"""
Causal transformer for streaming pose event detection.

Each frame attends only to past frames (like GPT). Produces a prediction
at every timestep, enabling real-time deployment on continuous video.

Training: full variable-length videos, per-frame cross-entropy loss.
Inference: feed frames as they arrive, get classification at each step.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from prepare import (
    BONE_PAIRS,
    CLASS_TO_IDX,
    DEVICE,
    EVENT_CLASSES,
    INPUT_DIM,
    MAX_TIME_BUDGET_SECONDS,
    NUM_BONES,
    NUM_KEYPOINTS,
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

NUM_CLASSES = len(EVENT_CLASSES)
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
DIM_FF = 256
MAX_SEQ_LEN = 256  # 99th percentile is 240 frames; caps outlier Le2i videos
WARMUP_EPOCHS = 5


# ============================================================================
# VARIABLE-LENGTH DATASET
# ============================================================================


class VariableLengthPoseDataset(Dataset):
    """Loads full-length pose sequences without truncation."""

    def __init__(self, data_dir: Path):
        self.samples: list[tuple[np.ndarray, int]] = []

        for json_file in data_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)

            label_str = data["label"]
            if label_str not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[label_str]

            frames = data["frames"]
            keypoints = []
            for frame in frames:
                flat = np.array(frame["keypoints"]).flatten()
                if len(flat) < INPUT_DIM:
                    flat = np.pad(flat, (0, INPUT_DIM - len(flat)))
                elif len(flat) > INPUT_DIM:
                    flat = flat[:INPUT_DIM]
                keypoints.append(flat)

            poses = np.array(keypoints, dtype=np.float32)
            # Cap at MAX_SEQ_LEN to keep batch padding manageable
            if len(poses) > MAX_SEQ_LEN:
                poses = poses[:MAX_SEQ_LEN]
            self.samples.append((poses, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        poses, label = self.samples[idx]
        return torch.from_numpy(poses.copy()), torch.tensor(label, dtype=torch.long)


class LengthBucketSampler(Sampler):
    """Batches sequences of similar length to minimize padding."""

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # Sort indices by sequence length
        self.sorted_indices = sorted(
            range(len(dataset)),
            key=lambda i: dataset.samples[i][0].shape[0],
        )

    def __iter__(self):
        # Create batches from sorted indices
        batches = [
            self.sorted_indices[i : i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.shuffle:
            import random
            random.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.sorted_indices)


def collate_variable_length(batch):
    """Pad sequences to max length in batch, return lengths for masking."""
    sequences, labels = zip(*batch)
    lengths = [s.shape[0] for s in sequences]
    max_len = max(lengths)
    dim = sequences[0].shape[1]

    padded = torch.zeros(len(sequences), max_len, dim)
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded[i, :length] = seq

    labels = torch.stack(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, labels, lengths


# ============================================================================
# MODEL: Causal Pose Transformer
# ============================================================================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CausalPoseTransformer(nn.Module):
    """Causal transformer for streaming pose event detection.

    Each frame attends only to current and past frames. Produces a class
    prediction at every timestep.

    For training: per-frame loss using the video-level label at every frame.
    For inference: feed frames incrementally, read prediction at each step.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        dim_ff: int = DIM_FF,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        # joint(51) + bone(48) + velocity(51) = 150
        full_input_dim = INPUT_DIM + NUM_BONES * 3 + INPUT_DIM

        self.input_proj = nn.Linear(full_input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def _compute_features(self, x):
        """Compute bone + velocity features from raw keypoints."""
        B, T, D = x.shape
        kps = x.view(B, T, NUM_KEYPOINTS, 3)

        # Velocity: frame-to-frame displacement
        vel = torch.zeros_like(kps)
        vel[:, 1:, :, :2] = kps[:, 1:, :, :2] - kps[:, :-1, :, :2]
        vel[:, :, :, 2] = kps[:, :, :, 2]
        vel_flat = vel.reshape(B, T, -1)

        # Bones: vectors between connected joints
        bone_parts = []
        for parent, child in BONE_PAIRS:
            dx = kps[:, :, child, 0] - kps[:, :, parent, 0]
            dy = kps[:, :, child, 1] - kps[:, :, parent, 1]
            conf = (kps[:, :, child, 2] + kps[:, :, parent, 2]) / 2
            bone_parts.append(torch.stack([dx, dy, conf], dim=2))
        bones_flat = torch.cat(bone_parts, dim=2)

        return torch.cat([x, bones_flat, vel_flat], dim=2)  # (B, T, 150)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (B, T, 51) raw keypoints
            lengths: (B,) actual sequence lengths (for padding mask)
        Returns:
            logits: (B, T, num_classes) — prediction at every frame
        """
        B, T, D = x.shape

        x = self._compute_features(x)  # (B, T, 150)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.pos_enc(x)
        x = self.input_dropout(x)

        # Causal mask: each position can only attend to itself and earlier
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device
        )

        # Padding mask: ignore padded positions
        pad_mask = None
        if lengths is not None:
            pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)

        return self.cls_head(x)  # (B, T, num_classes)


# ============================================================================
# TRAINING
# ============================================================================


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for poses, labels, lengths in dataloader:
        poses = poses.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(poses, lengths=lengths)  # (B, T, C)

        # Per-frame loss: every frame gets the video label
        B, T, C = logits.shape
        # Expand labels to every frame: (B,) -> (B, T)
        frame_labels = labels.unsqueeze(1).expand(B, T)

        # Mask out padded frames from loss
        loss_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        # Flatten for cross-entropy
        logits_flat = logits[loss_mask]  # (N, C)
        labels_flat = frame_labels[loss_mask]  # (N,)

        loss = criterion(logits_flat, labels_flat)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # Accuracy: use prediction at the last real frame of each sequence
        last_frame_idx = lengths - 1  # (B,)
        last_logits = logits[torch.arange(B, device=device), last_frame_idx]
        preds = torch.argmax(last_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += B

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for poses, labels, lengths in dataloader:
        poses = poses.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(poses, lengths=lengths)
        B, T, C = logits.shape

        # Loss on last frame only for eval
        last_idx = lengths - 1
        last_logits = logits[torch.arange(B, device=device), last_idx]
        loss = criterion(last_logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(last_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += B

    return correct / total, total_loss / len(dataloader)


@torch.no_grad()
def evaluate_per_class(model, dataloader, device):
    model.eval()
    class_correct = Counter()
    class_total = Counter()

    for poses, labels, lengths in dataloader:
        poses = poses.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        logits = model(poses, lengths=lengths)
        B = logits.shape[0]

        last_idx = lengths - 1
        last_logits = logits[torch.arange(B, device=device), last_idx]
        preds = torch.argmax(last_logits, dim=1)

        for pred, label in zip(preds, labels):
            cls = EVENT_CLASSES[label.item()]
            class_total[cls] += 1
            if pred.item() == label.item():
                class_correct[cls] += 1

    return {
        cls: class_correct[cls] / max(class_total[cls], 1) for cls in EVENT_CLASSES
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("POSE AUTORESEARCH - Causal Transformer Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: Causal Pose Transformer (d={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS})")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Time Budget: {MAX_TIME_BUDGET_SECONDS}s")
    print("=" * 70)

    splits_dir = Path("data/splits")
    train_ds = VariableLengthPoseDataset(splits_dir / "train")
    val_ds = VariableLengthPoseDataset(splits_dir / "val")
    test_ds = VariableLengthPoseDataset(splits_dir / "test")

    pin = DEVICE.type == "cuda"
    train_sampler = LengthBucketSampler(train_ds, BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=4, pin_memory=pin, collate_fn=collate_variable_length,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=pin, collate_fn=collate_variable_length,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=pin, collate_fn=collate_variable_length,
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print()

    model = CausalPoseTransformer().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    # Linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, 100 - WARMUP_EPOCHS)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Dynamic class weights
    label_counts = Counter()
    for _, label in train_ds:
        label_counts[label.item()] += 1

    total_samples = sum(label_counts.values())
    class_weights = torch.ones(NUM_CLASSES, device=DEVICE)
    for cls_idx in range(NUM_CLASSES):
        count = label_counts.get(cls_idx, 1)
        class_weights[cls_idx] = total_samples / (NUM_CLASSES * count)
    class_weights[0] *= 1.5  # fall boost
    print(f"Class weights: {', '.join(f'{EVENT_CLASSES[i]}={class_weights[i]:.2f}' for i in range(NUM_CLASSES))}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    start_time = time.time()
    epoch = 0
    best_val_acc = 0.0

    print("Training...")
    print()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= MAX_TIME_BUDGET_SECONDS:
            print(f"\nTime budget reached: {elapsed:.1f}s")
            break

        epoch += 1
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, DEVICE
        )
        epoch_time = time.time() - epoch_start

        val_acc, val_loss = evaluate_model(model, val_loader, DEVICE)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "config": {
                        "d_model": D_MODEL,
                        "n_heads": N_HEADS,
                        "n_layers": N_LAYERS,
                        "dim_ff": DIM_FF,
                        "dropout": DROPOUT,
                    },
                },
                "checkpoints/best_transformer.pt",
            )

        print(
            f"Epoch {epoch:3d} | {epoch_time:5.1f}s | "
            f"Train {train_loss:.4f}/{train_acc:.4f} | "
            f"Val {val_loss:.4f}/{val_acc:.4f} | "
            f"Best {best_val_acc:.4f}"
        )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    ckpt = torch.load("checkpoints/best_transformer.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    test_acc, test_loss = evaluate_model(model, test_loader, DEVICE)
    per_class = evaluate_per_class(model, test_loader, DEVICE)

    print(f"Best Val: {best_val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
    print()
    print("Per-class accuracy:")
    for cls, acc in per_class.items():
        print(f"  {cls:20s}: {acc:.4f}")
    print()

    return best_val_acc


if __name__ == "__main__":
    Path("checkpoints").mkdir(exist_ok=True)
    val_accuracy = main()
    print(f"\nFINAL VALIDATION ACCURACY: {val_accuracy:.4f}")
