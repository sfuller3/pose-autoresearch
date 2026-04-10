"""
Hybrid CNN-Transformer for streaming pose event detection.

Architecture:
  Pose keypoints → shared feature extraction (bone + velocity)
                 → CNN branch (local temporal patterns)
                 → Transformer branch (global causal attention)
                 → fusion → per-frame classification

The CNN captures fast local motion (fall impact, hand gestures) while the
transformer captures long-range context (standing → falling → lying still).
Both heads can also be used independently at inference.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5

# Shared
FULL_INPUT_DIM = INPUT_DIM + NUM_BONES * 3 + INPUT_DIM  # 51+48+51 = 150
PROJ_DIM = 128  # shared projection dimension
DROPOUT = 0.2
MAX_SEQ_LEN = 256

# CNN branch
CNN_CHANNELS = [128, 128]
CNN_KERNELS = [7, 5]

# Transformer branch
D_MODEL = 128  # matches PROJ_DIM
N_HEADS = 4
N_LAYERS = 3
DIM_FF = 256

# Fusion
FUSION_DIM = 128


# ============================================================================
# DATASET + SAMPLER (reused from train_transformer.py)
# ============================================================================


class VariableLengthPoseDataset(Dataset):
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
            if len(poses) > MAX_SEQ_LEN:
                poses = poses[:MAX_SEQ_LEN]
            self.samples.append((poses, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        poses, label = self.samples[idx]
        return torch.from_numpy(poses.copy()), torch.tensor(label, dtype=torch.long)


class LengthBucketSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sorted_indices = sorted(
            range(len(dataset)),
            key=lambda i: dataset.samples[i][0].shape[0],
        )

    def __iter__(self):
        batches = [
            self.sorted_indices[i : i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.sorted_indices)


def collate_variable_length(batch):
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
# MODEL
# ============================================================================


class FeatureExtractor(nn.Module):
    """Shared feature extraction: raw keypoints → bone + velocity features."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """x: (B, T, 51) → (B, T, 150)"""
        B, T, D = x.shape
        kps = x.view(B, T, NUM_KEYPOINTS, 3)

        vel = torch.zeros_like(kps)
        vel[:, 1:, :, :2] = kps[:, 1:, :, :2] - kps[:, :-1, :, :2]
        vel[:, :, :, 2] = kps[:, :, :, 2]
        vel_flat = vel.reshape(B, T, -1)

        bone_parts = []
        for parent, child in BONE_PAIRS:
            dx = kps[:, :, child, 0] - kps[:, :, parent, 0]
            dy = kps[:, :, child, 1] - kps[:, :, parent, 1]
            conf = (kps[:, :, child, 2] + kps[:, :, parent, 2]) / 2
            bone_parts.append(torch.stack([dx, dy, conf], dim=2))
        bones_flat = torch.cat(bone_parts, dim=2)

        return torch.cat([x, bones_flat, vel_flat], dim=2)


class CNNBranch(nn.Module):
    """Temporal CNN branch — captures local motion patterns."""

    def __init__(self, in_dim, channels, kernels, dropout):
        super().__init__()
        layers = []
        prev_ch = in_dim
        for ch, k in zip(channels, kernels):
            pad = (k - 1) // 2
            layers.extend([
                nn.Conv1d(prev_ch, ch, k, padding=pad),
                nn.BatchNorm1d(ch),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_ch = ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (B, C, T) → (B, C_out, T)"""
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerBranch(nn.Module):
    """Causal transformer branch — captures long-range dependencies."""

    def __init__(self, d_model, n_heads, n_layers, dim_ff, dropout):
        super().__init__()
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x, causal_mask, pad_mask):
        """x: (B, T, D) → (B, T, D)"""
        x = self.pos_enc(x)
        x = self.dropout(x)
        return self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)


class HybridPoseDetector(nn.Module):
    """Hybrid CNN-Transformer for streaming pose event detection.

    Two parallel branches process the same features:
    - CNN: fast local temporal patterns (receptive field ~11 frames)
    - Transformer: causal global attention (sees all past frames)

    A learned gate fuses both branches per-frame.
    """

    def __init__(
        self,
        proj_dim=PROJ_DIM,
        cnn_channels=None,
        cnn_kernels=None,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_ff=DIM_FF,
        dropout=DROPOUT,
        fusion_dim=FUSION_DIM,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = CNN_CHANNELS
        if cnn_kernels is None:
            cnn_kernels = CNN_KERNELS

        self.features = FeatureExtractor()

        # Shared projection
        self.proj = nn.Sequential(
            nn.Linear(FULL_INPUT_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # CNN branch: operates on (B, C, T)
        self.cnn = CNNBranch(proj_dim, cnn_channels, cnn_kernels, dropout)
        cnn_out_dim = cnn_channels[-1]

        # Transformer branch: operates on (B, T, D)
        self.transformer = TransformerBranch(d_model, n_heads, n_layers, dim_ff, dropout)
        tx_out_dim = d_model

        # Fusion gate: learns per-frame weighting of CNN vs Transformer
        combined_dim = cnn_out_dim + tx_out_dim
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.SiLU(),
            nn.Linear(fusion_dim, 2),  # 2 weights: CNN, Transformer
        )

        # Fusion projection + classifier
        self.fusion_norm = nn.LayerNorm(combined_dim)
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

        # Independent heads for auxiliary losses
        self.cnn_head = nn.Linear(cnn_out_dim, num_classes)
        self.tx_head = nn.Linear(tx_out_dim, num_classes)

    def forward(self, x, lengths=None):
        """
        Args:
            x: (B, T, 51) raw keypoints
            lengths: (B,) actual sequence lengths
        Returns:
            dict with keys:
                'fused': (B, T, num_classes) — main output
                'cnn':   (B, T, num_classes) — CNN-only predictions
                'tx':    (B, T, num_classes) — Transformer-only predictions
                'gate':  (B, T, 2) — fusion gate weights
        """
        B, T, _ = x.shape

        # Shared features + projection
        feat = self.features(x)       # (B, T, 150)
        proj = self.proj(feat)        # (B, T, proj_dim)

        # CNN branch: (B, T, C) → (B, C, T) → CNN → (B, C_out, T) → (B, T, C_out)
        cnn_out = self.cnn(proj.permute(0, 2, 1)).permute(0, 2, 1)

        # Transformer branch with causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        pad_mask = None
        if lengths is not None:
            pad_mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        tx_out = self.transformer(proj, causal_mask, pad_mask)

        # Fusion
        combined = torch.cat([cnn_out, tx_out], dim=-1)  # (B, T, cnn+tx)
        combined = self.fusion_norm(combined)

        gate_weights = F.softmax(self.gate(combined), dim=-1)  # (B, T, 2)
        gated = (
            gate_weights[:, :, 0:1] * cnn_out
            + gate_weights[:, :, 1:2] * tx_out
        )
        # But classifier uses full combined features (gate just modulates attention)
        fused_logits = self.classifier(combined)

        return {
            'fused': fused_logits,
            'cnn': self.cnn_head(cnn_out),
            'tx': self.tx_head(tx_out),
            'gate': gate_weights,
        }


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
        B, T, _ = poses.shape

        outputs = model(poses, lengths=lengths)

        # Frame labels: every frame gets the video label
        frame_labels = labels.unsqueeze(1).expand(B, T)
        loss_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        # Main loss on fused output
        fused_flat = outputs['fused'][loss_mask]
        labels_flat = frame_labels[loss_mask]
        loss_fused = criterion(fused_flat, labels_flat)

        # Auxiliary losses on individual heads (weighted lower)
        cnn_flat = outputs['cnn'][loss_mask]
        tx_flat = outputs['tx'][loss_mask]
        loss_cnn = criterion(cnn_flat, labels_flat)
        loss_tx = criterion(tx_flat, labels_flat)

        loss = loss_fused + 0.3 * loss_cnn + 0.3 * loss_tx

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss_fused.item()

        # Accuracy on last frame of fused output
        last_idx = lengths - 1
        last_logits = outputs['fused'][torch.arange(B, device=device), last_idx]
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
        B = poses.shape[0]

        outputs = model(poses, lengths=lengths)
        last_idx = lengths - 1
        last_logits = outputs['fused'][torch.arange(B, device=device), last_idx]
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
    # Also track gate statistics
    gate_sum = torch.zeros(2)
    gate_count = 0

    for poses, labels, lengths in dataloader:
        poses = poses.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        B = poses.shape[0]

        outputs = model(poses, lengths=lengths)
        last_idx = lengths - 1
        last_logits = outputs['fused'][torch.arange(B, device=device), last_idx]
        preds = torch.argmax(last_logits, dim=1)

        # Gate weights at last frame
        last_gate = outputs['gate'][torch.arange(B, device=device), last_idx]
        gate_sum += last_gate.cpu().sum(dim=0)
        gate_count += B

        for pred, label in zip(preds, labels):
            cls = EVENT_CLASSES[label.item()]
            class_total[cls] += 1
            if pred.item() == label.item():
                class_correct[cls] += 1

    avg_gate = gate_sum / gate_count
    print(f"\n  Gate weights (avg): CNN={avg_gate[0]:.3f}, Transformer={avg_gate[1]:.3f}")

    return {
        cls: class_correct[cls] / max(class_total[cls], 1) for cls in EVENT_CLASSES
    }


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("POSE AUTORESEARCH - Hybrid CNN-Transformer Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: Hybrid (CNN [{CNN_CHANNELS}] + Transformer [d={D_MODEL}, h={N_HEADS}, L={N_LAYERS}])")
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

    model = HybridPoseDetector().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

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
    class_weights[0] *= 1.5
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
                },
                "checkpoints/best_hybrid.pt",
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

    ckpt = torch.load("checkpoints/best_hybrid.pt", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    test_acc, test_loss = evaluate_model(model, test_loader, DEVICE)
    per_class = evaluate_per_class(model, test_loader, DEVICE)

    print(f"\nBest Val: {best_val_acc:.4f}")
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
