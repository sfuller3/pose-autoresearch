"""
Pose-based event detection training script.
The agent modifies this file to improve validation accuracy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from pathlib import Path

from prepare import (
    PoseDataset,
    get_dataloaders,
    evaluate_model,
    evaluate_per_class,
    DEVICE,
    EVENT_CLASSES,
    MAX_TIME_BUDGET_SECONDS,
    NUM_KEYPOINTS,
)

# ============================================================================
# HYPERPARAMETERS (agent can modify)
# ============================================================================

INPUT_DIM = 51       # 17 keypoints x 3 (x, y, confidence)
INPUT_CHANNELS = 3   # Per-joint: x, y, confidence
NUM_JOINTS = NUM_KEYPOINTS  # 17
NUM_CLASSES = len(EVENT_CLASSES)  # 7
SEQ_LEN = 150        # 5 seconds at 30fps
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# ============================================================================
# MODEL: Lightweight 1D Temporal CNN
# ============================================================================


class TemporalBlock(nn.Module):
    """1D temporal convolution block with residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (B, C, T)
        res = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x + res)
        return x


class PoseEventClassifier(nn.Module):
    """Lightweight 1D temporal CNN for pose event classification.

    Flattens all 51 keypoint features and applies temporal convolutions.
    Much faster than ST-GCN on CPU while capturing temporal patterns.
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_joints: int = NUM_JOINTS,
        num_classes: int = NUM_CLASSES,
        gcn_channels: list[int] | None = None,
        temporal_kernel: int = 9,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        in_dim = num_joints * input_channels  # 51

        self.input_bn = nn.BatchNorm1d(in_dim)

        self.blocks = nn.Sequential(
            TemporalBlock(in_dim, 128, kernel_size=7, dropout=dropout),
            TemporalBlock(128, 128, kernel_size=7, dropout=dropout),
            TemporalBlock(128, 256, kernel_size=5, stride=2, dropout=dropout),
            TemporalBlock(256, 256, kernel_size=5, dropout=dropout),
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 51) -- flattened keypoints
        Returns:
            logits: (batch, num_classes)
        """
        B, T, D = x.shape

        # Input normalization
        x = x.permute(0, 2, 1)  # (B, 51, T)
        x = self.input_bn(x)

        # Temporal blocks
        x = self.blocks(x)  # (B, 256, T')

        # Global average pooling over time
        x = x.mean(dim=2)  # (B, 256)

        # Classify
        logits = self.fc(x)
        return logits


# ============================================================================
# TRAINING
# ============================================================================


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for poses, labels in dataloader:
        poses = poses.to(device)
        labels = labels.to(device)

        logits = model(poses)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader), correct / total


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("POSE AUTORESEARCH - Training Run")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: Temporal CNN")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Time Budget: {MAX_TIME_BUDGET_SECONDS}s")
    print("=" * 70)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4,
        augment_train=True,
    )

    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    print()

    model = PoseEventClassifier(
        dropout=DROPOUT,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6,
    )

    # Class weights: boost fall (class 0)
    class_weights = torch.ones(NUM_CLASSES, device=DEVICE)
    class_weights[0] = 1.5  # fall priority
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
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                "checkpoints/best_model.pt",
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

    # Final evaluation
    ckpt = torch.load("checkpoints/best_model.pt", weights_only=True)
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
