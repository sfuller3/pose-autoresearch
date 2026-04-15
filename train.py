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
    BONE_PAIRS,
    DEVICE,
    EVENT_CLASSES,
    MAX_TIME_BUDGET_SECONDS,
    NUM_KEYPOINTS,
    NUM_BONES,
)
from collections import Counter

# ============================================================================
# HYPERPARAMETERS (agent can modify)
# ============================================================================

INPUT_DIM = 51       # 17 keypoints x 3 (x, y, confidence)
BONE_DIM = NUM_BONES * 3  # 16 bones x 3 (dx, dy, mean_conf)
VELOCITY_DIM = 51    # 17 keypoints x 3 (vx, vy, confidence)
FULL_INPUT_DIM = INPUT_DIM + BONE_DIM + VELOCITY_DIM  # 51 + 48 + 51 = 150
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
        self.act = nn.SiLU(inplace=True)

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
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.act(x + res)
        return x


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale 1D temporal convolution with residual connection.

    Parallel convolutions at kernel sizes 3, 7, 15 capture fast actions
    (fall impact ~0.3s), medium motions (eating cycles ~1s), and slow
    patterns (wandering ~5s) simultaneously.
    """

    def __init__(self, in_ch, out_ch, kernels=(3, 7, 15), stride=1, dropout=0.3):
        super().__init__()
        branch_ch = out_ch // len(kernels)
        remainder = out_ch - branch_ch * len(kernels)

        self.branches = nn.ModuleList()
        for i, k in enumerate(kernels):
            ch = branch_ch + (remainder if i == 0 else 0)
            pad = (k - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, k, stride=stride, padding=pad),
                nn.BatchNorm1d(ch),
                nn.SiLU(inplace=True),
            ))

        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU(inplace=True)

        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.residual = nn.Identity()

        self.se = SqueezeExcitation(out_ch)

    def forward(self, x):
        res = self.residual(x)
        branches = [branch(x) for branch in self.branches]
        x = torch.cat(branches, dim=1)  # (B, out_ch, T')
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = self.act(x + res)
        return x


class SqueezeExcitation(nn.Module):
    """Channel attention via squeeze-and-excitation."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.se(x).unsqueeze(2)  # (B, C, 1)
        return x * w


class TemporalAttentionPool(nn.Module):
    """Learned attention pooling over temporal dimension.

    Instead of average pooling (which weights all frames equally),
    learns which frames are most discriminative for classification.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.Tanh(),
            nn.Linear(channels // 4, 1),
        )

    def forward(self, x):
        # x: (B, C, T)
        x_t = x.permute(0, 2, 1)  # (B, T, C)
        weights = self.attn(x_t).squeeze(-1)  # (B, T)
        weights = F.softmax(weights, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), x_t).squeeze(1)  # (B, C)
        return pooled


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
        dropout: float = DROPOUT,
    ):
        super().__init__()

        in_dim = FULL_INPUT_DIM  # joint(51) + bone(48) + velocity(51) = 150

        # Channel progression for temporal blocks. The last value is the
        # feature dim consumed by the pool and classifier head.
        BLOCK_CHANNELS = (128, 128, 256, 256)
        final_channels = BLOCK_CHANNELS[-1]

        self.input_bn = nn.BatchNorm1d(in_dim)

        self.blocks = nn.ModuleList([
            MultiScaleTemporalBlock(in_dim, BLOCK_CHANNELS[0], kernels=(3, 7, 15), dropout=dropout),
            MultiScaleTemporalBlock(BLOCK_CHANNELS[0], BLOCK_CHANNELS[1], kernels=(3, 7, 15), dropout=dropout),
            MultiScaleTemporalBlock(BLOCK_CHANNELS[1], BLOCK_CHANNELS[2], kernels=(3, 7, 15), stride=2, dropout=dropout),
            MultiScaleTemporalBlock(BLOCK_CHANNELS[2], BLOCK_CHANNELS[3], kernels=(3, 7, 15), dropout=dropout),
        ])

        self.pool = TemporalAttentionPool(final_channels)

        self.fc = nn.Linear(final_channels, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 51) -- flattened keypoints
        Returns:
            logits: (batch, num_classes)
        """
        B, T, D = x.shape

        # Compute bone and velocity features in pure PyTorch (stays on GPU)
        kps = x.view(B, T, NUM_JOINTS, 3)  # (B, T, 17, 3)

        # Velocity: frame-to-frame displacement (uses raw kps; velocity stream
        # carries its own conf channel, so we don't gate it here)
        vel = torch.zeros_like(kps)
        vel[:, 1:, :, :2] = kps[:, 1:, :, :2] - kps[:, :-1, :, :2]
        vel[:, :, :, 2] = kps[:, :, :, 2]  # keep confidence
        vel_flat = vel.reshape(B, T, -1)  # (B, T, 51)

        # Confidence gating: attenuate features from low-confidence joints.
        # This prevents noisy YOLO detections from dominating the representation.
        # sigmoid(conf*5 - 2): conf<0.2 -> ~0.12, conf=0.4 -> 0.5, conf>0.6 -> ~0.73+
        conf = kps[:, :, :, 2:3]  # (B, T, 17, 1)
        conf_gate = torch.sigmoid(conf * 5 - 2)
        gated_kps = kps.clone()
        gated_kps[:, :, :, :2] = kps[:, :, :, :2] * conf_gate
        x_gated = gated_kps.reshape(B, T, -1)  # (B, T, 51) with gated coords

        # Bones: vectors between connected joints (computed from gated coords so
        # zero-confidence joints contribute zero displacement rather than noise)
        bone_parts = []
        for parent, child in BONE_PAIRS:
            dx = gated_kps[:, :, child, 0] - gated_kps[:, :, parent, 0]
            dy = gated_kps[:, :, child, 1] - gated_kps[:, :, parent, 1]
            bone_conf = (kps[:, :, child, 2] + kps[:, :, parent, 2]) / 2
            bone_parts.append(torch.stack([dx, dy, bone_conf], dim=2))  # (B, T, 3)
        bones_flat = torch.cat(bone_parts, dim=2)  # (B, T, 48)

        # Concatenate joint + bone + velocity
        x = torch.cat([x_gated, bones_flat, vel_flat], dim=2)  # (B, T, 150)

        # Input normalization
        x = x.permute(0, 2, 1)  # (B, 150, T)
        x = self.input_bn(x)

        # Temporal blocks
        for block in self.blocks:
            x = block(x)  # (B, 256, T')

        # Temporal attention pooling (learned per-frame importance)
        x = self.pool(x)  # (B, 256)

        # Classify
        logits = self.fc(x)
        return logits


# ============================================================================
# TRAINING
# ============================================================================


class FocalLoss(nn.Module):
    """Focal loss with per-class weights for hard-example mining.

    Reduces loss for well-classified examples, focusing training on
    confused classes (eating vs sitting_standing, unstable_gait detection).

    Note: ``pt`` is computed directly from the softmax probability of the
    true class (via ``log_softmax.gather``), so the focal modulation
    ``(1 - pt)**gamma`` is decoupled from both the per-class ``weight`` and
    ``label_smoothing``. The previous ``pt = exp(-ce)`` identity only holds
    for unweighted, unsmoothed CE on one-hot targets; with class weights it
    becomes ``p_true**w_y`` and with label smoothing it becomes a mixed
    quantity, both of which silently distort the focal factor.
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # True-class probability (independent of weight/smoothing)
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        # Standard CE with weight + label smoothing
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        # Focal modulation
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


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

    splits_dir = Path("data/splits")
    if splits_dir.exists() and (splits_dir / "train").exists():
        print("Loading from pre-split directories (data/splits/)")
        train_ds = PoseDataset(splits_dir / "train", augment=True)
        val_ds = PoseDataset(splits_dir / "val", augment=False)
        test_ds = PoseDataset(splits_dir / "test", augment=False)

        pin = DEVICE.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=pin,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=pin,
        )
    else:
        print("No data/splits/ found, using random split via get_dataloaders()")
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

    # Dynamic class weights from training set distribution
    label_counts = Counter()
    for _, label in train_loader.dataset:
        label_counts[label.item() if isinstance(label, torch.Tensor) else label] += 1

    total_samples = sum(label_counts.values())
    class_weights = torch.ones(NUM_CLASSES, device=DEVICE)
    if label_counts:
        for cls_idx in range(NUM_CLASSES):
            count = label_counts.get(cls_idx, 1)
            class_weights[cls_idx] = total_samples / (NUM_CLASSES * count)
        # Additional fall-priority boost
        class_weights[0] *= 1.5
        print(f"Class weights: {', '.join(f'{EVENT_CLASSES[i]}={class_weights[i]:.2f}' for i in range(NUM_CLASSES))}")

    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)

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
