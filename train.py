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
from pose_autoresearch.graph import (
    get_normalized_adjacency,
    adjacency_to_tensor,
    COCO_17_EDGES,
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
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
GCN_CHANNELS = [64, 128, 256]  # Graph conv channel progression
TEMPORAL_KERNEL_SIZE = 9        # Temporal conv kernel size

# ============================================================================
# GRAPH CONVOLUTION BLOCK
# ============================================================================


class SpatialGraphConv(nn.Module):
    """Single spatial graph convolution layer.

    Applies graph convolution: H' = A_norm @ H @ W
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)  # (num_joints, num_joints)
        self.W = nn.Linear(in_channels, out_channels, bias=False)
        self.bn = nn.BatchNorm1d(out_channels * A.shape[0])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_joints, channels)
        Returns:
            (batch, seq_len, num_joints, out_channels)
        """
        B, T, V, C = x.shape

        # Graph convolution: aggregate neighbor features
        x = torch.einsum("btvc,vw->btwc", x, self.A)

        # Linear transform
        x = self.W(x)

        # Batch norm
        C_out = x.shape[-1]
        x = x.reshape(B * T, V * C_out)
        x = self.bn(x)
        x = x.reshape(B, T, V, C_out)

        return x


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution block.

    1. Spatial graph conv (inter-joint relationships)
    2. Temporal conv (intra-joint motion over time)
    3. Residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.spatial = SpatialGraphConv(in_channels, out_channels, A)

        pad = (temporal_kernel - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (temporal_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_joints, in_channels)
        Returns:
            (batch, seq_len, num_joints, out_channels)
        """
        # Residual: reshape to (B, C, T, V) for Conv2d
        res = x.permute(0, 3, 1, 2)
        res = self.residual(res)

        # Spatial graph conv
        x = self.spatial(x)
        x = self.relu(x)

        # Temporal conv: (B, C_out, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.temporal(x)

        # Residual + activate
        x = x + res
        x = self.relu(x)
        x = self.dropout(x)

        # Back to (B, T, V, C)
        x = x.permute(0, 2, 3, 1)
        return x


# ============================================================================
# MODEL
# ============================================================================


class PoseEventClassifier(nn.Module):
    """Spatial-Temporal GCN for pose event classification.

    Architecture:
    1. Input projection: (x, y, conf) per joint -> feature channels
    2. Stack of ST-GCN blocks with increasing channels
    3. Global average pooling over time and joints
    4. Classification head

    Agent: You can modify anything here. Ideas to try:
    - Add attention (channel, temporal, or spatial)
    - Try CTR-GCN style channel-wise topology refinement
    - Replace temporal conv with Transformer encoder
    - Multi-scale temporal kernels
    - Multi-stream fusion (add bone/velocity inputs)
    - Deeper or wider GCN blocks
    - Different pooling strategies (attention pooling, max pooling)
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        num_joints: int = NUM_JOINTS,
        num_classes: int = NUM_CLASSES,
        gcn_channels: list[int] | None = None,
        temporal_kernel: int = TEMPORAL_KERNEL_SIZE,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        if gcn_channels is None:
            gcn_channels = list(GCN_CHANNELS)

        A = adjacency_to_tensor(get_normalized_adjacency(COCO_17_EDGES, num_joints))

        self.input_bn = nn.BatchNorm1d(input_channels * num_joints)

        layers = []
        in_ch = input_channels
        for out_ch in gcn_channels:
            layers.append(
                STGCNBlock(in_ch, out_ch, A, temporal_kernel, dropout=dropout)
            )
            in_ch = out_ch

        self.gcn_blocks = nn.ModuleList(layers)
        self.fc = nn.Linear(gcn_channels[-1], num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 51) -- flattened keypoints
        Returns:
            logits: (batch, num_classes)
        """
        B, T, _ = x.shape

        # Reshape to (batch, seq_len, 17, 3)
        x = x.view(B, T, NUM_JOINTS, INPUT_CHANNELS)

        # Input normalization
        x_flat = x.reshape(B * T, NUM_JOINTS * INPUT_CHANNELS)
        x_flat = self.input_bn(x_flat)
        x = x_flat.reshape(B, T, NUM_JOINTS, INPUT_CHANNELS)

        # ST-GCN blocks
        for block in self.gcn_blocks:
            x = block(x)

        # Global average pooling over time and joints
        x = x.mean(dim=[1, 2])

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
    print(f"Model: ST-GCN PoseEventClassifier")
    print(f"GCN Channels: {GCN_CHANNELS}")
    print(f"Temporal Kernel: {TEMPORAL_KERNEL_SIZE}")
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
        gcn_channels=GCN_CHANNELS,
        temporal_kernel=TEMPORAL_KERNEL_SIZE,
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

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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
