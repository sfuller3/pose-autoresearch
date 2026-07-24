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
    CLASS_TO_IDX,
    DEVICE,
    EVENT_CLASSES,
    MAX_TIME_BUDGET_SECONDS,
    NUM_KEYPOINTS,
    NUM_BONES,
)
from pose_autoresearch.augment import augment_pose_sequence
from collections import Counter
from datetime import datetime, timezone
import json
import os
import subprocess
import numpy as np

# ============================================================================
# RUN IDENTITY (set via environment so parallel runs never collide)
# ============================================================================

# POSE_RUN_NAME names the run. It determines the checkpoint filename and is
# recorded in the run log, so two concurrent experiments can't overwrite each
# other's weights (which silently corrupted the 2026-07-24 baseline run).
RUN_NAME = os.environ.get("POSE_RUN_NAME", "default")
RUN_NOTES = os.environ.get("POSE_RUN_NOTES", "")

_ckpt_default = (
    "checkpoints/best_model.pt" if RUN_NAME == "default"
    else f"checkpoints/best_model_{RUN_NAME}.pt"
)
CHECKPOINT_PATH = Path(os.environ.get("POSE_CHECKPOINT", _ckpt_default))
RUN_LOG_PATH = Path(os.environ.get("POSE_RUN_LOG", "experiments/runs.jsonl"))

# ============================================================================
# HYPERPARAMETERS (agent can modify)
# ============================================================================

INPUT_DIM = 51       # 17 keypoints x 3 (x, y, confidence)
BONE_DIM = NUM_BONES * 3  # 16 bones x 3 (dx, dy, mean_conf)
VELOCITY_DIM = 51    # 17 keypoints x 3 (vx, vy, confidence)
FULL_INPUT_DIM = INPUT_DIM + BONE_DIM + VELOCITY_DIM  # 51 + 48 + 51 = 150
# Inter-body interaction metadata appended per frame (see
# MultiPersonPoseDataset._compute_metadata). Widened from 3 -> 7 to add
# approach dynamics and limb-level proximity, which hip-to-hip distance alone
# can't express — targets the working_together/aggression/wandering confusion
# triangle (see STATUS.md confusion diagnostic, 2026-07-24).
META_DIM = 7
MULTI_INPUT_DIM = 102 + META_DIM       # primary(51) + neighbor(51) + metadata
MULTI_FULL_INPUT_DIM = 300 + META_DIM  # primary_feats(150) + neighbor_feats(150) + metadata
INPUT_CHANNELS = 3   # Per-joint: x, y, confidence
NUM_JOINTS = NUM_KEYPOINTS  # 17
NUM_CLASSES = len(EVENT_CLASSES)  # 7
SEQ_LEN = 150        # 5 seconds at 30fps
BATCH_SIZE = 256
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# Channel progression for the temporal blocks. The last value is the feature
# dim consumed by the pool and classifier head. The wide (192/384) setting beat
# the previous (128/256) one by +0.4pt val on equal wall-clock (2026-07-24).
# Override for sweeps with POSE_BLOCK_CHANNELS="128,128,256,256".
BLOCK_CHANNELS = tuple(
    int(c) for c in os.environ.get("POSE_BLOCK_CHANNELS", "192,192,384,384").split(",")
)

# MixUp augmentation
MIXUP_ALPHA = 0.2    # Beta(alpha, alpha) shape parameter; <1 is U-shaped
MIXUP_PROB = 0.5     # Probability of applying MixUp to a given batch
EMA_DECAY = 0.999    # Exponential moving average decay for model weights

# ============================================================================
# MULTI-PERSON DATASET
# ============================================================================


class MultiPersonPoseDataset(torch.utils.data.Dataset):
    """Dataset that pairs primary and neighbor bodies for multi-person tracking.

    Each JSON sample may contain a ``bodies`` list with multiple detected people.
    When two or more bodies are present, the dataset produces *two* training
    examples per sample (pair flipping) to double interaction data.  Metadata
    features (distance, relative position) are computed from hip midpoints.
    """

    # Hip joint indices in COCO-17 format
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    # Approximate frame diagonal for distance normalization (640x480 default)
    FRAME_DIAG = (640 ** 2 + 480 ** 2) ** 0.5

    def __init__(self, data_dir, seq_len: int = 150, augment: bool = False):
        self.seq_len = seq_len
        self.augment = augment
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        # Each entry is (primary_kps, neighbor_kps, metadata, label_idx)
        # primary_kps/neighbor_kps: (T, 51)   metadata: (T, META_DIM)

        data_dir = Path(data_dir)
        for json_path in sorted(data_dir.glob("*.json")):
            with open(json_path) as f:
                data = json.load(f)

            label_str = data["label"]
            if label_str not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[label_str]

            frames = data["frames"]
            T = len(frames)

            # Extract body 0 (primary) keypoints — always present
            primary = np.zeros((T, 17, 3), dtype=np.float32)
            for t, frame in enumerate(frames):
                kps = np.array(frame["keypoints"], dtype=np.float32).reshape(17, 3)
                primary[t] = kps

            # Check for multi-body data
            has_neighbor = False
            neighbor = np.zeros((T, 17, 3), dtype=np.float32)
            for t, frame in enumerate(frames):
                bodies = frame.get("bodies")
                if bodies is not None and len(bodies) >= 2:
                    has_neighbor = True
                    nb = np.array(bodies[1], dtype=np.float32).reshape(17, 3)
                    neighbor[t] = nb

            # Compute metadata from hip midpoints
            metadata = self._compute_metadata(primary, neighbor)

            # Flatten to (T, 51)
            primary_flat = primary.reshape(T, 51)
            neighbor_flat = neighbor.reshape(T, 51)

            self.samples.append((primary_flat, neighbor_flat, metadata, label))

            # Pair flipping: if we have a real neighbor, also add the reverse
            if has_neighbor:
                rev_metadata = self._compute_metadata(neighbor, primary)
                self.samples.append((
                    neighbor.reshape(T, 51),
                    primary.reshape(T, 51),
                    rev_metadata,
                    label,
                ))

    def _compute_metadata(self, primary: np.ndarray, neighbor: np.ndarray) -> np.ndarray:
        """Compute per-frame inter-body interaction metadata (META_DIM dims).

        Dims 0-2 are the original hip-midpoint geometry (unchanged so prior
        checkpoints stay interpretable). Dims 3-6 add approach dynamics and
        limb-level proximity — the signals that separate the two-body classes
        working_together / aggression / wandering, which hip distance alone
        cannot (see STATUS.md confusion diagnostic).

        Args:
            primary: (T, 17, 3) keypoints for the primary person
            neighbor: (T, 17, 3) keypoints for the neighbor person

        Returns:
            metadata: (T, META_DIM) —
                [0] dist_norm            hip-midpoint distance / frame diagonal
                [1] relative_x           hip-midpoint dx / frame diagonal
                [2] relative_y           hip-midpoint dy / frame diagonal
                [3] closing_speed        -d(dist_norm)/dt  (+ = approaching)
                [4] wrist_min_dist       nearest wrist-wrist distance / diagonal
                [5] wrist_closing_speed  -d(wrist_min_dist)/dt (+ = hands closing)
                [6] motion_align         cos angle of the two hip velocity vectors
        """
        T = primary.shape[0]
        metadata = np.zeros((T, META_DIM), dtype=np.float32)
        diag = self.FRAME_DIAG + 1e-8

        p_hip = (primary[:, self.LEFT_HIP, :2] + primary[:, self.RIGHT_HIP, :2]) / 2  # (T, 2)
        n_hip = (neighbor[:, self.LEFT_HIP, :2] + neighbor[:, self.RIGHT_HIP, :2]) / 2  # (T, 2)

        diff = n_hip - p_hip  # (T, 2)
        dist_norm = np.linalg.norm(diff, axis=1) / diag  # (T,)

        metadata[:, 0] = dist_norm
        metadata[:, 1] = diff[:, 0] / diag  # relative_x
        metadata[:, 2] = diff[:, 1] / diag  # relative_y

        # [3] closing speed: how fast hip-to-hip distance shrinks (frame diff).
        # Positive = the two bodies are approaching. Distinguishes a rapid
        # aggressive approach from slow independent drift (wandering).
        d_dist = np.zeros(T, dtype=np.float32)
        d_dist[1:] = dist_norm[1:] - dist_norm[:-1]
        metadata[:, 3] = -d_dist

        # [4] nearest wrist-to-wrist distance: hand proximity is the essence of
        # aggression (strikes/pushes) and separates it from working_together.
        p_wrists = primary[:, (self.LEFT_WRIST, self.RIGHT_WRIST), :2]   # (T, 2, 2)
        n_wrists = neighbor[:, (self.LEFT_WRIST, self.RIGHT_WRIST), :2]  # (T, 2, 2)
        # Pairwise over the 2x2 wrist cross-pairs -> (T, 2, 2) distances, min over pairs.
        pair = p_wrists[:, :, None, :] - n_wrists[:, None, :, :]  # (T, 2, 2, 2)
        wrist_dists = np.linalg.norm(pair, axis=-1)               # (T, 2, 2)
        wrist_min = wrist_dists.reshape(T, -1).min(axis=1) / diag  # (T,)
        metadata[:, 4] = wrist_min

        # [5] wrist closing speed: strike/reach dynamics (frame diff of [4]).
        d_wrist = np.zeros(T, dtype=np.float32)
        d_wrist[1:] = wrist_min[1:] - wrist_min[:-1]
        metadata[:, 5] = -d_wrist

        # [6] motion alignment: cos angle between the two bodies' hip-velocity
        # vectors. ~+1 = moving together (working_together / walking as a pair),
        # near 0 or negative = independent or converging motion.
        p_vel = np.zeros_like(p_hip)
        n_vel = np.zeros_like(n_hip)
        p_vel[1:] = p_hip[1:] - p_hip[:-1]
        n_vel[1:] = n_hip[1:] - n_hip[:-1]
        dot = (p_vel * n_vel).sum(axis=1)
        norm = np.linalg.norm(p_vel, axis=1) * np.linalg.norm(n_vel, axis=1)
        metadata[:, 6] = np.where(norm > 1e-6, dot / (norm + 1e-8), 0.0)

        return metadata

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        primary, neighbor, metadata, label = self.samples[idx]
        T = primary.shape[0]

        # Augmentation: apply independently to both bodies.
        # Independent augmentation adds slight spatial noise to the
        # inter-person relationship, acting as regularization.
        if self.augment:
            p_tensor = torch.from_numpy(primary.reshape(T, 17, 3).copy())
            p_tensor = augment_pose_sequence(p_tensor)
            primary = p_tensor.numpy().reshape(T, 51)

            # Augment neighbor too (if present, i.e., not all-zero)
            if neighbor.any():
                n_tensor = torch.from_numpy(neighbor.reshape(T, 17, 3).copy())
                n_tensor = augment_pose_sequence(n_tensor)
                neighbor = n_tensor.numpy().reshape(T, 51)
                # Recompute metadata after augmentation since positions changed
                metadata = self._compute_metadata(
                    p_tensor.numpy().reshape(T, 17, 3),
                    n_tensor.numpy().reshape(T, 17, 3),
                )

        # Pad or truncate to seq_len
        if T >= self.seq_len:
            primary = primary[:self.seq_len]
            neighbor = neighbor[:self.seq_len]
            metadata = metadata[:self.seq_len]
        else:
            pad_len = self.seq_len - T
            primary = np.concatenate([primary, np.zeros((pad_len, 51), dtype=np.float32)])
            neighbor = np.concatenate([neighbor, np.zeros((pad_len, 51), dtype=np.float32)])
            metadata = np.concatenate([metadata, np.zeros((pad_len, META_DIM), dtype=np.float32)])

        # Concatenate: primary(51) + neighbor(51) + metadata(META_DIM)
        combined = np.concatenate([primary, neighbor, metadata], axis=1)

        return torch.tensor(combined, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ============================================================================
# MODEL: Lightweight 1D Temporal CNN
# ============================================================================


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


class EnvironmentConditioner(nn.Module):
    """FiLM conditioning: environment features modulate temporal CNN channels.

    Given environment context vector, produces per-channel scale and shift
    that are applied after each temporal block's batch norm.
    """

    def __init__(self, env_dim: int, channel_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(env_dim, channel_dim),
            nn.SiLU(),
            nn.Linear(channel_dim, channel_dim * 2),  # gamma + beta
        )

    def forward(self, env_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gamma, beta) each of shape (B, C)."""
        out = self.fc(env_features)  # (B, 2*C)
        gamma, beta = out.chunk(2, dim=-1)
        return 1 + gamma, beta  # gamma centered at 1 for identity init


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
        env_dim: int = 0,  # 0 = no environment conditioning
        n_bodies: int = 1,  # 1 = single-body (51-dim), 2 = dual-body (102+META_DIM)
    ):
        super().__init__()
        self.env_dim = env_dim
        self.n_bodies = n_bodies

        if n_bodies == 2:
            in_dim = MULTI_FULL_INPUT_DIM  # 150 + 150 + META_DIM
        else:
            in_dim = FULL_INPUT_DIM  # joint(51) + bone(48) + velocity(51) = 150

        final_channels = BLOCK_CHANNELS[-1]

        self.input_bn = nn.BatchNorm1d(in_dim)

        self.blocks = nn.ModuleList([
            MultiScaleTemporalBlock(in_dim, BLOCK_CHANNELS[0], kernels=(3, 7, 15), dropout=dropout),
            MultiScaleTemporalBlock(BLOCK_CHANNELS[0], BLOCK_CHANNELS[1], kernels=(3, 7, 15), dropout=dropout),
            MultiScaleTemporalBlock(BLOCK_CHANNELS[1], BLOCK_CHANNELS[2], kernels=(3, 7, 15), stride=2, dropout=dropout),
            MultiScaleTemporalBlock(BLOCK_CHANNELS[2], BLOCK_CHANNELS[3], kernels=(3, 7, 15), dropout=dropout),
        ])

        # FiLM conditioning per block (only if env features provided)
        if env_dim > 0:
            self.conditioners = nn.ModuleList([
                EnvironmentConditioner(env_dim, ch)
                for ch in BLOCK_CHANNELS
            ])
        else:
            self.conditioners = None

        self.pool = TemporalAttentionPool(final_channels)

        self.fc = nn.Linear(final_channels, num_classes)

    def _process_body(self, kps):
        """Compute confidence gating, bones, and velocity for one body.

        Args:
            kps: (B, T, 17, 3) raw keypoints

        Returns:
            features: (B, T, 150) — gated_joints(51) + bones(48) + velocity(51)
        """
        B, T = kps.shape[:2]

        # Velocity: frame-to-frame displacement
        vel = torch.zeros_like(kps)
        vel[:, 1:, :, :2] = kps[:, 1:, :, :2] - kps[:, :-1, :, :2]
        vel[:, :, :, 2] = kps[:, :, :, 2]  # keep confidence
        vel_flat = vel.reshape(B, T, -1)  # (B, T, 51)

        # Confidence gating
        conf = kps[:, :, :, 2:3]  # (B, T, 17, 1)
        conf_gate = torch.sigmoid(conf * 5 - 2)
        gated_kps = kps.clone()
        gated_kps[:, :, :, :2] = kps[:, :, :, :2] * conf_gate
        x_gated = gated_kps.reshape(B, T, -1)  # (B, T, 51)

        # Bones
        bone_parts = []
        for parent, child in BONE_PAIRS:
            dx = gated_kps[:, :, child, 0] - gated_kps[:, :, parent, 0]
            dy = gated_kps[:, :, child, 1] - gated_kps[:, :, parent, 1]
            bone_conf = (kps[:, :, child, 2] + kps[:, :, parent, 2]) / 2
            bone_parts.append(torch.stack([dx, dy, bone_conf], dim=2))  # (B, T, 3)
        bones_flat = torch.cat(bone_parts, dim=2)  # (B, T, 48)

        return torch.cat([x_gated, bones_flat, vel_flat], dim=2)  # (B, T, 150)

    def forward(self, x, env_features=None):
        """
        Args:
            x: (batch, seq_len, D) -- D=51 for single-body, D=102+META_DIM for dual-body
            env_features: optional (batch, env_dim) -- environment context
        Returns:
            logits: (batch, num_classes)
        """
        B, T, D = x.shape

        if self.n_bodies == 2:
            # Split input: primary(51) + neighbor(51) + metadata(META_DIM)
            primary_raw = x[:, :, :51]
            neighbor_raw = x[:, :, 51:102]
            metadata = x[:, :, 102:102 + META_DIM]  # (B, T, META_DIM)

            primary_kps = primary_raw.view(B, T, NUM_JOINTS, 3)
            neighbor_kps = neighbor_raw.view(B, T, NUM_JOINTS, 3)

            primary_feats = self._process_body(primary_kps)   # (B, T, 150)
            neighbor_feats = self._process_body(neighbor_kps)  # (B, T, 150)

            # Soft distance decay: attenuate neighbor when far away
            # dist_norm is metadata[:,:,0]
            dist_norm = metadata[:, :, 0:1]  # (B, T, 1)
            decay = torch.sigmoid(5 - dist_norm * 10)  # (B, T, 1)
            neighbor_feats = neighbor_feats * decay

            # Concatenate: primary(150) + neighbor(150) + metadata(META_DIM)
            x = torch.cat([primary_feats, neighbor_feats, metadata], dim=2)
        else:
            # Single-body path: identical to original behavior
            kps = x.view(B, T, NUM_JOINTS, 3)  # (B, T, 17, 3)
            x = self._process_body(kps)  # (B, T, 150)

        # Input normalization
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = self.input_bn(x)

        # Temporal blocks with optional FiLM conditioning
        for i, block in enumerate(self.blocks):
            x = block(x)  # (B, C, T')
            if self.conditioners is not None and env_features is not None:
                gamma, beta = self.conditioners[i](env_features)
                # Broadcast: gamma/beta are (B, C), x is (B, C, T')
                x = gamma.unsqueeze(2) * x + beta.unsqueeze(2)

        # Temporal attention pooling (learned per-frame importance)
        x = self.pool(x)  # (B, 256)

        # Classify
        logits = self.fc(x)
        return logits


# ============================================================================
# TRAINING
# ============================================================================


class ModelEMA:
    """Exponential moving average of model parameters for smoother evaluation."""

    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply(self, model):
        """Temporarily apply EMA weights to model. Returns original state_dict."""
        original = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return original

    def restore(self, model, original):
        """Restore original weights after EMA evaluation."""
        model.load_state_dict(original)


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


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """Temporal MixUp: blend two sequences with random lambda."""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def train_epoch(model, dataloader, optimizer, criterion, scheduler, device,
                use_mixup=True, scaler=None, ema=None):
    """Train for one epoch with optional temporal mixup and AMP."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for poses, labels in dataloader:
        poses = poses.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_mixup and torch.rand(()).item() < MIXUP_PROB:
                poses, labels_a, labels_b, lam = mixup_data(poses, labels)
                logits = model(poses)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels_a).sum().item()
            else:
                logits = model(poses)
                loss = criterion(logits, labels)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

    return total_loss / len(dataloader), correct / total


# ============================================================================
# MAIN
# ============================================================================


def _git_rev() -> str:
    """Short HEAD hash, with a ``+dirty`` suffix if the tree has changes."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return f"{rev}+dirty" if dirty else rev
    except Exception:
        return "unknown"


def record_run(*, best_val_acc, test_acc, test_loss, per_class,
               epochs, elapsed, num_params) -> None:
    """Append this run's config and results to the run log.

    One JSON object per line so runs are never lost to a concurrent write and
    the log survives crashes.  ``scripts/leaderboard.py`` renders it.
    """
    record = {
        "run_name": RUN_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "notes": RUN_NOTES,
        "checkpoint": str(CHECKPOINT_PATH),
        "val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "per_class": {k: round(v, 4) for k, v in per_class.items()},
        "epochs": epochs,
        "elapsed_s": round(elapsed, 1),
        "num_params": num_params,
        "config": {
            "block_channels": list(BLOCK_CHANNELS),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "seq_len": SEQ_LEN,
            "mixup_alpha": MIXUP_ALPHA,
            "mixup_prob": MIXUP_PROB,
            "ema_decay": EMA_DECAY,
            "time_budget_s": MAX_TIME_BUDGET_SECONDS,
        },
    }
    RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"Run logged to {RUN_LOG_PATH} as '{RUN_NAME}'")


def main():
    print("=" * 70)
    print("POSE AUTORESEARCH - Training Run")
    print("=" * 70)
    print(f"Run: {RUN_NAME} -> {CHECKPOINT_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Model: Temporal CNN {BLOCK_CHANNELS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Time Budget: {MAX_TIME_BUDGET_SECONDS}s")
    print("=" * 70)

    n_bodies = 1
    splits_dir = Path("data/splits")
    if splits_dir.exists() and (splits_dir / "train").exists():
        print("Loading from pre-split directories (data/splits/) with multi-person tracking")
        n_bodies = 2
        train_ds = MultiPersonPoseDataset(splits_dir / "train", seq_len=SEQ_LEN, augment=True)
        val_ds = MultiPersonPoseDataset(splits_dir / "val", seq_len=SEQ_LEN, augment=False)
        test_ds = MultiPersonPoseDataset(splits_dir / "test", seq_len=SEQ_LEN, augment=False)

        pin = DEVICE.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=4, pin_memory=True,
        )
    else:
        print("No data/splits/ found, using random split via get_dataloaders()")
        train_loader, val_loader, test_loader = get_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=2,
            augment_train=True,
        )

    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    print()

    # Environment conditioning: infrastructure for future facility-specific env features.
    # FiLM conditioning layers are constructed but remain at identity initialization
    # (gamma=1, beta=0) until env feature extraction is run on real facility data.
    # See scripts/extract_env_features.py and docs/roboflow_guide.md.
    env_dim = 0
    env_features_dir = Path("data/env_features")
    if env_features_dir.exists():
        env_dim = 32  # 8 object classes × 4 features each
        print(f"Environment features found — conditioning with {env_dim}-dim context")

    model = PoseEventClassifier(
        dropout=DROPOUT,
        env_dim=env_dim,
        n_bodies=n_bodies,
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    num_batches = len(train_loader)
    estimated_epochs = max(int(MAX_TIME_BUDGET_SECONDS / 10), 30)  # rough estimate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=num_batches,
        epochs=estimated_epochs,
        pct_start=0.1,
        anneal_strategy="cos",
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

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    ema = ModelEMA(model, decay=EMA_DECAY)

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
            model, train_loader, optimizer, criterion, scheduler, DEVICE,
            scaler=scaler, ema=ema,
        )
        epoch_time = time.time() - epoch_start

        # Evaluate with EMA weights
        orig = ema.apply(model)
        val_acc, val_loss = evaluate_model(model, val_loader, DEVICE)
        ema.restore(model, orig)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": ema.shadow,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "run_name": RUN_NAME,
                    "block_channels": list(BLOCK_CHANNELS),
                },
                CHECKPOINT_PATH,
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
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
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

    record_run(
        best_val_acc=best_val_acc,
        test_acc=test_acc,
        test_loss=test_loss,
        per_class=per_class,
        epochs=epoch,
        elapsed=time.time() - start_time,
        num_params=num_params,
    )

    return best_val_acc


if __name__ == "__main__":
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    val_accuracy = main()
    print(f"\nFINAL VALIDATION ACCURACY: {val_accuracy:.4f}")
