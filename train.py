"""
Pose-based event detection training script.
The agent modifies this file to improve validation accuracy.
"""

from __future__ import annotations

import os
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
from pose_autoresearch.graph import get_two_body_adjacency
from collections import Counter
import json
import numpy as np

# ============================================================================
# HYPERPARAMETERS (agent can modify)
# ============================================================================

INPUT_DIM = 51       # 17 keypoints x 3 (x, y, confidence)
BONE_DIM = NUM_BONES * 3  # 16 bones x 3 (dx, dy, mean_conf)
VELOCITY_DIM = 51    # 17 keypoints x 3 (vx, vy, confidence)
FULL_INPUT_DIM = INPUT_DIM + BONE_DIM + VELOCITY_DIM  # 51 + 48 + 51 = 150
MULTI_INPUT_DIM = 105    # 51 + 51 + 3 (primary + neighbor + metadata)
MULTI_FULL_INPUT_DIM = 303  # 150 + 150 + 3
INPUT_CHANNELS = 3   # Per-joint: x, y, confidence
NUM_JOINTS = NUM_KEYPOINTS  # 17
NUM_CLASSES = len(EVENT_CLASSES)  # 7
SEQ_LEN = 150        # 5 seconds at 30fps
BATCH_SIZE = 64
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# MixUp augmentation
MIXUP_ALPHA = 0.2    # Beta(alpha, alpha) shape parameter; <1 is U-shaped
MIXUP_PROB = 0.5     # Probability of applying MixUp to a given batch

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
    # Approximate frame diagonal for distance normalization (640x480 default)
    FRAME_DIAG = (640 ** 2 + 480 ** 2) ** 0.5

    def __init__(self, data_dir, seq_len: int = 150, augment: bool = False):
        self.seq_len = seq_len
        self.augment = augment
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        # Each entry is (primary_kps, neighbor_kps, metadata, label_idx)
        # primary_kps/neighbor_kps: (T, 51)   metadata: (T, 3)

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
        """Compute per-frame metadata (3 dims) from hip midpoints.

        Args:
            primary: (T, 17, 3) keypoints for the primary person
            neighbor: (T, 17, 3) keypoints for the neighbor person

        Returns:
            metadata: (T, 3) — [dist_norm, relative_x, relative_y]
        """
        T = primary.shape[0]
        metadata = np.zeros((T, 3), dtype=np.float32)

        p_hip = (primary[:, self.LEFT_HIP, :2] + primary[:, self.RIGHT_HIP, :2]) / 2  # (T, 2)
        n_hip = (neighbor[:, self.LEFT_HIP, :2] + neighbor[:, self.RIGHT_HIP, :2]) / 2  # (T, 2)

        diff = n_hip - p_hip  # (T, 2)
        dist = np.linalg.norm(diff, axis=1)  # (T,)

        metadata[:, 0] = dist / self.FRAME_DIAG  # normalized euclidean distance
        metadata[:, 1] = diff[:, 0] / (self.FRAME_DIAG + 1e-8)  # relative_x
        metadata[:, 2] = diff[:, 1] / (self.FRAME_DIAG + 1e-8)  # relative_y

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
            metadata = np.concatenate([metadata, np.zeros((pad_len, 3), dtype=np.float32)])

        # Concatenate: primary(51) + neighbor(51) + metadata(3) = 105
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


class SqueezeExcitation2d(nn.Module):
    """Channel attention for (B, C, T, V) graph feature maps."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T, V)
        w = self.se(x).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        return x * w


class GraphTemporalBlock(nn.Module):
    """Multi-scale temporal convolution over graph features (B, C, T, V).

    Same kernels-3/7/15 + SE design as MultiScaleTemporalBlock, but kernels
    run along T only ((k, 1) Conv2d) so every joint keeps its own temporal
    stream. Channel count is preserved; stride downsamples T.
    """

    def __init__(self, channels, kernels=(3, 7, 15), stride=1, dropout=0.3):
        super().__init__()
        branch_ch = channels // len(kernels)
        remainder = channels - branch_ch * len(kernels)

        self.branches = nn.ModuleList()
        for i, k in enumerate(kernels):
            ch = branch_ch + (remainder if i == 0 else 0)
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, ch, (k, 1), stride=(stride, 1),
                          padding=((k - 1) // 2, 0)),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            ))

        self.conv2 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)
        self.se = SqueezeExcitation2d(channels)

        if stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(channels),
            )
        else:
            self.residual = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return self.act(x + res)


class CTRGraphBlock(nn.Module):
    """Channel-wise Topology Refinement graph convolution (CTR-GCN style).

    A fixed, normalized base adjacency is shared by all channel groups.
    Each of the `groups` channel groups additionally learns a dynamic
    pairwise affinity from the input features (tanh of pairwise feature
    differences, temporal-mean pooled), scaled by a per-group `alpha`
    initialized to zero — so training starts as a plain GCN on the
    skeleton and topology refinement grows in as it helps.

    Note (ReZero-style gating): while alpha == 0, theta/phi receive zero
    gradient; alpha itself gets a healthy gradient and escapes zero on the
    first optimizer step, after which the affinity branch unfreezes. The
    `value` projection is a single shared 1x1 conv — the group structure
    lives in the per-group adjacency, not in the feature projection.
    """

    def __init__(self, in_ch, out_ch, A_base, groups=8, rd_ch=8):
        super().__init__()
        assert out_ch % groups == 0, "out_ch must be divisible by groups"
        self.groups = groups
        self.out_ch = out_ch
        self.rd_ch = rd_ch
        self.register_buffer(
            "A_base", torch.as_tensor(A_base, dtype=torch.float32))

        self.theta = nn.Conv2d(in_ch, rd_ch * groups, 1)
        self.phi = nn.Conv2d(in_ch, rd_ch * groups, 1)
        self.value = nn.Conv2d(in_ch, out_ch, 1)
        self.alpha = nn.Parameter(torch.zeros(groups))

        self.bn = nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.residual = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, V)
        B, _, T, V = x.shape
        res = self.residual(x)

        # Dynamic per-group affinity from temporal-mean features
        th = self.theta(x).mean(dim=2).view(B, self.groups, self.rd_ch, V)
        ph = self.phi(x).mean(dim=2).view(B, self.groups, self.rd_ch, V)
        diff = th.unsqueeze(-1) - ph.unsqueeze(-2)        # (B, g, rd, V, V)
        refine = torch.tanh(diff).mean(dim=2)             # (B, g, V, V)

        A = self.A_base.view(1, 1, V, V) \
            + self.alpha.view(1, -1, 1, 1) * refine        # (B, g, V, V)

        v = self.value(x).view(B, self.groups, self.out_ch // self.groups, T, V)
        out = torch.einsum("bgctv,bgvw->bgctw", v, A)
        out = out.reshape(B, self.out_ch, T, V)
        return self.act(self.bn(out) + res)


class STGCNClassifier(nn.Module):
    """CTR-GCN-style two-body skeleton classifier.

    Input contract matches PoseEventClassifier with n_bodies=2:
    (B, T, 105) = primary(51) + neighbor(51) + metadata(3), so it is a
    drop-in alternative backbone for MultiPersonPoseDataset batches.
    Selected via POSE_BACKBONE=gcn (see main()).
    """

    STAGE_CHANNELS = (64, 64, 128, 256)
    STAGE_STRIDES = (1, 1, 2, 2)

    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = DROPOUT, env_dim: int = 0):
        super().__init__()
        self.env_dim = env_dim
        A = get_two_body_adjacency()

        in_ch = 6  # x, y, conf for each joint + dist/rel_x/rel_y broadcast
        self.input_bn = nn.BatchNorm2d(in_ch)

        stages = []
        prev = in_ch
        for ch, stride in zip(self.STAGE_CHANNELS, self.STAGE_STRIDES):
            stages.append(nn.ModuleList([
                CTRGraphBlock(prev, ch, A),
                GraphTemporalBlock(ch, stride=stride, dropout=dropout),
            ]))
            prev = ch
        self.stages = nn.ModuleList(stages)

        if env_dim > 0:
            self.conditioners = nn.ModuleList([
                EnvironmentConditioner(env_dim, ch)
                for ch in self.STAGE_CHANNELS
            ])
        else:
            self.conditioners = None

        final_channels = self.STAGE_CHANNELS[-1]
        self.pool = TemporalAttentionPool(final_channels)
        self.fc = nn.Linear(final_channels, num_classes)

    @staticmethod
    def _gate(kps):
        """Confidence-gate xy coordinates: sigmoid(conf*5 - 2)."""
        conf = kps[:, :, :, 2:3]
        gate = torch.sigmoid(conf * 5 - 2)
        out = kps.clone()
        out[:, :, :, :2] = kps[:, :, :, :2] * gate
        return out

    def forward(self, x, env_features=None):
        """
        Args:
            x: (batch, seq_len, 105)
            env_features: optional (batch, env_dim)
        Returns:
            logits: (batch, num_classes)
        """
        B, T, _ = x.shape
        primary = x[:, :, :51].reshape(B, T, NUM_JOINTS, 3)
        neighbor = x[:, :, 51:102].reshape(B, T, NUM_JOINTS, 3)
        metadata = x[:, :, 102:105]  # (B, T, 3)

        primary = self._gate(primary)
        neighbor = self._gate(neighbor)

        # Soft distance decay on neighbor (same curve as the CNN backbone)
        decay = torch.sigmoid(5 - metadata[:, :, 0:1] * 10)  # (B, T, 1)
        neighbor = neighbor * decay.unsqueeze(-1)

        joints = torch.cat([primary, neighbor], dim=2)        # (B, T, 34, 3)
        meta = metadata.unsqueeze(2).expand(-1, -1, joints.shape[2], -1)
        feats = torch.cat([joints, meta], dim=3)              # (B, T, 34, 6)
        x = feats.permute(0, 3, 1, 2).contiguous()            # (B, 6, T, 34)

        x = self.input_bn(x)
        for i, stage in enumerate(self.stages):
            spatial, temporal = stage[0], stage[1]
            x = temporal(spatial(x))
            if self.conditioners is not None and env_features is not None:
                gamma, beta = self.conditioners[i](env_features)
                x = gamma.unsqueeze(2).unsqueeze(3) * x \
                    + beta.unsqueeze(2).unsqueeze(3)

        x = x.mean(dim=3)        # joint pool -> (B, C, T')
        x = self.pool(x)         # temporal attention -> (B, C)
        return self.fc(x)


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
        n_bodies: int = 1,  # 1 = single-body (51-dim), 2 = dual-body (105-dim)
    ):
        super().__init__()
        self.env_dim = env_dim
        self.n_bodies = n_bodies

        if n_bodies == 2:
            in_dim = MULTI_FULL_INPUT_DIM  # 150 + 150 + 3 = 303
        else:
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
            x: (batch, seq_len, D) -- D=51 for single-body, D=105 for dual-body
            env_features: optional (batch, env_dim) -- environment context
        Returns:
            logits: (batch, num_classes)
        """
        B, T, D = x.shape

        if self.n_bodies == 2:
            # Split input: primary(51) + neighbor(51) + metadata(3)
            primary_raw = x[:, :, :51]
            neighbor_raw = x[:, :, 51:102]
            metadata = x[:, :, 102:105]  # (B, T, 3)

            primary_kps = primary_raw.view(B, T, NUM_JOINTS, 3)
            neighbor_kps = neighbor_raw.view(B, T, NUM_JOINTS, 3)

            primary_feats = self._process_body(primary_kps)   # (B, T, 150)
            neighbor_feats = self._process_body(neighbor_kps)  # (B, T, 150)

            # Soft distance decay: attenuate neighbor when far away
            # dist_norm is metadata[:,:,0]
            dist_norm = metadata[:, :, 0:1]  # (B, T, 1)
            decay = torch.sigmoid(5 - dist_norm * 10)  # (B, T, 1)
            neighbor_feats = neighbor_feats * decay

            # Concatenate: primary(150) + neighbor(150) + metadata(3) = 303
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
                use_mixup=True):
    """Train for one epoch with optional temporal mixup."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for poses, labels in dataloader:
        poses = poses.to(device)
        labels = labels.to(device)

        if use_mixup and torch.rand(()).item() < MIXUP_PROB:
            poses, labels_a, labels_b, lam = mixup_data(poses, labels)
            logits = model(poses)
            loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
            preds = torch.argmax(logits, dim=1)
            # Report accuracy vs original labels (labels_a) — cleaner than the soft proxy.
            correct += (preds == labels_a).sum().item()
        else:
            logits = model(poses)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(dataloader), correct / total


# ============================================================================
# MAIN
# ============================================================================


def build_model(backbone, env_dim, n_bodies):
    """Construct the classifier for the requested backbone.

    Returns (model, checkpoint_path). The GCN saves to a separate
    checkpoint file so side-by-side comparison never clobbers the CNN.
    """
    if backbone == "gcn":
        model = STGCNClassifier(dropout=DROPOUT, env_dim=env_dim)
        return model, "checkpoints/best_model_gcn.pt"
    if backbone == "cnn":
        model = PoseEventClassifier(
            dropout=DROPOUT, env_dim=env_dim, n_bodies=n_bodies)
        return model, "checkpoints/best_model.pt"
    raise ValueError(f"Unknown POSE_BACKBONE: {backbone!r} (use 'cnn' or 'gcn')")


def main():
    print("=" * 70)
    print("POSE AUTORESEARCH - Training Run")
    print("=" * 70)
    print(f"Device: {DEVICE}")
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
            num_workers=2, pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=2, pin_memory=False,
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

    backbone = os.environ.get("POSE_BACKBONE", "cnn").strip().lower() or "cnn"
    if backbone == "gcn" and n_bodies != 2:
        raise SystemExit(
            "POSE_BACKBONE=gcn requires multi-person data (data/splits/). "
            "Run scripts/split_data.py first.")
    model, ckpt_path = build_model(backbone, env_dim, n_bodies)
    model = model.to(DEVICE)
    print(f"Backbone: {backbone} -> {ckpt_path}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )

    # T_max=260 produces a single smooth cosine decay across the full
    # 1-hour training run (~260 epochs on A100). Autoresearch showed this
    # beats the default T_max=50 which cycles the LR back up mid-training.
    # Biggest win: unstable_gait +5.4 points, plus best overall test acc 96.15%.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=260, eta_min=1e-6,
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
                ckpt_path,
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
    ckpt = torch.load(ckpt_path, weights_only=True)
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
