"""Skeleton-specific data augmentation for pose sequences.

These augmentations operate on (seq_len, 17, 3) tensors where the
last dimension is (x, y, confidence). They preserve the skeleton
structure and produce physically plausible variations.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def random_rotation(poses: torch.Tensor, max_degrees: float = 15.0) -> torch.Tensor:
    """Rotate all keypoints around the center by a random angle.

    Args:
        poses: (seq_len, 17, 3) tensor -- (x, y, confidence).
        max_degrees: Maximum rotation angle.

    Returns:
        Rotated poses, same shape.
    """
    angle = random.uniform(-max_degrees, max_degrees) * (np.pi / 180)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    xy = poses[:, :, :2]
    conf = poses[:, :, 2:3]
    mask = conf > 0.1
    visible_count = mask.sum(dim=(0, 1)).clamp(min=1)
    center = (xy * mask).sum(dim=(0, 1)) / visible_count

    centered = xy - center
    rotated = torch.stack([
        centered[:, :, 0] * cos_a - centered[:, :, 1] * sin_a,
        centered[:, :, 0] * sin_a + centered[:, :, 1] * cos_a,
    ], dim=-1) + center

    return torch.cat([rotated, conf], dim=-1)


def random_scale(
    poses: torch.Tensor,
    scale_range: tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """Scale keypoints around center of mass.

    Args:
        poses: (seq_len, 17, 3) tensor.
        scale_range: (min_scale, max_scale).

    Returns:
        Scaled poses, same shape.
    """
    scale = random.uniform(*scale_range)

    xy = poses[:, :, :2]
    conf = poses[:, :, 2:3]
    mask = conf > 0.1
    visible_count = mask.sum(dim=(0, 1)).clamp(min=1)
    center = (xy * mask).sum(dim=(0, 1)) / visible_count

    scaled = (xy - center) * scale + center
    return torch.cat([scaled, conf], dim=-1)


def random_horizontal_flip(poses: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Mirror keypoints left<->right with probability p.

    Swaps left/right joint pairs and flips x coordinates.

    Args:
        poses: (seq_len, 17, 3) tensor.
        p: Flip probability.

    Returns:
        Possibly flipped poses.
    """
    if random.random() > p:
        return poses

    # COCO-17 left<->right pairs
    swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    flipped = poses.clone()

    # Flip x coordinate around center
    center_x = flipped[:, :, 0].mean()
    flipped[:, :, 0] = 2 * center_x - flipped[:, :, 0]

    # Swap left/right joints
    for left, right in swap_pairs:
        flipped[:, [left, right]] = flipped[:, [right, left]]

    return flipped


def random_temporal_crop(
    poses: torch.Tensor,
    target_len: int,
) -> torch.Tensor:
    """Randomly crop a temporal window from the sequence.

    Args:
        poses: (seq_len, 17, 3) tensor.
        target_len: Desired output length.

    Returns:
        (target_len, 17, 3) cropped tensor.
    """
    seq_len = poses.shape[0]
    if seq_len <= target_len:
        pad = torch.zeros(target_len - seq_len, 17, 3)
        return torch.cat([poses, pad], dim=0)

    start = random.randint(0, seq_len - target_len)
    return poses[start : start + target_len]


def random_temporal_resample(
    poses: torch.Tensor,
    speed_range: tuple[float, float] = (0.8, 1.2),
) -> torch.Tensor:
    """Resample temporal axis to simulate speed variation.

    Args:
        poses: (seq_len, 17, 3) tensor.
        speed_range: (min_speed, max_speed) multiplier.

    Returns:
        Resampled poses, same length as input.
    """
    seq_len = poses.shape[0]
    speed = random.uniform(*speed_range)
    new_len = max(2, int(seq_len * speed))

    indices = torch.linspace(0, seq_len - 1, new_len).long()
    resampled = poses[indices]

    if new_len < seq_len:
        pad = torch.zeros(seq_len - new_len, 17, 3)
        return torch.cat([resampled, pad], dim=0)
    else:
        return resampled[:seq_len]


def add_gaussian_noise(
    poses: torch.Tensor,
    std: float = 0.01,
) -> torch.Tensor:
    """Add Gaussian noise to joint positions (not confidence).

    Args:
        poses: (seq_len, 17, 3) tensor.
        std: Noise standard deviation (relative to coordinate range).

    Returns:
        Noisy poses, same shape.
    """
    noisy = poses.clone()
    noise = torch.randn_like(noisy[:, :, :2]) * std
    noisy[:, :, :2] += noise
    return noisy


def random_joint_dropout(
    poses: torch.Tensor,
    drop_prob: float = 0.05,
) -> torch.Tensor:
    """Randomly zero out joints to simulate occlusion.

    Args:
        poses: (seq_len, 17, 3) tensor.
        drop_prob: Per-joint drop probability.

    Returns:
        Poses with some joints zeroed out.
    """
    dropped = poses.clone()
    mask = torch.rand(17) > drop_prob
    dropped[:, ~mask, :] = 0.0
    return dropped


def augment_pose_sequence(
    poses: torch.Tensor,
    target_len: int | None = None,
) -> torch.Tensor:
    """Apply a random combination of augmentations.

    Args:
        poses: (seq_len, 17, 3) tensor.
        target_len: If set, crop/pad to this length.

    Returns:
        Augmented poses.
    """
    if random.random() < 0.5:
        poses = random_rotation(poses)
    if random.random() < 0.5:
        poses = random_scale(poses)
    if random.random() < 0.5:
        poses = random_horizontal_flip(poses)
    if random.random() < 0.3:
        poses = random_temporal_resample(poses)
    if random.random() < 0.5:
        poses = add_gaussian_noise(poses)
    if random.random() < 0.3:
        poses = random_joint_dropout(poses)
    if target_len is not None:
        poses = random_temporal_crop(poses, target_len)
    return poses
