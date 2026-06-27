"""Skeleton graph topology for GCN-based pose models.

Defines adjacency matrices for COCO-17 keypoint skeleton, plus
multi-hop and self-loop variants used by ST-GCN and CTR-GCN
style architectures.
"""

from __future__ import annotations

import numpy as np
import torch


# COCO-17 skeleton edges (bidirectional)
COCO_17_EDGES = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),       # nose <-> eyes <-> ears
    # Torso
    (5, 6),                                  # left_shoulder <-> right_shoulder
    (5, 11), (6, 12),                        # shoulders <-> hips
    (11, 12),                                # left_hip <-> right_hip
    # Left arm
    (5, 7), (7, 9),                          # shoulder -> elbow -> wrist
    # Right arm
    (6, 8), (8, 10),                         # shoulder -> elbow -> wrist
    # Left leg
    (11, 13), (13, 15),                      # hip -> knee -> ankle
    # Right leg
    (12, 14), (14, 16),                      # hip -> knee -> ankle
    # Head -> torso
    (0, 5), (0, 6),                          # nose <-> shoulders (implicit)
]

NUM_JOINTS = 17
CENTER_JOINT = 0  # Nose as center (or use hip midpoint)


def get_adjacency_matrix(
    edges: list[tuple[int, int]] = COCO_17_EDGES,
    num_joints: int = NUM_JOINTS,
    self_loops: bool = True,
) -> np.ndarray:
    """Build binary adjacency matrix from edge list.

    Args:
        edges: List of (i, j) joint connections.
        num_joints: Number of joints.
        self_loops: Add self-connections on diagonal.

    Returns:
        (num_joints, num_joints) binary adjacency matrix.
    """
    A = np.zeros((num_joints, num_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loops:
        np.fill_diagonal(A, 1.0)
    return A


def get_normalized_adjacency(
    edges: list[tuple[int, int]] = COCO_17_EDGES,
    num_joints: int = NUM_JOINTS,
) -> np.ndarray:
    """Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}.

    Standard normalization for GCN (Kipf & Welling 2017).
    """
    A = get_adjacency_matrix(edges, num_joints, self_loops=True)
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-6)))
    return D_inv_sqrt @ A @ D_inv_sqrt


def get_spatial_partitioning(
    edges: list[tuple[int, int]] = COCO_17_EDGES,
    num_joints: int = NUM_JOINTS,
    center: int = CENTER_JOINT,
) -> np.ndarray:
    """ST-GCN spatial partitioning: 3 subsets per joint.

    For each joint i and its neighbor j:
      - Subset 0: j == i (self-loop)
      - Subset 1: j is closer to center than i
      - Subset 2: j is farther from center than i

    Returns:
        (3, num_joints, num_joints) partitioned adjacency.
    """
    A = get_adjacency_matrix(edges, num_joints, self_loops=False)

    # BFS distance from center
    dist = _bfs_distance(edges, num_joints, center)

    partitions = np.zeros((3, num_joints, num_joints), dtype=np.float32)

    for i in range(num_joints):
        for j in range(num_joints):
            if i == j:
                partitions[0, i, j] = 1.0
            elif A[i, j] > 0:
                if dist[j] <= dist[i]:
                    partitions[1, i, j] = 1.0  # Centripetal
                else:
                    partitions[2, i, j] = 1.0  # Centrifugal

    # Normalize each partition
    for k in range(3):
        D = partitions[k].sum(axis=1)
        D[D == 0] = 1.0
        partitions[k] /= D[:, None]

    return partitions


def get_bone_pairs() -> list[tuple[int, int]]:
    """Return directed bone vectors (parent -> child) for COCO-17.

    Each bone is a vector from parent joint to child joint.
    Used for the bone-stream input in multi-stream models.
    """
    return [
        (0, 1), (0, 2), (1, 3), (2, 4),     # Head
        (5, 7), (7, 9),                       # Left arm
        (6, 8), (8, 10),                      # Right arm
        (5, 11), (11, 13), (13, 15),          # Left leg
        (6, 12), (12, 14), (14, 16),          # Right leg
        (5, 6), (11, 12),                     # Cross-body
    ]


def adjacency_to_tensor(A: np.ndarray) -> torch.Tensor:
    """Convert numpy adjacency to torch tensor."""
    return torch.from_numpy(A).float()


def _bfs_distance(
    edges: list[tuple[int, int]],
    num_nodes: int,
    source: int,
) -> list[int]:
    """BFS shortest distance from source to all nodes."""
    adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    dist = [-1] * num_nodes
    dist[source] = 0
    queue = [source]
    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if dist[neighbor] == -1:
                dist[neighbor] = dist[node] + 1
                queue.append(neighbor)

    # Unreachable nodes get max distance
    max_d = max(d for d in dist if d >= 0)
    return [d if d >= 0 else max_d + 1 for d in dist]


# Cross-body seed edges for the unified two-person graph.
# Primary joints are 0-16, neighbor joints are 17-33 (COCO index + 17).
TWO_BODY_CROSS_EDGES = [
    (11, 11 + NUM_JOINTS), (12, 12 + NUM_JOINTS),   # hip <-> hip
    (9, 9 + NUM_JOINTS), (10, 10 + NUM_JOINTS),     # wrist <-> wrist
    (9, 0 + NUM_JOINTS), (10, 0 + NUM_JOINTS),      # primary wrists <-> neighbor nose
]


def get_two_body_adjacency() -> np.ndarray:
    """Symmetric-normalized adjacency for a unified two-person skeleton graph.

    Joints 0..16 are the primary person, 17..33 the neighbor. Intra-body
    edges duplicate COCO_17_EDGES at a +17 offset; cross-body seed edges
    connect hips, wrists, and primary-wrist-to-neighbor-head so interaction
    geometry (aggression, working together) is reachable in one hop. The
    CTR mechanism refines this base dynamically at runtime.

    Returns:
        (34, 34) float32, D^{-1/2} (A + I) D^{-1/2} normalized.
    """
    edges = list(COCO_17_EDGES)
    edges += [(i + NUM_JOINTS, j + NUM_JOINTS) for i, j in COCO_17_EDGES]
    edges += TWO_BODY_CROSS_EDGES
    A = get_adjacency_matrix(edges, NUM_JOINTS * 2, self_loops=True)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(A.sum(axis=1), 1e-6)))
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)
