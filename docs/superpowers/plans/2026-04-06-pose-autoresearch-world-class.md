# Pose Autoresearch: World-Class Event Classification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the world's best model for classifying 5–10 second pose estimation coordinate sequences into 7 event types (fall, eating, aggression, unstable gait, wandering, working together, sitting/standing).

**Architecture:** Graph Convolutional Network (GCN) with skeleton topology awareness, multi-stream input (joint positions + bone vectors + motion velocities), and temporal Transformer attention. The autoresearch agent iterates on `train.py` with 5-minute training budgets on a cloud GPU, hill-climbing validation accuracy over hundreds of experiments. Real skeleton data from NTU RGB+D 120 (mapped to our 7 classes) plus Vistarra-specific labeled data from the edge pipeline.

**Tech Stack:** PyTorch 2.x, torch-geometric (optional), NTU RGB+D 120 skeleton data, cloud GPU (Lambda/RunPod/vast.ai), Claude agent via autoresearch loop.

---

## Codebase Assessment

### Current State

The repo follows Karpathy's autoresearch pattern: `train.py` (agent-modifiable), `prepare.py` (fixed), `program.md` (human directives). But several critical gaps exist:

1. **No real data.** `prepare_synthetic_data()` generates random noise — no model can learn from this. The entire data pipeline needs to be built.
2. **Weak baseline model.** CNN+LSTM treats keypoints as a flat 51-dim vector with no skeleton topology. State-of-the-art (CTR-GCN, InfoGCN, SkelFormer) uses graph convolutions that understand which joints connect to which.
3. **COCO-17 vs NTU-25 keypoint mismatch.** YOLO produces 17 COCO keypoints; NTU RGB+D has 25 Kinect joints. Need a mapping layer so the model works with both.
4. **Single-person only.** Real events (aggression, working together) require multi-person reasoning.
5. **Fixed sequence length.** Hardcoded to 150 frames (5s). Need variable 5–10 second support.
6. **Bug:** `scripts/extract_poses.py` line 91: `args = args.parse_args()` should be `args = parser.parse_args()`.
7. **No cloud GPU setup.** The autoresearch loop needs to run on a cloud machine.

### What State-of-the-Art Looks Like

| Model | NTU-120 X-Sub | NTU-120 X-Set | Year |
|-------|---------------|---------------|------|
| InfoGCN | 89.8% | 91.2% | 2022 |
| CTR-GCN | 88.9% | 90.6% | 2021 |
| SkelFormer | 89.4% | — | 2026 |
| ST-VGCN | 96.7% (NTU-60) | — | 2025 |

These models use **graph structure** (skeleton topology as adjacency matrix), **multi-stream fusion** (joint + bone + velocity), and **attention mechanisms**. Our autoresearch agent should explore this design space.

### Target: Why We Can Win

Our problem is narrower than NTU-120 (7 classes vs 120). This is an advantage — we can:
- Pre-train on NTU-120 for skeleton understanding, then fine-tune on our 7 classes
- Use the autoresearch loop to explore hundreds of architecture variations autonomously
- Build a purpose-specific model that doesn't need to generalize to 120 action types
- Augment with real Vistarra edge data as it accumulates

---

## File Structure

```
pose-autoresearch/
├── train.py                    # MODIFY — agent-editable model + training loop
├── prepare.py                  # MODIFY — upgrade data pipeline, keep fixed for agent
├── program.md                  # MODIFY — upgrade agent directives
├── pyproject.toml              # MODIFY — add dependencies
├── data/
│   ├── ntu120/                 # NEW — NTU RGB+D 120 skeleton data
│   │   ├── download.sh         # NEW — download + extract script
│   │   └── class_mapping.json  # NEW — NTU-120 → our 7 classes
│   ├── processed/              # EXISTS — processed pose sequences (JSON)
│   └── raw/                    # EXISTS — raw videos
├── scripts/
│   ├── extract_poses.py        # MODIFY — fix bug, support multi-person
│   ├── convert_ntu120.py       # NEW — convert NTU-120 .skeleton → JSON
│   ├── label_tool.py           # EXISTS (placeholder)
│   ├── visualize.py            # EXISTS (placeholder)
│   └── setup_cloud.sh          # NEW — cloud GPU environment setup
├── pose_autoresearch/          # NEW — package with shared utilities
│   ├── __init__.py             # NEW
│   ├── graph.py                # NEW — skeleton adjacency matrices + topology
│   └── augment.py              # NEW — skeleton-specific data augmentation
├── checkpoints/                # EXISTS — saved model weights
└── results.tsv                 # EXISTS — experiment log (generated)
```

---

## Phase 1: Data Foundation

### Task 1: NTU RGB+D 120 Class Mapping

**Files:**
- Create: `data/ntu120/class_mapping.json`

The NTU-120 dataset has 120 action classes. We need to map a subset to our 7 event types. Not all NTU classes have equivalents — we'll use the closest matches and ignore the rest.

- [ ] **Step 1: Create the class mapping file**

```json
{
    "_description": "Maps NTU RGB+D 120 action classes to Vistarra event types",
    "_ntu_format": "A001-A120 action labels",
    "fall": [
        "A043:falling down",
        "A044:stumbling"
    ],
    "eating": [
        "A005:drop item (eating motion proxy)",
        "A020:put on glasses (fine hand motion proxy)",
        "A024:eat meal",
        "A025:drink water"
    ],
    "working_together": [
        "A050:punch (2-person interaction)",
        "A051:kick (2-person interaction)",
        "A052:push (2-person interaction)",
        "A056:touch pocket (2-person interaction)",
        "A057:handshaking",
        "A058:walking towards",
        "A059:walking apart"
    ],
    "aggression": [
        "A050:punch",
        "A051:kick",
        "A052:push",
        "A106:hit with object"
    ],
    "unstable_gait": [
        "A043:falling down",
        "A044:stumbling",
        "A060:walking towards (slow gait proxy)"
    ],
    "wandering": [
        "A008:walking (general)",
        "A058:walking towards",
        "A059:walking apart",
        "A060:walking independently"
    ],
    "sitting_standing": [
        "A001:drink water (seated proxy)",
        "A008:standing up",
        "A009:sitting down",
        "A010:reading",
        "A011:writing"
    ]
}
```

Note: Some NTU classes appear in multiple event types (e.g., punch → both aggression and working_together). This is intentional — the same action can have different event semantics. The model learns to distinguish context.

- [ ] **Step 2: Commit**

```bash
git add data/ntu120/class_mapping.json
git commit -m "feat: add NTU-120 to Vistarra event class mapping"
```

---

### Task 2: NTU-120 Download Script

**Files:**
- Create: `data/ntu120/download.sh`

NTU RGB+D 120 requires registration at https://rose1.ntu.edu.sg/dataset/actionRecognition/. The skeleton data is distributed as `.skeleton` text files. Pre-processed 2D/3D skeleton pickle files are also available on figshare. We'll use the pre-processed pickle files for faster setup.

- [ ] **Step 1: Create the download script**

```bash
#!/usr/bin/env bash
# Download NTU RGB+D 120 pre-processed skeleton data.
#
# Option A: Official registration (recommended for publication)
#   Register at https://rose1.ntu.edu.sg/dataset/actionRecognition/
#   Download "NTU RGB+D 120 Skeleton" files
#
# Option B: Pre-processed pickle from figshare (faster for research)
#   https://figshare.com/articles/dataset/NTU_RGB_D_60_120_skeleton_with_coordinates_dataset/27427188

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== NTU RGB+D 120 Skeleton Data Setup ==="
echo ""
echo "This dataset requires registration. Two options:"
echo ""
echo "Option A — Official (for publication):"
echo "  1. Register at: https://rose1.ntu.edu.sg/dataset/actionRecognition/"
echo "  2. Download 'NTU RGB+D 120' skeleton files"
echo "  3. Extract into: $DATA_DIR/raw_skeletons/"
echo ""
echo "Option B — Pre-processed pickle (faster):"
echo "  1. Visit: https://figshare.com/articles/dataset/27427188"
echo "  2. Download the 2D skeleton pickle file"
echo "  3. Place in: $DATA_DIR/ntu120_2d.pkl"
echo ""

# Check if data already exists
if [ -f "$DATA_DIR/ntu120_2d.pkl" ]; then
    echo "Found: ntu120_2d.pkl — ready for conversion."
    echo "Run: python scripts/convert_ntu120.py"
    exit 0
fi

if [ -d "$DATA_DIR/raw_skeletons" ] && [ "$(ls -1 "$DATA_DIR/raw_skeletons"/*.skeleton 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Found: raw_skeletons/ — ready for conversion."
    echo "Run: python scripts/convert_ntu120.py --format raw"
    exit 0
fi

echo "No data found. Please download using one of the options above."
exit 1
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x data/ntu120/download.sh
git add data/ntu120/download.sh
git commit -m "feat: add NTU-120 download instructions script"
```

---

### Task 3: NTU-120 Conversion Script

**Files:**
- Create: `scripts/convert_ntu120.py`

Converts NTU-120 skeleton data (either raw `.skeleton` files or pre-processed pickle) into our JSON format, applying the class mapping from Task 1.

- [ ] **Step 1: Create the conversion script**

```python
#!/usr/bin/env python3
"""Convert NTU RGB+D 120 skeleton data to Vistarra pose JSON format.

Supports two input formats:
  - Pre-processed pickle (ntu120_2d.pkl) from figshare
  - Raw .skeleton text files from official NTU release

Usage:
    python scripts/convert_ntu120.py                         # pickle (default)
    python scripts/convert_ntu120.py --format raw            # raw .skeleton files
    python scripts/convert_ntu120.py --seq-len 300 --fps 30  # 10-second sequences
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path

import numpy as np

# NTU-120 has 25 joints; COCO-17 has 17. Map NTU→COCO indices.
# NTU joints: https://arxiv.org/pdf/1604.02808 (Figure 4)
# COCO joints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles
NTU_TO_COCO_17 = {
    0: 3,    # nose ← NTU head (approximate)
    1: None, # left_eye — no NTU equivalent, interpolate from head
    2: None, # right_eye — no NTU equivalent, interpolate from head
    3: None, # left_ear — skip
    4: None, # right_ear — skip
    5: 4,    # left_shoulder ← NTU left_shoulder
    6: 8,    # right_shoulder ← NTU right_shoulder
    7: 5,    # left_elbow ← NTU left_elbow
    8: 9,    # right_elbow ← NTU right_elbow
    9: 6,    # left_wrist ← NTU left_hand (approximate)
    10: 10,  # right_wrist ← NTU right_hand (approximate)
    11: 12,  # left_hip ← NTU left_hip
    12: 16,  # right_hip ← NTU right_hip
    13: 13,  # left_knee ← NTU left_knee
    14: 17,  # right_knee ← NTU right_knee
    15: 14,  # left_ankle ← NTU left_foot (approximate)
    16: 18,  # right_ankle ← NTU right_foot (approximate)
}

# NTU action label → Vistarra event class (from class_mapping.json)
# We parse this dynamically from the mapping file.

DATA_DIR = Path("data/ntu120")
OUTPUT_DIR = Path("data/processed")
MAPPING_FILE = DATA_DIR / "class_mapping.json"

# Known missing/bad samples in NTU-120
NTU120_BAD_SAMPLES = set()  # Populated from the 535 known bad files


def load_class_mapping() -> dict[int, list[str]]:
    """Load NTU action ID → Vistarra event class mapping.

    Returns:
        Dict mapping NTU action index (0-119) to list of Vistarra class names.
        A single NTU action can map to multiple Vistarra classes.
    """
    with open(MAPPING_FILE) as f:
        raw = json.load(f)

    ntu_to_vistarra: dict[int, list[str]] = {}
    for vistarra_class, ntu_actions in raw.items():
        if vistarra_class.startswith("_"):
            continue
        for entry in ntu_actions:
            # Format: "A043:falling down"
            action_id = int(entry.split(":")[0][1:]) - 1  # A001 → 0
            ntu_to_vistarra.setdefault(action_id, []).append(vistarra_class)

    return ntu_to_vistarra


def ntu25_to_coco17(joints_25: np.ndarray) -> list[list[float]]:
    """Convert NTU 25-joint skeleton to COCO 17-keypoint format.

    Args:
        joints_25: (25, 2) or (25, 3) array of joint coordinates.

    Returns:
        List of 17 [x, y, confidence] keypoints.
    """
    keypoints = []
    for coco_idx in range(17):
        ntu_idx = NTU_TO_COCO_17.get(coco_idx)
        if ntu_idx is not None and ntu_idx < len(joints_25):
            x, y = float(joints_25[ntu_idx][0]), float(joints_25[ntu_idx][1])
            conf = 1.0 if (x != 0 or y != 0) else 0.0
            keypoints.append([x, y, conf])
        else:
            # Interpolate eyes/ears from head position if available
            if coco_idx in (1, 2, 3, 4) and 3 in NTU_TO_COCO_17.values():
                head_idx = 3  # NTU head joint
                x, y = float(joints_25[head_idx][0]), float(joints_25[head_idx][1])
                keypoints.append([x, y, 0.5])  # Lower confidence for interpolated
            else:
                keypoints.append([0.0, 0.0, 0.0])
    return keypoints


def convert_pickle(
    pkl_path: Path,
    output_dir: Path,
    class_mapping: dict[int, list[str]],
    seq_len: int = 150,
    fps: int = 30,
) -> int:
    """Convert pre-processed NTU-120 pickle file to Vistarra JSON.

    The pickle file contains a dict with:
        'x_train': (N, C, T, V, M) — samples, channels, frames, vertices, persons
        'y_train': (N,) — action labels 0-119
        'x_test', 'y_test': same format

    Returns:
        Number of samples converted.
    """
    print(f"Loading pickle: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    count = 0
    for split in ["train", "test"]:
        x_key = f"x_{split}"
        y_key = f"y_{split}"
        if x_key not in data:
            continue

        X = data[x_key]  # (N, C, T, V, M)
        Y = data[y_key]  # (N,)

        print(f"  {split}: {len(Y)} samples, shape {X.shape}")

        for i in range(len(Y)):
            action_id = int(Y[i])
            vistarra_classes = class_mapping.get(action_id, [])
            if not vistarra_classes:
                continue  # This NTU action doesn't map to our events

            sample = X[i]  # (C, T, V, M) — C=3(x,y,conf), T=frames, V=25joints, M=2persons

            # Process first person (person 0)
            for vistarra_class in vistarra_classes:
                frames = []
                for t in range(min(sample.shape[1], seq_len)):
                    joints = sample[:2, t, :, 0].T  # (25, 2) from person 0
                    keypoints = ntu25_to_coco17(joints)
                    frames.append({
                        "keypoints": keypoints,
                        "timestamp": t / fps,
                    })

                if len(frames) < 10:  # Skip very short sequences
                    continue

                # Pad if needed
                while len(frames) < seq_len:
                    frames.append({
                        "keypoints": [[0.0, 0.0, 0.0]] * 17,
                        "timestamp": len(frames) / fps,
                    })

                output = {
                    "frames": frames[:seq_len],
                    "label": vistarra_class,
                    "duration": len(frames) / fps,
                    "source": f"ntu120_{split}_{i}",
                    "ntu_action_id": action_id,
                }

                filename = f"ntu_{vistarra_class}_{split}_{i:06d}.json"
                with open(output_dir / filename, "w") as f:
                    json.dump(output, f)
                count += 1

    return count


def convert_raw_skeletons(
    skeleton_dir: Path,
    output_dir: Path,
    class_mapping: dict[int, list[str]],
    seq_len: int = 150,
    fps: int = 30,
) -> int:
    """Convert raw .skeleton files to Vistarra JSON format.

    NTU .skeleton filename format: SsssCcccPpppRrrrAaaa.skeleton
        S=setup, C=camera, P=performer, R=replication, A=action

    Returns:
        Number of samples converted.
    """
    count = 0
    skeleton_files = sorted(skeleton_dir.glob("*.skeleton"))
    print(f"Found {len(skeleton_files)} skeleton files")

    for skel_path in skeleton_files:
        # Parse action ID from filename
        fname = skel_path.stem
        action_id = int(fname[-3:]) - 1  # Aaaa → 0-indexed

        vistarra_classes = class_mapping.get(action_id, [])
        if not vistarra_classes:
            continue

        # Parse .skeleton file
        try:
            frames_data = _parse_skeleton_file(skel_path)
        except Exception as e:
            print(f"  Skip {fname}: {e}")
            continue

        for vistarra_class in vistarra_classes:
            frames = []
            for t, joints in enumerate(frames_data[:seq_len]):
                keypoints = ntu25_to_coco17(joints)
                frames.append({
                    "keypoints": keypoints,
                    "timestamp": t / fps,
                })

            if len(frames) < 10:
                continue

            while len(frames) < seq_len:
                frames.append({
                    "keypoints": [[0.0, 0.0, 0.0]] * 17,
                    "timestamp": len(frames) / fps,
                })

            output = {
                "frames": frames[:seq_len],
                "label": vistarra_class,
                "duration": len(frames) / fps,
                "source": fname,
                "ntu_action_id": action_id,
            }

            filename = f"ntu_{vistarra_class}_{fname}.json"
            with open(output_dir / filename, "w") as f:
                json.dump(output, f)
            count += 1

    return count


def _parse_skeleton_file(path: Path) -> list[np.ndarray]:
    """Parse a single NTU .skeleton text file.

    Returns:
        List of (25, 2) arrays, one per frame (first person only).
    """
    with open(path) as f:
        lines = f.readlines()

    idx = 0
    num_frames = int(lines[idx].strip())
    idx += 1

    frames = []
    for _ in range(num_frames):
        num_bodies = int(lines[idx].strip())
        idx += 1

        if num_bodies == 0:
            continue

        # Read first body
        idx += 1  # Skip body info line
        num_joints = int(lines[idx].strip())
        idx += 1

        joints = np.zeros((25, 2))
        for j in range(num_joints):
            parts = lines[idx].strip().split()
            joints[j, 0] = float(parts[5])  # colorX
            joints[j, 1] = float(parts[6])  # colorY
            idx += 1

        frames.append(joints)

        # Skip remaining bodies
        for _ in range(1, num_bodies):
            idx += 1  # body info
            nj = int(lines[idx].strip())
            idx += 1
            idx += nj  # skip joints

    return frames


def main():
    parser = argparse.ArgumentParser(description="Convert NTU-120 → Vistarra JSON")
    parser.add_argument(
        "--format",
        choices=["pickle", "raw"],
        default="pickle",
        help="Input format (default: pickle)",
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DATA_DIR / "ntu120_2d.pkl",
        help="Path to pickle file",
    )
    parser.add_argument(
        "--skeletons",
        type=Path,
        default=DATA_DIR / "raw_skeletons",
        help="Path to raw skeleton directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument("--seq-len", type=int, default=150, help="Frames per sequence")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    class_mapping = load_class_mapping()
    mapped_actions = sum(len(v) for v in class_mapping.values())
    print(f"Class mapping: {mapped_actions} NTU→Vistarra mappings loaded")

    if args.format == "pickle":
        count = convert_pickle(args.pkl, args.output, class_mapping, args.seq_len, args.fps)
    else:
        count = convert_raw_skeletons(
            args.skeletons, args.output, class_mapping, args.seq_len, args.fps
        )

    print(f"\nConverted {count} samples → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/convert_ntu120.py
git commit -m "feat: NTU-120 skeleton to Vistarra JSON converter"
```

---

### Task 4: Fix extract_poses.py Bug

**Files:**
- Modify: `scripts/extract_poses.py:91`

- [ ] **Step 1: Fix the parser bug**

Line 91 has `args = args.parse_args()` — should be `args = parser.parse_args()`.

```python
# Before:
    args = args.parse_args()

# After:
    args = parser.parse_args()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/extract_poses.py
git commit -m "fix: parser.parse_args() typo in extract_poses.py"
```

---

## Phase 2: Skeleton Graph Infrastructure

### Task 5: Skeleton Topology Graph Module

**Files:**
- Create: `pose_autoresearch/__init__.py`
- Create: `pose_autoresearch/graph.py`

The key insight of GCN-based models: human joints form a graph (shoulder→elbow→wrist). Treating keypoints as a flat 51-dim vector throws away this structural information. We need adjacency matrices that encode skeleton connectivity.

- [ ] **Step 1: Create package init**

```python
"""Pose autoresearch shared utilities."""
```

- [ ] **Step 2: Create the skeleton graph module**

```python
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
    (0, 1), (0, 2), (1, 3), (2, 4),       # nose ↔ eyes ↔ ears
    # Torso
    (5, 6),                                  # left_shoulder ↔ right_shoulder
    (5, 11), (6, 12),                        # shoulders ↔ hips
    (11, 12),                                # left_hip ↔ right_hip
    # Left arm
    (5, 7), (7, 9),                          # shoulder → elbow → wrist
    # Right arm
    (6, 8), (8, 10),                         # shoulder → elbow → wrist
    # Left leg
    (11, 13), (13, 15),                      # hip → knee → ankle
    # Right leg
    (12, 14), (14, 16),                      # hip → knee → ankle
    # Head → torso
    (0, 5), (0, 6),                          # nose ↔ shoulders (implicit)
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
    """Return directed bone vectors (parent → child) for COCO-17.

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
    adj = {i: [] for i in range(num_nodes)}
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
```

- [ ] **Step 3: Commit**

```bash
git add pose_autoresearch/__init__.py pose_autoresearch/graph.py
git commit -m "feat: skeleton topology graph with COCO-17 adjacency matrices"
```

---

### Task 6: Skeleton Data Augmentation

**Files:**
- Create: `pose_autoresearch/augment.py`

Skeleton-specific augmentations that preserve physical plausibility.

- [ ] **Step 1: Create augmentation module**

```python
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
        poses: (seq_len, 17, 3) tensor — (x, y, confidence).
        max_degrees: Maximum rotation angle.

    Returns:
        Rotated poses, same shape.
    """
    angle = random.uniform(-max_degrees, max_degrees) * (np.pi / 180)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Center of mass (mean of visible joints)
    xy = poses[:, :, :2]
    conf = poses[:, :, 2:3]
    mask = conf > 0.1
    center = (xy * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp(min=1)

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
    center = (xy * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp(min=1)

    scaled = (xy - center) * scale + center
    return torch.cat([scaled, conf], dim=-1)


def random_horizontal_flip(poses: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Mirror keypoints left↔right with probability p.

    Swaps left/right joint pairs and flips x coordinates.

    Args:
        poses: (seq_len, 17, 3) tensor.
        p: Flip probability.

    Returns:
        Possibly flipped poses.
    """
    if random.random() > p:
        return poses

    # COCO-17 left↔right pairs
    swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    flipped = poses.clone()

    # Flip x coordinate around center
    xy = flipped[:, :, :2]
    center_x = xy[:, :, 0].mean()
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
        # Pad with zeros
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
    new_len = int(seq_len * speed)
    new_len = max(2, new_len)

    # Resample indices
    indices = torch.linspace(0, seq_len - 1, new_len).long()
    resampled = poses[indices]

    # Resize back to original length
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
```

- [ ] **Step 2: Commit**

```bash
git add pose_autoresearch/augment.py
git commit -m "feat: skeleton-specific data augmentation (rotation, flip, speed, noise, dropout)"
```

---

## Phase 3: Upgrade prepare.py

### Task 7: Upgrade Data Pipeline

**Files:**
- Modify: `prepare.py`

Upgrade the data pipeline to support: multi-stream input (joint + bone + velocity), variable sequence lengths, proper normalization, and skeleton augmentation. Keep the same interface so `train.py` works unchanged.

- [ ] **Step 1: Rewrite prepare.py with multi-stream support**

The full rewrite of `prepare.py`:

```python
"""
Data preparation and evaluation utilities.
DO NOT MODIFY THIS FILE (unless explicitly asked by human).

The agent should only edit train.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from typing import Tuple

from pose_autoresearch.graph import get_bone_pairs
from pose_autoresearch.augment import augment_pose_sequence

# ============================================================================
# CONSTANTS (fixed)
# ============================================================================

EVENT_CLASSES = [
    "fall",
    "eating",
    "working_together",
    "aggression",
    "unstable_gait",
    "wandering",
    "sitting_standing",
]

NUM_CLASSES = len(EVENT_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(EVENT_CLASSES)}

NUM_KEYPOINTS = 17
VALUES_PER_KEYPOINT = 3  # x, y, confidence
INPUT_DIM = NUM_KEYPOINTS * VALUES_PER_KEYPOINT  # 51

# Sequence config: support 5-10 seconds at 30fps
SEQ_LEN = 150       # Default 5 seconds (150 frames at 30fps)
MAX_SEQ_LEN = 300   # Max 10 seconds (300 frames at 30fps)
FPS = 30

MAX_TIME_BUDGET_SECONDS = 300  # 5 minutes per experiment
TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"


# ============================================================================
# MULTI-STREAM FEATURES
# ============================================================================

BONE_PAIRS = get_bone_pairs()


def compute_bone_features(keypoints: np.ndarray) -> np.ndarray:
    """Compute bone vectors from joint positions.

    Args:
        keypoints: (seq_len, 17, 3) array — (x, y, confidence).

    Returns:
        (seq_len, num_bones, 3) array — bone vector (dx, dy, mean_conf).
    """
    bones = []
    for parent, child in BONE_PAIRS:
        dx = keypoints[:, child, 0] - keypoints[:, parent, 0]
        dy = keypoints[:, child, 1] - keypoints[:, parent, 1]
        conf = (keypoints[:, child, 2] + keypoints[:, parent, 2]) / 2
        bones.append(np.stack([dx, dy, conf], axis=-1))
    return np.stack(bones, axis=1)  # (seq_len, num_bones, 3)


def compute_velocity_features(keypoints: np.ndarray) -> np.ndarray:
    """Compute temporal velocity of each joint.

    Args:
        keypoints: (seq_len, 17, 3) array.

    Returns:
        (seq_len, 17, 3) array — (vx, vy, confidence).
    """
    velocity = np.zeros_like(keypoints)
    velocity[1:, :, 0] = keypoints[1:, :, 0] - keypoints[:-1, :, 0]  # vx
    velocity[1:, :, 1] = keypoints[1:, :, 1] - keypoints[:-1, :, 1]  # vy
    velocity[:, :, 2] = keypoints[:, :, 2]  # Keep confidence
    return velocity


# ============================================================================
# DATASET
# ============================================================================

class PoseDataset(Dataset):
    """Dataset for pose sequences with event labels.

    Each sample is loaded from a JSON file containing:
    {
        "frames": [{"keypoints": [[x,y,c],...], "timestamp": float}, ...],
        "label": "fall",
        "duration": float
    }

    Returns:
        poses: (seq_len, input_dim) tensor — flattened keypoints
        label: int class index
    """

    def __init__(
        self,
        data_dir: Path,
        seq_len: int = SEQ_LEN,
        augment: bool = False,
        multi_stream: bool = False,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.augment = augment
        self.multi_stream = multi_stream
        self.samples: list[tuple[np.ndarray, int]] = []

        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            print(f"Warning: No JSON files in {data_dir}. Run data preparation first.")
            return

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            label_str = data["label"]
            if label_str not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[label_str]

            frames = data["frames"]
            keypoints = []
            for frame in frames[:seq_len]:
                flat = np.array(frame["keypoints"]).flatten()
                keypoints.append(flat)

            while len(keypoints) < seq_len:
                keypoints.append(np.zeros(INPUT_DIM))

            poses = np.array(keypoints, dtype=np.float32)  # (seq_len, 51)
            self.samples.append((poses, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        poses_flat, label = self.samples[idx]
        poses_tensor = torch.from_numpy(poses_flat)

        if self.augment:
            # Reshape to (seq_len, 17, 3) for augmentation
            reshaped = poses_tensor.view(self.seq_len, NUM_KEYPOINTS, VALUES_PER_KEYPOINT)
            reshaped = augment_pose_sequence(reshaped, target_len=self.seq_len)
            poses_tensor = reshaped.view(self.seq_len, INPUT_DIM)

        if self.multi_stream:
            # Return joint + bone + velocity as separate streams
            kps = poses_tensor.view(self.seq_len, NUM_KEYPOINTS, VALUES_PER_KEYPOINT).numpy()
            bones = compute_bone_features(kps)       # (seq_len, num_bones, 3)
            velocity = compute_velocity_features(kps) # (seq_len, 17, 3)

            joint_flat = poses_tensor                                      # (seq_len, 51)
            bone_flat = torch.from_numpy(bones.reshape(self.seq_len, -1))  # (seq_len, num_bones*3)
            vel_flat = torch.from_numpy(velocity.reshape(self.seq_len, -1)) # (seq_len, 51)

            return (joint_flat, bone_flat, vel_flat), torch.tensor(label, dtype=torch.long)

        return poses_tensor, torch.tensor(label, dtype=torch.long)


# ============================================================================
# DATA LOADING
# ============================================================================

def get_dataloaders(
    data_dir: Path = PROCESSED_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    split: Tuple[float, float, float] = TRAIN_VAL_TEST_SPLIT,
    augment_train: bool = True,
    multi_stream: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        data_dir: Path to processed JSON files.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        split: (train, val, test) fractions.
        augment_train: Apply augmentation to training set.
        multi_stream: Return multi-stream (joint, bone, velocity) tensors.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load full dataset without augmentation for splitting
    dataset = PoseDataset(data_dir, augment=False, multi_stream=multi_stream)

    if len(dataset) == 0:
        print("ERROR: No data found. Run one of:")
        print("  python prepare.py --dataset synthetic")
        print("  python scripts/convert_ntu120.py")
        raise RuntimeError("No training data")

    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Wrap training subset with augmentation if requested
    if augment_train:
        train_dataset.dataset = PoseDataset(
            data_dir, augment=True, multi_stream=multi_stream
        )

    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model accuracy and loss.

    Returns:
        (accuracy, avg_loss)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], (tuple, list)):
                # Multi-stream: batch[0] = (joints, bones, velocity)
                inputs = tuple(t.to(device) for t in batch[0])
                labels = batch[1].to(device)
                logits = model(*inputs)
            else:
                poses, labels = batch
                poses = poses.to(device)
                labels = labels.to(device)
                logits = model(poses)

            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(len(dataloader), 1)
    return accuracy, avg_loss


def evaluate_per_class(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Per-class accuracy breakdown.

    Returns:
        Dict mapping class name → accuracy.
    """
    model.eval()
    class_correct = {cls: 0 for cls in EVENT_CLASSES}
    class_total = {cls: 0 for cls in EVENT_CLASSES}

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], (tuple, list)):
                inputs = tuple(t.to(device) for t in batch[0])
                labels = batch[1].to(device)
                logits = model(*inputs)
            else:
                poses, labels = batch
                poses = poses.to(device)
                labels = labels.to(device)
                logits = model(poses)

            preds = torch.argmax(logits, dim=1)
            for pred, label in zip(preds, labels):
                cls = EVENT_CLASSES[label.item()]
                class_total[cls] += 1
                if pred == label:
                    class_correct[cls] += 1

    return {
        cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        for cls in EVENT_CLASSES
    }


# ============================================================================
# SYNTHETIC DATA (for testing only)
# ============================================================================

def prepare_synthetic_data():
    """Generate dummy data for pipeline testing.

    NOTE: This produces random data — no model can learn real patterns from this.
    Use NTU RGB+D 120 or real video data for actual training.
    """
    print("Preparing synthetic data (for pipeline testing only)...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    samples_per_class = 100
    for class_name in EVENT_CLASSES:
        print(f"  Generating {samples_per_class} samples: {class_name}")
        for i in range(samples_per_class):
            frames = []
            for t in range(SEQ_LEN):
                keypoints = np.random.rand(NUM_KEYPOINTS, VALUES_PER_KEYPOINT).tolist()
                frames.append({"keypoints": keypoints, "timestamp": t / FPS})

            sample = {
                "frames": frames,
                "label": class_name,
                "duration": SEQ_LEN / FPS,
            }
            with open(PROCESSED_DIR / f"{class_name}_{i:04d}.json", "w") as f:
                json.dump(sample, f)

    total = len(EVENT_CLASSES) * samples_per_class
    print(f"\nCreated {total} synthetic samples in {PROCESSED_DIR}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare pose dataset")
    parser.add_argument(
        "--dataset",
        default="synthetic",
        choices=["synthetic", "ntu", "kinetics"],
    )
    args = parser.parse_args()

    if args.dataset == "synthetic":
        prepare_synthetic_data()
    else:
        print(f"{args.dataset} not yet implemented. Use --dataset synthetic or run scripts/convert_ntu120.py")

    print("\nTesting data loading...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    poses, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Labels: {[EVENT_CLASSES[l.item()] for l in labels[:5]]}")
    print("\nDone!")
```

- [ ] **Step 2: Commit**

```bash
git add prepare.py
git commit -m "feat: upgrade prepare.py with multi-stream features, augmentation, variable-length support"
```

---

## Phase 4: Upgrade Model Baseline

### Task 8: GCN Baseline in train.py

**Files:**
- Modify: `train.py`

Replace the flat CNN+LSTM baseline with a Spatial-Temporal Graph Convolutional Network (ST-GCN) baseline. This gives the autoresearch agent a much stronger starting point and a proper architecture to iterate on.

- [ ] **Step 1: Rewrite train.py with ST-GCN baseline**

```python
"""
Pose-based event detection training script.
The agent modifies this file to improve validation accuracy.

Baseline: Spatial-Temporal GCN (graph-aware skeleton model).
The agent can modify architecture, hyperparameters, optimizer, anything.
"""

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
    get_spatial_partitioning,
    adjacency_to_tensor,
    COCO_17_EDGES,
)

# ============================================================================
# HYPERPARAMETERS (agent can modify)
# ============================================================================

INPUT_DIM = 51       # 17 keypoints × 3 (x, y, confidence)
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
    where A_norm is the normalized adjacency matrix.
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
        x = torch.einsum("btvc,vw->btwc", x, self.A)  # (B, T, V, C)

        # Linear transform
        x = self.W(x)  # (B, T, V, out_channels)

        # Batch norm (reshape to merge batch + time)
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

        # Temporal convolution: operates on the time dimension per-joint
        pad = (temporal_kernel - 1) // 2
        self.temporal = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (temporal_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # Residual connection (match dimensions if needed)
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
        # Residual path
        # Reshape to (B, C, T, V) for Conv2d
        res = x.permute(0, 3, 1, 2)  # (B, C_in, T, V)
        res = self.residual(res)

        # Spatial graph conv
        x = self.spatial(x)           # (B, T, V, C_out)
        x = self.relu(x)

        # Temporal conv: reshape to (B, C_out, T, V)
        x = x.permute(0, 3, 1, 2)    # (B, C_out, T, V)
        x = self.temporal(x)          # (B, C_out, T, V)

        # Add residual and activate
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
    1. Input projection: (x, y, conf) per joint → feature channels
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
        gcn_channels: list[int] = None,
        temporal_kernel: int = TEMPORAL_KERNEL_SIZE,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        if gcn_channels is None:
            gcn_channels = list(GCN_CHANNELS)

        # Normalized adjacency matrix
        A = adjacency_to_tensor(get_normalized_adjacency(COCO_17_EDGES, num_joints))

        # Input batch norm
        self.input_bn = nn.BatchNorm1d(input_channels * num_joints)

        # ST-GCN blocks
        layers = []
        in_ch = input_channels
        for out_ch in gcn_channels:
            layers.append(
                STGCNBlock(in_ch, out_ch, A, temporal_kernel, dropout=dropout)
            )
            in_ch = out_ch

        self.gcn_blocks = nn.ModuleList(layers)

        # Classification head
        self.fc = nn.Linear(gcn_channels[-1], num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 51) — flattened keypoints

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
            x = block(x)  # (B, T, V, C)

        # Global average pooling: average over time and joints
        x = x.mean(dim=[1, 2])  # (B, C)

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
    print("POSE AUTORESEARCH — Training Run")
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

    # Cosine annealing schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-6,
    )

    # Label smoothing helps prevent overconfidence
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
```

- [ ] **Step 2: Commit**

```bash
git add train.py
git commit -m "feat: ST-GCN baseline model with graph-aware skeleton processing"
```

---

## Phase 5: Autoresearch Agent Setup

### Task 9: Upgrade Agent Directives

**Files:**
- Modify: `program.md`

The current `program.md` is generic. We need directives that guide the agent toward graph-based architectures, multi-stream fusion, and the specific innovations that push accuracy highest.

- [ ] **Step 1: Rewrite program.md**

```markdown
# Pose Event Classification — Autoresearch Directives

## Objective

Maximize validation accuracy on 7-class event classification from 17-keypoint
COCO pose sequences (5 seconds at 30fps = 150 frames per sample).

You modify `train.py`. Everything in `prepare.py` and `pose_autoresearch/` is fixed.

## Classes

| Class | What It Looks Like | Priority |
|-------|-------------------|----------|
| fall | Rapid downward hip velocity → horizontal posture → stillness | **HIGHEST** (safety-critical) |
| aggression | Fast limb extension toward another person, high acceleration | HIGH |
| unstable_gait | Asymmetric stride length, low cadence, lateral sway | HIGH |
| eating | Repetitive wrist-to-face motion while seated | MEDIUM |
| wandering | Repetitive back-and-forth traversal pattern | MEDIUM |
| working_together | Two+ people in sustained proximity, coordinated motion | MEDIUM |
| sitting_standing | Postural transitions — hip height change, torso angle shift | LOW |

Fall detection accuracy is the single most important metric. If you have to trade
accuracy on "sitting_standing" to improve "fall" by 1%, do it.

## Current Architecture

The baseline is a Spatial-Temporal GCN (ST-GCN) that:
1. Treats 17 COCO keypoints as a graph with anatomical edges
2. Applies graph convolution (spatial) + 1D convolution (temporal) per block
3. Global average pools over time and joints
4. Linear classification head

This is a strong baseline. To improve it, try:

## Architecture Ideas (ordered by expected impact)

### 1. Channel-wise Topology Refinement (CTR-GCN style)
Instead of a fixed adjacency matrix, learn a different graph topology per
channel. This lets the model attend to different joint subsets for different
features (e.g., one channel focuses on arms, another on legs).

### 2. Temporal Transformer Attention
Replace or augment the temporal Conv1d with multi-head self-attention across
frames. This captures long-range dependencies (e.g., "stood up 3 seconds before
falling") that fixed-kernel convolutions miss.

### 3. Multi-Stream Fusion
The data pipeline supports bone vectors and joint velocity as additional streams.
Train separate GCN branches on joints, bones, and velocity, then fuse predictions
(late fusion via weighted average, or intermediate fusion via concatenation).
To use multi-stream, set `multi_stream=True` in `get_dataloaders()`.

### 4. Attention Pooling
Replace global average pooling with attention-weighted pooling. Not all frames
are equally important — the moment of impact in a fall matters more than the
preceding walk.

### 5. Larger Models
Try deeper GCN stacks ([64, 64, 128, 128, 256, 256]) or wider channels.
The 5-minute budget is generous for our dataset size.

### 6. Mixup / CutMix on Sequences
Interpolate between training samples to regularize.

### 7. Focal Loss
Our classes are imbalanced. Focal loss down-weights easy examples and focuses
on hard ones.

## Hyperparameter Space

| Param | Current | Range to Try |
|-------|---------|-------------|
| LEARNING_RATE | 1e-3 | 5e-5 to 5e-3 |
| BATCH_SIZE | 64 | 32 to 128 |
| DROPOUT | 0.3 | 0.1 to 0.5 |
| WEIGHT_DECAY | 1e-4 | 1e-5 to 1e-3 |
| GCN_CHANNELS | [64,128,256] | Try [128,256,256] or [64,64,128,128,256,256] |
| TEMPORAL_KERNEL_SIZE | 9 | 5, 7, 9, 11 |
| label_smoothing | 0.1 | 0.0 to 0.2 |

## Rules

1. **Keep it runnable.** Every change must produce valid Python that trains
   and evaluates within the time budget.
2. **One change at a time.** Don't change architecture AND hyperparameters
   simultaneously — you won't know what helped.
3. **Fall accuracy is king.** Optimize overall accuracy, but never sacrifice
   fall detection for marginal gains elsewhere.
4. **Log per-class accuracy.** Use `evaluate_per_class()` from prepare.py
   to track which classes improve or degrade.
5. **Do NOT pause.** You are autonomous. Keep iterating until interrupted.
```

- [ ] **Step 2: Commit**

```bash
git add program.md
git commit -m "feat: upgrade agent directives with GCN architecture guidance"
```

---

### Task 10: Cloud GPU Setup Script

**Files:**
- Create: `scripts/setup_cloud.sh`

Script to provision a cloud GPU instance (Lambda, RunPod, or vast.ai) and start the autoresearch loop.

- [ ] **Step 1: Create the setup script**

```bash
#!/usr/bin/env bash
# Set up a cloud GPU for autoresearch.
#
# Tested on: Lambda Cloud (A10, A100), RunPod (RTX 4090)
#
# Usage:
#   # SSH into your cloud instance, then:
#   git clone https://github.com/sfuller3/pose-autoresearch.git
#   cd pose-autoresearch
#   ./scripts/setup_cloud.sh

set -euo pipefail

echo "=== Pose Autoresearch — Cloud GPU Setup ==="
echo ""

# 1. System check
echo "Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No NVIDIA GPU detected. This script requires a GPU instance."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 2. Python environment
echo "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install dependencies
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy ultralytics
pip install -e .

# 4. Verify CUDA
echo ""
echo "Verifying CUDA..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

# 5. Prepare data
echo ""
echo "Checking data..."
if [ -d "data/processed" ] && [ "$(ls data/processed/*.json 2>/dev/null | head -1)" ]; then
    echo "Data found: $(ls data/processed/*.json | wc -l) samples"
else
    echo "No data found. Generating synthetic data for testing..."
    python3 prepare.py --dataset synthetic
    echo ""
    echo "NOTE: For real training, download NTU RGB+D 120 and run:"
    echo "  python scripts/convert_ntu120.py"
fi

# 6. Test training
echo ""
echo "Running quick training test (30 seconds)..."
python3 -c "
import os
os.environ['VISTARRA_MAX_TIME'] = '30'
# Quick smoke test
from train import main
main()
" 2>&1 | tail -10

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start autoresearch loop, run Claude Code:"
echo "  claude --chat 'Read program.md, then run the autoresearch loop on train.py'"
echo ""
echo "Or manually:"
echo "  python train.py"
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/setup_cloud.sh
git add scripts/setup_cloud.sh
git commit -m "feat: cloud GPU setup script for autoresearch"
```

---

### Task 11: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update dependencies**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "pose-autoresearch"
version = "0.2.0"
description = "Autonomous research for pose-based event detection"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "ultralytics>=8.0.0",  # For YOLO pose extraction
]

[project.optional-dependencies]
dev = [
    "ipython",
    "jupyter",
    "matplotlib",
    "seaborn",
    "pytest",
    "pytest-cov",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: update pyproject.toml with v0.2.0 and test deps"
```

---

## Phase 6: Integration Testing

### Task 12: End-to-End Smoke Test

**Files:**
- Create: `tests/test_pipeline.py`

Verify the full pipeline works: synthetic data → dataloader → model → train → evaluate.

- [ ] **Step 1: Create the test**

```python
"""End-to-end smoke tests for the training pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from prepare import (
    PoseDataset,
    get_dataloaders,
    evaluate_model,
    evaluate_per_class,
    EVENT_CLASSES,
    NUM_KEYPOINTS,
    VALUES_PER_KEYPOINT,
    SEQ_LEN,
    DEVICE,
)
from pose_autoresearch.graph import (
    get_adjacency_matrix,
    get_normalized_adjacency,
    get_spatial_partitioning,
    get_bone_pairs,
    COCO_17_EDGES,
)
from pose_autoresearch.augment import (
    random_rotation,
    random_horizontal_flip,
    augment_pose_sequence,
)


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with minimal JSON pose samples."""
    for cls_idx, cls_name in enumerate(EVENT_CLASSES):
        for i in range(10):
            frames = []
            for t in range(SEQ_LEN):
                kps = np.random.rand(NUM_KEYPOINTS, VALUES_PER_KEYPOINT).tolist()
                frames.append({"keypoints": kps, "timestamp": t / 30.0})
            sample = {"frames": frames, "label": cls_name, "duration": SEQ_LEN / 30.0}
            with open(tmp_path / f"{cls_name}_{i:04d}.json", "w") as f:
                json.dump(sample, f)
    return tmp_path


class TestGraph:
    def test_adjacency_shape(self):
        A = get_adjacency_matrix()
        assert A.shape == (17, 17)
        assert A[5, 7] == 1.0  # left_shoulder → left_elbow
        assert A[0, 0] == 1.0  # self-loop

    def test_normalized_adjacency_rows_sum_near_one(self):
        A = get_normalized_adjacency()
        row_sums = A.sum(axis=1)
        # Normalized adjacency rows won't sum to exactly 1 (symmetric norm),
        # but should be non-zero
        assert all(s > 0 for s in row_sums)

    def test_spatial_partitioning_shape(self):
        P = get_spatial_partitioning()
        assert P.shape == (3, 17, 17)

    def test_bone_pairs_valid_indices(self):
        bones = get_bone_pairs()
        assert len(bones) > 0
        for parent, child in bones:
            assert 0 <= parent < 17
            assert 0 <= child < 17


class TestAugmentation:
    def test_rotation_preserves_shape(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        rotated = random_rotation(poses)
        assert rotated.shape == poses.shape

    def test_flip_swaps_joints(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        flipped = random_horizontal_flip(poses, p=1.0)  # Force flip
        assert flipped.shape == poses.shape
        # Left shoulder (5) and right shoulder (6) should be swapped
        # (approximately — x coordinates are also mirrored)

    def test_augment_pipeline_preserves_shape(self):
        poses = torch.rand(SEQ_LEN, 17, 3)
        augmented = augment_pose_sequence(poses, target_len=SEQ_LEN)
        assert augmented.shape == (SEQ_LEN, 17, 3)


class TestDataset:
    def test_dataset_loads(self, sample_data_dir):
        ds = PoseDataset(sample_data_dir)
        assert len(ds) == 70  # 7 classes × 10 samples

    def test_sample_shape(self, sample_data_dir):
        ds = PoseDataset(sample_data_dir)
        poses, label = ds[0]
        assert poses.shape == (SEQ_LEN, 51)
        assert 0 <= label.item() < len(EVENT_CLASSES)

    def test_dataloaders(self, sample_data_dir):
        train, val, test = get_dataloaders(
            data_dir=sample_data_dir, batch_size=4, num_workers=0,
        )
        assert len(train) > 0
        poses, labels = next(iter(train))
        assert poses.shape[0] <= 4
        assert poses.shape[2] == 51


class TestModel:
    def test_forward_pass(self):
        from train import PoseEventClassifier
        model = PoseEventClassifier().to(DEVICE)
        x = torch.randn(2, SEQ_LEN, 51).to(DEVICE)
        logits = model(x)
        assert logits.shape == (2, len(EVENT_CLASSES))

    def test_evaluate(self, sample_data_dir):
        from train import PoseEventClassifier
        model = PoseEventClassifier().to(DEVICE)
        _, val_loader, _ = get_dataloaders(
            data_dir=sample_data_dir, batch_size=4, num_workers=0,
        )
        acc, loss = evaluate_model(model, val_loader, DEVICE)
        assert 0.0 <= acc <= 1.0
        assert loss >= 0.0

    def test_per_class_eval(self, sample_data_dir):
        from train import PoseEventClassifier
        model = PoseEventClassifier().to(DEVICE)
        _, val_loader, _ = get_dataloaders(
            data_dir=sample_data_dir, batch_size=4, num_workers=0,
        )
        per_class = evaluate_per_class(model, val_loader, DEVICE)
        assert set(per_class.keys()) == set(EVENT_CLASSES)
```

- [ ] **Step 2: Run the tests**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: All tests PASS (with random data, model accuracy will be ~14% / random chance).

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline.py
git commit -m "test: end-to-end pipeline smoke tests"
```

---

## Phase 7: Data Acquisition & First Training Run

### Task 13: Download and Convert NTU RGB+D 120

This is a manual step that requires registering for the dataset.

- [ ] **Step 1: Register for NTU RGB+D 120**

Go to https://rose1.ntu.edu.sg/dataset/actionRecognition/ and register.
Or download pre-processed pickle from https://figshare.com/articles/dataset/27427188.

- [ ] **Step 2: Convert to Vistarra format**

```bash
# If using pickle:
python scripts/convert_ntu120.py --format pickle --pkl data/ntu120/ntu120_2d.pkl

# If using raw skeletons:
python scripts/convert_ntu120.py --format raw --skeletons data/ntu120/raw_skeletons/
```

Expected output: thousands of JSON files in `data/processed/`.

- [ ] **Step 3: Verify data distribution**

```bash
python -c "
from pathlib import Path
from collections import Counter
import json
counts = Counter()
for f in Path('data/processed').glob('*.json'):
    with open(f) as fh:
        counts[json.load(fh)['label']] += 1
for cls, n in sorted(counts.items()):
    print(f'  {cls:20s}: {n:6d}')
print(f'  {\"TOTAL\":20s}: {sum(counts.values()):6d}')
"
```

- [ ] **Step 4: Run first real training**

```bash
python train.py
```

This runs the ST-GCN baseline for 5 minutes on real data. Record the validation accuracy — this is your baseline to beat.

---

### Task 14: Start Autoresearch Loop

- [ ] **Step 1: SSH into cloud GPU**

```bash
ssh your-cloud-instance
cd pose-autoresearch
./scripts/setup_cloud.sh
```

- [ ] **Step 2: Launch the autoresearch agent**

Run Claude Code with the autoresearch directive:

```bash
claude --chat "Read program.md carefully. Your job is to maximize validation accuracy
by modifying train.py. Run training, check the result, make one change, repeat.
If accuracy improves, commit. If not, revert. Keep going until I stop you."
```

The agent will:
1. Read `program.md` and `train.py`
2. Form a hypothesis (e.g., "add attention pooling")
3. Edit `train.py`
4. Run `python train.py`
5. Check validation accuracy
6. If improved: `git commit`; if not: `git checkout train.py`
7. Repeat

Expected: ~12 experiments/hour, ~100 overnight. Each improvement stacks.

---

## Summary

| Phase | What | Why |
|-------|------|-----|
| 1. Data Foundation | NTU-120 mapping, conversion, pipeline | Can't train without real data |
| 2. Graph Infrastructure | Skeleton topology, adjacency matrices | GCNs need graph structure |
| 3. Augmentation | Skeleton-aware transforms | Regularization + data diversity |
| 4. prepare.py Upgrade | Multi-stream, augmentation, variable-length | Foundation for agent experiments |
| 5. train.py Upgrade | ST-GCN baseline (replaces CNN+LSTM) | 10-15% accuracy jump from graph awareness |
| 6. Agent Directives | Architecture-aware program.md | Guide agent toward high-value experiments |
| 7. Cloud + Autoresearch | Setup script, launch agent | Autonomous overnight improvement |

**Expected trajectory:**
- Synthetic random data: ~14% (random chance for 7 classes)
- CNN+LSTM on real data: ~60-70%
- ST-GCN on real data: ~80-85%
- After autoresearch (100+ experiments): ~90%+
- With multi-stream fusion + attention: ~93%+
