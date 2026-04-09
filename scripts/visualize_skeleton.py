"""Render COCO-17 skeleton sequences to animated GIF or MP4.

Usage:
    python scripts/visualize_skeleton.py data/processed/sample.json -o output.gif
    python scripts/visualize_skeleton.py data/processed/ -o output_dir/ --batch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path so we can import pose_autoresearch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_autoresearch.graph import get_bone_pairs


def load_sample(path: str | Path) -> tuple[list[list[list[float]]], str]:
    """Load a JSON pose sample.

    Args:
        path: Path to a JSON file with keys: frames, label, duration.

    Returns:
        (frames, label) where frames is a list of 17-keypoint frames,
        each keypoint being [x, y, confidence].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sample not found: {path}")
    with open(path) as f:
        data = json.load(f)
    frames = [frame["keypoints"] for frame in data["frames"]]
    label = data["label"]
    return frames, label


# Body region colors (by joint index)
JOINT_COLORS = {}
_HEAD = [0, 1, 2, 3, 4]
_TORSO = [5, 6, 11, 12]
_ARMS = [7, 8, 9, 10]
_LEGS = [13, 14, 15, 16]

_COLOR_MAP = {
    "#4FC3F7": _HEAD,    # light blue
    "#81C784": _TORSO,   # green
    "#FFB74D": _ARMS,    # orange
    "#E57373": _LEGS,    # red
}
for color, indices in _COLOR_MAP.items():
    for idx in indices:
        JOINT_COLORS[idx] = color

# Map each joint to its body-region color for bone drawing
_CHILD_COLOR = {idx: color for color, indices in _COLOR_MAP.items() for idx in indices}

CONF_THRESHOLD = 0.1
BASE_JOINT_SIZE = 80


def draw_skeleton(
    ax: plt.Axes,
    keypoints: list[list[float]],
) -> dict:
    """Draw a single skeleton frame on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        keypoints: 17 keypoints, each [x, y, confidence].

    Returns:
        Dict with 'joints' (PathCollection) and 'bones' (list of Line2D).
    """
    kps = np.array(keypoints)  # (17, 3)
    xs, ys, confs = kps[:, 0], kps[:, 1], kps[:, 2]

    # Joint sizes: scale by confidence, hide low-confidence
    sizes = np.where(confs >= CONF_THRESHOLD, BASE_JOINT_SIZE * confs, 0)
    colors = [JOINT_COLORS.get(i, "#FFFFFF") for i in range(len(kps))]

    joints = ax.scatter(xs, ys, s=sizes, c=colors, zorder=5, edgecolors="white", linewidths=0.5)

    # Bones
    bone_pairs = get_bone_pairs()
    bones = []
    for parent, child in bone_pairs:
        if confs[parent] < CONF_THRESHOLD or confs[child] < CONF_THRESHOLD:
            continue
        color = _CHILD_COLOR.get(child, "#FFFFFF")
        line, = ax.plot(
            [xs[parent], xs[child]],
            [ys[parent], ys[child]],
            color=color, linewidth=2, zorder=3, alpha=0.8,
        )
        bones.append(line)

    return {"joints": joints, "bones": bones}
