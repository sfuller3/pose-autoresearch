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
from matplotlib.animation import FuncAnimation
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


BG_COLOR = "#1a1a2e"


def render_animation(
    frames: list[list[list[float]]],
    label: str,
    output_path: str,
    fps: int = 30,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 100,
    show_overlay: bool = True,
) -> None:
    """Render a skeleton sequence to GIF or MP4.

    Args:
        frames: List of frames, each a list of 17 [x, y, conf] keypoints.
        label: Action class label for overlay text.
        output_path: Output file path (.gif or .mp4).
        fps: Playback framerate.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch.
        show_overlay: Whether to draw text overlay.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    # Compute global bounds across all frames for stable view
    all_kps = np.array(frames)  # (num_frames, 17, 3)
    all_x = all_kps[:, :, 0]
    all_y = all_kps[:, :, 1]
    pad = 0.1
    x_min, x_max = all_x.min() - pad, all_x.max() + pad
    y_min, y_max = all_y.min() - pad, all_y.max() + pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y so head is at top

    num_frames = len(frames)

    def init():
        return []

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

        draw_skeleton(ax, frames[frame_idx])

        if show_overlay:
            kps = np.array(frames[frame_idx])
            mean_conf = kps[:, 2].mean()

            bbox_props = dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6)
            ax.text(
                0.02, 0.98, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="white",
                verticalalignment="top", bbox=bbox_props,
            )
            ax.text(
                0.98, 0.98, f"{frame_idx + 1} / {num_frames}",
                transform=ax.transAxes,
                fontsize=11, color="white",
                verticalalignment="top", horizontalalignment="right",
                bbox=bbox_props,
            )
            ax.text(
                0.02, 0.02, f"conf: {mean_conf:.2f}",
                transform=ax.transAxes,
                fontsize=11, color="white",
                verticalalignment="bottom", bbox=bbox_props,
            )

        return []

    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)

    output_path = Path(output_path)
    if output_path.suffix == ".gif":
        anim.save(str(output_path), writer="pillow", fps=fps)
    elif output_path.suffix == ".mp4":
        anim.save(str(output_path), writer="ffmpeg", fps=fps)
    else:
        raise ValueError(f"Unsupported format: {output_path.suffix}. Use .gif or .mp4")

    plt.close(fig)
    print(f"Saved: {output_path}")
