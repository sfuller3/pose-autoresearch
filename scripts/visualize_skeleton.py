"""Render COCO-17 skeleton sequences to animated GIF or MP4.

Usage:
    python scripts/visualize_skeleton.py data/processed/sample.json -o output.gif
    python scripts/visualize_skeleton.py data/processed/ -o output_dir/ --batch
"""

from __future__ import annotations

import json
from pathlib import Path


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
