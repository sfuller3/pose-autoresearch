#!/usr/bin/env python3
"""Convert NTU RGB+D 120 pre-processed 2D skeleton pickle to our JSON format.

The OpenMMLab ntu120_2d.pkl contains HRNet-extracted 17-keypoint skeletons
(same count as COCO-17). We filter to action classes that map to our 7 event
types and output JSON files compatible with prepare.py.

Usage:
    python scripts/convert_ntu120.py
    python scripts/convert_ntu120.py --seq-len 150 --max-per-class 2000
    python scripts/convert_ntu120.py --dry-run  # show counts without writing
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

DATA_DIR = Path("data/ntu120")
OUTPUT_DIR = Path("data/processed")
MAPPING_FILE = DATA_DIR / "class_mapping.json"
PKL_FILE = DATA_DIR / "ntu120_2d.pkl"


def load_class_mapping() -> dict[int, list[str]]:
    """Load NTU action ID -> list of Vistarra event class names."""
    with open(MAPPING_FILE) as f:
        raw = json.load(f)

    ntu_to_vistarra: dict[int, list[str]] = {}
    for vistarra_class, ntu_actions in raw.items():
        if vistarra_class.startswith("_"):
            continue
        for entry in ntu_actions:
            action_id = int(entry.split(":")[0][1:]) - 1  # A001 -> 0
            ntu_to_vistarra.setdefault(action_id, []).append(vistarra_class)

    return ntu_to_vistarra


def convert(
    seq_len: int = 150,
    fps: int = 30,
    max_per_class: int | None = None,
    dry_run: bool = False,
) -> None:
    class_mapping = load_class_mapping()
    mapped_actions = set(class_mapping.keys())
    print(f"Class mapping: {len(mapped_actions)} NTU actions -> our event types")
    for action_id, classes in sorted(class_mapping.items()):
        print(f"  A{action_id + 1:03d} -> {classes}")

    print(f"\nLoading {PKL_FILE}...")
    with open(PKL_FILE, "rb") as f:
        data = pickle.load(f)

    annotations = data["annotations"]
    print(f"Total samples in pickle: {len(annotations)}")

    # Count what we'll produce
    class_counts: Counter[str] = Counter()
    class_written: Counter[str] = Counter()
    skipped_short = 0
    converted = 0

    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        label = ann["label"]
        if label not in class_mapping:
            continue

        vistarra_classes = class_mapping[label]
        keypoints = ann["keypoint"]        # (M, T, 17, 2) float16
        scores = ann["keypoint_score"]     # (M, T, 17) float16
        total_frames = ann["total_frames"]
        frame_dir = ann["frame_dir"]

        # Use first person only
        kp = keypoints[0]    # (T, 17, 2)
        sc = scores[0]       # (T, 17)

        if total_frames < 10:
            skipped_short += 1
            continue

        for vistarra_class in vistarra_classes:
            class_counts[vistarra_class] += 1

            if max_per_class and class_written[vistarra_class] >= max_per_class:
                continue

            if dry_run:
                class_written[vistarra_class] += 1
                continue

            # Build frames list
            frames = []
            n_frames = min(total_frames, seq_len)
            for t in range(n_frames):
                kp_frame = []
                for j in range(17):
                    x = float(kp[t, j, 0])
                    y = float(kp[t, j, 1])
                    c = float(sc[t, j])
                    kp_frame.append([x, y, c])
                frames.append({
                    "keypoints": kp_frame,
                    "timestamp": t / fps,
                })

            # Pad short sequences to seq_len
            while len(frames) < seq_len:
                frames.append({
                    "keypoints": [[0.0, 0.0, 0.0]] * 17,
                    "timestamp": len(frames) / fps,
                })

            output = {
                "frames": frames[:seq_len],
                "label": vistarra_class,
                "duration": n_frames / fps,
                "source": f"ntu120:{frame_dir}",
            }

            filename = f"ntu_{vistarra_class}_{frame_dir}.json"
            with open(OUTPUT_DIR / filename, "w") as f:
                json.dump(output, f)

            class_written[vistarra_class] += 1
            converted += 1

            if converted % 1000 == 0:
                print(f"  Converted {converted} samples...")

    print(f"\nSkipped (too short): {skipped_short}")
    print(f"\nAvailable samples per class:")
    for cls, count in sorted(class_counts.items()):
        written = class_written[cls]
        cap = f" (capped at {max_per_class})" if max_per_class and written < count else ""
        print(f"  {cls:20s}: {count:6d} available, {written:6d} written{cap}")
    print(f"\nTotal written: {sum(class_written.values())}")


def main():
    parser = argparse.ArgumentParser(description="Convert NTU-120 2D skeletons to JSON")
    parser.add_argument("--seq-len", type=int, default=150, help="Frames per sequence (default: 150)")
    parser.add_argument("--fps", type=int, default=30, help="Assumed FPS (default: 30)")
    parser.add_argument("--max-per-class", type=int, default=None, help="Cap samples per class")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing files")
    args = parser.parse_args()
    convert(args.seq_len, args.fps, args.max_per_class, args.dry_run)


if __name__ == "__main__":
    main()
