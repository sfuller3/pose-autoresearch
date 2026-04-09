#!/usr/bin/env python3
"""Source-aware train/val/test splitting for multi-dataset pose data.

Splits data from NTU120, FallVision, and Le2i into train/val/test
directories with proper isolation to prevent data leakage:

  - NTU120: Cross-subject protocol (official 53 training subjects)
  - FallVision: Video-level grouping (all segments from same video together)
  - Le2i: Video-level grouping (same approach as FallVision)

Synthetic data (files without ntu_/fv_/le2i_ prefix) is skipped.

Usage:
    python scripts/split_data.py
    python scripts/split_data.py --source-dir data/processed --output-dir data/splits
    python scripts/split_data.py --copy  # copy files instead of symlinks
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


# NTU RGB+D 120 official cross-subject training subjects
# Source: https://rose1.ntu.edu.sg/dataset/actionRecognition/
NTU_TRAIN_SUBJECTS = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81,
    82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103,
}


def parse_ntu_subject(filename: str) -> int | None:
    """Extract performer ID from NTU filename like ntu_fall_S001C001P007R001A043.json."""
    match = re.search(r"P(\d{3})", filename)
    return int(match.group(1)) if match else None


def parse_fv_video_key(filename: str) -> str:
    """Extract video key from FallVision filename.

    fv_fall_bed_B_D_0014_keypoints_0002.json -> fv_fall_bed_B_D_0014_keypoints
    The last _NNNN.json is the segment index; everything before is the video.
    """
    stem = Path(filename).stem  # remove .json
    # Remove trailing _NNNN segment index
    match = re.match(r"(.+)_(\d{4})$", stem)
    return match.group(1) if match else stem


def parse_le2i_video_key(filename: str) -> str:
    """Extract video key from Le2i filename.

    le2i_fall_coffee_room_video01_0003.json -> le2i_fall_coffee_room_video01
    """
    stem = Path(filename).stem
    match = re.match(r"(.+)_(\d{4})$", stem)
    return match.group(1) if match else stem


def split_groups(groups: list[str], seed: int = 42, ratios: tuple = (0.7, 0.15, 0.15)):
    """Split group keys into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train = set(shuffled[:n_train])
    val = set(shuffled[n_train:n_train + n_val])
    test = set(shuffled[n_train + n_val:])
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Split pose data into train/val/test")
    parser.add_argument("--source-dir", type=Path, default=Path("data/processed"),
                        help="Directory containing all JSON files (default: data/processed)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"),
                        help="Output directory for splits (default: data/splits)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of creating symlinks")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir

    # Collect files by source
    ntu_files: list[tuple[str, Path]] = []       # (filename, path)
    fv_files: dict[str, list[Path]] = defaultdict(list)   # video_key -> [paths]
    le2i_files: dict[str, list[Path]] = defaultdict(list)  # video_key -> [paths]
    skipped = 0

    for f in sorted(source_dir.glob("*.json")):
        name = f.name
        if name.startswith("ntu_"):
            ntu_files.append((name, f))
        elif name.startswith("fv_"):
            key = parse_fv_video_key(name)
            fv_files[key].append(f)
        elif name.startswith("le2i_"):
            key = parse_le2i_video_key(name)
            le2i_files[key].append(f)
        else:
            skipped += 1

    print(f"Found: {len(ntu_files)} NTU, {sum(len(v) for v in fv_files.values())} FallVision "
          f"({len(fv_files)} videos), {sum(len(v) for v in le2i_files.values())} Le2i "
          f"({len(le2i_files)} videos)")
    if skipped:
        print(f"Skipped: {skipped} files (no ntu_/fv_/le2i_ prefix)")

    # Create output directories
    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    assignments: dict[str, list[Path]] = {"train": [], "val": [], "test": []}

    # --- NTU120: Cross-subject split ---
    ntu_val_test_subjects = set()
    for name, path in ntu_files:
        subj = parse_ntu_subject(name)
        if subj is not None and subj not in NTU_TRAIN_SUBJECTS:
            ntu_val_test_subjects.add(subj)

    # Split remaining subjects 50/50 into val/test
    val_test_list = sorted(ntu_val_test_subjects)
    rng = random.Random(42)
    rng.shuffle(val_test_list)
    half = len(val_test_list) // 2
    ntu_val_subjects = set(val_test_list[:half])
    ntu_test_subjects = set(val_test_list[half:])

    for name, path in ntu_files:
        subj = parse_ntu_subject(name)
        if subj is None:
            assignments["train"].append(path)  # fallback
        elif subj in NTU_TRAIN_SUBJECTS:
            assignments["train"].append(path)
        elif subj in ntu_val_subjects:
            assignments["val"].append(path)
        else:
            assignments["test"].append(path)

    print(f"\nNTU120 split: {len(NTU_TRAIN_SUBJECTS)} train subjects, "
          f"{len(ntu_val_subjects)} val subjects, {len(ntu_test_subjects)} test subjects")

    # --- FallVision: Video-level split ---
    if fv_files:
        fv_train, fv_val, fv_test = split_groups(sorted(fv_files.keys()))
        for key, paths in fv_files.items():
            if key in fv_train:
                assignments["train"].extend(paths)
            elif key in fv_val:
                assignments["val"].extend(paths)
            else:
                assignments["test"].extend(paths)
        print(f"FallVision split: {len(fv_train)} train videos, "
              f"{len(fv_val)} val videos, {len(fv_test)} test videos")

    # --- Le2i: Video-level split ---
    if le2i_files:
        le2i_train, le2i_val, le2i_test = split_groups(sorted(le2i_files.keys()))
        for key, paths in le2i_files.items():
            if key in le2i_train:
                assignments["train"].extend(paths)
            elif key in le2i_val:
                assignments["val"].extend(paths)
            else:
                assignments["test"].extend(paths)
        print(f"Le2i split: {len(le2i_train)} train videos, "
              f"{len(le2i_val)} val videos, {len(le2i_test)} test videos")

    # --- Write files to split directories ---
    print(f"\nWriting splits to {output_dir}/...")
    for split, paths in assignments.items():
        split_dir = output_dir / split
        # Clear existing files in split dir
        for existing in split_dir.glob("*.json"):
            existing.unlink()

        for src in paths:
            dst = split_dir / src.name
            if args.copy:
                shutil.copy2(src, dst)
            else:
                # Create relative symlink
                rel = os.path.relpath(src, split_dir)
                dst.symlink_to(rel)

    # --- Summary ---
    print(f"\n{'':=<70}")
    print(f"{'Split':<8} {'Total':>7}", end="")

    # Collect all class names
    import json
    class_counts: dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}
    source_counts: dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}

    for split, paths in assignments.items():
        for p in paths:
            with open(p) as f:
                data = json.load(f)
            label = data.get("label", "unknown")
            class_counts[split][label] += 1
            if p.name.startswith("ntu_"):
                source_counts[split]["NTU"] += 1
            elif p.name.startswith("fv_"):
                source_counts[split]["FV"] += 1
            elif p.name.startswith("le2i_"):
                source_counts[split]["Le2i"] += 1

    all_classes = sorted(set().union(*class_counts.values()))
    for cls in all_classes:
        print(f" {cls:>14}", end="")
    print()
    print(f"{'':=<70}")

    for split in ("train", "val", "test"):
        total = len(assignments[split])
        print(f"{split:<8} {total:>7}", end="")
        for cls in all_classes:
            print(f" {class_counts[split][cls]:>14}", end="")
        print()

        # Source breakdown
        src_str = ", ".join(f"{k}:{v}" for k, v in sorted(source_counts[split].items()))
        print(f"{'':8} {'':>7}  ({src_str})")

    print(f"{'':=<70}")
    total_all = sum(len(v) for v in assignments.values())
    print(f"{'Total':<8} {total_all:>7}", end="")
    for cls in all_classes:
        total_cls = sum(class_counts[s][cls] for s in class_counts)
        print(f" {total_cls:>14}", end="")
    print()


if __name__ == "__main__":
    main()
