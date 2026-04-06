#!/usr/bin/env python3
"""Convert FallVision and Le2i video datasets to Vistarra pose JSON format.

Downloads are manual (see below), then this script:
1. Runs YOLO pose estimation on each video
2. Segments into fixed-length sequences with sliding window
3. Saves as JSON with event labels

FallVision (Harvard Dataverse, CC0):
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK
  - Download MP4 files into data/raw/fallvision/fall/ and data/raw/fallvision/nofall/

Le2i (University of Burgundy):
  https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
  - Download and extract into data/raw/le2i/

Usage:
    python scripts/convert_fallvision.py --source fallvision
    python scripts/convert_fallvision.py --source le2i
    python scripts/convert_fallvision.py --source all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def extract_poses_from_video(
    video_path: Path,
    model,
    seq_len: int = 150,
    fps: int = 30,
    overlap: float = 0.5,
) -> list[list[dict]]:
    """Run YOLO pose estimation on a video and segment into sequences.

    Args:
        video_path: Path to video file.
        model: YOLO model instance.
        seq_len: Frames per output sequence.
        fps: Target frame rate.
        overlap: Sliding window overlap fraction.

    Returns:
        List of frame sequences, each a list of dicts with 'keypoints' and 'timestamp'.
    """
    results = model(str(video_path), stream=True, verbose=False)

    all_frames = []
    frame_idx = 0

    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
            # Take the first (most confident) person
            try:
                kp_xy = result.keypoints.xy[0].cpu().numpy()   # (17, 2)
                kp_conf = result.keypoints.conf[0].cpu().numpy()  # (17,)
            except (IndexError, AttributeError):
                frame_idx += 1
                continue

            keypoints = np.concatenate([
                kp_xy,
                kp_conf[:, None],
            ], axis=1).tolist()  # (17, 3)

            all_frames.append({
                "keypoints": keypoints,
                "timestamp": frame_idx / fps,
            })
        else:
            # No person detected — insert zero frame
            all_frames.append({
                "keypoints": [[0.0, 0.0, 0.0]] * 17,
                "timestamp": frame_idx / fps,
            })

        frame_idx += 1

    # Segment into fixed-length sequences with sliding window
    sequences = []
    stride = max(1, int(seq_len * (1 - overlap)))

    for start in range(0, len(all_frames) - seq_len + 1, stride):
        seq = all_frames[start : start + seq_len]
        sequences.append(seq)

    # If video is shorter than seq_len, pad and use as single sequence
    if not sequences and all_frames:
        while len(all_frames) < seq_len:
            all_frames.append({
                "keypoints": [[0.0, 0.0, 0.0]] * 17,
                "timestamp": len(all_frames) / fps,
            })
        sequences.append(all_frames[:seq_len])

    return sequences


def convert_fallvision(
    data_dir: Path,
    output_dir: Path,
    model,
    seq_len: int = 150,
    fps: int = 30,
) -> int:
    """Convert FallVision dataset videos to pose JSON.

    Expected directory structure:
        data/raw/fallvision/fall/*.mp4
        data/raw/fallvision/nofall/*.mp4

    FallVision labels map to our event classes:
        fall/ -> "fall"
        nofall/ -> "sitting_standing" (default non-fall class)
    """
    count = 0

    label_map = {
        "fall": "fall",
        "nofall": "sitting_standing",
    }

    for subdir, label in label_map.items():
        video_dir = data_dir / subdir
        if not video_dir.exists():
            print(f"  Skipping {video_dir} (not found)")
            continue

        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        print(f"  {subdir}/: {len(videos)} videos -> label '{label}'")

        for video_path in videos:
            try:
                sequences = extract_poses_from_video(video_path, model, seq_len, fps)
            except Exception as e:
                print(f"    Error processing {video_path.name}: {e}")
                continue

            for seq_idx, frames in enumerate(sequences):
                sample = {
                    "frames": frames,
                    "label": label,
                    "duration": len(frames) / fps,
                    "source": f"fallvision/{subdir}/{video_path.stem}",
                }

                filename = f"fv_{label}_{video_path.stem}_{seq_idx:04d}.json"
                with open(output_dir / filename, "w") as f:
                    json.dump(sample, f)
                count += 1

            if sequences:
                print(f"    {video_path.name}: {len(sequences)} sequences")

    return count


def convert_le2i(
    data_dir: Path,
    output_dir: Path,
    model,
    seq_len: int = 150,
    fps: int = 30,
) -> int:
    """Convert Le2i dataset videos to pose JSON.

    Expected directory structure:
        data/raw/le2i/Coffee_room_01/Videos/video_*.avi
        data/raw/le2i/Home_01/Videos/video_*.avi
        ...

    Le2i has fall and non-fall videos. Filenames or subdirectories
    typically indicate the label. We use directory naming convention:
    folders containing 'fall' in the path are falls.
    """
    count = 0

    videos = list(data_dir.rglob("*.avi")) + list(data_dir.rglob("*.mp4"))
    print(f"  Found {len(videos)} videos in Le2i")

    for video_path in videos:
        # Determine label from path
        path_lower = str(video_path).lower()
        if "fall" in video_path.stem.lower() or "chute" in path_lower:
            label = "fall"
        else:
            label = "sitting_standing"

        try:
            sequences = extract_poses_from_video(video_path, model, seq_len, fps)
        except Exception as e:
            print(f"    Error: {video_path.name}: {e}")
            continue

        scene = video_path.parent.parent.name if video_path.parent.name == "Videos" else video_path.parent.name

        for seq_idx, frames in enumerate(sequences):
            sample = {
                "frames": frames,
                "label": label,
                "duration": len(frames) / fps,
                "source": f"le2i/{scene}/{video_path.stem}",
            }

            filename = f"le2i_{label}_{scene}_{video_path.stem}_{seq_idx:04d}.json"
            with open(output_dir / filename, "w") as f:
                json.dump(sample, f)
            count += 1

        if sequences:
            print(f"    {video_path.name}: {len(sequences)} seq -> {label}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert FallVision/Le2i video datasets to pose JSON"
    )
    parser.add_argument(
        "--source",
        choices=["fallvision", "le2i", "all"],
        default="all",
        help="Which dataset to convert",
    )
    parser.add_argument(
        "--model",
        default="yolo11s-pose.pt",
        help="YOLO pose model (default: yolo11s-pose.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for pose JSON files",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=150,
        help="Frames per sequence (default: 150 = 5s at 30fps)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    args.output.mkdir(parents=True, exist_ok=True)

    total = 0

    if args.source in ("fallvision", "all"):
        fv_dir = Path("data/raw/fallvision")
        if fv_dir.exists():
            print(f"\nConverting FallVision from {fv_dir}...")
            n = convert_fallvision(fv_dir, args.output, model, args.seq_len, args.fps)
            total += n
            print(f"FallVision: {n} sequences")
        else:
            print(f"\nFallVision not found at {fv_dir}")
            print("Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK")
            print(f"Place fall videos in {fv_dir}/fall/ and non-fall in {fv_dir}/nofall/")

    if args.source in ("le2i", "all"):
        le2i_dir = Path("data/raw/le2i")
        if le2i_dir.exists():
            print(f"\nConverting Le2i from {le2i_dir}...")
            n = convert_le2i(le2i_dir, args.output, model, args.seq_len, args.fps)
            total += n
            print(f"Le2i: {n} sequences")
        else:
            print(f"\nLe2i not found at {le2i_dir}")
            print("Download from: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia")
            print(f"Extract into {le2i_dir}/")

    print(f"\nTotal: {total} sequences saved to {args.output}")


if __name__ == "__main__":
    main()
