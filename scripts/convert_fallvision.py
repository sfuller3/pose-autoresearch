#!/usr/bin/env python3
"""Convert FallVision and Le2i datasets to Vistarra pose JSON format.

FallVision: Pre-extracted COCO-17 keypoint CSVs in RAR archives.
  No YOLO needed — direct CSV → JSON conversion.

Le2i: Video files requiring YOLO pose extraction.

FallVision (Harvard Dataverse, CC0):
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/75QPKK
  Structure after download:
    data/raw/fallvision/Fall Detection Video Dataset/
      Fall/{Bed,Chair,Stand}/{Mask Video,Raw Video}/*.rar
      No Fall/{Bed,Chair,Stand}/{Mask Video,Raw Video}/*.rar
  We use the *_keypoints_csv.rar files (pre-extracted poses).

Le2i (University of Burgundy):
  https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
  Structure:
    data/raw/le2i/{Coffee_room_01,Home_01,...}/{subdir}/Videos/*.avi

Usage:
    python scripts/convert_fallvision.py --source fallvision
    python scripts/convert_fallvision.py --source le2i
    python scripts/convert_fallvision.py --source all
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# COCO-17 keypoint order (must match our model's expected input)
COCO_KEYPOINTS = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
]
KEYPOINT_TO_IDX = {name: idx for idx, name in enumerate(COCO_KEYPOINTS)}


# ============================================================================
# FallVision: CSV keypoints from RAR archives
# ============================================================================

def parse_keypoints_csv(csv_path: Path, fps: int = 30) -> list[dict]:
    """Parse a FallVision keypoints CSV into a list of frame dicts.

    CSV format: Frame,Keypoint,X,Y,Confidence
    Each frame has 17 rows (one per COCO keypoint).

    Returns:
        List of dicts with 'keypoints' (17x3 list) and 'timestamp'.
    """
    frames_data: dict[int, list[list[float]]] = defaultdict(
        lambda: [[0.0, 0.0, 0.0] for _ in range(17)]
    )

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_num = int(row["Frame"])
            keypoint_name = row["Keypoint"].strip()
            x = float(row["X"])
            y = float(row["Y"])
            conf = float(row["Confidence"])

            if keypoint_name in KEYPOINT_TO_IDX:
                idx = KEYPOINT_TO_IDX[keypoint_name]
                frames_data[frame_num][idx] = [x, y, conf]

    # Convert to sorted frame list
    if not frames_data:
        return []

    all_frames = []
    for frame_num in sorted(frames_data.keys()):
        all_frames.append({
            "keypoints": frames_data[frame_num],
            "timestamp": (frame_num - 1) / fps,  # Frame numbers are 1-based
        })

    return all_frames


def segment_frames(
    all_frames: list[dict],
    seq_len: int = 150,
    overlap: float = 0.5,
    fps: int = 30,
) -> list[list[dict]]:
    """Segment a list of frames into fixed-length sequences with sliding window."""
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


def extract_rar(rar_path: Path, dest_dir: Path) -> bool:
    """Extract a RAR archive. Returns True on success."""
    try:
        result = subprocess.run(
            ["unrar", "x", "-o+", str(rar_path), str(dest_dir) + "/"],
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    Error extracting {rar_path.name}: {e}")
        return False


def convert_fallvision(
    data_dir: Path,
    output_dir: Path,
    seq_len: int = 150,
    fps: int = 30,
) -> int:
    """Convert FallVision pre-extracted keypoint CSVs to pose JSON.

    Finds all *_keypoints_csv.rar files, extracts them, parses CSVs,
    segments into sequences, and saves as JSON.
    """
    count = 0

    # Find the actual dataset directory
    fv_root = data_dir / "Fall Detection Video Dataset"
    if not fv_root.exists():
        # Maybe files are directly in data_dir
        fv_root = data_dir

    # Find all keypoints RAR files
    keypoint_rars = list(fv_root.rglob("*keypoints*.rar"))
    if not keypoint_rars:
        print(f"  No keypoints RAR files found in {fv_root}")
        print(f"  Looking for pattern: *keypoints*.rar")
        return 0

    print(f"  Found {len(keypoint_rars)} keypoints RAR archives")

    for rar_path in sorted(keypoint_rars):
        # Determine label from path
        path_str = str(rar_path)
        if "/Fall/" in path_str and "/No Fall/" not in path_str:
            label = "fall"
        elif "/No Fall/" in path_str:
            label = "sitting_standing"
        else:
            # Try to infer from filename
            name_lower = rar_path.stem.lower()
            if name_lower.startswith("f_") or "fall" in name_lower:
                label = "fall"
            else:
                label = "sitting_standing"

        # Determine scene from path (Bed, Chair, Stand)
        scene = "unknown"
        for s in ["Bed", "Chair", "Stand"]:
            if s in path_str:
                scene = s.lower()
                break

        print(f"  Extracting {rar_path.name} (label={label}, scene={scene})...")

        # Extract to temp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            if not extract_rar(rar_path, tmp_path):
                print(f"    Failed to extract {rar_path.name}")
                continue

            # Find all CSV files in extracted archive
            csv_files = list(tmp_path.rglob("*.csv"))
            print(f"    {len(csv_files)} CSV files")

            for csv_file in sorted(csv_files):
                try:
                    all_frames = parse_keypoints_csv(csv_file, fps)
                except Exception as e:
                    print(f"    Error parsing {csv_file.name}: {e}")
                    continue

                if not all_frames:
                    continue

                sequences = segment_frames(all_frames, seq_len, overlap=0.5, fps=fps)

                for seq_idx, frames in enumerate(sequences):
                    sample = {
                        "frames": frames,
                        "label": label,
                        "duration": len(frames) / fps,
                        "source": f"fallvision/{label}/{scene}/{csv_file.stem}",
                    }

                    filename = f"fv_{label}_{scene}_{csv_file.stem}_{seq_idx:04d}.json"
                    # Sanitize filename
                    filename = filename.replace(" ", "_")
                    with open(output_dir / filename, "w") as f:
                        json.dump(sample, f)
                    count += 1

            rar_seqs = sum(1 for _ in output_dir.glob(f"fv_{label}_{scene}_*"))
            print(f"    -> {count} sequences so far")

    return count


# ============================================================================
# Le2i: Video files requiring YOLO pose extraction
# ============================================================================

def extract_poses_from_video(
    video_path: Path,
    model,
    seq_len: int = 150,
    fps: int = 30,
    overlap: float = 0.5,
) -> list[list[dict]]:
    """Run YOLO pose estimation on a video and segment into sequences."""
    results = model(str(video_path), stream=True, verbose=False)

    all_frames = []
    frame_idx = 0

    for result in results:
        if result.keypoints is not None and len(result.keypoints) > 0:
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
            all_frames.append({
                "keypoints": [[0.0, 0.0, 0.0]] * 17,
                "timestamp": frame_idx / fps,
            })

        frame_idx += 1

    return segment_frames(all_frames, seq_len, overlap, fps)


def convert_le2i(
    data_dir: Path,
    output_dir: Path,
    model,
    seq_len: int = 150,
    fps: int = 30,
) -> int:
    """Convert Le2i dataset videos to pose JSON.

    Le2i structure (from Kaggle):
        data/raw/le2i/Coffee_room_01/Coffee_room_01/Videos/*.avi
        data/raw/le2i/Home_01/Home_01/Videos/*.avi
        data/raw/le2i/Lecture_room/Lecture room/*.avi
        ...

    Label inference:
        - Videos in directories/filenames containing 'fall' or 'chute' = fall
        - Le2i annotation files have fall frame ranges, but for simplicity
          we label entire short clips
    """
    count = 0

    # Find all video files (handles spaces in paths via Path objects)
    videos = list(data_dir.rglob("*.avi")) + list(data_dir.rglob("*.mp4"))
    print(f"  Found {len(videos)} videos in Le2i")

    for i, video_path in enumerate(sorted(videos)):
        # Determine label from path and filename
        path_lower = str(video_path).lower()

        # Le2i labels: most videos in the dataset are labeled by annotation files
        # The dataset contains both fall and ADL (activities of daily living) videos
        # Fall videos typically have lower numbers, ADL have higher numbers
        # For now, we label all as "sitting_standing" (non-fall default)
        # and rely on annotation files if available
        label = "sitting_standing"

        # Check for annotation files near the video
        anno_dir = video_path.parent.parent
        if "annotation" in str(anno_dir).lower():
            anno_dir = video_path.parent

        # Check common annotation patterns
        for anno_pattern in ["Annotation*", "annotation*"]:
            anno_dirs = list(video_path.parent.parent.glob(anno_pattern))
            if not anno_dirs:
                anno_dirs = list(video_path.parent.glob(anno_pattern))
            for ad in anno_dirs:
                # If annotation file exists for this video, it's a fall video
                video_stem = video_path.stem.replace(" ", "")
                anno_files = list(ad.glob(f"*{video_path.stem}*")) + list(ad.glob("*.txt"))
                for af in anno_files:
                    try:
                        content = af.read_text().strip()
                        if content and any(c.isdigit() for c in content):
                            # Annotation file with frame numbers = fall video
                            label = "fall"
                            break
                    except Exception:
                        pass
                if label == "fall":
                    break

        # Determine scene from directory structure
        # Go up to find the environment name
        scene = "unknown"
        for parent in video_path.parents:
            if parent == data_dir:
                break
            name = parent.name
            if name and name != "Videos" and not name.startswith("."):
                scene = name.replace(" ", "_")

        try:
            sequences = extract_poses_from_video(video_path, model, seq_len, fps)
        except Exception as e:
            print(f"    Error: {video_path.name}: {e}")
            continue

        for seq_idx, frames in enumerate(sequences):
            sample = {
                "frames": frames,
                "label": label,
                "duration": len(frames) / fps,
                "source": f"le2i/{scene}/{video_path.stem}",
            }

            # Sanitize filename (handle spaces)
            safe_stem = video_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
            safe_scene = scene.replace(" ", "_")
            filename = f"le2i_{label}_{safe_scene}_{safe_stem}_{seq_idx:04d}.json"
            with open(output_dir / filename, "w") as f:
                json.dump(sample, f)
            count += 1

        if sequences:
            print(f"    [{i+1}/{len(videos)}] {video_path.name}: {len(sequences)} seq -> {label}")

    return count


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert FallVision/Le2i datasets to pose JSON"
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
        help="YOLO pose model for Le2i video extraction (default: yolo11s-pose.pt)",
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

    args.output.mkdir(parents=True, exist_ok=True)

    total = 0

    if args.source in ("fallvision", "all"):
        fv_dir = Path("data/raw/fallvision")
        if fv_dir.exists():
            print(f"\nConverting FallVision from {fv_dir}...")
            print(f"  (Using pre-extracted keypoint CSVs — no YOLO needed)")
            n = convert_fallvision(fv_dir, args.output, args.seq_len, args.fps)
            total += n
            print(f"FallVision: {n} sequences\n")
        else:
            print(f"\nFallVision not found at {fv_dir}")

    if args.source in ("le2i", "all"):
        le2i_dir = Path("data/raw/le2i")
        if le2i_dir.exists():
            print(f"\nConverting Le2i from {le2i_dir}...")

            # Only load YOLO for Le2i (video processing)
            from ultralytics import YOLO
            print(f"Loading YOLO model: {args.model}")
            model = YOLO(args.model)

            n = convert_le2i(le2i_dir, args.output, model, args.seq_len, args.fps)
            total += n
            print(f"Le2i: {n} sequences\n")
        else:
            print(f"\nLe2i not found at {le2i_dir}")

    print(f"\nTotal: {total} sequences saved to {args.output}")


if __name__ == "__main__":
    main()
