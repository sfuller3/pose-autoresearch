#!/usr/bin/env python3
"""Overlay extracted JSON skeleton keypoints onto original Le2i video frames.

Produces a side-by-side or overlay video/GIF for visual verification that
the AVI → JSON conversion preserved pose data correctly.

Usage:
    # Single video + its JSON sequence(s)
    python scripts/verify_le2i_overlay.py \\
        --video test_le2i/test_le2i/Coffee_room_01/Coffee_room_01/Videos/video\ \(1\).avi \\
        --json data/le2i_debug/le2i_fall_Coffee_room_01_video_1_0000.json \\
        -o output/verify_coffee1.mp4

    # Batch: auto-match all JSONs to their source videos
    python scripts/verify_le2i_overlay.py \\
        --video-dir test_le2i \\
        --json-dir data/le2i_debug \\
        -o output/le2i_verify/ \\
        --batch

    # Limit to first N frames for quick checks
    python scripts/verify_le2i_overlay.py \\
        --video test_le2i/.../video\ \(1\).avi \\
        --json data/le2i_debug/le2i_fall_*.json \\
        -o output/verify.mp4 --max-frames 60
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# COCO-17 skeleton connections
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

# Colors by body region (BGR)
JOINT_COLORS = {
    "head": (0, 255, 255),      # yellow
    "torso": (0, 255, 0),       # green
    "left_arm": (255, 128, 0),  # blue-ish
    "right_arm": (0, 128, 255), # orange
    "left_leg": (255, 0, 128),  # magenta
    "right_leg": (128, 0, 255), # purple
}

JOINT_REGION = {
    0: "head", 1: "head", 2: "head", 3: "head", 4: "head",
    5: "torso", 6: "torso",
    7: "left_arm", 9: "left_arm",
    8: "right_arm", 10: "right_arm",
    11: "torso", 12: "torso",
    13: "left_leg", 15: "left_leg",
    14: "right_leg", 16: "right_leg",
}

BONE_REGION = {
    (0, 1): "head", (0, 2): "head", (1, 3): "head", (2, 4): "head",
    (5, 6): "torso", (5, 11): "torso", (6, 12): "torso", (11, 12): "torso",
    (5, 7): "left_arm", (7, 9): "left_arm",
    (6, 8): "right_arm", (8, 10): "right_arm",
    (11, 13): "left_leg", (13, 15): "left_leg",
    (12, 14): "right_leg", (14, 16): "right_leg",
}

CONF_THRESHOLD = 0.1


def draw_skeleton_on_frame(
    frame: np.ndarray,
    keypoints: list[list[float]],
    thickness: int = 2,
    radius: int = 4,
) -> np.ndarray:
    """Draw COCO-17 skeleton onto a video frame (in-place)."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Draw bones first (behind joints)
    for (i, j) in SKELETON_EDGES:
        x1, y1, c1 = keypoints[i]
        x2, y2, c2 = keypoints[j]
        if c1 < CONF_THRESHOLD or c2 < CONF_THRESHOLD:
            continue
        if x1 == 0 and y1 == 0:
            continue
        if x2 == 0 and y2 == 0:
            continue

        region = BONE_REGION.get((i, j), "torso")
        color = JOINT_COLORS[region]

        # Alpha based on min confidence
        alpha = min(c1, c2)
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw joints
    for idx, (x, y, conf) in enumerate(keypoints):
        if conf < CONF_THRESHOLD or (x == 0 and y == 0):
            continue
        region = JOINT_REGION.get(idx, "torso")
        color = JOINT_COLORS[region]
        center = (int(x), int(y))
        cv2.circle(overlay, center, radius, color, -1, cv2.LINE_AA)
        # White outline for visibility
        cv2.circle(overlay, center, radius, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend overlay with original for slight transparency on bones
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return frame


def draw_info_text(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    label: str,
    source: str,
    keypoints: list[list[float]],
    seq_offset: int = 0,
) -> np.ndarray:
    """Draw metadata overlay on frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 if w < 400 else 0.55
    thick = 1

    # Detection stats for this frame
    detected = sum(1 for x, y, c in keypoints if c >= CONF_THRESHOLD and (x != 0 or y != 0))
    mean_conf = np.mean([c for _, _, c in keypoints if c >= CONF_THRESHOLD]) if detected > 0 else 0.0

    lines = [
        f"Label: {label}",
        f"Frame: {frame_idx + seq_offset}/{total_frames + seq_offset} | Joints: {detected}/17",
        f"Conf: {mean_conf:.2f} | Source: {source}",
    ]

    # Semi-transparent background bar
    bar_h = 18 * len(lines) + 8
    bar = frame[:bar_h, :].copy()
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(frame[:bar_h, :], 0.6, bar, 0.4, 0, frame[:bar_h, :])

    for i, line in enumerate(lines):
        y = 15 + i * 18
        cv2.putText(frame, line, (5, y), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    return frame


def match_json_to_video(
    json_path: Path,
    video_dir: Path,
) -> Path | None:
    """Find the source video for a Le2i JSON file based on the source field."""
    with open(json_path) as f:
        data = json.load(f)

    source = data.get("source", "")
    # source looks like "le2i/Coffee_room_01/video (10)"
    parts = source.replace("le2i/", "").split("/")
    if len(parts) < 2:
        return None

    scene = parts[0]
    video_stem = parts[1]  # e.g. "video (10)"

    # Search for the video file recursively
    for ext in ("*.avi", "*.mp4"):
        for vpath in video_dir.rglob(ext):
            if vpath.stem == video_stem and scene in str(vpath):
                return vpath

    return None


def get_sequence_offset(json_path: Path) -> int:
    """Extract sequence index from filename to compute frame offset.

    e.g. le2i_fall_Coffee_room_01_video_10_0001.json -> seq_idx=1, offset=75
    (50% overlap with 150-frame windows -> stride of 75)
    """
    match = re.search(r"_(\d{4})\.json$", json_path.name)
    if match:
        seq_idx = int(match.group(1))
        stride = 75  # 150 * 0.5 overlap
        return seq_idx * stride
    return 0


def render_overlay(
    video_path: Path,
    json_path: Path,
    output_path: Path,
    max_frames: int | None = None,
    fps_out: int | None = None,
) -> dict:
    """Render video with skeleton overlay from JSON keypoints.

    Returns:
        dict with stats: total_frames, detected_frames, mean_conf, output_path
    """
    with open(json_path) as f:
        data = json.load(f)

    frames_data = data["frames"]
    label = data.get("label", "unknown")
    source = data.get("source", json_path.stem)

    if max_frames:
        frames_data = frames_data[:max_frames]

    # Open source video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = fps_out or int(vid_fps)
    seq_offset = get_sequence_offset(json_path)

    # Seek to the start frame for this sequence
    if seq_offset > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seq_offset)

    # Set up output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (vid_w, vid_h))
    elif suffix == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (vid_w, vid_h))
    else:
        # For GIF, collect frames and use PIL
        writer = None
        gif_frames = []

    stats = {"total_frames": len(frames_data), "detected_frames": 0, "conf_sum": 0.0}

    for frame_idx, frame_data in enumerate(frames_data):
        ret, img = cap.read()
        if not ret:
            # Video shorter than JSON — use black frame
            img = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

        keypoints = frame_data["keypoints"]
        detected = sum(1 for x, y, c in keypoints if c >= CONF_THRESHOLD and (x != 0 or y != 0))
        if detected > 0:
            stats["detected_frames"] += 1
            stats["conf_sum"] += np.mean([c for _, _, c in keypoints if c >= CONF_THRESHOLD])

        # Draw skeleton
        img = draw_skeleton_on_frame(img, keypoints)

        # Draw info
        img = draw_info_text(img, frame_idx, len(frames_data), label, source, keypoints, seq_offset)

        if writer is not None:
            writer.write(img)
        else:
            # Convert BGR -> RGB for PIL
            gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    cap.release()

    if writer is not None:
        writer.release()
    elif gif_frames:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in gif_frames]
        pil_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000 / out_fps),
            loop=0,
        )

    stats["mean_conf"] = stats["conf_sum"] / max(stats["detected_frames"], 1)
    stats["detection_rate"] = stats["detected_frames"] / max(stats["total_frames"], 1)
    stats["output_path"] = str(output_path)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Overlay JSON skeleton keypoints on original Le2i videos for verification"
    )

    # Input modes
    parser.add_argument("--video", type=Path, help="Single source video path")
    parser.add_argument("--json", type=Path, nargs="+", help="JSON file(s) to overlay")
    parser.add_argument("--video-dir", type=Path, help="Directory of source videos (batch mode)")
    parser.add_argument("--json-dir", type=Path, help="Directory of JSON files (batch mode)")
    parser.add_argument("--batch", action="store_true", help="Auto-match all JSONs to videos")

    # Output
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file or directory")
    parser.add_argument("--format", choices=["mp4", "gif", "avi"], default="mp4")
    parser.add_argument("--fps", type=int, default=None, help="Output FPS (default: match source)")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per sequence")
    parser.add_argument("--max-sequences", type=int, default=None, help="Limit sequences in batch")

    args = parser.parse_args()

    if args.batch:
        if not args.video_dir or not args.json_dir:
            parser.error("--batch requires --video-dir and --json-dir")

        json_files = sorted(args.json_dir.glob("le2i_*.json"))
        if args.max_sequences:
            json_files = json_files[:args.max_sequences]

        args.output.mkdir(parents=True, exist_ok=True)

        print(f"Batch mode: {len(json_files)} JSON files")
        print(f"Video dir: {args.video_dir}")
        print(f"Output dir: {args.output}")
        print()

        results = []
        for i, jf in enumerate(json_files):
            video_path = match_json_to_video(jf, args.video_dir)
            if video_path is None:
                print(f"  [{i+1}/{len(json_files)}] {jf.name}: SKIP (no matching video)")
                continue

            out_name = jf.stem + f".{args.format}"
            out_path = args.output / out_name

            try:
                stats = render_overlay(video_path, jf, out_path, args.max_frames, args.fps)
                det = stats["detection_rate"] * 100
                conf = stats["mean_conf"]
                quality = "GOOD" if det > 80 else "FAIR" if det > 50 else "POOR"
                print(f"  [{i+1}/{len(json_files)}] {jf.name}: {det:.0f}% detected, "
                      f"conf={conf:.2f} [{quality}] -> {out_name}")
                results.append(stats)
            except Exception as e:
                print(f"  [{i+1}/{len(json_files)}] {jf.name}: ERROR: {e}")

        # Summary
        if results:
            avg_det = np.mean([r["detection_rate"] for r in results]) * 100
            avg_conf = np.mean([r["mean_conf"] for r in results])
            good = sum(1 for r in results if r["detection_rate"] > 0.8)
            fair = sum(1 for r in results if 0.5 < r["detection_rate"] <= 0.8)
            poor = sum(1 for r in results if r["detection_rate"] <= 0.5)
            print(f"\n{'='*60}")
            print(f"Summary: {len(results)} sequences rendered")
            print(f"  Avg detection rate: {avg_det:.1f}%")
            print(f"  Avg confidence:     {avg_conf:.3f}")
            print(f"  Quality: {good} GOOD / {fair} FAIR / {poor} POOR")
            print(f"  Output: {args.output}/")

    else:
        if not args.video or not args.json:
            parser.error("Provide --video and --json, or use --batch mode")

        for jf in args.json:
            if len(args.json) > 1:
                out_path = args.output / (jf.stem + f".{args.format}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = args.output

            print(f"Rendering: {jf.name}")
            print(f"  Video: {args.video}")
            print(f"  Output: {out_path}")

            stats = render_overlay(args.video, jf, out_path, args.max_frames, args.fps)
            det = stats["detection_rate"] * 100
            print(f"  Detection rate: {det:.0f}% ({stats['detected_frames']}/{stats['total_frames']})")
            print(f"  Mean confidence: {stats['mean_conf']:.3f}")
            print(f"  Done.")


if __name__ == "__main__":
    main()
