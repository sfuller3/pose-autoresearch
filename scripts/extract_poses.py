"""
Extract pose sequences from videos using YOLO26.

Usage:
    python scripts/extract_poses.py --video data/raw/*.mp4 --output data/processed/
"""

import argparse
from pathlib import Path
import json
import numpy as np
from ultralytics import YOLO


def extract_poses_from_video(video_path, model, seq_len=30, fps=30):
    """
    Extract pose sequences from a video file.
    
    Args:
        video_path: Path to video file
        model: YOLO pose model
        seq_len: Number of frames per sequence
        fps: Frames per second
    
    Returns:
        List of pose sequences, each with seq_len frames
    """
    print(f"Processing: {video_path}")
    
    # Run pose estimation
    results = model(video_path, stream=True, verbose=False)
    
    sequences = []
    current_seq = []
    frame_idx = 0
    
    for result in results:
        # Extract keypoints
        if len(result.keypoints) > 0:
            # Take first person if multiple detected
            keypoints = result.keypoints.xy[0].cpu().numpy()  # (17, 2)
            confidence = result.keypoints.conf[0].cpu().numpy()  # (17,)
            
            # Combine into (17, 3) array
            frame_data = np.concatenate([
                keypoints,
                confidence[:, None]
            ], axis=1).tolist()
            
            current_seq.append({
                "keypoints": frame_data,
                "timestamp": frame_idx / fps,
            })
            
            frame_idx += 1
            
            # When we have seq_len frames, save and start new sequence
            if len(current_seq) >= seq_len:
                sequences.append(current_seq[:seq_len])
                # Sliding window with 50% overlap
                current_seq = current_seq[seq_len // 2:]
    
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Extract poses from videos")
    parser.add_argument(
        "--video",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to video file(s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for pose JSON files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n-pose.pt",
        help="YOLO pose model",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=150,
        help="Frames per sequence",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Event label for these videos",
    )
    args = parser.parse_args()
    
    # Load YOLO model
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    total_sequences = 0
    
    for video_path in args.video:
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            continue
        
        # Extract poses
        sequences = extract_poses_from_video(
            video_path,
            model,
            seq_len=args.seq_len,
            fps=args.fps,
        )
        
        # Save each sequence as JSON
        video_name = video_path.stem
        
        for seq_idx, frames in enumerate(sequences):
            # Create sample
            sample = {
                "frames": frames,
                "label": args.label,
                "duration": len(frames) / args.fps,
                "source_video": str(video_path),
            }
            
            # Save
            output_file = args.output / f"{video_name}_{seq_idx:04d}.json"
            with open(output_file, 'w') as f:
                json.dump(sample, f, indent=2)
            
            total_sequences += 1
        
        print(f"  Extracted {len(sequences)} sequences")
    
    print()
    print(f"Total sequences extracted: {total_sequences}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
