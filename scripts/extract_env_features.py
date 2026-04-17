# scripts/extract_env_features.py
"""
Extract environment features from source videos for training data.

For each pose JSON sample, finds the corresponding video frame(s),
runs Roboflow object detection, and saves the spatial relationship
features as a companion .env.npy file.

Usage:
  python scripts/extract_env_features.py \
    --data-dir data/splits/train \
    --video-dir data/raw/le2i \
    --model-id elder-care-environment/1 \
    --api-key YOUR_KEY
"""
import argparse
import json
from pathlib import Path

import numpy as np

# Import from stream_detect to reuse EnvironmentDetector
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_features_for_split(
    data_dir: Path,
    video_dir: Path | None,
    model_id: str,
    api_key: str,
):
    """Extract env features for all JSON files in a split directory."""
    from stream_detect import EnvironmentDetector

    detector = EnvironmentDetector(
        model_id=model_id,
        api_key=api_key,
        infer_interval=1,  # detect every frame during extraction
    )

    json_files = sorted(data_dir.glob("*.json"))
    print(f"Processing {len(json_files)} samples from {data_dir}")

    for i, json_file in enumerate(json_files):
        env_path = json_file.with_suffix(".env.npy")
        if env_path.exists():
            continue

        with open(json_file) as f:
            data = json.load(f)

        # For NTU data (no source video): use zero features
        # For Le2i/FallVision: could extract from source video if available
        source = data.get("source", "")
        if not video_dir or "ntu" in source:
            features = np.zeros(
                EnvironmentDetector.NUM_CONTEXT_CLASSES * 4,
                dtype=np.float32,
            )
        else:
            # TODO: match JSON to source video frame and run detection
            # For now, use zero features — will be filled when
            # facility-specific video data is available
            print(f"  WARNING: No video source for {json_file.name}, using zero features")
            features = np.zeros(
                EnvironmentDetector.NUM_CONTEXT_CLASSES * 4,
                dtype=np.float32,
            )

        np.save(env_path, features)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(json_files)}")

    print(f"Done. Saved {len(json_files)} .env.npy files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()
    extract_features_for_split(args.data_dir, args.video_dir,
                                args.model_id, args.api_key)
