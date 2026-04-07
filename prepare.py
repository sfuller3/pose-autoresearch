"""
Data preparation and evaluation utilities.
DO NOT MODIFY THIS FILE (unless explicitly asked by human).

The agent should only edit train.py.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from typing import Tuple

from pose_autoresearch.graph import get_bone_pairs
from pose_autoresearch.augment import augment_pose_sequence

# ============================================================================
# CONSTANTS (fixed)
# ============================================================================

EVENT_CLASSES = [
    "fall",
    "eating",
    "working_together",
    "aggression",
    "unstable_gait",
    "wandering",
    "sitting_standing",
]

NUM_CLASSES = len(EVENT_CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(EVENT_CLASSES)}

NUM_KEYPOINTS = 17
VALUES_PER_KEYPOINT = 3  # x, y, confidence
INPUT_DIM = NUM_KEYPOINTS * VALUES_PER_KEYPOINT  # 51

# Temporal sequence: 5 seconds at 30fps
SEQ_LEN = 150
MAX_SEQ_LEN = 300  # 10 seconds max
FPS = 30

MAX_TIME_BUDGET_SECONDS = int(os.environ.get("POSE_AUTORESEARCH_MAX_TIME", "300"))
TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)

_force_device = os.environ.get("POSE_DEVICE", "").strip().lower()
if _force_device:
    DEVICE = torch.device(_force_device)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"


# ============================================================================
# MULTI-STREAM FEATURES
# ============================================================================

BONE_PAIRS = get_bone_pairs()
NUM_BONES = len(BONE_PAIRS)


def compute_bone_features(keypoints: np.ndarray) -> np.ndarray:
    """Compute bone vectors from joint positions.

    Args:
        keypoints: (seq_len, 17, 3) array -- (x, y, confidence).

    Returns:
        (seq_len, num_bones, 3) array -- bone vector (dx, dy, mean_conf).
    """
    bones = []
    for parent, child in BONE_PAIRS:
        dx = keypoints[:, child, 0] - keypoints[:, parent, 0]
        dy = keypoints[:, child, 1] - keypoints[:, parent, 1]
        conf = (keypoints[:, child, 2] + keypoints[:, parent, 2]) / 2
        bones.append(np.stack([dx, dy, conf], axis=-1))
    return np.stack(bones, axis=1)


def compute_velocity_features(keypoints: np.ndarray) -> np.ndarray:
    """Compute temporal velocity of each joint.

    Args:
        keypoints: (seq_len, 17, 3) array.

    Returns:
        (seq_len, 17, 3) array -- (vx, vy, confidence).
    """
    velocity = np.zeros_like(keypoints)
    velocity[1:, :, 0] = keypoints[1:, :, 0] - keypoints[:-1, :, 0]
    velocity[1:, :, 1] = keypoints[1:, :, 1] - keypoints[:-1, :, 1]
    velocity[:, :, 2] = keypoints[:, :, 2]
    return velocity


# ============================================================================
# DATASET
# ============================================================================

class PoseDataset(Dataset):
    """Dataset for pose sequences with event labels.

    Each sample is loaded from a JSON file containing:
    {
        "frames": [{"keypoints": [[x,y,c],...], "timestamp": float}, ...],
        "label": "fall",
        "duration": float
    }
    """

    def __init__(
        self,
        data_dir: Path,
        seq_len: int = SEQ_LEN,
        augment: bool = False,
        multi_stream: bool = False,
    ):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.augment = augment
        self.multi_stream = multi_stream
        self.samples: list[tuple[np.ndarray, int]] = []

        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            print(f"Warning: No JSON files in {data_dir}. Run data preparation first.")
            return

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            label_str = data["label"]
            if label_str not in CLASS_TO_IDX:
                continue
            label = CLASS_TO_IDX[label_str]

            frames = data["frames"]
            keypoints = []
            for frame in frames[:seq_len]:
                flat = np.array(frame["keypoints"]).flatten()
                if len(flat) < INPUT_DIM:
                    flat = np.pad(flat, (0, INPUT_DIM - len(flat)))
                elif len(flat) > INPUT_DIM:
                    flat = flat[:INPUT_DIM]
                keypoints.append(flat)

            while len(keypoints) < seq_len:
                keypoints.append(np.zeros(INPUT_DIM))

            poses = np.array(keypoints, dtype=np.float32)
            self.samples.append((poses, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        poses_flat, label = self.samples[idx]
        poses_tensor = torch.from_numpy(poses_flat.copy())

        if self.augment:
            reshaped = poses_tensor.view(self.seq_len, NUM_KEYPOINTS, VALUES_PER_KEYPOINT)
            reshaped = augment_pose_sequence(reshaped, target_len=self.seq_len)
            poses_tensor = reshaped.reshape(self.seq_len, INPUT_DIM)

        if self.multi_stream:
            kps = poses_tensor.view(self.seq_len, NUM_KEYPOINTS, VALUES_PER_KEYPOINT).numpy()
            bones = compute_bone_features(kps)
            velocity = compute_velocity_features(kps)

            joint_flat = poses_tensor
            bone_flat = torch.from_numpy(bones.reshape(self.seq_len, -1).copy())
            vel_flat = torch.from_numpy(velocity.reshape(self.seq_len, -1).copy())

            return (joint_flat, bone_flat, vel_flat), torch.tensor(label, dtype=torch.long)

        return poses_tensor, torch.tensor(label, dtype=torch.long)


# ============================================================================
# DATA LOADING
# ============================================================================

def get_dataloaders(
    data_dir: Path = PROCESSED_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    split: Tuple[float, float, float] = TRAIN_VAL_TEST_SPLIT,
    augment_train: bool = True,
    multi_stream: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders."""
    full_dataset = PoseDataset(data_dir, augment=False, multi_stream=multi_stream)

    if len(full_dataset) == 0:
        print("ERROR: No data found. Run one of:")
        print("  python prepare.py --dataset synthetic")
        print("  python scripts/convert_fallvision.py")
        raise RuntimeError("No training data")

    train_size = int(split[0] * len(full_dataset))
    val_size = int(split[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    if augment_train:
        aug_dataset = PoseDataset(data_dir, augment=True, multi_stream=multi_stream)
        train_subset = torch.utils.data.Subset(aug_dataset, train_subset.indices)

    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model accuracy and loss.

    Returns:
        (accuracy, avg_loss)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], (tuple, list)):
                inputs = tuple(t.to(device) for t in batch[0])
                labels = batch[1].to(device)
                logits = model(*inputs)
            else:
                poses, labels = batch
                poses = poses.to(device)
                labels = labels.to(device)
                logits = model(poses)

            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(len(dataloader), 1)
    return accuracy, avg_loss


def evaluate_per_class(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Per-class accuracy breakdown."""
    model.eval()
    class_correct = {cls: 0 for cls in EVENT_CLASSES}
    class_total = {cls: 0 for cls in EVENT_CLASSES}

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch[0], (tuple, list)):
                inputs = tuple(t.to(device) for t in batch[0])
                labels = batch[1].to(device)
                logits = model(*inputs)
            else:
                poses, labels = batch
                poses = poses.to(device)
                labels = labels.to(device)
                logits = model(poses)

            preds = torch.argmax(logits, dim=1)
            for pred, label in zip(preds, labels):
                cls = EVENT_CLASSES[label.item()]
                class_total[cls] += 1
                if pred == label:
                    class_correct[cls] += 1

    return {
        cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        for cls in EVENT_CLASSES
    }


# ============================================================================
# SYNTHETIC DATA (for testing only)
# ============================================================================

def prepare_synthetic_data():
    """Generate dummy data for pipeline testing."""
    print("Preparing synthetic data (for pipeline testing only)...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    samples_per_class = 100
    for class_name in EVENT_CLASSES:
        print(f"  Generating {samples_per_class} samples: {class_name}")
        for i in range(samples_per_class):
            frames = []
            for t in range(SEQ_LEN):
                keypoints = np.random.rand(NUM_KEYPOINTS, VALUES_PER_KEYPOINT).tolist()
                frames.append({"keypoints": keypoints, "timestamp": t / FPS})
            sample = {"frames": frames, "label": class_name, "duration": SEQ_LEN / FPS}
            with open(PROCESSED_DIR / f"{class_name}_{i:04d}.json", "w") as f:
                json.dump(sample, f)

    print(f"\nCreated {len(EVENT_CLASSES) * samples_per_class} synthetic samples in {PROCESSED_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare pose dataset")
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic"])
    args = parser.parse_args()

    if args.dataset == "synthetic":
        prepare_synthetic_data()

    print("\nTesting data loading...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=0)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    poses, labels = next(iter(train_loader))
    print(f"Batch shape: {poses.shape}")
    print(f"Labels: {[EVENT_CLASSES[l.item()] for l in labels[:5]]}")
    print("Done!")
