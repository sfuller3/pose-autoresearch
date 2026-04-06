"""
Data preparation and evaluation utilities.
DO NOT MODIFY THIS FILE (unless explicitly asked by human).

The agent should only edit train.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List

# ============================================================================
# CONSTANTS (fixed)
# ============================================================================

# Event classes
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

# Data format
NUM_KEYPOINTS = 17
VALUES_PER_KEYPOINT = 3  # x, y, confidence
INPUT_DIM = NUM_KEYPOINTS * VALUES_PER_KEYPOINT  # 51

# Temporal sequence
SEQ_LEN = 150  # 150 frames = 5 seconds at 30fps
FPS = 30

# Training
MAX_TIME_BUDGET_SECONDS = 300  # 5 minutes
TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"


# ============================================================================
# DATASET
# ============================================================================

class PoseDataset(Dataset):
    """
    Dataset for pose sequences with event labels.
    
    Each sample is:
    - poses: (seq_len, num_keypoints * 3) tensor of keypoint coordinates
    - label: int, class index
    
    File format (JSON):
    {
        "frames": [
            {
                "keypoints": [[x1, y1, c1], [x2, y2, c2], ..., [x17, y17, c17]],
                "timestamp": 0.033
            },
            ...
        ],
        "label": "fall",
        "duration": 1.0
    }
    """
    
    def __init__(self, data_dir: Path, seq_len: int = SEQ_LEN):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.samples = []
        
        # Load all JSON files
        for json_file in data_dir.glob("*.json"):
            with open(json_file) as f:
                data = json.load(f)
            
            # Extract pose sequence
            frames = data["frames"]
            keypoints = []
            
            for frame in frames[:seq_len]:  # Truncate to seq_len
                # Flatten keypoints: [[x,y,c], ...] → [x1,y1,c1, x2,y2,c2, ...]
                flat = np.array(frame["keypoints"]).flatten()
                keypoints.append(flat)
            
            # Pad if too short
            while len(keypoints) < seq_len:
                keypoints.append(np.zeros(INPUT_DIM))
            
            poses = np.array(keypoints, dtype=np.float32)  # (seq_len, 51)
            
            # Get label
            label_str = data["label"]
            label = CLASS_TO_IDX[label_str]
            
            self.samples.append((poses, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        poses, label = self.samples[idx]
        return torch.from_numpy(poses), torch.tensor(label, dtype=torch.long)


# ============================================================================
# DATA LOADING
# ============================================================================

def get_dataloaders(
    data_dir: Path = PROCESSED_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    split: Tuple[float, float, float] = TRAIN_VAL_TEST_SPLIT,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Load full dataset
    dataset = PoseDataset(data_dir)
    
    # Split
    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),  # Reproducible splits
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE.type == "cuda" else False,
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
    """
    Evaluate model on a dataset.
    
    Returns:
        accuracy: fraction of correct predictions
        loss: average cross-entropy loss
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for poses, labels in dataloader:
            poses = poses.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(poses)
            loss = criterion(logits, labels)
            
            # Compute accuracy
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss


def evaluate_per_class(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model per-class accuracy.
    
    Returns:
        dict mapping class name to accuracy
    """
    model.eval()
    
    class_correct = {cls: 0 for cls in EVENT_CLASSES}
    class_total = {cls: 0 for cls in EVENT_CLASSES}
    
    with torch.no_grad():
        for poses, labels in dataloader:
            poses = poses.to(device)
            labels = labels.to(device)
            
            logits = model(poses)
            preds = torch.argmax(logits, dim=1)
            
            for pred, label in zip(preds, labels):
                cls = EVENT_CLASSES[label.item()]
                class_total[cls] += 1
                if pred == label:
                    class_correct[cls] += 1
    
    per_class_acc = {
        cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        for cls in EVENT_CLASSES
    }
    
    return per_class_acc


# ============================================================================
# DATA PREPARATION (one-time setup)
# ============================================================================

def prepare_synthetic_data():
    """
    Download and prepare synthetic data from NTU RGB+D or Kinetics.
    
    This is a placeholder - you'll need to implement actual data download.
    For now, creates dummy data for testing.
    """
    
    print("Preparing synthetic data...")
    print("NOTE: This is dummy data for testing. Replace with actual dataset.")
    print()
    
    # Create directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy samples (100 per class)
    samples_per_class = 100
    
    for class_idx, class_name in enumerate(EVENT_CLASSES):
        print(f"Generating {samples_per_class} samples for class: {class_name}")
        
        for sample_idx in range(samples_per_class):
            # Generate random pose sequence
            frames = []
            for frame_idx in range(SEQ_LEN):
                # Random keypoints (normalized to [0, 1])
                keypoints = np.random.rand(NUM_KEYPOINTS, VALUES_PER_KEYPOINT).tolist()
                frames.append({
                    "keypoints": keypoints,
                    "timestamp": frame_idx / FPS,
                })
            
            # Create sample
            sample = {
                "frames": frames,
                "label": class_name,
                "duration": SEQ_LEN / FPS,
            }
            
            # Save to JSON
            filename = f"{class_name}_{sample_idx:04d}.json"
            with open(PROCESSED_DIR / filename, 'w') as f:
                json.dump(sample, f)
    
    print()
    print(f"Created {len(EVENT_CLASSES) * samples_per_class} samples")
    print(f"Saved to: {PROCESSED_DIR}")
    print()


# ============================================================================
# MAIN (one-time data preparation)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare pose dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "ntu", "kinetics"],
        help="Dataset to prepare",
    )
    args = parser.parse_args()
    
    if args.dataset == "synthetic":
        prepare_synthetic_data()
    elif args.dataset == "ntu":
        print("NTU RGB+D dataset preparation not yet implemented")
        print("Use --dataset synthetic for now")
    elif args.dataset == "kinetics":
        print("Kinetics-Skeleton dataset preparation not yet implemented")
        print("Use --dataset synthetic for now")
    
    # Test data loading
    print()
    print("Testing data loading...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    poses, labels = next(iter(train_loader))
    print()
    print(f"Sample batch:")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Label examples: {[EVENT_CLASSES[l.item()] for l in labels[:5]]}")
    print()
    print("Data preparation complete!")
