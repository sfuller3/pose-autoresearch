"""
Pose-based event detection training script.
The agent modifies this file to improve validation accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from pathlib import Path

# Import from prepare.py (fixed utilities)
from prepare import (
    PoseDataset,
    get_dataloaders,
    evaluate_model,
    DEVICE,
    EVENT_CLASSES,
    MAX_TIME_BUDGET_SECONDS,
)

# ============================================================================
# HYPERPARAMETERS (agent can modify)
# ============================================================================

INPUT_DIM = 51  # 17 keypoints × 3 (x, y, confidence)
HIDDEN_DIM = 256
NUM_LSTM_LAYERS = 2
NUM_CLASSES = len(EVENT_CLASSES)  # 7 events
SEQ_LEN = 30  # 30 frames = 1 second at 30fps
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

# ============================================================================
# MODEL ARCHITECTURE (agent can modify)
# ============================================================================

class PoseEventClassifier(nn.Module):
    """
    Baseline: CNN + LSTM architecture for temporal pose sequence classification.
    
    Architecture:
    1. Temporal 1D convolutions to extract local patterns
    2. LSTM to capture long-range temporal dependencies
    3. Linear classification head
    
    Agent: Feel free to completely redesign this! Try:
    - Deeper/wider networks
    - Transformers instead of LSTM
    - Attention mechanisms
    - Residual connections
    - Different activation functions
    - Multi-scale temporal convolutions
    """
    
    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LSTM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Temporal convolutions
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) - pose sequence
        
        Returns:
            logits: (batch, num_classes) - event classification scores
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Temporal convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Reshape back for LSTM: (batch, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for classification
        x = h_n[-1]  # (batch, hidden_dim)
        
        # Classification
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits


# ============================================================================
# TRAINING LOOP (agent can modify)
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (poses, labels) in enumerate(dataloader):
        poses = poses.to(device)
        labels = labels.to(device)
        
        # Forward pass
        logits = model(poses)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Main training loop with fixed 5-minute time budget.
    
    The agent can modify:
    - Model architecture (class above)
    - Hyperparameters (top of file)
    - Optimizer
    - Learning rate schedule
    - Data augmentation (in prepare.py or here)
    
    The agent should NOT modify:
    - Time budget (MAX_TIME_BUDGET_SECONDS)
    - Evaluation logic (in prepare.py)
    - Data loading logic (in prepare.py)
    """
    
    print("=" * 70)
    print("POSE AUTORESEARCH - Training Run")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Model: PoseEventClassifier")
    print(f"Hidden Dim: {HIDDEN_DIM}")
    print(f"LSTM Layers: {NUM_LSTM_LAYERS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Time Budget: {MAX_TIME_BUDGET_SECONDS}s")
    print("=" * 70)
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=4,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()
    
    # Initialize model
    model = PoseEventClassifier(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LSTM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Optimizer (agent can modify)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with time budget
    start_time = time.time()
    epoch = 0
    best_val_acc = 0.0
    
    print("Starting training...")
    print()
    
    while True:
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed >= MAX_TIME_BUDGET_SECONDS:
            print(f"Time budget reached: {elapsed:.1f}s")
            break
        
        epoch += 1
        epoch_start = time.time()
        
        # Train one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        
        epoch_time = time.time() - epoch_start
        
        # Evaluate on validation set
        val_acc, val_loss = evaluate_model(model, val_loader, DEVICE)
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoint
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                },
                'checkpoints/best_model.pt'
            )
        
        # Print progress
        print(f"Epoch {epoch:3d} | "
              f"Time: {epoch_time:5.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Best: {best_val_acc:.4f}")
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    # Final evaluation on test set
    model.load_state_dict(
        torch.load('checkpoints/best_model.pt')['model_state_dict']
    )
    test_acc, test_loss = evaluate_model(model, test_loader, DEVICE)
    
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print()
    
    # Return final metric (for agent to hill-climb on)
    return best_val_acc


if __name__ == "__main__":
    # Create checkpoint directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Run training
    val_accuracy = main()
    
    print()
    print(f"FINAL VALIDATION ACCURACY: {val_accuracy:.4f}")
