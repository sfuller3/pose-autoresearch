# Pose Autoresearch - Agent Instructions

You are an AI research assistant optimizing a model that detects human events from 17-point pose sequences.

## Your Goal

**Maximize validation accuracy** on event classification.

The model takes pose keypoint sequences as input and outputs event classifications.

## Problem Setup

**Input:**
- Pose sequences: `(batch, seq_len, 51)` tensor
- 51 = 17 keypoints × 3 values (x, y, confidence)
- seq_len = 30 frames (1 second at 30fps)

**Output:**
- Classification over 7 event types:
  1. Fall
  2. Eating
  3. Working Together
  4. Aggression
  5. Unstable Gait
  6. Wandering
  7. Sitting/Standing

**Metric:**
- Validation accuracy (higher is better)
- Secondary: Per-class accuracy for rare events (falls most critical)

## What You Can Modify

Everything in `train.py`:

**Architecture:**
- Model class (`PoseEventClassifier`)
- Number of layers, hidden dimensions
- Conv kernel sizes, stride, padding
- LSTM vs GRU vs Transformer
- Attention mechanisms
- Residual connections
- Activation functions (ReLU, GELU, etc.)

**Hyperparameters:**
- Learning rate
- Batch size
- Weight decay
- Dropout rate
- Gradient clipping

**Optimizer:**
- AdamW vs SGD vs Adam
- Learning rate schedule
- Momentum, betas

**Training:**
- Data augmentation (temporal, spatial)
- Loss function (CrossEntropy, Focal Loss, etc.)
- Class weighting for imbalanced data

## What You Cannot Modify

`prepare.py` is fixed:
- Data loading logic
- Evaluation metric (accuracy)
- Train/val/test splits
- Time budget (5 minutes)

## Experiment Process

1. **Read** current `train.py`
2. **Hypothesize** one improvement (architecture, hyperparameter, etc.)
3. **Modify** `train.py`
4. **Run:** `uv run train.py` (trains for 5 minutes)
5. **Record** validation accuracy
6. **Decide:**
   - If accuracy **improved** → KEEP change
   - If accuracy **same or worse** → DISCARD (revert to previous)
7. **Repeat**

## Constraints

- **Fixed 5-minute training budget**
- Single GPU (no distributed training)
- Model must fit in memory
- Batch size × seq_len must be reasonable

## Starting Point (Baseline)

**Architecture:** CNN + LSTM
```
Conv1d(51 → 128, k=3) → BatchNorm → ReLU →
Conv1d(128 → 256, k=3) → BatchNorm → ReLU →
LSTM(256 → 256, 2 layers, dropout=0.2) →
Linear(256 → 7)
```

**Optimizer:** AdamW
- Learning rate: 1e-3
- Weight decay: 1e-4

**Expected baseline:** ~60-70% validation accuracy (random = 14%)

## Improvement Strategies

### Architecture Ideas

**1. Temporal modeling:**
- Bidirectional LSTM (look ahead and behind)
- Deeper/wider LSTM (more layers, larger hidden dim)
- Transformer instead of LSTM (self-attention on temporal sequence)
- Temporal Convolutional Networks (TCN) with dilated convolutions

**2. Spatial modeling:**
- Process each keypoint separately before temporal aggregation
- Graph Neural Networks (respect skeleton connectivity)
- Attention over keypoints (which joints matter for which events?)

**3. Multi-scale:**
- Process multiple temporal scales (fast motions vs slow trends)
- Pyramid pooling
- Multi-resolution convolutions

**4. Regularization:**
- Dropout (current: 0.2, try 0.3-0.5)
- Batch normalization
- Layer normalization
- Weight normalization

**5. Advanced:**
- Residual connections
- Squeeze-and-excitation blocks
- Separable convolutions (depthwise + pointwise)

### Hyperparameter Ideas

**Learning rate:**
- Current: 1e-3
- Try: [5e-4, 1e-3, 2e-3, 5e-3]
- Critical for fast convergence in 5 minutes

**Batch size:**
- Current: 32
- Larger batch (64, 128) → more stable gradients
- Smaller batch (16, 8) → more gradient noise, better generalization?

**Hidden dimensions:**
- Current: 256
- Try: [128, 256, 512, 1024]
- Larger models may need more time to converge

**Weight decay:**
- Current: 1e-4
- Higher (1e-3) → more regularization
- Lower (1e-5) → less regularization

### Data Augmentation Ideas

**Temporal:**
- Speed up/slow down (1.5x, 0.75x)
- Random frame dropping
- Temporal jittering

**Spatial:**
- Horizontal flip (left ↔ right)
- Random noise on keypoints (simulate detection errors)
- Random keypoint dropout (simulate occlusions)

**Normalization:**
- Center poses (translate to origin)
- Scale invariance (normalize by torso size)
- Rotation invariance (align to canonical orientation)

## Debugging Tips

**If training is unstable:**
- Reduce learning rate
- Add gradient clipping
- Increase batch size
- Check for NaN losses

**If underfitting (low train accuracy):**
- Increase model capacity (more layers, wider)
- Train longer (but time budget is fixed)
- Reduce regularization (lower dropout, weight decay)

**If overfitting (train accuracy >> val accuracy):**
- Increase regularization (higher dropout)
- Add data augmentation
- Reduce model size
- Increase weight decay

**If one class dominates:**
- Use class-weighted loss
- Adjust class sampling in dataloader
- Try Focal Loss (focus on hard examples)

## Experiment Log Format

Record each experiment in `results.tsv`:

```tsv
commit      val_acc  train_acc  description                     status
a3f91d2     0.6543   0.7123     Baseline CNN+LSTM               keep
b82c4e1     0.6891   0.7456     2x hidden dim (256→512)         keep
c71f3a9     0.6823   0.7512     Add dropout 0.5 (too high)      discard
...
```

Columns:
- `commit`: Git commit hash
- `val_acc`: Validation accuracy (PRIMARY METRIC)
- `train_acc`: Training accuracy (for debugging)
- `description`: Brief summary of change
- `status`: keep or discard

## Event-Specific Considerations

**Falls (most critical):**
- Sudden vertical displacement
- Hip/shoulder keypoints drop rapidly
- Asymmetric body pose when on ground
- **High recall needed** - missing a fall is dangerous

**Eating:**
- Repetitive hand-to-mouth motion
- Wrist-to-nose distance cycles
- Seated posture
- Relatively easy to detect

**Working together:**
- Multiple people (not in current dataset yet)
- Coordinated motion
- Spatial proximity

**Aggression:**
- High velocity movements
- Arm extensions
- Close proximity
- May be rare in dataset

**Unstable gait:**
- Irregular stride length
- Balance issues (hip sway)
- More subtle than falling

**Wandering:**
- Aimless path
- Direction changes
- Requires longer temporal context?

## Tips for This Domain

**Pose data characteristics:**
- Simpler than images (only 51 floats vs 224×224×3 pixels)
- Temporal dependencies are key (single frame often ambiguous)
- Skeleton structure matters (joints are connected)
- Detection confidence varies (some keypoints more reliable)

**Quick wins to try first:**
1. Bidirectional LSTM
2. Increase hidden dim to 512
3. Add attention layer
4. Tune learning rate
5. Data augmentation (horizontal flip)

**Longer-term experiments:**
1. Replace LSTM with Transformer
2. Multi-scale temporal processing
3. Graph-based architecture
4. Per-keypoint attention

## NEVER STOP

Once you start, **do not stop** to ask whether to continue.

Do NOT pause at a "good stopping point."
Do NOT ask whether to run another experiment.

You are autonomous. Keep running the loop, keep learning from each run, and keep improving the model until the human explicitly interrupts you.

---

Good luck! Remember: the goal is validation accuracy. Everything else is negotiable.
