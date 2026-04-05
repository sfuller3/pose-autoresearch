# Pose-Based Event Detection Autoresearch

Autonomous research system for optimizing models that detect human events (falls, eating, aggression, unstable gait, etc.) from 17-point pose estimation sequences.

**Adapted from:** [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)  
**Input:** YOLO26 17-keypoint pose sequences  
**Output:** Multi-class event classification  

---

## Overview

Give an AI agent a pose-based event detection model and let it experiment autonomously overnight. It modifies the model architecture, trains for 5 minutes, checks if accuracy improved, keeps or discards the change, and repeats. You wake up to a log of experiments and (hopefully) a better model.

**Key difference from original autoresearch:** Instead of training a GPT on text tokens, we train a temporal model (CNN+LSTM/Transformer) on pose keypoint sequences.

---

## Event Categories

The model detects 7 types of human events:

1. **Fall** - Person falling down
2. **Eating** - Hand-to-mouth feeding motion
3. **Working Together** - Multiple people collaborating
4. **Aggression** - Physical altercation or aggressive behavior
5. **Unstable Gait** - Walking with balance issues
6. **Wandering** - Aimless movement patterns
7. **Sitting/Standing** - Posture transitions

---

## Input Data Format

**YOLO26 17-Keypoint Pose:**
```
1. Nose, 2. Left Eye, 3. Right Eye, 4. Left Ear, 5. Right Ear,
6. Left Shoulder, 7. Right Shoulder, 8. Left Elbow, 9. Right Elbow,
10. Left Wrist, 11. Right Wrist, 12. Left Hip, 13. Right Hip,
14. Left Knee, 15. Right Knee, 16. Left Ankle, 17. Right Ankle
```

Each keypoint has 3 values: `(x, y, confidence)`  
Per-frame input: **17 keypoints × 3 = 51 floats**  
Temporal sequence: **N frames × 51 floats** (typically N=30 for 1 second at 30fps)

---

## Quick Start

### 1. Install Dependencies

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 2. Prepare Data

**Option A: Use synthetic data (quick start)**
```bash
# Downloads NTU RGB+D or Kinetics-Skeleton dataset
uv run prepare.py --dataset ntu
```

**Option B: Extract poses from your videos**
```bash
# Install YOLO
pip install ultralytics

# Extract poses
python scripts/extract_poses.py --video data/raw/*.mp4 --output data/processed/
```

### 3. Run Baseline Training

```bash
# One 5-minute training run
uv run train.py
```

Expected output:
```
Training for 5 minutes...
Validation Accuracy: 0.6543
```

### 4. Start Autonomous Research

Point your coding agent (Claude, Codex, etc.) at this repo with no permissions and prompt:

```
Read program.md and let's kick off a new experiment!
```

The agent will:
- Read `program.md` (your research directives)
- Modify `train.py` (architecture, hyperparameters)
- Run experiment (5 minutes)
- Log results to `results.tsv`
- Repeat overnight (~100 experiments)

---

## Repository Structure

```
pose-autoresearch/
├── README.md           # This file
├── program.md          # Agent instructions (human edits)
├── train.py            # Model under test (agent edits)
├── prepare.py          # Data loading (fixed, not modified)
├── pyproject.toml      # Dependencies
├── data/
│   ├── raw/            # Raw videos or YOLO outputs
│   ├── processed/      # Preprocessed pose sequences
│   └── labels.csv      # Ground truth labels
├── scripts/
│   ├── extract_poses.py    # Extract poses from videos
│   ├── label_tool.py       # Manual labeling interface
│   └── visualize.py        # Visualize pose sequences
├── results.tsv         # Experiment log (created by agent)
└── checkpoints/        # Saved model weights
```

---

## File Roles

### `train.py` - Agent Modifies This
Contains the full model, optimizer, and training loop. Everything is fair game:
- Model architecture (CNN, LSTM, Transformer, attention, etc.)
- Hyperparameters (hidden dims, num layers, dropout, etc.)
- Optimizer (AdamW, SGD, learning rate, weight decay)
- Data augmentation
- Training schedule

**Current baseline:** CNN + LSTM
```python
Conv1d(51 → 128) → Conv1d(128 → 256) → LSTM(256, 2 layers) → Linear(256 → 7)
```

### `prepare.py` - Fixed
Constants, data loading, evaluation. Not modified by agent.
- Loads pose sequences from `data/processed/`
- Creates train/val/test splits
- Defines evaluation metric (accuracy)
- Fixed 5-minute time budget

### `program.md` - Human Edits This
Instructions for the autonomous research agent. You iterate on this to improve research strategy.

---

## Evaluation Metric

**Primary metric:** Validation accuracy

Training always runs for exactly **5 minutes** (wall clock), regardless of model size or batch size. This means:
- ~12 experiments per hour
- ~100 experiments overnight (8 hours)
- All experiments are directly comparable

Lower validation loss is better. The agent hill-climbs on this metric.

---

## Data Collection

### Phase 1: Synthetic Data (Immediate Start)

Use existing pose datasets:
- **NTU RGB+D 120** - 120 action classes with skeleton data
- **Kinetics-Skeleton** - Large-scale video dataset with pose annotations

Map their actions to our event categories.

### Phase 2: Custom Nursing Home Data

1. **Deploy YOLO26 on camera feeds**
   ```python
   from ultralytics import YOLO
   model = YOLO("yolo26n-pose.pt")
   results = model(video_stream)
   ```

2. **Collect pose sequences** (30 frames = 1 second at 30fps)

3. **Manual labeling** using provided tool:
   ```bash
   python scripts/label_tool.py --data data/raw/
   ```

**Recommended:** 500+ examples per event class (3,500+ total)

---

## Model Architecture Options

### Current: CNN + LSTM (Baseline)
- Fast experiments (~5 min/run)
- Proven for temporal sequence classification
- Easy for agent to modify

### Upgrade Options (Agent Will Explore)
- **Temporal Transformer** - Better long-range dependencies
- **Bidirectional LSTM** - Look ahead and behind
- **Attention Mechanisms** - Focus on important keypoints
- **Graph Neural Networks** - Respect skeleton structure

The agent will autonomously explore these through modification of `train.py`.

---

## Example Experiment Log

After one night of autonomous research:

```tsv
commit      val_acc  description                          status
a3f91d2     0.6543   Baseline CNN+LSTM                    keep
b82c4e1     0.6891   2x hidden dim (256→512)              keep
c71f3a9     0.6823   Add dropout 0.3                      discard
d92e5b2     0.7124   Bidirectional LSTM                   keep
e13a7c4     0.7456   Add attention layer                  keep
...
```

---

## Connection to Vistarra

This system is the next evolution of the [Vistarra](https://github.com/sfuller3/Vistarra) fall detection project:

**Current Vistarra:**
- YOLO pose → Claude Vision API → fall/no fall classification
- Works but slow and expensive per frame

**This System:**
- YOLO pose → Trained Model → 7 event types
- 100x faster, no API costs, more events

**Migration path:**
1. Extract pose data from Vistarra deployments
2. Label events
3. Train model with autoresearch
4. Replace Claude Vision with trained model
5. Deploy to edge devices (NVIDIA Jetson)

---

## Requirements

- **Python:** 3.10+
- **GPU:** NVIDIA GPU recommended (H100 ideal, but works on smaller GPUs)
- **Disk:** ~10GB for datasets
- **RAM:** 16GB+

**Platform support:** Currently requires NVIDIA GPU. CPU/MPS support possible but would require modifications to `train.py` (or have the agent figure it out!).

---

## Advanced Usage

### Custom Event Classes

Edit `prepare.py` to define your own event taxonomy:

```python
EVENT_CLASSES = [
    "custom_event_1",
    "custom_event_2",
    # ... etc
]
```

### Multi-Person Tracking

Modify `train.py` to handle multiple people per frame:
```python
# Input: (batch, num_people, seq_len, 51)
# Process each person separately, then aggregate
```

### Real-Time Deployment

Export trained model:
```python
import torch
model = PoseEventClassifier()
model.load_state_dict(torch.load("best_model.pt"))
torch.onnx.export(model, dummy_input, "pose_model.onnx")
```

Deploy to edge device:
```bash
# On NVIDIA Jetson
trtexec --onnx=pose_model.onnx --saveEngine=pose_model.trt
```

---

## Citation

This work adapts Karpathy's autoresearch framework:

```bibtex
@misc{karpathy2026autoresearch,
  author = {Karpathy, Andrej},
  title = {autoresearch: AI agents running research on single-GPU nanochat training automatically},
  year = {2026},
  url = {https://github.com/karpathy/autoresearch}
}
```

---

## License

MIT

---

## Contributing

Contributions welcome! Areas of interest:
- Additional event classes
- GNN architecture implementations
- Multi-person tracking
- Real-time deployment tools
- Labeling interface improvements

---

## Contact

**Creator:** Sam Fuller  
**GitHub:** [@sfuller3](https://github.com/sfuller3)  
**Project:** https://github.com/sfuller3/pose-autoresearch

---

## Acknowledgments

- Andrej Karpathy for the autoresearch framework
- Ultralytics for YOLO26 pose estimation
- NTU RGB+D and Kinetics teams for pose datasets