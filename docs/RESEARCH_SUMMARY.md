# Pose Autoresearch — Experiment Summary & Next Steps

**Date:** 2026-04-08
**Best Model:** Temporal 1D CNN (commit `606713d`)
**Best Val Accuracy:** 97.44% | **Fall Recall:** 97.63%

---

## Project Context

Autonomous research system for classifying human events from YOLO pose estimation sequences. Deployed in elder care facilities as Stage 2 of a two-stage pipeline (Stage 1: YOLO pose extraction on MemryX MX3, Stage 2: this model on CPU/edge). Classifies 5-second windows of 17 COCO keypoints into 7 event types: fall, eating, working_together, aggression, unstable_gait, wandering, sitting_standing.

## Dataset

- **5,201 samples** across 2 classes (fall: 3,087 / sitting_standing: 2,114)
- Sources: FallVision and Le2i video datasets
- Format: 150-frame sequences (5s at 30fps), 51 features per frame (17 keypoints x 3)
- **5 of 7 target classes have zero labeled data**

## Experiment History (16 experiments)

### Kept (3 experiments — monotonic improvement)

| # | Commit | Val Acc | Fall Recall | Change |
|---|--------|---------|-------------|--------|
| 1 | `b2930da` | 89.74% | 94.84% | Baseline ST-GCN [64,128,256] |
| 2 | `09fd115` | 97.05% | 97.20% | **Replace ST-GCN with temporal CNN** [128,128,256,256] |
| 3 | `606713d` | 97.44% | 97.63% | Learning rate 1e-3 -> 2e-3 |

### Discarded (13 experiments)

| # | Val Acc | Fall Recall | What Failed | Why |
|---|---------|-------------|-------------|-----|
| exp1 | 91.92% | 90.97% | Smaller model [64,128] bs=128 | Underfitting |
| exp2 | 88.72% | 92.90% | bs=128 + fall weight=2.0 | Fewer epochs in budget |
| exp5 | 96.92% | 97.42% | lr=3e-3 | Too aggressive |
| exp6 | 97.05% | 96.56% | dropout=0.2 | Fall recall dropped |
| exp7 | 97.44% | 95.91% | 5 temporal blocks | Fall recall dropped significantly |
| exp8 | 96.67% | 96.34% | weight_decay=5e-4 + fall_weight=2.0 | Both worse |
| exp9 | 96.03% | 96.77% | Inline velocity features | Overfitting |
| exp10 | 97.31% | 96.13% | T_max=100 cosine annealing | Slightly worse |
| exp11 | 96.67% | 96.99% | label_smoothing=0.05 | Worse |
| exp12 | 97.05% | 96.77% | Max pooling | Overfitting |
| exp13 | 97.31% | 96.99% | Mean+max pooling | Marginal, fall recall down |
| exp14 | 97.44% | 96.13% | Hidden FC head | Fall recall dropped |
| exp15 | 95.64% | — | Multi-scale temporal conv | Much worse |
| exp16 | 97.18% | 96.77% | batch_size=32 | Worse |

## Key Findings

1. **Flat temporal CNN >> Graph-based ST-GCN.** Switching from ST-GCN to a 1D temporal CNN yielded the biggest single gain: +7.3% val accuracy, +2.4% fall recall. With only 2 classes and limited data, the graph structure added complexity without benefit.

2. **The model is near the ceiling for 2-class classification.** At 97.44% accuracy, further gains from architecture tweaks are diminishing — 13 consecutive experiments failed to improve on the best.

3. **Fall recall is fragile.** Many changes that maintained or improved accuracy caused fall recall to drop (exp6, exp7, exp10, exp14). The 1.5x class weight + label smoothing 0.1 + lr=2e-3 combo is a stable sweet spot.

4. **Velocity features hurt when inline.** exp9 tried adding velocity features directly — it overfitted. The multi-stream approach in `prepare.py` (separate bone/velocity streams) may work better but hasn't been tested with the current CNN architecture.

5. **Model is well within constraints.** ~730K parameters (target <1M), lightweight enough for edge deployment.

## Current Architecture

```
Input: (batch, 150, 51) — 5-second pose sequences
    ↓
4 Temporal Blocks (Conv1d + BN + ReLU + Residual)
    Channels: 51 → 128 → 128 → 256 → 256
    Kernel size: 9, stride 1, padding 4
    ↓
Global Average Pooling
    ↓
Dropout(0.3) → Linear(256 → 7)
```

**Training:** AdamW, lr=2e-3, weight_decay=1e-4, CosineAnnealingLR (T_max=50), CrossEntropyLoss (fall weight=1.5, label_smoothing=0.1), grad clip=1.0, 300s budget.

---

## Next Steps (Priority Order)

### 1. DATA: Expand to all 7 classes (HIGHEST IMPACT)

The model plateau is almost certainly data-limited, not architecture-limited. 97.44% on a 2-class problem doesn't tell us how the model will perform on the real 7-class task.

- **NTU RGB+D 120 integration:** A class mapping and conversion pipeline are planned in the world-class roadmap (`data/ntu120/class_mapping.json`). NTU-120 has skeleton data for actions that map to eating, aggression, unstable_gait, and falling.
- **Label remaining classes:** The raw video data exists in `data/raw/`; pose sequences need to be labeled for eating, aggression, unstable_gait, wandering, and working_together.
- **Class imbalance strategy:** With 7 classes of varying frequency, weighted sampling or focal loss will become critical.

### 2. ARCHITECTURE: Multi-stream fusion

The infrastructure exists (`prepare.py` supports `multi_stream=True`) but hasn't been properly tested. With more classes, bone vectors (limb angles/lengths) and velocity (motion dynamics) will carry discriminative signal that raw joint positions miss.

- Test late fusion: three parallel CNN branches → concatenate → classify
- Start with joint+velocity (velocity is key for fall vs. unstable_gait distinction)

### 3. ARCHITECTURE: Attention mechanisms

For the 7-class problem, different temporal moments matter for different events:
- Falls: the moment of impact
- Eating: repetitive wrist-to-face cycling
- Wandering: direction-change pattern over full 5s

Attention pooling (replacing global average pooling) or lightweight self-attention blocks could help the model focus on class-relevant temporal windows.

### 4. ARCHITECTURE: Spatial graph structure (revisit)

ST-GCN failed on the 2-class problem, but graph structure may matter for the 7-class problem where spatial relationships between joints are discriminative (e.g., wrist-to-face for eating, hip sway for unstable gait). Consider CTR-GCN-style channel-wise topology refinement once more classes are available.

### 5. ROBUSTNESS: Noise and edge deployment testing

- Test with simulated YOLO noise (keypoint jitter, missing joints, confidence variation)
- Measure inference latency on CPU (<5ms target)
- Test with MemryX MX3 quantization noise (~1.5% added noise)

### 6. SCALE: Multi-person support

Working_together and aggression require reasoning about 2+ people. The current pipeline assumes single-person input. This is a significant engineering effort involving person tracking and multi-person input handling.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Only 2/7 classes have data | **Critical** | NTU-120 integration + labeling pipeline |
| 97% accuracy may not transfer to 7-class | High | Expect significant drop; budget for re-optimization |
| Multi-person events unsupported | High | Requires tracking layer + architecture changes |
| Edge latency untested | Medium | Profile on target hardware early |
| YOLO noise robustness unvalidated | Medium | Add noise injection to eval pipeline |

## Recommendation

**Stop architecture hill-climbing on the 2-class problem.** The model is near ceiling. The highest-ROI next step is expanding training data to all 7 classes (via NTU-120 mapping and/or manual labeling), then resuming the autoresearch loop on the full classification task. The current temporal CNN architecture is a strong starting point for the 7-class problem.
