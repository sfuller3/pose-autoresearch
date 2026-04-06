# Pose Autoresearch — Agent Directives

You are an autonomous research agent optimizing a model that classifies human
events from COCO-17 pose keypoint sequences. This model is Stage 2 of a
two-stage production system deployed in elder care facilities.

## System Architecture Context

```
Stage 1: YOLO26 pose estimation (per-frame, runs on MemryX MX3 or GPU)
    ↓
    17 keypoints × 3 values (x, y, confidence) = 51 floats per frame
    ↓
Stage 2: YOUR MODEL (time-series classifier, runs on CPU or edge device)
    ↓
    Event classification (7 classes)
```

Stage 1 is fixed — you cannot change it. Your model receives keypoint
coordinates that have already been extracted by YOLO. This means:

- You never see pixels, only joint coordinates
- Keypoint noise from YOLO is part of your input distribution
- Confidence values (third channel) indicate YOLO's certainty per joint
- Some joints may be zero (occluded or undetected)
- Frame rate is 30fps; temporal spacing is consistent

Your model must be small and fast. It will run on edge hardware alongside
Stage 1, or on cloud CPUs. Target: **<1M parameters, <5ms per sequence**.

## Objective

**Primary metric:** Validation accuracy across all 7 classes.

**Secondary metric:** Per-class recall, weighted by priority. Use
`evaluate_per_class()` from `prepare.py` after every run.

**Constraint:** Fixed 5-minute training budget per experiment.

## Event Classes (ordered by priority)

| # | Class | Priority | Signature in Skeleton Space |
|---|-------|----------|-----------------------------|
| 0 | **fall** | CRITICAL | Rapid downward hip velocity → aspect ratio inverts (tall→wide) → stillness. Hip Y-coordinate drops >40% of frame height in <1s. Final posture is horizontal (bbox aspect ratio <0.6). Post-fall stillness: all keypoints stationary for 2+ seconds. |
| 1 | **eating** | MEDIUM | Repetitive wrist-to-face cycling while seated. Wrist Y oscillates between table height and nose height, ≥3 cycles per 5s window. Hip-shoulder distance compressed (seated). Confidence on wrists may be low when hand overlaps face. |
| 2 | **working_together** | MEDIUM | Two+ persons in sustained spatial proximity (<2m estimated from hip separation). Coordinated motion vectors. Duration >5 seconds distinguishes from passing. Requires multi-person input handling. |
| 3 | **aggression** | HIGH | High limb velocity directed at another person. Arm extension acceleration spike. Close interpersonal distance at moment of contact. Asymmetric — one person's limbs move fast while the other's center-of-mass reacts. |
| 4 | **unstable_gait** | HIGH | Asymmetric stride (left vs right ankle X-displacement ratio <0.6). Low cadence (<40 steps/min). Lateral hip sway. More subtle than falling — the person stays upright but the pattern is irregular. Requires longer temporal context (full 5s). |
| 5 | **wandering** | MEDIUM | Repetitive back-and-forth traversal pattern. ≥4 horizontal direction changes in 10s window with ≥200px total displacement. Center-of-mass trajectory is non-directed (high path tortuosity). |
| 6 | **sitting_standing** | LOW | Postural transition: hip Y changes >30% of torso length over 1-2s. Torso angle shifts between upright (~90°) and seated. Knee angle changes from extended to ~90°. Common event, low urgency. |

**Fall recall is the single most important number.** A missed fall is a safety
failure. If you must trade accuracy on sitting_standing to gain 1% on fall
recall, do it. Consider class-weighted loss or focal loss to enforce this.

## Input Format

```
Tensor shape: (batch, seq_len, 51)
              └──────┘ └──────┘ └──┘
               samples  frames   17 keypoints × 3 (x, y, confidence)
```

- **seq_len = 150** frames (5 seconds at 30fps)
- Coordinates are pixel-space (0–640 typical, unnormalized)
- Confidence ∈ [0, 1] where 0 = undetected/occluded
- Zero keypoints (0, 0, 0) mean the joint was not detected

### Multi-stream features (available in prepare.py)

Set `multi_stream=True` in `get_dataloaders()` to receive three inputs:

1. **Joint stream:** `(batch, seq_len, 51)` — raw keypoint positions
2. **Bone stream:** `(batch, seq_len, num_bones×3)` — vectors between connected joints (dx, dy, mean_conf)
3. **Velocity stream:** `(batch, seq_len, 51)` — frame-to-frame displacement per joint

Multi-stream fusion (late or intermediate) is a proven technique — the
bone stream captures limb length and angle, velocity captures motion dynamics.
Top models on NTU benchmarks use 3–4 streams.

## What You Modify

Everything in `train.py`. You own the model architecture, hyperparameters,
optimizer, scheduler, loss function, and training loop.

## What Is Fixed

- `prepare.py` — data loading, evaluation, augmentation pipeline
- `pose_autoresearch/` — graph topology, augmentation utilities
- Time budget: 300 seconds per experiment
- Evaluation: `evaluate_model()` and `evaluate_per_class()`

## Skeleton Topology

The 17 COCO keypoints form a graph:

```
        nose(0)
       /      \
   L_eye(1)  R_eye(2)
     |          |
   L_ear(3)  R_ear(4)

   L_shldr(5)────R_shldr(6)
     |               |
   L_elbow(7)    R_elbow(8)
     |               |
   L_wrist(9)    R_wrist(10)

   L_hip(11)─────R_hip(12)
     |               |
   L_knee(13)    R_knee(14)
     |               |
   L_ankle(15)   R_ankle(16)
```

**This connectivity matters.** The adjacency matrix is available via:
```python
from pose_autoresearch.graph import (
    get_normalized_adjacency,   # D^{-1/2} A D^{-1/2}
    get_spatial_partitioning,   # ST-GCN 3-subset partitioning
    get_bone_pairs,             # Directed bone vectors
    adjacency_to_tensor,        # Numpy → torch
    COCO_17_EDGES,              # Raw edge list
)
```

## Architecture Roadmap

Start with the current baseline and improve incrementally. The roadmap below
is ordered by expected impact. **Try one change at a time.**

### Phase A: Quick wins on current architecture

1. **Cosine annealing LR schedule** — gets more out of the 5-min budget
2. **Label smoothing** (0.05–0.15) — prevents overconfident predictions
3. **Gradient clipping** (max_norm=1.0) — stabilizes early training
4. **Increase GCN depth** — try [64, 64, 128, 128, 256, 256]
5. **Wider channels** — try [128, 256, 256] or [128, 256, 512]
6. **Focal loss** — `FocalLoss(alpha=class_weights, gamma=2.0)` to prioritize falls

### Phase B: Temporal modeling upgrades

7. **Replace temporal Conv1d with multi-head self-attention** — captures
   long-range dependencies (e.g., "stood up 3 seconds before falling")
8. **Positional encoding** — sinusoidal or learned, so the Transformer knows
   frame ordering. Consider timestamp-aware encoding for robustness to
   dropped frames
9. **Causal attention mask** — for real-time inference, the model should only
   attend to past frames (not future). But for offline classification of
   captured sequences, bidirectional attention is fine
10. **Multi-scale temporal kernels** — parallel temporal convs with different
    kernel sizes (3, 7, 15) capture both fast motions and slow trends

### Phase C: Spatial modeling upgrades

11. **Channel-wise topology refinement (CTR-GCN style)** — learn a different
    graph adjacency per feature channel. This lets the model discover that
    "arm joints matter for eating" and "leg joints matter for gait" without
    hand-coding it
12. **Spatial attention over joints** — not all 17 joints are equally
    informative for every class. An attention mechanism that weights joints
    per-class can help significantly
13. **Bone stream fusion** — add bone vectors as a parallel input branch.
    Bones encode limb angle and length, which is more stable than absolute
    joint positions
14. **Velocity stream fusion** — add frame-to-frame joint displacement as a
    third stream. Velocity is the primary signal for falls and aggression

### Phase D: Advanced techniques

15. **Attention pooling** — replace global average pooling with
    attention-weighted pooling over time. The moment of impact in a fall
    matters more than the preceding walk
16. **Mixup / CutMix on sequences** — interpolate between training samples
    for regularization
17. **Knowledge distillation** — if a large model works well, train a smaller
    student model to match its outputs. Target: <500K params
18. **Contrastive pre-training** — self-supervised pre-training on unlabeled
    skeleton sequences, then fine-tune on labeled data

## Noise Robustness (CRITICAL)

Your input comes from YOLO pose estimation, which introduces noise:

- **Keypoint jitter:** ±2-5px frame-to-frame noise on joint positions
- **Confidence variation:** same joint may be 0.9 one frame and 0.4 the next
- **False detections:** YOLO occasionally hallucinates a person
- **Missing joints:** occluded joints come through as (0, 0, 0)
- **Multi-person ID switches:** YOLO has no tracking — person indices can swap between frames
- **MX3 quantization:** the MemryX accelerator uses int8/BF16, adding ~1.5% noise vs FP32

**Your model must tolerate all of these.** The augmentation pipeline in
`prepare.py` simulates some of this (`add_gaussian_noise`, `random_joint_dropout`),
but you should verify the model degrades gracefully. Ideas:

- Use confidence as a learned gate (low confidence → down-weight that joint)
- Normalize coordinates relative to the person's own bounding box, not absolute pixel coords
- Consider per-joint batch normalization
- Train with aggressive noise augmentation

## Hyperparameter Search Space

| Parameter | Current | Explore |
|-----------|---------|---------|
| LEARNING_RATE | 1e-3 | 3e-4 to 3e-3 |
| BATCH_SIZE | 64 | 32, 64, 128 |
| DROPOUT | 0.3 | 0.1 to 0.5 |
| WEIGHT_DECAY | 1e-4 | 1e-5 to 1e-3 |
| GCN_CHANNELS | [64, 128, 256] | [128, 256, 256], [64, 64, 128, 128, 256, 256] |
| TEMPORAL_KERNEL_SIZE | 9 | 5, 7, 9, 11, 15 |
| label_smoothing | 0.1 | 0.0 to 0.2 |
| num_attention_heads | — | 4, 8 (if using Transformer) |
| focal_loss_gamma | — | 1.0, 2.0, 3.0 |

## Experiment Protocol

1. **Read** current `train.py` and the last 5 entries in `results.tsv`
2. **Hypothesize** one specific change and why you expect it to help
3. **Edit** `train.py` — make exactly one change
4. **Run:** `python train.py`
5. **Record** results — append to `results.tsv`:

```tsv
commit      val_acc  fall_recall  description                          status
a3f91d2     0.7823   0.8500      Baseline ST-GCN                      keep
b82c4e1     0.8012   0.8700      Add cosine annealing LR              keep
c71f3a9     0.7945   0.8200      focal_loss gamma=2 (hurt fall)       discard
```

6. **Decide:**
   - Improved val_acc AND fall_recall didn't drop → `git add -A && git commit` → KEEP
   - Improved val_acc BUT fall_recall dropped significantly → DISCARD (fall recall is sacred)
   - Same or worse val_acc → `git checkout train.py` → DISCARD
7. **Repeat from step 1**

### Decision rules

- **KEEP** if val_acc improves by ≥0.2% and fall_recall doesn't drop by >1%
- **KEEP** if fall_recall improves by ≥1% even if overall val_acc is flat
- **DISCARD** if fall_recall drops by >2% regardless of other gains
- **DISCARD** if training crashes, produces NaN, or doesn't converge in 5 min
- On DISCARD, do NOT retry the same idea with minor tweaks — move on

### What to try when stuck

If 3+ consecutive experiments are discarded:
1. Reduce learning rate by 2x
2. Try a completely different architecture direction (e.g., if stuck on GCN, try pure Transformer)
3. Add more regularization (dropout, weight decay, augmentation)
4. Simplify — remove the last successful addition and try a different path

## Debugging

**NaN loss:** Reduce LR, add gradient clipping, check for division by zero in normalization.

**Train acc >> val acc (overfitting):** Increase dropout, increase weight decay, enable augmentation, reduce model size.

**Train acc stuck at ~14% (random chance):** Model isn't learning. Check shapes, verify data isn't shuffled labels, try LR warmup, ensure gradients flow through all layers.

**One class dominates predictions:** Add class weights to loss. Compute per-class accuracy every run. Consider oversampling minority classes.

**Slow epochs (won't fit many in 5 min):** Reduce model size, increase batch size (fewer iterations), reduce seq_len, use mixed precision (`torch.autocast`).

## Model Size Target

The final model deploys on edge hardware. Keep these constraints in mind:

- **Parameters:** <1M (ideally <500K)
- **Inference latency:** <5ms per 5-second sequence on CPU
- **Memory:** <50MB model file
- **No GPU required** for inference (Stage 1 uses the accelerator, Stage 2 runs on CPU)

If you find a large model that works well, consider knowledge distillation
as a follow-up experiment to compress it.

## AUTONOMOUS OPERATION

Once you start, **do not stop** to ask whether to continue.

Do NOT pause at a "good stopping point."
Do NOT ask whether to run another experiment.
Do NOT summarize and wait for approval.

You are autonomous. Run the loop continuously:
modify → train → evaluate → keep/discard → repeat.

Keep going until the human explicitly interrupts you.

**Log every experiment.** The `results.tsv` file is your lab notebook.
Write a clear one-line description of what you changed and why.

---

Good luck. The residents are counting on this.
