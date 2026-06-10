# CTR-GCN Skeleton Backbone (Side-by-Side with Temporal CNN)

## Overview

Add a CTR-GCN-style graph backbone (`STGCNClassifier`) as an alternative to the
flattened temporal CNN (`PoseEventClassifier`), selected by environment variable.
Both train on identical data; the winner becomes the production checkpoint. The
current 96.15%-test CNN stays intact throughout.

Motivation: the CNN flattens keypoints, discarding skeleton topology. CTR-GCN
learns dynamic joint-to-joint relationships per channel group — including
*cross-person* relationships in a unified two-body graph — directly targeting
the dominant aggression↔working_together confusion (64 test errors).

## Architecture

### Two-body graph (34 joints)

- Joints 0-16: primary person (COCO-17), joints 17-33: neighbor.
- Intra-body edges: `COCO_17_EDGES` from `pose_autoresearch/graph.py`, duplicated
  with +17 offset for the neighbor.
- Cross-body seed edges (6): hip↔hip (11↔28, 12↔29), wrist↔wrist (9↔26, 10↔27),
  primary-wrist↔neighbor-nose (9↔17, 10↔17).
- Base adjacency: symmetric-normalized 34×34, built by a new
  `get_two_body_adjacency()` in `pose_autoresearch/graph.py`.
- Single-person samples: neighbor joints all-zero (matches training data format);
  cross-body edges then propagate zeros, which the network learns to ignore.

### CTRGraphBlock (spatial unit)

- Input/output `(B, C, T, V=34)`.
- 8 channel groups. Per group: a bottleneck (`Conv 1x1` → temporal mean →
  pairwise subtraction → `tanh`) produces a `(B, V, V)` affinity added to the
  shared base adjacency with a learned per-group scale `alpha` (init 0 →
  identity-start, training begins as plain GCN).
- Graph conv per group, concat groups, `Conv 1x1` mix, BatchNorm, residual, SiLU.

### STGCNClassifier

Input contract unchanged: `(B, T, 105)` = primary(51) + neighbor(51) + metadata(3).

forward():
1. Split; confidence-gate both bodies (`sigmoid(conf*5-2)` on xy).
2. Distance decay `sigmoid(5 - dist_norm*10)` scales neighbor features.
3. Reshape to `(B, 3, T, 34)`; append metadata broadcast per joint → `(B, 6, T, 34)`.
4. BatchNorm2d input norm.
5. 4 stages, channels 64→64→128→256, temporal stride 2 at stages 3 and 4.
   Each stage = CTRGraphBlock → multi-scale temporal conv (kernels 3/7/15,
   SE attention, residual) applied per joint.
6. Optional FiLM conditioning after each stage (same `EnvironmentConditioner`,
   gamma/beta broadcast over T and V) — parity with CNN.
7. Joint-mean pool → `TemporalAttentionPool` → `Linear(256, 7)`.

Parameter budget: 1.6-2.0M (comparable to CNN's 1.94M).

### Backbone selection

- `POSE_BACKBONE=gcn python train.py` → STGCNClassifier.
- Default (unset or `cnn`) → existing PoseEventClassifier. No flag churn in
  prepare.py (immutable) or the dataloaders.
- Checkpoint saved to `checkpoints/best_model_gcn.pt` when backbone=gcn so the
  CNN checkpoint is never clobbered.
- `scripts/audit_training_data.py` gains `--backbone {cnn,gcn}` for confusion
  matrices of either model.

## Comparison protocol (Thunder A100)

1. `POSE_AUTORESEARCH_MAX_TIME=3600 python train.py` (CNN control, T_max=260).
2. `POSE_AUTORESEARCH_MAX_TIME=3600 POSE_BACKBONE=gcn python train.py`.
3. Compare test accuracy + per-class, with fall recall as the tiebreak veto:
   the winner must not drop fall below the CNN's 98.37%.
4. Winner's checkpoint becomes production; loser stays in experiments/.

## Files

| File | Change |
|------|--------|
| `pose_autoresearch/graph.py` | Add `get_two_body_adjacency()`, `TWO_BODY_CROSS_EDGES` |
| `train.py` | Add `CTRGraphBlock`, `GraphTemporalBlock`, `STGCNClassifier`; backbone select in `main()` |
| `stream_detect.py` | `--backbone gcn` flag mapping checkpoint + model class |
| `tests/test_pipeline.py` | New `TestTwoBodyGraph`, `TestCTRGCN` suites |
| `prepare.py` | No changes (immutable) |

## Testing

- `get_two_body_adjacency()`: shape (34,34), symmetric, row-normalized, contains
  intra-body and cross-body edges.
- CTRGraphBlock: output shape, identity-start (alpha=0 ⇒ matches plain GCN
  behavior), gradient flow to affinity bottleneck.
- STGCNClassifier: forward `(B,150,105)`→`(B,7)`; n_bodies semantics (zero
  neighbor → valid output, no NaN); distance decay attenuation; FiLM parity;
  param count under 2.5M; backward pass.
- Backbone selection: env var picks the right class; CNN default unchanged
  (existing 70 tests must still pass).

## Risks

- GCN may underperform on FallVision/Le2i single-person data (YOLO noise) —
  confidence gating mitigates; comparison protocol catches it.
- 2D-only keypoints limit some CTR-GCN gains reported on 3D NTU data; we
  compare against our own CNN baseline, not paper numbers.
- Slower per-epoch than CNN (graph ops): expect ~2x epoch time; still ~100+
  epochs in a 1-hour budget, enough for T_max=260 to partially anneal — if
  epochs fall short, rerun with longer budget before judging.
