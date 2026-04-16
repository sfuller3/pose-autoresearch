# Multi-Person Tracking & Per-Person Event Detection

## Overview

Expand the system from single-person to multi-person event detection. Each tracked person gets independent classification with nearest-neighbor context for interaction events (aggression, working_together).

## Architecture

### Tracking

`PoseExtractor` switches from `model.predict()` to `model.track(persist=True)`, returning `list[(track_id, keypoints_17x3)]` per frame. YOLO's built-in BoT-SORT tracker provides persistent IDs via IoU/motion matching.

`PersonTracker` manages per-person state:
- Dict of `{track_id: PersonState}` where `PersonState` holds a `KeypointBuffer`, `EventSmoother`, last-seen timestamp, and last-known position
- 5-second grace period for lost tracks. New unmatched tracks near a stale track's last position re-associate.
- After grace expiry, `PersonState` is dropped.
- `get_active_people()` returns tracked people with sufficient buffer depth for classification.

Fallback: `--no-tracking` flag uses `predict()` with synthetic ID 0 for single-person mode.

### Model Input Expansion

Per-person input per frame (105 dims raw):
- Primary person: 51 dims (17 joints x 3)
- Nearest neighbor: 51 dims (closest other tracked person, or zeros if alone)
- Neighbor metadata: 3 dims (euclidean distance normalized by frame diagonal, relative x, relative y)

Nearest neighbor keypoints are multiplied by a soft distance decay: `sigmoid(5 - dist_norm * 10)`. People within ~30% frame diagonal contribute strongly; beyond ~60% fades to near-zero.

After bone/velocity expansion in forward pass:
- Primary: joints 51 + bones 48 + velocity 51 = 150
- Neighbor: joints 51 + bones 48 + velocity 51 = 150
- Metadata: 3
- **Total: 303 dims** into temporal CNN

Confidence gating applies independently to both person streams. Neighbor gating stacks with distance decay.

Constructor parameter `n_bodies=2` (set to 1 for single-body backward compat, input stays 150 dims).

### Training Data

New `MultiPersonPoseDataset` in `train.py`:
- Reads `bodies` key from JSON for multi-body samples, `keypoints` for single-body
- Multi-body: body 0 = primary, body 1 = neighbor. Computes 3 metadata dims from hip midpoints.
- **Pair flipping:** each 2-body sample produces 2 training examples (body 0 as primary, body 1 as primary) — doubles interaction training data
- Single-body: neighbor slots zero-padded
- Output: `(B, T, 105)` tensors

### Streaming Pipeline

Per-frame loop:
1. `PoseExtractor.extract()` → list of (track_id, kps)
2. `PersonTracker.update(detections, timestamp)` → updates buffers, runs grace period
3. For each active person with sufficient history:
   - Find nearest neighbor from other active tracks
   - Compute distance decay + metadata
   - Build 105-dim input from person's buffer + neighbor's buffer
   - Run classifier → per-person probabilities
   - Run context rules + per-person `EventSmoother`
4. Events include track_id in alerts/logs

### HUD & Alerting

- Per-person colored skeleton + classification label overlay with track ID
- Event log includes person ID: "Person 3 — fall"
- Clip recording captures full scene; metadata tags which person triggered

### Backward Compatibility

- `n_bodies=1` parameter preserves single-body model behavior (150-dim input)
- `prepare.py` untouched — fallback `get_dataloaders()` path still works
- `--no-tracking` CLI flag forces single-person mode
- Checkpoints incompatible with prior model (input dim change) — full retrain required

## Files Modified/Created

| File | Change |
|------|--------|
| `train.py` | `MultiPersonPoseDataset`, expand `PoseEventClassifier` input, `FULL_INPUT_DIM` 150→303 |
| `stream_detect.py` | `PoseExtractor` tracking, new `PersonTracker` class, per-person classification loop, HUD updates |
| `tests/test_pipeline.py` | Tests for multi-person input, pair flipping, distance decay, tracker grace period |
| `scripts/convert_ntu120.py` | Already done — stores both bodies |
| `prepare.py` | No changes |
