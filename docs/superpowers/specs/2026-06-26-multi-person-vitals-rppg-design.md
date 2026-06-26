# Multi-Person Contactless Vital Signs (rPPG)

## Overview

Add contactless pulse rate (HR) and respiration rate (RR) estimation for every
tracked person in view, integrated into the live `stream_detect.py` pipeline.
Based on the rPPG method in Kolosov et al., *Contactless Camera-Based Heart Rate
and Respiratory Rate Monitoring Using AI on Hardware* (Sensors 2023, 23, 4550),
adapted from single-person to multi-person and from MediaPipe face models to the
YOLO pose keypoints we already track.

Decisions locked during brainstorming:
- **ROI source:** reuse YOLO COCO-17 face keypoints (no second face model).
- **Signal method:** rPPG only (GREEN-channel chain; no EVM).
- **Integration:** into `stream_detect.py`, signal chain isolated in `vitals.py`.
- **Control mechanism:** master enable flag, per-frame people cap, estimation
  cadence throttle, and a signal-quality reporting gate.
- **Liveness/occupancy signal:** a fused per-person state machine
  (`EMPTY → PRESENT_STATIC → PRESENT_MOVING → LIVE_CONFIRMED`) combining a
  keypoint-motion activity metric with rPPG pulse confirmation.

This is additive. With `--vitals` off (the default), the pipeline behaves
exactly as today — event detection is untouched.

## Why deviate from the paper

The paper targets a single subject (largest face) via BlazeFace + FaceMesh. Two
changes are required and two are improvements:

1. **Multi-person:** we already track N people with BoT-SORT and hold per-person
   `PersonState`. Reusing YOLO keypoints for ROIs gives multi-person vitals for
   free and avoids a second CNN whose cost would multiply per head (the paper
   measured FaceMesh roughly halving throughput for one subject).
2. **ROI from 5 keypoints** instead of 468 landmarks — coarser, but forehead and
   cheeks are geometrically predictable from nose/eyes/ears.
3. **Longer, separate windows (improvement):** the paper's 180-frame (6 s) buffer
   yields FFT bins of `fs/N = 30/180 = 0.167 Hz`. That is ~10 bpm for HR and ~10
   breaths/min for RR — unusably coarse for RR (range 11–30). We use a longer RR
   window and parabolic interpolation for sub-bin precision.
4. **Quality gating (improvement):** the paper always emits a reading; we gate on
   in-band SNR, buffer fullness, track stability, and head motion so we report
   only trustworthy values.

## Architecture

### `vitals.py` (new module — isolated signal chain)

**`FaceROIExtractor`**
- Input: COCO-17 keypoints `(17, 3)` for one person + the BGR frame.
- COCO indices: nose 0, left eye 1, right eye 2, left ear 3, right ear 4.
- Geometry: inter-ocular distance `d = ||eye_l - eye_r||`; eye-line roll angle
  `theta = atan2(dy, dx)`; eye midpoint `m`.
  - **Forehead:** rotated box centered above `m`, vertical extent roughly
    `[m_y - 0.9d, m_y - 0.25d]`, width `~1.3d`.
  - **Cheeks:** two rotated boxes below the eyes, lateral of the nose, each
    `~0.6d` square, centered near `(eye + nose)/2` offset outward.
- Validity gate: all required keypoints' confidence `>= KP_CONF_MIN` (default
  0.4); reject near-profile faces when `ear`/`eye` geometry implies yaw beyond a
  threshold or `d` is too small (face too far). Returns `None` when invalid.
- Output: list of ROI pixel masks/boxes (clipped to frame bounds) + a validity
  flag. Behind a clean interface so a FaceMesh-based extractor can be swapped in
  later without touching callers.

**`RPPGEstimator` (stateless compute)**
- `green_mean(frame, rois) -> float`: mean of the green channel (BGR index 1)
  over the union of valid ROI pixels.
- `estimate(timestamps, values, band) -> (freq_hz, quality)`:
  1. Detrend (remove DC + linear drift from lighting).
  2. Resample to even spacing via 1-D interpolation onto a uniform time grid at
     the nominal fps (handles dropped/variable frames).
  3. Hamming window.
  4. L2-normalize.
  5. Real FFT with zero-padding to `>= 4x` length for finer bins.
  6. Peak-pick the max amplitude bin within `band`, then parabolic interpolation
     across its two neighbors for a sub-bin frequency estimate.
  7. Quality = in-band SNR: peak power / mean in-band power (excluding the peak
     neighborhood).
- Bands: HR `0.83–3.0 Hz` (50–180 bpm), RR `0.18–0.5 Hz` (11–30 br/min).
- HR uses `HR_WINDOW` frames (default 256, ~8.5 s @30fps); RR uses `RR_WINDOW`
  frames (default 512, ~17 s @30fps). Buffers size to `max(HR_WINDOW, RR_WINDOW)`.
- Convert: `bpm = freq_hz * 60`.

### `PersonState` additions (in `stream_detect.py`)

- `vitals_buffer`: `collections.deque[(timestamp, green_mean)]`,
  `maxlen = max(HR_WINDOW, RR_WINDOW)`.
- `hr_ema`, `rr_ema`: smoothed estimates (EMA alpha default 0.3).
- `hr_quality`, `rr_quality`: last quality scores.
- `last_vitals_frame`: frame index of last estimation (for cadence throttle).
- `head_motion`: rolling variance of eye-midpoint position over the window
  (rPPG motion gate; a face-region subset of whole-body activity).
- `activity_ema`: smoothed whole-body keypoint-motion metric (continuous
  activity level).
- `liveness`: a `LivenessMonitor` instance (per-person state + timers).

### `VitalsController` (the control mechanism)

Config object (constructed from CLI args) with:
- `enabled: bool` — master switch (`--vitals`).
- `max_people: int` — per-frame cap; when more tracks are active, process the
  `max_people` largest faces by ROI area (`--vitals-max-people`, default 4).
- `cadence: int` — recompute vitals every `cadence` frames per person
  (`--vitals-cadence`, default 15 ≈ 0.5 s @30fps).
- `quality_min: float` — minimum SNR to report (`--vitals-quality-min`, default
  tuned during validation; placeholder 2.0).
- `motion_max: float` — maximum head-motion variance to report.
- `abnormal ranges` — HR `[40, 130]`, RR `[8, 25]`; outside → alert.
- **Liveness params:** `move_threshold`, `move_hold_s` (3 s), `pulse_hold_s`
  (10 s), `unresponsive_s` (30 s), `unresponsive_alert: bool` (default False,
  `--vitals-unresponsive-alert`).

`should_estimate(state, frame_idx)` and `should_report(state)` centralize the
gating so the pipeline loop stays readable.

### Pipeline integration (`run_pipeline` loop)

Inside the existing per-active-person loop, when `controller.enabled`:
1. `FaceROIExtractor.extract(kps, frame)` → ROIs or `None`. If `None`, push a
   gap marker (skip) and continue; the resampler tolerates gaps.
2. Append `(timestamp, green_mean)` to `state.vitals_buffer`.
3. If `controller.should_estimate(state, frame_idx)` and buffer full:
   - HR = `RPPGEstimator.estimate(..., HR_BAND)` over the last `HR_WINDOW`.
   - RR = `RPPGEstimator.estimate(..., RR_BAND)` over the last `RR_WINDOW`.
   - Update EMAs and quality; update `head_motion`.
4. If `controller.should_report(state)`: HUD shows values; log to
   `events/vitals_log.jsonl`; abnormal → `AlertDispatcher`. Else HUD shows
   "acquiring".

People beyond `max_people` still track and classify events; they just skip the
rPPG compute that frame.

### Liveness & activity signal (fused state machine)

A per-person signal answering "is a live person present, and are they active?"
It fuses an instant keypoint-motion read with the slower rPPG pulse confirmation,
so it degrades gracefully: movement-based occupancy works even when the face is
not visible or `--vitals` cannot confirm a pulse, and pulse upgrades it to
physiological confirmation (anti-spoof: a photo/mannequin never reaches
`LIVE_CONFIRMED`).

**`ActivityEstimator` (`vitals.py`, stateless helper)**
- `frame_activity(prev_kps, cur_kps) -> float`: mean over confident joints of
  `||p_t - p_{t-1}||`, normalized by a body-scale reference (shoulder width, or
  bbox diagonal fallback) so the metric is invariant to distance from the camera.
- The pipeline maintains a per-person `activity_ema` (EMA alpha default 0.3) over
  this metric — the continuous "activity level" shown on the HUD/log.

**`LivenessMonitor` (`vitals.py`, per-person instance in `PersonState`)**
Holds `last_move_time`, `last_pulse_time`, and the current state. `update(now,
is_tracked, activity, pulse_quality) -> LivenessState` applies, strongest
evidence first:
- `LIVE_CONFIRMED` — a valid pulse (`pulse_quality >= quality_min`) was seen
  within `pulse_hold_s` (default 10 s). Pulse latches over brief stillness.
- `PRESENT_MOVING` — no recent pulse, but `activity > move_threshold` within
  `move_hold_s` (default 3 s).
- `PRESENT_STATIC` — tracked, but neither recent pulse nor recent movement.
- `EMPTY` — no active track (grace period expired).

`states` is an `enum.IntEnum` so ordering ("highest achieved") is explicit.

**Unresponsiveness branch (opt-in):** a track held in `PRESENT_STATIC`
continuously for `unresponsive_s` (default 30 s) with no pulse raises a
"possible unresponsiveness" alert via `AlertDispatcher`. Default **off**
(`--vitals-unresponsive-alert`) because a still, face-occluded sleeping resident
can sit in `PRESENT_STATIC` legitimately; thresholds need site tuning before this
is trusted. The liveness *state* is always produced; only the *alert* is gated.

**Pipeline:** after the rPPG step in the per-person loop, compute
`frame_activity` from the keypoint buffer, update `activity_ema`, then
`state.liveness.update(...)`. Runs whenever the track is active — independent of
the `max_people` rPPG cap (activity is cheap), so every tracked person always has
at least a movement-based liveness state; only `LIVE_CONFIRMED` needs rPPG.

### Output

- **HUD:** per person, `HR 72 / RR 16` near the head, colored by quality
  (gray = acquiring, green = confident), plus a liveness badge (state name +
  color: EMPTY gray, STATIC amber, MOVING blue, LIVE green) and an activity bar.
  Reuses the existing overlay drawing.
- **Log:** `events/vitals_log.jsonl`, one record per report:
  `{track_id, timestamp, hr_bpm, rr_bpm, hr_quality, rr_quality, liveness,
  activity}`. Liveness state *changes* also emit a record so occupancy
  transitions are logged even when no vitals are reported.
- **Alerts:** abnormal HR/RR reuse `AlertDispatcher` (console + webhook), tagged
  with track_id, mirroring the event-alert path. Opt-in unresponsiveness alert
  (prolonged `PRESENT_STATIC` + no pulse) uses the same dispatcher.
- **CSV recorder** (validation aid): optional `--vitals-csv path` dumps
  `(timestamp, track_id, hr, rr, quality)` for offline comparison against a
  pulse-oximeter reference.

## Testing

Unit (`tests/test_pipeline.py` additions):
- **Estimator recovers known frequency:** synthesize a 1.2 Hz (72 bpm) sinusoid
  buffer → HR within ±2 bpm; 0.25 Hz (15 br/min) → RR within ±1 br/min.
- **Parabolic interpolation** beats raw-bin resolution on an off-bin frequency.
- **Even-resampling** with randomly dropped samples still recovers the frequency.
- **Quality gate:** pure Gaussian noise → low quality → `should_report` False.
- **ROI geometry:** synthetic upright keypoints → forehead above eyes, cheeks
  flanking nose; rolled keypoints → boxes rotate; low-confidence/profile → `None`.
- **Controller:** `max_people` cap selects largest faces; `cadence` throttles
  estimation; abnormal ranges flag correctly.
- **Activity metric:** moving synthetic keypoints → high activity, static → ~0;
  same motion at two bbox scales → similar normalized activity (scale invariance).
- **Liveness state machine:** scripted input sequences drive
  `EMPTY→PRESENT_STATIC→PRESENT_MOVING→LIVE_CONFIRMED` and back; pulse latch holds
  `LIVE_CONFIRMED` over brief stillness for `pulse_hold_s`; unresponsiveness timer
  fires only after `unresponsive_s` of continuous `PRESENT_STATIC` with no pulse.

Integration:
- Synthetic frames with a sinusoidally brightening face ROI through the full
  loop → a confident HR appears for the tracked person.
- `--vitals` off → pipeline output byte-identical to current behavior (event
  detection untouched).

Real-world validation (manual, documented, not automated): record `--vitals-csv`
while wearing a pulse oximeter; compare MAE. Success target from the paper's
neighborhood: HR MAE < 5 bpm at rest, good face visibility.

## Files

| File | Change |
|------|--------|
| `vitals.py` | New: `FaceROIExtractor`, `RPPGEstimator`, `ActivityEstimator`, `LivenessMonitor`, `LivenessState`, `VitalsController` |
| `stream_detect.py` | `PersonState` fields; per-person rPPG + liveness in `run_pipeline`; HUD; vitals log; CLI flags |
| `tests/test_pipeline.py` | New `TestFaceROIExtractor`, `TestRPPGEstimator`, `TestActivityEstimator`, `TestLivenessMonitor`, `TestVitalsController` |
| `prepare.py` | No changes (immutable) |

## Risks / open items

- **Skin-tone & lighting** affect green-channel SNR; the quality gate suppresses
  bad reads rather than emitting wrong ones. CHROM/POS (paper Table 1) are more
  robust than GREEN and can replace `RPPGEstimator.estimate` internals later
  without interface change.
- **Motion** is the dominant rPPG error source; the head-motion gate is a first
  defense, not a correction. Out of scope: motion compensation.
- **5-keypoint ROI** may yield lower SNR than FaceMesh; the extractor interface
  allows a FaceMesh swap if validation shows it's needed.
- **`quality_min` / `motion_max` thresholds** need empirical tuning; defaults are
  placeholders, finalized during real-world validation.
