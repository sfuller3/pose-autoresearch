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
  keypoint-motion activity metric with rPPG pulse confirmation. This *enhances*
  the existing coarse presence signal (a `PersonTracker` track = "body detected")
  with activity gradation and anti-spoof, and can optionally harden the event
  detector against static photos/posters — it is not a parallel mechanism.
- **Deployment topology:** all processing is edge-local; the edge is push-only to
  the cloud and never serves UIs directly. Cloud-down resilience is store-and-
  forward for the data record, plus a local (LAN) life-safety alarm fallback for
  critical events when the cloud is unreachable (see "Edge ↔ cloud topology &
  offline operation").

This is additive. With `--vitals` off (the default), the pipeline behaves
exactly as today — event detection is untouched.

## Deployment context

- **Room:** small, ~12 × 12 ft. Subjects sit at ~6–12 ft from the camera — the
  moderate-to-hard end of the rPPG distance range. At 1080p with a normal ~60°
  lens, a face at the far wall is near the resolution floor (~28 px inter-ocular),
  so **1080p input and a slightly narrow FOV lens are recommended when vitals are
  enabled.** Event detection is unaffected and still runs at any resolution.
- **Lighting:** varies through the day. The signal-quality gate is the primary
  defense — bad light yields low SNR and an "acquiring" state rather than a wrong
  reading.
- **Night / IR mode:** the camera switches to monochrome NIR (IR-LED
  illumination, single intensity channel). This is a first-class operating mode,
  not an edge case — see "Illumination regimes" below. Per the brainstorm
  decision: **liveness/occupancy/activity work 24/7; HR/RR are best-effort at
  night** (NIR single-channel rPPG, quality-gated).

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
- Validity gate (returns `None` when any fails):
  - all required keypoints' confidence `>= KP_CONF_MIN` (default 0.4);
  - near-profile rejection when `ear`/`eye` geometry implies excessive yaw;
  - **resolution floor:** inter-ocular `d >= MIN_INTEROCULAR_PX` (default 28 px)
    and forehead ROI `>= MIN_ROI_PX` (default 30×30 px). This is the
    distance/resolution gate — below it the face has too few skin pixels for a
    trustworthy signal (SNR scales ~√pixels), so the person stays occupancy/
    activity-tracked but rPPG is not attempted. Pixel-based, not distance-based,
    so it self-adjusts to resolution and lens.
- Output: list of ROI pixel masks/boxes (clipped to frame bounds) + a validity
  flag. Behind a clean interface so a FaceMesh-based extractor can be swapped in
  later without touching callers.

**`RPPGEstimator` (stateless compute)**
- `roi_mean(frame, rois, regime) -> float`: mean pixel value over the union of
  valid ROI pixels, channel chosen by illumination regime:
  - **`DAY` (RGB):** green channel (BGR index 1) — the strongest visible PPG
    channel.
  - **`NIGHT` (IR/monochrome):** the single intensity channel (any channel; they
    are equal in a grayscale frame). NIR PPG amplitude is lower, so night reads
    lean harder on the quality gate.
  The downstream `estimate(...)` chain is identical for both — only the source
  scalar differs.
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

### Illumination regimes (day RGB / night IR)

The camera runs visible RGB by day and monochrome NIR at night. The green-channel
method has no green channel at night, so the source channel switches by regime.

**`detect_regime(frame) -> DAY | NIGHT`** (`vitals.py`): a frame is `NIGHT` when
its mean chroma saturation is near zero (R≈G≈B everywhere) — the reliable
signature of IR/monochrome output, independent of camera metadata we may not get.
Evaluated on a throttled cadence (e.g. once per second) and smoothed with
hysteresis so a brief lighting change doesn't flap the regime.

Behavior by regime:
- **DAY:** green-channel rPPG as specified; HR and RR attempted normally.
- **NIGHT:** single NIR intensity channel. HR attempted (best-effort, lower SNR);
  the quality gate suppresses weak reads, so the HUD shows a night HR only when
  it clears threshold. RR via rPPG is the hardest case and will frequently sit in
  "acquiring" — acceptable per the decision that night vitals are best-effort.

What does **not** change at night: tracking, the activity metric, and liveness up
to `PRESENT_MOVING` all run on grayscale keypoints exactly as in daylight, so the
occupancy/liveness signal is unaffected. Only the `LIVE_CONFIRMED` pulse rung is
harder to reach, which the latch (`pulse_hold_s`) and quality gate handle
honestly. The regime is logged with each vitals record so day/night reads are
distinguishable downstream.

### `PersonState` additions (in `stream_detect.py`)

- `vitals_buffer`: `collections.deque[(timestamp, roi_mean)]`,
  `maxlen = max(HR_WINDOW, RR_WINDOW)` (the scalar is green by day, NIR intensity
  at night).
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
- `live_gating: bool` — when set (`--live-gating`), `apply_context_rules`
  receives a liveness-derived "live person present" instead of raw body
  detection. Default False (event path unchanged).

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

**Relationship to existing presence detection.** The pipeline already has a
coarse presence signal: a `PersonTracker` track existing (`get_active_people` /
the `people` dict) means "a body was detected," and `person_detected` in
`apply_context_rules` gates fall/aggression on it. That signal only answers "did
YOLO detect a human shape" — it cannot distinguish a live person from a photo,
poster, mannequin, or a person on a TV, and it has no activity gradation. The
liveness state machine *consumes* that existing presence rather than re-detecting:
`is_tracked` (the bottom rung, `EMPTY` vs `PRESENT_*`) comes straight from the
tracker, and activity + pulse are the new rungs layered above it. This is an
accuracy enhancement of an existing feature, not a parallel mechanism.

**Optional event-detector hardening.** `apply_context_rules` currently takes a
boolean `person_detected`. With liveness available, the caller can pass
`liveness >= PRESENT_MOVING` (or `LIVE_CONFIRMED` when rPPG is on) instead of raw
body detection, so a static photo/poster on the wall no longer suppresses or
triggers events. This is gated behind `--vitals` (liveness must be running) and
behind a `--live-gating` flag so the default event path is unchanged; it is a
small, isolated change at the existing call site, not a rework of the rules.

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
  activity, regime}`. Liveness state *changes* also emit a record so occupancy
  transitions are logged even when no vitals are reported.
- **Alerts:** abnormal HR/RR reuse `AlertDispatcher` (console + webhook), tagged
  with track_id, mirroring the event-alert path. Opt-in unresponsiveness alert
  (prolonged `PRESENT_STATIC` + no pulse) uses the same dispatcher.
- **CSV recorder** (validation aid): optional `--vitals-csv path` dumps
  `(timestamp, track_id, hr, rr, quality)` for offline comparison against a
  pulse-oximeter reference.

## Data flow & resolution

Diagram: `docs/superpowers/specs/2026-06-26-vitals-data-flow.svg`.

The camera frame is decoded **once at full resolution** and forks:

- **Full-res → local rotating buffer** (the existing `ClipRecorder`, kept at
  full-res, review-only) and **→ per-ROI vitals sampling**. rPPG is the only
  consumer that needs full-res pixels (skin-pixel SNR).
- **Full-res → single downscale to 320/640 → pose + tracking.** Pose then fans
  out to fall/event detection, liveness/activity, the privacy-shape display, and
  face-ROI derivation. All of these run on the cheap reduced stream.

**Coordinate scaling (load-bearing).** Keypoints come back in reduced-frame
coordinates; the ROI extractor multiplies them by `full_res / pose_res` before
indexing the full-res frame. The distance/resolution validity gate
(`MIN_INTEROCULAR_PX`, `MIN_ROI_PX`) is evaluated in **full-res pixels**, so the
threshold is meaningful regardless of the pose downscale factor.

**Compute scales with vitals, not headcount.** Pose, fall, liveness, and privacy
cost is fixed per frame; only the `max_people` capped faces incur full-res ROI
sampling + rPPG.

**Privacy boundary.** The live display renders only pose-derived shapes. Full-res
video exists solely for (a) the local review buffer and (b) per-ROI vitals
sampling — it is never sent to the display, and a static photo/poster can never
reach `LIVE_CONFIRMED`.

## Edge ↔ cloud topology & offline operation

**Trust boundary — the edge never serves data to workers or families directly.**
The edge device is **push-only to the cloud**; the cloud is the sole system that
serves dashboards and apps to caregivers and families. No inbound viewing
connections to the edge — this is both the product rule and the security/privacy
posture. All CV/vitals processing is edge-local; raw full-res never leaves the
edge unless a clip is explicitly escalated to the cloud for review.

**Normal flow:** edge → cloud carries events (falls, abnormal vitals,
unresponsiveness), vitals records, liveness/occupancy transitions, and clip
references (clips uploaded only on escalation). Cloud → consumers (web/app).

**Offline operation (cloud unreachable)** — split by stakes:

- **Convenience monitoring** (live view, history, family access) is **cloud-only**
  and simply unavailable until reconnect. Acceptable.
- **The data record is never lost — store-and-forward (always on):** every event,
  vitals record, liveness transition, and escalated clip persists to a durable
  local queue and backfills on reconnect. The local store is bounded (ring) with
  safety events retained at the highest priority.
- **Edge self-health heartbeat:** the edge periodically reports its own liveness
  to the cloud so a unit going offline is detectable and surfaced to staff.
- **Local life-safety fallback (DECIDED):** for *critical* events only (fall,
  prolonged unresponsiveness, severe abnormal vitals), the edge raises a one-way
  alarm over the **local network** to the facility's own on-prem infrastructure
  (nurse-call relay / LAN alert appliance / optional cellular SMS), independent of
  the cloud. This is an alarm *to facility systems*, not a data-serving UI, so it
  preserves "the edge never serves workers/families directly" while ensuring
  life-safety alerts survive a cloud outage. Cellular/secondary-WAN escalation to
  the cloud is explicitly out of scope for this version (future option).
  - **Seam:** the existing `AlertDispatcher` (console + webhook) generalizes to a
    pluggable sink list. A `LANAlarmSink` (HTTP/webhook to a configured on-prem
    endpoint, e.g. `--lan-alarm-url`) is added alongside the cloud sink. Critical
    events fan out to all configured sinks; the LAN sink fires regardless of cloud
    reachability, the cloud sink rides the store-and-forward queue.
  - **Severity tiering:** only events classified *critical* reach the LAN alarm;
    routine vitals/occupancy never do. The abnormal-vitals ranges and the
    unresponsiveness branch already defined above are the critical triggers.

## Vistarra coordination (revision required before implementation)

This repo (`pose-autoresearch`) is the model-research arm of the **Vistarra**
monorepo (consolidates `memryx-fall-demo` + `pose-autoresearch`). A 2026-06-28
review of the Vistarra repo found this spec's standalone assumptions must be
reconciled with Vistarra's existing edge/cloud architecture:

- **Event contract:** emit Vistarra's `EdgeEvent`/`EventEnvelope`
  (`edge/src/detection/events.py`, `envelope.py`) — `event_type` (EventType
  enum), `urgency` (EventUrgency: immediate/alert/log_only), `camera_id`,
  `facility_id`, etc. Do NOT use the ad-hoc `{"class"/"event"}` dicts sketched in
  these plans. Vitals need NEW EventTypes (e.g. `VITALS_READING`,
  `ABNORMAL_VITALS`) added to that enum + matching detail models.
- **Severity → urgency:** drop the invented `Severity` enum; reuse `EventUrgency`.
  Its documented routing already is the hybrid we want: immediate/alert → cloud
  VLM (Claude Vision), log_only → edge only.
- **Cloud delivery:** reuse `edge/src/upload/cloud_client.py`
  (`submit_envelope` → `POST /api/analyze`, `UploadWorker` store-and-forward,
  `/api/health`). **Plan 2 (`edge_sync.py`) is largely redundant with this and
  should be dropped/retargeted to thin glue, not a parallel stack.**
- **Claude Vision retained (hybrid):** the edge trained model does fast
  first-pass; the cloud VLM remains the confirmation for high-stakes events —
  specifically fall (`FALL`), unstable gait (`GAIT_ANOMALY`), and getting out of
  bed (`BED_EXIT_PREDICTED`). This is existing urgency→VLM routing, not new work.
- **Liveness overlap:** `inactivity_detector` and `nighttime_detector` already
  track movement/`max_keypoint_displacement_px`/`consecutive_checks_missed`.
  Integrate the liveness/activity signal with those rather than re-implementing;
  the rPPG `LIVE_CONFIRMED` rung is the genuinely new contribution.
- **Edge hardware:** MemryX MX3 (ONNX→MX3 export, `edge/models/mx3_*.py`), not
  Jetson/TensorRT. Axelera/DeepX are alternate backends.

Plan 1 (signal chain `vitals.py`) is mostly net-new and survives revision; its
`stream_detect.py` integration points become Vistarra `edge/` integration points
and its alert emission must build `EdgeEvent`s routed via `cloud_client`.

## Testing

Unit (`tests/test_pipeline.py` additions):
- **Estimator recovers known frequency:** synthesize a 1.2 Hz (72 bpm) sinusoid
  buffer → HR within ±2 bpm; 0.25 Hz (15 br/min) → RR within ±1 br/min.
- **Parabolic interpolation** beats raw-bin resolution on an off-bin frequency.
- **Even-resampling** with randomly dropped samples still recovers the frequency.
- **Quality gate:** pure Gaussian noise → low quality → `should_report` False.
- **ROI geometry:** synthetic upright keypoints → forehead above eyes, cheeks
  flanking nose; rolled keypoints → boxes rotate; low-confidence/profile → `None`.
- **Distance/resolution gate:** keypoints with inter-ocular below
  `MIN_INTEROCULAR_PX` → `None` (rPPG skipped) while the track still reports
  occupancy/activity; above the floor → valid ROIs.
- **Regime detection:** a saturated RGB frame → `DAY` (samples green); a
  grayscale/IR frame (R==G==B) → `NIGHT` (samples intensity); hysteresis holds
  through a one-frame blip. NIR recovery: a known-frequency sinusoid in the
  intensity channel still yields the right HR.
- **Controller:** `max_people` cap selects largest faces; `cadence` throttles
  estimation; abnormal ranges flag correctly.
- **Activity metric:** moving synthetic keypoints → high activity, static → ~0;
  same motion at two bbox scales → similar normalized activity (scale invariance).
- **Liveness state machine:** scripted input sequences drive
  `EMPTY→PRESENT_STATIC→PRESENT_MOVING→LIVE_CONFIRMED` and back; pulse latch holds
  `LIVE_CONFIRMED` over brief stillness for `pulse_hold_s`; unresponsiveness timer
  fires only after `unresponsive_s` of continuous `PRESENT_STATIC` with no pulse.
- **Alert sinks & severity:** a critical event fans out to all configured sinks;
  `LANAlarmSink` fires even when the cloud sink is unreachable; routine
  vitals/occupancy events never reach the LAN alarm (severity tiering).
- **Live-gating:** with `live_gating` on, a `PRESENT_STATIC` (photo-like, no
  motion/pulse) track yields "not a live person" to `apply_context_rules`, while a
  `PRESENT_MOVING`/`LIVE_CONFIRMED` track yields "live"; with `live_gating` off,
  the boolean passed is the unchanged raw body-detection value.

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
| `vitals.py` | New: `FaceROIExtractor`, `RPPGEstimator`, `ActivityEstimator`, `LivenessMonitor`, `LivenessState`, `VitalsController`, `detect_regime` / regime enum |
| `stream_detect.py` | `PersonState` fields; per-person rPPG + liveness in `run_pipeline`; HUD; vitals log; CLI flags; optional `--live-gating`; `AlertDispatcher` → pluggable sink list with a `LANAlarmSink` for critical events (`--lan-alarm-url`) |
| `tests/test_pipeline.py` | New `TestFaceROIExtractor`, `TestRPPGEstimator`, `TestActivityEstimator`, `TestLivenessMonitor`, `TestVitalsController` |
| `prepare.py` | No changes (immutable) |

## Risks / open items

- **Skin-tone & lighting** affect green-channel SNR; the quality gate suppresses
  bad reads rather than emitting wrong ones. CHROM/POS (paper Table 1) are more
  robust than GREEN and can replace the `DAY` channel extraction later without
  interface change.
- **Night/IR is physics-limited:** NIR PPG amplitude is inherently lower, so
  night HR will report less often and RR rarely. This is expected and handled by
  the quality gate + best-effort decision, not a defect. If night HR proves too
  sparse in validation, a NIR-tuned method (e.g. larger ROI, longer window, or an
  NIR-specific algorithm) is the upgrade path — behind the same interface.
- **Regime mis-detection** (e.g. a very desaturated daytime scene) could pick the
  wrong channel; hysteresis + the saturation threshold mitigate, and at worst the
  quality gate rejects the resulting weak signal.
- **Motion** is the dominant rPPG error source; the head-motion gate is a first
  defense, not a correction. Out of scope: motion compensation.
- **5-keypoint ROI** may yield lower SNR than FaceMesh; the extractor interface
  allows a FaceMesh swap if validation shows it's needed.
- **`quality_min` / `motion_max` thresholds** need empirical tuning; defaults are
  placeholders, finalized during real-world validation.
