# Vitals Signal Chain + Liveness Implementation Plan (Plan 1 of 2)

> **SUPERSEDED (2026-06-28):** Replaced by the Vistarra-targeted design at
> `Vistarra/docs/superpowers/specs/2026-06-28-contactless-vitals-design.md`. This
> standalone plan predated the Vistarra architecture review. The `vitals.py` DSP
> here is portable, but integration moves to Vistarra's `edge/` (BaseDetector,
> EventEnvelope, EventQueue/UploadWorker, dual-stream). Do not execute as-is.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On-device contactless HR/RR (rPPG) and a fused liveness/occupancy signal for every tracked person, integrated into `stream_detect.py`, all runnable and testable without any network.

**Architecture:** A new `vitals.py` holds the isolated, stateless-where-possible signal chain (regime detection, rPPG DSP, face-ROI geometry, activity metric, liveness state machine, controller). `stream_detect.py` gains per-person vitals fields and a small block in the per-person loop that calls into `vitals.py`, plus HUD + a `vitals_log.jsonl` writer and CLI flags. Source paper: Kolosov et al. 2023 (Sensors 23, 4550).

**Tech Stack:** Python 3.9 (no 3.10+ syntax — the repo's venv is 3.9), NumPy, OpenCV (cv2, BGR frames). Tests via `.venv/bin/python -m pytest`.

**Conventions:**
- All commands from repo root `/Users/samfuller/Projects/pose-autoresearch`.
- `prepare.py` is IMMUTABLE.
- The existing 70 `tests/test_pipeline.py` tests must keep passing after every task.
- Scope boundary: this plan does NOT touch networking, cloud sync, or alert-sink refactoring — that is Plan 2 (`2026-06-28-vitals-infrastructure.md`). Abnormal-vitals alerts here use the existing `AlertDispatcher.dispatch` as-is.

**File structure:**
- Create `vitals.py` — all signal-chain units (Tasks 1-7).
- Modify `stream_detect.py` — `PersonState` fields + per-person loop + HUD + log + CLI (Tasks 8-10).
- Modify `tests/test_pipeline.py` — append test classes per task.

---

### Task 1: vitals.py skeleton — constants, LivenessState, Regime

**Files:**
- Create: `vitals.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
# ============================================================================
# VITALS SIGNAL CHAIN TESTS
# ============================================================================


class TestVitalsConstants:
    def test_bands_and_windows(self):
        import vitals
        assert vitals.HR_BAND == (0.83, 3.0)
        assert vitals.RR_BAND == (0.18, 0.5)
        assert vitals.RR_WINDOW >= vitals.HR_WINDOW
        assert vitals.FPS == 30

    def test_liveness_state_ordering(self):
        from vitals import LivenessState
        assert LivenessState.EMPTY < LivenessState.PRESENT_STATIC
        assert LivenessState.PRESENT_STATIC < LivenessState.PRESENT_MOVING
        assert LivenessState.PRESENT_MOVING < LivenessState.LIVE_CONFIRMED

    def test_regime_values(self):
        from vitals import Regime
        assert Regime.DAY != Regime.NIGHT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsConstants -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vitals'`

- [ ] **Step 3: Create vitals.py**

```python
"""Contactless vital-signs signal chain (rPPG) + liveness.

Isolated from stream_detect.py so the DSP/geometry units stay testable on
their own. Based on Kolosov et al. 2023 (Sensors 23, 4550), adapted from
MediaPipe face models to YOLO pose keypoints and from single- to multi-person.
"""

from __future__ import annotations

import enum

import numpy as np

# Frequency bands of interest
HR_BAND = (0.83, 3.0)    # 50-180 bpm
RR_BAND = (0.18, 0.5)    # 11-30 breaths/min

# Buffer windows (frames @ FPS). RR needs a longer window for usable resolution.
FPS = 30
HR_WINDOW = 256          # ~8.5 s
RR_WINDOW = 512          # ~17 s

# Face-ROI validity thresholds (evaluated in FULL-RES pixels)
KP_CONF_MIN = 0.4
MIN_INTEROCULAR_PX = 28
MIN_ROI_PX = 30

# COCO-17 facial keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6


class LivenessState(enum.IntEnum):
    EMPTY = 0
    PRESENT_STATIC = 1
    PRESENT_MOVING = 2
    LIVE_CONFIRMED = 3


class Regime(str, enum.Enum):
    DAY = "day"      # RGB available -> green channel
    NIGHT = "night"  # monochrome NIR -> single intensity channel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsConstants -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add vitals.py tests/test_pipeline.py
git commit -m "feat: vitals.py skeleton — constants, LivenessState, Regime"
```

---

### Task 2: Illumination regime detection

**Files:**
- Modify: `vitals.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestDetectRegime:
    def test_color_frame_is_day(self):
        from vitals import detect_regime, Regime
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        frame[..., 1] = 200  # strong green only -> saturated
        frame[..., 0] = 20
        assert detect_regime(frame) == Regime.DAY

    def test_grayscale_frame_is_night(self):
        from vitals import detect_regime, Regime
        gray = np.full((48, 48, 3), 120, dtype=np.uint8)  # R==G==B
        assert detect_regime(gray) == Regime.NIGHT

    def test_near_gray_is_night(self):
        from vitals import detect_regime, Regime
        frame = np.full((48, 48, 3), 120, dtype=np.uint8)
        frame[..., 0] = 122  # tiny channel difference -> still night
        assert detect_regime(frame) == Regime.NIGHT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestDetectRegime -v`
Expected: FAIL with `ImportError: cannot import name 'detect_regime'`

- [ ] **Step 3: Implement in vitals.py**

Append:

```python
SATURATION_NIGHT_MAX = 0.04  # mean normalized saturation below this => NIGHT


def detect_regime(frame, threshold: float = SATURATION_NIGHT_MAX) -> Regime:
    """Classify a BGR frame as DAY (color) or NIGHT (IR/monochrome).

    Uses mean per-pixel saturation (max-min across channels, normalized by max).
    A monochrome/IR frame has R==G==B so saturation ~ 0.
    """
    f = frame.astype(np.float32)
    mx = f.max(axis=2)
    mn = f.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    return Regime.NIGHT if float(sat.mean()) < threshold else Regime.DAY
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestDetectRegime -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add vitals.py tests/test_pipeline.py
git commit -m "feat: saturation-based day/night regime detection"
```

---

### Task 3: rPPG estimator — the DSP chain

**Files:**
- Modify: `vitals.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestRPPGEstimator:
    def _synth(self, freq_hz, n=512, fps=30, noise=0.0, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(n) / fps
        sig = np.sin(2 * np.pi * freq_hz * t)
        if noise:
            sig = sig + noise * rng.standard_normal(n)
        return t, sig

    def test_recovers_hr_frequency(self):
        from vitals import RPPGEstimator, HR_BAND
        t, sig = self._synth(1.2)  # 72 bpm
        freq, quality = RPPGEstimator.estimate(t, sig, HR_BAND)
        assert abs(freq * 60 - 72) < 2.0
        assert quality > 2.0

    def test_recovers_rr_frequency(self):
        from vitals import RPPGEstimator, RR_BAND
        t, sig = self._synth(0.25, n=512)  # 15 br/min
        freq, quality = RPPGEstimator.estimate(t, sig, RR_BAND)
        assert abs(freq * 60 - 15) < 1.0

    def test_parabolic_beats_raw_bin(self):
        from vitals import RPPGEstimator, HR_BAND
        # off-bin frequency: 1.27 Hz = 76.2 bpm
        t, sig = self._synth(1.27)
        freq, _ = RPPGEstimator.estimate(t, sig, HR_BAND)
        assert abs(freq * 60 - 76.2) < 1.5

    def test_dropped_samples_still_recover(self):
        from vitals import RPPGEstimator, HR_BAND
        t, sig = self._synth(1.5, n=400)  # 90 bpm
        keep = np.ones(len(t), dtype=bool)
        keep[::7] = False  # drop ~14% of samples
        freq, _ = RPPGEstimator.estimate(t[keep], sig[keep], HR_BAND)
        assert abs(freq * 60 - 90) < 3.0

    def test_noise_gives_low_quality(self):
        from vitals import RPPGEstimator, HR_BAND
        rng = np.random.default_rng(1)
        t = np.arange(512) / 30
        noise = rng.standard_normal(512)
        _, quality = RPPGEstimator.estimate(t, noise, HR_BAND)
        assert quality < 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestRPPGEstimator -v`
Expected: FAIL with `ImportError: cannot import name 'RPPGEstimator'`

- [ ] **Step 3: Implement in vitals.py**

Append:

```python
class RPPGEstimator:
    """Stateless rPPG signal processing: signal -> (frequency, quality)."""

    @staticmethod
    def roi_mean(frame, boxes, regime: Regime) -> float:
        """Mean pixel value over ROI boxes; green by day, intensity at night.

        boxes: list of (x1, y1, x2, y2) integer pixel rects (full-res frame).
        """
        chan = 1 if regime == Regime.DAY else 0  # BGR green by day; any chan at night
        vals = []
        h, w = frame.shape[:2]
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                vals.append(float(frame[y1:y2, x1:x2, chan].mean()))
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def estimate(timestamps, values, band, fps: int = FPS):
        """Return (freq_hz, quality) for the dominant peak in `band`.

        quality = in-band SNR (peak power / mean of remaining in-band power).
        """
        t = np.asarray(timestamps, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        if len(v) < 16:
            return 0.0, 0.0

        # 1. detrend (remove mean + linear drift)
        coeffs = np.polyfit(t, v, 1)
        v = v - np.polyval(coeffs, t)

        # 2. resample to an even grid at fps
        n = len(v)
        even_t = np.linspace(t[0], t[-1], n)
        v = np.interp(even_t, t, v)

        # 3. Hamming window
        v = v * np.hamming(n)

        # 4. L2 normalize
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return 0.0, 0.0
        v = v / norm

        # 5. zero-padded real FFT
        nfft = 1
        while nfft < n * 4:
            nfft *= 2
        spectrum = np.abs(np.fft.rfft(v, n=nfft))
        freqs = np.fft.rfftfreq(nfft, d=1.0 / fps)

        # 6. peak within band + parabolic interpolation
        lo, hi = band
        band_mask = (freqs >= lo) & (freqs <= hi)
        if not band_mask.any():
            return 0.0, 0.0
        band_idx = np.where(band_mask)[0]
        local_peak = band_idx[np.argmax(spectrum[band_idx])]
        peak_freq = freqs[local_peak]
        if 0 < local_peak < len(spectrum) - 1:
            a, b, c = spectrum[local_peak - 1], spectrum[local_peak], spectrum[local_peak + 1]
            denom = (a - 2 * b + c)
            if abs(denom) > 1e-12:
                delta = 0.5 * (a - c) / denom
                peak_freq = peak_freq + delta * (freqs[1] - freqs[0])

        # 7. in-band SNR quality
        peak_power = spectrum[local_peak] ** 2
        others = spectrum[band_idx] ** 2
        others_mean = (others.sum() - peak_power) / max(len(others) - 1, 1)
        quality = float(peak_power / (others_mean + 1e-12))
        return float(peak_freq), quality
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestRPPGEstimator -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add vitals.py tests/test_pipeline.py
git commit -m "feat: RPPGEstimator — detrend/resample/FFT/parabolic peak + SNR + roi_mean"
```

---

### Task 4: Face ROI extractor (forehead + cheeks, resolution gate)

**Files:**
- Modify: `vitals.py`
- Test: `tests/test_pipeline.py`

Note: ROIs are axis-aligned boxes derived from keypoints. Eye-line roll
rotation is deferred (a future enhancement noted in the spec) — axis-aligned is
robust and sufficient for v1. Coordinates are scaled to full-res by the caller
passing `pose_scale` (full_res / pose_res).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestFaceROIExtractor:
    def _face(self, cx=320, cy=240, d=60, conf=0.9):
        # upright face: eyes at +/- d/2, nose below midpoint, ears outside eyes
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[1] = [cx - d / 2, cy, conf]        # left eye
        kps[2] = [cx + d / 2, cy, conf]        # right eye
        kps[0] = [cx, cy + d * 0.4, conf]      # nose
        kps[3] = [cx - d, cy, conf]            # left ear
        kps[4] = [cx + d, cy, conf]            # right ear
        kps[5] = [cx - d, cy + 3 * d, conf]    # left shoulder
        kps[6] = [cx + d, cy + 3 * d, conf]    # right shoulder
        return kps

    def test_returns_three_boxes(self):
        from vitals import FaceROIExtractor
        boxes = FaceROIExtractor.extract(self._face(), (480, 640))
        assert boxes is not None
        assert len(boxes) == 3

    def test_forehead_above_eyes(self):
        from vitals import FaceROIExtractor
        boxes = FaceROIExtractor.extract(self._face(cy=240), (480, 640))
        forehead = boxes[0]
        assert forehead[3] <= 240  # forehead bottom at or above eye line

    def test_low_confidence_rejected(self):
        from vitals import FaceROIExtractor
        assert FaceROIExtractor.extract(self._face(conf=0.1), (480, 640)) is None

    def test_too_far_rejected(self):
        from vitals import FaceROIExtractor
        # d=20 -> inter-ocular 20 px < MIN_INTEROCULAR_PX (28)
        assert FaceROIExtractor.extract(self._face(d=20), (480, 640)) is None

    def test_pose_scale_upsamples_coords(self):
        from vitals import FaceROIExtractor
        # face detected on a 0.5x downscaled frame; full-res is 2x
        small = self._face(cx=160, cy=120, d=30)
        boxes = FaceROIExtractor.extract(small, (480, 640), pose_scale=2.0)
        assert boxes is not None
        # boxes should be in full-res coords (~around 320,240), not 160,120
        forehead = boxes[0]
        assert forehead[0] > 200

    def test_profile_rejected(self):
        from vitals import FaceROIExtractor
        kps = self._face()
        kps[1, 2] = 0.1  # left eye not visible -> profile/occluded
        assert FaceROIExtractor.extract(kps, (480, 640)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestFaceROIExtractor -v`
Expected: FAIL with `ImportError: cannot import name 'FaceROIExtractor'`

- [ ] **Step 3: Implement in vitals.py**

Append:

```python
class FaceROIExtractor:
    """Derive forehead + cheek ROI boxes from COCO-17 keypoints.

    Returns a list of three (x1, y1, x2, y2) integer boxes in full-resolution
    pixel coordinates, or None when the face is invalid (low confidence,
    profile/occluded, or too small for a trustworthy signal).
    """

    @staticmethod
    def extract(kps, frame_shape, pose_scale: float = 1.0):
        h, w = frame_shape[:2]
        required = (NOSE, L_EYE, R_EYE)
        if any(kps[i, 2] < KP_CONF_MIN for i in required):
            return None
        # Profile/occlusion: both eyes must be confident
        if kps[L_EYE, 2] < KP_CONF_MIN or kps[R_EYE, 2] < KP_CONF_MIN:
            return None

        le = kps[L_EYE, :2] * pose_scale
        re = kps[R_EYE, :2] * pose_scale
        nose = kps[NOSE, :2] * pose_scale
        d = float(np.linalg.norm(re - le))
        if d < MIN_INTEROCULAR_PX:
            return None

        m = (le + re) / 2.0  # eye midpoint
        # axis-aligned boxes sized by inter-ocular distance d
        fore = (
            int(m[0] - 0.65 * d), int(m[1] - 0.9 * d),
            int(m[0] + 0.65 * d), int(m[1] - 0.25 * d),
        )
        half = 0.3 * d
        l_cheek = (
            int(le[0] - half), int(le[1] + 0.3 * d),
            int(le[0] + half), int(le[1] + 0.9 * d),
        )
        r_cheek = (
            int(re[0] - half), int(re[1] + 0.3 * d),
            int(re[0] + half), int(re[1] + 0.9 * d),
        )

        boxes = []
        for (x1, y1, x2, y2) in (fore, l_cheek, r_cheek):
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) >= MIN_ROI_PX and (y2 - y1) >= MIN_ROI_PX:
                boxes.append((x1, y1, x2, y2))
        # forehead is required; if it clipped away, the face is too marginal
        if not boxes or (fore[2] - fore[0]) < MIN_ROI_PX:
            return None
        return boxes
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestFaceROIExtractor -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add vitals.py tests/test_pipeline.py
git commit -m "feat: FaceROIExtractor — forehead/cheek boxes with full-res resolution gate"
```

---

### Task 5: Activity estimator (scale-invariant motion)

**Files:**
- Modify: `vitals.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestActivityEstimator:
    def _pose(self, cx=320, cy=240, scale=60, conf=0.9):
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[5] = [cx - scale, cy, conf]   # left shoulder
        kps[6] = [cx + scale, cy, conf]   # right shoulder
        for j in range(7, 17):
            kps[j] = [cx, cy + scale, conf]
        return kps

    def test_static_is_near_zero(self):
        from vitals import ActivityEstimator
        p = self._pose()
        assert ActivityEstimator.frame_activity(p, p) < 1e-6

    def test_motion_is_positive(self):
        from vitals import ActivityEstimator
        p0 = self._pose(cx=320)
        p1 = self._pose(cx=330)  # moved 10 px
        assert ActivityEstimator.frame_activity(p0, p1) > 0.01

    def test_scale_invariance(self):
        from vitals import ActivityEstimator
        # same proportional motion at two body scales -> similar activity
        near0, near1 = self._pose(cx=320, scale=120), self._pose(cx=344, scale=120)
        far0, far1 = self._pose(cx=320, scale=60), self._pose(cx=332, scale=60)
        a_near = ActivityEstimator.frame_activity(near0, near1)
        a_far = ActivityEstimator.frame_activity(far0, far1)
        assert abs(a_near - a_far) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestActivityEstimator -v`
Expected: FAIL with `ImportError: cannot import name 'ActivityEstimator'`

- [ ] **Step 3: Implement in vitals.py**

Append:

```python
class ActivityEstimator:
    """Scale-invariant whole-body keypoint motion between two frames."""

    @staticmethod
    def frame_activity(prev_kps, cur_kps) -> float:
        """Mean per-joint displacement over confident joints, normalized by
        shoulder width (so distance from the camera does not change the metric).
        """
        conf = (prev_kps[:, 2] >= KP_CONF_MIN) & (cur_kps[:, 2] >= KP_CONF_MIN)
        if not conf.any():
            return 0.0
        disp = np.linalg.norm(cur_kps[conf, :2] - prev_kps[conf, :2], axis=1)
        shoulder = np.linalg.norm(cur_kps[L_SHOULDER, :2] - cur_kps[R_SHOULDER, :2])
        scale = shoulder if shoulder > 1.0 else 1.0
        return float(disp.mean() / scale)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestActivityEstimator -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add vitals.py tests/test_pipeline.py
git commit -m "feat: ActivityEstimator — scale-invariant keypoint motion metric"
```

---

### Task 6: LivenessMonitor state machine

**Files:**
- Modify: `vitals.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestLivenessMonitor:
    def test_untracked_is_empty(self):
        from vitals import LivenessMonitor, LivenessState
        m = LivenessMonitor()
        assert m.update(now=0.0, is_tracked=False, activity=0.0, pulse_quality=0.0) == LivenessState.EMPTY

    def test_tracked_still_is_static(self):
        from vitals import LivenessMonitor, LivenessState
        m = LivenessMonitor()
        assert m.update(now=1.0, is_tracked=True, activity=0.0, pulse_quality=0.0) == LivenessState.PRESENT_STATIC

    def test_motion_is_moving(self):
        from vitals import LivenessMonitor, LivenessState
        m = LivenessMonitor(move_threshold=0.02)
        assert m.update(now=1.0, is_tracked=True, activity=0.5, pulse_quality=0.0) == LivenessState.PRESENT_MOVING

    def test_pulse_is_confirmed(self):
        from vitals import LivenessMonitor, LivenessState
        m = LivenessMonitor(quality_min=2.0)
        assert m.update(now=1.0, is_tracked=True, activity=0.0, pulse_quality=5.0) == LivenessState.LIVE_CONFIRMED

    def test_pulse_latches_over_brief_stillness(self):
        from vitals import LivenessMonitor, LivenessState
        m = LivenessMonitor(quality_min=2.0, pulse_hold_s=10.0)
        m.update(now=1.0, is_tracked=True, activity=0.0, pulse_quality=5.0)
        # 5 s later, no pulse, still tracked -> still LIVE (within hold)
        assert m.update(now=6.0, is_tracked=True, activity=0.0, pulse_quality=0.0) == LivenessState.LIVE_CONFIRMED

    def test_unresponsive_after_threshold(self):
        from vitals import LivenessMonitor
        m = LivenessMonitor(unresponsive_s=30.0)
        m.update(now=0.0, is_tracked=True, activity=0.0, pulse_quality=0.0)
        assert m.unresponsive(now=10.0) is False
        m.update(now=10.0, is_tracked=True, activity=0.0, pulse_quality=0.0)
        assert m.unresponsive(now=31.0) is True

    def test_motion_resets_unresponsive(self):
        from vitals import LivenessMonitor
        m = LivenessMonitor(unresponsive_s=30.0, move_threshold=0.02)
        m.update(now=0.0, is_tracked=True, activity=0.0, pulse_quality=0.0)
        m.update(now=20.0, is_tracked=True, activity=0.5, pulse_quality=0.0)  # moved
        assert m.unresponsive(now=31.0) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestLivenessMonitor -v`
Expected: FAIL with `ImportError: cannot import name 'LivenessMonitor'`

- [ ] **Step 3: Implement in vitals.py**

Append:

```python
class LivenessMonitor:
    """Per-person fused liveness state machine.

    Strongest-evidence-first: a recent pulse -> LIVE_CONFIRMED (latched for
    pulse_hold_s), recent motion -> PRESENT_MOVING, tracked-but-still ->
    PRESENT_STATIC, untracked -> EMPTY.
    """

    def __init__(self, move_threshold: float = 0.02, move_hold_s: float = 3.0,
                 pulse_hold_s: float = 10.0, quality_min: float = 2.0,
                 unresponsive_s: float = 30.0):
        self.move_threshold = move_threshold
        self.move_hold_s = move_hold_s
        self.pulse_hold_s = pulse_hold_s
        self.quality_min = quality_min
        self.unresponsive_s = unresponsive_s
        self.last_move_time = None
        self.last_pulse_time = None
        self.state = LivenessState.EMPTY

    def update(self, now: float, is_tracked: bool, activity: float,
               pulse_quality: float) -> LivenessState:
        if not is_tracked:
            self.state = LivenessState.EMPTY
            return self.state
        if activity > self.move_threshold:
            self.last_move_time = now
        if pulse_quality >= self.quality_min:
            self.last_pulse_time = now

        if self.last_pulse_time is not None and (now - self.last_pulse_time) <= self.pulse_hold_s:
            self.state = LivenessState.LIVE_CONFIRMED
        elif self.last_move_time is not None and (now - self.last_move_time) <= self.move_hold_s:
            self.state = LivenessState.PRESENT_MOVING
        else:
            self.state = LivenessState.PRESENT_STATIC
        return self.state

    def unresponsive(self, now: float) -> bool:
        """True when continuously static (no motion, no pulse) for unresponsive_s."""
        if self.state != LivenessState.PRESENT_STATIC:
            return False
        last_activity = max(
            self.last_move_time if self.last_move_time is not None else -1e9,
            self.last_pulse_time if self.last_pulse_time is not None else -1e9,
        )
        return (now - last_activity) >= self.unresponsive_s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestLivenessMonitor -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add vitals.py tests/test_pipeline.py
git commit -m "feat: LivenessMonitor — fused state machine + unresponsiveness timer"
```

---

### Task 7: VitalsController (config + gating + people selection)

**Files:**
- Modify: `vitals.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline.py`:

```python
class TestVitalsController:
    def test_disabled_by_default(self):
        from vitals import VitalsController
        assert VitalsController().enabled is False

    def test_cadence_throttle(self):
        from vitals import VitalsController
        c = VitalsController(enabled=True, cadence=15)
        assert c.should_estimate(last_frame=0, frame_idx=10) is False
        assert c.should_estimate(last_frame=0, frame_idx=15) is True

    def test_quality_gate(self):
        from vitals import VitalsController
        c = VitalsController(enabled=True, quality_min=2.0)
        assert c.should_report(2.5) is True
        assert c.should_report(1.0) is False

    def test_select_largest_faces(self):
        from vitals import VitalsController
        c = VitalsController(enabled=True, max_people=2)
        # (id, area) pairs; expect the two largest ids returned
        people = [(1, 100.0), (2, 400.0), (3, 250.0)]
        chosen = c.select_people(people, key=lambda p: p[1])
        ids = {p[0] for p in chosen}
        assert ids == {2, 3}

    def test_abnormal_ranges(self):
        from vitals import VitalsController
        c = VitalsController(enabled=True)
        assert c.is_abnormal(hr=72, rr=16) is False
        assert c.is_abnormal(hr=35, rr=16) is True   # bradycardia
        assert c.is_abnormal(hr=72, rr=28) is True   # tachypnea
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsController -v`
Expected: FAIL with `ImportError: cannot import name 'VitalsController'`

- [ ] **Step 3: Implement in vitals.py**

Append:

```python
class VitalsController:
    """Runtime control + gating for the vitals subsystem."""

    def __init__(self, enabled: bool = False, max_people: int = 4,
                 cadence: int = 15, quality_min: float = 2.0,
                 hr_range=(40, 130), rr_range=(8, 25),
                 unresponsive_alert: bool = False, live_gating: bool = False):
        self.enabled = enabled
        self.max_people = max_people
        self.cadence = cadence
        self.quality_min = quality_min
        self.hr_range = hr_range
        self.rr_range = rr_range
        self.unresponsive_alert = unresponsive_alert
        self.live_gating = live_gating

    def should_estimate(self, last_frame: int, frame_idx: int) -> bool:
        return (frame_idx - last_frame) >= self.cadence

    def should_report(self, quality: float) -> bool:
        return quality >= self.quality_min

    def select_people(self, people, key):
        """Return up to max_people items, largest-by-key first."""
        return sorted(people, key=key, reverse=True)[:self.max_people]

    def is_abnormal(self, hr=None, rr=None) -> bool:
        if hr is not None and not (self.hr_range[0] <= hr <= self.hr_range[1]):
            return True
        if rr is not None and not (self.rr_range[0] <= rr <= self.rr_range[1]):
            return True
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsController -v`
Expected: 5 passed

- [ ] **Step 5: Run the FULL suite (regression)**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: all pass (70 baseline + new vitals tests)

- [ ] **Step 6: Commit**

```bash
git add vitals.py tests/test_pipeline.py
git commit -m "feat: VitalsController — enable/cadence/quality/people-cap/abnormal gating"
```

---

### Task 8: PersonState vitals fields + per-person loop integration

**Files:**
- Modify: `stream_detect.py` (`PersonState.__init__` ~line 213; `run_pipeline` per-person loop)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Add PersonState fields**

In `stream_detect.py`, extend `PersonState.__init__` (currently ends at
`self.last_position = np.zeros(2)`):

```python
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.buffer = KeypointBuffer(max_len=256)
        self.smoother = EventSmoother()
        self.last_seen = 0.0   # timestamp
        self.last_position = np.zeros(2)  # hip midpoint (x, y)
        # --- vitals ---
        import collections as _c
        from vitals import HR_WINDOW, RR_WINDOW, LivenessMonitor
        self.vitals_buffer = _c.deque(maxlen=max(HR_WINDOW, RR_WINDOW))
        self.hr_ema = None
        self.rr_ema = None
        self.hr_quality = 0.0
        self.rr_quality = 0.0
        self.last_vitals_frame = -10_000
        self.activity_ema = 0.0
        self.prev_kps = None
        self.liveness = LivenessMonitor()
        self.liveness_state = None
```

- [ ] **Step 2: Write the integration test**

Append to `tests/test_pipeline.py`:

```python
class TestVitalsIntegration:
    def test_person_state_has_vitals_fields(self):
        from stream_detect import PersonState
        s = PersonState(track_id=1)
        assert s.vitals_buffer is not None
        assert s.liveness is not None
        assert s.activity_ema == 0.0

    def test_update_person_vitals_fills_buffer(self):
        import numpy as np
        from stream_detect import PersonState, update_person_vitals
        from vitals import VitalsController, Regime
        ctrl = VitalsController(enabled=True, cadence=1)
        s = PersonState(track_id=1)
        # upright face keypoints, brightening green ROI over time
        for i in range(20):
            kps = np.zeros((17, 3), dtype=np.float32)
            kps[1] = [300, 240, 0.9]; kps[2] = [360, 240, 0.9]
            kps[0] = [330, 264, 0.9]
            kps[5] = [300, 420, 0.9]; kps[6] = [360, 420, 0.9]
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[..., 1] = 100 + int(20 * np.sin(2 * np.pi * 1.2 * i / 30))
            frame[..., 0] = 10
            update_person_vitals(s, kps, frame, Regime.DAY, ctrl,
                                 frame_idx=i, timestamp=i / 30.0, pose_scale=1.0)
        assert len(s.vitals_buffer) > 0
        assert s.liveness_state is not None
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsIntegration -v`
Expected: FAIL with `ImportError: cannot import name 'update_person_vitals'`

- [ ] **Step 4: Implement `update_person_vitals` in stream_detect.py**

Add this helper function above `run_pipeline` (after `apply_context_rules`):

```python
def update_person_vitals(state, kps, frame, regime, controller,
                         frame_idx, timestamp, pose_scale):
    """Update one person's rPPG buffer, vitals estimate, and liveness state.

    Mutates `state` in place. Safe to call every frame; estimation is throttled
    by the controller cadence.
    """
    import vitals

    # activity (always, cheap)
    if state.prev_kps is not None:
        act = vitals.ActivityEstimator.frame_activity(state.prev_kps, kps)
        state.activity_ema = 0.3 * act + 0.7 * state.activity_ema
    state.prev_kps = kps.copy()

    # rPPG sampling into the buffer (only if face ROI valid)
    boxes = vitals.FaceROIExtractor.extract(kps, frame.shape, pose_scale=pose_scale)
    if boxes:
        val = vitals.RPPGEstimator.roi_mean(frame, boxes, regime)
        state.vitals_buffer.append((timestamp, val))

    # throttled estimation
    if controller.should_estimate(state.last_vitals_frame, frame_idx):
        state.last_vitals_frame = frame_idx
        buf = list(state.vitals_buffer)
        if len(buf) >= vitals.HR_WINDOW:
            ts = [b[0] for b in buf]
            vals = [b[1] for b in buf]
            hr_f, hr_q = vitals.RPPGEstimator.estimate(
                ts[-vitals.HR_WINDOW:], vals[-vitals.HR_WINDOW:], vitals.HR_BAND)
            state.hr_quality = hr_q
            if controller.should_report(hr_q):
                hr = hr_f * 60.0
                state.hr_ema = hr if state.hr_ema is None else 0.3 * hr + 0.7 * state.hr_ema
        if len(buf) >= vitals.RR_WINDOW:
            ts = [b[0] for b in buf]
            vals = [b[1] for b in buf]
            rr_f, rr_q = vitals.RPPGEstimator.estimate(ts, vals, vitals.RR_BAND)
            state.rr_quality = rr_q
            if controller.should_report(rr_q):
                rr = rr_f * 60.0
                state.rr_ema = rr if state.rr_ema is None else 0.3 * rr + 0.7 * state.rr_ema

    # liveness fusion
    state.liveness_state = state.liveness.update(
        now=timestamp, is_tracked=True,
        activity=state.activity_ema, pulse_quality=state.hr_quality)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsIntegration -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add stream_detect.py tests/test_pipeline.py
git commit -m "feat: per-person vitals + liveness update in stream_detect"
```

---

### Task 9: CLI flags + wire into run_pipeline (compute path)

**Files:**
- Modify: `stream_detect.py` (`main()` argparse ~line 1239; `run_pipeline` loop ~line 962)

- [ ] **Step 1: Add CLI flags in main()**

In the argparse block of `main()` add:

```python
    parser.add_argument("--vitals", action="store_true",
                        help="Enable contactless HR/RR + liveness estimation")
    parser.add_argument("--vitals-max-people", type=int, default=4,
                        help="Max faces to run rPPG on per frame")
    parser.add_argument("--vitals-cadence", type=int, default=15,
                        help="Estimate vitals every N frames per person")
    parser.add_argument("--vitals-quality-min", type=float, default=2.0,
                        help="Min in-band SNR to report a vitals reading")
    parser.add_argument("--vitals-pose-res", type=int, default=0,
                        help="If >0, downscale frame to this width for pose; "
                             "ROIs still sampled from full-res")
    parser.add_argument("--vitals-csv", default=None,
                        help="Optional CSV path for (ts, track_id, hr, rr, quality)")
    parser.add_argument("--vitals-unresponsive-alert", action="store_true",
                        help="Alert on prolonged PRESENT_STATIC + no pulse")
    parser.add_argument("--live-gating", action="store_true",
                        help="Use liveness (not raw body detection) for apply_context_rules")
```

Also pass these into the controller construction in Step 2:

```python
        unresponsive_alert=getattr(args, "vitals_unresponsive_alert", False),
        live_gating=getattr(args, "live_gating", False),
```

- [ ] **Step 2: Construct the controller in run_pipeline**

Near the top of `run_pipeline`, after the tracker is created, add:

```python
    from vitals import VitalsController, detect_regime, Regime
    vitals_ctrl = VitalsController(
        enabled=getattr(args, "vitals", False),
        max_people=getattr(args, "vitals_max_people", 4),
        cadence=getattr(args, "vitals_cadence", 15),
        quality_min=getattr(args, "vitals_quality_min", 2.0),
    )
    vitals_regime = Regime.DAY
    frame_idx = 0
```

- [ ] **Step 3: Call the vitals update inside the per-person loop**

In `run_pipeline`, inside the loop over `active_people` (after the existing
per-person event-classification work), add:

```python
                if vitals_ctrl.enabled:
                    state_box = (state.last_position, state.buffer)
                    # newest keypoints for this person (full-res coords)
                    cur_kps = state.buffer.latest_keypoints() \
                        if hasattr(state.buffer, "latest_keypoints") else None
                    if cur_kps is not None:
                        update_person_vitals(
                            state, cur_kps, frame_bgr, vitals_regime, vitals_ctrl,
                            frame_idx=frame_idx, timestamp=timestamp,
                            pose_scale=pose_scale)
```

And once per frame, before the per-person loop, set the regime and frame index:

```python
            frame_idx += 1
            if vitals_ctrl.enabled and frame_idx % 30 == 1:
                vitals_regime = detect_regime(frame_bgr)
```

- [ ] **Step 4: Add `latest_keypoints` to KeypointBuffer**

`update_person_vitals` needs the most recent (17,3) keypoints. Add to
`KeypointBuffer` (in `stream_detect.py`):

```python
    def latest_keypoints(self):
        """Return the most recent (17, 3) keypoints, or None if empty."""
        if not self.buffer:
            return None
        return self.buffer[-1].reshape(17, 3)
```

(Confirm the attribute name: read the existing `KeypointBuffer.__init__` to use
its real deque attribute — if it is `self.frames` rather than `self.buffer`,
use that. The push method stores `kps.flatten()`.)

- [ ] **Step 5: Define `pose_scale`**

If `--vitals-pose-res` is 0 (default), pose runs on the full frame, so
`pose_scale = 1.0`. Add near the controller construction:

```python
    pose_scale = 1.0  # full-res pose; >1.0 when --vitals-pose-res downscales
```

(The actual downscaling of the pose input is wired in Task 10; for now
`pose_scale` stays 1.0 so behavior is correct and tests pass.)

- [ ] **Step 6: Smoke-test the CLI parses and `--vitals` off is unchanged**

Run: `.venv/bin/python stream_detect.py --help 2>&1 | grep -- --vitals`
Expected: shows the new flags.

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add stream_detect.py
git commit -m "feat: --vitals CLI flags + per-person vitals call in run_pipeline"
```

---

### Task 10: HUD overlay, vitals log, CSV recorder, downscaled-pose path

**Files:**
- Modify: `stream_detect.py` (`draw_overlay` / a new `draw_vitals`; `run_pipeline`)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test for the log writer**

Append to `tests/test_pipeline.py`:

```python
class TestVitalsLog:
    def test_vitals_log_writer(self, tmp_path):
        from stream_detect import write_vitals_record
        path = tmp_path / "vitals_log.jsonl"
        write_vitals_record(str(path), {
            "track_id": 1, "timestamp": 1.0, "hr_bpm": 72.0, "rr_bpm": 16.0,
            "hr_quality": 3.1, "rr_quality": 2.2, "liveness": "LIVE_CONFIRMED",
            "activity": 0.05, "regime": "day"})
        import json
        line = path.read_text().strip()
        rec = json.loads(line)
        assert rec["track_id"] == 1
        assert rec["hr_bpm"] == 72.0
        assert rec["regime"] == "day"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsLog -v`
Expected: FAIL with `ImportError: cannot import name 'write_vitals_record'`

- [ ] **Step 3: Implement the log writer + HUD draw + downscale**

In `stream_detect.py` add the log writer:

```python
def write_vitals_record(path, record):
    """Append one vitals record as a JSON line."""
    record = dict(record)
    record["written_at"] = datetime.now().isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
```

Add a HUD helper:

```python
LIVENESS_COLORS = {
    "EMPTY": (130, 130, 130),
    "PRESENT_STATIC": (60, 170, 230),
    "PRESENT_MOVING": (230, 170, 60),
    "LIVE_CONFIRMED": (60, 200, 60),
}


def draw_vitals(frame, state):
    """Draw HR/RR + liveness badge near a person's head position."""
    x, y = int(state.last_position[0]), int(state.last_position[1])
    state_name = state.liveness_state.name if state.liveness_state else "EMPTY"
    color = LIVENESS_COLORS.get(state_name, (200, 200, 200))
    hr = f"{state.hr_ema:.0f}" if state.hr_ema else "--"
    rr = f"{state.rr_ema:.0f}" if state.rr_ema else "--"
    label = f"HR {hr} / RR {rr}"
    cv2.putText(frame, label, (x - 40, max(20, y - 60)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.putText(frame, state_name.replace("_", " ").title(), (x - 40, max(36, y - 44)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return frame
```

For the downscaled-pose path: when `args.vitals_pose_res > 0`, build a reduced
frame for the YOLO call and set `pose_scale = full_w / args.vitals_pose_res`.
In `run_pipeline`, where the pose model is invoked, resize the input and record
the scale. (Follow the existing `PoseExtractor` call; pass a resized copy and
multiply returned keypoints' handling via `pose_scale`, which `update_person_vitals`
already applies to ROI coords.) Keep default (0) → full-res, `pose_scale=1.0`.

- [ ] **Step 4: Wire log + HUD into run_pipeline**

After `update_person_vitals(...)` in the loop, add:

```python
                    if state.liveness_state is not None:
                        if args.vitals_csv and state.hr_ema:
                            import csv as _csv
                            with open(args.vitals_csv, "a", newline="") as f:
                                _csv.writer(f).writerow(
                                    [timestamp, state.track_id,
                                     state.hr_ema or "", state.rr_ema or "",
                                     state.hr_quality])
                        write_vitals_record(
                            str(Path(args.output_dir) / "vitals_log.jsonl"), {
                                "track_id": state.track_id, "timestamp": timestamp,
                                "hr_bpm": state.hr_ema, "rr_bpm": state.rr_ema,
                                "hr_quality": state.hr_quality,
                                "rr_quality": state.rr_quality,
                                "liveness": state.liveness_state.name,
                                "activity": state.activity_ema,
                                "regime": vitals_regime.value})
```

In the display block (where `draw_overlay` is called), add when vitals on:

```python
                if vitals_ctrl.enabled:
                    for state in active_people:
                        draw_vitals(display, state)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsLog -v`
Expected: 1 passed

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add stream_detect.py tests/test_pipeline.py
git commit -m "feat: vitals HUD, vitals_log.jsonl, CSV recorder, downscaled-pose path"
```

---

### Task 11: Abnormal-vitals + unresponsiveness alerts; live-gating

**Files:**
- Modify: `stream_detect.py` (`run_pipeline` per-person loop; `apply_context_rules` call site)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
class TestVitalsAlertsAndGating:
    def test_abnormal_vitals_builds_event(self):
        from stream_detect import build_abnormal_vitals_event
        ev = build_abnormal_vitals_event(track_id=3, hr=35, rr=16, timestamp=5.0)
        assert ev["event"] == "abnormal_vitals"
        assert ev["track_id"] == 3
        assert ev["hr_bpm"] == 35

    def test_unresponsive_builds_event(self):
        from stream_detect import build_unresponsive_event
        ev = build_unresponsive_event(track_id=2, timestamp=9.0)
        assert ev["event"] == "unresponsive"
        assert ev["track_id"] == 2

    def test_live_gating_maps_state_to_bool(self):
        from stream_detect import live_person_present
        from vitals import LivenessState
        assert live_person_present(LivenessState.PRESENT_STATIC) is False
        assert live_person_present(LivenessState.PRESENT_MOVING) is True
        assert live_person_present(LivenessState.LIVE_CONFIRMED) is True
        assert live_person_present(None) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsAlertsAndGating -v`
Expected: FAIL with `ImportError: cannot import name 'build_abnormal_vitals_event'`

- [ ] **Step 3: Implement helpers in stream_detect.py**

Add near `update_person_vitals`:

```python
def build_abnormal_vitals_event(track_id, hr, rr, timestamp):
    return {"event": "abnormal_vitals", "track_id": track_id,
            "hr_bpm": hr, "rr_bpm": rr, "timestamp": timestamp,
            "confidence": 1.0}


def build_unresponsive_event(track_id, timestamp):
    return {"event": "unresponsive", "track_id": track_id,
            "timestamp": timestamp, "confidence": 1.0}


def live_person_present(liveness_state) -> bool:
    """Map a liveness state to the boolean apply_context_rules expects."""
    from vitals import LivenessState
    if liveness_state is None:
        return False
    return liveness_state >= LivenessState.PRESENT_MOVING
```

- [ ] **Step 4: Emit alerts + apply live-gating in run_pipeline**

After `update_person_vitals(...)` in the per-person loop, add abnormal +
unresponsiveness emission (deduped by the per-person `EventSmoother` cooldown
pattern is overkill here; use a simple per-state cooldown via `last_seen`-style
guard — emit at most once per `vitals_cadence*4` frames per person):

```python
                    if state.hr_ema and vitals_ctrl.is_abnormal(
                            hr=state.hr_ema, rr=state.rr_ema):
                        alerter.dispatch(build_abnormal_vitals_event(
                            state.track_id, state.hr_ema, state.rr_ema, timestamp))
                    if vitals_ctrl.unresponsive_alert and \
                            state.liveness.unresponsive(timestamp):
                        alerter.dispatch(build_unresponsive_event(
                            state.track_id, timestamp))
```

For live-gating, where `apply_context_rules` is called with `person_detected`,
replace the boolean source when gating is on:

```python
                    if vitals_ctrl.live_gating:
                        person_detected = live_person_present(state.liveness_state)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py::TestVitalsAlertsAndGating -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add stream_detect.py tests/test_pipeline.py
git commit -m "feat: abnormal-vitals + unresponsiveness alerts and live-gating"
```

---

### Task 12: Final verification

- [ ] **Step 1: Full suite**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v 2>&1 | tail -20`
Expected: all pass, zero failures.

- [ ] **Step 2: Confirm `--vitals` off leaves event path unchanged**

Run: `.venv/bin/python -c "import stream_detect, vitals; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 3: Push**

```bash
git push origin HEAD
```

Note: real-world accuracy validation (record `--vitals-csv` against a pulse
oximeter, compare MAE; target HR MAE < 5 bpm at rest, good face visibility) is a
manual step outside this automated plan.
