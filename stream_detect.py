#!/usr/bin/env python3
"""
Real-time streaming pose event detection.

Takes video input (camera, RTSP, or file), runs YOLO pose estimation,
feeds keypoints into the causal hybrid CNN-Transformer model, and
detects events in real-time.

On event detection:
  - Logs to events/event_log.jsonl
  - Saves a video clip (pre-roll + event + post-roll)
  - Prints alert to stdout (or sends webhook)

Usage:
  python stream_detect.py --source 0                    # webcam
  python stream_detect.py --source video.mp4            # file
  python stream_detect.py --source rtsp://...           # RTSP stream
  python stream_detect.py --source video.mp4 --display  # with live overlay
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import savgol_coeffs

from prepare import BONE_PAIRS, CLASS_TO_IDX, EVENT_CLASSES, INPUT_DIM, NUM_BONES, NUM_KEYPOINTS
from train_hybrid import HybridPoseDetector


# ============================================================================
# DEVICE
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# VIDEO SOURCE
# ============================================================================

class VideoSource:
    """Video source supporting camera, RTSP, or file input.

    Uses decord for file playback (handles corrupted AVI audio headers).
    Falls back to cv2 for cameras and RTSP streams.
    """

    def __init__(self, source):
        self._is_camera = False
        self._decord_vr = None
        self._cv2_cap = None

        # Camera index or RTSP → use cv2
        try:
            source_int = int(source)
            self._is_camera = True
            self._cv2_cap = cv2.VideoCapture(source_int)
            if not self._cv2_cap.isOpened():
                raise RuntimeError(f"Cannot open camera: {source_int}")
            self._fps = self._cv2_cap.get(cv2.CAP_PROP_FPS) or 30.0
        except (ValueError, TypeError):
            if str(source).startswith(("rtsp://", "http://", "https://")):
                self._is_camera = True
                self._cv2_cap = cv2.VideoCapture(source)
                if not self._cv2_cap.isOpened():
                    raise RuntimeError(f"Cannot open stream: {source}")
                self._fps = self._cv2_cap.get(cv2.CAP_PROP_FPS) or 30.0
            else:
                # File → use decord
                from decord import VideoReader, cpu
                self._decord_vr = VideoReader(str(source), ctx=cpu(0))
                self._fps = float(self._decord_vr.get_avg_fps()) or 30.0

        self._frame_count = 0

    @property
    def fps(self):
        return self._fps

    def __iter__(self):
        return self

    def __next__(self):
        if self._decord_vr is not None:
            if self._frame_count >= len(self._decord_vr):
                raise StopIteration
            frame = self._decord_vr[self._frame_count].asnumpy()
            # decord returns RGB, convert to BGR for cv2 compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self._cv2_cap.read()
            if not ret:
                raise StopIteration

        timestamp = self._frame_count / self._fps
        self._frame_count += 1
        return frame, timestamp

    def release(self):
        if self._cv2_cap is not None:
            self._cv2_cap.release()
        self._decord_vr = None


# ============================================================================
# POSE EXTRACTION
# ============================================================================

class PoseExtractor:
    """Extracts COCO-17 keypoints from a video frame using YOLO."""

    def __init__(self, model_path: str, device: torch.device, conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Returns (17, 3) keypoints [x, y, conf] for the primary person, or None."""
        result = self.model(frame_bgr, verbose=False, conf=self.conf)[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # Select person with largest bounding box
        if len(result.boxes) > 1:
            areas = (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]) * \
                    (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1])
            idx = areas.argmax().item()
        else:
            idx = 0

        kp_xy = result.keypoints.xy[idx].cpu().numpy()    # (17, 2)
        kp_conf = result.keypoints.conf[idx].cpu().numpy()  # (17,)
        return np.concatenate([kp_xy, kp_conf[:, None]], axis=1)  # (17, 3)


# ============================================================================
# ENVIRONMENT DETECTION (Roboflow)
# ============================================================================

class EnvironmentDetector:
    """Roboflow-based object detector for environmental context.

    Detects furniture (beds, chairs, tables) and fixtures (doors, handrails)
    to provide spatial context to the event classifier.

    Uses Roboflow Inference SDK for edge-optimized inference.
    """

    # Object classes that affect event classification
    CONTEXT_CLASSES = {
        "bed", "chair", "table", "door", "wheelchair",
        "walker", "handrail", "floor-area",
    }
    NUM_CONTEXT_CLASSES = len(CONTEXT_CLASSES)

    def __init__(
        self,
        model_id: str,
        api_key: str | None = None,
        conf: float = 0.3,
        infer_interval: int = 15,  # run every N frames (not every frame)
    ):
        from inference import get_model
        self.model = get_model(model_id=model_id, api_key=api_key)
        self.conf = conf
        self.infer_interval = infer_interval
        self._frame_count = 0
        self._cached_detections: list[dict] = []
        self._class_list = sorted(self.CONTEXT_CLASSES)
        self._class_to_idx = {c: i for i, c in enumerate(self._class_list)}

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Run object detection (or return cached result if not due).

        Returns list of dicts: [{"class": str, "bbox": [x1,y1,x2,y2],
                                  "confidence": float}, ...]
        """
        self._frame_count += 1
        if self._frame_count % self.infer_interval != 1 and self._cached_detections:
            return self._cached_detections

        results = self.model.infer(frame_bgr, confidence=self.conf)

        detections = []
        if hasattr(results, "predictions"):
            for pred in results.predictions:
                cls_name = pred.class_name.lower().replace(" ", "-")
                if cls_name in self.CONTEXT_CLASSES:
                    detections.append({
                        "class": cls_name,
                        "bbox": [pred.x - pred.width/2, pred.y - pred.height/2,
                                 pred.x + pred.width/2, pred.y + pred.height/2],
                        "confidence": pred.confidence,
                    })

        self._cached_detections = detections
        return detections

    def compute_spatial_features(
        self,
        detections: list[dict],
        keypoints: np.ndarray | None,
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        """Compute spatial relationship features between person and objects.

        Returns a fixed-size feature vector encoding:
        - Per-class: present (0/1), nearest distance to person center,
          relative position (above/below/left/right), overlap ratio
        - Total: NUM_CONTEXT_CLASSES * 4 = 32 features

        Args:
            detections: output from detect()
            keypoints: (17, 3) array or None
            frame_shape: (height, width)
        """
        h, w = frame_shape
        num_features = self.NUM_CONTEXT_CLASSES * 4
        features = np.zeros(num_features, dtype=np.float32)

        if keypoints is None or len(detections) == 0:
            return features

        # Person center (hip midpoint if available, else bbox center)
        hip_l = keypoints[11, :2]
        hip_r = keypoints[12, :2]
        if keypoints[11, 2] > 0.1 and keypoints[12, 2] > 0.1:
            person_center = (hip_l + hip_r) / 2
        else:
            valid = keypoints[keypoints[:, 2] > 0.1, :2]
            if len(valid) == 0:
                return features
            person_center = valid.mean(axis=0)

        px, py = person_center / np.array([w, h])  # normalize to [0,1]

        for det in detections:
            cls = det["class"]
            if cls not in self._class_to_idx:
                continue
            idx = self._class_to_idx[cls]
            x1, y1, x2, y2 = np.array(det["bbox"]) / np.array([w, h, w, h])
            obj_cx, obj_cy = (x1 + x2) / 2, (y1 + y2) / 2

            base = idx * 4
            features[base + 0] = 1.0  # present
            dist = np.sqrt((px - obj_cx)**2 + (py - obj_cy)**2)
            features[base + 1] = max(0, 1.0 - dist)  # proximity (1=close, 0=far)
            features[base + 2] = py - obj_cy  # relative Y (-1=person above, +1=below)
            features[base + 3] = px - obj_cx  # relative X (-1=person left, +1=right)

        return features


# ============================================================================
# KEYPOINT BUFFER
# ============================================================================

class KeypointBuffer:
    """Ring buffer of keypoints with Savitzky-Golay temporal smoothing.

    Applies a causal Savitzky-Golay filter to smooth jittery keypoint
    positions and interpolate over missing detections (zero frames).
    Only x,y coordinates are smoothed — confidence values pass through raw.
    """

    def __init__(self, max_len: int = 256, sg_window: int = 7, sg_polyorder: int = 2):
        self.max_len = max_len
        self.buffer = collections.deque(maxlen=max_len)
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        # Precompute causal SG coefficients (only use past + current frames)
        # savgol_coeffs returns coefficients for centered filter; we shift
        # to make it causal by using pos=window_length-1
        self._sg_coeffs = savgol_coeffs(sg_window, sg_polyorder, pos=sg_window - 1)

    def push(self, keypoints_flat: np.ndarray):
        """Push (51,) flattened keypoints."""
        self.buffer.append(keypoints_flat.astype(np.float32))

    def get_tensor(self, device: torch.device) -> torch.Tensor:
        """Return (1, T, 51) smoothed tensor from current buffer."""
        arr = np.stack(list(self.buffer), axis=0)  # (T, 51)
        arr = self._smooth(arr)
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        """Apply causal Savitzky-Golay filter to x,y coordinates.

        Smooths each joint's x and y independently. Confidence (every 3rd
        value) is left unsmoothed. Missing frames (all zeros) are interpolated
        before filtering.
        """
        T, D = arr.shape
        if T < self.sg_window:
            return arr  # not enough frames yet

        smoothed = arr.copy()
        coeffs = self._sg_coeffs

        # Process x,y for each of the 17 joints (skip confidence at idx 2,5,8,...)
        for j in range(17):
            for offset in range(2):  # 0=x, 1=y
                col_idx = j * 3 + offset
                signal = arr[:, col_idx].copy()

                # Interpolate over zero gaps (missing detections)
                nonzero = np.nonzero(signal)[0]
                if len(nonzero) > 1:
                    zero_mask = signal == 0.0
                    if zero_mask.any():
                        signal[zero_mask] = np.interp(
                            np.where(zero_mask)[0], nonzero, signal[nonzero]
                        )

                # Apply causal SG filter via convolution
                # For frames before sg_window, use raw values
                for t in range(self.sg_window - 1, T):
                    window = signal[t - self.sg_window + 1 : t + 1]
                    smoothed[t, col_idx] = np.dot(coeffs, window)

        return smoothed

    def __len__(self):
        return len(self.buffer)


# ============================================================================
# STREAMING DETECTOR
# ============================================================================

class StreamingDetector:
    """Loads the hybrid model and runs per-frame inference."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.model = HybridPoseDetector().to(device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, kp_buffer: KeypointBuffer) -> np.ndarray:
        """Return softmax probabilities (7,) for the latest frame."""
        x = kp_buffer.get_tensor(self.device)  # (1, T, 51)
        lengths = torch.tensor([x.shape[1]], device=self.device)
        outputs = self.model(x, lengths=lengths)
        logits = outputs["fused"][0, -1]  # last frame prediction
        return F.softmax(logits, dim=0).cpu().numpy()


# ============================================================================
# EVENT SMOOTHER
# ============================================================================

class EventSmoother:
    """Converts noisy per-frame probabilities into discrete event detections.

    Uses exponential moving average + streak counting + cooldown.
    """

    def __init__(
        self,
        num_classes: int = len(EVENT_CLASSES),
        alpha: float = 0.3,
        threshold: float = 0.6,
        min_frames: int = 15,
        cooldown_frames: int = 300,
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.min_frames = min_frames
        self.cooldown_frames = cooldown_frames

        self.ema = np.ones(num_classes) / num_classes
        self.streak = np.zeros(num_classes, dtype=int)
        self.cooldown = np.zeros(num_classes, dtype=int)

    def update(self, probs: np.ndarray) -> tuple[int, float] | None:
        """Returns (class_idx, confidence) if event detected, else None."""
        # Update EMA
        self.ema = self.alpha * probs + (1 - self.alpha) * self.ema

        # Tick cooldowns
        self.cooldown = np.maximum(0, self.cooldown - 1)

        # Find dominant class
        dominant = int(np.argmax(self.ema))
        confidence = float(self.ema[dominant])

        # Reset non-dominant streaks
        mask = np.ones(len(self.streak), dtype=bool)
        mask[dominant] = False
        self.streak[mask] = 0

        # Check threshold and cooldown
        if confidence >= self.threshold and self.cooldown[dominant] == 0:
            self.streak[dominant] += 1
            if self.streak[dominant] >= self.min_frames:
                # Event detected
                self.cooldown[dominant] = self.cooldown_frames
                self.streak[dominant] = 0
                return dominant, confidence

        return None

    @property
    def current_class(self) -> int:
        return int(np.argmax(self.ema))

    @property
    def current_confidence(self) -> float:
        return float(self.ema.max())


# ============================================================================
# CLIP RECORDER
# ============================================================================

class ClipRecorder:
    """Maintains a rolling frame buffer and saves clips on event detection."""

    def __init__(
        self,
        pre_roll: int = 150,
        post_roll: int = 90,
        fps: float = 30.0,
        output_dir: str = "events",
    ):
        self.pre_roll = pre_roll
        self.post_roll = post_roll
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        buffer_size = pre_roll + post_roll + 300  # extra headroom
        self.frame_buffer = collections.deque(maxlen=buffer_size)

        self._pending = None  # (class_name, confidence, timestamp, frames_remaining)
        self._event_start_idx = 0

    def push_frame(self, frame_bgr: np.ndarray, timestamp: float):
        self.frame_buffer.append((frame_bgr.copy(), timestamp))

    def start_clip(self, class_name: str, confidence: float, timestamp: float):
        """Mark an event — will save clip after post-roll completes."""
        if self._pending is not None:
            return  # already recording
        self._pending = {
            "class": class_name,
            "confidence": confidence,
            "timestamp": timestamp,
            "post_remaining": self.post_roll,
            "trigger_idx": len(self.frame_buffer) - 1,
        }

    def tick(self) -> dict | None:
        """Call each frame. Returns event dict with clip_path when clip is ready."""
        if self._pending is None:
            return None

        self._pending["post_remaining"] -= 1
        if self._pending["post_remaining"] > 0:
            return None

        # Post-roll complete — write clip
        event = self._pending
        self._pending = None

        clip_path = self._write_clip(event)
        return {
            "class": event["class"],
            "confidence": event["confidence"],
            "timestamp": event["timestamp"],
            "clip_path": str(clip_path),
        }

    def _write_clip(self, event: dict) -> Path:
        """Write buffered frames to a video file."""
        trigger_idx = event["trigger_idx"]
        start_idx = max(0, trigger_idx - self.pre_roll)

        frames = list(self.frame_buffer)
        clip_frames = frames[start_idx:]

        if not clip_frames:
            clip_frames = frames  # fallback

        ts = datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        filename = f"{ts}_{event['class']}_{event['confidence']:.2f}.avi"
        clip_path = self.output_dir / filename

        h, w = clip_frames[0][0].shape[:2]
        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            self.fps,
            (w, h),
        )
        for frame, _ in clip_frames:
            writer.write(frame)
        writer.release()

        return clip_path


# ============================================================================
# ALERT DISPATCHER
# ============================================================================

class AlertDispatcher:
    """Logs events and optionally sends webhook alerts."""

    def __init__(self, output_dir: str = "events", webhook_url: str | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "event_log.jsonl"
        self.webhook_url = webhook_url

    def dispatch(self, event: dict):
        """Log event and send alert."""
        event["detected_at"] = datetime.now().isoformat()

        # Append to JSONL log
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

        # Console alert
        print(f"\n{'='*60}")
        print(f"  EVENT DETECTED: {event['class'].upper()}")
        print(f"  Confidence: {event['confidence']:.1%}")
        print(f"  Video time: {event['timestamp']:.1f}s")
        if event.get("clip_path"):
            print(f"  Clip saved: {event['clip_path']}")
        print(f"{'='*60}\n")

        # Webhook (if configured)
        if self.webhook_url:
            self._send_webhook(event)

    def _send_webhook(self, event: dict):
        try:
            import urllib.request
            data = json.dumps(event).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"  Webhook failed: {e}")


# ============================================================================
# DISPLAY OVERLAY
# ============================================================================

# Colors for environment objects (BGR)
ENV_COLORS = {
    "bed":        (200, 150, 50),   # teal
    "chair":      (50, 200, 50),    # green
    "table":      (50, 150, 200),   # amber
    "door":       (200, 100, 200),  # pink
    "wheelchair": (100, 200, 200),  # yellow
    "walker":     (200, 200, 100),  # cyan
    "handrail":   (100, 100, 200),  # salmon
    "floor-area": (150, 150, 150),  # gray
}


def draw_environment(
    frame: np.ndarray,
    detections: list[dict],
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw environment object bounding boxes on frame."""
    if not detections:
        return frame

    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        cls = det["class"]
        bbox = det["bbox"]
        conf = det["confidence"]
        color = ENV_COLORS.get(cls, (180, 180, 180))

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        label = f"{cls} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_overlay(frame: np.ndarray, smoother: EventSmoother, fps: float) -> np.ndarray:
    """Draw detection status overlay on frame."""
    overlay = frame.copy()

    # Current prediction bar
    class_idx = smoother.current_class
    class_name = EVENT_CLASSES[class_idx]
    confidence = smoother.current_confidence

    # Color: red for fall/aggression, yellow for unstable, green for normal
    colors = {
        "fall": (0, 0, 255),
        "aggression": (0, 0, 200),
        "unstable_gait": (0, 200, 255),
        "wandering": (0, 180, 255),
        "eating": (0, 200, 0),
        "sitting_standing": (0, 200, 0),
        "working_together": (200, 200, 0),
    }
    color = colors.get(class_name, (200, 200, 200))

    # Background bar
    cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, overlay)

    # Text
    cv2.putText(overlay, f"{class_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(overlay, f"{confidence:.0%} | {fps:.0f} fps", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Confidence bar
    bar_w = int(300 * confidence)
    cv2.rectangle(overlay, (20, 72), (20 + bar_w, 78), color, -1)

    return overlay


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(args):
    device = get_device()
    print(f"Device: {device}")

    # Initialize components
    print("Loading YOLO pose model...")
    source = VideoSource(args.source)
    pose = PoseExtractor(args.yolo_model, device, conf=args.pose_conf)

    env_detector = None
    if args.env_model:
        api_key = args.roboflow_key or os.environ.get("ROBOFLOW_API_KEY")
        print(f"  Loading environment detector: {args.env_model}")
        env_detector = EnvironmentDetector(
            model_id=args.env_model,
            api_key=api_key,
            infer_interval=args.env_interval,
        )

    buffer = KeypointBuffer(max_len=256, sg_window=args.sg_window, sg_polyorder=args.sg_polyorder)

    print("Loading event detection model...")
    detector = StreamingDetector(args.checkpoint, device)

    smoother = EventSmoother(
        alpha=args.ema_alpha,
        threshold=args.threshold,
        min_frames=int(args.min_duration * source.fps),
        cooldown_frames=int(args.cooldown * source.fps),
    )
    clip_recorder = ClipRecorder(
        pre_roll=int(args.pre_roll * source.fps),
        post_roll=int(args.post_roll * source.fps),
        fps=source.fps,
        output_dir=args.output_dir,
    )
    alerter = AlertDispatcher(output_dir=args.output_dir, webhook_url=args.webhook)

    print(f"Source: {args.source} ({source.fps:.0f} fps)")
    print(f"Threshold: {args.threshold}, EMA alpha: {args.ema_alpha}")
    print(f"Min duration: {args.min_duration}s, Cooldown: {args.cooldown}s")
    print(f"Clip pre-roll: {args.pre_roll}s, post-roll: {args.post_roll}s")
    print(f"Output: {args.output_dir}/")
    print()
    print("Streaming... (Ctrl+C to stop)")
    print()

    frame_times = collections.deque(maxlen=30)
    events_detected = 0

    try:
        for frame_bgr, timestamp in source:
            t0 = time.time()

            # 1. Pose estimation
            kps = pose.extract(frame_bgr)

            # Environment context (optional)
            env_detections = []
            env_features = np.zeros(EnvironmentDetector.NUM_CONTEXT_CLASSES * 4,
                                    dtype=np.float32)
            if env_detector is not None:
                env_detections = env_detector.detect(frame_bgr)
                env_features = env_detector.compute_spatial_features(
                    env_detections, kps, frame_bgr.shape[:2]
                )

            kps_flat = kps.flatten() if kps is not None else np.zeros(INPUT_DIM, dtype=np.float32)

            # 2. Buffer keypoints
            buffer.push(kps_flat)

            # 3. Record frame for clips
            clip_recorder.push_frame(frame_bgr, timestamp)

            # 4. Model inference (need minimum context)
            if len(buffer) < args.min_context:
                if args.display:
                    cv2.imshow("Stream Detect", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            probs = detector.predict(buffer)

            # 5. Event smoothing
            event = smoother.update(probs)

            # 6. Handle events
            if event:
                class_idx, confidence = event
                class_name = EVENT_CLASSES[class_idx]
                clip_recorder.start_clip(class_name, confidence, timestamp)
                events_detected += 1

            clip_event = clip_recorder.tick()
            if clip_event:
                alerter.dispatch(clip_event)

            # 7. Display
            frame_times.append(time.time() - t0)
            current_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0

            if args.display:
                # Draw environment objects
                if env_detector is not None and env_detections:
                    draw_environment(frame_bgr, env_detections)

                display = draw_overlay(frame_bgr, smoother, current_fps)
                cv2.imshow("Stream Detect", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Status line (every 30 frames)
            if len(buffer) % 30 == 0:
                cls = EVENT_CLASSES[smoother.current_class]
                conf = smoother.current_confidence
                person = "✓" if kps is not None else "✗"
                print(
                    f"\r  t={timestamp:6.1f}s | {current_fps:4.0f} fps | "
                    f"person={person} | {cls}={conf:.0%} | "
                    f"events={events_detected}",
                    end="", flush=True,
                )

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    finally:
        source.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"\nTotal events detected: {events_detected}")
    print(f"Event log: {args.output_dir}/event_log.jsonl")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Real-time streaming pose event detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True,
                        help="Video source: camera index (0), file path, or RTSP URL")
    parser.add_argument("--checkpoint", default="checkpoints/best_hybrid.pt",
                        help="Model checkpoint path")
    parser.add_argument("--yolo-model", default="yolo11s-pose.pt",
                        help="YOLO pose model path")
    parser.add_argument("--pose-conf", type=float, default=0.25,
                        help="YOLO confidence threshold")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Event detection threshold")
    parser.add_argument("--ema-alpha", type=float, default=0.3,
                        help="EMA smoothing factor (higher = more responsive)")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum event duration in seconds")
    parser.add_argument("--cooldown", type=float, default=10.0,
                        help="Cooldown between same-class alerts (seconds)")
    parser.add_argument("--pre-roll", type=float, default=5.0,
                        help="Seconds of video before event in clip")
    parser.add_argument("--post-roll", type=float, default=3.0,
                        help="Seconds of video after event in clip")
    parser.add_argument("--min-context", type=int, default=10,
                        help="Minimum frames before inference starts")
    parser.add_argument("--output-dir", default="events",
                        help="Output directory for clips and logs")
    parser.add_argument("--webhook", default=None,
                        help="Webhook URL for alert notifications")
    parser.add_argument("--sg-window", type=int, default=7,
                        help="Savitzky-Golay filter window size (odd, >= 3)")
    parser.add_argument("--sg-polyorder", type=int, default=2,
                        help="Savitzky-Golay polynomial order")
    parser.add_argument("--display", action="store_true",
                        help="Show live video with detection overlay")
    parser.add_argument("--env-model", default=None,
                        help="Roboflow model ID for environment detection (e.g. 'elder-care/1')")
    parser.add_argument("--roboflow-key", default=None,
                        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--env-interval", type=int, default=15,
                        help="Run environment detection every N frames")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
