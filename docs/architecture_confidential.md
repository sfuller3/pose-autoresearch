# CONFIDENTIAL — System Architecture Detail

**Classification: Internal use only. Do not distribute.**

---

## Two-Stage Pipeline

### Stage 1: Pose Estimation

YOLO11-pose extracts 17 COCO keypoints per frame at 30fps. Each keypoint produces (x, y, confidence) — 51 floats per frame. On edge hardware, this runs on a MemryX MX3 neural accelerator (<2W power draw).

### Stage 1.5: Signal Conditioning

- **Causal Savitzky-Golay filter** on the keypoint stream reduces YOLO detection jitter without introducing latency (uses only past frames, configurable window/polynomial order)
- **Confidence gating** applies a steep sigmoid gate `sigmoid(conf * 5 - 2)` to attenuate x,y coordinates from low-confidence detections. This gracefully suppresses noisy joints rather than zeroing them (conf=0.2 retains ~12% signal)

### Stage 2: Temporal CNN Classifier (PoseEventClassifier)

**Input expansion** (51 → 150 features per frame):
- Joints: 17 x 3 = 51 (gated x, y, raw confidence)
- Bones: 16 bone vectors x 3 = 48 (dx, dy between connected joints from gated coords, averaged confidence from raw)
- Velocity: 17 x 3 = 51 (frame-to-frame displacement on raw coords, preserving confidence channel)

**Architecture** (1,761,460 parameters at env_dim=0; 2,116,020 at env_dim=32):
1. BatchNorm1d on 150-dim input
2. 4 x MultiScaleTemporalBlock — parallel Conv1d at kernel sizes 3, 7, 15:
   - Kernel 3: fast actions (fall impact ~0.3s, 9 frames)
   - Kernel 7: medium motions (eating cycles ~1s, 30 frames)
   - Kernel 15: slow patterns (wandering ~5s, 150 frames)
   - Each block includes Squeeze-and-Excitation channel attention (AdaptiveAvgPool1d → FC → SiLU → FC → Sigmoid, reduction=4)
   - Channel progression: 150→128, 128→128, 128→256 (stride 2), 256→256
   - Residual connections with 1x1 conv when dimensions change
3. Temporal Attention Pooling — Bahdanau-style additive attention over frames (Linear→Tanh→Linear→Softmax), replacing global average pooling. Learns which frames are most discriminative.
4. Linear(256, 7) → logits

**Training configuration:**
- Focal loss: `(1-pt)^gamma * CE` with gamma=2.0, label_smoothing=0.1
  - pt computed via explicit `log_softmax.gather().exp()` — decoupled from class weights and smoothing
- Dynamic class weighting: inverse-frequency from training set distribution, with 1.5x fall-priority boost
- Temporal MixUp: `lam ~ Beta(0.2, 0.2)`, applied to 50% of batches (coin flip), lambda-weighted loss against both label sets. Uses torch RNG for reproducibility with `torch.manual_seed`.
- AdamW optimizer, lr=2e-3, weight_decay=1e-4
- CosineAnnealingLR scheduler, T_max=50, eta_min=1e-6
- Gradient clipping at norm 1.0

### Environment Context (Roboflow Integration)

**Environment detection** runs a Roboflow-trained object detector every 15th frame (amortized cost ~20ms). Detects 8 object classes: bed, chair, table, door, wheelchair, walker, handrail, floor-area.

**Spatial features** (32-dim vector): For each of the 8 classes, 4 features are computed relative to the person's hip-midpoint:
- Present (0/1)
- Proximity (1 - euclidean distance, normalized to frame dimensions)
- Relative Y (person above/below object)
- Relative X (person left/right of object)

**FiLM conditioning** (Feature-wise Linear Modulation): Per temporal block, a 2-layer MLP (env_dim → channel_dim → channel_dim * 2) produces gamma (scale, centered at 1) and beta (shift) vectors. Applied channel-wise after each temporal block: `x' = gamma * x + beta`. This allows the environment context to modulate how the CNN processes skeleton motion at each layer depth. Currently infrastructure-only — activates when facility-specific environment training data is collected.

### Stage 3: Event Detection Pipeline (Streaming)

1. **Context rules** (heuristic, environment-aware):
   - No person → suppress fall (0.1x) and aggression (0.1x)
   - Person near bed (proximity > 0.5) → suppress fall (0.3x), boost sitting_standing (1.5x)
   - Person near table (proximity > 0.4) → boost eating (1.8x)
   - Person near door (proximity > 0.5) → suppress wandering (0.4x)
   - Walker or wheelchair detected → boost unstable_gait (1.5x)
   - Renormalize after adjustments
2. **EMA smoothing** (alpha=0.3)
3. **Streak counting** — minimum consecutive frames above threshold before trigger
4. **Per-class cooldown** — configurable seconds between repeated alerts of same class
5. **Event trigger** → JSONL log entry, video clip (5s pre-roll + 3s post-roll), webhook alert

## 7 Event Classes

| Index | Class | Temporal Signature |
|-------|-------|-------------------|
| 0 | fall | Sudden downward acceleration + ground-level posture |
| 1 | eating | Repetitive hand-to-mouth oscillation near table height |
| 2 | working_together | Multiple-person coordinated movement patterns |
| 3 | aggression | Rapid, erratic multi-person limb movements |
| 4 | unstable_gait | Irregular step cadence, lateral sway, asymmetric limb motion |
| 5 | wandering | Sustained locomotion without directional purpose |
| 6 | sitting_standing | Vertical center-of-mass transitions |

## Training Data Sources

- **NTU RGB+D 120** (pickle format): All 7 classes, cross-subject train/val/test split using official 53 training subjects
- **FallVision** (CSV keypoints): Fall + sitting_standing classes, video-level grouping to prevent data leakage
- **Le2i** (video, YOLO extraction): Fall class, video-level grouping
- Source-aware splitting via `scripts/split_data.py` with symlinks to save disk

## Edge Deployment

| Component | Hardware | Latency | Power |
|-----------|----------|---------|-------|
| YOLO11-pose (Stage 1) | MemryX MX3 | ~33ms (30fps) | <2W |
| Temporal CNN (Stage 2) | CPU | <5ms/sequence | negligible |
| Roboflow env detection | CPU/GPU | ~20ms every 15th frame | variable |
| **Total end-to-end** | | **<50ms** | **<2W additional** |

## Differentiation from Competitors

- **vs SafelyYou**: They process raw video pixels. We classify skeleton-only — fundamentally different feature space.
- **vs Inspiren/AUGi**: They use custom depth/IR sensor hardware for 3D skeletal reconstruction. We use commodity RGB + software-only 2D pose estimation.
- **Novel combination**: Two-stage skeleton pipeline + separate environment object detection + FiLM conditioning + edge deployment on heterogeneous hardware.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Model definition, training loop, focal loss, MixUp |
| `stream_detect.py` | Streaming pipeline, EnvironmentDetector, context rules, HUD |
| `prepare.py` | Dataset loading, PoseDataset class (immutable) |
| `train_hybrid.py` | Hybrid CNN-Transformer variant (separate model) |
| `scripts/setup_roboflow.py` | Roboflow project initialization |
| `scripts/extract_env_features.py` | Batch env feature extraction for training |
| `scripts/split_data.py` | Source-aware train/val/test splitting |
| `tests/test_pipeline.py` | 50 unit tests |
