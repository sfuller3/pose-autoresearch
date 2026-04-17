# Patent Strategy: Pose-Based Event Detection in Elder Care

## Competitive Patent Landscape

### SafelyYou (4 patents)
- **US 11,430,088 B2** — "Systems and methods for fall detection"
  - Claims: continuous video monitoring, AI-based fall detection, automatic
    alert generation, video clip saving around fall events
  - Key limitation: VIDEO-based detection (processes raw pixels/frames)
  - Our differentiation: We use SKELETON-ONLY classification — our Stage 2
    model never sees pixels, only joint coordinates

- **US 11,682,196 B2** — "Fall detection with video review"
  - Claims: fall detection → video review workflow → caregiver notification
  - Our differentiation: We use real-time streaming classification with
    temporal smoothing, not post-hoc video review

- **US 11,756,339 B2** — "Machine learning for fall detection refinement"
  - Claims: ML model trained on facility-specific fall videos, active
    learning loop, false positive reduction via human review
  - Our differentiation: Our model trains on skeleton sequences, not video.
    Our active learning (Roboflow) is for ENVIRONMENT objects, not falls.

- **US 12,002,584 B2** — "Fall detection in memory care environments"
  - Claims: camera placement optimization, multi-camera fusion, memory
    care-specific fall patterns
  - Our differentiation: We are camera-agnostic (works with any YOLO-
    compatible camera), single-camera, and detect 7 event types (not just falls)

**Summary:** SafelyYou's IP is centered on VIDEO-based fall detection with
pixel-level analysis. Our two-stage pipeline (pose extraction → skeleton
classifier) operates in a fundamentally different feature space. Their claims
do not cover skeleton-only classification.

### Inspiren (AUGi device)
- **Technology:** "Geometric Exoskeletal Monitoring (GEM)"
  - Wall-mounted sensor creates 3D skeletal representation
  - Uses depth/IR sensors (not RGB cameras) for privacy
  - Claims focus on the specific sensor hardware and 3D reconstruction

- **Key patents (pending/granted):**
  - Skeletal imaging methods using depth sensors
  - Privacy-preserving monitoring via non-photographic sensing
  - Bed exit prediction from skeletal state changes

- **Our differentiation:**
  - We use standard RGB cameras + software-based pose estimation (YOLO)
  - We do not use depth sensors or custom hardware
  - We extract 2D skeletons (not 3D GEM representations)
  - We classify 7 event types (not just falls and bed exits)
  - Our environmental context comes from a SEPARATE object detection
    model (Roboflow), not from the skeletal sensor itself

**Summary:** Inspiren's IP is tied to their custom depth-sensor hardware
and 3D skeletal reconstruction. Our approach uses commodity cameras with
software-only processing, operating in 2D skeleton space.

---

## Defensive Patent Opportunities

### Patent Claim 1: "Multi-Modal Edge Pipeline for Activity Detection"
**Novel combination:**
- Two-stage pipeline: pose estimation (Stage 1) → temporal classifier (Stage 2)
  running on heterogeneous edge hardware (MemryX MX3 for Stage 1, CPU for Stage 2)
- Environment-aware context injection via parallel object detection model
  that provides spatial relationship features to the skeleton classifier
- FiLM conditioning mechanism that modulates temporal CNN activations
  based on detected furniture/fixtures

**Why novel:** No prior art combines (a) skeleton-only temporal classification
with (b) separate environment object detection providing (c) FiLM-based
feature modulation on (d) heterogeneous edge hardware.

### Patent Claim 2: "Causal Temporal Smoothing for Streaming Skeleton Classification"
**Novel combination:**
- Causal Savitzky-Golay filter on YOLO keypoint streams to reduce
  detection jitter without introducing latency
- EMA probability smoothing with streak counting and per-class cooldown
  for converting noisy per-frame predictions to discrete events
- Confidence gating that attenuates features from low-confidence joints
  before classification

**Why novel:** The specific combination of SG smoothing → confidence gating
→ temporal CNN → EMA+streak event detection is a novel streaming pipeline
not found in prior art.

### Patent Claim 3: "Environment-Conditioned Fall Suppression"
**Novel combination:**
- Object detection model identifies environmental context (bed, chair,
  table, door, wheelchair, walker, handrail)
- Spatial relationship features (distance, relative position, overlap)
  computed between detected person skeleton and environment objects
- Rule-based and learned (FiLM) probability modulation:
  - Person overlapping bed → fall probability suppressed
  - Person near table → eating probability boosted
  - Person near door → wandering probability suppressed
  - Mobility aid detected → unstable gait prior increased

**Why novel:** Using environment object detection to MODULATE skeleton-based
event classification (not replace it) is a novel approach. SafelyYou works
at the pixel level; Inspiren uses custom depth hardware. Neither uses a
separate environment model to condition a skeleton classifier.

### Patent Claim 4: "Facility-Adaptive Retraining via Active Learning"
**Novel combination:**
- Roboflow-based annotation and retraining pipeline for environment
  objects specific to each facility deployment
- Transfer learning: base model pre-trained on generic furniture,
  fine-tuned on facility-specific objects
- Environment model updates WITHOUT retraining the event classifier
  (modular architecture allows independent model updates)

---

## IP Development Timeline

| Phase | Timeframe | Action |
|-------|-----------|--------|
| 1. Document | Now | Write detailed technical descriptions of all novel methods |
| 2. Provisional | 1-2 months | File provisional patent applications (12-month priority) |
| 3. Prior art search | 2-3 months | Engage patent attorney for formal prior art analysis |
| 4. Utility filing | 6-12 months | Convert provisionals to utility applications |
| 5. Continuation | Ongoing | File continuations as architecture evolves |

## Key Principles
1. **Don't claim what SafelyYou claims** — avoid video-based fall detection claims
2. **Don't claim what Inspiren claims** — avoid depth sensor / 3D skeleton claims
3. **DO claim the combination** — two-stage skeleton + environment context is novel
4. **DO claim the edge deployment** — heterogeneous hardware pipeline is novel
5. **DO claim the streaming pipeline** — SG smoothing + EMA + streak is novel
6. **File provisionals early** — establish priority date before publishing
