# System Architecture — Patent Reference

## Figure 1: Two-Stage Pipeline with Environment Context

```
┌─────────────┐
│  RGB Camera  │
└──────┬──────┘
       │ frame (H×W×3)
       │
       ├───────────────────────────┐
       │                           │
       ▼                           ▼
┌──────────────┐          ┌────────────────┐
│  YOLO Pose   │          │  Roboflow Obj  │
│  (MemryX     │          │  Detection     │
│   MX3)       │          │  (CPU/GPU)     │
│              │          │                │
│  17 joints   │          │  furniture +   │
│  × (x,y,c)  │          │  fixtures      │
└──────┬───────┘          └───────┬────────┘
       │ 51 floats                │ N × (class, bbox, conf)
       │                          │
       ▼                          │
┌──────────────┐                  │
│  Causal SG   │                  │
│  Temporal    │                  │
│  Smoothing   │                  │
└──────┬───────┘                  │
       │                          │
       ▼                          ▼
┌──────────────┐          ┌────────────────┐
│  Feature     │          │  Spatial       │
│  Extraction  │          │  Relationship  │
│              │          │  Features      │
│  joints: 51  │          │                │
│  bones:  48  │          │  per-class:    │
│  velocity:51 │          │  present,      │
│  ──────────  │          │  proximity,    │
│  total: 150  │          │  relative pos  │
└──────┬───────┘          └───────┬────────┘
       │                          │
       ▼                          ▼
┌─────────────────────────────────────────┐
│  Context-Conditioned Temporal CNN       │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  MultiScale Block 1 (3,7,15)   │◄─FiLM── env features
│  │  128 ch + SE attention          │    │
│  ├─────────────────────────────────┤    │
│  │  MultiScale Block 2 (3,7,15)   │◄─FiLM── env features
│  │  128 ch + SE attention          │    │
│  ├─────────────────────────────────┤    │
│  │  MultiScale Block 3 (3,7,15)   │◄─FiLM── env features
│  │  256 ch, stride 2 + SE          │    │
│  ├─────────────────────────────────┤    │
│  │  MultiScale Block 4 (3,7,15)   │◄─FiLM── env features
│  │  256 ch + SE attention          │    │
│  ├─────────────────────────────────┤    │
│  │  Temporal Attention Pooling     │    │
│  ├─────────────────────────────────┤    │
│  │  Linear → 7 classes             │    │
│  └─────────────────────────────────┘    │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Event Detection Pipeline               │
│                                         │
│  Confidence Gating                      │
│       ↓                                 │
│  Context Rules (bed→no fall, etc.)      │
│       ↓                                 │
│  EMA Smoothing (α=0.3)                  │
│       ↓                                 │
│  Streak Counting (min 15 frames)        │
│       ↓                                 │
│  Per-Class Cooldown (300 frames)        │
│       ↓                                 │
│  Event Trigger → Alert + Clip           │
└─────────────────────────────────────────┘
```

## Figure 2: FiLM Conditioning Detail

```
Environment Features (32-dim)
    │
    ▼
┌──────────┐     ┌──────────────┐
│ FC → SiLU │────►│ FC → γ, β   │
│ (32→C)   │     │ (C → 2C)     │
└──────────┘     └──────┬───────┘
                        │
                   γ (scale), β (shift)
                        │
Temporal Block output ──┤
    x: (B, C, T)        │
                        ▼
              x' = γ · x + β
              (channel-wise modulation)
```

## Figure 3: Edge Deployment Configuration

```
┌──────────────────────────────────────────┐
│  Edge Device                              │
│                                           │
│  ┌───────────┐    ┌──────────────────┐   │
│  │ MemryX    │    │  CPU             │   │
│  │ MX3       │    │                  │   │
│  │           │    │  Stage 2:        │   │
│  │ Stage 1:  │    │  Temporal CNN    │   │
│  │ YOLO Pose │    │  (<5ms/seq)     │   │
│  │ (30fps)   │    │                  │   │
│  │           │    │  Roboflow Obj    │   │
│  │           │    │  (every 15th     │   │
│  │           │    │   frame, ~20ms)  │   │
│  └───────────┘    └──────────────────┘   │
│                                           │
│  Total: <2W additional power              │
│  Latency: <50ms end-to-end               │
└──────────────────────────────────────────┘
```
