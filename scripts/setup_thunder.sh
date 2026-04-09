#!/usr/bin/env bash
# Full setup on Thunder Compute: video → poses → training → autoresearch.
#
# Everything runs on one A100 GPU instance (~$1.10/hr).
# YOLO extraction is fast on GPU (~5s/video), training epochs are ~1-3s.
# Total cost for full pipeline (extract + 100 experiments): ~$2-5.
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FROM YOUR MAC:
#
#   1. Install Thunder CLI:
#      curl -fsSL https://raw.githubusercontent.com/Thunder-Compute/thunder-cli/main/scripts/install.sh | bash
#
#   2. Login & create A100 instance (~$1.10/hr):
#      tnr login
#      tnr create --gpu a100
#
#   3. Connect and run (datasets download automatically):
#      tnr connect <instance-id>
#      git clone https://github.com/sfuller3/pose-autoresearch.git
#      cd pose-autoresearch
#      ./scripts/setup_thunder.sh
#
#      NOTE: Le2i requires a Kaggle API token. Before running setup:
#        mkdir -p ~/.kaggle
#        echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
#        chmod 600 ~/.kaggle/kaggle.json
#      Get your token at: https://www.kaggle.com/settings → Create New API Token
#
#   4. Launch autoresearch:
#      ./run_autoresearch.sh
#
#   5. When done, snapshot before stopping:
#      tnr snapshot <instance-id>
#      tnr stop <instance-id>
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pose Autoresearch — Thunder Compute GPU Setup           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. System Check ────────────────────────────────────────────────
echo "▶ System info..."
echo "  OS:     $(uname -s) $(uname -r)"
echo "  CPU:    $(nproc) cores"
echo "  RAM:    $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'unknown')"
echo "  Disk:   $(df -h . | awk 'NR==2{print $4}') available"

if nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  GPU:    $GPU_NAME ($GPU_MEM)"
    HAS_GPU=true
else
    echo "  GPU:    None detected (will use CPU)"
    HAS_GPU=false
fi
echo ""

# ── 2. Python Environment ─────────────────────────────────────────
echo "▶ Setting up Python environment..."

# Thunder Compute pre-installs Python + PyTorch. Check first.
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NONE")

if [ "$TORCH_VER" != "NONE" ]; then
    echo "  Pre-installed PyTorch: $TORCH_VER"
    CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
    echo "  CUDA available: $CUDA_AVAIL"

    # Thunder Compute has system Python with PyTorch — use it directly
    # Create venv only if system Python is missing packages we need
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv --system-site-packages
        echo "  Created .venv (with system site-packages for pre-installed PyTorch)"
    fi
else
    echo "  No pre-installed PyTorch. Installing..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip -q

    if [ "$HAS_GPU" = true ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    fi
fi

source .venv/bin/activate
pip install --upgrade pip -q 2>/dev/null
echo ""

# ── 3. Install Dependencies ────────────────────────────────────────
echo "▶ Installing dependencies..."

pip install numpy -q
# Ultralytics for YOLO pose extraction from video files
pip install ultralytics opencv-python-headless -q

# Install the pose-autoresearch package
pip install -e . -q 2>/dev/null || {
    echo "  Note: editable install issue — using PYTHONPATH fallback"
    export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
}

echo "  Done."
echo ""

# ── 4. Verify Stack ────────────────────────────────────────────────
echo "▶ Verifying compute stack..."
python3 << 'PYCHECK'
import torch
import sys

print(f"  Python:    {sys.version.split()[0]}")
print(f"  PyTorch:   {torch.__version__}")

if torch.cuda.is_available():
    print(f"  CUDA:      {torch.version.cuda}")
    print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram = (getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)) / 1e9
    print(f"  VRAM:      {vram:.1f} GB")

    # GPU benchmark
    import time
    x = torch.randn(4096, 4096, device='cuda')
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()
    tflops = 10 * 2 * 4096**3 / (time.time() - t0) / 1e12
    print(f"  Benchmark: {tflops:.1f} TFLOPS")
else:
    import multiprocessing
    print(f"  Device:    CPU ({multiprocessing.cpu_count()} cores)")

    import time
    x = torch.randn(2048, 2048)
    t0 = time.time()
    for _ in range(10):
        y = x @ x
    gflops = 10 * 2 * 2048**3 / (time.time() - t0) / 1e9
    print(f"  Benchmark: {gflops:.0f} GFLOPS")
PYCHECK
echo ""

# ── 5. Extract Poses from Videos (GPU-accelerated) ─────────────────
echo "▶ Checking data..."
PROCESSED_COUNT=0
if [ -d "data/processed" ]; then
    PROCESSED_COUNT=$(find data/processed -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
fi

RAW_VIDEO_COUNT=0
if [ -d "data/raw" ]; then
    RAW_VIDEO_COUNT=$(find data/raw -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) 2>/dev/null | wc -l | tr -d ' ')
fi

if [ "$PROCESSED_COUNT" -gt "50" ]; then
    echo "  Found $PROCESSED_COUNT processed samples. Skipping extraction."
elif [ "$RAW_VIDEO_COUNT" -gt "0" ]; then
    echo "  Found $RAW_VIDEO_COUNT raw video files."
    echo ""

    if [ -d "data/raw/fallvision" ]; then
        FV_COUNT=$(find data/raw/fallvision -type f \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | wc -l | tr -d ' ')
        echo "  FallVision: $FV_COUNT videos"
    fi
    if [ -d "data/raw/le2i" ]; then
        LE_COUNT=$(find data/raw/le2i -type f \( -name "*.mp4" -o -name "*.avi" \) 2>/dev/null | wc -l | tr -d ' ')
        echo "  Le2i:       $LE_COUNT videos"
    fi

    echo ""
    if [ "$HAS_GPU" = true ]; then
        echo "  Extracting poses with YOLO on GPU (~5-15s per video)..."
    else
        echo "  Extracting poses with YOLO on CPU (~1-3min per video)..."
    fi

    time python3 scripts/convert_fallvision.py --source all

    NEW_COUNT=$(find data/processed -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ✓ Extracted $NEW_COUNT pose sequences."
else
    echo "  No data found. Downloading datasets..."
    echo ""
    ./scripts/download_data.sh all

    # Re-check for videos after download
    RAW_VIDEO_COUNT=$(find data/raw -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mov" \) 2>/dev/null | wc -l | tr -d ' ')
    if [ "$RAW_VIDEO_COUNT" -gt "0" ]; then
        echo ""
        echo "  Found $RAW_VIDEO_COUNT videos. Extracting poses..."
        if [ "$HAS_GPU" = true ]; then
            echo "  (GPU-accelerated — ~5-15s per video)"
        else
            echo "  (CPU mode — ~1-3min per video)"
        fi
        time python3 scripts/convert_fallvision.py --source all
        NEW_COUNT=$(find data/processed -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
        echo "  ✓ Extracted $NEW_COUNT pose sequences."
    else
        echo "  Download may have failed. Generating synthetic data for smoke test..."
        python3 prepare.py --dataset synthetic
    fi
fi
echo ""

# ── 6. Smoke Test ──────────────────────────────────────────────────
echo "▶ Running smoke test..."
python3 << 'PYSMOKE'
import os, time, torch
os.environ['POSE_AUTORESEARCH_MAX_TIME'] = '30'

from prepare import get_dataloaders, evaluate_model, DEVICE
from train import PoseEventClassifier

train_ld, val_ld, test_ld = get_dataloaders(batch_size=64, num_workers=2)
print(f"  Data:     {len(train_ld.dataset)} train, {len(val_ld.dataset)} val")

model = PoseEventClassifier().to(DEVICE)
params = sum(p.numel() for p in model.parameters())
print(f"  Model:    {params:,} parameters on {DEVICE}")

# Inference timing
batch = next(iter(train_ld))
x = batch[0].to(DEVICE)
if DEVICE.type == 'cuda':
    torch.cuda.synchronize()

t0 = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(x)

if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
ms_per_batch = (time.time() - t0) / 100 * 1000
ms_per_sample = ms_per_batch / x.shape[0]
print(f"  Inference: {ms_per_batch:.1f}ms/batch ({ms_per_sample:.2f}ms/sample)")

acc, loss = evaluate_model(model, val_ld, DEVICE)
print(f"  Baseline:  acc={acc:.4f} (untrained, expected ~0.14)")
print("  ✓ Smoke test passed.")
PYSMOKE
echo ""

# ── 7. Estimate Throughput ─────────────────────────────────────────
echo "▶ Estimating experiment throughput..."
python3 << 'PYESTIMATE'
import os, time, torch, torch.nn as nn
os.environ['POSE_AUTORESEARCH_MAX_TIME'] = '60'

from prepare import get_dataloaders, DEVICE
from train import PoseEventClassifier

train_ld, val_ld, _ = get_dataloaders(batch_size=64, num_workers=2)
model = PoseEventClassifier().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

if DEVICE.type == 'cuda':
    torch.cuda.synchronize()

t0 = time.time()
model.train()
for batch_x, batch_y in train_ld:
    batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
    optimizer.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    loss.backward()
    optimizer.step()

if DEVICE.type == 'cuda':
    torch.cuda.synchronize()
epoch_time = time.time() - t0

epochs_per_5min = int(300 / epoch_time)
experiment_time = 305
experiments_per_hour = 3600 / experiment_time
hourly_cost = 0.27  # A6000 prototyping tier

print(f"  Epoch time:             {epoch_time:.1f}s")
print(f"  Epochs per 5min run:    ~{epochs_per_5min}")
print(f"  Experiments per hour:   ~{experiments_per_hour:.0f}")
hours_for_100 = 100 * experiment_time / 3600
print(f"  100 experiments:        ~{hours_for_100:.1f} hours")
print(f"  Estimated cost:         ~${hours_for_100 * hourly_cost:.2f} (A6000 @ $0.27/hr)")
PYESTIMATE
echo ""

# ── 8. Create tmux launch script ──────────────────────────────────
echo "▶ Creating launch scripts..."
cat > run_autoresearch.sh << 'LAUNCH'
#!/usr/bin/env bash
# Launch autoresearch in a tmux session (survives SSH disconnect).
#
# Usage: ./run_autoresearch.sh
#
# Monitor: tmux attach -t autoresearch
# Detach:  Ctrl+B, then D

set -euo pipefail
cd "$(dirname "$0")"

source .venv/bin/activate 2>/dev/null || true

SESSION="autoresearch"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running. Attach with: tmux attach -t $SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -x 200 -y 50

# Main pane: Claude autoresearch agent
tmux send-keys -t "$SESSION" "source .venv/bin/activate 2>/dev/null; claude 'Read program.md and run the autoresearch loop. Modify only train.py. Run each experiment for 5 minutes max. Keep changes that improve validation accuracy (especially fall recall). Discard regressions. Target: 100 experiments.'" Enter

# Split for monitoring
tmux split-window -t "$SESSION" -v -p 20
tmux send-keys -t "$SESSION" "watch -n 30 'tail -5 checkpoints/experiment_log.txt 2>/dev/null || echo \"Waiting for first experiment...\"'" Enter

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Autoresearch launched in tmux session                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Attach:  tmux attach -t autoresearch                   ║"
echo "║  Detach:  Ctrl+B, then D                                ║"
echo "║  Kill:    tmux kill-session -t autoresearch              ║"
echo "║                                                         ║"
echo "║  GPU monitor (separate terminal):                       ║"
echo "║    watch -n 1 nvidia-smi                                ║"
echo "║                                                         ║"
echo "║  REMEMBER: Snapshot before stopping instance!            ║"
echo "║    tnr snapshot <instance-id>                            ║"
echo "╚══════════════════════════════════════════════════════════╝"

LAUNCH
chmod +x run_autoresearch.sh
echo "  Written: run_autoresearch.sh"
echo ""

# ── Done ───────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Setup Complete                                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                         ║"
echo "║  Quick start:                                           ║"
echo "║    ./run_autoresearch.sh                                ║"
echo "║                                                         ║"
echo "║  Manual training:                                       ║"
echo "║    source .venv/bin/activate                            ║"
echo "║    python train.py                                      ║"
echo "║                                                         ║"
echo "║  Monitor GPU:                                           ║"
echo "║    watch -n 1 nvidia-smi                                ║"
echo "║                                                         ║"
echo "║  Cost tracking:                                         ║"
echo "║    Thunder console → Instance → Usage                   ║"
echo "║                                                         ║"
echo "║  SAVE WORK: Snapshot before stopping!                   ║"
echo "║    tnr snapshot <instance-id>                            ║"
echo "║                                                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
