#!/usr/bin/env bash
# End-to-end training pipeline for Thunder A100.
#
# Runs: download → convert → clean → split → train
#
# Prerequisites:
#   - Thunder A100 instance with GPU
#   - Kaggle API token at ~/.kaggle/kaggle.json (for Le2i)
#
# Usage:
#   ./scripts/run_thunder_pipeline.sh           # Full pipeline
#   ./scripts/run_thunder_pipeline.sh --skip-download  # Skip data download
#   ./scripts/run_thunder_pipeline.sh --train-only     # Only train (data ready)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SKIP_DOWNLOAD=false
TRAIN_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --train-only) TRAIN_ONLY=true ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pose AutoResearch - Thunder A100 Pipeline               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 0: Environment ──────────────────────────────────────────
echo "▶ Step 0: Environment setup"

if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv .venv --system-site-packages 2>/dev/null || python3 -m venv .venv
fi
source .venv/bin/activate

pip install -q numpy torch ultralytics matplotlib pillow pytest 2>/dev/null
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  GPU: $GPU"
else
    echo "  WARNING: No GPU detected. Le2i YOLO extraction will be slow."
fi
echo ""

if [ "$TRAIN_ONLY" = true ]; then
    echo "▶ Skipping to training (--train-only)"
    echo ""
    exec python train.py
fi

# ── Step 1: Download datasets ────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "▶ Step 1: Downloading datasets"

    # Check for Kaggle token
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo "  ERROR: Kaggle API token not found at ~/.kaggle/kaggle.json"
        echo "  Le2i download requires it. Set up:"
        echo "    1. Go to https://www.kaggle.com/settings"
        echo "    2. Create New API Token"
        echo "    3. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/"
        echo "    4. chmod 600 ~/.kaggle/kaggle.json"
        exit 1
    fi

    ./scripts/download_data.sh all
    echo ""
else
    echo "▶ Step 1: Skipping download (--skip-download)"
    echo ""
fi

# ── Step 2: Convert all datasets ─────────────────────────────────
echo "▶ Step 2: Converting datasets to JSON"

echo "  Converting FallVision + Le2i..."
python scripts/convert_fallvision.py --source all 2>&1 | tail -5
echo ""

echo "  Converting NTU RGB+D 120..."
python scripts/convert_ntu120.py --max-per-class 3000 2>&1 | tail -5
echo ""

# ── Step 3: Remove synthetic data ────────────────────────────────
echo "▶ Step 3: Removing synthetic data"

SYNTH_COUNT=0
for f in data/processed/*.json; do
    basename=$(basename "$f")
    # Skip files with source prefixes
    case "$basename" in
        ntu_*|fv_*|le2i_*) continue ;;
    esac
    rm -f "$f"
    SYNTH_COUNT=$((SYNTH_COUNT + 1))
done
echo "  Removed $SYNTH_COUNT synthetic files"
echo ""

# ── Step 4: Split into train/val/test ────────────────────────────
echo "▶ Step 4: Source-aware train/val/test split"

python scripts/split_data.py --copy
echo ""

# ── Step 5: Audit training data ─────────────────────────────────
echo "▶ Step 5: Auditing training data (visual verification)"
echo ""

python scripts/audit_training_data.py --quick
echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║  PAUSE: Review data_audit/ before continuing.         ║"
echo "  ║                                                       ║"
echo "  ║  Check:                                               ║"
echo "  ║    data_audit/split_summary.txt   — class counts      ║"
echo "  ║    data_audit/class_distribution.png — bar chart       ║"
echo "  ║    data_audit/samples/<class>/    — stick figures      ║"
echo "  ║                                                       ║"
echo "  ║  Does each class look right? Labels match motion?     ║"
echo "  ║  Press Enter to continue training, Ctrl+C to abort.   ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""
read -r -p "  Continue? [Enter] "
echo ""

# ── Step 6: Train ────────────────────────────────────────────────
echo "▶ Step 6: Training"
echo ""

python train.py

echo ""

# ── Step 7: Post-training audit ──────────────────────────────────
echo "▶ Step 7: Post-training confusion matrix"
echo ""

python scripts/audit_training_data.py --checkpoint checkpoints/best_model.pt --quick

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Pipeline Complete                                     ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                         ║"
echo "║  Review results:                                        ║"
echo "║    data_audit/confusion_matrix.png — model accuracy     ║"
echo "║    data_audit/samples/             — class samples       ║"
echo "║    checkpoints/best_model.pt       — trained model       ║"
echo "║                                                         ║"
echo "║  Copy results to local machine:                         ║"
echo "║    scp -r thunder:pose-autoresearch/data_audit ./        ║"
echo "║                                                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
