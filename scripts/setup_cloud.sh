#!/usr/bin/env bash
# Set up a cloud GPU for autoresearch.
#
# Tested on: Lambda Cloud (A10, A100), RunPod (RTX 4090)
#
# Usage:
#   ssh your-cloud-instance
#   git clone https://github.com/sfuller3/pose-autoresearch.git
#   cd pose-autoresearch
#   ./scripts/setup_cloud.sh

set -euo pipefail

echo "=== Pose Autoresearch - Cloud GPU Setup ==="
echo ""

# 1. System check
echo "Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No NVIDIA GPU detected."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 2. Python environment
echo "Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install dependencies
echo "Installing PyTorch with CUDA..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy ultralytics
pip install -e .

# 4. Verify CUDA
echo ""
echo "Verifying CUDA..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

# 5. Check data
echo ""
echo "Checking data..."
if [ -d "data/processed" ] && [ "$(ls data/processed/*.json 2>/dev/null | wc -l)" -gt "0" ]; then
    COUNT=$(ls data/processed/*.json | wc -l)
    echo "Data found: $COUNT samples"
else
    echo "No data found. Generating synthetic data for smoke test..."
    python3 prepare.py --dataset synthetic
    echo ""
    echo "For real training, download FallVision/Le2i and run:"
    echo "  python scripts/convert_fallvision.py --source all"
fi

# 6. Quick smoke test
echo ""
echo "Running 30-second smoke test..."
python3 -c "
import os, time
os.environ['POSE_AUTORESEARCH_MAX_TIME'] = '30'
from prepare import get_dataloaders, evaluate_model, DEVICE
from train import PoseEventClassifier
train_ld, val_ld, _ = get_dataloaders(batch_size=32, num_workers=2)
model = PoseEventClassifier().to(DEVICE)
acc, loss = evaluate_model(model, val_ld, DEVICE)
print(f'Untrained model: val_acc={acc:.4f}, loss={loss:.4f}')
print('Smoke test passed.')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start autoresearch, run Claude Code:"
echo "  claude 'Read program.md, then run the autoresearch loop on train.py'"
echo ""
echo "Or train manually:"
echo "  source .venv/bin/activate"
echo "  python train.py"
