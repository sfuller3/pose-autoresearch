#!/usr/bin/env bash
# Download FallVision and Le2i datasets directly to this machine.
#
# FallVision: Public, no auth needed (Harvard Dataverse API)
# Le2i:       Requires Kaggle API token (one-time setup)
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KAGGLE SETUP (one-time, needed for Le2i only):
#   1. Go to https://www.kaggle.com/settings
#   2. Click "Create New API Token" → downloads kaggle.json
#   3. Copy to server: scp kaggle.json thunder:~/.kaggle/kaggle.json
#   4. chmod 600 ~/.kaggle/kaggle.json
#
# USAGE:
#   ./scripts/download_data.sh              # Download both datasets
#   ./scripts/download_data.sh fallvision   # FallVision only
#   ./scripts/download_data.sh le2i         # Le2i only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$PROJECT_DIR/data/raw"

SOURCE="${1:-all}"

mkdir -p "$RAW_DIR"

# ── FallVision (Harvard Dataverse, CC0 license) ───────────────────
download_fallvision() {
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Downloading FallVision from Harvard Dataverse           ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    FV_DIR="$RAW_DIR/fallvision"
    mkdir -p "$FV_DIR"

    # First, list files to show what we're downloading
    echo "▶ Listing dataset contents..."
    curl -s "https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/75QPKK" \
        | python3 -c "
import sys, json
data = json.load(sys.stdin)
files = data.get('data', {}).get('latestVersion', {}).get('files', [])
total_bytes = sum(f.get('dataFile', {}).get('filesize', 0) for f in files)
print(f'  Files: {len(files)}')
print(f'  Total: {total_bytes / 1e9:.1f} GB')
print()
# Show a few examples
for f in files[:5]:
    df = f.get('dataFile', {})
    print(f'    {df.get(\"filename\", \"?\"):40s} {df.get(\"filesize\", 0)/1e6:>8.1f} MB')
if len(files) > 5:
    print(f'    ... and {len(files) - 5} more files')
" || {
        echo "  WARNING: Could not list files. Proceeding with download anyway."
    }

    echo ""
    echo "▶ Downloading dataset (this may take a while)..."

    # Download entire dataset as zip
    curl -L --progress-bar -o "$FV_DIR/fallvision_dataset.zip" \
        "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/75QPKK"

    echo ""
    echo "▶ Extracting..."
    cd "$FV_DIR"
    unzip -q fallvision_dataset.zip

    # Organize into fall/nofall if not already structured
    # (Harvard Dataverse may flatten the directory structure in the zip)
    if [ ! -d "fall" ] && [ ! -d "nofall" ]; then
        echo "  Organizing files..."
        mkdir -p fall nofall
        # Move files based on naming convention (adjust if needed)
        # FallVision typically names files with "fall" or "adl" prefixes
        find . -maxdepth 1 -name "*fall*" -name "*.mp4" -exec mv {} fall/ \; 2>/dev/null || true
        find . -maxdepth 1 -name "*adl*" -name "*.mp4" -exec mv {} nofall/ \; 2>/dev/null || true
        find . -maxdepth 1 -name "*nofall*" -name "*.mp4" -exec mv {} nofall/ \; 2>/dev/null || true
    fi

    # Clean up zip to save space
    rm -f fallvision_dataset.zip

    FALL_COUNT=$(find fall -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
    NOFALL_COUNT=$(find nofall -name "*.mp4" 2>/dev/null | wc -l | tr -d ' ')
    TOTAL_SIZE=$(du -sh "$FV_DIR" | cut -f1)

    echo ""
    echo "  ✓ FallVision: $FALL_COUNT fall + $NOFALL_COUNT nofall videos ($TOTAL_SIZE)"

    if [ "$FALL_COUNT" -eq "0" ] && [ "$NOFALL_COUNT" -eq "0" ]; then
        echo ""
        echo "  ⚠ No videos found in fall/nofall dirs."
        echo "  Check the extracted structure:"
        ls -la "$FV_DIR" | head -20
        echo ""
        echo "  You may need to manually organize files into fall/ and nofall/ subdirs."
    fi
}

# ── Le2i (Kaggle, academic use) ───────────────────────────────────
download_le2i() {
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Downloading Le2i from Kaggle                            ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    # Check for Kaggle credentials
    if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
        echo "  ERROR: Kaggle API token not found."
        echo ""
        echo "  Setup (one-time):"
        echo "    1. Go to https://www.kaggle.com/settings"
        echo "    2. Click 'Create New API Token' → downloads kaggle.json"
        echo "    3. Copy to this machine:"
        echo "       mkdir -p ~/.kaggle"
        echo "       # paste your kaggle.json content:"
        echo "       cat > ~/.kaggle/kaggle.json << 'EOF'"
        echo '       {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}'
        echo "       EOF"
        echo "       chmod 600 ~/.kaggle/kaggle.json"
        echo ""
        echo "  Then rerun: ./scripts/download_data.sh le2i"
        return 1
    fi

    # Install kaggle CLI if needed
    pip install kaggle -q 2>/dev/null

    LE2I_DIR="$RAW_DIR/le2i"
    mkdir -p "$LE2I_DIR"

    echo "▶ Downloading Le2i fall dataset..."
    kaggle datasets download -d tuyenldvn/falldataset-imvia --unzip -p "$LE2I_DIR"

    VIDEO_COUNT=$(find "$LE2I_DIR" -type f \( -name "*.avi" -o -name "*.mp4" \) 2>/dev/null | wc -l | tr -d ' ')
    TOTAL_SIZE=$(du -sh "$LE2I_DIR" | cut -f1)

    echo ""
    echo "  ✓ Le2i: $VIDEO_COUNT videos ($TOTAL_SIZE)"
    echo ""

    # Show directory structure
    echo "  Structure:"
    find "$LE2I_DIR" -type d -maxdepth 2 | head -15 | while read -r dir; do
        count=$(find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' ')
        echo "    $dir ($count files)"
    done
}

# ── Main ──────────────────────────────────────────────────────────
echo ""

case "$SOURCE" in
    fallvision)
        download_fallvision
        ;;
    le2i)
        download_le2i
        ;;
    all)
        download_fallvision
        echo ""
        download_le2i
        ;;
    *)
        echo "Usage: $0 [fallvision|le2i|all]"
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ Download Complete                                    ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                         ║"
echo "║  Next: Run setup to extract poses and start training:   ║"
echo "║    ./scripts/setup_thunder.sh                           ║"
echo "║                                                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
