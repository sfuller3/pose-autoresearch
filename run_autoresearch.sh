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
