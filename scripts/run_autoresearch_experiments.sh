#!/usr/bin/env bash
# Autoresearch experiments targeting aggression↔working_together and
# wandering↔unstable_gait confusion patterns.
#
# Each experiment modifies train.py, runs for 1 hour, logs results,
# then restores train.py for the next experiment.
#
# Usage:
#   ./scripts/run_autoresearch_experiments.sh
#
# Results logged to: experiments/experiment_log.txt
# Best checkpoint per experiment saved to: experiments/exp_N_best_model.pt

set -euo pipefail

cd "$(dirname "$0")/.."

RESULTS_DIR="experiments"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/experiment_log.txt"
TRAIN_PY="train.py"
BACKUP="$RESULTS_DIR/train.py.baseline"

# Save baseline
cp "$TRAIN_PY" "$BACKUP"

echo "╔══════════════════════════════════════════════════════════╗" | tee -a "$LOG"
echo "║  Autoresearch Experiments — $(date '+%Y-%m-%d %H:%M')              ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════════════════╝" | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_experiment() {
    local exp_num="$1"
    local exp_name="$2"
    local description="$3"
    shift 3
    # Remaining args are sed commands to apply

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo "▶ Experiment $exp_num: $exp_name" | tee -a "$LOG"
    echo "  $description" | tee -a "$LOG"
    echo "  Started: $(date '+%H:%M:%S')" | tee -a "$LOG"

    # Restore baseline and apply this experiment's changes
    cp "$BACKUP" "$TRAIN_PY"
    # Always apply the worker fix
    sed -i 's/num_workers=4/num_workers=2/g' "$TRAIN_PY"
    sed -i 's/pin_memory=pin/pin_memory=False/g' "$TRAIN_PY"

    for cmd in "$@"; do
        eval "$cmd"
    done

    # Show the diff
    echo "  Changes:" | tee -a "$LOG"
    diff "$BACKUP" "$TRAIN_PY" | grep '^[<>]' | head -10 | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    # Run training
    POSE_AUTORESEARCH_MAX_TIME=3600 python3 "$TRAIN_PY" 2>&1 | tee "$RESULTS_DIR/exp_${exp_num}_output.txt"

    # Extract results
    local final_line
    final_line=$(grep "FINAL VALIDATION ACCURACY" "$RESULTS_DIR/exp_${exp_num}_output.txt" || echo "FAILED")
    local test_line
    test_line=$(grep "Test Acc" "$RESULTS_DIR/exp_${exp_num}_output.txt" || echo "FAILED")

    echo "  $final_line" | tee -a "$LOG"
    echo "  $test_line" | tee -a "$LOG"

    # Save per-class results
    grep -A 8 "Per-class accuracy" "$RESULTS_DIR/exp_${exp_num}_output.txt" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    # Save checkpoint
    if [ -f "checkpoints/best_model.pt" ]; then
        cp checkpoints/best_model.pt "$RESULTS_DIR/exp_${exp_num}_best_model.pt"
        echo "  Checkpoint: $RESULTS_DIR/exp_${exp_num}_best_model.pt" | tee -a "$LOG"
    fi
    echo "  Finished: $(date '+%H:%M:%S')" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

# ── Experiment 0: Baseline (reproduce current best) ──────────────
run_experiment 0 "baseline" \
    "Reproduce current best with worker fix (control run)" \

# ── Experiment 1: Single cosine decay ────────────────────────────
run_experiment 1 "cosine-single" \
    "T_max=260 — single smooth LR decay instead of cycling every 50 epochs" \
    "sed -i 's/T_max=50/T_max=260/g' $TRAIN_PY"

# ── Experiment 2: Larger batch size ──────────────────────────────
run_experiment 2 "batch-256" \
    "Batch size 256 — more stable gradients for distinguishing similar classes" \
    "sed -i 's/BATCH_SIZE = 64/BATCH_SIZE = 256/g' $TRAIN_PY"

# ── Experiment 3: Lower learning rate ────────────────────────────
run_experiment 3 "lr-1e3" \
    "LR 1e-3 — finer weight updates for subtle distinctions" \
    "sed -i 's/LEARNING_RATE = 2e-3/LEARNING_RATE = 1e-3/g' $TRAIN_PY"

# ── Experiment 4: Aggression class weight boost ──────────────────
run_experiment 4 "aggression-boost" \
    "2x weight on aggression class (index 3) to reduce aggression→working_together confusion" \
    "sed -i 's/class_weights\[0\] \*= 1.5/class_weights[0] *= 1.5\n        class_weights[3] *= 2.0  # aggression boost/g' $TRAIN_PY"

# ── Experiment 5: Combined best (cosine + batch + lr) ────────────
run_experiment 5 "combined" \
    "T_max=260 + batch 256 + LR 1e-3 — combine structural improvements" \
    "sed -i 's/T_max=50/T_max=260/g' $TRAIN_PY" \
    "sed -i 's/BATCH_SIZE = 64/BATCH_SIZE = 256/g' $TRAIN_PY" \
    "sed -i 's/LEARNING_RATE = 2e-3/LEARNING_RATE = 1e-3/g' $TRAIN_PY"

# ── Experiment 6: Combined + aggression boost ────────────────────
run_experiment 6 "combined-aggression" \
    "Experiment 5 + 2x aggression weight — full kitchen sink" \
    "sed -i 's/T_max=50/T_max=260/g' $TRAIN_PY" \
    "sed -i 's/BATCH_SIZE = 64/BATCH_SIZE = 256/g' $TRAIN_PY" \
    "sed -i 's/LEARNING_RATE = 2e-3/LEARNING_RATE = 1e-3/g' $TRAIN_PY" \
    "sed -i 's/class_weights\[0\] \*= 1.5/class_weights[0] *= 1.5\n        class_weights[3] *= 2.0  # aggression boost/g' $TRAIN_PY"

# ── Restore baseline ─────────────────────────────────────────────
cp "$BACKUP" "$TRAIN_PY"
# Re-apply worker fix permanently
sed -i 's/num_workers=4/num_workers=2/g' "$TRAIN_PY"
sed -i 's/pin_memory=pin/pin_memory=False/g' "$TRAIN_PY"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
echo "▶ All experiments complete" | tee -a "$LOG"
echo "  Results: $LOG" | tee -a "$LOG"
echo "  Checkpoints: $RESULTS_DIR/exp_*_best_model.pt" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Pick the best checkpoint and copy to checkpoints/       ║"
echo "║                                                         ║"
echo "║  Compare results:                                       ║"
echo "║    cat $LOG                                    ║"
echo "║                                                         ║"
echo "║  Deploy winner:                                         ║"
echo "║    cp experiments/exp_N_best_model.pt checkpoints/       ║"
echo "╚══════════════════════════════════════════════════════════╝"
