#!/usr/bin/env bash
# Autoresearch experiments targeting aggression↔working_together and
# wandering↔unstable_gait confusion patterns.
#
# Each experiment modifies train.py, runs for 1 hour, saves its checkpoint
# and results immediately, then moves to the next. Progress is saved after
# every experiment so partial runs are still useful.
#
# Usage:
#   ./scripts/run_autoresearch_experiments.sh
#   ./scripts/run_autoresearch_experiments.sh --resume 3   # skip to experiment 3
#
# Results:
#   experiments/experiment_log.txt          — summary of all runs
#   experiments/exp_N_best_model.pt         — checkpoint per experiment
#   experiments/exp_N_output.txt            — full training output
#   experiments/exp_N_confusion_matrix.png  — confusion matrix per experiment
#   experiments/summary.txt                 — side-by-side comparison table

cd "$(dirname "$0")/.."

RESULTS_DIR="experiments"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/experiment_log.txt"
SUMMARY="$RESULTS_DIR/summary.txt"
TRAIN_PY="train.py"
BACKUP="$RESULTS_DIR/train.py.baseline"

# Parse --resume flag
RESUME_FROM=0
if [ "${1:-}" = "--resume" ] && [ -n "${2:-}" ]; then
    RESUME_FROM="$2"
    echo "Resuming from experiment $RESUME_FROM"
fi

# Save baseline (only if not resuming)
if [ "$RESUME_FROM" -eq 0 ]; then
    cp "$TRAIN_PY" "$BACKUP"
fi

# Ensure baseline exists
if [ ! -f "$BACKUP" ]; then
    cp "$TRAIN_PY" "$BACKUP"
fi

echo "╔══════════════════════════════════════════════════════════╗" | tee -a "$LOG"
echo "║  Autoresearch Experiments — $(date '+%Y-%m-%d %H:%M')              ║" | tee -a "$LOG"
echo "╚══════════════════════════════════════════════════════════╝" | tee -a "$LOG"
echo "" | tee -a "$LOG"

run_experiment() {
    local exp_num="$1"
    local exp_name="$2"
    local description="$3"
    shift 3

    # Skip if resuming past this experiment
    if [ "$exp_num" -lt "$RESUME_FROM" ]; then
        echo "  Skipping experiment $exp_num (resuming from $RESUME_FROM)" | tee -a "$LOG"
        return 0
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$LOG"
    echo "▶ Experiment $exp_num: $exp_name" | tee -a "$LOG"
    echo "  $description" | tee -a "$LOG"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"

    # Restore baseline and apply this experiment's changes
    cp "$BACKUP" "$TRAIN_PY"
    # Always apply the worker fix
    sed -i 's/num_workers=4/num_workers=2/g' "$TRAIN_PY"
    sed -i 's/pin_memory=pin/pin_memory=False/g' "$TRAIN_PY"

    # Apply experiment-specific changes
    for cmd in "$@"; do
        eval "$cmd"
    done

    # Show the diff (|| true because diff returns 1 when files differ)
    echo "  Changes:" | tee -a "$LOG"
    diff "$BACKUP" "$TRAIN_PY" | grep '^[<>]' | head -10 | tee -a "$LOG" || true
    echo "" | tee -a "$LOG"

    # Run training
    POSE_AUTORESEARCH_MAX_TIME=3600 python3 "$TRAIN_PY" 2>&1 | tee "$RESULTS_DIR/exp_${exp_num}_output.txt"
    local train_exit=$?

    # ── Save checkpoint immediately ──────────────────────────────
    if [ -f "checkpoints/best_model.pt" ]; then
        cp checkpoints/best_model.pt "$RESULTS_DIR/exp_${exp_num}_best_model.pt"
        echo "  ✓ Checkpoint saved: $RESULTS_DIR/exp_${exp_num}_best_model.pt" | tee -a "$LOG"
    else
        echo "  ✗ No checkpoint found" | tee -a "$LOG"
    fi

    # ── Extract and save results ─────────────────────────────────
    local val_acc="FAILED"
    local test_acc="FAILED"
    local test_loss="N/A"

    if [ -f "$RESULTS_DIR/exp_${exp_num}_output.txt" ]; then
        val_acc=$(grep "FINAL VALIDATION ACCURACY" "$RESULTS_DIR/exp_${exp_num}_output.txt" | awk '{print $NF}' || echo "FAILED")
        test_acc=$(grep "Test Acc:" "$RESULTS_DIR/exp_${exp_num}_output.txt" | awk '{print $3}' || echo "FAILED")
        test_loss=$(grep "Test Acc:" "$RESULTS_DIR/exp_${exp_num}_output.txt" | awk '{print $NF}' || echo "N/A")
    fi

    echo "  Val Acc:  $val_acc" | tee -a "$LOG"
    echo "  Test Acc: $test_acc" | tee -a "$LOG"

    # Save per-class results
    if [ -f "$RESULTS_DIR/exp_${exp_num}_output.txt" ]; then
        grep -A 8 "Per-class accuracy" "$RESULTS_DIR/exp_${exp_num}_output.txt" | tee -a "$LOG" || true
    fi

    # ── Generate confusion matrix for this experiment ────────────
    if [ -f "$RESULTS_DIR/exp_${exp_num}_best_model.pt" ]; then
        echo "  Generating confusion matrix..." | tee -a "$LOG"
        cp "$RESULTS_DIR/exp_${exp_num}_best_model.pt" checkpoints/best_model.pt
        python3 scripts/audit_training_data.py --checkpoint checkpoints/best_model.pt --quick 2>&1 | tail -3 || true
        if [ -f "data_audit/confusion_matrix.png" ]; then
            cp data_audit/confusion_matrix.png "$RESULTS_DIR/exp_${exp_num}_confusion_matrix.png"
            echo "  ✓ Confusion matrix: $RESULTS_DIR/exp_${exp_num}_confusion_matrix.png" | tee -a "$LOG"
        fi
    fi

    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
    echo "" | tee -a "$LOG"

    # ── Update summary table after each experiment ───────────────
    update_summary
}

update_summary() {
    # Build a comparison table from all completed experiments
    echo "Autoresearch Results — $(date '+%Y-%m-%d %H:%M')" > "$SUMMARY"
    echo "" >> "$SUMMARY"
    printf "%-4s %-22s %-10s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n" \
        "#" "Name" "Val" "Test" "fall" "eat" "work" "aggr" "gait" "wand" "sit" >> "$SUMMARY"
    printf "%-4s %-22s %-10s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n" \
        "---" "----" "---" "----" "----" "---" "----" "----" "----" "----" "---" >> "$SUMMARY"

    for f in "$RESULTS_DIR"/exp_*_output.txt; do
        [ -f "$f" ] || continue
        local num
        num=$(echo "$f" | grep -o 'exp_[0-9]*' | grep -o '[0-9]*')
        local name
        name=$(grep "Experiment $num:" "$LOG" | head -1 | sed "s/.*Experiment $num: //" || echo "unknown")

        local val test fall eat work aggr gait wand sit
        val=$(grep "FINAL VALIDATION ACCURACY" "$f" | awk '{print $NF}' || echo "-")
        test=$(grep "Test Acc:" "$f" | awk '{print $3}' || echo "-")

        fall=$(grep "fall " "$f" | tail -1 | awk '{print $3}' || echo "-")
        eat=$(grep "eating " "$f" | tail -1 | awk '{print $3}' || echo "-")
        work=$(grep "working_together " "$f" | tail -1 | awk '{print $3}' || echo "-")
        aggr=$(grep "aggression " "$f" | tail -1 | awk '{print $3}' || echo "-")
        gait=$(grep "unstable_gait " "$f" | tail -1 | awk '{print $3}' || echo "-")
        wand=$(grep "wandering " "$f" | tail -1 | awk '{print $3}' || echo "-")
        sit=$(grep "sitting_standing " "$f" | tail -1 | awk '{print $3}' || echo "-")

        printf "%-4s %-22s %-10s %-10s %-8s %-8s %-8s %-8s %-8s %-8s %-8s\n" \
            "$num" "$name" "$val" "$test" "$fall" "$eat" "$work" "$aggr" "$gait" "$wand" "$sit" >> "$SUMMARY"
    done

    echo "" >> "$SUMMARY"
    echo "Updated: $(date '+%Y-%m-%d %H:%M:%S')" >> "$SUMMARY"

    echo ""
    echo "  ── Current standings ──"
    cat "$SUMMARY"
    echo ""
}

# ── Experiment 0: Baseline (reproduce current best) ──────────────
run_experiment 0 "baseline" \
    "Reproduce current best with worker fix (control run)"

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
    "2x weight on aggression class (index 3) to reduce aggression->working_together confusion" \
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
echo "" | tee -a "$LOG"

# Final summary
update_summary

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  All experiments complete!                               ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                         ║"
echo "║  Review results:                                        ║"
echo "║    cat experiments/summary.txt                          ║"
echo "║    cat experiments/experiment_log.txt                    ║"
echo "║                                                         ║"
echo "║  Copy results to local machine:                         ║"
echo "║    scp -r thunder:~/.kaggle/pose-autoresearch/experiments ./  ║"
echo "║                                                         ║"
echo "║  Deploy winner:                                         ║"
echo "║    cp experiments/exp_N_best_model.pt checkpoints/       ║"
echo "╚══════════════════════════════════════════════════════════╝"
