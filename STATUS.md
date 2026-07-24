# Project status

**Read this first when resuming work.** Updated at the end of each working session.

Last updated: 2026-07-24

## Where things stand

Pose-based event detection: a multi-scale temporal CNN over COCO-17 keypoints
(+ bone and velocity channels, 303-dim dual-body input) classifying 7 events.
Fall detection is the priority class.

**Current best:** `wide-30min` — val 0.9579 / test 0.9590, fall recall 97.9%.
Checkpoint: `checkpoints/best_model_wide-30min.pt`. Config is now the default
in `train.py` (block channels 192/192/384/384, bs 256, lr 3e-3, AMP + EMA +
OneCycleLR, focal loss with dynamic class weights and a 1.5x fall boost).

Full history: `experiments/LEADERBOARD.md` (generated from
`experiments/runs.jsonl`).

## Weak classes (from the best run's test set)

| class | acc | |
|---|---|---|
| fall | 0.979 | effectively solved — protect this number |
| sitting_standing | 0.971 | fine |
| working_together | 0.968 | fine |
| eating | 0.956 | fine |
| wandering | 0.951 | **weak** |
| unstable_gait | 0.945 | **weak** |
| aggression | 0.944 | **weak** |

The three weak classes are where the remaining headroom is. Note the historical
context: `unstable_gait` and `wandering` were the two classes fixed by the
class-mapping correction at exp18 (see `results.tsv`), and they remain the
hardest — they are long-horizon gait/trajectory patterns that a 5-second window
may simply be too short to characterize.

## Next steps

1. **Longer wide run** (in progress / next): val was still trending up at epoch
   95 of the 30-minute run. A 90-minute run at the same width is the cheapest
   remaining win.
2. **Longer temporal window** for `unstable_gait` / `wandering`: try
   `SEQ_LEN=300` (10s) or a dilated/strided branch, since these classes are
   defined by trajectory over time rather than a single motion burst.
3. **Aggression** confusion is most likely with `working_together` (both are
   two-body interactions) — inspect the confusion matrix before tuning blindly.

## How to run an experiment

Every run is identified by `POSE_RUN_NAME`, which determines the checkpoint
filename. **Always set it** — two runs sharing the default path will silently
overwrite each other's weights (this corrupted a run on 2026-07-24).

```bash
source .venv/bin/activate
POSE_RUN_NAME=my-experiment \
POSE_RUN_NOTES="what I changed and why" \
POSE_AUTORESEARCH_MAX_TIME=1800 \
python train.py 2>&1 | tee /tmp/run_my-experiment.log
```

Environment overrides:

| var | default | effect |
|---|---|---|
| `POSE_RUN_NAME` | `default` | run id; sets `checkpoints/best_model_<name>.pt` |
| `POSE_RUN_NOTES` | `""` | free text recorded in the run log |
| `POSE_CHECKPOINT` | derived | explicit checkpoint path |
| `POSE_RUN_LOG` | `experiments/runs.jsonl` | run log path |
| `POSE_BLOCK_CHANNELS` | `192,192,384,384` | model width, no code edit needed |
| `POSE_AUTORESEARCH_MAX_TIME` | `300` | wall-clock training budget (seconds) |
| `POSE_DEVICE` | auto | `cpu` to force CPU |

On completion `train.py` appends a full record (config, val/test, per-class,
epochs, params, git rev) to `experiments/runs.jsonl`. Then:

```bash
python scripts/leaderboard.py    # prints the table, refreshes LEADERBOARD.md
```

## Tracking files

| file | role |
|---|---|
| `STATUS.md` | this file — narrative state and next steps, updated per session |
| `experiments/runs.jsonl` | append-only machine-written record of every run |
| `experiments/LEADERBOARD.md` | generated table, sorted by val accuracy |
| `results.tsv` | historical keep/discard log from the autoresearch loop (pre-dates the run log; kept for the exp1–exp27 history) |
