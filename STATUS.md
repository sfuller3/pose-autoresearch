# Project status

**Read this first when resuming work.** Updated at the end of each working session.

Last updated: 2026-07-24

## Where things stand

Pose-based event detection: a multi-scale temporal CNN over COCO-17 keypoints
(+ bone and velocity channels, 303-dim dual-body input) classifying 7 events.
Fall detection is the priority class.

**Current best:** `wide-90min` — val 0.9629 / test 0.9659, fall recall 98.5%.
Checkpoint: `checkpoints/best_model_wide-90min.pt`. Same default config as
`wide-30min` (block channels 192/192/384/384, bs 256, lr 3e-3, AMP + EMA +
OneCycleLR, focal loss with dynamic class weights and a 1.5x fall boost), just
trained longer — 331 epochs, best val at epoch 299. The run was interrupted
before its final test eval, so it was evaluated and recorded post-hoc via
`scripts/eval_checkpoint.py` (2026-07-24).

Previous best was `wide-30min` (val 0.9579 / test 0.9590), still on the
leaderboard.

Full history: `experiments/LEADERBOARD.md` (generated from
`experiments/runs.jsonl`).

## Weak classes (from the best run's test set)

| class | acc | |
|---|---|---|
| fall | 0.985 | effectively solved — protect this number |
| working_together | 0.983 | fine |
| sitting_standing | 0.976 | fine |
| unstable_gait | 0.971 | recovered — see below |
| eating | 0.963 | fine |
| aggression | 0.955 | **weak** |
| wandering | 0.943 | **weak** |

(Per-class from the `wide-90min` test set.)

The longer run reshuffled the weak classes. `unstable_gait`, previously one of
the three weak spots (0.945), jumped to 0.971 with more training — consistent
with it being a long-horizon pattern that benefits from more optimization.
`wandering`, its sibling from the exp18 class-mapping correction, actually
*dropped* slightly (0.951 → 0.943) and is now the single weakest class.
`aggression` barely moved (0.944 → 0.955) and remains weak.

So the remaining headroom is now `wandering` and `aggression`.

### Confusion-matrix diagnostic (2026-07-24, `scripts/confusion_matrix.py`)

Ran the full test set through `wide-90min`. **The errors are concentrated, not
diffuse, and they overturn the "wandering needs a longer window" theory.** The
three two-body-interaction classes form an error triangle:

- `wandering → aggression`: 55 (4.3% of wandering) — its single dominant error
- `aggression → working_together`: 47 (3.9%) — confirms the long-standing guess
- `wandering → working_together`: 18 (1.4%)
- `working_together → aggression / wandering`: 9 / 9

Nearly all remaining headroom lives in `{working_together, aggression,
wandering}` — the model struggles to disambiguate *what the two tracked bodies
are doing relative to each other*, not sequence length. Non-interaction errors
are minor: `fall ↔ sitting_standing` (10–12 each way) and a small reciprocal
`eating ↔ unstable_gait` (11 each way).

**Implication:** the higher-value next experiment is explicit **inter-body
interaction features** (relative distance, closing velocity, relative
orientation between the two bodies) rather than a longer temporal window. The
303-dim input concatenates two bodies but may not encode their interaction.

## Next steps

1. ~~**Longer wide run**~~ — **DONE**. `wide-90min` is the new best (test 0.9659,
   fall 0.985). Note val was *still* trending up when the run was interrupted at
   epoch 331 (best at 299), so an even longer run may still have headroom, but
   with diminishing returns — the val curve had flattened to ~0.960–0.963 for
   the last ~30 epochs.
2. **Inter-body interaction features** (NEW top priority — see the confusion
   diagnostic above). Add explicit relative-geometry channels between the two
   tracked bodies: pairwise distance (e.g. centroid + wrist-to-wrist), closing
   velocity, and relative orientation. This targets the entire
   `{working_together, aggression, wandering}` error triangle at once, which is
   where essentially all remaining headroom is.
3. **Longer temporal window** (`SEQ_LEN=300`) — now demoted. The matrix shows
   `wandering`'s errors go to `aggression`, not a diffuse temporal smear, so a
   longer window is a weaker bet than #2. Still worth trying if #2 stalls.
4. **Aggression → working_together** confirmed as the dominant aggression error
   (47 samples). Whatever helps #2 should help this directly.

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
