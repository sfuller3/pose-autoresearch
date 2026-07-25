# Project status

**Read this first when resuming work.** Updated at the end of each working session.

Last updated: 2026-07-25

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

`wide-90min` **survived a direct challenge on 2026-07-25**: the inter-body
interaction features (`META_DIM` 7) were tested at matched 90-minute budget and
lost on test and on fall recall. Details in the negative-result section below.
Note `LEADERBOARD.md` ranks by val and therefore shows the losing run at #1 —
see the warning there.

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

### Interaction features — first result (2026-07-24, commit `2e67042`)

Acted on the implication above: `META_DIM` 3 → 7, adding four inter-body
channels — centroid closing speed, wrist-pair minimum distance, wrist closing
speed, and hip-velocity motion alignment.

`interaction-feats-30min-v2`: val 0.9582 / test 0.9618, 117 epochs.
(v1 was interrupted at epoch 57 and is not on the leaderboard.)

**Compare it against `wide-30min`, not `wide-90min`** — matched 30-minute
budget. Against the 90-minute champion it looks like a regression, but that is
a budget artifact, not a feature verdict:

| class | wide-30min | interaction-v2 | Δ |
|---|---|---|---|
| working_together | 0.9682 | 0.9811 | **+1.3** |
| unstable_gait | 0.9453 | 0.9583 | **+1.3** |
| aggression | 0.9439 | 0.9480 | +0.4 |
| eating | 0.9555 | 0.9555 | 0.0 |
| fall | 0.9793 | 0.9778 | −0.2 |
| sitting_standing | 0.9711 | 0.9734 | +0.2 |
| wandering | 0.9512 | 0.9442 | **−0.7** |

Overall test +0.28pt at equal budget. The two classes the channels were
designed for moved the right way — `working_together` most of all — and fall
recall held at 97.8%. But `wandering` went the *wrong* way, so the error
triangle is not closed. Verdict: promising, not proven.

The confound is budget. The only comparison available is 30 min vs the
champion's 90, so the feature change has never been tested at full training
length.

### Interaction features — settled at 90 min (2026-07-25). NEGATIVE RESULT.

`interaction-feats-90min`: val 0.9634 / test **0.9645**, 452 epochs, clean run.
The apples-to-apples test against `wide-90min` (val 0.9629 / test 0.9659).

**It did not beat the champion, and the 30-minute result did not replicate.**

| class | wide-90min | interaction-90min | Δ |
|---|---|---|---|
| unstable_gait | 0.9714 | 0.9766 | **+0.52** |
| working_together | 0.9828 | 0.9854 | +0.26 |
| eating | 0.9634 | 0.9634 | 0.00 |
| sitting_standing | 0.9758 | 0.9746 | −0.12 |
| wandering | 0.9434 | 0.9411 | −0.23 |
| fall | 0.9852 | 0.9822 | **−0.30** |
| aggression | 0.9546 | 0.9488 | **−0.58** |

It won val by +0.0005 and lost test by −0.0014 — both inside noise. That split
is the finding: no evidence the channels help at full budget.

The damning part is `aggression`. At 30 min the channels appeared to help it
(+0.4); at 90 min it is the **worst** class delta (−0.58) — sign-flipped.
`working_together`'s +1.3 shrank to +0.26. Read together, the 30-minute signal
was mostly noise. Two further points that close off the easy excuses:
`fall` **dropped to 0.9822** from 0.9852, against the one class this project
protects; and the run trained *longer* than the champion (452 vs 299 epochs),
so "needs more epochs" is not available as an explanation.

**Verdict: the four channels as currently defined do not work.** Not promoted.
`checkpoints/best_model.pt` untouched, `META_DIM` stays 3 in the `train.py`
defaults. The feature code remains on `2e67042` for anyone who wants to iterate
on the channel definitions rather than discard the idea.

> ⚠️ **`LEADERBOARD.md` will disagree with this section.** It sorts by *val*, so
> it lists `interaction-feats-90min` at #1 and prints "Best so far:
> interaction-feats-90min". That is a sorting artifact — the run lost on test
> and on fall. **`wide-90min` is the best model.** Trust this file over the
> leaderboard header until `scripts/leaderboard.py` is taught to rank by test.

## Next steps

1. ~~**Longer wide run**~~ — **DONE**. `wide-90min` is the new best (test 0.9659,
   fall 0.985). Note val was *still* trending up when the run was interrupted at
   epoch 331 (best at 299), so an even longer run may still have headroom, but
   with diminishing returns — the val curve had flattened to ~0.960–0.963 for
   the last ~30 epochs.
2. ~~**Inter-body interaction features**~~ — **DONE, NEGATIVE.** Built
   (`2e67042`), tested at 30 min and 90 min. Lost to `wide-90min` on test and on
   fall; the 30-min gains did not replicate. Not promoted. See the section above.
3. ~~**Settle the interaction features at 90 min**~~ — **DONE 2026-07-25.**
   Answered: no. `wide-90min` remains the best model.
4. **Treat run-to-run noise as a first-class problem** (NEW top priority). This
   session cost 90 minutes to discover that a +0.28pt "win" was noise. Every
   comparison in this project is single-seed, and the gaps that decide promotion
   (0.001–0.003) are the same size as the noise. Before the next feature
   experiment, run `wide-90min`'s exact config 3× under different seeds and
   measure the spread. That number is the minimum bar any future change must
   clear, and without it the leaderboard's ranking is not meaningful.
5. **Fix `wandering` specifically.** Still the weakest class (0.9411–0.9434) and
   unmoved by anything tried so far. Its dominant error is
   `wandering → aggression` (55 samples). The interaction channels made it
   slightly worse in both runs, which weakly suggests closing-speed and
   wrist-distance make an aimless walker near another body look like an
   approach — but treat that as a hypothesis, not a finding, given #4.
6. **Longer temporal window** (`SEQ_LEN=300`) — still unrun, and now the main
   untested structural idea, since the interaction-feature route is closed. The
   confusion matrix argued against it (`wandering`'s errors are concentrated on
   `aggression`, not smeared), but that argument has not actually been tested.
7. **Teach `scripts/leaderboard.py` to rank by test, not val** — small chore,
   but it currently crowns a run that lost. See the warning above.

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
