# pose-autoresearch

Pose-based event detection (7 classes, fall detection is the priority class).

## Start here

**Read `STATUS.md` first** — it holds the current best model, the weak classes,
the agreed next steps, and how to launch a run. It is the session-to-session
memory for this project; update it at the end of a working session.

`experiments/LEADERBOARD.md` has the full run history (generated from
`experiments/runs.jsonl` by `python scripts/leaderboard.py`).

## Running training

Always set `POSE_RUN_NAME` — it scopes the checkpoint filename. Two runs
without it write to the same file and silently clobber each other.

```bash
source .venv/bin/activate
POSE_RUN_NAME=<name> POSE_RUN_NOTES="<why>" POSE_AUTORESEARCH_MAX_TIME=1800 \
  python train.py 2>&1 | tee /tmp/run_<name>.log
```

Model width, batch size, and other knobs are env-overridable — see the table in
`STATUS.md`. Prefer an env override to editing `train.py` for a one-off sweep;
edit the defaults only when promoting a config that won.

## After a run

1. `python scripts/leaderboard.py` — refreshes the leaderboard.
2. If it won, promote it: copy its checkpoint to `checkpoints/best_model.pt`,
   fold the config into `train.py` defaults, update `STATUS.md`.
3. If it lost, leave the run log entry — negative results are the point of it.
