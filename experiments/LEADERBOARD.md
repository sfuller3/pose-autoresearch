# Run leaderboard

Generated from `experiments/runs.jsonl` (4 runs logged). Regenerate with `python scripts/leaderboard.py`.

**Best so far:** `wide-90min` — val 0.9629, test 0.9659, checkpoint `checkpoints/best_model_wide-90min.pt`

| # | run | val | test | fall | eat | work | aggr | gait | wand | sit | channels | ep | min | params | rev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | wide-90min | 0.9629 | 0.9659 | 0.985 | 0.963 | 0.983 | 0.955 | 0.971 | 0.943 | 0.976 | 192/192/384/384 | 299 | 0 | 4.1M | cee74dc+dirty |
| 2 | wide-30min | 0.9579 | 0.9590 | 0.979 | 0.956 | 0.968 | 0.944 | 0.945 | 0.951 | 0.971 | 192/192/384/384 | 95 | 30 | - | d35aef0+dirty |
| 3 | baseline-30min | 0.9536 | - | - | - | - | - | - | - | - | 128/128/256/256 | 95 | 30 | - | d35aef0+dirty |
| 4 | amp-ema-onecycle | 0.9487 | - | 0.978 | - | - | - | - | - | - | 128/128/256/256 | 60 | 15 | - | d35aef0 |

## Notes

- **wide-90min** — Longer wide run (STATUS next-step #1); 331 epochs, best val at epoch 299. Test evaluated post-hoc via scripts/eval_checkpoint.py after the original run was interrupted before its final eval/record.
- **wide-30min** — Wide model 192/384 via train_wide.py, 30min budget. Best so far.
- **baseline-30min** — Same config as amp-ema-onecycle, 30min budget. CRASHED at final eval: the concurrent wide run overwrote checkpoints/best_model.pt (shared path bug, fixed by POSE_RUN_NAME). No test numbers.
- **amp-ema-onecycle** — AMP + EMA(0.999) + OneCycleLR, bs=256 lr=3e-3. Committed baseline.
