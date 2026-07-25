# Run leaderboard

Generated from `experiments/runs.jsonl` (6 runs logged). Regenerate with `python scripts/leaderboard.py`.

**Best so far:** `interaction-feats-90min` — val 0.9634, test 0.9645, checkpoint `checkpoints/best_model_interaction-feats-90min.pt`

| # | run | val | test | fall | eat | work | aggr | gait | wand | sit | channels | ep | min | params | rev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | interaction-feats-90min | 0.9634 | 0.9645 | 0.982 | 0.963 | 0.985 | 0.949 | 0.977 | 0.941 | 0.975 | 192/192/384/384 | 452 | 90 | 4.1M | 2e67042+dirty |
| 2 | wide-90min | 0.9629 | 0.9659 | 0.985 | 0.963 | 0.983 | 0.955 | 0.971 | 0.943 | 0.976 | 192/192/384/384 | 299 | 0 | 4.1M | cee74dc+dirty |
| 3 | interaction-feats-30min-v2 | 0.9582 | 0.9618 | 0.978 | 0.956 | 0.981 | 0.948 | 0.958 | 0.944 | 0.973 | 192/192/384/384 | 117 | 30 | 4.1M | 2e67042+dirty |
| 4 | wide-30min | 0.9579 | 0.9590 | 0.979 | 0.956 | 0.968 | 0.944 | 0.945 | 0.951 | 0.971 | 192/192/384/384 | 95 | 30 | - | d35aef0+dirty |
| 5 | baseline-30min | 0.9536 | - | - | - | - | - | - | - | - | 128/128/256/256 | 95 | 30 | - | d35aef0+dirty |
| 6 | amp-ema-onecycle | 0.9487 | - | 0.978 | - | - | - | - | - | - | 128/128/256/256 | 60 | 15 | - | d35aef0 |

## Notes

- **interaction-feats-90min** — Interaction features (META_DIM 3->7) at 90min budget — apples-to-apples vs wide-90min. Tests whether the inter-body relative-geometry channels hold up with full training; the 30min v2 run beat wide-30min on test (+0.28pt) and on working_together/aggression but regressed wandering.
- **wide-90min** — Longer wide run (STATUS next-step #1); 331 epochs, best val at epoch 299. Test evaluated post-hoc via scripts/eval_checkpoint.py after the original run was interrupted before its final eval/record.
- **interaction-feats-30min-v2** — META_DIM 3->7: closing speed, wrist min-dist, wrist closing speed, hip-velocity motion align. Targets the working_together/aggression/wandering confusion triangle. Rerun after v1 was interrupted at epoch 57.
- **wide-30min** — Wide model 192/384 via train_wide.py, 30min budget. Best so far.
- **baseline-30min** — Same config as amp-ema-onecycle, 30min budget. CRASHED at final eval: the concurrent wide run overwrote checkpoints/best_model.pt (shared path bug, fixed by POSE_RUN_NAME). No test numbers.
- **amp-ema-onecycle** — AMP + EMA(0.999) + OneCycleLR, bs=256 lr=3e-3. Committed baseline.
