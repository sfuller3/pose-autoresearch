# Thunder A100 Training Plan

## Overview

Train the pose event classifier on all three real datasets (NTU RGB+D 120, FallVision, Le2i) using a Thunder Compute A100 GPU instance. The model was previously trained on synthetic random data — this plan produces a properly trained model on ~25K+ real pose sequences with source-aware train/val/test splits.

**Estimated cost:** ~$2-5 total (~$1.10/hr × 2-4 hours)
**Estimated time:** 2-4 hours total (setup + data + 100 experiments)

---

## Prerequisites

Before connecting to Thunder, ensure:

1. **Thunder CLI installed** on your Mac:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/Thunder-Compute/thunder-cli/main/scripts/install.sh | bash
   ```

2. **Kaggle API token ready** (needed for Le2i dataset):
   - Go to https://www.kaggle.com/settings → Create New API Token
   - You'll paste the credentials on the Thunder instance

3. **NTU RGB+D 120 pickle** is already in the repo (`data/ntu120/ntu120_2d.pkl`, ~1.8GB). It will be cloned with the repo.

---

## Step-by-Step Execution

### Phase 1: Instance Setup (~5 min)

```bash
# From your Mac
tnr login
tnr create --gpu a100
tnr connect <instance-id>

# On the Thunder instance
git clone https://github.com/sfuller3/pose-autoresearch.git
cd pose-autoresearch

# Set up Kaggle credentials (for Le2i download)
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Phase 2: Automated Pipeline (~30-60 min)

Run the full pipeline script. It handles everything:

```bash
./scripts/run_thunder_pipeline.sh
```

This script executes:
1. **Environment setup** — creates venv, installs PyTorch + ultralytics + deps
2. **Download datasets** — FallVision (Harvard Dataverse, ~2GB) and Le2i (Kaggle, ~1GB)
3. **Convert FallVision** — extracts keypoints from pre-extracted CSVs in RAR archives (~5,200 sequences). Requires `unrar` — install with `apt-get install unrar` if needed.
4. **Convert Le2i** — runs YOLO pose estimation on ~200 video files using A100 GPU (~5-15s/video, ~15 min total). Each video processed in isolated subprocess for crash safety.
5. **Convert NTU RGB+D 120** — converts 2D skeleton pickle to JSON, capped at 3,000 samples/class (~18K total)
6. **Remove synthetic data** — deletes files without `ntu_`/`fv_`/`le2i_` prefix
7. **Source-aware split** — runs `scripts/split_data.py` to create `data/splits/{train,val,test}/`:
   - NTU: 53 official cross-subject training subjects; remaining split 50/50 val/test
   - FallVision: video-level grouping (all segments from same video in same split)
   - Le2i: video-level grouping (same as FallVision)
8. **Train** — runs `python train.py` (5-minute budget, ~150+ epochs on A100)

### Phase 3: Autoresearch Loop (~2-3 hours)

After the pipeline completes and you have a baseline, launch the autonomous experiment loop:

```bash
# Option A: Use the tmux launcher (survives SSH disconnect)
./run_autoresearch.sh

# Option B: Run Claude directly
source .venv/bin/activate
claude 'Read program.md and run the autoresearch loop. Modify only train.py. Run each experiment for 5 minutes max. Keep changes that improve validation accuracy (especially fall recall). Discard regressions. Target: 100 experiments.'
```

**Monitor progress:**
```bash
tmux attach -t autoresearch          # See Claude working
watch -n 1 nvidia-smi                # GPU utilization (separate terminal)
tail -f results.tsv                  # Experiment results
tail -5 checkpoints/experiment_log.txt
```

### Phase 4: Save & Shutdown

```bash
# Snapshot the instance (saves all state including trained models)
# From your Mac:
tnr snapshot <instance-id>
tnr stop <instance-id>
```

**Retrieve results before stopping:**
```bash
# From the Thunder instance, push results
cd pose-autoresearch
git add results.tsv train.py checkpoints/best_model.pt
git commit -m "trained model: val_acc=X.XXXX fall_recall=X.XXXX on NTU+FV+Le2i"
git push origin main
```

---

## What Claude Should Do (Autoresearch Agent Instructions)

When Claude runs on the Thunder instance, it reads `program.md` which contains the full agent directive. Here is the execution contract:

### Before Starting Experiments

1. **Verify data is ready.** Check that `data/splits/{train,val,test}/` exist and contain JSON files. If not, run the pipeline:
   ```bash
   ./scripts/run_thunder_pipeline.sh
   ```

2. **Run a baseline.** Execute `python train.py` once to establish the starting accuracy. Record in `results.tsv`.

3. **Read `results.tsv`** to understand what has already been tried (experiments exp1-exp27 were on synthetic + partial data; the real-data baseline starts fresh from this pipeline).

### During Experiments

- **Modify only `train.py`.** Everything else is fixed.
- **One change per experiment.** Isolate variables.
- **5-minute training budget** per experiment (enforced by `MAX_TIME_BUDGET_SECONDS`).
- **Log every result** in `results.tsv` (commit, val_acc, fall_recall, description, status).
- **Keep/discard rules:**
  - KEEP if val_acc improves ≥0.2% and fall_recall doesn't drop >1%
  - KEEP if fall_recall improves ≥1% even if val_acc is flat
  - DISCARD if fall_recall drops >2% regardless of other gains
  - DISCARD if training crashes or doesn't converge
- **Commit on KEEP:** `git add train.py results.tsv && git commit -m "expN: description"`
- **Revert on DISCARD:** `git checkout train.py`
- **Never stop.** Run continuously until interrupted or 100 experiments done.

### Architecture Direction

The temporal CNN (not GCN) is the correct architecture. Previous experiments proved:
- Temporal CNN beats ST-GCN by +7.3% accuracy on this data
- Conv1D is 3-5x faster on CPU (critical for edge deployment)
- With ~25K samples / 7 classes, graph inductive bias is unnecessary

**Highest-priority improvements to try:**
1. Velocity + bone features are already concatenated (150-dim input) — tune this
2. Focal loss for fall-priority training
3. Multi-scale temporal kernels (3, 7, 15)
4. Squeeze-and-excitation channel attention
5. Confidence gating for noisy YOLO joints
6. Attention pooling over time
7. Mixup/CutMix temporal augmentation
8. Hyperparameter tuning (LR, dropout, weight decay, channel sizes)

### Expected Performance

On synthetic data (meaningless): ~95% val_acc
On real NTU+FV data (exp19 baseline): ~95% val_acc, ~97% fall_recall
On real NTU+FV+Le2i (this pipeline): target ≥95% val_acc, ≥97% fall_recall

The Le2i dataset adds real-world fall videos from 4 environments (coffee room, home, lecture room, office). This should improve fall detection generalization significantly.

---

## Dataset Summary

| Dataset | Source | Classes Covered | Extraction Method | Expected Samples |
|---------|--------|----------------|-------------------|-----------------|
| NTU RGB+D 120 | Pre-processed 2D pickle | All 7 | HRNet keypoints (pre-extracted) | ~18,000 |
| FallVision | Harvard Dataverse (CC0) | fall, sitting_standing | COCO-17 CSV (pre-extracted) | ~5,200 |
| Le2i | Kaggle (academic) | fall, sitting_standing | YOLO pose on video (GPU) | ~1,000-2,000 |
| **Total** | | | | **~24,000-25,000** |

### Class Distribution (approximate)

| Class | NTU | FallVision | Le2i | Total |
|-------|-----|-----------|------|-------|
| fall | 541 | ~1,700 | ~500 | ~2,700 |
| eating | 3,000 | — | — | 3,000 |
| working_together | 3,000 | — | — | 3,000 |
| aggression | 3,000 | — | — | 3,000 |
| unstable_gait | 1,200 | — | — | 1,200 |
| wandering | 3,000 | — | — | 3,000 |
| sitting_standing | 3,000 | ~3,500 | ~800 | ~7,300 |

Dynamic class weighting in `train.py` compensates for imbalance (inverse-frequency + 1.5x fall boost).

---

## File Reference

| File | Role | Modifiable? |
|------|------|-------------|
| `train.py` | Model + training loop | YES (agent modifies) |
| `prepare.py` | Data loading, evaluation | NO |
| `program.md` | Agent directives | NO |
| `results.tsv` | Experiment log | APPEND ONLY |
| `scripts/run_thunder_pipeline.sh` | End-to-end orchestrator | NO |
| `scripts/setup_thunder.sh` | Environment + smoke test | NO |
| `scripts/download_data.sh` | Dataset downloader | NO |
| `scripts/convert_fallvision.py` | FallVision + Le2i → JSON | NO |
| `scripts/convert_ntu120.py` | NTU pickle → JSON | NO |
| `scripts/split_data.py` | Source-aware splitting | NO |
| `checkpoints/best_model.pt` | Best model checkpoint | AUTO (saved by train.py) |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `unrar` not found | `apt-get update && apt-get install -y unrar` |
| Kaggle auth fails | Verify `~/.kaggle/kaggle.json` exists with correct credentials |
| CUDA out of memory | Reduce `BATCH_SIZE` in `train.py` (128→64→32) |
| Le2i video crashes | Expected — subprocess isolation handles it. Check stderr for pattern. |
| No data in splits | Run `python scripts/split_data.py --copy` after conversion |
| `results.tsv` stale | Previous results were on synthetic/partial data. Start fresh baseline. |
| SSH disconnects | Use tmux: `tmux attach -t autoresearch` to reconnect |
| Instance stops | `tnr snapshot` first! Then `tnr start <id>` to resume |
