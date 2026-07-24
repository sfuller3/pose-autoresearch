"""Confusion matrix + top-confusion breakdown for a trained checkpoint.

Diagnostic, read-only w.r.t. the model: loads a checkpoint, runs the test set,
and reports where each class's errors actually go. Mirrors the forward/batch
handling in prepare.evaluate_per_class so dual-body inputs are handled.

Usage:
    POSE_CHECKPOINT=checkpoints/best_model_wide-90min.pt \
        python scripts/confusion_matrix.py
"""
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("POSE_RUN_NAME", "wide-90min")
os.environ.setdefault("POSE_CHECKPOINT", "checkpoints/best_model_wide-90min.pt")

import numpy as np
import torch

import train as T
from prepare import EVENT_CLASSES


def collect_predictions(model, loader, device):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch[0], (tuple, list)):
                inputs = tuple(t.to(device) for t in batch[0])
                labels = batch[1].to(device)
                logits = model(*inputs)
            else:
                poses, labels = batch
                logits = model(poses.to(device))
                labels = labels.to(device)
            ps.append(torch.argmax(logits, dim=1).cpu())
            ys.append(labels.cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ckpt = torch.load(T.CHECKPOINT_PATH, weights_only=True, map_location=T.DEVICE)
    n = len(EVENT_CLASSES)

    splits_dir = Path("data/splits")
    n_bodies = 2 if (splits_dir / "train").exists() else 1
    test_ds = T.MultiPersonPoseDataset(
        splits_dir / "test", seq_len=T.SEQ_LEN, augment=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=T.BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=T.DEVICE.type == "cuda",
    )
    env_dim = 32 if Path("data/env_features").exists() else 0
    model = T.PoseEventClassifier(
        dropout=T.DROPOUT, env_dim=env_dim, n_bodies=n_bodies
    ).to(T.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    y, p = collect_predictions(model, test_loader, T.DEVICE)

    cm = np.zeros((n, n), dtype=int)
    for yi, pi in zip(y, p):
        cm[yi, pi] += 1

    short = [c[:8] for c in EVENT_CLASSES]
    print(f"Checkpoint: {T.CHECKPOINT_PATH}  ({len(y)} test samples)\n")
    print("Rows = true class, Cols = predicted. Diagonal = correct.\n")
    header = "true \\ pred   " + " ".join(f"{s:>8}" for s in short) + "   | recall"
    print(header)
    print("-" * len(header))
    for i, cls in enumerate(EVENT_CLASSES):
        row_total = cm[i].sum()
        recall = cm[i, i] / row_total if row_total else 0.0
        cells = " ".join(f"{cm[i, j]:>8}" for j in range(n))
        print(f"{cls:>12}  {cells}   | {recall:.3f}")

    print("\nWhere each class's errors go (true class -> top wrong predictions):")
    for i, cls in enumerate(EVENT_CLASSES):
        row_total = cm[i].sum()
        errs = [(cm[i, j], EVENT_CLASSES[j]) for j in range(n) if j != i and cm[i, j] > 0]
        errs.sort(reverse=True)
        if not errs:
            print(f"  {cls:>16}: no errors")
            continue
        frac_wrong = sum(c for c, _ in errs) / row_total
        top = ", ".join(f"{name} ({c}, {c/row_total:.1%})" for c, name in errs[:3])
        print(f"  {cls:>16}: {frac_wrong:.1%} wrong -> {top}")

    # Also report precision-side: what gets misclassified INTO the weak classes.
    print("\nWhat gets misclassified INTO each class (false positives, top sources):")
    for j, cls in enumerate(EVENT_CLASSES):
        col = [(cm[i, j], EVENT_CLASSES[i]) for i in range(n) if i != j and cm[i, j] > 0]
        col.sort(reverse=True)
        pred_total = cm[:, j].sum()
        prec = cm[j, j] / pred_total if pred_total else 0.0
        if not col:
            print(f"  {cls:>16}: precision {prec:.3f}, no false positives")
            continue
        top = ", ".join(f"{name} ({c})" for c, name in col[:3])
        print(f"  {cls:>16}: precision {prec:.3f} <- {top}")


if __name__ == "__main__":
    main()
