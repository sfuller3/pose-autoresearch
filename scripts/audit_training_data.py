#!/usr/bin/env python3
"""
Pre-training data audit: visually verify what the model will learn from.

Generates:
  1. data_audit/class_distribution.png — per-class, per-source sample counts
  2. data_audit/samples/<class>/*.gif  — 3 random GIF samples per class
  3. data_audit/split_summary.txt     — text summary of train/val/test splits
  4. data_audit/confusion_matrix.png  — (post-training only, with --checkpoint)

Usage:
  # Before training: audit the data
  python scripts/audit_training_data.py

  # After training: add confusion matrix
  python scripts/audit_training_data.py --checkpoint checkpoints/best_model.pt

  # Quick mode (static frames instead of GIFs, much faster)
  python scripts/audit_training_data.py --quick
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVENT_CLASSES = [
    "fall", "eating", "working_together", "aggression",
    "unstable_gait", "wandering", "sitting_standing",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(EVENT_CLASSES)}

OUTPUT_DIR = Path("data_audit")


def gather_split_stats(splits_dir: Path) -> dict:
    """Collect per-split, per-source, per-class counts."""
    stats = {}
    for split in ["train", "val", "test"]:
        split_dir = splits_dir / split
        if not split_dir.exists():
            continue
        counts = Counter()
        source_counts = defaultdict(Counter)
        frame_lengths = []
        for f in split_dir.glob("*.json"):
            with open(f) as fh:
                d = json.load(fh)
            label = d["label"]
            prefix = f.name.split("_")[0]
            counts[label] += 1
            source_counts[prefix][label] += 1
            frame_lengths.append(len(d["frames"]))
        stats[split] = {
            "counts": counts,
            "source_counts": dict(source_counts),
            "total": sum(counts.values()),
            "frame_lengths": frame_lengths,
        }
    return stats


def write_split_summary(stats: dict, out_path: Path):
    """Write a human-readable text summary."""
    with open(out_path, "w") as f:
        for split, s in stats.items():
            f.write(f"{'='*60}\n")
            f.write(f"{split.upper()} — {s['total']} samples\n")
            f.write(f"{'='*60}\n\n")

            f.write("By class:\n")
            for cls in EVENT_CLASSES:
                n = s["counts"].get(cls, 0)
                pct = n / s["total"] * 100 if s["total"] else 0
                bar = "#" * int(pct / 2)
                f.write(f"  {cls:20s}: {n:5d} ({pct:5.1f}%) {bar}\n")
            f.write("\n")

            f.write("By source:\n")
            for src, sc in sorted(s["source_counts"].items()):
                total_src = sum(sc.values())
                f.write(f"  {src}: {total_src} — {dict(sc)}\n")
            f.write("\n")

            lengths = s["frame_lengths"]
            if lengths:
                f.write(f"Sequence lengths: min={min(lengths)}, "
                        f"max={max(lengths)}, "
                        f"mean={np.mean(lengths):.0f}, "
                        f"median={np.median(lengths):.0f}\n")
            f.write("\n")
    print(f"  Saved: {out_path}")


def plot_class_distribution(stats: dict, out_path: Path):
    """Bar chart: sample counts per class, colored by source."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(stats), figsize=(6 * len(stats), 6),
                             sharey=True)
    if len(stats) == 1:
        axes = [axes]

    all_sources = set()
    for s in stats.values():
        all_sources.update(s["source_counts"].keys())
    sources = sorted(all_sources)
    source_colors = {
        "ntu": "#4FC3F7",
        "fv": "#81C784",
        "le2i": "#FFB74D",
    }

    for ax, (split, s) in zip(axes, stats.items()):
        x = np.arange(len(EVENT_CLASSES))
        width = 0.8 / max(len(sources), 1)
        for i, src in enumerate(sources):
            sc = s["source_counts"].get(src, {})
            vals = [sc.get(cls, 0) for cls in EVENT_CLASSES]
            color = source_colors.get(src, "#B0BEC5")
            ax.bar(x + i * width, vals, width, label=src, color=color, alpha=0.85)

        ax.set_title(f"{split.upper()} ({s['total']} samples)", fontsize=14)
        ax.set_xticks(x + width * (len(sources) - 1) / 2)
        ax.set_xticklabels(EVENT_CLASSES, rotation=45, ha="right", fontsize=9)
        ax.legend(title="Source")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Training Data Distribution by Class and Source", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def render_sample_gifs(splits_dir: Path, out_dir: Path, n_per_class: int = 3,
                       quick: bool = False):
    """Render N random samples per class as GIF (or static PNG in quick mode)."""
    from scripts.visualize_skeleton import load_sample, render_animation

    train_dir = splits_dir / "train"
    if not train_dir.exists():
        print("  No train split found, skipping sample rendering.")
        return

    # Group files by label
    by_class = defaultdict(list)
    for f in sorted(train_dir.glob("*.json")):
        with open(f) as fh:
            label = json.load(fh)["label"]
        by_class[label].append(f)

    random.seed(42)

    for cls in EVENT_CLASSES:
        cls_files = by_class.get(cls, [])
        if not cls_files:
            print(f"  {cls}: no samples found")
            continue

        samples = random.sample(cls_files, min(n_per_class, len(cls_files)))
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i, sample_path in enumerate(samples):
            frames, label = load_sample(sample_path)

            if quick:
                # Static PNG: show 4 evenly-spaced frames
                _render_static_strip(frames, label, sample_path.stem,
                                     cls_dir / f"{sample_path.stem}.png")
            else:
                out_file = cls_dir / f"{sample_path.stem}.gif"
                render_animation(frames, label, str(out_file), fps=15, dpi=80,
                                 figsize=(6, 4.5))

        print(f"  {cls}: {len(samples)} samples → {cls_dir}/")


def _render_static_strip(frames, label, name, out_path):
    """Render 4 evenly-spaced frames as a horizontal strip (fast)."""
    import matplotlib.pyplot as plt
    from scripts.visualize_skeleton import draw_skeleton, BG_COLOR, CONF_THRESHOLD

    n = len(frames)
    indices = [0, n // 3, 2 * n // 3, n - 1]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.patch.set_facecolor(BG_COLOR)

    # Compute global bounds
    all_xs, all_ys = [], []
    for frame_bodies in frames:
        for body in frame_bodies:
            kps = np.array(body)
            visible = kps[:, 2] >= CONF_THRESHOLD
            if visible.any():
                all_xs.extend(kps[visible, 0].tolist())
                all_ys.extend(kps[visible, 1].tolist())
    if all_xs:
        pad = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys)) * 0.1 + 10
        x_min, x_max = min(all_xs) - pad, max(all_xs) + pad
        y_min, y_max = min(all_ys) - pad, max(all_ys) + pad
    else:
        x_min, x_max, y_min, y_max = 0, 1, 0, 1

    for ax, idx in zip(axes, indices):
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        for person_idx, body in enumerate(frames[idx]):
            draw_skeleton(ax, body, person_idx=person_idx)
        ax.set_title(f"Frame {idx + 1}/{n}", color="white", fontsize=9)

    fig.suptitle(f"{label} — {name}", color="white", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=100, facecolor=BG_COLOR, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(checkpoint_path: Path, splits_dir: Path, out_path: Path,
                          backbone: str = "cnn"):
    """Generate confusion matrix from trained model on test set."""
    import torch
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prepare import DEVICE
    from train import PoseEventClassifier, STGCNClassifier, MultiPersonPoseDataset

    test_dir = splits_dir / "test"
    if not test_dir.exists():
        print("  No test split found, skipping confusion matrix.")
        return

    # Load model
    if backbone == "gcn":
        model = STGCNClassifier().to(DEVICE)
    else:
        model = PoseEventClassifier(n_bodies=2).to(DEVICE)
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load test data
    test_ds = MultiPersonPoseDataset(test_dir, augment=False)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Build confusion matrix
    n_classes = len(EVENT_CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # Normalize rows (recall-based)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)

    # Per-class accuracy
    acc = np.diag(cm).sum() / cm.sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # Annotate cells with count and percentage
    for i in range(n_classes):
        for j in range(n_classes):
            count = cm[i][j]
            pct = cm_norm[i][j]
            color = "white" if pct > 0.5 else "black"
            ax.text(j, i, f"{count}\n{pct:.0%}", ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(EVENT_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(EVENT_CLASSES)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — Test Set (acc={acc:.1%})", fontsize=14)

    fig.colorbar(im, ax=ax, label="Recall")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit training data before/after training")
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Model checkpoint for post-training confusion matrix")
    parser.add_argument("--quick", action="store_true",
                        help="Static PNG strips instead of animated GIFs (faster)")
    parser.add_argument("--backbone", choices=["cnn", "gcn"], default="cnn",
                        help="Backbone of the checkpoint being audited")
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Training Data Audit                                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # 1. Split statistics
    print("▶ Gathering split statistics...")
    stats = gather_split_stats(args.splits_dir)
    if not stats:
        print("  ERROR: No splits found at", args.splits_dir)
        return
    write_split_summary(stats, args.output_dir / "split_summary.txt")
    print()

    # 2. Class distribution chart
    print("▶ Plotting class distribution...")
    plot_class_distribution(stats, args.output_dir / "class_distribution.png")
    print()

    # 3. Sample visualizations per class
    print("▶ Rendering sample sequences per class...")
    fmt = "PNG strips" if args.quick else "GIFs"
    print(f"  ({fmt}, {args.samples_per_class} per class)")
    render_sample_gifs(args.splits_dir, args.output_dir / "samples",
                       n_per_class=args.samples_per_class, quick=args.quick)
    print()

    # 4. Confusion matrix (post-training only)
    if args.checkpoint:
        print("▶ Generating confusion matrix from trained model...")
        plot_confusion_matrix(args.checkpoint, args.splits_dir,
                              args.output_dir / "confusion_matrix.png",
                              backbone=args.backbone)
        print()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Audit Complete                                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Output: {args.output_dir}/")
    print("║")
    print("║  Review:")
    print("║    split_summary.txt      — counts per class/source/split")
    print("║    class_distribution.png — visual bar chart")
    print("║    samples/<class>/*.gif  — stick figure animations")
    if args.checkpoint:
        print("║    confusion_matrix.png  — where the model gets confused")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
