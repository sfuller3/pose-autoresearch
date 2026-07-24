"""Post-hoc test evaluation of an already-trained checkpoint.

Used to record the `wide-90min` run, which finished training (best val at
epoch 299) but was interrupted before train.py ran its final test evaluation
and appended to experiments/runs.jsonl. Reuses train.py's own model, data,
and record helpers so the emitted record is identical in shape to a live run.
"""
import os
from pathlib import Path

# Must be set before importing train (module-level globals read these).
os.environ.setdefault("POSE_RUN_NAME", "wide-90min")
os.environ.setdefault("POSE_CHECKPOINT", "checkpoints/best_model_wide-90min.pt")

import torch

import train as T


def main():
    ckpt = torch.load(T.CHECKPOINT_PATH, weights_only=True, map_location=T.DEVICE)
    block_channels = tuple(ckpt.get("block_channels", T.BLOCK_CHANNELS))
    val_acc = ckpt["val_acc"]
    best_epoch = ckpt.get("epoch")
    print(f"Checkpoint: {T.CHECKPOINT_PATH}")
    print(f"  run_name={ckpt.get('run_name')} best_epoch={best_epoch} "
          f"val_acc={val_acc:.4f} block_channels={block_channels}")

    if list(block_channels) != list(T.BLOCK_CHANNELS):
        raise SystemExit(
            f"Width mismatch: checkpoint {block_channels} != configured "
            f"{T.BLOCK_CHANNELS}. Re-run with "
            f"POSE_BLOCK_CHANNELS={','.join(map(str, block_channels))}"
        )

    # Data — mirror train.main()'s split detection.
    splits_dir = Path("data/splits")
    n_bodies = 1
    if splits_dir.exists() and (splits_dir / "train").exists():
        n_bodies = 2
        test_ds = T.MultiPersonPoseDataset(
            splits_dir / "test", seq_len=T.SEQ_LEN, augment=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=T.BATCH_SIZE, shuffle=False, num_workers=4,
            pin_memory=T.DEVICE.type == "cuda",
        )
    else:
        _, _, test_loader = T.get_dataloaders(
            batch_size=T.BATCH_SIZE, num_workers=2, augment_train=False
        )

    env_dim = 32 if Path("data/env_features").exists() else 0
    model = T.PoseEventClassifier(
        dropout=T.DROPOUT, env_dim=env_dim, n_bodies=n_bodies
    ).to(T.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    num_params = sum(p.numel() for p in model.parameters())

    test_acc, test_loss = T.evaluate_model(model, test_loader, T.DEVICE)
    per_class = T.evaluate_per_class(model, test_loader, T.DEVICE)

    print(f"\nTest: {len(test_loader.dataset)} samples")
    print(f"Best Val: {val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f} | Test Loss: {test_loss:.4f}\n")
    print("Per-class accuracy:")
    for cls, acc in per_class.items():
        print(f"  {cls:20s}: {acc:.4f}")

    T.record_run(
        best_val_acc=val_acc,
        test_acc=test_acc,
        test_loss=test_loss,
        per_class=per_class,
        epochs=best_epoch or 0,
        elapsed=0.0,  # post-hoc eval; wall-clock of the interrupted run is unknown
        num_params=num_params,
    )


if __name__ == "__main__":
    main()
