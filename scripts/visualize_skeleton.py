"""Render COCO-17 skeleton sequences to animated GIF or MP4.

Usage:
    python scripts/visualize_skeleton.py data/processed/sample.json -o output.gif
    python scripts/visualize_skeleton.py data/processed/ -o output_dir/ --batch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Add project root to path so we can import pose_autoresearch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_autoresearch.graph import get_bone_pairs

# Event classes matching prepare.py
EVENT_CLASSES = [
    "fall", "eating", "working_together", "aggression",
    "unstable_gait", "wandering", "sitting_standing",
]


def predict_sample(frames: list[list[list[list[float]]]], model_path: str | Path) -> tuple[str, float, list[tuple[str, float]]]:
    """Run model inference on a pose sequence.

    Args:
        frames: List of frames, each a list of bodies, each body 17 keypoints.
        model_path: Path to best_model.pt checkpoint.

    Returns:
        (predicted_class, confidence, top3) where top3 is [(class, prob), ...].
    """
    import torch

    # Lazy import to avoid loading torch when not needed
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train import PoseEventClassifier

    device = torch.device("cpu")
    model = PoseEventClassifier()
    ckpt = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Build input tensor from first body: (1, seq_len, 51)
    seq_len = 150
    kps = []
    for frame_bodies in frames[:seq_len]:
        body = frame_bodies[0]  # Use first body for prediction
        flat = np.array(body).flatten()
        if len(flat) < 51:
            flat = np.pad(flat, (0, 51 - len(flat)))
        kps.append(flat[:51])
    while len(kps) < seq_len:
        kps.append(np.zeros(51))
    tensor = torch.from_numpy(np.array(kps, dtype=np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top3_idx = probs.argsort(descending=True)[:3]
    top3 = [(EVENT_CLASSES[i], probs[i].item()) for i in top3_idx]
    pred_class = top3[0][0]
    pred_conf = top3[0][1]

    return pred_class, pred_conf, top3


def load_sample(path: str | Path) -> tuple[list[list[list[list[float]]]], str]:
    """Load a JSON pose sample.

    Args:
        path: Path to a JSON file with frames, label, duration.

    Returns:
        (frames, label) where frames is a list of bodies per frame.
        Each body is a list of 17 [x, y, confidence] keypoints.
        Single-body files are normalized to [[body]] per frame.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sample not found: {path}")
    with open(path) as f:
        data = json.load(f)

    frames = []
    for frame in data["frames"]:
        if "bodies" in frame:
            # Multi-body format: frames[].bodies[]
            frames.append(frame["bodies"])
        else:
            # Legacy single-body format: frames[].keypoints
            frames.append([frame["keypoints"]])

    label = data["label"]
    return frames, label


# Body region indices
_HEAD = [0, 1, 2, 3, 4]
_TORSO = [5, 6, 11, 12]
_ARMS = [7, 8, 9, 10]
_LEGS = [13, 14, 15, 16]
_REGIONS = [_HEAD, _TORSO, _ARMS, _LEGS]

# Per-person color palettes (body regions: head, torso, arms, legs)
_PERSON_PALETTES = [
    ["#4FC3F7", "#81C784", "#FFB74D", "#E57373"],  # Person 0: blue/green/orange/red
    ["#CE93D8", "#4DD0E1", "#AED581", "#FFD54F"],  # Person 1: purple/cyan/lime/yellow
    ["#F48FB1", "#80CBC4", "#DCE775", "#FF8A65"],  # Person 2: pink/teal/yellow-green/deep-orange
    ["#B0BEC5", "#A1887F", "#90A4AE", "#BCAAA4"],  # Person 3: greys/browns (fallback)
]

CONF_THRESHOLD = 0.1
BASE_JOINT_SIZE = 80


def _get_person_colors(person_idx: int) -> tuple[dict[int, str], dict[int, str]]:
    """Get joint and bone color maps for a specific person index."""
    palette = _PERSON_PALETTES[person_idx % len(_PERSON_PALETTES)]
    joint_colors = {}
    for region_idx, region in enumerate(_REGIONS):
        for joint_idx in region:
            joint_colors[joint_idx] = palette[region_idx]
    return joint_colors, joint_colors  # child_color same mapping


def draw_skeleton(
    ax: plt.Axes,
    keypoints: list[list[float]],
    person_idx: int = 0,
) -> dict:
    """Draw a single skeleton frame on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        keypoints: 17 keypoints, each [x, y, confidence].
        person_idx: Index of the person (for color palette selection).

    Returns:
        Dict with 'joints' (PathCollection) and 'bones' (list of Line2D).
    """
    joint_colors, child_colors = _get_person_colors(person_idx)

    kps = np.array(keypoints)  # (17, 3)
    xs, ys, confs = kps[:, 0], kps[:, 1], kps[:, 2]

    # Joint sizes: scale by confidence, hide low-confidence
    sizes = np.where(confs >= CONF_THRESHOLD, BASE_JOINT_SIZE * confs, 0)
    colors = [joint_colors.get(i, "#FFFFFF") for i in range(len(kps))]

    joints = ax.scatter(xs, ys, s=sizes, c=colors, zorder=5, edgecolors="white", linewidths=0.5)

    # Bones
    bone_pairs = get_bone_pairs()
    bones = []
    for parent, child in bone_pairs:
        if confs[parent] < CONF_THRESHOLD or confs[child] < CONF_THRESHOLD:
            continue
        color = child_colors.get(child, "#FFFFFF")
        line, = ax.plot(
            [xs[parent], xs[child]],
            [ys[parent], ys[child]],
            color=color, linewidth=2, zorder=3, alpha=0.8,
        )
        bones.append(line)

    return {"joints": joints, "bones": bones}


BG_COLOR = "#1a1a2e"


def render_animation(
    frames: list[list[list[list[float]]]],
    label: str,
    output_path: str,
    fps: int = 30,
    figsize: tuple[float, float] = (8, 6),
    dpi: int = 100,
    show_overlay: bool = True,
    prediction: tuple[str, float, list[tuple[str, float]]] | None = None,
) -> None:
    """Render a skeleton sequence to GIF or MP4.

    Args:
        frames: List of frames, each a list of bodies, each body a list
                of 17 [x, y, conf] keypoints.
        label: Action class label for overlay text.
        output_path: Output file path (.gif or .mp4).
        fps: Playback framerate.
        figsize: Figure size in inches (width, height).
        dpi: Resolution in dots per inch.
        show_overlay: Whether to draw text overlay.
        prediction: Optional (pred_class, pred_conf, top3) from predict_sample.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    # Compute global bounds across all frames and all bodies for stable view
    all_xs, all_ys = [], []
    for frame_bodies in frames:
        for body in frame_bodies:
            kps = np.array(body)
            # Only consider visible joints for bounds
            visible = kps[:, 2] >= CONF_THRESHOLD
            if visible.any():
                all_xs.extend(kps[visible, 0].tolist())
                all_ys.extend(kps[visible, 1].tolist())

    if all_xs:
        x_range = max(all_xs) - min(all_xs)
        y_range = max(all_ys) - min(all_ys)
        pad = max(x_range, y_range) * 0.1 + 10  # padding proportional to data range
        x_min, x_max = min(all_xs) - pad, max(all_xs) + pad
        y_min, y_max = min(all_ys) - pad, max(all_ys) + pad
    else:
        x_min, x_max, y_min, y_max = 0, 1, 0, 1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y so head is at top

    num_frames = len(frames)

    def init():
        return []

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

        frame_bodies = frames[frame_idx]
        for person_idx, body_kps in enumerate(frame_bodies):
            draw_skeleton(ax, body_kps, person_idx=person_idx)

        if show_overlay:
            # Mean confidence across all bodies
            all_confs = []
            for body in frame_bodies:
                kps = np.array(body)
                all_confs.extend(kps[:, 2].tolist())
            mean_conf = np.mean(all_confs) if all_confs else 0.0

            bbox_props = dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6)

            # Ground truth label (top-left)
            ax.text(
                0.02, 0.98, f"GT: {label}", transform=ax.transAxes,
                fontsize=14, fontweight="bold", color="white",
                verticalalignment="top", bbox=bbox_props,
            )

            # Frame counter + body count (top-right)
            n_bodies = len(frame_bodies)
            frame_text = f"{frame_idx + 1} / {num_frames}"
            if n_bodies > 1:
                frame_text += f"  ({n_bodies} bodies)"
            ax.text(
                0.98, 0.98, frame_text,
                transform=ax.transAxes,
                fontsize=11, color="white",
                verticalalignment="top", horizontalalignment="right",
                bbox=bbox_props,
            )

            # Model prediction (bottom-right) if available
            if prediction:
                pred_class, pred_conf, top3 = prediction
                correct = pred_class == label
                pred_color = "#81C784" if correct else "#E57373"  # green/red
                pred_bbox = dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)

                ax.text(
                    0.98, 0.16, f"Pred: {pred_class} ({pred_conf:.0%})",
                    transform=ax.transAxes,
                    fontsize=13, fontweight="bold", color=pred_color,
                    verticalalignment="bottom", horizontalalignment="right",
                    bbox=pred_bbox,
                )
                top3_str = "  ".join(f"{c}: {p:.0%}" for c, p in top3)
                ax.text(
                    0.98, 0.02, top3_str,
                    transform=ax.transAxes,
                    fontsize=9, color="#B0BEC5",
                    verticalalignment="bottom", horizontalalignment="right",
                    bbox=bbox_props,
                )

            # Keypoint confidence (bottom-left)
            ax.text(
                0.02, 0.02, f"kp conf: {mean_conf:.2f}",
                transform=ax.transAxes,
                fontsize=11, color="white",
                verticalalignment="bottom", bbox=bbox_props,
            )

        return []

    anim = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)

    output_path = Path(output_path)
    if output_path.suffix == ".gif":
        anim.save(str(output_path), writer="pillow", fps=fps)
    elif output_path.suffix == ".mp4":
        anim.save(str(output_path), writer="ffmpeg", fps=fps)
    else:
        raise ValueError(f"Unsupported format: {output_path.suffix}. Use .gif or .mp4")

    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render skeleton sequences to GIF/MP4.")
    parser.add_argument("input", help="Path to a JSON sample file or directory (with --batch)")
    parser.add_argument("-o", "--output", required=True, help="Output file path (.gif/.mp4) or directory (with --batch)")
    parser.add_argument("--batch", action="store_true", help="Process all JSON files in input directory")
    parser.add_argument("--fps", type=int, default=30, help="Playback framerate (default: 30)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[8, 6], metavar=("W", "H"), help="Figure size in inches (default: 8 6)")
    parser.add_argument("--dpi", type=int, default=100, help="Resolution (default: 100)")
    parser.add_argument("--no-overlay", action="store_true", help="Hide text overlay")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif", help="Output format for batch mode (default: gif)")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint for detection overlay")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.batch:
        if not input_path.is_dir():
            parser.error(f"--batch requires a directory, got: {input_path}")
        output_path.mkdir(parents=True, exist_ok=True)
        json_files = sorted(input_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {input_path}")
            return
        ext = f".{args.format}"
        for i, jf in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] {jf.name}")
            frames, label = load_sample(jf)
            prediction = None
            if args.model:
                prediction = predict_sample(frames, args.model)
                print(f"  Prediction: {prediction[0]} ({prediction[1]:.0%})")
            out_file = output_path / f"{jf.stem}{ext}"
            render_animation(
                frames, label, str(out_file),
                fps=args.fps, figsize=tuple(args.figsize), dpi=args.dpi,
                show_overlay=not args.no_overlay,
                prediction=prediction,
            )
    else:
        frames, label = load_sample(input_path)
        prediction = None
        if args.model:
            prediction = predict_sample(frames, args.model)
            print(f"Prediction: {prediction[0]} ({prediction[1]:.0%})")
            for cls, prob in prediction[2]:
                print(f"  {cls}: {prob:.1%}")
        render_animation(
            frames, label, str(output_path),
            fps=args.fps, figsize=tuple(args.figsize), dpi=args.dpi,
            show_overlay=not args.no_overlay,
            prediction=prediction,
        )


if __name__ == "__main__":
    main()
