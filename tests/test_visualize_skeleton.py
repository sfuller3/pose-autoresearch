"""Tests for skeleton visualization script."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


def _load_mod():
    """Helper to import the visualize_skeleton module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "visualize_skeleton",
        str(Path(__file__).resolve().parents[1] / "scripts" / "visualize_skeleton.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_load_sample_returns_frames_and_label(tmp_path):
    """load_sample should parse JSON and return frames array + label string."""
    sample = {
        "frames": [
            {"keypoints": [[0.5, 0.5, 0.9]] * 17, "timestamp": 0.0},
            {"keypoints": [[0.6, 0.4, 0.8]] * 17, "timestamp": 0.033},
        ],
        "label": "fall",
        "duration": 0.066,
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample))

    mod = _load_mod()
    frames, label = mod.load_sample(path)

    assert label == "fall"
    assert len(frames) == 2
    assert len(frames[0]) == 17
    assert len(frames[0][0]) == 3
    assert frames[0][0][0] == pytest.approx(0.5)


def test_load_sample_missing_file():
    """load_sample should raise FileNotFoundError for missing files."""
    mod = _load_mod()

    with pytest.raises(FileNotFoundError):
        mod.load_sample("/nonexistent/path.json")


def test_draw_skeleton_creates_artists():
    """draw_skeleton should add scatter and line artists to the axes."""
    mod = _load_mod()
    fig, ax = plt.subplots()

    # 17 keypoints, all visible (conf=0.9)
    keypoints = [[0.3 + i * 0.02, 0.5 - i * 0.01, 0.9] for i in range(17)]

    artists = mod.draw_skeleton(ax, keypoints)

    # Should return a dict with 'joints' (PathCollection) and 'bones' (list of Line2D)
    assert "joints" in artists
    assert "bones" in artists
    assert len(artists["bones"]) > 0
    plt.close(fig)


def test_draw_skeleton_hides_low_confidence():
    """Keypoints with confidence < 0.1 should not be drawn."""
    mod = _load_mod()
    fig, ax = plt.subplots()

    # All keypoints have zero confidence
    keypoints = [[0.5, 0.5, 0.0]] * 17

    artists = mod.draw_skeleton(ax, keypoints)

    # Joints scatter should have sizes of 0 for invisible points
    sizes = artists["joints"].get_sizes()
    assert all(s == 0 for s in sizes)
    plt.close(fig)


def test_render_animation_creates_gif(tmp_path):
    """render_animation should produce a GIF file."""
    mod = _load_mod()

    # Create a minimal 5-frame sample
    frames = [
        [[0.3 + i * 0.02 + f * 0.01, 0.5 - i * 0.01, 0.9] for i in range(17)]
        for f in range(5)
    ]

    out_path = tmp_path / "test.gif"
    mod.render_animation(frames, label="fall", output_path=str(out_path), fps=10, figsize=(4, 3), dpi=50)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_render_animation_no_overlay(tmp_path):
    """render_animation with show_overlay=False should still produce output."""
    mod = _load_mod()

    frames = [
        [[0.5, 0.5, 0.9]] * 17
        for _ in range(3)
    ]

    out_path = tmp_path / "no_overlay.gif"
    mod.render_animation(frames, label="test", output_path=str(out_path), fps=10, figsize=(4, 3), dpi=50, show_overlay=False)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


import subprocess


def test_cli_single_file(tmp_path):
    """CLI should render a single JSON file to GIF."""
    # Create a sample JSON
    sample = {
        "frames": [
            {"keypoints": [[0.5 + i * 0.01, 0.5, 0.9] for i in range(17)], "timestamp": t * 0.033}
            for t in range(5)
        ],
        "label": "eating",
        "duration": 0.165,
    }
    in_path = tmp_path / "sample.json"
    in_path.write_text(json.dumps(sample))

    out_path = tmp_path / "out.gif"
    result = subprocess.run(
        [".venv/bin/python", "scripts/visualize_skeleton.py", str(in_path), "-o", str(out_path),
         "--fps", "10", "--figsize", "4", "3", "--dpi", "50"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()


def test_cli_batch_mode(tmp_path):
    """CLI --batch should process all JSON files in a directory."""
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    for name in ["a.json", "b.json"]:
        sample = {
            "frames": [{"keypoints": [[0.5, 0.5, 0.9]] * 17, "timestamp": 0.0}] * 3,
            "label": "fall",
            "duration": 0.1,
        }
        (in_dir / name).write_text(json.dumps(sample))

    result = subprocess.run(
        [".venv/bin/python", "scripts/visualize_skeleton.py", str(in_dir), "-o", str(out_dir),
         "--batch", "--fps", "10", "--figsize", "4", "3", "--dpi", "50"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert (out_dir / "a.gif").exists()
    assert (out_dir / "b.gif").exists()
