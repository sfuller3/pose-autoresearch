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
