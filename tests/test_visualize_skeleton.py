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
