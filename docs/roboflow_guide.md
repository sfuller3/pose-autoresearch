# Roboflow Environment Detection Guide

## Overview

This project uses Roboflow to train an object detection model that identifies
furniture and fixtures in elder care facilities. The detected objects provide
**environment context** to the fall-detection classifier -- knowing that a person
is near a bed vs. in an open hallway changes how we interpret their pose.

The trained model runs locally via the Roboflow Inference SDK and feeds
environment features into the streaming detection pipeline (`stream_detect.py`).

## Prerequisites

1. Create a free account at [roboflow.com](https://roboflow.com)
2. Get your API key from **Settings > API Key** in the Roboflow dashboard
3. Install dependencies:
   ```bash
   pip install roboflow inference
   ```

## Target Classes

Annotate the following objects in facility images:

| Class | Annotation Notes |
|-------|-----------------|
| `bed` | Full bed frame including mattress; draw tight bounding box |
| `chair` | Any seating: armchair, dining chair, recliner |
| `table` | Dining tables, bedside tables, desks |
| `door` | Door frame area including open or closed door |
| `wheelchair` | Include wheels and footrests in bounding box |
| `walker` | Walkers and rollators |
| `handrail` | Wall-mounted grab bars and corridor handrails |
| `floor-area` | Open floor regions where falls are most dangerous |

Aim for at least 200 annotated images before training. More images with varied
lighting and camera angles will improve accuracy.

## Workflow

### 1. Create the Roboflow Project

```bash
python scripts/setup_roboflow.py init --api-key YOUR_KEY
```

This creates an `elder-care-environment` project in your Roboflow workspace.

### 2. Upload Images

Upload facility images or extracted video frames through the Roboflow web UI.
To extract frames from surveillance video:

```bash
ffmpeg -i video.mp4 -vf "fps=1" frames/frame_%04d.jpg
```

Then drag-and-drop the frames into your Roboflow project.

### 3. Annotate

Use the Roboflow annotation tool to label objects in each image. Use the class
names listed above exactly as shown (lowercase, hyphenated).

### 4. Train

Generate a dataset version in Roboflow (apply augmentations as needed), then
start training with Roboflow Train. A YOLOv8 model typically trains in under
an hour on Roboflow's servers.

### 5. Download the Trained Model

```bash
python scripts/setup_roboflow.py download --api-key YOUR_KEY --version 1
```

Weights are saved to `models/roboflow/` in YOLOv8 format.

## Pipeline Integration

The `EnvironmentDetector` class (defined in the streaming pipeline) loads the
downloaded model and runs inference on each video frame:

```
Video Frame
    |
    v
EnvironmentDetector  -->  detected objects + bounding boxes
    |
    v
Feature vector: [bed_nearby, chair_nearby, floor_open, ...]
    |
    v
Fall classifier (pose + environment features)
```

Environment features are concatenated with pose features before classification.
This allows the model to learn context-dependent patterns -- for example,
a person lowering themselves near a bed is likely sitting down, not falling.

## Troubleshooting

- **Import error for `roboflow`**: Run `pip install roboflow`
- **API key invalid**: Check your key at roboflow.com Settings > API Key
- **Model download fails**: Ensure you have trained at least one version
- **Low detection accuracy**: Add more annotated images with diverse conditions
