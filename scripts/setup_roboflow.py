# scripts/setup_roboflow.py
"""
Initialize Roboflow project for environment object detection.

Usage:
  # Create project and upload initial dataset
  python scripts/setup_roboflow.py init --api-key YOUR_KEY

  # Download trained model for local inference
  python scripts/setup_roboflow.py download --api-key YOUR_KEY --version 1
"""
import argparse
from pathlib import Path


def init_project(api_key: str):
    """Create Roboflow project with target classes."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace()

    project = workspace.create_project(
        project_name="elder-care-environment",
        project_type="object-detection",
        annotation="elder care facility furniture and fixtures",
    )

    print(f"Project created: {project.id}")
    print(f"URL: https://app.roboflow.com/{workspace.name}/{project.id}")
    print()
    print("Target classes to annotate:")
    for cls in ["bed", "chair", "table", "door", "wheelchair",
                "walker", "handrail", "floor-area"]:
        print(f"  - {cls}")
    print()
    print("Next steps:")
    print("  1. Upload facility images/video frames to Roboflow")
    print("  2. Annotate objects using Roboflow web UI")
    print("  3. Train model (Roboflow Train or export to YOLO format)")
    print("  4. Download: python scripts/setup_roboflow.py download --version N")


def download_model(api_key: str, project_id: str, version: int):
    """Download trained model weights for local inference."""
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_id)
    model = project.version(version)

    # Export as YOLO format for inference SDK compatibility
    export_path = Path("models/roboflow")
    export_path.mkdir(parents=True, exist_ok=True)
    model.download("yolov8", location=str(export_path))
    print(f"Model downloaded to {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    init_p = sub.add_parser("init")
    init_p.add_argument("--api-key", required=True)

    dl_p = sub.add_parser("download")
    dl_p.add_argument("--api-key", required=True)
    dl_p.add_argument("--project", default="elder-care-environment")
    dl_p.add_argument("--version", type=int, required=True)

    args = parser.parse_args()
    if args.command == "init":
        init_project(args.api_key)
    elif args.command == "download":
        download_model(args.api_key, args.project, args.version)
