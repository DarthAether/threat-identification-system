#!/usr/bin/env python3
"""Train a YOLOv5/YOLOv8 model for threat detection.

This script wraps the Ultralytics training CLI with project-specific defaults
and validation. It produces trained weights in the specified project directory.

Usage:
    python scripts/train_yolov5.py --data data/dataset.yaml --epochs 100
    python scripts/train_yolov5.py --data data/dataset.yaml --epochs 50 --batch-size 32 --device 0
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model for threat detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the dataset YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size in pixels.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov5s.pt",
        help="Pretrained weights file or model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="CUDA device(s) to use, e.g. '0' or '0,1' or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Directory to save training results.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate that required files and directories exist."""
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset config not found: %s", data_path)
        sys.exit(1)

    project_path = Path(args.project)
    project_path.mkdir(parents=True, exist_ok=True)
    logger.info("Training results will be saved to: %s", project_path)


def train(args: argparse.Namespace) -> None:
    """Run the Ultralytics training process."""
    validate_inputs(args)

    cmd = [
        sys.executable,
        "-m",
        "ultralytics",
        "detect",
        "train",
        f"data={args.data}",
        f"epochs={args.epochs}",
        f"batch={args.batch_size}",
        f"imgsz={args.img_size}",
        f"model={args.weights}",
        f"project={args.project}",
        "verbose=True",
    ]

    if args.device:
        cmd.append(f"device={args.device}")

    logger.info("Starting training with command:")
    logger.info("  %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error("Training failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    logger.info("Training completed successfully.")
    logger.info("Results saved to: %s", args.project)


if __name__ == "__main__":
    args = parse_args()
    train(args)
