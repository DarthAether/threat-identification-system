#!/usr/bin/env python3
"""Export a trained YOLO model to ONNX format.

This script converts a PyTorch (.pt) model to ONNX for optimized inference.
The exported model can be used with ONNX Runtime or TensorRT.

Usage:
    python scripts/export_onnx.py --weights models/best.pt --output models/best.onnx
    python scripts/export_onnx.py --weights models/best.pt --output models/best.onnx --opset 17
"""

from __future__ import annotations

import argparse
import logging
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
        description="Export a YOLO model to ONNX format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained PyTorch model weights (.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the ONNX model.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size for the exported model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate that the weights file exists."""
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error("Weights file not found: %s", weights_path)
        sys.exit(1)

    if not weights_path.suffix == ".pt":
        logger.warning("Weights file does not have .pt extension: %s", weights_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def export(args: argparse.Namespace) -> None:
    """Export the model to ONNX format using Ultralytics."""
    validate_inputs(args)

    logger.info("Loading model from: %s", args.weights)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics package not found. Install it with: pip install ultralytics"
        )
        sys.exit(1)

    model = YOLO(args.weights)

    logger.info("Exporting to ONNX (opset=%d, imgsz=%d)...", args.opset, args.img_size)

    export_path = model.export(
        format="onnx",
        imgsz=args.img_size,
        opset=args.opset,
    )

    # Move to the requested output path if different
    export_path = Path(export_path)
    output_path = Path(args.output)

    if export_path.resolve() != output_path.resolve():
        import shutil

        shutil.move(str(export_path), str(output_path))
        logger.info("Moved exported model to: %s", output_path)

    # Validate the output
    validate_output(output_path, args.img_size)

    logger.info("Export completed successfully: %s", output_path)


def validate_output(output_path: Path, img_size: int) -> None:
    """Validate the exported ONNX model."""
    if not output_path.exists():
        logger.error("Export failed: output file not found at %s", output_path)
        sys.exit(1)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX model size: %.2f MB", file_size_mb)

    try:
        import onnx

        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
        logger.info("ONNX model validation passed.")

        # Log input/output info
        graph = model.graph
        for inp in graph.input:
            dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            logger.info("  Input:  %s  shape=%s", inp.name, dims)
        for out in graph.output:
            dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
            logger.info("  Output: %s  shape=%s", out.name, dims)

    except ImportError:
        logger.warning(
            "onnx package not installed; skipping validation. "
            "Install with: pip install onnx"
        )
    except Exception as e:
        logger.error("ONNX validation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    args = parse_args()
    export(args)
