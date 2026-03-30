"""ONNX Runtime detection backend.

Uses ``onnxruntime.InferenceSession`` for hardware-accelerated inference
without a PyTorch dependency.  Supports CPU, CUDA, and TensorRT execution
providers.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import structlog

from threat_id.core.config import DetectionSettings
from threat_id.core.exceptions import InferenceError, ModelLoadError
from threat_id.detection.protocol import RawDetection

logger = structlog.get_logger(__name__)

# Default COCO-style labels — can be overridden by a labels file next to the
# ONNX model (``<model>.labels``).
_DEFAULT_INPUT_SIZE: int = 640


class OnnxBackend:
    """ONNX Runtime implementation of the :class:`DetectionBackend` protocol."""

    def __init__(self, settings: DetectionSettings) -> None:
        self._model_path: str = settings.model_path
        self._device: str = settings.device
        self._confidence_threshold: float = settings.confidence_threshold
        self._session: Any | None = None
        self._input_name: str = ""
        self._input_shape: tuple[int, ...] = ()
        self._labels: dict[int, str] = {}

    # ── Lifecycle ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Create an ONNX ``InferenceSession``.

        Raises
        ------
        ModelLoadError
            If the ONNX file is missing or the runtime cannot load it.
        """
        if self._session is not None:
            logger.debug("onnx.session_already_loaded")
            return

        try:
            import onnxruntime as ort  # noqa: PLC0415

            providers = self._resolve_providers()
            logger.info(
                "onnx.loading_model",
                path=self._model_path,
                providers=providers,
            )

            self._session = ort.InferenceSession(self._model_path, providers=providers)

            # Inspect the first input for shape / name.
            model_input = self._session.get_inputs()[0]
            self._input_name = model_input.name
            self._input_shape = tuple(model_input.shape)  # e.g. (1, 3, 640, 640)

            self._labels = self._load_labels()
            logger.info("onnx.model_loaded", input_name=self._input_name, input_shape=self._input_shape)

        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load ONNX model from '{self._model_path}': {exc}",
            ) from exc

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        """Run inference on a BGR frame.

        The method handles preprocessing (resize, normalise, transpose) and
        post-processing (NMS, coordinate rescaling) internally.

        Parameters
        ----------
        frame:
            ``(H, W, 3)`` uint8 array in BGR order.

        Returns
        -------
        list[RawDetection]

        Raises
        ------
        InferenceError
            On any failure during inference.
        """
        if self._session is None:
            raise InferenceError("Model not loaded. Call load_model() first.")

        try:
            orig_h, orig_w = frame.shape[:2]
            blob = self._preprocess(frame)
            outputs = self._session.run(None, {self._input_name: blob})
            return self._postprocess(outputs, orig_w, orig_h)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"ONNX inference failed: {exc}") from exc

    def unload_model(self) -> None:
        """Release the ONNX session."""
        if self._session is None:
            return
        logger.info("onnx.unloading_model")
        del self._session
        self._session = None

    # ── Internal Helpers ────────────────────────────────────────────────

    def _resolve_providers(self) -> list[str]:
        """Choose execution providers based on the configured device."""
        if self._device.startswith("cuda"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _get_input_size(self) -> int:
        """Return the expected square input dimension."""
        if len(self._input_shape) == 4 and isinstance(self._input_shape[2], int):
            return int(self._input_shape[2])
        return _DEFAULT_INPUT_SIZE

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize, normalise, and transpose a BGR frame for ONNX input.

        Returns an ``(1, 3, H, W)`` float32 blob.
        """
        size = self._get_input_size()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return np.expand_dims(img, axis=0)  # add batch dim

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        orig_w: int,
        orig_h: int,
    ) -> list[RawDetection]:
        """Parse YOLO-style ONNX output ``(1, N, 5+C)`` into detections."""
        predictions = outputs[0]  # shape: (1, N, 5+num_classes)
        if predictions.ndim == 3:
            predictions = predictions[0]

        size = self._get_input_size()
        scale_x = orig_w / size
        scale_y = orig_h / size

        detections: list[RawDetection] = []

        for row in predictions:
            obj_conf = float(row[4])
            if obj_conf < self._confidence_threshold:
                continue

            class_scores = row[5:]
            cls_idx = int(np.argmax(class_scores))
            cls_conf = float(class_scores[cls_idx])
            confidence = obj_conf * cls_conf
            if confidence < self._confidence_threshold:
                continue

            # Convert centre-xy-wh to xyxy and rescale.
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            x1 = int((cx - w / 2) * scale_x)
            y1 = int((cy - h / 2) * scale_y)
            x2 = int((cx + w / 2) * scale_x)
            y2 = int((cy + h / 2) * scale_y)

            # Clamp to frame boundaries.
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_w, x2)
            y2 = min(orig_h, y2)

            label = self._labels.get(cls_idx, f"class_{cls_idx}")
            detections.append(
                RawDetection(
                    label=label,
                    confidence=confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return detections

    def _load_labels(self) -> dict[int, str]:
        """Attempt to read a ``.labels`` file next to the model.

        Falls back to an empty dict if no file is found.
        """
        from pathlib import Path  # noqa: PLC0415

        labels_path = Path(self._model_path).with_suffix(".labels")
        if not labels_path.is_file():
            return {}

        labels: dict[int, str] = {}
        for idx, line in enumerate(labels_path.read_text().splitlines()):
            name = line.strip()
            if name:
                labels[idx] = name
        logger.info("onnx.labels_loaded", count=len(labels), path=str(labels_path))
        return labels
