"""YOLOv5 detection backend backed by ``torch.hub``.

The model is loaded lazily (only when :meth:`load_model` is called)
so that importing this module has zero side effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from threat_id.core.config import DetectionSettings
from threat_id.core.exceptions import InferenceError, ModelLoadError
from threat_id.detection.protocol import RawDetection

logger = structlog.get_logger(__name__)


class YoloBackend:
    """YOLOv5 implementation of the :class:`DetectionBackend` protocol."""

    def __init__(self, settings: DetectionSettings) -> None:
        self._model_path: str = settings.model_path
        self._device: str = settings.device
        self._confidence_threshold: float = settings.confidence_threshold
        self._model: Any | None = None

    # ── Lifecycle ───────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the YOLOv5 model via ``torch.hub``.

        Raises
        ------
        ModelLoadError
            If the model cannot be loaded for any reason (missing weights,
            unsupported device, etc.).
        """
        if self._model is not None:
            logger.debug("yolo.model_already_loaded")
            return

        try:
            import torch  # noqa: PLC0415 — lazy import

            logger.info(
                "yolo.loading_model",
                path=self._model_path,
                device=self._device,
            )

            # Support both hub models (e.g. "yolov5s") and local .pt files.
            if self._model_path.endswith(".pt"):
                self._model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=self._model_path,
                    device=self._device,
                    force_reload=False,
                )
            else:
                self._model = torch.hub.load(
                    "ultralytics/yolov5",
                    self._model_path,
                    device=self._device,
                    force_reload=False,
                )

            self._model.conf = self._confidence_threshold
            logger.info("yolo.model_loaded")
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load YOLOv5 model from '{self._model_path}': {exc}",
            ) from exc

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        """Run YOLOv5 inference on a single BGR frame.

        Parameters
        ----------
        frame:
            ``(H, W, 3)`` uint8 NumPy array.

        Returns
        -------
        list[RawDetection]
            Detections above the confidence threshold.

        Raises
        ------
        InferenceError
            On any runtime / hardware failure during inference.
        """
        if self._model is None:
            raise InferenceError("Model not loaded. Call load_model() first.")

        try:
            # YOLOv5 accepts BGR numpy directly.
            results = self._model(frame)
            predictions = results.xyxy[0].cpu().numpy()  # (N, 6): x1 y1 x2 y2 conf cls
        except Exception as exc:
            raise InferenceError(f"YOLOv5 inference failed: {exc}") from exc

        detections: list[RawDetection] = []
        names: dict[int, str] = self._model.names  # type: ignore[assignment]

        for row in predictions:
            x1, y1, x2, y2, conf, cls_idx = row
            label = names.get(int(cls_idx), f"class_{int(cls_idx)}")
            detections.append(
                RawDetection(
                    label=label,
                    confidence=float(conf),
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                )
            )

        return detections

    def unload_model(self) -> None:
        """Release model and GPU memory."""
        if self._model is None:
            return

        logger.info("yolo.unloading_model")
        del self._model
        self._model = None

        # Attempt to free GPU cache if torch is available.
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
