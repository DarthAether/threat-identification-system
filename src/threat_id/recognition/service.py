"""High-level face recognition service.

Orchestrates a :class:`RecognitionBackend` and :class:`FaceStore` to
produce :class:`FaceMatch` results from raw camera frames.
"""

from __future__ import annotations

import time

import numpy as np
import structlog

from threat_id.core.config import RecognitionSettings
from threat_id.detection.models import BoundingBox
from threat_id.recognition.face_store import FaceStore
from threat_id.recognition.models import FaceMatch, Identity
from threat_id.recognition.protocol import RecognitionBackend

logger = structlog.get_logger(__name__)


class RecognitionService:
    """Stateless service that wraps a backend and a face store.

    Parameters
    ----------
    settings:
        Recognition configuration (model name, threshold, etc.)
    backend:
        Any object satisfying the :class:`RecognitionBackend` protocol.
    face_store:
        An initialised :class:`FaceStore` with loaded embeddings.
    """

    def __init__(
        self,
        settings: RecognitionSettings,
        backend: RecognitionBackend,
        face_store: FaceStore,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._face_store = face_store

    def recognize(self, frame: np.ndarray) -> list[FaceMatch]:
        """Run face recognition on a single BGR frame.

        Parameters
        ----------
        frame:
            ``(H, W, 3)`` uint8 BGR image.

        Returns
        -------
        list[FaceMatch]
            Zero or more matched identities with bounding boxes and
            confidence scores.
        """
        known = self._face_store.get_all()
        if not known:
            logger.debug("recognition.no_known_faces")
            return []

        start = time.perf_counter()
        raw_matches = self._backend.find_matches(frame, known)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.debug(
            "recognition.inference_complete",
            raw_matches=len(raw_matches),
            elapsed_ms=round(elapsed_ms, 2),
        )

        results: list[FaceMatch] = []
        for m in raw_matches:
            if m.confidence < self._settings.similarity_threshold:
                continue

            results.append(
                FaceMatch(
                    identity=Identity(id=m.identity, name=m.identity),
                    confidence=round(m.confidence, 4),
                    bbox=BoundingBox(x1=m.x1, y1=m.y1, x2=m.x2, y2=m.y2),
                )
            )

        return results
