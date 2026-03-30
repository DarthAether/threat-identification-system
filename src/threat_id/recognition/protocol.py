"""Recognition backend protocol and shared data structures.

Any object satisfying :class:`RecognitionBackend` can be plugged into
:class:`~threat_id.recognition.service.RecognitionService`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class RawFaceMatch:
    """A single face match returned by a recognition backend.

    Attributes
    ----------
    identity:
        The name or identifier of the matched person.
    confidence:
        Similarity score (0..1, higher = better).
    x1, y1, x2, y2:
        Pixel-space bounding box of the detected face.
    """

    identity: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@runtime_checkable
class RecognitionBackend(Protocol):
    """Structural interface for face recognition backends."""

    def compute_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Compute a fixed-length embedding vector for a face crop.

        Parameters
        ----------
        face_image:
            ``(H, W, 3)`` uint8 BGR image tightly cropped around a face.

        Returns
        -------
        np.ndarray
            1-D float32 embedding vector.
        """
        ...

    def find_matches(
        self,
        frame: np.ndarray,
        known_embeddings: dict[str, np.ndarray],
    ) -> list[RawFaceMatch]:
        """Detect faces in *frame* and match against known embeddings.

        Parameters
        ----------
        frame:
            Full-resolution ``(H, W, 3)`` uint8 BGR image.
        known_embeddings:
            Mapping of ``{name: embedding_vector}``.

        Returns
        -------
        list[RawFaceMatch]
        """
        ...
