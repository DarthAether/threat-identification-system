"""DeepFace recognition backend.

Wraps the `deepface <https://github.com/serengil/deepface>`_ library to
implement the :class:`RecognitionBackend` protocol.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from threat_id.core.config import RecognitionSettings
from threat_id.core.exceptions import RecognitionError
from threat_id.recognition.protocol import RawFaceMatch

logger = structlog.get_logger(__name__)


class DeepFaceBackend:
    """DeepFace implementation of :class:`RecognitionBackend`."""

    def __init__(self, settings: RecognitionSettings) -> None:
        self._model_name: str = settings.model_name
        self._distance_metric: str = settings.distance_metric
        self._similarity_threshold: float = settings.similarity_threshold

    # ── Protocol Methods ────────────────────────────────────────────────

    def compute_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Compute a 1-D embedding for a cropped face image.

        Parameters
        ----------
        face_image:
            ``(H, W, 3)`` uint8 BGR array.

        Returns
        -------
        np.ndarray
            Float32 embedding vector.

        Raises
        ------
        RecognitionError
            If DeepFace fails to produce an embedding.
        """
        try:
            from deepface import DeepFace  # noqa: PLC0415

            representations = DeepFace.represent(
                img_path=face_image,
                model_name=self._model_name,
                enforce_detection=False,
            )

            if not representations:
                raise RecognitionError("DeepFace returned no embeddings for the input image.")

            embedding = np.asarray(representations[0]["embedding"], dtype=np.float32)
            return embedding

        except RecognitionError:
            raise
        except Exception as exc:
            raise RecognitionError(f"DeepFace embedding failed: {exc}") from exc

    def find_matches(
        self,
        frame: np.ndarray,
        known_embeddings: dict[str, np.ndarray],
    ) -> list[RawFaceMatch]:
        """Detect faces and match against known embeddings.

        Parameters
        ----------
        frame:
            Full-resolution ``(H, W, 3)`` BGR image.
        known_embeddings:
            ``{name: embedding}`` mapping.

        Returns
        -------
        list[RawFaceMatch]
        """
        try:
            from deepface import DeepFace  # noqa: PLC0415

            # Step 1: extract face regions + embeddings from the frame.
            extracted = DeepFace.represent(
                img_path=frame,
                model_name=self._model_name,
                enforce_detection=False,
            )

            if not extracted:
                return []

            matches: list[RawFaceMatch] = []

            for face_data in extracted:
                query_emb = np.asarray(face_data["embedding"], dtype=np.float32)
                facial_area: dict[str, int] = face_data.get("facial_area", {})

                x1 = int(facial_area.get("x", 0))
                y1 = int(facial_area.get("y", 0))
                w = int(facial_area.get("w", 0))
                h = int(facial_area.get("h", 0))
                x2 = x1 + w
                y2 = y1 + h

                best_name, best_score = self._find_closest(query_emb, known_embeddings)
                if best_name is not None and best_score >= self._similarity_threshold:
                    matches.append(
                        RawFaceMatch(
                            identity=best_name,
                            confidence=best_score,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )

            return matches

        except RecognitionError:
            raise
        except Exception as exc:
            raise RecognitionError(f"DeepFace recognition failed: {exc}") from exc

    # ── Internal Helpers ────────────────────────────────────────────────

    def _find_closest(
        self,
        query: np.ndarray,
        known: dict[str, np.ndarray],
    ) -> tuple[str | None, float]:
        """Return the identity with the highest cosine similarity.

        Returns
        -------
        tuple[str | None, float]
            ``(name, similarity)`` or ``(None, 0.0)`` when *known* is empty.
        """
        if not known:
            return None, 0.0

        best_name: str | None = None
        best_sim: float = -1.0

        q_norm = query / (np.linalg.norm(query) + 1e-10)

        for name, emb in known.items():
            e_norm = emb / (np.linalg.norm(emb) + 1e-10)
            similarity = float(np.dot(q_norm, e_norm))
            if similarity > best_sim:
                best_sim = similarity
                best_name = name

        return best_name, max(best_sim, 0.0)
