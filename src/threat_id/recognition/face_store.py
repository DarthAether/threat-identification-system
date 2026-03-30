"""In-memory face embedding store backed by filesystem or database.

:class:`FaceStore` holds pre-computed face embeddings in memory for fast
nearest-neighbour lookups during live recognition.  Embeddings can be
loaded from a directory of ``.npy`` files **or** from the database via
``FaceRecord`` rows.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

from threat_id.core.exceptions import FaceNotFoundError, FaceStoreError

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from threat_id.db.models import FaceRecord

logger = structlog.get_logger(__name__)


class FaceStore:
    """Thread-safe in-memory cache of named face embeddings.

    Parameters
    ----------
    faces_dir:
        Directory containing ``<name>.npy`` embedding files.
        If ``None``, embeddings must be loaded from the database.
    """

    def __init__(self, faces_dir: Path | None = None) -> None:
        self._faces_dir = faces_dir
        self._lock = threading.Lock()
        self._embeddings: dict[str, np.ndarray] = {}

    # ── Bulk Loading ────────────────────────────────────────────────────

    def load(self) -> int:
        """Load all ``.npy`` files from the configured directory.

        Returns the number of embeddings loaded.

        Raises
        ------
        FaceStoreError
            If the directory does not exist or a file is unreadable.
        """
        if self._faces_dir is None:
            logger.warning("face_store.no_directory_configured")
            return 0

        directory = self._faces_dir
        if not directory.is_dir():
            raise FaceStoreError(f"Faces directory does not exist: {directory}")

        count = 0
        with self._lock:
            for npy_file in sorted(directory.glob("*.npy")):
                try:
                    embedding = np.load(npy_file).astype(np.float32)
                    name = npy_file.stem
                    self._embeddings[name] = embedding
                    count += 1
                except Exception as exc:
                    logger.error(
                        "face_store.load_file_error",
                        file=str(npy_file),
                        error=str(exc),
                    )

        logger.info("face_store.loaded", count=count, directory=str(directory))
        return count

    async def load_from_db(self, session: AsyncSession) -> int:
        """Load active face embeddings from the database.

        Each ``FaceRecord.embedding_path`` is expected to point to a
        ``.npy`` file.

        Returns the number of embeddings loaded.
        """
        from sqlalchemy import select  # noqa: PLC0415

        from threat_id.db.models import FaceRecord  # noqa: PLC0415

        stmt = select(FaceRecord).where(FaceRecord.is_active.is_(True))
        result = await session.execute(stmt)
        records: list[FaceRecord] = list(result.scalars().all())

        count = 0
        with self._lock:
            for record in records:
                try:
                    path = Path(record.embedding_path)
                    if not path.is_file():
                        logger.warning(
                            "face_store.missing_embedding_file",
                            name=record.name,
                            path=str(path),
                        )
                        continue
                    embedding = np.load(path).astype(np.float32)
                    self._embeddings[record.name] = embedding
                    count += 1
                except Exception as exc:
                    logger.error(
                        "face_store.db_load_error",
                        name=record.name,
                        error=str(exc),
                    )

        logger.info("face_store.loaded_from_db", count=count, total_records=len(records))
        return count

    # ── Single-entry Mutations ──────────────────────────────────────────

    def add_face(self, name: str, embedding: np.ndarray) -> None:
        """Add or replace an embedding in the in-memory cache.

        Parameters
        ----------
        name:
            Unique identifier for this face.
        embedding:
            1-D float32 vector.
        """
        with self._lock:
            self._embeddings[name] = embedding.astype(np.float32)
        logger.info("face_store.face_added", name=name)

    def remove_face(self, name: str) -> None:
        """Remove an embedding.

        Raises
        ------
        FaceNotFoundError
            If *name* is not in the store.
        """
        with self._lock:
            if name not in self._embeddings:
                raise FaceNotFoundError(f"Face '{name}' not found in store.")
            del self._embeddings[name]
        logger.info("face_store.face_removed", name=name)

    # ── Queries ─────────────────────────────────────────────────────────

    def get_all(self) -> dict[str, np.ndarray]:
        """Return a shallow copy of the full embeddings dict.

        The copy prevents callers from mutating internal state.
        """
        with self._lock:
            return dict(self._embeddings)

    def find_match(
        self,
        embedding: np.ndarray,
        *,
        threshold: float = 0.6,
    ) -> tuple[str | None, float]:
        """Find the best-matching identity via cosine similarity.

        Parameters
        ----------
        embedding:
            Query embedding.
        threshold:
            Minimum similarity to consider a match.

        Returns
        -------
        tuple[str | None, float]
            ``(name, similarity)`` if a match is found, else ``(None, 0.0)``.
        """
        with self._lock:
            if not self._embeddings:
                return None, 0.0

            q_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
            best_name: str | None = None
            best_sim: float = -1.0

            for name, emb in self._embeddings.items():
                e_norm = emb / (np.linalg.norm(emb) + 1e-10)
                sim = float(np.dot(q_norm, e_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

            if best_sim >= threshold:
                return best_name, best_sim
            return None, 0.0

    @property
    def count(self) -> int:
        """Number of embeddings currently cached."""
        with self._lock:
            return len(self._embeddings)
