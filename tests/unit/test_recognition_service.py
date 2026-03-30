"""Tests for threat_id.recognition.service — RecognitionService logic."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import FakeRecognitionBackend
from tests.factories import make_frame
from threat_id.core.config import RecognitionSettings
from threat_id.recognition.face_store import FaceStore
from threat_id.recognition.models import FaceMatch
from threat_id.recognition.protocol import RawFaceMatch
from threat_id.recognition.service import RecognitionService


def _make_face_store_with_entries() -> FaceStore:
    """Create a FaceStore with one known embedding."""
    store = FaceStore(faces_dir=None)
    store.add_face("john_doe", np.random.default_rng(1).standard_normal(128).astype(np.float32))
    return store


def _make_empty_face_store() -> FaceStore:
    """Create an empty FaceStore."""
    return FaceStore(faces_dir=None)


class TestRecognizeReturnsFaceMatchList:
    """recognize() returns a list of FaceMatch objects."""

    def test_returns_face_match_list(self) -> None:
        backend = FakeRecognitionBackend(
            matches=[
                RawFaceMatch(identity="john_doe", confidence=0.95, x1=30, y1=40, x2=130, y2=180),
            ]
        )
        settings = RecognitionSettings(similarity_threshold=0.5)
        store = _make_face_store_with_entries()
        service = RecognitionService(settings, backend, store)

        results = service.recognize(make_frame())

        assert len(results) == 1
        assert isinstance(results[0], FaceMatch)
        assert results[0].identity.name == "john_doe"
        assert results[0].confidence == 0.95

    def test_multiple_matches(self) -> None:
        backend = FakeRecognitionBackend(
            matches=[
                RawFaceMatch(identity="alice", confidence=0.90, x1=10, y1=10, x2=80, y2=80),
                RawFaceMatch(identity="bob", confidence=0.85, x1=100, y1=100, x2=200, y2=200),
            ]
        )
        settings = RecognitionSettings(similarity_threshold=0.5)
        store = _make_face_store_with_entries()
        service = RecognitionService(settings, backend, store)

        results = service.recognize(make_frame())
        assert len(results) == 2
        names = {r.identity.name for r in results}
        assert names == {"alice", "bob"}


class TestRecognizeWithNoFaces:
    """recognize() with no faces in the store returns empty list."""

    def test_empty_store_returns_empty(self) -> None:
        backend = FakeRecognitionBackend(matches=[])
        settings = RecognitionSettings(similarity_threshold=0.5)
        store = _make_empty_face_store()
        service = RecognitionService(settings, backend, store)

        results = service.recognize(make_frame())
        assert results == []


class TestRecognizeFiltersBySimilarityThreshold:
    """recognize() filters out matches below the similarity threshold."""

    def test_below_threshold_filtered(self) -> None:
        backend = FakeRecognitionBackend(
            matches=[
                RawFaceMatch(identity="john_doe", confidence=0.3, x1=30, y1=40, x2=130, y2=180),
                RawFaceMatch(identity="alice", confidence=0.8, x1=10, y1=10, x2=80, y2=80),
            ]
        )
        settings = RecognitionSettings(similarity_threshold=0.6)
        store = _make_face_store_with_entries()
        service = RecognitionService(settings, backend, store)

        results = service.recognize(make_frame())
        assert len(results) == 1
        assert results[0].identity.name == "alice"

    def test_all_below_threshold_returns_empty(self) -> None:
        backend = FakeRecognitionBackend(
            matches=[
                RawFaceMatch(identity="john_doe", confidence=0.2, x1=30, y1=40, x2=130, y2=180),
            ]
        )
        settings = RecognitionSettings(similarity_threshold=0.5)
        store = _make_face_store_with_entries()
        service = RecognitionService(settings, backend, store)

        results = service.recognize(make_frame())
        assert results == []

    def test_exact_threshold_included(self) -> None:
        backend = FakeRecognitionBackend(
            matches=[
                RawFaceMatch(identity="john_doe", confidence=0.6, x1=30, y1=40, x2=130, y2=180),
            ]
        )
        settings = RecognitionSettings(similarity_threshold=0.6)
        store = _make_face_store_with_entries()
        service = RecognitionService(settings, backend, store)

        results = service.recognize(make_frame())
        assert len(results) == 1
