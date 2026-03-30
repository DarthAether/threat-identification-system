"""Sound alert channel — plays an audible alarm on the host machine.

Playback is delegated to a platform-native CLI tool (``aplay`` on
Linux, ``afplay`` on macOS, ``start`` on Windows) and executed in a
thread-pool executor so the async loop is never blocked.
"""

from __future__ import annotations

import asyncio
import platform
import subprocess
from pathlib import Path

import structlog

from threat_id.alerting.models import AlertPayload
from threat_id.core.exceptions import SoundPlaybackError

logger = structlog.get_logger(__name__)


class SoundAlertChannel:
    """Plays a local sound file when an alert is triggered."""

    def __init__(self, sound_path: str | Path) -> None:
        self._sound_path = Path(sound_path).resolve()
        if not self._sound_path.is_file():
            raise SoundPlaybackError(
                f"Sound file not found: {self._sound_path}",
                detail="Verify the ALERT_SOUND_PATH environment variable",
            )

        self._system = platform.system()
        self._available = True

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "sound"

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Public API ───────────────────────────────────────────────────

    async def send(self, payload: AlertPayload) -> None:
        """Play the configured alert sound.

        Raises:
            SoundPlaybackError: If the subprocess exits non-zero.
        """
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._play)
        except SoundPlaybackError:
            raise
        except Exception as exc:
            raise SoundPlaybackError(
                f"Sound playback failed: {exc}",
                detail=str(exc),
            ) from exc

        logger.info(
            "sound_alert.played",
            camera_id=payload.camera_id,
            threat=payload.threat_label,
            file=str(self._sound_path),
        )

    # ── Internals ────────────────────────────────────────────────────

    def _play(self) -> None:
        """Blocking subprocess call — runs inside the thread pool."""
        cmd = self._build_command()
        try:
            subprocess.run(  # noqa: S603
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=10,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            raise SoundPlaybackError(
                f"Sound player exited with code {exc.returncode}",
                detail=stderr,
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SoundPlaybackError(
                "Sound playback timed out",
                detail=str(exc),
            ) from exc

    def _build_command(self) -> list[str]:
        file_str = str(self._sound_path)
        if self._system == "Linux":
            return ["aplay", "-q", file_str]
        if self._system == "Darwin":
            return ["afplay", file_str]
        if self._system == "Windows":
            # `start` is a cmd.exe built-in; /wait blocks until playback ends.
            return ["cmd", "/c", "start", "/wait", "", file_str]

        raise SoundPlaybackError(
            f"Unsupported platform: {self._system}",
            detail="Only Linux, macOS, and Windows are supported",
        )
