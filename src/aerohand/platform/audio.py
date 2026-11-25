"""
Non-blocking macOS click audio feedback (`click_down` / `click_up`).
Uses native AppKit `NSSound` (preloaded system sounds or optional .wav paths).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class AudioFeedback:
    """Zero-stall macOS auditory feedback for pinch-down and click-up confirmation."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        c = cfg or {}
        self.enabled: bool = bool(c.get("enabled", True))
        self.volume: float = max(0.0, min(1.0, float(c.get("volume", 0.45))))
        self._down_name: str = str(c.get("click_down_sound", "Tink"))
        self._up_name: str = str(c.get("click_up_sound", "Pop"))
        self._sounds: Dict[str, Any] = {}
        if self.enabled:
            self._preload("down", self._down_name)
            self._preload("up", self._up_name)

    def _preload(self, key: str, name_or_path: str) -> None:
        try:
            from AppKit import NSSound

            p = Path(name_or_path)
            if p.is_file():
                snd = NSSound.alloc().initWithContentsOfFile_byReference_(str(p), True)
            else:
                snd = NSSound.soundNamed_(name_or_path)
                if snd is None:
                    sys_aiff = Path(f"/System/Library/Sounds/{name_or_path}.aiff")
                    if sys_aiff.is_file():
                        snd = NSSound.alloc().initWithContentsOfFile_byReference_(str(sys_aiff), True)
            if snd is not None:
                snd.setVolume_(self.volume)
                self._sounds[key] = snd
        except Exception as exc:
            logger.debug(f"[AUDIO] Unable to preload sound '{name_or_path}': {exc}")

    def play_click_down(self) -> None:
        self._play("down")

    def play_click_up(self) -> None:
        self._play("up")

    def _play(self, key: str) -> None:
        if not self.enabled:
            return
        snd = self._sounds.get(key)
        if snd is None:
            return
        try:
            if snd.isPlaying():
                snd.stop()
            snd.play()
        except Exception:
            pass
