"""
Non-blocking macOS tactile click & wheel audio feedback.
Synthesizes crisp clean-room subtle WAV clicks (`click_down.wav`, `click_up.wav`,
`wheel_tick.wav`) on first run and plays them via preloaded native AppKit `NSSound`
with live volume & mute controls.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from aerohand.config import repo_root

logger = logging.getLogger(__name__)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples * 32767.0, -32767.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def ensure_default_sound_assets(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Generate clean-room tactile macOS-style click/tick WAVs if not present."""
    sdir = (base_dir or repo_root()) / "assets" / "sounds"
    paths = {
        "down": sdir / "click_down.wav",
        "up": sdir / "click_up.wav",
        "tick": sdir / "wheel_tick.wav",
    }
    sr = 44100
    if not paths["down"].is_file():
        t = np.linspace(0.0, 0.016, int(sr * 0.016), endpoint=False)
        env = np.exp(-t * 280.0) * (1.0 - np.exp(-t * 4000.0))
        sig = 0.55 * env * (0.7 * np.sin(2 * np.pi * 1350.0 * t) + 0.3 * np.sin(2 * np.pi * 2700.0 * t))
        _write_wav(paths["down"], sig, sr)

    if not paths["up"].is_file():
        t = np.linspace(0.0, 0.018, int(sr * 0.018), endpoint=False)
        env = np.exp(-t * 240.0) * (1.0 - np.exp(-t * 4500.0))
        sig = 0.52 * env * (0.65 * np.sin(2 * np.pi * 1950.0 * t) + 0.35 * np.sin(2 * np.pi * 3400.0 * t))
        _write_wav(paths["up"], sig, sr)

    if not paths["tick"].is_file():
        t = np.linspace(0.0, 0.014, int(sr * 0.014), endpoint=False)
        env = np.exp(-t * 310.0) * (1.0 - np.exp(-t * 5000.0))
        sig = 0.42 * env * np.sin(2 * np.pi * 2400.0 * t)
        _write_wav(paths["tick"], sig, sr)

    return paths


class AudioFeedback:
    """Zero-stall macOS auditory feedback for pinch-down, click-up, and Palm Wheel ticks."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        c = cfg or {}
        self.enabled: bool = bool(c.get("enabled", True))
        self.muted: bool = bool(c.get("muted", False))
        self.volume: float = max(0.0, min(1.0, float(c.get("volume", 0.50))))
        self._sounds: Dict[str, Any] = {}

        defaults = ensure_default_sound_assets()
        self._down_name: str = str(c.get("click_down_sound", str(defaults["down"])))
        self._up_name: str = str(c.get("click_up_sound", str(defaults["up"])))
        self._tick_name: str = str(c.get("wheel_tick_sound", str(defaults["tick"])))

        self._preload("down", self._down_name)
        self._preload("up", self._up_name)
        self._preload("tick", self._tick_name)

    def _resolve_path(self, name_or_path: str) -> Path:
        p = Path(name_or_path)
        if p.is_file():
            return p
        rel = repo_root() / name_or_path
        if rel.is_file():
            return rel
        return p

    def _preload(self, key: str, name_or_path: str) -> None:
        try:
            from AppKit import NSSound

            p = self._resolve_path(name_or_path)
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

    def update_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        c = cfg or {}
        self.enabled = bool(c.get("enabled", self.enabled))
        self.muted = bool(c.get("muted", self.muted))
        self.set_volume(float(c.get("volume", self.volume)))

    def set_volume(self, volume: float) -> None:
        self.volume = max(0.0, min(1.0, float(volume)))
        for snd in self._sounds.values():
            try:
                snd.setVolume_(self.volume)
            except Exception:
                pass

    def play_click_down(self) -> None:
        self._play("down")

    def play_click_up(self) -> None:
        self._play("up")

    def play_wheel_tick(self) -> None:
        self._play("tick")

    def _play(self, key: str) -> None:
        if not self.enabled or self.muted or self.volume <= 0.001:
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
