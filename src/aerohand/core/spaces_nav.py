"""
Spaces Navigation Slide Controller (`thumb_pinky_base` + slide).
Ported from Airpoint SDK `SpacesNavController.ts` (Apache-2.0) with cooldown guard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from aerohand.core.types import SlideDirection


@dataclass
class SpacesNavOutput:
    direction: SlideDirection = None
    is_active: bool = False
    progress_x: float = 0.0
    progress_y: float = 0.0
    triggered: bool = False


class SpacesNavController:
    """
    Replaces tiring 4-finger desktop swipes with `thumb_pinky_base` hold + short slide:
      - Horizontal (left/right): switch macOS Desktop Spaces
      - Vertical (top/bottom): Mission Control / App Exposé
    Fires immediately (edge-triggered once per hold) when threshold is reached.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        c = cfg or {}
        self.enabled: bool = bool(c.get("enabled", True))
        self.lock_threshold_px: float = float(c.get("lock_threshold_px", 8.0))
        self.sensitivity_px: float = float(c.get("sensitivity_px", 20.0))
        self.invert: bool = bool(c.get("invert", True))
        self.cooldown_ms: float = float(c.get("cooldown_ms", 450.0))

        self._acc_x: float = 0.0
        self._acc_y: float = 0.0
        self._locked_dir: SlideDirection = None
        self._is_active: bool = False
        self._already_fired: bool = False
        self._last_fire_ms: float = -1e9

    def reset(self) -> None:
        self._acc_x = 0.0
        self._acc_y = 0.0
        self._locked_dir = None
        self._is_active = False
        self._already_fired = False

    def process(
        self,
        delta_x: float,
        delta_y: float,
        is_gesture_active: bool,
        timestamp_ms: float = 0.0,
    ) -> SpacesNavOutput:
        if not self.enabled:
            self.reset()
            return SpacesNavOutput()

        if is_gesture_active and not self._is_active:
            self._is_active = True
            self._acc_x = 0.0
            self._acc_y = 0.0
            self._locked_dir = None
            self._already_fired = False

        if not is_gesture_active:
            if self._is_active:
                self.reset()
            return SpacesNavOutput()

        self._acc_x += delta_x
        self._acc_y += delta_y
        abs_x = abs(self._acc_x)
        abs_y = abs(self._acc_y)

        if self._locked_dir is None:
            if abs_x > self.lock_threshold_px or abs_y > self.lock_threshold_px:
                if abs_x > abs_y:
                    if self.invert:
                        self._locked_dir = "left" if self._acc_x > 0 else "right"
                    else:
                        self._locked_dir = "right" if self._acc_x > 0 else "left"
                else:
                    self._locked_dir = "bottom" if self._acc_y > 0 else "top"

        just_triggered = False
        if self._locked_dir is not None and not self._already_fired:
            relevant = abs_x if self._locked_dir in ("left", "right") else abs_y
            in_cooldown = (timestamp_ms - self._last_fire_ms) < self.cooldown_ms
            if relevant >= self.sensitivity_px and not in_cooldown:
                self._already_fired = True
                self._last_fire_ms = timestamp_ms
                just_triggered = True

        return SpacesNavOutput(
            direction=self._locked_dir,
            is_active=True,
            progress_x=self._acc_x,
            progress_y=self._acc_y,
            triggered=just_triggered,
        )
