"""
Window Tiling Slide Controller (`thumb_pinky_pinch` + slide).
Ported from Airpoint SDK `WindowTileController.ts` (Apache-2.0):
  - Locks axis after `lock_threshold_px` (8 px)
  - Arms once slide exceeds `sensitivity_px` (20 px)
  - Dispatches final direction cleanly ON RELEASE so user can preview/cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from aerohand.core.types import SlideDirection


@dataclass
class WindowTileOutput:
    direction: SlideDirection = None
    is_active: bool = False
    progress_x: float = 0.0
    progress_y: float = 0.0
    triggered: bool = False


class WindowTileController:
    """Handles thumb-pinky pinch + slide for macOS window tiling."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        c = cfg or {}
        self.enabled: bool = bool(c.get("enabled", True))
        self.lock_threshold_px: float = float(c.get("lock_threshold_px", 8.0))
        self.sensitivity_px: float = float(c.get("sensitivity_px", 20.0))
        self.cooldown_ms: float = float(c.get("cooldown_ms", 450.0))

        self._acc_x: float = 0.0
        self._acc_y: float = 0.0
        self._locked_dir: SlideDirection = None
        self._is_active: bool = False
        self._armed: bool = False
        self._last_fire_ms: float = -1e9

    def reset(self) -> None:
        self._acc_x = 0.0
        self._acc_y = 0.0
        self._locked_dir = None
        self._is_active = False
        self._armed = False

    def process(
        self,
        delta_x: float,
        delta_y: float,
        is_gesture_active: bool,
        timestamp_ms: float = 0.0,
    ) -> WindowTileOutput:
        if not self.enabled:
            self.reset()
            return WindowTileOutput()

        if is_gesture_active and not self._is_active:
            self._is_active = True
            self._acc_x = 0.0
            self._acc_y = 0.0
            self._locked_dir = None
            self._armed = False

        # Release edge -> dispatch if armed
        if not is_gesture_active and self._is_active:
            direction = self._locked_dir
            armed = self._armed
            self.reset()
            in_cooldown = (timestamp_ms - self._last_fire_ms) < self.cooldown_ms
            if direction is not None and armed and not in_cooldown:
                self._last_fire_ms = timestamp_ms
                return WindowTileOutput(direction=direction, is_active=False, triggered=True)
            return WindowTileOutput()

        if not is_gesture_active:
            return WindowTileOutput()

        self._acc_x += delta_x
        self._acc_y += delta_y
        abs_x = abs(self._acc_x)
        abs_y = abs(self._acc_y)

        if self._locked_dir is None:
            if abs_x > self.lock_threshold_px or abs_y > self.lock_threshold_px:
                if abs_x > abs_y:
                    self._locked_dir = "right" if self._acc_x > 0 else "left"
                else:
                    self._locked_dir = "bottom" if self._acc_y > 0 else "top"

        if self._locked_dir is not None and not self._armed:
            relevant = abs_x if self._locked_dir in ("left", "right") else abs_y
            if relevant >= self.sensitivity_px:
                self._armed = True

        return WindowTileOutput(
            direction=self._locked_dir,
            is_active=True,
            progress_x=self._acc_x,
            progress_y=self._acc_y,
            triggered=False,
        )
