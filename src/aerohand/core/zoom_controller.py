"""
Two-Hand Pinch-to-Zoom Controller (Airpoint Parity).

When both Left and Right hands are visible and both actively pinch, tracks changes
in inter-hand distance (normalized by average palm size) and emits smooth discrete
zoom steps (`+1` zoom in / `-1` zoom out) with deadzone & rate-limiting.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from aerohand.core.types import ZoomOutput


class ZoomController:
    """Two-hand spread/pinch zoom gesture detector."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.update_config(cfg)
        self._active: bool = False
        self._anchor_dist: Optional[float] = None
        self._ema_dist: Optional[float] = None
        self._accum: float = 0.0
        self._last_step_ms: float = -1e9

    def update_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        c = cfg or {}
        self.enabled: bool = bool(c.get("enable_zoom", True))
        self.zoom_speed: float = max(0.2, float(c.get("zoom_speed", 1.0)))
        self.step_threshold: float = 0.045 / self.zoom_speed
        self.deadzone: float = 0.004
        self.cooldown_ms: float = 95.0

    def reset(self) -> None:
        self._active = False
        self._anchor_dist = None
        self._ema_dist = None
        self._accum = 0.0

    def process(
        self,
        left_pt: Optional[Tuple[float, float]],
        right_pt: Optional[Tuple[float, float]],
        both_pinching: bool,
        timestamp_ms: float,
    ) -> ZoomOutput:
        if not self.enabled or not both_pinching or left_pt is None or right_pt is None:
            self.reset()
            return ZoomOutput()

        raw_dist = math.hypot(right_pt[0] - left_pt[0], right_pt[1] - left_pt[1])
        if not self._active or self._ema_dist is None or self._anchor_dist is None:
            self._active = True
            self._anchor_dist = raw_dist
            self._ema_dist = raw_dist
            self._accum = 0.0
            return ZoomOutput(is_zooming=True, delta=0.0, step=0, distance=raw_dist)

        prev = self._ema_dist
        self._ema_dist = 0.65 * prev + 0.35 * raw_dist
        delta = self._ema_dist - prev
        if abs(delta) < self.deadzone:
            delta = 0.0

        self._accum += delta
        step = 0
        if (timestamp_ms - self._last_step_ms) >= self.cooldown_ms:
            if self._accum >= self.step_threshold:
                step = 1
                self._accum -= self.step_threshold
                self._last_step_ms = timestamp_ms
            elif self._accum <= -self.step_threshold:
                step = -1
                self._accum += self.step_threshold
                self._last_step_ms = timestamp_ms

        return ZoomOutput(is_zooming=True, delta=delta, step=step, distance=self._ema_dist)
