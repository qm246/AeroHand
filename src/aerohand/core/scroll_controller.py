"""
2D Axis-Locked Scroll Controller with Rate-Independent Inertia.
Ported from Airpoint SDK `ScrollController.ts` (Apache-2.0), upgraded with dt-normalized
inertia decay so coasting feels identical across 30 fps and 60 fps webcams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

ScrollAxis = Optional[Literal["x", "y", "free"]]


@dataclass
class ScrollOutput:
    dx: float = 0.0
    dy: float = 0.0
    is_scrolling: bool = False
    is_inertia: bool = False
    axis: ScrollAxis = None


class ScrollController:
    """Continuous 2D scroll with axis lock, breakout, and smooth exponential inertia."""

    def __init__(self, scroll_cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = scroll_cfg or {}
        self.enabled: bool = bool(cfg.get("enabled", True))
        self.speed: float = float(cfg.get("speed", 1.0))
        self.deadzone: float = float(cfg.get("deadzone", 1.0))
        self.enable_axis_lock: bool = bool(cfg.get("enable_axis_lock", True))
        self.axis_lock_distance: float = float(cfg.get("axis_lock_distance", 15.0))
        self.axis_lock_ratio: float = float(cfg.get("axis_lock_ratio", 2.0))
        self.breakout_threshold: float = float(cfg.get("breakout_threshold", 8.0))
        self.inertia_friction: float = float(cfg.get("inertia_friction", 0.95))
        self.min_inertia_speed: float = float(cfg.get("min_inertia_speed", 5.0))
        self.natural_scrolling: bool = bool(cfg.get("natural_scrolling", True))

        self._axis: ScrollAxis = None
        self._acc_x: float = 0.0
        self._acc_y: float = 0.0
        self._vel_x: float = 0.0
        self._vel_y: float = 0.0
        self._inertia_x: float = 0.0
        self._inertia_y: float = 0.0
        self._is_scrolling: bool = False
        self._is_inertia: bool = False
        self._last_ts_ms: Optional[float] = None

    def reset(self) -> None:
        self._axis = None
        self._acc_x = 0.0
        self._acc_y = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._inertia_x = 0.0
        self._inertia_y = 0.0
        self._is_scrolling = False
        self._is_inertia = False
        self._last_ts_ms = None

    def process(
        self,
        delta_x: float,
        delta_y: float,
        is_gesture_active: bool,
        timestamp_ms: float,
    ) -> ScrollOutput:
        if not self.enabled:
            self.reset()
            return ScrollOutput()

        dt_s = (
            max(0.001, (timestamp_ms - self._last_ts_ms) / 1000.0)
            if self._last_ts_ms is not None and timestamp_ms > self._last_ts_ms
            else (1.0 / 60.0)
        )
        self._last_ts_ms = timestamp_ms

        dx = 0.0
        dy = 0.0
        processed = False

        # Transition: active scroll -> inertia coasting
        if self._is_scrolling and not is_gesture_active:
            speed_sq = self._vel_x * self._vel_x + self._vel_y * self._vel_y
            if not self._is_inertia and speed_sq > self.min_inertia_speed * self.min_inertia_speed:
                self._is_inertia = True
                self._inertia_x = self._vel_x
                self._inertia_y = self._vel_y
            self._is_scrolling = False

        if is_gesture_active:
            self._is_inertia = False
            self._is_scrolling = True
            processed = True

            raw_dx = 0.0 if abs(delta_x) < self.deadzone else delta_x
            raw_dy = 0.0 if abs(delta_y) < self.deadzone else delta_y

            sign = -1.0 if self.natural_scrolling else 1.0
            vx = sign * raw_dx * self.speed
            vy = sign * raw_dy * self.speed
            final_vx, final_vy = vx, vy

            if self.enable_axis_lock:
                if self._axis is None:
                    self._acc_x += vx
                    self._acc_y += vy
                    acc_x = abs(self._acc_x)
                    acc_y = abs(self._acc_y)
                    if math.hypot(acc_x, acc_y) > self.axis_lock_distance:
                        if acc_x > acc_y * self.axis_lock_ratio:
                            self._axis = "x"
                            final_vx, final_vy = self._acc_x, 0.0
                        elif acc_y > acc_x * self.axis_lock_ratio:
                            self._axis = "y"
                            final_vx, final_vy = 0.0, self._acc_y
                        else:
                            self._axis = "free"
                            final_vx, final_vy = self._acc_x, self._acc_y
                    else:
                        final_vx, final_vy = 0.0, 0.0
                elif self._axis == "x":
                    if abs(vy) > self.breakout_threshold and abs(vy) > abs(vx) * 1.5:
                        self._axis = "free"
                    else:
                        final_vy = 0.0
                elif self._axis == "y":
                    if abs(vx) > self.breakout_threshold and abs(vx) > abs(vy) * 1.5:
                        self._axis = "free"
                    else:
                        final_vx = 0.0
            else:
                self._axis = "free"

            dx, dy = final_vx, final_vy
            alpha = 0.5
            self._vel_x = self._vel_x * (1.0 - alpha) + dx * alpha
            self._vel_y = self._vel_y * (1.0 - alpha) + dy * alpha

        elif self._is_inertia:
            # Rate-independent exponential decay calibrated to 60 Hz reference step
            steps = max(0.25, min(4.0, dt_s / (1.0 / 60.0)))
            decay = self.inertia_friction ** steps
            self._inertia_x *= decay
            self._inertia_y *= decay
            dx, dy = self._inertia_x, self._inertia_y
            if (self._inertia_x * self._inertia_x + self._inertia_y * self._inertia_y) < 0.5:
                self._is_inertia = False
                self._inertia_x = 0.0
                self._inertia_y = 0.0
                dx, dy = 0.0, 0.0
        else:
            self._axis = None
            self._acc_x = 0.0
            self._acc_y = 0.0

        return ScrollOutput(
            dx=dx,
            dy=dy,
            is_scrolling=processed,
            is_inertia=self._is_inertia,
            axis=self._axis,
        )
