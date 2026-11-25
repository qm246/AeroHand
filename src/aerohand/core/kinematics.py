"""
Relative Cursor Kinematics & 60 Hz Lerp Interpolator.

Combines:
- Airpoint SDK `CursorKinematics.ts` (Apache-2.0): Stage-2 OneEuro filter, distance
  compensation (`reference_palm_size / palm_size` clamped `[0.5, 3.0]`), speed-adaptive
  pointer acceleration, jump rejection (`> 15% width`), and exponential lerp (`lerpSpeed=55`).
- GestureCtl relative pointer hardening: OS cursor anchor sync (no jump on session entry),
  reacquire skip frames, micro-jitter deadzone, and sub-pixel stability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from aerohand.core.filters import OneEuroFilter1D
from aerohand.core.types import Landmark


@dataclass
class KinematicsOutput:
    x: float = 0.0
    y: float = 0.0
    delta_x: float = 0.0
    delta_y: float = 0.0
    speed: float = 0.0
    finger_speed: float = 0.0
    is_valid: bool = False


class CursorKinematics:
    """Computes smooth, distance-compensated relative cursor movement."""

    def __init__(self, cursor_cfg: Optional[Dict[str, Any]] = None) -> None:
        cfg = cursor_cfg or {}
        self.enable_filter: bool = bool(cfg.get("enable_filter", True))
        self.filter_min_cutoff: float = float(cfg.get("filter_min_cutoff", 1.0))
        self.filter_beta: float = float(cfg.get("filter_beta", 0.01))
        self.filter_d_cutoff: float = float(cfg.get("filter_d_cutoff", 1.0))
        self.pointer_sensitivity: float = float(cfg.get("pointer_sensitivity", 2.2))
        self.pointer_acceleration: float = float(cfg.get("pointer_acceleration", 1.4))
        self.enable_distance_compensation: bool = bool(cfg.get("enable_distance_compensation", True))
        self.reference_palm_size: float = float(cfg.get("reference_palm_size", 0.10))
        self.max_jump_ratio: float = float(cfg.get("max_jump_ratio", 0.15))
        self.deadzone_px: float = float(cfg.get("deadzone_px", 0.35))
        self.reacquire_skip_frames: int = int(cfg.get("reacquire_skip_frames", 2))

        self.filter_x = OneEuroFilter1D(
            self.filter_min_cutoff, self.filter_beta, self.filter_d_cutoff, gap_reset_ms=250.0
        )
        self.filter_y = OneEuroFilter1D(
            self.filter_min_cutoff, self.filter_beta, self.filter_d_cutoff, gap_reset_ms=250.0
        )

        self._last_finger_pos: Optional[Tuple[float, float]] = None
        self._last_pointer_pos: Optional[Tuple[float, float]] = None
        self._last_frame_ms: Optional[float] = None
        self._skip_remaining: int = 0
        self._display_speed: float = 0.0
        self._display_finger_speed: float = 0.0

    def reset(self) -> None:
        self.filter_x.reset()
        self.filter_y.reset()
        self._last_finger_pos = None
        self._last_pointer_pos = None
        self._last_frame_ms = None
        self._skip_remaining = self.reacquire_skip_frames
        self._display_speed = 0.0
        self._display_finger_speed = 0.0

    def set_cursor_position(self, x: float, y: float) -> None:
        """Anchor pointer position (e.g. from real OS cursor or recenter) without jumping."""
        self._last_pointer_pos = (float(x), float(y))
        self._last_finger_pos = None
        self._skip_remaining = self.reacquire_skip_frames
        self.filter_x.reset()
        self.filter_y.reset()

    @property
    def current_position(self) -> Optional[Tuple[float, float]]:
        return self._last_pointer_pos

    def process(
        self,
        raw_target_x: float,
        raw_target_y: float,
        landmarks: Sequence[Landmark],
        timestamp_ms: float,
        width: float,
        height: float,
    ) -> KinematicsOutput:
        out = KinematicsOutput()

        if self._last_finger_pos is None:
            self._last_finger_pos = (raw_target_x, raw_target_y)
            if self._last_pointer_pos is None:
                self._last_pointer_pos = (raw_target_x, raw_target_y)
            self._last_frame_ms = timestamp_ms
            out.x, out.y = self._last_pointer_pos
            return out

        if self._skip_remaining > 0:
            self._skip_remaining -= 1
            self._last_finger_pos = (raw_target_x, raw_target_y)
            self._last_frame_ms = timestamp_ms
            assert self._last_pointer_pos is not None
            out.x, out.y = self._last_pointer_pos
            return out

        prev_fx, prev_fy = self._last_finger_pos
        dx = raw_target_x - prev_fx
        dy = raw_target_y - prev_fy

        max_jump = max(1.0, width * self.max_jump_ratio)
        jump_dist_sq = dx * dx + dy * dy
        assert self._last_pointer_pos is not None
        prev_px, prev_py = self._last_pointer_pos

        if jump_dist_sq < max_jump * max_jump:
            sensitivity = self.pointer_sensitivity
            distance_scale = 1.0
            if self.enable_distance_compensation and len(landmarks) >= 10:
                wrist = landmarks[0]
                middle_mcp = landmarks[9]
                palm_size = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)
                if palm_size > 0.01:
                    distance_scale = max(0.5, min(3.0, self.reference_palm_size / palm_size))
            sensitivity *= distance_scale

            prev_ts = self._last_frame_ms if self._last_frame_ms is not None else timestamp_ms
            dt_s = max(0.001, (timestamp_ms - prev_ts) / 1000.0)

            norm_dx = dx * distance_scale
            norm_dy = dy * distance_scale
            finger_speed = math.hypot(norm_dx, norm_dy) / dt_s

            if self.pointer_acceleration > 0.0 and finger_speed > 0.0:
                boost = finger_speed * self.pointer_acceleration * 0.0005
                sensitivity *= 1.0 + min(2.0, boost)

            raw_x = max(0.0, min(width, prev_px + dx * sensitivity))
            raw_y = max(0.0, min(height, prev_py + dy * sensitivity))

            if self.enable_filter:
                final_x = self.filter_x.filter(raw_x, timestamp_ms)
                final_y = self.filter_y.filter(raw_y, timestamp_ms)
            else:
                final_x, final_y = raw_x, raw_y

            step_dx = final_x - prev_px
            step_dy = final_y - prev_py
            if math.hypot(step_dx, step_dy) < self.deadzone_px:
                final_x, final_y = prev_px, prev_py
                step_dx, step_dy = 0.0, 0.0

            self._last_pointer_pos = (final_x, final_y)
            cursor_speed = math.hypot(step_dx, step_dy) / dt_s

            alpha = 0.15
            self._display_speed = self._display_speed * (1.0 - alpha) + cursor_speed * alpha
            self._display_finger_speed = self._display_finger_speed * (1.0 - alpha) + finger_speed * alpha

            out.x = final_x
            out.y = final_y
            out.delta_x = step_dx
            out.delta_y = step_dy
            out.speed = self._display_speed
            out.finger_speed = self._display_finger_speed
            out.is_valid = True
        else:
            out.x = prev_px
            out.y = prev_py
            out.speed = self._display_speed
            out.finger_speed = self._display_finger_speed
            out.is_valid = False

        self._last_finger_pos = (raw_target_x, raw_target_y)
        self._last_frame_ms = timestamp_ms
        return out


class CursorLerpInterpolator:
    """
    Smoothly interpolates ~30 Hz camera cursor updates to 60 Hz OS cursor motion
    using Airpoint's frame-rate-independent exponential lerp (`1 - exp(-lerp_speed * dt)`).
    """

    def __init__(self, lerp_speed: float = 55.0, snap_threshold_px: float = 0.25) -> None:
        self.lerp_speed = float(lerp_speed)
        self.snap_threshold_px = float(snap_threshold_px)
        self._current: Optional[Tuple[float, float]] = None
        self._target: Optional[Tuple[float, float]] = None

    def reset(self, pos: Optional[Tuple[float, float]] = None) -> None:
        self._current = pos
        self._target = pos

    def set_target(self, x: float, y: float) -> None:
        self._target = (float(x), float(y))
        if self._current is None:
            self._current = (float(x), float(y))

    def step(self, dt_s: float) -> Optional[Tuple[float, float, float, float]]:
        """
        Advance one interpolation tick.
        Returns `(new_x, new_y, dx, dy)` if cursor moved, else `None`.
        """
        if self._current is None or self._target is None:
            return None
        cx, cy = self._current
        tx, ty = self._target
        err_x = tx - cx
        err_y = ty - cy
        err = math.hypot(err_x, err_y)
        if err < 1e-4:
            return None
        if err <= self.snap_threshold_px or self.lerp_speed <= 0.0:
            nx, ny = tx, ty
        else:
            factor = 1.0 - math.exp(-self.lerp_speed * max(0.001, dt_s))
            nx = cx + err_x * factor
            ny = cy + err_y * factor
        self._current = (nx, ny)
        return nx, ny, nx - cx, ny - cy
