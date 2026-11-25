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
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

from aerohand.core.filters import OneEuroFilter1D
from aerohand.core.types import Landmark


def adaptive_lerp_speed(
    speed_px_s: float,
    *,
    min_speed: float = 40.0,
    max_speed: float = 70.0,
    low_vel: float = 150.0,
    high_vel: float = 1200.0,
) -> float:
    """Monotonically scale lerp stiffness from `min_speed` (precise) to `max_speed` (fast flick)."""
    v = max(0.0, float(speed_px_s))
    if v <= low_vel or high_vel <= low_vel:
        return float(min_speed)
    if v >= high_vel:
        return float(max_speed)
    t = (v - low_vel) / (high_vel - low_vel)
    return float(min_speed + t * (max_speed - min_speed))


def isotropic_distance_2d(
    a: Landmark,
    b: Landmark,
    frame_aspect: float = 1.0,
) -> float:
    """
    Pixel-true isotropic 2D distance between normalized landmarks `a` and `b`
    given camera frame aspect ratio `width / height` (area-preserving scaling).
    """
    ar = max(0.25, float(frame_aspect))
    sx = math.sqrt(ar)
    sy = 1.0 / sx
    return math.hypot((a.x - b.x) * sx, (a.y - b.y) * sy)


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
        self._history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=64
        )  # (ts_ms, x, y, speed_px_s) over ~0.5s

    def reset(self) -> None:
        self.filter_x.reset()
        self.filter_y.reset()
        self._last_finger_pos = None
        self._last_pointer_pos = None
        self._last_frame_ms = None
        self._skip_remaining = self.reacquire_skip_frames
        self._display_speed = 0.0
        self._display_finger_speed = 0.0
        self._history.clear()

    def record_history(
        self,
        timestamp_ms: float,
        x: float,
        y: float,
        speed_px_s: float = 0.0,
        window_ms: float = 500.0,
    ) -> None:
        self._history.append((float(timestamp_ms), float(x), float(y), max(0.0, float(speed_px_s))))
        cutoff = float(timestamp_ms) - float(window_ms)
        while len(self._history) > 1 and self._history[0][0] < cutoff:
            self._history.popleft()

    def lookup_position(self, target_timestamp_ms: float) -> Optional[Tuple[float, float]]:
        """Return the buffered cursor position closest to `target_timestamp_ms` (pre-pinch anchor)."""
        if not self._history:
            return self._last_pointer_pos
        target = float(target_timestamp_ms)
        best = min(self._history, key=lambda item: abs(item[0] - target))
        return (best[1], best[2])

    def lookup_position_guarded(
        self,
        target_timestamp_ms: float,
        live_pos: Tuple[float, float],
        *,
        max_speed_px_s: float = 250.0,
        max_offset_px: float = 24.0,
        fallback_pos: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """
        Speed-guarded & distance-clamped pre-pinch lookback (V031-04):
          - If pointer speed at lookback target > `max_speed_px_s` (>0), ignore lookback and
            return `fallback_pos` (dwell lock) or `live_pos`.
          - Clamp the anchor offset from `live_pos` to `<= max_offset_px` (>0).
        """
        if not self._history:
            return fallback_pos if fallback_pos is not None else live_pos
        target = float(target_timestamp_ms)
        best = min(self._history, key=lambda item: abs(item[0] - target))
        _, hx, hy, spd = best
        if max_speed_px_s > 0.0 and spd > max_speed_px_s:
            return fallback_pos if fallback_pos is not None else live_pos
        lx, ly = float(live_pos[0]), float(live_pos[1])
        dx = hx - lx
        dy = hy - ly
        dist = math.hypot(dx, dy)
        if max_offset_px > 0.0 and dist > max_offset_px:
            scale = max_offset_px / dist
            return (lx + dx * scale, ly + dy * scale)
        return (hx, hy)

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
        *,
        frame_aspect: float = 1.0,
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
                palm_size = isotropic_distance_2d(wrist, middle_mcp, frame_aspect)
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
            self.record_history(timestamp_ms, final_x, final_y, speed_px_s=cursor_speed)

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
    using Airpoint's frame-rate-independent exponential lerp (`1 - exp(-lerp_speed * dt)`),
    with optional velocity-adaptive stiffness (40 when precise -> 70 when flicking).
    """

    def __init__(
        self,
        lerp_speed: float = 55.0,
        snap_threshold_px: float = 0.25,
        adaptive: bool = True,
    ) -> None:
        self.lerp_speed = float(lerp_speed)
        self.snap_threshold_px = float(snap_threshold_px)
        self.adaptive = bool(adaptive)
        self._current: Optional[Tuple[float, float]] = None
        self._target: Optional[Tuple[float, float]] = None
        self._speed_px_s: float = 0.0

    def reset(self, pos: Optional[Tuple[float, float]] = None) -> None:
        self._current = pos
        self._target = pos
        self._speed_px_s = 0.0

    def sync_current_to_anchor(
        self,
        anchor_x: float,
        anchor_y: float,
        live_x: float,
        live_y: float,
    ) -> None:
        """Align lerp current position to an anchored click so subsequent ticks ease smoothly."""
        self._current = (float(anchor_x), float(anchor_y))
        self._target = (float(live_x), float(live_y))

    def set_target(self, x: float, y: float, *, speed_px_s: Optional[float] = None) -> None:
        self._target = (float(x), float(y))
        if speed_px_s is not None:
            self._speed_px_s = max(0.0, float(speed_px_s))
        if self._current is None:
            self._current = (float(x), float(y))

    def effective_speed(self) -> float:
        if not self.adaptive or self.lerp_speed <= 0.0:
            return self.lerp_speed
        base_scale = self.lerp_speed / 55.0
        return adaptive_lerp_speed(self._speed_px_s) * base_scale

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
        eff_speed = self.effective_speed()
        if err <= self.snap_threshold_px or eff_speed <= 0.0:
            nx, ny = tx, ty
        else:
            factor = 1.0 - math.exp(-eff_speed * max(0.001, dt_s))
            nx = cx + err_x * factor
            ny = cy + err_y * factor
        self._current = (nx, ny)
        return nx, ny, nx - cx, ny - cy
