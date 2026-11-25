"""
3D Palm Cylinder Radial Heuristic & Palm Wheel Menu Detector.
Ported from Airpoint SDK `AirKeyboardRadial.ts` & `PalmWheelDetector.ts` (Apache-2.0).

Geometry (rotation-invariant 3D cylinder):
  - Origin = midpoint of Wrist (0) and Middle MCP (9).
  - Axis   = unit palm normal `normalize((LM9 - LM0) x (LM17 - LM5))`.
  - Radius = `radius_multiplier * ||LM9 - LM0||` (default `0.5 * palm_length`).
  - For each fingertip `P`: orthogonal distance to axis `||(P - origin) x axis|| / radius`.
    `< 1.0` => finger curled inside cylinder (pressed); `> 1.1` => extended (released).

Interaction:
  - Hold open palm facing camera (`palm_facing_dot >= 0.50`) for `activation_hold_ms` (200 ms)
    -> enters `active` state.
  - Curl any single finger inside the cylinder and release -> fires that finger's bound action
    (e.g. `toggle_pause`, `mission_control`, `recenter`, `spotlight`, `media_play_pause`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence, Tuple

from aerohand.core.palm_frame import compute_palm_facing_dot
from aerohand.core.types import Handedness, Landmark

FingerName = Literal["thumb", "index", "middle", "ring", "pinky"]
PalmWheelState = Literal["inactive", "activating", "active", "confirmed"]

_FINGER_TIP_MAP: Tuple[Tuple[FingerName, int, float], ...] = (
    ("thumb", 4, 0.70),
    ("index", 8, 1.00),
    ("middle", 12, 1.00),
    ("ring", 16, 1.00),
    ("pinky", 20, 0.90),
)


@dataclass(frozen=True)
class RadialCylinderResult:
    origin: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    palm_size: float
    radius: float
    normalized_distances: Dict[FingerName, float]


def calculate_radial_cylinder(
    landmarks: Sequence[Landmark],
    radius_multiplier: float = 0.50,
) -> Optional[RadialCylinderResult]:
    """Compute 3D Palm Cylinder radial distances normalized by cylinder radius."""
    if len(landmarks) < 21:
        return None

    w = landmarks[0]
    m = landmarks[9]
    i_mcp = landmarks[5]
    p_mcp = landmarks[17]

    ox = 0.5 * (w.x + m.x)
    oy = 0.5 * (w.y + m.y)
    oz = 0.5 * (w.z + m.z)

    v1x, v1y, v1z = m.x - w.x, m.y - w.y, m.z - w.z
    v2x, v2y, v2z = p_mcp.x - i_mcp.x, p_mcp.y - i_mcp.y, p_mcp.z - i_mcp.z

    nx = v1y * v2z - v1z * v2y
    ny = v1z * v2x - v1x * v2z
    nz = v1x * v2y - v1y * v2x
    n_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    palm_size = math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
    if n_len < 1e-7 or palm_size < 1e-7:
        return None

    ax, ay, az = nx / n_len, ny / n_len, nz / n_len
    radius = max(1e-7, radius_multiplier * palm_size)

    norm_dists: Dict[FingerName, float] = {}
    for fname, tip_idx, _ in _FINGER_TIP_MAP:
        tip = landmarks[tip_idx]
        vx, vy, vz = tip.x - ox, tip.y - oy, tip.z - oz
        cx = vy * az - vz * ay
        cy = vz * ax - vx * az
        cz = vx * ay - vy * ax
        dist = math.sqrt(cx * cx + cy * cy + cz * cz)
        norm_dists[fname] = dist / radius

    return RadialCylinderResult(
        origin=(ox, oy, oz),
        axis=(ax, ay, az),
        palm_size=palm_size,
        radius=radius,
        normalized_distances=norm_dists,
    )


@dataclass
class PalmWheelOutput:
    state: PalmWheelState = "inactive"
    progress: float = 0.0
    selected_finger: Optional[FingerName] = None
    triggered_action: Optional[str] = None
    finger_extended: Dict[FingerName, bool] = field(
        default_factory=lambda: {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}
    )


class PalmWheelDetector:
    """Open-palm 5-finger radial quick-menu detector."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        c = cfg or {}
        self.enabled: bool = bool(c.get("enabled", True))
        self.activation_hold_ms: float = float(c.get("activation_hold_ms", 200.0))
        self.press_down_threshold: float = float(c.get("press_down_threshold", 0.95))
        self.press_up_threshold: float = float(c.get("press_up_threshold", 1.10))
        self.cylinder_radius_multiplier: float = float(c.get("cylinder_radius_multiplier", 0.50))
        self.palm_facing_dot_threshold: float = float(c.get("palm_facing_dot_threshold", 0.50))
        self.palm_facing_dot_hysteresis: float = float(c.get("palm_facing_dot_hysteresis", 0.05))
        self.open_palm_grace_ms: float = float(c.get("open_palm_grace_ms", 100.0))
        self.confirmation_hold_ms: float = float(c.get("confirmation_hold_ms", 200.0))
        self.cooldown_ms: float = float(c.get("cooldown_ms", 350.0))

        default_slots: Dict[FingerName, Optional[str]] = {
            "thumb": "toggle_pause",
            "index": "mission_control",
            "middle": "recenter",
            "ring": "spotlight",
            "pinky": "media_play_pause",
        }
        raw_slots = c.get("slots") or {}
        self.slots: Dict[FingerName, Optional[str]] = {
            fname: raw_slots.get(fname, default_slots[fname]) for fname, _, _ in _FINGER_TIP_MAP
        }

        self._state: PalmWheelState = "inactive"
        self._activation_start_ms: Optional[float] = None
        self._confirm_start_ms: Optional[float] = None
        self._last_action_ms: float = -1e9
        self._last_open_palm_ms: Optional[float] = None
        self._open_palm_stable: bool = False
        self._selected_finger: Optional[FingerName] = None
        self._finger_pressed: Dict[FingerName, bool] = {f: False for f, _, _ in _FINGER_TIP_MAP}

    def reset(self) -> None:
        self._state = "inactive"
        self._activation_start_ms = None
        self._confirm_start_ms = None
        self._last_open_palm_ms = None
        self._open_palm_stable = False
        self._selected_finger = None
        self._finger_pressed = {f: False for f, _, _ in _FINGER_TIP_MAP}

    def process(
        self,
        landmarks: Optional[Sequence[Landmark]],
        timestamp_ms: float,
        handedness: Handedness = "Right",
    ) -> PalmWheelOutput:
        if not self.enabled or not landmarks or len(landmarks) < 21:
            self.reset()
            return PalmWheelOutput()

        if self._state == "inactive" and (timestamp_ms - self._last_action_ms) < self.cooldown_ms:
            return PalmWheelOutput(state="inactive")

        cyl = calculate_radial_cylinder(landmarks, radius_multiplier=self.cylinder_radius_multiplier)
        if cyl is None:
            self.reset()
            return PalmWheelOutput()

        dists = cyl.normalized_distances
        facing_dot = compute_palm_facing_dot(landmarks, is_left_hand=(handedness == "Left"), mirrored=True)
        hyst = self.palm_facing_dot_hysteresis
        facing_thresh = max(
            0.0,
            min(1.0, self.palm_facing_dot_threshold + (-hyst if self._open_palm_stable else hyst)),
        )
        all_up = all(dists[f] > self.press_up_threshold * scale for f, _, scale in _FINGER_TIP_MAP)
        open_raw = all_up and (facing_dot >= facing_thresh)

        open_now = open_raw
        if open_raw:
            self._open_palm_stable = True
            self._last_open_palm_ms = timestamp_ms
        else:
            within_grace = (
                self._state == "activating"
                and self._last_open_palm_ms is not None
                and (timestamp_ms - self._last_open_palm_ms) <= self.open_palm_grace_ms
            )
            if within_grace:
                open_now = True
            else:
                self._open_palm_stable = False

        finger_ext: Dict[FingerName, bool] = {
            f: (dists[f] > self.press_down_threshold * scale) for f, _, scale in _FINGER_TIP_MAP
        }

        out = PalmWheelOutput(state=self._state, finger_extended=finger_ext)

        if self._state == "inactive":
            if open_now:
                self._state = "activating"
                self._activation_start_ms = timestamp_ms
                out.state = "activating"
                out.progress = 0.0
        elif self._state == "activating":
            if not open_now:
                self.reset()
                return PalmWheelOutput(state="inactive", finger_extended=finger_ext)
            start_ms = self._activation_start_ms if self._activation_start_ms is not None else timestamp_ms
            elapsed = timestamp_ms - start_ms
            out.progress = min(1.0, elapsed / max(1.0, self.activation_hold_ms))
            if elapsed >= self.activation_hold_ms:
                self._state = "active"
                self._finger_pressed = {f: False for f, _, _ in _FINGER_TIP_MAP}
                out.state = "active"
                out.progress = 1.0
        elif self._state == "active":
            # Exit menu if palm turns away from camera
            if facing_dot < max(0.0, self.palm_facing_dot_threshold - 0.15):
                self.reset()
                return PalmWheelOutput(state="inactive", finger_extended=finger_ext)

            out.progress = 1.0
            for fname, _, scale in _FINGER_TIP_MAP:
                d = dists[fname]
                down_th = self.press_down_threshold * scale
                up_th = self.press_up_threshold * scale
                was_pressed = self._finger_pressed[fname]
                if not was_pressed and d < down_th:
                    self._finger_pressed[fname] = True
                    self._selected_finger = fname
                elif was_pressed and d > up_th:
                    self._finger_pressed[fname] = False
                    action = self.slots.get(fname)
                    if action and action != "unassigned":
                        self._state = "confirmed"
                        self._confirm_start_ms = timestamp_ms
                        self._last_action_ms = timestamp_ms
                        self._selected_finger = fname
                        out.state = "confirmed"
                        out.selected_finger = fname
                        out.triggered_action = action
                        return out
            out.selected_finger = self._selected_finger
        elif self._state == "confirmed":
            out.progress = 1.0
            out.selected_finger = self._selected_finger
            c_start = self._confirm_start_ms if self._confirm_start_ms is not None else timestamp_ms
            elapsed = timestamp_ms - c_start
            if elapsed >= self.confirmation_hold_ms:
                self.reset()
                out.state = "inactive"
                out.progress = 0.0

        return out
