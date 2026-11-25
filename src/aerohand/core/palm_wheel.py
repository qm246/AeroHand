"""
3D Palm Cylinder Radial Heuristic & Full Airpoint-Parity Palm Wheel Detector.
Ported from Airpoint SDK `AirKeyboardRadial.ts` & `PalmWheelDetector.ts` (Apache-2.0)
with GestureCtl false-alarm hardening.

Features:
  - Rotation-invariant 3D Palm Cylinder (`R = 0.5 * palm_length`) around palm normal.
  - Per-hand independent 5-slot trees (`Left` L1..L5 and `Right` R1..R5).
  - Arbitrary nested multi-level Submenus (`Tracking`, `Player` -> `Volume`, `Window`, `Edit`, `Apps`).
  - `submenu_back` navigation + stay-open after confirmation when palm remains extended
    (so repeated commands like Volume Up / Next Track / Copy -> Paste work seamlessly).
  - Rapid dismiss on closing fist (`down_count > 1`).
  - Motion-anchored 350 ms activation hold to eliminate waving/hair-touch false alarms.
  - Emits fingertip screen positions, active slot metadata, breadcrumb path, and press audio events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from aerohand.core.palm_frame import compute_palm_facing_dot
from aerohand.core.types import Handedness, Landmark

FingerName = Literal["thumb", "index", "middle", "ring", "pinky"]
PalmWheelState = Literal["inactive", "activating", "active", "confirmed"]

_FINGER_TIP_MAP: Tuple[Tuple[FingerName, int, int, float], ...] = (
    ("thumb", 4, 2, 0.70),
    ("index", 8, 5, 1.00),
    ("middle", 12, 9, 1.00),
    ("ring", 16, 13, 1.00),
    ("pinky", 20, 17, 0.90),
)

_ACTION_ICON_HINTS: Dict[str, str] = {
    "toggle_pause": "pause",
    "tracking_toggle": "pause",
    "recenter": "crosshair",
    "show_main_window": "sliders",
    "toggle_camera_hud": "camera",
    "stop_tracking": "power",
    "mission_control": "layout",
    "spotlight": "search",
    "dictation_toggle": "mic",
    "dictation": "mic",
    "siri_listen": "sparkles",
    "media_play_pause": "play",
    "media_next": "skip-forward",
    "media_prev": "skip-back",
    "volume_up": "volume-2",
    "volume_down": "volume-1",
    "volume_mute": "volume-x",
    "window_hide": "eye-off",
    "window_fullscreen": "maximize",
    "window_minimize": "minimize",
    "window_close": "x",
    "edit_copy": "copy",
    "edit_paste": "clipboard",
    "edit_cut": "scissors",
    "edit_undo": "undo",
    "edit_redo": "redo",
    "keyboard_emoji": "keyboard",
    "submenu_back": "arrow-left",
    "unassigned": "minus",
}


def _default_hand_bindings(hand: Handedness) -> Dict[str, Any]:
    if hand == "Left":
        return {
            "thumb": "siri_listen",
            "index": {
                "type": "submenu",
                "label": "Window",
                "icon": "window",
                "items": {
                    "thumb": "window_hide",
                    "index": "window_fullscreen",
                    "middle": "window_minimize",
                    "ring": "window_close",
                    "pinky": "submenu_back",
                },
            },
            "middle": {
                "type": "submenu",
                "label": "Edit",
                "icon": "edit",
                "items": {
                    "thumb": "edit_paste",
                    "index": "edit_copy",
                    "middle": "edit_cut",
                    "ring": "edit_redo",
                    "pinky": "edit_undo",
                },
            },
            "ring": {
                "type": "submenu",
                "label": "Apps",
                "icon": "apps",
                "items": {
                    "thumb": "open_app:/System/Library/CoreServices/Finder.app",
                    "index": "open_app:/Applications/Safari.app",
                    "middle": "open_app:/System/Applications/Notes.app",
                    "ring": "open_app:/System/Applications/Utilities/Terminal.app",
                    "pinky": "submenu_back",
                },
            },
            "pinky": "keyboard_emoji",
        }
    return {
        "thumb": {
            "type": "submenu",
            "label": "Tracking",
            "icon": "pointer",
            "items": {
                "thumb": "toggle_pause",
                "index": "recenter",
                "middle": "show_main_window",
                "ring": "toggle_camera_hud",
                "pinky": "submenu_back",
            },
        },
        "index": "mission_control",
        "middle": "spotlight",
        "ring": "dictation_toggle",
        "pinky": {
            "type": "submenu",
            "label": "Player",
            "icon": "play",
            "items": {
                "thumb": "submenu_back",
                "index": "media_prev",
                "middle": "media_play_pause",
                "ring": "media_next",
                "pinky": {
                    "type": "submenu",
                    "label": "Volume",
                    "icon": "volume",
                    "items": {
                        "thumb": "submenu_back",
                        "index": "volume_down",
                        "middle": "volume_mute",
                        "ring": "volume_up",
                        "pinky": "unassigned",
                    },
                },
            },
        },
    }


def _describe_binding(finger: FingerName, raw: Any) -> Dict[str, Any]:
    if not raw or raw == "unassigned":
        return {"finger": finger, "id": "unassigned", "label": "", "icon": "minus", "is_submenu": False}
    if isinstance(raw, dict) and raw.get("type") == "submenu":
        return {
            "finger": finger,
            "id": f"submenu:{raw.get('label', 'Menu')}",
            "label": str(raw.get("label", "Menu")),
            "icon": str(raw.get("icon", "folder")),
            "is_submenu": True,
        }
    act = str(raw)
    if act.startswith("open_app:"):
        app_path = act.split(":", 1)[1].strip()
        app_name = app_path.rstrip("/").rsplit("/", 1)[-1].replace(".app", "")
        return {"finger": finger, "id": act, "label": app_name or "App", "icon": "app", "is_submenu": False}
    return {
        "finger": finger,
        "id": act,
        "label": act,
        "icon": _ACTION_ICON_HINTS.get(act, "circle"),
        "is_submenu": False,
    }


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
    for fname, tip_idx, _, _ in _FINGER_TIP_MAP:
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
    press_event: Optional[Literal["down", "up"]] = None
    menu_path: List[str] = field(default_factory=list)
    menu_label: Optional[str] = None
    slots: List[Dict[str, Any]] = field(default_factory=list)
    finger_positions: Optional[Dict[str, Tuple[float, float]]] = None
    finger_extended: Dict[FingerName, bool] = field(
        default_factory=lambda: {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}
    )


class PalmWheelDetector:
    """Open-palm 5-finger radial quick-menu detector with multi-level Submenus."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None, hand: Handedness = "Right") -> None:
        self.hand: Handedness = hand
        self._state: PalmWheelState = "inactive"
        self._activation_start_ms: Optional[float] = None
        self._activation_anchor: Optional[Tuple[float, float]] = None
        self._confirm_start_ms: Optional[float] = None
        self._last_action_ms: float = -1e9
        self._last_open_palm_ms: Optional[float] = None
        self._open_palm_stable: bool = False
        self._selected_finger: Optional[FingerName] = None
        self._finger_pressed: Dict[FingerName, bool] = {f: False for f, _, _, _ in _FINGER_TIP_MAP}
        self._menu_stack: List[Tuple[str, Dict[str, Any]]] = []

        self.update_config(cfg)

    def update_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        c = cfg or {}
        enabled_global = bool(c.get("enabled", True))
        eh = c.get("enabled_hands")
        if isinstance(eh, dict):
            self.enabled = enabled_global and bool(eh.get(self.hand, True))
        else:
            self.enabled = enabled_global

        self.activation_hold_ms: float = float(c.get("activation_hold_ms", 350.0))
        self.max_activation_motion_norm: float = float(c.get("max_activation_motion_norm", 0.035))
        self.press_down_threshold: float = float(c.get("press_down_threshold", 0.95))
        self.press_up_threshold: float = float(c.get("press_up_threshold", 1.10))
        self.cylinder_radius_multiplier: float = float(c.get("cylinder_radius_multiplier", 0.50))
        self.palm_facing_dot_threshold: float = float(c.get("palm_facing_dot_threshold", 0.60))
        self.palm_facing_dot_hysteresis: float = float(c.get("palm_facing_dot_hysteresis", 0.05))
        self.open_palm_grace_ms: float = float(c.get("open_palm_grace_ms", 100.0))
        self.confirmation_hold_ms: float = float(c.get("confirmation_hold_ms", 200.0))
        self.confirmation_grace_ms: float = float(c.get("confirmation_grace_ms", 200.0))
        self.cooldown_ms: float = float(c.get("cooldown_ms", 300.0))

        # Support new nested `bindings.<Hand>` as well as legacy flat `slots`
        bindings_cfg = c.get("bindings")
        if isinstance(bindings_cfg, dict) and isinstance(bindings_cfg.get(self.hand), dict):
            self._root_bindings: Dict[str, Any] = dict(bindings_cfg[self.hand])
        elif isinstance(c.get("slots"), dict):
            self._root_bindings = dict(c["slots"])
        else:
            self._root_bindings = _default_hand_bindings(self.hand)

    @property
    def _current_bindings(self) -> Dict[str, Any]:
        if self._menu_stack:
            return self._menu_stack[-1][1]
        return self._root_bindings

    def _current_slots_view(self) -> List[Dict[str, Any]]:
        cur = self._current_bindings
        return [_describe_binding(fname, cur.get(fname)) for fname, _, _, _ in _FINGER_TIP_MAP]

    def _current_menu_path(self) -> List[str]:
        return [lbl for lbl, _ in self._menu_stack]

    def reset(self) -> None:
        self._state = "inactive"
        self._activation_start_ms = None
        self._activation_anchor = None
        self._confirm_start_ms = None
        self._last_open_palm_ms = None
        self._open_palm_stable = False
        self._selected_finger = None
        self._finger_pressed = {f: False for f, _, _, _ in _FINGER_TIP_MAP}
        self._menu_stack.clear()

    @staticmethod
    def _extract_finger_positions(landmarks: Sequence[Landmark]) -> Dict[str, Tuple[float, float]]:
        w = landmarks[0]
        m = landmarks[9]
        pos: Dict[str, Tuple[float, float]] = {
            "palmCenter": (0.5 * (w.x + m.x), 0.5 * (w.y + m.y)),
        }
        for fname, _, mcp_idx, _ in _FINGER_TIP_MAP:
            pos[fname] = (landmarks[mcp_idx].x, landmarks[mcp_idx].y)
        return pos

    def process(
        self,
        landmarks: Optional[Sequence[Landmark]],
        timestamp_ms: float,
        handedness: Optional[Handedness] = None,
    ) -> PalmWheelOutput:
        hand = handedness or self.hand
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
        facing_dot = compute_palm_facing_dot(landmarks, is_left_hand=(hand == "Left"), mirrored=True)
        hyst = self.palm_facing_dot_hysteresis
        facing_thresh = max(
            0.0,
            min(1.0, self.palm_facing_dot_threshold + (-hyst if self._open_palm_stable else hyst)),
        )
        all_up = all(dists[f] > self.press_up_threshold * scale for f, _, _, scale in _FINGER_TIP_MAP)
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
            f: (dists[f] > self.press_down_threshold * scale) for f, _, _, scale in _FINGER_TIP_MAP
        }
        finger_pos = self._extract_finger_positions(landmarks)

        out = PalmWheelOutput(
            state=self._state,
            finger_extended=finger_ext,
            finger_positions=finger_pos,
            menu_path=self._current_menu_path(),
            menu_label=self._menu_stack[-1][0] if self._menu_stack else None,
            slots=self._current_slots_view(),
        )

        palm_cx, palm_cy = finger_pos["palmCenter"]

        if self._state == "inactive":
            if open_now:
                self._state = "activating"
                self._activation_start_ms = timestamp_ms
                self._activation_anchor = (palm_cx, palm_cy)
                out.state = "activating"
                out.progress = 0.0
        elif self._state == "activating":
            if not open_now:
                self.reset()
                return PalmWheelOutput(state="inactive", finger_extended=finger_ext)
            if self._activation_anchor is not None and self.max_activation_motion_norm > 0.0:
                drift = math.hypot(palm_cx - self._activation_anchor[0], palm_cy - self._activation_anchor[1])
                if drift > self.max_activation_motion_norm:
                    self._activation_start_ms = timestamp_ms
                    self._activation_anchor = (palm_cx, palm_cy)
            start_ms = self._activation_start_ms if self._activation_start_ms is not None else timestamp_ms
            elapsed = timestamp_ms - start_ms
            out.progress = min(1.0, elapsed / max(1.0, self.activation_hold_ms))
            if elapsed >= self.activation_hold_ms:
                self._state = "active"
                self._finger_pressed = {f: False for f, _, _, _ in _FINGER_TIP_MAP}
                out.state = "active"
                out.progress = 1.0
        elif self._state == "active":
            # Dismiss if palm turns away from camera
            if facing_dot < max(0.0, self.palm_facing_dot_threshold - 0.18):
                self.reset()
                return PalmWheelOutput(state="inactive", finger_extended=finger_ext)

            # Airpoint rapid dismiss: closing > 1 finger simultaneously (fist) cancels menu cleanly
            down_count = sum(1 for fname, _, _, scale in _FINGER_TIP_MAP if dists[fname] < self.press_down_threshold * scale)
            if down_count > 1:
                self.reset()
                return PalmWheelOutput(state="inactive", finger_extended=finger_ext)

            out.progress = 1.0
            cur_bindings = self._current_bindings

            for fname, _, _, scale in _FINGER_TIP_MAP:
                binding = cur_bindings.get(fname)
                if not binding or binding == "unassigned":
                    self._finger_pressed[fname] = False
                    continue

                d = dists[fname]
                down_th = self.press_down_threshold * scale
                up_th = self.press_up_threshold * scale
                was_pressed = self._finger_pressed[fname]

                if not was_pressed and d < down_th:
                    self._finger_pressed[fname] = True
                    self._selected_finger = fname
                    out.press_event = "down"
                elif was_pressed and d > up_th:
                    self._finger_pressed[fname] = False
                    out.press_event = "up"
                    self._selected_finger = fname

                    # Handle Submenu entry or Back navigation inside the wheel
                    if isinstance(binding, dict) and binding.get("type") == "submenu":
                        items = binding.get("items") or {}
                        label = str(binding.get("label", "Submenu"))
                        self._menu_stack.append((label, dict(items)))
                        self._state = "confirmed"
                        self._confirm_start_ms = timestamp_ms
                        out.state = "confirmed"
                        out.selected_finger = fname
                        out.menu_path = self._current_menu_path()
                        out.menu_label = label
                        out.slots = self._current_slots_view()
                        return out

                    action = str(binding)
                    if action == "submenu_back":
                        if self._menu_stack:
                            self._menu_stack.pop()
                        self._state = "confirmed"
                        self._confirm_start_ms = timestamp_ms
                        out.state = "confirmed"
                        out.selected_finger = fname
                        out.menu_path = self._current_menu_path()
                        out.menu_label = self._menu_stack[-1][0] if self._menu_stack else None
                        out.slots = self._current_slots_view()
                        return out

                    self._state = "confirmed"
                    self._confirm_start_ms = timestamp_ms
                    self._last_action_ms = timestamp_ms
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
                # Airpoint stay-open behavior: keep menu open if user still holds palm open!
                if open_now:
                    self._state = "active"
                    self._selected_finger = None
                    self._finger_pressed = {f: False for f, _, _, _ in _FINGER_TIP_MAP}
                    out.state = "active"
                    out.selected_finger = None
                else:
                    within_grace = (
                        self.confirmation_grace_ms > 0.0
                        and self._last_open_palm_ms is not None
                        and (timestamp_ms - self._last_open_palm_ms) <= self.confirmation_grace_ms
                    )
                    if not within_grace:
                        self.reset()
                        out.state = "inactive"
                        out.progress = 0.0
                        out.selected_finger = None

        return out
