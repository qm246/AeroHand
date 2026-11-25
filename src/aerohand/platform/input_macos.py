"""
Native macOS Quartz & Accessibility (AXUIElement) Input Controller
with Zero-Permission Session-Tap Typing Guard.

Capabilities:
  - Session-tap `CGEventPost` (`kCGSessionEventTap`) so `HIDSystemState` measures PURE
    physical keyboard/mouse activity — ZERO Input Monitoring permission required.
  - Multi-monitor virtual desktop bounds (`CGGetActiveDisplayList`).
  - Left/Right click, Double-click, Left/Right drag, failsafe `release_all()`.
  - Native 2D pixel scroll with BOTH `ScrollPhase` and `MomentumPhase` (rubber-band Trackpad parity).
  - True `AXUIElement` Window Tiling (`left`, `right`, `top`/`fill`, `bottom`, `center`) with
    optional smooth animation — works across all apps without macOS Sequoia shortcut dependencies.
  - Complete Airpoint Palm Wheel macOS actions: Edit (Copy/Paste/Cut/Undo/Redo), Window
    (Hide/Minimize/Fullscreen/Close), Media/Volume (`NX_KEYTYPE_*`), Siri, Spotlight,
    Mission Control, App Exposé, Emoji/Keyboard viewer, Launch App, Two-Hand Zoom, and
    UTF-8 safe Vietnamese clipboard text insertion.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

try:
    import Quartz
except Exception:  # pragma: no cover
    Quartz = None  # type: ignore[assignment]

try:
    import ApplicationServices as AppServices
except Exception:  # pragma: no cover
    AppServices = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

AEROHAND_EVENT_TAG = 0x4145524F  # ASCII 'AERO'

# macOS Virtual Keycodes (US ANSI)
VK_A = 0
VK_C = 8
VK_F = 3
VK_H = 4
VK_M = 46
VK_V = 9
VK_W = 13
VK_X = 7
VK_Z = 6
VK_EQUAL = 24
VK_MINUS = 27
VK_SPACE = 49
VK_LEFT_ARROW = 123
VK_RIGHT_ARROW = 124
VK_DOWN_ARROW = 125
VK_UP_ARROW = 126

# NX_KEYTYPE media key codes
NX_KEYTYPE_SOUND_UP = 0
NX_KEYTYPE_SOUND_DOWN = 1
NX_KEYTYPE_MUTE = 7
NX_KEYTYPE_PLAY = 16
NX_KEYTYPE_NEXT = 17
NX_KEYTYPE_PREVIOUS = 18

DEFAULT_KEYBOARD_BLOCKS = frozenset({"pointer", "click", "drag", "scroll", "swipe", "window_tile", "zoom"})
DEFAULT_MOUSE_BLOCKS = frozenset({"pointer", "click", "drag", "scroll"})


def get_main_display_bounds() -> Tuple[float, float, float, float]:
    """Return `(x, y, width, height)` of the main display in Retina points."""
    if Quartz is None:
        return 0.0, 0.0, 1920.0, 1080.0
    try:
        rect = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return float(rect.origin.x), float(rect.origin.y), float(rect.size.width), float(rect.size.height)
    except Exception:
        return 0.0, 0.0, 1920.0, 1080.0


def get_virtual_display_bounds(multi_monitor: bool = True) -> Tuple[float, float, float, float]:
    """Return `(min_x, min_y, width, height)` across all active displays (or main only)."""
    if not multi_monitor or Quartz is None:
        return get_main_display_bounds()
    try:
        err, display_ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
        if err != 0 or not count:
            return get_main_display_bounds()
        min_x = 1e9
        min_y = 1e9
        max_x = -1e9
        max_y = -1e9
        for did in display_ids[:count]:
            r = Quartz.CGDisplayBounds(did)
            min_x = min(min_x, float(r.origin.x))
            min_y = min(min_y, float(r.origin.y))
            max_x = max(max_x, float(r.origin.x + r.size.width))
            max_y = max(max_y, float(r.origin.y + r.size.height))
        if max_x > min_x and max_y > min_y:
            return min_x, min_y, max_x - min_x, max_y - min_y
    except Exception:
        pass
    return get_main_display_bounds()


def get_current_cursor_position() -> Tuple[float, float]:
    """Query the live macOS cursor position in Retina points."""
    if Quartz is not None:
        try:
            ev = Quartz.CGEventCreate(None)
            if ev is not None:
                pt = Quartz.CGEventGetLocation(ev)
                return float(pt.x), float(pt.y)
        except Exception:
            pass
    ox, oy, w, h = get_main_display_bounds()
    return ox + w * 0.5, oy + h * 0.5


def get_frontmost_app_bundle_id() -> Optional[str]:
    """Return the macOS bundle identifier of the currently focused application."""
    try:
        import AppKit

        app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is not None:
            bid = app.bundleIdentifier()
            return str(bid) if bid else None
    except Exception:
        pass
    return None


def _visible_frame_quartz_for_point(px: float, py: float) -> Tuple[float, float, float, float]:
    """Find the active screen visible frame in Quartz top-left coordinates for point `(px, py)`."""
    _, _, main_w, main_h = get_main_display_bounds()
    try:
        import AppKit

        screens = list(AppKit.NSScreen.screens() or [])
        chosen = None
        for sc in screens:
            f = sc.frame()
            q_x = float(f.origin.x)
            q_y = main_h - float(f.origin.y + f.size.height)
            q_w = float(f.size.width)
            q_h = float(f.size.height)
            if q_x <= px <= q_x + q_w and q_y <= py <= q_y + q_h:
                chosen = sc
                break
        if chosen is None and screens:
            chosen = AppKit.NSScreen.mainScreen() or screens[0]
        if chosen is not None:
            vf = chosen.visibleFrame()
            vx = float(vf.origin.x)
            vy = main_h - float(vf.origin.y + vf.size.height)
            vw = float(vf.size.width)
            vh = float(vf.size.height)
            return vx, vy, vw, vh
    except Exception:
        pass
    return 0.0, 28.0, main_w, max(200.0, main_h - 28.0)


def compute_tile_rect(
    direction: str,
    visible_rect: Tuple[float, float, float, float],
) -> Optional[Tuple[float, float, float, float]]:
    """Compute target `(x, y, w, h)` in Quartz points for a given tile direction."""
    vx, vy, vw, vh = visible_rect
    pad = 4.0
    if direction == "left":
        return (vx + pad, vy + pad, vw * 0.5 - pad * 1.5, vh - pad * 2.0)
    if direction == "right":
        return (vx + vw * 0.5 + pad * 0.5, vy + pad, vw * 0.5 - pad * 1.5, vh - pad * 2.0)
    if direction in ("top", "fill"):
        return (vx + pad, vy + pad, vw - pad * 2.0, vh - pad * 2.0)
    if direction == "bottom":
        cw = vw * 0.72
        ch = vh * 0.78
        return (vx + (vw - cw) * 0.5, vy + (vh - ch) * 0.5, cw, ch)
    if direction == "center":
        cw = vw * 0.70
        ch = vh * 0.76
        return (vx + (vw - cw) * 0.5, vy + (vh - ch) * 0.5, cw, ch)
    return None


class MacOSInputController:
    """Low-latency Quartz + AXUIElement input injector with session-tap Typing Guard."""

    def __init__(
        self,
        typing_guard_cfg: Optional[Dict[str, Any]] = None,
        *,
        dry_run: bool = False,
        counter_query_fn: Optional[Callable[[int, int], float]] = None,
    ) -> None:
        self.dry_run: bool = bool(dry_run)
        tg = typing_guard_cfg or {}
        self.typing_guard_enabled: bool = bool(tg.get("enabled", True))
        self.keyboard_guard_s: float = float(tg.get("keyboard_ms", 1500)) / 1000.0
        self.mouse_guard_s: float = float(tg.get("mouse_ms", 1000)) / 1000.0
        self.keyboard_blocks: Set[str] = {
            str(g).lower() for g in (tg.get("keyboard_blocks") or DEFAULT_KEYBOARD_BLOCKS)
        }
        self.mouse_blocks: Set[str] = {
            str(g).lower() for g in (tg.get("mouse_blocks") or DEFAULT_MOUSE_BLOCKS)
        }

        self._tap_location = Quartz.kCGSessionEventTap if Quartz is not None else 0
        self._counter_query_fn = counter_query_fn or (
            Quartz.CGEventSourceSecondsSinceLastEventType if Quartz is not None else None
        )

        self._pressed_mouse_buttons: Set[str] = set()
        self._pressed_keys: Set[int] = set()
        self._last_mouse_pos: Optional[Tuple[float, float]] = None
        self._scroll_in_progress: bool = False
        self._momentum_in_progress: bool = False
        self._last_scroll_time: float = 0.0
        self._last_guard_reason: str = ""
        self.suppressed_count: int = 0

    @property
    def last_guard_reason(self) -> str:
        return self._last_guard_reason

    def _query_hid_remaining(self) -> Tuple[float, float]:
        if not self.typing_guard_enabled or self._counter_query_fn is None or Quartz is None:
            return 0.0, 0.0

        sec_key = 9999.0
        for ev_t in (Quartz.kCGEventKeyDown, Quartz.kCGEventKeyUp, Quartz.kCGEventFlagsChanged):
            try:
                sec = float(self._counter_query_fn(Quartz.kCGEventSourceStateHIDSystemState, ev_t))
                if 0.0 <= sec < sec_key:
                    sec_key = sec
            except Exception:
                continue

        sec_mouse = 9999.0
        for ev_t in (
            Quartz.kCGEventMouseMoved,
            Quartz.kCGEventLeftMouseDown,
            Quartz.kCGEventLeftMouseUp,
            Quartz.kCGEventRightMouseDown,
            Quartz.kCGEventLeftMouseDragged,
            Quartz.kCGEventScrollWheel,
        ):
            try:
                sec = float(self._counter_query_fn(Quartz.kCGEventSourceStateHIDSystemState, ev_t))
                if 0.0 <= sec < sec_mouse:
                    sec_mouse = sec
            except Exception:
                continue

        key_rem = max(0.0, self.keyboard_guard_s - sec_key) if sec_key < self.keyboard_guard_s else 0.0
        mouse_rem = max(0.0, self.mouse_guard_s - sec_mouse) if sec_mouse < self.mouse_guard_s else 0.0
        return key_rem, mouse_rem

    def is_action_blocked(self, action_group: str) -> bool:
        group = action_group.lower()
        key_rem, mouse_rem = self._query_hid_remaining()
        if key_rem > 0.0 and group in self.keyboard_blocks:
            self._last_guard_reason = f"KEYBOARD ({key_rem:.1f}s)"
            self.suppressed_count += 1
            return True
        if mouse_rem > 0.0 and group in self.mouse_blocks:
            self._last_guard_reason = f"MOUSE ({mouse_rem:.1f}s)"
            self.suppressed_count += 1
            return True
        self._last_guard_reason = ""
        return False

    def _post(self, event: Any) -> bool:
        if self.dry_run or Quartz is None or event is None:
            return True
        try:
            Quartz.CGEventSetIntegerValueField(event, Quartz.kCGEventSourceUserData, AEROHAND_EVENT_TAG)
            Quartz.CGEventPost(self._tap_location, event)
            return True
        except Exception as exc:
            logger.error(f"[INPUT] CGEventPost failed: {exc}")
            return False

    def mouse_move(
        self,
        x: float,
        y: float,
        delta_x: Optional[float] = None,
        delta_y: Optional[float] = None,
    ) -> bool:
        group = "drag" if self._pressed_mouse_buttons else "pointer"
        if self.is_action_blocked(group):
            return False
        self._last_mouse_pos = (x, y)
        if self.dry_run or Quartz is None:
            return True

        pt = Quartz.CGPoint(x, y)
        if "left" in self._pressed_mouse_buttons:
            ev_type = Quartz.kCGEventLeftMouseDragged
            btn = Quartz.kCGMouseButtonLeft
        elif "right" in self._pressed_mouse_buttons:
            ev_type = Quartz.kCGEventRightMouseDragged
            btn = Quartz.kCGMouseButtonRight
        else:
            ev_type = Quartz.kCGEventMouseMoved
            btn = Quartz.kCGMouseButtonLeft

        ev = Quartz.CGEventCreateMouseEvent(None, ev_type, pt, btn)
        if ev is None:
            return False
        if delta_x is not None:
            Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventDeltaX, int(round(delta_x)))
        if delta_y is not None:
            Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGMouseEventDeltaY, int(round(delta_y)))
        return self._post(ev)

    def mouse_click(
        self,
        x: float,
        y: float,
        button: str = "left",
        click_count: int = 1,
    ) -> bool:
        if self.is_action_blocked("click"):
            return False
        self._last_mouse_pos = (x, y)
        if self.dry_run or Quartz is None:
            return True

        pt = Quartz.CGPoint(x, y)
        is_left = button == "left"
        btn_id = Quartz.kCGMouseButtonLeft if is_left else Quartz.kCGMouseButtonRight
        down_t = Quartz.kCGEventLeftMouseDown if is_left else Quartz.kCGEventRightMouseDown
        up_t = Quartz.kCGEventLeftMouseUp if is_left else Quartz.kCGEventRightMouseUp

        ev_d = Quartz.CGEventCreateMouseEvent(None, down_t, pt, btn_id)
        ev_u = Quartz.CGEventCreateMouseEvent(None, up_t, pt, btn_id)
        if ev_d is None or ev_u is None:
            return False
        if click_count > 1:
            Quartz.CGEventSetIntegerValueField(ev_d, Quartz.kCGMouseEventClickState, int(click_count))
            Quartz.CGEventSetIntegerValueField(ev_u, Quartz.kCGMouseEventClickState, int(click_count))
        ok_d = self._post(ev_d)
        time.sleep(0.012)
        ok_u = self._post(ev_u)
        return ok_d and ok_u

    def mouse_down(self, x: float, y: float, button: str = "left") -> bool:
        if self.is_action_blocked("drag") or self.is_action_blocked("click"):
            return False
        self._last_mouse_pos = (x, y)
        self._pressed_mouse_buttons.add(button)
        if self.dry_run or Quartz is None:
            return True
        pt = Quartz.CGPoint(x, y)
        is_left = button == "left"
        btn_id = Quartz.kCGMouseButtonLeft if is_left else Quartz.kCGMouseButtonRight
        down_t = Quartz.kCGEventLeftMouseDown if is_left else Quartz.kCGEventRightMouseDown
        return self._post(Quartz.CGEventCreateMouseEvent(None, down_t, pt, btn_id))

    def mouse_up(self, x: float, y: float, button: str = "left") -> bool:
        self._last_mouse_pos = (x, y)
        self._pressed_mouse_buttons.discard(button)
        if self.dry_run or Quartz is None:
            return True
        pt = Quartz.CGPoint(x, y)
        is_left = button == "left"
        btn_id = Quartz.kCGMouseButtonLeft if is_left else Quartz.kCGMouseButtonRight
        up_t = Quartz.kCGEventLeftMouseUp if is_left else Quartz.kCGEventRightMouseUp
        return self._post(Quartz.CGEventCreateMouseEvent(None, up_t, pt, btn_id))

    def scroll_pixels(
        self,
        delta_y: float,
        delta_x: float = 0.0,
        use_phases: bool = True,
        is_inertia: bool = False,
    ) -> bool:
        """
        Post 2D pixel-level scroll event with native macOS ScrollPhase + MomentumPhase.
        Matches Apple Trackpad & Airpoint momentum physics.
        """
        if self.is_action_blocked("scroll"):
            return False
        dy_int = int(round(delta_y))
        dx_int = int(round(delta_x))
        if dy_int == 0 and dx_int == 0:
            return True
        if self.dry_run or Quartz is None:
            return True

        now = time.time()
        scroll_phase = 0
        momentum_phase = 0
        phase_field = getattr(Quartz, "kCGScrollWheelEventScrollPhase", 99)
        mom_field = getattr(Quartz, "kCGScrollWheelEventMomentumPhase", 123)

        if use_phases:
            if not is_inertia:
                if self._momentum_in_progress:
                    self._send_phase_marker(0, getattr(Quartz, "kCGMomentumScrollPhaseEnd", 3))
                    self._momentum_in_progress = False
                if not self._scroll_in_progress or (now - self._last_scroll_time > 0.25):
                    scroll_phase = getattr(Quartz, "kCGScrollPhaseBegan", 1)
                    self._scroll_in_progress = True
                else:
                    scroll_phase = getattr(Quartz, "kCGScrollPhaseChanged", 2)
                momentum_phase = getattr(Quartz, "kCGMomentumScrollPhaseNone", 0)
            else:
                if self._scroll_in_progress:
                    self._send_phase_marker(getattr(Quartz, "kCGScrollPhaseEnded", 4), 0)
                    self._scroll_in_progress = False
                if not self._momentum_in_progress:
                    momentum_phase = getattr(Quartz, "kCGMomentumScrollPhaseBegin", 1)
                    self._momentum_in_progress = True
                else:
                    momentum_phase = getattr(Quartz, "kCGMomentumScrollPhaseContinue", 2)
                scroll_phase = 0
            self._last_scroll_time = now

        wheel_count = 2 if dx_int != 0 else 1
        ev = Quartz.CGEventCreateScrollWheelEvent2(
            None,
            Quartz.kCGScrollEventUnitPixel,
            wheel_count,
            dy_int,
            dx_int,
            0,
        )
        if ev is None:
            return False
        if use_phases:
            Quartz.CGEventSetIntegerValueField(ev, phase_field, scroll_phase)
            Quartz.CGEventSetIntegerValueField(ev, mom_field, momentum_phase)
        return self._post(ev)

    def _send_phase_marker(self, scroll_phase: int, momentum_phase: int) -> None:
        if self.dry_run or Quartz is None:
            return
        ev = Quartz.CGEventCreateScrollWheelEvent2(None, Quartz.kCGScrollEventUnitPixel, 2, 0, 0, 0)
        if ev is not None:
            phase_field = getattr(Quartz, "kCGScrollWheelEventScrollPhase", 99)
            mom_field = getattr(Quartz, "kCGScrollWheelEventMomentumPhase", 123)
            Quartz.CGEventSetIntegerValueField(ev, phase_field, scroll_phase)
            Quartz.CGEventSetIntegerValueField(ev, mom_field, momentum_phase)
            self._post(ev)

    def end_scroll_phase(self) -> None:
        if self._scroll_in_progress:
            self._scroll_in_progress = False
            self._send_phase_marker(getattr(Quartz, "kCGScrollPhaseEnded", 4), 0)
        if self._momentum_in_progress:
            self._momentum_in_progress = False
            self._send_phase_marker(0, getattr(Quartz, "kCGMomentumScrollPhaseEnd", 3))

    def key_combo(self, keycode: int, flags: int = 0, hold_s: float = 0.025) -> bool:
        if self.dry_run or Quartz is None:
            return True
        ev_d = Quartz.CGEventCreateKeyboardEvent(None, keycode, True)
        ev_u = Quartz.CGEventCreateKeyboardEvent(None, keycode, False)
        if ev_d is None or ev_u is None:
            return False
        if flags:
            Quartz.CGEventSetFlags(ev_d, flags)
            Quartz.CGEventSetFlags(ev_u, flags)
        self._pressed_keys.add(keycode)
        ok_d = self._post(ev_d)
        if hold_s > 0:
            time.sleep(hold_s)
        ok_u = self._post(ev_u)
        self._pressed_keys.discard(keycode)
        return ok_d and ok_u

    # ------------------------------------------------------------------
    # Navigation, System, Edit, Window & Media Shortcuts (Airpoint Parity)
    # ------------------------------------------------------------------

    def switch_space(self, direction: str) -> bool:
        if self.is_action_blocked("swipe") or Quartz is None:
            return False
        kc = VK_LEFT_ARROW if direction == "left" else VK_RIGHT_ARROW
        flags = (
            Quartz.kCGEventFlagMaskControl
            | Quartz.kCGEventFlagMaskSecondaryFn
            | Quartz.kCGEventFlagMaskNumericPad
        )
        return self.key_combo(kc, flags=flags)

    def toggle_mission_control(self) -> bool:
        if self.is_action_blocked("swipe") or Quartz is None:
            return False
        flags = (
            Quartz.kCGEventFlagMaskControl
            | Quartz.kCGEventFlagMaskSecondaryFn
            | Quartz.kCGEventFlagMaskNumericPad
        )
        return self.key_combo(VK_UP_ARROW, flags=flags)

    def exit_mission_control(self) -> bool:
        if self.is_action_blocked("swipe") or Quartz is None:
            return False
        flags = (
            Quartz.kCGEventFlagMaskControl
            | Quartz.kCGEventFlagMaskSecondaryFn
            | Quartz.kCGEventFlagMaskNumericPad
        )
        return self.key_combo(VK_DOWN_ARROW, flags=flags)

    def open_spotlight(self) -> bool:
        if self.is_action_blocked("swipe") or Quartz is None:
            return False
        return self.key_combo(VK_SPACE, flags=Quartz.kCGEventFlagMaskCommand)

    def siri_listen(self) -> bool:
        if self.dry_run:
            return True
        try:
            subprocess.Popen(
                ["open", "/System/Applications/Siri.app"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def open_app(self, app_path_or_name: str) -> bool:
        if self.dry_run or not app_path_or_name:
            return True
        try:
            cmd = ["open", app_path_or_name] if app_path_or_name.startswith("/") else ["open", "-a", app_path_or_name]
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

    def keyboard_emoji(self) -> bool:
        if Quartz is None:
            return False
        flags = Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskCommand
        return self.key_combo(VK_SPACE, flags=flags)

    def zoom_step(self, step: int) -> bool:
        if step == 0 or self.is_action_blocked("zoom") or Quartz is None:
            return False
        kc = VK_EQUAL if step > 0 else VK_MINUS
        return self.key_combo(kc, flags=Quartz.kCGEventFlagMaskCommand)

    def edit_copy(self) -> bool:
        return Quartz is not None and self.key_combo(VK_C, flags=Quartz.kCGEventFlagMaskCommand)

    def edit_paste(self) -> bool:
        return Quartz is not None and self.key_combo(VK_V, flags=Quartz.kCGEventFlagMaskCommand)

    def edit_cut(self) -> bool:
        return Quartz is not None and self.key_combo(VK_X, flags=Quartz.kCGEventFlagMaskCommand)

    def edit_undo(self) -> bool:
        return Quartz is not None and self.key_combo(VK_Z, flags=Quartz.kCGEventFlagMaskCommand)

    def edit_redo(self) -> bool:
        if Quartz is None:
            return False
        return self.key_combo(VK_Z, flags=Quartz.kCGEventFlagMaskCommand | Quartz.kCGEventFlagMaskShift)

    def window_hide(self) -> bool:
        return Quartz is not None and self.key_combo(VK_H, flags=Quartz.kCGEventFlagMaskCommand)

    def window_minimize(self) -> bool:
        return Quartz is not None and self.key_combo(VK_M, flags=Quartz.kCGEventFlagMaskCommand)

    def window_close(self) -> bool:
        return Quartz is not None and self.key_combo(VK_W, flags=Quartz.kCGEventFlagMaskCommand)

    def window_fullscreen(self) -> bool:
        if Quartz is None:
            return False
        return self.key_combo(VK_F, flags=Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskCommand)

    def post_media_key(self, nx_key: int) -> bool:
        if self.dry_run:
            return True
        try:
            import AppKit

            mk_ev = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_  # noqa: E501
            ev_d = mk_ev(14, (0, 0), 0xA00, 0, 0, None, 8, (nx_key << 16) | (0xA << 8), -1)
            ev_u = mk_ev(14, (0, 0), 0xB00, 0, 0, None, 8, (nx_key << 16) | (0xB << 8), -1)
            if ev_d is not None and ev_u is not None:
                self._post(ev_d.CGEvent())
                self._post(ev_u.CGEvent())
                return True
        except Exception:
            pass
        return False

    def media_play_pause(self) -> bool:
        return self.post_media_key(NX_KEYTYPE_PLAY)

    def media_next(self) -> bool:
        return self.post_media_key(NX_KEYTYPE_NEXT)

    def media_prev(self) -> bool:
        return self.post_media_key(NX_KEYTYPE_PREVIOUS)

    def volume_up(self) -> bool:
        return self.post_media_key(NX_KEYTYPE_SOUND_UP)

    def volume_down(self) -> bool:
        return self.post_media_key(NX_KEYTYPE_SOUND_DOWN)

    def volume_mute(self) -> bool:
        return self.post_media_key(NX_KEYTYPE_MUTE)

    def paste_text_utf8(self, text: str) -> bool:
        """Insert arbitrary UTF-8 text (full Vietnamese diacritics safe) preserving clipboard."""
        if not text:
            return False
        if self.dry_run:
            return True
        try:
            import AppKit

            pb = AppKit.NSPasteboard.generalPasteboard()
            prev = pb.stringForType_(AppKit.NSPasteboardTypeString)
            pb.clearContents()
            pb.setString_forType_(text, AppKit.NSPasteboardTypeString)
            ok = self.edit_paste()

            def _restore(saved: Optional[str]) -> None:
                time.sleep(0.15)
                try:
                    pb2 = AppKit.NSPasteboard.generalPasteboard()
                    if pb2.stringForType_(AppKit.NSPasteboardTypeString) == text:
                        pb2.clearContents()
                        if saved is not None:
                            pb2.setString_forType_(saved, AppKit.NSPasteboardTypeString)
                except Exception:
                    pass

            threading.Thread(target=_restore, args=(prev,), daemon=True).start()
            return ok
        except Exception as exc:
            logger.warning(f"[INPUT] paste_text_utf8 failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # True AXUIElement Window Tiling (Zero shortcut dependency, multi-monitor aware)
    # ------------------------------------------------------------------

    @staticmethod
    def _ax_get_attr(elem: Any, attr: str) -> Any:
        if AppServices is None:
            return None
        try:
            err, val = AppServices.AXUIElementCopyAttributeValue(elem, attr, None)
            if err == 0:
                return val
        except Exception:
            pass
        return None

    @staticmethod
    def _ax_set_rect(win: Any, x: float, y: float, w: float, h: float) -> bool:
        if AppServices is None or Quartz is None:
            return False
        try:
            pos_val = AppServices.AXValueCreate(AppServices.kAXValueCGPointType, Quartz.CGPoint(x, y))
            size_val = AppServices.AXValueCreate(AppServices.kAXValueCGSizeType, Quartz.CGSize(w, h))
            if pos_val is None or size_val is None:
                return False
            e1 = AppServices.AXUIElementSetAttributeValue(win, AppServices.kAXPositionAttribute, pos_val)
            e2 = AppServices.AXUIElementSetAttributeValue(win, AppServices.kAXSizeAttribute, size_val)
            # Re-apply position in case window size constraint shifted origin across monitors
            AppServices.AXUIElementSetAttributeValue(win, AppServices.kAXPositionAttribute, pos_val)
            return (e1 == 0) or (e2 == 0)
        except Exception:
            return False

    def _tile_via_ax(self, direction: str, animate: bool = True) -> bool:
        if AppServices is None or Quartz is None:
            return False
        try:
            sys_wide = AppServices.AXUIElementCreateSystemWide()
            app = self._ax_get_attr(sys_wide, AppServices.kAXFocusedApplicationAttribute)
            if app is None:
                return False
            win = self._ax_get_attr(app, AppServices.kAXFocusedWindowAttribute)
            if win is None:
                return False

            cur_pos_val = self._ax_get_attr(win, AppServices.kAXPositionAttribute)
            cur_size_val = self._ax_get_attr(win, AppServices.kAXSizeAttribute)
            cx, cy = get_current_cursor_position()
            start_rect: Optional[Tuple[float, float, float, float]] = None
            if cur_pos_val is not None and cur_size_val is not None:
                ok_p, pt = AppServices.AXValueGetValue(cur_pos_val, AppServices.kAXValueCGPointType, None)
                ok_s, sz = AppServices.AXValueGetValue(cur_size_val, AppServices.kAXValueCGSizeType, None)
                if ok_p and ok_s:
                    start_rect = (float(pt.x), float(pt.y), float(sz.width), float(sz.height))
                    cx = start_rect[0] + start_rect[2] * 0.5
                    cy = start_rect[1] + start_rect[3] * 0.5

            vis = _visible_frame_quartz_for_point(cx, cy)
            target = compute_tile_rect(direction, vis)
            if target is None:
                return False

            if animate and start_rect is not None:
                def _anim() -> None:
                    steps = 4
                    sx, sy, sw, sh = start_rect
                    tx, ty, tw, th = target
                    for i in range(1, steps + 1):
                        t = i / steps
                        ease = 1.0 - (1.0 - t) ** 2
                        self._ax_set_rect(
                            win,
                            sx + (tx - sx) * ease,
                            sy + (ty - sy) * ease,
                            sw + (tw - sw) * ease,
                            sh + (th - sh) * ease,
                        )
                        if i < steps:
                            time.sleep(0.012)

                threading.Thread(target=_anim, daemon=True).start()
                return True

            return self._ax_set_rect(win, *target)
        except Exception as exc:
            logger.debug(f"[AX TILE] Fallback triggered: {exc}")
            return False

    def tile_focused_window(self, direction: str, animate: bool = True) -> bool:
        """
        Tile the frontmost window (`left`, `right`, `top`/`fill`, `bottom`, `center`)
        using native Accessibility API (`AXUIElement`) with safe Sequoia shortcut fallback.
        """
        if self.is_action_blocked("window_tile"):
            return False
        if self.dry_run or Quartz is None:
            return True

        if self._tile_via_ax(direction, animate=animate):
            return True

        # Safe fallback (macOS Sequoia native tiling shortcuts — NEVER Ctrl+C!)
        flags = Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskSecondaryFn
        if direction == "left":
            return self.key_combo(VK_LEFT_ARROW, flags=flags | Quartz.kCGEventFlagMaskNumericPad)
        if direction == "right":
            return self.key_combo(VK_RIGHT_ARROW, flags=flags | Quartz.kCGEventFlagMaskNumericPad)
        if direction in ("top", "fill"):
            return self.key_combo(VK_F, flags=flags)
        return False

    def release_all(self) -> None:
        """Failsafe cleanup: guarantee zero stuck mouse buttons, keys, or scroll phases."""
        pos = self._last_mouse_pos or get_current_cursor_position()
        for btn in list(self._pressed_mouse_buttons):
            self.mouse_up(pos[0], pos[1], button=btn)
        self._pressed_mouse_buttons.clear()
        if Quartz is not None and not self.dry_run:
            for kc in list(self._pressed_keys):
                try:
                    ev = Quartz.CGEventCreateKeyboardEvent(None, kc, False)
                    if ev is not None:
                        self._post(ev)
                except Exception:
                    pass
        self._pressed_keys.clear()
        self.end_scroll_phase()
