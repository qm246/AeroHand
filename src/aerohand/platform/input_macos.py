"""
Native macOS Quartz Input Controller + Zero-Permission Session-Tap Typing Guard.

Combines GestureCtl's proven safety foundation with AeroHand upgrades:
  - Posts synthetic CGEvents to `kCGSessionEventTap` so `kCGEventSourceStateHIDSystemState`
    reflects PURE physical keyboard/mouse activity — ZERO Input Monitoring permission needed.
  - Full left/right click, double-click, left/right drag.
  - Upgraded 2D pixel scroll (`wheelCount=2` when `dx != 0` so horizontal scroll works on macOS!).
  - Desktop Spaces (`Ctrl+Left/Right`), Mission Control (`Ctrl+Up`), App Exposé (`Ctrl+Down`),
    Spotlight (`Cmd+Space`), Window Tiling (macOS native / AXUIElement), and `release_all()` failsafe.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple

try:
    import Quartz
except Exception:  # pragma: no cover - allows non-macOS unit testing
    Quartz = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

AEROHAND_EVENT_TAG = 0x4145524F  # ASCII 'AERO'

VK_LEFT_ARROW = 123
VK_RIGHT_ARROW = 124
VK_DOWN_ARROW = 125
VK_UP_ARROW = 126
VK_SPACE = 49
VK_F = 3

DEFAULT_KEYBOARD_BLOCKS = frozenset({"pointer", "click", "drag", "scroll", "swipe", "window_tile"})
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
    _, _, w, h = get_main_display_bounds()
    return w * 0.5, h * 0.5


class MacOSInputController:
    """Low-latency Quartz input injector with session-tap Typing Guard & failsafe release."""

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

        # Always post to Session tap in counter mode -> never pollutes HIDSystemState
        self._tap_location = Quartz.kCGSessionEventTap if Quartz is not None else 0
        self._counter_query_fn = counter_query_fn or (
            Quartz.CGEventSourceSecondsSinceLastEventType if Quartz is not None else None
        )

        self._pressed_mouse_buttons: Set[str] = set()
        self._pressed_keys: Set[int] = set()
        self._last_mouse_pos: Optional[Tuple[float, float]] = None
        self._scroll_in_progress: bool = False
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
    ) -> bool:
        """Post 2D pixel-level scroll event (`wheelCount=2` whenever horizontal delta is non-zero)."""
        if self.is_action_blocked("scroll"):
            return False
        dy_int = int(round(delta_y))
        dx_int = int(round(delta_x))
        if dy_int == 0 and dx_int == 0:
            return True
        if self.dry_run or Quartz is None:
            return True

        now = time.time()
        phase = Quartz.kCGScrollPhaseChanged
        if use_phases:
            if not self._scroll_in_progress or (now - self._last_scroll_time > 0.25):
                phase = Quartz.kCGScrollPhaseBegan
                self._scroll_in_progress = True
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
            Quartz.CGEventSetIntegerValueField(ev, Quartz.kCGScrollWheelEventScrollPhase, phase)
        return self._post(ev)

    def end_scroll_phase(self) -> None:
        if not self._scroll_in_progress:
            return
        self._scroll_in_progress = False
        if self.dry_run or Quartz is None:
            return
        ev = Quartz.CGEventCreateScrollWheelEvent2(None, Quartz.kCGScrollEventUnitPixel, 2, 0, 0, 0)
        if ev is not None:
            Quartz.CGEventSetIntegerValueField(
                ev, Quartz.kCGScrollWheelEventScrollPhase, Quartz.kCGScrollPhaseEnded
            )
            self._post(ev)

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

    def media_play_pause(self) -> bool:
        if self.dry_run:
            return True
        # Non-intrusive fallback using NSEvent media key or AppleScript Music/System
        try:
            import AppKit

            mk_ev = AppKit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_  # noqa: E501
            ev_d = mk_ev(14, (0, 0), 0xA00, 0, 0, None, 8, (16 << 16) | (0xA << 8), -1)
            ev_u = mk_ev(14, (0, 0), 0xB00, 0, 0, None, 8, (16 << 16) | (0xB << 8), -1)
            if ev_d is not None and ev_u is not None:
                self._post(ev_d.CGEvent())
                self._post(ev_u.CGEvent())
                return True
        except Exception:
            pass
        return False

    def tile_focused_window(self, direction: str) -> bool:
        """
        Tile the frontmost window (`left`, `right`, `top`=fill/maximize, `bottom`=center).
        Uses macOS Sequoia native Window Tiling shortcuts (Fn+Ctrl+Arrows / Fn+Ctrl+F)
        with non-blocking fallback.
        """
        if self.is_action_blocked("window_tile"):
            return False
        if self.dry_run or Quartz is None:
            return True
        flags = Quartz.kCGEventFlagMaskControl | Quartz.kCGEventFlagMaskSecondaryFn
        if direction == "left":
            return self.key_combo(VK_LEFT_ARROW, flags=flags | Quartz.kCGEventFlagMaskNumericPad)
        if direction == "right":
            return self.key_combo(VK_RIGHT_ARROW, flags=flags | Quartz.kCGEventFlagMaskNumericPad)
        if direction == "top":
            return self.key_combo(VK_F, flags=flags)
        if direction == "bottom":
            # Minimize/restore via Cmd+M or Escape depending on user preference; keep non-destructive center
            try:
                subprocess.Popen(
                    [
                        "osascript",
                        "-e",
                        'tell application "System Events" to keystroke "c" using {control down}',
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            return True
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
