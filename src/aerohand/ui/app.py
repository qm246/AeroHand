"""
Native macOS Desktop Application for AeroHand v0.2.0
(WKWebView Dark Glass Control Center + Transparent Click-Through Screen Overlays
+ Hover-Fade Floating Camera HUD + Global Hotkeys + URL Scheme + Bilingual vi/en).
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import AppKit
import cv2
import Foundation
import numpy as np
import objc
import WebKit

from aerohand import __version__
from aerohand.config import (
    apply_gesture_preset,
    default_config_path,
    load_config,
    save_user_overrides,
    set_nested_key,
)
from aerohand.i18n import get_lang_pack
from aerohand.platform.control_api import (
    handle_aerohand_url,
    is_launch_at_login_enabled,
    set_launch_at_login,
)
from aerohand.platform.input_macos import ScreenGeometry
from aerohand.platform.permissions import is_accessibility_trusted
from aerohand.runtime import AeroHandRuntime, RuntimeSnapshot

logger = logging.getLogger("aerohand.app")


def _app_support_dir() -> Path:
    d = Path.home() / "Library" / "Application Support" / "AeroHand"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def _app_logs_dir() -> Path:
    d = Path.home() / "Library" / "Logs" / "AeroHand"
    d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def _lock_path() -> Path:
    return _app_support_dir() / "aerohand.lock"


def _show_trigger_path() -> Path:
    return _app_support_dir() / "show_window.trigger"


def _log_path() -> Path:
    return _app_logs_dir() / "aerohand.log"


_LOCK_FD = None


class DebouncedConfigPersister:
    """Coalesces rapid slider updates into a single trailing disk write (default 500 ms)."""

    def __init__(self, save_fn: Callable[[Dict[str, Any]], Any], delay_s: float = 0.5) -> None:
        self._save_fn = save_fn
        self._delay_s = max(0.01, float(delay_s))
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._pending_cfg: Optional[Dict[str, Any]] = None

    def schedule(self, cfg: Dict[str, Any]) -> None:
        snap = deepcopy(cfg)
        with self._lock:
            self._pending_cfg = snap
            if self._timer is not None:
                self._timer.cancel()
            t = threading.Timer(self._delay_s, self._run)
            t.daemon = True
            self._timer = t
            t.start()

    def _run(self) -> None:
        with self._lock:
            cfg = self._pending_cfg
            self._pending_cfg = None
            self._timer = None
        if cfg is not None:
            try:
                self._save_fn(cfg)
            except Exception as exc:
                logger.debug(f"[UI] Debounced override save failed: {exc}")

    def flush(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            cfg = self._pending_cfg
            self._pending_cfg = None
        if cfg is not None:
            try:
                self._save_fn(cfg)
            except Exception as exc:
                logger.debug(f"[UI] Flush override save failed: {exc}")


_MOD_NAME_MAP = {
    "cmd": "cmd",
    "command": "cmd",
    "meta": "cmd",
    "shift": "shift",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "opt",
    "opt": "opt",
    "option": "opt",
}


def parse_hotkey_spec(spec: str) -> Optional[Tuple[frozenset[str], str]]:
    """Parse `'cmd+shift+a'` into `(frozenset({'cmd', 'shift'}), 'a')`."""
    raw = str(spec or "").strip().lower()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split("+") if p.strip()]
    if len(parts) < 2:
        return None
    mods = set()
    key = ""
    for p in parts:
        mapped = _MOD_NAME_MAP.get(p)
        if mapped is not None:
            mods.add(mapped)
        elif not key and len(p) == 1 and p.isalnum():
            key = p
        else:
            return None
    if not mods or not key:
        return None
    return frozenset(mods), key


def detect_hotkey_conflicts(hk_cfg: Dict[str, Any]) -> List[str]:
    """Return list of action names that collide on the exact same normalized hotkey."""
    seen: Dict[Tuple[frozenset[str], str], str] = {}
    conflicts: List[str] = []
    for action in ("toggle_pause", "recenter", "toggle_hud", "mark_not_intended"):
        parsed = parse_hotkey_spec(str(hk_cfg.get(action, "")))
        if parsed is None:
            continue
        if parsed in seen:
            if seen[parsed] not in conflicts:
                conflicts.append(seen[parsed])
            conflicts.append(action)
        else:
            seen[parsed] = action
    return conflicts


def screen_frames_signature(
    frames: Sequence[Tuple[float, float, float, float]],
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Stable geometry signature of display frames so Dock/Space animations do not rebuild overlays."""
    return tuple((int(round(x)), int(round(y)), int(round(w)), int(round(h))) for x, y, w, h in frames)


class OverlayPayloadBuilder:
    """
    Incremental overlay payload builder:
    - Sends static `i18n` pack ONLY on initial sync or when language changes.
    - Sends ONLY changed `ui` keys per tick (empty dict omitted) to minimize JS bridge JSON overhead.
    """

    def __init__(self) -> None:
        self._last_lang: Optional[str] = None
        self._last_ui: Dict[str, Any] = {}

    def reset(self) -> None:
        self._last_lang = None
        self._last_ui = {}

    def build(
        self,
        snap_payload: Dict[str, Any],
        ui_cfg: Dict[str, Any],
        lang: str,
        i18n_pack: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"snap": snap_payload}
        if lang != self._last_lang:
            payload["i18n"] = i18n_pack
            payload["ui"] = deepcopy(ui_cfg)
            self._last_lang = lang
            self._last_ui = deepcopy(ui_cfg)
            return payload

        ui_delta: Dict[str, Any] = {}
        for k, v in ui_cfg.items():
            if self._last_ui.get(k) != v:
                ui_delta[k] = deepcopy(v)
        if ui_delta:
            payload["ui_delta"] = ui_delta
            self._last_ui = deepcopy(ui_cfg)
        return payload


_HUD_SIZES = {
    "small": (220, 154),
    "medium": (300, 210),
    "large": (390, 273),
}


def compute_camera_hud_rect(
    hud_cfg: Dict[str, Any],
    visible_frames: Sequence[Tuple[float, float, float, float]],
    main_visible_frame: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Compute the Cocoa `(x, y, w, h)` rect for the Floating Camera HUD.
    - If `custom_x` and `custom_y` are saved (user dragged the HUD), keep that position
      and clamp it onto the nearest connected display's `visibleFrame`.
    - Otherwise place it at the configured corner preset on `main_visible_frame`.
    """
    size_key = str(hud_cfg.get("size", "medium"))
    pw, ph = _HUD_SIZES.get(size_key, (300, 210))
    raw_cx = hud_cfg.get("custom_x")
    raw_cy = hud_cfg.get("custom_y")
    if (
        raw_cx is not None
        and raw_cy is not None
        and not isinstance(raw_cx, bool)
        and not isinstance(raw_cy, bool)
    ):
        try:
            cx = float(raw_cx)
            cy = float(raw_cy)
        except (TypeError, ValueError):
            cx = cy = None  # type: ignore[assignment]
        if cx is not None and cy is not None:
            frames = list(visible_frames) if visible_frames else [main_visible_frame]
            mid_x = cx + pw * 0.5
            mid_y = cy + ph * 0.5
            best_sf = frames[0]
            best_d2 = float("inf")
            for sf in frames:
                sx, sy, sw, sh = sf
                clx = min(max(mid_x, sx), sx + sw)
                cly = min(max(mid_y, sy), sy + sh)
                d2 = (mid_x - clx) ** 2 + (mid_y - cly) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_sf = sf
            sx, sy, sw, sh = best_sf
            x = min(max(cx, sx), max(sx, sx + sw - pw))
            y = min(max(cy, sy), max(sy, sy + sh - ph))
            return (float(x), float(y), float(pw), float(ph))

    pos = str(hud_cfg.get("position", "bottom-right"))
    sx, sy, sw, sh = main_visible_frame
    pad = 22.0
    x = sx + pad if "left" in pos else (sx + sw - pw - pad)
    y = sy + pad if "bottom" in pos else (sy + sh - ph - pad)
    return (float(x), float(y), float(pw), float(ph))


def _rect_contains_pt(rect: Tuple[float, float, float, float], x: float, y: float) -> bool:
    rx, ry, rw, rh = rect
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


def resolve_active_overlay_index(
    quartz_rects: Sequence[Tuple[float, float, float, float]],
    cursor_x: float,
    cursor_y: float,
    hands_count: int,
    fallback_point: Optional[Tuple[float, float]] = None,
) -> int:
    """Resolve which display index should receive cursor-bound hints/toasts."""
    if not quartz_rects:
        return 0
    if hands_count > 0:
        for idx, rect in enumerate(quartz_rects):
            if _rect_contains_pt(rect, cursor_x, cursor_y):
                return idx
    if fallback_point is not None:
        fx, fy = fallback_point
        for idx, rect in enumerate(quartz_rects):
            if _rect_contains_pt(rect, fx, fy):
                return idx
    for idx, rect in enumerate(quartz_rects):
        if _rect_contains_pt(rect, cursor_x, cursor_y):
            return idx
    return 0


def build_overlay_snap_payloads(
    quartz_rects: Sequence[Tuple[float, float, float, float]],
    snap: RuntimeSnapshot,
    has_collector_overlay: bool,
    compact_collector: Optional[Dict[str, Any]],
    *,
    fallback_point: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Build per-display overlay snapshot payloads.
    - Pointer/click/tile/palm-wheel visuals route to the display containing their Quartz coordinates.
    - When ML Studio recording is active (`has_collector_overlay`), `collector_guide` is mirrored
      onto EVERY display so the floating teleprompter is always visible on multi-monitor setups
      even when no hand is currently in front of the camera.
    """
    if not quartz_rects:
        return []
    active_idx = resolve_active_overlay_index(
        quartz_rects,
        float(snap.cursor_x),
        float(snap.cursor_y),
        int(snap.hands_count),
        fallback_point=fallback_point,
    )
    single_display = len(quartz_rects) == 1
    out: List[Dict[str, Any]] = []
    for idx, q_rect in enumerate(quartz_rects):
        qx, qy, _qw, _qh = q_rect
        is_active_panel = single_display or (idx == active_idx)

        local_click = None
        if snap.click_event is not None and (
            single_display
            or _rect_contains_pt(q_rect, float(snap.click_event["x"]), float(snap.click_event["y"]))
        ):
            local_click = dict(snap.click_event)
            local_click["x"] = round(float(snap.click_event["x"]) - qx, 1)
            local_click["y"] = round(float(snap.click_event["y"]) - qy, 1)

        local_tile = None
        if snap.window_tile_preview is not None:
            tcx = float(snap.window_tile_preview["x"]) + float(snap.window_tile_preview["w"]) * 0.5
            tcy = float(snap.window_tile_preview["y"]) + float(snap.window_tile_preview["h"]) * 0.5
            if single_display or _rect_contains_pt(q_rect, tcx, tcy):
                local_tile = dict(snap.window_tile_preview)
                local_tile["x"] = round(float(snap.window_tile_preview["x"]) - qx, 1)
                local_tile["y"] = round(float(snap.window_tile_preview["y"]) - qy, 1)

        local_wheels = []
        for pw in snap.palm_wheels:
            px_w = float(pw.get("cursor_x", snap.cursor_x))
            py_w = float(pw.get("cursor_y", snap.cursor_y))
            if (
                single_display
                or _rect_contains_pt(q_rect, px_w, py_w)
                or (
                    is_active_panel
                    and not any(_rect_contains_pt(r2, px_w, py_w) for r2 in quartz_rects)
                )
            ):
                pw_local = dict(pw)
                pw_local["cursor_x"] = round(px_w - qx, 1)
                pw_local["cursor_y"] = round(py_w - qy, 1)
                local_wheels.append(pw_local)

        out.append(
            {
                "click_event_seq": snap.click_event_seq if local_click is not None else 0,
                "click_event": local_click,
                "palm_wheels": local_wheels,
                "window_tile_preview": local_tile,
                "active_hint": snap.active_hint if is_active_panel else None,
                "active_hint_text": snap.active_hint_text if is_active_panel else None,
                "warning": snap.warning if is_active_panel else None,
                "dictation": snap.dictation if is_active_panel else {},
                "collector_guide": compact_collector if has_collector_overlay else None,
            }
        )
    return out


def _acquire_single_instance() -> bool:
    global _LOCK_FD
    lock_fp = _lock_path()
    try:
        fd = os.open(str(lock_fp), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.chmod(lock_fp, 0o600)
        except OSError:
            pass
        _LOCK_FD = os.fdopen(fd, "w+", encoding="utf-8")
        fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _LOCK_FD.seek(0)
        _LOCK_FD.truncate()
        _LOCK_FD.write(str(os.getpid()))
        _LOCK_FD.flush()
        atexit.register(_release_lock)
        return True
    except OSError:
        trig = _show_trigger_path()
        try:
            tfd = os.open(str(trig), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(tfd, b"show")
            finally:
                os.close(tfd)
            os.chmod(trig, 0o600)
        except OSError:
            pass
        return False


def _release_lock() -> None:
    global _LOCK_FD
    lock_fp = _lock_path()
    try:
        if _LOCK_FD is not None:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
            _LOCK_FD.close()
            _LOCK_FD = None
        if lock_fp.exists():
            lock_fp.unlink(missing_ok=True)
    except OSError:
        pass


def _bgr_to_nsimage(frame_bgr: np.ndarray, target_w: int, target_h: int) -> Optional[AppKit.NSImage]:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    resized = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
    h, w, _ = rgba.shape
    raw_bytes = rgba.tobytes()
    rep = AppKit.NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(  # noqa: E501
        None, w, h, 8, 4, True, False, AppKit.NSCalibratedRGBColorSpace, w * 4, 32
    )
    if rep is None:
        return None
    buf = rep.bitmapData()
    buf[: len(raw_bytes)] = raw_bytes
    img = AppKit.NSImage.alloc().initWithSize_(Foundation.NSMakeSize(w, h))
    img.addRepresentation_(rep)
    return img


def _open_privacy_pane(pane: str) -> None:
    subprocess.Popen(
        ["open", f"x-apple.systempreferences:com.apple.preference.security?{pane}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _to_py(obj: Any) -> Any:
    """Recursively convert Foundation NS* bridged collections into pure Python primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict) or hasattr(obj, "keys"):
        return {str(k): _to_py(obj[k]) for k in obj.keys()}
    if isinstance(obj, (list, tuple)) or hasattr(obj, "__iter__"):
        return [_to_py(x) for x in obj]
    return str(obj)


class WindowDragView(AppKit.NSView):
    """Transparent native drag region overlaid on full-bleed WKWebView titlebars."""

    def acceptsFirstMouse_(self, _event) -> bool:
        return True

    def mouseDownCanMoveWindow(self) -> bool:
        return True

    def mouseDown_(self, event) -> None:
        win = self.window()
        if win is not None:
            win.performWindowDragWithEvent_(event)


class NonInteractiveImageView(AppKit.NSImageView):
    """NSImageView that passes mouse events through to its parent HoverCameraView for dragging."""

    def hitTest_(self, _point):
        return None


class HoverCameraView(AppKit.NSView):
    """NSView hosting the Floating Camera HUD with smooth hover opacity and free mouse dragging."""

    def initWithFrame_idleAlpha_hoverAlpha_onMoved_(
        self,
        frame,
        idle_alpha: float,
        hover_alpha: float,
        on_moved: Optional[Callable[[float, float], None]] = None,
    ):
        self = objc.super(HoverCameraView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._idle_alpha = float(idle_alpha)
        self._hover_alpha = float(hover_alpha)
        self._on_moved = on_moved
        self._tracking_area = None
        self._hovered = False
        return self

    def acceptsFirstMouse_(self, _event) -> bool:
        return True

    def mouseDownCanMoveWindow(self) -> bool:
        return True

    def resetCursorRects(self) -> None:
        objc.super(HoverCameraView, self).resetCursorRects()
        self.addCursorRect_cursor_(self.bounds(), AppKit.NSCursor.openHandCursor())

    def mouseDown_(self, event) -> None:
        win = self.window()
        if win is None:
            return
        AppKit.NSCursor.closedHandCursor().push()
        try:
            win.performWindowDragWithEvent_(event)
        finally:
            AppKit.NSCursor.pop()
        if self._on_moved is not None:
            try:
                f = win.frame()
                self._on_moved(float(f.origin.x), float(f.origin.y))
            except Exception as exc:
                logger.debug(f"[HUD] Failed persisting dragged HUD origin: {exc}")

    def setAlphas_(self, pair) -> None:
        self._idle_alpha = float(pair[0])
        self._hover_alpha = float(pair[1])
        win = self.window()
        if win is not None:
            win.setAlphaValue_(self._hover_alpha if self._hovered else self._idle_alpha)

    def updateTrackingAreas(self) -> None:
        objc.super(HoverCameraView, self).updateTrackingAreas()
        if self._tracking_area is not None:
            self.removeTrackingArea_(self._tracking_area)
        opts = (
            AppKit.NSTrackingMouseEnteredAndExited
            | AppKit.NSTrackingActiveAlways
            | AppKit.NSTrackingInVisibleRect
        )
        self._tracking_area = AppKit.NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
            self.bounds(), opts, self, None
        )
        self.addTrackingArea_(self._tracking_area)

    def mouseEntered_(self, _event) -> None:
        self._hovered = True
        win = self.window()
        if win is not None:
            win.animator().setAlphaValue_(self._hover_alpha)

    def mouseExited_(self, _event) -> None:
        self._hovered = False
        win = self.window()
        if win is not None:
            win.animator().setAlphaValue_(self._idle_alpha)


class WebBridgeHandler(AppKit.NSObject):
    """WKScriptMessageHandler receiving JS messages from the Control Center WKWebView."""

    def initWithCallback_(self, callback: Callable[[Dict[str, Any]], None]):
        self = objc.super(WebBridgeHandler, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def userContentController_didReceiveScriptMessage_(self, _controller, message) -> None:
        with objc.autorelease_pool():
            try:
                body = _to_py(message.body())
                if isinstance(body, dict):
                    self._callback(body)
            except Exception as exc:
                logger.exception(f"[BRIDGE] Failed handling JS message: {exc}")


class OverlayNavigationDelegate(AppKit.NSObject):
    """WKNavigationDelegate that marks overlay WebViews ready and auto-reloads if WebContent crashes."""

    def initWithTerminateCallback_finishCallback_(
        self,
        on_terminated: Callable[[Any], None],
        on_finished: Optional[Callable[[Any], None]] = None,
    ):
        self = objc.super(OverlayNavigationDelegate, self).init()
        if self is None:
            return None
        self._on_terminated = on_terminated
        self._on_finished = on_finished
        return self

    def webView_didFinishNavigation_(self, webview, _navigation) -> None:
        if self._on_finished is not None:
            try:
                self._on_finished(webview)
            except Exception as exc:
                logger.warning(f"[OVERLAY] Navigation finish callback failed: {exc}")

    def webViewWebContentProcessDidTerminate_(self, webview) -> None:
        try:
            self._on_terminated(webview)
        except Exception as exc:
            logger.warning(f"[OVERLAY] WebContent recovery callback failed: {exc}")


class AeroHandAppDelegate(AppKit.NSObject):
    """Main-thread AppKit controller for Menu Bar, WKWebView Control Center, Overlay & HUD."""

    def initWithConfig_(self, config: Dict[str, Any]):
        self = objc.super(AeroHandAppDelegate, self).init()
        if self is None:
            return None
        self.config: Dict[str, Any] = config
        self.runtime: Optional[AeroHandRuntime] = None
        self.status_item = None
        self.window = None
        self.control_webview = None
        self.overlay_panel = None
        self.overlay_webview = None
        self._overlay_entries: list[Dict[str, Any]] = []
        self._overlay_bridge_handler = None
        self._overlay_nav_delegate = None
        self._overlay_screen_signature: Tuple[Tuple[int, int, int, int], ...] = ()
        self._overlay_had_visuals: bool = False
        self._overlay_collector_was_active: bool = False
        self._screen_geom = ScreenGeometry()
        self.camera_panel = None
        self.camera_container = None
        self.camera_image_view = None
        self.ui_timer = None
        self._bridge_handler = None
        self._menu_items: Dict[str, Any] = {}
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._control_ready: bool = False
        self._overlay_ready: bool = False
        self._tick_count: int = 0
        self._global_hotkey_mon = None
        self._local_hotkey_mon = None
        self._hotkeys_local_only: bool = False
        self._hud_size_px = (300, 210)
        self._persister = DebouncedConfigPersister(save_user_overrides, delay_s=0.5)
        self._last_click_dispatched_seq: int = 0
        return self

    def _enqueue_ui(self, fn: Callable[[], None]) -> None:
        self._ui_queue.put(fn)

    def applicationDidFinishLaunching_(self, _notification) -> None:
        AppKit.NSApp.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
        icon_fp = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "AppIcon.png"
        if icon_fp.is_file():
            try:
                app_icon = AppKit.NSImage.alloc().initWithContentsOfFile_(str(icon_fp))
                if app_icon is not None:
                    AppKit.NSApp.setApplicationIconImage_(app_icon)
            except Exception:
                pass

        acc_ok = is_accessibility_trusted(prompt_user=True)
        start_dry_run = not acc_ok

        self.runtime = AeroHandRuntime(
            self.config,
            dry_run=start_dry_run,
            preview=False,
            render_hud_frames=True,
            on_show_window=lambda: self._enqueue_ui(lambda: self.showControlCenter_(None)),
            on_toggle_camera_hud=lambda: self._enqueue_ui(lambda: self.toggleCameraHud_(None)),
            on_config_updated=lambda cfg: self._enqueue_ui(lambda: self._sync_external_config(cfg)),
        )

        self._build_status_bar()
        self._build_control_center()
        self._build_screen_overlay()
        self._build_floating_camera_panel()
        self._register_hotkeys()
        Foundation.NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self,
            "onScreenParametersChanged:",
            AppKit.NSApplicationDidChangeScreenParametersNotification,
            None,
        )
        self.runtime.start_background()

        hud_vis = bool(((self.config.get("ui") or {}).get("camera_hud") or {}).get("visible", True))
        if hud_vis and self.camera_panel is not None:
            self.camera_panel.orderFront_(None)
        for entry in self._overlay_entries:
            entry["panel"].orderFront_(None)
        if self.window is not None:
            self.window.makeKeyAndOrderFront_(None)
        AppKit.NSApp.activateIgnoringOtherApps_(True)

        # 20 Hz UI & Overlay loop
        self.ui_timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.05, self, "onUiTick:", None, True
        )

    def applicationShouldHandleReopen_hasVisibleWindows_(self, _sender, _flag) -> bool:
        self.showControlCenter_(None)
        return True

    def application_openURLs_(self, _app, urls) -> None:
        for u in list(urls or []):
            url_str = str(u.absoluteString() if hasattr(u, "absoluteString") else u)
            handle_aerohand_url(
                url_str,
                on_show=lambda: self.showControlCenter_(None),
                on_tracking=lambda en: self.runtime.set_paused(not en) if self.runtime else None,
                on_toggle_tracking=lambda: self.togglePause_(None),
                on_recenter=lambda: self.recenterCursor_(None),
                on_toggle_hud=lambda: self.toggleCameraHud_(None),
                on_toggle_dictation=lambda: self.toggleDictation_(None),
            )

    def applicationWillTerminate_(self, _notification) -> None:
        try:
            Foundation.NSNotificationCenter.defaultCenter().removeObserver_(self)
        except Exception:
            pass
        if self.ui_timer is not None:
            self.ui_timer.invalidate()
            self.ui_timer = None
        for mon in (self._global_hotkey_mon, self._local_hotkey_mon):
            if mon is not None:
                try:
                    AppKit.NSEvent.removeMonitor_(mon)
                except Exception:
                    pass
        self._persister.flush()
        if self.runtime is not None:
            self.runtime.stop()

    # ------------------------------------------------------------------
    # Configurable Global & Local Hotkeys (graceful local-only fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _event_matches_spec(event, spec: str) -> bool:
        parsed = parse_hotkey_spec(spec)
        if parsed is None:
            return False
        mods_req, key_req = parsed
        flags = int(event.modifierFlags())
        active_mods = set()
        if flags & AppKit.NSEventModifierFlagCommand:
            active_mods.add("cmd")
        if flags & AppKit.NSEventModifierFlagShift:
            active_mods.add("shift")
        if flags & AppKit.NSEventModifierFlagControl:
            active_mods.add("ctrl")
        if flags & AppKit.NSEventModifierFlagOption:
            active_mods.add("opt")
        if active_mods != mods_req:
            return False
        chars = str(event.charactersIgnoringModifiers() or "").lower()
        return chars == key_req

    def _handle_hotkey_event(self, event) -> bool:
        try:
            hk_cfg = self.config.get("hotkeys") or {}
            if not hk_cfg.get("enabled", True):
                return False
            if self._event_matches_spec(event, str(hk_cfg.get("toggle_pause", "cmd+shift+a"))):
                self.togglePause_(None)
                return True
            if self._event_matches_spec(event, str(hk_cfg.get("recenter", "cmd+shift+r"))):
                self.recenterCursor_(None)
                return True
            if self._event_matches_spec(event, str(hk_cfg.get("toggle_hud", "cmd+shift+h"))):
                self.toggleCameraHud_(None)
                return True
            if self._event_matches_spec(event, str(hk_cfg.get("mark_not_intended", "cmd+shift+n"))):
                self.markNotIntended_(None)
                return True
            if self._event_matches_spec(
                event, str(hk_cfg.get("mark_deliberate_gesture", "cmd+ctrl+opt+m"))
            ):
                if self.runtime is not None and self.runtime.passive_recorder is not None:
                    self.runtime.mark_deliberate_gesture()
                return True
        except Exception:
            pass
        return False

    def _register_hotkeys(self) -> None:
        self._hotkeys_local_only = False
        try:
            self._global_hotkey_mon = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                AppKit.NSEventMaskKeyDown,
                lambda ev: self._handle_hotkey_event(ev),
            )
            if self._global_hotkey_mon is None:
                self._hotkeys_local_only = True
        except Exception as exc:
            self._hotkeys_local_only = True
            logger.debug(f"[HOTKEYS] Global monitor unavailable (falling back to local-only): {exc}")
        try:
            self._local_hotkey_mon = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                AppKit.NSEventMaskKeyDown,
                lambda ev: (None if self._handle_hotkey_event(ev) else ev),
            )
        except Exception as exc:
            logger.debug(f"[HOTKEYS] Unable to register local monitor: {exc}")

    # ------------------------------------------------------------------
    # Localized Menu Bar (`NSStatusBar`)
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        bar = AppKit.NSStatusBar.systemStatusBar()
        self.status_item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        btn = self.status_item.button()
        if btn is not None:
            btn.setTitle_("✦ AeroHand")

        menu = AppKit.NSMenu.alloc().init()

        def _add(key: str, title: str, action: str, key_equiv: str = "") -> None:
            item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key_equiv)
            item.setTarget_(self)
            menu.addItem_(item)
            self._menu_items[key] = item

        _add("status", "AeroHand", "showControlCenter:")
        self._menu_items["status"].setEnabled_(False)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        _add("toggle_pause", "Pause", "togglePause:")
        _add("toggle_mode", "Live / Dry-Run", "toggleDryRun:")
        _add("toggle_cam", "Camera HUD", "toggleCameraHud:")
        _add("recenter", "Recenter", "recenterCursor:")
        _add("dictation", "Dictation", "toggleDictation:")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        _add("open_cc", "Control Center", "showControlCenter:", ",")
        _add("open_acc", "Accessibility Settings", "openAccessibility:")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        _add("quit", "Quit AeroHand", "quitApp:", "q")

        self.status_item.setMenu_(menu)

    # ------------------------------------------------------------------
    # Modern Dark Glass Control Center (`NSWindow` + `WKWebView`)
    # ------------------------------------------------------------------

    def _build_control_center(self) -> None:
        w, h = 920, 670
        screen = AppKit.NSScreen.mainScreen()
        sf = screen.visibleFrame() if screen else Foundation.NSMakeRect(0, 0, 1440, 900)
        rect = Foundation.NSMakeRect(
            sf.origin.x + (sf.size.width - w) * 0.5,
            sf.origin.y + (sf.size.height - h) * 0.5,
            w,
            h,
        )

        style = (
            AppKit.NSWindowStyleMaskTitled
            | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskMiniaturizable
            | AppKit.NSWindowStyleMaskResizable
            | AppKit.NSWindowStyleMaskFullSizeContentView
        )
        win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, AppKit.NSBackingStoreBuffered, False
        )
        win.setTitle_("AeroHand Control Center")
        win.setTitlebarAppearsTransparent_(True)
        win.setTitleVisibility_(AppKit.NSWindowTitleHidden)
        win.setMovable_(True)
        win.setMovableByWindowBackground_(True)
        win.setReleasedWhenClosed_(False)
        win.setMinSize_(Foundation.NSMakeSize(780, 560))
        try:
            win.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua))
        except Exception:
            pass

        wk_cfg = WebKit.WKWebViewConfiguration.alloc().init()
        ucc = WebKit.WKUserContentController.alloc().init()
        self._bridge_handler = WebBridgeHandler.alloc().initWithCallback_(self._on_js_message)
        ucc.addScriptMessageHandler_name_(self._bridge_handler, "aerohand")
        wk_cfg.setUserContentController_(ucc)

        root_view = AppKit.NSView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, w, h))
        root_view.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)

        wv = WebKit.WKWebView.alloc().initWithFrame_configuration_(
            Foundation.NSMakeRect(0, 0, w, h), wk_cfg
        )
        wv.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        try:
            wv.setValue_forKey_(Foundation.NSNumber.numberWithBool_(False), "drawsBackground")
        except Exception:
            pass

        html_path = Path(__file__).resolve().parent / "web" / "control_center.html"
        file_url = Foundation.NSURL.fileURLWithPath_(str(html_path))
        dir_url = Foundation.NSURL.fileURLWithPath_(str(html_path.parent))
        wv.loadFileURL_allowingReadAccessToURL_(file_url, dir_url)
        root_view.addSubview_(wv)

        # Native transparent drag regions over the hidden titlebar, main header (.topbar),
        # and sidebar brand header (.brand), staying clear of traffic lights and resize edges.
        drag_top = WindowDragView.alloc().initWithFrame_(
            Foundation.NSMakeRect(78, h - 30, w - 84, 26)
        )
        drag_top.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewMinYMargin)
        root_view.addSubview_(drag_top)

        drag_topbar = WindowDragView.alloc().initWithFrame_(
            Foundation.NSMakeRect(214, h - 68, w - 220, 64)
        )
        drag_topbar.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewMinYMargin)
        root_view.addSubview_(drag_topbar)

        drag_brand = WindowDragView.alloc().initWithFrame_(
            Foundation.NSMakeRect(6, h - 80, 204, 48)
        )
        drag_brand.setAutoresizingMask_(AppKit.NSViewMaxXMargin | AppKit.NSViewMinYMargin)
        root_view.addSubview_(drag_brand)

        win.setContentView_(root_view)
        self.window = win
        self.control_webview = wv

    # ------------------------------------------------------------------
    # Full-Screen Transparent Click-Through Screen Overlays (1 NSPanel per Display)
    # ------------------------------------------------------------------

    def _destroy_screen_overlays(self) -> None:
        for entry in self._overlay_entries:
            try:
                entry["panel"].orderOut_(None)
                entry["panel"].close()
            except Exception:
                pass
        self._overlay_entries = []
        self.overlay_panel = None
        self.overlay_webview = None
        self._overlay_ready = False

    def _build_screen_overlay(self) -> None:
        with objc.autorelease_pool():
            self._destroy_screen_overlays()
            screens = list(AppKit.NSScreen.screens() or [])
            if not screens:
                main_sc = AppKit.NSScreen.mainScreen()
                if main_sc is not None:
                    screens = [main_sc]

            html_path = Path(__file__).resolve().parent / "web" / "overlay.html"
            file_url = Foundation.NSURL.fileURLWithPath_(str(html_path))
            dir_url = Foundation.NSURL.fileURLWithPath_(str(html_path.parent))
            style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
            if self._overlay_bridge_handler is None:
                self._overlay_bridge_handler = WebBridgeHandler.alloc().initWithCallback_(self._on_js_message)
            if self._overlay_nav_delegate is None:
                self._overlay_nav_delegate = OverlayNavigationDelegate.alloc().initWithTerminateCallback_finishCallback_(
                    self._on_overlay_webcontent_terminated,
                    self._on_overlay_did_finish_navigation,
                )

            raw_frames: List[Tuple[float, float, float, float]] = []
            for idx, sc in enumerate(screens):
                sf = sc.frame()
                raw_frames.append(
                    (float(sf.origin.x), float(sf.origin.y), float(sf.size.width), float(sf.size.height))
                )
                q_rect = self._screen_geom.to_quartz_rect(raw_frames[-1])
                panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                    sf, style, AppKit.NSBackingStoreBuffered, False
                )
                panel.setFrame_display_(sf, False)
                panel.setLevel_(AppKit.NSScreenSaverWindowLevel)
                panel.setHidesOnDeactivate_(False)
                panel.setOpaque_(False)
                panel.setBackgroundColor_(AppKit.NSColor.clearColor())
                panel.setHasShadow_(False)
                panel.setIgnoresMouseEvents_(True)
                panel.setCollectionBehavior_(
                    AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                    | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
                    | AppKit.NSWindowCollectionBehaviorStationary
                )

                wk_cfg = WebKit.WKWebViewConfiguration.alloc().init()
                ucc = WebKit.WKUserContentController.alloc().init()
                ucc.addScriptMessageHandler_name_(self._overlay_bridge_handler, "aerohand")
                wk_cfg.setUserContentController_(ucc)
                wv = WebKit.WKWebView.alloc().initWithFrame_configuration_(
                    Foundation.NSMakeRect(0, 0, sf.size.width, sf.size.height), wk_cfg
                )
                wv.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
                try:
                    wv.setValue_forKey_(Foundation.NSNumber.numberWithBool_(False), "drawsBackground")
                except Exception:
                    pass
                wv.setNavigationDelegate_(self._overlay_nav_delegate)
                wv.loadFileURL_allowingReadAccessToURL_(file_url, dir_url)
                panel.setContentView_(wv)

                is_primary = abs(q_rect[0]) < 2.0 and abs(q_rect[1]) < 2.0
                self._overlay_entries.append(
                    {
                        "panel": panel,
                        "webview": wv,
                        "quartz_rect": q_rect,
                        "is_primary": is_primary or idx == 0,
                        "ready": False,
                        "payload_builder": OverlayPayloadBuilder(),
                    }
                )

            self._overlay_screen_signature = screen_frames_signature(raw_frames)
            logger.info(
                f"[OVERLAY] Built {len(self._overlay_entries)} per-display overlay panel(s): "
                f"{[e['quartz_rect'] for e in self._overlay_entries]}"
            )
            if self._overlay_entries:
                prim = next((e for e in self._overlay_entries if e["is_primary"]), self._overlay_entries[0])
                self.overlay_panel = prim["panel"]
                self.overlay_webview = prim["webview"]
                self._overlay_ready = True

    def _on_overlay_did_finish_navigation(self, webview: Any) -> None:
        for entry in self._overlay_entries:
            if entry["webview"] is webview:
                if not entry.get("ready", False):
                    entry["ready"] = True
                    entry["payload_builder"].reset()
                break

    def _on_overlay_webcontent_terminated(self, webview: Any) -> None:
        logger.warning("[OVERLAY] WKWebView WebContent process terminated; self-healing reload triggered.")
        html_path = Path(__file__).resolve().parent / "web" / "overlay.html"
        file_url = Foundation.NSURL.fileURLWithPath_(str(html_path))
        dir_url = Foundation.NSURL.fileURLWithPath_(str(html_path.parent))
        for entry in self._overlay_entries:
            if entry["webview"] is webview:
                entry["ready"] = False
                entry["payload_builder"].reset()
                try:
                    webview.loadFileURL_allowingReadAccessToURL_(file_url, dir_url)
                except Exception as exc:
                    logger.warning(f"[OVERLAY] Failed to reload WebView: {exc}")
                break

    def onScreenParametersChanged_(self, _notification) -> None:
        with objc.autorelease_pool():
            screens = list(AppKit.NSScreen.screens() or [])
            cur_sig = screen_frames_signature(
                [
                    (
                        float(sc.frame().origin.x),
                        float(sc.frame().origin.y),
                        float(sc.frame().size.width),
                        float(sc.frame().size.height),
                    )
                    for sc in screens
                ]
            )
            if cur_sig and cur_sig == self._overlay_screen_signature:
                # Dock/menu bar/Space change without display topology change — keep existing WebViews alive!
                self._apply_camera_hud_layout()
                if self.runtime is not None:
                    self.runtime._refresh_display_bounds()
                return
            logger.info("[UI] Display geometry changed; rebuilding per-display click-through overlays.")
            self._build_screen_overlay()
            for entry in self._overlay_entries:
                entry["panel"].orderFront_(None)
            self._apply_camera_hud_layout()
            if self.runtime is not None:
                self.runtime._refresh_display_bounds()

    @staticmethod
    def _rect_contains(rect: tuple[float, float, float, float], x: float, y: float) -> bool:
        rx, ry, rw, rh = rect
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

    # ------------------------------------------------------------------
    # Hover-Aware Floating Camera HUD (4 Corners, 3 Sizes, Smooth Hover Opacity)
    # ------------------------------------------------------------------

    def _compute_camera_hud_frame(self) -> Any:
        hud_cfg = ((self.config.get("ui") or {}).get("camera_hud")) or {}
        screens = list(AppKit.NSScreen.screens() or [])
        visible_frames = [
            (
                float(sc.visibleFrame().origin.x),
                float(sc.visibleFrame().origin.y),
                float(sc.visibleFrame().size.width),
                float(sc.visibleFrame().size.height),
            )
            for sc in screens
        ]
        main_sc = AppKit.NSScreen.mainScreen()
        if main_sc is not None:
            msf = main_sc.visibleFrame()
            main_vf = (float(msf.origin.x), float(msf.origin.y), float(msf.size.width), float(msf.size.height))
        elif visible_frames:
            main_vf = visible_frames[0]
        else:
            main_vf = (0.0, 0.0, 1440.0, 900.0)
        x, y, pw, ph = compute_camera_hud_rect(hud_cfg, visible_frames, main_vf)
        self._hud_size_px = (int(round(pw)), int(round(ph)))
        return Foundation.NSMakeRect(x, y, pw, ph)

    def _on_camera_hud_moved(self, x: float, y: float) -> None:
        set_nested_key(self.config, "ui.camera_hud.custom_x", round(float(x), 1))
        set_nested_key(self.config, "ui.camera_hud.custom_y", round(float(y), 1))
        self._persister.schedule(self.config)

    def _build_floating_camera_panel(self) -> None:
        rect = self._compute_camera_hud_frame()
        pw, ph = self._hud_size_px
        hud_cfg = ((self.config.get("ui") or {}).get("camera_hud")) or {}
        idle_op = float(hud_cfg.get("opacity", 0.55))
        hover_op = float(hud_cfg.get("hover_opacity", 1.0))

        style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, AppKit.NSBackingStoreBuffered, False
        )
        panel.setLevel_(AppKit.NSFloatingWindowLevel)
        panel.setHidesOnDeactivate_(False)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setMovable_(True)
        panel.setMovableByWindowBackground_(True)
        panel.setAlphaValue_(idle_op)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        container = HoverCameraView.alloc().initWithFrame_idleAlpha_hoverAlpha_onMoved_(
            Foundation.NSMakeRect(0, 0, pw, ph), idle_op, hover_op, self._on_camera_hud_moved
        )
        container.setWantsLayer_(True)
        container.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        layer = container.layer()
        layer.setCornerRadius_(18.0)
        layer.setMasksToBounds_(True)
        layer.setBorderWidth_(1.5)
        layer.setBorderColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.18, 0.84, 0.75, 0.55).CGColor())
        layer.setBackgroundColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.05, 0.08, 0.13, 0.94).CGColor())

        img_view = NonInteractiveImageView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, pw, ph))
        img_view.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        img_view.setImageScaling_(AppKit.NSImageScaleAxesIndependently)
        container.addSubview_(img_view)

        panel.setContentView_(container)
        self.camera_panel = panel
        self.camera_container = container
        self.camera_image_view = img_view

    def _apply_camera_hud_layout(self) -> None:
        if self.camera_panel is None:
            return
        hud_cfg = ((self.config.get("ui") or {}).get("camera_hud")) or {}
        rect = self._compute_camera_hud_frame()
        self.camera_panel.setFrame_display_animate_(rect, True, True)
        if self.camera_container is not None:
            self.camera_container.setAlphas_(
                (float(hud_cfg.get("opacity", 0.55)), float(hud_cfg.get("hover_opacity", 1.0)))
            )
        if bool(hud_cfg.get("visible", True)):
            self.camera_panel.orderFront_(None)
        else:
            self.camera_panel.orderOut_(None)

    # ------------------------------------------------------------------
    # JS <-> Python Bridge
    # ------------------------------------------------------------------

    def _hydrate_control_center(self) -> None:
        if self.control_webview is None or not self._control_ready:
            return
        lang = str((self.config.get("ui") or {}).get("language", "vi"))
        payload = {
            "version": __version__,
            "cfg": self.config,
            "i18n": get_lang_pack(lang),
            "cameras": self.runtime.list_available_cameras() if self.runtime else [],
            "launch_at_login": is_launch_at_login_enabled(),
            "hotkeys_local_only": self._hotkeys_local_only,
            "hotkey_conflicts": detect_hotkey_conflicts(self.config.get("hotkeys") or {}),
        }
        js = f"window.hydrateState && window.hydrateState({json.dumps(payload, ensure_ascii=False)});"
        self.control_webview.evaluateJavaScript_completionHandler_(js, None)

    def _sync_external_config(self, cfg: Dict[str, Any]) -> None:
        self.config = cfg
        self._apply_camera_hud_layout()
        self._hydrate_control_center()

    def _persist_and_apply(self, *, rehydrate: bool = False) -> None:
        if self.runtime is not None:
            self.runtime.apply_live_settings(self.config)
        self._apply_camera_hud_layout()
        self._persister.schedule(self.config)
        if rehydrate:
            self._hydrate_control_center()

    def _on_js_message(self, msg: Dict[str, Any]) -> None:
        act = str(msg.get("action", ""))
        if act == "ready":
            self._control_ready = True
            self._hydrate_control_center()
        elif act in ("overlay_ready", "overlay_need_sync"):
            for entry in self._overlay_entries:
                entry["ready"] = True
                entry["payload_builder"].reset()
        elif act == "window_drag":
            ev = AppKit.NSApp.currentEvent()
            if ev is not None and self.window is not None:
                try:
                    self.window.performWindowDragWithEvent_(ev)
                except Exception as exc:
                    logger.debug(f"[UI] JS-triggered window_drag ignored: {exc}")
        elif act == "set_key":
            key = str(msg.get("key", ""))
            val = msg.get("value")
            if key:
                set_nested_key(self.config, key, val)
                if key == "ui.camera_hud.position":
                    hud_cfg = (self.config.get("ui") or {}).get("camera_hud")
                    if isinstance(hud_cfg, dict):
                        hud_cfg.pop("custom_x", None)
                        hud_cfg.pop("custom_y", None)
                need_rehydrate = (
                    key in ("ui.language", "hand.gesture_preset")
                    or key.startswith("palm_wheel.bindings.")
                    or key.startswith("hotkeys.")
                )
                self._persist_and_apply(rehydrate=need_rehydrate)
        elif act == "set_keys":
            updates = msg.get("updates") or {}
            if isinstance(updates, dict):
                for k, v in updates.items():
                    set_nested_key(self.config, str(k), v)
                self._persist_and_apply(rehydrate=False)
        elif act == "apply_preset":
            preset = str(msg.get("preset", "airpoint"))
            self.config = apply_gesture_preset(self.config, preset)
            self._persist_and_apply(rehydrate=True)
        elif act == "toggle_pause":
            self.togglePause_(None)
        elif act == "toggle_dry_run":
            self.toggleDryRun_(None)
        elif act == "toggle_camera_hud":
            self.toggleCameraHud_(None)
        elif act == "recenter":
            self.recenterCursor_(None)
        elif act == "toggle_dictation":
            self.toggleDictation_(None)
        elif act == "open_privacy":
            _open_privacy_pane(str(msg.get("pane", "Privacy_Accessibility")))
        elif act == "refresh_cameras":
            self._hydrate_control_center()
        elif act == "test_sound":
            if self.runtime is not None:
                self.runtime.audio.play_click_down()
                self.runtime.audio.play_click_up()
        elif act == "set_launch_at_login":
            enabled = bool(msg.get("enabled", False))
            set_launch_at_login(enabled)
            set_nested_key(self.config, "system.launch_at_login", enabled)
            self._persist_and_apply(rehydrate=True)
        elif act == "collector_toggle":
            if self.runtime is not None:
                self.runtime.collector_toggle_recording()
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_set_recording":
            if self.runtime is not None:
                self.runtime.collector_set_recording(bool(msg.get("active", False)))
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_pose":
            if self.runtime is not None:
                self.runtime.collector_set_pose(str(msg.get("pose", "idle")))
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_meta":
            if self.runtime is not None and isinstance(msg.get("meta"), dict):
                self.runtime.collector_set_meta(msg["meta"])
        elif act == "collector_undo":
            if self.runtime is not None:
                self.runtime.collector_undo(float(msg.get("seconds", 5.0)))
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_guided_start":
            if self.runtime is not None:
                self.runtime.collector_start_guided()
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_guided_stop":
            if self.runtime is not None:
                self.runtime.collector_stop_guided()
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_cued_start":
            if self.runtime is not None:
                self.runtime.collector_start_cued(
                    cues_per_trigger=int(msg.get("cues_per_trigger", 10)),
                    hand=msg.get("hand"),
                    window_ms=int(msg.get("window_ms", 3500)),
                    gap_ms=int(msg.get("gap_ms", 1800)),
                    seed=msg.get("seed"),
                )
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_cued_stop":
            if self.runtime is not None:
                self.runtime.collector_stop_cued()
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_cued_save":
            if self.runtime is not None:
                self.runtime.collector_save_cued()
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_save":
            if self.runtime is not None:
                self.runtime.collector_save(allow_low_quality=bool(msg.get("allow_low_quality", False)))
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "collector_clear":
            if self.runtime is not None:
                self.runtime.collector_clear()
                self._on_ui_tick_inner(force_control_telemetry=True)
        elif act == "classifier_rearm":
            if self.runtime is not None:
                self.runtime.rearm_classifier()
        elif act == "mark_not_intended":
            self.markNotIntended_(None)
        elif act == "reset_practice":
            if self.runtime is not None:
                self.runtime.reset_practice()
        elif act == "calibration_start":
            if self.runtime is not None:
                self.runtime.calibration_start()
        elif act == "calibration_cancel":
            if self.runtime is not None:
                self.runtime.calibration_cancel()
        elif act == "calibration_apply":
            if self.runtime is not None:
                updated = self.runtime.calibration_apply()
                if updated is not None:
                    self.config = updated
                    self._persist_and_apply(rehydrate=True)
        elif act == "reset_defaults":
            self.resetDefaults_(None)
        elif act == "quit":
            self.quitApp_(None)

    # ------------------------------------------------------------------
    # Periodic 20 Hz UI, Menu Bar, Camera HUD & Screen Overlay Sync
    # ------------------------------------------------------------------

    def onUiTick_(self, _timer) -> None:
        with objc.autorelease_pool():
            self._on_ui_tick_inner()

    def _on_ui_tick_inner(self, *, force_control_telemetry: bool = False) -> None:
        self._tick_count += 1

        while not self._ui_queue.empty():
            try:
                fn = self._ui_queue.get_nowait()
                fn()
            except Exception as exc:
                logger.debug(f"[UI] Queued callback error: {exc}")

        trig_fp = _show_trigger_path()
        if trig_fp.exists():
            try:
                trig_fp.unlink(missing_ok=True)
            except OSError:
                pass
            self.showControlCenter_(None)

        if self.runtime is None:
            return
        snap: RuntimeSnapshot = self.runtime.get_snapshot()
        window_visible = bool(self.window is not None and self.window.isVisible())
        col_snap = snap.collector or {}
        col_guided = col_snap.get("guided") or {}
        col_cued = col_snap.get("cued") or {}
        has_collector_overlay = bool(
            col_snap.get("recording")
            or col_guided.get("active")
            or col_guided.get("phase") == "done"
            or col_cued.get("active")
            or (col_cued.get("phase") == "done" and col_cued.get("has_unsaved"))
        )
        has_urgent_overlay = (
            snap.click_event_seq != self._last_click_dispatched_seq
            or bool(snap.palm_wheels)
            or (snap.window_tile_preview is not None)
            or bool(snap.dictation.get("recording"))
            or bool(snap.dictation.get("transcribing"))
            or bool(snap.dictation.get("discard_active"))
            or (snap.active_hint is not None)
            or (snap.warning is not None)
            or has_collector_overlay
        )
        need_clear_overlay = self._overlay_had_visuals and (not has_urgent_overlay)
        self._overlay_had_visuals = has_urgent_overlay
        # Drop tick rate from 20 Hz to 5 Hz when Control Center is hidden and overlays are idle
        if (not window_visible) and (not has_urgent_overlay) and (not need_clear_overlay) and ((self._tick_count % 4) != 0):
            return

        lang = str((self.config.get("ui") or {}).get("language", "vi"))
        pack = get_lang_pack(lang)
        mtxt = pack.get("menu") or {}

        # 1. Menu Bar title & localized labels (at 5 Hz to minimize AppKit layout churn)
        if ((self._tick_count % 4) == 0 or self._tick_count == 1) and self.status_item is not None:
            btn = self.status_item.button()
            if btn is not None:
                if snap.paused:
                    btn.setTitle_("⏸ AeroHand")
                elif snap.dictation.get("recording"):
                    btn.setTitle_("🎙 AeroHand")
                elif snap.guard_reason:
                    btn.setTitle_("⌨ AeroHand")
                elif snap.dry_run:
                    btn.setTitle_("◌ AeroHand")
                else:
                    btn.setTitle_("✦ AeroHand")

            hud_vis = bool(((self.config.get("ui") or {}).get("camera_hud") or {}).get("visible", True))
            self._menu_items["status"].setTitle_(
                f"AeroHand v{__version__} • {snap.fps:.0f} FPS ({snap.delegate})"
            )
            self._menu_items["toggle_pause"].setTitle_(mtxt.get("resume" if snap.paused else "pause", "Pause"))
            self._menu_items["toggle_mode"].setTitle_(mtxt.get("to_live" if snap.dry_run else "to_dry", "Mode"))
            self._menu_items["toggle_cam"].setTitle_(mtxt.get("hide_hud" if hud_vis else "show_hud", "Camera HUD"))
            self._menu_items["recenter"].setTitle_(mtxt.get("recenter", "Recenter"))
            self._menu_items["dictation"].setTitle_(mtxt.get("dictation", "Dictation"))
            self._menu_items["open_cc"].setTitle_(mtxt.get("open_cc", "Control Center"))
            self._menu_items["open_acc"].setTitle_(mtxt.get("open_acc", "Accessibility"))
            self._menu_items["quit"].setTitle_(mtxt.get("quit", "Quit"))
        else:
            hud_vis = bool(((self.config.get("ui") or {}).get("camera_hud") or {}).get("visible", True))

        # 2. Floating Camera HUD image — explicit autorelease pool avoids ObjC bitmap buildup
        if hud_vis and snap.frame_bgr is not None and self.camera_image_view is not None:
            pw, ph = self._hud_size_px
            with objc.autorelease_pool():
                ns_img = _bgr_to_nsimage(snap.frame_bgr, pw, ph)
                if ns_img is not None:
                    self.camera_image_view.setImage_(ns_img)

        # 3. Transparent Screen Overlays (incremental i18n/ui, routed per display)
        if self._overlay_ready and self._overlay_entries:
            self._last_click_dispatched_seq = snap.click_event_seq
            ui_cfg = self.config.get("ui") or {}
            if has_collector_overlay and not self._overlay_collector_was_active:
                for entry in self._overlay_entries:
                    try:
                        entry["panel"].orderFrontRegardless()
                    except Exception:
                        pass
            self._overlay_collector_was_active = has_collector_overlay

            fallback_pt: Optional[Tuple[float, float]] = None
            if snap.hands_count == 0:
                if window_visible and self.window is not None:
                    wf = self.window.frame()
                    q_win = self._screen_geom.to_quartz_rect(
                        (float(wf.origin.x), float(wf.origin.y), float(wf.size.width), float(wf.size.height))
                    )
                    fallback_pt = (q_win[0] + q_win[2] * 0.5, q_win[1] + q_win[3] * 0.5)
                else:
                    try:
                        ml = AppKit.NSEvent.mouseLocation()
                        q_m = self._screen_geom.to_quartz_rect((float(ml.x), float(ml.y), 1.0, 1.0))
                        fallback_pt = (q_m[0], q_m[1])
                    except Exception:
                        pass

            compact_collector = None
            if has_collector_overlay:
                compact_collector = {
                    "recording": bool(col_snap.get("recording", False)),
                    "current_pose": col_snap.get("current_pose", "idle"),
                    "total": int(col_snap.get("total", 0)),
                    "counts": col_snap.get("counts") or {},
                    "session_meta": col_snap.get("session_meta") or {},
                    "guided": col_guided,
                    "cued": col_cued,
                    "hand": snap.hand,
                    "hands_count": int(snap.hands_count),
                    "live_pose": snap.active_pose or snap.pose,
                }

            quartz_rects = [e["quartz_rect"] for e in self._overlay_entries]
            snap_payloads = build_overlay_snap_payloads(
                quartz_rects,
                snap,
                has_collector_overlay,
                compact_collector,
                fallback_point=fallback_pt,
            )
            for entry, snap_payload in zip(self._overlay_entries, snap_payloads):
                if not entry.get("ready", False):
                    continue
                builder: OverlayPayloadBuilder = entry["payload_builder"]
                ov_payload = builder.build(snap_payload, ui_cfg, lang, pack)
                js_ov = f"window.updateOverlay && window.updateOverlay({json.dumps(ov_payload, ensure_ascii=False)});"
                entry["webview"].evaluateJavaScript_completionHandler_(js_ov, None)

        # 4. Control Center telemetry (10 Hz, or immediate on user button click when window visible)
        if (
            (force_control_telemetry or (self._tick_count % 2) == 0)
            and self._control_ready
            and self.control_webview is not None
            and window_visible
        ):
            tel = {
                "running": snap.running,
                "paused": snap.paused,
                "dry_run": snap.dry_run,
                "fps": round(snap.fps, 1),
                "delegate": snap.delegate,
                "classifier_name": snap.classifier_name,
                "classifier_telemetry": snap.classifier_telemetry,
                "accessibility_ok": snap.accessibility_ok,
                "permissions": snap.permissions,
                "guard_reason": snap.guard_reason,
                "active_app": snap.active_app,
                "hand": snap.hand,
                "hands_count": snap.hands_count,
                "is_pointing": snap.is_pointing,
                "dir_z": round(snap.dir_z, 3),
                "pose": snap.pose,
                "active_pose": snap.active_pose,
                "last_action": snap.last_action,
                "refusal_code": snap.refusal_code,
                "refusal_reason": snap.refusal_reason,
                "probabilities": snap.probabilities,
                "dictation": snap.dictation,
                "timing": snap.timing,
                "vitals": snap.vitals,
                "collector": snap.collector,
                "calibration": snap.calibration,
                "practice": snap.practice,
                "error": snap.error,
            }
            js_cc = f"window.updateTelemetry && window.updateTelemetry({json.dumps(tel, ensure_ascii=False)});"
            self.control_webview.evaluateJavaScript_completionHandler_(js_cc, None)

    # ------------------------------------------------------------------
    # Menu & Action Selectors
    # ------------------------------------------------------------------

    def resetDefaults_(self, _sender) -> None:
        self.config = load_config(config_path=default_config_path(), overrides_path="/nonexistent_overrides.yaml")
        self._persist_and_apply(rehydrate=True)

    def togglePause_(self, _sender) -> None:
        if self.runtime is not None:
            self.runtime.set_paused(not self.runtime.is_paused)

    def toggleDryRun_(self, _sender) -> None:
        if self.runtime is None:
            return
        want_live = self.runtime.dry_run
        if want_live and not is_accessibility_trusted(prompt_user=True):
            _open_privacy_pane("Privacy_Accessibility")
            return
        self.runtime.set_dry_run(not self.runtime.dry_run)

    def toggleCameraHud_(self, _sender) -> None:
        cur = bool(((self.config.get("ui") or {}).get("camera_hud") or {}).get("visible", True))
        set_nested_key(self.config, "ui.camera_hud.visible", not cur)
        self._persist_and_apply(rehydrate=False)

    def recenterCursor_(self, _sender) -> None:
        if self.runtime is not None:
            self.runtime.trigger_recenter()

    def toggleDictation_(self, _sender) -> None:
        if self.runtime is not None:
            self.runtime.toggle_dictation()

    def markNotIntended_(self, _sender) -> None:
        if self.runtime is not None:
            self.runtime.mark_last_trigger_not_intended()

    def showControlCenter_(self, _sender) -> None:
        if self.window is not None:
            self.window.makeKeyAndOrderFront_(None)
            AppKit.NSApp.activateIgnoringOtherApps_(True)

    def openAccessibility_(self, _sender) -> None:
        _open_privacy_pane("Privacy_Accessibility")

    def quitApp_(self, _sender) -> None:
        AppKit.NSApp.terminate_(None)


def main() -> None:
    log_fp = _log_path()
    try:
        lfd = os.open(str(log_fp), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.close(lfd)
        os.chmod(log_fp, 0o600)
    except OSError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(str(log_fp), encoding="utf-8"),
        ],
        force=True,
    )
    if not _acquire_single_instance():
        logger.info("AeroHand is already running; surfaced existing Control Center.")
        return

    cfg = load_config()
    app = AppKit.NSApplication.sharedApplication()
    delegate = AeroHandAppDelegate.alloc().initWithConfig_(cfg)
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
