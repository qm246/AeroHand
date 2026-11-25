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
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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
from aerohand.platform.permissions import is_accessibility_trusted
from aerohand.runtime import AeroHandRuntime, RuntimeSnapshot

logger = logging.getLogger("aerohand.app")

_LOCK_PATH = "/tmp/aerohand_app.lock"
_SHOW_TRIGGER_PATH = "/tmp/aerohand_show_window"
_LOCK_FD = None

_HUD_SIZES = {
    "small": (220, 154),
    "medium": (300, 210),
    "large": (390, 273),
}


def _acquire_single_instance() -> bool:
    global _LOCK_FD
    try:
        _LOCK_FD = open(_LOCK_PATH, "w")
        fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _LOCK_FD.write(str(os.getpid()))
        _LOCK_FD.flush()
        atexit.register(_release_lock)
        return True
    except OSError:
        try:
            Path(_SHOW_TRIGGER_PATH).write_text("show", encoding="utf-8")
        except OSError:
            pass
        return False


def _release_lock() -> None:
    global _LOCK_FD
    try:
        if _LOCK_FD is not None:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
            _LOCK_FD.close()
            _LOCK_FD = None
        if os.path.exists(_LOCK_PATH):
            os.unlink(_LOCK_PATH)
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


class HoverCameraView(AppKit.NSView):
    """NSView hosting the Floating Camera HUD with smooth Airpoint-style hover opacity."""

    def initWithFrame_idleAlpha_hoverAlpha_(self, frame, idle_alpha: float, hover_alpha: float):
        self = objc.super(HoverCameraView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._idle_alpha = float(idle_alpha)
        self._hover_alpha = float(hover_alpha)
        self._tracking_area = None
        self._hovered = False
        return self

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
        try:
            body = _to_py(message.body())
            if isinstance(body, dict):
                self._callback(body)
        except Exception as exc:
            logger.debug(f"[BRIDGE] Failed handling JS message: {exc}")


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
        self._hud_size_px = (300, 210)
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
        self.runtime.start_background()

        hud_vis = bool(((self.config.get("ui") or {}).get("camera_hud") or {}).get("visible", True))
        if hud_vis and self.camera_panel is not None:
            self.camera_panel.orderFront_(None)
        if self.overlay_panel is not None:
            self.overlay_panel.orderFront_(None)
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
        if self.ui_timer is not None:
            self.ui_timer.invalidate()
            self.ui_timer = None
        for mon in (self._global_hotkey_mon, self._local_hotkey_mon):
            if mon is not None:
                try:
                    AppKit.NSEvent.removeMonitor_(mon)
                except Exception:
                    pass
        if self.runtime is not None:
            self.runtime.stop()

    # ------------------------------------------------------------------
    # Global & Local Hotkeys (`Cmd+Shift+A/R/H`)
    # ------------------------------------------------------------------

    def _handle_hotkey_event(self, event) -> bool:
        try:
            hk_cfg = self.config.get("hotkeys") or {}
            if not hk_cfg.get("enabled", True):
                return False
            flags = int(event.modifierFlags())
            need = AppKit.NSEventModifierFlagCommand | AppKit.NSEventModifierFlagShift
            if (flags & need) != need:
                return False
            chars = str(event.charactersIgnoringModifiers() or "").lower()
            if chars == "a":
                self.togglePause_(None)
                return True
            if chars == "r":
                self.recenterCursor_(None)
                return True
            if chars == "h":
                self.toggleCameraHud_(None)
                return True
        except Exception:
            pass
        return False

    def _register_hotkeys(self) -> None:
        try:
            self._global_hotkey_mon = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                AppKit.NSEventMaskKeyDown,
                lambda ev: self._handle_hotkey_event(ev),
            )
            self._local_hotkey_mon = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                AppKit.NSEventMaskKeyDown,
                lambda ev: (None if self._handle_hotkey_event(ev) else ev),
            )
        except Exception as exc:
            logger.debug(f"[HOTKEYS] Unable to register monitors: {exc}")

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

        win.setContentView_(wv)
        self.window = win
        self.control_webview = wv

    # ------------------------------------------------------------------
    # Full-Screen Transparent Click-Through Screen Overlay (`NSPanel` + `WKWebView`)
    # ------------------------------------------------------------------

    def _build_screen_overlay(self) -> None:
        screen = AppKit.NSScreen.mainScreen()
        sf = screen.frame() if screen else Foundation.NSMakeRect(0, 0, 1440, 900)
        style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            sf, style, AppKit.NSBackingStoreBuffered, False
        )
        panel.setLevel_(AppKit.NSScreenSaverWindowLevel)
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
        wv = WebKit.WKWebView.alloc().initWithFrame_configuration_(
            Foundation.NSMakeRect(0, 0, sf.size.width, sf.size.height), wk_cfg
        )
        wv.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        try:
            wv.setValue_forKey_(Foundation.NSNumber.numberWithBool_(False), "drawsBackground")
        except Exception:
            pass

        html_path = Path(__file__).resolve().parent / "web" / "overlay.html"
        file_url = Foundation.NSURL.fileURLWithPath_(str(html_path))
        dir_url = Foundation.NSURL.fileURLWithPath_(str(html_path.parent))
        wv.loadFileURL_allowingReadAccessToURL_(file_url, dir_url)

        panel.setContentView_(wv)
        self.overlay_panel = panel
        self.overlay_webview = wv
        self._overlay_ready = True

    # ------------------------------------------------------------------
    # Hover-Aware Floating Camera HUD (4 Corners, 3 Sizes, Smooth Hover Opacity)
    # ------------------------------------------------------------------

    def _compute_camera_hud_frame(self) -> Any:
        hud_cfg = ((self.config.get("ui") or {}).get("camera_hud")) or {}
        size_key = str(hud_cfg.get("size", "medium"))
        pw, ph = _HUD_SIZES.get(size_key, (300, 210))
        self._hud_size_px = (pw, ph)
        pos = str(hud_cfg.get("position", "bottom-right"))

        screen = AppKit.NSScreen.mainScreen()
        sf = screen.visibleFrame() if screen else Foundation.NSMakeRect(0, 0, 1440, 900)
        pad = 22.0
        x = sf.origin.x + pad if "left" in pos else (sf.origin.x + sf.size.width - pw - pad)
        y = sf.origin.y + pad if "bottom" in pos else (sf.origin.y + sf.size.height - ph - pad)
        return Foundation.NSMakeRect(x, y, pw, ph)

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
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setMovableByWindowBackground_(True)
        panel.setAlphaValue_(idle_op)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        container = HoverCameraView.alloc().initWithFrame_idleAlpha_hoverAlpha_(
            Foundation.NSMakeRect(0, 0, pw, ph), idle_op, hover_op
        )
        container.setWantsLayer_(True)
        container.setAutoresizingMask_(AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable)
        layer = container.layer()
        layer.setCornerRadius_(18.0)
        layer.setMasksToBounds_(True)
        layer.setBorderWidth_(1.5)
        layer.setBorderColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.18, 0.84, 0.75, 0.55).CGColor())
        layer.setBackgroundColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.05, 0.08, 0.13, 0.94).CGColor())

        img_view = AppKit.NSImageView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, pw, ph))
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
        try:
            save_user_overrides(self.config)
        except Exception as exc:
            logger.debug(f"[UI] Could not persist overrides: {exc}")
        if rehydrate:
            self._hydrate_control_center()

    def _on_js_message(self, msg: Dict[str, Any]) -> None:
        act = str(msg.get("action", ""))
        if act == "ready":
            self._control_ready = True
            self._hydrate_control_center()
        elif act == "set_key":
            key = str(msg.get("key", ""))
            val = msg.get("value")
            if key:
                set_nested_key(self.config, key, val)
                need_rehydrate = key in ("ui.language", "hand.gesture_preset") or key.startswith("palm_wheel.bindings.")
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
        elif act == "collector_pose":
            if self.runtime is not None:
                self.runtime.collector_set_pose(str(msg.get("pose", "idle")))
        elif act == "collector_save":
            if self.runtime is not None:
                self.runtime.collector_save()
        elif act == "collector_clear":
            if self.runtime is not None:
                self.runtime.collector_clear()
        elif act == "reset_practice":
            if self.runtime is not None:
                self.runtime.reset_practice()
        elif act == "reset_defaults":
            self.resetDefaults_(None)
        elif act == "quit":
            self.quitApp_(None)

    # ------------------------------------------------------------------
    # Periodic 20 Hz UI, Menu Bar, Camera HUD & Screen Overlay Sync
    # ------------------------------------------------------------------

    def onUiTick_(self, _timer) -> None:
        self._tick_count += 1

        while not self._ui_queue.empty():
            try:
                fn = self._ui_queue.get_nowait()
                fn()
            except Exception as exc:
                logger.debug(f"[UI] Queued callback error: {exc}")

        if os.path.exists(_SHOW_TRIGGER_PATH):
            try:
                os.unlink(_SHOW_TRIGGER_PATH)
            except OSError:
                pass
            self.showControlCenter_(None)

        if self.runtime is None:
            return
        snap: RuntimeSnapshot = self.runtime.get_snapshot()
        lang = str((self.config.get("ui") or {}).get("language", "vi"))
        pack = get_lang_pack(lang)
        mtxt = pack.get("menu") or {}

        # 1. Menu Bar title & localized labels
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

        # 2. Floating Camera HUD image (20 Hz)
        if hud_vis and snap.frame_bgr is not None and self.camera_image_view is not None:
            pw, ph = self._hud_size_px
            ns_img = _bgr_to_nsimage(snap.frame_bgr, pw, ph)
            if ns_img is not None:
                self.camera_image_view.setImage_(ns_img)

        # 3. Transparent Screen Overlay (20 Hz for silky Palm Wheel & Click Pulse)
        if self._overlay_ready and self.overlay_webview is not None:
            ov_payload = {
                "snap": {
                    "click_event_seq": snap.click_event_seq,
                    "click_event": snap.click_event,
                    "palm_wheels": snap.palm_wheels,
                    "window_tile_preview": snap.window_tile_preview,
                    "active_hint": snap.active_hint,
                    "warning": snap.warning,
                    "dictation": snap.dictation,
                },
                "ui": self.config.get("ui") or {},
                "i18n": pack,
            }
            js_ov = f"window.updateOverlay && window.updateOverlay({json.dumps(ov_payload, ensure_ascii=False)});"
            self.overlay_webview.evaluateJavaScript_completionHandler_(js_ov, None)

        # 4. Control Center telemetry (10 Hz, only when window visible)
        if (
            (self._tick_count % 2) == 0
            and self._control_ready
            and self.control_webview is not None
            and self.window is not None
            and self.window.isVisible()
        ):
            tel = {
                "running": snap.running,
                "paused": snap.paused,
                "dry_run": snap.dry_run,
                "fps": round(snap.fps, 1),
                "delegate": snap.delegate,
                "classifier_name": snap.classifier_name,
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
                "probabilities": snap.probabilities,
                "dictation": snap.dictation,
                "timing": snap.timing,
                "vitals": snap.vitals,
                "collector": snap.collector,
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

    def showControlCenter_(self, _sender) -> None:
        if self.window is not None:
            self.window.makeKeyAndOrderFront_(None)
            AppKit.NSApp.activateIgnoringOtherApps_(True)

    def openAccessibility_(self, _sender) -> None:
        _open_privacy_pane("Privacy_Accessibility")

    def quitApp_(self, _sender) -> None:
        AppKit.NSApp.terminate_(None)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler("/tmp/aerohand.log", encoding="utf-8"),
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
