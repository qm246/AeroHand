"""
Native macOS Desktop Application for AeroHand (Airpoint-style Dark Glass UI + GestureCtl Safety).

Features:
  - Menu Bar controller (`NSStatusBar`) with live status dot & quick actions
  - Airpoint-style Dark Glass Control Center (`NSWindow` + `NSVisualEffectView`)
    with live telemetry, real-time tuning sliders/toggles (saved as override-only YAML),
    and gesture cheat-sheet
  - Floating Camera Pill (`NSPanel` at bottom-right, draggable, rounded 18px, always-on-top)
    rendering anti-aliased 21-landmark skeleton, pointing gate, and Palm Wheel progress ring
  - Single-instance PID lock so re-clicking `AeroHand.app` surfaces the Control Center cleanly
"""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import AppKit
import cv2
import Foundation
import numpy as np
import objc

from aerohand import __version__
from aerohand.config import default_config_path, load_config, save_user_overrides
from aerohand.platform.permissions import is_accessibility_trusted
from aerohand.runtime import AeroHandRuntime, RuntimeSnapshot

logger = logging.getLogger("aerohand.app")

_LOCK_PATH = "/tmp/aerohand_app.lock"
_SHOW_TRIGGER_PATH = "/tmp/aerohand_show_window"
_LOCK_FD = None


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
        None,
        w,
        h,
        8,
        4,
        True,
        False,
        AppKit.NSCalibratedRGBColorSpace,
        w * 4,
        32,
    )
    if rep is None:
        return None
    buf = rep.bitmapData()
    buf[: len(raw_bytes)] = raw_bytes
    img = AppKit.NSImage.alloc().initWithSize_(Foundation.NSMakeSize(w, h))
    img.addRepresentation_(rep)
    return img


def _make_label(
    frame: Any,
    text: str,
    *,
    size: float = 12.0,
    bold: bool = False,
    color: Optional[Any] = None,
    align: int = AppKit.NSTextAlignmentLeft,
) -> AppKit.NSTextField:
    lbl = AppKit.NSTextField.alloc().initWithFrame_(frame)
    lbl.setStringValue_(text)
    lbl.setBezeled_(False)
    lbl.setDrawsBackground_(False)
    lbl.setEditable_(False)
    lbl.setSelectable_(False)
    lbl.setAlignment_(align)
    font = AppKit.NSFont.boldSystemFontOfSize_(size) if bold else AppKit.NSFont.systemFontOfSize_(size)
    lbl.setFont_(font)
    lbl.setTextColor_(color or AppKit.NSColor.labelColor())
    return lbl


def _open_privacy_pane(pane: str) -> None:
    subprocess.Popen(
        ["open", f"x-apple.systempreferences:com.apple.preference.security?{pane}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class AeroHandAppDelegate(AppKit.NSObject):
    """Main-thread AppKit controller hosting Menu Bar, Control Center, and Floating Camera HUD."""

    def initWithConfig_(self, config: Dict[str, Any]):
        self = objc.super(AeroHandAppDelegate, self).init()
        if self is None:
            return None
        self.config: Dict[str, Any] = config
        self.runtime: Optional[AeroHandRuntime] = None
        self.status_item = None
        self.window = None
        self.camera_panel = None
        self.camera_image_view = None
        self.camera_hud_visible: bool = True
        self.ui_timer = None
        self._widgets: Dict[str, Any] = {}
        self._menu_items: Dict[str, Any] = {}
        return self

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

        # Prompt Accessibility smoothly on launch if needed; start in Live if granted else Dry-Run safely
        acc_ok = is_accessibility_trusted(prompt_user=True)
        start_dry_run = not acc_ok

        self.runtime = AeroHandRuntime(
            self.config,
            dry_run=start_dry_run,
            preview=False,
            render_hud_frames=True,
        )

        self._build_status_bar()
        self._build_control_center()
        self._build_floating_camera_panel()
        self.runtime.start_background()

        if self.camera_hud_visible:
            self.camera_panel.orderFront_(None)
        self.window.makeKeyAndOrderFront_(None)
        AppKit.NSApp.activateIgnoringOtherApps_(True)

        # 20 Hz UI refresh timer (keeps CPU ultra-low while floating camera & telemetry look fluid)
        self.ui_timer = Foundation.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.05, self, "onUiTick:", None, True
        )

    def applicationShouldHandleReopen_hasVisibleWindows_(self, _sender, _flag) -> bool:
        self.showControlCenter_(None)
        return True

    def applicationWillTerminate_(self, _notification) -> None:
        if self.ui_timer is not None:
            self.ui_timer.invalidate()
            self.ui_timer = None
        if self.runtime is not None:
            self.runtime.stop()

    # ------------------------------------------------------------------
    # Menu Bar
    # ------------------------------------------------------------------

    def _build_status_bar(self) -> None:
        bar = AppKit.NSStatusBar.systemStatusBar()
        self.status_item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
        btn = self.status_item.button()
        if btn is not None:
            btn.setTitle_("✦ AeroHand")

        menu = AppKit.NSMenu.alloc().init()

        def _add_item(key: str, title: str, action: str, key_equiv: str = "") -> None:
            item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, key_equiv)
            item.setTarget_(self)
            menu.addItem_(item)
            self._menu_items[key] = item

        _add_item("status", "Status: Starting...", "showControlCenter:")
        self._menu_items["status"].setEnabled_(False)
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        _add_item("toggle_pause", "Pause Tracking", "togglePause:")
        _add_item("toggle_mode", "Switch to Live Control", "toggleDryRun:")
        _add_item("toggle_cam", "Hide Floating Camera HUD", "toggleCameraHud:")
        _add_item("recenter", "Recenter Cursor", "recenterCursor:")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        _add_item("open_cc", "Open Control Center", "showControlCenter:", ",")
        _add_item("open_acc", "Open Accessibility Settings...", "openAccessibility:")
        menu.addItem_(AppKit.NSMenuItem.separatorItem())
        _add_item("quit", "Quit AeroHand", "quitApp:", "q")

        self.status_item.setMenu_(menu)

    # ------------------------------------------------------------------
    # Airpoint-Style Floating Camera Pill (Bottom-Right)
    # ------------------------------------------------------------------

    def _build_floating_camera_panel(self) -> None:
        pw, ph = 300, 210
        screen = AppKit.NSScreen.mainScreen()
        sf = screen.visibleFrame() if screen else Foundation.NSMakeRect(0, 0, 1440, 900)
        rect = Foundation.NSMakeRect(sf.origin.x + sf.size.width - pw - 22, sf.origin.y + 22, pw, ph)

        style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, AppKit.NSBackingStoreBuffered, False
        )
        panel.setLevel_(AppKit.NSFloatingWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setMovableByWindowBackground_(True)
        panel.setAlphaValue_(0.90)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        container = AppKit.NSView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, pw, ph))
        container.setWantsLayer_(True)
        layer = container.layer()
        layer.setCornerRadius_(18.0)
        layer.setMasksToBounds_(True)
        layer.setBorderWidth_(1.5)
        layer.setBorderColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.35, 0.72, 0.98, 0.65).CGColor())
        layer.setBackgroundColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.06, 0.08, 0.12, 0.92).CGColor())

        img_view = AppKit.NSImageView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, pw, ph))
        img_view.setImageScaling_(AppKit.NSImageScaleAxesIndependently)
        container.addSubview_(img_view)

        panel.setContentView_(container)
        self.camera_panel = panel
        self.camera_image_view = img_view

    # ------------------------------------------------------------------
    # Airpoint-Style Dark Glass Control Center
    # ------------------------------------------------------------------

    def _build_control_center(self) -> None:
        w, h = 540, 690
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
            | AppKit.NSWindowStyleMaskFullSizeContentView
        )
        win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, AppKit.NSBackingStoreBuffered, False
        )
        win.setTitle_("AeroHand Control Center")
        win.setTitlebarAppearsTransparent_(True)
        win.setTitleVisibility_(AppKit.NSWindowTitleHidden)
        win.setReleasedWhenClosed_(False)
        try:
            win.setAppearance_(AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua))
        except Exception:
            pass

        effect = AppKit.NSVisualEffectView.alloc().initWithFrame_(Foundation.NSMakeRect(0, 0, w, h))
        effect.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
        effect.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
        effect.setState_(AppKit.NSVisualEffectStateActive)
        win.setContentView_(effect)

        c_white = AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.96, 1.0)
        c_muted = AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.68, 1.0)
        c_teal = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.18, 0.85, 0.75, 1.0)

        # Header
        effect.addSubview_(
            _make_label(Foundation.NSMakeRect(24, h - 56, 260, 28), f"✦ AeroHand v{__version__}", size=20, bold=True, color=c_white)
        )
        effect.addSubview_(
            _make_label(
                Foundation.NSMakeRect(24, h - 76, 360, 18),
                "Airpoint Ergonomics × GestureCtl macOS Safety",
                size=11.5,
                color=c_muted,
            )
        )
        badge = _make_label(
            Foundation.NSMakeRect(w - 190, h - 54, 166, 22),
            "STARTING...",
            size=12,
            bold=True,
            color=c_teal,
            align=AppKit.NSTextAlignmentRight,
        )
        effect.addSubview_(badge)
        self._widgets["badge"] = badge

        # Primary Action Buttons Row
        y = h - 120
        btn_specs = [
            ("btn_pause", "Pause Tracking", "togglePause:", 24, 130),
            ("btn_mode", "Mode: Live", "toggleDryRun:", 164, 130),
            ("btn_cam", "Camera HUD: ON", "toggleCameraHud:", 304, 126),
            ("btn_recenter", "Recenter", "recenterCursor:", 440, 76),
        ]
        for key, title, sel, bx, bw in btn_specs:
            b = AppKit.NSButton.alloc().initWithFrame_(Foundation.NSMakeRect(bx, y, bw, 32))
            b.setTitle_(title)
            b.setBezelStyle_(AppKit.NSBezelStyleRounded)
            b.setTarget_(self)
            b.setAction_(sel)
            effect.addSubview_(b)
            self._widgets[key] = b

        # Telemetry Card
        y -= 102
        card = AppKit.NSView.alloc().initWithFrame_(Foundation.NSMakeRect(24, y, w - 48, 88))
        card.setWantsLayer_(True)
        card.layer().setCornerRadius_(12.0)
        card.layer().setBackgroundColor_(AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.10, 0.65).CGColor())
        card.layer().setBorderWidth_(1.0)
        card.layer().setBorderColor_(AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.12).CGColor())

        tel1 = _make_label(Foundation.NSMakeRect(14, 56, w - 76, 20), "Tracker: GPU Metal | FPS: -- | Guard: OK", size=12, bold=True, color=c_white)
        tel2 = _make_label(Foundation.NSMakeRect(14, 32, w - 76, 18), "Hand: None | Pointing Gate (dirZ): -- | Pose: idle", size=11.5, color=c_teal)
        tel3 = _make_label(Foundation.NSMakeRect(14, 10, w - 76, 18), "Last Action: Ready", size=11.5, color=c_muted)
        card.addSubview_(tel1)
        card.addSubview_(tel2)
        card.addSubview_(tel3)
        effect.addSubview_(card)
        self._widgets["tel1"] = tel1
        self._widgets["tel2"] = tel2
        self._widgets["tel3"] = tel3

        # Tuning Sliders
        y -= 28
        effect.addSubview_(_make_label(Foundation.NSMakeRect(24, y, 300, 18), "REAL-TIME TUNING (AUTO-SAVED)", size=11, bold=True, color=c_muted))

        cur_cfg = self.config.get("cursor") or {}
        scr_cfg = self.config.get("scroll") or {}
        sliders = [
            ("sens", "Pointer Sensitivity", 0.8, 4.5, float(cur_cfg.get("pointer_sensitivity", 2.2))),
            ("accel", "Pointer Acceleration", 0.0, 2.5, float(cur_cfg.get("pointer_acceleration", 1.4))),
            ("scroll_spd", "Scroll Speed", 0.3, 2.5, float(scr_cfg.get("speed", 1.0))),
            ("dirz", "Pointing Gate (dirZ)", -0.30, 0.05, float(cur_cfg.get("pointing_forward_z_threshold", -0.15))),
        ]
        for key, title, lo, hi, val in sliders:
            y -= 34
            effect.addSubview_(_make_label(Foundation.NSMakeRect(24, y + 2, 165, 20), title, size=12, color=c_white))
            sld = AppKit.NSSlider.alloc().initWithFrame_(Foundation.NSMakeRect(195, y, 250, 22))
            sld.setMinValue_(lo)
            sld.setMaxValue_(hi)
            sld.setDoubleValue_(val)
            sld.setContinuous_(True)
            sld.setTarget_(self)
            sld.setAction_("onSettingChanged:")
            effect.addSubview_(sld)
            vlbl = _make_label(Foundation.NSMakeRect(452, y + 2, 64, 20), f"{val:.2f}", size=12, bold=True, color=c_teal, align=AppKit.NSTextAlignmentRight)
            effect.addSubview_(vlbl)
            self._widgets[f"sld_{key}"] = sld
            self._widgets[f"val_{key}"] = vlbl

        # Feature Toggles (2 columns x 3 rows)
        y -= 18
        tg_cfg = self.config.get("typing_guard") or {}
        pw_cfg = self.config.get("palm_wheel") or {}
        aud_cfg = self.config.get("audio") or {}
        toggles = [
            ("chk_lerp", "60 Hz Smooth Lerp", bool(cur_cfg.get("enable_lerp_60hz", True))),
            ("chk_gate", "3D Pointing Gate", bool(cur_cfg.get("enable_pointing_gate", True))),
            ("chk_guard", "Typing Guard (0-Perm)", bool(tg_cfg.get("enabled", True))),
            ("chk_inertia", "Scroll Inertia", True),
            ("chk_pw", "Palm Wheel Menu (350ms)", bool(pw_cfg.get("enabled", True))),
            ("chk_audio", "Click Audio Feedback", bool(aud_cfg.get("enabled", True))),
        ]
        for idx, (key, title, state) in enumerate(toggles):
            col = idx % 2
            row = idx // 2
            cx = 24 + col * 255
            cy = y - (row + 1) * 26
            chk = AppKit.NSButton.alloc().initWithFrame_(Foundation.NSMakeRect(cx, cy, 240, 22))
            chk.setButtonType_(AppKit.NSButtonTypeSwitch)
            chk.setTitle_(title)
            chk.setState_(AppKit.NSControlStateValueOn if state else AppKit.NSControlStateValueOff)
            chk.setTarget_(self)
            chk.setAction_("onSettingChanged:")
            effect.addSubview_(chk)
            self._widgets[key] = chk

        # Gesture Guide Card
        guide_y = 64
        guide_h = 190
        gcard = AppKit.NSView.alloc().initWithFrame_(Foundation.NSMakeRect(24, guide_y, w - 48, guide_h))
        gcard.setWantsLayer_(True)
        gcard.layer().setCornerRadius_(12.0)
        gcard.layer().setBackgroundColor_(AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.55).CGColor())
        gcard.layer().setBorderWidth_(1.0)
        gcard.layer().setBorderColor_(AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.10).CGColor())

        gcard.addSubview_(_make_label(Foundation.NSMakeRect(14, guide_h - 28, 300, 18), "GESTURE QUICK GUIDE (RIGHT HAND)", size=11, bold=True, color=c_teal))
        guide_lines = [
            "• Point Index toward screen        → Move Cursor (60 Hz smooth)",
            "• Tap Thumb + Middle               → Left Click / Double Click",
            "• Hold Thumb + Index + Middle      → Click & Drag",
            "• Hold Thumb + Ring (still)        → Right Click (on release)",
            "• Hold Thumb + Ring + Move         → 2D Scroll with Inertia",
            "• Hold Thumb + Pinky Base + Slide  → Switch Spaces / Mission Control",
            "• Hold Thumb + Pinky Tip + Slide   → Tile Window (Left/Right/Full)",
            "• Open Palm steady 350ms + Curl 1  → Palm Wheel Quick Menu",
        ]
        for i, line in enumerate(guide_lines):
            gcard.addSubview_(
                _make_label(
                    Foundation.NSMakeRect(14, guide_h - 50 - i * 18, w - 76, 17),
                    line,
                    size=11,
                    color=c_white,
                )
            )
        effect.addSubview_(gcard)

        # Footer buttons
        footer_btns = [
            ("Reset Defaults", "resetDefaults:", 24, 120),
            ("Accessibility Settings", "openAccessibility:", 154, 155),
            ("Camera Settings", "openCameraSettings:", 319, 125),
            ("Quit", "quitApp:", 454, 62),
        ]
        for title, sel, fx, fw in footer_btns:
            b = AppKit.NSButton.alloc().initWithFrame_(Foundation.NSMakeRect(fx, 18, fw, 30))
            b.setTitle_(title)
            b.setBezelStyle_(AppKit.NSBezelStyleRounded)
            b.setTarget_(self)
            b.setAction_(sel)
            effect.addSubview_(b)

        self.window = win

    # ------------------------------------------------------------------
    # Actions & Periodic UI Sync
    # ------------------------------------------------------------------

    def onUiTick_(self, _timer) -> None:
        # Check single-instance surface signal
        if os.path.exists(_SHOW_TRIGGER_PATH):
            try:
                os.unlink(_SHOW_TRIGGER_PATH)
            except OSError:
                pass
            self.showControlCenter_(None)

        if self.runtime is None:
            return
        snap: RuntimeSnapshot = self.runtime.get_snapshot()

        # Status bar title
        btn = self.status_item.button()
        if btn is not None:
            if snap.paused:
                btn.setTitle_("⏸ AeroHand")
            elif snap.guard_reason:
                btn.setTitle_("⌨ AeroHand")
            elif snap.dry_run:
                btn.setTitle_("◌ AeroHand")
            else:
                btn.setTitle_("✦ AeroHand")

        # Badge & buttons
        badge = self._widgets.get("badge")
        if badge is not None:
            if snap.error:
                badge.setStringValue_("ERROR")
                badge.setTextColor_(AppKit.NSColor.systemRedColor())
            elif snap.paused:
                badge.setStringValue_("PAUSED")
                badge.setTextColor_(AppKit.NSColor.systemYellowColor())
            elif not snap.accessibility_ok and not snap.dry_run:
                badge.setStringValue_("NEED ACCESSIBILITY")
                badge.setTextColor_(AppKit.NSColor.systemOrangeColor())
            elif snap.dry_run:
                badge.setStringValue_("DRY-RUN (SAFE)")
                badge.setTextColor_(AppKit.NSColor.systemYellowColor())
            else:
                badge.setStringValue_("LIVE • ACTIVE")
                badge.setTextColor_(AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(0.18, 0.90, 0.55, 1.0))

        self._widgets["btn_pause"].setTitle_("Resume Tracking" if snap.paused else "Pause Tracking")
        self._widgets["btn_mode"].setTitle_("Mode: Dry-Run" if snap.dry_run else "Mode: Live")
        self._widgets["btn_cam"].setTitle_("Camera HUD: ON" if self.camera_hud_visible else "Camera HUD: OFF")

        self._menu_items["status"].setTitle_(
            f"Status: {'Paused' if snap.paused else ('Dry-Run' if snap.dry_run else 'Live')} • {snap.fps:.0f} FPS ({snap.delegate})"
        )
        self._menu_items["toggle_pause"].setTitle_("Resume Tracking" if snap.paused else "Pause Tracking")
        self._menu_items["toggle_mode"].setTitle_("Switch to Live Control" if snap.dry_run else "Switch to Dry-Run")
        self._menu_items["toggle_cam"].setTitle_("Hide Floating Camera HUD" if self.camera_hud_visible else "Show Floating Camera HUD")

        # Telemetry labels
        guard_str = snap.guard_reason if snap.guard_reason else ("Active" if self.runtime.input_ctl.typing_guard_enabled else "Off")
        acc_str = "OK" if snap.accessibility_ok else "Missing!"
        self._widgets["tel1"].setStringValue_(
            f"Tracker: {snap.delegate} Metal  |  FPS: {snap.fps:4.1f}  |  Typing Guard: {guard_str}  |  Access: {acc_str}"
        )
        gate_txt = "OPEN" if snap.is_pointing else "BLOCKED"
        self._widgets["tel2"].setStringValue_(
            f"Hand: {snap.hand or 'None'}  |  Pointing dirZ: {snap.dir_z:+.2f} ({gate_txt})  |  Pose: {snap.active_pose or snap.pose}"
        )
        self._widgets["tel3"].setStringValue_(
            f"Last Action: {snap.last_action}  |  PalmWheel: {snap.palm_wheel_state}"
        )

        # Floating Camera HUD frame update
        if self.camera_hud_visible and snap.frame_bgr is not None and self.camera_image_view is not None:
            ns_img = _bgr_to_nsimage(snap.frame_bgr, 300, 210)
            if ns_img is not None:
                self.camera_image_view.setImage_(ns_img)

    def onSettingChanged_(self, _sender) -> None:
        if self.runtime is None:
            return
        sens = float(self._widgets["sld_sens"].doubleValue())
        accel = float(self._widgets["sld_accel"].doubleValue())
        sc_spd = float(self._widgets["sld_scroll_spd"].doubleValue())
        dirz = float(self._widgets["sld_dirz"].doubleValue())

        self._widgets["val_sens"].setStringValue_(f"{sens:.2f}")
        self._widgets["val_accel"].setStringValue_(f"{accel:.2f}")
        self._widgets["val_scroll_spd"].setStringValue_(f"{sc_spd:.2f}")
        self._widgets["val_dirz"].setStringValue_(f"{dirz:.2f}")

        self.config.setdefault("cursor", {}).update(
            {
                "pointer_sensitivity": round(sens, 2),
                "pointer_acceleration": round(accel, 2),
                "pointing_forward_z_threshold": round(dirz, 2),
                "enable_lerp_60hz": self._widgets["chk_lerp"].state() == AppKit.NSControlStateValueOn,
                "enable_pointing_gate": self._widgets["chk_gate"].state() == AppKit.NSControlStateValueOn,
            }
        )
        self.config.setdefault("scroll", {}).update(
            {
                "speed": round(sc_spd, 2),
                "enable_inertia": self._widgets["chk_inertia"].state() == AppKit.NSControlStateValueOn,
            }
        )
        self.config.setdefault("typing_guard", {})["enabled"] = (
            self._widgets["chk_guard"].state() == AppKit.NSControlStateValueOn
        )
        self.config.setdefault("palm_wheel", {})["enabled"] = (
            self._widgets["chk_pw"].state() == AppKit.NSControlStateValueOn
        )
        self.config.setdefault("audio", {})["enabled"] = (
            self._widgets["chk_audio"].state() == AppKit.NSControlStateValueOn
        )

        self.runtime.apply_live_settings(self.config)
        try:
            save_user_overrides(self.config)
        except Exception as exc:
            logger.debug(f"[UI] Could not persist overrides: {exc}")

    def resetDefaults_(self, _sender) -> None:
        fresh = load_config(config_path=default_config_path(), overrides_path="/nonexistent_overrides.yaml")
        self.config = fresh
        cur = fresh.get("cursor") or {}
        scr = fresh.get("scroll") or {}
        self._widgets["sld_sens"].setDoubleValue_(float(cur.get("pointer_sensitivity", 2.2)))
        self._widgets["sld_accel"].setDoubleValue_(float(cur.get("pointer_acceleration", 1.4)))
        self._widgets["sld_scroll_spd"].setDoubleValue_(float(scr.get("speed", 1.0)))
        self._widgets["sld_dirz"].setDoubleValue_(float(cur.get("pointing_forward_z_threshold", -0.15)))
        self._widgets["chk_lerp"].setState_(AppKit.NSControlStateValueOn)
        self._widgets["chk_gate"].setState_(AppKit.NSControlStateValueOn)
        self._widgets["chk_guard"].setState_(AppKit.NSControlStateValueOn)
        self._widgets["chk_inertia"].setState_(AppKit.NSControlStateValueOn)
        self._widgets["chk_pw"].setState_(AppKit.NSControlStateValueOn)
        self._widgets["chk_audio"].setState_(AppKit.NSControlStateValueOn)
        self.onSettingChanged_(None)

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
        self.camera_hud_visible = not self.camera_hud_visible
        if self.camera_panel is not None:
            if self.camera_hud_visible:
                self.camera_panel.orderFront_(None)
            else:
                self.camera_panel.orderOut_(None)

    def recenterCursor_(self, _sender) -> None:
        if self.runtime is not None:
            self.runtime.trigger_recenter()

    def showControlCenter_(self, _sender) -> None:
        if self.window is not None:
            self.window.makeKeyAndOrderFront_(None)
            AppKit.NSApp.activateIgnoringOtherApps_(True)

    def openAccessibility_(self, _sender) -> None:
        _open_privacy_pane("Privacy_Accessibility")

    def openCameraSettings_(self, _sender) -> None:
        _open_privacy_pane("Privacy_Camera")

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
