"""
AeroHand Unified Runtime (CLI + Native macOS Background Worker):
  - AVFoundation ThreadedCamera + 8 FPS Idle Throttle (GestureCtl)
  - MediaPipe HandLandmarker with GPU Metal -> CPU auto-fallback (Airpoint)
  - HandCursorEngine + pluggable Classifier (3D Heuristic MVP / self-trained ONNX TCN)
  - 60 Hz CursorLerpInterpolator thread for silky cursor motion from 30 Hz webcam
  - Zero-permission Session-Tap Typing Guard + Failsafe release_all()
  - Thread-safe `RuntimeSnapshot` for the native macOS Control Center & Floating Camera HUD
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from aerohand.core.engine import HandCursorEngine
from aerohand.core.kinematics import CursorLerpInterpolator
from aerohand.core.types import ALL_POSE_NAMES, HandOutput, PoseName, idle_probabilities
from aerohand.ml.base import PinchClassifier
from aerohand.ml.collector import PinchDataCollector
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier
from aerohand.ml.onnx_classifier import ONNXPinchClassifier
from aerohand.platform.audio import AudioFeedback
from aerohand.platform.camera import ThreadedCamera
from aerohand.platform.input_macos import (
    MacOSInputController,
    get_current_cursor_position,
    get_main_display_bounds,
)
from aerohand.platform.permissions import is_accessibility_trusted, verify_permissions
from aerohand.platform.tracker import MediaPipeHandTracker
from aerohand.ui.overlay_renderer import render_camera_hud

logger = logging.getLogger("aerohand.runtime")


@dataclass
class RuntimeSnapshot:
    """Thread-safe snapshot consumed by the native macOS UI on the main thread."""

    running: bool = False
    paused: bool = False
    dry_run: bool = False
    is_idle: bool = False
    fps: float = 0.0
    delegate: str = "CPU"
    accessibility_ok: bool = True
    guard_reason: str = ""
    hand: Optional[str] = None
    is_pointing: bool = False
    dir_z: float = 0.0
    pose: str = "idle"
    active_pose: Optional[str] = None
    palm_wheel_state: str = "inactive"
    palm_wheel_progress: float = 0.0
    last_action: str = "Ready"
    probabilities: Dict[str, float] = field(
        default_factory=lambda: {p.value: (1.0 if p == PoseName.IDLE else 0.0) for p in ALL_POSE_NAMES}
    )
    frame_bgr: Optional[np.ndarray] = None
    error: Optional[str] = None


def build_classifier(config: Dict[str, Any]) -> PinchClassifier:
    cls_cfg = config.get("classifier") or {}
    backend = str(cls_cfg.get("backend", "heuristic_3d")).strip().lower()
    if backend == "onnx_tcn":
        mpath = cls_cfg.get("onnx_model_path")
        npath = cls_cfg.get("onnx_normalizer_path")
        if mpath and npath:
            return ONNXPinchClassifier(
                model_path=mpath,
                normalizer_path=npath,
                sequence_length=int(cls_cfg.get("sequence_length", 5)),
            )
        logger.warning("[ML] onnx_tcn selected without model/normalizer paths; falling back to heuristic_3d.")
    return Heuristic3DPinchClassifier(cls_cfg.get("heuristic_3d"))


class AeroHandRuntime:
    """End-to-end live camera -> engine -> macOS Quartz runtime."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        dry_run: bool = False,
        preview: bool = False,
        camera_index: Optional[int] = None,
        collect_dir: Optional[str | Path] = None,
        render_hud_frames: bool = False,
    ) -> None:
        self.config = config
        self.dry_run = bool(dry_run)
        self.preview = bool(preview or (collect_dir is not None))
        self.render_hud_frames = bool(render_hud_frames or self.preview)
        self.collect_dir = Path(collect_dir) if collect_dir else None

        cam_cfg = config.get("camera") or {}
        self.camera_index = int(camera_index if camera_index is not None else cam_cfg.get("index", 0))
        self.cam_width = int(cam_cfg.get("width", 640))
        self.cam_height = int(cam_cfg.get("height", 480))
        self.fps_cap = int(cam_cfg.get("fps_cap", 60))
        self.mirror = bool(cam_cfg.get("mirror", True))

        rt_cfg = config.get("runtime") or {}
        self.idle_throttle_enabled: bool = bool(rt_cfg.get("idle_throttle_enabled", True))
        self.idle_timeout_s: float = float(rt_cfg.get("idle_timeout_s", 3.0))
        self.idle_fps: float = float(rt_cfg.get("idle_fps", 8.0))
        self.max_drag_duration_ms: float = float(rt_cfg.get("max_drag_duration_ms", 20000.0))
        self.hand_lost_release_ms: float = float(rt_cfg.get("hand_lost_release_ms", 350.0))

        cur_cfg = config.get("cursor") or {}
        self.enable_lerp_60hz: bool = bool(cur_cfg.get("enable_lerp_60hz", True))
        self.lerp_rate_hz: float = float(cur_cfg.get("lerp_rate_hz", 60.0))
        self._lerp = CursorLerpInterpolator(lerp_speed=float(cur_cfg.get("lerp_speed", 55.0)))
        self._lerp_lock = threading.Lock()
        self._lerp_active: bool = False

        self.classifier = build_classifier(config)
        self.engine = HandCursorEngine(config, classifier=self.classifier)
        self.input_ctl = MacOSInputController(config.get("typing_guard"), dry_run=self.dry_run)
        self.audio = AudioFeedback(config.get("audio"))
        self.collector: Optional[PinchDataCollector] = PinchDataCollector() if self.collect_dir else None

        self._stop_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._snapshot_lock = threading.Lock()
        self._snapshot = RuntimeSnapshot(dry_run=self.dry_run)

        self._paused: bool = False
        self._drag_started_ms: Optional[float] = None
        self._last_hand_ms: float = 0.0
        self._hand_present_prev: bool = False
        self._screen_w: float = 1920.0
        self._screen_h: float = 1080.0
        self._scroll_acc_x: float = 0.0
        self._scroll_acc_y: float = 0.0
        self._last_action_text: str = "Ready"

    # ------------------------------------------------------------------
    # Live Controls (safe to call from macOS UI main thread)
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        return self._paused

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)
        if self._paused:
            with self._lerp_lock:
                self._lerp_active = False
            self._drag_started_ms = None
            self.input_ctl.release_all()
            self._last_action_text = "Paused"
        else:
            self._last_action_text = "Tracking resumed"

    def set_dry_run(self, dry_run: bool) -> None:
        self.dry_run = bool(dry_run)
        self.input_ctl.dry_run = self.dry_run
        if self.dry_run:
            self.input_ctl.release_all()
        self._last_action_text = "Mode: Dry-Run" if self.dry_run else "Mode: Live Control"

    def trigger_recenter(self) -> None:
        cx, cy = self._screen_w * 0.5, self._screen_h * 0.5
        self.engine.recenter(self._screen_w, self._screen_h)
        with self._lerp_lock:
            self._lerp.reset((cx, cy))
        if not self.dry_run:
            self.input_ctl.mouse_move(cx, cy)
        self._last_action_text = "Cursor recentered"

    def apply_live_settings(self, config: Dict[str, Any]) -> None:
        """Apply slider/checkbox tweaks in realtime without restarting the camera."""
        self.config = config
        cur = config.get("cursor") or {}
        self.enable_lerp_60hz = bool(cur.get("enable_lerp_60hz", True))
        self._lerp.lerp_speed = float(cur.get("lerp_speed", 55.0))
        self.engine.enable_pointing_gate = bool(cur.get("enable_pointing_gate", True))
        self.engine.pointing_forward_z_threshold = float(cur.get("pointing_forward_z_threshold", -0.15))

        scr = config.get("scroll") or {}
        pw = config.get("palm_wheel") or {}
        for hs in self.engine._hands.values():
            hs.kinematics.pointer_sensitivity = float(cur.get("pointer_sensitivity", 2.2))
            hs.kinematics.pointer_acceleration = float(cur.get("pointer_acceleration", 1.4))
            hs.scroll.speed = float(scr.get("speed", 1.0))
            hs.scroll.inertia_friction = float(scr.get("inertia_friction", 0.95))
            hs.scroll.min_inertia_speed = (
                float(scr.get("min_inertia_speed", 5.0)) if scr.get("enable_inertia", True) else 1e9
            )
            hs.palm_wheel.enabled = bool(pw.get("enabled", True))

        tg = config.get("typing_guard") or {}
        self.input_ctl.typing_guard_enabled = bool(tg.get("enabled", True))

        aud = config.get("audio") or {}
        self.audio.enabled = bool(aud.get("enabled", True))

    def get_snapshot(self) -> RuntimeSnapshot:
        with self._snapshot_lock:
            return self._snapshot

    def start_background(self) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stop_event.clear()
        self.preview = False  # Native AppKit displays frames on main thread; never call cv2.imshow in thread
        self.render_hud_frames = True
        self._bg_thread = threading.Thread(target=self._run_loop, name="AeroHandRuntime", daemon=True)
        self._bg_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=2.0)
        self._bg_thread = None

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _lerp_worker(self) -> None:
        period = 1.0 / max(20.0, self.lerp_rate_hz)
        last_t = time.monotonic()
        while not self._stop_event.is_set():
            time.sleep(period)
            now = time.monotonic()
            dt = now - last_t
            last_t = now
            if self._paused or not self.enable_lerp_60hz:
                continue
            with self._lerp_lock:
                if not self._lerp_active:
                    continue
                step = self._lerp.step(dt)
            if step is not None:
                nx, ny, dx, dy = step
                self.input_ctl.mouse_move(nx, ny, delta_x=dx, delta_y=dy)

    def _select_primary_hand(self, outputs: List[HandOutput]) -> Optional[HandOutput]:
        if not outputs:
            return None
        pref = str((self.config.get("hand") or {}).get("cursor_hand", "Right"))
        if pref in ("Left", "Right"):
            for o in outputs:
                if o.hand == pref:
                    return o
            return None
        for o in outputs:
            if o.pose.is_grabbing or o.pose.is_clicking:
                return o
        for o in outputs:
            if o.hand == "Right":
                return o
        return outputs[0]

    def _dispatch_palm_wheel_action(self, action: str) -> None:
        logger.info(f"[PALM WHEEL] Triggered action: {action}")
        self._last_action_text = f"PalmWheel: {action}"
        if action == "toggle_pause":
            self.set_paused(not self._paused)
        elif action == "recenter":
            self.trigger_recenter()
        elif action == "mission_control":
            self.input_ctl.toggle_mission_control()
        elif action == "spotlight":
            self.input_ctl.open_spotlight()
        elif action == "media_play_pause":
            self.input_ctl.media_play_pause()

    def _dispatch_actions(self, out: HandOutput, ts_ms: float) -> None:
        p = out.pose
        c = out.cursor

        if p.palm_wheel_action:
            self.audio.play_click_up()
            self._dispatch_palm_wheel_action(p.palm_wheel_action)
            return

        if self._paused:
            return

        suppress_cursor = p.is_scrolling or p.is_spaces_nav or p.is_window_tiling
        if c.is_pointing and c.is_valid and not suppress_cursor:
            if self.enable_lerp_60hz:
                with self._lerp_lock:
                    self._lerp.set_target(c.x, c.y)
                    self._lerp_active = True
            else:
                self.input_ctl.mouse_move(c.x, c.y, delta_x=c.delta_x, delta_y=c.delta_y)
        else:
            with self._lerp_lock:
                self._lerp_active = False

        if p.is_pinching:
            self.audio.play_click_down()

        if p.is_clicking:
            self.audio.play_click_up()
            cnt = max(1, p.click_count)
            self._last_action_text = "Double Click" if cnt == 2 else "Left Click"
            self.input_ctl.mouse_click(c.x, c.y, button="left", click_count=cnt)

        if p.is_right_clicking:
            self.audio.play_click_up()
            self._last_action_text = "Right Click"
            self.input_ctl.mouse_click(c.x, c.y, button="right", click_count=1)

        if p.grab_just_started:
            self.audio.play_click_down()
            self._drag_started_ms = ts_ms
            self._last_action_text = "Drag Start"
            self.input_ctl.mouse_down(c.x, c.y, button="left")
        elif p.grab_just_ended:
            self.audio.play_click_up()
            self._drag_started_ms = None
            self._last_action_text = "Drag Release"
            self.input_ctl.mouse_up(c.x, c.y, button="left")
        elif self._drag_started_ms is not None and (ts_ms - self._drag_started_ms) > self.max_drag_duration_ms:
            logger.warning("[SAFETY] Max drag duration exceeded; auto-releasing mouse button.")
            self._drag_started_ms = None
            self.input_ctl.mouse_up(c.x, c.y, button="left")

        scroll_use_phases = bool((self.config.get("scroll") or {}).get("use_phases", True))
        if p.is_scrolling:
            self._scroll_acc_x += p.scroll_dx
            self._scroll_acc_y += p.scroll_dy
            sx = int(self._scroll_acc_x)
            sy = int(self._scroll_acc_y)
            if sx != 0 or sy != 0:
                self._scroll_acc_x -= sx
                self._scroll_acc_y -= sy
                self._last_action_text = f"Scroll ({sx:+d}, {sy:+d})"
                self.input_ctl.scroll_pixels(sy, sx, use_phases=scroll_use_phases)
        else:
            self._scroll_acc_x = 0.0
            self._scroll_acc_y = 0.0
            self.input_ctl.end_scroll_phase()

        if p.spaces_nav_direction == "left":
            self._last_action_text = "Space Left"
            self.input_ctl.switch_space("left")
        elif p.spaces_nav_direction == "right":
            self._last_action_text = "Space Right"
            self.input_ctl.switch_space("right")
        elif p.spaces_nav_direction == "top":
            self._last_action_text = "Mission Control"
            self.input_ctl.toggle_mission_control()
        elif p.spaces_nav_direction == "bottom":
            self._last_action_text = "App Exposé"
            self.input_ctl.exit_mission_control()

        if p.window_tile_direction is not None:
            self._last_action_text = f"Tile Window: {p.window_tile_direction}"
            self.input_ctl.tile_focused_window(p.window_tile_direction)

    def _run_loop(self) -> None:
        acc_ok = True if self.dry_run else is_accessibility_trusted(prompt_user=False)
        _, _, self._screen_w, self._screen_h = get_main_display_bounds()
        cur_x, cur_y = get_current_cursor_position()
        self.engine.anchor_cursor("Right", cur_x, cur_y)
        self.engine.anchor_cursor("Left", cur_x, cur_y)
        with self._lerp_lock:
            self._lerp.reset((cur_x, cur_y))

        tracker: Optional[MediaPipeHandTracker] = None
        camera: Optional[ThreadedCamera] = None
        lerp_thread: Optional[threading.Thread] = None

        try:
            trk_cfg = self.config.get("tracker") or {}
            tracker = MediaPipeHandTracker(
                model_path=str(trk_cfg.get("model_path", "models/hand_landmarker.task")),
                num_hands=int(trk_cfg.get("num_hands", 1)),
                min_hand_detection_confidence=float(trk_cfg.get("min_hand_detection_confidence", 0.55)),
                min_hand_presence_confidence=float(trk_cfg.get("min_hand_presence_confidence", 0.55)),
                min_tracking_confidence=float(trk_cfg.get("min_tracking_confidence", 0.55)),
                delegate=str(trk_cfg.get("delegate", "GPU")),
                mirror_handedness=self.mirror,
            )
            camera = ThreadedCamera(
                camera_index=self.camera_index,
                width=self.cam_width,
                height=self.cam_height,
                fps_cap=self.fps_cap,
                mirror=self.mirror,
            )

            lerp_thread = threading.Thread(target=self._lerp_worker, name="AeroHandLerp60Hz", daemon=True)
            lerp_thread.start()

            logger.info(
                f"[AeroHand] Started ({'DRY-RUN' if self.dry_run else 'LIVE'} | "
                f"delegate={tracker.delegate_used} | screen={int(self._screen_w)}x{int(self._screen_h)})"
            )

            camera.start()
            last_hand_wall = time.monotonic()
            fps_ema = 30.0
            prev_wall = time.monotonic()
            last_acc_check = time.monotonic()

            while not self._stop_event.is_set():
                now_wall = time.monotonic()
                if now_wall - last_acc_check >= 2.0:
                    acc_ok = True if self.dry_run else is_accessibility_trusted(prompt_user=False)
                    last_acc_check = now_wall

                is_idle = self.idle_throttle_enabled and ((now_wall - last_hand_wall) >= self.idle_timeout_s)
                if is_idle:
                    time.sleep(max(0.01, 1.0 / max(1.0, self.idle_fps) - 0.015))

                cf = camera.read(timeout_s=0.04)
                if not cf.success or cf.frame is None:
                    continue

                dt_wall = max(1e-4, time.monotonic() - prev_wall)
                prev_wall = time.monotonic()
                fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / dt_wall)

                hands = tracker.process_frame(cf.frame, cf.timestamp_ms)
                if hands:
                    last_hand_wall = time.monotonic()
                    self._last_hand_ms = float(cf.timestamp_ms)
                    if not self._hand_present_prev:
                        cx, cy = get_current_cursor_position()
                        for h_inp in hands:
                            self.engine.anchor_cursor(h_inp.handedness, cx, cy)
                        with self._lerp_lock:
                            self._lerp.reset((cx, cy))
                    self._hand_present_prev = True

                    if self.collector is not None:
                        h0 = hands[0]
                        self.collector.add_frame(
                            h0.landmarks, h0.world_landmarks, h0.handedness, h0.confidence, cf.timestamp_ms
                        )
                else:
                    if self._hand_present_prev:
                        self._hand_present_prev = False
                        with self._lerp_lock:
                            self._lerp_active = False
                    if (
                        self._drag_started_ms is not None
                        and (float(cf.timestamp_ms) - self._last_hand_ms) >= self.hand_lost_release_ms
                    ):
                        self._drag_started_ms = None
                        self.input_ctl.release_all()

                outputs = self.engine.process_frame(hands, float(cf.timestamp_ms), self._screen_w, self._screen_h)
                primary = self._select_primary_hand(outputs)
                if primary is not None:
                    self._dispatch_actions(primary, float(cf.timestamp_ms))

                vis_frame: Optional[np.ndarray] = None
                if self.render_hud_frames:
                    vis_frame = render_camera_hud(
                        cf.frame,
                        hands,
                        primary,
                        fps=fps_ema,
                        paused=self._paused,
                        dry_run=self.dry_run,
                        is_idle=is_idle,
                        guard_reason=self.input_ctl.last_guard_reason,
                        compact=not self.preview,
                    )

                probs_map = (
                    {k.value: float(v) for k, v in primary.pose.pose_probabilities.items()}
                    if primary is not None
                    else {k.value: float(v) for k, v in idle_probabilities().items()}
                )
                snap = RuntimeSnapshot(
                    running=True,
                    paused=self._paused,
                    dry_run=self.dry_run,
                    is_idle=is_idle,
                    fps=fps_ema,
                    delegate=tracker.delegate_used,
                    accessibility_ok=acc_ok,
                    guard_reason=self.input_ctl.last_guard_reason,
                    hand=primary.hand if primary is not None else None,
                    is_pointing=primary.cursor.is_pointing if primary is not None else False,
                    dir_z=primary.cursor.pointing_dir_z if primary is not None else 0.0,
                    pose=primary.pose.pose.value if primary is not None else "idle",
                    active_pose=(
                        primary.pose.active_pose.value
                        if (primary is not None and primary.pose.active_pose is not None)
                        else None
                    ),
                    palm_wheel_state=primary.pose.palm_wheel_state if primary is not None else "inactive",
                    palm_wheel_progress=primary.pose.palm_wheel_progress if primary is not None else 0.0,
                    last_action=self._last_action_text,
                    probabilities=probs_map,
                    frame_bgr=vis_frame,
                )
                with self._snapshot_lock:
                    self._snapshot = snap

                if self.preview and vis_frame is not None:
                    if self.collector is not None:
                        rec_tag = "REC ON" if self.collector.is_recording else "REC OFF"
                        col_col = (60, 80, 255) if self.collector.is_recording else (180, 180, 180)
                        cv2.putText(
                            vis_frame,
                            f"[COLLECT {rec_tag}] Label(0-6): {self.collector.current_pose.value} | r:rec s:save",
                            (10, vis_frame.shape[0] - 36),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.44,
                            col_col,
                            1,
                        )
                    cv2.imshow("AeroHand Preview (q to quit)", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if self.collector is not None:
                        if key == ord("r"):
                            st = self.collector.toggle_recording()
                            print(f"[COLLECT] recording={'ON' if st else 'OFF'} counts={self.collector.counts()}")
                        elif key == ord("s") and self.collect_dir is not None:
                            fp = self.collector.save(self.collect_dir)
                            print(f"[COLLECT] Saved -> {fp} ({self.collector.counts()})")
                        elif ord("0") <= key <= ord("6"):
                            idx = key - ord("0")
                            if idx < len(ALL_POSE_NAMES):
                                self.collector.set_pose(ALL_POSE_NAMES[idx])
        except KeyboardInterrupt:
            pass
        except Exception as exc:
            logger.exception(f"[AeroHand] Runtime error: {exc}")
            with self._snapshot_lock:
                self._snapshot = RuntimeSnapshot(running=False, error=str(exc), last_action=f"Error: {exc}")
        finally:
            self._stop_event.set()
            if lerp_thread is not None and lerp_thread.is_alive():
                lerp_thread.join(timeout=0.5)
            self.input_ctl.release_all()
            if camera is not None:
                camera.stop()
            if tracker is not None:
                tracker.close()
            if self.preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if (
                self.collector is not None
                and self.collect_dir is not None
                and sum(self.collector.counts().values()) > 0
            ):
                fp = self.collector.save(self.collect_dir)
                print(f"[COLLECT] Auto-saved on exit -> {fp}")
            with self._snapshot_lock:
                self._snapshot.running = False
            logger.info("[AeroHand] Stopped cleanly (release_all executed).")

    def run(self) -> None:
        ok, msg = verify_permissions(dry_run=self.dry_run)
        if not ok:
            logger.error(msg)
            print(msg)
            return
        self._run_loop()
