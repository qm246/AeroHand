"""
AeroHand Unified Runtime (v0.2.0 — Full Airpoint Stage 1–3 Parity + GestureCtl Safety):
  - Multi-camera AVFoundation capture + luminance monitoring + 8 FPS Idle Throttle
  - MediaPipe 2-Hand GPU Metal -> CPU auto-fallback
  - HandCursorEngine: Dual OneEuro, 60 Hz Lerp, 3D Pointing Gate, Two-Hand Zoom,
    Left & Right Palm Wheel with nested Submenus, Window Tile Ghost Preview, Momentum Scroll
  - Native macOS AXUIElement Window Tiling + Session-Tap 0-Permission Typing Guard
  - Local macOS Voice Dictation (Vietnamese `vi-VN` & English `en-US`)
  - Per-App Dynamic Config Profiles (frontmost app bundle ID detection)
  - Local HTTP Control API (`127.0.0.1:17420`) for CLI & Apple Shortcuts
  - Built-in Interactive Data Collector Studio & Onboarding Practice Tracker
  - Thread-safe `RuntimeSnapshot` + screen overlay event stream for WKWebView UI & Overlays
"""

from __future__ import annotations

import logging
import math
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from aerohand import __version__
from aerohand.config import (
    repo_root,
    resolve_profile_config,
)
from aerohand.core.engine import (
    HandCursorEngine,
    PinchCalibrationWizard,
    format_live_hint_toast,
    format_refusal_reason,
)
from aerohand.core.kinematics import CursorLerpInterpolator
from aerohand.core.types import ALL_POSE_NAMES, HandOutput, PoseName
from aerohand.ml.base import PinchClassifier
from aerohand.ml.collector import (
    FrameRingBuffer,
    PassiveIdleRecorder,
    PinchDataCollector,
    QualityGateError,
    clean_recording_samples,
    compute_corpus_condition_matrix,
)
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier
from aerohand.ml.onnx_classifier import GuardedClassifier, ONNXPinchClassifier, resolve_classifier_mode
from aerohand.ml.provenance import compute_heuristic_source_sha256
from aerohand.platform.audio import AudioFeedback
from aerohand.platform.camera import ThreadedCamera, list_cameras
from aerohand.platform.control_api import LocalControlAPIServer
from aerohand.platform.dictation import VoiceDictationController
from aerohand.platform.input_macos import (
    MacOSInputController,
    ScreenGeometry,
    compute_tile_rect,
    get_current_cursor_position,
    get_frontmost_app_bundle_id,
)
from aerohand.platform.permissions import check_all_permissions, verify_permissions
from aerohand.platform.tracker import MediaPipeHandTracker
from aerohand.platform.vitals import SystemVitalsMonitor
from aerohand.ui.overlay_renderer import render_camera_hud

logger = logging.getLogger("aerohand.runtime")


@dataclass
class RuntimeSnapshot:
    """Thread-safe snapshot consumed by the native macOS UI, Screen Overlay, and Control API."""

    running: bool = False
    paused: bool = False
    dry_run: bool = False
    is_idle: bool = False
    fps: float = 0.0
    delegate: str = "CPU"
    classifier_name: str = "Heuristic3DPinchClassifier"
    classifier_telemetry: Dict[str, Any] = field(default_factory=dict)
    accessibility_ok: bool = True
    permissions: Dict[str, bool] = field(default_factory=dict)
    guard_reason: str = ""
    active_app: Optional[str] = None
    hand: Optional[str] = None
    hands_count: int = 0
    is_pointing: bool = False
    dir_z: float = 0.0
    cursor_x: float = 0.0
    cursor_y: float = 0.0
    pose: str = "idle"
    active_pose: Optional[str] = None
    is_scrolling: bool = False
    is_grabbing: bool = False
    is_zooming: bool = False
    palm_wheel_state: str = "inactive"
    palm_wheel_progress: float = 0.0
    palm_wheels: List[Dict[str, Any]] = field(default_factory=list)
    window_tile_preview: Optional[Dict[str, Any]] = None
    spaces_nav_preview: Optional[str] = None
    active_hint: Optional[str] = None
    active_hint_text: Optional[str] = None
    refusal_code: Optional[str] = None
    refusal_reason: Optional[str] = None
    warning: Optional[Dict[str, str]] = None
    dictation: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    vitals: Dict[str, Any] = field(default_factory=dict)
    collector: Dict[str, Any] = field(default_factory=dict)
    calibration: Dict[str, Any] = field(default_factory=dict)
    practice: Dict[str, bool] = field(default_factory=dict)
    click_event_seq: int = 0
    click_event: Optional[Dict[str, Any]] = None
    last_action: str = "Ready"
    probabilities: Dict[str, float] = field(
        default_factory=lambda: {p.value: (1.0 if p == PoseName.IDLE else 0.0) for p in ALL_POSE_NAMES}
    )
    frame_bgr: Optional[np.ndarray] = None
    error: Optional[str] = None


def build_classifier(
    config: Dict[str, Any],
    *,
    on_error: Optional[Callable[[], None]] = None,
    preserved_counters: Optional[Dict[str, Any]] = None,
) -> PinchClassifier:
    cls_cfg = config.get("classifier") or {}
    mode = resolve_classifier_mode(cls_cfg)
    max_p95_ms = float(cls_cfg.get("max_onnx_p95_ms", 8.0))
    veto_min_prob = float(cls_cfg.get("veto_min_prob", 0.35))
    veto_idle_prob = float(cls_cfg.get("veto_idle_prob", 0.60))
    veto_scope = str(cls_cfg.get("veto_scope", "entry"))
    veto_entry_window_ms = float(cls_cfg.get("veto_entry_window_ms", 300.0))
    veto_confirm_frames = max(1, int(cls_cfg.get("veto_confirm_frames", 3)))
    veto_on_error = str(cls_cfg.get("veto_on_error", "pass_through"))
    heuristic = Heuristic3DPinchClassifier(cls_cfg.get("heuristic_3d"))

    if mode in ("shadow", "veto", "onnx"):
        mpath = cls_cfg.get("onnx_model_path")
        npath = cls_cfg.get("onnx_normalizer_path")
        if mpath and npath:
            mp = Path(mpath)
            np_ = Path(npath)
            if not mp.is_file():
                mp = repo_root() / mpath
            if not np_.is_file():
                np_ = repo_root() / npath
            if mp.is_file() and np_.is_file():
                try:
                    onnx_clf = ONNXPinchClassifier(
                        model_path=mp,
                        normalizer_path=np_,
                        sequence_length=int(cls_cfg.get("sequence_length", 5)),
                        feature_set=str(cls_cfg.get("feature_set", "auto")),
                    )
                    guarded = GuardedClassifier(
                        onnx_clf,
                        heuristic,
                        mode=mode,
                        veto_min_prob=veto_min_prob,
                        veto_idle_prob=veto_idle_prob,
                        veto_scope=veto_scope,
                        veto_entry_window_ms=veto_entry_window_ms,
                        veto_confirm_frames=veto_confirm_frames,
                        veto_on_error=veto_on_error,
                        max_p95_ms=max_p95_ms,
                        on_error=on_error,
                    )
                    guarded.import_counters(preserved_counters)
                    return guarded
                except Exception as exc:
                    logger.warning(f"[ML] Failed loading ONNX TCN ({exc}); falling back to heuristic_3d.")
            else:
                logger.info(
                    f"[ML] ONNX model not trained yet for mode={mode}; using heuristic_3d until model is exported."
                )
    return heuristic


class AeroHandRuntime:
    """End-to-end live camera -> 2-hand engine -> macOS Quartz/AX/Dictation runtime."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        dry_run: bool = False,
        preview: bool = False,
        camera_index: Optional[int] = None,
        collect_dir: Optional[str | Path] = None,
        passive_record_dir: Optional[str | Path] = None,
        passive_max_daily_mb: float = 256.0,
        render_hud_frames: bool = False,
        on_show_window: Optional[Callable[[], None]] = None,
        on_toggle_camera_hud: Optional[Callable[[], None]] = None,
        on_config_updated: Optional[Callable[[Dict[str, Any]], None]] = None,
        clock_ms: Optional[Callable[[], float]] = None,
        screen_geom: Optional[ScreenGeometry] = None,
        cursor_pos_fn: Optional[Callable[[], Tuple[float, float]]] = None,
    ) -> None:
        self.base_config: Dict[str, Any] = deepcopy(config)
        self.config: Dict[str, Any] = deepcopy(config)
        self._passive_record_dir: Optional[Path] = Path(passive_record_dir) if passive_record_dir else None
        # V060-06: Passive idle recording permanently forces dry_run=True (zero HID emission)
        self.dry_run = True if self._passive_record_dir is not None else bool(dry_run)
        self.preview = bool(preview or (collect_dir is not None))
        self.render_hud_frames = bool(render_hud_frames or self.preview)
        self.collect_dir = Path(collect_dir) if collect_dir else (repo_root() / "data" / "sessions")
        self._cli_collect_mode = collect_dir is not None
        self._clock_ms: Callable[[], float] = clock_ms or (lambda: time.monotonic() * 1000.0)
        self.screen_geom: ScreenGeometry = screen_geom or ScreenGeometry()
        self._cursor_pos_fn: Callable[[], Tuple[float, float]] = cursor_pos_fn or get_current_cursor_position

        self._on_show_window = on_show_window
        self._on_toggle_camera_hud = on_toggle_camera_hud
        self._on_config_updated = on_config_updated

        cam_cfg = config.get("camera") or {}
        self.camera_index = int(camera_index if camera_index is not None else cam_cfg.get("index", 0))
        self.camera_device_id = str(cam_cfg.get("device_id", ""))
        self.cam_width = int(cam_cfg.get("width", 640))
        self.cam_height = int(cam_cfg.get("height", 480))
        self.cam_aspect = str(cam_cfg.get("aspect_ratio", "4:3"))
        self.fps_cap = int(cam_cfg.get("fps_cap", 60))
        self.mirror = bool(cam_cfg.get("mirror", True))

        rt_cfg = config.get("runtime") or {}
        self.idle_throttle_enabled: bool = bool(rt_cfg.get("idle_throttle_enabled", True))
        self.idle_timeout_s: float = float(rt_cfg.get("idle_timeout_s", 3.0))
        self.idle_fps: float = float(rt_cfg.get("idle_fps", 8.0))
        self.max_drag_duration_ms: float = float(rt_cfg.get("max_drag_duration_ms", 20000.0))
        self.hand_lost_release_ms: float = float(rt_cfg.get("hand_lost_release_ms", 350.0))
        self.camera_stall_release_ms: float = float(rt_cfg.get("camera_stall_release_ms", 500.0))

        cur_cfg = config.get("cursor") or {}
        self.enable_lerp_60hz: bool = bool(cur_cfg.get("enable_lerp_60hz", True))
        self.lerp_rate_hz: float = float(cur_cfg.get("lerp_rate_hz", 60.0))
        self._lerp = CursorLerpInterpolator(
            lerp_speed=float(cur_cfg.get("lerp_speed", 55.0)),
            adaptive=bool(cur_cfg.get("adaptive_lerp", True)),
        )
        self._lerp_lock = threading.Lock()
        self._lerp_active: bool = False

        self.input_ctl = MacOSInputController(config.get("typing_guard"), dry_run=self.dry_run)
        self.classifier = build_classifier(config, on_error=self._on_classifier_error)
        self.engine = HandCursorEngine(
            config, classifier=self.classifier, on_classifier_error=self._on_classifier_error
        )
        self.audio = AudioFeedback(config.get("audio"))
        self.dictation = VoiceDictationController(
            config.get("dictation"),
            on_text_ready=self._on_dictation_text,
            on_discard=self._on_dictation_discard,
        )
        self.vitals_mon = SystemVitalsMonitor()
        col_cfg = config.get("collector") or {}
        self.collector_subject: str = str(col_cfg.get("subject", "minh")).strip() or "minh"
        self.collector_record_hand: str = str(col_cfg.get("record_hand", "primary")).lower()
        neg_raw = str(col_cfg.get("negatives_dir", "data/negatives"))
        self.negatives_dir: Path = Path(neg_raw) if Path(neg_raw).is_absolute() else (repo_root() / neg_raw)
        recall_raw = str(col_cfg.get("recall_dir", "data/recall"))
        self.recall_dir: Path = Path(recall_raw) if Path(recall_raw).is_absolute() else (repo_root() / recall_raw)
        self._cued_saved_dry_run: Optional[bool] = None
        self.negative_window_s: float = max(0.5, min(10.0, float(col_cfg.get("negative_window_s", 3.0))))
        self.frame_ring: FrameRingBuffer = FrameRingBuffer(max_frames=100)
        self._negatives_saved_count: int = 0
        self._last_camera_ts_ms: int = 0
        self.collector: PinchDataCollector = PinchDataCollector(
            trim_edge_ms=float(col_cfg.get("trim_edge_ms", 300.0)),
            countdown_s=float(col_cfg.get("countdown_s", 3.0)),
            step_duration_s=float(col_cfg.get("step_duration_s", 5.0)),
            quality_gate_mode=str(col_cfg.get("quality_gate_mode", "clean")),
            default_subject=self.collector_subject,
        )
        self.passive_recorder: Optional[PassiveIdleRecorder] = (
            PassiveIdleRecorder(
                self._passive_record_dir,
                session_meta=self.collector.session_meta,
                default_subject=self.collector_subject,
                max_daily_mb=float(passive_max_daily_mb),
            )
            if self._passive_record_dir is not None
            else None
        )
        self.calibration: PinchCalibrationWizard = PinchCalibrationWizard()
        self._cached_saved_sessions: Optional[List[str]] = None
        self._cached_corpus_counts: Dict[str, Dict[str, int]] = {}
        self._cached_condition_matrix: Optional[Dict[str, Any]] = None
        self._collector_last_save_error: Optional[List[str]] = None

        self._api_server: Optional[LocalControlAPIServer] = None
        self._stop_event = threading.Event()
        self._restart_camera_event = threading.Event()
        self._restart_tracker_event = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._processing_thread_id: Optional[int] = None
        self._config_lock = threading.Lock()
        self._classifier_swap_lock = threading.Lock()
        self._classifier_swap_gen: int = 0
        self._classifier_swap_thread: Optional[threading.Thread] = None
        self._pending_cfg: Optional[Tuple[Dict[str, Any], bool]] = None
        self._snapshot_lock = threading.Lock()
        self._snapshot = RuntimeSnapshot(dry_run=self.dry_run)

        self._paused: bool = False
        self._drag_started_ms: Optional[float] = None
        self._mouse_button_held: bool = False
        self._is_scroll_held: bool = False
        self._last_primary_ms: float = 0.0
        self._last_frame_ms: float = 0.0
        self._last_hand_ms: float = 0.0
        self._hand_present_prev: bool = False
        self._screen_ox: float = 0.0
        self._screen_oy: float = 0.0
        self._screen_w: float = 1920.0
        self._screen_h: float = 1080.0
        self._scroll_acc_x: float = 0.0
        self._scroll_acc_y: float = 0.0
        self._last_action_text: str = "Ready"
        self._active_bundle_id: Optional[str] = None
        self._click_seq: int = 0
        self._last_click_event: Optional[Dict[str, Any]] = None
        self._practice: Dict[str, bool] = {
            "move": False,
            "click": False,
            "right_click": False,
            "drag": False,
            "scroll": False,
            "window_tile": False,
            "spaces_nav": False,
            "palm_wheel": False,
            "dictation": False,
            "zoom": False,
        }
        self._move_accum_px: float = 0.0

    # ------------------------------------------------------------------
    # Live Controls (Thread-safe for WKWebView UI, Menu Bar & HTTP API)
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
            self._mouse_button_held = False
            self._is_scroll_held = False
            self.dictation.stop_and_transcribe()
            self.input_ctl.release_all()
            self._last_action_text = "Paused"
        else:
            self._last_action_text = "Tracking resumed"

    def set_dry_run(self, dry_run: bool) -> bool:
        target = bool(dry_run)
        if not target and (self.collector.is_cued_active or self.passive_recorder is not None):
            logger.warning("[COLLECT] Refusing live HID output while cued or passive recording is active.")
            return False
        if target:
            with self._lerp_lock:
                self._lerp_active = False
            self._drag_started_ms = None
            self._mouse_button_held = False
            self._is_scroll_held = False
            # Release any held macOS HID button/modifiers BEFORE silencing input_ctl with dry_run=True
            self.input_ctl.release_all()
        self.dry_run = target
        self.input_ctl.dry_run = self.dry_run
        self._last_action_text = "Mode: Dry-Run" if self.dry_run else "Mode: Live Control"
        return True

    def trigger_recenter(self) -> None:
        self._refresh_display_bounds()
        cx = self._screen_ox + self._screen_w * 0.5
        cy = self._screen_oy + self._screen_h * 0.5
        self.engine.recenter(self._screen_w, self._screen_h)
        with self._lerp_lock:
            self._lerp.reset((cx, cy))
        if not self.dry_run:
            self.input_ctl.mouse_move(cx, cy)
        self._last_action_text = "Cursor recentered"

    def toggle_dictation(self) -> bool:
        active = self.dictation.toggle()
        self._last_action_text = "Dictation ON" if active else "Dictation transcribing..."
        return active

    def _on_dictation_text(self, text: str) -> None:
        if text:
            self._last_action_text = f"Dictated: {text[:36]}"
            self.input_ctl.paste_text_utf8(text)

    def _on_dictation_discard(self, reason: str, _ev: Dict[str, Any]) -> None:
        self._last_action_text = f"Dictation discarded ({reason})"

    def _on_classifier_error(self) -> None:
        self._force_release_held_inputs("Classifier inference error")

    def rearm_classifier(self) -> bool:
        """Re-arm ONNX after a latency/error auto-fallback (V051-03)."""
        if isinstance(self.classifier, GuardedClassifier):
            self.classifier.rearm()
            self._last_action_text = "Re-armed ONNX classifier"
            return True
        return False

    def _export_classifier_counters(self) -> Dict[str, int]:
        if isinstance(self.classifier, GuardedClassifier):
            return self.classifier.export_counters()
        return dict(getattr(self, "_cumulative_classifier_counters", {}))

    def _swap_classifier_async(self, cfg_snapshot: Dict[str, Any]) -> threading.Thread:
        """
        Load ONNX classifier on a background thread and swap atomically under locks (V051-06 / V052-05c).
        Exports live counters from the old classifier RIGHT BEFORE swapping so events accumulated
        during background model load & warm-up are never lost, and discards stale generations.
        """
        with self._classifier_swap_lock:
            self._classifier_swap_gen += 1
            gen = self._classifier_swap_gen

            def _worker() -> None:
                new_clf = build_classifier(
                    cfg_snapshot,
                    on_error=self._on_classifier_error,
                )
                with self._classifier_swap_lock, self.engine._classifier_lock:
                    if self._classifier_swap_gen != gen:
                        return
                    latest = self._export_classifier_counters()
                    if latest:
                        self._cumulative_classifier_counters = dict(latest)
                    if isinstance(new_clf, GuardedClassifier):
                        new_clf.import_counters(latest or getattr(self, "_cumulative_classifier_counters", None))
                    self.classifier = new_clf
                    self.engine.set_classifier(new_clf)

            t = threading.Thread(target=_worker, name="aerohand-clf-reload", daemon=True)
            self._classifier_swap_thread = t
        t.start()
        return t

    def classifier_telemetry(self) -> Dict[str, Any]:
        with self._classifier_swap_lock:
            clf = self.classifier
            if isinstance(clf, GuardedClassifier):
                tel = clf.telemetry()
                self._cumulative_classifier_counters = clf.export_counters()
                return tel
            cls_cfg = self.config.get("classifier") or {}
            cum = dict(getattr(self, "_cumulative_classifier_counters", {}))
            backend_name = type(clf).__name__
        s_frames = int(cum.get("shadow_frames", 0))
        s_dis = int(cum.get("shadow_disagreements", 0))
        s_ev = int(cum.get("shadow_events", 0))
        s_wv_ev = int(cum.get("shadow_would_veto_events", 0))
        s_sw_ev = int(cum.get("shadow_would_swallow_events", 0))
        return {
            "mode": "heuristic",
            "active_backend": backend_name,
            "shadow_mode": False,
            "shadow_frames": s_frames,
            "shadow_disagreements": s_dis,
            "shadow_disagreement_rate": round(s_dis / float(s_frames), 4) if s_frames > 0 else 0.0,
            "shadow_events": s_ev,
            "shadow_would_veto_events": s_wv_ev,
            "shadow_would_veto_frames": int(cum.get("shadow_would_veto_frames", 0)),
            "shadow_would_swallow_events": s_sw_ev,
            "shadow_would_pending_frames": int(cum.get("shadow_would_pending_frames", 0)),
            "shadow_would_veto_rate": round(s_wv_ev / float(s_ev), 4) if s_ev > 0 else 0.0,
            "shadow_would_block_rate": round((s_wv_ev + s_sw_ev) / float(s_ev), 4) if s_ev > 0 else 0.0,
            "veto_count": int(cum.get("veto_count", 0)),
            "veto_frames": int(cum.get("veto_frames", 0)),
            "veto_swallowed_events": int(cum.get("veto_swallowed_events", 0)),
            "veto_pending_frames": int(cum.get("veto_pending_frames", 0)),
            "veto_retried_within_1s": int(cum.get("veto_retried_within_1s", 0)),
            "veto_abstain_not_ready": int(cum.get("veto_abstain_not_ready", 0)),
            "veto_scope": str(cls_cfg.get("veto_scope", "entry")),
            "veto_entry_window_ms": float(cls_cfg.get("veto_entry_window_ms", 300.0)),
            "veto_confirm_frames": max(1, int(cls_cfg.get("veto_confirm_frames", 3))),
            "veto_on_error": str(cls_cfg.get("veto_on_error", "pass_through")),
            "veto_min_prob": float(cls_cfg.get("veto_min_prob", 0.35)),
            "veto_idle_prob": float(cls_cfg.get("veto_idle_prob", 0.60)),
            "onnx_p95_ms": 0.0,
            "feature_p95_ms": 0.0,
            "onnx_latency_samples": 0,
            "onnx_errors": int(cum.get("onnx_errors", 0)),
            "fell_back": False,
            "fallback_reason": None,
            "guard_active": True,
            "max_onnx_p95_ms": float(cls_cfg.get("max_onnx_p95_ms", 8.0)),
            "model_hash": None,
            "model_sha256": None,
            "heuristic_source_sha256": compute_heuristic_source_sha256(),
        }

    def _emit_click_pulse(self, kind: str, x: float, y: float) -> None:
        self._click_seq += 1
        self._last_click_event = {"seq": self._click_seq, "kind": kind, "x": round(x, 1), "y": round(y, 1)}

    def apply_live_settings(self, config: Dict[str, Any], *, from_profile: bool = False) -> None:
        """
        Queue config updates from UI/HTTP threads to apply cleanly on the processing thread
        at the start of the next frame (prevents cross-thread filter/engine races).
        """
        cfg_copy = deepcopy(config)
        with self._config_lock:
            if not from_profile:
                self.base_config = deepcopy(cfg_copy)
            proc_tid = self._processing_thread_id
            if proc_tid is not None and threading.get_ident() != proc_tid:
                self._pending_cfg = (cfg_copy, bool(from_profile))
                return
        self._apply_live_settings_now(cfg_copy, from_profile=from_profile)

    def _flush_pending_config(self) -> bool:
        """Swap and apply any queued configuration update on the processing thread."""
        with self._config_lock:
            pending = self._pending_cfg
            self._pending_cfg = None
        if pending is None:
            return False
        cfg, from_profile = pending
        self._apply_live_settings_now(cfg, from_profile=from_profile)
        return True

    def _apply_live_settings_now(self, config: Dict[str, Any], *, from_profile: bool = False) -> None:
        """Apply any config update in realtime; restarts camera/tracker only when hardware params change."""
        if not from_profile:
            with self._config_lock:
                self.base_config = deepcopy(config)
        effective = (
            config
            if from_profile
            else resolve_profile_config(self.base_config, self._active_bundle_id)
        )
        old_cam = self.config.get("camera") or {}
        old_trk = self.config.get("tracker") or {}
        old_cls = self.config.get("classifier") or {}
        self.config = deepcopy(effective)

        cur = effective.get("cursor") or {}
        self.enable_lerp_60hz = bool(cur.get("enable_lerp_60hz", True))
        self._lerp.lerp_speed = float(cur.get("lerp_speed", 55.0))
        self._lerp.adaptive = bool(cur.get("adaptive_lerp", True))

        rt_cfg = effective.get("runtime") or {}
        self.idle_throttle_enabled = bool(rt_cfg.get("idle_throttle_enabled", True))
        self.idle_timeout_s = float(rt_cfg.get("idle_timeout_s", 3.0))
        self.idle_fps = float(rt_cfg.get("idle_fps", 8.0))
        self.max_drag_duration_ms = float(rt_cfg.get("max_drag_duration_ms", 20000.0))
        self.hand_lost_release_ms = float(rt_cfg.get("hand_lost_release_ms", 350.0))
        self.camera_stall_release_ms = float(rt_cfg.get("camera_stall_release_ms", 500.0))

        self.engine.update_config(effective)

        tg = effective.get("typing_guard") or {}
        self.input_ctl.typing_guard_enabled = bool(tg.get("enabled", True))
        self.input_ctl.keyboard_guard_s = float(tg.get("keyboard_ms", 1500)) / 1000.0
        self.input_ctl.mouse_guard_s = float(tg.get("mouse_ms", 1000)) / 1000.0

        col_cfg = effective.get("collector") or {}
        new_subj = str(col_cfg.get("subject", self.collector_subject)).strip() or "minh"
        if new_subj != self.collector_subject:
            self.collector_subject = new_subj
            self.collector.set_session_meta({"subject": new_subj})
        self.collector_record_hand = str(col_cfg.get("record_hand", "primary")).lower()

        self.audio.update_config(effective.get("audio"))
        self.dictation.update_config(effective.get("dictation"))

        new_cam = effective.get("camera") or {}
        cam_changed = (
            str(new_cam.get("device_id", "")) != str(old_cam.get("device_id", ""))
            or int(new_cam.get("index", 0)) != int(old_cam.get("index", 0))
            or int(new_cam.get("width", 640)) != int(old_cam.get("width", 640))
            or int(new_cam.get("height", 480)) != int(old_cam.get("height", 480))
            or str(new_cam.get("aspect_ratio", "4:3")) != str(old_cam.get("aspect_ratio", "4:3"))
            or bool(new_cam.get("mirror", True)) != bool(old_cam.get("mirror", True))
        )
        if cam_changed:
            self.camera_device_id = str(new_cam.get("device_id", ""))
            self.camera_index = int(new_cam.get("index", 0))
            self.cam_width = int(new_cam.get("width", 640))
            self.cam_height = int(new_cam.get("height", 480))
            self.cam_aspect = str(new_cam.get("aspect_ratio", "4:3"))
            self.mirror = bool(new_cam.get("mirror", True))
            self._restart_camera_event.set()

        new_trk = effective.get("tracker") or {}
        if (
            int(new_trk.get("num_hands", 2)) != int(old_trk.get("num_hands", 2))
            or str(new_trk.get("delegate", "CPU")) != str(old_trk.get("delegate", "CPU"))
        ):
            self._restart_tracker_event.set()

        new_cls = effective.get("classifier") or {}
        new_mode = resolve_classifier_mode(new_cls)
        old_mode = resolve_classifier_mode(old_cls)
        model_paths_changed = (
            str(new_cls.get("onnx_model_path", "")) != str(old_cls.get("onnx_model_path", ""))
            or str(new_cls.get("onnx_normalizer_path", "")) != str(old_cls.get("onnx_normalizer_path", ""))
            or int(new_cls.get("sequence_length", 5)) != int(old_cls.get("sequence_length", 5))
            or str(new_cls.get("feature_set", "invariant348")) != str(old_cls.get("feature_set", "invariant348"))
        )
        heuristic_changed = (new_cls.get("heuristic_3d") or {}) != (old_cls.get("heuristic_3d") or {})
        guard_params_changed = (
            float(new_cls.get("veto_min_prob", 0.35)) != float(old_cls.get("veto_min_prob", 0.35))
            or float(new_cls.get("veto_idle_prob", 0.60)) != float(old_cls.get("veto_idle_prob", 0.60))
            or str(new_cls.get("veto_scope", "entry")) != str(old_cls.get("veto_scope", "entry"))
            or float(new_cls.get("veto_entry_window_ms", 300.0)) != float(old_cls.get("veto_entry_window_ms", 300.0))
            or int(new_cls.get("veto_confirm_frames", 3)) != int(old_cls.get("veto_confirm_frames", 3))
            or str(new_cls.get("veto_on_error", "pass_through")) != str(old_cls.get("veto_on_error", "pass_through"))
            or float(new_cls.get("max_onnx_p95_ms", 8.0)) != float(old_cls.get("max_onnx_p95_ms", 8.0))
        )

        cls_changed = model_paths_changed or (new_mode != old_mode) or heuristic_changed or guard_params_changed
        with self._classifier_swap_lock:
            swap_in_flight = self._classifier_swap_thread is not None and self._classifier_swap_thread.is_alive()

        # V051-06 / V053-04 (B4): Update heuristic/veto parameters in-place without wiping counters or
        # reloading ONNX — EXCEPT when an async swap is currently in flight and any classifier setting
        # changed, in which case we route through rebuild so `_classifier_swap_gen` increments and the
        # stale background thread cannot overwrite newer user settings.
        if (not (swap_in_flight and cls_changed)) and (not model_paths_changed) and (new_mode == old_mode):
            if heuristic_changed:
                if isinstance(self.classifier, GuardedClassifier):
                    self.classifier.update_heuristic_config(new_cls.get("heuristic_3d"))
                elif isinstance(self.classifier, Heuristic3DPinchClassifier):
                    self.classifier.update_config(new_cls.get("heuristic_3d"))
            if guard_params_changed and isinstance(self.classifier, GuardedClassifier):
                self.classifier.update_veto_config(
                    veto_min_prob=float(new_cls.get("veto_min_prob", 0.35)),
                    veto_idle_prob=float(new_cls.get("veto_idle_prob", 0.60)),
                    veto_scope=str(new_cls.get("veto_scope", "entry")),
                    veto_entry_window_ms=float(new_cls.get("veto_entry_window_ms", 300.0)),
                    veto_confirm_frames=int(new_cls.get("veto_confirm_frames", 3)),
                    veto_on_error=str(new_cls.get("veto_on_error", "pass_through")),
                    max_p95_ms=float(new_cls.get("max_onnx_p95_ms", 8.0)),
                )
        elif (
            (not (swap_in_flight and cls_changed))
            and (not model_paths_changed)
            and isinstance(self.classifier, GuardedClassifier)
        ):
            # Mode toggle on an already-loaded GuardedClassifier: switch in-place (clears buffers, keeps counters!)
            self.classifier.set_mode(new_mode)
            if heuristic_changed:
                self.classifier.update_heuristic_config(new_cls.get("heuristic_3d"))
            if guard_params_changed:
                self.classifier.update_veto_config(
                    veto_min_prob=float(new_cls.get("veto_min_prob", 0.35)),
                    veto_idle_prob=float(new_cls.get("veto_idle_prob", 0.60)),
                    veto_scope=str(new_cls.get("veto_scope", "entry")),
                    veto_entry_window_ms=float(new_cls.get("veto_entry_window_ms", 300.0)),
                    veto_confirm_frames=int(new_cls.get("veto_confirm_frames", 3)),
                    veto_on_error=str(new_cls.get("veto_on_error", "pass_through")),
                    max_p95_ms=float(new_cls.get("max_onnx_p95_ms", 8.0)),
                )
        else:
            # Need to load/rebuild ONNX model (preserve cumulative counters; load off-thread if loop active)
            if self._processing_thread_id is not None and new_mode in ("shadow", "veto", "onnx"):
                self._swap_classifier_async(deepcopy(effective))
            else:
                with self._classifier_swap_lock, self.engine._classifier_lock:
                    self._classifier_swap_gen += 1
                    preserved = self._export_classifier_counters()
                    if preserved:
                        self._cumulative_classifier_counters = dict(preserved)
                    self.classifier = build_classifier(
                        effective,
                        on_error=self._on_classifier_error,
                        preserved_counters=preserved or getattr(self, "_cumulative_classifier_counters", None),
                    )
                    self.engine.set_classifier(self.classifier)

    # ------------------------------------------------------------------
    # Data Collector Studio Controls (callable from WKWebView UI)
    # ------------------------------------------------------------------

    def _refresh_saved_sessions_cache(self) -> List[str]:
        import json as _json

        sessions = sorted(self.collect_dir.glob("*.json")) if self.collect_dir.is_dir() else []
        self._cached_saved_sessions = [fp.name for fp in sessions[-10:]]
        agg: Dict[str, Dict[str, int]] = {p.value: {"total": 0, "Right": 0, "Left": 0} for p in ALL_POSE_NAMES}
        for fp in sessions:
            try:
                data = _json.loads(fp.read_text(encoding="utf-8"))
                if isinstance(data, dict) and str(data.get("recording_kind") or "").strip().lower() == "cued_events":
                    continue
                cbh = data.get("counts_by_hand")
                if isinstance(cbh, dict):
                    for pname, hand_map in cbh.items():
                        if pname in agg and isinstance(hand_map, dict):
                            r = int(hand_map.get("Right", 0))
                            l_cnt = int(hand_map.get("Left", 0))
                            agg[pname]["Right"] += r
                            agg[pname]["Left"] += l_cnt
                            agg[pname]["total"] += r + l_cnt
                else:
                    cnts = data.get("counts") or {}
                    for pname, n in cnts.items():
                        if pname in agg:
                            agg[pname]["total"] += int(n)
            except Exception:
                continue
        self._cached_corpus_counts = agg
        self._cached_condition_matrix = compute_corpus_condition_matrix(
            self.collect_dir,
            negatives_dir=self.negatives_dir,
        )
        return self._cached_saved_sessions

    def collector_toggle_recording(self) -> Dict[str, Any]:
        active = self.collector.toggle_recording()
        if not active:
            self.collector.autosave_draft(self.collect_dir)
        self._refresh_saved_sessions_cache()
        return self.collector_status()

    def collector_set_recording(self, active: bool) -> Dict[str, Any]:
        was_rec = self.collector.is_recording
        self.collector.set_recording(active)
        if was_rec and not active:
            self.collector.autosave_draft(self.collect_dir)
        return self.collector_status()

    def collector_set_pose(self, pose_name: str) -> Dict[str, Any]:
        try:
            self.collector.set_pose(PoseName(pose_name))
        except ValueError:
            pass
        return self.collector_status()

    def collector_set_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        self.collector.set_session_meta(meta)
        return self.collector_status()

    def collector_undo(self, seconds: float = 5.0) -> Dict[str, Any]:
        removed = self.collector.undo_last(seconds=seconds)
        self._last_action_text = f"Undid {removed} samples"
        return self.collector_status()

    def collector_start_guided(self) -> Dict[str, Any]:
        now_ms = float(self._last_camera_ts_ms) if self._last_camera_ts_ms > 0 else None
        self.collector.start_guided(now_ms=now_ms)
        return self.collector_status()

    def collector_stop_guided(self) -> Dict[str, Any]:
        self.collector.stop_guided()
        self.collector.autosave_draft(self.collect_dir)
        return self.collector_status()

    def _restore_cued_dry_run(self) -> None:
        """Restore the pre-cued `dry_run` flag safely after stop, completion, error, or shutdown."""
        if self.collector.is_cued_active:
            self.collector.stop_cued_session()
        if self._cued_saved_dry_run is not None:
            prev = self._cued_saved_dry_run
            self._cued_saved_dry_run = None
            self.set_dry_run(prev)

    def collector_start_cued(
        self,
        *,
        cues_per_trigger: int = 10,
        hand: Optional[str] = None,
        window_ms: int = 3500,
        gap_ms: int = 1800,
        seed: Optional[int] = None,
        triggers: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Start a cued-event recall session (forces `dry_run=True` after releasing held inputs)."""
        if self._cued_saved_dry_run is None:
            self._cued_saved_dry_run = bool(self.dry_run)
        if self.dictation.is_recording:
            self.dictation.cancel(reason="cued_recording")
        # Release any held HID state BEFORE forcing dry_run=True
        self.set_dry_run(True)
        eff_hand = hand or str((self.config.get("hand") or {}).get("cursor_hand", "Right"))
        self.collector.start_cued_session(
            cues_per_trigger=cues_per_trigger,
            hand=eff_hand,
            window_ms=window_ms,
            gap_ms=gap_ms,
            seed=seed,
            triggers=triggers,
        )
        self._last_action_text = "Cued recall recording started (Dry-Run)"
        return self.collector_status()

    def collector_stop_cued(self) -> Dict[str, Any]:
        self.collector.stop_cued_session()
        try:
            self.collector.autosave_cued_draft(self.recall_dir)
        except Exception:
            pass
        self._restore_cued_dry_run()
        self._last_action_text = "Cued recall recording stopped"
        return self.collector_status()

    def collector_save_cued(self) -> Dict[str, Any]:
        if self.collector.is_cued_active:
            self.collector.stop_cued_session()
        self._restore_cued_dry_run()
        out = self.collector.save_cued(self.recall_dir)
        self._last_action_text = f"Saved cued session: {out.name}"
        st = self.collector_status()
        st["saved"] = True
        st["saved_path"] = str(out)
        return st

    def collector_save(self, *, allow_low_quality: bool = False) -> Dict[str, Any]:
        try:
            out = self.collector.save(self.collect_dir, allow_low_quality=allow_low_quality)
        except QualityGateError as exc:
            self._collector_last_save_error = list(exc.warnings)
            self._last_action_text = f"Save blocked ({', '.join(exc.warnings)})"
            st = self.collector_status()
            st["saved"] = False
            st["blocked_warnings"] = list(exc.warnings)
            return st
        self._collector_last_save_error = None
        self._last_action_text = f"Saved dataset: {out.name}"
        self._refresh_saved_sessions_cache()
        st = self.collector_status()
        st["saved"] = True
        st["saved_path"] = str(out)
        return st

    def collector_clear(self) -> Dict[str, Any]:
        self._collector_last_save_error = None
        self.collector.clear()
        self._refresh_saved_sessions_cache()
        return self.collector_status()

    def mark_deliberate_gesture(self, timestamp_ms: Optional[int] = None, *, note: str = "") -> Dict[str, Any]:
        """Mark a deliberate gesture during `--passive-record` (+/-1.5s excluded from idle exposure; NEVER for FT)."""
        if self.passive_recorder is None:
            return {"ok": False, "error": "Passive recording is not active."}
        ts = int(timestamp_ms) if timestamp_ms is not None else int(self._last_camera_ts_ms)
        res = self.passive_recorder.mark_deliberate_gesture(ts, note=note)
        self._last_action_text = f"Marked deliberate gesture (#{res['marks_count']})"
        logger.info(
            f"[PASSIVE] Marked deliberate gesture #{res['marks_count']} at t={ts}ms (+/-1.5s excluded)."
        )
        return res

    def mark_last_trigger_not_intended(self, out_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        """Save +/- 1.5s of landmark ring buffer around the last false trigger as an `idle` hard-negative."""
        if self.dictation.is_recording:
            self.dictation.cancel(reason="not_intended")
        target_dir = Path(out_dir) if out_dir is not None else self.negatives_dir
        saved = self.frame_ring.save_negative_snapshot(
            target_dir,
            window_s=self.negative_window_s,
            session_meta=self.collector.session_meta,
        )
        trig = self.frame_ring.last_trigger or {}
        if saved is None:
            self._last_action_text = "No recent hand frames for hard-negative"
            return {"ok": False, "path": None, "negatives_saved": self._negatives_saved_count}
        self._negatives_saved_count += 1
        self._last_action_text = f"Saved hard-negative: {saved.name}"
        logger.info(f"[COLLECT] Saved false-trigger hard-negative -> {saved}")
        return {
            "ok": True,
            "path": str(saved),
            "trigger_type": trig.get("trigger_type", "manual"),
            "negatives_saved": self._negatives_saved_count,
        }

    def collector_status(self) -> Dict[str, Any]:
        from aerohand.ml.collector import COLLECTOR_CLASS_TARGETS

        counts = self.collector.counts()
        by_hand = self.collector.counts_by_hand()
        if self._cached_saved_sessions is None:
            self._refresh_saved_sessions_cache()
        combined: Dict[str, Dict[str, int]] = {}
        for p in ALL_POSE_NAMES:
            pk = p.value
            base = self._cached_corpus_counts.get(pk, {"total": 0, "Right": 0, "Left": 0})
            cur_r = by_hand.get(pk, {}).get("Right", 0)
            cur_l = by_hand.get(pk, {}).get("Left", 0)
            combined[pk] = {
                "total": base["total"] + counts.get(pk, 0),
                "Right": base["Right"] + cur_r,
                "Left": base["Left"] + cur_l,
                "target": COLLECTOR_CLASS_TARGETS[pk]["total"],
            }
        _, clean_preview = clean_recording_samples(self.collector._samples)
        return {
            "recording": self.collector.is_recording,
            "current_pose": self.collector.current_pose.value,
            "counts": counts,
            "counts_by_hand": by_hand,
            "corpus_progress": combined,
            "total": sum(counts.values()),
            "session_meta": self.collector.session_meta,
            "guided": self.collector.guided_status(
                now_ms=float(self._last_camera_ts_ms) if self._last_camera_ts_ms > 0 else None
            ),
            "cued": self.collector.cued_status(
                now_ms=float(self._last_camera_ts_ms) if self._last_camera_ts_ms > 0 else None
            ),
            "recall_dir": str(self.recall_dir),
            "quality": self.collector.evaluate_quality(),
            "quality_gate_mode": self.collector.quality_gate_mode,
            "cleaned_preview": clean_preview,
            "draft_path": str(self.collector._last_draft_path) if self.collector._last_draft_path else None,
            "record_hand": self.collector_record_hand,
            "sessions_dir": str(self.collect_dir),
            "negatives_dir": str(self.negatives_dir),
            "negatives_saved": self._negatives_saved_count,
            "last_trigger": self.frame_ring.last_trigger,
            "last_save_blocked_warnings": list(self._collector_last_save_error or []),
            "condition_matrix": self._cached_condition_matrix or {},
            "saved_sessions": list(self._cached_saved_sessions or []),
        }

    def reset_practice(self) -> Dict[str, bool]:
        for k in self._practice:
            self._practice[k] = False
        self._move_accum_px = 0.0
        return dict(self._practice)

    def calibration_start(self) -> Dict[str, Any]:
        self.calibration.start()
        return self.calibration.status()

    def calibration_cancel(self) -> Dict[str, Any]:
        self.calibration.cancel()
        return self.calibration.status()

    def calibration_apply(self) -> Optional[Dict[str, Any]]:
        sugg = self.calibration.suggestion
        if not sugg:
            return None
        new_cfg = deepcopy(self.base_config)
        h3d = new_cfg.setdefault("classifier", {}).setdefault("heuristic_3d", {})
        h3d["pinch_on_dist"] = float(sugg["pinch_on_dist"])
        if "pinch_off_dist" in sugg:
            h3d["pinch_off_dist"] = float(sugg["pinch_off_dist"])
        h3d["pinky_base_on_dist"] = float(sugg["pinky_base_on_dist"])
        h3d["dual_pinch_on_dist"] = float(sugg["dual_pinch_on_dist"])
        th_map = new_cfg.setdefault("pose_state_machine", {}).setdefault("thresholds", {})
        for pose_k, val in (sugg.get("enter_thresholds") or {}).items():
            th_map.setdefault(pose_k, {})["enter_threshold"] = float(val)
        self.apply_live_settings(new_cfg)
        self.calibration.cancel()
        return self.base_config

    def get_snapshot(self) -> RuntimeSnapshot:
        with self._snapshot_lock:
            return self._snapshot

    def list_available_cameras(self) -> List[Dict[str, Any]]:
        return [d.to_dict() for d in list_cameras()]

    def start_background(self) -> None:
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stop_event.clear()
        self.preview = False
        self.render_hud_frames = True
        self._start_api_if_enabled()
        self._bg_thread = threading.Thread(target=self._run_loop, name="AeroHandRuntime", daemon=True)
        self._bg_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if sum(self.collector.counts().values()) > 0 or self.collector._segment_buf:
            try:
                self.collector.autosave_draft(self.collect_dir)
            except Exception:
                pass
        if self.collector._cued_samples or self.collector._cued_events:
            try:
                self.collector.autosave_cued_draft(self.recall_dir)
            except Exception:
                pass
        self._restore_cued_dry_run()
        if self.passive_recorder is not None:
            try:
                summary = self.passive_recorder.close()
                logger.info(
                    f"[PASSIVE] Closed recorder: chunks={summary['chunks_written']} "
                    f"idle_h={summary['hand_present_idle_hours']:.4f}h marks={summary['deliberate_marks_count']}"
                )
            except Exception:
                pass
        if self._api_server is not None:
            self._api_server.stop()
            self._api_server = None
        self.dictation.close()
        if self._bg_thread is not None and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=2.5)
        self._bg_thread = None

    def _start_api_if_enabled(self) -> None:
        api_cfg = self.config.get("api") or {}
        if not api_cfg.get("enabled", True):
            return
        if self._api_server is not None:
            return

        def _get_status() -> Dict[str, Any]:
            s = self.get_snapshot()
            return {
                "ok": True,
                "version": __version__,
                "running": s.running,
                "tracking": not s.paused,
                "paused": s.paused,
                "dry_run": s.dry_run,
                "fps": round(s.fps, 1),
                "delegate": s.delegate,
                "active_app": s.active_app,
                "hand": s.hand,
                "pose": s.active_pose or s.pose,
                "language": (self.base_config.get("ui") or {}).get("language", "vi"),
                "dictation_active": bool(s.dictation.get("recording", False)),
            }

        def _get_vitals() -> Dict[str, Any]:
            v = self.vitals_mon.read(force=True).to_dict()
            v["timing"] = self.get_snapshot().timing
            v["classifier"] = self.classifier_telemetry()
            v["dictation"] = self.dictation.status_dict()
            v["engine"] = self.engine.engine_telemetry()
            return v

        def _set_tracking(enabled: bool) -> bool:
            self.set_paused(not enabled)
            return not self._paused

        def _apply_from_api(new_cfg: Dict[str, Any]) -> None:
            self.apply_live_settings(new_cfg)
            if self._on_config_updated is not None:
                self._on_config_updated(self.base_config)

        self._api_server = LocalControlAPIServer(
            host=str(api_cfg.get("host", "127.0.0.1")),
            port=int(api_cfg.get("port", 17420)),
            get_status=_get_status,
            get_vitals=_get_vitals,
            get_config=lambda: deepcopy(self.base_config),
            apply_config=_apply_from_api,
            set_tracking_enabled=_set_tracking,
            trigger_recenter=self.trigger_recenter,
            set_dry_run=self.set_dry_run,
            show_window=self._on_show_window,
            mark_deliberate_gesture=self.mark_deliberate_gesture,
        )
        self._api_server.start()

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

    def _record_collector_frame(
        self,
        hands: List[Any],
        primary: Optional[HandOutput],
        timestamp_ms: int,
    ) -> None:
        """Record landmarks for cued sessions (all hands + hand-absent) or normal collector (`primary`/`both`)."""
        if self.collector.is_cued_active:
            prev_issued = len(self.collector._cued_events)
            self.collector.tick_cued(now_ms=float(timestamp_ms))
            if len(self.collector._cued_events) > prev_issued:
                try:
                    self.audio.play_wheel_tick()
                except Exception:
                    pass
            if self.collector.is_cued_active:
                self.collector.add_cued_frame(hands, timestamp_ms)
            else:
                try:
                    self.collector.autosave_cued_draft(self.recall_dir)
                except Exception:
                    pass
                self._restore_cued_dry_run()
                self._last_action_text = "Cued recall session complete — ready to save"
            return

        self.collector.tick_guided(now_ms=float(timestamp_ms))
        if not self.collector.is_recording or not hands:
            return
        if self.collector_record_hand == "both":
            for h_inp in hands:
                self.collector.add_frame(
                    h_inp.landmarks,
                    h_inp.world_landmarks,
                    h_inp.handedness,
                    h_inp.confidence,
                    timestamp_ms,
                    hand_id=h_inp.handedness,
                )
            return

        target_hand: Optional[str] = primary.hand if primary is not None else None
        chosen = None
        if target_hand is not None:
            for h_inp in hands:
                if h_inp.handedness == target_hand:
                    chosen = h_inp
                    break
        if chosen is None:
            pref = str((self.config.get("hand") or {}).get("cursor_hand", "Right"))
            for h_inp in hands:
                if pref == "Both" or h_inp.handedness == pref:
                    chosen = h_inp
                    break
        if chosen is not None:
            self.collector.add_frame(
                chosen.landmarks,
                chosen.world_landmarks,
                chosen.handedness,
                chosen.confidence,
                timestamp_ms,
                hand_id=chosen.handedness,
            )

    def _select_click_hand(
        self, outputs: List[HandOutput], primary: Optional[HandOutput]
    ) -> Optional[HandOutput]:
        """Resolve which hand drives click/right-click/drag actions (supports split cursor/click hands)."""
        hand_cfg = self.config.get("hand") or {}
        click_pref = str(hand_cfg.get("click_hand", "Right"))
        if click_pref.lower() == "same":
            return primary
        if click_pref in ("Left", "Right"):
            for o in outputs:
                if o.hand == click_pref:
                    return o
            return None
        return primary

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
        self._practice["palm_wheel"] = True

        if action in ("toggle_pause", "tracking_toggle"):
            self.set_paused(not self._paused)
        elif action == "stop_tracking":
            self.set_paused(True)
        elif action == "recenter":
            self.trigger_recenter()
        elif action == "show_main_window":
            if self._on_show_window is not None:
                self._on_show_window()
        elif action == "toggle_camera_hud":
            if self._on_toggle_camera_hud is not None:
                self._on_toggle_camera_hud()
        elif action == "mission_control":
            self.input_ctl.toggle_mission_control()
        elif action == "spotlight":
            self.input_ctl.open_spotlight()
        elif action in ("dictation_toggle", "dictation"):
            self.toggle_dictation()
        elif action == "siri_listen":
            self.input_ctl.siri_listen()
        elif action == "media_play_pause":
            self.input_ctl.media_play_pause()
        elif action == "media_next":
            self.input_ctl.media_next()
        elif action == "media_prev":
            self.input_ctl.media_prev()
        elif action == "volume_up":
            self.input_ctl.volume_up()
        elif action == "volume_down":
            self.input_ctl.volume_down()
        elif action == "volume_mute":
            self.input_ctl.volume_mute()
        elif action == "window_hide":
            self.input_ctl.window_hide()
        elif action == "window_fullscreen":
            self.input_ctl.window_fullscreen()
        elif action == "window_minimize":
            self.input_ctl.window_minimize()
        elif action == "window_close":
            self.input_ctl.window_close()
        elif action == "edit_copy":
            self.input_ctl.edit_copy()
        elif action == "edit_paste":
            self.input_ctl.edit_paste()
        elif action == "edit_cut":
            self.input_ctl.edit_cut()
        elif action == "edit_undo":
            self.input_ctl.edit_undo()
        elif action == "edit_redo":
            self.input_ctl.edit_redo()
        elif action == "keyboard_emoji":
            self.input_ctl.keyboard_emoji()
        elif action.startswith("open_app:"):
            self.input_ctl.open_app(action.split(":", 1)[1].strip())

    def _now_ms(self) -> float:
        return float(self._clock_ms())

    def _has_held_input(self) -> bool:
        pressed_btns = getattr(self.input_ctl, "_pressed_mouse_buttons", None)
        pressed_keys = getattr(self.input_ctl, "_pressed_keys", None)
        return (
            self._drag_started_ms is not None
            or self._mouse_button_held
            or self._is_scroll_held
            or bool(pressed_btns)
            or bool(pressed_keys)
        )

    def _force_release_held_inputs(self, reason: str) -> None:
        logger.warning(f"[SAFETY] {reason}; releasing all held inputs.")
        self._drag_started_ms = None
        self._mouse_button_held = False
        if self._is_scroll_held:
            self._is_scroll_held = False
            self._scroll_acc_x = 0.0
            self._scroll_acc_y = 0.0
            self.input_ctl.end_scroll_phase()
        self.input_ctl.release_all()
        for hs in self.engine._hands.values():
            hs.is_grabbing = False

    def _enforce_input_safety(self, now_ms: Optional[float] = None) -> bool:
        """
        Level-triggered watchdog guaranteeing zero stuck mouse buttons or scroll phases
        across primary-hand loss, secondary-hand-only views, max drag timeout, or camera stalls.
        """
        if not self._has_held_input():
            return False
        now = self._now_ms() if now_ms is None else float(now_ms)

        if self._drag_started_ms is not None and (now - self._drag_started_ms) > self.max_drag_duration_ms:
            self._force_release_held_inputs("Max drag duration exceeded")
            return True

        if (
            (self._drag_started_ms is not None or self._mouse_button_held or self._is_scroll_held)
            and self._last_primary_ms > 0.0
            and (now - self._last_primary_ms) >= self.hand_lost_release_ms
        ):
            self._force_release_held_inputs(
                f"Primary hand/pose absent for {now - self._last_primary_ms:.0f}ms"
            )
            return True

        if self._last_frame_ms > 0.0 and (now - self._last_frame_ms) >= self.camera_stall_release_ms:
            self._force_release_held_inputs(
                f"Camera stalled for {now - self._last_frame_ms:.0f}ms while input held"
            )
            return True

        return False

    def _dispatch_actions(self, outputs: List[HandOutput], primary: Optional[HandOutput], ts_ms: float) -> Optional[str]:
        """Dispatch OS events for Palm Wheels (both hands), Zoom, and primary cursor/gestures."""
        active_hint: Optional[str] = None
        now_ms = float(ts_ms)

        # Level-triggered max-drag check runs BEFORE any early return (even when primary is None)
        if self._drag_started_ms is not None and (now_ms - self._drag_started_ms) > self.max_drag_duration_ms:
            self._force_release_held_inputs("Max drag duration exceeded")

        # 1. Process Palm Wheel audio & actions across BOTH hands (even when paused, so user can unpause!)
        any_pw_active = False
        for ho in outputs:
            p = ho.pose
            if p.palm_wheel_press_event == "down":
                self.audio.play_wheel_tick()
            elif p.palm_wheel_press_event == "up":
                self.audio.play_click_up()
            if p.palm_wheel_state in ("active", "confirmed"):
                any_pw_active = True
            if p.palm_wheel_action:
                self._dispatch_palm_wheel_action(p.palm_wheel_action)
                return "palm_wheel"

        if self._paused:
            return None

        # 2. Two-Hand Pinch-to-Zoom takes priority when both hands actively zoom
        # Always honour dictation_cancelled across ANY hand before returning early for Zoom!
        for ho in outputs:
            if ho.pose.dictation_cancelled:
                self.dictation.cancel(reason="two_hand_zoom")
                self._last_action_text = "Dictation cancelled (Zoom)"

        z_out = self.engine.last_zoom
        if z_out.is_zooming:
            if self.dictation.is_recording:
                self.dictation.cancel(reason="two_hand_zoom")
            with self._lerp_lock:
                self._lerp_active = False
            if z_out.step != 0:
                self.audio.play_wheel_tick()
                self._practice["zoom"] = True
                self._last_action_text = "Zoom In" if z_out.step > 0 else "Zoom Out"
                self.input_ctl.zoom_step(z_out.step)
            return "zoom"

        if primary is None:
            self._enforce_input_safety(now_ms)
            return None

        p = primary.pose
        c = primary.cursor
        abs_x = self._screen_ox + c.x
        abs_y = self._screen_oy + c.y
        click_ho = self._select_click_hand(outputs, primary)
        cp = click_ho.pose if click_ho is not None else None

        if (
            p.is_grabbing
            or p.grab_just_started
            or p.is_scrolling
            or p.is_dictating
            or p.active_pose is not None
            or (cp is not None and (cp.is_grabbing or cp.grab_just_started))
        ):
            self._last_primary_ms = now_ms

        suppress_cursor = p.is_scrolling or p.is_spaces_nav or p.is_window_tiling or any_pw_active
        if c.is_pointing and c.is_valid and not suppress_cursor:
            self._move_accum_px += abs(c.delta_x) + abs(c.delta_y)
            if self._move_accum_px > 180.0:
                self._practice["move"] = True
            if self.enable_lerp_60hz:
                with self._lerp_lock:
                    self._lerp.set_target(abs_x, abs_y, speed_px_s=c.speed)
                    self._lerp_active = True
            else:
                self.input_ctl.mouse_move(abs_x, abs_y, delta_x=c.delta_x, delta_y=c.delta_y)
        else:
            with self._lerp_lock:
                self._lerp_active = False

        if cp is not None:
            if cp.is_pinching:
                self.audio.play_click_down()

            if cp.is_clicking:
                self.audio.play_click_up()
                cnt = max(1, cp.click_count)
                use_anchor = (click_ho is primary) and (cp.click_x is not None) and (cp.click_y is not None)
                clk_x = (self._screen_ox + float(cp.click_x)) if use_anchor else abs_x
                clk_y = (self._screen_oy + float(cp.click_y)) if use_anchor else abs_y
                self._last_action_text = "Double Click" if cnt == 2 else "Left Click"
                self._practice["click"] = True
                self.frame_ring.record_trigger(
                    "click", click_ho.hand, self._last_camera_ts_ms or int(now_ms)
                )
                self._emit_click_pulse("double_click" if cnt == 2 else "left_click", clk_x, clk_y)
                self.input_ctl.mouse_click(clk_x, clk_y, button="left", click_count=cnt)
                if use_anchor and math.hypot(clk_x - abs_x, clk_y - abs_y) > 2.0:
                    with self._lerp_lock:
                        self._lerp.sync_current_to_anchor(clk_x, clk_y, abs_x, abs_y)
                active_hint = "click"

            if cp.is_right_clicking:
                self.audio.play_click_up()
                use_rc_anchor = (
                    (click_ho is primary)
                    and (cp.right_click_x is not None)
                    and (cp.right_click_y is not None)
                )
                rc_x = (self._screen_ox + float(cp.right_click_x)) if use_rc_anchor else abs_x
                rc_y = (self._screen_oy + float(cp.right_click_y)) if use_rc_anchor else abs_y
                self._last_action_text = "Right Click"
                self._practice["right_click"] = True
                self.frame_ring.record_trigger(
                    "right_click", click_ho.hand, self._last_camera_ts_ms or int(now_ms)
                )
                self._emit_click_pulse("right_click", rc_x, rc_y)
                self.input_ctl.mouse_click(rc_x, rc_y, button="right", click_count=1)
                if use_rc_anchor and math.hypot(rc_x - abs_x, rc_y - abs_y) > 2.0:
                    with self._lerp_lock:
                        self._lerp.sync_current_to_anchor(rc_x, rc_y, abs_x, abs_y)
                active_hint = "right_click"

            if cp.grab_just_started:
                self.audio.play_click_down()
                self._drag_started_ms = now_ms
                self._last_primary_ms = now_ms
                if self._last_frame_ms <= 0.0:
                    self._last_frame_ms = now_ms
                self._mouse_button_held = True
                use_drag_anchor = (
                    (click_ho is primary)
                    and (cp.drag_x is not None)
                    and (cp.drag_y is not None)
                )
                dg_x = (self._screen_ox + float(cp.drag_x)) if use_drag_anchor else abs_x
                dg_y = (self._screen_oy + float(cp.drag_y)) if use_drag_anchor else abs_y
                self._last_action_text = "Drag Start"
                self._practice["drag"] = True
                self.frame_ring.record_trigger(
                    "drag_start", click_ho.hand, self._last_camera_ts_ms or int(now_ms)
                )
                self._emit_click_pulse("grab", dg_x, dg_y)
                self.input_ctl.mouse_down(dg_x, dg_y, button="left")
                if use_drag_anchor and math.hypot(dg_x - abs_x, dg_y - abs_y) > 2.0:
                    with self._lerp_lock:
                        self._lerp.sync_current_to_anchor(dg_x, dg_y, abs_x, abs_y)
            elif cp.grab_just_ended:
                self.audio.play_click_up()
                self._drag_started_ms = None
                self._mouse_button_held = False
                self._last_action_text = "Drag Release"
                self.input_ctl.mouse_up(abs_x, abs_y, button="left")
            if cp.is_grabbing:
                active_hint = "grab"

        # Voice Dictation hold gesture (`thumb_index_pinch`, arbitrated against 2-hand Zoom)
        if p.dictation_cancelled:
            self.dictation.cancel(reason="two_hand_zoom")
            self._last_action_text = "Dictation cancelled (Zoom)"
        elif p.dictation_just_started:
            self.audio.play_wheel_tick()
            self._practice["dictation"] = True
            pose_key = p.active_pose or p.pose
            self.frame_ring.record_trigger(
                "dictation_start",
                primary.hand,
                self._last_camera_ts_ms or int(now_ms),
                meta={"pose": pose_key.value},
            )
            self.dictation.start(
                trigger_meta={
                    "source": "gesture",
                    "hand": primary.hand,
                    "pose": pose_key.value,
                    "pose_prob": float(p.pose_probabilities.get(pose_key, 0.0)),
                    "dir_z": float(c.pointing_dir_z),
                    "palm_size": float(c.palm_size),
                }
            )
            self._last_action_text = "Dictation listening..."
        elif p.dictation_just_ended:
            self.audio.play_click_up()
            self.dictation.stop_and_transcribe()
            self._last_action_text = "Dictation transcribing..."
        if p.is_dictating or self.dictation.is_recording:
            active_hint = "dictation"

        # 2D Scroll with Momentum Phase
        scroll_use_phases = bool((self.config.get("scroll") or {}).get("use_phases", True))
        if p.is_scrolling:
            self._practice["scroll"] = True
            active_hint = "scroll"
            self._scroll_acc_x += p.scroll_dx
            self._scroll_acc_y += p.scroll_dy
            sx = int(self._scroll_acc_x)
            sy = int(self._scroll_acc_y)
            if sx != 0 or sy != 0:
                self._scroll_acc_x -= sx
                self._scroll_acc_y -= sy
                self._is_scroll_held = True
                self._last_action_text = f"Scroll ({sx:+d}, {sy:+d})"
                self.input_ctl.scroll_pixels(
                    sy, sx, use_phases=scroll_use_phases, is_inertia=p.is_scroll_inertia
                )
        else:
            self._scroll_acc_x = 0.0
            self._scroll_acc_y = 0.0
            if self._is_scroll_held:
                self._is_scroll_held = False
            self.input_ctl.end_scroll_phase()

        if p.is_spaces_nav:
            active_hint = "spaces_nav"
        if p.spaces_nav_direction == "left":
            self._last_action_text = "Space Left"
            self._practice["spaces_nav"] = True
            self.input_ctl.switch_space("left")
        elif p.spaces_nav_direction == "right":
            self._last_action_text = "Space Right"
            self._practice["spaces_nav"] = True
            self.input_ctl.switch_space("right")
        elif p.spaces_nav_direction == "top":
            self._last_action_text = "Mission Control"
            self._practice["spaces_nav"] = True
            self.input_ctl.toggle_mission_control()
        elif p.spaces_nav_direction == "bottom":
            self._last_action_text = "App Exposé"
            self._practice["spaces_nav"] = True
            self.input_ctl.exit_mission_control()

        if p.is_window_tiling:
            active_hint = "window_tile"
        if p.window_tile_direction is not None:
            wt_animate = bool((self.config.get("window_tile") or {}).get("animate", True))
            self._last_action_text = f"Tile Window: {p.window_tile_direction}"
            self._practice["window_tile"] = True
            self.input_ctl.tile_focused_window(p.window_tile_direction, animate=wt_animate)

        return active_hint

    def _create_tracker(self) -> MediaPipeHandTracker:
        trk_cfg = self.config.get("tracker") or {}
        return MediaPipeHandTracker(
            model_path=str(trk_cfg.get("model_path", "models/hand_landmarker.task")),
            num_hands=int(trk_cfg.get("num_hands", 2)),
            min_hand_detection_confidence=float(trk_cfg.get("min_hand_detection_confidence", 0.55)),
            min_hand_presence_confidence=float(trk_cfg.get("min_hand_presence_confidence", 0.55)),
            min_tracking_confidence=float(trk_cfg.get("min_tracking_confidence", 0.55)),
            delegate=str(trk_cfg.get("delegate", "CPU")),
            mirror_handedness=self.mirror,
        )

    def _create_camera(self) -> ThreadedCamera:
        return ThreadedCamera(
            camera_index=self.camera_index,
            width=self.cam_width,
            height=self.cam_height,
            fps_cap=self.fps_cap,
            mirror=self.mirror,
            aspect_ratio=self.cam_aspect,
            device_id=self.camera_device_id,
        )

    def _refresh_display_bounds(self) -> None:
        cur_cfg = self.config.get("cursor") or {}
        mode = str(cur_cfg.get("display", "cursor")).strip().lower()
        multi = bool((self.config.get("system") or {}).get("multi_monitor", True))
        if not multi and mode == "all":
            mode = "main"
        cursor_pt = self._cursor_pos_fn()
        self._screen_ox, self._screen_oy, self._screen_w, self._screen_h = (
            self.screen_geom.resolve_control_surface(mode=mode, cursor_pos=cursor_pt)
        )

    def _run_loop(self) -> None:
        self._processing_thread_id = threading.get_ident()
        self._flush_pending_config()
        perms = check_all_permissions()
        acc_ok = True if self.dry_run else perms.get("accessibility", False)
        self._refresh_display_bounds()
        cur_x, cur_y = self._cursor_pos_fn()
        self.engine.anchor_cursor("Right", cur_x - self._screen_ox, cur_y - self._screen_oy)
        self.engine.anchor_cursor("Left", cur_x - self._screen_ox, cur_y - self._screen_oy)
        with self._lerp_lock:
            self._lerp.reset((cur_x, cur_y))

        tracker: Optional[MediaPipeHandTracker] = None
        camera: Optional[ThreadedCamera] = None
        lerp_thread: Optional[threading.Thread] = None

        try:
            tracker = self._create_tracker()
            camera = self._create_camera()

            lerp_thread = threading.Thread(target=self._lerp_worker, name="AeroHandLerp60Hz", daemon=True)
            lerp_thread.start()

            logger.info(
                f"[AeroHand] Started v{__version__} ({'DRY-RUN' if self.dry_run else 'LIVE'} | "
                f"delegate={tracker.delegate_used} | display={int(self._screen_w)}x{int(self._screen_h)})"
            )

            # Wait gracefully while user grants macOS Camera permission on first launch
            cam_started = False
            while not self._stop_event.is_set() and not cam_started:
                try:
                    camera.start()
                    cam_started = True
                except RuntimeError as cam_exc:
                    logger.warning(f"[CAMERA] Waiting for camera access ({cam_exc}); retrying in 1.5s...")
                    with self._snapshot_lock:
                        self._snapshot = RuntimeSnapshot(
                            running=False,
                            dry_run=self.dry_run,
                            delegate=tracker.delegate_used,
                            permissions=check_all_permissions(),
                            last_action="Waiting for Camera permission...",
                            error=str(cam_exc),
                        )
                    self._stop_event.wait(timeout=1.5)
            if not cam_started:
                return
            last_hand_wall = time.monotonic()
            self._last_frame_ms = self._now_ms()
            fps_ema = 30.0
            prev_wall = time.monotonic()
            last_slow_poll = time.monotonic()

            while not self._stop_event.is_set():
                # Apply any queued UI/API live settings safely on the processing thread
                self._flush_pending_config()
                # Level-triggered stuck-input watchdog runs every iteration regardless of camera/hand state
                self._enforce_input_safety(self._now_ms())

                # Handle hot camera / tracker switches cleanly
                if self._restart_camera_event.is_set():
                    self._restart_camera_event.clear()
                    try:
                        camera.stop()
                        camera = self._create_camera()
                        camera.start()
                        logger.info("[CAMERA] Hot-restarted camera stream.")
                    except Exception as exc:
                        logger.error(f"[CAMERA] Hot-restart failed: {exc}")

                if self._restart_tracker_event.is_set():
                    self._restart_tracker_event.clear()
                    try:
                        tracker.close()
                        tracker = self._create_tracker()
                        logger.info("[TRACKER] Hot-restarted MediaPipe tracker.")
                    except Exception as exc:
                        logger.error(f"[TRACKER] Hot-restart failed: {exc}")

                now_wall = time.monotonic()
                if now_wall - last_slow_poll >= 1.0:
                    last_slow_poll = now_wall
                    perms = check_all_permissions()
                    acc_ok = True if self.dry_run else perms.get("accessibility", False)
                    self._refresh_display_bounds()

                    # Per-App Profile switching
                    bid = get_frontmost_app_bundle_id()
                    if bid != self._active_bundle_id:
                        self._active_bundle_id = bid
                        merged = resolve_profile_config(self.base_config, bid)
                        self.apply_live_settings(merged, from_profile=True)

                is_idle = self.idle_throttle_enabled and ((now_wall - last_hand_wall) >= self.idle_timeout_s)
                if (
                    is_idle
                    and not self.collector.is_recording
                    and not self.collector.is_cued_active
                    and self.passive_recorder is None
                ):
                    time.sleep(max(0.01, 1.0 / max(1.0, self.idle_fps) - 0.015))

                cf = camera.read(timeout_s=0.04)
                if not cf.success or cf.frame is None:
                    self._enforce_input_safety(self._now_ms())
                    continue

                frame_wall_ms = self._now_ms()
                self._last_frame_ms = frame_wall_ms
                dt_wall = max(1e-4, time.monotonic() - prev_wall)
                prev_wall = time.monotonic()
                fps_ema = 0.85 * fps_ema + 0.15 * (1.0 / dt_wall)

                t_mp0 = time.perf_counter()
                hands = tracker.process_frame(cf.frame, cf.timestamp_ms)
                mp_ms = (time.perf_counter() - t_mp0) * 1000.0

                if hands:
                    last_hand_wall = time.monotonic()
                    self._last_hand_ms = frame_wall_ms
                    if not self._hand_present_prev:
                        cx, cy = get_current_cursor_position()
                        for h_inp in hands:
                            self.engine.anchor_cursor(
                                h_inp.handedness, cx - self._screen_ox, cy - self._screen_oy
                            )
                        with self._lerp_lock:
                            self._lerp.reset((cx, cy))
                    self._hand_present_prev = True
                else:
                    if self._hand_present_prev:
                        self._hand_present_prev = False
                        with self._lerp_lock:
                            self._lerp_active = False

                outputs = self.engine.process_frame(
                    hands,
                    float(cf.timestamp_ms),
                    self._screen_w,
                    self._screen_h,
                    frame_brightness=cf.brightness,
                    frame_aspect=cf.frame_aspect,
                )
                self._last_camera_ts_ms = int(cf.timestamp_ms)
                out_by_hand = {o.hand: o for o in outputs}
                for h_inp in hands:
                    ho_match = out_by_hand.get(h_inp.handedness)
                    probs_d = (
                        {k.value: float(v) for k, v in ho_match.pose.pose_probabilities.items()}
                        if ho_match is not None
                        else None
                    )
                    self.frame_ring.push(
                        h_inp.landmarks,
                        h_inp.world_landmarks,
                        h_inp.handedness,
                        h_inp.confidence,
                        int(cf.timestamp_ms),
                        hand_id=h_inp.handedness,
                        probabilities=probs_d,
                    )
                primary = self._select_primary_hand(outputs)
                self._record_collector_frame(hands, primary, cf.timestamp_ms)
                if self.passive_recorder is not None:
                    self.passive_recorder.add_frame(hands, int(cf.timestamp_ms))
                if self.calibration.state == "running":
                    cal_hand = next(
                        (h for h in hands if primary is not None and h.handedness == primary.hand),
                        hands[0] if hands else None,
                    )
                    self.calibration.update(
                        cal_hand.landmarks if cal_hand is not None else None,
                        cal_hand.world_landmarks if cal_hand is not None else None,
                        cal_hand.handedness if cal_hand is not None else "Right",
                    )
                active_hint = self._dispatch_actions(outputs, primary, frame_wall_ms)
                self._enforce_input_safety(frame_wall_ms)

                t_ren0 = time.perf_counter()
                vis_frame: Optional[np.ndarray] = None
                if self.render_hud_frames:
                    hud_cfg = (self.config.get("ui") or {}).get("camera_hud") or {}
                    vis_frame = render_camera_hud(
                        cf.frame,
                        hands,
                        outputs,
                        primary,
                        fps=fps_ema,
                        paused=self._paused,
                        dry_run=self.dry_run,
                        is_idle=is_idle,
                        guard_reason=self.input_ctl.last_guard_reason,
                        dictating=self.dictation.is_recording,
                        zooming=self.engine.last_zoom.is_zooming,
                        show_connections=bool(hud_cfg.get("show_connections", True)),
                        show_landmarks=bool(hud_cfg.get("show_landmarks", True)),
                        compact=not self.preview,
                    )
                ren_ms = (time.perf_counter() - t_ren0) * 1000.0

                timing = self.engine.last_timing
                timing.mediapipe_ms = mp_ms
                timing.render_ms = ren_ms
                timing.total_ms = mp_ms + timing.pre_filters_ms + timing.cursor_ms + timing.gesture_ms + timing.gesture_ml_ms + ren_ms
                timing.headroom_ms = max(0.0, timing.frame_budget_ms - timing.total_ms)
                timing.utilization_pct = min(100.0, (timing.total_ms / timing.frame_budget_ms) * 100.0)

                # Build Palm Wheel payloads for On-Screen Radial Overlay
                pw_payloads: List[Dict[str, Any]] = []
                for ho in outputs:
                    po = ho.pose
                    if po.palm_wheel_state != "inactive":
                        pw_payloads.append(
                            {
                                "hand": ho.hand,
                                "state": po.palm_wheel_state,
                                "progress": round(po.palm_wheel_progress, 3),
                                "selected_finger": po.palm_wheel_selected_finger,
                                "menu_path": po.palm_wheel_menu_path,
                                "menu_label": po.palm_wheel_menu_label,
                                "slots": po.palm_wheel_slots,
                                "palm_center": (
                                    po.palm_wheel_finger_positions.get("palmCenter")
                                    if po.palm_wheel_finger_positions
                                    else None
                                ),
                                "cursor_x": round(self._screen_ox + ho.cursor.x, 1),
                                "cursor_y": round(self._screen_oy + ho.cursor.y, 1),
                            }
                        )

                # Build Window Tile Ghost Preview payload (in Quartz global coordinates on active display)
                wt_preview: Optional[Dict[str, Any]] = None
                if primary is not None and primary.pose.window_tile_preview_direction:
                    px_abs = self._screen_ox + primary.cursor.x
                    py_abs = self._screen_oy + primary.cursor.y
                    vis_rect = self.screen_geom.target_display_for((px_abs, py_abs)).visible_tuple()
                    rect = compute_tile_rect(
                        primary.pose.window_tile_preview_direction,
                        vis_rect,
                    )
                    if rect is not None:
                        wt_preview = {
                            "direction": primary.pose.window_tile_preview_direction,
                            "x": round(rect[0], 1),
                            "y": round(rect[1], 1),
                            "w": round(rect[2], 1),
                            "h": round(rect[3], 1),
                        }

                warn_dict = (
                    {"level": self.engine.last_warning.level, "code": self.engine.last_warning.code}
                    if self.engine.last_warning is not None
                    else None
                )
                probs_map = (
                    {k.value: round(float(v), 3) for k, v in primary.pose.pose_probabilities.items()}
                    if primary is not None
                    else {k.value: (1.0 if k == PoseName.IDLE else 0.0) for k in ALL_POSE_NAMES}
                )

                cls_tel = self.classifier_telemetry()
                if warn_dict is None and cls_tel.get("fell_back") and cls_tel.get("fallback_reason"):
                    warn_dict = {"level": "warning", "code": str(cls_tel["fallback_reason"])}

                dict_status = self.dictation.status_dict()
                refusal_code: Optional[str] = None
                if self._paused:
                    refusal_code = "paused"
                elif self.input_ctl.last_guard_reason:
                    refusal_code = "typing_guard_active"
                elif dict_status.get("discard_active") and dict_status.get("last_discard_reason"):
                    dr = str(dict_status["last_discard_reason"])
                    refusal_code = "two_hand_zoom_active" if dr == "two_hand_zoom" else f"dictation_{dr}"
                elif self.engine.last_refusal_code:
                    refusal_code = self.engine.last_refusal_code
                refusal_reason = format_refusal_reason(refusal_code, self.config)

                snap = RuntimeSnapshot(
                    running=True,
                    paused=self._paused,
                    dry_run=self.dry_run,
                    is_idle=is_idle,
                    fps=fps_ema,
                    delegate=tracker.delegate_used,
                    classifier_name=str(cls_tel.get("active_backend", type(self.classifier).__name__)),
                    classifier_telemetry=cls_tel,
                    accessibility_ok=acc_ok,
                    permissions=perms,
                    guard_reason=self.input_ctl.last_guard_reason,
                    active_app=self._active_bundle_id,
                    hand=primary.hand if primary is not None else None,
                    hands_count=len(outputs),
                    is_pointing=primary.cursor.is_pointing if primary is not None else False,
                    dir_z=primary.cursor.pointing_dir_z if primary is not None else 0.0,
                    cursor_x=round(self._screen_ox + primary.cursor.x, 1) if primary is not None else 0.0,
                    cursor_y=round(self._screen_oy + primary.cursor.y, 1) if primary is not None else 0.0,
                    pose=primary.pose.pose.value if primary is not None else "idle",
                    active_pose=(
                        primary.pose.active_pose.value
                        if (primary is not None and primary.pose.active_pose is not None)
                        else None
                    ),
                    is_scrolling=primary.pose.is_scrolling if primary is not None else False,
                    is_grabbing=primary.pose.is_grabbing if primary is not None else False,
                    is_zooming=self.engine.last_zoom.is_zooming,
                    palm_wheel_state=primary.pose.palm_wheel_state if primary is not None else "inactive",
                    palm_wheel_progress=primary.pose.palm_wheel_progress if primary is not None else 0.0,
                    palm_wheels=pw_payloads,
                    window_tile_preview=wt_preview,
                    spaces_nav_preview=(
                        primary.pose.spaces_nav_preview_direction if primary is not None else None
                    ),
                    active_hint=active_hint,
                    active_hint_text=format_live_hint_toast(active_hint, self.config),
                    refusal_code=refusal_code,
                    refusal_reason=refusal_reason,
                    warning=warn_dict,
                    dictation=dict_status,
                    timing=timing.to_dict(),
                    vitals=self.vitals_mon.read().to_dict(),
                    collector=self.collector_status(),
                    calibration=self.calibration.status(),
                    practice=dict(self._practice),
                    click_event_seq=self._click_seq,
                    click_event=self._last_click_event,
                    last_action=self._last_action_text,
                    probabilities=probs_map,
                    frame_bgr=vis_frame,
                )
                with self._snapshot_lock:
                    self._snapshot = snap

                if self.preview and vis_frame is not None:
                    if self._cli_collect_mode:
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
                    if self._cli_collect_mode:
                        if key == ord("r"):
                            st = self.collector.toggle_recording()
                            print(f"[COLLECT] recording={'ON' if st else 'OFF'} counts={self.collector.counts()}")
                        elif key == ord("s"):
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
            self._processing_thread_id = None
            self._stop_event.set()
            if self.collector._cued_samples or self.collector._cued_events:
                try:
                    self.collector.autosave_cued_draft(self.recall_dir)
                except Exception:
                    pass
            self._restore_cued_dry_run()
            if self.passive_recorder is not None:
                try:
                    summary = self.passive_recorder.close()
                    logger.info(
                        f"[PASSIVE] Saved {summary['chunks_written']} chunk(s) to {summary['out_dir']} | "
                        f"hand_present_idle_hours={summary['hand_present_idle_hours']:.4f}h | "
                        f"deliberate_marks={summary['deliberate_marks_count']}"
                    )
                except Exception:
                    pass
            if lerp_thread is not None and lerp_thread.is_alive():
                lerp_thread.join(timeout=0.5)
            self.dictation.close()
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
            if self._cli_collect_mode and sum(self.collector.counts().values()) > 0:
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
        self._start_api_if_enabled()
        try:
            self._run_loop()
        finally:
            if self._api_server is not None:
                self._api_server.stop()
                self._api_server = None
