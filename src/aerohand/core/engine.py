"""
HandCursorEngine — Central Airpoint-Parity Orchestrator for AeroHand.
Combines Airpoint SDK `HandCursorEngine.ts` (Apache-2.0) with GestureCtl safety:

Pipeline per frame:
  1. Stage-1 LandmarkSmoother (fast OneEuro, mc=2.0, beta=45.0, AM-7 gap-reset).
  2. Per-hand PalmWheelDetector (Left L1..L5 & Right R1..R5 with nested Submenus +
     motion-anchored 350 ms activation hold + palm cursor handoff).
  3. CursorKinematics (index tip LM8, Stage-2 OneEuro, distance compensation, accel,
     jump rejection, OS cursor anchor).
  4. 3D Forward-Pointing Gate with hysteresis (`(LM8 - LM0).z / len < -0.15`).
  5. Pluggable PinchClassifier (`heuristic_3d` or self-trained `onnx_tcn`) with
     fail-open exception safety.
  6. PoseStateMachine -> Click (single/double), Right-Click, Drag, Voice Dictation,
     Scroll (2D + inertia + momentum phase), Spaces Navigation, Window Tiling (with live
     preview before release), and Two-Hand Pinch-to-Zoom.
  7. Detailed per-stage PipelineTiming & smart TrackingWarning generation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from aerohand.core.filters import LandmarkSmoother
from aerohand.core.kinematics import CursorKinematics, isotropic_distance_2d
from aerohand.core.palm_frame import (
    INDEX_TIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    compute_hand_rotation,
    compute_palm_facing_dot,
    compute_palm_frame,
    compute_pointing_dir_z,
)
from aerohand.core.palm_wheel import PalmWheelDetector, PalmWheelOutput
from aerohand.core.pose_state_machine import PoseStateMachine
from aerohand.core.scroll_controller import ScrollController
from aerohand.core.spaces_nav import SpacesNavController
from aerohand.core.types import (
    CursorOutput,
    Handedness,
    HandInput,
    HandOutput,
    Landmark,
    PipelineTiming,
    PoseName,
    PoseOutput,
    TrackingWarning,
    ZoomOutput,
    idle_probabilities,
)
from aerohand.core.window_tile import WindowTileController
from aerohand.core.zoom_controller import ZoomController
from aerohand.i18n import get_lang_pack
from aerohand.ml.base import PinchClassifier

logger = logging.getLogger(__name__)

CALIBRATION_POSES: Tuple[PoseName, ...] = (
    PoseName.THUMB_INDEX_PINCH,
    PoseName.THUMB_MIDDLE_PINCH,
    PoseName.THUMB_RING_PINCH,
    PoseName.THUMB_PINKY_PINCH,
    PoseName.THUMB_PINKY_BASE,
    PoseName.THUMB_INDEX_MIDDLE_PINCH,
)
LOOSE_PINCH_STEP: str = "loose_pinch"
CALIBRATION_STEPS: Tuple[str, ...] = tuple(p.value for p in CALIBRATION_POSES) + (LOOSE_PINCH_STEP,)

# Safe threshold bounds so calibration can never create hyper-sensitive or unusable settings
SAFE_PINCH_ON_BOUNDS: Tuple[float, float] = (0.24, 0.46)
SAFE_PINCH_OFF_BOUNDS: Tuple[float, float] = (0.48, 0.78)
SAFE_PINKY_BASE_ON_BOUNDS: Tuple[float, float] = (0.30, 0.54)
SAFE_DUAL_PINCH_ON_BOUNDS: Tuple[float, float] = (0.28, 0.50)
SAFE_ENTER_THRESHOLD_BOUNDS: Tuple[float, float] = (0.65, 0.88)


def measure_pose_palm_distance(
    landmarks_2d: List[Landmark],
    world_landmarks: Optional[List[Landmark]],
    handedness: Handedness,
    pose: PoseName | str,
) -> Optional[float]:
    """Measure fingertip contact (or loose-pinch near-miss) distance in 3D `palm_frame` units."""
    src = world_landmarks if (world_landmarks is not None and len(world_landmarks) >= 21) else landmarks_2d
    pf = compute_palm_frame(src, anchor="palm_center", is_left_hand=(handedness == "Left"), scale_metric="palm_width")
    if pf is None:
        return None
    key = pose.value if isinstance(pose, PoseName) else str(pose)
    if key == LOOSE_PINCH_STEP:
        return min(pf.dist(THUMB_TIP, INDEX_TIP), pf.dist(THUMB_TIP, MIDDLE_TIP))
    if key == PoseName.THUMB_INDEX_PINCH.value:
        return pf.dist(THUMB_TIP, INDEX_TIP)
    if key == PoseName.THUMB_MIDDLE_PINCH.value:
        return pf.dist(THUMB_TIP, MIDDLE_TIP)
    if key == PoseName.THUMB_RING_PINCH.value:
        return pf.dist(THUMB_TIP, RING_TIP)
    if key == PoseName.THUMB_PINKY_PINCH.value:
        return pf.dist(THUMB_TIP, PINKY_TIP)
    if key == PoseName.THUMB_PINKY_BASE.value:
        return pf.dist(THUMB_TIP, PINKY_MCP)
    if key == PoseName.THUMB_INDEX_MIDDLE_PINCH.value:
        return max(pf.dist(THUMB_TIP, INDEX_TIP), pf.dist(THUMB_TIP, MIDDLE_TIP))
    return None


def suggest_calibrated_thresholds(medians: Dict[str, float]) -> Dict[str, Any]:
    """
    Convert per-pose median `palm_frame` contact distances (plus optional `loose_pinch` near-miss)
    into safe suggested overrides. Never exceeds documented safe bounds.
    """
    def _clamp(val: float, lo: float, hi: float) -> float:
        return round(max(lo, min(hi, float(val))), 3)

    single_vals = [
        float(medians[p.value])
        for p in (
            PoseName.THUMB_INDEX_PINCH,
            PoseName.THUMB_MIDDLE_PINCH,
            PoseName.THUMB_RING_PINCH,
            PoseName.THUMB_PINKY_PINCH,
        )
        if p.value in medians and medians[p.value] > 0.0
    ]
    if single_vals:
        med_single = sorted(single_vals)[len(single_vals) // 2]
        raw_pinch_on = med_single + 0.08
    else:
        raw_pinch_on = 0.34

    loose_raw = medians.get(LOOSE_PINCH_STEP)
    if loose_raw is not None and float(loose_raw) > 0.0:
        # Keep pinch_on safely below user's loose near-miss distance so 1-2 cm hover won't false-trigger
        raw_pinch_on = min(raw_pinch_on, float(loose_raw) - 0.06)
        pinch_on = _clamp(raw_pinch_on, *SAFE_PINCH_ON_BOUNDS)
        pinch_off = _clamp(max(pinch_on + 0.16, float(loose_raw) + 0.05), *SAFE_PINCH_OFF_BOUNDS)
    else:
        pinch_on = _clamp(raw_pinch_on, *SAFE_PINCH_ON_BOUNDS)
        pinch_off = _clamp(max(0.62, pinch_on + 0.24), *SAFE_PINCH_OFF_BOUNDS)

    pb_raw = medians.get(PoseName.THUMB_PINKY_BASE.value)
    pinky_base_on = _clamp((pb_raw + 0.09) if pb_raw else 0.42, *SAFE_PINKY_BASE_ON_BOUNDS)

    dual_raw = medians.get(PoseName.THUMB_INDEX_MIDDLE_PINCH.value)
    dual_on = _clamp((dual_raw + 0.08) if dual_raw else 0.38, *SAFE_DUAL_PINCH_ON_BOUNDS)

    enter_thresholds: Dict[str, float] = {}
    for p in CALIBRATION_POSES:
        d = medians.get(p.value)
        if d is None or d <= 0.0:
            enter_thresholds[p.value] = 0.75
        else:
            raw_th = 0.82 - max(0.0, (d - 0.22)) * 0.45
            enter_thresholds[p.value] = _clamp(raw_th, *SAFE_ENTER_THRESHOLD_BOUNDS)

    return {
        "pinch_on_dist": pinch_on,
        "pinch_off_dist": pinch_off,
        "pinky_base_on_dist": pinky_base_on,
        "dual_pinch_on_dist": dual_on,
        "enter_thresholds": enter_thresholds,
        "medians": {k: round(float(v), 3) for k, v in medians.items()},
    }


class PinchCalibrationWizard:
    """Interactive personal calibration wizard (6 pinch poses + 1 loose-pinch near-miss step)."""

    def __init__(self, hold_seconds: float = 3.0) -> None:
        self.hold_seconds: float = max(1.0, float(hold_seconds))
        self.state: str = "idle"  # "idle" | "running" | "done"
        self.step_index: int = 0
        self._step_started_s: float = 0.0
        self._samples: Dict[str, List[float]] = {k: [] for k in CALIBRATION_STEPS}
        self.suggestion: Optional[Dict[str, Any]] = None

    def start(self, now_s: Optional[float] = None) -> None:
        t = time.monotonic() if now_s is None else float(now_s)
        self.state = "running"
        self.step_index = 0
        self._step_started_s = t
        self._samples = {k: [] for k in CALIBRATION_STEPS}
        self.suggestion = None

    def cancel(self) -> None:
        self.state = "idle"
        self.step_index = 0
        self.suggestion = None

    def update(
        self,
        landmarks_2d: Optional[List[Landmark]],
        world_landmarks: Optional[List[Landmark]],
        handedness: Handedness,
        now_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self.state != "running":
            return self.status(now_s=now_s)
        t = time.monotonic() if now_s is None else float(now_s)
        step_key = CALIBRATION_STEPS[self.step_index]
        if landmarks_2d is not None and len(landmarks_2d) >= 21:
            d = measure_pose_palm_distance(landmarks_2d, world_landmarks, handedness, step_key)
            if d is not None and 0.01 < d < 1.8:
                self._samples[step_key].append(d)

        elapsed = t - self._step_started_s
        if elapsed >= self.hold_seconds:
            self.step_index += 1
            self._step_started_s = t
            if self.step_index >= len(CALIBRATION_STEPS):
                medians: Dict[str, float] = {}
                for k in CALIBRATION_STEPS:
                    vals = sorted(self._samples[k])
                    if vals:
                        medians[k] = vals[len(vals) // 2]
                self.suggestion = suggest_calibrated_thresholds(medians)
                self.state = "done"
        return self.status(now_s=t)

    def status(self, now_s: Optional[float] = None) -> Dict[str, Any]:
        t = time.monotonic() if now_s is None else float(now_s)
        cur_pose = (
            CALIBRATION_STEPS[self.step_index]
            if self.state == "running" and self.step_index < len(CALIBRATION_STEPS)
            else None
        )
        remaining_s = (
            max(0.0, self.hold_seconds - (t - self._step_started_s))
            if self.state == "running"
            else 0.0
        )
        return {
            "state": self.state,
            "step_index": self.step_index,
            "total_steps": len(CALIBRATION_STEPS),
            "current_pose": cur_pose,
            "remaining_s": round(remaining_s, 1),
            "sample_counts": {k: len(v) for k, v in self._samples.items()},
            "suggestion": self.suggestion,
        }


def format_live_hint_toast(
    hint_key: Optional[str],
    config: Dict[str, Any],
    lang: Optional[str] = None,
) -> Optional[str]:
    """
    Build a bilingual contextual toast message reading the LIVE gesture map
    (`pose_state_machine.tap_actions` / `hold_actions`) with zero hard-coded gesture names.
    """
    if not hint_key:
        return None
    eff_lang = lang or str((config.get("ui") or {}).get("language", "vi"))
    pack = get_lang_pack(eff_lang)
    base_label = (pack.get("hints") or {}).get(hint_key) or hint_key
    poses_i18n = pack.get("poses") or {}

    psm = config.get("pose_state_machine") or {}
    tap_map = psm.get("tap_actions") or {}
    hold_map = psm.get("hold_actions") or {}
    click_hold = bool(psm.get("enable_click_hold", False))

    action_lookup = {
        "click": ("tap", "left_click"),
        "right_click": ("dual_right", "right_click"),
        "grab": ("hold_drag", "drag"),
        "scroll": ("hold", "scroll"),
        "window_tile": ("hold", "window_tile"),
        "spaces_nav": ("hold", "spaces_nav"),
        "dictation": ("hold", "dictation"),
    }
    entry = action_lookup.get(hint_key)
    if entry is None:
        return str(base_label)

    kind, act_name = entry
    matched_poses: List[str] = []
    if kind == "tap":
        matched_poses = [pk for pk, act in tap_map.items() if act == act_name]
    elif kind == "hold":
        matched_poses = [pk for pk, act in hold_map.items() if act == act_name]
    elif kind == "hold_drag":
        matched_poses = [pk for pk, act in hold_map.items() if act == "drag"]
        if click_hold:
            for pk, act in tap_map.items():
                if act == "left_click" and pk not in matched_poses:
                    matched_poses.append(pk)
    elif kind == "dual_right":
        matched_poses = [pk for pk, act in tap_map.items() if act == "right_click"]
        if not matched_poses:
            matched_poses = [pk for pk, act in hold_map.items() if act == "scroll"]

    if not matched_poses:
        return str(base_label)
    pose_names = " / ".join(poses_i18n.get(pk, pk) for pk in matched_poses)
    return f"{base_label} • {pose_names}"


def format_refusal_reason(
    code: Optional[str],
    config: Dict[str, Any],
    *,
    lang: Optional[str] = None,
) -> Optional[str]:
    """
    Format the live bilingual "Why didn't it trigger?" reason line reading the LIVE gesture map
    and active settings (zero hard-coded gesture names, V040-04).
    """
    if not code:
        return None
    eff_lang = lang or str((config.get("ui") or {}).get("language", "vi"))
    pack = get_lang_pack(eff_lang)
    reasons = pack.get("refusal_reasons") or {}
    poses_i18n = pack.get("poses") or {}
    hands_i18n = pack.get("hands") or {}
    tmpl = reasons.get(code)
    if not tmpl:
        return str(code)

    psm = config.get("pose_state_machine") or {}
    tap_map = psm.get("tap_actions") or {}
    hold_map = psm.get("hold_actions") or {}
    click_poses = [pk for pk, act in tap_map.items() if act == "left_click"]
    dict_poses = [pk for pk, act in hold_map.items() if act == "dictation"]

    click_pose_lbl = " / ".join(poses_i18n.get(pk, pk) for pk in click_poses) or (
        (pack.get("actions") or {}).get("left_click", "Left Click")
    )
    dict_pose_lbl = " / ".join(poses_i18n.get(pk, pk) for pk in dict_poses) or (
        (pack.get("actions") or {}).get("dictation", "Dictation")
    )
    raw_click_hand = str((config.get("hand") or {}).get("click_hand", "Right"))
    hand_key = "same_as_cursor" if raw_click_hand.lower() == "same" else raw_click_hand.lower()
    click_hand_lbl = hands_i18n.get(hand_key, raw_click_hand)

    dir_z_th = float((config.get("cursor") or {}).get("pointing_forward_z_threshold", -0.15))
    arm_ms = int(round(float((config.get("gestures") or {}).get("dictation_arm_ms", 350))))

    return str(tmpl).format(
        click_pose=click_pose_lbl,
        dictation_pose=dict_pose_lbl,
        click_hand=click_hand_lbl,
        dir_z=f"{dir_z_th:+.2f}",
        arm_ms=arm_ms,
    )


@dataclass
class _PerHandState:
    hand: Handedness
    kinematics: CursorKinematics
    pose_sm: PoseStateMachine
    scroll: ScrollController
    spaces_nav: SpacesNavController
    window_tile: WindowTileController
    palm_wheel: PalmWheelDetector
    is_grabbing: bool = False
    is_dictating: bool = False
    dictation_pose_enter_ms: Optional[float] = None
    dictation_started_ms: Optional[float] = None
    zoom_pose_enter_ms: Optional[float] = None
    is_pointing_latched: bool = False
    last_click_ms: float = -1e9
    last_click_count: int = 0
    palm_wheel_was_active: bool = False
    palm_wheel_offset: Tuple[float, float] = (0.0, 0.0)
    cursor_delta: Tuple[float, float] = (0.0, 0.0)
    last_seen_ms: float = -1e9
    click_anchor_pos: Optional[Tuple[float, float]] = None
    drag_anchor_pos: Optional[Tuple[float, float]] = None
    right_click_anchor_pos: Optional[Tuple[float, float]] = None
    prev_click_prob: float = 0.0
    dwell_lock_until_ms: float = 0.0
    dwell_lock_pos: Optional[Tuple[float, float]] = None
    dwell_lock_used_in_attempt: bool = False
    dwell_lock_cooldown_until_ms: float = 0.0
    last_low_click_prob_ms: Optional[float] = None
    prev_drag_prob: float = 0.0
    drag_dwell_lock_until_ms: float = 0.0
    drag_dwell_lock_pos: Optional[Tuple[float, float]] = None
    drag_dwell_lock_used: bool = False

    def reset(self) -> None:
        self.kinematics.reset()
        self.pose_sm.reset()
        self.scroll.reset()
        self.spaces_nav.reset()
        self.window_tile.reset()
        self.palm_wheel.reset()
        self.is_grabbing = False
        self.is_dictating = False
        self.dictation_pose_enter_ms = None
        self.dictation_started_ms = None
        self.zoom_pose_enter_ms = None
        self.is_pointing_latched = False
        self.palm_wheel_was_active = False
        self.palm_wheel_offset = (0.0, 0.0)
        self.cursor_delta = (0.0, 0.0)
        self.click_anchor_pos = None
        self.drag_anchor_pos = None
        self.right_click_anchor_pos = None
        self.prev_click_prob = 0.0
        self.dwell_lock_until_ms = 0.0
        self.dwell_lock_pos = None
        self.dwell_lock_used_in_attempt = False
        self.dwell_lock_cooldown_until_ms = 0.0
        self.last_low_click_prob_ms = None
        self.prev_drag_prob = 0.0
        self.drag_dwell_lock_until_ms = 0.0
        self.drag_dwell_lock_pos = None
        self.drag_dwell_lock_used = False


class HandCursorEngine:
    """Unified 2-hand cursor, gesture, Palm Wheel, Dictation, and Zoom engine."""

    def __init__(
        self,
        config: Dict[str, Any],
        classifier: Optional[PinchClassifier] = None,
        on_classifier_error: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.classifier = classifier
        self._on_classifier_error = on_classifier_error
        self.smoother = LandmarkSmoother()
        self.zoom = ZoomController(config.get("gestures"))
        self._hands: Dict[Handedness, _PerHandState] = {
            "Left": self._make_hand_state("Left"),
            "Right": self._make_hand_state("Right"),
        }
        self._active_hands: set[Handedness] = set()
        self.last_zoom: ZoomOutput = ZoomOutput()
        self.last_timing: PipelineTiming = PipelineTiming()
        self.last_warning: Optional[TrackingWarning] = None
        self.last_refusal_code: Optional[str] = None
        self.dwell_lock_activations: int = 0
        self.dwell_unlock_without_click: int = 0
        self._classifier_lock = threading.RLock()
        self.update_config(config)

    def engine_telemetry(self) -> Dict[str, Any]:
        return {
            "dwell_lock_activations": self.dwell_lock_activations,
            "dwell_unlock_without_click": self.dwell_unlock_without_click,
        }

    def _make_hand_state(self, hand: Handedness) -> _PerHandState:
        return _PerHandState(
            hand=hand,
            kinematics=CursorKinematics(self.config.get("cursor")),
            pose_sm=PoseStateMachine(self.config.get("pose_state_machine")),
            scroll=ScrollController(self.config.get("scroll")),
            spaces_nav=SpacesNavController(self.config.get("spaces_nav")),
            window_tile=WindowTileController(self.config.get("window_tile")),
            palm_wheel=PalmWheelDetector(self.config.get("palm_wheel"), hand=hand),
        )

    def update_config(self, config: Dict[str, Any]) -> None:
        """Hot-reload every engine parameter cleanly without losing active cursor position."""
        self.config = config

        sm_cfg = config.get("smoothing") or {}
        self.enable_landmark_smoothing: bool = bool(sm_cfg.get("enable_landmark_smoothing", True))
        prev_excl = getattr(self, "landmark_exclude_joints", None)
        new_excl = list(sm_cfg.get("landmark_exclude_joints") or [])
        self.landmark_exclude_joints: List[int] = new_excl
        self.smoother.update_config(
            min_cutoff=float(sm_cfg.get("landmark_min_cutoff", 2.0)),
            beta=float(sm_cfg.get("landmark_beta", 45.0)),
            d_cutoff=float(sm_cfg.get("landmark_d_cutoff", 1.0)),
        )
        if prev_excl is not None and prev_excl != new_excl:
            self.smoother.reset()

        hand_cfg = config.get("hand") or {}
        self.cursor_hand: str = str(hand_cfg.get("cursor_hand", "Right"))
        self.click_hand: str = str(hand_cfg.get("click_hand", "Right"))

        gest_cfg = config.get("gestures") or {}
        self.enable_click: bool = bool(gest_cfg.get("enable_click", True))
        self.enable_right_click: bool = bool(gest_cfg.get("enable_right_click", True))
        self.enable_grab: bool = bool(gest_cfg.get("enable_grab", True))
        self.enable_scroll: bool = bool(gest_cfg.get("enable_scroll", True))
        self.enable_window_tile: bool = bool(gest_cfg.get("enable_window_tile", True))
        self.enable_spaces_nav: bool = bool(gest_cfg.get("enable_spaces_nav", True))
        self.enable_dictation: bool = bool(gest_cfg.get("enable_dictation", True))
        self.enable_zoom: bool = bool(gest_cfg.get("enable_zoom", True))
        try:
            self.zoom_pose: PoseName = PoseName(str(gest_cfg.get("zoom_pose", "thumb_index_pinch")))
        except ValueError:
            self.zoom_pose = PoseName.THUMB_INDEX_PINCH
        self.zoom_arbitration_ms: float = max(0.0, min(1000.0, float(gest_cfg.get("zoom_arbitration_ms", 300.0))))
        self.dictation_arm_ms: float = max(200.0, min(800.0, float(gest_cfg.get("dictation_arm_ms", 350.0))))
        self.dictation_priority_ms: float = max(0.0, min(5000.0, float(gest_cfg.get("dictation_priority_ms", 1000.0))))
        self.latch_gate_while_pose_active: bool = bool(gest_cfg.get("latch_gate_while_pose_active", True))
        self.zoom.update_config(gest_cfg)

        cur_cfg = config.get("cursor") or {}
        self.enable_pointing_gate: bool = bool(cur_cfg.get("enable_pointing_gate", True))
        self.pointing_forward_z_threshold: float = float(cur_cfg.get("pointing_forward_z_threshold", -0.15))
        self.pointing_gate_hysteresis: float = float(cur_cfg.get("pointing_gate_hysteresis", 0.04))
        self.click_lookback_ms: float = max(0.0, float(cur_cfg.get("click_lookback_ms", 90.0)))
        self.click_lookback_max_speed_px_s: float = max(
            0.0, float(cur_cfg.get("click_lookback_max_speed_px_s", 250.0))
        )
        self.click_lookback_max_offset_px: float = max(
            0.0, float(cur_cfg.get("click_lookback_max_offset_px", 24.0))
        )
        self.click_dwell_lock_ms: float = max(0.0, float(cur_cfg.get("click_dwell_lock_ms", 120.0)))
        self.click_dwell_lock_cooldown_ms: float = max(
            0.0, float(cur_cfg.get("click_dwell_lock_cooldown_ms", 400.0))
        )
        self.drag_lookback_ms: float = max(0.0, float(cur_cfg.get("drag_lookback_ms", 60.0)))
        self.drag_dwell_lock_ms: float = max(0.0, float(cur_cfg.get("drag_dwell_lock_ms", 80.0)))

        psm_cfg = config.get("pose_state_machine") or {}
        self.enable_click_debounce: bool = bool(psm_cfg.get("enable_click_debounce", True))
        self.click_debounce_ms: float = float(psm_cfg.get("click_debounce_ms", 200.0))
        self.double_click_window_ms: float = float(psm_cfg.get("double_click_window_ms", 350.0))
        self.still_right_click_motion_px: float = float(psm_cfg.get("pose_hold_motion_threshold_px", 15.0))

        cls_cfg = (config.get("classifier") or {}).get("heuristic_3d") or {}
        self.require_pointing_gate_for_actions: bool = bool(
            cls_cfg.get("require_pointing_gate_for_actions", True)
        )
        self.spaces_nav_palm_facing_min: float = max(
            0.0, min(1.0, float(cls_cfg.get("spaces_nav_palm_facing_min", 0.55)))
        )

        scr_cfg = config.get("scroll") or {}
        wt_cfg = config.get("window_tile") or {}
        sn_cfg = config.get("spaces_nav") or {}
        pw_cfg = config.get("palm_wheel") or {}

        for hs in self._hands.values():
            hs.kinematics.enable_filter = bool(cur_cfg.get("enable_filter", True))
            hs.kinematics.filter_x.min_cutoff = float(cur_cfg.get("filter_min_cutoff", 1.0))
            hs.kinematics.filter_x.beta = float(cur_cfg.get("filter_beta", 0.01))
            hs.kinematics.filter_y.min_cutoff = float(cur_cfg.get("filter_min_cutoff", 1.0))
            hs.kinematics.filter_y.beta = float(cur_cfg.get("filter_beta", 0.01))
            hs.kinematics.pointer_sensitivity = float(cur_cfg.get("pointer_sensitivity", 2.2))
            hs.kinematics.pointer_acceleration = float(cur_cfg.get("pointer_acceleration", 1.4))
            hs.kinematics.enable_distance_compensation = bool(cur_cfg.get("enable_distance_compensation", True))
            hs.kinematics.reference_palm_size = float(cur_cfg.get("reference_palm_size", 0.10))

            hs.pose_sm.update_config(psm_cfg)

            hs.scroll.enabled = self.enable_scroll and bool(scr_cfg.get("enabled", True))
            hs.scroll.speed = float(scr_cfg.get("speed", 1.0))
            hs.scroll.deadzone = float(scr_cfg.get("deadzone", 1.0))
            hs.scroll.enable_axis_lock = bool(scr_cfg.get("enable_axis_lock", True))
            hs.scroll.inertia_friction = float(scr_cfg.get("inertia_friction", 0.95))
            hs.scroll.natural_scrolling = bool(scr_cfg.get("natural_scrolling", True))
            hs.scroll.min_inertia_speed = (
                float(scr_cfg.get("min_inertia_speed", 5.0))
                if bool(scr_cfg.get("enable_inertia", True))
                else 1e9
            )

            hs.window_tile.enabled = self.enable_window_tile and bool(wt_cfg.get("enabled", True))
            hs.window_tile.sensitivity_px = float(wt_cfg.get("sensitivity_px", 20.0))

            hs.spaces_nav.enabled = self.enable_spaces_nav and bool(sn_cfg.get("enabled", True))
            hs.spaces_nav.sensitivity_px = float(sn_cfg.get("sensitivity_px", 20.0))
            hs.spaces_nav.invert = bool(sn_cfg.get("invert", True))

            hs.palm_wheel.update_config(pw_cfg)

    def _reset_classifier(self, hand: Optional[Handedness] = None) -> None:
        with self._classifier_lock:
            clf = self.classifier
        if clf is None:
            return
        try:
            clf.reset(hand)  # type: ignore[call-arg]
        except TypeError:
            if hand is None:
                clf.reset()

    def set_classifier(self, classifier: Optional[PinchClassifier]) -> None:
        """Hot-swap active classifier atomically while clearing temporal buffers cleanly (V051-06)."""
        with self._classifier_lock:
            old_clf = self.classifier
            self.classifier = classifier
        for clf in (old_clf, classifier):
            if clf is None:
                continue
            try:
                clf.reset(None)  # type: ignore[call-arg]
            except TypeError:
                clf.reset()

    def reset(self) -> None:
        self.smoother.reset()
        self.zoom.reset()
        for hs in self._hands.values():
            hs.reset()
        self._active_hands.clear()
        self.last_zoom = ZoomOutput()
        self.last_warning = None
        self._reset_classifier(None)

    def anchor_cursor(self, hand: Handedness, x: float, y: float) -> None:
        """Synchronize engine pointer state to real OS cursor without jumping."""
        self._hands[hand].kinematics.set_cursor_position(x, y)

    def recenter(self, width: float, height: float, hand: Optional[Handedness] = None) -> None:
        targets: Tuple[Handedness, ...] = (hand,) if hand else ("Left", "Right")
        for h in targets:
            hs = self._hands[h]
            hs.kinematics.set_cursor_position(width * 0.5, height * 0.5)
            hs.pose_sm.reset()
            hs.scroll.reset()
            hs.spaces_nav.reset()
            hs.window_tile.reset()
            hs.is_grabbing = False
            hs.is_dictating = False
            hs.dictation_pose_enter_ms = None
            hs.dictation_started_ms = None
            hs.zoom_pose_enter_ms = None

    def _effective_click_hand(self) -> str:
        return self.cursor_hand if str(self.click_hand).lower() == "same" else self.click_hand

    def _is_hand_allowed_for_click_actions(self, hand: Handedness) -> bool:
        eff = self._effective_click_hand()
        return eff == "Both" or eff == hand

    def _is_click_allowed_for_hand(self, hand: Handedness) -> bool:
        return self.enable_click and self._is_hand_allowed_for_click_actions(hand)

    def _update_pointing_gate(self, hs: _PerHandState, dir_z: float) -> bool:
        if not self.enable_pointing_gate:
            hs.is_pointing_latched = True
            return True
        if not hs.is_pointing_latched:
            if dir_z < self.pointing_forward_z_threshold:
                hs.is_pointing_latched = True
        else:
            if dir_z > (self.pointing_forward_z_threshold + self.pointing_gate_hysteresis):
                hs.is_pointing_latched = False
        return hs.is_pointing_latched

    @staticmethod
    def _evaluate_warning(
        lms: List[Any],
        palm_size: float,
        confidence: float,
        frame_brightness: Optional[float],
    ) -> Optional[TrackingWarning]:
        if frame_brightness is not None and frame_brightness < 38.0:
            return TrackingWarning(level="warning", code="low_light")
        if palm_size > 0.0:
            if palm_size < 0.055:
                return TrackingWarning(level="warning", code="hand_too_far")
            if palm_size > 0.34:
                return TrackingWarning(level="warning", code="hand_too_close")
        w = lms[0]
        tip = lms[8]
        if min(w.x, tip.x) < 0.03 or max(w.x, tip.x) > 0.97 or min(w.y, tip.y) < 0.03 or max(w.y, tip.y) > 0.97:
            return TrackingWarning(level="warning", code="near_edge")
        if confidence < 0.58:
            return TrackingWarning(level="warning", code="low_confidence")
        return None

    def process_frame(
        self,
        inputs: List[HandInput],
        timestamp_ms: float,
        width: float,
        height: float,
        *,
        frame_brightness: Optional[float] = None,
        frame_aspect: float = 4.0 / 3.0,
    ) -> List[HandOutput]:
        """Process one camera frame of 0..2 detected hands."""
        t0 = time.perf_counter()
        pre_ms = 0.0
        cur_ms = 0.0
        gest_ms = 0.0
        ml_ms = 0.0

        seen: set[Handedness] = set()
        outputs: List[HandOutput] = []
        warning: Optional[TrackingWarning] = None

        hand_pinch_pts: Dict[Handedness, Tuple[float, float]] = {}
        hand_pinching: Dict[Handedness, bool] = {}

        for inp in inputs:
            hand = inp.handedness
            seen.add(hand)
            hs = self._hands[hand]
            hs.last_seen_ms = timestamp_ms

            raw_lms = inp.landmarks
            if len(raw_lms) < 21:
                continue

            # 1. Stage-1 Landmark Smoothing
            tp0 = time.perf_counter()
            if self.enable_landmark_smoothing:
                lms = self.smoother.smooth(raw_lms, hand, timestamp_ms, self.landmark_exclude_joints)
            else:
                lms = list(raw_lms)
            pre_ms += (time.perf_counter() - tp0) * 1000.0

            # 2. Palm Wheel detection FIRST so cursor smoothly switches to palm center
            # Suppress Palm Wheel while an active pinch pose (e.g. drag/scroll) is held
            tg0 = time.perf_counter()
            if hs.pose_sm.active_pose is not None:
                hs.palm_wheel.reset()
                pw_out = PalmWheelOutput()
            else:
                pw_out = hs.palm_wheel.process(lms, timestamp_ms, handedness=hand)
            pw_active = pw_out.state in ("active", "confirmed")
            gest_ms += (time.perf_counter() - tg0) * 1000.0

            # 3. Cursor Kinematics
            tc0 = time.perf_counter()
            index_tip = lms[8]
            wrist = lms[0]
            middle_mcp = lms[9]
            palm_size = isotropic_distance_2d(wrist, middle_mcp, frame_aspect)

            index_x = index_tip.x * width
            index_y = index_tip.y * height
            palm_x = 0.5 * (wrist.x + middle_mcp.x) * width
            palm_y = 0.5 * (wrist.y + middle_mcp.y) * height

            if pw_active:
                if not hs.palm_wheel_was_active:
                    hs.palm_wheel_offset = (index_x - palm_x, index_y - palm_y)
                    hs.palm_wheel_was_active = True
                src_x = palm_x + hs.palm_wheel_offset[0]
                src_y = palm_y + hs.palm_wheel_offset[1]
            else:
                if hs.palm_wheel_was_active:
                    cur_px = palm_x + hs.palm_wheel_offset[0]
                    cur_py = palm_y + hs.palm_wheel_offset[1]
                    hs.palm_wheel_offset = (cur_px - index_x, cur_py - index_y)
                    hs.palm_wheel_was_active = False
                src_x = index_x + hs.palm_wheel_offset[0]
                src_y = index_y + hs.palm_wheel_offset[1]
                ox, oy = hs.palm_wheel_offset[0] * 0.9, hs.palm_wheel_offset[1] * 0.9
                if abs(ox) < 0.5:
                    ox = 0.0
                if abs(oy) < 0.5:
                    oy = 0.0
                hs.palm_wheel_offset = (ox, oy)

            kin = hs.kinematics.process(
                src_x, src_y, lms, timestamp_ms, width, height, frame_aspect=frame_aspect
            )
            hs.cursor_delta = (kin.delta_x, kin.delta_y) if kin.is_valid else (0.0, 0.0)

            dir_z = compute_pointing_dir_z(lms)
            is_pointing = self._update_pointing_gate(hs, dir_z)
            rotation = compute_hand_rotation(lms)
            cur_ms += (time.perf_counter() - tc0) * 1000.0

            if warning is None:
                warning = self._evaluate_warning(lms, palm_size, inp.confidence, frame_brightness)

            cursor_out = CursorOutput(
                hand=hand,
                x=kin.x,
                y=kin.y,
                z=index_tip.z,
                delta_x=kin.delta_x,
                delta_y=kin.delta_y,
                speed=kin.speed,
                finger_speed=kin.finger_speed,
                rotation=rotation,
                is_pointing=is_pointing,
                pointing_dir_z=dir_z,
                is_valid=kin.is_valid,
                handedness_score=inp.confidence,
                palm_size=palm_size,
            )

            # 5. Pluggable Pinch Classification (fail-open to IDLE; latch gate while pose is active)
            # Upright open 4-finger palm facing the camera (`palm_facing_dot >= spaces_nav_palm_facing_min`)
            # may classify ONLY `thumb_pinky_base` (Spaces Nav) even when fingers point upward (`dirZ > -0.15`),
            # while all other pinch actions remain strictly gated by the forward-pointing `dirZ` gate.
            pose_probs = idle_probabilities()
            pred_pose = PoseName.IDLE
            pose_active = hs.pose_sm.active_pose is not None
            gate_allows = (
                (not self.require_pointing_gate_for_actions)
                or is_pointing
                or (self.latch_gate_while_pose_active and pose_active)
            )
            spaces_palm_open = False
            if (not gate_allows) and self.enable_spaces_nav and (not pw_active):
                facing_dot = compute_palm_facing_dot(lms, is_left_hand=(hand == "Left"), mirrored=True)
                spaces_palm_open = facing_dot >= self.spaces_nav_palm_facing_min
            spaces_only_mode = (
                self.require_pointing_gate_for_actions
                and (not is_pointing)
                and (
                    spaces_palm_open
                    or (self.latch_gate_while_pose_active and hs.pose_sm.active_pose == PoseName.THUMB_PINKY_BASE)
                )
            )
            tm0 = time.perf_counter()
            with self._classifier_lock:
                active_clf = self.classifier
            if active_clf is not None and not pw_active and (gate_allows or spaces_palm_open):
                try:
                    if not spaces_only_mode:
                        pred = active_clf.predict(
                            landmarks_2d=lms,
                            world_landmarks=inp.world_landmarks,
                            handedness=hand,
                            timestamp_ms=timestamp_ms,
                        )
                        pred_pose = pred.pose
                        pose_probs = pred.probabilities
                    else:
                        predict_only = getattr(active_clf, "predict_only_pose", None)
                        if callable(predict_only):
                            pred = predict_only(
                                landmarks_2d=lms,
                                world_landmarks=inp.world_landmarks,
                                handedness=hand,
                                timestamp_ms=timestamp_ms,
                                only_pose=PoseName.THUMB_PINKY_BASE,
                            )
                        else:
                            pred = active_clf.predict(
                                landmarks_2d=lms,
                                world_landmarks=inp.world_landmarks,
                                handedness=hand,
                                timestamp_ms=timestamp_ms,
                            )
                        pb_p = float(pred.probabilities.get(PoseName.THUMB_PINKY_BASE, 0.0))
                        pose_probs = idle_probabilities()
                        if pb_p > 0.0:
                            pose_probs[PoseName.THUMB_PINKY_BASE] = pb_p
                            pose_probs[PoseName.IDLE] = max(0.0, 1.0 - pb_p)
                            pred_pose = PoseName.THUMB_PINKY_BASE if pb_p >= 0.5 else PoseName.IDLE
                        else:
                            pred_pose = PoseName.IDLE
                except Exception as exc:
                    logger.warning(f"[ENGINE] Classifier failed (fail-open to IDLE): {exc}")
                    pose_probs = idle_probabilities()
                    pred_pose = PoseName.IDLE
                    if self._on_classifier_error is not None:
                        try:
                            self._on_classifier_error()
                        except Exception:
                            pass
            ml_ms += (time.perf_counter() - tm0) * 1000.0

            # Record dedicated zoom pose eligibility (never mixes click/drag/scroll/palm-wheel)
            hand_pinch_pts[hand] = (lms[4].x, lms[4].y)
            in_zoom_pose = (pred_pose == self.zoom_pose) and (not pw_active)
            hand_pinching[hand] = in_zoom_pose
            if in_zoom_pose:
                if hs.zoom_pose_enter_ms is None:
                    hs.zoom_pose_enter_ms = timestamp_ms
            else:
                hs.zoom_pose_enter_ms = None

            # 6. Pre-click dwell lock (V031-03 single-arm + cooldown limiter) & Pose State Machine
            tg1 = time.perf_counter()
            click_poses = [
                pose_k
                for pose_k, act in hs.pose_sm.tap_actions.items()
                if act == "left_click"
            ]
            click_prob = max((float(pose_probs.get(pk, 0.0)) for pk in click_poses), default=0.0)
            if (
                self.click_dwell_lock_ms > 0.0
                and self._is_click_allowed_for_hand(hand)
                and not hs.is_grabbing
                and not pw_active
            ):
                # Expire active dwell lock cleanly & enforce cooldown if no click fired
                if hs.dwell_lock_pos is not None and timestamp_ms >= hs.dwell_lock_until_ms:
                    hs.dwell_lock_pos = None
                    hs.dwell_lock_until_ms = 0.0
                    hs.dwell_lock_cooldown_until_ms = timestamp_ms + self.click_dwell_lock_cooldown_ms
                    self.dwell_unlock_without_click += 1

                if click_prob < 0.20:
                    hs.last_low_click_prob_ms = timestamp_ms
                    if hs.pose_sm.active_pose is None:
                        hs.dwell_lock_used_in_attempt = False
                        hs.dwell_lock_until_ms = 0.0
                        hs.dwell_lock_pos = None
                else:
                    rising_from_low = (
                        hs.last_low_click_prob_ms is not None
                        and (timestamp_ms - hs.last_low_click_prob_ms) <= 250.0
                    ) or (hs.last_low_click_prob_ms is None and hs.prev_click_prob < 0.20)
                    if (
                        (not hs.dwell_lock_used_in_attempt)
                        and (hs.dwell_lock_pos is None)
                        and (timestamp_ms >= hs.dwell_lock_cooldown_until_ms)
                        and click_prob > 0.40
                        and click_prob > hs.prev_click_prob
                        and rising_from_low
                    ):
                        hs.dwell_lock_until_ms = timestamp_ms + self.click_dwell_lock_ms
                        hs.dwell_lock_pos = (cursor_out.x, cursor_out.y)
                        hs.dwell_lock_used_in_attempt = True
                        self.dwell_lock_activations += 1

                if timestamp_ms < hs.dwell_lock_until_ms and hs.dwell_lock_pos is not None:
                    cursor_out.x, cursor_out.y = hs.dwell_lock_pos
                    cursor_out.delta_x = 0.0
                    cursor_out.delta_y = 0.0
                    hs.cursor_delta = (0.0, 0.0)
            else:
                if click_prob < 0.20:
                    hs.last_low_click_prob_ms = timestamp_ms
                    if hs.pose_sm.active_pose is None:
                        hs.dwell_lock_used_in_attempt = False
                hs.dwell_lock_until_ms = 0.0
                hs.dwell_lock_pos = None
            hs.prev_click_prob = click_prob

            # Brief pre-drag dwell lock (V031-05): stabilizes 3-finger pinch closure, NEVER locks active drag
            drag_poses = [
                pose_k
                for pose_k, act in hs.pose_sm.hold_actions.items()
                if act == "drag"
            ]
            drag_prob = max((float(pose_probs.get(pk, 0.0)) for pk in drag_poses), default=0.0)
            if (
                self.drag_dwell_lock_ms > 0.0
                and self._is_hand_allowed_for_click_actions(hand)
                and not hs.is_grabbing
                and not pw_active
            ):
                if hs.drag_dwell_lock_pos is not None and timestamp_ms >= hs.drag_dwell_lock_until_ms:
                    hs.drag_dwell_lock_pos = None
                    hs.drag_dwell_lock_until_ms = 0.0
                if drag_prob < 0.20 and hs.pose_sm.active_pose is None:
                    hs.drag_dwell_lock_used = False
                    hs.drag_dwell_lock_pos = None
                    hs.drag_dwell_lock_until_ms = 0.0
                elif (
                    (not hs.drag_dwell_lock_used)
                    and hs.drag_dwell_lock_pos is None
                    and drag_prob > 0.40
                    and drag_prob > hs.prev_drag_prob
                ):
                    hs.drag_dwell_lock_until_ms = timestamp_ms + self.drag_dwell_lock_ms
                    hs.drag_dwell_lock_pos = (cursor_out.x, cursor_out.y)
                    hs.drag_dwell_lock_used = True

                click_locked = timestamp_ms < hs.dwell_lock_until_ms and hs.dwell_lock_pos is not None
                if (
                    (not click_locked)
                    and timestamp_ms < hs.drag_dwell_lock_until_ms
                    and hs.drag_dwell_lock_pos is not None
                ):
                    cursor_out.x, cursor_out.y = hs.drag_dwell_lock_pos
                    cursor_out.delta_x = 0.0
                    cursor_out.delta_y = 0.0
                    hs.cursor_delta = (0.0, 0.0)
            else:
                if drag_prob < 0.20 and hs.pose_sm.active_pose is None:
                    hs.drag_dwell_lock_used = False
                hs.drag_dwell_lock_until_ms = 0.0
                hs.drag_dwell_lock_pos = None
            hs.prev_drag_prob = drag_prob

            if hs.pose_sm.active_pose is not None:
                hs.pose_sm.add_motion_delta(hs.cursor_delta[0], hs.cursor_delta[1])

            events = hs.pose_sm.process(
                pose_probs, timestamp_ms, cursor_speed_px_s=cursor_out.speed
            )

            is_pinching = False
            is_clicking = False
            click_count = 0
            click_x: Optional[float] = None
            click_y: Optional[float] = None
            is_right_clicking = False
            right_click_x: Optional[float] = None
            right_click_y: Optional[float] = None
            grab_started = False
            grab_ended = False
            drag_x: Optional[float] = None
            drag_y: Optional[float] = None
            dictation_started = False
            dictation_ended = False

            for ev in events:
                if ev.type == "pose_enter":
                    if hs.pose_sm.get_tap_action(ev.pose) == "left_click" and self._is_click_allowed_for_hand(hand):
                        is_pinching = True
                        if self.click_lookback_ms > 0.0:
                            hs.click_anchor_pos = hs.kinematics.lookup_position_guarded(
                                timestamp_ms - self.click_lookback_ms,
                                (cursor_out.x, cursor_out.y),
                                max_speed_px_s=self.click_lookback_max_speed_px_s,
                                max_offset_px=self.click_lookback_max_offset_px,
                                fallback_pos=hs.dwell_lock_pos,
                            )
                        else:
                            hs.click_anchor_pos = (cursor_out.x, cursor_out.y)
                    if hs.pose_sm.get_hold_action(ev.pose) == "drag" and self._is_hand_allowed_for_click_actions(hand):
                        if self.drag_lookback_ms > 0.0:
                            hs.drag_anchor_pos = hs.kinematics.lookup_position_guarded(
                                timestamp_ms - self.drag_lookback_ms,
                                (cursor_out.x, cursor_out.y),
                                max_speed_px_s=self.click_lookback_max_speed_px_s,
                                max_offset_px=self.click_lookback_max_offset_px,
                                fallback_pos=hs.drag_dwell_lock_pos,
                            )
                        else:
                            hs.drag_anchor_pos = (cursor_out.x, cursor_out.y)
                    if (
                        ev.pose == PoseName.THUMB_RING_PINCH
                        or hs.pose_sm.get_tap_action(ev.pose) == "right_click"
                    ) and self._is_hand_allowed_for_click_actions(hand):
                        if self.drag_lookback_ms > 0.0:
                            hs.right_click_anchor_pos = hs.kinematics.lookup_position_guarded(
                                timestamp_ms - self.drag_lookback_ms,
                                (cursor_out.x, cursor_out.y),
                                max_speed_px_s=self.click_lookback_max_speed_px_s,
                                max_offset_px=self.click_lookback_max_offset_px,
                            )
                        else:
                            hs.right_click_anchor_pos = (cursor_out.x, cursor_out.y)
                    if hs.pose_sm.get_hold_action(ev.pose) == "dictation" and self.enable_dictation:
                        hs.dictation_pose_enter_ms = timestamp_ms
                elif ev.type == "tap":
                    tap_act = hs.pose_sm.get_tap_action(ev.pose)
                    if hs.pose_sm.get_hold_action(ev.pose) == "dictation":
                        hs.dictation_pose_enter_ms = None
                    if hs.pose_sm.get_hold_action(ev.pose) == "drag":
                        hs.drag_anchor_pos = None
                    if tap_act == "left_click" and self._is_click_allowed_for_hand(hand):
                        debounce_ok = (not self.enable_click_debounce) or (
                            (timestamp_ms - hs.last_click_ms) > self.click_debounce_ms
                        )
                        if debounce_ok:
                            if (
                                hs.last_click_count == 1
                                and (timestamp_ms - hs.last_click_ms) <= self.double_click_window_ms
                            ):
                                click_count = 2
                            else:
                                click_count = 1
                            is_clicking = True
                            if hs.click_anchor_pos is not None:
                                click_x, click_y = hs.click_anchor_pos
                            else:
                                click_x, click_y = cursor_out.x, cursor_out.y
                            hs.last_click_ms = timestamp_ms
                            hs.last_click_count = click_count
                        hs.click_anchor_pos = None
                        hs.dwell_lock_until_ms = 0.0
                        hs.dwell_lock_pos = None
                    elif tap_act == "right_click" and self.enable_right_click and self._is_hand_allowed_for_click_actions(hand):
                        is_right_clicking = True
                        if hs.right_click_anchor_pos is not None:
                            right_click_x, right_click_y = hs.right_click_anchor_pos
                        else:
                            right_click_x, right_click_y = cursor_out.x, cursor_out.y
                        hs.right_click_anchor_pos = None
                    elif ev.pose == PoseName.THUMB_RING_PINCH:
                        hs.right_click_anchor_pos = None
                elif ev.type == "hold_start":
                    hold_act = hs.pose_sm.get_hold_action(ev.pose)
                    if hold_act == "drag" and self.enable_grab and self._is_hand_allowed_for_click_actions(hand):
                        hs.is_grabbing = True
                        grab_started = True
                        if hs.drag_anchor_pos is not None:
                            drag_x, drag_y = hs.drag_anchor_pos
                        else:
                            drag_x, drag_y = cursor_out.x, cursor_out.y
                        hs.drag_anchor_pos = None
                        # Release all dwell locks immediately once drag starts so dragging moves freely
                        hs.drag_dwell_lock_until_ms = 0.0
                        hs.drag_dwell_lock_pos = None
                        hs.dwell_lock_until_ms = 0.0
                        hs.dwell_lock_pos = None
                    elif hold_act == "dictation" and self.enable_dictation:
                        # Arming is deferred to post-classification arbitration (`dictation_arm_ms`)
                        if hs.dictation_pose_enter_ms is None:
                            hs.dictation_pose_enter_ms = timestamp_ms
                elif ev.type == "hold_end":
                    hold_act = hs.pose_sm.get_hold_action(ev.pose)
                    if hold_act == "drag" and hs.is_grabbing:
                        hs.is_grabbing = False
                        grab_ended = True
                        hs.drag_anchor_pos = None
                    elif hold_act == "dictation":
                        hs.dictation_pose_enter_ms = None
                        if hs.is_dictating:
                            hs.is_dictating = False
                            hs.dictation_started_ms = None
                            dictation_ended = True
                    # Dual behavior: thumb_ring_pinch held still -> right click on release
                    if (
                        ev.pose == PoseName.THUMB_RING_PINCH
                        and self.enable_right_click
                        and self._is_hand_allowed_for_click_actions(hand)
                        and ev.motion_distance < self.still_right_click_motion_px
                    ):
                        is_right_clicking = True
                        if hs.right_click_anchor_pos is not None:
                            right_click_x, right_click_y = hs.right_click_anchor_pos
                        else:
                            right_click_x, right_click_y = cursor_out.x, cursor_out.y
                    if ev.pose == PoseName.THUMB_RING_PINCH:
                        hs.right_click_anchor_pos = None

            # 7. Downstream Hold Controllers
            active_pose = hs.pose_sm.active_pose
            active_hold = (
                hs.pose_sm.get_hold_action(active_pose)
                if (active_pose is not None and hs.pose_sm.is_hold_active)
                else None
            )

            scroll_active = (
                self.enable_scroll
                and active_hold == "scroll"
                and (
                    active_pose != PoseName.THUMB_RING_PINCH
                    or hs.pose_sm.motion_distance >= self.still_right_click_motion_px
                )
            )
            scroll_out = hs.scroll.process(
                hs.cursor_delta[0], hs.cursor_delta[1], scroll_active, timestamp_ms
            )

            wt_active = self.enable_window_tile and (active_hold == "window_tile")
            wt_out = hs.window_tile.process(hs.cursor_delta[0], hs.cursor_delta[1], wt_active, timestamp_ms)

            sn_active = self.enable_spaces_nav and (active_hold == "spaces_nav")
            sn_out = hs.spaces_nav.process(hs.cursor_delta[0], hs.cursor_delta[1], sn_active, timestamp_ms)
            gest_ms += (time.perf_counter() - tg1) * 1000.0

            pose_out = PoseOutput(
                hand=hand,
                pose=pred_pose,
                active_pose=active_pose,
                pose_probabilities=pose_probs,
                is_pinching=is_pinching,
                is_clicking=is_clicking,
                click_count=click_count,
                click_x=click_x,
                click_y=click_y,
                is_right_clicking=is_right_clicking,
                right_click_x=right_click_x,
                right_click_y=right_click_y,
                is_grabbing=hs.is_grabbing,
                grab_just_started=grab_started,
                grab_just_ended=grab_ended,
                drag_x=drag_x,
                drag_y=drag_y,
                is_dictating=hs.is_dictating,
                dictation_just_started=dictation_started,
                dictation_just_ended=dictation_ended,
                is_scrolling=scroll_out.is_scrolling or scroll_out.is_inertia,
                is_scroll_inertia=scroll_out.is_inertia,
                scroll_dx=scroll_out.dx,
                scroll_dy=scroll_out.dy,
                is_window_tiling=wt_out.is_active,
                window_tile_direction=wt_out.direction if wt_out.triggered else None,
                window_tile_preview_direction=wt_out.direction if wt_out.is_active else None,
                is_spaces_nav=sn_out.is_active,
                spaces_nav_direction=sn_out.direction if sn_out.triggered else None,
                spaces_nav_preview_direction=sn_out.direction if sn_out.is_active else None,
                palm_wheel_state=pw_out.state,
                palm_wheel_action=pw_out.triggered_action,
                palm_wheel_progress=pw_out.progress,
                palm_wheel_selected_finger=pw_out.selected_finger,
                palm_wheel_press_event=pw_out.press_event,
                palm_wheel_menu_path=pw_out.menu_path,
                palm_wheel_menu_label=pw_out.menu_label,
                palm_wheel_slots=pw_out.slots,
                palm_wheel_finger_positions=pw_out.finger_positions,
                click_timestamp_ms=int(hs.last_click_ms) if hs.last_click_ms > 0 else 0,
            )

            outputs.append(HandOutput(hand=hand, cursor=cursor_out, pose=pose_out))

        # 8. Two-Hand Zoom vs Single-Hand Dictation Arbitration (V031-01 + V041-04 priority)
        any_busy_for_zoom = any(
            o.pose.is_grabbing
            or o.pose.is_scrolling
            or o.pose.palm_wheel_state in ("active", "confirmed")
            for o in outputs
        )
        raw_both_in_zoom_pose = (
            self.enable_zoom
            and (not any_busy_for_zoom)
            and len(outputs) == 2
            and hand_pinching.get("Left", False)
            and hand_pinching.get("Right", False)
        )
        # V041-04: Once dictation has been active >= dictation_priority_ms, ignore incidental 2nd-hand zoom
        dictation_has_priority = any(
            self._hands[o.hand].is_dictating
            and self._hands[o.hand].dictation_started_ms is not None
            and (timestamp_ms - self._hands[o.hand].dictation_started_ms) >= self.dictation_priority_ms
            for o in outputs
        )
        dictation_priority_suppressed_zoom = dictation_has_priority and raw_both_in_zoom_pose
        both_in_zoom_pose = raw_both_in_zoom_pose and (not dictation_has_priority)

        zoom_intent = both_in_zoom_pose
        # Also detect second hand entering zoom_pose within zoom_arbitration_ms of an active/arming dictation
        if (
            (not zoom_intent)
            and (not dictation_has_priority)
            and self.enable_zoom
            and len(outputs) == 2
            and (not any_busy_for_zoom)
        ):
            l_enter = self._hands["Left"].zoom_pose_enter_ms
            r_enter = self._hands["Right"].zoom_pose_enter_ms
            if l_enter is not None and r_enter is not None and abs(l_enter - r_enter) <= self.zoom_arbitration_ms:
                zoom_intent = True

        # Count hands currently in whichever pose is mapped to dictation (preset-agnostic)
        dict_pose_hands = [
            o.hand
            for o in outputs
            if (
                (
                    o.pose.active_pose is not None
                    and self._hands[o.hand].pose_sm.get_hold_action(o.pose.active_pose) == "dictation"
                )
                or (
                    o.pose.pose != PoseName.IDLE
                    and self._hands[o.hand].pose_sm.get_hold_action(o.pose.pose) == "dictation"
                )
            )
        ]

        for o in outputs:
            hs = self._hands[o.hand]
            if zoom_intent:
                # Suppress arming and cancel any already-started dictation without transcribing!
                hs.dictation_pose_enter_ms = None
                o.pose.dictation_just_started = False
                o.pose.dictation_just_ended = False
                if hs.is_dictating:
                    hs.is_dictating = False
                    hs.dictation_started_ms = None
                    o.pose.is_dictating = False
                    o.pose.dictation_cancelled = True
                continue

            # Single-hand dictation arming: requires exactly 1 hand in dictation pose AND held >= dictation_arm_ms
            if (
                self.enable_dictation
                and (not hs.is_dictating)
                and hs.dictation_pose_enter_ms is not None
                and o.pose.active_pose is not None
                and hs.pose_sm.get_hold_action(o.pose.active_pose) == "dictation"
                and len(dict_pose_hands) == 1
                and (timestamp_ms - hs.dictation_pose_enter_ms) >= self.dictation_arm_ms
            ):
                hs.is_dictating = True
                hs.dictation_started_ms = timestamp_ms
                o.pose.is_dictating = True
                o.pose.dictation_just_started = True
            elif o.pose.active_pose is None or hs.pose_sm.get_hold_action(o.pose.active_pose) != "dictation":
                hs.dictation_pose_enter_ms = None

        self.last_zoom = self.zoom.process(
            hand_pinch_pts.get("Left"),
            hand_pinch_pts.get("Right"),
            both_in_zoom_pose,
            timestamp_ms,
        )

        # Reset vanished hands cleanly (including per-hand classifier temporal ring buffers)
        for lost in list(self._active_hands - seen):
            self._hands[lost].reset()
            self.smoother.reset(lost)
            self._reset_classifier(lost)
        self._active_hands = seen

        if not inputs and frame_brightness is not None and frame_brightness < 38.0:
            warning = TrackingWarning(level="warning", code="low_light")
        self.last_warning = warning

        # Compute concise "Why didn't it trigger?" diagnostic code (V040-04 + V041-04)
        refusal_code: Optional[str] = None
        if dictation_priority_suppressed_zoom:
            refusal_code = "dictation_priority"
        elif zoom_intent or self.last_zoom.is_zooming:
            refusal_code = "two_hand_zoom_active"
        elif any(
            (not self._hands[o.hand].is_dictating)
            and self._hands[o.hand].dictation_pose_enter_ms is not None
            and o.pose.active_pose is not None
            and self._hands[o.hand].pose_sm.get_hold_action(o.pose.active_pose) == "dictation"
            for o in outputs
        ):
            refusal_code = "dictation_arming"
        elif outputs and all((not o.cursor.is_pointing) and (o.pose.active_pose is None) for o in outputs):
            if self.enable_pointing_gate:
                refusal_code = "pointing_gate_closed"
        elif any(
            o.pose.pose != PoseName.IDLE
            and (
                self._hands[o.hand].pose_sm.get_tap_action(o.pose.pose) == "left_click"
                or self._hands[o.hand].pose_sm.get_hold_action(o.pose.pose) == "drag"
            )
            and (not self._is_hand_allowed_for_click_actions(o.hand))
            for o in outputs
        ):
            refusal_code = "non_click_hand"
        elif warning is not None and warning.code == "hand_too_far":
            refusal_code = "hand_too_far"
        self.last_refusal_code = refusal_code

        total_ms = (time.perf_counter() - t0) * 1000.0
        self.last_timing = PipelineTiming(
            pre_filters_ms=pre_ms,
            cursor_ms=cur_ms,
            gesture_ms=gest_ms,
            gesture_ml_ms=ml_ms,
            total_ms=total_ms,
            frame_budget_ms=33.33,
            headroom_ms=max(0.0, 33.33 - total_ms),
            utilization_pct=min(100.0, (total_ms / 33.33) * 100.0),
        )
        return outputs
