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
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from aerohand.core.filters import LandmarkSmoother
from aerohand.core.kinematics import CursorKinematics
from aerohand.core.palm_frame import compute_hand_rotation, compute_pointing_dir_z
from aerohand.core.palm_wheel import PalmWheelDetector
from aerohand.core.pose_state_machine import PoseStateMachine
from aerohand.core.scroll_controller import ScrollController
from aerohand.core.spaces_nav import SpacesNavController
from aerohand.core.types import (
    CursorOutput,
    Handedness,
    HandInput,
    HandOutput,
    PipelineTiming,
    PoseName,
    PoseOutput,
    TrackingWarning,
    ZoomOutput,
    idle_probabilities,
)
from aerohand.core.window_tile import WindowTileController
from aerohand.core.zoom_controller import ZoomController
from aerohand.ml.base import PinchClassifier

logger = logging.getLogger(__name__)


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
    is_pointing_latched: bool = False
    last_click_ms: float = -1e9
    last_click_count: int = 0
    palm_wheel_was_active: bool = False
    palm_wheel_offset: Tuple[float, float] = (0.0, 0.0)
    cursor_delta: Tuple[float, float] = (0.0, 0.0)
    last_seen_ms: float = -1e9

    def reset(self) -> None:
        self.kinematics.reset()
        self.pose_sm.reset()
        self.scroll.reset()
        self.spaces_nav.reset()
        self.window_tile.reset()
        self.palm_wheel.reset()
        self.is_grabbing = False
        self.is_dictating = False
        self.is_pointing_latched = False
        self.palm_wheel_was_active = False
        self.palm_wheel_offset = (0.0, 0.0)
        self.cursor_delta = (0.0, 0.0)


class HandCursorEngine:
    """Unified 2-hand cursor, gesture, Palm Wheel, Dictation, and Zoom engine."""

    def __init__(
        self,
        config: Dict[str, Any],
        classifier: Optional[PinchClassifier] = None,
    ) -> None:
        self.config = config
        self.classifier = classifier
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
        self.update_config(config)

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
        self.zoom.update_config(gest_cfg)

        cur_cfg = config.get("cursor") or {}
        self.enable_pointing_gate: bool = bool(cur_cfg.get("enable_pointing_gate", True))
        self.pointing_forward_z_threshold: float = float(cur_cfg.get("pointing_forward_z_threshold", -0.15))
        self.pointing_gate_hysteresis: float = float(cur_cfg.get("pointing_gate_hysteresis", 0.04))

        psm_cfg = config.get("pose_state_machine") or {}
        self.enable_click_debounce: bool = bool(psm_cfg.get("enable_click_debounce", True))
        self.click_debounce_ms: float = float(psm_cfg.get("click_debounce_ms", 200.0))
        self.double_click_window_ms: float = float(psm_cfg.get("double_click_window_ms", 350.0))
        self.still_right_click_motion_px: float = float(psm_cfg.get("pose_hold_motion_threshold_px", 15.0))

        cls_cfg = (config.get("classifier") or {}).get("heuristic_3d") or {}
        self.require_pointing_gate_for_actions: bool = bool(
            cls_cfg.get("require_pointing_gate_for_actions", True)
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

    def reset(self) -> None:
        self.smoother.reset()
        self.zoom.reset()
        for hs in self._hands.values():
            hs.reset()
        self._active_hands.clear()
        self.last_zoom = ZoomOutput()
        self.last_warning = None
        if self.classifier is not None:
            self.classifier.reset()

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

    def _is_click_allowed_for_hand(self, hand: Handedness) -> bool:
        return self.enable_click and (self.click_hand == "Both" or self.click_hand == hand)

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
            tg0 = time.perf_counter()
            pw_out = hs.palm_wheel.process(lms, timestamp_ms, handedness=hand)
            pw_active = pw_out.state in ("active", "confirmed")
            gest_ms += (time.perf_counter() - tg0) * 1000.0

            # 3. Cursor Kinematics
            tc0 = time.perf_counter()
            index_tip = lms[8]
            wrist = lms[0]
            middle_mcp = lms[9]
            palm_size = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)

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

            kin = hs.kinematics.process(src_x, src_y, lms, timestamp_ms, width, height)
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

            # 5. Pluggable Pinch Classification (fail-open to IDLE)
            pose_probs = idle_probabilities()
            pred_pose = PoseName.IDLE
            gate_allows = (not self.require_pointing_gate_for_actions) or is_pointing
            tm0 = time.perf_counter()
            if self.classifier is not None and not pw_active and gate_allows:
                try:
                    pred = self.classifier.predict(
                        landmarks_2d=lms,
                        world_landmarks=inp.world_landmarks,
                        handedness=hand,
                        timestamp_ms=timestamp_ms,
                    )
                    pred_pose = pred.pose
                    pose_probs = pred.probabilities
                except Exception as exc:
                    logger.warning(f"[ENGINE] Classifier failed (fail-open to IDLE): {exc}")
                    pose_probs = idle_probabilities()
                    pred_pose = PoseName.IDLE
            ml_ms += (time.perf_counter() - tm0) * 1000.0

            # Record pinch coordinates for two-hand zoom check
            hand_pinch_pts[hand] = (lms[4].x, lms[4].y)
            hand_pinching[hand] = pred_pose in (
                PoseName.THUMB_INDEX_PINCH,
                PoseName.THUMB_MIDDLE_PINCH,
                PoseName.THUMB_INDEX_MIDDLE_PINCH,
            )

            # 6. Pose State Machine
            tg1 = time.perf_counter()
            if hs.pose_sm.active_pose is not None:
                hs.pose_sm.add_motion(math.hypot(hs.cursor_delta[0], hs.cursor_delta[1]))

            events = hs.pose_sm.process(pose_probs, timestamp_ms)

            is_pinching = False
            is_clicking = False
            click_count = 0
            is_right_clicking = False
            grab_started = False
            grab_ended = False
            dictation_started = False
            dictation_ended = False

            for ev in events:
                if ev.type == "pose_enter":
                    if hs.pose_sm.get_tap_action(ev.pose) == "left_click" and self._is_click_allowed_for_hand(hand):
                        is_pinching = True
                elif ev.type == "tap":
                    tap_act = hs.pose_sm.get_tap_action(ev.pose)
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
                            hs.last_click_ms = timestamp_ms
                            hs.last_click_count = click_count
                    elif tap_act == "right_click" and self.enable_right_click:
                        is_right_clicking = True
                elif ev.type == "hold_start":
                    hold_act = hs.pose_sm.get_hold_action(ev.pose)
                    if hold_act == "drag" and self.enable_grab:
                        hs.is_grabbing = True
                        grab_started = True
                    elif hold_act == "dictation" and self.enable_dictation:
                        hs.is_dictating = True
                        dictation_started = True
                elif ev.type == "hold_end":
                    hold_act = hs.pose_sm.get_hold_action(ev.pose)
                    if hold_act == "drag" and hs.is_grabbing:
                        hs.is_grabbing = False
                        grab_ended = True
                    elif hold_act == "dictation" and hs.is_dictating:
                        hs.is_dictating = False
                        dictation_ended = True
                    # Dual behavior: thumb_ring_pinch held still -> right click on release
                    if (
                        ev.pose == PoseName.THUMB_RING_PINCH
                        and self.enable_right_click
                        and ev.motion_distance < self.still_right_click_motion_px
                    ):
                        is_right_clicking = True

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
                is_right_clicking=is_right_clicking,
                is_grabbing=hs.is_grabbing,
                grab_just_started=grab_started,
                grab_just_ended=grab_ended,
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

        # 8. Two-Hand Pinch-to-Zoom
        both_pinching = (
            self.enable_zoom
            and len(outputs) == 2
            and hand_pinching.get("Left", False)
            and hand_pinching.get("Right", False)
        )
        self.last_zoom = self.zoom.process(
            hand_pinch_pts.get("Left"),
            hand_pinch_pts.get("Right"),
            both_pinching,
            timestamp_ms,
        )

        # Reset vanished hands cleanly
        for lost in list(self._active_hands - seen):
            self._hands[lost].reset()
            self.smoother.reset(lost)
        self._active_hands = seen

        if not inputs and frame_brightness is not None and frame_brightness < 38.0:
            warning = TrackingWarning(level="warning", code="low_light")
        self.last_warning = warning

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
