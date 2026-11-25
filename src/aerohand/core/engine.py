"""
HandCursorEngine — Central Airpoint-Style Orchestrator for AeroHand.
Ported & adapted from Airpoint SDK `HandCursorEngine.ts` (Apache-2.0) + GestureCtl safety:

Pipeline per frame:
  1. Stage-1 LandmarkSmoother (fast OneEuro, mc=2.0, beta=45.0, gap-reset).
  2. PalmWheelDetector (open palm facing camera -> radial quick menu; smoothly anchors
     cursor source to palm center during active wheel).
  3. CursorKinematics (index tip LM8, Stage-2 OneEuro, distance compensation, accel,
     jump rejection, OS cursor anchor).
  4. 3D Forward-Pointing Gate with hysteresis (`(LM8 - LM0).z / len < -0.15`).
  5. Pluggable PinchClassifier (`heuristic_3d` MVP or self-trained `onnx_tcn`) with
     fail-open exception safety (returns idle, never crashes or sticks drag).
  6. PoseStateMachine -> Click (single/double), Right-Click, Drag, Scroll (2D + inertia),
     Spaces Navigation, and Window Tiling.
"""

from __future__ import annotations

import logging
import math
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
    PoseName,
    PoseOutput,
    idle_probabilities,
)
from aerohand.core.window_tile import WindowTileController
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
        self.is_pointing_latched = False
        self.palm_wheel_was_active = False
        self.palm_wheel_offset = (0.0, 0.0)
        self.cursor_delta = (0.0, 0.0)


class HandCursorEngine:
    """Unified multi-hand cursor and gesture engine."""

    def __init__(
        self,
        config: Dict[str, Any],
        classifier: Optional[PinchClassifier] = None,
    ) -> None:
        self.config = config
        self.classifier = classifier

        sm_cfg = config.get("smoothing") or {}
        self.enable_landmark_smoothing: bool = bool(sm_cfg.get("enable_landmark_smoothing", True))
        self.landmark_exclude_joints: List[int] = list(sm_cfg.get("landmark_exclude_joints") or [])
        self.smoother = LandmarkSmoother(
            min_cutoff=float(sm_cfg.get("landmark_min_cutoff", 2.0)),
            beta=float(sm_cfg.get("landmark_beta", 45.0)),
            d_cutoff=float(sm_cfg.get("landmark_d_cutoff", 1.0)),
            gap_reset_ms=float(sm_cfg.get("gap_reset_ms", 100.0)),
        )

        hand_cfg = config.get("hand") or {}
        self.cursor_hand: str = str(hand_cfg.get("cursor_hand", "Right"))
        self.click_hand: str = str(hand_cfg.get("click_hand", "Right"))

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

        self._hands: Dict[Handedness, _PerHandState] = {
            "Left": self._make_hand_state("Left"),
            "Right": self._make_hand_state("Right"),
        }
        self._active_hands: set[Handedness] = set()

    def _make_hand_state(self, hand: Handedness) -> _PerHandState:
        return _PerHandState(
            hand=hand,
            kinematics=CursorKinematics(self.config.get("cursor")),
            pose_sm=PoseStateMachine(self.config.get("pose_state_machine")),
            scroll=ScrollController(self.config.get("scroll")),
            spaces_nav=SpacesNavController(self.config.get("spaces_nav")),
            window_tile=WindowTileController(self.config.get("window_tile")),
            palm_wheel=PalmWheelDetector(self.config.get("palm_wheel")),
        )

    def reset(self) -> None:
        self.smoother.reset()
        for hs in self._hands.values():
            hs.reset()
        self._active_hands.clear()
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

    def _is_click_allowed_for_hand(self, hand: Handedness) -> bool:
        return self.click_hand == "Both" or self.click_hand == hand

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

    def process_frame(
        self,
        inputs: List[HandInput],
        timestamp_ms: float,
        width: float,
        height: float,
    ) -> List[HandOutput]:
        """Process one camera frame of 0..2 detected hands."""
        seen: set[Handedness] = set()
        outputs: List[HandOutput] = []

        for inp in inputs:
            hand = inp.handedness
            seen.add(hand)
            hs = self._hands[hand]
            hs.last_seen_ms = timestamp_ms

            raw_lms = inp.landmarks
            if len(raw_lms) < 21:
                continue

            # 1. Stage-1 Landmark Smoothing (keeps unfiltered copy for ML if needed)
            if self.enable_landmark_smoothing:
                lms = self.smoother.smooth(raw_lms, hand, timestamp_ms, self.landmark_exclude_joints)
            else:
                lms = list(raw_lms)

            # 2. Palm Wheel detection FIRST so cursor can smoothly switch to palm center
            pw_out = hs.palm_wheel.process(lms, timestamp_ms, handedness=hand)
            pw_active = pw_out.state in ("active", "confirmed")

            # 3. Cursor target resolution (Index tip LM8, or palm center during Palm Wheel)
            index_tip = lms[8]
            wrist = lms[0]
            middle_mcp = lms[9]
            # Note: `camera.py` already mirrors frame horizontally when `camera.mirror=true`,
            # so `index_tip.x` already moves rightward when user moves hand right.
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

            # 4. 3D Forward-Pointing Gate + Hand Rotation
            dir_z = compute_pointing_dir_z(lms)
            is_pointing = self._update_pointing_gate(hs, dir_z)
            rotation = compute_hand_rotation(lms)

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
            )

            # 5. Pluggable Pinch Classification (fail-open to IDLE on any exception)
            pose_probs = idle_probabilities()
            pred_pose = PoseName.IDLE
            gate_allows = (not self.require_pointing_gate_for_actions) or is_pointing
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

            # 6. Pose State Machine (Tap / Hold / Drag / Dual Right-Click-Scroll)
            if hs.pose_sm.active_pose is not None:
                hs.pose_sm.add_motion(math.hypot(hs.cursor_delta[0], hs.cursor_delta[1]))

            events = hs.pose_sm.process(pose_probs, timestamp_ms)

            is_pinching = False
            is_clicking = False
            click_count = 0
            is_right_clicking = False
            grab_started = False
            grab_ended = False

            for ev in events:
                if ev.type == "pose_enter":
                    if (
                        hs.pose_sm.get_tap_action(ev.pose) == "left_click"
                        and ev.pose == PoseName.THUMB_MIDDLE_PINCH
                    ):
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
                    elif tap_act == "right_click":
                        is_right_clicking = True
                elif ev.type == "hold_start":
                    hold_act = hs.pose_sm.get_hold_action(ev.pose)
                    if hold_act == "drag":
                        hs.is_grabbing = True
                        grab_started = True
                elif ev.type == "hold_end":
                    hold_act = hs.pose_sm.get_hold_action(ev.pose)
                    if hold_act == "drag" and hs.is_grabbing:
                        hs.is_grabbing = False
                        grab_ended = True
                    # Dual behavior: thumb_ring_pinch held still -> right click on release
                    if (
                        ev.pose == PoseName.THUMB_RING_PINCH
                        and ev.motion_distance < self.still_right_click_motion_px
                    ):
                        is_right_clicking = True

            # 7. Downstream Hold Controllers (Scroll, Window Tile, Spaces Nav)
            active_pose = hs.pose_sm.active_pose
            active_hold = (
                hs.pose_sm.get_hold_action(active_pose)
                if (active_pose is not None and hs.pose_sm.is_hold_active)
                else None
            )

            scroll_active = active_hold == "scroll" and (
                active_pose != PoseName.THUMB_RING_PINCH
                or hs.pose_sm.motion_distance >= self.still_right_click_motion_px
            )
            scroll_out = hs.scroll.process(
                hs.cursor_delta[0], hs.cursor_delta[1], scroll_active, timestamp_ms
            )

            wt_out = hs.window_tile.process(
                hs.cursor_delta[0], hs.cursor_delta[1], active_hold == "window_tile", timestamp_ms
            )
            sn_out = hs.spaces_nav.process(
                hs.cursor_delta[0], hs.cursor_delta[1], active_hold == "spaces_nav", timestamp_ms
            )

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
                is_scrolling=scroll_out.is_scrolling or scroll_out.is_inertia,
                is_scroll_inertia=scroll_out.is_inertia,
                scroll_dx=scroll_out.dx,
                scroll_dy=scroll_out.dy,
                is_window_tiling=wt_out.is_active,
                window_tile_direction=wt_out.direction if wt_out.triggered else None,
                is_spaces_nav=sn_out.is_active,
                spaces_nav_direction=sn_out.direction if sn_out.triggered else None,
                palm_wheel_state=pw_out.state,
                palm_wheel_action=pw_out.triggered_action,
                palm_wheel_progress=pw_out.progress,
                palm_wheel_selected_finger=pw_out.selected_finger,
                click_timestamp_ms=int(hs.last_click_ms) if hs.last_click_ms > 0 else 0,
            )

            outputs.append(HandOutput(hand=hand, cursor=cursor_out, pose=pose_out))

        # Reset vanished hands cleanly so no stale filter or stuck drag persists
        for lost in list(self._active_hands - seen):
            self._hands[lost].reset()
            self.smoother.reset(lost)
        self._active_hands = seen
        return outputs
