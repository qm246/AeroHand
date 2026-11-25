"""
Unified Probability-Based Asymmetric Schmitt Pose State Machine.
Ported from Airpoint SDK `PoseStateMachine.ts` & `poseActions.ts` (Apache-2.0).

Handles:
- Per-pose enter/exit thresholds & consecutive frame counts (asymmetric hysteresis).
- Dual promotion from Tap -> Hold via elapsed time (`pose_hold_threshold_ms`) OR accumulated
  cursor motion (`pose_hold_motion_threshold_px`).
- Dual behavior on `thumb_ring_pinch`: stillness (< motion threshold) emits right-click on
  hold_end; movement (>= motion threshold) activates continuous scroll.
- Live config hot-reload (`update_config`) and `enable_click_hold` drag promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from aerohand.core.types import ALL_POSE_NAMES, PoseName, PoseProbabilities, PoseThresholds

PoseEventType = Literal["pose_enter", "tap", "hold_start", "hold_end"]

_FALLBACK_THRESHOLDS = PoseThresholds(
    enter_threshold=0.90,
    exit_threshold=0.50,
    enter_frames=3,
    exit_frames=3,
)


@dataclass(frozen=True)
class PoseEvent:
    type: PoseEventType
    pose: PoseName
    motion_distance: float = 0.0


class PoseStateMachine:
    """Asymmetric hysteresis & tap/hold resolver operating on soft pose probabilities."""

    def __init__(self, psm_cfg: Optional[Dict[str, Any]] = None) -> None:
        self._active_pose: Optional[PoseName] = None
        self._candidate_pose: Optional[PoseName] = None
        self._enter_count: int = 0
        self._exit_count: int = 0
        self._start_time_ms: float = 0.0
        self._hold_active: bool = False
        self._motion_distance: float = 0.0
        self.update_config(psm_cfg)

    def update_config(self, psm_cfg: Optional[Dict[str, Any]]) -> None:
        cfg = psm_cfg or {}
        self.hold_threshold_ms: float = float(cfg.get("pose_hold_threshold_ms", 180.0))
        self.hold_motion_threshold_px: float = float(cfg.get("pose_hold_motion_threshold_px", 15.0))
        self.enable_click_hold: bool = bool(cfg.get("enable_click_hold", False))

        raw_thresh = cfg.get("thresholds") or {}
        self.thresholds: Dict[PoseName, PoseThresholds] = {}
        for pose in ALL_POSE_NAMES:
            if pose == PoseName.IDLE:
                continue
            item = raw_thresh.get(pose.value)
            if isinstance(item, dict):
                self.thresholds[pose] = PoseThresholds(
                    enter_threshold=float(item.get("enter_threshold", _FALLBACK_THRESHOLDS.enter_threshold)),
                    exit_threshold=float(item.get("exit_threshold", _FALLBACK_THRESHOLDS.exit_threshold)),
                    enter_frames=int(item.get("enter_frames", _FALLBACK_THRESHOLDS.enter_frames)),
                    exit_frames=int(item.get("exit_frames", _FALLBACK_THRESHOLDS.exit_frames)),
                )

        self.tap_actions: Dict[PoseName, str] = {}
        raw_taps = cfg.get("tap_actions")
        if raw_taps is None:
            raw_taps = {"thumb_middle_pinch": "left_click"}
        for k, v in raw_taps.items():
            if not v or v == "none":
                continue
            try:
                self.tap_actions[PoseName(k)] = str(v)
            except ValueError:
                pass

        self.hold_actions: Dict[PoseName, str] = {}
        raw_holds = cfg.get("hold_actions")
        if raw_holds is None:
            raw_holds = {
                "thumb_ring_pinch": "scroll",
                "thumb_index_middle_pinch": "drag",
                "thumb_index_pinch": "dictation",
                "thumb_pinky_pinch": "window_tile",
                "thumb_pinky_base": "spaces_nav",
            }
        for k, v in raw_holds.items():
            if not v or v == "none":
                continue
            try:
                self.hold_actions[PoseName(k)] = str(v)
            except ValueError:
                pass

    def reset(self) -> None:
        self._active_pose = None
        self._candidate_pose = None
        self._enter_count = 0
        self._exit_count = 0
        self._start_time_ms = 0.0
        self._hold_active = False
        self._motion_distance = 0.0

    @property
    def active_pose(self) -> Optional[PoseName]:
        return self._active_pose

    @property
    def is_hold_active(self) -> bool:
        return self._hold_active

    @property
    def motion_distance(self) -> float:
        return self._motion_distance

    def add_motion(self, distance_px: float) -> None:
        if self._active_pose is not None and distance_px > 0.0:
            self._motion_distance += float(distance_px)

    def get_tap_action(self, pose: PoseName) -> Optional[str]:
        return self.tap_actions.get(pose)

    def get_hold_action(self, pose: PoseName) -> Optional[str]:
        mapped = self.hold_actions.get(pose)
        if mapped:
            return mapped
        if self.enable_click_hold and self.tap_actions.get(pose) == "left_click":
            return "drag"
        return None

    def resolve_thresholds(self, pose: PoseName) -> PoseThresholds:
        return self.thresholds.get(pose, _FALLBACK_THRESHOLDS)

    @staticmethod
    def _dominant_non_idle_pose(probs: PoseProbabilities) -> Optional[PoseName]:
        best_pose: Optional[PoseName] = None
        best_prob = -1.0
        for pose in ALL_POSE_NAMES:
            if pose == PoseName.IDLE:
                continue
            p = float(probs.get(pose, 0.0))
            if p > best_prob:
                best_prob = p
                best_pose = pose
        return best_pose

    def process(
        self,
        probs: PoseProbabilities,
        timestamp_ms: float,
        *,
        cursor_speed_px_s: float = 0.0,
    ) -> List[PoseEvent]:
        events: List[PoseEvent] = []

        # --- CASE 1: No active pose, looking for entry ---
        if self._active_pose is None:
            candidate = self._dominant_non_idle_pose(probs)
            if candidate is None:
                self._candidate_pose = None
                self._enter_count = 0
                return events

            if self._candidate_pose != candidate:
                self._candidate_pose = candidate
                self._enter_count = 0

            th = self.resolve_thresholds(candidate)
            if float(probs.get(candidate, 0.0)) >= th.enter_threshold:
                self._enter_count += 1
            else:
                self._enter_count = 0

            if self._enter_count >= th.enter_frames:
                self._active_pose = candidate
                self._start_time_ms = timestamp_ms
                self._exit_count = 0
                self._hold_active = False
                self._motion_distance = 0.0
                self._candidate_pose = None
                self._enter_count = 0
                events.append(PoseEvent(type="pose_enter", pose=candidate))
            return events

        # --- CASE 2: Active pose, check exit or hold promotion ---
        active = self._active_pose
        th = self.resolve_thresholds(active)

        if float(probs.get(active, 0.0)) < th.exit_threshold:
            self._exit_count += 1
        else:
            self._exit_count = 0

        hold_action = self.get_hold_action(active)
        hold_eligible = hold_action is not None

        can_promote_by_motion = hold_eligible and self.hold_motion_threshold_px > 0.0
        hold_by_motion = can_promote_by_motion and (self._motion_distance >= self.hold_motion_threshold_px)

        is_thumb_ring = active == PoseName.THUMB_RING_PINCH
        scroll_requires_motion = (hold_action == "scroll") and (not is_thumb_ring)
        elapsed_ms = timestamp_ms - self._start_time_ms
        hold_by_time = (
            hold_eligible
            and (not scroll_requires_motion)
            and (elapsed_ms >= self.hold_threshold_ms)
        )

        if (not self._hold_active) and self._exit_count == 0 and (hold_by_time or hold_by_motion):
            self._hold_active = True
            events.append(PoseEvent(type="hold_start", pose=active))

        effective_exit_frames = th.exit_frames
        if active == PoseName.THUMB_INDEX_MIDDLE_PINCH and cursor_speed_px_s > 300.0:
            effective_exit_frames = max(th.exit_frames, 6)

        if self._exit_count >= effective_exit_frames:
            was_holding = self._hold_active
            captured_motion = self._motion_distance
            self._active_pose = None
            self._exit_count = 0
            self._hold_active = False
            self._motion_distance = 0.0

            if was_holding and hold_eligible:
                events.append(PoseEvent(type="hold_end", pose=active, motion_distance=captured_motion))
            else:
                events.append(PoseEvent(type="tap", pose=active, motion_distance=captured_motion))

        return events
