"""Core data types shared across AeroHand engine, ML classifiers, UI overlays, and macOS platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple


class PoseName(str, Enum):
    """Canonical 7-class pinch vocabulary matching Airpoint SDK (Apache-2.0)."""

    IDLE = "idle"
    THUMB_INDEX_PINCH = "thumb_index_pinch"
    THUMB_MIDDLE_PINCH = "thumb_middle_pinch"
    THUMB_RING_PINCH = "thumb_ring_pinch"
    THUMB_PINKY_PINCH = "thumb_pinky_pinch"
    THUMB_PINKY_BASE = "thumb_pinky_base"
    THUMB_INDEX_MIDDLE_PINCH = "thumb_index_middle_pinch"


ALL_POSE_NAMES: Tuple[PoseName, ...] = (
    PoseName.IDLE,
    PoseName.THUMB_INDEX_PINCH,
    PoseName.THUMB_MIDDLE_PINCH,
    PoseName.THUMB_RING_PINCH,
    PoseName.THUMB_PINKY_PINCH,
    PoseName.THUMB_PINKY_BASE,
    PoseName.THUMB_INDEX_MIDDLE_PINCH,
)

PoseProbabilities = Dict[PoseName, float]
Handedness = Literal["Left", "Right"]
SlideDirection = Optional[Literal["left", "right", "top", "bottom", "fill"]]


def idle_probabilities() -> PoseProbabilities:
    """Return a clean 1.0-idle probability map across all canonical poses."""
    probs: PoseProbabilities = {p: 0.0 for p in ALL_POSE_NAMES}
    probs[PoseName.IDLE] = 1.0
    return probs


@dataclass(frozen=True)
class Landmark:
    """Single 3D landmark point."""

    x: float
    y: float
    z: float
    visibility: float = 1.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class HandRotation:
    """Estimated 3D hand Euler angles (degrees) from palm basis."""

    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class PoseThresholds:
    """Asymmetric Schmitt trigger thresholds for a single pose."""

    enter_threshold: float = 0.75
    exit_threshold: float = 0.30
    enter_frames: int = 2
    exit_frames: int = 2


@dataclass
class PipelineTiming:
    """Per-stage frame latency telemetry matching Airpoint's PipelineTiming."""

    mediapipe_ms: float = 0.0
    pre_filters_ms: float = 0.0
    cursor_ms: float = 0.0
    gesture_ms: float = 0.0
    feature_extraction_ms: float = 0.0
    gesture_ml_ms: float = 0.0
    render_ms: float = 0.0
    total_ms: float = 0.0
    frame_budget_ms: float = 33.33
    headroom_ms: float = 33.33
    utilization_pct: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "mediapipe_ms": round(self.mediapipe_ms, 2),
            "pre_filters_ms": round(self.pre_filters_ms, 2),
            "cursor_ms": round(self.cursor_ms, 2),
            "gesture_ms": round(self.gesture_ms, 2),
            "feature_extraction_ms": round(self.feature_extraction_ms, 2),
            "gesture_ml_ms": round(self.gesture_ml_ms, 2),
            "render_ms": round(self.render_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "frame_budget_ms": round(self.frame_budget_ms, 2),
            "headroom_ms": round(self.headroom_ms, 2),
            "utilization_pct": round(self.utilization_pct, 1),
        }


@dataclass(frozen=True)
class TrackingWarning:
    """Smart environment/posture warning emitted for toast overlay."""

    level: Literal["warning", "error"]
    code: str


@dataclass(frozen=True)
class ZoomOutput:
    """Two-hand pinch-to-zoom frame output."""

    is_zooming: bool = False
    delta: float = 0.0
    step: int = 0
    distance: float = 0.0


@dataclass
class HandInput:
    """Per-frame tracker hand payload consumed by HandCursorEngine."""

    handedness: Handedness
    landmarks: List[Landmark]
    world_landmarks: Optional[List[Landmark]] = None
    landmarks_unfiltered: Optional[List[Landmark]] = None
    confidence: float = 1.0
    timestamp_ms: int = 0


@dataclass
class CursorOutput:
    """Per-hand cursor kinematics output for one frame."""

    hand: Handedness
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    delta_x: float = 0.0
    delta_y: float = 0.0
    speed: float = 0.0
    finger_speed: float = 0.0
    rotation: Optional[HandRotation] = None
    is_pointing: bool = False
    pointing_dir_z: float = 0.0
    is_valid: bool = False
    handedness_score: float = 1.0
    palm_size: float = 0.0


@dataclass
class PoseOutput:
    """Per-hand gesture & action output for one frame."""

    hand: Handedness
    pose: PoseName = PoseName.IDLE
    active_pose: Optional[PoseName] = None
    pose_probabilities: PoseProbabilities = field(default_factory=idle_probabilities)
    is_pinching: bool = False
    is_clicking: bool = False
    click_count: int = 0
    click_x: Optional[float] = None
    click_y: Optional[float] = None
    is_right_clicking: bool = False
    is_grabbing: bool = False
    grab_just_started: bool = False
    grab_just_ended: bool = False
    is_dictating: bool = False
    dictation_just_started: bool = False
    dictation_just_ended: bool = False
    is_scrolling: bool = False
    is_scroll_inertia: bool = False
    scroll_dx: float = 0.0
    scroll_dy: float = 0.0
    is_window_tiling: bool = False
    window_tile_direction: SlideDirection = None
    window_tile_preview_direction: SlideDirection = None
    is_spaces_nav: bool = False
    spaces_nav_direction: SlideDirection = None
    spaces_nav_preview_direction: SlideDirection = None
    palm_wheel_state: str = "inactive"
    palm_wheel_action: Optional[str] = None
    palm_wheel_progress: float = 0.0
    palm_wheel_selected_finger: Optional[str] = None
    palm_wheel_press_event: Optional[str] = None
    palm_wheel_menu_path: List[str] = field(default_factory=list)
    palm_wheel_menu_label: Optional[str] = None
    palm_wheel_slots: List[Dict[str, Any]] = field(default_factory=list)
    palm_wheel_finger_positions: Optional[Dict[str, Tuple[float, float]]] = None
    click_timestamp_ms: int = 0


@dataclass
class HandOutput:
    """Unified per-hand cursor + gesture output for a single frame."""

    hand: Handedness
    cursor: CursorOutput
    pose: PoseOutput
