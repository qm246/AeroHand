"""Core data types shared across AeroHand engine, ML classifiers, and macOS platform."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple


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
SlideDirection = Optional[Literal["left", "right", "top", "bottom"]]


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
    is_right_clicking: bool = False
    is_grabbing: bool = False
    grab_just_started: bool = False
    grab_just_ended: bool = False
    is_scrolling: bool = False
    is_scroll_inertia: bool = False
    scroll_dx: float = 0.0
    scroll_dy: float = 0.0
    is_window_tiling: bool = False
    window_tile_direction: SlideDirection = None
    is_spaces_nav: bool = False
    spaces_nav_direction: SlideDirection = None
    palm_wheel_state: str = "inactive"
    palm_wheel_action: Optional[str] = None
    palm_wheel_progress: float = 0.0
    palm_wheel_selected_finger: Optional[str] = None
    click_timestamp_ms: int = 0


@dataclass
class HandOutput:
    """Unified per-hand cursor + gesture output for a single frame."""

    hand: Handedness
    cursor: CursorOutput
    pose: PoseOutput
