"""
Orthonormal 3D Palm Coordinate Frame & Hand Rotation Estimation.

Combines:
- GestureCtl's Gram-Schmidt 3D orthonormal palm basis (`palm_frame.py`): strictly
  invariant to 3D translation, 3D rigid rotation (yaw/pitch/roll), and camera distance.
- Airpoint SDK's `getHandRotation` (`gestures.ts`, Apache-2.0) for diagnostic pitch/yaw/roll.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from aerohand.core.types import HandRotation, Landmark

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

PALM_KNUCKLE_INDICES = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


@dataclass(frozen=True)
class PalmFrameResult:
    """21 landmarks transformed into a scale-normalized orthonormal palm basis."""

    origin: np.ndarray            # (3,)
    x_axis: np.ndarray            # (3,) transverse axis (index_mcp -> pinky_mcp, mirrored for Left)
    y_axis: np.ndarray            # (3,) longitudinal axis (wrist -> middle_mcp)
    z_axis: np.ndarray            # (3,) palm normal (x_axis x y_axis)
    scale: float                  # palm_width (||LM5 - LM17||) or palm_length (||LM0 - LM9||)
    coords_normalized: np.ndarray # (21, 3) float64
    is_left_hand: bool

    def dist(self, i: int, j: int) -> float:
        """Euclidean 3D distance between landmarks i and j in palm-scale units."""
        diff = self.coords_normalized[i] - self.coords_normalized[j]
        return float(np.linalg.norm(diff))


def landmarks_to_array(landmarks: Any) -> Optional[np.ndarray]:
    """Convert a 21-landmark sequence or ndarray into shape (21, 3) float64."""
    if landmarks is None:
        return None
    if isinstance(landmarks, np.ndarray):
        if landmarks.ndim == 2 and landmarks.shape[0] >= 21 and landmarks.shape[1] >= 3:
            return landmarks[:21, :3].astype(np.float64, copy=False)
        return None
    if not isinstance(landmarks, (list, tuple)) or len(landmarks) < 21:
        return None
    arr = np.empty((21, 3), dtype=np.float64)
    try:
        for i in range(21):
            pt = landmarks[i]
            if isinstance(pt, Landmark):
                arr[i, 0] = pt.x
                arr[i, 1] = pt.y
                arr[i, 2] = pt.z
            elif isinstance(pt, dict):
                arr[i, 0] = float(pt["x"])
                arr[i, 1] = float(pt["y"])
                arr[i, 2] = float(pt.get("z", 0.0))
            elif isinstance(pt, (tuple, list)):
                arr[i, 0] = float(pt[0])
                arr[i, 1] = float(pt[1])
                arr[i, 2] = float(pt[2]) if len(pt) > 2 else 0.0
            else:
                arr[i, 0] = float(getattr(pt, "x"))
                arr[i, 1] = float(getattr(pt, "y"))
                arr[i, 2] = float(getattr(pt, "z", 0.0))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def compute_palm_frame(
    landmarks: Any,
    *,
    anchor: str = "palm_center",
    is_left_hand: bool = False,
    scale_metric: str = "palm_width",
) -> Optional[PalmFrameResult]:
    """Build an orthonormal, scale-normalized palm coordinate frame."""
    coords = landmarks_to_array(landmarks)
    if coords is None:
        return None

    if anchor == "palm_center":
        origin = np.mean(coords[list(PALM_KNUCKLE_INDICES)], axis=0)
    elif anchor in ("index_mcp", "mcp"):
        origin = coords[INDEX_MCP].copy()
    elif anchor == "wrist":
        origin = coords[WRIST].copy()
    else:
        raise ValueError(f"Unsupported anchor '{anchor}'.")

    v_y = coords[MIDDLE_MCP] - coords[WRIST]
    norm_y = float(np.linalg.norm(v_y))
    if norm_y < 1e-7:
        return None
    y_axis = v_y / norm_y

    v_span = coords[PINKY_MCP] - coords[INDEX_MCP]
    v_x = v_span - float(np.dot(v_span, y_axis)) * y_axis
    norm_x = float(np.linalg.norm(v_x))
    if norm_x < 1e-7:
        return None
    x_axis = v_x / norm_x
    if is_left_hand:
        x_axis = -x_axis

    z_axis = np.cross(x_axis, y_axis)
    norm_z = float(np.linalg.norm(z_axis))
    if norm_z < 1e-7:
        return None
    z_axis = z_axis / norm_z

    if scale_metric == "palm_width":
        scale = float(np.linalg.norm(coords[INDEX_MCP] - coords[PINKY_MCP]))
    elif scale_metric == "palm_length":
        scale = float(np.linalg.norm(coords[MIDDLE_MCP] - coords[WRIST]))
    else:
        raise ValueError(f"Unsupported scale_metric '{scale_metric}'.")
    if scale < 1e-7:
        return None

    rot = np.vstack([x_axis, y_axis, z_axis])
    coords_norm = ((coords - origin) @ rot.T) / scale

    return PalmFrameResult(
        origin=origin,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        scale=scale,
        coords_normalized=coords_norm,
        is_left_hand=is_left_hand,
    )


def compute_hand_rotation(landmarks: Sequence[Landmark]) -> Optional[HandRotation]:
    """Estimate (pitch, yaw, roll) in degrees following Airpoint SDK `getHandRotation`."""
    if len(landmarks) < 21:
        return None
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    pinky_mcp = landmarks[PINKY_MCP]

    up_x = middle_mcp.x - wrist.x
    up_y = middle_mcp.y - wrist.y
    up_z = middle_mcp.z - wrist.z

    ac_x = index_mcp.x - pinky_mcp.x
    ac_y = index_mcp.y - pinky_mcp.y
    ac_z = index_mcp.z - pinky_mcp.z

    nx = ac_y * up_z - ac_z * up_y
    ny = ac_z * up_x - ac_x * up_z
    nz = ac_x * up_y - ac_y * up_x

    up_mag = math.sqrt(up_x * up_x + up_y * up_y + up_z * up_z)
    n_mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if up_mag < 1e-7 or n_mag < 1e-7:
        return None

    up_y /= up_mag
    up_z /= up_mag
    nx /= n_mag
    nz /= n_mag

    pitch = math.degrees(math.atan2(-up_z, -up_y))
    yaw = math.degrees(math.atan2(nx, -nz))
    roll = math.degrees(math.atan2(ac_y, ac_x))
    return HandRotation(pitch=pitch, yaw=yaw, roll=roll)


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex `b` formed by segments `a-b` and `c-b`."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-7 or n2 < 1e-7:
        return 0.0
    cos_val = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_val))


def compute_pointing_dir_z(landmarks: Sequence[Landmark]) -> float:
    """
    Airpoint 3D Forward-Pointing Gate metric:
    Normalized Z component of vector Wrist (LM0) -> Index Tip (LM8).
    Negative Z points toward the camera/screen; typical threshold is < -0.15.
    """
    if len(landmarks) < 9:
        return 0.0
    dx = landmarks[INDEX_TIP].x - landmarks[WRIST].x
    dy = landmarks[INDEX_TIP].y - landmarks[WRIST].y
    dz = landmarks[INDEX_TIP].z - landmarks[WRIST].z
    mag = math.sqrt(dx * dx + dy * dy + dz * dz)
    return (dz / mag) if mag > 1e-7 else 0.0


def compute_palm_facing_dot(
    landmarks: Sequence[Landmark],
    is_left_hand: bool = False,
    mirrored: bool = True,
) -> float:
    """
    Compute unit palm normal facing score toward camera in [-1, 1] (~+1.0 = palm facing camera).
    When `mirrored=True` (camera frame already flipped horizontally), the X axis is reversed,
    so the camera-facing sign flips compared to unmirrored video coordinates.
    """
    if len(landmarks) < 18:
        return 0.0
    wrist = landmarks[WRIST]
    index_mcp = landmarks[INDEX_MCP]
    middle_mcp = landmarks[MIDDLE_MCP]
    pinky_mcp = landmarks[PINKY_MCP]

    up_x = middle_mcp.x - wrist.x
    up_y = middle_mcp.y - wrist.y
    up_z = middle_mcp.z - wrist.z

    ac_x = index_mcp.x - pinky_mcp.x
    ac_y = index_mcp.y - pinky_mcp.y
    ac_z = index_mcp.z - pinky_mcp.z
    if is_left_hand:
        ac_x, ac_y, ac_z = -ac_x, -ac_y, -ac_z

    nx = ac_y * up_z - ac_z * up_y
    ny = ac_z * up_x - ac_x * up_z
    nz = ac_x * up_y - ac_y * up_x
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag < 1e-7:
        return 0.0
    return (nz / mag) if mirrored else (-nz / mag)
