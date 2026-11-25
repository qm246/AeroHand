"""
Airpoint 885-Dim Full Z-Depth Feature Extractor (Apache-2.0 clean-room Python port
of `airmouse885Features.ts`).

Provides an A/B comparable feature representation alongside AeroHand's 348-dim
strictly rotation-invariant `palm_frame` features (`features_3d.py`), including
consistent Left-hand X mirroring (`1 - x` in 2D, `-x` in 3D) so one trained TCN
generalizes seamlessly to both Left and Right hands.

Layout (885 float32 values):
  1. raw3D (63)
  2. raw2DWithZ (63)
  3. wristRelative3D (63)
  4. wristRelative2DWithZ (63)
  5. palmNormalizedWithZ (63)
  6. geometry2D with full z (210 pairwise + 60 bones + 15 angles = 285)
  7. geometry3D (210 pairwise + 60 bones + 15 angles = 285)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from aerohand.core.types import Handedness, Landmark

AIRMOUSE_885_DIM = 885
_EPS = 1e-6

_TRI_I, _TRI_J = np.triu_indices(21, k=1)

_BONE_SRC = np.array(
    [0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], dtype=np.int32
)
_BONE_DST = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int32
)

_ANG_A = np.array([1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 0, 0, 0, 0, 0], dtype=np.int32)
_ANG_B = np.array([2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 5, 9, 13, 17, 1], dtype=np.int32)
_ANG_C = np.array([3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 6, 10, 14, 18, 2], dtype=np.int32)


def mirror_landmarks_arrays(
    handedness: Handedness,
    pts2d: np.ndarray,
    pts3d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror Left hand coordinates into Right-hand canonical space."""
    if handedness != "Left":
        return pts2d, pts3d
    m2 = pts2d.copy()
    m2[:, 0] = 1.0 - m2[:, 0]
    m3 = pts3d.copy()
    m3[:, 0] = -m3[:, 0]
    return m2, m3


def _geometry_285(pts: np.ndarray) -> np.ndarray:
    diff = pts[_TRI_I] - pts[_TRI_J]
    pairwise = np.linalg.norm(diff, axis=1)  # (210,)

    bones = (pts[_BONE_DST] - pts[_BONE_SRC]).reshape(-1)  # (60,)

    v1 = pts[_ANG_A] - pts[_ANG_B]
    v2 = pts[_ANG_C] - pts[_ANG_B]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    cos_a = np.where(denom > _EPS, np.sum(v1 * v2, axis=1) / np.maximum(denom, _EPS), 0.0)
    angles = np.where(denom > _EPS, np.arccos(np.clip(cos_a, -1.0, 1.0)), 0.0)  # (15,)

    return np.concatenate([pairwise, bones, angles]).astype(np.float32)


def extract_airmouse_885_features(
    landmarks_2d: Sequence[Landmark],
    world_landmarks: Optional[Sequence[Landmark]] = None,
    handedness: Handedness = "Right",
) -> np.ndarray:
    """Vectorized 885-dim AirMouse feature extractor with Left-hand mirroring."""
    if len(landmarks_2d) < 21:
        return np.zeros(AIRMOUSE_885_DIM, dtype=np.float32)

    pts2d = np.array([(lm.x, lm.y, lm.z) for lm in landmarks_2d[:21]], dtype=np.float64)
    src3d = world_landmarks if (world_landmarks is not None and len(world_landmarks) >= 21) else landmarks_2d
    pts3d = np.array([(lm.x, lm.y, lm.z) for lm in src3d[:21]], dtype=np.float64)

    pts2d, pts3d = mirror_landmarks_arrays(handedness, pts2d, pts3d)

    wrist3 = pts3d[0]
    wrist2 = pts2d[0]
    palm_size = max(float(np.hypot(wrist2[0] - pts2d[5, 0], wrist2[1] - pts2d[5, 1])), _EPS)

    raw3 = pts3d.reshape(-1)
    raw2 = pts2d.reshape(-1)
    rel3 = (pts3d - wrist3).reshape(-1)
    rel2 = (pts2d - wrist2).reshape(-1)
    palm_norm = ((pts3d - wrist3) / palm_size).reshape(-1)
    geo2 = _geometry_285(pts2d)
    geo3 = _geometry_285(pts3d)

    out = np.concatenate([raw3, raw2, rel3, rel2, palm_norm, geo2, geo3]).astype(np.float32)
    return out
