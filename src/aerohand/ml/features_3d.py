"""
Vectorized Rotation-, Translation-, and Scale-Invariant 3D Feature Extractor (`348` dims).

Adapted from Airpoint SDK's `airmouse885Features.ts` (Apache-2.0) and hardened for
single-user training by DROPPING screen-absolute 2D `(x, y)` coordinates (which cause
desk-framing overfitting) and expressing all geometry inside GestureCtl's orthonormal
3D `palm_frame`:
  1. `palm_frame` 3D coords:   63  (21 × 3, mirrored for Left hand, normalized by palm_width)
  2. 3D pairwise distances:   210  (C(21, 2) on normalized 3D coords)
  3. 3D bone unit vectors:     60  (20 bones × 3 inside palm_frame)
  4. 3D joint angles (rad):    15  (same 15 joint triplets as Airpoint SDK)
  Total = 348 float32 features (~0.05 ms/frame vectorized in NumPy).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from aerohand.core.palm_frame import compute_palm_frame
from aerohand.core.types import Handedness, Landmark

BONE_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

ANGLE_TRIPLETS: Tuple[Tuple[int, int, int], ...] = (
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
    (0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18), (0, 1, 2),
)

_PAIR_I, _PAIR_J = np.triu_indices(21, k=1)
_BONE_A = np.array([a for a, _ in BONE_PAIRS], dtype=np.int32)
_BONE_B = np.array([b for _, b in BONE_PAIRS], dtype=np.int32)
_ANG_A = np.array([a for a, _, _ in ANGLE_TRIPLETS], dtype=np.int32)
_ANG_B = np.array([b for _, b, _ in ANGLE_TRIPLETS], dtype=np.int32)
_ANG_C = np.array([c for _, _, c in ANGLE_TRIPLETS], dtype=np.int32)

FEATURE_DIM_3D = 63 + 210 + 60 + 15  # 348


def extract_invariant_3d_features(
    landmarks_2d: Sequence[Landmark],
    world_landmarks: Optional[Sequence[Landmark]],
    handedness: Handedness = "Right",
) -> Optional[np.ndarray]:
    """Return a `(348,)` float32 feature vector invariant to 3D pose & camera framing."""
    src = world_landmarks if (world_landmarks is not None and len(world_landmarks) >= 21) else landmarks_2d
    pf = compute_palm_frame(
        src,
        anchor="palm_center",
        is_left_hand=(handedness == "Left"),
        scale_metric="palm_width",
    )
    if pf is None:
        return None

    pts = pf.coords_normalized  # (21, 3) float64

    # 1. Normalized palm-frame coordinates (63)
    f_coords = pts.reshape(-1)

    # 2. Pairwise 3D distances (210)
    diffs = pts[_PAIR_I] - pts[_PAIR_J]
    f_pair = np.linalg.norm(diffs, axis=1)

    # 3. Bone vectors in palm_frame (60)
    bones = pts[_BONE_B] - pts[_BONE_A]
    f_bones = bones.reshape(-1)

    # 4. Joint angles in radians (15)
    v1 = pts[_ANG_A] - pts[_ANG_B]
    v2 = pts[_ANG_C] - pts[_ANG_B]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    dots = np.einsum("ij,ij->i", v1, v2)
    cos_ang = np.where(denom > 1e-6, np.clip(dots / np.maximum(denom, 1e-6), -1.0, 1.0), 1.0)
    f_angles = np.arccos(cos_ang)

    return np.concatenate([f_coords, f_pair, f_bones, f_angles]).astype(np.float32, copy=False)
