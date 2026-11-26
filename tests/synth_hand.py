"""
Synthetic 3D Hand Trajectory Generator for Heuristic Gate & Recall Testing (V051-09).

Generates 21-landmark world + 2D sequences starting from an open right/left hand and
linearly closing the thumb tip toward any of the 6 canonical pinch targets, with optional:
  - Natural curling of non-participating fingers (tight curl or half-curl toward palm center)
  - Constant wrist translation velocity
  - Uniform hand scaling (near vs far camera distance)
  - Static already-closed contact (for approach-gate testing)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from aerohand.core.palm_frame import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_MCP,
    RING_TIP,
    THUMB_TIP,
)
from aerohand.core.types import Landmark, PoseName

# Finger landmark chains: (mcp, pip, dip, tip)
_FINGER_CHAINS: Dict[str, Tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

_POSE_TARGET_FINGERS: Dict[PoseName, Tuple[str, ...]] = {
    PoseName.THUMB_INDEX_PINCH: ("index",),
    PoseName.THUMB_MIDDLE_PINCH: ("middle",),
    PoseName.THUMB_RING_PINCH: ("ring",),
    PoseName.THUMB_PINKY_PINCH: ("pinky",),
    PoseName.THUMB_PINKY_BASE: ("pinky",),
    PoseName.THUMB_INDEX_MIDDLE_PINCH: ("index", "middle"),
}


def make_base_open_hand(*, forward_z: float = -0.25, scale: float = 1.0) -> List[Landmark]:
    """Open right hand centered near (0.5, 0.6) with configurable uniform scale."""
    pts = [
        (0.50, 0.70, 0.00),             # 0: wrist
        (0.45, 0.66, -0.01),            # 1: thumb cmc
        (0.42, 0.62, -0.02),            # 2: thumb mcp
        (0.39, 0.58, -0.03),            # 3: thumb ip
        (0.36, 0.54, -0.04),            # 4: thumb tip
        (0.46, 0.60, -0.02),            # 5: index mcp
        (0.46, 0.54, -0.04),            # 6: index pip
        (0.46, 0.49, -0.06),            # 7: index dip
        (0.46, 0.44, forward_z),        # 8: index tip
        (0.50, 0.59, -0.02),            # 9: middle mcp
        (0.50, 0.52, -0.04),            # 10: middle pip
        (0.50, 0.46, -0.05),            # 11: middle dip
        (0.50, 0.41, -0.06),            # 12: middle tip
        (0.54, 0.60, -0.02),            # 13: ring mcp
        (0.54, 0.54, -0.04),            # 14: ring pip
        (0.54, 0.49, -0.05),            # 15: ring dip
        (0.54, 0.44, -0.06),            # 16: ring tip
        (0.58, 0.62, -0.02),            # 17: pinky mcp
        (0.58, 0.57, -0.04),            # 18: pinky pip
        (0.58, 0.53, -0.05),            # 19: pinky dip
        (0.58, 0.49, -0.06),            # 20: pinky tip
    ]
    cx, cy, cz = pts[0]
    out: List[Landmark] = []
    for x, y, z in pts:
        out.append(
            Landmark(
                x=cx + (x - cx) * scale,
                y=cy + (y - cy) * scale,
                z=cz + (z - cz) * scale,
            )
        )
    return out


def _lerp_lm(a: Landmark, b: Landmark, t: float) -> Landmark:
    return Landmark(
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        z=a.z + (b.z - a.z) * t,
    )


def _palm_center(lms: Sequence[Landmark]) -> Landmark:
    ks = (lms[INDEX_MCP], lms[MIDDLE_MCP], lms[RING_MCP], lms[PINKY_MCP])
    return Landmark(
        x=sum(p.x for p in ks) / 4.0,
        y=sum(p.y for p in ks) / 4.0,
        z=sum(p.z for p in ks) / 4.0,
    )


def apply_finger_curl(
    lms: Sequence[Landmark],
    fingers: Sequence[str],
    *,
    curl_amount: float = 0.85,
) -> List[Landmark]:
    """
    Curl specified fingers anatomically toward their own base on the palm (`curl_amount` in [0, 1]):
      - `curl_amount=0.85` -> tightly curled into palm (`dist_to_palm < 0.72` triggers fist guard)
      - `curl_amount=0.55` -> half-curled (`dist_to_palm < 1.05` triggers half-curled grip guard)
    Each finger curls inward (+z into the palm) toward its own MCP/lower-palm column.
    """
    out = list(lms)
    pc = _palm_center(out)
    wrist = out[0]
    t = max(0.0, min(1.0, float(curl_amount)))
    for fname in fingers:
        chain = _FINGER_CHAINS.get(fname)
        if chain is None:
            continue
        mcp, pip, dip, tip = chain
        mcp_lm = out[mcp]
        target = Landmark(
            x=mcp_lm.x * 0.85 + pc.x * 0.15,
            y=mcp_lm.y * 0.55 + wrist.y * 0.45,
            z=mcp_lm.z + 0.045,
        )
        out[pip] = _lerp_lm(out[pip], mcp_lm, t * 0.35)
        out[dip] = _lerp_lm(out[dip], target, t * 0.65)
        out[tip] = _lerp_lm(out[tip], target, t)
    return out


def make_pinch_frame(
    base_open: Sequence[Landmark],
    target_pose: PoseName,
    closure: float,
    *,
    curl_fingers: Sequence[str] = (),
    curl_amount: float = 0.0,
) -> List[Landmark]:
    """Build a single hand frame at pinch `closure` in [0.0=open .. 1.0=full contact]."""
    lms = list(base_open)
    if curl_fingers and curl_amount > 0.0:
        lms = apply_finger_curl(lms, curl_fingers, curl_amount=curl_amount)

    t = max(0.0, min(1.0, float(closure)))
    thumb0 = lms[THUMB_TIP]

    if target_pose == PoseName.THUMB_INDEX_PINCH:
        tgt = lms[INDEX_TIP]
        lms[THUMB_TIP] = _lerp_lm(thumb0, tgt, t * 0.96)
    elif target_pose == PoseName.THUMB_MIDDLE_PINCH:
        tgt = lms[MIDDLE_TIP]
        lms[THUMB_TIP] = _lerp_lm(thumb0, tgt, t * 0.96)
    elif target_pose == PoseName.THUMB_RING_PINCH:
        tgt = lms[RING_TIP]
        lms[THUMB_TIP] = _lerp_lm(thumb0, tgt, t * 0.96)
    elif target_pose == PoseName.THUMB_PINKY_PINCH:
        # Thumb and pinky tip meet out in front of the palm (-z) near pinky base so curled middle/ring stay clear
        # while pinky tip itself stays within < 0.72 palm widths of palm center (testing fist-guard exclusion!)
        pmcp = lms[PINKY_MCP]
        meet = Landmark(
            x=pmcp.x + 0.008,
            y=pmcp.y - 0.015,
            z=pmcp.z - 0.048,
        )
        lms[PINKY_TIP] = _lerp_lm(lms[PINKY_TIP], meet, t)
        lms[THUMB_TIP] = _lerp_lm(thumb0, meet, t * 0.96)
    elif target_pose == PoseName.THUMB_PINKY_BASE:
        # Thumb reaches across palm front to pinky MCP while pinky tip stays clear of its own base
        pmcp = lms[PINKY_MCP]
        meet = Landmark(x=pmcp.x, y=pmcp.y, z=pmcp.z - 0.015)
        lms[THUMB_TIP] = _lerp_lm(thumb0, meet, t * 0.96)
        # Keep pinky tip anatomically bent outward/upward (half-curled, not collapsed onto MCP)
        if t > 0.0 and "pinky" in curl_fingers:
            lms[PINKY_TIP] = Landmark(x=pmcp.x + 0.015, y=pmcp.y - 0.085, z=pmcp.z - 0.045)
    elif target_pose == PoseName.THUMB_INDEX_MIDDLE_PINCH:
        # Bring index & middle tips together at their midpoint and thumb to meet both
        mid_pt = _lerp_lm(lms[INDEX_TIP], lms[MIDDLE_TIP], 0.5)
        lms[INDEX_TIP] = _lerp_lm(lms[INDEX_TIP], mid_pt, t * 0.85)
        lms[MIDDLE_TIP] = _lerp_lm(lms[MIDDLE_TIP], mid_pt, t * 0.85)
        lms[THUMB_TIP] = _lerp_lm(thumb0, mid_pt, t * 0.96)
    return lms


def make_synth_pinch_sequence(
    target_pose: PoseName,
    *,
    open_frames: int = 5,
    close_frames: int = 6,
    hold_frames: int = 6,
    fps: float = 30.0,
    curl_fingers: Sequence[str] = (),
    curl_amount: float = 0.0,
    wrist_vel_xy: Tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    start_closed: bool = False,
) -> List[Tuple[float, List[Landmark]]]:
    """Generate a chronological `(timestamp_ms, landmarks)` trajectory for synthetic recall tests."""
    base = make_base_open_hand(scale=scale)
    dt_ms = 1000.0 / float(fps)
    out: List[Tuple[float, List[Landmark]]] = []
    frame_idx = 0

    def _emit(closure_val: float, cur_curl: float) -> None:
        nonlocal frame_idx
        ts_ms = frame_idx * dt_ms
        t_s = ts_ms / 1000.0
        dx = wrist_vel_xy[0] * t_s
        dy = wrist_vel_xy[1] * t_s
        lms = make_pinch_frame(
            base,
            target_pose,
            closure_val,
            curl_fingers=curl_fingers,
            curl_amount=cur_curl,
        )
        if dx != 0.0 or dy != 0.0:
            lms = [Landmark(p.x + dx, p.y + dy, p.z) for p in lms]
        out.append((ts_ms, lms))
        frame_idx += 1

    if start_closed:
        for _ in range(open_frames + close_frames + hold_frames):
            _emit(1.0, curl_amount)
        return out

    for _ in range(open_frames):
        _emit(0.0, curl_amount * 0.2)
    for k in range(1, max(1, close_frames) + 1):
        frac = k / float(max(1, close_frames))
        _emit(frac, curl_amount * frac)
    for _ in range(hold_frames):
        _emit(1.0, curl_amount)
    return out
