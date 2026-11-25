"""
Airpoint-style visual overlay renderer for the Floating Camera HUD & Preview window.
Draws anti-aliased 21-landmark hand skeletons for BOTH hands (teal Left, blue Right),
active pinch/dictation/scroll/drag fingertip highlights, 3D pointing-gate status,
and Palm Wheel activation progress rings.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from aerohand.core.types import HandInput, HandOutput, PoseName

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

# BGR colors matching Airpoint's clean teal/blue dark glass aesthetic
COL_TEAL = (191, 212, 45)      # #2dd4bf (Left hand)
COL_BLUE = (250, 165, 96)      # #60a5fa (Right hand)
COL_GREEN = (128, 232, 74)
COL_AMBER = (36, 191, 251)
COL_ROSE = (113, 113, 248)
COL_PURPLE = (240, 130, 190)
COL_WHITE = (245, 247, 250)
COL_MUTED = (165, 172, 184)
COL_BONE = (210, 218, 228)

_POSE_HIGHLIGHT_TIPS: Dict[PoseName, Tuple[int, ...]] = {
    PoseName.THUMB_INDEX_PINCH: (4, 8),
    PoseName.THUMB_MIDDLE_PINCH: (4, 12),
    PoseName.THUMB_INDEX_MIDDLE_PINCH: (4, 8, 12),
    PoseName.THUMB_RING_PINCH: (4, 16),
    PoseName.THUMB_PINKY_PINCH: (4, 20),
    PoseName.THUMB_PINKY_BASE: (4, 17),
}


def render_camera_hud(
    frame_bgr: np.ndarray,
    raw_hands: List[HandInput],
    outputs: List[HandOutput],
    primary: Optional[HandOutput],
    *,
    fps: float,
    paused: bool,
    dry_run: bool,
    is_idle: bool,
    guard_reason: str = "",
    dictating: bool = False,
    zooming: bool = False,
    show_connections: bool = True,
    show_landmarks: bool = True,
    compact: bool = True,
) -> np.ndarray:
    """Render an Airpoint-style floating camera HUD frame (BGR)."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    out_by_hand: Dict[str, HandOutput] = {o.hand: o for o in outputs}

    for hand_inp in raw_hands:
        accent = COL_BLUE if hand_inp.handedness == "Right" else COL_TEAL
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_inp.landmarks[:21]]

        if show_connections:
            for a, b in HAND_CONNECTIONS:
                cv2.line(vis, pts[a], pts[b], COL_BONE, 2, cv2.LINE_AA)
        if show_landmarks:
            for idx, (px, py) in enumerate(pts):
                if idx in (4, 8, 12, 16, 20):
                    cv2.circle(vis, (px, py), 5, accent, -1, cv2.LINE_AA)
                    cv2.circle(vis, (px, py), 6, COL_WHITE, 1, cv2.LINE_AA)
                else:
                    cv2.circle(vis, (px, py), 3, accent, -1, cv2.LINE_AA)

        ho = out_by_hand.get(hand_inp.handedness)
        if ho is not None:
            p = ho.pose
            c = ho.cursor
            ap = p.active_pose or p.pose
            ring_tips = _POSE_HIGHLIGHT_TIPS.get(ap)
            if p.is_clicking or p.is_pinching:
                for tip_idx in ring_tips or (4, 12):
                    cv2.circle(vis, pts[tip_idx], 14, COL_GREEN, 2, cv2.LINE_AA)
            elif p.is_grabbing:
                for tip_idx in ring_tips or (4, 8, 12):
                    cv2.circle(vis, pts[tip_idx], 14, COL_AMBER, 3 if tip_idx != 4 else 2, cv2.LINE_AA)
            elif p.is_dictating:
                for tip_idx in ring_tips or (4, 8):
                    cv2.circle(vis, pts[tip_idx], 15, COL_PURPLE, 3 if tip_idx != 4 else 2, cv2.LINE_AA)
            elif p.is_scrolling:
                for tip_idx in ring_tips or (4, 16):
                    cv2.circle(vis, pts[tip_idx], 14, COL_TEAL, 2, cv2.LINE_AA)
            elif p.is_window_tiling:
                for tip_idx in ring_tips or (4, 20):
                    cv2.circle(vis, pts[tip_idx], 14, COL_BLUE, 2, cv2.LINE_AA)
            elif p.is_spaces_nav:
                for tip_idx in ring_tips or (4, 17):
                    cv2.circle(vis, pts[tip_idx], 14, COL_BLUE, 2, cv2.LINE_AA)
            elif c.is_pointing:
                cv2.circle(vis, pts[8], 10, accent, 2, cv2.LINE_AA)

            # Palm Wheel activation progress arc
            if p.palm_wheel_state in ("activating", "active", "confirmed"):
                pcx = (pts[0][0] + pts[9][0]) // 2
                pcy = (pts[0][1] + pts[9][1]) // 2
                rad = 36
                ring_col = COL_GREEN if p.palm_wheel_state == "active" else COL_AMBER
                cv2.circle(vis, (pcx, pcy), rad, (50, 58, 72), 3, cv2.LINE_AA)
                end_angle = int(360 * max(0.0, min(1.0, p.palm_wheel_progress)))
                cv2.ellipse(vis, (pcx, pcy), (rad, rad), -90, 0, end_angle, ring_col, 4, cv2.LINE_AA)

    # Top translucent status pill bar
    overlay = vis.copy()
    bar_h = 32 if compact else 48
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (12, 16, 24), -1)
    cv2.addWeighted(overlay, 0.72, vis, 0.28, 0.0, vis)

    if paused:
        dot_col, state_txt = COL_AMBER, "PAUSED"
    elif dictating:
        dot_col, state_txt = COL_PURPLE, "DICTATING"
    elif zooming:
        dot_col, state_txt = COL_TEAL, "ZOOMING"
    elif guard_reason:
        dot_col, state_txt = COL_ROSE, f"GUARD: {guard_reason}"
    elif dry_run:
        dot_col, state_txt = COL_AMBER, "DRY-RUN"
    elif is_idle:
        dot_col, state_txt = COL_MUTED, "IDLE (8 FPS)"
    else:
        dot_col, state_txt = COL_GREEN, "LIVE"

    cv2.circle(vis, (16, 16), 5, dot_col, -1, cv2.LINE_AA)
    cv2.putText(
        vis,
        f"AeroHand  {state_txt}",
        (28, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COL_WHITE,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        f"{fps:4.0f} FPS",
        (w - 76, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COL_MUTED,
        1,
        cv2.LINE_AA,
    )

    # Bottom status badge
    if primary is not None:
        p = primary.pose
        c = primary.cursor
        pose_label = p.active_pose.value if p.active_pose else p.pose.value
        if pose_label == PoseName.IDLE.value:
            pose_label = "pointing" if c.is_pointing else "resting (gate)"
        gate_col = COL_GREEN if c.is_pointing else COL_MUTED
        pill_y = h - 28
        ov2 = vis.copy()
        cv2.rectangle(ov2, (0, pill_y), (w, h), (12, 16, 24), -1)
        cv2.addWeighted(ov2, 0.72, vis, 0.28, 0.0, vis)
        cv2.putText(
            vis,
            f"{primary.hand[0]}: {pose_label} | dirZ {c.pointing_dir_z:+.2f}",
            (12, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            gate_col,
            1,
            cv2.LINE_AA,
        )

    return vis
