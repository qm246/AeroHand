"""
Zero-setup 3D Geometric Pinch Probability Estimator (MVP Brain — V050-01 / V051-07 / V051-08).

Operates strictly in the rotation-, translation-, and scale-invariant 3D `palm_frame`
computed from MediaPipe's metric `world_landmarks` (falling back to 2D+z if unavailable).
Outputs calibrated soft probabilities over the 7 Airpoint pinch classes so it plugs
directly into `PoseStateMachine` and can be hot-swapped later for a self-trained ONNX TCN.

Anti-false-activation guards built in (all individually configurable via `classifier.heuristic_3d`):
  1. Fist / grip rejection: when >= 3 non-thumb fingertips are tightly curled into the palm
     simultaneously (drinking, holding phone, resting fist), suppresses pinch scores to `idle`.
  2. Approach gate (`enable_approach_gate`): requires the pinching finger pair to have been
     clearly separated (`dist >= on_dist + approach_open_margin`) within `approach_window_ms`
     with FPS-independent exponential decay (`approach_decay_tau_ms`) prior to entering a new pinch.
  3. Wrist motion & rotation gate (`enable_motion_gate`): rejects new pinch entries during
     rapid palm-width-normalized wrist translation (`max_wrist_speed_pw`) or 3D palm rotation
     (`max_wrist_ang_speed_deg_s`) using FPS-independent time-based EMA (`motion_ema_tau_ms`),
     while never interrupting an already-latched hold.
  4. Index posture gate (`enable_posture_gate`): requires extended index finger (`index_ext >=
     index_ext_min`) before entering non-index pointer pinches (`thumb_middle_pinch`,
     `thumb_ring_pinch`, `thumb_pinky_pinch`).
  5. Half-curled loose-grip guard (`enable_grip_guard`): blocks loose/ambiguous contact when
     >= `grip_half_curled_min` fingers are half-curled (cup/phone grip) without affecting tight
     intentional pinches (`dist <= pinch_on_dist`).
  6. Margin disambiguation: distinguishes `thumb_index_middle_pinch` (Drag) from
     `thumb_middle_pinch` (Left Click) and damps ambiguous ties toward `idle`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from aerohand.core.palm_frame import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
    compute_finger_extension_ratios,
    compute_palm_frame,
)
from aerohand.core.types import (
    ALL_POSE_NAMES,
    Handedness,
    Landmark,
    PoseName,
    PoseProbabilities,
    idle_probabilities,
)
from aerohand.ml.base import PinchPrediction

_NON_INDEX_POSTURE_POSES: Tuple[PoseName, ...] = (
    PoseName.THUMB_MIDDLE_PINCH,
    PoseName.THUMB_RING_PINCH,
    PoseName.THUMB_PINKY_PINCH,
)
_REF_PALM_WIDTH_2D: float = 0.12


def _smooth_closeness(dist: float, on_dist: float, off_dist: float) -> float:
    """Map distance in palm-width units to [0, 1] contact score (1 at <= on_dist, 0 at >= off_dist)."""
    if dist <= on_dist:
        return 1.0
    if dist >= off_dist or off_dist <= on_dist:
        return 0.0
    t = (off_dist - dist) / (off_dist - on_dist)
    # Smoothstep for clean transition across Schmitt thresholds
    return t * t * (3.0 - 2.0 * t)


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """3D Euclidean distance from point `p` to closed line segment `a`–`b`."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / denom)))
    return float(np.linalg.norm(p - (a + t * ab)))


class _PerHandIntentState:
    __slots__ = (
        "last_ts_ms",
        "wrist_xy",
        "rot_mat",
        "speed_pw_ema",
        "speed_img_ema",
        "ang_ema",
        "last_open_ms",
        "last_open_dist",
        "last_shaft_open_ms",
        "last_shaft_open_dist",
        "latched",
        "latch_info",
        "shaft_ineligible_frames",
    )

    def __init__(self, ts_ms: float, wrist_xy: Tuple[float, float], rot_mat: np.ndarray) -> None:
        self.last_ts_ms: float = ts_ms
        self.wrist_xy: Tuple[float, float] = wrist_xy
        self.rot_mat: np.ndarray = rot_mat
        self.speed_pw_ema: float = 0.0
        self.speed_img_ema: float = 0.0
        self.ang_ema: float = 0.0
        self.last_open_ms: Dict[PoseName, float] = {}
        self.last_open_dist: Dict[PoseName, float] = {}
        self.last_shaft_open_ms: Dict[PoseName, float] = {}
        self.last_shaft_open_dist: Dict[PoseName, float] = {}
        self.latched: Set[PoseName] = set()
        self.latch_info: Dict[PoseName, Dict[str, Any]] = {}
        self.shaft_ineligible_frames: Dict[PoseName, int] = {}


class Heuristic3DPinchClassifier:
    """Rotation-invariant 3D distance estimator with FPS- & scale-invariant intent gates."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self._hands: Dict[Handedness, _PerHandIntentState] = {}
        self.last_debug: Dict[Handedness, Dict[str, Any]] = {}
        self.update_config(cfg)

    def update_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        """Update thresholds and intent-gate parameters in-place without clearing hand state (V051-06)."""
        c = cfg or {}
        self.pinch_on_dist: float = float(c.get("pinch_on_dist", 0.34))
        self.pinch_off_dist: float = float(c.get("pinch_off_dist", 0.62))
        self.pinky_base_on_dist: float = float(c.get("pinky_base_on_dist", 0.42))
        self.pinky_base_off_dist: float = float(c.get("pinky_base_off_dist", 0.72))
        self.dual_pinch_on_dist: float = float(c.get("dual_pinch_on_dist", 0.38))
        self.dual_pinch_off_dist: float = float(c.get("dual_pinch_off_dist", 0.65))
        self.min_margin: float = float(c.get("min_margin", 0.08))

        # V050-01 / V051-07: Data-driven FPS- & scale-invariant intent gates
        self.enable_approach_gate: bool = bool(c.get("enable_approach_gate", True))
        self.approach_window_ms: float = max(0.0, float(c.get("approach_window_ms", 1500.0)))
        self.approach_open_margin: float = max(0.0, float(c.get("approach_open_margin", 0.04)))
        self.approach_decay_tau_s: float = max(0.05, float(c.get("approach_decay_tau_ms", 400.0)) / 1000.0)

        self.enable_posture_gate: bool = bool(c.get("enable_posture_gate", True))
        self.index_ext_min: float = max(0.5, float(c.get("index_ext_min", 1.25)))

        self.enable_motion_gate: bool = bool(c.get("enable_motion_gate", True))
        self.max_wrist_speed: float = max(0.05, float(c.get("max_wrist_speed", 0.18)))
        if "max_wrist_speed_pw" in c and c["max_wrist_speed_pw"] is not None:
            self.max_wrist_speed_pw: float = max(0.2, float(c["max_wrist_speed_pw"]))
        elif "max_wrist_speed" in c and c["max_wrist_speed"] is not None:
            self.max_wrist_speed_pw = max(0.2, self.max_wrist_speed / _REF_PALM_WIDTH_2D)
        else:
            self.max_wrist_speed_pw = 1.50
        self.motion_ema_tau_s: float = max(0.005, float(c.get("motion_ema_tau_ms", 32.0)) / 1000.0)
        self.max_wrist_ang_speed_deg_s: float = max(20.0, float(c.get("max_wrist_ang_speed_deg_s", 95.0)))

        self.enable_grip_guard: bool = bool(c.get("enable_grip_guard", True))
        self.grip_half_curled_min: int = max(2, int(c.get("grip_half_curled_min", 3)))
        self.grip_loose_ratio: float = max(0.7, float(c.get("grip_loose_ratio", 1.05)))
        self.fist_guard_requires_index_curled: bool = bool(c.get("fist_guard_requires_index_curled", True))
        self.fist_guard_protect_latch_ms: float = max(0.0, float(c.get("fist_guard_protect_latch_ms", 500.0)))

        # Airpoint ergonomic curled-finger shaft presses (1/2/3 extended fingers) + relaxed
        # thumb-in-palm tuck (4 slightly extended/curved fingers -> spaces_nav).
        self.enable_airpoint_shaft_press: bool = bool(c.get("enable_airpoint_shaft_press", True))
        self.shaft_on_dist: float = float(c.get("shaft_on_dist", 0.29))
        self.shaft_off_dist: float = float(c.get("shaft_off_dist", 0.48))
        self.shaft_open_dist: float = float(c.get("shaft_open_dist", 0.52))
        self.shaft_eligibility_grace_frames: int = max(0, int(c.get("shaft_eligibility_grace_frames", 3)))
        self.shaft_grace_min_latch_age_ms: float = max(0.0, float(c.get("shaft_grace_min_latch_age_ms", 200.0)))
        self.index_shaft_on_dist: float = float(c.get("index_shaft_on_dist", 0.24))
        self.index_shaft_off_dist: float = float(c.get("index_shaft_off_dist", 0.42))
        self.index_shaft_open_dist: float = float(c.get("index_shaft_open_dist", 0.60))
        self.palm_tuck_on_dist: float = float(c.get("palm_tuck_on_dist", 0.44))
        self.palm_tuck_off_dist: float = float(c.get("palm_tuck_off_dist", 0.66))
        self.palm_tuck_open_dist: float = float(c.get("palm_tuck_open_dist", 0.72))
        self.spaces_nav_max_wrist_speed_pw: float = max(
            self.max_wrist_speed_pw, float(c.get("spaces_nav_max_wrist_speed_pw", 4.2))
        )
        self.spaces_nav_max_wrist_ang_deg_s: float = max(
            self.max_wrist_ang_speed_deg_s, float(c.get("spaces_nav_max_wrist_ang_deg_s", 240.0))
        )

        self._state_gap_reset_ms: float = float(c.get("state_gap_reset_ms", 250.0))
        self._latch_score_threshold: float = float(c.get("latch_score_threshold", 0.85))

    def reset(self, hand: Optional[Handedness] = None) -> None:
        if hand is None:
            self._hands.clear()
            self.last_debug.clear()
        else:
            self._hands.pop(hand, None)
            self.last_debug.pop(hand, None)

    def predict_only_pose(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
        only_pose: PoseName,
    ) -> PinchPrediction:
        """
        Evaluate ONLY `only_pose` (e.g. `THUMB_PINKY_BASE` under upright-palm Spaces Nav)
        without mutating open-history or latches for any other pinch class.
        """
        return self._predict_impl(
            landmarks_2d, world_landmarks, handedness, timestamp_ms, only_pose=only_pose
        )

    def predict(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
    ) -> PinchPrediction:
        return self._predict_impl(landmarks_2d, world_landmarks, handedness, timestamp_ms, only_pose=None)

    def _predict_impl(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
        *,
        only_pose: Optional[PoseName] = None,
    ) -> PinchPrediction:
        src = world_landmarks if (world_landmarks is not None and len(world_landmarks) >= 21) else landmarks_2d
        pf = compute_palm_frame(
            src,
            anchor="palm_center",
            is_left_hand=(handedness == "Left"),
            scale_metric="palm_width",
        )
        if pf is None:
            self.last_debug[handedness] = {
                "valid_palm_frame": False,
                "blocked_by": {},
                "latched": [],
                "latch_age_ms": {},
                "latch_entry": {},
            }
            return PinchPrediction(pose=PoseName.IDLE, probabilities=idle_probabilities(), confidence="low")

        # Update per-hand FPS- & palm-size-invariant motion & rotation kinematics (V051-07)
        rot_mat = np.vstack([pf.x_axis, pf.y_axis, pf.z_axis])
        wx = float(landmarks_2d[0].x) if landmarks_2d else 0.0
        wy = float(landmarks_2d[0].y) if landmarks_2d else 0.0
        palm_width_2d = 0.0
        if landmarks_2d and len(landmarks_2d) >= 18:
            palm_width_2d = math.hypot(
                float(landmarks_2d[INDEX_MCP].x) - float(landmarks_2d[PINKY_MCP].x),
                float(landmarks_2d[INDEX_MCP].y) - float(landmarks_2d[PINKY_MCP].y),
            )
        norm_pw = palm_width_2d if palm_width_2d > 1e-4 else _REF_PALM_WIDTH_2D

        st = self._hands.get(handedness)
        dt_s = 0.0
        if (
            st is None
            or timestamp_ms < st.last_ts_ms
            or (timestamp_ms - st.last_ts_ms) > self._state_gap_reset_ms
        ):
            st = _PerHandIntentState(timestamp_ms, (wx, wy), rot_mat)
            self._hands[handedness] = st
            wrist_speed_img = 0.0
            wrist_speed_pw = 0.0
            wrist_ang = 0.0
        else:
            dt_s = max(1e-3, (timestamp_ms - st.last_ts_ms) / 1000.0)
            wrist_speed_img = math.hypot(wx - st.wrist_xy[0], wy - st.wrist_xy[1]) / dt_s
            wrist_speed_pw = wrist_speed_img / norm_pw
            r_rel = rot_mat @ st.rot_mat.T
            cos_theta = max(-1.0, min(1.0, (float(np.trace(r_rel)) - 1.0) * 0.5))
            wrist_ang = math.degrees(math.acos(cos_theta)) / dt_s

            alpha = 1.0 - math.exp(-dt_s / self.motion_ema_tau_s)
            st.speed_pw_ema = alpha * wrist_speed_pw + (1.0 - alpha) * st.speed_pw_ema
            st.speed_img_ema = alpha * wrist_speed_img + (1.0 - alpha) * st.speed_img_ema
            st.ang_ema = alpha * wrist_ang + (1.0 - alpha) * st.ang_ema
            st.last_ts_ms = timestamp_ms
            st.wrist_xy = (wx, wy)
            st.rot_mat = rot_mat

        eff_speed_pw = max(wrist_speed_pw, st.speed_pw_ema)
        eff_speed_img = max(wrist_speed_img, st.speed_img_ema)
        eff_ang = max(wrist_ang, st.ang_ema)

        # In single-pose mode (e.g. upright-palm spaces_only_mode), release any stale latches
        # belonging to other poses immediately so they can never survive past the 250ms state reset.
        if only_pose is not None:
            for lp in list(st.latched):
                if lp != only_pose:
                    st.latched.discard(lp)
                    st.latch_info.pop(lp, None)
                    st.shaft_ineligible_frames.pop(lp, None)

        # 3D normalized distances from thumb tip (LM4) in palm-width units (LM5-LM17 = 1.0)
        c = pf.coords_normalized
        thumb = c[THUMB_TIP]
        exts = compute_finger_extension_ratios(pf)
        idx_ext = exts.get("index", 2.0)
        mid_ext = exts.get("middle", 2.0)
        ring_ext = exts.get("ring", 2.0)
        pinky_ext = exts.get("pinky", 2.0)
        idx_dist_to_palm = float(np.linalg.norm(c[INDEX_TIP]))

        d_idx_tip = pf.dist(THUMB_TIP, INDEX_TIP)
        d_mid_tip = pf.dist(THUMB_TIP, MIDDLE_TIP)
        d_ring_tip = pf.dist(THUMB_TIP, RING_TIP)
        d_pinky_tip = pf.dist(THUMB_TIP, PINKY_TIP)
        d_pbase_mcp = pf.dist(THUMB_TIP, PINKY_MCP)
        d_idx_mid_tips = pf.dist(INDEX_TIP, MIDDLE_TIP)
        d_dual = max(d_idx_tip, d_mid_tip)

        tip_specs: Dict[PoseName, Tuple[float, float, float]] = {
            PoseName.THUMB_INDEX_PINCH: (d_idx_tip, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_MIDDLE_PINCH: (d_mid_tip, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_RING_PINCH: (d_ring_tip, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_PINKY_PINCH: (d_pinky_tip, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_PINKY_BASE: (d_pbase_mcp, self.pinky_base_on_dist, self.pinky_base_off_dist),
            PoseName.THUMB_INDEX_MIDDLE_PINCH: (d_dual, self.dual_pinch_on_dist, self.dual_pinch_off_dist),
        }

        # Airpoint ergonomic curled-finger shaft presses (1/2/3 extended fingers) + relaxed
        # thumb-in-palm tuck (4 slightly extended/curved fingers -> spaces_nav).
        # Shaft open history is tracked independently from tip-to-tip distances so switching
        # metrics never synthesizes a false approach event.
        shaft_eligible: Dict[PoseName, bool] = {}
        shaft_specs: Dict[PoseName, Tuple[float, float, float, float]] = {}
        if self.enable_airpoint_shaft_press:
            d_idx_seg = min(
                _point_to_segment_dist(thumb, c[INDEX_PIP], c[INDEX_DIP]),
                _point_to_segment_dist(thumb, c[INDEX_DIP], c[INDEX_TIP]),
            )
            d_mid_seg = min(
                _point_to_segment_dist(thumb, c[MIDDLE_PIP], c[MIDDLE_DIP]),
                _point_to_segment_dist(thumb, c[MIDDLE_DIP], c[MIDDLE_TIP]),
            )
            d_ring_seg = min(
                _point_to_segment_dist(thumb, c[RING_PIP], c[RING_DIP]),
                _point_to_segment_dist(thumb, c[RING_DIP], c[RING_TIP]),
            )
            d_pinky_seg = min(
                _point_to_segment_dist(thumb, c[PINKY_PIP], c[PINKY_DIP]),
                _point_to_segment_dist(thumb, c[PINKY_DIP], c[PINKY_TIP]),
            )
            palm_tuck_a = 0.65 * c[MIDDLE_MCP] + 0.35 * c[WRIST]
            palm_tuck_b = 0.85 * c[PINKY_MCP] + 0.15 * c[WRIST]
            d_tuck_raw = _point_to_segment_dist(thumb, palm_tuck_a, palm_tuck_b)

            # Strict Airpoint curled-vs-extended progression: ALL non-extended fingers must
            # actually be curled toward the palm (ext < 1.15 AND tip dist_to_palm < 1.05),
            # rejecting open/phone-wrap grips where fingers stay stretched out.
            mid_dp = float(np.linalg.norm(c[MIDDLE_TIP]))
            ring_dp = float(np.linalg.norm(c[RING_TIP]))
            pinky_dp = float(np.linalg.norm(c[PINKY_TIP]))
            mid_curled = (mid_ext < 1.15) and (mid_dp < 1.05)
            ring_curled = (ring_ext < 1.12) and (ring_dp < 1.05)
            pinky_curled = (pinky_ext < 1.18) and (pinky_dp < 1.05)

            shaft_eligible = {
                PoseName.THUMB_INDEX_PINCH: (
                    self.index_shaft_on_dist > 0.0
                    and idx_dist_to_palm >= 0.85
                    and idx_ext >= 1.15
                    and mid_curled
                    and ring_curled
                    and pinky_curled
                ),
                PoseName.THUMB_MIDDLE_PINCH: (
                    idx_ext >= self.index_ext_min and mid_curled and ring_curled and pinky_curled
                ),
                PoseName.THUMB_RING_PINCH: (
                    idx_ext >= self.index_ext_min and mid_ext >= 1.35 and ring_curled and pinky_curled
                ),
                PoseName.THUMB_PINKY_PINCH: (
                    idx_ext >= self.index_ext_min
                    and mid_ext >= 1.35
                    and ring_ext >= 1.35
                    and pinky_curled
                ),
                PoseName.THUMB_PINKY_BASE: (
                    idx_ext >= 1.20
                    and mid_ext >= 1.20
                    and ring_ext >= 1.20
                    and pinky_ext >= 1.15
                    and d_tuck_raw < d_pinky_seg
                ),
            }
            shaft_specs = {
                PoseName.THUMB_INDEX_PINCH: (
                    d_idx_seg,
                    self.index_shaft_on_dist,
                    self.index_shaft_off_dist,
                    self.index_shaft_open_dist,
                ),
                PoseName.THUMB_MIDDLE_PINCH: (
                    d_mid_seg,
                    self.shaft_on_dist,
                    self.shaft_off_dist,
                    self.shaft_open_dist,
                ),
                PoseName.THUMB_RING_PINCH: (
                    d_ring_seg,
                    self.shaft_on_dist,
                    self.shaft_off_dist,
                    self.shaft_open_dist,
                ),
                PoseName.THUMB_PINKY_PINCH: (
                    d_pinky_seg,
                    self.shaft_on_dist,
                    self.shaft_off_dist,
                    self.shaft_open_dist,
                ),
                PoseName.THUMB_PINKY_BASE: (
                    d_tuck_raw,
                    self.palm_tuck_on_dist,
                    self.palm_tuck_off_dist,
                    self.palm_tuck_open_dist,
                ),
            }

        # Snapshot prior open timestamps & distances BEFORE updating with current frame so
        # a static finger hover can never satisfy its own approach gate without real closure.
        prev_open_ms = dict(st.last_open_ms)
        prev_open_dist = dict(st.last_open_dist)
        prev_shaft_open_ms = dict(st.last_shaft_open_ms)
        prev_shaft_open_dist = dict(st.last_shaft_open_dist)
        decay_factor = math.exp(-dt_s / self.approach_decay_tau_s) if dt_s > 0.0 else 1.0
        for pose_key, (dist_val, on_d, _off_d) in tip_specs.items():
            if only_pose is not None and pose_key != only_pose:
                continue
            if dist_val >= (on_d + self.approach_open_margin):
                prev_t = st.last_open_ms.get(pose_key)
                prev_d = st.last_open_dist.get(pose_key, 0.0)
                if prev_t is None or (timestamp_ms - prev_t) > self.approach_window_ms or dist_val >= prev_d:
                    st.last_open_dist[pose_key] = dist_val
                else:
                    # FPS-independent exponential decay toward current open distance (V051-07)
                    st.last_open_dist[pose_key] = max(
                        dist_val, decay_factor * prev_d + (1.0 - decay_factor) * dist_val
                    )
                st.last_open_ms[pose_key] = timestamp_ms

        for pose_key, (s_val, _s_on, _s_off, s_open) in shaft_specs.items():
            if only_pose is not None and pose_key != only_pose:
                continue
            if s_val >= s_open:
                prev_t = st.last_shaft_open_ms.get(pose_key)
                prev_d = st.last_shaft_open_dist.get(pose_key, 0.0)
                if prev_t is None or (timestamp_ms - prev_t) > self.approach_window_ms or s_val >= prev_d:
                    st.last_shaft_open_dist[pose_key] = s_val
                else:
                    st.last_shaft_open_dist[pose_key] = max(
                        s_val, decay_factor * prev_d + (1.0 - decay_factor) * s_val
                    )
                st.last_shaft_open_ms[pose_key] = timestamp_ms

        # Apply a brief eligibility grace period (default 3 frames) for ESTABLISHED shaft holds
        # (`used_shaft == True` AND latch age >= `shaft_grace_min_latch_age_ms`) whose physical
        # thumb-to-shaft contact stays below `s_off`. This prevents 1-3 frame finger-extension
        # flaps from dropping an active hold (e.g. dictation/scroll/tile/spaces) without stretching
        # 1-frame transient false contacts into confirmed triggers.
        eff_shaft_eligible: Dict[PoseName, bool] = dict(shaft_eligible)
        for pose_key, (s_val, _s_on, s_off, _s_open) in shaft_specs.items():
            if only_pose is not None and pose_key != only_pose:
                continue
            info = st.latch_info.get(pose_key)
            latch_age = (float(timestamp_ms) - float(info["since_ms"])) if info is not None else 0.0
            was_established_shaft = (
                (pose_key in st.latched)
                and bool((info or {}).get("used_shaft", False))
                and latch_age >= self.shaft_grace_min_latch_age_ms
            )
            if shaft_eligible.get(pose_key, False):
                st.shaft_ineligible_frames.pop(pose_key, None)
            elif was_established_shaft and s_val < s_off and self.shaft_eligibility_grace_frames > 0:
                miss_cnt = st.shaft_ineligible_frames.get(pose_key, 0) + 1
                st.shaft_ineligible_frames[pose_key] = miss_cnt
                if miss_cnt <= self.shaft_eligibility_grace_frames:
                    eff_shaft_eligible[pose_key] = True
            else:
                st.shaft_ineligible_frames.pop(pose_key, None)

        for pose_key, (dist_val, _on_d, off_d) in tip_specs.items():
            if only_pose is not None and pose_key != only_pose:
                continue
            tip_off = dist_val >= off_d
            if pose_key in shaft_specs and eff_shaft_eligible.get(pose_key, False):
                s_val, _s_on, s_off, _s_open = shaft_specs[pose_key]
                shaft_off = s_val >= s_off
            else:
                shaft_off = True
            if tip_off and shaft_off:
                st.latched.discard(pose_key)
                st.latch_info.pop(pose_key, None)
                st.shaft_ineligible_frames.pop(pose_key, None)

        # Raw scores in [0, 1]: best of classic tip-to-tip contact and eligible Airpoint shaft/tuck
        raw_tip_scores: Dict[PoseName, float] = {
            PoseName.THUMB_INDEX_PINCH: _smooth_closeness(d_idx_tip, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_MIDDLE_PINCH: _smooth_closeness(d_mid_tip, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_RING_PINCH: _smooth_closeness(d_ring_tip, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_PINKY_PINCH: _smooth_closeness(d_pinky_tip, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_PINKY_BASE: _smooth_closeness(d_pbase_mcp, self.pinky_base_on_dist, self.pinky_base_off_dist),
            PoseName.THUMB_INDEX_MIDDLE_PINCH: _smooth_closeness(
                d_dual, self.dual_pinch_on_dist, self.dual_pinch_off_dist
            ),
        }
        raw_shaft_scores: Dict[PoseName, float] = {}
        for pose_key, (s_val, s_on, s_off, _s_open) in shaft_specs.items():
            if eff_shaft_eligible.get(pose_key, False) and s_on > 0.0:
                raw_shaft_scores[pose_key] = _smooth_closeness(s_val, s_on, s_off)
            else:
                raw_shaft_scores[pose_key] = 0.0

        used_shaft: Set[PoseName] = set()
        scores: Dict[PoseName, float] = {}
        eff_dist_on: Dict[PoseName, Tuple[float, float]] = {}
        for pose_key, t_sc in raw_tip_scores.items():
            s_sc = raw_shaft_scores.get(pose_key, 0.0)
            # Prefer shaft mode when shaft score strictly beats tip score, OR when both saturate
            # on a curled non-index finger (enforcing the wider shaft anti-jitter open threshold).
            prefer_shaft = (s_sc > t_sc) or (
                s_sc > 0.0
                and s_sc >= t_sc
                and pose_key in _NON_INDEX_POSTURE_POSES
            )
            if prefer_shaft:
                scores[pose_key] = s_sc
                used_shaft.add(pose_key)
                s_val, s_on, _s_off, _s_open = shaft_specs[pose_key]
                eff_dist_on[pose_key] = (s_val, s_on)
            else:
                scores[pose_key] = t_sc
                t_val, t_on, _t_off = tip_specs[pose_key]
                eff_dist_on[pose_key] = (t_val, t_on)

        # Dual pinch (Drag): both index AND middle close to thumb, and tips close together
        if d_idx_mid_tips > 0.55:
            scores[PoseName.THUMB_INDEX_MIDDLE_PINCH] *= 0.25

        # Disambiguate dual pinch vs single index/middle pinch
        s_dual = scores[PoseName.THUMB_INDEX_MIDDLE_PINCH]
        if s_dual >= 0.50:
            scores[PoseName.THUMB_INDEX_PINCH] *= max(0.0, 1.0 - s_dual)
            scores[PoseName.THUMB_MIDDLE_PINCH] *= max(0.0, 1.0 - s_dual)

        # Check whether any canonical pinch pose is currently in tight contact (V051-09)
        any_tight_pinch = any(
            d_v <= on_v
            for pk, (d_v, on_v) in eff_dist_on.items()
            if pk != PoseName.THUMB_INDEX_MIDDLE_PINCH
        )
        active_tip_idx: Optional[int] = None
        if any_tight_pinch:
            tip_candidates = (
                (INDEX_TIP, PoseName.THUMB_INDEX_PINCH),
                (MIDDLE_TIP, PoseName.THUMB_MIDDLE_PINCH),
                (RING_TIP, PoseName.THUMB_RING_PINCH),
                (PINKY_TIP, PoseName.THUMB_PINKY_PINCH),
            )
            best_tip, best_ratio = min(
                (
                    (ti, eff_dist_on[pk][0] / max(1e-6, eff_dist_on[pk][1]))
                    for ti, pk in tip_candidates
                ),
                key=lambda kv: kv[1],
            )
            if best_ratio <= 1.0:
                active_tip_idx = best_tip

        # Fist / grip guard: count how many non-pinching fingertips are tightly or half curled toward palm center
        curled_count = 0
        all_curled_count = 0
        half_curled_count = 0
        for tip_idx in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
            dist_to_palm = idx_dist_to_palm if tip_idx == INDEX_TIP else float(np.linalg.norm(c[tip_idx]))
            if dist_to_palm < 0.72:
                all_curled_count += 1
                if tip_idx != active_tip_idx:
                    curled_count += 1
            if dist_to_palm < 1.05:
                half_curled_count += 1

        # If >= 3 non-pinching fingers are curled tightly into a fist/grip (holding phone/cup),
        # suppress loose pinch — never cancelling an already-latched hold. When all 4 non-thumb
        # fingertips (including index: `idx_dist_to_palm < 0.72`) are tightly curled into a full
        # fist (V052-04 / V053-01), drop ONLY young/transient latches (< fist_guard_protect_latch_ms)
        # while preserving long-held drags/dictations (scoring only those protected latches and
        # disallowing any new latch).
        is_full_curled_fist = (
            self.fist_guard_requires_index_curled
            and curled_count >= 3
            and idx_dist_to_palm < 0.72
            and all_curled_count >= 4
        )
        protected_fist_latches: Optional[Set[PoseName]] = None
        if is_full_curled_fist:
            for lp in list(st.latched):
                info = st.latch_info.get(lp)
                age_ms = (float(timestamp_ms) - float(info["since_ms"])) if info is not None else 0.0
                if age_ms < self.fist_guard_protect_latch_ms:
                    st.latched.discard(lp)
                    st.latch_info.pop(lp, None)
                    st.shaft_ineligible_frames.pop(lp, None)
            if st.latched:
                protected_fist_latches = set(st.latched)

        pointing_middle_shaft = (
            shaft_eligible.get(PoseName.THUMB_MIDDLE_PINCH, False)
            and idx_dist_to_palm >= 1.15
            and raw_shaft_scores.get(PoseName.THUMB_MIDDLE_PINCH, 0.0) >= 0.65
        )
        if (is_full_curled_fist and not st.latched) or (
            curled_count >= 3
            and (not st.latched)
            and (not any_tight_pinch)
            and (not pointing_middle_shaft)
        ):
            self.last_debug[handedness] = {
                "valid_palm_frame": True,
                "latched": [],
                "latch_age_ms": {},
                "latch_entry": {},
                "blocked_by": {"all": ["fist"]},
                "blocked_reasons": ["fist"],
                "eff_speed_pw": round(eff_speed_pw, 4),
                "eff_speed_img": round(eff_speed_img, 4),
                "eff_ang_deg_s": round(eff_ang, 2),
                "palm_width_2d": round(palm_width_2d, 5),
                "open_dist": {k.value: round(v, 4) for k, v in prev_open_dist.items()},
                "idx_ext": round(idx_ext, 4),
                "finger_ext": {k: round(v, 3) for k, v in exts.items()},
                "used_shaft": sorted(p.value for p in used_shaft),
                "curled_count": curled_count,
                "half_curled_count": half_curled_count,
                "raw_top_pose": PoseName.IDLE.value,
                "raw_top_score": 0.0,
            }
            return PinchPrediction(pose=PoseName.IDLE, probabilities=idle_probabilities(), confidence="high")

        # Prevent pinky tip/shaft pinch from being confused with pinky base / palm tuck slide.
        # Use effective 3D distance as tie-breaker when both saturate at 1.0 (short curved pinky).
        s_pinky = scores[PoseName.THUMB_PINKY_PINCH]
        s_pbase = scores[PoseName.THUMB_PINKY_BASE]
        d_pinky_eff = eff_dist_on[PoseName.THUMB_PINKY_PINCH][0]
        d_pbase_eff = eff_dist_on[PoseName.THUMB_PINKY_BASE][0]
        if s_pinky > s_pbase or (s_pinky >= 0.5 and s_pbase >= 0.5 and d_pinky_eff < d_pbase_eff):
            scores[PoseName.THUMB_PINKY_BASE] *= 0.4
        elif (
            s_pbase > s_pinky or (s_pinky >= 0.5 and s_pbase >= 0.5 and d_pbase_eff < d_pinky_eff)
        ) and d_pinky_tip > self.pinch_off_dist:
            scores[PoseName.THUMB_PINKY_PINCH] *= 0.3

        if only_pose is not None:
            for pose_key in list(scores.keys()):
                if pose_key != only_pose:
                    scores[pose_key] = 0.0

        raw_top_pose, raw_top_score = max(scores.items(), key=lambda kv: kv[1])
        blocked_by: Dict[str, List[str]] = {}
        all_reasons: Set[str] = set()

        # Under a full curled fist with protected long-held latches (V053-01), score ONLY the
        # protected latches and block all other poses (never allowing new latches to be acquired).
        if protected_fist_latches is not None:
            for pose_key, raw_sc in list(scores.items()):
                if pose_key not in protected_fist_latches:
                    if raw_sc > 0.0:
                        blocked_by[pose_key.value] = ["fist"]
                        all_reasons.add("fist")
                    scores[pose_key] = 0.0
        # Otherwise, if the leading pose is already latched, preserve the full 3D score distribution so
        # ongoing holds remain strictly rotation-invariant and uninterrupted by wrist motion.
        elif raw_top_pose not in st.latched:
            for pose_key, raw_sc in list(scores.items()):
                if raw_sc <= 0.0 or pose_key in st.latched:
                    continue
                dist_val, on_d = eff_dist_on[pose_key]
                reasons: List[str] = []
                if self.enable_approach_gate:
                    if pose_key in used_shaft:
                        _sv, s_on, _s_off, s_open = shaft_specs[pose_key]
                        last_open = prev_shaft_open_ms.get(pose_key)
                        open_dist = prev_shaft_open_dist.get(pose_key, 0.0)
                        req_delta = max(self.approach_open_margin, s_open - s_on)
                        if (
                            last_open is None
                            or (timestamp_ms - last_open) > self.approach_window_ms
                            or open_dist < s_open
                            or (open_dist - dist_val) < req_delta
                        ):
                            reasons.append("approach")
                    else:
                        last_open = prev_open_ms.get(pose_key)
                        open_dist = prev_open_dist.get(pose_key, 0.0)
                        if (
                            last_open is None
                            or (timestamp_ms - last_open) > self.approach_window_ms
                            or (open_dist - dist_val) < self.approach_open_margin
                        ):
                            reasons.append("approach")
                if self.enable_motion_gate:
                    if (
                        only_pose == PoseName.THUMB_PINKY_BASE
                        and pose_key == PoseName.THUMB_PINKY_BASE
                        and shaft_eligible.get(PoseName.THUMB_PINKY_BASE, False)
                    ):
                        spd_lim = self.spaces_nav_max_wrist_speed_pw
                        ang_lim = self.spaces_nav_max_wrist_ang_deg_s
                    else:
                        spd_lim = self.max_wrist_speed_pw
                        ang_lim = self.max_wrist_ang_speed_deg_s
                    if eff_speed_pw > spd_lim or eff_ang > ang_lim:
                        reasons.append("motion")
                if self.enable_posture_gate:
                    if pose_key in _NON_INDEX_POSTURE_POSES and idx_ext < self.index_ext_min:
                        reasons.append("posture")
                if self.enable_grip_guard:
                    if half_curled_count >= self.grip_half_curled_min and dist_val > (on_d * self.grip_loose_ratio):
                        reasons.append("grip")

                if reasons:
                    scores[pose_key] = 0.0
                    blocked_by[pose_key.value] = reasons
                    all_reasons.update(reasons)
                elif raw_sc >= self._latch_score_threshold:
                    st.latched.add(pose_key)
                    st.latch_info[pose_key] = {
                        "since_ms": round(float(timestamp_ms), 2),
                        "dist": round(float(dist_val), 4),
                        "raw_sc": round(float(raw_sc), 4),
                        "used_shaft": pose_key in used_shaft,
                        "eff_speed_pw": round(float(eff_speed_pw), 4),
                        "eff_ang_deg_s": round(float(eff_ang), 2),
                        "open_dist": round(
                            float(
                                prev_shaft_open_dist.get(pose_key, 0.0)
                                if pose_key in used_shaft
                                else prev_open_dist.get(pose_key, 0.0)
                            ),
                            4,
                        ),
                        "idx_ext": round(float(idx_ext), 4),
                        "half_curled_count": int(half_curled_count),
                    }

        latched_sorted = sorted(st.latched, key=lambda p: p.value)
        latch_age_ms: Dict[str, float] = {}
        latch_entry: Dict[str, Dict[str, Any]] = {}
        for lp in latched_sorted:
            info = st.latch_info.get(lp)
            if info is not None:
                latch_age_ms[lp.value] = round(max(0.0, float(timestamp_ms) - float(info["since_ms"])), 2)
                latch_entry[lp.value] = dict(info)

        self.last_debug[handedness] = {
            "valid_palm_frame": True,
            "latched": [p.value for p in latched_sorted],
            "latch_age_ms": latch_age_ms,
            "latch_entry": latch_entry,
            "blocked_by": blocked_by,
            "blocked_reasons": sorted(all_reasons),
            "eff_speed_pw": round(eff_speed_pw, 4),
            "eff_speed_img": round(eff_speed_img, 4),
            "eff_ang_deg_s": round(eff_ang, 2),
            "palm_width_2d": round(palm_width_2d, 5),
            "open_dist": {k.value: round(v, 4) for k, v in prev_open_dist.items()},
            "idx_ext": round(idx_ext, 4),
            "finger_ext": {k: round(v, 3) for k, v in exts.items()},
            "used_shaft": sorted(p.value for p in used_shaft),
            "curled_count": curled_count,
            "half_curled_count": half_curled_count,
            "raw_top_pose": raw_top_pose.value if raw_top_score > 0.0 else PoseName.IDLE.value,
            "raw_top_score": round(raw_top_score, 4),
        }

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_pose, top_score = ranked[0]
        second_score = ranked[1][1]

        # Ambiguity damping: if two unrelated pinches are nearly tied, damp toward IDLE
        if top_score > 0.0 and (top_score - second_score) < self.min_margin:
            damp = max(0.0, (top_score - second_score) / max(1e-6, self.min_margin))
            top_score *= 0.5 + 0.5 * damp
            scores[top_pose] = top_score

        # Build clean probability distribution summing to 1.0
        probs: PoseProbabilities = {p: 0.0 for p in ALL_POSE_NAMES}
        idle_score = max(0.0, 1.0 - top_score)
        non_idle_sum = sum(scores.values())

        if non_idle_sum <= 1e-6:
            probs[PoseName.IDLE] = 1.0
            return PinchPrediction(pose=PoseName.IDLE, probabilities=probs, confidence="high")

        # Keep top pose score sharp (faithful to Schmitt enter/exit semantics) while normalizing
        remaining = max(0.0, 1.0 - idle_score)
        other_sum = max(0.0, non_idle_sum - top_score)
        probs[PoseName.IDLE] = idle_score
        probs[top_pose] = remaining * (top_score / (top_score + 0.35 * other_sum))
        leftover = max(0.0, 1.0 - probs[PoseName.IDLE] - probs[top_pose])
        if other_sum > 1e-6 and leftover > 0.0:
            for p, sc in scores.items():
                if p != top_pose:
                    probs[p] = leftover * (sc / other_sum)

        best_pose = max(probs.items(), key=lambda kv: kv[1])[0]
        best_prob = probs[best_pose]
        conf = "high" if best_prob > 0.8 else ("medium" if best_prob > 0.6 else "low")
        return PinchPrediction(pose=best_pose, probabilities=probs, confidence=conf)
