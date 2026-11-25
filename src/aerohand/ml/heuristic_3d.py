"""
Zero-setup 3D Geometric Pinch Probability Estimator (MVP Brain — V050-01 Data-Driven Intent Gates).

Operates strictly in the rotation-, translation-, and scale-invariant 3D `palm_frame`
computed from MediaPipe's metric `world_landmarks` (falling back to 2D+z if unavailable).
Outputs calibrated soft probabilities over the 7 Airpoint pinch classes so it plugs
directly into `PoseStateMachine` and can be hot-swapped later for a self-trained ONNX TCN.

Anti-false-activation guards built in (all individually configurable via `classifier.heuristic_3d`):
  1. Fist / grip rejection: when >= 3 non-thumb fingertips are tightly curled into the palm
     simultaneously (drinking, holding phone, resting fist), suppresses pinch scores to `idle`.
  2. Approach gate (`enable_approach_gate`): requires the pinching finger pair to have been
     clearly separated (`dist >= on_dist + approach_open_margin`) within `approach_window_ms`
     prior to entering a new pinch, rejecting static contact when a hand enters already gripped.
  3. Wrist motion & rotation gate (`enable_motion_gate`): rejects new pinch entries during
     rapid linear wrist translation or 3D palm rotation (talking/waving/turning chair) using
     smoothed velocity (`max(instant, EMA)`), while never interrupting an already-latched hold.
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
from typing import Any, Dict, Optional, Sequence, Set, Tuple

import numpy as np

from aerohand.core.palm_frame import (
    INDEX_TIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
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


def _smooth_closeness(dist: float, on_dist: float, off_dist: float) -> float:
    """Map distance in palm-width units to [0, 1] contact score (1 at <= on_dist, 0 at >= off_dist)."""
    if dist <= on_dist:
        return 1.0
    if dist >= off_dist or off_dist <= on_dist:
        return 0.0
    t = (off_dist - dist) / (off_dist - on_dist)
    # Smoothstep for clean transition across Schmitt thresholds
    return t * t * (3.0 - 2.0 * t)


class _PerHandIntentState:
    __slots__ = (
        "last_ts_ms",
        "wrist_xy",
        "rot_mat",
        "speed_ema",
        "ang_ema",
        "last_open_ms",
        "last_open_dist",
        "latched",
    )

    def __init__(self, ts_ms: float, wrist_xy: Tuple[float, float], rot_mat: np.ndarray) -> None:
        self.last_ts_ms: float = ts_ms
        self.wrist_xy: Tuple[float, float] = wrist_xy
        self.rot_mat: np.ndarray = rot_mat
        self.speed_ema: float = 0.0
        self.ang_ema: float = 0.0
        self.last_open_ms: Dict[PoseName, float] = {}
        self.last_open_dist: Dict[PoseName, float] = {}
        self.latched: Set[PoseName] = set()


class Heuristic3DPinchClassifier:
    """Rotation-invariant 3D distance estimator with data-driven intent gates."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        c = cfg or {}
        self.pinch_on_dist: float = float(c.get("pinch_on_dist", 0.34))
        self.pinch_off_dist: float = float(c.get("pinch_off_dist", 0.62))
        self.pinky_base_on_dist: float = float(c.get("pinky_base_on_dist", 0.42))
        self.pinky_base_off_dist: float = float(c.get("pinky_base_off_dist", 0.72))
        self.dual_pinch_on_dist: float = float(c.get("dual_pinch_on_dist", 0.38))
        self.dual_pinch_off_dist: float = float(c.get("dual_pinch_off_dist", 0.65))
        self.min_margin: float = float(c.get("min_margin", 0.08))

        # V050-01: Data-driven intent gates (tuned on D-dev + cadence positives)
        self.enable_approach_gate: bool = bool(c.get("enable_approach_gate", True))
        self.approach_window_ms: float = max(0.0, float(c.get("approach_window_ms", 1500.0)))
        self.approach_open_margin: float = max(0.0, float(c.get("approach_open_margin", 0.04)))

        self.enable_posture_gate: bool = bool(c.get("enable_posture_gate", True))
        self.index_ext_min: float = max(0.5, float(c.get("index_ext_min", 1.25)))

        self.enable_motion_gate: bool = bool(c.get("enable_motion_gate", True))
        self.max_wrist_speed: float = max(0.05, float(c.get("max_wrist_speed", 0.18)))
        self.max_wrist_ang_speed_deg_s: float = max(20.0, float(c.get("max_wrist_ang_speed_deg_s", 95.0)))

        self.enable_grip_guard: bool = bool(c.get("enable_grip_guard", True))
        self.grip_half_curled_min: int = max(2, int(c.get("grip_half_curled_min", 3)))
        self.grip_loose_ratio: float = max(0.7, float(c.get("grip_loose_ratio", 1.05)))

        self._state_gap_reset_ms: float = float(c.get("state_gap_reset_ms", 250.0))
        self._latch_score_threshold: float = float(c.get("latch_score_threshold", 0.85))
        self._hands: Dict[Handedness, _PerHandIntentState] = {}

    def reset(self, hand: Optional[Handedness] = None) -> None:
        if hand is None:
            self._hands.clear()
        else:
            self._hands.pop(hand, None)

    def predict(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
    ) -> PinchPrediction:
        src = world_landmarks if (world_landmarks is not None and len(world_landmarks) >= 21) else landmarks_2d
        pf = compute_palm_frame(
            src,
            anchor="palm_center",
            is_left_hand=(handedness == "Left"),
            scale_metric="palm_width",
        )
        if pf is None:
            return PinchPrediction(pose=PoseName.IDLE, probabilities=idle_probabilities(), confidence="low")

        # Update per-hand temporal motion & rotation kinematics
        rot_mat = np.vstack([pf.x_axis, pf.y_axis, pf.z_axis])
        wx = float(landmarks_2d[0].x) if landmarks_2d else 0.0
        wy = float(landmarks_2d[0].y) if landmarks_2d else 0.0
        st = self._hands.get(handedness)
        if (
            st is None
            or timestamp_ms < st.last_ts_ms
            or (timestamp_ms - st.last_ts_ms) > self._state_gap_reset_ms
        ):
            st = _PerHandIntentState(timestamp_ms, (wx, wy), rot_mat)
            self._hands[handedness] = st
            wrist_speed = 0.0
            wrist_ang = 0.0
        else:
            dt_s = max(1e-3, (timestamp_ms - st.last_ts_ms) / 1000.0)
            wrist_speed = math.hypot(wx - st.wrist_xy[0], wy - st.wrist_xy[1]) / dt_s
            r_rel = rot_mat @ st.rot_mat.T
            cos_theta = max(-1.0, min(1.0, (float(np.trace(r_rel)) - 1.0) * 0.5))
            wrist_ang = math.degrees(math.acos(cos_theta)) / dt_s
            st.speed_ema = 0.65 * wrist_speed + 0.35 * st.speed_ema
            st.ang_ema = 0.65 * wrist_ang + 0.35 * st.ang_ema
            st.last_ts_ms = timestamp_ms
            st.wrist_xy = (wx, wy)
            st.rot_mat = rot_mat

        eff_speed = max(wrist_speed, st.speed_ema)
        eff_ang = max(wrist_ang, st.ang_ema)

        # 3D normalized distances from thumb tip (LM4) in palm-width units (LM5-LM17 = 1.0)
        d_idx = pf.dist(THUMB_TIP, INDEX_TIP)
        d_mid = pf.dist(THUMB_TIP, MIDDLE_TIP)
        d_ring = pf.dist(THUMB_TIP, RING_TIP)
        d_pinky = pf.dist(THUMB_TIP, PINKY_TIP)
        d_pbase = pf.dist(THUMB_TIP, PINKY_MCP)
        d_idx_mid_tips = pf.dist(INDEX_TIP, MIDDLE_TIP)
        d_dual = max(d_idx, d_mid)

        pose_dists: Dict[PoseName, Tuple[float, float, float]] = {
            PoseName.THUMB_INDEX_PINCH: (d_idx, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_MIDDLE_PINCH: (d_mid, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_RING_PINCH: (d_ring, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_PINKY_PINCH: (d_pinky, self.pinch_on_dist, self.pinch_off_dist),
            PoseName.THUMB_PINKY_BASE: (d_pbase, self.pinky_base_on_dist, self.pinky_base_off_dist),
            PoseName.THUMB_INDEX_MIDDLE_PINCH: (d_dual, self.dual_pinch_on_dist, self.dual_pinch_off_dist),
        }
        # Snapshot prior open timestamps & distances BEFORE updating with current frame so
        # a static finger hover can never satisfy its own approach gate without real closure.
        prev_open_ms = dict(st.last_open_ms)
        prev_open_dist = dict(st.last_open_dist)
        for pose_key, (dist_val, on_d, off_d) in pose_dists.items():
            if dist_val >= (on_d + self.approach_open_margin):
                prev_t = st.last_open_ms.get(pose_key)
                prev_d = st.last_open_dist.get(pose_key, 0.0)
                if prev_t is None or (timestamp_ms - prev_t) > self.approach_window_ms or dist_val >= prev_d:
                    st.last_open_dist[pose_key] = dist_val
                else:
                    # Keep recent peak separation within window, decaying slowly toward current open dist
                    st.last_open_dist[pose_key] = max(dist_val, 0.92 * prev_d + 0.08 * dist_val)
                st.last_open_ms[pose_key] = timestamp_ms
            if dist_val >= off_d:
                st.latched.discard(pose_key)

        # Fist / grip guard: count how many fingertips are tightly or half curled toward palm center
        curled_count = 0
        half_curled_count = 0
        for tip_idx in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
            dist_to_palm = float(np.linalg.norm(pf.coords_normalized[tip_idx]))
            if dist_to_palm < 0.72:
                curled_count += 1
            if dist_to_palm < 1.05:
                half_curled_count += 1
        # If >= 3 fingers are curled tightly into a fist/grip (holding phone/cup), suppress loose pinch
        if curled_count >= 3 and min(d_idx, d_mid) > self.pinch_on_dist:
            st.latched.clear()
            return PinchPrediction(pose=PoseName.IDLE, probabilities=idle_probabilities(), confidence="high")

        # Raw scores in [0, 1]
        s_idx = _smooth_closeness(d_idx, self.pinch_on_dist, self.pinch_off_dist)
        s_mid = _smooth_closeness(d_mid, self.pinch_on_dist, self.pinch_off_dist)
        s_ring = _smooth_closeness(d_ring, self.pinch_on_dist, self.pinch_off_dist)
        s_pinky = _smooth_closeness(d_pinky, self.pinch_on_dist, self.pinch_off_dist)
        s_pbase = _smooth_closeness(d_pbase, self.pinky_base_on_dist, self.pinky_base_off_dist)

        # Dual pinch (Drag): both index AND middle close to thumb, and tips close together
        s_dual = _smooth_closeness(d_dual, self.dual_pinch_on_dist, self.dual_pinch_off_dist)
        if d_idx_mid_tips > 0.55:
            s_dual *= 0.25

        # Disambiguate dual pinch vs single index/middle pinch
        if s_dual >= 0.50:
            s_idx *= max(0.0, 1.0 - s_dual)
            s_mid *= max(0.0, 1.0 - s_dual)

        # Prevent pinky tip pinch from being confused with pinky base slide
        if s_pinky > s_pbase:
            s_pbase *= 0.4
        elif s_pbase > s_pinky and d_pinky > self.pinch_off_dist:
            s_pinky *= 0.3

        scores: Dict[PoseName, float] = {
            PoseName.THUMB_INDEX_PINCH: s_idx,
            PoseName.THUMB_MIDDLE_PINCH: s_mid,
            PoseName.THUMB_RING_PINCH: s_ring,
            PoseName.THUMB_PINKY_PINCH: s_pinky,
            PoseName.THUMB_PINKY_BASE: s_pbase,
            PoseName.THUMB_INDEX_MIDDLE_PINCH: s_dual,
        }

        idx_ext = compute_finger_extension_ratios(pf).get("index", 2.0) if self.enable_posture_gate else 2.0

        # If the leading pose is already latched, preserve the full 3D score distribution so
        # ongoing holds remain strictly rotation-invariant and uninterrupted by wrist motion.
        raw_top_pose = max(scores.items(), key=lambda kv: kv[1])[0]
        if raw_top_pose not in st.latched:
            for pose_key, raw_sc in list(scores.items()):
                if raw_sc <= 0.0 or pose_key in st.latched:
                    continue
                dist_val, on_d, _ = pose_dists[pose_key]
                blocked = False
                if self.enable_approach_gate:
                    last_open = prev_open_ms.get(pose_key)
                    open_dist = prev_open_dist.get(pose_key, 0.0)
                    if (
                        last_open is None
                        or (timestamp_ms - last_open) > self.approach_window_ms
                        or (open_dist - dist_val) < self.approach_open_margin
                    ):
                        blocked = True
                if not blocked and self.enable_motion_gate:
                    if eff_speed > self.max_wrist_speed or eff_ang > self.max_wrist_ang_speed_deg_s:
                        blocked = True
                if not blocked and self.enable_posture_gate:
                    if pose_key in _NON_INDEX_POSTURE_POSES and idx_ext < self.index_ext_min:
                        blocked = True
                if not blocked and self.enable_grip_guard:
                    if half_curled_count >= self.grip_half_curled_min and dist_val > (on_d * self.grip_loose_ratio):
                        blocked = True

                if blocked:
                    scores[pose_key] = 0.0
                elif raw_sc >= self._latch_score_threshold:
                    st.latched.add(pose_key)

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
