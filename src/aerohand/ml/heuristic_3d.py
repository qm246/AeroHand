"""
Zero-setup 3D Geometric Pinch Probability Estimator (MVP Brain).

Operates strictly in the rotation-, translation-, and scale-invariant 3D `palm_frame`
computed from MediaPipe's metric `world_landmarks` (falling back to 2D+z if unavailable).
Outputs calibrated soft probabilities over the 7 Airpoint pinch classes so it plugs
directly into `PoseStateMachine` and can be hot-swapped later for a self-trained ONNX TCN.

Anti-false-activation guards built in:
  1. Fist / grip rejection: when >= 3 non-thumb fingertips are curled close to palm/thumb
     simultaneously (drinking, grabbing phone, resting fist), suppresses single-finger
     pinch scores in favor of `idle`.
  2. Margin disambiguation: distinguishes `thumb_index_middle_pinch` (Drag) from
     `thumb_middle_pinch` (Left Click) and suppresses ambiguous ties.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from aerohand.core.palm_frame import (
    INDEX_TIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
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


def _smooth_closeness(dist: float, on_dist: float, off_dist: float) -> float:
    """Map distance in palm-width units to [0, 1] contact score (1 at <= on_dist, 0 at >= off_dist)."""
    if dist <= on_dist:
        return 1.0
    if dist >= off_dist or off_dist <= on_dist:
        return 0.0
    t = (off_dist - dist) / (off_dist - on_dist)
    # Smoothstep for clean transition across Schmitt thresholds
    return t * t * (3.0 - 2.0 * t)


class Heuristic3DPinchClassifier:
    """Rotation-invariant 3D distance estimator producing soft 7-class probabilities."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        c = cfg or {}
        self.pinch_on_dist: float = float(c.get("pinch_on_dist", 0.34))
        self.pinch_off_dist: float = float(c.get("pinch_off_dist", 0.62))
        self.pinky_base_on_dist: float = float(c.get("pinky_base_on_dist", 0.42))
        self.pinky_base_off_dist: float = float(c.get("pinky_base_off_dist", 0.72))
        self.dual_pinch_on_dist: float = float(c.get("dual_pinch_on_dist", 0.38))
        self.dual_pinch_off_dist: float = float(c.get("dual_pinch_off_dist", 0.65))
        self.min_margin: float = float(c.get("min_margin", 0.08))

    def reset(self) -> None:
        return None

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

        # 3D normalized distances from thumb tip (LM4) in palm-width units (LM5-LM17 = 1.0)
        d_idx = pf.dist(THUMB_TIP, INDEX_TIP)
        d_mid = pf.dist(THUMB_TIP, MIDDLE_TIP)
        d_ring = pf.dist(THUMB_TIP, RING_TIP)
        d_pinky = pf.dist(THUMB_TIP, PINKY_TIP)
        d_pbase = pf.dist(THUMB_TIP, PINKY_MCP)
        d_idx_mid_tips = pf.dist(INDEX_TIP, MIDDLE_TIP)

        # Fist / grip guard: count how many fingertips are tightly curled into palm center
        import numpy as np

        curled_count = 0
        for tip_idx in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
            dist_to_palm = float(np.linalg.norm(pf.coords_normalized[tip_idx]))
            if dist_to_palm < 0.72:
                curled_count += 1
        # If >= 3 fingers are curled tightly into a fist/grip (holding phone/cup), suppress pinch
        if curled_count >= 3 and min(d_idx, d_mid) > self.pinch_on_dist * 0.9:
            return PinchPrediction(pose=PoseName.IDLE, probabilities=idle_probabilities(), confidence="high")

        # Raw scores in [0, 1]
        s_idx = _smooth_closeness(d_idx, self.pinch_on_dist, self.pinch_off_dist)
        s_mid = _smooth_closeness(d_mid, self.pinch_on_dist, self.pinch_off_dist)
        s_ring = _smooth_closeness(d_ring, self.pinch_on_dist, self.pinch_off_dist)
        s_pinky = _smooth_closeness(d_pinky, self.pinch_on_dist, self.pinch_off_dist)
        s_pbase = _smooth_closeness(d_pbase, self.pinky_base_on_dist, self.pinky_base_off_dist)

        # Dual pinch (Drag): both index AND middle close to thumb, and tips close together
        d_dual = max(d_idx, d_mid)
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
