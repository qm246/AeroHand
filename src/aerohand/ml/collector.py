"""
Privacy-Preserving Landmarks-Only Pinch Data Collector (v2 — Guided Protocol, Metadata & Quality Gate).
Adapted from Airpoint SDK `PinchDataCollector.ts` (Apache-2.0): NEVER saves video pixels;
records raw 2D+3D landmarks, handedness/hand_id, confidence, session metadata, and pose label.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName

POSE_TO_ID: Dict[PoseName, int] = {pose: i for i, pose in enumerate(ALL_POSE_NAMES)}

# Target frame counts (Section 4.2 of engineering plan)
COLLECTOR_CLASS_TARGETS: Dict[str, Dict[str, int]] = {
    PoseName.IDLE.value: {"total": 8000, "Right": 5000, "Left": 3000},
    PoseName.THUMB_INDEX_PINCH.value: {"total": 3000, "Right": 2000, "Left": 1000},
    PoseName.THUMB_MIDDLE_PINCH.value: {"total": 3000, "Right": 2000, "Left": 1000},
    PoseName.THUMB_RING_PINCH.value: {"total": 3000, "Right": 2000, "Left": 1000},
    PoseName.THUMB_PINKY_PINCH.value: {"total": 3000, "Right": 2000, "Left": 1000},
    PoseName.THUMB_PINKY_BASE.value: {"total": 3000, "Right": 2000, "Left": 1000},
    PoseName.THUMB_INDEX_MIDDLE_PINCH.value: {"total": 3000, "Right": 2000, "Left": 1000},
}

HARD_NEGATIVE_GROUPS: Tuple[str, ...] = (
    "typing",
    "resting",
    "hair_face",
    "cup_phone",
    "waving",
    "fist_open",
    "near_miss",
    "transitions",
)


@dataclass(frozen=True)
class GuidedStep:
    pose: PoseName
    duration_s: float = 5.0
    condition_note: str = ""
    hard_negative_group: str = ""


def default_guided_steps(step_duration_s: float = 5.0) -> List[GuidedStep]:
    """Build a structured 7-class + hard-negative guided recording script."""
    d = max(2.0, float(step_duration_s))
    return [
        GuidedStep(PoseName.IDLE, d, "Resting open/relaxed hand", "resting"),
        GuidedStep(PoseName.IDLE, d, "Near-miss pinch (1-2 cm apart)", "near_miss"),
        GuidedStep(PoseName.IDLE, d, "Typing / fist / natural hand motion", "typing"),
        GuidedStep(PoseName.THUMB_INDEX_PINCH, d, "Thumb + Index pinch (hold & loose)"),
        GuidedStep(PoseName.THUMB_MIDDLE_PINCH, d, "Thumb + Middle pinch (tap & hold)"),
        GuidedStep(PoseName.THUMB_RING_PINCH, d, "Thumb + Ring pinch (still & sliding)"),
        GuidedStep(PoseName.THUMB_PINKY_PINCH, d, "Thumb + Pinky tip pinch"),
        GuidedStep(PoseName.THUMB_PINKY_BASE, d, "Thumb + Pinky base touch"),
        GuidedStep(PoseName.THUMB_INDEX_MIDDLE_PINCH, d, "3-Finger drag pinch (still & moving)"),
    ]


def _slugify(val: str, fallback: str = "x") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "", str(val or "").strip())
    return cleaned[:24] if cleaned else fallback


def trim_segment_edges(samples: Sequence[Dict[str, Any]], trim_ms: float = 300.0) -> List[Dict[str, Any]]:
    """
    Remove transition frames within `trim_ms` of the start AND end of a recorded segment.
    Preserves chronological order and avoids mislabeled transition frames.
    """
    if not samples:
        return []
    if trim_ms <= 0.0:
        return list(samples)
    ts_list = [int(s.get("timestamp_ms", 0)) for s in samples]
    t_start = min(ts_list)
    t_end = max(ts_list)
    if (t_end - t_start) <= 2.0 * trim_ms:
        return []
    lo = t_start + trim_ms
    hi = t_end - trim_ms
    return [s for s in samples if lo <= float(s.get("timestamp_ms", 0)) <= hi]


class PinchDataCollector:
    """Interactive & guided landmarks-only session recorder with metadata, trimming, undo & quality gate."""

    def __init__(
        self,
        session_meta: Optional[Dict[str, Any]] = None,
        *,
        trim_edge_ms: float = 300.0,
        countdown_s: float = 3.0,
        step_duration_s: float = 5.0,
    ) -> None:
        self._samples: List[Dict[str, Any]] = []
        self._segment_buf: List[Dict[str, Any]] = []
        self._recording: bool = False
        self._current_pose: PoseName = PoseName.IDLE
        self._started_at_iso: str = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._trim_edge_ms: float = max(0.0, float(trim_edge_ms))
        self._countdown_s: float = max(0.5, float(countdown_s))
        self._step_duration_s: float = max(1.0, float(step_duration_s))

        self._session_meta: Dict[str, Any] = {
            "subject": "user",
            "hand": "Right",
            "wrist_angle": "0",
            "distance": "mid",
            "lighting": "bright",
            "camera": "default",
        }
        if session_meta:
            self._session_meta.update(session_meta)

        # Guided protocol state
        self._guided_active: bool = False
        self._guided_phase: str = "idle"  # "idle" | "countdown" | "recording" | "done"
        self._guided_steps: List[GuidedStep] = default_guided_steps(self._step_duration_s)
        self._guided_idx: int = 0
        self._guided_phase_start_ms: float = 0.0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_pose(self) -> PoseName:
        return self._current_pose

    @property
    def session_meta(self) -> Dict[str, Any]:
        return dict(self._session_meta)

    def set_session_meta(self, meta: Dict[str, Any]) -> None:
        for k, v in meta.items():
            if v is not None:
                self._session_meta[str(k)] = str(v)

    def toggle_recording(self) -> bool:
        self._recording = not self._recording
        return self._recording

    def set_recording(self, active: bool) -> bool:
        self._recording = bool(active)
        return self._recording

    def set_pose(self, pose: PoseName) -> None:
        self._current_pose = pose

    def clear(self) -> None:
        self._samples.clear()
        self._segment_buf.clear()
        self._started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.stop_guided()

    def undo_last(self, seconds: float = 5.0) -> int:
        """Remove samples recorded within the last `seconds` of the latest timestamp."""
        if not self._samples:
            return 0
        latest_ts = max(int(s.get("timestamp_ms", 0)) for s in self._samples)
        cutoff = latest_ts - int(round(max(0.1, float(seconds)) * 1000.0))
        before = len(self._samples)
        self._samples = [s for s in self._samples if int(s.get("timestamp_ms", 0)) < cutoff]
        return before - len(self._samples)

    # ------------------------------------------------------------------
    # Guided Protocol (countdown -> record segment -> auto-trim +/-300ms)
    # ------------------------------------------------------------------

    def start_guided(
        self,
        steps: Optional[Sequence[GuidedStep]] = None,
        *,
        now_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        self._guided_steps = list(steps) if steps else default_guided_steps(self._step_duration_s)
        self._guided_idx = 0
        self._guided_active = True
        self._guided_phase = "countdown"
        self._guided_phase_start_ms = (time.monotonic() * 1000.0) if now_ms is None else float(now_ms)
        self._segment_buf = []
        self._recording = False
        if self._guided_steps:
            self._current_pose = self._guided_steps[0].pose
        return self.guided_status(now_ms=self._guided_phase_start_ms)

    def stop_guided(self) -> None:
        self._guided_active = False
        self._guided_phase = "idle"
        self._segment_buf = []
        self._recording = False

    def tick_guided(self, now_ms: Optional[float] = None) -> Dict[str, Any]:
        if not self._guided_active or self._guided_idx >= len(self._guided_steps):
            return self.guided_status(now_ms=now_ms)
        t = (time.monotonic() * 1000.0) if now_ms is None else float(now_ms)
        step = self._guided_steps[self._guided_idx]
        elapsed_s = (t - self._guided_phase_start_ms) / 1000.0

        if self._guided_phase == "countdown":
            self._recording = False
            self._current_pose = step.pose
            if elapsed_s >= self._countdown_s:
                self._guided_phase = "recording"
                self._guided_phase_start_ms = t
                self._segment_buf = []
                self._recording = True
        elif self._guided_phase == "recording":
            self._current_pose = step.pose
            self._recording = True
            if elapsed_s >= step.duration_s:
                self._recording = False
                trimmed = trim_segment_edges(self._segment_buf, trim_ms=self._trim_edge_ms)
                self._samples.extend(trimmed)
                self._segment_buf = []
                self._guided_idx += 1
                if self._guided_idx >= len(self._guided_steps):
                    self._guided_active = False
                    self._guided_phase = "done"
                else:
                    self._guided_phase = "countdown"
                    self._guided_phase_start_ms = t
                    self._current_pose = self._guided_steps[self._guided_idx].pose
        return self.guided_status(now_ms=t)

    def guided_status(self, now_ms: Optional[float] = None) -> Dict[str, Any]:
        t = (time.monotonic() * 1000.0) if now_ms is None else float(now_ms)
        step = (
            self._guided_steps[self._guided_idx]
            if self._guided_idx < len(self._guided_steps)
            else None
        )
        rem_s = 0.0
        if self._guided_active and step is not None:
            target = self._countdown_s if self._guided_phase == "countdown" else step.duration_s
            rem_s = max(0.0, target - (t - self._guided_phase_start_ms) / 1000.0)
        return {
            "active": self._guided_active,
            "phase": self._guided_phase,
            "step_index": self._guided_idx,
            "total_steps": len(self._guided_steps),
            "pose": step.pose.value if step is not None else None,
            "condition_note": step.condition_note if step is not None else "",
            "remaining_s": round(rem_s, 1),
        }

    # ------------------------------------------------------------------
    # Frame ingestion, counts, quality gate, and persistence
    # ------------------------------------------------------------------

    def add_frame(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        confidence: float,
        timestamp_ms: int,
        *,
        hand_id: Optional[str] = None,
    ) -> None:
        if not self._recording or len(landmarks_2d) < 21:
            return
        flat_2d = [c for lm in landmarks_2d[:21] for c in (lm.x, lm.y, lm.z)]
        flat_3d = (
            [c for lm in world_landmarks[:21] for c in (lm.x, lm.y, lm.z)]
            if (world_landmarks is not None and len(world_landmarks) >= 21)
            else []
        )
        sample: Dict[str, Any] = {
            "timestamp_ms": int(timestamp_ms),
            "label": POSE_TO_ID[self._current_pose],
            "label_name": self._current_pose.value,
            "handedness": handedness,
            "hand_id": str(hand_id or handedness),
            "confidence": round(float(confidence), 4),
            "landmarks_2d": flat_2d,
            "world_landmarks": flat_3d,
        }
        if self._guided_active and self._guided_phase == "recording":
            step = self._guided_steps[self._guided_idx]
            if step.hard_negative_group:
                sample["hard_negative_group"] = step.hard_negative_group
            self._segment_buf.append(sample)
        else:
            self._samples.append(sample)

    def counts(self) -> Dict[str, int]:
        out = {p.value: 0 for p in ALL_POSE_NAMES}
        for s in self._samples:
            name = s["label_name"]
            if name in out:
                out[name] += 1
        return out

    def counts_by_hand(self) -> Dict[str, Dict[str, int]]:
        out = {p.value: {"Right": 0, "Left": 0} for p in ALL_POSE_NAMES}
        for s in self._samples:
            name = s.get("label_name", "")
            h = "Left" if s.get("handedness") == "Left" else "Right"
            if name in out:
                out[name][h] += 1
        return out

    def evaluate_quality(self) -> Dict[str, Any]:
        """
        Pre-save quality gate:
          - Flags mean confidence < 0.58 (`low_confidence`)
          - Flags hands clipped near frame border x/y < 0.03 or > 0.97 (`hand_clipped_edge`)
          - Flags timestamp gaps > 100 ms within the same hand (`time_gaps`)
        """
        n = len(self._samples)
        if n == 0:
            return {
                "ok": True,
                "warnings": [],
                "mean_confidence": 1.0,
                "edge_clipped_count": 0,
                "gap_count": 0,
            }

        confs = [float(s.get("confidence", 1.0)) for s in self._samples]
        mean_conf = sum(confs) / float(n)

        edge_clipped = 0
        gap_count = 0
        last_ts_by_hand: Dict[str, int] = {}
        for s in self._samples:
            lms = s.get("landmarks_2d") or []
            if len(lms) >= 63:
                xs = lms[0:63:3]
                ys = lms[1:63:3]
                if min(xs) < 0.03 or max(xs) > 0.97 or min(ys) < 0.03 or max(ys) > 0.97:
                    edge_clipped += 1
            hid = str(s.get("hand_id") or s.get("handedness") or "Right")
            ts = int(s.get("timestamp_ms", 0))
            if hid in last_ts_by_hand and (ts - last_ts_by_hand[hid]) > 100:
                gap_count += 1
            last_ts_by_hand[hid] = ts

        warnings: List[str] = []
        if mean_conf < 0.58:
            warnings.append("low_confidence")
        if edge_clipped > 0:
            warnings.append("hand_clipped_edge")
        if gap_count > 0:
            warnings.append("time_gaps")

        return {
            "ok": len(warnings) == 0,
            "warnings": warnings,
            "mean_confidence": round(mean_conf, 3),
            "edge_clipped_count": edge_clipped,
            "gap_count": gap_count,
        }

    def save(self, out_dir: str | Path) -> Path:
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        m = self._session_meta
        subj = _slugify(m.get("subject", "user"), "user")
        hand = _slugify(m.get("hand", "Right"), "Right")
        angle = _slugify(m.get("wrist_angle", "0"), "0")
        dist = _slugify(m.get("distance", "mid"), "mid")
        light = _slugify(m.get("lighting", "bright"), "bright")
        out_fp = out_root / f"aerohand_pinch_{subj}_{hand}_{angle}deg_{dist}_{light}_{stamp}.json"
        quality = self.evaluate_quality()
        payload = {
            "schema": "aerohand_pinch_v2",
            "schema_version": 2,
            "collected_at": self._started_at_iso,
            "session_meta": dict(self._session_meta),
            "quality_report": quality,
            "total_samples": len(self._samples),
            "counts": self.counts(),
            "counts_by_hand": self.counts_by_hand(),
            "samples": self._samples,
        }
        out_fp.write_text(json.dumps(payload), encoding="utf-8")
        return out_fp


PoseCollector = PinchDataCollector
