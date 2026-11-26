"""
Privacy-Preserving Landmarks-Only Pinch Data Collector (v2 — Guided Protocol, Metadata & Quality Gate).
Adapted from Airpoint SDK `PinchDataCollector.ts` (Apache-2.0): NEVER saves video pixels;
records raw 2D+3D landmarks, handedness/hand_id, confidence, session metadata, and pose label.
"""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName

logger = logging.getLogger(__name__)

POSE_TO_ID: Dict[PoseName, int] = {pose: i for i, pose in enumerate(ALL_POSE_NAMES)}

# Target new AeroHand frame counts (V050-02: ~31.6k 3D negatives already exist from GestureCtl,
# so only 3,000 new AeroHand idle frames + 18,000 positive frames = 21,000 new frames are needed)
COLLECTOR_CLASS_TARGETS: Dict[str, Dict[str, int]] = {
    PoseName.IDLE.value: {"total": 3000, "Right": 2000, "Left": 1000},
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

WRIST_ANGLE_BUCKETS: Tuple[str, ...] = ("0", "30", "45", "60")
DISTANCE_BUCKETS: Tuple[str, ...] = ("near", "mid", "far")
LIGHTING_BUCKETS: Tuple[str, ...] = ("bright", "backlit", "dim")
HARD_NEGATIVE_MIN_PER_GROUP: int = 500
MIN_ANGLE_SHARE: float = 0.15
CORPUS_TOTAL_TARGET: int = sum(v["total"] for v in COLLECTOR_CLASS_TARGETS.values())  # 21000
DEFAULT_COLLECTOR_SUBJECT: str = "minh"
DEFAULT_CUE_WINDOW_MS: int = 2500
DEFAULT_CUE_GAP_MS: int = 1200
MIN_CUE_GAP_MS: int = 1000


@dataclass(frozen=True)
class CuedTriggerSpec:
    """Canonical pose-to-trigger specification for cued event recall evaluation (V060-01a)."""

    trigger: str
    pose: PoseName
    instruction_en: str
    instruction_vi: str


CUED_TRIGGER_SPECS: Dict[str, CuedTriggerSpec] = {
    "left_click": CuedTriggerSpec(
        trigger="left_click",
        pose=PoseName.THUMB_MIDDLE_PINCH,
        instruction_en="Quick tap Thumb + Middle pinch, then release.",
        instruction_vi="Chạm thả nhanh Cái + Giữa (Click trái).",
    ),
    "right_click": CuedTriggerSpec(
        trigger="right_click",
        pose=PoseName.THUMB_RING_PINCH,
        instruction_en="Pinch Thumb + Ring, keep hand still (< 15 px), then release.",
        instruction_vi="Chụm Cái + Áp út, giữ tay yên (< 15 px) rồi thả (Chuột phải).",
    ),
    "scroll_stroke": CuedTriggerSpec(
        trigger="scroll_stroke",
        pose=PoseName.THUMB_RING_PINCH,
        instruction_en="Pinch Thumb + Ring and slide smoothly (>= 15 px), then release.",
        instruction_vi="Chụm Cái + Áp út rồi trượt tay mượt (>= 15 px) để cuộn, sau đó thả.",
    ),
    "drag_start": CuedTriggerSpec(
        trigger="drag_start",
        pose=PoseName.THUMB_INDEX_MIDDLE_PINCH,
        instruction_en="Pinch Thumb + Index + Middle and hold (> 200 ms) or drag, then release.",
        instruction_vi="Chụm 3 ngón (Cái + Trỏ + Giữa), giữ (> 200 ms) hoặc kéo rồi thả.",
    ),
    "dictation_start": CuedTriggerSpec(
        trigger="dictation_start",
        pose=PoseName.THUMB_INDEX_PINCH,
        instruction_en="Pinch Thumb + Index and hold >= 400 ms, then release.",
        instruction_vi="Chụm Cái + Trỏ và giữ >= 400 ms (Đọc giọng nói), rồi thả.",
    ),
    "window_tile": CuedTriggerSpec(
        trigger="window_tile",
        pose=PoseName.THUMB_PINKY_PINCH,
        instruction_en="Pinch Thumb + Pinky tip, slide >= 20 px along one axis, then RELEASE.",
        instruction_vi="Chụm đầu Cái + Út, vuốt >= 20 px theo 1 trục rồi THẢ TAY.",
    ),
    "spaces_nav": CuedTriggerSpec(
        trigger="spaces_nav",
        pose=PoseName.THUMB_PINKY_BASE,
        instruction_en="Touch Thumb + Pinky base, slide >= 20 px along one axis, then release.",
        instruction_vi="Chạm gốc Cái + Út, vuốt >= 20 px theo 1 trục rồi thả.",
    ),
}

CUED_TRIGGER_ORDER: Tuple[str, ...] = tuple(CUED_TRIGGER_SPECS.keys())


def build_cued_schedule(
    *,
    cues_per_trigger: int = 10,
    hand: str = "Right",
    window_ms: int = DEFAULT_CUE_WINDOW_MS,
    gap_ms: int = DEFAULT_CUE_GAP_MS,
    seed: Optional[int] = None,
    triggers: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Build a randomized cued-event schedule with unique cue IDs and enforced gap >= 1000 ms."""
    n_per = max(1, int(cues_per_trigger))
    win = max(250, int(window_ms))
    gap = int(gap_ms)
    if gap < MIN_CUE_GAP_MS:
        raise ValueError(f"gap_ms must be >= {MIN_CUE_GAP_MS} ms to prevent cue overlap (got {gap})")
    norm_hand = "Left" if str(hand or "Right").strip().lower() == "left" else "Right"
    chosen = list(triggers) if triggers else list(CUED_TRIGGER_ORDER)
    unknown = [t for t in chosen if t not in CUED_TRIGGER_SPECS]
    if unknown:
        raise ValueError(f"Unknown cued trigger(s): {unknown}. Valid: {list(CUED_TRIGGER_ORDER)}")

    items: List[CuedTriggerSpec] = []
    for trig in chosen:
        spec = CUED_TRIGGER_SPECS[trig]
        items.extend([spec] * n_per)

    rng = random.Random(seed)
    rng.shuffle(items)

    schedule: List[Dict[str, Any]] = []
    for idx, spec in enumerate(items, start=1):
        schedule.append(
            {
                "cue_id": f"cue_{idx:04d}",
                "pose": spec.pose.value,
                "expected_trigger": spec.trigger,
                "hand": norm_hand,
                "window_ms": win,
                "gap_ms": gap,
                "instruction_en": spec.instruction_en,
                "instruction_vi": spec.instruction_vi,
            }
        )
    return schedule


class QualityGateError(ValueError):
    """Raised when `PinchDataCollector.save()` blocks a session that fails pre-save quality checks."""

    def __init__(self, warnings: Sequence[str], report: Optional[Dict[str, Any]] = None) -> None:
        self.warnings: List[str] = list(warnings)
        self.report: Dict[str, Any] = dict(report or {})
        super().__init__(f"Session blocked by quality gate: {', '.join(self.warnings)}")


@dataclass(frozen=True)
class GuidedStep:
    pose: PoseName
    duration_s: float = 5.0
    condition_note: str = ""
    hard_negative_group: str = ""


def default_guided_steps(step_duration_s: float = 5.0) -> List[GuidedStep]:
    """Build a structured 7-class + all-8 hard-negative guided recording script (14 steps)."""
    d = max(2.0, float(step_duration_s))
    return [
        GuidedStep(PoseName.IDLE, d, "Resting open/relaxed hand", "resting"),
        GuidedStep(PoseName.IDLE, d, "Near-miss pinch (1-2 cm apart)", "near_miss"),
        GuidedStep(PoseName.IDLE, d, "Typing on keyboard", "typing"),
        GuidedStep(PoseName.IDLE, d, "Touching hair / face / glasses", "hair_face"),
        GuidedStep(PoseName.IDLE, d, "Holding cup / phone", "cup_phone"),
        GuidedStep(PoseName.IDLE, d, "Waving / gesturing casually", "waving"),
        GuidedStep(PoseName.IDLE, d, "Fist open/close & stretching", "fist_open"),
        GuidedStep(PoseName.IDLE, d, "Fast pose-to-pose transitions", "transitions"),
        GuidedStep(PoseName.THUMB_INDEX_PINCH, d, "Thumb + Index pinch (hold & loose)"),
        GuidedStep(PoseName.THUMB_MIDDLE_PINCH, d, "Thumb + Middle pinch (tap & hold)"),
        GuidedStep(PoseName.THUMB_RING_PINCH, d, "Thumb + Ring pinch (still & sliding)"),
        GuidedStep(PoseName.THUMB_PINKY_PINCH, d, "Thumb + Pinky tip pinch"),
        GuidedStep(PoseName.THUMB_PINKY_BASE, d, "Thumb + Pinky base touch"),
        GuidedStep(PoseName.THUMB_INDEX_MIDDLE_PINCH, d, "3-Finger drag pinch (still & moving)"),
    ]


def compute_condition_id(meta: Dict[str, Any]) -> str:
    """Derive a canonical `condition_id` from wrist angle, distance, and lighting metadata."""
    existing = str(meta.get("condition_id") or "").strip()
    if existing:
        return existing
    angle = _slugify(str(meta.get("wrist_angle", "0")), "0")
    dist = _slugify(str(meta.get("distance", "mid")), "mid")
    light = _slugify(str(meta.get("lighting", "bright")), "bright")
    return f"{angle}deg_{dist}_{light}"


def compute_corpus_condition_matrix(
    sessions_dir: Optional[str | Path] = None,
    *,
    negatives_dir: Optional[str | Path] = None,
    live_samples: Optional[Sequence[Dict[str, Any]]] = None,
    live_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute the 26k dataset Class x Condition progress matrix across saved sessions,
    hard-negative clips, and optional in-memory unsaved samples (V040-02).
    """
    by_class: Dict[str, Dict[str, int]] = {
        p.value: {
            "total": 0,
            "Right": 0,
            "Left": 0,
            "target": COLLECTOR_CLASS_TARGETS[p.value]["total"],
            "target_right": COLLECTOR_CLASS_TARGETS[p.value]["Right"],
            "target_left": COLLECTOR_CLASS_TARGETS[p.value]["Left"],
        }
        for p in ALL_POSE_NAMES
    }
    by_class_angle: Dict[str, Dict[str, int]] = {
        p.value: {a: 0 for a in WRIST_ANGLE_BUCKETS} for p in ALL_POSE_NAMES
    }
    by_angle: Dict[str, int] = {a: 0 for a in WRIST_ANGLE_BUCKETS}
    by_distance: Dict[str, int] = {d: 0 for d in DISTANCE_BUCKETS}
    by_lighting: Dict[str, int] = {lt: 0 for lt in LIGHTING_BUCKETS}
    by_hard_neg: Dict[str, int] = {g: 0 for g in HARD_NEGATIVE_GROUPS}
    total_frames = 0

    def _ingest(samples: Sequence[Dict[str, Any]], meta: Dict[str, Any], force_idle: bool = False) -> None:
        nonlocal total_frames
        angle = str(meta.get("wrist_angle", "0")).replace("deg", "").strip()
        if angle not in by_angle:
            angle = "0"
        dist = str(meta.get("distance", "mid")).strip().lower()
        light = str(meta.get("lighting", "bright")).strip().lower()
        for s in samples:
            if not isinstance(s, dict):
                continue
            pname = PoseName.IDLE.value if force_idle else str(s.get("label_name") or s.get("labelName") or "")
            if pname not in by_class:
                continue
            hand = "Left" if str(s.get("handedness", "Right")) == "Left" else "Right"
            by_class[pname]["total"] += 1
            by_class[pname][hand] += 1
            by_class_angle[pname][angle] += 1
            by_angle[angle] += 1
            if dist in by_distance:
                by_distance[dist] += 1
            if light in by_lighting:
                by_lighting[light] += 1
            hng = str(s.get("hard_negative_group") or "").strip()
            if hng in by_hard_neg:
                by_hard_neg[hng] += 1
            total_frames += 1

    skipped_cued = 0
    for folder, force_idle in ((sessions_dir, False), (negatives_dir, True)):
        if folder is None:
            continue
        p = Path(folder)
        if not p.is_dir():
            continue
        for fp in sorted(p.glob("*.json")):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
                if str(data.get("recording_kind") or "").strip().lower() == "cued_events":
                    skipped_cued += 1
                    continue
                meta = data.get("session_meta") if isinstance(data.get("session_meta"), dict) else {}
                samples = data.get("samples")
                if isinstance(samples, list):
                    _ingest(samples, meta, force_idle=force_idle)
            except Exception:
                continue
    if skipped_cued > 0:
        logger.info(f"[COLLECT] Excluded {skipped_cued} cued_events file(s) from corpus condition matrix.")

    if live_samples:
        _ingest(live_samples, live_meta or {}, force_idle=False)

    angle_shares = {
        a: round((by_angle[a] / total_frames), 4) if total_frames > 0 else 0.0
        for a in WRIST_ANGLE_BUCKETS
    }
    angle_balanced = total_frames > 0 and all(share >= MIN_ANGLE_SHARE for share in angle_shares.values())
    hard_neg_ready = all(cnt >= HARD_NEGATIVE_MIN_PER_GROUP for cnt in by_hard_neg.values())
    return {
        "total_frames": total_frames,
        "target_total": CORPUS_TOTAL_TARGET,
        "by_class": by_class,
        "by_class_angle": by_class_angle,
        "by_angle": by_angle,
        "angle_shares": angle_shares,
        "angle_balanced": angle_balanced,
        "by_distance": by_distance,
        "by_lighting": by_lighting,
        "by_hard_negative_group": by_hard_neg,
        "hard_negative_min_per_group": HARD_NEGATIVE_MIN_PER_GROUP,
        "hard_negatives_ready": hard_neg_ready,
    }


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


def clean_recording_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    min_confidence: float = 0.58,
    max_gap_ms: float = 100.0,
    edge_margin: float = 0.03,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Annotate & clean a recorded session instead of discarding it (V041-01):
      - Drops frames with `confidence < min_confidence`.
      - Drops frames with hand landmarks clipped within `edge_margin` of frame borders.
      - Splits samples into separate `segment_id`s whenever a same-hand timestamp gap > `max_gap_ms`.
    """
    cleaned: List[Dict[str, Any]] = []
    dropped_low_conf = 0
    dropped_edge = 0
    gap_splits = 0
    last_ts_by_hand: Dict[str, int] = {}
    seg_id_by_hand: Dict[str, int] = {}
    next_seg_id = 0
    cleaned_counts: Dict[str, int] = {p.value: 0 for p in ALL_POSE_NAMES}

    for raw in samples:
        if not isinstance(raw, dict):
            continue
        conf = float(raw.get("confidence", 1.0))
        if conf < min_confidence:
            dropped_low_conf += 1
            continue
        lms = raw.get("landmarks_2d") or []
        if len(lms) >= 63:
            xs = lms[0:63:3]
            ys = lms[1:63:3]
            if (
                min(xs) < edge_margin
                or max(xs) > (1.0 - edge_margin)
                or min(ys) < edge_margin
                or max(ys) > (1.0 - edge_margin)
            ):
                dropped_edge += 1
                continue

        hid = str(raw.get("hand_id") or raw.get("handedness") or "Right")
        ts = int(raw.get("timestamp_ms", 0))
        if hid not in seg_id_by_hand:
            seg_id_by_hand[hid] = next_seg_id
            next_seg_id += 1
        elif hid in last_ts_by_hand and ((ts - last_ts_by_hand[hid]) > max_gap_ms or ts < last_ts_by_hand[hid]):
            gap_splits += 1
            seg_id_by_hand[hid] = next_seg_id
            next_seg_id += 1
        last_ts_by_hand[hid] = ts

        item = dict(raw)
        item["segment_id"] = seg_id_by_hand[hid]
        cleaned.append(item)
        pname = str(item.get("label_name", ""))
        if pname in cleaned_counts:
            cleaned_counts[pname] += 1

    unique_segments = len({s["segment_id"] for s in cleaned}) if cleaned else 0
    report = {
        "input_count": len(samples),
        "kept_count": len(cleaned),
        "dropped_low_confidence": dropped_low_conf,
        "dropped_edge_clipped": dropped_edge,
        "dropped_total": dropped_low_conf + dropped_edge,
        "gap_splits": gap_splits,
        "segments_count": unique_segments,
        "cleaned_counts": cleaned_counts,
    }
    return cleaned, report


class PinchDataCollector:
    """Interactive & guided landmarks-only session recorder with metadata, trimming, undo & quality gate."""

    def __init__(
        self,
        session_meta: Optional[Dict[str, Any]] = None,
        *,
        trim_edge_ms: float = 300.0,
        countdown_s: float = 3.0,
        step_duration_s: float = 5.0,
        quality_gate_mode: str = "clean",
        default_subject: str = DEFAULT_COLLECTOR_SUBJECT,
    ) -> None:
        self._samples: List[Dict[str, Any]] = []
        self._segment_buf: List[Dict[str, Any]] = []
        self._recording: bool = False
        self._current_pose: PoseName = PoseName.IDLE
        self._started_at_iso: str = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._trim_edge_ms: float = max(0.0, float(trim_edge_ms))
        self._countdown_s: float = max(0.5, float(countdown_s))
        self._step_duration_s: float = max(1.0, float(step_duration_s))
        mode = str(quality_gate_mode or "clean").strip().lower()
        self.quality_gate_mode: str = mode if mode in ("clean", "strict") else "clean"
        self._default_subject: str = str(default_subject or "").strip() or DEFAULT_COLLECTOR_SUBJECT
        self._draft_id: str = time.strftime("%Y%m%d_%H%M%S")
        self._last_draft_path: Optional[Path] = None

        self._session_meta: Dict[str, Any] = {
            "subject": self._default_subject,
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

        # Cued event recording state (V060-01a — continuous capture, no trim/clean, dry_run enforced by runtime)
        self._cued_active: bool = False
        self._cued_phase: str = "idle"  # "idle" | "rest" | "cue" | "done" | "stopped"
        self._cued_schedule: List[Dict[str, Any]] = []
        self._cued_idx: int = 0
        self._cued_phase_start_ms: Optional[float] = None
        self._cued_gap_ms: int = DEFAULT_CUE_GAP_MS
        self._cued_window_ms: int = DEFAULT_CUE_WINDOW_MS
        self._cued_seed: Optional[int] = None
        self._cued_hand: str = "Right"
        self._cued_events: List[Dict[str, Any]] = []
        self._cued_samples: List[Dict[str, Any]] = []
        self._cued_started_at_iso: str = self._started_at_iso
        self._cued_draft_id: str = self._draft_id
        self._cued_last_draft_path: Optional[Path] = None

    @property
    def is_cued_active(self) -> bool:
        return self._cued_active

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
        self._draft_id = time.strftime("%Y%m%d_%H%M%S")
        self.clear_draft()
        self.stop_guided()

    def clear_draft(self) -> None:
        if self._last_draft_path is not None:
            try:
                self._last_draft_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._last_draft_path = None

    def autosave_draft(self, sessions_dir: str | Path) -> Optional[Path]:
        """Persist unsaved samples to `<sessions_dir>/_drafts/` so sessions are never lost."""
        all_samples = list(self._samples) + list(self._segment_buf)
        if not all_samples:
            return None
        drafts_dir = Path(sessions_dir) / "_drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        meta = dict(self._session_meta)
        meta["subject"] = str(meta.get("subject") or "").strip() or self._default_subject
        meta["condition_id"] = compute_condition_id(meta)
        draft_fp = drafts_dir / f"draft_{_slugify(meta['subject'], self._default_subject)}_{self._draft_id}.json"
        cleaned, clean_report = clean_recording_samples(all_samples)
        payload = {
            "schema": "aerohand_pinch_v2",
            "schema_version": 2,
            "is_draft": True,
            "collected_at": self._started_at_iso,
            "session_meta": meta,
            "quality_report": clean_report,
            "total_samples": len(cleaned),
            "counts": clean_report["cleaned_counts"],
            "samples": cleaned,
        }
        tmp_fp = draft_fp.with_suffix(".json.tmp")
        tmp_fp.write_text(json.dumps(payload), encoding="utf-8")
        tmp_fp.replace(draft_fp)
        self._last_draft_path = draft_fp
        return draft_fp

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
    # Cued Event Recording (V060-01a — event-level recall evaluation)
    # ------------------------------------------------------------------

    def start_cued_session(
        self,
        *,
        cues_per_trigger: int = 10,
        hand: str = "Right",
        window_ms: int = DEFAULT_CUE_WINDOW_MS,
        gap_ms: int = DEFAULT_CUE_GAP_MS,
        seed: Optional[int] = None,
        triggers: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Start a continuous cued-event session (`recording_kind='cued_events'`)."""
        schedule = build_cued_schedule(
            cues_per_trigger=cues_per_trigger,
            hand=hand,
            window_ms=window_ms,
            gap_ms=gap_ms,
            seed=seed,
            triggers=triggers,
        )
        self.stop_guided()
        self._recording = False
        self._cued_schedule = schedule
        self._cued_idx = 0
        self._cued_active = True
        self._cued_phase = "rest"
        self._cued_phase_start_ms = None
        self._cued_gap_ms = int(gap_ms)
        self._cued_window_ms = int(window_ms)
        self._cued_seed = seed
        self._cued_hand = "Left" if str(hand or "Right").strip().lower() == "left" else "Right"
        self._cued_events = []
        self._cued_samples = []
        self._cued_started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._cued_draft_id = time.strftime("%Y%m%d_%H%M%S")
        self._cued_last_draft_path = None
        return self.cued_status()

    def stop_cued_session(self) -> Dict[str, Any]:
        """Stop an in-progress cued-event session without discarding recorded samples/cues."""
        self._cued_active = False
        if self._cued_phase not in ("done", "idle"):
            self._cued_phase = "stopped"
        return self.cued_status()

    def tick_cued(self, now_ms: float) -> Dict[str, Any]:
        """Advance cued-event state machine using the exact camera `timestamp_ms` clock."""
        t = float(now_ms)
        if not self._cued_active or self._cued_idx >= len(self._cued_schedule):
            return self.cued_status(now_ms=t)

        if self._cued_phase_start_ms is None:
            self._cued_phase_start_ms = t
            return self.cued_status(now_ms=t)

        elapsed_ms = t - self._cued_phase_start_ms
        item = self._cued_schedule[self._cued_idx]

        if self._cued_phase == "rest":
            if elapsed_ms >= float(self._cued_gap_ms):
                cue_ts = int(round(t))
                win_ms = int(item["window_ms"])
                ev = {
                    "cue_id": item["cue_id"],
                    "pose": item["pose"],
                    "expected_trigger": item["expected_trigger"],
                    "hand": item["hand"],
                    "cue_ts_ms": cue_ts,
                    "window_ms": win_ms,
                    "t_start_ms": cue_ts,
                    "t_end_ms": cue_ts + win_ms,
                }
                self._cued_events.append(ev)
                self._cued_phase = "cue"
                self._cued_phase_start_ms = t
        elif self._cued_phase == "cue":
            if elapsed_ms >= float(item["window_ms"]):
                self._cued_idx += 1
                if self._cued_idx >= len(self._cued_schedule):
                    self._cued_active = False
                    self._cued_phase = "done"
                else:
                    self._cued_phase = "rest"
                    self._cued_phase_start_ms = t

        return self.cued_status(now_ms=t)

    def add_cued_frame(self, hands: Sequence[Any], timestamp_ms: int) -> None:
        """
        Record ALL detected hands continuously (no trim/clean/quality drop, ignoring `record_hand`),
        and emit an explicit `hand_present=False` marker when `hands` is empty so offline replay
        preserves exact lost-hand resets.
        """
        if not self._cued_active:
            return
        active_cue_id = (
            self._cued_events[-1]["cue_id"]
            if (self._cued_phase == "cue" and self._cued_events)
            else None
        )
        added = 0
        for h_inp in hands:
            lms2d = getattr(h_inp, "landmarks", None)
            if lms2d is None or len(lms2d) < 21:
                continue
            lms3d = getattr(h_inp, "world_landmarks", None)
            flat_2d = [c for lm in lms2d[:21] for c in (lm.x, lm.y, lm.z)]
            flat_3d = (
                [c for lm in lms3d[:21] for c in (lm.x, lm.y, lm.z)]
                if (lms3d is not None and len(lms3d) >= 21)
                else []
            )
            handed: Handedness = "Left" if str(getattr(h_inp, "handedness", "Right")) == "Left" else "Right"
            conf = float(getattr(h_inp, "confidence", 1.0))
            sample: Dict[str, Any] = {
                "timestamp_ms": int(timestamp_ms),
                "hand_present": True,
                "label": 0,
                "label_name": PoseName.IDLE.value,
                "handedness": handed,
                "hand_id": handed,
                "confidence": round(conf, 4),
                "landmarks_2d": flat_2d,
                "world_landmarks": flat_3d,
            }
            if active_cue_id is not None:
                sample["cue_id"] = active_cue_id
            self._cued_samples.append(sample)
            added += 1

        if added == 0:
            marker: Dict[str, Any] = {
                "timestamp_ms": int(timestamp_ms),
                "hand_present": False,
                "label": 0,
                "label_name": PoseName.IDLE.value,
                "handedness": "Right",
                "hand_id": "Right",
                "confidence": 0.0,
                "landmarks_2d": [],
                "world_landmarks": [],
            }
            if active_cue_id is not None:
                marker["cue_id"] = active_cue_id
            self._cued_samples.append(marker)

    def _build_cued_payload(self, *, is_draft: bool = False) -> Dict[str, Any]:
        meta = dict(self._session_meta)
        meta["subject"] = str(meta.get("subject") or "").strip() or self._default_subject
        meta["hand"] = self._cued_hand
        meta["condition_id"] = compute_condition_id(meta)
        meta["cued_window_ms"] = int(self._cued_window_ms)
        meta["cued_gap_ms"] = int(self._cued_gap_ms)
        if self._cued_seed is not None:
            meta["cued_seed"] = int(self._cued_seed)
        payload: Dict[str, Any] = {
            "schema": "aerohand_pinch_v2",
            "schema_version": 2,
            "recording_kind": "cued_events",
            "collected_at": self._cued_started_at_iso,
            "session_meta": meta,
            "cue_events": list(self._cued_events),
            "total_cues": len(self._cued_events),
            "total_samples": len(self._cued_samples),
            "samples": list(self._cued_samples),
        }
        if is_draft:
            payload["is_draft"] = True
        return payload

    def autosave_cued_draft(self, recall_dir: str | Path) -> Optional[Path]:
        """Atomically persist unsaved cued-event samples & cues under `<recall_dir>/_drafts/`."""
        if not self._cued_samples and not self._cued_events:
            return None
        drafts_dir = Path(recall_dir) / "_drafts"
        drafts_dir.mkdir(parents=True, exist_ok=True)
        payload = self._build_cued_payload(is_draft=True)
        subj = _slugify(payload["session_meta"]["subject"], self._default_subject)
        draft_fp = drafts_dir / f"cued_draft_{subj}_{self._cued_draft_id}.json"
        tmp_fp = draft_fp.with_suffix(".json.tmp")
        tmp_fp.write_text(json.dumps(payload), encoding="utf-8")
        tmp_fp.replace(draft_fp)
        self._cued_last_draft_path = draft_fp
        return draft_fp

    def save_cued(self, recall_dir: str | Path) -> Path:
        """Save cued-event session (`recording_kind='cued_events'`) to `recall_dir` with zero trimming/cleaning."""
        out_root = Path(recall_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        payload = self._build_cued_payload(is_draft=False)
        meta = payload["session_meta"]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        subj = _slugify(meta.get("subject", self._default_subject), self._default_subject)
        hand = _slugify(meta.get("hand", "Right"), "Right")
        angle = _slugify(meta.get("wrist_angle", "0"), "0")
        dist = _slugify(meta.get("distance", "mid"), "mid")
        light = _slugify(meta.get("lighting", "bright"), "bright")
        out_fp = out_root / f"aerohand_cued_{subj}_{hand}_{angle}deg_{dist}_{light}_{stamp}.json"
        if out_fp.exists():
            out_fp = out_root / (
                f"aerohand_cued_{subj}_{hand}_{angle}deg_{dist}_{light}_{stamp}_{int(time.time() * 1000) % 1000:03d}.json"
            )
        out_fp.write_text(json.dumps(payload), encoding="utf-8")
        if self._cued_last_draft_path is not None:
            try:
                self._cued_last_draft_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._cued_last_draft_path = None
        return out_fp

    def cued_status(self, now_ms: Optional[float] = None) -> Dict[str, Any]:
        item = (
            self._cued_schedule[self._cued_idx]
            if self._cued_idx < len(self._cued_schedule)
            else None
        )
        rem_s = 0.0
        if self._cued_active and item is not None and self._cued_phase_start_ms is not None and now_ms is not None:
            target_ms = float(self._cued_gap_ms if self._cued_phase == "rest" else item["window_ms"])
            rem_s = max(0.0, (target_ms - (float(now_ms) - self._cued_phase_start_ms)) / 1000.0)
        return {
            "active": self._cued_active,
            "phase": self._cued_phase,
            "cue_index": self._cued_idx,
            "cues_issued": len(self._cued_events),
            "total_cues": len(self._cued_schedule),
            "samples": len(self._cued_samples),
            "hand": self._cued_hand,
            "pose": item["pose"] if (self._cued_phase == "cue" and item is not None) else None,
            "expected_trigger": item["expected_trigger"] if (self._cued_phase == "cue" and item is not None) else None,
            "instruction_en": item["instruction_en"] if (self._cued_phase == "cue" and item is not None) else "",
            "instruction_vi": item["instruction_vi"] if (self._cued_phase == "cue" and item is not None) else "",
            "remaining_s": round(rem_s, 1),
            "has_unsaved": len(self._cued_samples) > 0,
            "draft_path": str(self._cued_last_draft_path) if self._cued_last_draft_path else None,
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

    def save(
        self,
        out_dir: str | Path,
        *,
        allow_low_quality: bool = False,
        quality_gate_mode: Optional[str] = None,
    ) -> Path:
        raw_quality = self.evaluate_quality()
        eff_mode = str(quality_gate_mode or self.quality_gate_mode or "clean").strip().lower()
        if eff_mode == "strict" and (not allow_low_quality) and (not raw_quality["ok"]):
            raise QualityGateError(raw_quality["warnings"], raw_quality)

        if eff_mode == "clean":
            saved_samples, clean_report = clean_recording_samples(self._samples)
            quality_report = {**raw_quality, **clean_report, "mode": "clean"}
        else:
            saved_samples = [dict(s) for s in self._samples]
            quality_report = {
                **raw_quality,
                "mode": "strict",
                "kept_count": len(saved_samples),
                "dropped_low_confidence": 0,
                "dropped_edge_clipped": 0,
                "dropped_total": 0,
                "gap_splits": raw_quality.get("gap_count", 0),
                "segments_count": 1 if saved_samples else 0,
            }

        meta = dict(self._session_meta)
        meta["subject"] = str(meta.get("subject") or "").strip() or self._default_subject
        meta["condition_id"] = compute_condition_id(meta)

        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        subj = _slugify(meta.get("subject", self._default_subject), self._default_subject)
        hand = _slugify(meta.get("hand", "Right"), "Right")
        angle = _slugify(meta.get("wrist_angle", "0"), "0")
        dist = _slugify(meta.get("distance", "mid"), "mid")
        light = _slugify(meta.get("lighting", "bright"), "bright")
        out_fp = out_root / f"aerohand_pinch_{subj}_{hand}_{angle}deg_{dist}_{light}_{stamp}.json"
        if out_fp.exists():
            out_fp = out_root / (
                f"aerohand_pinch_{subj}_{hand}_{angle}deg_{dist}_{light}_{stamp}_{int(time.time() * 1000) % 1000:03d}.json"
            )

        saved_counts = {p.value: 0 for p in ALL_POSE_NAMES}
        saved_by_hand = {p.value: {"Right": 0, "Left": 0} for p in ALL_POSE_NAMES}
        for s in saved_samples:
            pname = str(s.get("label_name", ""))
            if pname in saved_counts:
                saved_counts[pname] += 1
                h = "Left" if s.get("handedness") == "Left" else "Right"
                saved_by_hand[pname][h] += 1

        payload = {
            "schema": "aerohand_pinch_v2",
            "schema_version": 2,
            "collected_at": self._started_at_iso,
            "session_meta": meta,
            "quality_report": quality_report,
            "total_samples": len(saved_samples),
            "counts": saved_counts,
            "counts_by_hand": saved_by_hand,
            "samples": saved_samples,
        }
        out_fp.write_text(json.dumps(payload), encoding="utf-8")
        self.clear_draft()
        return out_fp


class FrameRingBuffer:
    """
    Bounded per-hand rolling landmark buffer (max 100 frames/hand) for V031-06 hard-negative mining.
    Never stores camera pixels — only 2D/3D landmarks, handedness, confidence, timestamps, and pose probs.
    """

    def __init__(self, max_frames: int = 100) -> None:
        self.max_frames: int = max(10, int(max_frames))
        self._lock = threading.Lock()
        self._buffers: Dict[str, Deque[Dict[str, Any]]] = {
            "Right": deque(maxlen=self.max_frames),
            "Left": deque(maxlen=self.max_frames),
        }
        self._last_trigger: Optional[Dict[str, Any]] = None

    def push(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        confidence: float,
        timestamp_ms: int,
        *,
        hand_id: Optional[str] = None,
        probabilities: Optional[Dict[str, float]] = None,
    ) -> None:
        if len(landmarks_2d) < 21:
            return
        flat_2d = [c for lm in landmarks_2d[:21] for c in (lm.x, lm.y, lm.z)]
        flat_3d = (
            [c for lm in world_landmarks[:21] for c in (lm.x, lm.y, lm.z)]
            if (world_landmarks is not None and len(world_landmarks) >= 21)
            else []
        )
        item: Dict[str, Any] = {
            "timestamp_ms": int(timestamp_ms),
            "label": 0,
            "label_name": PoseName.IDLE.value,
            "handedness": handedness,
            "hand_id": str(hand_id or handedness),
            "confidence": round(float(confidence), 4),
            "landmarks_2d": flat_2d,
            "world_landmarks": flat_3d,
            "hard_negative_group": "false_trigger",
        }
        if probabilities:
            item["trigger_probs"] = {str(k): round(float(v), 4) for k, v in probabilities.items()}
        key = "Left" if handedness == "Left" else "Right"
        with self._lock:
            self._buffers[key].append(item)

    def record_trigger(
        self,
        trigger_type: str,
        hand: str,
        timestamp_ms: int,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = "Left" if hand == "Left" else "Right"
        with self._lock:
            self._last_trigger = {
                "trigger_type": str(trigger_type),
                "hand": key,
                "timestamp_ms": int(timestamp_ms),
                "wall_ts": round(time.time(), 2),
                "meta": dict(meta or {}),
            }

    @property
    def last_trigger(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last_trigger) if self._last_trigger is not None else None

    def len_for_hand(self, hand: str) -> int:
        key = "Left" if hand == "Left" else "Right"
        with self._lock:
            return len(self._buffers[key])

    def total_frames(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._buffers.values())

    def extract_window(
        self,
        *,
        hand: Optional[str] = None,
        center_ts_ms: Optional[int] = None,
        window_s: float = 3.0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Extract chronological `idle`-labeled samples around the last trigger (or latest frames)."""
        half_ms = max(250.0, float(window_s) * 500.0)
        with self._lock:
            trig = dict(self._last_trigger) if self._last_trigger is not None else None
            target_hand = hand or (trig["hand"] if trig else None)
            if target_hand is None:
                target_hand = "Right" if len(self._buffers["Right"]) >= len(self._buffers["Left"]) else "Left"
            key = "Left" if target_hand == "Left" else "Right"
            buf = list(self._buffers[key])
            if not buf:
                other = "Left" if key == "Right" else "Right"
                buf = list(self._buffers[other])
                key = other

        if not buf:
            return [], {"hand": key, "trigger_type": (trig or {}).get("trigger_type", "manual"), "center_ts_ms": 0}

        if center_ts_ms is None:
            center = int(trig["timestamp_ms"]) if trig is not None else int(buf[-1]["timestamp_ms"])
        else:
            center = int(center_ts_ms)

        lo = center - half_ms
        hi = center + half_ms
        selected = [dict(s) for s in buf if lo <= float(s["timestamp_ms"]) <= hi]
        if not selected:
            # Fallback to the most recent frames within window_s of buffer tail
            tail_ts = float(buf[-1]["timestamp_ms"])
            selected = [dict(s) for s in buf if (tail_ts - float(s["timestamp_ms"])) <= (2.0 * half_ms)]

        selected.sort(key=lambda s: int(s["timestamp_ms"]))
        for s in selected:
            s["label"] = 0
            s["label_name"] = PoseName.IDLE.value
            s["hard_negative_group"] = "false_trigger"

        info = {
            "hand": key,
            "trigger_type": (trig or {}).get("trigger_type", "manual"),
            "center_ts_ms": center,
        }
        return selected, info

    def save_negative_snapshot(
        self,
        out_dir: str | Path,
        *,
        window_s: float = 3.0,
        session_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save the +/- window_s/2 landmark clip around the last false trigger as an `idle` v2 JSON."""
        samples, info = self.extract_window(window_s=window_s)
        if not samples:
            return None
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        trig_slug = _slugify(info["trigger_type"], "manual")
        hand_slug = _slugify(info["hand"], "Right")
        meta = dict(session_meta or {})
        meta.setdefault("subject", DEFAULT_COLLECTOR_SUBJECT)
        meta["hand"] = info["hand"]
        meta["source"] = "hard_negative_mine"
        meta["trigger_type"] = info["trigger_type"]
        meta.setdefault("condition_id", f"hardneg_{trig_slug}_{stamp}")

        out_fp = out_root / f"aerohand_neg_{trig_slug}_{hand_slug}_{stamp}.json"
        if out_fp.exists():
            out_fp = out_root / f"aerohand_neg_{trig_slug}_{hand_slug}_{stamp}_{int(time.time() * 1000) % 1000}.json"

        by_hand = {p.value: {"Right": 0, "Left": 0} for p in ALL_POSE_NAMES}
        by_hand[PoseName.IDLE.value][info["hand"]] = len(samples)
        counts = {p.value: (len(samples) if p == PoseName.IDLE else 0) for p in ALL_POSE_NAMES}

        payload = {
            "schema": "aerohand_pinch_v2",
            "schema_version": 2,
            "hard_negative": True,
            "trigger_type": info["trigger_type"],
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "session_meta": meta,
            "total_samples": len(samples),
            "counts": counts,
            "counts_by_hand": by_hand,
            "samples": samples,
        }
        out_fp.write_text(json.dumps(payload), encoding="utf-8")
        return out_fp


PoseCollector = PinchDataCollector
