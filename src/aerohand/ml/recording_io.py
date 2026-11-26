"""
Shared Recording I/O & Held-Out Hygiene Guard for AeroHand (V041-02).

Provides:
  - `assert_not_heldout(path)`: refuses any path whose resolved form contains `heldout`
    (unless `allow_sealed_aerohand=True` is explicitly passed for AeroHand's own sealed
    single-touch held-out evaluation in V050-02).
  - `flip_mirrored_handedness(raw)`: converts GestureCtl's mirrored MediaPipe handedness
    (`"Left"` <-> `"Right"`) into AeroHand's anatomical handedness.
  - `load_recording_frames(path)`: unified loader supporting:
      * AeroHand v1 / v2 & converted v2 (`samples[]` with `landmarks_2d` + `world_landmarks`)
      * Airpoint SDK v4.0 (`samples[]` with `landmarks` + `worldLandmarks` + `labelName`)
      * GestureCtl `m01_v2` (`frames[].hand`) and `dev2_batch1` (`frames[].landmarks`)
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName

logger = logging.getLogger(__name__)

POSE_TO_IDX: Dict[str, int] = {p.value: i for i, p in enumerate(ALL_POSE_NAMES)}
MANIFEST_FILENAMES = frozenset({"manifest.json", "session_manifest.json", "sealed_manifest.json"})


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sealed_hashes(data_root: str | Path = "data") -> Set[str]:
    """Collect all file SHA-256 digests referenced in any `sealed_manifest.json` under `data_root`."""
    root = Path(data_root)
    if not root.exists():
        return set()
    hashes: Set[str] = set()
    for mf in sorted(root.rglob("sealed_manifest.json")):
        try:
            payload = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in payload.get("files") or []:
            digest = str(entry.get("sha256") or "").strip().lower()
            if len(digest) == 64:
                hashes.add(digest)
    return hashes


def exclude_sealed_content(
    paths: Sequence[Path],
    *,
    data_root: str | Path = "data",
    sealed_hashes: Optional[Set[str]] = None,
) -> Tuple[List[Path], List[Path]]:
    """
    Filter out any recording whose content SHA-256 matches a sealed held-out manifest (V051-02).
    Prevents train <-> held-out leakage even if a file was copied or renamed.
    """
    hashes = sealed_hashes if sealed_hashes is not None else load_sealed_hashes(data_root)
    if not hashes:
        return list(paths), []
    kept: List[Path] = []
    excluded: List[Path] = []
    for fp in paths:
        try:
            digest = sha256_file(fp).lower()
        except OSError:
            kept.append(fp)
            continue
        if digest in hashes:
            excluded.append(fp)
            logger.warning(
                f"[HELD-OUT GUARD] Excluding {fp} (SHA-256 {digest[:12]}... matches sealed_manifest.json)."
            )
        else:
            kept.append(fp)
    return kept, excluded


def assert_not_heldout(
    path: str | Path,
    *,
    allow_sealed_aerohand: bool = False,
) -> None:
    """
    Refuse any path containing `heldout` (case-insensitive on resolved path) to enforce
    single-touch held-out hygiene across all offline scripts.
    """
    p = Path(path)
    raw_lower = str(p).lower()
    resolved_lower = str(p.resolve()).lower()
    if "heldout" not in raw_lower and "heldout" not in resolved_lower:
        return
    if allow_sealed_aerohand and "heldout_aerohand" in resolved_lower and "gesturectl" not in resolved_lower:
        return
    raise ValueError(f"REFUSED: Held-out path access is forbidden: {path}")


def flip_mirrored_handedness(raw: Any) -> Handedness:
    """Flip GestureCtl's mirrored MediaPipe handedness (`Left` <-> `Right`) to physical hand."""
    low = str(raw or "Left").strip().lower()
    if low == "right":
        return "Left"
    return "Right"


def to_landmarks(raw: Any) -> List[Landmark]:
    """Parse 21 3D landmarks from flat-63 floats, list-of-21 dicts, or list-of-21 sequences."""
    if not raw or not isinstance(raw, list):
        return []
    first = raw[0]
    if isinstance(first, (int, float)) and len(raw) >= 63:
        return [
            Landmark(x=float(raw[i]), y=float(raw[i + 1]), z=float(raw[i + 2]))
            for i in range(0, 63, 3)
        ]
    if isinstance(first, dict) and len(raw) >= 21:
        return [
            Landmark(
                x=float(p["x"]),
                y=float(p["y"]),
                z=float(p.get("z", 0.0)),
                visibility=float(p.get("visibility", 1.0)),
            )
            for p in raw[:21]
        ]
    if isinstance(first, (list, tuple)) and len(raw) >= 21:
        return [
            Landmark(
                x=float(p[0]),
                y=float(p[1]),
                z=float(p[2]) if len(p) > 2 else 0.0,
            )
            for p in raw[:21]
        ]
    return []


def flatten_landmarks(lms: Sequence[Landmark]) -> List[float]:
    return [c for lm in lms[:21] for c in (lm.x, lm.y, lm.z)]


def collect_json_paths(
    inputs: Sequence[str | Path],
    *,
    allow_sealed_aerohand: bool = False,
    skip_manifests: bool = True,
) -> List[Path]:
    """Resolve JSON files from files/directories while strictly enforcing `assert_not_heldout`."""
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        assert_not_heldout(p, allow_sealed_aerohand=allow_sealed_aerohand)
        if p.is_file() and p.suffix == ".json":
            if skip_manifests and p.name in MANIFEST_FILENAMES:
                continue
            out.append(p)
        elif p.is_dir():
            for fp in sorted(p.rglob("*.json")):
                if skip_manifests and fp.name in MANIFEST_FILENAMES:
                    continue
                if "_drafts" in fp.parts:
                    continue
                assert_not_heldout(fp, allow_sealed_aerohand=allow_sealed_aerohand)
                out.append(fp)
    return sorted(set(out))


def load_recording_frames(
    path: str | Path,
    *,
    force_idle: bool = False,
    allow_sealed_aerohand: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load any supported recording JSON into normalized chronological frame dicts plus metadata.

    Each returned frame dict contains:
      `timestamp_ms`, `label_name`, `label_idx`, `handedness`, `hand_id`, `confidence`,
      `landmarks_2d` (List[Landmark] or None if no-hand frame), `world_landmarks`,
      `is_idle_gt`, `hand_present`, `segment_id`, `scenario`.
    """
    fp = Path(path)
    assert_not_heldout(fp, allow_sealed_aerohand=allow_sealed_aerohand)
    raw = json.loads(fp.read_text(encoding="utf-8"))

    session_meta: Dict[str, Any] = {}
    if isinstance(raw, dict) and isinstance(raw.get("session_meta"), dict):
        session_meta = dict(raw["session_meta"])

    scenario = str(session_meta.get("scenario") or fp.stem)
    source = str(session_meta.get("source") or ("gesturectl" if isinstance(raw, dict) and "frames" in raw else "aerohand"))

    frames_out: List[Dict[str, Any]] = []
    raw_timestamps: List[float] = []

    if isinstance(raw, dict) and isinstance(raw.get("frames"), list):
        # GestureCtl m01_v2 (`frames[].hand`) or dev2_batch1 (`frames[].landmarks`)
        raw_frames = raw["frames"]
        fallback_ts = 0.0
        for fr in raw_frames:
            if not isinstance(fr, dict):
                continue
            ts_raw = fr.get("timestamp_ms")
            if ts_raw is None:
                ts_ms = fallback_ts
                fallback_ts += 33.33
            else:
                ts_ms = float(ts_raw)
                fallback_ts = ts_ms + 33.33
            raw_timestamps.append(ts_ms)

            hand_obj = fr.get("hand")
            if isinstance(hand_obj, dict):
                raw_2d = hand_obj.get("landmarks_unfiltered") or hand_obj.get("landmarks")
                raw_3d = hand_obj.get("world_landmarks")
                handed_raw = hand_obj.get("handedness", "Left")
                conf = float(hand_obj.get("score", 1.0))
            else:
                raw_2d = fr.get("landmarks")
                raw_3d = fr.get("world_landmarks")
                handed_raw = fr.get("handedness", "Left")
                conf = float(fr.get("score", 1.0)) if fr.get("score") is not None else 1.0

            lms2d = to_landmarks(raw_2d)
            lms3d = to_landmarks(raw_3d) or None
            if len(lms2d) < 21:
                frames_out.append(
                    {
                        "timestamp_ms": ts_ms,
                        "label_name": PoseName.IDLE.value,
                        "label_idx": 0,
                        "handedness": "Right",
                        "hand_id": "Right",
                        "confidence": 0.0,
                        "landmarks_2d": None,
                        "world_landmarks": None,
                        "is_idle_gt": True,
                        "hand_present": False,
                        "segment_id": 0,
                        "scenario": scenario,
                    }
                )
                continue

            handed = flip_mirrored_handedness(handed_raw)
            label_name = PoseName.IDLE.value
            if not force_idle:
                cue = str(fr.get("cue_phase") or "").strip().lower()
                if cue == "press":
                    label_name = PoseName.THUMB_INDEX_PINCH.value
            frames_out.append(
                {
                    "timestamp_ms": ts_ms,
                    "label_name": label_name,
                    "label_idx": POSE_TO_IDX.get(label_name, 0),
                    "handedness": handed,
                    "hand_id": handed,
                    "confidence": conf,
                    "landmarks_2d": lms2d,
                    "world_landmarks": lms3d,
                    "is_idle_gt": label_name == PoseName.IDLE.value,
                    "hand_present": True,
                    "segment_id": 0,
                    "scenario": scenario,
                }
            )
    else:
        # AeroHand v1/v2 or Airpoint SDK v4.0 (`samples[]` or raw list)
        samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
        if not isinstance(samples, list):
            samples = []
        fallback_ts = 0.0
        for s in samples:
            if not isinstance(s, dict):
                continue
            ts_raw = s.get("timestamp_ms")
            if ts_raw is None:
                ts_raw = s.get("timestamp")
            if ts_raw is None:
                ts_ms = fallback_ts
                fallback_ts += 33.33
            else:
                ts_ms = float(ts_raw)
                fallback_ts = ts_ms + 33.33
            raw_timestamps.append(ts_ms)

            raw_2d = s.get("landmarks_2d")
            if raw_2d is None:
                raw_2d = s.get("landmarks")
            lms2d = to_landmarks(raw_2d)
            if len(lms2d) < 21:
                continue

            raw_3d = s.get("world_landmarks")
            if raw_3d is None:
                raw_3d = s.get("worldLandmarks")
            lms3d = to_landmarks(raw_3d) or None

            if force_idle:
                label_name = PoseName.IDLE.value
            else:
                raw_lbl = s.get("label_name")
                if raw_lbl is None:
                    raw_lbl = s.get("labelName")
                if raw_lbl is None and s.get("label") == 0:
                    raw_lbl = PoseName.IDLE.value
                label_name = str(raw_lbl).strip() if raw_lbl is not None else PoseName.IDLE.value
                if label_name not in POSE_TO_IDX:
                    continue

            handed: Handedness = "Left" if str(s.get("handedness", "Right")) == "Left" else "Right"
            hand_id = str(s.get("hand_id") or handed)
            conf = float(s.get("confidence", 0.95))
            seg_id = int(s.get("segment_id", 0))
            frames_out.append(
                {
                    "timestamp_ms": ts_ms,
                    "label_name": label_name,
                    "label_idx": POSE_TO_IDX[label_name],
                    "handedness": handed,
                    "hand_id": hand_id,
                    "confidence": conf,
                    "landmarks_2d": lms2d,
                    "world_landmarks": lms3d,
                    "is_idle_gt": label_name == PoseName.IDLE.value,
                    "hand_present": True,
                    "segment_id": seg_id,
                    "scenario": str(s.get("scenario") or scenario),
                }
            )

    frames_out.sort(key=lambda item: item["timestamp_ms"])
    wall_clock_ms = 0.0
    if isinstance(raw, dict) and raw.get("wall_clock_duration_s") is not None:
        wall_clock_ms = max(0.0, float(raw["wall_clock_duration_s"]) * 1000.0)
    elif len(raw_timestamps) >= 2:
        wall_clock_ms = max(0.0, max(raw_timestamps) - min(raw_timestamps))
    elif len(raw_timestamps) == 1:
        wall_clock_ms = 33.33

    meta = {
        "file": fp.name,
        "path": str(fp),
        "scenario": scenario,
        "source": source,
        "session_meta": session_meta,
        "wall_clock_ms": wall_clock_ms,
        "raw_frame_count": len(raw_timestamps),
    }
    return frames_out, meta
