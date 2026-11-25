#!/usr/bin/env python3
"""
GestureCtl -> AeroHand Schema v2 Converter (V041-02).

Strictly read-only on local copies under `data/gesturectl_recordings/`; writes converted
`schema_version: 2` files into `data/converted/` ONLY.

Guarantees:
  - Refuses any path containing `heldout` (`assert_not_heldout`).
  - Refuses files without 3D `world_landmarks` (requires 21 landmarks in BOTH 2D and 3D).
  - Flips GestureCtl's mirrored MediaPipe handedness (`Left` <-> `Right`) to physical hand.
  - Labels `s2b_dev_d_set`, `s2b_dev_t_set`, `p_dev_07/08`, `target_acquisition*` as `idle`.
  - For `clicks_pinch_cadence_near_*`: trims `+/-300 ms` around `cue_phase` transitions,
    maps `press` -> `thumb_index_pinch` ONLY when 3D palm-frame thumb-index distance < `pinch_on_dist`
    (drops unclosed press frames and reports exact drop counts), and maps `rest` -> `idle`.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aerohand.core.palm_frame import INDEX_TIP, THUMB_TIP, compute_palm_frame
from aerohand.core.types import ALL_POSE_NAMES, PoseName
from aerohand.ml.collector import POSE_TO_ID
from aerohand.ml.recording_io import (
    assert_not_heldout,
    flatten_landmarks,
    flip_mirrored_handedness,
    to_landmarks,
)

_PINCH_CADENCE_RE = re.compile(r"clicks_pinch_cadence", re.IGNORECASE)


def _thumb_index_3d_dist(lms3d: Sequence[Any], is_left: bool) -> Optional[float]:
    pf = compute_palm_frame(lms3d, anchor="palm_center", is_left_hand=is_left, scale_metric="palm_width")
    if pf is None:
        return None
    return pf.dist(THUMB_TIP, INDEX_TIP)


def _infer_condition_from_name(stem: str, raw: Dict[str, Any]) -> Tuple[str, str, str]:
    low = stem.lower()
    conds = raw.get("conditions") if isinstance(raw.get("conditions"), dict) else {}
    angle = "45" if ("tilted" in low or "tilt" in str(conds.get("tilt", "")).lower()) else "0"
    dist_raw = str(conds.get("distance", "")).lower()
    if "near" in low or "near" in dist_raw:
        dist = "near"
    elif "far" in low or "far" in dist_raw:
        dist = "far"
    else:
        dist = "mid"
    light_raw = str(conds.get("lighting", "")).lower()
    if "dim" in low or "low_light" in low or "dim" in light_raw:
        light = "dim"
    elif "backlit" in low or "backlit" in light_raw:
        light = "backlit"
    else:
        light = "bright"
    return angle, dist, light


def convert_gesturectl_file(
    src_path: str | Path,
    out_dir: str | Path,
    *,
    subject: str = "minh",
    pinch_on_dist: float = 0.34,
    trim_ms: float = 300.0,
    force_idle: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Convert a single GestureCtl recording JSON into an AeroHand `schema_version: 2` JSON file.
    Raises `ValueError` on `heldout` paths or recordings lacking 3D `world_landmarks`.
    """
    src = Path(src_path)
    out_root = Path(out_dir)
    assert_not_heldout(src)
    assert_not_heldout(out_root)

    raw = json.loads(src.read_text(encoding="utf-8"))
    raw_frames = raw.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError(f"Unsupported or empty GestureCtl recording (missing `frames[]`): {src}")

    is_cadence_pinch = (not force_idle) and bool(_PINCH_CADENCE_RE.search(src.stem))
    if force_idle is True:
        is_cadence_pinch = False

    # Inspect timestamps & wall clock across ALL frames (including no-hand frames)
    all_ts = [float(fr["timestamp_ms"]) for fr in raw_frames if isinstance(fr, dict) and "timestamp_ms" in fr]
    wall_clock_s = round(max(0.0, (max(all_ts) - min(all_ts)) / 1000.0), 3) if len(all_ts) >= 2 else 0.0
    fps_est = float(raw.get("fps_est") or raw.get("fps") or raw.get("target_fps") or 30.0)

    # Identify cue transition timestamps when converting cadence pinch sessions
    transition_ts: List[float] = []
    if is_cadence_pinch:
        prev_cue: Optional[str] = None
        for fr in raw_frames:
            if not isinstance(fr, dict):
                continue
            cue = str(fr.get("cue_phase") or "").strip().lower()
            ts = float(fr.get("timestamp_ms", 0.0))
            if prev_cue is not None and cue != prev_cue:
                transition_ts.append(ts)
            prev_cue = cue

    samples: List[Dict[str, Any]] = []
    stats = {
        "source_file": str(src),
        "raw_frames": len(raw_frames),
        "hand_present_frames": 0,
        "world_present_frames": 0,
        "dropped_no_hand": 0,
        "dropped_no_world": 0,
        "dropped_edge_trim": 0,
        "dropped_unclosed_press": 0,
        "kept_samples": 0,
    }

    for fr in raw_frames:
        if not isinstance(fr, dict):
            continue
        ts_ms = int(round(float(fr.get("timestamp_ms", 0.0))))
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
        if len(lms2d) < 21:
            stats["dropped_no_hand"] += 1
            continue
        stats["hand_present_frames"] += 1

        lms3d = to_landmarks(raw_3d)
        if len(lms3d) < 21:
            stats["dropped_no_world"] += 1
            continue
        stats["world_present_frames"] += 1

        handed = flip_mirrored_handedness(handed_raw)
        pose = PoseName.IDLE

        if is_cadence_pinch:
            if trim_ms > 0.0 and any(abs(ts_ms - t_tr) <= trim_ms for t_tr in transition_ts):
                stats["dropped_edge_trim"] += 1
                continue
            cue = str(fr.get("cue_phase") or "").strip().lower()
            if cue == "press":
                dist = _thumb_index_3d_dist(lms3d, is_left=(handed == "Left"))
                if dist is None or dist >= float(pinch_on_dist):
                    stats["dropped_unclosed_press"] += 1
                    continue
                pose = PoseName.THUMB_INDEX_PINCH
            elif cue == "rest":
                pose = PoseName.IDLE
            else:
                stats["dropped_edge_trim"] += 1
                continue

        samples.append(
            {
                "timestamp_ms": ts_ms,
                "label": POSE_TO_ID[pose],
                "label_name": pose.value,
                "handedness": handed,
                "hand_id": handed,
                "confidence": round(conf, 4),
                "landmarks_2d": flatten_landmarks(lms2d),
                "world_landmarks": flatten_landmarks(lms3d),
                "scenario": src.stem,
            }
        )

    if stats["world_present_frames"] == 0:
        raise ValueError(f"REFUSED: Recording lacks 3D `world_landmarks`: {src}")

    stats["kept_samples"] = len(samples)
    angle, dist, light = _infer_condition_from_name(src.stem, raw)
    counts = {p.value: 0 for p in ALL_POSE_NAMES}
    by_hand = {p.value: {"Right": 0, "Left": 0} for p in ALL_POSE_NAMES}
    for s in samples:
        counts[s["label_name"]] += 1
        by_hand[s["label_name"]][s["handedness"]] += 1

    meta = {
        "source": "gesturectl",
        "subject": str(subject),
        "scenario": src.stem,
        "condition_id": f"gctl_{src.stem}_{angle}deg_{dist}_{light}",
        "wrist_angle": angle,
        "distance": dist,
        "lighting": light,
        "fps": round(fps_est, 2),
        "has_world": True,
        "handedness_flipped": True,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    sub_bucket = "positives" if is_cadence_pinch else "negatives"
    target_dir = out_root / sub_bucket
    target_dir.mkdir(parents=True, exist_ok=True)
    out_fp = target_dir / f"gctl_{src.stem}.json"

    payload = {
        "schema": "aerohand_pinch_v2",
        "schema_version": 2,
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_clock_duration_s": wall_clock_s,
        "session_meta": meta,
        "conversion_report": stats,
        "total_samples": len(samples),
        "counts": counts,
        "counts_by_hand": by_hand,
        "samples": samples,
    }
    out_fp.write_text(json.dumps(payload), encoding="utf-8")
    return {"output_path": str(out_fp), "counts": counts, **stats}


def discover_default_sources(base_dir: Path) -> List[Path]:
    """Select the usable 3D GestureCtl dev recordings (never heldout, never 2D-only)."""
    assert_not_heldout(base_dir)
    candidates: List[Path] = []
    for rel_dir in ("m01_v2/s2b_dev_d_set", "m01_v2/s2b_dev_t_set", "dev2_batch1"):
        d = base_dir / rel_dir
        if d.is_dir():
            for fp in sorted(d.glob("*.json")):
                if fp.name in ("session_manifest.json", "manifest.json"):
                    continue
                assert_not_heldout(fp)
                candidates.append(fp)
    for rel_file in (
        "m01_v2/s2b_dev_p_set/p_dev_07_pause_fist.json",
        "m01_v2/s2b_dev_p_set/p_dev_08_pointer_track.json",
    ):
        fp = base_dir / rel_file
        if fp.is_file():
            assert_not_heldout(fp)
            candidates.append(fp)
    return candidates


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert local GestureCtl 3D recordings to AeroHand schema v2 (V041-02)."
    )
    parser.add_argument(
        "--src",
        nargs="*",
        default=[],
        help="Input JSON files or directories (defaults to usable 3D dev sets under data/gesturectl_recordings)",
    )
    parser.add_argument("--out-dir", type=str, default="data/converted", help="Output directory for converted v2 JSONs")
    parser.add_argument("--subject", type=str, default="minh", help="Subject ID metadata")
    parser.add_argument("--pinch-on-dist", type=float, default=0.34, help="3D palm_frame threshold for cleaning cadence press labels")
    parser.add_argument("--trim-ms", type=float, default=300.0, help="Edge trim window around cue transitions (ms)")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else (repo_root / args.out_dir)
    assert_not_heldout(out_dir)

    if args.src:
        files: List[Path] = []
        for s in args.src:
            p = Path(s)
            assert_not_heldout(p)
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                for fp in sorted(p.rglob("*.json")):
                    if fp.name in ("session_manifest.json", "manifest.json"):
                        continue
                    assert_not_heldout(fp)
                    files.append(fp)
    else:
        files = discover_default_sources(repo_root / "data" / "gesturectl_recordings")

    if not files:
        raise SystemExit("No usable GestureCtl 3D recordings found.")

    results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for fp in files:
        try:
            res = convert_gesturectl_file(
                fp,
                out_dir,
                subject=args.subject,
                pinch_on_dist=args.pinch_on_dist,
                trim_ms=args.trim_ms,
            )
            results.append(res)
        except ValueError as exc:
            skipped.append({"file": str(fp), "reason": str(exc)})

    summary = {
        "converted_files": len(results),
        "skipped_files": len(skipped),
        "total_kept_samples": sum(r["kept_samples"] for r in results),
        "total_dropped_unclosed_press": sum(r["dropped_unclosed_press"] for r in results),
        "total_dropped_edge_trim": sum(r["dropped_edge_trim"] for r in results),
        "results": results,
        "skipped": skipped,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
