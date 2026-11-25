#!/usr/bin/env python3
"""
Read-only offline dev benchmark for AeroHand on GestureCtl's Day-1 `D-dev` (32 min daily negatives).

STRICT HYGIENE:
  - Refuses to touch any path containing `heldout` (preserves AM-4 single-touch held-out rule).
  - Measures:
      1. Pure Python engine + Heuristic3DPinchClassifier CPU time (med / p95 / p99 ms/frame).
      2. False activations (destructive vs non-destructive) across all 12 daily-activity scenarios.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from aerohand.config import load_config
from aerohand.core.engine import HandCursorEngine
from aerohand.core.types import Handedness, HandInput, Landmark
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier


def _assert_not_heldout(path: Path) -> None:
    if "heldout" in str(path.resolve()).lower():
        raise ValueError(f"REFUSED: Never run offline tuning/benchmarks on held-out path: {path}")


def _to_landmarks(raw: Any) -> List[Landmark]:
    out: List[Landmark] = []
    for p in raw:
        if isinstance(p, dict):
            out.append(Landmark(float(p["x"]), float(p["y"]), float(p.get("z", 0.0))))
        else:
            out.append(Landmark(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0))
    return out


def benchmark_dir(dev_dir: Path) -> Dict[str, Any]:
    _assert_not_heldout(dev_dir)
    files = sorted(p for p in dev_dir.glob("*.json") if p.name != "session_manifest.json")
    if not files:
        raise FileNotFoundError(f"No recording JSON files found in {dev_dir}")

    cfg = load_config()
    frame_times_ms: List[float] = []
    per_file: List[Dict[str, Any]] = []
    totals = {
        "left_click": 0,
        "right_click": 0,
        "drag_start": 0,
        "spaces_nav": 0,
        "window_tile": 0,
        "palm_wheel": 0,
        "scroll_strokes": 0,
    }
    total_frames = 0
    total_present = 0
    total_dur_s = 0.0

    for fp in files:
        _assert_not_heldout(fp)
        payload = json.loads(fp.read_text(encoding="utf-8"))
        frames = payload.get("frames") or []
        if len(frames) >= 2:
            total_dur_s += (int(frames[-1]["timestamp_ms"]) - int(frames[0]["timestamp_ms"])) / 1000.0

        clf = Heuristic3DPinchClassifier((cfg.get("classifier") or {}).get("heuristic_3d"))
        eng = HandCursorEngine(cfg, classifier=clf)
        counts = {k: 0 for k in totals}
        prev_scrolling = False

        for fr in frames:
            total_frames += 1
            ts_ms = float(fr["timestamp_ms"])
            hand = fr.get("hand")
            inputs: List[HandInput] = []
            if isinstance(hand, dict):
                total_present += 1
                raw_2d = hand.get("landmarks_unfiltered") or hand.get("landmarks")
                raw_3d = hand.get("world_landmarks")
                if raw_2d and len(raw_2d) >= 21:
                    h_label_raw = str(hand.get("handedness", "Right")).strip().lower()
                    # GestureCtl recorded mirrored MediaPipe handedness ("Left" for Minh's right hand)
                    handedness: Handedness = "Right" if h_label_raw in ("left", "right") else "Right"
                    inputs.append(
                        HandInput(
                            handedness=handedness,
                            landmarks=_to_landmarks(raw_2d[:21]),
                            world_landmarks=_to_landmarks(raw_3d[:21]) if (raw_3d and len(raw_3d) >= 21) else None,
                            confidence=float(hand.get("score", 1.0)),
                            timestamp_ms=int(ts_ms),
                        )
                    )

            t0 = time.perf_counter()
            outs = eng.process_frame(inputs, ts_ms, 1920.0, 1080.0)
            frame_times_ms.append((time.perf_counter() - t0) * 1000.0)

            if outs:
                p = outs[0].pose
                if p.is_clicking:
                    counts["left_click"] += 1
                if p.is_right_clicking:
                    counts["right_click"] += 1
                if p.grab_just_started:
                    counts["drag_start"] += 1
                if p.spaces_nav_direction is not None:
                    counts["spaces_nav"] += 1
                if p.window_tile_direction is not None:
                    counts["window_tile"] += 1
                if p.palm_wheel_action is not None:
                    counts["palm_wheel"] += 1
                if p.is_scrolling and not prev_scrolling:
                    counts["scroll_strokes"] += 1
                prev_scrolling = p.is_scrolling
            else:
                prev_scrolling = False

        for k, v in counts.items():
            totals[k] += v
        destr = (
            counts["left_click"]
            + counts["right_click"]
            + counts["drag_start"]
            + counts["spaces_nav"]
            + counts["window_tile"]
            + counts["palm_wheel"]
        )
        per_file.append({"file": fp.name, "destructive": destr, **counts})

    ft_sorted = sorted(frame_times_ms)

    def _q(p: float) -> float:
        idx = min(len(ft_sorted) - 1, max(0, int(round(p * (len(ft_sorted) - 1)))))
        return round(ft_sorted[idx], 4)

    destr_total = (
        totals["left_click"]
        + totals["right_click"]
        + totals["drag_start"]
        + totals["spaces_nav"]
        + totals["window_tile"]
        + totals["palm_wheel"]
    )
    return {
        "dataset": str(dev_dir),
        "files": len(files),
        "duration_min": round(total_dur_s / 60.0, 2),
        "frames_total": total_frames,
        "frames_hand_present": total_present,
        "cpu_ms_per_frame": {
            "median": round(statistics.median(ft_sorted), 4),
            "p95": _q(0.95),
            "p99": _q(0.99),
            "max": round(ft_sorted[-1], 4),
        },
        "destructive_total": destr_total,
        "all_fa_total": destr_total + totals["scroll_strokes"],
        "breakdown": totals,
        "per_file": per_file,
    }


def main() -> None:
    default_dev_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "gesturectl_recordings"
        / "m01_v2"
        / "s2b_dev_d_set"
    )
    parser = argparse.ArgumentParser(description="AeroHand read-only dev replay benchmark")
    parser.add_argument(
        "dev_dir",
        nargs="?",
        default=str(default_dev_dir),
        help="Path to dev recordings directory (never heldout)",
    )
    args = parser.parse_args()
    res = benchmark_dir(Path(args.dev_dir))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
