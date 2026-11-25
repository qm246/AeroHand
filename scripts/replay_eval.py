#!/usr/bin/env python3
"""
Deterministic Offline Replay & Evaluation Harness for AeroHand (V040-01).

Pure-Python (zero camera or macOS UI APIs — runs cleanly on macOS & Linux CI).
Replays recorded landmark sessions (v1/v2 or SDK v4.0 JSON) and hard-negative clips
chronologically through `HandCursorEngine` with either `heuristic_3d` or `onnx_tcn`.

Reports:
  - Per-class precision, recall, F1, macro-F1, idle FP rate, and confusion matrix
  - False triggers per hour on `idle` segments / hard-negative recordings:
    clicks, right-clicks, drags, dictation starts, and total FT/hr
  - Click-anchor offset statistics (mean / p95 / max px)
  - Optional regression diff against `--baseline metrics.json`

Exits non-zero if `total_ft_per_hour > --max-ft-per-hour` (default 2.0) or if
`macro_f1` regresses by more than `--max-f1-regression` (default 0.01) vs baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aerohand.config import load_config
from aerohand.core.engine import HandCursorEngine
from aerohand.core.types import ALL_POSE_NAMES, Handedness, HandInput, Landmark, PoseName
from aerohand.ml.base import PinchClassifier
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier
from aerohand.ml.onnx_classifier import ONNXPinchClassifier

POSE_TO_IDX: Dict[str, int] = {p.value: i for i, p in enumerate(ALL_POSE_NAMES)}
NUM_CLASSES: int = len(ALL_POSE_NAMES)


def _to_landmarks(raw: Any) -> List[Landmark]:
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
            Landmark(x=float(p["x"]), y=float(p["y"]), z=float(p.get("z", 0.0)))
            for p in raw[:21]
        ]
    return []


def _collect_json_paths(inputs: Sequence[str | Path]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_file() and p.suffix == ".json":
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(p.rglob("*.json")))
    return sorted(set(out))


def load_replay_samples(path: Path, *, force_idle: bool = False) -> List[Dict[str, Any]]:
    """Parse a v1/v2 or Airpoint SDK v4.0 JSON recording into normalized chronological frames."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = raw.get("samples", raw) if isinstance(raw, dict) else raw
    if not isinstance(samples, list):
        return []

    parsed: List[Dict[str, Any]] = []
    fallback_ts = 0.0
    for s in samples:
        if not isinstance(s, dict):
            continue
        raw_2d = s.get("landmarks_2d")
        if raw_2d is None:
            raw_2d = s.get("landmarks")
        lms2d = _to_landmarks(raw_2d)
        if len(lms2d) < 21:
            continue
        raw_3d = s.get("world_landmarks")
        if raw_3d is None:
            raw_3d = s.get("worldLandmarks")
        lms3d = _to_landmarks(raw_3d) or None

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

        ts_raw = s.get("timestamp_ms")
        if ts_raw is None:
            ts_raw = s.get("timestamp")
        if ts_raw is None:
            ts_ms = fallback_ts
            fallback_ts += 33.33
        else:
            ts_ms = float(ts_raw)
            fallback_ts = ts_ms + 33.33

        handed: Handedness = "Left" if str(s.get("handedness", "Right")) == "Left" else "Right"
        conf = float(s.get("confidence", 0.95))
        parsed.append(
            {
                "timestamp_ms": ts_ms,
                "label_name": label_name,
                "label_idx": POSE_TO_IDX[label_name],
                "handedness": handed,
                "confidence": conf,
                "landmarks_2d": lms2d,
                "world_landmarks": lms3d,
                "is_idle_gt": label_name == PoseName.IDLE.value,
            }
        )

    parsed.sort(key=lambda item: item["timestamp_ms"])
    return parsed


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < NUM_CLASSES and 0 <= int(p) < NUM_CLASSES:
            cm[int(t), int(p)] += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    f1s: List[float] = []
    for idx, pose in enumerate(ALL_POSE_NAMES):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        sup = int(cm[idx, :].sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[pose.value] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sup,
        }
        if sup > 0:
            f1s.append(f1)

    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    idle_total = int(cm[0, :].sum())
    idle_fp_rate = float((idle_total - int(cm[0, 0])) / idle_total) if idle_total > 0 else 0.0
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "idle_fp_rate": round(idle_fp_rate, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_recordings(
    *,
    session_paths: Sequence[Path] = (),
    negative_paths: Sequence[Path] = (),
    config: Optional[Dict[str, Any]] = None,
    classifier: Optional[PinchClassifier] = None,
    backend_name: str = "heuristic_3d",
    width: float = 1920.0,
    height: float = 1080.0,
    max_ft_per_hour: float = 2.0,
    baseline: Optional[Dict[str, Any]] = None,
    max_f1_regression: float = 0.01,
) -> Dict[str, Any]:
    """
    Deterministically replay all session and negative JSON files through `HandCursorEngine`
    and aggregate classification, false-trigger-per-hour, and click-anchor telemetry.
    """
    base_cfg = deepcopy(config) if config is not None else load_config()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    ft_counts = {"clicks": 0, "right_clicks": 0, "drags": 0, "dictation_starts": 0}
    anchor_offsets_px: List[float] = []
    idle_duration_ms = 0.0
    total_duration_ms = 0.0
    evaluated_files: List[str] = []

    work_items: List[Tuple[Path, bool]] = [(p, False) for p in session_paths] + [
        (p, True) for p in negative_paths
    ]

    for fp, is_neg in work_items:
        frames = load_replay_samples(fp, force_idle=is_neg)
        if not frames:
            continue
        evaluated_files.append(fp.name)

        clf = classifier
        if clf is None:
            clf = Heuristic3DPinchClassifier((base_cfg.get("classifier") or {}).get("heuristic_3d"))
        else:
            try:
                clf.reset()
            except Exception:
                pass

        eng = HandCursorEngine(deepcopy(base_cfg), classifier=clf)
        eng.anchor_cursor("Right", width * 0.5, height * 0.5)
        eng.anchor_cursor("Left", width * 0.5, height * 0.5)

        prev_ts: Optional[float] = None
        for fr in frames:
            ts = float(fr["timestamp_ms"])
            if prev_ts is not None:
                dt = ts - prev_ts
                if 0.0 < dt <= 500.0:
                    total_duration_ms += dt
                    if fr["is_idle_gt"]:
                        idle_duration_ms += dt
            else:
                # Account for nominal 1-frame duration on single/first frame
                total_duration_ms += 33.33
                if fr["is_idle_gt"]:
                    idle_duration_ms += 33.33
            prev_ts = ts

            h_in = HandInput(
                handedness=fr["handedness"],
                landmarks=fr["landmarks_2d"],
                world_landmarks=fr["world_landmarks"],
                confidence=fr["confidence"],
                timestamp_ms=int(round(ts)),
            )
            outs = eng.process_frame([h_in], ts, width, height)
            if not outs:
                continue
            ho = outs[0]
            y_true_all.append(int(fr["label_idx"]))
            y_pred_all.append(POSE_TO_IDX.get(ho.pose.pose.value, 0))

            if ho.pose.is_clicking and ho.pose.click_x is not None and ho.pose.click_y is not None:
                anchor_offsets_px.append(
                    math.hypot(float(ho.pose.click_x) - ho.cursor.x, float(ho.pose.click_y) - ho.cursor.y)
                )

            if fr["is_idle_gt"]:
                if ho.pose.is_clicking:
                    ft_counts["clicks"] += 1
                if ho.pose.is_right_clicking:
                    ft_counts["right_clicks"] += 1
                if ho.pose.grab_just_started:
                    ft_counts["drags"] += 1
                if ho.pose.dictation_just_started:
                    ft_counts["dictation_starts"] += 1

    cls_metrics = compute_classification_metrics(
        np.asarray(y_true_all, dtype=np.int64),
        np.asarray(y_pred_all, dtype=np.int64),
    )

    idle_hours = idle_duration_ms / 3_600_000.0
    total_ft = sum(ft_counts.values())

    def _per_hour(cnt: int) -> float:
        if cnt == 0:
            return 0.0
        if idle_hours <= 1e-9:
            return float("inf")
        return round(float(cnt) / idle_hours, 3)

    ft_report = {
        **ft_counts,
        "total": total_ft,
        "idle_duration_s": round(idle_duration_ms / 1000.0, 3),
        "clicks_per_hour": _per_hour(ft_counts["clicks"]),
        "right_clicks_per_hour": _per_hour(ft_counts["right_clicks"]),
        "drags_per_hour": _per_hour(ft_counts["drags"]),
        "dictation_per_hour": _per_hour(ft_counts["dictation_starts"]),
        "total_ft_per_hour": _per_hour(total_ft),
    }

    if anchor_offsets_px:
        arr_off = np.asarray(anchor_offsets_px, dtype=np.float64)
        anchor_stats = {
            "anchored_clicks": len(anchor_offsets_px),
            "mean_offset_px": round(float(np.mean(arr_off)), 3),
            "p95_offset_px": round(float(np.percentile(arr_off, 95)), 3),
            "max_offset_px": round(float(np.max(arr_off)), 3),
        }
    else:
        anchor_stats = {"anchored_clicks": 0, "mean_offset_px": 0.0, "p95_offset_px": 0.0, "max_offset_px": 0.0}

    baseline_diff: Optional[Dict[str, Any]] = None
    regressed_f1 = False
    if baseline is not None:
        base_f1 = float(
            (baseline.get("classification") or {}).get("macro_f1", baseline.get("macro_f1", 0.0))
        )
        base_ft = float(
            (baseline.get("false_triggers") or {}).get("total_ft_per_hour", 0.0)
        )
        f1_delta = round(cls_metrics["macro_f1"] - base_f1, 4)
        ft_delta = round(ft_report["total_ft_per_hour"] - base_ft, 3) if math.isfinite(ft_report["total_ft_per_hour"]) else float("inf")
        regressed_f1 = f1_delta < -abs(float(max_f1_regression))
        baseline_diff = {
            "baseline_macro_f1": base_f1,
            "macro_f1_delta": f1_delta,
            "baseline_ft_per_hour": base_ft,
            "ft_per_hour_delta": ft_delta,
            "regressed_macro_f1": regressed_f1,
        }

    ft_ok = ft_report["total_ft_per_hour"] <= float(max_ft_per_hour)
    passed = bool(ft_ok and (not regressed_f1))

    return {
        "backend": backend_name,
        "sessions_evaluated": evaluated_files,
        "total_frames": len(y_true_all),
        "total_duration_s": round(total_duration_ms / 1000.0, 3),
        "classification": cls_metrics,
        "false_triggers": ft_report,
        "click_anchor_stats": anchor_stats,
        "baseline_diff": baseline_diff,
        "max_ft_per_hour": float(max_ft_per_hour),
        "passed": passed,
    }


def _build_cli_classifier(args: argparse.Namespace, cfg: Dict[str, Any]) -> PinchClassifier:
    if args.backend == "onnx_tcn":
        if not args.model or not args.normalizer:
            raise SystemExit("--model and --normalizer are required when --backend onnx_tcn")
        return ONNXPinchClassifier(
            model_path=args.model,
            normalizer_path=args.normalizer,
            sequence_length=int((cfg.get("classifier") or {}).get("sequence_length", 5)),
            feature_set=str((cfg.get("classifier") or {}).get("feature_set", "auto")),
        )
    return Heuristic3DPinchClassifier((cfg.get("classifier") or {}).get("heuristic_3d"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="AeroHand deterministic offline replay & evaluation harness (V040-01)."
    )
    parser.add_argument("--sessions", nargs="*", default=[], help="Session JSON files or directories")
    parser.add_argument("--negatives", nargs="*", default=[], help="Hard-negative JSON files or directories (forced idle)")
    parser.add_argument("--backend", choices=["heuristic_3d", "onnx_tcn"], default="heuristic_3d")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model (for onnx_tcn)")
    parser.add_argument("--normalizer", type=str, default=None, help="Path to normalizer JSON (for onnx_tcn)")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline JSON report for regression diff")
    parser.add_argument("--max-ft-per-hour", type=float, default=2.0, help="Fail if false triggers/hour exceeds this cap")
    parser.add_argument("--max-f1-regression", type=float, default=0.01, help="Fail if macro-F1 drops more than this vs baseline")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write JSON evaluation report")
    args = parser.parse_args(argv)

    session_paths = _collect_json_paths(args.sessions)
    negative_paths = _collect_json_paths(args.negatives)
    if not session_paths and not negative_paths:
        raise SystemExit("No JSON recordings found via --sessions or --negatives.")

    cfg = load_config(config_path=args.config) if args.config else load_config()
    baseline_data = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    clf = _build_cli_classifier(args, cfg)

    report = evaluate_recordings(
        session_paths=session_paths,
        negative_paths=negative_paths,
        config=cfg,
        classifier=clf,
        backend_name=args.backend,
        max_ft_per_hour=args.max_ft_per_hour,
        baseline=baseline_data,
        max_f1_regression=args.max_f1_regression,
    )

    out_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        out_fp = Path(args.out)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(out_text, encoding="utf-8")
    print(out_text)

    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
