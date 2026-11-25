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
from aerohand.core.types import ALL_POSE_NAMES, HandInput
from aerohand.ml.base import PinchClassifier
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier
from aerohand.ml.onnx_classifier import ONNXPinchClassifier
from aerohand.ml.recording_io import (
    POSE_TO_IDX,
    assert_not_heldout,
    collect_json_paths,
    load_recording_frames,
)

NUM_CLASSES: int = len(ALL_POSE_NAMES)


def _collect_json_paths(inputs: Sequence[str | Path]) -> List[Path]:
    return collect_json_paths(inputs)


def load_replay_samples(path: Path, *, force_idle: bool = False) -> List[Dict[str, Any]]:
    """Load any supported recording (v1/v2, SDK v4.0, or GestureCtl) via shared `recording_io`."""
    frames, _ = load_recording_frames(path, force_idle=force_idle)
    return frames


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


def poisson_upper_bound_count(count: int, ci_level: float = 0.95) -> float:
    """
    Exact one-sided Poisson / Garwood upper confidence bound on expected event count `mu`
    such that `P(Poisson(mu) <= count) = 1 - ci_level`.
    For `count == 0` and `ci_level == 0.95`, `mu = -ln(0.05) ~= 2.9957` (Rule of Three: ~3/T).
    """
    k = max(0, int(count))
    alpha = max(1e-9, min(0.5, 1.0 - float(ci_level)))
    if k == 0:
        return -math.log(alpha)

    def _cdf(mu: float) -> float:
        if mu <= 0.0:
            return 1.0
        log_mu = math.log(mu)
        total = 0.0
        for j in range(k + 1):
            total += math.exp(-mu + j * log_mu - math.lgamma(j + 1))
        return total

    lo = float(k)
    hi = max(12.0, float(k) + 8.0 * math.sqrt(float(k) + 1.0) + 30.0)
    while _cdf(hi) > alpha:
        hi *= 2.0
    for _ in range(72):
        mid = 0.5 * (lo + hi)
        if _cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def poisson_upper_bound_rate(count: int, exposure_hours: float, ci_level: float = 0.95) -> float:
    """Exact Poisson upper confidence bound on event rate per hour (`mu_upper / exposure_hours`)."""
    if exposure_hours <= 1e-12:
        return float("inf")
    return round(poisson_upper_bound_count(count, ci_level=ci_level) / float(exposure_hours), 3)


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
    ci_level: float = 0.95,
    min_exposure_hours_for_claim: float = 1.5,
    baseline: Optional[Dict[str, Any]] = None,
    max_f1_regression: float = 0.01,
) -> Dict[str, Any]:
    """
    Deterministically replay all session and negative JSON files through `HandCursorEngine`
    and aggregate classification, Poisson-UCB false-trigger-per-hour, and click-anchor telemetry.

    Exposure definitions (reported separately):
      - `hand_present_idle_hours`: cumulative frame-to-frame time (gaps <= 500 ms) where a hand
        is present and ground-truth label is `idle`.
      - `wall_clock_hours`: cumulative chronological span of all replayed files (including
        no-hand gaps), reported alongside hand-present exposure for transparency.
    """
    base_cfg = deepcopy(config) if config is not None else load_config()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    ft_counts = {"clicks": 0, "right_clicks": 0, "drags": 0, "dictation_starts": 0}
    anchor_offsets_px: List[float] = []
    idle_duration_ms = 0.0
    total_duration_ms = 0.0
    wall_clock_ms = 0.0
    evaluated_files: List[str] = []

    work_items: List[Tuple[Path, bool]] = [(p, False) for p in session_paths] + [
        (p, True) for p in negative_paths
    ]

    for fp, is_neg in work_items:
        assert_not_heldout(fp)
        frames, rec_meta = load_recording_frames(fp, force_idle=is_neg)
        if not frames:
            continue
        evaluated_files.append(fp.name)
        wall_clock_ms += float(rec_meta.get("wall_clock_ms", 0.0))

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

        prev_present_ts: Optional[float] = None
        for fr in frames:
            ts = float(fr["timestamp_ms"])
            if not fr["hand_present"] or not fr["landmarks_2d"]:
                eng.process_frame([], ts, width, height)
                prev_present_ts = None
                continue

            if prev_present_ts is not None:
                dt = ts - prev_present_ts
                if 0.0 < dt <= 500.0:
                    total_duration_ms += dt
                    if fr["is_idle_gt"]:
                        idle_duration_ms += dt
            else:
                total_duration_ms += 33.33
                if fr["is_idle_gt"]:
                    idle_duration_ms += 33.33
            prev_present_ts = ts

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
    wall_hours = wall_clock_ms / 3_600_000.0
    total_ft = sum(ft_counts.values())

    def _per_hour(cnt: int) -> float:
        if cnt == 0:
            return 0.0
        if idle_hours <= 1e-9:
            return float("inf")
        return round(float(cnt) / idle_hours, 3)

    def _cat_stats(cnt: int) -> Dict[str, Any]:
        return {
            "count": int(cnt),
            "exposure_hours": round(idle_hours, 6),
            "wall_clock_hours": round(wall_hours, 6),
            "rate_per_hour": _per_hour(cnt),
            "upper_bound_per_hour": poisson_upper_bound_rate(cnt, idle_hours, ci_level=ci_level),
        }

    total_ucb = poisson_upper_bound_rate(total_ft, idle_hours, ci_level=ci_level)
    required_for_target = (
        poisson_upper_bound_count(0, ci_level=ci_level) / float(max_ft_per_hour)
        if max_ft_per_hour > 0.0
        else float("inf")
    )
    eff_min_exp = max(float(min_exposure_hours_for_claim), required_for_target) if min_exposure_hours_for_claim > 0.0 else 0.0
    exposure_sufficient = idle_hours >= eff_min_exp
    exposure_notice = (
        "sufficient exposure to certify target"
        if exposure_sufficient
        else f"insufficient exposure to certify target (need >= {eff_min_exp:.2f}h, got {idle_hours:.4f}h)"
    )

    ft_report = {
        **ft_counts,
        "total": total_ft,
        "ci_level": float(ci_level),
        "idle_duration_s": round(idle_duration_ms / 1000.0, 3),
        "wall_clock_duration_s": round(wall_clock_ms / 1000.0, 3),
        "hand_present_idle_hours": round(idle_hours, 6),
        "wall_clock_hours": round(wall_hours, 6),
        "clicks_per_hour": _per_hour(ft_counts["clicks"]),
        "right_clicks_per_hour": _per_hour(ft_counts["right_clicks"]),
        "drags_per_hour": _per_hour(ft_counts["drags"]),
        "dictation_per_hour": _per_hour(ft_counts["dictation_starts"]),
        "total_ft_per_hour": _per_hour(total_ft),
        "total_upper_bound_per_hour": total_ucb,
        "exposure_sufficient": exposure_sufficient,
        "exposure_notice": exposure_notice,
        "categories": {
            "clicks": _cat_stats(ft_counts["clicks"]),
            "right_clicks": _cat_stats(ft_counts["right_clicks"]),
            "drags": _cat_stats(ft_counts["drags"]),
            "dictation_starts": _cat_stats(ft_counts["dictation_starts"]),
            "total": _cat_stats(total_ft),
        },
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

    # Gate on UPPER CONFIDENCE BOUND and sufficient exposure (V041-01)
    ft_ok = (total_ucb <= float(max_ft_per_hour)) and exposure_sufficient
    passed = bool(ft_ok and (not regressed_f1))

    return {
        "backend": backend_name,
        "sessions_evaluated": evaluated_files,
        "total_frames": len(y_true_all),
        "total_duration_s": round(total_duration_ms / 1000.0, 3),
        "wall_clock_duration_s": round(wall_clock_ms / 1000.0, 3),
        "classification": cls_metrics,
        "false_triggers": ft_report,
        "click_anchor_stats": anchor_stats,
        "baseline_diff": baseline_diff,
        "max_ft_per_hour": float(max_ft_per_hour),
        "ci_level": float(ci_level),
        "min_exposure_hours_for_claim": float(min_exposure_hours_for_claim),
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
    parser.add_argument("--max-ft-per-hour", type=float, default=2.0, help="Fail if FT upper confidence bound exceeds this cap")
    parser.add_argument("--ci-level", type=float, default=0.95, help="One-sided Poisson confidence level for FT upper bound (default 0.95)")
    parser.add_argument(
        "--min-exposure-hours-for-claim",
        type=float,
        default=1.5,
        help="Minimum idle hand-present hours required to certify target (default 1.5h ~= 3/2.0)",
    )
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
        ci_level=args.ci_level,
        min_exposure_hours_for_claim=args.min_exposure_hours_for_claim,
        baseline=baseline_data,
        max_f1_regression=args.max_f1_regression,
    )
    if not report["false_triggers"]["exposure_sufficient"]:
        print(f"[WARN] {report['false_triggers']['exposure_notice']}", file=sys.stderr)

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
