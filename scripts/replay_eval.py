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
import hashlib
import importlib.util
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aerohand.config import load_config
from aerohand.core.engine import HandCursorEngine
from aerohand.core.palm_frame import compute_hand_rotation, extract_frame_diagnostics
from aerohand.core.types import ALL_POSE_NAMES, HandInput
from aerohand.ml.base import PinchClassifier
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier
from aerohand.ml.onnx_classifier import ONNXPinchClassifier
from aerohand.ml.recording_io import (
    POSE_TO_IDX,
    assert_not_heldout,
    collect_json_paths,
    exclude_sealed_content,
    load_recording_frames,
    sha256_file,
)

NUM_CLASSES: int = len(ALL_POSE_NAMES)
ALL_TRIGGER_TYPES: Tuple[str, ...] = (
    "left_click",
    "right_click",
    "drag_start",
    "scroll_stroke",
    "palm_wheel",
    "window_tile",
    "spaces_nav",
    "dictation_start",
)


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
    ft_jsonl_path: Optional[str | Path] = None,
    allow_sealed_aerohand: bool = False,
) -> Dict[str, Any]:
    """
    Deterministically replay all session and negative JSON files through `HandCursorEngine`
    and aggregate classification, Poisson-UCB false-trigger-per-hour (all 8 trigger types),
    per-scenario breakdown, optional per-event FT JSONL diagnostics, and click-anchor stats.

    Exposure definitions (reported separately):
      - `hand_present_idle_hours`: cumulative frame-to-frame time (gaps <= 500 ms) where a hand
        is present and ground-truth label is `idle`.
      - `wall_clock_hours`: cumulative chronological span of all replayed files (including
        no-hand gaps), reported alongside hand-present exposure for transparency.
    """
    base_cfg = deepcopy(config) if config is not None else load_config()
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    ft_by_type: Dict[str, int] = {k: 0 for k in ALL_TRIGGER_TYPES}
    ft_events: List[Dict[str, Any]] = []
    scenario_stats: Dict[str, Dict[str, Any]] = {}
    anchor_offsets_px: List[float] = []
    idle_duration_ms = 0.0
    total_duration_ms = 0.0
    wall_clock_ms = 0.0
    evaluated_files: List[str] = []

    work_items: List[Tuple[Path, bool]] = [(p, False) for p in session_paths] + [
        (p, True) for p in negative_paths
    ]

    for fp, is_neg in work_items:
        assert_not_heldout(fp, allow_sealed_aerohand=allow_sealed_aerohand)
        frames, rec_meta = load_recording_frames(
            fp, force_idle=is_neg, allow_sealed_aerohand=allow_sealed_aerohand
        )
        if not frames:
            continue
        evaluated_files.append(fp.name)
        sc_name = str(rec_meta.get("scenario") or fp.stem)
        file_wall_ms = float(rec_meta.get("wall_clock_ms", 0.0))
        wall_clock_ms += file_wall_ms

        sc = scenario_stats.setdefault(
            sc_name,
            {
                "scenario": sc_name,
                "files": [],
                "idle_duration_ms": 0.0,
                "wall_clock_ms": 0.0,
                "counts": {k: 0 for k in ALL_TRIGGER_TYPES},
            },
        )
        sc["files"].append(fp.name)
        sc["wall_clock_ms"] += file_wall_ms

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
        prev_scrolling = False
        visible_since_by_hand: Dict[str, float] = {}
        prev_wrist_by_hand: Dict[str, Tuple[float, float, float]] = {}
        prev_rot_by_hand: Dict[str, Tuple[float, float, float, float]] = {}

        for fr in frames:
            ts = float(fr["timestamp_ms"])
            if not fr["hand_present"] or not fr["landmarks_2d"]:
                eng.process_frame([], ts, width, height)
                prev_present_ts = None
                prev_scrolling = False
                visible_since_by_hand.clear()
                prev_wrist_by_hand.clear()
                prev_rot_by_hand.clear()
                continue

            if prev_present_ts is not None:
                dt = ts - prev_present_ts
                if 0.0 < dt <= 500.0:
                    total_duration_ms += dt
                    if fr["is_idle_gt"]:
                        idle_duration_ms += dt
                        sc["idle_duration_ms"] += dt
                else:
                    visible_since_by_hand.clear()
                    prev_wrist_by_hand.clear()
                    prev_rot_by_hand.clear()
            else:
                total_duration_ms += 33.33
                if fr["is_idle_gt"]:
                    idle_duration_ms += 33.33
                    sc["idle_duration_ms"] += 33.33
            prev_present_ts = ts

            hand = str(fr["handedness"])
            if hand not in visible_since_by_hand:
                visible_since_by_hand[hand] = ts
            hand_visible_ms = max(0.0, ts - visible_since_by_hand[hand])

            w0 = fr["landmarks_2d"][0]
            wrist_speed = 0.0
            if hand in prev_wrist_by_hand:
                px, py, pts = prev_wrist_by_hand[hand]
                dt_s = (ts - pts) / 1000.0
                if 0.005 <= dt_s <= 0.35:
                    wrist_speed = math.hypot(w0.x - px, w0.y - py) / dt_s
            prev_wrist_by_hand[hand] = (w0.x, w0.y, ts)

            rot = compute_hand_rotation(fr["landmarks_2d"])
            wrist_ang_speed = 0.0
            if rot is not None:
                if hand in prev_rot_by_hand:
                    pp, pyaw, proll, pts = prev_rot_by_hand[hand]
                    dt_s = (ts - pts) / 1000.0
                    if 0.005 <= dt_s <= 0.35:
                        dp = ((rot.pitch - pp + 180.0) % 360.0) - 180.0
                        dy = ((rot.yaw - pyaw + 180.0) % 360.0) - 180.0
                        dr = ((rot.roll - proll + 180.0) % 360.0) - 180.0
                        wrist_ang_speed = math.sqrt(dp * dp + dy * dy + dr * dr) / dt_s
                prev_rot_by_hand[hand] = (rot.pitch, rot.yaw, rot.roll, ts)

            h_in = HandInput(
                handedness=fr["handedness"],
                landmarks=fr["landmarks_2d"],
                world_landmarks=fr["world_landmarks"],
                confidence=fr["confidence"],
                timestamp_ms=int(round(ts)),
            )
            outs = eng.process_frame([h_in], ts, width, height)
            if not outs:
                prev_scrolling = False
                continue
            ho = outs[0]
            y_true_all.append(int(fr["label_idx"]))
            y_pred_all.append(POSE_TO_IDX.get(ho.pose.pose.value, 0))

            if ho.pose.is_clicking and ho.pose.click_x is not None and ho.pose.click_y is not None:
                anchor_offsets_px.append(
                    math.hypot(float(ho.pose.click_x) - ho.cursor.x, float(ho.pose.click_y) - ho.cursor.y)
                )

            fired: List[str] = []
            if fr["is_idle_gt"]:
                p = ho.pose
                if p.is_clicking:
                    fired.append("left_click")
                if p.is_right_clicking:
                    fired.append("right_click")
                if p.grab_just_started:
                    fired.append("drag_start")
                if p.is_scrolling and not prev_scrolling:
                    fired.append("scroll_stroke")
                if p.palm_wheel_action is not None:
                    fired.append("palm_wheel")
                if p.window_tile_direction is not None:
                    fired.append("window_tile")
                if p.spaces_nav_direction is not None:
                    fired.append("spaces_nav")
                if p.dictation_just_started:
                    fired.append("dictation_start")
            prev_scrolling = bool(ho.pose.is_scrolling)

            if fired:
                diag = extract_frame_diagnostics(
                    fr["landmarks_2d"], fr["world_landmarks"], fr["handedness"]
                )
                probs_d = {k.value: round(float(v), 4) for k, v in ho.pose.pose_probabilities.items()}
                for trig in fired:
                    ft_by_type[trig] += 1
                    sc["counts"][trig] += 1
                    ft_events.append(
                        {
                            "scenario": sc_name,
                            "file": fp.name,
                            "timestamp_ms": round(ts, 2),
                            "hand": hand,
                            "trigger_type": trig,
                            "pose_probabilities": probs_d,
                            "thumb_distances": diag["thumb_distances"],
                            "curled_count": diag["curled_count"],
                            "half_curled_count": diag["half_curled_count"],
                            "finger_extensions": diag["finger_extensions"],
                            "wrist_speed": round(wrist_speed, 4),
                            "wrist_ang_speed_deg_s": round(wrist_ang_speed, 2),
                            "dir_z": round(float(ho.cursor.pointing_dir_z), 4),
                            "palm_size": round(float(ho.cursor.palm_size), 5),
                            "palm_facing_dot": diag["palm_facing_dot"],
                            "hand_visible_ms": round(hand_visible_ms, 1),
                        }
                    )

    if ft_jsonl_path is not None:
        jfp = Path(ft_jsonl_path)
        jfp.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(ev, ensure_ascii=False) for ev in ft_events]
        jfp.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

    cls_metrics = compute_classification_metrics(
        np.asarray(y_true_all, dtype=np.int64),
        np.asarray(y_pred_all, dtype=np.int64),
    )

    idle_hours = idle_duration_ms / 3_600_000.0
    wall_hours = wall_clock_ms / 3_600_000.0
    total_ft = sum(ft_by_type.values())

    def _per_hour(cnt: int, exp_h: float = idle_hours) -> float:
        if cnt == 0:
            return 0.0
        if exp_h <= 1e-9:
            return float("inf")
        return round(float(cnt) / exp_h, 3)

    def _cat_stats(cnt: int) -> Dict[str, Any]:
        return {
            "count": int(cnt),
            "exposure_hours": round(idle_hours, 6),
            "wall_clock_hours": round(wall_hours, 6),
            "rate_per_hour": _per_hour(cnt),
            "upper_bound_per_hour": poisson_upper_bound_rate(cnt, idle_hours, ci_level=ci_level),
        }

    per_scenario: List[Dict[str, Any]] = []
    for sc_key in sorted(scenario_stats.keys()):
        entry = scenario_stats[sc_key]
        sc_idle_h = entry["idle_duration_ms"] / 3_600_000.0
        sc_wall_h = entry["wall_clock_ms"] / 3_600_000.0
        sc_total = sum(entry["counts"].values())
        per_scenario.append(
            {
                "scenario": sc_key,
                "files": entry["files"],
                "hand_present_idle_hours": round(sc_idle_h, 6),
                "wall_clock_hours": round(sc_wall_h, 6),
                "total_ft": sc_total,
                "rate_per_hour": _per_hour(sc_total, sc_idle_h),
                "upper_bound_per_hour": poisson_upper_bound_rate(sc_total, sc_idle_h, ci_level=ci_level),
                "counts": dict(entry["counts"]),
            }
        )

    total_ucb = poisson_upper_bound_rate(total_ft, idle_hours, ci_level=ci_level)
    required_for_target = (
        poisson_upper_bound_count(0, ci_level=ci_level) / float(max_ft_per_hour)
        if max_ft_per_hour > 0.0
        else float("inf")
    )
    eff_min_exp = (
        max(float(min_exposure_hours_for_claim), required_for_target)
        if min_exposure_hours_for_claim > 0.0
        else 0.0
    )
    exposure_sufficient = idle_hours >= eff_min_exp
    exposure_notice = (
        "sufficient exposure to certify target"
        if exposure_sufficient
        else f"insufficient exposure to certify target (need >= {eff_min_exp:.2f}h, got {idle_hours:.4f}h)"
    )

    ft_report = {
        # Legacy aliases preserved for backward compatibility
        "clicks": ft_by_type["left_click"],
        "right_clicks": ft_by_type["right_click"],
        "drags": ft_by_type["drag_start"],
        "dictation_starts": ft_by_type["dictation_start"],
        # Complete 8-trigger breakdown (V041-03)
        "by_type": dict(ft_by_type),
        "total": total_ft,
        "ci_level": float(ci_level),
        "idle_duration_s": round(idle_duration_ms / 1000.0, 3),
        "wall_clock_duration_s": round(wall_clock_ms / 1000.0, 3),
        "hand_present_idle_hours": round(idle_hours, 6),
        "wall_clock_hours": round(wall_hours, 6),
        "clicks_per_hour": _per_hour(ft_by_type["left_click"]),
        "right_clicks_per_hour": _per_hour(ft_by_type["right_click"]),
        "drags_per_hour": _per_hour(ft_by_type["drag_start"]),
        "dictation_per_hour": _per_hour(ft_by_type["dictation_start"]),
        "total_ft_per_hour": _per_hour(total_ft),
        "total_upper_bound_per_hour": total_ucb,
        "exposure_sufficient": exposure_sufficient,
        "exposure_notice": exposure_notice,
        "categories": {
            **{k: _cat_stats(ft_by_type[k]) for k in ALL_TRIGGER_TYPES},
            "clicks": _cat_stats(ft_by_type["left_click"]),
            "drags": _cat_stats(ft_by_type["drag_start"]),
            "dictation_starts": _cat_stats(ft_by_type["dictation_start"]),
            "total": _cat_stats(total_ft),
        },
        "per_scenario": per_scenario,
        "ft_events_count": len(ft_events),
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


def _load_seal_heldout_module():
    fp = Path(__file__).resolve().parent / "seal_heldout.py"
    spec = importlib.util.spec_from_file_location("_aerohand_seal_heldout", fp)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {fp}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_eval_model_hash(
    backend_name: str,
    cfg: Dict[str, Any],
    *,
    model_path: Optional[str | Path] = None,
    normalizer_path: Optional[str | Path] = None,
    heuristic_source_bytes: Optional[bytes] = None,
) -> str:
    """
    Compute a deterministic 24-hex SHA-256 fingerprint identifying the evaluated model/config (V051-02).
    For `heuristic_3d`, hashes BOTH `cfg["classifier"]` AND the `heuristic_3d.py` source file so
    code edits produce a distinct hash even when YAML thresholds are unchanged.
    """
    h = hashlib.sha256()
    h.update(str(backend_name).encode("utf-8"))
    if backend_name == "onnx_tcn" and model_path and normalizer_path:
        for p in (Path(model_path), Path(normalizer_path)):
            h.update(p.read_bytes())
    else:
        cls_blob = json.dumps(cfg.get("classifier") or {}, sort_keys=True, ensure_ascii=False)
        h.update(cls_blob.encode("utf-8"))
        if heuristic_source_bytes is not None:
            h.update(heuristic_source_bytes)
        else:
            import inspect

            src_file = inspect.getsourcefile(Heuristic3DPinchClassifier)
            if src_file and Path(src_file).is_file():
                h.update(Path(src_file).read_bytes())
    return h.hexdigest()[:24]


def evaluate_sealed_heldout(
    heldout_dir: str | Path,
    *,
    config: Optional[Dict[str, Any]] = None,
    classifier: Optional[PinchClassifier] = None,
    backend_name: str = "heuristic_3d",
    model_path: Optional[str | Path] = None,
    normalizer_path: Optional[str | Path] = None,
    model_hash: Optional[str] = None,
    force_idle: bool = False,
    max_ft_per_hour: float = 2.0,
    ci_level: float = 0.95,
    min_exposure_hours_for_claim: float = 1.5,
    baseline: Optional[Dict[str, Any]] = None,
    max_f1_regression: float = 0.01,
    ft_jsonl_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Single-touch evaluation on AeroHand's own sealed `heldout_aerohand` directory (V050-02 / V051-02):
      1. Validates 24-hex `model_hash`.
      2. Verifies SHA-256 of every file listed in `sealed_manifest.json`.
      3. Refuses a second evaluation for the same `model_hash` or exceeding `max_touches`.
      4. Runs `evaluate_recordings(..., allow_sealed_aerohand=True)`.
      5. Increments `touched` and records `model_hash` in `sealed_manifest.json`.
    """
    seal_mod = _load_seal_heldout_module()
    cfg = deepcopy(config) if config is not None else load_config()
    eff_hash = seal_mod.validate_model_hash(
        model_hash
        if model_hash is not None
        else compute_eval_model_hash(
            backend_name, cfg, model_path=model_path, normalizer_path=normalizer_path
        )
    )
    manifest, verified_files = seal_mod.verify_sealed_heldout_manifest(heldout_dir)
    seal_mod.check_model_not_evaluated(manifest, eff_hash)

    report = evaluate_recordings(
        session_paths=() if force_idle else verified_files,
        negative_paths=verified_files if force_idle else (),
        config=cfg,
        classifier=classifier,
        backend_name=backend_name,
        max_ft_per_hour=max_ft_per_hour,
        ci_level=ci_level,
        min_exposure_hours_for_claim=min_exposure_hours_for_claim,
        baseline=baseline,
        max_f1_regression=max_f1_regression,
        ft_jsonl_path=ft_jsonl_path,
        allow_sealed_aerohand=True,
    )

    updated_manifest = seal_mod.record_heldout_touch(
        heldout_dir,
        eff_hash,
        summary={
            "backend": backend_name,
            "total_ft": report["false_triggers"]["total"],
            "hand_present_idle_hours": report["false_triggers"]["hand_present_idle_hours"],
            "total_upper_bound_per_hour": report["false_triggers"]["total_upper_bound_per_hour"],
            "macro_f1": report["classification"]["macro_f1"],
            "passed": report["passed"],
        },
    )
    model_sha256 = (
        sha256_file(model_path) if (backend_name == "onnx_tcn" and model_path and Path(model_path).is_file()) else None
    )
    report["model_hash"] = eff_hash
    report["model_sha256"] = model_sha256
    report["heldout_audit"] = {
        "heldout_dir": str(Path(heldout_dir)),
        "model_hash": eff_hash,
        "model_sha256": model_sha256,
        "touched": updated_manifest["touched"],
        "max_touches": updated_manifest.get("max_touches", 3),
        "files_verified": len(verified_files),
    }
    return report


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
        description="AeroHand deterministic offline replay & evaluation harness (V040-01 / V050-02)."
    )
    parser.add_argument("--sessions", nargs="*", default=[], help="Session JSON files or directories")
    parser.add_argument("--negatives", nargs="*", default=[], help="Hard-negative JSON files or directories (forced idle)")
    parser.add_argument(
        "--heldout",
        type=str,
        default=None,
        help="Path to AeroHand sealed heldout_aerohand directory (verifies SHA-256 & single-touch manifest)",
    )
    parser.add_argument("--model-hash", type=str, default=None, help="Explicit model fingerprint for --heldout audit")
    parser.add_argument(
        "--heldout-force-idle",
        action="store_true",
        help="Treat all files in --heldout as ground-truth idle negatives",
    )
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
    parser.add_argument("--ft-jsonl", type=str, default=None, help="Optional JSONL path to write per-event false-trigger diagnostics")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write JSON evaluation report")
    args = parser.parse_args(argv)

    cfg = load_config(config_path=args.config) if args.config else load_config()
    baseline_data = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    clf = _build_cli_classifier(args, cfg)

    if args.heldout:
        if args.sessions or args.negatives:
            raise SystemExit("--heldout cannot be combined with --sessions or --negatives.")
        report = evaluate_sealed_heldout(
            args.heldout,
            config=cfg,
            classifier=clf,
            backend_name=args.backend,
            model_path=args.model,
            normalizer_path=args.normalizer,
            model_hash=args.model_hash,
            force_idle=args.heldout_force_idle,
            max_ft_per_hour=args.max_ft_per_hour,
            ci_level=args.ci_level,
            min_exposure_hours_for_claim=args.min_exposure_hours_for_claim,
            baseline=baseline_data,
            max_f1_regression=args.max_f1_regression,
            ft_jsonl_path=args.ft_jsonl,
        )
    else:
        session_paths, excl_s = exclude_sealed_content(_collect_json_paths(args.sessions))
        negative_paths, excl_n = exclude_sealed_content(_collect_json_paths(args.negatives))
        if excl_s or excl_n:
            print(
                f"[WARN] Excluded {len(excl_s) + len(excl_n)} file(s) matching sealed_manifest.json SHA-256.",
                file=sys.stderr,
            )
        if not session_paths and not negative_paths:
            raise SystemExit("No JSON recordings found via --sessions, --negatives, or --heldout.")

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
            ft_jsonl_path=args.ft_jsonl,
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
