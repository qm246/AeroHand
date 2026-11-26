#!/usr/bin/env python3
"""
Event-Level Cued Recall Evaluator & Paired Heuristic-vs-Veto Comparator (V060-01b).

Replays `recording_kind="cued_events"` files (`data/recall/`) chronologically through
`HandCursorEngine` (including multi-hand simultaneous frames and `hand_present=False`
lost-hand resets) and scores event-level trigger recall per canonical gesture.

Safety & Hygiene:
  - Strictly refuses any held-out path (`assert_not_heldout`) and excludes files whose
    SHA-256 matches any `sealed_manifest.json` (`exclude_sealed_content`).
  - Matches at most ONE `expected_trigger` hit per cue inside `[t_start_ms, t_end_ms]`.
  - Computes 95% Wilson score confidence intervals, activation latency p50/p95, wrong-first
    triggers inside cue windows, and extra triggers outside all cue windows.
  - Paired comparison (`--compare-veto`): evaluates `heuristic_3d` and `veto` on the exact
    same cued files and gates **per trigger**: `heuristic_recall - veto_recall <= 0.02`
    (<= 2 percentage points drop per trigger; at n >= 50 allows at most 1 lost cue).
"""

from __future__ import annotations

import argparse
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
from aerohand.core.types import HandInput
from aerohand.ml.base import PinchClassifier
from aerohand.ml.collector import CUED_TRIGGER_ORDER
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier
from aerohand.ml.recording_io import (
    assert_not_heldout,
    collect_json_paths,
    exclude_sealed_content,
    group_frames_by_timestamp,
    load_recording_frames,
)

Z_95: float = 1.959963984540054


def _load_replay_eval_module():
    fp = Path(__file__).resolve().parent / "replay_eval.py"
    spec = importlib.util.spec_from_file_location("_aerohand_replay_eval_recall", fp)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {fp}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wilson_interval(hits: int, n: int, *, z: float = Z_95) -> Tuple[float, float]:
    """Compute exact two-sided Wilson score confidence interval (default 95%)."""
    if n <= 0:
        return 0.0, 0.0
    p = max(0.0, min(1.0, float(hits) / float(n)))
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / float(n)
    center = (p + z2 / (2.0 * float(n))) / denom
    half = (float(z) / denom) * math.sqrt((p * (1.0 - p) / float(n)) + (z2 / (4.0 * float(n) * float(n))))
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def _replay_single_cued_file(
    fp: Path,
    *,
    config: Dict[str, Any],
    classifier: PinchClassifier,
    width: float = 1920.0,
    height: float = 1080.0,
) -> Dict[str, Any]:
    """Replay one `cued_events` file and match fired engine triggers to cue windows."""
    assert_not_heldout(fp)
    frames, rec_meta = load_recording_frames(fp, force_idle=False)
    if str(rec_meta.get("recording_kind") or "").strip().lower() != "cued_events":
        raise ValueError(f"Recording {fp.name} is not recording_kind='cued_events'.")
    cue_events: List[Dict[str, Any]] = list(rec_meta.get("cue_events") or [])

    try:
        classifier.reset(None)  # type: ignore[call-arg]
    except TypeError:
        try:
            classifier.reset()
        except Exception:
            pass

    eng = HandCursorEngine(deepcopy(config), classifier=classifier)
    eng.anchor_cursor("Right", width * 0.5, height * 0.5)
    eng.anchor_cursor("Left", width * 0.5, height * 0.5)

    prev_scrolling: Dict[str, bool] = {"Right": False, "Left": False}
    fired_events: List[Dict[str, Any]] = []

    for ts, batch in group_frames_by_timestamp(frames):
        hand_inputs: List[HandInput] = []
        for fr in batch:
            if fr["hand_present"] and fr["landmarks_2d"]:
                hand_inputs.append(
                    HandInput(
                        handedness=fr["handedness"],
                        landmarks=fr["landmarks_2d"],
                        world_landmarks=fr["world_landmarks"],
                        confidence=fr["confidence"],
                        timestamp_ms=int(round(ts)),
                    )
                )

        if not hand_inputs:
            eng.process_frame([], ts, width, height)
            prev_scrolling["Right"] = False
            prev_scrolling["Left"] = False
            continue

        outs = eng.process_frame(hand_inputs, ts, width, height)
        present_hands = {ho.hand for ho in outs}
        for h_key in ("Right", "Left"):
            if h_key not in present_hands:
                prev_scrolling[h_key] = False

        for ho in outs:
            p = ho.pose
            h_key = ho.hand
            trigs: List[str] = []
            if p.is_clicking:
                trigs.append("left_click")
            if p.is_right_clicking:
                trigs.append("right_click")
            if p.grab_just_started:
                trigs.append("drag_start")
            if p.is_scrolling and not prev_scrolling.get(h_key, False):
                trigs.append("scroll_stroke")
            if p.palm_wheel_action is not None:
                trigs.append("palm_wheel")
            if p.window_tile_direction is not None:
                trigs.append("window_tile")
            if p.spaces_nav_direction is not None:
                trigs.append("spaces_nav")
            if p.dictation_just_started:
                trigs.append("dictation_start")
            prev_scrolling[h_key] = bool(p.is_scrolling)

            for trig in trigs:
                fired_events.append({"timestamp_ms": ts, "hand": h_key, "trigger_type": trig})

    # Match at most ONE expected hit per cue inside [t_start_ms, t_end_ms]
    cue_results: List[Dict[str, Any]] = []
    used_in_any_window: set[int] = set()

    for ev in cue_events:
        cid = str(ev.get("cue_id") or "")
        expected = str(ev.get("expected_trigger") or "")
        cue_hand = str(ev.get("hand") or "Right")
        cue_ts = float(ev.get("cue_ts_ms", ev.get("t_start_ms", 0.0)))
        t_start = float(ev.get("t_start_ms", cue_ts))
        t_end = float(ev.get("t_end_ms", t_start))

        in_window: List[Tuple[int, Dict[str, Any]]] = []
        for idx, fe in enumerate(fired_events):
            fts = float(fe["timestamp_ms"])
            hand_ok = (cue_hand.lower() == "both") or (fe["hand"] == cue_hand)
            if hand_ok and (t_start <= fts <= t_end):
                in_window.append((idx, fe))
                used_in_any_window.add(idx)

        hit_ts: Optional[float] = None
        wrong_first: Optional[str] = None
        wrong_triggers_in_window: List[str] = []
        for _idx, fe in in_window:
            if fe["trigger_type"] == expected:
                if hit_ts is None:
                    hit_ts = float(fe["timestamp_ms"])
            else:
                wrong_triggers_in_window.append(fe["trigger_type"])
                if hit_ts is None and wrong_first is None:
                    wrong_first = fe["trigger_type"]

        latency_ms = round(max(0.0, hit_ts - cue_ts), 2) if hit_ts is not None else None
        cue_results.append(
            {
                "file": fp.name,
                "cue_id": cid,
                "key": f"{fp.name}::{cid}",
                "expected_trigger": expected,
                "hand": cue_hand,
                "cue_ts_ms": cue_ts,
                "t_start_ms": t_start,
                "t_end_ms": t_end,
                "hit": hit_ts is not None,
                "hit_ts_ms": hit_ts,
                "activation_latency_ms": latency_ms,
                "wrong_trigger_first": wrong_first,
                "wrong_triggers_count": len(wrong_triggers_in_window),
            }
        )

    extra_by_type: Dict[str, int] = {}
    extra_count = 0
    for idx, fe in enumerate(fired_events):
        if idx not in used_in_any_window:
            extra_count += 1
            tt = fe["trigger_type"]
            extra_by_type[tt] = extra_by_type.get(tt, 0) + 1

    return {
        "file": fp.name,
        "cue_results": cue_results,
        "extra_triggers_outside_windows": extra_count,
        "extra_by_type": extra_by_type,
    }


def _aggregate_cue_results(
    file_reports: Sequence[Dict[str, Any]],
    *,
    backend_name: str,
    classifier: Optional[PinchClassifier] = None,
) -> Dict[str, Any]:
    all_cues: List[Dict[str, Any]] = []
    extra_total = 0
    extra_by_type: Dict[str, int] = {}
    for fr in file_reports:
        all_cues.extend(fr["cue_results"])
        extra_total += int(fr["extra_triggers_outside_windows"])
        for k, v in (fr.get("extra_by_type") or {}).items():
            extra_by_type[k] = extra_by_type.get(k, 0) + int(v)

    by_trigger: Dict[str, Dict[str, Any]] = {}
    seen_triggers = list(CUED_TRIGGER_ORDER)
    for c in all_cues:
        t = c["expected_trigger"]
        if t and t not in seen_triggers:
            seen_triggers.append(t)

    for trig in seen_triggers:
        sub = [c for c in all_cues if c["expected_trigger"] == trig]
        n = len(sub)
        hits = sum(1 for c in sub if c["hit"])
        wrong_first = sum(1 for c in sub if c["wrong_trigger_first"] is not None)
        lats = [float(c["activation_latency_ms"]) for c in sub if c["activation_latency_ms"] is not None]
        rec = round(hits / float(n), 4) if n > 0 else 0.0
        w_lo, w_hi = wilson_interval(hits, n)
        by_trigger[trig] = {
            "n": n,
            "hits": hits,
            "misses": n - hits,
            "wrong_trigger_first_count": wrong_first,
            "recall": rec,
            "wilson_lower": w_lo,
            "wilson_upper": w_hi,
            "latency_p50_ms": round(float(np.percentile(lats, 50)), 2) if lats else None,
            "latency_p95_ms": round(float(np.percentile(lats, 95)), 2) if lats else None,
        }

    total_n = len(all_cues)
    total_hits = sum(1 for c in all_cues if c["hit"])
    ov_lo, ov_hi = wilson_interval(total_hits, total_n)
    out: Dict[str, Any] = {
        "backend": backend_name,
        "files_evaluated": [fr["file"] for fr in file_reports],
        "total_cues": total_n,
        "total_hits": total_hits,
        "overall_recall": round(total_hits / float(total_n), 4) if total_n > 0 else 0.0,
        "overall_wilson_lower": ov_lo,
        "overall_wilson_upper": ov_hi,
        "by_trigger": by_trigger,
        "extra_triggers_outside_windows": extra_total,
        "extra_by_type": extra_by_type,
        "cue_results": all_cues,
    }
    if classifier is not None and hasattr(classifier, "telemetry"):
        out["classifier_telemetry"] = classifier.telemetry()
    return out


def evaluate_event_recall(
    cued_paths: Sequence[str | Path],
    *,
    config: Optional[Dict[str, Any]] = None,
    classifier: Optional[PinchClassifier] = None,
    backend_name: str = "heuristic_3d",
    width: float = 1920.0,
    height: float = 1080.0,
    data_root: str | Path = "data",
    sealed_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Replay cued_events files under one classifier backend and compute per-trigger recall + Wilson CI."""
    cfg = deepcopy(config) if config is not None else load_config()
    resolved_paths = collect_json_paths([Path(p) for p in cued_paths])
    kept_paths, excluded_sealed = exclude_sealed_content(
        resolved_paths, data_root=data_root, sealed_root=sealed_root
    )
    if excluded_sealed:
        raise ValueError(
            f"REFUSED: {len(excluded_sealed)} cued file(s) match sealed_manifest.json SHA-256: "
            + ", ".join(p.name for p in excluded_sealed)
        )

    clf = classifier
    if clf is None:
        clf = Heuristic3DPinchClassifier((cfg.get("classifier") or {}).get("heuristic_3d"))

    file_reports: List[Dict[str, Any]] = []
    for fp in kept_paths:
        file_reports.append(
            _replay_single_cued_file(fp, config=cfg, classifier=clf, width=width, height=height)
        )
    return _aggregate_cue_results(file_reports, backend_name=backend_name, classifier=clf)


def evaluate_paired_event_recall(
    cued_paths: Sequence[str | Path],
    *,
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[str | Path] = None,
    normalizer_path: Optional[str | Path] = None,
    onnx_classifier: Optional[PinchClassifier] = None,
    max_recall_drop: float = 0.02,
    width: float = 1920.0,
    height: float = 1080.0,
    data_root: str | Path = "data",
    sealed_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Paired event-level recall evaluation comparing `heuristic_3d` vs `veto` on identical cued files.
    Enforces per-trigger recall gate: `heuristic_recall - veto_recall <= max_recall_drop` (default 2 pp).
    """
    cfg = deepcopy(config) if config is not None else load_config()
    replay_mod = _load_replay_eval_module()
    heur_clf = replay_mod.build_replay_classifier("heuristic_3d", cfg)
    veto_clf = replay_mod.build_replay_classifier(
        "veto",
        cfg,
        model_path=model_path,
        normalizer_path=normalizer_path,
        onnx_classifier=onnx_classifier,
    )

    rep_h = evaluate_event_recall(
        cued_paths,
        config=cfg,
        classifier=heur_clf,
        backend_name="heuristic_3d",
        width=width,
        height=height,
        data_root=data_root,
        sealed_root=sealed_root,
    )
    rep_v = evaluate_event_recall(
        cued_paths,
        config=cfg,
        classifier=veto_clf,
        backend_name="veto",
        width=width,
        height=height,
        data_root=data_root,
        sealed_root=sealed_root,
    )

    h_by_key = {c["key"]: c for c in rep_h["cue_results"]}
    v_by_key = {c["key"]: c for c in rep_v["cue_results"]}

    paired_by_trigger: Dict[str, Dict[str, Any]] = {}
    all_overheads_ms: List[float] = []
    all_passed = True
    for trig, h_stats in rep_h["by_trigger"].items():
        v_stats = rep_v["by_trigger"].get(trig, {"n": 0, "hits": 0, "recall": 0.0})
        n = int(h_stats["n"])
        h_rec = float(h_stats["recall"])
        v_rec = float(v_stats["recall"])
        drop = h_rec - v_rec
        delta_pp = round((v_rec - h_rec) * 100.0, 2)

        keys = [k for k, c in h_by_key.items() if c["expected_trigger"] == trig]
        h_only = sum(1 for k in keys if h_by_key[k]["hit"] and not v_by_key.get(k, {}).get("hit", False))
        v_only = sum(1 for k in keys if not h_by_key[k]["hit"] and v_by_key.get(k, {}).get("hit", False))
        both_h = sum(1 for k in keys if h_by_key[k]["hit"] and v_by_key.get(k, {}).get("hit", False))
        both_m = sum(1 for k in keys if not h_by_key[k]["hit"] and not v_by_key.get(k, {}).get("hit", False))

        # Pure veto overhead latency (veto_hit_ts - heuristic_hit_ts) on cues hit by both backends
        trig_overheads = [
            float(v_by_key[k]["hit_ts_ms"]) - float(h_by_key[k]["hit_ts_ms"])
            for k in keys
            if h_by_key[k]["hit"]
            and v_by_key.get(k, {}).get("hit", False)
            and h_by_key[k].get("hit_ts_ms") is not None
            and v_by_key[k].get("hit_ts_ms") is not None
        ]
        all_overheads_ms.extend(trig_overheads)

        trig_pass = True if n == 0 else (drop <= float(max_recall_drop) + 1e-9)
        if not trig_pass:
            all_passed = False
        paired_by_trigger[trig] = {
            "n": n,
            "heuristic_hits": int(h_stats["hits"]),
            "veto_hits": int(v_stats["hits"]),
            "heuristic_recall": h_rec,
            "veto_recall": v_rec,
            "recall_delta_pp": delta_pp,
            "heuristic_only_hits": h_only,
            "veto_only_hits": v_only,
            "both_hits": both_h,
            "both_misses": both_m,
            "veto_latency_overhead_p50_ms": (
                round(float(np.percentile(trig_overheads, 50)), 2) if trig_overheads else None
            ),
            "veto_latency_overhead_p95_ms": (
                round(float(np.percentile(trig_overheads, 95)), 2) if trig_overheads else None
            ),
            "veto_latency_overhead_max_ms": (
                round(float(max(trig_overheads)), 2) if trig_overheads else None
            ),
            "max_recall_drop": float(max_recall_drop),
            "passed": trig_pass,
        }

    return {
        "schema": "aerohand_event_recall_paired_v1",
        "max_recall_drop": float(max_recall_drop),
        "heuristic": rep_h,
        "veto": rep_v,
        "paired": {
            "passed": all_passed,
            "gate_scope": "per_trigger",
            "veto_latency_overhead_p50_ms": (
                round(float(np.percentile(all_overheads_ms, 50)), 2) if all_overheads_ms else None
            ),
            "veto_latency_overhead_p95_ms": (
                round(float(np.percentile(all_overheads_ms, 95)), 2) if all_overheads_ms else None
            ),
            "veto_latency_overhead_max_ms": (
                round(float(max(all_overheads_ms)), 2) if all_overheads_ms else None
            ),
            "by_trigger": paired_by_trigger,
        },
        "veto_telemetry": rep_v.get("classifier_telemetry"),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Event-level cued recall evaluator & paired heuristic-vs-veto comparator (V060-01b)."
    )
    parser.add_argument("--recall", nargs="+", required=True, help="Cued-event JSON files or directories (data/recall)")
    parser.add_argument(
        "--backend",
        choices=["heuristic_3d", "veto", "onnx_tcn"],
        default="heuristic_3d",
        help="Single-backend mode (ignored when --compare-veto is set)",
    )
    parser.add_argument(
        "--compare-veto",
        action="store_true",
        help="Run paired heuristic_3d vs veto recall comparison and enforce per-trigger recall gate",
    )
    parser.add_argument("--model", type=str, default=None, help="ONNX model path (for veto / onnx_tcn / --compare-veto)")
    parser.add_argument("--normalizer", type=str, default=None, help="Normalizer JSON path")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument(
        "--max-recall-drop",
        type=float,
        default=0.02,
        help="Max allowed per-trigger recall drop (heuristic - veto, default 0.02 = 2 pp)",
    )
    parser.add_argument("--data-root", type=str, default="data", help="Data root for sealed_manifest.json discovery")
    parser.add_argument("--sealed-root", type=str, default=None, help="Optional sealed manifest directory")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON report path")
    args = parser.parse_args(argv)

    cfg = load_config(config_path=args.config) if args.config else load_config()

    if args.compare_veto:
        if not args.model or not args.normalizer:
            raise SystemExit("--model and --normalizer are required with --compare-veto.")
        report = evaluate_paired_event_recall(
            args.recall,
            config=cfg,
            model_path=args.model,
            normalizer_path=args.normalizer,
            max_recall_drop=args.max_recall_drop,
            data_root=args.data_root,
            sealed_root=args.sealed_root,
        )
        passed = bool(report["paired"]["passed"])
    else:
        replay_mod = _load_replay_eval_module()
        clf = replay_mod.build_replay_classifier(
            args.backend,
            cfg,
            model_path=args.model,
            normalizer_path=args.normalizer,
        )
        report = evaluate_event_recall(
            args.recall,
            config=cfg,
            classifier=clf,
            backend_name=args.backend,
            data_root=args.data_root,
            sealed_root=args.sealed_root,
        )
        passed = True

    # Strip verbose per-cue list only if not needed? Keep full cue_results in JSON for auditability.
    out_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        out_fp = Path(args.out)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(out_text, encoding="utf-8")
    print(out_text)

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
