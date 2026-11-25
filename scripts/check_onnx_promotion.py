#!/usr/bin/env python3
"""
ONNX Full-Promotion Readiness Checker for AeroHand (V050-04).

Verifies all 4 mandatory promotion gates before switching `classifier.mode` from
`veto` to full `onnx`:
  1. Subject-held-out recall >= 0.97 for EVERY canonical class (from `metrics.json`).
  2. Sealed AeroHand negatives FT 95% Poisson upper confidence bound <= 2.0/h over
     >= 1.5 hours of hand-present idle exposure (from `replay_eval.py --heldout` JSON).
  3. M1 CPU p95 ONNX inference latency < 3.0 ms (measured directly via onnxruntime or
     provided via `--latency-p95-ms` / `--vitals`).
  4. Runtime safety guard active (`max_onnx_p95_ms <= 8.0` and error fallback enabled).

Prints structured pass/fail per criterion and exits non-zero if ANY criterion fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from aerohand.config import load_config
from aerohand.core.types import ALL_POSE_NAMES


def benchmark_onnx_p95_ms(
    model_path: str | Path,
    normalizer_path: str | Path,
    *,
    warmup: int = 10,
    iterations: int = 100,
) -> float:
    """Measure single-threaded CPU p95 inference latency (ms) on the current host."""
    import onnxruntime as ort

    norm = json.loads(Path(normalizer_path).read_text(encoding="utf-8"))
    feat_dim = int(norm.get("feature_dim") or len(norm["mean"]))
    seq_len = int(norm.get("sequence_length") or 5)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    x = np.zeros((1, seq_len, feat_dim), dtype=np.float32)

    for _ in range(max(1, warmup)):
        sess.run(None, {in_name: x})

    times_ms: List[float] = []
    for _ in range(max(10, iterations)):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return round(float(np.percentile(times_ms, 95)), 3)


def check_onnx_promotion(
    *,
    metrics_report: Dict[str, Any],
    sealed_eval_report: Dict[str, Any],
    latency_p95_ms: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    vitals_report: Optional[Dict[str, Any]] = None,
    min_class_recall: float = 0.97,
    max_ft_ucb_per_hour: float = 2.0,
    min_exposure_hours: float = 1.5,
    max_latency_p95_ms: float = 3.0,
    max_guard_p95_ms: float = 8.0,
) -> Dict[str, Any]:
    """Evaluate all 4 ONNX promotion criteria and return a structured pass/fail audit."""
    cfg = config if config is not None else load_config()
    criteria: Dict[str, Dict[str, Any]] = {}

    # Criterion 1: Subject-held-out recall >= 0.97 for EVERY canonical class
    ho_metrics = metrics_report.get("subject_heldout_metrics")
    per_cls = (ho_metrics or {}).get("per_class") if isinstance(ho_metrics, dict) else None
    failing_classes: Dict[str, Dict[str, Any]] = {}
    if not isinstance(per_cls, dict):
        c1_pass = False
        c1_detail = "Missing `subject_heldout_metrics.per_class` in metrics report."
    else:
        for pose in ALL_POSE_NAMES:
            entry = per_cls.get(pose.value)
            if not isinstance(entry, dict) or int(entry.get("support", 0)) <= 0:
                failing_classes[pose.value] = {"recall": None, "support": 0}
            else:
                rec = float(entry.get("recall", 0.0))
                if rec < float(min_class_recall):
                    failing_classes[pose.value] = {"recall": rec, "support": int(entry["support"])}
        c1_pass = len(failing_classes) == 0
        c1_detail = (
            f"All {len(ALL_POSE_NAMES)} canonical classes achieved recall >= {min_class_recall:.2f}."
            if c1_pass
            else f"Classes below {min_class_recall:.2f} recall or missing support: {failing_classes}"
        )
    criteria["subject_heldout_recall"] = {
        "passed": c1_pass,
        "required": f">= {min_class_recall:.2f} for all {len(ALL_POSE_NAMES)} classes",
        "failing_classes": failing_classes,
        "detail": c1_detail,
    }

    # Criterion 2: Sealed negatives FT 95% UCB <= 2.0/h over >= 1.5h hand-present idle exposure
    ft = sealed_eval_report.get("false_triggers") or {}
    ho_audit = sealed_eval_report.get("heldout_audit")
    exp_h = float(ft.get("hand_present_idle_hours", 0.0))
    ucb = float(ft.get("total_upper_bound_per_hour", float("inf")))
    has_sealed_audit = isinstance(ho_audit, dict) and int(ho_audit.get("touched", 0)) >= 1
    c2_reasons: List[str] = []
    if not has_sealed_audit:
        c2_reasons.append("missing sealed heldout_audit (must come from `replay_eval.py --heldout`)")
    if exp_h < float(min_exposure_hours):
        c2_reasons.append(f"hand_present_idle_hours={exp_h:.4f}h < {min_exposure_hours:.2f}h")
    if ucb > float(max_ft_ucb_per_hour):
        c2_reasons.append(f"total_upper_bound_per_hour={ucb:.3f}/h > {max_ft_ucb_per_hour:.2f}/h")
    c2_pass = len(c2_reasons) == 0
    criteria["sealed_negatives_ft_ucb"] = {
        "passed": c2_pass,
        "hand_present_idle_hours": round(exp_h, 6),
        "total_upper_bound_per_hour": round(ucb, 3) if np.isfinite(ucb) else None,
        "has_sealed_audit": has_sealed_audit,
        "required": f"sealed UCB <= {max_ft_ucb_per_hour:.2f}/h over >= {min_exposure_hours:.2f}h",
        "detail": "Passed sealed FT UCB and exposure gate." if c2_pass else "; ".join(c2_reasons),
    }

    # Criterion 3: M1 CPU p95 latency < 3.0 ms
    eff_lat = latency_p95_ms
    if eff_lat is None and isinstance(vitals_report, dict):
        cls_v = vitals_report.get("classifier") or {}
        if cls_v.get("onnx_p95_ms") is not None and float(cls_v["onnx_p95_ms"]) > 0.0:
            eff_lat = float(cls_v["onnx_p95_ms"])
    c3_pass = eff_lat is not None and 0.0 < float(eff_lat) < float(max_latency_p95_ms)
    criteria["cpu_latency_p95_ms"] = {
        "passed": c3_pass,
        "observed_ms": round(float(eff_lat), 3) if eff_lat is not None else None,
        "required": f"< {max_latency_p95_ms:.1f} ms on CPUExecutionProvider",
        "detail": (
            f"CPU p95 latency {eff_lat:.3f} ms < {max_latency_p95_ms:.1f} ms."
            if c3_pass
            else (
                "Latency measurement missing (pass --model/--normalizer, --latency-p95-ms, or --vitals)."
                if eff_lat is None
                else f"CPU p95 latency {eff_lat:.3f} ms >= {max_latency_p95_ms:.1f} ms."
            )
        ),
    }

    # Criterion 4: Runtime fallback guard active (`max_onnx_p95_ms <= 8.0`)
    cls_cfg = cfg.get("classifier") or {}
    guard_cap = float(cls_cfg.get("max_onnx_p95_ms", 8.0))
    guard_active = 0.5 <= guard_cap <= float(max_guard_p95_ms)
    if isinstance(vitals_report, dict) and isinstance(vitals_report.get("classifier"), dict):
        cls_v = vitals_report["classifier"]
        if cls_v.get("guard_active") is False:
            guard_active = False
        if cls_v.get("fell_back") is True:
            guard_active = False
    criteria["runtime_guard_active"] = {
        "passed": guard_active,
        "max_onnx_p95_ms": guard_cap,
        "required": f"GuardedClassifier active with max_onnx_p95_ms <= {max_guard_p95_ms:.1f} ms and not fell_back",
        "detail": (
            f"Safety fallback guard active (max_onnx_p95_ms={guard_cap:.1f} ms)."
            if guard_active
            else "Safety guard misconfigured, disabled, or currently in fallback state."
        ),
    }

    overall_passed = all(c["passed"] for c in criteria.values())
    return {
        "schema": "aerohand_onnx_promotion_check_v1",
        "passed": overall_passed,
        "criteria": criteria,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify all 4 AeroHand criteria before promoting ONNX from veto-only to full mode (V050-04)."
    )
    parser.add_argument("--metrics", type=str, required=True, help="Path to train_tcn.py metrics.json")
    parser.add_argument(
        "--sealed-eval",
        type=str,
        required=True,
        help="Path to replay_eval.py --heldout JSON report",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional ONNX model to benchmark CPU p95 latency")
    parser.add_argument("--normalizer", type=str, default=None, help="Optional normalizer JSON for latency benchmark")
    parser.add_argument("--latency-p95-ms", type=float, default=None, help="Explicit measured CPU p95 latency (ms)")
    parser.add_argument("--vitals", type=str, default=None, help="Optional `aerohand vitals --json` output file")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output report path")
    args = parser.parse_args(argv)

    metrics_data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    sealed_data = json.loads(Path(args.sealed_eval).read_text(encoding="utf-8"))
    vitals_data = json.loads(Path(args.vitals).read_text(encoding="utf-8")) if args.vitals else None
    cfg = load_config(config_path=args.config) if args.config else load_config()

    lat_ms = args.latency_p95_ms
    if lat_ms is None and args.model and args.normalizer:
        lat_ms = benchmark_onnx_p95_ms(args.model, args.normalizer)

    report = check_onnx_promotion(
        metrics_report=metrics_data,
        sealed_eval_report=sealed_data,
        latency_p95_ms=lat_ms,
        config=cfg,
        vitals_report=vitals_data,
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
