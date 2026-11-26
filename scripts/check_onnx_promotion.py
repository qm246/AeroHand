#!/usr/bin/env python3
"""
ONNX Promotion Readiness Checker for AeroHand (V050-04 / V051-11).

Supports two promotion stages (`--stage`):
  - `veto`: promotes an ONNX model into `classifier.mode="veto"` (does NOT require volunteer
    `subject_heldout_metrics`; still enforces provenance binding, sealed FT UCB, CPU latency,
    and active runtime safety guard).
  - `onnx` (default): full promotion from `veto` to `classifier.mode="onnx"` — enforces ALL 5 gates:
      1. Provenance binding (`provenance_hash_match`): `model_hash` and/or `model_sha256` match
         across `metrics.json`, sealed `replay_eval.py --heldout` report, `--model` file (if given),
         and `--vitals` (if given).
      2. Subject-held-out recall >= 0.97 for EVERY canonical class (stage=`onnx` only).
      3. Sealed AeroHand negatives FT 95% Poisson upper confidence bound <= 2.0/h over >= 1.5h
         hand-present idle exposure (Garwood: k=0 -> 1.50h, k=1 -> 2.37h, k=2 -> 3.15h).
      4. M1 CPU p95 ONNX inference latency < 3.0 ms.
      5. Runtime safety guard active (`max_onnx_p95_ms <= 8.0` and not in fallback state).

Prints structured pass/fail per criterion and exits non-zero if ANY criterion fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from aerohand.config import load_config
from aerohand.core.types import ALL_POSE_NAMES
from aerohand.ml.provenance import compute_onnx_model_hash
from aerohand.ml.recording_io import sha256_file


def _load_replay_eval_helpers():
    fp = Path(__file__).resolve().parent / "replay_eval.py"
    spec = importlib.util.spec_from_file_location("_aerohand_replay_eval_promo", fp)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {fp}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compute_onnx_model_hash(model_path: str | Path, normalizer_path: str | Path) -> str:
    return compute_onnx_model_hash(model_path, normalizer_path)


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


def _norm_opt_hex(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s else None


def check_onnx_promotion(
    *,
    metrics_report: Dict[str, Any],
    sealed_eval_report: Dict[str, Any],
    latency_p95_ms: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    vitals_report: Optional[Dict[str, Any]] = None,
    model_path: Optional[str | Path] = None,
    normalizer_path: Optional[str | Path] = None,
    stage: str = "onnx",
    min_class_recall: float = 0.97,
    max_ft_ucb_per_hour: float = 2.0,
    min_exposure_hours: float = 1.5,
    max_latency_p95_ms: float = 3.0,
    max_guard_p95_ms: float = 8.0,
) -> Dict[str, Any]:
    """Evaluate all ONNX promotion criteria for `stage` (`veto` | `onnx`) with provenance binding (V051-11)."""
    stage_norm = str(stage or "onnx").strip().lower()
    if stage_norm not in ("veto", "onnx"):
        raise ValueError(f"Unsupported promotion stage '{stage}' (expected 'veto' or 'onnx').")

    cfg = config if config is not None else load_config()
    replay_mod = _load_replay_eval_helpers()
    criteria: Dict[str, Dict[str, Any]] = {}

    # Criterion 0: Provenance binding across metrics, sealed_eval, --model, and --vitals (V051-11)
    ho_audit = sealed_eval_report.get("heldout_audit")
    ho_audit_dict = ho_audit if isinstance(ho_audit, dict) else {}

    m_hash = _norm_opt_hex(metrics_report.get("model_hash"))
    m_sha = _norm_opt_hex(metrics_report.get("model_sha256"))
    s_hash = _norm_opt_hex(sealed_eval_report.get("model_hash") or ho_audit_dict.get("model_hash"))
    s_sha = _norm_opt_hex(sealed_eval_report.get("model_sha256") or ho_audit_dict.get("model_sha256"))

    file_sha = _norm_opt_hex(sha256_file(model_path)) if (model_path and Path(model_path).is_file()) else None
    file_hash = (
        _norm_opt_hex(_compute_onnx_model_hash(model_path, normalizer_path))
        if (model_path and normalizer_path and Path(model_path).is_file() and Path(normalizer_path).is_file())
        else None
    )

    cls_v = (vitals_report.get("classifier") or {}) if isinstance(vitals_report, dict) else {}
    v_hash = _norm_opt_hex(cls_v.get("model_hash"))
    v_sha = _norm_opt_hex(cls_v.get("model_sha256"))

    prov_reasons: List[str] = []
    if not (m_hash or m_sha):
        prov_reasons.append("metrics_report missing model_hash and model_sha256")
    if not (s_hash or s_sha):
        prov_reasons.append("sealed_eval_report missing model_hash and model_sha256")
    # V052-02: When --vitals is provided, vitals MUST report provenance hash and match metrics/sealed_eval
    if vitals_report is not None and not (v_hash or v_sha):
        prov_reasons.append("vitals_report missing classifier.model_hash and classifier.model_sha256")

    observed_hashes = {
        k: v
        for k, v in (
            ("metrics", m_hash),
            ("sealed_eval", s_hash),
            ("model_files", file_hash),
            ("vitals", v_hash),
        )
        if v is not None
    }
    observed_shas = {
        k: v
        for k, v in (
            ("metrics", m_sha),
            ("sealed_eval", s_sha),
            ("model_file", file_sha),
            ("vitals", v_sha),
        )
        if v is not None
    }
    if len(set(observed_hashes.values())) > 1:
        prov_reasons.append(f"model_hash mismatch across artifacts: {observed_hashes}")
    if len(set(observed_shas.values())) > 1:
        prov_reasons.append(f"model_sha256 mismatch across artifacts: {observed_shas}")
    # Ensure metrics & sealed_eval share at least one verified common identifier
    if (m_hash or m_sha) and (s_hash or s_sha):
        shared_hash_ok = bool(m_hash and s_hash and m_hash == s_hash)
        shared_sha_ok = bool(m_sha and s_sha and m_sha == s_sha)
        if not (shared_hash_ok or shared_sha_ok):
            prov_reasons.append(
                f"metrics and sealed_eval do not share a matching model_hash or model_sha256 "
                f"(metrics={{hash:{m_hash}, sha:{m_sha}}}, sealed={{hash:{s_hash}, sha:{s_sha}}})"
            )
    if vitals_report is not None and (v_hash or v_sha) and (m_hash or m_sha or s_hash or s_sha):
        ref_hash = m_hash or s_hash
        ref_sha = m_sha or s_sha
        v_hash_ok = bool(v_hash and ref_hash and v_hash == ref_hash)
        v_sha_ok = bool(v_sha and ref_sha and v_sha == ref_sha)
        if not (v_hash_ok or v_sha_ok):
            prov_reasons.append(
                f"vitals_report does not share a matching model_hash or model_sha256 with metrics/sealed_eval "
                f"(vitals={{hash:{v_hash}, sha:{v_sha}}})"
            )

    c0_pass = len(prov_reasons) == 0
    criteria["provenance_hash_match"] = {
        "passed": c0_pass,
        "observed_model_hashes": observed_hashes,
        "observed_model_sha256s": observed_shas,
        "required": "Matching model_hash and/or model_sha256 across metrics, sealed_eval, model, and vitals",
        "detail": (
            "All artifacts bound to identical model provenance."
            if c0_pass
            else "; ".join(prov_reasons)
        ),
    }

    # Criterion 1: Subject-held-out recall >= 0.97 for EVERY canonical class (required for `onnx`, skipped for `veto`)
    if stage_norm == "veto":
        criteria["subject_heldout_recall"] = {
            "passed": True,
            "skipped": True,
            "required": "Not required for stage='veto' (enforced when promoting to stage='onnx')",
            "failing_classes": {},
            "detail": "Skipped for stage='veto'; volunteer subject-held-out recall is checked on full 'onnx' promotion.",
        }
    else:
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
            "skipped": False,
            "required": f">= {min_class_recall:.2f} for all {len(ALL_POSE_NAMES)} classes",
            "failing_classes": failing_classes,
            "detail": c1_detail,
        }

    # Criterion 2: Sealed negatives FT 95% UCB <= 2.0/h over >= 1.5h hand-present idle exposure
    ft = sealed_eval_report.get("false_triggers") or {}
    exp_h = float(ft.get("hand_present_idle_hours", 0.0))
    if ft.get("total_upper_bound_per_hour") is not None:
        ucb = float(ft["total_upper_bound_per_hour"])
    elif ft.get("total") is not None and exp_h > 0.0:
        ucb = float(replay_mod.poisson_upper_bound_rate(int(ft["total"]), exp_h))
    else:
        ucb = float("inf")
    has_sealed_audit = isinstance(ho_audit, dict) and int(ho_audit.get("touched", 0)) >= 1

    garwood_table = {
        f"k={k}": replay_mod.required_exposure_hours(k, max_ucb=max_ft_ucb_per_hour)
        for k in (0, 1, 2, 3)
    }
    exposure_note = (
        f"UCB <= {max_ft_ucb_per_hour:.1f}/h at {min_exposure_hours:.1f}h requires 0 FT "
        f"(Garwood 95% min exposure: k=0 -> {garwood_table['k=0']:.2f}h, "
        f"k=1 -> {garwood_table['k=1']:.2f}h, k=2 -> {garwood_table['k=2']:.2f}h)."
    )

    c2_reasons: List[str] = []
    if not has_sealed_audit:
        c2_reasons.append("missing sealed heldout_audit (must come from `replay_eval.py --heldout`)")
    if exp_h + 1e-9 < float(min_exposure_hours):
        c2_reasons.append(f"hand_present_idle_hours={exp_h:.4f}h < {min_exposure_hours:.2f}h")
    if ucb > float(max_ft_ucb_per_hour) + 1e-9:
        c2_reasons.append(f"total_upper_bound_per_hour={ucb:.3f}/h > {max_ft_ucb_per_hour:.2f}/h")
    c2_pass = len(c2_reasons) == 0
    criteria["sealed_negatives_ft_ucb"] = {
        "passed": c2_pass,
        "hand_present_idle_hours": round(exp_h, 6),
        "total_upper_bound_per_hour": round(ucb, 3) if np.isfinite(ucb) else None,
        "has_sealed_audit": has_sealed_audit,
        "garwood_exposure_table": garwood_table,
        "note": exposure_note,
        "required": f"sealed UCB <= {max_ft_ucb_per_hour:.2f}/h over >= {min_exposure_hours:.2f}h",
        "detail": "Passed sealed FT UCB and exposure gate." if c2_pass else "; ".join(c2_reasons),
    }

    # Criterion 3: M1 CPU p95 latency < 3.0 ms
    eff_lat = latency_p95_ms
    if eff_lat is None and isinstance(vitals_report, dict):
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
        "schema": "aerohand_onnx_promotion_check_v2",
        "stage": stage_norm,
        "passed": overall_passed,
        "garwood_exposure_table": garwood_table,
        "exposure_note": exposure_note,
        "criteria": criteria,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify AeroHand criteria & provenance before promoting ONNX to veto or full onnx mode (V051-11)."
    )
    parser.add_argument("--metrics", type=str, required=True, help="Path to train_tcn.py *_metrics.json")
    parser.add_argument(
        "--sealed-eval",
        type=str,
        required=True,
        help="Path to replay_eval.py --heldout JSON report",
    )
    parser.add_argument(
        "--stage",
        choices=["veto", "onnx"],
        default="onnx",
        help="Target promotion stage: 'veto' (skips subject-held-out recall) or 'onnx' (full promotion, default)",
    )
    parser.add_argument("--model", type=str, default=None, help="Optional ONNX model to verify SHA-256 & benchmark latency")
    parser.add_argument("--normalizer", type=str, default=None, help="Optional normalizer JSON for model_hash & latency benchmark")
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
        model_path=args.model,
        normalizer_path=args.normalizer,
        stage=args.stage,
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
