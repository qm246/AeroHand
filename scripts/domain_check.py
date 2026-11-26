#!/usr/bin/env python3
"""
Domain Shift Diagnostic between Converted GestureCtl Data and Fresh AeroHand Sessions (V041-02).

Compares feature distributions (palm size, 3D palm-frame thumb-index distance, wrist speed)
using a pure-NumPy two-sample Kolmogorov-Smirnov test and median relative difference.
Warns whenever p < 0.01 or |median_a - median_b| / max(|median_b|, 1e-6) > 0.15.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aerohand.core.palm_frame import INDEX_TIP, THUMB_TIP, compute_palm_frame
from aerohand.ml.recording_io import (
    collect_json_paths,
    exclude_cued_events,
    exclude_sealed_content,
    load_recording_frames,
)


def ks_2samp_stat_and_pvalue(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Pure-NumPy two-sample Kolmogorov-Smirnov statistic `D` and asymptotic p-value."""
    xa = np.sort(np.asarray(a, dtype=np.float64))
    xb = np.sort(np.asarray(b, dtype=np.float64))
    na, nb = xa.size, xb.size
    if na == 0 or nb == 0:
        return 0.0, 1.0
    combined = np.concatenate([xa, xb])
    cdf_a = np.searchsorted(xa, combined, side="right") / float(na)
    cdf_b = np.searchsorted(xb, combined, side="right") / float(nb)
    d_stat = float(np.max(np.abs(cdf_a - cdf_b)))

    n_eff = (na * nb) / float(na + nb)
    sqrt_n = math.sqrt(max(1.0, n_eff))
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d_stat
    if lam <= 1e-9:
        return round(d_stat, 5), 1.0
    # Kolmogorov survival series: 2 * sum_{k=1..100} (-1)^(k-1) exp(-2 k^2 lam^2)
    p_val = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * (k * k) * (lam * lam))
        p_val += term if (k % 2 == 1) else -term
        if term < 1e-12:
            break
    p_val = max(0.0, min(1.0, 2.0 * p_val))
    return round(d_stat, 5), round(p_val, 6)


def extract_domain_features(paths: Sequence[Path]) -> Dict[str, List[float]]:
    """Extract `palm_size`, `thumb_index_dist_3d`, and `wrist_speed` across recordings (excluding `cued_events`)."""
    palm_sizes: List[float] = []
    ti_dists: List[float] = []
    wrist_speeds: List[float] = []

    kept_paths, _ = exclude_cued_events(paths)
    for fp in kept_paths:
        frames, _ = load_recording_frames(fp)
        prev_wrist: Optional[Tuple[float, float, float]] = None
        prev_ts: Optional[float] = None
        for fr in frames:
            if not fr["hand_present"] or not fr["landmarks_2d"]:
                prev_wrist = None
                prev_ts = None
                continue
            lms2d = fr["landmarks_2d"]
            lms3d = fr["world_landmarks"] or lms2d
            ts = float(fr["timestamp_ms"])
            w, m = lms2d[0], lms2d[9]
            palm_sizes.append(math.hypot(w.x - m.x, w.y - m.y))

            pf = compute_palm_frame(
                lms3d,
                anchor="palm_center",
                is_left_hand=(fr["handedness"] == "Left"),
                scale_metric="palm_width",
            )
            if pf is not None:
                ti_dists.append(pf.dist(THUMB_TIP, INDEX_TIP))

            cur_w = (lms2d[0].x, lms2d[0].y, lms2d[0].z)
            if prev_wrist is not None and prev_ts is not None:
                dt_s = (ts - prev_ts) / 1000.0
                if 0.005 <= dt_s <= 0.25:
                    spd = math.hypot(cur_w[0] - prev_wrist[0], cur_w[1] - prev_wrist[1]) / dt_s
                    wrist_speeds.append(spd)
            prev_wrist = cur_w
            prev_ts = ts

    return {
        "palm_size": palm_sizes,
        "thumb_index_dist_3d": ti_dists,
        "wrist_speed": wrist_speeds,
    }


def compare_domains(
    converted_paths: Sequence[Path],
    aerohand_paths: Sequence[Path],
) -> Dict[str, Any]:
    fa = extract_domain_features(converted_paths)
    fb = extract_domain_features(aerohand_paths)
    metrics: Dict[str, Any] = {}
    warnings: List[str] = []

    for feat in ("palm_size", "thumb_index_dist_3d", "wrist_speed"):
        va, vb = fa[feat], fb[feat]
        med_a = float(np.median(va)) if va else 0.0
        med_b = float(np.median(vb)) if vb else 0.0
        rel_diff = abs(med_a - med_b) / max(abs(med_b), 1e-6) if (va and vb) else 0.0
        ks_d, p_val = ks_2samp_stat_and_pvalue(va, vb)
        warn = bool((p_val < 0.01) or (rel_diff > 0.15))
        if warn:
            warnings.append(f"{feat}: ks_p={p_val:.4f}, median_diff={rel_diff*100:.1f}%")
        metrics[feat] = {
            "n_converted": len(va),
            "n_aerohand": len(vb),
            "median_converted": round(med_a, 4),
            "median_aerohand": round(med_b, 4),
            "median_rel_diff": round(rel_diff, 4),
            "ks_stat": ks_d,
            "ks_pvalue": p_val,
            "warn": warn,
        }

    return {"features": metrics, "warnings": warnings, "domain_shift_detected": len(warnings) > 0}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Two-sample KS domain check: GestureCtl vs AeroHand (V041-02).")
    parser.add_argument("--converted", nargs="+", required=True, help="Converted GestureCtl JSON files or dirs")
    parser.add_argument("--aerohand", nargs="+", required=True, help="Fresh AeroHand session JSON files or dirs")
    parser.add_argument(
        "--sealed-root",
        type=str,
        default=None,
        help="Optional directory scanned for sealed_manifest.json SHA-256 exclusion",
    )
    args = parser.parse_args(argv)

    conv_paths, _ = exclude_sealed_content(collect_json_paths(args.converted), sealed_root=args.sealed_root)
    aero_paths, _ = exclude_sealed_content(collect_json_paths(args.aerohand), sealed_root=args.sealed_root)
    conv_paths, _ = exclude_cued_events(conv_paths)
    aero_paths, _ = exclude_cued_events(aero_paths)
    report = compare_domains(conv_paths, aero_paths)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
