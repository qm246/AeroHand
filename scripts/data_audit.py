#!/usr/bin/env python3
"""
Read-Only Dataset, Negative-Exposure & Cued-Recall Auditor for AeroHand (V060-07).

Audits local recording directories (strictly excluding `_drafts/` and never reading
sealed `heldout` directories directly) and outputs a structured JSON report plus
an optional human-readable summary table.

Checks performed:
  1. Integrity (blocking errors — always exit non-zero, even with `--allow-partial`):
       - Corrupt / unreadable JSON recordings
       - Duplicate SHA-256 content across audited files
       - SHA-256 overlap with any discovered `sealed_manifest.json`
       - Missing `world_landmarks` in AeroHand v2 sessions
       - In `cued_events` files (`data/recall/`): duplicate `cue_id`, overlapping or
         invalid `[t_start_ms, t_end_ms]` windows, unknown `expected_trigger`, or
         `(pose, expected_trigger)` mismatch against `CUED_TRIGGER_SPECS`
  2. Readiness (reported in `readiness_gaps`; fails exit code unless `--allow-partial`):
       - Class x hand progress vs `ML_EVALUATION.md` targets (21k fresh AeroHand frames)
       - Wrist angle coverage (`0/30/45/60` >= 15% each) & 8 hard-negative groups (>= 500 each)
       - Estimated gap-free 5-frame training windows per class >= `--min-train-windows` (300)
       - Cued recall counts per trigger >= `--min-cues-per-trigger` (50)
  3. Diagnostics & Warnings:
       - Hand-balance warning (> 70/30 Left/Right skew per class)
       - Median FPS & same-hand timestamp gaps > 100 ms
       - `hand_present_idle_hours` broken down by source (`sessions`, `negatives`, `converted`, `passive`)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from aerohand.core.types import ALL_POSE_NAMES
from aerohand.ml.collector import (
    COLLECTOR_CLASS_TARGETS,
    CORPUS_TOTAL_TARGET,
    CUED_TRIGGER_ORDER,
    CUED_TRIGGER_SPECS,
    DISTANCE_BUCKETS,
    HARD_NEGATIVE_GROUPS,
    HARD_NEGATIVE_MIN_PER_GROUP,
    LIGHTING_BUCKETS,
    MIN_ANGLE_SHARE,
    WRIST_ANGLE_BUCKETS,
)
from aerohand.ml.recording_io import (
    collect_json_paths,
    find_sealed_manifests,
    is_cued_events_recording,
    load_recording_frames,
    read_sealed_manifest_hashes,
    sha256_file,
)


def _existing_paths(items: Sequence[str | Path]) -> List[Path]:
    out: List[Path] = []
    for raw in items:
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            out.append(p)
    return out


def _compute_idle_hours_and_timing(
    frames: Sequence[Dict[str, Any]],
    *,
    deliberate_marks_ms: Sequence[float] = (),
    mark_half_window_ms: float = 1500.0,
) -> Dict[str, Any]:
    """Compute `hand_present_idle_hours`, median FPS, and same-hand gaps > 100 ms."""
    idle_ms = 0.0
    dts_ms: List[float] = []
    gaps_gt_100ms = 0
    prev_present_ts: Optional[float] = None
    last_ts_by_hand: Dict[str, float] = {}
    marks = [float(m) for m in deliberate_marks_ms]

    def _in_marked_window(ts_val: float) -> bool:
        for m in marks:
            if abs(ts_val - m) <= mark_half_window_ms:
                return True
        return False

    for fr in frames:
        ts = float(fr["timestamp_ms"])
        if not fr["hand_present"] or not fr["landmarks_2d"]:
            prev_present_ts = None
            continue

        hid = str(fr.get("hand_id") or fr.get("handedness") or "Right")
        if hid in last_ts_by_hand:
            dt_h = ts - last_ts_by_hand[hid]
            if dt_h > 100.0:
                gaps_gt_100ms += 1
            elif dt_h > 0.0:
                dts_ms.append(dt_h)
        last_ts_by_hand[hid] = ts

        excluded = _in_marked_window(ts) if marks else False
        if prev_present_ts is not None:
            dt = ts - prev_present_ts
            if 0.0 < dt <= 500.0 and fr["is_idle_gt"] and not excluded:
                idle_ms += dt
        else:
            if fr["is_idle_gt"] and not excluded:
                idle_ms += 33.33
        prev_present_ts = ts

    med_dt = float(np.median(dts_ms)) if dts_ms else 0.0
    med_fps = round(1000.0 / med_dt, 2) if med_dt > 1e-6 else 0.0
    return {
        "hand_present_idle_hours": idle_ms / 3_600_000.0,
        "gaps_gt_100ms": gaps_gt_100ms,
        "median_fps": med_fps,
        "sample_dts_ms": dts_ms,
    }


def _estimate_train_windows(
    frames: Sequence[Dict[str, Any]],
    *,
    seq_len: int = 5,
    max_gap_ms: float = 100.0,
    deliberate_marks_ms: Sequence[float] = (),
    mark_half_window_ms: float = 1500.0,
) -> Dict[str, int]:
    """Count gap-free, per-hand sliding windows of length `seq_len` per class (2-hand safe)."""
    counts: Dict[str, int] = {p.value: 0 for p in ALL_POSE_NAMES}
    runs_by_hand: Dict[str, List[str]] = {}
    prev_ts_by_hand: Dict[str, float] = {}
    marks = [float(m) for m in deliberate_marks_ms]

    def _flush_hand(hid: str) -> None:
        run_labels = runs_by_hand.pop(hid, None)
        prev_ts_by_hand.pop(hid, None)
        if run_labels and len(run_labels) >= seq_len:
            for lbl in run_labels[seq_len - 1 :]:
                if lbl in counts:
                    counts[lbl] += 1

    def _flush_all() -> None:
        for hid in list(runs_by_hand.keys()):
            _flush_hand(hid)

    for fr in frames:
        ts = float(fr["timestamp_ms"])
        if not fr["hand_present"] or not fr["landmarks_2d"]:
            _flush_all()
            continue
        if marks and any(abs(ts - m) <= mark_half_window_ms for m in marks):
            _flush_all()
            continue
        hid = str(fr.get("hand_id") or fr.get("handedness") or "Right")
        prev_ts = prev_ts_by_hand.get(hid)
        if hid in runs_by_hand and prev_ts is not None and ((ts - prev_ts) > max_gap_ms or ts < prev_ts):
            _flush_hand(hid)
        runs_by_hand.setdefault(hid, []).append(str(fr["label_name"]))
        prev_ts_by_hand[hid] = ts

    _flush_all()
    return counts


def _audit_cued_file(
    fp: Path,
    *,
    blocking_errors: List[str],
    warnings: List[str],
) -> Dict[str, Any]:
    """Validate one `cued_events` recording and return per-trigger cue counts."""
    by_trigger: Dict[str, int] = {t: 0 for t in CUED_TRIGGER_ORDER}
    try:
        raw = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as exc:
        blocking_errors.append(f"Corrupt JSON in cued file {fp.name}: {exc}")
        return {"file": fp.name, "valid": False, "cues": 0, "by_trigger": by_trigger}

    if not isinstance(raw, dict):
        blocking_errors.append(f"Cued file {fp.name} root must be a JSON object.")
        return {"file": fp.name, "valid": False, "cues": 0, "by_trigger": by_trigger}

    kind = str(raw.get("recording_kind") or "").strip().lower()
    if kind != "cued_events":
        blocking_errors.append(f"Recall file {fp.name} has recording_kind={kind!r} (expected 'cued_events').")

    schema = str(raw.get("schema") or "")
    if schema != "aerohand_pinch_v2":
        warnings.append(f"Cued file {fp.name} schema is {schema!r} (expected 'aerohand_pinch_v2').")

    cues = raw.get("cue_events")
    if not isinstance(cues, list):
        blocking_errors.append(f"Cued file {fp.name} missing 'cue_events' list.")
        return {"file": fp.name, "valid": False, "cues": 0, "by_trigger": by_trigger}

    seen_ids: Set[str] = set()
    windows: List[Tuple[float, float, str]] = []
    file_ok = True

    for idx, ev in enumerate(cues):
        if not isinstance(ev, dict):
            blocking_errors.append(f"Cued file {fp.name} cue #{idx} is not a dict.")
            file_ok = False
            continue
        cid = str(ev.get("cue_id") or "").strip()
        if not cid:
            blocking_errors.append(f"Cued file {fp.name} cue #{idx} missing cue_id.")
            file_ok = False
        elif cid in seen_ids:
            blocking_errors.append(f"Cued file {fp.name} has duplicate cue_id '{cid}'.")
            file_ok = False
        else:
            seen_ids.add(cid)

        trig = str(ev.get("expected_trigger") or "").strip()
        pose = str(ev.get("pose") or "").strip()
        spec = CUED_TRIGGER_SPECS.get(trig)
        if spec is None:
            blocking_errors.append(
                f"Cued file {fp.name} cue '{cid or idx}' has unknown expected_trigger '{trig}'."
            )
            file_ok = False
        elif pose != spec.pose.value:
            blocking_errors.append(
                f"Cued file {fp.name} cue '{cid or idx}' pose/trigger mismatch: "
                f"pose='{pose}' vs expected '{spec.pose.value}' for trigger '{trig}'."
            )
            file_ok = False
        else:
            by_trigger[trig] += 1

        try:
            t_start = float(ev["t_start_ms"])
            t_end = float(ev["t_end_ms"])
        except Exception:
            blocking_errors.append(
                f"Cued file {fp.name} cue '{cid or idx}' missing numeric t_start_ms/t_end_ms."
            )
            file_ok = False
            continue
        if t_end <= t_start:
            blocking_errors.append(
                f"Cued file {fp.name} cue '{cid or idx}' invalid window [{t_start}, {t_end}]."
            )
            file_ok = False
        else:
            windows.append((t_start, t_end, cid or str(idx)))

    windows.sort(key=lambda item: (item[0], item[1]))
    for i in range(1, len(windows)):
        prev_s, prev_e, prev_id = windows[i - 1]
        cur_s, cur_e, cur_id = windows[i]
        if cur_s < prev_e:
            blocking_errors.append(
                f"Cued file {fp.name} overlapping cue windows: "
                f"'{prev_id}' [{prev_s}, {prev_e}] overlaps '{cur_id}' [{cur_s}, {cur_e}]."
            )
            file_ok = False

    return {
        "file": fp.name,
        "valid": file_ok,
        "cues": len(cues),
        "samples": len(raw.get("samples") or []) if isinstance(raw.get("samples"), list) else 0,
        "by_trigger": by_trigger,
    }


def run_data_audit(
    *,
    sessions_dirs: Sequence[str | Path] = (),
    negatives_dirs: Sequence[str | Path] = (),
    converted_dirs: Sequence[str | Path] = (),
    recall_dirs: Sequence[str | Path] = (),
    passive_dirs: Sequence[str | Path] = (),
    data_root: str | Path = "data",
    sealed_root: Optional[str | Path] = None,
    min_cues_per_trigger: int = 50,
    min_train_windows: int = 300,
    seq_len: int = 5,
) -> Dict[str, Any]:
    """Run read-only data integrity & readiness audit across all AeroHand dataset folders."""
    blocking_errors: List[str] = []
    readiness_gaps: List[str] = []
    warnings: List[str] = []

    source_inputs: Dict[str, List[Path]] = {
        "sessions": collect_json_paths(_existing_paths(sessions_dirs)),
        "negatives": collect_json_paths(_existing_paths(negatives_dirs)),
        "converted": collect_json_paths(_existing_paths(converted_dirs)),
        "recall": collect_json_paths(_existing_paths(recall_dirs)),
        "passive": collect_json_paths(_existing_paths(passive_dirs)),
    }

    all_paths: List[Path] = []
    for plist in source_inputs.values():
        all_paths.extend(plist)

    # 1. Discover sealed manifests & check SHA-256 overlap
    sealed_hashes: Set[str] = set()
    manifests = find_sealed_manifests(
        data_root=data_root,
        input_paths=all_paths,
        sealed_root=sealed_root,
    )
    for mf in manifests:
        try:
            sealed_hashes.update(read_sealed_manifest_hashes(mf))
        except ValueError as exc:
            blocking_errors.append(str(exc))

    # 2. Duplicate SHA-256 & sealed overlap scan across all audited files
    hash_to_paths: Dict[str, List[str]] = {}
    sealed_overlaps: List[Dict[str, str]] = []
    for src_name, plist in source_inputs.items():
        for fp in plist:
            try:
                dig = sha256_file(fp).lower()
            except OSError as exc:
                blocking_errors.append(f"Unable to hash {fp}: {exc}")
                continue
            hash_to_paths.setdefault(dig, []).append(f"{src_name}:{fp.name}")
            if dig in sealed_hashes:
                sealed_overlaps.append({"source": src_name, "file": str(fp), "sha256": dig})
                blocking_errors.append(
                    f"Sealed held-out overlap: {src_name}/{fp.name} matches sealed_manifest.json ({dig[:12]})."
                )

    duplicates: List[Dict[str, Any]] = []
    for dig, names in sorted(hash_to_paths.items()):
        if len(names) > 1:
            duplicates.append({"sha256": dig, "files": names})
            blocking_errors.append(f"Duplicate SHA-256 ({dig[:12]}) across files: {', '.join(names)}")

    # 3. Audit corpus files (sessions, negatives, converted, passive)
    by_class: Dict[str, Dict[str, int]] = {
        p.value: {
            "total": 0,
            "Right": 0,
            "Left": 0,
            "target": COLLECTOR_CLASS_TARGETS[p.value]["total"],
            "target_right": COLLECTOR_CLASS_TARGETS[p.value]["Right"],
            "target_left": COLLECTOR_CLASS_TARGETS[p.value]["Left"],
        }
        for p in ALL_POSE_NAMES
    }
    fresh_by_class: Dict[str, Dict[str, int]] = {
        p.value: {"total": 0, "Right": 0, "Left": 0} for p in ALL_POSE_NAMES
    }
    by_angle: Dict[str, int] = {a: 0 for a in WRIST_ANGLE_BUCKETS}
    by_distance: Dict[str, int] = {d: 0 for d in DISTANCE_BUCKETS}
    by_lighting: Dict[str, int] = {lt: 0 for lt in LIGHTING_BUCKETS}
    by_hard_neg: Dict[str, int] = {g: 0 for g in HARD_NEGATIVE_GROUPS}
    train_windows_by_class: Dict[str, int] = {p.value: 0 for p in ALL_POSE_NAMES}
    idle_hours_by_source: Dict[str, float] = {
        "sessions": 0.0,
        "negatives": 0.0,
        "converted": 0.0,
        "passive": 0.0,
    }
    all_dts_ms: List[float] = []
    total_gaps_gt_100ms = 0
    total_frames_all = 0
    fresh_total_frames = 0
    world_landmarks_present = 0
    world_landmarks_checked = 0
    v2_files_count = 0
    non_v2_files: List[str] = []

    for src_name in ("sessions", "negatives", "converted", "passive"):
        force_idle = src_name in ("negatives", "passive")
        for fp in source_inputs[src_name]:
            if is_cued_events_recording(fp):
                warnings.append(
                    f"Misplaced cued_events file '{fp.name}' inside '{src_name}' directory (ignored for corpus)."
                )
                continue
            try:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                frames, _rec_meta = load_recording_frames(fp, force_idle=force_idle)
            except Exception as exc:
                blocking_errors.append(f"Corrupt or unreadable recording {src_name}/{fp.name}: {exc}")
                continue

            if not isinstance(raw, dict):
                blocking_errors.append(f"Recording {src_name}/{fp.name} root must be a JSON object.")
                continue

            schema = str(raw.get("schema") or "")
            is_v2 = schema == "aerohand_pinch_v2"
            if is_v2:
                v2_files_count += 1
            else:
                non_v2_files.append(f"{src_name}:{fp.name}")

            meta = raw.get("session_meta") if isinstance(raw.get("session_meta"), dict) else {}
            angle = str(meta.get("wrist_angle", "0")).replace("deg", "").strip()
            dist = str(meta.get("distance", "mid")).strip().lower()
            light = str(meta.get("lighting", "bright")).strip().lower()

            # Check world_landmarks presence in AeroHand v2 recordings
            file_present_frames = 0
            file_world_frames = 0
            for fr in frames:
                if not fr["hand_present"] or not fr["landmarks_2d"]:
                    continue
                file_present_frames += 1
                world_landmarks_checked += 1
                if fr.get("world_landmarks"):
                    file_world_frames += 1
                    world_landmarks_present += 1

                pname = str(fr["label_name"])
                hand = "Left" if fr["handedness"] == "Left" else "Right"
                if pname in by_class:
                    by_class[pname]["total"] += 1
                    by_class[pname][hand] += 1
                    total_frames_all += 1
                    if src_name in ("sessions", "negatives"):
                        fresh_by_class[pname]["total"] += 1
                        fresh_by_class[pname][hand] += 1
                        fresh_total_frames += 1
                        if angle in by_angle:
                            by_angle[angle] += 1
                        if dist in by_distance:
                            by_distance[dist] += 1
                        if light in by_lighting:
                            by_lighting[light] += 1

            if is_v2 and file_present_frames > 0 and file_world_frames == 0:
                blocking_errors.append(
                    f"Schema v2 recording {src_name}/{fp.name} has 0/{file_present_frames} frames with world_landmarks."
                )

            # Hard-negative group tags from raw samples
            raw_samples = raw.get("samples")
            if isinstance(raw_samples, list):
                for s in raw_samples:
                    if isinstance(s, dict):
                        hng = str(s.get("hard_negative_group") or "").strip()
                        if hng in by_hard_neg:
                            by_hard_neg[hng] += 1

            # Exposure & timing (respect deliberate_marks_ms if present on passive files)
            raw_marks = raw.get("deliberate_marks_ms") or meta.get("deliberate_marks_ms") or []
            marks_ms = [float(m) for m in raw_marks if isinstance(m, (int, float))] if isinstance(raw_marks, list) else []
            timing = _compute_idle_hours_and_timing(frames, deliberate_marks_ms=marks_ms)
            idle_hours_by_source[src_name] += timing["hand_present_idle_hours"]
            total_gaps_gt_100ms += timing["gaps_gt_100ms"]
            all_dts_ms.extend(timing["sample_dts_ms"])

            if src_name in ("sessions", "negatives", "converted"):
                w_cnts = _estimate_train_windows(
                    frames,
                    seq_len=seq_len,
                    deliberate_marks_ms=marks_ms,
                )
                for k, v in w_cnts.items():
                    train_windows_by_class[k] += v

    # 4. Corpus readiness & balance checks
    angle_shares = {
        a: round((by_angle[a] / fresh_total_frames), 4) if fresh_total_frames > 0 else 0.0
        for a in WRIST_ANGLE_BUCKETS
    }
    if fresh_total_frames < CORPUS_TOTAL_TARGET:
        readiness_gaps.append(
            f"Fresh AeroHand corpus has {fresh_total_frames}/{CORPUS_TOTAL_TARGET} frames."
        )
    low_angles = {a: sh for a, sh in angle_shares.items() if sh < MIN_ANGLE_SHARE}
    if low_angles:
        readiness_gaps.append(
            f"Wrist angle buckets below {int(MIN_ANGLE_SHARE * 100)}% share: {low_angles}."
        )
    low_hng = {g: c for g, c in by_hard_neg.items() if c < HARD_NEGATIVE_MIN_PER_GROUP}
    if low_hng:
        readiness_gaps.append(
            f"Hard-negative groups below {HARD_NEGATIVE_MIN_PER_GROUP} frames: {low_hng}."
        )
    low_win = {k: v for k, v in train_windows_by_class.items() if v < int(min_train_windows)}
    if low_win:
        readiness_gaps.append(
            f"Classes below {min_train_windows} estimated train windows: {low_win}."
        )
        warnings.append(f"Low train-window classes (< {min_train_windows}): {low_win}")

    # Hand balance (>70/30 warning)
    hand_imbalanced: Dict[str, Dict[str, Any]] = {}
    for pname, cnts in by_class.items():
        tot = cnts["total"]
        if tot > 0:
            r_share = cnts["Right"] / float(tot)
            l_share = cnts["Left"] / float(tot)
            if r_share > 0.70 or l_share > 0.70:
                hand_imbalanced[pname] = {
                    "Right": cnts["Right"],
                    "Left": cnts["Left"],
                    "right_share": round(r_share, 3),
                }
    if hand_imbalanced:
        warnings.append(
            f"Hand balance >70/30 skew in {len(hand_imbalanced)} class(es): {list(hand_imbalanced.keys())}"
        )

    # 5. Audit cued recall files
    cued_by_trigger: Dict[str, int] = {t: 0 for t in CUED_TRIGGER_ORDER}
    cued_files_report: List[Dict[str, Any]] = []
    for fp in source_inputs["recall"]:
        f_rep = _audit_cued_file(fp, blocking_errors=blocking_errors, warnings=warnings)
        cued_files_report.append(f_rep)
        for trig, cnt in f_rep["by_trigger"].items():
            if trig in cued_by_trigger:
                cued_by_trigger[trig] += cnt

    low_cues = {t: c for t, c in cued_by_trigger.items() if c < int(min_cues_per_trigger)}
    if low_cues:
        readiness_gaps.append(
            f"Cued recall triggers below target {min_cues_per_trigger} cues: {low_cues}."
        )

    med_dt_all = float(np.median(all_dts_ms)) if all_dts_ms else 0.0
    overall_fps = round(1000.0 / med_dt_all, 2) if med_dt_all > 1e-6 else 0.0
    idle_rounded = {k: round(v, 6) for k, v in idle_hours_by_source.items()}
    idle_rounded["total"] = round(sum(idle_hours_by_source.values()), 6)

    integrity_ok = len(blocking_errors) == 0
    ready_for_training = integrity_ok and (len(readiness_gaps) == 0)

    return {
        "schema": "aerohand_data_audit_v1",
        "integrity_ok": integrity_ok,
        "ready_for_training": ready_for_training,
        "passed": ready_for_training,
        "blocking_errors": blocking_errors,
        "readiness_gaps": readiness_gaps,
        "warnings": warnings,
        "sealed_manifests_checked": [str(m) for m in manifests],
        "sealed_overlaps": sealed_overlaps,
        "duplicates": duplicates,
        "files_count": {k: len(v) for k, v in source_inputs.items()},
        "schema_summary": {
            "v2_files": v2_files_count,
            "non_v2_files": non_v2_files,
            "world_landmarks_ratio": (
                round(world_landmarks_present / float(world_landmarks_checked), 4)
                if world_landmarks_checked > 0
                else 0.0
            ),
        },
        "corpus": {
            "total_frames_all_sources": total_frames_all,
            "fresh_aerohand_frames": fresh_total_frames,
            "target_fresh_frames": CORPUS_TOTAL_TARGET,
            "by_class": by_class,
            "fresh_by_class": fresh_by_class,
            "train_windows_by_class": train_windows_by_class,
            "min_train_windows": int(min_train_windows),
            "by_angle": by_angle,
            "angle_shares": angle_shares,
            "by_distance": by_distance,
            "by_lighting": by_lighting,
            "by_hard_negative_group": by_hard_neg,
            "hand_imbalanced_classes": hand_imbalanced,
        },
        "cued_recall": {
            "files": cued_files_report,
            "total_cues": sum(cued_by_trigger.values()),
            "min_cues_per_trigger": int(min_cues_per_trigger),
            "by_trigger": cued_by_trigger,
        },
        "fps_gaps": {
            "median_fps": overall_fps,
            "gaps_gt_100ms": total_gaps_gt_100ms,
        },
        "idle_hours_by_source": idle_rounded,
    }


def format_summary_table(report: Dict[str, Any]) -> str:
    """Format a compact terminal summary table for humans alongside JSON output."""
    lines: List[str] = []
    status = "PASS (READY)" if report["ready_for_training"] else (
        "PARTIAL (INTEGRITY OK)" if report["integrity_ok"] else "FAIL (INTEGRITY ERRORS)"
    )
    lines.append(f"=== AeroHand Data Audit: {status} ===")
    c = report["corpus"]
    lines.append(
        f"Fresh AeroHand frames: {c['fresh_aerohand_frames']} / {c['target_fresh_frames']} | "
        f"All sources: {c['total_frames_all_sources']} | "
        f"World 3D: {report['schema_summary']['world_landmarks_ratio'] * 100:.1f}%"
    )
    lines.append(
        f"FPS median: {report['fps_gaps']['median_fps']} | Gaps >100ms: {report['fps_gaps']['gaps_gt_100ms']} | "
        f"Idle exposure (h): {report['idle_hours_by_source']}"
    )
    lines.append("Class counts (total / R / L / est.windows):")
    for pname, d in c["by_class"].items():
        w = c["train_windows_by_class"].get(pname, 0)
        lines.append(f"  {pname:26s} {d['total']:7d} (R:{d['Right']:6d} L:{d['Left']:6d}) win={w:6d}")
    cr = report["cued_recall"]
    lines.append(f"Cued recall ({cr['total_cues']} cues, target >= {cr['min_cues_per_trigger']}/trigger):")
    for trig, cnt in cr["by_trigger"].items():
        lines.append(f"  {trig:20s} {cnt:5d}")
    if report["blocking_errors"]:
        lines.append(f"BLOCKING ERRORS ({len(report['blocking_errors'])}):")
        for e in report["blocking_errors"][:15]:
            lines.append(f"  [ERROR] {e}")
    if report["readiness_gaps"]:
        lines.append(f"READINESS GAPS ({len(report['readiness_gaps'])}):")
        for g in report["readiness_gaps"][:15]:
            lines.append(f"  [GAP]   {g}")
    if report["warnings"]:
        lines.append(f"WARNINGS ({len(report['warnings'])}):")
        for w in report["warnings"][:15]:
            lines.append(f"  [WARN]  {w}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Read-only AeroHand dataset, negative-exposure & cued-recall auditor (V060-07)."
    )
    parser.add_argument("--data-root", type=str, default="data", help="Base data root directory (default: data)")
    parser.add_argument("--sessions", nargs="*", default=None, help="Session directories/files (default: <data-root>/sessions)")
    parser.add_argument("--negatives", nargs="*", default=None, help="Hard-negative directories/files (default: <data-root>/negatives)")
    parser.add_argument("--converted", nargs="*", default=None, help="Optional converted directories/files (default: <data-root>/converted)")
    parser.add_argument("--recall", nargs="*", default=None, help="Cued recall directories/files (default: <data-root>/recall)")
    parser.add_argument("--passive", nargs="*", default=None, help="Passive idle directories/files (default: <data-root>/passive)")
    parser.add_argument("--sealed-root", type=str, default=None, help="Optional directory scanned for sealed_manifest.json")
    parser.add_argument("--min-cues-per-trigger", type=int, default=50, help="Target cues per trigger in data/recall (default: 50)")
    parser.add_argument("--min-train-windows", type=int, default=300, help="Target train windows per class (default: 300)")
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length for train window estimation (default: 5)")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 when integrity_ok is True even if corpus/cue targets are still incomplete",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only (no summary table)")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save JSON report")
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    explicit_any = any(
        x is not None for x in (args.sessions, args.negatives, args.converted, args.recall, args.passive)
    )
    sessions_dirs = args.sessions if args.sessions is not None else ([] if explicit_any else [root / "sessions"])
    negatives_dirs = args.negatives if args.negatives is not None else ([] if explicit_any else [root / "negatives"])
    converted_dirs = args.converted if args.converted is not None else ([] if explicit_any else [root / "converted"])
    recall_dirs = args.recall if args.recall is not None else ([] if explicit_any else [root / "recall"])
    passive_dirs = args.passive if args.passive is not None else ([] if explicit_any else [root / "passive"])

    report = run_data_audit(
        sessions_dirs=sessions_dirs,
        negatives_dirs=negatives_dirs,
        converted_dirs=converted_dirs,
        recall_dirs=recall_dirs,
        passive_dirs=passive_dirs,
        data_root=args.data_root,
        sealed_root=args.sealed_root,
        min_cues_per_trigger=args.min_cues_per_trigger,
        min_train_windows=args.min_train_windows,
        seq_len=args.seq_len,
    )

    out_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        out_fp = Path(args.out)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(out_text, encoding="utf-8")

    if args.json:
        print(out_text)
    else:
        print(format_summary_table(report), file=sys.stderr)
        print(out_text)

    if not report["integrity_ok"]:
        sys.exit(1)
    if not report["ready_for_training"] and not args.allow_partial:
        sys.exit(1)


if __name__ == "__main__":
    main()
