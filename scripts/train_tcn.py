#!/usr/bin/env python3
"""
Train a Causal 1D-TCN Gesture Classifier for AeroHand and Export to ONNX + Normalizer JSON + Metrics.

Matches Airpoint's `ONNXGestureClassifier` architecture (5-frame sliding causal window,
dilated residual 1D convolutions, 7 canonical pose classes) while supporting BOTH:
  - `--feature-set invariant348` (default: 3D rotation/translation/scale-invariant palm-frame features)
  - `--feature-set airmouse885`  (Airpoint 885-dim full z-depth feature layout for A/B comparison)

Zero-leakage evaluation & calibration (P2-01):
  - Splits strictly by `(subject, condition_id)` metadata group (fallback: recording file) and
    validates that all 7 canonical classes are represented in validation.
  - Breaks sliding windows across timestamp gaps > 100 ms or hand_id changes.
  - Landmark-level augmentations BEFORE feature extraction (z-rotation +/-15 deg, scale 0.85-1.15,
    Gaussian noise sigma=0.003, 1-frame drop/duplicate jitter, left-right mirror).
  - Focal loss (gamma=2) or label-smoothing / weighted CE; selects best epoch by
    `macro_f1 - 0.5 * idle_fp_rate`.
  - Post-hoc validation temperature scaling saved to `*_normalizer.json`.
  - Exports `metrics.json` with per-class P/R/F1, confusion matrix, and confusable-pair analysis.

Usage:
  uv sync --extra train
  uv run python scripts/train_tcn.py --sessions data/sessions --out-dir models --feature-set invariant348
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName
from aerohand.ml.features_3d import FEATURE_DIM_3D, extract_invariant_3d_features
from aerohand.ml.features_885 import AIRMOUSE_885_DIM, extract_airmouse_885_features
from aerohand.ml.provenance import compute_onnx_model_hash as _shared_compute_onnx_model_hash
from aerohand.ml.recording_io import (
    assert_not_heldout,
    collect_json_paths,
    exclude_cued_events,
    exclude_sealed_content,
    load_recording_frames,
    sha256_file,
)

POSE_TO_IDX: Dict[str, int] = {p.value: idx for idx, p in enumerate(ALL_POSE_NAMES)}
NUM_CLASSES: int = len(ALL_POSE_NAMES)

CONFUSABLE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("thumb_middle_pinch", "thumb_index_middle_pinch"),
    ("thumb_ring_pinch", "thumb_pinky_pinch"),
    ("thumb_pinky_pinch", "thumb_pinky_base"),
    ("thumb_index_pinch", "idle"),
)


class ParsedSample:
    __slots__ = ("label_idx", "lms2d", "lms3d", "handedness", "hand_id", "timestamp_ms")

    def __init__(
        self,
        label_idx: int,
        lms2d: List[Landmark],
        lms3d: Optional[List[Landmark]],
        handedness: Handedness,
        hand_id: str,
        timestamp_ms: Optional[float],
    ) -> None:
        self.label_idx = label_idx
        self.lms2d = lms2d
        self.lms3d = lms3d
        self.handedness = handedness
        self.hand_id = hand_id
        self.timestamp_ms = timestamp_ms


class SessionBundle:
    __slots__ = ("name", "group_key", "windows", "labels", "scenario", "subject")

    def __init__(
        self,
        name: str,
        group_key: str,
        windows: List[List[ParsedSample]],
        labels: np.ndarray,
        *,
        scenario: str = "",
        subject: str = "anon",
    ) -> None:
        self.name = name
        self.group_key = group_key
        self.windows = windows
        self.labels = labels
        self.scenario = scenario or name
        self.subject = subject or "anon"


def _to_landmarks(flat_or_list: object) -> List[Landmark]:
    if not flat_or_list:
        return []
    if isinstance(flat_or_list, list) and flat_or_list:
        first = flat_or_list[0]
        if isinstance(first, (int, float)) and len(flat_or_list) >= 63:
            return [
                Landmark(
                    x=float(flat_or_list[i]),
                    y=float(flat_or_list[i + 1]),
                    z=float(flat_or_list[i + 2]),
                )
                for i in range(0, 63, 3)
            ]
        if isinstance(first, dict) and len(flat_or_list) >= 21:
            return [
                Landmark(x=float(p["x"]), y=float(p["y"]), z=float(p.get("z", 0.0)))
                for p in flat_or_list[:21]
            ]
    return []


def _resolve_sample_label(sample: Dict[str, Any]) -> Optional[str]:
    raw = sample.get("label_name")
    if raw is None:
        raw = sample.get("labelName")
    if raw is None:
        return None
    return str(raw).strip()


def _resolve_sample_timestamp_ms(sample: Dict[str, Any]) -> Optional[float]:
    raw = sample.get("timestamp_ms")
    if raw is None:
        raw = sample.get("timestamp")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _extract_feature(
    lms2d: Sequence[Landmark],
    lms3d: Optional[Sequence[Landmark]],
    handedness: Handedness,
    feature_set: str,
) -> Optional[np.ndarray]:
    if feature_set == "invariant348":
        return extract_invariant_3d_features(lms2d, lms3d, handedness=handedness)
    return extract_airmouse_885_features(lms2d, lms3d, handedness=handedness)


def _segment_samples_into_windows(
    samples: Sequence[Dict[str, Any]],
    *,
    seq_len: int = 5,
    force_idle: bool = False,
    max_gap_ms: float = 100.0,
    skip_stats: Optional[Dict[str, int]] = None,
) -> Tuple[List[List[ParsedSample]], np.ndarray]:
    """Parse raw sample dicts and build gap-free, single-hand sliding windows of length `seq_len`."""
    stats = skip_stats if skip_stats is not None else {}

    def _bump(reason: str) -> None:
        stats[reason] = stats.get(reason, 0) + 1

    runs: List[List[ParsedSample]] = []
    cur_run: List[ParsedSample] = []

    for s in samples:
        if not isinstance(s, dict):
            _bump("non_dict")
            continue

        if force_idle:
            label_idx = 0
        else:
            lname = _resolve_sample_label(s)
            if lname is None and s.get("label") == 0:
                lname = PoseName.IDLE.value
            if lname is None or lname not in POSE_TO_IDX:
                _bump("unknown_label")
                continue
            label_idx = POSE_TO_IDX[lname]

        raw_2d = s.get("landmarks_2d")
        if raw_2d is None:
            raw_2d = s.get("landmarks")
        lms2d = _to_landmarks(raw_2d)
        if len(lms2d) < 21:
            _bump("bad_landmarks")
            continue

        raw_3d = s.get("world_landmarks")
        if raw_3d is None:
            raw_3d = s.get("worldLandmarks")
        lms3d = _to_landmarks(raw_3d) or None
        handed: Handedness = "Left" if str(s.get("handedness", "Right")) == "Left" else "Right"
        hand_id = str(s.get("hand_id") or handed)
        ts_ms = _resolve_sample_timestamp_ms(s)

        parsed = ParsedSample(
            label_idx=label_idx,
            lms2d=lms2d,
            lms3d=lms3d,
            handedness=handed,
            hand_id=hand_id,
            timestamp_ms=ts_ms,
        )

        # Break contiguous run on hand_id change or timestamp gap > max_gap_ms
        if cur_run:
            prev = cur_run[-1]
            hand_changed = prev.hand_id != parsed.hand_id
            time_gap = (
                max_gap_ms > 0.0
                and prev.timestamp_ms is not None
                and parsed.timestamp_ms is not None
                and (
                    (parsed.timestamp_ms - prev.timestamp_ms) > max_gap_ms
                    or (parsed.timestamp_ms < prev.timestamp_ms)
                )
            )
            if hand_changed or time_gap:
                runs.append(cur_run)
                cur_run = []
        cur_run.append(parsed)

    if cur_run:
        runs.append(cur_run)

    windows: List[List[ParsedSample]] = []
    ys: List[int] = []
    for run in runs:
        if len(run) < seq_len:
            continue
        for i in range(len(run) - seq_len + 1):
            win = run[i : i + seq_len]
            windows.append(win)
            ys.append(win[-1].label_idx)
    return windows, np.asarray(ys, dtype=np.int64)


def _featurize_windows(
    windows: Sequence[Sequence[ParsedSample]],
    *,
    feature_set: str = "invariant348",
    seq_len: int = 5,
    skip_stats: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    feat_dim = FEATURE_DIM_3D if feature_set == "invariant348" else AIRMOUSE_885_DIM
    stats = skip_stats if skip_stats is not None else {}

    seqs_out: List[np.ndarray] = []
    ys_out: List[int] = []
    for win in windows:
        if len(win) != seq_len:
            continue
        vecs: List[np.ndarray] = []
        ok = True
        for item in win:
            vec = _extract_feature(item.lms2d, item.lms3d, item.handedness, feature_set)
            if vec is None or vec.shape[0] != feat_dim or not np.all(np.isfinite(vec)):
                ok = False
                stats["non_finite_features"] = stats.get("non_finite_features", 0) + 1
                break
            vecs.append(vec)
        if ok:
            seqs_out.append(np.stack(vecs, axis=0))
            ys_out.append(win[-1].label_idx)

    if not seqs_out:
        return np.zeros((0, seq_len, feat_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(seqs_out, axis=0).astype(np.float32), np.asarray(ys_out, dtype=np.int64)


def samples_to_sequences(
    samples: Sequence[Dict[str, Any]],
    *,
    feature_set: str = "invariant348",
    seq_len: int = 5,
    force_idle: bool = False,
    max_gap_ms: float = 100.0,
    skip_stats: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw sample dicts (snake_case v1/v2 or SDK camelCase v4.0) into gap-safe `(N, seq_len, F)` arrays."""
    windows, _ = _segment_samples_into_windows(
        samples,
        seq_len=seq_len,
        force_idle=force_idle,
        max_gap_ms=max_gap_ms,
        skip_stats=skip_stats,
    )
    return _featurize_windows(windows, feature_set=feature_set, seq_len=seq_len, skip_stats=skip_stats)


def load_session_sequences(
    session_path: Path,
    *,
    feature_set: str = "invariant348",
    seq_len: int = 5,
    force_idle: bool = False,
    max_gap_ms: float = 100.0,
    skip_stats: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one JSON session into gap-safe sliding `(N, seq_len, F)` sequences and labels `(N,)`."""
    data = json.loads(session_path.read_text(encoding="utf-8"))
    samples = data.get("samples") or data.get("frames") or []
    return samples_to_sequences(
        samples,
        feature_set=feature_set,
        seq_len=seq_len,
        force_idle=force_idle,
        max_gap_ms=max_gap_ms,
        skip_stats=skip_stats,
    )


def _session_group_key(data: Dict[str, Any], fallback_name: str) -> str:
    meta = data.get("session_meta")
    if isinstance(meta, dict) and meta:
        subj = str(meta.get("subject") or "").strip()
        cond = str(meta.get("condition_id") or "").strip()
        if not cond:
            parts = [
                str(meta.get(k) or "").strip()
                for k in ("wrist_angle", "distance", "lighting", "hand", "camera")
                if meta.get(k) not in (None, "")
            ]
            cond = "_".join(parts)
        if subj or cond:
            return f"{subj or 'anon'}::{cond or fallback_name}"
    return fallback_name


def load_session_bundle(
    session_path: Path,
    *,
    seq_len: int = 5,
    force_idle: bool = False,
    max_gap_ms: float = 100.0,
    skip_stats: Optional[Dict[str, int]] = None,
) -> SessionBundle:
    assert_not_heldout(session_path)
    frames, rec_meta = load_recording_frames(session_path, force_idle=force_idle)
    # Convert normalized `recording_io` frames into sample dicts compatible with `_segment_samples_into_windows`
    norm_samples: List[Dict[str, Any]] = []
    for fr in frames:
        if not fr["hand_present"] or not fr["landmarks_2d"]:
            continue
        norm_samples.append(
            {
                "timestamp_ms": fr["timestamp_ms"],
                "label_name": fr["label_name"],
                "handedness": fr["handedness"],
                "hand_id": f"{fr['hand_id']}#s{fr.get('segment_id', 0)}",
                "landmarks_2d": [c for lm in fr["landmarks_2d"][:21] for c in (lm.x, lm.y, lm.z)],
                "world_landmarks": (
                    [c for lm in fr["world_landmarks"][:21] for c in (lm.x, lm.y, lm.z)]
                    if fr["world_landmarks"]
                    else []
                ),
            }
        )
    windows, ys = _segment_samples_into_windows(
        norm_samples,
        seq_len=seq_len,
        force_idle=force_idle,
        max_gap_ms=max_gap_ms,
        skip_stats=skip_stats,
    )
    prefix = "neg:" if force_idle else ""
    smeta = rec_meta.get("session_meta") or {}
    group_key = prefix + _session_group_key({"session_meta": smeta}, session_path.name)
    scenario = str(rec_meta.get("scenario") or session_path.stem)
    subject = str(smeta.get("subject") or "anon").strip() or "anon"
    return SessionBundle(
        name=f"{prefix}{session_path.name}",
        group_key=group_key,
        windows=windows,
        labels=ys,
        scenario=scenario,
        subject=subject,
    )


DEFAULT_UNASSIGNED_SUBJECTS = frozenset({"", "user", "anon", "default", "unknown"})


def summarize_bundles_by_subject(bundles: Sequence[SessionBundle]) -> Dict[str, Dict[str, int]]:
    """Return `{subject: {"bundles": n_files, "windows": n_windows}}` summary table."""
    table: Dict[str, Dict[str, int]] = {}
    for b in bundles:
        subj = (b.subject or "anon").strip() or "anon"
        entry = table.setdefault(subj, {"bundles": 0, "windows": 0})
        entry["bundles"] += 1
        entry["windows"] += len(b.windows)
    return dict(sorted(table.items()))


def split_subject_heldout(
    bundles: Sequence[SessionBundle],
    *,
    primary_subject: str = "",
    heldout_subjects: Sequence[str] = (),
) -> Tuple[List[SessionBundle], List[SessionBundle]]:
    """
    Separate non-primary or explicitly held-out volunteer subjects out of dev training
    so subject generalization is evaluated strictly out-of-subject (V050-02).
    """
    primary_norm = primary_subject.strip().lower()
    ho_set = {s.strip().lower() for s in heldout_subjects if s and s.strip()}
    if not primary_norm and not ho_set:
        return list(bundles), []

    dev_bundles: List[SessionBundle] = []
    ho_bundles: List[SessionBundle] = []
    for b in bundles:
        subj = b.subject.strip().lower()
        is_ho = (subj in ho_set) or (bool(primary_norm) and subj != primary_norm)
        if is_ho:
            ho_bundles.append(b)
        else:
            dev_bundles.append(b)
    return dev_bundles, ho_bundles


def validate_subject_split(
    all_bundles: Sequence[SessionBundle],
    dev_bundles: Sequence[SessionBundle],
    ho_bundles: Sequence[SessionBundle],
    *,
    primary_subject: str = "",
    heldout_subjects: Sequence[str] = (),
) -> Dict[str, Any]:
    """
    Guard against silent data exclusion when `--primary-subject` mismatches collector subject
    names (V051-01). Raises `ValueError` if primary subject matches 0 bundles OR if
    placeholder/default subject names (`user`/`anon`) are silently demoted into held-out.
    """
    summary = summarize_bundles_by_subject(all_bundles)
    primary_norm = primary_subject.strip().lower()
    explicit_ho = {s.strip().lower() for s in heldout_subjects if s and s.strip()}
    if primary_norm and not dev_bundles:
        raise ValueError(
            f"Primary subject '{primary_subject}' matched 0 bundles! "
            f"Available subjects: {summary}. "
            "Set `collector.subject` or fix `session_meta.subject` before training."
        )
    demoted_defaults = sorted(
        {
            b.subject
            for b in ho_bundles
            if b.subject.strip().lower() in DEFAULT_UNASSIGNED_SUBJECTS
            and b.subject.strip().lower() not in explicit_ho
        }
    )
    if primary_norm and demoted_defaults:
        raise ValueError(
            f"Refusing silent subject exclusion: bundles with placeholder subject {demoted_defaults} "
            f"would be pushed out of primary '{primary_subject}' into subject-held-out! "
            f"Subject summary: {summary}. Update `collector.subject` / `session_meta.subject` "
            "(or pass `--heldout-subjects` explicitly if intended)."
        )
    return {
        "by_subject": summary,
        "dev_bundles": len(dev_bundles),
        "heldout_bundles": len(ho_bundles),
        "demoted_default_subjects": demoted_defaults,
    }


def balance_scenario_samples(
    bundles: Sequence[SessionBundle],
    *,
    max_scenario_share: float = 0.15,
    seed: int = 42,
) -> Tuple[List[SessionBundle], Dict[str, Any]]:
    """
    Per-scenario balanced idle sampler (V050-02):
    Caps every single scenario's share of `idle` (class 0) windows to `<= max_scenario_share`
    using exact water-filling subsampling while preserving 100% of positive pinch windows.
    """
    share = max(0.01, min(1.0, float(max_scenario_share)))
    # Collect idle window indices grouped by scenario
    idle_by_sc: Dict[str, List[Tuple[int, int]]] = {}
    for bi, b in enumerate(bundles):
        for wi, lbl in enumerate(b.labels.tolist()):
            if int(lbl) == 0:
                idle_by_sc.setdefault(b.scenario, []).append((bi, wi))

    total_idle_before = sum(len(v) for v in idle_by_sc.values())
    if total_idle_before == 0 or len(idle_by_sc) <= 1 or share >= 1.0:
        return list(bundles), {
            "total_idle_before": total_idle_before,
            "total_idle_after": total_idle_before,
            "max_share_before": 1.0 if total_idle_before > 0 else 0.0,
            "max_share_after": 1.0 if total_idle_before > 0 else 0.0,
            "capped_scenarios": {},
        }

    counts = {sc: len(idxs) for sc, idxs in idle_by_sc.items()}
    max_share_before = max(counts.values()) / float(total_idle_before)

    # Exact closed-form water-filling cap:
    # If `m` scenarios are capped at `L` and uncapped sum is `U`,
    # `L / (U + m * L) <= share`  <=>  `L <= share * U / (1 - m * share)`.
    n_sc = len(counts)
    if n_sc * share < 1.0:
        # Fewer than ceil(1 / share) scenarios: balance equally at min count
        min_c = min(counts.values())
        caps = {sc: min_c for sc in counts}
    else:
        capped: set[str] = set()
        limit = total_idle_before
        while True:
            uncapped_sum = sum(c for sc, c in counts.items() if sc not in capped)
            m = len(capped)
            if m == 0:
                cand_limit = int(math.floor(share * total_idle_before))
            else:
                denom = 1.0 - m * share
                if denom <= 1e-9:
                    cand_limit = min(counts.values())
                else:
                    cand_limit = int(math.floor((share * uncapped_sum) / denom))
            cand_limit = max(1, cand_limit)
            newly_capped = {sc for sc, c in counts.items() if sc not in capped and c > cand_limit}
            if not newly_capped:
                limit = cand_limit
                break
            capped.update(newly_capped)
        caps = {sc: min(c, limit) for sc, c in counts.items()}

    rng = np.random.default_rng(int(seed))
    keep_idle_by_bundle: Dict[int, set[int]] = {}
    capped_info: Dict[str, Dict[str, int]] = {}
    for sc, pairs in sorted(idle_by_sc.items()):
        cap_n = caps[sc]
        if len(pairs) > cap_n:
            chosen_idx = rng.choice(len(pairs), size=cap_n, replace=False)
            chosen = [pairs[int(i)] for i in sorted(chosen_idx.tolist())]
            capped_info[sc] = {"before": len(pairs), "after": cap_n}
        else:
            chosen = pairs
        for bi, wi in chosen:
            keep_idle_by_bundle.setdefault(bi, set()).add(wi)

    out_bundles: List[SessionBundle] = []
    total_idle_after = 0
    kept_counts_by_sc: Dict[str, int] = {}
    for bi, b in enumerate(bundles):
        keep_idle = keep_idle_by_bundle.get(bi, set())
        new_wins: List[List[ParsedSample]] = []
        new_lbls: List[int] = []
        for wi, (w, lbl) in enumerate(zip(b.windows, b.labels.tolist())):
            if int(lbl) != 0 or wi in keep_idle:
                new_wins.append(w)
                new_lbls.append(int(lbl))
                if int(lbl) == 0:
                    total_idle_after += 1
                    kept_counts_by_sc[b.scenario] = kept_counts_by_sc.get(b.scenario, 0) + 1
        if new_wins:
            out_bundles.append(
                SessionBundle(
                    name=b.name,
                    group_key=b.group_key,
                    windows=new_wins,
                    labels=np.asarray(new_lbls, dtype=np.int64),
                    scenario=b.scenario,
                    subject=b.subject,
                )
            )

    max_share_after = (
        max(kept_counts_by_sc.values()) / float(total_idle_after) if total_idle_after > 0 else 0.0
    )
    return out_bundles, {
        "total_idle_before": total_idle_before,
        "total_idle_after": total_idle_after,
        "max_share_before": round(max_share_before, 4),
        "max_share_after": round(max_share_after, 4),
        "capped_scenarios": capped_info,
    }


def _get_git_commit() -> str:
    """Return current short git HEAD commit hash (or 'unknown' outside a git worktree)."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        out = (res.stdout or "").strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def compute_onnx_model_hash(onnx_path: str | Path, normalizer_path: str | Path) -> str:
    """Compute 24-hex fingerprint identical to `aerohand.ml.provenance.compute_onnx_model_hash`."""
    return _shared_compute_onnx_model_hash(onnx_path, normalizer_path)


def augment_window(
    window: Sequence[ParsedSample],
    rng: np.random.Generator,
) -> List[ParsedSample]:
    """
    Apply landmark-level augmentations BEFORE feature extraction (V051-10):
      - Z-rotation +/-15 deg around wrist
      - Uniform scale 0.85..1.15
      - Palm-width-scaled Gaussian noise: shared per-window jitter `sigma = 0.01 * palm_width`
        plus smaller independent per-frame jitter `sigma = 0.002 * palm_width` (avoids artificial velocity)
      - 1-frame drop/duplicate temporal jitter (preserves exact sequence length & final label)
      - Random 50% Left<->Right X-mirror matching AeroHand handedness convention
    """
    seq_len = len(window)
    if seq_len == 0:
        return []

    angle_rad = math.radians(float(rng.uniform(-15.0, 15.0)))
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    scale = float(rng.uniform(0.85, 1.15))
    do_mirror = bool(rng.random() < 0.5)

    # Temporal jitter: with 50% prob, drop one earlier frame and duplicate a neighbor (keeps last label!)
    indices = list(range(seq_len))
    if seq_len >= 3 and rng.random() < 0.5:
        drop_i = int(rng.integers(0, seq_len - 1))
        dup_i = max(0, min(seq_len - 1, drop_i + (1 if rng.random() < 0.5 else -1)))
        indices[drop_i] = dup_i

    # Shared per-window unit noise (scaled per frame by palm_width ||LM5 - LM17||)
    n_pts = len(window[0].lms2d)
    win_noise_unit_2d = rng.normal(0.0, 0.01, size=(n_pts, 3))
    win_noise_unit_3d = rng.normal(0.0, 0.01, size=(n_pts, 3))

    def _xform_landmarks(lms: List[Landmark], *, is_world: bool) -> List[Landmark]:
        arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
        anchor = arr[0].copy()
        rel = (arr - anchor) * scale
        rx = rel[:, 0] * cos_a - rel[:, 1] * sin_a
        ry = rel[:, 0] * sin_a + rel[:, 1] * cos_a
        rel[:, 0] = rx
        rel[:, 1] = ry
        out_arr = rel + anchor
        if len(rel) >= 18:
            pw = float(np.linalg.norm(rel[5] - rel[17]))
        else:
            pw = 0.0
        if not math.isfinite(pw) or pw < 1e-5:
            pw = 0.08 if is_world else 0.12
        shared_unit = win_noise_unit_3d if is_world else win_noise_unit_2d
        if shared_unit.shape[0] != out_arr.shape[0]:
            shared_unit = rng.normal(0.0, 0.01, size=out_arr.shape)
        out_arr += shared_unit * pw
        out_arr += rng.normal(0.0, 0.002 * pw, size=out_arr.shape)
        if do_mirror:
            if is_world:
                out_arr[:, 0] = -out_arr[:, 0]
            else:
                out_arr[:, 0] = 1.0 - out_arr[:, 0]
        return [
            Landmark(x=float(out_arr[i, 0]), y=float(out_arr[i, 1]), z=float(out_arr[i, 2]), visibility=lms[i].visibility)
            for i in range(len(lms))
        ]

    aug_out: List[ParsedSample] = []
    for idx in indices:
        s = window[idx]
        new_hand: Handedness = ("Left" if s.handedness == "Right" else "Right") if do_mirror else s.handedness
        aug_out.append(
            ParsedSample(
                label_idx=s.label_idx,
                lms2d=_xform_landmarks(s.lms2d, is_world=False),
                lms3d=_xform_landmarks(s.lms3d, is_world=True) if s.lms3d is not None else None,
                handedness=new_hand,
                hand_id=new_hand,
                timestamp_ms=s.timestamp_ms,
            )
        )
    return aug_out


def split_dev_test_groups(
    bundles: Sequence[SessionBundle],
    *,
    test_ratio: float = 0.15,
    min_groups_for_dev_test: int = 4,
    num_classes: int = NUM_CLASSES,
) -> Tuple[List[SessionBundle], List[SessionBundle]]:
    """
    Carve out a disjoint `dev_test` group split when `>= min_groups_for_dev_test` distinct
    groups exist (V051-10). Never used for epoch selection or temperature calibration.
    Guarantees every class present in `bundles` keeps at least 2 groups in `dev_train_val`.
    """
    grouped: Dict[str, List[SessionBundle]] = {}
    for b in bundles:
        grouped.setdefault(b.group_key, []).append(b)

    group_keys = sorted(grouped.keys())
    if len(group_keys) < max(3, int(min_groups_for_dev_test)):
        return list(bundles), []

    group_classes: Dict[str, set[int]] = {}
    class_group_counts: Dict[int, int] = {}
    for gk in group_keys:
        cls_set: set[int] = set()
        for b in grouped[gk]:
            cls_set.update(int(x) for x in b.labels.tolist())
        group_classes[gk] = cls_set
        for c in cls_set:
            class_group_counts[c] = class_group_counts.get(c, 0) + 1

    target_test = max(1, int(round(len(group_keys) * max(0.05, min(0.4, float(test_ratio))))))
    test_keys: List[str] = []
    remaining = list(group_keys)
    covered: set[int] = set()

    while remaining and len(test_keys) < target_test and len(remaining) > 3:
        eligible = [
            gk for gk in remaining if all(class_group_counts[c] >= 3 for c in group_classes[gk])
        ]
        if not eligible:
            break
        best_gk = max(eligible, key=lambda gk: (len(group_classes[gk] - covered), gk))
        test_keys.append(best_gk)
        covered.update(group_classes[best_gk])
        for c in group_classes[best_gk]:
            class_group_counts[c] -= 1
        remaining.remove(best_gk)

    if not test_keys:
        return list(bundles), []

    test_key_set = set(test_keys)
    train_val_bundles = [b for gk in group_keys if gk not in test_key_set for b in grouped[gk]]
    dev_test_bundles = [b for gk in group_keys if gk in test_key_set for b in grouped[gk]]
    return train_val_bundles, dev_test_bundles


def split_session_groups(
    bundles: Sequence[SessionBundle],
    *,
    val_ratio: float = 0.2,
    num_classes: int = NUM_CLASSES,
) -> Tuple[List[SessionBundle], List[SessionBundle], List[str]]:
    """
    Group-stratified split by `group_key` (`(subject, condition_id)` or file) (V051-10).
    Ensures zero frame leakage across groups, NEVER leaves any class without training windows,
    and falls back to a class-stratified temporal block cut (with 5-window gap) for single-group data.
    Returns `(train_bundles, val_bundles, missing_val_class_names)`.
    """
    grouped: Dict[str, List[SessionBundle]] = {}
    for b in bundles:
        grouped.setdefault(b.group_key, []).append(b)

    group_keys = sorted(grouped.keys())
    if len(group_keys) < 2:
        # Single-group fallback: class-stratified chronological block cut with 5-window gap per class
        merged_wins = [w for b in bundles for w in b.windows]
        merged_ys = np.concatenate([b.labels for b in bundles], axis=0) if bundles else np.zeros((0,), dtype=np.int64)
        gap = 5
        tr_idxs: List[int] = []
        va_idxs: List[int] = []
        by_class: Dict[int, List[int]] = {}
        for idx, lbl in enumerate(merged_ys.tolist()):
            by_class.setdefault(int(lbl), []).append(idx)
        for _cls, idxs in sorted(by_class.items()):
            n = len(idxs)
            if n <= 1:
                tr_idxs.extend(idxs)
                continue
            val_n = max(1, int(round(n * val_ratio)))
            if val_n >= n:
                val_n = max(1, n - 1)
            eff_gap = min(gap, max(0, n - val_n - 1))
            tr_end = max(1, n - val_n - eff_gap)
            va_start = min(n, tr_end + eff_gap)
            tr_idxs.extend(idxs[:tr_end])
            va_idxs.extend(idxs[va_start:])
        tr_idxs.sort()
        va_idxs.sort()
        tr_wins = [merged_wins[i] for i in tr_idxs]
        tr_ys = np.asarray([int(merged_ys[i]) for i in tr_idxs], dtype=np.int64)
        va_wins = [merged_wins[i] for i in va_idxs]
        va_ys = np.asarray([int(merged_ys[i]) for i in va_idxs], dtype=np.int64)
        tr_b = SessionBundle("single:train", "single:train", tr_wins, tr_ys)
        va_b = SessionBundle("single:val", "single:val", va_wins, va_ys)
        val_present = set(int(x) for x in va_b.labels.tolist())
        missing = [ALL_POSE_NAMES[i].value for i in range(num_classes) if i not in val_present]
        return [tr_b], [va_b], missing

    target_val_groups = max(1, int(round(len(group_keys) * val_ratio)))
    min_train_groups = 2 if len(group_keys) >= 3 else 1
    group_classes: Dict[str, set[int]] = {}
    train_class_counts: Dict[int, int] = {}
    for gk in group_keys:
        cls_set: set[int] = set()
        for b in grouped[gk]:
            cls_set.update(int(x) for x in b.labels.tolist())
        group_classes[gk] = cls_set
        for c in cls_set:
            train_class_counts[c] = train_class_counts.get(c, 0) + 1

    # Greedy selection: NEVER remove the last remaining group containing any class from train!
    val_keys: List[str] = []
    covered: set[int] = set()
    remaining = list(group_keys)

    while remaining and (len(val_keys) < target_val_groups or len(covered) < num_classes):
        if len(remaining) <= min_train_groups:
            break
        eligible = [
            gk for gk in remaining if all(train_class_counts[c] > 1 for c in group_classes[gk])
        ]
        if not eligible:
            break
        best_gk = max(eligible, key=lambda gk: (len(group_classes[gk] - covered), gk))
        gain = len(group_classes[best_gk] - covered)
        if len(val_keys) >= target_val_groups and gain == 0:
            break
        val_keys.append(best_gk)
        covered.update(group_classes[best_gk])
        for c in group_classes[best_gk]:
            train_class_counts[c] -= 1
        remaining.remove(best_gk)

    val_key_set = set(val_keys)
    train_bundles = [b for gk in group_keys if gk not in val_key_set for b in grouped[gk]]
    val_bundles = [b for gk in group_keys if gk in val_key_set for b in grouped[gk]]

    val_present = set()
    for b in val_bundles:
        val_present.update(int(x) for x in b.labels.tolist())
    missing = [ALL_POSE_NAMES[i].value for i in range(num_classes) if i not in val_present]
    return train_bundles, val_bundles, missing


def validate_train_classes(
    train_bundles: Sequence[SessionBundle],
    *,
    min_windows: int = 300,
    allow_low_count: bool = False,
    allow_missing: bool = False,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, int]:
    """
    Verify that every canonical class has training windows (`>= min_windows` unless `allow_low_count=True`)
    so training never silently proceeds with a missing or severely starved class (V051-10).
    """
    counts_arr = np.zeros((num_classes,), dtype=np.int64)
    for b in train_bundles:
        if len(b.labels):
            counts_arr += np.bincount(b.labels, minlength=num_classes)[:num_classes]
    counts_map = {ALL_POSE_NAMES[i].value: int(counts_arr[i]) for i in range(num_classes)}
    missing = [k for k, v in counts_map.items() if v <= 0]
    if missing and not allow_missing:
        raise ValueError(
            f"Training split has 0 windows for canonical class(es): {missing}. "
            f"Counts: {counts_map}."
        )
    if not allow_low_count and min_windows > 0:
        low = {k: v for k, v in counts_map.items() if (v < int(min_windows)) and (not allow_missing or v > 0)}
        if low:
            raise ValueError(
                f"Training split has < {min_windows} windows for class(es): {low}. "
                "Record more samples (or pass --allow-low-train-counts / --min-train-windows for smoke runs)."
            )
    return counts_map


def validate_validation_classes(
    val_bundles: Sequence[SessionBundle],
    *,
    allow_missing: bool = False,
    num_classes: int = NUM_CLASSES,
) -> List[str]:
    """
    Verify that all `num_classes` canonical pose classes are present in validation bundles (V050-03).
    Returns missing class names; raises `ValueError` if any class is missing and `allow_missing=False`.
    """
    present: set[int] = set()
    for b in val_bundles:
        present.update(int(x) for x in b.labels.tolist())
    missing = [ALL_POSE_NAMES[i].value for i in range(num_classes) if i not in present]
    if missing and not allow_missing:
        raise ValueError(
            f"Validation split is missing canonical classes: {missing}. "
            "Record sessions covering all 7 classes (or pass --allow-missing-val-classes for smoke runs)."
        )
    return missing


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Any]:
    """Compute per-class P/R/F1, macro-F1, idle FP rate, confusion matrix, and confusable-pair counts."""
    y_t = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_t, y_p):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class: Dict[str, Dict[str, float | int]] = {}
    f1s: List[float] = []
    for idx in range(num_classes):
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        sup = int(cm[idx, :].sum())
        prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        per_class[ALL_POSE_NAMES[idx].value] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sup,
        }
        if sup > 0:
            f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    val_acc = float((y_t == y_p).mean()) if len(y_t) else 0.0

    idle_total = int(cm[0, :].sum())
    idle_fp = int(cm[0, 1:].sum())
    idle_fp_rate = (idle_fp / float(idle_total)) if idle_total > 0 else 0.0

    confusable: Dict[str, Dict[str, int]] = {}
    for a_name, b_name in CONFUSABLE_PAIRS:
        ia = POSE_TO_IDX[a_name]
        ib = POSE_TO_IDX[b_name]
        confusable[f"{a_name} <-> {b_name}"] = {
            f"{a_name}_as_{b_name}": int(cm[ia, ib]),
            f"{b_name}_as_{a_name}": int(cm[ib, ia]),
            "total_errors": int(cm[ia, ib] + cm[ib, ia]),
        }

    return {
        "val_accuracy": round(val_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "idle_fp_rate": round(idle_fp_rate, 4),
        "selection_score": round(macro_f1 - 0.5 * idle_fp_rate, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusable_pairs": confusable,
    }


def fit_temperature_scaling(val_logits: np.ndarray, y_val: np.ndarray) -> float:
    """Calibrate scalar softmax temperature `T > 0` minimizing validation NLL (pure NumPy)."""
    logits = np.asarray(val_logits, dtype=np.float64)
    targets = np.asarray(y_val, dtype=np.int64).reshape(-1)
    if logits.ndim != 2 or len(targets) == 0:
        return 1.0

    def _nll(temp: float) -> float:
        scaled = logits / temp
        scaled = scaled - np.max(scaled, axis=1, keepdims=True)
        log_probs = scaled - np.log(np.sum(np.exp(scaled), axis=1, keepdims=True))
        return float(-np.mean(log_probs[np.arange(len(targets)), targets]))

    grid = np.concatenate([np.linspace(0.25, 1.5, 51), np.linspace(1.55, 5.0, 70)])
    best_t = float(min(grid, key=_nll))
    fine = np.linspace(max(0.15, best_t - 0.1), best_t + 0.1, 41)
    best_t = float(min(fine, key=_nll))
    return round(best_t, 4)


AB_DEFAULT_SEEDS: Tuple[int, ...] = (42, 1337, 2026)


def aggregate_seed_metrics(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multi-seed evaluation metrics (mean/std) + average confusable-pair error totals."""
    if not runs:
        return {
            "num_seeds": 0,
            "macro_f1_mean": 0.0,
            "macro_f1_std": 0.0,
            "val_accuracy_mean": 0.0,
            "idle_fp_rate_mean": 0.0,
            "idle_fp_rate_std": 0.0,
            "selection_score_mean": 0.0,
            "ft_per_hour_mean": None,
            "ft_upper_bound_per_hour_mean": None,
            "confusable_pairs_mean": {},
            "runs": [],
        }

    f1s = [float(r.get("macro_f1", 0.0)) for r in runs]
    accs = [float(r.get("val_accuracy", 0.0)) for r in runs]
    ifps = [float(r.get("idle_fp_rate", 0.0)) for r in runs]
    sel_scores = [
        float(r.get("selection_score", float(r.get("macro_f1", 0.0)) - 0.5 * float(r.get("idle_fp_rate", 0.0))))
        for r in runs
    ]
    fts = [float(r["ft_per_hour"]) for r in runs if r.get("ft_per_hour") is not None]
    ft_ucbs = [
        float(r["ft_upper_bound_per_hour"])
        for r in runs
        if r.get("ft_upper_bound_per_hour") is not None
    ]

    pair_totals: Dict[str, List[float]] = {}
    for r in runs:
        cp = r.get("confusable_pairs") or {}
        if isinstance(cp, dict):
            for k, v in cp.items():
                if isinstance(v, dict):
                    pair_totals.setdefault(k, []).append(float(v.get("total_errors", 0.0)))

    return {
        "num_seeds": len(runs),
        "macro_f1_mean": round(float(np.mean(f1s)), 4),
        "macro_f1_std": round(float(np.std(f1s)), 4),
        "val_accuracy_mean": round(float(np.mean(accs)), 4),
        "idle_fp_rate_mean": round(float(np.mean(ifps)), 4),
        "idle_fp_rate_std": round(float(np.std(ifps)), 4),
        "selection_score_mean": round(float(np.mean(sel_scores)), 4),
        "ft_per_hour_mean": round(float(np.mean(fts)), 3) if fts else None,
        "ft_upper_bound_per_hour_mean": round(float(np.mean(ft_ucbs)), 3) if ft_ucbs else None,
        "confusable_pairs_mean": {
            k: round(float(np.mean(vs)), 2) for k, vs in pair_totals.items()
        },
        "runs": list(runs),
    }


def evaluate_ab_decision(
    summary_348: Dict[str, Any],
    summary_885: Dict[str, Any],
    *,
    min_f1_gain: float = 0.02,
) -> Dict[str, Any]:
    """
    Documented V040-03 / V050-03 feature-set selection rule:
      - Default to `invariant348` (3D rotation/scale invariant, compact).
      - Switch to `airmouse885` ONLY if it improves `macro_f1_mean` by strictly more than
        `min_f1_gain` (+0.02) AND does not increase `idle_fp_rate_mean`, `ft_per_hour_mean`,
        or `ft_upper_bound_per_hour_mean` (95% Poisson UCB).
    """
    f1_348 = float(summary_348.get("macro_f1_mean", 0.0))
    f1_885 = float(summary_885.get("macro_f1_mean", 0.0))
    fp_348 = float(summary_348.get("idle_fp_rate_mean", 0.0))
    fp_885 = float(summary_885.get("idle_fp_rate_mean", 0.0))
    sel_348 = float(summary_348.get("selection_score_mean", f1_348 - 0.5 * fp_348))
    sel_885 = float(summary_885.get("selection_score_mean", f1_885 - 0.5 * fp_885))
    ft_348 = summary_348.get("ft_per_hour_mean")
    ft_885 = summary_885.get("ft_per_hour_mean")
    ucb_348 = summary_348.get("ft_upper_bound_per_hour_mean")
    ucb_885 = summary_885.get("ft_upper_bound_per_hour_mean")

    f1_delta = round(f1_885 - f1_348, 4)
    fp_delta = round(fp_885 - fp_348, 4)
    sel_delta = round(sel_885 - sel_348, 4)
    ft_delta = (
        round(float(ft_885) - float(ft_348), 3)
        if (ft_348 is not None and ft_885 is not None)
        else None
    )
    ucb_delta = (
        round(float(ucb_885) - float(ucb_348), 3)
        if (ucb_348 is not None and ucb_885 is not None)
        else None
    )

    f1_beats = f1_delta > float(min_f1_gain)
    fp_ok = fp_885 <= fp_348 + 1e-9
    ft_ok = True if (ft_delta is None) else (ft_delta <= 1e-9)
    ucb_ok = True if (ucb_delta is None) else (ucb_delta <= 1e-9)

    if f1_beats and fp_ok and ft_ok and ucb_ok:
        winner = "airmouse885"
        reason = (
            f"airmouse885 beat invariant348 by +{f1_delta:.4f} macro-F1 (> {min_f1_gain:.2f}) "
            "without raising idle FP rate, FT/hour, or FT 95% UCB."
        )
    else:
        winner = "invariant348"
        reasons = []
        if not f1_beats:
            reasons.append(f"macro-F1 gain {f1_delta:+.4f} <= +{min_f1_gain:.2f}")
        if not fp_ok:
            reasons.append(f"idle_fp_rate increased by {fp_delta:+.4f}")
        if not ft_ok and ft_delta is not None:
            reasons.append(f"ft_per_hour increased by {ft_delta:+.3f}")
        if not ucb_ok and ucb_delta is not None:
            reasons.append(f"ft_upper_bound_per_hour increased by {ucb_delta:+.3f}")
        reason = "Retained default invariant348 (" + "; ".join(reasons) + ")."

    return {
        "winner": winner,
        "min_f1_gain_required": float(min_f1_gain),
        "macro_f1_delta": f1_delta,
        "idle_fp_delta": fp_delta,
        "selection_score_delta": sel_delta,
        "ft_per_hour_delta": ft_delta,
        "ft_upper_bound_delta": ucb_delta,
        "reason": reason,
    }


def build_ab_report(
    runs_348: Sequence[Dict[str, Any]],
    runs_885: Sequence[Dict[str, Any]],
    *,
    seeds: Sequence[int] = AB_DEFAULT_SEEDS,
    min_f1_gain: float = 0.02,
) -> Dict[str, Any]:
    """Build structured `ab_report.json` comparing `invariant348` vs `airmouse885` across seeds."""
    s348 = aggregate_seed_metrics(runs_348)
    s885 = aggregate_seed_metrics(runs_885)
    decision = evaluate_ab_decision(s348, s885, min_f1_gain=min_f1_gain)
    return {
        "schema": "aerohand_ab_report_v1",
        "seeds": [int(s) for s in seeds],
        "feature_sets": {
            "invariant348": s348,
            "airmouse885": s885,
        },
        "decision": decision,
    }


def collect_training_files(
    inputs: Sequence[str | Path],
    *,
    data_root: str | Path = "data",
    sealed_root: Optional[str | Path] = None,
) -> Tuple[List[Path], List[Path]]:
    """Collect training JSON files while excluding `cued_events` and sealed held-out SHA-256 files (V051-02 / V060-01a)."""
    paths_in = [Path(s) for s in inputs if s]
    for sp in paths_in:
        assert_not_heldout(sp)
    raw_files = collect_json_paths([sp for sp in paths_in if sp.exists()])
    raw_files, _ = exclude_cued_events(raw_files)
    return exclude_sealed_content(raw_files, data_root=data_root, sealed_root=sealed_root)


def validate_disjoint_ab_negatives(
    train_neg_paths: Sequence[Path],
    eval_neg_paths: Sequence[Path],
) -> None:
    """Refuse `--ab-eval-negatives` overlapping with training negatives by content SHA-256 (V051-02)."""
    if not train_neg_paths or not eval_neg_paths:
        return
    train_shas = {sha256_file(p).lower(): p for p in train_neg_paths if Path(p).is_file()}
    overlaps: List[str] = []
    for ep in eval_neg_paths:
        if not Path(ep).is_file():
            continue
        dig = sha256_file(ep).lower()
        if dig in train_shas:
            overlaps.append(f"{ep.name} == {train_shas[dig].name} ({dig[:12]})")
    if overlaps:
        raise ValueError(
            "REFUSED: --ab-eval-negatives overlaps with training negatives (in-sample leak): "
            + ", ".join(overlaps)
        )


def write_training_artifacts(
    *,
    out_dir: str | Path,
    name: str,
    onnx_path: str | Path,
    feature_set: str,
    feat_dim: int,
    seq_len: int,
    hidden: int,
    loss: str,
    augmented: bool,
    mean: np.ndarray,
    scale: np.ndarray,
    temperature: float,
    train_sequences: int,
    val_sequences: int,
    best_metrics: Dict[str, Any],
    balance_stats: Dict[str, Any],
    dev_test_metrics: Optional[Dict[str, Any]],
    subject_ho_metrics: Optional[Dict[str, Any]],
    onnx_parity_samples: int,
    onnx_max_diff: float,
    seed: int,
    data_files_sha256_list: Sequence[Dict[str, str]],
    torch_version: str = "unknown",
    git_commit: Optional[str] = None,
    set_latest: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Write normalizer JSON and metrics JSON with cryptographic provenance (`model_sha256`,
    `normalizer_sha256`, 24-hex `model_hash`, `git_commit`, `seed`, `data_files_sha256_list`,
    `torch_version`). Only overwrites `metrics.json` when `set_latest=True` (V051-10).
    """
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    onnx_fp = Path(onnx_path)
    norm_fp = out_p / f"{name}_normalizer.json"
    metrics_fp = out_p / f"{name}_metrics.json"
    latest_metrics_fp = out_p / "metrics.json"

    eff_commit = git_commit if git_commit is not None else _get_git_commit()
    model_sha256 = sha256_file(onnx_fp)

    norm_payload = {
        "feature_set": feature_set,
        "feature_dim": int(feat_dim),
        "sequence_length": int(seq_len),
        "classes": [p.value for p in ALL_POSE_NAMES],
        "mean": np.asarray(mean, dtype=np.float32).tolist(),
        "scale": np.asarray(scale, dtype=np.float32).tolist(),
        "temperature": float(temperature),
        "val_accuracy": best_metrics.get("val_accuracy", 0.0),
        "macro_f1": best_metrics.get("macro_f1", 0.0),
        "idle_fp_rate": best_metrics.get("idle_fp_rate", 0.0),
        "model_sha256": model_sha256,
        "git_commit": eff_commit,
        "seed": int(seed),
        "torch_version": str(torch_version),
    }
    norm_fp.write_text(json.dumps(norm_payload, indent=2), encoding="utf-8")
    normalizer_sha256 = sha256_file(norm_fp)
    model_hash = compute_onnx_model_hash(onnx_fp, norm_fp)

    metrics_payload = {
        "model_name": name,
        "model_sha256": model_sha256,
        "normalizer_sha256": normalizer_sha256,
        "model_hash": model_hash,
        "git_commit": eff_commit,
        "seed": int(seed),
        "torch_version": str(torch_version),
        "data_files_sha256_list": list(data_files_sha256_list),
        "feature_set": feature_set,
        "feature_dim": int(feat_dim),
        "seq_len": int(seq_len),
        "hidden": int(hidden),
        "loss": loss,
        "augmented": bool(augmented),
        "temperature": float(temperature),
        "train_sequences": int(train_sequences),
        "val_sequences": int(val_sequences),
        "idle_scenario_balance": balance_stats,
        "dev_test_metrics": dev_test_metrics,
        "subject_heldout_metrics": subject_ho_metrics,
        "onnx_parity_samples": int(onnx_parity_samples),
        "onnx_max_diff": round(float(onnx_max_diff), 8),
        "classes": [p.value for p in ALL_POSE_NAMES],
        **best_metrics,
    }
    metrics_json = json.dumps(metrics_payload, indent=2)
    metrics_fp.write_text(metrics_json, encoding="utf-8")
    if set_latest:
        latest_metrics_fp.write_text(metrics_json, encoding="utf-8")
    return norm_fp, metrics_fp, metrics_payload


def train_and_export(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        import onnxruntime as ort
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            f"Missing training dependency ({exc}). Install with:\n"
            "  cd /Users/macos/Documents/AeroHand && uv sync --extra train"
        ) from None

    class CausalConv1dBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_ch)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_ch)
            self.drop = nn.Dropout(dropout)
            self.act = nn.ReLU(inplace=True)
            self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = F.pad(x, (self.pad, 0))
            h = self.drop(self.act(self.bn1(self.conv1(h))))
            h = F.pad(h, (self.pad, 0))
            h = self.drop(self.act(self.bn2(self.conv2(h))))
            res = x if self.downsample is None else self.downsample(x)
            return self.act(h + res)

    class AeroHandTCN(nn.Module):
        def __init__(self, feat_dim: int, num_classes: int = NUM_CLASSES, hidden: int = 96):
            super().__init__()
            self.blocks = nn.Sequential(
                CausalConv1dBlock(feat_dim, hidden, kernel_size=3, dilation=1, dropout=0.20),
                CausalConv1dBlock(hidden, hidden, kernel_size=3, dilation=2, dropout=0.20),
            )
            self.head = nn.Linear(hidden, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.blocks(x.transpose(1, 2))
            return self.head(h[:, :, -1])

    class FocalLoss(nn.Module):
        def __init__(self, weight: torch.Tensor, gamma: float = 2.0) -> None:
            super().__init__()
            self.weight = weight
            self.gamma = gamma

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
            pt = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
            return (((1.0 - pt) ** self.gamma) * ce).mean()

    data_root = getattr(args, "data_root", "data") or "data"
    sealed_root = getattr(args, "sealed_root", None)
    raw_sessions = args.sessions if isinstance(args.sessions, (list, tuple)) else [args.sessions]
    files, excl_s = collect_training_files(raw_sessions, data_root=data_root, sealed_root=sealed_root)
    if excl_s:
        print(f"[WARN] Excluded {len(excl_s)} session file(s) whose SHA-256 matches sealed_manifest.json.")
    if not files:
        raise SystemExit(
            f"No session JSON files found in {raw_sessions}.\n"
            "Record labeled data first inside AeroHand.app -> 'Dữ liệu ML' tab (or `aerohand run --collect-dir data/sessions`)."
        )

    skip_stats: Dict[str, int] = {}
    bundles: List[SessionBundle] = []
    input_files: List[Path] = list(files)
    for fp in files:
        b = load_session_bundle(fp, seq_len=args.seq_len, max_gap_ms=args.max_gap_ms, skip_stats=skip_stats)
        if b.windows:
            bundles.append(b)

    if args.negatives_dir:
        raw_negs = (
            args.negatives_dir
            if isinstance(args.negatives_dir, (list, tuple))
            else [args.negatives_dir]
        )
        neg_files, excl_n = collect_training_files(raw_negs, data_root=data_root, sealed_root=sealed_root)
        if excl_n:
            print(f"[WARN] Excluded {len(excl_n)} negative file(s) whose SHA-256 matches sealed_manifest.json.")
        input_files.extend(neg_files)
        for fp in neg_files:
            b = load_session_bundle(
                fp, seq_len=args.seq_len, force_idle=True, max_gap_ms=args.max_gap_ms, skip_stats=skip_stats
            )
            if b.windows:
                bundles.append(b)

    data_files_sha256_list = [
        {"file": fp.name, "sha256": sha256_file(fp)}
        for fp in sorted(set(input_files))
        if fp.is_file()
    ]

    if skip_stats:
        print(f"[DATA] Skipped samples by reason: {skip_stats}")
    if not bundles:
        raise SystemExit("Loaded 0 valid sequences across all session files.")

    # Separate non-primary / volunteer subjects strictly out of dev training (V050-02 / V051-01)
    ho_subjs = [s.strip() for s in str(getattr(args, "heldout_subjects", "") or "").split(",") if s.strip()]
    primary_subj = str(getattr(args, "primary_subject", "") or "")
    dev_bundles, subject_ho_bundles = split_subject_heldout(
        bundles,
        primary_subject=primary_subj,
        heldout_subjects=ho_subjs,
    )
    subj_summary = summarize_bundles_by_subject(bundles)
    print(f"[DATA] Bundles/windows by subject: {subj_summary}")
    try:
        validate_subject_split(
            bundles,
            dev_bundles,
            subject_ho_bundles,
            primary_subject=primary_subj,
            heldout_subjects=ho_subjs,
        )
    except ValueError as exc:
        raise SystemExit(f"Aborting: {exc}") from None
    if subject_ho_bundles:
        ho_names = sorted({b.subject for b in subject_ho_bundles})
        print(f"[DATA] Held out volunteer subjects for out-of-subject eval: {ho_names} ({len(subject_ho_bundles)} files)")
    if not dev_bundles:
        raise SystemExit("No development bundles remain after subject-held-out filtering.")

    # Carve out a disjoint dev_test group split when >= 4 distinct groups exist (V051-10)
    dev_train_val_bundles, dev_test_bundles = split_dev_test_groups(dev_bundles)
    train_bundles, val_bundles, _ = split_session_groups(dev_train_val_bundles, val_ratio=args.val_ratio)
    try:
        missing_val = validate_validation_classes(
            val_bundles, allow_missing=bool(args.allow_missing_val_classes)
        )
        validate_train_classes(
            train_bundles,
            min_windows=int(getattr(args, "min_train_windows", 300)),
            allow_low_count=bool(
                getattr(args, "allow_low_train_counts", False)
                or getattr(args, "allow_missing_val_classes", False)
            ),
            allow_missing=bool(getattr(args, "allow_missing_val_classes", False)),
        )
    except ValueError as exc:
        raise SystemExit(f"Aborting: {exc}") from None
    if missing_val:
        print(f"[WARN] Validation split is missing classes: {missing_val}")

    max_sc_share = float(getattr(args, "max_idle_scenario_share", 0.15))
    balance_stats: Dict[str, Any] = {}
    if 0.0 < max_sc_share < 1.0:
        train_bundles, balance_stats = balance_scenario_samples(
            train_bundles, max_scenario_share=max_sc_share, seed=args.seed
        )
        if balance_stats.get("capped_scenarios"):
            print(
                f"[DATA] Balanced idle scenarios (cap={max_sc_share:.2f}): "
                f"idle {balance_stats['total_idle_before']} -> {balance_stats['total_idle_after']}, "
                f"max_share {balance_stats['max_share_before']:.2f} -> {balance_stats['max_share_after']:.2f}"
            )

    train_windows = [w for b in train_bundles for w in b.windows]
    val_windows = [w for b in val_bundles for w in b.windows]

    rng = np.random.default_rng(args.seed)
    if args.augment and args.augment_factor > 1:
        extra_windows: List[List[ParsedSample]] = []
        for _ in range(args.augment_factor - 1):
            for w in train_windows:
                extra_windows.append(augment_window(w, rng))
        train_windows = train_windows + extra_windows

    X_train, y_train = _featurize_windows(
        train_windows, feature_set=args.feature_set, seq_len=args.seq_len, skip_stats=skip_stats
    )
    X_val, y_val = _featurize_windows(
        val_windows, feature_set=args.feature_set, seq_len=args.seq_len, skip_stats=skip_stats
    )
    if len(X_train) == 0 or len(X_val) == 0:
        raise SystemExit(f"Insufficient sequences after featurization: train={len(X_train)}, val={len(X_val)}")

    total_counts = np.bincount(np.concatenate([y_train, y_val]), minlength=NUM_CLASSES)
    print("[DATA] Sequence counts per class (train + val):")
    for idx, pose in enumerate(ALL_POSE_NAMES):
        print(f"  [{idx}] {pose.value:26s}: {int(total_counts[idx])}")

    feat_dim = X_train.shape[2]
    flat_train = X_train.reshape(-1, feat_dim)
    mean = flat_train.mean(axis=0).astype(np.float32)
    scale = flat_train.std(axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)

    X_train_n = (X_train - mean) / scale
    X_val_n = (X_val - mean) / scale

    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = np.where(counts > 0, counts.sum() / np.maximum(counts * NUM_CLASSES, 1.0), 0.0)
    class_weights = np.clip(class_weights, 0.2, 6.0).astype(np.float32)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(
        f"[TRAIN] device={device} feature_set={args.feature_set} ({feat_dim}d) "
        f"loss={args.loss} augment={args.augment} train={len(X_train)} val={len(X_val)}"
    )

    model = AeroHandTCN(feat_dim=feat_dim, num_classes=NUM_CLASSES, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    w_t = torch.tensor(class_weights, device=device)
    if args.loss == "focal":
        loss_fn: nn.Module = FocalLoss(weight=w_t, gamma=2.0)
    elif args.loss == "ls":
        loss_fn = nn.CrossEntropyLoss(weight=w_t, label_smoothing=0.05)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=w_t)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_n), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best_score = -1e9
    best_metrics: Dict[str, Any] = {}
    best_state = None
    best_val_logits = np.zeros((len(X_val_n), NUM_CLASSES), dtype=np.float32)
    X_val_t = torch.from_numpy(X_val_n).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(bx)
        sched.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(X_val_t).detach().cpu().numpy()
            v_preds = np.argmax(v_logits, axis=1)
        m = compute_classification_metrics(y_val, v_preds)
        if m["selection_score"] >= best_score:
            best_score = m["selection_score"]
            best_metrics = m
            best_val_logits = v_logits
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"  epoch {epoch:02d}/{args.epochs} | loss={total_loss/max(1,len(X_train)):.4f} "
                f"| val_acc={m['val_accuracy']*100:.2f}% | macro_f1={m['macro_f1']*100:.2f}% "
                f"| idle_fp={m['idle_fp_rate']*100:.2f}%"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu().eval()

    temperature = fit_temperature_scaling(best_val_logits, y_val)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_fp = out_dir / f"{args.name}.onnx"

    dummy = torch.zeros(1, args.seq_len, feat_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_fp),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )

    # Verify ONNX Runtime parity across up to 256 validation samples (V051-10)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_fp), sess_options=opts, providers=["CPUExecutionProvider"])
    n_parity = min(256, len(X_val_n))
    max_diff = 0.0
    if n_parity > 0:
        for i in range(n_parity):
            sample_x = X_val_n[i : i + 1]
            pt_out = model(torch.from_numpy(sample_x)).detach().numpy()
            ort_out = sess.run(None, {"input": sample_x})[0]
            diff = float(np.max(np.abs(pt_out - ort_out)))
            if diff > max_diff:
                max_diff = diff
    else:
        sample_x = np.zeros((1, args.seq_len, feat_dim), dtype=np.float32)
        pt_out = model(torch.from_numpy(sample_x)).detach().numpy()
        ort_out = sess.run(None, {"input": sample_x})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
    if max_diff >= 1e-4:
        raise RuntimeError(f"ONNX export parity check failed: max_diff={max_diff:.6e} >= 1e-4 over {n_parity} samples")

    dev_test_metrics: Optional[Dict[str, Any]] = None
    if dev_test_bundles:
        dt_windows = [w for b in dev_test_bundles for w in b.windows]
        X_dt, y_dt = _featurize_windows(
            dt_windows, feature_set=args.feature_set, seq_len=args.seq_len, skip_stats=skip_stats
        )
        if len(X_dt) > 0:
            X_dt_n = (X_dt - mean) / scale
            with torch.no_grad():
                dt_logits = model(torch.from_numpy(X_dt_n)).detach().numpy()
                dt_preds = np.argmax(dt_logits, axis=1)
            dev_test_metrics = compute_classification_metrics(y_dt, dt_preds)
            dev_test_metrics["groups"] = sorted({b.group_key for b in dev_test_bundles})
            dev_test_metrics["sequences"] = int(len(X_dt))

    subject_ho_metrics: Optional[Dict[str, Any]] = None
    if subject_ho_bundles:
        ho_windows = [w for b in subject_ho_bundles for w in b.windows]
        X_ho, y_ho = _featurize_windows(
            ho_windows, feature_set=args.feature_set, seq_len=args.seq_len, skip_stats=skip_stats
        )
        if len(X_ho) > 0:
            X_ho_n = (X_ho - mean) / scale
            with torch.no_grad():
                ho_logits = model(torch.from_numpy(X_ho_n)).detach().numpy()
                ho_preds = np.argmax(ho_logits, axis=1)
            subject_ho_metrics = compute_classification_metrics(y_ho, ho_preds)
            subject_ho_metrics["subjects"] = sorted({b.subject for b in subject_ho_bundles})
            subject_ho_metrics["sequences"] = int(len(X_ho))

    norm_fp, metrics_fp, metrics_payload = write_training_artifacts(
        out_dir=out_dir,
        name=args.name,
        onnx_path=onnx_fp,
        feature_set=args.feature_set,
        feat_dim=feat_dim,
        seq_len=args.seq_len,
        hidden=args.hidden,
        loss=args.loss,
        augmented=bool(args.augment),
        mean=mean,
        scale=scale,
        temperature=temperature,
        train_sequences=len(X_train),
        val_sequences=len(X_val),
        best_metrics=best_metrics,
        balance_stats=balance_stats,
        dev_test_metrics=dev_test_metrics,
        subject_ho_metrics=subject_ho_metrics,
        onnx_parity_samples=n_parity,
        onnx_max_diff=max_diff,
        seed=int(args.seed),
        data_files_sha256_list=data_files_sha256_list,
        torch_version=str(getattr(torch, "__version__", "unknown")),
        set_latest=bool(getattr(args, "set_latest", False)),
    )

    print(
        f"[EXPORT] Saved {onnx_fp}, {norm_fp}, {metrics_fp} | "
        f"macro_f1={best_metrics.get('macro_f1', 0.0)*100:.2f}% | "
        f"idle_fp={best_metrics.get('idle_fp_rate', 0.0)*100:.2f}% | "
        f"T={temperature:.3f} | ONNX max_diff={max_diff:.2e}"
    )
    return metrics_payload


def run_ab_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    """Train & compare `invariant348` vs `airmouse885` across multiple seeds and write `ab_report.json`."""
    import importlib.util

    seeds = [int(s.strip()) for s in str(args.ab_seeds).split(",") if s.strip()]
    if not seeds:
        seeds = list(AB_DEFAULT_SEEDS)

    data_root = getattr(args, "data_root", "data") or "data"
    sealed_root = getattr(args, "sealed_root", None)
    eval_neg_paths = (
        collect_training_files(args.ab_eval_negatives, data_root=data_root, sealed_root=sealed_root)[0]
        if getattr(args, "ab_eval_negatives", None)
        else []
    )
    if eval_neg_paths and getattr(args, "negatives_dir", None):
        raw_negs = (
            args.negatives_dir
            if isinstance(args.negatives_dir, (list, tuple))
            else [args.negatives_dir]
        )
        train_neg_files, _ = collect_training_files(raw_negs, data_root=data_root, sealed_root=sealed_root)
        validate_disjoint_ab_negatives(train_neg_files, eval_neg_paths)
    replay_mod = None
    if eval_neg_paths:
        rfp = Path(__file__).resolve().parent / "replay_eval.py"
        spec = importlib.util.spec_from_file_location("_aerohand_replay_eval_ab", rfp)
        if spec is not None and spec.loader is not None:
            replay_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(replay_mod)

    out_dir = Path(args.out_dir)
    runs_by_fs: Dict[str, List[Dict[str, Any]]] = {"invariant348": [], "airmouse885": []}
    for fs in ("invariant348", "airmouse885"):
        for sd in seeds:
            run_args = argparse.Namespace(**vars(args))
            run_args.feature_set = fs
            run_args.seed = sd
            run_args.name = f"{args.name}_{fs}_s{sd}"
            run_args.set_latest = False
            m = train_and_export(run_args)
            m["seed"] = sd
            if replay_mod is not None and eval_neg_paths:
                from aerohand.ml.onnx_classifier import ONNXPinchClassifier

                clf = ONNXPinchClassifier(
                    model_path=out_dir / f"{run_args.name}.onnx",
                    normalizer_path=out_dir / f"{run_args.name}_normalizer.json",
                    sequence_length=args.seq_len,
                    feature_set=fs,
                )
                rep = replay_mod.evaluate_recordings(
                    negative_paths=eval_neg_paths,
                    classifier=clf,
                    backend_name="onnx_tcn",
                    min_exposure_hours_for_claim=0.0,
                    max_ft_per_hour=1e9,
                )
                m["ft_per_hour"] = rep["false_triggers"]["total_ft_per_hour"]
                m["ft_upper_bound_per_hour"] = rep["false_triggers"]["total_upper_bound_per_hour"]
            runs_by_fs[fs].append(m)

    report = build_ab_report(runs_by_fs["invariant348"], runs_by_fs["airmouse885"], seeds=seeds)
    out_dir.mkdir(parents=True, exist_ok=True)
    ab_fp = out_dir / "ab_report.json"
    ab_fp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    dec = report["decision"]
    print(f"[A/B] Winner: {dec['winner']} | {dec['reason']} | Saved -> {ab_fp}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Train AeroHand Causal 1D-TCN and export ONNX + metrics.")
    p.add_argument(
        "--sessions",
        nargs="+",
        default=["data/sessions"],
        help="One or more directories/files containing collected JSON sessions",
    )
    p.add_argument(
        "--negatives-dir",
        nargs="*",
        default=None,
        help="Optional directories/files of hard-negative JSON recordings",
    )
    p.add_argument("--out-dir", type=str, default="models", help="Output directory for .onnx, normalizer, and metrics.json")
    p.add_argument("--name", type=str, default="aerohand_tcn", help="Base filename for exported artifacts")
    p.add_argument("--feature-set", choices=["invariant348", "airmouse885"], default="invariant348")
    p.add_argument("--ab-compare", action="store_true", help="Run multi-seed A/B evaluation comparing invariant348 vs airmouse885")
    p.add_argument("--ab-seeds", type=str, default="42,1337,2026", help="Comma-separated seeds for --ab-compare")
    p.add_argument(
        "--ab-eval-negatives",
        nargs="*",
        default=None,
        help="Optional offline negative recordings used to compute FT/hour & 95% UCB per A/B seed",
    )
    p.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Root data directory scanned for sealed_manifest.json SHA-256 exclusion (default 'data')",
    )
    p.add_argument(
        "--sealed-root",
        type=str,
        default=None,
        help="Explicit directory scanned for sealed_manifest.json SHA-256 exclusion (overrides/augments --data-root)",
    )
    p.add_argument("--loss", choices=["ce", "focal", "ls"], default="focal")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable landmark augmentation (default on)")
    p.add_argument("--augment-factor", type=int, default=2, help="Total training expansion factor when --augment is on")
    p.add_argument("--max-gap-ms", type=float, default=100.0, help="Break sequence windows on gaps > this threshold (ms)")
    p.add_argument(
        "--max-idle-scenario-share",
        type=float,
        default=0.15,
        help="Cap any single scenario to <= this fraction of idle training windows (default 0.15)",
    )
    p.add_argument(
        "--primary-subject",
        type=str,
        default="",
        help="If set, keep all non-primary subjects strictly out of training for subject-held-out evaluation",
    )
    p.add_argument(
        "--heldout-subjects",
        type=str,
        default="",
        help="Comma-separated volunteer subject IDs held out strictly for subject-held-out evaluation",
    )
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--allow-missing-val-classes", action="store_true", help="Allow smoke training even if val lacks some classes")
    p.add_argument("--min-train-windows", type=int, default=300, help="Minimum required training windows per canonical class (default 300)")
    p.add_argument("--allow-low-train-counts", action="store_true", help="Allow smoke training with fewer than --min-train-windows per class")
    p.add_argument("--set-latest", action="store_true", help="Also overwrite models/metrics.json with this run's metrics")
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--seed", type=int, default=42)
    ns = p.parse_args()
    if ns.ab_compare:
        run_ab_comparison(ns)
    else:
        train_and_export(ns)


if __name__ == "__main__":
    main()
