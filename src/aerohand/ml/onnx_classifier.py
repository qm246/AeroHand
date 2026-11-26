"""
Self-Trained ONNX Causal TCN Classifier (Dual-Feature: 348-dim Invariant3D & 885-dim AirMouse).
Adapted from Airpoint SDK `ONNXGestureClassifier.ts` (Apache-2.0) for Python `onnxruntime`.

Upgrades (v0.5.1 — V051-03 / V051-04 / V051-05 / V051-06):
  - Per-hand independent 5-frame causal ring buffers (`Left` & `Right` never cross-contaminate).
  - Auto-detects `invariant348` vs `airmouse885` feature extractor from normalizer JSON metadata/length.
  - Warm-up (20 runs) in `_load()` so cold-start ORT allocation never trips the latency guard.
  - Separates `feature_ms` from `infer_ms` (`session.run` only) and skips buffer-fill frames (`ready=False`).
  - Raises `ValueError` on feature-dim mismatch so `GuardedClassifier` triggers error fallback instead of silent veto.
  - Entry-only event-counted veto mode (`veto_scope="entry"`, `veto_entry_window_ms=300`, `veto_on_error="pass_through"`).
  - Supports `rearm()`, in-place heuristic config updates, and cumulative counter export/import.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Sequence, Tuple

import numpy as np

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName, idle_probabilities
from aerohand.ml.base import PinchClassifier, PinchPrediction
from aerohand.ml.features_3d import FEATURE_DIM_3D, extract_invariant_3d_features
from aerohand.ml.features_885 import AIRMOUSE_885_DIM, extract_airmouse_885_features
from aerohand.ml.provenance import compute_heuristic_source_sha256, compute_onnx_model_hash
from aerohand.ml.recording_io import sha256_file

logger = logging.getLogger(__name__)

VALID_CLASSIFIER_MODES = ("heuristic", "shadow", "veto", "onnx")
VALID_VETO_SCOPES = ("entry", "always")
VALID_VETO_ON_ERROR = ("pass_through", "idle")


def resolve_classifier_mode(cls_cfg: Optional[Dict[str, Any]]) -> str:
    """Resolve effective classifier mode (`heuristic | shadow | veto | onnx`) with legacy compatibility."""
    cfg = cls_cfg or {}
    raw_mode = str(cfg.get("mode", "")).strip().lower()
    if raw_mode in ("shadow", "veto", "onnx"):
        return raw_mode
    if bool(cfg.get("shadow_mode", False)):
        return "shadow"
    if str(cfg.get("backend", "heuristic_3d")).strip().lower() == "onnx_tcn":
        return "onnx"
    return "heuristic"


class ONNXPinchClassifier:
    """Streaming per-hand sliding-window ONNX Runtime classifier."""

    def __init__(
        self,
        model_path: str | Path,
        normalizer_path: str | Path,
        sequence_length: int = 5,
        feature_set: Optional[str] = None,
        *,
        warmup_runs: int = 20,
    ) -> None:
        self.model_path = Path(model_path)
        self.normalizer_path = Path(normalizer_path)
        self.sequence_length = max(1, int(sequence_length))
        self.feature_set: str = feature_set or "auto"
        self._warmup_runs: int = max(0, int(warmup_runs))
        self._buffers: Dict[Handedness, Deque[np.ndarray]] = {
            "Left": deque(maxlen=self.sequence_length),
            "Right": deque(maxlen=self.sequence_length),
        }
        self._session: Any = None
        self._input_name: str = "input"
        self._mean: np.ndarray = np.zeros(FEATURE_DIM_3D, dtype=np.float32)
        self._scale: np.ndarray = np.ones(FEATURE_DIM_3D, dtype=np.float32)
        self._temperature: float = 1.0
        self.model_sha256: Optional[str] = None
        self.normalizer_sha256: Optional[str] = None
        self.model_hash: Optional[str] = None
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed. Run `uv add onnxruntime`.") from exc

        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.normalizer_path.is_file():
            raise FileNotFoundError(f"Normalizer JSON not found: {self.normalizer_path}")

        norm = json.loads(self.normalizer_path.read_text(encoding="utf-8"))
        self._mean = np.asarray(norm["mean"], dtype=np.float32)
        scale = np.asarray(norm["scale"], dtype=np.float32)
        self._scale = np.where(np.abs(scale) > 1e-8, scale, 1.0).astype(np.float32)
        raw_temp = norm.get("temperature", 1.0)
        try:
            self._temperature = max(1e-3, float(raw_temp))
        except (TypeError, ValueError):
            self._temperature = 1.0

        if self.feature_set == "auto":
            meta_fs = str(norm.get("feature_set", "")).lower()
            if meta_fs in ("invariant348", "airmouse885"):
                self.feature_set = meta_fs
            elif self._mean.shape[0] == AIRMOUSE_885_DIM:
                self.feature_set = "airmouse885"
            else:
                self.feature_set = "invariant348"

        seq_meta = norm.get("sequence_length")
        if isinstance(seq_meta, int) and seq_meta > 0:
            self.sequence_length = seq_meta
            self._buffers = {
                "Left": deque(maxlen=self.sequence_length),
                "Right": deque(maxlen=self.sequence_length),
            }

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

        # V052-02: Compute cryptographic provenance hashes matching train_tcn & replay_eval
        self.model_sha256 = sha256_file(self.model_path)
        self.normalizer_sha256 = sha256_file(self.normalizer_path)
        self.model_hash = compute_onnx_model_hash(self.model_path, self.normalizer_path)

        # V051-03: Warm-up ORT session so cold-start allocation never trips the latency guard
        if self._warmup_runs > 0:
            dummy = np.zeros((1, self.sequence_length, int(self._mean.shape[0])), dtype=np.float32)
            for _ in range(self._warmup_runs):
                self._session.run(None, {self._input_name: dummy})

        logger.info(
            f"[ONNX] Loaded TCN model: {self.model_path.name} "
            f"(feature_set={self.feature_set}, dim={self._mean.shape[0]}, seq={self.sequence_length}, "
            f"warmup={self._warmup_runs}, model_hash={self.model_hash})"
        )

    def reset(self, hand: Optional[Handedness] = None) -> None:
        if hand is None:
            for buf in self._buffers.values():
                buf.clear()
        else:
            self._buffers[hand].clear()

    def predict(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
    ) -> PinchPrediction:
        if self._session is None:
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low", ready=False)

        tf0 = time.perf_counter()
        if self.feature_set == "airmouse885":
            feat = extract_airmouse_885_features(landmarks_2d, world_landmarks, handedness=handedness)
        else:
            feat = extract_invariant_3d_features(landmarks_2d, world_landmarks, handedness=handedness)
        feat_ms = (time.perf_counter() - tf0) * 1000.0

        if feat is None:
            return PinchPrediction(
                PoseName.IDLE, idle_probabilities(), "low", ready=False, feature_ms=feat_ms
            )
        if feat.shape[0] != self._mean.shape[0]:
            raise ValueError(
                f"ONNX feature_dim mismatch: extracted {feat.shape[0]} ({self.feature_set}), "
                f"normalizer expects {self._mean.shape[0]}."
            )

        norm_feat = (feat - self._mean) / self._scale
        buf = self._buffers[handedness]
        buf.append(norm_feat)
        if len(buf) < self.sequence_length:
            return PinchPrediction(
                PoseName.IDLE, idle_probabilities(), "low", ready=False, feature_ms=feat_ms
            )

        seq = np.stack(list(buf), axis=0)[None, :, :]  # (1, seq_len, F)
        ti0 = time.perf_counter()
        raw_out = self._session.run(None, {self._input_name: seq})[0]
        infer_ms = (time.perf_counter() - ti0) * 1000.0

        logits = np.asarray(raw_out).reshape(-1).astype(np.float64)
        logits = logits / self._temperature
        logits = logits - float(np.max(logits))
        exps = np.exp(logits)
        probs_arr = exps / float(np.sum(exps))

        probs = idle_probabilities()
        for i, pose in enumerate(ALL_POSE_NAMES):
            if i < len(probs_arr):
                probs[pose] = float(probs_arr[i])

        best_pose = max(probs.items(), key=lambda kv: kv[1])[0]
        best_p = probs[best_pose]
        conf = "high" if best_p > 0.8 else ("medium" if best_p > 0.6 else "low")
        return PinchPrediction(
            pose=best_pose,
            probabilities=probs,
            confidence=conf,
            ready=True,
            infer_ms=infer_ms,
            feature_ms=feat_ms,
        )


class _HeurEventState:
    """Per-hand state tracking multi-frame ONNX confirmation for one heuristic pose attempt (V052-01 / V053-02)."""

    __slots__ = (
        "pose",
        "enter_ts_ms",
        "last_ts_ms",
        "accepted",
        "ever_accepted",
        "event_vetoed",
        "approve_streak",
        "saw_ready",
    )

    def __init__(self, pose: PoseName, enter_ts_ms: float) -> None:
        self.pose: PoseName = pose
        self.enter_ts_ms: float = enter_ts_ms
        self.last_ts_ms: float = enter_ts_ms
        self.accepted: bool = False
        self.ever_accepted: bool = False
        self.event_vetoed: bool = False
        self.approve_streak: int = 0
        self.saw_ready: bool = False


class GuardedClassifier:
    """
    Production safety wrapper for ONNX TCN + Heuristic3D fallback (V040-04 / V050-04 / V051-03..06 / V052-01):
      - Modes:
          * `heuristic`: runs Heuristic3D only.
          * `shadow`: runs ONNX alongside Heuristic3D and tracks both frame disagreements AND
            event-level `shadow_would_veto_*` counters using the exact K-frame confirmation rule.
          * `veto`: Heuristic3D proposes; ONNX can ONLY veto non-idle poses when `ready=True`.
            Default `veto_scope="entry"` requires `K = veto_confirm_frames` consecutive ready
            ONNX approval frames (`p_proposed >= veto_min_prob` and `p_idle <= veto_idle_prob`)
            before marking the attempt `accepted`; once accepted, ongoing holds are never interrupted.
            ONNX can NEVER create a new activation when Heuristic3D predicts IDLE.
          * `onnx`: full ONNX 1D-TCN predictions.
      - Latency guard: tracks ONLY real inference calls (`ready=True`, `infer_ms` around `session.run`),
        requires >= `min_latency_samples` (30) real inferences, and uses `np.percentile(..., 95)`.
      - Re-arm: `rearm()` clears fallback state & rolling windows cleanly.
      - Buffer hygiene: clears per-hand 5-frame temporal buffers on `reset(hand)` AND `set_mode()`.
    """

    def __init__(
        self,
        onnx_classifier: PinchClassifier,
        fallback_classifier: PinchClassifier,
        *,
        mode: Optional[str] = None,
        shadow_mode: bool = False,
        veto_min_prob: float = 0.35,
        veto_idle_prob: float = 0.60,
        veto_scope: str = "entry",
        veto_entry_window_ms: float = 300.0,
        veto_confirm_frames: int = 3,
        veto_on_error: str = "pass_through",
        max_p95_ms: float = 8.0,
        min_latency_samples: int = 30,
        max_errors: int = 3,
        error_window_s: float = 10.0,
        on_error: Optional[Callable[[], None]] = None,
        clock_s: Optional[Callable[[], float]] = None,
    ) -> None:
        self._onnx = onnx_classifier
        self._fallback = fallback_classifier
        if mode is not None:
            norm_mode = str(mode).strip().lower()
            if norm_mode not in VALID_CLASSIFIER_MODES:
                raise ValueError(f"Unsupported classifier mode '{mode}'; expected one of {VALID_CLASSIFIER_MODES}")
            self._mode: str = norm_mode
        else:
            self._mode = "shadow" if bool(shadow_mode) else "onnx"

        scope_norm = str(veto_scope or "entry").strip().lower()
        self.veto_scope: str = scope_norm if scope_norm in VALID_VETO_SCOPES else "entry"
        self.veto_entry_window_ms: float = max(0.0, float(veto_entry_window_ms))
        self.veto_confirm_frames: int = max(1, int(veto_confirm_frames))
        err_pol = str(veto_on_error or "pass_through").strip().lower()
        self.veto_on_error: str = err_pol if err_pol in VALID_VETO_ON_ERROR else "pass_through"

        self.veto_min_prob: float = max(0.05, min(0.95, float(veto_min_prob)))
        self.veto_idle_prob: float = max(0.05, min(0.95, float(veto_idle_prob)))
        self.max_p95_ms: float = max(0.5, float(max_p95_ms))
        self.min_latency_samples: int = max(5, int(min_latency_samples))
        self.max_errors: int = max(1, int(max_errors))
        self.error_window_s: float = max(1.0, float(error_window_s))
        self._on_error = on_error
        self._clock_s: Callable[[], float] = clock_s or time.monotonic

        self._latencies_ms: Deque[float] = deque(maxlen=128)
        self._feature_latencies_ms: Deque[float] = deque(maxlen=128)
        self._error_times: Deque[float] = deque()
        self._total_errors: int = 0
        self._fell_back: bool = False
        self._fallback_reason: Optional[str] = None
        self._shadow_frames: int = 0
        self._shadow_disagreements: int = 0
        self._shadow_events: int = 0
        self._shadow_would_veto_events: int = 0
        self._shadow_would_veto_frames: int = 0
        self._shadow_would_swallow_events: int = 0
        self._shadow_would_pending_frames: int = 0
        self._veto_count: int = 0
        self._veto_frames: int = 0
        self._veto_swallowed_events: int = 0
        self._veto_pending_frames: int = 0
        self._veto_retried_within_1s: int = 0
        self._veto_abstain_not_ready: int = 0
        # Per-hand rising-edge event trackers (V051-05 / V052-01)
        self._heur_event_by_hand: Dict[Handedness, _HeurEventState] = {}
        self._shadow_event_by_hand: Dict[Handedness, _HeurEventState] = {}
        # Per-hand last vetoed event for retry-within-1s detection: (pose, veto_ts_ms, released_to_idle)
        self._last_veto_by_hand: Dict[Handedness, Tuple[PoseName, float, bool]] = {}

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def shadow_mode(self) -> bool:
        return self._mode == "shadow"

    @shadow_mode.setter
    def shadow_mode(self, enabled: bool) -> None:
        self.set_mode("shadow" if enabled else "onnx")

    def set_mode(self, new_mode: str) -> None:
        """Switch operating mode and clear all per-hand temporal buffers cleanly."""
        norm = str(new_mode).strip().lower()
        if norm not in VALID_CLASSIFIER_MODES:
            raise ValueError(f"Unsupported classifier mode '{new_mode}'; expected one of {VALID_CLASSIFIER_MODES}")
        if norm != self._mode:
            self._mode = norm
            self._last_veto_by_hand.clear()
            self._heur_event_by_hand.clear()
            self._shadow_event_by_hand.clear()
            self.reset(None)

    def rearm(self) -> None:
        """Re-arm ONNX after an auto-fallback event, clearing stale latency/error windows (V051-03)."""
        self._fell_back = False
        self._fallback_reason = None
        self._latencies_ms.clear()
        self._feature_latencies_ms.clear()
        self._error_times.clear()
        self.reset(None)

    def update_veto_config(
        self,
        *,
        veto_min_prob: Optional[float] = None,
        veto_idle_prob: Optional[float] = None,
        veto_scope: Optional[str] = None,
        veto_entry_window_ms: Optional[float] = None,
        veto_confirm_frames: Optional[int] = None,
        veto_on_error: Optional[str] = None,
        max_p95_ms: Optional[float] = None,
    ) -> None:
        """Update guard/veto settings in-place without rebuilding ONNX or resetting counters (V051-06 / V052-01)."""
        if veto_min_prob is not None:
            self.veto_min_prob = max(0.05, min(0.95, float(veto_min_prob)))
        if veto_idle_prob is not None:
            self.veto_idle_prob = max(0.05, min(0.95, float(veto_idle_prob)))
        if veto_scope is not None:
            s = str(veto_scope).strip().lower()
            if s in VALID_VETO_SCOPES:
                self.veto_scope = s
        if veto_entry_window_ms is not None:
            self.veto_entry_window_ms = max(0.0, float(veto_entry_window_ms))
        if veto_confirm_frames is not None:
            self.veto_confirm_frames = max(1, int(veto_confirm_frames))
        if veto_on_error is not None:
            p = str(veto_on_error).strip().lower()
            if p in VALID_VETO_ON_ERROR:
                self.veto_on_error = p
        if max_p95_ms is not None:
            self.max_p95_ms = max(0.5, float(max_p95_ms))

    def update_heuristic_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        """Forward in-place heuristic threshold updates without losing state/counters (V051-06)."""
        upd = getattr(self._fallback, "update_config", None)
        if callable(upd):
            upd(cfg)

    def export_counters(self) -> Dict[str, int]:
        """Export cumulative shadow/veto/error counters so rebuilds never wipe evidence (V051-06 / V053-02)."""
        return {
            "shadow_frames": self._shadow_frames,
            "shadow_disagreements": self._shadow_disagreements,
            "shadow_events": self._shadow_events,
            "shadow_would_veto_events": self._shadow_would_veto_events,
            "shadow_would_veto_frames": self._shadow_would_veto_frames,
            "shadow_would_swallow_events": self._shadow_would_swallow_events,
            "shadow_would_pending_frames": self._shadow_would_pending_frames,
            "veto_count": self._veto_count,
            "veto_frames": self._veto_frames,
            "veto_swallowed_events": self._veto_swallowed_events,
            "veto_pending_frames": self._veto_pending_frames,
            "veto_retried_within_1s": self._veto_retried_within_1s,
            "veto_abstain_not_ready": self._veto_abstain_not_ready,
            "onnx_errors": self._total_errors,
        }

    def import_counters(self, counters: Optional[Dict[str, Any]]) -> None:
        """Restore cumulative shadow/veto/error counters onto a freshly built guard (V051-06 / V053-02)."""
        if not counters:
            return
        self._shadow_frames = int(counters.get("shadow_frames", self._shadow_frames))
        self._shadow_disagreements = int(counters.get("shadow_disagreements", self._shadow_disagreements))
        self._shadow_events = int(counters.get("shadow_events", self._shadow_events))
        self._shadow_would_veto_events = int(counters.get("shadow_would_veto_events", self._shadow_would_veto_events))
        self._shadow_would_veto_frames = int(counters.get("shadow_would_veto_frames", self._shadow_would_veto_frames))
        self._shadow_would_swallow_events = int(
            counters.get("shadow_would_swallow_events", self._shadow_would_swallow_events)
        )
        self._shadow_would_pending_frames = int(
            counters.get("shadow_would_pending_frames", self._shadow_would_pending_frames)
        )
        self._veto_count = int(counters.get("veto_count", self._veto_count))
        self._veto_frames = int(counters.get("veto_frames", self._veto_frames))
        self._veto_swallowed_events = int(counters.get("veto_swallowed_events", self._veto_swallowed_events))
        self._veto_pending_frames = int(counters.get("veto_pending_frames", self._veto_pending_frames))
        self._veto_retried_within_1s = int(counters.get("veto_retried_within_1s", self._veto_retried_within_1s))
        self._veto_abstain_not_ready = int(counters.get("veto_abstain_not_ready", self._veto_abstain_not_ready))
        self._total_errors = int(counters.get("onnx_errors", self._total_errors))

    @property
    def fell_back(self) -> bool:
        return self._fell_back

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @property
    def active_backend_name(self) -> str:
        if self._mode == "heuristic":
            return type(self._fallback).__name__
        if self._fell_back:
            return f"{type(self._fallback).__name__} (Fallback)"
        if self._mode == "shadow":
            return f"{type(self._fallback).__name__} (Shadow: ONNX)"
        if self._mode == "veto":
            return f"{type(self._fallback).__name__} (Veto: ONNX)"
        return type(self._onnx).__name__

    def set_on_error(self, callback: Optional[Callable[[], None]]) -> None:
        self._on_error = callback

    def reset(self, hand: Optional[Handedness] = None) -> None:
        if hand is None:
            # Global reset (mode switch, rearm, fallback): do NOT count in-flight events as swallowed (V053-02)
            self._last_veto_by_hand.clear()
            self._heur_event_by_hand.clear()
            self._shadow_event_by_hand.clear()
        else:
            # Per-hand reset (hand lost mid-event): finalize unaccepted pending events as swallowed (V053-02)
            prev_v_ev = self._heur_event_by_hand.pop(hand, None)
            self._finalize_veto_event(hand, prev_v_ev, None, released_to_idle=True)
            prev_s_ev = self._shadow_event_by_hand.pop(hand, None)
            self._finalize_shadow_event(prev_s_ev)
            self._last_veto_by_hand.pop(hand, None)
        for clf in (self._onnx, self._fallback):
            try:
                clf.reset(hand)  # type: ignore[call-arg]
            except TypeError:
                clf.reset()

    def _p95_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        return float(np.percentile(np.asarray(self._latencies_ms, dtype=np.float64), 95))

    def _feature_p95_ms(self) -> float:
        if not self._feature_latencies_ms:
            return 0.0
        return float(np.percentile(np.asarray(self._feature_latencies_ms, dtype=np.float64), 95))

    def _record_error(self, exc: Exception, *, invoke_on_error: bool = True) -> None:
        now = self._clock_s()
        self._total_errors += 1
        self._error_times.append(now)
        cutoff = now - self.error_window_s
        while self._error_times and self._error_times[0] < cutoff:
            self._error_times.popleft()
        logger.warning(f"[ML] ONNX classifier error ({exc}).")
        if invoke_on_error and self._on_error is not None:
            try:
                self._on_error()
            except Exception:
                pass
        if len(self._error_times) >= self.max_errors:
            self._fell_back = True
            self._fallback_reason = "onnx_fallback_errors"
            self.reset()
            logger.error(
                f"[ML] Auto-fallback to {type(self._fallback).__name__}: "
                f"{len(self._error_times)} ONNX errors in {self.error_window_s:.0f}s."
            )

    def _record_and_check_latency(self, pred: PinchPrediction, wall_ms: float, *, enforce: bool = True) -> None:
        if pred.feature_ms is not None:
            self._feature_latencies_ms.append(max(0.0, float(pred.feature_ms)))
        # V051-03: Exclude buffer-fill / unready frames from ONNX latency tracking
        if not pred.ready:
            return
        elapsed_ms = max(0.0, float(pred.infer_ms)) if pred.infer_ms is not None else max(0.0, float(wall_ms))
        self._latencies_ms.append(elapsed_ms)
        if enforce and len(self._latencies_ms) >= self.min_latency_samples and self._p95_ms() > self.max_p95_ms:
            self._fell_back = True
            self._fallback_reason = "onnx_fallback_latency"
            self.reset()
            logger.warning(
                f"[ML] Auto-fallback to {type(self._fallback).__name__}: "
                f"ONNX p95={self._p95_ms():.2f}ms > {self.max_p95_ms:.1f}ms "
                f"over {len(self._latencies_ms)} inferences."
            )

    def _is_onnx_approving(self, heur_pose: PoseName, onnx_pred: PinchPrediction) -> bool:
        p_proposed = float(onnx_pred.probabilities.get(heur_pose, 0.0))
        p_idle = float(onnx_pred.probabilities.get(PoseName.IDLE, 0.0))
        return (p_proposed >= self.veto_min_prob) and (p_idle <= self.veto_idle_prob)

    def _finalize_veto_event(
        self,
        handedness: Handedness,
        ev: Optional[_HeurEventState],
        ts_ms: Optional[float],
        *,
        released_to_idle: bool,
    ) -> None:
        """Finalize an unaccepted veto event that saw ready ONNX without explicit veto as swallowed (V053-02)."""
        if ev is not None and ev.saw_ready and (not ev.ever_accepted) and (not ev.event_vetoed):
            self._veto_swallowed_events += 1
            eff_ts = float(ts_ms) if ts_ms is not None else float(ev.last_ts_ms)
            self._last_veto_by_hand[handedness] = (ev.pose, eff_ts, released_to_idle)

    def _finalize_shadow_event(self, ev: Optional[_HeurEventState]) -> None:
        """Finalize an unaccepted shadow event that saw ready ONNX without explicit veto as swallowed (V053-02)."""
        if ev is not None and ev.saw_ready and (not ev.ever_accepted) and (not ev.event_vetoed):
            self._shadow_would_swallow_events += 1

    def _track_shadow_veto_rule(
        self,
        heur_pred: PinchPrediction,
        onnx_pred: PinchPrediction,
        handedness: Handedness,
        timestamp_ms: float,
    ) -> None:
        """Mirror the exact K-frame veto confirmation rule in shadow mode so counters reflect real veto."""
        if heur_pred.pose == PoseName.IDLE:
            prev_ev = self._shadow_event_by_hand.pop(handedness, None)
            self._finalize_shadow_event(prev_ev)
            return

        cur_ev = self._shadow_event_by_hand.get(handedness)
        if cur_ev is None or cur_ev.pose != heur_pred.pose:
            self._finalize_shadow_event(cur_ev)
            cur_ev = _HeurEventState(heur_pred.pose, float(timestamp_ms))
            self._shadow_event_by_hand[handedness] = cur_ev
            self._shadow_events += 1
        cur_ev.last_ts_ms = float(timestamp_ms)

        # V053-03 (B3): Mirror consistent warm-up abstain rule in shadow mode
        if not onnx_pred.ready:
            if not cur_ev.saw_ready:
                cur_ev.ever_accepted = True
                if self.veto_scope == "entry":
                    cur_ev.accepted = True
            return

        cur_ev.saw_ready = True
        if self.veto_scope == "entry" and cur_ev.accepted:
            return

        elapsed_ms = float(timestamp_ms) - cur_ev.enter_ts_ms
        if self.veto_scope == "entry" and elapsed_ms > self.veto_entry_window_ms and (not cur_ev.event_vetoed):
            cur_ev.event_vetoed = True
            self._shadow_would_veto_events += 1

        if self._is_onnx_approving(heur_pred.pose, onnx_pred):
            cur_ev.approve_streak += 1
            if cur_ev.approve_streak >= self.veto_confirm_frames:
                cur_ev.ever_accepted = True
                if self.veto_scope == "entry":
                    cur_ev.accepted = True
            else:
                self._shadow_would_pending_frames += 1
        else:
            cur_ev.approve_streak = 0
            self._shadow_would_veto_frames += 1
            if not cur_ev.event_vetoed:
                cur_ev.event_vetoed = True
                self._shadow_would_veto_events += 1

    def predict(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
    ) -> PinchPrediction:
        if self._fell_back or self._mode == "heuristic":
            return self._fallback.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)

        if self._mode == "shadow":
            heur_pred = self._fallback.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)
            t0 = self._clock_s()
            try:
                onnx_pred = self._onnx.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)
                wall_ms = max(0.0, (self._clock_s() - t0) * 1000.0)
                self._record_and_check_latency(onnx_pred, wall_ms, enforce=False)
                if onnx_pred.ready:
                    self._shadow_frames += 1
                    if onnx_pred.pose != heur_pred.pose:
                        self._shadow_disagreements += 1
                self._track_shadow_veto_rule(heur_pred, onnx_pred, handedness, timestamp_ms)
            except Exception as exc:
                self._record_error(exc, invoke_on_error=False)
            return heur_pred

        if self._mode == "veto":
            heur_pred = self._fallback.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)
            t0 = self._clock_s()
            onnx_pred: Optional[PinchPrediction] = None
            try:
                onnx_pred = self._onnx.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)
                wall_ms = max(0.0, (self._clock_s() - t0) * 1000.0)
                self._record_and_check_latency(onnx_pred, wall_ms, enforce=True)
            except Exception as exc:
                pass_through = self.veto_on_error == "pass_through"
                self._record_error(exc, invoke_on_error=(not pass_through))
                if pass_through:
                    return heur_pred
                return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

            # CRITICAL INVARIANT: ONNX can NEVER create an activation when Heuristic predicts IDLE!
            if heur_pred.pose == PoseName.IDLE:
                prev_ev = self._heur_event_by_hand.pop(handedness, None)
                self._finalize_veto_event(handedness, prev_ev, float(timestamp_ms), released_to_idle=True)
                prev_v = self._last_veto_by_hand.get(handedness)
                if prev_v is not None and not prev_v[2]:
                    self._last_veto_by_hand[handedness] = (prev_v[0], prev_v[1], True)
                return heur_pred

            # Track heuristic rising-edge event per hand (V051-05 / V052-01 / V053-02)
            cur_ev = self._heur_event_by_hand.get(handedness)
            if cur_ev is None or cur_ev.pose != heur_pred.pose:
                if cur_ev is not None:
                    self._finalize_veto_event(handedness, cur_ev, float(timestamp_ms), released_to_idle=False)
                # Rising edge of a new heuristic pose attempt
                prev_v = self._last_veto_by_hand.get(handedness)
                if prev_v is not None and prev_v[2]:
                    if prev_v[0] == heur_pred.pose and 0.0 <= (timestamp_ms - prev_v[1]) <= 1000.0:
                        self._veto_retried_within_1s += 1
                    self._last_veto_by_hand.pop(handedness, None)
                cur_ev = _HeurEventState(heur_pred.pose, float(timestamp_ms))
                self._heur_event_by_hand[handedness] = cur_ev
            cur_ev.last_ts_ms = float(timestamp_ms)

            if self._fell_back or onnx_pred is None:
                return heur_pred

            # V051-04 / V052-01 / V053-03 (B3): Consistent abstain when ONNX is not ready:
            #   - Never seen a ready frame in this event -> pass heuristic through & mark accepted (entry scope)
            #     so subsequent ready frames do not dip to IDLE and split a click/hold.
            #   - Already accepted -> pass heuristic through.
            #   - Already saw ready frame(s), not yet accepted -> keep returning IDLE without changing streak.
            if not onnx_pred.ready:
                self._veto_abstain_not_ready += 1
                if not cur_ev.saw_ready:
                    cur_ev.ever_accepted = True
                    if self.veto_scope == "entry":
                        cur_ev.accepted = True
                    return heur_pred
                if cur_ev.accepted:
                    return heur_pred
                return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high")

            cur_ev.saw_ready = True

            # V051-05 / V052-01: Once accepted in entry scope, never interrupt an active hold
            if self.veto_scope == "entry" and cur_ev.accepted:
                return heur_pred

            # Past entry window without prior acceptance: count 1 veto event and keep requiring K approvals (late accept)
            elapsed_ms = float(timestamp_ms) - cur_ev.enter_ts_ms
            if self.veto_scope == "entry" and elapsed_ms > self.veto_entry_window_ms and (not cur_ev.event_vetoed):
                cur_ev.event_vetoed = True
                self._veto_count += 1
                self._last_veto_by_hand[handedness] = (heur_pred.pose, float(timestamp_ms), False)

            if self._is_onnx_approving(heur_pred.pose, onnx_pred):
                cur_ev.approve_streak += 1
                if cur_ev.approve_streak >= self.veto_confirm_frames:
                    cur_ev.ever_accepted = True
                    if self.veto_scope == "entry":
                        cur_ev.accepted = True
                    return heur_pred
                # Pending confirmation (< K consecutive ready approvals): stay IDLE & count pending frame
                self._veto_pending_frames += 1
                return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high")

            # Rejected by ONNX -> reset confirmation streak and veto
            cur_ev.approve_streak = 0
            self._veto_frames += 1
            if not cur_ev.event_vetoed:
                cur_ev.event_vetoed = True
                self._veto_count += 1
                self._last_veto_by_hand[handedness] = (heur_pred.pose, float(timestamp_ms), False)
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high")

        t0 = self._clock_s()
        try:
            pred = self._onnx.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)
            wall_ms = max(0.0, (self._clock_s() - t0) * 1000.0)
            self._record_and_check_latency(pred, wall_ms, enforce=True)
            return pred
        except Exception as exc:
            self._record_error(exc, invoke_on_error=True)
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

    def telemetry(self) -> Dict[str, Any]:
        rate = (self._shadow_disagreements / float(self._shadow_frames)) if self._shadow_frames > 0 else 0.0
        wv_rate = (
            self._shadow_would_veto_events / float(self._shadow_events)
            if self._shadow_events > 0
            else 0.0
        )
        return {
            "mode": self._mode,
            "active_backend": self.active_backend_name,
            "shadow_mode": self.shadow_mode,
            "shadow_frames": self._shadow_frames,
            "shadow_disagreements": self._shadow_disagreements,
            "shadow_disagreement_rate": round(rate, 4),
            "shadow_events": self._shadow_events,
            "shadow_would_veto_events": self._shadow_would_veto_events,
            "shadow_would_veto_frames": self._shadow_would_veto_frames,
            "shadow_would_swallow_events": self._shadow_would_swallow_events,
            "shadow_would_pending_frames": self._shadow_would_pending_frames,
            "shadow_would_veto_rate": round(wv_rate, 4),
            "veto_count": self._veto_count,
            "veto_frames": self._veto_frames,
            "veto_swallowed_events": self._veto_swallowed_events,
            "veto_pending_frames": self._veto_pending_frames,
            "veto_retried_within_1s": self._veto_retried_within_1s,
            "veto_abstain_not_ready": self._veto_abstain_not_ready,
            "veto_scope": self.veto_scope,
            "veto_entry_window_ms": self.veto_entry_window_ms,
            "veto_confirm_frames": self.veto_confirm_frames,
            "veto_on_error": self.veto_on_error,
            "veto_min_prob": self.veto_min_prob,
            "veto_idle_prob": self.veto_idle_prob,
            "onnx_p95_ms": round(self._p95_ms(), 2),
            "feature_p95_ms": round(self._feature_p95_ms(), 2),
            "onnx_latency_samples": len(self._latencies_ms),
            "onnx_errors": self._total_errors,
            "fell_back": self._fell_back,
            "fallback_reason": self._fallback_reason,
            "guard_active": True,
            "max_onnx_p95_ms": self.max_p95_ms,
            "model_hash": getattr(self._onnx, "model_hash", None),
            "model_sha256": getattr(self._onnx, "model_sha256", None),
            "heuristic_source_sha256": compute_heuristic_source_sha256(),
        }
