"""
Self-Trained ONNX Causal TCN Classifier (Dual-Feature: 348-dim Invariant3D & 885-dim AirMouse).
Adapted from Airpoint SDK `ONNXGestureClassifier.ts` (Apache-2.0) for Python `onnxruntime`.

Upgrades:
  - Per-hand independent 5-frame causal ring buffers (`Left` & `Right` never cross-contaminate).
  - Auto-detects `invariant348` vs `airmouse885` feature extractor from normalizer JSON metadata/length.
  - Consistent Left-hand X mirroring for both feature sets.
  - Fail-open to IDLE if model/normalizer is missing or mismatched.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional, Sequence

import numpy as np

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName, idle_probabilities
from aerohand.ml.base import PinchClassifier, PinchPrediction
from aerohand.ml.features_3d import FEATURE_DIM_3D, extract_invariant_3d_features
from aerohand.ml.features_885 import AIRMOUSE_885_DIM, extract_airmouse_885_features

logger = logging.getLogger(__name__)


class ONNXPinchClassifier:
    """Streaming per-hand sliding-window ONNX Runtime classifier."""

    def __init__(
        self,
        model_path: str | Path,
        normalizer_path: str | Path,
        sequence_length: int = 5,
        feature_set: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.normalizer_path = Path(normalizer_path)
        self.sequence_length = max(1, int(sequence_length))
        self.feature_set: str = feature_set or "auto"
        self._buffers: Dict[Handedness, Deque[np.ndarray]] = {
            "Left": deque(maxlen=self.sequence_length),
            "Right": deque(maxlen=self.sequence_length),
        }
        self._session: Any = None
        self._input_name: str = "input"
        self._mean: np.ndarray = np.zeros(FEATURE_DIM_3D, dtype=np.float32)
        self._scale: np.ndarray = np.ones(FEATURE_DIM_3D, dtype=np.float32)
        self._temperature: float = 1.0
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
        logger.info(
            f"[ONNX] Loaded TCN model: {self.model_path.name} "
            f"(feature_set={self.feature_set}, dim={self._mean.shape[0]}, seq={self.sequence_length})"
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
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

        if self.feature_set == "airmouse885":
            feat = extract_airmouse_885_features(landmarks_2d, world_landmarks, handedness=handedness)
        else:
            feat = extract_invariant_3d_features(landmarks_2d, world_landmarks, handedness=handedness)

        if feat is None or feat.shape[0] != self._mean.shape[0]:
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

        norm_feat = (feat - self._mean) / self._scale
        buf = self._buffers[handedness]
        buf.append(norm_feat)
        if len(buf) < self.sequence_length:
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

        seq = np.stack(list(buf), axis=0)[None, :, :]  # (1, seq_len, F)
        logits = np.asarray(self._session.run(None, {self._input_name: seq})[0]).reshape(-1).astype(np.float64)
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
        return PinchPrediction(pose=best_pose, probabilities=probs, confidence=conf)


class GuardedClassifier:
    """
    Production safety wrapper for ONNX TCN + Heuristic3D fallback:
      - Shadow mode: runs ONNX alongside Heuristic3D and logs disagreement rates without acting on ONNX.
      - Auto-fallback: switches back to Heuristic3D if ONNX inference p95 exceeds `max_p95_ms` (default 8 ms)
        or if ONNX errors occur >= `max_errors` (3) times within `error_window_s` (10 s).
      - Fail-open: any ONNX error returns `PoseName.IDLE` and invokes `on_error` (`release_all()`).
      - Buffer hygiene: propagates per-hand `reset(hand)` to both classifiers.
    """

    def __init__(
        self,
        onnx_classifier: PinchClassifier,
        fallback_classifier: PinchClassifier,
        *,
        shadow_mode: bool = False,
        max_p95_ms: float = 8.0,
        max_errors: int = 3,
        error_window_s: float = 10.0,
        on_error: Optional[Callable[[], None]] = None,
        clock_s: Optional[Callable[[], float]] = None,
    ) -> None:
        self._onnx = onnx_classifier
        self._fallback = fallback_classifier
        self.shadow_mode: bool = bool(shadow_mode)
        self.max_p95_ms: float = max(0.5, float(max_p95_ms))
        self.max_errors: int = max(1, int(max_errors))
        self.error_window_s: float = max(1.0, float(error_window_s))
        self._on_error = on_error
        self._clock_s: Callable[[], float] = clock_s or time.monotonic

        self._latencies_ms: Deque[float] = deque(maxlen=64)
        self._error_times: Deque[float] = deque()
        self._total_errors: int = 0
        self._fell_back: bool = False
        self._fallback_reason: Optional[str] = None
        self._shadow_frames: int = 0
        self._shadow_disagreements: int = 0

    @property
    def fell_back(self) -> bool:
        return self._fell_back

    @property
    def fallback_reason(self) -> Optional[str]:
        return self._fallback_reason

    @property
    def active_backend_name(self) -> str:
        if self.shadow_mode:
            return f"{type(self._fallback).__name__} (Shadow: ONNX)"
        if self._fell_back:
            return f"{type(self._fallback).__name__} (Fallback)"
        return type(self._onnx).__name__

    def set_on_error(self, callback: Optional[Callable[[], None]]) -> None:
        self._on_error = callback

    def reset(self, hand: Optional[Handedness] = None) -> None:
        for clf in (self._onnx, self._fallback):
            try:
                clf.reset(hand)  # type: ignore[call-arg]
            except TypeError:
                clf.reset()

    def _p95_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        arr = sorted(self._latencies_ms)
        idx = min(len(arr) - 1, int(round(0.95 * (len(arr) - 1))))
        return float(arr[idx])

    def _record_error(self, exc: Exception) -> None:
        now = self._clock_s()
        self._total_errors += 1
        self._error_times.append(now)
        cutoff = now - self.error_window_s
        while self._error_times and self._error_times[0] < cutoff:
            self._error_times.popleft()
        logger.warning(f"[ML] ONNX classifier error ({exc}); fail-open to IDLE.")
        if self._on_error is not None:
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

    def _check_latency_guard(self, elapsed_ms: float) -> None:
        self._latencies_ms.append(elapsed_ms)
        if len(self._latencies_ms) >= 10 and self._p95_ms() > self.max_p95_ms:
            self._fell_back = True
            self._fallback_reason = "onnx_fallback_latency"
            self.reset()
            logger.warning(
                f"[ML] Auto-fallback to {type(self._fallback).__name__}: "
                f"ONNX p95={self._p95_ms():.2f}ms > {self.max_p95_ms:.1f}ms."
            )

    def predict(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
    ) -> PinchPrediction:
        if self._fell_back:
            return self._fallback.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)

        if self.shadow_mode:
            heur_pred = self._fallback.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)
            t0 = self._clock_s()
            try:
                onnx_pred = self._onnx.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)
                dt_ms = max(0.0, (self._clock_s() - t0) * 1000.0)
                self._latencies_ms.append(dt_ms)
                self._shadow_frames += 1
                if onnx_pred.pose != heur_pred.pose:
                    self._shadow_disagreements += 1
            except Exception as exc:
                self._record_error(exc)
            return heur_pred

        t0 = self._clock_s()
        try:
            pred = self._onnx.predict(landmarks_2d, world_landmarks, handedness, timestamp_ms)
            dt_ms = max(0.0, (self._clock_s() - t0) * 1000.0)
            self._check_latency_guard(dt_ms)
            return pred
        except Exception as exc:
            self._record_error(exc)
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

    def telemetry(self) -> Dict[str, Any]:
        rate = (self._shadow_disagreements / float(self._shadow_frames)) if self._shadow_frames > 0 else 0.0
        return {
            "active_backend": self.active_backend_name,
            "shadow_mode": self.shadow_mode,
            "shadow_frames": self._shadow_frames,
            "shadow_disagreements": self._shadow_disagreements,
            "shadow_disagreement_rate": round(rate, 4),
            "onnx_p95_ms": round(self._p95_ms(), 2),
            "onnx_errors": self._total_errors,
            "fell_back": self._fell_back,
            "fallback_reason": self._fallback_reason,
        }
