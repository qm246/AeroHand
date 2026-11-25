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
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Sequence

import numpy as np

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName, idle_probabilities
from aerohand.ml.base import PinchPrediction
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
        logits = np.asarray(self._session.run(None, {self._input_name: seq})[0]).reshape(-1)
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
