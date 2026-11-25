"""
Self-Trained ONNX Causal TCN Classifier Slot.
Adapted from Airpoint SDK `ONNXGestureClassifier.ts` (Apache-2.0) for Python `onnxruntime`.

Expects a user-trained ONNX model taking `[1, sequence_length, feature_dim]` float32
(default `[1, 5, 348]` from `extract_invariant_3d_features`) and outputting `[1, 7]` logits
ordered by `ALL_POSE_NAMES`, paired with a normalizer JSON `{"mean": [...], "scale": [...]}`.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional, Sequence

import numpy as np

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName, idle_probabilities
from aerohand.ml.base import PinchPrediction
from aerohand.ml.features_3d import FEATURE_DIM_3D, extract_invariant_3d_features

logger = logging.getLogger(__name__)


class ONNXPinchClassifier:
    """Streaming sliding-window ONNX Runtime classifier (fail-open to IDLE if unloaded)."""

    def __init__(
        self,
        model_path: str | Path,
        normalizer_path: str | Path,
        sequence_length: int = 5,
    ) -> None:
        self.model_path = Path(model_path)
        self.normalizer_path = Path(normalizer_path)
        self.sequence_length = max(1, int(sequence_length))
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self._session: Any = None
        self._input_name: str = "input"
        self._mean: np.ndarray = np.zeros(FEATURE_DIM_3D, dtype=np.float32)
        self._scale: np.ndarray = np.ones(FEATURE_DIM_3D, dtype=np.float32)
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Run `uv sync --extra ml` to enable ONNX TCN inference."
            ) from exc

        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.normalizer_path.is_file():
            raise FileNotFoundError(f"Normalizer JSON not found: {self.normalizer_path}")

        norm = json.loads(self.normalizer_path.read_text(encoding="utf-8"))
        self._mean = np.asarray(norm["mean"], dtype=np.float32)
        scale = np.asarray(norm["scale"], dtype=np.float32)
        self._scale = np.where(np.abs(scale) > 1e-8, scale, 1.0).astype(np.float32)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(f"[ONNX] Loaded self-trained TCN model: {self.model_path.name}")

    def reset(self) -> None:
        self._buffer.clear()

    def predict(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
    ) -> PinchPrediction:
        if self._session is None:
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

        feat = extract_invariant_3d_features(landmarks_2d, world_landmarks, handedness=handedness)
        if feat is None or feat.shape[0] != self._mean.shape[0]:
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

        norm_feat = (feat - self._mean) / self._scale
        self._buffer.append(norm_feat)
        if len(self._buffer) < self.sequence_length:
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low")

        seq = np.stack(list(self._buffer), axis=0)[None, :, :]  # (1, seq_len, F)
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
