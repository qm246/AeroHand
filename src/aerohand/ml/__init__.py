"""Pluggable gesture classifier backends (3D Heuristic MVP & Self-Trained ONNX TCN)."""

from aerohand.ml.base import PinchClassifier, PinchPrediction
from aerohand.ml.collector import PinchDataCollector
from aerohand.ml.features_3d import FEATURE_DIM_3D, extract_invariant_3d_features
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier
from aerohand.ml.onnx_classifier import ONNXPinchClassifier

__all__ = [
    "FEATURE_DIM_3D",
    "Heuristic3DPinchClassifier",
    "ONNXPinchClassifier",
    "PinchClassifier",
    "PinchDataCollector",
    "PinchPrediction",
    "extract_invariant_3d_features",
]
