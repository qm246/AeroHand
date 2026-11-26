"""Abstract interface for pluggable AeroHand pinch/pose classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence

from aerohand.core.types import Handedness, Landmark, PoseName, PoseProbabilities

ConfidenceLevel = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class PinchPrediction:
    pose: PoseName
    probabilities: PoseProbabilities
    confidence: ConfidenceLevel
    ready: bool = True
    infer_ms: Optional[float] = None
    feature_ms: Optional[float] = None


class PinchClassifier(Protocol):
    """Any heuristic or ML model outputting canonical 7-class PoseProbabilities."""

    def predict(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        timestamp_ms: float,
    ) -> PinchPrediction:
        ...

    def reset(self, hand: Optional[Handedness] = None) -> None:
        ...
