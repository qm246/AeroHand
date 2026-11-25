"""
One-Euro adaptive filtering (Casiez et al., CHI 2012) & Two-Stage LandmarkSmoother.

Combines:
- Airpoint SDK's `OneEuroFilter.ts` & `LandmarkSmoother.ts` (Apache-2.0): fast Stage-1
  defaults (`min_cutoff=2.0, beta=45.0`) and `skip_indices` (pass-through raw joint while
  keeping filter state warm for seamless transitions).
- GestureCtl AM-7 §6 large-gap reset (`dt >= gap_reset_ms`): prevents stale velocity spikes
  after camera hiccups or hand reappearance.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from aerohand.core.types import Landmark


class OneEuroFilter1D:
    """Adaptive 1D low-pass filter with large-gap reset safety."""

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.5,
        d_cutoff: float = 1.0,
        gap_reset_ms: float = 100.0,
    ) -> None:
        self.min_cutoff: float = float(min_cutoff)
        self.beta: float = float(beta)
        self.d_cutoff: float = float(d_cutoff)
        self.gap_reset_ms: float = float(gap_reset_ms)

        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev_ms: Optional[float] = None

    @staticmethod
    def _alpha(te_s: float, cutoff: float) -> float:
        if cutoff <= 0.0 or te_s <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te_s)

    def filter(self, x: float, timestamp_ms: float) -> float:
        """Filter scalar `x` at monotonic timestamp `timestamp_ms`."""
        if self._t_prev_ms is None or self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev_ms = timestamp_ms
            return x

        dt_ms = timestamp_ms - self._t_prev_ms
        if dt_ms <= 0.0:
            return self._x_prev

        # Large-gap reset (GestureCtl AM-7 §6) prevents stale-state lag after hiccups
        if self.gap_reset_ms > 0.0 and dt_ms >= self.gap_reset_ms:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev_ms = timestamp_ms
            return x

        te_s = max(0.001, dt_ms / 1000.0)
        dx = (x - self._x_prev) / te_s
        a_d = self._alpha(te_s, self.d_cutoff)
        edx = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = edx

        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self._alpha(te_s, cutoff)
        result = a * x + (1.0 - a) * self._x_prev

        self._x_prev = result
        self._t_prev_ms = timestamp_ms
        return result

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev_ms = None


class LandmarkSmoother:
    """
    Stage-1 21-landmark One-Euro smoother keyed by hand label ("Left" / "Right").
    Supports `skip_indices` (keeps filter state warm while emitting raw coordinates).
    """

    def __init__(
        self,
        min_cutoff: float = 2.0,
        beta: float = 45.0,
        d_cutoff: float = 1.0,
        gap_reset_ms: float = 100.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.gap_reset_ms = float(gap_reset_ms)
        self._filters: Dict[str, List[Tuple[OneEuroFilter1D, OneEuroFilter1D, OneEuroFilter1D]]] = {}

    def _init_hand(self, label: str) -> None:
        self._filters[label] = [
            (
                OneEuroFilter1D(self.min_cutoff, self.beta, self.d_cutoff, self.gap_reset_ms),
                OneEuroFilter1D(self.min_cutoff, self.beta, self.d_cutoff, self.gap_reset_ms),
                OneEuroFilter1D(self.min_cutoff, self.beta, self.d_cutoff, self.gap_reset_ms),
            )
            for _ in range(21)
        ]

    def update_config(self, min_cutoff: float, beta: float, d_cutoff: Optional[float] = None) -> None:
        new_dc = self.d_cutoff if d_cutoff is None else float(d_cutoff)
        if self.min_cutoff != float(min_cutoff) or self.beta != float(beta) or self.d_cutoff != new_dc:
            self.min_cutoff = float(min_cutoff)
            self.beta = float(beta)
            self.d_cutoff = new_dc
            self.reset()

    def reset(self, label: Optional[str] = None) -> None:
        if label is None:
            self._filters.clear()
        else:
            self._filters.pop(label, None)

    def smooth(
        self,
        landmarks: Sequence[Landmark],
        label: str,
        timestamp_ms: float,
        skip_indices: Iterable[int] = (),
    ) -> List[Landmark]:
        if len(landmarks) < 21:
            return list(landmarks)
        if label not in self._filters:
            self._init_hand(label)

        skip_set: Set[int] = set(skip_indices)
        hand_filters = self._filters[label]
        out: List[Landmark] = []
        for idx in range(21):
            lm = landmarks[idx]
            fx, fy, fz = hand_filters[idx]
            sx = fx.filter(lm.x, timestamp_ms)
            sy = fy.filter(lm.y, timestamp_ms)
            sz = fz.filter(lm.z, timestamp_ms)
            if idx in skip_set:
                out.append(lm)
            else:
                out.append(Landmark(x=sx, y=sy, z=sz, visibility=lm.visibility))
        return out
