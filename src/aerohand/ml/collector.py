"""
Privacy-Preserving Landmarks-Only Pinch Data Collector.
Adapted from Airpoint SDK `PinchDataCollector.ts` (Apache-2.0): NEVER saves video pixels;
records raw 2D+3D landmarks, handedness, confidence, and pose label for self-training TCN.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from aerohand.core.types import ALL_POSE_NAMES, Handedness, Landmark, PoseName

POSE_TO_ID: Dict[PoseName, int] = {pose: i for i, pose in enumerate(ALL_POSE_NAMES)}


class PinchDataCollector:
    """Interactive labeled session recorder for training a personal ONNX TCN model."""

    def __init__(self) -> None:
        self._samples: List[Dict[str, Any]] = []
        self._recording: bool = False
        self._current_pose: PoseName = PoseName.IDLE
        self._started_at_iso: str = time.strftime("%Y-%m-%dT%H:%M:%S")

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_pose(self) -> PoseName:
        return self._current_pose

    def toggle_recording(self) -> bool:
        self._recording = not self._recording
        return self._recording

    def set_pose(self, pose: PoseName) -> None:
        self._current_pose = pose

    def add_frame(
        self,
        landmarks_2d: Sequence[Landmark],
        world_landmarks: Optional[Sequence[Landmark]],
        handedness: Handedness,
        confidence: float,
        timestamp_ms: int,
    ) -> None:
        if not self._recording or len(landmarks_2d) < 21:
            return
        flat_2d = [c for lm in landmarks_2d[:21] for c in (lm.x, lm.y, lm.z)]
        flat_3d = (
            [c for lm in world_landmarks[:21] for c in (lm.x, lm.y, lm.z)]
            if (world_landmarks is not None and len(world_landmarks) >= 21)
            else []
        )
        self._samples.append(
            {
                "timestamp_ms": int(timestamp_ms),
                "label": POSE_TO_ID[self._current_pose],
                "label_name": self._current_pose.value,
                "handedness": handedness,
                "confidence": round(float(confidence), 4),
                "landmarks_2d": flat_2d,
                "world_landmarks": flat_3d,
            }
        )

    def counts(self) -> Dict[str, int]:
        out = {p.value: 0 for p in ALL_POSE_NAMES}
        for s in self._samples:
            name = s["label_name"]
            if name in out:
                out[name] += 1
        return out

    def save(self, out_dir: str | Path) -> Path:
        out_root = Path(out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_fp = out_root / f"aerohand_pinch_{stamp}.json"
        payload = {
            "schema": "aerohand_pinch_v1",
            "collected_at": self._started_at_iso,
            "total_samples": len(self._samples),
            "counts": self.counts(),
            "samples": self._samples,
        }
        out_fp.write_text(json.dumps(payload), encoding="utf-8")
        return out_fp
