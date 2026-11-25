"""
MediaPipe HandLandmarker wrapper with automatic GPU -> CPU fallback (Airpoint pattern)
and strictly monotonic VIDEO-mode timestamps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from aerohand.config import repo_root
from aerohand.core.types import Handedness, HandInput, Landmark

logger = logging.getLogger(__name__)


def _safe_float(val: object, default: float = 1.0) -> float:
    return default if val is None else float(val)


def resolve_model_path(model_path: str) -> Path:
    p = Path(model_path)
    if p.is_absolute() and p.is_file():
        return p
    cand = repo_root() / model_path
    if cand.is_file():
        return cand
    if p.is_file():
        return p.resolve()
    raise FileNotFoundError(f"MediaPipe hand model not found at '{model_path}' or '{cand}'.")


class MediaPipeHandTracker:
    """Runs MediaPipe HandLandmarker in VIDEO mode and returns raw 2D + metric 3D landmarks."""

    def __init__(
        self,
        model_path: str = "models/hand_landmarker.task",
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.55,
        min_hand_presence_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        delegate: str = "GPU",
        mirror_handedness: bool = True,
    ) -> None:
        self.model_path = str(resolve_model_path(model_path))
        self.num_hands = int(num_hands)
        self.mirror_handedness = bool(mirror_handedness)
        self._last_ts_ms: int = -1
        self._delegate_used: str = "CPU"
        self._landmarker: Optional[HandLandmarker] = None

        preferred = str(delegate).upper()
        order = ["GPU", "CPU"] if preferred == "GPU" else ["CPU"]
        last_err: Optional[Exception] = None
        for deg in order:
            try:
                deg_opt = BaseOptions.Delegate.GPU if deg == "GPU" else BaseOptions.Delegate.CPU
                opts = HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=self.model_path, delegate=deg_opt),
                    running_mode=RunningMode.VIDEO,
                    num_hands=self.num_hands,
                    min_hand_detection_confidence=float(min_hand_detection_confidence),
                    min_hand_presence_confidence=float(min_hand_presence_confidence),
                    min_tracking_confidence=float(min_tracking_confidence),
                )
                self._landmarker = HandLandmarker.create_from_options(opts)
                self._delegate_used = deg
                logger.info(f"[TRACKER] MediaPipe initialized (delegate={deg}, num_hands={self.num_hands})")
                break
            except Exception as exc:
                last_err = exc
                logger.warning(f"[TRACKER] Delegate {deg} unavailable ({exc}); trying fallback...")
        if self._landmarker is None:
            raise RuntimeError(f"Failed to initialize MediaPipe HandLandmarker: {last_err}")

    @property
    def delegate_used(self) -> str:
        return self._delegate_used

    def _normalize_handedness(self, raw_label: str) -> Handedness:
        label = raw_label.strip().lower()
        # When camera frame is already horizontally mirrored, MediaPipe reports the mirror
        # handedness (a user's physical Right hand appears as "Left" in selfie view).
        if self.mirror_handedness:
            return "Right" if label == "left" else "Left"
        return "Left" if label == "left" else "Right"

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> List[HandInput]:
        assert self._landmarker is not None
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms

        if self._delegate_used == "GPU":
            frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        res = self._landmarker.detect_for_video(mp_img, timestamp_ms)
        if not res.hand_landmarks:
            return []

        hands: List[HandInput] = []
        for idx, raw_lms in enumerate(res.hand_landmarks):
            lms = [
                Landmark(
                    x=float(p.x),
                    y=float(p.y),
                    z=float(p.z),
                    visibility=_safe_float(getattr(p, "visibility", None), 1.0),
                )
                for p in raw_lms
            ]
            wlms: Optional[List[Landmark]] = None
            if res.hand_world_landmarks and idx < len(res.hand_world_landmarks):
                wlms = [
                    Landmark(
                        x=float(p.x),
                        y=float(p.y),
                        z=float(p.z),
                        visibility=_safe_float(getattr(p, "visibility", None), 1.0),
                    )
                    for p in res.hand_world_landmarks[idx]
                ]
            raw_label = "Right"
            score = 1.0
            if res.handedness and idx < len(res.handedness) and res.handedness[idx]:
                cat = res.handedness[idx][0]
                raw_label = cat.category_name or cat.display_name or "Right"
                score = _safe_float(getattr(cat, "score", None), 1.0)
            hands.append(
                HandInput(
                    handedness=self._normalize_handedness(raw_label),
                    landmarks=lms,
                    world_landmarks=wlms,
                    landmarks_unfiltered=list(lms),
                    confidence=score,
                    timestamp_ms=timestamp_ms,
                )
            )
        return hands

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
