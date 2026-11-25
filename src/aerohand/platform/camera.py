"""
Threaded AVFoundation webcam capture (adapted from GestureCtl `camera.py`).
Drops stale backlog frames so downstream tracking always runs on the freshest image.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraFrame:
    success: bool
    frame: Optional[np.ndarray]
    timestamp_ms: int
    frame_id: int


class ThreadedCamera:
    """Zero-backlog background camera reader with horizontal mirroring and monotonic timestamps."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps_cap: int = 60,
        mirror: bool = True,
    ) -> None:
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.fps_cap = int(fps_cap)
        self.mirror = bool(mirror)

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts_ms: int = 0
        self._frame_id: int = 0
        self._has_new: bool = False

    def start(self) -> ThreadedCamera:
        if self._running:
            return self
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self.camera_index}. "
                "Verify Camera permission in System Settings -> Privacy & Security -> Camera."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps_cap)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="AeroHandCamera", daemon=True)
        self._thread.start()
        return self

    def _worker(self) -> None:
        while self._running and self._cap is not None:
            ok, raw = self._cap.read()
            if not ok or raw is None:
                time.sleep(0.004)
                continue
            ts_ms = int(time.monotonic_ns() // 1_000_000)
            frame = cv2.flip(raw, 1) if self.mirror else raw
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            with self._cond:
                self._frame_id += 1
                self._latest_frame = frame
                self._latest_ts_ms = ts_ms
                self._has_new = True
                self._cond.notify_all()

    def read(self, timeout_s: float = 0.04) -> CameraFrame:
        with self._cond:
            if not self._has_new and self._running:
                self._cond.wait(timeout=timeout_s)
            if not self._has_new or self._latest_frame is None:
                return CameraFrame(False, None, 0, self._frame_id)
            frame = self._latest_frame.copy()
            ts_ms = self._latest_ts_ms
            fid = self._frame_id
            self._has_new = False
            return CameraFrame(True, frame, ts_ms, fid)

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        with self._cond:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._latest_frame = None
