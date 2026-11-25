"""
Threaded AVFoundation webcam capture & multi-camera device discovery (Airpoint CameraManager parity).
Drops stale backlog frames, supports switching camera devices / resolution / aspect ratios,
and computes low-cost luminance for smart low-light warnings.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraDeviceInfo:
    index: int
    unique_id: str
    name: str
    model_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "unique_id": self.unique_id,
            "name": self.name,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class CameraFrame:
    success: bool
    frame: Optional[np.ndarray]
    timestamp_ms: int
    frame_id: int
    brightness: Optional[float] = None


def list_cameras() -> List[CameraDeviceInfo]:
    """Enumerate available macOS video capture devices via AVFoundation."""
    devices: List[CameraDeviceInfo] = []
    try:
        import AVFoundation

        raw_devs = list(AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo) or [])
        for idx, dev in enumerate(raw_devs):
            uid = str(dev.uniqueID() or f"cam_{idx}")
            name = str(dev.localizedName() or f"Camera {idx}")
            model = str(dev.modelID() or "")
            devices.append(CameraDeviceInfo(index=idx, unique_id=uid, name=name, model_id=model))
    except Exception as exc:
        logger.debug(f"[CAMERA] AVFoundation enumeration fallback: {exc}")

    if not devices:
        devices.append(CameraDeviceInfo(index=0, unique_id="default", name="Default Camera (0)"))
    return devices


def resolve_camera_index(device_id: str = "", fallback_index: int = 0) -> int:
    """Resolve an AVFoundation `unique_id` to an OpenCV AVFoundation index."""
    if not device_id or device_id == "default":
        return int(fallback_index)
    for dev in list_cameras():
        if dev.unique_id == device_id:
            return dev.index
    return int(fallback_index)


def _crop_aspect_ratio(frame: np.ndarray, aspect_ratio: str) -> np.ndarray:
    if aspect_ratio in ("auto", "", "4:3"):
        return frame
    h, w = frame.shape[:2]
    targets = {"1:1": 1.0, "16:9": 16.0 / 9.0, "9:16": 9.0 / 16.0}
    target = targets.get(aspect_ratio)
    if target is None or h <= 0 or w <= 0:
        return frame
    cur = w / float(h)
    if abs(cur - target) < 0.03:
        return frame
    if cur > target:
        new_w = max(16, int(round(h * target)))
        x0 = (w - new_w) // 2
        return frame[:, x0 : x0 + new_w]
    new_h = max(16, int(round(w / target)))
    y0 = (h - new_h) // 2
    return frame[y0 : y0 + new_h, :]


class ThreadedCamera:
    """Zero-backlog background camera reader with horizontal mirroring and monotonic timestamps."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps_cap: int = 60,
        mirror: bool = True,
        aspect_ratio: str = "4:3",
        device_id: str = "",
    ) -> None:
        self.device_id = str(device_id or "")
        self.camera_index = resolve_camera_index(self.device_id, camera_index)
        self.width = int(width)
        self.height = int(height)
        self.fps_cap = int(fps_cap)
        self.mirror = bool(mirror)
        self.aspect_ratio = str(aspect_ratio or "4:3")

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts_ms: int = 0
        self._latest_brightness: Optional[float] = None
        self._frame_id: int = 0
        self._has_new: bool = False

    def start(self) -> ThreadedCamera:
        if self._running:
            return self
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened() and self.camera_index != 0:
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
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
            frame = _crop_aspect_ratio(frame, self.aspect_ratio)
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            brightness = self._latest_brightness
            if (self._frame_id % 10) == 0:
                small = frame[::8, ::8]
                brightness = float(np.mean(small))

            with self._cond:
                self._frame_id += 1
                self._latest_frame = frame
                self._latest_ts_ms = ts_ms
                self._latest_brightness = brightness
                self._has_new = True
                self._cond.notify_all()

    def read(self, timeout_s: float = 0.04) -> CameraFrame:
        with self._cond:
            if not self._has_new and self._running:
                self._cond.wait(timeout=timeout_s)
            if not self._has_new or self._latest_frame is None:
                return CameraFrame(False, None, 0, self._frame_id, self._latest_brightness)
            frame = self._latest_frame.copy()
            ts_ms = self._latest_ts_ms
            fid = self._frame_id
            br = self._latest_brightness
            self._has_new = False
            return CameraFrame(True, frame, ts_ms, fid, br)

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
