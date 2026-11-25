"""
Threaded AVFoundation webcam capture & multi-camera device discovery (Airpoint CameraManager parity).
Drops stale backlog frames, supports switching camera devices / resolution / aspect ratios,
and computes low-cost luminance for smart low-light warnings.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    frame_aspect: float = 4.0 / 3.0


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
    """Center-crop `frame` to the configured aspect ratio without stretching."""
    ar = str(aspect_ratio or "auto").strip()
    if ar in ("auto", ""):
        return frame
    h, w = frame.shape[:2]
    targets = {
        "4:3": 4.0 / 3.0,
        "1:1": 1.0,
        "16:9": 16.0 / 9.0,
        "9:16": 9.0 / 16.0,
    }
    target = targets.get(ar)
    if target is None or h <= 0 or w <= 0:
        return frame
    cur = w / float(h)
    if abs(cur - target) < 0.01:
        return frame
    if cur > target:
        new_w = max(16, int(round(h * target)))
        x0 = (w - new_w) // 2
        return frame[:, x0 : x0 + new_w]
    new_h = max(16, int(round(w / target)))
    y0 = (h - new_h) // 2
    return frame[y0 : y0 + new_h, :]


def prepare_camera_frame(
    raw_frame: np.ndarray,
    *,
    target_width: int = 640,
    target_height: int = 480,
    aspect_ratio: str = "4:3",
    mirror: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Mirror, center-crop to `aspect_ratio`, and uniformly resize (`cv2.INTER_AREA`)
    so geometry stays strictly isotropic (circles stay circular, zero axis stretch).
    Returns `(processed_frame, frame_aspect)`.
    """
    frame = cv2.flip(raw_frame, 1) if mirror else raw_frame
    frame = _crop_aspect_ratio(frame, aspect_ratio)
    h, w = frame.shape[:2]
    if h > 0 and w > 0 and (target_width > 0 and target_height > 0):
        scale = min(target_width / float(w), target_height / float(h))
        if abs(scale - 1.0) > 1e-3:
            new_w = max(16, int(round(w * scale)))
            new_h = max(16, int(round(h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
            h, w = frame.shape[:2]
    aspect = (w / float(h)) if h > 0 else (4.0 / 3.0)
    return frame, aspect


class ThreadedCamera:
    """
    Zero-backlog background camera reader with a size-1 drop-oldest queue.
    The capture thread performs ONLY raw capture; mirroring, aspect-correct cropping,
    uniform resizing, and luminance estimation execute on the consumer processing thread.
    """

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
        self._raw_queue: queue.Queue[Tuple[np.ndarray, int, int]] = queue.Queue(maxsize=1)
        self._latest_brightness: Optional[float] = None
        self._frame_id: int = 0

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

    def _enqueue_latest(self, item: Tuple[np.ndarray, int, int]) -> None:
        while True:
            try:
                self._raw_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._raw_queue.get_nowait()
                except queue.Empty:
                    pass

    def _worker(self) -> None:
        while self._running and self._cap is not None:
            ok, raw = self._cap.read()
            if not ok or raw is None:
                time.sleep(0.004)
                continue
            ts_ms = int(time.monotonic_ns() // 1_000_000)
            self._frame_id += 1
            self._enqueue_latest((raw, ts_ms, self._frame_id))

    def read(self, timeout_s: float = 0.04) -> CameraFrame:
        try:
            raw, ts_ms, fid = self._raw_queue.get(timeout=timeout_s)
        except queue.Empty:
            return CameraFrame(False, None, 0, self._frame_id, self._latest_brightness)

        frame, frame_aspect = prepare_camera_frame(
            raw,
            target_width=self.width,
            target_height=self.height,
            aspect_ratio=self.aspect_ratio,
            mirror=self.mirror,
        )
        brightness = self._latest_brightness
        if brightness is None or (fid % 10) == 0:
            small = frame[::8, ::8]
            brightness = float(np.mean(small))
            self._latest_brightness = brightness

        return CameraFrame(True, frame, ts_ms, fid, brightness, frame_aspect)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        while True:
            try:
                self._raw_queue.get_nowait()
            except queue.Empty:
                break
        if self._cap is not None:
            self._cap.release()
            self._cap = None
