"""
Local macOS Voice Dictation Engine (Vietnamese `vi-VN` & English `en-US`).

Supports:
  - Hold-to-dictate (`thumb_index_pinch`) and Toggle-to-dictate (`Palm Wheel R4`).
  - Live microphone RMS level (`0.0..1.0`) for UI/overlay waveform feedback.
  - Primary engine: Native macOS `Speech.SFSpeechRecognizer` (on-device when supported,
    full Vietnamese diacritics with zero extra model downloads).
  - Optional engine: `mlx_whisper` (Apple Silicon local Whisper if user installs it).
  - UTF-8 safe insertion via `MacOSInputController.paste_text_utf8` (prevents Telex/EVKey corruption).
"""

from __future__ import annotations

import logging
import re
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

DEFAULT_HALLUCINATION_BLOCKLIST: Dict[str, List[str]] = {
    "vi-VN": [
        "hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn",
        "cảm ơn các bạn đã theo dõi và hẹn gặp lại",
        "cảm ơn đã xem video",
        "đăng ký kênh",
        "nhớ đăng ký kênh nhé",
        "phụ đề bởi",
    ],
    "en-US": [
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "subtitles by",
        "you",
    ],
}

_PUNCT_STRIP_RE = re.compile(r"^[\s\.,!?;:\"'`~…\-–—()\[\]]+|[\s\.,!?;:\"'`~…\-–—()\[\]]+$")
_WS_COLLAPSE_RE = re.compile(r"\s+")


def normalize_transcript_for_blocklist(text: str) -> str:
    """Normalize a recognized transcript for exact-phrase hallucination blocklist matching."""
    cleaned = _WS_COLLAPSE_RE.sub(" ", str(text or "").strip().lower())
    return _PUNCT_STRIP_RE.sub("", cleaned).strip()


def is_hallucinated_transcript(
    text: str,
    language: str,
    blocklist: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Return True if the entire normalized transcript matches a known silence-hallucination phrase."""
    norm = normalize_transcript_for_blocklist(text)
    if not norm:
        return True
    bl = blocklist if blocklist is not None else DEFAULT_HALLUCINATION_BLOCKLIST
    candidates: List[str] = []
    if language in bl and isinstance(bl[language], list):
        candidates.extend(bl[language])
    lang_prefix = (language or "").split("-")[0].lower()
    if lang_prefix:
        for k, vals in bl.items():
            if k != language and k.split("-")[0].lower() == lang_prefix and isinstance(vals, list):
                candidates.extend(vals)
    for phrase in candidates:
        if normalize_transcript_for_blocklist(phrase) == norm:
            return True
    return False


def evaluate_audio_vad(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    min_duration_s: float = 0.6,
    min_rms: float = 0.01,
    min_voiced_ms: float = 250.0,
    frame_ms: float = 20.0,
) -> Tuple[bool, str, Dict[str, float]]:
    """
    Pure energy-based speech VAD gate (V031-02).
    Returns `(accepted, reason, metrics)` where reason is `"ok"`, `"too_short"`, or `"silent"`.
    """
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr = max(1, int(sample_rate))
    duration_s = float(arr.size) / float(sr)
    if arr.size == 0:
        return False, "too_short", {"duration_s": 0.0, "overall_rms": 0.0, "peak_rms": 0.0, "voiced_ms": 0.0}

    overall_rms = float(np.sqrt(np.mean(arr * arr)))
    frame_len = max(1, int(round(sr * (frame_ms / 1000.0))))
    n_full = arr.size // frame_len
    peak_rms = 0.0
    voiced_ms = 0.0
    if n_full > 0:
        frames = arr[: n_full * frame_len].reshape(n_full, frame_len)
        frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
        peak_rms = float(np.max(frame_rms))
        voiced_ms = float(np.sum(frame_rms >= min_rms)) * frame_ms
    rem = arr[n_full * frame_len :]
    if rem.size >= max(1, frame_len // 2):
        rem_rms = float(np.sqrt(np.mean(rem * rem)))
        peak_rms = max(peak_rms, rem_rms)
        if rem_rms >= min_rms:
            voiced_ms += (float(rem.size) / float(sr)) * 1000.0

    metrics = {
        "duration_s": round(duration_s, 3),
        "overall_rms": round(overall_rms, 5),
        "peak_rms": round(peak_rms, 5),
        "voiced_ms": round(voiced_ms, 1),
    }
    if duration_s < float(min_duration_s):
        return False, "too_short", metrics
    if peak_rms < float(min_rms) or voiced_ms < float(min_voiced_ms):
        return False, "silent", metrics
    return True, "ok", metrics


def _write_pcm16_wav(path: Path, audio_f32: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = np.clip(audio_f32 * 32767.0, -32767.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _is_mlx_whisper_installed() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("mlx_whisper") is not None
    except Exception:
        return False


def check_apple_speech_on_device(language: str = "vi-VN") -> Dict[str, Any]:
    """Inspect local macOS on-device Speech availability for `language` without network access."""
    info: Dict[str, Any] = {
        "language": language,
        "available": False,
        "supports_on_device": False,
        "mlx_whisper_available": _is_mlx_whisper_installed(),
    }
    try:
        import Foundation
        import Speech

        locale = Foundation.NSLocale.localeWithLocaleIdentifier_(language)
        recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
        if recognizer is not None:
            info["available"] = bool(recognizer.isAvailable())
            try:
                info["supports_on_device"] = bool(recognizer.supportsOnDeviceRecognition())
            except Exception:
                info["supports_on_device"] = False
    except Exception:
        pass
    return info


def _transcribe_apple_speech(
    wav_path: Path,
    language: str = "vi-VN",
    timeout_s: float = 15.0,
    *,
    require_on_device: bool = True,
) -> str:
    """Transcribe a short WAV clip via macOS native Speech.framework (`SFSpeechRecognizer`, local-only)."""
    import Foundation
    import Speech

    auth_done = threading.Event()
    auth_status = [Speech.SFSpeechRecognizer.authorizationStatus()]

    # 3 == SFSpeechRecognizerAuthorizationStatusAuthorized
    if auth_status[0] == 0:  # NotDetermined
        def _auth_cb(st: int) -> None:
            auth_status[0] = st
            auth_done.set()

        Speech.SFSpeechRecognizer.requestAuthorization_(_auth_cb)
        auth_done.wait(timeout=8.0)

    if auth_status[0] != 3:
        raise RuntimeError(
            "Speech Recognition permission not granted (System Settings -> Privacy & Security -> Speech Recognition)."
        )

    locale = Foundation.NSLocale.localeWithLocaleIdentifier_(language)
    recognizer = Speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
    if recognizer is None or not recognizer.isAvailable():
        raise RuntimeError(f"Apple Speech Recognizer unavailable for locale '{language}'.")

    supports_local = False
    try:
        supports_local = bool(recognizer.supportsOnDeviceRecognition())
    except Exception:
        supports_local = False

    if require_on_device and not supports_local:
        raise RuntimeError(f"On-device Apple Speech is unavailable for locale '{language}'.")

    url = Foundation.NSURL.fileURLWithPath_(str(wav_path))
    req = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
    try:
        if supports_local:
            req.setRequiresOnDeviceRecognition_(True)
    except Exception:
        pass
    try:
        req.setShouldReportPartialResults_(False)
    except Exception:
        pass

    done = threading.Event()
    result_box: Dict[str, Any] = {"text": "", "error": None}

    def _handler(result: Any, error: Any) -> None:
        if error is not None:
            result_box["error"] = str(error.localizedDescription() if hasattr(error, "localizedDescription") else error)
            done.set()
            return
        if result is not None:
            try:
                best = result.bestTranscription()
                if best is not None:
                    result_box["text"] = str(best.formattedString() or "")
            except Exception:
                pass
            if result.isFinal():
                done.set()

    task = recognizer.recognitionTaskWithRequest_resultHandler_(req, _handler)
    finished = done.wait(timeout=timeout_s)
    if not finished:
        try:
            task.cancel()
        except Exception:
            pass
    if result_box["error"] and not result_box["text"]:
        raise RuntimeError(result_box["error"])
    return str(result_box["text"]).strip()


def _transcribe_mlx_whisper(wav_path: Path, language: str = "vi-VN") -> str:
    """Optional local Whisper transcription on Apple Silicon (`mlx_whisper`)."""
    import mlx_whisper  # type: ignore[import-not-found]

    lang_code = language.split("-")[0].lower() if language else "vi"
    out = mlx_whisper.transcribe(str(wav_path), language=lang_code)
    return str((out or {}).get("text", "")).strip()


class VoiceDictationController:
    """Non-blocking hold/toggle microphone capture & local speech-to-text inserter."""

    def __init__(
        self,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        on_text_ready: Optional[Callable[[str], None]] = None,
        on_discard: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self._on_text_ready = on_text_ready
        self._on_discard = on_discard
        self._lock = threading.Lock()
        self._recording: bool = False
        self._transcribing: bool = False
        self._stream: Any = None
        self._chunks: List[np.ndarray] = []
        self._started_wall: float = 0.0
        self._audio_level: float = 0.0
        self._last_transcript: str = ""
        self._last_error: Optional[str] = None
        self._on_device_info: Dict[str, Any] = {}
        self._notice_code: str = "on_device_ok"
        self._active_backend: str = "apple_speech"
        self._events: Deque[Dict[str, Any]] = deque(maxlen=50)
        self._stats: Dict[str, int] = {"started": 0, "committed": 0, "discarded": 0, "cancelled": 0}
        self._current_trigger_meta: Dict[str, Any] = {}
        self._last_discard_reason: Optional[str] = None
        self._last_discard_seq: int = 0
        self._last_discard_wall: float = 0.0
        self.update_config(cfg)

    def update_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        c = cfg or {}
        self.enabled: bool = bool(c.get("enabled", True))
        self.engine: str = str(c.get("engine", "auto")).lower()
        self.language: str = str(c.get("language", "vi-VN"))
        self.max_duration_s: float = max(2.0, float(c.get("max_duration_s", 30.0)))
        self.min_duration_s: float = max(0.1, min(10.0, float(c.get("min_duration_s", 0.6))))
        self.min_rms: float = max(0.0005, min(0.5, float(c.get("min_rms", 0.01))))
        self.min_voiced_ms: float = max(50.0, min(5000.0, float(c.get("min_voiced_ms", 250.0))))
        bl = c.get("hallucination_blocklist")
        if isinstance(bl, dict):
            self.hallucination_blocklist: Dict[str, List[str]] = {
                str(k): [str(x) for x in v] for k, v in bl.items() if isinstance(v, list)
            }
        else:
            self.hallucination_blocklist = {k: list(v) for k, v in DEFAULT_HALLUCINATION_BLOCKLIST.items()}
        self._refresh_on_device_status()

    def _record_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)

    def _refresh_on_device_status(self) -> None:
        info = check_apple_speech_on_device(self.language)
        self._on_device_info = info
        if self.engine == "mlx_whisper":
            self._active_backend = "mlx_whisper"
            self._notice_code = "fallback_mlx_whisper" if info.get("mlx_whisper_available") else "on_device_unavailable"
        elif info.get("supports_on_device"):
            self._active_backend = "apple_speech"
            self._notice_code = "on_device_ok"
        elif info.get("mlx_whisper_available"):
            self._active_backend = "mlx_whisper"
            self._notice_code = "fallback_mlx_whisper"
        else:
            self._active_backend = "none"
            self._notice_code = "on_device_unavailable"

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_transcribing(self) -> bool:
        return self._transcribing

    def recent_events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def status_dict(self) -> Dict[str, Any]:
        discard_active = (
            self._last_discard_reason is not None
            and (time.monotonic() - self._last_discard_wall) < 2.8
        )
        return {
            "enabled": self.enabled,
            "recording": self._recording,
            "transcribing": self._transcribing,
            "audio_level": round(self._audio_level, 3),
            "language": self.language,
            "engine": self.engine,
            "active_backend": self._active_backend,
            "on_device_supported": bool(self._on_device_info.get("supports_on_device", False)),
            "mlx_whisper_available": bool(self._on_device_info.get("mlx_whisper_available", False)),
            "notice_code": self._notice_code,
            "last_transcript": self._last_transcript,
            "last_error": self._last_error,
            "last_discard_reason": self._last_discard_reason,
            "last_discard_seq": self._last_discard_seq,
            "discard_active": discard_active,
            "stats": dict(self._stats),
            "recent_events": self.recent_events(),
        }

    def start(self, trigger_meta: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False
        meta = dict(trigger_meta or {})
        with self._lock:
            if self._recording:
                return True
            self._chunks = []
            self._audio_level = 0.0
            self._last_error = None
            self._current_trigger_meta = meta
            # Initialize wall timestamp BEFORE stream.start() so audio callback never sees stale 0.0
            self._started_wall = time.monotonic()
            try:
                import sounddevice as sd

                def _cb(indata: np.ndarray, _frames: int, _time_info: Any, _status: Any) -> None:
                    mono = indata[:, 0].copy()
                    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
                    self._audio_level = min(1.0, rms * 10.0)
                    self._chunks.append(mono)
                    if (time.monotonic() - self._started_wall) >= self.max_duration_s:
                        threading.Thread(target=self.stop_and_transcribe, daemon=True).start()

                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    callback=_cb,
                )
                stream.start()
                self._stream = stream
                self._recording = True
                self._stats["started"] += 1
                ev = {
                    "ts": round(time.time(), 2),
                    "event": "start",
                    "language": self.language,
                    "source": str(meta.get("source", "gesture")),
                    "hand": meta.get("hand"),
                    "pose": meta.get("pose"),
                    "pose_prob": round(float(meta.get("pose_prob", 0.0)), 3),
                    "dir_z": round(float(meta.get("dir_z", 0.0)), 3),
                    "palm_size": round(float(meta.get("palm_size", 0.0)), 4),
                }
                self._events.append(ev)
                logger.info(
                    f"[DICTATION] start lang={self.language} src={ev['source']} "
                    f"hand={ev['hand']} pose={ev['pose']} prob={ev['pose_prob']:.2f} "
                    f"dir_z={ev['dir_z']:+.2f} palm={ev['palm_size']:.3f}"
                )
                return True
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning(f"[DICTATION] Failed to start microphone stream: {exc}")
                return False

    def toggle(self, trigger_meta: Optional[Dict[str, Any]] = None) -> bool:
        if self._recording:
            self.stop_and_transcribe()
            return False
        meta = dict(trigger_meta or {"source": "toggle"})
        return self.start(trigger_meta=meta)

    def cancel(self, reason: str = "cancelled") -> bool:
        """Stop microphone stream immediately and DISCARD audio (no transcription, no paste)."""
        with self._lock:
            was_recording = self._recording
            self._recording = False
            self._audio_level = 0.0
            stream = self._stream
            self._stream = None
            self._chunks = []
            dur = round(max(0.0, time.monotonic() - self._started_wall), 3) if self._started_wall > 0 else 0.0
            meta = dict(self._current_trigger_meta)
            self._current_trigger_meta = {}
            if was_recording:
                self._stats["cancelled"] += 1
                self._last_discard_reason = reason
                self._last_discard_seq += 1
                self._last_discard_wall = time.monotonic()
                self._events.append(
                    {
                        "ts": round(time.time(), 2),
                        "event": "cancelled",
                        "reason": reason,
                        "duration_s": dur,
                        "source": meta.get("source", "gesture"),
                        "hand": meta.get("hand"),
                    }
                )
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        if was_recording:
            logger.info(f"[DICTATION] cancelled reason={reason} duration_s={dur:.2f}")
        return was_recording

    def _emit_discard(self, reason: str, metrics: Dict[str, Any], meta: Dict[str, Any]) -> None:
        ev = {
            "ts": round(time.time(), 2),
            "event": "discarded",
            "reason": reason,
            "language": self.language,
            "source": meta.get("source", "gesture"),
            "hand": meta.get("hand"),
            **metrics,
        }
        with self._lock:
            self._stats["discarded"] += 1
            self._last_discard_reason = reason
            self._last_discard_seq += 1
            self._last_discard_wall = time.monotonic()
            self._events.append(ev)
        logger.info(
            f"[DICTATION] discarded reason={reason} dur={metrics.get('duration_s', 0.0):.2f}s "
            f"peak_rms={metrics.get('peak_rms', 0.0):.4f} voiced_ms={metrics.get('voiced_ms', 0.0):.0f}"
        )
        if self._on_discard is not None:
            try:
                self._on_discard(reason, ev)
            except Exception:
                pass

    def stop_and_transcribe(self, *, sync: bool = False) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._audio_level = 0.0
            stream = self._stream
            self._stream = None
            chunks = list(self._chunks)
            self._chunks = []
            meta = dict(self._current_trigger_meta)
            self._current_trigger_meta = {}

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        if not chunks:
            self._emit_discard(
                "too_short",
                {"duration_s": 0.0, "overall_rms": 0.0, "peak_rms": 0.0, "voiced_ms": 0.0},
                meta,
            )
            return

        audio = np.concatenate(chunks, axis=0)
        ok, reason, metrics = evaluate_audio_vad(
            audio,
            sample_rate=SAMPLE_RATE,
            min_duration_s=self.min_duration_s,
            min_rms=self.min_rms,
            min_voiced_ms=self.min_voiced_ms,
        )
        if not ok:
            self._emit_discard(reason, metrics, meta)
            return

        if sync:
            self._run_transcription(audio, metrics=metrics, meta=meta)
        else:
            threading.Thread(
                target=self._run_transcription,
                args=(audio,),
                kwargs={"metrics": metrics, "meta": meta},
                daemon=True,
            ).start()

    def _run_transcription(
        self,
        audio: np.ndarray,
        *,
        metrics: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._transcribing = True
        tmp_path: Optional[Path] = None
        m = dict(metrics or {})
        tr_meta = dict(meta or {})
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", prefix="aerohand_dict_", delete=False) as tf:
                tmp_path = Path(tf.name)
            _write_pcm16_wav(tmp_path, audio, SAMPLE_RATE)

            text = ""
            if self.engine == "mlx_whisper":
                text = _transcribe_mlx_whisper(tmp_path, self.language)
                self._active_backend = "mlx_whisper"
                self._notice_code = "fallback_mlx_whisper"
            elif self.engine == "apple_speech":
                text = _transcribe_apple_speech(tmp_path, self.language, require_on_device=True)
                self._active_backend = "apple_speech"
                self._notice_code = "on_device_ok"
            else:
                # auto (strictly 100% local): Apple Speech on-device first -> local mlx_whisper fallback
                try:
                    text = _transcribe_apple_speech(tmp_path, self.language, require_on_device=True)
                    self._active_backend = "apple_speech"
                    self._notice_code = "on_device_ok"
                except Exception as exc_sp:
                    logger.debug(f"[DICTATION] Apple Speech on-device unavailable ({exc_sp}); trying mlx_whisper...")
                    try:
                        text = _transcribe_mlx_whisper(tmp_path, self.language)
                        self._active_backend = "mlx_whisper"
                        self._notice_code = "fallback_mlx_whisper"
                    except Exception:
                        self._notice_code = "on_device_unavailable"
                        raise exc_sp

            if not text or not text.strip():
                self._emit_discard("silent", m, tr_meta)
                return

            if is_hallucinated_transcript(text, self.language, self.hallucination_blocklist):
                self._emit_discard("hallucination", {**m, "blocked_text": text[:64]}, tr_meta)
                return

            self._last_transcript = text
            self._last_error = None
            with self._lock:
                self._stats["committed"] += 1
                self._events.append(
                    {
                        "ts": round(time.time(), 2),
                        "event": "committed",
                        "language": self.language,
                        "backend": self._active_backend,
                        "chars": len(text),
                        "source": tr_meta.get("source", "gesture"),
                        **m,
                    }
                )
            logger.info(
                f"[DICTATION] committed lang={self.language} backend={self._active_backend} "
                f"chars={len(text)} dur={m.get('duration_s', 0.0):.2f}s"
            )
            if self._on_text_ready is not None:
                self._on_text_ready(text)
        except Exception as exc:
            self._last_error = str(exc)
            self._record_event(
                {
                    "ts": round(time.time(), 2),
                    "event": "error",
                    "error": str(exc)[:120],
                    "language": self.language,
                }
            )
            logger.warning(f"[DICTATION] Transcription failed: {exc}")
        finally:
            self._transcribing = False
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def close(self) -> None:
        with self._lock:
            self._recording = False
            stream = self._stream
            self._stream = None
            self._chunks = []
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
