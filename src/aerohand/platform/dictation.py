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
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _write_pcm16_wav(path: Path, audio_f32: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = np.clip(audio_f32 * 32767.0, -32767.0, 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _transcribe_apple_speech(wav_path: Path, language: str = "vi-VN", timeout_s: float = 15.0) -> str:
    """Transcribe a short WAV clip via macOS native Speech.framework (`SFSpeechRecognizer`)."""
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

    url = Foundation.NSURL.fileURLWithPath_(str(wav_path))
    req = Speech.SFSpeechURLRecognitionRequest.alloc().initWithURL_(url)
    try:
        if recognizer.supportsOnDeviceRecognition():
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
    ) -> None:
        self._on_text_ready = on_text_ready
        self._lock = threading.Lock()
        self._recording: bool = False
        self._transcribing: bool = False
        self._stream: Any = None
        self._chunks: List[np.ndarray] = []
        self._started_wall: float = 0.0
        self._audio_level: float = 0.0
        self._last_transcript: str = ""
        self._last_error: Optional[str] = None
        self.update_config(cfg)

    def update_config(self, cfg: Optional[Dict[str, Any]]) -> None:
        c = cfg or {}
        self.enabled: bool = bool(c.get("enabled", True))
        self.engine: str = str(c.get("engine", "auto")).lower()
        self.language: str = str(c.get("language", "vi-VN"))
        self.max_duration_s: float = float(c.get("max_duration_s", 30.0))

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_transcribing(self) -> bool:
        return self._transcribing

    def status_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "recording": self._recording,
            "transcribing": self._transcribing,
            "audio_level": round(self._audio_level, 3),
            "language": self.language,
            "engine": self.engine,
            "last_transcript": self._last_transcript,
            "last_error": self._last_error,
        }

    def start(self) -> bool:
        if not self.enabled:
            return False
        with self._lock:
            if self._recording:
                return True
            self._chunks = []
            self._audio_level = 0.0
            self._last_error = None
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
                self._started_wall = time.monotonic()
                logger.info(f"[DICTATION] Recording started ({self.language})")
                return True
            except Exception as exc:
                self._last_error = str(exc)
                logger.warning(f"[DICTATION] Failed to start microphone stream: {exc}")
                return False

    def toggle(self) -> bool:
        if self._recording:
            self.stop_and_transcribe()
            return False
        return self.start()

    def stop_and_transcribe(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._audio_level = 0.0
            stream = self._stream
            self._stream = None
            chunks = list(self._chunks)
            self._chunks = []

        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        if not chunks:
            return
        audio = np.concatenate(chunks, axis=0)
        # Skip ultra-short or silent accidental pinches (< 0.35s or below speech noise floor)
        if audio.size < int(SAMPLE_RATE * 0.35):
            return
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio * audio)))
        if peak < 0.015 or rms < 0.003:
            logger.debug(f"[DICTATION] Skipped silent clip (peak={peak:.4f}, rms={rms:.4f})")
            return

        threading.Thread(target=self._run_transcription, args=(audio,), daemon=True).start()

    def _run_transcription(self, audio: np.ndarray) -> None:
        self._transcribing = True
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", prefix="aerohand_dict_", delete=False) as tf:
                tmp_path = Path(tf.name)
            _write_pcm16_wav(tmp_path, audio, SAMPLE_RATE)

            text = ""
            if self.engine == "mlx_whisper":
                text = _transcribe_mlx_whisper(tmp_path, self.language)
            elif self.engine == "apple_speech":
                text = _transcribe_apple_speech(tmp_path, self.language)
            else:
                # auto: Apple Speech on-device first -> mlx_whisper fallback
                try:
                    text = _transcribe_apple_speech(tmp_path, self.language)
                except Exception as exc_sp:
                    logger.debug(f"[DICTATION] Apple Speech fallback ({exc_sp}); trying mlx_whisper...")
                    try:
                        text = _transcribe_mlx_whisper(tmp_path, self.language)
                    except Exception:
                        raise exc_sp

            if text:
                self._last_transcript = text
                self._last_error = None
                logger.info(f"[DICTATION] Transcribed ({self.language}): {text!r}")
                if self._on_text_ready is not None:
                    self._on_text_ready(text)
        except Exception as exc:
            self._last_error = str(exc)
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
