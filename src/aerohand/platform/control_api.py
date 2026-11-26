"""
Local HTTP Control API (`127.0.0.1:17420`), URL Scheme (`aerohand://`), and Launch-at-Login helper.
Provides full Airpoint CLI parity (`status`, `tracking`, `settings`, `vitals`, `see`, `recenter`, `open`)
while staying strictly localhost-bound and stdlib-only.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import plistlib
import secrets
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

import yaml

from aerohand import __version__
from aerohand.config import (
    default_config_path,
    get_nested_key,
    resolve_setting_key,
    save_user_overrides,
    set_nested_key,
)
from aerohand.core.types import ALL_POSE_NAMES

logger = logging.getLogger(__name__)

LAUNCH_AGENT_LABEL = "com.qm246.aerohand"
MAX_API_BODY_BYTES = 65536  # 64 KiB hard limit
_VALID_POSES = frozenset(p.value for p in ALL_POSE_NAMES if p.value != "idle")
_VALID_HANDS = frozenset({"Left", "Right"})
_VALID_FINGERS = frozenset({"thumb", "index", "middle", "ring", "pinky"})
_BLOCKED_KEY_PREFIXES = ("api.", "profiles.")
_BLOCKED_KEY_SUFFIXES = (".model_path", ".onnx_model_path", ".onnx_normalizer_path", "_sound")
_CACHED_DEFAULTS: Optional[Dict[str, Any]] = None


def app_support_dir() -> Path:
    return Path.home() / "Library" / "Application Support" / "AeroHand"


def api_token_path() -> Path:
    return app_support_dir() / "api_token"


def read_api_token(token_path: Optional[Path] = None) -> str:
    fp = Path(token_path) if token_path is not None else api_token_path()
    try:
        return fp.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _write_token_secure(fp: Path, token: str) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(fp.parent, 0o700)
    except OSError:
        pass
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(fp), flags, 0o600)
    try:
        os.write(fd, (token + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    try:
        os.chmod(fp, 0o600)
    except OSError:
        pass


def _load_default_schema() -> Dict[str, Any]:
    global _CACHED_DEFAULTS
    if _CACHED_DEFAULTS is None:
        with open(default_config_path(), "r", encoding="utf-8") as f:
            _CACHED_DEFAULTS = yaml.safe_load(f) or {}
    return _CACHED_DEFAULTS


def is_writable_setting_key(
    key: str,
    value: Any,
    defaults: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Validate that `key` is an allow-listed leaf setting and `value` is a safe scalar."""
    resolved = resolve_setting_key(key)
    if not resolved or ".." in resolved or resolved.startswith(".") or resolved.endswith("."):
        return False, f"Invalid setting key: {key}"
    if any(resolved.startswith(p) for p in _BLOCKED_KEY_PREFIXES) or any(
        resolved.endswith(s) for s in _BLOCKED_KEY_SUFFIXES
    ):
        return False, f"Setting key is not writable via HTTP API: {resolved}"

    # Reject non-scalar values (dicts/lists) to prevent structural config injection
    if not isinstance(value, (bool, int, float, str)):
        return False, f"Setting value for '{resolved}' must be a scalar (bool, int, float, or str)."

    # Dynamic gesture/palm-wheel binding maps
    parts = resolved.split(".")
    if (
        len(parts) == 3
        and parts[0] == "pose_state_machine"
        and parts[1] in ("tap_actions", "hold_actions")
        and parts[2] in _VALID_POSES
    ):
        if not isinstance(value, str):
            return False, f"Action binding for '{resolved}' must be a string."
        return True, resolved
    if (
        len(parts) == 4
        and parts[0] == "palm_wheel"
        and parts[1] == "bindings"
        and parts[2] in _VALID_HANDS
        and parts[3] in _VALID_FINGERS
    ):
        if not isinstance(value, str):
            return False, f"Palm wheel binding for '{resolved}' must be a string."
        return True, resolved

    schema = defaults if defaults is not None else _load_default_schema()
    found, default_val = get_nested_key(schema, resolved)
    if not found:
        return False, f"Unknown setting key: {resolved}"
    if isinstance(default_val, (dict, list)):
        return False, f"Setting key '{resolved}' refers to a composite object, not a scalar leaf."
    return True, resolved


def launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"


def is_launch_at_login_enabled() -> bool:
    return launch_agent_path().is_file()


def set_launch_at_login(enabled: bool) -> bool:
    """Create or remove a macOS LaunchAgent plist for automatic login startup."""
    plist_fp = launch_agent_path()
    if not enabled:
        try:
            plist_fp.unlink(missing_ok=True)
            return True
        except OSError as exc:
            logger.warning(f"[LAUNCH] Unable to remove LaunchAgent: {exc}")
            return False

    desktop_bin = Path.home() / "Desktop" / "AeroHand.app" / "Contents" / "MacOS" / "AeroHand"
    apps_bin = Path("/Applications/AeroHand.app/Contents/MacOS/AeroHand")
    if apps_bin.is_file():
        prog = [str(apps_bin)]
    elif desktop_bin.is_file():
        prog = [str(desktop_bin)]
    else:
        prog = ["/usr/bin/env", "aerohand-app"]

    payload = {
        "Label": LAUNCH_AGENT_LABEL,
        "ProgramArguments": prog,
        "RunAtLoad": True,
        "KeepAlive": False,
    }
    try:
        plist_fp.parent.mkdir(parents=True, exist_ok=True)
        with open(plist_fp, "wb") as f:
            plistlib.dump(payload, f)
        return True
    except Exception as exc:
        logger.warning(f"[LAUNCH] Unable to write LaunchAgent: {exc}")
        return False


def capture_screenshot_png() -> Dict[str, Any]:
    """Capture the current macOS display into Application Support screenshots folder."""
    shots_dir = Path.home() / "Library" / "Application Support" / "AeroHand" / "screenshots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"screen_{ts}.png"
    out_path = shots_dir / fname
    subprocess.run(["screencapture", "-x", str(out_path)], check=True, timeout=10)
    return {
        "ok": True,
        "filename": fname,
        "path": str(out_path),
    }


def handle_aerohand_url(
    url_str: str,
    *,
    on_show: Optional[Callable[[], None]] = None,
    on_tracking: Optional[Callable[[bool], None]] = None,
    on_toggle_tracking: Optional[Callable[[], None]] = None,
    on_recenter: Optional[Callable[[], None]] = None,
    on_toggle_hud: Optional[Callable[[], None]] = None,
    on_toggle_dictation: Optional[Callable[[], None]] = None,
) -> bool:
    """Dispatch `aerohand://<action>` URL scheme links (Apple Shortcuts / Raycast / Alfred)."""
    parsed = urlparse(url_str)
    if parsed.scheme.lower() != "aerohand":
        return False
    host = (parsed.netloc or "").lower()
    sub = parsed.path.strip("/").lower()
    if host in ("open", "show", "settings", ""):
        if on_show:
            on_show()
        return True
    if host == "tracking":
        if sub in ("on", "start", "resume", "true"):
            if on_tracking:
                on_tracking(True)
            return True
        if sub in ("off", "stop", "pause", "false"):
            if on_tracking:
                on_tracking(False)
            return True
        if on_toggle_tracking:
            on_toggle_tracking()
        return True
    if host == "recenter":
        if on_recenter:
            on_recenter()
        return True
    if host == "hud":
        if on_toggle_hud:
            on_toggle_hud()
        return True
    if host == "dictation":
        if on_toggle_dictation:
            on_toggle_dictation()
        return True
    return False


class _PayloadTooLargeError(Exception):
    pass


class LocalControlAPIServer:
    """Thread-safe hardened localhost HTTP API powering `aerohand` CLI and Shortcuts automation."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 17420,
        get_status: Callable[[], Dict[str, Any]],
        get_vitals: Callable[[], Dict[str, Any]],
        get_config: Callable[[], Dict[str, Any]],
        apply_config: Callable[[Dict[str, Any]], None],
        set_tracking_enabled: Callable[[bool], bool],
        trigger_recenter: Callable[[], None],
        set_dry_run: Optional[Callable[[bool], None]] = None,
        show_window: Optional[Callable[[], None]] = None,
        mark_deliberate_gesture: Optional[Callable[[], Dict[str, Any]]] = None,
        token: Optional[str] = None,
        token_path: Optional[Path] = None,
    ) -> None:
        if host != "127.0.0.1":
            logger.warning(f"[API] Refusing non-loopback bind host {host!r}; forcing 127.0.0.1.")
        self.host = "127.0.0.1"
        self.port = int(port)
        self._bound_port: int = self.port
        self._get_status = get_status
        self._get_vitals = get_vitals
        self._get_config = get_config
        self._apply_config = apply_config
        self._set_tracking = set_tracking_enabled
        self._recenter = trigger_recenter
        self._set_dry_run = set_dry_run
        self._show_window = show_window
        self._mark_deliberate_gesture = mark_deliberate_gesture
        self._token: str = token if token is not None else secrets.token_hex(32)
        self._token_path: Path = Path(token_path) if token_path is not None else api_token_path()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def token(self) -> str:
        return self._token

    @property
    def bound_port(self) -> int:
        return self._bound_port

    def start(self) -> bool:
        if self._server is not None:
            return True
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _validate_security(self, *, require_json_ct: bool = False) -> bool:
                # 1. DNS-rebinding defense: Host must match exact loopback address & bound port
                host_hdr = (self.headers.get("Host") or "").strip().lower()
                allowed_hosts = {
                    f"127.0.0.1:{parent._bound_port}",
                    f"localhost:{parent._bound_port}",
                }
                if host_hdr not in allowed_hosts:
                    self._send_json(403, {"error": "Forbidden: invalid Host header."})
                    return False

                # 2. CSRF defense: reject any request carrying a browser Origin header
                if self.headers.get("Origin") is not None:
                    self._send_json(403, {"error": "Forbidden: Origin header is not allowed."})
                    return False

                # 3. Local bearer token authentication (0600 file in Application Support)
                supplied = (self.headers.get("X-AeroHand-Token") or "").strip()
                if not supplied or not hmac.compare_digest(supplied, parent._token):
                    self._send_json(401, {"error": "Unauthorized: missing or invalid X-AeroHand-Token."})
                    return False

                # 4. Require application/json Content-Type on state-changing writes
                if require_json_ct:
                    ct = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
                    if ct != "application/json":
                        self._send_json(415, {"error": "Unsupported Media Type: expected application/json."})
                        return False

                return True

            def _read_json(self) -> Dict[str, Any]:
                raw_len = self.headers.get("Content-Length", "0")
                try:
                    length = int(raw_len)
                except ValueError:
                    raise ValueError("Invalid Content-Length header.") from None
                if length < 0:
                    raise ValueError("Negative Content-Length header.")
                if length > MAX_API_BODY_BYTES:
                    raise _PayloadTooLargeError(f"Request body exceeds {MAX_API_BODY_BYTES} bytes.")
                if length == 0:
                    return {}
                raw = self.rfile.read(length)
                if len(raw) > MAX_API_BODY_BYTES:
                    raise _PayloadTooLargeError(f"Request body exceeds {MAX_API_BODY_BYTES} bytes.")
                data = json.loads(raw.decode("utf-8"))
                return data if isinstance(data, dict) else {}

            def do_GET(self) -> None:  # noqa: N802
                if not self._validate_security(require_json_ct=False):
                    return
                path = urlparse(self.path).path.rstrip("/")
                try:
                    if path == "/api/status":
                        st = parent._get_status()
                        st.setdefault("version", __version__)
                        self._send_json(200, st)
                    elif path == "/api/tracking":
                        st = parent._get_status()
                        self._send_json(200, {"enabled": bool(st.get("tracking", False))})
                    elif path == "/api/settings":
                        self._send_json(200, {"config": parent._get_config()})
                    elif path.startswith("/api/settings/"):
                        key = unquote(path[len("/api/settings/") :])
                        found, val = get_nested_key(parent._get_config(), key)
                        if not found:
                            self._send_json(404, {"error": f"Unknown setting key: {key}"})
                        else:
                            self._send_json(200, {"key": key, "value": val})
                    elif path == "/api/system/vitals":
                        self._send_json(200, parent._get_vitals())
                    else:
                        self._send_json(404, {"error": f"Not found: {path}"})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})

            def do_POST(self) -> None:  # noqa: N802
                if not self._validate_security(require_json_ct=True):
                    return
                path = urlparse(self.path).path.rstrip("/")
                try:
                    body = self._read_json()
                    if path == "/api/tracking":
                        enabled = bool(body.get("enabled", True))
                        res = parent._set_tracking(enabled)
                        self._send_json(200, {"enabled": res})
                    elif path == "/api/recenter":
                        parent._recenter()
                        self._send_json(200, {"ok": True})
                    elif path == "/api/mode":
                        if parent._set_dry_run is not None:
                            parent._set_dry_run(bool(body.get("dry_run", False)))
                        self._send_json(200, {"ok": True})
                    elif path == "/api/show":
                        if parent._show_window is not None:
                            parent._show_window()
                        self._send_json(200, {"ok": True})
                    elif path == "/api/screenshot":
                        self._send_json(200, capture_screenshot_png())
                    elif path == "/api/passive/mark":
                        if parent._mark_deliberate_gesture is None:
                            self._send_json(409, {"ok": False, "error": "Passive recording is not active."})
                        else:
                            res = parent._mark_deliberate_gesture()
                            status_code = 200 if res.get("ok") else 409
                            self._send_json(status_code, res)
                    else:
                        self._send_json(404, {"error": f"Not found: {path}"})
                except _PayloadTooLargeError as exc:
                    self._send_json(413, {"error": str(exc)})
                except (ValueError, json.JSONDecodeError) as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})

            def do_PATCH(self) -> None:  # noqa: N802
                if not self._validate_security(require_json_ct=True):
                    return
                path = urlparse(self.path).path.rstrip("/")
                try:
                    if path != "/api/settings":
                        self._send_json(404, {"error": f"Not found: {path}"})
                        return
                    body = self._read_json()
                    key = str(body.get("key", "")).strip()
                    if not key or "value" not in body:
                        self._send_json(400, {"error": "Expected JSON body with 'key' and 'value'."})
                        return
                    ok_key, msg_or_resolved = is_writable_setting_key(key, body["value"])
                    if not ok_key:
                        self._send_json(400, {"error": msg_or_resolved})
                        return
                    cfg = parent._get_config()
                    resolved = set_nested_key(cfg, msg_or_resolved, body["value"])
                    parent._apply_config(cfg)
                    save_user_overrides(cfg)
                    self._send_json(200, {"ok": True, "key": resolved, "value": body["value"]})
                except _PayloadTooLargeError as exc:
                    self._send_json(413, {"error": str(exc)})
                except (ValueError, json.JSONDecodeError) as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
            self._server.daemon_threads = True
            self._bound_port = int(self._server.server_address[1])
            _write_token_secure(self._token_path, self._token)
            self._thread = threading.Thread(target=self._server.serve_forever, name="AeroHandAPI", daemon=True)
            self._thread.start()
            logger.info(f"[API] Local Control API listening on http://{self.host}:{self._bound_port}")
            return True
        except OSError as exc:
            logger.warning(f"[API] Could not bind http://{self.host}:{self.port}: {exc}")
            self._server = None
            return False

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        self._thread = None
