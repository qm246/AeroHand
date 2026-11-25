"""
Local HTTP Control API (`127.0.0.1:17420`), URL Scheme (`aerohand://`), and Launch-at-Login helper.
Provides full Airpoint CLI parity (`status`, `tracking`, `settings`, `vitals`, `see`, `recenter`, `open`)
while staying strictly localhost-bound and stdlib-only.
"""

from __future__ import annotations

import json
import logging
import plistlib
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import unquote, urlparse

from aerohand import __version__
from aerohand.config import get_nested_key, save_user_overrides, set_nested_key

logger = logging.getLogger(__name__)

LAUNCH_AGENT_LABEL = "com.qm246.aerohand"


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


class LocalControlAPIServer:
    """Thread-safe localhost HTTP API powering `aerohand` CLI and Shortcuts automation."""

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
    ) -> None:
        self.host = host
        self.port = int(port)
        self._get_status = get_status
        self._get_vitals = get_vitals
        self._get_config = get_config
        self._apply_config = apply_config
        self._set_tracking = set_tracking_enabled
        self._recenter = trigger_recenter
        self._set_dry_run = set_dry_run
        self._show_window = show_window
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

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
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> Dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length <= 0:
                    return {}
                raw = self.rfile.read(min(length, 262144))
                data = json.loads(raw.decode("utf-8"))
                return data if isinstance(data, dict) else {}

            def do_GET(self) -> None:  # noqa: N802
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
                    else:
                        self._send_json(404, {"error": f"Not found: {path}"})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})

            def do_PATCH(self) -> None:  # noqa: N802
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
                    cfg = parent._get_config()
                    resolved = set_nested_key(cfg, key, body["value"])
                    parent._apply_config(cfg)
                    save_user_overrides(cfg)
                    self._send_json(200, {"ok": True, "key": resolved, "value": body["value"]})
                except Exception as exc:
                    self._send_json(500, {"error": str(exc)})

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
            self._server.daemon_threads = True
            self._thread = threading.Thread(target=self._server.serve_forever, name="AeroHandAPI", daemon=True)
            self._thread.start()
            logger.info(f"[API] Local Control API listening on http://{self.host}:{self.port}")
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
