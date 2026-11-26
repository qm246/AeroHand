"""
AeroHand CLI Entrypoint (Airpoint CLI parity + standalone runner).

Desktop control subcommands (talks to running AeroHand.app on 127.0.0.1:17420):
  aerohand status [--json]
  aerohand tracking [on|off] [--json]
  aerohand settings list [--json]
  aerohand settings get <key> [--json]
  aerohand settings set <key> <value> [--json]
  aerohand vitals [--json]
  aerohand see [--json]
  aerohand recenter [--json]
  aerohand open [--json]

Standalone runtime mode:
  aerohand run [--dry-run] [--preview] [--collect-dir PATH] [--camera 0]
  aerohand [--dry-run] [--preview] ...   (backward-compatible flags)
"""

from __future__ import annotations

import argparse
import json
import logging
import select
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from aerohand import __version__
from aerohand.config import load_config
from aerohand.platform.control_api import read_api_token
from aerohand.runtime import AeroHandRuntime

_API_COMMANDS = frozenset(
    {"status", "tracking", "settings", "vitals", "see", "recenter", "open", "run", "passive-mark"}
)


def _parse_cli_value(raw: str) -> Any:
    low = raw.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _api_request(
    path: str,
    *,
    method: str = "GET",
    body: Optional[Dict[str, Any]] = None,
    port: int = 17420,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{port}{path}"
    is_write = method.upper() in ("POST", "PATCH", "PUT")
    if body is not None:
        data: Optional[bytes] = json.dumps(body).encode("utf-8")
    elif is_write:
        data = b"{}"
    else:
        data = None
    headers: Dict[str, str] = {}
    if is_write or data is not None:
        headers["Content-Type"] = "application/json"
    tok = token if token is not None else read_api_token()
    if tok:
        headers["X-AeroHand-Token"] = tok
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError:
        raise SystemExit(
            f"AeroHand.app is not running (cannot reach http://127.0.0.1:{port}). Run `aerohand open` first."
        ) from None


def _run_standalone(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="aerohand run",
        description="Run AeroHand tracking directly in terminal.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config")
    parser.add_argument("--overrides", type=str, default=None, help="Path to user_overrides.yaml")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--dry-run", action="store_true", help="Disable real OS clicks/moves/keys")
    parser.add_argument("--preview", action="store_true", help="Show live OpenCV HUD preview window")
    parser.add_argument(
        "--collect-dir",
        type=str,
        default=None,
        help="Enable interactive PinchDataCollector and save JSON sessions to this directory",
    )
    parser.add_argument(
        "--passive-record",
        type=str,
        default=None,
        help="Enable chunked dry-run passive idle recorder saving schema-v2 chunks to DIR (V060-06)",
    )
    parser.add_argument(
        "--passive-max-daily-mb",
        type=float,
        default=256.0,
        help="Daily disk cap (MB) for --passive-record chunks (default: 256)",
    )
    parser.add_argument("--no-audio", action="store_true", help="Disable click sound effects")
    parser.add_argument("--no-typing-guard", action="store_true", help="Disable Typing Guard (debug only)")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(config_path=args.config, overrides_path=args.overrides)
    if args.no_audio:
        cfg.setdefault("audio", {})["enabled"] = False
    if args.no_typing_guard:
        cfg.setdefault("typing_guard", {})["enabled"] = False

    eff_dry_run = bool(args.dry_run or args.passive_record)
    rt = AeroHandRuntime(
        cfg,
        dry_run=eff_dry_run,
        preview=args.preview,
        camera_index=args.camera,
        collect_dir=args.collect_dir,
        passive_record_dir=args.passive_record,
        passive_max_daily_mb=args.passive_max_daily_mb,
    )

    tty_stop = threading.Event()
    tty_thread: Optional[threading.Thread] = None
    if args.passive_record:
        print(
            "[PASSIVE] Forced DRY-RUN (zero HID output). Press 'm' here (or run `aerohand passive-mark`) "
            "ONLY after a DELIBERATE gesture (+/-1.5s excluded from idle exposure; NEVER for false triggers).",
            flush=True,
        )
        if sys.stdin.isatty():
            def _watch_tty_keys() -> None:
                try:
                    import termios
                    import tty
                except ImportError:
                    return
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not tty_stop.is_set():
                        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                        if ready:
                            ch = sys.stdin.read(1)
                            if ch.lower() == "m":
                                res = rt.mark_deliberate_gesture()
                                if res.get("ok"):
                                    print(
                                        f"[PASSIVE] Marked deliberate gesture #{res['marks_count']} "
                                        f"(t={res['timestamp_ms']}ms, +/-1.5s excluded)",
                                        flush=True,
                                    )
                except Exception:
                    return
                finally:
                    try:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    except Exception:
                        pass

            tty_thread = threading.Thread(target=_watch_tty_keys, name="AeroHandPassiveTTY", daemon=True)
            tty_thread.start()

    try:
        rt.run()
    finally:
        tty_stop.set()
        if tty_thread is not None and tty_thread.is_alive():
            tty_thread.join(timeout=0.5)
        if rt.passive_recorder is not None:
            st = rt.passive_recorder.status()
            print(
                f"[PASSIVE] Summary: chunks={st['chunks_written']} | "
                f"hand_present_idle_hours={st['hand_present_idle_hours']:.4f}h | "
                f"deliberate_marks={st['deliberate_marks_count']} | out={st['out_dir']}",
                flush=True,
            )


def main(argv: Optional[list[str]] = None) -> None:
    args_in = list(sys.argv[1:] if argv is None else argv)

    # Backward-compatible mode: if invoked with no subcommand or legacy flags like `--dry-run`
    if not args_in or (args_in[0].startswith("-") and args_in[0] not in ("-h", "--help", "--version")):
        _run_standalone(args_in)
        return

    if args_in[0] == "run":
        _run_standalone(args_in[1:])
        return

    cfg = load_config()
    port = int((cfg.get("api") or {}).get("port", 17420))

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Output raw JSON")
    common.add_argument(
        "--port", type=int, default=argparse.SUPPRESS, help=f"Local Control API port (default {port})"
    )

    parser = argparse.ArgumentParser(
        prog="aerohand",
        description="AeroHand — Touchless macOS Hand Controller CLI (Airpoint-parity API + Standalone Runner).",
    )
    parser.add_argument("--version", action="version", version=f"AeroHand {__version__}")
    parser.add_argument("--json", action="store_true", default=False, help="Output raw JSON")
    parser.add_argument("--port", type=int, default=port, help=f"Local Control API port (default {port})")

    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("status", parents=[common], help="Show AeroHand status, tracking state, and version")

    p_trk = sub.add_parser("tracking", parents=[common], help="Get or toggle hand tracking (on|off)")
    p_trk.add_argument("state", nargs="?", choices=["on", "off", "true", "false"])

    p_set = sub.add_parser("settings", parents=[common], help="Inspect or update AeroHand settings live")
    set_sub = p_set.add_subparsers(dest="settings_cmd", required=True)
    set_sub.add_parser("list", parents=[common], help="List full active configuration")
    p_get = set_sub.add_parser("get", parents=[common], help="Read one setting by dot-key (e.g. cursor.sensitivity)")
    p_get.add_argument("key")
    p_put = set_sub.add_parser("set", parents=[common], help="Update one setting live (auto-saved)")
    p_put.add_argument("key")
    p_put.add_argument("value")

    sub.add_parser("vitals", parents=[common], help="Show CPU, RAM, and pipeline timing telemetry")
    sub.add_parser("see", parents=[common], help="Capture a screenshot of the current screen")
    sub.add_parser("recenter", parents=[common], help="Recenter cursor to screen center")
    sub.add_parser(
        "passive-mark",
        parents=[common],
        help="Mark a deliberate gesture during --passive-record (+/-1.5s excluded from idle exposure)",
    )
    sub.add_parser("open", parents=[common], help="Open or surface AeroHand.app")
    sub.add_parser("run", help="Run standalone tracker in terminal")

    ns = parser.parse_args(args_in)
    api_port = int(ns.port)

    if ns.cmd == "status":
        data = _api_request("/api/status", port=api_port)
        if ns.json:
            print(json.dumps(data, ensure_ascii=False))
            return
        trk = "ON" if data.get("tracking") else "OFF"
        mode = "DRY-RUN" if data.get("dry_run") else "LIVE"
        print(f"OK AeroHand v{data.get('version', __version__)} ({mode})")
        print(f"Tracking: {trk} | FPS: {data.get('fps', 0):.1f} | Delegate: {data.get('delegate', '?')}")
        print(f"Active App: {data.get('active_app') or 'N/A'} | Pose: {data.get('pose', 'idle')}")
    elif ns.cmd == "tracking":
        if not ns.state:
            data = _api_request("/api/tracking", port=api_port)
            if ns.json:
                print(json.dumps(data))
            else:
                print(f"Tracking: {'ON' if data.get('enabled') else 'OFF'}")
        else:
            enabled = ns.state in ("on", "true")
            data = _api_request("/api/tracking", method="POST", body={"enabled": enabled}, port=api_port)
            if ns.json:
                print(json.dumps(data))
            else:
                print(f"OK Tracking {'enabled' if data.get('enabled') else 'disabled'}")
    elif ns.cmd == "settings":
        if ns.settings_cmd == "list":
            data = _api_request("/api/settings", port=api_port)
            print(json.dumps(data.get("config", {}), indent=2, ensure_ascii=False))
        elif ns.settings_cmd == "get":
            encoded = urllib.parse.quote(ns.key, safe="")
            data = _api_request(f"/api/settings/{encoded}", port=api_port)
            if ns.json:
                print(json.dumps(data, ensure_ascii=False))
            else:
                print(f"{data['key']} = {json.dumps(data['value'], ensure_ascii=False)}")
        elif ns.settings_cmd == "set":
            val = _parse_cli_value(ns.value)
            data = _api_request(
                "/api/settings",
                method="PATCH",
                body={"key": ns.key, "value": val},
                port=api_port,
            )
            if ns.json:
                print(json.dumps(data, ensure_ascii=False))
            else:
                print(f"OK {data['key']} = {json.dumps(data['value'], ensure_ascii=False)}")
    elif ns.cmd == "vitals":
        data = _api_request("/api/system/vitals", port=api_port)
        if ns.json:
            print(json.dumps(data, ensure_ascii=False))
            return
        print(
            f"CPU: {data.get('cpu_usage', 0):.1f}% | "
            f"RAM: {data.get('ram_used_gb', 0):.1f} / {data.get('ram_total_gb', 0):.1f} GB "
            f"(AeroHand: {data.get('process_ram_mb', 0):.0f} MB)"
        )
    elif ns.cmd == "see":
        data = _api_request("/api/screenshot", method="POST", port=api_port)
        if ns.json:
            print(json.dumps(data, ensure_ascii=False))
        else:
            print(f"OK Screenshot saved -> {data.get('path')}")
    elif ns.cmd == "recenter":
        data = _api_request("/api/recenter", method="POST", port=api_port)
        print(json.dumps(data) if ns.json else "OK Cursor recentered")
    elif ns.cmd == "passive-mark":
        data = _api_request("/api/passive/mark", method="POST", port=api_port)
        if ns.json:
            print(json.dumps(data, ensure_ascii=False))
        else:
            print(
                f"OK Marked deliberate gesture #{data.get('marks_count', 0)} "
                f"(t={data.get('timestamp_ms', 0)}ms, +/-1.5s excluded)"
            )
    elif ns.cmd == "open":
        desktop_app = Path.home() / "Desktop" / "AeroHand.app"
        apps_app = Path("/Applications/AeroHand.app")
        target = str(apps_app if apps_app.exists() else desktop_app)
        if Path(target).exists():
            subprocess.Popen(["open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["aerohand-app"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(json.dumps({"ok": True, "target": target}) if ns.json else f"OK Launched {target}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
