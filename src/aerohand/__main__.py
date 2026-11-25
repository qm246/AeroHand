"""AeroHand CLI entrypoint."""

from __future__ import annotations

import argparse
import logging

from aerohand import __version__
from aerohand.config import load_config
from aerohand.runtime import AeroHandRuntime


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aerohand",
        description="AeroHand — Low-latency touchless macOS hand controller.",
    )
    parser.add_argument("--version", action="version", version=f"AeroHand {__version__}")
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
    parser.add_argument("--no-audio", action="store_true", help="Disable click sound effects")
    parser.add_argument("--no-typing-guard", action="store_true", help="Disable Typing Guard (debug only)")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(config_path=args.config, overrides_path=args.overrides)
    if args.no_audio:
        cfg.setdefault("audio", {})["enabled"] = False
    if args.no_typing_guard:
        cfg.setdefault("typing_guard", {})["enabled"] = False

    rt = AeroHandRuntime(
        cfg,
        dry_run=args.dry_run,
        preview=args.preview,
        camera_index=args.camera,
        collect_dir=args.collect_dir,
    )
    rt.run()


if __name__ == "__main__":
    main()
