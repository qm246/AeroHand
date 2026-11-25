#!/usr/bin/env python3
"""
Read-only offline dev benchmark for AeroHand on GestureCtl's Day-1 `D-dev` daily negatives.

V041-02: Delegates directly to the unified `scripts/replay_eval.py` harness and shared
`aerohand.ml.recording_io.assert_not_heldout` guard (eliminating duplicate engine loops),
while preserving CLI compatibility.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

from aerohand.ml.recording_io import assert_not_heldout, collect_json_paths


def _load_replay_eval():
    fp = Path(__file__).resolve().parent / "replay_eval.py"
    spec = importlib.util.spec_from_file_location("aerohand_replay_eval_delegate", fp)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aerohand_replay_eval_delegate"] = mod
    spec.loader.exec_module(mod)
    return mod


def benchmark_dir(dev_dir: Path) -> Dict[str, Any]:
    assert_not_heldout(dev_dir)
    files = collect_json_paths([dev_dir])
    if not files:
        raise FileNotFoundError(f"No recording JSON files found in {dev_dir}")
    replay_mod = _load_replay_eval()
    return replay_mod.evaluate_recordings(
        negative_paths=files,
        max_ft_per_hour=2.0,
        ci_level=0.95,
        min_exposure_hours_for_claim=1.5,
    )


def main() -> None:
    default_dev_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "gesturectl_recordings"
        / "m01_v2"
        / "s2b_dev_d_set"
    )
    parser = argparse.ArgumentParser(description="AeroHand read-only dev replay benchmark (delegates to replay_eval)")
    parser.add_argument(
        "dev_dir",
        nargs="?",
        default=str(default_dev_dir),
        help="Path to dev recordings directory (never heldout)",
    )
    args = parser.parse_args()
    res = benchmark_dir(Path(args.dev_dir))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
