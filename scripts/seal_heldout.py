#!/usr/bin/env python3
"""
Sealed AeroHand Held-Out Dataset Manager (V050-02).

Enforces single-touch held-out evaluation discipline for AeroHand's own sealed test set
(`data/heldout_aerohand/`):
  - Copies (or moves with `--move`) chosen session JSON files into a `heldout_aerohand` folder.
  - Writes `sealed_manifest.json` containing per-file SHA-256 checksums, creation timestamp,
    touch counter (`touched: 0`), and `evaluated_model_hashes: []`.
  - Refuses any GestureCtl `heldout` source path (`assert_not_heldout`).
  - `verify_sealed_heldout_manifest()` validates integrity of every sealed file before evaluation.
  - `record_heldout_touch()` refuses evaluating the same `model_hash` twice on the sealed set
    and increments `touched` atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from aerohand.ml.recording_io import assert_not_heldout, collect_json_paths

MANIFEST_NAME = "sealed_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def seal_heldout_sessions(
    src_inputs: Sequence[str | Path],
    out_dir: str | Path = "data/heldout_aerohand",
    *,
    move: bool = False,
    force: bool = False,
    note: str = "",
) -> Dict[str, Any]:
    """
    Seal one or more AeroHand session JSON files into `out_dir` (whose path must include
    `heldout_aerohand` and never `gesturectl`) with a SHA-256 `sealed_manifest.json`.
    """
    dest = Path(out_dir)
    assert_not_heldout(dest, allow_sealed_aerohand=True)
    if "heldout_aerohand" not in str(dest.resolve()).lower():
        raise ValueError(
            f"Sealed AeroHand held-out directory name must include 'heldout_aerohand': {dest}"
        )

    manifest_fp = dest / MANIFEST_NAME
    if manifest_fp.exists() and not force:
        raise ValueError(
            f"REFUSED: Sealed manifest already exists at {manifest_fp}. "
            "Do not re-seal or modify a sealed held-out set (pass --force only for brand-new sets)."
        )

    src_files = collect_json_paths(src_inputs, allow_sealed_aerohand=False)
    if not src_files:
        raise ValueError("No valid JSON session files found to seal.")

    dest.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    seen_names: set[str] = set()

    for idx, src_fp in enumerate(src_files):
        assert_not_heldout(src_fp, allow_sealed_aerohand=False)
        target_name = src_fp.name
        if target_name in seen_names:
            target_name = f"{src_fp.stem}_{idx:02d}{src_fp.suffix}"
        seen_names.add(target_name)
        dst_fp = dest / target_name
        if move:
            shutil.move(str(src_fp), str(dst_fp))
        else:
            shutil.copy2(str(src_fp), str(dst_fp))
        entries.append(
            {
                "filename": target_name,
                "source_name": src_fp.name,
                "sha256": sha256_file(dst_fp),
                "bytes": int(dst_fp.stat().st_size),
            }
        )

    manifest: Dict[str, Any] = {
        "schema": "aerohand_sealed_heldout_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": str(note),
        "touched": 0,
        "evaluated_model_hashes": [],
        "touch_history": [],
        "files": entries,
    }
    manifest_fp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def verify_sealed_heldout_manifest(heldout_dir: str | Path) -> Tuple[Dict[str, Any], List[Path]]:
    """Verify `sealed_manifest.json` exists and every sealed file matches its recorded SHA-256."""
    root = Path(heldout_dir)
    assert_not_heldout(root, allow_sealed_aerohand=True)
    manifest_fp = root / MANIFEST_NAME
    if not manifest_fp.is_file():
        raise FileNotFoundError(f"Missing sealed manifest: {manifest_fp}")

    manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Sealed manifest lists no files: {manifest_fp}")

    verified_paths: List[Path] = []
    for entry in files:
        fname = str(entry.get("filename") or "")
        expected_sha = str(entry.get("sha256") or "")
        fp = root / fname
        if not fp.is_file():
            raise ValueError(f"REFUSED: Sealed file missing: {fp}")
        actual_sha = sha256_file(fp)
        if actual_sha != expected_sha:
            raise ValueError(
                f"REFUSED: Sealed file checksum mismatch for {fname} "
                f"(expected {expected_sha[:12]}..., got {actual_sha[:12]}...)"
            )
        verified_paths.append(fp)
    return manifest, verified_paths


def check_model_not_evaluated(manifest: Dict[str, Any], model_hash: str) -> None:
    """Refuse re-evaluating the exact same model hash on the sealed held-out set."""
    evaluated = [str(x) for x in (manifest.get("evaluated_model_hashes") or [])]
    if str(model_hash) in evaluated:
        raise ValueError(
            f"REFUSED: Model hash '{model_hash}' was already evaluated on this sealed held-out set "
            "(single-touch discipline forbids repeated evaluations of the same model)."
        )


def record_heldout_touch(
    heldout_dir: str | Path,
    model_hash: str,
    *,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Increment `touched` and append `model_hash` into `sealed_manifest.json`."""
    root = Path(heldout_dir)
    manifest_fp = root / MANIFEST_NAME
    manifest, _ = verify_sealed_heldout_manifest(root)
    check_model_not_evaluated(manifest, model_hash)

    manifest["touched"] = int(manifest.get("touched", 0)) + 1
    evaluated = list(manifest.get("evaluated_model_hashes") or [])
    evaluated.append(str(model_hash))
    manifest["evaluated_model_hashes"] = evaluated

    hist = list(manifest.get("touch_history") or [])
    hist.append(
        {
            "touch_index": manifest["touched"],
            "model_hash": str(model_hash),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": summary or {},
        }
    )
    manifest["touch_history"] = hist

    tmp_fp = manifest_fp.with_suffix(".json.tmp")
    tmp_fp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_fp.replace(manifest_fp)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Seal AeroHand held-out sessions with SHA-256 manifest & single-touch counter (V050-02)."
    )
    parser.add_argument("--src", nargs="+", required=True, help="Session JSON files or directories to seal")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/heldout_aerohand",
        help="Destination sealed directory (must contain 'heldout_aerohand')",
    )
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing sealed_manifest.json")
    parser.add_argument("--note", type=str, default="", help="Optional description note stored in manifest")
    args = parser.parse_args(argv)

    manifest = seal_heldout_sessions(
        args.src,
        out_dir=args.out_dir,
        move=args.move,
        force=args.force,
        note=args.note,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
