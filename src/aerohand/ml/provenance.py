"""
Shared Cryptographic Provenance Helpers for AeroHand ML & Heuristic Classifiers (V052-02).

Provides a single source of truth shared across training (`scripts/train_tcn.py`),
offline evaluation (`scripts/replay_eval.py`), promotion checking
(`scripts/check_onnx_promotion.py`), and live runtime telemetry (`ONNXPinchClassifier`,
`GuardedClassifier`, `AeroHandRuntime`):
  - `compute_onnx_model_hash(model_path, normalizer_path)` -> deterministic 24-hex fingerprint
  - `compute_heuristic_source_sha256(...)` -> deterministic 64-hex SHA-256 of
    `heuristic_3d.py` + `palm_frame.py` source bytes
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from pathlib import Path
from typing import Optional


def compute_onnx_model_hash(model_path: str | Path, normalizer_path: str | Path) -> str:
    """
    Compute the canonical 24-hex SHA-256 fingerprint for an ONNX model + normalizer pair.
    Matches `train_tcn.compute_onnx_model_hash` and `replay_eval.compute_eval_model_hash('onnx_tcn', ...)`.
    """
    h = hashlib.sha256()
    h.update(b"onnx_tcn")
    h.update(Path(model_path).read_bytes())
    h.update(Path(normalizer_path).read_bytes())
    return h.hexdigest()[:24]


def _read_module_source_bytes(obj: object) -> bytes:
    src_file = inspect.getsourcefile(obj)  # type: ignore[arg-type]
    if src_file and Path(src_file).is_file():
        return Path(src_file).read_bytes()
    return b""


@functools.lru_cache(maxsize=1)
def _default_heuristic_source_sha256() -> str:
    from aerohand.core import palm_frame
    from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier

    h = hashlib.sha256()
    h.update(_read_module_source_bytes(Heuristic3DPinchClassifier))
    h.update(_read_module_source_bytes(palm_frame))
    return h.hexdigest()


def compute_heuristic_source_sha256(
    *,
    heuristic_source_bytes: Optional[bytes] = None,
    palm_frame_source_bytes: Optional[bytes] = None,
) -> str:
    """Return 64-hex SHA-256 of `heuristic_3d.py` + `palm_frame.py` source bytes (cached when default)."""
    if heuristic_source_bytes is None and palm_frame_source_bytes is None:
        return _default_heuristic_source_sha256()

    from aerohand.core import palm_frame
    from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier

    h = hashlib.sha256()
    h.update(
        heuristic_source_bytes
        if heuristic_source_bytes is not None
        else _read_module_source_bytes(Heuristic3DPinchClassifier)
    )
    h.update(
        palm_frame_source_bytes
        if palm_frame_source_bytes is not None
        else _read_module_source_bytes(palm_frame)
    )
    return h.hexdigest()
