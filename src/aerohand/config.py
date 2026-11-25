"""
Override-only configuration loader and deep-merge utilities.
Adapted from Airpoint SDK's `config-core.ts` design (Apache-2.0):
  - `config/default.yaml` holds canonical defaults.
  - Optional `config/user_overrides.yaml` (or CLI `--config`) stores ONLY user deltas.
  - Defaults can evolve without breaking user customizations.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_config_path() -> Path:
    return repo_root() / "config" / "default.yaml"


def user_overrides_path() -> Path:
    return repo_root() / "config" / "user_overrides.yaml"


def deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge `override` onto a deep copy of `base`."""
    result = deepcopy(base)
    if not override:
        return result
    for key, val in override.items():
        if isinstance(result.get(key), dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        elif val is not None:
            result[key] = deepcopy(val)
    return result


def extract_overrides(
    current: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract only keys whose values differ from `defaults`."""
    out: Dict[str, Any] = {}
    for key, val in current.items():
        if key not in defaults:
            out[key] = deepcopy(val)
            continue
        base_val = defaults[key]
        if isinstance(base_val, dict) and isinstance(val, dict):
            nested = extract_overrides(val, base_val)
            if nested:
                out[key] = nested
        elif val != base_val:
            out[key] = deepcopy(val)
    return out


def load_config(
    config_path: Optional[str | Path] = None,
    overrides_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load canonical defaults and apply optional user override file(s)."""
    base_fp = Path(config_path) if config_path else default_config_path()
    with open(base_fp, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    ov_fp = Path(overrides_path) if overrides_path else user_overrides_path()
    if ov_fp.is_file():
        with open(ov_fp, "r", encoding="utf-8") as f:
            ov = yaml.safe_load(f) or {}
        if isinstance(ov, dict):
            cfg = deep_merge(cfg, ov)
    return cfg


def save_user_overrides(
    current: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
    overrides_path: Optional[str | Path] = None,
) -> Path:
    """Persist only keys differing from defaults into `user_overrides.yaml`."""
    if defaults is None:
        with open(default_config_path(), "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}
    diff = extract_overrides(current, defaults)
    out_fp = Path(overrides_path) if overrides_path else user_overrides_path()
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as f:
        yaml.safe_dump(diff, f, sort_keys=False, allow_unicode=True)
    return out_fp
