"""
Override-only configuration loader, dot-path accessors, presets, and Per-App Profiles.
Adapted from Airpoint SDK's `config-core.ts` design (Apache-2.0):
  - `config/default.yaml` holds canonical defaults.
  - `config/user_overrides.yaml` stores ONLY user deltas.
  - Per-App Profiles under `profiles.apps.<bundle_id>` overlay cleanly on top of active config.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

# Friendly CLI / Airpoint aliases -> canonical AeroHand dotted keys
SETTING_KEY_ALIASES: Dict[str, str] = {
    "cursor.sensitivity": "cursor.pointer_sensitivity",
    "cursor.acceleration": "cursor.pointer_acceleration",
    "scroll.sensitivity": "scroll.speed",
    "scroll.inertia": "scroll.enable_inertia",
    "language": "ui.language",
    "dictation.lang": "dictation.language",
    "ui.hotkeys.toggle_tracking": "hotkeys.toggle_pause",
    "ui.hotkeys.recenter": "hotkeys.recenter",
    "ui.hotkeys.toggle_hud": "hotkeys.toggle_hud",
}

GESTURE_PRESETS: Dict[str, Dict[str, Any]] = {
    "airpoint": {
        "hand": {"gesture_preset": "airpoint"},
        "pose_state_machine": {
            "enable_click_hold": False,
            "tap_actions": {
                "thumb_middle_pinch": "left_click",
            },
            "hold_actions": {
                "thumb_ring_pinch": "scroll",
                "thumb_index_middle_pinch": "drag",
                "thumb_index_pinch": "dictation",
                "thumb_pinky_pinch": "window_tile",
                "thumb_pinky_base": "spaces_nav",
            },
        },
    },
    "classic": {
        "hand": {"gesture_preset": "classic"},
        "pose_state_machine": {
            "enable_click_hold": True,
            "tap_actions": {
                "thumb_index_pinch": "left_click",
            },
            "hold_actions": {
                "thumb_index_pinch": "drag",
                "thumb_middle_pinch": "dictation",
                "thumb_ring_pinch": "scroll",
                "thumb_pinky_pinch": "window_tile",
                "thumb_pinky_base": "spaces_nav",
            },
        },
    },
}


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


def resolve_setting_key(key: str) -> str:
    cleaned = key.strip()
    return SETTING_KEY_ALIASES.get(cleaned, cleaned)


def get_nested_key(cfg: Dict[str, Any], dotted_key: str) -> Tuple[bool, Any]:
    """Read a dot-separated key from config. Returns `(found, value)`."""
    resolved = resolve_setting_key(dotted_key)
    cur: Any = cfg
    for part in resolved.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def set_nested_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> str:
    """Set a dot-separated key inside `cfg` (creating intermediate dicts if needed)."""
    resolved = resolve_setting_key(dotted_key)
    parts = resolved.split(".")
    cur: Dict[str, Any] = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
    return resolved


def apply_gesture_preset(cfg: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Apply `'airpoint'` or `'classic'` preset cleanly (replacing tap/hold maps)."""
    preset = GESTURE_PRESETS.get(preset_name)
    if preset is None:
        return cfg
    out = deepcopy(cfg)
    out.setdefault("hand", {})["gesture_preset"] = preset_name
    psm = out.setdefault("pose_state_machine", {})
    p_psm = preset["pose_state_machine"]
    psm["enable_click_hold"] = bool(p_psm["enable_click_hold"])
    psm["tap_actions"] = deepcopy(p_psm["tap_actions"])
    psm["hold_actions"] = deepcopy(p_psm["hold_actions"])
    return out


def resolve_profile_config(base_cfg: Dict[str, Any], bundle_id: Optional[str]) -> Dict[str, Any]:
    """Overlay active frontmost-app profile settings on top of `base_cfg`."""
    if not bundle_id:
        return base_cfg
    prof_root = base_cfg.get("profiles") or {}
    if not prof_root.get("enabled", True):
        return base_cfg
    apps = prof_root.get("apps") or {}
    app_ov = apps.get(bundle_id)
    if not isinstance(app_ov, dict):
        return base_cfg
    filtered = {k: v for k, v in app_ov.items() if k != "name"}
    if not filtered:
        return base_cfg
    return deep_merge(base_cfg, filtered)


# Documented safe numeric ranges `(min_val, max_val)` enforced by `validate_and_clamp_config`
SAFE_NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "gestures.dictation_arm_ms": (200.0, 800.0),
    "gestures.dictation_priority_ms": (0.0, 5000.0),
    "gestures.zoom_arbitration_ms": (0.0, 1000.0),
    "gestures.zoom_hold_ms": (0.0, 1000.0),
    "dictation.min_duration_s": (0.1, 10.0),
    "dictation.min_rms": (0.0005, 0.5),
    "dictation.min_voiced_ms": (50.0, 5000.0),
    "cursor.click_dwell_lock_ms": (0.0, 500.0),
    "cursor.click_dwell_lock_cooldown_ms": (0.0, 2000.0),
    "cursor.click_lookback_ms": (0.0, 300.0),
    "cursor.click_lookback_max_speed_px_s": (0.0, 5000.0),
    "cursor.click_lookback_max_offset_px": (0.0, 200.0),
    "cursor.drag_lookback_ms": (0.0, 200.0),
    "cursor.drag_dwell_lock_ms": (0.0, 300.0),
    "collector.negative_window_s": (0.5, 10.0),
    "classifier.heuristic_3d.approach_window_ms": (0.0, 5000.0),
    "classifier.heuristic_3d.approach_open_margin": (0.0, 0.40),
    "classifier.heuristic_3d.approach_decay_tau_ms": (50.0, 5000.0),
    "classifier.heuristic_3d.index_ext_min": (0.5, 2.0),
    "classifier.heuristic_3d.max_wrist_speed": (0.05, 2.0),
    "classifier.heuristic_3d.max_wrist_speed_pw": (0.2, 20.0),
    "classifier.heuristic_3d.motion_ema_tau_ms": (5.0, 500.0),
    "classifier.heuristic_3d.max_wrist_ang_speed_deg_s": (20.0, 600.0),
    "classifier.heuristic_3d.grip_half_curled_min": (2.0, 4.0),
    "classifier.heuristic_3d.grip_loose_ratio": (0.7, 1.5),
    "classifier.veto_min_prob": (0.05, 0.95),
    "classifier.veto_idle_prob": (0.05, 0.95),
    "classifier.veto_entry_window_ms": (0.0, 2000.0),
    "classifier.max_onnx_p95_ms": (1.0, 30.0),
}


def validate_and_clamp_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp documented safety-critical numeric parameters into safe bounds (in-place)."""
    for dotted_key, (lo, hi) in SAFE_NUMERIC_RANGES.items():
        found, val = get_nested_key(cfg, dotted_key)
        if not found or isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        clamped = max(lo, min(hi, float(val)))
        if isinstance(val, int) and float(lo).is_integer() and float(hi).is_integer():
            set_nested_key(cfg, dotted_key, int(round(clamped)))
        else:
            set_nested_key(cfg, dotted_key, clamped)
    return cfg


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
    return validate_and_clamp_config(cfg)


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
