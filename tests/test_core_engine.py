"""Comprehensive unit & integration tests for AeroHand core, ML, and safety invariants."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from aerohand.config import deep_merge, extract_overrides, load_config
from aerohand.core.engine import HandCursorEngine
from aerohand.core.filters import LandmarkSmoother, OneEuroFilter1D
from aerohand.core.kinematics import CursorKinematics, CursorLerpInterpolator
from aerohand.core.palm_wheel import PalmWheelDetector, calculate_radial_cylinder
from aerohand.core.pose_state_machine import PoseStateMachine
from aerohand.core.scroll_controller import ScrollController
from aerohand.core.spaces_nav import SpacesNavController
from aerohand.core.types import HandInput, Landmark, PoseName, idle_probabilities
from aerohand.core.window_tile import WindowTileController
from aerohand.ml.collector import PinchDataCollector
from aerohand.ml.features_3d import FEATURE_DIM_3D, extract_invariant_3d_features
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier
from aerohand.platform.input_macos import MacOSInputController


def _make_open_hand(forward_z: float = -0.25) -> List[Landmark]:
    """Synthetic flat right hand facing camera in 3D coordinates (origin near wrist)."""
    # Palm length ~0.10, palm width ~0.08
    pts = [
        (0.50, 0.70, 0.00),             # 0: wrist
        (0.45, 0.66, -0.01),            # 1: thumb cmc
        (0.42, 0.62, -0.02),            # 2: thumb mcp
        (0.39, 0.58, -0.03),            # 3: thumb ip
        (0.36, 0.54, -0.04),            # 4: thumb tip
        (0.46, 0.60, -0.02),            # 5: index mcp
        (0.46, 0.54, -0.04),            # 6: index pip
        (0.46, 0.49, -0.06),            # 7: index dip
        (0.46, 0.44, forward_z),        # 8: index tip
        (0.50, 0.59, -0.02),            # 9: middle mcp
        (0.50, 0.52, -0.04),            # 10: middle pip
        (0.50, 0.46, -0.05),            # 11: middle dip
        (0.50, 0.41, -0.06),            # 12: middle tip
        (0.54, 0.60, -0.02),            # 13: ring mcp
        (0.54, 0.54, -0.04),            # 14: ring pip
        (0.54, 0.49, -0.05),            # 15: ring dip
        (0.54, 0.44, -0.06),            # 16: ring tip
        (0.58, 0.62, -0.02),            # 17: pinky mcp
        (0.58, 0.57, -0.04),            # 18: pinky pip
        (0.58, 0.53, -0.05),            # 19: pinky dip
        (0.58, 0.49, -0.06),            # 20: pinky tip
    ]
    return [Landmark(x=x, y=y, z=z) for x, y, z in pts]


def _rotate_translate_scale(
    lms: List[Landmark],
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    scale: float = 1.35,
    offset=(0.15, -0.10, 0.08),
) -> List[Landmark]:
    yr, pr, rr = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(yr), 0, math.sin(yr)], [0, 1, 0], [-math.sin(yr), 0, math.cos(yr)]])
    rx = np.array([[1, 0, 0], [0, math.cos(pr), -math.sin(pr)], [0, math.sin(pr), math.cos(pr)]])
    rz = np.array([[math.cos(rr), -math.sin(rr), 0], [math.sin(rr), math.cos(rr), 0], [0, 0, 1]])
    r = rz @ ry @ rx
    arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float64)
    arr = (arr * scale) @ r.T + np.array(offset)
    return [Landmark(float(p[0]), float(p[1]), float(p[2])) for p in arr]


def test_config_override_roundtrip():
    base = load_config()
    modified = deep_merge(base, {"cursor": {"pointer_sensitivity": 3.1}})
    diff = extract_overrides(modified, base)
    assert diff == {"cursor": {"pointer_sensitivity": 3.1}}


def test_one_euro_gap_reset_and_landmark_smoother_skip():
    flt = OneEuroFilter1D(min_cutoff=1.0, beta=0.1, gap_reset_ms=100.0)
    assert flt.filter(0.0, 0.0) == 0.0
    v1 = flt.filter(1.0, 33.0)
    assert 0.0 < v1 < 1.0
    # After >= 100ms hiccup, filter must snap cleanly with zero lag
    assert flt.filter(5.0, 200.0) == 5.0

    sm = LandmarkSmoother(min_cutoff=2.0, beta=45.0)
    h0 = _make_open_hand()
    h1 = [Landmark(p.x + 0.02, p.y + 0.01, p.z) for p in h0]
    sm.smooth(h0, "Right", 0.0, skip_indices=[8])
    out = sm.smooth(h1, "Right", 33.0, skip_indices=[8])
    assert out[8].x == pytest.approx(h1[8].x)
    assert out[0].x != pytest.approx(h1[0].x)


def test_cursor_kinematics_jump_reject_and_lerp():
    kin = CursorKinematics({"reacquire_skip_frames": 0, "enable_filter": False, "pointer_sensitivity": 2.0})
    lms = _make_open_hand()
    kin.set_cursor_position(500.0, 500.0)
    kin._skip_remaining = 0
    o0 = kin.process(100.0, 100.0, lms, 0.0, 1920.0, 1080.0)
    assert not o0.is_valid
    o1 = kin.process(110.0, 105.0, lms, 33.0, 1920.0, 1080.0)
    assert o1.is_valid and o1.x > 500.0 and o1.y > 500.0
    # Huge jump > 15% of 1920 (= 288 px) must be rejected
    o2 = kin.process(600.0, 105.0, lms, 66.0, 1920.0, 1080.0)
    assert not o2.is_valid and o2.x == pytest.approx(o1.x)

    lerp = CursorLerpInterpolator(lerp_speed=55.0)
    lerp.reset((0.0, 0.0))
    lerp.set_target(100.0, 0.0)
    s = lerp.step(1.0 / 60.0)
    assert s is not None and 40.0 < s[0] < 90.0


def test_pose_state_machine_tap_hold_and_dual_thumb_ring():
    cfg = load_config()["pose_state_machine"]
    psm = PoseStateMachine(cfg)

    def _p(pose: PoseName, val: float):
        d = idle_probabilities()
        d[PoseName.IDLE] = 1.0 - val
        d[pose] = val
        return d

    # 1. Tap thumb_middle_pinch -> left_click on exit
    assert psm.process(_p(PoseName.THUMB_MIDDLE_PINCH, 0.9), 0.0) == []
    evs = psm.process(_p(PoseName.THUMB_MIDDLE_PINCH, 0.9), 33.0)
    assert [e.type for e in evs] == ["pose_enter"]
    assert psm.process(_p(PoseName.THUMB_MIDDLE_PINCH, 0.1), 66.0) == []
    evs = psm.process(_p(PoseName.THUMB_MIDDLE_PINCH, 0.1), 100.0)
    assert [e.type for e in evs] == ["tap"] and evs[0].pose == PoseName.THUMB_MIDDLE_PINCH

    # 2. Dual thumb_ring_pinch: hold still >= 180ms -> hold_start, then hold_end with low motion
    psm.process(_p(PoseName.THUMB_RING_PINCH, 0.9), 200.0)
    psm.process(_p(PoseName.THUMB_RING_PINCH, 0.9), 233.0)
    evs = psm.process(_p(PoseName.THUMB_RING_PINCH, 0.9), 450.0)
    assert [e.type for e in evs] == ["hold_start"]
    # 5 exit frames for thumb_ring_pinch
    final_evs = []
    for k in range(5):
        final_evs = psm.process(_p(PoseName.THUMB_RING_PINCH, 0.1), 483.0 + k * 33.0)
    assert [e.type for e in final_evs] == ["hold_end"]
    assert final_evs[0].motion_distance == 0.0


def test_scroll_spaces_and_window_tile_controllers():
    sc = ScrollController()
    # Accumulate vertical motion > 15px -> locks to 'y'
    out = None
    for i in range(5):
        out = sc.process(1.0, -5.0, True, i * 33.0)
    assert out is not None and out.is_scrolling and out.axis == "y" and out.dx == 0.0 and out.dy > 0.0
    # Release -> enters inertia
    out_rel = sc.process(0.0, 0.0, False, 200.0)
    assert out_rel.is_inertia

    sn = SpacesNavController({"sensitivity_px": 20.0, "invert": True})
    assert not sn.process(10.0, 0.0, True, 0.0).triggered
    trig = sn.process(15.0, 0.0, True, 33.0)
    assert trig.triggered and trig.direction == "left"
    # Does not re-trigger during same hold
    assert not sn.process(20.0, 0.0, True, 66.0).triggered

    wt = WindowTileController({"sensitivity_px": 20.0})
    wt.process(10.0, 0.0, True, 0.0)
    assert not wt.process(15.0, 0.0, True, 33.0).triggered
    rel = wt.process(0.0, 0.0, False, 66.0)
    assert rel.triggered and rel.direction == "right"


def test_palm_cylinder_and_palm_wheel_activation():
    lms = _make_open_hand(forward_z=-0.06)
    cyl = calculate_radial_cylinder(lms)
    assert cyl is not None
    assert all(v > 1.1 for v in cyl.normalized_distances.values())

    pw = PalmWheelDetector({"activation_hold_ms": 200.0, "cooldown_ms": 100.0})
    assert pw.process(lms, 0.0, "Right").state == "activating"
    assert pw.process(lms, 220.0, "Right").state == "active"

    # Curl index tip right onto palm center -> pressed, then release -> fires 'mission_control'
    curled = list(lms)
    cx = 0.5 * (lms[0].x + lms[9].x)
    cy = 0.5 * (lms[0].y + lms[9].y)
    cz = 0.5 * (lms[0].z + lms[9].z)
    curled[8] = Landmark(cx, cy, cz)
    pw.process(curled, 260.0, "Right")
    out_fire = pw.process(lms, 300.0, "Right")
    assert out_fire.state == "confirmed"
    assert out_fire.triggered_action == "mission_control"


def test_heuristic_3d_pinch_and_rotation_invariance():
    clf = Heuristic3DPinchClassifier()
    open_hand = _make_open_hand()
    pred_open = clf.predict(open_hand, open_hand, "Right", 0.0)
    assert pred_open.pose == PoseName.IDLE

    # Move thumb tip close to middle tip -> thumb_middle_pinch
    pinched = list(open_hand)
    m_tip = pinched[12]
    pinched[4] = Landmark(m_tip.x - 0.005, m_tip.y, m_tip.z)
    pred_mid = clf.predict(pinched, pinched, "Right", 33.0)
    assert pred_mid.pose == PoseName.THUMB_MIDDLE_PINCH
    assert pred_mid.probabilities[PoseName.THUMB_MIDDLE_PINCH] > 0.80

    # Apply extreme 3D yaw +60, pitch +45, roll -30 + scale 1.4x + translation
    tilted = _rotate_translate_scale(pinched, yaw_deg=60.0, pitch_deg=45.0, roll_deg=-30.0)
    pred_tilted = clf.predict(tilted, tilted, "Right", 66.0)
    assert pred_tilted.pose == PoseName.THUMB_MIDDLE_PINCH
    assert pred_tilted.probabilities[PoseName.THUMB_MIDDLE_PINCH] == pytest.approx(
        pred_mid.probabilities[PoseName.THUMB_MIDDLE_PINCH], abs=1e-4
    )

    # Check 348-dim ML feature invariance under 3D rigid transform + scale
    f1 = extract_invariant_3d_features(pinched, pinched, "Right")
    f2 = extract_invariant_3d_features(tilted, tilted, "Right")
    assert f1 is not None and f2 is not None
    assert f1.shape == (FEATURE_DIM_3D,)
    assert np.allclose(f1, f2, atol=1e-4)


def test_engine_pointing_gate_fail_open_and_typing_guard():
    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0

    class _BoomClassifier:
        def predict(self, *args, **kwargs):
            raise RuntimeError("boom")

        def reset(self):
            pass

    eng = HandCursorEngine(cfg, classifier=_BoomClassifier())
    # Pointing away (dirZ > 0) -> gate blocks pointing
    away_hand = _make_open_hand(forward_z=+0.10)
    outs = eng.process_frame([HandInput("Right", away_hand, away_hand)], 0.0, 1920.0, 1080.0)
    assert len(outs) == 1 and not outs[0].cursor.is_pointing

    # Pointing toward screen (dirZ < -0.15) with crashing classifier -> fail-open to IDLE
    fwd_hand = _make_open_hand(forward_z=-0.25)
    outs2 = eng.process_frame([HandInput("Right", fwd_hand, fwd_hand)], 33.0, 1920.0, 1080.0)
    assert outs2[0].cursor.is_pointing
    assert outs2[0].pose.pose == PoseName.IDLE

    # Typing guard counter mock blocks keyboard gestures without Input Monitoring
    ctl = MacOSInputController(
        cfg["typing_guard"],
        dry_run=True,
        counter_query_fn=lambda _state, _ev: 0.2,  # 200ms since last physical event (< 1.5s)
    )
    if ctl._counter_query_fn is not None:
        # On macOS with Quartz installed, typing guard blocks clicks immediately
        try:
            import Quartz  # noqa: F401
            assert ctl.is_action_blocked("click") is True
        except ImportError:
            pass


def test_data_collector_save(tmp_path: Path):
    col = PinchDataCollector()
    col.set_pose(PoseName.THUMB_MIDDLE_PINCH)
    col.toggle_recording()
    h = _make_open_hand()
    col.add_frame(h, h, "Right", 0.95, 1000)
    fp = col.save(tmp_path)
    assert fp.is_file()
    assert col.counts()[PoseName.THUMB_MIDDLE_PINCH.value] == 1


def test_palm_wheel_nested_submenu_back_and_fist_dismiss():
    lms = _make_open_hand(forward_z=-0.06)
    pw = PalmWheelDetector({"activation_hold_ms": 200.0, "cooldown_ms": 50.0}, hand="Right")
    assert pw.process(lms, 0.0, "Right").state == "activating"
    act = pw.process(lms, 220.0, "Right")
    assert act.state == "active"
    assert len(act.slots) == 5

    # Curl pinky (slot R5 = Player submenu) and release -> enters Player submenu!
    cx = 0.5 * (lms[0].x + lms[9].x)
    cy = 0.5 * (lms[0].y + lms[9].y)
    cz = 0.5 * (lms[0].z + lms[9].z)
    curled_pinky = list(lms)
    curled_pinky[20] = Landmark(cx, cy, cz)
    pw.process(curled_pinky, 260.0, "Right")
    entered = pw.process(lms, 300.0, "Right")
    assert entered.menu_label == "Player"
    assert entered.menu_path == ["Player"]

    # After confirmation_hold_ms (200ms) with open palm -> stays active inside Player submenu!
    stay = pw.process(lms, 550.0, "Right")
    assert stay.state == "active"
    assert stay.menu_label == "Player"

    # Curl thumb (submenu_back in Player) -> returns to root
    curled_thumb = list(lms)
    curled_thumb[4] = Landmark(cx, cy, cz)
    pw.process(curled_thumb, 590.0, "Right")
    back = pw.process(lms, 630.0, "Right")
    assert back.menu_path == []
    assert back.menu_label is None

    # Stay active again, then close fist (>1 finger down) -> rapid dismiss to inactive
    pw.process(lms, 860.0, "Right")
    fist = list(lms)
    fist[8] = Landmark(cx, cy, cz)
    fist[12] = Landmark(cx, cy, cz)
    dismissed = pw.process(fist, 900.0, "Right")
    assert dismissed.state == "inactive"


def test_two_hand_zoom_controller():
    from aerohand.core.zoom_controller import ZoomController

    zc = ZoomController({"enable_zoom": True, "zoom_speed": 1.0})
    o0 = zc.process((0.35, 0.5), (0.65, 0.5), True, 0.0)
    assert o0.is_zooming and o0.step == 0
    steps = []
    for k in range(1, 10):
        d = 0.02 * k
        o = zc.process((0.35 - d, 0.5), (0.65 + d, 0.5), True, k * 120.0)
        if o.step != 0:
            steps.append(o.step)
    assert any(s > 0 for s in steps)
    assert not zc.process((0.35, 0.5), (0.65, 0.5), False, 2000.0).is_zooming


def test_presets_profiles_and_nested_config_keys():
    from aerohand.config import (
        apply_gesture_preset,
        get_nested_key,
        resolve_profile_config,
        set_nested_key,
    )

    cfg = load_config()
    classic = apply_gesture_preset(cfg, "classic")
    assert classic["pose_state_machine"]["enable_click_hold"] is True
    assert classic["pose_state_machine"]["tap_actions"]["thumb_index_pinch"] == "left_click"

    # Alias resolution
    found, val = get_nested_key(cfg, "cursor.sensitivity")
    assert found and val == cfg["cursor"]["pointer_sensitivity"]
    set_nested_key(cfg, "cursor.sensitivity", 2.95)
    assert cfg["cursor"]["pointer_sensitivity"] == 2.95

    # Per-app profile merge
    prof = resolve_profile_config(cfg, "com.microsoft.VSCode")
    assert prof["scroll"]["speed"] == 1.2


def test_airmouse_885_features_and_left_hand_mirror():
    from aerohand.ml.features_885 import AIRMOUSE_885_DIM, extract_airmouse_885_features

    right = _make_open_hand()
    f_r = extract_airmouse_885_features(right, right, "Right")
    assert f_r.shape == (AIRMOUSE_885_DIM,)
    assert np.all(np.isfinite(f_r))

    # Construct exact X-mirrored left hand
    left_2d = [Landmark(1.0 - p.x, p.y, p.z) for p in right]
    left_3d = [Landmark(-p.x, p.y, p.z) for p in right]
    f_l = extract_airmouse_885_features(left_2d, left_3d, "Left")
    assert np.allclose(f_r, f_l, atol=1e-5)


def test_ax_tile_rects_i18n_audio_and_local_control_api(tmp_path: Path):
    import json
    import urllib.request

    from aerohand.i18n import TRANSLATIONS
    from aerohand.platform.audio import ensure_default_sound_assets
    from aerohand.platform.control_api import LocalControlAPIServer, handle_aerohand_url
    from aerohand.platform.input_macos import compute_tile_rect

    # 1. Tile rect calculations
    vis = (0.0, 28.0, 1440.0, 872.0)
    left = compute_tile_rect("left", vis)
    right = compute_tile_rect("right", vis)
    top = compute_tile_rect("top", vis)
    bottom = compute_tile_rect("bottom", vis)
    assert left and right and top and bottom
    assert left[2] == pytest.approx(right[2])
    assert top[2] > left[2] * 1.8

    # 2. Bilingual vi / en translation completeness
    assert set(TRANSLATIONS["vi"].keys()) == set(TRANSLATIONS["en"].keys())
    assert set(TRANSLATIONS["vi"]["tabs"].keys()) == set(TRANSLATIONS["en"]["tabs"].keys())

    # 3. Clean-room WAV click synthesis
    wavs = ensure_default_sound_assets(tmp_path)
    assert all(p.is_file() and p.stat().st_size > 200 for p in wavs.values())

    # 4. URL scheme handler
    hit = []
    assert handle_aerohand_url("aerohand://tracking/off", on_tracking=lambda v: hit.append(v))
    assert hit == [False]

    # 5. Local Control HTTP API roundtrip on ephemeral port
    state = {"tracking": True, "cfg": load_config()}
    tok_fp = tmp_path / "api_token"
    srv = LocalControlAPIServer(
        host="127.0.0.1",
        port=0,
        get_status=lambda: {"ok": True, "tracking": state["tracking"]},
        get_vitals=lambda: {"cpu_usage": 5.0, "ram_used_gb": 4.0, "ram_total_gb": 16.0},
        get_config=lambda: state["cfg"],
        apply_config=lambda c: state.__setitem__("cfg", c),
        set_tracking_enabled=lambda en: (state.__setitem__("tracking", en) or en),
        trigger_recenter=lambda: None,
        token_path=tok_fp,
    )
    assert srv.start()
    try:
        actual_port = srv.bound_port
        assert tok_fp.is_file()
        assert (tok_fp.stat().st_mode & 0o777) == 0o600
        tok = tok_fp.read_text(encoding="utf-8").strip()
        assert tok == srv.token

        req_get = urllib.request.Request(
            f"http://127.0.0.1:{actual_port}/api/status",
            headers={"X-AeroHand-Token": tok},
            method="GET",
        )
        with urllib.request.urlopen(req_get, timeout=3) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True and body["tracking"] is True

        req = urllib.request.Request(
            f"http://127.0.0.1:{actual_port}/api/tracking",
            data=json.dumps({"enabled": False}).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-AeroHand-Token": tok},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            body2 = json.loads(resp.read().decode("utf-8"))
        assert body2["enabled"] is False
        assert state["tracking"] is False
    finally:
        srv.stop()


class _FakeInputCtl:
    def __init__(self) -> None:
        self.events: List[str] = []
        self.dry_run = False
        self._pressed_mouse_buttons: set[str] = set()
        self._pressed_keys: set[int] = set()
        self.last_guard_reason = ""

    def mouse_move(self, x: float, y: float, delta_x=None, delta_y=None) -> bool:
        self.events.append("mouse_move")
        return True

    def mouse_click(self, x: float, y: float, button: str = "left", click_count: int = 1) -> bool:
        self.events.append(f"mouse_click:{button}:{click_count}")
        return True

    def mouse_down(self, x: float, y: float, button: str = "left") -> bool:
        self._pressed_mouse_buttons.add(button)
        self.events.append(f"mouse_down:{button}")
        return True

    def mouse_up(self, x: float, y: float, button: str = "left") -> bool:
        self._pressed_mouse_buttons.discard(button)
        self.events.append(f"mouse_up:{button}")
        return True

    def end_scroll_phase(self) -> None:
        self.events.append("end_scroll_phase")

    def release_all(self) -> None:
        self._pressed_mouse_buttons.clear()
        self._pressed_keys.clear()
        self.events.append("release_all")


def _make_hand_output(hand: str, *, grab_started: bool = False, is_grabbing: bool = False):
    from aerohand.core.types import CursorOutput, HandOutput, PoseOutput

    return HandOutput(
        hand=hand,  # type: ignore[arg-type]
        cursor=CursorOutput(hand=hand, x=500.0, y=500.0, is_pointing=True, is_valid=True),  # type: ignore[arg-type]
        pose=PoseOutput(
            hand=hand,  # type: ignore[arg-type]
            pose=PoseName.THUMB_INDEX_MIDDLE_PINCH if is_grabbing else PoseName.IDLE,
            active_pose=PoseName.THUMB_INDEX_MIDDLE_PINCH if is_grabbing else None,
            grab_just_started=grab_started,
            is_grabbing=is_grabbing,
        ),
    )


def test_drag_released_when_only_secondary_hand_visible():
    from aerohand.runtime import AeroHandRuntime

    now = [1000.0]
    cfg = load_config()
    cfg["hand"]["cursor_hand"] = "Right"
    cfg["runtime"]["hand_lost_release_ms"] = 350
    cfg["audio"]["enabled"] = False
    rt = AeroHandRuntime(cfg, dry_run=False, clock_ms=lambda: now[0])
    fake = _FakeInputCtl()
    rt.input_ctl = fake  # type: ignore[assignment]

    # Right hand starts drag at t=1000ms
    right_drag = _make_hand_output("Right", grab_started=True, is_grabbing=True)
    rt._last_frame_ms = now[0]
    rt._dispatch_actions([right_drag], right_drag, now[0])
    assert fake.events.count("mouse_down:left") == 1
    assert fake.events.count("release_all") == 0

    # Only Left (secondary) hand remains visible; Right vanished -> primary is None
    left_only = _make_hand_output("Left", grab_started=False, is_grabbing=False)
    now[0] = 1200.0
    rt._last_frame_ms = now[0]
    primary = rt._select_primary_hand([left_only])
    assert primary is None
    rt._dispatch_actions([left_only], primary, now[0])
    assert fake.events.count("release_all") == 0

    # At t=1400ms (400ms without primary >= 350ms) -> release_all called exactly once
    now[0] = 1400.0
    rt._last_frame_ms = now[0]
    rt._dispatch_actions([left_only], None, now[0])
    assert fake.events.count("release_all") == 1

    # Subsequent ticks must NOT spam release_all
    now[0] = 2000.0
    rt._last_frame_ms = now[0]
    rt._dispatch_actions([left_only], None, now[0])
    rt._enforce_input_safety(now[0])
    assert fake.events.count("release_all") == 1


def test_drag_released_on_camera_stall():
    from aerohand.runtime import AeroHandRuntime

    now = [5000.0]
    cfg = load_config()
    cfg["runtime"]["hand_lost_release_ms"] = 1000
    cfg["runtime"]["camera_stall_release_ms"] = 500
    cfg["audio"]["enabled"] = False
    rt = AeroHandRuntime(cfg, dry_run=False, clock_ms=lambda: now[0])
    fake = _FakeInputCtl()
    rt.input_ctl = fake  # type: ignore[assignment]

    right_drag = _make_hand_output("Right", grab_started=True, is_grabbing=True)
    rt._last_frame_ms = now[0]
    rt._dispatch_actions([right_drag], right_drag, now[0])
    assert fake.events.count("mouse_down:left") == 1

    # Camera stops yielding frames (cf.success == False -> _last_frame_ms does not advance)
    now[0] = 5400.0
    assert rt._enforce_input_safety(now[0]) is False
    assert fake.events.count("release_all") == 0

    now[0] = 5550.0
    assert rt._enforce_input_safety(now[0]) is True
    assert fake.events.count("release_all") == 1
    # Calling again while nothing is held does nothing
    now[0] = 6000.0
    assert rt._enforce_input_safety(now[0]) is False
    assert fake.events.count("release_all") == 1


def test_max_drag_watchdog_runs_without_primary():
    from aerohand.runtime import AeroHandRuntime

    now = [1000.0]
    cfg = load_config()
    cfg["runtime"]["max_drag_duration_ms"] = 20000
    cfg["runtime"]["hand_lost_release_ms"] = 999999
    cfg["runtime"]["camera_stall_release_ms"] = 999999
    cfg["audio"]["enabled"] = False
    rt = AeroHandRuntime(cfg, dry_run=False, clock_ms=lambda: now[0])
    fake = _FakeInputCtl()
    rt.input_ctl = fake  # type: ignore[assignment]

    right_drag = _make_hand_output("Right", grab_started=True, is_grabbing=True)
    rt._last_frame_ms = now[0]
    rt._dispatch_actions([right_drag], right_drag, now[0])
    assert fake.events.count("mouse_down:left") == 1

    # Exceed max_drag_duration_ms with primary=None -> watchdog still fires before early return!
    now[0] = 22000.0
    rt._last_frame_ms = now[0]
    rt._dispatch_actions([], None, now[0])
    assert fake.events.count("release_all") == 1
    assert rt._drag_started_ms is None


def _start_test_api(tmp_path: Path):
    from aerohand.platform.control_api import LocalControlAPIServer

    state = {"tracking": True, "cfg": load_config()}
    srv = LocalControlAPIServer(
        host="0.0.0.0",  # must be forced to 127.0.0.1
        port=0,
        get_status=lambda: {"ok": True, "tracking": state["tracking"]},
        get_vitals=lambda: {"cpu_usage": 1.0},
        get_config=lambda: state["cfg"],
        apply_config=lambda c: state.__setitem__("cfg", c),
        set_tracking_enabled=lambda en: (state.__setitem__("tracking", en) or en),
        trigger_recenter=lambda: None,
        token_path=tmp_path / "api_token",
    )
    assert srv.host == "127.0.0.1"
    assert srv.start()
    return srv, state


def test_api_rejects_foreign_host(tmp_path: Path):
    import http.client

    srv, _ = _start_test_api(tmp_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.bound_port, timeout=3)
        conn.putrequest("GET", "/api/status", skip_host=True)
        conn.putheader("Host", "evil.rebind.example:17420")
        conn.putheader("X-AeroHand-Token", srv.token)
        conn.endheaders()
        resp = conn.getresponse()
        assert resp.status == 403
        conn.close()
    finally:
        srv.stop()


def test_api_rejects_origin_header(tmp_path: Path):
    import http.client
    import json

    srv, _ = _start_test_api(tmp_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.bound_port, timeout=3)
        payload = json.dumps({"key": "cursor.sensitivity", "value": 3.0}).encode("utf-8")
        conn.request(
            "PATCH",
            "/api/settings",
            body=payload,
            headers={
                "Host": f"127.0.0.1:{srv.bound_port}",
                "Origin": "http://evil.example",
                "Content-Type": "application/json",
                "X-AeroHand-Token": srv.token,
            },
        )
        resp = conn.getresponse()
        assert resp.status == 403
        conn.close()
    finally:
        srv.stop()


def test_api_requires_token(tmp_path: Path):
    import http.client

    srv, _ = _start_test_api(tmp_path)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.bound_port, timeout=3)
        conn.request("GET", "/api/status", headers={"Host": f"127.0.0.1:{srv.bound_port}"})
        resp = conn.getresponse()
        assert resp.status == 401
        conn.close()

        conn2 = http.client.HTTPConnection("127.0.0.1", srv.bound_port, timeout=3)
        conn2.request(
            "GET",
            "/api/status",
            headers={"Host": f"127.0.0.1:{srv.bound_port}", "X-AeroHand-Token": "wrong-token"},
        )
        resp2 = conn2.getresponse()
        assert resp2.status == 401
        conn2.close()
    finally:
        srv.stop()


def test_api_rejects_unknown_setting_key(tmp_path: Path):
    import http.client
    import json

    srv, _ = _start_test_api(tmp_path)
    try:
        for body in (
            {"key": "nonexistent.evil_key", "value": 123},
            {"key": "api.host", "value": "0.0.0.0"},
            {"key": "cursor", "value": {"pointer_sensitivity": 9.9}},
            {"key": "cursor.pointer_sensitivity", "value": [1, 2, 3]},
        ):
            conn = http.client.HTTPConnection("127.0.0.1", srv.bound_port, timeout=3)
            raw = json.dumps(body).encode("utf-8")
            conn.request(
                "PATCH",
                "/api/settings",
                body=raw,
                headers={
                    "Host": f"127.0.0.1:{srv.bound_port}",
                    "Content-Type": "application/json",
                    "X-AeroHand-Token": srv.token,
                },
            )
            resp = conn.getresponse()
            assert resp.status == 400, f"Expected 400 for {body}, got {resp.status}"
            conn.close()

        # Oversized body (> 64 KiB) returns 413
        conn3 = http.client.HTTPConnection("127.0.0.1", srv.bound_port, timeout=3)
        big = b'{"key":"ui.language","value":"' + (b"a" * 70000) + b'"}'
        conn3.request(
            "PATCH",
            "/api/settings",
            body=big,
            headers={
                "Host": f"127.0.0.1:{srv.bound_port}",
                "Content-Type": "application/json",
                "X-AeroHand-Token": srv.token,
            },
        )
        resp3 = conn3.getresponse()
        assert resp3.status == 413
        conn3.close()
    finally:
        srv.stop()


def test_collector_roundtrip_to_trainer(tmp_path: Path):
    import importlib.util
    import json

    from aerohand.ml.collector import PoseCollector

    spec = importlib.util.spec_from_file_location(
        "train_tcn_mod", Path(__file__).resolve().parent.parent / "scripts" / "train_tcn.py"
    )
    assert spec is not None and spec.loader is not None
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    col = PoseCollector(session_meta={"subject": "minh", "lighting": "bright"})
    col.toggle_recording()
    h = _make_open_hand()
    ts = 1000
    col.set_pose(PoseName.IDLE)
    for _ in range(20):
        col.add_frame(h, h, "Right", 0.95, ts)
        ts += 33
    col.set_pose(PoseName.THUMB_MIDDLE_PINCH)
    for _ in range(20):
        col.add_frame(h, h, "Right", 0.95, ts)
        ts += 33

    fp = col.save(tmp_path)
    saved = json.loads(fp.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["session_meta"]["subject"] == "minh"

    xs, ys = train_mod.load_session_sequences(fp, feature_set="invariant348", seq_len=5)
    assert xs.shape[0] == 36  # 40 - 5 + 1
    assert xs.shape[1] == 5
    assert xs.shape[2] == FEATURE_DIM_3D
    assert int((ys == 0).sum()) > 0
    assert int((ys == 2).sum()) > 0  # thumb_middle_pinch index == 2


def test_trainer_reads_sdk_camelcase_session(tmp_path: Path):
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "train_tcn_mod2", Path(__file__).resolve().parent.parent / "scripts" / "train_tcn.py"
    )
    assert spec is not None and spec.loader is not None
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    h = _make_open_hand()
    lms_dicts = [{"x": p.x, "y": p.y, "z": p.z} for p in h]
    samples = []
    for i in range(12):
        samples.append(
            {
                "timestamp": 2000 + i * 33,
                "labelName": "thumb_ring_pinch",
                "handedness": "Right",
                "landmarks": lms_dicts,
                "worldLandmarks": lms_dicts,
            }
        )
    # Include an unknown label sample that must be skipped and counted
    samples.append(
        {
            "timestamp": 2500,
            "labelName": "bogus_pose",
            "handedness": "Right",
            "landmarks": lms_dicts,
            "worldLandmarks": lms_dicts,
        }
    )
    fp = tmp_path / "sdk_v4_session.json"
    fp.write_text(json.dumps({"version": "4.0", "samples": samples}), encoding="utf-8")

    skip_stats = {}
    xs, ys = train_mod.load_session_sequences(
        fp, feature_set="invariant348", seq_len=5, skip_stats=skip_stats
    )
    assert xs.shape == (8, 5, FEATURE_DIM_3D)
    assert np.all(ys == 3)  # thumb_ring_pinch index == 3
    assert skip_stats.get("unknown_label") == 1


def test_to_quartz_to_cocoa_roundtrip():
    from aerohand.platform.input_macos import DisplayRect, ScreenGeometry

    displays = [
        DisplayRect(x=0.0, y=0.0, width=1512.0, height=982.0, is_primary=True),
        DisplayRect(x=-1920.0, y=-98.0, width=1920.0, height=1080.0, is_primary=False),
    ]
    geom = ScreenGeometry(displays_provider=lambda: displays)

    for pt in [(0.0, 0.0), (756.0, 491.0), (-960.0, 442.0), (-1920.0, -98.0)]:
        q = geom.to_quartz(pt)
        c = geom.to_cocoa(q)
        assert c == pytest.approx(pt)

    # Rect roundtrip on negative-origin secondary monitor
    cocoa_r = (-1920.0, 0.0, 1920.0, 1080.0)
    quartz_r = geom.to_quartz_rect(cocoa_r)
    assert quartz_r == pytest.approx((-1920.0, -98.0, 1920.0, 1080.0))
    assert geom.to_cocoa_rect(quartz_r) == pytest.approx(cocoa_r)

    # Target display selection across both monitors
    assert geom.target_display_for((-500.0, 300.0)) == displays[1]
    assert geom.target_display_for((500.0, 300.0)) == displays[0]


def test_recenter_uses_active_display_rect():
    from aerohand.platform.input_macos import DisplayRect, ScreenGeometry
    from aerohand.runtime import AeroHandRuntime

    displays = [
        DisplayRect(x=0.0, y=0.0, width=1512.0, height=982.0, is_primary=True),
        DisplayRect(x=-1920.0, y=0.0, width=1920.0, height=1080.0, is_primary=False),
    ]
    geom = ScreenGeometry(displays_provider=lambda: displays)
    moved_to: List[tuple[float, float]] = []

    class _RecInput(_FakeInputCtl):
        def mouse_move(self, x: float, y: float, delta_x=None, delta_y=None) -> bool:
            moved_to.append((x, y))
            return True

    cfg = load_config()
    cfg["cursor"]["display"] = "cursor"
    cfg["audio"]["enabled"] = False
    # Cursor is currently on the left (negative-X) secondary monitor
    rt = AeroHandRuntime(
        cfg,
        dry_run=False,
        screen_geom=geom,
        cursor_pos_fn=lambda: (-400.0, 200.0),
    )
    rt.input_ctl = _RecInput()  # type: ignore[assignment]
    rt.trigger_recenter()

    # Must center onto secondary display (-1920 + 960 = -960, 540), matching main loop bounds
    assert (rt._screen_ox, rt._screen_oy, rt._screen_w, rt._screen_h) == (-1920.0, 0.0, 1920.0, 1080.0)
    assert moved_to == [(-960.0, 540.0)]


def test_update_config_does_not_reset_filter_state():
    sm = LandmarkSmoother(min_cutoff=2.0, beta=45.0, d_cutoff=1.0)
    h0 = _make_open_hand()
    h1 = [Landmark(p.x + 0.01, p.y + 0.01, p.z) for p in h0]
    sm.smooth(h0, "Right", 0.0)
    sm.smooth(h1, "Right", 33.0)
    prev_x_before = sm._filters["Right"][8][0]._x_prev
    prev_t_before = sm._filters["Right"][8][0]._t_prev_ms
    assert prev_x_before is not None and prev_t_before == 33.0

    # Changing min_cutoff / beta updates coefficients in place without clearing filter history
    sm.update_config(min_cutoff=1.5, beta=30.0)
    assert "Right" in sm._filters
    assert sm._filters["Right"][8][0]._x_prev == prev_x_before
    assert sm._filters["Right"][8][0]._t_prev_ms == 33.0
    assert sm._filters["Right"][8][0].min_cutoff == 1.5
    assert sm._filters["Right"][8][0].beta == 30.0

    # Changing d_cutoff structure resets filters as specified
    sm.update_config(min_cutoff=1.5, beta=30.0, d_cutoff=2.0)
    assert sm._filters == {}


def test_pending_config_applied_on_processing_thread():
    import threading

    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    rt = AeroHandRuntime(cfg, dry_run=True)

    # Simulate an active processing thread owning the engine
    rt._processing_thread_id = threading.get_ident()
    applied_threads: List[int] = []
    orig_apply = rt._apply_live_settings_now

    def _spy(c, *, from_profile=False):
        applied_threads.append(threading.get_ident())
        orig_apply(c, from_profile=from_profile)

    rt._apply_live_settings_now = _spy  # type: ignore[method-assign]

    # Call apply_live_settings from a DIFFERENT (UI/HTTP) thread -> must queue, NOT apply immediately
    ui_cfg = deep_merge(cfg, {"cursor": {"pointer_sensitivity": 3.4}})
    worker_tid: List[int] = []

    def _ui_thread():
        worker_tid.append(threading.get_ident())
        rt.apply_live_settings(ui_cfg)

    t = threading.Thread(target=_ui_thread)
    t.start()
    t.join(timeout=2.0)

    assert len(applied_threads) == 0
    assert rt._pending_cfg is not None
    assert rt.engine.config["cursor"]["pointer_sensitivity"] != 3.4

    # Processing thread flushes pending config at start of frame
    assert rt._flush_pending_config() is True
    assert applied_threads == [threading.get_ident()]
    assert applied_threads[0] != worker_tid[0]
    assert rt.engine.config["cursor"]["pointer_sensitivity"] == 3.4
    assert rt._flush_pending_config() is False


def test_active_drag_survives_gate_closing():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["gestures"]["latch_gate_while_pose_active"] = True

    class _ConstDragClassifier:
        def __init__(self) -> None:
            self.pose = PoseName.THUMB_INDEX_MIDDLE_PINCH

        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            if self.pose != PoseName.IDLE:
                probs[PoseName.IDLE] = 0.05
                probs[self.pose] = 0.95
            return PinchPrediction(self.pose, probs, "high")

        def reset(self):
            pass

    clf = _ConstDragClassifier()
    eng = HandCursorEngine(cfg, classifier=clf)

    # Point forward and hold 3-finger pinch long enough to start drag
    fwd = _make_open_hand(forward_z=-0.25)
    eng.process_frame([HandInput("Right", fwd, fwd)], 0.0, 1920.0, 1080.0)
    eng.process_frame([HandInput("Right", fwd, fwd)], 33.0, 1920.0, 1080.0)
    outs_hold = eng.process_frame([HandInput("Right", fwd, fwd)], 260.0, 1920.0, 1080.0)
    assert outs_hold[0].pose.is_grabbing is True

    # Hand tilts upward so dirZ crosses -0.11 (pointing gate closes) mid-drag
    tilted_away = _make_open_hand(forward_z=+0.05)
    for k in range(6):
        outs_tilted = eng.process_frame(
            [HandInput("Right", tilted_away, tilted_away)], 300.0 + k * 33.0, 1920.0, 1080.0
        )
        assert outs_tilted[0].cursor.is_pointing is False
        assert outs_tilted[0].pose.is_grabbing is True

    # When pinch actually releases -> drag ends cleanly even with gate closed
    clf.pose = PoseName.IDLE
    ended = False
    for k in range(5):
        outs_rel = eng.process_frame(
            [HandInput("Right", tilted_away, tilted_away)], 550.0 + k * 33.0, 1920.0, 1080.0
        )
        if outs_rel[0].pose.grab_just_ended:
            ended = True
    assert ended is True


def test_new_pose_blocked_when_gate_closed():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["gestures"]["latch_gate_while_pose_active"] = True

    class _AlwaysClickClassifier:
        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_MIDDLE_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_MIDDLE_PINCH, probs, "high")

        def reset(self):
            pass

    eng = HandCursorEngine(cfg, classifier=_AlwaysClickClassifier())
    away = _make_open_hand(forward_z=+0.10)
    for k in range(5):
        outs = eng.process_frame([HandInput("Right", away, away)], k * 33.0, 1920.0, 1080.0)
        assert outs[0].cursor.is_pointing is False
        assert outs[0].pose.active_pose is None
        assert outs[0].pose.is_pinching is False


def test_right_click_blocked_for_non_click_hand():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False
    cfg["hand"]["cursor_hand"] = "Left"
    cfg["hand"]["click_hand"] = "Right"

    class _PerHandPoseClassifier:
        def __init__(self) -> None:
            self.by_hand = {"Left": PoseName.THUMB_RING_PINCH, "Right": PoseName.IDLE}

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            pose = self.by_hand.get(handedness, PoseName.IDLE)
            probs = idle_probabilities()
            if pose != PoseName.IDLE:
                probs[PoseName.IDLE] = 0.05
                probs[pose] = 0.95
            return PinchPrediction(pose, probs, "high")

        def reset(self):
            pass

    clf = _PerHandPoseClassifier()
    eng = HandCursorEngine(cfg, classifier=clf)
    fwd = _make_open_hand(forward_z=-0.25)

    # Hold thumb_ring_pinch still on Left hand (non-click hand), then release
    eng.process_frame([HandInput("Left", fwd, fwd)], 0.0, 1920.0, 1080.0)
    eng.process_frame([HandInput("Left", fwd, fwd)], 33.0, 1920.0, 1080.0)
    eng.process_frame([HandInput("Left", fwd, fwd)], 260.0, 1920.0, 1080.0)
    clf.by_hand["Left"] = PoseName.IDLE
    right_clicked = False
    for k in range(6):
        outs = eng.process_frame([HandInput("Left", fwd, fwd)], 300.0 + k * 33.0, 1920.0, 1080.0)
        if outs[0].pose.is_right_clicking:
            right_clicked = True
    assert right_clicked is False

    # Now repeat on Right hand (click hand) -> MUST emit right_click
    clf.by_hand["Right"] = PoseName.THUMB_RING_PINCH
    eng.process_frame([HandInput("Right", fwd, fwd)], 600.0, 1920.0, 1080.0)
    eng.process_frame([HandInput("Right", fwd, fwd)], 633.0, 1920.0, 1080.0)
    eng.process_frame([HandInput("Right", fwd, fwd)], 860.0, 1920.0, 1080.0)
    clf.by_hand["Right"] = PoseName.IDLE
    right_clicked_r = False
    for k in range(6):
        outs = eng.process_frame([HandInput("Right", fwd, fwd)], 900.0 + k * 33.0, 1920.0, 1080.0)
        if outs[0].pose.is_right_clicking:
            right_clicked_r = True
    assert right_clicked_r is True


def test_drag_blocked_for_non_click_hand():
    from aerohand.ml.base import PinchPrediction
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False
    cfg["hand"]["cursor_hand"] = "Left"
    cfg["hand"]["click_hand"] = "Right"
    cfg["audio"]["enabled"] = False

    class _PerHandClf:
        def __init__(self) -> None:
            self.by_hand = {"Left": PoseName.THUMB_INDEX_MIDDLE_PINCH, "Right": PoseName.IDLE}

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            pose = self.by_hand.get(handedness, PoseName.IDLE)
            probs = idle_probabilities()
            if pose != PoseName.IDLE:
                probs[PoseName.IDLE] = 0.05
                probs[pose] = 0.95
            return PinchPrediction(pose, probs, "high")

        def reset(self):
            pass

    clf = _PerHandClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    fwd = _make_open_hand(forward_z=-0.25)

    # Left hand (non-click hand) holds drag pose -> must NOT start grab
    eng.process_frame([HandInput("Left", fwd, fwd)], 0.0, 1920.0, 1080.0)
    eng.process_frame([HandInput("Left", fwd, fwd)], 33.0, 1920.0, 1080.0)
    outs = eng.process_frame([HandInput("Left", fwd, fwd)], 260.0, 1920.0, 1080.0)
    assert outs[0].pose.is_grabbing is False
    assert outs[0].pose.grab_just_started is False

    # Split cursor=Left / click=Right runtime routing: Right click fires at Left hand's cursor position!
    rt = AeroHandRuntime(cfg, dry_run=False)
    clicks: List[tuple[float, float, str]] = []

    class _CapInput(_FakeInputCtl):
        def mouse_click(self, x: float, y: float, button: str = "left", click_count: int = 1) -> bool:
            clicks.append((x, y, button))
            return True

    rt.input_ctl = _CapInput()  # type: ignore[assignment]
    from aerohand.core.types import CursorOutput, HandOutput, PoseOutput

    left_out = HandOutput(
        hand="Left",
        cursor=CursorOutput(hand="Left", x=321.0, y=456.0, is_pointing=True, is_valid=True),
        pose=PoseOutput(hand="Left"),
    )
    right_out = HandOutput(
        hand="Right",
        cursor=CursorOutput(hand="Right", x=999.0, y=999.0, is_pointing=True, is_valid=True),
        pose=PoseOutput(hand="Right", is_clicking=True, click_count=1),
    )
    primary = rt._select_primary_hand([left_out, right_out])
    assert primary is not None and primary.hand == "Left"
    rt._dispatch_actions([left_out, right_out], primary, 1000.0)
    assert clicks == [(321.0, 456.0, "left")]


def test_crop_preserves_aspect_ratio():
    import cv2

    from aerohand.platform.camera import prepare_camera_frame

    # Draw an exact circle of radius 180 px at the center of a 1280x720 (16:9) frame
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(img, (640, 360), 180, (255, 255, 255), thickness=-1)

    for ar in ("4:3", "16:9", "1:1", "auto"):
        out, out_ar = prepare_camera_frame(img, target_width=640, target_height=480, aspect_ratio=ar, mirror=True)
        ys, xs = np.where(out[:, :, 0] > 128)
        bw = float(xs.max() - xs.min())
        bh = float(ys.max() - ys.min())
        ratio = bw / max(1.0, bh)
        assert abs(ratio - 1.0) < 0.01, f"Circle distorted under aspect_ratio={ar}: bw={bw}, bh={bh}"
        assert out_ar == pytest.approx(out.shape[1] / float(out.shape[0]))


def test_distance_isotropy_16x9():
    from aerohand.core.kinematics import isotropic_distance_2d

    w, h = 1280.0, 720.0
    ar = w / h
    # Two segments of identical pixel length (100 px): one horizontal, one vertical
    c = Landmark(0.5, 0.5, 0.0)
    horiz = Landmark(0.5 + 100.0 / w, 0.5, 0.0)
    vert = Landmark(0.5, 0.5 + 100.0 / h, 0.0)
    d_h = isotropic_distance_2d(c, horiz, frame_aspect=ar)
    d_v = isotropic_distance_2d(c, vert, frame_aspect=ar)
    assert d_h == pytest.approx(d_v, rel=1e-6)


def test_click_lookback_returns_pre_pinch_position():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["cursor"]["enable_filter"] = False
    cfg["cursor"]["click_lookback_ms"] = 90.0
    cfg["cursor"]["click_lookback_max_offset_px"] = 0.0
    cfg["cursor"]["click_dwell_lock_ms"] = 0.0
    cfg["palm_wheel"]["enabled"] = False

    class _SeqClf:
        def __init__(self) -> None:
            self.p = 0.0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.THUMB_MIDDLE_PINCH] = self.p
            probs[PoseName.IDLE] = max(0.0, 1.0 - self.p)
            return PinchPrediction(
                PoseName.THUMB_MIDDLE_PINCH if self.p >= 0.5 else PoseName.IDLE,
                probs,
                "high",
            )

        def reset(self):
            pass

    clf = _SeqClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    eng.anchor_cursor("Right", 500.0, 500.0)
    eng._hands["Right"].kinematics._skip_remaining = 0

    # Steady aim at t=0..90ms, then fingertip shifts during pinch at t=120..180ms
    base = _make_open_hand(forward_z=-0.25)
    eng.process_frame([HandInput("Right", base, base)], 0.0, 1920.0, 1080.0)
    out_30 = eng.process_frame([HandInput("Right", base, base)], 30.0, 1920.0, 1080.0)[0]
    pre_x, pre_y = out_30.cursor.x, out_30.cursor.y
    eng.process_frame([HandInput("Right", base, base)], 60.0, 1920.0, 1080.0)
    eng.process_frame([HandInput("Right", base, base)], 90.0, 1920.0, 1080.0)

    # Shift index tip during pinch closure (simulates accidental click drift)
    shifted = [Landmark(p.x + 0.015, p.y + 0.015, p.z) for p in base]
    clf.p = 0.92
    eng.process_frame([HandInput("Right", shifted, shifted)], 120.0, 1920.0, 1080.0)
    out_enter = eng.process_frame([HandInput("Right", shifted, shifted)], 150.0, 1920.0, 1080.0)[0]
    assert out_enter.cursor.x > pre_x + 20.0  # cursor visibly drifted on closure

    # Release -> tap click MUST anchor to ~90ms pre-pinch position (~t=60ms), NOT drifted cursor!
    clf.p = 0.05
    eng.process_frame([HandInput("Right", shifted, shifted)], 180.0, 1920.0, 1080.0)
    out_tap = eng.process_frame([HandInput("Right", shifted, shifted)], 210.0, 1920.0, 1080.0)[0]
    assert out_tap.pose.is_clicking is True
    assert out_tap.pose.click_x == pytest.approx(pre_x, abs=2.0)
    assert out_tap.pose.click_y == pytest.approx(pre_y, abs=2.0)


def test_adaptive_lerp_speed_monotonic():
    from aerohand.core.kinematics import adaptive_lerp_speed

    speeds = [0.0, 50.0, 150.0, 300.0, 600.0, 900.0, 1200.0, 2000.0]
    vals = [adaptive_lerp_speed(s) for s in speeds]
    assert vals[0] == 40.0
    assert vals[2] == 40.0
    assert vals[-2] == 70.0
    assert vals[-1] == 70.0
    for a, b in zip(vals, vals[1:]):
        assert b >= a

    # Also verify fast drag increases exit_frames from 4 to 6 in PoseStateMachine
    psm = PoseStateMachine(load_config()["pose_state_machine"])
    hi = idle_probabilities()
    hi[PoseName.IDLE] = 0.05
    hi[PoseName.THUMB_INDEX_MIDDLE_PINCH] = 0.95
    lo = idle_probabilities()
    psm.process(hi, 0.0)
    psm.process(hi, 33.0)
    psm.process(hi, 250.0)  # hold_start
    # 5 low-prob frames at high cursor speed (> 300 px/s) must NOT drop drag yet
    for k in range(5):
        evs = psm.process(lo, 283.0 + k * 33.0, cursor_speed_px_s=500.0)
        assert evs == []
    # 6th frame drops drag cleanly
    evs6 = psm.process(lo, 450.0, cursor_speed_px_s=500.0)
    assert [e.type for e in evs6] == ["hold_end"]


def test_override_save_is_debounced():
    pytest.importorskip("AppKit")
    import time

    from aerohand.ui.app import DebouncedConfigPersister, detect_hotkey_conflicts, parse_hotkey_spec

    writes: List[float] = []
    persister = DebouncedConfigPersister(
        lambda cfg: writes.append(float(cfg["cursor"]["pointer_sensitivity"])),
        delay_s=0.12,
    )
    for i in range(20):
        persister.schedule({"cursor": {"pointer_sensitivity": 1.0 + i * 0.1}})
        time.sleep(0.005)
    time.sleep(0.22)
    assert len(writes) == 1
    assert writes[0] == pytest.approx(2.9)

    # Also verify hotkey parsing & conflict detection
    assert parse_hotkey_spec("Cmd+Shift+A") == (frozenset({"cmd", "shift"}), "a")
    assert detect_hotkey_conflicts(
        {"toggle_pause": "cmd+shift+a", "recenter": "cmd+shift+a", "toggle_hud": "cmd+shift+h"}
    ) == ["toggle_pause", "recenter"]


def test_ui_tick_payload_excludes_static_i18n():
    pytest.importorskip("AppKit")
    from aerohand.i18n import get_lang_pack
    from aerohand.ui.app import OverlayPayloadBuilder

    builder = OverlayPayloadBuilder()
    ui_cfg = {"language": "vi", "show_click_feedback": True, "show_palm_wheel_overlay": True}
    pack_vi = get_lang_pack("vi")

    # First tick sends i18n + full ui
    p1 = builder.build({"click_event_seq": 0}, ui_cfg, "vi", pack_vi)
    assert "i18n" in p1 and "ui" in p1

    # Second tick with unchanged language & ui excludes static i18n and ui
    p2 = builder.build({"click_event_seq": 1}, ui_cfg, "vi", pack_vi)
    assert "i18n" not in p2
    assert "ui" not in p2
    assert "ui_delta" not in p2

    # Changing a single ui key sends only that key in ui_delta (still no static i18n)
    ui_cfg2 = dict(ui_cfg, show_click_feedback=False)
    p3 = builder.build({"click_event_seq": 1}, ui_cfg2, "vi", pack_vi)
    assert "i18n" not in p3
    assert p3.get("ui_delta") == {"show_click_feedback": False}

    # Changing language re-sends new i18n pack
    p4 = builder.build({"click_event_seq": 1}, dict(ui_cfg2, language="en"), "en", get_lang_pack("en"))
    assert "i18n" in p4 and p4["i18n"]["tabs"]["dashboard"] == "Dashboard"


def test_zoom_not_triggered_during_drag():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False
    cfg["hand"]["click_hand"] = "Both"

    class _DualHandClf:
        def __init__(self) -> None:
            self.left_pose = PoseName.THUMB_INDEX_MIDDLE_PINCH
            self.right_pose = PoseName.THUMB_INDEX_PINCH

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            pose = self.left_pose if handedness == "Left" else self.right_pose
            probs = idle_probabilities()
            if pose != PoseName.IDLE:
                probs[PoseName.IDLE] = 0.05
                probs[pose] = 0.95
            return PinchPrediction(pose, probs, "high")

        def reset(self):
            pass

    clf = _DualHandClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    left_h = _make_open_hand(forward_z=-0.25)
    right_h = [Landmark(p.x + 0.25, p.y, p.z) for p in left_h]

    # Left hand enters drag while right hand pinches -> zoom MUST NOT trigger
    for k in range(10):
        spread = [Landmark(p.x + 0.25 + k * 0.02, p.y, p.z) for p in left_h]
        eng.process_frame(
            [HandInput("Left", left_h, left_h), HandInput("Right", spread, spread)],
            k * 40.0,
            1920.0,
            1080.0,
        )
        assert eng.last_zoom.is_zooming is False
        assert eng.last_zoom.step == 0

    # When BOTH hands hold dedicated zoom pose (thumb_index_pinch) >= 120ms without drag -> zoom triggers
    eng.reset()
    clf.left_pose = PoseName.THUMB_INDEX_PINCH
    clf.right_pose = PoseName.THUMB_INDEX_PINCH
    eng.process_frame(
        [HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)], 0.0, 1920.0, 1080.0
    )
    assert eng.last_zoom.is_zooming is False  # < 120ms hold gate
    eng.process_frame(
        [HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)], 140.0, 1920.0, 1080.0
    )
    assert eng.last_zoom.is_zooming is True


def test_compute_tile_rect_halves():
    from aerohand.platform.input_macos import compute_tile_rect

    vis = (0.0, 28.0, 1440.0, 872.0)
    left = compute_tile_rect("left", vis)
    right = compute_tile_rect("right", vis)
    top = compute_tile_rect("top", vis)
    bottom = compute_tile_rect("bottom", vis)
    fill = compute_tile_rect("fill", vis)
    center = compute_tile_rect("center", vis)
    assert left and right and top and bottom and fill and center

    # Left/right are exact vertical halves
    assert left[2] == pytest.approx(right[2])
    assert left[3] == pytest.approx(right[3])
    assert left[0] + left[2] < right[0]

    # Top/bottom are exact horizontal halves (distinct from fill & center!)
    assert top[2] == pytest.approx(bottom[2])
    assert top[3] == pytest.approx(bottom[3])
    assert top[1] + top[3] < bottom[1]
    assert top[3] < fill[3] * 0.55

    # Fill covers full visible rect, center is 72% x 78% centered
    assert fill[2] == pytest.approx(1440.0 - 8.0)
    assert fill[3] == pytest.approx(872.0 - 8.0)
    assert center[2] == pytest.approx(1440.0 * 0.72)
    assert center[3] == pytest.approx(872.0 * 0.78)


def test_calibration_suggestion_within_bounds():
    from aerohand.core.engine import (
        SAFE_DUAL_PINCH_ON_BOUNDS,
        SAFE_ENTER_THRESHOLD_BOUNDS,
        SAFE_PINCH_ON_BOUNDS,
        SAFE_PINKY_BASE_ON_BOUNDS,
        suggest_calibrated_thresholds,
    )

    # Extreme too-small and too-large measurements must still clamp cleanly within safe bounds
    for raw_val in (0.01, 0.25, 0.95):
        sugg = suggest_calibrated_thresholds(
            {
                "thumb_index_pinch": raw_val,
                "thumb_middle_pinch": raw_val,
                "thumb_ring_pinch": raw_val,
                "thumb_pinky_pinch": raw_val,
                "thumb_pinky_base": raw_val,
                "thumb_index_middle_pinch": raw_val,
            }
        )
        assert SAFE_PINCH_ON_BOUNDS[0] <= sugg["pinch_on_dist"] <= SAFE_PINCH_ON_BOUNDS[1]
        assert SAFE_PINKY_BASE_ON_BOUNDS[0] <= sugg["pinky_base_on_dist"] <= SAFE_PINKY_BASE_ON_BOUNDS[1]
        assert SAFE_DUAL_PINCH_ON_BOUNDS[0] <= sugg["dual_pinch_on_dist"] <= SAFE_DUAL_PINCH_ON_BOUNDS[1]
        for th in sugg["enter_thresholds"].values():
            assert SAFE_ENTER_THRESHOLD_BOUNDS[0] <= th <= SAFE_ENTER_THRESHOLD_BOUNDS[1]


def test_toast_uses_live_gesture_map():
    from aerohand.config import apply_gesture_preset
    from aerohand.core.engine import format_live_hint_toast

    cfg_ap = load_config()
    msg_vi_ap = format_live_hint_toast("click", cfg_ap, lang="vi")
    msg_en_ap = format_live_hint_toast("click", cfg_ap, lang="en")
    assert msg_vi_ap and "Cái + Giữa" in msg_vi_ap
    assert msg_en_ap and "Thumb + Middle" in msg_en_ap

    # Switch to Classic preset -> toast dynamically updates to Thumb + Index (no hard-coded gesture names!)
    cfg_cl = apply_gesture_preset(cfg_ap, "classic")
    msg_vi_cl = format_live_hint_toast("click", cfg_cl, lang="vi")
    msg_en_cl = format_live_hint_toast("click", cfg_cl, lang="en")
    assert msg_vi_cl and "Cái + Trỏ" in msg_vi_cl
    assert msg_en_cl and "Thumb + Index" in msg_en_cl


def test_pasteboard_snapshot_restores_non_string_types():
    from aerohand.platform.input_macos import (
        clipboard_restore_delay_s,
        restore_pasteboard,
        snapshot_pasteboard,
    )

    class _FakePBItem:
        def __init__(self, data_map: dict[str, bytes]) -> None:
            self._map = dict(data_map)

        def types(self):
            return list(self._map.keys())

        def dataForType_(self, ptype: str):
            return self._map.get(ptype)

    class _FakePasteboard:
        def __init__(self) -> None:
            self.items: List[_FakePBItem] = []

        def pasteboardItems(self):
            return list(self.items)

        def types(self):
            return self.items[0].types() if self.items else []

        def clearContents(self) -> None:
            self.items = []

        def setString_forType_(self, text: str, ptype: str) -> None:
            self.items = [_FakePBItem({ptype: text.encode("utf-8")})]

        def stringForType_(self, ptype: str) -> str | None:
            if not self.items:
                return None
            raw = self.items[0].dataForType_(ptype)
            return raw.decode("utf-8") if raw is not None else None

        def restore_snapshot_items(self, snap: List[dict[str, bytes]]) -> None:
            self.items = [_FakePBItem(m) for m in snap]

    png_bytes = b"\x89PNG\r\n\x1a\nFAKE_IMAGE_BYTES"
    file_url_bytes = b"file:///Users/macos/Desktop/demo.png"
    pb = _FakePasteboard()
    pb.items = [
        _FakePBItem({"public.png": png_bytes, "public.utf8-plain-text": b"caption"}),
        _FakePBItem({"public.file-url": file_url_bytes}),
    ]

    snap = snapshot_pasteboard(pb)
    assert len(snap) == 2 and snap[0]["public.png"] == png_bytes

    ctl = MacOSInputController(dry_run=True)
    ok = ctl.paste_text_utf8("Xin chào tiếng Việt!", pasteboard=pb, restore_delay_s=0.0, sync_restore=True)
    assert ok is True
    # Original image & file URL items must be completely preserved!
    assert len(pb.items) == 2
    assert pb.items[0].dataForType_("public.png") == png_bytes
    assert pb.items[0].dataForType_("public.utf8-plain-text") == b"caption"
    assert pb.items[1].dataForType_("public.file-url") == file_url_bytes

    # Also test direct restore_pasteboard + Electron/Chromium restore delay
    pb.clearContents()
    restore_pasteboard(pb, snap)
    assert pb.items[0].dataForType_("public.png") == png_bytes
    assert clipboard_restore_delay_s("com.microsoft.VSCode") == 0.35
    assert clipboard_restore_delay_s("com.google.Chrome") == 0.35
    assert clipboard_restore_delay_s("com.apple.Notes") == 0.15


def _load_train_tcn_module(name: str = "train_tcn_p2"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent.parent / "scripts" / "train_tcn.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_split_requires_all_classes_in_val():
    mod = _load_train_tcn_module("train_tcn_split")
    h = _make_open_hand()

    def _mk_bundle(name: str, group_key: str, class_ids: List[int]):
        wins = []
        for cid in class_ids:
            wins.append([mod.ParsedSample(cid, h, h, "Right", "Right", float(i * 33)) for i in range(5)])
        return mod.SessionBundle(name=name, group_key=group_key, windows=wins, labels=np.array(class_ids, dtype=np.int64))

    # Case 1: Only 2 classes across bundles -> missing_val reports the 5 missing classes
    b_partial_1 = _mk_bundle("s1.json", "minh::bright", [0, 1])
    b_partial_2 = _mk_bundle("s2.json", "minh::dim", [0, 1])
    _tr, _va, missing = mod.split_session_groups([b_partial_1, b_partial_2])
    assert len(missing) == 5

    # Case 2: Multiple condition groups covering all 7 classes -> val split covers all 7 classes with 0 leakage
    all_7 = list(range(7))
    bundles = [
        _mk_bundle("s1.json", "minh::0deg_bright", all_7),
        _mk_bundle("s2.json", "minh::30deg_bright", all_7),
        _mk_bundle("s3.json", "minh::45deg_dim", all_7),
        _mk_bundle("s4.json", "minh::60deg_backlit", all_7),
    ]
    tr, va, missing2 = mod.split_session_groups(bundles, val_ratio=0.25)
    assert missing2 == []
    assert set(b.group_key for b in tr).isdisjoint(set(b.group_key for b in va))


def test_sequences_break_on_time_gap():
    mod = _load_train_tcn_module("train_tcn_gap")
    h = _make_open_hand()
    flat = [c for lm in h for c in (lm.x, lm.y, lm.z)]

    # 6 contiguous frames (t=0..165ms) + 250ms gap (>100ms) + 6 contiguous frames (t=415..580ms)
    samples = []
    for i in range(6):
        samples.append(
            {"timestamp_ms": i * 33, "label_name": "idle", "handedness": "Right", "landmarks_2d": flat, "world_landmarks": flat}
        )
    for i in range(6):
        samples.append(
            {"timestamp_ms": 415 + i * 33, "label_name": "thumb_middle_pinch", "handedness": "Right", "landmarks_2d": flat, "world_landmarks": flat}
        )

    xs, ys = mod.samples_to_sequences(samples, feature_set="invariant348", seq_len=5, max_gap_ms=100.0)
    # Without breaking, 12 frames would produce 8 windows; breaking across the 250ms gap gives (6-5+1)+(6-5+1) = 4 windows!
    assert xs.shape == (4, 5, FEATURE_DIM_3D)
    assert list(ys) == [0, 0, 2, 2]


def test_augmentation_preserves_label_and_shape():
    mod = _load_train_tcn_module("train_tcn_aug")
    h = _make_open_hand()
    win = [mod.ParsedSample(2, h, h, "Right", "Right", float(i * 33)) for i in range(5)]
    rng = np.random.default_rng(123)

    aug = mod.augment_window(win, rng)
    assert len(aug) == 5
    assert aug[-1].label_idx == 2
    assert all(len(s.lms2d) == 21 and s.lms3d is not None and len(s.lms3d) == 21 for s in aug)

    xs, ys = mod._featurize_windows([aug], feature_set="invariant348", seq_len=5)
    assert xs.shape == (1, 5, FEATURE_DIM_3D)
    assert np.all(np.isfinite(xs))
    assert ys[0] == 2


def test_temperature_loaded_by_onnx_classifier(tmp_path: Path, monkeypatch):
    import json

    import onnxruntime as ort

    from aerohand.ml.onnx_classifier import ONNXPinchClassifier

    norm_fp = tmp_path / "norm.json"
    norm_fp.write_text(
        json.dumps(
            {
                "feature_set": "invariant348",
                "sequence_length": 1,
                "mean": [0.0] * FEATURE_DIM_3D,
                "scale": [1.0] * FEATURE_DIM_3D,
                "temperature": 2.0,
            }
        ),
        encoding="utf-8",
    )
    model_fp = tmp_path / "dummy.onnx"
    model_fp.write_bytes(b"placeholder")

    class _FakeInput:
        name = "input"

    class _FakeOrtSession:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def get_inputs(self):
            return [_FakeInput()]

        def run(self, _out, _feed):
            return [np.array([[0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _FakeOrtSession)

    clf = ONNXPinchClassifier(model_path=model_fp, normalizer_path=norm_fp, sequence_length=1)
    assert clf._temperature == pytest.approx(2.0)

    h = _make_open_hand()
    pred_t2 = clf.predict(h, h, "Right", 0.0)
    # With T=2.0, scaled logit is 2.0 -> softmax prob = exp(2) / (exp(2) + 6)
    expected_p = math.exp(2.0) / (math.exp(2.0) + 6.0)
    assert pred_t2.pose == PoseName.THUMB_MIDDLE_PINCH
    assert pred_t2.probabilities[PoseName.THUMB_MIDDLE_PINCH] == pytest.approx(expected_p, abs=1e-5)

    # Compare against T=1.0 -> sharper probability exp(4)/(exp(4)+6) > T=2.0
    clf._temperature = 1.0
    pred_t1 = clf.predict(h, h, "Right", 33.0)
    assert pred_t1.probabilities[PoseName.THUMB_MIDDLE_PINCH] > pred_t2.probabilities[PoseName.THUMB_MIDDLE_PINCH]


def test_collector_trims_edges():
    from aerohand.ml.collector import GuidedStep, PinchDataCollector

    col = PinchDataCollector(trim_edge_ms=300.0, countdown_s=1.0, step_duration_s=2.0)
    h = _make_open_hand()
    col.start_guided([GuidedStep(PoseName.THUMB_MIDDLE_PINCH, duration_s=2.0)], now_ms=0.0)
    # Countdown ends at t=1000ms
    col.tick_guided(now_ms=1000.0)
    assert col.is_recording is True
    # Feed frames every 100ms from 1000..3000ms (21 frames covering 2000ms)
    for ts in range(1000, 3001, 100):
        col.add_frame(h, h, "Right", 0.95, ts)
    # Complete step at t=3000ms -> first 300ms (1000..1200) & last 300ms (2800..3000) MUST be trimmed!
    col.tick_guided(now_ms=3000.0)
    assert col.is_recording is False
    kept_ts = [s["timestamp_ms"] for s in col._samples]
    assert min(kept_ts) >= 1300
    assert max(kept_ts) <= 2700
    assert len(kept_ts) == 15

    # Undo last 0.5s removes samples >= 2200ms
    removed = col.undo_last(seconds=0.5)
    assert removed > 0
    assert all(s["timestamp_ms"] < 2200 for s in col._samples)


def test_collector_records_primary_hand():
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["hand"]["cursor_hand"] = "Right"
    cfg["collector"]["record_hand"] = "primary"
    cfg["audio"]["enabled"] = False
    rt = AeroHandRuntime(cfg, dry_run=True)
    rt.collector.set_pose(PoseName.THUMB_MIDDLE_PINCH)
    rt.collector.set_recording(True)

    h = _make_open_hand()
    # hands[0] is Left, hands[1] is Right; primary is Right -> MUST record Right, NOT hands[0]!
    hands = [HandInput("Left", h, h, confidence=0.9), HandInput("Right", h, h, confidence=0.95)]
    primary = _make_hand_output("Right")
    rt._record_collector_frame(hands, primary, 1000)
    assert len(rt.collector._samples) == 1
    assert rt.collector._samples[0]["handedness"] == "Right"
    assert rt.collector._samples[0]["hand_id"] == "Right"


def test_collector_metadata_in_header(tmp_path: Path):
    import json

    col = PinchDataCollector(
        session_meta={
            "subject": "minh",
            "hand": "Right",
            "wrist_angle": "45",
            "distance": "mid",
            "lighting": "dim",
            "camera": "FaceTimeHD",
        }
    )
    col.set_pose(PoseName.THUMB_INDEX_PINCH)
    col.toggle_recording()
    h = _make_open_hand()
    col.add_frame(h, h, "Right", 0.92, 1000)
    fp = col.save(tmp_path)

    assert "minh" in fp.name and "45deg" in fp.name and "dim" in fp.name
    data = json.loads(fp.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["session_meta"]["subject"] == "minh"
    assert data["session_meta"]["wrist_angle"] == "45"
    assert data["session_meta"]["lighting"] == "dim"


def test_quality_gate_flags_low_confidence():
    col = PinchDataCollector()
    col.set_pose(PoseName.THUMB_RING_PINCH)
    col.toggle_recording()
    h = _make_open_hand()
    # Low confidence (< 0.58), clipped at edge (x=0.01), and a 250ms gap (> 100ms)
    clipped = [Landmark(0.01 if i == 0 else p.x, p.y, p.z) for i, p in enumerate(h)]
    col.add_frame(clipped, h, "Right", 0.42, 1000)
    col.add_frame(h, h, "Right", 0.50, 1250)
    q = col.evaluate_quality()
    assert q["ok"] is False
    assert "low_confidence" in q["warnings"]
    assert "hand_clipped_edge" in q["warnings"]
    assert "time_gaps" in q["warnings"]


def test_onnx_error_falls_back_to_heuristic():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    now_s = [10.0]
    releases: List[int] = []

    class _FlakyOnnx:
        def __init__(self) -> None:
            self.fail = True
            self.pose = PoseName.THUMB_INDEX_PINCH

        def predict(self, *args, **kwargs):
            if self.fail:
                raise RuntimeError("onnx crash")
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.1
            probs[self.pose] = 0.9
            return PinchPrediction(self.pose, probs, "high")

        def reset(self, hand=None):
            pass

    fallback = Heuristic3DPinchClassifier()
    onnx = _FlakyOnnx()
    guarded = GuardedClassifier(
        onnx,
        fallback,
        shadow_mode=False,
        max_p95_ms=8.0,
        max_errors=3,
        error_window_s=10.0,
        on_error=lambda: releases.append(1),
        clock_s=lambda: now_s[0],
    )

    pinched = list(_make_open_hand())
    m_tip = pinched[12]
    pinched[4] = Landmark(m_tip.x - 0.005, m_tip.y, m_tip.z)

    # First 3 errors within 10s: each fails open to IDLE and triggers release_all callback; 3rd switches to heuristic
    for i in range(3):
        now_s[0] = 10.0 + i * 1.0
        pred = guarded.predict(pinched, pinched, "Right", float(i * 33))
        assert pred.pose == PoseName.IDLE
    assert len(releases) == 3
    assert guarded.fell_back is True
    assert guarded.fallback_reason == "onnx_fallback_errors"

    # Next frames use Heuristic3D seamlessly (open -> pinch approach) and detect the middle pinch
    open_h = _make_open_hand()
    guarded.predict(open_h, open_h, "Right", 100.0)
    pred_fb = guarded.predict(pinched, pinched, "Right", 133.0)
    assert pred_fb.pose == PoseName.THUMB_MIDDLE_PINCH

    # Shadow mode logs disagreement rate without acting on ONNX
    onnx2 = _FlakyOnnx()
    onnx2.fail = False
    onnx2.pose = PoseName.THUMB_PINKY_PINCH  # disagrees with heuristic (THUMB_MIDDLE_PINCH)
    shadow = GuardedClassifier(onnx2, fallback, shadow_mode=True)
    pred_sh = shadow.predict(pinched, pinched, "Right", 200.0)
    assert pred_sh.pose == PoseName.THUMB_MIDDLE_PINCH  # acts on heuristic, NOT ONNX
    tel = shadow.telemetry()
    assert tel["shadow_mode"] is True
    assert tel["shadow_frames"] == 1
    assert tel["shadow_disagreement_rate"] == 1.0


def test_buffers_cleared_on_hand_loss(tmp_path: Path, monkeypatch):
    import json

    import onnxruntime as ort

    from aerohand.ml.onnx_classifier import ONNXPinchClassifier

    norm_fp = tmp_path / "norm_buf.json"
    norm_fp.write_text(
        json.dumps(
            {
                "feature_set": "invariant348",
                "sequence_length": 5,
                "mean": [0.0] * FEATURE_DIM_3D,
                "scale": [1.0] * FEATURE_DIM_3D,
            }
        ),
        encoding="utf-8",
    )
    model_fp = tmp_path / "dummy_buf.onnx"
    model_fp.write_bytes(b"placeholder")

    class _FakeInput:
        name = "input"

    class _FakeOrt:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def get_inputs(self):
            return [_FakeInput()]

        def run(self, _out, _feed):
            return [np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _FakeOrt)

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False

    clf = ONNXPinchClassifier(model_path=model_fp, normalizer_path=norm_fp, sequence_length=5)
    eng = HandCursorEngine(cfg, classifier=clf)
    fwd = _make_open_hand(forward_z=-0.25)

    # Feed 3 frames with both Left & Right hands -> both buffers have 3 entries
    for k in range(3):
        eng.process_frame(
            [HandInput("Left", fwd, fwd), HandInput("Right", fwd, fwd)], k * 33.0, 1920.0, 1080.0
        )
    assert len(clf._buffers["Right"]) == 3
    assert len(clf._buffers["Left"]) == 3

    # Right hand vanishes on next frame while Left remains -> Right buffer cleared, Left keeps accumulating!
    eng.process_frame([HandInput("Left", fwd, fwd)], 100.0, 1920.0, 1080.0)
    assert len(clf._buffers["Right"]) == 0
    assert len(clf._buffers["Left"]) == 4

    # Switching classifier clears all remaining buffers cleanly
    eng.set_classifier(Heuristic3DPinchClassifier())
    assert len(clf._buffers["Left"]) == 0


def test_overlay_hydration_reset_and_screen_signature():
    pytest.importorskip("AppKit")
    from aerohand.i18n import get_lang_pack
    from aerohand.ui.app import OverlayPayloadBuilder, screen_frames_signature

    builder = OverlayPayloadBuilder()
    ui_cfg = {"language": "vi", "show_palm_wheel_overlay": True, "palm_wheel_fixed": False}
    pack_vi = get_lang_pack("vi")

    # Initial sync delivers full ui + i18n; subsequent tick sends delta only
    p1 = builder.build({"palm_wheels": []}, ui_cfg, "vi", pack_vi)
    assert p1["ui"]["show_palm_wheel_overlay"] is True and "i18n" in p1
    p2 = builder.build({"palm_wheels": [{"hand": "Right", "state": "active"}]}, ui_cfg, "vi", pack_vi)
    assert "ui" not in p2 and "i18n" not in p2

    # When overlay_ready / overlay_need_sync triggers builder.reset(), next tick MUST re-deliver full ui + i18n
    builder.reset()
    p3 = builder.build({"palm_wheels": [{"hand": "Right", "state": "active"}]}, ui_cfg, "vi", pack_vi)
    assert p3["ui"]["show_palm_wheel_overlay"] is True and "i18n" in p3

    # Identical screen geometry (modulo sub-pixel Dock/menu bar noise) keeps the same signature
    s1 = screen_frames_signature([(0.0, 0.0, 1512.0, 982.0)])
    s2 = screen_frames_signature([(0.0, 0.0, 1512.1, 982.0)])
    s3 = screen_frames_signature([(0.0, 0.0, 1512.0, 982.0), (-1920.0, 0.0, 1920.0, 1080.0)])
    assert s1 == s2
    assert s1 != s3


def test_zoom_two_hands_never_starts_dictation():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False

    class _BothIndexPinch:
        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_INDEX_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_INDEX_PINCH, probs, "high")

        def reset(self, hand=None):
            pass

    eng = HandCursorEngine(cfg, classifier=_BothIndexPinch())
    left_h = _make_open_hand(forward_z=-0.25)
    right_h = [Landmark(p.x + 0.25, p.y, p.z) for p in left_h]

    # Hold both hands in thumb_index_pinch for 600 ms -> Zoom engages, Dictation NEVER starts
    dict_starts = 0
    for k in range(18):
        outs = eng.process_frame(
            [HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)],
            k * 35.0,
            1920.0,
            1080.0,
        )
        for o in outs:
            if o.pose.dictation_just_started or o.pose.is_dictating:
                dict_starts += 1
    assert dict_starts == 0
    assert eng.last_zoom.is_zooming is True


def test_dictation_cancelled_when_second_hand_pinches_within_arbitration_window():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False
    cfg["gestures"]["dictation_arm_ms"] = 350

    class _StaggeredZoom:
        def __init__(self) -> None:
            self.left_active = False

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            active = (handedness == "Right") or self.left_active
            if active:
                probs[PoseName.IDLE] = 0.05
                probs[PoseName.THUMB_INDEX_PINCH] = 0.95
                return PinchPrediction(PoseName.THUMB_INDEX_PINCH, probs, "high")
            return PinchPrediction(PoseName.IDLE, probs, "high")

        def reset(self, hand=None):
            pass

    clf = _StaggeredZoom()
    eng = HandCursorEngine(cfg, classifier=clf)
    left_h = _make_open_hand(forward_z=-0.25)
    right_h = [Landmark(p.x + 0.25, p.y, p.z) for p in left_h]

    # Right hand pinches alone and arms dictation after 350ms
    started = False
    for ts in [0.0, 33.0, 66.0, 200.0, 380.0, 420.0]:
        outs = eng.process_frame(
            [HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)],
            ts,
            1920.0,
            1080.0,
        )
        r_out = next(o for o in outs if o.hand == "Right")
        if r_out.pose.dictation_just_started:
            started = True
    assert started is True

    # Left hand now enters zoom pose -> dictation MUST be cancelled (NOT ended for transcription!)
    clf.left_active = True
    outs_cancel = eng.process_frame(
        [HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)],
        460.0,
        1920.0,
        1080.0,
    )
    r_cancel = next(o for o in outs_cancel if o.hand == "Right")
    assert r_cancel.pose.dictation_cancelled is True
    assert r_cancel.pose.dictation_just_ended is False
    assert r_cancel.pose.is_dictating is False


def test_single_hand_dictation_requires_arm_ms():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False
    cfg["gestures"]["dictation_arm_ms"] = 350

    class _IndexPinch:
        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_INDEX_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_INDEX_PINCH, probs, "high")

        def reset(self, hand=None):
            pass

    eng = HandCursorEngine(cfg, classifier=_IndexPinch())
    fwd = _make_open_hand(forward_z=-0.25)

    # Pose enters on 3rd frame (enter_frames=3 -> t=66ms). At 260ms (< 66+350=416ms), NOT armed yet!
    for ts in [0.0, 33.0, 66.0, 260.0, 390.0]:
        o = eng.process_frame([HandInput("Right", fwd, fwd)], ts, 1920.0, 1080.0)[0]
        assert o.pose.dictation_just_started is False
        assert o.pose.is_dictating is False

    # At t=420ms (>= 66+350ms) -> arms dictation cleanly once!
    o_arm = eng.process_frame([HandInput("Right", fwd, fwd)], 420.0, 1920.0, 1080.0)[0]
    assert o_arm.pose.dictation_just_started is True
    assert o_arm.pose.is_dictating is True
    # Subsequent frame keeps is_dictating=True without re-firing dictation_just_started
    o_next = eng.process_frame([HandInput("Right", fwd, fwd)], 460.0, 1920.0, 1080.0)[0]
    assert o_next.pose.dictation_just_started is False
    assert o_next.pose.is_dictating is True


def test_dictation_cancel_discards_audio():
    from aerohand.platform.dictation import VoiceDictationController

    pasted: List[str] = []
    ctrl = VoiceDictationController({"enabled": True}, on_text_ready=lambda t: pasted.append(t))
    # Simulate active recording with loud audio chunks
    ctrl._recording = True
    ctrl._chunks = [np.ones(16000, dtype=np.float32) * 0.2]
    assert ctrl.cancel(reason="two_hand_zoom") is True
    assert ctrl.is_recording is False
    assert ctrl._chunks == []
    assert pasted == []


def test_dictation_discards_short_recording(monkeypatch):
    from aerohand.platform import dictation as dict_mod
    from aerohand.platform.dictation import VoiceDictationController

    called = {"recognize": 0}
    pasted: List[str] = []
    monkeypatch.setattr(
        dict_mod, "_transcribe_apple_speech", lambda *_a, **_kw: (called.__setitem__("recognize", called["recognize"] + 1) or "hello")
    )
    monkeypatch.setattr(
        dict_mod, "_transcribe_mlx_whisper", lambda *_a, **_kw: (called.__setitem__("recognize", called["recognize"] + 1) or "hello")
    )

    ctrl = VoiceDictationController(
        {"enabled": True, "min_duration_s": 0.6, "min_rms": 0.01, "min_voiced_ms": 250},
        on_text_ready=lambda t: pasted.append(t),
    )
    # 0.30s loud audio -> shorter than 0.6s -> MUST discard without calling recognizer or pasting
    ctrl._recording = True
    ctrl._chunks = [np.ones(int(16000 * 0.30), dtype=np.float32) * 0.25]
    ctrl.stop_and_transcribe(sync=True)

    assert called["recognize"] == 0
    assert pasted == []
    st = ctrl.status_dict()
    assert st["last_discard_reason"] == "too_short"
    assert st["stats"]["discarded"] == 1
    assert st["discard_active"] is True


def test_dictation_discards_silent_recording(monkeypatch):
    from aerohand.platform import dictation as dict_mod
    from aerohand.platform.dictation import VoiceDictationController

    called = {"recognize": 0}
    pasted: List[str] = []
    monkeypatch.setattr(
        dict_mod, "_transcribe_apple_speech", lambda *_a, **_kw: (called.__setitem__("recognize", called["recognize"] + 1) or "hello")
    )

    ctrl = VoiceDictationController(
        {"enabled": True, "min_duration_s": 0.6, "min_rms": 0.01, "min_voiced_ms": 250},
        on_text_ready=lambda t: pasted.append(t),
    )
    # 1.2s near-silence (rms=0.001 < 0.01) -> MUST discard as silent
    ctrl._recording = True
    ctrl._chunks = [np.ones(int(16000 * 1.2), dtype=np.float32) * 0.001]
    ctrl.stop_and_transcribe(sync=True)

    assert called["recognize"] == 0
    assert pasted == []
    assert ctrl.status_dict()["last_discard_reason"] == "silent"

    # Also test brief single click spike (100ms loud, 1.1s silent -> voiced_ms=100 < 250) -> MUST discard as silent
    spike = np.zeros(int(16000 * 1.2), dtype=np.float32)
    spike[: int(16000 * 0.10)] = 0.2
    ctrl._recording = True
    ctrl._chunks = [spike]
    ctrl.stop_and_transcribe(sync=True)
    assert called["recognize"] == 0
    assert ctrl.status_dict()["last_discard_reason"] == "silent"


def test_dictation_blocklist_drops_hallucination(monkeypatch):
    from aerohand.platform import dictation as dict_mod
    from aerohand.platform.dictation import VoiceDictationController

    pasted: List[str] = []
    monkeypatch.setattr(
        dict_mod,
        "_transcribe_mlx_whisper",
        lambda *_a, **_kw: "  Cảm ơn các bạn đã theo dõi và hẹn gặp lại!  ",
    )

    ctrl = VoiceDictationController(
        {
            "enabled": True,
            "engine": "mlx_whisper",
            "language": "vi-VN",
            "min_duration_s": 0.6,
            "min_rms": 0.01,
            "min_voiced_ms": 250,
        },
        on_text_ready=lambda t: pasted.append(t),
    )
    # 1.0s voiced audio passes VAD, then recognizer returns known Whisper silence hallucination -> dropped!
    t = np.linspace(0.0, 1.0, 16000, dtype=np.float32)
    voiced = 0.15 * np.sin(2.0 * math.pi * 440.0 * t)
    ctrl._recording = True
    ctrl._chunks = [voiced]
    ctrl.stop_and_transcribe(sync=True)

    assert pasted == []
    st = ctrl.status_dict()
    assert st["last_discard_reason"] == "hallucination"
    assert st["stats"]["committed"] == 0

    # Genuine Vietnamese phrase passes blocklist and pastes cleanly
    monkeypatch.setattr(dict_mod, "_transcribe_mlx_whisper", lambda *_a, **_kw: "Xin chào AeroHand")
    ctrl._recording = True
    ctrl._chunks = [voiced]
    ctrl.stop_and_transcribe(sync=True)
    assert pasted == ["Xin chào AeroHand"]
    assert ctrl.status_dict()["stats"]["committed"] == 1


def test_dictation_event_ring_buffer_capped_at_50():
    from aerohand.platform.dictation import VoiceDictationController

    ctrl = VoiceDictationController({"enabled": True, "min_duration_s": 0.6})
    for i in range(75):
        ctrl._recording = True
        ctrl._chunks = [np.zeros(1600, dtype=np.float32)]  # 0.1s -> discarded
        ctrl.stop_and_transcribe(sync=True)

    evs = ctrl.recent_events()
    assert len(evs) == 50
    assert ctrl.status_dict()["stats"]["discarded"] == 75


def test_dwell_lock_does_not_rearm_while_probability_hovers():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["cursor"]["enable_filter"] = False
    cfg["cursor"]["click_dwell_lock_ms"] = 120.0
    cfg["cursor"]["click_dwell_lock_cooldown_ms"] = 400.0
    cfg["palm_wheel"]["enabled"] = False

    class _HoverClf:
        def __init__(self) -> None:
            self.p = 0.0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.THUMB_MIDDLE_PINCH] = self.p
            probs[PoseName.IDLE] = max(0.0, 1.0 - self.p)
            return PinchPrediction(PoseName.IDLE, probs, "high")

        def reset(self, hand=None):
            pass

    clf = _HoverClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    eng.anchor_cursor("Right", 500.0, 500.0)

    # Frame 0: clean idle (< 0.20)
    clf.p = 0.05
    h = _make_open_hand(forward_z=-0.25)
    eng.process_frame([HandInput("Right", h, h)], 0.0, 1920.0, 1080.0)

    # Oscillate between 0.42 and 0.68 for 2000ms (never dropping below 0.20) while hand moves smoothly
    pattern = [0.45, 0.62, 0.42, 0.65, 0.44, 0.68, 0.41, 0.66]
    xs = []
    for i in range(40):
        ts = 50.0 + i * 50.0
        clf.p = pattern[i % len(pattern)]
        shifted = [Landmark(p.x + 0.002 * i, p.y, p.z) for p in h]
        o = eng.process_frame([HandInput("Right", shifted, shifted)], ts, 1920.0, 1080.0)[0]
        xs.append(o.cursor.x)

    # Lock must activate AT MOST ONCE and never re-arm during continuous hovering
    assert eng.dwell_lock_activations == 1
    assert eng.dwell_unlock_without_click == 1
    # Pointer continues moving freely after the initial 120ms lock expires
    assert xs[-1] > xs[6] + 40.0


def test_dwell_lock_arms_once_per_pinch_attempt():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["cursor"]["enable_filter"] = False
    cfg["cursor"]["click_dwell_lock_ms"] = 120.0
    cfg["cursor"]["click_dwell_lock_cooldown_ms"] = 400.0
    cfg["palm_wheel"]["enabled"] = False

    class _RampClf:
        def __init__(self) -> None:
            self.p = 0.0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.THUMB_MIDDLE_PINCH] = self.p
            probs[PoseName.IDLE] = max(0.0, 1.0 - self.p)
            return PinchPrediction(
                PoseName.THUMB_MIDDLE_PINCH if self.p >= 0.75 else PoseName.IDLE, probs, "high"
            )

        def reset(self, hand=None):
            pass

    clf = _RampClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    eng.anchor_cursor("Right", 600.0, 400.0)
    h = _make_open_hand(forward_z=-0.25)

    # Idle -> rising pinch -> tap click -> return to idle
    seq = [
        (0.0, 0.05, 0.0),
        (33.0, 0.50, 0.005),   # arms lock once
        (66.0, 0.85, 0.010),   # enters pose (locked)
        (99.0, 0.90, 0.015),   # active pose (locked)
        (132.0, 0.10, 0.020),  # release 1
        (165.0, 0.05, 0.025),  # release 2 -> tap fires
    ]
    clicked = False
    for ts, prob, dx in seq:
        clf.p = prob
        shifted = [Landmark(p.x + dx, p.y, p.z) for p in h]
        o = eng.process_frame([HandInput("Right", shifted, shifted)], ts, 1920.0, 1080.0)[0]
        if o.pose.is_clicking:
            clicked = True
    assert clicked is True
    assert eng.dwell_lock_activations == 1


def test_dwell_lock_cooldown_enforced():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["cursor"]["enable_filter"] = False
    cfg["cursor"]["click_dwell_lock_ms"] = 120.0
    cfg["cursor"]["click_dwell_lock_cooldown_ms"] = 400.0
    cfg["palm_wheel"]["enabled"] = False

    class _StepClf:
        def __init__(self) -> None:
            self.p = 0.0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.THUMB_MIDDLE_PINCH] = self.p
            probs[PoseName.IDLE] = max(0.0, 1.0 - self.p)
            return PinchPrediction(PoseName.IDLE, probs, "high")

        def reset(self, hand=None):
            pass

    clf = _StepClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    h = _make_open_hand(forward_z=-0.25)

    # Attempt 1: arm at t=100ms, let it expire at t=220ms -> cooldown active until 620ms
    for ts, prob in [(0.0, 0.05), (100.0, 0.55), (230.0, 0.50)]:
        clf.p = prob
        eng.process_frame([HandInput("Right", h, h)], ts, 1920.0, 1080.0)
    assert eng.dwell_lock_activations == 1
    assert eng.dwell_unlock_without_click == 1

    # Drop below 0.20 at t=260ms, then try re-arming at t=300ms (inside 400ms cooldown) -> MUST NOT arm
    for ts, prob in [(260.0, 0.05), (300.0, 0.55)]:
        clf.p = prob
        eng.process_frame([HandInput("Right", h, h)], ts, 1920.0, 1080.0)
    assert eng.dwell_lock_activations == 1

    # Drop below 0.20 at t=640ms (after 620ms cooldown) and rise cleanly at t=680ms -> MUST arm again!
    for ts, prob in [(640.0, 0.05), (680.0, 0.55)]:
        clf.p = prob
        eng.process_frame([HandInput("Right", h, h)], ts, 1920.0, 1080.0)
    assert eng.dwell_lock_activations == 2


def test_lookback_ignored_when_pointer_fast():
    kin = CursorKinematics()
    # Fast pointer movement at 900 px/s recorded at t=100ms
    kin.record_history(100.0, 400.0, 400.0, speed_px_s=900.0)
    kin.record_history(190.0, 490.0, 400.0, speed_px_s=850.0)
    # Speed > 250 px/s -> lookback MUST be ignored and return live_pos (490, 400), NOT (400, 400)
    anchored = kin.lookup_position_guarded(
        100.0,
        (490.0, 400.0),
        max_speed_px_s=250.0,
        max_offset_px=24.0,
    )
    assert anchored == (490.0, 400.0)
    # If dwell_lock fallback is provided while fast, returns fallback
    anchored_fb = kin.lookup_position_guarded(
        100.0,
        (490.0, 400.0),
        max_speed_px_s=250.0,
        max_offset_px=24.0,
        fallback_pos=(485.0, 400.0),
    )
    assert anchored_fb == (485.0, 400.0)


def test_lookback_offset_clamped():
    kin = CursorKinematics()
    # Steady slow point at (500, 500) at t=100ms; live drifted 60px right to (560, 500)
    kin.record_history(100.0, 500.0, 500.0, speed_px_s=20.0)
    anchored = kin.lookup_position_guarded(
        100.0,
        (560.0, 500.0),
        max_speed_px_s=250.0,
        max_offset_px=24.0,
    )
    assert math.hypot(anchored[0] - 560.0, anchored[1] - 500.0) == pytest.approx(24.0, abs=1e-5)
    assert anchored[0] == pytest.approx(536.0, abs=1e-5)
    assert anchored[1] == pytest.approx(500.0, abs=1e-5)


def test_lerp_reset_after_anchored_click():
    from aerohand.core.types import CursorOutput, HandOutput, PoseOutput
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["cursor"]["enable_lerp_60hz"] = True
    rt = AeroHandRuntime(cfg, dry_run=True)
    rt._screen_ox, rt._screen_oy = 0.0, 0.0
    rt._lerp.reset((520.0, 500.0))

    # Anchored click lands at (500, 500) while live cursor is at (520, 500) (offset = 20px > 2px)
    ho = HandOutput(
        hand="Right",
        cursor=CursorOutput(hand="Right", x=520.0, y=500.0, is_pointing=True, is_valid=True),
        pose=PoseOutput(
            hand="Right",
            is_clicking=True,
            click_count=1,
            click_x=500.0,
            click_y=500.0,
        ),
    )
    rt._dispatch_actions([ho], ho, 1000.0)
    # Lerp current MUST sync to anchor (500, 500) while target stays at live (520, 500)
    assert rt._lerp._current == (500.0, 500.0)
    assert rt._lerp._target == (520.0, 500.0)
    # First lerp tick moves smoothly forward from 500 toward 520 without snapping past 520
    step = rt._lerp.step(1.0 / 60.0)
    assert step is not None
    nx, ny, dx, _ = step
    assert 500.0 < nx < 520.0
    assert dx > 0.0


def test_drag_start_uses_pre_pinch_anchor():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["cursor"]["enable_filter"] = False
    cfg["cursor"]["drag_lookback_ms"] = 60.0
    cfg["cursor"]["drag_dwell_lock_ms"] = 0.0
    cfg["cursor"]["click_lookback_max_offset_px"] = 24.0
    cfg["palm_wheel"]["enabled"] = False

    class _DragClf:
        def __init__(self) -> None:
            self.p = 0.0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.THUMB_INDEX_MIDDLE_PINCH] = self.p
            probs[PoseName.IDLE] = max(0.0, 1.0 - self.p)
            return PinchPrediction(
                PoseName.THUMB_INDEX_MIDDLE_PINCH if self.p >= 0.75 else PoseName.IDLE, probs, "high"
            )

        def reset(self, hand=None):
            pass

    clf = _DragClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    eng.anchor_cursor("Right", 500.0, 500.0)
    eng._hands["Right"].kinematics._skip_remaining = 0

    h = _make_open_hand(forward_z=-0.25)
    # Aim steadily at t=0..90ms
    for ts in [0.0, 30.0, 60.0, 90.0]:
        out = eng.process_frame([HandInput("Right", h, h)], ts, 1920.0, 1080.0)[0]
    pre_x, pre_y = out.cursor.x, out.cursor.y

    # Close 3-finger drag pinch with ~15px fingertip drift and hold past 180ms hold threshold
    shifted = [Landmark(p.x + 0.0035, p.y + 0.002, p.z) for p in h]
    clf.p = 0.92
    drag_start_out = None
    for ts in [120.0, 150.0, 200.0, 260.0, 350.0]:
        o = eng.process_frame([HandInput("Right", shifted, shifted)], ts, 1920.0, 1080.0)[0]
        if o.pose.grab_just_started:
            drag_start_out = o
            break
    assert drag_start_out is not None
    assert drag_start_out.cursor.x > pre_x + 10.0  # live fingertip drifted
    # Drag anchor MUST be the pre-pinch position, NOT the drifted live fingertip!
    assert drag_start_out.pose.drag_x == pytest.approx(pre_x, abs=2.0)
    assert drag_start_out.pose.drag_y == pytest.approx(pre_y, abs=2.0)


def test_right_click_uses_anchor():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["cursor"]["enable_filter"] = False
    cfg["cursor"]["drag_lookback_ms"] = 60.0
    cfg["cursor"]["click_lookback_max_offset_px"] = 24.0
    cfg["palm_wheel"]["enabled"] = False

    class _RingClf:
        def __init__(self) -> None:
            self.p = 0.0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.THUMB_RING_PINCH] = self.p
            probs[PoseName.IDLE] = max(0.0, 1.0 - self.p)
            return PinchPrediction(
                PoseName.THUMB_RING_PINCH if self.p >= 0.75 else PoseName.IDLE, probs, "high"
            )

        def reset(self, hand=None):
            pass

    clf = _RingClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    eng.anchor_cursor("Right", 400.0, 300.0)
    eng._hands["Right"].kinematics._skip_remaining = 0

    h = _make_open_hand(forward_z=-0.25)
    for ts in [0.0, 30.0, 60.0, 90.0]:
        out = eng.process_frame([HandInput("Right", h, h)], ts, 1920.0, 1080.0)[0]
    pre_x, pre_y = out.cursor.x, out.cursor.y

    # Ring pinch closes with slight drift (~10px < 15px still_right_click_motion_px), holds >180ms, then releases
    shifted = [Landmark(p.x + 0.0022, p.y, p.z) for p in h]
    clf.p = 0.92
    for ts in [120.0, 150.0, 220.0, 360.0]:
        eng.process_frame([HandInput("Right", shifted, shifted)], ts, 1920.0, 1080.0)

    clf.p = 0.05
    rc_out = None
    for ts in [390.0, 420.0, 450.0, 480.0, 510.0, 540.0]:
        o = eng.process_frame([HandInput("Right", shifted, shifted)], ts, 1920.0, 1080.0)[0]
        if o.pose.is_right_clicking:
            rc_out = o
            break
    assert rc_out is not None
    assert rc_out.pose.right_click_x == pytest.approx(pre_x, abs=2.0)
    assert rc_out.pose.right_click_y == pytest.approx(pre_y, abs=2.0)


def test_drag_lock_released_on_grab_start():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["cursor"]["enable_filter"] = False
    cfg["cursor"]["drag_dwell_lock_ms"] = 250.0  # intentionally longer than hold threshold to prove early release
    cfg["pose_state_machine"]["pose_hold_threshold_ms"] = 80
    cfg["palm_wheel"]["enabled"] = False

    class _DragClf:
        def __init__(self) -> None:
            self.p = 0.0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.THUMB_INDEX_MIDDLE_PINCH] = self.p
            probs[PoseName.IDLE] = max(0.0, 1.0 - self.p)
            return PinchPrediction(
                PoseName.THUMB_INDEX_MIDDLE_PINCH if self.p >= 0.75 else PoseName.IDLE, probs, "high"
            )

        def reset(self, hand=None):
            pass

    clf = _DragClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    eng.anchor_cursor("Right", 500.0, 500.0)
    eng._hands["Right"].kinematics._skip_remaining = 0

    h = _make_open_hand(forward_z=-0.25)
    eng.process_frame([HandInput("Right", h, h)], 0.0, 1920.0, 1080.0)

    # Arm drag dwell lock at t=30ms (would hold until 280ms if not cleared on grab_just_started!)
    clf.p = 0.92
    eng.process_frame([HandInput("Right", h, h)], 30.0, 1920.0, 1080.0)
    assert eng._hands["Right"].drag_dwell_lock_pos is not None
    eng.process_frame([HandInput("Right", h, h)], 60.0, 1920.0, 1080.0)  # pose_enter
    o_start = eng.process_frame([HandInput("Right", h, h)], 150.0, 1920.0, 1080.0)[0]  # hold_start
    assert o_start.pose.grab_just_started is True
    assert eng._hands["Right"].drag_dwell_lock_pos is None

    # Next frame at t=180ms (still < 280ms!): dragging hand MUST move cursor freely!
    moved = [Landmark(p.x + 0.02, p.y, p.z) for p in h]
    o_drag = eng.process_frame([HandInput("Right", moved, moved)], 180.0, 1920.0, 1080.0)[0]
    assert o_drag.pose.is_grabbing is True
    assert o_drag.cursor.delta_x > 20.0


def test_ring_buffer_capped():
    from aerohand.ml.collector import FrameRingBuffer

    ring = FrameRingBuffer(max_frames=100)
    h = _make_open_hand()
    for i in range(250):
        ring.push(h, h, "Right", 0.95, i * 33)
        ring.push(h, h, "Left", 0.92, i * 33)
    assert ring.len_for_hand("Right") == 100
    assert ring.len_for_hand("Left") == 100
    assert ring.total_frames() == 200


def test_negative_snapshot_roundtrips_to_trainer(tmp_path: Path):
    import json

    from aerohand.ml.collector import FrameRingBuffer

    ring = FrameRingBuffer(max_frames=100)
    h = _make_open_hand()
    # Push 40 frames at 33ms intervals (~1.3s), mark false trigger at frame 20
    for i in range(40):
        ring.push(
            h,
            h,
            "Right",
            0.94,
            1000 + i * 33,
            probabilities={"idle": 0.2, "thumb_index_pinch": 0.8},
        )
    ring.record_trigger("dictation_start", "Right", 1000 + 20 * 33)

    fp = ring.save_negative_snapshot(tmp_path, window_s=3.0)
    assert fp is not None and fp.is_file()
    raw = json.loads(fp.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    assert raw["hard_negative"] is True
    assert raw["trigger_type"] == "dictation_start"
    assert len(raw["samples"]) == 40
    assert all(s["label"] == 0 and s["label_name"] == "idle" for s in raw["samples"])

    # Round-trip directly into train_tcn.py with force_idle=True (same path as --negatives-dir)
    mod = _load_train_tcn_module("train_tcn_neg_rt")
    xs, ys = mod.samples_to_sequences(raw["samples"], feature_set="invariant348", seq_len=5, force_idle=True)
    assert xs.shape == (40 - 5 + 1, 5, FEATURE_DIM_3D)
    assert np.all(ys == 0)
    assert np.all(np.isfinite(xs))


def _load_replay_eval_module(name: str = "replay_eval_mod"):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent.parent / "scripts" / "replay_eval.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_replay_is_deterministic(tmp_path: Path):
    import json

    mod = _load_replay_eval_module("replay_eval_det")
    col = PinchDataCollector(session_meta={"subject": "eval", "wrist_angle": "0", "distance": "mid", "lighting": "bright"})
    col.set_recording(True)
    h = _make_open_hand(forward_z=-0.25)
    col.set_pose(PoseName.IDLE)
    for i in range(15):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    col.set_pose(PoseName.THUMB_MIDDLE_PINCH)
    pinched = list(h)
    pinched[4] = Landmark(h[12].x + 0.004, h[12].y + 0.004, h[12].z)
    for i in range(15, 25):
        col.add_frame(pinched, pinched, "Right", 0.95, 1000 + i * 33)
    fp = col.save(tmp_path)

    cfg = load_config()
    r1 = mod.evaluate_recordings(session_paths=[fp], config=cfg)
    r2 = mod.evaluate_recordings(session_paths=[fp], config=cfg)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert r1["total_frames"] == 25
    assert r1["classification"]["accuracy"] > 0.5


def test_replay_reports_false_triggers(tmp_path: Path):
    from aerohand.ml.base import PinchPrediction

    mod = _load_replay_eval_module("replay_eval_ft")
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    h = _make_open_hand(forward_z=-0.25)
    # 20 frames (~660 ms) of ground-truth idle
    for i in range(20):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    neg_fp = col.save(tmp_path)

    class _FalseClickClf:
        def __init__(self) -> None:
            self.n = 0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            self.n += 1
            probs = idle_probabilities()
            # Trigger one spurious tap (enter on frames 4-6, release on frames 7-10)
            if 4 <= self.n <= 6:
                probs[PoseName.IDLE] = 0.05
                probs[PoseName.THUMB_MIDDLE_PINCH] = 0.95
                return PinchPrediction(PoseName.THUMB_MIDDLE_PINCH, probs, "high")
            return PinchPrediction(PoseName.IDLE, probs, "high")

        def reset(self, hand=None):
            self.n = 0

    cfg = load_config()
    cfg["palm_wheel"]["enabled"] = False
    rep = mod.evaluate_recordings(
        negative_paths=[neg_fp],
        config=cfg,
        classifier=_FalseClickClf(),
        max_ft_per_hour=2.0,
    )
    assert rep["false_triggers"]["clicks"] == 1
    assert rep["false_triggers"]["total"] == 1
    assert rep["false_triggers"]["total_ft_per_hour"] > 2.0
    assert rep["passed"] is False

    # Baseline regression gate check
    rep_reg = mod.evaluate_recordings(
        negative_paths=[neg_fp],
        config=cfg,
        classifier=_FalseClickClf(),
        max_ft_per_hour=100000.0,
        baseline={"classification": {"macro_f1": 0.99}, "false_triggers": {"total_ft_per_hour": 0.0}},
        max_f1_regression=0.01,
    )
    assert rep_reg["baseline_diff"]["regressed_macro_f1"] is True
    assert rep_reg["passed"] is False


def test_guided_targets_cover_all_classes_and_conditions(tmp_path: Path):
    from aerohand.core.types import ALL_POSE_NAMES
    from aerohand.ml.collector import (
        COLLECTOR_CLASS_TARGETS,
        CORPUS_TOTAL_TARGET,
        DISTANCE_BUCKETS,
        HARD_NEGATIVE_GROUPS,
        LIGHTING_BUCKETS,
        WRIST_ANGLE_BUCKETS,
        compute_corpus_condition_matrix,
        default_guided_steps,
    )

    assert CORPUS_TOTAL_TARGET == 21000
    assert COLLECTOR_CLASS_TARGETS["idle"]["total"] == 3000
    for p in ALL_POSE_NAMES:
        assert p.value in COLLECTOR_CLASS_TARGETS
        assert COLLECTOR_CLASS_TARGETS[p.value] == {"total": 3000, "Right": 2000, "Left": 1000}

    steps = default_guided_steps()
    covered_poses = {s.pose for s in steps}
    covered_hn = {s.hard_negative_group for s in steps if s.hard_negative_group}
    assert covered_poses == set(ALL_POSE_NAMES)
    assert covered_hn == set(HARD_NEGATIVE_GROUPS)

    # Save sessions spanning all 4 wrist-angle buckets (25% each >= 15%) and check matrix
    h = _make_open_hand()
    for angle in WRIST_ANGLE_BUCKETS:
        col = PinchDataCollector(
            session_meta={"subject": "minh", "wrist_angle": angle, "distance": "mid", "lighting": "bright"}
        )
        col.set_recording(True)
        col.set_pose(PoseName.THUMB_MIDDLE_PINCH)
        for i in range(5):
            col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
        col.save(tmp_path)

    mat = compute_corpus_condition_matrix(tmp_path)
    assert mat["total_frames"] == 20
    assert mat["angle_balanced"] is True
    assert set(mat["by_angle"].keys()) == set(WRIST_ANGLE_BUCKETS)
    assert set(mat["by_distance"].keys()) == set(DISTANCE_BUCKETS)
    assert set(mat["by_lighting"].keys()) == set(LIGHTING_BUCKETS)
    assert set(mat["by_hard_negative_group"].keys()) == set(HARD_NEGATIVE_GROUPS)


def test_save_blocked_by_quality_gate(tmp_path: Path):
    from aerohand.ml.collector import QualityGateError
    from aerohand.runtime import AeroHandRuntime

    col = PinchDataCollector(
        session_meta={"subject": "minh", "wrist_angle": "0", "distance": "mid", "lighting": "dim"},
        quality_gate_mode="strict",
    )
    col.set_pose(PoseName.THUMB_RING_PINCH)
    col.set_recording(True)
    h = _make_open_hand()
    clipped = [Landmark(0.01 if i == 0 else p.x, p.y, p.z) for i, p in enumerate(h)]
    col.add_frame(clipped, h, "Right", 0.40, 1000)
    col.add_frame(h, h, "Right", 0.45, 1300)  # gap > 100ms, low confidence, clipped

    # Strict mode save MUST block and raise QualityGateError (subclass of ValueError)
    with pytest.raises(QualityGateError) as exc_info:
        col.save(tmp_path)
    assert "low_confidence" in exc_info.value.warnings
    assert "hand_clipped_edge" in exc_info.value.warnings
    assert "time_gaps" in exc_info.value.warnings
    assert list(tmp_path.glob("*.json")) == []

    # Runtime wrapper catches cleanly without crashing and reports saved=False + blocked_warnings
    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["collector"]["quality_gate_mode"] = "strict"
    rt = AeroHandRuntime(cfg, dry_run=True, collect_dir=tmp_path)
    rt.collector = col
    st = rt.collector_save()
    assert st["saved"] is False
    assert "low_confidence" in st["blocked_warnings"]

    # Explicit override saves cleanly with condition_id populated
    fp = col.save(tmp_path, allow_low_quality=True)
    assert fp.is_file()


def test_ab_report_contains_both_feature_sets():
    mod = _load_train_tcn_module("train_tcn_ab_rep")
    runs_348 = [
        {"seed": 42, "macro_f1": 0.91, "val_accuracy": 0.93, "idle_fp_rate": 0.012, "ft_per_hour": 1.1, "confusable_pairs": {"thumb_index_pinch <-> idle": {"total_errors": 4}}},
        {"seed": 1337, "macro_f1": 0.92, "val_accuracy": 0.94, "idle_fp_rate": 0.010, "ft_per_hour": 0.9, "confusable_pairs": {"thumb_index_pinch <-> idle": {"total_errors": 2}}},
    ]
    runs_885 = [
        {"seed": 42, "macro_f1": 0.925, "val_accuracy": 0.94, "idle_fp_rate": 0.011, "ft_per_hour": 1.0, "confusable_pairs": {"thumb_index_pinch <-> idle": {"total_errors": 3}}},
        {"seed": 1337, "macro_f1": 0.930, "val_accuracy": 0.945, "idle_fp_rate": 0.010, "ft_per_hour": 0.9, "confusable_pairs": {"thumb_index_pinch <-> idle": {"total_errors": 3}}},
    ]
    rep = mod.build_ab_report(runs_348, runs_885, seeds=[42, 1337])
    assert set(rep["feature_sets"].keys()) == {"invariant348", "airmouse885"}
    assert rep["feature_sets"]["invariant348"]["num_seeds"] == 2
    assert rep["feature_sets"]["airmouse885"]["num_seeds"] == 2
    assert "thumb_index_pinch <-> idle" in rep["feature_sets"]["invariant348"]["confusable_pairs_mean"]
    assert rep["decision"]["winner"] in ("invariant348", "airmouse885")


def test_default_feature_set_decision_rule():
    mod = _load_train_tcn_module("train_tcn_ab_rule")

    # Case 1: airmouse885 gain is <= 0.02 -> retain default invariant348
    d1 = mod.evaluate_ab_decision(
        {"macro_f1_mean": 0.90, "idle_fp_rate_mean": 0.01, "ft_per_hour_mean": 1.0},
        {"macro_f1_mean": 0.915, "idle_fp_rate_mean": 0.01, "ft_per_hour_mean": 1.0},
    )
    assert d1["winner"] == "invariant348"

    # Case 2: airmouse885 beats by > 0.02 BUT increases idle_fp_rate -> retain invariant348
    d2 = mod.evaluate_ab_decision(
        {"macro_f1_mean": 0.90, "idle_fp_rate_mean": 0.01, "ft_per_hour_mean": 1.0},
        {"macro_f1_mean": 0.94, "idle_fp_rate_mean": 0.015, "ft_per_hour_mean": 1.0},
    )
    assert d2["winner"] == "invariant348"

    # Case 3: airmouse885 beats by > 0.02 AND does not raise idle_fp_rate or ft_per_hour -> select airmouse885
    d3 = mod.evaluate_ab_decision(
        {"macro_f1_mean": 0.90, "idle_fp_rate_mean": 0.012, "ft_per_hour_mean": 1.2},
        {"macro_f1_mean": 0.935, "idle_fp_rate_mean": 0.010, "ft_per_hour_mean": 0.8},
    )
    assert d3["winner"] == "airmouse885"


def test_i18n_vi_en_keys_equal():
    import re
    import shutil
    import subprocess

    from aerohand.i18n.translations import TRANSLATIONS

    def _collect_keys(obj, prefix=""):
        keys = set()
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                keys |= _collect_keys(v, full)
            else:
                assert isinstance(v, str) and v.strip(), f"Empty or non-string i18n leaf: {full}"
                keys.add(full)
        return keys

    vi_keys = _collect_keys(TRANSLATIONS["vi"])
    en_keys = _collect_keys(TRANSLATIONS["en"])
    assert vi_keys == en_keys, f"i18n mismatch: vi-only={vi_keys - en_keys}, en-only={en_keys - vi_keys}"

    # Verify JS syntax of both WKWebView HTML UIs on macOS CI
    if shutil.which("osascript"):
        root = Path(__file__).resolve().parent.parent / "src" / "aerohand" / "ui" / "web"
        for fname in ("overlay.html", "control_center.html"):
            txt = (root / fname).read_text(encoding="utf-8")
            m = re.search(r"<script>(.*)</script>", txt, re.S)
            assert m is not None, f"Missing <script> in {fname}"
            res = subprocess.run(
                ["osascript", "-l", "JavaScript", "-e", "new Function(" + repr(m.group(1)) + '); "OK";'],
                capture_output=True,
                text=True,
                check=False,
            )
            assert res.returncode == 0, f"JS syntax error in {fname}: {res.stderr}"


def test_reason_line_uses_live_gesture_map():
    from aerohand.config import apply_gesture_preset
    from aerohand.core.engine import format_refusal_reason

    cfg_air = load_config()
    msg_air_vi = format_refusal_reason("dictation_arming", cfg_air, lang="vi")
    assert msg_air_vi is not None and "Chụm Cái + Trỏ" in msg_air_vi

    # Switch to Classic preset (dictation -> thumb_middle_pinch) -> reason line updates dynamically!
    cfg_cls = apply_gesture_preset(cfg_air, "classic")
    msg_cls_vi = format_refusal_reason("dictation_arming", cfg_cls, lang="vi")
    assert msg_cls_vi is not None and "Chụm Cái + Giữa" in msg_cls_vi
    assert "Chụm Cái + Trỏ" not in msg_cls_vi

    # Custom remap for click hand & left click pose in English
    cfg_custom = load_config()
    cfg_custom["hand"]["click_hand"] = "Left"
    cfg_custom["pose_state_machine"]["tap_actions"] = {"thumb_pinky_pinch": "left_click"}
    msg_en = format_refusal_reason("non_click_hand", cfg_custom, lang="en")
    assert msg_en is not None and "Thumb + Pinky Pinch" in msg_en and "Left Hand" in msg_en


def test_calibration_includes_loose_pinch_step():
    from aerohand.core.engine import (
        CALIBRATION_STEPS,
        LOOSE_PINCH_STEP,
        SAFE_PINCH_OFF_BOUNDS,
        SAFE_PINCH_ON_BOUNDS,
        PinchCalibrationWizard,
    )

    assert LOOSE_PINCH_STEP in CALIBRATION_STEPS
    assert len(CALIBRATION_STEPS) == 7

    wiz = PinchCalibrationWizard(hold_seconds=1.0)
    wiz.start(now_s=0.0)
    h = _make_open_hand()
    seen_steps = []
    t = 0.0
    for _ in range(len(CALIBRATION_STEPS)):
        st = wiz.status(now_s=t)
        seen_steps.append(st["current_pose"])
        wiz.update(h, h, "Right", now_s=t + 0.2)
        wiz.update(h, h, "Right", now_s=t + 1.05)
        t += 1.05

    assert LOOSE_PINCH_STEP in seen_steps
    assert wiz.state == "done"
    assert wiz.suggestion is not None
    assert SAFE_PINCH_ON_BOUNDS[0] <= wiz.suggestion["pinch_on_dist"] <= SAFE_PINCH_ON_BOUNDS[1]
    assert SAFE_PINCH_OFF_BOUNDS[0] <= wiz.suggestion["pinch_off_dist"] <= SAFE_PINCH_OFF_BOUNDS[1]


def test_quality_gate_clean_mode_saves_session_with_gaps(tmp_path: Path):
    import json

    col = PinchDataCollector(session_meta={"subject": "minh", "wrist_angle": "0", "distance": "mid", "lighting": "bright"})
    assert col.quality_gate_mode == "clean"
    col.set_pose(PoseName.THUMB_MIDDLE_PINCH)
    col.set_recording(True)
    h = _make_open_hand()
    clipped = [Landmark(0.01 if i == 0 else p.x, p.y, p.z) for i, p in enumerate(h)]

    # Segment 0: 10 clean frames + 1 low-conf + 1 clipped edge
    for i in range(10):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    col.add_frame(h, h, "Right", 0.40, 1330)       # dropped low confidence
    col.add_frame(clipped, h, "Right", 0.95, 1363) # dropped clipped edge
    # Gap of 200ms (>100ms) -> Segment 1: 10 clean frames
    for i in range(10):
        col.add_frame(h, h, "Right", 0.94, 1563 + i * 33)

    fp = col.save(tmp_path)
    assert fp.is_file()
    data = json.loads(fp.read_text(encoding="utf-8"))
    qr = data["quality_report"]
    assert qr["mode"] == "clean"
    assert qr["kept_count"] == 20
    assert qr["dropped_low_confidence"] == 1
    assert qr["dropped_edge_clipped"] == 1
    assert qr["segments_count"] == 2


def test_session_split_at_time_gaps():
    from aerohand.ml.collector import clean_recording_samples

    h = _make_open_hand()
    flat = [c for lm in h for c in (lm.x, lm.y, lm.z)]
    raw = [
        {"timestamp_ms": 1000, "label_name": "idle", "handedness": "Right", "confidence": 0.95, "landmarks_2d": flat},
        {"timestamp_ms": 1033, "label_name": "idle", "handedness": "Right", "confidence": 0.95, "landmarks_2d": flat},
        {"timestamp_ms": 1200, "label_name": "idle", "handedness": "Right", "confidence": 0.95, "landmarks_2d": flat},
        {"timestamp_ms": 1233, "label_name": "idle", "handedness": "Right", "confidence": 0.95, "landmarks_2d": flat},
    ]
    cleaned, rep = clean_recording_samples(raw)
    assert rep["gap_splits"] == 1
    assert rep["segments_count"] == 2
    assert [s["segment_id"] for s in cleaned] == [0, 0, 1, 1]


def test_autosave_draft_written_on_stop(tmp_path: Path):
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    rt = AeroHandRuntime(cfg, dry_run=True, collect_dir=tmp_path)
    h = _make_open_hand()
    rt.collector_set_recording(True)
    rt.collector.add_frame(h, h, "Right", 0.95, 1000)
    rt.collector.add_frame(h, h, "Right", 0.95, 1033)
    # Stopping recording MUST write a draft under _drafts/
    rt.collector_set_recording(False)
    drafts = list((tmp_path / "_drafts").glob("draft_*.json"))
    assert len(drafts) == 1

    # Saving formally MUST clean up the draft file
    st = rt.collector_save()
    assert st["saved"] is True
    assert list((tmp_path / "_drafts").glob("draft_*.json")) == []


def test_upper_bound_zero_events_is_three_over_t():
    mod = _load_replay_eval_module("replay_eval_ucb")
    # 24.62 minutes ~= 0.410333 hours -> 0 events gives ~2.9957 / 0.410333 ~= 7.30 FT/h (Rule of Three: 3/T)
    t_h = 24.62 / 60.0
    ucb = mod.poisson_upper_bound_rate(0, t_h, ci_level=0.95)
    assert ucb == pytest.approx(3.0 / t_h, rel=0.01)
    assert ucb > 7.0
    # Over 1.5 hours, 0 events gives ~1.997 <= 2.0 FT/h
    assert mod.poisson_upper_bound_rate(0, 1.5, ci_level=0.95) <= 2.0


def test_replay_eval_fails_when_exposure_insufficient(tmp_path: Path):
    from aerohand.ml.base import PinchPrediction

    mod = _load_replay_eval_module("replay_eval_exp")
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    h = _make_open_hand(forward_z=-0.25)
    for i in range(30):  # ~1 second of idle -> way below 1.5h exposure!
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    neg_fp = col.save(tmp_path)

    class _AlwaysIdle:
        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high")

        def reset(self, hand=None):
            pass

    cfg = load_config()
    cfg["palm_wheel"]["enabled"] = False
    rep = mod.evaluate_recordings(
        negative_paths=[neg_fp],
        config=cfg,
        classifier=_AlwaysIdle(),
        max_ft_per_hour=2.0,
        min_exposure_hours_for_claim=1.5,
    )
    assert rep["false_triggers"]["total"] == 0
    assert rep["false_triggers"]["exposure_sufficient"] is False
    assert "insufficient exposure" in rep["false_triggers"]["exposure_notice"]
    assert rep["passed"] is False


def _load_script_module(rel_path: str, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parent.parent / rel_path
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_converter_flips_handedness_like_bench(tmp_path: Path):
    import json

    conv_mod = _load_script_module("scripts/convert_gesturectl.py", "convert_gctl_flip")
    h = _make_open_hand()
    lms_dicts = [{"x": p.x, "y": p.y, "z": p.z, "visibility": 1.0} for p in h]
    src = tmp_path / "d_dev_sample.json"
    src.write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [
                    {
                        "timestamp_ms": 1000,
                        "hand": {"handedness": "Left", "score": 0.95, "landmarks": lms_dicts, "world_landmarks": lms_dicts},
                    },
                    {
                        "timestamp_ms": 1033,
                        "hand": {"handedness": "Right", "score": 0.92, "landmarks": lms_dicts, "world_landmarks": lms_dicts},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "converted"
    res = conv_mod.convert_gesturectl_file(src, out_dir)
    out_data = json.loads(Path(res["output_path"]).read_text(encoding="utf-8"))
    assert out_data["schema_version"] == 2
    assert out_data["session_meta"]["handedness_flipped"] is True
    assert [s["handedness"] for s in out_data["samples"]] == ["Right", "Left"]


def test_converter_refuses_heldout_path(tmp_path: Path):
    conv_mod = _load_script_module("scripts/convert_gesturectl.py", "convert_gctl_ho")
    ho_dir = tmp_path / "m01_v2" / "heldout"
    ho_dir.mkdir(parents=True)
    ho_fp = ho_dir / "d_heldout_01.json"
    ho_fp.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="REFUSED"):
        conv_mod.convert_gesturectl_file(ho_fp, tmp_path / "out")


def test_cue_phase_press_label_requires_pinch_distance(tmp_path: Path):
    import json

    conv_mod = _load_script_module("scripts/convert_gesturectl.py", "convert_gctl_cue")
    open_h = _make_open_hand()
    closed_h = list(open_h)
    closed_h[4] = Landmark(open_h[8].x + 0.002, open_h[8].y + 0.002, open_h[8].z)

    def _to_d(lms):
        return [{"x": p.x, "y": p.y, "z": p.z} for p in lms]

    # Build a cadence session: rest (0..1000ms), press with 1 unclosed frame + 5 closed frames (1000..2200ms)
    frames = []
    for i in range(15):  # 0..462ms rest (well before 1000ms transition)
        frames.append({"timestamp_ms": i * 33, "cue_phase": "rest", "handedness": "Left", "score": 0.95, "landmarks": _to_d(open_h), "world_landmarks": _to_d(open_h)})
    # Transition at 1000ms; frames at 1500..1700ms are outside +/-300ms trim window
    frames.append({"timestamp_ms": 1000, "cue_phase": "press", "handedness": "Left", "score": 0.95, "landmarks": _to_d(closed_h), "world_landmarks": _to_d(closed_h)})
    # Unclosed press frame at 1500ms MUST be dropped by 3D distance check
    frames.append({"timestamp_ms": 1500, "cue_phase": "press", "handedness": "Left", "score": 0.95, "landmarks": _to_d(open_h), "world_landmarks": _to_d(open_h)})
    # Truly closed press frames at 1533..1632ms MUST be kept as thumb_index_pinch
    for i in range(4):
        frames.append({"timestamp_ms": 1533 + i * 33, "cue_phase": "press", "handedness": "Left", "score": 0.95, "landmarks": _to_d(closed_h), "world_landmarks": _to_d(closed_h)})

    src = tmp_path / "clicks_pinch_cadence_near_test.json"
    src.write_text(json.dumps({"fps": 30.0, "frames": frames}), encoding="utf-8")
    res = conv_mod.convert_gesturectl_file(src, tmp_path / "converted", pinch_on_dist=0.34, trim_ms=300.0)
    assert res["dropped_unclosed_press"] == 1
    assert res["dropped_edge_trim"] >= 1
    assert res["counts"]["thumb_index_pinch"] == 4
    assert res["counts"]["idle"] > 0


def test_converted_file_roundtrips_to_replay_eval(tmp_path: Path):
    import json

    conv_mod = _load_script_module("scripts/convert_gesturectl.py", "convert_gctl_rt")
    replay_mod = _load_replay_eval_module("replay_eval_conv_rt")
    dom_mod = _load_script_module("scripts/domain_check.py", "domain_check_rt")

    h = _make_open_hand(forward_z=-0.25)
    lms_d = [{"x": p.x, "y": p.y, "z": p.z} for p in h]
    src = tmp_path / "d_dev_01_drink_water.json"
    src.write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [
                    {"timestamp_ms": 1000 + i * 33, "hand": {"handedness": "Left", "score": 0.95, "landmarks": lms_d, "world_landmarks": lms_d}}
                    for i in range(12)
                ],
            }
        ),
        encoding="utf-8",
    )
    res = conv_mod.convert_gesturectl_file(src, tmp_path / "converted")
    out_fp = Path(res["output_path"])
    rep = replay_mod.evaluate_recordings(negative_paths=[out_fp], min_exposure_hours_for_claim=0.0)
    assert rep["total_frames"] == 12
    assert rep["sessions_evaluated"] == [out_fp.name]

    # Also verify domain_check KS comparison works cleanly
    dom = dom_mod.compare_domains([out_fp], [out_fp])
    assert dom["domain_shift_detected"] is False
    assert dom["features"]["palm_size"]["ks_stat"] == 0.0


def test_bench_delegates_to_replay_eval(tmp_path: Path):
    import json

    bench_mod = _load_script_module("scripts/bench_d_dev_replay.py", "bench_delegate_test")
    h = _make_open_hand(forward_z=-0.25)
    lms_d = [{"x": p.x, "y": p.y, "z": p.z} for p in h]
    dev_dir = tmp_path / "dev_set"
    dev_dir.mkdir()
    (dev_dir / "d_dev_01_test.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "frames": [
                    {"timestamp_ms": 1000 + i * 33, "hand": {"handedness": "Left", "score": 0.95, "landmarks": lms_d, "world_landmarks": lms_d}}
                    for i in range(10)
                ],
            }
        ),
        encoding="utf-8",
    )
    rep = bench_mod.benchmark_dir(dev_dir)
    assert rep["total_frames"] == 10
    assert "false_triggers" in rep and "total_upper_bound_per_hour" in rep["false_triggers"]


def test_replay_counts_dictation_starts(tmp_path: Path):
    from aerohand.ml.base import PinchPrediction

    mod = _load_replay_eval_module("replay_eval_dict_ft")
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    h = _make_open_hand(forward_z=-0.25)
    # 25 frames spanning ~800ms (> dictation_arm_ms=350ms + enter_frames=3)
    for i in range(25):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    neg_fp = col.save(tmp_path)

    class _FalseDictClf:
        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_INDEX_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_INDEX_PINCH, probs, "high")

        def reset(self, hand=None):
            pass

    cfg = load_config()
    cfg["palm_wheel"]["enabled"] = False
    rep = mod.evaluate_recordings(
        negative_paths=[neg_fp],
        config=cfg,
        classifier=_FalseDictClf(),
        min_exposure_hours_for_claim=0.0,
    )
    assert rep["false_triggers"]["dictation_starts"] == 1
    assert rep["false_triggers"]["by_type"]["dictation_start"] == 1
    assert len(rep["false_triggers"]["per_scenario"]) == 1


def test_ft_jsonl_contains_feature_fields(tmp_path: Path):
    import json

    from aerohand.ml.base import PinchPrediction

    mod = _load_replay_eval_module("replay_eval_jsonl")
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    h = _make_open_hand(forward_z=-0.25)
    for i in range(20):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    neg_fp = col.save(tmp_path)

    class _FalseClickClf:
        def __init__(self) -> None:
            self.n = 0

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            self.n += 1
            probs = idle_probabilities()
            if 4 <= self.n <= 6:
                probs[PoseName.IDLE] = 0.05
                probs[PoseName.THUMB_MIDDLE_PINCH] = 0.95
                return PinchPrediction(PoseName.THUMB_MIDDLE_PINCH, probs, "high")
            return PinchPrediction(PoseName.IDLE, probs, "high")

        def reset(self, hand=None):
            self.n = 0

    jsonl_fp = tmp_path / "ft_events.jsonl"
    cfg = load_config()
    cfg["palm_wheel"]["enabled"] = False
    rep = mod.evaluate_recordings(
        negative_paths=[neg_fp],
        config=cfg,
        classifier=_FalseClickClf(),
        ft_jsonl_path=jsonl_fp,
        min_exposure_hours_for_claim=0.0,
    )
    lines = [json.loads(ln) for ln in jsonl_fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == rep["false_triggers"]["total"] == 1
    ev = lines[0]
    for req_key in (
        "scenario",
        "timestamp_ms",
        "hand",
        "trigger_type",
        "pose_probabilities",
        "thumb_distances",
        "curled_count",
        "finger_extensions",
        "wrist_speed",
        "dir_z",
        "palm_size",
        "hand_visible_ms",
    ):
        assert req_key in ev
    assert set(ev["thumb_distances"].keys()) == {"d_idx", "d_mid", "d_ring", "d_pinky", "d_pbase", "d_dual"}
    assert set(ev["finger_extensions"].keys()) == {"index", "middle", "ring", "pinky"}


def test_second_hand_pinch_does_not_cancel_long_dictation():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False
    cfg["gestures"]["dictation_arm_ms"] = 350
    cfg["gestures"]["dictation_priority_ms"] = 1000

    class _LongDictClf:
        def __init__(self) -> None:
            self.left_active = False

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            if handedness == "Right" or self.left_active:
                probs[PoseName.IDLE] = 0.05
                probs[PoseName.THUMB_INDEX_PINCH] = 0.95
                return PinchPrediction(PoseName.THUMB_INDEX_PINCH, probs, "high")
            return PinchPrediction(PoseName.IDLE, probs, "high")

        def reset(self, hand=None):
            pass

    clf = _LongDictClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    left_h = _make_open_hand(forward_z=-0.25)
    right_h = [Landmark(p.x + 0.25, p.y, p.z) for p in left_h]

    # Arm dictation on Right hand (~t=420ms) and keep speaking for 1.2s (t=1650ms)
    for ts in [0.0, 33.0, 66.0, 200.0, 420.0, 800.0, 1200.0, 1650.0]:
        outs = eng.process_frame(
            [HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)], ts, 1920.0, 1080.0
        )
    r_now = next(o for o in outs if o.hand == "Right")
    assert r_now.pose.is_dictating is True

    # Left hand now incidentally forms thumb-index curl at t=1700..1900ms -> MUST NOT cancel dictation or zoom!
    clf.left_active = True
    for ts in [1700.0, 1750.0, 1850.0, 1950.0]:
        outs2 = eng.process_frame(
            [HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)], ts, 1920.0, 1080.0
        )
        r_after = next(o for o in outs2 if o.hand == "Right")
        assert r_after.pose.dictation_cancelled is False
        assert r_after.pose.is_dictating is True
        assert eng.last_zoom.is_zooming is False
        assert eng.last_refusal_code == "dictation_priority"


def test_zoom_still_cancels_dictation_within_arbitration_window():
    from aerohand.ml.base import PinchPrediction

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False
    cfg["gestures"]["dictation_arm_ms"] = 350
    cfg["gestures"]["dictation_priority_ms"] = 1000
    cfg["gestures"]["zoom_arbitration_ms"] = 300

    class _JoinZoomClf:
        def __init__(self) -> None:
            self.left_active = False

        def predict(self, landmarks_2d=None, world_landmarks=None, handedness="Right", timestamp_ms=0.0):
            probs = idle_probabilities()
            if handedness == "Right" or self.left_active:
                probs[PoseName.IDLE] = 0.05
                probs[PoseName.THUMB_INDEX_PINCH] = 0.95
                return PinchPrediction(PoseName.THUMB_INDEX_PINCH, probs, "high")
            return PinchPrediction(PoseName.IDLE, probs, "high")

        def reset(self, hand=None):
            pass

    clf = _JoinZoomClf()
    eng = HandCursorEngine(cfg, classifier=clf)
    left_h = _make_open_hand(forward_z=-0.25)
    right_h = [Landmark(p.x + 0.25, p.y, p.z) for p in left_h]

    # Arm dictation at t=420ms, then second hand joins 80ms later (well below 1000ms priority)
    for ts in [0.0, 33.0, 66.0, 200.0, 420.0]:
        eng.process_frame([HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)], ts, 1920.0, 1080.0)
    clf.left_active = True
    outs = eng.process_frame([HandInput("Left", left_h, left_h), HandInput("Right", right_h, right_h)], 500.0, 1920.0, 1080.0)
    r_out = next(o for o in outs if o.hand == "Right")
    assert r_out.pose.dictation_cancelled is True
    assert r_out.pose.is_dictating is False



def test_approach_gate_blocks_static_contact():
    # Hand appears already in static thumb-middle contact without ever opening -> MUST stay IDLE
    clf = Heuristic3DPinchClassifier({"enable_approach_gate": True, "approach_window_ms": 1500.0})
    open_h = _make_open_hand()
    pinched = list(open_h)
    pinched[4] = Landmark(open_h[12].x - 0.004, open_h[12].y, open_h[12].z)

    for i in range(10):
        pred = clf.predict(pinched, pinched, "Right", i * 33.0)
        assert pred.pose == PoseName.IDLE

    # Once the hand genuinely opens then closes -> MUST detect THUMB_MIDDLE_PINCH
    pred_open = clf.predict(open_h, open_h, "Right", 400.0)
    assert pred_open.pose == PoseName.IDLE
    pred_close = clf.predict(pinched, pinched, "Right", 433.0)
    assert pred_close.pose == PoseName.THUMB_MIDDLE_PINCH
    assert pred_close.probabilities[PoseName.THUMB_MIDDLE_PINCH] >= 0.75


def test_posture_gate_requires_extended_index_for_click():
    clf = Heuristic3DPinchClassifier(
        {"enable_approach_gate": True, "enable_posture_gate": True, "index_ext_min": 1.25}
    )
    open_h = _make_open_hand()
    clf.predict(open_h, open_h, "Right", 0.0)

    # Curl index tip right onto index MCP (ext ratio ~1.0 < 1.25) while thumb touches middle -> blocked!
    curled_click = list(open_h)
    curled_click[8] = Landmark(open_h[5].x, open_h[5].y, open_h[5].z)
    curled_click[4] = Landmark(open_h[12].x - 0.004, open_h[12].y, open_h[12].z)
    pred_blocked = clf.predict(curled_click, curled_click, "Right", 33.0)
    assert pred_blocked.pose == PoseName.IDLE

    # Extend index normally while thumb touches middle -> allowed!
    clean_click = list(open_h)
    clean_click[4] = Landmark(open_h[12].x - 0.004, open_h[12].y, open_h[12].z)
    pred_ok = clf.predict(clean_click, clean_click, "Right", 66.0)
    assert pred_ok.pose == PoseName.THUMB_MIDDLE_PINCH


def test_motion_gate_blocks_fast_wrist():
    clf = Heuristic3DPinchClassifier(
        {
            "enable_approach_gate": True,
            "enable_motion_gate": True,
            "max_wrist_speed": 0.18,
            "max_wrist_ang_speed_deg_s": 95.0,
        }
    )
    open_h = _make_open_hand()
    clf.predict(open_h, open_h, "Right", 0.0)

    # Translate whole hand quickly by dx=0.05 in 33ms (~1.5 screen/s >> 0.18) while pinching -> blocked!
    fast_pinch = [Landmark(p.x + 0.05, p.y, p.z) for p in open_h]
    fast_pinch[4] = Landmark(fast_pinch[12].x - 0.004, fast_pinch[12].y, fast_pinch[12].z)
    pred_fast = clf.predict(fast_pinch, fast_pinch, "Right", 33.0)
    assert pred_fast.pose == PoseName.IDLE

    # Steady wrist after reset + open -> allowed!
    clf.reset("Right")
    clf.predict(open_h, open_h, "Right", 500.0)
    steady_pinch = list(open_h)
    steady_pinch[4] = Landmark(open_h[12].x - 0.004, open_h[12].y, open_h[12].z)
    pred_steady = clf.predict(steady_pinch, steady_pinch, "Right", 533.0)
    assert pred_steady.pose == PoseName.THUMB_MIDDLE_PINCH


def test_gates_do_not_block_clean_pinch_fixture():
    # Clean open -> pinch -> hold (even with wrist drag movement) -> release works with zero gate penalty
    cfg = load_config()
    hcfg = dict(cfg["classifier"]["heuristic_3d"])
    clf = Heuristic3DPinchClassifier(hcfg)
    ungated = Heuristic3DPinchClassifier(
        {
            **hcfg,
            "enable_approach_gate": False,
            "enable_posture_gate": False,
            "enable_motion_gate": False,
            "enable_grip_guard": False,
        }
    )
    open_h = _make_open_hand(forward_z=-0.25)

    for target_pose, tip_indices in (
        (PoseName.THUMB_INDEX_PINCH, (8,)),
        (PoseName.THUMB_MIDDLE_PINCH, (12,)),
        (PoseName.THUMB_RING_PINCH, (16,)),
        (PoseName.THUMB_PINKY_PINCH, (20,)),
    ):
        clf.reset("Right")
        # 3 steady open frames
        for i in range(3):
            assert clf.predict(open_h, open_h, "Right", i * 33.0).pose == PoseName.IDLE
        # Clean steady pinch entry matches ungated score identically
        pinched = list(open_h)
        t = open_h[tip_indices[0]]
        pinched[4] = Landmark(t.x - 0.003, t.y, t.z)
        pred_ref = ungated.predict(pinched, pinched, "Right", 100.0)
        pred_enter = clf.predict(pinched, pinched, "Right", 100.0)
        assert pred_enter.pose == target_pose
        assert pred_enter.probabilities[target_pose] == pytest.approx(
            pred_ref.probabilities[target_pose], abs=1e-6
        )
        # Once latched, moving the wrist rapidly during a hold/drag MUST keep the pose active!
        moved = [Landmark(p.x + 0.04, p.y - 0.02, p.z) for p in pinched]
        pred_hold = clf.predict(moved, moved, "Right", 133.0)
        assert pred_hold.pose == target_pose



def test_seal_heldout_manifest_and_counter(tmp_path_factory):
    import json

    tmp_path = tmp_path_factory.mktemp("seal_manifest")
    seal_mod = _load_script_module("scripts/seal_heldout.py", "seal_heldout_test")
    replay_mod = _load_replay_eval_module("replay_eval_seal_test")

    src_dir = tmp_path / "raw_sessions"
    col = PinchDataCollector(session_meta={"subject": "minh", "wrist_angle": "0", "distance": "mid", "lighting": "bright"})
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    h = _make_open_hand(forward_z=-0.25)
    for i in range(15):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    src_fp = col.save(src_dir)

    ho_dir = tmp_path / "heldout_aerohand"
    manifest = seal_mod.seal_heldout_sessions([src_fp], out_dir=ho_dir)
    assert manifest["touched"] == 0
    assert manifest["evaluated_model_hashes"] == []
    assert len(manifest["files"]) == 1
    assert len(manifest["files"][0]["sha256"]) == 64

    # Single-touch eval verifies checksums, runs replay, and increments touched -> 1
    h1 = "a" * 24
    rep = replay_mod.evaluate_sealed_heldout(
        ho_dir,
        model_hash=h1,
        min_exposure_hours_for_claim=0.0,
        max_ft_per_hour=100.0,
    )
    assert rep["heldout_audit"]["touched"] == 1
    assert rep["heldout_audit"]["model_hash"] == h1

    persisted = json.loads((ho_dir / "sealed_manifest.json").read_text(encoding="utf-8"))
    assert persisted["touched"] == 1
    assert persisted["evaluated_model_hashes"] == [h1]

    # Tampering with a sealed file MUST be detected and refused!
    sealed_fp = ho_dir / manifest["files"][0]["filename"]
    sealed_fp.write_text(sealed_fp.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        seal_mod.verify_sealed_heldout_manifest(ho_dir)


def test_heldout_second_run_refused_for_same_model(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("seal_dup")
    seal_mod = _load_script_module("scripts/seal_heldout.py", "seal_heldout_dup")
    replay_mod = _load_replay_eval_module("replay_eval_seal_dup")

    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    h = _make_open_hand(forward_z=-0.25)
    for i in range(12):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    src_fp = col.save(tmp_path / "src")

    ho_dir = tmp_path / "heldout_aerohand"
    seal_mod.seal_heldout_sessions([src_fp], out_dir=ho_dir)

    # First run succeeds
    h_same = "b" * 24
    rep1 = replay_mod.evaluate_sealed_heldout(
        ho_dir, model_hash=h_same, min_exposure_hours_for_claim=0.0
    )
    assert rep1["heldout_audit"]["touched"] == 1

    # Second run with the exact same model hash MUST be refused!
    with pytest.raises(ValueError, match="already evaluated"):
        replay_mod.evaluate_sealed_heldout(
            ho_dir, model_hash=h_same, min_exposure_hours_for_claim=0.0
        )

    # A distinct new model hash is permitted and increments touched -> 2
    rep2 = replay_mod.evaluate_sealed_heldout(
        ho_dir, model_hash="c" * 24, min_exposure_hours_for_claim=0.0
    )
    assert rep2["heldout_audit"]["touched"] == 2


def test_balanced_sampler_caps_scenario_share():
    mod = _load_train_tcn_module("train_tcn_balance_test")
    h = _make_open_hand()

    def _make_bundle(sc_name: str, n_idle: int, n_pos: int = 0, subject: str = "minh"):
        wins = []
        lbls = []
        for _ in range(n_idle):
            wins.append([mod.ParsedSample(0, h, h, "Right", "Right", 0.0)] * 5)
            lbls.append(0)
        for _ in range(n_pos):
            wins.append([mod.ParsedSample(2, h, h, "Right", "Right", 0.0)] * 5)
            lbls.append(2)
        return mod.SessionBundle(
            name=f"{sc_name}.json",
            group_key=f"{subject}::{sc_name}",
            windows=wins,
            labels=np.asarray(lbls, dtype=np.int64),
            scenario=sc_name,
            subject=subject,
        )

    # 1 dominant scenario with 5,000 idle windows + 9 smaller scenarios with 100 idle each (+ 40 positives)
    bundles = [_make_bundle("dominant_typing", 5000, n_pos=40)] + [
        _make_bundle(f"scenario_{i}", 100, n_pos=5) for i in range(9)
    ]
    balanced, stats = mod.balance_scenario_samples(bundles, max_scenario_share=0.15, seed=42)
    assert stats["max_share_before"] > 0.80
    assert stats["max_share_after"] <= 0.15 + 1e-9
    assert "dominant_typing" in stats["capped_scenarios"]

    # Positive windows MUST be 100% preserved (40 + 9*5 = 85)
    pos_after = sum(int(np.sum(b.labels != 0)) for b in balanced)
    assert pos_after == 85

    # Subject-held-out splitter separates non-primary volunteer subjects cleanly
    b_vol = _make_bundle("vol_session", 50, n_pos=10, subject="volunteer_a")
    dev_b, ho_b = mod.split_subject_heldout(bundles + [b_vol], primary_subject="minh")
    assert len(ho_b) == 1 and ho_b[0].subject == "volunteer_a"
    assert all(b.subject == "minh" for b in dev_b)



def test_ab_selection_rule_respects_ft_and_f1():
    mod = _load_train_tcn_module("train_tcn_ab_v050")

    # 1. airmouse885 wins macro-F1 by +0.03 and keeps idle_fp & ft_per_hour flat,
    #    BUT increases 95% Poisson UCB (`ft_upper_bound_per_hour`) -> MUST retain invariant348!
    d_ucb_reg = mod.evaluate_ab_decision(
        {
            "macro_f1_mean": 0.90,
            "idle_fp_rate_mean": 0.01,
            "ft_per_hour_mean": 1.0,
            "ft_upper_bound_per_hour_mean": 1.8,
        },
        {
            "macro_f1_mean": 0.935,
            "idle_fp_rate_mean": 0.01,
            "ft_per_hour_mean": 1.0,
            "ft_upper_bound_per_hour_mean": 2.4,
        },
    )
    assert d_ucb_reg["winner"] == "invariant348"
    assert "ft_upper_bound_per_hour" in d_ucb_reg["reason"]

    # 2. airmouse885 wins macro-F1 by +0.03 and does NOT increase idle_fp, ft/h, or 95% UCB -> airmouse885 wins!
    d_win = mod.evaluate_ab_decision(
        {
            "macro_f1_mean": 0.90,
            "idle_fp_rate_mean": 0.012,
            "ft_per_hour_mean": 1.2,
            "ft_upper_bound_per_hour_mean": 1.9,
        },
        {
            "macro_f1_mean": 0.935,
            "idle_fp_rate_mean": 0.010,
            "ft_per_hour_mean": 0.9,
            "ft_upper_bound_per_hour_mean": 1.7,
        },
    )
    assert d_win["winner"] == "airmouse885"
    assert d_win["selection_score_delta"] > 0.02

    # 3. aggregate_seed_metrics carries ft_upper_bound_per_hour_mean cleanly
    agg = mod.aggregate_seed_metrics(
        [
            {"macro_f1": 0.91, "val_accuracy": 0.92, "idle_fp_rate": 0.01, "ft_per_hour": 1.0, "ft_upper_bound_per_hour": 1.9},
            {"macro_f1": 0.93, "val_accuracy": 0.94, "idle_fp_rate": 0.01, "ft_per_hour": 0.8, "ft_upper_bound_per_hour": 1.7},
        ]
    )
    assert agg["ft_upper_bound_per_hour_mean"] == pytest.approx(1.8, abs=1e-3)


def test_validation_requires_all_classes():
    mod = _load_train_tcn_module("train_tcn_val_cls")
    h = _make_open_hand()

    def _bundle_with_labels(name: str, lbls: List[int]):
        wins = [[mod.ParsedSample(l_i, h, h, "Right", "Right", 0.0)] * 5 for l_i in lbls]
        return mod.SessionBundle(name, name, wins, np.asarray(lbls, dtype=np.int64))

    # Missing classes 5 & 6 -> MUST raise ValueError by default
    partial = [_bundle_with_labels("partial", [0, 1, 2, 3, 4])]
    with pytest.raises(ValueError, match="missing canonical classes"):
        mod.validate_validation_classes(partial, allow_missing=False)

    # allow_missing=True returns the missing class names without raising
    missing = mod.validate_validation_classes(partial, allow_missing=True)
    assert len(missing) == 2

    # All 7 canonical classes present -> passes cleanly with empty missing list
    complete = [_bundle_with_labels("complete", list(range(mod.NUM_CLASSES)))]
    assert mod.validate_validation_classes(complete, allow_missing=False) == []



def test_veto_mode_never_creates_activation():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _StubClf:
        def __init__(self, pose: PoseName, probs_override=None) -> None:
            self.pose = pose
            self.probs_override = probs_override

        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            if self.probs_override is not None:
                probs.update(self.probs_override)
            elif self.pose != PoseName.IDLE:
                probs[PoseName.IDLE] = 0.05
                probs[self.pose] = 0.95
            return PinchPrediction(self.pose, probs, "high")

        def reset(self, hand=None):
            pass

    # Heuristic says IDLE while ONNX strongly predicts THUMB_MIDDLE_PINCH -> veto mode MUST stay IDLE!
    heur_idle = _StubClf(PoseName.IDLE)
    onnx_pinch = _StubClf(PoseName.THUMB_MIDDLE_PINCH)
    guarded = GuardedClassifier(onnx_pinch, heur_idle, mode="veto", veto_min_prob=0.35, veto_idle_prob=0.60)
    h = _make_open_hand()
    for i in range(5):
        pred = guarded.predict(h, h, "Right", i * 33.0)
        assert pred.pose == PoseName.IDLE
    tel = guarded.telemetry()
    assert tel["mode"] == "veto"
    assert tel["veto_count"] == 0


def test_veto_blocks_low_probability_pose():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _DynamicClf:
        def __init__(self) -> None:
            self.pose = PoseName.IDLE
            self.probs = idle_probabilities()

        def set(self, pose: PoseName, p_pose: float, p_idle: float):
            self.pose = pose
            self.probs = idle_probabilities()
            self.probs[PoseName.IDLE] = p_idle
            if pose != PoseName.IDLE:
                self.probs[pose] = p_pose

        def predict(self, *args, **kwargs):
            return PinchPrediction(self.pose, dict(self.probs), "high")

        def reset(self, hand=None):
            pass

    heur = _DynamicClf()
    onnx = _DynamicClf()
    guarded = GuardedClassifier(onnx, heur, mode="veto", veto_min_prob=0.35, veto_idle_prob=0.60)
    h = _make_open_hand()

    # 1. Heuristic proposes THUMB_INDEX_PINCH, ONNX gives prob=0.20 (<0.35) -> VETOED to IDLE!
    heur.set(PoseName.THUMB_INDEX_PINCH, 0.90, 0.10)
    onnx.set(PoseName.IDLE, 0.0, 0.80)
    onnx.probs[PoseName.THUMB_INDEX_PINCH] = 0.20
    pred_vetoed = guarded.predict(h, h, "Right", 100.0)
    assert pred_vetoed.pose == PoseName.IDLE
    assert guarded.telemetry()["veto_count"] == 1

    # 2. User releases to IDLE then retries the exact same pose within 1s -> increments veto_retried_within_1s!
    heur.set(PoseName.IDLE, 0.0, 1.0)
    guarded.predict(h, h, "Right", 250.0)
    heur.set(PoseName.THUMB_INDEX_PINCH, 0.92, 0.08)
    onnx.set(PoseName.THUMB_INDEX_PINCH, 0.82, 0.18)
    # Default veto_confirm_frames=3 requires 3 consecutive approving ready frames before accepting
    for i in range(2):
        assert guarded.predict(h, h, "Right", 400.0 + i * 33.0).pose == PoseName.IDLE
    pred_ok = guarded.predict(h, h, "Right", 466.0)
    assert pred_ok.pose == PoseName.THUMB_INDEX_PINCH
    tel = guarded.telemetry()
    assert tel["veto_count"] == 1
    assert tel["veto_retried_within_1s"] == 1


def test_mode_switch_clears_buffers():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _BufferedClf:
        def __init__(self) -> None:
            self.reset_calls: List[object] = []
            self.buf: List[float] = []

        def predict(self, _l2, _l3, hand, ts):
            self.buf.append(ts)
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high")

        def reset(self, hand=None):
            self.reset_calls.append(hand)
            self.buf.clear()

    onnx = _BufferedClf()
    heur = _BufferedClf()
    guarded = GuardedClassifier(onnx, heur, mode="shadow")
    h = _make_open_hand()
    guarded.predict(h, h, "Right", 0.0)
    guarded.predict(h, h, "Right", 33.0)
    assert len(onnx.buf) == 2 and len(heur.buf) == 2

    # Switching shadow -> veto MUST clear both classifiers' buffers!
    guarded.set_mode("veto")
    assert guarded.mode == "veto"
    assert onnx.buf == [] and heur.buf == []
    assert onnx.reset_calls == [None] and heur.reset_calls == [None]

    # Switching veto -> onnx clears buffers again
    guarded.predict(h, h, "Right", 66.0)
    assert len(onnx.buf) == 1
    guarded.set_mode("onnx")
    assert onnx.buf == [] and len(onnx.reset_calls) == 2


def test_promotion_check_script_rejects_insufficient_exposure():
    from aerohand.core.types import ALL_POSE_NAMES

    promo_mod = _load_script_module("scripts/check_onnx_promotion.py", "check_onnx_promo_test")
    all_high_per_class = {
        p.value: {"precision": 0.99, "recall": 0.985, "f1": 0.987, "support": 500}
        for p in ALL_POSE_NAMES
    }
    good_metrics = {
        "model_hash": "a" * 24,
        "subject_heldout_metrics": {"per_class": all_high_per_class},
    }
    cfg = load_config()

    # Case 1: Insufficient sealed exposure (0.41h < 1.5h) -> MUST fail!
    short_sealed = {
        "heldout_audit": {"touched": 1, "model_hash": "a" * 24},
        "false_triggers": {
            "hand_present_idle_hours": 0.41,
            "total_upper_bound_per_hour": 7.31,
        },
    }
    res_fail = promo_mod.check_onnx_promotion(
        metrics_report=good_metrics,
        sealed_eval_report=short_sealed,
        latency_p95_ms=1.4,
        config=cfg,
    )
    assert res_fail["passed"] is False
    assert res_fail["criteria"]["sealed_negatives_ft_ucb"]["passed"] is False
    assert res_fail["criteria"]["subject_heldout_recall"]["passed"] is True
    assert res_fail["criteria"]["cpu_latency_p95_ms"]["passed"] is True
    assert res_fail["criteria"]["runtime_guard_active"]["passed"] is True

    # Case 2: All criteria satisfied -> MUST pass!
    zero_by_type = {
        "left_click": 0,
        "right_click": 0,
        "drag_start": 0,
        "scroll_stroke": 0,
        "palm_wheel": 0,
        "window_tile": 0,
        "spaces_nav": 0,
        "dictation_start": 0,
    }
    good_sealed = {
        "heldout_audit": {"touched": 1, "model_hash": "a" * 24},
        "false_triggers": {
            "hand_present_idle_hours": 1.60,
            "total_upper_bound_per_hour": 1.87,
            "by_type": dict(zero_by_type),
        },
    }
    res_ok = promo_mod.check_onnx_promotion(
        metrics_report=good_metrics,
        sealed_eval_report=good_sealed,
        latency_p95_ms=1.4,
        config=cfg,
    )
    assert res_ok["passed"] is True
    assert all(c["passed"] for c in res_ok["criteria"].values())

    # Case 3: One class recall < 0.97 -> MUST fail!
    weak_cls = dict(all_high_per_class)
    weak_cls["thumb_pinky_base"] = {"precision": 0.96, "recall": 0.94, "f1": 0.95, "support": 200}
    res_weak = promo_mod.check_onnx_promotion(
        metrics_report={"model_hash": "a" * 24, "subject_heldout_metrics": {"per_class": weak_cls}},
        sealed_eval_report=good_sealed,
        latency_p95_ms=1.4,
        config=cfg,
    )
    assert res_weak["passed"] is False
    assert res_weak["criteria"]["subject_heldout_recall"]["passed"] is False



def test_primary_subject_zero_bundles_aborts():
    mod = _load_train_tcn_module("train_tcn_v051_01_zero")
    h = _make_open_hand()
    wins = [[mod.ParsedSample(0, h, h, "Right", "Right", 0.0)] * 5] * 4
    b_other = mod.SessionBundle(
        "other.json", "other::c1", wins, np.zeros(4, dtype=np.int64), scenario="s1", subject="alice"
    )
    dev_b, ho_b = mod.split_subject_heldout([b_other], primary_subject="minh")
    assert dev_b == [] and len(ho_b) == 1
    with pytest.raises(ValueError, match="matched 0 bundles"):
        mod.validate_subject_split([b_other], dev_b, ho_b, primary_subject="minh")


def test_collector_default_subject_from_config(tmp_path: Path):
    import json

    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    assert cfg["collector"]["subject"] == "minh"
    cfg["audio"]["enabled"] = False
    cfg["collector"]["subject"] = "minh_custom"
    rt = AeroHandRuntime(cfg, dry_run=True, collect_dir=tmp_path)
    assert rt.collector.session_meta["subject"] == "minh_custom"

    h = _make_open_hand()
    rt.collector.set_recording(True)
    rt.collector.set_pose(PoseName.IDLE)
    rt.collector.add_frame(h, h, "Right", 0.95, 1000)
    rt.collector.add_frame(h, h, "Right", 0.95, 1033)
    saved_fp = rt.collector.save(tmp_path)
    data = json.loads(saved_fp.read_text(encoding="utf-8"))
    assert data["session_meta"]["subject"] == "minh_custom"


def test_split_subject_heldout_warns_on_default_subject_names():
    mod = _load_train_tcn_module("train_tcn_v051_01_warn")
    h = _make_open_hand()
    wins = [[mod.ParsedSample(0, h, h, "Right", "Right", 0.0)] * 5] * 4
    b_minh = mod.SessionBundle(
        "minh.json", "minh::c1", wins, np.zeros(4, dtype=np.int64), scenario="s1", subject="minh"
    )
    b_user = mod.SessionBundle(
        "ui.json", "user::c1", wins, np.zeros(4, dtype=np.int64), scenario="s2", subject="user"
    )
    dev_b, ho_b = mod.split_subject_heldout([b_minh, b_user], primary_subject="minh")
    assert len(dev_b) == 1 and len(ho_b) == 1
    with pytest.raises(ValueError, match="placeholder subject"):
        mod.validate_subject_split([b_minh, b_user], dev_b, ho_b, primary_subject="minh")

    # Named volunteer subject passes cleanly
    b_vol = mod.SessionBundle(
        "vol.json", "linh::c1", wins, np.zeros(4, dtype=np.int64), scenario="s3", subject="linh"
    )
    dev2, ho2 = mod.split_subject_heldout([b_minh, b_vol], primary_subject="minh")
    res = mod.validate_subject_split([b_minh, b_vol], dev2, ho2, primary_subject="minh")
    assert res["dev_bundles"] == 1 and res["heldout_bundles"] == 1



def test_train_excludes_files_whose_sha_is_in_sealed_manifest(tmp_path_factory):
    import shutil

    tmp = tmp_path_factory.mktemp("v051_02_sha")
    seal_mod = _load_script_module("scripts/seal_heldout.py", "seal_v051_02_sha")
    train_mod = _load_train_tcn_module("train_v051_02_sha")

    data_root = tmp / "data"
    sess_dir = data_root / "sessions"
    ho_dir = data_root / "heldout_aerohand"

    h = _make_open_hand()
    col1 = PinchDataCollector()
    col1.set_recording(True)
    col1.set_pose(PoseName.IDLE)
    for i in range(8):
        col1.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    fp_sealed = col1.save(sess_dir)

    col2 = PinchDataCollector(session_meta={"wrist_angle": "30"})
    col2.set_recording(True)
    col2.set_pose(PoseName.THUMB_MIDDLE_PINCH)
    for i in range(8):
        col2.add_frame(h, h, "Right", 0.95, 2000 + i * 33)
    fp_keep = col2.save(sess_dir)

    # Even if copied & renamed back into sessions/, SHA-256 exclusion MUST drop it!
    seal_mod.seal_heldout_sessions([fp_sealed], out_dir=ho_dir, move=False)
    renamed_copy = sess_dir / "renamed_leaked_copy.json"
    shutil.copy2(str(fp_sealed), str(renamed_copy))

    kept, excluded = train_mod.collect_training_files([sess_dir], data_root=data_root)
    assert fp_keep in kept
    assert fp_sealed not in kept and renamed_copy not in kept
    assert len(excluded) == 2


def test_seal_default_moves_files_out_of_source_dirs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v051_02_move")
    seal_mod = _load_script_module("scripts/seal_heldout.py", "seal_v051_02_move")

    src_dir = tmp / "sessions"
    h = _make_open_hand()
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    col.add_frame(h, h, "Right", 0.95, 1000)
    src_fp = col.save(src_dir)
    assert src_fp.is_file()

    ho_dir = tmp / "heldout_aerohand"
    manifest = seal_mod.seal_heldout_sessions([src_fp], out_dir=ho_dir)
    # Default mode MUST move source file out so it cannot leak back into training
    assert not src_fp.exists()
    assert (ho_dir / manifest["files"][0]["filename"]).is_file()


def test_seal_force_refused_after_touch(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v051_02_force")
    seal_mod = _load_script_module("scripts/seal_heldout.py", "seal_v051_02_force")
    replay_mod = _load_replay_eval_module("replay_v051_02_force")

    h = _make_open_hand(forward_z=-0.25)
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    for i in range(10):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    src_fp = col.save(tmp / "src")

    ho_dir = tmp / "heldout_aerohand"
    seal_mod.seal_heldout_sessions([src_fp], out_dir=ho_dir, move=False)

    # Untouched manifest allows --force and writes a .bak.json backup
    seal_mod.seal_heldout_sessions([src_fp], out_dir=ho_dir, move=False, force=True)
    assert len(list(ho_dir.glob("sealed_manifest.*.bak.json"))) == 1

    # Once touched, --force MUST be refused!
    replay_mod.evaluate_sealed_heldout(ho_dir, model_hash="d" * 24, min_exposure_hours_for_claim=0.0)
    with pytest.raises(ValueError, match="already been touched"):
        seal_mod.seal_heldout_sessions([src_fp], out_dir=ho_dir, move=False, force=True)


def test_manifest_max_touches_enforced(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v051_02_maxt")
    seal_mod = _load_script_module("scripts/seal_heldout.py", "seal_v051_02_maxt")
    replay_mod = _load_replay_eval_module("replay_v051_02_maxt")

    h = _make_open_hand(forward_z=-0.25)
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    for i in range(10):
        col.add_frame(h, h, "Right", 0.95, 1000 + i * 33)
    src_fp = col.save(tmp / "src")

    ho_dir = tmp / "heldout_aerohand"
    seal_mod.seal_heldout_sessions([src_fp], out_dir=ho_dir, max_touches=3)

    # Invalid arbitrary string hash rejected
    with pytest.raises(ValueError, match="Invalid model_hash"):
        replay_mod.evaluate_sealed_heldout(ho_dir, model_hash="arbitrary_bypass_hash", min_exposure_hours_for_claim=0.0)

    # First 3 distinct valid 24-hex hashes succeed
    for ch in ("1", "2", "3"):
        replay_mod.evaluate_sealed_heldout(ho_dir, model_hash=ch * 24, min_exposure_hours_for_claim=0.0)

    # 4th touch MUST be refused regardless of new hash!
    with pytest.raises(ValueError, match="max_touches"):
        replay_mod.evaluate_sealed_heldout(ho_dir, model_hash="4" * 24, min_exposure_hours_for_claim=0.0)


def test_heuristic_model_hash_changes_with_source_version():
    replay_mod = _load_replay_eval_module("replay_v051_02_hhash")
    cfg = load_config()
    h_v1 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg, heuristic_source_bytes=b"v1_code")
    h_v2 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg, heuristic_source_bytes=b"v2_code")
    assert len(h_v1) == 24 and len(h_v2) == 24
    assert h_v1 != h_v2


def test_ab_eval_negatives_overlap_with_train_rejected(tmp_path: Path):
    import shutil

    train_mod = _load_train_tcn_module("train_v051_02_abov")
    f1 = tmp_path / "train_neg.json"
    f1.write_text('{"samples": []}', encoding="utf-8")
    f1_copy = tmp_path / "eval_neg_copy.json"
    shutil.copy2(str(f1), str(f1_copy))
    f2 = tmp_path / "eval_neg_disjoint.json"
    f2.write_text('{"samples": [1]}', encoding="utf-8")

    # Disjoint passes cleanly
    train_mod.validate_disjoint_ab_negatives([f1], [f2])
    # Content-identical copy MUST raise ValueError
    with pytest.raises(ValueError, match="overlaps with training negatives"):
        train_mod.validate_disjoint_ab_negatives([f1], [f1_copy])



def test_latency_guard_ignores_cold_first_inference(tmp_path: Path, monkeypatch):
    import json

    import onnxruntime as ort

    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier, ONNXPinchClassifier

    norm_fp = tmp_path / "norm_wu.json"
    norm_fp.write_text(
        json.dumps(
            {
                "feature_set": "invariant348",
                "sequence_length": 5,
                "mean": [0.0] * FEATURE_DIM_3D,
                "scale": [1.0] * FEATURE_DIM_3D,
            }
        ),
        encoding="utf-8",
    )
    model_fp = tmp_path / "wu.onnx"
    model_fp.write_bytes(b"placeholder")

    run_calls = {"n": 0}

    class _WarmupOrt:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def get_inputs(self):
            class _In:
                name = "input"

            return [_In()]

        def run(self, _out, _feed):
            run_calls["n"] += 1
            return [np.array([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _WarmupOrt)

    # _load() MUST execute 20 warm-up inferences before any live prediction
    ONNXPinchClassifier(model_fp, norm_fp, sequence_length=5, warmup_runs=20)
    assert run_calls["n"] == 20

    # Even if first live inference is slow (25 ms cold spike) followed by 29 fast (1.0 ms) calls,
    # p95 over 30 samples is <= 8.0 ms so GuardedClassifier MUST NOT fall back!
    class _ColdSpikeClf:
        def __init__(self) -> None:
            self.n = 0

        def predict(self, *args, **kwargs):
            self.n += 1
            ms = 25.0 if self.n == 1 else 1.0
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high", ready=True, infer_ms=ms)

        def reset(self, hand=None):
            pass

    guarded = GuardedClassifier(
        _ColdSpikeClf(), Heuristic3DPinchClassifier(), mode="onnx", max_p95_ms=8.0, min_latency_samples=30
    )
    h = _make_open_hand()
    for i in range(30):
        guarded.predict(h, h, "Right", i * 33.0)
    assert guarded.fell_back is False
    assert guarded.telemetry()["onnx_p95_ms"] < 8.0


def test_latency_guard_requires_30_real_samples():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _SlowClf:
        def predict(self, *args, **kwargs):
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high", ready=True, infer_ms=12.0)

        def reset(self, hand=None):
            pass

    guarded = GuardedClassifier(
        _SlowClf(), Heuristic3DPinchClassifier(), mode="onnx", max_p95_ms=8.0, min_latency_samples=30
    )
    h = _make_open_hand()
    # First 29 slow frames MUST NOT trigger premature fallback yet
    for i in range(29):
        guarded.predict(h, h, "Right", i * 33.0)
    assert guarded.fell_back is False
    # 30th slow frame pushes p95=12.0 > 8.0 -> triggers auto-fallback!
    guarded.predict(h, h, "Right", 29 * 33.0)
    assert guarded.fell_back is True
    assert guarded.fallback_reason == "onnx_fallback_latency"


def test_latency_guard_excludes_buffer_fill_frames():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _WarmingClf:
        def __init__(self) -> None:
            self.n = 0

        def predict(self, *args, **kwargs):
            self.n += 1
            if self.n <= 4:
                return PinchPrediction(
                    PoseName.IDLE, idle_probabilities(), "low", ready=False, infer_ms=None, feature_ms=0.2
                )
            return PinchPrediction(
                PoseName.IDLE, idle_probabilities(), "high", ready=True, infer_ms=1.5, feature_ms=0.3
            )

        def reset(self, hand=None):
            pass

    guarded = GuardedClassifier(_WarmingClf(), Heuristic3DPinchClassifier(), mode="onnx", max_p95_ms=8.0)
    h = _make_open_hand()
    for i in range(4):
        guarded.predict(h, h, "Right", i * 33.0)
    # 4 unready buffer-fill frames MUST contribute 0 latency samples!
    assert guarded.telemetry()["onnx_latency_samples"] == 0
    guarded.predict(h, h, "Right", 4 * 33.0)
    assert guarded.telemetry()["onnx_latency_samples"] == 1
    assert guarded.telemetry()["onnx_p95_ms"] == pytest.approx(1.5, abs=0.01)


def test_rearm_clears_fallback_state():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _SlowThenFast:
        def __init__(self) -> None:
            self.ms = 15.0

        def predict(self, *args, **kwargs):
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high", ready=True, infer_ms=self.ms)

        def reset(self, hand=None):
            pass

    clf = _SlowThenFast()
    guarded = GuardedClassifier(
        clf, Heuristic3DPinchClassifier(), mode="onnx", max_p95_ms=8.0, min_latency_samples=10
    )
    h = _make_open_hand()
    for i in range(10):
        guarded.predict(h, h, "Right", i * 33.0)
    assert guarded.fell_back is True

    clf.ms = 1.2
    guarded.rearm()
    assert guarded.fell_back is False
    assert guarded.fallback_reason is None
    assert guarded.telemetry()["onnx_latency_samples"] == 0
    guarded.predict(h, h, "Right", 500.0)
    assert guarded.fell_back is False


def test_veto_abstains_while_onnx_buffer_warming():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _AlwaysPinch:
        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_MIDDLE_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_MIDDLE_PINCH, probs, "high", ready=True)

        def reset(self, hand=None):
            pass

    class _NotReadyOnnx:
        def predict(self, *args, **kwargs):
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "low", ready=False)

        def reset(self, hand=None):
            pass

    guarded = GuardedClassifier(_NotReadyOnnx(), _AlwaysPinch(), mode="veto")
    h = _make_open_hand()
    for i in range(4):
        pred = guarded.predict(h, h, "Right", i * 33.0)
        assert pred.pose == PoseName.THUMB_MIDDLE_PINCH
    tel = guarded.telemetry()
    assert tel["veto_count"] == 0
    assert tel["veto_abstain_not_ready"] == 4


def test_veto_passes_heuristic_after_single_frame_dropout_reset(tmp_path: Path, monkeypatch):
    import json

    import onnxruntime as ort

    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier, ONNXPinchClassifier

    norm_fp = tmp_path / "norm_drop.json"
    norm_fp.write_text(
        json.dumps(
            {
                "feature_set": "invariant348",
                "sequence_length": 5,
                "mean": [0.0] * FEATURE_DIM_3D,
                "scale": [1.0] * FEATURE_DIM_3D,
            }
        ),
        encoding="utf-8",
    )
    model_fp = tmp_path / "drop.onnx"
    model_fp.write_bytes(b"placeholder")

    class _ApproveOrt:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def get_inputs(self):
            class _In:
                name = "input"

            return [_In()]

        def run(self, _out, _feed):
            # High logit for class 2 (THUMB_MIDDLE_PINCH)
            return [np.array([[0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _ApproveOrt)

    class _HeurClick:
        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_MIDDLE_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_MIDDLE_PINCH, probs, "high")

        def reset(self, hand=None):
            pass

    onnx = ONNXPinchClassifier(model_fp, norm_fp, sequence_length=5, warmup_runs=1)
    guarded = GuardedClassifier(onnx, _HeurClick(), mode="veto")
    h = _make_open_hand()
    # Fill buffer, then simulate 1-frame MediaPipe dropout -> reset("Right")
    for i in range(6):
        guarded.predict(h, h, "Right", i * 33.0)
    guarded.reset("Right")
    # Immediately after dropout reset, the 4 warming frames MUST pass heuristic through (not veto!)
    for i in range(4):
        pred = guarded.predict(h, h, "Right", 300.0 + i * 33.0)
        assert pred.pose == PoseName.THUMB_MIDDLE_PINCH
    assert guarded.telemetry()["veto_count"] == 0
    assert guarded.telemetry()["veto_abstain_not_ready"] >= 4


def test_feature_dim_mismatch_counts_as_error_and_falls_back(tmp_path: Path, monkeypatch):
    import json

    import onnxruntime as ort

    from aerohand.ml.onnx_classifier import GuardedClassifier, ONNXPinchClassifier

    # Normalizer claims invariant348 but wrong mean length (100 != 348)
    norm_fp = tmp_path / "norm_bad.json"
    norm_fp.write_text(
        json.dumps(
            {
                "feature_set": "invariant348",
                "sequence_length": 5,
                "mean": [0.0] * 100,
                "scale": [1.0] * 100,
            }
        ),
        encoding="utf-8",
    )
    model_fp = tmp_path / "bad.onnx"
    model_fp.write_bytes(b"placeholder")

    class _DummyOrt:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def get_inputs(self):
            class _In:
                name = "input"

            return [_In()]

        def run(self, _out, _feed):
            return [np.zeros((1, 7), dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _DummyOrt)
    onnx = ONNXPinchClassifier(model_fp, norm_fp, sequence_length=5, warmup_runs=1)
    guarded = GuardedClassifier(onnx, Heuristic3DPinchClassifier(), mode="veto", max_errors=3)
    h = _make_open_hand()
    for i in range(3):
        guarded.predict(h, h, "Right", i * 33.0)
    assert guarded.telemetry()["onnx_errors"] == 3
    assert guarded.fell_back is True
    assert guarded.fallback_reason == "onnx_fallback_errors"


def test_veto_does_not_interrupt_active_hold():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _HeurDrag:
        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_INDEX_MIDDLE_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_INDEX_MIDDLE_PINCH, probs, "high")

        def reset(self, hand=None):
            pass

    class _FlipOnnx:
        def __init__(self) -> None:
            self.approve = True

        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            if self.approve:
                probs[PoseName.IDLE] = 0.10
                probs[PoseName.THUMB_INDEX_MIDDLE_PINCH] = 0.90
                return PinchPrediction(PoseName.THUMB_INDEX_MIDDLE_PINCH, probs, "high", ready=True)
            return PinchPrediction(PoseName.IDLE, probs, "high", ready=True)

        def reset(self, hand=None):
            pass

    onnx = _FlipOnnx()
    guarded = GuardedClassifier(
        onnx, _HeurDrag(), mode="veto", veto_scope="entry", veto_entry_window_ms=300.0, veto_confirm_frames=3
    )
    h = _make_open_hand()
    # Entry frames approved consecutively at t=0, 33, 66 -> accepted on 3rd frame
    assert guarded.predict(h, h, "Right", 0.0).pose == PoseName.IDLE
    assert guarded.predict(h, h, "Right", 33.0).pose == PoseName.IDLE
    assert guarded.predict(h, h, "Right", 66.0).pose == PoseName.THUMB_INDEX_MIDDLE_PINCH
    # Mid-drag ONNX flickers to IDLE at t=100..600ms -> entry-only veto MUST NOT interrupt active drag!
    onnx.approve = False
    for ts in (100.0, 250.0, 400.0, 600.0):
        assert guarded.predict(h, h, "Right", ts).pose == PoseName.THUMB_INDEX_MIDDLE_PINCH
    assert guarded.telemetry()["veto_count"] == 0


def test_veto_count_counts_events_not_frames():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _HeurToggle:
        def __init__(self) -> None:
            self.active = True

        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            if self.active:
                probs[PoseName.IDLE] = 0.05
                probs[PoseName.THUMB_INDEX_PINCH] = 0.95
                return PinchPrediction(PoseName.THUMB_INDEX_PINCH, probs, "high")
            return PinchPrediction(PoseName.IDLE, probs, "high")

        def reset(self, hand=None):
            pass

    class _RejectOnnx:
        def predict(self, *args, **kwargs):
            return PinchPrediction(PoseName.IDLE, idle_probabilities(), "high", ready=True)

        def reset(self, hand=None):
            pass

    heur = _HeurToggle()
    guarded = GuardedClassifier(_RejectOnnx(), heur, mode="veto", veto_scope="entry")
    h = _make_open_hand()
    # 8 consecutive frames of ONE continuous vetoed pinch attempt -> veto_count == 1, veto_frames == 8
    for i in range(8):
        assert guarded.predict(h, h, "Right", i * 33.0).pose == PoseName.IDLE
    tel = guarded.telemetry()
    assert tel["veto_count"] == 1
    assert tel["veto_frames"] == 8

    # Release to IDLE then start a 2nd attempt within 1s -> veto_count == 2, veto_retried_within_1s == 1
    heur.active = False
    guarded.predict(h, h, "Right", 300.0)
    heur.active = True
    for i in range(3):
        assert guarded.predict(h, h, "Right", 350.0 + i * 33.0).pose == PoseName.IDLE
    tel2 = guarded.telemetry()
    assert tel2["veto_count"] == 2
    assert tel2["veto_retried_within_1s"] == 1


def test_veto_error_passes_heuristic_through_by_default():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _HeurClick:
        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_MIDDLE_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_MIDDLE_PINCH, probs, "high")

        def reset(self, hand=None):
            pass

    class _CrashOnnx:
        def predict(self, *args, **kwargs):
            raise RuntimeError("transient onnx error")

        def reset(self, hand=None):
            pass

    releases: List[int] = []
    guarded = GuardedClassifier(
        _CrashOnnx(),
        _HeurClick(),
        mode="veto",
        veto_on_error="pass_through",
        on_error=lambda: releases.append(1),
    )
    h = _make_open_hand()
    pred = guarded.predict(h, h, "Right", 0.0)
    # Default veto_on_error="pass_through" MUST preserve heuristic click and NOT force release_all
    assert pred.pose == PoseName.THUMB_MIDDLE_PINCH
    assert releases == []
    assert guarded.telemetry()["onnx_errors"] == 1


def test_veto_scope_always_preserves_legacy_behavior():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier

    class _HeurDrag:
        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.05
            probs[PoseName.THUMB_INDEX_MIDDLE_PINCH] = 0.95
            return PinchPrediction(PoseName.THUMB_INDEX_MIDDLE_PINCH, probs, "high")

        def reset(self, hand=None):
            pass

    class _FlipOnnx:
        def __init__(self) -> None:
            self.approve = True

        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            if self.approve:
                probs[PoseName.IDLE] = 0.10
                probs[PoseName.THUMB_INDEX_MIDDLE_PINCH] = 0.90
                return PinchPrediction(PoseName.THUMB_INDEX_MIDDLE_PINCH, probs, "high", ready=True)
            return PinchPrediction(PoseName.IDLE, probs, "high", ready=True)

        def reset(self, hand=None):
            pass

    onnx = _FlipOnnx()
    guarded = GuardedClassifier(onnx, _HeurDrag(), mode="veto", veto_scope="always", veto_confirm_frames=1)
    h = _make_open_hand()
    assert guarded.predict(h, h, "Right", 0.0).pose == PoseName.THUMB_INDEX_MIDDLE_PINCH
    onnx.approve = False
    # Under veto_scope="always", mid-hold rejection STILL vetoes immediately (legacy behavior)
    assert guarded.predict(h, h, "Right", 500.0).pose == PoseName.IDLE


def test_heuristic_config_update_keeps_state_and_counters():
    from aerohand.ml.base import PinchPrediction
    from aerohand.ml.onnx_classifier import GuardedClassifier
    from aerohand.runtime import AeroHandRuntime

    class _StubOnnx:
        def predict(self, *args, **kwargs):
            probs = idle_probabilities()
            probs[PoseName.IDLE] = 0.1
            probs[PoseName.THUMB_PINKY_PINCH] = 0.9
            return PinchPrediction(PoseName.THUMB_PINKY_PINCH, probs, "high", ready=True, infer_ms=1.0)

        def reset(self, hand=None):
            pass

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["classifier"]["mode"] = "shadow"
    rt = AeroHandRuntime(cfg, dry_run=True)
    heur = Heuristic3DPinchClassifier(cfg["classifier"]["heuristic_3d"])
    guarded = GuardedClassifier(_StubOnnx(), heur, mode="shadow")
    rt.classifier = guarded
    rt.engine.set_classifier(guarded)

    open_h = _make_open_hand()
    guarded.predict(open_h, open_h, "Right", 0.0)
    assert guarded.telemetry()["shadow_frames"] == 1
    assert "Right" in heur._hands

    # Tweaking heuristic_3d slider MUST update in-place without rebuilding or wiping shadow counters/state!
    new_cfg = deep_merge(cfg, {"classifier": {"heuristic_3d": {"pinch_on_dist": 0.36}}})
    rt._apply_live_settings_now(new_cfg)
    assert rt.classifier is guarded
    assert heur.pinch_on_dist == pytest.approx(0.36)
    assert "Right" in heur._hands
    assert guarded.telemetry()["shadow_frames"] == 1


def test_set_classifier_swap_atomic_under_concurrent_predict():
    import threading

    cfg = load_config()
    cfg["smoothing"]["enable_landmark_smoothing"] = False
    cfg["cursor"]["reacquire_skip_frames"] = 0
    cfg["palm_wheel"]["enabled"] = False
    eng = HandCursorEngine(cfg, classifier=Heuristic3DPinchClassifier())
    h = _make_open_hand(forward_z=-0.25)
    errors: List[Exception] = []
    stop = threading.Event()

    def _swapper() -> None:
        try:
            for _ in range(2000):
                eng.set_classifier(Heuristic3DPinchClassifier())
        except Exception as exc:
            errors.append(exc)
        finally:
            stop.set()

    def _predictor() -> None:
        ts = 0.0
        try:
            while not stop.is_set() and ts < 2000 * 33.0:
                eng.process_frame([HandInput("Right", h, h)], ts, 1920.0, 1080.0)
                ts += 33.0
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_swapper)
    t2 = threading.Thread(target=_predictor)
    t1.start()
    t2.start()
    t1.join(timeout=15.0)
    t2.join(timeout=15.0)
    assert not t1.is_alive() and not t2.is_alive()
    assert errors == []


def test_rebuild_preserves_cumulative_shadow_counters(tmp_path: Path, monkeypatch):
    import json

    import onnxruntime as ort

    from aerohand.ml.onnx_classifier import GuardedClassifier
    from aerohand.runtime import AeroHandRuntime

    norm_fp = tmp_path / "norm_cum.json"
    norm_fp.write_text(
        json.dumps(
            {
                "feature_set": "invariant348",
                "sequence_length": 5,
                "mean": [0.0] * FEATURE_DIM_3D,
                "scale": [1.0] * FEATURE_DIM_3D,
            }
        ),
        encoding="utf-8",
    )
    model_fp1 = tmp_path / "m1.onnx"
    model_fp2 = tmp_path / "m2.onnx"
    model_fp1.write_bytes(b"m1")
    model_fp2.write_bytes(b"m2")

    class _DummyOrt:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def get_inputs(self):
            class _In:
                name = "input"

            return [_In()]

        def run(self, _out, _feed):
            return [np.array([[0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _DummyOrt)

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["classifier"]["mode"] = "shadow"
    cfg["classifier"]["onnx_model_path"] = str(model_fp1)
    cfg["classifier"]["onnx_normalizer_path"] = str(norm_fp)
    rt = AeroHandRuntime(cfg, dry_run=True)
    assert isinstance(rt.classifier, GuardedClassifier)

    h = _make_open_hand()
    for i in range(7):  # 4 warmup + 3 ready shadow frames
        rt.classifier.predict(h, h, "Right", i * 33.0)
    assert rt.classifier.telemetry()["shadow_frames"] == 3

    # Trigger full model rebuild by switching onnx_model_path -> cumulative shadow_frames MUST survive!
    cfg2 = deep_merge(cfg, {"classifier": {"onnx_model_path": str(model_fp2)}})
    rt._apply_live_settings_now(cfg2)
    assert isinstance(rt.classifier, GuardedClassifier)
    assert rt.classifier.telemetry()["shadow_frames"] == 3


# ==============================================================================
# V051-07 / V051-08: FPS- & Scale-Invariant Intent Gates + Gate Debug
# ==============================================================================


def _make_scaled_shifted_hand(scale: float, dx: float = 0.0, dy: float = 0.0) -> List[Landmark]:
    base = _make_open_hand(forward_z=-0.25)
    w = base[0]
    return [
        Landmark(
            x=w.x + (p.x - w.x) * scale + dx,
            y=w.y + (p.y - w.y) * scale + dy,
            z=w.z + (p.z - w.z) * scale,
        )
        for p in base
    ]


def _pinch_middle_copy(lms: List[Landmark]) -> List[Landmark]:
    out = list(lms)
    out[4] = Landmark(out[12].x, out[12].y, out[12].z)
    return out


def test_motion_gate_same_decision_at_30_and_60fps():
    cfg = load_config()["classifier"]["heuristic_3d"]
    pw = math.hypot(
        _make_open_hand()[5].x - _make_open_hand()[17].x,
        _make_open_hand()[5].y - _make_open_hand()[17].y,
    )

    for speed_pw, expect_blocked in ((0.8, False), (2.3, True)):
        decisions = []
        ema_vals = []
        for fps in (30.0, 60.0):
            clf = Heuristic3DPinchClassifier(cfg)
            dt_ms = 1000.0 / fps
            n_open = int(round(0.30 * fps))  # 300ms moving open hand
            for i in range(n_open):
                t_s = (i * dt_ms) / 1000.0
                h = _make_scaled_shifted_hand(1.0, dx=speed_pw * pw * t_s)
                clf.predict(h, h, "Right", i * dt_ms)
            # Pinch on next frame while still translating at the exact same physical speed
            t_s = (n_open * dt_ms) / 1000.0
            h_pinch = _pinch_middle_copy(_make_scaled_shifted_hand(1.0, dx=speed_pw * pw * t_s))
            pred = clf.predict(h_pinch, h_pinch, "Right", n_open * dt_ms)
            dbg = clf.last_debug["Right"]
            decisions.append(pred.pose == PoseName.IDLE)
            ema_vals.append(dbg["eff_speed_pw"])
        assert decisions == [expect_blocked, expect_blocked], (
            f"speed_pw={speed_pw}: 30fps vs 60fps mismatch {decisions}, ema={ema_vals}"
        )
        assert abs(ema_vals[0] - ema_vals[1]) < 0.15


def test_motion_gate_invariant_to_palm_size():
    cfg = load_config()["classifier"]["heuristic_3d"]
    base = _make_open_hand()
    base_pw = math.hypot(base[5].x - base[17].x, base[5].y - base[17].y)

    # Near hand (scale=1.6, pw~0.19) vs far hand (scale=0.55, pw~0.066):
    # At 1.0 pw/s, near image speed is 0.19 (> old 0.18) yet palm-normalized speed is 1.0 (< 1.50) -> ALLOWED!
    # At 2.1 pw/s, far image speed is 0.139 (< old 0.18) yet palm-normalized speed is 2.1 (> 1.50) -> BLOCKED!
    for speed_pw, expect_pose in ((1.0, PoseName.THUMB_MIDDLE_PINCH), (2.1, PoseName.IDLE)):
        for scale in (0.55, 1.60):
            clf = Heuristic3DPinchClassifier(cfg)
            pw = base_pw * scale
            for i in range(8):
                t_s = (i * 33.33) / 1000.0
                h = _make_scaled_shifted_hand(scale, dx=speed_pw * pw * t_s)
                clf.predict(h, h, "Right", i * 33.33)
            t_s = (8 * 33.33) / 1000.0
            h_p = _pinch_middle_copy(_make_scaled_shifted_hand(scale, dx=speed_pw * pw * t_s))
            pred = clf.predict(h_p, h_p, "Right", 8 * 33.33)
            assert pred.pose == expect_pose, (
                f"scale={scale}, speed_pw={speed_pw}: expected {expect_pose}, got {pred.pose}, dbg={clf.last_debug['Right']}"
            )


def test_approach_decay_time_based():
    cfg = dict(load_config()["classifier"]["heuristic_3d"])
    cfg["enable_motion_gate"] = False
    open_dists = []
    for fps in (30.0, 60.0):
        clf = Heuristic3DPinchClassifier(cfg)
        dt_ms = 1000.0 / fps
        base = _make_open_hand()
        # Frame 0: wide open
        clf.predict(base, base, "Right", 0.0)
        # Next 400ms: hold thumb moderately closer (still open, d_mid ~ 0.48 > 0.38)
        semi = list(base)
        semi[4] = Landmark(
            x=base[12].x - 0.055,
            y=base[12].y + 0.015,
            z=base[12].z,
        )
        steps = int(round(0.40 * fps))
        for i in range(1, steps + 1):
            clf.predict(semi, semi, "Right", i * dt_ms)
        open_dists.append(clf.last_debug["Right"]["open_dist"]["thumb_middle_pinch"])
    assert abs(open_dists[0] - open_dists[1]) < 0.04, f"FPS-dependent decay mismatch: {open_dists}"


def test_v050_regression_ddev_ft_not_worse():
    ddev_files = sorted(Path("data/converted/negatives").glob("gctl_d_dev_*.json"))
    if not ddev_files:
        pytest.skip("Local converted D-dev corpus not present in workspace")
    mod = _load_replay_eval_module("replay_eval_ddev_reg")
    rep = mod.evaluate_recordings(
        negative_paths=ddev_files,
        min_exposure_hours_for_claim=0.0,
        max_ft_per_hour=1e9,
    )
    assert rep["false_triggers"]["total"] <= 4, f"D-dev FT regressed above 4: {rep['false_triggers']}"


def test_predict_exposes_gate_debug_blocked_reasons():
    cfg = load_config()["classifier"]["heuristic_3d"]
    # 1. Static contact -> approach blocked
    clf_app = Heuristic3DPinchClassifier(cfg)
    h_pinch = _pinch_middle_copy(_make_open_hand())
    pred_app = clf_app.predict(h_pinch, h_pinch, "Right", 100.0)
    assert pred_app.pose == PoseName.IDLE
    dbg_app = clf_app.last_debug["Right"]
    assert "approach" in dbg_app["blocked_reasons"]
    assert "approach" in dbg_app["blocked_by"].get("thumb_middle_pinch", [])

    # 2. Fast wrist swipe -> motion blocked
    clf_mot = Heuristic3DPinchClassifier(cfg)
    h0 = _make_open_hand()
    clf_mot.predict(h0, h0, "Right", 0.0)
    h1 = _pinch_middle_copy(_make_scaled_shifted_hand(1.0, dx=0.05))
    pred_mot = clf_mot.predict(h1, h1, "Right", 33.0)
    assert pred_mot.pose == PoseName.IDLE
    dbg_mot = clf_mot.last_debug["Right"]
    assert "motion" in dbg_mot["blocked_reasons"]
    assert dbg_mot["eff_speed_pw"] > clf_mot.max_wrist_speed_pw
    assert dbg_mot["palm_width_2d"] > 0.0


def test_ft_jsonl_contains_gate_debug(tmp_path: Path):
    import json

    mod = _load_replay_eval_module("replay_eval_gate_dbg")
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    h_open = _make_open_hand(forward_z=-0.25)
    h_pinch = _pinch_middle_copy(h_open)
    # 6 open frames -> 3 pinched frames -> 6 open frames => intentional-looking left click on idle file
    for i in range(6):
        col.add_frame(h_open, h_open, "Right", 0.95, 1000 + i * 33)
    for i in range(3):
        col.add_frame(h_pinch, h_pinch, "Right", 0.95, 1000 + (6 + i) * 33)
    for i in range(6):
        col.add_frame(h_open, h_open, "Right", 0.95, 1000 + (9 + i) * 33)
    neg_fp = col.save(tmp_path)

    jsonl_fp = tmp_path / "ft_gate_dbg.jsonl"
    cfg = load_config()
    cfg["palm_wheel"]["enabled"] = False
    rep = mod.evaluate_recordings(
        negative_paths=[neg_fp],
        config=cfg,
        ft_jsonl_path=jsonl_fp,
        min_exposure_hours_for_claim=0.0,
        max_ft_per_hour=1e9,
    )
    assert rep["false_triggers"]["total"] >= 1
    lines = [json.loads(ln) for ln in jsonl_fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) >= 1
    gd = lines[0].get("gate_debug")
    assert isinstance(gd, dict)
    for k in ("latched", "blocked_by", "blocked_reasons", "eff_speed_pw", "eff_ang_deg_s", "palm_width_2d", "open_dist"):
        assert k in gd


# ==============================================================================
# V051-10: Hardened train_tcn Split, Augmentation & Provenance
# ==============================================================================


def _mk_tcn_bundle(mod, name: str, group_key: str, class_ids: List[int]):
    h = _make_open_hand()
    wins = [[mod.ParsedSample(cid, h, h, "Right", "Right", float(i * 33)) for i in range(5)] for cid in class_ids]
    return mod.SessionBundle(name=name, group_key=group_key, windows=wins, labels=np.array(class_ids, dtype=np.int64))


def test_split_never_leaves_class_without_train_windows():
    mod = _load_train_tcn_module("train_tcn_v051_10_split")
    # Class 5 appears ONLY in group g2; class 6 appears ONLY in g3 -> greedy val MUST NOT steal g2 or g3!
    b1 = _mk_tcn_bundle(mod, "s1.json", "minh::g1", [0, 1, 2, 3, 4])
    b2 = _mk_tcn_bundle(mod, "s2.json", "minh::g2", [0, 1, 2, 3, 4, 5])
    b3 = _mk_tcn_bundle(mod, "s3.json", "minh::g3", [0, 1, 2, 3, 4, 6])
    b4 = _mk_tcn_bundle(mod, "s4.json", "minh::g4", [0, 1, 2, 3, 4])

    tr, va, _missing = mod.split_session_groups([b1, b2, b3, b4], val_ratio=0.25)
    tr_classes = {int(c) for b in tr for c in b.labels.tolist()}
    # Every class present anywhere in input MUST still be represented in train!
    assert {0, 1, 2, 3, 4, 5, 6}.issubset(tr_classes)
    assert set(b.group_key for b in tr).isdisjoint(set(b.group_key for b in va))


def test_single_group_fallback_is_class_stratified():
    mod = _load_train_tcn_module("train_tcn_v051_10_single")
    # Single group recorded sequentially class-by-class: 0,0..0, 1,1..1, ..., 6,6..6
    seq_labels: List[int] = []
    for cid in range(7):
        seq_labels.extend([cid] * 25)
    b = _mk_tcn_bundle(mod, "single.json", "minh::only_group", seq_labels)

    tr, va, missing = mod.split_session_groups([b], val_ratio=0.2)
    assert missing == []
    tr_classes = {int(c) for c in tr[0].labels.tolist()}
    va_classes = {int(c) for c in va[0].labels.tolist()}
    assert tr_classes == set(range(7))
    assert va_classes == set(range(7))
    # Verify 5-window temporal gap dropped per class: 25 * 7 = 175 total -> 7 * 5 = 35 gap windows dropped
    assert len(tr[0].labels) + len(va[0].labels) <= 175 - 7 * 5


def test_validate_train_classes_min_windows():
    mod = _load_train_tcn_module("train_tcn_v051_10_minwin")
    # Missing class 6 -> raises even when allow_low_count=True
    b_miss = _mk_tcn_bundle(mod, "miss.json", "minh::g1", [0, 1, 2, 3, 4, 5] * 60)
    with pytest.raises(ValueError, match="0 windows"):
        mod.validate_train_classes([b_miss], min_windows=10, allow_low_count=True)

    # All 7 classes present, but class 6 has only 50 < 300 -> raises when allow_low_count=False
    labels = [c for c in range(6) for _ in range(320)] + [6] * 50
    b_low = _mk_tcn_bundle(mod, "low.json", "minh::g2", labels)
    with pytest.raises(ValueError, match="< 300"):
        mod.validate_train_classes([b_low], min_windows=300, allow_low_count=False)

    # All 7 classes >= 300 -> passes cleanly
    labels_ok = [c for c in range(7) for _ in range(305)]
    b_ok = _mk_tcn_bundle(mod, "ok.json", "minh::g3", labels_ok)
    counts = mod.validate_train_classes([b_ok], min_windows=300, allow_low_count=False)
    assert all(v == 305 for v in counts.values())


def test_dev_test_split_disjoint_from_train_val_by_group():
    mod = _load_train_tcn_module("train_tcn_v051_10_devtest")
    all_7 = list(range(7))
    bundles = [_mk_tcn_bundle(mod, f"s{i}.json", f"minh::cond_{i}", all_7) for i in range(5)]

    tv, dev_test = mod.split_dev_test_groups(bundles, test_ratio=0.2)
    assert len(dev_test) >= 1
    tr, va, missing = mod.split_session_groups(tv, val_ratio=0.25)
    assert missing == []

    g_tr = {b.group_key for b in tr}
    g_va = {b.group_key for b in va}
    g_dt = {b.group_key for b in dev_test}
    assert g_tr.isdisjoint(g_va)
    assert g_tr.isdisjoint(g_dt)
    assert g_va.isdisjoint(g_dt)


def test_augment_noise_scales_with_palm_width():
    mod = _load_train_tcn_module("train_tcn_v051_10_augscale")
    # Compare frame-to-frame jitter std on small hand (scale=0.35) vs large hand (scale=2.0)
    # Using a static window so frame-to-frame diff isolates ONLY per-frame noise!
    small_h = _make_scaled_shifted_hand(0.35)
    large_h = _make_scaled_shifted_hand(2.00)
    win_small = [mod.ParsedSample(1, small_h, small_h, "Right", "Right", float(i * 33)) for i in range(5)]
    win_large = [mod.ParsedSample(1, large_h, large_h, "Right", "Right", float(i * 33)) for i in range(5)]

    def _mean_frame_jitter(win):
        rng = np.random.default_rng(77)
        diffs = []
        for _ in range(40):
            aug = mod.augment_window(win, rng)
            arr = np.array([[[p.x, p.y, p.z] for p in s.lms2d] for s in aug], dtype=np.float64)
            # Adjacent frame diffs cancel shared window noise (except duplicated jitter frames which are 0)
            step = np.abs(np.diff(arr, axis=0))
            nz = step[step > 1e-12]
            if len(nz):
                diffs.append(float(np.mean(nz)))
        return float(np.mean(diffs))

    j_small = _mean_frame_jitter(win_small)
    j_large = _mean_frame_jitter(win_large)
    ratio = j_large / max(1e-9, j_small)
    # Expected ratio ~ 2.0 / 0.35 = 5.71 (with fixed sigma=0.003, ratio would be ~1.0!)
    assert ratio > 3.5, f"Augmentation noise did not scale with palm_width: ratio={ratio:.2f}"


def test_metrics_not_overwritten_without_set_latest(tmp_path: Path):
    import json

    mod = _load_train_tcn_module("train_tcn_v051_10_latest")
    onnx_fp = tmp_path / "run_a.onnx"
    onnx_fp.write_bytes(b"fake-onnx-run-a")
    sentinel = tmp_path / "metrics.json"
    sentinel.write_text(json.dumps({"sentinel": "preserve_me"}), encoding="utf-8")

    # Default set_latest=False MUST NOT overwrite metrics.json
    mod.write_training_artifacts(
        out_dir=tmp_path,
        name="run_a",
        onnx_path=onnx_fp,
        feature_set="invariant348",
        feat_dim=FEATURE_DIM_3D,
        seq_len=5,
        hidden=96,
        loss="focal",
        augmented=True,
        mean=np.zeros(FEATURE_DIM_3D, dtype=np.float32),
        scale=np.ones(FEATURE_DIM_3D, dtype=np.float32),
        temperature=1.0,
        train_sequences=100,
        val_sequences=20,
        best_metrics={"val_accuracy": 0.95, "macro_f1": 0.94, "idle_fp_rate": 0.01},
        balance_stats={},
        dev_test_metrics=None,
        subject_ho_metrics=None,
        onnx_parity_samples=20,
        onnx_max_diff=1e-6,
        seed=42,
        data_files_sha256_list=[],
        set_latest=False,
    )
    assert (tmp_path / "run_a_metrics.json").is_file()
    assert json.loads(sentinel.read_text(encoding="utf-8")) == {"sentinel": "preserve_me"}

    # With set_latest=True, metrics.json IS updated
    mod.write_training_artifacts(
        out_dir=tmp_path,
        name="run_a",
        onnx_path=onnx_fp,
        feature_set="invariant348",
        feat_dim=FEATURE_DIM_3D,
        seq_len=5,
        hidden=96,
        loss="focal",
        augmented=True,
        mean=np.zeros(FEATURE_DIM_3D, dtype=np.float32),
        scale=np.ones(FEATURE_DIM_3D, dtype=np.float32),
        temperature=1.0,
        train_sequences=100,
        val_sequences=20,
        best_metrics={"val_accuracy": 0.95, "macro_f1": 0.94, "idle_fp_rate": 0.01},
        balance_stats={},
        dev_test_metrics=None,
        subject_ho_metrics=None,
        onnx_parity_samples=20,
        onnx_max_diff=1e-6,
        seed=42,
        data_files_sha256_list=[],
        set_latest=True,
    )
    assert json.loads(sentinel.read_text(encoding="utf-8"))["model_name"] == "run_a"


def test_export_records_model_sha256(tmp_path: Path):
    import json

    from aerohand.ml.recording_io import sha256_file

    mod = _load_train_tcn_module("train_tcn_v051_10_prov")
    replay_mod = _load_replay_eval_module("replay_eval_v051_10_prov")

    onnx_fp = tmp_path / "prov_model.onnx"
    onnx_fp.write_bytes(b"onnx-payload-bytes-v051")
    norm_fp, metrics_fp, payload = mod.write_training_artifacts(
        out_dir=tmp_path,
        name="prov_model",
        onnx_path=onnx_fp,
        feature_set="invariant348",
        feat_dim=FEATURE_DIM_3D,
        seq_len=5,
        hidden=96,
        loss="focal",
        augmented=True,
        mean=np.zeros(FEATURE_DIM_3D, dtype=np.float32),
        scale=np.ones(FEATURE_DIM_3D, dtype=np.float32),
        temperature=1.1,
        train_sequences=200,
        val_sequences=50,
        best_metrics={"val_accuracy": 0.98, "macro_f1": 0.97, "idle_fp_rate": 0.005},
        balance_stats={},
        dev_test_metrics=None,
        subject_ho_metrics=None,
        onnx_parity_samples=50,
        onnx_max_diff=2e-6,
        seed=1337,
        data_files_sha256_list=[{"file": "s1.json", "sha256": "0" * 64}],
        torch_version="2.5.0-test",
    )
    expected_sha = sha256_file(onnx_fp)
    expected_hash = replay_mod.compute_eval_model_hash(
        "onnx_tcn", load_config(), model_path=onnx_fp, normalizer_path=norm_fp
    )
    norm_data = json.loads(norm_fp.read_text(encoding="utf-8"))
    met_data = json.loads(metrics_fp.read_text(encoding="utf-8"))

    assert norm_data["model_sha256"] == expected_sha
    assert norm_data["seed"] == 1337
    assert met_data["model_sha256"] == expected_sha
    assert met_data["normalizer_sha256"] == sha256_file(norm_fp)
    assert met_data["model_hash"] == expected_hash == payload["model_hash"]
    assert met_data["data_files_sha256_list"] == [{"file": "s1.json", "sha256": "0" * 64}]
    assert met_data["torch_version"] == "2.5.0-test"


# ==============================================================================
# V051-11: Provenance-Bound Promotion Checker & veto vs onnx Stages
# ==============================================================================


def test_promotion_rejects_metrics_from_different_model_hash(tmp_path: Path):
    from aerohand.core.types import ALL_POSE_NAMES

    promo_mod = _load_script_module("scripts/check_onnx_promotion.py", "check_onnx_promo_v051_prov")
    all_high = {p.value: {"precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 500} for p in ALL_POSE_NAMES}
    cfg = load_config()

    metrics_a = {
        "model_hash": "a" * 24,
        "model_sha256": "1" * 64,
        "subject_heldout_metrics": {"per_class": all_high},
    }
    sealed_b = {
        "model_hash": "b" * 24,
        "model_sha256": "2" * 64,
        "heldout_audit": {"touched": 1, "model_hash": "b" * 24, "model_sha256": "2" * 64},
        "false_triggers": {"hand_present_idle_hours": 1.60, "total_upper_bound_per_hour": 1.87},
    }
    res_mismatch = promo_mod.check_onnx_promotion(
        metrics_report=metrics_a,
        sealed_eval_report=sealed_b,
        latency_p95_ms=1.2,
        config=cfg,
    )
    assert res_mismatch["passed"] is False
    assert res_mismatch["criteria"]["provenance_hash_match"]["passed"] is False

    # Missing provenance hash altogether must also fail!
    res_missing = promo_mod.check_onnx_promotion(
        metrics_report={"subject_heldout_metrics": {"per_class": all_high}},
        sealed_eval_report=sealed_b,
        latency_p95_ms=1.2,
        config=cfg,
    )
    assert res_missing["passed"] is False
    assert res_missing["criteria"]["provenance_hash_match"]["passed"] is False

    # Mismatch against actual --model file bytes must also fail!
    fake_model = tmp_path / "wrong.onnx"
    fake_model.write_bytes(b"different-model-bytes")
    sealed_a = {
        "model_hash": "a" * 24,
        "model_sha256": "1" * 64,
        "heldout_audit": {"touched": 1, "model_hash": "a" * 24, "model_sha256": "1" * 64},
        "false_triggers": {"hand_present_idle_hours": 1.60, "total_upper_bound_per_hour": 1.87},
    }
    res_file_mismatch = promo_mod.check_onnx_promotion(
        metrics_report=metrics_a,
        sealed_eval_report=sealed_a,
        latency_p95_ms=1.2,
        config=cfg,
        model_path=fake_model,
    )
    assert res_file_mismatch["passed"] is False
    assert res_file_mismatch["criteria"]["provenance_hash_match"]["passed"] is False


def test_promotion_stage_veto_does_not_require_subject_heldout():
    promo_mod = _load_script_module("scripts/check_onnx_promotion.py", "check_onnx_promo_v051_veto")
    cfg = load_config()
    metrics_no_ho = {"model_hash": "c" * 24, "subject_heldout_metrics": None}
    sealed_ok = {
        "model_hash": "c" * 24,
        "heldout_audit": {"touched": 1, "model_hash": "c" * 24},
        "false_triggers": {
            "hand_present_idle_hours": 1.60,
            "total_upper_bound_per_hour": 1.87,
            "by_type": {
                "left_click": 0,
                "right_click": 0,
                "drag_start": 0,
                "scroll_stroke": 0,
                "palm_wheel": 0,
                "window_tile": 0,
                "spaces_nav": 0,
                "dictation_start": 0,
            },
        },
    }

    res_veto = promo_mod.check_onnx_promotion(
        metrics_report=metrics_no_ho,
        sealed_eval_report=sealed_ok,
        latency_p95_ms=1.3,
        config=cfg,
        stage="veto",
    )
    assert res_veto["passed"] is True
    assert res_veto["stage"] == "veto"
    assert res_veto["criteria"]["subject_heldout_recall"]["skipped"] is True

    res_onnx = promo_mod.check_onnx_promotion(
        metrics_report=metrics_no_ho,
        sealed_eval_report=sealed_ok,
        latency_p95_ms=1.3,
        config=cfg,
        stage="onnx",
    )
    assert res_onnx["passed"] is False
    assert res_onnx["criteria"]["subject_heldout_recall"]["passed"] is False


def test_required_exposure_hours_matches_garwood_table():
    replay_mod = _load_replay_eval_module("replay_eval_v051_11_garwood")
    assert abs(replay_mod.required_exposure_hours(0, max_ucb=2.0) - 1.4979) < 0.01
    assert abs(replay_mod.required_exposure_hours(1, max_ucb=2.0) - 2.3719) < 0.01
    assert abs(replay_mod.required_exposure_hours(2, max_ucb=2.0) - 3.1479) < 0.01


def test_promotion_zero_ft_at_1p5h_borderline_passes():
    from aerohand.core.types import ALL_POSE_NAMES

    promo_mod = _load_script_module("scripts/check_onnx_promotion.py", "check_onnx_promo_v051_border")
    replay_mod = _load_replay_eval_module("replay_eval_v051_11_border")
    all_high = {p.value: {"precision": 0.99, "recall": 0.985, "f1": 0.987, "support": 500} for p in ALL_POSE_NAMES}
    cfg = load_config()

    ucb_0_at_1p5 = replay_mod.poisson_upper_bound_rate(0, 1.5)
    assert ucb_0_at_1p5 <= 2.0  # 2.9957 / 1.5 = 1.997 <= 2.0

    res = promo_mod.check_onnx_promotion(
        metrics_report={"model_hash": "d" * 24, "subject_heldout_metrics": {"per_class": all_high}},
        sealed_eval_report={
            "model_hash": "d" * 24,
            "heldout_audit": {"touched": 1, "model_hash": "d" * 24},
            "false_triggers": {
                "total": 0,
                "hand_present_idle_hours": 1.5,
                "total_upper_bound_per_hour": ucb_0_at_1p5,
                "by_type": {k: 0 for k in replay_mod.ALL_TRIGGER_TYPES},
            },
        },
        latency_p95_ms=1.5,
        config=cfg,
        stage="onnx",
    )
    assert res["passed"] is True
    assert "requires 0 FT" in res["exposure_note"]
    assert "k=0" in res["garwood_exposure_table"]


# ==============================================================================
# V051-12: Hygiene (recording_io parent scan + control_api body limit)
# ==============================================================================


def test_collect_json_paths_skips_heldout_dir_when_scanning_parent(tmp_path: Path, caplog):
    from aerohand.ml.recording_io import collect_json_paths

    good_dir = tmp_path / "sessions"
    ho_dir = tmp_path / "heldout_aerohand"
    good_dir.mkdir()
    ho_dir.mkdir()
    good_fp = good_dir / "rec1.json"
    ho_fp = ho_dir / "secret_heldout.json"
    good_fp.write_text("{}", encoding="utf-8")
    ho_fp.write_text("{}", encoding="utf-8")

    with caplog.at_level("WARNING"):
        found = collect_json_paths([tmp_path])
    assert found == [good_fp]
    assert any("Skipping held-out path" in rec.message for rec in caplog.records)


def test_collect_json_paths_explicit_heldout_file_still_raises(tmp_path: Path):
    from aerohand.ml.recording_io import collect_json_paths

    ho_dir = tmp_path / "heldout_aerohand"
    ho_dir.mkdir()
    ho_fp = ho_dir / "secret.json"
    ho_fp.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Held-out path access is forbidden"):
        collect_json_paths([ho_fp])
    with pytest.raises(ValueError, match="Held-out path access is forbidden"):
        collect_json_paths([ho_dir])


def test_api_rejects_body_over_limit(tmp_path: Path):
    import http.client
    import json

    from aerohand.platform.control_api import MAX_API_BODY_BYTES, LocalControlAPIServer

    cfg = load_config()
    cfg["api"]["port"] = 0
    srv = LocalControlAPIServer(
        host="127.0.0.1",
        port=0,
        get_status=lambda: {"tracking": True},
        get_config=lambda: cfg,
        apply_config=lambda _c: None,
        set_tracking_enabled=lambda v: v,
        trigger_recenter=lambda: None,
        get_vitals=lambda: {"ok": True},
        token_path=tmp_path / "api_token",
    )
    srv.start()
    try:
        port = srv.bound_port
        assert port > 0
        oversized = json.dumps({"key": "cursor.gain", "value": "x" * (MAX_API_BODY_BYTES + 2048)}).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            "PATCH",
            "/api/settings",
            body=oversized,
            headers={
                "X-AeroHand-Token": srv.token,
                "Authorization": f"Bearer {srv.token}",
                "Content-Type": "application/json",
                "Content-Length": str(len(oversized)),
            },
        )
        resp = conn.getresponse()
        body = resp.read()
        conn.close()
        assert resp.status == 413, f"Expected HTTP 413 Payload Too Large, got {resp.status}: {body}"
    finally:
        srv.stop()


# ==============================================================================
# V052-01: Multi-Frame Veto Confirmation & Aligned Shadow Event Counting
# ==============================================================================


class _ScriptedClf:
    """Deterministic stub classifier yielding scripted `(pose, p_pose, p_idle, ready)` tuples."""

    def __init__(self, script: List[tuple[PoseName, float, float, bool]]) -> None:
        self._script = list(script)
        self._idx = 0

    def predict(self, *args, **kwargs):
        from aerohand.ml.base import PinchPrediction

        pose, p_pose, p_idle, ready = self._script[min(self._idx, len(self._script) - 1)]
        self._idx += 1
        probs = idle_probabilities()
        probs[PoseName.IDLE] = p_idle
        if pose != PoseName.IDLE:
            probs[pose] = p_pose
        return PinchPrediction(pose, probs, "high" if ready else "low", ready=ready)

    def reset(self, hand=None):
        pass


def test_veto_requires_k_consecutive_approvals():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_MIDDLE_PINCH
    heur = _ScriptedClf([(p, 0.95, 0.05, True)])
    # Approve x2, reject x1 (resets streak!), then approve x3 -> only 6th frame accepts!
    onnx = _ScriptedClf(
        [
            (p, 0.85, 0.15, True),
            (p, 0.85, 0.15, True),
            (PoseName.IDLE, 0.0, 0.90, True),
            (p, 0.85, 0.15, True),
            (p, 0.85, 0.15, True),
            (p, 0.85, 0.15, True),
        ]
    )
    guarded = GuardedClassifier(onnx, heur, mode="veto", veto_confirm_frames=3, veto_entry_window_ms=300.0)
    h = _make_open_hand()
    outs = [guarded.predict(h, h, "Right", i * 33.0).pose for i in range(6)]
    assert outs == [PoseName.IDLE, PoseName.IDLE, PoseName.IDLE, PoseName.IDLE, PoseName.IDLE, p]
    tel = guarded.telemetry()
    assert tel["veto_confirm_frames"] == 3
    assert tel["veto_count"] == 1
    assert tel["veto_frames"] == 1


def test_veto_single_approval_frame_does_not_accept():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_INDEX_PINCH
    heur = _ScriptedClf([(p, 0.95, 0.05, True)])
    # 1 approval frame followed by rejections -> must NEVER accept
    onnx = _ScriptedClf(
        [
            (p, 0.90, 0.10, True),
            (PoseName.IDLE, 0.0, 0.95, True),
            (PoseName.IDLE, 0.0, 0.95, True),
            (PoseName.IDLE, 0.0, 0.95, True),
        ]
    )
    guarded = GuardedClassifier(onnx, heur, mode="veto", veto_confirm_frames=3)
    h = _make_open_hand()
    outs = [guarded.predict(h, h, "Right", i * 33.0).pose for i in range(4)]
    assert all(out == PoseName.IDLE for out in outs)
    assert guarded.telemetry()["veto_count"] == 1


def test_veto_confirm_frames_1_matches_legacy_first_approval():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_RING_PINCH
    heur = _ScriptedClf([(p, 0.95, 0.05, True)])
    onnx = _ScriptedClf(
        [
            (p, 0.85, 0.15, True),
            (PoseName.IDLE, 0.0, 0.95, True),
        ]
    )
    guarded = GuardedClassifier(onnx, heur, mode="veto", veto_confirm_frames=1, veto_scope="entry")
    h = _make_open_hand()
    assert guarded.predict(h, h, "Right", 0.0).pose == p
    assert guarded.predict(h, h, "Right", 33.0).pose == p
    assert guarded.telemetry()["veto_count"] == 0


def test_veto_confirm_adds_exactly_k_minus_1_frames_latency():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_MIDDLE_PINCH
    h = _make_open_hand()
    for k in (1, 2, 3, 4):
        heur = _ScriptedClf([(p, 0.95, 0.05, True)])
        onnx = _ScriptedClf([(p, 0.90, 0.10, True)])
        guarded = GuardedClassifier(onnx, heur, mode="veto", veto_confirm_frames=k)
        first_accept_idx = None
        for idx in range(k + 2):
            out = guarded.predict(h, h, "Right", idx * 33.0).pose
            if out == p and first_accept_idx is None:
                first_accept_idx = idx
        assert first_accept_idx == k - 1, f"K={k}: expected first accept at frame {k - 1}, got {first_accept_idx}"


def test_veto_unready_frames_do_not_reset_streak():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_MIDDLE_PINCH
    heur = _ScriptedClf([(p, 0.95, 0.05, True)])
    # Frame 0: ready approve (streak=1 -> IDLE)
    # Frame 1: unready mid-pending -> V053-03 returns IDLE (no spurious pose flash), streak stays 1
    # Frame 2: ready approve (streak=2 -> IDLE, proves unready did not increment streak)
    # Frame 3: ready approve (streak=3 -> accepted -> p, proves unready did not reset streak)
    onnx = _ScriptedClf(
        [
            (p, 0.90, 0.10, True),
            (PoseName.IDLE, 0.0, 1.0, False),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
        ]
    )
    guarded = GuardedClassifier(onnx, heur, mode="veto", veto_confirm_frames=3)
    h = _make_open_hand()
    outs = [guarded.predict(h, h, "Right", i * 33.0).pose for i in range(4)]
    assert outs == [PoseName.IDLE, PoseName.IDLE, PoseName.IDLE, p]
    tel = guarded.telemetry()
    assert tel["veto_abstain_not_ready"] == 1
    assert tel["veto_count"] == 0


def test_veto_late_accept_after_window_when_streak_reached():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_INDEX_MIDDLE_PINCH
    heur = _ScriptedClf([(p, 0.95, 0.05, True)])
    # Reject during window, then after 300 ms approve 3 consecutive frames -> late accept & 1 veto_count
    onnx = _ScriptedClf(
        [
            (PoseName.IDLE, 0.0, 0.95, True),  # t=0 (in window, vetoed)
            (PoseName.IDLE, 0.0, 0.95, True),  # t=150 (in window)
            (p, 0.90, 0.10, True),             # t=350 (past window, streak=1 -> still IDLE)
            (p, 0.90, 0.10, True),             # t=383 (past window, streak=2 -> still IDLE)
            (p, 0.90, 0.10, True),             # t=416 (past window, streak=3 -> LATE ACCEPT!)
            (PoseName.IDLE, 0.0, 0.95, True),  # t=450 (accepted hold -> never interrupted!)
        ]
    )
    guarded = GuardedClassifier(
        onnx, heur, mode="veto", veto_scope="entry", veto_entry_window_ms=300.0, veto_confirm_frames=3
    )
    h = _make_open_hand()
    timestamps = [0.0, 150.0, 350.0, 383.0, 416.0, 450.0]
    outs = [guarded.predict(h, h, "Right", ts).pose for ts in timestamps]
    assert outs == [PoseName.IDLE, PoseName.IDLE, PoseName.IDLE, PoseName.IDLE, p, p]
    assert guarded.telemetry()["veto_count"] == 1


def test_veto_never_interrupts_accepted_hold():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_RING_PINCH
    heur = _ScriptedClf([(p, 0.95, 0.05, True)])
    onnx = _ScriptedClf(
        [
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (PoseName.IDLE, 0.0, 0.99, True),
            (PoseName.IDLE, 0.0, 0.99, True),
            (PoseName.IDLE, 0.0, 0.99, True),
        ]
    )
    guarded = GuardedClassifier(
        onnx, heur, mode="veto", veto_scope="entry", veto_entry_window_ms=300.0, veto_confirm_frames=3
    )
    h = _make_open_hand()
    outs = [guarded.predict(h, h, "Right", i * 33.0).pose for i in range(6)]
    assert outs == [PoseName.IDLE, PoseName.IDLE, p, p, p, p]
    assert guarded.telemetry()["veto_count"] == 0


def test_shadow_mode_counts_would_veto_with_k_frame_confirm_rule():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_MIDDLE_PINCH
    # Event 1: 1 approval frame then 2 rejections (would be vetoed!) -> release to IDLE
    # Event 2: 3 consecutive approvals -> would be accepted (NOT vetoed) -> release to IDLE
    heur = _ScriptedClf(
        [
            (p, 0.95, 0.05, True),
            (p, 0.95, 0.05, True),
            (p, 0.95, 0.05, True),
            (PoseName.IDLE, 0.0, 1.0, True),
            (p, 0.95, 0.05, True),
            (p, 0.95, 0.05, True),
            (p, 0.95, 0.05, True),
            (PoseName.IDLE, 0.0, 1.0, True),
        ]
    )
    onnx = _ScriptedClf(
        [
            (p, 0.90, 0.10, True),
            (PoseName.IDLE, 0.0, 0.95, True),
            (PoseName.IDLE, 0.0, 0.95, True),
            (PoseName.IDLE, 0.0, 1.0, True),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (PoseName.IDLE, 0.0, 1.0, True),
        ]
    )
    guarded = GuardedClassifier(onnx, heur, mode="shadow", veto_confirm_frames=3)
    h = _make_open_hand()
    for i in range(8):
        out = guarded.predict(h, h, "Right", i * 33.0)
        # Shadow mode NEVER alters heuristic output
        assert out.pose == (PoseName.IDLE if i in (3, 7) else p)
    tel = guarded.telemetry()
    assert tel["shadow_events"] == 2
    assert tel["shadow_would_veto_events"] == 1
    assert tel["shadow_would_veto_rate"] == pytest.approx(0.5)
    assert load_config()["classifier"]["veto_confirm_frames"] == 3


# ==============================================================================
# V052-02: Provenance Binding in Runtime Telemetry & Strict Vitals Promotion Gate
# ==============================================================================


def test_onnx_model_hash_matches_train_formula(tmp_path: Path, monkeypatch):
    import json

    import onnxruntime as ort

    from aerohand.ml.onnx_classifier import ONNXPinchClassifier
    from aerohand.ml.recording_io import sha256_file

    train_mod = _load_train_tcn_module("train_v052_02_hash")
    replay_mod = _load_replay_eval_module("replay_v052_02_hash")

    model_fp = tmp_path / "m_prov.onnx"
    model_fp.write_bytes(b"fake-onnx-model-payload-v052")
    norm_fp = tmp_path / "m_prov_normalizer.json"
    norm_fp.write_text(
        json.dumps(
            {
                "feature_set": "invariant348",
                "sequence_length": 5,
                "mean": [0.0] * FEATURE_DIM_3D,
                "scale": [1.0] * FEATURE_DIM_3D,
            }
        ),
        encoding="utf-8",
    )

    class _DummyOrt:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def get_inputs(self):
            class _In:
                name = "input"

            return [_In()]

        def run(self, _out, _feed):
            return [np.zeros((1, 7), dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _DummyOrt)
    clf = ONNXPinchClassifier(model_fp, norm_fp, sequence_length=5, warmup_runs=1)
    expected_train_hash = train_mod.compute_onnx_model_hash(model_fp, norm_fp)
    expected_replay_hash = replay_mod.compute_eval_model_hash(
        "onnx_tcn", load_config(), model_path=model_fp, normalizer_path=norm_fp
    )
    assert clf.model_hash == expected_train_hash == expected_replay_hash
    assert len(clf.model_hash) == 24
    assert clf.model_sha256 == sha256_file(model_fp)


def test_guarded_telemetry_includes_model_hash(tmp_path: Path, monkeypatch):
    import json

    import onnxruntime as ort

    from aerohand.ml.onnx_classifier import GuardedClassifier, ONNXPinchClassifier
    from aerohand.ml.provenance import compute_heuristic_source_sha256

    model_fp = tmp_path / "m_tel.onnx"
    model_fp.write_bytes(b"guarded-telemetry-onnx-bytes")
    norm_fp = tmp_path / "m_tel_norm.json"
    norm_fp.write_text(
        json.dumps(
            {
                "feature_set": "invariant348",
                "sequence_length": 5,
                "mean": [0.0] * FEATURE_DIM_3D,
                "scale": [1.0] * FEATURE_DIM_3D,
            }
        ),
        encoding="utf-8",
    )

    class _DummyOrt:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def get_inputs(self):
            class _In:
                name = "input"

            return [_In()]

        def run(self, _out, _feed):
            return [np.zeros((1, 7), dtype=np.float32)]

    monkeypatch.setattr(ort, "InferenceSession", _DummyOrt)
    onnx = ONNXPinchClassifier(model_fp, norm_fp, sequence_length=5, warmup_runs=1)
    guarded = GuardedClassifier(onnx, Heuristic3DPinchClassifier(), mode="veto")
    tel = guarded.telemetry()
    assert tel["model_hash"] == onnx.model_hash
    assert tel["model_sha256"] == onnx.model_sha256
    assert tel["heuristic_source_sha256"] == compute_heuristic_source_sha256()
    assert len(tel["heuristic_source_sha256"]) == 64


def test_promotion_fails_when_vitals_lacks_hash():
    promo_mod = _load_script_module("scripts/check_onnx_promotion.py", "check_onnx_promo_v052_vh_missing")
    cfg = load_config()
    metrics_ok = {"model_hash": "e" * 24, "model_sha256": "f" * 64, "subject_heldout_metrics": None}
    sealed_ok = {
        "model_hash": "e" * 24,
        "model_sha256": "f" * 64,
        "heldout_audit": {"touched": 1, "model_hash": "e" * 24, "model_sha256": "f" * 64},
        "false_triggers": {
            "hand_present_idle_hours": 1.60,
            "total_upper_bound_per_hour": 1.87,
            "by_type": {
                "left_click": 0,
                "right_click": 0,
                "drag_start": 0,
                "scroll_stroke": 0,
                "palm_wheel": 0,
                "window_tile": 0,
                "spaces_nav": 0,
                "dictation_start": 0,
            },
        },
    }
    vitals_no_hash = {"classifier": {"onnx_p95_ms": 1.2, "guard_active": True, "fell_back": False}}

    res = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok,
        sealed_eval_report=sealed_ok,
        vitals_report=vitals_no_hash,
        config=cfg,
        stage="veto",
    )
    assert res["passed"] is False
    assert res["criteria"]["provenance_hash_match"]["passed"] is False
    assert "vitals_report missing" in res["criteria"]["provenance_hash_match"]["detail"]


def test_promotion_fails_when_vitals_hash_differs():
    promo_mod = _load_script_module("scripts/check_onnx_promotion.py", "check_onnx_promo_v052_vh_diff")
    cfg = load_config()
    metrics_ok = {"model_hash": "e" * 24, "model_sha256": "f" * 64, "subject_heldout_metrics": None}
    sealed_ok = {
        "model_hash": "e" * 24,
        "model_sha256": "f" * 64,
        "heldout_audit": {"touched": 1, "model_hash": "e" * 24, "model_sha256": "f" * 64},
        "false_triggers": {
            "hand_present_idle_hours": 1.60,
            "total_upper_bound_per_hour": 1.87,
            "by_type": {
                "left_click": 0,
                "right_click": 0,
                "drag_start": 0,
                "scroll_stroke": 0,
                "palm_wheel": 0,
                "window_tile": 0,
                "spaces_nav": 0,
                "dictation_start": 0,
            },
        },
    }
    vitals_wrong = {
        "classifier": {
            "onnx_p95_ms": 1.2,
            "guard_active": True,
            "fell_back": False,
            "model_hash": "9" * 24,
            "model_sha256": "8" * 64,
        }
    }
    res_bad = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok,
        sealed_eval_report=sealed_ok,
        vitals_report=vitals_wrong,
        config=cfg,
        stage="veto",
    )
    assert res_bad["passed"] is False
    assert res_bad["criteria"]["provenance_hash_match"]["passed"] is False

    # Matching vitals hash MUST pass cleanly
    vitals_good = {
        "classifier": {
            "onnx_p95_ms": 1.2,
            "guard_active": True,
            "fell_back": False,
            "model_hash": "e" * 24,
            "model_sha256": "f" * 64,
        }
    }
    res_good = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok,
        sealed_eval_report=sealed_ok,
        vitals_report=vitals_good,
        config=cfg,
        stage="veto",
    )
    assert res_good["passed"] is True
    assert res_good["criteria"]["provenance_hash_match"]["passed"] is True


# ==============================================================================
# V052-03: Latch-Entry Debug Diagnostics (`latch_entry`, `latch_age_ms`)
# ==============================================================================


def test_latch_entry_debug_recorded_at_latch_time():
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    h_open = _make_open_hand()
    h_pinch = _pinch_middle_copy(h_open)

    # Open frames to prime approach gate
    for i in range(4):
        clf.predict(h_open, h_open, "Right", i * 33.0)
    assert clf.last_debug["Right"]["latch_entry"] == {}
    assert clf.last_debug["Right"]["latch_age_ms"] == {}

    # Enter pinch slowly at t=132 -> latches and snapshots entry metrics
    pred_enter = clf.predict(h_pinch, h_pinch, "Right", 132.0)
    assert pred_enter.pose == PoseName.THUMB_MIDDLE_PINCH
    dbg_enter = clf.last_debug["Right"]
    assert "thumb_middle_pinch" in dbg_enter["latched"]
    entry = dbg_enter["latch_entry"]["thumb_middle_pinch"]
    for key in (
        "since_ms",
        "dist",
        "raw_sc",
        "eff_speed_pw",
        "eff_ang_deg_s",
        "open_dist",
        "idx_ext",
        "half_curled_count",
    ):
        assert key in entry
    assert entry["since_ms"] == pytest.approx(132.0)
    assert entry["raw_sc"] >= 0.85
    assert entry["open_dist"] > entry["dist"]
    assert dbg_enter["latch_age_ms"]["thumb_middle_pinch"] == pytest.approx(0.0)

    # Move wrist fast on the next frame while staying pinched -> latch_entry stays frozen at entry values!
    h_fast_pinch = _pinch_middle_copy(_make_scaled_shifted_hand(1.0, dx=0.06))
    clf.predict(h_fast_pinch, h_fast_pinch, "Right", 165.0)
    dbg_moved = clf.last_debug["Right"]
    assert dbg_moved["eff_speed_pw"] > clf.max_wrist_speed_pw
    assert dbg_moved["latch_entry"]["thumb_middle_pinch"]["since_ms"] == pytest.approx(132.0)
    assert dbg_moved["latch_entry"]["thumb_middle_pinch"]["eff_speed_pw"] == entry["eff_speed_pw"]


def test_gate_debug_latch_age_increases_while_held():
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    h_open = _make_open_hand()
    h_pinch = _pinch_middle_copy(h_open)

    for i in range(4):
        clf.predict(h_open, h_open, "Right", i * 33.0)
    clf.predict(h_pinch, h_pinch, "Right", 200.0)
    assert clf.last_debug["Right"]["latch_age_ms"]["thumb_middle_pinch"] == pytest.approx(0.0)
    clf.predict(h_pinch, h_pinch, "Right", 266.0)
    assert clf.last_debug["Right"]["latch_age_ms"]["thumb_middle_pinch"] == pytest.approx(66.0)
    clf.predict(h_pinch, h_pinch, "Right", 399.0)
    assert clf.last_debug["Right"]["latch_age_ms"]["thumb_middle_pinch"] == pytest.approx(199.0)


def test_latch_info_cleared_when_latch_released():
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    h_open = _make_open_hand()
    h_pinch = _pinch_middle_copy(h_open)

    for i in range(4):
        clf.predict(h_open, h_open, "Right", i * 33.0)
    clf.predict(h_pinch, h_pinch, "Right", 200.0)
    assert PoseName.THUMB_MIDDLE_PINCH in clf._hands["Right"].latch_info
    assert "thumb_middle_pinch" in clf.last_debug["Right"]["latch_entry"]

    # Release back to open hand (dist >= off_dist) -> latch and latch_info MUST both clear!
    clf.predict(h_open, h_open, "Right", 233.0)
    assert PoseName.THUMB_MIDDLE_PINCH not in clf._hands["Right"].latched
    assert PoseName.THUMB_MIDDLE_PINCH not in clf._hands["Right"].latch_info
    assert clf.last_debug["Right"]["latch_entry"] == {}
    assert clf.last_debug["Right"]["latch_age_ms"] == {}


def test_ft_jsonl_contains_latch_entry(tmp_path: Path):
    import json

    mod = _load_replay_eval_module("replay_eval_v052_latch_entry")
    col = PinchDataCollector()
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    h_open = _make_open_hand(forward_z=-0.25)
    # Build a held thumb_index_pinch (fires dictation_start while actively latched!)
    h_idx_pinch = list(h_open)
    h_idx_pinch[4] = Landmark(h_open[8].x, h_open[8].y, h_open[8].z)
    for i in range(6):
        col.add_frame(h_open, h_open, "Right", 0.95, 1000 + i * 33)
    for i in range(20):  # ~660 ms hold > 350 ms dictation_arm_ms
        col.add_frame(h_idx_pinch, h_idx_pinch, "Right", 0.95, 1000 + (6 + i) * 33)
    neg_fp = col.save(tmp_path)

    jsonl_fp = tmp_path / "ft_latch_entry.jsonl"
    cfg = load_config()
    cfg["palm_wheel"]["enabled"] = False
    rep = mod.evaluate_recordings(
        negative_paths=[neg_fp],
        config=cfg,
        ft_jsonl_path=jsonl_fp,
        min_exposure_hours_for_claim=0.0,
        max_ft_per_hour=1e9,
    )
    assert rep["false_triggers"]["by_type"]["dictation_start"] >= 1
    lines = [json.loads(ln) for ln in jsonl_fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dict_events = [ev for ev in lines if ev["trigger_type"] == "dictation_start"]
    assert len(dict_events) >= 1
    gd = dict_events[0]["gate_debug"]
    assert "thumb_index_pinch" in gd["latched"]
    assert "thumb_index_pinch" in gd["latch_entry"]
    assert "thumb_index_pinch" in gd["latch_age_ms"]
    assert gd["latch_age_ms"]["thumb_index_pinch"] > 0.0
    le = gd["latch_entry"]["thumb_index_pinch"]
    for k in ("since_ms", "dist", "raw_sc", "eff_speed_pw", "eff_ang_deg_s", "open_dist", "idx_ext", "half_curled_count"):
        assert k in le


def test_gate_debug_logging_does_not_change_predictions():
    from aerohand.core.types import ALL_POSE_NAMES

    cfg = load_config()["classifier"]["heuristic_3d"]
    clf_a = Heuristic3DPinchClassifier(cfg)
    clf_b = Heuristic3DPinchClassifier(cfg)
    h_open = _make_open_hand()
    h_pinch = _pinch_middle_copy(h_open)
    h_pinch_moved = _pinch_middle_copy(_make_scaled_shifted_hand(1.0, dx=0.05))
    seq = (
        [(i * 33.0, h_open) for i in range(5)]
        + [(165.0 + i * 33.0, h_pinch) for i in range(3)]
        + [(264.0 + i * 33.0, h_pinch_moved) for i in range(3)]
        + [(363.0 + i * 33.0, h_open) for i in range(3)]
    )
    for ts, lms in seq:
        pa = clf_a.predict(lms, lms, "Right", ts)
        pb = clf_b.predict(lms, lms, "Right", ts)
        assert pa.pose == pb.pose
        assert pa.confidence == pb.confidence
        for k in ALL_POSE_NAMES:
            assert pa.probabilities[k] == pytest.approx(pb.probabilities[k])
    # Verify latch_entry stayed purely observational and matches expected deterministic behavior
    assert clf_a.predict(h_open, h_open, "Right", 500.0).pose == PoseName.IDLE


# ==============================================================================
# V052-05: Hygiene (Sealed-Hash Discovery, Tighter Heuristic Hash, Atomic Async Swap)
# ==============================================================================


def test_exclude_sealed_warns_when_no_manifest_found(tmp_path: Path, caplog):
    from aerohand.ml.recording_io import exclude_sealed_content

    sess_dir = tmp_path / "isolated_exp" / "sessions"
    sess_dir.mkdir(parents=True)
    rec_fp = sess_dir / "rec1.json"
    rec_fp.write_text('{"samples": []}', encoding="utf-8")

    with caplog.at_level("WARNING", logger="aerohand.ml.recording_io"):
        kept, excluded = exclude_sealed_content([rec_fp], data_root=tmp_path / "nonexistent_data")
    assert kept == [rec_fp]
    assert excluded == []
    assert any("Found 0 sealed_manifest.json" in rec.message for rec in caplog.records)


def test_exclude_sealed_finds_manifest_from_input_parent(tmp_path: Path, monkeypatch):
    import json

    from aerohand.ml.recording_io import exclude_sealed_content, sha256_file

    # Place sessions and heldout_aerohand under an external parent dir and run from an unrelated CWD
    exp_root = tmp_path / "external_corpus"
    sess_dir = exp_root / "sessions" / "day1"
    ho_dir = exp_root / "heldout_aerohand"
    other_cwd = tmp_path / "somewhere_else"
    sess_dir.mkdir(parents=True)
    ho_dir.mkdir(parents=True)
    other_cwd.mkdir(parents=True)

    leaked = sess_dir / "leaked_copy.json"
    clean = sess_dir / "clean_train.json"
    leaked.write_text('{"samples": [{"id": "sealed_secret"}]}', encoding="utf-8")
    clean.write_text('{"samples": [{"id": "train_only"}]}', encoding="utf-8")

    (ho_dir / "sealed_manifest.json").write_text(
        json.dumps({"files": [{"filename": "orig.json", "sha256": sha256_file(leaked)}]}),
        encoding="utf-8",
    )

    monkeypatch.chdir(other_cwd)
    kept, excluded = exclude_sealed_content([leaked, clean], data_root="data")
    assert kept == [clean]
    assert excluded == [leaked]


def test_exclude_sealed_cli_sealed_root_override(tmp_path: Path):
    import json

    from aerohand.ml.recording_io import sha256_file

    train_mod = _load_train_tcn_module("train_v052_05_sroot")
    sess_dir = tmp_path / "train_pool"
    custom_sealed = tmp_path / "unrelated_vault" / "heldout_aerohand"
    sess_dir.mkdir(parents=True)
    custom_sealed.mkdir(parents=True)

    leaked = sess_dir / "dup.json"
    keep = sess_dir / "keep.json"
    leaked.write_text('{"samples": [{"secret": 1}]}', encoding="utf-8")
    keep.write_text('{"samples": [{"secret": 2}]}', encoding="utf-8")

    (custom_sealed / "sealed_manifest.json").write_text(
        json.dumps({"files": [{"filename": "sealed.json", "sha256": sha256_file(leaked)}]}),
        encoding="utf-8",
    )

    kept, excluded = train_mod.collect_training_files(
        [sess_dir],
        data_root=tmp_path / "empty_data",
        sealed_root=custom_sealed,
    )
    assert kept == [keep]
    assert excluded == [leaked]


def test_heuristic_hash_ignores_veto_config():
    replay_mod = _load_replay_eval_module("replay_v052_05_hhash_veto")
    cfg1 = load_config()
    cfg2 = deep_merge(
        cfg1,
        {
            "classifier": {
                "mode": "veto",
                "veto_min_prob": 0.48,
                "veto_idle_prob": 0.52,
                "veto_confirm_frames": 2,
                "veto_entry_window_ms": 450.0,
            }
        },
    )
    h1 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg1)
    h2 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg2)
    assert len(h1) == 24
    assert h1 == h2


def test_heuristic_hash_changes_with_palm_frame_source():
    replay_mod = _load_replay_eval_module("replay_v052_05_hhash_pf")
    cfg = load_config()
    h1 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg, palm_frame_source_bytes=b"palm_v1")
    h2 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg, palm_frame_source_bytes=b"palm_v2")
    assert len(h1) == 24 and len(h2) == 24
    assert h1 != h2


def test_heuristic_hash_changes_with_heuristic_threshold():
    replay_mod = _load_replay_eval_module("replay_v052_05_hhash_thr")
    cfg1 = load_config()
    cfg2 = deep_merge(cfg1, {"classifier": {"heuristic_3d": {"pinch_on_dist": 0.36}}})
    cfg3 = deep_merge(cfg1, {"smoothing": {"landmark_beta": 60.0}})
    h1 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg1)
    h2 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg2)
    h3 = replay_mod.compute_eval_model_hash("heuristic_3d", cfg3)
    assert h1 != h2
    assert h1 != h3


def test_swap_classifier_async_merges_counters_at_swap_time(monkeypatch):
    import threading

    import aerohand.runtime as rt_mod
    from aerohand.ml.onnx_classifier import GuardedClassifier
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["classifier"]["mode"] = "shadow"
    rt = AeroHandRuntime(cfg, dry_run=True)

    old_onnx = _ScriptedClf([(PoseName.THUMB_MIDDLE_PINCH, 0.9, 0.1, True)])
    old_guarded = GuardedClassifier(old_onnx, Heuristic3DPinchClassifier(), mode="shadow")
    rt.classifier = old_guarded
    rt.engine.set_classifier(old_guarded)

    h = _make_open_hand()
    # 2 shadow frames BEFORE async swap begins
    old_guarded.predict(h, h, "Right", 0.0)
    old_guarded.predict(h, h, "Right", 33.0)
    assert old_guarded.telemetry()["shadow_frames"] == 2

    started_ev = threading.Event()
    release_ev = threading.Event()
    new_guarded = GuardedClassifier(
        _ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]),
        Heuristic3DPinchClassifier(),
        mode="shadow",
    )

    def _slow_build(_cfg, *, on_error=None, preserved_counters=None):
        started_ev.set()
        assert release_ev.wait(timeout=5.0)
        return new_guarded

    monkeypatch.setattr(rt_mod, "build_classifier", _slow_build)

    t = rt._swap_classifier_async(cfg)
    assert started_ev.wait(timeout=5.0)

    # Accumulate 5 MORE shadow frames on the old classifier WHILE background build is running!
    for i in range(5):
        old_guarded.predict(h, h, "Right", 100.0 + i * 33.0)
    assert old_guarded.telemetry()["shadow_frames"] == 7

    release_ev.set()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert rt.classifier is new_guarded
    # All 7 shadow frames (2 before + 5 during load) MUST be preserved with zero loss!
    assert rt.classifier.telemetry()["shadow_frames"] == 7


def test_swap_classifier_async_stale_generation_discarded(monkeypatch):
    import threading

    import aerohand.runtime as rt_mod
    from aerohand.ml.onnx_classifier import GuardedClassifier
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    rt = AeroHandRuntime(cfg, dry_run=True)

    ev1_started = threading.Event()
    ev1_release = threading.Event()
    clf1 = GuardedClassifier(_ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]), Heuristic3DPinchClassifier(), mode="veto")
    clf2 = GuardedClassifier(_ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]), Heuristic3DPinchClassifier(), mode="veto")
    call_n = {"n": 0}

    def _build(_cfg, *, on_error=None, preserved_counters=None):
        call_n["n"] += 1
        idx = call_n["n"]
        if idx == 1:
            ev1_started.set()
            assert ev1_release.wait(timeout=5.0)
            return clf1
        return clf2

    monkeypatch.setattr(rt_mod, "build_classifier", _build)

    t1 = rt._swap_classifier_async(cfg)
    assert ev1_started.wait(timeout=5.0)
    # Trigger a newer swap #2 while swap #1 is still blocked -> swap #2 completes first
    t2 = rt._swap_classifier_async(cfg)
    t2.join(timeout=5.0)
    assert rt.classifier is clf2

    # Now let stale swap #1 finish -> MUST be discarded, leaving clf2 installed!
    ev1_release.set()
    t1.join(timeout=5.0)
    assert rt.classifier is clf2
    assert rt.engine.classifier is clf2


def test_version_string_consistent_across_pyproject_and_package():
    import tomllib

    from aerohand import __version__
    from aerohand.config import default_config_path, repo_root

    pyproj = tomllib.loads((repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    proj_version = str(pyproj["project"]["version"]).strip()
    assert __version__ == proj_version
    first_cfg_line = default_config_path().read_text(encoding="utf-8").splitlines()[0]
    assert f"v{proj_version}" in first_cfg_line


# ==============================================================================
# V053-02 (B2): Swallowed Events & Pending Frames Parity Across Veto & Shadow
# ==============================================================================


def test_veto_and_shadow_record_swallowed_event_when_released_before_k():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_MIDDLE_PINCH
    h = _make_open_hand()
    # 2 positive frames (ONNX approves both), release to IDLE, then retry within 1s (3 approvals -> accepted)
    heur_seq = [
        (p, 0.95, 0.05, True),
        (p, 0.95, 0.05, True),
        (PoseName.IDLE, 0.0, 1.0, True),
        (p, 0.95, 0.05, True),
        (p, 0.95, 0.05, True),
        (p, 0.95, 0.05, True),
        (PoseName.IDLE, 0.0, 1.0, True),
    ]
    onnx_seq = [
        (p, 0.90, 0.10, True),
        (p, 0.90, 0.10, True),
        (PoseName.IDLE, 0.0, 1.0, True),
        (p, 0.90, 0.10, True),
        (p, 0.90, 0.10, True),
        (p, 0.90, 0.10, True),
        (PoseName.IDLE, 0.0, 1.0, True),
    ]

    gv = GuardedClassifier(_ScriptedClf(onnx_seq), _ScriptedClf(heur_seq), mode="veto", veto_confirm_frames=3)
    gs = GuardedClassifier(_ScriptedClf(onnx_seq), _ScriptedClf(heur_seq), mode="shadow", veto_confirm_frames=3)
    for i in range(len(heur_seq)):
        gv.predict(h, h, "Right", i * 33.0)
        gs.predict(h, h, "Right", i * 33.0)

    tv = gv.telemetry()
    ts = gs.telemetry()
    # Veto: 1st tap swallowed (not explicit veto), 2+2 pending frames across both attempts, 1 retry within 1s
    assert tv["veto_count"] == 0
    assert tv["veto_frames"] == 0
    assert tv["veto_swallowed_events"] == 1
    assert tv["veto_pending_frames"] == 4
    assert tv["veto_retried_within_1s"] == 1
    # Shadow mirrors the EXACT same event & frame definitions
    assert ts["shadow_events"] == 2
    assert ts["shadow_would_veto_events"] == 0
    assert ts["shadow_would_veto_frames"] == 0
    assert ts["shadow_would_swallow_events"] == 1
    assert ts["shadow_would_pending_frames"] == 4


def test_swallowed_event_finalized_on_pose_change_and_hand_reset_not_global_reset():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p1 = PoseName.THUMB_MIDDLE_PINCH
    p2 = PoseName.THUMB_RING_PINCH
    h = _make_open_hand()

    for mode, sw_key in (("veto", "veto_swallowed_events"), ("shadow", "shadow_would_swallow_events")):
        # 1. Pose switch after 2 approved frames -> finalizes p1 as 1 swallowed event
        heur = _ScriptedClf(
            [
                (p1, 0.95, 0.05, True),
                (p1, 0.95, 0.05, True),
                (p2, 0.95, 0.05, True),
                (p2, 0.95, 0.05, True),
                (p2, 0.95, 0.05, True),
            ]
        )
        onnx = _ScriptedClf(
            [
                (p1, 0.90, 0.10, True),
                (p1, 0.90, 0.10, True),
                (p2, 0.90, 0.10, True),
                (p2, 0.90, 0.10, True),
                (p2, 0.90, 0.10, True),
            ]
        )
        g = GuardedClassifier(onnx, heur, mode=mode, veto_confirm_frames=3)
        for i in range(5):
            g.predict(h, h, "Right", i * 33.0)
        assert g.telemetry()[sw_key] == 1

        # 2. Per-hand reset("Right") (hand lost mid-pending) -> finalizes as swallowed
        g2 = GuardedClassifier(
            _ScriptedClf([(p1, 0.90, 0.10, True)]),
            _ScriptedClf([(p1, 0.95, 0.05, True)]),
            mode=mode,
            veto_confirm_frames=3,
        )
        g2.predict(h, h, "Right", 0.0)
        g2.predict(h, h, "Right", 33.0)
        g2.reset("Right")
        assert g2.telemetry()[sw_key] == 1

        # 3. Global reset(None) (mode switch / rearm / fallback) -> MUST NOT count as swallowed
        g3 = GuardedClassifier(
            _ScriptedClf([(p1, 0.90, 0.10, True)]),
            _ScriptedClf([(p1, 0.95, 0.05, True)]),
            mode=mode,
            veto_confirm_frames=3,
        )
        g3.predict(h, h, "Right", 0.0)
        g3.predict(h, h, "Right", 33.0)
        g3.reset(None)
        assert g3.telemetry()[sw_key] == 0


def test_swallowed_and_pending_counters_exported_and_in_heuristic_telemetry():
    from aerohand.ml.onnx_classifier import GuardedClassifier
    from aerohand.runtime import AeroHandRuntime

    p = PoseName.THUMB_MIDDLE_PINCH
    h = _make_open_hand()
    gv = GuardedClassifier(
        _ScriptedClf([(p, 0.90, 0.10, True), (p, 0.90, 0.10, True), (PoseName.IDLE, 0.0, 1.0, True)]),
        _ScriptedClf([(p, 0.95, 0.05, True), (p, 0.95, 0.05, True), (PoseName.IDLE, 0.0, 1.0, True)]),
        mode="veto",
        veto_confirm_frames=3,
    )
    for i in range(3):
        gv.predict(h, h, "Right", i * 33.0)
    exp = gv.export_counters()
    assert exp["veto_swallowed_events"] == 1
    assert exp["veto_pending_frames"] == 2
    assert "shadow_would_swallow_events" in exp
    assert "shadow_would_pending_frames" in exp

    gv2 = GuardedClassifier(
        _ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]),
        _ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]),
        mode="veto",
    )
    gv2.import_counters(exp)
    assert gv2.telemetry()["veto_swallowed_events"] == 1
    assert gv2.telemetry()["veto_pending_frames"] == 2

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["classifier"]["mode"] = "heuristic"
    rt = AeroHandRuntime(cfg, dry_run=True)
    htel = rt.classifier_telemetry()
    for k in (
        "veto_swallowed_events",
        "veto_pending_frames",
        "shadow_would_swallow_events",
        "shadow_would_pending_frames",
    ):
        assert k in htel


# ==============================================================================
# V053-03 (B3): Consistent Warm-Up Abstain (No Mid-Event Drop to IDLE)
# ==============================================================================


def test_veto_abstain_on_warmup_does_not_split_event_into_idle():
    """V053-03 (B3): 4 unready warm-up frames followed by ready approvals must NOT dip to IDLE mid-event."""
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_MIDDLE_PINCH
    h = _make_open_hand()
    heur = _ScriptedClf([(p, 0.95, 0.05, True)])
    onnx = _ScriptedClf(
        [
            (PoseName.IDLE, 0.0, 1.0, False),
            (PoseName.IDLE, 0.0, 1.0, False),
            (PoseName.IDLE, 0.0, 1.0, False),
            (PoseName.IDLE, 0.0, 1.0, False),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
        ]
    )
    guarded = GuardedClassifier(onnx, heur, mode="veto", veto_scope="entry", veto_confirm_frames=3)
    outs = [guarded.predict(h, h, "Right", i * 33.0).pose for i in range(7)]
    # Once abstained on warm-up frames, event stays continuously `p` (never splits into p -> IDLE -> p)
    assert outs == [p] * 7, f"Event split into IDLE mid-warmup: {outs}"
    assert guarded.telemetry()["veto_abstain_not_ready"] == 4
    assert guarded.telemetry()["veto_count"] == 0

    # Also verify already-accepted event encountering an unready frame stays `p` (never drops to IDLE)
    onnx2 = _ScriptedClf(
        [
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (PoseName.IDLE, 0.0, 1.0, False),
        ]
    )
    g2 = GuardedClassifier(onnx2, _ScriptedClf([(p, 0.95, 0.05, True)]), mode="veto", veto_confirm_frames=3)
    outs2 = [g2.predict(h, h, "Right", i * 33.0).pose for i in range(4)]
    assert outs2 == [PoseName.IDLE, PoseName.IDLE, p, p]


# ==============================================================================
# V053-04 (B4): In-Flight Async Swap Must Not Overwrite Newer Classifier Config
# ==============================================================================


def test_async_swap_does_not_overwrite_mode_switch_to_heuristic(monkeypatch):
    """V053-04 (B4): Switching back to `heuristic` while an async `veto` reload is in flight must win."""
    import threading

    import aerohand.runtime as rt_mod
    from aerohand.ml.onnx_classifier import GuardedClassifier
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["classifier"]["mode"] = "shadow"
    rt = AeroHandRuntime(cfg, dry_run=True)
    # Start with an active GuardedClassifier so switching mode later would normally hit the in-place branch
    initial_guarded = GuardedClassifier(
        _ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]),
        Heuristic3DPinchClassifier(),
        mode="shadow",
    )
    rt.classifier = initial_guarded
    rt.engine.set_classifier(initial_guarded)
    rt._processing_thread_id = threading.get_ident()

    build_started = threading.Event()
    build_release = threading.Event()
    stale_veto_clf = GuardedClassifier(
        _ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]),
        Heuristic3DPinchClassifier(),
        mode="veto",
    )

    def _mock_build(c, *, on_error=None, preserved_counters=None):
        mode = (c.get("classifier") or {}).get("mode")
        if mode == "veto":
            build_started.set()
            assert build_release.wait(timeout=5.0)
            return stale_veto_clf
        return Heuristic3DPinchClassifier((c.get("classifier") or {}).get("heuristic_3d"))

    monkeypatch.setattr(rt_mod, "build_classifier", _mock_build)

    # 1. Trigger async reload for a new model path in mode="veto"
    cfg_veto = deep_merge(cfg, {"classifier": {"mode": "veto", "onnx_model_path": "models/new_v2.onnx"}})
    rt._apply_live_settings_now(cfg_veto)
    t_bg = rt._classifier_swap_thread
    assert t_bg is not None
    assert build_started.wait(timeout=5.0)

    # 2. While background reload is in flight, user switches back to mode="heuristic"
    cfg_heur = deep_merge(cfg_veto, {"classifier": {"mode": "heuristic"}})
    rt._apply_live_settings_now(cfg_heur)

    # 3. Release stale background loader and wait -> final classifier MUST NOT be stale_veto_clf!
    build_release.set()
    t_bg.join(timeout=5.0)
    assert rt.classifier is not stale_veto_clf
    assert rt.classifier_telemetry()["mode"] == "heuristic"


def test_async_swap_does_not_overwrite_veto_param_change_in_flight(monkeypatch):
    """V053-04 (B4): Changing `veto_confirm_frames` while async `veto` load is in flight must apply latest value."""
    import threading

    import aerohand.runtime as rt_mod
    from aerohand.ml.onnx_classifier import GuardedClassifier
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["classifier"]["mode"] = "heuristic"
    rt = AeroHandRuntime(cfg, dry_run=True)
    rt._processing_thread_id = threading.get_ident()

    first_started = threading.Event()
    first_release = threading.Event()
    built: List[GuardedClassifier] = []

    def _mock_build(c, *, on_error=None, preserved_counters=None):
        k = int((c.get("classifier") or {}).get("veto_confirm_frames", 3))
        g = GuardedClassifier(
            _ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]),
            Heuristic3DPinchClassifier(),
            mode="veto",
            veto_confirm_frames=k,
        )
        built.append(g)
        if k == 3:
            first_started.set()
            assert first_release.wait(timeout=5.0)
        return g

    monkeypatch.setattr(rt_mod, "build_classifier", _mock_build)

    # 1. Switch heuristic -> veto (k=3): starts async load #1
    cfg_k3 = deep_merge(cfg, {"classifier": {"mode": "veto", "veto_confirm_frames": 3}})
    rt._apply_live_settings_now(cfg_k3)
    t1 = rt._classifier_swap_thread
    assert t1 is not None
    assert first_started.wait(timeout=5.0)

    # 2. While load #1 is still in flight, change veto_confirm_frames -> 2 (same mode="veto")
    cfg_k2 = deep_merge(cfg_k3, {"classifier": {"veto_confirm_frames": 2}})
    rt._apply_live_settings_now(cfg_k2)
    t2 = rt._classifier_swap_thread
    assert t2 is not None
    if t2 is not t1:
        t2.join(timeout=5.0)

    # 3. Release load #1 -> must be discarded; final classifier MUST have veto_confirm_frames == 2!
    first_release.set()
    t1.join(timeout=5.0)
    assert isinstance(rt.classifier, GuardedClassifier)
    assert rt.classifier.veto_confirm_frames == 2


# ==============================================================================
# V053-05 (B5): Fail Loudly on Corrupt or Empty `sealed_manifest.json`
# ==============================================================================


def test_sealed_guard_raises_on_corrupt_or_empty_manifest(tmp_path: Path):
    """V053-05 (B5): Corrupt JSON or zero valid SHA-256 hashes in sealed_manifest.json MUST raise with path."""
    import json

    from aerohand.ml.recording_io import exclude_sealed_content, load_sealed_hashes

    # Case 1: Malformed JSON manifest
    bad_dir = tmp_path / "case_corrupt" / "heldout_aerohand"
    bad_dir.mkdir(parents=True)
    bad_mf = bad_dir / "sealed_manifest.json"
    bad_mf.write_text("{corrupt json", encoding="utf-8")
    rec_fp = tmp_path / "case_corrupt" / "rec.json"
    rec_fp.write_text('{"samples": []}', encoding="utf-8")

    with pytest.raises(ValueError, match=str(bad_mf.name)):
        load_sealed_hashes(data_root=tmp_path / "case_corrupt")
    with pytest.raises(ValueError, match=str(bad_mf.name)):
        exclude_sealed_content([rec_fp], data_root=tmp_path / "case_corrupt")

    # Case 2: Valid JSON but zero valid 64-hex SHA-256 digests
    empty_dir = tmp_path / "case_empty" / "heldout_aerohand"
    empty_dir.mkdir(parents=True)
    empty_mf = empty_dir / "sealed_manifest.json"
    empty_mf.write_text(json.dumps({"files": [{"filename": "a.json", "sha256": "invalid"}]}), encoding="utf-8")
    rec_fp2 = tmp_path / "case_empty" / "rec.json"
    rec_fp2.write_text('{"samples": []}', encoding="utf-8")

    with pytest.raises(ValueError, match=str(empty_mf.name)):
        load_sealed_hashes(data_root=tmp_path / "case_empty")
    with pytest.raises(ValueError, match=str(empty_mf.name)):
        exclude_sealed_content([rec_fp2], data_root=tmp_path / "case_empty")


# ==============================================================================
# V060-00: v0.5.3 Review Leftovers (always-scope unready, partial corrupt manifest, block_rate)
# ==============================================================================


def test_veto_scope_always_unready_after_k_approvals_passes_heuristic():
    """V060-00 (1): In veto_scope='always', ready=False after approve_streak >= K returns heuristic pose."""
    from aerohand.ml.onnx_classifier import GuardedClassifier

    p = PoseName.THUMB_MIDDLE_PINCH
    h = _make_open_hand()
    heur = _ScriptedClf([(p, 0.95, 0.05, True)])
    onnx = _ScriptedClf(
        [
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),            # streak == 3 >= K -> passes `p`
            (PoseName.IDLE, 0.0, 1.0, False),  # unready after K approvals -> MUST still pass `p`!
            (PoseName.IDLE, 0.0, 0.95, True),  # explicit rejection resets streak to 0 -> IDLE
            (PoseName.IDLE, 0.0, 1.0, False),  # unready with streak < K -> IDLE
        ]
    )
    g = GuardedClassifier(onnx, heur, mode="veto", veto_scope="always", veto_confirm_frames=3)
    outs = [g.predict(h, h, "Right", i * 33.0).pose for i in range(6)]
    assert outs == [PoseName.IDLE, PoseName.IDLE, p, p, PoseName.IDLE, PoseName.IDLE]


def test_sealed_manifest_raises_on_any_invalid_entry(tmp_path: Path):
    """V060-00 (2): Even if one entry has a valid SHA-256, ANY malformed/invalid entry must raise."""
    import json

    from aerohand.ml.recording_io import read_sealed_manifest_hashes

    sub = tmp_path / "case_invalid"
    sub.mkdir()
    mf = sub / "sealed_manifest.json"
    mf.write_text(
        json.dumps(
            {
                "files": [
                    {"filename": "ok.json", "sha256": "a" * 64},
                    {"filename": "bad.json", "sha256": "not-a-valid-sha"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed_manifest.json"):
        read_sealed_manifest_hashes(mf)

    mf2 = sub / "sealed_manifest_nondict.json"
    mf2.write_text(
        json.dumps({"files": [{"filename": "ok.json", "sha256": "b" * 64}, "corrupt_string_entry"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sealed_manifest_nondict.json"):
        read_sealed_manifest_hashes(mf2)


def test_shadow_would_block_rate_includes_veto_and_swallow():
    """V060-00 (3): shadow_would_block_rate = (would_veto + would_swallow) / shadow_events (incl. heuristic dict)."""
    from aerohand.ml.onnx_classifier import GuardedClassifier
    from aerohand.runtime import AeroHandRuntime

    p = PoseName.THUMB_MIDDLE_PINCH
    h = _make_open_hand()
    # Event 1: 2 approvals then IDLE -> swallowed (1)
    # Event 2: 1 rejection then IDLE -> vetoed (1)
    # Event 3: 3 approvals then IDLE -> accepted (0)
    # Total events=3 -> shadow_would_veto_rate=1/3, shadow_would_block_rate=2/3
    heur = _ScriptedClf(
        [
            (p, 0.95, 0.05, True),
            (p, 0.95, 0.05, True),
            (PoseName.IDLE, 0.0, 1.0, True),
            (p, 0.95, 0.05, True),
            (PoseName.IDLE, 0.0, 1.0, True),
            (p, 0.95, 0.05, True),
            (p, 0.95, 0.05, True),
            (p, 0.95, 0.05, True),
            (PoseName.IDLE, 0.0, 1.0, True),
        ]
    )
    onnx = _ScriptedClf(
        [
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (PoseName.IDLE, 0.0, 1.0, True),
            (PoseName.IDLE, 0.0, 0.95, True),
            (PoseName.IDLE, 0.0, 1.0, True),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (p, 0.90, 0.10, True),
            (PoseName.IDLE, 0.0, 1.0, True),
        ]
    )
    g = GuardedClassifier(onnx, heur, mode="shadow", veto_confirm_frames=3)
    for i in range(9):
        g.predict(h, h, "Right", i * 33.0)
    tel = g.telemetry()
    assert tel["shadow_events"] == 3
    assert tel["shadow_would_veto_events"] == 1
    assert tel["shadow_would_swallow_events"] == 1
    assert tel["shadow_would_veto_rate"] == pytest.approx(round(1 / 3, 4))
    assert tel["shadow_would_block_rate"] == pytest.approx(round(2 / 3, 4))

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["classifier"]["mode"] = "heuristic"
    rt = AeroHandRuntime(cfg, dry_run=True)
    rt._cumulative_classifier_counters = g.export_counters()
    htel = rt.classifier_telemetry()
    assert htel["shadow_would_block_rate"] == pytest.approx(round(2 / 3, 4))


# ==============================================================================
# V060-01a: Cued Event Recording (`recording_kind="cued_events"`, `data/recall/`)
# ==============================================================================


def test_cued_session_save_load_roundtrip_preserves_all_cue_fields_and_empty_frames(tmp_path: Path):
    import json

    from aerohand.ml.recording_io import load_recording_frames

    col = PinchDataCollector(default_subject="minh")
    st = col.start_cued_session(cues_per_trigger=1, hand="Right", window_ms=600, gap_ms=1000, seed=42, triggers=["left_click"])
    assert st["active"] is True
    assert col.is_cued_active is True

    h_r = _make_open_hand(forward_z=-0.25)
    h_l = _make_open_hand(forward_z=-0.25)
    # 1. Rest phase (t=0..1000ms) with both hands present + one hand-absent frame + one low-conf frame
    col.tick_cued(0)
    col.add_cued_frame([HandInput("Right", h_r, h_r, confidence=0.40, timestamp_ms=0), HandInput("Left", h_l, h_l, confidence=0.95, timestamp_ms=0)], 0)
    col.tick_cued(500)
    col.add_cued_frame([], 500)  # Hand-absent frame MUST be preserved for lost-hand reset on replay!
    # 2. At t=1000ms -> gap >= 1000ms -> issues Cue #1 with cue_ts_ms == 1000
    col.tick_cued(1000)
    col.add_cued_frame([HandInput("Right", h_r, h_r, confidence=0.95, timestamp_ms=1000)], 1000)
    # Autosave draft while active -> must keep recording_kind and cue_events
    draft_fp = col.autosave_cued_draft(tmp_path)
    assert draft_fp is not None and draft_fp.is_file()
    draft_raw = json.loads(draft_fp.read_text(encoding="utf-8"))
    assert draft_raw["recording_kind"] == "cued_events"
    assert len(draft_raw["cue_events"]) == 1

    # 3. End cue window at t=1600ms -> finishes session
    col.tick_cued(1600)
    assert col.is_cued_active is False

    out_fp = col.save_cued(tmp_path)
    raw = json.loads(out_fp.read_text(encoding="utf-8"))
    assert raw["schema"] == "aerohand_pinch_v2"
    assert raw["recording_kind"] == "cued_events"
    assert len(raw["cue_events"]) == 1
    ev = raw["cue_events"][0]
    for k in ("cue_id", "pose", "expected_trigger", "hand", "cue_ts_ms", "window_ms", "t_start_ms", "t_end_ms"):
        assert k in ev
    assert ev["pose"] == PoseName.THUMB_MIDDLE_PINCH.value
    assert ev["expected_trigger"] == "left_click"
    assert ev["cue_ts_ms"] == 1000
    assert ev["t_start_ms"] == 1000
    assert ev["t_end_ms"] == 1600
    assert ev["window_ms"] == 600
    # Low-conf frame (0.40), both hands at t=0, AND hand-absent frame at t=500 must ALL survive (no trim/clean!)
    assert len(raw["samples"]) == 4

    frames, meta = load_recording_frames(out_fp)
    assert meta["recording_kind"] == "cued_events"
    assert meta["cue_events"] == raw["cue_events"]
    assert any(fr["hand_present"] is False and fr["timestamp_ms"] == 500.0 for fr in frames)
    assert any(fr["handedness"] == "Left" for fr in frames)


def test_cued_events_excluded_from_train_matrix_domain_and_replay(tmp_path: Path, caplog):
    from aerohand.ml.collector import compute_corpus_condition_matrix

    train_mod = _load_train_tcn_module("train_v060_01a_excl")
    replay_mod = _load_replay_eval_module("replay_v060_01a_excl")
    dom_mod = _load_script_module("scripts/domain_check.py", "domain_v060_01a_excl")

    # Isolate tmp_path as its own project root so sibling pytest temp dirs are never scanned
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    rec_dir = tmp_path / "sessions"
    rec_dir.mkdir()

    # Save 1 normal session and 1 cued_events session in the same directory
    h = _make_open_hand(forward_z=-0.25)
    norm_col = PinchDataCollector(default_subject="minh")
    norm_col.set_recording(True)
    norm_col.set_pose(PoseName.IDLE)
    for i in range(6):
        norm_col.add_frame(h, h, "Right", 0.95, i * 33)
    norm_fp = norm_col.save(rec_dir)

    cued_col = PinchDataCollector(default_subject="minh")
    cued_col.start_cued_session(cues_per_trigger=1, window_ms=500, gap_ms=1000, seed=1, triggers=["left_click"])
    cued_col.tick_cued(0)
    cued_col.add_cued_frame([HandInput("Right", h, h, confidence=0.95, timestamp_ms=0)], 0)
    cued_col.tick_cued(1000)
    cued_col.add_cued_frame([HandInput("Right", h, h, confidence=0.95, timestamp_ms=1000)], 1000)
    cued_col.stop_cued_session()
    cued_fp = cued_col.save_cued(rec_dir)

    with caplog.at_level("INFO"):
        kept_train, _ = train_mod.collect_training_files([rec_dir], data_root=tmp_path / "no_data")
    assert norm_fp in kept_train
    assert cued_fp not in kept_train

    mat = compute_corpus_condition_matrix(rec_dir)
    assert mat["total_frames"] == 6  # Only normal session counted; cued_events excluded!

    rep = replay_mod.evaluate_recordings(
        session_paths=[norm_fp, cued_fp],
        min_exposure_hours_for_claim=0.0,
        max_ft_per_hour=1e9,
    )
    assert norm_fp.name in rep["sessions_evaluated"]
    assert cued_fp.name not in rep["sessions_evaluated"]

    dom_feats = dom_mod.extract_domain_features([norm_fp, cued_fp])
    assert len(dom_feats["palm_size"]) == 6


def test_cued_session_forces_and_restores_dry_run_on_stop_and_error(tmp_path: Path):
    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    rt = AeroHandRuntime(cfg, dry_run=False, collect_dir=tmp_path / "sessions")
    rt.recall_dir = tmp_path / "recall"
    assert rt.dry_run is False

    # Simulate held mouse button -> start cued MUST release inputs and force dry_run=True
    rt._mouse_button_held = True
    rt.collector_start_cued(cues_per_trigger=1, window_ms=500, gap_ms=1000, seed=7, triggers=["left_click"])
    assert rt.dry_run is True
    assert rt.input_ctl.dry_run is True
    assert rt._mouse_button_held is False

    # Normal stop restores dry_run=False
    rt.collector_stop_cued()
    assert rt.dry_run is False
    assert rt.input_ctl.dry_run is False

    # Error / shutdown path also restores dry_run=False
    rt.collector_start_cued(cues_per_trigger=1, window_ms=500, gap_ms=1000, seed=7, triggers=["left_click"])
    assert rt.dry_run is True
    rt._restore_cued_dry_run()
    assert rt.dry_run is False


def test_cued_session_replays_expected_trigger_and_resets_lost_hand(tmp_path: Path):
    from aerohand.ml.recording_io import load_recording_frames

    h_open = _make_open_hand(forward_z=-0.25)
    h_pinch = _pinch_middle_copy(h_open)

    col = PinchDataCollector(default_subject="minh")
    col.start_cued_session(cues_per_trigger=1, hand="Right", window_ms=1200, gap_ms=1000, seed=3, triggers=["left_click"])

    # Lead-in open frames (0..990ms)
    for ts in range(0, 1000, 33):
        col.tick_cued(ts)
        col.add_cued_frame([HandInput("Right", h_open, h_open, confidence=0.95, timestamp_ms=ts)], ts)
    # Cue issues at ts=1000 -> open -> pinch (4 frames) -> release (4 frames) -> hand-absent frame
    ts = 1000
    col.tick_cued(ts)
    col.add_cued_frame([HandInput("Right", h_open, h_open, confidence=0.95, timestamp_ms=ts)], ts)
    for k in range(1, 5):
        t = ts + k * 33
        col.tick_cued(t)
        col.add_cued_frame([HandInput("Right", h_pinch, h_pinch, confidence=0.95, timestamp_ms=t)], t)
    for k in range(5, 11):
        t = ts + k * 33
        col.tick_cued(t)
        col.add_cued_frame([HandInput("Right", h_open, h_open, confidence=0.95, timestamp_ms=t)], t)
    # Lost hand frame at t=1400ms
    col.tick_cued(1400)
    col.add_cued_frame([], 1400)
    col.stop_cued_session()
    fp = col.save_cued(tmp_path)

    frames, meta = load_recording_frames(fp)
    cue = meta["cue_events"][0]
    cfg = load_config()
    cfg["palm_wheel"]["enabled"] = False
    clf = Heuristic3DPinchClassifier((cfg.get("classifier") or {}).get("heuristic_3d"))
    eng = HandCursorEngine(cfg, classifier=clf)
    eng.anchor_cursor("Right", 960.0, 540.0)
    clicks_in_window = 0
    lost_hand_reset_seen = False
    for fr in frames:
        t = float(fr["timestamp_ms"])
        if not fr["hand_present"] or not fr["landmarks_2d"]:
            eng.process_frame([], t, 1920.0, 1080.0)
            if "Right" not in eng._active_hands:
                lost_hand_reset_seen = True
            continue
        outs = eng.process_frame(
            [HandInput(fr["handedness"], fr["landmarks_2d"], fr["world_landmarks"], fr["confidence"], int(t))],
            t,
            1920.0,
            1080.0,
        )
        if outs and outs[0].pose.is_clicking and cue["t_start_ms"] <= t <= cue["t_end_ms"]:
            clicks_in_window += 1
    assert clicks_in_window >= 1
    assert lost_hand_reset_seen is True


# ==============================================================================
# V060-07: Read-Only Dataset & Cued Recall Auditor (`scripts/data_audit.py`)
# ==============================================================================


def test_data_audit_catches_duplicates_sealed_overlap_and_bad_cues(tmp_path: Path):
    import json

    from aerohand.ml.recording_io import sha256_file

    audit_mod = _load_script_module("scripts/data_audit.py", "data_audit_v060_07_a")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    sess_dir = tmp_path / "sessions"
    neg_dir = tmp_path / "negatives"
    recall_dir = tmp_path / "recall"
    sealed_dir = tmp_path / "heldout_aerohand"
    for d in (sess_dir, neg_dir, recall_dir, sealed_dir):
        d.mkdir(parents=True)

    # 1. Valid session + exact duplicate copy -> duplicate SHA-256 error
    h = _make_open_hand(forward_z=-0.25)
    col = PinchDataCollector(default_subject="minh")
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    for i in range(10):
        col.add_frame(h, h, "Right", 0.95, i * 33)
    s1 = col.save(sess_dir)
    s_dup = sess_dir / "dup_copy.json"
    s_dup.write_bytes(s1.read_bytes())

    # 2. Sealed manifest containing a hash that overlaps with a file in neg_dir
    neg_fp = neg_dir / "leaked_neg.json"
    neg_fp.write_bytes(s1.read_bytes())
    (sealed_dir / "sealed_manifest.json").write_text(
        json.dumps({"files": [{"filename": "sealed_01.json", "sha256": sha256_file(neg_fp)}]}),
        encoding="utf-8",
    )

    # 3. Draft file under _drafts/ MUST be ignored
    drafts = sess_dir / "_drafts"
    drafts.mkdir()
    (drafts / "draft_ignore.json").write_text("{corrupt json ignored", encoding="utf-8")

    # 4. Bad cued recall file: duplicate cue_id + overlapping windows + wrong pose for trigger
    bad_cued = recall_dir / "bad_cued.json"
    bad_cued.write_text(
        json.dumps(
            {
                "schema": "aerohand_pinch_v2",
                "schema_version": 2,
                "recording_kind": "cued_events",
                "session_meta": {"subject": "minh", "hand": "Right"},
                "cue_events": [
                    {
                        "cue_id": "cue_0001",
                        "pose": PoseName.THUMB_MIDDLE_PINCH.value,
                        "expected_trigger": "left_click",
                        "hand": "Right",
                        "cue_ts_ms": 1000,
                        "window_ms": 1000,
                        "t_start_ms": 1000,
                        "t_end_ms": 2000,
                    },
                    {
                        "cue_id": "cue_0001",  # duplicate!
                        "pose": PoseName.THUMB_INDEX_PINCH.value,  # wrong pose for left_click!
                        "expected_trigger": "left_click",
                        "hand": "Right",
                        "cue_ts_ms": 1500,  # overlaps [1000, 2000]!
                        "window_ms": 1000,
                        "t_start_ms": 1500,
                        "t_end_ms": 2500,
                    },
                ],
                "samples": [],
            }
        ),
        encoding="utf-8",
    )

    rep = audit_mod.run_data_audit(
        sessions_dirs=[sess_dir],
        negatives_dirs=[neg_dir],
        recall_dirs=[recall_dir],
        sealed_root=sealed_dir,
    )
    assert rep["integrity_ok"] is False
    assert rep["passed"] is False
    assert len(rep["duplicates"]) >= 1
    assert len(rep["sealed_overlaps"]) >= 1
    err_text = " ".join(rep["blocking_errors"]).lower()
    assert "duplicate" in err_text
    assert "sealed" in err_text
    assert "cue_id" in err_text or "overlap" in err_text
    assert "mismatch" in err_text or "pose" in err_text


def test_data_audit_reports_clean_partial_corpus_and_allow_partial_exit(tmp_path: Path):
    audit_mod = _load_script_module("scripts/data_audit.py", "data_audit_v060_07_b")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    sess_dir = tmp_path / "sessions"
    recall_dir = tmp_path / "recall"
    sess_dir.mkdir()
    recall_dir.mkdir()

    h = _make_open_hand(forward_z=-0.25)
    col = PinchDataCollector(default_subject="minh")
    col.set_recording(True)
    col.set_pose(PoseName.IDLE)
    for i in range(30):
        col.add_frame(h, h, "Right", 0.95, i * 33)
    col.save(sess_dir)

    cued = PinchDataCollector(default_subject="minh")
    cued.start_cued_session(cues_per_trigger=2, hand="Right", window_ms=500, gap_ms=1000, seed=5, triggers=["left_click"])
    for ts in (0, 1000, 1500, 2500, 3000):
        cued.tick_cued(ts)
        cued.add_cued_frame([HandInput("Right", h, h, confidence=0.95, timestamp_ms=ts)], ts)
    cued.save_cued(recall_dir)

    rep = audit_mod.run_data_audit(
        sessions_dirs=[sess_dir],
        recall_dirs=[recall_dir],
        min_cues_per_trigger=50,
    )
    assert rep["integrity_ok"] is True
    assert rep["ready_for_training"] is False  # partial dataset below 21k / 50 cues
    assert rep["cued_recall"]["by_trigger"]["left_click"] == 2
    assert rep["idle_hours_by_source"]["sessions"] > 0.0
    assert any("hand" in w.lower() or "balance" in w.lower() for w in rep["warnings"])

    # CLI exits non-zero without --allow-partial, and 0 with --allow-partial when integrity_ok is True
    with pytest.raises(SystemExit) as exc_strict:
        audit_mod.main(["--sessions", str(sess_dir), "--recall", str(recall_dir), "--json"])
    assert exc_strict.value.code != 0

    audit_mod.main(["--sessions", str(sess_dir), "--recall", str(recall_dir), "--allow-partial", "--json"])


# ==============================================================================
# V060-03: Tiered False-Trigger Ceilings (Tier A <= 2/h, B <= 6/h, C <= 6/h)
# ==============================================================================


def test_tiered_ft_ucb_gate_and_promotion_checks():
    from aerohand.core.types import ALL_POSE_NAMES

    replay_mod = _load_replay_eval_module("replay_eval_v060_03_tiers")
    promo_mod = _load_script_module("scripts/check_onnx_promotion.py", "promo_v060_03_tiers")

    cfg = load_config()
    assert "eval" in cfg and "ft_tiers" in cfg["eval"]
    assert cfg["eval"]["ft_tiers"]["A"]["max_ucb_per_hour"] == pytest.approx(2.0)
    assert cfg["eval"]["ft_tiers"]["B"]["max_ucb_per_hour"] == pytest.approx(6.0)
    assert cfg["eval"]["ft_tiers"]["C"]["max_ucb_per_hour"] == pytest.approx(6.0)

    all_high = {p.value: {"precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 500} for p in ALL_POSE_NAMES}
    metrics_ok = {"model_hash": "a" * 24, "subject_heldout_metrics": {"per_class": all_high}}

    def _by_type(**overrides: int) -> dict[str, int]:
        base = {k: 0 for k in replay_mod.ALL_TRIGGER_TYPES}
        base.update(overrides)
        return base

    # 1. At 1.5h: 0 Tier A, 3 Tier B (UCB=5.169<=6), 3 Tier C (UCB=5.169<=6) -> MUST PASS!
    sealed_pass = {
        "model_hash": "a" * 24,
        "heldout_audit": {"touched": 1, "model_hash": "a" * 24},
        "false_triggers": {
            "hand_present_idle_hours": 1.5,
            "total": 6,
            "by_type": _by_type(scroll_stroke=3, dictation_start=3),
        },
    }
    res_pass = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok,
        sealed_eval_report=sealed_pass,
        latency_p95_ms=1.2,
        config=cfg,
        stage="veto",
    )
    assert res_pass["passed"] is True
    tiers = res_pass["criteria"]["sealed_negatives_ft_ucb"]["tiers"]
    assert tiers["A"]["passed"] is True and tiers["A"]["count"] == 0
    assert tiers["B"]["passed"] is True and tiers["B"]["count"] == 3
    assert tiers["C"]["passed"] is True and tiers["C"]["count"] == 3

    # 2. At 1.5h: 1 Tier A click (UCB=3.163 > 2.0) -> MUST FAIL!
    sealed_fail_a = {
        "model_hash": "a" * 24,
        "heldout_audit": {"touched": 1, "model_hash": "a" * 24},
        "false_triggers": {
            "hand_present_idle_hours": 1.5,
            "total": 1,
            "by_type": _by_type(left_click=1),
        },
    }
    res_a = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok, sealed_eval_report=sealed_fail_a, latency_p95_ms=1.2, config=cfg, stage="veto"
    )
    assert res_a["passed"] is False
    assert res_a["criteria"]["sealed_negatives_ft_ucb"]["tiers"]["A"]["passed"] is False

    # 3. At 1.5h: 4 Tier B scroll (UCB=6.102 > 6.0) -> MUST FAIL!
    sealed_fail_b = {
        "model_hash": "a" * 24,
        "heldout_audit": {"touched": 1, "model_hash": "a" * 24},
        "false_triggers": {
            "hand_present_idle_hours": 1.5,
            "total": 4,
            "by_type": _by_type(scroll_stroke=4),
        },
    }
    res_b = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok, sealed_eval_report=sealed_fail_b, latency_p95_ms=1.2, config=cfg, stage="veto"
    )
    assert res_b["passed"] is False
    assert res_b["criteria"]["sealed_negatives_ft_ucb"]["tiers"]["B"]["passed"] is False

    # 4. Missing `by_type` -> MUST FAIL explicitly (no silent pooled fallback)!
    sealed_no_by_type = {
        "model_hash": "a" * 24,
        "heldout_audit": {"touched": 1, "model_hash": "a" * 24},
        "false_triggers": {"hand_present_idle_hours": 1.6, "total": 0, "total_upper_bound_per_hour": 1.87},
    }
    res_missing = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok, sealed_eval_report=sealed_no_by_type, latency_p95_ms=1.2, config=cfg, stage="veto"
    )
    assert res_missing["passed"] is False
    assert "by_type" in res_missing["criteria"]["sealed_negatives_ft_ucb"]["detail"]

    # 5. Unknown trigger type in `by_type` -> MUST FAIL explicitly!
    bad_bt = _by_type()
    bad_bt["unmapped_trigger"] = 1
    sealed_unknown = {
        "model_hash": "a" * 24,
        "heldout_audit": {"touched": 1, "model_hash": "a" * 24},
        "false_triggers": {"hand_present_idle_hours": 1.6, "total": 1, "by_type": bad_bt},
    }
    res_unk = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok, sealed_eval_report=sealed_unknown, latency_p95_ms=1.2, config=cfg, stage="veto"
    )
    assert res_unk["passed"] is False
    assert "unknown" in res_unk["criteria"]["sealed_negatives_ft_ucb"]["detail"].lower()

    # 6. Config override of `eval.ft_tiers` is respected!
    strict_cfg = load_config()
    strict_cfg["eval"]["ft_tiers"]["B"]["max_ucb_per_hour"] = 4.0
    res_ovr = promo_mod.check_onnx_promotion(
        metrics_report=metrics_ok, sealed_eval_report=sealed_pass, latency_p95_ms=1.2, config=strict_cfg, stage="veto"
    )
    assert res_ovr["passed"] is False
    assert res_ovr["criteria"]["sealed_negatives_ft_ucb"]["tiers"]["B"]["passed"] is False


def test_replay_eval_emits_tiered_ft_report_and_respects_config_override(tmp_path: Path):
    import json

    replay_mod = _load_replay_eval_module("replay_eval_v060_03_rep")
    h = _make_open_hand(forward_z=-0.25)
    flat = [c for lm in h for c in (lm.x, lm.y, lm.z)]
    samples = [
        {
            "timestamp_ms": i * 33,
            "label": 0,
            "label_name": "idle",
            "handedness": "Right",
            "hand_id": "Right",
            "confidence": 0.95,
            "landmarks_2d": flat,
            "world_landmarks": flat,
        }
        for i in range(15)
    ]
    fp = tmp_path / "idle_short.json"
    fp.write_text(
        json.dumps({"schema": "aerohand_pinch_v2", "session_meta": {"subject": "minh"}, "samples": samples}),
        encoding="utf-8",
    )
    cfg = load_config()
    cfg["eval"]["ft_tiers"]["A"]["max_ucb_per_hour"] = 1.5
    rep = replay_mod.evaluate_recordings(negative_paths=[fp], config=cfg, min_exposure_hours_for_claim=0.0)
    tiers = rep["false_triggers"]["tiers"]
    assert set(tiers.keys()) == {"A", "B", "C"}
    assert tiers["A"]["max_ucb_per_hour"] == pytest.approx(1.5)
    assert tiers["A"]["count"] == 0
    assert "garwood_exposure_table" in tiers["A"]


# ==============================================================================
# V060-01b: `veto` Replay Backend & Paired Event Recall (`scripts/eval_event_recall.py`)
# ==============================================================================


def test_replay_eval_supports_veto_backend_without_latency_fallback():
    from aerohand.ml.onnx_classifier import GuardedClassifier

    replay_mod = _load_replay_eval_module("replay_eval_v060_01b_veto")
    cfg = load_config()
    cfg["classifier"]["veto_confirm_frames"] = 3
    cfg["classifier"]["veto_min_prob"] = 0.42

    # Stub ONNX classifier passed via helper or verify CLI builder sets max_p95_ms >= 1e6
    clf = replay_mod.build_replay_classifier(
        "veto",
        cfg,
        onnx_classifier=_ScriptedClf([(PoseName.IDLE, 0.0, 1.0, True)]),
    )
    assert isinstance(clf, GuardedClassifier)
    assert clf.mode == "veto"
    assert clf.veto_confirm_frames == 3
    assert clf.veto_min_prob == pytest.approx(0.42)
    assert clf.max_p95_ms >= 1e6  # Latency guard disabled in offline replay


def test_eval_event_recall_wilson_ci_paired_gate_and_sealed_refusal(tmp_path: Path):
    import json

    recall_mod = _load_script_module("scripts/eval_event_recall.py", "eval_recall_v060_01b")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")

    # 1. Exact Wilson 95% CI sanity check
    lo, hi = recall_mod.wilson_interval(49, 50)
    assert 0.89 < lo < 0.91 and 0.99 < hi <= 1.0

    # 2. Build a synthetic cued_events recording with 2 `left_click` cues (both perform a clean pinch+release)
    h_open = _make_open_hand(forward_z=-0.25)
    h_pinch = _pinch_middle_copy(h_open)

    col = PinchDataCollector(default_subject="minh")
    col.start_cued_session(cues_per_trigger=2, hand="Right", window_ms=1000, gap_ms=1000, seed=11, triggers=["left_click"])
    # Cue 1 starts at t=1000..2000; Cue 2 starts at t=3000..4000
    t = 0
    while t <= 4200:
        col.tick_cued(t)
        if not col.is_cued_active:
            break
        in_pinch_1 = 1066 <= t <= 1198
        in_pinch_2 = 3066 <= t <= 3198
        if t == 2500:
            col.add_cued_frame([], t)  # Hand-absent gap between cues
        else:
            h_cur = h_pinch if (in_pinch_1 or in_pinch_2) else h_open
            col.add_cued_frame([HandInput("Right", h_cur, h_cur, confidence=0.95, timestamp_ms=t)], t)
        t += 33
    fp = col.save_cued(tmp_path / "recall")

    cfg = load_config()
    cfg["palm_wheel"]["enabled"] = False

    # 3a. Approving ONNX veto (always approves thumb_middle_pinch) -> 2/2 hits, delta = 0 pp -> PASS!
    approve_onnx = _ScriptedClf([(PoseName.THUMB_MIDDLE_PINCH, 0.92, 0.05, True)])
    rep_ok = recall_mod.evaluate_paired_event_recall(
        [fp],
        config=cfg,
        onnx_classifier=approve_onnx,
        max_recall_drop=0.02,
    )
    assert rep_ok["heuristic"]["by_trigger"]["left_click"]["n"] == 2
    assert rep_ok["heuristic"]["by_trigger"]["left_click"]["hits"] == 2
    assert rep_ok["veto"]["by_trigger"]["left_click"]["hits"] == 2
    assert rep_ok["paired"]["by_trigger"]["left_click"]["recall_delta_pp"] == pytest.approx(0.0)
    assert rep_ok["paired"]["passed"] is True

    # 3b. Always-idle ONNX veto -> blocks both clicks -> heuristic 2/2 vs veto 0/2 (-100 pp) -> FAIL!
    reject_onnx = _ScriptedClf([(PoseName.IDLE, 0.02, 0.96, True)])
    rep_fail = recall_mod.evaluate_paired_event_recall(
        [fp],
        config=cfg,
        onnx_classifier=reject_onnx,
        max_recall_drop=0.02,
    )
    assert rep_fail["heuristic"]["by_trigger"]["left_click"]["hits"] == 2
    assert rep_fail["veto"]["by_trigger"]["left_click"]["hits"] == 0
    assert rep_fail["paired"]["by_trigger"]["left_click"]["heuristic_only_hits"] == 2
    assert rep_fail["paired"]["by_trigger"]["left_click"]["passed"] is False
    assert rep_fail["paired"]["passed"] is False

    # 4. Must refuse heldout paths!
    ho_dir = tmp_path / "heldout_aerohand"
    ho_dir.mkdir()
    ho_fp = ho_dir / "sealed_cued.json"
    ho_fp.write_text(json.dumps({"recording_kind": "cued_events", "cue_events": [], "samples": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Held-out"):
        recall_mod.evaluate_event_recall([ho_fp], config=cfg)


# ==============================================================================
# V060-06: `--passive-record` Chunked Dry-Run Dogfood Idle Recorder & Mark API
# ==============================================================================


def test_passive_idle_recorder_chunks_segments_daily_cap_and_deliberate_marks(tmp_path: Path):
    import json

    from aerohand.ml.collector import PassiveIdleRecorder
    from aerohand.ml.recording_io import load_recording_frames

    replay_mod = _load_replay_eval_module("replay_eval_v060_06_passive")
    h_open = _make_open_hand(forward_z=-0.25)

    out_dir = tmp_path / "passive"
    rec = PassiveIdleRecorder(
        out_dir,
        default_subject="minh",
        chunk_max_frames=10,
        max_daily_mb=10.0,
        mark_half_window_ms=1500.0,
    )
    # Feed 25 frames with a 300ms gap (>100ms -> segment_id increments) + 1 deliberate mark at t=5000ms
    timestamps = [i * 33 for i in range(10)] + [1000 + i * 33 for i in range(10)] + [5000 + i * 33 for i in range(5)]
    for ts in timestamps:
        rec.add_frame([HandInput("Right", h_open, h_open, confidence=0.95, timestamp_ms=ts)], ts)
    mark_res = rec.mark_deliberate_gesture(5000)
    assert mark_res["ok"] is True
    assert mark_res["marks_count"] == 1
    summary = rec.close()

    # Chunked flushing: 25 frames / chunk_max_frames=10 -> 3 chunk files written, RAM buffer empty
    assert summary["chunks_written"] == 3
    assert len(rec._buf) == 0
    chunk_files = sorted(out_dir.glob("*.json"))
    assert len(chunk_files) == 3

    # Verify schema v2, kind, segment_id bump across gap, and deliberate mark saved in chunk 3
    c1 = json.loads(chunk_files[0].read_text(encoding="utf-8"))
    c2 = json.loads(chunk_files[1].read_text(encoding="utf-8"))
    c3 = json.loads(chunk_files[2].read_text(encoding="utf-8"))
    assert c1["schema"] == "aerohand_pinch_v2"
    assert c1["session_meta"]["kind"] == "passive_idle_claim"
    assert c1["session_meta"]["subject"] == "minh"
    assert c2["samples"][0]["segment_id"] > c1["samples"][-1]["segment_id"]
    assert 5000 in c3["deliberate_marks_ms"]
    assert c3["deliberate_marks_count"] == 1

    # Loader + replay_eval preserve deliberate_marks_ms and report deliberate_marks_count
    _frs, meta3 = load_recording_frames(chunk_files[2])
    assert meta3["deliberate_marks_ms"] == [5000]
    rep = replay_mod.evaluate_recordings(
        negative_paths=chunk_files,
        min_exposure_hours_for_claim=0.0,
        max_ft_per_hour=1e9,
    )
    assert rep["false_triggers"]["deliberate_marks_count"] == 1
    # Chunk 3 frames (t=5000..5132) are within +/-1.5s of mark=5000 -> excluded from idle exposure!
    assert rep["false_triggers"]["hand_present_idle_hours"] < (len(timestamps) * 33.0 / 3_600_000.0)

    # Daily cap enforcement: tiny cap (0.0001 MB ~ 100 bytes) prevents writing further chunks after limit
    cap_dir = tmp_path / "passive_capped"
    capped = PassiveIdleRecorder(cap_dir, default_subject="minh", chunk_max_frames=5, max_daily_mb=0.0001)
    for i in range(15):
        capped.add_frame([HandInput("Right", h_open, h_open, confidence=0.95, timestamp_ms=i * 33)], i * 33)
    cap_sum = capped.close()
    assert cap_sum["cap_reached"] is True
    assert len(list(cap_dir.glob("*.json"))) <= 1


def test_passive_record_runtime_forces_dry_run_and_api_mark_endpoint(tmp_path: Path):
    import json
    import urllib.request

    from aerohand.runtime import AeroHandRuntime

    cfg = load_config()
    cfg["audio"]["enabled"] = False
    cfg["api"]["enabled"] = True
    cfg["api"]["port"] = 0  # Ephemeral localhost port
    assert cfg["hotkeys"]["mark_deliberate_gesture"] == "cmd+ctrl+opt+m"

    rt = AeroHandRuntime(
        cfg,
        dry_run=False,  # MUST be overridden to True by passive_record_dir!
        passive_record_dir=tmp_path / "passive_rt",
    )
    assert rt.dry_run is True
    assert rt.input_ctl.dry_run is True
    # Attempting to switch to Live HID while passive recording is active MUST be refused!
    rt.set_dry_run(False)
    assert rt.dry_run is True
    assert rt.input_ctl.dry_run is True

    # Start Local Control API and hit POST /api/passive/mark
    rt._last_camera_ts_ms = 2400
    rt._start_api_if_enabled()
    try:
        assert rt._api_server is not None
        port = rt._api_server.bound_port
        tok = rt._api_server.token
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/passive/mark",
            data=b"{}",
            headers={"Content-Type": "application/json", "X-AeroHand-Token": tok},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
        assert body["ok"] is True
        assert body["timestamp_ms"] == 2400
        assert body["marks_count"] == 1
    finally:
        rt.stop()
