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
