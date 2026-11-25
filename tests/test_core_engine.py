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
