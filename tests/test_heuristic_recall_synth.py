"""
Synthetic 5-Class Recall & Guard-Edge-Case Tests for Heuristic3DPinchClassifier (V051-09).

Verifies whether the 4 data-driven intent gates (`approach`, `motion`, `posture`, `grip`)
and the `fist` guard accidentally break recall on any of the 5 non-index-pinch classes
under realistic linear closing trajectories and natural finger curl.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from synth_hand import (  # noqa: E402
    apply_finger_curl,
    make_airpoint_ergonomic_sequence,
    make_base_open_hand,
    make_fist_thumb_on_index_sequence,
    make_ok_sign_pinch_sequence,
    make_pinch_frame,
    make_synth_pinch_sequence,
)

from aerohand.config import load_config  # noqa: E402
from aerohand.core.engine import HandCursorEngine  # noqa: E402
from aerohand.core.types import HandInput, PoseName  # noqa: E402
from aerohand.ml.heuristic_3d import Heuristic3DPinchClassifier  # noqa: E402


def _run_seq_and_collect_hold_poses(clf: Heuristic3DPinchClassifier, seq, hold_tail: int = 6):
    preds = []
    for ts_ms, lms in seq:
        preds.append(clf.predict(lms, lms, "Right", ts_ms))
    return preds[-hold_tail:]


def test_synth_recall_thumb_middle_closure_0p1s_and_0p5s():
    cfg = load_config()["classifier"]["heuristic_3d"]
    for close_frames in (3, 15):  # 0.1s and 0.5s at 30 fps
        clf = Heuristic3DPinchClassifier(cfg)
        seq = make_synth_pinch_sequence(
            PoseName.THUMB_MIDDLE_PINCH,
            open_frames=5,
            close_frames=close_frames,
            hold_frames=6,
            fps=30.0,
        )
        tail = _run_seq_and_collect_hold_poses(clf, seq)
        assert all(p.pose == PoseName.THUMB_MIDDLE_PINCH for p in tail), (
            f"close_frames={close_frames} failed: {[p.pose for p in tail]}, dbg={clf.last_debug.get('Right')}"
        )


def test_synth_recall_thumb_ring_with_natural_curl():
    # Ring pinch while pinky naturally half-curls (index stays extended for aiming)
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    seq = make_synth_pinch_sequence(
        PoseName.THUMB_RING_PINCH,
        open_frames=5,
        close_frames=6,
        hold_frames=6,
        curl_fingers=("pinky",),
        curl_amount=0.55,
    )
    tail = _run_seq_and_collect_hold_poses(clf, seq)
    assert all(p.pose == PoseName.THUMB_RING_PINCH for p in tail), (
        f"ring curl failed: {[p.pose for p in tail]}, dbg={clf.last_debug.get('Right')}"
    )


def test_synth_recall_thumb_pinky_pinch_with_curled_others():
    # Pinky tip pinch while middle & ring naturally curl tightly AND pinky curves toward thumb
    # -> 3 fingers curled into palm (middle, ring, pinky) while index stays extended!
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    seq = make_synth_pinch_sequence(
        PoseName.THUMB_PINKY_PINCH,
        open_frames=5,
        close_frames=6,
        hold_frames=6,
        curl_fingers=("middle", "ring", "pinky"),
        curl_amount=0.82,
    )
    tail = _run_seq_and_collect_hold_poses(clf, seq)
    assert all(p.pose == PoseName.THUMB_PINKY_PINCH for p in tail), (
        f"pinky pinch curled failed: {[p.pose for p in tail]}, dbg={clf.last_debug.get('Right')}"
    )


def test_synth_recall_thumb_pinky_base_half_curled():
    # Thumb reaches across palm to pinky MCP while middle, ring, pinky naturally half-curl (>=3 half-curled)
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    seq = make_synth_pinch_sequence(
        PoseName.THUMB_PINKY_BASE,
        open_frames=5,
        close_frames=6,
        hold_frames=6,
        curl_fingers=("middle", "ring", "pinky"),
        curl_amount=0.60,
    )
    tail = _run_seq_and_collect_hold_poses(clf, seq)
    assert all(p.pose == PoseName.THUMB_PINKY_BASE for p in tail), (
        f"pinky base half-curled failed: {[p.pose for p in tail]}, dbg={clf.last_debug.get('Right')}"
    )


def test_synth_recall_dual_pinch_drag():
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    seq = make_synth_pinch_sequence(
        PoseName.THUMB_INDEX_MIDDLE_PINCH,
        open_frames=5,
        close_frames=6,
        hold_frames=6,
        curl_fingers=("ring", "pinky"),
        curl_amount=0.55,
    )
    tail = _run_seq_and_collect_hold_poses(clf, seq)
    assert all(p.pose == PoseName.THUMB_INDEX_MIDDLE_PINCH for p in tail), (
        f"dual drag failed: {[p.pose for p in tail]}, dbg={clf.last_debug.get('Right')}"
    )


def test_fist_guard_does_not_cancel_latched_pinch():
    # Establish a latched THUMB_PINKY_PINCH cleanly, then curl middle+ring+pinky tightly mid-hold
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    base = make_base_open_hand()

    for i in range(5):
        clf.predict(base, base, "Right", i * 33.0)
    pinched = make_pinch_frame(base, PoseName.THUMB_PINKY_PINCH, 1.0)
    pred_enter = clf.predict(pinched, pinched, "Right", 200.0)
    assert pred_enter.pose == PoseName.THUMB_PINKY_PINCH
    assert PoseName.THUMB_PINKY_PINCH.value in clf.last_debug["Right"]["latched"]

    # Now curl middle + ring + pinky tightly toward palm while keeping thumb firmly on pinky tip
    curled_hold = apply_finger_curl(base, ("middle", "ring", "pinky"), curl_amount=0.85)
    curled_hold = make_pinch_frame(curled_hold, PoseName.THUMB_PINKY_PINCH, 1.0)
    pred_hold = clf.predict(curled_hold, curled_hold, "Right", 233.0)
    assert pred_hold.pose == PoseName.THUMB_PINKY_PINCH, (
        f"Fist guard cancelled latched pinch! got={pred_hold.pose}, dbg={clf.last_debug.get('Right')}"
    )


def test_synth_static_contact_without_approach_stays_idle():
    cfg = load_config()["classifier"]["heuristic_3d"]
    for pose in (
        PoseName.THUMB_INDEX_PINCH,
        PoseName.THUMB_MIDDLE_PINCH,
        PoseName.THUMB_RING_PINCH,
        PoseName.THUMB_PINKY_PINCH,
        PoseName.THUMB_PINKY_BASE,
        PoseName.THUMB_INDEX_MIDDLE_PINCH,
    ):
        clf = Heuristic3DPinchClassifier(cfg)
        seq = make_synth_pinch_sequence(pose, open_frames=3, close_frames=4, hold_frames=6, start_closed=True)
        for ts_ms, lms in seq:
            pred = clf.predict(lms, lms, "Right", ts_ms)
            assert pred.pose == PoseName.IDLE, f"Static {pose.value} leaked through approach gate!"


@pytest.mark.parametrize("pose", [PoseName.THUMB_MIDDLE_PINCH, PoseName.THUMB_INDEX_MIDDLE_PINCH])
def test_synth_smoke_pose_sequence_finite(pose):
    seq = make_synth_pinch_sequence(pose, open_frames=2, close_frames=3, hold_frames=2)
    assert len(seq) == 7


def _run_engine_dictation_armed(cfg, seq) -> bool:
    eng_cfg = dict(cfg)
    eng_cfg["smoothing"] = dict(cfg["smoothing"], enable_landmark_smoothing=False)
    eng_cfg["cursor"] = dict(cfg["cursor"], reacquire_skip_frames=0)
    eng_cfg["palm_wheel"] = dict(cfg["palm_wheel"], enabled=False)
    clf = Heuristic3DPinchClassifier(eng_cfg["classifier"]["heuristic_3d"])
    eng = HandCursorEngine(eng_cfg, classifier=clf)
    armed = False
    for ts_ms, lms in seq:
        outs = eng.process_frame([HandInput("Right", lms, lms, timestamp_ms=int(ts_ms))], ts_ms, 1920.0, 1080.0)
        if outs and (outs[0].pose.dictation_just_started or outs[0].pose.is_dictating):
            armed = True
    return armed


def test_synth_fist_thumb_on_index_does_not_arm_dictation():
    cfg = load_config()
    seq = make_fist_thumb_on_index_sequence(open_frames=5, close_frames=6, hold_frames=18, fps=30.0)
    assert _run_engine_dictation_armed(cfg, seq) is False, (
        "Closed fist with thumb resting on curled index tip should NOT arm dictation!"
    )


def test_synth_ok_sign_thumb_index_pinch_still_arms():
    cfg = load_config()
    for curl_others in (False, True):
        seq = make_ok_sign_pinch_sequence(
            open_frames=5, close_frames=6, hold_frames=18, fps=30.0, curl_others=curl_others
        )
        assert _run_engine_dictation_armed(cfg, seq) is True, (
            f"Real OK-sign thumb_index_pinch failed to arm dictation (curl_others={curl_others})!"
        )


def test_fist_guard_index_curled_param_off_restores_legacy():
    cfg = load_config()
    assert cfg["classifier"]["heuristic_3d"]["fist_guard_requires_index_curled"] is True
    legacy_cfg = dict(cfg)
    legacy_cfg["classifier"] = dict(
        cfg["classifier"],
        heuristic_3d=dict(cfg["classifier"]["heuristic_3d"], fist_guard_requires_index_curled=False),
    )
    seq = make_fist_thumb_on_index_sequence(open_frames=5, close_frames=6, hold_frames=18, fps=30.0)
    assert _run_engine_dictation_armed(cfg, seq) is False
    assert _run_engine_dictation_armed(legacy_cfg, seq) is True


def test_fist_guard_preserves_long_held_drag_when_all_four_curl():
    """V053-01 (B1): Drag latched >= 600 ms MUST survive if all 4 fingertips curl < 0.72 mid-drag."""
    from aerohand.core.palm_frame import INDEX_TIP, MIDDLE_TIP, THUMB_TIP

    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    base = make_base_open_hand()

    # Open frames -> enter clean dual-pinch drag -> hold >= 600 ms
    for i in range(5):
        clf.predict(base, base, "Right", i * 33.0)
    drag_frame = make_pinch_frame(base, PoseName.THUMB_INDEX_MIDDLE_PINCH, 1.0)
    t_enter = 200.0
    p_enter = clf.predict(drag_frame, drag_frame, "Right", t_enter)
    assert p_enter.pose == PoseName.THUMB_INDEX_MIDDLE_PINCH
    assert PoseName.THUMB_INDEX_MIDDLE_PINCH.value in clf.last_debug["Right"]["latched"]

    # Hold steady for 650 ms (step every 50 ms so gap reset never triggers)
    for k in range(1, 14):
        p_hold = clf.predict(drag_frame, drag_frame, "Right", t_enter + k * 50.0)
        assert p_hold.pose == PoseName.THUMB_INDEX_MIDDLE_PINCH

    # Now curl all 4 non-thumb fingers tightly (< 0.72 palm width) while keeping thumb touching index+middle
    curled_drag = apply_finger_curl(base, ("index", "middle", "ring", "pinky"), curl_amount=0.88)
    mid_pt = type(base[0])(
        x=(curled_drag[INDEX_TIP].x + curled_drag[MIDDLE_TIP].x) * 0.5,
        y=(curled_drag[INDEX_TIP].y + curled_drag[MIDDLE_TIP].y) * 0.5,
        z=(curled_drag[INDEX_TIP].z + curled_drag[MIDDLE_TIP].z) * 0.5,
    )
    curled_drag[INDEX_TIP] = mid_pt
    curled_drag[MIDDLE_TIP] = mid_pt
    curled_drag[THUMB_TIP] = mid_pt

    t_curled = t_enter + 700.0
    for j in range(5):
        p_after = clf.predict(curled_drag, curled_drag, "Right", t_curled + j * 33.0)
        assert p_after.pose == PoseName.THUMB_INDEX_MIDDLE_PINCH, (
            f"Long-held drag dropped on frame {j} when all 4 fingers curled! "
            f"got={p_after.pose}, dbg={clf.last_debug.get('Right')}"
        )
        # No unrelated pose may acquire a new latch while all 4 fingers are curled into a fist
        assert clf.last_debug["Right"]["latched"] == [PoseName.THUMB_INDEX_MIDDLE_PINCH.value]


def test_fist_guard_preserves_long_held_dictation_when_all_four_curl():
    """V053-01 (B1): Dictation latched >= 600 ms MUST survive if all 4 fingertips curl < 0.72 mid-hold."""
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    ok_seq = make_ok_sign_pinch_sequence(open_frames=5, close_frames=5, hold_frames=22, fps=30.0)
    last_ts = 0.0
    for ts_ms, lms in ok_seq:
        p = clf.predict(lms, lms, "Right", ts_ms)
        last_ts = ts_ms
    assert p.pose == PoseName.THUMB_INDEX_PINCH
    assert clf.last_debug["Right"]["latch_age_ms"]["thumb_index_pinch"] >= 600.0

    # Now curl all 4 fingers into palm (< 0.72) while thumb stays on curled index tip (same pose as fist)
    fist_tail = make_fist_thumb_on_index_sequence(open_frames=1, close_frames=1, hold_frames=5, fps=30.0)[-5:]
    for j, (_, fist_lms) in enumerate(fist_tail):
        p_after = clf.predict(fist_lms, fist_lms, "Right", last_ts + (j + 1) * 33.0)
        assert p_after.pose == PoseName.THUMB_INDEX_PINCH, (
            f"Long-held dictation dropped on frame {j} when all 4 fingers curled! "
            f"got={p_after.pose}, dbg={clf.last_debug.get('Right')}"
        )


def test_synth_slow_fist_thumb_on_index_does_not_arm_dictation():
    """V053-01: Slow fist closure (close_frames=15 = 500 ms) must still NOT arm dictation."""
    cfg = load_config()
    seq = make_fist_thumb_on_index_sequence(open_frames=5, close_frames=15, hold_frames=18, fps=30.0)
    assert _run_engine_dictation_armed(cfg, seq) is False, (
        "Slow fist closure with thumb on curled index tip should NOT arm dictation!"
    )


@pytest.mark.parametrize(
    "pose",
    [
        PoseName.THUMB_MIDDLE_PINCH,
        PoseName.THUMB_RING_PINCH,
        PoseName.THUMB_PINKY_PINCH,
        PoseName.THUMB_PINKY_BASE,
        PoseName.THUMB_INDEX_PINCH,
        PoseName.THUMB_INDEX_MIDDLE_PINCH,
    ],
)
def test_airpoint_ergonomic_shaft_and_palm_tuck_recall(pose):
    """All 6 Airpoint ergonomic postures (curled-finger shaft press & relaxed thumb-in-palm) fire cleanly."""
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    seq = make_airpoint_ergonomic_sequence(pose, open_frames=6, close_frames=6, hold_frames=6, fps=30.0)
    tail = _run_seq_and_collect_hold_poses(clf, seq)
    assert all(p.pose == pose for p in tail), (
        f"Airpoint ergonomic {pose.value} failed: {[p.pose for p in tail]}, dbg={clf.last_debug.get('Right')}"
    )
    if pose != PoseName.THUMB_INDEX_MIDDLE_PINCH:
        assert pose.value in clf.last_debug["Right"]["used_shaft"]


def test_airpoint_ergonomic_static_shaft_contact_stays_idle():
    """Static resting thumb on curled finger shaft or inner palm without prior separation stays IDLE."""
    cfg = load_config()["classifier"]["heuristic_3d"]
    for pose in (
        PoseName.THUMB_MIDDLE_PINCH,
        PoseName.THUMB_RING_PINCH,
        PoseName.THUMB_PINKY_PINCH,
        PoseName.THUMB_PINKY_BASE,
        PoseName.THUMB_INDEX_PINCH,
    ):
        clf = Heuristic3DPinchClassifier(cfg)
        seq = make_airpoint_ergonomic_sequence(pose, open_frames=3, close_frames=4, hold_frames=6, start_closed=True)
        for ts_ms, lms in seq:
            pred = clf.predict(lms, lms, "Right", ts_ms)
            assert pred.pose == PoseName.IDLE, (
                f"Static Airpoint {pose.value} leaked through shaft approach gate! dbg={clf.last_debug.get('Right')}"
            )


def test_airpoint_relaxed_palm_tuck_does_not_require_reaching_pinky_mcp():
    """4 slightly curved fingers + thumb folded into inner palm fires THUMB_PINKY_BASE even far from PINKY_MCP."""
    from aerohand.core.palm_frame import PINKY_MCP, THUMB_TIP, compute_palm_frame

    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    seq = make_airpoint_ergonomic_sequence(PoseName.THUMB_PINKY_BASE, open_frames=6, close_frames=6, hold_frames=6)
    tail = _run_seq_and_collect_hold_poses(clf, seq)
    assert all(p.pose == PoseName.THUMB_PINKY_BASE for p in tail)
    # Confirm thumb truly did NOT have to reach PINKY_MCP (> pinky_base_on_dist = 0.42)
    pf = compute_palm_frame(seq[-1][1], anchor="palm_center", is_left_hand=False, scale_metric="palm_width")
    assert pf is not None
    assert pf.dist(THUMB_TIP, PINKY_MCP) > cfg["pinky_base_on_dist"]
    # Confirm disabling Airpoint shaft press cleanly restores legacy behavior
    legacy_clf = Heuristic3DPinchClassifier(dict(cfg, enable_airpoint_shaft_press=False))
    legacy_tail = _run_seq_and_collect_hold_poses(legacy_clf, seq)
    assert all(p.pose == PoseName.IDLE for p in legacy_tail)


def _inject_shaft_eligibility_flap(lms, pose: PoseName):
    """Perturb finger extension enough to break strict `shaft_eligible[pose]` while keeping
    physical thumb-to-shaft contact `< s_off` and avoiding a full 4-finger curled fist."""
    import math

    from aerohand.core.palm_frame import (
        INDEX_MCP,
        INDEX_TIP,
        MIDDLE_MCP,
        MIDDLE_TIP,
        PINKY_MCP,
        PINKY_TIP,
        RING_MCP,
        RING_TIP,
        WRIST,
    )
    from aerohand.core.types import Landmark

    out = list(lms)
    w = out[WRIST]

    def _set_finger_ext(tip_idx: int, mcp_idx: int, target_ext: float, side_dir: tuple[float, float, float]) -> None:
        m = out[mcp_idx]
        mcp_d = math.sqrt((m.x - w.x) ** 2 + (m.y - w.y) ** 2 + (m.z - w.z) ** 2)
        dx, dy, dz = side_dir
        n = math.sqrt(dx * dx + dy * dy + dz * dz)
        r = target_ext * mcp_d / max(n, 1e-9)
        out[tip_idx] = Landmark(w.x + dx * r, w.y + dy * r, w.z + dz * r)

    if pose in (PoseName.THUMB_INDEX_PINCH, PoseName.THUMB_MIDDLE_PINCH):
        # Dip index_ext to 1.10 (< 1.15 / < 1.20) while keeping idx_dist_to_palm > 0.90 (not a fist)
        # and leaving INDEX_PIP / INDEX_DIP untouched so thumb-to-shaft contact stays < s_off.
        _set_finger_ext(INDEX_TIP, INDEX_MCP, 1.10, (-0.58, -0.33, -0.74))
    elif pose == PoseName.THUMB_RING_PINCH:
        # Dip mid_ext to 1.25 (< 1.35) while keeping thumb on ring PIP-DIP
        _set_finger_ext(MIDDLE_TIP, MIDDLE_MCP, 1.25, (0.0, -0.50, -0.86))
    elif pose == PoseName.THUMB_PINKY_PINCH:
        # Dip ring_ext to 1.25 (< 1.35) while keeping thumb on pinky PIP-DIP
        _set_finger_ext(RING_TIP, RING_MCP, 1.25, (0.25, -0.50, -0.83))
    elif pose == PoseName.THUMB_PINKY_BASE:
        # Dip pinky_ext to 1.05 (< 1.15) while keeping thumb tucked inside inner palm
        _set_finger_ext(PINKY_TIP, PINKY_MCP, 1.05, (0.55, -0.45, -0.70))
    return out


@pytest.mark.parametrize(
    "pose",
    [
        PoseName.THUMB_INDEX_PINCH,
        PoseName.THUMB_MIDDLE_PINCH,
        PoseName.THUMB_RING_PINCH,
        PoseName.THUMB_PINKY_PINCH,
        PoseName.THUMB_PINKY_BASE,
    ],
)
@pytest.mark.parametrize("flap_frames", [1, 2, 3])
def test_shaft_latched_poses_survive_transient_eligibility_flap(pose, flap_frames):
    """P0.2: After a 2 s shaft hold, a 1-3 frame eligibility flap with intact thumb-to-shaft
    contact must NOT drop the latch or trigger post-flap approach lockout for any shaft pose."""
    cfg = load_config()["classifier"]["heuristic_3d"]
    assert int(cfg["shaft_eligibility_grace_frames"]) == 3
    clf = Heuristic3DPinchClassifier(cfg)
    fps = 30.0
    dt_ms = 1000.0 / fps

    # 1. Enter cleanly and hold for 2.0 s (60 frames at 30 fps)
    seq = make_airpoint_ergonomic_sequence(pose, open_frames=6, close_frames=6, hold_frames=60, fps=fps)
    last_ts = 0.0
    hold_frame = seq[-1][1]
    for ts_ms, lms in seq:
        pred = clf.predict(lms, lms, "Right", ts_ms)
        last_ts = ts_ms
    assert pred.pose == pose
    assert pose.value in clf.last_debug["Right"]["latched"]
    assert clf.last_debug["Right"]["latch_age_ms"][pose.value] >= 1800.0

    # 2. Inject 1-3 frames where strict shaft_eligible drops to False while contact stays < s_off
    flap_frame = _inject_shaft_eligibility_flap(hold_frame, pose)
    for k in range(1, flap_frames + 1):
        last_ts += dt_ms
        p_flap = clf.predict(flap_frame, flap_frame, "Right", last_ts)
        st = clf._hands["Right"]
        assert st.shaft_ineligible_frames.get(pose, 0) == k, (
            f"Expected strict shaft_eligible to fail on flap frame {k} for {pose.value}, "
            f"dbg={clf.last_debug.get('Right')}"
        )
        assert pose.value in clf.last_debug["Right"]["latched"], (
            f"Shaft latch dropped on flap frame {k}/{flap_frames} for {pose.value}! "
            f"dbg={clf.last_debug.get('Right')}"
        )
        assert p_flap.pose == pose, (
            f"Predicted {p_flap.pose} instead of {pose} on flap frame {k}/{flap_frames}! "
            f"dbg={clf.last_debug.get('Right')}"
        )

    # 3. Return to steady hold for 5 frames: zero approach lockout & grace counter resets cleanly
    for j in range(1, 6):
        last_ts += dt_ms
        p_rec = clf.predict(hold_frame, hold_frame, "Right", last_ts)
        assert p_rec.pose == pose, (
            f"Recovery frame {j} after {flap_frames}-frame flap failed for {pose.value}: "
            f"got={p_rec.pose}, dbg={clf.last_debug.get('Right')}"
        )
        assert clf.last_debug["Right"]["blocked_reasons"] == []
        assert clf._hands["Right"].shaft_ineligible_frames.get(pose, 0) == 0


def test_shaft_grace_still_unlatches_on_sustained_ineligibility_or_open_thumb():
    """P0.2: >3 consecutive ineligible frames OR physically releasing the thumb must still unlatch."""
    cfg = load_config()["classifier"]["heuristic_3d"]
    dt_ms = 1000.0 / 30.0

    # Case A: 4 consecutive ineligible frames (> grace=3) drops the shaft latch cleanly
    clf_a = Heuristic3DPinchClassifier(cfg)
    seq = make_airpoint_ergonomic_sequence(
        PoseName.THUMB_INDEX_PINCH, open_frames=6, close_frames=6, hold_frames=60, fps=30.0
    )
    last_ts = 0.0
    for ts_ms, lms in seq:
        clf_a.predict(lms, lms, "Right", ts_ms)
        last_ts = ts_ms
    flap_frame = _inject_shaft_eligibility_flap(seq[-1][1], PoseName.THUMB_INDEX_PINCH)
    for k in range(1, 5):
        last_ts += dt_ms
        pred = clf_a.predict(flap_frame, flap_frame, "Right", last_ts)
    assert PoseName.THUMB_INDEX_PINCH.value not in clf_a.last_debug["Right"]["latched"]
    assert pred.pose == PoseName.IDLE

    # Case B: Opening the thumb immediately unlatches on frame 1 (even during an eligibility flap)
    clf_b = Heuristic3DPinchClassifier(cfg)
    for ts_ms, lms in seq:
        clf_b.predict(lms, lms, "Right", ts_ms)
        last_ts = ts_ms
    open_frame = seq[0][1]
    pred_open = clf_b.predict(open_frame, open_frame, "Right", last_ts + dt_ms)
    assert pred_open.pose == PoseName.IDLE
    assert clf_b.last_debug["Right"]["latched"] == []


def test_spaces_only_mode_expires_stale_non_target_latches():
    """P0.3: Entering spaces-only mode must immediately discard non-target latches so they
    cannot survive past the 250ms state-gap reset and bypass approach gates later."""
    cfg = load_config()["classifier"]["heuristic_3d"]
    clf = Heuristic3DPinchClassifier(cfg)
    dt_ms = 1000.0 / 30.0

    # 1. Latch THUMB_MIDDLE_PINCH and hold past approach_window_ms (1500 ms) so ONLY the latch
    #    is keeping the pose alive (the exact stale-latch scenario if spaces-only kept it).
    mid_seq = make_airpoint_ergonomic_sequence(
        PoseName.THUMB_MIDDLE_PINCH, open_frames=6, close_frames=6, hold_frames=50, fps=30.0
    )
    last_ts = 0.0
    for ts_ms, lms in mid_seq:
        p = clf.predict(lms, lms, "Right", ts_ms)
        last_ts = ts_ms
    assert p.pose == PoseName.THUMB_MIDDLE_PINCH
    assert PoseName.THUMB_MIDDLE_PINCH in clf._hands["Right"].latched

    # 2. Switch into spaces-only mode for 330 ms (> 250 ms gap reset)
    upright_open = make_airpoint_ergonomic_sequence(
        PoseName.THUMB_PINKY_BASE, open_frames=10, close_frames=1, hold_frames=1, fps=30.0
    )[:10]
    for _, lms in upright_open:
        last_ts += dt_ms
        clf.predict_only_pose(lms, lms, "Right", last_ts, only_pose=PoseName.THUMB_PINKY_BASE)
        assert PoseName.THUMB_MIDDLE_PINCH not in clf._hands["Right"].latched
        assert PoseName.THUMB_MIDDLE_PINCH not in clf._hands["Right"].latch_info

    # 3. Return to normal mode with a static closed THUMB_MIDDLE_PINCH (no fresh open approach):
    #    must be blocked by approach gate rather than riding a stale latch.
    closed_mid = mid_seq[-1][1]
    p_back = clf.predict(closed_mid, closed_mid, "Right", last_ts + dt_ms)
    assert PoseName.THUMB_MIDDLE_PINCH not in clf._hands["Right"].latched
    assert p_back.pose == PoseName.IDLE
