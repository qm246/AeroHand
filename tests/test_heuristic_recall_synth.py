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
