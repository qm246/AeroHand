# AeroHand Architecture

## Design Goal
Combine the **best ergonomic & control algorithms** of `andrewslabmd/airpoint-sdk` (Apache-2.0) with the **native macOS safety, low-CPU runtime, and scientific evaluation discipline** built in `GestureCtl`, inside a clean standalone Python/PyObjC codebase.

## Runtime Data Flow

```text
AVFoundation Webcam (~30 fps, ThreadedCamera, zero backlog)
  │
  ▼  (Idle Throttle drops to 8 fps after 3s with no hand)
MediaPipe HandLandmarker (`models/hand_landmarker.task`, GPU -> CPU auto-fallback)
  │  outputs: 21 image landmarks + 21 metric 3D world_landmarks + handedness
  ▼
HandCursorEngine (`src/aerohand/core/engine.py`)
  ├─ 1. LandmarkSmoother (Stage-1 OneEuro: mc=2.0, beta=45.0, gap_reset=100ms)
  ├─ 2. PalmWheelDetector (3D Palm Cylinder R=0.5*palm_len + palm_facing_dot >= 0.5)
  ├─ 3. CursorKinematics (index tip LM8, Stage-2 OneEuro mc=1.0 beta=0.01,
  │      distance compensation, acceleration, 15% jump reject, OS cursor anchor)
  ├─ 4. 3D Forward-Pointing Gate (`(LM8-LM0).z / len < -0.15`, hysteresis +0.04)
  ├─ 5. Pluggable PinchClassifier (`src/aerohand/ml/`) — fail-open to IDLE on error
  │      • MVP: `Heuristic3DPinchClassifier` (3D `palm_frame` normalized distances)
  │      • Upgrade: `ONNXPinchClassifier` (self-trained causal 1D TCN, seq_len=5,
  │                 312 strictly rotation/translation/scale-invariant 3D features)
  ├─ 6. PoseStateMachine (per-pose asymmetric Schmitt + dual time/motion Tap->Hold)
  └─ 7. Hold Controllers:
         • ScrollController (2D X/Y axis lock + breakout + rate-independent inertia)
         • SpacesNavController (thumb_pinky_base + 20px slide)
         • WindowTileController (thumb_pinky_pinch + 20px slide, fire on release)
  │
  ▼
macOS Native Platform (`src/aerohand/platform/`)
  ├─ 60 Hz CursorLerpInterpolator (`lerp_speed=55`) smooths 30 Hz camera to 60 Hz motion
  ├─ Typing Guard (`kCGSessionEventTap` counter — ZERO Input Monitoring permission!)
  ├─ Quartz `CGEventPost` (left/right click, drag, 2D `wheelCount=2` scroll, Spaces)
  ├─ Window Tiling (AXUIElement native resize) + Audio feedback (`NSSound`)
  └─ Failsafe `release_all()` on hand loss, pause, exception, or shutdown
```

## Safety Invariants
1. **Zero stuck buttons/modifiers**: Hand loss > `hand_lost_release_ms`, drag > `max_drag_duration_ms`, pause, or quit always invokes `InputController.release_all()`.
2. **Typing Guard always active**: Physical keyboard input suppresses gestures for `1.5 s`; physical mouse/trackpad input suppresses pointer/click/drag/scroll for `1.0 s`. Posting synthetic events to `kCGSessionEventTap` avoids polluting `HIDSystemState`, eliminating Input Monitoring permission friction.
3. **Fail-open classifier**: Any runtime error in heuristic or ONNX inference falls back to `PoseName.IDLE` rather than crashing or latching a click.
4. **Clean-room ML**: Feature vectors exclude raw screen-absolute `(x, y)` coordinates so single-user TCN models cannot overfit to desk framing. No proprietary AirMouse assets are ever included.
