# Third-Party Notices

AeroHand is licensed under the Apache License, Version 2.0 (see `LICENSE`).

## 1. Airpoint SDK (`andrewslabmd/airpoint-sdk`)
- **License**: Apache License, Version 2.0
- **Source**: https://github.com/andrewslabmd/airpoint-sdk
- **Scope Used**: Selected open-source algorithmic designs and state-machine patterns from `airpoint-sdk` (`src/internal-engine/`) were ported and adapted to Python in `src/aerohand/core/` and `src/aerohand/ml/`, including:
  - Two-stage 1€ landmark/cursor smoothing and relative kinematics (`filters.py`, `kinematics.py`)
  - Unified probability-based asymmetric Schmitt pose state machine (`pose_state_machine.py`)
  - 2D axis-locked scroll controller with inertia (`scroll_controller.py`)
  - Spaces navigation and window tiling slide controllers (`spaces_nav.py`, `window_tile.py`)
  - 3D Palm Cylinder radial heuristic and Palm Wheel detector (`palm_wheel.py`)
  - Forward-pointing 3D depth gate and causal TCN/ONNX feature/inference interface (`engine.py`, `features_3d.py`, `onnx_classifier.py`, `collector.py`)
- **Strict Non-Inclusion of Proprietary Assets**: AeroHand contains **ZERO** proprietary AirMouse weights, normalizers, license keys, or extracted assets from `Airpoint.app`. Any neural gesture classifier used with AeroHand must be independently trained on the user's own recordings.
- **Trademarks**: "Airpoint" and "AirMouse" are trademarks of their respective owners and are referenced solely for attribution and technical comparison under Apache-2.0 Section 6.

## 2. MediaPipe Hand Landmarker
- **License**: Apache License, Version 2.0 (Google LLC)
- **Asset**: `models/hand_landmarker.task`

## 3. GestureCtl (Author's Prior Project)
- Native macOS Quartz `CGEvent` injection, `release_all()` failsafe, session-counter `Typing Guard`, `ThreadedCamera`, `Idle Throttle`, and Gram-Schmidt 3D `palm_frame` geometry are adapted from Minh's `GestureCtl` project.
