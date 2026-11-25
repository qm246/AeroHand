# AeroHand

**Low-latency, touchless macOS hand controller** combining:
- **Airpoint SDK's ergonomics & controllers** (Apache-2.0): decoupled index-tip aiming + `thumb_middle` click, 2-stage OneEuro smoothing, 60 Hz lerp cursor, 2D axis-locked scroll with inertia, short-slide Spaces & Window Tiling, 3D Forward-Pointing Gate, and 3D Palm Cylinder Palm Wheel.
- **GestureCtl's native macOS safety & ML foundation**: zero-Electron Quartz `CGEvent` injection, session-counter **Typing Guard** (0 Input Monitoring permission), `release_all()` failsafe, 8 FPS Idle Throttle, Gram-Schmidt 3D `palm_frame`, and pluggable **3D Heuristic → Self-Trained ONNX TCN** classifier.

> **License & Clean-Room Note**: Licensed under **Apache-2.0** (see [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). Contains **zero** proprietary AirMouse weights or extracted assets from `Airpoint.app`.

---

## Gesture Cheat-Sheet (Single Hand — Right Hand Default)

| Action | Gesture | Notes |
| :--- | :--- | :--- |
| **Move Cursor** | Point index finger toward screen (`dirZ < -0.15`) | 2-stage OneEuro + distance comp + 60 Hz lerp |
| **Left Click / Double Click** | Tap **Thumb + Middle** (`thumb_middle_pinch`) | Decoupled from index tip → eliminates cursor shift |
| **Drag** | Hold **Thumb + Index + Middle** (`thumb_index_middle_pinch`) | Never confuses slow clicks with drags |
| **Right Click** | Hold **Thumb + Ring** still (`< 15 px`) & release | Dual-behavior pose |
| **Scroll (2D + Inertia)** | Hold **Thumb + Ring** & move (`>= 15 px`) | X/Y axis lock + natural coasting inertia |
| **Spaces / Mission Control** | Hold **Thumb + Pinky Base** & slide (`>= 20 px`) | Left/Right = Spaces; Up/Down = Mission Control / Exposé |
| **Window Tiling** | Hold **Thumb + Pinky Tip** & slide (`>= 20 px`), release | Left / Right / Maximize / Restore |
| **Quick Menu (Palm Wheel)** | Open palm facing camera (`>= 200 ms`), curl 1 finger & release | Thumb=Pause, Index=Mission Control, Middle=Recenter, Ring=Spotlight, Pinky=Play/Pause |

---

## Quick Start (macOS Apple Silicon)

```bash
cd /Users/macos/Documents/AeroHand
uv sync --extra dev

# 1. Run unit tests
uv run pytest -q

# 2. Dry-run with live camera preview HUD (no OS clicks/moves injected)
uv run aerohand --dry-run --preview

# 3. Live macOS control (requires Camera + Accessibility permissions)
uv run aerohand --preview

# 4. Headless low-CPU mode (no OpenCV preview window)
uv run aerohand
```

Press `q` in preview window (or `Ctrl+C` in terminal) to stop safely (`release_all()` is always called on exit).

---

## Upgrading from 3D Heuristic MVP to Self-Trained ONNX TCN

AeroHand ships with **`Heuristic3DPinchClassifier`** working out-of-the-box using `world_landmarks` + 3D `palm_frame`. When you want higher accuracy under extreme hand tilts:
1. Install ML extras: `uv sync --extra ml --extra train`
2. Record labeled pinch data via `aerohand --collect-dir recordings/pinch_v1` (press `0..6` for class labels, `r` to toggle recording, `s` to save JSON).
3. Train a causal 1D dilated TCN on the strictly rotation/translation-invariant 3D features in `aerohand.ml.features_3d` (no screen-absolute 2D coordinates) and export to `.onnx`.
4. Point `config/user_overrides.yaml` to your `.onnx` + normalizer JSON (`classifier.backend: "onnx_tcn"`) — zero engine code changes required.

---

## Project Layout

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full architecture, data flow, and safety invariants.
