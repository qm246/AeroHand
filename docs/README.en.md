<div align="center">

# ✦ AeroHand

**Low-latency, touchless macOS hand controller — 100% on-device.**

[🇻🇳 Tiếng Việt](../README.md) · 🇺🇸 **English** · [📐 Architecture](./ARCHITECTURE.md)

</div>

---

**AeroHand** combines:
- **Airpoint SDK's ergonomics & UX** (Apache-2.0): decoupled index-tip aiming + `thumb_middle` click, 2-stage One-Euro smoothing, 60 Hz lerp cursor, 2D axis-locked scroll with inertia + macOS Momentum Phase, `AXUIElement` Window Tiling, Spaces navigation, Two-Hand Pinch-to-Zoom, 2-Hand Nested Submenu Palm Wheel, Voice Dictation, Local HTTP Control API/CLI, and transparent Screen Overlays.
- **GestureCtl's native macOS safety & ML discipline**: Quartz `CGEventPost` injection, zero-Input-Monitoring **Typing Guard**, `release_all()` failsafe, 8 FPS Idle Throttle, AM-7 filter gap reset, Gram-Schmidt 3D `palm_frame`, zero-leak MediaPipe CPU XNNPACK default on macOS, and pluggable **Heuristic 3D → Self-Trained ONNX 1D-TCN** (`invariant348` / `airmouse885`).

> **License & Clean-Room Note**: Apache-2.0 ([`LICENSE`](../LICENSE), [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)). Contains zero proprietary AirMouse model weights or extracted commercial assets.

---

## Default Airpoint-Standard Gestures

*(Switchable in 1 click to **Classic / GestureCtl** preset or fully remappable in the **Gestures** tab.)*

| Action | Gesture (Right Hand Default) | Notes |
| :--- | :--- | :--- |
| **Move Cursor** | Extend **1 finger (Index)** toward screen (`dirZ < -0.15`), curl others | 2-stage One-Euro + distance comp + 60 Hz lerp |
| **Left Click / Double Click** | **1 ext (Index)** → tap **Thumb on curled Middle shaft** (`thumb_middle_pinch`) | Decoupled from index tip → zero aim drift |
| **Click & Drag** | Curl Ring+Pinky → hold **Thumb + Index + Middle** (`thumb_index_middle_pinch`) | Distinct from tap click |
| **Right Click** | **2 ext (Index+Middle)** → press **Thumb on curled Ring shaft** still (`< 38 px`) & release | Dual-behavior with Scroll |
| **2D Inertial Scroll** | **2 ext (Index+Middle)** → press **Thumb on curled Ring shaft** & slide (`>= 38 px`) | X/Y axis lock + native macOS Momentum Phase |
| **Voice Dictation** | Index extended/slightly curved (others curled) → hold **Thumb on Index side** (`thumb_index_pinch`) | On-device Vietnamese (`vi-VN`) & English (`en-US`) |
| **Window Tiling** | **3 ext (Index+Middle+Ring)** → press **Thumb on curled Pinky**, slide & release | Direct `AXUIElement` resize + glass ghost preview |
| **Spaces / Mission Control** | **4 slightly extended/curved** → fold **Thumb into palm** (`thumb_pinky_base`) & slide | 4-finger trackpad swipe equivalent |
| **2-Hand Palm Wheel** | Hold open palm (`350 ms`), curl 1 finger | 10 root slots (`L1–L5`, `R1–R5`) + nested Submenus |
| **Two-Hand Pinch Zoom** | Pinch both hands & spread / pull | Zoom In / Out (`Cmd +` / `Cmd -`) |

---

## Quick Start (macOS Apple Silicon)

```bash
git clone https://github.com/qm246/AeroHand.git
cd AeroHand
uv sync --extra dev
uv run pytest -q
uv run python scripts/build_macos_app.py
open ~/Desktop/AeroHand.app
```

CLI automation (connects to running `AeroHand.app` at `127.0.0.1:17420`):
```bash
uv run aerohand status --json
uv run aerohand tracking on | off
uv run aerohand settings set cursor.sensitivity 2.4
uv run aerohand vitals --json
uv run aerohand see
```

Train your own ONNX 1D-TCN from recorded sessions:
```bash
uv sync --extra train
uv run python scripts/train_tcn.py --sessions data/sessions --out-dir models --feature-set invariant348
```
