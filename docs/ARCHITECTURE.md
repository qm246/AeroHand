# Kiến Trúc Hệ Thống AeroHand (v0.5.1)

## 1. Mục tiêu thiết kế
Kết hợp **thuật toán công thái học & trải nghiệm UX hoàn chỉnh** của `andrewslabmd/airpoint-sdk` (Apache-2.0) với **nền tảng an toàn macOS bản địa, tiêu thụ tài nguyên thấp và kỷ luật đánh giá ML** từ `GestureCtl` trong một ứng dụng macOS gọn nhẹ chạy hoàn toàn bằng Python + PyObjC + `WKWebView` bản địa (không cần Electron/Node).

---

## 2. Luồng xử lý dữ liệu thời gian thực

```text
AVFoundation Multi-Camera (`platform/camera.py`, zero backlog + đo độ sáng khung hình)
  │
  ▼  (Idle Throttle hạ xuống 8 FPS sau 3 giây không thấy tay)
MediaPipe HandLandmarker (`models/hand_landmarker.task`, num_hands=2)
  │  • Mặc định: CPU XNNPACK NEON (0 rò rỉ bộ nhớ, ~14 ms/frame trên Apple M1 Pro)
  │  • Tránh lỗi rò rỉ CVPixelBuffer/IOSurface (~1.25 MB/frame) của MediaPipe Metal trên macOS
  ▼
HandCursorEngine (`src/aerohand/core/engine.py`)
  ├─ 1. Stage-1 LandmarkSmoother (One-Euro mc=2.0, beta=45.0 + AM-7 gap_reset=100ms)
  ├─ 2. PalmWheelDetector 2 tay (`Left` L1..L5 & `Right` R1..R5, Submenu nhiều tầng,
  │      3D Palm Cylinder R=0.5*palm_len, hold 350ms + khóa chống rung + nắm tay thoát nhanh)
  ├─ 3. CursorKinematics (đầu ngón trỏ LM8, Stage-2 One-Euro mc=1.0 beta=0.01,
  │      bù khoảng cách, gia tốc, loại bước nhảy >15%, neo con trỏ OS)
  ├─ 4. 3D Forward-Pointing Gate (`(LM8 - LM0).z / len < -0.15`, hysteresis +0.04)
  ├─ 5. Pluggable PinchClassifier (`src/aerohand/ml/`) — fail-open về IDLE khi có lỗi
  │      • Mặc định: `Heuristic3DPinchClassifier` (khoảng cách 3D trong `palm_frame`)
  │      • Nâng cấp: `ONNXPinchClassifier` (Causal 1D-TCN, seq_len=5, tự nhận diện
  │                  `invariant348` 348 chiều bất biến 3D hoặc `airmouse885` 885 chiều)
  ├─ 6. PoseStateMachine (ngưỡng Schmitt bất đối xứng từng pose + chuyển Tap → Hold)
  ├─ 7. Bộ điều khiển hành động:
  │      • ScrollController (khóa trục X/Y + quán tính độc lập tốc độ khung hình)
  │      • SpacesNavController (`thumb_pinky_base` + vuốt 20px)
  │      • WindowTileController (`thumb_pinky_pinch` + vuốt 20px, xem trước trước khi thả)
  │      • ZoomController (chụm 2 tay kéo giãn / thu lại)
  └─ 8. PipelineTiming Profiler & Smart Tracking Warnings (thiếu sáng, quá xa/gần, mép khung)
  │
  ▼
Tầng macOS Bản Địa (`src/aerohand/platform/`) & Giao Diện (`src/aerohand/ui/`)
  ├─ 60 Hz CursorLerpInterpolator (`lerp_speed=55`) nội suy mượt từ webcam 30 Hz
  ├─ Typing Guard (`kCGSessionEventTap` + `HIDSystemState` — 0 quyền Input Monitoring!)
  ├─ Quartz `CGEventPost` (Click, Drag, 2D Scroll có ScrollPhase + MomentumPhase, Media/Volume)
  ├─ Accessibility `AXUIElement` Window Tiling (hỗ trợ đa màn hình + animation mượt)
  ├─ VoiceDictationController (`SFSpeechRecognizer` local `vi-VN`/`en-US` + chèn UTF-8 an toàn)
  ├─ Per-App Profiles (theo dõi `frontmostApplication` bundle ID) + Local HTTP API (`127.0.0.1:17420`)
  ├─ Control Center 8 tab (`NSWindow` + `WKWebView`, song ngữ Việt/Anh `i18n`)
  ├─ Screen Overlay trong suốt click-through (`overlay.html`: Palm Wheel 5 cánh, Click Pulse, Tile Preview, Toasts)
  └─ Hover-Fade Floating Camera HUD (`HoverCameraView` + `objc.autorelease_pool()`)
```

---

## 3. Các bất biến an toàn (Safety Invariants)
1. **Không bao giờ kẹt nút chuột / phím tắt (Level-Triggered Failsafe)**: Watchdog `_enforce_input_safety` chạy mỗi vòng lặp (kể cả khi `cf.success == False` hoặc chỉ thấy tay phụ): mất tay chính quá `hand_lost_release_ms` (`350 ms`), treo camera quá `camera_stall_release_ms` (`500 ms`), kéo thả quá `20 s`, tạm dừng, lỗi classifier hoặc thoát app đều gọi `MacOSInputController.release_all()` + `end_scroll_phase()`.
2. **Bảo mật Local Control API**: Chỉ bind `127.0.0.1`, kiểm tra chặt `Host` (chống DNS rebinding), chặn mọi request mang header `Origin` (chống CSRF trình duyệt), yêu cầu `X-AeroHand-Token` (tạo ngẫu nhiên 32 byte, lưu quyền `0600` trong `~/Library/Application Support/AeroHand/api_token`), giới hạn body 64 KiB (413) và whitelist khóa cấu hình scalar cho `PATCH /api/settings`.
3. **Typing Guard luôn bảo vệ không cần Input Monitoring**: Gõ phím vật lý chặn cử chỉ tay trong `1.5 s`; chạm chuột/trackpad chặn trong `1.0 s`. Mọi sự kiện ảo đều bơm vào `kCGSessionEventTap` để không làm nhiễm bộ đếm `HIDSystemState`.
4. **Fail-open & Guarded ML**: Bất kỳ lỗi nào trong heuristic hoặc ONNX Runtime đều rơi an toàn về `PoseName.IDLE` và giải phóng input; `GuardedClassifier` tự động fallback về `Heuristic3DPinchClassifier` khi p95 > `8 ms` hoặc ≥ 3 lỗi trong `10 s`, đồng thời xóa ring buffer 5 frame khi mất tay hoặc đổi model.
5. **Ổn định bộ nhớ, WebView tự phục hồi & Không Race đa luồng**: MediaPipe chạy `CPU XNNPACK` mặc định (0 rò rỉ `CVPixelBuffer`) kết hợp `objc.autorelease_pool()` ở mọi tick UI/overlay; Overlay `WKWebView` dùng `screen_frames_signature()` tránh rebuild khi đổi Space/Dock và gắn `WKNavigationDelegate` (`webViewWebContentProcessDidTerminate_`) tự tải lại nếu macOS dừng tiến trình WebContent; cập nhật cấu hình từ UI/API được xếp hàng áp dụng trên luồng xử lý và kẹp biên an toàn qua `validate_and_clamp_config()`.
6. **Phân xử Zoom vs Dictation & Chống kích hoạt nhầm**: Dictation 1 tay yêu cầu giữ tối thiểu `dictation_arm_ms` (`350 ms`) và bị hủy ngay (không nhận diện, không dán) nếu tay thứ 2 chụm Zoom trong `zoom_arbitration_ms`; cổng VAD loại đoạn quá ngắn (`< 0.6 s`) hoặc im lặng (`< 250 ms` có tiếng) kèm blocklist chặn chữ ảo; Dwell Lock chỉ khóa tối đa 1 lần mỗi lượt chụm kèm cooldown `400 ms`; Lookback có chốt tốc độ (`<= 250 px/s`) và kẹp độ lệch (`<= 24 px`) cho Click, Drag Start và Chuột phải.
7. **Clean-Room & Quyền riêng tư 100% cục bộ**: Không có dữ liệu ảnh camera, giọng nói hay tệp trọng số độc quyền nào rời khỏi máy hoặc bị đóng gói trong repo. Xem thêm quy trình thu thập dữ liệu 26k, so sánh A/B và đánh giá ngoại tuyến tại [`docs/ML_EVALUATION.md`](./ML_EVALUATION.md).
