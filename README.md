<div align="center">

# ✦ AeroHand

**Ứng dụng điều khiển macOS không chạm độ trễ thấp — xử lý 100% cục bộ trên máy.**

🇻🇳 **Tiếng Việt** · [🇺🇸 English](./docs/README.en.md) · [📐 Kiến trúc hệ thống](./docs/ARCHITECTURE.md)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-teal.svg)](LICENSE)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-blue.svg)](#cài-đặt--chạy-nhanh-macos-apple-silicon)
[![Version: v0.2.0](https://img.shields.io/badge/Version-0.2.0-emerald.svg)](#)

</div>

---

**AeroHand** kết hợp trọn vẹn hai thế mạnh:
- **Công thái học, bộ điều khiển & trải nghiệm từ [Airpoint SDK](https://github.com/andrewslabmd/airpoint-sdk) (Apache-2.0)**: tách ngón trỏ chỉ chuột + chụm **Cái + Giữa** click trái chống lệch con trỏ, lọc mượt 2 tầng One-Euro + nội suy 60 Hz Lerp, cuộn 2D khóa trục có quán tính & Momentum Phase, chia cửa sổ, chuyển Space, Zoom 2 tay, Palm Wheel 2 tay kèm Submenu lồng nhau, đọc giọng nói (Voice Dictation), Local HTTP API/CLI, và Overlay trực quan nổi trên màn hình.
- **Nền tảng an toàn macOS bản địa & khoa học ML từ GestureCtl**: bơm sự kiện Quartz `CGEventPost` + `AXUIElement` (không cần Electron), **Typing Guard 0-Permission** (không đòi quyền Input Monitoring), **Idle Throttle 8 FPS** tiết kiệm pin, **AM-7 Large-Gap Filter Reset**, hệ trục 3D **Gram-Schmidt `palm_frame`**, và kiến trúc phân loại có thể nâng cấp liền mạch từ **Heuristic 3D → ONNX 1D-TCN tự huấn luyện**.

> **Minh bạch giấy phép (Clean-Room)**: Mã nguồn phát hành theo giấy phép **Apache-2.0** (xem [`LICENSE`](LICENSE) và [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). Hoàn toàn **không** chứa trọng số độc quyền `AirMouse` hay tài nguyên trích xuất từ ứng dụng thương mại. Toàn bộ dữ liệu camera và giọng nói xử lý cục bộ trên máy.

---

## Tính năng nổi bật (v0.2.0)

- **Giao diện macOS Dark Glass bản địa (`WKWebView` + Menu Bar)**: 8 tab chuyên biệt (*Tổng quan*, *Cử chỉ*, *Con trỏ & Cuộn*, *Palm Wheel*, *Camera & HUD*, *Giọng nói & API*, *Dữ liệu ML*, *Hướng dẫn tương tác*), **hỗ trợ song ngữ Tiếng Việt (mặc định) ↔ English** đổi tức thì.
- **4 lớp Overlay nổi trên toàn màn hình (click-through)**:
  - Đĩa tròn **Palm Wheel 5 cánh** kèm breadcrumb menu con (theo tay hoặc cố định giữa màn hình).
  - Vòng sóng **Click Pulse** ngay tại con trỏ khi Click / Double-Click / Chuột phải / Drag.
  - Khung kính mờ **Window Tile Ghost Preview** xem trước vùng snap cửa sổ.
  - **Toast gợi ý cử chỉ**, sóng âm **Voice Dictation** và **cảnh báo thông minh** (ánh sáng yếu, tay quá xa/gần, sát mép khung hình).
- **Floating Camera HUD Hover-Fade**: chọn 4 góc màn hình, 3 kích thước (S/M/L), tự mờ khi rảnh và sáng 100% khi rê chuột qua.
- **Chia cửa sổ chuẩn bằng Accessibility API (`AXUIElement`)**: hoạt động mọi ứng dụng macOS, hỗ trợ đa màn hình (Multi-Monitor) và hiệu ứng animation mượt.
- **Voice Dictation Tiếng Việt (`vi-VN`) & Tiếng Anh (`en-US`)**: dùng Apple `SFSpeechRecognizer` cục bộ (hỗ trợ `mlx_whisper`), chèn chữ có dấu UTF-8 an toàn không lỗi bộ gõ Telex/EVKey.
- **Hệ sinh thái tự động hóa**: Local HTTP Control API (`127.0.0.1:17420`), `aerohand` CLI, URL Scheme `aerohand://` (cho Apple Shortcuts / Raycast), **Hồ sơ theo ứng dụng (Per-App Profiles)**, phím tắt toàn cục (`Cmd+Shift+A/R/H`), và tự khởi động cùng macOS (**Launch at Login**).
- **Ổn định 24/7 trên macOS**: chạy MediaPipe **CPU XNNPACK NEON** mặc định (`0` rò rỉ bộ nhớ, ~30 FPS, khắc phục triệt để lỗi rò rỉ `CVPixelBuffer` của MediaPipe Metal trên macOS) + `autorelease_pool`.

---

## Bảng 10 Cử Chỉ Chuẩn Airpoint (Mặc định)

*(Bạn có thể chuyển 1 chạm sang **Preset Cổ điển: Cái + Trỏ bấm Click** hoặc tự remap từng cử chỉ trong tab **Cử chỉ**.)*

| # | Thao tác | Cử chỉ (Tay phải mặc định) | Ghi chú |
| :---: | :--- | :--- | :--- |
| 1 | **Di chuyển con trỏ** | Giơ **ngón trỏ** hướng về màn hình (`dirZ < -0.15`) | Lọc One-Euro 2 tầng + bù khoảng cách + 60 Hz Lerp |
| 2 | **Click trái / Double Click** | Chạm thả **Ngón cái + Ngón giữa** (`thumb_middle_pinch`) | Tách khỏi ngón trỏ → không làm lệch điểm ngắm |
| 3 | **Kéo thả (Drag)** | Chụm giữ **3 ngón: Cái + Trỏ + Giữa** (`thumb_index_middle_pinch`) | Không bao giờ nhầm giữa bấm chậm và kéo thả |
| 4 | **Chuột phải** | Chụm **Cái + Áp út** giữ yên (`< 15 px`) rồi thả | Hành vi kép thông minh với Cuộn |
| 5 | **Cuộn 2D quán tính** | Chụm **Cái + Áp út** và di lên/xuống/trái/phải (`>= 15 px`) | Khóa trục X/Y + Momentum Phase như Trackpad Apple |
| 6 | **Đọc giọng nói (Dictation)** | Chụm giữ **Cái + Trỏ** (`thumb_index_pinch`) để nói, thả để chèn chữ | Hỗ trợ Tiếng Việt có dấu (`vi-VN`) & Tiếng Anh (`en-US`) |
| 7 | **Chia cửa sổ (Window Tile)** | Chụm **Cái + Út** và vuốt Trái / Phải / Lên / Xuống, thả ra | Resize trực tiếp bằng `AXUIElement` + khung xem trước |
| 8 | **Chuyển Space / Mission Control** | Chạm **Cái + Gốc ngón út** và vuốt ngang / lên / xuống | Tương đương vuốt 4 ngón Trackpad |
| 9 | **Menu nhanh Palm Wheel (2 tay)** | Xòe lòng bàn tay hướng camera giữ `0.35s`, gập 1 ngón chọn | **Tay phải**: Tracking, Mission Control, Spotlight, Giọng nói, Nhạc→Âm lượng · **Tay trái**: Siri, Cửa sổ, Chỉnh sửa, Ứng dụng, Emoji |
| 10 | **Phóng to / Thu nhỏ (Zoom)** | Chụm cả **2 tay** kéo giãn ra / thu lại | Phóng to (`Cmd +`) / Thu nhỏ (`Cmd -`) |

---

## Cài đặt & Chạy nhanh (macOS Apple Silicon)

### 1. Cài đặt môi trường (`uv`)
```bash
git clone https://github.com/qm246/AeroHand.git
cd AeroHand
uv sync --extra dev
```

### 2. Chạy ứng dụng Desktop (`AeroHand.app`)
```bash
# Đóng gói ứng dụng native AeroHand.app ra Desktop (đã ký ad-hoc & live-link mã nguồn)
uv run python scripts/build_macos_app.py

# Mở ứng dụng
open ~/Desktop/AeroHand.app
```
> **Cấp quyền lần đầu**: Mở tab **Tổng quan** trong AeroHand để kiểm tra và bật nhanh quyền **Camera**, **Trợ năng (Accessibility)**, và (nếu dùng giọng nói) **Microphone** + **Nhận dạng giọng nói**. Khi chưa cấp Trợ năng, app tự khởi động ở chế độ **Chạy thử (Dry-Run)** an toàn.

### 3. Điều khiển bằng CLI & Apple Shortcuts
Khi `AeroHand.app` đang chạy, Local Control API lắng nghe tại `http://127.0.0.1:17420`:
```bash
uv run aerohand status --json                 # Xem trạng thái, FPS, pose hiện tại
uv run aerohand tracking off                  # Tạm dừng theo dõi
uv run aerohand tracking on                   # Tiếp tục theo dõi
uv run aerohand settings set cursor.sensitivity 2.4
uv run aerohand vitals                        # Xem CPU, RAM, độ trễ từng tầng
uv run aerohand see                           # Chụp ảnh màn hình
```
Liên kết nhanh cho **Apple Shortcuts / Raycast**:
`aerohand://open` · `aerohand://tracking/on` · `aerohand://tracking/off` · `aerohand://recenter` · `aerohand://dictation`

### 4. Chạy chế độ dòng lệnh & kiểm thử
```bash
uv run pytest -q                              # Chạy toàn bộ 14 unit/integration test
uv run ruff check src tests scripts           # Kiểm tra lint
uv run aerohand run --dry-run --preview       # Chạy thử bằng cửa sổ OpenCV
```

---

## Nâng cấp Model ML: Từ Heuristic 3D lên ONNX 1D-TCN tự huấn luyện

AeroHand chạy sẵn ngay bằng **`Heuristic3DPinchClassifier`** (hình học 3D trong `palm_frame` bất biến góc xoay). Để nâng độ chính xác pinch khi nghiêng tay gắt lên ngang các model thương mại:

1. Mở tab **Dữ liệu ML** ngay trong `AeroHand.app`, quay ~300–500 mẫu cho từng cử chỉ (`0..6`) và bấm **Lưu JSON** (lưu vào `data/sessions/`).
2. Cài nhóm thư viện huấn luyện và chạy script train Causal 1D-TCN (hỗ trợ so sánh A/B giữa **`invariant348`** và **`airmouse885`**):
```bash
uv sync --extra train
uv run python scripts/train_tcn.py --sessions data/sessions --out-dir models --feature-set invariant348
```
3. Mở tab **Dữ liệu ML** trong app → đổi **Bộ phân loại Pinch** sang **`ONNX 1D-TCN`** (không cần sửa code).

---

## Cấu trúc dự án

Xem chi tiết kiến trúc dữ liệu, luồng xử lý và các bất biến an toàn tại **[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)**.
