# Đánh giá Mô hình ML & Giao thức Thu thập Dữ liệu AeroHand (v0.4.0)

Tài liệu này quy định quy trình thu thập dữ liệu, huấn luyện Causal 1D-TCN, đánh giá A/B giữa hai tập đặc trưng (`invariant348` và `airmouse885`), kiểm thử hồi quy ngoại tuyến (`replay_eval.py`) và triển khai an toàn qua Shadow Mode.

---

## 1. Mục tiêu Bộ dữ liệu 26.000 khung hình (Landmarks-Only)

AeroHand **không bao giờ lưu điểm ảnh camera** — chỉ lưu tọa độ 21 khớp tay 2D/3D, tay trái/phải, độ tin cậy và siêu dữ liệu điều kiện.

| Lớp cử chỉ (`PoseName`) | Tổng mục tiêu | Tay phải (`Right`) | Tay trái (`Left`) | Ghi chú |
|---|---:|---:|---:|---|
| `idle` | 8.000 | 5.000 | 3.000 | Bao gồm 8 nhóm âm tính khó (≥ 500 mẫu/nhóm) |
| `thumb_index_pinch` | 3.000 | 2.000 | 1.000 | Giữ đọc giọng nói / Zoom 2 tay |
| `thumb_middle_pinch` | 3.000 | 2.000 | 1.000 | Click trái / Double click |
| `thumb_ring_pinch` | 3.000 | 2.000 | 1.000 | Chuột phải (giữ yên) / Cuộn 2D (di chuyển) |
| `thumb_pinky_pinch` | 3.000 | 2.000 | 1.000 | Chia cửa sổ (`AXUIElement`) |
| `thumb_pinky_base` | 3.000 | 2.000 | 1.000 | Chuyển Space / Mission Control |
| `thumb_index_middle_pinch` | 3.000 | 2.000 | 1.000 | Kéo thả 3 ngón (Drag) |
| **Tổng cộng** | **26.000** | **17.000** | **9.000** | Cắt tự động ±300 ms biên mỗi phân đoạn |

### Phân bố điều kiện bắt buộc (`condition_id`)
- **Góc cổ tay (`wrist_angle`)**: `{0, 30, 45, 60}` độ — **mỗi góc ≥ 15% tổng số mẫu**.
- **Khoảng cách (`distance`)**: `{near, mid, far}`.
- **Ánh sáng (`lighting`)**: `{bright, backlit, dim}`.
- **8 nhóm âm tính khó (`hard_negative_group`)** (mỗi nhóm ≥ 500 khung `idle`): `typing`, `resting`, `hair_face`, `cup_phone`, `waving`, `fist_open`, `near_miss` (chụm lỏng 1–2 cm), `transitions`.
- **Khai thác âm tính thực tế (V031-06)**: Bất kỳ kích hoạt nhầm nào trong sử dụng hàng ngày đều có thể nhấn nút **"Không phải tôi muốn"** (`Cmd+Shift+N`) để lưu đoạn ±1,5 giây thành mẫu `idle` trong `data/negatives/`.

### Cổng chất lượng trước khi lưu (Quality Gate)
`PinchDataCollector.save()` tự động **chặn lưu** (`QualityGateError`) nếu phiên vi phạm bất kỳ điều kiện nào:
1. `low_confidence`: Độ tin cậy MediaPipe trung bình < `0.58`.
2. `hand_clipped_edge`: Có khung hình bàn tay chạm mép khung hình (`x` hoặc `y` < `0.03` hoặc > `0.97`).
3. `time_gaps`: Có khoảng trống thời gian > `100 ms` giữa hai khung liên tiếp của cùng một bàn tay.

---

## 2. Giao thức Chia tập & Huấn luyện Không rò rỉ

- **Chia tập theo nhóm `(subject, condition_id)`**: Không bao giờ cắt ngẫu nhiên từng khung hình từ cùng một phiên vào cả train và val.
- **Ngắt chuỗi trượt**: Tự động ngắt cửa sổ 5 khung hình (`seq_len=5`) khi đổi `hand_id` hoặc khi khoảng trống > `100 ms`.
- **Tăng cường dữ liệu trên tọa độ khớp (trước khi trích đặc trưng)**: Xoay quanh trục Z ±15°, tỷ lệ 0.85–1.15, nhiễu Gaussian σ = 0.003, rơi/lặp 1 khung hình, lật gương trái/phải.
- **Hàm mất mát & Chọn epoch**: Focal Loss (`gamma=2.0`) chọn epoch tốt nhất theo `macro_f1 - 0.5 * idle_fp_rate`.
- **Hiệu chuẩn xác suất (Temperature Scaling)**: Tối ưu nhiệt độ `T > 0` trên tập validation và lưu trực tiếp vào `*_normalizer.json` để `ONNXPinchClassifier` áp dụng khi suy luận.

---

## 3. So sánh A/B: `invariant348` vs `airmouse885`

### Lệnh chạy so sánh đa hạt giống (3 seed: `42, 1337, 2026`)
```bash
cd /Users/macos/Documents/AeroHand
uv sync --extra train
uv run python scripts/train_tcn.py \
  --ab-compare \
  --ab-seeds 42,1337,2026 \
  --sessions data/sessions \
  --negatives-dir data/negatives \
  --out-dir models \
  --loss focal \
  --augment
```

Kết quả được xuất ra `models/ab_report.json` chứa:
- Trung bình và độ lệch chuẩn (`mean` / `std`) của `macro_f1`, `val_accuracy`, `idle_fp_rate` cho cả `invariant348` và `airmouse885`.
- Phân tích lỗi từng cặp dễ nhầm lẫn:
  - `thumb_middle_pinch <-> thumb_index_middle_pinch`
  - `thumb_ring_pinch <-> thumb_pinky_pinch`
  - `thumb_pinky_pinch <-> thumb_pinky_base`
  - `thumb_index_pinch <-> idle`
- Trường `decision` ghi rõ tập đặc trưng được chọn.

### Quy tắc ra quyết định (Decision Rule)
- **Mặc định giữ `invariant348`** (bất biến xoay/tịnh tiến/tỷ lệ trong hệ tọa độ lòng bàn tay 3D, kích thước gọn và suy luận nhanh).
- **Chỉ chuyển sang `airmouse885`** nếu đồng thời thỏa mãn:
  1. `macro_f1_mean(airmouse885) - macro_f1_mean(invariant348) > +0.02` (vượt trội trên 2 điểm phần trăm).
  2. `idle_fp_rate_mean(airmouse885) <= idle_fp_rate_mean(invariant348)` (không làm tăng báo động giả ở trạng thái nghỉ).
  3. Không làm tăng tổng số kích hoạt nhầm trên giờ (`total_ft_per_hour`) trên bộ kiểm thử hồi quy ngoại tuyến.

---

## 4. Kiểm thử Hồi quy Ngoại tuyến (`scripts/replay_eval.py`)

Chạy phát lại tuần tự toàn bộ phiên và tệp âm tính khó qua `HandCursorEngine` thực (không cần camera hay API đồ họa macOS):

```bash
uv run python scripts/replay_eval.py \
  --sessions data/sessions \
  --negatives data/negatives \
  --backend onnx_tcn \
  --model models/aerohand_tcn.onnx \
  --normalizer models/aerohand_tcn_normalizer.json \
  --max-ft-per-hour 2.0 \
  --baseline models/baseline_replay.json \
  --out models/latest_replay.json
```

Tiêu chí đạt chuẩn CI/phát hành:
- `total_ft_per_hour <= 2.0` kích hoạt nhầm / giờ trên dữ liệu nghỉ & âm tính khó.
- `macro_f1` không giảm quá `0.01` so với `--baseline`.

---

## 5. Quy trình Đưa mô hình ONNX vào Sử dụng An toàn (Shadow Rollout)

1. Bật `classifier.shadow_mode: true` trong tab **Dữ liệu ML** (hoặc `aerohand settings set classifier.shadow_mode true`). Lúc này hệ thống vẫn điều khiển chuột bằng `Heuristic3DPinchClassifier`, đồng thời chạy song song ONNX và ghi nhận tỷ lệ bất đồng (`shadow_disagreement_rate`) + độ trễ `onnx_p95_ms` qua `aerohand vitals --json`.
2. Theo dõi phiên thực tế ≥ 15 phút: kiểm tra `onnx_p95_ms < 8.0 ms` và `onnx_errors == 0`.
3. Chuyển `classifier.backend: "onnx_tcn"` (`shadow_mode: false`). Lớp bảo vệ `GuardedClassifier` vẫn giám sát liên tục và tự động hạ về `Heuristic3DPinchClassifier` kèm `release_all()` nếu độ trễ `p95 > 8 ms` hoặc có ≥ 3 lỗi trong 10 giây.
