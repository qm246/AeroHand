# Đánh giá Mô hình ML & Giao thức Thu thập Dữ liệu AeroHand (v0.4.0)

Tài liệu này quy định quy trình thu thập dữ liệu, huấn luyện Causal 1D-TCN, đánh giá A/B giữa hai tập đặc trưng (`invariant348` và `airmouse885`), kiểm thử hồi quy ngoại tuyến (`replay_eval.py`) và triển khai an toàn qua Shadow Mode.

---

## 1. Mục tiêu Bộ dữ liệu 21.000 khung hình mới (Landmarks-Only — `V050-02`)

AeroHand **không bao giờ lưu điểm ảnh camera** — chỉ lưu tọa độ 21 khớp tay 2D/3D, tay trái/phải, độ tin cậy và siêu dữ liệu điều kiện. Vì kho âm tính 3D đã quy đổi từ GestureCtl (`data/converted/negatives/`) có sẵn ~31.600 khung `idle`, mục tiêu thu thập mới của AeroHand tập trung vào 6 lớp dương tính và 3.000 khung `idle` bổ sung (tổng **21.000 khung mới**):

| Lớp cử chỉ (`PoseName`) | Tổng mục tiêu mới | Tay phải (`Right`) | Tay trái (`Left`) | Ghi chú |
|---|---:|---:|---:|---|
| `idle` | 3.000 | 2.000 | 1.000 | Bổ sung vào ~31,6k khung 3D GestureCtl (cân bằng ≤ 15%/kịch bản) |
| `thumb_index_pinch` | 3.000 | 2.000 | 1.000 | Giữ đọc giọng nói / Zoom 2 tay |
| `thumb_middle_pinch` | 3.000 | 2.000 | 1.000 | Click trái / Double click |
| `thumb_ring_pinch` | 3.000 | 2.000 | 1.000 | Chuột phải (giữ yên) / Cuộn 2D (di chuyển) |
| `thumb_pinky_pinch` | 3.000 | 2.000 | 1.000 | Chia cửa sổ (`AXUIElement`) |
| `thumb_pinky_base` | 3.000 | 2.000 | 1.000 | Chuyển Space / Mission Control |
| `thumb_index_middle_pinch` | 3.000 | 2.000 | 1.000 | Kéo thả 3 ngón (Drag) |
| **Tổng cộng mới** | **21.000** | **14.000** | **7.000** | Cắt tự động ±300 ms biên mỗi phân đoạn |

### Kỷ luật Held-Out đóng niêm phong & Đa người dùng (`V050-02`)
- **Niêm phong tập held-out riêng của AeroHand** (`scripts/seal_heldout.py`):
  ```bash
  uv run python scripts/seal_heldout.py --src data/sessions/heldout_candidates --out-dir data/heldout_aerohand
  uv run python scripts/replay_eval.py --heldout data/heldout_aerohand --backend onnx_tcn --model models/aerohand_tcn.onnx --normalizer models/aerohand_tcn_normalizer.json
  ```
  `sealed_manifest.json` lưu mã băm SHA-256 từng tệp, bộ đếm `touched` và `evaluated_model_hashes`. `replay_eval.py --heldout` xác minh toàn vẹn SHA-256 và **từ chối chạy lần thứ 2** cho cùng một `model_hash`.
- **Đánh giá ngoài người dùng (`subject-held-out`)**: Khi quay thêm tình nguyện viên (chỉ lưu 21 khớp cục bộ, có thông báo quyền riêng tư trên ML Studio), truyền `--primary-subject <tên>` hoặc `--heldout-subjects <danh_sách>` khi chạy `scripts/train_tcn.py` để tách hoàn toàn người dùng phụ ra khỏi tập huấn luyện.
- **Lấy mẫu cân bằng theo kịch bản (`--max-idle-scenario-share 0.15`)**: Tự động giới hạn mỗi kịch bản âm tính chiếm **≤ 15%** tổng số cửa sổ `idle` huấn luyện (giữ nguyên 100% cửa sổ dương tính).

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

### Lệnh chạy so sánh đa hạt giống trên kho hợp nhất (3 seed: `42, 1337, 2026` — `V050-03`)
```bash
cd /Users/macos/Documents/AeroHand
uv sync --extra train

# 1. Kiểm tra lệch miền (KS-test) giữa kho GestureCtl đã quy đổi và phiên AeroHand mới
uv run python scripts/domain_check.py \
  --a data/converted/negatives \
  --b data/sessions

# 2. Huấn luyện & so sánh A/B trên kho hợp nhất (cân bằng kịch bản <= 15%, tách subject-held-out)
uv run python scripts/train_tcn.py \
  --ab-compare \
  --ab-seeds 42,1337,2026 \
  --sessions data/sessions data/converted/positives \
  --negatives-dir data/negatives data/converted/negatives \
  --ab-eval-negatives data/converted/negatives/gctl_d_dev_*.json \
  --max-idle-scenario-share 0.15 \
  --primary-subject minh \
  --out-dir models \
  --loss focal \
  --augment
```

Kết quả được xuất ra `models/ab_report.json` chứa:
- Trung bình và độ lệch chuẩn (`mean` / `std`) của `macro_f1`, `val_accuracy`, `idle_fp_rate`, `selection_score` (`macro_f1 - 0.5 * idle_fp_rate`), `ft_per_hour` và `ft_upper_bound_per_hour` (95% Poisson UCB) cho cả `invariant348` và `airmouse885`.
- Phân tích lỗi từng cặp dễ nhầm lẫn:
  - `thumb_middle_pinch <-> thumb_index_middle_pinch`
  - `thumb_ring_pinch <-> thumb_pinky_pinch`
  - `thumb_pinky_pinch <-> thumb_pinky_base`
  - `thumb_index_pinch <-> idle`
- Trường `decision` ghi rõ tập đặc trưng được chọn.

### Quy tắc ra quyết định (`V050-03` Decision Rule)
- **Bắt buộc đủ cả 7 lớp chuẩn** trong tập validation (nếu thiếu lớp, `validate_validation_classes` từ chối huấn luyện trừ khi chỉ định `--allow-missing-val-classes` cho smoke test).
- **Mặc định giữ `invariant348`** (bất biến xoay/tịnh tiến/tỷ lệ trong hệ tọa độ lòng bàn tay 3D, kích thước gọn và suy luận nhanh).
- **Chỉ chuyển sang `airmouse885`** nếu đồng thời thỏa mãn:
  1. `macro_f1_mean(airmouse885) - macro_f1_mean(invariant348) > +0.02` (vượt trội trên 2 điểm phần trăm).
  2. `idle_fp_rate_mean(airmouse885) <= idle_fp_rate_mean(invariant348)` (không làm tăng báo động giả ở trạng thái nghỉ).
  3. Không làm tăng `total_ft_per_hour` **và** không làm tăng cận trên 95% Poisson `ft_upper_bound_per_hour` trên bộ kiểm thử hồi quy ngoại tuyến.

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

---

## 6. Cổng Ý định dựa trên Dữ liệu cho Heuristic 3D (`V050-01`)

Phân tích JSONL sự kiện kích hoạt nhầm trên bộ âm tính đời thực `D-dev` (12 kịch bản, 24,62 phút đồng hồ / 7,42 phút `hand_present_idle`, đã sửa đảo nhãn tay gương GestureCtl `Left <-> Right`) cho thấy đa số kích hoạt nhầm xảy ra khi cổ tay dịch chuyển/xoay nhanh hoặc bàn tay xuất hiện trong tư thế nắm cốc/điện thoại.

4 cổng ý định cấu hình độc lập dưới `classifier.heuristic_3d` (chỉ áp dụng khi **vào** cử chỉ mới chưa chốt `latch`, tuyệt đối không cắt ngang lúc đang giữ kéo thả/cuộn/đọc giọng nói):
1. `enable_approach_gate` (`approach_window_ms: 1500.0`, `approach_open_margin: 0.04`): yêu cầu cặp ngón tay từng tách ra trước khi khép vào; chặn hoàn toàn tiếp xúc tĩnh khi tay mới vào khung hình.
2. `enable_motion_gate` (`max_wrist_speed: 0.18`, `max_wrist_ang_speed_deg_s: 95.0`): chặn vào cử chỉ mới khi cổ tay dịch chuyển hoặc xoay 3D nhanh (`max(instant, EMA)`).
3. `enable_posture_gate` (`index_ext_min: 1.25`): yêu cầu ngón trỏ duỗi khi thực hiện các cử chỉ không dùng ngón trỏ (`thumb_middle_pinch`, `thumb_ring_pinch`, `thumb_pinky_pinch`).
4. `enable_grip_guard` (`grip_half_curled_min: 3`, `grip_loose_ratio: 1.05`): chặn tiếp xúc lỏng khi ≥ 3 ngón đang gập nửa (cầm cốc/điện thoại) mà không ảnh hưởng tới chụm chặt chủ ý.

### Kết quả kiểm chứng Trước / Sau trên `D-dev` & mẫu dương tính liên tục (`dev2_batch1`)

| Chỉ số | Trước (`v0.4.1`, tắt cổng) | Sau (`V050-01`, bật 4 cổng) | Thay đổi |
|---|---:|---:|---:|
| Tổng FT trên `D-dev` (12 kịch bản) | 16 | **4** | **-75,0%** |
| Tốc độ FT `hand_present_idle` (95% Poisson UCB) | 129,3 / giờ (UCB 196,4) | **32,3 / giờ (UCB 74,0)** | **-75,0%** |
| Tốc độ FT theo `wall_clock` (24,62 phút) | 39,0 / giờ | **9,7 / giờ** | **-75,0%** |
| `left_click` / `drag_start` / `window_tile` / `palm_wheel` | 5 / 2 / 1 / 0 | **0 / 0 / 0 / 0** | **-100%** |
| `scroll_stroke` / `dictation_start` | 5 / 3 | **2 / 2** | -60% / -33% |
| Tổng FT trên `T-dev` (5 kịch bản chỉ tay) | 29 | **19** | **-34,5%** |
| Recall dương tính `clicks_pinch_cadence_near_145s` | 99,2% khung / 50/50 lượt | **95,0% khung / 49/50 lượt (98,0%)** | Đạt ≥ 95% |
| Recall dương tính `clicks_pinch_cadence_near_250s` | 94,4% khung / 53/53 lượt | **94,4% khung / 53/53 lượt (100,0%)** | Không giảm |

> **Lưu ý thống kê**: Phơi nhiễm `D-dev` chỉ đạt `0,1237 giờ` `hand_present_idle` (cận trên 95% với 4 sự kiện là `74,0 FT/giờ`). Các cổng hình học chỉ giảm tải bước đầu; để chứng minh ≤ 2 FT/giờ (UCB 95%) cần huấn luyện TCN trên bộ dữ liệu đa lớp và đánh giá 1 lần trên tập held-out đóng niêm phong ≥ 1,5 giờ (`V050-02`–`V050-04`).
