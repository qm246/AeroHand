# Đánh giá Mô hình ML & Giao thức Thu thập Dữ liệu AeroHand (v0.5.3)

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

## 5. Quy trình Đưa mô hình ONNX vào Sử dụng An toàn (4 Giai đoạn — `V050-04`)

Cấu hình `classifier.mode` hỗ trợ 4 mức an toàn (`heuristic` → `shadow` → `veto` → `onnx`); mọi chuyển chế độ đều xóa sạch bộ đệm 5 khung hình của từng tay:

1. **`heuristic`** (mặc định): dùng `Heuristic3DPinchClassifier` kèm 4 cổng ý định (`V050-01`).
2. **`shadow`**: chạy song song ONNX để đo tỷ lệ bất đồng từng khung (`shadow_disagreement_rate`), tỷ lệ sự kiện bị phủ quyết mô phỏng theo đúng luật xác nhận `K` khung (`shadow_would_veto_events` / `shadow_events`, `shadow_would_veto_rate`) và độ trễ `onnx_p95_ms` qua `aerohand vitals --json` mà chưa tác động chuột.
3. **`veto` (Veto-Only)**: `Heuristic3DPinchClassifier` đề xuất cử chỉ; ONNX **chỉ được phép phủ quyết (trả về `idle`)** nếu `P_onnx(pose) < veto_min_prob` (`0.35`) hoặc `P_onnx(idle) > veto_idle_prob` (`0.60`). Trong `veto_scope="entry"` (`veto_entry_window_ms=300`), sự kiện chỉ chuyển `accepted` khi có `K = veto_confirm_frames` (mặc định `3`, thêm `(K-1)` khung ≈ `+66 ms` ở 30 fps) khung `ready` liên tiếp được duyệt; một khung từ chối đặt lại `approve_streak = 0`; sau khi `accepted` không bao giờ ngắt thao tác đang giữ; quá cửa sổ mà chưa `accepted` giữ `idle` (đếm 1 `veto_count`) cho tới khi đủ `K` khung duyệt liên tiếp hoặc tay nhả. ONNX **không bao giờ tự tạo kích hoạt mới** khi Heuristic dự đoán `idle`. Bộ đếm `veto_count` và chỉ số bỏ lỡ gần đúng `veto_retried_within_1s` được theo dõi trực tiếp qua `aerohand vitals --json` và tab **Dữ liệu ML**.
4. **`onnx` (Toàn phần — chỉ bật sau khi vượt qua `scripts/check_onnx_promotion.py`)**:
   ```bash
   uv run python scripts/check_onnx_promotion.py \
     --metrics models/metrics.json \
     --sealed-eval runs/sealed_heldout_eval.json \
     --model models/aerohand_tcn.onnx \
     --normalizer models/aerohand_tcn_normalizer.json
   ```
   Bắt buộc đạt đồng thời **4 tiêu chí**:
   - Recall ngoài người dùng (`subject_heldout_metrics`) ≥ `0.97` trên **tất cả 7 lớp chuẩn**.
   - Cận trên 95% Poisson trên tập âm tính held-out đóng niêm phong ≤ `2.0 FT/giờ` trên phơi nhiễm `hand_present_idle_hours` ≥ `1.5 giờ`.
   - Độ trễ p95 CPU M1 `< 3.0 ms`.
   - Cổng tự động hạ về `Heuristic3DPinchClassifier` (`max_onnx_p95_ms <= 8.0` và lỗi ≥ 3 lần / 10 giây) đang kích hoạt kèm `release_all()`.

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

---

## 7. Gia cố An toàn, Provenance & Kiểm chứng `v0.5.1` (`V051-01` → `V051-12`)

Bản vá `v0.5.1` khóa toàn bộ các lỗ hổng rò rỉ dữ liệu, chuẩn hóa độc lập FPS/khoảng cách và kiểm chứng recall 5 lớp tổng hợp trước khi thu thập bộ dữ liệu 21.000 khung:

- **Chống rò rỉ & Khóa Provenance (`V051-01`, `V051-02`, `V051-10`, `V051-11`)**:
  - Thống nhất `collector.subject: "minh"` và chặn bỏ sót dữ liệu ngầm khi `--primary-subject` không khớp `session_meta.subject`.
  - `seal_heldout.py` mặc định **di chuyển** (`move=True`, `--copy` tùy chọn), cấm `--force` sau khi đã chạm (`touched > 0`), giới hạn `max_touches=3`, và kiểm tra `model_hash` đúng 24 ký tự hex.
  - `train_tcn.py`, `replay_eval.py` và `domain_check.py` lọc loại mọi tệp có mã băm SHA-256 trùng với `sealed_manifest.json` (kể cả khi đã sao chép/đổi tên) và cấm `--ab-eval-negatives` trùng với tập âm tính huấn luyện.
  - `train_tcn.py` ghi đầy đủ `model_sha256`, `normalizer_sha256`, `model_hash`, `git_commit`, `seed`, `data_files_sha256_list`, `torch_version`; tách `dev_test` độc lập khi có ≥ 4 nhóm; chia phân tầng theo lớp khi chỉ có 1 nhóm; chuẩn hóa nhiễu tăng cường theo `palm_width`; kiểm parity ONNX trên tới 256 mẫu; chỉ ghi đè `models/metrics.json` khi truyền `--set-latest`.
  - `check_onnx_promotion.py` hỗ trợ `--stage veto` (chưa yêu cầu tình nguyện viên `subject-held-out`) và `--stage onnx` (đầy đủ 5 cổng), bắt buộc khớp `model_hash` / `model_sha256` xuyên suốt mọi tệp, kèm bảng Garwood 95% (`UCB <= 2.0/h`: `k=0 -> 1.50h`, `k=1 -> 2.37h`, `k=2 -> 3.15h`).
- **Ổn định Runtime ONNX & Veto-Only (`V051-03` → `V051-06`)**:
  - Warm-up 20 lần nạp ONNX, tách `feature_ms` vs `infer_ms` (`session.run`), yêu cầu `>= 30` mẫu thật trước khi kích hoạt Latency Guard, bổ sung nút **"Kích hoạt lại ONNX"** (`rearm()`).
  - Bỏ qua sạch các khung `ready=False` lúc nạp đệm 5 khung; phạm vi phủ quyết mặc định `veto_scope: "entry"` (`veto_entry_window_ms: 300 ms`) không cắt ngang thao tác đang giữ; đếm `veto_count` theo sự kiện kèm `veto_frames`; đổi ngưỡng live không hủy trạng thái/bộ đếm và nạp ONNX ở luồng nền có khóa hoán đổi nguyên tử `_classifier_lock`.
- **Cổng Ý định bất biến FPS/khoảng cách & Giải mã `gate_debug` (`V051-07`, `V051-08`)**:
  - Chuyển vận tốc cổ tay sang chuẩn hóa theo bề rộng lòng bàn tay (`max_wrist_speed_pw: 1.50`), EMA theo thời gian (`motion_ema_tau_ms: 32.0`) và suy giảm cửa sổ tiếp cận theo thời gian (`approach_decay_tau_ms: 400.0`).
  - `last_debug` xuất thẳng `gate_debug` vào `--ft-jsonl`: xác nhận nguyên nhân 4 sự kiện còn lại trên `D-dev` (2 `scroll_stroke`, 2 `dictation_start`) đều đã **chốt `latched`** ở khung chậm trước đó (còn `wrist_speed = 0.2549` ở `replay_eval` là vận tốc thô chưa lọc One-Euro so với `eff_speed_img = 0.0903` sau Stage-1 smoother mà classifier nhận).
- **Kiểm chứng Recall 5 lớp tổng hợp & Sửa Fist Guard (`V051-09`)**:
  - Trước sửa: 3/9 bài test tổng hợp trong `tests/test_heuristic_recall_synth.py` **thất bại** do fist guard cũ (`curled_count >= 3 and min(d_idx, d_mid) > pinch_on_dist`) chặn `thumb_pinky_pinch`, `thumb_pinky_base` và xóa nhầm `st.latched`, đồng thời khi `s_pinky == s_pbase == 1.0` hai điểm bị hòa nhau kéo về `idle`.
  - Sau sửa (bỏ qua fist guard khi đang `latched` hoặc có bất kỳ pose chuẩn nào chụm chặt, loại trừ ngón đang chụm khỏi `curled_count`, phân giải hòa `s_pinky`/`s_pbase` theo khoảng cách 3D): **9/9 bài test tổng hợp đạt**, `D-dev` giữ nguyên **4 FT (`32,3/h`, UCB `74,0/h`)**, `T-dev` giảm **19 → 17 FT (`100,6/h`, UCB `150,9/h`)**.

### Bản vá hậu kiểm `v0.5.2` (`V052-01` → `V052-05`)

Hoàn tất 5 hạng mục theo **Plan v5** trước khi bắt đầu thu bằng chứng `shadow` và tối ưu cổng `v0.6.0`:

- **Xác nhận đa khung trong Veto & đồng bộ luật Shadow (`V052-01`)**:
  - Bổ sung `classifier.veto_confirm_frames: 3` (mặc định `3`, thêm `(K-1)` khung ≈ `+66 ms` ở 30 fps; `1` giữ tương thích ngược). Trong `veto_scope="entry"`, sự kiện chỉ chuyển `accepted` khi có `K` khung `ready` liên tiếp được ONNX duyệt; khung từ chối đặt lại `approve_streak = 0`; khung `ready=False` abstain mà không tăng/reset chuỗi; quá `veto_entry_window_ms` chưa `accepted` giữ `idle` (đếm 1 `veto_count`) cho tới khi đủ `K` khung duyệt liên tiếp (nhận muộn) hoặc tay nhả.
  - Chế độ `shadow` chạy song song đúng luật xác nhận `K` khung và xuất `shadow_events`, `shadow_would_veto_events`, `shadow_would_veto_frames`, `shadow_would_veto_rate` trong telemetry.
- **Khóa Provenance xuyên suốt Runtime Vitals (`V052-02`)**:
  - Hợp nhất công thức `compute_onnx_model_hash` và `compute_heuristic_source_sha256` tại `aerohand/ml/provenance.py`. `ONNXPinchClassifier._load()` tính `model_sha256` và `model_hash` 24-hex khớp `train_tcn.py` / `replay_eval.py`; `GuardedClassifier.telemetry()` và `aerohand vitals --json` xuất `model_hash`, `model_sha256`, `heuristic_source_sha256`.
  - `check_onnx_promotion.py` bắt buộc **FAIL** `provenance_hash_match` nếu `--vitals` thiếu hash hoặc lệch với `metrics.json` / báo cáo held-out.
- **Nhật ký thời điểm vào Latch (`V052-03`)**:
  - `Heuristic3DPinchClassifier` lưu `latch_info[pose]` (`since_ms`, `dist`, `raw_sc`, `eff_speed_pw`, `eff_ang_deg_s`, `open_dist`, `idx_ext`, `half_curled_count`) đúng khung chốt latch, xóa sạch khi nhả latch, và xuất `latch_age_ms` + `latch_entry` trong `gate_debug` / `--ft-jsonl`.
  - Xác minh trên `D-dev` (4 FT, `32,33/h`, UCB `73,99/h`) và `T-dev` (17 FT, `100,60/h`, UCB `150,89/h`): **100%** pose đang `latched` có đầy đủ `latch_entry` (0 thiếu); ví dụ trên `D-dev` cả 4 FT đều chốt từ trước `65–1034 ms` (`eff_ang_deg_s` từ `50,8` đến `91,3°/s`).
- **Chặn nắm đấm có ngón cái chạm đầu ngón trỏ (`V052-04`)**:
  - Bổ sung `classifier.heuristic_3d.fist_guard_requires_index_curled: true`: chặn nắm đấm kín cả 4 ngón cuộn vào lòng bàn tay (`idx_dist_to_palm < 0.72` và `all_curled_count == 4`) không cho arm dictation, trong khi giữ nguyên OK-sign `thumb_index_pinch`, không tăng FT trên `D-dev`/`T-dev` và không đổi kết quả trên bộ cadence positives cục bộ.
- **Vệ sinh hệ thống (`V052-05`)**:
  - `recording_io.exclude_sealed_content` tìm `sealed_manifest.json` trong `data_root`, `--sealed-root` (hỗ trợ ở `train_tcn.py`, `replay_eval.py`, `domain_check.py`) và các thư mục cha của đầu vào, đồng thời cảnh báo rõ khi tìm thấy `0` manifest.
  - `replay_eval.compute_eval_model_hash("heuristic_3d", ...)` chỉ băm `classifier.heuristic_3d` + `smoothing` + mã nguồn `heuristic_3d.py` & `palm_frame.py` (không đổi hash khi chỉnh `veto_*`).
  - `runtime._swap_classifier_async` dùng khóa `_classifier_swap_lock` + `engine._classifier_lock` (`RLock`) để export counter sát lúc hoán đổi và loại bỏ thế hệ nạp cũ bị trễ.

### Bản vá ổn định `v0.5.3` (`V053-01` → `V053-06` — Plan v6)

Mọi lỗi `B1`–`B5` đều được tái hiện bằng test đỏ trên `56fee6a` trước khi sửa:

- **Bảo vệ latch giữ lâu dưới fist guard (`V053-01` / B1)**: thêm `classifier.heuristic_3d.fist_guard_protect_latch_ms: 500.0`. Khi `is_full_curled_fist` (cả 4 ngón `< 0.72`), chỉ xóa latch trẻ (`< 500 ms`), giữ nguyên latch giữ lâu và chỉ chấm điểm các pose được bảo vệ (không cho chốt latch mới). Đã kiểm chứng trên kịch bản tổng hợp drag/dictation giữ ≥ 600 ms và nắm tay chậm (`close_frames=15`); *lưu ý: drag mới kiểm bằng dữ liệu tổng hợp, chưa kiểm trên bản ghi drag thật*.
- **Đồng nhất bộ đếm nuốt sự kiện & khung chờ giữa `veto` và `shadow` (`V053-02` / B2)**: bổ sung `veto_swallowed_events`, `veto_pending_frames`, `shadow_would_swallow_events`, `shadow_would_pending_frames` (kèm `export_counters`/`import_counters` và telemetry heuristic). Chốt sự kiện chưa `accepted` đã qua khung `ready` khi về `idle`, đổi pose hoặc `reset(hand)` (không chốt khi `reset(None)`), đồng thời ghi `_last_veto_by_hand` để `veto_retried_within_1s` bắt được lần thử lại.
- **Abstain nhất quán lúc warm-up (`V053-03` / B3)**: khi gặp khung `ready=False` lúc chưa thấy khung `ready` nào trong sự kiện → cho pose heuristic đi qua và đánh dấu `accepted` (scope `entry`) để không tách thành 2 nhịp/double-click khi buffer đầy; nếu đã thấy `ready` nhưng chưa `accepted` → tiếp tục trả `idle` và giữ nguyên `approve_streak`. *(Đánh đổi chấp nhận ở `v0.5.3`, sẽ đo lại ở `V060-04` khi `seq_len=15`.)*
- **Chống ghi đè cấu hình khi đang nạp ONNX nền (`V053-04` / B4)**: khi luồng `_swap_classifier_async` còn sống mà có thay đổi cấu hình classifier → bỏ qua nhánh in-place để đi qua rebuild (tăng `_classifier_swap_gen`), loại bỏ hoàn toàn thế hệ nạp cũ.
- **Chặn rào held-out hỏng im lặng (`V053-05` / B5)**: gom bộ đọc chung `read_sealed_manifest_hashes` tại `aerohand/ml/recording_io.py`, `raise ValueError` kèm đường dẫn nếu `sealed_manifest.json` lỗi JSON hoặc có `0` mã SHA-256 hợp lệ.
- **Nghiệm thu macOS M1 Pro**: `ruff` 0 lỗi, **199 passed**, i18n `vi`↔`en` khớp `15/15`, `D-dev` giữ nguyên **4 FT** (`32,33/h`, UCB `73,99/h`), `T-dev` giữ nguyên **17 FT** (`100,60/h`, UCB `150,89/h`), positives cục bộ giữ nguyên (`145s: 4 fr/1 bout`; `250s: 177 fr/28 bouts`).
