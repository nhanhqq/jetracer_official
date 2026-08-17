# Smart City

Pipeline riêng cho bài Smart City:

1. `SemanticLaneONNX` tái sử dụng `yolo_lane_following/artifacts/track_yolo26n_sem_best.onnx` để bám divider/road và không đi vào vùng forbidden.
2. YOLO26 detector nhận biển đi thẳng, trái, phải, cấm, đèn đỏ/xanh và các vạch `crosswalk`/`stop_line`.
3. `SmartCityPolicy` xác nhận tín hiệu theo nhiều frame. Đèn đỏ có ưu tiên cao nhất và giữ trạng thái dừng; chỉ đủ số frame đèn xanh mới cho đi.
4. Runtime lưu route command, xử lý hướng tương đối tại giao lộ, ghi CSV latency/FPS và mặc định dry-run.

Config hiện chạy bản `smart_city_yolo26n_urban_pseudo.onnx`, là bản provisional
đã fine-tune thêm trên pseudo-label urban. Bản này giúp chạy thử trên dữ liệu
urban ngay, nhưng không được xem là nhãn thật; bản reviewed phải thay thế nó
trước khi thi.

Lane target ngoài vùng camera bị coi là mất lane và phát lệnh dừng, thay vì
cho phép polynomial extrapolation tạo ra một góc lái không an toàn.

## Dataset

Đặt dataset detection chuẩn YOLO vào `smart_city/datasets/traffic/`:

```text
images/train/*.jpg
images/val/*.jpg
labels/train/*.txt
labels/val/*.txt
```

The currently available `JetRacer_AI/jetracer/notebooks/urban/urban_dataset_A`
contains 427 images but no annotations. Prepare and label it with:

```bash
python3 smart_city/prepare_dataset.py \
  --source ../JetRacer_AI/jetracer/notebooks/urban/urban_dataset_A
python3 smart_city/label_traffic_gui.py
```

The GUI uses keys `0`-`7` to select the 8 classes in `data.yaml`, mouse drag to
draw a box, `s` to save, and `n`/`p` to change image.

For a faster first pass, generate reviewable pseudo-labels from the trained
model:

```bash
python3 smart_city/pseudo_label_urban.py \
  --source ../JetRacer_AI/jetracer/notebooks/urban/urban_dataset_A \
  --device 0
```

Use `--device cpu` when reviewing on a machine without CUDA.

Inspect `datasets/urban_pseudo/previews/`, correct labels with the GUI, and
press `s` for every reviewed image. The GUI updates `review_manifest.csv`.
Then build a safe fine-tune set:

```bash
python3 smart_city/merge_reviewed_dataset.py
python3 smart_city/audit_dataset.py \
  --data smart_city/datasets/traffic_reviewed/data.yaml
```

For a temporary provisional model while urban review is incomplete, use a
separate output and mark the result as pseudo-trained:

```bash
python3 smart_city/merge_reviewed_dataset.py \
  --allow-unreviewed \
  --output smart_city/datasets/traffic_pseudo_bootstrap
python3 smart_city/train_traffic.py \
  --data smart_city/datasets/traffic_pseudo_bootstrap/data.yaml \
  --model smart_city/artifacts/smart_city_yolo26n_best.pt \
  --epochs 10 --name smart_city_yolo26n_urban_pseudo \
  --output smart_city/artifacts/smart_city_yolo26n_urban_pseudo.onnx
```

This provisional ONNX is not competition ground truth and must be replaced
after the manifest is fully reviewed.

The included synthetic dataset is complete enough for a temporary bootstrap
train; the 427-image urban set is still review data until every manifest row
is reviewed.

Tên lớp nằm trong [data.yaml](data.yaml). Thứ tự policy là đèn trước, biển cấm sau,
rồi mới đến biển chỉ dẫn. Với `bien_cam`, policy chọn trái/phải random theo
`traffic.forbidden_random_seed`; biển chỉ dẫn `di_thang`, `re_trai`, `re_phai`
chỉ được dùng khi không có biển cấm. Các alias `forbidden_left/right/straight`
vẫn hỗ trợ chọn hướng thay thế tương ứng.

## Train/export

```bash
python3 smart_city/generate_dataset.py \
  --roads notebook3/old_codes/road_following_A/apex \
  --signs notebook3/bien_bao \
  --output smart_city/datasets/traffic \
  --imgsz 320
python3 smart_city/audit_dataset.py
python3 smart_city/train_traffic.py \
  --data smart_city/datasets/traffic/data.yaml \
  --model yolo26n.pt --device 0 --workers 0
python3 smart_city/export_lane_onnx.py
```

Lệnh train sẽ tạo checkpoint rồi export `smart_city/artifacts/smart_city_yolo26n.onnx`. Cần chạy trong đúng Docker Python 3.8/Ultralytics của Jetson hoặc môi trường tương thích. Audit chỉ chặn khi thiếu file label; nhãn tổng hợp hiện đã đủ để bootstrap, còn urban label vẫn phải người duyệt lại.

## Offline/runtime

```bash
python3 -m unittest discover -s smart_city/tests -v
python3 smart_city/runtime.py --source path/to/replay.mp4 --display
python3 smart_city/replay_folder.py \
  --source ../JetRacer_AI/jetracer/notebooks/urban/urban_dataset_A
```

`replay_folder.py` writes a disarmed JSON report with lane-valid percentage,
red/lane-loss stop counts, detections, and latency percentiles.

Không dùng `--arm` cho đến khi đã kiểm tra replay, CSV, red-light stop và chiều lái trên giá kê bánh. ONNX engine phải được benchmark lại trên Jetson Nano; FPS workstation không phải bằng chứng xe thật đạt ngưỡng 300 ms.
