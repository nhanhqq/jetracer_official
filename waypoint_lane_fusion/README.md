# JetRacer waypoint lane fusion (YOLOv5n)

Module mới này tham khảo calibration/camera trong `notebook3` và smoothing, fail-safe,
telemetry trong `yolo_lane_following`, nhưng **không dùng YOLO để bám lane**:

```text
CSI frame -> ResNet18 waypoint -> EMA -> heading + PD -> low-pass/rate limit -> steering
        `-> async YOLOv5n -> state machine -> adaptive throttle -------------> car
```

Lane chạy ở fast loop. YOLOv5n nhận frame qua queue một phần tử; frame cũ bị bỏ và
lane loop không bao giờ đợi YOLO. Runtime mặc định dry-run, chỉ `--arm` mới cấp ga.

Kết quả train/inference thực tế hiện tại được ghi tại [EVALUATION.md](EVALUATION.md).

Notebook chạy camera/UI thời gian thực trên Jetson:

```text
waypoint_lane_fusion/lane_fusion_live.ipynb
```

Notebook tự chọn TensorRT engine nếu engine đã được build, fallback về ONNX, và mặc
định disarm motor. Video kiểm thử tổng hợp có thể tái tạo bằng:

```bash
python3 waypoint_lane_fusion/make_combined_inference.py
```

## Những phần đã triển khai

- collector và GUI click nhãn normalized `x,y`;
- ResNet18 head `512 -> 128 -> 3` (`x,y,confidence`), train trên workstation;
- TorchScript -> ONNX opset 13 để build TensorRT FP16 trên Nano;
- EMA waypoint, heading error, PD, low-pass và giới hạn thay đổi servo;
- ga theo độ cua/confidence, tăng chậm và giảm nhanh;
- hysteresis lane-lost; obstacle/red-light có ưu tiên dừng;
- YOLOv5n bất đồng bộ, TTL chống dùng detection cũ;
- overlay và CSV đầy đủ để active learning.

## Quy trình

Từ thư mục gốc repo:

```bash
python -m waypoint_lane_fusion.collect_frames --source camera --output waypoint_lane_fusion/dataset/images
python -m waypoint_lane_fusion.label_waypoints waypoint_lane_fusion/dataset/images
python waypoint_lane_fusion/train_waypoint.py waypoint_lane_fusion/dataset/images/labels.csv \
  --output waypoint_lane_fusion/artifacts/lane_resnet18.pt
python waypoint_lane_fusion/export_onnx.py waypoint_lane_fusion/artifacts/lane_resnet18.pt \
  --output waypoint_lane_fusion/artifacts/lane_resnet18.onnx
```

Kiểm tra video/camera không cấp ga:

```bash
python -m waypoint_lane_fusion.run --source VIDEO.mp4 --display --output-video preview.mp4
python -m waypoint_lane_fusion.run --source camera --display
```

Sau khi đặt `yolov5n.pt` hoặc weights biển báo YOLOv5n custom vào folder và sửa class
mapping trong `behavior.py` nếu cần:

```bash
python -m waypoint_lane_fusion.run --source camera --yolo --display
```

Chỉ arm sau khi kê bánh, xác nhận chiều steering và fail-safe:

```bash
python -m waypoint_lane_fusion.run --source camera --yolo --arm
```

## Jetson Nano: build TensorRT và chạy thử

Engine phải được build trên chính Jetson đích, không copy `.engine` từ RTX/x86:

```bash
chmod +x waypoint_lane_fusion/build_engine_jetson.sh
./waypoint_lane_fusion/build_engine_jetson.sh
```

Backend dùng trực tiếp CUDA Driver API có sẵn trong JetPack; không cần cài PyCUDA.

Kiểm tra engine bằng video trước, motor luôn tắt nếu không có `--arm`:

```bash
python3 -m waypoint_lane_fusion.run \
  --lane-backend tensorrt \
  --lane-model waypoint_lane_fusion/artifacts/lane_resnet18_bootstrap_fp16.engine \
  --source notebook3/test/1786085420913_202326730929621029_7442541399315262114.mp4 \
  --output-video waypoint_lane_fusion/artifacts/demo/nano_tensorrt_test.mp4
```

Sau đó dry-run camera:

```bash
python3 -m waypoint_lane_fusion.run \
  --lane-backend tensorrt \
  --lane-model waypoint_lane_fusion/artifacts/lane_resnet18_bootstrap_fp16.engine \
  --source camera --display
```

## Đánh giá khả thi

Kiến trúc phù hợp Jetson Nano hơn YOLO26 lane segmentation: lane ResNet18 có pipeline
nhẹ và ổn định tần số, YOLOv5n chạy thưa/bất đồng bộ. Tuy vậy `confidence` học từ sai
số regression chỉ là proxy, không phải xác suất đã hiệu chuẩn; phải đo threshold trên
validation và các frame out-of-distribution. Dynamic lookahead đúng nghĩa cần dataset
có nhiều waypoint/lookahead hoặc output path nhiều điểm; baseline hiện tại dùng `y`
được model dự đoán và heading geometry, chưa giả vờ thay đổi waypoint sau inference.

Các target 20 Hz lane và 5-15 FPS YOLO chỉ là tiêu chí nghiệm thu. Chúng chưa được xác
nhận cho đến khi ONNX được build TensorRT FP16 và log trên đúng Nano. YOLOv5 qua
`torch.hub` là baseline tích hợp; deployment thi đấu nên export YOLOv5n ONNX/TensorRT
và thay backend worker để giảm RAM/latency.
