# Kết quả train và inference thử nghiệm

Ngày chạy: 2026-08-08. Motor luôn tắt (offline dry-run).

## Dữ liệu

- 92 ảnh JetRacer legacy có waypoint từ tên file.
- 498 pseudo-label từ video `1786085420908...mp4`, tạo bằng CV detector trong
  `notebook3`. Đây là training video, không dùng để báo cáo inference cuối.
- Tổng 590 mẫu; split cố định 501 train / 89 validation.
- Video đánh giá held-out: `1786085420913...mp4`, 102 frame, chưa xuất hiện trong
  pseudo-label training.

Pseudo-label không thay thế ground truth do người label. Kết quả này xác nhận pipeline
hoạt động và tạo baseline để active learning, chưa chứng minh xe sẵn sàng chạy thật.

## Train

```bash
python waypoint_lane_fusion/train_waypoint.py \
  waypoint_lane_fusion/dataset/train_combined.csv \
  --epochs 25 --batch 64 --workers 0 \
  --output waypoint_lane_fusion/artifacts/lane_resnet18_bootstrap.pt
```

Best validation normalized L2: **0.02851** (epoch 19), tương đương khoảng 6.4 px
trên ảnh 224 nếu quy đổi trực tiếp. Validation chứa pseudo-label nên metric này chủ
yếu đo khả năng bắt chước bootstrap detector.

## Inference video held-out

- 102/102 frame được xử lý và ghi video.
- CPU ONNX median: **21.48 FPS**, minimum 13.28 FPS.
- Mean confidence: 0.7756.
- Predicted x: 0.4886–0.6154; predicted y: 0.5972–0.6080.
- Mean absolute filtered steering: 0.1004.
- State: 2 frame startup `LANE_LOST`, sau đó 100 frame `NORMAL`.
- Mean throttle preview: 0.1385 (chỉ telemetry, không cấp motor).

So với CV detector như một pseudo-reference (không phải ground truth), 65 frame mà
CV detector báo confident đạt MAE x = 0.1096 và correlation = 0.7061. Vì hai hệ thống
không hoàn toàn độc lập, con số này chỉ dùng để kiểm tra sanity.

Artifacts:

- `artifacts/lane_resnet18_bootstrap.onnx`
- `artifacts/lane_resnet18_bootstrap.metrics.json`
- `artifacts/combined/combined_source_5fps.mp4`
- `artifacts/combined/combined_resnet18_inference_5fps.mp4`
- `artifacts/combined/combined_resnet18_metrics.csv`
- `artifacts/combined/combined_resnet18_summary.json`

## Kết luận

Pipeline train/export/inference đã chạy end-to-end. Model bootstrap tốt hơn model chỉ
train trên 92 ảnh legacy, nhưng chưa đủ bằng chứng để bật `--arm`. Bước tiếp theo đúng
là label tay 3k–6k frame đa dạng, giữ nguyên từng video hoàn chỉnh làm validation/test,
sau đó build TensorRT FP16 và benchmark trên đúng Jetson Nano.
