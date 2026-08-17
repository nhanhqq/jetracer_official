# YOLO26 Semantic Lane Following cho JetRacer

Pipeline chính dùng `yolo26n-sem` (semantic segmentation), không dùng YOLO instance segmentation cũ. Runtime lane-following hiện dùng best model engine `artifacts/track_yolo26n_sem_nano_fp16.engine` và chỉ dùng các mask `road`, `divider`, `forbidden`; nhánh vật cản/cube đã tắt. Target mặc định là divider cam, còn vùng `forbidden` (phần trắng) vẫn là vùng không được đi vào. Mất lane quá ngưỡng thì dừng.

## Train Semantic

```bash
python yolo_lane_following/prepare_semantic_dataset.py --clean
python yolo_lane_following/train_semantic.py --epochs 80 --device 0
```

Dataset sinh ra tại `semantic_dataset/`: 1 PNG mask lossless cho mỗi ảnh, class ID 0..4 theo `data.yaml`. `workers=0` là mặc định để không lỗi shared-memory trên Jetson/container.

Model thi đấu hiện tại là `track_yolo26n_sem_cube_best.pt`. Nó được fine-tune từ
model lane gốc (không train lại từ đầu) với cube 5--10 cm có nhiều màu, bóng và
phối cảnh. Tạo lại mẫu và đo riêng obstacle bằng:

```bash
python3 yolo_lane_following/augment_cube_obstacles.py --copies 1
python3 yolo_lane_following/train_semantic.py \
  --model yolo_lane_following/artifacts/track_yolo26n_sem_best.pt \
  --epochs 24 --batch 4 --name track_yolo26n_sem_cube_finetune \
  --output-name track_yolo26n_sem_cube_best.pt
python3 yolo_lane_following/evaluate_semantic_obstacle.py \
  --model yolo_lane_following/artifacts/track_yolo26n_sem_cube_best.pt \
  --pattern '*_cube_*.jpg' --all-classes
```

Trên split validation hiện tại, checkpoint cube đạt obstacle recall `0.9755` và
IoU `0.9161` trên 98 ảnh cube tổng hợp. So với engine gốc, mean IoU trên cube tăng
`0.7243 -> 0.8255`; mean IoU trên ảnh gốc thay đổi `0.8279 -> 0.8227`. Trung bình
cân bằng hai split tăng `0.7761 -> 0.8241`, nên checkpoint cube hiện được chọn thay
vì train thêm khi chưa có ảnh cube thật. Đây không thay thế ảnh cube thật trên sa bàn.

Trên replay 102 frame không có cube, engine mới có cùng median lane confidence
`0.5645`, target jump median thấp hơn (`5.129` so với `5.511` pixel), cùng giới hạn
steering delta `0.16`; có 3 dropout đơn-frame được controller giảm ga mượt, trong
khi engine gốc không dropout trên đoạn ngắn này.

Smoke tích hợp dưới đây tạo lane lock bằng ảnh sạch, sau đó lặp ảnh cube để kiểm tra
chuỗi model -> temporal confirmation -> planner -> controller mà không load motor:

```bash
python3 yolo_lane_following/smoke_semantic_controller.py \
  --preroll-image yolo_lane_following/semantic_dataset/images/val/track_s01_000032.jpg \
  --image yolo_lane_following/semantic_dataset/images/val/track_s01_000032_cube_09cm_00.jpg \
  --model yolo_lane_following/artifacts/track_yolo26n_sem_cube_nano_fp16.engine
```

Kết quả hiện tại xác nhận obstacle ở frame thứ ba (`risk=0.798`), luôn tiến với
`avoid:obstacle`, ga ramp `0.03 -> 0.06 -> 0.09 -> 0.12` và không phát
`reverse:obstacle`. Nếu sau `4.0 s` vẫn không lấy lại divider, controller chuyển
sang `stop:obstacle_no_divider`, ramp lái về 0 và giữ ga 0 cho đến khi risk giảm.

## Dry Run

```bash
python yolo_lane_following/run.py \
  --source notebook3/test/1786085420908_202326730929621029_7442541399315262114.mp4 \
  --output-video yolo_lane_following/artifacts/semantic_test.mp4
```

Không có `--arm` nghĩa là không ghi ga/lái ra phần cứng. Chỉ dùng `--arm` sau khi đã kiểm tra overlay, CSV telemetry và thử xe trên giá kê bánh.

Hướng dẫn cài đặt và build TensorRT trên Jetson Nano nằm tại [JETSON_NANO_SETUP.md](JETSON_NANO_SETUP.md). Notebook live tương ứng là [semantic_lane_live.ipynb](semantic_lane_live.ipynb).

---

## Pipeline instance cũ

Pipeline này kế thừa camera, calibration xe và video kiểm thử trong `notebook3`,
nhưng thay nhận diện lane theo ngưỡng màu bằng YOLO26 segmentation.

## Kiến trúc

1. Một `lane_yolo26n_seg` duy nhất sinh mask `road`, `divider`, `obstacle`;
   mỗi instance mask đồng thời có detection box.
2. Pipeline này hoàn toàn không load hay thay đổi model/dataset biển báo Smart City.
3. Mask divider được fit đa thức theo trục dọc. Điểm preview của divider là mục tiêu,
   vì `target_mode: divider` yêu cầu tâm camera nằm trên đường phân cách.
4. PID + heading preview tự sinh steering. Throttle tự giảm theo độ cong, confidence
   và khoảng gần vật cản; mất divider quá 3 frame mà không có road hint thì dừng.
   Đoạn thẳng sạch tăng dần đến `throttle_max`; góc cua yêu cầu giảm ga ngay cả khi
   steering đang được slew/filter để tránh lao vào cua rồi mới giảm.

Không có thuật toán nào bảo đảm bám đường “tuyệt đối”. Chất lượng phụ thuộc nhãn,
camera calibration, độ trễ và độ bám cơ khí. Vì an toàn, chương trình mặc định dry-run;
chỉ `--arm` mới cấp ga.

## 1. Tạo và sửa dataset segmentation

Dataset road cũ không có polygon segmentation. Công cụ dưới đây bootstrap road/divider,
giữ trọn một video làm validation và sinh thêm obstacle có mask chính xác:

```bash
python yolo_lane_following/bootstrap_dataset.py notebook3/test/*.mp4 --every 5
```

Không đưa video `_inference.mp4` vào dataset vì chúng đã bị vẽ overlay. Các nhãn
pseudo-label vẫn nên được rà soát bằng CVAT trước một đợt train thi đấu cuối cùng.

## 2. Train và export trên đúng Jetson đích

```bash
python yolo_lane_following/train_segmentation.py --epochs 80 --device 0
python yolo_lane_following/export_tensorrt.py
```

TensorRT engine phụ thuộc GPU/CUDA/TensorRT, vì vậy phải build và benchmark trên xe
thi đấu. Export dùng `nms=False`; YOLO26 là end-to-end/NMS-free, code cũng không thêm
NMS phía ứng dụng.

### ONNX dành cho Jetson Nano 4GB

Jetson Nano dùng JetPack 4.6.x/TensorRT 8.2. ONNX export trực tiếp của YOLO26 có toán
tử `Mod` mà parser TensorRT 8.2 không hỗ trợ. Không dùng file
`lane_yolo26n_seg_best.onnx` thô. Script dưới đây export opset 13, batch tĩnh 1,
224x224 và thay đúng phép modulo chỉ số bằng các toán tử tương đương được TensorRT
8.2 hỗ trợ:

```bash
python yolo_lane_following/export_onnx_jetson_nano.py
```

Artifact cần chép sang Nano là `artifacts/lane_yolo26n_seg_nano.onnx`. Build engine
ngay trên Nano (không chép engine build từ RTX/x86 sang):

```bash
chmod +x yolo_lane_following/build_engine_jetson_nano.sh
yolo_lane_following/build_engine_jetson_nano.sh
```

Script chỉ thành công sau khi `trtexec` parse, build FP16 và benchmark engine. Dùng
workspace 512 MiB để phù hợp RAM 4 GB. Nếu bước này chưa chạy thành công trên đúng
Nano thì chưa được coi là đã xác nhận deployment phần cứng.

## 3. Kiểm thử không cấp ga

```bash
python yolo_lane_following/run.py --source notebook3/test/VIDEO.mp4 --display
python yolo_lane_following/run.py --source camera --display
```

Tạo video từ toàn bộ 689 ảnh dataset và inference bằng ONNX:

```bash
python yolo_lane_following/images_to_video.py --fps 20
python yolo_lane_following/run.py \
  --model yolo_lane_following/artifacts/lane_yolo26n_seg_nano.onnx \
  --device cpu \
  --source yolo_lane_following/artifacts/dataset_images.mp4 \
  --output-video yolo_lane_following/artifacts/dataset_images_inference_onnx.mp4
```

Kiểm tra CSV trong `logs`: FPS pipeline phải >=20 theo đề. Mục tiêu 50–100 Hz chỉ có
thể xác nhận bằng log trên phần cứng; dùng `imgsz=224`, YOLO26n-seg TensorRT FP16 và
tắt display để giảm latency.

Một dry-run tham chiếu bằng model `.pt` trên RTX 3090, video 102 frame đạt median
40.1 FPS / 24.4 ms sau warm-up. Đây chỉ là kiểm tra pipeline, không thay thế benchmark
TensorRT trên Jetson đích.

Dry-run CSI trực tiếp trên Jetson Nano với engine cube FP16 hiện tại đạt `23.9 FPS`
và median latency `41.8 ms` trên 709 frame. Camera lúc đo nhìn vào một phòng clutter,
không phải sa bàn; dù obstacle risk lên `0.729`, lane-lock confirmation giữ `0`
throttle trên toàn bộ 709 frame. Đây là bằng chứng runtime/camera và fail-safe,
chưa phải kiểm thử lái xe.

## 4. Chạy xe

Đặt xe lên giá kê bánh, chỉnh `steering_gain/offset`, xác nhận chiều lái, thử ga thấp,
sau đó mới chạy trong vùng có nút dừng khẩn cấp:

```bash
python yolo_lane_following/run.py --source camera --arm
```

Các ngưỡng ga/góc lái nằm trong `config.yaml`. Không tăng `throttle_max` trước khi
log cho thấy divider confidence ổn định và fail-safe dừng đúng.

Controller không được khởi tạo maneuver trắng hoặc vật cản trước khi có lane hợp lệ
hoặc đã từng khóa lane. Vì vậy ARM khi camera chưa nhìn thấy đường vẫn giữ ga bằng 0.

### Thu bằng chứng cube và đèn xanh thật, không có motor

Đặt xe/camera cố định, lần lượt đưa cube và hình tròn xanh sáng vào khung hình rồi chạy:

```bash
python3 yolo_lane_following/validate_live_safety.py --duration 30
```

Script này không import `JetRacerController`, không có tùy chọn ARM và không thể ghi
lệnh motor. CSV cùng ảnh `obstacle_confirmed_*.jpg` / `green_confirmed_*.jpg` được
lưu trong `artifacts/live_validation/`. Chỉ coi validation đạt khi ảnh lưu cho thấy
đúng vật thật, không dựa riêng vào số risk.

Trong notebook, `COMPETITION` tắt nghĩa là chỉ cần bật `ARM MOTOR`: notebook tự đổi
sang `live` và xe chạy khi controller có lane-lock an toàn. Khi bật `COMPETITION`,
ARM cũng tự mở live nhưng motor vẫn dừng cho đến khi thấy hình tròn xanh lá sáng đủ
3 frame liên tiếp; quyền start sau đó được latch đến khi Stop, bỏ ARM hoặc đổi chế độ.
CSV ghi thêm `start_latency_ms`; detector xanh riêng lẻ đo median `1.236 ms`, p99
`1.408 ms` trên Jetson. Latency end-to-end vẫn phải xác nhận bằng đèn BTC thật.
