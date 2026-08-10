# YOLO26 Semantic Lane Following cho JetRacer

Pipeline chính dùng `yolo26n-sem` (semantic segmentation), không dùng YOLO instance segmentation cũ. Mỗi pixel được phân loại thành `background`, `road`, `divider`, `forbidden` hoặc `obstacle`. Target mặc định là divider cam. Xe chỉ rời divider khi mask obstacle cắt corridor của xe; corridor mới phải nằm trong road và không đè lên `forbidden` (phần trắng). Khi obstacle không còn cắt corridor, target trở về divider ngay. Không đủ corridor an toàn hoặc mất divider thì dừng.

## Train Semantic

```bash
python yolo_lane_following/prepare_semantic_dataset.py --clean
python yolo_lane_following/train_semantic.py --epochs 80 --device 0
```

Dataset sinh ra tại `semantic_dataset/`: 1 PNG mask lossless cho mỗi ảnh, class ID 0..4 theo `data.yaml`. `workers=0` là mặc định để không lỗi shared-memory trên Jetson/container.

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
   và khoảng gần vật cản; mất divider quá 3 frame hoặc vật cản quá gần thì dừng.

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

## 4. Chạy xe

Đặt xe lên giá kê bánh, chỉnh `steering_gain/offset`, xác nhận chiều lái, thử ga thấp,
sau đó mới chạy trong vùng có nút dừng khẩn cấp:

```bash
python yolo_lane_following/run.py --source camera --arm
```

Các ngưỡng ga/góc lái nằm trong `config.yaml`. Không tăng `throttle_max` trước khi
log cho thấy divider confidence ổn định và fail-safe dừng đúng.
