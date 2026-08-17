import cv2
import os
import sys
from ultralytics import YOLO

def main():
    pt_path = 'best.pt'
    trt_path = 'best.engine'

    # Kiểm tra xem file weights gốc có tồn tại không
    if not os.path.exists(pt_path):
        print(f"Không tìm thấy file weights gốc tại {pt_path}")
        sys.exit(1)

    # 1. Dịch model sang TensorRT (nếu chưa dịch)
    if not os.path.exists(trt_path):
        print("Bắt đầu dịch model từ PyTorch (.pt) sang TensorRT (.engine)...")
        print("Quá trình này có thể mất vài phút. Vui lòng đợi!")
        try:
            model = YOLO(pt_path)
            # YOLOv8 hỗ trợ export tự động sang TensorRT thông qua format='engine'
            # Yêu cầu hệ thống phải cài đặt sẵn tensorrt (pip install tensorrt)
            model.export(format='engine', device=0, half=True) # half=True để dùng FP16 tối ưu tốc độ
            print("Đã dịch xong sang TensorRT!")
        except Exception as e:
            print(f"\nLỗi khi dịch sang TensorRT: {e}")
            print("Đang chuyển sang chạy fallback bằng file .pt gốc...")
            trt_path = pt_path # Fallback lại file gốc nếu lỗi
            
    # 2. Load model để Inference
    print(f"Đang load mô hình từ: {trt_path}")
    try:
        model = YOLO(trt_path)
    except Exception as e:
        print(f"Không thể load model: {e}")
        sys.exit(1)

    # 3. Mở Camera Laptop (ID = 0)
    print("Đang mở Camera laptop...")
    cap = cv2.VideoCapture(0)
    
    # Tuỳ chỉnh độ phân giải camera nếu cần
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Lỗi: Không thể kết nối với Webcam!")
        sys.exit(1)

    print("Camera đã mở. Nhấn phím 'q' trên cửa sổ video để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera.")
            break
            
        # 4. Inference trực tiếp
        # verbose=False để không in log liên tục làm rác terminal
        results = model(frame, verbose=False)
        
        # Lấy ảnh đã vẽ sẵn Bounding Box và Label
        annotated_frame = results[0].plot()
        
        # 5. Hiển thị lên màn hình
        cv2.imshow("YOLOv8 Live Inference (Nhấn Q để thoát)", annotated_frame)
        
        # Thoát vòng lặp khi nhấn 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
