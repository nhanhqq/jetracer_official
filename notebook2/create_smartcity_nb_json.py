import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.6.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

text_cells = [
    "### 1. Khởi tạo Camera và Xe",
    "### 2. Khởi tạo 2 Mô hình AI (YOLOv8 + ResNet18)",
    "### 3. Giao diện điều khiển & Trạng thái",
    "### 4. Logic Xử lý Giao lộ (State Machine) & Ghi Log"
]

code_cells = [
    # 1. Khởi tạo Camera và Xe
    """from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
import cv2
import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display
from jetcam.utils import bgr8_to_jpeg

car = NvidiaRacecar()
camera = CSICamera(width=224, height=224, capture_fps=30) # Hạ FPS xuống 30 để tránh quá tải khi chạy 2 AI""",

    # 2. Khởi tạo 2 Mô hình
    """import torch
import torchvision
import torchvision.transforms as transforms
import PIL.Image

device = torch.device('cuda')

# ----- AI 1: BÁM ĐƯỜNG (ResNet18) -----
output_dim = 2 # Chỉ cần 2 tọa độ (x, y) của Apex, vì né vật cản tĩnh có thể dùng chung YOLO
model_lane = torchvision.models.resnet18(pretrained=False)
model_lane.fc = torch.nn.Linear(512, output_dim)
model_lane = model_lane.to(device)
# Bạn thay đường dẫn tới model bám lane đã train cho Smart City nhé
# model_lane.load_state_dict(torch.load('lane_smartcity_model.pth'))
model_lane = model_lane.eval().half()

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_lane(image):
    image = PIL.Image.fromarray(image)
    image = TRANSFORMS(image).to(device).half()
    return image[None, ...]

# ----- AI 2: NHẬN DIỆN BIỂN BÁO (YOLOv8) -----
# Yêu cầu: pip install ultralytics
# Lưu ý: Đây là code mẫu, bạn cần thay file 'best.pt' bằng trọng số YOLO bạn tự train!
try:
    from ultralytics import YOLO
    model_yolo = YOLO('yolov8n.pt') # Thay bằng 'best.pt' sau khi train biển báo
    print("YOLO Loaded!")
except Exception as e:
    print("Vui lòng cài đặt Ultralytics để dùng YOLO: pip install ultralytics")
    model_yolo = None""",

    # 3. Giao diện
    """state_widget = widgets.ToggleButtons(options=['On', 'Off'], description='Camera', value='Off')
prediction_widget = widgets.Image(format='jpeg', width=camera.width, height=camera.height)

status_html = widgets.HTML(value="<b>Trạng thái xe:</b> Sẵn sàng")
memory_html = widgets.HTML(value="<b>Bộ nhớ đệm (Biển báo):</b> Trống")

steering_gain_slider  = widgets.FloatSlider(description='Steering Gain', min=-1.0, max=1.0, value=-0.65, step=0.01)
steering_bias_slider  = widgets.FloatSlider(description='Steering Bias', min=-0.5, max=0.5, value=0.0, step=0.01)
throttle_slider = widgets.FloatSlider(description='Throttle (Tiến)', min=0.0, max=1.0, value=0.15, step=0.01)
intersection_flag = widgets.ToggleButton(description='Đang ở Ngã Tư', value=False, button_style='warning') # Nút thủ công để test rẽ ngã tư

ui = widgets.HBox([
    widgets.VBox([
        status_html,
        memory_html,
        steering_gain_slider,
        steering_bias_slider,
        intersection_flag,
        state_widget
    ]),
    prediction_widget,
    throttle_slider
])
display(ui)""",

    # 4. Logic & Ghi Log
    """import time
import pandas as pd
import datetime

# Biến State Machine
current_state = 'DRIVE' # 'DRIVE' hoặc 'STOP' (khi gặp đèn đỏ)
pending_turn = None # 'LEFT', 'RIGHT', 'STRAIGHT'
log_data = []

def update(change):
    global current_state, pending_turn, log_data
    start_time = time.time()
    new_image = change['new']
    
    if state_widget.value == 'Off':
        car.throttle = 0.0
        return
        
    prediction = new_image.copy()
    detected_signs = []
    
    # -----------------------------------------
    # PHASE 1: NHẬN THỨC (PERCEPTION)
    # -----------------------------------------
    
    # 1. Chạy YOLO tìm biển báo / đèn đỏ
    if model_yolo is not None:
        results = model_yolo(new_image, verbose=False, imgsz=224)[0]
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model_yolo.names[class_id]
            
            if conf > 0.5: # Độ tin cậy > 50%
                detected_signs.append(label)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(prediction, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(prediction, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
    # 2. Chạy ResNet18 tìm Tim đường
    img_tensor = preprocess_lane(new_image)
    # Tạm comment để mô phỏng nếu chưa có model ResNet 2 output:
    # output = model_lane(img_tensor).detach().cpu().numpy().flatten()
    # ax, ay = output[0], output[1]
    ax, ay = 0.0, 0.0 # Giả lập Apex đang ở giữa đường (Chờ bạn train model)
    
    # -----------------------------------------
    # PHASE 2: CẬP NHẬT TRẠNG THÁI (STATE MACHINE)
    # -----------------------------------------
    for sign in detected_signs:
        if "red" in sign.lower():
            current_state = 'STOP'
        elif "green" in sign.lower():
            current_state = 'DRIVE'
        elif "left" in sign.lower():
            pending_turn = 'LEFT'
        elif "right" in sign.lower():
            pending_turn = 'RIGHT'
        elif "straight" in sign.lower():
            pending_turn = 'STRAIGHT'
            
    memory_html.value = f"<b>Bộ nhớ đệm (Biển báo):</b> {pending_turn}"
    status_html.value = f"<b>Trạng thái xe:</b> {current_state}"
    
    # -----------------------------------------
    # PHASE 3: RA QUYẾT ĐỊNH ĐIỀU KHIỂN (CONTROL)
    # -----------------------------------------
    final_steering = 0.0
    final_throttle = 0.0
    decision_str = "Đi thẳng"
    
    if current_state == 'STOP':
        final_throttle = 0.0
        decision_str = "Dừng (Đèn Đỏ)"
    else:
        # Nếu đang chạy bình thường
        final_throttle = throttle_slider.value
        
        # Kiểm tra xem có đang ở ngã tư không (intersection_flag dùng để test mô phỏng)
        # Trong thực tế, bạn có thể dùng một Model AI thứ 3, hoặc dựa vào Tọa độ Y của Apex để biết đã đến ngã tư chưa
        if intersection_flag.value and pending_turn is not None:
            # Thực thi lệnh rẽ từ bộ nhớ
            if pending_turn == 'LEFT':
                final_steering = -1.0 # Bẻ lái gắt trái
                decision_str = "Rẽ Trái (Giao lộ)"
            elif pending_turn == 'RIGHT':
                final_steering = 1.0 # Bẻ lái gắt phải
                decision_str = "Rẽ Phải (Giao lộ)"
                
            # Tuỳ chọn: Tự động xóa bộ nhớ sau khi rẽ xong
            # pending_turn = None
        else:
            # Bám lane bình thường bằng điểm Apex
            final_steering = ax * steering_gain_slider.value + steering_bias_slider.value
            decision_str = "Bám Lane"
            
    # Gửi lệnh xuống Motor
    # car.steering = final_steering
    # car.throttle = final_throttle
    
    # -----------------------------------------
    # PHASE 4: GHI LOG SMART CITY
    # -----------------------------------------
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0
    
    log_data.append({
        'timestamp': datetime.datetime.now().isoformat(),
        'fps': round(fps, 2),
        'detected_object/sign': ', '.join(detected_signs) if detected_signs else 'None',
        'confidence': 1.0, # Lấy trung bình YOLO conf nếu cần
        'decision': decision_str,
        'latency_ms': round(latency_ms, 2),
        'control_output': f"steering:{final_steering:.2f},throttle:{final_throttle:.2f}",
        'event': "Vào Giao Lộ" if intersection_flag.value else "Bình thường"
    })
    
    if len(log_data) % 50 == 0:
        pd.DataFrame(log_data).to_csv('smartcity_log.csv', index=False)
        
    # Cập nhật hình ảnh UI
    px = int(camera.width * (ax / 2.0 + 0.5))
    py = int(camera.height * (ay / 2.0 + 0.5))
    cv2.circle(prediction, (px, py), 8, (0, 0, 255), -1)
    prediction_widget.value = bgr8_to_jpeg(prediction)

# Chạy Live
update({'new': camera.value})
camera.observe(update, names='value')
camera.running = True"""
]

for idx in range(len(text_cells)):
    # Markdown cell
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text_cells[idx]]
    })
    
    # Code cell
    lines = code_cells[idx].split('\n')
    source_lines = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
    
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

with open('smart_city_v1.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
