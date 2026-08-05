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
    "### Khởi tạo Camera và Xe",
    "### Tải Mô hình AI (Apex + Obstacle 10 tọa độ)",
    "### Giao diện Điều khiển (Live UI Tuning)",
    "### Logic Điều khiển (Road Following + Obstacle Avoidance + Fallback)"
]

code_cells = [
    # 1. Khởi tạo Camera và Xe
    """from working_jetracer import WorkingJetRacer as NvidiaRacecar
from jetcam.csi_camera import CSICamera
# from jetcam.usb_camera import USBCamera

car = NvidiaRacecar()
camera = CSICamera(width=224, height=224, capture_fps=65)
# camera = USBCamera(width=224, height=224, capture_fps=65)""",

    # 2. Tải Mô hình
    """import torch
import torchvision

device = torch.device('cuda')
output_dim = 10  # 2 cho apex, 8 cho bounding box (4 điểm)

model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, output_dim)
model = model.to(device)

# Load trọng số bạn vừa train ở file trước
model.load_state_dict(torch.load('road_obstacle_model.pth'))
model = model.eval().half() # half() để chạy nhanh hơn (FP16)""",

    # 3. Giao diện điều khiển
    """import cv2
import ipywidgets
import traitlets
from IPython.display import display
from ipywidgets import Layout, Button, Box
import ipywidgets.widgets as widgets
from jetcam.utils import bgr8_to_jpeg

state_widget = ipywidgets.ToggleButtons(options=['On', 'Off'], description='Camera', value='Off')
prediction_widget = ipywidgets.Image(format='jpeg', width=camera.width, height=camera.height)
status_widget = widgets.HTML(value="<b>Trạng thái:</b> Sẵn sàng")

steering_gain_slider  = widgets.FloatSlider(description='Steering Gain', min=-1.0, max=1.0, value=-0.65, step=0.01, orientation='horizontal', layout={'width': '300px'})
steering_bias_slider  = widgets.FloatSlider(description='Steering Bias', min=-0.5, max=0.5, value=0.0, step=0.01, orientation='horizontal', layout={'width': '300px'})
throttle_slider = widgets.FloatSlider(description='Throttle (Tiến)', min=0.0, max=1.0, value=0.15, step=0.01, orientation='vertical')
obstacle_sensitivity_slider = widgets.FloatSlider(description='Sợ Vật cản', min=0.0, max=2.0, value=0.8, step=0.01, orientation='horizontal', layout={'width': '300px'})

steering_gain_link   = traitlets.link((steering_gain_slider, 'value'), (car, 'steering_gain'))
steering_offset_link = traitlets.link((steering_bias_slider, 'value'), (car, 'steering_offset'))

ui = widgets.HBox([
    widgets.VBox([
        status_widget,
        steering_gain_slider,
        steering_bias_slider,
        obstacle_sensitivity_slider,
        state_widget
    ]),
    prediction_widget,
    throttle_slider
])
display(ui)""",

    # 4. Logic điều khiển
    """import time
import numpy as np
import torchvision.transforms as transforms
import PIL.Image
import pandas as pd
import datetime

TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = TRANSFORMS(image).to(device).half()
    return image[None, ...]

# Khởi tạo Log theo yêu cầu đề bài
log_data = []

# Biến trạng thái cho Fallback (Lùi xe)
is_reversing = False
reverse_start_time = 0
REVERSE_DURATION = 3.0 # Lùi xe trong 3 giây
APEX_Y_THRESHOLD = 0.8 # Nếu y > 0.8 (ở sát mép dưới màn hình) tức là mất đường

def update(change):
    global is_reversing, reverse_start_time, log_data
    start_time = time.time()
    new_image = change['new']
    
    if state_widget.value == 'Off':
        car.throttle = 0.0
        return
        
    image = preprocess(new_image)
    output = model(image).detach().cpu().numpy().flatten()
    
    # Lấy tọa độ (trong khoảng -1 đến 1)
    ax, ay = output[0], output[1]
    
    # Fallback Logic: Lỡ đi quá vạch / mất đường
    if not is_reversing and ay > APEX_Y_THRESHOLD:
        is_reversing = True
        reverse_start_time = time.time()
        status_widget.value = "<b style='color:red;'>Trạng thái: MẤT ĐƯỜNG! ĐANG ĐI LÙI...</b>"
        
    if is_reversing:
        if time.time() - reverse_start_time < REVERSE_DURATION:
            # Thực hiện lùi xe và bẻ vô lăng ngược lại để tìm đường
            car.throttle = -0.2
            car.steering = -ax # Đánh lái ngược để quay đầu
            
            # Chỉ vẽ hình để debug, bỏ qua logic né vật cản
            prediction = new_image.copy()
            cv2.putText(prediction, 'REVERSING', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            prediction_widget.value = bgr8_to_jpeg(prediction)
            return
        else:
            is_reversing = False
            status_widget.value = "<b style='color:green;'>Trạng thái: Đã bắt lại đường</b>"
            
    # ------ Normal Driving Logic ------
    
    # 1. Tính diện tích Bounding Box vật cản để xem nó to/nhỏ
    bbox_x = output[2::2] # Lấy mảng [x1, x2, x3, x4]
    bbox_y = output[3::2] # Lấy mảng [y1, y2, y3, y4]
    
    min_x, max_x = np.min(bbox_x), np.max(bbox_x)
    min_y, max_y = np.min(bbox_y), np.max(bbox_y)
    
    bbox_area = (max_x - min_x) * (max_y - min_y)
    bbox_center_x = (min_x + max_x) / 2.0
    
    # 2. Logic Tránh vật cản
    avoidance_steering = 0.0
    # Nếu diện tích đủ lớn (chướng ngại vật ở gần)
    if bbox_area > 0.05: 
        # Nếu vật cản ở bên phải tâm (bbox_center_x > 0), bẻ lái mạnh sang trái (âm)
        # Nếu vật cản ở bên trái tâm (bbox_center_x < 0), bẻ lái mạnh sang phải (dương)
        avoidance_steering = -np.sign(bbox_center_x) * obstacle_sensitivity_slider.value * bbox_area
        
    # 3. Tổng hợp Steering (Bám đường + Né)
    base_steering = ax
    final_steering = base_steering + avoidance_steering
    
    # Giới hạn lái trong khoảng [-1, 1]
    final_steering = max(-1.0, min(1.0, final_steering))
    
    car.steering = final_steering
    car.throttle = throttle_slider.value
    
    # 4. Vẽ hiển thị
    def to_pixel(val, max_val):
        return int(max_val * (val / 2.0 + 0.5))
        
    prediction = new_image.copy()
    
    # Vẽ Apex
    px, py = to_pixel(ax, camera.width), to_pixel(ay, camera.height)
    cv2.circle(prediction, (px, py), 8, (0, 0, 255), -1)
    
    # Vẽ Bounding Box
    pts = []
    for i in range(4):
        x = to_pixel(output[2 + i*2], camera.width)
        y = to_pixel(output[2 + i*2 + 1], camera.height)
        pts.append((x, y))
    
    # Đổi màu Bounding box: Đỏ nếu né, Xanh lá nếu an toàn
    bbox_color = (0, 0, 255) if bbox_area > 0.05 else (0, 255, 0)
    for i in range(4):
        cv2.line(prediction, pts[i], pts[(i+1)%4], bbox_color, 2)
        
    prediction_widget.value = bgr8_to_jpeg(prediction)
        
    # 5. Ghi log
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0
    event_str = "Lệch lane / Fallback" if is_reversing else ("Né vật cản" if bbox_area > 0.05 else "Bình thường")
    decision_str = "Lùi" if is_reversing else ("Rẽ trái" if avoidance_steering < 0 else "Rẽ phải" if avoidance_steering > 0 else "Đi thẳng")
    
    log_data.append({
        'timestamp': datetime.datetime.now().isoformat(),
        'fps': round(fps, 2),
        'detected_object/sign': 'obstacle' if bbox_area > 0.05 else 'apex',
        'confidence': 1.0,
        'decision': decision_str,
        'latency_ms': round(latency_ms, 2),
        'control_output': f"steering:{final_steering:.2f},throttle:{car.throttle:.2f}",
        'event': event_str
    })
    
    # Save log ra file sau mỗi 50 frames để tránh mất data
    if len(log_data) % 50 == 0:
        pd.DataFrame(log_data).to_csv('run_log.csv', index=False)

# Bắt đầu vòng lặp Live
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

with open('road_followingv2.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
