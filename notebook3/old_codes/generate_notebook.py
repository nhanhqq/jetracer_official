import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

nb.cells.append(new_markdown_cell("# JetRacer Final - Lane & Obstacle Avoidance\n\nSử dụng thuật toán nhận diện vạch đường mới nhất (1D Spatial Clustering) kết hợp né vật cản."))

nb.cells.append(new_code_cell("""\
import os
import time
import cv2
import numpy as np
import ipywidgets
import threading
import csv
from datetime import datetime
from IPython.display import display
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
from basic_motion import JetRacerController
import importlib
import lane_detection_v2

# Reload on each notebook initialisation so the latest lane logic is used even
# when the Jupyter kernel has already imported an older version.
lane_detection_v2 = importlib.reload(lane_detection_v2)
get_detector = lane_detection_v2.get_detector
print('Lane detector loaded: nearest-corridor lane tracking')

# Restart NVArgus Daemon
os.system('echo "jetson" | sudo -S systemctl restart nvargus-daemon')
time.sleep(2)

try:
    if 'camera' in globals():
        camera.running = False
        camera.unobserve_all()
except:
    pass

camera = CSICamera(width=224, height=224, capture_fps=15)
car = JetRacerController()
detector = get_detector(224, 224)
"""))

nb.cells.append(new_code_cell("""\
# Setup CSV Logging
log_filename = f"jetracer_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(log_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['timestamp', 'fps', 'detected_object', 'confidence', 'decision', 'latency_ms', 'control_output', 'event'])

# Setup UI Widgets
state_widget = ipywidgets.ToggleButtons(options=['stop', 'live'], description='State', value='stop')
raw_widget = ipywidgets.Image(format='jpeg', width=224, height=224)
debug_widget = ipywidgets.Image(format='jpeg', width=224, height=224)

raw_widget.value = bgr8_to_jpeg(np.zeros((224, 224, 3), dtype=np.uint8))
debug_widget.value = bgr8_to_jpeg(np.zeros((224, 224, 3), dtype=np.uint8))

steering_gain_slider = ipywidgets.FloatSlider(description='Steering Gain', min=0.0, max=2.0, value=1.0, step=0.05)
throttle_slider = ipywidgets.FloatSlider(description='Throttle', min=0.0, max=0.5, value=0.15, step=0.01)

ui_widget = ipywidgets.VBox([
    ipywidgets.HBox([raw_widget, debug_widget]),
    ipywidgets.HBox([state_widget]),
    ipywidgets.HBox([steering_gain_slider, throttle_slider])
])

display(ui_widget)
"""))

nb.cells.append(new_code_cell("""\
import time

frame_count = 0
last_time = time.time()
fps = 0.0

def live_update(change):
    global frame_count, last_time, fps
    if state_widget.value != 'live':
        return
        
    start_time = time.time()
    img = change['new']
    
    # Process Frame
    debug_img, raw_steering, info = detector.process_frame(img, draw_debug=True)
    
    # Calculate Latency
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)
    
    # FPS Calculation
    frame_count += 1
    if end_time - last_time >= 1.0:
        fps = frame_count / (end_time - last_time)
        frame_count = 0
        last_time = end_time
        
    # Control Logic
    steering = raw_steering * steering_gain_slider.value
    steering = max(min(steering, 1.0), -1.0)
    throttle = throttle_slider.value
    
    # Fail safe: stop instead of continuing with a stale steering command when
    # both boundaries have been missing for several consecutive frames.
    lane_confident = info.get('lane_confident', False)
    if lane_confident:
        car.set_steering(steering)
        car.set_throttle(throttle)
    else:
        steering = 0.0
        throttle = 0.0
        car.stop()
    
    # Display Update
    raw_widget.value = bgr8_to_jpeg(img)
    debug_widget.value = bgr8_to_jpeg(debug_img)
    
    # Log to CSV
    obstacle = info.get('obstacle')
    detected_obj = 'Obstacle' if obstacle else 'Lane'
    decision = ('Lane Lost - Stop' if not lane_confident else
                ('Avoid Obstacle' if obstacle else 'Follow Lane'))
    control_out = f"S:{steering:.2f} T:{throttle:.2f}"
    event = info.get('case', '')
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    with open(log_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, f"{fps:.1f}", detected_obj, "1.0", decision, latency_ms, control_out, event])

def state_changed(change):
    if change['new'] == 'stop':
        car.stop()

state_widget.observe(state_changed, names='value')
"""))

nb.cells.append(new_code_cell("""\
camera.observe(live_update, names='value')
camera.running = True
"""))

with open('final_racer_v2.ipynb', 'w') as f:
    nbformat.write(nb, f)
print("Notebook created successfully.")
