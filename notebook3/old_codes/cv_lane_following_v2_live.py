#!/usr/bin/env python3
"""
cv_lane_following_v2_live.py — Lane Following V2 Live Control cho JetRacer
==========================================================================

Script điều khiển xe JetRacer live sử dụng thuật toán lane detection v2 nâng cao.
Cấu trúc giống các file follow lane cũ (cv_road_following_live.ipynb) để dễ tùy chỉnh.

Tính năng:
- Phát hiện 2 lề đường (cam/đỏ/vàng/trắng)
- Phát hiện dải phân cách đứt đoạn ở giữa
- Phát hiện và tránh vật cản
- Chống bóng, lóe sáng, ánh sáng chập chờn (CLAHE)
- Smoothing steering để xe chạy mượt
- FPS logging cho cuộc thi

Sử dụng (Jupyter Notebook):
    Copy nội dung các cell vào notebook mới hoặc chạy trực tiếp file này.

Sử dụng (Terminal):
    python3 cv_lane_following_v2_live.py
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
from datetime import datetime

# Thêm thư mục project vào path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from lane_detection_v2 import LaneDetector
from basic_motion import JetRacerController


# =============================================================================
# CONFIGURATION — Điều chỉnh các tham số tại đây
# =============================================================================

# Steering
STEERING_GAIN = 1.0       # Hệ số nhân steering (tăng nếu xe cua thiếu)
STEERING_BIAS = 0.0       # Offset steering (dương = lệch phải, âm = lệch trái)
MAX_STEERING = 1.0        # Giới hạn steering tối đa

# Throttle
DEFAULT_THROTTLE = 0.15   # Tốc độ mặc định
SLOW_THROTTLE = 0.12      # Tốc độ khi phát hiện vật cản
TURN_THROTTLE = 0.12      # Tốc độ khi cua gấp

# Obstacle avoidance
OBSTACLE_SLOW_THRESHOLD = 0.3  # Giảm tốc khi steering > threshold

# Lane detector tuning
ROI_TOP_RATIO = 0.35      # Vùng ROI (0.35 = lấy 65% dưới ảnh)

# Logging
ENABLE_LOG = True         # Bật/tắt logging
LOG_FILE = os.path.join(current_dir, f'race_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')


# =============================================================================
# LOGGING
# =============================================================================

class RaceLogger:
    """Logger cho cuộc thi - ghi FPS, steering, throttle, events."""

    def __init__(self, filepath, enabled=True):
        self.enabled = enabled
        self.filepath = filepath
        self.frame_count = 0
        self.start_time = None
        self.fps_values = []

        if self.enabled:
            with open(filepath, 'w') as f:
                f.write("timestamp,fps,steering,throttle,left_detected,"
                        "right_detected,center_detected,obstacle_detected,"
                        "decision,latency_ms,event\n")

    def log(self, fps, steering, throttle, info, decision, latency_ms, event=""):
        if not self.enabled:
            return
        self.frame_count += 1
        self.fps_values.append(fps)

        timestamp = datetime.now().isoformat()
        left = 1 if info.get('left_x') is not None else 0
        right = 1 if info.get('right_x') is not None else 0
        center = 1 if info.get('center_x') is not None else 0
        obstacle = 1 if info.get('obstacle') is not None else 0

        try:
            with open(self.filepath, 'a') as f:
                f.write(f"{timestamp},{fps:.1f},{steering:.3f},{throttle:.3f},"
                        f"{left},{right},{center},{obstacle},"
                        f"{decision},{latency_ms:.1f},{event}\n")
        except Exception:
            pass

    def get_avg_fps(self):
        if not self.fps_values:
            return 0
        return np.mean(self.fps_values)


# =============================================================================
# MAIN CONTROLLER (Terminal mode)
# =============================================================================

def run_terminal_mode():
    """
    Chế độ chạy từ Terminal (không cần Jupyter).
    Sử dụng OpenCV window để hiển thị và keyboard để điều khiển.
    """
    print("=" * 60)
    print("  JETRACER LANE FOLLOWING V2 — TERMINAL MODE")
    print("=" * 60)

    # --- Restart camera daemon ---
    print("\n[1/4] Khởi động lại camera daemon...")
    os.system('echo "jetson" | sudo -S systemctl restart nvargus-daemon')
    time.sleep(2)

    # --- Initialize Camera ---
    print("[2/4] Khởi tạo CSI Camera...")
    try:
        from jetcam.csi_camera import CSICamera
        camera = CSICamera(width=224, height=224, capture_fps=0)
        camera.running = True
        time.sleep(1)
        print("  Camera đã sẵn sàng!")
    except Exception as e:
        print(f"  LỖI Camera: {e}")
        print("  Thử chạy ở chế độ test với dataset images...")
        run_test_mode()
        return

    # --- Initialize Car ---
    print("[3/4] Khởi tạo bộ điều khiển xe...")
    try:
        car = JetRacerController()
        print("  Xe đã sẵn sàng!")
    except Exception as e:
        print(f"  LỖI Xe: {e}")
        return

    # --- Initialize Lane Detector ---
    print("[4/4] Khởi tạo Lane Detector V2...")
    detector = LaneDetector(224, 224)
    detector.roi_top_ratio = ROI_TOP_RATIO
    logger = RaceLogger(LOG_FILE, ENABLE_LOG)

    print("\n" + "=" * 60)
    print("  SẴN SÀNG! Nhấn SPACE để bắt đầu, Q để thoát")
    print("=" * 60)
    print("  [SPACE] Start/Stop")
    print("  [+/=]   Tăng ga")
    print("  [-/_]   Giảm ga")
    print("  [Q]     Thoát")
    print("=" * 60)

    running = False
    throttle = DEFAULT_THROTTLE
    frame_count = 0
    fps_timer = time.time()

    try:
        while True:
            # Read frame
            frame_start = time.time()
            image = camera.read()
            if image is None:
                continue

            # Process with lane detection
            result_img, steering, info = detector.process_frame(image, draw_debug=True)

            # Apply steering gain and bias
            final_steering = steering * STEERING_GAIN + STEERING_BIAS
            final_steering = max(-MAX_STEERING, min(MAX_STEERING, final_steering))

            # Determine throttle
            decision = "straight"
            current_throttle = throttle

            if info['obstacle'] is not None:
                current_throttle = SLOW_THROTTLE
                decision = "obstacle_avoid"
            elif abs(final_steering) > OBSTACLE_SLOW_THRESHOLD:
                current_throttle = TURN_THROTTLE
                decision = "turn"

            # Apply controls if running
            if running:
                car.set_steering(final_steering)
                car.set_throttle(current_throttle)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed > 0:
                fps = frame_count / elapsed
            else:
                fps = 0

            # Reset FPS counter every 2 seconds
            if elapsed > 2.0:
                frame_count = 0
                fps_timer = time.time()

            # Latency
            latency_ms = (time.time() - frame_start) * 1000

            # Log
            logger.log(fps, final_steering, current_throttle, info,
                       decision, latency_ms)

            # Add FPS and status to display
            status = "RUNNING" if running else "STOPPED"
            cv2.putText(result_img, f"FPS:{fps:.0f} {status}", (5, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cv2.putText(result_img, f"T:{current_throttle:.2f}", (160, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

            # Show frame
            cv2.imshow("JetRacer Lane V2", result_img)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord(' '):  # SPACE
                running = not running
                if not running:
                    car.stop()
                    logger.log(fps, 0, 0, info, "stop", 0, "manual_stop")
                else:
                    logger.log(fps, 0, 0, info, "start", 0, "manual_start")
                print(f"\n{'▶ STARTED' if running else '■ STOPPED'}")
            elif key in [ord('+'), ord('=')]:
                throttle = min(1.0, throttle + 0.01)
                print(f"  Throttle: {throttle:.2f}")
            elif key in [ord('-'), ord('_')]:
                throttle = max(0.0, throttle - 0.01)
                print(f"  Throttle: {throttle:.2f}")

    except KeyboardInterrupt:
        print("\n\nĐã ngắt chương trình!")
    finally:
        car.stop()
        camera.running = False
        cv2.destroyAllWindows()

        if ENABLE_LOG:
            print(f"\nFPS trung bình: {logger.get_avg_fps():.1f}")
            print(f"Log đã lưu: {LOG_FILE}")

        print("Đã dừng xe an toàn.")


# =============================================================================
# JUPYTER NOTEBOOK MODE — Copy từng cell vào notebook
# =============================================================================
"""
=== CELL 1: Khởi tạo Camera và Xe ===
"""


def jupyter_init():
    """
    Cell 1: Khởi tạo Camera, Xe, và Lane Detector.
    Copy code trong hàm này vào cell đầu tiên của notebook.
    """
    code = '''
import os
import time
import cv2
import numpy as np
import ipywidgets
import threading
from IPython.display import display
from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
from basic_motion import JetRacerController
from lane_detection_v2 import LaneDetector

# Khởi động lại camera daemon
os.system('echo "jetson" | sudo -S systemctl restart nvargus-daemon')
time.sleep(2)

try:
    if 'camera' in globals():
        camera.running = False
        camera.unobserve_all()
except:
    pass

camera = CSICamera(width=224, height=224, capture_fps=0)

# Khởi tạo bộ điều khiển xe
car = JetRacerController()

# Khởi tạo Lane Detector V2
detector = LaneDetector(224, 224)
'''
    return code


def jupyter_ui():
    """
    Cell 2: Tạo giao diện Live View.
    """
    code = '''
state_widget = ipywidgets.ToggleButtons(
    options=['stop', 'live'],
    description='Trạng thái',
    value='stop'
)
prediction_widget = ipywidgets.Image(
    format='jpeg', width=camera.width, height=camera.height
)
prediction_widget.value = bgr8_to_jpeg(np.zeros((224, 224, 3), dtype=np.uint8))

steering_gain_slider = ipywidgets.FloatSlider(
    description='Steering Gain', min=0.0, max=3.0,
    layout=ipywidgets.Layout(width='500px'),
    value=1.0, step=0.05, orientation='horizontal'
)
throttle_slider = ipywidgets.FloatSlider(
    description='Throttle', min=0.0, max=0.5,
    layout=ipywidgets.Layout(width='500px'),
    value=0.15, step=0.01, orientation='horizontal'
)
obstacle_offset_slider = ipywidgets.IntSlider(
    description='Obs Offset', min=20, max=100,
    layout=ipywidgets.Layout(width='500px'),
    value=60, step=5, orientation='horizontal'
)

# FPS display
fps_label = ipywidgets.Label(value="FPS: --")

ui_widget = ipywidgets.VBox([
    prediction_widget,
    ipywidgets.HBox([state_widget]),
    ipywidgets.HBox([steering_gain_slider, throttle_slider]),
    ipywidgets.HBox([obstacle_offset_slider, fps_label])
])

display(ui_widget)
'''
    return code


def jupyter_algorithm():
    """
    Cell 3: Thuật toán xử lý và điều khiển.
    """
    code = '''
import time as _time
_frame_times = []

def live_update(change):
    global _frame_times

    if state_widget.value != 'live':
        return

    frame_start = _time.time()
    img = change['new']

    # Update detector params from UI
    detector.obstacle_avoidance_offset = obstacle_offset_slider.value

    # Process frame
    processed_img, steering, info = detector.process_frame(img.copy(), draw_debug=True)

    # Apply steering gain
    final_steering = steering * steering_gain_slider.value
    final_steering = max(-1.0, min(1.0, final_steering))

    # Determine throttle
    current_throttle = throttle_slider.value
    if info['obstacle'] is not None:
        current_throttle *= 0.7  # Giảm tốc khi gặp vật cản
    elif abs(final_steering) > 0.3:
        current_throttle *= 0.85  # Giảm nhẹ khi cua

    # Apply controls
    car.set_steering(final_steering)
    car.set_throttle(current_throttle)

    # Calculate FPS
    frame_time = _time.time() - frame_start
    _frame_times.append(frame_time)
    if len(_frame_times) > 30:
        _frame_times = _frame_times[-30:]
    avg_fps = 1.0 / (sum(_frame_times) / len(_frame_times)) if _frame_times else 0

    # Update FPS display
    fps_label.value = f"FPS: {avg_fps:.1f} | Steer: {final_steering:.2f}"

    # Add FPS text to image
    cv2.putText(processed_img, f"FPS:{avg_fps:.0f}", (5, 220),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    prediction_widget.value = bgr8_to_jpeg(processed_img)

def state_changed(change):
    if change['new'] == 'stop':
        car.stop()

state_widget.observe(state_changed, names='value')
'''
    return code


def jupyter_start():
    """
    Cell 4: Bắt đầu camera.
    """
    code = '''
camera.observe(live_update, names='value')
camera.running = True
'''
    return code


def jupyter_stop():
    """
    Cell 5: Dừng camera an toàn.
    """
    code = '''
camera.running = False
camera.unobserve_all()
car.stop()
print("Đã dừng xe và camera an toàn.")
'''
    return code


# =============================================================================
# TEST MODE — Chạy thuật toán trên dataset images (không cần xe/camera)
# =============================================================================

def run_test_mode():
    """Chạy test thuật toán trên dataset images."""
    import glob

    print("\n" + "=" * 60)
    print("  TEST MODE — Chạy trên dataset images")
    print("=" * 60)

    dataset_dir = '/home/jetson/jetracer_official/notebook3/road_following_A/apex'
    images = sorted(glob.glob(os.path.join(dataset_dir, '*.jpg')))

    if not images:
        print("ERROR: Không tìm thấy ảnh trong dataset!")
        return

    detector = LaneDetector(224, 224)
    print(f"Đã load {len(images)} ảnh. Nhấn bất kỳ phím để chuyển frame, Q để thoát.\n")

    for i, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            continue

        result, steering, info = detector.process_frame(img, draw_debug=True)

        # Add file info
        fname = os.path.basename(img_path)
        cv2.putText(result, f"[{i + 1}/{len(images)}] {fname}", (5, 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)

        cv2.imshow("Lane Detection V2 Test", result)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    print("Test hoàn tất!")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='JetRacer Lane Following V2')
    parser.add_argument('--test', action='store_true',
                       help='Chạy ở chế độ test (không cần xe/camera)')
    parser.add_argument('--jupyter', action='store_true',
                       help='In ra code cells cho Jupyter notebook')
    args = parser.parse_args()

    if args.test:
        run_test_mode()
    elif args.jupyter:
        print("\n=== CELL 1: Khởi tạo ===")
        print(jupyter_init())
        print("\n=== CELL 2: Tạo UI ===")
        print(jupyter_ui())
        print("\n=== CELL 3: Thuật toán ===")
        print(jupyter_algorithm())
        print("\n=== CELL 4: Bắt đầu ===")
        print(jupyter_start())
        print("\n=== CELL 5: Dừng ===")
        print(jupyter_stop())
    else:
        run_terminal_mode()
