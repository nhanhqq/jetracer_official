#!/usr/bin/env python3
"""
notebook3/basic_motion.py
Module điều khiển chuyển động cơ bản cho xe NVIDIA JetRacer.
Cung cấp đầy đủ các hàm:
- Bẻ bánh trái (steer_left)
- Bẻ bánh phải (steer_right)
- Trả lái thẳng (center_steering)
- Đi lên / Tiến (move_forward)
- Đi lùi (move_backward)
- Tăng ga (increase_speed)
- Giảm ga (decrease_speed)
- Dừng xe (stop)
- Hiệu chỉnh góc lái và ga (calibrate)
"""

import sys
import os
import time

# Tự động thêm thư mục gốc dự án vào sys.path nếu cần
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jetracer.nvidia_racecar import NvidiaRacecar


class JetRacerController:
    """
    Lớp điều khiển xe JetRacer mở rộng với các hàm chức năng cơ bản.
    """
    def __init__(self, steering_gain=-0.65, steering_offset=0.0,
                 throttle_gain=0.8, max_throttle=0.5, verbose=False):
        self.car = NvidiaRacecar()
        self.car.steering_gain = steering_gain
        self.car.steering_offset = steering_offset
        self.car.throttle_gain = throttle_gain
        self.max_throttle = max_throttle
        self.verbose = bool(verbose)
        
        # Đảm bảo khởi tạo an toàn
        self.stop()
        self.center_steering()

    def set_steering(self, value: float):
        """
        Đặt góc lái trực tiếp [-1.0: Trái tối đa, 1.0: Phải tối đa, 0.0: Thẳng].
        """
        clamped_val = max(-1.0, min(1.0, float(value)))
        self.car.steering = clamped_val
        if self.verbose:
            print(f"[JetRacer] Góc lái (steering): {clamped_val:.2f}")

    def steer_left(self, amount: float = 0.5):
        """
        Bẻ bánh trái với độ bẻ amount (0.0 đến 1.0).
        """
        amount = max(0.0, min(1.0, float(amount)))
        # Lái trái tương ứng steering âm (-amount)
        self.set_steering(-amount)

    def steer_right(self, amount: float = 0.5):
        """
        Bẻ bánh phải với độ bẻ amount (0.0 đến 1.0).
        """
        amount = max(0.0, min(1.0, float(amount)))
        # Lái phải tương ứng steering dương (+amount)
        self.set_steering(amount)

    def center_steering(self):
        """
        Trả bánh xe về vị trí thẳng (0.0).
        """
        self.set_steering(0.0)

    def set_throttle(self, value: float):
        """
        Đặt ga (throttle) [-1.0: Lùi tối đa, 1.0: Tiến tối đa, 0.0: Dừng].
        Áp dụng giới hạn an toàn max_throttle cho chiều tiến/lùi.
        """
        clamped_val = max(-self.max_throttle, min(self.max_throttle, float(value)))
        self.car.throttle = clamped_val
        if self.verbose:
            print(f"[JetRacer] Mức ga (throttle): {clamped_val:.2f}")

    def move_forward(self, speed: float = 0.2, duration: float = None):
        """
        Đi lên (Tiến).
        - speed: tốc độ tiến (0.0 đến max_throttle)
        - duration: thời gian chạy tính bằng giây (nếu None sẽ giữ ga)
        """
        speed = max(0.0, min(self.max_throttle, float(speed)))
        print(f"[JetRacer] Đi lên với tốc độ {speed:.2f}...")
        self.set_throttle(speed)
        
        if duration is not None and duration > 0:
            time.sleep(duration)
            self.stop()

    def move_backward(self, speed: float = 0.2, duration: float = None):
        """
        Đi lùi.
        - speed: tốc độ lùi (0.0 đến max_throttle)
        - duration: thời gian chạy tính bằng giây (nếu None sẽ giữ ga)
        """
        speed = max(0.0, min(self.max_throttle, float(speed)))
        print(f"[JetRacer] Đi lùi với tốc độ -{speed:.2f}...")
        self.set_throttle(-speed)
        
        if duration is not None and duration > 0:
            time.sleep(duration)
            self.stop()

    def increase_speed(self, step: float = 0.05):
        """
        Tăng ga hiện tại lên một khoảng step.
        """
        new_throttle = self.car.throttle + step
        print(f"[JetRacer] Tăng ga (+{step:.2f}): {new_throttle:.2f}")
        self.set_throttle(new_throttle)

    def decrease_speed(self, step: float = 0.05):
        """
        Giảm ga hiện tại một khoảng step.
        """
        new_throttle = self.car.throttle - step
        print(f"[JetRacer] Giảm ga (-{step:.2f}): {new_throttle:.2f}")
        self.set_throttle(new_throttle)

    def stop(self):
        """
        Dừng xe lập tức (throttle = 0.0).
        """
        self.set_throttle(0.0)
        print("[JetRacer] Đã dừng xe!")

    def calibrate_steering(self, gain: float = None, offset: float = None):
        """
        Hiệu chỉnh gain và offset của steering.
        """
        if gain is not None:
            self.car.steering_gain = float(gain)
        if offset is not None:
            self.car.steering_offset = float(offset)
        print(f"[JetRacer] Steering calibration -> gain: {self.car.steering_gain:.2f}, offset: {self.car.steering_offset:.2f}")

    def calibrate_throttle(self, gain: float = None, max_throttle: float = None):
        """
        Hiệu chỉnh gain và max_throttle của động cơ.
        """
        if gain is not None:
            self.car.throttle_gain = float(gain)
        if max_throttle is not None:
            self.max_throttle = float(max_throttle)
        print(f"[JetRacer] Throttle calibration -> gain: {self.car.throttle_gain:.2f}, max_throttle: {self.max_throttle:.2f}")


# -------------------------------------------------------------
# Standalone Global Functions (Hàm độc lập cho thao tác nhanh)
# -------------------------------------------------------------
_controller_instance = None

def get_controller():
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = JetRacerController()
    return _controller_instance

def steer_left(amount=0.5):
    """Bẻ bánh trái"""
    get_controller().steer_left(amount)

def steer_right(amount=0.5):
    """Bẻ bánh phải"""
    get_controller().steer_right(amount)

def center_steering():
    """Trả lái thẳng"""
    get_controller().center_steering()

def move_forward(speed=0.2, duration=None):
    """Đi lên (Tiến)"""
    get_controller().move_forward(speed, duration)

def move_backward(speed=0.2, duration=None):
    """Đi lùi"""
    get_controller().move_backward(speed, duration)

def increase_speed(step=0.05):
    """Tăng ga"""
    get_controller().increase_speed(step)

def decrease_speed(step=0.05):
    """Giảm ga"""
    get_controller().decrease_speed(step)

def stop():
    """Dừng xe"""
    get_controller().stop()


# -------------------------------------------------------------
# Interactive Test / CLI
# -------------------------------------------------------------
def run_interactive_demo():
    print("=" * 50)
    print("      NVIDIA JETRACER - BASIC MOTION DEMO      ")
    print("=" * 50)
    print("Hướng dẫn điều khiển:")
    print("  [W]           : Đi lên (Tiến)")
    print("  [S]           : Đi lùi")
    print("  [A]           : Bẻ bánh trái")
    print("  [D]           : Bẻ bánh phải")
    print("  [C]           : Trả lái thẳng")
    print("  [+]           : Tăng ga (+0.05)")
    print("  [-]           : Giảm ga (-0.05)")
    print("  [SPACE] / [X] : Dừng xe (Stop)")
    print("  [Q]           : Thoát chương trình")
    print("=" * 50)

    controller = get_controller()

    try:
        import tty, termios
        def getch():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
    except Exception:
        def getch():
            return input("Nhập lệnh (W/A/S/D/C/+/ -/X/Q): ")

    while True:
        try:
            cmd = getch().lower()
            if cmd == 'w':
                controller.move_forward(0.2)
            elif cmd == 's':
                controller.move_backward(0.2)
            elif cmd == 'a':
                controller.steer_left(0.6)
            elif cmd == 'd':
                controller.steer_right(0.6)
            elif cmd == 'c':
                controller.center_steering()
            elif cmd in ['+', '=']:
                controller.increase_speed(0.05)
            elif cmd in ['-', '_']:
                controller.decrease_speed(0.05)
            elif cmd in [' ', 'x']:
                controller.stop()
            elif cmd == 'q':
                print("\nThoát chương trình. Đang dừng xe...")
                controller.stop()
                break
        except KeyboardInterrupt:
            print("\nĐã ngắt chương trình! Dừng xe an toàn...")
            controller.stop()
            break

if __name__ == '__main__':
    run_interactive_demo()
