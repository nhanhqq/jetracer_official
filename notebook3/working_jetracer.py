import time
import board
import busio
import traitlets
from traitlets.config.configurable import Configurable
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

class WorkingJetRacer(Configurable):
    steering_gain = traitlets.Float(default_value=-0.65).tag(config=True)
    steering_offset = traitlets.Float(default_value=0.0).tag(config=True)
    steering = traitlets.Float(default_value=0.0).tag(config=True)
    throttle_gain = traitlets.Float(default_value=0.3).tag(config=True)
    throttle = traitlets.Float(default_value=0.0).tag(config=True)

    def __init__(self, i2c_address=0x40, steering_channel=0, throttle_channel=1, *args, **kwargs):
        super(WorkingJetRacer, self).__init__(*args, **kwargs)
        # Khởi tạo bus I2C và mạch PCA9685 duy nhất tại 0x40
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c, address=i2c_address)
        self.pca.frequency = 50  # Tần số 50Hz chuẩn cho Servo và ESC
        
        # Gán kênh lái và kênh ga
        self.steering_servo = servo.Servo(self.pca.channels[steering_channel])
        self.throttle_servo = servo.ContinuousServo(self.pca.channels[throttle_channel])
        
        self.stop()

    @traitlets.observe('steering')
    def _on_steering(self, change):
        self.set_steering(change['new'])

    @traitlets.observe('throttle')
    def _on_throttle(self, change):
        self.set_throttle(change['new'])

    def set_steering(self, value):
        """
        value từ -1.0 (trái) đến 1.0 (phải)
        Mặc định góc quay Servo từ 0 đến 180 độ (90 là đi thẳng)
        """
        value = value * self.steering_gain + self.steering_offset
        # Quy đổi từ [-1, 1] sang góc [0, 180]
        angle = 90 + (value * 45)  # Giới hạn góc bẻ trong khoảng 45 đến 135 độ
        angle = max(0, min(180, angle))
        self.steering_servo.angle = angle

    def set_throttle(self, value):
        """
        value từ -1.0 đến 1.0
        """
        value = max(-1.0, min(1.0, value)) * self.throttle_gain
        # ContinuousServo tự động nhận giá trị từ -1.0 đến 1.0 và xuất ra xung chuẩn
        self.throttle_servo.throttle = value

    def stop(self):
        self.steering = 0.0
        self.throttle = 0.0
