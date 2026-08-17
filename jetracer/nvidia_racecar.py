from .racecar import Racecar
import traitlets
from adafruit_servokit import ServoKit


class NvidiaRacecar(Racecar):
    
    i2c_address = traitlets.Integer(default_value=0x40)
    steering_gain = traitlets.Float(default_value=-0.65)
    steering_offset = traitlets.Float(default_value=0)
    steering_channel = traitlets.Integer(default_value=0)
    throttle_gain = traitlets.Float(default_value=0.8)
    throttle_channel = traitlets.Integer(default_value=1)
    
    def __init__(self, *args, **kwargs):
        super(NvidiaRacecar, self).__init__(*args, **kwargs)
        self.kit = ServoKit(channels=16, address=self.i2c_address)
        self.steering_motor = self.kit.continuous_servo[self.steering_channel]
        self.throttle_motor = self.kit.continuous_servo[self.throttle_channel]
        # Force initial motor positions to neutral
        self._on_steering()
        self._on_throttle()
    
    @traitlets.observe('steering', 'steering_gain', 'steering_offset')
    def _on_steering(self, change=None):
        val = self.steering * self.steering_gain + self.steering_offset
        # Clamp to [-1.0, 1.0] to prevent ServoKit ValueError exception
        self.steering_motor.throttle = max(-1.0, min(1.0, float(val)))
    
    @traitlets.observe('throttle', 'throttle_gain')
    def _on_throttle(self, change=None):
        val = self.throttle * self.throttle_gain
        # Clamp to [-1.0, 1.0] to prevent ServoKit ValueError exception
        self.throttle_motor.throttle = max(-1.0, min(1.0, float(val)))