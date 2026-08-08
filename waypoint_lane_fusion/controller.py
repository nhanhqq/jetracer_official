import math
import numpy as np
from .types import DriveCommand, DriveState, Waypoint


class WaypointFilter:
    def __init__(self, alpha=0.30):
        self.alpha, self.value = float(alpha), None

    def update(self, point):
        if self.value is None: self.value = point
        else:
            a, old = self.alpha, self.value
            self.value = Waypoint(a*point.x+(1-a)*old.x, a*point.y+(1-a)*old.y,
                                  a*point.confidence+(1-a)*old.confidence)
        return self.value


class SteeringController:
    def __init__(self, cfg):
        self.c, self.last_error, self.last = cfg, 0.0, 0.0

    def update(self, point, dt, bias=0.0):
        dt = float(np.clip(dt, .005, .2))
        error = 2.0 * (point.x - .5) + bias
        # Heading from bottom-centre to waypoint, normalized to roughly [-1, 1].
        heading = math.atan2(point.x-.5, max(.05, 1.0-point.y)) / (math.pi/2)
        raw = self.c["kp"]*error + self.c["kd"]*(error-self.last_error)/dt + self.c["heading_gain"]*heading
        raw = float(np.clip(raw, -self.c["max_steering"], self.c["max_steering"]))
        lowpass = self.c["steering_alpha"]*raw + (1-self.c["steering_alpha"])*self.last
        steer = float(self.last + np.clip(lowpass-self.last, -self.c["max_steering_change"], self.c["max_steering_change"]))
        self.last_error, self.last = error, steer
        return raw, steer


class SpeedController:
    def __init__(self, cfg): self.c, self.last = cfg, 0.0

    def update(self, steering, confidence, state):
        if state in (DriveState.LANE_LOST, DriveState.OBSTACLE, DriveState.RED_LIGHT): target = 0.0
        else:
            target = self.c["throttle_max"] - self.c["curve_penalty"]*abs(steering)
            target -= self.c["confidence_penalty"]*(1.0-confidence)
            if state == DriveState.SLOWDOWN: target = min(target, self.c["throttle_min"])
            target = float(np.clip(target, self.c["throttle_min"], self.c["throttle_max"]))
        rate = self.c["accel_rate"] if target > self.last else self.c["brake_rate"]
        self.last = float(self.last + np.clip(target-self.last, -rate, rate))
        return self.last


class DriveController:
    def __init__(self, cfg):
        self.steering, self.speed = SteeringController(cfg), SpeedController(cfg)

    def update(self, point, state, dt, bias=0.0):
        raw, steering = self.steering.update(point, dt, bias)
        if state == DriveState.LANE_LOST:
            self.steering.last = float(self.steering.last + np.clip(-self.steering.last, -.04, .04))
            steering = self.steering.last
        throttle = self.speed.update(steering, point.confidence, state)
        return DriveCommand(raw, steering, throttle, state)
