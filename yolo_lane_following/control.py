from dataclasses import dataclass

import numpy as np

from .geometry import LaneEstimate


@dataclass
class ControlCommand:
    steering: float
    throttle: float
    state: str


class AdaptiveController:
    """PID steering plus confidence/curve/obstacle-aware longitudinal control."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.integral = 0.0
        self.last_error = 0.0
        self.last_steering = 0.0
        self.last_throttle = 0.0
        self.lost_frames = 0
        self.has_lane_lock = False

    @staticmethod
    def _approach(current: float, target: float, up: float, down: float) -> float:
        step = up if target > current else down
        return float(current + np.clip(target - current, -step, step))

    def update(self, lane: LaneEstimate, obstacle: float, width: int, dt: float) -> ControlCommand:
        c = self.cfg
        dt = float(np.clip(dt, 1e-3, 0.2))
        if not lane.valid:
            self.lost_frames += 1
        else:
            self.lost_frames = 0
            self.has_lane_lock = True

        if self.lost_frames > int(c["max_lost_frames"]) or obstacle >= c["emergency_obstacle_ratio"]:
            self.integral = 0.0
            self.last_throttle = 0.0
            self.last_steering = self._approach(self.last_steering, 0.0, c["max_steering_step"], c["max_steering_step"])
            return ControlCommand(self.last_steering, 0.0, "stop:obstacle" if obstacle >= c["emergency_obstacle_ratio"] else "stop:lane_lost")

        # Never start moving before perception has acquired a lane.  Once locked,
        # a short dropout is debounced while throttle ramps down instead of up.
        if not lane.valid:
            self.last_throttle = self._approach(
                self.last_throttle, 0.0, c["throttle_step_up"], c["throttle_step_down"]
            )
            state = "wait:lane_lock" if not self.has_lane_lock else "slow:lane_dropout"
            return ControlCommand(self.last_steering, self.last_throttle, state)

        error = (lane.target_x - width / 2.0) / max(1.0, width / 2.0)
        self.integral = float(np.clip(self.integral + error * dt, -0.5, 0.5))
        derivative = (error - self.last_error) / dt
        raw = c["kp"] * error + c["ki"] * self.integral + c["kd"] * derivative + c["heading_gain"] * lane.heading_error
        raw = float(np.clip(raw, -c["max_steering"], c["max_steering"]))
        steering = self._approach(self.last_steering, raw, c["max_steering_step"], c["max_steering_step"])

        curve_load = max(abs(steering), lane.curvature)
        speed_scale = max(0.28, 1.0 - c["curve_slowdown"] * curve_load)
        confidence_scale = c["low_confidence_slowdown"] + (1.0 - c["low_confidence_slowdown"]) * lane.confidence
        obstacle_scale = max(0.25, 1.0 - obstacle) if obstacle >= c["obstacle_slow_ratio"] else 1.0
        if lane.source == "avoid":
            obstacle_scale = min(obstacle_scale, c["avoidance_speed_scale"])
        target_throttle = float(np.clip(c["throttle_cruise"] * speed_scale * confidence_scale * obstacle_scale,
                                        c["throttle_min"], c["throttle_max"]))
        throttle = self._approach(self.last_throttle, target_throttle, c["throttle_step_up"], c["throttle_step_down"])
        self.last_error, self.last_steering, self.last_throttle = error, steering, throttle
        state = "avoid" if lane.source == "avoid" else ("slow:obstacle" if obstacle >= c["obstacle_slow_ratio"] else "follow")
        return ControlCommand(steering, throttle, state)
