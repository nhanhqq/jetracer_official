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

    def update(self, lane: LaneEstimate, obstacle: float, width: int, dt: float,
               forbidden_left: float = 0.0, forbidden_right: float = 0.0,
               escape_steering: float = 0.0) -> ControlCommand:
        c = self.cfg
        dt = float(np.clip(dt, 1e-3, 0.2))
        reverse_time = float(c.get("reverse_time", 0.35))
        neutral_time = float(c.get("reverse_neutral_time", 0.12))
        white_threshold = float(c.get("white_side_threshold", 0.12))
        white_margin = float(c.get("white_side_margin", 0.04))
        obstacle_trigger = float(c.get("obstacle_slow_ratio", 0.58))

        # Safety manoeuvres are latched.  Without a latch, a changing mask can
        # alternate between forward and stop/steer commands every frame.
        if not hasattr(self, "maneuver"):
            self.maneuver = None
            self.maneuver_time = 0.0
            self.maneuver_steering = 0.0
        if self.maneuver is None:
            white_left = forbidden_left >= white_threshold and forbidden_left > forbidden_right + white_margin
            white_right = forbidden_right >= white_threshold and forbidden_right > forbidden_left + white_margin
            if obstacle >= obstacle_trigger:
                # Pass obstacles while moving forward.  Prefer the side with
                # more free road, then keep that direction until the divider
                # corridor is clear again.
                self.maneuver = "obstacle_avoid"
                self.maneuver_time = float(c.get("obstacle_avoid_time", 2.0))
                self.maneuver_steering = float(np.sign(lane.target_x - width / 2.0))
                if escape_steering != 0.0:
                    self.maneuver_steering = float(np.sign(escape_steering))
                if self.maneuver_steering == 0.0:
                    self.maneuver_steering = -1.0 if forbidden_left > forbidden_right else 1.0
            elif white_left or white_right:
                moving_forward = self.last_throttle > 0.0
                self.maneuver = "white_neutral" if moving_forward else "white_reverse"
                self.maneuver_time = neutral_time if moving_forward else reverse_time * 0.75
                # White on the left means move right, and vice versa.
                self.maneuver_steering = 1.0 if white_left else -1.0

        if self.maneuver is not None:
            self.maneuver_time -= dt
            steer = self._approach(self.last_steering,
                                   self.maneuver_steering * c.get("recovery_steering", 0.34),
                                   c["max_steering_step"], c["max_steering_step"])
            if self.maneuver.endswith("_neutral"):
                # This is deliberately a real 0-throttle motor command, not a
                # skipped frame.  The next command can therefore engage reverse.
                self.last_steering = steer
                self.last_throttle = 0.0
                if self.maneuver_time <= 0.0:
                    prefix = self.maneuver.split("_")[0]
                    self.maneuver = prefix + "_reverse"
                    self.maneuver_time = reverse_time * (0.75 if prefix == "white" else 1.0)
                return ControlCommand(steer, 0.0, "neutral:" + self.maneuver.split("_")[0])
            if self.maneuver_time > 0.0 and self.maneuver.endswith("_reverse"):
                # Reverse first; never convert a safety event into a stationary
                # command.  The motor adapter accepts negative throttle.
                self.last_steering = steer
                self.last_throttle = -abs(float(c.get("reverse_throttle", 0.11)))
                return ControlCommand(steer, self.last_throttle, "reverse:" + self.maneuver.split("_")[0])
            if self.maneuver == "white_reverse":
                # Give the car a short forward correction after backing away
                # from the white shoulder.  Keep steering opposite the white.
                self.maneuver = "white_correct"
                self.maneuver_time = float(c.get("white_correct_time", 0.30))
            elif self.maneuver == "obstacle_reverse":
                self.maneuver = "obstacle_avoid"
                self.maneuver_time = float(c.get("obstacle_avoid_time", 2.0))
            elif self.maneuver == "white_correct":
                if forbidden_left < white_threshold * 0.65 and forbidden_right < white_threshold * 0.65:
                    self.maneuver = None
                else:
                    self.maneuver_time = float(c.get("white_correct_time", 0.30))
            elif self.maneuver == "obstacle_avoid":
                if (self.maneuver_time <= 1e-6 and
                        obstacle < obstacle_trigger * 0.70 and lane.source == "divider"):
                    self.maneuver = None
                elif self.maneuver_time <= 1e-6:
                    # Divider is still blocked: remain in the safe side for a
                    # short retry interval instead of cutting back across it.
                    self.maneuver_time = float(c.get("avoid_retry_time", 0.25))
            if self.maneuver is not None:
                steer = self._approach(self.last_steering,
                                       self.maneuver_steering * c.get("recovery_steering", 0.34),
                                       c["max_steering_step"], c["max_steering_step"])
                throttle = float(c.get(
                    "obstacle_avoid_throttle" if self.maneuver == "obstacle_avoid" else "recovery_throttle",
                    c["throttle_min"],
                ))
                self.last_steering, self.last_throttle = steer, throttle
                return ControlCommand(steer, throttle, "avoid:" + self.maneuver.split("_")[0])

        if not lane.valid:
            self.lost_frames += 1
        else:
            self.lost_frames = 0
            self.has_lane_lock = True

        if self.lost_frames > int(c["max_lost_frames"]):
            self.integral = 0.0
            self.last_throttle = 0.0
            self.last_steering = self._approach(self.last_steering, 0.0, c["max_steering_step"], c["max_steering_step"])
            return ControlCommand(self.last_steering, 0.0, "stop:lane_lost")

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
