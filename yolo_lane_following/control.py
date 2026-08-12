from dataclasses import dataclass

import numpy as np

from .geometry import LaneEstimate


@dataclass
class ControlCommand:
    steering: float
    throttle: float
    state: str


def is_motor_command_state(state: str) -> bool:
    """Return whether a controller state is allowed to reach the motor adapter."""
    return (state in ("follow", "avoid", "reacquire:road") or
            state.startswith("avoid:") or
            state.startswith("reverse:white") or
            state.startswith("neutral:white") or
            state.startswith("reverse:obstacle") or
            state.startswith("neutral:obstacle"))


class AdaptiveController:
    """PID steering plus confidence/curve/obstacle-aware longitudinal control."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_throttle_cfg = {
            key: float(cfg[key]) for key in (
                "throttle_min", "throttle_cruise", "throttle_max",
                "throttle_step_up", "obstacle_avoid_throttle",
                "recovery_throttle", "lane_reacquire_throttle",
                "reverse_throttle",
            ) if key in cfg
        }
        self.integral = 0.0
        self.last_error = 0.0
        self.last_steering = 0.0
        self.last_throttle = 0.0
        self.lost_frames = 0
        self.has_lane_lock = False

    def set_throttle_limit(self, requested: float) -> float:
        """Apply a live speed cap without allowing fixed manoeuvre speeds past it."""
        base = self.base_throttle_cfg
        base_max = max(1e-6, base.get("throttle_max", 1.0))
        hard_min = float(self.cfg.get("throttle_limit_min", 0.08))
        hard_max = float(self.cfg.get("throttle_limit_max", 0.60))
        limit = float(np.clip(requested, hard_min, hard_max))
        ratio = limit / base_max

        self.cfg["throttle_max"] = limit
        for key in ("throttle_min", "throttle_cruise", "obstacle_avoid_throttle",
                    "recovery_throttle", "lane_reacquire_throttle"):
            if key in base:
                self.cfg[key] = min(limit, base[key] * ratio)
        if "throttle_step_up" in base:
            self.cfg["throttle_step_up"] = base["throttle_step_up"] * max(1.0, ratio)
        if "reverse_throttle" in base:
            # Raising forward speed must not make reverse more aggressive.
            self.cfg["reverse_throttle"] = min(limit, base["reverse_throttle"])
        return limit

    @staticmethod
    def _approach(current: float, target: float, up: float, down: float) -> float:
        step = up if target > current else down
        return float(current + np.clip(target - current, -step, step))

    def update(self, lane: LaneEstimate, obstacle: float, width: int, dt: float,
               forbidden_left: float = 0.0, forbidden_right: float = 0.0,
               escape_steering: float = 0.0,
               forbidden_front: float = 0.0) -> ControlCommand:
        c = self.cfg
        elapsed = float(np.clip(dt, 1e-3, 0.5))
        # Keep PID derivatives bounded at low inference FPS, but let latched
        # manoeuvres use wall-clock time so a configured 2 s remains 2 s.
        dt = min(elapsed, 0.2)
        reverse_time = float(c.get("reverse_time", 0.35))
        neutral_time = float(c.get("reverse_neutral_time", 0.12))
        white_threshold = float(c.get("white_side_threshold", 0.12))
        white_margin = float(c.get("white_side_margin", 0.04))
        white_front_threshold = float(c.get("white_front_reverse_threshold", 0.58))
        white_front_required = int(c.get("white_front_reverse_frames", 2))
        obstacle_trigger = float(c.get("obstacle_slow_ratio", 0.58))

        # Safety manoeuvres are latched.  Without a latch, a changing mask can
        # alternate between forward and stop/steer commands every frame.
        if not hasattr(self, "maneuver"):
            self.maneuver = None
            self.maneuver_time = 0.0
            self.maneuver_steering = 0.0
            self.maneuver_frames = 0
            self.maneuver_preempted = False
            self.white_cooldown_time = 0.0
            self.white_front_frames = 0
        self.white_cooldown_time = max(0.0, self.white_cooldown_time - elapsed)
        white_left = forbidden_left >= white_threshold and forbidden_left > forbidden_right + white_margin
        white_right = forbidden_right >= white_threshold and forbidden_right > forbidden_left + white_margin
        obstacle_active = obstacle >= obstacle_trigger
        if forbidden_front >= white_front_threshold:
            self.white_front_frames += 1
        else:
            self.white_front_frames = 0
        white_front_blocked = self.white_front_frames >= white_front_required

        # Obstacle has priority over shoulder recovery. Live telemetry showed
        # long white-mask latches suppressing avoidance despite high YOLO risk.
        if obstacle_active and (self.maneuver is None or not self.maneuver.startswith("obstacle_")):
            self.maneuver_preempted = self.maneuver is not None
            blocked = not lane.valid or lane.source == "blocked"
            moving_forward = self.last_throttle > 0.0
            if blocked:
                self.maneuver = "obstacle_neutral" if moving_forward else "obstacle_reverse"
                self.maneuver_time = neutral_time if moving_forward else reverse_time
            elif self.last_throttle < 0.0:
                self.maneuver = "obstacle_reverse"
                self.maneuver_time = reverse_time
            else:
                self.maneuver = "obstacle_avoid"
                self.maneuver_time = float(c.get("obstacle_avoid_time", 2.0))
            self.maneuver_frames = 0
            self.maneuver_steering = float(np.sign(lane.target_x - width / 2.0))
            if escape_steering != 0.0:
                self.maneuver_steering = float(np.sign(escape_steering))
            if self.maneuver_steering == 0.0:
                self.maneuver_steering = -1.0 if forbidden_left > forbidden_right else 1.0
        elif self.maneuver is None:
            if self.white_cooldown_time <= 0.0 and white_front_blocked:
                moving_forward = self.last_throttle > 0.0
                self.maneuver = "white_neutral" if moving_forward else "white_reverse"
                self.maneuver_time = neutral_time if moving_forward else reverse_time * 0.75
                self.maneuver_frames = 0
                self.maneuver_steering = (float(np.sign(escape_steering))
                                          if escape_steering != 0.0
                                          else (1.0 if forbidden_left > forbidden_right else -1.0))
            elif self.white_cooldown_time <= 0.0 and (white_left or white_right):
                # A white shoulder is not a reason to reverse. Keep moving and
                # steer away from it toward the divider/remaining road.
                self.maneuver = "white_correct"
                self.maneuver_time = float(c.get("white_side_correct_time", 0.22))
                self.maneuver_frames = 0
                self.maneuver_steering = 1.0 if white_left else -1.0

        if self.maneuver is not None:
            if self.maneuver_frames > 0:
                self.maneuver_time -= elapsed
            self.maneuver_frames += 1
            maneuver_gain = (c.get("obstacle_avoid_steering", 0.72)
                             if self.maneuver.startswith("obstacle_")
                             else c.get("recovery_steering", 0.42))
            maneuver_step = (c.get("obstacle_preempt_steering_step", 0.72)
                             if self.maneuver_preempted
                             else c.get("maneuver_steering_step", c["max_steering_step"]))
            steer = self._approach(self.last_steering,
                                   self.maneuver_steering * maneuver_gain,
                                   maneuver_step, maneuver_step)
            if self.maneuver.endswith("_neutral"):
                # This is deliberately a real 0-throttle motor command, not a
                # skipped frame.  The next command can therefore engage reverse.
                self.last_steering = steer
                self.last_throttle = 0.0
                if self.maneuver_time <= 0.0:
                    prefix = self.maneuver.split("_")[0]
                    self.maneuver = prefix + "_reverse"
                    self.maneuver_time = reverse_time * (0.75 if prefix == "white" else 1.0)
                    self.maneuver_frames = 0
                self.maneuver_preempted = False
                return ControlCommand(steer, 0.0, "neutral:" + self.maneuver.split("_")[0])
            if self.maneuver_time > 0.0 and self.maneuver.endswith("_reverse"):
                # Reverse first; never convert a safety event into a stationary
                # command.  The motor adapter accepts negative throttle.
                self.last_steering = steer
                self.last_throttle = -abs(float(c.get("reverse_throttle", 0.11)))
                self.maneuver_preempted = False
                return ControlCommand(steer, self.last_throttle, "reverse:" + self.maneuver.split("_")[0])
            if self.maneuver == "white_reverse":
                # Give the car a short forward correction after backing away
                # from the white shoulder.  Keep steering opposite the white.
                self.maneuver = "white_correct"
                self.maneuver_time = float(c.get("white_correct_time", 0.30))
                self.maneuver_frames = 0
            elif self.maneuver == "obstacle_reverse":
                self.maneuver = "obstacle_avoid"
                self.maneuver_time = float(c.get("obstacle_avoid_time", 2.0))
                self.maneuver_frames = 0
            elif self.maneuver == "white_correct":
                if self.maneuver_time <= 1e-6:
                    self.maneuver = None
                    self.white_cooldown_time = float(c.get("white_recovery_cooldown", 0.40))
            elif self.maneuver == "obstacle_avoid":
                if (self.maneuver_time <= 1e-6 and
                        obstacle < obstacle_trigger * 0.70 and lane.source == "divider"):
                    self.maneuver = None
                elif self.maneuver_time <= 1e-6:
                    # Divider is still blocked: remain in the safe side for a
                    # short retry interval instead of cutting back across it.
                    self.maneuver_time = float(c.get("avoid_retry_time", 0.25))
                    self.maneuver_frames = 0
            if self.maneuver is not None:
                steer = self._approach(self.last_steering,
                                       self.maneuver_steering * maneuver_gain,
                                       maneuver_step, maneuver_step)
                throttle = float(c.get(
                    "obstacle_avoid_throttle" if self.maneuver == "obstacle_avoid" else "recovery_throttle",
                    c["throttle_min"],
                ))
                self.last_steering, self.last_throttle = steer, throttle
                self.maneuver_preempted = False
                return ControlCommand(steer, throttle, "avoid:" + self.maneuver.split("_")[0])
            self.maneuver_preempted = False

        if not lane.valid:
            self.lost_frames += 1
        else:
            self.lost_frames = 0
            self.has_lane_lock = True

        if self.lost_frames > int(c["max_lost_frames"]):
            self.integral = 0.0
            if self.has_lane_lock and escape_steering != 0.0:
                target = float(np.sign(escape_steering)) * float(c.get("lane_reacquire_steering", 0.58))
                self.last_steering = self._approach(
                    self.last_steering, target,
                    c.get("maneuver_steering_step", c["max_steering_step"]),
                    c.get("maneuver_steering_step", c["max_steering_step"]),
                )
                self.last_throttle = float(c.get("lane_reacquire_throttle", 0.10))
                return ControlCommand(self.last_steering, self.last_throttle, "reacquire:road")
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
