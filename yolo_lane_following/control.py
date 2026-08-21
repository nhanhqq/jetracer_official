from dataclasses import dataclass

import numpy as np

from .geometry import LaneEstimate


@dataclass
class ControlCommand:
    steering: float
    throttle: float
    state: str
    raw_steering: float = 0.0
    filtered_steering: float = 0.0
    curve_load: float = 0.0


def is_motor_command_state(state: str) -> bool:
    """Return whether a controller state is allowed to reach the motor adapter."""
    return (state in ("follow", "avoid", "reacquire:road") or
            state.startswith("avoid:") or
            state.startswith("reverse:white") or
            state.startswith("neutral:white") or
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
        self.last_raw_steering = 0.0
        self.last_steering = 0.0
        self.last_throttle = 0.0
        self.last_throttle_target = 0.0
        self.filtered_derivative = 0.0
        self.curve_load_ema = 0.0
        self.lost_frames = 0
        # The first valid segmentation is immediately usable.  Do not gate
        # driving behind a confirmation/lock period when entering a new area.
        self.seen_lane = False
        self.steering_reversal_time = 0.0
        self.steering_direction = 0

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

    @staticmethod
    def _approach_rate(current: float, target: float, rate_up: float,
                       rate_down: float, dt: float) -> float:
        """Slew-limit a command in units/second, independent of inference FPS."""
        rate = rate_up if target > current else rate_down
        maximum_delta = max(0.0, float(rate)) * max(float(dt), 1e-3)
        return float(current + np.clip(target - current, -maximum_delta, maximum_delta))

    def _slew(self, current: float, target: float, dt: float, rate_up_key: str,
              rate_down_key: str, legacy_up_key: str, legacy_down_key: str) -> float:
        """Use time-based config, with an old per-frame config compatibility path.

        Legacy values are interpreted at the historical nominal 20 Hz only for
        callers/configurations that have not yet migrated to rate keys.
        """
        c = self.cfg
        nominal_dt = 0.05
        fallback_up = (c.get("max_steering_step", 0.08)
                       if "steering" in legacy_up_key else c.get("throttle_step_up", 0.045))
        fallback_down = (c.get("max_steering_step", 0.08)
                         if "steering" in legacy_down_key else c.get("throttle_step_down", 0.045))
        legacy_up = float(c.get(legacy_up_key, fallback_up))
        legacy_down = float(c.get(legacy_down_key, fallback_down))
        rate_up = float(c.get(rate_up_key, legacy_up / nominal_dt))
        rate_down = float(c.get(rate_down_key, legacy_down / nominal_dt))
        return self._approach_rate(current, target, rate_up, rate_down, dt)

    def _steering_slew(self, current: float, target: float, dt: float,
                       maneuver: bool = False) -> float:
        if maneuver:
            return self._slew(current, target, dt,
                              "steering_maneuver_rate", "steering_maneuver_rate",
                              "maneuver_steering_step", "maneuver_steering_step")
        return self._slew(current, target, dt,
                          "steering_rate", "steering_rate",
                          "max_steering_step", "max_steering_step")

    def _throttle_slew(self, current: float, target: float, dt: float,
                       maneuver: bool = False) -> float:
        if maneuver:
            return self._slew(current, target, dt,
                              "maneuver_throttle_accel_rate", "maneuver_throttle_brake_rate",
                              "maneuver_throttle_step_up", "maneuver_throttle_step_down")
        return self._slew(current, target, dt,
                          "throttle_accel_rate", "throttle_brake_rate",
                          "throttle_step_up", "throttle_step_down")

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
        if lane.valid:
            self.seen_lane = True

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
            self.obstacle_avoid_elapsed = 0.0
            self.obstacle_timeout_latched = False
        self.white_cooldown_time = max(0.0, self.white_cooldown_time - elapsed)
        white_left = forbidden_left >= white_threshold and forbidden_left > forbidden_right + white_margin
        white_right = forbidden_right >= white_threshold and forbidden_right > forbidden_left + white_margin
        # Lane-only deployments intentionally disable the legacy obstacle
        # manoeuvre path even if a model/backend returns an extra class.
        obstacle_active = (not bool(c.get("lane_only", False)) and
                           obstacle >= obstacle_trigger)
        if obstacle < obstacle_trigger * 0.70:
            self.obstacle_timeout_latched = False
        if forbidden_front >= white_front_threshold:
            self.white_front_frames += 1
        else:
            self.white_front_frames = 0
        white_front_blocked = self.white_front_frames >= white_front_required

        # Obstacle has priority over shoulder recovery. Live telemetry showed
        # long white-mask latches suppressing avoidance despite high YOLO risk.
        has_safety_context = self.seen_lane
        obstacle_context = has_safety_context and lane.source != "lost"
        if (obstacle_context and obstacle_active and not self.obstacle_timeout_latched and
                (self.maneuver is None or not self.maneuver.startswith("obstacle_"))):
            self.maneuver_preempted = self.maneuver is not None
            if self.last_throttle < 0.0:
                # Ordinary obstacles never request reverse. If white recovery
                # was already reversing, give the ESC a neutral interval before
                # resuming the requested forward detour.
                self.maneuver = "obstacle_forward_neutral"
                self.maneuver_time = neutral_time
            else:
                self.maneuver = "obstacle_avoid"
                self.maneuver_time = float(c.get("obstacle_avoid_time", 2.0))
            self.maneuver_frames = 0
            self.obstacle_avoid_elapsed = 0.0
            self.maneuver_steering = float(np.sign(lane.target_x - width / 2.0))
            if escape_steering != 0.0:
                self.maneuver_steering = float(np.sign(escape_steering))
            if self.maneuver_steering == 0.0:
                self.maneuver_steering = -1.0 if forbidden_left > forbidden_right else 1.0
        elif self.maneuver is None and has_safety_context:
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
            if self.maneuver.startswith("obstacle_"):
                self.obstacle_avoid_elapsed += elapsed
            maneuver_gain = (c.get("obstacle_avoid_steering", 0.72)
                             if self.maneuver.startswith("obstacle_")
                             else c.get("recovery_steering", 0.42))
            maneuver_target = self.maneuver_steering * maneuver_gain
            # An explicit pre-empt rate is reserved for obstacle hand-off;
            # ordinary recovery remains time based as well.
            if self.maneuver_preempted and "obstacle_preempt_steering_rate" in c:
                steer = self._approach_rate(self.last_steering, maneuver_target,
                                            c["obstacle_preempt_steering_rate"],
                                            c["obstacle_preempt_steering_rate"], elapsed)
            else:
                steer = self._steering_slew(self.last_steering, maneuver_target,
                                            elapsed, maneuver=True)
            if self.maneuver.endswith("_neutral"):
                # This is deliberately a real 0-throttle motor command, not a
                # skipped frame.  The next command can therefore engage reverse.
                self.last_steering = steer
                self.last_throttle = 0.0
                if self.maneuver_time <= 0.0 and self.maneuver == "obstacle_forward_neutral":
                    self.maneuver = "obstacle_avoid"
                    self.maneuver_time = float(c.get("obstacle_avoid_time", 2.0))
                    self.maneuver_frames = 0
                elif self.maneuver_time <= 0.0:
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
            elif self.maneuver == "white_correct":
                if self.maneuver_time <= 1e-6:
                    self.maneuver = None
                    self.white_cooldown_time = float(c.get("white_recovery_cooldown", 0.40))
            elif self.maneuver == "obstacle_avoid":
                if (self.obstacle_avoid_elapsed >=
                        float(c.get("obstacle_avoid_max_time", 4.0))):
                    self.maneuver = None
                    self.obstacle_timeout_latched = True
                    self.last_throttle = 0.0
                    self.last_steering = self._steering_slew(
                        self.last_steering, 0.0, elapsed)
                    return ControlCommand(self.last_steering, 0.0,
                                          "stop:obstacle_no_divider")
                if (self.maneuver_time <= 1e-6 and
                        obstacle < obstacle_trigger * 0.70 and lane.source == "divider"):
                    self.maneuver = None
                elif self.maneuver_time <= 1e-6:
                    # Divider is still blocked: remain in the safe side for a
                    # short retry interval instead of cutting back across it.
                    self.maneuver_time = float(c.get("avoid_retry_time", 0.25))
                    self.maneuver_frames = 0
            if self.maneuver is not None:
                steer = self._steering_slew(self.last_steering,
                                            self.maneuver_steering * maneuver_gain,
                                            elapsed, maneuver=True)
                throttle_key = ("obstacle_avoid_throttle"
                                if self.maneuver == "obstacle_avoid"
                                else "recovery_throttle")
                throttle = float(c.get(throttle_key, c["throttle_min"]))
                throttle = self._throttle_slew(self.last_throttle, throttle,
                                               elapsed, maneuver=True)
                self.last_steering, self.last_throttle = steer, throttle
                self.maneuver_preempted = False
                return ControlCommand(steer, throttle, "avoid:" + self.maneuver.split("_")[0])
            self.maneuver_preempted = False

        if self.obstacle_timeout_latched:
            self.last_throttle = 0.0
            self.last_steering = self._steering_slew(
                self.last_steering, 0.0, elapsed)
            return ControlCommand(self.last_steering, 0.0,
                                  "stop:obstacle_no_divider")

        if not lane.valid:
            self.lost_frames += 1
        else:
            self.lost_frames = 0

        if self.lost_frames > int(c["max_lost_frames"]):
            self.integral = 0.0
            if self.seen_lane and escape_steering != 0.0:
                target = float(np.sign(escape_steering)) * float(c.get("lane_reacquire_steering", 0.58))
                self.last_steering = self._steering_slew(
                    self.last_steering, target, elapsed, maneuver=True)
                self.last_throttle = float(c.get("lane_reacquire_throttle", c["throttle_min"]))
                return ControlCommand(self.last_steering, self.last_throttle, "reacquire:road")
            self.last_throttle = 0.0
            self.last_steering = self._steering_slew(self.last_steering, 0.0, elapsed)
            return ControlCommand(self.last_steering, 0.0, "stop:lane_lost")

        # Before the first valid segmentation there is no safe direction to
        # drive. The first valid lane estimate below immediately enters follow.
        if not self.seen_lane:
            self.last_throttle = self._throttle_slew(self.last_throttle, 0.0, elapsed)
            return ControlCommand(self.last_steering, self.last_throttle, "stop:lane_lost")

        # A short segmentation dropout must not become a full-throttle event.
        # Hold steering briefly and ease toward a conservative speed until the
        # divider is visible again.  Accelerating while blind caused the car to
        # leave the track before reacquisition could take effect.
        if not lane.valid:
            dropout_scale = float(c.get("lane_dropout_throttle_scale", 0.70))
            dropout_target = min(
                float(c["throttle_cruise"]) * dropout_scale,
                self.last_throttle_target if self.last_throttle_target > 0.0 else self.last_throttle,
                self.last_throttle,
            )
            self.last_throttle = self._throttle_slew(
                self.last_throttle, dropout_target, elapsed)
            return ControlCommand(self.last_steering, self.last_throttle, "slow:lane_dropout")

        error = (lane.target_x - width / 2.0) / max(1.0, width / 2.0)
        self.integral = float(np.clip(self.integral + error * dt, -0.5, 0.5))
        derivative = (error - self.last_error) / dt
        derivative_alpha = float(np.clip(c.get("derivative_alpha", 1.0), 0.0, 1.0))
        self.filtered_derivative = (derivative_alpha * derivative +
                                    (1.0 - derivative_alpha) * self.filtered_derivative)
        raw = (c["kp"] * error + c["ki"] * self.integral +
               c["kd"] * self.filtered_derivative +
               c["heading_gain"] * lane.heading_error)
        # A right-hand curve needs an earlier turn-in on this chassis. Apply
        # the configurable anticipation only while the requested turn is
        # right; left steering and all pre-corner speed braking stay unchanged.
        if raw > 0.0:
            raw *= float(c.get("right_turn_anticipation_scale", 1.0))
        raw = float(np.clip(raw, -c["max_steering"], c["max_steering"]))
        # Filter the target before applying the slew limit. Segmentation target
        # jitter can otherwise alternate the wheel direction at inference FPS.
        alpha = float(np.clip(c.get("steering_target_alpha", 1.0), 0.0, 1.0))
        filtered_raw = alpha * raw + (1.0 - alpha) * self.last_raw_steering
        # A single noisy segmentation frame must not immediately command the
        # opposite lock.  Keep the previous direction until the new direction
        # is persistent (or clearly large), which prevents left/right chatter.
        deadband = float(c.get("steering_direction_deadband", 0.08))
        direction = int(np.sign(filtered_raw)) if abs(filtered_raw) >= deadband else 0
        if self.steering_direction and direction and direction != self.steering_direction:
            self.steering_reversal_time += elapsed
            # New configurations express the debounce in seconds.  Preserve
            # the old three-frame setting as a 20 Hz compatibility fallback.
            required = float(c.get("steering_reversal_confirm_time",
                                   float(c.get("steering_reversal_confirm_frames", 3)) * 0.05))
            strong = abs(filtered_raw) >= float(c.get("steering_reversal_force", 0.55))
            if self.steering_reversal_time < required and not strong:
                filtered_raw = float(np.sign(self.steering_direction) *
                                     min(abs(filtered_raw), abs(self.last_raw_steering)))
            else:
                self.steering_direction = direction
                self.steering_reversal_time = 0.0
        elif direction:
            self.steering_direction = direction
            self.steering_reversal_time = 0.0
        steering = self._steering_slew(self.last_steering, filtered_raw, elapsed)

        # Compute the normal speed demand for configurations that want
        # curve-aware throttle. The competition setup can explicitly request
        # full throttle at all times; steering remains responsible for lane
        # following while throttle no longer hunts with segmentation noise.
        measured_curve_load = max(abs(filtered_raw), abs(lane.heading_error), abs(lane.curvature))
        # Remember an approaching curve: react quickly when curvature rises,
        # then release it slowly.  This brakes before/during the corner and
        # prevents alternating fast/slow commands from frame-level mask noise.
        curve_alpha = float(np.clip(
            c.get("curve_load_attack_alpha", 0.65)
            if measured_curve_load > self.curve_load_ema
            else c.get("curve_load_release_alpha", 0.12), 0.0, 1.0))
        self.curve_load_ema += curve_alpha * (measured_curve_load - self.curve_load_ema)
        curve_load = self.curve_load_ema
        speed_scale = max(0.28, 1.0 - c["curve_slowdown"] * curve_load)
        # A good semantic mask should not hold back straight-line speed merely
        # because the geometric confidence metric rarely reaches exactly 1.0.
        # Above the configured threshold confidence has no speed penalty;
        # below it, speed falls smoothly toward the safe floor.
        confidence_full = max(1e-6, float(c.get("confidence_full_speed", 1.0)))
        confidence_ratio = float(np.clip(lane.confidence / confidence_full, 0.0, 1.0))
        confidence_scale = (c["low_confidence_slowdown"] +
                            (1.0 - c["low_confidence_slowdown"]) * confidence_ratio)
        obstacle_scale = max(0.25, 1.0 - obstacle) if obstacle >= c["obstacle_slow_ratio"] else 1.0
        if lane.source == "avoid":
            obstacle_scale = min(obstacle_scale, c["avoidance_speed_scale"])
        boost_start = float(c.get("straight_boost_start", 0.06))
        boost_end = max(boost_start + 1e-6, float(c.get("straight_boost_end", 0.22)))
        straightness = float(np.clip((boost_end - curve_load) / (boost_end - boost_start), 0.0, 1.0))
        speed_base = c["throttle_cruise"] + straightness * (c["throttle_max"] - c["throttle_cruise"])
        target_throttle = float(np.clip(speed_base * speed_scale * confidence_scale * obstacle_scale,
                                        0.0, c["throttle_max"]))
        # A motor's minimum moving command is not a controller speed floor:
        # retain it for gentle travel, but allow a genuine zero-throttle coast
        # before a U-turn or other very sharp corner.
        if curve_load >= float(c.get("corner_curve_load", 0.62)):
            target_throttle = min(target_throttle, float(c.get("corner_throttle", 0.0)))
        elif curve_load >= float(c.get("curve_curve_load", 0.40)):
            target_throttle = min(target_throttle, float(c.get("curve_throttle", target_throttle)))
        # Preserve maximum straight-line speed, but unload the chassis during
        # asymmetric right turns.  The right-side linkage produces more wheel
        # angle than the left, so full throttle there can make the car tip.
        if steering > 0.0:
            right_limit = max(1e-6, float(c.get("max_steering_right", c["max_steering"])))
            right_scale = float(c.get("right_turn_throttle_scale", 1.0))
            turn_fraction = float(np.clip(steering / right_limit, 0.0, 1.0))
            target_throttle *= 1.0 - (1.0 - right_scale) * turn_fraction
        if bool(c.get("throttle_always_max", False)):
            target_throttle = float(c["throttle_max"])
        motion_min = float(c.get("throttle_motion_min", c["throttle_min"]))
        if 0.0 < target_throttle < motion_min:
            target_throttle = motion_min
        if self.last_throttle_target <= 0.0:
            smoothed_target = target_throttle
        else:
            # Drop the demand promptly for a detected curve, but restore speed
            # slowly on corner exit.  Separate rates avoid both late braking
            # and the fast/slow hunting produced by one symmetric filter.
            target_alpha = float(np.clip(
                c.get("throttle_target_brake_alpha", c.get("throttle_target_alpha", 0.35))
                if target_throttle < self.last_throttle_target
                else c.get("throttle_target_accel_alpha", c.get("throttle_target_alpha", 0.35)),
                0.0, 1.0))
            smoothed_target = (target_alpha * target_throttle +
                               (1.0 - target_alpha) * self.last_throttle_target)
        throttle = self._throttle_slew(self.last_throttle, smoothed_target, elapsed)
        self.last_throttle_target = smoothed_target
        self.last_error, self.last_raw_steering = error, filtered_raw
        self.last_steering, self.last_throttle = steering, throttle
        obstacle_state = (not bool(c.get("lane_only", False)) and
                          obstacle >= c["obstacle_slow_ratio"])
        state = "avoid" if lane.source == "avoid" else ("slow:obstacle" if obstacle_state else "follow")
        return ControlCommand(steering, throttle, state, raw, filtered_raw, curve_load)
