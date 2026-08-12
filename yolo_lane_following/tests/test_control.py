import unittest

import numpy as np

from yolo_lane_following.control import AdaptiveController, is_motor_command_state
from yolo_lane_following.geometry import LaneEstimate, estimate_lane, obstacle_risk, plan_semantic_lane


CFG = dict(kp=0.9, ki=0.02, kd=0.05, heading_gain=0.3, max_steering=0.82,
           max_steering_step=0.16, throttle_min=0.09, throttle_cruise=0.18,
           throttle_max=0.24, throttle_step_up=0.008, throttle_step_down=0.035,
           curve_slowdown=0.62, low_confidence_slowdown=0.55,
           emergency_obstacle_ratio=0.78, obstacle_slow_ratio=0.58,
           avoidance_speed_scale=0.45,
           reverse_neutral_time=0.10, reverse_time=0.20,
           max_lost_frames=3)


class GeometryTests(unittest.TestCase):
    def test_vertical_divider_centres_camera(self):
        divider = np.zeros((224, 224), np.uint8); divider[100:, 110:114] = 255
        lane = estimate_lane(divider, np.zeros_like(divider))
        self.assertTrue(lane.valid)
        self.assertAlmostEqual(lane.target_x, 111.5, delta=1.0)

    def test_strict_divider_mode_does_not_invent_lane_from_road(self):
        road = np.zeros((224, 224), np.uint8); road[100:, 30:190] = 255
        self.assertFalse(estimate_lane(np.zeros_like(road), road, target_mode="divider").valid)

    def test_only_obstacle_on_path_has_risk(self):
        self.assertGreater(obstacle_risk([[95, 120, 130, 215]], 224, 224, 112), 0.7)
        self.assertEqual(obstacle_risk([[0, 120, 20, 215]], 224, 224, 112), 0.0)

    def test_semantic_planner_avoids_obstacle_without_entering_forbidden(self):
        road = np.zeros((224, 224), np.uint8); road[90:, 24:200] = 255
        divider = np.zeros_like(road); divider[90:, 110:114] = 255
        forbidden = np.zeros_like(road); forbidden[90:, :24] = 255; forbidden[90:, 200:] = 255
        obstacle = np.zeros_like(road); obstacle[120:205, 92:132] = 255
        lane = plan_semantic_lane(divider, road, forbidden, obstacle)
        self.assertTrue(lane.valid)
        self.assertEqual(lane.source, "avoid")
        self.assertTrue(lane.target_x < 88 or lane.target_x > 136)

    def test_obstacle_class_overrides_any_visual_background(self):
        for background in ("road", "divider", "forbidden"):
            road = np.zeros((224, 224), np.uint8); road[90:, 24:200] = 255
            divider = np.zeros_like(road); divider[90:, 110:114] = 255
            forbidden = np.zeros_like(road); forbidden[90:, :24] = 255; forbidden[90:, 200:] = 255
            obstacle = np.zeros_like(road); obstacle[120:205, 96:128] = 255
            # Semantic output is single-class per pixel. Simulate an obstacle
            # replacing whichever background class its colour resembles.
            if background == "road":
                road[obstacle > 0] = 0
            elif background == "divider":
                divider[obstacle > 0] = 0
                divider[90:120, 110:114] = 255
            else:
                forbidden[obstacle > 0] = 0
            lane = plan_semantic_lane(divider, road, forbidden, obstacle)
            self.assertIn(lane.source, ("avoid", "blocked"), background)

    def test_obstacle_touching_path_corridor_triggers_risk(self):
        risk = obstacle_risk([[145, 135, 166, 215]], 224, 224, 112)
        self.assertGreaterEqual(risk, CFG["obstacle_slow_ratio"])

    def test_semantic_planner_stops_when_white_shoulders_leave_no_clearance(self):
        road = np.zeros((224, 224), np.uint8); road[90:, 94:130] = 255
        divider = np.zeros_like(road); divider[90:, 110:114] = 255
        forbidden = np.zeros_like(road); forbidden[90:, :108] = 255; forbidden[90:, 116:] = 255
        lane = plan_semantic_lane(divider, road, forbidden, np.zeros_like(road))
        self.assertFalse(lane.valid)


class ControlTests(unittest.TestCase):
    def test_motor_gate_allows_recovery_and_obstacle_reverse(self):
        for state in ("follow", "avoid:obstacle", "neutral:obstacle",
                      "reverse:obstacle", "neutral:white", "reverse:white",
                      "reacquire:road"):
            self.assertTrue(is_motor_command_state(state), state)
        for state in ("stop:lane_lost", "wait:lane_lock", "slow:lane_dropout"):
            self.assertFalse(is_motor_command_state(state), state)

    def test_live_throttle_limit_scales_all_forward_modes(self):
        cfg = dict(CFG, obstacle_avoid_throttle=0.16, recovery_throttle=0.11,
                   lane_reacquire_throttle=0.10, reverse_throttle=0.11,
                   throttle_limit_min=0.08, throttle_limit_max=0.60)
        ctl = AdaptiveController(cfg)
        applied = ctl.set_throttle_limit(0.48)
        self.assertEqual(applied, 0.48)
        self.assertAlmostEqual(ctl.cfg["throttle_cruise"], 0.36)
        self.assertAlmostEqual(ctl.cfg["obstacle_avoid_throttle"], 0.32)
        self.assertAlmostEqual(ctl.cfg["recovery_throttle"], 0.22)
        self.assertAlmostEqual(ctl.cfg["lane_reacquire_throttle"], 0.20)
        self.assertEqual(ctl.cfg["reverse_throttle"], 0.11)

    def test_live_throttle_limit_is_hard_clamped(self):
        cfg = dict(CFG, obstacle_avoid_throttle=0.16, recovery_throttle=0.11,
                   lane_reacquire_throttle=0.10, reverse_throttle=0.11,
                   throttle_limit_min=0.08, throttle_limit_max=0.60)
        ctl = AdaptiveController(cfg)
        self.assertEqual(ctl.set_throttle_limit(2.0), 0.60)
        self.assertLessEqual(ctl.cfg["throttle_cruise"], 0.60)
        self.assertLessEqual(ctl.cfg["obstacle_avoid_throttle"], 0.60)
        self.assertEqual(ctl.set_throttle_limit(0.01), 0.08)
        self.assertLessEqual(ctl.cfg["reverse_throttle"], 0.08)

    def test_steering_and_throttle_are_automatic(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(True, 160, 150, 0.1, 0.1, 1.0, "divider")
        cmd = ctl.update(lane, 0.0, 224, 0.05)
        self.assertGreater(cmd.steering, 0)
        self.assertGreater(cmd.throttle, 0)

    def test_close_obstacle_drives_forward_then_returns_to_divider(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(True, 160, 112, 0, 0, 1, "avoid")
        avoid = ctl.update(lane, 0.9, 224, 0.05, escape_steering=-1.0)
        self.assertGreater(avoid.throttle, 0)
        self.assertEqual(avoid.state, "avoid:obstacle")
        self.assertLess(avoid.steering, 0)
        divider = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        for _ in range(45):
            avoid = ctl.update(divider, 0.0, 224, 0.05)
        self.assertEqual(avoid.state, "follow")

    def test_obstacle_hold_uses_wall_clock_time_at_low_fps(self):
        ctl = AdaptiveController(dict(CFG, obstacle_avoid_time=0.6))
        lane = LaneEstimate(True, 160, 112, 0, 0, 1, "avoid")
        cmd = ctl.update(lane, 0.9, 224, 0.4, escape_steering=-1.0)
        self.assertEqual(cmd.state, "avoid:obstacle")
        divider = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        cmd = ctl.update(divider, 0.0, 224, 0.4)
        self.assertEqual(cmd.state, "avoid:obstacle")
        cmd = ctl.update(divider, 0.0, 224, 0.4)
        self.assertEqual(cmd.state, "follow")

    def test_white_left_drives_forward_and_steers_right(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        cmd = ctl.update(lane, 0.0, 224, 0.05, forbidden_left=0.30, forbidden_right=0.02)
        self.assertGreater(cmd.throttle, 0)
        self.assertGreater(cmd.steering, 0)
        self.assertEqual(cmd.state, "avoid:white")

    def test_single_front_white_frame_never_reverses(self):
        ctl = AdaptiveController(dict(CFG, white_front_reverse_threshold=0.58,
                                      white_front_reverse_frames=2))
        lane = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        cmd = ctl.update(lane, 0.0, 224, 0.05, forbidden_front=0.80)
        self.assertGreater(cmd.throttle, 0)
        self.assertEqual(cmd.state, "follow")

    def test_repeated_front_white_reverses_briefly(self):
        ctl = AdaptiveController(dict(CFG, white_front_reverse_threshold=0.58,
                                      white_front_reverse_frames=2))
        lane = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        first = ctl.update(lane, 0.0, 224, 0.05, escape_steering=-1.0,
                           forbidden_front=0.80)
        second = ctl.update(lane, 0.0, 224, 0.05, escape_steering=-1.0,
                            forbidden_front=0.80)
        self.assertGreater(first.throttle, 0)
        self.assertEqual(second.state, "neutral:white")
        reverse = None
        for _ in range(3):
            reverse = ctl.update(lane, 0.0, 224, 0.15, escape_steering=-1.0,
                                 forbidden_front=0.80)
            if reverse.state == "reverse:white":
                break
        self.assertEqual(reverse.state, "reverse:white")
        self.assertLess(reverse.throttle, 0)

    def test_normal_road_never_commands_reverse(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        for _ in range(30):
            cmd = ctl.update(lane, 0.0, 224, 0.05)
            self.assertGreaterEqual(cmd.throttle, 0)
            self.assertEqual(cmd.state, "follow")

    def test_obstacle_preempts_active_white_recovery(self):
        ctl = AdaptiveController(CFG)
        divider = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        ctl.update(divider, 0.0, 224, 0.05, forbidden_left=0.30)
        ctl.update(divider, 0.0, 224, 0.25, forbidden_left=0.30)
        avoid = LaneEstimate(True, 160, 112, 0, 0, 1, "avoid")
        cmd = ctl.update(avoid, 0.9, 224, 0.05, forbidden_left=0.30,
                         escape_steering=-1.0)
        self.assertEqual(cmd.state, "avoid:obstacle")
        self.assertLess(cmd.steering, 0)

    def test_persistent_white_mask_does_not_latch_forever(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        ctl.update(lane, 0.0, 224, 0.05, forbidden_left=0.30)
        ctl.update(lane, 0.0, 224, 0.25, forbidden_left=0.30)
        ctl.update(lane, 0.0, 224, 0.40, forbidden_left=0.30)
        cmd = ctl.update(lane, 0.0, 224, 0.40, forbidden_left=0.30)
        self.assertEqual(cmd.state, "follow")

    def test_blocked_obstacle_uses_wider_side_hint(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(False, 112, 112, 0, 1, 0, "blocked")
        cmd = ctl.update(lane, 0.9, 224, 0.05, escape_steering=-1.0)
        self.assertLess(cmd.throttle, 0)
        self.assertLess(cmd.steering, 0)

    def test_low_fps_blocked_obstacle_still_emits_reverse(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(False, 112, 112, 0, 1, 0, "blocked")
        first = ctl.update(lane, 0.9, 224, 0.4, escape_steering=1.0)
        self.assertEqual(first.state, "reverse:obstacle")
        self.assertLess(first.throttle, 0)

    def test_lane_loss_reacquires_toward_remaining_road_after_lock(self):
        ctl = AdaptiveController(CFG)
        locked = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        ctl.update(locked, 0, 224, 0.05)
        lost = LaneEstimate(False, 112, 112, 0, 1, 0, "lost")
        for _ in range(4):
            cmd = ctl.update(lost, 0, 224, 0.05, escape_steering=-1.0)
        self.assertEqual(cmd.state, "reacquire:road")
        self.assertLess(cmd.steering, 0)
        self.assertGreater(cmd.throttle, 0)

    def test_lane_loss_stops_without_any_road_hint(self):
        ctl = AdaptiveController(CFG)
        locked = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        ctl.update(locked, 0, 224, 0.05)
        lost = LaneEstimate(False, 112, 112, 0, 1, 0, "lost")
        for _ in range(4):
            cmd = ctl.update(lost, 0, 224, 0.05)
        self.assertEqual(cmd.state, "stop:lane_lost")

    def test_startup_never_accelerates_without_lane_lock(self):
        ctl = AdaptiveController(CFG)
        lost = LaneEstimate(False, 112, 112, 0, 1, 0, "lost")
        for _ in range(3):
            cmd = ctl.update(lost, 0, 224, 0.05)
            self.assertEqual(cmd.throttle, 0)
            self.assertEqual(cmd.state, "wait:lane_lock")

    def test_short_dropout_decelerates_after_lane_lock(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        for _ in range(4):
            moving = ctl.update(lane, 0, 224, 0.05)
        lost = LaneEstimate(False, 112, 112, 0, 1, 0, "lost")
        dropout = ctl.update(lost, 0, 224, 0.05)
        self.assertLess(dropout.throttle, moving.throttle)
        self.assertEqual(dropout.state, "slow:lane_dropout")


if __name__ == "__main__":
    unittest.main()
