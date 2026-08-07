import unittest

import numpy as np

from yolo_lane_following.control import AdaptiveController
from yolo_lane_following.geometry import LaneEstimate, estimate_lane, obstacle_risk


CFG = dict(kp=0.9, ki=0.02, kd=0.05, heading_gain=0.3, max_steering=0.82,
           max_steering_step=0.16, throttle_min=0.09, throttle_cruise=0.18,
           throttle_max=0.24, throttle_step_up=0.008, throttle_step_down=0.035,
           curve_slowdown=0.62, low_confidence_slowdown=0.55,
           emergency_obstacle_ratio=0.78, obstacle_slow_ratio=0.58,
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


class ControlTests(unittest.TestCase):
    def test_steering_and_throttle_are_automatic(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(True, 160, 150, 0.1, 0.1, 1.0, "divider")
        cmd = ctl.update(lane, 0.0, 224, 0.05)
        self.assertGreater(cmd.steering, 0)
        self.assertGreater(cmd.throttle, 0)

    def test_close_obstacle_stops(self):
        ctl = AdaptiveController(CFG)
        lane = LaneEstimate(True, 112, 112, 0, 0, 1, "divider")
        self.assertEqual(ctl.update(lane, 0.9, 224, 0.05).throttle, 0)

    def test_lane_loss_stops_after_debounce(self):
        ctl = AdaptiveController(CFG)
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
