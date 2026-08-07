import unittest

from lane_detection_v2 import LaneDetector


class ObstaclePlannerTests(unittest.TestCase):
    def setUp(self):
        self.detector = LaneDetector(224, 224)
        self.detector.active_lane_pair = (70.0, 140.0)
        self.detector.last_lane_pair = (70.0, 140.0)
        self.detector.last_target_x = 105

    @staticmethod
    def obstacle(center_x):
        return {"center_x": center_x, "center_y": 170,
                "x": center_x - 10, "y": 150, "w": 20, "h": 40, "area": 800}

    def test_object_in_other_lane_is_ignored(self):
        self.detector._last_detected_pairs = [(5.0, 70.0), (70.0, 140.0)]
        target, action = self.detector.plan_lane_target(105, self.obstacle(35), 224)
        self.assertEqual((target, action), (105, "follow:active_lane"))

    def test_object_in_active_lane_switches_and_holds_adjacent_lane(self):
        self.detector._last_detected_pairs = [(5.0, 70.0), (70.0, 140.0)]
        target, action = self.detector.plan_lane_target(105, self.obstacle(105), 224)
        self.assertEqual(action, "avoid:switch_lane")
        self.assertEqual(target, 37)
        for _ in range(self.detector.return_after_clear_frames - 1):
            target, action = self.detector.plan_lane_target(105, None, 224)
            self.assertEqual(action, "avoid:hold_other_lane")
            self.assertEqual(target, 37)

    def test_returns_only_after_sustained_clear_frames(self):
        self.detector._last_detected_pairs = [(5.0, 70.0), (70.0, 140.0)]
        self.detector.plan_lane_target(105, self.obstacle(105), 224)
        for _ in range(self.detector.return_after_clear_frames):
            target, action = self.detector.plan_lane_target(105, None, 224)
        self.assertEqual(action, "avoid:return_lane")
        self.assertEqual(target, 105)

    def test_without_adjacent_lane_correction_stays_inside_boundaries(self):
        self.detector._last_detected_pairs = [(70.0, 140.0)]
        target, action = self.detector.plan_lane_target(105, self.obstacle(95), 224)
        self.assertEqual(action, "avoid:within_lane")
        self.assertGreaterEqual(target, 86)
        self.assertLessEqual(target, 124)


if __name__ == "__main__":
    unittest.main()
