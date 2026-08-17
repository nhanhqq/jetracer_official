import unittest

import cv2
import numpy as np

from yolo_lane_following.start_gate import (
    CompetitionStartGate,
    competition_motor_allowed,
    detect_green_circle,
)


CFG = dict(hsv_lower=[38, 100, 120], hsv_upper=[88, 255, 255],
           min_area_ratio=0.002, max_area_ratio=0.20,
           min_circularity=0.82, confirm_frames=3, latch_start=True)


class StartGateTests(unittest.TestCase):
    def test_green_circle_is_segmented(self):
        frame = np.zeros((224, 224, 3), np.uint8)
        cv2.circle(frame, (112, 90), 18, (0, 255, 0), -1)
        result = detect_green_circle(frame, CFG)
        self.assertTrue(result.detected)
        self.assertAlmostEqual(result.center[0], 112, delta=2)

    def test_mild_perspective_green_ellipse_is_accepted(self):
        frame = np.zeros((224, 224, 3), np.uint8)
        cv2.ellipse(frame, (112, 90), (20, 16), 12, 0, 360, (0, 255, 0), -1)
        self.assertTrue(detect_green_circle(frame, CFG).detected)

    def test_green_rectangle_is_rejected(self):
        frame = np.zeros((224, 224, 3), np.uint8)
        cv2.rectangle(frame, (40, 90), (185, 110), (0, 255, 0), -1)
        self.assertFalse(detect_green_circle(frame, CFG).detected)

    def test_green_square_is_rejected(self):
        frame = np.zeros((224, 224, 3), np.uint8)
        cv2.rectangle(frame, (82, 82), (142, 142), (0, 255, 0), -1)
        self.assertFalse(detect_green_circle(frame, CFG).detected)

    def test_dark_green_circle_is_rejected(self):
        frame = np.zeros((224, 224, 3), np.uint8)
        cv2.circle(frame, (112, 90), 18, (0, 80, 0), -1)
        self.assertFalse(detect_green_circle(frame, CFG).detected)

    def test_three_frames_authorize_and_reset_revokes(self):
        frame = np.zeros((224, 224, 3), np.uint8)
        cv2.circle(frame, (112, 90), 18, (0, 255, 0), -1)
        gate = CompetitionStartGate(CFG)
        for expected in (False, False, True):
            gate.update(frame)
            self.assertEqual(gate.authorized, expected)
        gate.reset()
        self.assertFalse(gate.authorized)
        self.assertIsNone(gate.authorization_latency_ms)

    def test_authorization_latency_is_measured_from_first_green_frame(self):
        frame = np.zeros((224, 224, 3), np.uint8)
        cv2.circle(frame, (112, 90), 18, (0, 255, 0), -1)
        gate = CompetitionStartGate(CFG)
        gate.update(frame, now=10.00)
        gate.update(frame, now=10.05)
        gate.update(frame, now=10.10)
        self.assertTrue(gate.authorized)
        self.assertAlmostEqual(gate.authorization_latency_ms, 100.0)

    def test_competition_motor_gate_truth_table(self):
        # Competition mode requires all three independent permissions.
        for armed in (False, True):
            for green in (False, True):
                for safe_state in (False, True):
                    expected = armed and green and safe_state
                    self.assertEqual(
                        competition_motor_allowed(armed, True, green, safe_state),
                        expected,
                    )
        # Normal mode retains the deliberate ARM + controller-state interlock.
        self.assertTrue(competition_motor_allowed(True, False, False, True))
        self.assertFalse(competition_motor_allowed(False, False, True, True))
        self.assertFalse(competition_motor_allowed(True, False, True, False))


if __name__ == "__main__":
    unittest.main()
