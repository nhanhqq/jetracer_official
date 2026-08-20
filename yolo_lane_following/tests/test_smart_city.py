import unittest
import cv2
import numpy as np

from yolo_lane_following.crosswalk_detector import CrosswalkDetector
from yolo_lane_following.decision import choose_branch
from yolo_lane_following.intersection_geometry import BranchExtractor
from yolo_lane_following.traffic_light import TrafficLightDetector
from yolo_lane_following.geometry import estimate_lane


class SmartCityTests(unittest.TestCase):
    def test_crosswalk_requires_temporal_confirmation(self):
        detector = CrosswalkDetector(dict(roi_top=.45, roi_bottom=.95, min_bars=4,
                                          confirm_frames=3, history_frames=5))
        image = np.zeros((224, 224, 3), np.uint8)
        for y in range(120, 180, 12):
            cv2.rectangle(image, (35, y), (190, y + 5), (255, 255, 255), -1)
        self.assertFalse(detector.update(image).present)
        self.assertFalse(detector.update(image).present)
        estimate = detector.update(image)
        self.assertTrue(estimate.present)
        self.assertGreaterEqual(estimate.bars, 4)

    def test_branch_scores_require_component_connected_to_ego(self):
        cfg = dict(homography=np.eye(3).tolist(), bev_width=224, bev_height=224,
                   branch_min_score=.08)
        road = np.zeros((224, 224), np.uint8)
        # ego stem joined to left and right exits; no top/straight connection
        road[145:224, 100:124] = 255
        road[100:165, :124] = 255
        road[100:165, 100:224] = 255
        result = BranchExtractor(cfg).update(road, np.zeros_like(road), np.zeros_like(road), np.zeros_like(road))
        self.assertTrue(result.valid)
        self.assertTrue(result.available["left"])
        self.assertTrue(result.available["right"])
        self.assertFalse(result.available["straight"])

    def test_sign_rules_filter_existing_branches_only(self):
        available = dict(left=True, straight=False, right=True)
        self.assertEqual(choose_branch(available, "NO_LEFT", ["straight", "right", "left"]), "right")
        self.assertIsNone(choose_branch(available, "MUST_STRAIGHT", ["right", "left"]))

    def test_traffic_light_needs_persistent_compact_green_or_red(self):
        cfg = dict(roi_top=0, roi_bottom=1, min_blob_ratio=.00002, max_blob_ratio=.01,
                   max_blob_aspect=2.2, score_min=.05, history_frames=5, confirm_frames=3)
        frame = np.zeros((224, 224, 3), np.uint8)
        cv2.circle(frame, (112, 70), 8, (0, 255, 0), -1)
        detector = TrafficLightDetector(cfg)
        self.assertEqual(detector.update(frame).state, "UNKNOWN")
        self.assertEqual(detector.update(frame).state, "UNKNOWN")
        self.assertEqual(detector.update(frame).state, "GREEN")

    def test_explicit_road_mode_ignores_an_edge_divider(self):
        road = np.zeros((224, 224), np.uint8); road[90:, 45:180] = 255
        divider = np.zeros_like(road); divider[90:, 45:49] = 255
        lane = estimate_lane(divider, road, target_mode="road")
        self.assertTrue(lane.valid)
        self.assertEqual(lane.source, "road_center")
        self.assertAlmostEqual(lane.target_x, 112, delta=2)
