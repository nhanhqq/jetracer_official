"""Temporal red/green light gate. UNKNOWN is intentionally not permission to go."""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrafficLightEstimate:
    state: str                 # RED, GREEN, UNKNOWN
    red_score: float
    green_score: float


class TrafficLightDetector:
    """Detect compact red/green lamps, excluding large green map islands.

    This is a calibrated MVP detector for the fixed tabletop setup. A trained
    sign/light detector can replace it later, but its output must retain this
    temporal interface and the fail-safe UNKNOWN state.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.history = deque(maxlen=max(1, int(cfg.get("history_frames", 5))))

    def update(self, frame: np.ndarray) -> TrafficLightEstimate:
        h, w = frame.shape[:2]
        roi_top, roi_bottom = int(h * self.cfg.get("roi_top", .05)), int(h * self.cfg.get("roi_bottom", .70))
        hsv = cv2.cvtColor(frame[roi_top:roi_bottom], cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, np.array([0, 100, 120]), np.array([12, 255, 255])) | cv2.inRange(hsv, np.array([168, 100, 120]), np.array([180, 255, 255]))
        green = cv2.inRange(hsv, np.array([40, 80, 90]), np.array([90, 255, 255]))
        max_area = max(1, int(h * w * float(self.cfg.get("max_blob_ratio", .008))))
        min_area = max(1, int(h * w * float(self.cfg.get("min_blob_ratio", .00002))))
        def score(mask):
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            best = 0.0
            for _x, _y, bw, bh, area in stats[1:count]:
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if min_area <= area <= max_area and aspect <= float(self.cfg.get("max_blob_aspect", 2.2)):
                    best = max(best, float(area) / max_area)
            return min(1.0, best)
        red_score, green_score = score(red), score(green)
        raw = "RED" if red_score >= green_score and red_score >= float(self.cfg.get("score_min", .12)) else ("GREEN" if green_score >= float(self.cfg.get("score_min", .12)) else "UNKNOWN")
        self.history.append(raw)
        red_votes, green_votes = self.history.count("RED"), self.history.count("GREEN")
        required = int(self.cfg.get("confirm_frames", 3))
        state = "RED" if red_votes >= required else ("GREEN" if green_votes >= required else "UNKNOWN")
        return TrafficLightEstimate(state, red_score, green_score)
