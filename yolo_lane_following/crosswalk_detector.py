"""Cheap, temporal zebra-crossing detector used only as a junction trigger."""
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CrosswalkEstimate:
    present: bool
    score: float
    y: float
    mask: np.ndarray
    bars: int
    traffic_light: Optional[object] = None


class CrosswalkDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.history = deque(maxlen=max(1, int(cfg.get("history_frames", 5))))
        # Kept here as a compatibility path for the live notebook: it lets
        # the trigger deliver a light state in the same frame. The standalone
        # runtime also owns its explicitly configured detector for logging.
        from .traffic_light import TrafficLightDetector
        self.traffic_light = TrafficLightDetector(cfg.get("traffic_light", {}))

    def update(self, frame: np.ndarray) -> CrosswalkEstimate:
        h, w = frame.shape[:2]
        y0, y1 = int(h * self.cfg.get("roi_top", .45)), int(h * self.cfg.get("roi_bottom", .95))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # White is low-saturation and bright. This intentionally does not use
        # the semantic forbidden class: markings are traversable.
        white = cv2.inRange(hsv, np.array([0, 0, self.cfg.get("white_value", 165)]),
                            np.array([180, self.cfg.get("white_saturation", 80), 255]))
        roi = white[y0:y1]
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
        centres = []
        min_width = int(w * self.cfg.get("min_bar_width_ratio", .13))
        for x, y, bw, bh, area in stats[1:count]:
            ratio = bw / max(1, bh)
            if bw >= min_width and bh >= 2 and ratio >= self.cfg.get("min_bar_aspect", 2.2) and area >= bw * 2:
                centres.append(y0 + y + bh / 2.0)
        centres.sort()
        # A true zebra has several horizontal bars, not just one road marking.
        bars = len(centres)
        raw = bars >= int(self.cfg.get("min_bars", 4))
        self.history.append(raw)
        confirmed = sum(self.history) >= int(self.cfg.get("confirm_frames", 3))
        score = min(1.0, bars / max(1, int(self.cfg.get("min_bars", 4))))
        return CrosswalkEstimate(confirmed, score, (max(centres) / h if centres else 0.0), white, bars,
                                 self.traffic_light.update(frame))
