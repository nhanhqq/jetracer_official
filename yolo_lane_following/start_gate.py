from dataclasses import dataclass
from math import pi
import time
from typing import Tuple

import cv2
import numpy as np


@dataclass
class GreenCircleResult:
    detected: bool
    center: Tuple[int, int]
    radius: int
    area_ratio: float
    circularity: float
    mask: np.ndarray


def detect_green_circle(frame_bgr: np.ndarray, cfg: dict) -> GreenCircleResult:
    """Segment a sufficiently large, circular green start marker."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.asarray(cfg.get("hsv_lower", [38, 100, 120]), dtype=np.uint8)
    upper = np.asarray(cfg.get("hsv_upper", [88, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    image_area = float(max(1, frame_bgr.shape[0] * frame_bgr.shape[1]))
    best = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        ratio = area / image_area
        circularity = 4.0 * pi * area / max(1e-6, perimeter * perimeter)
        if not (float(cfg.get("min_area_ratio", 0.002)) <= ratio <=
                float(cfg.get("max_area_ratio", 0.20))):
            continue
        if circularity < float(cfg.get("min_circularity", 0.82)):
            continue
        if best is None or area > best[0]:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            best = area, int(round(cx)), int(round(cy)), int(round(radius)), circularity, ratio
    if best is None:
        return GreenCircleResult(False, (-1, -1), 0, 0.0, 0.0, mask)
    _, cx, cy, radius, circularity, ratio = best
    return GreenCircleResult(True, (cx, cy), radius, ratio, circularity, mask)


class CompetitionStartGate:
    """Debounce the green marker and optionally latch start authorization."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.consecutive = 0
        self.authorized = False
        self.first_detected_at = None
        self.authorization_latency_ms = None

    def reset(self) -> None:
        self.consecutive = 0
        self.authorized = False
        self.first_detected_at = None
        self.authorization_latency_ms = None

    def update(self, frame_bgr: np.ndarray, now: float = None) -> GreenCircleResult:
        now = time.perf_counter() if now is None else float(now)
        result = detect_green_circle(frame_bgr, self.cfg)
        if result.detected:
            if self.consecutive == 0:
                self.first_detected_at = now
            self.consecutive += 1
        else:
            self.consecutive = 0
            self.first_detected_at = None
        if (not self.authorized and
                self.consecutive >= int(self.cfg.get("confirm_frames", 3))):
            self.authorized = True
            self.authorization_latency_ms = 1000.0 * (
                now - (self.first_detected_at if self.first_detected_at is not None else now))
        if not bool(self.cfg.get("latch_start", True)) and not result.detected:
            self.authorized = False
            self.authorization_latency_ms = None
        return result


def competition_motor_allowed(armed: bool, competition_ready: bool,
                              start_authorized: bool, safe_state: bool) -> bool:
    """Single motor gate used by the notebook and covered by a truth-table test."""
    start_gate_open = (not competition_ready) or start_authorized
    return bool(armed and start_gate_open and safe_state)
