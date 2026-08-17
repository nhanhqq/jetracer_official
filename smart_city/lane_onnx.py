from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class LaneResult:
    valid: bool
    target_x: float
    heading_error: float
    confidence: float
    forbidden_front: float
    mask: np.ndarray


class SemanticLaneONNX:
    """Runtime adapter for track_yolo26n_sem_best.onnx."""

    def __init__(self, model_path: Path, input_size: int = 224,
                 road_class: int = 1, divider_class: int = 2,
                 forbidden_class: int = 3):
        import onnxruntime as ort

        self.input_size = int(input_size)
        self.road_class = int(road_class)
        self.divider_class = int(divider_class)
        self.forbidden_class = int(forbidden_class)
        self.session = ort.InferenceSession(str(model_path),
                                             providers=["CUDAExecutionProvider",
                                                        "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, frame: np.ndarray, lookahead_ratio: float = 0.60,
              bottom_ratio: float = 0.93, min_pixels: int = 35) -> LaneResult:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: rgb.transpose(2, 0, 1)[None]})[0]
        labels = np.asarray(output).squeeze().astype(np.int32)
        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        divider = labels == self.divider_class
        road = labels == self.road_class
        forbidden = labels == self.forbidden_class
        ys, xs = np.where(divider)
        if len(xs) < int(min_pixels):
            return LaneResult(False, w / 2.0, 0.0, 0.0, self._front(forbidden), labels)
        look_y = int(np.clip(h * float(lookahead_ratio), 0, h - 1))
        near_y = int(np.clip(h * float(bottom_ratio), look_y, h - 1))
        points = []
        for y in np.linspace(look_y, near_y, 8).astype(np.int32):
            row = np.where(divider[y])[0]
            if len(row):
                points.append((float(y), float(np.mean(row))))
        if not points:
            return LaneResult(False, w / 2.0, 0.0, 0.0, self._front(forbidden), labels)
        fit = np.polyfit([p[0] for p in points], [p[1] for p in points], 1) if len(points) >= 2 else [0.0, points[0][1]]
        raw_target_x = float(np.polyval(fit, look_y))
        # A polynomial can extrapolate far outside the image when a few
        # divider pixels are spurious. Treat that as lane loss; never turn a
        # vehicle toward an invented target outside the camera view.
        if raw_target_x < -0.10 * w or raw_target_x > 1.10 * w:
            return LaneResult(False, w / 2.0, 0.0, 0.0, self._front(forbidden), labels)
        target_x = float(np.clip(raw_target_x, 0.0, float(w - 1)))
        heading = float(np.clip(fit[0] * h / max(1.0, w), -1.0, 1.0))
        support = float(np.count_nonzero(road[look_y:near_y + 1])) / max(1, road[look_y:near_y + 1].size)
        confidence = float(np.clip(0.5 * min(1.0, len(xs) / 500.0) + 0.5 * min(1.0, support / 0.5), 0.0, 1.0))
        return LaneResult(True, target_x, heading, confidence, self._front(forbidden), labels)

    @staticmethod
    def _front(mask: np.ndarray) -> float:
        h, w = mask.shape[:2]
        roi = mask[int(h * 0.68):, int(w * 0.30):int(w * 0.70)]
        return float(np.count_nonzero(roi)) / max(1, roi.size)
