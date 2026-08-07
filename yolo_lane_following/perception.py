from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from .config import resolve_path
from .geometry import LaneEstimate, estimate_lane, obstacle_risk


@dataclass
class PerceptionResult:
    lane: LaneEstimate
    obstacle_risk: float
    obstacle_boxes: List[List[float]]
    masks: Dict[str, np.ndarray]
    annotated: np.ndarray


class YoloPerception:
    """YOLO26 segmentation + detection adapter with no application-side NMS."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        self.cfg = cfg
        model_cfg = cfg["models"]
        lane_path = resolve_path(cfg, model_cfg["lane"])
        fallback = resolve_path(cfg, model_cfg["lane_pt_fallback"])
        if not lane_path.exists():
            lane_path = fallback
        if not lane_path.exists():
            raise FileNotFoundError(
                f"Missing lane segmentation model: {lane_path}. "
                "Run train_segmentation.py first."
            )
        self.lane_model = YOLO(str(lane_path), task="segment")
        self.imgsz = int(model_cfg["imgsz"])
        self.device = model_cfg["device"]
        self.class_groups = cfg["classes"]
        self.confidence_alpha = float(cfg["tracking"].get("confidence_ema", 1.0))
        self._confidence_ema = None

    @staticmethod
    def _names(result) -> Dict[int, str]:
        names = result.names
        return names if isinstance(names, dict) else dict(enumerate(names))

    def _collect_masks(self, result, shape: Sequence[int]) -> Dict[str, np.ndarray]:
        h, w = shape[:2]
        grouped = {key: np.zeros((h, w), np.uint8) for key in self.class_groups}
        if result.masks is None or result.boxes is None:
            return grouped
        names = self._names(result)
        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for raw_mask, cls_id in zip(masks, classes):
            name = str(names.get(int(cls_id), "")).lower()
            resized = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
            for group, accepted in self.class_groups.items():
                if name in {str(v).lower() for v in accepted}:
                    grouped[group][resized] = 255
        return grouped

    def infer(self, frame: np.ndarray) -> PerceptionResult:
        m = self.cfg["models"]
        # TensorRT precision is fixed at export time.  Passing ``half`` to the
        # current YOLO26 predictor is deprecated and emits a warning per frame.
        lane_result = self.lane_model.predict(
            frame, imgsz=self.imgsz, conf=float(m["confidence"]), device=self.device,
            verbose=False,
        )[0]
        masks = self._collect_masks(lane_result, frame.shape)
        boxes: List[List[float]] = []
        # Obstacles labelled in the segmentation set are preferred.
        if np.any(masks["obstacle"]):
            count, _, stats, _ = cv2.connectedComponentsWithStats((masks["obstacle"] > 0).astype(np.uint8), 8)
            for x, y, bw, bh, area in stats[1:count]:
                if area >= 20:
                    boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
        t = self.cfg["tracking"]
        lane = estimate_lane(masks["divider"], masks["road"], t["lookahead_ratio"],
                             t["bottom_ratio"], t["roi_top_ratio"], t["min_mask_pixels"],
                             t["target_mode"])
        if lane.valid:
            alpha = float(np.clip(self.confidence_alpha, 0.0, 1.0))
            self._confidence_ema = (lane.confidence if self._confidence_ema is None else
                                    alpha * lane.confidence + (1.0 - alpha) * self._confidence_ema)
            lane.confidence = float(self._confidence_ema)
        else:
            self._confidence_ema = None
        risk = obstacle_risk(boxes, frame.shape[1], frame.shape[0], lane.near_x)
        annotated = lane_result.plot()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        colour = (0, 255, 0) if lane.valid else (0, 0, 255)
        cv2.line(annotated, (frame.shape[1] // 2, frame.shape[0]),
                 (int(lane.target_x), int(frame.shape[0] * t["lookahead_ratio"])), colour, 2)
        return PerceptionResult(lane, risk, boxes, masks, annotated)
