from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .config import resolve_path
from .geometry import LaneEstimate, estimate_lane, obstacle_risk, plan_semantic_lane


@dataclass
class SemanticPerceptionResult:
    lane: LaneEstimate
    obstacle_risk: float
    obstacle_boxes: List[List[float]]
    masks: Dict[str, np.ndarray]
    annotated: np.ndarray
    forbidden_left: float = 0.0
    forbidden_right: float = 0.0
    escape_steering: float = 0.0


class YoloSemanticPerception:
    """Dense YOLO26 semantic output adapter for the track safety policy."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        model_cfg = cfg["models"]
        model_path = resolve_path(cfg, model_cfg["semantic"])
        if not model_path.exists():
            raise FileNotFoundError(f"Missing semantic model: {model_path}. Run train_semantic.py first.")
        self.model = YOLO(str(model_path), task="semantic")
        self.cfg = cfg
        self.imgsz = int(model_cfg["imgsz"])
        self.device = model_cfg["device"]
        self.class_ids = {str(name).lower(): int(class_id)
                          for name, class_id in cfg["semantic_classes"].items()}
        self.last_target_x = None

    def _masks(self, result, shape: tuple[int, int]) -> Dict[str, np.ndarray]:
        height, width = shape
        labels = result.semantic_mask.data.detach().cpu().numpy().astype(np.uint8)
        if labels.shape != (height, width):
            labels = cv2.resize(labels, (width, height), interpolation=cv2.INTER_NEAREST)
        return {name: ((labels == class_id).astype(np.uint8) * 255)
                for name, class_id in self.class_ids.items()}

    @staticmethod
    def _boxes(mask: np.ndarray) -> List[List[float]]:
        count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
        boxes = []
        for x, y, width, height, area in stats[1:count]:
            if area >= 20:
                boxes.append([float(x), float(y), float(x + width), float(y + height)])
        return boxes

    def infer(self, frame: np.ndarray) -> SemanticPerceptionResult:
        result = self.model.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        masks = self._masks(result, frame.shape[:2])
        t = self.cfg["tracking"]
        divider_lane = estimate_lane(masks["divider"], masks["road"],
                                     t["lookahead_ratio"], t["bottom_ratio"],
                                     t["roi_top_ratio"], t["min_mask_pixels"], "divider")
        lane = plan_semantic_lane(masks["divider"], masks["road"], masks["forbidden"], masks["obstacle"],
                                  t["lookahead_ratio"], t["bottom_ratio"], t["roi_top_ratio"],
                                  t["min_mask_pixels"], t["vehicle_half_width"])
        if lane.valid:
            if self.last_target_x is not None:
                lane.target_x = float(np.clip(lane.target_x, self.last_target_x - 28, self.last_target_x + 28))
            self.last_target_x = lane.target_x
        else:
            self.last_target_x = None
        boxes = self._boxes(masks["obstacle"])
        # Risk is measured against the divider corridor, not the temporary
        # avoidance target.  Otherwise selecting an escape route hides the
        # obstacle from the controller on the same frame.
        risk = obstacle_risk(boxes, frame.shape[1], frame.shape[0], divider_lane.near_x)
        roi = masks["forbidden"][int(frame.shape[0] * t["roi_top_ratio"]):]
        mid = roi.shape[1] // 2
        forbidden_left = float(np.count_nonzero(roi[:, :mid])) / max(1, roi[:, :mid].size)
        forbidden_right = float(np.count_nonzero(roi[:, mid:])) / max(1, roi[:, mid:].size)
        safe = ((masks["road"] > 0) & (masks["forbidden"] == 0) &
                (masks["obstacle"] == 0))
        low = safe[int(frame.shape[0] * t["lookahead_ratio"]):]
        left_clear = int(np.count_nonzero(low[:, :mid]))
        right_clear = int(np.count_nonzero(low[:, mid:]))
        escape_steering = 1.0 if right_clear > left_clear else (-1.0 if left_clear > right_clear else 0.0)
        overlay = frame.copy()
        colours = {"road": (70, 160, 70), "divider": (0, 110, 255),
                   "forbidden": (255, 80, 220), "obstacle": (0, 0, 255)}
        for name, colour in colours.items():
            overlay[masks[name] > 0] = colour
        annotated = cv2.addWeighted(frame, 0.60, overlay, 0.40, 0)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        colour = (0, 255, 0) if lane.valid else (0, 0, 255)
        cv2.line(annotated, (frame.shape[1] // 2, frame.shape[0]),
                 (int(lane.target_x), int(frame.shape[0] * t["lookahead_ratio"])), colour, 2)
        return SemanticPerceptionResult(lane, risk, boxes, masks, annotated,
                                        forbidden_left, forbidden_right, escape_steering)
