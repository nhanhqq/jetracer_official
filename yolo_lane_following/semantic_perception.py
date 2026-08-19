from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import resolve_path
from .geometry import LaneEstimate, estimate_lane, plan_semantic_lane


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
    forbidden_front: float = 0.0


class ConsecutiveRiskGate:
    """Require a persistent on-path obstacle before changing the corridor."""

    def __init__(self, required_frames: int):
        self.required_frames = max(1, int(required_frames))
        self.count = 0

    def update(self, active: bool) -> bool:
        self.count = self.count + 1 if active else 0
        return self.count >= self.required_frames


class YoloSemanticPerception:
    """Dense YOLO26 adapter for lane following.

    The model may still contain a legacy obstacle class in its output tensor,
    but this runtime deliberately ignores it. Lane following uses only road,
    divider and forbidden/shoulder masks.
    """

    def __init__(self, cfg: dict):
        model_cfg = cfg["models"]
        model_path = resolve_path(cfg, model_cfg["semantic"])
        if not model_path.exists():
            raise FileNotFoundError(f"Missing semantic model: {model_path}. Run train_semantic.py first.")

        from ultralytics import YOLO

        self.model = YOLO(str(model_path), task="semantic")
        self.cfg = cfg
        self.imgsz = int(model_cfg["imgsz"])
        self.device = model_cfg["device"]
        self.class_ids = {str(name).lower(): int(class_id)
                          for name, class_id in cfg["semantic_classes"].items()}
        self.last_target_x = None

    def warmup(self) -> None:
        """Initialize the lazy backend before camera callbacks or motor gating."""
        height = int(self.cfg["camera"]["height"])
        width = int(self.cfg["camera"]["width"])
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        self.model.predict(blank, imgsz=self.imgsz, device=self.device, verbose=False)

    def _masks(self, result, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
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
        # Keep the planner contract stable, but pass an empty obstacle mask so
        # no obstacle prediction can alter the target or controller state.
        no_obstacles = np.zeros_like(masks["road"])
        lane = plan_semantic_lane(masks["divider"], masks["road"], masks["forbidden"], no_obstacles,
                                  t["lookahead_ratio"], t["bottom_ratio"], t["roi_top_ratio"],
                                  t["min_mask_pixels"], t["vehicle_half_width"], divider_lane)
        if lane.valid:
            if self.last_target_x is not None:
                # Reject isolated segmentation jumps, but allow a real sharp
                # turn to enter the controller promptly (the old fixed 28px
                # clamp made the car react late at high speed).
                jump = float(t.get("max_target_jump", 48.0))
                lane.target_x = float(np.clip(lane.target_x,
                                              self.last_target_x - jump,
                                              self.last_target_x + jump))
            self.last_target_x = lane.target_x
        else:
            self.last_target_x = None
        boxes = []
        risk = 0.0
        roi = masks["forbidden"][int(frame.shape[0] * t["roi_top_ratio"]):]
        mid = roi.shape[1] // 2
        forbidden_left = float(np.count_nonzero(roi[:, :mid])) / max(1, roi[:, :mid].size)
        forbidden_right = float(np.count_nonzero(roi[:, mid:])) / max(1, roi[:, mid:].size)
        front_y1 = int(frame.shape[0] * float(t.get("front_roi_top_ratio", 0.62)))
        front_x1 = int(frame.shape[1] * float(t.get("front_roi_left_ratio", 0.30)))
        front_x2 = int(frame.shape[1] * float(t.get("front_roi_right_ratio", 0.70)))
        front = masks["forbidden"][front_y1:, front_x1:front_x2]
        forbidden_front = float(np.count_nonzero(front)) / max(1, front.size)
        safe = ((masks["road"] > 0) & (masks["forbidden"] == 0))
        low = safe[int(frame.shape[0] * t["lookahead_ratio"]):]
        left_clear = int(np.count_nonzero(low[:, :mid]))
        right_clear = int(np.count_nonzero(low[:, mid:]))
        min_hint = max(1, int(low[:, :mid].size * float(t.get("road_hint_min_ratio", 0.005))))
        if max(left_clear, right_clear) < min_hint:
            escape_steering = 0.0
        else:
            escape_steering = 1.0 if right_clear > left_clear else (-1.0 if left_clear > right_clear else 0.0)
        overlay = frame.copy()
        colours = {"road": (70, 160, 70), "divider": (0, 110, 255),
                   "forbidden": (255, 80, 220)}
        for name, colour in colours.items():
            overlay[masks[name] > 0] = colour
        annotated = cv2.addWeighted(frame, 0.60, overlay, 0.40, 0)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        colour = (0, 255, 0) if lane.valid else (0, 0, 255)
        cv2.line(annotated, (frame.shape[1] // 2, frame.shape[0]),
                 (int(lane.target_x), int(frame.shape[0] * t["lookahead_ratio"])), colour, 2)
        return SemanticPerceptionResult(lane, risk, boxes, masks, annotated,
                                        forbidden_left, forbidden_right, escape_steering,
                                        forbidden_front)
