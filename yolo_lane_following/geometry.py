from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class LaneEstimate:
    valid: bool
    target_x: float
    near_x: float
    heading_error: float
    curvature: float
    confidence: float
    source: str


def _fit_x(mask: np.ndarray, top_y: int, min_pixels: int) -> Optional[np.ndarray]:
    roi = mask[top_y:] > 0
    row_counts = np.count_nonzero(roi, axis=1)
    valid_rows = row_counts > 0
    if int(row_counts.sum()) < min_pixels or np.count_nonzero(valid_rows) < 6:
        return None
    # Equalise row influence so a thick blob at the bottom cannot dominate.
    # A vectorised row centroid is equivalent to the median for the thin,
    # connected divider strip and avoids hundreds of tiny NumPy median calls.
    x_coordinates = np.arange(mask.shape[1], dtype=np.float32)
    row_centres = np.sum(roi * x_coordinates, axis=1)[valid_rows] / row_counts[valid_rows]
    sampled_y = np.flatnonzero(valid_rows).astype(np.float32) + float(top_y)
    return np.polyfit(sampled_y, row_centres, 2)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def _path_component(mask: np.ndarray) -> np.ndarray:
    """Choose the marking approaching camera centre, not simply the largest blob."""
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary
    best_label, best_score = 0, -1e9
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < 12 or bh < 5:
            continue
        # Inspect only this component's lower bounding-box slice. Building a
        # full-frame boolean mask per label made cluttered CSI scenes scale as
        # O(number_of_components * image_pixels).
        lower_y = y + int(bh * 0.70)
        crop = labels[lower_y:y + bh, x:x + bw] == label
        crop_x = np.nonzero(crop)[1]
        near_x = (float(np.median(crop_x) + x) if crop_x.size
                  else float(centroids[label, 0]))
        bottom = float((y + bh) / h)
        coverage = float(bh / h)
        centre_penalty = abs(near_x - w / 2.0) / w
        score = 2.0 * bottom + coverage + min(area / 300.0, 1.0) - 2.2 * centre_penalty
        if score > best_score:
            best_label, best_score = label, score
    return (labels == best_label).astype(np.uint8) if best_label else binary


def estimate_lane(
    divider_mask: np.ndarray,
    road_mask: np.ndarray,
    lookahead_ratio: float = 0.62,
    bottom_ratio: float = 0.93,
    roi_top_ratio: float = 0.45,
    min_pixels: int = 35,
    target_mode: str = "divider",
) -> LaneEstimate:
    """Convert segmentation masks to a preview target in full-image coordinates."""
    if divider_mask.shape != road_mask.shape:
        raise ValueError("divider_mask and road_mask must have identical HxW shape")
    h, w = divider_mask.shape
    top_y = int(h * roi_top_ratio)
    look_y, near_y = int(h * lookahead_ratio), int(h * bottom_ratio)
    source, mask = "divider", _path_component(divider_mask)
    coeff = _fit_x(mask, top_y, min_pixels)

    if coeff is None and target_mode != "divider":
        # Fallback to the centre of the drivable component per image row.
        road = _largest_component(road_mask)
        centre = np.zeros_like(road)
        for y in range(top_y, h):
            xs = np.flatnonzero(road[y])
            if xs.size:
                centre[y, int(np.median(xs))] = 1
        coeff = _fit_x(centre, top_y, min_pixels)
        mask, source = road, "road_center"

    if coeff is None:
        return LaneEstimate(False, w / 2, w / 2, 0.0, 1.0, 0.0, "lost")

    target_x = float(np.clip(np.polyval(coeff, look_y), 0, w - 1))
    near_x = float(np.clip(np.polyval(coeff, near_y), 0, w - 1))
    heading = float(np.clip((target_x - near_x) / max(1.0, near_y - look_y), -1, 1))
    curvature = float(np.clip(abs(2.0 * coeff[0]) * h, 0, 1))
    roi_pixels = max(1, int(np.count_nonzero(mask[top_y:])))
    vertical_coverage = np.unique(np.nonzero(mask[top_y:])[0]).size / max(1, h - top_y)
    confidence = float(np.clip(0.75 * vertical_coverage + 0.25 * min(1, roi_pixels / 350), 0, 1))
    return LaneEstimate(True, target_x, near_x, heading, curvature, confidence, source)


def obstacle_risk(boxes: Sequence[Sequence[float]], width: int, height: int, path_x: float) -> float:
    """Return 0..1 proximity risk for boxes intersecting the planned path corridor."""
    best = 0.0
    corridor = width * 0.20
    for x1, y1, x2, y2 in boxes:
        centre_x = (x1 + x2) / 2.0
        half_width = max(0.0, (x2 - x1) / 2.0)
        if abs(centre_x - path_x) > corridor + half_width:
            continue
        bottom = np.clip(y2 / max(1.0, height), 0.0, 1.0)
        area = np.clip(((x2 - x1) * (y2 - y1)) / max(1.0, width * height), 0.0, 1.0)
        best = max(best, float(0.8 * bottom + 0.2 * min(1.0, area * 8.0)))
    return best


def _row_candidates(safe: np.ndarray, y: int, half_width: int) -> list[float]:
    """Return centres that have a complete vehicle-width support in one row."""
    row = safe[y] > 0
    candidates = []
    start = None
    for x, value in enumerate(row):
        if value and start is None:
            start = x
        if start is not None and (not value or x == len(row) - 1):
            end = x if value else x - 1
            if end - start + 1 >= half_width * 2 + 1:
                candidates.append((start + end) / 2.0)
            start = None
    return candidates


def plan_semantic_lane(
    divider_mask: np.ndarray,
    road_mask: np.ndarray,
    forbidden_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    lookahead_ratio: float = 0.62,
    bottom_ratio: float = 0.93,
    roi_top_ratio: float = 0.45,
    min_pixels: int = 35,
    vehicle_half_width: int = 13,
    divider_lane: Optional[LaneEstimate] = None,
) -> LaneEstimate:
    """Plan inside semantic road only, never using white forbidden pixels.

    The normal target is the orange divider.  When an obstacle blocks its
    corridor, a road-supported side target is selected.  Once clear, the normal
    divider target is returned automatically on the following frame.
    """
    lane = divider_lane
    if lane is None:
        lane = estimate_lane(divider_mask, road_mask, lookahead_ratio, bottom_ratio,
                             roi_top_ratio, min_pixels, target_mode="divider")
    if not lane.valid:
        return lane
    h, w = road_mask.shape
    look_y = int(np.clip(round(h * lookahead_ratio), 0, h - 1))
    near_y = int(np.clip(round(h * bottom_ratio), 0, h - 1))
    blocked = cv2.dilate(((forbidden_mask > 0) | (obstacle_mask > 0)).astype(np.uint8),
                         np.ones((vehicle_half_width * 2 + 1, vehicle_half_width * 2 + 1), np.uint8))
    # The orange divider is intentionally followed, so it is navigable.  Join
    # its thin semantic strip back to the road before checking vehicle width.
    safe = ((road_mask > 0) | (divider_mask > 0)).astype(np.uint8)
    safe = cv2.morphologyEx(safe, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    safe[blocked > 0] = 0
    # Require free space at both preview and near rows.  This rejects a path
    # that would start safely but run into a white shoulder or obstacle.
    candidates = [x for x in _row_candidates(safe, look_y, vehicle_half_width)
                  if any(abs(x - near) <= vehicle_half_width * 5
                         for near in _row_candidates(safe, near_y, vehicle_half_width))]
    if not candidates:
        return LaneEstimate(False, lane.target_x, lane.near_x, 0.0, lane.curvature, 0.0, "blocked")
    desired = lane.target_x
    corridor_x1 = max(0, int(round(desired - vehicle_half_width)))
    corridor_x2 = min(w, int(round(desired + vehicle_half_width + 1)))
    obstacle_blocks_divider = bool(np.any(obstacle_mask[look_y:near_y + 1, corridor_x1:corridor_x2]))
    forbidden_blocks_divider = bool(np.any(forbidden_mask[look_y:near_y + 1, corridor_x1:corridor_x2]))
    if not obstacle_blocks_divider:
        # Divider is the primary signal.  Its thin class often replaces road
        # pixels in the semantic map, so road support is not a valid reason to
        # abandon it.  White in the full vehicle corridor is a hard stop.
        if forbidden_blocks_divider:
            return LaneEstimate(False, lane.target_x, lane.near_x, 0.0, lane.curvature, 0.0, "blocked")
        return LaneEstimate(True, float(desired), lane.near_x, lane.heading_error,
                            lane.curvature, lane.confidence, "divider")

    # Divider corridor is blocked: choose the closest road-supported side with
    # a full vehicle-width clearance.  A crossing through white is impossible
    # because forbidden pixels were removed before candidates were made.
    side_candidates = [x for x in candidates if abs(x - desired) > vehicle_half_width * 2]
    if not side_candidates:
        return LaneEstimate(False, lane.target_x, lane.near_x, 0.0, lane.curvature, 0.0, "blocked")
    target = min(side_candidates, key=lambda x: abs(x - desired))
    avoiding = True
    near_candidates = _row_candidates(safe, near_y, vehicle_half_width)
    near = min(near_candidates, key=lambda x: abs(x - target))
    heading = float(np.clip((target - near) / max(1.0, near_y - look_y), -1, 1))
    confidence = lane.confidence * (0.78 if avoiding else 1.0)
    return LaneEstimate(True, float(target), float(near), heading, lane.curvature,
                        float(confidence), "avoid" if avoiding else "divider")
