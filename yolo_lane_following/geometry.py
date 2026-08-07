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
    ys, xs = np.nonzero(mask)
    keep = ys >= top_y
    xs, ys = xs[keep], ys[keep]
    if xs.size < min_pixels or np.unique(ys).size < 6:
        return None
    # Equalise row influence so a thick blob at the bottom cannot dominate.
    sampled_y, sampled_x = [], []
    for y in np.unique(ys):
        row = xs[ys == y]
        sampled_y.append(y)
        sampled_x.append(float(np.median(row)))
    return np.polyfit(np.asarray(sampled_y), np.asarray(sampled_x), 2)


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
        component = labels == label
        ys, xs = np.nonzero(component)
        lower = ys >= np.percentile(ys, 70)
        near_x = float(np.median(xs[lower])) if np.any(lower) else float(centroids[label, 0])
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
