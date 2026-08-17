"""Perspective-aware local corridor and receding-horizon trajectory."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class RoadGeometry:
    valid: bool
    points: list
    left: list
    right: list
    center_x: float
    heading: float
    curvature: float
    confidence: float
    occupancy: float
    white_left: float
    white_right: float
    white_center: float
    divider_x: float
    divider_confidence: float
    obstacle: float
    obstacle_offset: float
    warning: str


def _row_center(mask, y):
    xs = np.flatnonzero(mask[y] > 0)
    return (float(xs[0]), float(xs[-1])) if xs.size else None


def estimate_geometry(road, outside, marking, cfg, obstacle=None):
    h, w = road.shape[:2]; ys = [int(v * h) for v in cfg['bands']]
    left, right, centers, widths = [], [], [], []
    for y in ys:
        span = _row_center(road, min(h - 1, y))
        if span:
            left.append((y, span[0])); right.append((y, span[1])); centers.append((y, sum(span) / 2.0)); widths.append(span[1] - span[0])
    near = float(np.count_nonzero(road[int(h*.72):]) / max(1, road[int(h*.72):].size))
    occupancy = float(np.count_nonzero(road[int(h*.48):]) / max(1, road[int(h*.48):].size))
    if not centers or widths[-1] < 20:
        return RoadGeometry(False, [], left, right, w/2, 0, 1, 0, occupancy, 0, 0, 0, w/2, 0, 0, 0, 'ROAD_LOST')
    # Width should decrease toward the horizon; a mild violation only lowers confidence.
    shrink_ok = all(a >= b * (1.0 - cfg.get('width_shrink_tolerance', .2)) for a, b in zip(widths[:-1], widths[1:]))
    yy = np.asarray([p[0] for p in centers], dtype=np.float64) / max(1, h)
    xx = np.asarray([p[1] for p in centers], dtype=np.float64) / max(1, w)
    fit = np.polyfit(yy, xx, 2) if len(centers) >= 3 else np.polyfit(yy, xx, 1)
    points = [(y, float(np.clip(np.polyval(fit, y / float(h)) * w, 0, w - 1))) for y in ys]
    heading = float(np.clip((points[0][1] - points[-1][1]) / max(1, ys[-1]-ys[0]), -1, 1))
    curvature = float(np.clip(abs(fit[0]) if len(fit) == 3 else 0, 0, 1))
    mark = (marking > 0).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mark, 8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= cfg.get('min_boundary_pixels', 8) and bh >= 5:
            score = bh + min(area, 80) - abs(cents[i][0] - w/2) * .2
            if best is None or score > best[0]: best = (score, i)
    divider_x, divider_conf = w/2, 0.0
    if best:
        yy, xx = np.nonzero(labels == best[1]); coef = np.polyfit(yy, xx, 1)
        divider_x = float(np.clip(np.polyval(coef, int(h*.64)), 0, w-1)); divider_conf = float(min(1, best[0]/160))
    outside_bin = outside > 0
    left_white = float(np.mean(outside_bin[int(h*.60):, :int(w*.45)]))
    right_white = float(np.mean(outside_bin[int(h*.60):, int(w*.55):]))
    center_white = float(np.mean(outside_bin[int(h*.72):, int(w*.35):int(w*.65)]))
    safe = 1.0 - min(1.0, (left_white + right_white + center_white) * 2.0)
    confidence = float(np.clip(.35 * min(1, len(centers)/len(ys)) + .30 * safe + .20 * (1 if shrink_ok else .25) + .15 * min(1, near/.18), 0, 1))
    obstacle_ratio = 0.0
    if obstacle is not None:
        near_obstacle = obstacle[int(h*.52):int(h*.86)]
        obstacle_ratio = float(np.count_nonzero(near_obstacle) / max(1, near_obstacle.size))
    obstacle_offset = 0.0
    if obstacle_ratio > 0 and obstacle is not None:
        oy, ox = np.nonzero(obstacle[int(h*.52):int(h*.86)] > 0)
        if ox.size:
            obstacle_offset = float(np.clip(np.mean(ox) / float(w) - .5, -.5, .5))
    warning = 'OBSTACLE' if obstacle_ratio > cfg.get('obstacle_ratio_caution', .08) else ('' if shrink_ok else 'PERSPECTIVE_WARNING')
    return RoadGeometry(True, points, left, right, points[-2][1], heading, curvature, confidence, occupancy, left_white, right_white, center_white, divider_x, divider_conf, obstacle_ratio, obstacle_offset, warning)
