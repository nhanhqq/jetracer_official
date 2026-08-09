"""Waypoint validation and correction against the local drivable corridor."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FusedTarget:
    x: float
    y: float
    confidence: float
    source: str
    original_x: float
    corrected: bool


def fuse(waypoint, geometry, width, height):
    wx, wy, wc = waypoint
    wx = float(np.clip(wx, 0, 1)); wy = float(np.clip(wy, .35, 1))
    original = wx
    target_y = int(np.clip(wy * height, height*.45, height*.93))
    if not geometry.valid:
        return FusedTarget(.5, wy, min(wc, .15), 'road_lost', original, False)
    road_x = geometry.center_x / width
    if geometry.points:
        road_x = min(1, max(0, geometry.points[-1][1] / width))
    # Waypoint outside or too close to a boundary is rejected as likely outer edge.
    inside = any(y >= target_y and abs(x - wx*width) <= max(8, width*.08) for y, x in geometry.points)
    white_side = geometry.white_left > .08 and wx < .5 or geometry.white_right > .08 and wx > .5
    bad = not inside or white_side or abs(wx - road_x) > .30
    if bad:
        alpha = .22 if geometry.divider_confidence < .55 else .38
        x = alpha * wx + (1-alpha) * road_x
        return FusedTarget(float(np.clip(x, .05, .95)), wy, float(min(wc*.55, geometry.confidence)), 'road_corrected', original, True)
    return FusedTarget(wx, wy, float(min(1, .55*wc + .45*geometry.confidence)), 'waypoint', original, False)

