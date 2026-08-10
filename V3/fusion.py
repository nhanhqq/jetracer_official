"""Divider-first target fusion. Segmentation constrains identity and safety."""
from dataclasses import dataclass
import numpy as np

@dataclass
class Target:
    x: float; y: float; confidence: float; heading: float; source: str; corrected: bool

def fuse(waypoint, divider, geometry, width, height, cfg):
    wx, wy, wc = waypoint; wx = float(np.clip(wx, 0., 1.)); wy = float(np.clip(wy, .42, .9))
    reliable = divider.confidence >= cfg.get('divider_min_confidence', .32)
    # Divider wins whenever visible. The waypoint is only a stabilizer for jitter.
    if reliable:
        dx = divider.x / float(width)
        # Do not let the old waypoint pull the car toward the outer lane.
        # Once divider segmentation is reliable, it owns lateral target x.
        x = dx
        source = 'divider_first'
        conf = min(1., divider.confidence)
        heading = divider.heading
    else:
        x, source, conf, heading = wx, 'waypoint_fallback', wc * .65, 0.
    corrected = False
    # White/outside is an absolute forbidden region; road geometry is a safety veto.
    if geometry is not None:
        if geometry.white_center >= cfg.get('white_hard_ratio', .32):
            x = geometry.center_x / float(width); source += '+white_recover'; conf *= .35; corrected = True
        elif geometry.white_left >= cfg.get('white_side_ratio', .12) and x < .5:
            road_center = geometry.center_x / float(width)
            x = .35 * x + .65 * road_center; source += '+white_left_recover'; conf *= .72; corrected = True
        elif geometry.white_right >= cfg.get('white_side_ratio', .12) and x > .5:
            road_center = geometry.center_x / float(width)
            x = .35 * x + .65 * road_center; source += '+white_right_recover'; conf *= .72; corrected = True
        elif not geometry.valid:
            conf *= .35; source += '+road_uncertain'
        elif x < geometry.left_x / width or x > geometry.right_x / width:
            x = float(np.clip(x, geometry.left_x/width+.03, geometry.right_x/width-.03))
            source += '+corridor_veto'; corrected = True
        if geometry.obstacle >= cfg.get('obstacle_ratio_caution', .08):
            # obstacle_offset > 0 means obstacle is right of car; detour left.
            shift = -.16 if geometry.obstacle_offset > .03 else .16 if geometry.obstacle_offset < -.03 else 0.
            if shift:
                x = float(np.clip(x + shift, geometry.left_x/width+.04, geometry.right_x/width-.04))
                source += '+obstacle_avoid'; conf *= .82; corrected = True
    return Target(float(np.clip(x, .03, .97)), wy, float(np.clip(conf,0,1)), heading, source, corrected)
