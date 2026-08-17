"""Small adapter keeping the validated V2 perception backends out of control."""
import cv2
from V2.geometry import estimate_geometry
from V3.perception import Perception

def perceive(frame, perception, cfg):
    (road, outside, marking), mode = perception.infer(frame)
    geom = estimate_geometry(road, outside, marking, cfg['geometry'], perception.segmenter.obstacle)
    if geom.left:
        geom.left_x = geom.left[-1][1]
    else:
        geom.left_x = 0.
    if geom.right:
        geom.right_x = geom.right[-1][1]
    else:
        geom.right_x = frame.shape[1]-1.
    return (road, outside, marking), geom, mode

def resize(frame, cfg):
    return cv2.resize(frame, (int(cfg['camera']['width']), int(cfg['camera']['height'])))
