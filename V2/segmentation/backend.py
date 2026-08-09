"""Segmentation backend for V2.

YOLOv5 TensorRT is the primary perception branch. Classical CV is retained
only for the white-forbidden safety signal and as a road-lost fallback.
"""
import cv2
import numpy as np

from V2.pseudo_label import make_masks


class Segmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.neural = None
        self.mode = 'cv'
        self.obstacle = np.zeros((int(cfg['camera']['height']), int(cfg['camera']['width'])), np.uint8)
        models = cfg.get('models', {})
        backend = models.get('backend', 'cv')
        if backend == 'yolov5_tensorrt':
            try:
                from V2.segmentation.yolov5_trt import Yolov5SegTensorRT
                path = cfg['_root'] + '/' + models.get('segmentation_engine', '')
                self.neural = Yolov5SegTensorRT(path, models.get('input_size', 224), 3,
                                                models.get('segmentation_confidence', .25),
                                                models.get('segmentation_max_detections', 16))
                self.mode = 'yolov5-tensorrt'
            except (ImportError, IOError, OSError, RuntimeError) as exc:
                self.mode = 'cv_pending_yolov5_engine'
                self.error = str(exc)

    def infer(self, frame):
        cv_road, cv_outside, cv_marking = make_masks(frame)
        self.obstacle = np.zeros_like(cv_road)
        if self.neural is None:
            return (cv_road, cv_outside, cv_marking), self.mode
        try:
            road, divider, obstacle = self.neural.infer(frame)
            # White outside is a hard safety constraint. It is never replaced
            # by a learned prediction that could misclassify reflection.
            outside = cv_outside
            if np.count_nonzero(road) < int(self.cfg['geometry'].get('min_road_pixels', 180)):
                road = cv_road
                mode = 'yolov5-tensorrt+cv-road-fallback'
            else:
                mode = self.mode
            self.obstacle = cv2.bitwise_and(obstacle, road)
            return (road, outside, divider), mode
        except Exception:
            # Keep the safety path explicit: callers can see the mode and logs
            # can distinguish an engine failure from a neural result.
            self.obstacle = np.zeros_like(cv_road)
            return (cv_road, cv_outside, cv_marking), 'cv_fallback_error'
