"""Per-frame white safety with decimated neural segmentation."""
import cv2
from V2.pseudo_label import make_masks

class Perception:
    def __init__(self, cfg):
        self.cfg=cfg; self.segmenter=__import__('V2.segmentation',fromlist=['Segmenter']).Segmenter(cfg)
        self.interval=max(1,int(cfg.get('runtime',{}).get('segmentation_interval',2))); self.index=0; self.cached=None; self.mode='cv'
    def infer(self, frame):
        cv_size=int(self.cfg.get('runtime',{}).get('cv_safety_size', frame.shape[1]))
        if cv_size > 0 and frame.shape[1] != cv_size:
            cv_frame=cv2.resize(frame,(cv_size,cv_size),interpolation=cv2.INTER_AREA)
            small_masks=make_masks(cv_frame)
            cv_road,cv_outside,cv_marking=[cv2.resize(mask,(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST) for mask in small_masks]
        else:
            cv_road, cv_outside, cv_marking=make_masks(frame)
        if self.segmenter.neural is None:
            self.index += 1
            self.segmenter.obstacle=cv_road*0
            return (cv_road,cv_outside,cv_marking), self.mode
        refresh=self.cached is None or self.index % self.interval == 0
        if refresh:
            (road, outside, marking), self.mode=self.segmenter.infer(frame)
            self.cached=(road.copy(), marking.copy(), self.segmenter.obstacle.copy())
        else:
            road, marking, obstacle=self.cached
            self.segmenter.obstacle=obstacle.copy()
        self.index+=1
        # White is recomputed every frame and always wins over stale/neural masks.
        if not refresh: road, marking, obstacle=self.cached
        road=cv2.bitwise_and(road, cv2.bitwise_not(cv_outside))
        outside=cv_outside
        marking=cv2.bitwise_and(marking, cv2.bitwise_not(cv_outside))
        self.segmenter.obstacle=cv2.bitwise_and(self.segmenter.obstacle, road)
        return (road,outside,marking), self.mode
