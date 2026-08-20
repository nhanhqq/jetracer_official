"""Extract connected L/S/R exits from the semantic drivable mask in BEV."""
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class BranchEstimate:
    available: Dict[str, bool]
    scores: Dict[str, float]
    component: np.ndarray
    valid: bool


class BranchExtractor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        matrix = cfg.get("homography")
        self.H: Optional[np.ndarray] = (np.asarray(matrix, dtype=np.float32).reshape(3, 3)
                                        if matrix is not None else None)

    def update(self, road: np.ndarray, divider: np.ndarray, forbidden: np.ndarray,
               crosswalk: np.ndarray) -> BranchEstimate:
        empty = np.zeros_like(road, dtype=np.uint8)
        scores = {"left": 0.0, "straight": 0.0, "right": 0.0}
        if self.H is None:
            return BranchEstimate({k: False for k in scores}, scores, empty, False)
        h, w = road.shape
        drivable = ((road > 0) | (divider > 0) | (crosswalk > 0)).astype(np.uint8)
        # Crosswalk is deliberately restored after removing hard forbidden.
        drivable[(forbidden > 0) & (crosswalk == 0)] = 0
        size = (int(self.cfg.get("bev_width", w)), int(self.cfg.get("bev_height", h)))
        bev = cv2.warpPerspective(drivable, self.H, size, flags=cv2.INTER_NEAREST)
        bev = cv2.morphologyEx(bev, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        bh, bw = bev.shape
        labels_count, labels, _stats, _ = cv2.connectedComponentsWithStats(bev, 8)
        # Component touching the car anchor at bottom-centre; search nearby so
        # calibration does not require the anchor to land on an exact pixel.
        anchor = labels[max(0, bh - int(bh * .12)):bh, int(bw * .40):int(bw * .60)]
        ids, counts = np.unique(anchor[anchor > 0], return_counts=True)
        if not len(ids):
            return BranchEstimate({k: False for k in scores}, scores, empty, False)
        label = int(ids[np.argmax(counts)])
        component = (labels == label).astype(np.uint8)
        # ROIs are deliberately high/outer: mere road at the car does not
        # qualify as an exit. Scores are occupancy of a connected component.
        rois = {"left": (slice(int(.25 * bh), int(.72 * bh)), slice(0, int(.32 * bw))),
                "straight": (slice(0, int(.42 * bh)), slice(int(.36 * bw), int(.64 * bw))),
                "right": (slice(int(.25 * bh), int(.72 * bh)), slice(int(.68 * bw), bw))}
        for name, (ys, xs) in rois.items():
            area = max(1, component[ys, xs].size)
            scores[name] = float(np.count_nonzero(component[ys, xs])) / area
        threshold = float(self.cfg.get("branch_min_score", .18))
        return BranchEstimate({k: v >= threshold for k, v in scores.items()}, scores, component, True)
