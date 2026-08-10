"""Fast, temporal divider extraction; never substitutes road centre for divider."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class DividerPath:
    points: list
    x: float
    heading: float
    confidence: float
    source: str

class DividerTracker:
    def __init__(self, cfg):
        self.cfg = cfg; self.last = None; self.last_conf = 0.0; self.last_heading = 0.; self.misses = 0

    @staticmethod
    def _fit(yy, xx, h, w):
        """Fit normalized y->x; quadratic captures long tight bends."""
        z = yy.astype(np.float64) / float(h) - .65
        xn = xx.astype(np.float64) / float(w)
        degree = 2 if np.unique(yy).size >= 8 and yy.size >= 16 else 1
        return np.polyfit(z, xn, degree), degree

    @staticmethod
    def _eval(coef, degree, y, h, w):
        return float(np.polyval(coef, float(y) / float(h) - .65) * w)

    def _candidates(self, marking, road, outside=None):
        mask = (marking > 0).astype(np.uint8)
        if outside is not None:
            mask[outside > 0] = 0
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        h, w = mask.shape; out = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < self.cfg.get('divider_min_pixels', 8) or bh < 5: continue
            yy, xx = np.nonzero(labels == i)
            if yy.size >= 2:
                coef, degree = self._fit(yy, xx, h, w)
            else:
                coef, degree = np.asarray([float(cents[i][0] / w)]), 0
            # Long, vertically persistent and forward-facing markings score best.
            score = min(1., bh / (h * .55)) * .58 + min(1., area / 90.) * .27
            score += min(1., max(0., 1. - abs(cents[i][0] / w - .5) * 1.3)) * .15
            out.append((score, i, coef, degree, yy, xx))
        return out

    def update(self, marking, road, outside=None):
        h, w = marking.shape[:2]; candidates = self._candidates(marking, road, outside)
        best = None
        for item in candidates:
            score, _, coef, degree, yy, xx = item
            x_near = self._eval(coef, degree, h * .78, h, w)
            x_far = self._eval(coef, degree, h * .46, h, w)
            if self.last is not None and abs(x_near - self.last) > w * self.cfg.get('divider_max_jump', .22):
                score *= .25
            if best is None or score > best[0]: best = (score, coef, degree, yy, xx)
        if best is None:
            self.misses += 1; self.last_conf *= .82
            if self.last is not None and self.misses <= 3:
                return DividerPath([], self.last, self.last_heading, self.last_conf, 'temporal_hold')
            return DividerPath([], self.last if self.last is not None else w*.5, self.last_heading, self.last_conf, 'lost')
        score, coef, degree, yy, xx = best
        # Use a fixed lookahead path, not only the bottom-most pixel.
        bands = [int(v*h) for v in self.cfg.get('bands', [.46,.56,.66,.78,.90])]
        points = [(y, float(np.clip(self._eval(coef, degree, y, h, w), 0, w-1))) for y in bands]
        x = points[2][1] if len(points) > 2 else points[-1][1]
        heading = float(np.clip((points[0][1] - points[-1][1]) / max(1., bands[-1]-bands[0]), -1., 1.))
        conf = float(np.clip(score * (.75 + .25 * min(1., len(np.unique(yy))/max(1.,len(bands)))), 0., 1.))
        source = 'segmentation'
        threshold = self.cfg.get('divider_min_confidence', .32)
        if self.last is not None and conf < threshold:
            previous_x, previous_conf = self.last, self.last_conf
            x = .75 * previous_x + .25 * x
            # Do not discard a good track because one neural mask is weak.
            conf = max(conf, previous_conf * .82)
            heading = .75 * self.last_heading + .25 * heading
            source = 'temporal_hold'
        self.misses = 0; self.last, self.last_conf, self.last_heading = x, conf, heading
        return DividerPath(points, x, heading, conf, source)
