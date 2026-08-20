"""Low-rate optional sign detector with a latched majority vote."""
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SignEstimate:
    raw: Optional[str]
    confidence: float
    locked: Optional[str]


class SignPerception:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.history = deque(maxlen=max(1, int(cfg.get("vote_window", 7))))
        self.locked: Optional[str] = None
        self.last = SignEstimate(None, 0.0, None)
        self.model = None
        path = cfg.get("model")
        if path:
            from ultralytics import YOLO
            self.model = YOLO(path)

    def clear(self) -> None:
        self.history.clear(); self.locked = None; self.last = SignEstimate(None, 0.0, None)

    def update(self, frame: np.ndarray) -> SignEstimate:
        if self.locked:
            self.last = SignEstimate(self.locked, 1.0, self.locked)
            return self.last
        if self.model is None:
            self.last = SignEstimate(None, 0.0, None)
            return self.last
        crop = frame[:int(frame.shape[0] * float(self.cfg.get("roi_bottom", .65)))]
        result = self.model.predict(crop, imgsz=int(self.cfg.get("imgsz", 320)), verbose=False)[0]
        raw, conf = None, 0.0
        if result.boxes is not None and len(result.boxes):
            index = int(np.argmax(result.boxes.conf.detach().cpu().numpy()))
            conf = float(result.boxes.conf[index])
            if conf >= float(self.cfg.get("confidence", .55)):
                raw = str(result.names[int(result.boxes.cls[index])]).upper()
        if raw:
            self.history.append(raw)
        if self.history:
            label, votes = Counter(self.history).most_common(1)[0]
            if votes >= int(self.cfg.get("vote_min", 4)):
                self.locked = label
        self.last = SignEstimate(raw, conf, self.locked)
        return self.last
