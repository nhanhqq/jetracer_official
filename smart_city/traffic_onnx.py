from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


class TrafficONNX:
    """Small compatibility wrapper for common Ultralytics YOLO ONNX outputs."""

    def __init__(self, model_path: Path, class_names: Dict[int, str],
                 confidence: float = 0.55, input_size: int = 224):
        import onnxruntime as ort

        self.classes = class_names
        self.confidence = float(confidence)
        self.input_size = int(input_size)
        self.session = ort.InferenceSession(str(model_path),
                                             providers=["CUDAExecutionProvider",
                                                        "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> List[dict]:
        h, w = frame.shape[:2]
        image = cv2.cvtColor(cv2.resize(frame, (self.input_size, self.input_size)), cv2.COLOR_BGR2RGB)
        tensor = image.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        raw = np.asarray(self.session.run(None, {self.input_name: tensor})[0])
        raw = np.squeeze(raw)
        if raw.ndim != 2:
            return []
        if raw.shape[0] < raw.shape[1] and raw.shape[0] <= 64:
            raw = raw.T
        detections = []
        for row in raw:
            if row.size == 6:
                x1, y1, x2, y2, conf, class_id = row
            elif row.size >= 7:
                x, y, bw, bh = row[:4]
                scores = row[4:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                x1, y1, x2, y2 = x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2
            else:
                continue
            if float(conf) < self.confidence:
                continue
            # Exported coordinates may be input-size pixels or normalized.
            scale_x, scale_y = (w / float(self.input_size), h / float(self.input_size))
            values = np.array([x1, y1, x2, y2], dtype=np.float32)
            if np.max(np.abs(values)) <= 1.5:
                values[[0, 2]] *= w
                values[[1, 3]] *= h
            else:
                values[[0, 2]] *= scale_x
                values[[1, 3]] *= scale_y
            x1, y1, x2, y2 = values.tolist()
            detections.append({"label": self.classes.get(int(class_id), str(int(class_id))),
                               "class_id": int(class_id), "confidence": float(conf),
                               "bbox": [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)]})
        return detections

