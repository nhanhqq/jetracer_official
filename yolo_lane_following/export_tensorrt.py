#!/usr/bin/env python3
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
model = ROOT / "artifacts" / "lane_yolo26n_seg_best.pt"
if not model.exists():
    raise SystemExit(f"Missing {model}; train the segmentation model first")
exported = YOLO(str(model)).export(format="engine", imgsz=224, batch=1, half=True,
                                   dynamic=False, nms=False, workspace=2)
print("TensorRT engine:", exported)
