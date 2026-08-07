#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="lane_yolo26n_seg")
    args = parser.parse_args()
    model = YOLO(args.model)
    result = model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                         batch=args.batch, device=args.device, workers=args.workers, cache=True,
                         project=str(ROOT / "runs"), name=args.name, exist_ok=True,
                         patience=15, seed=2608, degrees=4, translate=0.08, scale=0.25,
                         hsv_h=0.01, hsv_s=0.25, hsv_v=0.30, mosaic=0.2, close_mosaic=10)
    best = Path(result.save_dir) / "weights" / "best.pt"
    artifacts = ROOT / "artifacts"; artifacts.mkdir(exist_ok=True)
    shutil.copy2(best, artifacts / "lane_yolo26n_seg_best.pt")
    print("Best model:", artifacts / "lane_yolo26n_seg_best.pt")
    print("Export on the target Jetson: python export_tensorrt.py")


if __name__ == "__main__":
    main()
