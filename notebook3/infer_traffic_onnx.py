"""Small inference example for the exported traffic detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Image, video, folder, webcam index, or stream URL")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/traffic_detector/traffic_yolo26n.onnx")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    YOLO(args.model).predict(source=source, conf=args.conf, imgsz=args.imgsz, save=True)


if __name__ == "__main__":
    main()
