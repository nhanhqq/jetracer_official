#!/usr/bin/env python3
"""Train the nano YOLO26 semantic model for the physical track."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "semantic_dataset" / "data.yaml")
    parser.add_argument("--model", default="yolo26n-sem.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0,
                        help="0 avoids shared-memory DataLoader failures on Jetson/containers")
    parser.add_argument("--name", default="track_yolo26n_sem")
    parser.add_argument("--output-name", default="track_yolo26n_sem_best.pt",
                        help="artifact filename for the selected best checkpoint")
    parser.add_argument("--cache", action="store_true",
                        help="cache images in RAM; leave off on Jetson Nano")
    args = parser.parse_args()
    model = YOLO(args.model)
    result = model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                         batch=args.batch, device=args.device, workers=args.workers, cache=args.cache,
                         project=str(ROOT / "runs"), name=args.name, exist_ok=True, patience=20,
                         seed=2608, degrees=4, translate=0.08, scale=0.22, hsv_h=0.01,
                         hsv_s=0.20, hsv_v=0.25, mosaic=0.15, close_mosaic=10)
    best = Path(result.save_dir) / "weights" / "best.pt"
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    output = artifacts / args.output_name
    shutil.copy2(best, output)
    print("Best semantic model:", output)


if __name__ == "__main__":
    main()
