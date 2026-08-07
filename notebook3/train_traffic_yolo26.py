"""Fine-tune YOLO26n and export the best checkpoint to ONNX."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "traffic_sign_dataset" / "data.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    parser.add_argument("--name", default="traffic_yolo26n")
    args = parser.parse_args()

    model = YOLO(args.model)
    result = model.train(
        data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        project=str(args.project.resolve()), name=args.name, exist_ok=True,
        pretrained=True, patience=8, cache=True, seed=2608,
        degrees=5.0, translate=0.08, scale=0.30, fliplr=0.5,
        hsv_h=0.01, hsv_s=0.35, hsv_v=0.25, mosaic=0.35, close_mosaic=5,
        plots=True, verbose=True,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    best_model = YOLO(best)
    metrics = best_model.val(data=str(args.data.resolve()), imgsz=args.imgsz, device=args.device)
    onnx = Path(best_model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False))

    output = ROOT / "artifacts" / "traffic_detector"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output / "traffic_yolo26n_best.pt")
    shutil.copy2(onnx, output / "traffic_yolo26n.onnx")
    summary = {
        "classes": best_model.names,
        "imgsz": args.imgsz,
        "epochs_requested": args.epochs,
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"ONNX: {output / 'traffic_yolo26n.onnx'}")


if __name__ == "__main__":
    main()
