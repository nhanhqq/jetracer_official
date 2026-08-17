#!/usr/bin/env python3
"""Train and export the Smart City traffic detector with YOLO26.

The lane model is reused from ``yolo_lane_following`` and is not retrained by
this script. Dataset layout follows the standard YOLO detection format.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    parser.add_argument("--model", type=Path,
                        default=ROOT.parent / "yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="smart_city_yolo26n")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "smart_city_yolo26n.onnx")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="checkpoint destination; defaults beside --output")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit("Missing dataset yaml: %s" % args.data)
    if not args.model.exists():
        raise SystemExit("Missing YOLO26 weight: %s" % args.model)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    result = model.train(data=str(args.data.resolve()), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                workers=args.workers, name=args.name, project=str(ROOT / "runs"),
                exist_ok=True, patience=20, seed=2608)
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit("Training finished but best checkpoint is missing: %s" % best)
    exported = YOLO(str(best)).export(format="onnx", imgsz=args.imgsz,
                                      batch=1, opset=13, simplify=False,
                                      nms=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exported_path = Path(str(exported))
    args.output.write_bytes(exported_path.read_bytes())
    checkpoint = args.checkpoint or (args.output.parent / (args.output.stem + "_best.pt"))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, checkpoint)
    print("best_pt:", checkpoint.resolve())
    print("traffic_onnx:", args.output.resolve())


if __name__ == "__main__":
    main()
