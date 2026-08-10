#!/usr/bin/env python3
"""Export the trained semantic checkpoint to a static Nano-friendly ONNX graph."""
import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "track_yolo26n_sem_best.pt")
    parser.add_argument("--imgsz", type=int, default=224)
    args = parser.parse_args()
    exported = YOLO(str(args.model), task="semantic").export(
        format="onnx", imgsz=args.imgsz, batch=1, dynamic=False,
        simplify=False, opset=13, device="cpu")
    print(f"Semantic ONNX: {Path(exported).resolve()}")


if __name__ == "__main__":
    main()
