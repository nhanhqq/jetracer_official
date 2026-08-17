#!/usr/bin/env python3
"""Verify/copy the lane ONNX artifact used by Smart City."""
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=ROOT.parent / "yolo_lane_following/artifacts/track_yolo26n_sem_best.onnx")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/smart_city_lane.onnx")
    args = parser.parse_args()
    if not args.source.exists():
        raise SystemExit("Missing lane ONNX: %s" % args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(args.source.read_bytes())
    print("lane_onnx:", args.output.resolve())


if __name__ == "__main__":
    main()

