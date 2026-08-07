#!/usr/bin/env python3
"""YOLOv8 apex detection and lane following for JetRacer.

The original ``road_following_A/apex`` images store the desired apex as
``<x>_<y>_<uuid>.jpg``.  This module turns those points into one small YOLO
bounding box per image (class ``apex``), fine-tunes an existing YOLOv8 weight,
and converts the detected box centre to a steering command.

Run from ``notebook3`` using its existing environment:

    ./venv/bin/python lanedetectionv3.py prepare
    ./venv/bin/python lanedetectionv3.py train --epochs 80
    ./venv/bin/python lanedetectionv3.py infer --source 0

``ultralytics`` is intentionally imported only when a YOLO command is used so
the module can still be imported in Jupyter to show a helpful error instead of
failing at notebook start-up.
"""

from __future__ import print_function

import argparse
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_DATASET = ROOT / "old_codes" / "road_following_A" / "apex"
DATASET_ROOT = ROOT / "yolo_lane_dataset"
DATA_YAML = DATASET_ROOT / "data.yaml"
BASE_WEIGHTS = ROOT.parent / "notebook2" / "best.pt"
RUNS_DIR = ROOT / "runs" / "yolo_lane"
BEST_WEIGHTS = RUNS_DIR / "weights" / "best.pt"
CLASS_NAME = "apex"


def _require_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Không tìm thấy ultralytics trong notebook3/venv. Cài một bản "
            "Ultralytics tương thích Jetson/Python 3.6 trước, rồi chạy lại."
        ) from exc
    return YOLO


def _apex_from_filename(path):
    match = re.match(r"^(-?\d+)_(-?\d+)_.*\.jpg$", path.name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def prepare_dataset(source=SOURCE_DATASET, output=DATASET_ROOT, val_fraction=0.2,
                    box_size=18, seed=42):
    """Make a deterministic YOLO dataset from the legacy apex image names.

    A square box is used because YOLO detection requires boxes while this
    legacy set has point labels.  Images with x=-1 are clamped to the visible
    left edge instead of being silently dropped.
    """
    source, output = Path(source), Path(output)
    examples = []
    for image_path in sorted(source.glob("*.jpg")):
        apex = _apex_from_filename(image_path)
        if apex is not None:
            examples.append((image_path, apex))
    if len(examples) < 2:
        raise RuntimeError("Cần ít nhất 2 ảnh có tên <x>_<y>_<uuid>.jpg trong %s" % source)

    random.Random(seed).shuffle(examples)
    val_count = max(1, int(round(len(examples) * val_fraction)))
    splits = {"train": examples[val_count:], "val": examples[:val_count]}

    if output.exists():
        shutil.rmtree(str(output))
    for split, rows in splits.items():
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, (x, y) in rows:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError("Không đọc được ảnh: %s" % image_path)
            height, width = image.shape[:2]
            # Dataset has a few points at x=-1; this preserves them as an edge target.
            x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
            half = box_size / 2.0
            x1, x2 = max(0.0, x - half), min(float(width), x + half)
            y1, y2 = max(0.0, y - half), min(float(height), y + half)
            xc, yc = (x1 + x2) / 2.0 / width, (y1 + y2) / 2.0 / height
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            shutil.copy2(str(image_path), str(image_dir / image_path.name))
            (label_dir / (image_path.stem + ".txt")).write_text(
                "0 %.7f %.7f %.7f %.7f\n" % (xc, yc, bw, bh), encoding="utf-8"
            )

    yaml_text = "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: %s\n" % (
        output.resolve(), CLASS_NAME
    )
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    return {key: len(value) for key, value in splits.items()}


class YoloLaneDetector:
    """Detect the labelled apex and map its x-coordinate to steering [-1, 1]."""

    def __init__(self, weights=BEST_WEIGHTS, confidence=0.25, smoothing=0.35):
        YOLO = _require_yolo()
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                "Không có weights fine-tuned: %s. Hãy chạy lệnh train trước." % self.weights
            )
        self.model = YOLO(str(self.weights))
        self.confidence = confidence
        self.smoothing = smoothing
        self.last_steering = 0.0

    def process_frame(self, image, draw_debug=True):
        result = self.model(image, imgsz=224, conf=self.confidence, verbose=False)[0]
        output = image.copy()
        info = {"detected": False, "lane_confident": False, "steering": self.last_steering}
        if result.boxes is None or len(result.boxes) == 0:
            if draw_debug:
                cv2.putText(output, "YOLO: apex lost - STOP", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
            return output, self.last_steering, info

        boxes = result.boxes
        best_index = int(np.argmax(boxes.conf.cpu().numpy()))
        x1, y1, x2, y2 = boxes.xyxy[best_index].cpu().numpy().astype(int)
        confidence = float(boxes.conf[best_index])
        height, width = image.shape[:2]
        apex_x, apex_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        raw_steering = float(np.clip((apex_x / (width / 2.0)) - 1.0, -1.0, 1.0))
        steering = self.smoothing * raw_steering + (1.0 - self.smoothing) * self.last_steering
        self.last_steering = float(np.clip(steering, -1.0, 1.0))
        info.update({"detected": True, "lane_confident": True, "steering": self.last_steering,
                     "confidence": confidence, "apex_x": apex_x, "apex_y": apex_y,
                     "raw_steering": raw_steering})
        if draw_debug:
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(output, (int(apex_x), int(apex_y)), 4, (0, 255, 0), -1)
            cv2.putText(output, "apex %.2f steer %.2f" % (confidence, self.last_steering),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return output, self.last_steering, info


def train(epochs=80, imgsz=224, batch=8, device=0):
    if not DATA_YAML.exists():
        print("Preparing YOLO dataset from legacy apex images...")
        print(prepare_dataset())
    YOLO = _require_yolo()
    if not BASE_WEIGHTS.exists():
        raise FileNotFoundError("Không tìm thấy weight gốc: %s" % BASE_WEIGHTS)
    model = YOLO(str(BASE_WEIGHTS))
    return model.train(data=str(DATA_YAML), epochs=epochs, imgsz=imgsz, batch=batch,
                       device=device, project=str(RUNS_DIR.parent), name=RUNS_DIR.name,
                       exist_ok=True, pretrained=True, patience=20, workers=0, seed=42)


def infer(source=0, weights=BEST_WEIGHTS):
    detector = YoloLaneDetector(weights)
    source = int(source) if str(source).isdigit() else str(source)
    camera = cv2.VideoCapture(source)
    if not camera.isOpened():
        raise RuntimeError("Không mở được nguồn video: %s" % source)
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        annotated, steering, _ = detector.process_frame(frame)
        cv2.imshow("YOLOv8 Lane Following - q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 lane apex pipeline")
    sub = parser.add_subparsers(dest="command")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--box-size", type=int, default=18)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=80)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--imgsz", type=int, default=224)
    train_parser.add_argument("--device", default=0)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--source", default=0)
    infer_parser.add_argument("--weights", default=str(BEST_WEIGHTS))
    args = parser.parse_args()
    if not args.command:
        parser.error("chọn một lệnh: prepare, train, hoặc infer")
    if args.command == "prepare":
        print("Dataset ready:", prepare_dataset(box_size=args.box_size))
    elif args.command == "train":
        train(args.epochs, args.imgsz, args.batch, args.device)
    else:
        infer(args.source, args.weights)


if __name__ == "__main__":
    main()
