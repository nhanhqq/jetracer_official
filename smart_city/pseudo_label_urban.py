#!/usr/bin/env python3
"""Create reviewable pseudo-labels for the real urban image collection.

This is an annotation assistant, not ground truth. It copies images, writes an
empty YOLO label file when nothing passes the threshold, and renders previews
so a human can delete/fix false boxes before using the set for training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path,
                        default=ROOT / "artifacts/smart_city_yolo26n_best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/urban_pseudo")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="0",
                        help="Ultralytics device, for example 0 or cpu")
    args = parser.parse_args()
    images = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise SystemExit("No urban images found: %s" % args.source)
    model = YOLO(str(args.model))
    names = {int(k): str(v) for k, v in model.names.items()}
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "previews" / split).mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "review_manifest.csv"
    counts = {}
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "split", "detections", "max_confidence", "reviewed"])
        for index in range(0, len(images), 1):
            source = images[index]
            digest = int(hashlib.sha1(source.name.encode("utf-8")).hexdigest()[:8], 16) / 0xffffffff
            split = "val" if digest < 0.2 else "train"
            destination = args.output / "images" / split / source.name
            label_path = args.output / "labels" / split / (source.stem + ".txt")
            preview_path = args.output / "previews" / split / (source.stem + ".jpg")
            frame = cv2.imread(str(source))
            if frame is None:
                continue
            result = model.predict(frame, imgsz=args.imgsz, conf=args.confidence,
                                   device=args.device, verbose=False)[0]
            h, w = frame.shape[:2]
            lines = []
            preview = frame.copy()
            confidences = []
            if result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = [float(x) for x in box.xyxy[0]]
                    lines.append("%d %.6f %.6f %.6f %.6f" %
                                 (cls, ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                                  (x2 - x1) / w, (y2 - y1) / h))
                    confidences.append(conf)
                    counts[names.get(cls, str(cls))] = counts.get(names.get(cls, str(cls)), 0) + 1
                    cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)
                    cv2.putText(preview, "%s %.2f" % (names.get(cls, str(cls)), conf),
                                (int(x1), max(14, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1)
            shutil.copy2(str(source), str(destination))
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            cv2.imwrite(str(preview_path), preview)
            writer.writerow([source.name, split, len(lines), max(confidences or [0.0]), 0])
    (args.output / "data.yaml").write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n" % args.output.resolve() +
        "".join("  %d: %s\n" % (i, names[i]) for i in sorted(names)), encoding="utf-8")
    print("images=%d output=%s" % (len(images), args.output.resolve()))
    print("pseudo_instances=%s" % counts)
    print("Review previews under %s/previews before training." % args.output)


if __name__ == "__main__":
    main()
