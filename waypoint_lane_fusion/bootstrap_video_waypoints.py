#!/usr/bin/env python3
"""Bootstrap waypoint labels from notebook3 CV on a training video.

Pseudo-labels are only a starting point and should be reviewed/augmented manually.
Never evaluate on the same video used here.
"""
import argparse
import csv
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "notebook3"))
from lane_detection_v2 import LaneDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=1)
    args = parser.parse_args()
    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError("Cannot open %s" % args.video)
    detector = LaneDetector(224, 224)
    rows = []
    frame_index = 0
    while True:
        ok, source = capture.read()
        if not ok:
            break
        if frame_index % max(1, args.every):
            frame_index += 1
            continue
        frame = cv2.resize(source, (224, 224), interpolation=cv2.INTER_AREA)
        _, _, info = detector.process_frame(frame, draw_debug=False)
        if info["lane_confident"]:
            name = "frame_%06d.jpg" % frame_index
            image_path = images_dir / name
            cv2.imwrite(str(image_path), frame)
            rows.append({"image": str(image_path.resolve()),
                         "x": min(1.0, max(0.0, info["target_x"] / 224.0)),
                         "y": 0.60})
        frame_index += 1
    capture.release()
    csv_path = args.output / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "x", "y"])
        writer.writeheader(); writer.writerows(rows)
    print("Pseudo-labeled %d/%d frames -> %s" % (len(rows), frame_index, csv_path))


if __name__ == "__main__":
    main()
