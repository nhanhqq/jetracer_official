#!/usr/bin/env python3
"""Convert JetRacer `{x}_{y}_{uuid}.jpg` labels to normalized CSV."""
import argparse
import csv
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.images / "labels.csv"
    rows = []
    for image_path in sorted(args.images.glob("*.jpg")):
        try:
            x_px, y_px = map(int, image_path.stem.split("_")[:2])
        except (ValueError, IndexError):
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        rows.append({
            # Absolute paths keep the CSV valid when --output is outside the
            # legacy image directory.
            "image": str(image_path.resolve()),
            "x": min(1.0, max(0.0, x_px / float(width))),
            "y": min(1.0, max(0.0, y_px / float(height))),
        })
    if not rows:
        raise RuntimeError("No valid legacy waypoint images found")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)
    print("Converted %d samples -> %s" % (len(rows), output))


if __name__ == "__main__":
    main()
