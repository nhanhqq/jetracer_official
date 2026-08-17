#!/usr/bin/env python3
"""Audit the Smart City YOLO dataset before starting a costly train."""
from pathlib import Path
import argparse
import yaml


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.data.read_text())
    base = args.data.parent / cfg.get("path", ".")
    if not base.exists():
        raise SystemExit("Dataset directory does not exist: %s" % base)
    total_images = total_labels = missing = empty = 0
    for split in ("train", "val"):
        image_dir = base / cfg[split]
        label_dir = base / cfg[split].replace("images", "labels")
        images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        split_missing = split_empty = 0
        for image in images:
            label = label_dir / (image.stem + ".txt")
            if not label.exists():
                split_missing += 1
            elif not label.read_text().strip():
                split_empty += 1
            else:
                total_labels += len(label.read_text().splitlines())
        total_images += len(images)
        missing += split_missing
        empty += split_empty
        print("%s images=%d missing_labels=%d empty_labels=%d" % (split, len(images), split_missing, split_empty))
    print("total_images=%d total_labels=%d missing_labels=%d empty_labels=%d" %
          (total_images, total_labels, missing, empty))
    if not total_images or missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

