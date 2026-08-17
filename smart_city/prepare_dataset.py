#!/usr/bin/env python3
"""Copy an image-only Smart City collection into YOLO train/val folders."""
from pathlib import Path
import argparse
import hashlib
import shutil


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/traffic")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    images = sorted(p for p in args.source.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit("No images found in %s" % args.source)
    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)
    for image in images:
        digest = int(hashlib.sha1(image.name.encode("utf-8")).hexdigest()[:8], 16) / 0xffffffff
        split = "val" if digest < args.val_ratio else "train"
        shutil.copy2(str(image), str(args.output / "images" / split / image.name))
    print("copied_images=%d output=%s" % (len(images), args.output.resolve()))
    print("Next: annotate the copied images, then run smart_city/audit_dataset.py")


if __name__ == "__main__":
    main()

