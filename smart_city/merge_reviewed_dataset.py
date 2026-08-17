#!/usr/bin/env python3
"""Build a fine-tune set from reviewed urban labels and bootstrap data."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import yaml


NAMES = ["bien_cam", "di_thang", "re_phai", "re_trai",
         "den_do", "den_xanh", "crosswalk", "stop_line"]


def read_manifest(path: Path, allow_unreviewed: bool = False):
    if not path.exists():
        raise SystemExit("Missing review manifest: %s" % path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("Review manifest is empty: %s" % path)
    pending = [row.get("image", "") for row in rows if row.get("reviewed") != "1"]
    if pending and not allow_unreviewed:
        raise SystemExit("%d urban images are not reviewed; first=%s" %
                         (len(pending), pending[0]))
    if pending:
        print("WARNING: merging %d unreviewed pseudo-labeled images" % len(pending))
    return rows


def copy_split(source: Path, output: Path, split: str, prefix: str = ""):
    image_dir = source / "images" / split
    label_dir = source / "labels" / split
    for image in sorted(image_dir.glob("*")):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = prefix + image.stem
        destination_image = output / "images" / split / (stem + image.suffix.lower())
        destination_label = output / "labels" / split / (stem + ".txt")
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(image), str(destination_image))
        label = label_dir / (image.stem + ".txt")
        if not label.exists():
            raise SystemExit("Missing label for %s" % image)
        shutil.copy2(str(label), str(destination_label))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urban", type=Path, default=Path("smart_city/datasets/urban_pseudo"))
    parser.add_argument("--bootstrap", type=Path, default=Path("smart_city/datasets/traffic"))
    parser.add_argument("--output", type=Path, default=Path("smart_city/datasets/traffic_reviewed"))
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="build a provisional pseudo-label dataset; never use as ground truth")
    args = parser.parse_args()
    read_manifest(args.urban / "review_manifest.csv", args.allow_unreviewed)
    if args.output.exists():
        raise SystemExit("Output already exists; choose a new --output: %s" % args.output)
    for split in ("train", "val"):
        copy_split(args.bootstrap, args.output, split, prefix="boot_")
        copy_split(args.urban, args.output, split, prefix="urban_")
    data = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val",
            "names": {i: name for i, name in enumerate(NAMES)}}
    (args.output / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print("merged_dataset:", args.output.resolve())


if __name__ == "__main__":
    main()
