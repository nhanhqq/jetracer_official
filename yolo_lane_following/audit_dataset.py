#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "dataset")
    args = parser.parse_args()
    report = {"splits": {}, "errors": []}
    for split in ("train", "val"):
        images = {p.stem: p for p in (args.dataset / "images" / split).glob("*.jpg")}
        labels = {p.stem: p for p in (args.dataset / "labels" / split).glob("*.txt")}
        classes = Counter()
        for stem, path in labels.items():
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) < 7 or len(fields) % 2 == 0:
                    report["errors"].append(f"{path}:{line_no}: invalid polygon")
                    continue
                try:
                    cls = int(fields[0]); coords = [float(v) for v in fields[1:]]
                except ValueError:
                    report["errors"].append(f"{path}:{line_no}: non-numeric value"); continue
                if cls not in range(3) or any(v < 0 or v > 1 for v in coords):
                    report["errors"].append(f"{path}:{line_no}: class/coordinate out of range")
                classes[cls] += 1
        for stem, path in images.items():
            image = cv2.imread(str(path))
            if image is None:
                report["errors"].append(f"{path}: unreadable")
        missing_labels = sorted(set(images) - set(labels))
        missing_images = sorted(set(labels) - set(images))
        report["splits"][split] = {"images": len(images), "labels": len(labels),
                                    "class_instances": dict(classes),
                                    "missing_labels": missing_labels, "missing_images": missing_images}
    train_stems = set((args.dataset / "images" / "train").glob("*.jpg"))
    val_names = {p.name for p in (args.dataset / "images" / "val").glob("*.jpg")}
    report["filename_leakage"] = sorted(p.name for p in train_stems if p.name in val_names)
    print(json.dumps(report, indent=2))
    if report["errors"] or report["filename_leakage"] or any(
        s["missing_labels"] or s["missing_images"] for s in report["splits"].values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
