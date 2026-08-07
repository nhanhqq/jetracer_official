#!/usr/bin/env python3
"""Build a deterministic MP4 preview from a directory of dataset images."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.as_posix())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an image tree to an MP4 video")
    parser.add_argument("--images", type=Path, default=ROOT / "dataset" / "images")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "dataset_images.mp4")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--exclude-augmented", action="store_true",
                        help="exclude synthetic *_obs frames")
    args = parser.parse_args()

    paths = sorted(args.images.rglob("*.jpg"), key=natural_key)
    if args.exclude_augmented:
        paths = [path for path in paths if not path.stem.endswith("_obs")]
    if not paths:
        raise SystemExit(f"No JPG images found under {args.images}")

    first = cv2.imread(str(paths[0]))
    if first is None:
        raise SystemExit(f"Cannot read {paths[0]}")
    height, width = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot create {args.output}")

    written = 0
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Cannot read {path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            label = path.relative_to(args.images).as_posix()
            cv2.rectangle(frame, (0, 0), (width, 18), (0, 0, 0), -1)
            cv2.putText(frame, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    print(f"Video: {args.output.resolve()}")
    print(f"Frames: {written} | size: {width}x{height} | FPS: {args.fps:g}")


if __name__ == "__main__":
    main()
