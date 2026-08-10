#!/usr/bin/env python3
from __future__ import annotations
"""Build a YOLO26 instance-segmentation dataset from repository road footage.

The existing notebook3 lane detector supplies conservative pseudo masks for road.
Red/orange lane markings become divider instances. Real obstacle boxes and additional
synthetic obstacles are composited only inside the road region. Validation is
split by source, not adjacent frames, to avoid temporal leakage.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "notebook3"))
from lane_detection_v2 import LaneDetector  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
CLASS_NAMES = {0: "road", 1: "divider", 2: "obstacle"}


def iter_frames(source: Path, every: int) -> Iterator[Tuple[int, np.ndarray]]:
    if source.is_dir():
        for index, path in enumerate(sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)):
            frame = cv2.imread(str(path))
            if frame is not None:
                yield index, frame
        return
    if source.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(source))
        if frame is not None:
            yield 0, frame
        return
    if source.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported source: {source}")
    capture = cv2.VideoCapture(str(source))
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % every == 0:
            yield index, frame
        index += 1
    capture.release()


def polygons(mask: np.ndarray, cls: int, min_area: float = 18.0) -> list[str]:
    h, w = mask.shape
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    labels = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        contour = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2)
        if len(contour) < 3:
            continue
        coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in contour)
        labels.append(f"{cls} {coords}")
    return labels


def real_obstacle_mask(info: dict, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    obstacle = info.get("obstacle")
    if obstacle:
        x, y, w, h = (int(obstacle[k]) for k in ("x", "y", "w", "h"))
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def divider_mask(frame: np.ndarray, road: np.ndarray) -> np.ndarray:
    """Extract every red/orange tape marking; runtime selects the followed instance."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    warm = cv2.inRange(hsv, np.array([0, 65, 45]), np.array([25, 255, 255]))
    warm |= cv2.inRange(hsv, np.array([155, 65, 45]), np.array([180, 255, 255]))
    zone = cv2.dilate(road, np.ones((9, 9), np.uint8), iterations=1)
    warm = cv2.bitwise_and(warm, zone)
    return cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def add_synthetic_obstacle(frame: np.ndarray, road: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Composite a high-contrast block/cup silhouette on a valid lower-road point."""
    h, w = road.shape
    ys, xs = np.nonzero((road > 0) & (np.indices(road.shape)[0] > int(h * 0.55)))
    mask = np.zeros_like(road)
    if xs.size < 100:
        return frame, mask
    pick = rng.randrange(xs.size)
    cx, cy = int(xs[pick]), int(ys[pick])
    object_w = rng.randint(max(10, w // 18), max(16, w // 9))
    object_h = rng.randint(max(13, h // 16), max(24, h // 7))
    x1 = int(np.clip(cx - object_w // 2, 1, w - object_w - 1))
    y2 = int(np.clip(cy, object_h + 1, h - 2))
    y1, x2 = y2 - object_h, x1 + object_w
    road_support = np.mean(road[y1:y2, x1:x2] > 0)
    if road_support < 0.65:
        return frame, mask
    colour = rng.choice([(235, 235, 235), (25, 180, 245), (35, 35, 200), (200, 80, 30)])
    result = frame.copy()
    if rng.random() < 0.5:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, -1)
    else:
        centre = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (object_w // 2, object_h // 2)
        cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, -1)
        cv2.ellipse(result, centre, axes, 0, 0, 360, colour, -1)
    # Add a stable contact shadow, but keep it out of the object mask.
    cv2.ellipse(result, ((x1 + x2) // 2, y2), (object_w // 2, max(2, object_h // 8)), 0, 0, 360, (20, 20, 20), -1)
    return result, mask


def write_sample(output: Path, split: str, stem: str, frame: np.ndarray, masks: dict[int, np.ndarray]) -> bool:
    labels = []
    for cls, mask in masks.items():
        labels.extend(polygons(mask, cls))
    if not labels or not polygons(masks[1], 1):
        return False
    image_dir, label_dir = output / "images" / split, output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_dir / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
    (label_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "dataset")
    parser.add_argument("--every", type=int, default=3, help="video sampling interval")
    parser.add_argument("--val-source-index", type=int, default=1, help="whole source reserved for validation")
    parser.add_argument("--synthetic-prob", type=float, default=0.55)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=2608)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    detector = LaneDetector(args.size, args.size)
    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0, "obstacle": 0}
    for source_index, source in enumerate(args.sources):
        split = "val" if source_index == args.val_source_index else "train"
        for frame_index, frame in iter_frames(source, args.every):
            frame = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA)
            _, _, info = detector.process_frame(frame, draw_debug=False)
            masks = {0: info["mask_road"], 1: divider_mask(frame, info["mask_road"]),
                     2: real_obstacle_mask(info, (args.size, args.size))}
            stem = f"s{source_index:02d}_{frame_index:06d}"
            if write_sample(args.output, split, stem, frame, masks):
                counts[split] += 1
                counts["obstacle"] += int(np.any(masks[2]))
            if rng.random() < args.synthetic_prob:
                augmented, obstacle = add_synthetic_obstacle(frame, masks[0], rng)
                if np.any(obstacle):
                    augmented_masks = dict(masks); augmented_masks[2] = obstacle
                    if write_sample(args.output, split, stem + "_obs", augmented, augmented_masks):
                        counts[split] += 1; counts["obstacle"] += 1

    data_yaml = args.output / "data.yaml"
    data_yaml.write_text(
        f"path: {args.output.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {idx}: {name}\n" for idx, name in CLASS_NAMES.items()), encoding="utf-8")
    print(f"Dataset: {args.output.resolve()}")
    print(f"train={counts['train']} val={counts['val']} samples_with_obstacle={counts['obstacle']}")


if __name__ == "__main__":
    main()
