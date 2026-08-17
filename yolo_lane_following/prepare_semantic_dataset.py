#!/usr/bin/env python3
from __future__ import annotations
"""Create dense YOLO26 semantic masks for the physical JetRacer track.

Class IDs are intentionally safety-oriented: road and orange divider are
separate, while the white shoulders are forbidden.  Obstacles override all
other labels so a white cup is never learned as a safe shoulder.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from bootstrap_dataset import divider_mask
from lane_detection_v2 import LaneDetector


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SOURCE = ROOT / "dataset"
APEX = PROJECT / "notebook3" / "old_codes" / "road_following_A" / "apex"

BACKGROUND, ROAD, DIVIDER, FORBIDDEN, OBSTACLE = range(5)


def polygon_masks(label_path: Path, shape: tuple[int, int]) -> dict[int, np.ndarray]:
    height, width = shape
    masks = {ROAD: np.zeros(shape, np.uint8), DIVIDER: np.zeros(shape, np.uint8),
             OBSTACLE: np.zeros(shape, np.uint8)}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        source_class = int(values[0])
        target_class = {0: ROAD, 1: DIVIDER, 2: OBSTACLE}.get(source_class)
        if target_class is None:
            continue
        points = np.asarray([float(v) for v in values[1:]], np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        cv2.fillPoly(masks[target_class], [np.round(points).astype(np.int32)], 255)
    return masks


def forbidden_mask(image: np.ndarray, road: np.ndarray) -> np.ndarray:
    """White material outside the dark road is a prohibited shoulder.

    A small road dilation prevents specular highlights on the glossy carriageway
    becoming forbidden labels.  Actual obstacle labels are applied afterwards.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([180, 82, 255]))
    outside_road = cv2.bitwise_not(cv2.dilate(road, np.ones((7, 7), np.uint8), iterations=1))
    return cv2.bitwise_and(white, outside_road)


def write_sample(image: np.ndarray, masks: dict[int, np.ndarray], destination: Path, split: str, stem: str) -> None:
    image_path = destination / "images" / split / f"{stem}.jpg"
    mask_path = destination / "masks" / split / f"{stem}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    dense = np.full(image.shape[:2], BACKGROUND, np.uint8)
    dense[masks[ROAD] > 0] = ROAD
    dense[masks[DIVIDER] > 0] = DIVIDER
    dense[forbidden_mask(image, masks[ROAD]) > 0] = FORBIDDEN
    dense[masks[OBSTACLE] > 0] = OBSTACLE
    cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    cv2.imwrite(str(mask_path), dense)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense semantic masks from track footage and apex images")
    parser.add_argument("--output", type=Path, default=ROOT / "semantic_dataset")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    counts = {"train": 0, "val": 0}
    for split in ("train", "val"):
        for image_path in sorted((SOURCE / "images" / split).glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            masks = polygon_masks(SOURCE / "labels" / split / f"{image_path.stem}.txt", image.shape[:2])
            write_sample(image, masks, args.output, split, f"track_{image_path.stem}")
            counts[split] += 1

    # Apex labels encode a future centre point, so bootstrap road/divider masks
    # from the established real-track detector.  Keep a deterministic holdout.
    detector = LaneDetector(224, 224)
    apex_paths = sorted(APEX.glob("*.jpg"))
    for index, image_path in enumerate(apex_paths):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        _, _, info = detector.process_frame(image, draw_debug=False)
        split = "val" if index % 8 == 0 else "train"
        road = info["mask_road"]
        masks = {ROAD: road, DIVIDER: divider_mask(image, road),
                 OBSTACLE: np.zeros_like(road)}
        write_sample(image, masks, args.output, split, f"apex_{image_path.stem}")
        counts[split] += 1

    (args.output / "data.yaml").write_text(
        "path: " + str(args.output.resolve()) + "\n"
        "train: images/train\nval: images/val\nmasks_dir: masks\nnames:\n"
        "  0: background\n  1: road\n  2: divider\n  3: forbidden\n  4: obstacle\n",
        encoding="utf-8",
    )
    print(f"Semantic dataset: {args.output.resolve()} train={counts['train']} val={counts['val']}")


if __name__ == "__main__":
    main()
