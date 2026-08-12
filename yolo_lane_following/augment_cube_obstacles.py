#!/usr/bin/env python3
"""Add perspective cube obstacles to an existing dense semantic dataset.

The generated cube has independently shaded front, top and side faces plus a
contact shadow and visible edges. Colours include camouflage sampled from the
underlying road so obstacle learning cannot depend on colour alone.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
OBSTACLE = 4


def _shade(colour: np.ndarray, scale: float, offset: float = 0.0) -> Tuple[int, int, int]:
    value = np.clip(colour.astype(np.float32) * scale + offset, 0, 255)
    return tuple(int(v) for v in value)


def _colour(image: np.ndarray, cx: int, cy: int, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(4)
    if mode == 0:
        # Camouflage against road/divider/white under the cube.
        y1, y2 = max(0, cy - 4), min(image.shape[0], cy + 5)
        x1, x2 = max(0, cx - 4), min(image.shape[1], cx + 5)
        return np.median(image[y1:y2, x1:x2].reshape(-1, 3), axis=0).astype(np.uint8)
    if mode == 1:
        value = rng.randint(25, 235)
        return np.asarray([value, value, value], np.uint8)
    return np.asarray([rng.randint(15, 240) for _ in range(3)], np.uint8)


def _anchor(mask: np.ndarray, rng: random.Random) -> Tuple[int, int]:
    h, w = mask.shape
    yy, xx = np.indices(mask.shape)
    valid = (mask == 1) & (yy >= int(h * 0.56)) & (yy <= int(h * 0.92))
    ys, xs = np.nonzero(valid)
    if xs.size < 50:
        raise ValueError("not enough road pixels for a cube")
    index = rng.randrange(xs.size)
    return int(xs[index]), int(ys[index])


def render_cube(image: np.ndarray, dense: np.ndarray, size_cm: int,
                rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    h, w = dense.shape
    cx, ground_y = _anchor(dense, rng)
    perspective = 0.58 + 0.75 * (ground_y / float(h))
    face = int(round((8.0 + size_cm * 2.2) * perspective))
    face = int(np.clip(face, 14, 42))
    depth_x = max(4, int(round(face * rng.uniform(0.22, 0.36))))
    depth_y = max(3, int(round(face * rng.uniform(0.16, 0.28))))
    direction = rng.choice((-1, 1))
    x1 = int(np.clip(cx - face // 2, depth_x + 2, w - face - depth_x - 3))
    x2 = x1 + face
    y2 = int(np.clip(ground_y, face + depth_y + 2, h - 3))
    y1 = y2 - face
    dx = direction * depth_x

    front = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.int32)
    top = np.asarray([[x1, y1], [x2, y1], [x2 + dx, y1 - depth_y],
                      [x1 + dx, y1 - depth_y]], np.int32)
    side_x = x2 if direction > 0 else x1
    side = np.asarray([[side_x, y1], [side_x + dx, y1 - depth_y],
                       [side_x + dx, y2 - depth_y], [side_x, y2]], np.int32)

    result = image.copy()
    mask = np.zeros_like(dense, np.uint8)
    colour = _colour(image, cx, ground_y, rng)
    shadow_centre = (cx + dx // 2, min(h - 1, y2 + max(2, face // 12)))
    cv2.ellipse(result, shadow_centre, (face // 2 + depth_x, max(2, face // 7)),
                0, 0, 360, (18, 18, 18), -1, cv2.LINE_AA)
    cv2.fillPoly(result, [front], _shade(colour, 0.82))
    cv2.fillPoly(result, [side], _shade(colour, 0.58))
    cv2.fillPoly(result, [top], _shade(colour, 1.12, 8.0))
    cv2.fillPoly(mask, [front, side, top], 255)
    edge = _shade(colour, 0.28)
    for polygon in (front, side, top):
        cv2.polylines(result, [polygon], True, edge, rng.choice((1, 1, 2)), cv2.LINE_AA)

    output_mask = dense.copy()
    output_mask[mask > 0] = OBSTACLE
    return result, output_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semantic cube obstacle samples")
    parser.add_argument("--dataset", type=Path, default=ROOT / "semantic_dataset")
    parser.add_argument("--copies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2608)
    parser.add_argument("--sizes-cm", type=int, nargs="+", default=list(range(5, 11)))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    written = {"train": 0, "val": 0}

    for split in ("train", "val"):
        image_dir = args.dataset / "images" / split
        mask_dir = args.dataset / "masks" / split
        sources = sorted(p for p in image_dir.glob("*.jpg") if "_cube_" not in p.stem)
        for image_path in sources:
            mask_path = mask_dir / (image_path.stem + ".png")
            image, dense = cv2.imread(str(image_path)), cv2.imread(str(mask_path), 0)
            if image is None or dense is None:
                continue
            for copy_index in range(args.copies):
                try:
                    size_cm = rng.choice(args.sizes_cm)
                    augmented, augmented_mask = render_cube(image, dense, size_cm, rng)
                except ValueError:
                    continue
                stem = "%s_cube_%02dcm_%02d" % (image_path.stem, size_cm, copy_index)
                cv2.imwrite(str(image_dir / (stem + ".jpg")), augmented,
                            [cv2.IMWRITE_JPEG_QUALITY, 94])
                cv2.imwrite(str(mask_dir / (stem + ".png")), augmented_mask)
                written[split] += 1

    data_path = args.dataset / "data.yaml"
    data_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\nmasks_dir: masks\nnames:\n"
        "  0: background\n  1: road\n  2: divider\n  3: forbidden\n  4: obstacle\n"
        % args.dataset.resolve(), encoding="utf-8")
    print("cube samples train=%d val=%d" % (written["train"], written["val"]))
    print("dataset:", args.dataset.resolve())


if __name__ == "__main__":
    main()
