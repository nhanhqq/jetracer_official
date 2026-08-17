#!/usr/bin/env python3
"""Measure obstacle-class pixel metrics for dense semantic model samples."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# TensorRT 8.2 still asks NumPy for the removed alias. Keep this compatibility
# local to the offline evaluator; the deployed perception core is unchanged.
if "bool" not in np.__dict__:
    np.bool = np.bool_

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "track_yolo26n_sem_best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "semantic_dataset")
    parser.add_argument("--split", default="val")
    parser.add_argument("--pattern", default="*_cube_*.jpg")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0")
    parser.add_argument("--all-classes", action="store_true",
                        help="also report per-class IoU and mean IoU")
    args = parser.parse_args()

    paths = sorted(p for p in (args.dataset / "images" / args.split).glob(args.pattern)
                   if not any(token in p.stem for token in args.exclude))
    if not paths:
        raise SystemExit("no evaluation images matched %s" % args.pattern)
    model = YOLO(str(args.model), task="semantic")
    tp = fp = fn = 0
    class_intersection = np.zeros(5, dtype=np.int64)
    class_union = np.zeros(5, dtype=np.int64)
    detected = 0
    predicted_images = 0
    truth_images = 0
    obstacle_class = 4
    for image_path in paths:
        truth = cv2.imread(str(args.dataset / "masks" / args.split / (image_path.stem + ".png")), 0)
        result = model.predict(str(image_path), imgsz=args.imgsz, device=args.device, verbose=False)[0]
        prediction = result.semantic_mask.data.detach().cpu().numpy().astype(np.uint8)
        if prediction.shape != truth.shape:
            prediction = cv2.resize(prediction, (truth.shape[1], truth.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        pred_obstacle = prediction == obstacle_class
        true_obstacle = truth == obstacle_class
        intersection = int(np.count_nonzero(pred_obstacle & true_obstacle))
        tp += intersection
        fp += int(np.count_nonzero(pred_obstacle & ~true_obstacle))
        fn += int(np.count_nonzero(~pred_obstacle & true_obstacle))
        detected += int(intersection > 0)
        predicted_images += int(np.any(pred_obstacle))
        truth_images += int(np.any(true_obstacle))
        if args.all_classes:
            for class_id in range(5):
                pred_class = prediction == class_id
                true_class = truth == class_id
                class_intersection[class_id] += np.count_nonzero(pred_class & true_class)
                class_union[class_id] += np.count_nonzero(pred_class | true_class)

    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    iou = tp / float(max(1, tp + fp + fn))
    print("images=%d truth_images=%d predicted_images=%d detected=%d detection_rate=%.4f" %
          (len(paths), truth_images, predicted_images, detected,
           detected / float(max(1, truth_images))))
    print("obstacle_precision=%.4f obstacle_recall=%.4f obstacle_iou=%.4f" %
          (precision, recall, iou))
    if args.all_classes:
        names = ("background", "road", "divider", "forbidden", "obstacle")
        valid_ious = []
        for class_id, name in enumerate(names):
            if class_union[class_id] == 0:
                continue
            class_iou = class_intersection[class_id] / float(class_union[class_id])
            valid_ious.append(class_iou)
            print("%s_iou=%.4f" % (name, class_iou))
        print("mean_iou=%.4f" % (sum(valid_ious) / max(1, len(valid_ious))))


if __name__ == "__main__":
    main()
