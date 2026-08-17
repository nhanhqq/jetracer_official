#!/usr/bin/env python3
"""Offline lighting/glare stress test for the lane follower."""

import argparse
import glob
import json
import os
import uuid

import cv2
import numpy as np

from lane_detection_v2 import LaneDetector


def frame_key(path):
    return uuid.UUID(os.path.splitext(os.path.basename(path))[0].split("_", 2)[2]).time


def lighting_variant(image, mode, index):
    if mode == "clean":
        return image
    if mode == "dark":
        return cv2.convertScaleAbs(image, alpha=0.58, beta=-8)
    if mode == "bright":
        return cv2.convertScaleAbs(image, alpha=1.32, beta=24)
    if mode == "flicker":
        alpha, beta = ((0.62, -5) if index % 2 else (1.28, 18))
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    if mode == "glare":
        result = image.copy()
        overlay = np.zeros_like(result)
        center = (70 + (index * 17) % 100, 170)
        cv2.ellipse(overlay, center, (24, 65), 12, 0, 360, (255, 255, 255), -1)
        return cv2.addWeighted(result, 1.0, overlay, 0.38, 0)
    raise ValueError(mode)


def run(paths, mode):
    detector = LaneDetector(224, 224)
    targets, confidence = [], []
    for index, path in enumerate(paths):
        image = lighting_variant(cv2.imread(path), mode, index)
        _, _, info = detector.process_frame(image, draw_debug=False)
        targets.append(int(info["target_x"]))
        confidence.append(bool(info["lane_confident"]))
    return np.asarray(targets), np.asarray(confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="old_codes/road_following_A/apex")
    parser.add_argument("--output", default="artifacts/lighting_benchmark.json")
    args = parser.parse_args()
    paths = sorted(glob.glob(os.path.join(args.dataset, "*.jpg")), key=frame_key)
    if not paths:
        raise SystemExit("Dataset is empty")

    clean, _ = run(paths, "clean")
    report = {}
    for mode in ("dark", "bright", "flicker", "glare"):
        target, confident = run(paths, mode)
        delta = np.abs(target - clean)
        report[mode] = {
            "mean_delta_from_clean_px": round(float(np.mean(delta)), 3),
            "p90_delta_from_clean_px": round(float(np.percentile(delta, 90)), 3),
            "max_delta_from_clean_px": int(np.max(delta)),
            "confidence_ratio": round(float(np.mean(confident)), 4),
            "jumps_over_45px": int(np.sum(np.abs(np.diff(target)) > 45)),
        }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
