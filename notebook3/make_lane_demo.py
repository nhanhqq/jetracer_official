#!/usr/bin/env python3
"""Create a chronological offline demo and objective metrics from XYDataset frames."""

import argparse
import csv
import glob
import json
import os
import time
import uuid

import cv2
import numpy as np

from lane_detection_v2 import LaneDetector


def annotation(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    x_text, y_text, uid_text = stem.split("_", 2)
    return int(x_text), int(y_text), uuid.UUID(uid_text).time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="old_codes/road_following_A/apex")
    parser.add_argument("--output", default="artifacts/lane_following_demo.mp4")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--hold", type=int, default=3,
                        help="Repeat each source image for easier visual inspection")
    args = parser.parse_args()

    paths = glob.glob(os.path.join(args.dataset, "*.jpg"))
    paths.sort(key=lambda path: annotation(path)[2])
    if not paths:
        raise SystemExit("No JPG frames found in %s" % args.dataset)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    csv_path = os.path.splitext(output)[0] + "_metrics.csv"
    json_path = os.path.splitext(output)[0] + "_summary.json"

    detector = LaneDetector(224, 224)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (448, 224))
    if not writer.isOpened():
        raise RuntimeError("Cannot open video writer: %s" % output)

    rows = []
    started = time.perf_counter()
    previous_target = None
    try:
        for index, path in enumerate(paths):
            image = cv2.imread(path)
            if image is None:
                continue
            gt_x, gt_y, _ = annotation(path)
            tick = time.perf_counter()
            debug, steering, info = detector.process_frame(image, draw_debug=True)
            latency_ms = (time.perf_counter() - tick) * 1000.0

            target_x = int(info["target_x"])
            error_x = abs(target_x - gt_x)
            jump_px = 0 if previous_target is None else abs(target_x - previous_target)
            previous_target = target_x
            rows.append({
                "frame": index,
                "file": os.path.basename(path),
                "gt_x": gt_x,
                "gt_y": gt_y,
                "target_x": target_x,
                "error_x": error_x,
                "jump_px": jump_px,
                "steering": "%.5f" % steering,
                "latency_ms": "%.3f" % latency_ms,
                "lane_confident": int(bool(info["lane_confident"])),
                "obstacle": int(info["obstacle"] is not None),
                "decision": info["lane_action"],
                "case": info["case"],
            })

            raw = image.copy()
            cv2.circle(raw, (max(0, min(223, gt_x)), max(0, min(223, gt_y))),
                       6, (255, 0, 255), 2)
            cv2.putText(raw, "GT apex", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(raw, "err=%dpx" % error_x, (4, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            canvas = np.hstack((raw, debug))
            for _ in range(max(1, args.hold)):
                writer.write(canvas)
    finally:
        writer.release()

    elapsed = time.perf_counter() - started
    errors = np.asarray([row["error_x"] for row in rows], dtype=np.float32)
    jumps = np.asarray([row["jump_px"] for row in rows], dtype=np.float32)
    latencies = np.asarray([float(row["latency_ms"]) for row in rows], dtype=np.float32)
    summary = {
        "frames": len(rows),
        "mae_x_px": round(float(np.mean(errors)), 3),
        "median_x_px": round(float(np.median(errors)), 3),
        "p90_x_px": round(float(np.percentile(errors, 90)), 3),
        "within_20px_ratio": round(float(np.mean(errors <= 20)), 4),
        "large_jump_over_45px": int(np.sum(jumps > 45)),
        "lane_confident_ratio": round(float(np.mean([r["lane_confident"] for r in rows])), 4),
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "offline_pipeline_fps": round(len(rows) / max(elapsed, 1e-6), 2),
        "video": output,
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
