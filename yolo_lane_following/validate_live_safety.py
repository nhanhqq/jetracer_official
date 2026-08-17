#!/usr/bin/env python3
"""Collect CSI evidence for cube segmentation and the green start gate.

This utility intentionally has no motor/JetRacer import and cannot arm the car.
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_lane_following.config import load_config
from yolo_lane_following.semantic_perception import YoloSemanticPerception
from yolo_lane_following.start_gate import CompetitionStartGate


def main() -> None:
    parser = argparse.ArgumentParser(description="Motor-free CSI safety validation")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "yolo_lane_following" / "config.yaml")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "yolo_lane_following" / "artifacts" /
                        "track_yolo26n_sem_cube_nano_fp16.engine")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "yolo_lane_following" / "artifacts" /
                        "live_validation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["models"]["semantic"] = str(args.model.resolve())
    perception = YoloSemanticPerception(cfg)
    perception.warmup()
    green_gate = CompetitionStartGate(cfg["competition"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from jetcam.csi_camera import CSICamera

    camera_cfg = cfg["camera"]
    camera = CSICamera(width=camera_cfg["width"], height=camera_cfg["height"],
                       capture_fps=camera_cfg["capture_fps"])
    camera.running = True
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = args.output_dir / ("live_validation_%s.csv" % stamp)
    obstacle_path = args.output_dir / ("obstacle_confirmed_%s.jpg" % stamp)
    green_path = args.output_dir / ("green_confirmed_%s.jpg" % stamp)
    obstacle_saved = green_saved = False
    frames = obstacle_frames = green_frames = 0
    started = time.perf_counter()
    try:
        with log_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["timestamp", "lane_valid", "lane_confidence", "lane_source",
                             "obstacle_risk", "obstacle_boxes", "green_detected",
                             "green_authorized", "green_latency_ms"])
            while not stopping and time.perf_counter() - started < args.duration:
                frame = camera.value
                if frame is None:
                    continue
                green = green_gate.update(frame)
                result = perception.infer(frame)
                frames += 1
                obstacle_active = result.obstacle_risk >= float(
                    cfg["control"].get("obstacle_slow_ratio", 0.58))
                obstacle_frames += int(obstacle_active)
                green_frames += int(green.detected)
                writer.writerow([
                    time.time(), int(result.lane.valid), result.lane.confidence,
                    result.lane.source, result.obstacle_risk,
                    len(result.obstacle_boxes), int(green.detected),
                    int(green_gate.authorized),
                    green_gate.authorization_latency_ms
                    if green_gate.authorization_latency_ms is not None else -1.0,
                ])
                if obstacle_active and not obstacle_saved:
                    cv2.imwrite(str(obstacle_path), result.annotated)
                    obstacle_saved = True
                if green_gate.authorized and not green_saved:
                    marked = result.annotated.copy()
                    if green.detected:
                        cv2.circle(marked, green.center, green.radius, (0, 255, 0), 2)
                    cv2.imwrite(str(green_path), marked)
                    green_saved = True
    finally:
        camera.running = False
    elapsed = max(1e-6, time.perf_counter() - started)
    print("log:", log_path)
    print("frames=%d fps=%.1f obstacle_frames=%d green_frames=%d green_authorized=%s latency_ms=%s" %
          (frames, frames / elapsed, obstacle_frames, green_frames,
           green_gate.authorized,
           "%.1f" % green_gate.authorization_latency_ms
           if green_gate.authorization_latency_ms is not None else "n/a"))
    print("obstacle_evidence:", obstacle_path if obstacle_saved else "not observed")
    print("green_evidence:", green_path if green_saved else "not observed")


if __name__ == "__main__":
    main()
