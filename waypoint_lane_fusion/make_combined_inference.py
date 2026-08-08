#!/usr/bin/env python3
"""Sample the two raw notebook3 videos every 0.2 s and run lane inference."""
import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from waypoint_lane_fusion.behavior import BehaviorStateMachine
from waypoint_lane_fusion.config import load_config, resolve_path
from waypoint_lane_fusion.controller import DriveController, WaypointFilter
from waypoint_lane_fusion.lane_model import OnnxWaypointModel
from waypoint_lane_fusion.telemetry import overlay
from waypoint_lane_fusion.types import DetectionSnapshot


def raw_test_videos(test_dir):
    videos = sorted(path for path in test_dir.glob("*.mp4")
                    if "_inference" not in path.stem)
    if len(videos) != 2:
        raise RuntimeError("Expected exactly 2 raw MP4 files in %s, found %d" %
                           (test_dir, len(videos)))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "notebook3/test")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="seconds between sampled frames and output-frame duration")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "waypoint_lane_fusion/artifacts/lane_resnet18_bootstrap.onnx")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "waypoint_lane_fusion/artifacts/combined")
    args = parser.parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_fps = 1.0 / args.interval
    size = (224, 224)
    source_path = args.output_dir / "combined_source_5fps.mp4"
    inference_path = args.output_dir / "combined_resnet18_inference_5fps.mp4"
    csv_path = args.output_dir / "combined_resnet18_metrics.csv"
    summary_path = args.output_dir / "combined_resnet18_summary.json"
    source_writer = cv2.VideoWriter(str(source_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                    output_fps, size)
    inference_writer = cv2.VideoWriter(str(inference_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                       output_fps, size)
    if not source_writer.isOpened() or not inference_writer.isOpened():
        raise RuntimeError("Cannot create output videos in %s" % args.output_dir)

    cfg = load_config(); control_cfg = cfg["control"]
    model = OnnxWaypointModel(args.model)
    waypoint_filter = WaypointFilter(control_cfg["waypoint_ema"])
    behavior = BehaviorStateMachine(control_cfg)
    controller = DriveController(control_cfg)
    snapshot = DetectionSnapshot()
    rows, latencies, selected_per_video = [], [], {}
    output_index = 0
    try:
        for video_path in raw_test_videos(args.test_dir):
            capture = cv2.VideoCapture(str(video_path))
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            if not source_fps or source_fps <= 0:
                source_fps = 30.0
            sample_step = max(1, int(round(source_fps * args.interval)))
            frame_index = selected = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_step:
                    frame_index += 1; continue
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                started = time.perf_counter()
                raw = model.predict(frame); filtered = waypoint_filter.update(raw)
                state, bias = behavior.update(filtered, snapshot)
                command = controller.update(filtered, state, args.interval, bias)
                latency_ms = (time.perf_counter() - started) * 1000.0
                infer_fps = 1000.0 / max(latency_ms, 1e-6)
                rendered = overlay(frame, raw, filtered, command, snapshot, infer_fps)
                cv2.putText(rendered, "%s  t=%.1fs" %
                            (video_path.name[:20], frame_index / source_fps), (6, 91),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
                source_writer.write(frame); inference_writer.write(rendered)
                rows.append({"output_frame": output_index, "source_video": video_path.name,
                             "source_frame": frame_index, "source_time_s": frame_index/source_fps,
                             "target_x": raw.x, "target_y": raw.y,
                             "filtered_x": filtered.x, "filtered_y": filtered.y,
                             "confidence": filtered.confidence,
                             "steering_raw": command.steering_raw,
                             "steering": command.steering, "throttle_preview": command.throttle,
                             "state": state.value, "latency_ms": latency_ms})
                latencies.append(latency_ms); selected += 1; output_index += 1
                frame_index += 1
            capture.release(); selected_per_video[video_path.name] = selected
    finally:
        source_writer.release(); inference_writer.release()

    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    def portable(path):
        try: return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError: return str(path.resolve())
    summary = {"inputs": selected_per_video, "sample_interval_s": args.interval,
               "output_fps": output_fps, "frames": len(rows),
               "duration_s": len(rows) / output_fps,
               "median_inference_ms": statistics.median(latencies),
               "median_inference_fps": 1000.0/statistics.median(latencies),
               "source_video": portable(source_path),
               "inference_video": portable(inference_path),
               "metrics_csv": portable(csv_path), "motor_output": False}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
