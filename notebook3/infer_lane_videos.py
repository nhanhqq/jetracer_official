#!/usr/bin/env python3
"""Run lane/obstacle inference on videos without using manual XY labels."""

import argparse
import csv
import glob
import json
import os
import time

import cv2
import numpy as np

from lane_detection_v2 import LaneDetector


def adaptive_throttle(lane_confident, steering, obstacle, previous):
    """Offline-only longitudinal controller mirroring the safe car policy.

    It starts at zero until a lane is locked, slows down for large steering
    commands and obstacles, and ramps down faster than it ramps up.  The
    returned value is only written into the review video/CSV: this script has
    no JetRacer motor import and cannot arm the vehicle.
    """
    if not lane_confident:
        return max(0.0, previous - 0.035), "stop:lane_lost"
    if obstacle:
        target, state = 0.09, "slow:obstacle"
    else:
        target = max(0.09, min(0.18, 0.18 * (1.0 - 0.62 * abs(steering))))
        state = "follow"
    step = 0.008 if target > previous else 0.035
    return float(previous + np.clip(target - previous, -step, step)), state


def infer_video(input_path, output_dir):
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError("Cannot open %s" % input_path)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 20.0
    stem = os.path.splitext(os.path.basename(input_path))[0]
    video_path = os.path.join(output_dir, stem + "_inference.mp4")
    csv_path = os.path.join(output_dir, stem + "_inference_metrics.csv")
    json_path = os.path.join(output_dir, stem + "_inference_summary.json")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (448, 224))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Cannot create %s" % video_path)

    detector = LaneDetector(224, 224)
    rows = []
    previous_target = None
    throttle = 0.0
    started = time.perf_counter()
    try:
        frame_index = 0
        while True:
            ok, source = capture.read()
            if not ok:
                break
            # This is exactly the spatial input used by final_racer_v2.ipynb.
            frame = cv2.resize(source, (224, 224), interpolation=cv2.INTER_AREA)
            tick = time.perf_counter()
            debug, steering, info = detector.process_frame(frame, draw_debug=True)
            latency_ms = (time.perf_counter() - tick) * 1000.0
            target = int(info["target_x"])
            jump = 0 if previous_target is None else abs(target - previous_target)
            previous_target = target

            lane_confident = bool(info["lane_confident"])
            obstacle = info["obstacle"] is not None
            throttle, control_state = adaptive_throttle(lane_confident, steering,
                                                         obstacle, throttle)
            decision = info["lane_action"] if lane_confident else "stop:lane_lost"
            cv2.putText(frame, "frame=%d" % frame_index, (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, decision[:27], (4, 31), cv2.FONT_HERSHEY_SIMPLEX,
                        0.34, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "steer=%+.2f  gas=%.3f  %s" %
                        (steering, throttle, control_state), (4, 47),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 0), 1, cv2.LINE_AA)
            writer.write(np.hstack((frame, debug)))
            rows.append({
                "frame": frame_index,
                "time_s": "%.3f" % (frame_index / fps),
                "target_x": target,
                "target_jump_px": jump,
                "steering": "%.5f" % steering,
                "throttle_preview": "%.5f" % throttle,
                "control_state": control_state,
                "lane_confident": int(lane_confident),
                "obstacle": int(obstacle),
                "decision": decision,
                "case": info["case"],
                "latency_ms": "%.3f" % latency_ms,
            })
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started
    steering = np.asarray([float(row["steering"]) for row in rows])
    jumps = np.asarray([int(row["target_jump_px"]) for row in rows])
    latency = np.asarray([float(row["latency_ms"]) for row in rows])
    summary = {
        "input": os.path.abspath(input_path),
        "output": os.path.abspath(video_path),
        "frames": len(rows),
        "source_fps": round(fps, 3),
        "duration_s": round(len(rows) / fps, 3),
        "lane_confident_ratio": round(float(np.mean([r["lane_confident"] for r in rows])), 4),
        "obstacle_frame_ratio": round(float(np.mean([r["obstacle"] for r in rows])), 4),
        "left_turn_frame_ratio": round(float(np.mean(steering < -0.15)), 4),
        "right_turn_frame_ratio": round(float(np.mean(steering > 0.15)), 4),
        "mean_abs_steering": round(float(np.mean(np.abs(steering))), 4),
        "mean_throttle_preview": round(float(np.mean(
            [float(row["throttle_preview"]) for row in rows])), 4),
        "target_jumps_over_45px": int(np.sum(jumps > 45)),
        "mean_latency_ms": round(float(np.mean(latency)), 3),
        "offline_pipeline_fps": round(len(rows) / max(elapsed, 1e-6), 2),
        "uses_manual_labels": False,
        "motor_output": False,
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Video paths or glob patterns")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    inputs = []
    for item in args.inputs:
        matches = glob.glob(item)
        inputs.extend(matches if matches else [item])
    reports = []
    for path in inputs:
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(path))
        reports.append(infer_video(path, output_dir))
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
