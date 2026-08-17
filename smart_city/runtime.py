#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# Support direct execution under Python 3.8 as well as ``python -m``.
ROOT = Path(__file__).resolve().parent
if __package__ in (None, "") and str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from smart_city.lane_onnx import SemanticLaneONNX
from smart_city.policy import SmartCityPolicy
from smart_city.traffic_onnx import TrafficONNX


CLASS_NAMES = {
    0: "bien_cam", 1: "di_thang", 2: "re_phai", 3: "re_trai",
    4: "den_do", 5: "den_xanh", 6: "crosswalk", 7: "stop_line",
    8: "forbidden_left", 9: "forbidden_right", 10: "forbidden_straight",
}


class SmartCityRuntime:
    def __init__(self, cfg: dict, arm: bool = False):
        self.cfg = cfg
        self.arm = bool(arm)
        lane_path = (ROOT / cfg["models"]["lane_onnx"]).resolve()
        traffic_path = (ROOT / cfg["models"]["traffic_onnx"]).resolve()
        if not lane_path.exists():
            raise FileNotFoundError("Missing lane ONNX: %s" % lane_path)
        if not traffic_path.exists():
            raise FileNotFoundError("Missing traffic ONNX: %s" % traffic_path)
        lane_cfg = cfg["lane"]
        self.lane = SemanticLaneONNX(lane_path, cfg["models"]["lane_input"],
                                      lane_cfg["road_class"], lane_cfg["divider_class"],
                                      lane_cfg["forbidden_class"])
        self.traffic = TrafficONNX(traffic_path, CLASS_NAMES,
                                    cfg["models"]["traffic_confidence"],
                                    cfg["models"]["traffic_input"])
        traffic_cfg = cfg["traffic"]
        self.policy = SmartCityPolicy(
            traffic_cfg["red_confirm_frames"], traffic_cfg["green_confirm_frames"],
            traffic_cfg["sign_confirm_frames"], traffic_cfg["default_route"],
            traffic_cfg.get("forbidden_direction"),
            traffic_cfg.get("forbidden_random_seed", 2608))
        self.last_error = 0.0
        self.turn_until = 0.0
        self.turn_route = None
        self.intersection_cooldown = 0.0

    def process(self, frame: np.ndarray, now: Optional[float] = None):
        now = time.monotonic() if now is None else float(now)
        lane_cfg = self.cfg["lane"]
        lane = self.lane.infer(frame, lane_cfg["lookahead_ratio"],
                               lane_cfg["bottom_ratio"], lane_cfg["min_pixels"])
        detections = self.traffic.detect(frame)
        decision = self.policy.update(detections, lane.valid, lane.forbidden_front)
        labels = {str(d["label"]).lower() for d in detections}
        intersection = bool(labels.intersection({"crosswalk", "stop_line"})) or lane.forbidden_front > 0.35
        if self.intersection_cooldown > 0:
            self.intersection_cooldown = max(0.0, self.intersection_cooldown - 0.033)
        if (decision.state == "DRIVE" and intersection and self.turn_until <= now and
                self.intersection_cooldown <= 0 and self.policy.pending_route in ("LEFT", "RIGHT")):
            self.turn_route = self.policy.pending_route
            self.turn_until = now + float(self.cfg["traffic"].get("turn_time_s", 0.65))
            self.intersection_cooldown = float(self.cfg["traffic"].get("intersection_timeout_s", 4.0))
        if self.turn_until > now and decision.state == "DRIVE":
            turn_sign = -1.0 if self.turn_route == "LEFT" else 1.0
            steering = turn_sign * float(self.cfg["control"]["max_steering"])
            throttle = float(self.cfg["control"]["turn_throttle"])
            action = "TURN_" + str(self.turn_route)
        elif decision.state != "DRIVE":
            steering, throttle, action = 0.0, float(self.cfg["control"]["stop_throttle"]), decision.state
        else:
            error = (lane.target_x - frame.shape[1] / 2.0) / max(1.0, frame.shape[1] / 2.0)
            dt = 1.0 / max(1.0, float(self.cfg["camera"]["capture_fps"]))
            derivative = (error - self.last_error) / dt
            self.last_error = error
            c = self.cfg["control"]
            steering = c["steering_gain"] * (c["kp"] * error + c["kd"] * derivative + c["heading_gain"] * lane.heading_error)
            steering = float(np.clip(steering, -c["max_steering"], c["max_steering"]))
            throttle = float(c["throttle"])
            action = "FOLLOW_LANE"
        annotated = frame.copy()
        cv2.line(annotated, (frame.shape[1] // 2, frame.shape[0]),
                 (int(lane.target_x), int(frame.shape[0] * lane_cfg["lookahead_ratio"])),
                 (0, 255, 0) if lane.valid else (0, 0, 255), 2)
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 1)
            cv2.putText(annotated, "%s %.2f" % (d["label"], d["confidence"]),
                        (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 180, 0), 1)
        cv2.putText(annotated, "%s route=%s signal=%s" % (action, decision.route, self.policy.signal),
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
        return {"lane": lane, "detections": detections, "decision": decision,
                "steering": steering, "throttle": throttle, "action": action,
                "annotated": annotated}

    def command(self, steering: float, throttle: float):
        """Hardware hook intentionally remains disarmed unless --arm is given."""
        if not self.arm:
            return
        from jetracer.nvidia_racecar import NvidiaRacecar
        if not hasattr(self, "car"):
            self.car = NvidiaRacecar()
        self.car.steering = float(steering)
        self.car.throttle = float(throttle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart City lane/sign/light runtime")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--source", default="camera", help="camera, webcam index, image or video")
    parser.add_argument("--arm", action="store_true", help="enable motor output")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    runtime = SmartCityRuntime(cfg, arm=args.arm)
    if args.source == "camera":
        from jetcam.csi_camera import CSICamera
        camera = CSICamera(width=cfg["camera"]["width"], height=cfg["camera"]["height"],
                           capture_fps=cfg["camera"]["capture_fps"])
        camera.running = True
        capture = None
    else:
        camera = None
        source = int(args.source) if str(args.source).isdigit() else args.source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError("Cannot open source: %s" % args.source)
    log_dir = (ROOT / cfg["runtime"]["log_dir"]).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("smart_city_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "fps", "detected_object/sign", "confidence",
                         "decision", "latency_ms", "control_output"])
        try:
            while not stop:
                if camera is not None:
                    frame = camera.value
                    if frame is None:
                        continue
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                frame = cv2.resize(frame, (cfg["camera"]["width"], cfg["camera"]["height"]))
                started = time.perf_counter()
                result = runtime.process(frame)
                runtime.command(result["steering"], result["throttle"])
                latency = (time.perf_counter() - started) * 1000.0
                labels = [d["label"] for d in result["detections"]]
                confidence = max([d["confidence"] for d in result["detections"]] or [0.0])
                writer.writerow([time.time(), 1000.0 / max(latency, 1e-3), json.dumps(labels),
                                 confidence, result["action"], latency,
                                 "steering=%.3f,throttle=%.3f" % (result["steering"], result["throttle"])])
                stream.flush()
                if args.display:
                    cv2.imshow("Smart City", result["annotated"])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            if camera is not None:
                camera.running = False
            if capture is not None:
                capture.release()
            if args.display:
                cv2.destroyAllWindows()
            if getattr(runtime, "car", None) is not None:
                runtime.car.throttle = 0.0
    print("Log:", log_path)


if __name__ == "__main__":
    main()
