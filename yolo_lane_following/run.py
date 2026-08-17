#!/usr/bin/env python3
import argparse
import csv
import signal
import statistics
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_lane_following.config import load_config, resolve_path
from yolo_lane_following.control import AdaptiveController
from yolo_lane_following.semantic_perception import YoloSemanticPerception


class CarOutput:
    def __init__(self, cfg: dict, dry_run: bool):
        self.dry_run = dry_run
        self.car = None
        if not dry_run:
            from notebook3.basic_motion import JetRacerController
            c = cfg["control"]
            self.car = JetRacerController(c["steering_gain"], c["steering_offset"],
                                          c["throttle_gain"], c["throttle_max"])

    def set(self, steering: float, throttle: float) -> None:
        if self.car:
            self.car.set_steering(steering)
            self.car.set_throttle(throttle)

    def stop(self) -> None:
        if self.car:
            self.car.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 lane following runtime")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source", default="camera", help="camera, webcam index, image or video")
    parser.add_argument("--model", type=Path,
                        help="override config model (.engine, .onnx or .pt)")
    parser.add_argument("--device", help="override inference device, e.g. cpu or 0")
    parser.add_argument("--arm", action="store_true", help="enable motor output; default is safe dry-run")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--output-video", type=Path,
                        help="save annotated segmentation/control output as MP4")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["models"]["semantic"] = str(args.model.resolve())
    if args.device:
        cfg["models"]["device"] = args.device
    output = CarOutput(cfg, dry_run=not args.arm)
    perception = YoloSemanticPerception(cfg)
    perception.warmup()
    controller_cfg = dict(
        cfg["control"],
        max_lost_frames=cfg["tracking"]["max_lost_frames"],
        lane_lock_confirm_frames=cfg["tracking"].get("lane_lock_confirm_frames", 1),
        lane_lock_min_confidence=cfg["tracking"].get("lane_lock_min_confidence", 0.0),
        lane_only=cfg["models"].get("lane_only", True),
    )
    controller = AdaptiveController(controller_cfg)

    camera = None
    capture = None
    if args.source == "camera":
        from jetcam.csi_camera import CSICamera
        c = cfg["camera"]
        camera = CSICamera(width=c["width"], height=c["height"], capture_fps=c["capture_fps"])
        camera.running = True
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")

    log_dir = resolve_path(cfg, cfg["runtime"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / time.strftime("run_%Y%m%d_%H%M%S.csv")
    stopping = False
    video_writer = None

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    last = time.perf_counter()
    rows = []
    fps_samples = []
    latency_samples = []
    frame_count = 0
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time", "fps", "latency_ms", "lane_conf", "target_x", "steering", "throttle", "state"])
        try:
            while not stopping:
                if camera is not None:
                    frame = camera.value
                    ok = frame is not None
                else:
                    ok, frame = capture.read()
                if not ok:
                    break
                # Match the live CSI path: semantic inference, geometry and
                # control all operate on the configured 224x224 frame. Feeding
                # a 1280x720 replay here made postprocessing needlessly resize
                # full-resolution masks and hid the actual Nano throughput.
                inference_size = (int(cfg["camera"]["width"]), int(cfg["camera"]["height"]))
                if (frame.shape[1], frame.shape[0]) != inference_size:
                    frame = cv2.resize(frame, inference_size, interpolation=cv2.INTER_AREA)
                started = time.perf_counter()
                result = perception.infer(frame)
                now = time.perf_counter()
                dt, last = now - last, now
                command = controller.update(result.lane, 0.0, frame.shape[1], dt,
                                            result.forbidden_left, result.forbidden_right,
                                            result.escape_steering, result.forbidden_front)
                output.set(command.steering, command.throttle)
                elapsed = time.perf_counter() - started
                annotated = result.annotated
                cv2.putText(annotated, command.state, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 255), 1, cv2.LINE_AA)
                if args.output_video:
                    if video_writer is None:
                        args.output_video.parent.mkdir(parents=True, exist_ok=True)
                        source_fps = capture.get(cv2.CAP_PROP_FPS) if capture is not None else cfg["camera"]["capture_fps"]
                        if not source_fps or source_fps <= 0:
                            source_fps = 20.0
                        video_writer = cv2.VideoWriter(
                            str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), source_fps,
                            (annotated.shape[1], annotated.shape[0]))
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Cannot create output video: {args.output_video}")
                    video_writer.write(annotated)
                frame_count += 1
                # Exclude model warm-up from the performance summary.
                if frame_count > 1:
                    fps_samples.append(1.0 / max(dt, 1e-6))
                    latency_samples.append(elapsed * 1000)
                rows.append([time.time(), 1.0 / max(dt, 1e-6), elapsed * 1000,
                             result.lane.confidence, result.lane.target_x,
                             command.steering, command.throttle, command.state])
                if len(rows) >= int(cfg["runtime"]["log_every"]):
                    writer.writerows(rows); stream.flush(); rows.clear()
                if args.display:
                    cv2.imshow("YOLO26 lane following", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            output.stop()
            if rows:
                writer.writerows(rows)
            if camera is not None:
                camera.running = False
            if capture is not None:
                capture.release()
            if video_writer is not None:
                video_writer.release()
            if args.display:
                cv2.destroyAllWindows()
    print(f"Log: {log_path}")
    if fps_samples:
        print(f"Frames: {frame_count} | median FPS: {statistics.median(fps_samples):.1f} | "
              f"median latency: {statistics.median(latency_samples):.1f} ms")


if __name__ == "__main__":
    main()
