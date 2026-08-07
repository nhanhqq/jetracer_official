"""Test the exported YOLO26n ONNX detector with a live camera.

Examples:
    python test_cam.py
    python test_cam.py --camera 1 --conf 0.4
    python test_cam.py --csi --sensor-id 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "artifacts" / "traffic_detector" / "traffic_yolo26n.onnx"


def gstreamer_pipeline(
    sensor_id: int,
    capture_width: int,
    capture_height: int,
    display_width: int,
    display_height: int,
    framerate: int,
    flip_method: int,
) -> str:
    """Return an nvarguscamerasrc pipeline for Jetson CSI cameras."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, "
        f"height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, "
        "format=(string)BGRx ! videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live camera test for traffic_yolo26n.onnx")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0, help="USB/webcam index")
    parser.add_argument("--csi", action="store_true", help="Use a Jetson CSI camera")
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.csi:
        pipeline = gstreamer_pipeline(
            sensor_id=args.sensor_id,
            capture_width=args.width,
            capture_height=args.height,
            display_width=args.width,
            display_height=args.height,
            framerate=args.fps,
            flip_method=args.flip_method,
        )
        print("Opening Jetson CSI camera with GStreamer")
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    print(f"Opening USB/webcam camera index {args.camera}")
    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    print(f"Loading ONNX model: {model_path}")
    model = YOLO(str(model_path), task="detect")
    camera = open_camera(args)
    if not camera.isOpened():
        raise RuntimeError(
            "Cannot open camera. Try another --camera index, or use --csi on Jetson."
        )

    window_name = "YOLO26n Traffic Detector - Q/ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Camera frame could not be read.")
                break

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )[0]
            annotated = result.plot()

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            cv2.putText(
                annotated,
                f"FPS: {smoothed_fps:.1f} | detections: {len(result.boxes)}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
