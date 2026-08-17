#!/usr/bin/env python3
import argparse, json, signal, sys, time
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from waypoint_lane_fusion.behavior import BehaviorStateMachine
from waypoint_lane_fusion.camera import FrameSource
from waypoint_lane_fusion.config import load_config, resolve_path
from waypoint_lane_fusion.controller import DriveController, WaypointFilter
from waypoint_lane_fusion.lane_model import OnnxWaypointModel, TensorRTWaypointModel, TorchWaypointModel
from waypoint_lane_fusion.telemetry import Telemetry, overlay
from waypoint_lane_fusion.types import DetectionSnapshot
from waypoint_lane_fusion.yolo_v5 import AsyncYoloV5


class CarOutput:
    def __init__(self, cfg, armed):
        self.car = None
        if armed:
            from notebook3.basic_motion import JetRacerController
            h = cfg["hardware"]
            self.car = JetRacerController(h["steering_gain"], h["steering_offset"], h["throttle_gain"], cfg["control"]["throttle_max"])
    def set(self, cmd):
        if self.car: self.car.set_steering(cmd.steering); self.car.set_throttle(cmd.throttle)
    def stop(self):
        if self.car: self.car.stop(); self.car.center_steering()


def main():
    p=argparse.ArgumentParser(description="Waypoint lane + asynchronous YOLOv5n for JetRacer")
    p.add_argument("--config"); p.add_argument("--source",default="camera"); p.add_argument("--lane-model")
    p.add_argument("--lane-backend",choices=("onnx","tensorrt","torchscript"))
    p.add_argument("--arm",action="store_true",help="actually enable motor; default dry-run")
    p.add_argument("--yolo",action="store_true",help="enable asynchronous YOLOv5n")
    p.add_argument("--display",action="store_true"); p.add_argument("--output-video")
    args=p.parse_args(); cfg=load_config(args.config); c=cfg["control"]
    model_path=Path(args.lane_model) if args.lane_model else resolve_path(cfg,cfg["models"]["lane"])
    backend=args.lane_backend or cfg["models"]["lane_backend"]
    lane = {"onnx":OnnxWaypointModel,"tensorrt":TensorRTWaypointModel,"torchscript":TorchWaypointModel}[backend](model_path)
    source=FrameSource(args.source,cfg["camera"]); filt=WaypointFilter(c["waypoint_ema"])
    behavior=BehaviorStateMachine(c); controller=DriveController(c); car=CarOutput(cfg,args.arm)
    yolo=None
    if args.yolo or cfg["models"]["yolo_enabled"]:
        yolo=AsyncYoloV5(resolve_path(cfg,cfg["models"]["yolo"]),cfg["models"]["yolo_confidence"],cfg["models"]["yolo_size"],cfg["models"]["yolo_interval"]).start()
    log_dir=resolve_path(cfg,cfg["runtime"]["log_dir"]); log=Telemetry(log_dir/time.strftime("run_%Y%m%d_%H%M%S.csv"))
    stopping=False; writer=None; last=time.perf_counter()
    def stop(*_):
        nonlocal stopping; stopping=True
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    try:
        while not stopping:
            frame=source.read()
            if frame is None: break
            now=time.perf_counter(); dt=now-last; last=now; raw=lane.predict(frame); filtered=filt.update(raw)
            if yolo: yolo.submit(frame); snapshot=yolo.get_latest()
            else: snapshot=DetectionSnapshot()
            state,bias=behavior.update(filtered,snapshot); command=controller.update(filtered,state,dt,bias); car.set(command)
            fps=1/max(dt,1e-6); rendered=overlay(frame,raw,filtered,command,snapshot,fps)
            log.write(timestamp=time.time(),target_x=raw.x,target_y=raw.y,filtered_x=filtered.x,filtered_y=filtered.y,
                      steering_raw=command.steering_raw,steering_filtered=command.steering,throttle=command.throttle,
                      lane_confidence=filtered.confidence,yolo_objects=json.dumps([d.label for d in snapshot.detections]),state=state.value,fps=fps,yolo_fps=snapshot.fps)
            if args.output_video:
                if writer is None: writer=cv2.VideoWriter(args.output_video,cv2.VideoWriter_fourcc(*"mp4v"),cfg["camera"]["capture_fps"],(frame.shape[1],frame.shape[0]))
                writer.write(rendered)
            if args.display:
                cv2.imshow("Waypoint lane fusion",rendered)
                if cv2.waitKey(1)&0xff==ord("q"): break
    finally:
        car.stop(); source.close(); log.close()
        if yolo: yolo.close()
        if writer: writer.release()
        cv2.destroyAllWindows()


if __name__=="__main__": main()
