#!/usr/bin/env python3
"""Smart City runtime. Default is dry-run; --arm also requires calibrated BEV."""
import argparse, csv, signal, sys, time
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from yolo_lane_following.config import load_config, resolve_path
from yolo_lane_following.control import AdaptiveController
from yolo_lane_following.crosswalk_detector import CrosswalkDetector
from yolo_lane_following.intersection_control import IntersectionController
from yolo_lane_following.intersection_fsm import IntersectionFSM
from yolo_lane_following.intersection_geometry import BranchExtractor
from yolo_lane_following.run import CarOutput, LatestFrameReader
from yolo_lane_following.semantic_perception import YoloSemanticPerception
from yolo_lane_following.sign_perception import SignPerception
from yolo_lane_following.smart_city_perception import SmartCityScene
from yolo_lane_following.traffic_light import TrafficLightDetector

def main():
    p = argparse.ArgumentParser(description="Smart City semantic lane runtime")
    p.add_argument("--config", type=Path, default=Path(__file__).with_name("config_smart_city.yaml"))
    p.add_argument("--source", required=True, help="recorded video, image, or camera")
    p.add_argument("--arm", action="store_true"); p.add_argument("--display", action="store_true")
    p.add_argument("--output-video", type=Path); args = p.parse_args()
    cfg = load_config(args.config); sc = cfg["smart_city"]
    if args.arm and sc["intersection"].get("homography") is None:
        raise SystemExit("Refusing --arm: calibrate and set smart_city.intersection.homography first.")
    semantic = YoloSemanticPerception(cfg); semantic.warmup()
    lane_ctl = AdaptiveController(dict(cfg["control"], max_lost_frames=cfg["tracking"]["max_lost_frames"], lane_only=True))
    crosswalk, branches, signs = CrosswalkDetector(sc["crosswalk"]), BranchExtractor(sc["intersection"]), SignPerception(sc["signs"])
    traffic_light = TrafficLightDetector(sc["traffic_light"])
    fsm, turn_ctl, output = IntersectionFSM(sc), IntersectionController(sc), CarOutput(cfg, dry_run=not args.arm)
    camera = None; reader = None; sequence = 0
    if args.source == "camera":
        from jetcam.csi_camera import CSICamera
        c = cfg["camera"]; camera = CSICamera(width=c["width"], height=c["height"], capture_fps=c["capture_fps"]); camera.running = True
    else: reader = LatestFrameReader(int(args.source) if args.source.isdigit() else args.source)
    log_dir = resolve_path(cfg, cfg["runtime"]["log_dir"]); log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / time.strftime("smart_city_%Y%m%d_%H%M%S.csv")).open("w", newline="", encoding="utf8")
    writer = csv.DictWriter(log, fieldnames="time,fps,semantic_ms,crosswalk_score,crosswalk_y,traffic_light,red_score,green_score,sign_raw,sign_locked,left_score,straight_score,right_score,decision,state,steering,throttle".split(",")); writer.writeheader()
    stopping = False; video = None; last = time.perf_counter(); frame_id = 0
    def stop(*_): nonlocal_stopping[0] = True
    nonlocal_stopping = [False]; signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    try:
      while not nonlocal_stopping[0]:
        if camera is not None:
            raw = camera.value; frame = raw.copy() if raw is not None else None
            if frame is None: continue
        else:
            frame, sequence, ended = reader.latest(sequence)
            if frame is None:
                if ended: break
                time.sleep(.001); continue
        frame = cv2.resize(frame, (cfg["camera"]["width"], cfg["camera"]["height"]), interpolation=cv2.INTER_AREA)
        started = time.perf_counter(); sem = semantic.infer(frame); cw = crosswalk.update(frame)
        sign = (signs.update(frame) if frame_id % max(1, int(sc["signs"].get("infer_stride", 3))) == 0
                else signs.last)
        branch = branches.update(sem.masks["road"], sem.masks["divider"], sem.masks["forbidden"], cw.mask)
        now = time.perf_counter(); dt = now-last; last = now
        light = traffic_light.update(frame)
        scene = SmartCityScene(sem.lane, sem.masks["road"], sem.masks["divider"], sem.masks["forbidden"], cw, branch, sign, light)
        intent = fsm.update(scene, dt)
        if intent.mode == "lane":
            lane_ctl.set_throttle_limit(intent.speed_limit); cmd = lane_ctl.update(sem.lane, 0, frame.shape[1], dt, sem.forbidden_left, sem.forbidden_right, sem.escape_steering, sem.forbidden_front)
        elif intent.mode == "turn":
            cmd = turn_ctl.update(intent.maneuver, fsm.turn_elapsed)
            if cmd.state == "turn_complete": fsm.turn_complete()
        else: cmd = type("C", (), {"steering":0., "throttle":0., "state":intent.state})()
        output.set(cmd.steering, cmd.throttle); annotated = sem.annotated
        cv2.putText(annotated, f"{intent.state} CW:{cw.score:.2f} LIGHT:{light.state}", (4, 16), 0, .38, (0,255,255), 1)
        cv2.putText(annotated, f"L:{branch.scores['left']:.2f} S:{branch.scores['straight']:.2f} R:{branch.scores['right']:.2f}", (4, 32), 0, .38, (0,255,255), 1)
        writer.writerow(dict(time=time.time(), fps=1/max(dt,1e-6), semantic_ms=(time.perf_counter()-started)*1000, crosswalk_score=cw.score, crosswalk_y=cw.y, traffic_light=light.state, red_score=light.red_score, green_score=light.green_score, sign_raw=sign.raw or "", sign_locked=sign.locked or "", left_score=branch.scores["left"], straight_score=branch.scores["straight"], right_score=branch.scores["right"], decision=intent.maneuver or "", state=intent.state, steering=cmd.steering, throttle=cmd.throttle)); log.flush()
        if args.output_video:
            if video is None: args.output_video.parent.mkdir(parents=True, exist_ok=True); video=cv2.VideoWriter(str(args.output_video), cv2.VideoWriter_fourcc(*"mp4v"), reader.fps if reader and reader.fps > 0 else 20, (224,224))
            video.write(annotated)
        if args.display:
            cv2.imshow("Smart City", annotated)
            if cv2.waitKey(1) == ord("q"): break
        frame_id += 1
    finally:
      output.stop(); log.close()
      if camera: camera.running = False
      if reader: reader.stop()
      if video: video.release()
      if args.display: cv2.destroyAllWindows()
if __name__ == "__main__": main()
