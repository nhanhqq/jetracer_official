"""Dry-run video evaluator for V3; no motor imports or hardware writes."""
import argparse, csv, time
from pathlib import Path
import cv2
from V2.config import load_config
from V3.pipeline import perceive, resize
from V3.divider import DividerTracker
from V3.fusion import fuse
from V3.control import Controller
from V3.waypoint import WaypointModel
from V3.perception import Perception

def run(source, output_video, log_path, cfg_path=None, max_frames=0):
    cfg=load_config(cfg_path or Path(__file__).with_name('config.yaml')); cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise RuntimeError('Cannot open source: %s' % source)
    perception=Perception(cfg); tracker=DividerTracker(cfg['geometry']); ctl=Controller(cfg['control']); waypoint=WaypointModel(cfg); writer=None; last=time.time(); count=0
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    with Path(log_path).open('w',newline='') as stream:
        fields=['frame','divider_x','divider_conf','target_x','target_conf','source','steering','throttle','state','fps']; wr=csv.DictWriter(stream,fieldnames=fields); wr.writeheader()
        while True:
            ok, frame=cap.read()
            if not ok or (max_frames and count>=max_frames): break
            small=resize(frame,cfg); (road,outside,marking),geom,mode=perceive(small,perception,cfg); divider=tracker.update(marking,road,outside); wp=waypoint.predict(small)
            now=time.time(); dt=now-last; last=now; target=fuse(wp,divider,geom,small.shape[1],small.shape[0],cfg['geometry']); cmd=ctl.update(target,geom,dt)
            view=small.copy(); view[road>0]=(30,110,30); view[outside>0]=(220,220,220); view[marking>0]=(0,80,220)
            if divider.points: cv2.polylines(view,[__import__('numpy').asarray([(int(x),int(y)) for y,x in divider.points],__import__('numpy').int32)],False,(0,255,0),2)
            cv2.putText(view,'%s %.2ffps div %.2f target %.2f'%(cmd.state,1/max(1e-3,dt),divider.confidence,target.x),(2,15),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
            if writer is None:
                Path(output_video).parent.mkdir(parents=True,exist_ok=True); writer=cv2.VideoWriter(str(output_video),cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS) or 20.,(small.shape[1],small.shape[0]))
            writer.write(view); wr.writerow({'frame':count,'divider_x':divider.x,'divider_conf':divider.confidence,'target_x':target.x,'target_conf':target.confidence,'source':target.source,'steering':cmd.steering,'throttle':cmd.throttle,'state':cmd.state,'fps':1/max(1e-3,dt)}); count+=1
    cap.release();
    if writer: writer.release()
    return count

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--output-video',required=True); p.add_argument('--log',required=True); p.add_argument('--config'); p.add_argument('--max-frames',type=int,default=0); a=p.parse_args(); print('processed',run(a.source,a.output_video,a.log,a.config,a.max_frames))
