"""Offline/video runner. Motor output is intentionally absent and dry-run only."""
import argparse, csv, time
from pathlib import Path
import cv2
import numpy as np
from .config import load_config
from .pseudo_label import make_masks
from .geometry import estimate_geometry
from .fusion import fuse
from .control import Controller
from .segmentation import Segmenter

def waypoint_from_marking(marking, fallback_x=.5):
    h, w = marking.shape; ys, xs = np.nonzero(marking > 0); keep = ys > int(h*.48)
    if np.count_nonzero(keep) < 8 or np.unique(ys[keep]).size < 3: return fallback_x, .64, .0
    coef = np.polyfit(ys[keep], xs[keep], 1)
    x = float(np.clip(np.polyval(coef, int(h*.64))/w, 0, 1))
    return x, .64, float(min(1, np.count_nonzero(keep)/80.))

def overlay(frame, road, outside, marking, geom, target, command, obstacle=None):
    out = frame.copy(); tint = np.zeros_like(out); tint[road > 0] = (30, 110, 30); tint[outside > 0] = (220, 220, 220); tint[marking > 0] = (0, 80, 220)
    out = cv2.addWeighted(out, .68, tint, .32, 0)
    if obstacle is not None:
        out[obstacle > 0] = (0, 0, 255)
    if geom.points:
        pts=np.asarray([(int(x),int(y)) for y,x in geom.points],np.int32); cv2.polylines(out,[pts],False,(0,255,0),2)
    for y,x in geom.left+geom.right: cv2.circle(out,(int(x),int(y)),2,(255,150,0),-1)
    cv2.circle(out,(int(target.x*out.shape[1]),int(target.y*out.shape[0])),5,(255,0,255),-1)
    text='%s road=%.2f occ=%.2f wp=%.2f steer=%+.2f gas=%+.2f' % (command.state,geom.confidence,geom.occupancy,target.confidence,command.steering,command.throttle)
    cv2.putText(out,text,(3,15),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1,cv2.LINE_AA)
    if command.warning: cv2.putText(out,command.warning,(3,31),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,255),1,cv2.LINE_AA)
    return out

def run(source, output_video, log_path, cfg_path=None, max_frames=0):
    cfg=load_config(cfg_path); gcfg=cfg['geometry']; ctl=Controller(cfg['control']); segmenter=Segmenter(cfg); cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise RuntimeError('Cannot open source: %s' % source)
    fps=cap.get(cv2.CAP_PROP_FPS) or 20.; writer=None; rows=[]; index=0; last=time.time()
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    with Path(log_path).open('w',newline='') as stream:
        fields=['frame_index','timestamp','waypoint_x','waypoint_y','corrected_x','corrected_y','waypoint_confidence','segmentation_confidence','road_occupancy','road_center','curvature','heading_error','white_left_ratio','white_right_ratio','white_center_ratio','obstacle_ratio','steering','throttle','state','warning']
        wr=csv.DictWriter(stream,fieldnames=fields); wr.writeheader()
        while True:
            ok, frame=cap.read()
            if not ok or (max_frames and index>=max_frames): break
            small=cv2.resize(frame,(int(cfg['camera']['width']),int(cfg['camera']['height'])))
            (road,outside,marking), perception_mode=segmenter.infer(small); geom=estimate_geometry(road,outside,marking,gcfg,segmenter.obstacle)
            wx,wy,wc=waypoint_from_marking(marking,geom.center_x/small.shape[1]); target=fuse((wx,wy,wc),geom,small.shape[1],small.shape[0]); now=time.time(); command=ctl.update(target,geom,now-last); last=now
            rendered=overlay(small,road,outside,marking,geom,target,command,segmenter.obstacle)
            if writer is None:
                Path(output_video).parent.mkdir(parents=True,exist_ok=True); writer=cv2.VideoWriter(str(output_video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(rendered.shape[1],rendered.shape[0]))
                if not writer.isOpened(): raise RuntimeError('Cannot create output video: %s' % output_video)
            writer.write(rendered)
            rows.append({'frame_index':index,'timestamp':time.time(),'waypoint_x':wx,'waypoint_y':wy,'corrected_x':target.x,'corrected_y':target.y,'waypoint_confidence':wc,'segmentation_confidence':geom.confidence,'road_occupancy':geom.occupancy,'road_center':geom.center_x/small.shape[1],'curvature':geom.curvature,'heading_error':geom.heading,'white_left_ratio':geom.white_left,'white_right_ratio':geom.white_right,'white_center_ratio':geom.white_center,'obstacle_ratio':geom.obstacle,'steering':command.steering,'throttle':command.throttle,'state':command.state,'warning':command.warning})
            if len(rows)>=50: wr.writerows(rows); stream.flush(); rows=[]
            index+=1
        if rows: wr.writerows(rows)
    cap.release()
    if writer: writer.release()
    return index

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--output-video',required=True); p.add_argument('--log',required=True); p.add_argument('--config'); p.add_argument('--max-frames',type=int,default=0); a=p.parse_args(); print('processed',run(a.source,a.output_video,a.log,a.config,a.max_frames))
