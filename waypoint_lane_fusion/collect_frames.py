#!/usr/bin/env python3
import argparse, time
from pathlib import Path
import cv2
from waypoint_lane_fusion.camera import FrameSource
from waypoint_lane_fusion.config import load_config

p=argparse.ArgumentParser(); p.add_argument("--source",default="camera"); p.add_argument("--output",type=Path,default=Path("dataset/images")); p.add_argument("--every",type=int,default=2); a=p.parse_args()
cfg=load_config(); source=FrameSource(a.source,cfg["camera"]); a.output.mkdir(parents=True,exist_ok=True); count=0
try:
    while True:
        frame=source.read()
        if frame is None: break
        if count%a.every==0: cv2.imwrite(str(a.output/("frame_%013d.jpg"%int(time.time()*1000))),frame)
        count+=1; cv2.imshow("collect q=quit",frame)
        if cv2.waitKey(1)&0xff==ord("q"): break
finally: source.close(); cv2.destroyAllWindows()
