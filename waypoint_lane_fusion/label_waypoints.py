#!/usr/bin/env python3
"""Click the desired future waypoint; writes normalized labels.csv."""
import argparse, csv
from pathlib import Path
import cv2

p=argparse.ArgumentParser(); p.add_argument("images",type=Path); p.add_argument("--output",type=Path); a=p.parse_args()
images=sorted([x for x in a.images.iterdir() if x.suffix.lower() in {".jpg",".jpeg",".png"}]); output=a.output or a.images/"labels.csv"
existing={}
if output.exists(): existing={r["image"]:r for r in csv.DictReader(output.open())}
rows=[]
for path in images:
    frame=cv2.imread(str(path)); selected=[]
    def click(event,x,y,*_):
        if event==cv2.EVENT_LBUTTONDOWN: selected[:]=[(x,y)]
    cv2.namedWindow("Click waypoint | s=skip q=quit"); cv2.setMouseCallback("Click waypoint | s=skip q=quit",click)
    while True:
        shown=frame.copy()
        if selected: cv2.circle(shown,selected[0],7,(0,255,0),-1)
        cv2.imshow("Click waypoint | s=skip q=quit",shown); key=cv2.waitKey(20)&0xff
        if selected and key in (13,32):
            x,y=selected[0]; rows.append({"image":path.name,"x":x/frame.shape[1],"y":y/frame.shape[0]}); break
        if key==ord("s"): break
        if key==ord("q"): images=[]; break
cv2.destroyAllWindows(); merged=dict(existing); merged.update({r["image"]:r for r in rows})
with output.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["image","x","y"]); w.writeheader(); w.writerows(merged.values())
print("Saved %d labels to %s"%(len(merged),output))
