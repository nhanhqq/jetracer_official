import argparse, json, time
from pathlib import Path
import sys
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V2.live import LiveRunner

p=argparse.ArgumentParser(); p.add_argument('--source',default='notebook3/test/1786085420913_202326730929621029_7442541399315262114.mp4'); p.add_argument('--frames',type=int,default=50); p.add_argument('--output',default='V2/results/evaluation/live_pipeline_benchmark.json'); a=p.parse_args()
runner=LiveRunner('V2/config.yaml'); cap=cv2.VideoCapture(a.source); times=[]; n=0
while n<a.frames:
    ok,frame=cap.read()
    if not ok: break
    t=time.time(); runner._command(frame); times.append((time.time()-t)*1000.); n+=1
cap.release(); times=times[2:]
out={'frames':n,'backend':runner.mode,'mean_ms':sum(times)/max(1,len(times)),'median_ms':sorted(times)[len(times)//2] if times else 0,'fps':1000./(sum(times)/max(1,len(times)))}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
