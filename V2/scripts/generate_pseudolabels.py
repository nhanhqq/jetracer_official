import argparse, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from V2.pseudo_label import generate
p=argparse.ArgumentParser(); p.add_argument('--source',default='yolo_lane_following/dataset/images/train'); p.add_argument('--output',default='V2/data/pseudo_labels'); p.add_argument('--every',type=int,default=1); a=p.parse_args(); print('frames',generate(a.source,a.output,a.every))

