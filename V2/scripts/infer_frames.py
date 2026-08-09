#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from V2.config import load_config
from V2.segmentation import Segmenter

p = argparse.ArgumentParser()
p.add_argument('--source', default='notebook3')
p.add_argument('--config', default='V2/config.yaml')
p.add_argument('--output', default='V2/results/frame_masks')
a = p.parse_args()
cfg = load_config(a.config)
segmenter = Segmenter(cfg)
files = sorted(Path(a.source).rglob('*'))
files = [x for x in files if x.suffix.lower() in ('.jpg', '.jpeg', '.png')]
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
for index, path in enumerate(files):
    frame = cv2.imread(str(path))
    if frame is None:
        continue
    frame = cv2.resize(frame, (cfg['camera']['width'], cfg['camera']['height']))
    (road, outside, marking), mode = segmenter.infer(frame)
    mask = np.zeros(road.shape, np.uint8)
    mask[road > 0] = 1
    mask[marking > 0] = 2
    mask[outside > 0] = 3
    cv2.imwrite(str(out / ('frame_%06d_mask.png' % index)), mask)
    overlay = frame.copy()
    overlay[road > 0] = (30, 110, 30)
    overlay[outside > 0] = (220, 220, 220)
    overlay[marking > 0] = (0, 80, 220)
    cv2.imwrite(str(out / ('frame_%06d_overlay.jpg' % index)), overlay)
print('frames=%d backend=%s' % (len(files), segmenter.mode))
