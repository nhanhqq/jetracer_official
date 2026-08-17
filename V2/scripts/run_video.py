#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from V2.runtime import run

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--output-video', required=True)
p.add_argument('--log', required=True)
p.add_argument('--config', default='V2/config.yaml')
p.add_argument('--max-frames', type=int, default=0)
a = p.parse_args()
print('processed', run(a.source, a.output_video, a.log, a.config, a.max_frames))
