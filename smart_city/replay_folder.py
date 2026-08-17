#!/usr/bin/env python3
"""Disarmed folder replay for Smart City runtime evidence."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import yaml

# Support direct execution under the Python 3.8 Docker environment.
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from smart_city.runtime import SmartCityRuntime


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Smart City on an image folder")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--report", type=Path,
                        default=Path(__file__).resolve().parent / "logs/replay_report.json")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    runtime = SmartCityRuntime(cfg, arm=False)
    paths = sorted(p for p in args.source.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise SystemExit("No images found under %s" % args.source)
    states = Counter()
    actions = Counter()
    labels = Counter()
    latencies = []
    lane_valid = 0
    processed = 0
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frame = cv2.resize(frame, (cfg["camera"]["width"], cfg["camera"]["height"]))
        started = time.perf_counter()
        result = runtime.process(frame)
        latencies.append((time.perf_counter() - started) * 1000.0)
        states[result["decision"].state] += 1
        actions[result["action"]] += 1
        lane_valid += int(result["lane"].valid)
        for detection in result["detections"]:
            labels[detection["label"]] += 1
        processed += 1
    if not latencies:
        raise SystemExit("No readable images under %s" % args.source)
    ordered = sorted(latencies)
    percentile = lambda ratio: ordered[min(len(ordered) - 1, int(len(ordered) * ratio))]
    report = {
        "source": str(args.source.resolve()),
        "traffic_model": cfg["models"]["traffic_onnx"],
        "processed": processed,
        "lane_valid": lane_valid,
        "lane_valid_pct": 100.0 * lane_valid / processed,
        "states": dict(states),
        "actions": dict(actions),
        "detections": dict(labels),
        "latency_ms": {"median": percentile(.50), "p95": percentile(.95),
                       "max": max(ordered)},
        "armed": runtime.arm,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("report:", args.report.resolve())


if __name__ == "__main__":
    main()
