#!/usr/bin/env python3
"""Run repeated still-frame perception/controller smoke tests without motors."""
import argparse
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_lane_following.config import load_config
from yolo_lane_following.control import AdaptiveController
from yolo_lane_following.semantic_perception import YoloSemanticPerception


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--preroll-image", type=Path,
                        help="optional clear-lane frame used to warm up the controller")
    parser.add_argument("--preroll-repeats", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config()
    cfg["models"]["semantic"] = str(args.model.resolve())
    perception = YoloSemanticPerception(cfg)
    controller = AdaptiveController(dict(
        cfg["control"], max_lost_frames=cfg["tracking"]["max_lost_frames"]))
    def load_frame(path: Path):
        loaded = cv2.imread(str(path))
        if loaded is None:
            raise SystemExit("cannot read image: %s" % path)
        return cv2.resize(loaded, (cfg["camera"]["width"], cfg["camera"]["height"]),
                          interpolation=cv2.INTER_AREA)

    def step(frame, phase: str, index: int) -> None:
        result = perception.infer(frame)
        command = controller.update(
            result.lane, result.obstacle_risk, frame.shape[1], 0.05,
            result.forbidden_left, result.forbidden_right,
            result.escape_steering, result.forbidden_front)
        print("phase=%s frame=%d boxes=%d risk=%.3f lane=%s state=%s steer=%+.3f throttle=%+.3f" %
              (phase, index + 1, len(result.obstacle_boxes), result.obstacle_risk,
               result.lane.source, command.state, command.steering, command.throttle))

    if args.preroll_image:
        clear_frame = load_frame(args.preroll_image)
        for index in range(max(1, args.preroll_repeats)):
            step(clear_frame, "preroll", index)
    frame = load_frame(args.image)
    for index in range(max(1, args.repeats)):
        step(frame, "test", index)


if __name__ == "__main__":
    main()
