#!/usr/bin/env python3
"""Click four ground-plane points, then paste the printed homography into config."""
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(); parser.add_argument("image"); parser.add_argument("--width", type=int, default=224); parser.add_argument("--height", type=int, default=224)
args = parser.parse_args(); image = cv2.imread(args.image)
if image is None: raise SystemExit("cannot read image")
points = []
def click(_event, x, y, _flags, _data):
    if _event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y)); cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
cv2.namedWindow("click: far-left, far-right, near-right, near-left")
cv2.setMouseCallback("click: far-left, far-right, near-right, near-left", click)
while len(points) < 4:
    cv2.imshow("click: far-left, far-right, near-right, near-left", image)
    if cv2.waitKey(20) == 27: raise SystemExit("cancelled")
dst = np.float32([[0, 0], [args.width - 1, 0], [args.width - 1, args.height - 1], [0, args.height - 1]])
H = cv2.getPerspectiveTransform(np.float32(points), dst)
print("homography:"); [print("  - [" + ", ".join(f"{v:.8f}" for v in row) + "]") for row in H]
cv2.imshow("BEV", cv2.warpPerspective(image, H, (args.width, args.height))); cv2.waitKey(0)
