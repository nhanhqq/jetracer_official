"""Conservative pseudo labels for this track's dark-road/white-outside prior."""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np


def _largest_bottom_component(binary):
    b = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    if n <= 1:
        return b
    h, w = b.shape
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = labels[h - 2, max(0, w // 2 - 8):min(w, w // 2 + 9)] == i
        if np.any(touches):
            candidates.append((area + 2.0 * bh * w, i))
    if not candidates:
        return np.zeros_like(b)
    return (labels == max(candidates)[1]).astype(np.uint8)


def make_masks(image):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # White is forbidden. Keep it separate before constructing drivable road.
    white = ((gray > 145) & (hsv[:, :, 1] < 85)).astype(np.uint8)
    white[:int(h * .30)] = 0
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # Dark candidate is only accepted if connected to bottom-center, preventing
    # isolated dark objects on the outside from becoming road.
    dark = ((gray < 145) | (lab[:, :, 0] < 145)) & (white == 0)
    road = _largest_bottom_component(dark.astype(np.uint8))
    road[:int(h * .30)] = 0
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Orange/red markings are a geometry hint, not the road mask itself.
    orange = (((hsv[:, :, 0] < 18) | (hsv[:, :, 0] > 170)) &
              (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 55) & (road > 0)).astype(np.uint8)
    orange = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return (road * 255).astype(np.uint8), (white * 255).astype(np.uint8), (orange * 255).astype(np.uint8)


def generate(source, output, every=1):
    source, output = Path(source), Path(output)
    images = sorted(p for p in source.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    (output / 'road').mkdir(parents=True, exist_ok=True)
    (output / 'outside').mkdir(parents=True, exist_ok=True)
    (output / 'marking').mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(images[::max(1, int(every))]):
        image = cv2.imread(str(path))
        if image is None:
            continue
        road, outside, marking = make_masks(image)
        name = '%06d.png' % index
        cv2.imwrite(str(output / 'road' / name), road)
        cv2.imwrite(str(output / 'outside' / name), outside)
        cv2.imwrite(str(output / 'marking' / name), marking)
        rows.append([name, str(path), int(np.count_nonzero(road)), int(np.count_nonzero(outside))])
    with (output / 'manifest.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['mask', 'image', 'road_pixels', 'outside_pixels']); writer.writerows(rows)
    return len(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('--output', default='data/pseudo_labels'); p.add_argument('--every', type=int, default=1); a = p.parse_args()
    print('generated', generate(a.source, a.output, a.every))
