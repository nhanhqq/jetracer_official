#!/usr/bin/env python3
"""Minimal OpenCV YOLO bbox labeler for the Smart City classes."""
from pathlib import Path
import argparse
import cv2
import csv


NAMES = ["bien_cam", "di_thang", "re_phai", "re_trai",
         "den_do", "den_xanh", "crosswalk", "stop_line"]
KEYS = {str(i): i for i in range(len(NAMES))}


class Labeler:
    def __init__(self, root: Path):
        self.root = root
        self.images = sorted((root / "images").glob("*/*.jpg"))
        self.images += sorted((root / "images").glob("*/*.jpeg"))
        self.index = 0
        self.active = 0
        self.boxes = []
        self.start = None
        self.window = "Smart City YOLO labeler"
        self.manifest_path = root / "review_manifest.csv"
        self.reviewed = self._load_review_status()
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self.mouse)

    def _load_review_status(self):
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open(newline="", encoding="utf-8") as stream:
            return {row.get("image", ""): row.get("reviewed", "0") == "1"
                    for row in csv.DictReader(stream)}

    def _mark_reviewed(self):
        name = self.image.name
        self.reviewed[name] = True
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row.get("image") == name:
                row["reviewed"] = "1"
        fields = list(rows[0].keys()) if rows else ["image", "reviewed"]
        with self.manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @property
    def image(self):
        return self.images[self.index]

    @property
    def label_path(self):
        return self.root / "labels" / self.image.parent.name / (self.image.stem + ".txt")

    def load(self):
        self.boxes = []
        if self.label_path.exists():
            for line in self.label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    self.boxes.append(tuple([int(parts[0])] + [float(x) for x in parts[1:]]))

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start:
            x1, y1 = self.start
            x2, y2 = x, y
            if abs(x2 - x1) > 4 and abs(y2 - y1) > 4:
                h, w = self.frame.shape[:2]
                xa, xb = sorted((x1, x2)); ya, yb = sorted((y1, y2))
                self.boxes.append((self.active, (xa + xb) / 2 / w, (ya + yb) / 2 / h,
                                   (xb - xa) / w, (yb - ya) / h))
            self.start = None

    def save(self):
        self.label_path.parent.mkdir(parents=True, exist_ok=True)
        self.label_path.write_text("".join("%d %.6f %.6f %.6f %.6f\n" % b for b in self.boxes))
        self._mark_reviewed()

    def draw(self):
        self.frame = cv2.imread(str(self.image))
        h, w = self.frame.shape[:2]
        out = self.frame.copy()
        for cls, cx, cy, bw, bh in self.boxes:
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(out, NAMES[cls], (x1, max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1)
        status = "reviewed" if self.reviewed.get(self.image.name, False) else "UNREVIEWED"
        cv2.putText(out, "%d/%d class=%s boxes=%d %s | 0-7 select s save n next c clear q quit" %
                    (self.index + 1, len(self.images), NAMES[self.active], len(self.boxes), status),
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 0), 1)
        cv2.imshow(self.window, out)

    def run(self):
        if not self.images:
            raise SystemExit("No images under %s/images/{train,val}" % self.root)
        self.load()
        while True:
            self.draw()
            key = cv2.waitKey(30) & 0xff
            if key == ord("q") or key == 27:
                break
            if chr(key) in KEYS:
                self.active = KEYS[chr(key)]
            elif key == ord("s"):
                self.save()
            elif key == ord("c"):
                self.boxes = []
            elif key == ord("n") or key == 83:
                self.save()
                self.index = min(len(self.images) - 1, self.index + 1)
                self.load()
            elif key == ord("p") or key == 81:
                self.index = max(0, self.index - 1)
                self.load()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("smart_city/datasets/traffic"))
    args = parser.parse_args()
    Labeler(args.dataset).run()
