import csv, json
from pathlib import Path
import cv2

FIELDS = ["timestamp","target_x","target_y","filtered_x","filtered_y","steering_raw","steering_filtered","throttle","lane_confidence","yolo_objects","state","fps","yolo_fps"]


class Telemetry:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stream = Path(path).open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS); self.writer.writeheader()
    def write(self, **row): self.writer.writerow(row); self.stream.flush()
    def close(self): self.stream.close()


def overlay(frame, raw, filtered, command, snapshot, fps):
    out = frame.copy(); h, w = out.shape[:2]
    cv2.circle(out, (int(filtered.x*w), int(filtered.y*h)), 6, (0,255,0), -1)
    cv2.circle(out, (int(raw.x*w), int(raw.y*h)), 5, (0,165,255), 2)
    cv2.line(out, (w//2,h-1), (int(filtered.x*w),int(filtered.y*h)), (0,255,0), 2)
    lines = [f"FPS {fps:.1f} | YOLOv5 {snapshot.fps:.1f}", f"State {command.state.value}",
             f"steer {command.steering:+.3f}  throttle {command.throttle:+.3f}", f"confidence {filtered.confidence:.2f}"]
    for i, text in enumerate(lines): cv2.putText(out, text, (6,18+i*18), cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1,cv2.LINE_AA)
    return out
