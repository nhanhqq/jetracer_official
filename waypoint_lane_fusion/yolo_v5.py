"""Non-blocking YOLOv5n worker. The fast lane loop never waits for YOLO."""
import queue, threading, time
from .types import Detection, DetectionSnapshot


class AsyncYoloV5:
    def __init__(self, weights, confidence=.4, size=320, interval=2):
        self.weights, self.confidence, self.size = weights, confidence, size
        self.interval, self.count = max(1, int(interval)), 0
        self.frames, self.latest = queue.Queue(maxsize=1), DetectionSnapshot()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self): self.thread.start(); return self
    def submit(self, frame):
        self.count += 1
        if self.count % self.interval: return
        try: self.frames.put_nowait(frame.copy())
        except queue.Full:
            try: self.frames.get_nowait()
            except queue.Empty: pass
            try: self.frames.put_nowait(frame.copy())
            except queue.Full: pass
    def get_latest(self): return self.latest
    def close(self): self.stop_event.set(); self.thread.join(timeout=2)

    def _run(self):
        import torch
        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(self.weights), trust_repo=True)
        model.conf = self.confidence
        while not self.stop_event.is_set():
            try: frame = self.frames.get(timeout=.1)
            except queue.Empty: continue
            started = time.perf_counter(); result = model(frame, size=self.size)
            rows = result.pandas().xyxy[0]
            detections = [Detection(str(r["name"]), float(r["confidence"]), float(r.xmin), float(r.ymin), float(r.xmax), float(r.ymax)) for _, r in rows.iterrows()]
            elapsed = time.perf_counter()-started
            self.latest = DetectionSnapshot(detections, time.time(), 1/max(elapsed, 1e-6))
