import cv2


class FrameSource:
    def __init__(self, source, cfg):
        self.camera, self.capture = None, None
        if source == "camera":
            from jetcam.csi_camera import CSICamera
            self.camera = CSICamera(width=cfg["width"], height=cfg["height"], capture_fps=cfg["capture_fps"])
            self.camera.running = True
        else:
            source = int(source) if str(source).isdigit() else source
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened(): raise RuntimeError("Cannot open source: %s" % source)
    def read(self):
        if self.camera is not None: return self.camera.value
        ok, frame = self.capture.read(); return frame if ok else None
    def close(self):
        if self.camera is not None: self.camera.running = False
        if self.capture is not None: self.capture.release()
