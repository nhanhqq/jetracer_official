"""Lane waypoint inference. Output is normalized (x, y, confidence)."""
from pathlib import Path
import cv2
import numpy as np
from .types import Waypoint


class OnnxWaypointModel:
    def __init__(self, model_path, input_size=224):
        import onnxruntime as ort
        self.size = int(input_size)
        available = ort.get_available_providers()
        preferred = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
        self.session = ort.InferenceSession(str(Path(model_path)), providers=preferred or available)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame):
        rgb = cv2.cvtColor(cv2.resize(frame, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        tensor = (tensor - np.array([.485, .456, .406], np.float32)[None, :, None, None]) / np.array([.229, .224, .225], np.float32)[None, :, None, None]
        out = np.asarray(self.session.run(None, {self.input_name: tensor})[0]).reshape(-1)
        if out.size < 2:
            raise RuntimeError("Lane model must output x,y or x,y,confidence")
        x, y = np.clip(out[:2], 0.0, 1.0)
        confidence = float(np.clip(out[2], 0.0, 1.0)) if out.size >= 3 else 1.0
        return Waypoint(float(x), float(y), confidence)


class TorchWaypointModel:
    def __init__(self, model_path, input_size=224, device="cuda"):
        import torch
        self.torch, self.size = torch, int(input_size)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()

    def predict(self, frame):
        torch = self.torch
        rgb = cv2.cvtColor(cv2.resize(frame, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0
        mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]
        with torch.no_grad(): out = self.model((t - mean) / std).flatten().cpu().numpy()
        return Waypoint(float(np.clip(out[0], 0, 1)), float(np.clip(out[1], 0, 1)),
                        float(np.clip(out[2], 0, 1)) if out.size >= 3 else 1.0)
