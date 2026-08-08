"""Lane waypoint inference. Output is normalized (x, y, confidence)."""
from pathlib import Path
import cv2
import numpy as np
from .types import Waypoint


def _preprocess(frame, size):
    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(
        (tensor - np.array([.485, .456, .406], np.float32)[None, :, None, None]) /
        np.array([.229, .224, .225], np.float32)[None, :, None, None]
    )


def _waypoint(output):
    out = np.asarray(output).reshape(-1)
    if out.size < 2:
        raise RuntimeError("Lane model must output x,y or x,y,confidence")
    x, y = np.clip(out[:2], 0.0, 1.0)
    confidence = float(np.clip(out[2], 0.0, 1.0)) if out.size >= 3 else 1.0
    return Waypoint(float(x), float(y), confidence)


class OnnxWaypointModel:
    def __init__(self, model_path, input_size=224):
        import onnxruntime as ort
        self.size = int(input_size)
        available = ort.get_available_providers()
        preferred = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available]
        self.session = ort.InferenceSession(str(Path(model_path)), providers=preferred or available)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, frame):
        return _waypoint(self.session.run(None, {self.input_name: _preprocess(frame, self.size)})[0])


class TensorRTWaypointModel:
    """Static-batch TensorRT 8.x runner for JetPack 4/5 Jetsons."""
    def __init__(self, model_path, input_size=224):
        import tensorrt as trt
        import pycuda.autoinit  # noqa: F401 - creates the CUDA context
        import pycuda.driver as cuda
        self.trt, self.cuda, self.size = trt, cuda, int(input_size)
        logger = trt.Logger(trt.Logger.WARNING)
        with Path(model_path).open("rb") as stream:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(stream.read())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize TensorRT engine: %s" % model_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings, self.host, self.device = [], {}, {}
        self.input_index = self.output_index = None
        for index in range(self.engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(index))
            if any(dim < 0 for dim in shape):
                raise RuntimeError("Dynamic TensorRT bindings are not supported: %s" % (shape,))
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            host = cuda.pagelocked_empty(int(trt.volume(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self.host[index], self.device[index] = host, device
            self.bindings.append(int(device))
            if self.engine.binding_is_input(index): self.input_index = index
            else: self.output_index = index
        if self.input_index is None or self.output_index is None:
            raise RuntimeError("Expected one input and one output binding")

    def predict(self, frame):
        tensor = _preprocess(frame, self.size).astype(self.host[self.input_index].dtype, copy=False)
        np.copyto(self.host[self.input_index], tensor.reshape(-1))
        self.cuda.memcpy_htod_async(self.device[self.input_index], self.host[self.input_index], self.stream)
        if not self.context.execute_async_v2(self.bindings, self.stream.handle):
            raise RuntimeError("TensorRT execution failed")
        self.cuda.memcpy_dtoh_async(self.host[self.output_index], self.device[self.output_index], self.stream)
        self.stream.synchronize()
        return _waypoint(self.host[self.output_index])


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
        return _waypoint(out)
