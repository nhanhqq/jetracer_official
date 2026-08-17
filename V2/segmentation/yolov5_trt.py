"""Small YOLOv5-seg TensorRT 7 runner for Jetson Nano.

The engine is expected to be exported from the custom three-class YOLOv5n-seg
checkpoint. It returns masks in the V2 representation: road, outside, marking.
Outside is intentionally not a learned class here; white forbidden space is
measured by the deterministic safety mask in ``backend.py``.
"""
import ctypes
from pathlib import Path

import cv2
import numpy as np


class _Cuda:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so")
        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        self.allocations = []
        self.lib.cuInit.argtypes = [ctypes.c_uint]
        self.lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.lib.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        self.lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        self.lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self._check(self.lib.cuInit(0), "cuInit")
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self._check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.context), self.device), "ctxRetain")
        self.make_current()

    @staticmethod
    def _check(code, name):
        if code != 0:
            raise RuntimeError("CUDA %s failed: %d" % (name, code))

    def make_current(self):
        self._check(self.lib.cuCtxSetCurrent(self.context), "ctxSetCurrent")

    def alloc(self, size):
        ptr = ctypes.c_uint64()
        self._check(self.lib.cuMemAlloc_v2(ctypes.byref(ptr), size), "memAlloc")
        self.allocations.append(ptr)
        return ptr

    def h2d(self, ptr, array):
        self._check(self.lib.cuMemcpyHtoD_v2(ptr, ctypes.c_void_p(array.ctypes.data), array.nbytes), "h2d")

    def d2h(self, array, ptr):
        self._check(self.lib.cuMemcpyDtoH_v2(ctypes.c_void_p(array.ctypes.data), ptr, array.nbytes), "d2h")

    def close(self):
        if self.lib is None:
            return
        for ptr in self.allocations:
            self.lib.cuMemFree_v2(ptr)
        self.allocations = []
        self.lib.cuDevicePrimaryCtxRelease(self.device)
        self.lib = None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class Yolov5SegTensorRT:
    def __init__(self, engine_path, input_size=224, class_count=3, conf=0.25, max_detections=16):
        import tensorrt as trt
        self.trt = trt
        self.size = int(input_size)
        self.class_count = int(class_count)
        self.conf = float(conf)
        self.max_detections = int(max_detections)
        self.cuda = _Cuda()
        with Path(engine_path).open("rb") as stream:
            self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(stream.read())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize YOLOv5 segmentation engine: %s" % engine_path)
        self.context = self.engine.create_execution_context()
        self.host, self.device, self.bindings = {}, {}, []
        self.input_index, self.output_indices = None, []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(i))
            if any(d < 0 for d in shape):
                raise RuntimeError("Dynamic YOLOv5 bindings are unsupported: %s" % (shape,))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host = np.empty(int(trt.volume(shape)), dtype=dtype).reshape(shape)
            ptr = self.cuda.alloc(host.nbytes)
            self.host[i], self.device[i] = host, ptr
            self.bindings.append(ptr.value)
            if self.engine.binding_is_input(i):
                self.input_index = i
            else:
                self.output_indices.append(i)
        if self.input_index is None or len(self.output_indices) < 2:
            raise RuntimeError("Expected YOLOv5 input plus detection/prototype outputs")
        self.output_indices.sort(key=lambda i: self.host[i].size, reverse=True)

    def _input(self, frame):
        rgb = cv2.cvtColor(cv2.resize(frame, (self.size, self.size)), cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(tensor).astype(self.host[self.input_index].dtype, copy=False)

    def infer(self, frame):
        h, w = frame.shape[:2]
        self.cuda.make_current()
        tensor = self._input(frame)
        np.copyto(self.host[self.input_index], tensor)
        self.cuda.h2d(self.device[self.input_index], self.host[self.input_index])
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("YOLOv5 TensorRT execution failed")
        for i in self.output_indices:
            self.cuda.d2h(self.host[i], self.device[i])
        det = self.host[self.output_indices[0]].reshape(-1, self.host[self.output_indices[0]].shape[-1])
        proto = self.host[self.output_indices[1]]
        if proto.ndim == 4:
            proto = proto[0]
        if proto.shape[0] != 32 and proto.shape[-1] == 32:
            proto = proto.transpose(2, 0, 1)
        masks = [np.zeros((h, w), np.uint8) for _ in range(3)]
        if det.shape[1] < 5 + self.class_count + 1:
            raise RuntimeError("Unexpected YOLOv5 output width %d for %d classes" % (det.shape[1], self.class_count))
        coeff_start = 5 + self.class_count
        cls_probs = det[:, 5:coeff_start]
        classes = np.argmax(cls_probs, axis=1).astype(np.int32)
        scores = det[:, 4] * cls_probs[np.arange(det.shape[0]), classes]
        candidates = np.flatnonzero(scores >= self.conf)
        if candidates.size > self.max_detections:
            candidates = candidates[np.argsort(scores[candidates])[-self.max_detections:]]
        for index in candidates:
            row = det[index]
            cls = int(classes[index])
            score = float(scores[index])
            if cls not in (0, 1, 2):
                continue
            x, y, bw, bh = row[:4]
            x1 = int(np.clip((x - bw / 2.0) * w / self.size, 0, w - 1))
            y1 = int(np.clip((y - bh / 2.0) * h / self.size, 0, h - 1))
            x2 = int(np.clip((x + bw / 2.0) * w / self.size, 0, w))
            y2 = int(np.clip((y + bh / 2.0) * h / self.size, 0, h))
            if x2 <= x1 or y2 <= y1:
                continue
            coeff = row[coeff_start:coeff_start + proto.shape[0]]
            logits = np.matmul(coeff.astype(np.float32), proto.reshape(proto.shape[0], -1))
            mask = _sigmoid(logits).reshape(proto.shape[1:])
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            cropped = np.zeros_like(mask, dtype=np.uint8)
            cropped[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] * 255).astype(np.uint8)
            masks[cls] = np.maximum(masks[cls], cropped)
        return masks[0], masks[1], masks[2]

    def __del__(self):
        cuda = getattr(self, "cuda", None)
        if cuda is not None:
            self.context = None
            self.engine = None
            cuda.close()
