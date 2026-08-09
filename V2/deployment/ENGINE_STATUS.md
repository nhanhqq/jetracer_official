# Engine status

`V2/models/waypoint_baseline_fp16_rebuilt.engine` is the newly compiled Jetson TensorRT
FP16 engine for the identical `waypoint_baseline.onnx` graph. It is stored in V2 so the notebook
does not depend on a file outside V2. The engine must be rebuilt on the target Nano if
the GPU or JetPack/TensorRT version changes.

The local TensorRT 7.1 parser accepts the waypoint ONNX. The rebuilt engine was validated
with `/usr/src/tensorrt/bin/trtexec`: TensorRT passed, GPU latency averaged 10.96 ms,
host latency 11.02 ms, end-to-end latency 11.03 ms, P99 end-to-end 18.75 ms, and
throughput was 90.69 qps with CUDA Graphs. This is the engine used by the notebook.

The official YOLOv5n-seg ONNX (`V2/models/yolov5n-seg.onnx`, opset 12) parses and
benchmarks on TensorRT 7.1 at about 10 ms GPU for 224x224. It is a COCO checkpoint and
is not used as road perception. The custom three-class checkpoint is being fine-tuned
from it; its artifacts will be `yolov5n-seg-road.onnx` and
`yolov5n-seg-road_fp16.engine`. Traffic sign/light classes are excluded.

Rebuild command for the waypoint engine:

```bash
sh V2/deployment/build_waypoint_engine.sh
```
