# JetRacer Lane Fusion V2

V2 is an additive implementation. `waypoint_lane_fusion/` is treated as a read-only
baseline and is never imported by mutation. The runtime uses a local, perspective-aware
road corridor as a safety constraint around the existing center-divider waypoint model.

## Pipeline

`frame -> road/outside/marking perception -> horizontal-band corridor -> waypoint validation -> local trajectory -> rate-limited steering/throttle -> next frame`

The segmentation model is explicitly **YOLOv5n-seg (v7.0 or lower)** for Jetson Nano
compatibility. The custom dataset has only `road`, `divider`, and `obstacle`; traffic
signs and traffic lights are intentionally ignored. `V2/segmentation/yolov5_trt.py`
runs the engine without Ultralytics. CV remains only as the hard white-forbidden safety
check and road fallback.

## Dataset and pseudo labels

Inspection found 689 224x224 images under `yolo_lane_following/dataset` and 92 legacy
224x224 images in `notebook3/old_codes/road_following_A`. The copied YOLOv5 dataset in
`V2/data/yolov5_seg_dataset` contains only label IDs 0/1/2 for road/divider/obstacle.
The pseudo-label generator
uses HSV/LAB/gray thresholds, white forbidden masks, morphology, and a connected
component touching bottom-center. Isolated dark blobs outside the track are rejected.

```bash
python3 V2/scripts/generate_pseudolabels.py
python3 V2/scripts/infer_frames.py --source notebook3
```

Masks are binary road/outside/marking artifacts. They are bootstrapping labels, not
ground truth; a manually reviewed subset is still required before competition training.

The YOLOv5n-seg fine-tune used 689 images, 3 classes, 224x224 input, batch 2 and 5
epochs from the official nano checkpoint. The resulting model has about 1.89M parameters
and is stored as `models/yolov5n-seg-road.onnx` and
`models/yolov5n-seg-road_fp16.engine`.

## Video inference and evaluation

The two original test videos contain 498 and 102 frames at 1280x720. Run both:

```bash
python3 V2/scripts/run_video.py --source notebook3/test/1786085420908_202326730929621029_7442541399315262114.mp4 --output-video V2/results/videos/long.mp4 --log V2/results/logs/long.csv
python3 V2/scripts/run_video.py --source notebook3/test/1786085420913_202326730929621029_7442541399315262114.mp4 --output-video V2/results/videos/short.mp4 --log V2/results/logs/short.csv
python3 V2/scripts/evaluate.py
```

Each log contains waypoint/corrected target, confidence, occupancy, road center,
curvature, heading, white intrusion, steering, throttle, state, and warning. Overlay
videos show road, forbidden white, orange marking, corridor center and commands.

## Control and recovery

Steering is a filtered PD plus heading term and symmetric white-space repulsion. Throttle
is reduced by steering load, curvature, confidence and white intrusion, with separate
acceleration/deceleration rate limits. The state machine is `NORMAL -> CAUTION -> STOP ->
RECOVERY_REVERSE -> RECOVERY_TURN -> REACQUIRE`; recovery direction is selected from
which side still contains road, and every transition is reevaluated from new frames.

## Deployment

Open `V2/live_lane_fusion.ipynb` on the Jetson for the CSI camera callback runner.
It follows the existing baseline notebook pattern, but uses the V2 corridor safety
controller and exposes an ARM checkbox plus an explicit STOP cell. The notebook first
looks for `models/waypoint_baseline_fp16.engine`, then falls back to ONNX, then to
geometry-only if the corresponding runtime is unavailable.

Export the custom YOLOv5 checkpoint and build the engine on the target Nano; never copy
an engine across GPU/runtime versions:

```bash
python3 V2/third_party/yolov5/export.py --weights V2/results/yolov5_train/lane_v5n_seg_b2/weights/best.pt --include onnx --imgsz 224 224 --batch-size 1 --device cpu --opset 12
cp V2/results/yolov5_train/lane_v5n_seg_b2/weights/best.onnx V2/models/yolov5n-seg-road.onnx
sh V2/deployment/build_yolov5_seg_engine.sh
```

Expected deployment input is static batch 1, RGB `1x3x224x224`, FP16 where supported.
The validated waypoint benchmark is recorded in
`results/evaluation/waypoint_trt_summary.json`. Rebuild/benchmark with
`sh V2/deployment/build_waypoint_engine.sh` on the target Nano.

Measured on this Jetson: TensorRT-only waypoint inference reached 90.69 qps
(10.96 ms GPU compute with CUDA Graphs), while the complete Python loop including CV road safety,
geometry, fusion and control reached about 20.0 FPS median on a test video. Camera
callback overhead and CSI delivery can change the final live rate.

The custom YOLOv5 TensorRT engine was independently loaded with `trtexec`: 97.87 qps,
9.96 ms mean GPU compute, 10.21 ms end-to-end mean, P99 end-to-end 11.73 ms. The
complete Python loop with waypoint TensorRT, YOLOv5 mask postprocessing, geometry and
control measured 7.95 FPS after top-8 candidate limiting.

Replay artifacts produced by the current backend are:
`results/videos/yolov5_fusion_short_top8.mp4`,
`results/videos/yolov5_fusion_long_top8.mp4`, and matching CSV files in
`results/logs/`. All 825 collected images were processed into
`results/frame_masks_yolov5/`.

## Limitations and comparison

The existing waypoint ONNX is preserved and referenced, but `onnxruntime` is absent so
the offline runner currently derives a waypoint from orange center-marking geometry.
The video outputs therefore validate safety geometry/control and not the neural
waypoint branch. Baseline-vs-fusion quantitative comparison must be rerun on Nano (or
an environment with the original ONNX runtime) with identical frame timestamps.
The current checkpoint is a short Jetson fine-tune and its raw confidence is still low
on difficult frames; CV white detection and corridor validation remain mandatory safety
layers. Obstacle avoidance currently reduces throttle and applies a bounded lateral
offset, but it is not a full obstacle tracker. Live motor operation remains arm-gated
and requires supervised low-throttle validation.
