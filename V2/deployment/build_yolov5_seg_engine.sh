#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
ONNX="${1:-$ROOT/models/yolov5n-seg-road.onnx}"
ENGINE="${2:-$ROOT/models/yolov5n-seg-road_fp16.engine}"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --explicitBatch --fp16 --workspace=256 --minTiming=1 --avgTiming=1
"$TRTEXEC" --loadEngine="$ENGINE" --warmUp=100 --duration=3
