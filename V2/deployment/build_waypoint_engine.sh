#!/bin/sh
set -eu
ONNX="${1:-V2/models/waypoint_baseline.onnx}"
ENGINE="${2:-V2/models/waypoint_baseline_fp16.engine}"
TRTEXEC="${TRTEXEC:-$(command -v trtexec 2>/dev/null || true)}"
if [ ! -x "$TRTEXEC" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then TRTEXEC=/usr/src/tensorrt/bin/trtexec; fi
[ -x "$TRTEXEC" ] || { echo 'TensorRT trtexec not found' >&2; exit 2; }
mkdir -p "$(dirname "$ENGINE")"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --explicitBatch --fp16 --workspace=512
"$TRTEXEC" --loadEngine="$ENGINE" --warmUp=100 --duration=3

