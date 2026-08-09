#!/bin/sh
set -eu
ONNX="${1:-V2/models/segmentation.onnx}"
ENGINE="${2:-V2/models/segmentation.engine}"
TRTEXEC="${TRTEXEC:-$(command -v trtexec 2>/dev/null || true)}"
[ -x "$TRTEXEC" ] || [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
[ -x "$TRTEXEC" ] || { echo 'trtexec not installed; build this on Jetson Nano with TensorRT.' >&2; exit 2; }
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --workspace=512 --fp16 --explicitBatch
