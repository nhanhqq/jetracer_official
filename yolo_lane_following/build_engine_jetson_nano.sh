#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_PATH="${1:-${SCRIPT_DIR}/artifacts/lane_yolo26n_seg_nano.onnx}"
ENGINE_PATH="${2:-${SCRIPT_DIR}/artifacts/lane_yolo26n_seg_nano_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

test -f "${ONNX_PATH}" || { echo "Missing ONNX: ${ONNX_PATH}" >&2; exit 1; }
test -x "${TRTEXEC}" || { echo "Missing trtexec: ${TRTEXEC}" >&2; exit 1; }

echo "Jetson release: $(head -n 1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
"${TRTEXEC}" --version
"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --explicitBatch \
  --fp16 \
  --workspace=512 \
  --verbose
"${TRTEXEC}" --loadEngine="${ENGINE_PATH}" --warmUp=1000 --duration=10 --useCudaGraph
echo "Verified engine: ${ENGINE_PATH}"
