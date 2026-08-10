#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_PATH="${1:-${SCRIPT_DIR}/artifacts/track_yolo26n_sem_best.onnx}"
ENGINE_PATH="${2:-${SCRIPT_DIR}/artifacts/track_yolo26n_sem_nano_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

test -f "${ONNX_PATH}" || { echo "Missing ONNX: ${ONNX_PATH}" >&2; exit 1; }
mkdir -p "$(dirname "${ENGINE_PATH}")"

echo "Jetson release: $(head -n 1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
if [[ "${BUILD_BACKEND:-python}" == "python" ]]; then
  python3 "${SCRIPT_DIR}/build_semantic_engine.py" \
    --onnx "${ONNX_PATH}" --engine "${ENGINE_PATH}" --workspace 512
else
  test -x "${TRTEXEC}" || { echo "Missing trtexec: ${TRTEXEC}" >&2; exit 1; }
  "${TRTEXEC}" --help >/dev/null
  "${TRTEXEC}" --onnx="${ONNX_PATH}" --saveEngine="${ENGINE_PATH}" \
    --explicitBatch --fp16 --workspace=512 --minTiming=1 --avgTiming=1
  "${TRTEXEC}" --loadEngine="${ENGINE_PATH}" --warmUp=1000 --duration=10 --useCudaGraph
fi
echo "Verified semantic engine: ${ENGINE_PATH}"
