#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_PATH="${1:-${SCRIPT_DIR}/artifacts/track_yolo26n_sem_best.onnx}"
ENGINE_PATH="${2:-${SCRIPT_DIR}/artifacts/track_yolo26n_sem_nano_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-300}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-60}"

test -f "${ONNX_PATH}" || { echo "Missing ONNX: ${ONNX_PATH}" >&2; exit 1; }
mkdir -p "$(dirname "${ENGINE_PATH}")"

echo "Jetson release: $(head -n 1 /etc/nv_tegra_release 2>/dev/null || echo unknown)"
if [[ ! -x "${TRTEXEC}" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
test -n "${TRTEXEC}" && test -x "${TRTEXEC}" || {
  echo "Missing trtexec. Set TRTEXEC=/path/to/trtexec inside the TensorRT container." >&2
  exit 1
}
"${TRTEXEC}" --help >/dev/null

echo "TensorRT executable: ${TRTEXEC}"
echo "TensorRT/CUDA libraries:"
ldconfig -p 2>/dev/null | grep -E 'libnvinfer\.so|libcublas\.so|libcudnn\.so' | head -20 || true

# Build and benchmark on the target board/container.  Never replace a known
# good engine until both parse/build and deserialization have succeeded.
TMP_ENGINE="${ENGINE_PATH}.tmp.$$"
trap 'rm -f "${TMP_ENGINE}"' EXIT
timeout --foreground "${BUILD_TIMEOUT}" "${TRTEXEC}" --onnx="${ONNX_PATH}" --saveEngine="${TMP_ENGINE}" \
  --explicitBatch --fp16 --workspace=512 --minTiming=1 --avgTiming=1
# CUDA graphs are optional on JetPack 4 and can make a valid engine appear to
# hang during startup on some driver/container combinations.  Enable them only
# when explicitly requested after the ordinary inference path is verified.
BENCH_FLAGS=(--loadEngine="${TMP_ENGINE}" --warmUp=1000 --duration="${BENCH_DURATION:-10}")
if [[ "${USE_CUDA_GRAPH:-0}" == "1" ]]; then
  BENCH_FLAGS+=(--useCudaGraph)
fi
timeout --foreground "${BENCH_TIMEOUT}" "${TRTEXEC}" "${BENCH_FLAGS[@]}"
mv -f "${TMP_ENGINE}" "${ENGINE_PATH}"
echo "Verified semantic engine: ${ENGINE_PATH}"
