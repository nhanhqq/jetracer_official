#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENGINE="${1:-${SCRIPT_DIR}/artifacts/track_yolo26n_sem_nano_fp16.engine}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-45}"

if [[ ! -x "${TRTEXEC}" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
test -n "${TRTEXEC}" && test -x "${TRTEXEC}" || {
  echo "Missing trtexec; run this inside the TensorRT Docker container." >&2
  exit 1
}
test -s "${ENGINE}" || { echo "Missing/empty engine: ${ENGINE}" >&2; exit 1; }

echo "[0/3] CUDA device preflight"
if ! timeout --foreground 10 python3 - <<'PY'
import ctypes
c = ctypes.CDLL("libcudart.so.10.2")
n = ctypes.c_int()
rc = c.cudaGetDeviceCount(ctypes.byref(n))
if rc != 0 or n.value < 1:
    raise SystemExit("CUDA device unavailable (rc=%d, count=%d)" % (rc, n.value))
if c.cudaFree(None) != 0 or c.cudaDeviceSynchronize() != 0:
    raise SystemExit("CUDA context initialization failed")
print("CUDA device count:", n.value)
PY
then
  echo "ERROR: CUDA preflight failed. Reboot Jetson or repair the NVIDIA runtime before rebuilding." >&2
  exit 1
fi

echo "[1/3] TensorRT engine load + inference"
TRT_LOG="$(mktemp)"
trap 'rm -f "${TRT_LOG}"' EXIT
if ! timeout --foreground "${TIMEOUT_SECONDS}" "${TRTEXEC}" \
  --loadEngine="${ENGINE}" --warmUp=1000 --duration=3 2>&1 | tee "${TRT_LOG}"; then
  if grep -q 'linked against cuBLAS.*loaded cuBLAS' "${TRT_LOG}"; then
    echo "ERROR: TensorRT/cuBLAS version mismatch detected." >&2
    echo "Use matching JetPack R32.5.x TensorRT and cuBLAS libraries in the container; do not mix a newer TensorRT wheel/image with the host libraries." >&2
  elif grep -q 'CUDA Graph\|FAILED TensorRT' "${TRT_LOG}"; then
    echo "ERROR: TensorRT inference failed or timed out; engine was not accepted." >&2
  else
    echo "ERROR: TensorRT inference timed out/failed; inspect the log above." >&2
  fi
  exit 1
fi

echo "[2/3] Python/TensorRT version and library paths"
python3 - <<'PY'
import sys
import tensorrt as trt
print("python:", sys.version.split()[0])
print("tensorrt:", trt.__version__)
PY
ldconfig -p 2>/dev/null | grep -E 'libnvinfer\.so|libcublas\.so|libcudnn\.so' | head -20 || true

echo "[3/3] Ultralytics engine warmup (timeout ${TIMEOUT_SECONDS}s)"
cd "${ROOT_DIR}"
timeout "${TIMEOUT_SECONDS}" python3 - "${ENGINE}" <<'PY'
import sys
import numpy as np
from ultralytics import YOLO

engine = sys.argv[1]
model = YOLO(engine, task="semantic")
frame = np.zeros((224, 224, 3), dtype=np.uint8)
result = model.predict(frame, imgsz=224, device="0", verbose=False)[0]
mask = getattr(result, "semantic_mask", None)
if mask is None:
    raise RuntimeError("Ultralytics returned no semantic_mask")
print("semantic warmup OK; mask shape:", tuple(mask.data.shape))
PY
echo "Semantic TensorRT runtime verification passed."
