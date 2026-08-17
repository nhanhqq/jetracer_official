#!/bin/sh
set -eu
ENGINE="${1:-V2/models/waypoint_baseline_fp16.engine}"
TRTEXEC="${TRTEXEC:-$(command -v trtexec 2>/dev/null || true)}"
[ -x "$TRTEXEC" ] || [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
[ -x "$TRTEXEC" ] || { echo 'trtexec is required on the Jetson for a real benchmark.' >&2; exit 2; }
"$TRTEXEC" --loadEngine="$ENGINE" --warmUp=1000 --duration=10 --useCudaGraph
