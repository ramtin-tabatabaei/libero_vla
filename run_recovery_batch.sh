#!/usr/bin/env bash
# Save all live-GUI recovery combinations; resume by reusing --output.
set -euo pipefail
RECOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RECOVERY_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/libero_vla_numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/libero_vla_matplotlib}"
exec "${LIBERO_PY:-/home/ramtin/miniconda3/envs/libero/bin/python}" \
    "$RECOVERY_ROOT/waypoint_recovery_batch.py" "$@"
