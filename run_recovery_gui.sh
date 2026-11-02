#!/usr/bin/env bash
# Live LIBERO conditions. Existing plans/layouts are read; no results are saved.
set -euo pipefail
RECOVERY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RECOVERY_ROOT"
export MUJOCO_GL="${MUJOCO_GL:-glx}"
export PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/libero_vla_numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/libero_vla_matplotlib}"
exec "${LIBERO_PY:-/home/ramtin/miniconda3/envs/libero/bin/python}" \
    "$RECOVERY_ROOT/waypoint_recovery_gui.py" "$@"
