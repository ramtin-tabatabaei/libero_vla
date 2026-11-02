#!/usr/bin/env bash
# Interactive RRT grasp -> pi0.5 LIBERO handoff in a single simulation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export MUJOCO_GL="${MUJOCO_GL:-glx}"
exec /home/ramtin/miniconda3/envs/libero/bin/python robot_server.py \
    --mode hybrid --policy pi05_libero \
    --task-suite-name libero_object --task-id 0 "$@"
