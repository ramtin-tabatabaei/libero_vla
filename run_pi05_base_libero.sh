#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT/GalaxeaVLA"
export LIBERO_CONFIG_PATH="$PWD/experiments/libero"
export PYTHONPATH="$PWD/src:/home/ramtin/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python experiments/libero/eval_libero_parallel.py \
  --policy_backend pi05 --server_uri ws://127.0.0.1:8001 \
  --task_suite_name libero_goal --task_id 2 \
  --num_trials_per_task 1 --num_parallel 1 --num_steps_wait 20 \
  --env_resolution 256 --live_view --save_videos \
  --output_dir outputs/pi05_base_wine "$@"
