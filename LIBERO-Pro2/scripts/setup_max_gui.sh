#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$task_root/openpi/.venv/bin/python"
export UV_CACHE_DIR="$task_root/.cache/uv"
uv pip install --python "$python" --target "$task_root/.max-runtime/python" --no-deps \
  robosuite==1.4.0 robomimic==0.2.0 bddl==1.0.1 easydict==1.9 gym==0.25.2 \
  mujoco==3.2.3 cloudpickle==2.1.0 future==0.18.2 Wand==0.7.2 \
  scikit-image==0.24.0 tifffile==2025.5.10 lazy-loader==0.4
"$python" "$task_root/scripts/prepare_max_runtime.py"
