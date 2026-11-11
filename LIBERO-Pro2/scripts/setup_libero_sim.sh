#!/usr/bin/env bash
# Prepare the official simulator environment and start its local evaluation API.
set -euo pipefail

task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bundle="$task_root/libero-pro-install/LIBERO-Pro-evaluation"
prefix="$bundle/.envs/libero-pro"
export CONDA_PKGS_DIRS="$task_root/.cache/conda/pkgs"
export PIP_CACHE_DIR="$task_root/.cache/pip"
export PIP_DEFAULT_TIMEOUT=120
export CONDA_REMOTE_READ_TIMEOUT_SECS=120

command -v conda >/dev/null || {
  echo "Conda must be available in this terminal." >&2
  exit 1
}

if [[ ! -f "$prefix/.libero-pro2-install-complete" ]]; then
  echo "Installing simulator dependencies (no datasets or model weights are downloaded)..."
  bash "$bundle/install.sh" "$prefix"
  touch "$prefix/.libero-pro2-install-complete"
fi

echo "Installing the lightweight OpenPI websocket client in the simulator environment..."
"$prefix/bin/python" -m pip install -e "$task_root/openpi/packages/openpi-client"

if [[ ! -f "$bundle/server.env" && ! -f "$bundle/client.env" ]]; then
  bash "$bundle/configure.sh" "$prefix"
elif [[ ! -f "$bundle/server.env" || ! -f "$bundle/client.env" ]]; then
  echo "Incomplete configuration: both server.env and client.env must exist." >&2
  exit 1
fi

bash "$bundle/start.sh"
bash "$bundle/healthcheck.sh"
set -a
source "$bundle/client.env"
set +a
"$prefix/bin/python" "$bundle/smoke_test.py"
echo "Simulator ready. Run scripts/run_pi05_task.py using $prefix/bin/python."
