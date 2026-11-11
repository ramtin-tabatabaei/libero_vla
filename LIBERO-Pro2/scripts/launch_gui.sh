#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FONTCONFIG_FILE="$task_root/scripts/gui-fonts.conf"
python="$task_root/libero-pro-install/LIBERO-Pro-evaluation/.envs/libero-pro/bin/python"
if [[ ! -x "$python" ]]; then
  echo "Install the simulator first: bash scripts/setup_libero_sim.sh" >&2
  exit 1
fi
exec "$python" "$task_root/scripts/libero_gui.py" "$@"
