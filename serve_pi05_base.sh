#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /home/ramtin/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec .venv/bin/python "$TASK_ROOT/serve_pi05_base.py" "$@"
