#!/usr/bin/env bash
# Keep the LIBERO policy loaded until Ctrl+C. Run tasks in another terminal.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
POLICY="${POLICY:-pi05_libero}"
PORT="${PORT:-8000}"
CKPT="$HOME/.cache/openpi/openpi-assets/checkpoints/${POLICY}"
[ -d "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
if [ "$POLICY" = "pi05_libero" ]; then
    exec /home/ramtin/openpi/.venv/bin/python "$HERE/serve_pi05_libero.py" --port "$PORT" "$@"
fi
exec /home/ramtin/openpi/.venv/bin/python /home/ramtin/openpi/scripts/serve_policy.py \
    --env LIBERO --port "$PORT" \
    policy:checkpoint --policy.config "$POLICY" --policy.dir "$CKPT"
