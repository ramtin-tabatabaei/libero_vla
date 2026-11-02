#!/usr/bin/env bash
# Run LIBERO in sim with a pi0 / pi0.5 VLA driving the robot.
#
#   ./run.sh                          # libero_object, task 0, 3 trials, pi05
#   ./run.sh --task-id 4              # a different task
#   ./run.sh --num-trials-per-task 10
#   ./run.sh --render --drop-object  # force a drop, watch pi05_libero recover
#   ./run.sh --drop-object --drop-delay-steps 20 --drop-open-steps 10
#   USE_EXISTING_SERVER=1 ./run.sh --render  # reuse serve_pi05_libero.sh
#   POLICY=pi0_libero ./run.sh        # the pi0 checkpoint instead of pi0.5
#
# Starts the openpi policy server (JAX, its own venv), waits for it to load the
# checkpoint, runs the rollout in the libero conda env, then shuts the server down.

set -euo pipefail

OPENPI_DIR=/home/ramtin/openpi
LIBERO_PY=/home/ramtin/miniconda3/envs/libero/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POLICY="${POLICY:-pi05_libero}"           # pi05_libero | pi0_libero
PORT="${PORT:-8000}"

# EGL is headless and cannot open a window, so --render forces GLX.
if [[ " $* " == *" --render "* ]]; then
    export MUJOCO_GL="${MUJOCO_GL:-glx}"
else
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

case "${USE_EXISTING_SERVER:-0}" in
    0)
        CKPT="$HOME/.cache/openpi/openpi-assets/checkpoints/${POLICY}"
        [ -d "$CKPT" ] || { echo "no checkpoint at $CKPT" >&2; exit 1; }

        LOG="$HERE/server.log"
        echo "starting policy server: $POLICY (log: $LOG)"

        "$OPENPI_DIR/.venv/bin/python" "$OPENPI_DIR/scripts/serve_policy.py" \
            --env LIBERO --port "$PORT" \
            policy:checkpoint --policy.config "$POLICY" --policy.dir "$CKPT" \
            > "$LOG" 2>&1 &
        SERVER_PID=$!
        trap 'echo; echo "stopping server ($SERVER_PID)"; kill $SERVER_PID 2>/dev/null || true' EXIT

        # Loading the checkpoint takes a while; poll the port rather than sleeping blind.
        SERVER_READY=0
        echo -n "waiting for server on :$PORT "
        for _ in $(seq 1 180); do
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                echo; echo "server died during startup -- last 30 lines of $LOG:" >&2
                tail -30 "$LOG" >&2
                exit 1
            fi
            if "$LIBERO_PY" - <<PY 2>/dev/null
import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", $PORT)) == 0 else 1)
PY
            then
                SERVER_READY=1
                echo " up"
                break
            fi
            echo -n "."
            sleep 2
        done
        if [[ "$SERVER_READY" != 1 ]]; then
            echo "policy server did not become ready on :$PORT; see $LOG" >&2
            exit 1
        fi
        ;;
    1)
        echo "using existing policy server on :$PORT (leave serve_pi05_libero.sh running)"
        ;;
    *)
        echo "USE_EXISTING_SERVER must be 0 or 1" >&2
        exit 1
        ;;
esac

# Anything passed on the command line comes last, so it overrides these.
cd "$HERE"
"$LIBERO_PY" run_vla.py \
    --task-suite-name "${TASK_SUITE:-libero_object}" \
    --task-id "${TASK_ID:-0}" \
    --num-trials-per-task "${TRIALS:-3}" \
    --port "$PORT" \
    "$@"
