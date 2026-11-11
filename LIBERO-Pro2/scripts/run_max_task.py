#!/usr/bin/env python3
"""Run or preview one real LIBERO-Max source case using the existing pi0.5 server.

This is an interactive single-arm demonstration, not a matched-pair benchmark
score. Source scenarios and events are executed by the upstream MAX runtime.
"""
import argparse
import base64
from collections import deque
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / ".max-runtime/python"))


def main():
    from libero_gui_catalog import MaxCatalog, MAX_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["lite", "full"], default="lite")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--arm", choices=["control", "intervention"], default="intervention")
    parser.add_argument("--mode", choices=["preview", "run"], default="run")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--init-state-id", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--ipc", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.replan_steps < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Step counts must be positive")
    catalog = MaxCatalog(args.profile)
    case = next((c for c in catalog.raw_cases if c["case_id"] == args.case_id), None)
    if case is None:
        parser.error("This case is not in the selected manifest")
    runtime = ROOT / ".max-runtime"
    is_pro = bool(case.get("substrate_variant"))
    source = MAX_ROOT / ".deps/LIBERO-PRO/libero" if is_pro else runtime / "plus"
    source_inner = source / "libero"
    config = runtime / "config" / ("pro" if is_pro else "plus")
    config.mkdir(parents=True, exist_ok=True)
    data = MAX_ROOT / ".deps/libero-pro-data"
    paths = dict(assets=str(source_inner / "assets"), benchmark_root=str(source_inner),
                 bddl_files=str(data / "bddl_files" if is_pro else source_inner / "bddl_files"),
                 init_states=str(data / "init_files" if is_pro else source_inner / "init_files"),
                 datasets=str(data))
    (config / "config.yaml").write_text(json.dumps(paths))
    os.environ.update(LIBERO_SOURCE_PACKAGE_ROOT=str(source), LIBERO_CONFIG_PATH=str(config),
                      MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    sys.path[:0] = [str(MAX_ROOT / "scripts/libero_source_overlay"), str(MAX_ROOT / "src")]
    import numpy as np
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_max.substrate import load_case_task
    from libero_max.pro_runtime import wrap_case_env
    from libero_max.cosmos_integration import CosmosInterventionEnv
    from libero_max.env_factory import create_libero_env_with_retry
    from run_pi05_task import model_observation, LIMITS, LivePreview, PreviewStopped
    from libero_gui_transport import OpenPiClient, Cancelled

    stop = threading.Event()
    if args.ipc:
        def listen():
            for line in sys.stdin:
                if line.strip() == "STOP":
                    stop.set()
                    return
        threading.Thread(target=listen, daemon=True).start()
    def emit(kind, **payload):
        print(json.dumps(dict(type=kind, **payload)), flush=True)
    preview = LivePreview("LIBERO-Max · " + args.case_id) if args.live else None
    def frame(observation, step):
        if stop.is_set():
            raise Cancelled()
        if preview:
            preview.write(observation)
        if args.ipc:
            encoded = []
            for key in ["agentview_image", "robot0_eye_in_hand_image"]:
                stream = BytesIO()
                Image.fromarray(observation[key]).save(stream, format="JPEG", quality=85)
                encoded.append(base64.b64encode(stream.getvalue()).decode())
            emit("frame", images=encoded, step=step)
    seed = case["policy_seed"] if args.seed is None else args.seed
    init_id = case["init_state_index"] if args.init_state_id is None else args.init_state_id
    steps = args.max_steps or LIMITS[case["task_suite_name"]]
    def reseed(value):
        random.seed(value)
        np.random.seed(value)
    reseed(seed)
    task, states = load_case_task(case, benchmark)
    if not 0 <= init_id < len(states):
        raise ValueError(f"Initial state must be between 0 and {len(states)-1}")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    emit("status", text="Loading LIBERO-Max source scene…")
    env, _ = create_libero_env_with_retry(lambda: (OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"]), task.language),
        policy_seed=seed, reseed=reseed)
    policy = None
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    trace = ROOT / "results/max-traces" / f"{stamp}.jsonl"
    try:
        env = wrap_case_env(env, case)
        env.seed(seed)
        wrapped = CosmosInterventionEnv(env, task.language, case["scenario"], args.arm,
            trace, case["task_index"], init_state_index=init_id)
        wrapped.configure_episode(case["task_suite_name"], seed, args.replan_steps, steps)
        wrapped.reset()
        observation = wrapped.set_init_state(states[init_id])
        for _ in range(wrapped.warmup_steps):
            if stop.is_set(): raise Cancelled()
            observation, _, _, _ = wrapped.step([0, 0, 0, 0, 0, 0, -1])
        emit("instruction", text=wrapped.runtime.current_instruction)
        frame(observation, 0)
        if args.mode == "preview":
            emit("preview")
            return
        policy = OpenPiClient(args.host, args.port, stop)
        plan, success = deque(), False
        emit("status", text="Running LIBERO-Max")
        event_count = 0
        completed_steps = 0
        for step in range(steps):
            if stop.is_set(): raise Cancelled()
            if not plan:
                canonical = dict(observation)
                canonical["robot0_eef_quat_xyzw"] = observation["robot0_eef_quat"]
                canonical["instruction"] = wrapped.runtime.current_instruction
                chunk = np.asarray(policy.infer(model_observation(canonical))["actions"], dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < args.replan_steps or not np.isfinite(chunk).all():
                    raise ValueError("Invalid policy action chunk")
                wrapped.record_policy_query(chunk, instruction=canonical["instruction"], source="model")
                plan.extend(chunk[:args.replan_steps].copy())
            action = plan.popleft()
            observation, _, done, _ = wrapped.step(action.tolist())
            completed_steps = step + 1
            emit("action", values=action.tolist())
            frame(observation, completed_steps)
            if len(wrapped.events) != event_count:
                event_count = len(wrapped.events)
                emit("intervention", text="Selected condition has been applied", events=wrapped.events[-1:])
            if done:
                success = True
                break
        outcome = wrapped.record_outcome(success)
        result = dict(benchmark=catalog.display_name, suite=case["task_suite_name"], task_id=case["task_index"],
            case_id=case["case_id"], variant=case["scenario"]["change_type"], arm=args.arm, seed=seed,
            init_state_id=init_id, success=bool(outcome["success"]), steps=completed_steps,
            duration_seconds=time.monotonic()-started, event_applied=wrapped.runtime.applied,
            replan_steps=args.replan_steps, policy_rng_reset=False, purpose="single-arm interactive demonstration",
            source_category=case["substrate_category"], trace_path=str(trace))
        path = ROOT / "results" / f"gui_max_{stamp}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        emit("result", result=result, path=str(path))
        if preview:
            preview.finish(result["success"])
    except (Cancelled, PreviewStopped):
        emit("stopped")
    finally:
        if policy:
            policy._ws.close()
        env.close()
        if preview:
            preview.close()


if __name__ == "__main__":
    main()
