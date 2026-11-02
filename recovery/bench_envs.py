"""
Benchmark: how fast can we collect PPO rollout data from renderer-free LIBERO
ControlEnv instances running in separate subprocesses?

Measures aggregate env steps/s for N = 1, 4, 8, 16 workers, under both
multiprocessing start methods ('fork' and 'spawn'), and probes the
MuJoCo/EGL/robosuite gotchas empirically.

Run:
  MUJOCO_GL=egl /home/ramtin/miniconda3/envs/libero/bin/python recovery/bench_envs.py
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

# Keep BLAS/OMP single-threaded in every worker, otherwise 16 procs x 16 threads
# thrashes the 16 physical cores and throughput collapses.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")


# --------------------------------------------------------------------------
# helpers that must be importable in a 'spawn' child -> module level
# --------------------------------------------------------------------------
def _gl_probe():
    """Which GL libraries are actually mapped into this process?"""
    hits = set()
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                low = line.lower()
                for key in ("libegl", "libgl.so", "libglew", "libosmesa",
                            "libglfw", "libcuda.so", "libnvidia-egl"):
                    if key in low:
                        hits.add(key)
    except Exception:
        pass
    return sorted(hits)


def _make_env(bddl, horizon=1000):
    from libero.libero.envs.env_wrapper import ControlEnv
    return ControlEnv(
        bddl_file_name=bddl,
        controller="OSC_POSE",
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        camera_names=[],
        horizon=horizon,
    )


def _resolve_bddl(suite="libero_spatial", task_id=0):
    from libero.libero import benchmark, get_libero_path
    bdict = benchmark.get_benchmark_dict()
    s = bdict[suite]()
    task = s.get_task(task_id)
    return os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)


def worker(rank, bddl, n_steps, reset_every, seed, barrier, result_q, probe):
    """One env in one process: build, reset, step n_steps with random actions."""
    import numpy as np
    t_import0 = time.time()
    import robosuite  # noqa: F401
    import torch
    torch.set_num_threads(1)  # default is 8 here -> 16 workers x 8 threads = thrash
    t_import = time.time() - t_import0
    # fork() children inherit the parent's global RNG state verbatim: seed or
    # every worker explores with the identical action sequence.
    np.random.seed(seed)

    rng = np.random.default_rng(seed)

    t0 = time.time()
    env = _make_env(bddl)
    t_build = time.time() - t0

    t0 = time.time()
    env.reset()
    t_reset0 = time.time() - t0

    info = {}
    if probe:
        raw = env.env
        info["gl_libs"] = _gl_probe()
        info["mujoco_gl_envvar"] = os.environ.get("MUJOCO_GL")
        info["has_offscreen_ctx"] = getattr(raw.sim, "_render_context_offscreen", None) is not None
        info["has_window_ctx"] = getattr(raw.sim, "_render_context_window", None) is not None
        try:
            import mujoco
            info["mujoco_version"] = mujoco.__version__
        except Exception as e:
            info["mujoco_version"] = f"err:{e}"
        # is the process-global RNG actually distinct per worker?
        info["np_global_first"] = float(np.random.rand())
        info["obs_keys"] = sorted(env.reset().keys())[:4]

    # warm up JIT-ish paths, then sync all workers so we time true parallelism
    for _ in range(20):
        env.step(rng.uniform(-1, 1, 7))
    if barrier is not None:
        barrier.wait()

    n_resets = 0
    t_reset_total = 0.0
    t_start = time.time()
    steps = 0
    while steps < n_steps:
        a = rng.uniform(-1.0, 1.0, 7).astype("float64")
        env.step(a)
        steps += 1
        if reset_every and steps % reset_every == 0:
            tr = time.time()
            env.reset()
            t_reset_total += time.time() - tr
            n_resets += 1
    elapsed = time.time() - t_start

    env.close()
    result_q.put(dict(
        rank=rank, steps=steps, elapsed=elapsed,
        step_rate=steps / elapsed,
        t_build=t_build, t_reset0=t_reset0, t_import=t_import,
        n_resets=n_resets,
        t_reset_mean=(t_reset_total / n_resets) if n_resets else None,
        info=info,
    ))


def run_trial(method, n_workers, bddl, n_steps, reset_every, probe=False):
    ctx = mp.get_context(method)
    q = ctx.Queue()
    barrier = ctx.Barrier(n_workers)
    procs = []
    t0 = time.time()
    for r in range(n_workers):
        p = ctx.Process(target=worker,
                        args=(r, bddl, n_steps, reset_every, 1234 + r, barrier, q, probe))
        p.start()
        procs.append(p)
    results = [q.get() for _ in range(n_workers)]
    for p in procs:
        p.join()
    wall_total = time.time() - t0

    # aggregate throughput = total steps / slowest worker's timed window
    total_steps = sum(r["steps"] for r in results)
    slowest = max(r["elapsed"] for r in results)
    return dict(
        method=method, n=n_workers,
        total_steps=total_steps,
        slowest_elapsed=slowest,
        aggregate_sps=total_steps / slowest,
        per_worker_sps_mean=sum(r["step_rate"] for r in results) / len(results),
        per_worker_sps_min=min(r["step_rate"] for r in results),
        t_build_mean=sum(r["t_build"] for r in results) / len(results),
        t_reset0_mean=sum(r["t_reset0"] for r in results) / len(results),
        t_reset_mid_mean=(sum(r["t_reset_mean"] for r in results) / len(results))
        if results[0]["t_reset_mean"] is not None else None,
        wall_incl_startup=wall_total,
        results=results,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--reset-every", type=int, default=500)
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--ns", default="1,4,8,16")
    ap.add_argument("--methods", default="fork,spawn")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bddl = _resolve_bddl(args.suite, args.task_id)
    print(f"bddl = {bddl}")
    print(f"parent MUJOCO_GL={os.environ.get('MUJOCO_GL')}  "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"parent GL libs mapped BEFORE any env: {_gl_probe()}")

    out = {"bddl": bddl, "steps_per_worker": args.steps, "trials": []}
    first = True
    for method in args.methods.split(","):
        for n in [int(x) for x in args.ns.split(",")]:
            t = run_trial(method, n, bddl, args.steps, args.reset_every, probe=first)
            if first:
                print("\n--- probe from worker 0 ---")
                print(json.dumps(t["results"][0]["info"], indent=2))
                print("---------------------------\n")
                first = False
            print(f"[{method:5s}] N={n:2d}  aggregate={t['aggregate_sps']:8.1f} steps/s   "
                  f"per-env={t['per_worker_sps_mean']:6.1f}  "
                  f"build={t['t_build_mean']:.2f}s  reset0={t['t_reset0_mean']:.2f}s  "
                  f"reset_mid={t['t_reset_mid_mean']:.3f}s  "
                  f"wall_incl_startup={t['wall_incl_startup']:.1f}s")
            t.pop("results", None)
            out["trials"].append(t)
            sys.stdout.flush()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
