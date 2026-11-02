"""Empirical probes for the MuJoCo / EGL / robosuite / multiprocessing gotchas.

Run each probe in its own subprocess so the environments don't contaminate
each other:
  /home/ramtin/miniconda3/envs/libero/bin/python recovery/probe_gotchas.py <probe>
"""
import os
import sys
import time


def gl_probe():
    hits = set()
    with open("/proc/self/maps") as f:
        for line in f:
            low = line.lower()
            for key in ("libegl", "libgl.so", "libglew", "libosmesa", "libglfw",
                        "libcuda.so", "libnvidia-egl", "libgldispatch"):
                if key in low:
                    hits.add(key)
    return sorted(hits)


def _bddl():
    from libero.libero import benchmark, get_libero_path
    s = benchmark.get_benchmark_dict()["libero_spatial"]()
    t = s.get_task(0)
    return os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)


def _env(bddl):
    from libero.libero.envs.env_wrapper import ControlEnv
    return ControlEnv(bddl_file_name=bddl, controller="OSC_POSE", use_camera_obs=False,
                      has_renderer=False, has_offscreen_renderer=False,
                      camera_names=[], horizon=1000)


# ---------------------------------------------------------------- probes
def p_gl():
    """Is any GL context created at all when use_camera_obs=False?"""
    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    print("0. bare python      :", gl_probe())
    import mujoco
    print("1. after import mujoco:", gl_probe())
    import robosuite  # noqa
    print("2. after import robosuite:", gl_probe())
    import torch  # noqa
    print("3. after import torch:", gl_probe())
    from libero.libero import benchmark  # noqa
    print("4. after import libero.benchmark:", gl_probe())
    e = _env(_bddl())
    e.reset()
    for _ in range(5):
        e.step([0] * 7)
    print("5. after env build+step:", gl_probe())
    raw = e.env
    print("   sim._render_context_offscreen =", getattr(raw.sim, "_render_context_offscreen", "N/A"))
    print("   sim._render_context_window    =", getattr(raw.sim, "_render_context_window", "N/A"))
    try:
        import mujoco.egl
        print("   mujoco.egl importable:", True)
    except Exception as ex:
        print("   mujoco.egl import err:", ex)
    # does MjRenderContext exist anywhere?
    print("   has_offscreen_renderer flag  =", raw.has_offscreen_renderer)
    print("   use_camera_obs flag          =", raw.use_camera_obs)
    e.close()
    print("6. after close:", gl_probe())


def p_fork():
    """Does a MuJoCo/robosuite env survive fork()? Two cases:
    A) parent builds env THEN forks (child inherits a live MjModel/MjData)
    B) parent imports only, child builds its own env
    """
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    bddl = _bddl()

    def _case_b(q):
        try:
            e = _env(bddl)
            e.reset()
            for _ in range(50):
                e.step([0.0] * 7)
            q.put(("B-ok", float(e.env.sim.data.qpos[0])))
        except Exception as ex:
            q.put(("B-FAIL", repr(ex)))

    q = ctx.Queue()
    p = ctx.Process(target=_case_b, args=(q,))
    p.start()
    p.join(120)
    print("case B (child builds own env after fork):", q.get() if not q.empty() else "TIMEOUT/CRASH",
          "exitcode=", p.exitcode)

    # now case A: build in parent first, then fork
    parent_env = _env(bddl)
    parent_env.reset()
    for _ in range(10):
        parent_env.step([0.0] * 7)

    def _case_a(q2):
        try:
            for _ in range(50):
                parent_env.step([0.0] * 7)  # inherited env, copy-on-write
            q2.put(("A-ok-inherited", float(parent_env.env.sim.data.qpos[0])))
            e2 = _env(bddl)  # AND build a fresh one in the forked child
            e2.reset()
            for _ in range(50):
                e2.step([0.0] * 7)
            q2.put(("A-ok-newenv-after-fork", float(e2.env.sim.data.qpos[0])))
        except Exception as ex:
            q2.put(("A-FAIL", repr(ex)))

    q2 = ctx.Queue()
    p2 = ctx.Process(target=_case_a, args=(q2,))
    p2.start()
    p2.join(180)
    got = []
    while not q2.empty():
        got.append(q2.get())
    print("case A (fork AFTER parent built an env):", got, "exitcode=", p2.exitcode)
    print("parent still alive after fork? stepping...")
    for _ in range(10):
        parent_env.step([0.0] * 7)
    print("   parent OK, qpos0 =", float(parent_env.env.sim.data.qpos[0]))


def _seed_worker(rank, qq, bddl, do_seed):
    import numpy as np
    if do_seed:
        np.random.seed(1000 + rank)
    e = _env(bddl)
    o = e.reset()
    qq.put((rank,
            float(np.random.rand()),                  # global numpy RNG
            round(float(np.sum(o["robot0_joint_pos"])), 6),
            round(float(np.sum(e.env.sim.data.qpos)), 6)))
    e.close()


def p_seed():
    """Do forked children share RNG state / initial qpos? Does seeding matter?"""
    import multiprocessing as mp
    import numpy as np
    bddl = _bddl()
    for method, do_seed in (("fork", False), ("spawn", False), ("fork", True)):
        ctx = mp.get_context(method)
        q = ctx.Queue()
        ps = [ctx.Process(target=_seed_worker, args=(i, q, bddl, do_seed)) for i in range(3)]
        [p.start() for p in ps]
        res = sorted(q.get() for _ in range(3))
        [p.join() for p in ps]
        print(f"{method} (explicit per-rank seed={do_seed}): "
              f"(rank, np.random.rand(), sum joint_pos, sum qpos)")
        for r in res:
            print("   ", r)


def p_benchmark_global():
    """Is there global state in libero's benchmark loader that breaks under MP?
    It calls torch.load on init-state .pruned_init files."""
    import multiprocessing as mp
    import torch
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    print("BENCHMARK_MAPPING keys:", sorted(bd.keys()))
    s = bd["libero_spatial"]()
    print("n_tasks:", s.n_tasks)
    t0 = time.time()
    st = s.get_task_init_states(0)
    print("get_task_init_states(0): shape", st.shape, "dtype", st.dtype,
          "type", type(st), "took %.3fs" % (time.time() - t0))
    print("torch threads:", torch.get_num_threads())

    # does constructing the benchmark object in 8 forked children work / cost?
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def _w(rank, qq):
        t = time.time()
        from libero.libero import benchmark as b
        s2 = b.get_benchmark_dict()["libero_spatial"]()
        st2 = s2.get_task_init_states(rank % 10)
        qq.put((rank, tuple(st2.shape), round(time.time() - t, 3),
                round(float(st2.sum()), 4)))

    ps = [ctx.Process(target=_w, args=(i, q)) for i in range(8)]
    t0 = time.time()
    [p.start() for p in ps]
    res = sorted(q.get() for _ in range(8))
    [p.join() for p in ps]
    print("8 forked children each loading benchmark+init_states: %.2fs total" % (time.time() - t0))
    for r in res:
        print("   ", r)
    print("exitcodes:", [p.exitcode for p in ps])


def p_reset_cost():
    """How expensive is reset(), and reset()+set_init_state()?"""
    import numpy as np
    from libero.libero import benchmark
    s = benchmark.get_benchmark_dict()["libero_spatial"]()
    init_states = s.get_task_init_states(0)
    e = _env(_bddl())
    t = time.time(); e.reset(); print("first reset: %.3fs" % (time.time() - t))
    ts = []
    for i in range(10):
        t = time.time(); e.reset(); ts.append(time.time() - t)
    print("plain reset:            mean %.4fs  (n=10)" % (sum(ts) / len(ts)))
    ts = []
    for i in range(10):
        t = time.time()
        e.reset()
        e.set_init_state(init_states[i % len(init_states)])
        ts.append(time.time() - t)
    print("reset + set_init_state: mean %.4fs  (n=10)" % (sum(ts) / len(ts)))
    # step cost at different action magnitudes
    for scale, name in [(0.0, "zero"), (0.2, "small"), (1.0, "full-random")]:
        e.reset()
        rng = np.random.default_rng(0)
        for _ in range(20):
            e.step(rng.uniform(-scale, scale, 7))
        t = time.time()
        n = 600
        for _ in range(n):
            e.step(rng.uniform(-scale, scale, 7))
        dt = time.time() - t
        print(f"step rate, action scale {name:12s}: {n/dt:6.1f} steps/s ({1000*dt/n:.2f} ms/step)")
    e.close()


def p_env_var_matters():
    """Does MUJOCO_GL setting change anything when there is no renderer?"""
    import numpy as np
    t = time.time()
    e = _env(_bddl())
    build = time.time() - t
    e.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):
        e.step(rng.uniform(-1, 1, 7))
    t = time.time()
    for _ in range(400):
        e.step(rng.uniform(-1, 1, 7))
    dt = time.time() - t
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')!r:12s} build={build:.2f}s "
          f"rate={400/dt:6.1f} steps/s  gl_libs={gl_probe()}")
    e.close()


if __name__ == "__main__":
    globals()["p_" + sys.argv[1]]()
