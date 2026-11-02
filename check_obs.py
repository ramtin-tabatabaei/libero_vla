"""Checks 1 and 2: is the observation we hand the policy actually live?

  1. proprioception changes when the end-effector moves
  2. camera frames are fresh, not stale or repeated

    MUJOCO_GL=egl /home/ramtin/miniconda3/envs/libero/bin/python check_obs.py
"""

import hashlib
import pathlib

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from shuffle_eval import _observation

OUT = pathlib.Path("obs_check")
LABELS = ["x", "y", "z", "ax", "ay", "az", "grip_l", "grip_r"]


def digest(a):
    return hashlib.sha1(np.ascontiguousarray(a)).hexdigest()[:12]


def main():
    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task = suite.get_task(0)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256, controller="OSC_POSE"
    )
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

    OUT.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- check 1
    print("=" * 72)
    print("CHECK 1  proprioception tensor vs end-effector motion")
    print("=" * 72)
    el_before, _ = _observation(obs, 224, "x")
    before = el_before["observation/state"]
    print(f"raw obs['robot0_eef_pos']      {np.round(obs['robot0_eef_pos'], 4)}")
    print(f"observation/state  dtype={before.dtype} shape={before.shape}")
    print("  " + "  ".join(f"{l}={v:+.4f}" for l, v in zip(LABELS, before)))

    # drive the arm up and back until the eef has moved ~10cm
    start = obs["robot0_eef_pos"].copy()
    for _ in range(40):
        obs, _, _, _ = env.step([-0.6, 0.0, 0.8, 0.0, 0.0, 0.0, -1.0])
        if np.linalg.norm(obs["robot0_eef_pos"] - start) > 0.10:
            break

    el_after, _ = _observation(obs, 224, "x")
    after = el_after["observation/state"]
    print(f"\nraw obs['robot0_eef_pos']      {np.round(obs['robot0_eef_pos'], 4)}")
    print("  " + "  ".join(f"{l}={v:+.4f}" for l, v in zip(LABELS, after)))
    moved = np.linalg.norm(obs["robot0_eef_pos"] - start)
    print(f"\neef moved {moved*100:.1f} cm")
    print(f"observation/state delta        {np.round(after - before, 4)}")
    print(f"  -> state {'CHANGES' if np.abs(after - before).max() > 1e-6 else 'IS STALE'}"
          f" (max |delta| {np.abs(after - before).max():.4f})")
    print(f"  -> state[:3] matches eef_pos: {np.allclose(after[:3], obs['robot0_eef_pos'])}")

    # ---------------------------------------------------------------- check 2
    print("\n" + "=" * 72)
    print("CHECK 2  camera frames immediately before each inference call")
    print("=" * 72)
    print(f"{'t':>3}  {'agentview sha1':>14} {'wrist sha1':>14}   "
          f"{'agent d(prev)':>13} {'wrist d(prev)':>13}")
    prev_a = prev_w = None
    for t in range(8):
        element, _ = _observation(obs, 224, "x")
        a, w = element["observation/image"], element["observation/wrist_image"]
        da = "-" if prev_a is None else f"{np.abs(a.astype(int) - prev_a.astype(int)).mean():.3f}"
        dw = "-" if prev_w is None else f"{np.abs(w.astype(int) - prev_w.astype(int)).mean():.3f}"
        print(f"{t:>3}  {digest(a):>14} {digest(w):>14}   {da:>13} {dw:>13}")
        imageio.imwrite(OUT / f"t{t}_agentview.png", a)
        imageio.imwrite(OUT / f"t{t}_wrist.png", w)
        prev_a, prev_w = a, w
        for _ in range(5):
            obs, _, _, _ = env.step([0.0, -0.5, -0.4, 0.0, 0.0, 0.0, -1.0])

    print(f"\nframes written to {OUT}/  (t0..t7, agentview + wrist)")
    env.close()


if __name__ == "__main__":
    main()
