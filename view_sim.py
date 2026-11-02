"""Open a LIBERO task in MuJoCo's interactive viewer and look around.

This is the simulator UI, not the passive robosuite render window: free camera,
body tree, joint sliders, contact forces, pause / single-step, drag-to-perturb.

    MUJOCO_GL=glx python view_sim.py
    MUJOCO_GL=glx python view_sim.py --task-id 3
    MUJOCO_GL=glx python view_sim.py --task-suite-name libero_goal --task-id 2
    python view_sim.py --list-objects            # no window, just poses

Needs a real display. Over SSH you need X forwarding.

Mouse:  left-drag orbit | right-drag pan | scroll zoom | ctrl+drag shove a body
Panels: left = model tree + joint sliders, right = sim controls
Keys:   space pause | right-arrow single-step (while paused) | h help
"""

import dataclasses
import pathlib
import time
from typing import Optional

import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_object"
    task_id: int = 0
    seed: int = 0

    list_objects: bool = False
    """Print object names and world poses, then exit without opening a window."""

    freerun: bool = False
    """Step raw physics instead of holding the robot. The arm will sag under
    gravity -- there is no controller holding it up. Useful for watching objects
    settle, misleading if you are looking at the arm."""

    realtime: float = 1.0
    """Slow-motion factor. 0.25 = quarter speed."""


def main(args: Args) -> None:
    # Imported here so --list-objects works on machines with no GL at all.
    import mujoco
    import mujoco.viewer

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=128, camera_widths=128
    )
    env.seed(args.seed)
    obs = env.reset()

    print(f"\ntask [{args.task_id}] {args.task_suite_name}: {task.language}\n")
    print("objects in scene (world XYZ):")
    for k in sorted(obs):
        if k.endswith("_pos") and not k.startswith("robot0") and "_to_robot0" not in k:
            print(f"  {k[:-4]:26s} {np.round(obs[k], 4)}")
    print(f"  {'robot0_eef':26s} {np.round(obs['robot0_eef_pos'], 4)}")
    print(f"\n  obj_of_interest: {list(env.env.obj_of_interest)}")

    if args.list_objects:
        env.close()
        return

    # robosuite wraps MjModel/MjData; the viewer needs the raw structs.
    m = env.sim.model._model
    d = env.sim.data._data

    action_dim = env.env.action_dim
    dt = env.env.control_timestep

    print("\nopening viewer -- close the window to quit\n")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            t0 = time.time()

            if args.freerun:
                # Raw physics: nothing holds the arm up.
                mujoco.mj_step(m, d)
            else:
                # Zero action = controller holds the current pose, so the scene
                # stays as the task intends while you inspect it.
                env.step(np.zeros(action_dim))

            viewer.sync()

            if args.realtime > 0:
                slack = dt / args.realtime - (time.time() - t0)
                if slack > 0:
                    time.sleep(slack)

    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
