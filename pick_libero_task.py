"""Numbered LIBERO task menu and interactive MuJoCo scene viewer.

Run: GalaxeaVLA/.venv/bin/python pick_libero_task.py
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import textwrap

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / "GalaxeaVLA/experiments/libero"))
sys.path.insert(0, str(ROOT / "LIBERO"))
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def catalog():
    from libero.libero.benchmark import task_maps
    return [(suite, index, task) for suite in SUITES
            for index, task in enumerate(task_maps[suite].values())]


def show_menu(tasks):
    print("\nLIBERO — all 130 tasks (scene inspection, no policy)")
    previous = None
    for number, (suite, _, task) in enumerate(tasks, 1):
        if suite != previous:
            print(f"\n[{suite}]")
            previous = suite
        print(f"{number:3d}. {task.language}")
    print("\nEnter a task number; L lists tasks again; Q quits.")


def open_scene(selection, tasks, seconds=None):
    import mujoco
    import mujoco.viewer
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    suite_name, task_id, task = tasks[selection - 1]
    suite = benchmark.get_benchmark_dict()[suite_name]()
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    # The native viewer renders directly; observation cameras aren't needed.
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False,
                     has_renderer=False, has_offscreen_renderer=False,
                     ignore_done=True)
    try:
        env.seed(7)
        env.reset()
        env.set_init_state(suite.get_task_init_states(task_id)[0])
        action = np.zeros(env.env.action_dim)
        action[-1] = -1  # Hold pose with the gripper open.
        for _ in range(20):
            env.step(action)
        print(f"\nOpening {selection}: [{suite_name}] {task.language}", flush=True)
        print("Mouse drag: orbit/pan; wheel: zoom. Close the window to return to the menu.", flush=True)
        model, data = env.sim.model._model, env.sim.data._data
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT,
                               f"Task {selection} — {suite_name}",
                               textwrap.fill(task.language, width=65) + "\nScene inspection — no policy running")])
            start = time.monotonic()
            while viewer.is_running():
                tick = time.monotonic()
                if seconds is not None and tick - start >= seconds:
                    break
                with viewer.lock():
                    env.step(action)
                viewer.sync()
                time.sleep(max(0, env.env.control_timestep - (time.monotonic() - tick)))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print all tasks and exit")
    parser.add_argument("--open", type=int, metavar="NUMBER", help="Open a numbered task directly")
    parser.add_argument("--seconds", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    tasks = catalog()
    if args.open is not None:
        if not 1 <= args.open <= len(tasks):
            parser.error(f"Task number must be between 1 and {len(tasks)}")
        open_scene(args.open, tasks, args.seconds)
        return
    show_menu(tasks)
    if args.list:
        return
    while True:
        try:
            choice = input("\nTask number (1–130), L, or Q: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice in ("q", "quit", "exit"):
            break
        if choice == "l":
            show_menu(tasks)
            continue
        if not choice.isdigit() or not 1 <= int(choice) <= len(tasks):
            print("Please enter a number from 1 to 130.")
            continue
        # A fresh process keeps graphics resources isolated between scenes.
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--open", choice])
        if result.returncode:
            print(f"Viewer exited with code {result.returncode}. See the error above.")


if __name__ == "__main__":
    main()
