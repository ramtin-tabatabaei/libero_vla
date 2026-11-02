"""Run LIBERO in simulation with a pi0 / pi0.5 VLA driving the Panda.

The VLA runs in a separate process (openpi policy server, JAX) and is reached
over a websocket; this process owns the MuJoCo sim and does the stepping.
Start the server first, or just use run.sh which does both.

Observation/action contract matches openpi's examples/libero/main.py exactly --
images rotated 180 deg, padded-resized to 224, state = eef_pos + axis-angle +
gripper qpos. Deviating from it silently tanks the success rate.
"""

import collections
import dataclasses
import logging
import math
import pathlib
from typing import List, Optional

import imageio
import numpy as np
import tqdm
import tyro
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# 6-DoF zero delta + gripper open; used while the sim settles.
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution the training data was rendered at

# Longest training demo per suite, rounded up.
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_object"
    task_id: Optional[int] = None
    """Single task to run. None runs every task in the suite."""
    num_trials_per_task: int = 5

    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    """Steps of each predicted chunk to execute before re-querying the policy."""

    num_steps_wait: int = 10
    """The sim drops objects on reset; hold still until they land."""
    seed: int = 7
    video_out_path: str = "videos"
    save_video: bool = True

    drop_object: bool = False
    """Force a one-time gripper opening after grasping, then resume policy control."""
    drop_delay_steps: int = 10
    """Normal control steps after two consecutive grasp contacts before opening."""
    drop_open_steps: int = 10
    """Control steps to force the gripper open."""
    drop_min_lift: float = 0.03
    """Required object lift in meters before arming the drop; 0 disables lift check."""

    render: bool = False
    """Open a live MuJoCo window. Needs MUJOCO_GL=glx (run.sh sets it for you)."""


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils (xyzw ordering)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _make_env(task, resolution, seed, render=False):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kwargs = dict(
        bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution
    )
    if render:
        # ControlEnv rather than OffScreenRenderEnv: the latter hardcodes
        # has_renderer=False. We still need the offscreen cameras for the policy.
        env = ControlEnv(has_renderer=True, has_offscreen_renderer=True, **kwargs)
    else:
        env = OffScreenRenderEnv(**kwargs)
    # Seeding matters even with fixed initial states -- it moves object positions.
    env.seed(seed)
    return env


def _observation(obs, resize_size, task_description):
    # 180 deg rotation to match how the training data was rendered.
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist, resize_size, resize_size)
    )
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
        "prompt": str(task_description),
    }
    return element, img


def run_episode(env, client, task_description, init_state, args, max_steps):
    env.reset()
    obs = env.set_init_state(init_state)

    action_plan = collections.deque()
    frames = []
    done = False
    if args.drop_object:
        logging.info("Drop enabled: waiting for grasp and %.3f m lift; delay=%d, open=%d steps",
                     args.drop_min_lift, args.drop_delay_steps, args.drop_open_steps)

    for t in range(max_steps + args.num_steps_wait):
        if args.render:
            env.env.render()

        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            continue

        element, frame = _observation(obs, args.resize_size, task_description)
        frames.append(frame)

        if not action_plan:
            chunk = client.infer(element)["actions"]
            if len(chunk) < args.replan_steps:
                raise RuntimeError(
                    f"policy returned {len(chunk)} actions, need {args.replan_steps}"
                )
            action_plan.extend(chunk[: args.replan_steps])

        obs, _, done, info = env.step(action_plan.popleft().tolist())
        if info.get("drop_event"):
            logging.info("step %d forced drop: %s", t - args.num_steps_wait, info["drop_event"])
        if done:
            break

    if args.drop_object and not env.triggered:
        logging.warning("Drop did NOT trigger before episode ended (object=%s, countdown=%s)",
                        env.object_name, env.countdown)
    return done, frames


def main(args: Args) -> None:
    from libero.libero import benchmark
    from GalaxeaVLA.experiments.libero.drop_object import DropObject

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(args.seed)

    if args.drop_delay_steps < 0 or args.drop_open_steps < 1 or not math.isfinite(args.drop_min_lift) or args.drop_min_lift < 0:
        raise ValueError("Drop delay and minimum lift must be nonnegative; open steps must be >= 1")

    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"unknown suite {args.task_suite_name}; pick one of {list(MAX_STEPS)}")
    max_steps = MAX_STEPS[args.task_suite_name]

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = [args.task_id] if args.task_id is not None else range(task_suite.n_tasks)

    out_dir = pathlib.Path(args.video_out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info(f"connected to policy server at {args.host}:{args.port}")
    logging.info(f"server metadata: {client.get_server_metadata()}")

    total, successes = 0, 0
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env = _make_env(task, LIBERO_ENV_RESOLUTION, args.seed, render=args.render)
        if args.drop_object:
            env = DropObject(env, args.drop_delay_steps, args.drop_open_steps, args.drop_min_lift)

        task_total, task_successes = 0, 0
        desc = f"[{task_id}] {task.language}"
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=desc):
            try:
                done, frames = run_episode(
                    env, client, task.language, init_states[episode_idx], args, max_steps
                )
            except Exception as e:
                logging.error(f"  episode {episode_idx} failed: {e}")
                continue

            task_total += 1
            total += 1
            task_successes += int(done)
            successes += int(done)

            if args.save_video and frames:
                tag = "success" if done else "failure"
                name = task.language.replace(" ", "_").replace("/", "_")
                imageio.mimwrite(
                    out_dir / f"{args.task_suite_name}_t{task_id}_e{episode_idx}_{tag}.mp4",
                    [np.asarray(f) for f in frames],
                    fps=10,
                )

        env.close()
        if task_total:
            logging.info(f"{desc}: {task_successes}/{task_total} = {task_successes / task_total:.0%}")

    if total:
        logging.info(f"\nTOTAL: {successes}/{total} = {successes / total:.1%}")
    else:
        logging.warning("no episodes completed")


if __name__ == "__main__":
    main(tyro.cli(Args))
