"""Does pi0.5 actually ground the instruction, or does it replay the task?

Four conditions on libero_object, all sharing one scene (7 objects, one basket):

  baseline     original layout,  original instruction   -- sanity check
  instruction  original layout,  instruction names a DIFFERENT object
  layout       shuffled layout,  original instruction
  both         shuffled layout,  instruction names a DIFFERENT object

"Shuffled layout" permutes the xy placements of the six groceries among each
other (z and orientation stay with the object, so nothing ends up inside the
table). The benchmark's own check_success() only ever scores the task's own
goal, so it cannot judge a swapped instruction -- instead we report which
object ended up in the basket and how far each one was moved.

    MUJOCO_GL=egl /home/ramtin/miniconda3/envs/libero/bin/python shuffle_eval.py

Needs an openpi policy server on :8000 (run.sh or robot_server.py start one).
"""

import collections
import dataclasses
import json
import math
import pathlib
import time
from typing import List, Optional

import imageio
import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
BASKET = "basket_1"

# in-basket test: horizontal distance to the basket centre, and a rise in z
IN_BASKET_XY = 0.10
MOVED = 0.05


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_object"
    task_id: int = 0
    """Scene + original instruction come from this task."""
    other_task_id: int = 7
    """The instruction borrowed for the 'instruction' / 'both' conditions.

    Must name an object that exists in task_id's scene -- libero_object task 0
    has alphabet soup, salad dressing, cream cheese, milk, tomato sauce, butter,
    so 1/2/5/6/7 work and the rest (bbq sauce, ketchup, ...) do not.
    """
    trials: int = 3
    conditions: Optional[List[str]] = None
    """Subset of baseline/instruction/layout/both. None runs all four."""

    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    max_steps: int = 280
    seed: int = 7
    out_dir: str = "shuffle_eval_out"
    save_video: bool = True


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=float).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _observation(obs, resize_size, instruction):
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
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": str(instruction),
    }
    return element, img


def _free_joint_slots(env):
    """{object name: index of its x in the flattened sim state}."""
    sim = env.env.sim
    slots = {}
    for name in env.env.objects_dict:
        jid = sim.model.joint_name2id(f"{name}_joint0")
        slots[name] = 1 + sim.model.jnt_qposadr[jid]  # +1: flattened state is [time, qpos, qvel]
    return slots


def _derangement(n, rng):
    """A permutation with no fixed point, so every object really moves."""
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def shuffle_state(init_state, slots, rng):
    """Permute the xy placements of the graspable objects among each other."""
    state = np.asarray(init_state, dtype=float).copy()
    names = sorted(n for n in slots if n != BASKET)
    xy = np.array([state[slots[n] : slots[n] + 2] for n in names])
    perm = _derangement(len(names), rng)
    for i, n in enumerate(names):
        state[slots[n] : slots[n] + 2] = xy[perm[i]]
    mapping = {n: names[perm[i]] for i, n in enumerate(names)}  # n now sits where mapping[n] was
    return state, mapping


def object_positions(obs, names):
    return {n: np.asarray(obs[f"{n}_pos"], dtype=float) for n in names}


def score(start, end):
    """Which object was moved, and which one ended up in the basket."""
    basket = end[BASKET]
    moved, in_basket = {}, []
    for n, p in end.items():
        if n == BASKET:
            continue
        d = float(np.linalg.norm(p - start[n]))
        moved[n] = d
        if np.linalg.norm(p[:2] - basket[:2]) < IN_BASKET_XY and p[2] > basket[2] and d > MOVED:
            in_basket.append(n)
    top = max(moved, key=moved.get)
    return {
        "moved": {n: round(d, 4) for n, d in sorted(moved.items(), key=lambda kv: -kv[1])},
        "most_moved": top if moved[top] > MOVED else None,
        "in_basket": in_basket,
    }


def run_episode(env, client, instruction, init_state, args):
    env.reset()
    obs = env.set_init_state(init_state)

    names = list(env.env.objects_dict)
    plan = collections.deque()
    frames = []
    for t in range(args.max_steps + args.num_steps_wait):
        if t < args.num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            if t == args.num_steps_wait - 1:
                start = object_positions(obs, names)  # after the objects have settled
            continue
        element, frame = _observation(obs, args.resize_size, instruction)
        frames.append(frame)
        if not plan:
            plan.extend(client.infer(element)["actions"][: args.replan_steps])
        obs, _, _, _ = env.step(plan.popleft().tolist())

    result = score(start, object_positions(obs, names))
    result["task_success"] = bool(env.check_success())
    return result, frames


def main(args: Args) -> None:
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    other = suite.get_task(args.other_task_id)
    init_states = suite.get_task_init_states(args.task_id)

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(args.seed)
    env.reset()
    slots = _free_joint_slots(env)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"scene   : task {args.task_id} of {args.task_suite_name}")
    print(f'original: "{task.language}"')
    print(f'borrowed: "{other.language}"  (task {args.other_task_id})')
    print(f"objects : {sorted(n for n in slots if n != BASKET)}\n")

    wanted = args.conditions or ["baseline", "instruction", "layout", "both"]
    records = []
    for cond in wanted:
        shuffled = cond in ("layout", "both")
        instruction = other.language if cond in ("instruction", "both") else task.language
        rng = np.random.default_rng(args.seed)
        print(f'=== {cond}: layout={"shuffled" if shuffled else "original"}  "{instruction}"')
        for ep in range(args.trials):
            state = np.asarray(init_states[ep], dtype=float)
            mapping = None
            if shuffled:
                state, mapping = shuffle_state(state, slots, rng)
            t0 = time.time()
            result, frames = run_episode(env, client, instruction, state, args)
            result.update(
                condition=cond, episode=ep, instruction=instruction, layout_swap=mapping,
                seconds=round(time.time() - t0, 1),
            )
            records.append(result)
            print(
                f"  ep{ep}: most moved = {result['most_moved']}"
                f"   in basket = {result['in_basket'] or '-'}"
                f"   task_success = {result['task_success']}   ({result['seconds']}s)"
            )
            if args.save_video and frames:
                imageio.mimwrite(out / f"{cond}_ep{ep}.mp4", [np.asarray(f) for f in frames], fps=20)
        print()

    (out / "results.json").write_text(json.dumps(records, indent=2))

    print("summary (target = object named in the instruction)")
    for cond in wanted:
        rows = [r for r in records if r["condition"] == cond]
        hits = sum(
            any(o.rsplit("_", 1)[0].replace("_", " ") in r["instruction"] for o in r["in_basket"])
            for r in rows
        )
        print(f"  {cond:12s} {hits}/{len(rows)} episodes put an instruction-named object in the basket")
    print(f"\nvideos + results.json in {out}/")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
