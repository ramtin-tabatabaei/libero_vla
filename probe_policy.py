"""Is the policy actually a function of what we send it?

Queries the running policy server with one fixed observation and varies one
thing at a time. If the actions were a canned trajectory, every chunk would come
back identical no matter what we sent.

    MUJOCO_GL=egl /home/ramtin/miniconda3/envs/libero/bin/python probe_policy.py
"""

import pathlib

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy

from shuffle_eval import _observation

SUITE, TASK_ID = "libero_object", 0
PROMPT_A = "pick up the butter and place it in the basket"
PROMPT_B = "pick up the milk and place it in the basket"


def main():
    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(TASK_ID)[0])
    for _ in range(10):  # let the objects settle
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

    client = websocket_client_policy.WebsocketClientPolicy("0.0.0.0", 8000)
    base, _ = _observation(obs, 224, PROMPT_A)

    def ask(element):
        return np.asarray(client.infer(element)["actions"], dtype=float)

    # The policy samples, so one query per condition cannot separate a real
    # effect from sampling noise. Draw N per condition and compare the
    # between-condition distance against the within-baseline distance.
    N = 10
    conditions = {
        "identical input (noise floor)": base,
        "different prompt": {**base, "prompt": PROMPT_B},
        "agentview image blanked": {**base, "observation/image": np.zeros_like(base["observation/image"])},
        "wrist image blanked": {**base, "observation/wrist_image": np.zeros_like(base["observation/wrist_image"])},
        "eef state shifted 10cm": {**base, "observation/state": base["observation/state"] + np.r_[0.1, 0.1, 0.1, np.zeros(5)]},
        "eef state shifted 50cm": {**base, "observation/state": base["observation/state"] + np.r_[0.5, 0.5, 0.5, np.zeros(5)]},
        "state zeroed entirely": {**base, "observation/state": np.zeros_like(base["observation/state"])},
        "gripper state flipped": {**base, "observation/state": np.r_[base["observation/state"][:6], -base["observation/state"][6:]]},
    }
    samples = {name: [ask(el) for _ in range(N)] for name, el in conditions.items()}
    ref = samples["identical input (noise floor)"]

    def cross(a, b):
        return float(np.mean([np.abs(x - y).mean() for x in a for y in b if x is not y]))

    floor = cross(ref, ref)
    print(f"chunk shape: {ref[0].shape}  (horizon x action dim), {N} samples per condition\n")
    print(f"mean |action difference| from the baseline distribution  (noise floor {floor:.4f})")
    for name, s in samples.items():
        v = cross(ref, s)
        tag = "" if name.startswith("identical") else f"   {v / floor:.2f}x the noise floor"
        print(f"  {name:32s} {v:.4f}{tag}")
    print("\na canned trajectory would give 0.0000 on every row")
    env.close()


if __name__ == "__main__":
    main()
