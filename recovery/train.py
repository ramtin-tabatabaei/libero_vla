"""Train the PPO reach policy on one LIBERO task or on all 130 at once.

Because every term of the observation is relative to the grip site and carries
the target's own size, a single policy can cover the whole benchmark -- the
scenes differ in absolute height by a metre, but not in what the arm has to do.
That multi-task run is the default and the one worth having; the single-task
mode is there to debug a scene that the shared policy struggles with.

    # one policy over every task in every suite
    NUMBA_CACHE_DIR=/tmp/libero_numba MUJOCO_GL=egl \
      /home/ramtin/miniconda3/envs/libero/bin/python -m recovery.train

    # one scene, quickly
    ... -m recovery.train --suite libero_object --task-id 0 --ppo.total-timesteps 300000

    # pick up where a killed run left off
    ... -m recovery.train --resume
"""
import dataclasses
import json
import pathlib
import time
from typing import Optional

import numpy as np
import torch
import tyro

from recovery.ppo import PPO, PPOConfig
from recovery.reach_env import ACTION_DIM, OBS_DIM, OBS_SCHEMA, ReachConfig, SUITES, task_list
from recovery.policy import check_schema
from recovery.vec_env import SubprocVecEnv

from build_waypoint_library import write_json

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs"


@dataclasses.dataclass
class Args:
    suite: Optional[str] = None
    """Train on one suite. None trains on all five."""
    task_id: Optional[int] = None
    """Train on one task within the suite. None trains on every task in it."""

    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: Optional[str] = None
    """Directory under recovery/runs. Defaults to the task selection."""
    resume: bool = False
    """Continue from the run directory's last checkpoint."""
    init_from: Optional[str] = None
    """Start from another run's weights and observation statistics, keeping a
    fresh optimiser and a fresh step count. This is how `sweep.py` fine-tunes
    the shared multi-task policy onto a single stubborn scene: carrying the
    optimiser's moments and its annealed learning rate across would start the
    new run at a learning rate near zero."""

    # --- reporting ---
    log_every: int = 1
    """Iterations between progress lines."""
    save_every: int = 5
    """Iterations between checkpoints."""
    snapshot: bool = True
    """Also keep every checkpoint under snapshots/, so the policy can be
    watched improving rather than only inspected once it has finished."""

    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    env: ReachConfig = dataclasses.field(default_factory=ReachConfig)


def run_name_for(args):
    if args.run_name:
        return args.run_name
    if args.suite and args.task_id is not None:
        return "%s_task%03d" % (args.suite, args.task_id)
    return args.suite or "all_tasks"


def summarise(episodes):
    """Success rate, length and final distance over the finished episodes."""
    if not episodes:
        return dict(episodes=0, success_rate=float("nan"), grasp_rate=float("nan"),
                    mean_steps=float("nan"), mean_distance=float("nan"),
                    mean_progress=float("nan"))
    return dict(
        episodes=len(episodes),
        success_rate=float(np.mean([e["success"] for e in episodes])),
        # Grasp rate leads success by a long way and is the first sign the
        # gripper dimension is being used at all; success alone stays at zero
        # through the whole period where that is being learned.
        grasp_rate=float(np.mean([e.get("grasped", False) for e in episodes])),
        mean_progress=float(np.mean([e.get("progress", 0.0) for e in episodes])),
        mean_steps=float(np.mean([e["steps"] for e in episodes])),
        mean_distance=float(np.mean([e["distance"] for e in episodes])))


def main(args: Args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = task_list(suite=args.suite, task_id=args.task_id)
    directory = RUNS / run_name_for(args)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "checkpoint.pt"

    print("tasks      : %d" % len(tasks))
    print("run        : %s" % directory)
    print("device     : %s" % args.device)
    print("batch      : %d envs x %d steps = %d, %d minibatches of %d"
          % (args.ppo.num_envs, args.ppo.num_steps, args.ppo.batch_size,
             args.ppo.num_minibatches, args.ppo.minibatch_size))

    args.env.validate()
    args.ppo.validate()
    state = None
    source = checkpoint_path if args.resume else args.init_from
    if args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path.exists() and not args.resume:
        raise ValueError("Run already has a checkpoint; choose a new --run-name or --resume.")
    if source:
        state = torch.load(source, map_location=args.device)
        check_schema(state)
        if args.resume and state.get("env_config") != dataclasses.asdict(args.env):
            raise ValueError("Resume requires the original environment/reward configuration.")

    envs = SubprocVecEnv(tasks, num_envs=args.ppo.num_envs, config=args.env,
                         seed=args.seed, gamma=args.ppo.gamma,
                         cache_size=2)
    agent = PPO(OBS_DIM, ACTION_DIM, config=args.ppo, device=args.device)

    global_step = 0
    start_iteration = 0
    if args.resume and checkpoint_path.exists():
        agent.load_state_dict(state)
        envs.load_state_dict(state["normalisers"])
        global_step, start_iteration = state["global_step"], state["iteration"]
        print("resumed    : step %d, iteration %d" % (global_step, start_iteration))
    elif args.init_from:
        agent.policy.load_state_dict(state["policy"])
        envs.load_state_dict(state["normalisers"])
        print("warm start : %s" % args.init_from)

    write_json(directory / "config.json",
               dict(tasks=[dict(suite=s, task_id=t) for s, t in tasks],
                    ppo=dataclasses.asdict(args.ppo),
                    env=dataclasses.asdict(args.env),
                    seed=args.seed, obs_dim=OBS_DIM, obs_schema=OBS_SCHEMA, action_dim=ACTION_DIM))

    iterations = args.ppo.total_timesteps // args.ppo.batch_size
    obs = torch.as_tensor(envs.reset(), dtype=torch.float32, device=args.device)
    done = torch.zeros(args.ppo.num_envs, device=args.device)

    history_path = directory / "history.json"
    history = json.loads(history_path.read_text()) if args.resume and history_path.exists() else []
    completed_iterations = start_iteration
    starting_step = global_step
    started = time.time()
    try:
        for iteration in range(start_iteration, iterations):
            episodes = []
            batch, obs, done = agent.collect(envs, obs, done, episodes)
            global_step += args.ppo.batch_size
            stats = agent.update(batch, iteration / max(iterations, 1))
            completed_iterations = iteration + 1
            report = summarise(episodes)
            report.update(stats)
            report["calibration"] = batch["calibration"]
            report.update(iteration=iteration, global_step=global_step,
                          sps=int((global_step - starting_step) / (time.time() - started)))
            history.append(report)
            with (directory / "calibration.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(iteration=iteration, global_step=global_step,
                                             **batch["calibration"]), allow_nan=False) + "\n")

            if iteration % args.log_every == 0:
                print("it %4d  step %8d  sps %4d  eps %3d  ok %5.1f%%  grasp %5.1f%%  "
                      "prog %.2f  dist %.3f  sat %4.1f%%  kl %.4f  ev %6.3f%s"
                      % (iteration, global_step, report["sps"], report["episodes"],
                         100 * report["success_rate"], 100 * report["grasp_rate"],
                         report["mean_progress"], report["mean_distance"],
                         100 * report["saturated"], report["approx_kl"],
                         report["explained_variance"],
                         "  [kl stop]" if report["early_stopped"] else ""),
                      flush=True)

            if iteration % args.log_every == 0:
                raw = batch["calibration"]["overall"]
                if raw["peak_force_n"]["count"]:
                    print("  calibration: force p95/max %.1f/%.1f N  bad-sum p95 %.1f N  "
                          "collision mean %.3f  task mean %.3f  severe %.2f%%"
                          % (raw["peak_force_n"]["p95"], raw["peak_force_n"]["max"],
                             raw["sum_bad_force_n"]["p95"], raw["r_collision_total"]["mean"],
                             raw["r_task"]["mean"], 100 * raw["severe_event"]["mean"]), flush=True)

            if iteration % args.save_every == 0:
                save(agent, envs, checkpoint_path, global_step, completed_iterations, history,
                     directory, snapshot=args.snapshot, env_config=args.env)
    except KeyboardInterrupt:
        print("\ninterrupted -- saving")
    finally:
        save(agent, envs, checkpoint_path, global_step, completed_iterations, history, directory,
             env_config=args.env)
        envs.close()


def save(agent, envs, path, global_step, iteration, history, directory, snapshot=False,
         env_config=None):
    state = agent.state_dict()
    state.update(normalisers=envs.state_dict(), global_step=global_step,
                 iteration=iteration, obs_schema=OBS_SCHEMA,
                 env_config=dataclasses.asdict(env_config or ReachConfig()))
    torch.save(state, path)
    if snapshot:
        snapshots = directory / "snapshots"
        snapshots.mkdir(exist_ok=True)
        torch.save(state, snapshots / ("it%05d.pt" % iteration))
    write_json(directory / "history.json", history)


if __name__ == "__main__":
    main(tyro.cli(Args))
