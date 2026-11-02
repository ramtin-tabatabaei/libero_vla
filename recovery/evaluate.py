"""Score a trained reach policy, per task and overall.

Runs with the policy's mean action rather than a sample -- exploration noise is
a training device, and leaving it on costs a few points of success rate that
have nothing to do with what was learned -- and with the observation normaliser
frozen, so a long evaluation cannot drift the statistics the policy was trained
under.

    NUMBA_CACHE_DIR=/tmp/libero_numba MUJOCO_GL=egl \
      /home/ramtin/miniconda3/envs/libero/bin/python -m recovery.evaluate \
        --checkpoint recovery/runs/all_tasks/checkpoint.pt --episodes 10

A `--baseline` run replaces the network with the proportional controller the
repo already has in `RobotServer._goto_xyz_osc`: clip the position error by the
OSC scale and feed it straight in. That is the number PPO has to beat, and on
the scenes where a straight line is enough it is a hard number to beat.
"""
import collections
import dataclasses
import pathlib
from typing import Optional

import numpy as np
import torch
import tyro

from recovery.ppo import ActorCritic, PPOConfig
from recovery.policy import check_schema, config_from_dict
from recovery.reach_env import (ACTION_DIM, OBS_DIM, ReachConfig, ScriptedPick,
                                SUITES, task_list)
from recovery.vec_env import RunningMeanStd, SubprocVecEnv

from build_waypoint_library import write_json

ROOT = pathlib.Path(__file__).resolve().parent
OSC_POS_SCALE = 0.05
"""OSC_POSE maps action [-1,1] -> +-0.05 m; robot_server.py uses the same name."""


@dataclasses.dataclass
class Args:
    checkpoint: Optional[str] = None
    """Path to checkpoint.pt. Omit it only with --baseline."""
    suite: Optional[str] = None
    task_id: Optional[int] = None
    episodes: int = 10
    """Episodes per task."""
    num_envs: int = 16
    seed: int = 11
    device: str = "cpu"
    baseline: bool = False
    """Score the scripted top-down pick instead of a network. This is the
    number PPO has to beat, and on the scenes where a canonical top-down grasp
    is enough it is a hard number to beat."""
    servo: bool = False
    """Score the repo's straight-line reach instead. It cannot grasp; it is
    here to show how far short of the task bare reaching falls."""
    audit: bool = False
    """Report how often the policy's actions saturate the controller's range,
    rather than scoring success. Only meaningful with --checkpoint."""
    out: Optional[str] = None
    """Where to write the per-task JSON report. Defaults to beside the checkpoint."""
    stochastic: bool = False
    """Sample actions instead of taking the distribution's mean."""

    use_checkpoint_env: bool = True
    """Use saved reward/reset settings. Disable to evaluate explicit --env overrides."""

    env: ReachConfig = dataclasses.field(default_factory=ReachConfig)


def load_policy(path, device):
    """Weights plus the observation statistics they were trained with."""
    state = torch.load(path, map_location=device)
    check_schema(state)
    saved = state.get("config") or {}
    policy = ActorCritic(state.get("obs_dim", OBS_DIM), state.get("action_dim", ACTION_DIM),
                         tuple(saved.get("hidden_sizes", PPOConfig.hidden_sizes)),
                         saved.get("log_std_init", PPOConfig.log_std_init),
                         # Must follow the checkpoint, not the current default:
                         # loading a squashed policy as unsquashed silently
                         # changes what every weight means.
                         squash=saved.get("squash", True)).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    policy.env_config = config_from_dict(state.get("env_config", {}))
    normalisers = state.get("normalisers") or {}
    obs_rms = None
    if normalisers.get("obs_rms"):
        obs_rms = RunningMeanStd((state.get("obs_dim", OBS_DIM),))
        obs_rms.load_state_dict(normalisers["obs_rms"])
    return policy, obs_rms


def servo_action(obs):
    """Straight-line servoing on the world-frame error, the repo's existing reach.

    The first three numbers of the observation are the world-frame vector from
    the grip site to the grasp point, which is exactly the error term
    `_goto_xyz_osc` divides by OSC_POS_SCALE. Kept for reference: it solves the
    old reach-only task almost perfectly and cannot pick anything up, because
    it drives diagonally into the object and shoves it.
    """
    action = np.zeros((obs.shape[0], ACTION_DIM))
    action[:, :3] = np.clip(obs[:, :3] / OSC_POS_SCALE, -1.0, 1.0)
    action[:, -1] = -1.0
    return action


def main(args: Args) -> None:
    scripted_mode = args.baseline or args.servo
    if not scripted_mode and not args.checkpoint:
        raise ValueError("Give a --checkpoint to evaluate, or pass --baseline or --servo.")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tasks = task_list(suite=args.suite, task_id=args.task_id)
    policy = obs_rms = None
    if not scripted_mode:
        policy, obs_rms = load_policy(args.checkpoint, args.device)
        if args.use_checkpoint_env:
            args.env = policy.env_config

    # The baseline reads a raw world-frame error, so it must not be handed
    # normalised observations; the network must. Same env, different switch.
    envs = SubprocVecEnv(tasks, num_envs=args.num_envs, config=args.env, seed=args.seed,
                         cache_size=2,
                         normalise_obs=False, normalise_reward=False)
    envs.training = False

    counts = collections.Counter()
    records = collections.defaultdict(list)
    scripted = ScriptedPick(args.num_envs)
    saturation = []
    obs = envs.reset()
    target_total = args.episodes * len(tasks)

    while sum(min(counts[t], args.episodes) for t in tasks) < target_total:
        if args.baseline:
            action = scripted.act(obs)
        elif args.servo:
            action = servo_action(obs)
        else:
            vector = obs
            if obs_rms is not None:
                vector = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10.0, 10.0)
            with torch.no_grad():
                sampled, _, _, _, _ = policy.act(
                    torch.as_tensor(vector, dtype=torch.float32, device=args.device),
                    deterministic=not args.stochastic)
            action = sampled.cpu().numpy()
            saturation.append(np.abs(action) > 0.99)

        obs, _, terminated, truncated, infos = envs.step(action)
        scripted.reset(terminated | truncated)
        for i, info in enumerate(infos):
            if terminated[i] or truncated[i]:
                key = (info["suite"], info["task_id"])
                if counts[key] < args.episodes:
                    records[key].append(dict(success=bool(info["success"]),
                                             steps=int(info["steps"]),
                                             distance=float(info["distance"]),
                                             mode=info["mode"],
                                             collision_total=info["collision_total"],
                                             severe_steps=info["severe_steps"]))
                counts[key] += 1
        done_tasks = sum(1 for t in tasks if counts[t] >= args.episodes)
        print("\r%d/%d tasks complete" % (done_tasks, len(tasks)), end="", flush=True)

    envs.close()
    print()

    rows = []
    for suite_name, task_id in tasks:
        episodes = records[(suite_name, task_id)]
        if not episodes:
            continue
        successes = [e["success"] for e in episodes]
        rows.append(dict(
            suite=suite_name, task_id=task_id, episodes=len(episodes),
            success_rate=float(np.mean(successes)),
            mode=episodes[0]["mode"],
            mean_collision_force_sum=float(np.mean([e["collision_total"] for e in episodes])),
            mean_severe_steps=float(np.mean([e["severe_steps"] for e in episodes])),
            mean_steps=float(np.mean([e["steps"] for e in episodes])),
            mean_distance=float(np.mean([e["distance"] for e in episodes])),
            # Steps over the episodes that actually arrived; averaging in the
            # timeouts would make a policy that never finishes look fast.
            mean_steps_on_success=float(np.mean(
                [e["steps"] for e in episodes if e["success"]])) if any(successes) else float("nan")))

    overall = dict(
        tasks=len(rows),
        episodes=int(sum(r["episodes"] for r in rows)),
        success_rate=float(np.mean([r["success_rate"] for r in rows])),
        mean_distance=float(np.mean([r["mean_distance"] for r in rows])),
        solved=int(sum(1 for r in rows if r["success_rate"] >= 0.9)))

    print("\n%-16s %5s %8s %8s %8s" % ("suite", "task", "success", "steps", "dist"))
    for row in sorted(rows, key=lambda r: r["success_rate"]):
        print("%-16s %5d %7.0f%% %8.1f %8.3f"
              % (row["suite"], row["task_id"], 100 * row["success_rate"],
                 row["mean_steps_on_success"], row["mean_distance"]))
    print("\n%d tasks, %d episodes: mean success %.1f%%, %d tasks at or above 90%%"
          % (overall["tasks"], overall["episodes"], 100 * overall["success_rate"],
             overall["solved"]))

    if saturation:
        saturated = np.concatenate(saturation)
        overall["saturated_dims"] = float(saturated.mean())
        overall["saturated_steps"] = float(saturated.any(axis=-1).mean())
        print("action saturation: %.1f%% of dimensions, %.1f%% of steps"
              % (100 * overall["saturated_dims"], 100 * overall["saturated_steps"]))

    by_suite = collections.defaultdict(list)
    for row in rows:
        by_suite[row["suite"]].append(row["success_rate"])
    for suite_name in SUITES:
        if by_suite[suite_name]:
            print("  %-16s %5.1f%%  (%d tasks)"
                  % (suite_name, 100 * np.mean(by_suite[suite_name]), len(by_suite[suite_name])))

    destination = pathlib.Path(args.out) if args.out else (
        pathlib.Path(args.checkpoint).parent / "evaluation.json" if args.checkpoint
        else ROOT / "runs" / "baseline_evaluation.json")
    write_json(destination, dict(overall=overall, tasks=rows,
                                 baseline=args.baseline, episodes=args.episodes))
    print("\nwrote %s" % destination)


if __name__ == "__main__":
    main(tyro.cli(Args))
