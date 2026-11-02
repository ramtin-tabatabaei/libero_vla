"""See what the reach policy is doing: a learning curve, and video of the arm.

Two views of the same run, because neither one alone is honest. The curve says
whether the numbers are going up; the video says whether they are going up for
the reason you think. A policy that learned to hover a centimetre outside the
success radius and one that learned to reach both show a falling mean distance,
and only the video tells them apart.

`--stages` renders the same scene from several training snapshots side by side,
oldest first, so the improvement is visible in one file rather than across
several.

    # curve for a run in progress, redrawn each time it is called
    /home/ramtin/miniconda3/envs/libero/bin/python -m recovery.watch --run watch_object0 --curve

    # video of the latest policy
    NUMBA_CACHE_DIR=/tmp/libero_numba MUJOCO_GL=egl \
      /home/ramtin/miniconda3/envs/libero/bin/python -m recovery.watch --run watch_object0 --episodes 3

    # early / middle / late, as one strip
    ... -m recovery.watch --run watch_object0 --stages 4
"""
import dataclasses
import json
import pathlib
import time
from typing import Optional

import numpy as np
import tyro

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs"


@dataclasses.dataclass
class Args:
    run: str = "watch_object0"
    """Run directory under recovery/runs."""
    suite: Optional[str] = None
    task_id: Optional[int] = None
    """Scene to film. Defaults to the first task the run was trained on."""

    live: bool = False
    """Open a MuJoCo window and keep replaying the newest checkpoint while the
    trainer runs, so the policy is watched improving in the simulator itself.
    Needs MUJOCO_GL=glx -- EGL cannot open a window."""
    curve: bool = False
    """Print the learning curve as text and write a PNG beside the run."""
    episodes: int = 2
    """Episodes to film per policy."""
    stages: int = 0
    """Film this many evenly spaced snapshots side by side instead of one policy."""
    checkpoint: Optional[str] = None
    """Film a specific checkpoint rather than the run's latest."""
    seed: int = 3
    fps: int = 20
    """The control rate, so the video plays at wall-clock speed."""
    stochastic: bool = False
    out: Optional[str] = None


# ============================================================ learning curve


def curve(directory):
    """Success rate and reach distance against environment steps."""
    history = json.loads((directory / "history.json").read_text())
    rows = [h for h in history if h["episodes"]]
    if not rows:
        print("No finished episodes logged yet -- give it a few more iterations.")
        return

    steps = np.array([r["global_step"] for r in rows])
    success = np.array([r["success_rate"] for r in rows])
    distance = np.array([r["mean_distance"] for r in rows])

    # A run's own episodes arrive in batches of a few dozen, so a raw per-
    # iteration success rate rattles between 0 and 1 and hides the trend.
    window = max(1, len(rows) // 20)
    smooth = np.convolve(success, np.ones(window) / window, mode="valid")

    print("%-10s %8s %9s %9s" % ("step", "success", "distance", "steps"))
    for row in rows[:: max(1, len(rows) // 25)]:
        bar = "#" * int(round(24 * row["success_rate"]))
        print("%-10d %7.0f%% %9.3f %9.1f  %s"
              % (row["global_step"], 100 * row["success_rate"],
                 row["mean_distance"], row["mean_steps"], bar))
    print("\nlatest: %.0f%% success, %.3f m mean final distance, %d env steps"
          % (100 * success[-1], distance[-1], steps[-1]))
    print("best  : %.0f%% success (smoothed %.0f%%)"
          % (100 * success.max(), 100 * smooth.max() if len(smooth) else 0))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- text curve only)")
        return

    figure, left = plt.subplots(figsize=(9, 4.5))
    left.plot(steps, 100 * success, color="#c8d6e5", linewidth=1)
    if len(smooth):
        left.plot(steps[window - 1:], 100 * smooth, color="#1e6091", linewidth=2,
                  label="success rate (smoothed)")
    left.set_xlabel("environment steps")
    left.set_ylabel("success rate (%)", color="#1e6091")
    left.set_ylim(-2, 102)
    right = left.twinx()
    right.plot(steps, distance, color="#c1121f", linewidth=1.5, label="final distance")
    right.set_ylabel("mean final distance (m)", color="#c1121f")
    right.axhline(0.03, color="#c1121f", linestyle=":", linewidth=1)
    left.set_title("%s -- reach policy" % directory.name)
    figure.tight_layout()
    destination = directory / "learning_curve.png"
    figure.savefig(destination, dpi=130)
    print("\nwrote %s" % destination)


# ================================================================ live window


def live(directory, suite, task_id, args):
    """Replay the newest checkpoint in an on-screen window, forever.

    Reads the checkpoint off disk before every episode rather than holding one
    in memory, so a trainer writing checkpoints in another process shows up
    here within an episode. That is the whole point: the window is watching
    training, not a finished artefact.

    Deliberately a separate process from the trainer. Rendering inside a rollout
    worker would tie the window's frame rate to the optimiser and put a GL
    context inside a forked process, and the trainer would slow down to draw.
    """
    import torch

    from recovery.evaluate import load_policy
    from recovery.reach_env import ReachEnv

    checkpoint = pathlib.Path(args.checkpoint) if args.checkpoint else directory / "checkpoint.pt"
    env = ReachEnv(suite, task_id, seed=args.seed, viewer=True)
    print("scene   : %s" % env.language)
    print("target  : %s" % env.target_name)
    print("watching: %s" % checkpoint)
    print("\nClose the window or press Ctrl-C to stop.\n")

    policy = obs_rms = None
    stamped = None
    episode = 0
    try:
        while True:
            # Pick up a newer checkpoint between episodes, never mid-reach --
            # swapping the policy under a moving arm makes the window lie about
            # what any one policy can do.
            if checkpoint.exists():
                current = checkpoint.stat().st_mtime
                if current != stamped:
                    for _ in range(5):
                        try:
                            policy, obs_rms = load_policy(checkpoint, "cpu")
                            state = torch.load(checkpoint, map_location="cpu")
                            stamped = current
                            break
                        except Exception:
                            # The trainer may be mid-write; torch.save is not
                            # atomic. Wait a beat and take the next one.
                            time.sleep(0.4)
            if policy is None:
                print("waiting for a checkpoint at %s ..." % checkpoint)
                time.sleep(3.0)
                continue

            obs = env.reset()
            step = 0
            while True:
                vector = obs
                if obs_rms is not None:
                    vector = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                with torch.no_grad():
                    action, _, _, _, _ = policy.act(
                        torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0),
                        deterministic=not args.stochastic)
                obs, _, terminated, truncated, info = env.step(action.squeeze(0).numpy())
                env.render()
                step += 1
                if args.fps:
                    time.sleep(1.0 / args.fps)
                if terminated or truncated:
                    break
            episode += 1
            print("episode %3d  trained to step %8d  %3d frames  %5.0f mm  %s"
                  % (episode, state.get("global_step", 0), step,
                     1000 * info["distance"],
                     "REACHED" if info["success"] else "timed out"), flush=True)
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        env.close()


# ==================================================================== video


def roll(env, policy, obs_rms, episodes, seed, stochastic, label):
    """Film `episodes` reaches and return the frames, with a caption burnt in."""
    import torch

    frames = []
    for episode in range(episodes):
        obs = env.reset()
        while True:
            vector = obs
            if obs_rms is not None:
                vector = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
            with torch.no_grad():
                action, _, _, _, _ = policy.act(
                    torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0),
                    deterministic=not stochastic)
            frames.append(annotate(env.frame(), label, env.previous_distance))
            obs, _, terminated, truncated, info = env.step(action.squeeze(0).numpy())
            if terminated or truncated:
                # Hold the last frame so a success is readable rather than a blink.
                closing = annotate(env.frame(), label, info["distance"],
                                   "REACHED" if info["success"] else "timed out")
                frames.extend([closing] * 10)
                break
    return frames


def annotate(frame, label, distance, verdict=None):
    """Burn the snapshot label and live distance into the top of the frame.

    Drawn as blocky pixels rather than with a font dependency -- the repo has
    no PIL in this env and a video that needs an install is a video nobody
    watches.
    """
    frame = np.ascontiguousarray(frame)
    text = "%s  %.0fmm" % (label, 1000 * distance)
    if verdict:
        text += "  " + verdict
    stamp(frame, text, 4, 4)
    return frame


FONT = {
    "0": ["111", "101", "101", "101", "111"], "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"], "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"], "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"], "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"], "9": ["111", "101", "111", "001", "111"],
    "m": ["000", "110", "111", "101", "101"], "R": ["111", "101", "111", "110", "101"],
    "E": ["111", "100", "111", "100", "111"], "A": ["111", "101", "111", "101", "101"],
    "C": ["111", "100", "100", "100", "111"], "H": ["101", "101", "111", "101", "101"],
    "D": ["110", "101", "101", "101", "110"], "t": ["010", "111", "010", "010", "011"],
    "i": ["010", "000", "010", "010", "010"], "e": ["111", "101", "111", "100", "111"],
    "d": ["001", "001", "111", "101", "111"], "o": ["000", "111", "101", "101", "111"],
    "u": ["000", "101", "101", "101", "111"], "n": ["000", "110", "101", "101", "101"],
    "s": ["000", "111", "100", "001", "111"], "p": ["000", "111", "101", "111", "100"],
    "a": ["000", "111", "101", "101", "111"], "g": ["111", "101", "111", "001", "111"],
    "l": ["110", "010", "010", "010", "111"], "r": ["000", "111", "100", "100", "100"],
    "k": ["100", "101", "110", "101", "101"], "c": ["000", "111", "100", "100", "111"],
    "y": ["101", "101", "111", "001", "111"], "b": ["100", "100", "111", "101", "111"],
    "j": ["001", "000", "001", "101", "111"], "v": ["000", "101", "101", "101", "010"],
    "w": ["000", "101", "101", "111", "111"], "x": ["000", "101", "010", "010", "101"],
    "f": ["011", "100", "111", "100", "100"], "h": ["100", "100", "111", "101", "101"],
    "q": ["111", "101", "111", "001", "001"], "z": ["111", "001", "010", "100", "111"],
    ".": ["000", "000", "000", "000", "010"], " ": ["000", "000", "000", "000", "000"],
    "-": ["000", "000", "111", "000", "000"], "_": ["000", "000", "000", "000", "111"],
    "/": ["001", "001", "010", "100", "100"], ":": ["000", "010", "000", "010", "000"],
}


def stamp(frame, text, x, y, scale=2):
    """Blit `text` into `frame` as 3x5 blocks, on a dark strip for contrast."""
    height, width = 5 * scale + 2 * scale, len(text) * 4 * scale + 2 * scale
    height = min(height, frame.shape[0] - y)
    width = min(width, frame.shape[1] - x)
    if height <= 0 or width <= 0:
        return
    patch = frame[y:y + height, x:x + width]
    patch[:] = (0.25 * patch).astype(frame.dtype)
    for index, character in enumerate(text):
        glyph = FONT.get(character, FONT.get(character.lower()))
        if glyph is None:
            continue
        for row, bits in enumerate(glyph):
            for column, bit in enumerate(bits):
                if bit != "1":
                    continue
                top = y + scale + row * scale
                left = x + scale + (index * 4 + column) * scale
                if top + scale <= frame.shape[0] and left + scale <= frame.shape[1]:
                    frame[top:top + scale, left:left + scale] = 255


def main(args: Args) -> None:
    import imageio
    import torch

    from recovery.evaluate import load_policy
    from recovery.reach_env import ReachEnv

    directory = RUNS / args.run
    if not directory.exists():
        raise ValueError("No run at %s. Train one first, or pass --run." % directory)

    if args.curve:
        curve(directory)
        return

    config = json.loads((directory / "config.json").read_text())
    suite = args.suite or config["tasks"][0]["suite"]
    task_id = args.task_id if args.task_id is not None else config["tasks"][0]["task_id"]

    if args.live:
        live(directory, suite, task_id, args)
        return

    if args.stages:
        snapshots = sorted((directory / "snapshots").glob("it*.pt"))
        if not snapshots:
            raise ValueError("No snapshots under %s -- train with --snapshot." % directory)
        picks = [snapshots[i] for i in
                 np.linspace(0, len(snapshots) - 1, min(args.stages, len(snapshots))).astype(int)]
    else:
        picks = [pathlib.Path(args.checkpoint) if args.checkpoint
                 else directory / "checkpoint.pt"]

    env = ReachEnv(suite, task_id, seed=args.seed, cameras=True)
    print("scene   : %s" % env.language)
    print("target  : %s" % env.target_name)

    frames = []
    for path in picks:
        state = torch.load(path, map_location="cpu")
        policy, obs_rms = load_policy(path, "cpu")
        label = "step %d" % state.get("global_step", 0)
        env.rng = np.random.default_rng(args.seed)  # same starts for every stage
        clip = roll(env, policy, obs_rms, args.episodes, args.seed, args.stochastic, label)
        print("  %-28s %4d frames" % (path.name, len(clip)))
        frames.extend(clip)
    env.close()

    destination = pathlib.Path(args.out) if args.out else directory / (
        "stages.mp4" if args.stages else "rollout.mp4")
    imageio.mimwrite(destination, frames, fps=args.fps, quality=8)
    print("\nwrote %s  (%d frames, %.1fs)" % (destination, len(frames), len(frames) / args.fps))


if __name__ == "__main__":
    main(tyro.cli(Args))
