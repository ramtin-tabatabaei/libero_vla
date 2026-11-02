"""Many LIBERO scenes stepping at once, in subprocesses, with running normalisers.

One renderer-free scene steps at about 100 Hz, which is nowhere near enough to
feed PPO; MuJoCo holds the GIL through `mj_step`, so the only way up is
separate processes. Envs are built inside the child and never in the parent:
that keeps a MuJoCo model out of the forking process entirely, which is what
makes plain `fork` safe here.

Two details that PPO gets wrong if this file gets them wrong:

  * On an episode boundary the child resets immediately and returns the FIRST
    observation of the next episode, so the terminal one is carried out of band
    in `info["final_observation"]`. Bootstrapping a truncated episode off the
    reset observation instead of the final one is a silent, and surprisingly
    large, bias.
  * Reward normalisation divides by the standard deviation of the discounted
    RETURN, not of the reward. Dividing by the reward's own std rescales the
    advantage of a shaped, near-zero-mean reward into noise.
"""
import ctypes
import multiprocessing as mp
import signal

import numpy as np

from recovery.reach_env import OBS_DIM, ACTION_DIM, ReachConfig, ReachEnv


def _die_with_parent():
    """Ask the kernel to SIGKILL this process when its parent goes.

    `daemon=True` only covers a clean parent exit. SIGKILL the trainer -- which
    is what Ctrl-C twice, a job scheduler, or the OOM killer does -- and the
    workers are reparented to init and keep running: sixteen of them, a
    gigabyte and half a core each, were left spinning after one such kill here
    and quietly starved every measurement taken afterwards. PR_SET_PDEATHSIG is
    Linux-only, which is the only platform this repo runs on, and failing to
    set it is not worth aborting a worker over.
    """
    try:
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


# ================================================================ normalisers


class RunningMeanStd:
    """Welford's online moments, in the parallel (batched) form.

    Kept here rather than pulled from a library because it has to be
    checkpointed alongside the weights: a policy restored without the exact
    observation statistics it trained under sees a different input space and
    scores near zero.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        self.update_from_moments(batch.mean(axis=0), batch.var(axis=0), batch.shape[0])

    def update_from_moments(self, mean, var, count):
        delta = mean - self.mean
        total = self.count + count
        new_mean = self.mean + delta * count / total
        m_a = self.var * self.count
        m_b = var * count
        m2 = m_a + m_b + np.square(delta) * self.count * count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def state_dict(self):
        return dict(mean=self.mean.copy(), var=self.var.copy(), count=self.count)

    def load_state_dict(self, state):
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])


# ===================================================================== worker


def _worker(remote, parent_remote, tasks, config, seed, cache_size):
    """Own a slice of the task list and step whichever scene is current.

    Rebuilding a scene costs about a second, so envs are cached and tasks are
    rotated one per episode rather than one per step. With the cache holding
    every task a worker owns, the cost is paid once.
    """
    _die_with_parent()
    parent_remote.close()
    envs = {}
    order = list(tasks)
    index = 0

    def current():
        nonlocal envs
        key = order[index % len(order)]
        if key not in envs:
            if len(envs) >= cache_size:
                envs.pop(next(iter(envs))).close()
            envs[key] = ReachEnv(key[0], key[1], config=config, seed=seed + 1000 * index)
        return envs[key]

    try:
        while True:
            command, data = remote.recv()
            if command == "step":
                env = current()
                obs, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    info["final_observation"] = obs
                    index += 1
                    obs = current().reset()
                remote.send((obs, reward, terminated, truncated, info))
            elif command == "reset":
                remote.send(current().reset())
            elif command == "spec":
                env = current()
                remote.send(dict(suite=env.suite_name, task_id=env.task_id,
                                 language=env.language, target=env.target_name,
                                 tasks=order))
            elif command == "close":
                break
            else:
                raise ValueError("Unknown worker command: " + str(command))
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        for env in envs.values():
            env.close()
        remote.close()


# ================================================================ vector env


class SubprocVecEnv:
    """`num_envs` scenes in `num_envs` processes, with shared normalisers.

    Auto-resetting lives in the child so a slow scene never blocks the others
    on a reset barrier.
    """

    def __init__(self, tasks, num_envs=8, config=None, seed=0, gamma=0.99,
                 cache_size=4, normalise_obs=True, normalise_reward=True,
                 clip_obs=10.0, clip_reward=10.0, start_method="fork"):
        tasks = list(tasks)
        if not tasks:
            raise ValueError("Give the vector env at least one task to train on.")
        if num_envs < 1:
            raise ValueError("Run at least one environment.")
        config = config or ReachConfig()

        # Deal the task list round-robin so consecutive workers hold different
        # scenes; a contiguous split would put all 90 libero_90 tasks on the
        # same few workers and skew every batch toward one scene family.
        shares = [tasks[i::num_envs] for i in range(num_envs)]
        shares = [s if s else [tasks[i % len(tasks)]] for i, s in enumerate(shares)]

        context = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[context.Pipe() for _ in range(num_envs)])
        self.processes = []
        for index, (work_remote, remote, share) in enumerate(
                zip(self.work_remotes, self.remotes, shares)):
            process = context.Process(
                target=_worker,
                args=(work_remote, remote, share, config, seed + 7919 * index, cache_size),
                daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.num_envs = num_envs
        self.gamma = gamma
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.obs_rms = RunningMeanStd((OBS_DIM,)) if normalise_obs else None
        self.return_rms = RunningMeanStd(()) if normalise_reward else None
        self.returns = np.zeros(num_envs)
        self.training = True

    # --- normalisation ----------------------------------------------------

    def normalise_obs(self, obs):
        if self.obs_rms is None:
            return obs.astype(np.float32)
        if self.training:
            self.obs_rms.update(obs)
        scaled = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(scaled, -self.clip_obs, self.clip_obs).astype(np.float32)

    def _normalise_reward(self, reward, done):
        if self.return_rms is None:
            return reward
        self.returns = self.returns * self.gamma + reward
        if self.training:
            self.return_rms.update(self.returns)
        scaled = reward / np.sqrt(self.return_rms.var + 1e-8)
        self.returns[done] = 0.0
        return np.clip(scaled, -self.clip_reward, self.clip_reward)

    # --- the vector API ---------------------------------------------------

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = np.stack([remote.recv() for remote in self.remotes])
        self.returns[:] = 0.0
        return self.normalise_obs(obs)

    def step(self, actions):
        actions = np.asarray(actions, dtype=float).reshape(self.num_envs, ACTION_DIM)
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, reward, terminated, truncated, infos = zip(*results)

        obs = np.stack(obs)
        reward = np.asarray(reward, dtype=np.float64)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)

        infos = list(infos)
        for info in infos:
            if "final_observation" in info:
                info["final_observation"] = self.normalise_obs(
                    np.asarray(info["final_observation"])[None])[0]

        return (self.normalise_obs(obs),
                self._normalise_reward(reward, terminated | truncated),
                terminated, truncated, infos)

    def spec(self):
        self.remotes[0].send(("spec", None))
        return self.remotes[0].recv()

    def state_dict(self):
        return dict(obs_rms=self.obs_rms.state_dict() if self.obs_rms else None,
                    return_rms=self.return_rms.state_dict() if self.return_rms else None)

    def load_state_dict(self, state):
        if state.get("obs_rms") and self.obs_rms:
            self.obs_rms.load_state_dict(state["obs_rms"])
        if state.get("return_rms") and self.return_rms:
            self.return_rms.load_state_dict(state["return_rms"])

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
