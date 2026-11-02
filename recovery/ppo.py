"""PPO for continuous control, written to the settings the literature agrees on.

Nothing here is invented. The defaults are the ones that survive ablation in
Andrychowicz et al., "What Matters in On-Policy Reinforcement Learning?"
(2021), the code-level details are the ones enumerated in Huang et al., "The 37
Implementation Details of Proximal Policy Optimization" (2022) and implemented
in CleanRL's `ppo_continuous_action.py`, and where those two disagree with
stable-baselines3's tuned MuJoCo configs the disagreement is noted on the field
itself. stable-baselines3 is not installed in the libero env and pulling it in
would drag a gymnasium/numpy upgrade through a pinned robosuite, so the
algorithm is written out rather than imported -- it is about two hundred lines.

The four details that most often go missing, all present below:

  * Adam's epsilon is 1e-5, not torch's 1e-8 default (37 details, #3).
  * Orthogonal initialisation with gain sqrt(2) on hidden layers, 0.01 on the
    policy head so the initial policy is near-uniform, and 1.0 on the value
    head (#2).
  * The value target bootstraps through a time-limit truncation and does NOT
    bootstrap through a real arrival. Conflating them biases every episode that
    ends on the horizon (Pardo et al., 2018).
  * Advantages are normalised per minibatch, not per batch (#7).
"""
import dataclasses
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from recovery.calibration import summarise_calibration


# ===================================================================== config


@dataclasses.dataclass
class PPOConfig:
    total_timesteps: int = 3_000_000
    """Environment steps. A dense-reward Panda reach converges well inside
    this; the budget is sized for the multi-task run over all 130 scenes."""

    num_envs: int = 8
    """One per worker process. MuJoCo is GIL-bound, so parallelism has to come
    from processes -- but this box has 8 physical cores behind its 16 threads
    and a worker costs about 1.1 GB, so 16 of them both oversubscribe the cores
    and exhaust a 31 GB box. Eight is the measured sweet spot here; raise it on
    a bigger machine."""
    num_steps: int = 512
    """Rollout length per env, so a batch is num_envs * num_steps = 4096.
    Andrychowicz et al. find batches of a few thousand best for continuous
    control; CleanRL's 2048-with-one-env is the same batch by another route."""
    num_minibatches: int = 32
    """4096 / 32 = 128 per minibatch, inside the 64-256 band that ablates best."""
    update_epochs: int = 10
    """CleanRL and stable-baselines3 both use 10 for continuous control."""

    learning_rate: float = 3e-4
    """The MuJoCo default everywhere: CleanRL, sb3 rl-zoo, the PPO paper's own
    continuous experiments."""
    anneal_lr: bool = True
    """Linear decay to zero over training (37 details, #4). Worth roughly a few
    per cent of final return and costs nothing."""
    adam_epsilon: float = 1e-5

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    """Clipped value loss, as in the original implementation. The 37-details
    study finds it neither helps nor hurts much; kept because it is what the
    reference implementations do and it bounds the value update."""
    ent_coef: float = 0.0
    """Zero for continuous control -- exploration comes from the Gaussian's own
    standard deviation, and a positive entropy bonus on an unbounded Gaussian
    just inflates it. Both CleanRL and the rl-zoo MuJoCo configs use 0.0."""
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    """Off, which is what CleanRL, stable-baselines3 and the rl-zoo MuJoCo
    configs all default to -- the clipped objective is already the trust
    region, and a second one on top mostly removes epochs.

    Measured rather than assumed: at 0.02 the early stop fired on every single
    iteration of a real run, with the approximate KL sitting at 0.021-0.048
    after the first epoch, so `update_epochs=10` was silently doing one epoch.
    Set it to 0.05 or so if a run ever shows a collapse; leave it off
    otherwise."""

    hidden_sizes: tuple = (256, 256)
    """Wider than CleanRL's 64x64, which is tuned for Gym's low-dimensional
    locomotion. Andrychowicz et al. find wider networks better as the task gets
    harder, and 256x256 is the robosuite/manipulation norm."""
    log_std_init: float = -0.5
    """std ~= 0.61 at the start. Andrychowicz et al. single out the initial
    action standard deviation as one of the highest-impact choices and land
    near 0.5; torch's and CleanRL's 0.0 (std 1.0) saturates a [-1, 1] action
    space from the first step."""
    squash: bool = True
    """Push actions through tanh and correct the log-probability by the
    Jacobian, instead of sampling an unbounded Gaussian and clipping.

    Clipping is the usual shortcut and it is a real bias: PPO scores the
    density of the sample it drew while the simulator executes the clipped
    value, so every saturated dimension has a ratio computed for an action
    that never happened. Measured on the clipping version of this policy, over
    a converged rollout: 26.1% of all action dimensions were clipped, 89.7% of
    steps had at least one, and on the two translation axes the DISTRIBUTION
    MEAN itself sat outside [-1, 1] about 60% of the time -- the policy had
    learned to run the mean off to -1.08 because overshooting the box is free.
    tanh makes that impossible, since the mean maps into the interior by
    construction. `evaluate.py --audit` reports the same statistics for any
    checkpoint."""

    def validate(self):
        batch = self.num_envs * self.num_steps
        if batch % self.num_minibatches:
            raise ValueError(
                "num_envs * num_steps (%d) must divide evenly into %d minibatches."
                % (batch, self.num_minibatches))
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("Discount and GAE lambda must lie in (0, 1].")
        if self.clip_coef <= 0:
            raise ValueError("Clip coefficient must be greater than zero.")

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches


# ==================================================================== network


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    """Orthogonal weights, constant bias -- 37 details, #2."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Separate trunks for policy and value, tanh throughout.

    Sharing a trunk makes the value loss and the policy loss fight over the
    same features, which is why every tuned continuous-control PPO keeps them
    apart. The log standard deviation is a bare parameter, not a network head:
    a state-dependent std is free to collapse early and stop exploring.
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256), log_std_init=-0.5,
                 squash=True):
        super().__init__()
        self.actor = self._trunk(obs_dim, hidden_sizes, action_dim, head_std=0.01)
        self.critic = self._trunk(obs_dim, hidden_sizes, 1, head_std=1.0)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.squash = squash

    @staticmethod
    def _trunk(obs_dim, hidden_sizes, out_dim, head_std):
        layers, last = [], obs_dim
        for size in hidden_sizes:
            layers += [layer_init(nn.Linear(last, size)), nn.Tanh()]
            last = size
        layers.append(layer_init(nn.Linear(last, out_dim), std=head_std))
        return nn.Sequential(*layers)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.expand_as(mean).exp())

    def act(self, obs, action=None, deterministic=False):
        """Sample (or re-score) an action, and return what the env should execute.

        Returns `(executed, raw, log_prob, entropy, value)`. `raw` is the
        pre-squash Gaussian sample and is what the rollout buffer must store:
        re-scoring an action later needs the point the density was evaluated
        at, and recovering it from the squashed value would mean an arctanh,
        which blows up exactly at the saturated actions this is meant to fix.

        With `squash`, the log-probability carries the tanh Jacobian,
        log|d tanh/du| = log(1 - tanh(u)^2), summed over dimensions. Without
        it, this degrades to the older clip-afterwards behaviour.
        """
        distribution = self.distribution(obs)
        if action is None:
            action = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action).sum(-1)
        entropy = distribution.entropy().sum(-1)

        if not self.squash:
            return action, action, log_prob, entropy, self.value(obs)

        executed = torch.tanh(action)
        # log(1 - tanh(u)^2) written as 2*(log 2 - u - softplus(-2u)): the
        # direct form underflows to log(0) once |u| passes about 9, which a
        # saturating policy reaches routinely.
        correction = 2.0 * (np.log(2.0) - action - nn.functional.softplus(-2.0 * action))
        return executed, action, log_prob - correction.sum(-1), entropy, self.value(obs)


# ================================================================== algorithm


class PPO:
    """Rollout, GAE, clipped-surrogate update. One instance per training run."""

    def __init__(self, obs_dim, action_dim, config=None, device="cpu"):
        self.config = config or PPOConfig()
        self.config.validate()
        self.device = torch.device(device)
        self.policy = ActorCritic(obs_dim, action_dim, self.config.hidden_sizes,
                                  self.config.log_std_init,
                                  squash=self.config.squash).to(self.device)
        self.optimiser = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.config.learning_rate,
                                          eps=self.config.adam_epsilon)
        self.obs_dim, self.action_dim = obs_dim, action_dim

    # --- advantage --------------------------------------------------------

    def advantages(self, rewards, values, dones, next_value, next_done):
        """Generalised advantage estimation, backwards over the rollout.

        `dones` here means "the episode ended", truncation included; the
        truncated step's reward has already had gamma * V(final obs) folded
        into it by `collect`, which is what keeps the horizon from reading as a
        real terminal state.
        """
        config = self.config
        advantages = torch.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                not_done, following = 1.0 - next_done, next_value
            else:
                not_done, following = 1.0 - dones[t + 1], values[t + 1]
            delta = rewards[t] + config.gamma * following * not_done - values[t]
            running = delta + config.gamma * config.gae_lambda * not_done * running
            advantages[t] = running
        return advantages, advantages + values

    # --- rollout ----------------------------------------------------------

    def collect(self, envs, obs, done, episodes):
        """One batch of experience, plus whatever episodes finished inside it."""
        config = self.config
        num_envs = envs.num_envs
        observations = torch.zeros((config.num_steps, num_envs, self.obs_dim), device=self.device)
        actions = torch.zeros((config.num_steps, num_envs, self.action_dim), device=self.device)
        log_probs = torch.zeros((config.num_steps, num_envs), device=self.device)
        rewards = torch.zeros((config.num_steps, num_envs), device=self.device)
        dones = torch.zeros((config.num_steps, num_envs), device=self.device)
        values = torch.zeros((config.num_steps, num_envs), device=self.device)

        saturated = 0.0
        calibration = []
        for step in range(config.num_steps):
            observations[step] = obs
            dones[step] = done
            with torch.no_grad():
                executed, raw, log_prob, _, value = self.policy.act(obs)
            # The buffer stores the PRE-squash sample, because that is the point
            # the density was evaluated at; the env gets the squashed one.
            actions[step], log_probs[step], values[step] = raw, log_prob, value
            saturated += (executed.abs() > 0.99).float().mean().item()

            next_obs, reward, terminated, truncated, infos = envs.step(
                executed.cpu().numpy())
            reward = np.asarray(reward, dtype=np.float32)

            # Time-limit bootstrapping. A truncated episode was still going, so
            # its value has to carry past the cut; a terminated one really
            # stopped and must not.
            cut = np.flatnonzero(truncated & ~terminated)
            if cut.size:
                finals = torch.as_tensor(
                    np.stack([infos[i]["final_observation"] for i in cut]),
                    dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    reward[cut] += config.gamma * self.policy.value(finals).cpu().numpy()

            for i, info in enumerate(infos):
                if "calibration" in info:
                    calibration.append((info["mode"], info["calibration"]))
                if terminated[i] or truncated[i]:
                    episodes.append(dict(success=bool(info["success"]),
                                         steps=int(info["steps"]),
                                         distance=float(info["distance"]),
                                         grasped=bool(info.get("ever_engaged", False)),
                                         progress=float(info.get("best_progress", 0.0)),
                                         suite=info["suite"], task_id=info["task_id"]))

            rewards[step] = torch.as_tensor(reward, device=self.device)
            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            done = torch.as_tensor((terminated | truncated).astype(np.float32),
                                   device=self.device)

        with torch.no_grad():
            next_value = self.policy.value(obs)
        advantages, returns = self.advantages(rewards, values, dones, next_value, done)
        batch = dict(
            obs=observations.reshape(-1, self.obs_dim),
            actions=actions.reshape(-1, self.action_dim),
            log_probs=log_probs.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
            values=values.reshape(-1),
            saturated=saturated / max(config.num_steps, 1),
            calibration=summarise_calibration(calibration))
        return batch, obs, done

    # --- update -----------------------------------------------------------

    def update(self, batch, progress):
        """The clipped surrogate, `update_epochs` passes over shuffled minibatches."""
        config = self.config
        if config.anneal_lr:
            for group in self.optimiser.param_groups:
                group["lr"] = config.learning_rate * (1.0 - progress)

        indices = np.arange(config.batch_size)
        clip_fractions, approx_kl = [], 0.0
        policy_loss = value_loss = entropy_loss = 0.0
        stopped = False

        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, config.batch_size, config.minibatch_size):
                chunk = indices[start:start + config.minibatch_size]
                _, _, log_prob, entropy, value = self.policy.act(
                    batch["obs"][chunk], batch["actions"][chunk])
                log_ratio = log_prob - batch["log_probs"][chunk]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # The low-variance estimator from Schulman's "Approximating
                    # KL divergence"; the naive -log_ratio mean is unbiased but
                    # noisy enough to trip the early stop at random.
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fractions.append(
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                advantage = batch["advantages"][chunk]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                loss_unclipped = -advantage * ratio
                loss_clipped = -advantage * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                policy = torch.max(loss_unclipped, loss_clipped).mean()

                if config.clip_vloss:
                    old = batch["values"][chunk]
                    clipped = old + torch.clamp(value - old, -config.clip_coef, config.clip_coef)
                    value_term = 0.5 * torch.max(
                        (value - batch["returns"][chunk]) ** 2,
                        (clipped - batch["returns"][chunk]) ** 2).mean()
                else:
                    value_term = 0.5 * ((value - batch["returns"][chunk]) ** 2).mean()

                entropy_term = entropy.mean()
                loss = policy - config.ent_coef * entropy_term + config.vf_coef * value_term

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), config.max_grad_norm)
                self.optimiser.step()

                policy_loss, value_loss, entropy_loss = (
                    policy.item(), value_term.item(), entropy_term.item())

            if config.target_kl is not None and approx_kl > config.target_kl:
                stopped = True
                break

        predicted, actual = batch["values"].cpu().numpy(), batch["returns"].cpu().numpy()
        variance = np.var(actual)
        return dict(policy_loss=policy_loss, value_loss=value_loss,
                    entropy=entropy_loss, approx_kl=approx_kl,
                    clip_fraction=float(np.mean(clip_fractions)),
                    explained_variance=float("nan") if variance == 0
                    else float(1 - np.var(actual - predicted) / variance),
                    learning_rate=self.optimiser.param_groups[0]["lr"],
                    saturated=float(batch.get("saturated", float("nan"))),
                    early_stopped=stopped)

    # --- persistence ------------------------------------------------------

    def state_dict(self):
        return dict(policy=self.policy.state_dict(),
                    optimiser=self.optimiser.state_dict(),
                    config=dataclasses.asdict(self.config),
                    obs_dim=self.obs_dim, action_dim=self.action_dim)

    def load_state_dict(self, state):
        self.policy.load_state_dict(state["policy"])
        if "optimiser" in state:
            self.optimiser.load_state_dict(state["optimiser"])
