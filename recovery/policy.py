"""Load a recovery checkpoint and use the training observation on a live scene."""
import numpy as np
import torch

import subgoal_planner as sp
from recovery.ppo import ActorCritic, PPOConfig
from recovery.reach_env import ACTION_DIM, OBS_DIM, OBS_SCHEMA, ReachConfig, ReachEnv
from recovery.reward import RewardConfig, alignment_error
from recovery.vec_env import RunningMeanStd


def config_from_dict(saved):
    saved = dict(saved)
    reward = dict(saved.get('reward', {}))
    if reward:
        # Preserve reward semantics of saved pre-smoothing experiments.
        reward.setdefault('smooth_alignment', False)
        reward.setdefault('terminal_stage_bonus', False)
        reward.setdefault('integrated_progress', False)
    saved['reward'] = RewardConfig(**reward)
    return ReachConfig(**saved)


def check_schema(state):
    if state.get('obs_dim') != OBS_DIM or state.get('obs_schema') != OBS_SCHEMA:
        raise ValueError('Checkpoint observation schema is incompatible with collision_geometry_v2. '
                         'Train a new run: obstacle slots now contain fixed and movable collision '
                         'geometries in the wrist frame. Older 84-input weights have different '
                         'feature meanings and cannot be resumed or deployed here.')


class ReachPolicy:
    def __init__(self, checkpoint, env, target=None, config=None, device='cpu', deterministic=True):
        state = torch.load(checkpoint, map_location=device)
        check_schema(state)
        self.config = config or config_from_dict(state.get('env_config', {}))
        self.config.validate()
        self.device = torch.device(device)
        self.deterministic = deterministic
        saved = state.get('config') or {}
        self.policy = ActorCritic(OBS_DIM, ACTION_DIM,
                                 tuple(saved.get('hidden_sizes', PPOConfig.hidden_sizes)),
                                 saved.get('log_std_init', PPOConfig.log_std_init),
                                 squash=saved.get('squash', True)).to(self.device)
        self.policy.load_state_dict(state['policy'])
        self.policy.eval()
        self.obs_rms = None
        normalisers = state.get('normalisers') or {}
        if normalisers.get('obs_rms'):
            self.obs_rms = RunningMeanStd((OBS_DIM,))
            self.obs_rms.load_state_dict(normalisers['obs_rms'])
        self.attach(env, target)

    def attach(self, env, target=None):
        self.env = env
        # A measurement-only adapter: no scene construction or reset. Sharing
        # these methods prevents training/deployment features from drifting.
        reader = ReachEnv.__new__(ReachEnv)
        reader.env, reader.config = env, self.config
        reader.world = sp.World(env)
        reader.target_name = target or list(env.env.obj_of_interest)[0]
        reader._configure_task()
        reader._reset_measurements()
        reader.previous_action = np.zeros(ACTION_DIM)
        self.reader = reader
        self.world, self.target_name = reader.world, reader.target_name
        self.movable = reader.mode == 'grasp'
        self.initial_height = reader.initial_height
        self.streak, self.arrived = 0, False
        self._measurement_time = None
        self._stage_cache = None

    def _stage(self):
        now = float(self.env.sim.data.time)
        if now != self._measurement_time:
            self._stage_cache = self.reader._stage()
            _, self._force_cache = self.reader.contacts.interval()
            self.reader.contacts.clear()
            self.streak = self.streak + 1 if self._stage_cache['finished'] else 0
            self.arrived = self.streak >= self.config.success_hold
            self._measurement_time = now
        return self._stage_cache

    def observe(self, obs):
        stage = self._stage()
        force = self._force_cache
        align = alignment_error(self.reader._grip_mat(), sp.down_yaw(self.reader._grasp_yaw()))
        vector = self.reader._observation(obs, stage, align, force)
        if self.obs_rms is not None:
            vector = np.clip((vector - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -10, 10)
        return vector.astype(np.float32)

    def distance(self, obs=None):
        return float(np.linalg.norm(self.reader._target() - self.world.eef()))

    def act(self, obs):
        vector = self.observe(obs)
        with torch.no_grad():
            action, _, _, _, _ = self.policy.act(
                torch.as_tensor(vector, device=self.device).unsqueeze(0),
                deterministic=self.deterministic)
        action = action.squeeze(0).cpu().numpy()
        self.reader.previous_action = action
        return action

    def run(self, obs, max_steps=300):
        for step in range(max_steps):
            obs, _, done, _ = self.env.step(self.act(obs))
            self._stage()
            if self.arrived:
                return obs, 'ok (recovered %s in %d steps)' % (self.target_name, step + 1)
            if done:
                return obs, 'failed: episode ended after %d steps' % (step + 1)
        return obs, 'failed: %s not recovered in %d steps' % (self.target_name, max_steps)
