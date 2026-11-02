"""Reward, observation and PPO-maths regressions for the recovery policy.

Pure numpy and torch -- no simulator, no display, no checkpoint, so this runs
anywhere in under a second. The simulator half lives in test_recovery_sim.py.

    /home/ramtin/miniconda3/envs/libero/bin/python -m unittest test_recovery -v
"""
import unittest

import numpy as np
import torch

from recovery.ppo import PPO, PPOConfig, ActorCritic
from recovery.reach_env import (ACTION_DIM, OBS_DIM, ReachConfig, observation_vector,
                                rotation_6d, task_list)
from recovery.reward import (RewardConfig, alignment_error, alignment_gate, collision_cost,
                             progress_gain, progress_potential, stage_bonus,
                             step_reward)
from recovery.scene import derangement
from recovery.vec_env import RunningMeanStd


class ObservationTest(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            grip=np.zeros(3), grip_mat=np.eye(3), target=np.array([0.1, 0.2, 0.2]),
            span=np.array([0.06, 0.06, 0.08]), joint_pos=np.zeros(7), joint_vel=np.zeros(7),
            gripper_qpos=np.array([0.02, -0.02]), gripper_qvel=np.zeros(2),
            previous_action=np.zeros(ACTION_DIM), engaged=0.0, progress=0.0)

    def test_stage_channels_are_the_last_two(self):
        """The value function cannot tell hovering from holding without them,
        and those two states have very different returns."""
        vector = observation_vector(**{**self.kwargs, "engaged": 1.0, "progress": 0.4})
        np.testing.assert_allclose(vector[48:50], [1.0, 0.4], atol=1e-6)

    def test_vector_has_the_width_the_network_is_built_for(self):
        vector = observation_vector(**self.kwargs)
        self.assertEqual(vector.shape, (OBS_DIM,))
        self.assertEqual(vector.dtype, np.float32)

    def test_distance_channel_matches_the_offset_it_is_derived_from(self):
        vector = observation_vector(**self.kwargs)
        np.testing.assert_allclose(vector[:3], [0.1, 0.2, 0.2], atol=1e-6)
        self.assertAlmostEqual(float(vector[6]), float(np.linalg.norm([0.1, 0.2, 0.2])), places=6)

    def test_gripper_frame_offset_rotates_with_the_wrist(self):
        """A wrist turned 90 deg about z must see the goal on a different axis."""
        turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        vector = observation_vector(**{**self.kwargs, "grip_mat": turned})
        np.testing.assert_allclose(vector[:3], [0.1, 0.2, 0.2], atol=1e-6)
        np.testing.assert_allclose(vector[3:6], turned.T @ np.array([0.1, 0.2, 0.2]), atol=1e-6)
        self.assertAlmostEqual(float(vector[6]), float(np.linalg.norm([0.1, 0.2, 0.2])), places=6)

    def test_observation_is_translation_invariant(self):
        """The floor-bin scenes sit a metre below the kitchen ones; the policy
        must not be able to tell, or one network cannot cover both."""
        low = observation_vector(**self.kwargs)
        shifted = dict(self.kwargs)
        shifted["grip"] = self.kwargs["grip"] + 1.0
        shifted["target"] = self.kwargs["target"] + 1.0
        np.testing.assert_allclose(low, observation_vector(**shifted), atol=1e-6)

    def test_rotation_6d_is_the_first_two_columns(self):
        matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(rotation_6d(matrix), [0.0, 1.0, 0.0, -1.0, 0.0, 0.0], atol=1e-9)


class RewardTest(unittest.TestCase):
    def test_legacy_formula_with_all_terms(self):
        c = RewardConfig(latch_bonuses=False, smooth_alignment=False, integrated_progress=False)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        r, _, terms = step_reward(.12, .05, rotation, np.eye(3),
                                  (True, True, True), (True, True, True),
                                  [10., 150.], np.ones(7), c, return_terms=True)
        expected = float(progress_gain(.05, c)) * .07 - .5*np.pi/2 + 17 - .02*60 - 1 - .07
        self.assertAlmostEqual(r, expected)
        self.assertAlmostEqual(sum(terms.values()), expected)

    def test_unlatched_ablation_pays_every_step(self):
        c = RewardConfig(latch_bonuses=False)
        bonus, latch = stage_bonus((True, True, False), (False,)*3, c)
        self.assertEqual(bonus, 7.)
        self.assertEqual(stage_bonus((True, True, False), latch, c)[0], 7.)

    def test_default_latching_survives_release_and_regrasp(self):
        c = RewardConfig()
        bonus, latch = stage_bonus((True, True, False), (False,)*3, c)
        self.assertEqual(bonus, 7.)
        self.assertEqual(stage_bonus((True, True, False), latch, c)[0], 0.)
        _, latch = stage_bonus((False, False, False), latch, c)
        self.assertEqual(stage_bonus((True, True, False), latch, c)[0], 0.)
        self.assertEqual(stage_bonus((True, True, True), latch, c)[0], 10.)
        self.assertEqual(stage_bonus((True, True, False), (False,)*3, c)[0], 7.)

    def test_severe_threshold_is_strict_and_independent_of_cap(self):
        c = RewardConfig()
        self.assertEqual(collision_cost([100.], c), (1., 0.))
        self.assertEqual(collision_cost([100.01], c), (1., 1.))
        self.assertEqual(collision_cost([], c, contact_force=101), (0., 1.))
        self.assertEqual(collision_cost([], c), (0., 0.))

    def test_legacy_alignment_is_only_active_inside_radius(self):
        c = RewardConfig(w_action=0, smooth_alignment=False)
        rot = np.diag([-1., -1., 1.])
        def reward(d):
            return step_reward(d, d, rot, np.eye(3), (False,)*3, (False,)*3,
                               [], np.zeros(7), c)[0]
        self.assertEqual(reward(c.align_distance), 0.)
        self.assertAlmostEqual(reward(c.align_distance-.001), -.5*np.pi)

    def test_smooth_alignment_midpoint_continuity_and_tails(self):
        c = RewardConfig()
        self.assertEqual(alignment_gate(c.align_distance, c), .5)
        self.assertLess(abs(alignment_gate(.1-1e-8,c)-alignment_gate(.1+1e-8,c)), 1e-6)
        self.assertGreater(alignment_gate(.09,c), alignment_gate(.11,c))
        self.assertAlmostEqual(alignment_gate(.11,c), 1/(1+np.exp(1)))
        with np.errstate(over='raise', invalid='raise'):
            self.assertEqual(alignment_gate(1e5,c), 0.)
        r, _ = step_reward(.1, .1, np.diag([-1.,-1.,1.]), np.eye(3),
                           (False,)*3, (False,)*3, [], np.zeros(7), c)
        self.assertAlmostEqual(r, -.5 * .5 * np.pi)

    def test_action_uses_squared_norm_not_mean(self):
        r, _ = step_reward(.2, .2, np.eye(3), None, (False,)*3, (False,)*3,
                           [], np.ones(7), RewardConfig())
        self.assertAlmostEqual(r, -.07)

    def test_default_progress_matches_antiderivative_and_telescopes(self):
        c = RewardConfig()
        self.assertTrue(c.integrated_progress)
        def reward(a, b):
            return step_reward(a, b, np.eye(3), None, (False,)*3, (False,)*3,
                               [], np.zeros(7), c)[0]
        self.assertAlmostEqual(reward(.2, .35) + reward(.35, .2), 0.)
        expected = 10*(.2-.1) + 2*(np.exp(-1)-np.exp(-2))
        self.assertAlmostEqual(reward(.2, .1), expected)
        self.assertAlmostEqual(reward(.2, .1) + reward(.1, .2), 0.)
        route = [.2, .1, .07, .25, .31, .2]
        self.assertAlmostEqual(sum(reward(a,b) for a,b in zip(route,route[1:])),0.,places=12)
        c.integrated_progress = False
        self.assertGreater(reward(.2,.1) + reward(.1,.2), .46)

    def test_bad_configs_are_refused(self):
        for bad in (ReachConfig(horizon=0), ReachConfig(success_radius=0),
                    ReachConfig(lift_target=0), ReachConfig(contact_hold=0),
                    ReachConfig(mode='fly'), ReachConfig(settle_steps=-1)):
            with self.assertRaises(ValueError): bad.validate()
        for bad in (RewardConfig(w_action=-1), RewardConfig(force_cap=0),
                    RewardConfig(w_reference=0), RewardConfig(w_align=float('nan')),
                    RewardConfig(align_temperature=0), RewardConfig(align_temperature=-.1)):
            with self.assertRaises(ValueError): bad.validate()


class CalibrationTest(unittest.TestCase):
    def test_quantiles_capture_rare_impacts_and_split_modes(self):
        from recovery.calibration import METRICS, summarise_calibration
        quiet = dict.fromkeys(METRICS, 0.)
        impact = dict(quiet, peak_force_n=200., sum_bad_force_n=250.,
                      r_collision=-2., r_severe=-1., r_collision_total=-3.,
                      r_task=7., severe_event=1.)
        samples = [('grasp',quiet)]*99 + [('articulate',impact)]
        stats = summarise_calibration(samples)
        self.assertEqual(stats['overall']['peak_force_n']['count'],100)
        self.assertEqual(stats['overall']['peak_force_n']['p95'],0.)
        self.assertEqual(stats['overall']['peak_force_n']['max'],200.)
        self.assertEqual(stats['overall']['peak_force_n']['nonzero']['p95'],200.)
        self.assertAlmostEqual(stats['overall']['r_collision_total']['mean'],-.03)
        self.assertEqual(stats['by_mode']['articulate']['r_task']['mean'],7.)
        self.assertEqual(summarise_calibration([])['overall']['r_task'], {'count':0})

    def test_rollout_logs_raw_metrics_without_completed_episodes(self):
        from recovery.calibration import METRICS
        class FakeEnv:
            num_envs = 2
            def step(self, action):
                raw = dict.fromkeys(METRICS, 0.)
                raw.update(peak_force_n=20., r_task=7., r_collision=-2.)
                info = dict(mode='grasp', calibration=raw)
                return np.zeros((2, OBS_DIM)), np.full(2,.01), np.zeros(2,bool), np.zeros(2,bool), [info,info]
        agent = PPO(OBS_DIM, ACTION_DIM, PPOConfig(num_envs=2, num_steps=4, num_minibatches=1))
        episodes=[]
        batch, _, _ = agent.collect(FakeEnv(), torch.zeros(2,OBS_DIM), torch.zeros(2), episodes)
        self.assertEqual(episodes, [])
        self.assertEqual(batch['calibration']['overall']['r_task']['count'],8)
        self.assertEqual(batch['calibration']['overall']['r_task']['mean'],7.)

    def test_old_checkpoint_reward_semantics_are_preserved(self):
        from recovery.policy import config_from_dict
        c = config_from_dict({'reward': {'latch_bonuses':False}})
        self.assertFalse(c.reward.smooth_alignment)
        self.assertFalse(c.reward.terminal_stage_bonus)
        self.assertFalse(c.reward.latch_bonuses)
        self.assertTrue(config_from_dict({}).reward.smooth_alignment)


class CollisionObstacleTest(unittest.TestCase):
    def setUp(self):
        import mujoco
        from types import SimpleNamespace
        from recovery.scene import ContactMonitor, CollisionObstacles
        self.m = mujoco.MjModel.from_xml_string("""
        <mujoco><worldbody>
          <body name="robot0_arm" pos="0 0 1"><freejoint/>
            <geom name="arm" type="sphere" size=".03"/>
            <body name="gripper0_pad" pos=".1 0 0">
              <geom name="pad" type="sphere" size=".01"/>
            </body>
          </body>
          <geom name="cabinet_face" type="box" pos="2 0 .2" size="2 .5 .05"/>
          <geom name="visual" type="box" pos="0 0 .01" size=".01 .01 .01" contype="0" conaffinity="0"/>
          <body name="target" pos=".4 0 0"><freejoint/>
            <geom name="target_geom" type="sphere" size=".05"/>
          </body>
        </worldbody></mujoco>""")
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m,self.d)
        self.world = SimpleNamespace(m=self.m,d=self.d)
        self.contacts = ContactMonitor(self.world)
        self.contacts.allow({self.m.geom('pad').id}, {self.m.geom('target_geom').id})
        self.obstacles = CollisionObstacles(self.world,self.contacts,4)

    def test_fixed_face_and_target_are_candidates_but_robot_and_visual_are_not(self):
        ids=set(self.obstacles.geom_ids)
        self.assertEqual(ids,{self.m.geom('cabinet_face').id,self.m.geom('target_geom').id})
        # The target can make undesired arm contact despite its pad exemption.
        self.assertIn(self.m.geom('target_geom').id, ids)

    def test_surface_distance_ranking_wrist_frame_and_padding(self):
        rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        obs=self.obstacles.observe(np.zeros(3),rot)
        self.assertEqual(self.obstacles.last_ids[0],self.m.geom('cabinet_face').id)
        np.testing.assert_allclose(obs[0,:3],rot.T@np.array([2.,0.,.2]))
        np.testing.assert_allclose(obs[0,3:6],[4,1,.1])
        np.testing.assert_array_equal(obs[:,6],[1,1,0,0])
        np.testing.assert_array_equal(obs[2:],np.zeros((2,7)))
        before=obs.copy()
        self.d.geom_xpos[:] += [3,4,5]
        np.testing.assert_allclose(self.obstacles.observe([3,4,5],rot),before,atol=1e-12)

    def test_live_geometry_is_updated(self):
        gid=self.m.geom('target_geom').id
        self.obstacles.observe(np.zeros(3),np.eye(3))
        self.d.geom_xpos[gid]=[.06,0,0]
        obs=self.obstacles.observe(np.zeros(3),np.eye(3))
        self.assertEqual(self.obstacles.last_ids[0],gid)
        np.testing.assert_allclose(obs[0,:3],[.06,0,0])

    def test_infinite_plane_has_finite_features(self):
        import mujoco
        from types import SimpleNamespace
        from recovery.scene import ContactMonitor, CollisionObstacles
        m=mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom name="floor" type="plane" size="0 0 .1"/>
          <body name="robot0_arm" pos="0 0 1"><freejoint/><geom type="sphere" size=".02"/></body>
          </worldbody></mujoco>""")
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        world=SimpleNamespace(m=m,d=d)
        obstacles=CollisionObstacles(world,ContactMonitor(world),4)
        with np.errstate(over='raise',invalid='raise'):
            obs=obstacles.observe([4,3,.1],np.eye(3))
        self.assertTrue(np.isfinite(obs).all())
        np.testing.assert_allclose(obs[0],[0,0,-.1,2,2,0,1])

    def test_schema_rejects_same_width_old_semantics(self):
        from recovery.policy import check_schema
        from recovery.reach_env import OBS_SCHEMA
        with self.assertRaisesRegex(ValueError,'wrist frame'):
            check_schema(dict(obs_dim=OBS_DIM,obs_schema='task_collision_v1'))
        check_schema(dict(obs_dim=OBS_DIM,obs_schema=OBS_SCHEMA))


class ContactTaxonomyTest(unittest.TestCase):
    def test_only_permitted_target_contacts_are_exempt(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from recovery.scene import ContactMonitor
        monitor = ContactMonitor.__new__(ContactMonitor)
        monitor.m = object()
        monitor.robot, monitor.pads, monitor.target = {0, 1}, {1}, {2}
        monitor._buffer = np.zeros(6)
        # arm-target, pad-target, pad-obstacle, scenery-scenery
        pairs = [(0,2), (1,2), (1,3), (2,3)]
        monitor.d = SimpleNamespace(ncon=4, contact=[SimpleNamespace(geom1=a, geom2=b) for a,b in pairs])
        def force(m, d, i, buf):
            buf[:] = 0
            buf[:3] = [3,4,0] if i != 1 else [120,0,0]
        with patch('recovery.scene.mujoco.mj_contactForce', side_effect=force):
            bad, valid, peak = monitor.measure()
        np.testing.assert_allclose(bad, [5,5])
        self.assertTrue(valid)
        self.assertEqual(peak, 120)
        self.assertEqual(collision_cost(bad, RewardConfig(), peak), (.2,1.))


class ShuffleTest(unittest.TestCase):
    def test_seeded_shuffle_preserves_pose_and_rejects_overlaps(self):
        from recovery.scene import shuffle_scene
        from types import SimpleNamespace
        slots = dict(a=1, b=8, c=15)
        state = np.zeros(22)
        state[1:4], state[8:11], state[15:18] = [0, 0, .05], [.3, 0, .05], [.38, 0, .05]
        # a is wider than b; swapping it into b intersects fixed c.
        bounds = dict(a=(np.array([-.1,-.1,0]), np.array([.1,.1,.1])),
                      b=(np.array([.28,-.02,0]), np.array([.32,.02,.1])),
                      c=(np.array([.36,-.02,0]), np.array([.4,.02,.1])))
        world = SimpleNamespace(aabb=lambda n: bounds[n])
        out, mapping = shuffle_scene(world, state, slots, ['a','b'], np.random.default_rng(2))
        np.testing.assert_array_equal(out, state)
        self.assertEqual(mapping, {})
        bounds['c'] = (np.array([.9,-.02,0]),np.array([1.,.02,.1]))
        out, mapping = shuffle_scene(world, state, slots, ['a','b'], np.random.default_rng(2))
        np.testing.assert_allclose(out[1:3], state[8:10])
        np.testing.assert_array_equal(out[4:8], state[4:8])
        self.assertEqual(mapping, dict(a='b', b='a'))


class NormaliserTest(unittest.TestCase):
    def test_running_moments_match_numpy_over_several_batches(self):
        """Tolerance is 1e-4, not 1e-6, on purpose.

        The accumulator starts at count=1e-4 rather than 0 so the first batch
        cannot divide by zero -- the same prior OpenAI's and stable-baselines3'
        versions carry. That pulls the mean towards zero by count/n, which over
        77 samples is a relative 1.3e-6: real, deliberate, and far below
        anything the policy can feel.
        """
        rng = np.random.default_rng(0)
        rms = RunningMeanStd((4,))
        everything = []
        for _ in range(7):
            batch = rng.normal(3.0, 2.0, size=(11, 4))
            rms.update(batch)
            everything.append(batch)
        everything = np.concatenate(everything)
        np.testing.assert_allclose(rms.mean, everything.mean(axis=0), rtol=1e-4)
        np.testing.assert_allclose(rms.var, everything.var(axis=0), rtol=1e-3)

    def test_statistics_survive_a_round_trip(self):
        """A policy restored without its normaliser sees a different input
        space and scores near zero, so the two travel together."""
        rms = RunningMeanStd((3,))
        rms.update(np.random.default_rng(1).normal(size=(20, 3)))
        restored = RunningMeanStd((3,))
        restored.load_state_dict(rms.state_dict())
        np.testing.assert_array_equal(rms.mean, restored.mean)
        np.testing.assert_array_equal(rms.var, restored.var)
        self.assertEqual(rms.count, restored.count)


class AdvantageTest(unittest.TestCase):
    def setUp(self):
        self.agent = PPO(OBS_DIM, ACTION_DIM, PPOConfig(num_envs=2, num_steps=4,
                                                        num_minibatches=1))

    def test_gae_matches_the_recursion_written_out_by_hand(self):
        gamma, lam = self.agent.config.gamma, self.agent.config.gae_lambda
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.tensor([[0.5], [0.5], [0.5]])
        dones = torch.zeros(3, 1)
        advantages, returns = self.agent.advantages(
            rewards, values, dones, torch.tensor([0.0]), torch.tensor([0.0]))

        d2 = 3.0 + gamma * 0.0 - 0.5
        d1 = 2.0 + gamma * 0.5 - 0.5
        d0 = 1.0 + gamma * 0.5 - 0.5
        a2 = d2
        a1 = d1 + gamma * lam * a2
        a0 = d0 + gamma * lam * a1
        np.testing.assert_allclose(advantages.squeeze(1).numpy(), [a0, a1, a2], rtol=1e-6)
        np.testing.assert_allclose(returns.squeeze(1).numpy(),
                                   [a0 + 0.5, a1 + 0.5, a2 + 0.5], rtol=1e-6)

    def test_a_terminal_step_cuts_the_bootstrap(self):
        """Value must not flow backwards across an episode boundary."""
        rewards = torch.tensor([[1.0], [1.0]])
        values = torch.tensor([[10.0], [10.0]])
        advantages, _ = self.agent.advantages(
            rewards, values, torch.tensor([[0.0], [1.0]]),
            torch.tensor([0.0]), torch.tensor([0.0]))
        # Step 0 is followed by a done, so its delta sees no future value.
        self.assertAlmostEqual(float(advantages[0]), 1.0 - 10.0, places=5)


class NetworkTest(unittest.TestCase):
    def test_initial_policy_is_quiet_and_its_spread_is_the_configured_one(self):
        """Head gain 0.01 keeps the first actions near zero, and Andrychowicz
        et al.'s initial standard deviation of about 0.5 keeps them inside the
        action range instead of saturating it."""
        policy = ActorCritic(OBS_DIM, ACTION_DIM, log_std_init=-0.5)
        mean = policy.actor(torch.zeros(64, OBS_DIM))
        self.assertLess(float(mean.abs().max()), 0.1)
        np.testing.assert_allclose(policy.log_std.exp().detach().numpy(),
                                   np.full(ACTION_DIM, np.exp(-0.5)), rtol=1e-6)

    def test_act_returns_matching_shapes_and_finite_log_probs(self):
        policy = ActorCritic(OBS_DIM, ACTION_DIM)
        executed, raw, log_prob, entropy, value = policy.act(torch.zeros(5, OBS_DIM))
        self.assertEqual(executed.shape, (5, ACTION_DIM))
        self.assertEqual(raw.shape, (5, ACTION_DIM))
        for tensor in (log_prob, entropy, value):
            self.assertEqual(tensor.shape, (5,))
            self.assertTrue(torch.isfinite(tensor).all())

    def test_deterministic_action_is_reproducible(self):
        policy = ActorCritic(OBS_DIM, ACTION_DIM)
        obs = torch.zeros(3, OBS_DIM)
        first, _, _, _, _ = policy.act(obs, deterministic=True)
        second, _, _, _, _ = policy.act(obs, deterministic=True)
        np.testing.assert_allclose(first.detach().numpy(), second.detach().numpy())

    def test_squashing_keeps_every_action_inside_the_box(self):
        """The whole point of the tanh. Measured on the clipping version, 26.1%
        of action dimensions were being clipped and the distribution mean was
        outside [-1, 1] about 60% of the time on two axes -- so PPO was scoring
        actions the simulator never ran."""
        policy = ActorCritic(OBS_DIM, ACTION_DIM, log_std_init=2.0, squash=True)
        executed, raw, log_prob, _, _ = policy.act(torch.randn(512, OBS_DIM))
        # Not `< 1.0`: tanh is asymptotic but float32 rounds tanh(9) to exactly
        # 1.0, and a deliberately wide policy reaches that. In range is the
        # invariant; strictly inside is not true in floating point.
        self.assertLessEqual(float(executed.abs().max()), 1.0)
        # A wide Gaussian must actually be producing out-of-range samples, or
        # this test would pass even with the tanh removed.
        self.assertGreater(float(raw.abs().max()), 1.0)
        # And the saturated tail must not take the log-probability with it --
        # log(1 - tanh(u)^2) underflows to -inf there, which is why the
        # correction is written via softplus.
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_squashed_log_probability_carries_the_jacobian(self):
        """log p(a) = log p(u) - sum log(1 - tanh(u)^2). Checked against the
        direct form in a range where the direct form is still numerically
        sound."""
        policy = ActorCritic(OBS_DIM, ACTION_DIM, squash=True)
        obs = torch.zeros(4, OBS_DIM)
        raw = torch.full((4, ACTION_DIM), 0.3)
        _, _, log_prob, _, _ = policy.act(obs, raw)
        gaussian = policy.distribution(obs).log_prob(raw).sum(-1)
        correction = torch.log(1 - torch.tanh(raw) ** 2).sum(-1)
        np.testing.assert_allclose(log_prob.detach().numpy(),
                                   (gaussian - correction).detach().numpy(), rtol=1e-5)

    def test_unsquashed_policy_passes_the_raw_sample_through(self):
        policy = ActorCritic(OBS_DIM, ACTION_DIM, squash=False)
        executed, raw, _, _, _ = policy.act(torch.zeros(6, OBS_DIM))
        np.testing.assert_allclose(executed.detach().numpy(), raw.detach().numpy())


class ConfigTest(unittest.TestCase):
    def test_batch_must_divide_into_minibatches(self):
        with self.assertRaises(ValueError):
            PPOConfig(num_envs=8, num_steps=512, num_minibatches=7).validate()

    def test_default_batch_arithmetic(self):
        config = PPOConfig()
        config.validate()
        self.assertEqual(config.batch_size, config.num_envs * config.num_steps)
        self.assertEqual(config.minibatch_size, config.batch_size // config.num_minibatches)

    def test_defaults_are_the_published_ones(self):
        """These are the numbers the module docstring claims to follow. If one
        is changed on a whim, this is the reminder that it came from
        somewhere."""
        config = PPOConfig()
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.gamma, 0.99)
        self.assertEqual(config.gae_lambda, 0.95)
        self.assertEqual(config.clip_coef, 0.2)
        self.assertEqual(config.vf_coef, 0.5)
        self.assertEqual(config.max_grad_norm, 0.5)
        self.assertEqual(config.ent_coef, 0.0)
        self.assertEqual(config.adam_epsilon, 1e-5)
        self.assertEqual(config.update_epochs, 10)


class TaskListTest(unittest.TestCase):
    def test_every_suite_is_enumerated(self):
        self.assertEqual(len(task_list()), 130)

    def test_one_suite_and_one_task(self):
        self.assertEqual(len(task_list(suite="libero_90")), 90)
        self.assertEqual(task_list(suite="libero_goal", task_id=3), [("libero_goal", 3)])

    def test_unknown_selections_are_refused(self):
        with self.assertRaises(ValueError):
            task_list(suite="libero_nope")
        with self.assertRaises(ValueError):
            task_list(suite="libero_goal", task_id=99)


if __name__ == "__main__":
    unittest.main()
