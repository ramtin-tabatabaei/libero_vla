"""Simulator regressions for the reach target, the reset spread and the MDP.

Guards the three things that were wrong in the first draft and are invisible
without a live scene: the target resolving to a point inside the object, the
arm always starting from the same pose, and a horizon cut-off being reported as
a real arrival.

    NUMBA_CACHE_DIR=/tmp/libero_numba MUJOCO_GL=egl \
      /home/ramtin/miniconda3/envs/libero/bin/python -m unittest test_recovery_sim -v
No display, no policy server and no checkpoint are required.
"""
import pathlib
import unittest

import numpy as np

import subgoal_planner as sp
from recovery.reach_env import (ACTION_DIM, OBS_DIM, OBS_SCHEMA, ReachConfig, ReachEnv,
                                ScriptedPick, grasp_point)

OSC_POS_SCALE = 0.05


def servo(observation):
    """The straight-line controller from RobotServer._goto_xyz_osc."""
    action = np.zeros(ACTION_DIM)
    action[:3] = np.clip(observation[:3] / OSC_POS_SCALE, -1.0, 1.0)
    action[-1] = -1.0
    return action




class ReachEnvTest(unittest.TestCase):
    """One scene, built once -- building costs about a second."""

    @classmethod
    def setUpClass(cls):
        cls.env = ReachEnv("libero_object", 0, seed=5)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.observation = self.env.reset()

    def test_observation_matches_the_declared_width(self):
        self.assertEqual(self.observation.shape, (OBS_DIM,))
        self.assertTrue(np.isfinite(self.observation).all())

    def test_distance_channel_agrees_with_the_simulator(self):
        measured = np.linalg.norm(
            grasp_point(self.env.world, self.env.target_name, self.env.config.grasp_depth)
            - self.env.world.eef())
        self.assertAlmostEqual(float(self.observation[6]), float(measured), places=5)

    def test_the_scripted_pick_lifts_the_object(self):
        """The sanity floor: if a canonical top-down pick cannot solve this
        scene, the task is mis-specified and PPO will not rescue it.

        Run from the benchmark's own start pose with randomisation off, so a
        failure here means the task changed and not that the dice came up bad.
        """
        env = ReachEnv("libero_object", 0, seed=0, config=ReachConfig(
            arm_noise=0.0, randomise_gripper=False, shuffle_objects=False))
        scripted = ScriptedPick()
        try:
            observation = env.reset()
            for _ in range(env.config.horizon):
                observation, _, terminated, truncated, info = env.step(
                    scripted.act(observation)[0])
                if terminated or truncated:
                    break
            self.assertTrue(info["success"], "scripted pick failed: %r" % info)
            self.assertTrue(info["engaged"])
            self.assertGreaterEqual(info["progress"], 1.0)
        finally:
            env.close()

    def test_servoing_straight_at_the_object_shoves_it(self):
        """Why the baseline had to change.

        The repo's existing straight-line reach drives diagonally at a point
        2 cm inside the can, catches it with the open fingers and pushes it
        across the table -- measured at 42 cm of travel. It solved the old
        reach-only task every time, which is what made that task the wrong one.
        """
        env = ReachEnv("libero_object", 0, seed=0, config=ReachConfig(
            arm_noise=0.0, randomise_gripper=False, shuffle_objects=False))
        try:
            observation = env.reset()
            start = env.world.pos(env.target_name).copy()
            for _ in range(env.config.horizon):
                observation, _, terminated, truncated, info = env.step(servo(observation))
                if terminated or truncated:
                    break
            self.assertFalse(info["success"])
            moved = float(np.linalg.norm(env.world.pos(env.target_name)[:2] - start[:2]))
            self.assertGreater(moved, 0.10, "expected the servo to shove the can")
        finally:
            env.close()

    def test_the_horizon_truncates_without_terminating(self):
        """PPO bootstraps through a truncation and not through a success, so
        the two must never be reported as the same thing."""
        still = ReachEnv("libero_object", 0, config=ReachConfig(horizon=12), seed=1)
        try:
            still.reset()
            for _ in range(12):
                _, _, terminated, truncated, info = still.step(np.zeros(ACTION_DIM))
            self.assertTrue(truncated)
            self.assertFalse(terminated)
            self.assertFalse(info["success"])
            self.assertEqual(info["steps"], 12)
        finally:
            still.close()

    def test_success_terminates_without_truncating(self):
        """Driven in `reach` mode, where the straight-line servo is reliable,
        so the terminal path is exercised deterministically."""
        env = ReachEnv("libero_object", 0, config=ReachConfig(mode="reach", shuffle_objects=False), seed=2)
        try:
            observation = env.reset()
            self.assertEqual(env.mode, "reach")
            for _ in range(env.config.horizon):
                observation, _, terminated, truncated, info = env.step(servo(observation))
                if terminated or truncated:
                    break
            self.assertTrue(terminated, "scripted reach failed in reach mode: %r" % info)
            self.assertFalse(truncated)
            self.assertLess(info["distance"], env.config.success_radius)
        finally:
            env.close()

    def test_a_drawer_task_is_classified_as_articulated(self):
        """The question bare reaching could not answer: a drawer has nothing to
        pick up, so success is its own joint moving."""
        env = ReachEnv("libero_goal", 0, seed=0)
        try:
            env.reset()
            self.assertEqual(env.mode, "articulate")
            self.assertEqual(env.joint_name, "wooden_cabinet_1_middle_level")
            self.assertGreater(env.joint_threshold, 0.0)
        finally:
            env.close()

    def test_a_movable_object_is_classified_as_graspable(self):
        self.assertEqual(self.env.mode, "grasp")
        self.assertIsNone(self.env.joint_name)

    def test_resets_spread_the_arm_out(self):
        """A recovery controller is entered from wherever the VLA left the arm.
        If every episode started from the benchmark's home pose the policy
        would only ever learn that one approach."""
        starts = np.array([self.env.reset()[:3] for _ in range(6)])
        self.assertGreater(float(np.linalg.norm(starts.std(axis=0))), 1e-3)

    def test_actions_are_clipped_before_they_reach_the_controller(self):
        observation, _, _, _, _ = self.env.step(np.full(ACTION_DIM, 5.0))
        self.assertTrue(np.isfinite(observation).all())
        np.testing.assert_array_less(np.abs(self.env.previous_action), 1.0 + 1e-9)


    def test_fixed_collision_surfaces_are_visible_near_the_wrist(self):
        import mujoco
        env=self.env
        env.reset()
        candidates=env.obstacles.geom_ids
        # Static support geometries were absent from the old object-only slots.
        fixed=[g for g in candidates if env.world.m.body_weldid[env.world.m.geom_bodyid[g]] == 0]
        self.assertTrue(fixed)
        geom=min(fixed,key=lambda g: np.linalg.norm(env.world.d.geom_xpos[g]-env.world.eef()))
        boxes=env.world.m.geom_aabb
        R=env.world.d.geom_xmat[geom].reshape(3,3)
        center=env.world.d.geom_xpos[geom].copy()
        if env.world.m.geom_type[geom] != mujoco.mjtGeom.mjGEOM_PLANE:
            center += R@boxes[geom,:3]
        env.obstacles.observe(center, np.eye(3))
        selected=env.obstacles.last_ids
        self.assertTrue(any(g in fixed for g in selected))
        self.assertTrue(all(g not in env.contacts.robot for g in selected))

    def test_final_bonus_waits_for_held_success_and_latches_reset(self):
        from unittest.mock import patch
        env = self.env
        env.reset()
        stage = dict(env._stage(), touching=True, stable=True, final_stage=True,
                     finished=True, engaged=True, progress=1.1)
        with patch.object(env, '_stage', return_value=stage):
            for i in range(env.config.success_hold):
                _, r, done, _, info = env.step(np.zeros(ACTION_DIM))
                expected = (7. if i == 0 else 0.) + (10. if done else 0.)
                self.assertEqual(info['reward_terms']['stages'], expected)
                self.assertEqual(done, i == env.config.success_hold-1)
                raw = info['calibration']
                self.assertAlmostEqual(raw['r_task'] + raw['r_collision_total']
                                       + info['reward_terms']['action'] + info['reward_terms']['time'], r)
        env.reset()
        self.assertEqual(env.awarded, (False,False,False))

    def test_arm_contacts_have_physical_forces_and_penalties(self):
        env = self.env
        sim = env.env.sim
        gid = sim.model.geom_name2id('robot0_link4_collision')
        self.assertNotEqual(int(sim.model.geom_contype[gid]), 0)
        self.assertNotEqual(int(sim.model.geom_conaffinity[gid]), 0)
        joint = sim.model.joint_name2id('butter_1_joint0')
        adr = int(sim.model.jnt_qposadr[joint])
        sim.data.qpos[adr:adr+3] = sim.data.geom_xpos[gid]
        sim.forward()
        force_seen = False
        for _ in range(5):
            _, _, _, _, info = env.step(np.zeros(ACTION_DIM))
            force_seen |= info['reward_terms']['collision'] < 0
        self.assertTrue(force_seen, 'Arm/object overlap produced no physical collision penalty')

    def test_shuffles_are_seeded_and_actually_move_objects(self):
        a = ReachEnv('libero_object', 0, seed=33)
        b = ReachEnv('libero_object', 0, seed=33)
        try:
            a.reset(); b.reset()
            self.assertTrue(a.shuffle_mapping)
            self.assertEqual(a.shuffle_mapping, b.shuffle_mapping)
            for name in a.shuffle_mapping:
                np.testing.assert_allclose(a.world.pos(name), b.world.pos(name), atol=1e-8)
        finally:
            a.close(); b.close()

    def test_articulation_requires_the_goal_predicate(self):
        env = ReachEnv('libero_goal', 0, seed=0)
        try:
            env.reset()
            jid = env.world.joint_info(env.joint_name)['id']
            adr = env.world.m.jnt_qposadr[jid]
            env.world.d.qpos[adr] = 0.0
            env.env.sim.forward()
            self.assertFalse(env._stage()['finished'])
            env.world.d.qpos[adr] = env.joint_goal
            env.env.sim.forward()
            self.assertTrue(env._stage()['finished'])
        finally:
            env.close()


class TargetTest(unittest.TestCase):
    """The grasp point, on the scenes where the body origin is unusable."""

    def resolve(self, suite, task_id):
        env = ReachEnv(suite, task_id, seed=0)
        try:
            env.reset()
            world = env.world
            name = env.target_name
            low, high = world.aabb(name)
            return dict(name=name, origin=world.pos(name),
                        target=grasp_point(world, name, env.config.grasp_depth),
                        low=np.asarray(low), high=np.asarray(high), grip=world.eef())
        finally:
            env.close()

    def test_a_fixture_target_is_lifted_onto_a_surface_the_gripper_can_occupy(self):
        """close the microwave: the body origin sits deep inside the shell, a
        measured 47.9 cm from the grip site and unreachable. The grasp point is
        on the top face, 18.3 cm away."""
        got = self.resolve("libero_90", 33)
        self.assertEqual(got["name"], "microwave_1")
        to_origin = np.linalg.norm(got["grip"] - got["origin"])
        to_target = np.linalg.norm(got["grip"] - got["target"])
        self.assertLess(to_target, to_origin - 0.1)
        self.assertGreaterEqual(got["target"][2], got["low"][2])

    def test_a_region_site_resolves_to_itself(self):
        """open the middle drawer: the target is a bare region site, whose
        bounds collapse to a point, so the grasp offset must be zero."""
        got = self.resolve("libero_goal", 0)
        self.assertEqual(got["name"], "wooden_cabinet_1_middle_region")
        np.testing.assert_allclose(got["target"], got["origin"], atol=1e-9)

    def test_a_bowl_is_aimed_at_its_rim_not_its_origin(self):
        """akita_black_bowl_1's origin sits below its own collision mesh --
        reaching for it drives the fingers into the table."""
        got = self.resolve("libero_goal", 1)
        self.assertEqual(got["name"], "akita_black_bowl_1")
        self.assertGreater(got["target"][2], got["origin"][2])
        self.assertGreaterEqual(got["target"][2], got["low"][2])


class RecoveryPolicyTest(unittest.TestCase):
    """The seam into run_vla.py, exercised against a live scene.

    Written against a checkpoint saved here rather than a trained one, the way
    the repo's other policy tests swap in a local double instead of reaching for
    a server: what is under test is the observation packing, the normaliser
    round trip and the hand-over bookkeeping, none of which need a good policy.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        import torch

        from recovery.ppo import PPO, PPOConfig

        agent = PPO(OBS_DIM, ACTION_DIM, PPOConfig(num_envs=2, num_steps=4,
                                                   num_minibatches=1))
        state = agent.state_dict()
        state["normalisers"] = dict(obs_rms=None, return_rms=None)
        state["obs_schema"] = OBS_SCHEMA
        cls.directory = tempfile.mkdtemp()
        cls.checkpoint = pathlib.Path(cls.directory) / "checkpoint.pt"
        torch.save(state, cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_it_drives_a_live_env_with_in_range_actions(self):
        from recovery.policy import ReachPolicy

        env = ReachEnv("libero_object", 0, seed=4)
        try:
            env.reset()
            raw = env.env.env._get_observations()
            policy = ReachPolicy(str(self.checkpoint), env.env, device="cpu")
            self.assertEqual(policy.target_name, "alphabet_soup_1")
            self.assertTrue(policy.movable)
            for _ in range(5):
                action = policy.act(raw)
                self.assertEqual(action.shape, (ACTION_DIM,))
                np.testing.assert_array_less(np.abs(action), 1.0 + 1e-9)
                raw, _, _, _ = env.env.step(action)
            self.assertFalse(policy.arrived)
        finally:
            env.close()

    def test_live_observation_matches_training(self):
        from recovery.policy import ReachPolicy
        for suite, tid in [('libero_object', 0), ('libero_goal', 0)]:
            env = ReachEnv(suite, tid, seed=4)
            try:
                expected = env.reset()
                raw = env.env.env._get_observations()
                policy = ReachPolicy(str(self.checkpoint), env.env, config=env.config)
                np.testing.assert_allclose(policy.observe(raw), expected, atol=1e-5)
                # Observing repeatedly must not manufacture a stable grasp.
                before = policy.reader.contact_streak
                policy.observe(raw)
                self.assertEqual(policy.reader.contact_streak, before)
            finally:
                env.close()

    def test_the_dropped_object_can_override_the_default_target(self):
        """DropObject names what it made fall, and on a multi-object task that
        is not necessarily obj_of_interest[0]."""
        from recovery.policy import ReachPolicy

        env = ReachEnv("libero_object", 0, seed=4)
        try:
            env.reset()
            policy = ReachPolicy(str(self.checkpoint), env.env, target="butter_1",
                                 device="cpu")
            self.assertEqual(policy.target_name, "butter_1")
            self.assertGreater(policy.distance(), 0.0)
        finally:
            env.close()


class RenderTest(unittest.TestCase):
    def test_a_camera_env_yields_an_upright_frame(self):
        env = ReachEnv("libero_object", 0, seed=0, cameras=True)
        try:
            env.reset()
            frame = env.frame()
            self.assertEqual(frame.shape, (256, 256, 3))
            self.assertEqual(frame.dtype, np.uint8)
            # The table fills the lower half of an upright agentview frame and
            # is brighter than the dark upper background.
            self.assertGreater(float(frame[160:, :].mean()), float(frame[:60, :].mean()))
        finally:
            env.close()

    def test_a_renderer_free_env_refuses_to_pretend(self):
        env = ReachEnv("libero_object", 0, seed=0)
        try:
            env.reset()
            with self.assertRaises(ValueError):
                env.frame()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
