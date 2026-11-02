"""Headless integration test; no policy server or checkpoint required.

MUJOCO_GL=egl python -m unittest test_hybrid_libero
"""
import unittest

import numpy as np

from robot_server import Args, RobotServer


class HoldingPolicy:
    def __init__(self):
        self.observations = []

    def infer(self, obs):
        self.observations.append(obs)
        return {"actions": np.array([[0.] * 6 + [1.]] * 5)}


class OfflineServer(RobotServer):
    def _connect_policy(self):
        self.policy = HoldingPolicy()


class HybridSimulationTest(unittest.TestCase):
    def test_rrt_grasp_handoff_and_reset(self):
        server = OfflineServer(Args(mode="hybrid", viewer=False))
        try:
            self.assertIn("unknown object", server.handle("pick missing"))
            result = server.handle("pick milk_1")
            self.assertTrue(result.startswith("RRT grasp confirmed"), result)
            qpos = server.env.sim.data.qpos.copy()
            qvel = server.env.sim.data.qvel.copy()
            grip_target = server.env.env.robots[0].gripper.current_action.copy()
            server._switch_controller(True)
            np.testing.assert_array_equal(qpos, server.env.sim.data.qpos)
            np.testing.assert_array_equal(qvel, server.env.sim.data.qvel)
            np.testing.assert_array_equal(grip_target, server.env.env.robots[0].gripper.current_action)
            self.assertEqual(server.action_dim, 7)
            result = server.handle('vla "Put the milk in the basket." 10')
            self.assertIn("ran 10 steps", result)
            self.assertEqual(len(server.policy.observations), 2)
            obs = server.policy.observations[0]
            self.assertEqual(obs["prompt"], "Put the milk in the basket.")
            self.assertEqual(obs["observation/image"].shape, (224, 224, 3))
            self.assertEqual(obs["observation/state"].shape, (8,))
            domain = server.env.env
            self.assertTrue(domain._check_grasp(domain.robots[0].gripper, domain.objects_dict["milk_1"].contact_geoms))
            self.assertEqual(server.handle("reset"), "task reset")
            self.assertFalse(server.vla_mode)
            self.assertEqual(server.action_dim, 8)
            self.assertIs(server.scene.sim, server.env.sim)
            server.idle()
        finally:
            server.env.close()
