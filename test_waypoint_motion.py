"""Simulator regressions for stable editor targets and continuous arm motion.

NUMBA_CACHE_DIR=/tmp/libero_numba MUJOCO_GL=egl python -m unittest test_waypoint_motion -v
No display or policy server is required.
"""
import dataclasses
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import patch

import numpy as np

import subgoal_planner as sp
from waypoint_chain import Chain, Waypoint, PREVIOUS
from waypoint_session import Session


class MotionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.session = Session(sp.Args(task_suite_name="libero_spatial", task_id=0))
        cls.session.load("libero_spatial", 0, cameras=False)
        cls.initial = cls.session.env.sim.get_state().flatten().copy()
        cls.original = Chain.load(Path(__file__).parent / "tests/fixtures/manual_bowl_chain.json",
                                  "libero_spatial", 0)

    @classmethod
    def tearDownClass(cls):
        cls.session.env.close()

    def setUp(self):
        self.s = self.session
        self.s.env.set_init_state(self.initial)
        self.s.attach()

    def test_previous_anchor_and_repeated_click_do_not_accumulate(self):
        start, ori = self.s.gripper_pose()
        w = Waypoint(reference=PREVIOUS, offset=[0, 0, -.03], orientation=ori.tolist(), motion="line")
        chain = dataclasses.replace(self.original, waypoints=[w])
        target = self.s.editor_target(chain, 0, w)[0]
        np.testing.assert_allclose(target, start+[0, 0, -.03])
        for _ in range(3):
            result = self.s.test_waypoint(chain, 0, w, lambda *_: None)
            self.assertTrue(result.startswith("Finished"), result)
            self.assertLess(np.linalg.norm(self.s.world.eef()-target), .002)
            np.testing.assert_allclose(self.s.editor_target(chain, 0, w)[0], target)
        before = self.s.env.sim.get_state().flatten().copy()
        self.s.test_waypoint(chain, 0, w, lambda *_: None)
        np.testing.assert_array_equal(self.s.env.sim.get_state().flatten(), before)
        edited = dataclasses.replace(w, offset=[0, 0, -.04])
        np.testing.assert_allclose(self.s.editor_target(chain, 0, edited)[0], start+[0, 0, -.04])

    def test_object_refresh_and_rotated_local_offset(self):
        name = self.original.waypoints[0].reference
        world = self.s.world
        body = world.domain.obj_body_id[name]
        joint = world.m.body_jntadr[body]
        adr = world.m.jnt_qposadr[joint]
        position = self.s.reference_poses[name][0].copy()
        w = Waypoint(reference=name, offset=[.04, 0, .1])
        old = self.s.editor_target(self.original, 0, w)[0]
        world.d.qpos[adr:adr+3] = position+[.05, .02, 0]
        world.d.qpos[adr+3:adr+7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        world.sim.forward()
        np.testing.assert_allclose(self.s.editor_target(self.original, 0, w)[0], old)
        self.s.refresh_references()
        np.testing.assert_allclose(self.s.editor_target(self.original, 0, w)[0],
                                   position+[.05, .06, .1], atol=1e-7)
        self.assertEqual(w.offset, [.04, 0, .1])

    def test_grasp_lift_and_connected_target_stay_fixed(self):
        chain = self.original
        for i, w in enumerate(chain.waypoints[:3]):
            target = self.s.editor_target(chain, i, w)[0].copy()
            result = self.s.test_waypoint(chain, i, w, lambda *_: None)
            self.assertTrue(result.startswith("Finished"), result)
            self.assertLess(np.linalg.norm(self.s.world.eef()-target), .003)
        before = self.s.env.sim.get_state().flatten().copy()
        for _ in range(3):
            np.testing.assert_allclose(self.s.editor_target(chain, 2, w)[0], target)
            self.s.test_waypoint(chain, 2, w, lambda *_: None)
        np.testing.assert_array_equal(self.s.env.sim.get_state().flatten(), before)
        # Subsequent targets do not get pulled along with the now-carried bowl.
        retreat = self.s.editor_target(chain, 5, chain.waypoints[5])[0]
        lower = self.s.editor_target(chain, 4, chain.waypoints[4])[0]
        np.testing.assert_allclose(retreat, lower+[0, 0, .15])

    def test_smooth_full_task_replay(self):
        positions = [self.s.executor.scene.arm_qpos()]
        self.s.executor.on_step = lambda: positions.append(self.s.executor.scene.arm_qpos())
        result = self.s.run(self.original, lambda *_: None)
        self.assertTrue(result.startswith("Finished"), result)
        self.assertTrue(self.s.env.check_success())
        dt = self.s.env.env.control_timestep
        velocities = np.diff(positions, axis=0)/dt
        accelerations = np.diff(velocities, axis=0)/dt
        # The old stop/go trajectory measured 1.72 peak and .204 RMS rad/s².
        self.assertLess(np.max(np.abs(velocities)), .2)
        self.assertLess(np.max(np.abs(accelerations)), 1.)
        self.assertLess(np.sqrt(np.mean(accelerations**2)), .09)

    def test_short_moves_limit_overshoot_and_hold_jitter(self):
        for offset in ([0, 0, -.06], [.05, 0, 0], [-.05, 0, .06]):
            start, orientation = self.s.gripper_pose()
            target = start + offset
            positions = []
            self.s.executor.on_step = lambda: positions.append(self.s.world.eef())
            result = self.s.executor.line(target, orientation)
            self.assertTrue(result.startswith("ok"), result)
            reached = len(positions)
            self.s.executor.hold(40)
            points = np.array(positions)
            direction = np.array(offset) / np.linalg.norm(offset)
            self.assertLess(np.max((points-target) @ direction), .005)
            self.assertLess(np.linalg.norm(points[-1]-target), .0015)
            self.assertLess(np.max(np.ptp(points[reached:], axis=0)), .001)

    def test_unsettled_move_reports_failure(self):
        position, orientation = self.s.gripper_pose()
        with patch.object(self.s.executor, "_settle", return_value=False):
            result = self.s.executor.line(position + [0, 0, -.01], orientation)
        self.assertTrue(result.startswith("failed: robot did not settle"), result)

    def test_gripper_on_arrival_waits_for_contact_pose(self):
        position, ori = self.s.gripper_pose()
        w = Waypoint(reference=PREVIOUS, offset=[0, 0, -.03], orientation=ori.tolist(),
                     motion="line", gripper="open", gripper_after="close")
        chain = dataclasses.replace(self.original, waypoints=[w])
        closing_positions = []
        def sample():
            if self.s.executor.gripper == sp.GRIPPER_CLOSE:
                closing_positions.append(self.s.world.eef().copy())
        self.s.executor.on_step = sample
        result = self.s.test_waypoint(chain, 0, w, lambda *_: None)
        self.assertTrue(result.startswith("Finished"), result)
        self.assertTrue(closing_positions)
        self.assertLess(np.linalg.norm(closing_positions[0]-(position+[0, 0, -.03])), .002)
        state = self.s.env.sim.get_state().flatten().copy()
        self.s.test_waypoint(chain, 0, w, lambda *_: None)
        np.testing.assert_array_equal(self.s.env.sim.get_state().flatten(), state)

    def test_release_with_pose_motion_opens_fingers_without_moving_the_arm(self):
        position, rotation = self.s.gripper_pose()
        self.s.executor.gripper = sp.GRIPPER_CLOSE
        waypoint = Waypoint("Release", PREVIOUS, [0, 0, 0], orientation=np.eye(3).tolist(),
                            orientation_relative_to_previous=True, motion="line", gripper="open")
        chain = dataclasses.replace(self.original, waypoints=[waypoint])
        result = self.s.test_waypoint(chain, 0, waypoint, lambda *_: None)
        self.assertTrue(result.startswith("Finished"), result)
        self.assertEqual(self.s.executor.gripper, sp.GRIPPER_OPEN)
        self.assertLess(np.linalg.norm(self.s.world.eef()-position), .002)

    def test_learning_runs_twice_and_archives_both(self):
        from test_waypoint_library import DemonstrationTest
        first, second = DemonstrationTest().phased_demo(), DemonstrationTest().phased_demo()
        first["success"] = False
        recordings = iter([first, second])
        def record(session, *args, **kwargs):
            session.last_demonstration = next(recordings)
        with tempfile.TemporaryDirectory() as folder:
            with patch("waypoint_demo.record", side_effect=record) as run:
                chain = self.s.record_policy(self.original, archive_dir=folder)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[2], self.s.args.policy_port)
            self.assertTrue((self.s.last_learning_archive / "rollout_1.json").exists())
            self.assertTrue((self.s.last_learning_archive / "rollout_2.json").exists())
            self.assertEqual(len(chain.waypoints), 6)
            self.assertIn("selected demonstration 2/2", chain.source)

    def test_learning_uses_requested_runs_and_waypoint_count(self):
        from test_waypoint_library import DemonstrationTest
        demo = DemonstrationTest().phased_demo()
        def record(session, *args, **kwargs):
            session.last_demonstration = demo
        with tempfile.TemporaryDirectory() as folder:
            with patch("waypoint_demo.record", side_effect=record) as run:
                chain = self.s.record_policy(self.original, archive_dir=folder,
                                             repeats=3, waypoint_count=8)
            self.assertEqual(run.call_count, 3)
            self.assertEqual(len(chain.waypoints), 8)
            self.assertEqual(len(list(self.s.last_learning_archive.glob("rollout_*.json"))), 3)


class MotionResultTest(unittest.TestCase):
    def test_a_missed_target_preserves_the_original_planner_failure(self):
        session = SimpleNamespace(injected_failure="", args=SimpleNamespace(reach_tolerance=.03),
                                  world=SimpleNamespace(eef=lambda: np.zeros(3)))
        result = Session.check_motion_result(session, Waypoint(), "no collision-free path",
                                             np.array([.189, 0, 0]), np.eye(3))
        self.assertEqual(result, "no collision-free path; target missed by 18.9 cm")
        result = Session.check_motion_result(session, Waypoint(), "ok",
                                             np.array([.189, 0, 0]), np.eye(3))
        self.assertEqual(result, "failed: target missed by 18.9 cm")


if __name__ == "__main__":
    unittest.main()
