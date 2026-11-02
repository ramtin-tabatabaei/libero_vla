"""Physical retention and feedback regressions for Long Tasks 10, Task 01."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

import subgoal_planner as sp
from waypoint_chain import Chain, Waypoint, WORLD
from waypoint_failures import Injector, held_object
from waypoint_gmm import GMMChain
from waypoint_gmm_editor import GMMSession
from waypoint_session import Session


class GraspFeedbackTest(unittest.TestCase):
    def session(self):
        s = Session(sp.Args())
        s.continue_on_failure = True
        s.poses = lambda: {}
        s.gripper_pose = lambda: (np.zeros(3), np.eye(3))
        s.update_run_frames = lambda *_: None
        s.env = SimpleNamespace(check_success=lambda: False)
        domain = SimpleNamespace(
            objects_dict={'can': SimpleNamespace(contact_geoms=['can_geom'])},
            robots=[SimpleNamespace(gripper='fingers')],
            _check_grasp=Mock(return_value=False))
        s.world = SimpleNamespace(domain=domain, eef=lambda: np.zeros(3))
        s.executor = SimpleNamespace(
            holding='', gripper=sp.GRIPPER_OPEN, _set_ignore=Mock(),
            line=Mock(return_value='ok'), run_subgoal=Mock(return_value='ok'))
        chain = Chain('libero_10', 0, 'test', [
            Waypoint('Grasp', WORLD, motion='line', gripper_after='close', grasp='can'),
            Waypoint('Lift', WORLD, motion='line')])
        return s, chain

    def test_failed_close_on_arrival_is_reported_and_playback_continues(self):
        s, chain = self.session()
        def grip(sg):
            if sg.check_grasp and not s.world.domain._check_grasp('fingers', ['can_geom']):
                return 'failed: nothing between the fingers'
            return 'ok'
        s.executor.run_subgoal.side_effect = grip
        result = s.run(chain, lambda *_: None)
        self.assertIn('Finished 2 waypoints', result)
        self.assertEqual(s.last_run_failures, [dict(row=0, result='failed: nothing between the fingers')])
        self.assertEqual(s.executor.line.call_count, 2)

    def test_lost_object_is_detected_even_when_wrist_reaches_target(self):
        s, chain = self.session()
        s.executor.holding = 'can'
        chain.waypoints[0].gripper_after = 'keep'
        reports = []
        result = s.run(chain, lambda *r: reports.append(r))
        self.assertIn('Finished 2 waypoints', result)
        self.assertIn('lost grasp on can', reports[-1][1])
        self.assertEqual([f['row'] for f in s.last_run_failures], [0, 1])
        self.assertEqual(s.executor.line.call_count, 2)

    def test_secure_grasp_passes_the_same_motion_check(self):
        s, chain = self.session()
        s.executor.holding = 'can'
        s.world.domain._check_grasp.return_value = True
        result = s.run(chain, lambda *_: None)
        self.assertIn('Finished 2 waypoints', result)
        self.assertEqual(s.last_run_failures, [])

    def test_release_does_not_require_a_grasp(self):
        s, chain = self.session()
        for before, after, motion in [('open', 'keep', 'line'),
                                      ('keep', 'open', 'line'), ('open', 'keep', 'grip')]:
            with self.subTest(before=before, after=after, motion=motion):
                chain.waypoints[0].gripper = before
                chain.waypoints[0].gripper_after = after
                chain.waypoints[0].motion = motion
                s.executor.run_subgoal.reset_mock()
                s.run(chain, lambda *_: None)
                self.assertEqual(s.executor.run_subgoal.call_args[0][0].check_grasp, '')


class CanTaskSimulatorTest(unittest.TestCase):
    def test_both_cans_stay_held_until_release_across_scene_seeds(self):
        path = Path(__file__).parent / 'task_waypoints_gmm/libero_10/task_000.json'
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                s = GMMSession(sp.Args(task_suite_name='libero_10', task_id=0, seed=seed))
                try:
                    s.load('libero_10', 0, cameras=False)
                    s.continue_on_failure = True
                    chain = GMMChain.load(path, 'libero_10', 0)
                    carrying = {row: name for rows, name in (
                        ((2, 3, 4), 'alphabet_soup_1'), ((9, 10, 11), 'tomato_sauce_1'))
                        for row in rows}
                    expected = ['']
                    largest_gap = [0.]
                    visited = []
                    def tick():
                        if expected[0]:
                            gap = np.linalg.norm(s.world.pos(expected[0])-s.world.eef())
                            largest_gap[0] = max(largest_gap[0], float(gap))
                    s.tick = tick
                    def report(row, result):
                        if result == 'Running':
                            injector.start_row(row)
                            expected[0] = carrying.get(row, '')
                            return
                        visited.append(row)
                        self.assertTrue(result.startswith('ok'), (seed, row+1, result))
                        name = carrying.get(row) or {1: 'alphabet_soup_1', 8: 'tomato_sauce_1'}.get(row)
                        if name:
                            self.assertEqual(held_object(s.world), name, (seed, row+1))
                        if row in (2, 9):
                            self.assertGreater(s.world.pos(name)[2], .65)
                    with Injector(s, {}) as injector:
                        result = s.run(chain, report)
                    self.assertEqual(injector.fired, [])
                    self.assertEqual(visited, list(range(13)))
                    self.assertEqual(s.last_run_failures, [])
                    self.assertLess(largest_gap[0], .06)
                    s.executor.hold(60)
                    self.assertTrue(s.env.check_success(), (seed, result))
                    print('Can task seed %d: both cans delivered; max grip offset %.1f mm' %
                          (seed, largest_gap[0]*1000), flush=True)
                finally:
                    if s.env is not None:
                        s.env.close()


if __name__ == '__main__':
    unittest.main()
