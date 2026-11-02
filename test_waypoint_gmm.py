"""Editable GMM chain migration and execution regressions."""
import dataclasses
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from scipy.spatial.transform import Rotation
import numpy as np
from waypoint_chain import Chain, Waypoint
from waypoint_gmm import GMMChain, chain_from_model, check_learned_pose
from waypoint_learning import load_model

MODEL = Path(__file__).parent/'waypoint_models/cached_goal0_smoke/model.json'


class GMMChainTest(unittest.TestCase):
    def test_model_conversion_keeps_adaptive_count_and_uncertainty(self):
        model = load_model(MODEL)
        chain = chain_from_model(model, MODEL)
        self.assertEqual(len(chain.waypoints), 15)
        for node, waypoint in zip(model['waypoints'], chain.waypoints):
            np.testing.assert_allclose(waypoint.offset, node['relative_position_mean'])
            self.assertEqual(waypoint.learned['position_covariance'], node['position_covariance'])
            self.assertEqual(waypoint.learned['reference_policy'], node['reference']['policy'])

    def test_edits_round_trip_without_legacy_limit_change(self):
        chain = chain_from_model(load_model(MODEL), MODEL)
        chain.waypoints[3].offset[0] += .01
        chain.waypoints.append(Waypoint('Manually added'))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'gmm.json'
            chain.save(path)
            loaded = GMMChain.load(path, chain.suite, chain.task_id)
            self.assertEqual(len(loaded.waypoints), 16)
            self.assertEqual(loaded.waypoints[3].learned, chain.waypoints[3].learned)
            self.assertEqual(loaded.waypoints[3].offset, chain.waypoints[3].offset)
            self.assertEqual(loaded.waypoints[-1].learned, {})
            with self.assertRaises(ValueError): Chain.load(path, chain.suite, chain.task_id)
            legacy = Chain(chain.suite, chain.task_id, chain.language, [Waypoint()]*10)
            with self.assertRaisesRegex(ValueError, 'at most 9'): legacy.save(Path(folder)/'legacy.json')

    def test_bad_covariance_rejected(self):
        chain = chain_from_model(load_model(MODEL), MODEL)
        chain.waypoints[0].learned['position_covariance'] = (-np.eye(3)).tolist()
        with self.assertRaisesRegex(ValueError, 'positive semidefinite'): chain.validate()


class LearnedArrivalTest(unittest.TestCase):
    def session(self, gaps, angles=None):
        angles = angles or [0.]*len(gaps)
        poses = [(np.array([gap, 0., 0.]), Rotation.from_rotvec([0., 0., angle]).as_matrix())
                 for gap, angle in zip(gaps, angles)]
        session = SimpleNamespace(pose=poses[0])
        remaining = iter(poses[1:])
        def line(*args):
            session.pose = next(remaining)
            return 'ok'
        session.gripper_pose = lambda: session.pose
        session.executor = SimpleNamespace(line=Mock(side_effect=line), hold=Mock())
        return session

    def check(self, session, result='ok', **kwargs):
        return check_learned_pose(session, {'position_tolerance': .002, 'orientation_tolerance': .02},
                                  result, np.zeros(3), np.eye(3), **kwargs)

    def test_small_residual_corrected_without_relaxing_tolerance(self):
        session = self.session([.0022, .0018])
        self.assertIn('1 correction', self.check(session))
        session.executor.line.assert_called_once()
        session.executor.hold.assert_called_once_with(5)

    def test_uncorrectable_miss_stops_after_two_attempts(self):
        session = self.session([.0022, .0021, .0021])
        result = self.check(session)
        self.assertTrue(result.startswith('failed: learned tolerance exceeded'))
        self.assertIn('2.00 mm limit', result)
        self.assertIn('2 corrections', result)
        self.assertEqual(session.executor.line.call_count, 2)

    def test_large_errors_and_executor_failures_are_not_retried(self):
        for gap, angle, result in ((.02, 0., 'ok'), (0., .2, 'ok'),
                                   (.0022, 0., 'unreachable: Cartesian path misses')):
            session = self.session([gap], [angle])
            self.assertFalse(self.check(session, result).startswith('ok'))
            session.executor.line.assert_not_called()

    def test_orientation_miss_is_corrected_too(self):
        session = self.session([0., 0.], [.03, .015])
        self.assertTrue(self.check(session).startswith('ok'))
        session.executor.line.assert_called_once()

    def test_correction_must_settle_inside_original_bounds(self):
        session = self.session([.0022, .0018])
        session.executor.hold.side_effect = lambda *_: setattr(session, 'pose', (np.array([.0021, 0., 0.]), np.eye(3)))
        self.assertTrue(self.check(session).startswith('failed: learned tolerance exceeded'))

    def test_reach_does_not_retry_with_unchecked_line(self):
        session = self.session([.0022])
        self.assertTrue(self.check(session, retry_line=False).startswith('failed'))
        session.executor.line.assert_not_called()

    def test_continuing_playback_reports_pose_miss_without_correction(self):
        session = self.session([.0022])
        session.continue_on_failure = True
        self.assertTrue(self.check(session).startswith('failed: learned tolerance exceeded'))
        session.executor.line.assert_not_called()
        session.executor.hold.assert_not_called()

    def test_continuing_playback_does_not_extend_expired_settling_grace(self):
        session = self.session([.0018])
        session.continue_on_failure = True
        result = self.check(session, result='failed: robot did not settle at the target')
        self.assertEqual(result, 'failed: robot did not settle at the target')
        session.executor.line.assert_not_called()
        session.executor.hold.assert_not_called()


if __name__ == '__main__': unittest.main()
