"""Regression tests for chain geometry and task-scoped persistence."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from waypoint_chain import Chain, Waypoint, PREVIOUS, resolve, relative_waypoint


class ChainTest(unittest.TestCase):
    def test_rotated_object_and_connected_lift(self):
        basis = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
        poses = {"cup": (np.array([1., 2., 3.]), basis)}
        start = (np.zeros(3), np.eye(3))
        target = resolve(Waypoint(reference="cup", offset=[.1, 0, .2]), poses, start)
        np.testing.assert_allclose(target[0], [1., 2.1, 3.2])
        lift = resolve(Waypoint(reference=PREVIOUS, offset=[0, 0, .15]), poses, target)
        np.testing.assert_allclose(lift[0], [1., 2.1, 3.35])
        poses["cup"] = (np.array([2., 3., 4.]), basis)
        moved = resolve(Waypoint(reference="cup", offset=[.1, 0, .2]), poses, start)
        np.testing.assert_allclose(moved[0], target[0] + 1)

    def test_capture_round_trip(self):
        basis = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        poses = {"cup": (np.array([.2, .3, .4]), basis)}
        position, orientation = np.array([.5, -.2, .8]), np.eye(3)
        w = relative_waypoint("Capture", position, orientation, "cup", poses, (position, orientation))
        actual = resolve(w, poses, (position, orientation))
        np.testing.assert_allclose(actual[0], position)
        np.testing.assert_allclose(actual[1], orientation)

    def test_previous_waypoint_orientation_tracks_preceding_wrist(self):
        preceding = Rotation.from_euler('xyz', [.3, -.2, .7]).as_matrix()
        desired = preceding @ Rotation.from_euler('z', .4).as_matrix()
        previous = (np.array([.1, .2, .3]), preceding)
        waypoint = relative_waypoint('Release', np.array([.1, .2, .3]), desired,
                                     PREVIOUS, {}, previous)
        self.assertTrue(waypoint.orientation_relative_to_previous)
        np.testing.assert_allclose(resolve(waypoint, {}, previous)[1], desired)
        turned = Rotation.from_euler('y', -.6).as_matrix()
        changed_previous = (previous[0], turned @ preceding)
        np.testing.assert_allclose(resolve(waypoint, {}, changed_previous)[1],
                                   turned @ desired)
        legacy = Waypoint(reference=PREVIOUS, orientation=desired.tolist())
        np.testing.assert_allclose(resolve(legacy, {}, changed_previous)[1], desired)

    def test_save_load_and_task_mismatch(self):
        chain = Chain("libero_goal", 3, "task", [Waypoint(reference="cup")])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"task.json"
            chain.save(path)
            self.assertEqual(Chain.load(path, "libero_goal", 3), chain)
            with self.assertRaises(ValueError):
                Chain.load(path, "libero_goal", 2)

    def test_invalid_targets(self):
        for offset in ([1, 2], [0, float('nan'), 0], [0, 0, float('inf')]):
            with self.assertRaises(ValueError):
                Waypoint(offset=offset).validate()
        with self.assertRaises(ValueError):
            Waypoint(orientation=np.zeros((3,3)).tolist()).validate()
        with self.assertRaises(ValueError):
            resolve(Waypoint(reference="missing"), {}, (np.zeros(3), np.eye(3)))


if __name__ == '__main__':
    unittest.main()
