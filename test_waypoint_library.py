"""Geometry and event-preservation tests for pi0.5 keyframe extraction."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from waypoint_chain import Chain, Waypoint, resolve
from waypoint_library import demonstration_plan, phase_plan, choose_demonstration


class DemonstrationTest(unittest.TestCase):
    def phased_demo(self):
        demo = self.demo()
        keyframes = [[0, -.15, .3], [0, 0, .2], [0, 0, .1], [0, 0, .3],
                     [.2, 0, .3], [.2, 0, .15], [.2, 0, .1]]
        points = np.array([np.interp(np.arange(61), np.arange(0, 61, 10), np.array(keyframes)[:, j])
                           for j in range(3)]).T
        demo['samples'] = [dict(position=p.tolist(), orientation=np.eye(3).tolist(),
                               gripper="open" if i < 17 or i >= 60 else "close",
                               grasped=["bowl"] if 20 <= i < 60 else []) for i, p in enumerate(points)]
        demo['end_position'] = points[-1].tolist()
        return demo

    def test_pick_place_has_six_distinct_phases(self):
        chain = phase_plan(self.phased_demo())
        self.assertEqual([w.name for w in chain.waypoints],
                         ["Approach", "Reach & close gripper", "Lift", "Transfer", "Going down", "Finish & open gripper"])
        self.assertEqual(chain.waypoints[1].gripper_after, "close")
        self.assertEqual(chain.waypoints[-1].gripper_after, "open")

    def test_selection_prefers_successful_second_demo(self):
        first, second = self.phased_demo(), self.phased_demo()
        first['success'] = False
        chain, selected = choose_demonstration([first, second])
        self.assertEqual(selected, 1)
        self.assertIn("selected demonstration 2/2", chain.source)
        self.assertIn("replay unverified", chain.source)

    def test_requested_count_preserves_actions_and_measured_final_pose(self):
        demo = self.phased_demo()
        for count in (3, 6, 9):
            chain, _ = choose_demonstration([demo], waypoint_count=count)
            self.assertEqual(len(chain.waypoints), count)
            self.assertEqual([w.gripper_after for w in chain.waypoints
                              if w.gripper_after != "keep"], ["close", "open"])
            poses = {n: (np.array(p), np.array(r)) for n, (p, r) in demo["poses"].items()}
            previous = (np.array(demo["samples"][0]["position"]), np.eye(3))
            for waypoint in chain.waypoints:
                previous = resolve(waypoint, poses, previous)
            np.testing.assert_allclose(previous[0], demo["end_position"])
        for success in (True, False):
            demo["success"] = success
            with self.assertRaisesRegex(ValueError, "Increase the waypoint count"):
                choose_demonstration([demo], waypoint_count=1)

    def test_exact_count_fills_straight_path_and_rejects_insufficient_samples(self):
        demo = self.demo()
        for sample in demo["samples"]:
            sample.update(position=[0, 0, 0], gripper="open", grasped=[])
        demo["end_position"] = [0, 0, 0]
        self.assertEqual(len(choose_demonstration([demo], waypoint_count=9)[0].waypoints), 9)
        demo["samples"] = demo["samples"][:2]
        with self.assertRaisesRegex(ValueError, "Not enough measured samples"):
            choose_demonstration([demo], waypoint_count=9)

    def demo(self):
        basis = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        poses = {"bowl": [[0, 0, 0], basis.tolist()], "plate": [[.2, 0, 0], np.eye(3).tolist()]}
        samples = []
        for i in range(80):
            samples.append(dict(position=[i/400, .04*np.sin(i/8), .15+abs(i-30)/200],
                                orientation=np.eye(3).tolist(), gripper="close" if 20 <= i < 65 else "open",
                                grasped=["bowl"] if 30 <= i < 65 else []))
        return dict(suite="libero_spatial", task_id=0, language="test", samples=samples, poses=poses,
                    objects=["bowl", "plate"], fixtures=[], end_position=[.2, 0, .15],
                    end_orientation=np.eye(3).tolist(), success=True)

    def test_contact_events_and_final_pose_survive_nine_point_limit(self):
        demo = self.demo()
        chain = demonstration_plan(demo)
        self.assertLessEqual(len(chain.waypoints), 9)
        self.assertGreater(len(chain.waypoints), 2)
        self.assertIn("pi0.5", chain.source)
        poses = {n: (np.array(p), np.array(r)) for n, (p, r) in demo["poses"].items()}
        previous = (np.zeros(3), np.eye(3))
        transitions = []
        measured = np.array([s["position"] for s in demo["samples"]]+[demo["end_position"]])
        for w in chain.waypoints:
            previous = resolve(w, poses, previous)
            self.assertLess(np.min(np.linalg.norm(measured-previous[0], axis=1)), 1e-9)
            if w.gripper_after != "keep":
                transitions.append(w.gripper_after)
                expected = demo["samples"][30 if w.gripper_after == "close" else 65]["position"]
                np.testing.assert_allclose(previous[0], expected)
        self.assertEqual(transitions, ["close", "open"])
        np.testing.assert_allclose(previous[0], demo["end_position"])

    def test_refuses_to_silently_drop_excess_gripper_events(self):
        demo = self.demo()
        for i, s in enumerate(demo["samples"]):
            s["gripper"] = "open" if (i//8)%2 else "close"
            s["grasped"] = []
        with self.assertRaisesRegex(ValueError, "cannot preserve"):
            demonstration_plan(demo)

    def test_short_gripper_pulses_do_not_spend_waypoint_budget(self):
        demo = self.demo()
        for i in (45, 50, 55):
            demo["samples"][i]["gripper"] = "open"
        chain = demonstration_plan(demo)
        self.assertEqual([w.gripper_after for w in chain.waypoints if w.gripper_after != "keep"], ["close", "open"])

    def test_failed_retries_are_explicitly_labelled_in_limited_draft(self):
        demo = self.demo()
        demo["success"] = False
        for i, s in enumerate(demo["samples"]):
            s["gripper"] = "open" if (i//8)%2 else "close"
            s["grasped"] = []
        chain = demonstration_plan(demo)
        self.assertLessEqual(len(chain.waypoints), 9)
        self.assertIn("limited draft", chain.source)
        self.assertIn("rollout success=False", chain.source)

    def test_save_enforces_limit_and_preserves_on_arrival_action(self):
        chain = Chain("libero_spatial", 0, "test", [Waypoint(gripper_after="close")]*9)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"chain.json"
            chain.save(path)
            saved = path.read_text()
            self.assertEqual(Chain.load(path, chain.suite, 0).waypoints[0].gripper_after, "close")
            chain.waypoints.append(Waypoint())
            with self.assertRaisesRegex(ValueError, "at most 9"):
                chain.save(path)
            self.assertEqual(path.read_text(), saved)


if __name__ == "__main__":
    unittest.main()
