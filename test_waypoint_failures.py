"""Injected failures: what is armed in the studio really happens in the sim.

A run recorded for a failure detector is only worth as much as the fault in it,
so these checks read the robot's own state after the run -- where the wrist
stopped, how it was turned, what it touched, what it is still holding -- and
not the message the studio printed about it.

Run with the libero Python inside xvfb-run.
"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

import subgoal_gui as gui  # Configure the smooth Tk runtime before importing Tk.
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy.spatial.transform import Rotation

import subgoal_planner as sp
from waypoint_chain import Chain, Waypoint, WORLD
from waypoint_session import Session
from waypoint_failures import (FAILURES, HINTS, UNSETTLED, HIGH_COLLISION_OFFSET, Injector,
                               TILT_AXES, TILT_BANDS, TRANSITION_BANDS, allowed_intensities, collision_target,
                               held_object, split_path, tilt, waypoint_primitive)

FIXTURE = Path(__file__).parent / "tests/fixtures/manual_bowl_chain.json"


class Bowl:
    """The smallest world a collision target can be chosen in."""
    class domain:
        objects_dict = {"bowl": None, "cup": None}
        fixtures_dict = {}

    def aabb(self, name):
        centre = np.array([.3, 0., .85]) if name == "bowl" else np.array([.3, .5, .85])
        return centre - [.05, .05, .03], centre + [.05, .05, .03]


class FailureGeometryTest(unittest.TestCase):
    def test_every_failure_is_named_and_explained(self):
        self.assertEqual(set(FAILURES), set(HINTS))
        self.assertEqual(list(FAILURES), ["collision", "transition",
                                          "orientation", "grasp_loss"])

    def test_collision_aims_past_the_middle_of_the_nearest_body(self):
        name, target = collision_target(Bowl(), [0, 0, 1.], [.6, 0, 1.])
        self.assertEqual(name, "bowl")
        # Level with the bowl's middle and past it, not stopped at its face:
        # open fingers descending from above would straddle it and touch nothing.
        np.testing.assert_allclose(target, [.33, 0., .85], atol=1e-9)

    def test_collision_skips_what_the_gripper_is_holding(self):
        name, _ = collision_target(Bowl(), [0, 0, 1.], [.6, 0, 1.], ignore=("bowl",))
        self.assertEqual(name, "cup")

    def test_collision_falls_back_to_the_surface_below_an_empty_scene(self):
        class Empty(Bowl):
            class domain:
                objects_dict, fixtures_dict = {}, {}
        name, target = collision_target(Empty(), [0, 0, 1.], [.6, 0, 1.])
        self.assertIsNone(name)
        np.testing.assert_allclose(target, [.6, 0, .88])

    def test_tilt_turns_the_wrist_about_its_own_axis(self):
        for index, axis in enumerate(TILT_AXES):
            with self.subTest(axis=axis):
                turned = tilt(np.eye(3), np.radians(35), axis)
                angle = np.arccos(np.clip((np.trace(turned) - 1) / 2, -1, 1))
                self.assertAlmostEqual(np.degrees(angle), 35, places=6)
                # The drawn axis is the one the wrist keeps.
                np.testing.assert_allclose(turned[:, index], np.eye(3)[index], atol=1e-9)

    def test_tilt_combines_roll_pitch_and_yaw_in_the_wrist_frame(self):
        original = Rotation.from_euler('xyz', [80, -25, 15], degrees=True).as_matrix()
        turned = tilt(original, np.radians([25, 35, 40]), 'xyz')
        measured = Rotation.from_matrix(original.T @ turned).as_euler('xyz', degrees=True)
        np.testing.assert_allclose(measured, [25, 35, 40], atol=1e-9)

    def test_two_point_route_is_split_midway_not_at_the_destination(self):
        first, rest = split_path([[0., 0.], [2., 0.]])
        np.testing.assert_allclose(first, [[0., 0.], [1., 0.]])
        np.testing.assert_allclose(rest, [[1., 0.], [2., 0.]])

    def test_route_split_preserves_all_edges_on_an_uneven_path(self):
        first, rest = split_path([[0., 0.], [.1, 0.], [.1, 2.]])
        np.testing.assert_allclose(first, [[0., 0.], [.1, 0.], [.1, .95]])
        np.testing.assert_allclose(rest, [[.1, .95], [.1, 2.]])


class InjectorArmingTest(unittest.TestCase):
    def injector(self, injections, **kw):
        session = type("Session", (), {})()
        session.executor, session.injected_failure = None, ""
        return Injector(session, injections, **kw)

    def test_a_delivered_fault_turns_off_that_row_s_arrival_check(self):
        injector = self.injector({0: "transition"})
        injector.start_row(0)
        for landed in ("ok (3.0cm)", UNSETTLED):
            with self.subTest(landed=landed):
                self.assertEqual(injector._delivered("transition", landed),
                                 "ok (injected transition deviation)")
                self.assertEqual(injector.session.injected_failure,
                                 "ok (injected transition deviation)")

    def test_a_move_that_never_happened_is_still_the_executor_s_news(self):
        injector = self.injector({0: "transition"})
        injector.start_row(0)
        missed = "unreachable: Cartesian path misses by 7.0 cm"
        self.assertEqual(injector._delivered("transition", missed), missed)
        self.assertEqual(injector.session.injected_failure, "")

    def test_the_next_row_is_checked_normally_again(self):
        injector = self.injector({0: "collision"})
        injector.start_row(0)
        injector._delivered("collision", "ok")
        injector.start_row(1)
        self.assertEqual(injector.session.injected_failure, "")

    def test_severity_and_failure_names_are_checked_before_the_robot_moves(self):
        for injections, kw, message in (
                ({0: "typo"}, {}, "Unknown injected failure: typo"),
                ({}, dict(offset=2.), "Distance off target"),
                ({}, dict(offset=.001), "Distance off target"),
                ({}, dict(angle=np.radians(120)), "Wrist rotation"),
                ({}, dict(angle=np.radians(1)), "Wrist rotation")):
            with self.subTest(message=message):
                with self.assertRaises(ValueError) as caught:
                    self.injector(injections, **kw)
                self.assertIn(message, str(caught.exception))

    def test_a_row_fires_once_however_many_moves_it_makes(self):
        injector = self.injector({1: "collision", 3: "grasp_loss"})
        # A guarded move may probe, plan, then fall back to a straight line;
        # one arming is one fault, not one per attempt.
        injector.start_row(1)
        self.assertEqual([injector.take(), injector.take()], ["collision", None])
        injector.start_row(2)
        self.assertIsNone(injector.take())
        injector.start_row(3)
        self.assertEqual(injector.take(), "grasp_loss")

    def test_patched_methods_are_put_back_even_when_the_run_raises(self):
        session = type("Session", (), {})()
        session.executor = sp.Executor.__new__(sp.Executor)
        injector = Injector(session, {})
        with self.assertRaises(sp.Abort):
            with injector:
                self.assertNotEqual(session.executor.line, sp.Executor.line)
                raise sp.Abort()
        for name in Injector.METHODS:
            self.assertNotIn(name, vars(session.executor))


class CollisionDetourTest(unittest.TestCase):
    def test_unreachable_collision_still_attempts_the_original_waypoint(self):
        line = Mock(side_effect=["unreachable: collision target", "ok"])
        session = SimpleNamespace(world=SimpleNamespace(eef=lambda: np.zeros(3)),
                                  executor=None, injected_failure="")
        injector = Injector(session, {0: "collision"})
        injector.saved = dict(line=line, reach=Mock())
        injector.start_row(0)
        goal, orientation = np.array([.2, .1, 1.]), np.eye(3)
        with patch("waypoint_failures.collision_target", return_value=("bowl", np.ones(3))), \
                patch("waypoint_failures.held_object", return_value=""):
            result = injector._line(goal, orientation)
        self.assertEqual(result, "ok")
        np.testing.assert_allclose(line.call_args_list[-1].args[0], goal)
        self.assertEqual(injector.fired, [])
        self.assertEqual(len(injector.missed), 1)
        self.assertEqual(session.injected_failure, "")

    def test_arrival_failure_after_the_collision_is_not_masked(self):
        line = Mock(side_effect=[UNSETTLED, UNSETTLED, "unreachable: original waypoint"])
        session = SimpleNamespace(world=SimpleNamespace(eef=lambda: np.zeros(3)),
                                  executor=None, injected_failure="",
                                  gripper_pose=lambda: (np.ones(3), sp.DOWN))
        injector = Injector(session, {0: "collision"})
        injector.saved = dict(line=line, reach=Mock())
        injector.start_row(0)
        with patch("waypoint_failures.collision_target", return_value=("bowl", np.ones(3))), \
                patch("waypoint_failures.held_object", return_value=""):
            result = injector._line(np.zeros(3), np.eye(3))
        self.assertEqual(result, "unreachable: original waypoint")
        self.assertEqual(len(injector.fired), 1)
        self.assertEqual(session.injected_failure, "")


class FailureIntensityTest(unittest.TestCase):
    def injector(self, kind, intensity, primitive=None):
        executor = SimpleNamespace(gripper=sp.GRIPPER_CLOSE, holding="",
                                   hold=Mock(), carry_offset=np.ones(3), carry_bottom=.1)
        session = SimpleNamespace(executor=executor, injected_failure="",
                                  world=SimpleNamespace(eef=lambda: np.zeros(3)),
                                  gripper_pose=lambda: (np.ones(3), sp.DOWN))
        injector = Injector(session, {0: dict(kind=kind, intensity=intensity)})
        injector.start_row(0, primitive=primitive)
        return injector

    def test_invalid_intensity_is_rejected_before_playback(self):
        with self.assertRaisesRegex(ValueError, "Unknown failure intensity"):
            self.injector("collision", "medium")

    def test_every_fault_is_blocked_in_move_even_when_armed_programmatically(self):
        for kind in FAILURES:
            for intensity in ("low", "high"):
                with self.subTest(kind=kind, intensity=intensity):
                    injector = self.injector(kind, intensity, primitive="move")
                    original = Mock(return_value="ok")
                    injector.saved["line"] = original
                    self.assertEqual(injector._line(np.ones(3), np.eye(3)), "ok")
                    np.testing.assert_array_equal(original.call_args.args[0], np.ones(3))
                    self.assertEqual(injector.fired, [])
                    self.assertIn("not allowed in Move", injector.missed[-1])

    def test_pose_faults_are_blocked_in_transport_even_when_armed_programmatically(self):
        for kind in ("collision", "transition", "orientation"):
            for intensity in ("low", "high"):
                with self.subTest(kind=kind, intensity=intensity):
                    injector = self.injector(kind, intensity, primitive="transport")
                    original = Mock(return_value="ok")
                    injector.saved["line"] = original
                    self.assertEqual(injector._line(np.ones(3), np.eye(3)), "ok")
                    original.assert_called_once()
                    np.testing.assert_array_equal(original.call_args.args[0], np.ones(3))
                    np.testing.assert_array_equal(original.call_args.args[1], np.eye(3))
                    self.assertEqual(injector.fired, [])
                    self.assertIn("not allowed in Transport", injector.missed[-1])

    def test_gripper_only_release_has_no_available_failure(self):
        for kind in FAILURES:
            self.assertEqual(allowed_intensities(kind, "release", "grip"), ())
        self.assertEqual(allowed_intensities("grasp_loss", "grasp", "grip"), ("low",))
        self.assertEqual(allowed_intensities("grasp_loss", "transport", "grip"), ())

    def test_waypoint_motion_rules_are_enforced_during_playback(self):
        session = SimpleNamespace(executor=None, injected_failure="")
        injector = Injector(session, {0: dict(kind="orientation", intensity="high")},
                            waypoints=[Waypoint("Release", motion="grip", gripper="open")])
        injector.start_row(0)
        self.assertIsNone(injector.armed)
        self.assertIn("not allowed in Release", injector.missed[0])

    def test_grasp_loss_rules_match_acquisition_and_transport(self):
        self.assertEqual(allowed_intensities("grasp_loss", "grasp"), ("low",))
        self.assertEqual(allowed_intensities("grasp_loss", "transport"), ("high",))
        for phase in ("move", "place", "release", "push"):
            self.assertEqual(allowed_intensities("grasp_loss", phase), ())
        for phase, intensity in (("grasp", "high"), ("transport", "low")):
            injector = self.injector("grasp_loss", intensity, primitive=phase)
            self.assertIsNone(injector.armed)
            self.assertEqual(injector.fired, [])

    def test_low_and_high_change_actual_pose_commands(self):
        for kind in ("transition", "orientation"):
            commands = []
            for intensity in ("low", "high"):
                injector = self.injector(kind, intensity)
                original = Mock(return_value="ok")
                injector.saved["line"] = original
                injector._line(np.array([.2, 0, 1.]), np.eye(3))
                commands.append(original.call_args.args)
                self.assertIn(intensity, injector.fired[0])
            if kind == "transition":
                for command, limits in zip(commands, ((.05, .15), (.15, .25))):
                    distance = np.linalg.norm(command[0] - [.2, 0, 1.])
                    self.assertGreaterEqual(distance, limits[0])
                    self.assertLessEqual(distance, limits[1])
                    self.assertEqual(command[0][2], 1.)
            else:
                for command, intensity in zip(commands, ("low", "high")):
                    angles = Rotation.from_matrix(command[1]).as_euler('xyz', degrees=True)
                    low, high = TILT_BANDS[intensity]
                    self.assertTrue(np.all(angles >= low))
                    self.assertTrue(np.all(angles <= high))

    def test_transition_randomizes_distance_and_xy_direction_in_intensity_band(self):
        self.assertEqual(TRANSITION_BANDS, {'low': (.05, .15), 'high': (.15, .25)})
        for intensity, limits in (('low', (.05, .15)), ('high', (.15, .25))):
            directions, distances = [], []
            for seed in range(30):
                injector = self.injector('transition', intensity)
                injector.random = np.random.default_rng(seed)
                original = Mock(return_value='ok')
                injector.saved['line'] = original
                position = np.array([.2, 0., 1.])
                injector._line(position, sp.DOWN)
                delta = original.call_args.args[0] - position
                distance = injector.transition_distance()
                self.assertAlmostEqual(np.linalg.norm(delta), distance)
                self.assertGreaterEqual(distance, limits[0])
                self.assertLessEqual(distance, limits[1])
                self.assertEqual(delta[2], 0.)
                np.testing.assert_allclose(original.call_args.args[1], sp.DOWN)
                self.assertIn('%.1f cm off target in 2D X/Y' % (distance*100), injector.fired[0])
                directions.append(delta / distance)
                distances.append(distance)
            directions = np.asarray(directions)
            self.assertGreater(np.std(distances), .02)
            for axis in range(2):
                self.assertTrue(np.any(directions[:, axis] > 0))
                self.assertTrue(np.any(directions[:, axis] < 0))
                self.assertGreater(np.std(directions[:, axis]), .1)

    def test_transition_keeps_the_same_direction_for_the_row_and_redraws_for_the_next(self):
        injector = self.injector('transition', 'high')
        injector.random = np.random.default_rng(0)
        position = np.array([.2, 0., 1.])
        changed = injector.deviated(position)
        distance = injector.transition_distance()
        random_state = copy.deepcopy(injector.random.bit_generator.state)
        injector.session.world.eef = lambda: np.array([2., -1., .5])
        np.testing.assert_allclose(injector.deviated(position), changed)
        self.assertEqual(injector.transition_distance(), distance)
        self.assertEqual(injector.random.bit_generator.state, random_state)
        injector.start_row(1)
        self.assertFalse(np.allclose(injector.deviated(position), changed))
        self.assertNotEqual(injector.transition_distance(), distance)

    def test_paired_offsets_are_identical_regardless_of_accessor_order(self):
        for intensity in ('low', 'high'):
            first = self.injector('transition', intensity)
            second = self.injector('transition', intensity)
            first.random = np.random.default_rng(17)
            second.random = np.random.default_rng(17)
            distance = first.transition_distance()
            direction = second.transition_direction()
            self.assertEqual(second.transition_distance(), distance)
            np.testing.assert_array_equal(first.transition_direction(), direction)
            np.testing.assert_array_equal(first.deviated(np.zeros(3)), second.deviated(np.zeros(3)))

    def test_each_orientation_fault_draws_all_three_angles_in_the_requested_band(self):
        self.assertEqual(TILT_BANDS, {'low': (10., 20.), 'high': (20., 40.)})
        nominal = Rotation.from_euler('xyz', [80, -25, 15], degrees=True).as_matrix()
        for intensity in ("low", "high"):
            draws = []
            for seed in range(30):
                injector = self.injector("orientation", intensity)
                injector.random = np.random.default_rng(seed)
                original = Mock(return_value="ok")
                injector.saved["line"] = original
                injector._line(np.array([.2, 0, 1.]), nominal)
                turned = original.call_args.args[1]
                degrees = Rotation.from_matrix(nominal.T @ turned).as_euler('xyz', degrees=True)
                low, high = TILT_BANDS[intensity]
                with self.subTest(intensity=intensity, seed=seed):
                    self.assertTrue(np.all(degrees >= low))
                    self.assertTrue(np.all(degrees <= high))
                    np.testing.assert_allclose(original.call_args.args[0], [.2, 0, 1.])
                    for name, angle in zip(TILT_AXES.values(), degrees):
                        self.assertIn('%s %.0f degrees' % (name, angle), injector.fired[0])
                    self.assertEqual(len(set(np.round(degrees, 6))), 3)
                draws.append(degrees)
            for axis in range(3):
                self.assertGreater(len(set(np.round(np.asarray(draws)[:, axis], 6))), 1)

    def test_one_orientation_fault_keeps_a_single_drawn_tilt(self):
        injector = self.injector("orientation", "high")
        injector.saved["line"] = Mock(return_value="ok")
        injector._line(np.array([.2, 0, 1.]), np.eye(3))
        random_state = copy.deepcopy(injector.random.bit_generator.state)
        self.assertEqual(injector.wrist_tilt(), injector.tilts[0])
        self.assertEqual(injector.random.bit_generator.state, random_state)
        self.assertEqual(injector.wrist_tilt()[0], 'xyz')

    def test_an_explicit_angle_still_fixes_the_wrist_deviation(self):
        session = SimpleNamespace(executor=None, injected_failure="")
        for intensity, degrees in (("low", 15), ("high", 60)):
            injector = Injector(session, {0: dict(kind="orientation", intensity=intensity)},
                                angle=np.radians(60))
            injector.start_row(0)
            axis, angle = injector.wrist_tilt()
            self.assertEqual(axis, "x")
            self.assertAlmostEqual(np.degrees(angle), degrees)

    def test_collision_intensity_changes_contact_depth(self):
        for intensity, depth in (("low", .01), ("high", .03)):
            injector = self.injector("collision", intensity)
            injector.saved["line"] = Mock(return_value="ok")
            injector.saved["reach"] = Mock()
            with patch("waypoint_failures.held_object", return_value=""), \
                    patch("waypoint_failures.collision_target", return_value=("bowl", np.ones(3))) as target:
                injector._line(np.zeros(3), np.eye(3))
            self.assertAlmostEqual(target.call_args.kwargs["penetration"], depth)
        _, low = collision_target(Bowl(), [0, 0, 1.], [.6, 0, 1.], penetration=.01)
        _, high = collision_target(Bowl(), [0, 0, 1.], [.6, 0, 1.], penetration=.03)
        self.assertGreater(high[0], low[0])

    def test_high_collision_pushes_along_measured_gripper_facing_then_resumes(self):
        for degrees in (0, 35, 90):
            with self.subTest(degrees=degrees):
                injector = self.injector("collision", "high")
                here, facing = np.array([.1, .2, .9]), tilt(sp.DOWN, np.radians(degrees))
                injector.session.gripper_pose = lambda: (here, facing)
                original = Mock(return_value="ok")
                injector.saved.update(line=original, reach=Mock())
                goal = np.array([.2, 0., 1.])
                with patch("waypoint_failures.held_object", return_value=""), \
                        patch("waypoint_failures.collision_target", return_value=("bowl", np.ones(3))):
                    self.assertEqual(injector._line(goal, sp.DOWN), "ok")
                self.assertEqual(original.call_count, 3)
                push = original.call_args_list[1]
                np.testing.assert_allclose(push.args[0]-here, facing[:, 2]*HIGH_COLLISION_OFFSET, atol=1e-12)
                np.testing.assert_allclose(push.args[1], facing)
                np.testing.assert_allclose(original.call_args_list[-1].args[0], goal)
                self.assertIn("12 cm", injector.fired[0])

    def test_low_collision_only_bumps_without_the_forward_push(self):
        injector = self.injector("collision", "low")
        original = Mock(return_value="ok")
        injector.saved.update(line=original, reach=Mock())
        injector.session.gripper_pose = Mock(side_effect=AssertionError("Low must not push"))
        with patch("waypoint_failures.held_object", return_value=""), \
                patch("waypoint_failures.collision_target", return_value=("bowl", np.ones(3))):
            injector._line(np.zeros(3), sp.DOWN)
        self.assertEqual(original.call_count, 2)
        injector.session.gripper_pose.assert_not_called()

    def test_unreachable_high_push_still_continues_to_original_target(self):
        injector = self.injector("collision", "high")
        original = Mock(side_effect=["ok", "unreachable: forward push", "ok"])
        injector.saved.update(line=original, reach=Mock())
        goal = np.array([.2, 0., 1.])
        with patch("waypoint_failures.held_object", return_value=""), \
                patch("waypoint_failures.collision_target", return_value=("bowl", np.ones(3))):
            self.assertEqual(injector._line(goal, sp.DOWN), "ok")
        np.testing.assert_allclose(original.call_args.args[0], goal)
        self.assertIn("unreachable: forward push", injector.missed[-1])
        self.assertEqual(injector.fired, [])

    def test_low_grasp_loss_prevents_close_and_every_retry(self):
        injector = self.injector("grasp_loss", "low")
        close = Mock(return_value="ok")
        injector.saved["run_subgoal"] = close
        sg = sp.Subgoal("grasp", "grip", gripper=sp.GRIPPER_CLOSE)
        with patch("waypoint_failures.held_object", return_value=""):
            first = injector._run_subgoal(sg)
            retry = injector._run_subgoal(sg)
        self.assertIn("never grasped", first)
        self.assertIn("remains open", retry)
        close.assert_not_called()
        self.assertEqual(len(injector.fired), 1)
        self.assertEqual(injector.session.executor.gripper, sp.GRIPPER_OPEN)
        np.testing.assert_array_equal(injector.session.executor.carry_offset, np.zeros(3))

    def test_transport_close_does_not_drop_before_motion(self):
        injector = self.injector("grasp_loss", "high")
        injector.primitive = "transport"
        injector.saved["run_subgoal"] = Mock(return_value="ok")
        with patch("waypoint_failures.held_object", return_value="bowl"), \
                patch.object(injector, "release") as release:
            result = injector._run_subgoal(sp.Subgoal("close", "grip", gripper=sp.GRIPPER_CLOSE))
        self.assertEqual(result, "ok")
        release.assert_not_called()
        self.assertEqual(injector.armed, "grasp_loss")

    def test_learned_primitive_metadata_overrides_manual_inference(self):
        waypoint = SimpleNamespace(skill_role="move", learned=dict(primitive="transport"))
        self.assertEqual(waypoint_primitive(waypoint), "move")
        waypoint.skill_role = ""
        self.assertEqual(waypoint_primitive(waypoint), "transport")

    def test_explicit_manual_primitive_overrides_gripper_settings(self):
        for role in ("move", "transport"):
            waypoint = Waypoint(role.title(), gripper="close", carry=True)
            self.assertEqual(waypoint_primitive(waypoint), role)

    def test_high_orientation_is_not_reduced_when_unreachable(self):
        injector = self.injector("orientation", "high")
        original = Mock(return_value="unreachable: Cartesian path misses")
        injector.saved["line"] = original
        self.assertTrue(injector._line(np.ones(3), np.eye(3)).startswith("unreachable"))
        original.assert_called_once()
        axis, angles = injector.tilts[0]
        np.testing.assert_allclose(original.call_args.args[1], tilt(np.eye(3), angles, axis))
        self.assertEqual(injector.fired, [])
        self.assertIn("unreachable", injector.missed[0])

    def test_random_low_orientation_is_never_reduced_below_its_band(self):
        for continuing in (False, True):
            with self.subTest(continuing=continuing):
                injector = self.injector("orientation", "low")
                injector.session.continue_on_failure = continuing
                original = Mock(return_value="unreachable: Cartesian path misses")
                injector.saved["line"] = original
                self.assertTrue(injector._line(np.ones(3), np.eye(3)).startswith("unreachable"))
                original.assert_called_once()
                self.assertEqual(injector.fired, [])
                angles = Rotation.from_matrix(original.call_args.args[1]).as_euler('xyz', degrees=True)
                self.assertTrue(np.all(angles >= 10))
                self.assertTrue(np.all(angles <= 20))


class PlaybackContinuationTest(unittest.TestCase):
    def session(self, keep_going=True):
        session = Session(sp.Args())
        session.continue_on_failure = keep_going
        session.poses = lambda: {}
        session.gripper_pose = lambda: (np.zeros(3), np.eye(3))
        session.update_run_frames = lambda *_: None
        session.check_motion_result = lambda waypoint, result, *args: result
        session.executor = SimpleNamespace(
            _set_ignore=Mock(), gripper=sp.GRIPPER_OPEN,
            line=Mock(side_effect=["failed: target missed", "ok"]),
            run_subgoal=Mock(return_value="ok"))
        session.env = SimpleNamespace(check_success=lambda: False)
        return session

    def chain(self):
        return Chain("libero_spatial", 0, "test", [
            Waypoint("first", WORLD, [.1, 0., 0.], motion="line"),
            Waypoint("second", WORLD, [.2, 0., 0.], motion="line")])

    def test_failed_motion_does_not_skip_the_next_waypoint(self):
        session = self.session()
        reports = []
        result = session.run(self.chain(), lambda *r: reports.append(r))
        self.assertTrue(result.startswith("Finished"), result)
        self.assertEqual(session.executor.line.call_count, 2)
        self.assertEqual(reports[-1], (1, "ok"))
        self.assertEqual(session.last_run_failures, [dict(row=0, result="failed: target missed")])

    def test_failed_grip_does_not_skip_the_move_or_next_waypoint(self):
        session = self.session()
        session.executor.run_subgoal.return_value = "failed: nothing between the fingers"
        session.executor.line.side_effect = ["ok", "ok"]
        chain = self.chain()
        chain.waypoints[0].gripper = "close"
        result = session.run(chain, lambda *_: None)
        self.assertTrue(result.startswith("Finished"), result)
        self.assertEqual(session.executor.line.call_count, 2)
        self.assertEqual(len(session.last_run_failures), 1)

    def test_normal_non_gui_playback_can_still_stop_on_failure(self):
        session = self.session(keep_going=False)
        result = session.run(self.chain(), lambda *_: None)
        self.assertTrue(result.startswith("Stopped at 1"), result)
        self.assertEqual(session.executor.line.call_count, 1)

    def test_explicit_stop_still_interrupts_continued_playback(self):
        session = self.session()
        def report(row, result):
            if row == 1 and result == "Running":
                raise sp.Abort()
        with self.assertRaises(sp.Abort):
            session.run(self.chain(), report)
        self.assertEqual(session.executor.line.call_count, 1)

    def test_completion_waits_for_final_gripper_check_then_stops_before_next_move(self):
        session = self.session()
        session.executor.run_subgoal.return_value = 'failed: nothing between the fingers'
        chain = self.chain()
        chain.waypoints[0].gripper_after = 'close'
        completed = []
        def finished(row, result):
            self.assertEqual(session.executor.line.call_count, 1)
            self.assertEqual(session.executor.run_subgoal.call_count, 1)
            completed.append((row, result))
            raise sp.Abort()
        session.on_waypoint_finished = finished
        with self.assertRaises(sp.Abort):
            session.run(chain, lambda *_: None)
        self.assertEqual(completed, [(0, 'failed: target missed; failed: nothing between the fingers')])
        self.assertEqual(session.executor.line.call_count, 1)

    def test_abort_before_motion_cannot_emit_completion(self):
        session = self.session(keep_going=False)
        session.executor.run_subgoal.return_value = 'failed: gripper did not finish'
        session.on_waypoint_finished = Mock()
        chain = self.chain()
        chain.waypoints[0].gripper = 'close'
        self.assertTrue(session.run(chain, lambda *_: None).startswith('Stopped'))
        session.executor.line.assert_not_called()
        session.on_waypoint_finished.assert_not_called()

    def test_motion_policy_is_applied_and_restored_after_playback(self):
        session = self.session()
        session.executor.arrival_grace_seconds = None
        session.executor.retry_motion = True
        policies = []
        def line(*_):
            policies.append((session.executor.arrival_grace_seconds, session.executor.retry_motion))
            return "failed: robot did not settle at the target"
        session.executor.line.side_effect = line
        result = session.run(self.chain(), lambda *_: None)
        self.assertTrue(result.startswith("Finished"), result)
        self.assertEqual(policies, [(1.0, False), (1.0, False)])
        self.assertIsNone(session.executor.arrival_grace_seconds)
        self.assertTrue(session.executor.retry_motion)

    def test_motion_policy_restores_missing_attributes_after_stop(self):
        session = self.session()
        with self.assertRaises(sp.Abort):
            with session.playback_policy():
                self.assertEqual(session.executor.arrival_grace_seconds, 1.0)
                raise sp.Abort()
        self.assertFalse(hasattr(session.executor, "arrival_grace_seconds"))
        self.assertFalse(hasattr(session.executor, "retry_motion"))


class PlaybackDeadlineTest(unittest.TestCase):
    def test_unsettled_target_gets_at_most_one_extra_second_of_playback(self):
        for dt in (.05, .02, .1):
            with self.subTest(dt=dt):
                ex = sp.Executor.__new__(sp.Executor)
                ex.arrival_grace_seconds = 1.0
                ex.env = SimpleNamespace(env=SimpleNamespace(control_timestep=dt))
                ex.scene = SimpleNamespace(arm_qpos=lambda: np.zeros(2))
                ex.q_hold = np.ones(2)
                elapsed = [0.]
                def step():
                    elapsed[0] += dt
                ex._apply = Mock(side_effect=step)
                with patch("subgoal_planner.time.monotonic", side_effect=lambda: elapsed[0]):
                    self.assertFalse(ex._settle(max_steps=300))
                self.assertLess(elapsed[0], 1.0+dt+1e-9)
                self.assertGreaterEqual(elapsed[0], 1.0-1e-9)

    def test_planned_path_is_not_redriven_after_arrival_deadline(self):
        ex = sp.Executor.__new__(sp.Executor)
        ex.scene = SimpleNamespace(arm_qpos=lambda: np.zeros(2))
        ex.retry_motion = False
        ex._follow_path = Mock()
        ex._settle = Mock(return_value=False)
        ex._drive_to_q = Mock()
        self.assertFalse(ex.follow([np.zeros(2), np.ones(2)]))
        ex._follow_path.assert_called_once()
        ex._settle.assert_called_once()
        ex._drive_to_q.assert_not_called()


@unittest.skipUnless(os.environ.get("DISPLAY"), "A real or virtual display is required")
class InjectedFailureTest(unittest.TestCase):
    """One studio, one scene, each fault armed and run for real."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.stores = gui.STORE, gui.SAVED
        gui.STORE, gui.SAVED = Path(cls.folder.name), Path(cls.folder.name) / "saved"
        cls.chain = Chain.load(FIXTURE, "libero_spatial", 0)
        cls.chain.save(gui.STORE / "libero_spatial/task_000.json")
        cls.errors = []
        cls.error_patch = patch.object(gui.messagebox, "showerror",
                                       side_effect=lambda title, msg, **kw: cls.errors.append(str(msg)))
        cls.error_patch.start()
        cls.root = tk.Tk()
        cls.root.report_callback_exception = lambda typ, value, tb: cls.errors.append(str(value))
        cls.panel = gui.Panel(cls.root, sp.Args(task_suite_name="libero_spatial", task_id=0))
        cls.panel.session.realtime = False
        cls.root.update()
        cls.panel.load_task()
        cls.root.update()
        cls.state = cls.panel.session.env.sim.get_state().flatten().copy()
        # Physics assertions do not need camera sensors at every control step.
        domain = cls.panel.session.env.env
        for name in domain._observables:
            if name.endswith("_image"):
                domain.modify_observable(name, "enabled", False)
                domain.modify_observable(name, "active", False)
        cls.panel.session.tick = cls.root.update


    @classmethod
    def tearDownClass(cls):
        cls.panel.form_dirty = False
        cls.panel.close()
        cls.error_patch.stop()
        gui.STORE, gui.SAVED = cls.stores
        cls.folder.cleanup()

    def setUp(self):
        p = self.panel
        p.session.env.set_init_state(self.state)
        p.session.attach(settle=False)
        p.session.executor.gripper = sp.GRIPPER_OPEN
        p.chain = copy.deepcopy(self.chain)
        p.injections = {}
        p.form_dirty = False
        p.editing_index = None
        p.redraw_rows(0)
        p.select()
        self.errors.clear()
        self.root.update()

    def tearDown(self):
        self.assertEqual(self.errors, [])
        self.assertFalse(self.panel.busy)

    def play(self, rows, injections):
        """Run the first `rows` waypoints and report what the robot did.

        The intended endpoint is read before the run, so a fault is measured
        against where the plan said to go, not where the arm ended up.
        """
        p = self.panel
        # Pose faults are exercised on a contact phase; free Move phases are
        # separately checked below to ensure they can never fire a fault.
        for row, selection in injections.items():
            kind = selection if isinstance(selection, str) else selection["kind"]
            if kind != "grasp_loss" and p.chain.waypoints[row].name == "approach":
                p.chain.waypoints[row].name = "Grasp contact"
        p.chain.waypoints = p.chain.waypoints[:rows]
        p.injections = dict(injections)
        p.redraw_rows(0)
        p.select()
        target = p.session.preview(p.chain)[-1]
        made, original = [], p.failure_injector
        with patch.object(p, "failure_injector",
                          side_effect=lambda: made.append(original()) or made[-1]):
            p.run()
        self.root.update()
        position, orientation = p.session.gripper_pose()
        turn = np.arccos(np.clip((np.trace(orientation @ np.asarray(target[1]).T)-1)/2, -1, 1))
        return dict(injector=made[0], status=p.status.get(),
                    gap=float(np.linalg.norm(position - target[0])),
                    residual=np.asarray(position) - np.asarray(target[0]),
                    turn=float(np.degrees(turn)))

    def touching(self):
        """Bodies the arm is in contact with that are not part of the arm."""
        session = self.panel.session
        robot = session.executor.scene.robot_geoms
        model, data = session.env.sim.model, session.env.sim.data
        names = set()
        for c in range(data.ncon):
            first, second = data.contact[c].geom1, data.contact[c].geom2
            if (first in robot) == (second in robot):
                continue
            other = second if first in robot else first
            names.add(model.body_id2name(model.geom_bodyid[other]) or "")
        return names

    def test_an_unarmed_run_moves_exactly_as_planned(self):
        outcome = self.play(1, {})
        self.assertEqual(outcome["injector"].fired, [])
        self.assertLess(outcome["gap"], self.panel.args.reach_tolerance)
        self.assertNotIn("injected", outcome["status"])
        for name in Injector.METHODS:
            self.assertNotIn(name, vars(self.panel.session.executor))

    def test_transition_deviation_stops_at_the_random_2d_offset(self):
        with patch.object(self.panel, 'failure_injector', side_effect=lambda:
                          Injector(self.panel.session, self.panel.injections, seed=0)):
            outcome = self.play(1, {0: "transition"})
        distance = outcome['injector'].distances[0]
        self.assertGreaterEqual(distance, .15)
        self.assertLessEqual(distance, .25)
        requested = distance * np.asarray(outcome['injector'].directions[0])
        self.assertLess(np.linalg.norm(outcome['residual'] - requested), self.panel.args.reach_tolerance)
        self.assertLess(outcome["turn"], 2)
        self.assertIn('%.1f cm off target in 2D X/Y' % (distance*100), outcome['injector'].fired[0])
        self.assertEqual(outcome['injector'].directions[0][2], 0.)
        self.assertLess(abs(outcome['residual'][2]), self.panel.args.reach_tolerance)
        self.assertIn("injected", outcome["status"])

    def test_low_transition_uses_5_to_15_cm_without_changing_target_height(self):
        with patch.object(self.panel, 'failure_injector', side_effect=lambda:
                          Injector(self.panel.session, self.panel.injections, seed=0)):
            outcome = self.play(1, {0: dict(kind='transition', intensity='low')})
        distance = outcome['injector'].distances[0]
        self.assertGreaterEqual(distance, .05)
        self.assertLessEqual(distance, .15)
        requested = distance * np.asarray(outcome['injector'].directions[0])
        self.assertEqual(requested[2], 0.)
        self.assertLess(np.linalg.norm(outcome['residual'] - requested), self.panel.args.reach_tolerance)
        self.assertLess(outcome['turn'], 2.)

    def test_the_deviation_is_the_distance_that_was_asked_for(self):
        with patch.object(self.panel, "failure_injector", side_effect=lambda:
                          Injector(self.panel.session, self.panel.injections, offset=.12, seed=0)):
            outcome = self.play(1, {0: "transition"})
        requested = .12 * np.asarray(outcome['injector'].directions[0])
        self.assertLess(np.linalg.norm(outcome['residual'] - requested), self.panel.args.reach_tolerance)

    def test_high_orientation_keeps_large_rotation_when_the_target_is_missed(self):
        with patch.object(self.panel, 'failure_injector', side_effect=lambda:
                          Injector(self.panel.session, self.panel.injections, seed=0)):
            outcome = self.play(1, {0: "orientation"})
        axis, angles = outcome["injector"].tilts[0]
        low, high = TILT_BANDS["high"]
        self.assertTrue(np.all(np.degrees(angles) >= low))
        self.assertTrue(np.all(np.degrees(angles) <= high))
        angle = Rotation.from_matrix(tilt(np.eye(3), angles, axis)).magnitude()
        self.assertGreater(outcome["turn"], .8 * np.degrees(angle))
        self.assertIn("wrist tilted", outcome["injector"].fired[0])
        for name in TILT_AXES.values():
            self.assertIn(name, outcome["injector"].fired[0])
        self.assertTrue(outcome["status"].startswith("Sequence"), outcome["status"])

    def test_high_orientation_target_miss_advances_without_restarting(self):
        running, report = [], self.panel.report
        def observe(index, result, follow=False):
            if result == "Running":
                running.append(index)
            report(index, result, follow)
        with patch.object(self.panel, "report", side_effect=observe), \
                patch.object(self.panel, 'failure_injector', side_effect=lambda:
                    Injector(self.panel.session, self.panel.injections, seed=0)):
            outcome = self.play(3, {0: dict(kind="orientation", intensity="high")})
        self.assertEqual(running, [0, 1, 2])
        self.assertTrue(outcome["status"].startswith("Sequence"), outcome["status"])
        self.assertEqual(len(outcome["injector"].fired), 1)

    def test_collision_drives_into_a_body_and_shifts_it(self):
        before = self.panel.session.world.pos("akita_black_bowl_1").copy()
        touched = set()
        tick = self.panel.session.tick
        self.panel.session.tick = lambda: (tick(), touched.update(self.touching()))
        try:
            outcome = self.play(1, {0: "collision"})
        finally:
            self.panel.session.tick = tick
        self.assertIn("drove into", outcome["injector"].fired[0])
        self.assertTrue(any(name.startswith("akita_black_bowl_1") for name in touched), touched)
        moved = np.linalg.norm(self.panel.session.world.pos("akita_black_bowl_1") - before)
        self.assertGreater(moved, .005)
        # The bump is an extra motion: the original waypoint must still be reached.
        self.assertLess(outcome["gap"], self.panel.args.reach_tolerance)
        self.assertEqual(self.panel.session.last_run_failures, [])

    def test_high_collision_makes_a_real_forward_stroke_that_low_does_not(self):
        for intensity, count in (("low", 1), ("high", 2)):
            with self.subTest(intensity=intensity):
                self.setUp()
                session = self.panel.session
                line, calls = session.executor.line, []
                def observe(position, orientation=None, *args, **kwargs):
                    start, facing = session.gripper_pose()
                    result = line(position, orientation, *args, **kwargs)
                    calls.append(dict(start=start, facing=facing, target=np.asarray(position).copy(),
                                      orientation=np.asarray(orientation).copy(), result=result,
                                      end=session.world.eef().copy()))
                    return result
                with patch.object(session.executor, "line", side_effect=observe):
                    outcome = self.play(1, {0: dict(kind="collision", intensity=intensity)})
                self.assertEqual(len(calls), count)
                self.assertEqual(len(outcome["injector"].fired), 1)
                self.assertLess(outcome["gap"], self.panel.args.reach_tolerance)
                if intensity == "high":
                    push = calls[1]
                    np.testing.assert_allclose(push["target"]-push["start"],
                        push["facing"][:, 2]*HIGH_COLLISION_OFFSET, atol=1e-12)
                    np.testing.assert_allclose(push["orientation"], push["facing"], atol=1e-12)
                    self.assertFalse(push["result"].startswith("unreachable"), push["result"])
                    forward_travel = (push["end"]-push["start"]) @ push["facing"][:, 2]
                    self.assertGreater(forward_travel, .005)
                    self.assertIn("12 cm", outcome["injector"].fired[0])

    def test_collision_between_waypoints_preserves_both_following_targets(self):
        p = self.panel
        base = p.chain.waypoints[0]
        p.chain.waypoints = [copy.deepcopy(base) for _ in range(3)]
        for index, waypoint in enumerate(p.chain.waypoints):
            waypoint.name = "Grasp contact %d" % (index+2)
            waypoint.offset[1] += .03*index
        targets = p.session.preview(p.chain)
        reached, touched = [], set()
        report, tick = p.report, p.session.tick
        def observe(index, result, follow=False):
            if result != "Running":
                reached.append((index, result, p.session.world.eef().copy()))
            report(index, result, follow)
        p.session.tick = lambda: (tick(), touched.update(self.touching()))
        try:
            with patch.object(p, "report", side_effect=observe):
                outcome = self.play(3, {1: "collision"})
        finally:
            p.session.tick = tick
        self.assertEqual([row for row, _, _ in reached], [0, 1, 2])
        self.assertTrue(touched)
        self.assertEqual(len(outcome["injector"].fired), 1)
        for row, result, position in reached:
            self.assertTrue(result.startswith("ok"), result)
            self.assertLess(np.linalg.norm(position-targets[row][0]), p.args.reach_tolerance)

    def test_collision_on_a_joint_route_still_reaches_its_endpoint(self):
        session = self.panel.session
        target, orientation = session.preview(self.panel.chain)[0]
        ex = session.executor
        q, error = ex.scene.ik(target, q_init=ex.scene.arm_qpos(), target_ori=orientation)
        self.assertLess(error, .01)
        path = ex.planner.plan(ex.scene.arm_qpos(), q)
        self.assertIsNotNone(path)
        touched = set()
        tick = session.tick
        session.tick = lambda: (tick(), touched.update(self.touching()))
        try:
            with Injector(session, {0: "collision"}) as injector:
                injector.start_row(0)
                self.assertTrue(ex.follow(path))
                self.assertEqual(session.injected_failure, "")
        finally:
            session.tick = tick
        self.assertTrue(touched)
        self.assertEqual(len(injector.fired), 1)
        self.assertLess(np.linalg.norm(session.world.eef()-target), session.args.reach_tolerance)

    def test_a_faulty_waypoint_does_not_stop_the_sequence(self):
        # The fault belongs to its own row. What it costs the rows after it is
        # the whole reason for recording the run, so they have to get to run.
        with patch.object(self.panel, 'failure_injector', side_effect=lambda:
                          Injector(self.panel.session, self.panel.injections, seed=0)):
            outcome = self.play(3, {0: "transition"})
        self.assertTrue(outcome["status"].startswith("Sequence"), outcome["status"])
        self.assertIn("injected: waypoint 1 transition deviation",
                      outcome["status"])
        # Tk gives back "" for a row with no tags and a tuple for one with any.
        self.assertNotIn("failed", self.panel.rows.item("0", "tags"))
        self.assertLess(outcome["gap"], self.panel.args.reach_tolerance)

    def test_grasp_loss_empties_the_gripper_mid_transfer(self):
        world = self.panel.session.world
        grasped = []
        # Row 3 carries the bowl across; check the fingers really had it first.
        report = self.panel.report
        def watch(index, result, follow=False):
            if index == 3 and result == "Running":
                grasped.append(held_object(world))
            report(index, result, follow)
        with patch.object(self.panel, "report", side_effect=watch):
            outcome = self.play(4, {3: dict(kind="grasp_loss", intensity="high")})
        self.assertEqual(grasped, ["akita_black_bowl_1"])
        self.assertEqual(held_object(world), "")
        self.assertEqual(outcome["injector"].fired,
                         ["waypoint 4 grasp loss / high (dropped the akita black bowl 1)"])
        self.assertLess(world.pos("akita_black_bowl_1")[2], .95)
        # A hand that has lost its grip stays open; closing it again would be
        # the arm recovering from the fault inside the fault.
        self.assertEqual(self.panel.session.executor.gripper, sp.GRIPPER_OPEN)

    def test_low_grasp_loss_never_closes_after_the_descent(self):
        # The actual learned GUI's Grasp row descends empty and closes AFTER.
        self.panel.chain.waypoints[1].gripper_after = "close"
        with patch.object(Injector, "release", wraps=Injector.release) as release:
            outcome = self.play(2, {1: dict(kind="grasp_loss", intensity="low")})
        release.assert_not_called()
        self.assertEqual(held_object(self.panel.session.world), "")
        self.assertEqual(self.panel.session.executor.gripper, sp.GRIPPER_OPEN)
        self.assertIn("Sequence finished with 1 failed waypoint", outcome["status"])
        self.assertEqual(outcome["injector"].fired,
                         ["waypoint 2 grasp loss / low (no grasp; fingers stayed open)"])
        self.assertEqual(outcome["injector"].missed, [])

    def test_low_grasp_only_step_stays_open_and_clears_carry_state(self):
        self.panel.chain.waypoints[2].motion = "grip"
        outcome = self.play(3, {2: dict(kind="grasp_loss", intensity="low")})
        self.assertEqual(held_object(self.panel.session.world), "")
        ex = self.panel.session.executor
        self.assertEqual(ex.gripper, sp.GRIPPER_OPEN)
        self.assertEqual(ex.holding, "")
        np.testing.assert_array_equal(ex.carry_offset, np.zeros(3))
        self.assertEqual(ex.carry_bottom, 0.)
        self.assertEqual(len(outcome["injector"].fired), 1)

    def test_empty_hand_is_not_reported_as_an_injected_drop(self):
        outcome = self.play(1, {0: "grasp_loss"})
        self.assertEqual(outcome["injector"].fired, [])
        self.assertIn("not injected", outcome["status"])
        self.assertIn("not allowed in Move", outcome["status"])
        self.assertLess(outcome["gap"], self.panel.args.reach_tolerance)

    def test_high_grasp_loss_is_rejected_in_grasp(self):
        outcome = self.play(3, {2: dict(kind="grasp_loss", intensity="high")})
        self.assertEqual(outcome["injector"].fired, [])
        self.assertIn("not allowed in Grasp", outcome["status"])
        self.assertEqual(held_object(self.panel.session.world), "akita_black_bowl_1")

    def test_planned_transport_drops_before_the_destination(self):
        self.play(3, {})
        session = self.panel.session
        self.assertEqual(held_object(session.world), "akita_black_bowl_1")
        start = session.world.eef().copy()
        target = start + [0., -.10, .06]
        ex = session.executor
        q, error = ex.scene.ik(target, q_init=ex.scene.arm_qpos(),
                               target_ori=session.gripper_pose()[1])
        self.assertLess(error, .01)
        released_at = []
        release = Injector.release
        def observe(injector):
            released_at.append(session.world.eef().copy())
            return release(injector)
        with patch.object(Injector, "release", observe):
            with Injector(session, {0: "grasp_loss"}) as injector:
                injector.start_row(0)
                self.assertTrue(ex.follow([ex.scene.arm_qpos(), q]))
        self.assertEqual(len(injector.fired), 1)
        self.assertEqual(held_object(session.world), "")
        self.assertGreater(np.linalg.norm(released_at[0]-start), .02)
        self.assertGreater(np.linalg.norm(released_at[0]-target), .02)
        self.assertLess(np.linalg.norm(session.world.eef()-target), .03)


@unittest.skipUnless(os.environ.get("DISPLAY"), "A real or virtual display is required")
class FailurePickerTest(unittest.TestCase):
    """Arming failures in the studio, without moving the robot."""

    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.stores = gui.STORE, gui.SAVED
        gui.STORE, gui.SAVED = Path(cls.folder.name), Path(cls.folder.name) / "saved"
        cls.chain = Chain.load(FIXTURE, "libero_spatial", 0)
        cls.chain.save(gui.STORE / "libero_spatial/task_000.json")
        cls.errors = []
        cls.error_patch = patch.object(gui.messagebox, "showerror",
                                       side_effect=lambda title, msg, **kw: cls.errors.append(str(msg)))
        cls.error_patch.start()
        cls.root = tk.Tk()
        cls.root.report_callback_exception = lambda typ, value, tb: cls.errors.append(str(value))
        cls.panel = gui.Panel(cls.root, sp.Args(task_suite_name="libero_spatial", task_id=0))
        cls.panel.session.realtime = False
        cls.root.update()
        cls.panel.load_task()
        cls.root.update()

    @classmethod
    def tearDownClass(cls):
        cls.panel.form_dirty = False
        cls.panel.close()
        cls.error_patch.stop()
        gui.STORE, gui.SAVED = cls.stores
        cls.folder.cleanup()

    def setUp(self):
        self.p = self.panel
        self.p.chain = copy.deepcopy(self.chain)
        self.p.injections = {}
        self.p.form_dirty = False
        self.p.editing_index = None
        self.p.redraw_rows(0)
        self.p.select()
        self.errors.clear()
        self.root.update()

    def tearDown(self):
        self.assertEqual(self.errors, [])

    @staticmethod
    def widgets(parent, kind, found=None):
        found = [] if found is None else found
        for child in parent.winfo_children():
            if isinstance(child, kind):
                found.append(child)
            FailurePickerTest.widgets(child, kind, found)
        return found

    def open_picker(self):
        self.p.failure_picker()
        self.root.update()
        dialog = self.widgets(self.root, tk.Toplevel)[-1]
        return dialog, self.row_boxes(dialog)

    @classmethod
    def row_boxes(cls, dialog, intensity=False):
        boxes = cls.widgets(dialog, ttk.Combobox)[int(intensity)::2]
        result = {}
        for box in boxes:
            label, = box.master.grid_slaves(row=box.grid_info()["row"], column=0)
            index = int(str(label.cget("text")).split()[0])-1
            result[index] = box
        return result

    def press(self, dialog, text):
        button, = [b for b in self.widgets(dialog, ttk.Button) if b.cget("text") == text]
        button.invoke()
        self.root.update()

    def test_picker_lists_eligible_waypoints_and_arms_their_original_indices(self):
        dialog, boxes = self.open_picker()
        self.assertEqual(set(boxes), {1, 2, 3, 4, 5})
        boxes[1].set(FAILURES["collision"])
        boxes[3].set(FAILURES["grasp_loss"])
        self.press(dialog, "Apply failures")
        self.assertEqual(self.p.injections, {1: dict(kind="collision", intensity="low"),
                                             3: dict(kind="grasp_loss", intensity="high")})
        self.assertIn("2 waypoints set to fail", self.p.failure_summary.get())
        self.assertIn(FAILURES["collision"], self.p.rows.item("1", "text"))
        self.assertNotIn("⚠", self.p.rows.item("0", "text"))

    def test_clear_all_disarms_everything(self):
        self.p.injections = {1: "collision", 2: "transition"}
        self.p.redraw_rows(0)
        dialog, boxes = self.open_picker()
        self.assertEqual(boxes[1].get(), FAILURES["collision"])
        self.press(dialog, "Clear all")
        self.press(dialog, "Apply failures")
        self.assertEqual(self.p.injections, {})
        self.assertEqual(self.p.failure_summary.get(), "No failures injected.")

    def test_move_hidden_and_transport_offers_only_high_grasp_loss(self):
        dialog, boxes = self.open_picker()
        strengths = self.row_boxes(dialog, intensity=True)
        self.assertNotIn(0, boxes)
        self.assertNotIn(0, strengths)
        self.assertEqual(tuple(boxes[3].cget("values")), ("No failure", FAILURES["grasp_loss"]))
        boxes[2].set(FAILURES["grasp_loss"])
        boxes[3].set(FAILURES["grasp_loss"])
        self.assertEqual(tuple(strengths[2].cget("values")), ("Low",))
        self.assertEqual(tuple(strengths[3].cget("values")), ("High",))
        self.assertNotIn(FAILURES["grasp_loss"], boxes[4].cget("values"))
        self.press(dialog, "Apply failures")
        self.assertEqual(self.p.injections[2], dict(kind="grasp_loss", intensity="low"))
        self.assertIn("Grasp loss / Low", self.p.rows.item("2", "text"))

    def test_gripper_only_release_row_is_hidden(self):
        self.p.chain.waypoints[5].motion = "grip"
        dialog, boxes = self.open_picker()
        self.assertNotIn(5, boxes)
        self.assertEqual(set(boxes), {1, 2, 3, 4})
        self.press(dialog, "Apply failures")

    def test_applying_removes_old_ineligible_failures(self):
        self.p.injections = {0: "collision", 3: "orientation"}
        dialog, boxes = self.open_picker()
        self.assertNotIn(0, boxes)
        self.assertEqual(boxes[3].get(), "No failure")
        self.press(dialog, "Apply failures")
        self.assertEqual(self.p.injections, {})

    def test_plan_without_eligible_rows_has_an_empty_state(self):
        self.p.chain.waypoints = [Waypoint("Move")]
        dialog, boxes = self.open_picker()
        self.assertEqual(boxes, {})
        text = [label.cget("text") for label in self.widgets(dialog, ttk.Label)]
        self.assertIn("No failures are available for these waypoints.", text)
        self.press(dialog, "Apply failures")
        self.assertEqual(self.p.injections, {})

    def test_transport_only_plan_hides_other_failure_types_in_the_guide(self):
        self.p.chain.waypoints = [Waypoint("Transport")]
        dialog, boxes = self.open_picker()
        self.assertEqual(set(boxes), {0})
        labels = "\n".join(str(label.cget("text")) for label in self.widgets(dialog, ttk.Label))
        for kind in ("collision", "transition", "orientation"):
            self.assertNotIn(FAILURES[kind], labels)
            self.assertNotIn(FAILURES[kind], boxes[0].cget("values"))
        boxes[0].set(FAILURES["grasp_loss"])
        self.press(dialog, "Apply failures")
        self.assertEqual(self.p.injections, {0: dict(kind="grasp_loss", intensity="high")})

    def test_intensity_survives_reopen_and_reorder(self):
        self.p.injections = {1: dict(kind="orientation", intensity="high")}
        dialog, boxes = self.open_picker()
        strengths = self.row_boxes(dialog, intensity=True)
        self.assertEqual(strengths[1].get(), "High")
        self.press(dialog, "Apply failures")
        self.p.rows.selection_set("1")
        self.p.reorder(1)
        self.assertEqual(self.p.injections, {2: dict(kind="orientation", intensity="high")})

    def test_editing_the_list_keeps_failures_on_their_own_waypoints(self):
        kinds = [waypoint.name for waypoint in self.p.chain.waypoints]
        self.p.injections = {1: "collision", 4: "grasp_loss"}
        self.p.rows.selection_set("0")
        self.p.duplicate()
        self.assertEqual(self.p.injections, {2: "collision", 5: "grasp_loss"})
        self.p.rows.selection_set("0")
        self.p.delete()
        self.assertEqual(self.p.injections, {1: "collision", 4: "grasp_loss"})
        self.p.rows.selection_set("1")
        self.p.reorder(1)
        self.assertEqual(self.p.injections, {2: "collision", 4: "grasp_loss"})
        self.assertEqual(self.p.chain.waypoints[2].name, kinds[1])

    def test_picker_has_no_unexplained_numeric_fields(self):
        dialog, _ = self.open_picker()
        entries = [w for w in self.widgets(dialog, ttk.Entry)
                   if not isinstance(w, ttk.Combobox)]
        self.assertEqual(entries, [])
        self.press(dialog, "Apply failures")

    def test_loading_a_task_disarms_the_previous_plan(self):
        self.p.injections = {0: "collision"}
        self.p.load_task()
        self.root.update()
        self.assertEqual(self.p.injections, {})


@unittest.skipUnless(os.environ.get("DISPLAY"), "A real or virtual display is required")
class LearnedGraspLossTest(unittest.TestCase):
    def test_natural_grasp_failure_never_restarts_the_approach(self):
        import waypoint_gmm_editor as gmm
        from waypoint_gmm import GMMChain
        session = gmm.GMMSession(sp.Args(task_suite_name="libero_goal", task_id=3))
        chain = GMMChain.load(Path(__file__).parent / "saved_plans/libero_goal/task_003.json",
                              "libero_goal", 3)
        session.continue_on_failure = True
        running = []
        try:
            session.load("libero_goal", 3, cameras=False)
            with tempfile.TemporaryDirectory() as folder, patch.object(gmm, "MODEL_STORE", Path(folder)), \
                    patch.object(session.world.domain, "_check_grasp", return_value=False):
                result = session.run(chain, lambda row, status:
                                     running.append(row) if status == "Running" else None)
                archive = json.loads(session.last_replay_report.read_text())
            self.assertTrue(result.startswith("Finished"), result)
            grasp_row = next(i for i, w in enumerate(chain.waypoints) if w.skill_role == "grasp")
            self.assertEqual(running.count(grasp_row), 1)
            self.assertEqual(running, sorted(running))
            self.assertEqual(running[-1], len(chain.waypoints)-1)
            self.assertTrue(archive["continued_after_failure"])
            self.assertIn("nothing between the fingers", archive["failures"][0]["result"])
            self.assertIsNone(session.executor.arrival_grace_seconds)
            self.assertTrue(session.executor.retry_motion)
        finally:
            if session.env is not None:
                session.env.close()

    def test_default_gui_task_releases_the_grasp_instead_of_reclosing(self):
        self.run_drop("grasp", keep_going=False)

    def test_default_gui_task_continues_after_grasp_loss(self):
        self.run_drop("grasp", keep_going=True)

    def test_default_gui_task_continues_after_transport_loss(self):
        self.run_drop("transport", keep_going=True)

    def run_drop(self, role, keep_going):
        import waypoint_gmm_editor as gmm
        from waypoint_gmm import GMMChain
        session = gmm.GMMSession(sp.Args(task_suite_name="libero_goal", task_id=3))
        path = Path(__file__).parent / "saved_plans/libero_goal/task_003.json"
        chain = GMMChain.load(path, "libero_goal", 3)
        matching = [i for i, w in enumerate(chain.waypoints) if w.skill_role == role]
        grasp_row = matching[-1] if role == "transport" else matching[0]
        session.continue_on_failure = keep_going
        acquired, running, after_drop = [], [], []
        release = Injector.release
        def observe(injector):
            acquired.append(held_object(session.world))
            return release(injector)
        try:
            session.load("libero_goal", 3, cameras=False)
            with tempfile.TemporaryDirectory() as folder, patch.object(gmm, "MODEL_STORE", Path(folder)):
                with patch.object(Injector, "release", observe):
                    intensity = "low" if role == "grasp" else "high"
                    with Injector(session, {grasp_row: dict(kind="grasp_loss", intensity=intensity)},
                                  waypoints=chain.waypoints) as injector:
                        def report(row, result):
                            if result == "Running":
                                injector.start_row(row)
                                running.append(row)
                                if injector.fired:
                                    after_drop.append(session.world.eef().copy())
                        result = session.run(chain, report)
                archive = json.loads(session.last_replay_report.read_text())
            self.assertEqual(acquired, [chain.execution_skill["source"]] if role == "transport" else [])
            if keep_going:
                self.assertTrue(result.startswith("Finished"), result)
                self.assertEqual(running[-1], len(chain.waypoints)-1)
                self.assertTrue(set(range(grasp_row+1, len(chain.waypoints))) <= set(running))
                self.assertGreater(np.linalg.norm(after_drop[-1]-after_drop[0]), .05)
                self.assertTrue(archive["continued_after_failure"])
                self.assertTrue(archive["failures"])
                self.assertFalse(archive["task_success"])
                self.assertTrue(session.last_run_failures)
            else:
                self.assertIn("Stopped: failed: injected grasp loss", result)
            self.assertEqual(len(injector.fired), 1)
            self.assertEqual(injector.missed, [])
            self.assertEqual(held_object(session.world), "")
            self.assertEqual(session.executor.gripper, sp.GRIPPER_OPEN)
            for name in Injector.METHODS:
                self.assertNotIn(name, vars(session.executor))
        finally:
            if session.env is not None:
                session.env.close()


if __name__ == "__main__":
    unittest.main()
