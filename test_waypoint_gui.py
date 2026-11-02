"""GUI integration checks. Run with the libero Python inside xvfb-run."""
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from PIL import Image
from unittest.mock import patch

import subgoal_gui as gui  # Configure the smooth Tk runtime before importing Tk.
import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from scipy.spatial.transform import Rotation

from waypoint_chain import Chain, Waypoint, PREVIOUS, WORLD
import subgoal_planner as sp


@unittest.skipUnless(os.environ.get("DISPLAY"), "A real or virtual display is required")
class StudioIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_store = gui.STORE
        cls.folder = tempfile.TemporaryDirectory()
        gui.STORE = Path(cls.folder.name)
        cls.original_saved = gui.SAVED
        gui.SAVED = Path(cls.folder.name)/"saved"
        cls.original = Chain.load(Path(__file__).parent / "tests/fixtures/manual_bowl_chain.json", "libero_spatial", 0)
        cls.original.save(gui.STORE / "libero_spatial/task_000.json")
        cls.errors = []
        cls.error_patch = patch.object(gui.messagebox, "showerror", side_effect=lambda title, msg, **kw: cls.errors.append(str(msg)))
        cls.error_patch.start()
        cls.root = tk.Tk()
        cls.root.report_callback_exception = lambda typ, value, tb: cls.errors.append(str(value))
        cls.panel = gui.Panel(cls.root, sp.Args(task_suite_name="libero_spatial", task_id=0))
        cls.root.update()
        cls.panel.load_task()
        cls.root.update()
        if os.environ.get("STUDIO_SCREENSHOT"):
            from PIL import ImageGrab
            root = cls.root
            ImageGrab.grab(bbox=(root.winfo_rootx(), root.winfo_rooty(),
                                root.winfo_rootx()+root.winfo_width(), root.winfo_rooty()+root.winfo_height())).save(os.environ["STUDIO_SCREENSHOT"])

    @classmethod
    def tearDownClass(cls):
        cls.panel.form_dirty = False
        cls.panel.close()
        cls.error_patch.stop()
        gui.STORE = cls.original_store
        gui.SAVED = cls.original_saved
        cls.folder.cleanup()

    def setUp(self):
        p = self.panel
        p.chain = copy.deepcopy(self.original)
        p.form_dirty = False
        p.editing_index = None
        p.redraw_rows(0)
        p.select()
        self.root.update()

    def tearDown(self):
        self.assertEqual(self.errors, [])

    def test_run_all_follows_active_waypoint_axes_and_reference(self):
        p = self.panel
        position, orientation = p.session.gripper_pose()
        p.chain.waypoints = [
            Waypoint("First", WORLD, position.tolist(), orientation.tolist()),
            Waypoint("Second", PREVIOUS, [0, 0, .04], orientation.tolist()),
        ]
        original = copy.deepcopy(p.chain)
        for outcome in ("Finished", "failed", "stopped"):
            with self.subTest(outcome=outcome):
                p.editing_index = 1
                p.redraw_rows(1)
                p.show_form(p.chain.waypoints[1])

                def playback(chain, report):
                    targets = p.session.preview(chain)
                    for index in range(len(chain.waypoints)):
                        with patch.object(p, "draw_camera_frame", wraps=p.draw_camera_frame) as draw:
                            report(index, "Running")
                        self.assertEqual(p.editing_index, index)
                        self.assertEqual(p.rows.selection(), (str(index),))
                        self.assertEqual(p.reference.get(), chain.waypoints[index].reference)
                        frame = draw.call_args_list[0].args
                        np.testing.assert_allclose(frame[2], targets[index][0], atol=1e-5)
                        np.testing.assert_allclose(frame[3], targets[index][1])
                        self.assertEqual(frame[4], p.plan_labels()[index])
                        if index:
                            reference = draw.call_args_list[1].args
                            np.testing.assert_allclose(reference[2], targets[index-1][0])
                            self.assertEqual(reference[4], "Previous waypoint 1")
                    if outcome == "stopped":
                        raise sp.Abort()
                    report(1, "ok" if outcome == "Finished" else "failed: target missed")
                    return outcome

                with patch.object(p.session, "run", side_effect=playback):
                    p.run()
                self.root.update()
                self.assertEqual(p.editing_index, 1)
                self.assertFalse(p.busy)
                self.assertFalse(p.form_dirty)
                self.assertEqual(p.chain, original)
                captions = [p.camera.itemcget(i, "text") for i in p.camera.find_all()
                            if p.camera.type(i) == "text"]
                self.assertIn(p.plan_labels()[1], captions)
                self.assertIn("Previous waypoint 1", captions)

    def test_playback_highlight_ignores_elapsed_time_selection_and_old_status(self):
        p = self.panel
        position, orientation = p.session.gripper_pose()
        p.chain.waypoints = [
            Waypoint("First", WORLD, position.tolist(), orientation.tolist()),
            Waypoint("Second", PREVIOUS, [0, 0, .04], orientation.tolist()),
        ]
        p.redraw_rows(0)

        def playback(chain, report):
            report(0, "Running")
            # Simulate a queued Treeview click during motion, then a long
            # camera interval. Neither is a start event for waypoint 2.
            p.rows.selection_set("1")
            p.select()
            with patch.object(gui.time, "monotonic", return_value=p.last_frame + 120):
                p.pump()
            self.assertEqual(p.editing_index, 0)
            self.assertEqual(p.rows.selection(), ("0",))
            with patch.object(p, "draw_camera_frame") as draw:
                p.paint_plan(480, 360)
            self.assertEqual(draw.call_args.args[4], p.plan_labels()[0])

            report(1, "Running")
            report(0, "ok (late completion)")
            self.root.update()
            self.assertEqual(p.editing_index, 1)
            self.assertEqual(p.rows.selection(), ("1",))
            self.assertEqual(p.rows.item("1", "tags"), ("running",))
            self.assertEqual(p.status.get(), "Waypoint 2: Running")
            report(1, "ok")
            return "Finished"

        with patch.object(p.session, "run", side_effect=playback):
            p.run()
        self.assertEqual(p.editing_index, 1)

    def test_running_axes_follow_the_pose_the_executor_was_given(self):
        p = self.panel
        position, orientation = p.session.gripper_pose()
        p.chain.waypoints = [Waypoint("First", WORLD, position.tolist(), orientation.tolist())]
        p.editing_index = 0
        p.redraw_rows(0)
        # A grasp adjusted for clearance, or a contact refreshed from an object
        # that moved during the run, is centimetres from the stored target and
        # is where the arm actually goes.
        commanded = position + [.04, -.03, 0]

        def drawn():
            with patch.object(p, "draw_camera_frame") as draw:
                p.paint_camera()
            return draw.call_args_list[0].args[2]

        # operation() turns an exception in here into a dialog, so the frames
        # are collected and compared once the run is over.
        seen = {}
        def playback(chain, report):
            report(0, "Running")
            seen["busy"] = p.busy
            p.session.running_target = (commanded, orientation)
            seen["live"] = drawn()
            # Nothing published for this row falls back to the stored target.
            p.session.running_target = None
            seen["stored"] = drawn()
            return "Finished"

        with patch.object(p.session, "run", side_effect=playback):
            p.run()
        self.root.update()
        self.assertTrue(seen["busy"])
        # The published pose is drawn as given; a stored one comes back through
        # the offset box, which displays centimetres to three decimals.
        np.testing.assert_allclose(seen["live"], commanded, atol=1e-9)
        np.testing.assert_allclose(seen["stored"], position, atol=1e-5)
        # Idle again, the axes belong to the editor's own target.
        self.assertIsNone(p.session.running_target)
        np.testing.assert_allclose(drawn(), position, atol=1e-5)

    def test_scene_overlay_refreshes_after_typing_offsets(self):
        p = self.panel
        with patch.object(p, "paint_camera", wraps=p.paint_camera) as paint:
            p.xyz[0].set(str(float(p.xyz[0].get()) + 2))
            self.root.after(250, self.root.quit)
            self.root.mainloop()
            self.assertGreater(paint.call_count, 0)
        with patch.object(p, "draw_camera_frame") as draw:
            p.paint_plan(480, 360)
        expected, _ = p.session.editor_target(p.chain, p.editing_index, p.read_form())
        np.testing.assert_allclose(draw.call_args.args[2], expected)

    def test_legacy_grip_row_uses_pose_motion_when_edited(self):
        p = self.panel
        start, rotation = p.session.gripper_pose()
        waypoint = Waypoint(reference=WORLD, offset=start.tolist(),
                            orientation=rotation.tolist(), motion="grip", gripper="keep")
        p.chain.waypoints = [waypoint]
        p.editing_index = 0
        p.redraw_rows(0)
        p.show_form(waypoint)
        self.assertEqual(p.read_form().motion, "line")
        before = [tuple(p.camera.coords(item)) for item in p.camera.find_withtag("target_axes")
                  if p.camera.type(item) == "line"]
        p.xyz[2].set(str(float(p.xyz[2].get())-2.))
        self.root.after(250, self.root.quit)
        self.root.mainloop()
        self.assertEqual(p.read_form().motion, "line")
        after = [tuple(p.camera.coords(item)) for item in p.camera.find_withtag("target_axes")
                 if p.camera.type(item) == "line"]
        self.assertTrue(before)
        self.assertNotEqual(after, before)
        target = p.session.editor_target(p.chain, 0, p.read_form())[0]
        np.testing.assert_allclose(target, start+[0, 0, -.02], atol=1e-5)
        p.move_button.invoke()
        self.assertLess(np.linalg.norm(p.session.world.eef()-target), .005)
        self.assertIn("Robot reached", p.status.get())

    def test_gripper_only_is_not_available_for_new_or_legacy_rows(self):
        p = self.panel
        waypoint = copy.deepcopy(p.draft)
        waypoint.motion = "grip"
        p.show_form(waypoint)
        p.xyz[0].set(p.shown_xyz[0])
        self.assertNotIn("grip", gui.MOTIONS.values())
        self.assertEqual(p.motion.get(), "Straight · allows contact")
        for value in ("", "nan", "inf"):
            p.xyz[0].set(value)
            self.assertEqual(p.motion.get(), "Straight · allows contact")
        p.xyz[0].set(p.shown_xyz[0])

    def test_relationships_show_world_previous_and_object_and_follow_edits(self):
        p = self.panel
        position, rotation = p.session.gripper_pose()
        reference = self.original.waypoints[0].reference
        p.chain.waypoints = [
            Waypoint("First", WORLD, position.tolist(), rotation.tolist()),
            Waypoint("Second", PREVIOUS, [0, 0, .03], rotation.tolist()),
            Waypoint("Contact", reference, [0, 0, .1]),
        ]
        p.editing_index = None
        p.redraw_rows(0)
        p.select()
        self.root.update()
        tabs = [p.view_tabs.tab(tab, "text") for tab in p.view_tabs.tabs()]
        self.assertEqual(tabs, ["Live camera", "Waypoint relationships"])
        table = p.relationship_rows
        self.assertEqual(len(table.get_children()), 3)
        self.assertIn("World", table.item("0", "values")[1])
        self.assertIn("Previous waypoint 01 · First", table.item("1", "values")[1])
        self.assertIn(gui.friendly_name(reference), table.item("2", "values")[1])
        table.selection_set("1")
        self.root.update()
        self.assertEqual(p.editing_index, 1)
        self.assertEqual(p.rows.selection(), ("1",))
        p.xyz[0].set("3")
        self.root.after(250, self.root.quit)
        self.root.mainloop()
        self.assertIn("X +3.00", table.item("1", "values")[1])
        self.assertEqual(table.item("1", "tags"), ("edited",))
        p.reference.set(reference)
        p.change_reference()
        self.assertIn(gui.friendly_name(reference), table.item("1", "values")[1])

    def test_relationship_selection_cancel_keeps_the_unsaved_target(self):
        p = self.panel
        p.nudge(0, .01)
        p.draw_relationships()
        with patch.object(gui.messagebox, "askyesnocancel", return_value=None):
            p.relationship_rows.selection_set("1")
            self.root.update()
        self.assertEqual(p.editing_index, 0)
        self.assertEqual(p.relationship_rows.selection(), ("0",))
        self.assertEqual(p.rows.selection(), ("0",))
        self.assertTrue(p.form_dirty)

    def test_camera_zoom_pan_and_fit_work_during_motion_without_changing_the_scene(self):
        p = self.panel
        p.reset_camera_zoom()
        state = p.session.env.sim.get_state().flatten().copy()
        original_size = p.photo.width(), p.photo.height()
        p.zoom_in_button.invoke()
        self.assertGreater(p.photo.width(), original_size[0])
        self.assertEqual(p.camera_zoom_text.get(), "125%")
        try:
            p.busy = True
            p.lock(True)
            for button in p.camera_controls:
                self.assertFalse(button.instate(["disabled"]))
            p.zoom_in_button.invoke()
            p.camera_pan_start(SimpleNamespace(x=100, y=100))
            p.camera_pan_drag(SimpleNamespace(x=140, y=135))
            self.assertGreater(np.linalg.norm(p.camera_pan), 0)
            p.camera_wheel(SimpleNamespace(num=4, delta=0, x=140, y=135))
            self.assertGreater(p.camera_zoom, 1.5)
            p.zoom_fit_button.invoke()
            self.assertEqual(p.camera_zoom, 1.)
            np.testing.assert_array_equal(p.camera_pan, [0., 0.])
            self.assertEqual((p.photo.width(), p.photo.height()), original_size)
        finally:
            p.lock(False)
            p.busy = False
        np.testing.assert_array_equal(state, p.session.env.sim.get_state().flatten())

    def test_waypoint_view_reports_measured_speed_and_restores_policy_images(self):
        p = self.panel
        domain = p.session.env.env
        images = {name: (obs.is_enabled(), obs.is_active())
                  for name, obs in domain._observables.items() if name.endswith("_image")}
        self.assertTrue(images)
        chain = copy.deepcopy(p.chain)
        previous_rate = p.session.playback_rate
        try:
            p.session.playback_rate = 1.
            sim_start = float(p.session.env.sim.data.time)
            wall_start = time.monotonic()
            def hold():
                for name in images:
                    self.assertFalse(domain._observables[name].is_enabled())
                    self.assertFalse(domain._observables[name].is_active())
                p.session.executor.hold(10)
            p.operation(hold, waypoint_playback=True)
            elapsed = time.monotonic() - wall_start
            simulated = float(p.session.env.sim.data.time) - sim_start
            measured = float(p.actual_speed.get().split()[1].rstrip("×"))
            self.assertAlmostEqual(simulated, 10*domain.control_timestep)
            self.assertAlmostEqual(measured, simulated/elapsed, delta=.05)
            self.assertEqual(p.chain, chain)
            for name, flags in images.items():
                observable = domain._observables[name]
                self.assertEqual((observable.is_enabled(), observable.is_active()), flags)
            self.assertIsNotNone(domain._get_observations()["agentview_image"])
        finally:
            p.session.playback_rate = previous_rate

    def test_single_waypoint_stop_restores_policy_images(self):
        p = self.panel
        domain = p.session.env.env
        images = {name: (obs.is_enabled(), obs.is_active())
                  for name, obs in domain._observables.items() if name.endswith("_image")}
        def stop(*args):
            for name in images:
                self.assertFalse(domain._observables[name].is_enabled())
            raise sp.Abort()
        with patch.object(p.session, "test_waypoint", side_effect=stop):
            p.test()
        self.assertFalse(p.busy)
        self.assertIn("Stopped", p.status.get())
        for name, flags in images.items():
            observable = domain._observables[name]
            self.assertEqual((observable.is_enabled(), observable.is_active()), flags)
        self.assertIsNotNone(domain._get_observations()["agentview_image"])

    def test_move_failure_names_the_selected_waypoint_and_keeps_the_planner_reason(self):
        p = self.panel
        p.rows.selection_set("2")
        p.select()
        self.root.update()
        result = "Stopped at 1: no collision-free path; target missed by 18.9 cm"
        with patch.object(p.session, "test_waypoint", return_value=result):
            p.move_button.invoke()
        self.assertEqual(p.status.get(), "Waypoint 3: no collision-free path; target missed by 18.9 cm")

    def test_previous_reference_shows_resolved_predecessor_and_start(self):
        p = self.panel
        start, _ = p.session.start_pose
        rotated = np.diag([-1., -1., 1.])
        first = Waypoint(reference=WORLD, offset=(start + [0, 0, .03]).tolist(),
                         orientation=rotated.tolist())
        second = Waypoint(reference=PREVIOUS, offset=[0, 0, .05])
        p.chain.waypoints = [first, second]
        for index, expected, label in (
                (1, np.asarray(first.offset), "Previous waypoint 1"),
                (None, np.asarray(first.offset) + [0, 0, .05], "Previous waypoint 2"),
                (0, start, "Start pose")):
            p.editing_index = index
            p.show_form(second)
            with patch.object(p, "draw_camera_frame") as draw:
                p.paint_offset_axes(480, 360)
            np.testing.assert_allclose(draw.call_args.args[2], expected)
            np.testing.assert_array_equal(draw.call_args.args[3], np.eye(3))
            self.assertEqual(draw.call_args.args[4], label)
        # A grip can occupy exactly the previous stop: retain both captions.
        p.chain.waypoints[1] = Waypoint(reference=PREVIOUS, motion="grip")
        p.editing_index = 1
        p.show_form(p.chain.waypoints[1])
        texts = [p.camera.itemcget(i, "text") for i in p.camera.find_all()
                 if p.camera.type(i) == "text"]
        self.assertIn("Previous waypoint 1", texts)
        self.assertIn(p.plan_labels()[1], texts)

    def test_reference_overlay_uses_editor_frame_instead_of_live_object(self):
        p = self.panel
        reference = p.reference.get()
        origin = np.array([.1, .2, .9])
        basis = np.diag([-1., -1., 1.])
        # This also covers predicted skill frames (e.g. an opened drawer).
        with patch.object(p, "editor_poses", return_value={reference: (origin, basis)}), \
                patch.object(p.session, "poses", side_effect=AssertionError("live frame used")), \
                patch.object(p, "draw_camera_frame") as draw:
            p.paint_offset_axes(480, 360)
        np.testing.assert_array_equal(draw.call_args.args[2], origin)
        np.testing.assert_array_equal(draw.call_args.args[3], basis)

    def test_smooth_fonts_and_visible_actions(self):
        self.assertEqual(tkfont.Font(family=self.panel.font_family).actual("family"), "DejaVu Sans")
        for size in ("1240x800", "1500x930"):
            self.root.geometry(size)
            self.root.update()
            for widget in (self.panel.save_waypoint_button, self.panel.move_button, self.panel.stop_button):
                self.assertTrue(widget.winfo_viewable())
                self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(), self.root.winfo_rooty()+self.root.winfo_height())
                self.assertLessEqual(widget.winfo_rootx()+widget.winfo_width(), self.root.winfo_rootx()+self.root.winfo_width())

    def test_centimetres_save_without_robot_motion(self):
        p = self.panel
        state = p.session.env.sim.get_state().flatten().copy()
        x = p.chain.waypoints[0].offset[0]
        p.nudge(0, .01)
        self.assertTrue(p.form_dirty)
        self.assertAlmostEqual(p.read_form().offset[0], x+.01, places=5)
        p.save_plan()
        saved = Chain.load(p.path(), p.chain.suite, p.chain.task_id)
        self.assertAlmostEqual(saved.waypoints[0].offset[0], x+.01, places=5)
        self.assertFalse(p.form_dirty)
        np.testing.assert_array_equal(state, p.session.env.sim.get_state().flatten())

    def test_saved_orientation_degrees_load_and_save_without_robot_motion(self):
        p = self.panel
        state = p.session.env.sim.get_state().flatten().copy()
        waypoint = copy.deepcopy(p.draft)
        waypoint.reference = WORLD
        waypoint.orientation = sp.DOWN.tolist()
        p.show_form(waypoint)
        np.testing.assert_allclose([float(var.get()) for var in p.rpy], [180., 0., 90.])
        np.testing.assert_array_equal(p.read_form().orientation, waypoint.orientation)
        p.nudge_rotation(0, 1)
        self.assertEqual(float(p.roll.get()), 181.)
        p.nudge_rotation(0, -1)
        p.roll.set("90")
        expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.assertTrue(p.form_dirty)
        np.testing.assert_allclose(p.read_form().orientation, expected, atol=1e-12)
        np.testing.assert_allclose(p.read_form().orientation, expected, atol=1e-12)
        p.save_plan()
        saved = Chain.load(p.path(), p.chain.suite, p.chain.task_id)
        np.testing.assert_allclose(saved.waypoints[0].orientation, expected, atol=1e-12)
        np.testing.assert_allclose([float(var.get()) for var in p.rpy], [90., 0., 90.])
        np.testing.assert_allclose(p.read_form().orientation, expected, atol=1e-12)
        np.testing.assert_array_equal(state, p.session.env.sim.get_state().flatten())

    def test_orientation_degrees_reject_invalid_angles(self):
        p = self.panel
        for axis, value in ((0, "nan"), (1, "inf"), (2, "not a number")):
            p.rpy[axis].set(value)
            with self.assertRaisesRegex(ValueError, "degrees"):
                p.read_form()
            p.rpy[axis].set(p.shown_rpy[axis])

    def test_move_robot_button_executes_orientation_degrees(self):
        p = self.panel
        p.new()
        _, before = p.session.gripper_pose()
        p.motion.set("Straight · allows contact")
        for axis, degrees in enumerate((5, -5, 10)):
            p.nudge_rotation(axis, degrees)
        target_position, target_rotation = p.session.editor_target(p.chain, None, p.read_form())
        p.move_button.invoke()
        position, rotation = p.session.gripper_pose()
        changed = Rotation.from_matrix(before.T @ rotation).magnitude()
        error = Rotation.from_matrix(np.asarray(target_rotation).T @ rotation).magnitude()
        self.assertGreater(np.degrees(changed), 5.)
        self.assertLess(np.degrees(error), 2.)
        self.assertLess(np.linalg.norm(position-target_position), .005)
        np.testing.assert_allclose(p.read_form().orientation, target_rotation, atol=1e-12)
        self.assertTrue(p.form_dirty)
        self.assertIn("Robot reached", p.status.get())

    def test_orientation_degrees_rotate_scene_axes_immediately_above_the_reference(self):
        p = self.panel
        waypoint = copy.deepcopy(p.draft)
        waypoint.offset = [0., 0., 0.]
        p.show_form(waypoint)
        self.root.update()
        def lines(tag):
            return [tuple(p.camera.coords(item)) for item in p.camera.find_withtag(tag)
                    if p.camera.type(item) == "line"]
        before_target, before_reference = lines("target_axes"), lines("reference_axes")
        self.assertTrue(before_target)
        self.assertTrue(before_reference)
        target_position = p.session.editor_target(p.chain, p.editing_index, p.read_form())[0]
        p.yaw.set(str(float(p.yaw.get())+45.))
        self.root.update()
        self.assertNotEqual(lines("target_axes"), before_target)
        self.assertEqual(lines("reference_axes"), before_reference)
        np.testing.assert_allclose(p.session.editor_target(p.chain, p.editing_index, p.read_form())[0],
                                   target_position, atol=1e-12)
        order = {item:index for index,item in enumerate(p.camera.find_all())}
        self.assertGreater(min(order[item] for item in p.camera.find_withtag("target_axes")),
                           max(order[item] for item in p.camera.find_withtag("reference_axes")))

    def test_unchanged_display_preserves_arbitrary_and_singular_saved_rotations(self):
        p = self.panel
        for angles in ([21.123456, -32.654321, 43.987654], [30., 90., 20.]):
            waypoint = copy.deepcopy(p.draft)
            waypoint.orientation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix().tolist()
            p.show_form(waypoint)
            np.testing.assert_array_equal(p.read_form().orientation, waypoint.orientation)

    def test_previous_relative_orientation_shows_the_full_scene_pose(self):
        p = self.panel
        world_rotation = Rotation.from_euler("xyz", [25., -35., 40.], degrees=True).as_matrix()
        first = Waypoint(reference=WORLD, offset=p.session.start_pose[0].tolist(),
                         orientation=world_rotation.tolist())
        second = Waypoint(reference=PREVIOUS, orientation=np.eye(3).tolist(),
                          orientation_relative_to_previous=True, motion="grip")
        p.chain.waypoints = [first, second]
        p.editing_index = 1
        p.show_form(second)
        np.testing.assert_allclose([float(var.get()) for var in p.rpy], [25., -35., 40.], atol=1e-3)
        np.testing.assert_array_equal(p.read_form().orientation, np.eye(3))
        p.yaw.set("70")
        self.root.update()
        self.assertEqual(p.read_form().motion, "line")
        _, rotation = p.session.editor_target(p.chain, 1, p.read_form())
        np.testing.assert_allclose(Rotation.from_matrix(rotation).as_euler("xyz", degrees=True),
                                   [25., -35., 70.], atol=1e-6)

    def test_object_relative_orientation_edits_use_world_angles(self):
        p = self.panel
        waypoint = copy.deepcopy(p.draft)
        reference = waypoint.reference
        poses = copy.deepcopy(p.editor_poses())
        world_rotation = Rotation.from_euler("xyz", [25., -35., 40.], degrees=True).as_matrix()
        poses[reference] = (poses[reference][0], Rotation.from_euler("z", 35., degrees=True).as_matrix())
        waypoint.orientation = (poses[reference][1].T @ world_rotation).tolist()
        with patch.object(p, "editor_poses", return_value=poses), \
                patch.object(p.session, "reference_poses", poses):
            p.show_form(waypoint)
            np.testing.assert_allclose([float(var.get()) for var in p.rpy], [25., -35., 40.], atol=1e-3)
            p.pitch.set("-20")
            edited = p.read_form()
            _, rotation = p.session.editor_target(p.chain, p.editing_index, edited)
            np.testing.assert_allclose(Rotation.from_matrix(rotation).as_euler("xyz", degrees=True),
                                       [25., -20., 40.], atol=1e-6)

    def test_move_robot_button_executes_draft(self):
        p = self.panel
        p.new()
        before = p.session.world.eef().copy()
        p.nudge(2, -.02)
        p.motion.set("Straight · allows contact")
        p.move_button.invoke()
        after = p.session.world.eef()
        self.assertGreater(np.linalg.norm(after-before), .005)
        self.assertLess(np.linalg.norm(after-(before+[0, 0, -.02])), .015)
        self.assertIn("Robot reached", p.status.get())
        self.assertTrue(p.form_dirty)
        self.assertEqual(len(p.chain.waypoints), len(self.original.waypoints))

    def test_repeated_move_button_uses_same_previous_target(self):
        p = self.panel
        p.session.refresh_references()
        before, ori = p.session.gripper_pose()
        w = Waypoint(reference=PREVIOUS, offset=[0, 0, -.02], orientation=ori.tolist(), motion="line")
        p.chain.waypoints = [w]
        p.editing_index = 0
        p.redraw_rows(0)
        p.show_form(w)
        p.move_button.invoke()
        self.assertLess(np.linalg.norm(p.session.world.eef()-(before+[0, 0, -.02])), .002)
        state = p.session.env.sim.get_state().flatten().copy()
        for _ in range(3):
            p.move_button.invoke()
        np.testing.assert_array_equal(p.session.env.sim.get_state().flatten(), state)

    def test_xy_edit_overrides_vertical_template_lock(self):
        p = self.panel
        p.editing_index = 2
        p.show_form(p.chain.waypoints[2])
        self.assertTrue(p.hold_xy.get())
        p.xyz[0].set("2.0")
        self.assertFalse(p.read_form().hold_xy)
        self.assertEqual(p.read_form().offset[0], .02)

    def test_selection_cancel_and_discard(self):
        p = self.panel
        old = p.chain.waypoints[0].offset.copy()
        p.nudge(0, .01)
        with patch.object(gui.messagebox, "askyesnocancel", return_value=None):
            p.rows.selection_set("1")
            p.select()
            self.root.update()
        self.assertEqual(p.editing_index, 0)
        self.assertTrue(p.form_dirty)
        with patch.object(gui.messagebox, "askyesnocancel", return_value=False):
            p.rows.selection_set("1")
            p.select()
            self.root.update()
        self.assertEqual(p.editing_index, 1)
        self.assertFalse(p.form_dirty)
        self.assertEqual(p.chain.waypoints[0].offset, old)

    def test_delete_loads_successor_before_saving(self):
        p = self.panel
        expected = p.chain.waypoints[1].name
        p.delete()
        self.root.update()
        self.assertEqual(p.name.get(), expected)
        p.save_waypoint()
        self.assertEqual(p.chain.waypoints[0].name, expected)
        self.assertEqual(len(p.chain.waypoints), len(self.original.waypoints)-1)

    def test_nine_waypoint_limit_does_not_mutate_plan(self):
        p = self.panel
        p.chain.waypoints = [copy.deepcopy(p.chain.waypoints[0]) for _ in range(9)]
        p.redraw_rows(0)
        with self.assertRaisesRegex(ValueError, "9 waypoints"):
            p.duplicate()
        self.assertEqual(len(p.chain.waypoints), 9)
        with self.assertRaisesRegex(ValueError, "9 waypoints"):
            p.new()

    def test_gripper_on_arrival_round_trip(self):
        p = self.panel
        p.grip_after.set("close")
        p.save_waypoint()
        saved = Chain.load(p.path(), p.chain.suite, p.chain.task_id)
        self.assertEqual(saved.waypoints[0].gripper_after, "close")
        self.assertEqual(p.grip_after_display.get(), "Close")

    def test_task_picker_filters_status_and_loads_another_suite(self):
        p = self.panel
        p.change_task_button.invoke()
        self.root.update()
        self.assertTrue(p.task_dialog.winfo_viewable())
        self.assertEqual(len(p.task_picker_rows.get_children()), 10)
        p.task_filter_suite.set(gui.SUITE_LABELS["libero_90"])
        p.task_filter_search.set(" ")
        self.assertEqual(len(p.task_picker_rows.get_children()), 90)
        p.task_filter_suite.set(gui.SUITE_LABELS["libero_spatial"])
        p.task_filter_search.set("")
        # The saved filter lists exactly the tasks with a kept plan, whatever
        # else has run before this test.
        p.task_filter_ready.set(True)
        listed = p.task_picker_rows.get_children()
        self.assertTrue(all(p.task_saved("libero_spatial", int(row)) for row in listed))
        self.assertEqual(len(listed),
                         sum(p.task_saved("libero_spatial", i) for i in range(10)))
        p.task_filter_ready.set(False)
        p.task_filter_suite.set(gui.SUITE_LABELS["libero_object"])
        p.task_filter_search.set("alphabet")
        self.root.update()
        rows = p.task_picker_rows.get_children()
        self.assertEqual(len(rows), 1)
        p.task_picker_rows.selection_set(rows[0])
        p.load_selected_task_button.invoke()
        self.root.update()
        self.assertEqual(p.chain.suite, "libero_object")
        self.assertIn("alphabet", p.chain.language)
        p.form_dirty = False
        p.suite.set("libero_spatial")
        p.task_id = 0
        p.load_task()

    def test_next_task_cancel_keeps_scene_and_edited_waypoint(self):
        p = self.panel
        scene = p.session.env
        p.nudge(0, .01)
        with patch.object(gui.messagebox, "askyesnocancel", return_value=None):
            p.next_task_button.invoke()
        self.assertIs(p.session.env, scene)
        self.assertEqual(p.task_id, 0)
        self.assertEqual(p.chain.suite, "libero_spatial")
        self.assertTrue(p.form_dirty)

    def test_visible_learn_button_replaces_plan_with_inferred_result(self):
        p = self.panel
        from test_waypoint_library import DemonstrationTest
        from waypoint_library import choose_demonstration
        demo = DemonstrationTest().phased_demo()
        learned, _ = choose_demonstration([demo, demo])
        self.assertTrue(p.learn_button.winfo_viewable())
        p.learn_repeats.set("4")
        p.learn_waypoints.set("6")
        with patch.object(p.session, "record_policy", return_value=learned) as generate:
            p.learn_button.invoke()
        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["repeats"], 4)
        self.assertEqual(generate.call_args.kwargs["waypoint_count"], 6)
        p.learn_repeats.set("2")
        p.learn_waypoints.set("Auto")
        self.assertEqual(len(p.chain.waypoints), 6)
        self.assertEqual(p.chain.waypoints[1].name, "Reach & close gripper")
        saved = Chain.load(p.path(), p.chain.suite, p.chain.task_id)
        self.assertEqual(saved, learned)
        self.assertIn("replay needed", p.plan_status.get())


@unittest.skipUnless(os.environ.get("DISPLAY"), "A real or virtual display is required")
class CameraProjectionTest(unittest.TestCase):
    def test_projection_matches_rendered_landmark_after_resizing(self):
        from robosuite.utils.binding_utils import MjSim, MjRenderContextOffscreen
        from waypoint_session import Session
        sim = MjSim.from_xml_string("""
            <mujoco><worldbody>
              <light pos="0 0 3"/>
              <camera name="agentview" pos="0 0 2" quat=".9659258 0 0 .258819" fovy="45"/>
              <geom type="sphere" pos=".2 .12 0" size=".025" rgba="1 0 0 1"/>
            </worldbody></mujoco>""")
        context = MjRenderContextOffscreen(sim, device_id=-1)
        try:
            session = SimpleNamespace(env=SimpleNamespace(sim=sim))
            source = Image.fromarray(Session.image(session))
            panel = gui.Panel.__new__(gui.Panel)
            panel.session = session
            panel.camera_image = source
            panel.camera = SimpleNamespace(winfo_width=lambda: 803, winfo_height=lambda: 601)
            for width, height, pan in ((480, 360, [0, 0]), (801, 600, [0, 0]),
                                      (320, 240, [0, 0]), (1602, 1200, [80, -40])):
                panel.camera_pan = np.asarray(pan)
                pixels = np.asarray(source.resize((width, height)))
                mask = (pixels[:, :, 0] > 100) & (pixels[:, :, 1] < 70) & (pixels[:, :, 2] < 70)
                y, x = np.nonzero(mask)
                self.assertGreater(len(x), 0)
                visible = np.array([x.mean() + .5 + (803-width)/2,
                                    y.mean() + .5 + (601-height)/2]) + pan
                projected = panel.camera_view(width, height)([.2, .12, 0])
                np.testing.assert_allclose(projected, visible, atol=1.)
        finally:
            sim.free()


if __name__ == "__main__":
    unittest.main()
