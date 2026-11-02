"""Batch learning autosaves drafts; only an explicit save confirms a kept plan."""
import dataclasses
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import subgoal_gui as gui
from waypoint_chain import Chain, Waypoint
from waypoint_gmm_editor import GMMPanel


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class BatchLearningTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.store = Path(folder.name) / "autosaved"
        self.saved = Path(folder.name) / "saved"
        for name, path in (("STORE", self.store), ("SAVED", self.saved)):
            patcher = patch.object(gui, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)
        p = self.panel = gui.Panel.__new__(gui.Panel)
        p.root = Mock()
        p.busy = p.stopped = p.closing = p.form_dirty = False
        p.last_frame = time.monotonic()
        p.chain = Chain("libero_goal", 7, "Original task")
        p.suite = Value("libero_goal")
        p.task_id = 7
        p.learn_repeats, p.learn_waypoints = Value("3"), Value("6")
        p.status, p.robot_state, p.save_state = Value(""), Value(""), Value("")
        p.rows, p.lock, p.refresh, p.show_task = Mock(), Mock(), Mock(), Mock()
        p.rows.get_children.return_value = ()
        p.leave_draft = Mock(return_value=True)
        p.session = Mock()
        p.session.load.side_effect = lambda suite, task: Chain(suite, task, f"Task {task}")
        p.learn_chain = Mock(side_effect=lambda chain, **opts: dataclasses.replace(
            chain, waypoints=[Waypoint("Learned target")], source="pi0.5; replay unverified"))
        patcher = patch("libero.libero.benchmark.get_benchmark_dict", return_value={
            "libero_spatial": lambda: Mock(n_tasks=3)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_batch_autosaves_drafts_and_preserves_explicitly_kept_plan(self):
        p = self.panel
        old = Chain("libero_spatial", 0, "Old plan", [Waypoint("Old target")])
        old.save(self.store / "libero_spatial/task_000.json")
        p.chain = old
        p.save_plan()
        p.save_plan = Mock(wraps=p.save_plan)

        def learn(chain, **options):
            self.assertTrue(p.busy)
            self.assertEqual(options, dict(repeats=3, waypoint_count=6))
            return dataclasses.replace(chain, waypoints=[Waypoint("Learned target")])

        p.learn_chain.side_effect = learn
        p.learn_all_tasks("libero_spatial")
        self.assertEqual(p.session.load.call_count, 3)
        self.assertEqual(p.learn_chain.call_count, 3)
        for task in range(3):
            chain = Chain.load(self.store / f"libero_spatial/task_{task:03d}.json", "libero_spatial", task)
            self.assertEqual(chain.waypoints[0].name, "Learned target")
        backup = Chain.load(self.store / "libero_spatial/task_000.backup.json", "libero_spatial", 0)
        self.assertEqual(backup, old)
        kept = Chain.load(self.saved / "libero_spatial/task_000.json", "libero_spatial", 0)
        self.assertEqual(kept, old)
        self.assertTrue(p.task_saved("libero_spatial", 0))
        for task in (1, 2):
            self.assertFalse((self.saved / f"libero_spatial/task_{task:03d}.json").exists())
            self.assertFalse(p.task_saved("libero_spatial", task))
        p.save_plan.assert_not_called()
        self.assertIn("learned 3/3", p.status.get())
        self.assertEqual((p.chain.suite, p.chain.task_id), ("libero_spatial", 2))
        self.assertFalse(p.busy)
        self.assertEqual([call.args for call in p.lock.call_args_list], [(True,), (False,)])

    def test_failed_learning_keeps_existing_plan_and_continues(self):
        p = self.panel
        old = Chain("libero_spatial", 1, "Existing plan", [Waypoint("Existing target")])
        old.save(self.store / "libero_spatial/task_001.json")
        old.save(self.saved / "libero_spatial/task_001.json")

        def learn(chain, **options):
            if chain.task_id == 1:
                raise ValueError("No successful demonstrations")
            return dataclasses.replace(chain, waypoints=[Waypoint("Learned target")])

        p.learn_chain.side_effect = learn
        p.learn_all_tasks("libero_spatial")
        self.assertEqual(p.learn_chain.call_count, 3)
        for store in (self.store, self.saved):
            self.assertEqual(Chain.load(store / "libero_spatial/task_001.json", "libero_spatial", 1), old)
        self.assertTrue((self.store / "libero_spatial/task_002.json").exists())
        self.assertFalse((self.saved / "libero_spatial/task_002.json").exists())
        self.assertIn("learned 2/3", p.status.get())
        self.assertIn("02 (No successful demonstrations)", p.status.get())

    def test_stop_keeps_completed_plans_and_does_not_start_next_task(self):
        p = self.panel

        def learn(chain, **options):
            if chain.task_id == 1:
                p.stop()
                p.pump()
            return dataclasses.replace(chain, waypoints=[Waypoint("Learned target")])

        p.learn_chain.side_effect = learn
        p.learn_all_tasks("libero_spatial")
        self.assertEqual(p.session.load.call_count, 2)
        self.assertTrue((self.store / "libero_spatial/task_000.json").exists())
        self.assertFalse((self.store / "libero_spatial/task_001.json").exists())
        self.assertFalse(self.saved.exists())
        self.assertIn("Stopped: learned 1/3", p.status.get())
        self.assertFalse(p.busy)

    def test_stop_after_fitting_does_not_save_unfinished_task(self):
        p = self.panel

        def learn(chain, **options):
            p.stop()
            return dataclasses.replace(chain, waypoints=[Waypoint("Learned target")])

        p.learn_chain.side_effect = learn
        p.learn_all_tasks("libero_spatial")
        self.assertEqual(p.session.load.call_count, 1)
        self.assertFalse(self.saved.exists())
        self.assertIn("Stopped: learned 0/3", p.status.get())

    def test_scene_load_failure_does_not_learn_the_previous_task(self):
        p = self.panel
        p.session.load.side_effect = [
            Chain("libero_spatial", 0, "First"),
            ValueError("Scene could not load"),
            Chain("libero_spatial", 2, "Third"),
        ]
        p.learn_all_tasks("libero_spatial")
        self.assertEqual([call.args[0].task_id for call in p.learn_chain.call_args_list], [0, 2])
        self.assertIn("learned 2/3", p.status.get())
        self.assertIn("02 (Scene could not load)", p.status.get())

    def test_invalid_settings_are_checked_before_changing_tasks(self):
        p = self.panel
        p.learn_repeats.set("0")
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            p.learn_all_tasks("libero_spatial")
        p.session.load.assert_not_called()
        self.assertEqual(p.suite.get(), "libero_goal")

    def test_cancelled_draft_keeps_current_task(self):
        p = self.panel
        p.leave_draft.return_value = False
        p.learn_all_tasks("libero_spatial")
        p.session.load.assert_not_called()
        p.lock.assert_not_called()
        self.assertEqual(p.task_id, 7)

    def test_old_automatically_saved_files_do_not_confirm_a_plan(self):
        p = self.panel
        old = Chain("libero_spatial", 0, "Previously auto-saved", [Waypoint("Target")])
        old.save(self.saved / "libero_spatial/task_000.json")
        self.assertFalse(p.task_saved("libero_spatial", 0))
        self.assertTrue((self.saved / "libero_spatial/task_000.json").exists())

    def test_only_explicit_save_confirms_the_selected_plan(self):
        p = self.panel
        p.learn_all_tasks("libero_spatial")
        for task in range(3):
            self.assertFalse(p.task_saved("libero_spatial", task))
        p.save_plan()
        self.assertTrue(p.task_saved("libero_spatial", 2))
        self.assertFalse(p.task_saved("libero_spatial", 0))
        self.assertFalse(p.task_saved("libero_spatial", 1))
        p.saved_path().write_text(p.saved_path().read_text() + "\n")
        self.assertFalse(p.task_saved("libero_spatial", 2))

    def test_gmm_settings_share_single_task_and_batch_validation(self):
        p = GMMPanel.__new__(GMMPanel)
        p.learn_repeats = Value("4")
        self.assertEqual(p.learning_options(), dict(successes_needed=4))
        p.learn_repeats.set("1")
        with self.assertRaisesRegex(ValueError, "between 2 and 100"):
            p.learning_options()


if __name__ == "__main__":
    unittest.main()
