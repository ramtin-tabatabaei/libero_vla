"""Both tabs share the full editor; option 2 learns and edits GMM/GMR plans."""
import copy
import dataclasses
import os
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import subgoal_gui as gui
import waypoint_gmm_editor as gmm
import tkinter as tk
import numpy as np
import subgoal_planner as sp
from waypoint_gmm import GMMChain, GMMWaypoint, chain_from_model
from waypoint_learning import load_model
from waypoint_ui import POLICY_LABELS, GREEN, MUTED, friendly_name


@unittest.skipUnless(os.environ.get('DISPLAY'), 'A real or virtual display is required')
class GMMEditorGuiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.old_store, cls.old_gmm_store = gui.STORE, gmm.GMM_STORE
        gui.STORE = Path(cls.folder.name)/'legacy'
        gmm.GMM_STORE = Path(cls.folder.name)/'gmm'
        cls.original_saved = gmm.SAVED_STORE
        gmm.SAVED_STORE = Path(cls.folder.name)/'saved'
        cls.errors = []
        cls.error_patch = patch.object(gui.messagebox, 'showerror', side_effect=lambda title,msg,**kw: cls.errors.append(str(msg)))
        cls.error_patch.start()
        cls.root = tk.Tk()
        cls.root.report_callback_exception = lambda typ,value,tb: cls.errors.append(str(value))
        cls.app = gui.StudioApplication(cls.root, sp.Args(task_suite_name='libero_goal', task_id=0))
        cls.root.update()
        cls.panel, = cls.app.panels
        cls.panel.session.realtime = False
        # The application loads the first task on a timer; tests cannot wait.
        cls.app.changed()
        cls.root.update()
        cls.model = load_model(Path(__file__).parent/'waypoint_models/cached_goal0_smoke/model.json')
        cls.panel.adopt_model(cls.model, Path(__file__).parent/'waypoint_models/cached_goal0_smoke/model.json')
        cls.original = copy.deepcopy(cls.panel.chain)


    @classmethod
    def tearDownClass(cls):
        for p in cls.app.panels:p.form_dirty=False
        cls.app.close()
        cls.error_patch.stop()
        gui.STORE, gmm.GMM_STORE = cls.old_store, cls.old_gmm_store
        cls.folder.cleanup()

    def setUp(self):
        self.p = self.panel
        self.p.learned_model = self.model
        self.p.learn_repeats.set("5")
        self.errors.clear()
        self.p.chain = copy.deepcopy(self.original)
        self.p.form_dirty = False
        self.p.editing_index = None
        self.p.policy_choice.set(POLICY_LABELS["pi05"])
        self.p.change_policy()
        self.p.redraw_rows(0)
        self.p.select()
        self.root.update()

    def tearDown(self):
        self.assertEqual(self.errors, [])
        self.assertFalse(self.p.busy)

    def test_same_controls_and_editable_adaptive_plan(self):
        self.assertEqual(len(self.app.panels), 1)
        for size in ('1240x840','1500x970'):
            self.root.geometry(size);self.root.update()
            for attr in ('change_task_button','save_waypoint_button','move_button','learn_button','stop_button','shuffle_button','random_start_check','random_start_button','policy_selector','run_policy_button'):
                widget=getattr(self.p,attr)
                self.assertTrue(widget.winfo_viewable(),attr)
                self.assertLessEqual(widget.winfo_rooty()+widget.winfo_height(),self.root.winfo_rooty()+self.root.winfo_height(),attr)
            self.assertGreaterEqual(self.p.random_start_check.winfo_width(), self.p.random_start_check.winfo_reqwidth())
        self.assertEqual(len(self.p.rows.get_children()),len(self.original.waypoints))
        self.assertEqual(self.p.learn_repeats.get(),'5')
        self.assertIn('VLA',self.p.learn_button.cget('text'))
        state=self.p.session.env.sim.get_state().flatten().copy()
        before=self.p.chain.waypoints[0].offset[0]
        metadata=copy.deepcopy(self.p.chain.waypoints[0].learned)
        self.p.nudge(0,.01);self.p.save_plan()
        saved=GMMChain.load(self.p.path(),self.p.chain.suite,self.p.chain.task_id)
        self.assertAlmostEqual(saved.waypoints[0].offset[0],before+.01,places=5)
        self.assertEqual(saved.waypoints[0].learned['position_covariance'],metadata['position_covariance'])
        self.assertTrue(saved.waypoints[0].learned['edited'])
        np.testing.assert_array_equal(state,self.p.session.env.sim.get_state().flatten())
        self.assertNotEqual(self.p.path(),self.p.saved_path())
        self.p.duplicate();self.assertEqual(len(self.p.chain.waypoints),len(self.original.waypoints)+1)
        self.p.delete();self.assertEqual(len(self.p.chain.waypoints),len(self.original.waypoints))
        self.p.new();self.p.save_waypoint();self.assertEqual(len(self.p.chain.waypoints),len(self.original.waypoints)+1)

    def test_run_policy_button_preserves_plan_and_skips_learning(self):
        chain = copy.deepcopy(self.p.chain)
        model = copy.deepcopy(self.p.learned_model)
        state = self.p.session.env.sim.get_state().flatten().copy()
        self.p.xyz[0].set("3.125")
        self.p.form_dirty = True
        with patch.object(self.p.session, 'run_policy', return_value={'success': True, 'steps': 42}) as run, \
             patch.object(self.p, 'learn') as learn, patch.object(self.p, 'fit') as fit:
            self.p.run_policy_button.invoke()
        run.assert_called_once_with(chain.language, report=self.p.status.set, backend='pi05')
        learn.assert_not_called()
        fit.assert_not_called()
        self.assertEqual(self.p.chain, chain)
        self.assertEqual(self.p.learned_model, model)
        self.assertEqual(self.p.xyz[0].get(), "3.125")
        self.assertTrue(self.p.form_dirty)
        np.testing.assert_array_equal(state, self.p.session.env.sim.get_state().flatten())
        self.assertIn('task succeeded after 42 steps', self.p.status.get())

    def test_orientation_degrees_are_saved_in_learned_plan_without_robot_motion(self):
        state = self.p.session.env.sim.get_state().flatten().copy()
        original = self.p.session.editor_target(self.p.chain, self.p.editing_index, self.p.draft)[1]
        self.p.nudge_rotation(1, 7)
        self.assertTrue(self.p.draft.learned['edited'])
        self.p.save_plan()
        saved = GMMChain.load(self.p.path(), self.p.chain.suite, self.p.chain.task_id)
        rotation = np.asarray(saved.waypoints[0].orientation)
        world_rotation = self.p.session.editor_target(saved, 0, saved.waypoints[0])[1]
        angle = np.degrees(np.arccos(np.clip((np.trace(np.asarray(original).T @ world_rotation)-1)/2, -1, 1)))
        # Editing uses the angles displayed to 0.001 degree precision.
        self.assertAlmostEqual(angle, 7., places=3)
        self.assertTrue(any(float(var.get()) != 0. for var in self.p.rpy))
        np.testing.assert_allclose(self.p.read_form().orientation, rotation, atol=1e-12)
        np.testing.assert_array_equal(state, self.p.session.env.sim.get_state().flatten())

    def test_policy_selection_runs_inference_and_preserves_the_plan(self):
        chain = copy.deepcopy(self.p.chain)
        state = self.p.session.env.sim.get_state().flatten().copy()
        self.p.policy_choice.set(POLICY_LABELS['pi05'])
        self.p.policy_selector.event_generate('<<ComboboxSelected>>')
        self.root.update()
        self.assertIn('Run Pi0.5', self.p.run_policy_button.cget('text'))
        with patch.object(self.p.session, 'run_policy', return_value={'success': True, 'steps': 137}) as run, \
             patch.object(self.p, 'learn') as learn, patch.object(self.p, 'fit') as fit:
            def invoke(*args, **kwargs):
                self.assertIn('disabled', self.p.policy_selector.state())
                self.assertNotIn('disabled', self.p.stop_button.state())
                return {'success': True, 'steps': 137}
            run.side_effect = invoke
            self.p.run_policy_button.invoke()
        run.assert_called_once_with(chain.language, report=self.p.status.set, backend='pi05')
        learn.assert_not_called()
        fit.assert_not_called()
        self.assertEqual(self.p.chain, chain)
        np.testing.assert_array_equal(state, self.p.session.env.sim.get_state().flatten())
        self.assertIn('Pi0.5: task succeeded after 137 steps', self.p.status.get())
        self.assertNotIn('disabled', self.p.policy_selector.state())

    def test_shuffle_preserves_waypoints_and_draft(self):
        chain = copy.deepcopy(self.p.chain)
        self.p.nudge(0, .01)
        draft = self.p.xyz[0].get()
        with patch.object(self.p.session, 'shuffle_scene', return_value={'source': 'other'}) as shuffle:
            self.p.shuffle_button.invoke()
            shuffle.assert_called_once_with()
        self.assertIn('New shuffled layout', self.p.status.get())
        self.assertEqual(self.p.chain, chain)
        self.assertEqual(self.p.xyz[0].get(), draft)
        self.assertTrue(self.p.form_dirty)

    def test_random_start_controls_preserve_plan_and_draft(self):
        chain = copy.deepcopy(self.p.chain)
        self.p.nudge(0, .01)
        draft = self.p.xyz[0].get()
        with patch.object(self.p, 'apply_random_robot_start') as randomize:
            self.p.random_start_button.invoke()
        randomize.assert_called_once_with()
        self.assertIn('New random robot start', self.p.status.get())
        self.assertEqual(self.p.chain, chain)
        self.assertEqual(self.p.xyz[0].get(), draft)
        self.assertTrue(self.p.form_dirty)
        try:
            for enabled in (True, False):
                self.p.random_robot_start.set(enabled)
                with patch.object(self.p.session, 'reset') as reset, \
                     patch.object(self.p, 'apply_random_robot_start') as randomize:
                    self.p.reset()
                reset.assert_called_once_with()
                self.assertEqual(randomize.call_count, int(enabled))
        finally:
            self.p.random_robot_start.set(False)

    def test_vla_coverage_observes_steps_and_preserves_plan_and_unsaved_draft(self):
        p = self.p
        position, orientation = p.session.gripper_pose()
        p.chain = GMMChain(p.chain.suite, p.chain.task_id, p.chain.language, [
            GMMWaypoint('Near start', 'World', position.tolist(), orientation.tolist()),
            GMMWaypoint('Far target', 'World', (position+[.5, 0, 0]).tolist(), orientation.tolist())])
        chain = copy.deepcopy(p.chain)
        p.redraw_rows(0)
        p.show_form(p.chain.waypoints[0])
        p.xyz[0].set('12.345')
        p.form_dirty = True
        p.policy_choice.set(POLICY_LABELS['pi05'])
        p.change_policy()
        original_step = p.session.env.step
        def run(*args, **kwargs):
            p.session.executor.hold(3)
            return {'success': False, 'steps': 3}
        with patch.object(p.session, 'run_policy', side_effect=run) as policy:
            p.run_policy_button.invoke()
        policy.assert_called_once_with(chain.language, report=p.status.set, backend='pi05')
        result = p.policy_coverage.monitor.report()
        self.assertEqual((result['steps'], result['visited'], result['ordered_prefix']), (3, 1, 1))
        self.assertEqual(result['rows'][1]['status'], 'Not visited')
        self.assertFalse(result['task_success'])
        events = result['failure_timeline']['events']
        unfinished = next(e for e in events if e['kind'] == 'unfinished')
        self.assertAlmostEqual(unfinished['onset_sim_seconds'], .15)
        self.assertEqual(unfinished['control_step'], 3)
        self.assertEqual(p.session.env.step, original_step)
        self.assertEqual(p.chain, chain)
        self.assertEqual(p.xyz[0].get(), '12.345')
        self.assertTrue(p.form_dirty)
        self.assertEqual(len(p.policy_coverage.table.get_children()), 2)
        self.assertIn('pose_visited', p.rows.item('0', 'tags'))
        self.assertIn('pose_missed', p.rows.item('1', 'tags'))

    def test_vla_coverage_stop_keeps_partial_report_and_restores_step(self):
        p = self.p
        original_step = p.session.env.step
        def stop(*args, **kwargs):
            p.session.executor.hold(1)
            raise sp.Abort()
        with patch.object(p.session, 'run_policy', side_effect=stop):
            p.run_policy_button.invoke()
        self.assertEqual(p.policy_coverage.monitor.state, 'stopped')
        self.assertEqual(p.policy_coverage.monitor.report()['steps'], 1)
        self.assertEqual(p.session.env.step, original_step)
        self.assertIn('Stopped', p.status.get())
        self.assertNotIn('unfinished', {e['kind'] for e in p.policy_coverage.monitor.report()['failure_timeline']['events']})

    def test_vla_coverage_can_be_disabled_and_rejects_invalid_tolerance(self):
        view = self.p.policy_coverage
        try:
            view.enabled.set(False)
            view.watch_failures.set(False)
            with patch.object(self.p.session, 'run_policy', return_value={'success': True, 'steps': 8}) as run:
                self.p.run_policy_button.invoke()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(view.summary.get(), 'Waypoint checking is off.')
            view.enabled.set(True)
            view.centimetres.set('0')
            with self.assertRaisesRegex(ValueError, 'distance tolerance'):
                view.prepare(self.p.session, self.p.chain, 'pi05')
        finally:
            view.enabled.set(True)
            view.watch_failures.set(True)
            view.centimetres.set('5')

    def test_failure_times_work_without_a_waypoint_plan_and_can_be_saved(self):
        p = self.p
        view = p.policy_coverage
        p.chain = GMMChain(p.chain.suite, p.chain.task_id, p.chain.language, [])
        original = p.session.env.sim.step
        try:
            view.enabled.set(False)
            def policy(*args, **kwargs):
                p.session.executor.hold(2)
                return {'success': False, 'steps': 2}
            with patch.object(p.session, 'run_policy', side_effect=policy):
                p.run_policy_button.invoke()
            monitor = view.monitor
            result = monitor.report()
            self.assertFalse(result['checking_waypoints'])
            self.assertEqual(result['rows'], [])
            self.assertFalse(result['failure_timeline']['off_plan_available'])
            self.assertEqual(p.session.env.sim.step, original)
            self.assertTrue(view.event_table.get_children())
            path = Path(self.folder.name)/'failure_timeline.csv'
            monitor.save_failures_csv(path)
            self.assertIn('Task not finished', path.read_text())
            path = path.with_suffix('.json')
            monitor.save_json(path)
            self.assertEqual(json.loads(path.read_text())['failure_timeline']['assessment'], 'step limit exhausted')
        finally:
            view.enabled.set(True)

    def test_failure_settings_reject_invalid_values_before_policy_execution(self):
        view = self.p.policy_coverage
        try:
            view.force_n.set('nan')
            with self.assertRaisesRegex(ValueError, 'failure timing settings'):
                view.prepare(self.p.session, self.p.chain, 'pi05')
        finally:
            view.force_n.set('10')

    def test_robust_replay_restores_policy_cameras_after_stop(self):
        from test_waypoint_learning import placed
        from waypoint_skill_editor import fit_editor_model
        model = fit_editor_model([placed(), placed()])
        chain = chain_from_model(model)
        domain = self.p.session.env.env
        before = {n: (o.is_enabled(), o.is_active()) for n,o in domain._observables.items() if n.endswith('_image')}
        self.assertTrue(before)
        def stop(*args, **kwargs):
            for n in before:
                self.assertFalse(domain._observables[n].is_enabled())
                self.assertFalse(domain._observables[n].is_active())
            raise sp.Abort()
        with patch('waypoint_skill.run_skill', side_effect=stop), patch.object(gmm, 'MODEL_STORE', Path(self.folder.name)/'logs'):
            with self.assertRaises(sp.Abort):
                self.p.session.run(chain, lambda *_: None)
        for n,flags in before.items():
            self.assertEqual((domain._observables[n].is_enabled(),domain._observables[n].is_active()), flags)
        self.assertTrue(self.p.session.last_replay_report.exists())
        self.assertFalse(json.loads(self.p.session.last_replay_report.read_text())['completed'])
        self.assertIsNotNone(domain._get_observations()['agentview_image'])

    def test_learning_button_uses_multi_demo_model(self):
        from test_waypoint_learning import demo
        demos=[demo(),demo(.01),demo(-.01)]
        # Real numerical fit through the editor, using synthetic task-labelled data.
        for d in demos:
            d['suite']=self.p.chain.suite;d['task_id']=self.p.chain.task_id
            for sample in d['samples']+[d['terminal_sample']]:
                sample['task_success']=bool(sample['task_success'])
        folder=Path(self.folder.name)/'fit'
        poses={n:(np.array(p),np.array(r)) for n,(p,r) in demos[0]['poses'].items()}
        with patch.object(self.p.session,'preview',return_value=[self.p.session.gripper_pose()]):
            chain=self.p.fit(demos,folder)
        self.assertIsInstance(chain,GMMChain)
        self.assertIn('minimal single-reference waypoints',chain.source)
        self.assertTrue((folder/'model.json').exists())
        self.assertTrue(chain.execution_skill)
        # move up, move across, grasp, lift, traverse, place, release
        self.assertEqual(len(chain.waypoints),7)
        # The main Learn button routes to recording successes then fitting, not baseline selection.
        collected=[]
        def record(session,language,**kwargs):
            d=json.loads(json.dumps(demos[len(collected)%3], default=lambda value: value.item()));d['initial_state']=session.env.sim.get_state().flatten().tolist()
            collected.append(d);session.last_demonstration=d
        self.p.learn_repeats.set('2')
        edited_scene = self.p.session.env.sim.get_state().flatten().copy()
        edited_scene[0] += 10.
        self.p.session.env.set_init_state(edited_scene)
        with patch('waypoint_demo.record',side_effect=record), patch.object(self.p,'fit',return_value=copy.deepcopy(self.original)) as fit, patch.object(self.p,'model_folder',return_value=Path(self.folder.name)/'record'):
            self.p.learn_button.invoke()
        self.assertEqual(len(collected),2)
        for d in collected:
            np.testing.assert_allclose(d['initial_state'], self.p.session.training_state, atol=1e-12)
        np.testing.assert_allclose(self.p.session.env.sim.get_state().flatten(), edited_scene, atol=1e-12)
        self.assertEqual(len(fit.call_args[0][0]),2)
        self.p.learn_repeats.set('5')
        self.p.learned_model=self.model

    def test_move_and_run_execute_edited_targets(self):
        self.p.restore_recorded_scene()
        self.p.chain.waypoints=self.p.chain.waypoints[:1]
        self.p.redraw_rows(0);self.p.editing_index=None;self.p.select()
        self.p.nudge(2,.005)
        target=self.p.session.editor_target(self.p.chain,0,self.p.read_form())[0]
        self.p.test()
        self.assertIn('Robot reached',self.p.status.get())
        self.assertLess(np.linalg.norm(self.p.session.gripper_pose()[0]-target),.005)
        self.p.save_plan();self.p.run()
        self.assertIn('Sequence',self.p.status.get())
        self.assertFalse(self.p.form_dirty)

    def test_task_picker_keeps_the_draft(self):
        self.p.nudge(0,.01)
        draft=self.p.xyz[0].get()
        self.root.update()
        self.assertEqual(self.p.xyz[0].get(),draft)
        self.assertTrue(self.p.form_dirty)
        self.p.form_dirty=False
        self.p.task_picker();self.root.update()
        self.assertTrue(self.p.task_picker_rows.exists('0'))
        self.p.task_dialog.destroy()

    def test_task_picker_learn_all_uses_selected_suite_despite_filters(self):
        self.p.task_picker()
        self.p.task_filter_suite.set(gui.SUITE_LABELS['libero_spatial'])
        self.p.task_filter_ready.set(True)
        self.p.task_filter_search.set('no matching task')
        self.root.update()
        self.assertEqual(self.p.task_picker_rows.get_children(), ())
        learn = self.p.learn_all_tasks_button
        load = self.p.load_selected_task_button
        self.assertTrue(learn.winfo_viewable())
        self.assertIs(learn.master, load.master)
        self.assertLess(learn.winfo_rootx(), load.winfo_rootx())
        self.assertLessEqual(learn.winfo_rootx() + learn.winfo_width(), load.winfo_rootx())
        with patch.object(self.p, 'learn_all_tasks') as run:
            learn.invoke()
        run.assert_called_once_with('libero_spatial')
        self.assertFalse(self.p.task_dialog.winfo_exists())

    def test_batch_learning_keeps_gray_drafts_until_save_plan_is_clicked(self):
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()['libero_goal']()
        for task in range(2):
            self.p.saved_task_path('libero_goal', task).with_suffix('.user_saved').unlink(missing_ok=True)
        self.p.learn_repeats.set('2')
        recorded_tasks = []

        def record(session, language, **kwargs):
            self.assertTrue(self.p.busy)
            self.assertTrue(self.p.learn_button.instate(['disabled']))
            self.assertFalse(self.p.stop_button.instate(['disabled']))
            recorded_tasks.append(session.args.task_id)
            session.last_demonstration = dict(
                suite='libero_goal', task_id=session.args.task_id, success=True,
                initial_state=session.env.sim.get_state().flatten().tolist(),
                samples=[dict(task_success=False)] * 3)

        def fit(demos, folder, **kwargs):
            self.assertEqual(len(demos), 2)
            self.assertEqual({d['task_id'] for d in demos}, {self.p.task_id})
            position, rotation = self.p.session.gripper_pose()
            return GMMChain('libero_goal', self.p.task_id, self.p.chain.language,
                            [GMMWaypoint('Learned target', 'World', position.tolist(), rotation.tolist())])

        try:
            with patch('libero.libero.benchmark.get_benchmark_dict', return_value={
                    'libero_goal': lambda: SimpleNamespace(n_tasks=2, get_task=suite.get_task)}), \
                    patch.object(gmm, 'MODEL_STORE', Path(self.folder.name) / 'batch_models'), \
                    patch('waypoint_demo.record', side_effect=record), \
                    patch('waypoint_skill_editor.can_fit_editor_model', return_value=True), \
                    patch.object(self.p, 'fit', side_effect=fit):
                self.p.learn_all_tasks('libero_goal')
            self.root.update()
            self.assertEqual(recorded_tasks, [0, 0, 1, 1])
            self.assertIn('learned 2/2', self.p.status.get())
            self.assertEqual(self.p.editing_index, 0)
            self.assertEqual(self.p.rows.selection(), ('0',))
            for task in range(2):
                chain = GMMChain.load(gmm.GMM_STORE / f'libero_goal/task_{task:03d}.json', 'libero_goal', task)
                self.assertEqual(chain.waypoints[0].name, 'Learned target')
                self.assertFalse(self.p.task_saved('libero_goal', task))
            self.p.task_picker(); self.root.update()
            rows = self.p.task_picker_rows
            self.assertEqual(str(rows.tag_configure('review', 'foreground')), MUTED)
            for task in range(2):
                self.assertEqual(rows.item(str(task), 'tags'), ('review',))
                self.assertEqual(rows.item(str(task), 'values')[2], '—')
            self.p.task_dialog.destroy()
            self.p.save_plan_button.invoke()
            self.assertTrue(self.p.task_saved('libero_goal', 1))
            self.assertFalse(self.p.task_saved('libero_goal', 0))
            self.p.task_picker(); self.root.update()
            rows = self.p.task_picker_rows
            self.assertEqual(rows.item('0', 'tags'), ('review',))
            self.assertEqual(rows.item('1', 'tags'), ('saved',))
            self.assertEqual(str(rows.tag_configure('saved', 'foreground')), GREEN)
            self.p.task_dialog.destroy()
        finally:
            if getattr(self.p, 'task_dialog', None) and self.p.task_dialog.winfo_exists():
                self.p.task_dialog.destroy()
            self.p.saved_task_path('libero_goal', 1).with_suffix('.user_saved').unlink(missing_ok=True)
            self.p.form_dirty = False
            self.p.suite.set('libero_goal')
            self.p.task_id = 0
            self.p.load_task()

    def test_reference_change_rotates_covariance_and_imports_edits(self):
        waypoint=self.p.chain.waypoints[0]
        waypoint.learned['position_covariance']=np.diag([.001,.002,.003]).tolist()
        self.p.show_form(waypoint)
        basis=self.p.session.reference_poses[waypoint.reference][1]
        self.p.reference.set('World');self.p.change_reference();self.p.save_plan()
        saved=self.p.chain.waypoints[0]
        np.testing.assert_allclose(saved.learned['position_covariance'],basis@np.diag([.001,.002,.003])@basis.T)
        self.assertEqual(saved.learned['original_reference'],'World')
        self.assertEqual(saved.learned['reference_policy'],'episode_start')
        path=Path(self.folder.name)/'exported_gmm.json';self.p.chain.save(path)
        self.p.chain.waypoints=self.p.chain.waypoints[:1]
        with patch.object(gui.filedialog,'askopenfilename',return_value=str(path)):
            self.p.import_chain()
        self.assertEqual(len(self.p.chain.waypoints),len(self.original.waypoints))
        self.assertIsNotNone(self.p.learned_model)

    def test_cancel_task_change_keeps_learned_model(self):
        old_model=self.p.learned_model;old_chain=self.p.chain
        self.p.nudge(0,.01)
        with patch.object(gui.messagebox,'askyesnocancel',return_value=None):
            self.p.change_task_by(1)
        self.assertIs(self.p.learned_model,old_model)
        self.assertIs(self.p.chain,old_chain)
        self.assertEqual(self.p.task_id,0)
        self.p.form_dirty=False

    def test_save_plan_keeps_a_json_the_picker_can_see(self):
        """Save plan writes a plan the user owns, apart from the autosave."""
        suite, task = self.p.chain.suite, self.p.chain.task_id
        # Another test in this class may have saved already.
        self.p.saved_path().unlink(missing_ok=True)
        self.assertFalse(self.p.task_saved(suite, task))
        self.p.save_plan()
        target = self.p.saved_path()
        self.assertTrue(target.exists())
        self.assertNotEqual(target, self.p.path())
        self.assertIn(str(target), self.p.status.get())
        self.assertTrue(self.p.task_saved(suite, task))
        kept = GMMChain.load(target, suite, task)
        self.assertEqual(len(kept.waypoints), len(self.p.chain.waypoints))
        # The picker marks it green and counts it among the saved.
        self.p.form_dirty = False
        self.p.task_picker(); self.root.update()
        rows = self.p.task_picker_rows
        self.assertEqual(rows.item(str(task), 'tags'), ('saved',))
        self.assertEqual(rows.item(str(task), 'values')[2], 'saved')
        other = next(i for i in range(10) if i != task and rows.exists(str(i)))
        self.assertNotEqual(rows.item(str(other), 'tags'), ('saved',))
        self.p.task_dialog.destroy()

    def test_relationships_keep_all_waypoints_that_share_a_place(self):
        """Grasp and release remain distinct rows even at the same position."""
        panel = self.p
        panel.rows.selection_set('0'); panel.select(); self.root.update()
        panel.draw_relationships(); self.root.update()
        self.assertEqual(panel.relationship_rows.get_children(),
                         tuple(str(i) for i in range(len(panel.chain.waypoints))))
        for i, waypoint in enumerate(panel.chain.waypoints):
            self.assertIn(gui.friendly_name(waypoint.reference) if waypoint.reference != 'Previous waypoint'
                          else ('Start pose' if i == 0 else f'Previous waypoint {i:02d}'),
                          panel.relationship_rows.item(str(i), 'values')[1])

    def test_camera_shows_the_target_frame_and_its_reference_only(self):
        """Two triads in the scene: where the wrist is going, and from what."""
        panel = self.p
        panel.rows.selection_set('1'); panel.select(); self.root.update()
        panel.paint_camera(); self.root.update()
        canvas = panel.camera
        texts = [canvas.itemcget(i, 'text') for i in canvas.find_all()
                 if canvas.type(i) == 'text']
        # Three arrows per triad, and nothing numbered or circled.
        self.assertEqual(sum(1 for i in canvas.find_all()
                             if canvas.type(i) == 'line'), 6)
        self.assertEqual([t for t in texts if t in ('+X', '+Y', '+Z')].count('+X'), 2)
        self.assertNotIn('1', texts)
        self.assertIn(friendly_name(panel.chain.waypoints[1].reference), texts)
        self.assertIn(panel.plan_labels()[1], texts)

    def test_stop_and_application_close(self):
        def stopped():
            self.app.set_busy(self.p,True)
            self.assertTrue(self.p.busy)
            self.p.stop();self.p.pump()
        self.p.operation(stopped)
        self.assertIn('Stopped',self.p.status.get())
        self.assertFalse(self.p.busy)
        self.p.busy=True
        self.app.close()
        self.assertTrue(self.p.stopped);self.assertTrue(self.p.closing)
        self.p.busy=False;self.p.closing=False


if __name__=='__main__':unittest.main()
