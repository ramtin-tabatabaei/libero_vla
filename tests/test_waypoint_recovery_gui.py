"""Real Tk/MuJoCo integration; opt in with LIBERO_LIVE_GUI_TEST=1 on a display."""
import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('LIBERO_LIVE_GUI_TEST') != '1',
                                reason='Requires a display and MuJoCo rendering')


def forbid_artifact_writers(monkeypatch):
    import waypoint_gmm_editor as editor
    import waypoint_recovery_benchmark as benchmark
    from waypoint_gmm import GMMChain
    def forbidden(*args, **kwargs):
        pytest.fail('Live viewer attempted to save an artifact')
    monkeypatch.setattr(editor, 'write_json', forbidden)
    monkeypatch.setattr(benchmark, 'write_json', forbidden)
    monkeypatch.setattr(benchmark, 'save_scene', forbidden)
    monkeypatch.setattr(GMMChain, 'save', forbidden)


def test_window_streams_motion_and_can_pause_and_stop(monkeypatch):
    from waypoint_recovery_gui import RecoveryViewer, ROOT, DEFAULT_EXPERIMENT, MODES
    import tkinter as tk
    forbid_artifact_writers(monkeypatch)
    root = tk.Tk()
    viewer = RecoveryViewer(root, SimpleNamespace(plan_root=ROOT/'saved_plans',
        experiment=DEFAULT_EXPERIMENT, suite='libero_goal', task_id=1, backend='pi05', port=8000))
    try:
        root.update()
        assert len(viewer.plans) >= 10
        assert not viewer.detection_latency.get()
        assert viewer.options().delay == 0.
        viewer.latency_check.invoke()
        assert viewer.options().delay == 0.
        assert viewer.options().handoff_at_end
        assert viewer.options().propagate_deviation
        assert all('Grasp' in row for row in viewer.row_choices)
        viewer.operate(False)
        assert viewer.stage.get() == 'Scene ready', viewer.detail.get()
        assert viewer.photo is not None
        initial = viewer.runner.session.env.sim.get_state().flatten().copy()
        viewer.mode.set(MODES['normal_waypoints'])
        viewer.speed.set('10×')
        viewer.update_conditions()
        assert viewer.latency_check.instate(['disabled'])
        assert viewer.options().delay == 0.
        assert not viewer.options().handoff_at_end
        paused_at = []
        resumed = []
        def pause_when_moving():
            elapsed = float(viewer.runner.session.env.sim.data.time)-viewer.episode_start
            if viewer.busy and viewer.stage.get() == 'Waypoints' and elapsed > .5:
                viewer.toggle_pause()
                paused_at.append(viewer.runner.session.env.sim.get_state().flatten().copy())
                root.after(80, resume)
            else:
                root.after(20, pause_when_moving)
        def resume():
            np.testing.assert_array_equal(viewer.runner.session.env.sim.get_state().flatten(), paused_at[0])
            resumed.append(True)
            viewer.toggle_pause()
            root.after(100, viewer.stop)
        root.after(20, pause_when_moving)
        viewer.operate(True)
        assert resumed
        assert viewer.stage.get() == 'Stopped', viewer.detail.get()
        assert not viewer.busy and not viewer.paused
        assert not np.array_equal(initial, viewer.runner.session.env.sim.get_state().flatten())
        assert 'Waypoints · ' in viewer.controller_times.get()
        assert 'VLA · 0.00 s simulation' in viewer.controller_times.get()
        assert viewer.runner.timings.snapshot(viewer.runner.session.env.sim.data.time)['waypoints_sim_seconds'] > .5
    finally:
        viewer.close()


@pytest.mark.parametrize('suite,task_id,mode,intensity', [
    ('libero_goal', 1, 'failure_only', 'low'),
    ('libero_goal', 1, 'recovery', 'low'),
    ('libero_goal', 1, 'failure_only', 'high'),
    ('libero_goal', 1, 'recovery', 'high'),
    ('libero_spatial', 0, 'recovery', 'low'),
    ('libero_object', 0, 'recovery', 'low'),
    ('libero_10', 0, 'recovery', 'low'),
    ('libero_90', 1, 'recovery', 'low'),
])
def test_real_failure_runs_to_handoff_without_saving(monkeypatch, suite, task_id, mode, intensity):
    from waypoint_recovery_live import LiveRunner, LiveOptions
    import waypoint_policy as policy
    from serve_pi05_libero import metadata
    forbid_artifact_writers(monkeypatch)
    viewer = LiveRunner()
    frames = []
    try:
        options = LiveOptions(suite, task_id, mode=mode, kind='grasp_loss',
                              intensity=intensity, max_policy_steps=5)
        viewer.prepare(options)
        connection = Mock()
        connection.connect.return_value = metadata()
        actions = np.zeros((5, 7))
        actions[:, -1] = -1.
        connection.infer.return_value = {'actions': actions}
        monkeypatch.setattr(policy, 'PolicyConnection', lambda port: connection)
        switches = Mock(wraps=policy.switch_controller)
        monkeypatch.setattr(policy, 'switch_controller', switches)
        # Starts may be restored in prepare; a live controller handoff may not.
        def no_reset(*args, **kwargs):
            pytest.fail('Live handoff reset the simulation')
        monkeypatch.setattr(viewer.session.env, 'reset', no_reset)
        monkeypatch.setattr(viewer.session.env, 'set_init_state', no_reset)
        def tick():
            if not frames or viewer.session.env.sim.data.time-frames[-1] >= .5:
                image = viewer.session.image()
                assert image.shape == (360, 480, 3)
                frames.append(float(viewer.session.env.sim.data.time))
        viewer.session.tick = tick
        result = viewer.run(options)
        assert result['status'] == ('failure_observed' if mode == 'failure_only' else 'recovery_failed'), result
        assert result['failure_event']['kind'] == 'grasp_loss'
        assert result['failure_event']['primitive'] == ('transport' if intensity == 'high' else 'grasp')
        assert result['waypoint_completed'] and result['fault_execution_completed']
        assert max(step['row'] for step in result['trace']) == result['injection_row']
        # Grasp's suppression/release and final check finish before handoff.
        assert result['actual_delay_seconds'] > .05
        assert result['handoff_sim_time'] == result['waypoint_finish_sim_time']
        assert result['trace'][-1]['result'] != 'Running'
        assert len(frames) > 2
        assert viewer.session.save_replay_archive is False
        counts = result['controller_timings']
        assert counts['waypoints_sim_seconds'] > 0.
        assert counts['vla_sim_seconds'] == pytest.approx(.25 if mode == 'recovery' else 0.)
        if mode == 'recovery':
            assert result['policy_result']['steps'] == 5
            assert [c.args[1] for c in switches.call_args_list] == ['OSC_POSE', 'JOINT_POSITION']
            assert viewer.session.env.sim.data.time >= result['handoff_sim_time']+.25-1e-8
            connection.close.assert_called_once()
    finally:
        viewer.close()


@pytest.mark.parametrize('action,fault_seed,reachable,abort_close,block_all', [
    ('continue', 1, True, False, False), ('stop', 1, True, False, False),
    ('continue', 101, True, False, False), ('continue', 1, True, True, False),
    ('continue', 101, False, False, True)])
def test_high_transition_finishes_grasp_then_waits_for_continue_vla(monkeypatch, action, fault_seed, reachable, abort_close, block_all):
    from waypoint_recovery_gui import RecoveryViewer, ROOT, DEFAULT_EXPERIMENT, FAILURES
    import waypoint_policy as policy
    import waypoint_recovery_live as live
    from serve_pi05_libero import metadata
    import tkinter as tk
    forbid_artifact_writers(monkeypatch)
    monkeypatch.setattr(live.secrets, 'randbits', lambda bits: fault_seed)
    root = tk.Tk()
    viewer = RecoveryViewer(root, SimpleNamespace(plan_root=ROOT/'saved_plans',
        experiment=DEFAULT_EXPERIMENT, suite='libero_goal', task_id=1, backend='pi05', port=8000))
    if block_all:
        monkeypatch.setattr(live, 'TRANSITION_PLAN_ATTEMPTS', 3)
    original_planner = live.guarded_move
    def checked_planner(session, target, rotation, **kwargs):
        before = session.env.sim.get_state().flatten().copy()
        outcome = (('no collision-free route to the target', 'blocked') if block_all else
                   original_planner(session, target, rotation, **kwargs))
        if outcome[1] in ('blocked', 'unchecked'):
            np.testing.assert_array_equal(session.env.sim.get_state().flatten(), before)
            assert viewer.continue_button.instate(['disabled'])
            assert not viewer.waiting_for_vla
        return outcome
    monkeypatch.setattr(live, 'guarded_move', checked_planner)
    connection = Mock()
    connection.connect.return_value = metadata()
    actions = np.zeros((5, 7))
    actions[:, -1] = -1.
    connection.infer.return_value = {'actions': actions}
    monkeypatch.setattr(policy, 'PolicyConnection', lambda port: connection)
    close_events = []
    original_grip = live.sp.Executor.run_subgoal
    def track_grip(executor, subgoal):
        closing = subgoal.kind == 'grip' and subgoal.gripper == live.sp.GRIPPER_CLOSE
        if closing:
            close_events.append(('started', float(executor.env.sim.data.time)))
            if abort_close:
                raise ValueError('Grasp aborted before the gripper finished')
        result = original_grip(executor, subgoal)
        if closing:
            close_events.append(('finished', float(executor.env.sim.data.time)))
        return result
    monkeypatch.setattr(live.sp.Executor, 'run_subgoal', track_grip)
    waiting = []
    errors = []
    timeout = None
    try:
        root.update()
        viewer.failure.set(FAILURES['transition'])
        viewer.intensity.set('High')
        viewer.steps.set('5')
        viewer.speed.set('10×')
        viewer.update_conditions()
        def handle_wait():
            if not viewer.waiting_for_vla:
                root.after(20, handle_wait)
                return
            try:
                connection.connect.assert_not_called()
                connection.infer.assert_not_called()
                assert not viewer.continue_button.instate(['disabled'])
                assert viewer.runner.timings.phase is None
                if reachable:
                    assert viewer.direction_info.get().startswith('Direction ')
                    assert 'cm · Y ' in viewer.direction_info.get()
                if reachable and not abort_close:
                    assert close_events and close_events[-1][0] == 'finished'
                    assert close_events[-1][1]-close_events[-2][1] >= 1.5-1e-8
                else:
                    assert viewer.stage.get() == 'Waypoint incomplete · Continue VLA'
                    assert 'attempt stopped' in viewer.detail.get()
                state = viewer.runner.session.env.sim.get_state().flatten().copy()
                counts = viewer.runner.timings.snapshot(viewer.runner.session.env.sim.data.time)
                waiting.append((state, counts))
                root.after(80, check_then_act)
            except BaseException as exc:
                errors.append(exc)
                viewer.stop()
        def check_then_act():
            try:
                state, counts = waiting[0]
                np.testing.assert_array_equal(viewer.runner.session.env.sim.get_state().flatten(), state)
                assert viewer.runner.timings.snapshot(viewer.runner.session.env.sim.data.time) == counts
                assert counts['vla_sim_seconds'] == 0.
                assert viewer.runner.session.injected_failure == ''
                connection.connect.assert_not_called()
                def no_reset(*args, **kwargs):
                    pytest.fail('Continue VLA reset the failure scene')
                monkeypatch.setattr(viewer.runner.session.env, 'reset', no_reset)
                monkeypatch.setattr(viewer.runner.session.env, 'set_init_state', no_reset)
                if action == 'continue':
                    viewer.continue_button.invoke()
                else:
                    viewer.stop_button.invoke()
            except BaseException as exc:
                errors.append(exc)
                viewer.stop()
        root.after(20, handle_wait)
        timeout = root.after(45000, viewer.stop)
        viewer.operate(True)
        assert not errors, errors
        assert not viewer.waiting_for_vla
        assert viewer.continue_button.instate(['disabled'])
        assert waiting, viewer.detail.get()
        if action == 'continue':
            result = viewer.last_result
            assert result is not None, viewer.detail.get()
            assert result['observed_failure'] == reachable, result
            assert result['handoff_ready'] and result['waypoint_execution_stopped']
            if reachable and not abort_close:
                assert result['waypoint_completed'] and result['fault_execution_completed']
                assert result['waypoint_completion_event']['row'] == result['injection_row']
                assert result['waypoint_completion_event']['sim_time'] >= close_events[-1][1]
            else:
                assert not result['waypoint_completed'] and not result['fault_execution_completed']
                assert result['waypoint_completion_event'] is None
            if reachable:
                assert result['failure_event']['kind'] == 'transition'
                assert result['failure_event']['primitive'] == 'grasp'
                assert result['failure_event']['requested_metres'] >= .15
                assert result['handoff_sim_time'] > result['failure_event']['sim_time']
            assert result['handoff_sim_time'] == result['waypoint_end_sim_time']
            assert result['trace'][-1]['result'] != 'Running'
            routes = [t for t in result['fault_targets'] if 'planning' in t]
            path = routes[-1]
            if fault_seed == 101 and reachable:
                assert len(routes) > 1
                assert routes[0]['planning'] in ('blocked', 'unchecked')
                assert all(route['planning'] in ('blocked', 'unchecked') for route in routes[:-1])
                assert len({tuple(route['target']) for route in routes}) == len(routes)
            if reachable:
                assert path['planning'] in ('planned', 'line')
                assert path['target_gap_metres'] < .03
            else:
                assert path['planning'] == 'blocked'
                assert 'no collision-free route' in path['attempt_result']
            assert max(step['row'] for step in result['trace']) == result['injection_row']
            assert result['controller_timings']['vla_sim_seconds'] == pytest.approx(.25)
            connection.connect.assert_called_once()
        else:
            assert viewer.stage.get() == 'Stopped', viewer.detail.get()
            connection.connect.assert_not_called()
            connection.infer.assert_not_called()
    finally:
        if timeout is not None:
            root.after_cancel(timeout)
        viewer.close()


@pytest.mark.parametrize('intensity', ['low', 'high'])
def test_repeated_real_transitions_execute_different_directions_without_saving(monkeypatch, intensity):
    import waypoint_recovery_live as live
    from waypoint_recovery_live import LiveRunner, LiveOptions
    from waypoint_failures import TRANSITION_BANDS
    forbid_artifact_writers(monkeypatch)
    monkeypatch.setattr(live.secrets, 'randbits', lambda bits: 1)
    runner = LiveRunner()
    directions = []
    try:
        options = LiveOptions('libero_goal', 1, mode='failure_only', kind='transition', intensity=intensity)
        for _ in range(4):
            result = runner.run(options)
            assert result['observed_failure'], result
            # A geometrically checked route can still stall under dynamics.
            # Failure-only mode preserves the failed result. Recovery may now
            # use that same stopped state after the user presses Continue VLA.
            if result['status'] == 'failure_waypoint_incomplete':
                assert not result['fault_execution_completed']
                assert result['handoff_ready'] and result['waypoint_execution_stopped']
                assert result['waypoint_failure_reason']
            else:
                assert result['status'] == 'failure_observed', result
                assert result['waypoint_completed'] and result['fault_execution_completed']
            direction = np.asarray(result['failure_event']['requested_direction'])
            if directions:
                gap = np.arccos(np.clip(direction @ directions[-1], -1., 1.))
                assert gap >= live.TRANSITION_DIRECTION_SEPARATION-1e-8
            directions.append(direction)
            offset = np.asarray(result['transition_offset_metres'])
            assert offset[2] == 0.
            low, high = TRANSITION_BANDS[intensity]
            assert low <= np.linalg.norm(offset) <= high
            assert result['controller_timings']['vla_sim_seconds'] == 0.
    finally:
        runner.close()


@pytest.mark.parametrize('kind,intensity,seed,latency', [
    ('collision', 'Low', 1, False), ('collision', 'High', 2, False),
    ('orientation', 'Low', 1, False), ('orientation', 'High', 2, False),
    ('collision', 'Low', 1, True), ('orientation', 'Low', 1, True),
    ('transition', 'High', 1, True), ('orientation', 'High', 2, True),
    ('grasp_loss', 'High', 1, False), ('grasp_loss', 'High', 1, True)])
def test_random_pose_failure_waits_for_continue_after_the_attempt_stops(monkeypatch, kind, intensity, seed, latency):
    import tkinter as tk
    import waypoint_recovery_live as live
    import waypoint_policy as policy
    from waypoint_recovery_gui import RecoveryViewer, ROOT, DEFAULT_EXPERIMENT, FAILURES
    from serve_pi05_libero import metadata
    forbid_artifact_writers(monkeypatch)
    seeds = iter([seed, seed+100])
    monkeypatch.setattr(live.secrets, 'randbits', lambda bits: next(seeds) if kind == 'orientation' else seed)
    root = tk.Tk()
    viewer = RecoveryViewer(root, SimpleNamespace(plan_root=ROOT/'saved_plans',
        experiment=DEFAULT_EXPERIMENT, suite='libero_goal', task_id=1, backend='pi05', port=8000))
    presses, closes, releases = [], [], []
    original_press = live.GraspOnlyInjector._collision
    original_grip = live.sp.Executor.run_subgoal
    original_release = live.GraspOnlyInjector.release
    def press(injector):
        assert injector.session.executor.gripper == live.sp.GRIPPER_OPEN
        assert len(closes) == len(presses)
        result = original_press(injector)
        assert injector.session.executor.gripper == live.sp.GRIPPER_OPEN
        presses.append((float(injector.session.env.sim.data.time), injector.session.world.eef().copy()))
        return result
    def grip(executor, subgoal):
        closing = kind == 'collision' and subgoal.kind == 'grip' and subgoal.gripper == live.sp.GRIPPER_CLOSE
        if closing:
            assert len(presses) == len(closes)+1, 'Gripper closed before collision motion finished'
            assert executor.gripper == live.sp.GRIPPER_OPEN
            assert float(executor.env.sim.data.time) == presses[-1][0]
            np.testing.assert_allclose(executor.world.eef(), presses[-1][1])
        result = original_grip(executor, subgoal)
        if closing:
            closes.append(float(executor.env.sim.data.time))
        return result
    monkeypatch.setattr(live.GraspOnlyInjector, '_collision', press)
    monkeypatch.setattr(live.sp.Executor, 'run_subgoal', grip)
    def release(injector):
        assert injector.primitive == 'transport'
        assert injector.clock.event is None
        assert live.held_object(injector.session.world)
        target = injector.session.running_target[0]
        assert np.linalg.norm(injector.session.world.eef()-target) <= .02
        started = float(injector.session.env.sim.data.time)
        result = original_release(injector)
        assert result and not live.held_object(injector.session.world)
        releases.append((started, float(injector.session.env.sim.data.time), injector.row))
        return result
    monkeypatch.setattr(live.GraspOnlyInjector, 'release', release)
    connection = Mock()
    connection.connect.return_value = metadata()
    actions = np.zeros((5, 7))
    actions[:, -1] = -1.
    connection.infer.return_value = {'actions': actions}
    monkeypatch.setattr(policy, 'PolicyConnection', lambda port: connection)
    original_wait = viewer.wait_for_vla
    handoffs = []
    def wait(outcome):
        assert outcome['handoff_ready'] and outcome['waypoint_execution_stopped']
        handoffs.append(outcome)
        return original_wait(outcome)
    viewer.wait_for_vla = wait
    errors, directions = [], []
    try:
        root.update()
        viewer.failure.set(FAILURES[kind])
        viewer.intensity.set(intensity)
        viewer.steps.set('5')
        viewer.speed.set('10×')
        viewer.update_conditions()
        assert not viewer.detection_latency.get()
        if latency:
            viewer.latency_check.invoke()
        assert viewer.options().delay == 0.
        assert viewer.options().handoff_at_end == latency
        assert viewer.options().propagate_deviation
        for run in range(2):
            connected = connection.connect.call_count
            inferred = connection.infer.call_count
            pauses = []
            def pause_during_latency():
                if not viewer.busy or viewer.waiting_for_vla:
                    return
                if viewer.stage.get() != 'Detection latency':
                    root.after(10, pause_during_latency)
                    return
                try:
                    assert viewer.continue_button.instate(['disabled'])
                    assert connection.connect.call_count == connected
                    assert connection.infer.call_count == inferred
                    viewer.toggle_pause()
                    state = viewer.runner.session.env.sim.get_state().flatten().copy()
                    def resume_latency():
                        try:
                            np.testing.assert_array_equal(viewer.runner.session.env.sim.get_state().flatten(), state)
                            assert viewer.runner.timings.paused
                            assert viewer.continue_button.instate(['disabled'])
                            assert connection.connect.call_count == connected
                            pauses.append(True)
                            viewer.toggle_pause()
                        except BaseException as exc:
                            errors.append(exc)
                            viewer.stop()
                    root.after(80, resume_latency)
                except BaseException as exc:
                    errors.append(exc)
                    viewer.stop()
            if latency:
                root.after(10, pause_during_latency)
            def handle_wait():
                if not viewer.waiting_for_vla:
                    root.after(20, handle_wait)
                    return
                try:
                    assert connection.connect.call_count == connected
                    assert connection.infer.call_count == inferred
                    assert viewer.runner.timings.phase is None
                    outcome = handoffs[-1]
                    assert outcome['detection_latency_seconds'] == 0.
                    if latency:
                        assert pauses
                        assert outcome['detection_latency_completed']
                        assert outcome['handoff_mode'] == 'end_of_plan'
                        assert outcome['waypoint_plan_completed']
                        assert max(step['row'] for step in outcome['trace']) == len(viewer.runner.chain.waypoints)-1
                        assert outcome['trace'][-1]['result'] != 'Running'
                        assert outcome['handoff_sim_time'] == outcome['trace'][-1]['sim_time']
                        assert outcome['detection_latency_start_sim_time'] >= outcome['failure_event']['sim_time']
                        if outcome['waypoint_completed']:
                            assert outcome['detection_latency_start_sim_time'] == outcome['waypoint_finish_sim_time']
                        assert any(step['row'] > outcome['injection_row'] for step in outcome['trace'])
                        start = np.asarray(outcome['detection_latency_start_pose']['position'])
                        assert np.linalg.norm(viewer.runner.session.world.eef()-start) > .01
                        if kind in ('transition', 'orientation'):
                            propagation = outcome['propagated_deviation']
                            assert propagation and propagation['kind'] == kind
                            assert len(propagation['targets']) == len(viewer.runner.chain.waypoints)-outcome['injection_row']-1
                            delta = np.asarray(propagation['rotation'])
                            p0 = np.asarray(propagation['nominal_position'])
                            p1 = np.asarray(propagation['actual_position'])
                            for target in propagation['targets']:
                                np.testing.assert_allclose(target['position'], p1+delta@(np.asarray(target['nominal_position'])-p0))
                                np.testing.assert_allclose(target['orientation'], delta@np.asarray(target['nominal_orientation']))
                    if kind == 'collision':
                        assert outcome['failure_event']['kind'] == 'collision'
                        assert viewer.direction_info.get().startswith('Collision direction ')
                    elif kind == 'orientation':
                        assert viewer.direction_info.get().startswith('Total deviation ')
                        low, high = live.ORIENTATION_BANDS[intensity.lower()]
                        assert low <= outcome['orientation_deviation_degrees'] <= high
                        assert abs(outcome['orientation_axes_degrees'][2]) >= (
                            outcome['orientation_deviation_degrees']*live.ORIENTATION_MIN_YAW_FRACTION-1e-10)
                        assert ' · yaw ' in viewer.direction_info.get()
                        assert ' · XY ' in viewer.direction_info.get()
                        offset = np.asarray(outcome['orientation_transition_offset_metres'])
                        minimum, maximum = live.ORIENTATION_XY_BANDS[intensity.lower()]
                        assert minimum <= np.linalg.norm(offset) <= maximum
                        assert offset[2] == 0.
                        targets = [t for t in outcome['fault_targets'] if t.get('kind') == 'orientation'
                                   and t.get('axis') == 'xyz']
                        assert targets
                        from scipy.spatial.transform import Rotation
                        for target in targets:
                            nominal, changed = map(np.asarray, (target['orientation'], target['changed']))
                            total = np.degrees(Rotation.from_matrix(nominal.T@changed).magnitude())
                            assert low <= total <= high
                        assert low <= outcome['orientation_measured_degrees'] <= high
                        if outcome['failure_event'] is not None:
                            assert low <= outcome['failure_event']['orientation_degrees'] <= high
                    elif kind == 'transition':
                        assert outcome['failure_event']['kind'] == 'transition'
                    else:
                        event = outcome['failure_event']
                        assert event['kind'] == 'grasp_loss' and event['primitive'] == 'transport'
                        assert len(releases) == run+1
                        started, finished, row = releases[-1]
                        assert row == outcome['injection_row']
                        assert started <= event['sim_time'] <= finished
                        assert outcome['waypoint_completion_event']['sim_time'] == finished
                        if not latency:
                            assert max(step['row'] for step in outcome['trace']) == row
                    if not outcome['fault_execution_completed'] and not latency:
                        assert viewer.stage.get() == 'Waypoint incomplete · Continue VLA'
                    state = viewer.runner.session.env.sim.get_state().flatten().copy()
                    times = viewer.runner.timings.snapshot(viewer.runner.session.env.sim.data.time)
                    def continue_from_frozen_state():
                        try:
                            np.testing.assert_array_equal(viewer.runner.session.env.sim.get_state().flatten(), state)
                            assert viewer.runner.timings.snapshot(viewer.runner.session.env.sim.data.time) == times
                            assert connection.connect.call_count == connected
                            viewer.continue_button.invoke()
                        except BaseException as exc:
                            errors.append(exc)
                            viewer.stop()
                    root.after(80, continue_from_frozen_state)
                except BaseException as exc:
                    errors.append(exc)
                    viewer.stop()
            root.after(20, handle_wait)
            timeout = root.after(60000, viewer.stop)
            try:
                viewer.operate(True)
            finally:
                root.after_cancel(timeout)
            assert not errors, errors
            assert len(handoffs) == run+1, viewer.detail.get()
            result = viewer.last_result
            assert result, viewer.detail.get()
            policy_result = result['policy_result']
            assert result['status'] == ('recovered' if policy_result['success'] else 'recovery_failed')
            if result['success_before_policy']:
                assert policy_result['steps'] == 0
                assert connection.infer.call_count == inferred
            else:
                assert 1 <= policy_result['steps'] <= 5
            if not policy_result['success']:
                assert policy_result['steps'] == 5
            assert result['controller_timings']['vla_sim_seconds'] == pytest.approx(.05*policy_result['steps'])
            assert result['handoff_sim_time'] == result['waypoint_end_sim_time']
            if kind == 'collision':
                press = result['fault_targets'][-1]
                assert press['phase'] == 'grasp_press_before_close'
                assert press['finished_sim_time']-press['started_sim_time'] == pytest.approx(3.)
                assert len(presses) == len(closes) == run+1
                assert press['started_sim_time'] <= result['failure_event']['sim_time'] <= press['finished_sim_time']
                assert press['finished_sim_time'] < closes[-1] <= result['handoff_sim_time']
                assert press['press_direction'][2] < 0.
                low, high, depth = (.04, .06, .06) if intensity == 'Low' else (.06, .10, .12)
                assert low <= press['xy_offset_metres'] <= high
                assert press['press_metres'] == depth
                np.testing.assert_allclose(np.asarray(press['target'])-press['offset_target'],
                                           np.asarray(press['press_direction'])*depth)
            if kind in ('collision', 'orientation'):
                direction = np.asarray(result['failure_event']['requested_local_direction'] if kind == 'collision'
                                       else result['orientation_axes_degrees'])
                if directions and kind == 'collision':
                    gap = np.arccos(np.clip(direction @ directions[-1], -1., 1.))
                    assert gap >= live.TRANSITION_DIRECTION_SEPARATION-1e-8
                elif directions:
                    assert not np.allclose(direction, directions[-1])
                directions.append(direction)
            assert connection.connect.call_count == connected+1
    finally:
        viewer.close()
