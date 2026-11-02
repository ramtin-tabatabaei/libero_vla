"""Live conditions must preserve scene continuity and never create artifacts."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import waypoint_recovery_live as live
from waypoint_gmm import GMMChain, GMMWaypoint
from waypoint_failures import Injector, TRANSITION_BANDS
from waypoint_recovery_live import ORIENTATION_BANDS as TILT_BANDS


def manual_plan(folder):
    chain = GMMChain('libero_goal', 1, 'put the bowl on the stove', [
        GMMWaypoint(name='Move', skill_role='move'),
        GMMWaypoint(name='Grasp', skill_role='grasp'),
        GMMWaypoint(name='Transport', skill_role='transport'),
    ])
    chain.save(folder/'libero_goal/task_001.json')
    return chain


@pytest.fixture
def runner(tmp_path, monkeypatch):
    chain = manual_plan(tmp_path/'plans')
    data = SimpleNamespace(time=0.)
    env = Mock()
    state = np.arange(5, dtype=float)
    env.sim.data = data
    env.sim.get_state.return_value.flatten.side_effect = lambda: state.copy()
    env.check_success.return_value = False
    def restore(values):
        state[:] = values
        data.time = float(values[0])
    env.set_init_state.side_effect = restore
    session = SimpleNamespace(env=env, args=SimpleNamespace(seed=0), load=Mock(),
        executor=SimpleNamespace(gripper=-1., arrival_grace_seconds=1.), training_state=state,
        world=SimpleNamespace(eef=lambda:np.zeros(3)),
        last_shuffle_mapping=None, attach=Mock(), continue_on_failure=False, tick=Mock())
    session.restore_training_scene = lambda: env.set_init_state(session.training_state)
    session.shuffle_scene = Mock()
    def run(plan, progress):
        assert session.save_replay_archive is False
        progress(0, 'Running')
        data.time += .05
        session.tick()
        return 'finished'
    session.run = Mock(side_effect=run)
    monkeypatch.setattr(live, 'GMMSession', lambda args: session)
    def fault(session, chain, spec, delay, limit, folder, **kwargs):
        assert folder is None
        assert kwargs.get('recorder') is None
        kwargs['capture'](session, folder, 'failure')
        kwargs['progress'](1, 'Running')
        data.time += 2.
        kwargs['capture'](session, folder, 'handoff')
        return dict(observed_failure=True, success_before_policy=False,
                    failure_event=dict(sim_time=0.), waypoint_completed=True,
                    waypoint_execution_stopped=True,
                    waypoint_completion_event=dict(row=spec['injection_row'], sim_time=data.time),
                    fault_targets=[dict(planning='planned', target=[0., 0., 0.], attempt_result='ok')])
    monkeypatch.setattr(live, 'run_fault', Mock(side_effect=fault))
    def policy(session, language, **kwargs):
        # The handoff must not reset the physical scene.
        assert data.time == pytest.approx(2.)
        data.time += .05
        return dict(success=True, steps=1, model_id='test')
    monkeypatch.setattr(live, 'run_policy', Mock(side_effect=policy))
    viewer = live.LiveRunner(tmp_path/'plans', tmp_path/'experiment')
    return viewer, session


def test_live_recovery_uses_physical_handoff_and_never_writes_results(runner, tmp_path, monkeypatch):
    viewer, session = runner
    options = live.LiveOptions('libero_goal', 1, injection_row=1)
    viewer.prepare(options)
    resets = session.env.set_init_state.call_count
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*'))
    def no_write(*args, **kwargs):
        pytest.fail('A live run attempted to write a file')
    monkeypatch.setattr(Path, 'write_text', no_write)
    monkeypatch.setattr(Path, 'write_bytes', no_write)
    result = viewer.run(options)
    assert result['status'] == 'recovered'
    assert result['elapsed_sim_seconds'] == pytest.approx(2.05)
    assert session.env.set_init_state.call_count == resets
    assert session.save_replay_archive is False
    assert live.run_fault.call_args.kwargs['folder'] is None
    assert live.run_fault.call_args.kwargs['handoff_after_waypoint'] is True
    assert live.run_fault.call_args.args[2]['injection_row'] == 1
    assert live.run_policy.call_args.kwargs['backend'] == 'pi05'
    assert result['controller_timings']['waypoints_sim_seconds'] == pytest.approx(2.)
    assert result['controller_timings']['vla_sim_seconds'] == pytest.approx(.05)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*')) == before


@pytest.mark.parametrize('observed,already,mode,status', [
    (False, False, 'failure_only', 'failure_waypoint_incomplete'),
    (True, False, 'failure_only', 'failure_observed'),
])
def test_policy_is_skipped_for_conditions_that_do_not_need_it(runner, observed, already, mode, status):
    viewer, session = runner
    live.run_fault.side_effect = None
    live.run_fault.return_value = dict(observed_failure=observed, success_before_policy=already,
        waypoint_completed=True, waypoint_execution_stopped=True,
        waypoint_completion_event=dict(row=1, sim_time=0.))
    result = viewer.run(live.LiveOptions('libero_goal', 1, mode=mode))
    assert result['status'] == status
    live.run_policy.assert_not_called()


def test_second_run_restarts_from_same_in_memory_layout(runner):
    viewer, session = runner
    options = live.LiveOptions('libero_goal', 1)
    assert viewer.run(options)['status'] == 'recovered'
    assert viewer.run(options)['status'] == 'recovered'
    session.load.assert_called_once()
    # Restore once to capture the layout, once per run to restore that snapshot.
    assert session.env.set_init_state.call_count == 3


def test_batch_fault_seed_is_replayed_without_changing_live_default(runner):
    viewer, session = runner
    for _ in range(2):
        result = viewer.run(live.LiveOptions('libero_goal', 1, fault_seed=1234))
        assert live.run_fault.call_args.args[2]['fault_seed'] == 1234
        assert result['fault_seed'] == 1234


def test_end_of_plan_options_reach_controller_and_prompt_after_plan(runner):
    viewer, _ = runner
    report = Mock()
    live.run_fault.side_effect = None
    live.run_fault.return_value = dict(observed_failure=True, waypoint_completed=True,
        waypoint_execution_stopped=True, waypoint_completion_event=dict(row=1, sim_time=2.),
        waypoint_plan_completed=True)
    live.run_policy.side_effect = None
    live.run_policy.return_value = dict(success=False, steps=1)
    viewer.run(live.LiveOptions('libero_goal', 1, injection_row=1,
        handoff_at_end=True, propagate_deviation=True), report, before_vla=Mock())
    assert live.run_fault.call_args.kwargs['handoff_at_end']
    assert live.run_fault.call_args.kwargs['propagate_deviation']
    assert any('Remaining waypoint plan finished' in call.args[1] for call in report.call_args_list)


def test_end_of_plan_rejects_a_simultaneous_timed_delay():
    with pytest.raises(ValueError, match='timed delay'):
        live.LiveOptions('libero_goal', 1, handoff_at_end=True, delay=10.).validate()


def test_normal_policy_skips_failure_injection(runner):
    viewer, session = runner
    live.run_policy.side_effect = None
    live.run_policy.return_value = dict(success=False, steps=2, model_id='test')
    result = viewer.run(live.LiveOptions('libero_goal', 1, mode='normal_policy'))
    assert result['status'] == 'normal_policy_failed'
    live.run_fault.assert_not_called()
    live.run_policy.assert_called_once()


def test_reads_existing_paired_layout_without_changing_it(runner, tmp_path):
    viewer, session = runner
    path = tmp_path/'experiment/layouts/libero_goal_t001_shuffled.json'
    path.parent.mkdir(parents=True)
    text = json.dumps(dict(state=[7., 1., 2., 3., 4.], seed=1000, shuffle_mapping={'a': 'b'}))
    path.write_text(text)
    viewer.prepare(live.LiveOptions('libero_goal', 1, scene='shuffled'))
    np.testing.assert_array_equal(session.env.set_init_state.call_args.args[0], [7, 1, 2, 3, 4])
    session.shuffle_scene.assert_not_called()
    assert 'Saved paired layout' in viewer.scene_origin
    assert path.read_text() == text


def test_incompatible_failure_rejected_before_loading_simulation(runner):
    viewer, session = runner
    with pytest.raises(ValueError, match='not applicable'):
        viewer.prepare(live.LiveOptions('libero_goal', 1, injection_row=0))
    session.load.assert_not_called()
    assert live.eligible_rows(viewer.chain or GMMChain.load(
        viewer.plan_root/'libero_goal/task_001.json', 'libero_goal', 1), 'grasp_loss', 'high') == [2]


def test_normal_waypoint_timeout_restores_tick_and_controller_flags(runner, monkeypatch):
    viewer, session = runner
    original_tick = session.tick
    def run(chain, progress):
        session.env.sim.data.time = 10.
        session.tick()
    session.run.side_effect = run
    monkeypatch.setattr(live, 'suspend_images', lambda session: __import__('contextlib').nullcontext())
    result = viewer.run(live.LiveOptions('libero_goal', 1, mode='normal_waypoints', max_waypoint_seconds=1.))
    assert result['status'] == 'normal_waypoints_failed'
    assert 'limit' in result['waypoint_result']
    assert session.tick is original_tick
    assert not session.continue_on_failure
    live.run_policy.assert_not_called()


@pytest.mark.parametrize('changes', [dict(delay=-1), dict(delay=float('nan')),
    dict(max_waypoint_seconds=float('inf')), dict(max_policy_steps=0), dict(seed=-1), dict(port=65536)])
def test_invalid_options_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(live.LiveOptions('libero_goal', 1), **changes).validate()


def test_detection_latency_is_allowed_and_forwarded_to_fault_execution(runner):
    viewer, session = runner
    options = live.LiveOptions('libero_goal', 1, delay=live.DETECTION_LATENCY_SECONDS)
    options.validate()
    viewer.run(options)
    assert live.run_fault.call_args.args[3] == 10.
    assert live.run_fault.call_args.kwargs['handoff_after_waypoint'] is True


def test_default_handoff_is_immediate():
    assert live.LiveOptions('libero_goal', 1).delay == 0.


@pytest.mark.parametrize('kind', tuple(live.FAILURES))
@pytest.mark.parametrize('intensity', ('low', 'high'))
def test_live_failures_use_grasp_except_high_loss_in_next_transport(runner, kind, intensity):
    viewer, session = runner
    options = live.LiveOptions('libero_goal', 1, kind=kind, intensity=intensity)
    result = viewer.run(options)
    assert result['status'] == 'recovered'
    spec = live.run_fault.call_args.args[2]
    assert spec['injection_row'] == (2 if kind == 'grasp_loss' and intensity == 'high' else 1)
    assert live.run_fault.call_args.kwargs['injector_class'] is live.GraspOnlyInjector
    assert live.run_fault.call_args.kwargs['intensity_rule'] is live.grasp_intensities


@pytest.mark.parametrize('kind', ['transition', 'orientation'])
def test_fresh_pose_draws_with_identical_layout_seed(runner, monkeypatch, kind):
    viewer, session = runner
    seeds = iter([101, 202])
    monkeypatch.setattr(live.secrets, 'randbits', lambda bits: next(seeds))
    options = live.LiveOptions('libero_goal', 1, kind=kind, intensity='high')
    viewer.run(options)
    viewer.run(options)
    specs = [call.args[2] for call in live.run_fault.call_args_list]
    assert [spec['fault_seed'] for spec in specs] == [101, 202]
    draws = []
    for spec in specs:
        injector = live.GraspOnlyInjector.__new__(live.GraspOnlyInjector)
        Injector.__init__(injector, session, {0: dict(kind=kind, intensity='high')},
                          waypoints=[GMMWaypoint(name='Grasp')], seed=spec['fault_seed'])
        injector.start_row(0)
        if kind == 'transition':
            offset = injector.deviated([0., 0., 1.])-np.array([0., 0., 1.])
            assert offset[2] == 0.
            assert TRANSITION_BANDS['high'][0] <= np.linalg.norm(offset) <= TRANSITION_BANDS['high'][1]
        else:
            offset = np.degrees(injector.wrist_tilt()[1])
            total = np.degrees(Rotation.from_euler('xyz', offset, degrees=True).magnitude())
            assert 40. <= total <= 80.
        draws.append(offset)
    assert not np.allclose(*draws)
    assert session.args.seed == 1000
    session.load.assert_called_once()


def test_transition_history_survives_scene_restore_and_stop_before_vla(runner):
    viewer, session = runner
    original_fault = live.run_fault.side_effect
    seen = []
    def fault(*args, **kwargs):
        history = kwargs['injector_options']['transition_history']
        seen.append(history.copy())
        history['direction'] = [1., 0., 0.]
        return original_fault(*args, **kwargs)
    live.run_fault.side_effect = fault
    options = live.LiveOptions('libero_goal', 1, kind='transition')
    def stop(outcome):
        raise live.sp.Abort()
    with pytest.raises(live.sp.Abort):
        viewer.run(options, before_vla=stop)
    assert viewer.run(options)['status'] == 'recovered'
    assert seen == [{}, {'direction': [1., 0., 0.]}]


def test_high_grasp_loss_only_selects_transport_immediately_after_grasp():
    chain = GMMChain('libero_goal', 1, 'test', [
        GMMWaypoint(name=name, skill_role=role) for name,role in [
            ('Move', 'move'), ('Grasp', 'grasp'), ('Transport', 'transport'),
            ('Transport', 'transport'), ('Release', 'release'),
            ('Grasp', 'grasp'), ('Transport', 'transport')]])
    assert live.eligible_rows(chain, 'grasp_loss', 'high') == [2, 6]
    assert live.eligible_rows(chain, 'grasp_loss', 'low') == [1, 5]


@pytest.mark.parametrize('held', ['', 'bowl'])
@pytest.mark.parametrize('status', ['ok (transported)', 'failed: unfinished Transport'])
def test_high_grasp_loss_releases_only_a_held_object_after_transport(runner, monkeypatch, held, status):
    viewer, session = runner
    injector = live.GraspOnlyInjector.__new__(live.GraspOnlyInjector)
    Injector.__init__(injector, session, {1: dict(kind='grasp_loss', intensity='high')},
                      waypoints=[GMMWaypoint(name='Grasp'), GMMWaypoint(name='Transport')])
    injector.saved['run_subgoal'] = Mock(return_value='ok')
    monkeypatch.setattr(live, 'held_object', lambda world: held)
    injector.release = Mock(return_value=True)
    injector.start_row(0)
    assert injector._run_subgoal(SimpleNamespace(kind='grip', gripper=live.sp.GRIPPER_CLOSE)) == 'ok'
    assert injector.finish_waypoint(0, 'ok (grasped)') is None
    injector.release.assert_not_called()
    injector.start_row(1)
    assert injector.take_motion() is None  # every path executes without a midpoint drop
    assert injector.armed == 'grasp_loss'
    injector.release.assert_not_called()
    result = injector.finish_waypoint(1, status)
    if held and status.startswith('ok'):
        injector.release.assert_called_once()
        assert 'released at Transport end' in result
    else:
        injector.release.assert_not_called()
        assert result is None if not status.startswith('ok') else 'could not release' in result
    assert injector.armed is None
    assert injector.finish_waypoint(1, status) is None  # never drop twice


def test_controller_timers_separate_phases_and_exclude_pause():
    wall = [0.]
    timer = live.ControllerTimings(clock=lambda: wall[0])
    timer.switch('waypoints', 10.)
    wall[0] = 3.
    timer.pause(12.)
    wall[0] = 30.
    assert timer.snapshot(12.)['waypoints_wall_seconds'] == 3.
    timer.resume(12.)
    wall[0] = 31.
    timer.switch('vla', 13.)
    wall[0] = 35.
    counts = timer.snapshot(13.5)
    assert counts == dict(waypoints_sim_seconds=3., waypoints_wall_seconds=4.,
                          vla_sim_seconds=.5, vla_wall_seconds=4.)
    timer.switch(None, 13.5)
    wall[0] = 90.
    assert timer.snapshot(13.5) == counts


def test_timer_stops_when_policy_is_interrupted(runner):
    viewer, session = runner
    live.run_policy.side_effect = live.sp.Abort()
    with pytest.raises(live.sp.Abort):
        viewer.run(live.LiveOptions('libero_goal', 1))
    assert viewer.timings.phase is None
    counts = viewer.timings.snapshot(session.env.sim.data.time)
    assert counts['waypoints_sim_seconds'] == 2.
    assert counts['vla_sim_seconds'] == 0.


def test_recovery_waits_with_frozen_timers_before_starting_vla(runner, monkeypatch):
    viewer, session = runner
    wall = [0.]
    monkeypatch.setattr(live.time, 'monotonic', lambda: wall[0])
    waited = []
    def before_vla(outcome):
        live.run_policy.assert_not_called()
        assert outcome['observed_failure']
        assert viewer.timings.phase is None
        state = session.env.sim.get_state().flatten().copy()
        timers = viewer.timings.snapshot(session.env.sim.data.time)
        wall[0] += 90.
        np.testing.assert_array_equal(session.env.sim.get_state().flatten(), state)
        assert viewer.timings.snapshot(session.env.sim.data.time) == timers
        waited.append(True)
    result = viewer.run(live.LiveOptions('libero_goal', 1), before_vla=before_vla)
    assert waited
    assert result['status'] == 'recovered'
    assert result['controller_timings']['vla_wall_seconds'] == 0.
    live.run_policy.assert_called_once()


def test_stop_while_waiting_does_not_start_vla(runner):
    viewer, session = runner
    def before_vla(outcome):
        raise live.sp.Abort()
    with pytest.raises(live.sp.Abort):
        viewer.run(live.LiveOptions('libero_goal', 1), before_vla=before_vla)
    live.run_policy.assert_not_called()
    assert viewer.timings.phase is None


def test_ended_unreachable_attempt_offers_vla_without_claiming_a_measured_failure(runner):
    viewer, session = runner
    live.run_fault.side_effect = None
    live.run_fault.return_value = dict(observed_failure=False, waypoint_interrupted=True,
                                      waypoint_execution_stopped=True)
    live.run_policy.side_effect = None
    live.run_policy.return_value = dict(success=False, steps=1)
    waiting = Mock()
    result = viewer.run(live.LiveOptions('libero_goal', 1, kind='transition', intensity='high'),
                        before_vla=waiting)
    waiting.assert_called_once()
    assert not result['observed_failure']
    assert result['injection_attempt_stopped']
    assert result['handoff_ready']
    assert not result['fault_execution_completed']
    assert result['status'] == 'recovery_failed'
    live.run_policy.assert_called_once()


@pytest.mark.parametrize('changes', [dict(waypoint_completed=False), dict(waypoint_completion_event=None),
    dict(waypoint_completion_event=dict(row=2, sim_time=0.)),
    dict(fault_targets=[dict(planning='planned', target=[.2, 0., 0.], attempt_result='ok')]),
    dict(fault_targets=[dict(planning='planned', target=[0., 0., 0.], attempt_result='failed: did not settle')])])
def test_ended_incomplete_transition_can_enable_continue(runner, changes):
    viewer, session = runner
    live.run_fault.side_effect = None
    live.run_fault.return_value = dict(observed_failure=True, waypoint_completed=True,
        waypoint_execution_stopped=True,
        waypoint_completion_event=dict(row=1, sim_time=0.),
        fault_targets=[dict(planning='planned', target=[0., 0., 0.], attempt_result='ok')])
    live.run_fault.return_value.update(changes)
    live.run_policy.side_effect = None
    live.run_policy.return_value = dict(success=False, steps=1)
    waiting = Mock()
    result = viewer.run(live.LiveOptions('libero_goal', 1, kind='transition'), before_vla=waiting)
    assert result['status'] == 'recovery_failed'
    assert not result['fault_execution_completed']
    assert result['handoff_ready']
    waiting.assert_called_once()
    live.run_policy.assert_called_once()


def test_continue_requires_stopped_execution_even_when_other_completion_flags_are_true(runner):
    viewer, session = runner
    live.run_fault.side_effect = None
    live.run_fault.return_value = dict(observed_failure=True, waypoint_completed=True,
        waypoint_completion_event=dict(row=1, sim_time=0.), waypoint_execution_stopped=False)
    waiting = Mock()
    result = viewer.run(live.LiveOptions('libero_goal', 1), before_vla=waiting)
    assert not result['handoff_ready']
    waiting.assert_not_called()
    live.run_policy.assert_not_called()


def transition_detector(position):
    session = SimpleNamespace(world=SimpleNamespace(eef=lambda: np.asarray(position)),
        executor=SimpleNamespace(arrival_grace_seconds=1.),
        tick=lambda: None,
        env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=10.))))
    injector = live.GraspOnlyInjector.__new__(live.GraspOnlyInjector)
    Injector.__init__(injector, session, {0: dict(kind='transition', intensity='high')})
    injector.start_row(0, primitive='grasp')
    injector.clock = __import__('waypoint_recovery_benchmark').HandoffClock(0.)
    injector.watch = dict(kind='transition', start=np.array([0., 0., .2]),
        position=np.zeros(3), direction=np.array([1., 0., 0.]), distance=.25,
        changed=np.array([.25, 0., 0.]))
    return injector


def orientation_detector(intensity, seed=7):
    injector = transition_detector([0., 0., .2])
    Injector.__init__(injector, injector.session,
                      {0: dict(kind='orientation', intensity=intensity)}, seed=seed)
    injector.start_row(0, primitive='grasp')
    injector.take_motion()
    injector.watch = None
    injector.fault_targets = []
    return injector


@pytest.mark.parametrize('intensity', ['low', 'high'])
def test_live_orientation_randomizes_axis_with_bounded_total_rotation(intensity):
    draws, offsets = [], []
    low, high = TILT_BANDS[intensity]
    nominal = Rotation.from_euler('xyz', [90., -30., 20.], degrees=True).as_matrix()
    for seed in range(300):
        injector = orientation_detector(intensity, seed)
        axis, angles = injector.wrist_tilt()
        assert axis == 'xyz'
        assert injector.wrist_tilt() == (axis, angles)
        changed = nominal@Rotation.from_euler(axis, angles).as_matrix()
        total = np.degrees(Rotation.from_matrix(nominal.T@changed).magnitude())
        assert low <= total <= high
        yaw = np.degrees(angles[2])
        assert total >= high*.9-1e-10
        assert abs(yaw) >= high*.81-1e-10
        assert abs(yaw) >= total*live.ORIENTATION_MIN_YAW_FRACTION-1e-10
        draws.append(np.degrees(angles))
        offset = injector.orientation_offset()
        np.testing.assert_array_equal(offset, injector.orientation_offset())
        minimum, maximum = live.ORIENTATION_XY_BANDS[intensity]
        assert minimum <= np.linalg.norm(offset) <= maximum
        assert offset[2] == 0.
        offsets.append(offset)
    draws = np.asarray(draws)
    assert np.all(np.any(draws < 0., axis=0))
    assert np.all(np.any(draws > 0., axis=0))
    assert len({tuple(draw) for draw in draws}) == len(draws)
    assert len({tuple(offset) for offset in offsets}) == len(offsets)
    assert np.all(np.any(np.asarray(offsets)[:, :2] > 0., axis=0))
    assert np.all(np.any(np.asarray(offsets)[:, :2] < 0., axis=0))


@pytest.mark.parametrize('intensity', ['low', 'high'])
@pytest.mark.parametrize('rejection', ['unreachable: target IK', 'no collision-free path'])
def test_orientation_resamples_rejected_targets_before_motion(intensity, rejection):
    injector = orientation_detector(intensity)
    actual = {'position': np.zeros(3), 'rotation': np.eye(3)}
    injector.session.gripper_pose = lambda: (actual['position'], actual['rotation'])
    injector.session.world.eef = lambda: actual['position']
    draws, offsets = [], []
    def move(target, rotation):
        assert injector.clock.event is None
        draws.append(np.degrees(injector.wrist_tilt()[1]))
        offsets.append(np.asarray(target).copy())
        if len(draws) < 3:
            assert injector.session.env.sim.data.time == 10.
            return rejection
        actual['rotation'] = rotation
        actual['position'] = np.asarray(target).copy()
        injector.session.env.sim.data.time += 1.
        injector.observe_control()
        return 'ok'
    result = injector._faulty('orientation', move, np.zeros(3), np.eye(3))
    assert result.startswith('ok')
    assert len(draws) == len({tuple(draw) for draw in draws}) == 3
    assert len({tuple(offset) for offset in offsets}) == 3
    np.testing.assert_allclose(injector.clock.event['transition_offset_metres'], offsets[-1])
    np.testing.assert_array_equal(injector.nominal_fault_pose[0], np.zeros(3))
    low, high = TILT_BANDS[intensity]
    totals = np.degrees(Rotation.from_euler('xyz', draws, degrees=True).magnitude())
    assert np.all((totals >= low) & (totals <= high))
    assert np.all(np.abs(np.asarray(draws)[:, 2]) >= totals*live.ORIENTATION_MIN_YAW_FRACTION-1e-10)
    np.testing.assert_allclose(list(injector.clock.event['requested_axes_degrees'].values()), draws[-1])
    assert injector.clock.event['requested_degrees'] == pytest.approx(totals[-1])
    assert len(injector.fired) == 1


@pytest.mark.parametrize('intensity,requested,outside', [
    ('low', 20.1, 19.), ('low', 39.9, 41.),
    ('high', 40.1, 39.), ('high', 79.9, 81.)])
def test_orientation_onset_requires_measured_total_inside_band(intensity, requested, outside):
    injector = orientation_detector(intensity)
    injector.tilts[0] = ('xyz', (np.radians(requested), 0., 0.))
    actual = {'position': np.zeros(3), 'rotation': np.eye(3)}
    injector.session.gripper_pose = lambda: (actual['position'], actual['rotation'])
    injector.session.world.eef = lambda: actual['position']
    def move(target, rotation):
        actual['position'] = np.asarray(target).copy()
        actual['rotation'] = Rotation.from_euler('x', outside, degrees=True).as_matrix()
        injector.observe_control()
        assert injector.clock.event is None
        actual['rotation'] = rotation
        injector.observe_control()
        assert injector.clock.event['orientation_degrees'] == pytest.approx(requested)
        return 'ok'
    assert injector._faulty('orientation', move, np.zeros(3), np.eye(3)).startswith('ok')


@pytest.mark.parametrize('intensity', ['low', 'high'])
def test_combined_orientation_fault_requires_measured_translation_too(intensity):
    injector = orientation_detector(intensity)
    actual = dict(position=np.zeros(3), rotation=np.eye(3))
    injector.session.world.eef = lambda: actual['position']
    injector.session.gripper_pose = lambda: (actual['position'], actual['rotation'])
    def move(target, rotation):
        actual['rotation'] = rotation
        injector.observe_control()
        assert injector.clock.event is None, 'Rotation alone cannot deliver the combined fault'
        actual['position'] = np.asarray(target).copy()
        injector.observe_control()
        assert injector.clock.event is not None
        return 'ok'
    result = injector._faulty('orientation', move, np.zeros(3), np.eye(3))
    assert result.startswith('ok')
    record = injector.fault_targets[-1]
    np.testing.assert_array_equal(record['nominal_position'], np.zeros(3))
    np.testing.assert_allclose(record['position'], record['transition_offset_metres'])
    assert injector.clock.event['measured_transition_metres'] == pytest.approx(
        np.linalg.norm(record['transition_offset_metres']))


@pytest.mark.parametrize('intensity', ['low', 'high'])
def test_combined_fault_accepts_contact_error_within_existing_arrival_tolerance(intensity):
    injector = orientation_detector(intensity)
    injector.orientation_offsets = {0: np.array([.02, 0., 0.])}
    actual = dict(position=np.zeros(3), rotation=np.eye(3))
    injector.session.world.eef = lambda: actual['position']
    injector.session.gripper_pose = lambda: (actual['position'], actual['rotation'])
    def move(target, rotation):
        actual['position'] = target+[.015, .015, .02]
        actual['rotation'] = rotation
        injector.observe_control()
        return 'ok'
    injector._faulty('orientation', move, np.zeros(3), np.eye(3))
    assert injector.clock.event is not None
    assert injector.clock.event['measured_transition_metres'] > .02


def test_orientation_never_resamples_after_physical_movement():
    injector = orientation_detector('high')
    def move(target, rotation):
        injector.session.env.sim.data.time += .05
        return 'unreachable: stopped after movement'
    motion = Mock(side_effect=move)
    assert injector._faulty('orientation', motion, np.zeros(3), np.eye(3)).startswith('unreachable')
    motion.assert_called_once()
    assert injector.clock.event is None


def test_impossible_orientation_exhausts_random_draws_without_claiming_a_failure():
    injector = orientation_detector('high')
    draws = []
    def move(target, rotation):
        draws.append(injector.wrist_tilt()[1])
        return 'no collision-free path'
    result = injector._faulty('orientation', move, np.zeros(3), np.eye(3))
    assert len(set(draws)) == live.TRANSITION_PLAN_ATTEMPTS
    assert result.startswith('unreachable: no reachable tilted Grasp target')
    assert injector.clock.event is None and not injector.fired
    assert injector.session.env.sim.data.time == 10.


def test_stop_cancels_orientation_resampling():
    injector = orientation_detector('low')
    injector.session.tick = Mock(side_effect=[None, live.sp.Abort()])
    motion = Mock(return_value='unreachable: target IK')
    with pytest.raises(live.sp.Abort):
        injector._faulty('orientation', motion, np.zeros(3), np.eye(3))
    motion.assert_called_once()
    assert injector.clock.event is None


def collision_detector(monkeypatch, intensity='high', rotation=None):
    position = np.array([.1, .1, 1.])
    rotation = live.sp.DOWN if rotation is None else rotation
    data = SimpleNamespace(time=10.)
    world = SimpleNamespace(eef=lambda: position.copy(), _geoms_of=lambda name: [3],
        m=SimpleNamespace(geom_bodyid={5: 1}, body=lambda index: SimpleNamespace(name='table')),
        d=SimpleNamespace(contact=[], ncon=0))
    executor = SimpleNamespace(q_hold=position.copy(), gripper=live.sp.GRIPPER_OPEN,
        scene=SimpleNamespace(arm_qpos=lambda: position.copy(),
            ik=Mock(side_effect=lambda via, **kwargs: (via.copy(), 0.))))
    session = SimpleNamespace(world=world, executor=executor, tick=Mock(),
        env=SimpleNamespace(sim=SimpleNamespace(data=data), env=SimpleNamespace(control_timestep=.05)),
        gripper_pose=lambda: (position.copy(), rotation.copy()))
    injector = live.GraspOnlyInjector.__new__(live.GraspOnlyInjector)
    Injector.__init__(injector, session, {0: dict(kind='collision', intensity=intensity)}, seed=9)
    injector.clock = __import__('waypoint_recovery_benchmark').HandoffClock(0.)
    injector.watch = None
    injector.fault_targets = []
    injector.robot_geoms = {2, 4}
    injector.collision_history = {}
    injector.collision_directions = {}
    injector.start_row(0, primitive='grasp')
    injector.take_motion()
    monkeypatch.setattr(live, 'held_object', lambda world: '')
    def apply():
        assert executor.gripper == live.sp.GRIPPER_OPEN
        position[:] = executor.q_hold
        data.time += .05
        # Self contact and contacts between scene objects are not robot hits.
        world.d.contact = [SimpleNamespace(geom1=2, geom2=4, dist=-.001),
                           SimpleNamespace(geom1=3, geom2=5, dist=-.001)]
        world.d.ncon = 2
        injector.observe_physics()
        assert injector.clock.event is None or injector.watch['active']
        if injector.watch['active']:
            world.d.contact.append(SimpleNamespace(geom1=2, geom2=5, dist=-.001))
            world.d.ncon = 3
            injector.observe_physics()
    executor._apply = Mock(side_effect=apply)
    return injector


@pytest.mark.parametrize('intensity,limits,depth', [('low', (.04, .06), .06), ('high', (.06, .10), .12)])
@pytest.mark.parametrize('rotation', [live.sp.DOWN,
    Rotation.from_euler('xyz', [20., 30., 10.], degrees=True).as_matrix()])
def test_collision_is_a_three_second_pre_close_press_without_return(monkeypatch, intensity, limits, depth, rotation):
    injector = collision_detector(monkeypatch, intensity, rotation)
    session = injector.session
    def descend(target, rotation):
        assert session.executor.gripper == live.sp.GRIPPER_OPEN
        assert injector.clock.event is None and not injector.fault_targets
        session.executor._apply.assert_not_called()
        return 'ok'
    nominal = Mock(side_effect=descend)
    target = np.array([.1, .1, 1.])
    result = injector._faulty('collision', nominal, target, rotation)
    assert result.startswith('ok (injected collision')
    # Pose validation must not restore the nominal target before closing.
    from waypoint_gmm import check_learned_pose
    assert check_learned_pose(session, {}, result, target, rotation) == result
    assert session.executor._apply.call_count == 60
    def close(subgoal):
        assert injector.clock.event is not None
        assert injector.fault_targets[-1]['finished_sim_time'] == pytest.approx(13.)
        session.executor.gripper = subgoal.gripper
        return 'ok'
    injector.saved['run_subgoal'] = Mock(side_effect=close)
    injector._run_subgoal(live.sp.Subgoal('close', 'grip', gripper=live.sp.GRIPPER_CLOSE))
    injector.saved['run_subgoal'].assert_called_once()
    nominal.assert_called_once()
    np.testing.assert_array_equal(nominal.call_args.args[0], target)
    event = injector.clock.event
    assert event['body'] == 'table'
    assert event['sim_time'] >= 10.75
    direction = np.asarray(event['requested_direction'])
    local_direction = np.asarray(event['requested_local_direction'])
    assert event['requested_plane'] == 'local_xy'
    assert local_direction[2] == 0.
    np.testing.assert_allclose(direction, rotation@local_direction)
    assert direction @ rotation[:, 2] == pytest.approx(0., abs=1e-10)
    np.testing.assert_array_equal(injector.collision_history['direction'], local_direction)
    attempt = injector.fault_targets[-1]
    assert attempt['phase'] == 'grasp_press_before_close'
    low, high = limits
    assert low <= attempt['xy_offset_metres'] <= high
    offset, final = map(np.asarray, (attempt['offset_target'], attempt['target']))
    np.testing.assert_allclose(offset-target, direction*attempt['xy_offset_metres'])
    down = np.asarray(attempt['press_direction'])
    assert down[2] < 0. and abs(down @ rotation[:, 2]) == pytest.approx(1.)
    assert attempt['press_metres'] == depth
    np.testing.assert_allclose(final-offset, down*depth)
    np.testing.assert_allclose(session.world.eef(), final)
    np.testing.assert_allclose(session.executor.q_hold, final)
    assert session.executor._apply.call_count == 60
    assert attempt['finished_sim_time']-attempt['started_sim_time'] == pytest.approx(3.)
    assert injector.watch is None


def test_collision_resamples_unreachable_offset_before_physics(monkeypatch):
    injector = collision_detector(monkeypatch)
    ik = injector.session.executor.scene.ik
    ik.side_effect = lambda via, **kwargs: (via.copy(), 1. if ik.call_count <= 2 else 0.)
    injector._collision()
    assert [t['planning'] for t in injector.fault_targets] == ['unreachable', 'unreachable', 'ready']
    assert len({tuple(t['direction']) for t in injector.fault_targets}) == 3
    assert injector.session.executor._apply.call_count == 60
    assert injector.session.env.sim.data.time == pytest.approx(13.)


def test_unreachable_collision_never_moves_or_reports_contact(monkeypatch):
    injector = collision_detector(monkeypatch)
    injector.session.executor.scene.ik.side_effect = lambda via, **kwargs: (via.copy(), 1.)
    assert injector._collision().startswith('unreachable')
    injector.session.executor._apply.assert_not_called()
    assert injector.clock.event is None and not injector.fired
    assert injector.collision_history == {}


def test_collision_without_scene_contact_is_reported_as_missed(monkeypatch):
    injector = collision_detector(monkeypatch)
    def apply():
        injector.session.env.sim.data.time += .05
    injector.session.executor._apply.side_effect = apply
    result = injector._collision()
    assert 'without measured scene contact' in result
    assert injector.clock.event is None and not injector.fired
    assert injector.missed
    assert injector.session.env.sim.data.time == pytest.approx(13.)


def test_stop_during_collision_press_cleans_up_without_retraction(monkeypatch):
    injector = collision_detector(monkeypatch)
    injector.session.executor._apply.side_effect = [None, live.sp.Abort()]
    with pytest.raises(live.sp.Abort):
        injector._collision()
    assert injector.session.executor._apply.call_count == 2
    assert injector.watch is None
    assert not injector.fired


@pytest.mark.parametrize('failure', ['unreachable: IK', 'failed: interrupted descent'])
def test_collision_does_not_skip_an_unfinished_nominal_descent(monkeypatch, failure):
    injector = collision_detector(monkeypatch)
    original = Mock(return_value=failure)
    result = injector._faulty('collision', original, np.zeros(3), live.sp.DOWN)
    assert result == failure
    assert not injector.fault_targets and injector.clock.event is None
    injector.session.executor._apply.assert_not_called()


def test_collision_directions_are_random_cached_and_separated_from_last_execution():
    injector = transition_detector([0., 0., .2])
    injector.collision_directions = {}
    injector.collision_history = {}
    injector.random = np.random.default_rng(13)
    directions = []
    for _ in range(100):
        injector.collision_directions.clear()
        direction = injector.collision_direction()
        np.testing.assert_array_equal(injector.collision_direction(), direction)
        directions.append(direction)
    assert {(v[0] > 0, v[1] > 0) for v in directions} == {(True, True), (True, False), (False, True), (False, False)}
    injector.collision_history['direction'] = directions[-1].tolist()
    previous = directions[-1]
    for _ in range(30):
        injector.collision_directions.clear()
        direction = injector.collision_direction()
        gap = np.arccos(np.clip(previous @ direction, -1., 1.))
        assert gap >= live.TRANSITION_DIRECTION_SEPARATION-1e-10


def test_high_transition_detected_at_onset_before_full_target_arrival():
    injector = transition_detector([.012, 0., .18])
    injector.observe_control()
    event = injector.clock.event
    assert event is not None
    assert event['offset_metres'] == pytest.approx(.012)
    assert event['requested_metres'] == .25
    assert event['onset_threshold_metres'] == .01


def test_transition_onset_does_not_count_normal_approach():
    injector = transition_detector([0., 0., .15])
    injector.observe_control()
    assert injector.clock.event is None


@pytest.mark.parametrize('intensity', ['low', 'high'])
def test_transition_directions_cover_all_quadrants_with_a_fixed_layout_seed(intensity):
    injector = transition_detector([0., 0., .2])
    injector.intensity = intensity
    injector.random = np.random.default_rng(7)
    angles = []
    for _ in range(200):
        injector.directions.clear()
        injector.distances.clear()
        direction = injector.transition_direction()
        assert direction[2] == 0.
        assert np.linalg.norm(direction) == pytest.approx(1.)
        low, high = TRANSITION_BANDS[intensity]
        assert low <= injector.transition_distance() <= high
        np.testing.assert_array_equal(injector.transition_direction(), direction)
        angles.append(np.arctan2(direction[1], direction[0]) % (2*np.pi))
    assert {int(angle/(np.pi/2)) for angle in angles} == {0, 1, 2, 3}


def test_consecutive_executed_transitions_avoid_the_previous_direction(monkeypatch):
    history = {}
    executed = []
    for _ in range(6):
        position = [0., 0., .2]
        injector = transition_detector(position)
        injector.transition_history = history
        # Even repeated seed input must not replay the same executed direction.
        injector.random = np.random.default_rng(1)
        injector.fault_targets = []
        injector.take_motion()
        monkeypatch.setattr(live, 'held_object', lambda world: '')
        def move(session, target, rotation, **kwargs):
            position[:] = target.tolist()
            session.env.sim.data.time += 1.
            injector.observe_control()
            return 'ok', 'planned'
        monkeypatch.setattr(live, 'guarded_move', move)
        assert injector._faulty('transition', Mock(), np.zeros(3), np.eye(3)).startswith('ok')
        direction = np.asarray(injector.clock.event['requested_direction'])
        np.testing.assert_array_equal(history['direction'], direction)
        np.testing.assert_allclose(injector.transition_direction(), direction)
        if executed:
            separation = np.arccos(np.clip(direction @ executed[-1], -1., 1.))
            assert separation >= live.TRANSITION_DIRECTION_SEPARATION-1e-10
        executed.append(direction)


@pytest.mark.parametrize('planner_result,route', [(None, 'unchecked'),
    ('no collision-free route to the target', 'blocked')])
def test_blocked_transition_never_uses_unchecked_fallback(runner, monkeypatch, planner_result, route):
    viewer, session = runner
    session.world = SimpleNamespace(eef=lambda: np.array([0., 0., .2]))
    injector = transition_detector([0., 0., .2])
    injector.fault_targets = []
    injector.session = session
    history = {'direction': [1., 0., 0.]}
    injector.transition_history = history
    injector.take_motion()
    monkeypatch.setattr(live, 'held_object', lambda world: '')
    planner = Mock(return_value=(planner_result, route))
    monkeypatch.setattr(live, 'guarded_move', planner)
    unchecked = Mock()
    result = injector._faulty('transition', unchecked, np.zeros(3), np.eye(3))
    assert result.startswith('unreachable') and 'random samples' in result
    unchecked.assert_not_called()
    assert planner.call_count == live.TRANSITION_PLAN_ATTEMPTS
    assert planner.call_args.kwargs == dict(ignore=(), carried=(), plan_first=True)
    assert session.executor.arrival_grace_seconds == 1.
    assert injector.clock.event is None
    assert history == {'direction': [1., 0., 0.]}
    assert any(target.get('planning') == route for target in injector.fault_targets)


@pytest.mark.parametrize('intensity', ['low', 'high'])
@pytest.mark.parametrize('rejected_route', ['blocked', 'unchecked'])
def test_transition_resamples_planning_rejections_before_moving(monkeypatch, intensity, rejected_route):
    position = [0., 0., .2]
    injector = transition_detector(position)
    injector.intensity = intensity
    injector.random = np.random.default_rng(101)
    injector.fault_targets = []
    injector.take_motion()
    monkeypatch.setattr(live, 'held_object', lambda world: '')
    targets = []
    def planner(session, target, rotation, **kwargs):
        targets.append(target.copy())
        assert injector.clock.event is None
        assert session.env.sim.data.time == 10.
        np.testing.assert_array_equal(position, [0., 0., .2])
        if len(targets) < 3:
            return ((None, 'unchecked') if rejected_route == 'unchecked' else
                    ('no collision-free route to the target', 'blocked'))
        position[:] = target.tolist()
        session.env.sim.data.time += 1.
        injector.observe_control()
        return 'ok (collision-free route)', 'planned'
    monkeypatch.setattr(live, 'guarded_move', planner)
    result = injector._faulty('transition', Mock(), np.zeros(3), np.eye(3))
    assert result.startswith('ok')
    assert len(targets) == 3
    assert len({tuple(target) for target in targets}) == 3
    low, high = TRANSITION_BANDS[intensity]
    assert all(target[2] == 0. and low <= np.linalg.norm(target) <= high for target in targets)
    distances = [np.linalg.norm(target) for target in targets]
    assert len(set(distances)) == 3
    routes = [target for target in injector.fault_targets if 'planning' in target]
    assert [target['planning'] for target in routes] == [rejected_route, rejected_route, 'planned']
    assert [target['sample'] for target in routes] == [1, 2, 3]
    event = injector.clock.event
    assert event['requested_metres'] == pytest.approx(distances[-1])
    np.testing.assert_allclose(event['requested_direction'], targets[-1]/distances[-1])
    assert len(injector.fired) == 1
    assert injector.session.executor.arrival_grace_seconds == 1.


@pytest.mark.parametrize('route', ['planned', 'line'])
def test_transition_does_not_resample_an_executed_route(monkeypatch, route):
    injector = transition_detector([0., 0., .2])
    injector.fault_targets = []
    injector.take_motion()
    monkeypatch.setattr(live, 'held_object', lambda world: '')
    planner = Mock(return_value=('failed: robot did not settle on the planned path', route))
    monkeypatch.setattr(live, 'guarded_move', planner)
    result = injector._faulty('transition', Mock(), np.zeros(3), np.eye(3))
    planner.assert_called_once()
    assert result.startswith('failed')


def test_stop_interrupts_resampling_before_another_planning_attempt(monkeypatch):
    injector = transition_detector([0., 0., .2])
    injector.fault_targets = []
    injector.take_motion()
    injector.session.tick = Mock(side_effect=[None, live.sp.Abort()])
    monkeypatch.setattr(live, 'held_object', lambda world: '')
    planner = Mock(return_value=('no collision-free route to the target', 'blocked'))
    monkeypatch.setattr(live, 'guarded_move', planner)
    with pytest.raises(live.sp.Abort):
        injector._faulty('transition', Mock(), np.zeros(3), np.eye(3))
    planner.assert_called_once()
    assert injector.clock.event is None
    assert injector.session.executor.arrival_grace_seconds == 1.


def test_transition_reaches_full_displaced_target_before_waypoint_handoff(monkeypatch):
    position = [0., 0., .2]
    injector = transition_detector(position)
    injector.fault_targets = []
    injector.take_motion()
    monkeypatch.setattr(live, 'held_object', lambda world: '')
    def planner(session, target, orientation, **kwargs):
        assert session.executor.arrival_grace_seconds is None
        position[:] = (np.asarray(position)+(target-position)*.5).tolist()
        injector.observe_control()
        assert injector.clock.event is not None
        # Onset is recorded, but the full route and the waypoint may finish.
        position[:] = target.tolist()
        session.env.sim.data.time += 2.
        return 'ok (collision-free route, 4 configs)', 'planned'
    monkeypatch.setattr(live, 'guarded_move', Mock(side_effect=planner))
    unchecked = Mock()
    result = injector._faulty('transition', unchecked, np.zeros(3), np.eye(3))
    assert result.startswith('ok')
    unchecked.assert_not_called()
    route = next(t for t in injector.fault_targets if t.get('planning') == 'planned')
    assert route['target_gap_metres'] == 0.
    assert .15 <= np.linalg.norm(route['target']) <= .25
    assert route['finished_sim_time'] > injector.clock.event['sim_time']
    assert injector.session.executor.arrival_grace_seconds == 1.


@pytest.mark.parametrize('delay', [0., 10.])
def test_transport_requires_latency_after_finished_injection_row(runner, delay):
    viewer, session = runner
    entered = []
    def fault(session, chain, spec, delay, limit, folder, **kwargs):
        try:
            kwargs['progress'](1, 'Running')
            kwargs['progress'](1, 'intermediate motion status')
            kwargs['progress'](2, 'Running')
            entered.append('transport')
            return dict(observed_failure=False, waypoint_execution_stopped=True,
                        waypoint_completed=True)
        except live.Takeover:
            return dict(observed_failure=False, waypoint_interrupted=True)
        pytest.fail('Waypoint playback continued beyond the failed Grasp')
    live.run_fault.side_effect = fault
    live.run_policy.side_effect = None
    live.run_policy.return_value = dict(success=True)
    result = viewer.run(live.LiveOptions('libero_goal', 1, delay=delay))
    assert bool(entered) == bool(delay)
    assert result['injection_attempt_stopped']
