"""Recovery scheduling must distinguish missed faults and preserve experiment pairs."""
from types import SimpleNamespace
from contextlib import contextmanager
from dataclasses import asdict
from unittest.mock import Mock
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from waypoint_gmm import GMMChain, GMMWaypoint as Waypoint
from waypoint_recovery_benchmark import (HandoffClock, MeasuredInjector, observe_physics,
                                         scenarios, select_row, summarize, trial_id)
import waypoint_recovery_benchmark as benchmark


@pytest.mark.parametrize('intensity, limits', [('low', (10., 20.)), ('high', (20., 40.))])
def test_measured_orientation_tracks_all_three_angles_and_the_combined_turn(intensity, limits):
    position = np.array([.2, 0., 1.])
    nominal = Rotation.from_euler('xyz', [90, -20, 10], degrees=True).as_matrix()
    actual = dict(position=position, rotation=nominal)
    session = SimpleNamespace(
        executor=None, injected_failure='',
        world=SimpleNamespace(eef=lambda: position),
        env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=10.))),
        gripper_pose=lambda: (actual['position'], actual['rotation']))
    injector = MeasuredInjector.__new__(MeasuredInjector)
    benchmark.Injector.__init__(injector, session, {0: dict(kind='orientation', intensity=intensity)}, seed=7)
    injector.clock = HandoffClock(2.)
    injector.watch = None
    injector.fault_targets = []
    injector.start_row(0, primitive='grasp')

    def move(target, orientation):
        actual.update(position=target, rotation=orientation)
        injector.observe_control()
        return 'ok'

    injector._faulty('orientation', move, position, nominal)
    event = injector.clock.event
    assert event is not None
    assert event['requested_axis'] == 'roll, pitch, yaw'
    assert set(event['requested_axes_degrees']) == {'roll', 'pitch', 'yaw'}
    assert all(limits[0] <= angle <= limits[1] for angle in event['requested_axes_degrees'].values())
    actual_turn = np.degrees(Rotation.from_matrix(nominal.T @ actual['rotation']).magnitude())
    assert event['requested_degrees'] == pytest.approx(actual_turn)
    assert event['orientation_degrees'] == pytest.approx(actual_turn)
    assert injector.fault_targets[0]['axis'] == 'xyz'
    assert len(injector.fault_targets[0]['angles']) == 3
    json.dumps(injector.fault_targets)


def test_high_orientation_feasibility_checks_combined_corner_rotations(monkeypatch):
    chain = SimpleNamespace(waypoints=[Waypoint(name='Grasp', skill_role='grasp')])
    session = SimpleNamespace(
        preview=lambda chain: [(np.array([.1, 0., 0.]), np.eye(3))],
        gripper_pose=lambda: (np.zeros(3), np.eye(3)),
        args=SimpleNamespace(line_step=.1),
        executor=SimpleNamespace(scene=SimpleNamespace(ik=Mock(return_value=(np.zeros(7), 0.)))))
    draw = Mock(wraps=benchmark.tilt)
    monkeypatch.setattr(benchmark, 'tilt', draw)
    assert benchmark.feasible_row(session, chain, dict(kind='orientation', intensity='high')) == 0
    assert draw.call_count == 8
    corners = set()
    for call in draw.call_args_list:
        assert call.args[2] == 'xyz'
        corners.add(tuple(np.round(np.degrees(call.args[1]), 6)))
    assert corners == {(roll, pitch, yaw) for roll in (20., 40.)
                       for pitch in (20., 40.) for yaw in (20., 40.)}


def test_handoff_uses_simulation_time_since_first_fault():
    clock = HandoffClock(2.)
    assert not clock.due(1000.)
    clock.mark(8.35, kind='collision')
    clock.mark(9.5, kind='collision')
    assert not clock.due(10.30)
    assert clock.due(10.35)
    assert clock.event['sim_time'] == 8.35


@pytest.mark.parametrize('delay', [-1., float('nan'), float('inf')])
def test_invalid_delays_rejected(delay):
    with pytest.raises(ValueError):
        HandoffClock(delay)


def test_full_matrix_is_paired_and_unique():
    matrix = scenarios()
    assert len(matrix) == 160
    assert len({trial_id(s) for s in matrix}) == 160
    assert len({(s['suite'], s['task_id']) for s in matrix}) == 10
    for suite, task in {(s['suite'], s['task_id']) for s in matrix}:
        rows = [s for s in matrix if (s['suite'], s['task_id']) == (suite, task)]
        assert len(rows) == 16
        assert len({s['seed'] for s in rows}) == 1
        assert {s['scene'] for s in rows} == {'original', 'shuffled'}


def test_fault_placement_respects_transport_and_move_exclusions():
    chain = SimpleNamespace(waypoints=[
        Waypoint(name='Move', skill_role='move'),
        Waypoint(name='Grasp', skill_role='grasp'),
        Waypoint(name='Transport', skill_role='transport'),
        Waypoint(name='Transport', skill_role='transport'),
        Waypoint(name='Release', skill_role='release', motion='grip')])
    for kind in ('collision', 'transition', 'orientation'):
        for intensity in ('low', 'high'):
            assert select_row(chain, kind, intensity) == 1
    assert select_row(chain, 'grasp_loss', 'low') == 1
    assert select_row(chain, 'grasp_loss', 'high') == 3
    with pytest.raises(ValueError, match='No applicable'):
        select_row(SimpleNamespace(waypoints=chain.waypoints[:1]), 'collision', 'low')


def test_retry_only_uses_untried_eligible_rows():
    from waypoint_recovery_retry import alternative_rows
    chain = SimpleNamespace(waypoints=[
        Waypoint(name='Move', skill_role='move'),
        Waypoint(name='Grasp', skill_role='grasp'),
        Waypoint(name='Transport', skill_role='transport'),
        Waypoint(name='Place', skill_role='place'),
        Waypoint(name='Release', skill_role='release', motion='grip')])
    missed = dict(injection_row=1, kind='orientation', intensity='high')
    assert alternative_rows(chain, missed) == [3]
    missed['attempts'] = [dict(injection_row=3)]
    assert alternative_rows(chain, missed) == []
    assert alternative_rows(chain, dict(injection_row=2, kind='grasp_loss', intensity='high')) == []


def test_paired_layout_restores_the_exact_post_shuffle_state_without_settling(tmp_path):
    demo = tmp_path/'demo.json'
    demo.write_text(json.dumps(dict(initial_state=[1., 2., 3.])))
    model = tmp_path/'model.json'
    model.write_text(json.dumps(dict(demonstration_files=[str(demo)])))
    chain = SimpleNamespace(model_path=str(model))
    state = [1., 2., 3.]
    def restore():
        state[:] = [1., 2., 3.]
    def shuffle(seed):
        state[:] = [4., 5., 6.]
        session.last_shuffle_mapping = {'a': 'b', 'b': 'a'}
    def set_state(value):
        state[:] = value.tolist()
    session = SimpleNamespace(restore_training_scene=Mock(side_effect=restore),
        shuffle_scene=Mock(side_effect=shuffle), attach=Mock(),
        executor=SimpleNamespace(gripper=benchmark.sp.GRIPPER_OPEN),
        env=SimpleNamespace(set_init_state=Mock(side_effect=set_state),
                            sim=SimpleNamespace(get_state=lambda:np.asarray(state))))
    spec = dict(suite='libero_goal', task_id=1, scene='shuffled', seed=1000)
    benchmark.prepare_scene(session, chain, spec, tmp_path/'layouts')
    state[:] = [99., 99., 99.]
    benchmark.prepare_scene(session, chain, spec, tmp_path/'layouts')
    assert state == [4., 5., 6.]
    session.shuffle_scene.assert_called_once_with(seed=1000)
    session.attach.assert_called_once_with(settle=False)
    assert session.last_shuffle_mapping == {'a': 'b', 'b': 'a'}


@pytest.mark.parametrize('row', [0, 2, 4, -1, 99])
def test_explicit_injection_row_cannot_bypass_primitive_rules(tmp_path, row):
    chain = SimpleNamespace(waypoints=[
        Waypoint(name='Move', skill_role='move'),
        Waypoint(name='Grasp', skill_role='grasp'),
        Waypoint(name='Transport', skill_role='transport'),
        Waypoint(name='Place', skill_role='place'),
        Waypoint(name='Release', skill_role='release', motion='grip')])
    session = Mock()
    with pytest.raises(ValueError):
        benchmark.run_fault(session, chain, dict(kind='collision', intensity='high', injection_row=row),
                            2., 180., tmp_path)
    session.run.assert_not_called()


def test_no_injection_and_preexisting_success_are_excluded_from_recovery_rate():
    base = dict(scene='original', kind='collision', intensity='low')
    report = summarize([dict(base, status='recovered', success=True),
                        dict(base, status='recovery_failed', success=False),
                        dict(base, status='failure_not_observed', success=False),
                        dict(base, status='already_successful', success=True),
                        dict(base, status='error', success=False)])
    assert report['total'] == 5
    assert report['recovery_trials'] == 2
    assert report['successes'] == 1
    assert report['groups']['original/collision/low'] == dict(trials=2, successes=1)


def test_physics_observation_restores_step_after_interruption():
    sim = SimpleNamespace(step=Mock(return_value=123))
    session = SimpleNamespace(env=SimpleNamespace(sim=sim))
    original = sim.step
    injector = SimpleNamespace(observe_physics=Mock())
    with pytest.raises(RuntimeError):
        with observe_physics(session, injector):
            assert sim.step() == 123
            raise RuntimeError('takeover')
    assert sim.step is original
    injector.observe_physics.assert_called_once()


def detector(watch, position):
    obj = object.__new__(MeasuredInjector)
    obj.clock = HandoffClock(2.)
    obj.watch = watch
    obj.mark = Mock()
    obj.session = SimpleNamespace(world=SimpleNamespace(eef=lambda: np.asarray(position)))
    return obj


def test_transition_requires_measured_offset_from_the_nominal_target():
    obj = detector(dict(kind='transition', position=np.array([.3, 0., .1]),
                        changed=np.array([.3, .12, .1]), direction=np.array([0., 1., 0.]),
                        distance=.12), [.3, .02, .1])
    obj.observe_control()
    obj.mark.assert_not_called()
    obj.session.world.eef = lambda: np.array([.3, .10, .1])
    obj.observe_control()
    assert obj.mark.call_args.kwargs['offset_metres'] == .10


@pytest.mark.parametrize('direction', [
    [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
    [0., -1., 0.], [.6, .8, 0.]])
def test_transition_detector_accepts_horizontal_directions(direction):
    direction = np.asarray(direction)
    nominal = np.array([.3, .2, .1])
    changed = nominal + .12*direction
    obj = detector(dict(kind='transition', position=nominal, changed=changed,
                        direction=direction, distance=.12), nominal)
    obj.observe_control()
    obj.mark.assert_not_called()
    obj.session.world.eef = lambda: changed
    obj.observe_control()
    assert obj.mark.call_args.kwargs['offset_metres'] == pytest.approx(.12)
    assert obj.mark.call_args.kwargs['requested_direction'] == direction.tolist()
    assert obj.mark.call_args.kwargs['requested_plane'] == 'xy'


@pytest.mark.parametrize('height', [-.5, .5])
def test_vertical_motion_cannot_trigger_a_horizontal_transition(height):
    nominal = np.array([.3, .2, .1])
    obj = detector(dict(kind='transition', position=nominal,
        changed=nominal+np.array([.12, 0., 0.]), direction=np.array([1., 0., 0.]),
        distance=.12), nominal+np.array([0., 0., height]))
    obj.observe_control()
    obj.mark.assert_not_called()


@pytest.mark.parametrize('direction', [[1., 0., 0.], [-1., 0., 0.]])
def test_transition_does_not_mark_normal_forward_or_backward_approach(direction):
    direction = np.asarray(direction)
    nominal = np.array([.3, 0., 0.])
    changed = nominal + .12*direction
    obj = detector(dict(kind='transition', position=nominal, changed=changed,
                        direction=direction, distance=.12), [.1, 0., 0.])
    obj.observe_control()
    obj.mark.assert_not_called()
    obj.session.world.eef = lambda: changed
    obj.observe_control()
    obj.mark.assert_called_once()


@pytest.mark.parametrize('intensity, limits', [('low', (.05, .15)), ('high', (.15, .25))])
def test_transition_monitor_and_controller_use_the_same_random_target(intensity, limits):
    nominal = np.array([.3, .2, 1.])
    actual = dict(position=np.zeros(3))
    session = SimpleNamespace(
        executor=None, injected_failure='',
        world=SimpleNamespace(eef=lambda: actual['position']),
        env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=10.))))
    injector = MeasuredInjector.__new__(MeasuredInjector)
    benchmark.Injector.__init__(injector, session, {0: dict(kind='transition', intensity=intensity)}, seed=7)
    injector.clock = HandoffClock(2.)
    injector.watch = None
    injector.fault_targets = []
    injector.start_row(0, primitive='grasp')

    def move(position, orientation):
        actual['position'] = position
        injector.observe_control()
        return 'ok'

    injector._faulty('transition', move, nominal, np.eye(3))
    event = injector.clock.event
    assert event is not None
    distance = injector.transition_distance()
    assert limits[0] <= distance <= limits[1]
    assert event['requested_metres'] == pytest.approx(distance)
    assert event['offset_metres'] == pytest.approx(distance)
    target = injector.fault_targets[0]
    np.testing.assert_allclose(actual['position'], target['changed'])
    np.testing.assert_allclose(actual['position'] - nominal, distance*np.asarray(target['direction']))
    assert target['direction'][2] == 0.
    assert actual['position'][2] == nominal[2]
    assert event['requested_plane'] == 'xy'
    json.dumps(injector.fault_targets)


def test_collision_requires_real_robot_contact_with_selected_body():
    obj = detector(dict(kind='collision', body='obstacle', geoms={5}), [0., 0., 0.])
    obj.robot_geoms = {2}
    contacts = [SimpleNamespace(geom1=5, geom2=6, dist=-.003),
                SimpleNamespace(geom1=2, geom2=8, dist=-.003),
                SimpleNamespace(geom1=2, geom2=5, dist=.001)]
    obj.session.world.d = SimpleNamespace(contact=contacts, ncon=3)
    obj.observe_physics()
    obj.mark.assert_not_called()
    contacts.append(SimpleNamespace(geom1=2, geom2=5, dist=-.002))
    obj.session.world.d.ncon = 4
    obj.observe_physics()
    assert obj.mark.call_args.kwargs['penetration_metres'] == .002


def test_high_collision_forward_push_is_recorded_without_restarting_the_clock():
    from waypoint_failures import HIGH_COLLISION_OFFSET
    obj = detector(dict(kind='collision', body='obstacle', geoms={5}, active=False), [.1, .2, .9])
    obj.session.gripper_pose = lambda: (obj.session.world.eef(), benchmark.sp.DOWN)
    obj.fault_targets = []
    obj.clock.mark(8.35, kind='collision')
    target, orientation = obj.collision_push_pose()
    np.testing.assert_allclose(target, [.1, .2, .9-HIGH_COLLISION_OFFSET])
    assert obj.fault_targets[-1]['phase'] == 'gripper_forward_push'
    assert obj.fault_targets[-1]['body'] == 'obstacle'
    assert obj.fault_targets[-1]['requested_forward_offset_metres'] == HIGH_COLLISION_OFFSET
    assert obj.watch['geoms'] == {5}
    assert obj.watch['active']
    assert obj.clock.event['sim_time'] == 8.35


def test_high_collision_approach_does_not_start_the_takeover_clock():
    obj = detector(dict(kind='collision', body='obstacle', geoms={5}, active=False), [0., 0., 0.])
    obj.robot_geoms = {2}
    obj.session.world.d = SimpleNamespace(
        contact=[SimpleNamespace(geom1=2, geom2=5, dist=-.002)], ncon=1)
    obj.observe_physics()
    obj.mark.assert_not_called()
    obj.watch['active'] = True
    obj.observe_physics()
    assert obj.mark.call_args.kwargs['penetration_metres'] == .002


@pytest.mark.parametrize('observed', [True, False])
def test_fault_run_continues_to_deadline_without_reset(monkeypatch, tmp_path, observed):
    data = SimpleNamespace(time=0.)
    env = Mock()
    env.sim.data = data
    env.check_success.return_value = False
    previous_tick = Mock()
    s = SimpleNamespace(env=env, tick=previous_tick, continue_on_failure=False)

    class FakeInjector:
        def __init__(self, session, injections, clock, waypoints):
            self.clock = clock
            self.fired, self.missed = [], []
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def start_row(self, row):
            if observed:
                self.clock.mark(data.time, kind='collision')
        def observe_control(self):
            pass
    @contextmanager
    def context(*args):
        yield
    def hold(steps):
        data.time += .05*steps
        s.tick()
    def run(chain, report):
        report(0, 'Running')
        hold(10)
        return 'Finished'
    s.executor = SimpleNamespace(hold=hold)
    s.run = run
    monkeypatch.setattr(benchmark, 'MeasuredInjector', FakeInjector)
    monkeypatch.setattr(benchmark, 'observe_physics', context)
    monkeypatch.setattr(benchmark, 'suspend_images', context)
    monkeypatch.setattr(benchmark, 'save_scene', Mock())
    chain = SimpleNamespace(waypoints=[Waypoint(name='Grasp', skill_role='grasp')])
    result = benchmark.run_fault(s, chain, dict(kind='collision', intensity='low'), 2., 180., tmp_path)
    assert result['observed_failure'] == observed
    if observed:
        assert result['actual_delay_seconds'] == pytest.approx(2.)
        assert result['waypoint_interrupted']
    else:
        assert data.time == .5
        assert 'handoff_sim_time' not in result
        benchmark.save_scene.assert_not_called()
    assert s.tick is previous_tick
    assert not s.continue_on_failure
    env.reset.assert_not_called()
    env.set_init_state.assert_not_called()


@pytest.fixture
def latency_case(monkeypatch):
    def execute(kind='transition', delay=10., duration=1.5, ending='complete', observed=True,
                limit=180., stop=False, continuation=2., handoff_at_end=False):
        from waypoint_recovery_live import grasp_intensities
        data = SimpleNamespace(time=0.)
        env = Mock()
        env.sim.data = data
        env.check_success.return_value = False
        original_tick = Mock()
        session = SimpleNamespace(env=env, tick=original_tick, continue_on_failure=False)
        target = np.array([.2, -.1, .8])
        holds, moves = [], []
        progress = Mock()
        capture = Mock()
        class Injector:
            def __init__(self, session, injections, clock, waypoints):
                self.clock = clock
                self.fired, self.missed = [], []
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def start_row(self, row):
                if observed:
                    self.clock.mark(data.time, kind=kind)
            def observe_control(self):
                pass
        def hold(steps):
            assert steps == 1
            np.testing.assert_array_equal(session.executor.q_hold, target)
            holds.append(float(data.time))
            if stop:
                raise benchmark.sp.Abort()
            data.time += .05
            session.tick()
        def run(chain, report):
            report(0, 'Running')
            data.time += duration
            session.tick()
            report(0, 'failed: displaced Grasp')
            if ending == 'complete':
                session.on_waypoint_finished(0, 'failed: displaced Grasp')
                if delay == 0 and not handoff_at_end:
                    pytest.fail('Playback advanced after Grasp completion without latency')
                report(1, 'Running')
                for _ in range(int(round(continuation/.05))):
                    if stop:
                        raise benchmark.sp.Abort()
                    target[0] += .001
                    session.executor.q_hold = target.copy()
                    moves.append(target.copy())
                    data.time += .05
                    session.tick()
                report(1, 'ok')
            return 'Stopped: waypoint plan ended'
        @contextmanager
        def context(*args):
            yield
        session.executor = SimpleNamespace(hold=hold, q_hold=target.copy())
        session.gripper_pose = lambda: (target.copy(), np.eye(3))
        session.run = Mock(side_effect=run)
        monkeypatch.setattr(benchmark, 'observe_physics', context)
        monkeypatch.setattr(benchmark, 'suspend_images', context)
        chain = SimpleNamespace(waypoints=[Waypoint(name='Grasp', skill_role='grasp'),
                                          Waypoint(name='Transport', skill_role='transport')])
        try:
            result = benchmark.run_fault(session, chain,
                dict(kind=kind, intensity='low', injection_row=0), delay, limit, None,
                injector_class=Injector, intensity_rule=grasp_intensities, capture=capture,
                progress=progress, handoff_after_waypoint=True, handoff_at_end=handoff_at_end)
        finally:
            assert session.tick is original_tick
            assert not session.continue_on_failure
            assert not hasattr(session, 'on_waypoint_finished')
            assert not hasattr(session, 'stop_after_waypoint')
            env.reset.assert_not_called()
            env.set_init_state.assert_not_called()
        session.run.assert_called_once()
        return result, holds, progress, capture, moves
    return execute


@pytest.mark.parametrize('kind', ['collision', 'transition', 'orientation', 'grasp_loss'])
@pytest.mark.parametrize('duration,delay,ending,observed,continuation,expected,steps,moving_steps', [
    (1.5, 10., 'complete', True, 2., 11.5, 160, 40),
    (16., 10., 'complete', True, 20., 26., 0, 200),
    (1.5, 0., 'complete', True, 2., 1.5, 0, 0),
    (1.5, 10., 'returned', True, 2., 11.5, 200, 0),
    (1.5, 10., 'returned', False, 2., 11.5, 200, 0),
    (1.5, 10., 'complete', False, 20., 11.5, 0, 200)])
def test_live_latency_continues_plan_for_ten_seconds_after_waypoint(latency_case, kind, duration,
        delay, ending, observed, continuation, expected, steps, moving_steps):
    result, holds, progress, capture, moves = latency_case(kind, delay, duration, ending, observed,
                                                         continuation=continuation)
    assert result['handoff_sim_time'] == pytest.approx(expected)
    assert result['detection_latency_seconds'] == delay
    assert result['detection_latency_completed']
    assert result['waypoint_completed'] == (ending == 'complete')
    assert result['waypoint_execution_stopped']
    assert result['actual_detection_latency_seconds'] == pytest.approx(delay)
    assert len(holds) == steps and len(moves) == moving_steps
    if delay:
        assert result['detection_latency_start_sim_time'] == duration
        np.testing.assert_array_equal(result['detection_latency_start_pose']['position'], [.2, -.1, .8])
    else:
        assert result['detection_latency_start_sim_time'] is None
    if moves:
        assert any(call.args[0] == 1 for call in progress.call_args_list)
        assert moves[-1][0] > moves[0][0]
    else:
        assert all(call.args[0] == 0 for call in progress.call_args_list)
    countdown = [call.args[1] for call in progress.call_args_list
                 if call.args[1].startswith('Detection latency')]
    assert bool(countdown) == bool(delay)
    if observed:
        assert result['actual_delay_seconds'] == pytest.approx(expected)
    else:
        assert result['failure_event'] is None
        assert 'actual_delay_seconds' not in result
    if ending == 'complete':
        assert result['waypoint_finish_sim_time'] == duration
    assert capture.call_args.args[2] == 'handoff'


@pytest.mark.parametrize('ending', ['complete', 'returned'])
def test_stop_cancels_live_latency_without_enabling_handoff(latency_case, ending):
    with pytest.raises(benchmark.sp.Abort):
        latency_case(stop=True, ending=ending)


@pytest.mark.parametrize('kind', ['collision', 'transition', 'orientation', 'grasp_loss'])
@pytest.mark.parametrize('continuation', [2., 20.])
def test_end_handoff_runs_the_whole_tail_without_countdown_or_hold(latency_case, kind, continuation):
    result, holds, progress, capture, moves = latency_case(kind=kind, delay=0.,
        handoff_at_end=True, continuation=continuation)
    assert result['handoff_mode'] == 'end_of_plan'
    assert result['waypoint_plan_completed'] and result['detection_latency_completed']
    assert result['handoff_sim_time'] == pytest.approx(1.5+continuation)
    assert result['waypoint_finish_sim_time'] == 1.5
    assert len(moves) == round(continuation/.05)
    assert holds == []
    assert 'end of waypoint plan' in next(c.args[1] for c in progress.call_args_list
                                         if c.args[1].startswith('Detection latency'))


def test_end_handoff_keeps_stop_and_total_waypoint_time_limit(latency_case):
    with pytest.raises(benchmark.sp.Abort):
        latency_case(delay=0., handoff_at_end=True, stop=True)
    result, holds, *_ = latency_case(delay=0., handoff_at_end=True, continuation=20., limit=3.)
    assert not result['waypoint_plan_completed']
    assert not result['detection_latency_completed']
    assert 'time limit' in result['waypoint_result']
    assert result['handoff_sim_time'] == pytest.approx(3.)
    assert holds == []


@pytest.mark.parametrize('kind', ['transition', 'orientation'])
def test_deviation_rebases_all_following_targets_in_one_frame(kind):
    original = Rotation.from_euler('xyz', [40., 15., -20.], degrees=True).as_matrix()
    changed = Rotation.from_euler('z', 30., degrees=True).as_matrix() @ original
    p0 = np.array([.1, -.2, .8]); p1 = p0 + [.07, -.03, .002]
    targets = [(p0, original), (p0+[0., 0., .2], original),
               (p0+[.3, .1, .2], original @ Rotation.from_euler('x', 10., degrees=True).as_matrix())]
    chain = GMMChain('libero_goal', 1, 'place object', [
        Waypoint(name='Grasp', skill_role='grasp'),
        Waypoint(name='Transport', skill_role='transport'),
        Waypoint(name='Place', skill_role='place')])
    before = asdict(chain)
    tail, metadata = benchmark.propagated_tail(chain, targets, 0, (p0, original), (p1, changed), kind)
    tail.validate()
    assert asdict(chain) == before
    assert len(tail.waypoints) == 2
    error = changed @ original.T if kind == 'orientation' else np.eye(3)
    for w, (p, r), record in zip(tail.waypoints, targets[1:], metadata['targets']):
        np.testing.assert_allclose(w.offset, p1+error@(p-p0), atol=1e-12)
        np.testing.assert_allclose(w.orientation, error@r, atol=1e-12)
        np.testing.assert_allclose(record['position'], w.offset)
        assert not w.carry and not w.seat and not w.hold_xy and w.learned == {}
    if kind == 'orientation':
        np.testing.assert_allclose(changed.T@(np.array(tail.waypoints[1].offset)-p1),
                                   original.T@(targets[2][0]-p0), atol=1e-12)


@pytest.mark.parametrize('kind', ['transition', 'orientation'])
@pytest.mark.parametrize('completed_grasp', [True, False])
@pytest.mark.parametrize('observed', [True, False])
def test_sequence_continuation_unwinds_feedback_once_and_preserves_callbacks(
        monkeypatch, kind, completed_grasp, observed):
    from waypoint_recovery_live import grasp_intensities
    from waypoint_session import Session
    data = SimpleNamespace(time=0.)
    p0 = np.array([0., 0., .8]); p1 = p0+[.1, -.1, 0.]
    changed = Rotation.from_euler('y', 15., degrees=True).as_matrix()
    targets = [(p0, np.eye(3)), (p0+[0., 0., .2], np.eye(3)),
               (p0+[.2, .1, 0.], np.eye(3))]
    chain = GMMChain('libero_goal', 1, 'place object', [
        Waypoint(name='Grasp', skill_role='grasp'),
        Waypoint(name='Transport', skill_role='transport'),
        Waypoint(name='Release', skill_role='release')])
    env = Mock(); env.sim.data = data; env.check_success.return_value = False
    old_callback, old_tick = Mock(), Mock()
    session = SimpleNamespace(env=env, continue_on_failure=False, tick=old_tick,
        on_waypoint_finished=old_callback, preview=lambda _:targets,
        gripper_pose=lambda:(p1.copy(), changed.copy()))
    class Injector:
        def __init__(self, session, injections, clock, waypoints):
            self.clock = clock
            self.nominal_fault_pose = targets[0]
            self.fired, self.missed = [], []
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def start_row(self, row):
            if row == 0 and observed:
                self.clock.mark(data.time, kind=kind)
        def observe_control(self):
            pass
    def nominal(plan, report):
        report(0, 'Running')
        data.time += 2.
        session.tick()
        report(0, 'failed grasp')
        if completed_grasp:
            session.on_waypoint_finished(0, 'failed grasp')
        report(1, 'Running')
        assert not observed, 'Adaptive feedback ran past the failed waypoint'
        data.time += 3.
        report(2, 'ok')
        return 'Finished'
    def tail(s, plan, report):
        assert s is session and s.env is env
        assert data.time == 2.
        assert len(plan.waypoints) == 2
        assert s.continue_on_failure
        for i in range(len(plan.waypoints)):
            report(i, 'Running')
            data.time += 1.
            s.tick()
            report(i, 'ok')
            s.on_waypoint_finished(i, 'ok')
        return 'Finished tail'
    session.run = Mock(side_effect=nominal)
    tail_run = Mock(side_effect=tail)
    monkeypatch.setattr(Session, 'run', tail_run)
    @contextmanager
    def context(*args):
        yield
    monkeypatch.setattr(benchmark, 'observe_physics', context)
    monkeypatch.setattr(benchmark, 'suspend_images', context)
    result = benchmark.run_fault(session, chain, dict(kind=kind, intensity='low', injection_row=0),
        0., 180., None, injector_class=Injector, intensity_rule=grasp_intensities, capture=Mock(),
        handoff_after_waypoint=True, handoff_at_end=True, propagate_deviation=True)
    assert tail_run.call_count == int(observed)
    session.run.assert_called_once()
    assert bool(result['propagated_deviation']) == observed
    assert result['waypoint_completed'] == completed_grasp
    assert result['waypoint_plan_completed']
    assert result['trace'][-1]['row'] == 2 and result['trace'][-1]['result'] == 'ok'
    assert session.on_waypoint_finished is old_callback
    assert session.tick is old_tick and not session.continue_on_failure
    assert not hasattr(session, 'stop_after_waypoint')
    env.set_init_state.assert_not_called()
    env.reset.assert_not_called()


def test_live_latency_has_its_own_budget_after_waypoint_time_limit(latency_case):
    result, holds, progress, capture, moves = latency_case(limit=3., duration=4.)
    assert 'time limit' in result['waypoint_result']
    assert result['detection_latency_completed']
    assert result['detection_latency_start_sim_time'] == 4.
    assert result['actual_detection_latency_seconds'] == pytest.approx(10.)
    assert result['handoff_sim_time'] == pytest.approx(14.)
    assert result['waypoint_execution_stopped'] and not result['waypoint_completed']


@pytest.mark.parametrize('observed', [True, False])
@pytest.mark.parametrize('row', [0, 2, 5, 8])
@pytest.mark.parametrize('kind', ['collision', 'transition', 'orientation', 'grasp_loss'])
def test_waypoint_handoff_finishes_faulted_row_even_after_onset(monkeypatch, observed, row, kind):
    from waypoint_recovery_live import grasp_intensities
    data = SimpleNamespace(time=0.)
    env = Mock()
    env.sim.data = data
    env.check_success.return_value = False
    previous_tick = Mock()
    session = SimpleNamespace(env=env, tick=previous_tick, continue_on_failure=False)
    finished = []
    class FakeInjector:
        def __init__(self, session, injections, clock, waypoints):
            self.clock = clock
            self.injections = injections
            self.fired, self.missed = [], []
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def start_row(self, row):
            if observed and row in self.injections:
                self.clock.mark(data.time, kind=kind)
        def observe_control(self):
            pass
    @contextmanager
    def context(*args):
        yield
    def run(chain, report):
        for previous_row in range(row):
            report(previous_row, 'Running')
            data.time += .1
            session.tick()
            report(previous_row, 'ok')
        report(row, 'Running')
        data.time += .5
        session.tick()  # A zero-delay onset must not interrupt this row.
        report(row, 'intermediate motion status')
        finished.append('moved to wrong waypoint')
        data.time += 1.
        session.tick()
        finished.append('closed and checked grasp')
        report(row, 'failed: nothing between the fingers')
        session.on_waypoint_finished(row, 'failed: nothing between the fingers')
        pytest.fail('Completion did not stop playback before the next waypoint')
    session.run = run
    monkeypatch.setattr(benchmark, 'observe_physics', context)
    monkeypatch.setattr(benchmark, 'suspend_images', context)
    chain = SimpleNamespace(waypoints=[Waypoint(name='Move', skill_role='move') for _ in range(row)]
        + [Waypoint(name='Failure', skill_role='transport' if kind == 'grasp_loss' else 'grasp'),
           Waypoint(name='Transport', skill_role='transport')])
    result = benchmark.run_fault(session, chain,
        dict(kind=kind, intensity='high', injection_row=row), 0., 180., None,
        injector_class=FakeInjector, intensity_rule=grasp_intensities,
        capture=Mock(), handoff_after_waypoint=True)
    assert finished == ['moved to wrong waypoint', 'closed and checked grasp']
    assert result['waypoint_finish_sim_time'] == pytest.approx(row*.1+1.5)
    assert result['waypoint_result'] == 'failed: nothing between the fingers'
    assert result['waypoint_interrupted']
    assert result['waypoint_completed']
    assert max(step['row'] for step in result['trace']) == row
    assert result['observed_failure'] == observed
    if observed:
        assert result['actual_delay_seconds'] == pytest.approx(1.5)
        assert result['handoff_sim_time'] == result['waypoint_finish_sim_time']
    assert session.tick is previous_tick
    assert not session.continue_on_failure
    assert not hasattr(session, 'stop_after_waypoint')
    assert not hasattr(session, 'on_waypoint_finished')
    env.reset.assert_not_called()
    env.set_init_state.assert_not_called()


@pytest.mark.parametrize('ending', ['return', 'next_row'])
def test_aborted_grasp_is_not_completion_even_with_a_measured_failure(monkeypatch, ending):
    data = SimpleNamespace(time=0.)
    session = SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data),
        check_success=lambda: False), tick=lambda: None, continue_on_failure=False)
    previous_finished = Mock()
    session.on_waypoint_finished = previous_finished
    session.stop_after_waypoint = 99
    class Injector:
        def __init__(self, session, injections, clock, waypoints):
            self.clock = clock
            self.fired, self.missed = [], []
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def start_row(self, row):
            self.clock.mark(data.time, kind='transition')
        def observe_control(self):
            pass
    def run(chain, report):
        report(0, 'Running')
        data.time += .5
        session.tick()
        report(0, 'failed: path attempt ended before close/check')
        if ending == 'next_row':
            report(1, 'Running')
            pytest.fail('Entered the next waypoint after an unfinished Grasp')
        return 'Stopped: Grasp did not finish'
    session.run = run
    monkeypatch.setattr(benchmark, 'observe_physics', lambda *_: __import__('contextlib').nullcontext())
    monkeypatch.setattr(benchmark, 'suspend_images', lambda *_: __import__('contextlib').nullcontext())
    chain = SimpleNamespace(waypoints=[Waypoint(name='Grasp', skill_role='grasp'),
                                      Waypoint(name='Transport', skill_role='transport')])
    result = benchmark.run_fault(session, chain,
        dict(kind='transition', intensity='high', injection_row=0), 0., 180., None,
        injector_class=Injector, capture=Mock(), handoff_after_waypoint=True)
    assert result['observed_failure']
    assert not result['waypoint_completed']
    assert result['waypoint_execution_stopped']
    assert result['handoff_sim_time'] == result['waypoint_end_sim_time']
    assert 'waypoint_finish_sim_time' not in result
    assert session.on_waypoint_finished is previous_finished
    assert session.stop_after_waypoint == 99
    previous_finished.assert_not_called()


@pytest.mark.parametrize('row', [0, 1, 2, 5, 8])
def test_failed_gmm_skill_returns_after_entire_grasp_without_replay(monkeypatch, row):
    from waypoint_gmm_editor import GMMSession
    from waypoint_session import Session
    session = GMMSession.__new__(GMMSession)
    env = Mock()
    env.env._observables = {}
    env.check_success.return_value = False
    session.env = env
    session.continue_on_failure = True
    session.stop_after_waypoint = row
    session.save_replay_archive = False
    session.preview = lambda chain: [(np.zeros(3), np.eye(3))]*len(chain.waypoints)
    chain = SimpleNamespace(execution_skill={'test': True}, validate=Mock(), waypoints=[
        Waypoint(name='Move') for _ in range(row)] + [Waypoint(name='Grasp'), Waypoint(name='Transport')])
    steps = []
    def run_skill(session, skill, progress, mode):
        progress(row, 'Running')
        progress(row, 'intermediate failure notification')
        steps.append('finished the motion and close/check')
        return dict(completed=False, result='failed grasp', task_success=False)
    monkeypatch.setattr(__import__('waypoint_skill'), 'run_skill', run_skill)
    monkeypatch.setattr(Session, 'run', Mock(side_effect=AssertionError('replayed remaining waypoints')))
    result = session._run_skill_chain(chain, lambda *args: None)
    assert result == 'Stopped: failed grasp'
    assert steps == ['finished the motion and close/check']
    Session.run.assert_not_called()


@pytest.mark.parametrize('prefix_rows,selected_row', [(1, 2), (3, 5), (7, 9)])
def test_fallback_playback_completion_keeps_original_row_numbers(monkeypatch, prefix_rows, selected_row):
    from waypoint_gmm_editor import GMMSession
    from waypoint_session import Session
    session = GMMSession.__new__(GMMSession)
    session.env = SimpleNamespace(env=SimpleNamespace(_observables={}), check_success=lambda: False)
    session.continue_on_failure = True
    session.stop_after_waypoint = selected_row
    session.save_replay_archive = False
    session.preview = lambda chain: [(np.zeros(3), np.eye(3))]*len(chain.waypoints)
    waypoints = [Waypoint(name='Move') for _ in range(selected_row+2)]
    chain = GMMChain('libero_spatial', 0, 'test', waypoints,
        execution_skill={'waypoint_template': [asdict(w) for w in waypoints]})
    session.on_waypoint_finished = Mock()
    original_finished = session.on_waypoint_finished
    def skill(session, data, progress, mode, waypoint_finished):
        progress(prefix_rows-1, 'Running')
        return dict(completed=False, result='prefix failed', task_success=False)
    def tail(session, plan, report):
        local_row = selected_row-prefix_rows
        report(local_row, 'Running')
        report(local_row, 'failed: nothing between the fingers')
        session.on_waypoint_finished(local_row, 'failed: nothing between the fingers')
        session.last_run_failures = []
        return 'Finished'
    monkeypatch.setattr(__import__('waypoint_skill'), 'run_skill', skill)
    monkeypatch.setattr(Session, 'run', tail)
    session._run_skill_chain(chain, lambda *_: None)
    original_finished.assert_called_once_with(selected_row, 'failed: nothing between the fingers')
    assert session.on_waypoint_finished is original_finished
