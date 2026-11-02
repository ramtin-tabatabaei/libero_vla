"""Failure timing distinguishes measured onset, confirmation and interruption."""
from types import SimpleNamespace

import numpy as np
import pytest

from waypoint_chain import Chain, Waypoint
from waypoint_task_events import (FailureConfig, FailureTimeline, SimulationFailureObserver,
                                  distance_to_route, task_intent)


def timeline(**kwargs):
    return FailureTimeline(FailureConfig(**kwargs), {'correct'}, 'test goal', [[0, 0, 0], [1, 0, 0]])


def test_wrong_pickup_needs_bilateral_grasp_lift_and_continuous_confirmation():
    monitor = timeline()
    monitor.update(.05, 1, [0, 0, 0], [], {'wrong': .2})  # bumped object, no grasp
    monitor.update(.1, 2, [0, 0, 0], ['wrong'], {'wrong': .01})  # grasp but not lifted
    monitor.update(.15, 3, [0, 0, 0], ['wrong'], {'wrong': .03})
    monitor.update(.2, 4, [0, 0, 0], [], {})  # release breaks the confirmation window
    monitor.update(.3, 5, [0, 0, 0], ['wrong'], {'wrong': .04})
    monitor.update(.4, 6, [0, 0, 0], ['wrong'], {'wrong': .04})
    assert not monitor.events
    monitor.update(.45, 7, [0, 0, 0], ['wrong'], {'wrong': .05})
    event = monitor.events['wrong_object']
    assert event['onset_sim_seconds'] == pytest.approx(.3)
    assert event['confirmed_sim_seconds'] == pytest.approx(.45)
    assert event['evidence']['object'] == 'wrong'
    monitor.update(.9, 9, [0, 0, 0], ['another'], {'another': .2})
    monitor.update(1.1, 10, [0, 0, 0], ['another'], {'another': .2})
    assert len(monitor.events) == 1
    assert len(monitor.history) == 2  # another pickup episode is retained


def test_correct_pickup_and_unknown_expected_identity_are_not_called_wrong():
    for expected in ({'correct'}, None):
        monitor = FailureTimeline(FailureConfig(), expected, 'test', [])
        for i in range(1, 10):
            monitor.update(.05*i, i, [0, 0, 0], ['correct'], {'correct': .1})
        assert not monitor.events
    assert not monitor.report()['off_plan_available']


def test_route_includes_segments_and_off_plan_requires_continuous_time():
    assert distance_to_route([.5, 0, 0], [[0, 0, 0], [1, 0, 0]]) == 0
    assert distance_to_route([.5, .4, 0], [[0, 0, 0], [1, 0, 0]]) == pytest.approx(.4)
    assert distance_to_route([0, 0, 0], []) is None
    monitor = timeline(off_plan_seconds=.2)
    monitor.update(.05, 1, [.5, 0, 0], [], {})
    monitor.update(.1, 2, [.5, .4, 0], [], {})
    monitor.update(.2, 3, [.5, 0, 0], [], {})
    monitor.update(.3, 4, [.5, .4, 0], [], {})
    monitor.update(.45, 5, [.5, .4, 0], [], {})
    assert not monitor.events
    monitor.update(.5, 6, [.5, .4, 0], [], {})
    assert monitor.events['off_plan']['onset_sim_seconds'] == pytest.approx(.3)
    assert monitor.events['off_plan']['confirmed_sim_seconds'] == pytest.approx(.5)


def test_collision_threshold_initial_contact_and_wall_time():
    monitor = FailureTimeline(FailureConfig(), {'correct'}, 'test', [], lambda: 4.5)
    monitor.collision(.002, 1, 9.9, 'wrist', 'table')
    assert not monitor.events
    monitor.collision(.004, 1, 12., 'wrist', 'table')
    event = monitor.events['collision']
    assert event['onset_sim_seconds'] == .004
    assert event['confirmed_wall_seconds'] == 4.5
    monitor.collision(.006, 1, 20., 'wrist', 'table')
    assert event['evidence']['force_n'] == 12.
    initial = timeline()
    initial.collision(0., 0, 12., 'wrist', 'table')
    assert initial.events['collision']['onset_sim_seconds'] is None
    assert initial.events['collision']['evidence']['already_present_at_start']


def test_repeated_collisions_and_off_plan_excursions_are_grouped_into_episodes():
    monitor = timeline(off_plan_seconds=.1)
    monitor.collision(.002, 1, 15., 'wrist', 'table')
    monitor.collision(.004, 1, 16., 'wrist', 'table')
    monitor.collision(.006, 1, 0., '', '')
    monitor.collision(.02, 1, 15., 'wrist', 'table')  # a short gap is the same impact
    assert len(monitor.history) == 1
    monitor.collision(.12, 3, 0., '', '')
    monitor.collision(.2, 4, 20., 'wrist', 'cabinet')
    assert len(monitor.history) == 2
    assert monitor.history[1]['onset_sim_seconds'] == .2
    monitor.update(.25, 5, [.5, .4, 0], [], {})
    monitor.update(.35, 7, [.5, .4, 0], [], {})
    monitor.update(.4, 8, [.5, .5, 0], [], {})
    monitor.update(.45, 9, [.5, 0, 0], [], {})
    monitor.update(.5, 10, [.5, .4, 0], [], {})
    monitor.update(.6, 12, [.5, .4, 0], [], {})
    events = monitor.report()['events']
    assert len([event for event in events if event['kind'] == 'off_plan']) == 2


@pytest.mark.parametrize('state,success,reason,unfinished', [
    ('finished', False, 'step_limit', True), ('finished', True, 'task_success', False),
    ('stopped', False, 'step_limit', False), ('error', False, 'step_limit', False),
    ('finished', None, None, False)])
def test_unfinished_only_at_an_exhausted_policy_budget(state, success, reason, unfinished):
    monitor = timeline()
    monitor.finish(5., 100, state, success, reason)
    assert ('unfinished' in monitor.events) is unfinished
    if unfinished:
        assert monitor.events['unfinished']['onset_sim_seconds'] == 5.
    elif state in ('stopped', 'error'):
        assert 'unassessed' in monitor.assessment


def test_goal_inference_does_not_accept_destination_as_pickup_and_handles_multiple_objects():
    plan = Chain('libero_goal', 0, 'place things', [])
    expected, contacts, source = task_intent(
        [['and', ['On', 'correct', 'plate'], ['In', 'another', 'drawer']]], plan,
        {'correct', 'another', 'plate'})
    assert expected == {'correct', 'another'} and 'plate' in contacts
    expected, _, _ = task_intent([['Open', 'drawer']], plan, {'bowl'})
    assert expected == set()
    expected, _, _ = task_intent([['Unrecognized', 'region']], plan, {'bowl'})
    assert expected is None
    plan.waypoints = [Waypoint('grasp', grasp='bowl')]
    assert task_intent([], plan, {'bowl'})[0] == {'bowl'}
    assert task_intent([['Open', 'drawer']], plan, {'bowl'})[0] == set()
    assert task_intent([], plan, {'bowl'}, 'none')[0] == set()
    assert task_intent([], plan, {'bowl'}, 'bowl')[0] == {'bowl'}
    with pytest.raises(ValueError, match='not present'):
        task_intent([], plan, {'bowl'}, 'missing')


@pytest.mark.parametrize('kwargs', [dict(collision_force_n=0), dict(off_plan_metres=-1),
    dict(off_plan_seconds=float('nan')), dict(pickup_lift_metres=0), dict(pickup_hold_seconds=-.1)])
def test_invalid_thresholds_rejected(kwargs):
    with pytest.raises(ValueError):
        FailureConfig(**kwargs).validate()


def native_contact_scene(permitted=False):
    """A real MuJoCo impact that appears and disappears within one control step."""
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.002" gravity="0 0 0"/>
      <worldbody><body name="robot_link"><geom name="finger" type="box" size=".05 .05 .05"/></body>
      <body name="obstacle" pos="1 0 0"><freejoint/><geom name="obstacle_geom" type="sphere" size=".02" mass="1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    class ModelNames:
        def geom_id2name(self, index):
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(index))
        def body_id2name(self, index):
            return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(index))
    class Sim:
        def __init__(self):
            self.model, self.data = ModelNames(), data
        def step(self):
            mujoco.mj_step(model, data)
    sim = Sim()
    fingers = ['finger'] if permitted else []
    robot = SimpleNamespace(robot_model=SimpleNamespace(bodies=['robot_link']),
        gripper=SimpleNamespace(bodies=[], important_geoms=dict(left_fingerpad=fingers, right_fingerpad=[])))
    domain = SimpleNamespace(robots=[robot], objects_dict={'obstacle': SimpleNamespace(contact_geoms=['obstacle_geom'])},
        fixtures_dict={}, object_sites_dict={}, _check_grasp=lambda *args: False,
        parsed_problem={'goal_state': [['On', 'obstacle', 'destination']] if permitted else [['Open', 'fixture']]})
    world = SimpleNamespace(m=model, d=data, pos=lambda _: data.xpos[2].copy())
    session = SimpleNamespace(env=SimpleNamespace(env=domain, sim=sim), world=world,
                              gripper_pose=lambda: (np.zeros(3), np.eye(3)), reference_poses={})
    return session, sim, model, data


@pytest.mark.parametrize('permitted', [False, True])
def test_physics_step_contact_is_captured_even_if_gone_at_control_boundary(permitted):
    import mujoco
    session, sim, model, data = native_contact_scene(permitted)
    observer = SimulationFailureObserver(session, Chain('test', 0, '', []), [], 0.,
                                         FailureConfig(), lambda: 1)
    original = sim.step
    observer.install()
    try:
        data.qpos[0] = .04  # penetrate the robot box at the first physics step
        sim.step()
        data.qpos[0] = 1.
        data.qvel[:] = 0.
        mujoco.mj_forward(model, data)
        for _ in range(24):
            sim.step()
        assert data.ncon == 0  # a control-rate-only detector would miss the impact
    finally:
        observer.close()
    assert sim.step == original and 'step' not in sim.__dict__
    assert ('collision' in observer.timeline.events) is not permitted
    if not permitted:
        event = observer.timeline.events['collision']
        assert event['onset_sim_seconds'] == pytest.approx(.002)
        assert event['evidence']['force_n'] >= 10.


def test_combined_monitor_restores_physics_hook_on_failed_start_and_exports(tmp_path):
    import csv
    import json
    from waypoint_coverage import RolloutCoverage
    session, sim, _, _ = native_contact_scene()
    def env_step(action):
        for _ in range(25):
            sim.step()
        return ({}, 0., False, {})
    session.env.step = env_step
    plan = Chain('test', 0, '', [])
    original = sim.step
    def fail_once(monitor):
        if monitor.state == 'running':
            raise RuntimeError('report failed')
    with pytest.raises(RuntimeError, match='report failed'):
        with RolloutCoverage(session, plan, 'pi05', failure_config=FailureConfig(),
                             check_waypoints=False, on_update=fail_once):
            pass
    assert sim.step == original and session.env.step is env_step
    monitor = RolloutCoverage(session, plan, 'pi05', failure_config=FailureConfig(), check_waypoints=False)
    with monitor:
        session.env.step(np.zeros(7))
        monitor.set_policy_result({'success': False, 'steps': 1})
    assert sim.step == original and session.env.step is env_step
    report = monitor.report()
    event = report['failure_timeline']['events'][0]
    assert event['kind'] == 'unfinished'
    assert event['onset_sim_seconds'] == pytest.approx(.05)
    assert event['confirmed_wall_seconds'] >= 0
    monitor.save_failures_csv(tmp_path/'events.csv')
    saved = list(csv.DictReader((tmp_path/'events.csv').open()))
    assert saved[0]['kind'] == 'unfinished'
    assert float(saved[0]['confirmed_sim_seconds']) == pytest.approx(.05)
    monitor.save_json(tmp_path/'report.json')
    assert json.loads((tmp_path/'report.json').read_text())['failure_timeline']['events'] == [event]
