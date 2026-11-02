"""Approximate coverage must reflect actual poses, timing and order only."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest

from waypoint_chain import Chain, Waypoint, WORLD
from waypoint_coverage import PoseCoverage, RolloutCoverage, frozen_targets


IDENTITY = np.eye(3)
HALF_TURN = np.diag([-1., -1., 1.])


def coverage(**kwargs):
    return PoseCoverage(['first', 'second', 'third'],
                        [(np.array([x, 0, 0]), IDENTITY) for x in (0., .2, .4)], **kwargs)


def test_approximate_position_hits_and_plan_order_are_reported_separately():
    monitor = coverage()
    monitor.update([.2, 0, 0], IDENTITY, 0., 0)
    assert monitor.at_start == [False, True, False]
    assert monitor.first_visit == [None, None, None]
    monitor.update([.23, 0, 0], HALF_TURN, .05, 1)
    assert monitor.first_visit == [None, .05, None]
    assert monitor.prefix == 0
    monitor.update([.04, 0, 0], HALF_TURN, .1, 2)
    assert monitor.prefix == 1
    # Seeing the second target before the first is not enough for an ordered match.
    assert monitor.ordered_visit[1] is None
    monitor.update([.19, 0, 0], IDENTITY, .15, 3)
    assert monitor.prefix == 2
    assert monitor.ordered_visit == [.1, .15, None]
    rows = monitor.rows(finished=True)
    assert rows[0]['closest_position_cm'] == pytest.approx(4.)
    assert rows[2]['status'] == 'Not visited'


def test_rotation_is_optional_and_joint_pose_bounds_must_hold_at_same_time():
    relaxed = coverage()
    relaxed.update([0, 0, 0], HALF_TURN, .05, 1)
    assert relaxed.first_visit[0] == .05
    strict = coverage(orientation_degrees=30)
    strict.update([0, 0, 0], HALF_TURN, .05, 1)
    strict.update([.1, 0, 0], IDENTITY, .1, 2)
    assert strict.first_visit[0] is None
    strict.update([.04, 0, 0], IDENTITY, .15, 3)
    assert strict.first_visit[0] == .15
    # Show the angle at the closest position, not an unrelated best angle.
    assert strict.rows()[0]['angle_at_closest_deg'] == pytest.approx(180.)


def test_initial_pose_alone_and_a_stopped_rollout_do_not_create_hits():
    monitor = coverage()
    monitor.update([0, 0, 0], IDENTITY, 0., 0)
    assert monitor.prefix == 0
    assert monitor.rows()[0]['status'] == 'At start only'
    assert monitor.rows()[1]['status'] == 'Pending'
    monitor.update([.05, 0, 0], IDENTITY, .05, 1)
    assert monitor.first_visit[0] == .05  # inclusive five-centimetre boundary


@pytest.mark.parametrize('kwargs', [dict(position_tolerance=0), dict(position_tolerance=float('nan')),
                                   dict(orientation_degrees=0), dict(orientation_degrees=181)])
def test_invalid_tolerances_are_rejected(kwargs):
    with pytest.raises(ValueError):
        coverage(**kwargs)


class Env:
    def __init__(self):
        self.sim = SimpleNamespace(data=SimpleNamespace(time=3.))
        self.position = np.array([0., 0, 0])
        self.actions = []
        self.output = ({'same': 'observation'}, 0., False, {})

    def step(self, action):
        self.actions.append(action)
        self.position = np.asarray(action[:3]).copy()
        self.sim.data.time += .05
        return self.output


def session_and_plan():
    env = Env()
    session = SimpleNamespace(env=env, reference_poses={'object': ([9., 0, 0], IDENTITY)},
                              start_pose=('original', 'reference'), previewed='original cache')
    session.gripper_pose = lambda: (env.position.copy(), IDENTITY.copy())
    def refresh():
        session.reference_poses = {'object': (np.array([.2, 0, 0]), IDENTITY)}
        session.start_pose = session.gripper_pose()
        session.previewed = None
    def preview(chain):
        return [(np.asarray(session.reference_poses['object'][0])+w.offset, w.orientation)
                for w in chain.waypoints]
    session.refresh_references, session.preview = refresh, preview
    plan = Chain('libero_goal', 1, 'test',
                 [Waypoint('first', 'object', [0., 0, 0], IDENTITY.tolist()),
                  Waypoint('second', 'object', [.2, 0, 0], IDENTITY.tolist())])
    return session, plan


def test_target_resolution_uses_current_scene_and_restores_editor_references():
    session, plan = session_and_plan()
    poses, start = session.reference_poses, session.start_pose
    targets = frozen_targets(session, plan)
    np.testing.assert_allclose(targets[0][0], [.2, 0, 0])
    assert session.reference_poses is poses and session.start_pose is start
    assert session.previewed == 'original cache'
    assert not session.env.actions and session.env.sim.data.time == 3.


def test_every_real_step_is_observed_without_changing_actions_or_output(tmp_path):
    session, plan = session_and_plan()
    actions = [np.array([x, 0, 0, 0, 0, 0, -1.]) for x in (.22, .41)]
    monitor = RolloutCoverage(session, plan, 'pi05')
    with monitor:
        # Moving object references later must not move the frozen checkpoints.
        session.reference_poses['object'] = ([50., 0, 0], IDENTITY)
        for action in actions:
            assert session.env.step(action) is session.env.output
        monitor.task_success = True
    assert 'step' not in session.env.__dict__
    assert all(a is b for a, b in zip(actions, session.env.actions))
    assert len(session.env.actions) == 2
    report = monitor.report()
    assert report['visited'] == report['ordered_prefix'] == report['steps'] == 2
    assert report['rows'][0]['first_visit_sim_seconds'] == pytest.approx(.05)
    assert report['rows'][1]['ordered_visit_sim_seconds'] == pytest.approx(.1)
    assert report['task_success'] and report['run_state'] == 'finished'
    assert 'does not verify grasp' in report['measurement']
    path = tmp_path/'report.csv'
    monitor.save_csv(path)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2 and rows[0]['backend'] == 'pi05'
    assert float(rows[0]['closest_position_cm']) == pytest.approx(2.)
    assert float(rows[0]['position_tolerance_cm']) == 5.


def test_stop_and_error_restore_an_existing_step_wrapper():
    class Abort(Exception):
        pass
    for error in (Abort(), RuntimeError('policy failed')):
        session, plan = session_and_plan()
        original = session.env.step
        def wrapped(action):
            return original(action)
        session.env.step = wrapped
        monitor = RolloutCoverage(session, plan, 'pi05')
        with pytest.raises(type(error)):
            with monitor:
                session.env.step([.2, 0, 0])
                raise error
        assert session.env.step is wrapped
        assert monitor.report()['visited'] == 1
        assert monitor.report()['rows'][1]['status'] == 'Pending'
        assert monitor.state == ('stopped' if isinstance(error, Abort) else 'error')
