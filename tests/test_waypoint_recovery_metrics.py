"""Timing and directness must describe physical behavior, not eventual success."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from waypoint_recovery_metrics import TrajectoryRecorder, timing_comparisons, trajectory_metrics
from waypoint_recovery_benchmark import timed_scenarios, trial_id, feasible_row
from waypoint_gmm import GMMWaypoint


TARGETS = dict(source='soup', destination='basket_site', destination_body='basket')


def sample(t, x, contacts=(), grasped=(), success=False, target=1.):
    return dict(time=t, wall_seconds=t*4, eef=[x, 0., 0.], focus_position=[target, 0., 0.],
                focus_distance=abs(target-x), source_distance=abs(target-x),
                contacts=list(contacts), grasped=list(grasped), success=success)


def test_reaching_the_right_object_and_finishing_are_separate_times():
    rows = [sample(i*.05, i*.01) for i in range(21)]
    rows[10]['contacts'] = ['soup']
    rows[15]['grasped'] = ['soup']
    rows[20]['success'] = True
    m = trajectory_metrics(rows, TARGETS, prompt_window=1.)
    assert m['first_goalward_motion_seconds'] == 0.
    assert m['goalward_response_confirmed_seconds'] == .25
    assert m['first_correct_contact_seconds'] == .5
    assert m['first_correct_grasp_seconds'] == .75
    assert m['completion_sim_seconds'] == 1.
    assert m['correct_grasp_to_completion_seconds'] == .25
    assert m['path_after_correct_grasp_metres'] == pytest.approx(.05)
    assert m['completion_wall_seconds'] == 4.
    assert m['first_correct_grasp_wall_seconds'] == 3.
    assert m['first_observed_goalward_motion_wall_seconds'] == .2
    assert m['prompt_correct_object_recovery']
    assert m['path_to_grasp_detour_ratio'] == pytest.approx(1.)


def test_eventual_success_does_not_make_a_wrong_object_response_direct():
    rows = [sample(i*.05, i*.01) for i in range(21)]
    rows[3]['contacts'] = ['milk']
    rows[4]['grasped'] = ['milk']
    rows[15]['grasped'] = ['soup']
    rows[20]['success'] = True
    m = trajectory_metrics(rows, TARGETS, prompt_window=1.)
    assert m['success'] and m['prompt_goalward_response']
    assert not m['prompt_correct_object_recovery']
    assert m['wrong_object_contacts_before_correct_grasp'] == ['milk']
    assert m['wrong_objects_grasped'] == ['milk']


def test_motionless_robot_cannot_get_credit_for_a_moving_object():
    rows = [sample(i*.05, 0., target=1.-i*.04) for i in range(21)]
    m = trajectory_metrics(rows, TARGETS, prompt_window=1.)
    assert m['first_goalward_motion_seconds'] is None
    assert not m['prompt_goalward_response']
    assert m['path_metres'] == 0.


def test_slow_sustained_approach_is_measured_without_a_speed_cutoff():
    rows = [sample(i*.05, i*.00025) for i in range(21)]
    m = trajectory_metrics(rows, TARGETS, prompt_window=None)
    assert m['first_goalward_motion_seconds'] == 0.
    assert m['goalward_response_confirmed_seconds'] == .25
    assert m['first_centimetre_goalward_seconds'] is None


def test_late_approach_is_not_prompt_even_if_task_eventually_finishes():
    rows = [sample(i*.05, 0.) for i in range(21)]
    rows.extend(sample(1.+i*.05, i*.01, success=i==20) for i in range(1,21))
    m = trajectory_metrics(rows, TARGETS, prompt_window=1.)
    assert m['success']
    assert m['first_goalward_motion_seconds'] >= 1.
    assert not m['prompt_correct_object_recovery']


def test_destination_contact_is_allowed_and_already_held_source_needs_no_regrasp():
    rows = [sample(i*.05, i*.01, contacts=['soup', 'basket'], grasped=['soup']) for i in range(11)]
    m = trajectory_metrics(rows, TARGETS, focus='destination', prompt_window=1.)
    assert m['source_already_grasped']
    assert m['first_correct_grasp_seconds'] == 0.
    assert m['wrong_objects_contacted'] == []
    assert m['prompt_correct_object_recovery']


def test_grasping_the_destination_is_a_wrong_object_grasp():
    rows = [sample(0.,0.), sample(.1,.01,contacts=['basket'],grasped=['basket']),
            sample(1.,.2,contacts=['soup'],grasped=['soup'],success=True)]
    m = trajectory_metrics(rows, TARGETS, prompt_window=None)
    assert m['first_object_contact_names'] == ['basket']
    assert m['first_object_contact_seconds'] == .1
    assert m['wrong_objects_grasped'] == ['basket']
    assert m['wrong_object_contacts_before_correct_grasp'] == ['basket']


def test_carrying_motion_is_not_misreported_as_approach_to_the_source():
    rows = [sample(0.,0.),sample(.05,0.,grasped=['soup'])]
    rows.extend(sample(i*.05,i*.01,grasped=['soup'],success=i==20) for i in range(2,21))
    m = trajectory_metrics(rows, TARGETS, prompt_window=None)
    assert m['first_correct_grasp_seconds'] == .05
    assert m['first_goalward_motion_seconds'] is None


def test_cutoff_can_be_disabled_while_raw_timing_is_preserved():
    rows = [sample(i*.05, i*.01) for i in range(11)]
    m = trajectory_metrics(rows, TARGETS)
    assert m['first_goalward_motion_seconds'] == 0.
    assert m['prompt_correct_object_recovery'] is None


def test_transient_goal_contact_does_not_count_as_completed():
    rows = [sample(0., 0.), sample(.1, .1, success=True), sample(.2, .2)]
    m = trajectory_metrics(rows, TARGETS)
    assert m['first_goal_success_seconds'] == .1
    assert m['completion_sim_seconds'] is None
    assert not m['success']


def test_comparison_distinguishes_total_and_after_takeover_completion():
    base = dict(suite='libero_object', task_id=0, scene='shuffled', success=True)
    results = [dict(base, trial_id='normal', mode='normal_waypoints', status='normal_waypoints_success',
                    total_task_sim_seconds=12., motion_metrics=dict(completion_sim_seconds=12.)),
               dict(base, trial_id='recovery', mode='recovery', status='recovered', kind='collision',
                    intensity='high', total_task_sim_seconds=16., failure_to_completion_sim_seconds=8.,
                    actual_delay_seconds=2., motion_metrics=dict(completion_sim_seconds=6.))]
    row = timing_comparisons(results)[1]
    assert row['completion_sim_seconds'] == 6.
    assert row['failure_to_completion_sim_seconds'] == 8.
    assert row['total_task_sim_seconds'] == 16.
    assert row['extra_over_normal_waypoints_sim_seconds'] == 4.


def test_baselines_are_paired_with_all_160_recoveries():
    matrix = timed_scenarios()
    assert len(matrix) == 200
    assert len({trial_id(r) for r in matrix}) == 200
    assert sum(r['mode']=='recovery' for r in matrix) == 160
    assert sum(r['mode']=='normal_policy' for r in matrix) == 20
    assert sum(r['mode']=='normal_waypoints' for r in matrix) == 20
    for suite, task in {(r['suite'],r['task_id']) for r in matrix}:
        subset = [r for r in matrix if (r['suite'],r['task_id'])==(suite,task)]
        assert len({r['seed'] for r in subset}) == 1


def test_high_orientation_can_use_a_feasible_place_row_without_reducing_intensity():
    chain = SimpleNamespace(waypoints=[GMMWaypoint(name=n, skill_role=n.lower())
                                      for n in ('Move','Grasp','Transport','Place')])
    def ik(position, **kwargs):
        return np.zeros(7), .03 if abs(position[0]-1.) < .1 else 0.
    session = SimpleNamespace(preview=lambda chain: [(np.array([float(i),0.,0.]), np.eye(3)) for i in range(4)],
        executor=SimpleNamespace(scene=SimpleNamespace(ik=ik)), args=SimpleNamespace(line_step=.2))
    assert feasible_row(session, chain, dict(kind='orientation', intensity='high')) == 3


@pytest.mark.parametrize('video', [False, True])
@pytest.mark.parametrize('photo_every', [None, .05])
def test_recorder_preserves_commands_step_count_and_controller_state(tmp_path, video, photo_every):
    domain = SimpleNamespace(
        parsed_problem=dict(goal_state=[['in','soup','basket_site']]),
        objects_dict=dict(soup=SimpleNamespace(contact_geoms=['soup_geom'])),
        robots=[SimpleNamespace(gripper=SimpleNamespace(important_geoms=dict(
            left_fingerpad=['left'], right_fingerpad=['right'])))],
        check_contact=lambda *args: False, _check_grasp=lambda *args: False, control_timestep=.05)
    class Env:
        def __init__(self):
            self.env = domain
            self.sim = SimpleNamespace(data=SimpleNamespace(time=7.))
            self.commands = []
        def step(self, action):
            self.commands.append(np.asarray(action).copy())
            self.sim.data.time += .05
            return {}, 0., False, {}
        def check_success(self):
            return False
    env = Env()
    world = SimpleNamespace(domain=domain, parent_body=lambda name:'basket',
                            pos=lambda name:np.array([1.,0.,0.]))
    session = SimpleNamespace(env=env, world=world,
        image=Mock(return_value=np.zeros((12,16,3), dtype=np.uint8)),
        gripper_pose=lambda:(np.array([env.sim.data.time-7.,0.,0.]),np.eye(3)))
    action = np.arange(7, dtype=float)
    with TrajectoryRecorder(session, tmp_path, 'policy', video=video, video_gate=lambda:False,
                            photo_every=photo_every) as recorder:
        env.step(action)
        if photo_every is not None:
            assert len(list((tmp_path/'photos'/'policy').glob('*.png'))) == 2
        env.step(action)
    assert 'step' not in vars(env)
    assert len(env.commands) == 2 and len(recorder.samples) == 3
    assert np.array_equal(env.commands[0], action)
    assert recorder.samples[-1]['time'] == pytest.approx(.1)
    assert recorder.first_action_sim == 0.
    assert (tmp_path/'policy.trajectory.json').exists()
    assert not (tmp_path/'policy.mp4').exists()
    if photo_every is not None:
        import json
        records = [json.loads(line) for line in (tmp_path/'photos'/'policy'/'index.jsonl').read_text().splitlines()]
        assert [r['time'] for r in records] == pytest.approx([0., .05, .1])
        assert all((tmp_path/r['file']).exists() for r in records)
        assert recorder.photo_count == 3


def test_parallel_workers_keep_each_task_and_its_paired_layouts_together():
    from waypoint_recovery_parallel import task_batches
    batches = task_batches(timed_scenarios())
    assert len(batches) == 10
    assert all(len(batch) == 20 for batch in batches)
    assert len({trial_id(s) for batch in batches for s in batch}) == 200
    for batch in batches:
        assert len({(s['suite'], s['task_id']) for s in batch}) == 1
        assert {s['scene'] for s in batch} == {'original', 'shuffled'}
