"""Batch pairing, recovery attribution, compact records and resumable exports."""
import csv
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import waypoint_recovery_batch as batch
from waypoint_gmm import GMMChain, GMMWaypoint
from waypoint_recovery_live import LiveOptions


def inventory(tmp_path):
    root = tmp_path / 'plans'
    GMMChain('libero_goal', 0, 'place object', [
        GMMWaypoint(name='Grasp', skill_role='grasp'),
        GMMWaypoint(name='Transport', skill_role='transport')]).save(root / 'libero_goal/task_000.json')
    # Backups must not create extra tasks or conditions.
    (root / 'libero_goal/task_000.backup.json').write_text('{}')
    return batch.build_conditions(root, suites={'libero_goal': 2})[0]


def test_inventory_complete_paired_grid_and_missing_plans(tmp_path):
    conditions = inventory(tmp_path)
    assert len(conditions) == 64
    assert sum(not c['unavailable'] for c in conditions) == 32
    assert sum(c['unavailable'] == 'missing_plan' for c in conditions) == 32
    assert len({c['trial_id'] for c in conditions}) == 64
    for off, on in zip(conditions[::2], conditions[1::2]):
        assert off['latency_seconds'] == 0
        assert on['latency_seconds'] is None
        assert on['detection_latency'] == 'end_of_plan'
        assert off['detection_latency'] == 'off'
        assert 'latencyEnd' in on['trial_id']
        assert off['fault_seed'] == on['fault_seed']
        assert off['layout_seed'] == on['layout_seed']
        assert off['injection_row'] == on['injection_row']
        if off['kind'] == 'grasp_loss' and off['intensity'] == 'high' and off['task_id'] == 0:
            assert off['injection_row'] == 1
    assert len({c['fault_seed'] for c in conditions[:32]}) == 16


@pytest.mark.parametrize('extra,status,evaluable,recovered', [
    ({}, 'recovered', True, True),
    ({'observed_failure': False}, 'failure_not_observed', False, False),
    ({'success_before_policy': True}, 'already_successful_at_handoff', False, False),
    ({'policy_result': {'success': True, 'steps': 0}}, 'policy_not_evaluated', False, False),
    ({'policy_result': {'success': False, 'steps': 400}}, 'recovery_failed', True, False),
    ({'error': 'lost connection'}, 'error', False, False),
    ({'scene_unavailable': True}, 'scene_unavailable', False, False),
])
def test_classification_does_not_inflate_recovery(extra, status, evaluable, recovered):
    result = dict(observed_failure=True, success_before_policy=False,
                  policy_result={'success': True, 'steps': 5})
    result.update(extra)
    assert batch.classify({}, result) == dict(status=status, recovery_evaluable=evaluable,
                                             vla_recovered=recovered)


def test_csv_rebuild_is_idempotent_and_excludes_missing_from_denominator(tmp_path):
    conditions = inventory(tmp_path)
    manifest = dict(conditions=conditions, runnable=32)
    for c in conditions:
        if c['unavailable']:
            result = {}
        elif c is conditions[0]:
            result = dict(observed_failure=True, success_before_policy=False, success=True,
                          policy_result={'success': True, 'steps': 2},
                          controller_timings={'waypoints_sim_seconds': 3., 'vla_sim_seconds': .1})
        else:
            continue
        batch.atomic_json(batch.result_path(tmp_path, c), dict(condition=c, result=result,
                                                              evaluation=batch.classify(c, result)))
    for _ in range(2):
        assert batch.publish_tables(tmp_path, manifest, 'running') == 33
    with (tmp_path / 'results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 33
    assert rows[0]['vla_sim_seconds'] == '0.1'
    progress = json.loads((tmp_path / 'progress.json').read_text())
    assert progress['runnable_recorded'] == 1
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert sum(g['evaluable'] for g in summary['groups']) == 1
    assert sum(g['recovered'] for g in summary['groups']) == 1
    assert summary['overall'] == dict(evaluated=1, recovered=1, recovery_percent=100.)
    with (tmp_path / 'recovery_percentages.csv').open() as stream:
        table = list(csv.DictReader(stream))
    assert table[0]['failure_type'] == 'collision'
    assert table[0]['original_low_off'] == '100.0% (1/1)'
    assert table[0]['original_low_delayed'] == 'not evaluated (0/0)'


def test_recorder_keeps_phase_and_restores_step_on_failure(tmp_path):
    class Env:
        def __init__(self):
            self.sim = SimpleNamespace(data=SimpleNamespace(time=0., qpos=np.zeros(3), qvel=np.zeros(3)))
            self.env = SimpleNamespace(control_timestep=.05)
        def step(self, action):
            self.sim.data.time += .05
            self.sim.data.qpos += 1
            return 'observation'
        def check_success(self):
            return self.sim.data.time > .09
    env = Env()
    session = SimpleNamespace(env=env, gripper_pose=lambda: (np.zeros(3), np.eye(3)))
    recorder = batch.CompactRecorder(session, tmp_path, 100000.)
    with pytest.raises(RuntimeError):
        with recorder:
            assert env.step(np.zeros(8)) == 'observation'
            recorder.phase = 1
            env.step(np.ones(7))
            raise RuntimeError('interrupted')
    assert 'step' not in env.__dict__
    with np.load(str(tmp_path / 'trajectory.npz')) as trajectory:
        np.testing.assert_array_equal(trajectory['phase'], [0, 1])
        np.testing.assert_allclose(trajectory['sim_time'], [.05, .1])
        np.testing.assert_array_equal(trajectory['success'], [False, True])
        assert trajectory['action'].shape == (2, 8)
        assert np.isnan(trajectory['action'][1, -1])


@pytest.mark.parametrize('seed', [-1, 2**64, True, 2.5])
def test_bad_fault_seed_rejected(seed):
    with pytest.raises(ValueError, match='Fault seed'):
        LiveOptions('libero_goal', 1, fault_seed=seed).validate()


def test_resume_rejects_changed_experiment_input(tmp_path, monkeypatch):
    source = tmp_path / 'input.py'
    source.write_text('value = 1\n')
    config = dict(speed=20., port=8000, seed=1000, repeats=1,
                  max_waypoint_seconds=180., max_policy_steps=None, trial_wall_timeout=900.,
                  latency_mode='end_of_plan')
    batch.atomic_json(tmp_path / 'manifest.json', dict(config=config, inputs={str(source): batch.file_hash(source)}))
    args = SimpleNamespace(output=tmp_path, **config)
    batch.initialize(args)
    source.write_text('value = 2\n')
    with pytest.raises(ValueError, match='input changed'):
        batch.initialize(args)


def test_batch_matches_current_gui_end_handoff_and_inherited_deviations(tmp_path):
    off, on = inventory(tmp_path)[:2]
    config = dict(port=8000, max_waypoint_seconds=180., max_policy_steps=None)
    immediate = batch.trial_options(off, config)
    delayed = batch.trial_options(on, config)
    immediate.validate(); delayed.validate()
    assert immediate.delay == delayed.delay == 0.
    assert not immediate.handoff_at_end and delayed.handoff_at_end
    assert immediate.propagate_deviation and delayed.propagate_deviation
    assert immediate.fault_seed == delayed.fault_seed


def test_layout_retries_a_new_seed_then_reuses_exact_cached_state(tmp_path):
    condition = dict(suite='libero_goal', task_id=1, scene='shuffled', repeat=0)
    options = LiveOptions('libero_goal', 1, scene='shuffled', use_saved_layout=False)
    runner = SimpleNamespace(layouts={}, prepare=Mock(side_effect=[
        ValueError('No different supported object position found: support'), None]))
    actual, attempts = batch.prepare_trial(runner, tmp_path, condition, options)
    assert actual.seed == options.seed+10007
    assert len(attempts) == 2 and attempts[-1]['status'] == 'ready'
    assert runner.prepare.call_args_list[0].args[0].seed != actual.seed
    state = [.75, 1., 2., 3.]
    batch.atomic_json(batch.layout_path(tmp_path, condition), dict(seed=actual.seed,
        state=state, scene_origin='verified shuffled layout', shuffle_mapping={'a':'b'},
        gripper=-1., seed_attempts=attempts))
    runner.prepare = Mock()
    restored, _ = batch.prepare_trial(runner, tmp_path, condition, options)
    assert restored.seed == actual.seed
    key = ('libero_goal', 1, 'shuffled', actual.seed, False)
    np.testing.assert_array_equal(runner.layouts[key][0], state)
    runner.prepare.assert_called_once_with(restored)


def test_unavailable_scene_is_cached_without_stopping_other_conditions(tmp_path):
    condition = dict(suite='libero_goal', task_id=1, scene='shuffled', repeat=0)
    options = LiveOptions('libero_goal', 1, scene='shuffled')
    runner = SimpleNamespace(layouts={}, prepare=Mock(side_effect=ValueError(
        'No different supported object position found: cannot verify support')))
    for _ in range(2):
        with pytest.raises(batch.SceneUnavailable, match='8 seeds'):
            batch.prepare_trial(runner, tmp_path, condition, options)
    assert runner.prepare.call_count == batch.SCENE_SEED_ATTEMPTS
    seeds = [call.args[0].seed for call in runner.prepare.call_args_list]
    assert len(set(seeds)) == batch.SCENE_SEED_ATTEMPTS


def test_other_preparation_errors_are_not_hidden_as_missing_layout(tmp_path):
    condition = dict(suite='libero_goal', task_id=1, scene='shuffled', repeat=0)
    runner = SimpleNamespace(layouts={}, prepare=Mock(side_effect=ValueError('Model belongs to another task')))
    with pytest.raises(ValueError, match='another task'):
        batch.prepare_trial(runner, tmp_path, condition, LiveOptions('libero_goal', 1, scene='shuffled'))
    runner.prepare.assert_called_once()


def test_reference_run_reuses_archived_plans_conditions_and_exact_states(tmp_path, monkeypatch):
    reference = tmp_path / 'previous'
    conditions = inventory(reference)
    monkeypatch.setattr(batch, 'SUITES', {'libero_goal': 2})
    config = dict(seed=1000, repeats=1, max_waypoint_seconds=180.,
                  max_policy_steps=None, latency_mode='end_of_plan')
    batch.atomic_json(reference / 'manifest.json', dict(config=config,
        conditions=conditions, inputs={}))
    for c in conditions:
        if not c['unavailable']:
            batch.atomic_json(batch.layout_path(reference, c), dict(seed=1000,
                state=[.125, 1.25, 3.75], scene_origin='paired reference'))
    args = SimpleNamespace(output=tmp_path/'new', reference_run=reference,
                           suites=['libero_goal'], **config)
    selected, inputs, plans, comparison = batch.reference_inventory(args)
    assert selected == conditions
    assert plans == reference/'plans'
    assert comparison['conditions'] == 64
    for c in conditions:
        if not c['unavailable']:
            assert batch.layout_path(args.output, c).read_bytes() == batch.layout_path(reference, c).read_bytes()
    # A changed handoff protocol is not a repeat of the same experiment.
    args.latency_mode = 'timed'
    with pytest.raises(ValueError, match='setting differs'):
        batch.reference_inventory(args)
