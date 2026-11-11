"""Batch planning and evidence retention; simulator rollouts use a stub policy."""
import json
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

import pi05_benchmark as batch


def test_full_schedule_has_2400_unique_paired_conditions():
    cases = batch.schedule(batch.Options())
    assert len(cases) == 2400
    assert len({batch.trial_id(spec) for spec in cases}) == 2400
    pairs = {}
    for case in cases:
        pairs.setdefault((case['suite'], case['task_id'], case['trial']), []).append(case)
    assert len(pairs) == 400
    for conditions in pairs.values():
        assert {(s['robot_start'], s['layout']) for s in conditions} == {
            (start, layout) for start in batch.STARTS for layout in batch.LAYOUTS}
        assert len({s['environment_seed'] for s in conditions}) == 1
        assert len({s['shuffle_seed'] for s in conditions}) == 1
        assert len({s['initial_state_index'] for s in conditions}) == 1
        for start in batch.STARTS:
            assert len({s['robot_seed'] for s in conditions if s['robot_start'] == start}) == 1
    assert {s['initial_state_index'] for s in cases} == set(range(10))
    for suite in batch.SUITES:
        for task in range(10):
            assert len({s['model_seed'] for s in cases if s['suite'] == suite and s['task_id'] == task}) == 1


@pytest.mark.parametrize('changes', [dict(trials=0), dict(trials=51), dict(starts=['normal','normal']),
    dict(layouts=['unknown']), dict(task_ids=[10]), dict(replan_steps=11), dict(image_every=0)])
def test_invalid_experiments_rejected(changes):
    with pytest.raises(ValueError):
        batch.schedule(batch.Options(**changes))


@pytest.fixture(scope='module')
def scene():
    from libero.libero import benchmark
    seed = batch.schedule(batch.Options())[600]['environment_seed']
    session = batch.make_session('libero_object', 0, seed, 64)
    official = np.asarray(benchmark.get_benchmark_dict()['libero_object']().get_task_init_states(0)[0])
    yield session, official
    session.env.close()


class StubPolicy:
    """A stationary gripper-open test policy, never advertised as Pi 0.5 results."""
    def __init__(self, metadata):
        self.metadata = metadata
        self.closed, self.inferences = False, 0

    def connect(self):
        return self.metadata

    def infer(self, obs):
        assert obs['observation/image'].shape == (224, 224, 3)
        assert obs['observation/wrist_image'].shape == (224, 224, 3)
        self.inferences += 1
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, -1] = -1
        return dict(actions=actions, server_timing={'infer_ms': 0.})

    def close(self):
        self.closed = True


def options(**kwargs):
    return batch.Options(**dict(dict(suites=['libero_object'], task_ids=[0], trials=1,
        resolution=64, max_steps=2, save_video=False, min_free_gib=0), **kwargs))


def test_all_six_conditions_retain_failed_images_actions_states_and_pairing(scene, tmp_path):
    session, official = scene
    opts = options()
    cases = batch.schedule(opts)
    assets = batch.task_assets(session, tmp_path, 'libero_object', 0)
    initial = {}
    configs = {}
    for spec in cases:
        policy = StubPolicy({'model_id': 'unit-test-stub'})
        result = batch.run_case(session, official, spec, opts, tmp_path, policy.metadata, assets, lambda port: policy)
        assert result['status'] == 'failed', result
        assert result['scored'] and result['success'] is False and policy.closed
        folder = tmp_path / 'trials' / batch.trial_id(spec)
        config = json.loads((folder / 'config.json').read_text())
        configs[spec['robot_start'], spec['layout']] = config
        assert config['prepared']
        assert config['assets']['instruction'] == session.task.language
        assert len(list((folder / 'frames').glob('*/*.png'))) == 8
        with h5py.File(folder / 'trajectory.h5') as data:
            assert len(data['states']) == 3
            assert len(data['integration_states']) == 3
            assert data['actions'].shape == (2, 7)
            assert data['policy_states'].shape == (3, 8)
            initial[spec['robot_start'], spec['layout']] = data['states'][0]
        with np.load(folder / 'initial.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['state'], initial[spec['robot_start'], spec['layout']])
        queries = [json.loads(line) for line in (folder / 'policy_queries.jsonl').read_text().splitlines()]
        assert len(queries) == 1 and len(queries[0]['response']['actions']) == 10
        assert (folder / 'terminal.npz').exists()
    # Robot perturbation changes only the arm; layouts are shared byte-for-byte.
    arm = session.executor.scene.qadr
    other = np.setdiff1d(np.arange(session.env.sim.model.nq), arm) + 1
    for layout in batch.LAYOUTS:
        for start in ['small', 'large']:
            np.testing.assert_array_equal(initial[start, layout][other], initial['normal', layout][other])
            assert configs[start, layout]['layout']['state_file_sha256'] == configs['normal', layout]['layout']['state_file_sha256']
            low, high = (.04, .08) if start == 'small' else (.20, .35)
            assert low <= configs[start, layout]['offset']['distance_m'] <= high
        assert configs['normal', layout]['offset']['distance_m'] == 0
    assert configs['normal', 'shuffled']['layout']['method'] in ['object_permutation', 'source_relocation']
    assert max(configs['normal', 'shuffled']['layout']['moved_xy_m'].values()) >= .02
    summary = batch.summarize(tmp_path, cases)
    assert summary['evaluated'] == 6 and summary['statuses'] == {'failed': 6}


def test_success_removes_bulk_media_but_keeps_configuration(scene, tmp_path, monkeypatch):
    session, official = scene
    goal = [False]
    monkeypatch.setattr(session.env, 'check_success', lambda: goal[0])
    policy = StubPolicy({'model_id': 'unit-test-stub'})
    infer = policy.infer
    def succeed(obs):
        result = infer(obs)
        goal[0] = True
        return result
    policy.infer = succeed
    spec = batch.schedule(options(starts=['normal'], layouts=['original']))[0]
    assets = batch.task_assets(session, tmp_path, 'libero_object', 0)
    result = batch.run_case(session, official, spec, options(), tmp_path, policy.metadata, assets, lambda port: policy)
    assert result['status'] == 'success' and result['steps'] == 1
    folder = tmp_path / 'trials' / batch.trial_id(spec)
    assert not (folder / 'frames').exists()
    assert not (folder / 'trajectory.h5').exists()
    assert all((folder / name).exists() for name in ['config.json', 'result.json', 'initial.npz', 'terminal.npz'])


def test_unavailable_shuffle_is_not_counted_as_a_policy_failure(scene, tmp_path, monkeypatch):
    import waypoint_replay
    session, official = scene
    monkeypatch.setattr(waypoint_replay, 'shuffle_objects', Mock(side_effect=ValueError('no permutation')))
    monkeypatch.setattr(waypoint_replay, 'shuffle_positions', Mock(side_effect=ValueError('no supported relocation')))
    opts = options(starts=['normal'], layouts=['shuffled'])
    spec = batch.schedule(opts)[0]
    factory = Mock()
    assets = batch.task_assets(session, tmp_path, 'libero_object', 0)
    result = batch.run_case(session, official, spec, opts, tmp_path, {}, assets, factory)
    assert result['status'] == 'setup_unavailable' and not result['scored'] and result['success'] is None
    factory.assert_not_called()
    folder = tmp_path / 'trials' / batch.trial_id(spec)
    assert (folder / 'terminal.npz').exists()
    assert list((folder / 'frames').glob('*/*.png'))
    assert batch.summarize(tmp_path, [spec])['evaluated'] == 0


def test_policy_error_is_preserved_and_retry_archives_old_attempt(scene, tmp_path):
    session, official = scene
    opts = options(starts=['normal'], layouts=['original'])
    spec = batch.schedule(opts)[0]
    assets = batch.task_assets(session, tmp_path, 'libero_object', 0)
    policy = StubPolicy({'model_id': 'unit-test-stub'})
    policy.infer = Mock(side_effect=RuntimeError('test server disconnected'))
    with pytest.raises(RuntimeError, match='saving error artifacts'):
        batch.run_case(session, official, spec, opts, tmp_path, policy.metadata, assets, lambda port: policy)
    folder = tmp_path / 'trials' / batch.trial_id(spec)
    old_result = (folder / 'result.json').read_bytes()
    assert json.loads(old_result)['status'] == 'error'
    assert (folder / 'error.txt').exists() and (folder / 'terminal.npz').exists()
    good = StubPolicy(policy.metadata)
    batch.run_case(session, official, spec, opts, tmp_path, good.metadata, assets, lambda port: good)
    previous = tmp_path / 'trials' / '_previous_attempts' / (batch.trial_id(spec) + '_001')
    assert (previous / 'result.json').read_bytes() == old_result
    assert json.loads((folder / 'result.json').read_text())['status'] == 'failed'


def test_batch_resume_skips_completed_runs_and_rejects_changed_configuration(tmp_path, monkeypatch):
    import waypoint_demo
    from serve_pi05_libero import metadata
    policy = StubPolicy(metadata())
    monkeypatch.setattr(waypoint_demo, 'PolicyConnection', lambda port: policy)
    args = batch.parser().parse_args(['run', '--output', str(tmp_path / 'run'), '--suites', 'libero_object',
        '--task-ids', '0', '--trials', '1', '--starts', 'normal', '--layouts', 'original',
        '--max-steps', '1', '--resolution', '64', '--no-video', '--min-free-gib', '0'])
    batch.run_batch(args)
    assert policy.inferences == 1
    result = next((args.output / 'trials').glob('*/result.json'))
    before = result.read_bytes()
    args.resume = True
    batch.run_batch(args)
    assert policy.inferences == 1 and result.read_bytes() == before
    args.trials = 2
    with pytest.raises(ValueError, match='identical options'):
        batch.run_batch(args)


def test_disk_reserve_stops_before_space_is_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(batch.shutil, 'disk_usage', lambda path: type('Disk', (), {'free': 1024})())
    with pytest.raises(batch.DiskReserveError):
        batch.check_disk(tmp_path, 5.)


def test_video_and_state_replay_need_no_policy_server(scene, tmp_path, capsys):
    from types import SimpleNamespace
    session, official = scene
    opts = options(starts=['normal'], layouts=['original'], save_video=True)
    spec = batch.schedule(opts)[0]
    assets = batch.task_assets(session, tmp_path, 'libero_object', 0)
    policy = StubPolicy({'model_id': 'unit-test-stub'})
    result = batch.run_case(session, official, spec, opts, tmp_path, policy.metadata, assets, lambda port: policy)
    folder = tmp_path / 'trials' / batch.trial_id(spec)
    assert not result.get('artifact_errors'), result
    assert (folder / 'rollout.mp4').stat().st_size > 0
    args = SimpleNamespace(trial=folder, mode='states', render=False, realtime=False,
                           max_steps=None, output=tmp_path / 'replay.mp4')
    batch.replay(args)
    assert args.output.stat().st_size > 0
    assert '"maximum_flat_state_error": 0.0' in capsys.readouterr().out
    # Action replay executes saved controls without invoking policy inference.
    args.mode, args.output = 'actions', None
    batch.replay(args)
    assert policy.inferences == 1


def test_interrupted_trial_keeps_partial_evidence_and_is_resumable(scene, tmp_path):
    session, official = scene
    opts = options(starts=['normal'], layouts=['original'])
    spec = batch.schedule(opts)[0]
    assets = batch.task_assets(session, tmp_path, 'libero_object', 0)
    policy = StubPolicy({'model_id': 'unit-test-stub'})
    policy.infer = Mock(side_effect=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        batch.run_case(session, official, spec, opts, tmp_path, policy.metadata, assets, lambda port: policy)
    folder = tmp_path / 'trials' / batch.trial_id(spec)
    result = json.loads((folder / 'result.json').read_text())
    assert result['status'] == 'interrupted' and not result['scored']
    assert result['status'] not in batch.COMPLETE
    assert policy.closed and (folder / 'terminal.npz').exists()
    with h5py.File(folder / 'trajectory.h5') as data:
        assert len(data['states']) == 1


def test_reset_retries_are_bounded_and_interrupts_are_not_swallowed():
    from robosuite.utils.errors import RandomizationError
    session = Mock()
    session.env.env.reset.side_effect = RandomizationError('no feasible layout')
    with pytest.raises(batch.SetupUnavailable, match='3 reset attempts'):
        batch.reset_environment(session, attempts=3)
    assert session.env.env.reset.call_count == 3
    session.env.reset.assert_not_called()
    session.env.env.reset.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        batch.reset_environment(session)


def test_official_states_do_not_rerandomize_fixture_geometry(scene, tmp_path, monkeypatch):
    session, official = scene
    opts = options(starts=['normal'], layouts=['original'])
    spec = batch.schedule(opts)[0]
    before = session.env.sim.model.body_pos.copy()
    monkeypatch.setattr(session.env.env, 'reset', Mock(side_effect=AssertionError('fixtures must stay fixed')))
    batch.prepare_layout(session, official, spec, opts, tmp_path)
    np.testing.assert_array_equal(session.env.sim.model.body_pos, before)
    session.env.env.reset.assert_not_called()
