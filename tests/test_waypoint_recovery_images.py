"""The compact benchmark must preserve pairs, measure handoffs and save only PNG/CSV."""
import csv
import hashlib
import json
import os
import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import waypoint_recovery_images as runner
from waypoint_gmm import GMMWaypoint
from waypoint_policy import StableSuccess
from serve_pi05_libero import metadata as pi05_metadata


def poses(x=0., angle=0.):
    return {'bowl': (np.array([x, 0., 1.]), Rotation.from_euler('z', angle, degrees=True).as_matrix())}


def test_stability_requires_goal_and_four_continuous_seconds():
    stable = StableSuccess(4.)
    assert not stable.update(0., False, poses())
    assert not stable.update(1., True, poses())
    assert not stable.update(4.95, True, poses())
    assert stable.update(5., True, poses())
    assert stable.duration == 4.
    assert not stable.update(5.1, False, poses())
    assert not stable.update(9., True, poses())
    assert stable.update(13., True, poses())


@pytest.mark.parametrize('moved', [poses(.02), poses(angle=6.)])
def test_object_translation_or_rotation_restarts_stability(moved):
    stable = StableSuccess(4.)
    stable.update(0., True, poses())
    assert not stable.update(3., True, moved)
    assert not stable.update(6.9, True, moved)
    assert stable.update(7., True, moved)
    assert stable.first_goal == 0.
    assert stable.start == 3.


def test_all_suites_and_full_unique_matrix_share_layouts_and_fault_draws():
    tasks = runner.all_tasks(runner.SUITES)
    assert len(tasks) == 130
    specs = runner.schedule(tasks)
    assert len(specs) == 4680
    assert len({runner.identifier(s) for s in specs}) == 4680
    for suite, task in tasks:
        rows = [s for s in specs if (s['suite'], s['task_id']) == (suite, task)]
        assert len(rows) == 36
        assert {s['scene'] for s in rows} == {'original', 'shuffled'}
        for scene in ('original', 'shuffled'):
            subset = [s for s in rows if s['scene'] == scene]
            assert len([s for s in subset if s['kind'] == 'none']) == 2
            faults = [s for s in subset if s['mode'] == 'recovery']
            assert {s['phase'] for s in faults} == {'grasp', 'transport'}
            assert {s['kind'] for s in faults} == set(runner.FAILURES)
            for first in faults[::2]:
                pair = [s for s in faults if (s['kind'], s['intensity'], s['phase']) ==
                        (first['kind'], first['intensity'], first['phase'])]
                assert {s['requested_delay_seconds'] for s in pair} == {0., 5.}
                assert len({s['fault_seed'] for s in pair}) == 1
                assert len({s['seed'] for s in pair}) == 1
            loss = [s for s in faults if s['kind'] == 'grasp_loss']
            assert {(s['phase'], s['intensity']) for s in loss} == {('grasp', 'low'), ('transport', 'high')}


def test_only_high_grasp_loss_can_arm_during_transport():
    chain = SimpleNamespace(waypoints=[GMMWaypoint(name=role.title(), skill_role=role)
        for role in ('move', 'push', 'grasp', 'transport', 'place', 'release')])
    for kind in runner.FAILURES:
        for intensity in runner.INTENSITIES:
            for phase in ('move', 'push', 'transport', 'place', 'release'):
                if (kind, intensity, phase) == ('grasp_loss', 'high', 'transport'):
                    assert runner.fault_intensities(kind, phase, 'line') == ('high',)
                    assert runner.eligible_row(chain, dict(kind=kind, intensity=intensity, phase=phase)) == 3
                else:
                    assert intensity not in runner.fault_intensities(kind, phase, 'line')
                    assert runner.eligible_row(chain, dict(kind=kind, intensity=intensity, phase=phase)) is None
    assert runner.eligible_row(chain, dict(kind='grasp_loss', intensity='high', phase='grasp')) is None
    assert runner.eligible_row(chain, dict(kind='grasp_loss', intensity='low', phase='grasp')) == 2
    injector = runner.GraspInjector.__new__(runner.GraspInjector)
    session = SimpleNamespace(injected_failure='')
    runner.bench.Injector.__init__(injector, session, {2: dict(kind='grasp_loss', intensity='low')}, waypoints=chain.waypoints)
    injector.start_row(2)
    assert injector.armed == 'grasp_loss'
    assert injector.primitive == 'grasp'
    assert injector.missed == []
    drop = runner.GraspInjector.__new__(runner.GraspInjector)
    runner.bench.Injector.__init__(drop, session, {3: dict(kind='grasp_loss', intensity='high')},
                                 waypoints=chain.waypoints)
    drop.start_row(3)
    assert drop.armed == 'grasp_loss'
    assert drop.primitive == 'transport'
    assert drop.missed == []
    assert runner.fault_intensities('grasp_loss', 'transport', 'follow') == ()
    blocked = runner.GraspInjector.__new__(runner.GraspInjector)
    runner.bench.Injector.__init__(blocked, session, {4: dict(kind='orientation', intensity='high')},
                                  waypoints=chain.waypoints)
    blocked.start_row(4)
    assert blocked.armed is None
    assert blocked.missed


def test_task_condition_hierarchy_keeps_all_4680_trials_separate():
    specs = runner.schedule(runner.all_tasks(runner.SUITES))
    assert len({runner.trial_path(spec) for spec in specs}) == 4680
    for spec in specs:
        assert runner.trial_path(runner.identifier(spec)) == runner.trial_path(spec)
    assert runner.trial_path(specs[0]).as_posix() == 'trials/libero_spatial/task_000/original/no_failure/waypoints'
    assert runner.trial_path(specs[1]).as_posix() == 'trials/libero_spatial/task_000/original/no_failure/vla'
    example = next(s for s in specs if s['suite'] == 'libero_spatial' and s['task_id'] == 0
        and s['scene'] == 'shuffled' and s['kind'] == 'orientation' and s['intensity'] == 'high'
        and s['phase'] == 'grasp' and s['requested_delay_seconds'] == 5.)
    assert runner.trial_path(example).as_posix() == (
        'trials/libero_spatial/task_000/shuffled/failures/orientation/high/grasp/handoff_5s')
    example = next(s for s in specs if s['suite'] == 'libero_spatial' and s['task_id'] == 0
        and s['scene'] == 'shuffled' and s['kind'] == 'grasp_loss' and s['intensity'] == 'high'
        and s['requested_delay_seconds'] == 5.)
    assert runner.trial_path(example).as_posix() == (
        'trials/libero_spatial/task_000/shuffled/failures/grasp_loss/high/transport/handoff_5s')


def test_grasp_only_resume_preserves_prior_rows_and_fault_seed(tmp_path):
    old = dict(version=1, policy=pi05_metadata(), image_layout=runner.IMAGE_LAYOUT,
        max_policy_steps=800, seed=1000,
        code={'waypoint_recovery_images.py': runner.PRE_GRASP_ONLY_IMAGES_SHA256,
              'waypoint_failures.py': 'same-faults', 'waypoint_policy.py': 'same-policy'})
    old_digest = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    config = dict(old, failure_phases=['grasp'], code=dict(old['code'],
        **{'waypoint_recovery_images.py': 'grasp-only-code'}))
    current, compatible = runner.configuration_digests(config)
    assert old_digest in compatible
    spec = runner.schedule([('libero_goal', 1)])[2]
    old_seed = int(hashlib.sha256(b'1000/libero_goal/1/original/collision/low/grasp').hexdigest()[:8], 16)
    assert spec['fault_seed'] == old_seed
    saved = runner.CsvResults(tmp_path, old_digest, False)
    saved.append(dict(spec, trial_id=runner.identifier(spec), configuration_sha256=old_digest,
                      success=True, status='recovered', initial_state_sha256='same-scene'))
    saved.close()
    original = (tmp_path/'timing.csv').read_bytes()
    resumed = runner.CsvResults(tmp_path, current, True, compatible)
    assert resumed.completed == {runner.identifier(spec)}
    assert resumed.state_hashes == {('libero_goal', 1, 'original'): 'same-scene'}
    resumed.close()
    assert (tmp_path/'timing.csv').read_bytes() == original


@pytest.mark.parametrize('change', ['fault_code', 'policy', 'seed', 'steps'])
def test_grasp_only_resume_still_rejects_protocol_changes(change):
    old = dict(version=1, policy=pi05_metadata(), image_layout=runner.IMAGE_LAYOUT,
        max_policy_steps=800, seed=1000,
        code={'waypoint_recovery_images.py': runner.PRE_GRASP_ONLY_IMAGES_SHA256,
              'waypoint_failures.py': 'same-faults', 'waypoint_policy.py': 'same-policy'})
    old_digest = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    config = dict(old, failure_phases=['grasp'], code=dict(old['code'],
        **{'waypoint_recovery_images.py': 'grasp-only-code'}))
    if change == 'fault_code':
        config['code']['waypoint_failures.py'] = 'different-faults'
    elif change == 'policy':
        config['policy'] = dict(pi05_metadata(), model_id='different-policy')
    elif change == 'seed':
        config['seed'] = 2000
    else:
        config['max_policy_steps'] = 400
    current, compatible = runner.configuration_digests(config)
    assert old_digest not in {current, *compatible}


def pi05_resume_configuration():
    return dict(version=1, policy=pi05_metadata(), image_layout=runner.IMAGE_LAYOUT,
        failure_phases=['grasp'], max_policy_steps=800, seed=1000,
        plans={'libero_goal/1': 'same-plan'},
        code={'waypoint_recovery_images.py': 'current-images-code',
              'waypoint_policy.py': runner.UDVLA_POLICY_SHA256,
              'waypoint_failures.py': 'same-faults'})


def pi05_previous_digests(config):
    previous = dict(config, code=dict(config['code'],
        **{'waypoint_recovery_images.py': runner.PRE_RESUME_IMAGES_SHA256,
           'waypoint_policy.py': runner.PRE_UDVLA_POLICY_SHA256}))
    grasp = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    previous.pop('failure_phases')
    previous['code']['waypoint_recovery_images.py'] = runner.PRE_GRASP_ONLY_IMAGES_SHA256
    full = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    return full, grasp


def test_pi05_resume_preserves_mixed_pre_udvla_rows(tmp_path):
    config = pi05_resume_configuration()
    full, grasp = pi05_previous_digests(config)
    saved = runner.CsvResults(tmp_path, full, False)
    specs = runner.schedule([('libero_goal', 1)])[:2]
    for spec, digest in zip(specs, (full, grasp)):
        saved.append(dict(spec, trial_id=runner.identifier(spec), configuration_sha256=digest,
                          status='normal_policy_success', success=True,
                          initial_state_sha256='same-scene', policy_steps=123))
    saved.close()
    path = tmp_path/'timing.csv'
    original = path.read_bytes()
    current, compatible = runner.configuration_digests(config)
    assert {full, grasp}.issubset(compatible)
    for _ in range(2):
        resumed = runner.CsvResults(tmp_path, current, True, compatible)
        assert resumed.completed == {runner.identifier(spec) for spec in specs}
        assert resumed.state_hashes == {('libero_goal', 1, 'original'): 'same-scene'}
        resumed.close()
        assert path.read_bytes() == original


@pytest.mark.parametrize('change', ['seed', 'plans', 'steps', 'policy', 'fault_code',
                                   'policy_code', 'phase'])
def test_pi05_pre_udvla_resume_rejects_experiment_changes(change):
    config = pi05_resume_configuration()
    previous = pi05_previous_digests(config)
    if change == 'seed':
        config['seed'] = 2000
    elif change == 'plans':
        config['plans'] = {'libero_goal/1': 'changed-plan'}
    elif change == 'steps':
        config['max_policy_steps'] = 400
    elif change == 'policy':
        config['policy'] = dict(pi05_metadata(), num_steps=5)
    elif change == 'phase':
        config['failure_phases'] = ['place']
    else:
        name = 'waypoint_policy.py' if change == 'policy_code' else 'waypoint_failures.py'
        config['code'][name] = 'changed-controller'
    current, compatible = runner.configuration_digests(config)
    assert set(previous).isdisjoint({current, *compatible})


def transport_resume_configurations():
    previous = pi05_resume_configuration()
    previous['code'] = dict(previous['code'],
        **{'waypoint_recovery_images.py': runner.PRE_TRANSPORT_IMAGES_SHA256})
    current = dict(previous, failure_phases=['grasp', 'transport'],
        code=dict(previous['code'], **{'waypoint_recovery_images.py': 'transport-images-code'}))
    return previous, current


def test_transport_resume_preserves_old_rows_and_adds_only_four_drop_conditions(tmp_path):
    previous, config = transport_resume_configurations()
    old_digest, _ = runner.configuration_digests(previous)
    old_digests = (old_digest, *pi05_previous_digests(previous))
    current, compatible = runner.configuration_digests(config)
    assert set(old_digests).issubset(compatible)
    specs = runner.schedule([('libero_goal', 1)])
    old_specs = [s for s in specs if s['phase'] != 'transport']
    saved = runner.CsvResults(tmp_path, old_digest, False)
    for i, spec in enumerate(old_specs):
        saved.append(dict(spec, trial_id=runner.identifier(spec),
                          configuration_sha256=old_digests[i % len(old_digests)],
                          status='recovered', success=True, initial_state_sha256=spec['scene']))
    saved.close()
    original = (tmp_path/'timing.csv').read_bytes()
    resumed = runner.CsvResults(tmp_path, current, True, compatible)
    pending = [s for s in specs if runner.identifier(s) not in resumed.completed]
    resumed.close()
    assert len(old_specs) == 32
    assert len(pending) == 4
    assert {(s['scene'], s['requested_delay_seconds']) for s in pending} == {
        ('original', 0.), ('original', 5.), ('shuffled', 0.), ('shuffled', 5.)}
    assert all((s['kind'], s['intensity'], s['phase']) ==
               ('grasp_loss', 'high', 'transport') for s in pending)
    assert (tmp_path/'timing.csv').read_bytes() == original


@pytest.mark.parametrize('change', ['seed', 'plans', 'steps', 'policy', 'fault_code',
                                   'policy_code', 'phase', 'delay', 'stability'])
def test_transport_resume_rejects_changes_to_existing_experiment(change):
    previous, config = transport_resume_configurations()
    old_digest, _ = runner.configuration_digests(previous)
    if change == 'seed':
        config['seed'] = 2000
    elif change == 'plans':
        config['plans'] = {'libero_goal/1': 'changed-plan'}
    elif change == 'steps':
        config['max_policy_steps'] = 400
    elif change == 'policy':
        config['policy'] = dict(pi05_metadata(), num_steps=5)
    elif change == 'phase':
        config['failure_phases'] = ['grasp', 'transport', 'place']
    elif change == 'delay':
        config['delays'] = [0., 2.]
    elif change == 'stability':
        config['stable_seconds'] = 2.
    else:
        name = 'waypoint_policy.py' if change == 'policy_code' else 'waypoint_failures.py'
        config['code'][name] = 'changed-controller'
    current, compatible = runner.configuration_digests(config)
    assert old_digest not in {current, *compatible}


def test_dry_run_resume_reports_transport_drops_without_writing(tmp_path, monkeypatch, capsys):
    specs = runner.schedule([('libero_goal', 1)])
    saved = runner.CsvResults(tmp_path, 'old-config', False)
    for spec in specs:
        if spec['phase'] != 'transport':
            saved.append(dict(spec, trial_id=runner.identifier(spec), configuration_sha256='old-config'))
    saved.close()
    original = (tmp_path/'timing.csv').read_bytes()
    monkeypatch.setattr(runner, 'all_tasks', lambda suites, task_ids: [('libero_goal', 1)])
    monkeypatch.setattr(runner, 'plan_path', lambda suite, task: None)
    args = SimpleNamespace(task_suites=['libero_goal'], task_ids=None, seed=1000,
        handoff_delays=[0., 5.], stable_seconds=4., stability_metres=.01, stability_degrees=5.,
        photo_every=.5, limit=None, dry_run=True, output=tmp_path, resume=True)
    runner.run_images(args, argparse.ArgumentParser())
    output = capsys.readouterr().out
    assert '32 scheduled conditions already recorded; 4 pending' in output
    assert 'grasp_loss_high_transport_delay0s' in output
    assert 'grasp_loss_high_transport_delay5s' in output
    assert (tmp_path/'timing.csv').read_bytes() == original
    assert list((tmp_path/'trials').iterdir()) == []


def fake_session():
    data = SimpleNamespace(time=0.)
    state = np.array([0., 1., 2.])
    def step(action):
        data.time += .05
        state[0] = data.time
        return {}, 0., False, {}
    def set_state(value):
        state[:] = value
        data.time = float(state[0])
    env = SimpleNamespace(sim=SimpleNamespace(data=data, step=lambda: None, get_state=lambda: state.copy()),
        env=SimpleNamespace(_observables={}, modify_observable=Mock(), control_timestep=.05),
        step=step, set_init_state=set_state, check_success=lambda: False)
    session = SimpleNamespace(env=env, executor=SimpleNamespace(gripper=-1.), tick=lambda: None,
        image=lambda: np.zeros((12, 16, 3), dtype=np.uint8),
        attach=Mock(), task=SimpleNamespace(language='put the bowl on the plate'),
        restore_training_scene=lambda: set_state(np.array([0., 1., 2.])),
        shuffle_scene=Mock(side_effect=lambda seed: set_state(np.array([0., 4., 5.]))))
    def hold(count):
        for _ in range(count):
            session.env.step(np.zeros(7))
            session.tick()
    session.executor.hold = hold
    return session


@pytest.mark.parametrize('delay', [0., 5.])
def test_handoff_uses_observed_failure_clock_and_restores_callbacks(delay):
    session = fake_session()
    original_tick = session.tick
    chain = SimpleNamespace(waypoints=[GMMWaypoint(name='Grasp', skill_role='grasp')])
    def run(chain, report):
        report(0, 'Running')
        session.executor.hold(3)
        return 'stopped'
    session.run = run
    class Detector:
        def __init__(self, session, injections, clock, waypoints):
            self.clock, self.fired, self.missed = clock, [], []
        def observe_control(self):
            if session.env.sim.data.time >= .1:
                self.clock.mark(session.env.sim.data.time, kind='collision')
        def observe_physics(self):
            pass
        def start_row(self, row):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
    captures = []
    result = runner.bench.run_fault(session, chain,
        dict(kind='collision', intensity='low', injection_row=0, fault_seed=7), delay, 30., None,
        injector_class=Detector, intensity_rule=runner.fault_intensities,
        capture=lambda session, folder, label: captures.append(label))
    assert result['observed_failure']
    assert result['failure_event']['sim_time'] == pytest.approx(.1)
    assert result['actual_delay_seconds'] == pytest.approx(delay, abs=.051)
    assert captures == ['failure', 'handoff']
    assert session.tick is original_tick
    assert not session.continue_on_failure


def test_paired_scenes_restore_exact_states_once_without_disk_files():
    session = fake_session()
    paired = runner.PairedScenes()
    spec = dict(suite='libero_goal', task_id=1, scene='shuffled', seed=1000)
    digest = paired.restore(session, None, spec)
    first = session.env.sim.get_state().copy()
    session.executor.hold(20)
    assert paired.restore(session, None, spec) == digest
    np.testing.assert_array_equal(session.env.sim.get_state(), first)
    session.shuffle_scene.assert_called_once_with(seed=1000)
    assert session.attach.call_args.kwargs == dict(settle=False)
    changed = runner.PairedScenes({('libero_goal', 1, 'shuffled'): 'different-state'})
    with pytest.raises(ValueError, match='starting scene changed'):
        changed.restore(session, None, spec)


def test_csv_and_images_are_the_only_outputs_and_resume_preserves_rows(tmp_path):
    session = fake_session()
    spec = runner.schedule([('libero_goal', 1)])[2]
    trial_id = runner.identifier(spec)
    results = runner.CsvResults(tmp_path, 'config', False)
    original = session.env.step
    with runner.Images(session, tmp_path, trial_id, .1) as images:
        images.capture('initial')
        session.env.step(np.zeros(7))
        images.controller = 'vla'
        session.env.step(np.zeros(7))
        images.capture('terminal')
    assert session.env.step is original
    results.append(dict(spec, trial_id=trial_id, configuration_sha256='config', status='recovered',
                        success=True, failure_measurement={'requested_metres': .3}))
    results.close()
    assert {p.name for p in tmp_path.iterdir()} == {'timing.csv', 'trials'}
    folder = tmp_path/runner.trial_path(spec)
    assert {p.name for p in folder.iterdir()} == {'photos', 'initial.png', 'terminal.png'}
    assert {p.name for p in (folder/'photos').iterdir()} == {'waypoints', 'policy'}
    assert all(p.suffix == '.png' for p in folder.rglob('*') if p.is_file())
    assert list((folder/'photos'/'waypoints').glob('frame_*_t_*.png'))
    assert list((folder/'photos'/'policy').glob('frame_*_t_*.png'))
    with (tmp_path/'timing.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['trial_dir'] == 'trials/libero_goal/task_001/original/failures/collision/low/grasp/handoff_0s'
    assert rows[0]['waypoint_image_dir'] == (runner.trial_path(spec)/'photos'/'waypoints').as_posix()
    assert rows[0]['vla_image_dir'] == (runner.trial_path(spec)/'photos'/'policy').as_posix()
    resumed = runner.CsvResults(tmp_path, 'config', True)
    assert resumed.completed == {trial_id}
    resumed.close()
    with pytest.raises(ValueError, match='same options'):
        runner.CsvResults(tmp_path, 'other-config', True)
    with pytest.raises(ValueError, match='--resume'):
        runner.CsvResults(tmp_path, 'config', False)


def test_resume_moves_flat_photos_into_trials_and_preserves_completed_csv(tmp_path):
    old = dict(version=1, policy=pi05_metadata(), photo_every=.5, code={
        'waypoint_recovery_images.py': runner.PRE_TRIALS_IMAGES_SHA256,
        'waypoint_policy.py': 'same-policy-code'})
    old_digest = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    config = dict(old, image_layout=runner.IMAGE_LAYOUT, code={
        'waypoint_recovery_images.py': 'new-output-code', 'waypoint_policy.py': 'same-policy-code'})
    current, compatible = runner.configuration_digests(config)
    assert old_digest in compatible
    spec = runner.schedule([('libero_goal', 1)])[2]
    trial_id = runner.identifier(spec)
    row = dict(spec, trial_id=trial_id, configuration_sha256=old_digest, initial_state_sha256='original-state',
        status='recovered', success=True, waypoint_images=3, vla_images=2)
    path = tmp_path/'timing.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.PRE_CONDITIONS_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    original = path.read_bytes()
    for name in ('waypoint_images', 'vla_images'):
        (tmp_path/name).mkdir()
    photos = [('waypoint', 0, 0., 'initial'), ('waypoint', 1, .1, 'failure'),
              ('waypoint', 2, .1, 'handoff'), ('vla', 0, .1, 'handoff'),
              ('vla', 1, 4.1, 'terminal')]
    for controller, count, timestamp, label in photos:
        name = 'waypoint_images' if controller == 'waypoint' else 'vla_images'
        filename = '%s_%s_%05d_t%.3f_%s.png' % (trial_id, controller, count, timestamp, label)
        (tmp_path/name/filename).write_bytes(('%s-%s' % (controller, label)).encode())
    changed = dict(config, photo_every=1.)
    changed_current, changed_compatible = runner.configuration_digests(changed)
    with pytest.raises(ValueError, match='same options'):
        runner.CsvResults(tmp_path, changed_current, True, changed_compatible)
    assert path.read_bytes() == original
    assert not (tmp_path/'trials').exists()
    results = runner.CsvResults(tmp_path, current, True, compatible)
    assert results.completed == {trial_id}
    assert results.state_hashes == {(spec['suite'], spec['task_id'], spec['scene']): 'original-state'}
    results.close()
    assert {p.name for p in tmp_path.iterdir()} == {'timing.csv', 'trials'}
    with path.open() as stream:
        saved_row = list(csv.DictReader(stream))[0]
    with path.open() as stream:
        assert csv.DictReader(stream).fieldnames == list(runner.FIELDS)
    original_row = list(csv.DictReader(original.decode().splitlines()))[0]
    assert all(saved_row[key] == value for key, value in original_row.items())
    assert saved_row['trial_dir'] == runner.trial_path(spec).as_posix()
    folder = tmp_path/runner.trial_path(spec)
    for controller, count, timestamp, label in photos:
        filename = 'frame_%06d_t_%09.3f.png' % (count, timestamp)
        destination = folder/'photos'/runner.CONTROLLER_FOLDERS[controller]/filename
        assert destination.read_bytes() == ('%s-%s' % (controller, label)).encode()
    assert (folder/'initial.png').read_bytes() == b'waypoint-initial'
    assert (folder/'failure.png').read_bytes() == b'waypoint-failure'
    assert (folder/'handoff.png').read_bytes() == b'vla-handoff'
    assert (folder/'terminal.png').read_bytes() == b'vla-terminal'
    # A second resume does not move or rewrite completed output again.
    updated = path.read_bytes()
    resumed = runner.CsvResults(tmp_path, current, True, compatible)
    resumed.close()
    assert path.read_bytes() == updated


def test_flat_migration_checks_all_files_before_moving_anything(tmp_path):
    trial_id = runner.identifier(runner.schedule([('libero_goal', 1)])[0])
    results = runner.CsvResults(tmp_path, 'config', False)
    results.append(dict(trial_id='trial', configuration_sha256='config'))
    results.close()
    (tmp_path/'trials').rmdir()
    directory = tmp_path/'waypoint_images'
    directory.mkdir()
    photo = directory/(trial_id+'_waypoint_00000_t0.000_initial.png')
    photo.write_bytes(b'preserve this image')
    unknown = directory/'unknown.png'
    unknown.write_bytes(b'preserve this too')
    with pytest.raises(ValueError, match='Unrecognized old image'):
        runner.CsvResults(tmp_path, 'config', True)
    assert photo.read_bytes() == b'preserve this image'
    assert unknown.read_bytes() == b'preserve this too'
    assert not (tmp_path/'trials').exists()


def test_resume_groups_old_trial_folders_and_backfills_csv_paths(tmp_path):
    old = dict(version=1, policy=pi05_metadata(), image_layout=runner.PRE_CONDITIONS_LAYOUT,
        code={'waypoint_recovery_images.py': runner.PRE_CONDITIONS_IMAGES_SHA256,
              'waypoint_policy.py': 'same-policy-code'})
    old_digest = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    config = dict(old, image_layout=runner.IMAGE_LAYOUT,
                  code=dict(old['code'], **{'waypoint_recovery_images.py': 'organized-output-code'}))
    current, compatible = runner.configuration_digests(config)
    assert old_digest in compatible
    scheduled = runner.schedule([('libero_spatial', 0), ('libero_goal', 1)])
    historical_place = dict(next(s for s in scheduled if
        s['suite'] == 'libero_goal' and s['scene'] == 'shuffled' and s['kind'] == 'orientation'
        and s['intensity'] == 'high' and s['phase'] == 'grasp' and s['requested_delay_seconds'] == 5.), phase='place')
    specs = [scheduled[0], scheduled[1], historical_place]
    path = tmp_path/'timing.csv'
    old_rows = []
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.PRE_CONDITIONS_FIELDS)
        writer.writeheader()
        for spec in specs:
            trial_id = runner.identifier(spec)
            row = dict(spec, trial_id=trial_id, configuration_sha256=old_digest,
                       status='completed', success=True, stable_seconds=4., worker=5, worker_pid=12345)
            writer.writerow(row)
            old_rows.append(row)
            folder = tmp_path/'trials'/trial_id
            for controller in ('waypoints', 'policy'):
                directory = folder/'photos'/controller
                directory.mkdir(parents=True)
                (directory/'frame_000000_t_00000.000.png').write_bytes((trial_id+'-'+controller).encode())
            (folder/'terminal.png').write_bytes(trial_id.encode())
    before = path.read_bytes()
    with pytest.raises(ValueError, match='same options'):
        runner.CsvResults(tmp_path, 'wrong-config', True)
    assert path.read_bytes() == before
    assert all((tmp_path/'trials'/r['trial_id']).is_dir() for r in old_rows)
    results = runner.CsvResults(tmp_path, current, True, compatible)
    assert results.completed == {r['trial_id'] for r in old_rows}
    results.close()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    previous = list(csv.DictReader(before.decode().splitlines()))
    for spec, old_row, new_row in zip(specs, previous, rows):
        assert all(new_row[key] == value for key, value in old_row.items())
        assert {key: new_row[key] for key in runner.image_paths(spec)} == runner.image_paths(spec)
        folder = tmp_path/new_row['trial_dir']
        assert (folder/'terminal.png').read_bytes() == new_row['trial_id'].encode()
        for controller in ('waypoints', 'policy'):
            assert (folder/'photos'/controller/'frame_000000_t_00000.000.png').read_bytes() == (
                new_row['trial_id']+'-'+controller).encode()
        assert not (tmp_path/'trials'/new_row['trial_id']).exists()
    after = path.read_bytes()
    resumed = runner.CsvResults(tmp_path, current, True, compatible)
    resumed.close()
    assert path.read_bytes() == after


def test_grouping_trial_folders_preserves_existing_destination(tmp_path):
    specs = runner.schedule([('libero_goal', 1)])[:2]
    for spec in specs:
        folder = tmp_path/'trials'/runner.identifier(spec)
        folder.mkdir(parents=True)
        (folder/'initial.png').write_bytes(b'old data')
    destination = runner.trial_folder(tmp_path, specs[1])
    (destination/'initial.png').write_bytes(b'existing destination data')
    with pytest.raises(ValueError, match='would overwrite'):
        runner.migrate_trial_folders(tmp_path)
    assert (destination/'initial.png').read_bytes() == b'existing destination data'
    assert all((tmp_path/'trials'/runner.identifier(spec)/'initial.png').read_bytes() == b'old data'
               for spec in specs)


def test_resume_cleans_only_the_pending_trial_images(tmp_path, monkeypatch):
    import waypoint_gmm_editor as editor
    monkeypatch.setattr(editor, 'GMMSession', lambda args: SimpleNamespace(args=args, env=None))
    specs = [s for s in runner.schedule([('libero_goal', 1)]) if s['mode'] == 'normal_waypoints']
    complete, pending = [runner.identifier(s) for s in specs]
    complete_folder = runner.trial_folder(tmp_path, complete)
    pending_folder = runner.trial_folder(tmp_path, pending)
    saved = complete_folder/'photos'/'waypoints'/'frame_000000_t_00000.000.png'
    saved.write_bytes(b'completed trial')
    stale = pending_folder/'photos'/'waypoints'/'frame_000000_t_00000.000.png'
    stale.write_bytes(b'interrupted trial')
    snapshot = pending_folder/'initial.png'
    snapshot.write_bytes(b'interrupted snapshot')
    args = SimpleNamespace(output=tmp_path, port=8000, max_policy_steps=800)
    key = specs[0]['suite'], specs[0]['task_id']
    entries = list(runner.iter_entries(args, specs, {key: None}, {str(key): None},
                                      'config', {complete}, {}))
    assert [e['trial_id'] for e in entries] == [pending]
    assert entries[0]['status'] == 'missing_plan'
    assert saved.read_bytes() == b'completed trial'
    assert not stale.exists() and not snapshot.exists()
    assert {p.name for p in (pending_folder/'photos').iterdir()} == {'waypoints', 'policy'}


def test_recovery_is_scored_only_after_stability_and_keeps_controller_images(tmp_path, monkeypatch):
    session = fake_session()
    runner.CsvResults(tmp_path, 'config', False).close()
    spec = runner.schedule([('libero_goal', 1)])[2]
    spec['injection_row'] = 0
    def fault(session, chain, spec, delay, *args, **kwargs):
        session.executor.hold(2)
        failure_time = session.env.sim.data.time
        session.executor.hold(round(delay/.05))
        kwargs['capture'](session, None, 'failure')
        kwargs['capture'](session, None, 'handoff')
        return dict(observed_failure=True, injection_row=0, waypoint_result=None,
            fault_targets=[], missed=[], failure_event=dict(sim_time=failure_time),
            handoff_sim_time=session.env.sim.data.time, actual_delay_seconds=delay, success_before_policy=False)
    def policy(session, language, **kwargs):
        assert kwargs['backend'] == 'pi05'
        assert kwargs['stable_seconds'] == 4.
        assert kwargs.get('report') is None  # No per-step policy logging.
        start = session.env.sim.data.time
        session.executor.hold(100)
        return dict(success=True, task_success=True, stable_seconds=4., steps=100,
            policy_start_sim_time=start, first_action_sim_time=start+.05,
            first_goal_sim_time=start+1., stable_start_sim_time=start+1.,
            policy_end_sim_time=session.env.sim.data.time)
    monkeypatch.setattr(runner.bench, 'run_fault', fault)
    monkeypatch.setattr(runner, 'run_policy', policy)
    args = SimpleNamespace(output=tmp_path, photo_every=.5, max_policy_steps=400,
        max_waypoint_seconds=180., stable_seconds=4., stability_metres=.01, stability_degrees=5., port=8000)
    entry = dict(spec, trial_id=runner.identifier(spec), success=False)
    runner.execute(session, None, spec, args, runner.PairedScenes(), entry)
    assert entry['status'] == 'recovered'
    assert entry['vla_recovered']
    assert entry['failure_to_stable_success_seconds'] == pytest.approx(5.)
    assert entry['waypoint_images'] > 0
    assert entry['vla_images'] > 0
    assert {p.name for p in tmp_path.iterdir()} == {'timing.csv', 'trials'}
    folder = tmp_path/runner.trial_path(spec)
    assert {p.name for p in folder.iterdir()} == {'photos', 'initial.png', 'failure.png', 'handoff.png', 'terminal.png'}
    assert len(list((folder/'photos'/'waypoints').glob('*.png'))) == entry['waypoint_images']
    assert len(list((folder/'photos'/'policy').glob('*.png'))) == entry['vla_images']


def test_main_records_missing_plans_and_resumes_without_duplicate_rows(tmp_path, monkeypatch):
    import waypoint_demo
    import waypoint_gmm_editor
    conn = Mock()
    conn.connect.return_value = pi05_metadata()
    monkeypatch.setattr(waypoint_demo, 'PolicyConnection', lambda port: conn)
    monkeypatch.setattr(runner, 'all_tasks', lambda suites, task_ids: [('libero_goal', 1)])
    monkeypatch.setattr(runner, 'plan_path', lambda suite, task: None)
    loads, executions = [], []
    def new_session(args):
        session = SimpleNamespace(args=args, env=None)
        def load(suite, task, cameras):
            loads.append((suite, task))
            session.env = SimpleNamespace(close=Mock())
        session.load = load
        return session
    monkeypatch.setattr(waypoint_gmm_editor, 'GMMSession', new_session)
    def execute(session, chain, spec, args, paired, entry):
        assert chain is None
        executions.append(spec)
        entry.update(status='normal_policy_failed', initial_state_sha256='fake-state')
    monkeypatch.setattr(runner, 'execute', execute)
    args = SimpleNamespace(task_suites=['libero_goal'], task_ids=None, seed=1000,
        handoff_delays=[0., 5.], stable_seconds=4., stability_metres=.01, stability_degrees=5.,
        photo_every=.5, limit=2, dry_run=False, port=8000, output=tmp_path,
        max_policy_steps=400, max_waypoint_seconds=180., resume=False)
    runner.run_images(args, argparse.ArgumentParser())
    with (tmp_path/'timing.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row['status'] for row in rows] == ['missing_plan', 'normal_policy_failed']
    assert len(loads) == len(executions) == 1
    args.resume = True
    runner.run_images(args, argparse.ArgumentParser())
    with (tmp_path/'timing.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 2
    assert len(loads) == len(executions) == 1
    assert {p.name for p in tmp_path.iterdir()} == {'timing.csv', 'trials'}
    assert {p.name for p in (tmp_path/'trials').iterdir()} == {'libero_goal'}
    assert all((tmp_path/r['trial_dir']).is_dir() for r in rows)


@pytest.mark.parametrize('metadata', [
    {},
    dict(model_id='RLinf/RLinf-Pi05-PPO-LIBERO-130', action_horizon=10,
         num_steps=5, discrete_state_input=False),
    dict(pi05_metadata(), rl_finetuned=True),
])
def test_main_rejects_rl_before_creating_output(tmp_path, monkeypatch, metadata):
    import waypoint_demo
    conn = Mock()
    conn.connect.return_value = metadata
    monkeypatch.setattr(waypoint_demo, 'PolicyConnection', lambda port: conn)
    monkeypatch.setattr(runner, 'all_tasks', lambda suites, task_ids: [('libero_goal', 1)])
    monkeypatch.setattr(runner, 'plan_path', lambda suite, task: None)
    args = SimpleNamespace(task_suites=['libero_goal'], task_ids=None, seed=1000,
        handoff_delays=[0., 5.], stable_seconds=4., stability_metres=.01, stability_degrees=5.,
        photo_every=.5, limit=2, dry_run=False, port=8000, output=tmp_path/'run', workers=5,
        max_policy_steps=400, max_waypoint_seconds=180., resume=False)
    with pytest.raises(ValueError, match='Expected standard Pi 0.5'):
        runner.run_images(args, argparse.ArgumentParser())
    conn.close.assert_called_once()
    assert not args.output.exists()


def test_pi05_results_cannot_resume_rl_results(tmp_path):
    config = dict(version=1, policy=dict(model_id='RLinf/RLinf-Pi05-PPO-LIBERO-130'),
                  code={'waypoint_recovery_images.py': 'same-code'})
    digest, compatible = runner.configuration_digests(config)
    results = runner.CsvResults(tmp_path, digest, False, compatible)
    results.append(dict(trial_id='rl-trial', configuration_sha256=digest,
                        success=True, status='normal_policy_success'))
    results.close()
    original = (tmp_path/'timing.csv').read_bytes()
    digest, compatible = runner.configuration_digests(dict(config, policy=pi05_metadata()))
    with pytest.raises(ValueError, match='same options'):
        runner.CsvResults(tmp_path, digest, True, compatible)
    assert (tmp_path/'timing.csv').read_bytes() == original


def test_images_cli_defaults_to_standard_pi05_port(monkeypatch):
    import sys
    captured = []
    monkeypatch.setattr(sys, 'argv', ['waypoint_recovery_benchmark.py', '--images-only'])
    monkeypatch.setattr(runner, 'run_images', lambda args, parser: captured.append(args))
    runner.bench.main()
    assert captured[0].port == 8000
    assert captured[0].workers == 5


@pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='Requires a real or virtual display')
@pytest.mark.parametrize('intensity', ['low', 'high'])
def test_real_simulation_handoffs_have_only_csv_and_controller_pngs(tmp_path, monkeypatch, intensity):
    """Exercise actual waypoint motion, physical failure and controller switching."""
    import waypoint_policy
    import waypoint_gmm_editor as editor
    from waypoint_gmm import GMMChain
    conn = Mock()
    conn.connect.return_value = pi05_metadata()
    actions = np.zeros((10, 7))
    actions[:, -1] = -1.
    conn.infer.return_value = dict(actions=actions)
    monkeypatch.setattr(waypoint_policy, 'PolicyConnection', lambda port: conn)
    monkeypatch.setattr(editor, 'MODEL_STORE', tmp_path/'unexpected_archives')
    args = SimpleNamespace(output=tmp_path, photo_every=1., max_policy_steps=2,
        max_waypoint_seconds=180., stable_seconds=4., stability_metres=.01, stability_degrees=5., port=8000)
    results = runner.CsvResults(tmp_path, 'config', False)
    session = editor.GMMSession(runner.bench.sp.Args(task_suite_name='libero_spatial', task_id=0))
    session.save_replay_archive = False
    paired = runner.PairedScenes()
    entries = []
    try:
        session.load('libero_spatial', 0, cameras=True)
        chain = GMMChain.load(runner.plan_path('libero_spatial', 0), 'libero_spatial', 0)
        specs = [s for s in runner.schedule([('libero_spatial', 0)]) if s['scene'] == 'original'
                 and s['kind'] == 'grasp_loss' and s['intensity'] == intensity]
        for spec in specs:
            spec['injection_row'] = runner.eligible_row(chain, spec)
            entry = dict(spec, trial_id=runner.identifier(spec), success=False, configuration_sha256='config')
            runner.execute(session, chain, spec, args, paired, entry)
            results.append(entry)
            entries.append(entry)
            assert entry['observed_failure']
            assert entry['failure_measurement']['primitive'] == ('transport' if intensity == 'high' else 'grasp')
            assert entry['actual_delay_seconds'] == pytest.approx(spec['requested_delay_seconds'], abs=.051)
            assert entry['status'] == 'recovery_failed'
            assert not entry['vla_recovered']
        assert entries[0]['initial_state_sha256'] == entries[1]['initial_state_sha256']
        # Continuing waypoints can already satisfy the goal before the delayed
        # handoff. Those trials hold for stability instead of requesting actions.
        assert conn.infer.call_count == sum(e['vla_action_steps'] > 0 for e in entries)
        assert {p.name for p in tmp_path.iterdir()} == {'timing.csv', 'trials'}
        for entry in entries:
            folder = tmp_path/runner.trial_path(entry)
            assert {p.name for p in folder.iterdir()} == {'photos', 'initial.png', 'failure.png', 'handoff.png', 'terminal.png'}
            assert all(p.suffix == '.png' for p in folder.rglob('*') if p.is_file())
    finally:
        results.close()
        if session.env is not None:
            session.env.close()
