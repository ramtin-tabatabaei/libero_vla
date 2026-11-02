"""Parallel trials keep pairs together and retain completed single-worker CSV rows."""
import csv
import hashlib
import json
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import waypoint_recovery_images as runner
from waypoint_recovery_workers import assign_workers, parallel_entries


def test_five_workers_own_entire_tasks_without_duplicates():
    specs = runner.schedule([('libero_spatial', task) for task in range(10)])
    buckets = assign_workers(specs, 5)
    assert len(buckets) == 5
    assert [len(bucket) for bucket in buckets] == [72]*5
    owners = {}
    for worker, bucket in enumerate(buckets, 1):
        for spec in bucket:
            key = spec['suite'], spec['task_id']
            assert owners.setdefault(key, worker) == worker
    assert len({runner.identifier(spec) for bucket in buckets for spec in bucket}) == len(specs)
    assert assign_workers([], 5) == []


def test_resume_migrates_only_known_compatible_rows_and_preserves_their_values(tmp_path):
    config = dict(version=1, seed=1000, code={
        'waypoint_recovery_images.py': 'new-images-code',
        'waypoint_policy.py': 'same-controller-code',
        'waypoint_recovery_workers.py': 'new-workers-code'})
    current, compatible = runner.configuration_digests(config)
    old = dict(config, code={'waypoint_recovery_images.py': runner.PRE_WORKERS_IMAGES_SHA256,
                            'waypoint_policy.py': 'same-controller-code'})
    old_digest = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    assert compatible == (old_digest,)
    path = tmp_path/'timing.csv'
    original = [dict(trial_id='completed_%d' % index, configuration_sha256=old_digest,
                     suite='libero_spatial', task_id='0', scene='original', mode='normal_policy',
                     status='normal_policy_success', success='True', policy_steps='154',
                     first_goal_sim_time='7.25000000001', initial_state_sha256='same-state')
                for index in range(2)]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.PRE_WORKERS_FIELDS)
        writer.writeheader()
        writer.writerows(original)
    unchanged = path.read_bytes()
    with pytest.raises(ValueError, match='same options'):
        runner.CsvResults(tmp_path, current, True)
    assert path.read_bytes() == unchanged
    results = runner.CsvResults(tmp_path, current, True, compatible)
    assert results.completed == {'completed_0', 'completed_1'}
    results.append(dict(trial_id='new_parallel_trial', configuration_sha256=current,
                        worker=5, worker_pid=12345, status='recovered', success=True))
    results.close()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for before, after in zip(original, rows):
        assert all(after[key] == value for key, value in before.items())
        assert after['worker'] == '1'
    assert rows[2]['worker'] == '5'
    resumed = runner.CsvResults(tmp_path, current, True, compatible)
    assert len(resumed.completed) == 3
    resumed.close()
    assert {p.name for p in tmp_path.iterdir()} == {'timing.csv', 'trials'}
    changed = dict(config, seed=999)
    changed_current, changed_compatible = runner.configuration_digests(changed)
    with pytest.raises(ValueError, match='same options'):
        runner.CsvResults(tmp_path, changed_current, True, changed_compatible)


@pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='Requires a real or virtual display')
def test_five_real_simulations_share_a_mock_pi05_server_and_one_csv(tmp_path, capfd):
    """Exercise spawned processes and live OpenPI requests without using the user's server."""
    from openpi_client import msgpack_numpy
    from websockets.sync.server import serve
    from serve_pi05_libero import metadata as pi05_metadata
    requests = []
    connections = []
    lock = threading.Lock()
    metadata = pi05_metadata()
    def handler(connection):
        with lock:
            connections.append(connection.remote_address)
        connection.send(msgpack_numpy.packb(metadata))
        for message in connection:
            obs = msgpack_numpy.unpackb(message)
            with lock:
                requests.append(obs['prompt'])
            actions = np.zeros((10, 7))
            actions[:, -1] = -1.
            connection.send(msgpack_numpy.packb(dict(actions=actions)))
    specs = [spec for spec in runner.schedule([('libero_spatial', task) for task in range(5)])
             if spec['scene'] == 'original' and spec['mode'] == 'normal_policy']
    paths = {(spec['suite'], spec['task_id']): runner.plan_path(spec['suite'], spec['task_id'])
             for spec in specs}
    hashes = {str(key): hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    results = runner.CsvResults(tmp_path, 'smoke-test', False)
    starts = []
    with serve(handler, '127.0.0.1', 0, compression=None, max_size=None) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        args = SimpleNamespace(output=tmp_path, photo_every=1., max_policy_steps=2,
            max_waypoint_seconds=180., stable_seconds=4., stability_metres=.01,
            stability_degrees=5., port=server.socket.getsockname()[1], workers=5)
        try:
            entries = []
            for entry in parallel_entries(args, specs, paths, hashes, 'smoke-test', set(), {},
                                           on_start=lambda worker, trial: starts.append((worker, trial))):
                results.append(entry)
                entries.append(entry)
                with (tmp_path/'timing.csv').open() as stream:
                    assert len(list(csv.DictReader(stream))) == len(entries)
        finally:
            results.close()
            server.shutdown()
            thread.join(timeout=2.)
    assert len(entries) == len(starts) == len(connections) == 5
    # A goal already true at takeover holds for stability without inference.
    assert len(requests) == sum(entry['vla_action_steps'] > 0 for entry in entries)
    assert {entry['worker'] for entry in entries} == {1, 2, 3, 4, 5}
    assert len({entry['worker_pid'] for entry in entries}) == 5
    assert all(entry['status'] == 'normal_policy_failed' for entry in entries)
    with (tmp_path/'timing.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 5
    assert {p.name for p in tmp_path.iterdir()} == {'timing.csv', 'trials'}
    assert {p.name for p in (tmp_path/'trials').iterdir()} == {'libero_spatial'}
    assert {p.name for p in (tmp_path/'trials'/'libero_spatial').iterdir()} == {'task_%03d' % i for i in range(5)}
    for entry in entries:
        folder = tmp_path/runner.trial_path(entry)
        assert {p.name for p in (folder/'photos').iterdir()} == {'waypoints', 'policy'}
        assert list((folder/'photos'/'policy').glob('*.png'))
        assert {p.name for p in folder.iterdir()} == {'initial.png', 'terminal.png', 'photos'}
    output = capfd.readouterr()
    assert 'RLinf VLA' not in output.out + output.err
    assert 'step 0/' not in output.out + output.err
