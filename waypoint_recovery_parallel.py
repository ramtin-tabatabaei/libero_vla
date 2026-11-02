"""Rerun paired recovery trials with one simulation process per task."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from waypoint_recovery_benchmark import ROOT, execute_trial, summarize, trial_id, write_json


def task_batches(schedule):
    """Assign both layouts of a task to the same worker to prevent state races."""
    batches = {}
    for spec in schedule:
        batches.setdefault((spec['suite'], spec['task_id']), []).append(spec)
    return list(batches.values())


def collect_results(output, schedule):
    results = []
    for spec in schedule:
        path = output/'trials'/trial_id(spec)/'result.json'
        if path.exists():
            results.append(json.loads(path.read_text()))
    return results


def run_task(worker, schedule, configuration, output):
    from waypoint_gmm import GMMChain
    import waypoint_gmm_editor as editor
    import subgoal_planner as sp
    from waypoint_recovery_retry import alternative_rows

    output = Path(output)
    options = SimpleNamespace(port=configuration['port'], delay=configuration['delay_seconds'],
        max_policy_steps=configuration['max_policy_steps'],
        max_waypoint_seconds=configuration['max_waypoint_seconds'],
        no_video=not configuration['save_video'], photo_every=configuration['photo_interval_sim_seconds'],
        layout_root=output/'layouts')
    editor.MODEL_STORE = output/'waypoint_archives'
    progress = output/'workers'/('worker_%02d.json'%worker)
    done = 0
    with (output/'workers'/('worker_%02d.log'%worker)).open('a', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            session = editor.GMMSession(sp.Args(policy_port=options.port,
                                               policy_max_steps=options.max_policy_steps))
            try:
                key = schedule[0]['suite'], schedule[0]['task_id']
                session.load(*key, cameras=True)
                chain = GMMChain.load(ROOT/'saved_plans'/key[0]/('task_%03d.json'%key[1]), *key)
                for spec in schedule:
                    identifier = trial_id(spec)
                    folder = output/'trials'/identifier
                    if (folder/'result.json').exists():
                        done += 1
                        continue
                    # A resumed incomplete trial must not append to old photo indexes.
                    if folder.exists():
                        shutil.rmtree(folder)
                    write_json(progress, dict(worker=worker, pid=os.getpid(), task=list(key),
                        completed=done, planned=len(schedule), current_trial=identifier, status='running'))
                    started = time.monotonic()
                    print('START', identifier, flush=True)
                    try:
                        entry = execute_trial(session, chain, spec, options, folder)
                        if entry['status'] == 'failure_not_observed':
                            attempts = [dict(injection_row=entry['injection_row'],
                                             status=entry['status'], path='trials/'+identifier)]
                            for row in alternative_rows(chain, entry):
                                relative = Path('attempts')/identifier/('attempt_%02d'%(len(attempts)+1))
                                attempt_folder = output/relative
                                if attempt_folder.exists():
                                    shutil.rmtree(attempt_folder)
                                print('RETRY', identifier, 'row', row, flush=True)
                                attempt = execute_trial(session, chain, dict(spec, injection_row=row),
                                                        options, attempt_folder)
                                write_json(attempt_folder/'result.json', attempt)
                                old = json.loads((folder/'initial.state.json').read_text())
                                new = json.loads((attempt_folder/'initial.state.json').read_text())
                                if old != new:
                                    raise ValueError('Retry changed the paired initial state')
                                attempts.append(dict(injection_row=row, status=attempt['status'], path=str(relative)))
                                if attempt.get('observed_failure'):
                                    backup = output/'attempts'/identifier/'attempt_01'
                                    backup.parent.mkdir(parents=True, exist_ok=True)
                                    if backup.exists():
                                        shutil.rmtree(backup)
                                    folder.rename(backup)
                                    attempts[0]['path'] = str(backup.relative_to(output))
                                    attempt_folder.rename(folder)
                                    attempts[-1]['path'] = 'trials/'+identifier
                                    entry = attempt
                                    entry['selected_attempt'] = len(attempts)
                                    break
                            entry['attempts'] = attempts
                        initial = json.loads((folder/'initial.state.json').read_text())['state']
                        layout_id = '{suite}_t{task_id:03d}_{scene}'.format(**spec)
                        layout = json.loads((output/'layouts'/(layout_id+'.json')).read_text())['state']
                        entry.update(layout_id=layout_id,
                            initial_state_sha256=hashlib.sha256(json.dumps(initial).encode()).hexdigest(),
                            initial_state_matches_layout=initial == layout)
                        if initial != layout:
                            raise ValueError('Initial simulation state differs from the paired layout')
                    except Exception as exc:
                        import traceback
                        traceback.print_exc()
                        entry = dict(spec, trial_id=identifier, status='error', success=False,
                                     error=type(exc).__name__+': '+str(exc))
                    entry.update(worker=worker, worker_pid=os.getpid(),
                                 wall_seconds=round(time.monotonic()-started, 3))
                    write_json(folder/'result.json', entry)
                    done += 1
                    print('RESULT', identifier, entry['status'], 'seconds', entry['wall_seconds'], flush=True)
                    write_json(progress, dict(worker=worker, pid=os.getpid(), task=list(key),
                        completed=done, planned=len(schedule), current_trial=identifier, status='running'))
            finally:
                if session.env is not None:
                    session.env.close()
            write_json(progress, dict(worker=worker, pid=os.getpid(), task=list(key),
                completed=done, planned=len(schedule), current_trial=None, status='complete'))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'experiments'/'pi05_failure_recovery_timed_paired_20261001')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--photo-every', type=float, default=.25,
                        help='Save a camera PNG every N simulation seconds, plus each phase endpoint')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.photo_every < float('inf'):
        parser.error('Worker count and photo interval must be positive and finite')
    source = args.source.resolve()
    output = args.output.resolve()
    if output == source:
        parser.error('Rerun output must differ from the earlier experiment')
    configuration = json.loads((source/'manifest.json').read_text())
    for plan in configuration['tasks']:
        if hashlib.sha256(Path(plan['plan']).read_bytes()).hexdigest() != plan['sha256']:
            raise ValueError('A saved plan changed since the original experiment: '+plan['plan'])
    from waypoint_demo import PolicyConnection
    from waypoint_recovery_metrics import export_timing_csv
    client = PolicyConnection(configuration['port'])
    try:
        metadata = client.connect()
    finally:
        client.close()
    if metadata != configuration['policy']:
        raise ValueError('The policy checkpoint or configuration differs from the original experiment')
    schedule = configuration['scenarios']
    batches = task_batches(schedule)
    configuration.update(version=4, workers=args.workers, concurrency='One process per task; shared policy server',
        photo_interval_sim_seconds=args.photo_every, photo_camera='agentview',
        photo_scope='Whole waypoint and policy episodes, plus initial, failure, handoff and terminal snapshots',
        rerun_source=str(source), retry_unobserved_faults=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'manifest.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != configuration:
            parser.error('Existing output requires --resume with the exact same configuration')
    else:
        write_json(manifest, configuration)
    (output/'workers').mkdir(exist_ok=True)
    (output/'layouts').mkdir(exist_ok=True)
    for path in (source/'layouts').glob('*.json'):
        target = output/'layouts'/path.name
        if target.exists() and target.read_bytes() != path.read_bytes():
            raise ValueError('Rerun layout differs from the saved original: '+path.name)
        shutil.copy2(path, target)
    count = -1
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(run_task, index, batch, configuration, str(output))
                   for index, batch in enumerate(batches, 1)}
        while pending:
            finished, pending = wait(pending, timeout=1)
            for future in finished:
                future.result()
            results = collect_results(output, schedule)
            if len(results) != count:
                count = len(results)
                write_json(output/'results.json', results)
                write_json(output/'summary.json', summarize(results))
                export_timing_csv(output, results)
                print('PROGRESS', count, '/', len(schedule), json.dumps(summarize(results)['statuses']), flush=True)
    if count != len(schedule):
        raise RuntimeError('Workers finished without producing every planned condition')
    from waypoint_recovery_metrics import recompute_metrics
    from waypoint_recovery_report import generate
    recompute_metrics(output)
    generate(output)
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
