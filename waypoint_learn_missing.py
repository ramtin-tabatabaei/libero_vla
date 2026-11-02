"""Collect VLA demonstrations and fit GUI plans for tasks without learned models."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def write_json(path, value, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    options = dict(separators=(',', ':')) if compact else dict(indent=2)
    temporary.write_text(json.dumps(value, allow_nan=False, **options)+'\n')
    temporary.replace(path)


def learned_tasks(root, suite):
    """The GUI's nonempty learned plans or successfully fitted model files."""
    root = Path(root)
    learned = set()
    for path in (root/'task_waypoints_gmm'/suite).glob('task_*.json'):
        if not path.stem[5:].isdigit():
            continue
        try:
            data = json.loads(path.read_text())
            if (data.get('suite') == suite and data.get('waypoints')
                    and data.get('kind') == 'gmm_editor_chain'):
                learned.add(data['task_id'])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for path in (root/'waypoint_models').rglob('model.json'):
        try:
            data = json.loads(path.read_text())
            if (data.get('suite') == suite and data.get('waypoints')
                    and len(data.get('used_demonstrations', [])) >= 2):
                learned.add(data['task_id'])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return learned


def install_plan(chain, path):
    """A concurrent GUI edit must take precedence over batch installation."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('waypoints'):
            return False
    chain.save(path)
    return True


def learn_task(task_id, configuration):
    import copy
    import traceback
    import subgoal_planner as sp
    from waypoint_demo import record
    from waypoint_gmm import chain_from_model
    from waypoint_gmm_editor import GMMSession
    from waypoint_learning import Config, save_model
    from waypoint_skill_editor import can_fit_editor_model, fit_editor_model

    suite = configuration['suite']
    output = Path(configuration['output'])
    folder = output/'tasks'/('task_%03d'%task_id)
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder/'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    progress_path = folder/'progress.json'
    model_folder = (ROOT/'waypoint_models'/'gui'/suite/('task_%03d'%task_id)
                    /configuration['run_id'])
    archive = model_folder/'demonstrations'
    archive.mkdir(parents=True, exist_ok=True)
    session = None
    demos, attempts = [], []
    started = time.monotonic()
    base = dict(task_id=task_id, gui_task=task_id+1, suite=suite, pid=os.getpid())
    def progress(**values):
        write_json(progress_path, dict(base, successes=len(demos), attempts=len(attempts),
                                     wall_seconds=round(time.monotonic()-started, 2), **values))
    result = dict(base, status='error', model_folder=str(model_folder))
    with (folder/'worker.log').open('a', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                # Recheck before collecting: the user may have just learned it.
                if task_id in learned_tasks(ROOT, suite):
                    result.update(status='skipped_already_learned')
                    return result
                session = GMMSession(sp.Args(policy_port=configuration['port'], seed=configuration['seed']))
                chain = session.load(suite, task_id, cameras=True)
                result['instruction'] = chain.language
                progress(status='recording', attempt=0, step=0)
                for attempt in range(configuration['max_attempts']):
                    session.restore_training_scene()
                    last_progress = [0.]
                    def recording_progress(step, total):
                        now = time.monotonic()
                        if now-last_progress[0] >= 5:
                            progress(status='recording', attempt=attempt+1, step=step,
                                     max_steps=total, instruction=chain.language)
                            last_progress[0] = now
                    path = archive/('demo_%03d.json'%attempt)
                    if path.exists() or (archive/'failed'/path.name).exists():
                        # Resume only whole recordings; never append to or erase evidence.
                        actual = path if path.exists() else archive/'failed'/path.name
                        demo = json.loads(actual.read_text())
                        usable = (demo.get('success') and len(demo.get('samples', [])) >= 3
                                  and not demo['samples'][0].get('task_success', False))
                    else:
                        try:
                            record(session, chain.language, configuration['port'],
                                   max_steps=configuration['max_steps'], progress=recording_progress)
                            demo = copy.deepcopy(session.last_demonstration)
                            usable = (demo['success'] and len(demo['samples']) >= 3
                                      and not demo['samples'][0].get('task_success', False))
                            actual = path if usable else archive/'failed'/path.name
                            state_path = actual.with_suffix('.state.json')
                            write_json(state_path, dict(initial_state=demo.pop('initial_state')), compact=True)
                            demo.update(replay_state_file=state_path.name, placement_mode='fixed_unshuffled',
                                        seed=configuration['seed'])
                            write_json(actual, demo, compact=True)
                        except Exception as exc:
                            traceback.print_exc()
                            attempts.append(dict(attempt=attempt+1, usable=False, error=str(exc)))
                            write_json(model_folder/'recording_report.json', dict(successes=len(demos), attempts=attempts))
                            progress(status='recording', attempt=attempt+1, step=0, last_error=str(exc))
                            continue
                    demo['_path'] = str(actual.resolve())
                    attempts.append(dict(attempt=attempt+1, success=bool(demo.get('success')), usable=bool(usable),
                                         steps=len(demo.get('samples', [])), file=str(actual)))
                    if usable:
                        demos.append(demo)
                    print('DEMO', task_id+1, attempt+1, 'usable', usable, 'successes', len(demos), flush=True)
                    write_json(model_folder/'recording_report.json', dict(successes=len(demos), attempts=attempts))
                    progress(status='recording', attempt=attempt+1, step=0)
                    if len(demos) >= configuration['successes'] and can_fit_editor_model(demos, model_folder.parent):
                        break
                if len(demos) < configuration['successes']:
                    result.update(status='insufficient_successes', error='Successful demonstration target not reached')
                else:
                    progress(status='fitting', step=0)
                    model = fit_editor_model(demos, Config(max_waypoints=100), reuse_recordings_from=model_folder.parent)
                    save_model(model, model_folder)
                    learned = chain_from_model(model, model_folder/'model.json')
                    session.preview(learned)
                    plan_path = ROOT/'task_waypoints_gmm'/suite/('task_%03d.json'%task_id)
                    installed = install_plan(learned, plan_path)
                    result.update(status='learned', plan_installed=installed, plan_path=str(plan_path),
                                  model_path=str(model_folder/'model.json'), waypoints=len(learned.waypoints),
                                  used_demonstrations=len(model['used_demonstrations']), replay_verified=False)
            except Exception as exc:
                traceback.print_exc()
                result.update(status='error', error=type(exc).__name__+': '+str(exc))
            finally:
                if session is not None and session.env is not None:
                    session.env.close()
                result.update(successes=len(demos), attempts=len(attempts),
                              wall_seconds=round(time.monotonic()-started, 2))
                write_json(result_path, result)
                progress(status=result['status'], step=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='libero_90')
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--successes', type=int, default=5)
    parser.add_argument('--max-attempts', type=int, default=30)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.successes < 2 or args.max_attempts < args.successes or args.max_steps < 3:
        parser.error('Require positive workers, at least 2 successes, enough attempts and at least 3 steps')
    output = args.output.resolve()
    manifest_path = output/'manifest.json'
    if args.resume:
        configuration = json.loads(manifest_path.read_text())
    else:
        if manifest_path.exists():
            parser.error('Output already contains a batch; use --resume')
        from libero.libero import benchmark
        count = benchmark.get_benchmark_dict()[args.suite]().n_tasks
        learned = learned_tasks(ROOT, args.suite)
        pending = [task_id for task_id in range(count) if task_id not in learned]
        configuration = dict(suite=args.suite, workers=args.workers, port=args.port,
            successes=args.successes, max_attempts=args.max_attempts, max_steps=args.max_steps,
            seed=args.seed, output=str(output), task_ids=pending, skipped_task_ids=sorted(learned),
            run_id=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_missing_batch')
        write_json(manifest_path, configuration)
    results = []
    def summary(status):
        counts = {}
        for entry in results:
            counts[entry['status']] = counts.get(entry['status'], 0)+1
        write_json(output/'summary.json', dict(status=status, planned=len(configuration['task_ids']),
            completed=len(results), counts=counts, skipped_task_ids=configuration['skipped_task_ids'], results=results))
    summary('running')
    print('START', len(configuration['task_ids']), 'missing tasks;', configuration['workers'], 'workers;', output, flush=True)
    with ProcessPoolExecutor(max_workers=configuration['workers'],
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(learn_task, task_id, configuration): task_id for task_id in configuration['task_ids']}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(task_id=task_id, gui_task=task_id+1, suite=configuration['suite'],
                              status='error', error=type(exc).__name__+': '+str(exc))
                write_json(output/'tasks'/('task_%03d'%task_id)/'result.json', result)
            results.append(result)
            summary('running')
            print('RESULT', result['gui_task'], result['status'],
                  result.get('successes', 0), 'successes;', len(results), '/', len(futures), flush=True)
    summary('complete')
    print('COMPLETE', output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
