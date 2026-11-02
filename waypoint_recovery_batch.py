"""Collect the current live GUI's recovery conditions, with durable resume.

Run through run_recovery_batch.sh. Each task runs in its own simulator process;
the existing Pi 0.5 server is shared. No GUI clicks are needed. Playback pacing
does not change controller gains, physics, fault duration or detection latency.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent
SUITES = {'libero_goal': 10, 'libero_object': 10, 'libero_spatial': 10,
          'libero_10': 10, 'libero_90': 90}
FAILURES = ('collision', 'transition', 'orientation', 'grasp_loss')
RESERVE_BYTES = 512 * 1024**2
SCENE_SEED_ATTEMPTS = 8
SOURCE_FILES = ('waypoint_recovery_batch.py', 'waypoint_recovery_live.py',
                'waypoint_recovery_benchmark.py', 'waypoint_failures.py',
                'waypoint_skill.py', 'waypoint_session.py', 'waypoint_gmm_editor.py',
                'waypoint_gmm.py', 'waypoint_chain.py', 'subgoal_planner.py',
                'waypoint_policy.py', 'waypoint_demo.py', 'serve_pi05_libero.py',
                'waypoint_replay.py', 'waypoint_rrt.py', 'waypoint_skill_editor.py',
                'waypoint_recovery_gui.py', 'run_recovery_batch.sh',
                'compare_recovery_runs.py', 'recovery_frame_images.py')
CSV_FIELDS = ('trial_id', 'suite', 'task_id', 'language', 'scene', 'kind', 'intensity',
              'detection_latency', 'latency_seconds', 'repeat', 'layout_seed', 'scene_seed', 'fault_seed', 'injection_row',
              'status', 'observed_failure', 'fault_execution_completed',
              'success_before_policy', 'success', 'recovery_evaluable', 'vla_recovered',
              'waypoints_sim_seconds', 'vla_sim_seconds', 'wall_seconds', 'policy_steps',
              'actual_detection_latency_seconds', 'error', 'result_file',
              'waypoint_images', 'vla_images', 'image_index')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, default=json_default, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_seed(seed, suite, task_id, scene, kind, intensity, repeat):
    # Excluding latency makes off/on a matched random-fault pair.
    key = '%s/%s/%s/%s/%s/%s/%s' % (seed, suite, task_id, scene, kind, intensity, repeat)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def build_conditions(plan_root, seed=1000, repeats=1, suites=None, latency_mode='end_of_plan'):
    from waypoint_gmm import GMMChain
    from waypoint_recovery_live import eligible_rows
    conditions, inputs = [], {}
    for suite, count in (suites or SUITES).items():
        for task_id in range(count):
            path = Path(plan_root) / suite / ('task_%03d.json' % task_id)
            reason, chain = '', None
            if not path.is_file():
                reason = 'missing_plan'
            else:
                inputs[str(path.resolve())] = file_hash(path)
                try:
                    chain = GMMChain.load(path, suite, task_id)
                    if chain.model_path:
                        model = (ROOT / chain.model_path).resolve()
                        inputs[str(model)] = file_hash(model)
                        for demo in json.loads(model.read_text()).get('demonstration_files', [])[:1]:
                            demo_path = (ROOT / demo).resolve()
                            inputs[str(demo_path)] = file_hash(demo_path)
                except Exception as exc:
                    reason = 'invalid_plan: %s' % exc
            for scene, kind, intensity, repeat, delayed in itertools.product(
                    ('original', 'shuffled'), FAILURES, ('low', 'high'), range(repeats), (False, True)):
                detection = latency_mode if delayed else 'off'
                latency = (None if delayed and latency_mode == 'end_of_plan' else 10. if delayed else 0.)
                rows = eligible_rows(chain, kind, intensity) if chain is not None and not reason else []
                unavailable = reason or ('' if rows else 'no_applicable_waypoint')
                label = 'End' if detection == 'end_of_plan' else '%02d' % latency
                trial_id = '%s_t%03d_%s_%s_%s_latency%s_r%02d' % (
                    suite, task_id, scene, kind, intensity, label, repeat)
                conditions.append(dict(trial_id=trial_id, suite=suite, task_id=task_id,
                    language=chain.language if chain is not None else '', scene=scene,
                    kind=kind, intensity=intensity, detection_latency=detection,
                    latency_seconds=latency, repeat=repeat,
                    layout_seed=(seed + repeat) % 2**32,
                    fault_seed=paired_seed(seed, suite, task_id, scene, kind, intensity, repeat),
                    injection_row=rows[0] if rows else None, unavailable=unavailable))
    return conditions, inputs


def check_policy(port, backend='pi05'):
    from waypoint_demo import PolicyConnection
    from serve_pi05_libero import metadata
    if backend != 'pi05':
        raise ValueError('Unsupported batch policy: ' + backend)
    client = PolicyConnection(port)
    try:
        actual = client.connect()
        for key, value in metadata().items():
            if actual.get(key) != value:
                raise RuntimeError('%s identity mismatch: %s' % (backend, key))
        return actual
    finally:
        client.close()


def classify(condition, result):
    """Never attribute an absent fault or an already met goal to VLA recovery."""
    policy = result.get('policy_result') or {}
    observed = bool(result.get('observed_failure'))
    evaluable = (not condition.get('unavailable') and observed
                 and not result.get('success_before_policy', False)
                 and 'success' in policy and policy.get('steps', 0) > 0
                 and not result.get('error') and not result.get('scene_unavailable'))
    if condition.get('unavailable'):
        status = condition['unavailable'].split(':', 1)[0]
    elif result.get('scene_unavailable'):
        status = 'scene_unavailable'
    elif result.get('error'):
        status = 'error'
    elif not observed:
        status = 'failure_not_observed'
    elif result.get('success_before_policy'):
        status = 'already_successful_at_handoff'
    elif not evaluable:
        status = 'policy_not_evaluated'
    else:
        status = 'recovered' if policy['success'] else 'recovery_failed'
    return dict(status=status, recovery_evaluable=bool(evaluable),
                vla_recovered=bool(evaluable and policy['success']))


def result_path(output, condition):
    return Path(output) / 'trials' / (condition['trial_id'] + '.json')


def storage_available(output, config):
    paths = [Path(output)]
    if config.get('all_images'):
        images = Path(config['image_root'])
        images.mkdir(parents=True, exist_ok=True)
        paths.append(images)
    return all(shutil.disk_usage(path).free >= RESERVE_BYTES for path in paths)


def publish_tables(output, manifest, state, **details):
    """Rebuild from committed per-trial JSON, so resume cannot duplicate CSV rows."""
    output = Path(output)
    rows, groups, statuses = [], defaultdict(Counter), Counter()
    for condition in manifest['conditions']:
        path = result_path(output, condition)
        if not path.exists():
            continue
        document = json.loads(path.read_text())
        result, evaluation = document['result'], document['evaluation']
        timing = result.get('controller_timings') or {}
        row = dict(condition, **evaluation)
        for field in ('observed_failure', 'fault_execution_completed', 'success_before_policy',
                      'success', 'wall_seconds', 'actual_detection_latency_seconds', 'error'):
            row[field] = result.get(field, '')
        row.update(scene_seed=document.get('options', {}).get('seed', ''),
                   waypoints_sim_seconds=timing.get('waypoints_sim_seconds', ''),
                   vla_sim_seconds=timing.get('vla_sim_seconds', ''),
                   policy_steps=(result.get('policy_result') or {}).get('steps', ''),
                   result_file=str(path.relative_to(output)))
        images = document.get('images') or {}
        for name in ('waypoint_images', 'vla_images', 'image_index'):
            row[name] = images.get(name, '')
        rows.append(row)
        statuses[evaluation['status']] += 1
        detection = condition.get('detection_latency', 'timed' if condition.get('latency_seconds') else 'off')
        key = (condition['scene'], condition['kind'], condition['intensity'], detection)
        groups[key]['recorded'] += 1
        groups[key]['evaluable'] += evaluation['recovery_evaluable']
        groups[key]['recovered'] += evaluation['vla_recovered']
    temp = output / 'results.csv.tmp'
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, CSV_FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(output / 'results.csv')
    summaries = []
    for key, counts in sorted(groups.items()):
        rate = counts['recovered']/counts['evaluable'] if counts['evaluable'] else None
        summaries.append(dict(zip(('scene', 'kind', 'intensity', 'detection_latency'), key),
            **counts, recovery_rate=rate, recovery_percent=100*rate if rate is not None else None))
    evaluated = sum(row['recovery_evaluable'] for row in rows)
    recovered = sum(row['vla_recovered'] for row in rows)
    overall = dict(evaluated=evaluated, recovered=recovered,
                   recovery_percent=100*recovered/evaluated if evaluated else None)
    columns = [(scene, intensity, delayed) for scene in ('original', 'shuffled')
               for intensity in ('low', 'high') for delayed in (False, True)]
    headers = ['failure_type'] + ['%s_%s_%s' % (s, i, 'delayed' if d else 'off') for s, i, d in columns]
    table = output / 'recovery_percentages.csv.tmp'
    with table.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for kind in FAILURES:
            cells = []
            for scene, intensity, delayed in columns:
                matches = [v for (s, k, i, d), v in groups.items()
                           if (s, k, i) == (scene, kind, intensity) and (d != 'off') == delayed]
                n = sum(v['evaluable'] for v in matches)
                r = sum(v['recovered'] for v in matches)
                cells.append('%.1f%% (%d/%d)' % (100*r/n, r, n) if n else 'not evaluated (0/0)')
            writer.writerow([kind] + cells)
    table.replace(output / 'recovery_percentages.csv')
    atomic_json(output / 'summary.json', dict(updated=utcnow(), status_counts=statuses, groups=summaries,
        overall=overall,
        denominator='Measured failure, unmet goal at handoff, completed policy evaluation with at least one action.'))
    atomic_json(output / 'progress.json', dict(updated=utcnow(), state=state, pid=os.getpid(),
        total=len(manifest['conditions']), runnable=manifest['runnable'],
        recorded=len(rows), runnable_recorded=sum(not c.get('unavailable') for c in rows),
        status_counts=statuses, overall=overall, **details))
    return len(rows)


class CompactRecorder:
    """Record actual simulator steps in either controller, without a reset."""
    def __init__(self, session, folder, speed, image_folder=None):
        self.session, self.folder, self.speed = session, Path(folder), speed
        self.phase = 0
        self.samples = []
        self.snapshots = {}
        self.original = None
        self.frame_images = None
        if image_folder is not None:
            from recovery_frame_images import FrameImages
            self.frame_images = FrameImages(session, self.folder, image_folder)

    def snapshot(self, name):
        state = self.session.env.sim.get_state().flatten().copy()
        self.snapshots[name] = dict(sim_time=float(self.session.env.sim.data.time), state=state,
            state_sha256=hashlib.sha256(np.asarray(state, dtype='<f8').tobytes()).hexdigest())
        from PIL import Image
        self.folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.session.image()).save(str(self.folder / (name + '.jpg')), quality=75)

    def __enter__(self):
        if self.frame_images is not None:
            self.frame_images.start()
        env = self.session.env
        self.had_instance = 'step' in env.__dict__
        self.original = env.step
        def step(action, *args, **kwargs):
            start = time.monotonic()
            out = self.original(action, *args, **kwargs)
            data = env.sim.data
            position, rotation = self.session.gripper_pose()
            a = np.full(8, np.nan, dtype=np.float32)
            action_array = np.asarray(action).reshape(-1)
            if len(action_array) > len(a):
                raise ValueError('Unexpected action size')
            a[:len(action_array)] = action_array
            self.samples.append((float(data.time), self.phase, a, data.qpos.copy(),
                data.qvel.copy(), np.r_[position, rotation.reshape(-1)], bool(env.check_success())))
            if self.frame_images is not None:
                self.frame_images.capture(self.phase, len(self.samples)-1)
            # The same pace applies to both joint-waypoint and OSC-policy steps.
            remaining = float(env.env.control_timestep) / self.speed - (time.monotonic() - start)
            if remaining > 0:
                time.sleep(remaining)
            return out
        env.step = step
        return self

    def __exit__(self, *exc):
        env = self.session.env
        if self.had_instance:
            env.step = self.original
        else:
            del env.step
        if self.frame_images is not None:
            self.frame_images.close(not exc or exc[0] is None)
        self.folder.mkdir(parents=True, exist_ok=True)
        if self.samples:
            arrays = list(zip(*self.samples))
            temp = self.folder / 'trajectory.tmp.npz'
            np.savez_compressed(str(temp), sim_time=np.asarray(arrays[0], dtype=np.float64),
                phase=np.asarray(arrays[1], dtype=np.uint8), action=np.asarray(arrays[2], dtype=np.float32),
                qpos=np.asarray(arrays[3], dtype=np.float32), qvel=np.asarray(arrays[4], dtype=np.float32),
                eef_pose=np.asarray(arrays[5], dtype=np.float32), success=np.asarray(arrays[6], dtype=bool))
            temp.replace(self.folder / 'trajectory.npz')


class SceneUnavailable(RuntimeError):
    pass


def layout_path(output, condition):
    return Path(output) / 'layouts' / ('%s_t%03d_%s_r%02d.json' % (
        condition['suite'], condition['task_id'], condition['scene'], condition['repeat']))


def prepare_trial(runner, output, condition, options):
    """Find one valid layout, then reuse its exact state for every paired trial."""
    path = layout_path(output, condition)
    if path.exists():
        stored = json.loads(path.read_text())
        actual = replace(options, seed=stored['seed'])
        key = (actual.suite, actual.task_id, actual.scene, actual.seed, actual.use_saved_layout)
        runner.layouts[key] = (np.asarray(stored['state']), stored.get('gripper', -1.),
                              stored.get('shuffle_mapping'), stored['scene_origin'])
        runner.prepare(actual)
        return actual, stored.get('seed_attempts', [])
    failed = path.with_suffix('.unavailable.json')
    if failed.exists():
        raise SceneUnavailable(json.loads(failed.read_text())['reason'])
    attempts = []
    for attempt in range(SCENE_SEED_ATTEMPTS if options.scene == 'shuffled' else 1):
        actual = replace(options, seed=(options.seed + attempt*10007) % 2**32)
        try:
            runner.prepare(actual)
            attempts.append(dict(seed=actual.seed, status='ready'))
            return actual, attempts
        except ValueError as exc:
            if options.scene != 'shuffled' or not str(exc).startswith((
                    'No different supported', 'No movable task objects to shuffle')):
                raise
            attempts.append(dict(seed=actual.seed, error=str(exc)))
    reason = 'No verified shuffled layout after %d seeds: %s' % (len(attempts), attempts[-1]['error'])
    atomic_json(failed, dict(reason=reason, seed_attempts=attempts, completed=utcnow()))
    raise SceneUnavailable(reason)


def trial_options(condition, config):
    from waypoint_recovery_live import LiveOptions
    from run_vla import MAX_STEPS
    return LiveOptions(condition['suite'], condition['task_id'], scene=condition['scene'],
        kind=condition['kind'], intensity=condition['intensity'], delay=condition['latency_seconds'] or 0.,
        handoff_at_end=condition.get('detection_latency') == 'end_of_plan', propagate_deviation=True,
        seed=condition['layout_seed'], fault_seed=condition['fault_seed'],
        injection_row=condition['injection_row'], use_saved_layout=False,
        backend=config.get('backend', 'pi05'),
        port=config['port'], max_waypoint_seconds=config['max_waypoint_seconds'],
        max_policy_steps=config['max_policy_steps'] or max(400, MAX_STEPS[condition['suite']]))


def run_one(runner, output, condition, config):
    session = runner.session
    options = trial_options(condition, config)
    folder = Path(output) / 'trials' / condition['trial_id']
    start, result = time.monotonic(), {}
    recorder = None
    def report(stage, message):
        atomic_json(Path(output) / 'worker_progress.json', dict(updated=utcnow(),
            trial_id=condition['trial_id'], stage=stage, message=message,
            started_monotonic=start, wall_seconds=time.monotonic()-start))
    report('Preparing', 'Restoring the paired scene')
    try:
        options, seed_attempts = prepare_trial(runner, output, condition, options)
        np.random.seed(options.seed)
        # set_init_state restores physics but leaves the preceding trial's
        # controller goal/interpolator alive. Rebuild it only at the initial
        # boundary, never at the waypoint-to-policy handoff.
        from waypoint_demo import switch_controller
        switch_controller(session.env, 'JOINT_POSITION')
        # Direction-history avoidance is useful in the GUI, but would change
        # the matched seed between latency off/on. Each trial starts fresh.
        runner.transition_history.clear()
        runner.collision_history.clear()
        session.realtime = False
        image_folder = (Path(config.get('image_root', Path(output)/'images')) / condition['trial_id']
                        if config.get('all_images') else None)
        recorder = CompactRecorder(session, folder, config['speed'], image_folder=image_folder)
        recorder.snapshot('initial')
        layout = layout_path(output, condition)
        if not layout.exists():
            atomic_json(layout, dict(seed=options.seed, scene_origin=runner.scene_origin,
                requested_seed=condition['layout_seed'], seed_attempts=seed_attempts,
                gripper=session.executor.gripper,
                shuffle_mapping=session.last_shuffle_mapping, **recorder.snapshots['initial']))
        def handoff(partial):
            result.update(partial)
            recorder.snapshot('handoff')
            recorder.phase = 1
        with recorder:
            result = runner.run(options, report=report, before_vla=handoff)
        recorder.snapshot('terminal')
    except SceneUnavailable as exc:
        result.update(scene_unavailable=True, scene_error=str(exc))
    except Exception as exc:
        result.update(error='%s: %s' % (type(exc).__name__, exc), traceback=traceback.format_exc(),
            controller_timings=runner.timings.snapshot(session.env.sim.data.time) if session.env else {})
        # The worker's simulator is discarded after an error, before another trial.
    result['wall_seconds'] = time.monotonic() - start
    document = dict(condition=condition, options=asdict(options), result=result,
        evaluation=classify(condition, result), completed=utcnow(),
        snapshots=recorder.snapshots if recorder else {}, scene_origin=runner.scene_origin,
        artifacts=str(folder.relative_to(output)))
    if recorder is not None and recorder.frame_images is not None:
        document['images'] = recorder.frame_images.summary()
    atomic_json(result_path(output, condition), document)
    print('%s %s (%.1fs wall)' % (condition['trial_id'], document['evaluation']['status'],
                                result['wall_seconds']), flush=True)
    return result


def worker(output, task, limit):
    from waypoint_recovery_live import LiveRunner
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    config = manifest['config']
    check_policy(config['port'], config.get('backend', 'pi05'))
    runner, done = LiveRunner(plan_root=output / 'plans'), 0
    try:
        for c in manifest['conditions']:
            if (c['suite'] + ':%d' % c['task_id'] != task or c['unavailable']
                    or result_path(output, c).exists()):
                continue
            if (output / 'STOP').exists() or (limit and done >= limit):
                break
            if not storage_available(output, config):
                return 3
            result = run_one(runner, output, c, config)
            done += 1
            if result.get('error'):
                return 1
    finally:
        runner.close()
    return 0


def reference_inventory(args):
    """Reuse archived plans, conditions and physical states from a completed run."""
    reference = args.reference_run.resolve()
    if args.output.resolve() == reference or reference in args.output.resolve().parents:
        raise ValueError('Use a separate output directory for the model comparison')
    manifest_path = reference / 'manifest.json'
    previous = json.loads(manifest_path.read_text())
    for key in ('seed', 'repeats', 'max_waypoint_seconds', 'max_policy_steps', 'latency_mode'):
        if previous['config'].get(key) != getattr(args, key):
            raise ValueError('Reference experiment setting differs: ' + key)
    suites = getattr(args, 'suites', None) or list(SUITES)
    conditions = [c for c in previous['conditions'] if c['suite'] in suites]
    if not conditions:
        raise ValueError('No matching conditions in the reference experiment')
    generated, inputs = build_conditions(reference / 'plans', args.seed, args.repeats,
        suites={name: SUITES[name] for name in SUITES if name in suites}, latency_mode=args.latency_mode)
    if generated != conditions:
        raise ValueError('Archived reference conditions differ from the current fault inventory')
    for source, digest in inputs.items():
        path = Path(source)
        if reference / 'plans' in path.parents:
            original = str(ROOT / 'saved_plans' / path.relative_to(reference / 'plans'))
            if original in previous['inputs'] and previous['inputs'][original] != digest:
                raise ValueError('Archived reference plan changed: ' + source)
    # Do not regenerate shuffled states: this is a comparison of the same scenes.
    layouts = {}
    for condition in conditions:
        if condition['unavailable']:
            continue
        source = layout_path(reference, condition)
        if not source.exists():
            source = source.with_suffix('.unavailable.json')
        if not source.is_file():
            raise FileNotFoundError('Reference layout is missing: ' + str(source))
        destination = args.output.resolve() / 'layouts' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inputs[str(source)] = file_hash(source)
        layouts[source.name] = file_hash(source)
    inputs[str(manifest_path)] = file_hash(manifest_path)
    # Model/demo inputs must still match those used by the reference; archived
    # plans are read from reference/plans, independent of later user edits.
    for source, digest in inputs.items():
        if source in previous['inputs'] and previous['inputs'][source] != digest:
            raise ValueError('Reference model or demonstration changed: ' + source)
    comparison = dict(run=str(reference), manifest_sha256=file_hash(manifest_path),
        suites=suites, conditions=len(conditions), layouts=layouts,
        rule='Identical archived plans, conditions, initial states and fault seeds; replace only recovery policy.')
    return conditions, inputs, reference / 'plans', comparison


def initialize(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    config = dict(speed=args.speed, port=args.port, seed=args.seed, repeats=args.repeats,
                  max_waypoint_seconds=args.max_waypoint_seconds, max_policy_steps=args.max_policy_steps,
                  trial_wall_timeout=args.trial_wall_timeout, latency_mode=args.latency_mode)
    backend = getattr(args, 'backend', 'pi05')
    if backend != 'pi05':
        config['backend'] = backend
    if getattr(args, 'reference_run', None):
        config['reference_run'] = str(args.reference_run.resolve())
    if getattr(args, 'suites', None):
        config['suites'] = args.suites
    if getattr(args, 'all_images', False):
        config['all_images'] = True
        config['image_root'] = str((getattr(args, 'image_root', None) or output/'images').resolve())
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['config'] != config:
            raise ValueError('Resume settings differ from manifest; use original settings or a new output folder.')
        for path, digest in manifest['inputs'].items():
            if not Path(path).exists() or file_hash(path) != digest:
                raise ValueError('Experiment input changed since launch: %s. Use a new output folder.' % path)
        return manifest
    comparison = None
    plan_root = args.plan_root
    if getattr(args, 'reference_run', None):
        conditions, inputs, plan_root, comparison = reference_inventory(args)
    else:
        selected = getattr(args, 'suites', None)
        conditions, inputs = build_conditions(plan_root, args.seed, args.repeats,
            suites={s: SUITES[s] for s in SUITES if s in selected} if selected else None,
            latency_mode=args.latency_mode)
    for name in SOURCE_FILES:
        inputs[str(ROOT / name)] = file_hash(ROOT / name)
    for suite in SUITES:
        if getattr(args, 'suites', None) and suite not in args.suites:
            continue
        destination = output / 'plans' / suite
        destination.mkdir(parents=True, exist_ok=True)
        for task_id in range(SUITES[suite]):
            path = plan_root / suite / ('task_%03d.json' % task_id)
            if path.exists():
                shutil.copy2(path, destination / path.name)
    for name in SOURCE_FILES:
        destination = output / 'source' / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    from waypoint_recovery_live import (ORIENTATION_BANDS, ORIENTATION_XY_BANDS,
                                        ORIENTATION_MIN_TOTAL_FRACTION, ORIENTATION_MIN_YAW_FRACTION)
    manifest = dict(version=2, created=utcnow(), config=config, inputs=inputs, conditions=conditions,
        runnable=sum(not c['unavailable'] for c in conditions),
        pairing='Exact same initial scene and fault seed within each latency off/on pair; new seed per repeat.',
        latency=('Off: after failed waypoint. On: after all remaining waypoint attempts, without a timed hold.'
                 if args.latency_mode == 'end_of_plan' else 'Off: after failed waypoint. On: 10 simulation seconds later.'),
        propagate_deviation=True, orientation_bands_degrees=ORIENTATION_BANDS,
        orientation_xy_bands_metres=ORIENTATION_XY_BANDS,
        orientation_min_total_fraction=ORIENTATION_MIN_TOTAL_FRACTION,
        orientation_min_yaw_fraction=ORIENTATION_MIN_YAW_FRACTION,
        scene_layout='Reuse exact stored state across faults and latency pairs. Try up to 8 layout seeds; log unavailable scenes separately.',
        pacing='Configured speed is a maximum simulation playback pace; physics/control/fault timings are unchanged.',
        data='trajectory.npz: post-step time, phase (0 waypoints/1 VLA), padded action, qpos, qvel, eef position + row-major rotation, success. JSON snapshots are full precision. No videos.',
        outcome_rule='Same env.check_success criterion and failure behavior as the live GUI. No stable-success hold.',
        ordering='One repeat of all conditions by task, original/shuffled, failure, intensity, repeat, latency.')
    if comparison is not None:
        manifest['reference'] = comparison
    if config.get('all_images'):
        manifest['image_capture'] = dict(cameras=['agentview', 'robot0_eye_in_hand'],
            width=256, height=256, format='PNG', capture='initial and every env.step',
            controller_folders=['waypoints', 'policy'], image_root=config['image_root'],
            trial_link='trials/<trial_id>/photos', index='photos/frames.csv',
            skipped_steps=0, provenance='captured live during the original evaluation')
        manifest['data'] += ' Lossless scene and wrist PNGs at every control step, indexed by controller and simulation time.'
    atomic_json(manifest_path, manifest)
    for c in conditions:
        if c['unavailable']:
            atomic_json(result_path(output, c), dict(condition=c, result={}, evaluation=classify(c, {})))
    return manifest


def collect(args, manifest):
    output, config = args.output.resolve(), manifest['config']
    atomic_json(output / 'policy.json', check_policy(config['port'], config.get('backend', 'pi05')))
    atomic_json(output / 'run.json', dict(pid=os.getpid(), started=utcnow(), argv=sys.argv))
    executed = 0
    tasks = list(dict.fromkeys(c['suite'] + ':%d' % c['task_id'] for c in manifest['conditions']
                              if not c['unavailable']))
    publish_tables(output, manifest, 'running')
    for task in tasks:
        failures = 0
        while True:
            pending = [c for c in manifest['conditions'] if not c['unavailable']
                and c['suite'] + ':%d' % c['task_id'] == task and not result_path(output, c).exists()]
            if not pending:
                break
            if (output / 'STOP').exists():
                publish_tables(output, manifest, 'stopped')
                return 0
            if args.limit and executed >= args.limit:
                publish_tables(output, manifest, 'limited')
                return 0
            if not storage_available(output, config):
                publish_tables(output, manifest, 'disk_full', message='Less than 512 MiB free on a result or image drive. Resume after freeing space.')
                return 3
            try:
                check_policy(config['port'], config.get('backend', 'pi05'))
            except Exception as exc:
                publish_tables(output, manifest, 'policy_unavailable', error=str(exc))
                return 2
            remaining = args.limit - executed if args.limit else 0
            command = [sys.executable, str(ROOT / 'waypoint_recovery_batch.py'), '--output', str(output),
                       '--worker', task, '--limit', str(remaining)]
            log = output / 'workers' / (task.replace(':', '_') + '.log')
            log.parent.mkdir(exist_ok=True)
            progress = output / 'worker_progress.json'
            if progress.exists():
                progress.unlink()
            started, last_publish = time.monotonic(), 0.
            with log.open('a') as stream:
                process = subprocess.Popen(command, cwd=str(ROOT), stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None:
                        current = json.loads(progress.read_text()) if progress.exists() else {}
                        if time.monotonic() - current.get('started_monotonic', started) > config['trial_wall_timeout']:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                            break
                        if time.monotonic() - last_publish >= 10:
                            publish_tables(output, manifest, 'running', current=current, worker_pid=process.pid)
                            last_publish = time.monotonic()
                        time.sleep(1.)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            newly_done = sum(result_path(output, c).exists() for c in pending)
            executed += newly_done
            if process.returncode == 3:
                publish_tables(output, manifest, 'disk_full')
                return 3
            if process.returncode:
                # A native crash or wall timeout must have a durable, explicit row.
                active = json.loads(progress.read_text()).get('trial_id') if progress.exists() else None
                condition = next((c for c in pending if c['trial_id'] == active), pending[0])
                if not result_path(output, condition).exists():
                    error = dict(error='Worker exited %s; see %s (wall timeout %ss)' % (
                        process.returncode, log.relative_to(output), config['trial_wall_timeout']))
                    atomic_json(result_path(output, condition), dict(condition=condition, result=error,
                        evaluation=classify(condition, error), completed=utcnow()))
                    executed += 1
            if process.returncode:
                failures += 1
                if failures >= 3:
                    for c in pending:
                        if not result_path(output, c).exists():
                            error = dict(error='Task worker failed three times; see %s' % log.relative_to(output))
                            atomic_json(result_path(output, c), dict(condition=c, result=error,
                                evaluation=classify(c, error), completed=utcnow()))
                    publish_tables(output, manifest, 'running', task=task,
                                   message='Task worker errors recorded; continuing other tasks.')
                    break
            publish_tables(output, manifest, 'running', task=task)
    publish_tables(output, manifest, 'complete')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-root', type=Path, default=ROOT / 'saved_plans')
    parser.add_argument('--reference-run', type=Path, help='Reuse archived plans, conditions and layouts from this run')
    parser.add_argument('--suites', nargs='+', choices=tuple(SUITES))
    parser.add_argument('--all-images', action='store_true',
                        help='Save scene and wrist PNGs at every control step, plus an initial pair')
    parser.add_argument('--image-root', type=Path, help='Frame storage directory, optionally on a larger disk')
    parser.add_argument('--speed', type=float, default=20.)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--latency-mode', choices=('end_of_plan', 'timed'), default='end_of_plan')
    parser.add_argument('--max-waypoint-seconds', type=float, default=180.)
    parser.add_argument('--max-policy-steps', type=int, default=None)
    parser.add_argument('--trial-wall-timeout', type=float, default=900.)
    parser.add_argument('--limit', type=int, default=0, help='Run at most N new supported trials; 0 means all.')
    parser.add_argument('--dry-run', action='store_true', help='Build inventory and tables without running the robot.')
    parser.add_argument('--stop', action='store_true', help='Request a clean stop after the current trial.')
    parser.add_argument('--resume', action='store_true', help='Existing output is always resumed automatically.')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop:
        if not (args.output / 'manifest.json').is_file():
            parser.error('No experiment exists at this output path.')
        (args.output / 'STOP').touch()
        print('Stop requested. Finish the current trial, then remove STOP before resuming.')
        return 0
    if args.worker:
        return worker(args.output, args.worker, args.limit)
    if (not np.isfinite([args.speed, args.max_waypoint_seconds, args.trial_wall_timeout]).all()
            or min(args.speed, args.max_waypoint_seconds, args.trial_wall_timeout) <= 0
            or args.repeats < 1 or args.limit < 0 or not 0 <= args.seed < 2**32
            or not 1 <= args.port <= 65535 or (args.max_policy_steps is not None and args.max_policy_steps < 1)):
        parser.error('Invalid speed, limits, repeat count, port or seed.')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This experiment already has a running collector.')
        manifest = initialize(args)
        print('%d conditions: %d runnable, %d unavailable. Speed %.1fx. Output: %s' % (
            len(manifest['conditions']), manifest['runnable'],
            len(manifest['conditions']) - manifest['runnable'], args.speed, args.output.resolve()), flush=True)
        if args.dry_run:
            publish_tables(args.output, manifest, 'inventory')
            return 0
        return collect(args, manifest)


if __name__ == '__main__':
    sys.exit(main())
