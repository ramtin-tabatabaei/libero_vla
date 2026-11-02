"""CSV and PNG execution mode for waypoint_recovery_benchmark.py."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

import waypoint_recovery_benchmark as bench
from waypoint_failures import FAILURES, INTENSITIES, waypoint_primitive
from waypoint_policy import StableSuccess, run as run_policy
from serve_pi05_libero import metadata as pi05_metadata


SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90')
# Known output-only revisions may resume without changing recorded outcomes.
PRE_WORKERS_IMAGES_SHA256 = 'd8dfe4afbbb2604a605b24aa4f735c2765bc444e0661dfde0c12195c566f34e2'
PRE_TRIALS_IMAGES_SHA256 = '6c2bead70748cecd4a0243b1488816e5f7589b6473825f40495e7f36079d7a41'
PRE_CONDITIONS_IMAGES_SHA256 = 'b65fe192a12544206d33804d05ce24c4e6fc710a543c699df123f5f50c7a51a5'
# Restricting the schedule preserves the previous Grasp trials and their seeds.
PRE_GRASP_ONLY_IMAGES_SHA256 = 'eb393889fdf51c4a587c150548d65b0fb535b0855f7566df6abede082b954dbc'
PRE_RESUME_IMAGES_SHA256 = 'a23509cd3a4bb2412fc0eed32b3332e85cd758eb1df9edb81d2be35ce05dd3bf'
# Adding Transport drops leaves the preceding Grasp conditions unchanged.
PRE_TRANSPORT_IMAGES_SHA256 = 'aeac48a8a2f58ed729fbbdf03f5a55f864306ef7312cd3a5c6600421262ba2aa'
PRE_UDVLA_POLICY_SHA256 = '60c023903a2e105c8b99ab3a9396fe49c352aab3dc747022c7abfb6b24c268d8'
UDVLA_POLICY_SHA256 = '92a7c5dd38237833c2c1fd229a44236b74e47160a97b5849cbbbab273dda68bd'
PRE_CONDITIONS_LAYOUT = 'trials/<trial_id>/photos/{waypoints,policy}'
IMAGE_LAYOUT = 'trials/<suite>/task_<id>/<scene>/<condition>/photos/{waypoints,policy}'
CONTROLLER_FOLDERS = dict(waypoint='waypoints', vla='policy')
FIELDS = (
    'trial_id', 'suite', 'task_id', 'language', 'scene', 'mode', 'kind', 'intensity',
    'phase', 'requested_delay_seconds', 'seed', 'fault_seed', 'status', 'success',
    'vla_recovered', 'task_success', 'observed_failure', 'injection_row',
    'success_before_policy', 'episode_start_sim_time', 'failure_sim_time',
    'handoff_sim_time', 'actual_delay_seconds', 'policy_start_sim_time',
    'first_action_sim_time', 'first_goal_sim_time', 'stable_start_sim_time',
    'episode_end_sim_time', 'waypoint_sim_seconds', 'policy_sim_seconds',
    'failure_to_goal_seconds', 'failure_to_stable_success_seconds',
    'total_sim_seconds', 'stable_seconds', 'policy_steps', 'vla_action_steps',
    'stability_hold_steps', 'wall_seconds',
    'waypoint_images', 'vla_images', 'initial_state_sha256', 'plan_path',
    'plan_sha256', 'failure_measurement', 'fault_targets', 'missed',
    'waypoint_result', 'error', 'configuration_sha256', 'worker', 'worker_pid',
    'trial_dir', 'waypoint_image_dir', 'vla_image_dir',
)
PRE_CONDITIONS_FIELDS = FIELDS[:-3]
PRE_WORKERS_FIELDS = PRE_CONDITIONS_FIELDS[:-2]


def add_arguments(parser):
    parser.add_argument('--images-only', action='store_true',
        help='All-task Grasp failures and High grasp loss in Transport; write CSV and PNGs')
    parser.add_argument('--task-suites', nargs='+', choices=SUITES, default=list(SUITES))
    parser.add_argument('--task-ids', nargs='+', type=int,
        help='Optional task IDs within each selected suite (images-only mode)')
    parser.add_argument('--handoff-delays', nargs='+', type=float, default=[0., 5.],
        help='Seconds of waypoint control after an observed failure (images-only mode)')
    parser.add_argument('--stable-seconds', type=float, default=4.,
        help='Continuous successful goal with stable objects, in simulation seconds')
    parser.add_argument('--stability-metres', type=float, default=.01)
    parser.add_argument('--stability-degrees', type=float, default=5.)
    parser.add_argument('--photo-every', type=float, default=.5,
        help='Simulation seconds between PNGs, plus initial/failure/handoff/terminal snapshots')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=5,
        help='Simulation processes sharing the Pi 0.5 server; one CSV writer (default: 5)')
    parser.add_argument('--dry-run', action='store_true',
        help='Print the images-only schedule and plan availability without moving the robot')


def all_tasks(suites, task_ids=None):
    from libero.libero import benchmark
    result = []
    for suite in dict.fromkeys(suites):
        count = benchmark.get_benchmark_dict()[suite]().n_tasks
        selected = range(count) if task_ids is None else dict.fromkeys(task_ids)
        for task in selected:
            if not 0 <= task < count:
                raise ValueError('%s has task IDs 0 through %d' % (suite, count-1))
            result.append((suite, task))
    return result


def fault_intensities(kind, phase, motion=None):
    """Pose faults and no-grasp happen in Grasp; acquired-object drops in Transport."""
    if kind == 'grasp_loss':
        if phase == 'grasp':
            return ('low',)
        if phase == 'transport' and motion in ('line', 'reach', None):
            return ('high',)
        return ()
    if phase != 'grasp':
        return ()
    return tuple(INTENSITIES) if motion in ('line', 'reach', None) else ()


class GraspInjector(bench.MeasuredInjector):
    def permitted_intensities(self, kind, primitive, motion=None):
        return fault_intensities(kind, primitive, motion)


def schedule(tasks, delays=(0., 5.), seed=1000):
    result = []
    for suite, task in tasks:
        for scene in ('original', 'shuffled'):
            common = dict(suite=suite, task_id=task, scene=scene, seed=seed+SUITES.index(suite)*1000+task)
            for mode in ('normal_waypoints', 'normal_policy'):
                result.append(dict(common, mode=mode, kind='none', intensity='none', phase='none',
                                   requested_delay_seconds=0.))
            for kind in FAILURES:
                for intensity in INTENSITIES:
                    phase = 'transport' if kind == 'grasp_loss' and intensity == 'high' else 'grasp'
                    if intensity not in fault_intensities(kind, phase):
                        continue
                    key = '%s/%s/%s/%s/%s/%s/%s' % (seed, suite, task, scene, kind, intensity, phase)
                    # The same fault draw and layout are reused for both delays.
                    fault_seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    for delay in delays:
                        result.append(dict(common, mode='recovery', kind=kind, intensity=intensity,
                            phase=phase, requested_delay_seconds=float(delay), fault_seed=fault_seed))
    return result


def identifier(spec):
    base = '{suite}_t{task_id:03d}_{scene}_{mode}'.format(**spec)
    if spec['mode'] == 'recovery':
        base += '_{kind}_{intensity}_{phase}_delay{delay}s'.format(
            delay=('%g' % spec['requested_delay_seconds']).replace('.', 'p'), **spec)
    return base


def plan_path(suite, task):
    for store in ('saved_plans', 'task_waypoints_gmm'):
        path = bench.ROOT/store/suite/('task_%03d.json' % task)
        if path.is_file():
            return path
    return None


class PairedScenes:
    """Freeze both layouts in memory; reconstruct deterministically on resume."""
    def __init__(self, expected_hashes=None):
        self.states = {}
        self.expected_hashes = expected_hashes or {}

    def restore(self, session, chain, spec):
        key = spec['suite'], spec['task_id'], spec['scene']
        if key not in self.states:
            if chain is not None and chain.model_path:
                from waypoint_replay import initial_state
                model_path = Path(chain.model_path)
                model = json.loads(model_path.read_text())
                demo_path = Path(model['demonstration_files'][0])
                demo = json.loads(demo_path.read_text())
                session.training_state = np.asarray(initial_state(demo, demo_path))
                session.training_gripper = bench.sp.GRIPPER_OPEN
            session.restore_training_scene()
            session.last_shuffle_mapping = None
            if spec['scene'] == 'shuffled':
                session.shuffle_scene(seed=spec['seed'])
            self.states[key] = session.env.sim.get_state().flatten().copy()
        state = self.states[key]
        digest = hashlib.sha256(state.tobytes()).hexdigest()
        if key in self.expected_hashes and self.expected_hashes[key] != digest:
            raise ValueError('The starting scene changed since the completed CSV trials; choose a new output directory')
        session.env.set_init_state(state.copy())
        session.attach(settle=False)
        session.executor.gripper = bench.sp.GRIPPER_OPEN
        return digest


def parse_identifier(trial_id):
    """Recover conditions for legacy image directories and interrupted trials."""
    match = re.fullmatch(r'(?P<suite>'+ '|'.join(SUITES) + r')_t(?P<task_id>\d+)_'
                         r'(?P<scene>original|shuffled)_(?P<condition>.+)', trial_id)
    if match is None:
        raise ValueError('Unrecognized trial ID: '+trial_id)
    spec = match.groupdict()
    spec['task_id'] = int(spec['task_id'])
    condition = spec.pop('condition')
    if condition in ('normal_waypoints', 'normal_policy'):
        return dict(spec, mode=condition, kind='none', intensity='none', phase='none',
                    requested_delay_seconds=0.)
    fault = re.fullmatch(r'recovery_(?P<kind>'+ '|'.join(FAILURES) + r')_'
        r'(?P<intensity>low|high)_(?P<phase>grasp|transport|place)_delay'
        r'(?P<delay>\d+(?:p\d+)?(?:e[+-]?\d+)?)s', condition)
    if fault is None:
        raise ValueError('Unrecognized trial ID: '+trial_id)
    details = fault.groupdict()
    delay = float(details.pop('delay').replace('p', '.'))
    return dict(spec, mode='recovery', requested_delay_seconds=delay, **details)


def trial_path(spec):
    """One unique relative folder for each task, layout and experimental condition."""
    if isinstance(spec, str):
        spec = parse_identifier(spec)
    folder = Path('trials')/spec['suite']/('task_%03d' % int(spec['task_id']))/spec['scene']
    if spec['mode'] in ('normal_waypoints', 'normal_policy'):
        controller = 'waypoints' if spec['mode'] == 'normal_waypoints' else 'vla'
        return folder/'no_failure'/controller
    delay = ('%g' % float(spec['requested_delay_seconds'])).replace('.', 'p')
    return folder/'failures'/spec['kind']/spec['intensity']/spec['phase']/('handoff_'+delay+'s')


def image_paths(spec):
    folder = trial_path(spec)
    return dict(trial_dir=folder.as_posix(),
                waypoint_image_dir=(folder/'photos'/'waypoints').as_posix(),
                vla_image_dir=(folder/'photos'/'policy').as_posix())


def trial_folder(output, spec):
    folder = output/trial_path(spec)
    for name in CONTROLLER_FOLDERS.values():
        (folder/'photos'/name).mkdir(parents=True, exist_ok=True)
    return folder


def migrate_trial_folders(output):
    """Preserve complete trial folders when grouping the previous flat trials/ layout."""
    root = output/'trials'
    if not root.exists():
        return
    moves, destinations = [], set()
    for source in sorted(root.iterdir()):
        if source.name in SUITES and source.is_dir():
            continue
        if not source.is_dir():
            raise ValueError('Unrecognized trial folder; preserved: '+str(source))
        destination = output/trial_path(source.name)
        if destination.exists() or destination in destinations:
            raise ValueError('Trial migration would overwrite '+str(destination))
        moves.append((source, destination))
        destinations.add(destination)
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)


def migrate_flat_images(output):
    """Move previous controller folders into trials after CSV compatibility checks."""
    pattern = re.compile(r'^([A-Za-z0-9_+-]+)_(waypoint|vla)_(\d+)_t(-?\d+(?:\.\d+)?)_'
                         r'(frame|initial|failure|handoff|terminal)\.png$')
    moves, directories = [], []
    for name, controller in (('waypoint_images', 'waypoint'), ('vla_images', 'vla')):
        directory = output/name
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError('Expected an image directory: '+str(directory))
        directories.append(directory)
        for source in sorted(directory.iterdir()):
            match = pattern.fullmatch(source.name)
            if not source.is_file() or match is None or match[2] != controller:
                raise ValueError('Unrecognized old image; preserved without migration: '+str(source))
            trial_id, _, count, timestamp, label = match.groups()
            folder = output/trial_path(trial_id)
            filename = 'frame_%06d_t_%09.3f.png' % (int(count), float(timestamp))
            destination = folder/'photos'/CONTROLLER_FOLDERS[controller]/filename
            if destination.exists():
                raise ValueError('Image migration would overwrite '+str(destination))
            moves.append((source, trial_id, folder, destination, label))
    for source, trial_id, folder, destination, label in moves:
        trial_folder(output, trial_id)
        if label != 'frame':
            shutil.copyfile(source, folder/(label+'.png'))
        source.rename(destination)
    for directory in directories:
        directory.rmdir()


class Images:
    """Record controller-labelled PNGs without video, state or trajectory files."""
    def __init__(self, session, output, trial_id, interval):
        self.session, self.output, self.trial_id, self.interval = session, output, trial_id, interval
        self.folder = trial_folder(output, trial_id)
        self.controller, self.current_row = 'waypoint', None
        self.counts = dict(waypoint=0, vla=0)
        self.last = {}

    def capture(self, label='frame'):
        import imageio
        now = float(self.session.env.sim.data.time)
        count = self.counts[self.controller]
        filename = 'frame_%06d_t_%09.3f.png' % (count, now)
        directory = self.folder/'photos'/CONTROLLER_FOLDERS[self.controller]
        picture = self.session.image()
        imageio.imwrite(directory/filename, picture)
        if label != 'frame':
            imageio.imwrite(self.folder/(label+'.png'), picture)
        self.counts[self.controller] += 1
        self.last[self.controller] = now

    def __enter__(self):
        self.original = self.session.env.step
        def step(action):
            result = self.original(action)
            now = float(self.session.env.sim.data.time)
            if now-self.last.get(self.controller, -float('inf')) + 1e-8 >= self.interval:
                self.capture()
            return result
        self.session.env.step = step
        return self

    def __exit__(self, *_):
        self.session.env.step = self.original


def eligible_row(chain, spec):
    return next((i for i, w in enumerate(chain.waypoints)
                 if waypoint_primitive(w) == spec['phase'] and
                 spec['intensity'] in fault_intensities(spec['kind'], spec['phase'], w.motion)), None)


def stable_waypoints(session, args):
    stable = StableSuccess(args.stable_seconds, args.stability_metres, args.stability_degrees)
    start = float(session.env.sim.data.time)
    deadline = start + args.stable_seconds + 2.
    while True:
        now = float(session.env.sim.data.time)
        goal = bool(session.env.check_success())
        if stable.update(now, goal, session.poses()) or not goal or now + 1e-8 >= deadline:
            return dict(success=goal and stable.duration + 1e-8 >= args.stable_seconds,
                        task_success=goal, stable_seconds=stable.duration,
                        first_goal_sim_time=stable.first_goal, stable_start_sim_time=stable.start)
        session.executor.hold(1)


def execute(session, chain, spec, args, paired, entry):
    entry['language'] = session.task.language
    entry['initial_state_sha256'] = paired.restore(session, chain, spec)
    start = float(session.env.sim.data.time)
    entry['episode_start_sim_time'] = start
    images = Images(session, args.output, entry['trial_id'], args.photo_every)
    with images:
        try:
            if spec['mode'] in ('normal_waypoints', 'recovery'):
                images.capture('initial')
                if spec['mode'] == 'normal_waypoints':
                    entry['waypoint_result'] = bench.run_normal_waypoints(
                        session, chain, images, args.max_waypoint_seconds)
                    with bench.suspend_images(session):
                        entry.update(stable_waypoints(session, args))
                    entry['status'] = 'normal_waypoints_success' if entry['success'] else 'normal_waypoints_failed'
                else:
                    detail = bench.run_fault(session, chain, spec, spec['requested_delay_seconds'],
                        args.max_waypoint_seconds, None, images, injector_class=GraspInjector,
                        intensity_rule=fault_intensities,
                        capture=lambda session, folder, label: images.capture(label))
                    entry.update({key: detail[key] for key in ('observed_failure', 'injection_row',
                        'waypoint_result', 'fault_targets', 'missed')})
                    if not detail['observed_failure']:
                        entry['status'] = 'failure_not_observed'
                        return
                    entry.update({key: detail[key] for key in ('handoff_sim_time',
                        'actual_delay_seconds', 'success_before_policy')})
                    entry['failure_sim_time'] = detail['failure_event']['sim_time']
                    entry['failure_measurement'] = detail['failure_event']
            entry['waypoint_sim_seconds'] = float(session.env.sim.data.time)-start
            if spec['mode'] != 'normal_waypoints':
                images.controller = 'vla'
                images.capture('handoff' if spec['mode'] == 'recovery' else 'initial')
                result = run_policy(session, session.task.language, port=args.port,
                    backend='pi05', max_steps=args.max_policy_steps, stable_seconds=args.stable_seconds,
                    stability_metres=args.stability_metres, stability_degrees=args.stability_degrees)
                entry.update({key: result[key] for key in ('success', 'task_success', 'stable_seconds',
                    'policy_start_sim_time', 'first_action_sim_time', 'first_goal_sim_time', 'stable_start_sim_time')})
                entry['policy_steps'] = result['steps']
                entry['vla_action_steps'] = result.get('vla_action_steps', result['steps'])
                entry['stability_hold_steps'] = result.get('stability_hold_steps', 0)
                entry['policy_sim_seconds'] = result['policy_end_sim_time']-result['policy_start_sim_time']
                if spec['mode'] == 'normal_policy':
                    entry['status'] = 'normal_policy_success' if result['success'] else 'normal_policy_failed'
                else:
                    entry['status'] = ('already_successful' if result['success'] and
                                       entry['success_before_policy'] and entry['vla_action_steps'] == 0 else
                                       'recovered' if result['success'] else 'recovery_failed')
                    entry['vla_recovered'] = entry['status'] == 'recovered'
                    if result['first_goal_sim_time'] is not None:
                        entry['failure_to_goal_seconds'] = result['first_goal_sim_time']-entry['failure_sim_time']
                    if result['success']:
                        entry['failure_to_stable_success_seconds'] = result['policy_end_sim_time']-entry['failure_sim_time']
        finally:
            images.capture('terminal')
            end = float(session.env.sim.data.time)
            entry.update(episode_end_sim_time=end, total_sim_seconds=end-start,
                waypoint_images=images.counts['waypoint'], vla_images=images.counts['vla'])
            if 'waypoint_sim_seconds' not in entry:
                entry['waypoint_sim_seconds'] = end-start


class CsvResults:
    """Flush every completed attempt; interrupted trials are rerun on resume."""
    def __init__(self, output, configuration, resume, compatible=()):
        output.mkdir(parents=True, exist_ok=True)
        allowed = {'timing.csv', 'trials'}
        if resume:
            allowed.update(('timing.csv.tmp', 'waypoint_images', 'vla_images'))
        if any(p.name not in allowed for p in output.iterdir()):
            raise ValueError('Choose a dedicated output directory; images-only output must contain timing.csv and trials/')
        self.completed = set()
        self.state_hashes = {}
        path = output/'timing.csv'
        rewrite, rows = False, []
        if path.exists():
            if not resume:
                raise ValueError('Output already exists; use --resume with the same configuration')
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames not in (list(FIELDS), list(PRE_CONDITIONS_FIELDS), list(PRE_WORKERS_FIELDS)):
                    raise ValueError('Existing CSV uses a different benchmark format')
                rewrite = reader.fieldnames != list(FIELDS)
                for row in reader:
                    if row.get('configuration_sha256') not in {configuration, *compatible}:
                        raise ValueError('Resume requires the same options, plans and policy server configuration')
                    if reader.fieldnames == list(PRE_WORKERS_FIELDS):
                        row.update(worker=1, worker_pid='')
                    if rewrite:
                        row.update(image_paths(row))
                        rows.append(row)
                    self.completed.add(row['trial_id'])
                    if row.get('initial_state_sha256'):
                        key = row['suite'], int(row['task_id']), row['scene']
                        previous = self.state_hashes.setdefault(key, row['initial_state_sha256'])
                        if previous != row['initial_state_sha256']:
                            raise ValueError('Existing CSV contains inconsistent paired starting scenes')
        elif any(output.iterdir()):
            raise ValueError('Existing image folders have no CSV; choose a new output directory')
        if resume:
            migrate_trial_folders(output)
            migrate_flat_images(output)
        temporary = path.with_suffix('.csv.tmp')
        if rewrite:
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        elif resume and temporary.exists():
            temporary.unlink()
        (output/'trials').mkdir(exist_ok=True)
        new = not path.exists()
        self.stream = path.open('a', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            self.writer.writeheader()
            self.stream.flush()

    def append(self, entry):
        if all(entry.get(key) not in (None, '') for key in ('suite', 'task_id', 'scene', 'mode')):
            entry.update(image_paths(entry))
        data = {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in entry.items()}
        self.writer.writerow(data)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.completed.add(entry['trial_id'])

    def close(self):
        self.stream.close()


def configuration_digests(config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    legacy = dict(config)
    legacy.pop('image_layout', None)
    compatible = []
    if tuple(config.get('failure_phases', ())) == ('grasp', 'transport'):
        # Only the schedule and phase permissions were extended. Reconstruct
        # the audited predecessor, including its exact options and other code,
        # so existing Grasp rows can resume without accepting protocol changes.
        previous = dict(config, failure_phases=['grasp'], code=dict(config['code'],
            **{'waypoint_recovery_images.py': PRE_TRANSPORT_IMAGES_SHA256}))
        previous_digest, previous_compatible = configuration_digests(previous)
        compatible.extend((previous_digest, *previous_compatible))
    if tuple(config.get('failure_phases', ())) == ('grasp',):
        # Admit the known preceding schedule, retaining exact checks for all
        # controller, policy, fault, timing and scene settings. Grasp behavior
        # and IDs are identical; only Place scenarios have been removed.
        previous = dict(config)
        previous.pop('failure_phases')
        previous['code'] = dict(config['code'],
            **{'waypoint_recovery_images.py': PRE_GRASP_ONLY_IMAGES_SHA256})
        compatible.append(hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest())
    if 'image_layout' in config:
        trials_code = dict(config['code'], **{'waypoint_recovery_images.py': PRE_CONDITIONS_IMAGES_SHA256})
        trials = dict(config, image_layout=PRE_CONDITIONS_LAYOUT, code=trials_code)
        compatible.append(hashlib.sha256(json.dumps(trials, sort_keys=True).encode()).hexdigest())
        flat_code = dict(config['code'], **{'waypoint_recovery_images.py': PRE_TRIALS_IMAGES_SHA256})
        flat = dict(legacy, code=flat_code)
        compatible.append(hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest())
    legacy_code = {name: value for name, value in config['code'].items()
                   if name != 'waypoint_recovery_workers.py'}
    legacy_code['waypoint_recovery_images.py'] = PRE_WORKERS_IMAGES_SHA256
    compatible.append(hashlib.sha256(json.dumps(dict(legacy, code=legacy_code), sort_keys=True).encode()).hexdigest())
    if (tuple(config.get('failure_phases', ())) == ('grasp',)
            and config.get('image_layout') == IMAGE_LAYOUT
            and config.get('policy') == pi05_metadata()):
        previous = dict(config, code=dict(config['code'],
            **{'waypoint_recovery_images.py': PRE_RESUME_IMAGES_SHA256}))
        compatible.append(hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest())
        # UD-VLA added a separate backend; Pi 0.5 control remained identical.
        # Admit only that audited policy revision, preserving every other check.
        if config['code'].get('waypoint_policy.py') == UDVLA_POLICY_SHA256:
            previous['code'] = dict(previous['code'],
                **{'waypoint_policy.py': PRE_UDVLA_POLICY_SHA256})
            compatible.append(hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest())
            previous.pop('failure_phases')
            previous['code']['waypoint_recovery_images.py'] = PRE_GRASP_ONLY_IMAGES_SHA256
            compatible.append(hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest())
    return digest, tuple(compatible)


def iter_entries(args, specs, paths, plan_hashes, digest, completed, state_hashes,
                 worker=1, on_start=None, stop_event=None):
    """One worker owns each task's scenes and emits completed trials immediately."""
    from waypoint_gmm import GMMChain
    from waypoint_gmm_editor import GMMSession
    session = GMMSession(bench.sp.Args(policy_port=args.port, policy_max_steps=args.max_policy_steps))
    session.save_replay_archive = False
    if stop_event is not None:
        def tick():
            if stop_event.is_set():
                raise KeyboardInterrupt()
        session.tick = tick
    paired = PairedScenes(state_hashes)
    current, chain, plan_error = None, None, None
    try:
        for spec in specs:
            if stop_event is not None and stop_event.is_set():
                break
            trial_id = identifier(spec)
            if trial_id in completed:
                continue
            folder = trial_folder(args.output, trial_id)
            for path in folder.rglob('*.png'):
                path.unlink()
            entry = dict(spec, trial_id=trial_id, success=False, vla_recovered=False,
                         configuration_sha256=digest, worker=worker, worker_pid=os.getpid())
            entry.update(image_paths(spec))
            started = time.monotonic()
            if on_start is not None:
                on_start(worker, trial_id)
            try:
                key = spec['suite'], spec['task_id']
                path = paths[key]
                entry.update(plan_path=str(path) if path else '', plan_sha256=plan_hashes[str(key)])
                if path is None and spec['mode'] != 'normal_policy':
                    entry.update(status='missing_plan', error='Learn waypoints for this task in the GUI first')
                else:
                    if current != key:
                        session.args.seed = spec['seed']
                        session.load(*key, cameras=True)
                        chain, plan_error = None, None
                        if path is not None:
                            try:
                                chain = GMMChain.load(path, *key)
                            except Exception as exc:
                                plan_error = '%s: %s' % (type(exc).__name__, exc)
                        current = key
                    if plan_error is not None and spec['mode'] != 'normal_policy':
                        entry.update(status='invalid_plan', error=plan_error)
                    elif spec['mode'] == 'recovery' and eligible_row(chain, spec) is None:
                        entry.update(status='not_applicable', error='No eligible '+spec['phase']+' waypoint; other phases are excluded')
                    else:
                        if spec['mode'] == 'recovery':
                            spec = dict(spec, injection_row=eligible_row(chain, spec))
                        execute(session, chain, spec, args, paired, entry)
            except Exception as exc:
                entry.update(status='error', error='%s: %s' % (type(exc).__name__, exc))
            entry['wall_seconds'] = round(time.monotonic()-started, 3)
            yield entry
    finally:
        if session.env is not None:
            session.env.close()


def run_images(args, parser):
    try:
        StableSuccess(args.stable_seconds, args.stability_metres, args.stability_degrees)
        workers = getattr(args, 'workers', 1)
        if workers < 1:
            raise ValueError('Worker count must be positive')
        if args.stable_seconds <= 0 or not np.isfinite(args.photo_every) or args.photo_every <= 0:
            raise ValueError('Stable-success duration and photo interval must be positive')
        if not 0 <= args.seed < 2**32-5000:
            raise ValueError('Seed must be nonnegative and leave room for the per-task seeds')
        for delay in args.handoff_delays:
            bench.HandoffClock(delay)
        args.handoff_delays = list(dict.fromkeys(args.handoff_delays))
        tasks = all_tasks(args.task_suites, args.task_ids)
        specs = schedule(tasks, args.handoff_delays, args.seed)
        paths = {key: plan_path(*key) for key in tasks}
        if args.limit is not None:
            specs = specs[:args.limit]
        print('%d tasks, %d trials; Grasp failures and High grasp loss in Transport; workers %d; handoff delays %s; stable success %.1fs; no video' %
              (len(tasks), len(specs), workers, args.handoff_delays, args.stable_seconds), flush=True)
        missing = ['%s/task_%03d' % key for key, path in paths.items() if path is None]
        if missing:
            print('Missing waypoint plans (recorded in CSV; VLA-only baselines still run): '+', '.join(missing), flush=True)
        if args.dry_run:
            pending = specs
            if args.resume and (args.output/'timing.csv').exists():
                with (args.output/'timing.csv').open(newline='') as stream:
                    completed = {row['trial_id'] for row in csv.DictReader(stream)}
                pending = [spec for spec in specs if identifier(spec) not in completed]
                print('Resume: %d scheduled conditions already recorded; %d pending' %
                      (len(specs)-len(pending), len(pending)), flush=True)
            for spec in pending[:6]:
                print(identifier(spec))
            print('Output:', args.output/'timing.csv', args.output/'trials', '|', IMAGE_LAYOUT)
            return
    except ValueError as exc:
        parser.error(str(exc))

    from waypoint_demo import PolicyConnection
    client = PolicyConnection(args.port)
    try:
        metadata = client.connect()
    finally:
        client.close()
    expected = pi05_metadata()
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected standard Pi 0.5 (pi05_libero, no RL). '
                         'Start ./serve_pi05_libero.sh on port 8000; received: '+repr(metadata))
    print('Pi 0.5:', metadata['model_id'], '(supervised, no RL) | simulation workers:', workers, flush=True)
    plan_hashes = {str(key): hashlib.sha256(path.read_bytes()).hexdigest() if path else None
                   for key, path in paths.items()}
    config = dict(version=1, tasks=tasks, delays=args.handoff_delays, seed=args.seed, failure_phases=['grasp', 'transport'],
        port=args.port, policy=expected, plans=plan_hashes, stable_seconds=args.stable_seconds,
        stability_metres=args.stability_metres, stability_degrees=args.stability_degrees,
        max_policy_steps=args.max_policy_steps, max_waypoint_seconds=args.max_waypoint_seconds,
        photo_every=args.photo_every, image_layout=IMAGE_LAYOUT)
    # Reject resumes after controller/injection changes, as well as option changes.
    config['code'] = {name: hashlib.sha256((bench.ROOT/name).read_bytes()).hexdigest()
                      for name in ('waypoint_recovery_images.py', 'waypoint_recovery_benchmark.py',
                                   'waypoint_failures.py', 'waypoint_policy.py', 'waypoint_gmm_editor.py',
                                   'waypoint_recovery_workers.py', 'serve_pi05_libero.py')}
    digest, compatible = configuration_digests(config)
    try:
        results = CsvResults(args.output, digest, args.resume, compatible)
    except ValueError as exc:
        parser.error(str(exc))
    def start(worker, trial):
        print('START worker=%d %s' % (worker, trial), flush=True)
    iterator = None
    try:
        pending = [spec for spec in specs if identifier(spec) not in results.completed]
        if workers == 1 or not pending:
            iterator = iter_entries(args, pending, paths, plan_hashes, digest,
                                    results.completed, results.state_hashes, on_start=start)
        else:
            from waypoint_recovery_workers import parallel_entries
            iterator = parallel_entries(args, pending, paths, plan_hashes, digest,
                                        results.completed, results.state_hashes, on_start=start)
        for entry in iterator:
            results.append(entry)
            print('RESULT %d/%d worker=%d %s %s success=%s%s' %
                  (len(results.completed), len(specs), entry['worker'], entry['trial_id'],
                   entry['status'], entry['success'],
                   ' '+entry['error'] if entry['status'] == 'error' else ''), flush=True)
    except KeyboardInterrupt:
        print('Stopped. Completed rows are saved; rerun the same command with --resume.', flush=True)
    finally:
        if iterator is not None:
            iterator.close()
        results.close()
    print('Saved:', args.output/'timing.csv', flush=True)
