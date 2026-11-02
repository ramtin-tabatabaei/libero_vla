"""Physical trajectory evidence and timing for normal and recovery rollouts."""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def task_targets(session):
    world = session.world
    goals = world.domain.parsed_problem.get('goal_state', [])
    goal = next(g for g in goals if len(g) >= 3 and g[0].lower() in ('in', 'on')
                and g[1] in world.domain.objects_dict)
    return dict(source=goal[1], destination=goal[2], destination_body=world.parent_body(goal[2]))


class TrajectoryRecorder:
    """Record every env.step, including policy steps, without changing commands."""
    def __init__(self, session, folder, name, video=True, video_gate=None, video_name=None,
                 photo_every=None):
        if photo_every is not None and (not np.isfinite(photo_every) or photo_every <= 0):
            raise ValueError('Photo interval must be finite and positive')
        self.session, self.folder, self.name = session, Path(folder), name
        self.targets = task_targets(session)
        self.samples = []
        self.current_row = None
        self.started_sim = float(session.env.sim.data.time)
        self.started_wall = time.monotonic()
        self.first_action_wall = None
        self.first_action_sim = None
        self.video = video
        self.video_gate = video_gate
        self.video_name = video_name or name
        self.video_start = None
        self.writer = None
        self.last_frame = -np.inf
        self.video_seconds = 0.
        self.focus = None
        self.photo_every = photo_every
        self.last_photo = -np.inf
        self.photo_count = 0
        self.photo_seconds = 0.

    def save_photo(self, sample, frame=None):
        import imageio
        started = time.monotonic()
        directory = self.folder/'photos'/self.name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/('frame_%06d_t_%09.3f.png' % (self.photo_count, sample['time']))
        imageio.imwrite(path, self.session.image() if frame is None else frame)
        record = dict(file=str(path.relative_to(self.folder)), time=sample['time'],
                      sim_time=sample['sim_time'], row=sample['row'], success=sample['success'])
        with (directory/'index.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        self.photo_count += 1
        self.last_photo = sample['time']
        self.photo_seconds += time.monotonic()-started

    def __enter__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        env = self.session.env
        self.original_step = env.step
        self.had_instance_step = 'step' in vars(env)
        self.capture(None)
        def step(action, *args, **kwargs):
            if self.first_action_wall is None:
                self.first_action_wall = time.monotonic()-self.started_wall
                self.first_action_sim = float(env.sim.data.time)-self.started_sim
            result = self.original_step(action, *args, **kwargs)
            obs = result[0] if isinstance(result, tuple) else None
            self.capture(action, obs)
            return result
        env.step = step
        return self

    def __exit__(self, *_):
        if self.had_instance_step:
            self.session.env.step = self.original_step
        else:
            delattr(self.session.env, 'step')
        try:
            if (self.photo_every is not None and self.samples and
                    self.samples[-1]['time'] > self.last_photo+1e-8):
                self.save_photo(self.samples[-1])
            if self.writer is not None:
                self.writer.close()
        finally:
            data = dict(version=1, name=self.name, targets=self.targets, focus=self.focus,
                        control_timestep=float(self.session.env.env.control_timestep),
                        started_sim_time=self.started_sim, first_action_wall_seconds=self.first_action_wall,
                        first_action_sim_seconds=self.first_action_sim,
                        wall_seconds=time.monotonic()-self.started_wall,
                        video_encoding_wall_seconds=self.video_seconds,
                        video_start_sim_seconds=self.video_start,
                        video_file=self.video_name+'.mp4' if self.writer is not None else None,
                        photo_interval_sim_seconds=self.photo_every, photo_count=self.photo_count,
                        photo_recording_wall_seconds=self.photo_seconds,
                        photo_index_file=('photos/'+self.name+'/index.jsonl'
                                          if self.photo_count else None),
                        samples=self.samples)
            (self.folder/(self.name+'.trajectory.json')).write_text(json.dumps(data, allow_nan=False)+'\n')
        return False

    def capture(self, action, obs=None):
        session = self.session
        domain = session.env.env
        objects = {name: session.world.pos(name).tolist() for name in domain.objects_dict}
        pads = domain.robots[0].gripper.important_geoms
        contacts, grasped = [], []
        for name, obj in domain.objects_dict.items():
            if (domain.check_contact(pads['left_fingerpad'], obj.contact_geoms) or
                    domain.check_contact(pads['right_fingerpad'], obj.contact_geoms)):
                contacts.append(name)
            if domain._check_grasp(domain.robots[0].gripper, obj.contact_geoms):
                grasped.append(name)
        if self.focus is None:
            self.focus = ('destination' if self.targets['source'] in grasped else 'source')
        position, rotation = session.gripper_pose()
        target = (session.world.pos(self.targets['destination']) if self.focus == 'destination'
                  else np.asarray(objects[self.targets['source']]))
        sample = dict(time=float(session.env.sim.data.time)-self.started_sim,
            sim_time=float(session.env.sim.data.time), wall_seconds=time.monotonic()-self.started_wall,
            row=self.current_row, eef=position.tolist(), orientation=rotation.tolist(),
            objects=objects, contacts=contacts, grasped=grasped,
            focus_position=np.asarray(target).tolist(), focus_distance=float(np.linalg.norm(position-target)),
            source_distance=float(np.linalg.norm(position-np.asarray(objects[self.targets['source']]))),
            success=bool(session.env.check_success()), action=None if action is None else np.asarray(action).tolist())
        self.samples.append(sample)
        frame = None
        if self.photo_every is not None and sample['time'] >= self.last_photo+self.photo_every-1e-8:
            frame = session.image()
            self.save_photo(sample, frame)
        enabled = self.video and (self.video_gate is None or self.video_gate())
        if enabled and sample['time'] >= self.last_frame+.1-1e-8:
            started = time.monotonic()
            if self.writer is None:
                import imageio
                self.writer = imageio.get_writer(self.folder/(self.video_name+'.mp4'), fps=10,
                                                codec='libx264', macro_block_size=2)
                self.video_start = sample['time']
            if frame is not None:
                pass
            elif obs is not None and 'agentview_image' in obs:
                frame = np.asarray(obs['agentview_image'])[::-1].copy()
            else:
                frame = session.image()
            from PIL import Image, ImageDraw, ImageOps
            picture = Image.fromarray(frame)
            picture = ImageOps.pad(picture, (480, 360))
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, picture.width, 30), fill='black')
            draw.text((5, 4), self.name+' | simulation t=%.2fs'%sample['time'], fill='white')
            self.writer.append_data(np.asarray(picture))
            self.last_frame = sample['time']
            self.video_seconds += time.monotonic()-started

    def metrics(self, prompt_window=None):
        result = trajectory_metrics(self.samples, self.targets, self.focus, prompt_window)
        result.update(first_action_wall_seconds=self.first_action_wall,
                      first_action_sim_seconds=self.first_action_sim,
                      wall_seconds=time.monotonic()-self.started_wall,
                      video_encoding_wall_seconds=self.video_seconds)
        return result


def trajectory_metrics(samples, targets, focus='source', prompt_window=None):
    """Keep elapsed time, approach, grasp and wrong contacts separate.

    A goalward response requires >=1 mm of robot movement projected toward
    the live target across a >=0.25 s window, with >=50% of path length toward
    it. Moving objects cannot create credit for stationary robot movement.
    This is a measurable proxy; the stored videos show the complete behavior.
    """
    if not samples:
        raise ValueError('Trajectory needs an initial sample')
    if prompt_window is not None and (not np.isfinite(prompt_window) or prompt_window <= 0):
        raise ValueError('Prompt-response window must be positive or None')
    source = targets['source']
    allowed = {source, targets['destination_body']}
    first_success = next((s['time'] for s in samples if s['success']), None)
    last_false = max((i for i, s in enumerate(samples) if not s['success']), default=-1)
    success = samples[last_false+1]['time'] if samples[-1]['success'] else None
    contact_index = next((i for i, s in enumerate(samples) if source in s['contacts']), None)
    contact = None if contact_index is None else samples[contact_index]['time']
    grasp_index = next((i for i, s in enumerate(samples) if source in s['grasped']), None)
    grasp = None if grasp_index is None else samples[grasp_index]['time']
    wrong_contacts = sorted({n for s in samples for n in s['contacts'] if n not in allowed})
    wrong_grasps = sorted({n for s in samples for n in s['grasped'] if n != source})
    before_grasp = samples if grasp_index is None else samples[:grasp_index+1]
    allowed_before = allowed if source in samples[0]['grasped'] else {source}
    wrong_before = sorted({n for s in before_grasp for n in s['contacts'] if n not in allowed_before})
    first_contact_index = next((i for i, s in enumerate(samples) if s['contacts']), None)
    positions = np.asarray([s['eef'] for s in samples])
    delta = np.diff(positions, axis=0)
    length = np.linalg.norm(delta, axis=1)
    projected = []
    for i, step in enumerate(delta):
        direction = np.asarray(samples[i]['focus_position'])-positions[i]
        norm = np.linalg.norm(direction)
        projected.append(0. if norm < 1e-9 else float(step@direction/norm))
    projected = np.asarray(projected)
    already_near = (samples[0]['focus_distance'] <= .05 or
                    focus == 'source' and source in samples[0]['contacts'])
    response = 0. if already_near else None
    confirmed = response
    response_index = 0 if response is not None else None
    confirmed_index = response_index
    if response is None:
        end_of_approach = grasp_index+1 if focus == 'source' and grasp_index is not None else len(samples)
        for end in range(1, end_of_approach):
            starts = [i for i in range(end) if samples[end]['time']-samples[i]['time'] >= .25-1e-8]
            if not starts:
                continue
            start = starts[-1]
            travel = float(length[start:end].sum())
            progress = float(projected[start:end].sum())
            active = np.flatnonzero(projected[start:end] > .00005)
            if (progress >= .001 and progress >= .5*travel and len(active) >= 3
                    and len(active) >= .6*(end-start)):
                response = samples[start+int(active[0])]['time']
                confirmed = samples[end]['time']
                response_index, confirmed_index = start+int(active[0])+1, end
                break
    early = np.array([s['time'] <= 1.+1e-8 for s in samples[1:]], dtype=bool)
    approach_projected = projected[:grasp_index] if focus == 'source' and grasp_index is not None else projected
    early_approach = early[:len(approach_projected)]
    centimetre = np.flatnonzero(np.cumsum(np.maximum(approach_projected, 0.)) >= .01)
    centimetre_time = None if not len(centimetre) else samples[int(centimetre[0])+1]['time']
    path_to_grasp = None if grasp_index is None else float(length[:grasp_index].sum())
    displacement = (None if grasp_index is None else
                    float(np.linalg.norm(positions[grasp_index]-positions[0])))
    detour_ratio = (None if displacement is None or displacement < .01 else path_to_grasp/displacement)
    prompt = (None if prompt_window is None else
              confirmed is not None and confirmed <= prompt_window+1e-8)
    def wall(index):
        return None if index is None else samples[index].get('wall_seconds')
    grasp_to_completion = (None if success is None or grasp is None or success < grasp
                           else success-grasp)
    return dict(focus=focus, success=success is not None,
        completion_sim_seconds=success, first_goal_success_seconds=first_success,
        completion_wall_seconds=wall(last_false+1) if success is not None else None,
        elapsed_sim_seconds=samples[-1]['time'],
        initial_source_distance_metres=samples[0]['source_distance'],
        source_already_grasped=source in samples[0]['grasped'],
        focus_already_near_or_contact=bool(already_near),
        first_goalward_motion_seconds=response, goalward_response_confirmed_seconds=confirmed,
        first_observed_goalward_motion_wall_seconds=wall(response_index),
        goalward_response_confirmed_wall_seconds=wall(confirmed_index),
        first_centimetre_goalward_seconds=centimetre_time,
        prompt_window_seconds=prompt_window, prompt_goalward_response=prompt,
        first_correct_contact_seconds=contact, first_correct_grasp_seconds=grasp,
        correct_grasp_to_completion_seconds=grasp_to_completion,
        path_after_correct_grasp_metres=(None if grasp_to_completion is None else
                                        float(length[grasp_index:last_false+1].sum())),
        first_object_contact_seconds=None if first_contact_index is None else samples[first_contact_index]['time'],
        first_object_contact_names=[] if first_contact_index is None else samples[first_contact_index]['contacts'],
        first_correct_contact_wall_seconds=wall(contact_index), first_correct_grasp_wall_seconds=wall(grasp_index),
        wrong_objects_contacted=wrong_contacts, wrong_objects_grasped=wrong_grasps,
        wrong_object_contacts_before_correct_grasp=wrong_before,
        first_second_goalward_metres=float(np.maximum(approach_projected[early_approach], 0.).sum()),
        first_second_away_metres=float(np.maximum(-approach_projected[early_approach], 0.).sum()),
        path_metres=float(length.sum()), path_to_correct_grasp_metres=path_to_grasp,
        path_to_grasp_detour_ratio=detour_ratio,
        prompt_correct_object_recovery=(None if prompt is None else
                                       bool(prompt and not wrong_before and not wrong_grasps)))


def timing_comparisons(results):
    baselines = {(r['suite'], r['task_id'], r['scene'], r['mode']): r
                 for r in results if r.get('mode') in ('normal_policy', 'normal_waypoints')}
    output = []
    for r in results:
        row = dict(trial_id=r['trial_id'], suite=r['suite'], task_id=r['task_id'],
                   scene=r['scene'], mode=r.get('mode', 'recovery'), kind=r.get('kind'),
                   intensity=r.get('intensity'), status=r['status'], success=r.get('success', False))
        for key in ('layout_id', 'initial_state_sha256', 'initial_state_matches_layout'):
            row[key] = r.get(key)
        metrics = r.get('motion_metrics', {})
        for key in ('completion_sim_seconds', 'first_action_wall_seconds',
                    'completion_wall_seconds',
                    'focus', 'source_already_grasped', 'focus_already_near_or_contact',
                    'first_goalward_motion_seconds', 'goalward_response_confirmed_seconds',
                    'first_centimetre_goalward_seconds',
                    'first_observed_goalward_motion_wall_seconds',
                    'first_correct_contact_seconds', 'first_correct_grasp_seconds',
                    'correct_grasp_to_completion_seconds', 'path_after_correct_grasp_metres',
                    'first_object_contact_seconds',
                    'first_correct_contact_wall_seconds', 'first_correct_grasp_wall_seconds',
                    'prompt_correct_object_recovery', 'path_to_grasp_detour_ratio'):
            row[key] = metrics.get(key)
        row['wrong_objects_contacted'] = ', '.join(metrics.get('wrong_objects_contacted', []))
        row['wrong_objects_grasped'] = ', '.join(metrics.get('wrong_objects_grasped', []))
        row['first_object_contact_names'] = ', '.join(metrics.get('first_object_contact_names', []))
        row['wrong_contacts_before_correct_grasp'] = ', '.join(metrics.get('wrong_object_contacts_before_correct_grasp', []))
        row['failure_to_completion_sim_seconds'] = r.get('failure_to_completion_sim_seconds')
        row['total_task_sim_seconds'] = r.get('total_task_sim_seconds')
        row['handoff_delay_sim_seconds'] = r.get('actual_delay_seconds')
        row['policy_steps'] = r.get('policy_result', {}).get('steps')
        row['wall_seconds_with_logging'] = r.get('wall_seconds')
        for mode in ('normal_policy', 'normal_waypoints'):
            baseline = baselines.get((r['suite'], r['task_id'], r['scene'], mode), {})
            row[mode+'_completion_sim_seconds'] = baseline.get('motion_metrics', {}).get('completion_sim_seconds')
            row[mode+'_correct_grasp_sim_seconds'] = baseline.get('motion_metrics', {}).get('first_correct_grasp_seconds')
        normal = row['normal_waypoints_completion_sim_seconds']
        row['extra_over_normal_waypoints_sim_seconds'] = (None if normal is None or
            row['total_task_sim_seconds'] is None else row['total_task_sim_seconds']-normal)
        normal_grasp = row['normal_policy_correct_grasp_sim_seconds']
        row['correct_grasp_delay_vs_normal_policy_seconds'] = (None if normal_grasp is None or
            row['first_correct_grasp_seconds'] is None else row['first_correct_grasp_seconds']-normal_grasp)
        output.append(row)
    return output


def export_timing_csv(folder, results):
    rows = timing_comparisons(results)
    if not rows:
        return
    folder = Path(folder)
    temporary = folder/'timing.csv.tmp'
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(folder/'timing.csv')


def derive_results(folder, prompt_window=None, write=False):
    """Derive final timing definitions from trajectories; leave live data alone by default."""
    from waypoint_recovery_benchmark import summarize, write_json
    folder = Path(folder)
    results = json.loads((folder/'results.json').read_text())
    for row in results:
        trial = folder/'trials'/row['trial_id']
        initial = json.loads((trial/'initial.state.json').read_text())['state']
        row['layout_id'] = '{suite}_t{task_id:03d}_{scene}'.format(**row)
        row['initial_state_sha256'] = hashlib.sha256(json.dumps(initial).encode()).hexdigest()
        layout_path = folder/'layouts'/(row['layout_id']+'.json')
        if layout_path.exists():
            layout = json.loads(layout_path.read_text())['state']
            row['initial_state_matches_layout'] = initial == layout
        traces = {}
        for phase, key in [('waypoints', 'waypoint_motion_metrics'), ('policy', 'motion_metrics')]:
            path = trial/(phase+'.trajectory.json')
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            traces[phase] = data
            metrics = trajectory_metrics(data['samples'], data['targets'], data['focus'], prompt_window)
            for name in ('first_action_wall_seconds', 'first_action_sim_seconds', 'wall_seconds',
                         'video_encoding_wall_seconds'):
                metrics[name] = data.get(name)
            row[key] = metrics
        if row['mode'] == 'normal_waypoints' and 'waypoints' in traces:
            row['motion_metrics'] = row['waypoint_motion_metrics']
            row['success'] = row['motion_metrics']['success']
            row['status'] = 'normal_waypoints_success' if row['success'] else 'normal_waypoints_failed'
            row['total_task_sim_seconds'] = row['motion_metrics']['completion_sim_seconds']
        elif 'policy' in traces:
            completed = row['motion_metrics']['completion_sim_seconds']
            if row['mode'] == 'normal_policy':
                row['total_task_sim_seconds'] = completed
            elif completed is not None:
                completed += traces['policy']['started_sim_time']
                row['total_task_sim_seconds'] = completed-traces['waypoints']['started_sim_time']
                row['failure_to_completion_sim_seconds'] = completed-row['failure_event']['sim_time']
        if write:
            write_json(trial/'result.json', row)
    if write:
        write_json(folder/'results.json', results)
        write_json(folder/'summary.json', summarize(results))
        export_timing_csv(folder, results)
    return results


def recompute_metrics(folder, prompt_window=None):
    return derive_results(folder, prompt_window, write=True)
