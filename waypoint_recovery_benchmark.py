"""Measure live Pi 0.5 recovery from waypoint faults in paired object layouts.

One fault is armed per episode. The waypoint controller continues for two
simulation seconds after a measured fault, then the policy receives the SAME
scene. Missed injections are reported separately from unsuccessful recovery.

Use --images-only for all-task, original/shuffled trials with immediate and
5-second handoffs, a 4-second stable-success check, and only a CSV and two
controller image folders. The original experiment format remains available.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from itertools import product
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import subgoal_planner as sp
from waypoint_failures import (FAILURES, INTENSITIES, PENETRATION, HIGH_COLLISION_OFFSET, Injector,
                               TILT_AXES, TILT_BANDS, allowed_intensities, collision_target,
                               held_object, tilt, waypoint_primitive)


ROOT = Path(__file__).resolve().parent
TASKS = ([('libero_goal', i) for i in (1, 2, 4, 6, 8, 9)] +
         [('libero_object', i) for i in (3, 5, 6, 7)])


class Takeover(Exception):
    """Unwind waypoint execution at a control boundary without resetting it."""


class EpisodeLimit(Exception):
    pass


class HandoffClock:
    def __init__(self, delay):
        if not np.isfinite(delay) or delay < 0:
            raise ValueError('Handoff delay must be finite and nonnegative')
        self.delay, self.event = float(delay), None

    def mark(self, sim_time, **details):
        if self.event is None:
            self.event = dict(sim_time=float(sim_time), **details)

    def due(self, sim_time):
        return (self.event is not None and
                sim_time + 1e-8 >= self.event['sim_time'] + self.delay)


class MeasuredInjector(Injector):
    """Observe actual contact, deviation, open fingers or loss of a held object."""
    def __init__(self, session, injections, clock, waypoints, seed=None):
        super().__init__(session, injections, waypoints=waypoints, seed=seed)
        self.clock = clock
        self.watch = None
        self.fault_targets = []
        self.robot_geoms = set(session.world._geoms_of('robot0'))
        self.robot_geoms.update(session.world._geoms_of('gripper0'))
        # Robot body names include robot0_link..., not robot0_link_... .
        import mujoco
        for g in range(session.world.m.ngeom):
            name = mujoco.mj_id2name(session.world.m, mujoco.mjtObj.mjOBJ_BODY,
                                    session.world.m.geom_bodyid[g]) or ''
            if name.startswith(('robot0', 'gripper0')):
                self.robot_geoms.add(g)

    def mark(self, kind, detail, **measurements):
        self.clock.mark(self.session.env.sim.data.time, row=self.row,
                        primitive=self.primitive, kind=kind, intensity=self.intensity,
                        detail=detail, **measurements)

    def _faulty(self, kind, original, position, orientation, args=(), kwargs=None):
        position = np.asarray(position, dtype=float)
        orientation = sp.DOWN if orientation is None else np.asarray(orientation)
        if kind == 'collision':
            name, _ = collision_target(self.session.world, self.session.world.eef(),
                position, ignore=(held_object(self.session.world),),
                penetration=PENETRATION / 3 if self.intensity == 'low' else PENETRATION)
            self.watch = dict(kind=kind, body=name,
                geoms=set(self.session.world._geoms_of(name)) if name else set(),
                active=self.intensity == 'low')
        elif kind == 'transition':
            changed = self.deviated(position)
            distance = self.transition_distance()
            self.watch = dict(kind=kind, position=position, start=self.session.world.eef().copy(),
                              changed=changed, direction=(changed-position)/distance, distance=distance)
        elif kind == 'orientation':
            axis, angle = self.wrist_tilt()
            changed = tilt(orientation, angle, axis)
            self.watch = dict(kind=kind, position=position, orientation=orientation, axis=axis,
                              changed=changed, angles=np.atleast_1d(angle),
                              angle=float(Rotation.from_matrix(orientation.T@changed).magnitude()))
        if self.watch is not None:
            self.fault_targets.append({key: value.tolist() if isinstance(value, np.ndarray) else value
                                       for key, value in self.watch.items() if key != 'geoms'})
        try:
            return super()._faulty(kind, original, position, orientation, args, kwargs)
        finally:
            self.watch = None

    def prevent_grasp(self):
        if not held_object(self.session.world):
            self.mark('grasp_loss', 'close command suppressed while the hand was empty')
        return super().prevent_grasp()

    def collision_push_pose(self):
        position, orientation = super().collision_push_pose()
        # High first stages behind the obstacle. Start observing the collision
        # when its forward stroke begins, so takeover follows the actual impact.
        self.watch['active'] = True
        self.fault_targets.append(dict(kind='collision', phase='gripper_forward_push',
            body=self.watch['body'], position=position.tolist(),
            orientation=orientation.tolist(), requested_forward_offset_metres=HIGH_COLLISION_OFFSET))
        return position, orientation

    def release(self):
        name = held_object(self.session.world)
        if name:
            self.watch = dict(kind='grasp_loss', object=name)
        try:
            return super().release()
        finally:
            self.watch = None

    def observe_physics(self):
        """Contact can last less than one control step; observe each physics step."""
        if (self.clock.event is not None or not self.watch or self.watch['kind'] != 'collision'
                or not self.watch.get('active', True)):
            return
        data = self.session.world.d
        for contact in data.contact[:data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if contact.dist > 0:
                continue
            external = g2 if g1 in self.robot_geoms else g1 if g2 in self.robot_geoms else None
            if external is not None and (not self.watch['geoms'] or external in self.watch['geoms']):
                self.mark('collision', 'physical robot contact with the injected obstacle',
                          body=self.watch['body'], penetration_metres=float(-contact.dist))
                return

    def observe_control(self):
        if self.clock.event is not None or not self.watch:
            return
        kind = self.watch['kind']
        if kind == 'transition':
            position = self.session.world.eef()
            residual = position-self.watch['position']
            error = float(residual @ self.watch['direction'])
            # For a forward/backward draw, normal approach motion can project
            # onto the same direction. Require arrival near the faulty target too.
            target_gap = float(np.linalg.norm(position-self.watch['changed']))
            if error >= .8*self.watch['distance'] and target_gap <= .2*self.watch['distance']:
                self.mark(kind, 'measured 2D X/Y deviation reached 80% of the injected offset',
                          offset_metres=error, requested_metres=self.watch['distance'],
                          requested_plane='xy',
                          requested_direction=self.watch['direction'].tolist(),
                          target_gap_metres=target_gap)
        elif kind == 'orientation':
            position, rotation = self.session.gripper_pose()
            nominal = float(Rotation.from_matrix(self.watch['orientation'].T@rotation).magnitude())
            changed = float(Rotation.from_matrix(self.watch['changed'].T@rotation).magnitude())
            if (np.linalg.norm(position-self.watch['position']) < .04 and
                    nominal >= .5*self.watch['angle'] and changed <= .35*self.watch['angle']):
                self.mark(kind, 'measured wrist reached the injected orientation',
                          orientation_degrees=float(np.degrees(nominal)),
                          requested_degrees=float(np.degrees(self.watch['angle'])),
                          requested_axis=', '.join(TILT_AXES[a] for a in self.watch['axis']),
                          requested_axes_degrees={TILT_AXES[a]:float(np.degrees(value))
                              for a,value in zip(self.watch['axis'],self.watch['angles'])})
        elif kind == 'grasp_loss':
            if self.session.executor.gripper == sp.GRIPPER_OPEN and not held_object(self.session.world):
                self.mark(kind, 'the previously held object lost finger contact during opening',
                          object=self.watch['object'])


@contextmanager
def observe_physics(session, injector):
    sim = session.env.sim
    original = sim.step
    def step():
        result = original()
        injector.observe_physics()
        return result
    sim.step = step
    try:
        yield
    finally:
        sim.step = original


@contextmanager
def suspend_images(session):
    domain = session.env.env
    saved = {name: (obs.is_enabled(), obs.is_active()) for name, obs in domain._observables.items()
             if name.endswith('_image')}
    try:
        for name in saved:
            domain.modify_observable(name, 'enabled', False)
            domain.modify_observable(name, 'active', False)
        yield
    finally:
        for name, (enabled, active) in saved.items():
            domain.modify_observable(name, 'enabled', enabled)
            domain.modify_observable(name, 'active', active)


def select_row(chain, kind, intensity):
    candidates = [i for i, w in enumerate(chain.waypoints)
                  if intensity in allowed_intensities(kind, waypoint_primitive(w), w.motion)]
    if not candidates:
        raise ValueError('No applicable waypoint for '+kind+'/'+intensity)
    if kind == 'grasp_loss':
        return candidates[-1] if intensity == 'high' else candidates[0]
    # A grasp descent exposes faults before the placement can satisfy the task.
    return next((i for i in candidates if waypoint_primitive(chain.waypoints[i]) == 'grasp'), candidates[0])


def feasible_row(session, chain, spec):
    """Prefer rows that can execute the High band's combined corner rotations."""
    preferred = select_row(chain, spec['kind'], spec['intensity'])
    if spec['kind'] != 'orientation' or spec['intensity'] != 'high':
        return preferred
    targets = session.preview(chain)
    candidates = [preferred] + [i for i, w in enumerate(chain.waypoints) if i != preferred
        and spec['intensity'] in allowed_intensities(spec['kind'], waypoint_primitive(w), w.motion)]
    for row in candidates:
        start, start_rotation = targets[row-1] if row else session.gripper_pose()
        end, rotation = targets[row]
        steps = max(2, int(np.ceil(np.linalg.norm(np.asarray(end)-start)/session.args.line_step)))
        executable = True
        for angles in product(TILT_BANDS['high'], repeat=3):
            curve = Slerp([0, 1], Rotation.from_matrix(
                [start_rotation, tilt(rotation, np.radians(angles), 'xyz')]))
            q, error = session.executor.scene.ik(start, target_ori=start_rotation)
            executable = error <= .02
            for i in range(1, steps+1):
                q, error = session.executor.scene.ik(np.asarray(start)+(np.asarray(end)-start)*i/steps,
                                                    q_init=q, target_ori=curve(i/steps).as_matrix())
                if error > .02:
                    executable = False
                    break
            if not executable:
                break
        if executable:
            return row
    return preferred


def scenarios(tasks=TASKS):
    return [dict(suite=suite, task_id=task, scene=scene, kind=kind, intensity=intensity,
                 seed=1000+index)
            for index, (suite, task) in enumerate(tasks)
            for scene in ('original', 'shuffled')
            for kind in FAILURES for intensity in INTENSITIES]


def timed_scenarios(tasks=TASKS):
    result = []
    for index, (suite, task) in enumerate(tasks):
        for scene in ('original', 'shuffled'):
            for mode in ('normal_waypoints', 'normal_policy'):
                result.append(dict(suite=suite, task_id=task, scene=scene, mode=mode, seed=1000+index))
        result.extend(dict(s, mode='recovery', seed=1000+index)
                      for s in scenarios([(suite, task)]))
    return result


def trial_id(spec):
    if spec.get('mode', 'recovery') != 'recovery':
        return '{suite}_t{task_id:03d}_{scene}_{mode}'.format(**spec)
    return '{suite}_t{task_id:03d}_{scene}_{kind}_{intensity}'.format(**spec)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_scene(session, folder, name):
    import imageio
    write_json(folder/(name+'.state.json'), dict(state=session.env.sim.get_state().flatten().tolist(),
                                               sim_time=float(session.env.sim.data.time)))
    imageio.imwrite(folder/(name+'.png'), session.image())


def propagated_tail(chain, targets, row, nominal, actual, kind):
    """Keep the planned relative sequence in the reached faulty wrist frame."""
    from waypoint_chain import WORLD
    p0, r0 = map(np.asarray, nominal)
    p1, r1 = map(np.asarray, actual)
    rotation = r1 @ r0.T if kind == 'orientation' else np.eye(3)
    remaining, commands = [], []
    for i in range(row+1, len(chain.waypoints)):
        p, r = map(np.asarray, targets[i])
        moved_p, moved_r = p1 + rotation @ (p-p0), rotation @ r
        w = chain.waypoints[i]
        changes = dict(reference=WORLD, offset=moved_p.tolist(), orientation=moved_r.tolist(),
                       orientation_relative_to_previous=False, carry=False, seat=False, hold_xy=False)
        if hasattr(w, 'learned'):
            changes['learned'] = {}
        remaining.append(replace(w, **changes))
        commands.append(dict(row=i, nominal_position=p.tolist(), nominal_orientation=r.tolist(),
                             position=moved_p.tolist(), orientation=moved_r.tolist()))
    changes = dict(waypoints=remaining)
    if hasattr(chain, 'execution_skill'):
        changes['execution_skill'] = {}
    return replace(chain, **changes), dict(kind=kind, anchor_row=row,
        nominal_position=p0.tolist(), nominal_orientation=r0.tolist(),
        actual_position=p1.tolist(), actual_orientation=r1.tolist(),
        rotation=rotation.tolist(), targets=commands)


def run_fault(session, chain, spec, delay, max_sim_seconds, folder, recorder=None,
              injector_class=None, intensity_rule=None, capture=None, progress=None,
              handoff_after_waypoint=False, injector_options=None, handoff_at_end=False,
              propagate_deviation=False):
    """Observe a fault and hand off from the same simulation state.

    Timed live delays start at the selected waypoint's usual handoff boundary.
    End-of-plan live mode runs the tail without a countdown or final hold.
    Historical batch mode keeps its failure-onset delay.
    """
    if handoff_at_end and (not handoff_after_waypoint or delay):
        raise ValueError('End-of-plan handoff requires live mode without a timed delay')
    injector_class = injector_class or MeasuredInjector
    intensity_rule = intensity_rule or allowed_intensities
    capture = capture or save_scene
    row = spec.get('injection_row')
    if row is None:
        row = feasible_row(session, chain, spec)
    if not isinstance(row, int) or not 0 <= row < len(chain.waypoints):
        raise ValueError('Injection row is outside the waypoint plan')
    waypoint = chain.waypoints[row]
    if spec['intensity'] not in intensity_rule(spec['kind'], waypoint_primitive(waypoint), waypoint.motion):
        raise ValueError('Failure is not allowed in the selected primitive')
    clock = HandoffClock(delay)
    if recorder is not None:
        recorder.video_gate = lambda: clock.event is not None
    injector = injector_class(session, {row: dict(kind=spec['kind'], intensity=spec['intensity'])},
                              clock, chain.waypoints, **(injector_options or {}))
    if 'fault_seed' in spec:
        injector.random = np.random.default_rng(spec['fault_seed'])
    start = float(session.env.sim.data.time)
    trace = []
    previous_tick = session.tick
    previous_continue = session.continue_on_failure
    missing = object()
    previous_stop = getattr(session, 'stop_after_waypoint', missing)
    previous_finished = getattr(session, 'on_waypoint_finished', missing)
    selected_started = timed_out = False
    completion = None
    latency_start = latency_pose = None
    latency_second = None
    plan_finished = False
    propagation = None
    sequence_targets = (session.preview(chain) if propagate_deviation
                        and spec['kind'] in ('transition', 'orientation')
                        and (handoff_at_end or delay > 0) else None)

    class ContinueSequence(Exception):
        """Leave adaptive feedback before it corrects the injected deviation."""

    def continue_sequence():
        if (sequence_targets is not None and propagation is None
                and row+1 < len(chain.waypoints) and clock.event is not None
                and getattr(injector, 'nominal_fault_pose', None) is not None):
            raise ContinueSequence()

    def begin_latency():
        nonlocal latency_start, latency_pose
        if handoff_after_waypoint and (delay > 0 or handoff_at_end) and latency_start is None:
            latency_start = float(session.env.sim.data.time)
            pose = getattr(session, 'gripper_pose', None)
            if pose is not None:
                position, rotation = pose()
                latency_pose = dict(position=np.asarray(position).tolist(),
                                    orientation=np.asarray(rotation).tolist())

    def latency_progress(now):
        nonlocal latency_second
        if latency_start is not None and progress is not None:
            if handoff_at_end:
                if latency_second is None:
                    progress(row, 'Detection latency · continuing to end of waypoint plan')
                    latency_second = 0
                return
            remaining = max(0., latency_start+delay-now)
            second = int(np.ceil(remaining))
            if second != latency_second:
                progress(row, 'Detection latency · %.1f s remaining · continuing waypoint plan' % remaining)
                latency_second = second

    def tick():
        nonlocal failure_saved
        previous_tick()
        injector.observe_control()
        if clock.event is not None and not failure_saved:
            capture(session, folder, 'failure')
            failure_saved = True
        now = float(session.env.sim.data.time)
        if not handoff_after_waypoint and clock.due(now):
            raise Takeover()
        if handoff_at_end:
            if now-start+1e-8 >= max_sim_seconds:
                raise EpisodeLimit('Waypoint plan time limit reached before end-of-plan handoff')
        elif latency_start is not None:
            if now+1e-8 >= latency_start+delay:
                raise Takeover()
            latency_progress(now)
        elif (clock.event is None or handoff_after_waypoint) and now-start >= max_sim_seconds:
            raise EpisodeLimit('Waypoint time limit reached' if handoff_after_waypoint else
                               'No measured failure before the waypoint time limit')

    def report(index, message):
        nonlocal outcome, selected_started
        if handoff_after_waypoint and index > row:
            if delay == 0 and not handoff_at_end:
                raise Takeover()
            # An aborted Grasp can reach the continuation boundary without a
            # close/check completion callback. This was its usual handoff point.
            begin_latency()
            continue_sequence()
            if not handoff_at_end and session.env.sim.data.time+1e-8 >= latency_start+delay:
                raise Takeover()
        if recorder is not None:
            recorder.current_row = index
        if message == 'Running':
            injector.start_row(index)
            if index == row:
                selected_started = True
        trace.append(dict(row=index, result=message, sim_time=float(session.env.sim.data.time)))
        # Status text, skill return and the next row's boundary can all follow
        # an aborted move. None of them proves the final Grasp check happened.
        if handoff_after_waypoint and index == row and message != 'Running':
            outcome = message
        if progress is not None:
            progress(index, message)

    def waypoint_finished(index, message):
        nonlocal completion, outcome
        if index == row and selected_started:
            finish = getattr(injector, 'finish_waypoint', None)
            if finish is not None:
                terminal_result = finish(index, message)
                if terminal_result is not None:
                    message = terminal_result
                    report(index, message)
            completion = dict(row=index, result=message,
                              sim_time=float(session.env.sim.data.time))
            outcome = message
            if delay > 0 or handoff_at_end:
                # Detection is delayed from the normal Continue-VLA boundary.
                # Allow the remaining plan to move from the failed scene.
                begin_latency()
                latency_progress(float(session.env.sim.data.time))
                continue_sequence()
            else:
                raise Takeover()

    session.tick = tick
    session.continue_on_failure = True
    interrupted = False
    outcome = None
    failure_saved = False
    try:
        if handoff_after_waypoint:
            session.stop_after_waypoint = row if delay == 0 and not handoff_at_end else None
            session.on_waypoint_finished = waypoint_finished
        with suspend_images(session), observe_physics(session, injector), injector:
            try:
                try:
                    try:
                        outcome = session.run(chain, report)
                    except ContinueSequence:
                        # Use the pre-injection targets once, anchored at the
                        # reached faulty pose. Adaptive grasp compensation and
                        # object-relative target refresh must not undo it.
                        from waypoint_session import Session
                        tail, propagation = propagated_tail(chain, sequence_targets, row,
                            injector.nominal_fault_pose, session.gripper_pose(), spec['kind'])
                        session.on_waypoint_finished = lambda i, text: waypoint_finished(row+1+i, text)
                        try:
                            outcome = Session.run(session, tail, lambda i, text: report(row+1+i, text))
                        finally:
                            session.on_waypoint_finished = waypoint_finished
                    plan_finished = True
                except Takeover:
                    interrupted = True
                except EpisodeLimit as exc:
                    timed_out = True
                    outcome = str(exc)
                if handoff_after_waypoint and handoff_at_end:
                    begin_latency()
                elif handoff_after_waypoint and delay > 0:
                    # If playback ended before the requested interval, keep the
                    # final controller target active for the remaining time.
                    begin_latency()
                    while session.env.sim.data.time+1e-8 < latency_start+delay:
                        latency_progress(float(session.env.sim.data.time))
                        session.executor.hold(1)
                elif clock.event is not None and not handoff_after_waypoint:
                    while not clock.due(session.env.sim.data.time):
                        session.executor.hold(1)
            except Takeover:
                interrupted = True
            except EpisodeLimit as exc:
                timed_out = True
                outcome = str(exc)
    finally:
        session.tick = previous_tick
        session.continue_on_failure = previous_continue
        if handoff_after_waypoint:
            if previous_stop is missing:
                del session.stop_after_waypoint
            else:
                session.stop_after_waypoint = previous_stop
            if previous_finished is missing:
                del session.on_waypoint_finished
            else:
                session.on_waypoint_finished = previous_finished
    data = dict(injection_row=row, failure_event=clock.event, observed_failure=clock.event is not None,
                waypoint_result=outcome, waypoint_interrupted=interrupted,
                injected=injector.fired, missed=injector.missed, trace=trace,
                fault_targets=getattr(injector, 'fault_targets', []))
    if handoff_after_waypoint:
        data['handoff_mode'] = 'end_of_plan' if handoff_at_end else 'after_waypoint'
        data['waypoint_plan_completed'] = plan_finished
        data['propagated_deviation'] = propagation
        data['detection_latency_seconds'] = float(delay)
        data['detection_latency_start_sim_time'] = latency_start
        data['detection_latency_start_pose'] = latency_pose
        data['actual_detection_latency_seconds'] = (float(session.env.sim.data.time)-latency_start
                                                    if latency_start is not None else 0.)
        data['detection_latency_completed'] = (plan_finished if handoff_at_end else delay == 0 or
            (latency_start is not None and session.env.sim.data.time+1e-8 >= latency_start+delay))
        data['waypoint_execution_stopped'] = True
        data['waypoint_end_sim_time'] = float(session.env.sim.data.time)
        data['waypoint_completed'] = completion is not None and not timed_out
        data['waypoint_completion_event'] = completion
        if data['waypoint_completed']:
            data['waypoint_finish_sim_time'] = completion['sim_time']
    if handoff_after_waypoint or clock.event is not None:
        now = float(session.env.sim.data.time)
        data.update(handoff_sim_time=now, success_before_policy=bool(session.env.check_success()))
        if clock.event is not None:
            data['actual_delay_seconds'] = now-clock.event['sim_time']
        capture(session, folder, 'handoff')
    return data


def run_normal_waypoints(session, chain, recorder, max_sim_seconds):
    previous_tick = session.tick
    start = float(session.env.sim.data.time)
    def tick():
        previous_tick()
        if session.env.sim.data.time-start >= max_sim_seconds:
            raise EpisodeLimit('Normal waypoint execution reached its time limit')
    def report(index, message):
        recorder.current_row = index
    session.tick = tick
    session.continue_on_failure = True
    try:
        with suspend_images(session):
            return session.run(chain, report)
    except EpisodeLimit as exc:
        return str(exc)
    finally:
        session.tick = previous_tick
        session.continue_on_failure = False


def prepare_scene(session, chain, spec, layout_root):
    """Freeze each shuffled physical state once, then restore it without settling."""
    from waypoint_replay import initial_state
    model = json.loads(Path(chain.model_path).read_text())
    demo_path = Path(model['demonstration_files'][0])
    demo = json.loads(demo_path.read_text())
    session.training_state = np.asarray(initial_state(demo, demo_path))
    session.training_gripper = sp.GRIPPER_OPEN
    session.restore_training_scene()
    session.last_shuffle_mapping = None
    path = Path(layout_root)/('{suite}_t{task_id:03d}_{scene}.json'.format(**spec))
    if path.exists():
        layout = json.loads(path.read_text())
        if layout['seed'] != spec['seed']:
            raise ValueError('Layout seed does not match the scheduled task')
        session.env.set_init_state(np.asarray(layout['state']))
        session.attach(settle=False)
        session.executor.gripper = sp.GRIPPER_OPEN
        session.last_shuffle_mapping = layout['shuffle_mapping']
    else:
        if spec['scene'] == 'shuffled':
            session.shuffle_scene(seed=spec['seed'])
        write_json(path, dict(state=session.env.sim.get_state().flatten().tolist(),
                             seed=spec['seed'], shuffle_mapping=session.last_shuffle_mapping))


def execute_trial(session, chain, spec, options, folder, prompt_window=None):
    """Restore a paired initial layout and record one complete benchmark attempt."""
    from waypoint_policy import run as run_policy
    from waypoint_recovery_metrics import TrajectoryRecorder
    entry = dict(spec, trial_id=trial_id(spec), success=False)
    identifier = entry['trial_id']
    prepare_scene(session, chain, spec, options.layout_root)
    entry['shuffle_mapping'] = session.last_shuffle_mapping
    entry['language'] = chain.language
    save_scene(session, folder, 'initial')
    episode_start = float(session.env.sim.data.time)
    if spec['mode'] in ('normal_waypoints', 'recovery'):
        with TrajectoryRecorder(session, folder, 'waypoints',
                video=spec['mode']=='recovery' and not options.no_video,
                video_gate=lambda: False, video_name='failure_to_handoff',
                photo_every=getattr(options, 'photo_every', None)) as recorder:
            if spec['mode'] == 'recovery':
                entry.update(run_fault(session, chain, spec, options.delay,
                                       options.max_waypoint_seconds, folder, recorder))
            else:
                entry['waypoint_result'] = run_normal_waypoints(session, chain, recorder,
                                                              options.max_waypoint_seconds)
        entry['waypoint_motion_metrics'] = recorder.metrics(prompt_window)
    if spec['mode'] == 'normal_waypoints':
        entry['motion_metrics'] = entry['waypoint_motion_metrics']
        entry['success'] = entry['motion_metrics']['success']
        entry['status'] = 'normal_waypoints_success' if entry['success'] else 'normal_waypoints_failed'
        entry['total_task_sim_seconds'] = entry['motion_metrics']['completion_sim_seconds']
        save_scene(session, folder, 'terminal')
    elif spec['mode'] == 'recovery' and not entry['observed_failure']:
        entry['status'] = 'failure_not_observed'
        save_scene(session, folder, 'terminal')
    else:
        with TrajectoryRecorder(session, folder, 'policy', not options.no_video,
                photo_every=getattr(options, 'photo_every', None)) as recorder:
            result = run_policy(session, chain.language, port=options.port,
                max_steps=options.max_policy_steps, backend='pi05',
                report=lambda message: print(identifier, message, flush=True))
        entry['policy_result'] = result
        entry['motion_metrics'] = recorder.metrics(prompt_window)
        entry['success'] = result['success']
        if spec['mode'] == 'normal_policy':
            entry['status'] = 'normal_policy_success' if result['success'] else 'normal_policy_failed'
            entry['total_task_sim_seconds'] = entry['motion_metrics']['completion_sim_seconds']
        else:
            entry['status'] = ('already_successful' if entry['success_before_policy'] else
                              'recovered' if result['success'] else 'recovery_failed')
            completed_at = entry['motion_metrics']['completion_sim_seconds']
            if completed_at is not None:
                completed_at += entry['handoff_sim_time']
            entry['total_task_sim_seconds'] = None if completed_at is None else completed_at-episode_start
            entry['failure_to_completion_sim_seconds'] = (None if completed_at is None else
                completed_at-entry['failure_event']['sim_time'])
        save_scene(session, folder, 'terminal')
    return entry


def summarize(results):
    valid = [r for r in results if r.get('status') in ('recovered', 'recovery_failed')]
    groups = {}
    for scene in ('original', 'shuffled'):
        for kind in FAILURES:
            for intensity in INTENSITIES:
                subset = [r for r in valid if (r['scene'], r['kind'], r['intensity']) == (scene, kind, intensity)]
                groups[scene+'/'+kind+'/'+intensity] = dict(trials=len(subset),
                    successes=sum(bool(r.get('success')) for r in subset))
    prompt = [r for r in valid if r.get('motion_metrics', {}).get('prompt_correct_object_recovery') is True]
    scored = [r for r in valid if r.get('motion_metrics', {}).get('prompt_correct_object_recovery') is not None]
    return dict(total=len(results), statuses=dict(Counter(r['status'] for r in results)),
                recovery_trials=len(valid), successes=sum(bool(r.get('success')) for r in valid),
                prompt_correct_object_responses=len(prompt) if scored else None,
                prompt_successes=sum(bool(r.get('success')) for r in prompt) if scored else None, groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments'/
                        ('pi05_failure_recovery_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--port', type=int,
                        help='Policy port (default: 8000 for --images-only, 8002 for legacy experiments)')
    parser.add_argument('--delay', type=float, default=2.)
    parser.add_argument('--max-policy-steps', type=int, default=400)
    parser.add_argument('--max-waypoint-seconds', type=float, default=180.)
    parser.add_argument('--prompt-window', type=float, default=0.,
                        help='Optional seconds to flag a prompt response; default 0 reports raw timing without a cutoff')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--limit', type=int, help='Run only the first N scenarios for a smoke check')
    parser.add_argument('--resume', action='store_true')
    from waypoint_recovery_images import add_arguments, run_images
    add_arguments(parser)
    args = parser.parse_args()
    if args.port is None:
        args.port = 8000 if args.images_only else 8002
    if (args.max_policy_steps < 1 or not np.isfinite(args.max_waypoint_seconds)
            or args.max_waypoint_seconds <= 0 or (args.limit is not None and args.limit < 1)):
        parser.error('Step, time and scenario limits must be positive')
    HandoffClock(args.delay)
    if args.images_only:
        run_images(args, parser)
        return
    if args.dry_run:
        parser.error('--dry-run requires --images-only')
    from waypoint_demo import PolicyConnection
    from serve_pi05_libero import metadata as pi05_metadata
    from waypoint_gmm import GMMChain
    import waypoint_gmm_editor as editor
    from waypoint_recovery_metrics import export_timing_csv

    # Fail before moving any scenes if the requested policy is unavailable.
    client = PolicyConnection(args.port)
    try:
        metadata = client.connect()
    finally:
        client.close()
    expected = pi05_metadata()
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError('Wrong recovery server: '+key+'='+repr(metadata.get(key)))

    if args.prompt_window < 0 or not np.isfinite(args.prompt_window):
        parser.error('Prompt-response window must be finite and nonnegative')
    prompt_window = args.prompt_window or None
    schedule = timed_scenarios()
    if args.limit is not None:
        schedule = schedule[:args.limit]
    plans = []
    for suite, task in TASKS:
        path = ROOT/'saved_plans'/suite/('task_%03d.json'%task)
        plans.append(dict(suite=suite, task_id=task, plan=str(path),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    configuration = dict(version=3, policy=metadata, port=args.port, delay_seconds=args.delay,
        timing='simulation time; observed physics contact or measured control-step fault',
        max_policy_steps=args.max_policy_steps, max_waypoint_seconds=args.max_waypoint_seconds,
        prompt_window_seconds=prompt_window, save_video=not args.no_video,
        video_scope='Normal Pi 0.5 and recovery rollouts, plus failure-to-handoff clips',
        approach_measurement=dict(window_seconds=.25, minimum_projection_metres=.001,
                                  minimum_active_steps=3, minimum_active_fraction=.6,
                                  active_step_projection_metres=.00005),
        placement_rule='Grasp for pose faults; use another eligible row if High orientation fails Cartesian IK preflight',
        pairing='One stored physical simulation state per task and layout, restored without settling',
        tasks=plans, scenarios=schedule)
    manifest = args.output/'manifest.json'
    results_path = args.output/'results.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != configuration:
            parser.error('Existing output requires --resume with the exact same configuration')
    else:
        write_json(manifest, configuration)
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    completed = {r['trial_id'] for r in results}
    args.layout_root = args.output/'layouts'
    editor.MODEL_STORE = args.output/'waypoint_archives'
    session = editor.GMMSession(sp.Args(policy_port=args.port, policy_max_steps=args.max_policy_steps))
    current = None
    try:
        for spec in schedule:
            identifier = trial_id(spec)
            if identifier in completed:
                continue
            started = time.monotonic()
            folder = args.output/'trials'/identifier
            entry = dict(spec, trial_id=identifier, success=False)
            print('START', len(results)+1, '/', len(schedule), identifier, flush=True)
            try:
                key = spec['suite'], spec['task_id']
                if key != current:
                    session.load(*key, cameras=True)
                    current = key
                chain = GMMChain.load(ROOT/'saved_plans'/spec['suite']/('task_%03d.json'%spec['task_id']), *key)
                entry.update(execute_trial(session, chain, spec, args, folder, prompt_window))
            except Exception as exc:
                entry.update(status='error', error=type(exc).__name__+': '+str(exc))
            entry['wall_seconds'] = round(time.monotonic()-started, 3)
            write_json(folder/'result.json', entry)
            results.append(entry)
            write_json(results_path, results)
            write_json(args.output/'summary.json', summarize(results))
            export_timing_csv(args.output, results)
            print('RESULT', len(results), '/', len(schedule), identifier, entry['status'],
                  'success', entry['success'], 'seconds', entry['wall_seconds'], flush=True)
    finally:
        if session.env is not None:
            session.env.close()
    print(json.dumps(summarize(results), indent=2), flush=True)


if __name__ == '__main__':
    main()
