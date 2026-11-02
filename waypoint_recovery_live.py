"""Run waypoint conditions live, keeping scenes and outcomes only in memory.

The GUI uses the benchmark's measured physical injections and controller handoff.
Existing plans, demonstrations and paired layouts are read; no recorder or result
writer is used. GMMSession's automatic replay archives are explicitly disabled.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import secrets
import time

import numpy as np
from scipy.spatial.transform import Rotation

import subgoal_planner as sp
from waypoint_failures import (FAILURES, INTENSITIES, allowed_intensities, held_object,
                               waypoint_primitive, TILT_BANDS)
from waypoint_gmm import GMMChain
from waypoint_gmm_editor import GMMSession
from waypoint_policy import run as run_policy
from waypoint_recovery_benchmark import (EpisodeLimit, MeasuredInjector, Takeover,
                                         run_fault, suspend_images)
from waypoint_replay import initial_state
from waypoint_skill import guarded_move


ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = ROOT / 'experiments/pi05_failure_recovery_timed_paired_20261001'
MODES = {'recovery': 'Failure + policy recovery', 'failure_only': 'Failure only',
         'normal_waypoints': 'Normal waypoints', 'normal_policy': 'Normal policy'}
BACKENDS = {'pi05': 'Pi 0.5'}
TRANSITION_PLAN_ATTEMPTS = 32
TRANSITION_DIRECTION_SEPARATION = np.radians(30.)
COLLISION_DURATION_SECONDS = 3.
COLLISION_XY_BANDS = {'low': (.04, .06), 'high': (.06, .10)}
COLLISION_PRESS_METRES = {'low': .06, 'high': .12}
DETECTION_LATENCY_SECONDS = 10.
ORIENTATION_MIN_YAW_FRACTION = .9
ORIENTATION_MIN_TOTAL_FRACTION = .9
ORIENTATION_BANDS = {key: tuple(2*angle for angle in band) for key, band in TILT_BANDS.items()}
ORIENTATION_XY_BANDS = {'low': (.01, .02), 'high': (.02, .04)}


def discover_plans(folder):
    """List only tasks with an existing runnable saved waypoint plan."""
    plans = []
    for path in sorted(Path(folder).glob('*/task_*.json')):
        data = json.loads(path.read_text())
        if data.get('kind') == 'gmm_editor_chain':
            plans.append(dict(suite=data['suite'], task_id=data['task_id'],
                              language=data['language'], path=path))
    return plans


def grasp_intensities(kind, primitive, motion=None):
    """Pose faults and Low loss use Grasp; High loss uses the next Transport."""
    if kind == 'grasp_loss':
        if primitive == 'grasp' and motion in (None, 'line', 'reach', 'grip'):
            return ('low',)
        if primitive == 'transport' and motion in (None, 'line', 'reach'):
            return ('high',)
        return ()
    if primitive != 'grasp':
        return ()
    return allowed_intensities(kind, primitive, motion)


def eligible_rows(chain, kind, intensity):
    rows = [i for i, w in enumerate(chain.waypoints)
            if intensity in grasp_intensities(kind, waypoint_primitive(w), w.motion)]
    if kind == 'grasp_loss' and intensity == 'high':
        rows = [i for i in rows if i > 0 and waypoint_primitive(chain.waypoints[i-1]) == 'grasp']
    return rows


class GraspOnlyInjector(MeasuredInjector):
    def __init__(self, session, injections, clock, waypoints, seed=None, transition_history=None,
                 collision_history=None):
        super().__init__(session, injections, clock, waypoints, seed)
        self.transition_history = transition_history if transition_history is not None else {}
        self.collision_history = collision_history if collision_history is not None else {}
        self.collision_directions = {}
        self.orientation_offsets = {}

    def permitted_intensities(self, kind, primitive, motion=None):
        return grasp_intensities(kind, primitive, motion)

    def wrist_tilt(self):
        if self.angle is not None:
            return super().wrist_tilt()
        if self.row not in self.tilts:
            # Bound the total rotation, rather than three Euler components
            # whose composition can exceed the selected intensity's limit.
            low, high = ORIENTATION_BANDS[self.intensity]
            # Use the upper end of the allowed band and put most of the
            # rotation into yaw, retaining smaller random roll/pitch changes.
            magnitude = np.radians(self.random.uniform(
                max(low, high*ORIENTATION_MIN_TOTAL_FRACTION), high))
            while True:
                direction = self.random.normal(size=3)
                length = np.linalg.norm(direction)
                if length <= 1e-12:
                    continue
                rotation = Rotation.from_rotvec(direction/length*magnitude)
                angles = rotation.as_euler('xyz')
                # Every run includes strong clockwise or counterclockwise
                # local yaw. Reject tilt-dominated axes without increasing
                # the sampled total angle or adding a second rotation to it.
                if abs(angles[2]) >= magnitude*ORIENTATION_MIN_YAW_FRACTION:
                    break
            self.tilts[self.row] = ('xyz', tuple(angles))
        return self.tilts[self.row]

    def orientation_offset(self):
        """Cache the small horizontal translation paired with this wrist fault."""
        if not hasattr(self, 'orientation_offsets'):
            self.orientation_offsets = {}
        if self.row not in self.orientation_offsets:
            angle = self.random.uniform(0., 2*np.pi)
            distance = self.random.uniform(*ORIENTATION_XY_BANDS[self.intensity])
            self.orientation_offsets[self.row] = distance*np.array([np.cos(angle), np.sin(angle), 0.])
        return self.orientation_offsets[self.row].copy()

    def transition_direction(self):
        cached = self.row in self.directions
        direction = super().transition_direction()
        previous = getattr(self, 'transition_history', {}).get('direction')
        if not cached and previous is not None:
            # Uniformly sample the remaining angular arc, rather than cycling
            # through fixed directions or letting consecutive runs look alike.
            fraction = (np.arctan2(direction[1], direction[0]) % (2*np.pi))/(2*np.pi)
            gap = TRANSITION_DIRECTION_SEPARATION
            angle = np.arctan2(previous[1], previous[0])+gap+fraction*(2*np.pi-2*gap)
            direction = np.array([np.cos(angle), np.sin(angle), 0.])
            self.directions[self.row] = tuple(direction)
        return direction

    def collision_direction(self):
        if self.row not in self.collision_directions:
            angle = self.random.uniform(0., 2*np.pi)
            previous = self.collision_history.get('direction')
            if previous is not None:
                gap = TRANSITION_DIRECTION_SEPARATION
                angle = np.arctan2(previous[1], previous[0])+gap+angle/(2*np.pi)*(2*np.pi-2*gap)
            self.collision_directions[self.row] = np.array([np.cos(angle), np.sin(angle), 0.])
        return self.collision_directions[self.row]

    def mark(self, kind, detail, **measurements):
        if kind == 'orientation' and self.watch and 'transition_offset_metres' in self.watch:
            offset = self.watch['transition_offset_metres']
            measurements.update(transition_offset_metres=offset.tolist(),
                requested_transition_metres=float(np.linalg.norm(offset)),
                measured_transition_metres=float(np.linalg.norm(
                    self.session.world.eef()[:2]-self.watch['nominal_position'][:2])))
        if kind == 'collision' and self.watch and self.watch.get('direction') is not None:
            direction = self.watch['direction']
            local_direction = self.watch['local_direction']
            measurements.update(requested_direction=direction.tolist(), requested_plane='local_xy',
                                requested_local_direction=local_direction.tolist())
            self.collision_history['direction'] = local_direction.tolist()
        super().mark(kind, detail, **measurements)

    def _collision(self):
        session, executor = self.session, self.session.executor
        position, rotation = session.gripper_pose()
        position, rotation = np.asarray(position), np.asarray(rotation)
        # Panda's grip-site +Z points into the work surface in a normal grasp.
        # Use the downward sign of the tool Z axis, rather than accidentally
        # lifting the arm when a user's tool convention calls down "local -Z".
        local_axis = -1. if rotation[2, 2] >= 0. else 1.
        down = local_axis*rotation[:, 2]
        depth = COLLISION_PRESS_METRES[self.intensity]
        dt = session.env.env.control_timestep
        steps = max(2, int(round(COLLISION_DURATION_SECONDS/dt)))
        lateral_steps = max(1, int(round(steps*.25)))
        for sample in range(TRANSITION_PLAN_ATTEMPTS):
            session.tick()
            local_direction = self.collision_direction()
            direction = rotation@local_direction
            distance = self.random.uniform(*COLLISION_XY_BANDS[self.intensity])
            offset = position+direction*distance
            target = offset+down*depth
            path = []
            q = executor.scene.arm_qpos()
            for step in range(1, steps+1):
                session.tick()
                if step <= lateral_steps:
                    fraction = step/lateral_steps
                    fraction = fraction**3*(10-15*fraction+6*fraction*fraction)
                    via = position+direction*distance*fraction
                else:
                    fraction = (step-lateral_steps)/(steps-lateral_steps)
                    fraction = fraction**3*(10-15*fraction+6*fraction*fraction)
                    via = offset+down*depth*fraction
                q, error = executor.scene.ik(via, q_init=q, target_ori=rotation)
                if error > .02 or not np.all(np.isfinite(q)):
                    break
                path.append(np.asarray(q).copy())
            attempt = dict(kind='collision', phase='grasp_press_before_close', sample=sample+1,
                direction=direction.tolist(), xy_offset_metres=float(distance),
                local_direction=local_direction.tolist(),
                start=position.tolist(), offset_target=offset.tolist(), target=target.tolist(),
                press_direction=down.tolist(), grip_site_z_sign=local_axis,
                press_metres=depth, requested_duration_seconds=steps*dt,
                planning='ready' if len(path) == steps else 'unreachable')
            self.fault_targets.append(attempt)
            if len(path) == steps:
                break
            self.collision_directions.pop(self.row, None)
        else:
            return 'unreachable: no pre-close Grasp collision press found after %d random samples' % TRANSITION_PLAN_ATTEMPTS
        held = held_object(session.world)
        own = set(self.robot_geoms)
        if held:
            own.update(session.world._geoms_of(held))
        self.watch = dict(kind='collision', direction=direction, active=False,
                          local_direction=local_direction, pre_close=True, collider_geoms=own)
        attempt['started_sim_time'] = float(session.env.sim.data.time)
        try:
            executor.gripper = sp.GRIPPER_OPEN
            # Stream exactly the scheduled control steps. Contact deliberately
            # prevents arrival; settling/retrying would prolong the press.
            for step, q in enumerate(path, 1):
                self.watch['active'] = step > lateral_steps
                executor.q_hold = q
                executor._apply()
                self.collision_history['direction'] = local_direction.tolist()
            attempt['attempt_result'] = ('ok (injected collision before Grasp close)'
                if self.clock.event is not None else 'failed: press ended without measured scene contact')
            if self.clock.event is not None:
                self.record('collision', 'random local XY offset and downward tool-axis press before Grasp close')
            else:
                self.missed.append('waypoint %d: no measured scene collision during the pre-close Grasp press' % (self.row+1))
            return attempt['attempt_result']
        finally:
            attempt['finished_sim_time'] = float(session.env.sim.data.time)
            attempt['actual_position'] = session.world.eef().tolist()
            self.watch = None

    def observe_physics(self):
        if not self.watch or not self.watch.get('pre_close'):
            return super().observe_physics()
        if self.clock.event is not None or not self.watch['active']:
            return
        world, own = self.session.world, self.watch['collider_geoms']
        for contact in world.d.contact[:world.d.ncon]:
            first, second = int(contact.geom1), int(contact.geom2)
            if contact.dist > 0. or (first in own) == (second in own):
                continue
            external = second if first in own else first
            body = world.m.body(int(world.m.geom_bodyid[external])).name
            self.mark('collision', 'physical scene contact during the pre-close Grasp press',
                      body=body, penetration_metres=float(-contact.dist))
            return

    def observe_control(self):
        if self.clock.event is None and self.watch and self.watch['kind'] == 'orientation':
            _, rotation = self.session.gripper_pose()
            measured = float(np.degrees(Rotation.from_matrix(self.watch['orientation'].T@rotation).magnitude()))
            low, high = ORIENTATION_BANDS[self.intensity]
            if not low <= measured <= high:
                return
            if 'transition_offset_metres' in self.watch:
                position = self.session.world.eef()
                offset = self.watch['transition_offset_metres']
                distance = float(np.linalg.norm(offset))
                projected = float((position-self.watch['nominal_position']) @ (offset/distance))
                # The base orientation observer still requires arrival within
                # 4 cm of the combined target. Contact can prevent a perfect
                # grasp endpoint; require a real XY shift without imposing a
                # second, tighter endpoint tolerance on this failed grasp.
                if projected < .5*distance:
                    return
        if self.clock.event is None and self.watch and self.watch['kind'] == 'transition':
            # Detect departure from the nominal approach, rather than waiting
            # for arrival at a possibly unreachable 15–25 cm displaced target.
            start, end = self.watch['start'], self.watch['position']
            position = self.session.world.eef()
            segment = end-start
            length2 = float(segment @ segment)
            fraction = 0. if length2 < 1e-12 else float(np.clip((position-start) @ segment/length2, 0., 1.))
            residual = position-(start+fraction*segment)
            error = float(residual @ self.watch['direction'])
            threshold = min(.01, .1*self.watch['distance'])
            if error >= threshold:
                self.mark('transition', 'measured departure from the nominal Grasp approach',
                    offset_metres=error, onset_threshold_metres=threshold,
                    requested_metres=self.watch['distance'], requested_plane='xy',
                    requested_direction=self.watch['direction'].tolist())
                history = getattr(self, 'transition_history', None)
                if history is not None:
                    # Remember an actually executed direction even if the user
                    # stops before Continue VLA. Planning rejections do not count.
                    history['direction'] = self.watch['direction'].tolist()
            return
        super().observe_control()

    def _faulty(self, kind, original, position, orientation, args=(), kwargs=None):
        if kind in ('transition', 'orientation'):
            self.nominal_fault_pose = (np.asarray(position).copy(),
                np.asarray(sp.DOWN if orientation is None else orientation).copy())
        if kind == 'collision':
            # Keep the fingers open throughout the descent and collision. The
            # caller closes/checks only after this whole Grasp motion returns.
            self.session.executor.gripper = sp.GRIPPER_OPEN
            approach = original(position, orientation, *args, **(kwargs or {}))
            if not approach.startswith('ok'):
                return approach
            # Mark the displaced pose intentional so arrival checks cannot
            # undo the collision with a correction to the nominal waypoint.
            return self._delivered(kind, self._collision())
        attempt = None
        if kind == 'transition':
            def original(target, rotation, *move_args, **move_kwargs):
                nonlocal attempt
                # Route to the full random endpoint with all scene obstacles
                # included. A blocked endpoint never falls back to unchecked IK.
                held = held_object(self.session.world)
                executor = self.session.executor
                previous_grace = executor.arrival_grace_seconds
                try:
                    # Use follow()'s bounded control-step settling budget. The
                    # editor's one wall-second failure grace can expire while
                    # a valid route is still converging under live rendering.
                    executor.arrival_grace_seconds = None
                    result, route = guarded_move(self.session, target, rotation,
                        ignore=(), carried=(held,) if held else (), plan_first=True)
                finally:
                    executor.arrival_grace_seconds = previous_grace
                if result is None:
                    result = 'unreachable: no checked route to the displaced target'
                attempt = dict(kind='transition', planning=route,
                    target=np.asarray(target).tolist(), attempt_result=result,
                    actual_position=self.session.world.eef().tolist(),
                    target_gap_metres=float(np.linalg.norm(self.session.world.eef()-target)),
                    finished_sim_time=float(self.session.env.sim.data.time))
                self.fault_targets.append(attempt)
                return result
        resample_target = kind in ('transition', 'orientation')
        samples = TRANSITION_PLAN_ATTEMPTS if resample_target else 1
        for sample in range(samples):
            if resample_target:
                # Keep Pause/Stop responsive even when every sample is rejected
                # before the planner takes a single simulation step.
                self.session.tick()
                started = float(self.session.env.sim.data.time)
            target, move = position, original
            if kind == 'orientation':
                offset = self.orientation_offset()
                target = np.asarray(position)+offset
                def move(target, rotation, *move_args, **move_kwargs):
                    # MeasuredInjector has created its watch and target record
                    # before calling the executor. Keep the true nominal pose
                    # distinct from the combined fault's shifted target.
                    details = dict(nominal_position=np.asarray(position).copy(),
                                   transition_offset_metres=offset.copy())
                    self.watch.update(details)
                    self.fault_targets[-1].update({k: v.tolist() for k, v in details.items()})
                    return original(target, rotation, *move_args, **move_kwargs)
            result = super()._faulty(kind, move, target, orientation, args, kwargs)
            if not resample_target:
                break
            if kind == 'transition':
                attempt['sample'] = sample+1
                rejected = attempt['planning'] in ('blocked', 'unchecked')
            else:
                rejected = result.startswith(('unreachable', 'no '))
            if (not rejected or self.clock.event is not None
                    or float(self.session.env.sim.data.time) != started):
                # A checked route was executed. Its outcome belongs to this
                # waypoint; resampling after movement would inject it twice.
                break
            if sample+1 < samples:
                # The rejected target never moved the robot. Draw a new target
                # in the same intensity band before attempting any movement.
                if kind == 'transition':
                    self.directions.pop(self.row, None)
                    self.distances.pop(self.row, None)
                else:
                    self.tilts.pop(self.row, None)
                    self.orientation_offsets.pop(self.row, None)
        else:
            target = 'collision-free displaced' if kind == 'transition' else 'reachable tilted'
            result = 'unreachable: no %s Grasp target found after %d random samples' % (target, samples)
        if self.clock.event is None:
            self.fault_targets.append(dict(kind=kind, attempt_result=result))
        # The selected Grasp still performs its close/check after this move.
        # Handoff waits until this whole attempt has returned or stopped.
        return result

    def take_motion(self):
        if self.armed == 'grasp_loss' and self.intensity == 'high':
            # Keep the object held for the entire line/reach/follow path.
            # Completion, rather than a path midpoint, triggers the release.
            return None
        return super().take_motion()

    def finish_waypoint(self, index, message):
        if (index != self.row or self.primitive != 'transport'
                or self.armed != 'grasp_loss' or self.intensity != 'high'):
            return None
        self.take()
        if not message.startswith('ok'):
            self.missed.append('waypoint %d: Transport did not complete; no deliberate release' % (index+1))
            return None
        if not held_object(self.session.world):
            self.missed.append('waypoint %d: no held object at Transport end' % (index+1))
            return 'failed: High grasp loss could not release an object at Transport end'
        if self.release():
            return 'failed: injected grasp loss / high; object released at Transport end'
        return 'failed: object did not release at Transport end'


class ControllerTimings:
    """Separate live simulation and elapsed seconds for each controller."""
    def __init__(self, clock=None):
        self.clock = clock or time.monotonic
        self.seconds = {phase+'_'+unit+'_seconds': 0.
                        for phase in ('waypoints', 'vla') for unit in ('sim', 'wall')}
        self.phase = None
        self.paused = False
        self.sim_start = self.wall_start = 0.

    def snapshot(self, sim_time):
        seconds = dict(self.seconds)
        if self.phase and not self.paused:
            seconds[self.phase+'_sim_seconds'] += max(0., float(sim_time)-self.sim_start)
            seconds[self.phase+'_wall_seconds'] += max(0., self.clock()-self.wall_start)
        return seconds

    def switch(self, phase, sim_time):
        if phase not in (None, 'waypoints', 'vla'):
            raise ValueError('Unknown live controller')
        self.seconds = self.snapshot(sim_time)
        self.phase = phase
        self.sim_start, self.wall_start = float(sim_time), self.clock()

    def pause(self, sim_time):
        if not self.paused:
            self.seconds = self.snapshot(sim_time)
            self.paused = True

    def resume(self, sim_time):
        if self.paused:
            self.sim_start, self.wall_start = float(sim_time), self.clock()
            self.paused = False


@dataclass(frozen=True)
class LiveOptions:
    suite: str
    task_id: int
    scene: str = 'original'
    mode: str = 'recovery'
    kind: str = 'collision'
    intensity: str = 'low'
    delay: float = 0.
    max_waypoint_seconds: float = 180.
    max_policy_steps: int = 400
    backend: str = 'pi05'
    port: int = 8000
    seed: int = 1000
    injection_row: int = None
    use_saved_layout: bool = True
    fault_seed: int = None
    handoff_at_end: bool = False
    propagate_deviation: bool = False

    def validate(self):
        if not isinstance(self.handoff_at_end, bool) or not isinstance(self.propagate_deviation, bool):
            raise ValueError('Choose a valid continuation mode.')
        if self.handoff_at_end and self.delay:
            raise ValueError('End-of-plan handoff cannot also have a timed delay.')
        if self.scene not in ('original', 'shuffled') or self.mode not in MODES:
            raise ValueError('Choose a scene and run condition.')
        if self.kind not in FAILURES or self.intensity not in INTENSITIES:
            raise ValueError('Choose a failure and intensity.')
        if self.backend not in BACKENDS:
            raise ValueError('Choose an available recovery policy.')
        if (not np.isfinite([self.delay, self.max_waypoint_seconds]).all()
                or self.delay < 0 or self.max_waypoint_seconds <= 0):
            raise ValueError('Detection latency must be nonnegative; waypoint limit must be positive.')
        for value, low, high, name in ((self.max_policy_steps, 1, 100000, 'Policy steps'),
                                     (self.port, 1, 65535, 'Port'),
                                     (self.seed, 0, 2**32-1, 'Seed')):
            if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
                raise ValueError(name + ' must be a whole number in %d–%d.' % (low, high))
        if self.injection_row is not None and (not isinstance(self.injection_row, int)
                                               or self.injection_row < 0):
            raise ValueError('Choose an applicable injection waypoint.')
        if self.fault_seed is not None and (not isinstance(self.fault_seed, int)
                or isinstance(self.fault_seed, bool) or not 0 <= self.fault_seed < 2**64):
            raise ValueError('Fault seed must be a whole number in 0–2**64-1.')


class LiveRunner:
    def __init__(self, plan_root=ROOT/'saved_plans', experiment=DEFAULT_EXPERIMENT):
        self.plan_root, self.experiment = Path(plan_root), Path(experiment)
        self.session = GMMSession(sp.Args())
        self.session.save_replay_archive = False
        self.chain = None
        self.task = None
        self.prepared = None
        self.layouts = {}
        self.transition_history = {}
        self.collision_history = {}
        self.scene_origin = ''
        self.timings = ControllerTimings()

    def prepare(self, options):
        """Restore the selected start, without writing a new layout to disk."""
        options.validate()
        self.prepared = None
        self.timings = ControllerTimings()
        key = options.suite, options.task_id
        chain = GMMChain.load(self.plan_root/options.suite/('task_%03d.json' % options.task_id), *key)
        if options.mode in ('recovery', 'failure_only'):
            rows = eligible_rows(chain, options.kind, options.intensity)
            if not rows or (options.injection_row is not None and options.injection_row not in rows):
                raise ValueError('This failure is not applicable at the selected waypoint.')
        self.session.args.seed = options.seed
        np.random.seed(options.seed)
        if self.task != key:
            # A cancelled load must not leave a previous task marked as loaded.
            self.task = None
            self.session.load(*key, cameras=True)
            if chain.model_path:
                model_path = Path(chain.model_path)
                if not model_path.is_absolute():
                    model_path = ROOT/model_path
                model = json.loads(model_path.read_text())
                demos = model.get('demonstration_files', [])
                if demos:
                    demo_path = Path(demos[0])
                    if not demo_path.is_absolute():
                        demo_path = ROOT/demo_path
                    demo = json.loads(demo_path.read_text())
                    self.session.training_state = np.asarray(initial_state(demo, demo_path))
                    self.session.training_gripper = sp.GRIPPER_OPEN
            self.task = key
        self.chain = chain
        cache_key = key + (options.scene, options.seed, options.use_saved_layout)
        if cache_key not in self.layouts:
            layout_path = self.experiment/'layouts'/(
                '%s_t%03d_%s.json' % (options.suite, options.task_id, options.scene))
            self.session.last_shuffle_mapping = None
            if options.use_saved_layout and layout_path.is_file():
                layout = json.loads(layout_path.read_text())
                self.session.env.set_init_state(np.asarray(layout['state']))
                self.session.attach(settle=False)
                self.session.executor.gripper = sp.GRIPPER_OPEN
                self.session.last_shuffle_mapping = layout.get('shuffle_mapping')
                origin = 'Saved paired layout (layout seed %s)' % layout.get('seed', '?')
            else:
                self.session.restore_training_scene()
                if options.scene == 'shuffled':
                    self.session.shuffle_scene(seed=options.seed)
                origin = ('Shuffled in memory (seed %d)' % options.seed if options.scene == 'shuffled'
                          else 'Original demonstration layout')
            self.layouts[cache_key] = (self.session.env.sim.get_state().flatten().copy(),
                self.session.executor.gripper, self.session.last_shuffle_mapping, origin)
        state, gripper, mapping, self.scene_origin = self.layouts[cache_key]
        self.session.env.set_init_state(state.copy())
        self.session.attach(settle=False)
        self.session.executor.gripper = gripper
        self.session.last_shuffle_mapping = mapping
        self.prepared = options
        return chain

    def run(self, options, report=None, before_vla=None):
        """Execute from the chosen start; return an in-memory outcome dictionary."""
        options.validate()
        if self.prepared != options:
            self.prepare(options)
        # Another Run always restores the start, including after Stop or an error.
        self.prepared = None
        self.timings = ControllerTimings()
        try:
            result = self._execute(options, report, before_vla)
        finally:
            self.timings.switch(None, float(self.session.env.sim.data.time))
        result['controller_timings'] = self.timings.snapshot(self.session.env.sim.data.time)
        return result

    def _execute(self, options, report, before_vla):
        report = report or (lambda stage, message: None)
        session, chain = self.session, self.chain
        start = float(session.env.sim.data.time)
        result = dict(mode=options.mode, success=False)
        fault_row = None
        attempt_finished = False

        def progress(index, message):
            nonlocal attempt_finished
            if fault_row is not None and index > fault_row:
                attempt_finished = True
                if options.delay == 0 and not options.handoff_at_end:
                    raise Takeover()
            if message.startswith('Detection latency'):
                report('Detection latency', message)
                return
            stage = ('Detection latency' if options.handoff_at_end and fault_row is not None
                     and index > fault_row else 'Waypoints')
            report(stage, 'Step %d/%d · %s · %s' % (
                index+1, len(chain.waypoints), chain.waypoints[index].name, message))

        if options.mode == 'normal_waypoints':
            self.timings.switch('waypoints', session.env.sim.data.time)
            report('Waypoints', 'Running the normal waypoint plan')
            previous_tick = session.tick
            previous_continue = session.continue_on_failure
            def tick():
                previous_tick()
                if session.env.sim.data.time-start >= options.max_waypoint_seconds:
                    raise EpisodeLimit('Waypoint time limit reached')
            session.tick = tick
            session.continue_on_failure = True
            try:
                with suspend_images(session):
                    result['waypoint_result'] = session.run(chain, progress)
            except EpisodeLimit as exc:
                result['waypoint_result'] = str(exc)
            finally:
                session.tick = previous_tick
                session.continue_on_failure = previous_continue
            result['success'] = bool(session.env.check_success())
            result['status'] = 'normal_waypoints_success' if result['success'] else 'normal_waypoints_failed'
        else:
            if options.mode in ('recovery', 'failure_only'):
                self.timings.switch('waypoints', session.env.sim.data.time)
                report('Waypoints', 'Running toward the selected failure')
                # Layout/planner seed stays reproducible. Fault parameters get
                # new entropy on every Run, even when the same scene is reused.
                rows = eligible_rows(chain, options.kind, options.intensity)
                spec = dict(kind=options.kind, intensity=options.intensity,
                            fault_seed=(secrets.randbits(64) if options.fault_seed is None
                                        else options.fault_seed),
                            injection_row=options.injection_row if options.injection_row is not None else rows[0])
                fault_row = spec['injection_row']
                result['fault_seed'] = spec['fault_seed']
                injector_options = None
                if options.kind in ('transition', 'collision'):
                    history_key = (options.suite, options.task_id, options.scene, options.seed,
                                   options.use_saved_layout, fault_row, options.intensity)
                    histories = self.transition_history if options.kind == 'transition' else self.collision_history
                    injector_options = {options.kind+'_history': histories.setdefault(history_key, {})}
                def capture(_session, _folder, name):
                    if name == 'failure':
                        primitive = waypoint_primitive(chain.waypoints[fault_row]).title()
                        report('Finishing failed waypoint', 'Failure detected; completing this %s waypoint before handoff' % primitive)
                    elif name == 'handoff':
                        report('Handoff', 'Waypoints stopped at the failure scene')
                result.update(run_fault(session, chain, spec, options.delay,
                    options.max_waypoint_seconds, folder=None, capture=capture, progress=progress,
                    injector_class=GraspOnlyInjector, intensity_rule=grasp_intensities,
                    handoff_after_waypoint=True, injector_options=injector_options,
                    handoff_at_end=options.handoff_at_end,
                    propagate_deviation=options.propagate_deviation))
                self.timings.switch(None, session.env.sim.data.time)
                event = result.get('failure_event') or {}
                if options.kind == 'transition' and event.get('requested_direction') is not None:
                    direction = np.asarray(event['requested_direction'])
                    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360)
                    offset = direction*event['requested_metres']
                    result.update(transition_direction_degrees=angle, transition_offset_metres=offset.tolist())
                    report('Transition deviation', 'Direction %.1f° · X %+.1f cm · Y %+.1f cm' % (
                        angle, offset[0]*100, offset[1]*100))
                elif options.kind == 'collision' and event.get('requested_direction') is not None:
                    direction = np.asarray(event.get('requested_local_direction', event['requested_direction']))
                    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 360)
                    result['collision_direction_degrees'] = angle
                    report('Collision direction', 'Collision direction %.1f° (local XY) · %s' % (angle, event.get('body', 'obstacle')))
                elif options.kind == 'orientation':
                    target = next((target for target in reversed(result.get('fault_targets', []))
                                   if target.get('kind') == 'orientation' and target.get('axis') == 'xyz'), None)
                    if target is not None:
                        angles = np.degrees(target['angles'])
                        result['orientation_axes_degrees'] = angles.tolist()
                        requested = float(np.degrees(target['angle']))
                        pose = result.get('detection_latency_start_pose')
                        rotation = np.asarray(pose['orientation']) if pose else session.gripper_pose()[1]
                        measured = float(np.degrees(Rotation.from_matrix(
                            np.asarray(target['orientation']).T@rotation).magnitude()))
                        result.update(orientation_deviation_degrees=requested,
                                      orientation_measured_degrees=measured)
                        offset = np.asarray(target.get('transition_offset_metres', [0., 0., 0.]))
                        nominal = np.asarray(target.get('nominal_position', target['position']))
                        reached = np.asarray(pose['position']) if pose else session.world.eef()
                        result.update(orientation_transition_offset_metres=offset.tolist(),
                            orientation_transition_measured_metres=float(np.linalg.norm((reached-nominal)[:2])))
                        report('Orientation deviation', 'Total deviation at Grasp end · requested %.1f° · measured %.1f° · yaw %+.1f° · XY %.1f cm' % (
                            requested, measured, angles[2], np.linalg.norm(offset)*100))
                result['injection_attempt_stopped'] = (attempt_finished or
                    result.get('waypoint_execution_stopped', False) or
                    result.get('waypoint_interrupted', False) or result.get('waypoint_completed', False))
                result['handoff_ready'] = bool(result.get('waypoint_execution_stopped'))
                completion = result.get('waypoint_completion_event')
                result['fault_execution_completed'] = bool(result.get('waypoint_completed')
                    and completion and completion['row'] == fault_row and result['observed_failure'])
                if options.kind == 'transition' and result['fault_execution_completed']:
                    route = next((target for target in reversed(result.get('fault_targets', []))
                                  if 'planning' in target), None)
                    result['fault_execution_completed'] = bool(route
                        and route.get('attempt_result', '').startswith('ok')
                        and route.get('target_gap_metres',
                            np.linalg.norm(session.world.eef()-np.asarray(route['target']))) <= .02)
                if not result['fault_execution_completed']:
                    result['waypoint_failure_reason'] = str(result.get('waypoint_result') or 'No measured failure.')
                    if options.mode == 'failure_only' or not result['handoff_ready']:
                        result['status'] = 'failure_waypoint_incomplete'
                elif options.mode == 'failure_only':
                    result['status'] = 'failure_observed'
                if not result['handoff_ready']:
                    result['status'] = 'failure_waypoint_incomplete'
                    result.setdefault('waypoint_failure_reason', 'Waypoint execution has not stopped.')
            if 'status' not in result:
                if options.mode == 'recovery' and before_vla is not None:
                    if options.handoff_at_end:
                        if result.get('waypoint_plan_completed'):
                            report('Waiting for Continue VLA', 'Remaining waypoint plan finished. Press Continue VLA.')
                        else:
                            report('Waypoint plan stopped · Continue VLA',
                                   str(result.get('waypoint_result'))+' Press Continue VLA.')
                    elif result['fault_execution_completed']:
                        name = chain.waypoints[fault_row].name
                        report('Waiting for Continue VLA', 'Faulted %s finished. Press Continue VLA.' % name)
                    else:
                        report('Waypoint incomplete · Continue VLA',
                            'Waypoint attempt stopped. '+result['waypoint_failure_reason']+' Press Continue VLA.')
                    before_vla(result)
                self.timings.switch('vla', session.env.sim.data.time)
                report('Policy', 'Starting ' + BACKENDS[options.backend])
                policy_result = run_policy(session, chain.language, backend=options.backend,
                    port=options.port, max_steps=options.max_policy_steps,
                    report=lambda message: report('Policy', message))
                result.update(policy_result=policy_result, success=policy_result['success'])
                result['status'] = (('recovered' if result['success'] else 'recovery_failed')
                    if options.mode == 'recovery' else
                    ('normal_policy_success' if result['success'] else 'normal_policy_failed'))
        result['elapsed_sim_seconds'] = float(session.env.sim.data.time)-start
        if result.get('status') == 'failure_waypoint_incomplete':
            report('Waypoint incomplete', result['waypoint_failure_reason'])
        else:
            report('Finished', result['status'].replace('_', ' ').capitalize())
        return result

    def close(self):
        if self.session.env is not None:
            self.session.env.close()
            self.session.env = None
