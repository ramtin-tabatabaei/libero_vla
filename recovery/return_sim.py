"""Record visits and physically return to them in the existing LIBERO scene."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from recovery.return_points import RobotState, SCENE_DIM, VisitedHistory


def snapshot(session, max_scene_tokens=32):
    """Privileged simulator observation with generic goal roles and geometry."""
    world, domain, scene = session.world, session.env.env, session.executor.scene
    position, orientation = session.gripper_pose()
    robot = domain.robots[0]
    grip_joints = [session.env.sim.model.joint_name2id(j) for j in robot.gripper.joints]
    grip_addresses = session.env.sim.model.jnt_qposadr[grip_joints]
    held = [name for name, obj in domain.objects_dict.items()
            if domain._check_grasp(robot.gripper, obj.contact_geoms)]
    predicates = ('in', 'on', 'open', 'close', 'turnon', 'turnoff', 'other')
    context = np.zeros(8)
    sources, destinations = set(), set()
    goals = world.domain.parsed_problem.get('goal_state', [])
    for goal in goals:
        kind = str(goal[0]).lower()
        context[predicates.index(kind) if kind in predicates else 6] += 1.
        if len(goal) > 1:
            sources.add(goal[1])
        if len(goal) > 2:
            destinations.add(goal[2])
    context[:7] /= max(1., context[:7].sum())
    context[-1] = bool(session.env.check_success())
    poses = session.poses()
    tokens = []
    names = list(domain.objects_dict) + list(domain.fixtures_dict) + list(domain.object_sites_dict)
    for name in names:
        if name not in poses:
            continue
        point, rotation = poses[name]
        low, high = world.aabb(name)
        span = np.asarray(high) - low
        # Infinite planes are represented by a finite local patch.
        span = np.nan_to_num(span, nan=0., posinf=2., neginf=0.).clip(0., 2.)
        joint = world.articulated_joint(name)
        joint_value = 0.
        if joint:
            info = world.joint_info(joint)
            lower, upper = info['range']
            joint_value = (info['qpos'] - lower) / max(float(upper - lower), 1e-6)
        token = np.r_[orientation.T @ (np.asarray(point) - position),
            (orientation.T @ rotation)[:, :2].reshape(-1), span,
            name in domain.objects_dict, name in sources, name in destinations,
            name in held, joint_value, bool(joint), name in sources or name in destinations, 1.,
            np.zeros(11)]
        assert len(token) == SCENE_DIM
        tokens.append((name not in sources and name not in destinations,
                       float(np.linalg.norm(np.asarray(point) - position)), token.tolist()))
    # Each predicate is a separate token: source/destination pairing survives
    # multi-object tasks, and open/close/turn-on/turn-off remain distinguishable.
    for goal in goals:
        source = poses.get(goal[1]) if len(goal) > 1 else None
        destination = poses.get(goal[2]) if len(goal) > 2 else None
        point, rotation = source if source is not None else (position, orientation)
        relation = (orientation.T @ (destination[0] - point)
                    if destination is not None else np.zeros(3))
        kind = str(goal[0]).lower()
        encoded_kind = np.zeros(7)
        encoded_kind[predicates.index(kind) if kind in predicates else 6] = 1.
        token = np.r_[orientation.T @ (point - position),
            (orientation.T @ rotation)[:, :2].reshape(-1), np.zeros(3),
            0., 1., bool(destination is not None), 0., 0., 0., 1., float(source is not None),
            relation, encoded_kind, 1.]
        assert len(token) == SCENE_DIM
        tokens.append((False, -1., token.tolist()))
    tokens.sort(key=lambda item: (item[0], item[1]))
    state = RobotState(float(session.env.sim.data.time), position.tolist(), orientation.tolist(),
        scene.arm_qpos().tolist(), session.env.sim.data.qvel[scene.qadr].tolist(),
        session.env.sim.data.qpos[grip_addresses].tolist(), float(session.executor.gripper),
        scene=[token for _, _, token in tokens[:max_scene_tokens]], context=context.tolist(), held=held)
    state.validate()
    return state


class RecordVisits:
    """Wrap execution locally; record actual poses at primitive starts and steps."""
    def __init__(self, session):
        self.session = session
        self.history = VisitedHistory()

    def __enter__(self):
        from waypoint_failures import waypoint_primitive
        session = self.session
        self.original_run, self.original_step = session.run, session.env.step
        self.had_run, self.had_step = 'run' in vars(session), 'step' in vars(session.env)

        def run(chain, report):
            def observed(index, message):
                if message == 'Running':
                    self.history.start(index, waypoint_primitive(chain.waypoints[index]), snapshot(session))
                elif message.startswith('ok'):
                    self.history.complete(snapshot(session))
                report(index, message)
            return self.original_run(chain, observed)

        def step(action, *args, **kwargs):
            result = self.original_step(action, *args, **kwargs)
            self.history.append(snapshot(session))
            return result

        session.run, session.env.step = run, step
        return self

    def __exit__(self, *_):
        if self.had_run:
            self.session.run = self.original_run
        else:
            delattr(self.session, 'run')
        if self.had_step:
            self.session.env.step = self.original_step
        else:
            delattr(self.session.env, 'step')


@dataclass
class ReturnSafety:
    max_recovery_seconds: float = 20.
    max_peak_force_n: float = 100.
    max_bad_force_n: float = 10.
    joint_tolerance: float = .02
    position_tolerance: float = .03
    orientation_tolerance: float = .15
    gripper_tolerance: float = .015

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all() or min(asdict(self).values()) <= 0:
            raise ValueError('Return safety thresholds must be finite and positive')


class ReturnViolation(Exception):
    pass


@contextmanager
def watch_contacts(session, config, stop=True):
    """Observe each physics step and stop on measured excessive contact force."""
    from recovery.scene import ContactMonitor, geom_ids
    world, sim = session.world, session.env.sim
    robot = session.env.env.robots[0]
    pads = robot.gripper.important_geoms
    pad_ids = geom_ids(world, pads['left_fingerpad'] + pads['right_fingerpad'])
    # Permit pad contact with goal objects and currently held objects only.
    names = set(snapshot(session).held)
    names.update(goal[1] for goal in world.domain.parsed_problem.get('goal_state', []) if len(goal) > 1)
    targets = set()
    for name in names:
        if name in session.env.env.objects_dict:
            targets.update(geom_ids(world, session.env.env.objects_dict[name].contact_geoms))
        elif name in session.env.env.object_sites_dict or name in session.env.env.fixtures_dict:
            targets.update(world._geoms_of(world.parent_body(name)))
    monitor = ContactMonitor(world, pad_ids, targets)
    previous, had_step = sim.step, 'step' in vars(sim)
    measurements = dict(peak_force_n=0., peak_bad_force_n=0., violating_contacts=[],
                        limit_exceeded=False, exceeding_physics_steps=0)

    def step():
        result = previous()
        bad, _, peak = monitor.measure()
        bad_sum = float(bad.sum())
        measurements['peak_force_n'] = max(measurements['peak_force_n'], peak)
        measurements['peak_bad_force_n'] = max(measurements['peak_bad_force_n'], bad_sum)
        if peak > config.max_peak_force_n or bad_sum > config.max_bad_force_n:
            measurements['exceeding_physics_steps'] += 1
            already_recorded = measurements['limit_exceeded']
            measurements['limit_exceeded'] = True
            import mujoco
            for index in range(0 if already_recorded else world.d.ncon):
                contact = world.d.contact[index]
                first, second = int(contact.geom1), int(contact.geom2)
                if (first in monitor.robot) == (second in monitor.robot):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(world.m, world.d, index, force)
                magnitude = float(np.linalg.norm(force[:3]))
                if magnitude < .1:
                    continue
                measurements['violating_contacts'].append(dict(
                    first=sim.model.geom_id2name(first), second=sim.model.geom_id2name(second),
                    force_n=magnitude, permitted_pad_contact=(
                        first in pad_ids and second in targets or second in pad_ids and first in targets)))
            if stop:
                raise ReturnViolation(f'contact limit exceeded: peak={peak:.2f} N, undesired sum={bad_sum:.2f} N')
        return result

    sim.step = step
    try:
        yield measurements
    finally:
        if had_step:
            sim.step = previous
        else:
            delattr(sim, 'step')


class PhysicalReturnController:
    """RRT to a historical joint configuration, followed by measured arrival.

    The target is read from history. The current scene remains live throughout.
    Carried-object geometry is included in the planner collision checks.
    """
    def __init__(self, session, safety=None):
        self.session, self.safety = session, safety or ReturnSafety()
        settings = {name: getattr(session.args, name) for name in
            ('motion_speed', 'travel_speed', 'travel_accel', 'travel_floor',
             'rrt_max_iters', 'rrt_step', 'collision_check_step', 'collision_check_cartesian',
             'shortcut_rounds', 'settle_steps')}
        sources = [Path(__file__), Path(__file__).parents[1] / 'waypoint_rrt.py',
                   Path(__file__).parents[1] / 'subgoal_planner.py', Path(__file__).with_name('scene.py')]
        settings['code'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        settings['safety'] = asdict(self.safety)
        self.id = 'historical_joint_rrt:' + hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    def plan(self, candidate):
        from waypoint_rrt import Scene, RRTConnect
        current = snapshot(self.session)
        if set(current.held) != set(candidate.state.held):
            return None
        bodies = [self.session.world.parent_body(name) for name in current.held]
        scene = Scene(self.session.env, carried_bodies=bodies, args=self.session.args)
        q = np.asarray(candidate.state.joints)
        if not scene.valid(q):
            return None
        planner = RRTConnect(scene, self.session.args, np.random.default_rng(self.session.args.seed))
        path = planner.plan(scene.arm_qpos(), q)
        return None if path is None else (scene, path)

    def feasible(self, candidate):
        return self.plan(candidate) is not None

    def run(self, candidate):
        from recovery.return_points import Outcome
        session = self.session
        start = float(session.env.sim.data.time)
        planned = self.plan(candidate)
        if planned is None:
            return Outcome(False, False, 0., detail='current scene has no verified return path')
        scene, path = planned
        ex = session.executor
        previous_step = ex.on_step
        previous_scene = ex.scene

        def control_boundary():
            if previous_step:
                previous_step()
            if float(session.env.sim.data.time) - start > self.safety.max_recovery_seconds:
                raise ReturnViolation('recovery simulation-time budget exceeded')
            if not scene.within_limits(scene.arm_qpos()):
                raise ReturnViolation('arm exceeded its joint limits')
            if set(snapshot(session).held) != set(candidate.state.held):
                raise ReturnViolation('held-object state changed during the return')

        reached, violation, detail, contacts = False, False, '', None
        ex.on_step, ex.scene = control_boundary, scene
        try:
            with watch_contacts(session, self.safety) as contacts:
                # Keep current grip while moving; reproduce the historical command
                # only at arrival, then verify measured fingers as well as the arm.
                if not ex.follow(path, tol=self.safety.joint_tolerance):
                    detail = 'historical joint configuration was not reached'
                else:
                    ex.gripper = candidate.state.gripper_command
                    ex.hold(30)
                    actual, target = snapshot(session), candidate.state
                    angle = np.arccos(np.clip((np.trace(np.asarray(target.orientation).T @ actual.orientation) - 1) / 2, -1, 1))
                    reached = bool(np.max(abs(np.asarray(actual.joints) - target.joints)) <= self.safety.joint_tolerance
                        and np.linalg.norm(np.asarray(actual.position) - target.position) <= self.safety.position_tolerance
                        and angle <= self.safety.orientation_tolerance
                        and np.max(abs(np.asarray(actual.gripper_qpos) - target.gripper_qpos)) <= self.safety.gripper_tolerance)
                    detail = 'visited robot state reached' if reached else 'measured historical-state tolerance was not met'
        except ReturnViolation as exc:
            violation, detail = True, str(exc)
        finally:
            ex.on_step, ex.scene = previous_step, previous_scene
        return Outcome(reached, violation, float(session.env.sim.data.time) - start,
                       detail=detail, recovery_contacts=contacts)


def resume_from_return(session, candidate, controller, language, port=8000,
                       max_policy_steps=400, stable_seconds=1., backend='pi05', policy_runner=None,
                       resume_contact_mode='stop'):
    """Execute return then VLA; a failed/unsafe return never hands off."""
    if policy_runner is None:
        from waypoint_policy import run as policy_runner
    if resume_contact_mode not in ('stop', 'observe'):
        raise ValueError('VLA contact mode must be stop or observe')
    outcome = controller.run(candidate)
    if not outcome.safe_reached:
        return outcome
    start = float(session.env.sim.data.time)
    try:
        with watch_contacts(session, controller.safety, stop=resume_contact_mode == 'stop') as contacts:
            result = policy_runner(session, language, port=port, backend=backend,
                max_steps=max_policy_steps, stable_seconds=stable_seconds)
        outcome.resume_success = bool(result['success'])
    except ReturnViolation as exc:
        outcome.resume_success, outcome.resume_violation = False, True
        outcome.detail += '; VLA stopped: ' + str(exc)
    outcome.rollout_seconds = float(session.env.sim.data.time) - start
    outcome.resume_contacts = contacts
    outcome.finish_seconds = outcome.rollout_seconds if outcome.resume_success else None
    outcome.validate()
    return outcome


def recover_with_selector(session, history, failure_time, selector, language,
                          vla_id, port=8000, max_policy_steps=400, stable_seconds=1.,
                          phases=(0., .1, .25, .5, .75, 1.), safety=None,
                          resume_contact_mode='stop'):
    """Live entry point: rank visits, check a path, return, then hand off to VLA."""
    from recovery.return_points import Decision
    controller = PhysicalReturnController(session, safety)
    decision = Decision(f'live:{failure_time}',
        f'{session.args.task_suite_name}:{session.args.task_id}', vla_id,
        controller.id, failure_time, snapshot(session), history.candidates(failure_time, phases),
        max_policy_steps=max_policy_steps, stable_seconds=stable_seconds,
        resume_contact_mode=resume_contact_mode)
    report = selector.select(decision, feasible=controller.feasible)
    if report['selected'] is None:
        return report, None
    candidate = next(c for c in decision.candidates if c.id == report['selected'])
    outcome = resume_from_return(session, candidate, controller, language,
                                 port, max_policy_steps, stable_seconds,
                                 resume_contact_mode=resume_contact_mode)
    report['actual_outcome'] = asdict(outcome)
    return report, outcome
