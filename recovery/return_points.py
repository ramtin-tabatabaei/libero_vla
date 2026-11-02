"""Observed return points for one recovery followed by a fixed VLA rollout.

An action selects a recorded robot state inside a previous primitive. No task
number or primitive name is used as a neural-network input.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

import numpy as np


SCHEMA = 'visited_return_bandit_v1'
SCENE_DIM = 31
CONTEXT_DIM = 8
CORE_DIM = 32
OPTION_DIM = 53


def finite_vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must have {size} finite values')
    return array


@dataclass
class RobotState:
    time: float
    position: list
    orientation: list
    joints: list
    velocity: list
    gripper_qpos: list
    gripper_command: float
    scene: list = field(default_factory=list)
    context: list = field(default_factory=lambda: [0.] * CONTEXT_DIM)
    held: list = field(default_factory=list)

    def validate(self):
        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError('State time must be finite and nonnegative')
        finite_vector(self.position, 3, 'Position')
        matrix = np.asarray(self.orientation, dtype=float)
        if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(matrix), 1.)):
            raise ValueError('Orientation must be a rotation matrix')
        finite_vector(self.joints, 7, 'Panda joints')
        finite_vector(self.velocity, 7, 'Joint velocities')
        finite_vector(self.gripper_qpos, 2, 'Gripper qpos')
        finite_vector(self.context, CONTEXT_DIM, 'Goal context')
        if self.gripper_command not in (-1., 1.):
            raise ValueError('Gripper command must be -1 (open) or +1 (close)')
        for token in self.scene:
            finite_vector(token, SCENE_DIM, 'Scene token')


@dataclass
class ReturnPoint:
    id: str
    waypoint: int
    primitive: str
    phase: float
    phase_basis: str
    state: RobotState


@dataclass
class Decision:
    id: str
    task_key: str  # splitting/auditing metadata, never a policy feature
    vla_id: str
    controller_id: str
    failure_time: float
    current: RobotState
    candidates: list
    max_policy_steps: int = 400
    stable_seconds: float = 1.
    resume_contact_mode: str = 'stop'

    def validate(self):
        self.current.validate()
        if (not np.isfinite(self.failure_time) or self.failure_time < 0
                or self.failure_time > self.current.time):
            raise ValueError('Failure must precede the current state')
        if not self.candidates or len({c.id for c in self.candidates}) != len(self.candidates):
            raise ValueError('Need nonempty, uniquely identified return points')
        if not all((self.id, self.task_key, self.vla_id, self.controller_id)):
            raise ValueError('Decision and controller identities are required')
        if (type(self.max_policy_steps) is not int or self.max_policy_steps < 1
                or not np.isfinite(self.stable_seconds) or self.stable_seconds < 0):
            raise ValueError('VLA horizon and stable-success duration must be valid')
        if self.resume_contact_mode not in ('stop', 'observe'):
            raise ValueError('VLA contact mode must be stop or observe')
        for candidate in self.candidates:
            candidate.state.validate()
            if candidate.state.time >= self.failure_time:
                raise ValueError('Return points must have been visited BEFORE the failure')
            if (not np.isfinite(candidate.phase) or not 0 <= candidate.phase <= 1
                    or candidate.phase_basis not in ('completed_duration', 'visited_prefix_duration')):
                raise ValueError('Invalid recorded primitive phase')

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data['current'] = RobotState(**data['current'])
        data['candidates'] = [ReturnPoint(**dict(c, state=RobotState(**c['state'])))
                              for c in data['candidates']]
        decision = cls(**data)
        decision.validate()
        return decision


@dataclass
class Outcome:
    reached: bool
    recovery_violation: bool
    recovery_seconds: float
    resume_success: object = None  # None means VLA was not resumed
    finish_seconds: object = None  # observed completion time; None is censored
    rollout_seconds: float = 0.
    resume_violation: bool = False
    detail: str = ''
    recovery_contacts: object = None
    resume_contacts: object = None

    @property
    def safe_reached(self):
        return bool(self.reached and not self.recovery_violation)

    @property
    def success(self):
        return bool(self.safe_reached and self.resume_success and not self.resume_violation)

    @property
    def force_compliant_success(self):
        return bool(self.success and not (self.resume_contacts or {}).get('limit_exceeded', False))

    def validate(self):
        for name in ('reached', 'recovery_violation', 'resume_violation'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if self.resume_success is not None and type(self.resume_success) is not bool:
            raise ValueError('resume_success must be a boolean or null')
        times = [self.recovery_seconds, self.rollout_seconds]
        if self.finish_seconds is not None:
            times.append(self.finish_seconds)
        if not np.isfinite(times).all() or min(times) < 0:
            raise ValueError('Durations must be finite and nonnegative')
        if self.resume_success is not None and not self.safe_reached:
            raise ValueError('Only safely reached states may hand control to the VLA')
        if self.safe_reached and self.resume_success is None:
            raise ValueError('Safely reached return points need a measured VLA outcome')
        if bool(self.resume_success) != (self.finish_seconds is not None):
            raise ValueError('Completion time must be present exactly when VLA succeeded')
        if self.finish_seconds is not None and self.finish_seconds > self.rollout_seconds + 1e-6:
            raise ValueError('Completion time cannot exceed observed rollout duration')


@dataclass
class Reward:
    success_bonus: float = 10.
    failure_cost: float = 2.
    violation_cost: float = 10.
    seconds_cost: float = .1

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError('Reward weights must be finite and nonnegative')

    def __call__(self, outcome):
        outcome.validate()
        return (self.success_bonus * outcome.success
                - self.failure_cost * (not outcome.success)
                - self.violation_cost * (outcome.recovery_violation or outcome.resume_violation)
                - self.seconds_cost * (outcome.recovery_seconds + outcome.rollout_seconds))


class VisitedHistory:
    """Keep dense measured samples, then select actual visits near phase bins.

    Phase is elapsed time within a recorded primitive, not an interpolation.
    An interrupted primitive is explicitly measured against its visited prefix.
    """
    def __init__(self):
        self.segments = []
        self.active = None

    def start(self, waypoint, primitive, state):
        self.active = dict(waypoint=waypoint, primitive=primitive,
                           states=[state], completed=False)
        self.segments.append(self.active)

    def append(self, state):
        if self.active is not None:
            states = self.active['states']
            if state.time > states[-1].time:
                states.append(state)

    def complete(self, state):
        self.append(state)
        if self.active is not None:
            self.active['completed'] = True
            self.active = None

    def candidates(self, failure_time, phases=(0., .1, .25, .5, .75, 1.)):
        phases = np.asarray(phases, dtype=float)
        if phases.ndim != 1 or not len(phases) or not np.isfinite(phases).all() or np.any((phases < 0) | (phases > 1)):
            raise ValueError('Phase bins must lie between 0 and 1')
        result = []
        for segment_index, segment in enumerate(self.segments):
            states = [s for s in segment['states'] if s.time < failure_time]
            if not states:
                continue
            span = states[-1].time - states[0].time
            measured = np.array([(s.time - states[0].time) / span if span else 0.
                                 for s in states])
            basis = ('completed_duration' if segment['completed'] and
                     len(states) == len(segment['states']) else 'visited_prefix_duration')
            indices = sorted({int(np.argmin(abs(measured - phase))) for phase in phases})
            for index in indices:
                result.append(ReturnPoint(f'{segment_index}:{index}', segment['waypoint'],
                    segment['primitive'], float(measured[index]), basis, states[index]))
        return result


def core(state):
    joints = np.asarray(state.joints)
    return np.r_[np.sin(joints), np.cos(joints), np.asarray(state.velocity) / 2.,
                 np.asarray(state.gripper_qpos) / .04, state.gripper_command,
                 state.context].astype(np.float32)


def option_features(current, point):
    target = point.state
    rotation = np.asarray(current.orientation)
    return np.r_[core(target),
        (np.asarray(target.joints) - current.joints) / np.pi,
        rotation.T @ (np.asarray(target.position) - current.position),
        (rotation.T @ target.orientation)[:, :2].reshape(-1),
        (np.asarray(target.gripper_qpos) - current.gripper_qpos) / .04,
        target.gripper_command,
        np.log1p(current.time - target.time) / 10., point.phase].astype(np.float32)


def write_transition(path, decision, action_id, outcome):
    decision.validate()
    outcome.validate()
    if action_id not in {c.id for c in decision.candidates}:
        raise ValueError('Action must identify a recorded candidate')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(schema=SCHEMA, decision=asdict(decision), action_id=action_id,
                  outcome=asdict(outcome))
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def load_transitions(path):
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if record.get('schema') != SCHEMA:
                    raise ValueError('Need visited return-point rollouts; aggregate timing CSV is insufficient')
                decision = Decision.from_dict(record['decision'])
                outcome = Outcome(**record['outcome'])
                outcome.validate()
                action = next(i for i, c in enumerate(decision.candidates)
                              if c.id == record['action_id'])
                rows.append((decision, action, outcome))
            except (KeyError, TypeError, ValueError, StopIteration) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
    if not rows:
        raise ValueError('No measured rollouts in dataset')
    return rows
