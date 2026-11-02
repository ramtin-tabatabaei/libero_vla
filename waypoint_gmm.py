"""Editable adaptive chains derived from learned models; separate legacy format."""
import dataclasses
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from waypoint_chain import Chain, Waypoint


@dataclass
class GMMWaypoint(Waypoint):
    learned: dict = field(default_factory=dict)
    skill_role: str = ''

    def validate(self):
        super().validate()
        if not self.learned:
            return
        for key in ('position_tolerance', 'orientation_tolerance'):
            value = self.learned.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError('Invalid learned '+key)
        if self.learned.get('reference_policy') not in ('episode_start', 'interval_start'):
            raise ValueError('Invalid learned frame policy')
        for key in ('position_covariance', 'orientation_covariance'):
            covariance = np.asarray(self.learned.get(key), dtype=float)
            if covariance.shape != (3, 3) or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T):
                raise ValueError('Invalid learned covariance')
            if np.linalg.eigvalsh(covariance).min() < -1e-9:
                raise ValueError('Learned covariance must be positive semidefinite')


@dataclass
class GMMChain(Chain):
    max_waypoints: int = 100
    model_path: str = ''
    execution_skill: dict = field(default_factory=dict)

    def validate(self):
        super().validate()
        if self.execution_skill:
            template = copy.deepcopy(self.execution_skill.get('waypoint_template'))
            if template is not None:
                for waypoint in template:
                    waypoint.setdefault('orientation_relative_to_previous', False)
            if template != [dataclasses.asdict(w) for w in self.waypoints]:
                raise ValueError('Adaptive keyframes were changed; save edits as a manual plan or rebuild the VLA waypoints.')
        if not isinstance(self.max_waypoints, int) or self.max_waypoints < 2:
            raise ValueError('Invalid adaptive waypoint safety bound')
        if len(self.waypoints) > self.max_waypoints:
            raise ValueError(f'Plan exceeds the safety bound of {self.max_waypoints} waypoints')

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(dict(version=2, kind='gmm_editor_chain', **dataclasses.asdict(self)), indent=2, allow_nan=False)+'\n')
        temporary.replace(path)

    @classmethod
    def load(cls, path, suite, task_id):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.pop('version', None) != 2 or data.pop('kind', None) != 'gmm_editor_chain':
            raise ValueError('Expected an editable GMM/GMR plan. Use Open learned GMM/GMR model for model.json files.')
        if (data.get('suite'), data.get('task_id')) != (suite, task_id):
            raise ValueError('This plan belongs to another task.')
        data['waypoints'] = [GMMWaypoint(**w) for w in data['waypoints']]
        result = cls(**data)
        result.validate()
        return result


def chain_from_model(model, model_path=''):
    if model.get('object_relative_skill'):
        from waypoint_skill_editor import chain_from_skill_model
        return chain_from_skill_model(model, model_path)
    model = copy.deepcopy(model)
    result = []
    events = set()
    for i, node in enumerate(model['waypoints']):
        event = node.get('semantic_event')
        key = event.get('key') if event else None
        command = node['gripper_command'] if key not in events else 'keep'
        if key:
            events.add(key)
        result.append(GMMWaypoint(
            name=node.get('primitive', node.get('semantic_phase', 'Waypoint')).replace('_', ' ').title(),
            reference=node['reference']['name'], offset=node['relative_position_mean'],
            orientation=node['relative_orientation_mean'], motion='line',
            gripper=('close' if model['segments'][0]['gripper_probability'][0] >= .5 else 'open') if i == 0 else 'keep',
            gripper_after=command,
            learned=dict(node=copy.deepcopy(node), primitive=node.get('primitive'),
                         phase=node['phase'], segment=node['segment'],
                         reference_policy=node['reference']['policy'],
                         original_reference=node['reference']['name'],
                         position_tolerance=node['position_tolerance'], orientation_tolerance=node['orientation_tolerance'],
                         position_covariance=node['position_covariance'], orientation_covariance=node['orientation_covariance'],
                         semantic_event=event, reasons=node.get('reasons', []), edited=False)))
    chain = GMMChain(model['suite'], model['task_id'], model['instruction'], result,
                     f"GMM/GMR from {len(model['used_demonstrations'])} pi0.5 demonstrations; replay unverified",
                     max_waypoints=model['config']['max_waypoints'], model_path=str(model_path))
    chain.validate()
    return chain


def check_learned_pose(session, metadata, result, position, orientation, retry_line=True):
    """Retry small controller residuals without changing learned acceptance bounds."""
    injected = getattr(session, 'injected_failure', '')
    if injected:
        # This waypoint was deliberately made to miss. Its learned tolerance has
        # nothing to say about that, and the corrective retry below would undo it.
        return injected
    pos_tol, ori_tol = metadata['position_tolerance'], metadata['orientation_tolerance']

    def measure():
        p, r = session.gripper_pose()
        gap = float(np.linalg.norm(p-position))
        angle = float(Rotation.from_matrix(np.asarray(orientation).T@r).magnitude())
        return p, r, gap, angle

    unsettled = 'failed: robot did not settle at the target'
    if not result.startswith('ok') and result != unsettled:
        return result
    if getattr(session, 'continue_on_failure', False):
        # The executor already used the row's settling grace. Repeating a
        # motion or holding longer here would delay the following waypoint.
        if result == unsettled:
            return result
        retry_line = False
    actual_p, actual_r, gap, angle = measure()
    retries = 0
    while (retry_line and retries < 2 and (gap > pos_tol or angle > ori_tol)
           and gap <= min(.01, 2*pos_tol) and angle <= min(.1, 2*ori_tol)):
        # Same target, same gripper; no events are applied until arrival passes.
        result = session.executor.line(position, orientation)
        retries += 1
        actual_p, actual_r, gap, angle = measure()
        if not result.startswith('ok') and result != unsettled:
            return result
    if (result == unsettled or retries) and gap <= pos_tol and angle <= ori_tol:
        before_p, before_r = actual_p.copy(), actual_r.copy()
        session.executor.hold(5)
        actual_p, actual_r, gap, angle = measure()
        stationary = (np.linalg.norm(actual_p-before_p) < min(.002, pos_tol/4)
                      and Rotation.from_matrix(before_r.T@actual_r).magnitude() < .01)
        if gap <= pos_tol and angle <= ori_tol:
            if not stationary:
                return unsettled
            return ('ok (within learned tolerance after %d correction%s)' %
                    (retries, '' if retries == 1 else 's')) if retries else 'ok (stationary within learned pose tolerance)'
    if gap > pos_tol or angle > ori_tol:
        return (f'failed: learned tolerance exceeded (position {gap*1000:.1f} mm / '
                f'{pos_tol*1000:.2f} mm limit; orientation {np.degrees(angle):.1f}° / '
                f'{np.degrees(ori_tol):.2f}° limit; {retries} corrections)')
    return result


def check_learned_motion(session, waypoint, result, position, orientation):
    metadata = getattr(waypoint, 'learned', {})
    if not metadata:
        return result
    return check_learned_pose(session, metadata, result, position, orientation,
                              retry_line=waypoint.motion == 'line')
