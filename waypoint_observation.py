"""LIBERO-only recording adapter. Learned trajectory features live elsewhere."""
import numpy as np
from scipy.spatial.transform import Rotation


def episode_metadata(session):
    domain = session.world.domain
    poses = {n: [p.tolist(), r.tolist()] for n, (p, r) in session.poses().items()}
    joints = {}
    for j in range(session.world.m.njnt):
        name = session.env.sim.model.joint_id2name(j)
        if name and not name.startswith(('robot', 'gripper')) and int(session.world.m.jnt_type[j]) in (2, 3):
            bid = int(session.world.m.jnt_bodyid[j])
            joints[name] = dict(body=session.env.sim.model.body_id2name(bid),
                                type=int(session.world.m.jnt_type[j]))
    return dict(version=2, task=session.task.name, instruction=session.task.language,
                control_timestep=float(domain.control_timestep),
                goals=domain.parsed_problem.get('goal_state', []),
                initial_object_poses=poses, joints=joints,
                reference_candidates=list(poses),
                initial_gripper='close' if session.executor.gripper > 0 else 'open')


def measured_sample(session, obs, action, step, previous=None):
    domain, sim, world = session.world.domain, session.env.sim, session.world
    p, r = session.gripper_pose()
    dt = float(domain.control_timestep)
    poses = {n: [v.tolist(), q.tolist()] for n, (v, q) in session.poses().items()}
    pads = domain.robots[0].gripper.important_geoms
    contacts, grasped = {}, []
    for name, obj in list(domain.objects_dict.items()) + list(domain.fixtures_dict.items()):
        geoms = obj.contact_geoms
        left = bool(domain.check_contact(pads['left_fingerpad'], geoms))
        right = bool(domain.check_contact(pads['right_fingerpad'], geoms))
        if left or right:
            contacts[name] = dict(left=left, right=right, both=left and right,
                                  kind='object' if name in domain.objects_dict else 'fixture')
        if name in domain.objects_dict and left and right:
            grasped.append(name)
    joints = {}
    for j in range(world.m.njnt):
        name = sim.model.joint_id2name(j)
        if not name or name.startswith(('robot', 'gripper')) or int(world.m.jnt_type[j]) not in (2, 3):
            continue
        joints[name] = float(world.d.qpos[world.m.jnt_qposadr[j]])
        body = int(world.m.jnt_bodyid[j])
        descendants = {body}
        for b in range(body+1, world.m.nbody):
            if int(world.m.body_parentid[b]) in descendants:
                descendants.add(b)
        geoms = [sim.model.geom_id2name(g) for g in range(world.m.ngeom)
                 if int(world.m.geom_bodyid[g]) in descendants and sim.model.geom_id2name(g)]
        left = bool(domain.check_contact(pads['left_fingerpad'], geoms))
        right = bool(domain.check_contact(pads['right_fingerpad'], geoms))
        if left or right:
            contacts[name] = dict(left=left, right=right, both=left and right, kind='fixture_link')
    qpos = np.asarray(obs['robot0_gripper_qpos'])
    linear = np.zeros(3) if previous is None else (p-np.asarray(previous['position']))/dt
    angular = np.zeros(3) if previous is None else Rotation.from_matrix(r @ np.asarray(previous['orientation']).T).as_rotvec()/dt
    qvel = np.zeros_like(qpos) if previous is None else (qpos-np.asarray(previous['finger_positions']))/dt
    return dict(step=step, timestamp=step*dt, position=p.tolist(), orientation=r.tolist(),
                eef_position_world=p.tolist(), eef_orientation_world=r.tolist(),
                task_object_poses={n: poses[n] for n in domain.objects_dict},
                task_fixture_poses={n: poses[n] for n in domain.fixtures_dict},
                eef_linear_velocity=linear.tolist(), eef_angular_velocity=angular.tolist(),
                gripper='close' if action[-1] > 0 else 'open', gripper_command=float(action[-1]),
                grasped=grasped, finger_positions=qpos.tolist(), gripper_qpos=qpos.tolist(),
                gripper_qvel=qvel.tolist(), contact_state=contacts, object_poses=poses,
                reference_candidates=list(poses), joint_positions=joints,
                goal_predicates=[bool(domain._eval_predicate(g)) for g in domain.parsed_problem.get('goal_state', [])],
                task_success=bool(session.env.check_success()), action=np.asarray(action).tolist())
