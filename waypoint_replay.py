"""Replay learned waypoints or dense GMR targets using the existing LIBERO executor."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from waypoint_learning import (ANCHOR_FRAME, Config, fuse_world, load_model, tolerance, world_pose,
    write_json)


def reconstruct_target(node,frame):
    """Single-frame reconstruction, for version 1 models and legacy callers."""
    return world_pose(np.asarray(node['relative_position_mean']),np.asarray(node['relative_orientation_mean']),frame)


def reconstruct_fused(node,frames,anchor=None):
    """Product-of-Gaussians consensus of every task frame this waypoint was fitted in."""
    if node.get('retargeting') == 'task_endpoint':
        if anchor is None:
            if len(frames) == 1:
                raise ValueError('Connected single-reference waypoint requires its previous waypoint pose')
            anchor = frames[-1]
        return reconstruct_connected(node, frames[0], anchor)
    blocks=node['frames']
    p,r,_,_=fuse_world(frames,[np.asarray(b['relative_position_mean']) for b in blocks],
                       [np.asarray(b['relative_orientation_mean']) for b in blocks],
                       [np.asarray(b['position_covariance']) for b in blocks],
                       [np.asarray(b['orientation_covariance']) for b in blocks],
                       [np.asarray(b['empirical_position_covariance']) for b in blocks],
                       [np.asarray(b['empirical_orientation_covariance']) for b in blocks])
    return p,r


def reconstruct_connected(node, frame, anchor):
    """Keep task endpoints exact even when all training scenes were identical.

    The GMR curve stays relative to its task object. For move/transport its
    starting discrepancy decays smoothly to zero, connecting the actual robot
    pose to the learned path without averaging away the grasp/place target.
    """
    p, r = reconstruct_target(node, frame)
    if node.get('primitive') in ('move', 'transport'):
        start_p, start_r = world_pose(np.asarray(node['interval_start_position']),
                                     np.asarray(node['interval_start_orientation']), frame)
        u = float(np.clip(node['local_phase'], 0, 1))
        weight = 1 - u*u*(3-2*u)
        p = p + weight*(np.asarray(anchor[0])-start_p)
        correction = Rotation.from_matrix(np.asarray(anchor[1])@start_r.T).as_rotvec()
        r = Rotation.from_rotvec(weight*correction).as_matrix()@r
    return p, r


def resolve_frames(session,node,initial,anchor):
    """Live pose of each task frame; the anchor is where the robot actually is."""
    frames=[]
    for block in node['frames']:
        if block['name']==ANCHOR_FRAME:
            frames.append(anchor);continue
        poses=session.poses() if block['policy']=='interval_start' else initial
        if block['name'] not in poses:raise ValueError('Reference missing at replay: '+block['name'])
        frames.append(poses[block['name']])
    return frames


def dense_nodes(model):
    c=Config(**model['config']);result=[]
    for si,seg in enumerate(model['segments']):
        blocks=seg.get('frames');references=seg.get('references')
        for i,phase in enumerate(seg['phase']):
            event=seg['start_event'] if i==0 else seg['end_event'] if i==len(seg['phase'])-1 else None
            node=dict(segment=si,index=i,phase=phase,reference=seg['reference'],semantic_event=event,
                      gripper_command=event.get('execution_gripper_command', event['kind'] if event['kind'] in ('close','open') else 'keep') if event else 'keep')
            from waypoint_primitives import retarget_metadata
            node.update(retarget_metadata(seg, i))
            if blocks:
                pcov=np.asarray(seg['world_position_covariance'])[i]
                rcov=np.asarray(seg['world_orientation_covariance'])[i]
                node.update(frames=[dict(name=ref['name'],role=ref['role'],policy=ref['policy'],
                        relative_position_mean=np.asarray(b['position'])[i].tolist(),
                        relative_orientation_mean=np.asarray(b['orientation'])[i].tolist(),
                        position_covariance=np.asarray(b['covariance'])[i,:3,:3].tolist(),
                        orientation_covariance=np.asarray(b['covariance'])[i,3:6,3:6].tolist(),
                        empirical_position_covariance=np.asarray(b['empirical_position_covariance'])[i].tolist(),
                        empirical_orientation_covariance=np.asarray(b['empirical_orientation_covariance'])[i].tolist())
                        for ref,b in zip(references,blocks)],
                    relative_position_mean=np.asarray(blocks[0]['position'])[i].tolist(),
                    relative_orientation_mean=np.asarray(blocks[0]['orientation'])[i].tolist(),
                    position_tolerance=tolerance(pcov,c),orientation_tolerance=tolerance(rcov,c,True))
            else:
                cov=np.asarray(seg['covariance'][i])
                node.update(relative_position_mean=seg['position'][i],relative_orientation_mean=seg['orientation'][i],
                            position_tolerance=tolerance(cov[:3,:3],c),orientation_tolerance=tolerance(cov[3:6,3:6],c,True))
            result.append(node)
    return result


def replay(session,model,mode='waypoints',progress=None):
    """Freeze object anchors per episode, articulated link anchors per interval."""
    import subgoal_planner as sp
    if model.get('object_relative_skill'):
        if mode != 'waypoints':
            raise ValueError('Object-relative skill models use --mode waypoints; no dense GMR trajectory is stored')
        from waypoint_skill import run_skill
        trace = []
        def report(i, message):
            trace.append(dict(index=i,result=message))
            if progress is not None:
                progress(i, model['waypoints'][min(i,len(model['waypoints'])-1)], message)
        result = run_skill(session, model['object_relative_skill'], report, mode='constrained')
        return dict(completed=result['completed'], task_success=result['task_success'],
                    replay_success=bool(result['completed'] and result['task_success']),
                    trace=trace, details=result)
    from waypoint_gmm import check_learned_pose
    nodes=model['waypoints'] if mode=='waypoints' else dense_nodes(model)
    initial=session.poses();interval_frames={};interval_anchors={};done_events=set();trace=[];ex=session.executor
    initial_grip='close' if model['segments'][0]['gripper_probability'][0]>=.5 else 'open'
    result=ex.run_subgoal(sp.Subgoal('Initial gripper','grip',gripper=sp.GRIPPER_CLOSE if initial_grip=='close' else sp.GRIPPER_OPEN))
    if not result.startswith('ok'):return dict(completed=False,task_success=bool(session.env.check_success()),replay_success=False,trace=[],error=result)
    for index,node in enumerate(nodes):
        if progress is not None:
            progress(index,node,'Running')
        session.tick()
        si=node['segment'];ref=node['reference']
        if si not in interval_frames:
            interval_anchors[si] = session.gripper_pose()
            if node.get('frames'):
                # The anchor frame is the pose actually reached at the previous
                # waypoint, so a shifted scene is corrected for as the replay runs.
                interval_frames[si]=resolve_frames(session,node,initial,session.gripper_pose())
            else:
                poses=session.poses() if ref['policy']=='interval_start' else initial
                if ref['name'] not in poses:raise ValueError('Reference missing at replay: '+ref['name'])
                interval_frames[si]=poses[ref['name']]
        p,r=(reconstruct_fused(node,interval_frames[si],interval_anchors[si]) if node.get('frames')
             else reconstruct_target(node,interval_frames[si]))
        result=ex.line(p,r)
        executor_result=result
        result=check_learned_pose(session,node,result,p,r)
        actual_p,actual_r=session.gripper_pose();gap=float(np.linalg.norm(actual_p-p))
        angle=float(Rotation.from_matrix(r.T@actual_r).magnitude())
        event=node.get('semantic_event')
        if result.startswith('ok') and event and event['key'] not in done_events:
            if node['gripper_command']!='keep':
                result=ex.run_subgoal(sp.Subgoal(event['kind'],'grip',gripper=sp.GRIPPER_CLOSE if node['gripper_command']=='close' else sp.GRIPPER_OPEN))
            done_events.add(event['key'])
        trace.append(dict(segment=si,primitive=node.get('primitive'),phase=node['phase'],event=event,position_error_m=gap,orientation_error_rad=angle,
                          position_tolerance_m=node['position_tolerance'],orientation_tolerance_rad=node['orientation_tolerance'],executor_result=executor_result,result=result))
        if progress is not None:
            progress(index,node,result)
        if not result.startswith('ok'):break
    completed=len(trace)==len(nodes) and trace[-1]['result'].startswith('ok')
    return dict(completed=completed,task_success=bool(session.env.check_success()),replay_success=bool(completed and session.env.check_success()),trace=trace)


def _static_supports(world, bottom_z):
    """Horizontal collision planes and fixed box tops at an object's height."""
    import mujoco
    supports = []
    for g in range(world.m.ngeom):
        if not (world.m.geom_contype[g] or world.m.geom_conaffinity[g]):
            continue
        body = int(world.m.geom_bodyid[g])
        joint = int(world.m.body_jntadr[body])
        if joint >= 0 and int(world.m.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        typ = int(world.m.geom_type[g])
        r = np.asarray(world.d.geom_xmat[g]).reshape(3,3)
        if r[2,2] < .999:
            continue
        centre = np.asarray(world.d.geom_xpos[g])
        if typ == int(mujoco.mjtGeom.mjGEOM_PLANE):
            top = float(centre[2])
            low = np.array([-np.inf,-np.inf,top])
            high = np.array([np.inf,np.inf,top])
        elif typ == int(mujoco.mjtGeom.mjGEOM_BOX):
            size = np.asarray(world.m.geom_size[g])
            top = float(centre[2]+size[2])
            span = np.abs(r[:2,:2])@size[:2]
            low = np.r_[centre[:2]-span, top]
            high = np.r_[centre[:2]+span, top]
        else:
            continue
        if abs(bottom_z-top) < .025:
            supports.append((low,high))
    return supports


def shift_object(session,name,delta):
    """Conservative horizontal free-object relocation with support/overlap checks."""
    world=session.world;delta=np.asarray(delta,dtype=float)
    if delta.shape!=(3,) or not np.isfinite(delta).all() or abs(delta[2])>1e-9:
        raise ValueError('Only finite horizontal object shifts are supported')
    if name not in world.domain.objects_dict:raise ValueError('Shift requires a movable task object')
    body=world.domain.obj_body_id[name];joint=int(world.m.body_jntadr[body])
    if joint<0 or int(world.m.jnt_type[joint])!=0:raise ValueError('Object has no free joint')
    low,high=world.aabb(name);nl,nh=low+delta,high+delta
    supports=[]
    for other in list(world.domain.objects_dict)+list(world.domain.fixtures_dict):
        if other==name:continue
        ol,oh=world.aabb(other)
        if abs(low[2]-oh[2])<.025 and np.all(low[:2]>=ol[:2]) and np.all(high[:2]<=oh[:2]):
            supports.append((ol,oh))
        overlap=np.minimum(nh,oh)-np.maximum(nl,ol)
        if np.all(overlap>.003):raise ValueError('Shift overlaps '+other)
    supports.extend(_static_supports(world, low[2]))
    if not any(np.all(nl[:2]>=ol[:2]) and np.all(nh[:2]<=oh[:2]) for ol,oh in supports):
        raise ValueError('Cannot verify that shifted object remains on its support')
    before=session.env.sim.get_state().flatten().copy();adr=int(world.m.jnt_qposadr[joint])
    world.d.qpos[adr:adr+3]+=delta;session.env.sim.forward()
    session.attach(settle=False);session.executor.hold(15)
    settled_low,_=world.aabb(name)
    if abs(settled_low[2]-low[2])>.02:
        session.env.set_init_state(before);session.attach(settle=False)
        raise ValueError('Shift did not settle on the verified support')
    return dict(object=name,requested_delta=delta.tolist(),position=world.pos(name).tolist())


def shuffle_positions(session, seed=None, names=None, exclude_mapping=None):
    """Move a goal source to a new verified place when full swaps do not fit.

    Some scenes have differently sized objects and no valid all-object
    derangement. This uses the same support, overlap and settling checks as
    ``shift_object`` and always relocates the goal source first.
    """
    world = session.world
    goals = world.domain.parsed_problem.get('goal_state', [])
    source_names = [g[1] for g in goals if len(g) >= 3 and
                    g[0].lower() in ('in', 'on') and g[1] in world.domain.objects_dict]
    order = list(dict.fromkeys(source_names + (list(names) if names is not None else
                                list(world.domain.objects_dict))))
    if not order:
        raise ValueError('No movable task objects to shuffle')
    rng = np.random.default_rng(seed)
    previous = exclude_mapping if isinstance(exclude_mapping, dict) else {}
    last_error = None
    for name in order:
        start = world.pos(name).copy()
        for _ in range(100):
            delta = np.r_[rng.uniform(-.09,.09,2), 0.]
            if np.linalg.norm(delta[:2]) < .025:
                continue
            prior = previous.get(name)
            if (isinstance(prior, dict) and 'position' in prior and
                    np.linalg.norm(start[:2]+delta[:2]-np.asarray(prior['position'])[:2]) < .025):
                continue
            try:
                moved = shift_object(session, name, delta)
                session.refresh_references()
                return {name: moved}
            except ValueError as exc:
                last_error = exc
    raise ValueError('No different supported object position found: '+str(last_error))


def shuffle_objects(session, seed=None, names=None, exclude_mapping=None):
    """Permute supported source objects together, rolling back invalid placements.

    Goal containers stay fixed by default. Object identity, height and rotation
    stay attached to each object; only horizontal positions are exchanged.
    exclude_mapping rejects the previous layout when shuffling a restored scene.
    """
    w = session.world
    goals = w.domain.parsed_problem.get('goal_state', [])
    destinations = {w.parent_body(g[2]) for g in goals
                    if len(g) >= 3 and g[0].lower() in ('in', 'on')}
    names = sorted(names if names is not None else set(w.domain.objects_dict)-destinations)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError('Shuffle needs at least two distinct movable source objects.')
    addresses, bounds = {}, {}
    for name in names:
        if name not in w.domain.objects_dict:
            raise ValueError('Not a movable object: '+name)
        joint = int(w.m.body_jntadr[w.domain.obj_body_id[name]])
        if joint < 0 or int(w.m.jnt_type[joint]) != 0:
            raise ValueError('Object has no free joint: '+name)
        addresses[name] = (int(w.m.jnt_qposadr[joint]), int(w.m.jnt_dofadr[joint]))
        bounds[name] = w.aabb(name)
    all_bounds = {n: w.aabb(n) for n in list(w.domain.objects_dict)+list(w.domain.fixtures_dict)}
    supports = {}
    for name, (low, high) in bounds.items():
        supports[name] = [(ol, oh) for n, (ol, oh) in all_bounds.items() if n not in names and
            abs(low[2]-oh[2]) < .025 and np.all(low[:2] >= ol[:2]) and np.all(high[:2] <= oh[:2])]
        supports[name].extend(_static_supports(w, low[2]))
    xy = np.array([w.d.qpos[addresses[n][0]:addresses[n][0]+2].copy() for n in names])
    rng = np.random.default_rng(seed)
    for _ in range(100):
        perm = rng.permutation(len(names))
        if np.any(perm == np.arange(len(names))):
            continue
        mapping = {name: names[int(perm[i])] for i, name in enumerate(names)}
        if mapping == exclude_mapping:
            continue
        proposed = dict(all_bounds)
        for i, n in enumerate(names):
            delta = np.r_[xy[perm[i]]-xy[i], 0.]
            proposed[n] = tuple(v+delta for v in bounds[n])
        valid = True
        for n in names:
            low, high = proposed[n]
            if not any(np.all(low[:2] >= ol[:2]) and np.all(high[:2] <= oh[:2]) for ol, oh in supports[n]):
                valid = False
            if any(np.all(np.minimum(high, oh)-np.maximum(low, ol) > .003)
                   for other, (ol, oh) in proposed.items() if other != n):
                valid = False
        if valid:
            break
    else:
        raise ValueError('No different supported, non-overlapping shuffle found for these objects.')
    before = session.env.sim.get_state().flatten().copy()
    gripper = session.executor.gripper
    try:
        for i, name in enumerate(names):
            q, v = addresses[name]
            w.d.qpos[q:q+2] = xy[perm[i]]
            w.d.qvel[v:v+6] = 0.
        session.env.sim.forward()
        session.attach(settle=False)
        session.executor.gripper = gripper
        session.executor.hold(15)
        for name in names:
            if abs(w.aabb(name)[0][2]-bounds[name][0][2]) > .02:
                raise ValueError('Shuffled object did not settle on its support: '+name)
        session.refresh_references()
    except BaseException:
        session.env.set_init_state(before)
        session.attach(settle=False)
        session.executor.gripper = gripper
        raise
    return {name: names[int(perm[i])] for i, name in enumerate(names)}


def initial_state(demo,path):
    if 'initial_state' in demo:return demo['initial_state']
    if 'replay_state_file' in demo:return json.loads((Path(path).parent/demo['replay_state_file']).read_text())['initial_state']
    raise ValueError('Exact original scene unavailable: no replay state in demonstration')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path);parser.add_argument('--demo',type=Path)
    parser.add_argument('--mode',choices=('waypoints','trajectory','both'),default='waypoints')
    parser.add_argument('--scene',choices=('original','shifted','shuffled','both','all'),default='both')
    parser.add_argument('--shuffle-seeds', nargs='+', type=int, default=[0, 1, 2],
                        help='Reproducible layouts for --scene shuffled/all (default: 0 1 2)')
    parser.add_argument('--shift-object');parser.add_argument('--shift',nargs=3,type=float,default=[.02,0,0])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--render',action='store_true',help='Show the replay in a simulator window (requires a desktop display)')
    parser.add_argument('--realtime',action='store_true',help='Pace simulation at its control timestep')
    args=parser.parse_args();model=load_model(args.model)
    path=args.demo or next((Path(p) for p in model['demonstration_files'] if p),None)
    if path is None:parser.error('Provide --demo with replay state')
    demo=json.loads(path.read_text())
    if (demo['suite'],demo['task_id'])!=(model['suite'],model['task_id']):parser.error('Replay demo belongs to another task')
    state=initial_state(demo,path)
    import subgoal_planner as sp
    from waypoint_session import Session
    session=Session(sp.Args());report=dict(demonstration_success=bool(demo['success']),results=[])
    # Session defaults to headless for the editor; opt this CLI into its renderer.
    session.args.render=args.render
    session.realtime=args.realtime
    try:
        session.load(model['suite'],model['task_id'],cameras=False)
        if args.render:
            session.env.env.viewer.set_camera(session.env.sim.model.camera_name2id('agentview'))
            session.env.env.render()
            print('Showing learned replay. Press Ctrl+C in this terminal to stop.',flush=True)
        scenes = (['original', 'shifted', 'shuffled'] if args.scene == 'all' else
                  ['original', 'shifted'] if args.scene == 'both' else [args.scene])
        trials = [(scene, seed) for scene in scenes
                  for seed in (args.shuffle_seeds if scene == 'shuffled' else [None])]
        for scene, seed in trials:
            for mode in (['waypoints','trajectory'] if args.mode=='both' else [args.mode]):
                session.env.set_init_state(state);session.attach(settle=False)
                result=dict(scene=scene,mode=mode)
                try:
                    if scene == 'shuffled':
                        result['shuffle_seed'] = seed
                        result['shuffle'] = shuffle_objects(session, seed=seed)
                    if scene=='shifted':
                        name=args.shift_object or model.get('object_relative_skill',{}).get('source') or next((s['reference']['name'] for s in model.get('segments',[]) if s['reference']['role']=='target'),None)
                        result['shift']=shift_object(session,name,args.shift)
                    result.update(replay(session,model,mode));result['status']='tested'
                except ValueError as exc:result.update(status='unsupported',error=str(exc))
                except Exception as exc:result.update(status='error',error='%s: %s'%(type(exc).__name__,exc))
                report['results'].append(result);print(scene,mode,result.get('replay_success'),result.get('error',''),flush=True)
    finally:
        if session.env is not None:session.env.close()
        write_json(args.output or (args.model if args.model.is_dir() else args.model.parent)/'replay_report.json',report)


if __name__=='__main__':main()
