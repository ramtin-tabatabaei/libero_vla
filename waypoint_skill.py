"""VLA-demonstrated object-relative keyframes with measured grasp/place feedback.

Successful pick/place demonstrations may include a preceding articulated push.
Contact keyframes are measured VLA poses; free-space connectors are recomputed
for the live layout. No policy inference is used during execution.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from waypoint_learning import samples, infer_goals, detect_events


ARTICULATED = ('open', 'close', 'turnon', 'turnoff')


def settled_placement(ss, goals, goal):
    """Index of a terminal placement made without the fingers ever opening.

    LIBERO ends the episode the moment the predicate holds, so a policy that
    sets an object down and stops is never recorded releasing it. The settled
    terminal pose is still a demonstrated placement; an episode that ends with
    the object in mid-air is not.
    """
    states = ss[-1].get('goal_predicates')
    if not states or goal not in goals:
        return None
    index = goals.index(goal)
    return len(ss)-1 if index < len(states) and states[index] else None


def relative(pose, frame):
    p, r = map(np.asarray, pose); q, s = map(np.asarray, frame)
    return [(s.T@(p-q)).tolist(), (s.T@r).tolist()]


def compose(frame, pose):
    p, r = map(np.asarray, frame); q, s = map(np.asarray, pose)
    return p+r@q, r@s


def inverse(pose):
    p, r = map(np.asarray, pose)
    return -r.T@p, r.T


# Clearance above the grasp contact, and above the tallest thing in the scene.
GRASP_CLEARANCE, SCENE_CLEARANCE = .10, .08
# Clearance a carried object keeps over the scene, and over its release pose.
CARRY_CLEARANCE, PLACE_CLEARANCE = .02, .08
# Height the wrist stages at above a surface contact it is about to make.
PUSH_CLEARANCE = .08
# How far one push stroke advances a free object, and when it has arrived.
PUSH_STEP, PUSH_TOLERANCE = .03, .015
def carry_limits(ex):
    """Speed limits for a move with an object in the gripper."""
    return ex.args.carry_speed, ex.args.carry_accel, ex.args.carry_floor
# Arrival precision demanded of a measured contact, and of a free-space
# connector, in metres and radians.
CONTACT_TOLERANCE, CONNECTOR_TOLERANCE = .003, .02
CONTACT_ROTATION, CONNECTOR_ROTATION = .03, .15
# Smallest descent a placement is approached from when the demonstrated
# clearance puts the traverse out of reach.
MINIMUM_DESCENT = .02


def clearance_height(grasp_z, source_z, transport_height, obstacle_top=None):
    """One nominal traverse height for preview, editing and execution alike.

    The editor shows a stored offset; the executor must aim at that same point,
    so both read this instead of repeating a clearance rule of their own.
    """
    heights = [float(grasp_z)+GRASP_CLEARANCE, float(source_z)+float(transport_height)]
    if obstacle_top is not None:
        heights.append(float(obstacle_top)+SCENE_CLEARANCE)
    return float(max(heights))


def carry_height(approach, wrist_z, obstacle_top=None, object_bottom=None):
    """Raise a loaded traverse until the held object itself clears the scene.

    ``object_bottom`` is the world height of the carried object's underside at
    the moment of grasp; the wrist must ride that far higher than the tallest
    obstacle for the object to pass over it.
    """
    height = float(approach)
    if obstacle_top is not None and object_bottom is not None:
        height = max(height, float(obstacle_top)+(float(wrist_z)-float(object_bottom))+CARRY_CLEARANCE)
    return height


def supporting_bodies(world, name):
    """The bodies a goal region belongs to.

    A destination is often a site — `basket_1_contain_region`,
    `wooden_cabinet_1_top_side` — and a site is not a body, so excluding it by
    name excludes no geometry at all. Placing into a container means reaching
    into it, so the body that carries the region has to come out of the check.
    """
    domain = world.domain
    bodies = list(domain.objects_dict)+list(domain.fixtures_dict)
    if name in bodies:
        return (name,)
    owners = tuple(body for body in bodies if name.startswith(body))
    return owners or (name,)


def path_obstacle_top(world, start, end, ignore=(), margin=.05):
    """Tallest body whose footprint lies under the straight path between two points.

    Clearing the tallest thing anywhere in the scene means climbing over a
    cabinet the arm never crosses, which can put the traverse outside its
    reach. Only what the move actually passes over has to be cleared.
    """
    domain = world.domain
    a, b = np.asarray(start, float)[:2], np.asarray(end, float)[:2]
    span = b-a
    length = float(np.linalg.norm(span))
    direction = span/length if length > 1e-9 else np.zeros(2)
    top = None
    for name in list(domain.objects_dict)+list(domain.fixtures_dict):
        if name in ignore:
            continue
        low, high = (np.asarray(v, float) for v in world.aabb(name))
        centre = (low[:2]+high[:2])/2
        radius = float(np.linalg.norm(high[:2]-centre))
        along = float(np.clip((centre-a)@direction, 0, length))
        if float(np.linalg.norm(a+direction*along-centre)) <= radius+margin:
            top = float(high[2]) if top is None else max(top, float(high[2]))
    return top


def traverse_pose(frame, position, ceiling, floor, orientations, reach):
    """Highest reachable pose for a loaded traverse over a cleared path.

    The ceiling is the demonstrated carry height and the floor the lowest that
    still clears whatever lies under the path, so any height between them is
    safe and a ceiling the arm cannot hold is lowered rather than refused. The
    demonstrated carry rotation is tried first; where the arm cannot hold it
    over this layout, the placement rotation is, since that is the one the
    move has to end in anyway.
    """
    ceiling, floor = float(ceiling), float(min(floor, ceiling))
    if reach is None:
        return raise_in_frame(frame, position, ceiling), orientations[0]
    steps = max(2, int(np.ceil((ceiling-floor)/.02))+1)
    for orientation in orientations:
        for height in np.linspace(ceiling, floor, steps):
            target = raise_in_frame(frame, position, float(height))
            if reach(target, orientation):
                return target, orientation
    return raise_in_frame(frame, position, ceiling), orientations[0]


def vertical_axis(frame):
    """The frame axis closest to world +Z, with its world-Z component."""
    rotation = np.asarray(frame[1], float)
    axis = int(np.argmax(np.abs(rotation[2])))
    component = float(rotation[2, axis])
    return axis, (component if abs(component) > 1e-6 else 1e-6)


def raise_in_frame(frame, position, height):
    """Lift a world point to `height` along the frame's own near-vertical axis.

    Travelling along one frame axis keeps the stored connector offset equal to
    its contact offset in the other two numbers, so the editor shows a move
    that changes a single coordinate.
    """
    axis, component = vertical_axis(frame)
    local = np.asarray(relative([position, np.eye(3)], frame)[0], float)
    local[axis] += (float(height)-float(position[2]))/component
    return compose(frame, [local, np.eye(3)])[0]


def learn_skill(demos):
    candidates = []
    identity = None
    for d in demos:
        if not d.get('success') or d.get('placement_mode') not in (None, 'fixed_unshuffled'):
            continue
        goals, _ = infer_goals(d)
        spatial = [goal for goal in goals if len(goal) == 3 and goal[0].lower() in ('in', 'on')]
        articulated = [goal for goal in goals if goal not in spatial]
        # "Turn on the stove and put the pot on it" is one transfer with a push
        # around it, not two transfers; the push is measured, never assumed.
        if len(spatial) != 1 or any(len(goal) != 2 or goal[0].lower() not in ARTICULATED
                                    for goal in articulated):
            continue
        predicate, source, destination = spatial[0]
        ss = samples(d)
        if source not in d['objects']:
            continue
        key = (d['suite'], d['task_id'], source, destination)
        if identity is not None and key != identity:
            raise ValueError('Skill demonstrations must describe the same task and objects')
        identity = key
        # Stable two-pad evidence must persist; a gripper command is insufficient.
        stable = [i for i in range(2, len(ss))
                  if all(source in sample.get('grasped', []) for sample in ss[i-2:i+1])]
        holds = []
        for i in stable:
            if holds and i == holds[-1][-1]+1:
                holds[-1].append(i)
            else:
                holds.append([i])
        settled = settled_placement(ss, goals, spatial[0])
        grasp = release = None
        # The last completed hold is the one that placed the object; earlier
        # holds are fumbles the policy recovered from.
        for hold in reversed(holds):
            end = next((i for i in range(hold[0]+1, len(ss)) if ss[i]['gripper'] == 'open'), settled)
            if end is not None:
                grasp, release = hold[0], end
                break
        if grasp is None:
            continue
        g, r = ss[grasp], ss[release]
        if source not in g.get('object_poses', {}) or destination not in r.get('object_poses', {}):
            continue
        stable_grasp = relative([g['position'], g['orientation']], g['object_poses'][source])
        close = next((i for i in range(grasp-1, 0, -1)
                      if ss[i]['gripper'] == 'close' and ss[i-1]['gripper'] == 'open'), None)
        entry = ss[close] if close is not None else g
        grasp_entry = relative([entry['position'], entry['orientation']], entry['object_poses'][source])
        # The object can rotate while the fingers close. Express the final VLA
        # wrist orientation in its pre-contact frame, rather than the rotated
        # post-grasp object frame that would change the replayed approach.
        contact_orientation = relative([g['position'], g['orientation']],
                                       entry['object_poses'][source])[1]
        grasp_pose = [stable_grasp[0], contact_orientation]
        peak = max(range(grasp, release), key=lambda i: float(ss[i]['position'][2]))
        transport_orientation = relative([ss[peak]['position'], ss[peak]['orientation']],
                                         ss[peak]['object_poses'][destination])[1]
        release_object = relative(r['object_poses'][source], r['object_poses'][destination])
        heights = [float(s['position'][2]) for s in ss[grasp:release+1]]
        candidates.append(dict(grasp=grasp_pose, stable_grasp=stable_grasp,
            grasp_entry=grasp_entry, grasp_entry_sample=close,
            approach_orientation=grasp_entry[1],
            transport_orientation=transport_orientation, transport_sample=peak,
            release_object=release_object,
            release_eef=relative([r['position'], r['orientation']], r['object_poses'][destination]),
            transport_height=max(heights)-float(d['initial_object_poses'][source][0][2]),
            demo_file=d.get('_path'), grasp_sample=grasp, release_sample=release,
            initial_source=d['initial_object_poses'][source], initial_destination=d['initial_object_poses'][destination],
            opened_destination=r['object_poses'][destination]))
    if not candidates:
        raise ValueError('Need successful pick/place VLA demonstrations with measured grasp and release object poses')
    # Keep an actual demonstrated grasp, with alternatives ordered by medoid cost.
    def distance(a, b):
        dp = np.linalg.norm(np.asarray(a['grasp'][0])-b['grasp'][0])
        dr = Rotation.from_matrix(np.asarray(a['grasp'][1]).T@b['grasp'][1]).magnitude()
        return dp+.03*dr
    scores = [sum(distance(a,b) for b in candidates) for a in candidates]
    order = np.argsort(scores)
    ordered = [candidates[int(i)] for i in order]
    targets = [identity[3]]+[goal[1] for goal in articulated]
    drawer = next(filter(None, (learn_pregrasp_push(demos, target) for target in targets)), None)
    closing = next(filter(None, (learn_pregrasp_push(demos, target, after_release=True)
                                 for target in targets)), None)
    if closing and drawer and closing['joint'] == drawer['joint'] and not articulated:
        closing = None  # one measured travel, not an open and a close
    if drawer:
        # A drawer bowl can shift after finger closure. The stable held pose
        # describes the shifted bowl, while the first close pose targets the
        # bowl where it actually starts. Keep both measured poses for audit.
        for candidate in ordered:
            candidate['grasp'] = candidate['grasp_entry']
    return dict(version=1, kind='vla_object_relative_skill', suite=identity[0], task_id=identity[1],
                source=identity[2], destination=identity[3], predicate=spatial[0][0],
                primitives=(['move','push'] if drawer else [])
                           +['move','grasp','transport','place','release']
                           +(['move','push'] if closing else []),
                pregrasp_push=drawer, postplace_push=closing, candidates=ordered,
                representation='single_reference_minimal',
                orientation_scheme='precontact_object_and_previous',
                training='successful unshuffled VLA recordings', replay_validation='not_run')


def learn_sequential_skill(demos):
    """Learn repeated VLA pick/place episodes when a task moves multiple objects.

    A clean training run must release each manipulated object once. Retries and
    unfinished terminal grasps remain available as evidence but cannot define
    the sequence to replay. Intermediate clearance placements use the task's
    fixed goal region as their single reference.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for demo in demos:
        if (not demo.get('success') or
                demo.get('placement_mode') not in (None, 'fixed_unshuffled')):
            continue
        goals, _ = infer_goals(demo)
        spatial = [goal for goal in goals if len(goal) == 3 and
                   goal[0].lower() in ('in', 'on')]
        if not spatial:
            continue
        states = demo['samples'][0].get('goal_predicates')
        nonspatial = [i for i, goal in enumerate(goals) if goal not in spatial]
        if nonspatial and (states is None or any(
                i >= len(states) or not states[i] for i in nonspatial)):
            continue
        events = detect_events(demo)
        ss = samples(demo)
        releases = [event for event in events
                    if event['kind'] == 'release' and
                    event['identity'] in demo.get('objects', [])]
        # A policy that stops the moment the predicate holds never opens its
        # fingers on the object it just set down.
        for name in ss[-1].get('grasped', []):
            goal = next((g for g in spatial if g[1] == name), None)
            if goal is not None and settled_placement(ss, goals, goal) is not None:
                releases.append(dict(kind='release', identity=name,
                                     index=len(ss)-1, settled=True))
        releases.sort(key=lambda event: event['index'])
        # A fumbled object picked straight back up is one transfer, not two.
        order = []
        for event in releases:
            if not order or order[-1] != event['identity']:
                order.append(event['identity'])
        order = tuple(order)
        if len(order) < 2 or len(order) != len(set(order)):
            continue
        # Contact loss while fingers are still closed is not a completed
        # placement, even if the task predicate happens to become true.
        final = {name: max(event['index'] for event in releases
                           if event['identity'] == name) for name in order}
        complete = True
        for name, index in final.items():
            if any(event['identity'] == name and event['index'] == index and
                   event.get('settled') for event in releases):
                continue
            stable = next((event for event in reversed(events)
                           if event['kind'] == 'stable_grasp' and
                           event['identity'] == name and event['index'] < index), None)
            if stable is None or not any(
                    event['kind'] == 'open' and
                    stable['index'] < event['index'] <= index for event in events):
                complete = False
                break
        if not complete:
            continue
        if not any(goal[1] == order[-1] for goal in spatial):
            continue
        key = (demo['suite'], demo['task_id'],
               demo.get('language', demo.get('instruction')), order)
        groups[key].append(demo)
    if not groups:
        raise ValueError('No complete multi-object VLA sequence')
    key, selected = max(groups.items(), key=lambda item: len(item[1]))
    if len(selected) < 2:
        raise ValueError('Need two compatible complete multi-object VLA sequences')
    goals, _ = infer_goals(selected[0])
    spatial = [goal for goal in goals if len(goal) == 3 and
               goal[0].lower() in ('in', 'on')]
    final_destination = next(goal[2] for goal in spatial if goal[1] == key[3][-1])
    stages = []
    for source in key[3]:
        spatial_goal = next((goal for goal in spatial if goal[1] == source), None)
        predicate = spatial_goal[0] if spatial_goal else 'on'
        destination = spatial_goal[2] if spatial_goal else final_destination
        virtual = []
        for demo in selected:
            copy_demo = dict(demo)
            copy_demo['goals'] = [[predicate, source, destination]]
            virtual.append(copy_demo)
        stages.append(learn_skill(virtual))
    return dict(version=1, kind='vla_sequential_object_relative_skill',
                suite=key[0], task_id=key[1], stages=stages,
                primitives=[primitive for stage in stages
                            for primitive in stage['primitives']],
                training_demo_files=[demo.get('_path') for demo in selected],
                representation='single_reference_minimal',
                orientation_scheme='precontact_object_and_previous',
                training='successful unshuffled VLA recordings',
                replay_validation='not_run')


def learn_push_skill(demos):
    """Learn a two-waypoint plan for a task whose only goal is articulation.

    Opening a drawer or turning a knob has no grasp and no destination object,
    so the pick/place learner rejects it. The measured approach and joint
    travel are the whole task: move to the demonstrated contact, then push.
    """
    identity, usable = None, []
    for d in demos:
        if not d.get('success') or d.get('placement_mode') not in (None, 'fixed_unshuffled'):
            continue
        goals, _ = infer_goals(d)
        articulated = [g for g in goals if len(g) == 2 and g[0].lower() in ARTICULATED]
        if len(articulated) != 1 or len(articulated) != len(goals):
            continue
        key = (d['suite'], d['task_id'], articulated[0][1], articulated[0][0].lower())
        if identity is not None and key != identity:
            raise ValueError('Push demonstrations must describe the same articulated goal')
        identity = key
        usable.append(d)
    if not usable:
        raise ValueError('Need successful VLA demonstrations of a single articulated goal')
    push = learn_pregrasp_push(usable, identity[2], require_free_hand=False)
    if not push:
        raise ValueError('No measured joint travel; recordings with joint and frame '
                         'telemetry are required to learn a minimal push')
    return dict(version=1, kind='vla_articulation_skill', suite=identity[0], task_id=identity[1],
                target=identity[2], predicate=identity[3], candidates=[],
                primitives=['move', 'push'], pregrasp_push=push,
                representation='single_reference_minimal',
                orientation_scheme='precontact_object_and_previous',
                training='successful unshuffled VLA recordings', replay_validation='not_run')


def push_strokes(poses, positions, limit=2):
    """Split a demonstrated slide into the intervals where the object moved.

    A long push is done in strokes: shove, back off, reposition, shove again,
    because the arm cannot reach the whole way in one pass. Each stroke is one
    move/push pair in the plan.
    """
    speed = [float(np.linalg.norm(positions[i+1]-positions[i])) for i in range(len(positions)-1)]
    runs, active = [], None
    for i, value in enumerate(speed):
        if value > .0008:
            active = [i, i+1] if active is None else [active[0], i+1]
        elif active is not None and i-active[1] > 8:
            runs.append(active)
            active = None
    if active is not None:
        runs.append(active)
    runs = [(a, b) for a, b in runs if np.linalg.norm(positions[b]-positions[a]) > .02]
    if len(runs) > limit:
        # Keep the strokes that moved the object furthest, in demonstrated order.
        keep = sorted(sorted(runs, key=lambda r: -np.linalg.norm(positions[r[1]]-positions[r[0]]))[:limit])
        runs = keep
    return runs


def learn_object_push_skill(demos, limit=2):
    """Learn a minimal plan for an object slid across a surface, not carried.

    Some tasks are solved by pushing a plate or bowl into place. There is no
    grasp, so the pick/place learner has nothing to fit, and no joint, so the
    articulation learner has nothing to measure. The evidence is where the
    wrist touched the object in that object's own frame, and where the object
    came to rest in the goal frame, for each demonstrated stroke.
    """
    identity, evidence = None, []
    for d in demos:
        if not d.get('success') or d.get('placement_mode') not in (None, 'fixed_unshuffled'):
            continue
        goals, _ = infer_goals(d)
        if len(goals) != 1 or len(goals[0]) != 3 or goals[0][0].lower() not in ('in', 'on'):
            continue
        predicate, source, destination = goals[0]
        if source not in d.get('objects', []):
            continue
        ss = samples(d)
        if any(source in sample.get('grasped', []) for sample in ss):
            continue  # a carried object is a pick and place, not a push
        key = (d['suite'], d['task_id'], source, destination, predicate)
        if identity is not None and key != identity:
            raise ValueError('Push demonstrations must describe the same task and objects')
        identity = key
        object_poses = [sample.get('object_poses', {}).get(source) for sample in ss]
        goal_frame = ss[-1].get('object_poses', {}).get(destination)
        if goal_frame is None or any(pose is None for pose in object_poses):
            continue
        positions = np.asarray([pose[0] for pose in object_poses], float)
        runs = push_strokes(object_poses, positions, limit)
        if not runs:
            continue
        strokes = []
        for begin, finish in runs:
            # Sample the wrist only where the object was actually moving. The
            # pose just before a stroke is still descending and would replay as
            # a swipe through the air above the object.
            engaged = [relative([ss[i]['position'], ss[i]['orientation']], object_poses[i])
                       for i in range(begin, finish+1)]
            local = np.asarray([pose[0] for pose in engaged], float)
            axis = int(np.argmax(local.max(axis=0)-local.min(axis=0)))
            contact = np.median(local, axis=0)
            contact[axis] = local[0][axis]
            strokes.append(dict(
                contact=[contact.tolist(), engaged[0][1]], push_axis=axis,
                finish=relative([ss[finish]['position'], ss[finish]['orientation']], goal_frame),
                # Where the object itself came to rest after this stroke.
                target_object=relative(object_poses[finish], goal_frame),
                gripper=ss[begin]['gripper'], entry_sample=begin, end_sample=finish,
                travel=float(np.linalg.norm(positions[finish]-positions[begin])),
                initial_source=object_poses[begin]))
        evidence.append(dict(source=source, destination=destination, strokes=strokes,
                             demo_file=d.get('_path'), initial_source=object_poses[0],
                             initial_destination=goal_frame,
                             travel=float(np.linalg.norm(positions[runs[-1][1]]-positions[0]))))
    counts = {len(e['strokes']) for e in evidence}
    for count in sorted(counts, reverse=True):
        group = [e for e in evidence if len(e['strokes']) == count]
        if len(group) >= 1:
            evidence = group
            break
    if not evidence:
        raise ValueError('Need successful VLA demonstrations that move an ungrasped '
                         'object with measured object poses')
    def cost(a, b):
        return sum(np.linalg.norm(np.asarray(x['contact'][0])-y['contact'][0]) +
                   np.linalg.norm(np.asarray(x['finish'][0])-y['finish'][0])
                   for x, y in zip(a['strokes'], b['strokes']))
    scores = [sum(cost(a, b) for b in evidence) for a in evidence]
    ordered = [evidence[int(i)] for i in np.argsort(scores)]
    return dict(version=1, kind='vla_object_push_skill', suite=identity[0], task_id=identity[1],
                predicate=identity[4], candidates=[],
                primitives=['move', 'push']*len(ordered[0]['strokes']),
                object_push=dict(source=identity[2], destination=identity[3],
                                 strokes=len(ordered[0]['strokes']), candidates=ordered,
                                 travel=float(np.median([e['travel'] for e in evidence]))),
                representation='single_reference_minimal',
                orientation_scheme='precontact_object_and_previous',
                training='successful unshuffled VLA recordings', replay_validation='not_run')


def run_object_push(session, shove, report):
    """Execute the measured strokes and steer the object with measured poses.

    A pushed object is not held: it slides, spins and lags behind the wrist.
    Re-measuring it every stroke step and re-aiming keeps the fingers behind
    it, which one straight sweep to the demonstrated end pose cannot do.
    """
    import subgoal_planner as sp
    ex, world = session.executor, session.world
    source, destination = shove['source'], shove['destination']
    candidate = shove['candidates'][0]
    trace, began = [], np.asarray(world.pos(source), float).copy()
    unsettled = 'failed: robot did not settle at the target'

    def record(row, primitive, position, orientation, result):
        trace.append(dict(primitive=primitive, position=np.asarray(position).tolist(),
                          orientation=np.asarray(orientation).tolist(), result=result,
                          object_travel=float(np.linalg.norm(world.pos(source)-began))))
        report(row, result)

    def move(row, position, orientation, guarded=False):
        session.running_target = (np.asarray(position), np.asarray(orientation))
        report(row, 'Running')
        result = (guarded_move(session, position, orientation, ignore=(source,),
                               plan_first=False)[0]
                  or ex.line(position, orientation)) if guarded else ex.line(position, orientation)
        from waypoint_gmm import check_learned_pose
        result = check_learned_pose(session, dict(position_tolerance=.006,
            orientation_tolerance=.06), result, position, orientation)
        record(row, 'move', position, orientation, result)
        if not result.startswith('ok'):
            raise ValueError('move: '+result)
    try:
        for index, stroke in enumerate(candidate['strokes']):
            if session.env.check_success():
                break
            row = 3*index
            command = stroke.get('gripper', 'close')
            result = ex.run_subgoal(sp.Subgoal('Hand for push', 'grip',
                gripper=sp.GRIPPER_CLOSE if command == 'close' else sp.GRIPPER_OPEN))
            if not result.startswith('ok'):
                raise ValueError('move: '+result)
            # Freeze the contact frame: the object moves, and a target chasing
            # it never arrives.
            frame = session.poses()[source]
            local_contact = stroke['contact']
            if 'initial_source' in stroke and 'initial_source' in candidate:
                # Contact was recorded after the plate started rocking. Undo
                # that moving frame before retargeting onto a resting plate.
                measured = compose(stroke['initial_source'], local_contact)
                stable = np.array(candidate['initial_source'][0], float)
                if index:
                    stable[:2] = stroke['initial_source'][0][:2]
                local_contact = relative(measured, (stable, candidate['initial_source'][1]))
            contact, orientation = compose(frame, local_contact)
            # A surface push must not follow the object's roll/pitch as it
            # rocks against the fingers. Keep the demonstrated wrist frame
            # and contact height, while tracking only planar object motion.
            # The demonstrated approach skims the surface, which no collision
            # check can accept. Stage above the measured contact, then descend.
            # A layout that leaves no room overhead is approached directly.
            for height in (PUSH_CLEARANCE, .06, .04, None):
                if height is None:
                    # Nothing overhead is reachable; the contact pose itself
                    # touches by design and can never pass a clearance check.
                    move(row+1, contact, orientation)
                    break
                staged = raise_in_frame(frame, contact, contact[2]+height)
                if reachable(session, staged, orientation, ignore=(source,)):
                    move(row, staged, orientation, guarded=True)
                    move(row+1, contact, orientation)
                    break
            goal = compose(session.poses()[destination], stroke['target_object'])
            stalls = 0
            for _ in range(int(np.ceil(stroke['travel']/PUSH_STEP))+8):
                if session.env.check_success():
                    break
                here = session.poses()[source]
                remaining = np.asarray(goal[0], float) - here[0]
                remaining[2] = 0.
                distance = float(np.linalg.norm(remaining))
                if distance < PUSH_TOLERANCE:
                    break
                # Advance the wrist itself. Re-anchoring a fixed penetration
                # depth to a stalled object would repeat the same target.
                wrist, _ = session.gripper_pose()
                position = (np.asarray(wrist)
                            + min(PUSH_STEP, distance)*remaining/distance)
                position[2] = contact[2]
                rotated = orientation
                report(row+2, 'Running')
                result = ex.line(position, rotated)
                distance_after = float(np.linalg.norm(
                    (np.asarray(goal[0])-world.pos(source))[:2]))
                record(row+2, 'push', position, rotated, result)
                # A wrist loaded against the object it is shoving cannot satisfy
                # a free-space settling criterion; the object advancing is the
                # evidence that the stroke worked.
                if not result.startswith('ok') and result != unsettled:
                    raise ValueError('push: '+result)
                stalls = 0 if distance-distance_after > .0005 else stalls+1
                if stalls >= 4:
                    raise ValueError('push: %s stopped moving %.0f mm short of stroke %d'
                                     % (source, distance*1000, index+1))
            # Lift clear so the next approach is not dragged across the object.
            here, rotation = session.gripper_pose()
            move(row+2, here+[0, 0, PUSH_CLEARANCE], rotation)
        travelled = float(np.linalg.norm(world.pos(source)-began))
        if not session.env.check_success() and travelled < .5*shove['travel']:
            raise ValueError('push: %s moved only %.0f mm of an expected %.0f mm'
                             % (source, travelled*1000, shove['travel']*1000))
        report(3*len(candidate['strokes'])-1,
               'ok (%s moved %.0f mm)' % (source, travelled*1000))
        return dict(completed=True, source=source, travelled=travelled, waypoints=trace)
    except ValueError as exc:
        report(min(len(trace), 3*len(candidate['strokes'])-1), str(exc))
        return dict(completed=False, result=str(exc), source=source, waypoints=trace)


def push_window(ss, after_release=False, require_free_hand=True):
    """The sample range in which a demonstrated push can be measured.

    A drawer opened to reach an object is pushed before the hand is ever full;
    a drawer closed afterwards is pushed once the fingers have let go.
    """
    grasped = [i for i, sample in enumerate(ss) if sample.get('grasped')]
    if after_release:
        if not grasped:
            return None
        release = next((i for i in range(grasped[-1]+1, len(ss))
                        if ss[i]['gripper'] == 'open'), None)
        return None if release is None or len(ss)-release < 13 else (release, len(ss))
    first = grasped[0] if grasped else len(ss)
    if require_free_hand and first < 12:
        return None
    return 0, first


def learn_pregrasp_push(demos, destination, require_free_hand=True, after_release=False):
    """Measure drawer motion around a grasp even if contact detection missed it."""
    evidence = []
    for d in demos:
        if not d.get('success') or d.get('placement_mode') not in (None, 'fixed_unshuffled'):
            continue
        ss = samples(d)
        window = push_window(ss, after_release, require_free_hand)
        if window is None:
            continue
        low, high = window
        joint_names = set(d.get('joints', {})) & set(ss[0].get('joint_positions', {}))
        for name in joint_names:
            # Keep an articulation belonging to the destination fixture.
            prefix = destination.split('_region')[0].rsplit('_', 1)[0]
            if not name.startswith(prefix):
                continue
            q = np.asarray([sample['joint_positions'][name] for sample in ss[low:high]], float)
            peak = int(np.argmax(abs(q-q[0])))
            if abs(q[peak]-q[0]) < .04 or peak < 12:
                continue
            onset = next((i for i in range(peak+1) if abs(q[i]-q[0]) >= .005), None)
            if onset is None:
                continue
            start, approach = max(0, onset-3), max(0, onset-15)
            # The frame is where the articulation stands when this push begins,
            # which for a closing push is the open position it was left in.
            frame = ss[low].get('object_poses', {}).get(name)
            if frame is None:
                continue
            def local(i):
                return relative([ss[low+i]['position'], ss[low+i]['orientation']], frame)
            # The finish is measured contact evidence, like a grasp pose. The
            # few millimetres the wrist descends across a pull keep the open
            # fingers behind the handle; squaring the push onto one frame axis
            # loses the drawer part way out.
            evidence.append(dict(joint=name, initial_joint=float(q[0]),
                                 target_joint=float(q[peak]), approach=local(approach),
                                 contact=local(start), finish=local(peak),
                                 demo_file=d.get('_path'), onset_sample=low+onset,
                                 peak_sample=low+peak, initial_frame=frame))
    if not evidence:
        return None
    counts = {name: sum(e['joint'] == name for e in evidence) for name in {e['joint'] for e in evidence}}
    joint = max(counts, key=counts.get)
    evidence = [e for e in evidence if e['joint'] == joint]
    # Keep actual demonstrated contacts, with the medoid first.
    scores = [sum(np.linalg.norm(np.asarray(a['contact'][0])-b['contact'][0]) +
                  np.linalg.norm(np.asarray(a['finish'][0])-b['finish'][0])
                  for b in evidence) for a in evidence]
    return dict(joint=joint, candidates=[evidence[int(i)] for i in np.argsort(scores)],
                target_joint=float(np.median([e['target_joint'] for e in evidence])))


def run_pregrasp_push(session, push, report):
    """Execute the measured open-hand push and verify the joint actually moved."""
    import subgoal_planner as sp
    ex, world = session.executor, session.world
    joint = push['joint']
    info = world.joint_info(joint)
    target = float(push['target_joint'])
    direction = np.sign(target-push['candidates'][0]['initial_joint'])
    trace = []
    if direction*(info['qpos']-target) >= -.008:
        return dict(completed=True, joint=joint, target_joint=target,
                    reached_joint=info['qpos'], waypoints=trace, skipped_already_open=True)
    result = ex.run_subgoal(sp.Subgoal('Open hand for the push', 'grip', gripper=sp.GRIPPER_OPEN))
    if not result.startswith('ok'):
        return dict(completed=False, result=result, joint=joint, waypoints=trace)
    # Freeze the live frame while the drawer moves. Using the moving frame at
    # every waypoint would chase the handle indefinitely.
    frame = session.poses()[joint]
    candidate = push['candidates'][0]
    # The fixture being opened cannot also be an obstacle: reaching a handle
    # means reaching into the cabinet that carries it.
    fixture = tuple(name for name in world.domain.fixtures_dict if joint.startswith(name))
    def move(index, primitive, pose, checked=False):
        p, r = compose(frame, pose)
        session.running_target = (p, r)
        report(index, 'Running')
        outcome = (guarded_move(session, p, r, ignore=fixture, plan_first=False)[0]
                   or ex.line(p, r)) if checked else ex.line(p, r)
        from waypoint_gmm import check_learned_pose
        outcome = check_learned_pose(session, dict(position_tolerance=.006,
            orientation_tolerance=.06), outcome, p, r)
        trace.append(dict(primitive=primitive, position=p.tolist(),
                          orientation=r.tolist(), result=outcome,
                          joint_position=world.joint_info(joint)['qpos']))
        report(index, outcome)
        if not outcome.startswith('ok'):
            raise ValueError(primitive+': '+outcome)
        return p, r
    try:
        move(0, 'move', candidate['approach'], checked=True)
        start, orientation = move(1, 'move', candidate['contact'])
        end, end_orientation = compose(frame, candidate['finish'])
        delta = end-start
        rotation_delta = Rotation.from_matrix(end_orientation@orientation.T).as_rotvec()
        initial_q = previous_q = world.joint_info(joint)['qpos']
        unsettled = 'failed: robot did not settle at the target'
        def stroke(position, orientation, outcome):
            nonlocal previous_q
            reached = world.joint_info(joint)['qpos']
            trace.append(dict(primitive='push', position=np.asarray(position).tolist(),
                              orientation=np.asarray(orientation).tolist(),
                              result=outcome, joint_position=reached))
            # A wrist loaded against a knob or handle cannot satisfy a
            # free-space settling criterion; the joint turning is what says
            # the stroke worked. A stalled joint is still a failure.
            if not outcome.startswith('ok') and (outcome != unsettled or
                                                 direction*(reached-previous_q) <= .0005):
                raise ValueError('push: '+outcome)
            advanced = direction*(reached-previous_q) > .0005
            previous_q = reached
            return advanced
        def arrived():
            return direction*(world.joint_info(joint)['qpos']-target) >= -.008

        def chord():
            """Follow the measured contact-to-finish translation."""
            for step in range(12):
                if arrived():
                    return
                report(2, 'Running')
                fraction = (step+1)/12
                p = start+fraction*delta
                turn = Rotation.from_rotvec(fraction*rotation_delta).as_matrix()@orientation
                stroke(p, turn, ex.line(p, turn))

        def sweep(rounds=14):
            """Turn about the joint's own axis, rotating the wrist with it."""
            axis = info['axis']/max(np.linalg.norm(info['axis']), 1e-9)
            for _ in range(rounds):
                if arrived():
                    return True
                here, rotation = session.gripper_pose()
                angle = float(np.clip(target-world.joint_info(joint)['qpos'], -.2, .2))
                report(2, 'Running')
                outcome = ex.arc(joint, angle,
                                 Rotation.from_rotvec(axis*angle).as_matrix()@rotation)
                if outcome.startswith('unreachable'):
                    # The arm cannot follow the arc any further from here.
                    stroke(here, rotation, outcome.replace('unreachable', 'ok (arc limit)'))
                    return arrived()
                if not stroke(*session.gripper_pose(), outcome):
                    return arrived()
            return arrived()

        # A drawer slides along the measured stroke. A knob turns about its
        # axis, and the straight chord between contact and finish slides off it
        # part way round, so sweep about the axis; where the fingers never
        # engage, the demonstrated chord seats them and the sweep resumes.
        if info['slide']:
            chord()
        elif not sweep():
            chord()
            sweep()
        final_q = world.joint_info(joint)['qpos']
        if direction*(final_q-target) < -.012:
            raise ValueError('push: %s did not reach its demonstrated travel; '
                             'joint %.3f, target %.3f' % (joint, final_q, target))
        report(2, 'ok (joint %.3f)' % final_q)
        return dict(completed=True, joint=joint, target_joint=target,
                    reached_joint=final_q, started_joint=initial_q, waypoints=trace)
    except ValueError as exc:
        report(2, str(exc))
        return dict(completed=False, result=str(exc), joint=joint,
                    reached_joint=world.joint_info(joint)['qpos'], waypoints=trace)


def run_skill(session, skill, report=None, mode='feedback', waypoint_finished=None):
    """Execute a minimal plan, reporting against its own plan rows.

    Every caller sees one row index per listed waypoint, so the editor list,
    the CLI trace and the executor cannot drift apart.
    waypoint_finished signals Grasp after its gripper action and acquisition
    check, and Transport after the movement and carried-object checks finish.
    """
    report = report or (lambda *_: None)
    waypoint_finished = waypoint_finished or (lambda *_: None)
    if skill.get('stages'):
        rows, trace, stages = 0, [], []
        for stage in skill['stages']:
            result = run_skill(session, stage,
                               lambda i, status, base=rows: report(base+i, status), mode,
                               lambda i, status, base=rows: waypoint_finished(base+i, status))
            stages.append(result)
            trace.extend(result.get('waypoints', []))
            if not result['completed']:
                return dict(completed=False, result=result['result'],
                            task_success=bool(session.env.check_success()),
                            stages=stages, waypoints=trace, runtime_vla_calls=0)
            rows += len(stage['primitives'])
        return dict(completed=True, result='Finished sequential object-relative skill',
                    task_success=bool(session.env.check_success()),
                    stages=stages, waypoints=trace, runtime_vla_calls=0)

    row, trace = 0, []
    detail = dict(completed=True, result='', waypoints=trace, runtime_vla_calls=0)

    def stopped(message):
        detail.update(completed=False, result=message,
                      task_success=bool(session.env.check_success()))
        return detail

    if skill.get('object_push'):
        shove = run_object_push(session, skill['object_push'],
                                lambda i, status: report(row+i, status))
        detail['push'] = shove
        trace.extend(shove['waypoints'])
        row += 3*skill['object_push']['strokes']
        if not shove['completed']:
            return stopped(shove['result'])
        detail['result'] = 'Finished learned object push'
    if skill.get('pregrasp_push'):
        opening = run_pregrasp_push(session, skill['pregrasp_push'],
                                    lambda i, status: report(row+i, status))
        detail['drawer'] = opening
        trace.extend(opening['waypoints'])
        row += 3
        if not opening['completed']:
            return stopped(opening['result'])
        detail['result'] = 'Finished learned articulation skill'
    if skill.get('candidates'):
        def pick_place_report(index, status, base=row):
            report(base+index, status)
        picked = _run_pick_place(session, skill, pick_place_report, mode,
            lambda i, status, base=row: waypoint_finished(base+i, status))
        detail.update({k: v for k, v in picked.items() if k != 'waypoints'})
        trace.extend(picked['waypoints'])
        detail['waypoints'] = trace
        row += 7
        if not picked['completed']:
            return detail
    if skill.get('postplace_push'):
        closing = run_pregrasp_push(session, skill['postplace_push'],
                                    lambda i, status: report(row+i, status))
        detail['closing'] = closing
        trace.extend(closing['waypoints'])
        row += 3
        if not closing['completed']:
            return stopped(closing['result'])
        detail['result'] = detail['result'].replace('object-relative skill',
                                                    'object-relative skill and closing push')
    detail['task_success'] = bool(session.env.check_success())
    detail['waypoints'] = trace
    return detail


def planning_scene(session, ignore=(), carried=()):
    from waypoint_rrt import Scene
    return Scene(session.env, ignore_bodies=tuple(ignore), carried_bodies=tuple(carried))


def probe_line(session, scene, position, orientation):
    """Walk the straight Cartesian move the executor would actually run.

    The planner's own joint-space shortcut is a different path through the
    scene, so a cleared RRT edge says nothing about this move. Walking the
    Cartesian line at the executor's own resolution, warm-starting each sample
    from the last, keeps the demonstrated straight traverse whenever it is safe.

    Returns ``clear`` when every sample is reachable and collision free,
    ``blocked`` when one is in collision, and ``unreachable`` when the arm
    cannot follow the line at all — the executor's own error to report, not a
    reason to go looking for a detour. The final configuration comes back as
    the planner's goal when a detour is wanted.
    """
    ex = session.executor
    start, start_orientation = session.gripper_pose()
    position = np.asarray(position, float)
    span = float(np.linalg.norm(position-start))
    angle = Rotation.from_matrix(np.asarray(orientation)@start_orientation.T).magnitude()
    count = max(2, int(np.ceil(span/min(ex.args.line_step, .005))), int(np.ceil(angle/.04)))
    rotations = Slerp([0, 1], Rotation.from_matrix([start_orientation, np.asarray(orientation)]))
    q, blocked = scene.arm_qpos(), False
    for i in range(1, count+1):
        session.tick()
        fraction = i/count
        # Each step is a few millimetres from a solved configuration; a warm
        # start that has not converged by here is not going to.
        step, error = scene.ik(start+(position-start)*fraction, q_init=q, iters=60,
                               target_ori=rotations(fraction).as_matrix())
        if error > .01:
            # The last solved configuration is still the best seed for a
            # joint-space search towards the same target.
            return 'unreachable', q
        q = step
        blocked = blocked or not scene.valid(q)
    return ('blocked' if blocked else 'clear'), q


def seated(session, source, goal, result, limit=.02):
    """Accept a placement the object's own support cut short.

    Lowering a bowl into a drawer or onto a plate stops when the bowl reaches
    what holds it, which can be a centimetre or two from the pose recorded in
    the demonstration's own layout. Where the wrist ended is not what the task
    asks about; where the object came to rest is, so that is what is checked.
    """
    world = session.world
    before = np.asarray(world.pos(source), float).copy()
    session.executor.hold(5)
    now = np.asarray(world.pos(source), float)
    if float(np.linalg.norm(now-before)) > .002:
        return result  # still settling, or sliding out of the gripper
    gap = float(np.linalg.norm(now-np.asarray(goal, float)))
    if gap > limit:
        return result
    return 'ok (object seated %.1f mm from the demonstrated pose)' % (gap*1000)


def reachable(session, position, orientation, ignore=(), carried=()):
    """Is there a collision-free arm configuration that holds this pose?"""
    scene = planning_scene(session, ignore, carried)
    q, error = scene.ik(position, q_init=scene.arm_qpos(), target_ori=orientation)
    return bool(error <= .01 and scene.valid(q))


def guarded_move(session, position, orientation, ignore=(), carried=(), plan_first=True):
    """Plan a collision-free route to a free-space pose between waypoints.

    The learned plan says *where* the arm has to be; how it travels between
    those poses is the planner's job. RRT-Connect checks every edge in a
    scene where a grasped object moves with the wrist, so the object is routed
    clear too, and it returns a straight joint-space edge where one is free.
    Only when no route exists does this fall back to the Cartesian line, and
    then it says so rather than reporting the move as checked.

    An approach that ends within reach of what it is about to touch passes
    ``plan_first=False``: a joint-space route into a handle presses the wrist
    against the fixture instead of arriving beside it, so there the checked
    straight move comes first and the planner is the fallback.

    Returns the executor result and how the clearance was established.
    """
    ex = session.executor
    # An arm with something in the gripper accelerates gently.
    limits = carry_limits(ex) if carried else None
    scene = planning_scene(session, ignore, carried)
    if not plan_first:
        status, _ = probe_line(session, scene, position, orientation)
        if status == 'clear':
            return ex.line(position, orientation, *(limits or ())), 'line'
    saved_scene, saved_planner = ex.scene, ex.planner.s
    ex.scene = ex.planner.s = scene
    status = None
    try:
        # Damped least squares is seed sensitive. Try the arm's present
        # configuration first and only pay for walking the line when that
        # branch fails to reach the target.
        q_goal, error = scene.ik(position, q_init=scene.arm_qpos(), target_ori=orientation)
        if error > .01:
            status, seed = probe_line(session, scene, position, orientation)
            if seed is not None:
                q_goal, error = min(
                    (scene.ik(position, q_init=q, target_ori=orientation)
                     for q in (seed, scene.arm_qpos())), key=lambda item: item[1])
        if error <= .01 and scene.valid(q_goal):
            path = ex.planner.plan(scene.arm_qpos(), q_goal)
            if path is not None:
                if not ex.follow(path, limits=limits):
                    return 'failed: robot did not settle on the planned path', 'planned'
                ex.hold(ex.args.settle_steps)
                return 'ok (collision-free route, %d configs)' % len(path), 'planned'
        if status is None:
            status, _ = probe_line(session, scene, position, orientation)
    finally:
        ex.scene, ex.planner.s = saved_scene, saved_planner
    if status == 'clear':
        # No route was found, but the straight move is checked and free.
        return ex.line(position, orientation,
                       *(limits or (ex.args.travel_speed, ex.args.travel_accel,
                                    ex.args.travel_floor))), 'line'
    if status == 'blocked':
        return 'no collision-free route to the target', 'blocked'
    # Nothing can track this pose; let the executor report its own failure.
    return None, 'unchecked'


def _run_pick_place(session, skill, report=None, mode='feedback', waypoint_finished=None):
    import subgoal_planner as sp
    report = report or (lambda *_: None)
    waypoint_finished = waypoint_finished or (lambda *_: None)
    ex, world, domain = session.executor, session.world, session.world.domain
    source, destination = skill['source'], skill['destination']
    detail = dict(completed=False, result='', grasp_attempts=[], waypoints=[], runtime_vla_calls=0)

    def held():
        return bool(domain._check_grasp(domain.robots[0].gripper, domain.objects_dict[source].contact_geoms))

    def motion(row, primitive, p, r, carrying=False, joint_target=None, guarded=False,
               ignore=(), seat=None, complete=True):
        """Free-space moves are collision checked; contact moves stay straight.

        Descending onto an object, lifting it and lowering it into a container
        are all motions that touch by design, so a checked planner refuses
        every one of them. The free-space connectors between them are checked.
        """
        # Rows describe plan phases, not the number of internal moves. A
        # grasp retry revisits its approach/grasp rows without shifting the rest.
        p, r = np.asarray(p), np.asarray(r)
        record = dict(primitive=primitive, position=p.tolist(), orientation=r.tolist())
        detail['waypoints'].append(record)
        session.running_target = (p, r)
        report(row, 'Running')
        if joint_target is not None:
            # The configuration itself was chosen with the target object
            # excluded, but the route to it is planned against the full scene:
            # an approach that clips the object it is about to grasp is a miss.
            path = ex.planner.plan(ex.scene.arm_qpos(), np.asarray(joint_target))
            if path is None:
                result = 'no collision-free approach path'
            elif not ex.follow(path):
                result = 'failed: robot did not settle on the planned path'
            else:
                result = 'ok (collision-free route, %d configs)' % len(path)
                ex.hold(ex.args.settle_steps)
            record['clearance'] = 'planned'
        elif guarded:
            # A joint-space route swings the wrist as it goes, which tips a
            # carried object out of the fingers. Loaded, the checked straight
            # move comes first and keeps the wrist level; the planner is still
            # there when that move is blocked.
            result, record['clearance'] = guarded_move(
                session, p, r, ignore, (source,) if carrying else (),
                plan_first=not carrying)
            if result is None:
                result = ex.line(p, r)
        else:
            # A contact stroke stays at the careful speed the whole way. Timing
            # its free part faster and only slowing for the touch saves about
            # three seconds a run and costs four task layouts across the suite:
            # the arm enters the slow phase still catching up, and finishes
            # centimetres short of a grasp or a placement.
            result = ex.line(p, r)
            record['clearance'] = 'contact'
        # Execution precision is a controller constraint, not demonstration
        # spread. A contact has to land where it was measured; a free-space
        # connector only has to get over the scene, and holding it to contact
        # precision fails moves at the edge of the arm's reach for no gain.
        from waypoint_gmm import check_learned_pose
        contact = primitive in ('grasp', 'place')
        result = check_learned_pose(session, dict(
            position_tolerance=CONTACT_TOLERANCE if contact else CONNECTOR_TOLERANCE,
            orientation_tolerance=CONTACT_ROTATION if contact else CONNECTOR_ROTATION),
            result, p, r)
        if seat is not None and not result.startswith('ok'):
            result = seated(session, source, seat, result)
        if result.startswith('ok') and carrying and not held():
            result = 'failed: target object lost during transport'
        record['result'] = result
        if complete or not result.startswith('ok'):
            report(row, result)
        if not result.startswith('ok'):
            raise ValueError(primitive+': '+result)
        if complete and primitive == 'transport':
            waypoint_finished(row, result)

    def grip(command, check=True):
        result = ex.run_subgoal(sp.Subgoal(command, 'grip', gripper=sp.GRIPPER_CLOSE if command=='close' else sp.GRIPPER_OPEN))
        if check and not result.startswith('ok'):
            raise ValueError(result)
        return result

    try:
        report(0, 'Running')
        grip('open')
        chosen = None
        choices = feasible_grasps(session, skill) if mode == 'constrained' else skill['candidates'][:3]
        if not choices:
            raise ValueError('No collision-free grasp near the demonstrated contacts')
        for attempt, candidate in enumerate(choices):
            source_pose = session.poses()[source]
            gp, gr = compose(source_pose, candidate['grasp'])
            approach_r = source_pose[1]@np.asarray(candidate.get('approach_orientation', candidate['grasp'][1]))
            # Retract and approach from above so a shuffle cannot drag the arm
            # through neighbouring objects along the old demonstration route.
            obstacle_top = path_obstacle_top(world, session.gripper_pose()[0], gp, ignore=(source,))
            safe_z = clearance_height(gp[2], source_pose[0][2], candidate['transport_height'], obstacle_top)
            if mode == 'constrained':
                safe_z = candidate['approach_height']
            here, rotation = session.gripper_pose()
            motion(0, 'move', [here[0],here[1],max(here[2],safe_z)], rotation,
                   guarded=mode == 'constrained')
            # The same one-axis approach point the editor stores and displays.
            motion(1, 'move', raise_in_frame(source_pose, gp, safe_z), approach_r,
                   joint_target=candidate.get('approach_q'))
            # Refresh the contact target just before descending.
            gp, gr = compose(session.poses()[source], candidate['grasp'])
            motion(2, 'grasp', gp, gr, complete=False)
            close_result = grip('close', check=False)
            grasped = held()
            grasp_result = (('ok (grasped)' if grasped else 'failed: nothing between the fingers')
                            if close_result.startswith('ok') else close_result)
            report(2, grasp_result)
            detail['grasp_attempts'].append(dict(candidate=attempt, grasped=grasped, adjustment=candidate.get('adjustment')))
            # This is the actual end of Grasp: the move returned, the gripper
            # finished, and physical acquisition was checked. An early motion
            # abort cannot emit this event, even if it returned from run_skill.
            waypoint_finished(2, grasp_result)
            if not close_result.startswith('ok'):
                raise ValueError(close_result)
            if grasped:
                chosen = candidate
                break
            if getattr(session, 'continue_on_failure', False):
                # GUI playback advances from this failed Grasp to Transport.
                # Another candidate would revisit Move and restart the approach.
                raise ValueError('failed: nothing between the fingers')
            grip('open')
        if chosen is None:
            raise ValueError('No demonstrated grasp acquired the target object')
        lifted_from = np.asarray(world.pos(source), float).copy()
        start_z = float(lifted_from[2])
        here, gr = session.gripper_pose()
        support = supporting_bodies(world, destination)
        carried_top = path_obstacle_top(world, here, session.poses()[destination][0],
                                        ignore=(source,)+support)
        underside = world.aabb(source)[0][2]
        safe_z = carry_height(safe_z, here[2], carried_top, underside)
        # Straight up in the world from the commanded grasp point. Vertical in
        # the *object's* frame is not vertical at all once that frame is tilted:
        # it drags the object sideways across its support before it clears.
        # Taking the commanded point rather than the reached one keeps this row
        # exactly where the plan says, instead of inheriting a settling error.
        motion(3, 'transport', [gp[0], gp[1], safe_z], gr, carrying=True, complete=False)
        if world.pos(source)[2]-start_z < .025:
            raise ValueError('Target did not lift with the gripper')
        report(3, 'ok (transported)')
        waypoint_finished(3, 'ok (transported)')
        # Actual EEF->object transform compensates for grasp position/orientation.
        carried = relative(session.poses()[source], session.gripper_pose())
        goal_object = compose(session.poses()[destination], chosen['release_object'])
        release_p, release_r = compose(goal_object, inverse(carried))
        if mode == 'keyframes':
            release_p, release_r = compose(session.poses()[destination], chosen['release_eef'])
        transport_r = session.poses()[destination][1]@np.asarray(
            chosen.get('transport_orientation', relative([release_p,release_r], session.poses()[destination])[1]))
        # The loaded traverse is the one long free-space move in the task, and
        # the only one that can sweep the carried object through the scene.
        # Its height comes from the same search the editor previews: lower it
        # until the arm can hold the pose, never below what the path needs.
        destination_pose = session.poses()[destination]
        floor = carry_height(release_p[2]+MINIMUM_DESCENT, here[2], carried_top, underside)
        carry_p, carry_r = traverse_pose(
            destination_pose, release_p,
            max(safe_z, release_p[2]+PLACE_CLEARANCE, floor), floor, (transport_r, release_r),
            (lambda p, r: reachable(session, p, r, ignore=support, carried=(source,)))
            if mode == 'constrained' else None)
        motion(4, 'transport', carry_p, carry_r, carrying=True,
               guarded=mode == 'constrained', ignore=support)
        motion(5, 'place', release_p, release_r, carrying=True, seat=goal_object[0])
        report(6, 'Running')
        grip('open')
        detail['waypoints'].append(dict(primitive='release', position=release_p.tolist(), orientation=release_r.tolist()))
        ex.hold(30)
        if not session.env.check_success():
            # Remove supporting finger contact before evaluating the settled goal.
            p, r = session.gripper_pose()
            motion(6, 'move', p+[0,0,.08], r, complete=False)
            ex.hold(30)
        report(6, 'ok (released)')
        detail.update(completed=True, result='Finished learned object-relative skill', task_success=bool(session.env.check_success()))
    except ValueError as exc:
        detail.update(result=str(exc), task_success=bool(session.env.check_success()))
    return detail


def feasible_grasps(session, skill, limit=3):
    """Choose the smallest collision-free adjustment to demonstrated contacts.

    Local offsets stay within 25 mm and yaw within 30 degrees. For a round
    bowl in a drawer task, rotated versions of the demonstrated rim grasp are
    considered only after direct copies fail. Every approach is checked in
    the live scene before contact execution.
    """
    from waypoint_rrt import Scene
    source = skill['source']
    frame = session.poses()[source]
    scene = Scene(session.env, ignore_bodies=(source,))
    proposals = []
    for index, c in enumerate(skill['candidates'][:3]):
        for dx in [0., -.0125, .0125, -.025, .025]:
            for dy in [0., -.0125, .0125, -.025, .025]:
                for yaw in [0., -.26, .26, -.52, .52]:
                    cost = np.hypot(dx,dy)+.03*abs(yaw)+index*.0005
                    proposals.append((cost,index,dx,dy,yaw,0.))
    if skill.get('pregrasp_push') and 'bowl' in source:
        # The bowl rim is rotationally symmetric. Reuse the measured EEF/object
        # relation at other rim angles when the original side faces a drawer.
        for symmetry in (-np.pi/2, np.pi/2, np.pi):
            for index, c in enumerate(skill['candidates'][:3]):
                for dx in [0., -.0125, .0125]:
                    for dy in [0., -.0125, .0125]:
                        for yaw in [0., -.26, .26]:
                            cost = .1+np.hypot(dx,dy)+.03*abs(yaw)+index*.0005
                            proposals.append((cost,index,dx,dy,yaw,symmetry))
    chosen = []
    for _, index, dx, dy, yaw, symmetry in sorted(proposals):
        session.tick()
        c = dict(skill['candidates'][index])
        p, r = map(np.asarray, c['grasp'])
        approach_r = np.asarray(c.get('approach_orientation', c['grasp'][1]))
        rim_rotation = Rotation.from_rotvec([0,0,symmetry]).as_matrix()
        yaw_rotation = Rotation.from_rotvec([0,0,yaw]).as_matrix()
        local = [rim_rotation@p+np.array([dx,dy,0.]),
                 yaw_rotation@rim_rotation@r]
        local_approach_r = yaw_rotation@rim_rotation@approach_r
        gp, gr = compose(frame, local)
        q = scene.arm_qpos()
        feasible = True
        for dz in [.06,.04,.02,0.]:
            q, error = scene.ik(gp+[0,0,dz],q_init=q,target_ori=gr)
            if error > .003 or not scene.valid(q):
                feasible = False
                break
        if feasible:
            obstacle_top = path_obstacle_top(session.world, session.gripper_pose()[0], gp,
                                             ignore=(source,))
            floor = max(gp[2]+SCENE_CLEARANCE,
                        (obstacle_top+SCENE_CLEARANCE) if obstacle_top is not None else -np.inf)
            # Search downward from the height the plan shows, so an unobstructed
            # approach runs at exactly the offset stored in the editor.
            preferred = max(floor, clearance_height(gp[2], frame[0][2], c['transport_height'], obstacle_top))
            approach_height = None
            for z in np.linspace(preferred, floor, max(2, int(np.ceil((preferred-floor)/.015))+1)):
                # The same one-axis point the plan stores, so the solved
                # configuration lands exactly on the displayed offset.
                high_q, error = scene.ik(raise_in_frame(frame, gp, z), q_init=q,
                                         target_ori=frame[1]@local_approach_r)
                if error < .001 and scene.valid(high_q):
                    approach_height = float(z)
                    break
            if approach_height is None:
                continue
            c['approach_height'] = approach_height
            c['approach_q'] = high_q.tolist()
            c['grasp'] = [local[0].tolist(),local[1].tolist()]
            c['approach_orientation'] = local_approach_r.tolist()
            c['adjustment'] = dict(demo_candidate=index, offset=[dx,dy,0.],
                                   yaw=yaw, symmetry_yaw=symmetry)
            chosen.append(c)
            if len(chosen) == limit:
                break
    return chosen
