"""Editable keyframe representation and GUI model packaging for VLA skills."""
import copy
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np

from waypoint_chain import PREVIOUS, WORLD, relative_waypoint
from waypoint_gmm import GMMChain, GMMWaypoint
from waypoint_learning import Config, learn, infer_goals, detect_events, event_signature
from waypoint_skill import (learn_skill, learn_sequential_skill, learn_push_skill,
                            learn_object_push_skill,
                            compose, inverse, relative, clearance_height, carry_height,
                            raise_in_frame, traverse_pose, PLACE_CLEARANCE, MINIMUM_DESCENT,
                            PUSH_CLEARANCE)


def model_from_skill(skill, demos, config=None):
    c = config or Config()
    skill = copy.deepcopy(skill)
    if skill.get('training_demo_files'):
        selected = set(skill['training_demo_files'])
        demos = [demo for demo in demos if demo.get('_path') in selected]
    for stage in skill.get('stages', [skill]):
        blocks = [stage.get(name) or {} for name in
                  ('pregrasp_push', 'postplace_push', 'object_push')]
        for candidate in (list(stage['candidates'])
                          + [c for block in blocks for c in block.get('candidates', [])]):
            demo = next((d for d in demos if d.get('_path') == candidate.get('demo_file')), demos[0])
            sample = demo['samples'][0]
            candidate['initial_eef'] = [sample['position'], sample['orientation']]
    return dict(version=3, model_type='object_relative_vla_waypoints',
                suite=skill['suite'], task_id=skill['task_id'],
                instruction=demos[0].get('instruction', demos[0].get('language', '')),
                config=asdict(c), object_relative_skill=skill,
                demonstration_files=[d.get('_path') for d in demos],
                used_demonstrations=list(range(len(demos))),
                waypoints=[dict(primitive=p) for p in skill['primitives']],
                replay_validation='not_run')


def missing_articulation_evidence(demos):
    """Old recordings cannot identify a drawer push from arm motion alone."""
    for demo in demos:
        goals, _ = infer_goals(demo)
        articulated = any(g and g[0].lower() in ('open','close','turnon','turnoff') for g in goals)
        if not articulated:
            continue
        measured = any(sample.get('joint_positions') or sample.get('contact_state')
                       for sample in demo.get('samples', []))
        if not measured:
            return True
    return False


def archived_matching_demos(demos, task_folder):
    """Reuse only successful recordings of the same task and initial layout."""
    if not demos or task_folder is None:
        return []
    first = demos[0]
    initial = first.get('initial_object_poses', first.get('poses', {}))
    known = {str(Path(d['_path']).resolve()) for d in demos if d.get('_path')}
    result = []
    for path in sorted(Path(task_folder).glob('*/demonstrations/demo_*.json')):
        if str(path.resolve()) in known:
            continue
        try:
            d = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        other = d.get('initial_object_poses', d.get('poses', {}))
        if (not d.get('success') or not d.get('samples') or
                d.get('placement_mode') not in (None, 'fixed_unshuffled') or
                (d.get('suite'), d.get('task_id')) != (first.get('suite'), first.get('task_id')) or
                d.get('language', d.get('instruction')) != first.get('language', first.get('instruction')) or
                set(other) != set(initial)):
            continue
        if not all(np.allclose(other[name][0], initial[name][0], atol=.003, rtol=0) and
                   np.allclose(other[name][1], initial[name][1], atol=.02, rtol=0)
                   for name in initial):
            continue
        result.append(dict(d, _path=str(path.resolve())))
    return result


def can_fit_editor_model(demos, task_folder=None):
    """Check whether enough compatible evidence exists before ending recording."""
    for learner in (learn_sequential_skill, learn_skill, learn_push_skill,
                    learn_object_push_skill):
        try:
            learner(demos)
            return True
        except ValueError:
            pass
    all_demos = list(demos) + archived_matching_demos(demos, task_folder)
    try:
        learn_sequential_skill(all_demos)
        return True
    except ValueError:
        pass
    counts = Counter(event_signature(detect_events(d)) for d in all_demos
                     if d.get('success') and d.get('samples'))
    return bool(counts) and max(counts.values()) >= 2


def fit_editor_model(demos, config=None, overrides=None, reuse_recordings_from=None):
    if overrides:
        return learn(demos, config, overrides)
    for learner in (learn_sequential_skill, learn_skill, learn_push_skill,
                    learn_object_push_skill):
        try:
            return model_from_skill(learner(demos), demos, config)
        except ValueError:
            pass
    if missing_articulation_evidence(demos):
        raise ValueError('Fresh VLA recordings with drawer joint/contact measurements are required to learn a minimal opening plan')
    try:
        return minimal_gmm_waypoints(learn(demos, config))
    except ValueError as exc:
        if ('No two demonstrations share an ordered event signature' not in str(exc)
                or reuse_recordings_from is None):
            raise
        combined = list(demos) + archived_matching_demos(demos, reuse_recordings_from)
        if len(combined) == len(demos):
            raise
        try:
            skill = learn_sequential_skill(combined)
            model = model_from_skill(skill, combined, config)
        except ValueError:
            model = minimal_gmm_waypoints(learn(combined, config))
        current_paths = {d.get('_path') for d in demos}
        model['supplemented_demonstrations'] = [
            path for path in model['demonstration_files']
            if path not in current_paths]
        return model


def minimal_gmm_waypoints(model):
    """Reduce a fallback GMM fit to one waypoint per demonstrated primitive.

    The probabilistic extractor keeps every geometric detail it can justify,
    which is informative to inspect and unusable as an editable plan. Option 2
    promises a minimal list, so only the closing boundary of each primitive run
    and the poses that carry a gripper event survive. Move and transport
    boundaries reconstruct at local phase 1, where the connector correction is
    already zero, so dropping their interior samples changes no target.
    """
    nodes = model.get('waypoints') or []
    if model.get('object_relative_skill') or len(nodes) < 2:
        return model

    def phase(node):
        # Segment boundaries are semantic anchors in their own right; only
        # repeated detail *inside* one interval is redundant.
        return (node.get('segment'), node.get('primitive') or node.get('semantic_phase') or '')

    keep = [node for i, node in enumerate(nodes)
            if i+1 == len(nodes) or phase(nodes[i+1]) != phase(node)
            or node.get('gripper_command', 'keep') != 'keep']
    if len(keep) == len(nodes):
        return model
    return dict(model, waypoints=keep, reference_scheme='primitive_boundaries',
                waypoint_reduction=dict(policy='primitive_boundaries',
                                        before=len(nodes), after=len(keep)))


def upgrade_model(model):
    """Use existing VLA evidence; never substitute symbolic grasp templates."""
    if ((model.get('object_relative_skill') and
         model['object_relative_skill'].get('representation') == 'single_reference_minimal' and
         model['object_relative_skill'].get('orientation_scheme') == 'precontact_object_and_previous') or
        model.get('reference_scheme') == 'single_reference_minimal'):
        return model
    paths = model.get('demonstration_files', [])
    if len(paths) < 2 or any(not p or not Path(p).is_file() for p in paths):
        return model
    # GMM may have discarded successful recordings with different contact timing.
    # Reuse those original recordings too when rebuilding the semantic skill.
    candidates = {Path(p).resolve() for p in paths}
    for parent in {p.parent for p in candidates}:
        candidates.update(p.resolve() for p in parent.glob('demo_*.json') if not p.name.endswith('.state.json'))
    demos = []
    for path in sorted(candidates):
        d = json.loads(path.read_text())
        if (d.get('suite'), d.get('task_id')) == (model['suite'], model['task_id']) and d.get('success') and d.get('samples'):
            demos.append(dict(d, _path=str(path)))
    if missing_articulation_evidence(demos):
        return model
    for learner in (learn_sequential_skill, learn_skill, learn_push_skill,
                    learn_object_push_skill):
        try:
            return model_from_skill(learner(demos), demos, Config(**model['config']))
        except ValueError:
            continue
    try:
        return minimal_gmm_waypoints(
            learn(demos, Config(**model['config']), model.get('reference_overrides')))
    except ValueError:
        return model


def nominal_targets(skill, poses, start, obstacles=None, bottom=None, reach=None,
                    support_of=None):
    """Nominal targets shared by the preview, the editor and the executor.

    ``obstacles(a, b, ignore)`` gives the height that has to be cleared between
    two points and ``bottom(name)`` the world height of a body's underside, so
    the preview computes exactly the clearances the executor will use. The live
    constrained planner may still lower a blocked approach.
    """
    if skill.get('stages'):
        targets = []
        for stage in skill['stages']:
            part = nominal_targets(stage, poses, start, obstacles, bottom, reach, support_of)
            targets.extend(part)
            start = part[-1]
        return targets
    push, shove, shut = (skill.get('pregrasp_push'), skill.get('object_push'),
                         skill.get('postplace_push'))
    push_targets = []
    if shove:
        # Each stroke leaves the object somewhere new; the next contact is
        # measured against that predicted pose, not the layout on the table.
        frame = poses[shove['source']]
        goal = poses[shove['destination']]
        for stroke in shove['candidates'][0]['strokes']:
            contact = compose(frame, stroke['contact'])
            staged = (raise_in_frame(frame, contact[0], contact[0][2]+PUSH_CLEARANCE),
                      contact[1])
            push_targets += [staged, contact, compose(goal, stroke['finish'])]
            frame = compose(goal, stroke['target_object'])
        start = push_targets[-1]
    if push:
        pc = push['candidates'][0]
        drawer_frame = poses[push['joint']]
        push_targets += [compose(drawer_frame, pc[k])
                         for k in ('approach', 'contact', 'finish')]
        start = push_targets[-1]
    if not skill.get('candidates'):
        return push_targets
    c = skill['candidates'][0]
    if push:
        poses = dict(poses, **{skill['destination']: tuple(np.asarray(v) for v in c['opened_destination'])})
    source = poses[skill['source']]
    gp, gr = compose(source, c['grasp'])
    approach_r = source[1] @ np.asarray(c.get('approach_orientation', c['grasp'][1]))
    obstacle = (lambda a, b, ignore: None) if obstacles is None else obstacles
    high = clearance_height(gp[2], source[0][2], c['transport_height'],
                            obstacle(start[0], gp, (skill['source'],)))
    carried = inverse(c.get('stable_grasp', c['grasp']))  # Measured EEF->object after closure
    desired = compose(poses[skill['destination']], c['release_object'])
    pp, pr = compose(desired, inverse(carried))
    destination = poses[skill['destination']]
    support = () if support_of is None else support_of(skill['destination'])
    carried_top = obstacle(gp, destination[0], (skill['source'],)+support)
    underside = None if bottom is None else bottom(skill['source'])
    floor = carry_height(pp[2]+MINIMUM_DESCENT, gp[2], carried_top, underside)
    ceiling = max(carry_height(high, gp[2], carried_top, underside),
                  pp[2]+PLACE_CLEARANCE, floor)
    transport_r = destination[1] @ np.asarray(
        c.get('transport_orientation', relative((pp, pr), destination)[1]))
    # Each connector travels along one axis of the frame it is stored against,
    # so its saved offset differs from the contact it serves in one number.
    carry_p, carry_r = traverse_pose(
        destination, pp, ceiling, floor, (transport_r, pr),
        None if reach is None else (lambda p, r: reach(p, r, support)))
    # The arm first lifts clear of wherever it stands, keeping its own x and y.
    retract = (np.array([start[0][0], start[0][1], max(start[0][2], high)]), start[1])
    approach = (raise_in_frame(source, gp, high), approach_r)
    # The grasped object leaves its support straight up in the world, on the
    # grasp's own x and y, before anything travels sideways. Vertical in the
    # object's frame would drag it sideways whenever that frame is tilted.
    lift = (np.array([gp[0], gp[1],
                      carry_height(high, gp[2], carried_top, underside)]), gr)
    targets = push_targets+[retract, approach, (gp, gr), lift,
            (carry_p, carry_r), (pp, pr), (pp, pr)]
    if shut:
        sc = shut['candidates'][0]
        frame = poses[shut['joint']]
        targets += [compose(frame, sc[k]) for k in ('approach', 'contact', 'finish')]
    return targets


def skill_rows(skill):
    """One row per primitive of a skill plan: name, reference, stage start.

    The editor list, the nominal preview and the executor all walk this, so a
    plan cannot grow a row in one place and lose it in another.
    """
    if skill.get('stages'):
        return [row for stage in skill['stages'] for row in skill_rows(stage)]
    push, shove, shut = (skill.get('pregrasp_push'), skill.get('object_push'),
                         skill.get('postplace_push'))
    rows = []
    if shove:
        # Stage above the object, descend onto it, then shove.
        rows += [('move', shove['source']), ('move', shove['source']),
                 ('push', shove['destination'])]*shove['strokes']
    if push:
        # Stand off the handle, close on it, then push.
        rows += [('move', push['joint']), ('move', push['joint']), ('push', push['joint'])]
    if skill.get('candidates'):
        # Every motion the executor drives has its own row: the arm lifts clear
        # of where it is, crosses to above the object, descends, lifts the
        # object, crosses to above the goal, lowers it, lets go.
        rows += [('move', PREVIOUS), ('move', skill['source']), ('grasp', skill['source']),
                 ('transport', skill['source']), ('transport', skill['destination']),
                 ('place', skill['destination']), ('release', PREVIOUS)]
    if shut:
        rows += [('move', shut['joint']), ('move', shut['joint']), ('push', shut['joint'])]
    return [(name, reference, i == 0) for i, (name, reference) in enumerate(rows)]


def effective_poses(skill, poses):
    """Frames a plan is stored against; a pushed drawer is where it ends up.

    Transport and place happen after the push, so their single reference is the
    opened drawer. Storing them against the closed frame would show an offset
    the executor never aims at.
    """
    result = dict(poses)
    for stage in skill.get('stages', [skill]):
        if stage.get('pregrasp_push') and stage.get('source'):
            result[stage['destination']] = tuple(
                np.asarray(v) for v in stage['candidates'][0]['opened_destination'])
    return result


def skill_waypoints(skill, targets, poses, start):
    """Express nominal targets as editable single-reference waypoints."""
    rows = skill_rows(skill)
    if len(targets) != len(rows):
        raise ValueError('Skill preview produced %d targets for %d plan rows'
                         % (len(targets), len(rows)))
    poses = effective_poses(skill, poses)
    waypoints, previous = [], tuple(np.asarray(v) for v in start)
    for (p, r), (primitive, reference, stage_start) in zip(targets, rows):
        release = primitive == 'release'
        w = relative_waypoint(primitive.title(), p, r, reference, poses, previous,
                              motion='line',
                              gripper='open' if release or stage_start else 'keep',
                              gripper_after='close' if primitive == 'grasp' else 'keep')
        waypoints.append(GMMWaypoint(**asdict(w), skill_role=primitive))
        previous = np.asarray(p), np.asarray(r)
    return waypoints


def training_scene(skill):
    """The recorded layout a skill was measured in, and its first wrist pose."""
    poses, start = {}, None
    for stage in skill.get('stages', [skill]):
        for push in (stage.get('pregrasp_push'), stage.get('postplace_push')):
            if not push:
                continue
            pc = push['candidates'][0]
            poses.setdefault(push['joint'], pc['initial_frame'])
            if start is None:
                start = pc.get('initial_eef') or compose(
                    tuple(np.asarray(v) for v in pc['initial_frame']), pc['approach'])
        shove = stage.get('object_push')
        if shove:
            sc = shove['candidates'][0]
            poses.setdefault(shove['source'], sc['initial_source'])
            poses.setdefault(shove['destination'], sc['initial_destination'])
            if start is None:
                start = sc.get('initial_eef') or compose(
                    tuple(np.asarray(v) for v in sc['initial_source']),
                    sc['strokes'][0]['contact'])
        if stage.get('candidates'):
            c = stage['candidates'][0]
            poses.setdefault(stage['source'], c['initial_source'])
            poses.setdefault(stage['destination'], c['initial_destination'])
            if start is None:
                start = c.get('initial_eef', [c['initial_source'][0], c['grasp'][1]])
    return ({n: tuple(np.asarray(v) for v in pose) for n, pose in poses.items()},
            tuple(np.asarray(v) for v in start))


SKILL_NOTES = dict(
    stages='pi0.5 sequential minimal single-reference waypoints; replay unverified',
    articulation='pi0.5 minimal articulation waypoints; measured joint travel; replay unverified',
    object_push='pi0.5 minimal push waypoints; measured object travel; replay unverified',
    pick_place='pi0.5 minimal single-reference waypoints; grasp and placement feedback; replay unverified')


def chain_from_skill_model(model, model_path=''):
    skill = copy.deepcopy(model['object_relative_skill'])
    poses, start = training_scene(skill)
    waypoints = skill_waypoints(skill, nominal_targets(skill, poses, start), poses, start)
    skill['waypoint_template'] = [asdict(w) for w in waypoints]
    note = ('stages' if skill.get('stages') else
            'pick_place' if skill.get('candidates') else
            'object_push' if skill.get('object_push') else 'articulation')
    chain = GMMChain(model['suite'], model['task_id'], model['instruction'], waypoints,
                     SKILL_NOTES[note], max_waypoints=model['config']['max_waypoints'],
                     model_path=str(model_path), execution_skill=skill)
    chain.validate()
    return chain
