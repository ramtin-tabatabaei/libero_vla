"""Compact task plans without removing contact or gripper timing constraints."""
import copy
import dataclasses

import numpy as np

from waypoint_chain import Chain, MAX_WAYPOINTS, PREVIOUS, WORLD, resolve, relative_waypoint


def phase_plan(demo):
    """Prefer six measured pick/place phases; retain compact multi-stage plans.

    Phases are inferred from contact, gripper changes and vertical travel, not
    task names. A drawer or multi-object demonstration keeps its extra events.
    """
    chain = demonstration_plan(demo)
    poses = {n: (np.array(p), np.array(r)) for n, (p, r) in demo["poses"].items()}
    samples = demo["samples"]
    points = np.array([s["position"] for s in samples] + [demo["end_position"]])
    orientations = np.array([s["orientation"] for s in samples] + [demo["end_orientation"]])
    previous = points[0], orientations[0]
    events = []
    for w in chain.waypoints:
        previous = resolve(w, poses, previous)
        if w.gripper_after != "keep":
            scores = np.linalg.norm(points-previous[0], axis=1) + np.linalg.norm(orientations-previous[1], axis=(1, 2))*.01
            events.append((int(np.argmin(scores)), w.gripper_after))
    simple = [state for _, state in events] in (["close"], ["close", "open"])
    if not simple:
        return chain
    grasp = events[0][0]
    release = events[1][0] if len(events) == 2 else len(points)-1
    # Only apply pick/place phases when contact actually identifies a movable
    # object. Closing on a drawer handle must retain its learned articulation.
    contacts = samples[min(grasp, len(samples)-1)].get("grasped", [])
    if not contacts or release-grasp < 4 or grasp < 3:
        return chain
    held = contacts[0]
    if held not in poses:
        return chain
    top = grasp + int(np.argmax(points[grasp:release+1, 2]))
    height = points[top, 2]-points[grasp, 2]
    if height < .04 or top <= grasp+1:
        return chain
    # Last high approach sample before the descent, first high sample after
    # grasp, then the last high sample before the final descent.
    before = np.flatnonzero(points[1:grasp, 2] >= points[grasp, 2]+min(.10, height)) + 1
    approach = int(before[-1]) if len(before) else max(1, grasp//2)
    high = np.flatnonzero(points[grasp+1:release, 2] >= points[top, 2]-.02) + grasp+1
    if not len(high):
        return chain
    lift = int(high[0])
    transfer = int(high[-1])
    down_candidates = np.arange(transfer+1, release)
    lower = (int(down_candidates[np.argmin(np.abs(points[down_candidates, 2]-(points[release, 2]+.035)))])
             if len(down_candidates) else release)
    destination_names = [n for n in demo["objects"]+demo["fixtures"] if n != held]
    destination = min(destination_names, key=lambda n: np.linalg.norm(points[release]-poses[n][0])) if destination_names else WORLD
    stages = [(approach, "Approach", held, "keep"),
              (grasp, "Reach & close gripper", held, "close"),
              (lift, "Lift", held, "keep"),
              (transfer, "Transfer", destination, "keep"),
              (lower, "Going down", destination, "keep"),
              (release, "Finish & open gripper", destination, "open")]
    # Keep substantial learned bends (e.g. approaching around a cabinet) when
    # six straight segments would erase them. These are visible extra rows.
    while len(stages) < MAX_WAYPOINTS:
        cuts = sorted({0, *(s[0] for s in stages)})
        worst = (0., 0)
        for a, b in zip(cuts, cuts[1:]):
            if b-a < 2:
                continue
            direction = points[b]-points[a]
            t = np.clip((points[a+1:b]-points[a]) @ direction/max(float(direction @ direction), 1e-12), 0, 1)
            deviations = np.linalg.norm(points[a+1:b]-(points[a]+t[:, None]*direction), axis=1)
            j = int(np.argmax(deviations))
            worst = max(worst, (float(deviations[j]), a+1+j))
        if worst[0] <= .05:
            break
        index = worst[1]
        stages.append((index, "Path turn", held if index <= lift else destination, "keep"))
    stages.sort(key=lambda s: s[0])
    result = []
    previous = points[0], orientations[0]
    last_index = 0
    for index, name, ref, after in stages:
        if index <= last_index:
            if after != "keep" and result:
                result[-1].gripper_after = after
                result[-1].name = name
            continue
        w = relative_waypoint(name, points[index], orientations[index], ref, poses, previous,
                              motion="line", gripper="open" if not result else "keep", gripper_after=after)
        result.append(w)
        previous = points[index], orientations[index]
        last_index = index
    # Closing/releasing can precede an essential retreat in multi-step tasks.
    # Fall back to the general extractor if substantial post-release motion exists.
    if np.linalg.norm(points[-1]-points[release]) > .08:
        return chain
    source = (f"pi0.5 phase plan; {len(samples)} measured steps → {len(result)} waypoints; "
              f"rollout success={demo['success']}; replay unverified")
    return dataclasses.replace(chain, waypoints=result, source=source)


def choose_demonstration(demonstrations, waypoint_count=None):
    """Choose a coherent measured route; do not average incompatible grasps."""
    candidates, errors = [], []
    for index, demo in enumerate(demonstrations):
        try:
            chain = (phase_plan(demo) if waypoint_count is None else
                     demonstration_plan(demo, max_waypoints=waypoint_count, exact_count=True))
            distance = np.sum(np.linalg.norm(np.diff([s["position"] for s in demo["samples"]], axis=0), axis=1))
            candidates.append(((bool(demo["success"]), -len(chain.waypoints), -distance), index, chain))
        except ValueError as exc:
            errors.append(str(exc))
    if not candidates:
        target = "at most 9" if waypoint_count is None else str(waypoint_count)
        raise ValueError(f"Could not extract a plan of {target} waypoints: " + "; ".join(errors))
    _, index, chain = max(candidates, key=lambda c: c[0])
    passed = sum(bool(d["success"]) for d in demonstrations)
    chain.source = chain.source.replace("; replay unverified", f"; selected demonstration {index+1}/{len(demonstrations)}; "
                                       f"{passed}/{len(demonstrations)} rollouts succeeded; replay unverified")
    return chain, index


def demonstration_plan(demo, max_waypoints=MAX_WAYPOINTS, position_tolerance=.015, contact_delay=0,
                       exact_count=False):
    """Select actual pi0.5 key poses, preserving every gripper transition.

    The start pose is implicit. Grip changes happen after reaching their measured
    contact pose, leaving the budget for approach/lift/transfer shape. No symbolic
    task templates or hidden intermediate motion waypoints are used.
    """
    from scipy.spatial.transform import Rotation, Slerp
    if not 1 <= max_waypoints <= MAX_WAYPOINTS:
        raise ValueError("Plans must contain between 1 and 9 waypoints.")
    samples = demo["samples"]
    if not samples:
        raise ValueError("pi0.5 did not produce any motion samples.")
    points = np.array([s["position"] for s in samples] + [demo["end_position"]])
    rotations = np.array([s["orientation"] for s in samples] + [demo["end_orientation"]])
    raw_grips = [s["gripper"] for s in samples]
    grips = raw_grips.copy()
    # Short open/close pulses often leave the actual fingers holding the same
    # object. Require a sustained command unless measured contact changes.
    state = raw_grips[0]
    a = 0
    while a < len(grips):
        b = a+1
        while b < len(grips) and raw_grips[b] == raw_grips[a]:
            b += 1
        before_contact = set(samples[max(0, a-1)].get("grasped", []))
        after_contact = set(samples[min(len(samples)-1, b)].get("grasped", []))
        if b-a >= 6 or before_contact != after_contact or b == len(grips):
            state = raw_grips[a]
        grips[a:b] = [state]*(b-a)
        a = b
    commanded = [i for i in range(1, len(grips)) if grips[i] != grips[i-1]]
    changes = {}
    for k, i in enumerate(commanded):
        index = i
        if grips[i] == "close":
            end = commanded[k+1] if k+1 < len(commanded) else len(grips)
            # pi0.5 closes WHILE descending. A waypoint executor closes at rest,
            # so use measured contact rather than the earlier close command.
            contact = next((j for j in range(i, min(end, i+25)) if samples[j].get("grasped")), None)
            if contact is not None:
                index = min(contact + contact_delay, end-1)
            elif "grasped" not in samples[i]:
                # Older archived rollouts did not include contact observations.
                index = i + int(np.argmin(points[i:min(end, i+15), 2]))
        changes[index] = grips[i]
    keep = {0, len(points)-1, *changes}
    omitted = 0
    # A failed rollout may repeat unsuccessful attempts dozens of times. Keep
    # its strongest complete grasp/release attempts as an explicitly limited
    # draft rather than inventing a successful demonstration or exceeding nine.
    if len(keep)-1 > max_waypoints and not demo["success"] and not exact_count:
        events = sorted(changes)
        pairs = [(a, b) for a, b in zip(events, events[1:])
                 if changes[a] == "close" and changes[b] == "open"]
        ranked = sorted(pairs, key=lambda pair: (
            any(s.get("grasped") for s in samples[pair[0]:pair[1]]),
            np.linalg.norm(points[pair[1]]-points[pair[0]])), reverse=True)
        selected_events = {i for pair in ranked[:max(1, (max_waypoints-1)//2)] for i in pair}
        omitted = len(changes)-len(selected_events)
        changes = {i: changes[i] for i in sorted(selected_events)}
        keep = {0, len(points)-1, *changes}
    if len(keep)-1 > max_waypoints:
        raise ValueError(f"Demonstration has {len(changes)} gripper transitions; cannot preserve them and the final pose in {max_waypoints} waypoints. Increase the waypoint count.")

    def segment_error(a, b):
        if b-a < 2:
            return 0., a
        line = points[b]-points[a]
        distance = float(line @ line)
        fractions = (np.clip((points[a+1:b]-points[a]) @ line / distance, 0, 1)
                     if distance > 1e-10 else np.linspace(0, 1, b-a+1)[1:-1])
        deviation = np.linalg.norm(points[a+1:b]-(points[a]+fractions[:, None]*line), axis=1)
        expected = Slerp([0, 1], Rotation.from_matrix(rotations[[a, b]]))(fractions)
        angles = (expected.inv()*Rotation.from_matrix(rotations[a+1:b])).magnitude()
        scores = np.maximum(deviation / position_tolerance, angles / np.radians(12))
        index = int(np.argmax(scores))
        return float(scores[index]), a+1+index

    while len(keep)-1 < max_waypoints:
        cuts = sorted(keep)
        score, index = max(segment_error(a, b) for a, b in zip(cuts, cuts[1:]))
        if score <= 1 and not exact_count:
            break
        if index in keep:
            # Straight or stationary sections can still supply measured poses.
            a, b = max(zip(cuts, cuts[1:]), key=lambda pair: pair[1]-pair[0])
            if b-a < 2:
                raise ValueError(f"Not enough measured samples for {max_waypoints} waypoints.")
            index = (a+b)//2
        keep.add(index)

    poses = {n: (np.array(p), np.array(r)) for n, (p, r) in demo["poses"].items()}
    anchors = demo["objects"] + demo["fixtures"]
    result, previous, held = [], (points[0], rotations[0]), ""
    cuts = sorted(keep)[1:]
    for k, index in enumerate(cuts):
        # Pick a frame for the destination of this grip phase, then express all
        # its learned bends relative to that frame. The held source is excluded.
        phase_end = next((j for j in sorted(changes) if j >= index), len(points)-1)
        candidates = [name for name in anchors if name != held]
        ref = min(candidates, key=lambda n: np.linalg.norm(points[phase_end]-poses[n][0])) if candidates else WORLD
        after = changes.get(index, "keep")
        if after == "close":
            name = "Reach & close gripper"
        elif after == "open":
            name = "Place & open gripper"
        elif index == len(points)-1:
            name = "Finish"
        elif k and points[index][2]-previous[0][2] > .025:
            name = "Lift" if held else "Raise"
        else:
            name = "Transfer" if held else "Approach"
        w = relative_waypoint(name, points[index], rotations[index], ref, poses, previous,
                              motion="line", gripper=grips[0] if not result else "keep", gripper_after=after)
        result.append(w)
        previous = points[index], rotations[index]
        if after == "close" and demo["objects"]:
            nearest = min(demo["objects"], key=lambda n: np.linalg.norm(points[index]-poses[n][0]))
            held = nearest if np.linalg.norm(points[index]-poses[nearest][0]) < .16 else ""
        elif after == "open":
            held = ""
    source = (f"pi0.5 demonstration; {len(samples)} measured steps → {len(result)} waypoints; "
              f"rollout success={demo['success']}" +
              (f"; limited draft: {omitted} retry transitions omitted" if omitted else "") + "; replay unverified")
    chain = Chain(demo["suite"], demo["task_id"], demo["language"], result, source)
    chain.validate()
    return chain


def compact_gripper_steps(chain):
    """Execute stationary gripper commands before the following motion.

    Closing happens after descent and before lift; opening happens after lowering
    and before retreat. No positions are discarded by this transformation.
    """
    nodes = copy.deepcopy(chain.waypoints)
    result = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if (node.motion == "grip" and i + 1 < len(nodes)
                and nodes[i + 1].motion in ("reach", "line")
                and nodes[i + 1].gripper in ("keep", node.gripper)
                and not nodes[i + 1].grasp):
            following = nodes[i + 1]
            following.gripper, following.grasp = node.gripper, node.grasp
            following.ignore = list(dict.fromkeys(node.ignore + following.ignore))
            following.name = ("Grasp & " if node.gripper == "close" else "Release & ") + following.name
            result.append(following)
            i += 2
        else:
            result.append(node)
            i += 1
    return dataclasses.replace(chain, waypoints=result)


def simplify_demonstration(chain, poses, start, tolerance=.025):
    """Remove redundant straight travel samples; retain contact transitions.

    This is only for sampled line trajectories. Symbolic targets with carrying,
    arcs or hold-XY constraints are intentionally left intact.
    """
    points, previous = [], start
    protected = {0, len(chain.waypoints)-1}
    for i, w in enumerate(chain.waypoints):
        if w.motion not in ("line", "grip") or w.carry or w.seat or w.hold_xy:
            return copy.deepcopy(chain)
        previous = previous if w.motion == "grip" else resolve(w, poses, previous)
        points.append(previous)
        if w.motion == "grip" or w.gripper != "keep":
            protected.update(j for j in (i-1, i, i+1) if 0 <= j < len(chain.waypoints))
        if w.reference == PREVIOUS and w.motion != "grip":
            protected.update((i-1, i))
    if len(points) < 3:
        return copy.deepcopy(chain)
    keep = set(protected)

    def reduce(a, b):
        if b - a < 2:
            return
        p, q = points[a][0], points[b][0]
        direction = q-p
        span = max(float(direction @ direction), 1e-12)
        scores = []
        for j in range(a+1, b):
            t = np.clip(float((points[j][0]-p) @ direction) / span, 0, 1)
            distance = np.linalg.norm(points[j][0]-(p+t*direction))
            # Executor targets an endpoint orientation per segment. Retain wrist
            # changes larger than 12 degrees, even if XYZ travel is straight.
            cosine = np.clip((np.trace(points[j][1].T @ points[b][1])-1)/2, -1, 1)
            angle = np.arccos(cosine)
            scores.append(max(distance / tolerance, angle / np.radians(12)))
        j = a + 1 + int(np.argmax(scores))
        if max(scores) > 1:
            keep.add(j)
            reduce(a, j)
            reduce(j, b)

    cuts = sorted(protected)
    for a, b in zip(cuts, cuts[1:]):
        reduce(a, b)
    return dataclasses.replace(chain, waypoints=[copy.deepcopy(chain.waypoints[i]) for i in sorted(keep)])
