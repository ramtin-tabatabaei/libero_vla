"""Evidence-based manipulation phases; never generate replacement trajectories."""
import numpy as np

PRIMITIVES = ('move', 'grasp', 'transport', 'place', 'release', 'push')


def label_samples(samples, events, stable_samples=3):
    """Separate free motion, grasp acquisition, carrying, placement and pushing.

    Placement is the final measured descent before opening a held object.
    Push requires measured object/link displacement without a grasp. Drawer
    joint travel remains evidence when open-hand contact sensors miss the handle;
    a gripper command or proximity alone is not push evidence.
    """
    labels = ['move'] * len(samples)
    for event in events:
        if event['kind'] != 'stable_grasp':
            continue
        start, name = event['index'], event['identity']
        closes = [e['index'] for e in events if e['kind'] == 'close' and e['index'] <= start]
        acquire = closes[-1] if closes else max(0, start-stable_samples+1)
        end = next((e['index'] for e in events if e['index'] > start and
                    (e['kind'] == 'open' or (e['kind'] == 'release' and e['identity'] == name))), len(samples))
        labels[acquire:start] = ['grasp'] * (start-acquire)
        labels[start:end] = ['transport'] * (end-start)
        if end < len(samples):
            # The final maximum preceding a real downward placement stroke.
            heights = np.array([s['position'][2] for s in samples[start:end+1]])
            peak = start + int(np.flatnonzero(heights >= heights.max()-1e-4)[-1])
            if heights.max()-heights[-1] > .005 and peak < end:
                labels[peak:end] = ['place'] * (end-peak)
            released = next((e['index'] for e in events if e['kind'] == 'release' and
                             e['identity'] == name and e['index'] >= end), end)
            stop = min(len(samples), max(end+1, released+1))
            labels[end:stop] = ['release'] * (stop-end)
    # Explicit open after a failed grasp still denotes releasing the gripper.
    for event in events:
        if event['kind'] == 'close' and labels[event['index']] == 'move':
            labels[event['index']] = 'grasp'
        if event['kind'] == 'open' and labels[event['index']] == 'move':
            labels[event['index']] = 'release'
    for i, sample in enumerate(samples):
        if labels[i] != 'move' or sample.get('grasped'):
            continue
        a, b = samples[max(0, i-1)], samples[min(len(samples)-1, i+1)]
        joints = set(a.get('joint_positions', {})) & set(b.get('joint_positions', {}))
        if any(abs(b['joint_positions'][joint]-a['joint_positions'][joint]) > .0005
               for joint in joints):
            labels[i] = 'push'
            continue
        for name, contact in sample.get('contact_state', {}).items():
            if contact.get('both'):  # possible two-finger hold, not a push
                continue
            # A short centred window detects motion even at first contact.
            pa, pb = a.get('object_poses', {}).get(name), b.get('object_poses', {}).get(name)
            moving = pa is not None and pb is not None and np.linalg.norm(np.asarray(pb[0])-pa[0]) > .0005
            if pa is not None and pb is not None:
                relative = np.asarray(pa[1]).T @ np.asarray(pb[1])
                angle = np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))
                moving = moving or angle > .005
            ja, jb = a.get('joint_positions', {}).get(name), b.get('joint_positions', {}).get(name)
            moving = moving or (ja is not None and jb is not None and abs(jb-ja) > .0005)
            if moving:
                labels[i] = 'push'
                break
    return labels


def annotate_events(samples, events, stable_samples=3):
    labels = label_samples(samples, events, stable_samples)
    for event in events:
        event['primitive'] = labels[event['index']]
    # Retain boundaries even where there is no gripper/contact transition (e.g.
    # the measured transport-to-place descent). Existing anchors keep priority.
    occupied = {e['index'] for e in events}
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1] and i not in occupied:
            events.append(dict(index=i, kind='primitive', identity=labels[i], primitive=labels[i]))
    return events


def retarget_metadata(segment, index):
    """Attach phase and starting mean for continuous, endpoint-constrained replay."""
    primitive = segment.get('primitive')
    if primitive is None:
        return {}
    return dict(primitive=primitive, retargeting='task_endpoint',
                local_phase=float(index/max(1, len(segment['phase'])-1)),
                interval_start_position=segment['position'][0],
                interval_start_orientation=segment['orientation'][0])
