"""Event-partitioned object-relative residual GMM/GMR; no simulator imports.

Each semantic interval owns a frame and an SO(3) chart along a nominal curve.
A joint GMM over phase and trajectory residuals models demonstration variation,
without interpreting deterministic travel within a Gaussian as uncertainty.
"""
from dataclasses import asdict, dataclass
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation, Slerp

ANCHOR_FRAME = '__anchor__'


@dataclass
class Config:
    min_episode_steps: int = 3
    samples_per_interval: int = 16
    components: int = 4
    position_scale: float = .05
    orientation_scale: float = .35
    velocity_scale: float = .2
    gripper_weight: float = 4.
    dtw_band: float = .3
    stable_samples: int = 3
    gripper_debounce: int = 4
    lift_threshold: float = .008
    joint_threshold: float = .005
    covariance_floor: float = 1e-7
    tolerance_sigma: float = 2.
    min_position_tolerance: float = .002
    max_position_tolerance: float = .04
    min_orientation_tolerance: float = .02
    max_orientation_tolerance: float = .35
    geometry_tolerance: float = .015
    rotation_tolerance: float = .15
    importance_threshold: float = 2.5
    max_waypoints: int = 100

    def validate(self):
        if min(self.samples_per_interval, self.components, self.stable_samples) < 2 or self.max_waypoints < 2:
            raise ValueError('Sample, component, stability, and waypoint bounds are too small')
        for key, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError('Config must be positive and finite: '+key)
        if self.dtw_band > 1:
            raise ValueError('dtw_band must be <= 1')
        if self.min_position_tolerance > self.max_position_tolerance or self.min_orientation_tolerance > self.max_orientation_tolerance:
            raise ValueError('Minimum tolerance exceeds maximum')


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_demos(path):
    paths = [Path(path)] if Path(path).is_file() else sorted(Path(path).rglob('*.json'))
    result = []
    for p in paths:
        data = json.loads(p.read_text())
        if isinstance(data, dict) and 'samples' in data:
            data['_path'] = str(p.resolve()); result.append(data)
    if not result:
        raise ValueError('No demonstration JSON files found')
    return result


def relative_pose(position, orientation, frame):
    p, r = map(np.asarray, frame)
    return (np.asarray(position)-p) @ r, np.einsum('ij,...jk->...ik', r.T, orientation)


def world_pose(position, orientation, frame):
    p, r = map(np.asarray, frame)
    return np.asarray(position) @ r.T+p, np.einsum('ij,...jk->...ik', r, orientation)


def samples(demo):
    result = list(demo['samples'])
    if not result:
        raise ValueError('Empty demonstration')
    if 'terminal_sample' in demo:
        result.append(demo['terminal_sample'])
    else:
        end = dict(result[-1], position=demo['end_position'], orientation=demo['end_orientation'],
                   task_success=bool(demo['success']))
        end['timestamp'] = result[-1].get('timestamp', (len(result)-1)*demo.get('control_timestep', .05))+demo.get('control_timestep', .05)
        result.append(end)
    return result


def gripper_states(demo, config=None):
    """Commanded gripper per sample, with transient toggles absorbed.

    A policy idling after the task flips the gripper for one or two control steps.
    Those are not grasp or release events: left in, they fragment the event
    signature and stop otherwise compatible demonstrations from being pooled. A
    genuine retry is longer than the debounce and still produces its own events.
    """
    c = config or Config()
    states = [s['gripper'] for s in samples(demo)]
    runs = []
    for state in states:
        if runs and runs[-1][0] == state:
            runs[-1][1] += 1
        else:
            runs.append([state, 1])
    resolved = []
    for i, (state, length) in enumerate(runs):
        resolved.append(resolved[-1] if i and length < c.gripper_debounce else state)
    return [state for state, (_, length) in zip(resolved, runs) for _ in range(length)]


def detect_events(demo, config=None):
    """Ordered, identity-bearing anchors. Repeated interactions retain occurrence IDs."""
    c = config or Config(); ss = samples(demo); events = []; grips = gripper_states(demo, c)
    def add(i, kind, identity=''):
        events.append(dict(index=int(i), kind=kind, identity=identity))
    add(0, 'start')
    prev_grip = demo.get('initial_gripper', grips[0])
    last_contacts, last_held = set(), set()
    counts = Counter(); stable = set(); lifted = set(); joint_active = set(); achieved = set()
    lift_base = {}
    for i, s in enumerate(ss):
        grip = grips[i]
        if grip != prev_grip:
            add(i, 'close' if grip == 'close' else 'open')
        prev_grip = grip
        held = set(s.get('grasped', []))
        contacts = s.get('contact_state')
        # Version 1 has two-pad grasp evidence only: never invent earlier contact.
        contacts = contacts if contacts is not None else {n: dict(kind='object') for n in held}
        now = set(contacts)
        # Contact sensors can drop out for one or two samples during an
        # otherwise continuous hold. Bridge only those short gaps; a longer
        # loss of contact still marks a distinct interaction/retry.
        for name in last_contacts-now:
            if any(name in (ss[j].get('contact_state') or {})
                   for j in range(i+1, min(i+3, len(ss)))):
                now.add(name)
        for name in sorted(now-last_contacts):
            add(i, 'fixture_contact' if contacts[name].get('kind', '').startswith('fixture') else 'contact', name)
        for name in sorted(held | set(counts)):
            counts[name] = counts[name]+1 if name in held else 0
            if counts[name] >= c.stable_samples and name not in stable:
                stable.add(name); add(i, 'stable_grasp', name)
                pose = ss[max(0, i-c.stable_samples+1)].get('object_poses', {}).get(name)
                if pose is not None:
                    lift_base[name] = float(pose[0][2])
            pose = s.get('object_poses', {}).get(name)
            if name in stable and name not in lifted and pose is not None and name in lift_base:
                if pose[0][2]-lift_base[name] >= c.lift_threshold:
                    add(i, 'lift', name); lifted.add(name)
        for name in sorted(last_held-held):
            add(i, 'release', name); stable.discard(name); lifted.discard(name); lift_base.pop(name, None)
        # A fixture interaction can release without a two-pad grasp.
        for name in sorted(last_contacts-now):
            if name not in last_held and name not in demo.get('objects', []):
                add(i, 'fixture_release', name)
        for name, value in s.get('joint_positions', {}).items():
            base = ss[0].get('joint_positions', {}).get(name, value)
            if abs(value-base) >= c.joint_threshold and name not in joint_active:
                add(i, 'joint_motion', name); joint_active.add(name)
        for k, value in enumerate(s.get('goal_predicates', [])):
            if value and k not in achieved:
                add(i, 'goal', str(k)); achieved.add(k)
        last_contacts, last_held = now, held
    if demo['success']:
        i = next((i for i, s in enumerate(ss) if s.get('task_success')), len(ss)-1)
        add(i, 'success')
    add(len(ss)-1, 'end')
    from waypoint_primitives import annotate_events
    annotate_events(ss, events, c.stable_samples)
    # Stable order for simultaneous anchors; do not merge distinct event kinds.
    priority = {n: i for i, n in enumerate(('start','close','contact','fixture_contact','stable_grasp',
                    'lift','joint_motion','open','release','fixture_release','primitive','goal','success','end'))}
    events.sort(key=lambda e: (e['index'], priority[e['kind']], e['identity']))
    occurrences = Counter()
    for e in events:
        stem = e['kind']+':'+e['identity']; occurrences[stem] += 1
        e['key'] = stem+':'+str(occurrences[stem])
    for k, event in enumerate(events):
        command = event['kind'] if event['kind'] in ('close', 'open') else 'keep'
        if event['kind'] == 'close':
            following = next((e['kind'] for e in events[k+1:] if e['kind'] in ('stable_grasp', 'open')), None)
            if following == 'stable_grasp':
                command = 'keep'
        elif event['kind'] == 'stable_grasp':
            command = 'close'
        # A waypoint executor stops at each target. Closing fully at the VLA's
        # first close command would close in free space, before the simultaneous
        # approach has finished. Close at its measured stable grasp pose instead.
        event['execution_gripper_command'] = command
    return events


def event_signature(events):
    return tuple(e['key'] for e in events)


def infer_goals(demo):
    if 'goals' in demo:
        return demo['goals'], 'recorded BDDL'
    if '_migrated_goals' in demo:
        return demo['_migrated_goals'], 'benchmark BDDL migration'
    # Legacy migration: deterministic benchmark BDDL, never proximity guessing.
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        task = benchmark.get_benchmark_dict()[demo['suite']]().get_task(demo['task_id'])
        path = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        goals = robosuite_parse_problem(str(path))['goal_state']
        demo['_migrated_goals'] = goals
        return goals, 'benchmark BDDL migration'
    except (ImportError, OSError, KeyError, ValueError):
        return [], 'unavailable'


def resolve_roles(demo, events, interval):
    """Identify the manipulated object, its goal destination and the interval role."""
    poses = demo.get('initial_object_poses', demo['poses'])
    left, right = events[interval:interval+2]
    goals, source = infer_goals(demo)
    grasp_events = [e for e in events if e['kind'] == 'stable_grasp']
    past = [e for e in grasp_events if e['index'] <= left['index']]
    future = [e for e in grasp_events if e['index'] >= right['index']]
    target = past[-1]['identity'] if past else (future[0]['identity'] if future else None)
    # After a release, the next distinct source becomes the approach frame.
    releases = [e for e in events[:interval+1] if e['kind'] == 'release']
    if releases and future and (not past or releases[-1]['index'] >= past[-1]['index']):
        target = future[0]['identity']
    destinations = [g[2] for g in goals if len(g) >= 3 and g[0].lower() in ('in','on') and g[1] == target]
    articulated = [g[1] for g in goals if len(g) >= 2 and g[0].lower() in ('open','close','turnon','turnoff')]
    interactions = [e['identity'] for e in events if e['kind'] in ('fixture_contact','joint_motion')]
    lift = [e for e in events[:interval+1] if e['kind'] == 'lift' and e['identity'] == target]
    # For legacy logs without measured lift: stable grasp is an explicit fallback.
    transfer = bool(lift) or (bool(past) and not any(e['kind']=='lift' and e['identity']==target for e in events))
    if target:
        role = 'destination' if transfer and destinations else 'target'
        candidates = destinations if role == 'destination' else [target]
    elif articulated or interactions:
        role = 'fixture'
        # A moving joint frame is preferred over its fixed parent fixture.
        joint_refs = [n for n in interactions if n in demo.get('joints', {})]
        candidates = list(dict.fromkeys(joint_refs or articulated or interactions))
    else:
        role = 'target'
        candidates = list(dict.fromkeys(g[1] for g in goals if len(g)>1 and g[1] in poses))
    return dict(poses=poses, left=left, role=role, candidates=candidates, source=source,
                target=target, destinations=list(dict.fromkeys(destinations)))


def select_reference(demo, events, interval, overrides=None):
    """Resolve the primary task frame; ambiguity is an error."""
    overrides = overrides or {}
    resolved = resolve_roles(demo, events, interval)
    poses, role, left, source = (resolved[k] for k in ('poses','role','left','source'))
    explicit = overrides.get(left['key'], overrides.get(role))
    candidates = list(dict.fromkeys(n for n in resolved['candidates'] if n in poses))
    if explicit is not None:
        if explicit not in poses:
            raise ValueError('Unknown explicit reference '+explicit)
        chosen, reason = explicit, 'explicit override'
    elif len(candidates) == 1:
        chosen, reason = candidates[0], source+' and measured interaction'
    else:
        raise ValueError('Ambiguous reference at %s (%s); candidates=%s. Supply --reference-map with role or event key.' % (left['key'], role, candidates or list(poses)))
    # Fixed episode-start object frames preserve lift/transport displacement.
    # Articulated links use the frame captured when that interval begins.
    policy = 'interval_start' if role == 'fixture' else 'episode_start'
    return dict(name=chosen, role=role, policy=policy, candidates=candidates, reason=reason)


def select_references(demo, events, interval, overrides=None):
    """Use one measured object, site or joint frame for each interval.

    The first move/transport target is connected to the reached previous
    waypoint during replay; that connection is not a second stored offset.
    """
    return [dict(select_reference(demo, events, interval, overrides), primary=True)]


def frame_for(demo, event, reference):
    if reference['name'] == ANCHOR_FRAME:
        sample = samples(demo)[event['index']]
        return [sample['position'], sample['orientation']]
    poses = demo.get('initial_object_poses', demo['poses'])
    if reference['policy'] == 'interval_start':
        poses = samples(demo)[event['index']].get('object_poses', poses)
    if reference['name'] not in poses:
        raise ValueError('Missing frame '+reference['name'])
    return poses[reference['name']]


def resample_pose(p, r, grip, times, query):
    p, r, grip, times, query = map(np.asarray, (p,r,grip,times,query))
    if len(p) == 1:
        return np.repeat(p, len(query), axis=0), np.repeat(r, len(query), axis=0), np.repeat(grip, len(query))
    pos = np.column_stack([np.interp(query, times, p[:, j]) for j in range(3)])
    rot = Slerp(times, Rotation.from_matrix(r))(np.clip(query, times[0], times[-1])).as_matrix()
    # Discrete commands are left-continuous at original sample times.
    g = grip[np.clip(np.searchsorted(times, query, side='right')-1, 0, len(grip)-1)]
    return pos, rot, g


def interval_data(demo, events, i, ref, count):
    ss = samples(demo); a, b = events[i]['index'], events[i+1]['index']
    raw = ss[a:b+1]
    p = np.array([s['position'] for s in raw]); r = np.array([s['orientation'] for s in raw])
    p, r = relative_pose(p, r, frame_for(demo, events[i], ref))
    grips = gripper_states(demo)[a:b+1]
    g = np.array([1. if state=='close' else 0. for state in grips])
    time = np.array([s.get('timestamp', (a+j)*demo.get('control_timestep', .05)) for j,s in enumerate(raw)])
    if len(time)>1 and np.any(np.diff(time)<=0):
        raise ValueError('Non-increasing sample timestamps')
    return (*resample_pose(p, r, g, np.arange(len(raw)), np.linspace(0,len(raw)-1,count)),
            float(time[-1]-time[0]))


def dtw_path(p, r, g, q, s, h, config=None):
    """Banded monotone DTW within one semantic interval, with fixed endpoints."""
    c = config or Config(); n,m = len(p),len(q)
    position = np.sum((p[:,None]-q[None,:])**2,axis=2)/c.position_scale**2
    relative = np.einsum('aij,bjk->abik', np.transpose(r,(0,2,1)), s)
    angle = np.arccos(np.clip((np.trace(relative,axis1=2,axis2=3)-1)/2,-1,1))
    cost = position+(angle/c.orientation_scale)**2+c.gripper_weight*(g[:,None]-h[None,:])**2
    dp = np.full((n+1,m+1),np.inf); dp[0,0]=0; parent={}
    for i in range(1,n+1):
        for j in range(1,m+1):
            if abs((i-1)/max(n-1,1)-(j-1)/max(m-1,1)) > c.dtw_band:
                continue
            prev = [(i-1,j-1),(i-1,j),(i,j-1)]
            v = min(prev,key=lambda v:dp[v]); dp[i,j]=cost[i-1,j-1]+dp[v];parent[i,j]=v
    if not np.isfinite(dp[n,m]):
        raise ValueError('DTW band admits no complete alignment')
    path=[]; i,j=n,m
    while i and j:
        path.append((i-1,j-1));i,j=parent[i,j]
    return np.array(path[::-1],dtype=int)


def warp_to(template, value, config):
    path=dtw_path(*template[:3],*value[:3],config)
    indices=np.array([np.mean(path[path[:,0]==i,1]) for i in range(len(template[0]))])
    indices[0],indices[-1]=0,len(value[0])-1
    return resample_pose(*value[:3],np.arange(len(value[0])),indices), indices


def fit_gmm(x, components=4, floor=1e-7, iterations=100):
    """Deterministic full-covariance EM initialized in phase order."""
    x=np.asarray(x);n,d=x.shape;k=min(components,n)
    groups=np.array_split(np.argsort(x[:,0],kind='stable'),k)
    means=np.array([x[g].mean(axis=0) for g in groups])
    covs=np.array([np.atleast_2d(np.cov(x[g],rowvar=False,bias=True)) if len(g)>1 else np.eye(d)*floor for g in groups])
    covs+=np.eye(d)*floor;weights=np.array([len(g)/n for g in groups]);previous=-np.inf
    for iteration in range(iterations):
        logp=[]
        for w,mu,cov in zip(weights,means,covs):
            chol=np.linalg.cholesky(cov);z=np.linalg.solve(chol,(x-mu).T).T
            logp.append(np.log(max(w,1e-15))-.5*(d*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+np.sum(z*z,axis=1)))
        logp=np.array(logp).T; normalizer=logsumexp(logp,axis=1);resp=np.exp(logp-normalizer[:,None])
        nk=resp.sum(axis=0)+1e-12; weights=nk/n;means=resp.T@x/nk[:,None]
        for j in range(k):
            delta=x-means[j];covs[j]=(delta.T*resp[:,j])@delta/nk[j]+np.eye(d)*floor
        likelihood=float(normalizer.mean())
        if abs(likelihood-previous)<1e-6:break
        previous=likelihood
    return dict(weights=weights.tolist(),means=means.tolist(),covariances=covs.tolist(),iterations=iteration+1,log_likelihood=likelihood)


def gmr(model, phase):
    weights,means,covs=(np.asarray(model[k]) for k in ('weights','means','covariances'))
    out,covout=[],[]
    for t in phase:
        variance=covs[:,0,0]; delta=t-means[:,0]
        logh=np.log(weights)-.5*(np.log(2*np.pi*variance)+delta**2/variance)
        h=np.exp(logh-logsumexp(logh))
        mu=means[:,1:]+covs[:,1:,0]* (delta/variance)[:,None]
        cc=covs[:,1:,1:]-covs[:,1:,0,None]*covs[:,None,0,1:]/variance[:,None,None]
        mean=h@mu; centered=mu-mean
        cov=np.einsum('k,kij->ij',h,cc)+np.einsum('k,ki,kj->ij',h,centered,centered)
        out.append(mean);covout.append((cov+cov.T)/2)
    return np.array(out),np.array(covout)


def log_normal(x, means, covs):
    """Per-component Gaussian log density; components along the last axis."""
    out = []
    for mu, cov in zip(means, covs):
        chol = np.linalg.cholesky(cov); z = np.linalg.solve(chol, (x-mu).T).T
        out.append(-.5*(cov.shape[0]*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()+np.sum(z*z, axis=1)))
    return np.array(out).T


def fit_tpgmm(xs, components=4, floor=1e-7, iterations=100):
    """Deterministic joint EM over P task frames sharing one set of components.

    Every frame block is [phase, residual] and every block sees the same phase
    column, so the component density is N(phase) * prod_p N(residual_p | phase);
    the shared phase factor is counted once instead of P times. This is the
    task-parameterized E-step: responsibilities are tied across frames, which is
    what makes the per-frame Gaussians describe the same component of the motion.
    """
    xs = [np.asarray(x) for x in xs]
    n = xs[0].shape[0]; k = min(components, n)
    if any(x.shape[0] != n for x in xs):
        raise ValueError('Task frames must contribute one observation per aligned sample')
    groups = np.array_split(np.argsort(xs[0][:, 0], kind='stable'), k)
    means = [np.array([x[g].mean(axis=0) for g in groups]) for x in xs]
    covs = [np.array([np.atleast_2d(np.cov(x[g], rowvar=False, bias=True)) if len(g) > 1
                      else np.eye(x.shape[1])*floor for g in groups])+np.eye(x.shape[1])*floor for x in xs]
    weights = np.array([len(g)/n for g in groups]); previous = -np.inf
    for iteration in range(iterations):
        logp = np.log(np.maximum(weights, 1e-15))+sum(log_normal(x, m, c) for x, m, c in zip(xs, means, covs))
        # The phase marginal is identical in every frame; keep exactly one copy.
        logp -= (len(xs)-1)*log_normal(xs[0][:, :1], means[0][:, :1], covs[0][:, :1, :1])
        normalizer = logsumexp(logp, axis=1); resp = np.exp(logp-normalizer[:, None])
        nk = resp.sum(axis=0)+1e-12; weights = nk/n
        for f, x in enumerate(xs):
            means[f] = resp.T@x/nk[:, None]
            for j in range(k):
                delta = x-means[f][j]
                covs[f][j] = (delta.T*resp[:, j])@delta/nk[j]+np.eye(x.shape[1])*floor
        likelihood = float(normalizer.mean())
        if abs(likelihood-previous) < 1e-6: break
        previous = likelihood
    return dict(weights=weights.tolist(), iterations=iteration+1, log_likelihood=likelihood,
                frames=[dict(means=m.tolist(), covariances=c.tolist()) for m, c in zip(means, covs)])


def tpgmr(mixture, phase):
    """Condition every frame on phase under one shared set of responsibilities."""
    weights = np.asarray(mixture['weights'])
    frames = [(np.asarray(f['means']), np.asarray(f['covariances'])) for f in mixture['frames']]
    shared = frames[0]; variance = shared[1][:, 0, 0]
    means, covariances = [[] for _ in frames], [[] for _ in frames]
    for t in phase:
        delta = t-shared[0][:, 0]
        logh = np.log(weights)-.5*(np.log(2*np.pi*variance)+delta**2/variance)
        h = np.exp(logh-logsumexp(logh))
        for f, (mu_k, cov_k) in enumerate(frames):
            v = cov_k[:, 0, 0]; d = t-mu_k[:, 0]
            mu = mu_k[:, 1:]+cov_k[:, 1:, 0]*(d/v)[:, None]
            cc = cov_k[:, 1:, 1:]-cov_k[:, 1:, 0, None]*cov_k[:, None, 0, 1:]/v[:, None, None]
            mean = h@mu; centered = mu-mean
            cov = np.einsum('k,kij->ij', h, cc)+np.einsum('k,ki,kj->ij', h, centered, centered)
            means[f].append(mean); covariances[f].append((cov+cov.T)/2)
    return [np.array(m) for m in means], [np.array(c) for c in covariances]


def fuse_gaussians(means, covs, floor=1e-9):
    """Product of Gaussians: the precision-weighted consensus of the task frames.

    A frame that is consistent across demonstrations contributes a tight covariance
    and therefore dominates; an inconsistent one is down-weighted automatically.
    """
    means = [np.asarray(m, dtype=float) for m in means]; covs = [np.asarray(c, dtype=float) for c in covs]
    d = means[0].shape[-1]; regulariser = np.eye(d)*floor
    precision = np.zeros((d, d)); weighted = np.zeros(d)
    for m, c in zip(means, covs):
        p = np.linalg.inv(c+regulariser); precision += p; weighted += p@m
    cov = np.linalg.inv(precision+regulariser)
    return cov@weighted, (cov+cov.T)/2


def shrink(cov, count, floor):
    """Shrink a small-sample covariance toward its isotropic average.

    A handful of demonstrations cannot support a full 3x3 spread: with n demos the
    estimate has rank n-1 at best, so its apparent tight directions are sampling
    noise. Left alone, a frame would veto every other frame along an axis it never
    actually observed. Shrinkage intensity falls as demonstrations accumulate.
    """
    cov = np.asarray(cov, dtype=float); d = cov.shape[-1]
    intensity = d/(d+count)
    isotropic = np.trace(cov, axis1=-2, axis2=-1)[..., None, None]/d*np.eye(d)
    return (1-intensity)*cov+intensity*isotropic+np.eye(d)*floor


def combine(means, covs, weights, floor=1e-9):
    """Product of Gaussians whose precisions come from `weights`, values from `covs`.

    Fusion must be driven by how reproducible each frame is at this instant, which
    is the empirical cross-demonstration spread, not by the GMR conditional
    covariance: detrending every frame against its own nominal curve leaves those
    conditionals at similar magnitudes and would blend the frames evenly. The
    reported covariance is the matching linear combination of `covs`, so tolerances
    keep their smoothed GMR meaning.
    """
    means = [np.asarray(m, dtype=float) for m in means]
    d = means[0].shape[-1]; regulariser = np.eye(d)*floor
    precisions = [np.linalg.inv(np.asarray(w, dtype=float)+regulariser) for w in weights]
    total = np.linalg.inv(sum(precisions)+regulariser)
    gains = [total@p for p in precisions]
    mean = sum(g@m for g, m in zip(gains, means))
    cov = sum(g@np.asarray(c, dtype=float)@g.T for g, c in zip(gains, covs))
    return mean, (cov+cov.T)/2


def fuse_world(frames, positions, orientations, position_covs, orientation_covs,
               position_weights=None, orientation_weights=None):
    """Fuse per-frame relative predictions into one world pose and covariance.

    Position transforms exactly. Orientation is fused in the tangent space of the
    primary frame's prediction; to first order a body-tangent perturbation carries
    across unchanged, which holds while the frames broadly agree on orientation.
    """
    position_weights = position_covs if position_weights is None else position_weights
    orientation_weights = orientation_covs if orientation_weights is None else orientation_weights
    world_p, world_pc, world_pw = [], [], []
    offsets, world_rc, world_rw, reference = [], [], [], None
    for frame, p, r, pc, rc, pw, rw in zip(frames, positions, orientations, position_covs,
                                           orientation_covs, position_weights, orientation_weights):
        origin, rotation = np.asarray(frame[0], dtype=float), np.asarray(frame[1], dtype=float)
        world_p.append(rotation@np.asarray(p)+origin)
        world_pc.append(rotation@np.asarray(pc)@rotation.T); world_pw.append(rotation@np.asarray(pw)@rotation.T)
        world_r = rotation@np.asarray(r)
        if reference is None:
            reference = world_r
        offsets.append(Rotation.from_matrix(reference.T@world_r).as_rotvec())
        world_rc.append(np.asarray(rc)); world_rw.append(np.asarray(rw))
    position, position_cov = combine(world_p, world_pc, world_pw)
    offset, orientation_cov = combine(offsets, world_rc, world_rw)
    return position, reference@Rotation.from_rotvec(offset).as_matrix(), position_cov, orientation_cov


def tolerance(cov, config, orientation=False):
    sigma=np.sqrt(max(float(np.linalg.eigvalsh(cov)[-1]),0.))
    low=config.min_orientation_tolerance if orientation else config.min_position_tolerance
    high=config.max_orientation_tolerance if orientation else config.max_position_tolerance
    return float(np.clip(config.tolerance_sigma*sigma,low,high))


def fit_interval_frames(aligned, config):
    """Fit one task-parameterized mixture spanning every frame of a semantic interval.

    `aligned` is indexed [frame][demonstration] -> (position, orientation, gripper),
    already warped onto the shared template, so sample i is the same instant in every
    frame and demonstration. Each frame contributes a [phase, position, rotation,
    velocity] block detrended against its own nominal curve; the gripper is frame
    independent and enters as one extra shared block rather than once per frame.
    """
    c = config
    g = np.array([a[2] for a in aligned[0]]); n, t = g.shape; phase = np.linspace(0, 1, t)
    scales = np.array([c.position_scale]*3+[c.orientation_scale]*3+[c.velocity_scale]*3)
    blocks, frames = [], []
    for per_demo in aligned:
        p = np.array([a[0] for a in per_demo]); r = np.array([a[1] for a in per_demo])
        nominal_p = p.mean(axis=0)
        nominal_r = np.array([Rotation.from_matrix(r[:, i]).mean().as_matrix() for i in range(t)])
        residual_r = np.array([Rotation.from_matrix(np.einsum('tji,tjk->tik', nominal_r, rr)).as_rotvec() for rr in r])
        if np.max(np.linalg.norm(residual_r, axis=2)) > 2.6:
            raise ValueError('Incompatible orientation modes exceed the local SO(3) chart; separate demonstrations')
        # Velocity is with respect to normalized interval phase, not physical seconds.
        velocity = np.gradient(p, phase, axis=1); nominal_v = velocity.mean(axis=0)
        residual = np.concatenate((p-nominal_p, residual_r, velocity-nominal_v), axis=2)
        blocks.append(np.concatenate((np.broadcast_to(phase, (n, t))[..., None], residual/scales), axis=2).reshape(-1, 10))
        # How reproducible this frame is at each aligned instant, across demonstrations.
        # A handful of demonstrations gives a rank-deficient spread, so each frame is
        # floored at the smallest tolerance the model will commit to; without it a
        # frame would claim unbounded precision along its own unobserved directions.
        empirical_p = np.array([np.cov(p[:, i]-nominal_p[i], rowvar=False, bias=True).reshape(3, 3) for i in range(t)])
        empirical_r = np.array([np.cov(residual_r[:, i], rowvar=False, bias=True).reshape(3, 3) for i in range(t)])
        empirical_p = shrink(empirical_p, n, c.min_position_tolerance**2)
        empirical_r = shrink(empirical_r, n, c.min_orientation_tolerance**2)
        frames.append((nominal_p, nominal_r, nominal_v, p, r, empirical_p, empirical_r))
    nominal_g = g.mean(axis=0)
    blocks.append(np.stack((np.broadcast_to(phase, (n, t)).ravel(), (g-nominal_g).ravel()), axis=1))
    mixture = fit_tpgmm(blocks, c.components, c.covariance_floor)
    means, covs = tpgmr(mixture, phase)
    fitted = []
    for i, (nominal_p, nominal_r, nominal_v, p, r, empirical_p, empirical_r) in enumerate(frames):
        mu = means[i]*scales; cov = covs[i]*scales[None, :, None]*scales[None, None, :]
        fitted.append(dict(position=(nominal_p+mu[:, :3]).tolist(),
                           orientation=np.einsum('tij,tjk->tik', nominal_r,
                                                 Rotation.from_rotvec(mu[:, 3:6]).as_matrix()).tolist(),
                           velocity=(nominal_v+mu[:, 6:9]).tolist(), covariance=cov.tolist(),
                           nominal_position=nominal_p.tolist(), nominal_orientation=nominal_r.tolist(),
                           nominal_velocity=nominal_v.tolist(),
                           aligned_position=p.tolist(), aligned_orientation=r.tolist(),
                           empirical_position_covariance=empirical_p.tolist(),
                           empirical_orientation_covariance=empirical_r.tolist()))
    return dict(frames=fitted, gripper_probability=np.clip(nominal_g+means[-1][:, 0], 0, 1).tolist(),
                nominal_gripper=nominal_g.tolist(), aligned_gripper=g.tolist(),
                scales=scales.tolist(), gmm=mixture)


def fused_trajectory(segment, frames):
    """World consensus trajectory for one interval under the supplied frame poses."""
    blocks = segment['frames']
    position = [np.asarray(b['position']) for b in blocks]
    orientation = [np.asarray(b['orientation']) for b in blocks]
    covariance = [np.asarray(b['covariance']) for b in blocks]
    weight_p = [np.asarray(b['empirical_position_covariance']) for b in blocks]
    weight_r = [np.asarray(b['empirical_orientation_covariance']) for b in blocks]
    out = [fuse_world(frames, [x[i] for x in position], [x[i] for x in orientation],
                      [x[i][:3, :3] for x in covariance], [x[i][3:6, 3:6] for x in covariance],
                      [w[i] for w in weight_p], [w[i] for w in weight_r])
           for i in range(len(position[0]))]
    return (np.array([o[0] for o in out]), np.array([o[1] for o in out]),
            np.array([o[2] for o in out]), np.array([o[3] for o in out]))


def fit_interval(aligned, config):
    c=config; p=np.array([a[0] for a in aligned]); r=np.array([a[1] for a in aligned]);g=np.array([a[2] for a in aligned])
    n,t,_=p.shape;phase=np.linspace(0,1,t)
    nominal_p=p.mean(axis=0)
    nominal_r=np.array([Rotation.from_matrix(r[:,i]).mean().as_matrix() for i in range(t)])
    residual_r=np.array([Rotation.from_matrix(np.einsum('tji,tjk->tik',nominal_r,rr)).as_rotvec() for rr in r])
    if np.max(np.linalg.norm(residual_r,axis=2)) > 2.6:
        raise ValueError('Incompatible orientation modes exceed the local SO(3) chart; separate demonstrations')
    # Velocity is with respect to normalized interval phase, not physical seconds.
    velocity=np.gradient(p,phase,axis=1);nominal_v=velocity.mean(axis=0);nominal_g=g.mean(axis=0)
    residual=np.concatenate((p-nominal_p,residual_r,velocity-nominal_v,(g-nominal_g)[...,None]),axis=2)
    scales=np.array([c.position_scale]*3+[c.orientation_scale]*3+[c.velocity_scale]*3+[1.])
    x=np.concatenate((np.broadcast_to(phase,(n,t))[...,None],residual/scales),axis=2).reshape(-1,11)
    mixture=fit_gmm(x,c.components,c.covariance_floor)
    mu,cov=gmr(mixture,phase);mu*=scales;cov*=scales[None,:,None]*scales[None,None,:]
    mean_p=nominal_p+mu[:,:3];mean_r=np.einsum('tij,tjk->tik',nominal_r,Rotation.from_rotvec(mu[:,3:6]).as_matrix())
    return dict(position=mean_p.tolist(),orientation=mean_r.tolist(),velocity=(nominal_v+mu[:,6:9]).tolist(),
                gripper_probability=np.clip(nominal_g+mu[:,9],0,1).tolist(),covariance=cov.tolist(),
                nominal_position=nominal_p.tolist(),nominal_orientation=nominal_r.tolist(),
                nominal_velocity=nominal_v.tolist(),nominal_gripper=nominal_g.tolist(),scales=scales.tolist(),gmm=mixture,
                aligned_position=p.tolist(),aligned_orientation=r.tolist(),aligned_gripper=g.tolist())


def select_waypoints(segments, config):
    c=config;nodes=[]; diagnostics=[]
    for si,seg in enumerate(segments):
        # All newly learned intervals have one task frame. Historical models can
        # still carry their previously fused world statistics.
        if 'world_position' in seg:
            p=np.array(seg['world_position']);r=np.array(seg['world_orientation'])
            pcov=np.array(seg['world_position_covariance']);rcov=np.array(seg['world_orientation_covariance'])
        else:
            p=np.array(seg['position']);r=np.array(seg['orientation']);cov=np.array(seg['covariance'])
            pcov=cov[:,:3,:3];rcov=cov[:,3:6,3:6]
        n=len(p)
        dp=np.diff(p,axis=0);directions=dp/np.maximum(np.linalg.norm(dp,axis=1)[:,None],1e-9)
        turns=np.zeros(n);turns[1:-1]=np.linalg.norm(np.diff(directions,axis=0),axis=1)
        rot=np.zeros(n);rot[1:]=Rotation.from_matrix(np.einsum('tji,tjk->tik',r[:-1],r[1:])).magnitude()/c.rotation_tolerance
        ps=np.sqrt(np.maximum(np.trace(pcov,axis1=1,axis2=2),0))
        rs=np.sqrt(np.maximum(np.trace(rcov,axis1=1,axis2=2),0))
        consistency=.5*(np.exp(-ps/c.position_scale)+np.exp(-rs/c.orientation_scale))
        score=consistency+turns+rot
        chosen={0,n-1};reasons={0:['semantic_anchor','frame_boundary'],n-1:['semantic_anchor','frame_boundary']}
        if seg.get('primitive') not in ('move', 'transport'):
            # Local extrema retain constraint/mode transitions without selecting every low-variance point.
            for i in range(1,n-1):
                if score[i]>=c.importance_threshold and score[i]>=score[i-1] and score[i]>score[i+1]:
                    chosen.add(i);reasons[i]=['consistency_geometry_mode']
            while True:
                worst=(1.,None)
                cuts=sorted(chosen)
                for a,b in zip(cuts,cuts[1:]):
                    if b-a<2:continue
                    f=np.arange(1,b-a)/(b-a)
                    expected=p[a]+f[:,None]*(p[b]-p[a])
                    expected_r=Slerp([0,1],Rotation.from_matrix(r[[a,b]]))(f)
                    angles=(expected_r.inv()*Rotation.from_matrix(r[a+1:b])).magnitude()
                    # Tight learned tolerances make precise regions retain more detail.
                    bounds=np.array([min(c.geometry_tolerance,tolerance(v,c)) for v in pcov[a+1:b]])
                    errors=np.maximum(np.linalg.norm(p[a+1:b]-expected,axis=1)/bounds,angles/c.rotation_tolerance)
                    j=int(np.argmax(errors));candidate=(float(errors[j]),a+1+j)
                    if candidate[0]>worst[0]:worst=candidate
                if worst[1] is None:break
                chosen.add(worst[1]);reasons[worst[1]]=['trajectory_error']
            # Evidence-based pre-contact/pre-release pose, just before the event.
            ending=seg['end_event']['kind']
            if ending in ('contact','stable_grasp','open','fixture_contact') and n>2 and np.linalg.norm(p[-1]-p[0])>.003:
                chosen.add(n-2);reasons.setdefault(n-2,[]).append('pre-release' if ending=='open' else 'pre-grasp' if ending in ('contact','stable_grasp') else 'pre-interaction')
        diagnostics.append(dict(importance=score.tolist(),position_uncertainty=ps.tolist(),orientation_uncertainty=rs.tolist(),turn_score=turns.tolist(),rotation_score=rot.tolist(),consistency_score=consistency.tolist()))
        for i in sorted(chosen):
            event=seg['start_event'] if i==0 else seg['end_event'] if i==n-1 else None
            kind=event['kind'] if event else ('pre-release' if 'pre-release' in reasons[i] else 'pre-grasp' if 'pre-grasp' in reasons[i] else 'motion')
            if kind == 'motion':
                delta=p[min(i+1,n-1)]-p[max(i-1,0)]
                if seg['reference']['role']=='fixture':kind='Fixture interaction'
                elif delta[2]<-.002 and abs(delta[2])>np.linalg.norm(delta[:2]):kind='Descend'
                elif delta[2]>.002 and abs(delta[2])>np.linalg.norm(delta[:2]):kind='Lift' if seg['start_event']['kind'] in ('stable_grasp','lift') else 'Raise'
                elif np.linalg.norm(delta)>.002:kind='Transfer' if seg['reference']['role']=='destination' else 'Approach'
            command=event.get('execution_gripper_command', 'close' if kind=='close' else 'open' if kind=='open' else 'keep') if event else 'keep'
            # The new model has one relative offset and covariance per waypoint.
            frames=[dict(name=ref['name'],role=ref['role'],policy=ref['policy'],
                         relative_position_mean=np.array(block['position'])[i].tolist(),
                         relative_orientation_mean=np.array(block['orientation'])[i].tolist(),
                         position_covariance=np.array(block['covariance'])[i,:3,:3].tolist(),
                         orientation_covariance=np.array(block['covariance'])[i,3:6,3:6].tolist(),
                         empirical_position_covariance=np.array(block['empirical_position_covariance'])[i].tolist(),
                         empirical_orientation_covariance=np.array(block['empirical_orientation_covariance'])[i].tolist())
                    for ref,block in zip(seg.get('references',[]),seg.get('frames',[]))]
            primary=frames[0] if frames else None
            from waypoint_primitives import retarget_metadata
            nodes.append(dict(segment=si,index=i,phase=float(seg['phase'][i]),reference=seg['reference'],
                              **retarget_metadata(seg, i),
                              frames=frames,
                              relative_position_mean=(primary['relative_position_mean'] if primary else p[i].tolist()),
                              relative_orientation_mean=(primary['relative_orientation_mean'] if primary else r[i].tolist()),
                              position_covariance=pcov[i].tolist(),orientation_covariance=rcov[i].tolist(),
                              position_tolerance=tolerance(pcov[i],c),orientation_tolerance=tolerance(rcov[i],c,True),
                              gripper_state='close' if seg['gripper_probability'][i]>=.5 else 'open',gripper_command=command,
                              semantic_event=event,semantic_phase=kind,importance=float(score[i]),reasons=reasons[i]))
    if len(nodes)>c.max_waypoints:
        raise ValueError('Adaptive extraction requires %d waypoints; safety bound %d. Increase --max-waypoints; anchors will not be discarded.'%(len(nodes),c.max_waypoints))
    return nodes,diagnostics


def learn(demonstrations, config=None, overrides=None):
    c=config or Config();c.validate()
    successful=[(i,d) for i,d in enumerate(demonstrations) if d.get('success') and len(d['samples'])>=c.min_episode_steps and not d['samples'][0].get('task_success',False)]
    if len(successful)<2:raise ValueError('At least two successful, nontrivial demonstrations required')
    for _,d in successful:
        ss=samples(d);p=np.asarray([s['position'] for s in ss]);r=np.asarray([s['orientation'] for s in ss])
        if p.shape!=(len(ss),3) or r.shape!=(len(ss),3,3) or not np.isfinite(p).all() or not np.isfinite(r).all():
            raise ValueError('Demonstration poses must be finite XYZ and 3x3 rotations')
        if not np.allclose(np.einsum('tji,tjk->tik',r,r),np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(r),1,atol=1e-5):
            raise ValueError('Demonstration orientations must be proper rotations')
        if any(s['gripper'] not in ('open','close') for s in ss):
            raise ValueError('Unknown gripper state')
    tasks={(d['suite'],d['task_id'],d.get('language',d.get('instruction'))) for _,d in successful}
    if len(tasks)!=1:raise ValueError('Demonstrations must belong to the same task and instruction')
    groups=defaultdict(list)
    for i,d in successful:
        ev=detect_events(d,c);groups[event_signature(ev)].append((i,d,ev))
    selected=max(groups.values(),key=len)
    if len(selected)<2:
        raise ValueError('No two demonstrations share an ordered event signature; inspect contacts/retries or collect more compatible runs')
    excluded=[dict(index=i,reason='unsuccessful' if not d.get('success') else 'trivial/already-solved episode' if len(d['samples'])<c.min_episode_steps or d['samples'][0].get('task_success',False) else 'incompatible event sequence') for i,d in enumerate(demonstrations) if i not in {v[0] for v in selected}]
    canonical=selected[0][2];count=len(canonical)-1;segments=[]
    durations=np.array([[e['index']/max(len(samples(d))-1,1) for e in ev] for _,d,ev in selected])
    # Average episode fractions allocate common phase, even for simultaneous events.
    widths=np.maximum(np.diff(durations.mean(axis=0)),1e-4);boundaries=np.r_[0,np.cumsum(widths)/sum(widths)]
    for j in range(count):
        per_demo=[select_references(d,ev,j,overrides) for _,d,ev in selected]
        if len({tuple((r['name'],r['policy'],r['role']) for r in refs) for refs in per_demo})!=1:
            raise ValueError('Demonstrations disagree about reference identity in interval '+str(j))
        references=per_demo[0];primary=references[0]
        raw=[interval_data(d,ev,j,primary,c.samples_per_interval) for _,d,ev in selected]
        # Deterministic medoid under unwarped position/rotation distances.
        costs=[]
        for a in raw:
            costs.append(sum(float(np.mean(np.linalg.norm(a[0]-b[0],axis=1)/c.position_scale)+np.mean(Rotation.from_matrix(np.einsum('tji,tjk->tik',a[1],b[1])).magnitude()/c.orientation_scale)) for b in raw))
        medoid=int(np.argmin(costs));template=raw[medoid];warped=[warp_to(template,v,c) for v in raw]
        # One warp, computed in the primary frame, is reused by every other frame so
        # that sample i denotes the same instant in all of them.
        indices=[v[1] for v in warped];steps=np.arange(c.samples_per_interval)
        aligned=[[warped[k][0] for k in range(len(selected))]]
        for ref in references[1:]:
            data=[interval_data(d,ev,j,ref,c.samples_per_interval) for _,d,ev in selected]
            aligned.append([resample_pose(*v[:3],steps,idx) for v,idx in zip(data,indices)])
        seg=fit_interval_frames(aligned,c)
        # Back-compatible view of the primary frame, for plotting and legacy readers.
        seg.update({k:seg['frames'][0][k] for k in seg['frames'][0]})
        medoid_demo=selected[medoid]
        frame_poses=[frame_for(medoid_demo[1],medoid_demo[2][j],ref) for ref in references]
        world=fused_trajectory(seg,frame_poses)
        seg.update(reference=primary,references=references,start_event=canonical[j],end_event=canonical[j+1],
                   frame_poses=[[np.asarray(f[0]).tolist(),np.asarray(f[1]).tolist()] for f in frame_poses],
                   world_position=world[0].tolist(),world_orientation=world[1].tolist(),
                   world_position_covariance=world[2].tolist(),world_orientation_covariance=world[3].tolist(),
                   phase=np.linspace(boundaries[j],boundaries[j+1],c.samples_per_interval).tolist(),
                   duration_seconds=float(np.mean([v[3] for v in raw])),
                   template=dict(position=template[0].tolist(),orientation=template[1].tolist(),gripper=template[2].tolist()),
                   warps=[v[1].tolist() for v in warped],medoid_demo=medoid_demo[0])
        seg['primitive'] = canonical[j]['primitive']
        segments.append(seg)
    waypoints,diagnostics=select_waypoints(segments,c)
    for seg,diagnostic in zip(segments,diagnostics):seg.update(diagnostic)
    d=selected[0][1]
    return dict(version=2,model_type='task_parameterized_gmm_gmr',suite=d['suite'],task_id=d['task_id'],
                instruction=d.get('instruction',d.get('language','')),config=asdict(c),reference_overrides=overrides or {},
                reference_scheme='single_reference_minimal',
                used_demonstrations=[i for i,_,_ in selected],excluded_demonstrations=excluded,
                event_modes=[dict(events=list(signature),demonstrations=[i for i,_,_ in members]) for signature,members in groups.items()],
                demonstration_files=[d.get('_path') for _,d,_ in selected],
                demonstration_success=[bool(d.get('success')) for d in demonstrations],
                events=canonical,event_episode_fraction_mean=durations.mean(axis=0).tolist(),
                event_episode_fraction_std=durations.std(axis=0).tolist(),segments=segments,waypoints=waypoints,
                replay_validation='not_run',warnings=['Version 1 contact/lift/fixture evidence may be incomplete'] if any(d.get('version',1)<2 for _,d,_ in selected) else [])


def save_model(model, folder):
    write_json(Path(folder)/'model.json',model)


def load_model(path):
    path=Path(path);model=json.loads((path/'model.json' if path.is_dir() else path).read_text())
    supported={(1,'event_frame_residual_gmm_gmr'),(2,'task_parameterized_gmm_gmr'),(3,'object_relative_vla_waypoints')}
    if (model.get('version'),model.get('model_type')) not in supported:
        raise ValueError('Unsupported learned trajectory model')
    Config(**model['config']).validate()
    return model


def predict_segment(segment, phase, frame=0):
    """Evaluate one task frame's fitted distribution at local phase in [0, 1]."""
    phase=np.atleast_1d(np.asarray(phase,dtype=float))
    if not np.isfinite(phase).all() or np.any((phase<0)|(phase>1)):
        raise ValueError('Local phase must lie in [0,1]')
    # Each task frame detrends against its own nominal curve, so the frame being
    # evaluated selects which curve the residual is added back onto.
    block=segment['frames'][frame] if 'frames' in segment else segment
    knots=np.linspace(0,1,len(block['nominal_position']))
    p,r,_=resample_pose(np.array(block['nominal_position']),np.array(block['nominal_orientation']),
                       np.array(segment['nominal_gripper']),knots,phase)
    velocity=np.column_stack([np.interp(phase,knots,np.array(block['nominal_velocity'])[:,j]) for j in range(3)])
    grip=np.interp(phase,knots,segment['nominal_gripper'])
    scales=np.asarray(segment['scales'])
    if 'frames' in segment['gmm']:
        means,covs=tpgmr(segment['gmm'],phase)
        mu=means[frame]*scales;cov=covs[frame]*scales[None,:,None]*scales[None,None,:]
        gripper=np.clip(grip+means[-1][:,0],0,1)
    else:
        mu,cov=gmr(segment['gmm'],phase)
        mu*=scales;cov*=scales[None,:,None]*scales[None,None,:]
        gripper=np.clip(grip+mu[:,9],0,1)
    return dict(position=p+mu[:,:3],orientation=np.einsum('tij,tjk->tik',r,Rotation.from_rotvec(mu[:,3:6]).as_matrix()),
                velocity=velocity+mu[:,6:9],gripper_probability=gripper,covariance=cov[:,:9,:9])


def predict_world(segment, phase, frames):
    """Fused world prediction at arbitrary local phase, given live frame poses.

    This is the task-parameterized reproduction step: every frame is evaluated in
    its own coordinates, mapped through the pose that frame actually has now, and
    combined by the product of Gaussians.
    """
    if 'frames' not in segment.get('gmm', {}):
        raise ValueError('Dense world prediction requires a task-parameterized segment')
    if len(frames) != len(segment['frames']):
        raise ValueError('Expected %d frame poses, received %d' % (len(segment['frames']), len(frames)))
    blocks = [predict_segment(segment, phase, frame=i) for i in range(len(segment['frames']))]
    knots = np.linspace(0, 1, len(segment['frames'][0]['position']))
    query = np.atleast_1d(np.asarray(phase, dtype=float))
    weights = []
    for key in ('empirical_position_covariance', 'empirical_orientation_covariance'):
        per_frame = []
        for block in segment['frames']:
            w = np.asarray(block[key])
            per_frame.append(np.array([[[np.interp(u, knots, w[:, a, b]) for b in range(3)]
                                        for a in range(3)] for u in query]))
        weights.append(per_frame)
    fused = [fuse_world(frames, [b['position'][i] for b in blocks], [b['orientation'][i] for b in blocks],
                        [b['covariance'][i][:3, :3] for b in blocks], [b['covariance'][i][3:6, 3:6] for b in blocks],
                        [w[i] for w in weights[0]], [w[i] for w in weights[1]])
             for i in range(len(query))]
    return dict(position=np.array([f[0] for f in fused]), orientation=np.array([f[1] for f in fused]),
                position_covariance=np.array([f[2] for f in fused]),
                orientation_covariance=np.array([f[3] for f in fused]),
                gripper_probability=blocks[0]['gripper_probability'])
