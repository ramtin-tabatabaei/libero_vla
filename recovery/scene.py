"""Contact forces and object shuffling -- the two things the reward needs from
the simulator that `World` does not already provide.

Contact bookkeeping is cached because it is per-step code on the hot path:
which geom belongs to which body never changes once the model is built, and
resolving 500 geom ids to names every control step was never going to be
affordable.
"""
import numpy as np
import mujoco


# ================================================================== contacts


class ContactMonitor:
    """Undesired contact forces between the robot and everything it should miss.

    The exemption is narrow on purpose: only the two finger PADS touching the
    target are free. That is what a grasp is, and charging for it would train
    the policy never to touch anything. Everything else is charged, including
    the target -- because the back of a finger against the can is not a grasp,
    it is a shove, and the straight-line baseline that pushed a can 42 cm
    across the table did it entirely with contacts an object-level exemption
    would have waved through.

    Cached by geom id because this runs every control step; which geom belongs
    to which body never changes once the model is built.
    """

    def __init__(self, world, pad_geoms=(), target_geoms=()):
        self.m, self.d = world.m, world.d
        self.robot = self._geoms_with_prefix(("robot0", "gripper0"))
        self.pads = set(pad_geoms)
        self.target = set(target_geoms)
        self._buffer = np.zeros(6, dtype=float)

    def install(self, sim, force_cap):
        """Sample every physics step; keep the worst capped-force snapshot.

        Replacing the active monitor avoids stacking callbacks on scene resets
        or when a live policy attaches to an existing environment.
        """
        self.force_cap = force_cap
        self.clear()
        sim._recovery_contact_monitor = self
        if not hasattr(sim, '_recovery_original_step'):
            sim._recovery_original_step = sim.step
            def step():
                result = sim._recovery_original_step()
                sim._recovery_contact_monitor.capture()
                return result
            sim.step = step

    def clear(self):
        self.worst_forces = np.empty(0)
        self.worst_cost = -1.0
        self.peak_force = 0.0
        self.peak_bad_force_sum = 0.0

    def capture(self):
        forces, _, peak = self.measure()
        cost = float(np.minimum(forces, self.force_cap).sum())
        if cost > self.worst_cost:
            self.worst_forces, self.worst_cost = forces, cost
        self.peak_force = max(self.peak_force, peak)
        self.peak_bad_force_sum = max(self.peak_bad_force_sum, float(forces.sum()))

    def interval(self):
        self.capture()
        return self.worst_forces, self.peak_force

    def _geoms_with_prefix(self, prefixes):
        out = set()
        for g in range(self.m.ngeom):
            body = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY,
                                     self.m.geom_bodyid[g]) or ""
            if body.startswith(prefixes):
                out.add(g)
        return out

    def allow(self, pad_geoms, target_geoms):
        self.pads = set(pad_geoms)
        self.target = set(target_geoms)

    def measure(self):
        """Undesired force magnitudes, valid touch, and peak robot contact force.

        Use the norm of the three force components (normal plus friction),
        excluding torque. The severe limit also applies to permitted contacts.
        Called at each physics substep by the installed monitor.
        """
        out, valid, peak = [], False, 0.0
        for i in range(self.d.ncon):
            contact = self.d.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            first, second = g1 in self.robot, g2 in self.robot
            if first == second:
                continue
            mine, other = (g1, g2) if first else (g2, g1)
            mujoco.mj_contactForce(self.m, self.d, i, self._buffer)
            force = float(np.linalg.norm(self._buffer[:3]))
            peak = max(peak, force)
            if mine in self.pads and other in self.target:
                valid = valid or force > 1e-6
            else:
                out.append(force)
        return np.asarray(out, dtype=float), valid, peak

    def forces(self):
        return self.measure()[0]


def enable_robot_collisions(world):
    """Keep physical collision meshes enabled; leave visual meshes alone."""
    m = world.m
    for g in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if body.startswith(("robot0", "gripper0")) and m.geom_group[g] == 0:
            m.geom_contype[g] |= 1
            m.geom_conaffinity[g] |= 1


def joint_geom_ids(world, joint_name):
    """Collision geometry of the moving link and its descendants only."""
    root = world.joint_info(joint_name)["body"]
    bodies = {root}
    for body in range(root + 1, world.m.nbody):
        if int(world.m.body_parentid[body]) in bodies:
            bodies.add(body)
    return {g for g in range(world.m.ngeom)
            if world.m.geom_bodyid[g] in bodies and world.m.geom_group[g] == 0}


def geom_ids(world, names):
    """Geom ids for a list of geom names, skipping any the model does not have."""
    out = set()
    for name in names:
        gid = mujoco.mj_name2id(world.m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            out.add(int(gid))
    return out


def body_geom_ids(world, body_prefix):
    """Every geom on a body or its named children -- for fixtures, which have no
    `contact_geoms` list of their own."""
    out = set()
    for g in range(world.m.ngeom):
        body = mujoco.mj_id2name(world.m, mujoco.mjtObj.mjOBJ_BODY,
                                 world.m.geom_bodyid[g]) or ""
        if body == body_prefix or body.startswith(body_prefix + "_"):
            out.add(g)
    return out


class CollisionObstacles:
    """Nearest external geometries that can make undesired robot contacts.

    Candidate IDs are static, but bounds and rankings are measured live. Target
    geometry remains visible when arm/palm contact is penalized, even though
    finger-pad contact with that same geometry is allowed. Robot self-geometry
    and collision-disabled visual meshes are excluded.
    """

    def __init__(self, world, contacts, count):
        self.m, self.d = world.m, world.d
        self.count = int(count)
        robot = np.array(sorted(contacts.robot), dtype=int)
        # Explicit MuJoCo contact pairs bypass collision mask filtering.
        explicit = {}
        for a, b in zip(self.m.pair_geom1, self.m.pair_geom2):
            if int(a) in contacts.robot and int(b) not in contacts.robot:
                explicit.setdefault(int(b), set()).add(int(a))
            if int(b) in contacts.robot and int(a) not in contacts.robot:
                explicit.setdefault(int(a), set()).add(int(b))
        candidates = []
        for geom in range(self.m.ngeom):
            if geom in contacts.robot:
                continue
            compatible = ((self.m.geom_contype[geom] & self.m.geom_conaffinity[robot])
                          | (self.m.geom_conaffinity[geom] & self.m.geom_contype[robot])) != 0
            possible = set(robot[compatible]) | explicit.get(geom, set())
            if geom in contacts.target:
                possible -= contacts.pads
            if possible:
                candidates.append(geom)
        self.geom_ids = np.asarray(candidates, dtype=int)
        self.last_ids = np.empty(0, dtype=int)

    def observe(self, grip, grip_mat):
        """[R_eef.T (p_geom - p_eef), world AABB span, presence] per slot.

        Rank by distance to the bounding box rather than its center, so a
        large nearby table surface is not hidden by a distant table origin.
        Ties use geom ID. Infinite planes use the closest plane point and a
        finite 2 m square surface patch for the span; this keeps inputs finite.
        """
        result = np.zeros((self.count, 7), dtype=float)
        ids = self.geom_ids
        if not ids.size:
            self.last_ids = ids.copy()
            return result
        grip = np.asarray(grip, dtype=float)
        grip_mat = np.asarray(grip_mat, dtype=float).reshape(3, 3)
        rotations = self.d.geom_xmat[ids].reshape(-1, 3, 3)
        boxes = self.m.geom_aabb[ids]
        # Plane AABBs can encode unbounded extents; never use them in matrix
        # arithmetic, which could overflow before we replace the plane rows.
        planes = self.m.geom_type[ids] == mujoco.mjtGeom.mjGEOM_PLANE
        centers = np.array(self.d.geom_xpos[ids], dtype=float)
        half = np.zeros_like(centers)
        finite = ~planes
        centers[finite] += np.einsum('nij,nj->ni', rotations[finite], boxes[finite, :3])
        half[finite] = np.einsum('nij,nj->ni', np.abs(rotations[finite]), boxes[finite, 3:])
        distance = np.linalg.norm(np.maximum(np.abs(centers - grip) - half, 0), axis=1)
        if planes.any():
            normals = rotations[planes, :, 2]
            signed = np.einsum('ni,ni->n', grip - centers[planes], normals)
            centers[planes] = grip - signed[:, None] * normals
            half[planes] = np.abs(rotations[planes, :, :2]).sum(axis=2)
            distance[planes] = np.abs(signed)
        selected = np.lexsort((ids, distance))[:self.count]
        self.last_ids = ids[selected].copy()
        n = len(selected)
        # Row-vector equivalent of R_eef.T @ delta.
        result[:n, :3] = (centers[selected] - grip) @ grip_mat
        result[:n, 3:6] = 2 * half[selected]
        result[:n, 6] = 1.0
        return result


# ================================================================= shuffling


def free_joint_slots(env):
    """{object name: index of its x inside the flattened sim state}.

    The flattened state is [time, qpos, qvel], hence the +1. Same trick
    shuffle_eval.py uses; not imported from there because that module pulls in
    the openpi websocket client at import time, which has no business inside
    eight training workers.
    """
    sim = env.sim
    slots = {}
    for name in env.env.objects_dict:
        joint = name + "_joint0"
        jid = mujoco.mj_name2id(sim.model._model if hasattr(sim.model, "_model") else sim.model,
                                mujoco.mjtObj.mjOBJ_JOINT, joint)
        if jid < 0 or sim.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
            continue
        slots[name] = 1 + int(sim.model.jnt_qposadr[jid])
    return slots


def derangement(n, rng):
    """A permutation with no fixed point, so every object really moves."""
    if n < 2:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def shufflable(world, names, max_span):
    """Objects small enough to swap places without wrecking the scene.

    A basket, a tray or a plate is a surface other things stand on, and
    dropping one onto the spot a 6 cm can vacated leaves it interpenetrating
    whatever was nearby. Restricting the permutation to objects of roughly
    graspable scale keeps the shuffle to things that genuinely sit side by side.
    """
    out = []
    for name in names:
        low, high = world.aabb(name)
        if float(np.max(np.asarray(high) - np.asarray(low))) <= max_span:
            out.append(name)
    return sorted(out)


def shuffle_state(state, slots, names, rng):
    """Permute the xy placements of `names` among each other, in place on a copy.

    Only xy: z and the orientation quaternion stay, so a can that was standing
    is still standing and everything keeps the height of the surface it was
    resting on.
    """
    state = np.asarray(state, dtype=float).copy()
    if len(names) < 2:
        return state, {}
    xy = np.array([state[slots[n]: slots[n] + 2] for n in names])
    perm = derangement(len(names), rng)
    for i, name in enumerate(names):
        state[slots[name]: slots[name] + 2] = xy[perm[i]]
    return state, {name: names[perm[i]] for i, name in enumerate(names)}


def shuffle_scene(world, state, slots, names, rng, attempts=40):
    """Swap compatible resting places, rejecting new object intersections.

    AABBs conservatively filter placements; different support heights are not
    mixed. Preserve each object's bottom height and orientation. If no valid
    permutation exists, return the original scene and an empty mapping.
    """
    fixtures = getattr(getattr(world, "domain", None), "fixtures_dict", {})
    bounds = {n: tuple(np.asarray(x).copy() for x in world.aabb(n))
              for n in set(slots) | set(fixtures)}
    original = np.asarray(state).copy()
    for _ in range(attempts):
        candidate, mapping = shuffle_state(original, slots, names, rng)
        if not mapping:
            return original, {}
        moved_bounds = dict(bounds)
        valid = True
        for name, destination in mapping.items():
            low, high = bounds[name]
            dest_low = bounds[destination][0]
            if abs(low[2] - dest_low[2]) > 0.02:
                valid = False
                break
            candidate[slots[name] + 2] += dest_low[2] - low[2]
            delta = candidate[slots[name]:slots[name]+3] - original[slots[name]:slots[name]+3]
            moved_bounds[name] = (low + delta, high + delta)
        if not valid:
            continue
        for name in mapping:
            lo, hi = moved_bounds[name]
            for other, (olo, ohi) in moved_bounds.items():
                if name != other and np.all(np.minimum(hi, ohi) - np.maximum(lo, olo) > 0.003):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return candidate, mapping
    return original, {}
