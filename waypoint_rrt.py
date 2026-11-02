"""Drive a LIBERO task from hand-defined Cartesian waypoints using RRT-Connect.

No VLA here -- this is the classic waypoint + planner path, so you can watch the
task in MuJoCo and control exactly where the arm goes.

Pipeline per waypoint:
    Cartesian target -> damped-least-squares IK -> joint config
    current config -> RRT-Connect (collision-checked) -> joint path
    joint path -> JOINT_POSITION controller -> execution

Collision checking counts contacts between robot/gripper geoms and everything
else. At rest that count is 0, so any non-zero value is a real collision.

    python waypoint_rrt.py --list-objects       # object names + poses
    python waypoint_rrt.py --render             # auto pick-and-place chain
    python waypoint_rrt.py --waypoints example_waypoints.json --render
"""

import dataclasses
import json
import pathlib
from typing import List, Optional, Sequence

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv

# JOINT_POSITION controller maps action [-1,1] -> +-0.05 rad delta per step.
CTRL_DELTA_MAX = 0.05

GRIPPER_OPEN, GRIPPER_CLOSE = -1.0, 1.0


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_object"
    task_id: int = 0
    render: bool = False
    """Open the MuJoCo window. run.sh-style: needs MUJOCO_GL=glx."""

    waypoints: Optional[str] = None
    """JSON file of waypoints. Omitted = auto pick-and-place on obj_of_interest."""
    list_objects: bool = False
    """Print object names/poses and exit -- use this to author waypoints."""

    force_rrt: bool = False
    """Skip the straight-line shortcut so the RRT search always runs. Without
    this, free segments return a 2-config path and RRT is never exercised."""
    rrt_max_iters: int = 4000
    rrt_step: float = 0.15
    """Joint-space extension step (rad)."""
    collision_check_step: float = 0.05
    """Resolution for checking an edge (rad). Smaller = safer, slower. An edge
    is sampled at this spacing, so an obstacle thinner than the end effector
    travels between two samples can be missed."""
    collision_check_cartesian: float = 0.0075
    """Largest end effector travel allowed between two checks on an edge (m).
    The joint spacing alone does not bound this: near a stretched-out arm the
    same 0.05 rad moves the wrist several centimetres, far enough to step over
    a thin object between samples. Measured against the chord between the
    edge's endpoints, so a curved edge realises roughly 1.4x this; 7.5 mm here
    measured 10.6 mm worst case, against a thinnest scene object of 18 mm."""
    shortcut_rounds: int = 150
    """Greedy smoothing passes over a found path. Each tries to replace a
    detour with a straight edge, and keeps it only if that edge is clear."""
    ik_iters: int = 300
    ik_tol: float = 1e-3
    ik_damping: float = 0.08
    """Damped least squares inverse kinematics: iteration cap, convergence
    tolerance (m), and damping."""
    seed: int = 0
    settle_steps: int = 10

    ignore_collision_with: Optional[List[str]] = None
    """Bodies excluded from collision checking. Defaults to obj_of_interest --
    you cannot plan into a basket that counts as an obstacle."""


# ----------------------------------------------------------------- sim helpers


def _raw(sim):
    """robosuite wraps MjModel/MjData; mujoco.* functions need the raw structs."""
    return getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)


def _mat_to_quat(mat):
    """MuJoCo free joints store w-first quaternions."""
    x, y, z, w = Rotation.from_matrix(np.asarray(mat)).as_quat()
    return np.array([w, x, y, z])


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


class Scene:
    """Thin helper around the LIBERO env exposing IK + collision checks.

    Which joints the arm has, where its end effector is, and which geometry
    belongs to it are all read from the loaded model. Nothing here assumes a
    joint count, a naming convention or a particular robot.
    """

    ik_iters, ik_tol, ik_damping = 300, 1e-3, 0.08
    """Damped least squares defaults, overridden per instance by an Args."""

    def __init__(self, env, ignore_bodies: Sequence[str] = (),
                 carried_bodies: Sequence[str] = (), args=None, robot: int = 0):
        self.env = env
        self.sim = env.sim
        m, _ = _raw(self.sim)
        for name in ("ik_iters", "ik_tol", "ik_damping"):
            if hasattr(args, name):
                setattr(self, name, getattr(args, name))
        arm = env.env.robots[robot]
        self.site_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, arm.gripper.important_sites["grip_site"])
        self.joints = list(arm.robot_joints)
        jids = [self.sim.model.joint_name2id(j) for j in self.joints]
        self.qadr = np.array([self.sim.model.jnt_qposadr[j] for j in jids])
        self.limits = np.array([self.sim.model.jnt_range[j] for j in jids])
        body_names = [
            self.sim.model.body_id2name(self.sim.model.geom_bodyid[i]) or ""
            for i in range(self.sim.model.ngeom)
        ]
        # A grasped object travels with the wrist. Counting it as part of the
        # arm both stops its own grip contacts reading as collisions and makes
        # the planner steer *it* clear of the scene, not just the fingers.
        own = set(arm.robot_model.bodies) | set(arm.gripper.bodies)
        self.robot_geoms = {
            i
            for i in range(self.sim.model.ngeom)
            if body_names[i] in own
            or any(body_names[i].startswith(b) for b in carried_bodies)
        }
        # You cannot plan a grasp or a place if the thing you are reaching into
        # counts as an obstacle, so those bodies are excluded from the check.
        self.ignored_geoms = {
            i for i in range(self.sim.model.ngeom)
            if any(body_names[i].startswith(b) for b in ignore_bodies)
        }
        self.carried = self._grip_relative_freejoints(carried_bodies)

    # --- state -------------------------------------------------------------
    def arm_qpos(self):
        return self.sim.data.qpos[self.qadr].copy()

    def _eef_frame(self):
        return (self.sim.data.site_xpos[self.site_id].copy(),
                self.sim.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def eef_frame(self, q=None):
        """Where the wrist sits: live, or where a configuration would put it.

        A planned joint route says nothing about the pose it ends at, so a
        caller holding only the route has to ask the model. The arm is put
        back exactly as it was found, the same way the collision check does.
        """
        if q is None:
            return self._eef_frame()
        saved = self.arm_qpos()
        held = [self.sim.data.qpos[adr:adr + 7].copy() for adr, _, _ in self.carried]
        self._set_arm(np.asarray(q, dtype=float))
        frame = self._eef_frame()
        self.sim.data.qpos[self.qadr] = saved
        for (adr, _, _), value in zip(self.carried, held):
            self.sim.data.qpos[adr:adr + 7] = value
        self.sim.forward()
        return frame

    def _grip_relative_freejoints(self, bodies):
        """Freeze each carried body's current pose in the wrist frame.

        Setting arm joints does not drag a free-joint object along, so a
        sampled configuration would leave the held object behind and report a
        clear path straight through whatever the object would have hit.
        """
        if not bodies:
            return []
        m = self.sim.model
        self.sim.forward()
        pos, mat = self._eef_frame()
        held = []
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = m.body_id2name(m.jnt_bodyid[j]) or ""
            if not any(name.startswith(b) for b in bodies):
                continue
            adr = int(m.jnt_qposadr[j])
            q = self.sim.data.qpos
            held.append((adr, mat.T @ (q[adr:adr + 3] - pos),
                         _quat_mul(_quat_conjugate(_mat_to_quat(mat)), q[adr + 3:adr + 7].copy())))
        return held

    def _place_carried(self):
        pos, mat = self._eef_frame()
        wrist = _mat_to_quat(mat)
        q = self.sim.data.qpos
        for adr, offset, rotation in self.carried:
            q[adr:adr + 3] = pos + mat @ offset
            q[adr + 3:adr + 7] = _quat_mul(wrist, rotation)

    def _set_arm(self, q):
        self.sim.data.qpos[self.qadr] = q
        self.sim.forward()
        if self.carried:
            self._place_carried()
            self.sim.forward()

    def eef_pos(self):
        return self.sim.data.site_xpos[self.site_id].copy()

    def eef_span(self, q1, q2):
        """How far the end effector moves between two configurations."""
        saved = self.arm_qpos()
        self._set_arm(q1)
        a = self.eef_pos()
        self._set_arm(q2)
        span = float(np.linalg.norm(self.eef_pos()-a))
        self._set_arm(saved)
        return span

    # --- collision ---------------------------------------------------------
    def in_collision(self, q) -> bool:
        saved = self.arm_qpos()
        held = [self.sim.data.qpos[adr:adr + 7].copy() for adr, _, _ in self.carried]
        self._set_arm(q)
        d = self.sim.data
        hit = False
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in self.robot_geoms) == (g2 in self.robot_geoms):
                continue
            if g1 in self.ignored_geoms or g2 in self.ignored_geoms:
                continue
            hit = True
            break
        self.sim.data.qpos[self.qadr] = saved
        for (adr, _, _), value in zip(self.carried, held):
            self.sim.data.qpos[adr:adr + 7] = value
        self.sim.forward()
        return hit

    def within_limits(self, q):
        return bool(np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1]))

    def valid(self, q):
        return self.within_limits(q) and not self.in_collision(q)

    # --- IK ----------------------------------------------------------------
    def ik(self, target_pos, q_init=None, iters=None, tol=None, damping=None, target_ori=None):
        """Damped least squares IK, optionally constraining wrist orientation."""
        iters = self.ik_iters if iters is None else iters
        tol = self.ik_tol if tol is None else tol
        damping = self.ik_damping if damping is None else damping
        m, d = _raw(self.sim)
        saved = self.arm_qpos()
        q = self.arm_qpos() if q_init is None else np.array(q_init, dtype=float)

        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        for _ in range(iters):
            self._set_arm(q)
            err = np.asarray(target_pos) - self.eef_pos()
            if target_ori is not None:
                ori = self.sim.data.site_xmat[self.site_id].reshape(3, 3)
                rot_err = 0.5 * sum(np.cross(ori[:, i], target_ori[:, i]) for i in range(3))
                err = np.concatenate((err, rot_err))
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.site_id)
            J = jacp[:, self.qadr] if target_ori is None else np.vstack((jacp[:, self.qadr], jacr[:, self.qadr]))
            # DLS: dq = J^T (J J^T + l^2 I)^-1 e
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(len(err)), err)
            q = np.clip(q + dq, self.limits[:, 0], self.limits[:, 1])

        self._set_arm(q)
        reached = np.linalg.norm(np.asarray(target_pos) - self.eef_pos())
        self._set_arm(saved)
        return q, reached


# ------------------------------------------------------------------ RRT-Connect


class RRTConnect:
    def __init__(self, scene: Scene, args: Args, rng):
        self.s = scene
        self.a = args
        self.rng = rng

    def _sample(self):
        lo, hi = self.s.limits[:, 0], self.s.limits[:, 1]
        return self.rng.uniform(lo, hi)

    def _edge_clear(self, q1, q2):
        """Sample an edge finely enough that nothing can pass between checks.

        A joint-space spacing alone is not a guarantee: near a stretched-out
        arm the same 0.05 rad moves the end effector several centimetres, far
        enough to step straight over a thin object between two samples. The
        spacing is therefore whichever of the joint and Cartesian limits asks
        for more samples.
        """
        q1, q2 = np.asarray(q1, dtype=float), np.asarray(q2, dtype=float)
        n = max(2, int(np.ceil(np.linalg.norm(q2-q1) / self.a.collision_check_step)),
                int(np.ceil(self.s.eef_span(q1, q2) / self.a.collision_check_cartesian)))
        for i in range(1, n + 1):
            if not self.s.valid(q1 + (q2 - q1) * (i / n)):
                return False
        return True

    @staticmethod
    def _nearest(tree, q):
        return int(np.argmin([np.linalg.norm(n - q) for n in tree]))

    def _steer(self, frm, to):
        d = to - frm
        n = np.linalg.norm(d)
        return to.copy() if n <= self.a.rrt_step else frm + d / n * self.a.rrt_step

    def plan(self, q_start, q_goal) -> Optional[List[np.ndarray]]:
        self.last_stats = dict(nodes=0, raw_len=0, searched=False)
        if not self.s.valid(q_goal):
            return None
        if not self.a.force_rrt and self._edge_clear(q_start, q_goal):
            return [q_start, q_goal]  # straight line is free, no need to search

        ta, tb = [q_start], [q_goal]
        pa, pb = {0: None}, {0: None}
        swapped = False

        for _ in range(self.a.rrt_max_iters):
            if getattr(self, "on_progress", None) is not None:
                self.on_progress()
            q_rand = self._sample()
            i = self._nearest(ta, q_rand)
            q_new = self._steer(ta[i], q_rand)
            if not (self.s.valid(q_new) and self._edge_clear(ta[i], q_new)):
                ta, tb, pa, pb = tb, ta, pb, pa
                swapped = not swapped
                continue
            ta.append(q_new)
            pa[len(ta) - 1] = i

            # try to connect the other tree straight to the new node
            j = self._nearest(tb, q_new)
            q_c = tb[j]
            if self._edge_clear(q_new, q_c):
                path_a = self._walk(ta, pa, len(ta) - 1)
                path_b = self._walk(tb, pb, j)
                if swapped:
                    path_a, path_b = path_b, path_a
                raw = path_a[::-1] + path_b
                self.last_stats = dict(
                    nodes=len(ta) + len(tb), raw_len=len(raw), searched=True
                )
                return self._shortcut(raw)

            ta, tb, pa, pb = tb, ta, pb, pa
            swapped = not swapped

        return None

    @staticmethod
    def _walk(tree, parent, idx):
        out = []
        while idx is not None:
            out.append(tree[idx])
            idx = parent[idx]
        return out  # leaf -> root

    def _shortcut(self, path, rounds=None):
        """RRT paths are jagged; greedily replace detours with straight edges."""
        path = [np.asarray(p) for p in path]
        for _ in range(self.a.shortcut_rounds if rounds is None else rounds):
            if getattr(self, "on_progress", None) is not None:
                self.on_progress()
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self._edge_clear(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path


# ------------------------------------------------------------------- execution


def drive_to(env, scene, q_target, gripper, render, max_steps=300, tol=0.02):
    """JOINT_POSITION control is delta-based; step toward the target config."""
    for _ in range(max_steps):
        q = scene.arm_qpos()
        err = q_target - q
        if np.max(np.abs(err)) < tol:
            return True
        action = np.zeros(env.env.action_dim)
        action[:len(q_target)] = np.clip(err / CTRL_DELTA_MAX, -1.0, 1.0)
        action[-1] = gripper
        env.step(action)
        if render:
            env.env.render()
    return False


def hold(env, steps, gripper, render):
    """Let the sim settle / let the gripper actually close."""
    for _ in range(steps):
        action = np.zeros(env.env.action_dim)
        action[-1] = gripper
        env.step(action)
        if render:
            env.env.render()


# ------------------------------------------------------------------- waypoints


def auto_waypoints(env, task):
    """A pick-and-place chain built from the task's own objects of interest."""
    obs = env.env._get_observations()
    names = [o for o in env.env.obj_of_interest]
    target, dest = names[0], names[-1]
    tp = obs[f"{target}_pos"]
    dp = obs[f"{dest}_pos"]

    return [
        {"name": "above_object", "pos": [tp[0], tp[1], tp[2] + 0.12], "gripper": "open"},
        {"name": "at_object", "pos": [tp[0], tp[1], tp[2] + 0.01], "gripper": "open"},
        {"name": "grasp", "pos": [tp[0], tp[1], tp[2] + 0.01], "gripper": "close"},
        {"name": "lift", "pos": [tp[0], tp[1], tp[2] + 0.18], "gripper": "close"},
        {"name": "above_dest", "pos": [dp[0], dp[1], dp[2] + 0.22], "gripper": "close"},
        {"name": "release", "pos": [dp[0], dp[1], dp[2] + 0.15], "gripper": "open"},
    ]


def main(args: Args) -> None:
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    kwargs = dict(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        controller="JOINT_POSITION",
    )
    env = (
        ControlEnv(has_renderer=True, has_offscreen_renderer=True, **kwargs)
        if args.render
        else OffScreenRenderEnv(**kwargs)
    )
    env.seed(args.seed)
    obs = env.reset()

    print(f"task [{args.task_id}]: {task.language}")

    if args.list_objects:
        print("\nobjects in scene (name, position):")
        for k in sorted(obs):
            if k.endswith("_pos") and not k.startswith("robot0") and "_to_robot0" not in k:
                print(f"  {k[:-4]:28s} {np.round(obs[k], 4)}")
        print(f"\n  obj_of_interest: {list(env.env.obj_of_interest)}")
        print(f"  eef_pos:         {np.round(obs['robot0_eef_pos'], 4)}")
        env.close()
        return

    ignore = args.ignore_collision_with
    if ignore is None:
        ignore = list(env.env.obj_of_interest)
    print(f"collision-ignored bodies: {ignore}")

    scene = Scene(env, ignore_bodies=ignore, args=args)
    hold(env, args.settle_steps, GRIPPER_OPEN, args.render)

    if args.waypoints:
        wps = json.loads(pathlib.Path(args.waypoints).read_text())
    else:
        wps = auto_waypoints(env, task)

    print(f"\n{len(wps)} waypoints, planning with RRT-Connect\n")

    planner = RRTConnect(scene, args, rng)
    q_cur = scene.arm_qpos()

    for wp in wps:
        name = wp.get("name", "?")
        grip = GRIPPER_CLOSE if wp.get("gripper", "open") == "close" else GRIPPER_OPEN
        target = np.asarray(wp["pos"], dtype=float)

        q_goal, ik_err = scene.ik(target, q_init=q_cur)
        status = "ok" if ik_err < 0.01 else f"IK off by {ik_err * 100:.1f}cm"
        print(f"[{name}] target {np.round(target, 3)}  ({status})")

        if np.linalg.norm(q_goal - q_cur) < 1e-4:
            path = [q_cur, q_goal]
        else:
            path = planner.plan(q_cur, q_goal)

        if path is None:
            print(f"  !! no collision-free path to '{name}', skipping")
            continue

        st = planner.last_stats
        if st["searched"]:
            print(
                f"  RRT searched: {st['nodes']} nodes, raw path {st['raw_len']}"
                f" -> {len(path)} after smoothing"
            )
        else:
            print(f"  straight line free, no search ({len(path)} configs)")
        for q in path[1:]:
            drive_to(env, scene, q, grip, args.render)

        hold(env, args.settle_steps, grip, args.render)
        q_cur = scene.arm_qpos()
        err = np.linalg.norm(scene.eef_pos() - target)
        print(f"  reached, eef error {err * 100:.1f}cm | success={env.check_success()}")

    print(f"\ntask success: {env.check_success()}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
