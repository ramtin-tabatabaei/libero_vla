"""Turn a LIBERO task's BDDL goal into subgoals and execute them with RRT-Connect.

No VLA. The task description is never read as text -- everything comes from the
symbolic goal the benchmark already ships in the .bddl file:

    (:goal (And (Open wooden_cabinet_1_top_region)
                (In  akita_black_bowl_1 wooden_cabinet_1_top_region)))

Three layers:

  1. decompose()  goal predicates -> ordered symbolic Steps, including the
                  preconditions BDDL leaves implicit (you cannot put a bowl in
                  a drawer that is shut, so an `open` Step is inserted).
  2. expand()     each Step -> a list of Subgoals, which are motion primitives
                  with targets named relative to scene geometry, not baked-in
                  numbers. Targets resolve against the live sim when executed.
  3. Executor     each Subgoal -> IK -> RRT-Connect -> JOINT_POSITION control.

Each Step is then verified with LIBERO's own predicate evaluator, so "did the
subgoal work" is the benchmark's answer, not ours.

    python subgoal_planner.py --task-suite-name libero_goal --task-id 3 --dry-run
    python subgoal_planner.py --task-suite-name libero_goal --task-id 3 --render
    python subgoal_planner.py --task-suite-name libero_10 --task-id 3
"""

import dataclasses
import json
import pathlib
import select
import sys
import termios
import time
import tty
from typing import Callable, List, Literal, Optional, Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv

from waypoint_rrt import CTRL_DELTA_MAX, GRIPPER_CLOSE, GRIPPER_OPEN, RRTConnect, Scene

# Fingers pointing straight down, jaws separating along world y. Position-only
# IK would happily corkscrew the wrist on the way into a grasp; this pins it.
DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
MAX_JAW = 0.07
"""Panda finger separation, less a margin. Anything wider cannot be pinched."""
HOOK_CLEARANCE = 0.009
"""Half a closed finger pair, plus margin -- how far behind a handle to hook."""


def down_yaw(theta: float) -> np.ndarray:
    """The downward grasp orientation, spun by `theta` about the vertical."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ DOWN


def jaw_dir(theta: float) -> np.ndarray:
    """Which way the fingers separate at a given wrist yaw."""
    return down_yaw(theta)[:, 0]


def yaw_for(direction: np.ndarray) -> float:
    """Wrist yaw that makes the jaws separate along `direction` (xy only)."""
    return float(-np.arctan2(direction[0], direction[1]))


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_goal"
    task_id: int = 3
    seed: int = 0

    policy_backend: Literal["pi05"] = "pi05"
    """Initial pretrained policy selection in the studio."""
    policy_port: int = 8000
    """Standard Pi 0.5 server port for VLA waypoint recording."""
    policy_max_steps: Optional[int] = None
    """Direct run limit; None uses the suite limit."""

    dry_run: bool = False
    """Print the subgoal decomposition and exit without moving the robot."""
    render: bool = False
    """Open the MuJoCo window. Needs MUJOCO_GL=glx."""
    pick: bool = False
    """List every task and choose one, instead of using --task-suite-name/--task-id."""
    step: bool = False
    """Wait for a keypress before each subgoal. Space runs it, c runs the rest, q quits."""

    subgoals: Optional[str] = None
    """JSON file of hand-written subgoals to run instead of decomposing the goal."""

    stop_on_failure: bool = False
    """Abort the whole plan the first time a Step fails its predicate check."""

    # --- geometry knobs, all in metres ---
    approach_height: float = 0.15
    """How far above a grasp the arm stages before descending."""
    grasp_depth: float = 0.025
    """How far below an object's top the fingers close on it."""
    lift_height: float = 0.15
    place_clearance: float = 0.07
    """Gap left between object and surface/region before the fingers open."""
    reach_tolerance: float = 0.03
    """How far a planned reach may end up from its target before it counts as failed."""
    line_step: float = 0.02
    """Spacing of the via-points on a straight Cartesian move."""
    motion_speed: float = 1.0
    """Overall trajectory speed scale: 0.5 is half speed; valid range is (0, 1]."""
    travel_speed: float = 0.6
    travel_accel: float = 1.2
    travel_floor: float = 0.2
    carry_speed: float = 0.2
    carry_accel: float = 0.3
    carry_floor: float = 0.4
    """Limits for a move with something in the gripper: gentler than empty
    travel, because acceleration is what shakes an object out of the fingers."""
    """Peak joint speed (rad/s), acceleration (rad/s^2) and shortest duration
    (s) of a free-space move. The controller saturates at 1 rad/s, so the peak
    leaves it room to track."""
    contact_speed: float = 0.075
    contact_accel: float = 0.075
    contact_floor: float = 0.8
    """The same limits for a move that touches: a descent onto an object, a
    lift, a placement, a push. Contact is where overshoot costs a grasp, so
    these are an eighth of the free-space limits."""

    # --- planner knobs, consumed by RRTConnect ---
    force_rrt: bool = False
    rrt_max_iters: int = 4000
    rrt_step: float = 0.15
    collision_check_step: float = 0.05
    """Resolution for checking a planner edge (rad). Smaller is safer and
    slower; an obstacle thinner than the end effector travels between two
    samples can be missed."""
    collision_check_cartesian: float = 0.0075
    """Largest end effector travel allowed between two checks on an edge (m).
    The joint spacing alone does not bound this: near a stretched-out arm the
    same 0.05 rad moves the wrist several centimetres, far enough to step over
    a thin object between samples. Measured against the chord between the
    edge's endpoints, so a curved edge realises roughly 1.4x this; 7.5 mm here
    measured 10.6 mm worst case, against a thinnest scene object of 18 mm."""
    shortcut_rounds: int = 150
    """Greedy smoothing passes over a found path."""
    ik_iters: int = 300
    ik_tol: float = 1e-3
    ik_damping: float = 0.08
    """Damped least squares inverse kinematics: iteration cap, convergence
    tolerance (m), and damping."""
    probe_iters: int = 600
    """Planner budget when testing a candidate grasp rather than committing to one."""
    settle_steps: int = 15
    horizon: int = 100_000
    """Plans run far longer than the benchmark's 1000-step evaluation episode."""


# ============================================================ geometric queries


class World:
    """Symbolic and geometric questions about the live scene.

    BDDL arguments are either objects (`akita_black_bowl_1`) or region sites
    (`wooden_cabinet_1_top_region`); this hides that difference from everything
    above it.
    """

    def __init__(self, env):
        self.env = env
        self.domain = env.env
        self.sim = env.sim
        self.m = getattr(self.sim.model, "_model", self.sim.model)
        self.d = getattr(self.sim.data, "_data", self.sim.data)

    # --- names ------------------------------------------------------------
    def is_site(self, name: str) -> bool:
        return name in self.domain.object_sites_dict

    def parent_body(self, name: str) -> str:
        """The rigid body a BDDL argument belongs to, for collision purposes."""
        if self.is_site(name):
            return self.domain.object_sites_dict[name].parent_name or name
        return name

    def is_movable(self, name: str) -> bool:
        return name in self.domain.objects_dict

    # --- poses ------------------------------------------------------------
    def pos(self, name: str) -> np.ndarray:
        if self.is_site(name):
            return np.array(self.sim.data.get_site_xpos(name), dtype=float)
        bid = self.domain.obj_body_id[name]
        return np.array(self.sim.data.body_xpos[bid], dtype=float)

    def _geoms_of(self, body_prefix: str, collision_only=True) -> List[int]:
        out = []
        for g in range(self.m.ngeom):
            if collision_only and self.m.geom_group[g] != 0:
                continue
            bid = self.m.geom_bodyid[g]
            bname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname == body_prefix or bname.startswith(body_prefix + "_"):
                out.append(g)
        return out

    def aabb(self, name: str):
        """World-axis-aligned bounds of a thing's geometry, as (low, high).

        Collision geometry only. geom_rbound is a bounding *sphere*, which would
        float a bowl a whole plate-radius above a plate, and a visual mesh's own
        box arrives rotated, so re-boxing it in world axes inflates a 11cm bowl
        to 16cm. The convex-decomposition pieces are tight and are what the
        fingers will actually hit.
        """
        p = self.pos(name)
        if self.is_site(name):
            return p.copy(), p.copy()
        geoms = self._geoms_of(name)
        if not geoms:
            return p.copy(), p.copy()
        lows, highs = [], []
        for g in geoms:
            box = self.m.geom_aabb[g]
            R = self.d.geom_xmat[g].reshape(3, 3)
            centre = self.d.geom_xpos[g] + R @ box[:3]
            half = np.abs(R) @ box[3:]
            lows.append(centre - half)
            highs.append(centre + half)
        return np.min(lows, axis=0), np.max(highs, axis=0)

    def top_z(self, name: str) -> float:
        return float(self.aabb(name)[1][2])

    def grasp_candidates(self, name: str, depth: float):
        """Ways to take hold of an object, best first, as (offset, yaw) pairs.

        The offset is from the object's origin to the grip site. Two families,
        decided by the object's own width. A box or a can is narrower than the
        jaws one way round, so spin the wrist to close across that axis and
        pinch near the top. A bowl is wider than the jaws every way round --
        closing on its body just shoves it across the table -- so the fingers
        straddle its rim instead, one inside and one outside, and there are
        four sides of the rim to choose from.

        More than one is offered because the best grasp depends on what else is
        in the way, which is not knowable here: after a drawer is pulled out
        over the table, only the rim facing away from it is still reachable.
        """
        low, high = self.aabb(name)
        origin = self.pos(name)
        centre = 0.5 * (low + high)
        span = high - low

        up = np.array([0.0, 0.0, 1.0])
        out = []
        for axis_i in sorted((0, 1), key=lambda i: span[i]):
            axis = np.eye(3)[axis_i]
            if span[axis_i] <= MAX_JAW:
                z = high[2] - min(depth, 0.4 * float(span[2]))
                out.append((np.array([centre[0], centre[1], z]) - origin,
                            yaw_for(axis), [up]))
                continue
            for side in (1.0, -1.0):
                rim = centre + axis * side * 0.5 * float(span[axis_i])
                # A rim can also be come at from out to the side and above, on
                # the diagonal. That is the only way in under an overhang -- an
                # open drawer, a shelf -- where the air directly over the rim
                # is not free. The climb matters: coming in level would drag
                # the inner finger straight through the wall of the bowl, where
                # on the diagonal it crosses the rim with height to spare.
                out.append((np.array([rim[0], rim[1], high[2] - 0.015]) - origin,
                            yaw_for(axis),
                            [up, (axis * side + up) / np.sqrt(2.0)]))
        # Prefer whatever keeps the wrist closest to the robot, whose base
        # stands at the origin -- a shorter reach is a steadier one.
        return sorted(out, key=lambda c: float(np.linalg.norm((origin + c[0])[:2])))

    def eef(self) -> np.ndarray:
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        return np.array(self.sim.data.site_xpos[sid], dtype=float)

    # --- articulation -----------------------------------------------------
    def joints_of(self, name: str) -> List[str]:
        """Joint names that actuate a region or object, or [] if it is rigid."""
        if self.is_site(name):
            return list(getattr(self.domain.object_sites_dict[name], "joints", []) or [])
        obj = self.domain.get_object(name)
        return list(getattr(obj, "joints", []) or [])

    def articulated_joint(self, name: str) -> Optional[str]:
        """The single non-free joint that opens/turns this thing."""
        for j in self.joints_of(name):
            jid = self.sim.model.joint_name2id(j)
            if self.m.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
                return j
        return None

    def joint_info(self, jname: str) -> dict:
        jid = self.sim.model.joint_name2id(jname)
        adr = self.m.jnt_qposadr[jid]
        return dict(
            id=jid,
            qpos=float(self.d.qpos[adr]),
            axis=np.array(self.d.xaxis[jid], dtype=float),
            anchor=np.array(self.d.xanchor[jid], dtype=float),
            range=np.array(self.m.jnt_range[jid], dtype=float),
            slide=self.m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE,
            body=int(self.m.jnt_bodyid[jid]),
        )

    def handle_grasp(self, jname: str, travel: Optional[np.ndarray] = None):
        """Where and how to take a moving panel by its handle. Returns (point, yaw).

        The grabbable feature is found from geometry alone: on a drawer it is
        whatever sticks furthest out along the direction the drawer travels; on
        anything hinged -- a door handle, the tip of a stove knob -- it is
        whatever sits furthest from the hinge. Neither needs the asset's geom
        names, which differ per object.

        The panel is then *hooked*, not pinched: the point returned sits just
        behind the handle and the wrist turns so the closed fingers present
        their narrow side to that gap. LIBERO's cabinet pull leaves 16mm
        between the bar and the drawer face, which no open jaw fits into, and a
        bar pinched front-to-back slides straight out of the pads the moment
        you pull on it. A closed pair of fingers behind the bar does not.
        """
        info = self.joint_info(jname)
        axis = info["axis"] / max(np.linalg.norm(info["axis"]), 1e-9)
        if info["slide"] and travel is not None:
            score_of = lambda p: float(p @ travel)
        else:
            def score_of(p):
                r = p - info["anchor"]
                return float(np.linalg.norm(r - axis * (axis @ r)))

        best, best_score = None, -np.inf
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] != info["body"] or self.m.geom_group[g] != 0:
                continue
            p = np.array(self.d.geom_xpos[g], dtype=float)
            if score_of(p) > best_score:
                best, best_score = g, score_of(p)
        if best is None:
            return np.array(self.d.xpos[info["body"]], dtype=float), 0.0

        box = self.m.geom_aabb[best]
        R = self.d.geom_xmat[best].reshape(3, 3)
        point = np.array(self.d.geom_xpos[best] + R @ box[:3], dtype=float)
        half = np.abs(R) @ box[3:]
        if travel is not None:
            direction = travel / max(np.linalg.norm(travel), 1e-9)
            behind = float(np.abs(direction) @ half) + HOOK_CLEARANCE
            point = point - direction * behind
        long_axis = np.eye(3)[0 if half[0] >= half[1] else 1]
        return point, yaw_for(long_axis)


def target_qpos(world: World, name: str, jname: str, test: str) -> float:
    """Where the joint has to end up for LIBERO to call the thing open/on.

    The thresholds live in each object class (is_open, turn_on, ...) and differ
    in sign between a cabinet and a microwave, so rather than reading the
    numbers we sweep the joint range and ask the object itself.

    Only far enough to clear the threshold, never the full travel: a drawer run
    all the way out reaches 16cm across the table, which on the very task that
    wants one opened parks it directly over the bowl that has to go inside.
    """
    parent = world.parent_body(name)
    obj = world.domain.get_object(parent)
    fn = getattr(obj, test)
    info = world.joint_info(jname)
    lo, hi = info["range"]
    qs = np.linspace(lo, hi, 401)
    ok = np.array([q for q in qs if fn(q)])
    if ok.size == 0:
        raise ValueError(f"no qpos in {lo:.3f}..{hi:.3f} satisfies {test} for {parent}")

    cur = info["qpos"]
    threshold = ok[np.argmin(np.abs(ok - cur))]
    span = float(ok.max() - ok.min())  # margin covers control error
    q = threshold + (0.3 * span if threshold > cur else -0.3 * span)
    return float(np.clip(q, ok.min(), ok.max()))


# ================================================================== subgoals


@dataclasses.dataclass
class Target:
    """A point named relative to the scene, resolved when the subgoal runs.

    Resolving late matters: the drawer moves while you pull it, and the object
    you are carrying moves with the gripper.
    """

    frame: str  # "world" | "object" | "site" | "eef" | "handle"
    name: str = ""
    offset: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    on_top: bool = False
    """Offset from the thing's top surface instead of its origin."""
    carry: bool = False
    """Correct for where the held object sits in the gripper."""
    seat: bool = False
    """Read the offset as a gap under the held object, not above its origin."""
    direction: Optional[np.ndarray] = None
    """Which way a drawer travels, for frame="handle"."""
    hold_xy: bool = False
    """Take only the height from this target; stay over the current spot."""

    def describe(self) -> str:
        off = np.round(self.offset, 3)
        base = {"world": "", "eef": "eef", "handle": f"handle({self.name})"}.get(
            self.frame, f"{'top of ' if self.on_top else ''}{self.name}"
        )
        tail = (" +carry" if self.carry else "") + (" +seat" if self.seat else "")
        return f"{base} {off}{tail}".strip()


@dataclasses.dataclass
class Subgoal:
    name: str
    kind: str
    """reach: plan around obstacles | line: straight, contact-tolerant
       grip: open/close the fingers | arc: sweep about a joint axis"""
    target: Optional[Target] = None
    gripper: Optional[float] = None
    ignore: Sequence[str] = ()
    """Bodies excluded from collision checks for this motion."""
    joint: str = ""
    sweep: float = 0.0
    """Radians about `joint` for kind="arc"."""
    yaw: float = 0.0
    """Wrist spin about the vertical, so the jaws close across the right axis."""
    check_grasp: str = ""
    """Object that must be between the fingers for the plan to continue."""

    def describe(self) -> str:
        bits = [f"{self.kind:5s} {self.name}"]
        if self.target is not None:
            bits.append(f"-> {self.target.describe()}")
        if self.gripper is not None:
            bits.append("[close]" if self.gripper > 0 else "[open]")
        if abs(self.yaw) > 1e-6:
            bits.append(f"yaw {np.degrees(self.yaw):+.0f}deg")
        if self.kind == "arc":
            bits.append(f"{self.sweep:+.2f} rad about {self.joint}")
        if self.check_grasp:
            bits.append(f"(verify holding {self.check_grasp})")
        return " ".join(bits)


@dataclasses.dataclass
class Step:
    """One symbolic predicate and the subgoals that achieve it."""

    predicate: List[str]
    subgoals: List[Subgoal]
    """The expansion as of planning time -- what --dry-run prints."""
    expand: Optional[Callable] = None
    """Re-expand against live geometry. The scene moves between steps: pulling
    a drawer out over the table changes which side of a bowl is reachable."""
    implicit: bool = False
    """Inserted as a precondition rather than asked for by the .bddl goal."""
    verify: Optional[Callable[[], bool]] = None

    def label(self) -> str:
        head = " ".join(self.predicate)
        return f"{head}{'   [implicit precondition]' if self.implicit else ''}"


# ------------------------------------------------------------------ expansion


def expand_place(world: World, args: Args, obj: str, dest: str, into: bool,
                 probe=None) -> List[Subgoal]:
    """Pick `obj` and put it in/on `dest`."""
    # Reaching for the object may touch the object -- otherwise no grasp is
    # ever plannable -- but nothing else is fair game. Only once it is in hand
    # does the destination stop being an obstacle, because you cannot lower a
    # bowl into a drawer that counts as a crash either.
    take = [world.parent_body(obj)]
    put = take + [world.parent_body(dest)]
    up = np.array([0.0, 0.0, 1.0])
    frame = "site" if world.is_site(dest) else "object"
    drop = args.place_clearance if into else args.place_clearance * 0.5

    candidates = world.grasp_candidates(obj, args.grasp_depth)
    grasp, yaw, stage = candidates[0][0], candidates[0][1], up * args.approach_height
    if probe is not None:
        # Staging away from a grasp keeps the last centimetres a clean straight
        # line, but the clear space around an object is not always itself
        # reachable. Try each candidate's approach directions, near before far.
        for offset, theta, dirs in candidates:
            base = world.pos(obj) + offset
            if not probe(base, theta, take):
                continue
            # The object is an obstacle on the way in and only stops being one
            # at the grasp itself, or the planner would barge through it.
            found = next(
                (d * h for d in dirs for h in (args.approach_height, 0.5 * args.approach_height)
                 if probe(base + d * h, theta, (), path=True)),
                None,
            )
            if found is not None:
                grasp, yaw, stage = offset, theta, found
                break

    # The whole chain keeps the grasp's wrist yaw: carry_offset is measured in
    # world axes, so spinning the wrist mid-carry would invalidate it.
    return [
        Subgoal("approach", "reach", Target("object", obj, grasp + stage),
                GRIPPER_OPEN, (), yaw=yaw),
        Subgoal("descend", "line", Target("object", obj, grasp), None, take, yaw=yaw),
        Subgoal("grasp", "grip", None, GRIPPER_CLOSE, take, yaw=yaw, check_grasp=obj),
        # Rise all the way to the carrying altitude before setting off, rather
        # than a fixed distance: the held object is not in the collision model,
        # so a traverse that dips is a traverse that sweeps it through whatever
        # it passes -- and closes the drawer it is being carried towards.
        Subgoal("lift", "line",
                Target(frame, dest, up * (drop + args.lift_height),
                       on_top=not into, carry=True, seat=True, hold_xy=True),
                None, take, yaw=yaw),
        # The carry is planned with the destination still an obstacle: it is
        # only in the last few centimetres straight down that touching it is
        # intended, and a route that ignores it shoves the open drawer shut on
        # the way over.
        Subgoal("transfer", "reach",
                Target(frame, dest, up * (drop + args.lift_height),
                       on_top=not into, carry=True, seat=True), None, take, yaw=yaw),
        Subgoal("lower", "line",
                Target(frame, dest, up * drop, on_top=not into, carry=True, seat=True),
                None, put, yaw=yaw),
        Subgoal("release", "grip", None, GRIPPER_OPEN, put, yaw=yaw),
        Subgoal("retreat", "line", Target("eef", offset=up * args.lift_height),
                None, put, yaw=yaw),
    ]


def expand_articulate(world: World, args: Args, region: str, opening: bool) -> List[Subgoal]:
    """Drag a drawer or door to its open or closed pose by its handle."""
    jname = world.articulated_joint(region)
    if jname is None:
        raise ValueError(f"{region} has no slide/hinge joint to actuate")
    info = world.joint_info(jname)
    q_goal = target_qpos(world, region, jname, "is_open" if opening else "is_close")
    delta = q_goal - info["qpos"]
    body = world.parent_body(region)
    up = np.array([0.0, 0.0, 1.0])

    # Approach the handle from the side it swings/slides towards, so the arm is
    # never inside the panel's own swept volume.
    travel = info["axis"] * np.sign(delta)
    if not info["slide"]:
        radial = world.handle_grasp(jname)[0] - info["anchor"]
        travel = np.cross(info["axis"], radial) * np.sign(delta)
        travel /= max(np.linalg.norm(travel), 1e-9)

    _, yaw = world.handle_grasp(jname, travel)
    pull_label = "pull" if opening else "push"
    # Fingers stay shut throughout: this is a hook, so there is nothing to
    # clamp, and an open jaw would not fit behind the handle in the first place.
    subgoals = [
        Subgoal("stage", "reach", Target("handle", jname, up * 0.12, direction=travel),
                GRIPPER_CLOSE, [body], yaw=yaw),
        Subgoal("engage", "line", Target("handle", jname, direction=travel),
                None, [body], yaw=yaw),
    ]
    if info["slide"]:
        subgoals.append(
            Subgoal(pull_label, "line", Target("eef", offset=travel * abs(delta)),
                    None, [body], yaw=yaw)
        )
    else:
        subgoals.append(Subgoal(pull_label, "arc", None, None, [body],
                                joint=jname, sweep=float(delta), yaw=yaw))
    subgoals.append(
        Subgoal("back_off", "line", Target("eef", offset=up * 0.12), None, [body], yaw=yaw)
    )
    return subgoals


def expand_switch(world: World, args: Args, obj: str, on: bool) -> List[Subgoal]:
    """Twist a knob with the closed gripper: press down, then sweep an arc."""
    jname = world.articulated_joint(obj)
    if jname is None:
        raise ValueError(f"{obj} has no knob joint")
    info = world.joint_info(jname)
    q_goal = target_qpos(world, obj, jname, "turn_on" if on else "turn_off")
    delta = q_goal - info["qpos"]
    body = world.parent_body(obj)
    up = np.array([0.0, 0.0, 1.0])
    return [
        Subgoal("stage", "reach", Target("handle", jname, up * 0.10),
                GRIPPER_CLOSE, [body]),
        Subgoal("engage", "line", Target("handle", jname, up * 0.005), None, [body]),
        Subgoal("turn", "arc", None, None, [body], joint=jname, sweep=float(delta)),
        Subgoal("withdraw", "line", Target("eef", offset=up * 0.12), None, [body]),
    ]


# --------------------------------------------------------------- decomposition

# Ordering between goal predicates. Opening comes before everything because a
# shut drawer blocks the placement; shutting comes last for the same reason;
# a stove is switched on once whatever belongs on it is already there.
PRIORITY = {"open": 0, "in": 10, "on": 10, "turnon": 20, "turnoff": 20, "close": 30}


def decompose(world: World, args: Args) -> List[Step]:
    goals = [[t.lower() for t in state] for state in world.domain.parsed_problem["goal_state"]]

    # BDDL states the goal, not the route to it. "open the top drawer and put
    # the bowl inside" has a one-predicate goal -- the opening is left implicit.
    implicit: List[List[str]] = []
    for pred, *rest in goals:
        if pred not in ("in", "on") or len(rest) != 2:
            continue
        dest = rest[1]
        jname = world.articulated_joint(dest)
        if jname is None:
            continue
        state = world.domain.object_states_dict[dest]
        if hasattr(state, "is_open") and not state.is_open():
            if ["open", dest] not in goals and ["open", dest] not in implicit:
                implicit.append(["open", dest])

    ordered = sorted(
        [(p, False) for p in goals] + [(p, True) for p in implicit],
        key=lambda item: PRIORITY.get(item[0][0], 15),
    )

    steps: List[Step] = []
    for pred_args, is_implicit in ordered:
        pred, rest = pred_args[0], pred_args[1:]
        if pred in ("in", "on"):
            expand = (lambda probe=None, r=rest, p=pred:
                      expand_place(world, args, r[0], r[1], into=p == "in", probe=probe))
        elif pred in ("open", "close"):
            expand = (lambda probe=None, r=rest, p=pred:
                      expand_articulate(world, args, r[0], opening=p == "open"))
        elif pred in ("turnon", "turnoff"):
            expand = (lambda probe=None, r=rest, p=pred:
                      expand_switch(world, args, r[0], on=p == "turnon"))
        else:
            print(f"  !! no skill for predicate {pred!r}, skipping")
            continue
        steps.append(
            Step(
                predicate=pred_args,
                subgoals=expand(),
                expand=expand,
                implicit=is_implicit,
                verify=(lambda s=pred_args: bool(world.domain._eval_predicate(s))),
            )
        )
    return steps


# =================================================================== execution


class Executor:
    def __init__(self, env, world: World, args: Args):
        if not np.isfinite(args.motion_speed) or not 0 < args.motion_speed <= 1:
            raise ValueError("Motion speed must be greater than 0 and at most 1.")
        self.env = env
        self.world = world
        self.args = args
        self._scenes = {}
        self.scene = Scene(env, args=args)
        self.planner = RRTConnect(self.scene, args, np.random.default_rng(args.seed))
        # Same planner on a short leash: a probe that fails must fail quickly,
        # since several candidate grasps may be tried before one is committed to.
        self.prober = RRTConnect(
            self.scene,
            dataclasses.replace(args, rrt_max_iters=args.probe_iters),
            np.random.default_rng(args.seed),
        )
        self.action_dim = env.env.action_dim
        self.grip_idx = self.action_dim - 1
        self.gripper = GRIPPER_OPEN
        self.on_step = None
        """Optional per-sim-step callback, for a host that owns the main thread."""
        self.q_hold = self.scene.arm_qpos()
        self.arm_dof = len(self.scene.qadr)
        self.bias = np.zeros(self.arm_dof)
        self.carry_offset = np.zeros(3)
        """Where the held object sits relative to the grip site, measured at grasp."""
        self.carry_bottom = 0.0
        """How far the held object's origin sits above its own underside."""
        self.holding = ""
        self.arrival_grace_seconds = None
        """Optional elapsed-time budget after a trajectory's scheduled endpoint."""
        self.retry_motion = True

    # --- primitives -------------------------------------------------------
    def _step(self, action):
        self.obs, _, _, _ = self.env.step(action)
        if self.args.render:
            self.env.env.render()
        if self.on_step is not None:
            # A GUI owning this process pumps its event loop here, so a motion
            # of several hundred sim steps does not read as a frozen window.
            self.on_step()

    def _apply(self) -> float:
        """One control step towards the standing joint target; returns the error.

        robosuite's JOINT_POSITION controller reads a delta from wherever the
        arm *is*, so under gravity a constant command has a constant droop as
        its fixed point -- repeating it forever never closes the gap. The wrist
        settles a tenth of a radian low and stays there, which is five
        centimetres at the fingertips, enough to miss every grasp. `bias` is
        the integral term the controller does not have; it is held at zero
        while the command is saturated, so it cannot wind up during long moves.
        """
        err = self.q_hold - self.scene.arm_qpos()
        near = np.abs(err) < CTRL_DELTA_MAX
        # Integrate slowly (1/s, or .05 per step at 20 Hz). The old .25/step
        # correction accumulated tracking lag during motion and then pushed
        # past the target when the trajectory decelerated.
        integral_step = self.env.env.control_timestep * err
        self.bias = np.where(near, np.clip(self.bias + integral_step, -0.6, 0.6), 0.0)
        a = np.zeros(self.action_dim)
        a[:self.arm_dof] = np.clip((err + self.bias) / CTRL_DELTA_MAX, -1.0, 1.0)
        a[self.grip_idx] = self.gripper
        self._step(a)
        return float(np.max(np.abs(err)))

    def hold(self, steps):
        for _ in range(steps):
            self._apply()

    def _drive_to_q(self, q_target, max_steps=300, tol=0.004):
        self._follow_path([self.scene.arm_qpos(), np.asarray(q_target, dtype=float)])
        return self._settle(max_steps, tol)

    def follow(self, path, max_steps=300, tol=0.004, limits=None):
        """Run a planned joint path as one trajectory and settle once at the end.

        Interpolation stays on the supplied edges, so nothing cuts a corner.
        Stopping and settling at every planner waypoint costs seconds per move
        and buys nothing: the corners are not targets, only the last one is.
        """
        path = [np.asarray(self.scene.arm_qpos(), dtype=float)]+[
            np.asarray(q, dtype=float) for q in path[1:]]
        self._follow_path(path, *(limits or ()))
        if self._settle(max_steps, tol):
            return True
        if not self.retry_motion:
            return False
        # Tracking lag accumulates across a path that is never re-referenced to
        # where the arm actually is. Re-drive the last edge from there.
        return self._drive_to_q(path[-1], max_steps, tol)

    def _settle(self, max_steps=120, tol=0.004):
        """Finish at rest, rather than stopping on the first crossing of a target."""
        deadline = (None if self.arrival_grace_seconds is None else
                    time.monotonic() + self.arrival_grace_seconds)
        quiet = 0
        for _ in range(max_steps):
            if deadline is not None and time.monotonic() >= deadline:
                return False
            before = self.scene.arm_qpos()
            self._apply()
            after = self.scene.arm_qpos()
            quiet = quiet + 1 if (np.max(np.abs(self.q_hold-after)) < tol
                                  and np.max(np.abs(after-before)) < .0003) else 0
            if quiet >= 3:
                return True
        return False

    def _follow_path(self, path, speed=None, accel=None, floor=None):
        """Stream a time-scaled path; intermediate Cartesian samples are not stops.

        Quintic timing starts and ends with zero velocity and acceleration.
        Interpolation stays on the supplied joint edges. RRT corners are handled
        separately by reach(), so smoothing does not cut across obstacles.
        """
        path = np.asarray(path)
        distance = np.r_[0., np.cumsum(np.max(np.abs(np.diff(path, axis=0)), axis=1))]
        keep = np.r_[True, np.diff(distance) > 1e-9]
        path, distance = path[keep], distance[keep]
        if len(path) < 2:
            self.q_hold = path[-1].copy()
            return
        dt = self.env.env.control_timestep
        length = distance[-1]
        speed = self.args.travel_speed if speed is None else speed
        accel = self.args.travel_accel if accel is None else accel
        floor = self.args.travel_floor if floor is None else floor
        duration = max(floor, 1.875 * length / speed, np.sqrt(5.774 * length / accel))
        steps = int(np.ceil(duration / (dt * self.args.motion_speed)))
        for i in range(1, steps+1):
            t = i / steps
            fraction = t**3 * (10 - 15*t + 6*t*t)
            self.q_hold = np.array([np.interp(fraction*length, distance, path[:, j])
                                    for j in range(path.shape[1])])
            self._apply()

    def _cartesian_path(self, positions, ori, speed=None, accel=None, floor=None,
                        slow_out=None, slow_in=None, quick=None):
        """Warm-start IK from the preceding solution and interpolate wrist rotation.

        ``slow_out`` and ``slow_in`` are counts of leading and trailing samples
        to time at the contact limits while the rest of the stroke runs at
        ``quick``. A descent onto an object is mostly air; timing the whole of
        it for the touch spends nearly all of it crossing that air. The phases
        meet at zero velocity, so nothing is thrown, and the move still settles
        once, at the end. Given neither, the whole stroke is a contact move.
        """
        start_ori = self.env.sim.data.site_xmat[self.scene.site_id].reshape(3, 3).copy()
        orientations = Slerp([0, 1], Rotation.from_matrix([start_ori, ori]))
        path = [self.scene.arm_qpos()]
        for i, via in enumerate(positions, 1):
            rotation = orientations(i / len(positions)).as_matrix()
            q, error = self.scene.ik(via, q_init=path[-1], target_ori=rotation)
            if error > .02:
                return f"unreachable: Cartesian path misses by {error*100:.1f} cm"
            path.append(q)
            if getattr(self.planner, "on_progress", None) is not None:
                self.planner.on_progress()
        # A Cartesian path is a contact move unless the caller says otherwise.
        slow = (self.args.contact_speed if speed is None else speed,
                self.args.contact_accel if accel is None else accel,
                self.args.contact_floor if floor is None else floor)
        if slow_out is None and slow_in is None:
            self._follow_path(path, *slow)
        else:
            quick = quick or (self.args.travel_speed, self.args.travel_accel,
                              self.args.travel_floor)
            first = max(0, min(slow_out or 0, len(path)-1))
            last = max(first, len(path)-1-max(0, slow_in or 0))
            for segment, limits in ((path[:first+1], slow), (path[first:last+1], quick),
                                    (path[last:], slow)):
                if len(segment) > 1:
                    self._follow_path(segment, *limits)
        if not self._settle():
            return "failed: robot did not settle at the target"
        self.hold(self.args.settle_steps)
        return "ok"

    def _set_ignore(self, bodies: Sequence[str]):
        key = tuple(sorted(bodies))
        if key not in self._scenes:
            self._scenes[key] = Scene(self.env, ignore_bodies=list(key), args=self.args)
        self.scene = self._scenes[key]
        self.planner.s = self.scene

    def probe(self, pos, yaw: float, ignore: Sequence[str], path=False) -> bool:
        """Can the arm get to this pose? Used to choose between candidate grasps.

        `path` asks the harder question -- whether the planner can actually
        route there from here, not merely whether standing there is legal. A
        drawer pulled out over the table leaves the near rim of a bowl in clear
        air and still walls off every way of reaching it.
        """
        key = tuple(sorted(ignore))
        if key not in self._scenes:
            self._scenes[key] = Scene(self.env, ignore_bodies=list(key), args=self.args)
        scene = self._scenes[key]
        q, err = scene.ik(pos, q_init=scene.arm_qpos(), target_ori=down_yaw(yaw))
        if err >= 0.02 or not scene.valid(q):
            return False
        if not path:
            return True
        self.prober.s = scene
        return self.prober.plan(scene.arm_qpos(), q) is not None

    def resolve(self, t: Target) -> np.ndarray:
        if t.frame == "world":
            base = np.zeros(3)
        elif t.frame == "eef":
            base = self.world.eef()
        elif t.frame == "handle":
            base, _ = self.world.handle_grasp(t.name, t.direction)
        else:
            base = self.world.pos(t.name)
            if t.on_top:
                base = np.array([base[0], base[1], self.world.top_z(t.name)])
        target = base + t.offset
        if t.seat:
            # The offset is the gap under the object, so lift by however far the
            # object's origin sits above its own underside.
            target = target + np.array([0.0, 0.0, self.carry_bottom])
        if t.carry:
            # Command where the gripper must be for the *object* to land there.
            target = target - self.carry_offset
        if t.hold_xy:
            here = self.world.eef()
            target = np.array([here[0], here[1], target[2]])
        return target

    def reach(self, pos, ori=None) -> str:
        ori = DOWN if ori is None else ori
        q_cur = self.scene.arm_qpos()
        q_goal, err = self.scene.ik(pos, q_init=q_cur, target_ori=ori)
        if err > 0.05:
            return f"unreachable, closest IK solution is {err * 100:.1f}cm off"
        path = self.planner.plan(q_cur, q_goal)
        if path is None:
            return "no collision-free path"
        if not self.follow(path):
            return "failed: robot did not settle on the planned path"
        self.hold(self.args.settle_steps)
        # A staged pose that is centimetres out turns the next descent into a
        # miss, so an unconverged drive counts as a failure, not a warning.
        gap = float(np.linalg.norm(self.world.eef() - pos))
        if gap > self.args.reach_tolerance:
            return f"failed: stopped {gap * 100:.1f}cm short of the target"
        return f"ok ({len(path)} configs, {gap * 100:.1f}cm)"

    def line(self, pos, ori=None, speed=None, accel=None, floor=None,
             slow_out=None, slow_in=None, quick=None) -> str:
        """Straight Cartesian move, not collision-checked.

        Descending onto an object, pulling a drawer and lifting a grasp are all
        motions that are *supposed* to be in contact; a checked planner refuses
        every one of them.

        Dense IK samples form one continuous trajectory, including wrist rotation.
        """
        ori = DOWN if ori is None else ori
        start = self.world.eef()
        span = float(np.linalg.norm(np.asarray(pos) - start))
        start_ori = self.env.sim.data.site_xmat[self.scene.site_id].reshape(3, 3)
        angle = Rotation.from_matrix(np.asarray(ori) @ start_ori.T).magnitude()
        waypoints = max(2, int(np.ceil(span / min(self.args.line_step, .005))), int(np.ceil(angle / .04)))
        # slow_out/slow_in arrive in metres; the via-points are evenly spaced.
        spacing = span/waypoints if span > 1e-9 else 0.
        def samples(metres):
            if metres is None:
                return None
            return 0 if spacing <= 0 else int(np.ceil(metres/spacing))
        result = self._cartesian_path([start + (np.asarray(pos)-start)*(i/waypoints)
                                       for i in range(1, waypoints+1)], ori,
                                      speed, accel, floor,
                                      samples(slow_out), samples(slow_in), quick)
        if result != "ok":
            return result
        return f"ok ({np.linalg.norm(self.world.eef() - pos) * 100:.1f}cm)"

    def arc(self, jname: str, sweep: float, ori=None, steps=12) -> str:
        """Sweep the gripper around a joint's axis, dragging whatever it holds."""
        ori = DOWN if ori is None else ori
        info = self.world.joint_info(jname)
        anchor, axis = info["anchor"], info["axis"] / max(np.linalg.norm(info["axis"]), 1e-9)
        start = self.world.eef()
        radial = start - anchor
        steps = max(steps, int(np.ceil(abs(sweep) / .025)))
        positions = []
        for i in range(1, steps + 1):
            theta = sweep * (i / steps)
            c, s = np.cos(theta), np.sin(theta)
            # Rodrigues rotation of the radius vector about the joint axis.
            rotated = (radial * c + np.cross(axis, radial) * s
                       + axis * float(axis @ radial) * (1 - c))
            positions.append(anchor + rotated)
        result = self._cartesian_path(positions, ori)
        if result != "ok":
            return result
        return f"ok (swept {sweep:+.2f} rad, joint now {self.world.joint_info(jname)['qpos']:+.3f})"

    # --- subgoal dispatch -------------------------------------------------
    def run_subgoal(self, sg: Subgoal) -> str:
        self._set_ignore(sg.ignore)
        if sg.gripper is not None:
            self.gripper = sg.gripper

        if sg.kind == "grip":
            self.hold(30)
            if sg.check_grasp:
                domain = self.world.domain
                obj = domain.objects_dict.get(sg.check_grasp)
                if obj is not None and not domain._check_grasp(
                    domain.robots[0].gripper, obj.contact_geoms
                ):
                    return "failed: nothing between the fingers"
                self.carry_offset = self.world.pos(sg.check_grasp) - self.world.eef()
                low, _ = self.world.aabb(sg.check_grasp)
                self.carry_bottom = float(self.world.pos(sg.check_grasp)[2] - low[2])
                self.holding = sg.check_grasp
            elif self.gripper == GRIPPER_OPEN:
                self.carry_offset, self.carry_bottom, self.holding = np.zeros(3), 0.0, ""
            return "ok"

        ori = down_yaw(sg.yaw)
        if sg.kind == "arc":
            return self.arc(sg.joint, sg.sweep, ori)

        pos = self.resolve(sg.target)
        return self.reach(pos, ori) if sg.kind == "reach" else self.line(pos, ori)

    def _pause(self, sg: Subgoal) -> None:
        """Hold before a subgoal until the user says go."""
        if not self.args.step:
            return
        tick = (lambda: self.env.env.render()) if self.args.render else None
        key = wait_for_key(tick, f"    >>> {sg.describe()}")
        # Wipe the prompt so the subgoal's result line lands in its place.
        print("\r\033[K" if sys.stdout.isatty() else "", end="")
        if key in ("q", "Q", "\x03"):
            raise Abort
        if key in ("c", "C"):
            self.args.step = False

    def run_step(self, step: Step) -> bool:
        subgoals = step.expand(self.probe) if step.expand else step.subgoals
        for sg in subgoals:
            self._pause(sg)
            result = self.run_subgoal(sg)
            print(f"    {sg.describe():58s} {result}")
            if result.startswith(("failed", "unreachable", "no collision-free")):
                return False
        return True


# ======================================================================== main


# ============================================================ hand-written plans

KEYS = {"name", "kind", "at", "pos", "offset", "on_top", "carry", "seat", "hold_xy",
        "gripper", "grasp", "yaw", "ignore", "joint", "sweep"}


def subgoals_from_json(world: World, spec) -> List[Subgoal]:
    """Read a subgoal chain written by hand.

    Every position is named relative to something that is looked up when the
    subgoal runs, not when the file is read:

        "at": an object or region -- "alphabet_soup_1", "basket_1_contain_region"
        "pos": absolute world xyz, for when you really do mean a fixed point
        neither: wherever the gripper already is, i.e. the previous subgoal

    so a file written against one layout still works when the objects move, and
    a lift is "15cm up from here" rather than a number that has to be redone
    every time something upstream changes.
    """
    out = []
    for i, entry in enumerate(spec):
        unknown = set(entry) - KEYS
        if unknown:
            raise ValueError(f"subgoal {i}: unknown field(s) {sorted(unknown)}")
        offset = np.array(entry.get("offset", [0.0, 0.0, 0.0]), dtype=float)
        located = any(k in entry for k in ("at", "pos", "offset"))
        kind = entry.get("kind") or ("reach" if located else "grip")

        if "pos" in entry:
            target = Target("world", offset=np.array(entry["pos"], dtype=float) + offset)
        elif "at" in entry:
            name = entry["at"]
            if not (world.is_site(name) or name in world.domain.obj_body_id):
                raise ValueError(f"subgoal {i}: nothing in the scene called {name!r}")
            target = Target(
                "site" if world.is_site(name) else "object", name, offset,
                on_top=entry.get("on_top", False),
                carry=entry.get("carry", False),
                seat=entry.get("seat", False),
                hold_xy=entry.get("hold_xy", False),
            )
        else:
            target = Target("eef", offset=offset)

        grip = entry.get("gripper")
        out.append(Subgoal(
            name=entry.get("name", kind),
            kind=kind,
            target=None if kind in ("grip", "arc") else target,
            gripper=None if grip is None else (GRIPPER_CLOSE if grip == "close" else GRIPPER_OPEN),
            ignore=tuple(entry.get("ignore", ())),
            joint=entry.get("joint", ""),
            sweep=np.radians(float(entry.get("sweep", 0.0))),
            yaw=np.radians(float(entry.get("yaw", 0.0))),
            check_grasp=entry.get("grasp", ""),
        ))
    return out


def subgoal_to_dict(sg: Subgoal) -> dict:
    """The inverse of subgoals_from_json, so an edited plan can be written back."""
    out = {"name": sg.name, "kind": sg.kind}
    t = sg.target
    if t is not None:
        if t.frame == "world":
            out["pos"] = [round(float(v), 4) for v in t.offset]
        else:
            if t.frame != "eef":
                out["at"] = t.name
            if np.any(t.offset):
                out["offset"] = [round(float(v), 4) for v in t.offset]
            for flag in ("on_top", "carry", "seat", "hold_xy"):
                if getattr(t, flag):
                    out[flag] = True
    if sg.gripper is not None:
        out["gripper"] = "close" if sg.gripper > 0 else "open"
    if sg.check_grasp:
        out["grasp"] = sg.check_grasp
    if abs(sg.yaw) > 1e-9:
        out["yaw"] = round(float(np.degrees(sg.yaw)), 1)
    if sg.ignore:
        out["ignore"] = list(sg.ignore)
    if sg.joint:
        out["joint"] = sg.joint
        out["sweep"] = round(float(np.degrees(sg.sweep)), 1)
    return out


# ===================================================================== console

SUITES = ["libero_goal", "libero_10", "libero_object", "libero_spatial", "libero_90"]


class Abort(Exception):
    """The user asked to stop part-way through a plan."""


def wait_for_key(tick=None, prompt="") -> str:
    """Block for one keypress, running `tick` meanwhile.

    The viewer only redraws when something calls render(), so a plain input()
    would leave a frozen window for as long as the user took to read the line.
    """
    if prompt:
        print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        return input()[:1] or " "
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if select.select([sys.stdin], [], [], 0.03)[0]:
                return sys.stdin.read(1)
            if tick is not None:
                tick()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def choose_task(args: Args):
    """Print every task in every suite and let the user pick one."""
    rows = []
    for suite_name in SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        print(f"\n{suite_name}")
        for i in range(suite.n_tasks):
            print(f"  {len(rows):3d}  {suite.get_task(i).language}")
            rows.append((suite_name, i))

    while True:
        raw = input(f"\ntask number [0-{len(rows) - 1}], or q to quit: ").strip()
        if raw[:1].lower() == "q":
            raise SystemExit
        if raw.isdigit() and int(raw) < len(rows):
            return rows[int(raw)]
        print("  not a task number")


def build_env(args: Args, task, camera_observations=True):
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kwargs = dict(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
        controller="JOINT_POSITION",
        use_camera_obs=camera_observations,
        # robosuite refuses to step a finished episode, and LIBERO marks one
        # finished the moment the goal holds. A multi-step plan has to keep
        # moving afterwards -- to close the drawer it just filled, say.
        ignore_done=True,
        horizon=args.horizon,
    )
    env = (
        ControlEnv(has_renderer=False, has_offscreen_renderer=False, **kwargs)
        if not camera_observations and not args.render else
        ControlEnv(has_renderer=True, has_offscreen_renderer=True, **kwargs)
        if args.render
        else OffScreenRenderEnv(**kwargs)
    )
    env.seed(args.seed)
    env.reset()
    return env


def main(args: Args) -> None:
    np.random.seed(args.seed)
    if args.pick:
        args.task_suite_name, args.task_id = choose_task(args)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    env = build_env(args, task)
    world = World(env)

    print(f"\ntask [{args.task_suite_name}:{args.task_id}] {task.language}")
    print(f"bddl goal: {world.domain.parsed_problem['goal_state']}\n")

    if args.subgoals:
        spec = json.loads(pathlib.Path(args.subgoals).read_text())
        steps = [Step(predicate=[f"run {args.subgoals}"],
                      subgoals=subgoals_from_json(world, spec),
                      verify=(lambda: bool(env.check_success())))]
    else:
        steps = decompose(world, args)

    print(f"{len(steps)} steps, {sum(len(s.subgoals) for s in steps)} subgoals\n")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step.label()}")
        for sg in step.subgoals:
            print(f"       {sg.describe()}")
    print()

    if args.dry_run:
        env.close()
        return

    ex = Executor(env, world, args)
    ex.hold(args.settle_steps)
    if args.step:
        print("stepping: space runs the next subgoal, c runs the rest, q quits\n")

    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step.label()}")
        try:
            reached = ex.run_step(step)
        except Abort:
            print("\n  stopped by user")
            break
        satisfied = step.verify() if step.verify else None
        print(f"     -> motion {'ok' if reached else 'FAILED'}, predicate {satisfied}\n")
        if not satisfied and args.stop_on_failure:
            print("  stopping: --stop-on-failure")
            break

    print(f"task success: {env.check_success()}")
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
