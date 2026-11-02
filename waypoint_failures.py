"""Deliberate failures for waypoint replay: what goes wrong, and exactly where.

The studio arms at most one failure per waypoint before *Run all waypoints*.
Faults change the real controller commands. Low grasp loss prevents acquisition;
high grasp loss verifies a real drop during transport. Move rows never fail.
A collision is a detour before the original target, whose arrival is still
checked. Low bumps sideways; High approaches from behind the obstacle and
commands a 12 cm stroke along the gripper's measured forward axis. An
orientation deviation changes roll, pitch and yaw together, with each angle
drawn independently from that intensity's band. Transition deviations offset
the target in a random horizontal X/Y direction, with distance drawn from the
intensity's band. Arrival corrections
are suppressed for deliberate pose deviations; grasp and task checks still expose
the consequences. Faults that could not run are listed separately.
"""
import numpy as np
from scipy.spatial.transform import Rotation

import subgoal_planner as sp
from waypoint_primitives import PRIMITIVES

FAILURES = {
    "collision": "Collision",
    "transition": "Transition deviation",
    "orientation": "Orientation deviation",
    "grasp_loss": "Grasp loss",
}

HINTS = {
    "collision": "Low: a light sideways bump; High: a 12 cm stroke into the obstacle in the gripper's facing direction",
    "transition": "A random 2D X/Y direction; Low: 5 to 15 cm off target; High: 15 to 25 cm off target",
    "orientation": "Roll, pitch and yaw change together; each angle is random: Low 10 to 20 degrees, High 20 to 40 degrees",
    "grasp_loss": "Low: never grasp the object in Grasp; High: drop it midway through Transport",
}

INTENSITIES = {"low": "Low", "high": "High"}


def failure_spec(value):
    """Accept legacy failure names as high intensity, and explicit selections."""
    if isinstance(value, str):
        kind, intensity = value, "high"
    elif isinstance(value, dict):
        kind, intensity = value.get("kind"), value.get("intensity", "high")
    else:
        raise ValueError("Choose a failure and Low or High intensity.")
    if not isinstance(kind, str) or kind not in FAILURES:
        raise ValueError("Unknown injected failure: " + str(kind))
    if not isinstance(intensity, str) or intensity not in INTENSITIES:
        raise ValueError("Unknown failure intensity: " + str(intensity))
    return kind, intensity


def failure_label(value):
    kind, intensity = failure_spec(value)
    return FAILURES[kind] + (" / " + INTENSITIES[intensity] if isinstance(value, dict) else "")


def waypoint_primitive(waypoint):
    """Prefer learned phase metadata; infer the phase for manual/older plans."""
    learned = getattr(waypoint, "learned", {}) or {}
    role = (getattr(waypoint, "skill_role", "") or learned.get("primitive")
            or learned.get("node", {}).get("primitive"))
    if role:
        return str(role).lower()
    name = waypoint.name.lower()
    if name in PRIMITIVES:
        return name
    if waypoint.gripper_after == "close" or waypoint.gripper == "close" or waypoint.grasp:
        return "grasp"
    if "release" in name or waypoint.motion == "grip" and waypoint.gripper == "open":
        return "release"
    if "place" in name or "lower" in name:
        return "place"
    if waypoint.carry or "transport" in name or "transfer" in name:
        return "transport"
    if "grasp" in name or "descend" in name:
        return "grasp"
    if waypoint.motion == "arc" or "push" in name:
        return "push"
    return "move"


def allowed_intensities(kind, primitive, motion=None):
    """The same primitive rules are used by the picker and the controller."""
    if primitive == "move":
        return ()
    if kind == "grasp_loss":
        if primitive == "transport" and motion not in (None, "line", "reach"):
            return ()
        return {"grasp": ("low",), "transport": ("high",)}.get(primitive, ())
    if primitive == "transport" or motion not in (None, "line", "reach"):
        return ()
    return tuple(INTENSITIES)


PENETRATION = .03
"""How far past an obstacle's own middle a collision drives, in metres."""

HIGH_COLLISION_OFFSET = .12
"""High collision stroke along the gripper's forward (local +Z) axis."""

OPEN_STEPS = 40
"""Sim steps spent waiting, fingers wide, for the dropped object to fall clear."""

UNSETTLED = "failed: robot did not settle at the target"
"""The executor's own arrival verdict, spelled the way the executor spells it."""

TILT_BANDS = {"low": (10., 20.), "high": (20., 40.)}
"""Degrees drawn independently for roll, pitch and yaw at each intensity."""

TRANSITION_BANDS = {"low": (.05, .15), "high": (.15, .25)}
"""Horizontal transition target-offset ranges, in metres."""

TILT_AXES = {"x": "roll", "y": "pitch", "z": "yaw"}
"""Names of the wrist's roll, pitch and yaw axes."""


def label(name):
    return (name or "").replace("_", " ")


def collision_target(world, start, goal, ignore=(), penetration=PENETRATION):
    """Where to drive so the arm hits the body nearest the route to ``goal``.

    The target is the far side of that body's middle, approached level from
    where the wrist already is. Both halves of that matter. Descending onto a
    narrow object from above leaves it standing between open fingers, touching
    nothing at all, so the stroke has to be sideways; and stopping at the near
    surface leaves the fingertips at the object with the gap between them
    still clear of it, so it has to go past the middle.

    Returns the body hit and the target, or ``(None, ...)`` with a target below
    the goal when the scene offers nothing to hit.
    """
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    leg = goal - start
    span = float(leg @ leg)
    best = None
    for name in list(world.domain.objects_dict) + list(world.domain.fixtures_dict):
        if name in ignore:
            continue
        low, high = world.aabb(name)
        centre, half = (low + high) / 2, (high - low) / 2
        if not np.all(np.isfinite(centre)) or float(np.max(half)) < 1e-4:
            continue
        along = 0. if span < 1e-12 else float(np.clip((centre - start) @ leg / span, 0, 1))
        on_route = start + leg * along
        gap = float(np.linalg.norm(centre - on_route))
        if best is None or gap < best[0]:
            best = (gap, name, centre, half, on_route)
    if best is None:
        return None, np.array([goal[0], goal[1], goal[2] - 4 * penetration])
    _, name, centre, _, _ = best
    approach = np.array([centre[0] - start[0], centre[1] - start[1], 0.])
    distance = float(np.linalg.norm(approach))
    approach = np.array([1., 0., 0.]) if distance < 1e-6 else approach / distance
    return name, centre + approach * penetration


def held_object(world):
    """Which movable object is actually between the fingers right now.

    The executor tracks what it *meant* to pick up, and only when a subgoal
    told it to check; the skill executor never does. What a dropped grasp is
    about is what is really in the hand, so that is what is asked.
    """
    domain = world.domain
    gripper = domain.robots[0].gripper
    for name, item in domain.objects_dict.items():
        if domain._check_grasp(gripper, item.contact_geoms):
            return name
    return ""


def tilt(orientation, angle, axis="x"):
    """Apply a local wrist rotation, combining Euler angles when axis is xyz."""
    return np.asarray(orientation, float) @ Rotation.from_euler(axis, angle).as_matrix()


def split_path(path):
    """Split a joint route at half its travelled distance, retaining the join."""
    path = np.asarray(path, float)
    if len(path) < 2:
        return list(path), list(path)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance = np.r_[0., np.cumsum(lengths)]
    half = distance[-1] / 2
    index = min(int(np.searchsorted(distance, half, side="right")), len(path)-1)
    fraction = (half-distance[index-1])/lengths[index-1] if lengths[index-1] else 0.
    midpoint = path[index-1] + fraction*(path[index]-path[index-1])
    return list(path[:index]) + [midpoint], [midpoint] + list(path[index:])


class Injector:
    """Arms one deliberate failure per plan row for the length of one run.

    Every move a plan makes leaves the executor through ``line``, ``reach`` or
    ``follow``, whichever route the skill executor picked for that row, so
    patching those three for the duration of a ``with`` block catches a row's
    move whatever that row happens to be doing. A fault fires once per row:
    a move that is retried, or a planner that falls back to a straight line,
    must not fail twice for one arming.
    """

    METHODS = ("line", "reach", "follow", "run_subgoal")

    def __init__(self, session, injections, offset=None, angle=None, waypoints=None, seed=None):
        selections = {int(row): value for row, value in dict(injections).items()}
        specs = {row: failure_spec(value) for row, value in selections.items()}
        if offset is not None and not .005 <= float(offset) <= .5:
            raise ValueError("Distance off target must be between 0.5 and 50 cm.")
        if angle is not None and not np.radians(2) <= float(angle) <= np.radians(90):
            raise ValueError("Wrist rotation must be between 2 and 90 degrees.")
        self.session = session
        self.plan = {row: spec[0] for row, spec in specs.items()}
        self.intensities = {row: spec[1] for row, spec in specs.items()}
        self.labels = {row: failure_label(value).lower() for row, value in selections.items()}
        self.primitives = (None if waypoints is None else
                           [waypoint_primitive(w) for w in waypoints])
        self.motions = None if waypoints is None else [w.motion for w in waypoints]
        self.offset = None if offset is None else float(offset)
        self.angle = None if angle is None else float(angle)
        """A fixed wrist deviation, or ``None`` to draw one from the intensity band."""
        self.random = np.random.default_rng(seed)
        self.tilts = {}
        self.directions = {}
        self.distances = {}
        self.missed = []
        self.visited = set()
        self.consumed = set()
        self.dropped_row = None
        self.no_grasp_rows = set()
        self.fired = []
        """What actually happened, one line per fault, in the order they fired."""
        self.row = self.armed = None
        self.intensity = "high"
        self.primitive = None
        self.saved = {}

    # --- arming -----------------------------------------------------------
    def permitted_intensities(self, kind, primitive, motion=None):
        return allowed_intensities(kind, primitive, motion)

    def start_row(self, row, primitive=None, motion=None):
        """Called as each plan row starts, before the executor is asked to move."""
        if row != self.row:
            self.finish_row()
            self.row = row
            self.visited.add(row)
            self.armed = self.plan.get(row) if row not in self.consumed else None
            self.intensity = self.intensities.get(row, "high")
            self.session.injected_failure = ""
            if primitive is None and self.primitives is not None:
                primitive = self.primitives[row] if 0 <= row < len(self.primitives) else "move"
            if motion is None and self.motions is not None and 0 <= row < len(self.motions):
                motion = self.motions[row]
            self.primitive = primitive
            if (self.armed and primitive is not None
                    and self.intensity not in self.permitted_intensities(self.armed, primitive, motion)):
                self.missed.append("waypoint %d %s (not allowed in %s)" %
                                   (row + 1, self.labels[row], primitive.title()))
                self.take()

    def finish_row(self):
        if self.armed:
            reason = "this step did not execute a supported move"
            if self.armed == "grasp_loss":
                reason = ("no close command was executed" if self.intensity == "low" else
                          "no object was grasped during Transport")
            self.missed.append("waypoint %d %s (%s)" %
                               (self.row + 1, self.labels[self.row], reason))
            self.take()

    def _delivered(self, kind, result):
        """Report a row that has just carried out its fault, and stop checking it.

        The row was told to miss its target, so the plan's arrival tolerance has
        nothing useful left to say about where the wrist ended up, and the small
        corrective retry it makes first would quietly undo the fault. A move
        that never happened at all -- an unreachable pose, no route to it -- is
        still the executor's own news, and is passed straight through.
        """
        if not (result.startswith("ok") or result == UNSETTLED):
            return result
        self.session.injected_failure = "ok (injected %s)" % self.labels.get(self.row, FAILURES[kind].lower())
        return self.session.injected_failure

    def take(self):
        kind, self.armed = self.armed, None
        if kind:
            self.consumed.add(self.row)
        return kind

    def take_motion(self):
        # Low prevents the upcoming close. High waits until something is held
        # so the fault releases an acquired object during the carrying motion.
        if self.row in self.no_grasp_rows or (self.row is not None and self.dropped_row == self.row):
            self.session.executor.gripper = sp.GRIPPER_OPEN
        if self.armed == "grasp_loss" and (self.intensity == "low" or not held_object(self.session.world)):
            return None
        return self.take()

    def record(self, kind, detail):
        self.fired.append("waypoint %d %s (%s)"
                          % ((self.row or 0) + 1, self.labels.get(self.row, FAILURES[kind].lower()), detail))

    def pose_result(self, kind, result, detail):
        """A pose rejected before execution did not deliver an injected fault."""
        if result.startswith(("unreachable", "no ")):
            self.missed.append("waypoint %d %s (%s)" %
                               (self.row + 1, self.labels[self.row], result))
        else:
            self.record(kind, detail)
        return self._delivered(kind, result)

    def __enter__(self):
        ex = self.session.executor
        self.session.injected_failure = ""
        self.saved = {name: getattr(ex, name) for name in self.METHODS}
        ex.line, ex.reach, ex.follow = self._line, self._reach, self._follow
        ex.run_subgoal = self._run_subgoal
        return self

    def __exit__(self, *_):
        ex = self.session.executor
        self.finish_row()
        for row in sorted(set(self.plan) - self.visited):
            self.missed.append("waypoint %d %s (step was not reached)" %
                               (row + 1, self.labels[row]))
        self.session.injected_failure = ""
        for name, method in self.saved.items():
            # Restore the class method, not a leftover instance attribute.
            if getattr(type(ex), name, None) is getattr(method, "__func__", None):
                delattr(ex, name)
            else:
                setattr(ex, name, method)
        self.saved = {}
        return False

    # --- faults -----------------------------------------------------------
    def transition_distance(self):
        # Sample magnitude and direction together, so observers and controllers
        # get the same seeded offset regardless of which accessor runs first.
        self.transition_direction()
        return self.distances[self.row]

    def wrist_tilt(self):
        """This row's combined, independently drawn roll, pitch and yaw deviation.

        Drawn once per row and remembered, because the
        commanded pose and anything watching for it must mean the same
        deviation. An explicit ``angle`` keeps the old fixed roll instead.
        """
        drawn = self.tilts.get(self.row)
        if drawn is None:
            if self.angle is not None:
                drawn = ("x", self.angle * (.25 if self.intensity == "low" else 1.))
            else:
                low, high = TILT_BANDS[self.intensity]
                drawn = ("xyz", tuple(np.radians(self.random.uniform(low, high, size=3))))
            self.tilts[self.row] = drawn
        return drawn

    def transition_direction(self):
        """Draw one horizontal direction and distance per waypoint."""
        drawn = self.directions.get(self.row)
        if drawn is None:
            if self.offset is None:
                low, high = TRANSITION_BANDS[self.intensity]
                distance = float(self.random.uniform(low, high))
            else:
                # Preserve explicitly requested fixed distances for legacy callers.
                distance = self.offset * (.4 if self.intensity == "low" else 1.)
            angle = float(self.random.uniform(0., 2*np.pi))
            drawn = (float(np.cos(angle)), float(np.sin(angle)), 0.)
            self.directions[self.row] = drawn
            self.distances[self.row] = distance
        return np.asarray(drawn, float)

    def deviated(self, position):
        """Offset X/Y by the sampled distance while preserving target height."""
        return np.asarray(position, float) + self.transition_direction() * self.transition_distance()

    def collision_push_pose(self):
        """Push from the actual approach pose in the direction the fingers face.

        Read the measured frame after approaching: a blocked arm can differ from
        the requested waypoint in both position and orientation. Local +Z is
        forward through the fingers (downward for the normal grasp pose).
        """
        position, orientation = self.session.gripper_pose()
        orientation = np.asarray(orientation, float)
        return np.asarray(position, float) + orientation[:, 2] * HIGH_COLLISION_OFFSET, orientation

    def prevent_grasp(self):
        """Keep the fingers open before acquisition, including close retries."""
        self.take()
        ex = self.session.executor
        if held_object(self.session.world):
            self.missed.append("waypoint %d %s (object was already grasped)" %
                               (self.row + 1, self.labels[self.row]))
            return "failed: no-grasp injection requires an empty hand"
        self.no_grasp_rows.add(self.row)
        ex.gripper = sp.GRIPPER_OPEN
        ex.hold(30)
        ex.holding = ""
        ex.carry_offset, ex.carry_bottom = np.zeros(3), 0.
        if held_object(self.session.world):
            self.missed.append("waypoint %d %s (object acquired despite open fingers)" %
                               (self.row + 1, self.labels[self.row]))
            return "failed: no-grasp injection could not prevent acquisition"
        self.record("grasp_loss", "no grasp; fingers stayed open")
        return "failed: injected grasp loss / low; object was never grasped"

    def release(self):
        """Release a measured grasp and clear the executor's carried-object state."""
        ex = self.session.executor
        held = held_object(self.session.world)
        if not held:
            self.missed.append("waypoint %d grasp loss (nothing was held at release)"
                               % (self.row + 1))
            return False
        ex.gripper = sp.GRIPPER_OPEN
        ex.hold(OPEN_STEPS)
        if held_object(self.session.world):
            self.missed.append("waypoint %d grasp loss (object did not release)"
                               % (self.row + 1))
            return False
        ex.holding = ""
        ex.carry_offset, ex.carry_bottom = np.zeros(3), 0.
        self.dropped_row = self.row
        self.record("grasp_loss", "dropped the " + label(held))
        self._delivered("grasp_loss", "ok")
        return True

    def _run_subgoal(self, sg):
        """Catch the close after a grasp move, including gripper-only steps."""
        original = self.saved["run_subgoal"]
        if sg.kind != "grip":
            return original(sg)
        if sg.gripper == sp.GRIPPER_CLOSE:
            if self.row in self.no_grasp_rows:
                self.session.executor.gripper = sp.GRIPPER_OPEN
                return "failed: injected grasp loss / low; gripper remains open"
            if self.armed == "grasp_loss" and self.intensity == "low":
                return self.prevent_grasp()
        if self.row is not None and self.dropped_row == self.row and sg.gripper == sp.GRIPPER_CLOSE:
            # A retry must not silently recover the grasp in the same faulted row.
            self.session.executor.gripper = sp.GRIPPER_OPEN
            return "failed: injected grasp loss; gripper remains open"
        result = original(sg)
        if (self.armed == "grasp_loss" and self.intensity == "high"
                and self.primitive is None and held_object(self.session.world)):
            self.take()
            if self.release():
                # The object was acquired and deliberately lost. Stop grasp
                # retries here instead of letting the skill quietly pick it up again.
                return "failed: injected grasp loss; gripper released the object"
        return result

    def _drop_through(self, original, position, orientation, args, kwargs):
        """Make the same move in two halves, letting go in the middle of it.

        Dropping before the move starts would read as a release at the previous
        waypoint. What a lost grasp looks like is an arm that carries on to
        where it was going with nothing in its hand.
        """
        here = self.session.world.eef()
        result = original(here + (position - here) * .5, orientation, *args, **kwargs)
        if not (result.startswith("ok") or result == UNSETTLED):
            self.missed.append("waypoint %d grasp loss (%s)" % (self.row + 1, result))
            return result
        if not self.release():
            return "failed: grasp loss could not release an object"
        return original(position, orientation, *args, **kwargs)

    def _faulty(self, kind, original, position, orientation, args=(), kwargs=None):
        """Run one move with the armed fault in place, and say what it did."""
        kwargs = kwargs or {}
        position = np.asarray(position, float)
        orientation = sp.DOWN if orientation is None else np.asarray(orientation, float)
        if kind == "collision":
            name, target = collision_target(
                self.session.world, self.session.world.eef(), position,
                ignore=(held_object(self.session.world),),
                penetration=PENETRATION / 3 if self.intensity == "low" else PENETRATION)
            # Insert the contact before the requested move; the collision
            # target must never replace the waypoint or bypass its arrival check.
            if self.intensity == "high":
                # Starting already pressed against the obstacle hides the
                # difference from Low. Stage behind it, then drive forward
                # through its middle instead of attempting a blocked shove.
                target = target - orientation[:, 2] * (HIGH_COLLISION_OFFSET - PENETRATION)
            contact = self.saved["line"](target, orientation)
            if contact.startswith("ok") or contact == UNSETTLED:
                detail = "drove into the " + label(name) if name else "drove down into the surface below"
                if self.intensity == "high":
                    push_position, push_orientation = self.collision_push_pose()
                    push = self.saved["line"](push_position, push_orientation)
                    if push.startswith("ok") or push == UNSETTLED:
                        detail += "; commanded a %.0f cm stroke in the gripper's facing direction" % (HIGH_COLLISION_OFFSET * 100)
                    else:
                        self.missed.append("waypoint %d collision / high forward push (%s)" %
                                           (self.row + 1, push))
                        detail = None
                if detail is not None:
                    self.record("collision", detail)
            else:
                self.missed.append("waypoint %d collision (%s)" % (self.row + 1, contact))
            result = original(position, orientation, *args, **kwargs)
            if original == self.saved["reach"] and result == "no collision-free path":
                # Contact can make the RRT start invalid. Leave contact using
                # the same Cartesian controller that produced the bump.
                result = self.saved["line"](position, orientation)
            return result
        if kind == "transition":
            distance = self.transition_distance()
            changed = self.deviated(position)
            result = original(changed, orientation, *args, **kwargs)
            delta = (changed - position) * 100
            return self.pose_result(kind, result,
                "%.1f cm off target in 2D X/Y; X %+.1f cm, Y %+.1f cm, Z %+.1f cm"
                % (distance * 100, *delta))
        if kind == "orientation":
            axis, angle = self.wrist_tilt()
            result = original(position, tilt(orientation, angle, axis), *args, **kwargs)
            if (self.angle is not None and result.startswith("unreachable")
                    and angle > .12 and self.intensity == "low"
                    and not getattr(self.session, "continue_on_failure", False)):
                # Keep legacy fixed-angle retries; random angles stay in their band.
                angle = angle / 2
                result = original(position, tilt(orientation, angle, axis), *args, **kwargs)
            detail = ", ".join("%s %.0f degrees" % (TILT_AXES[a], np.degrees(value))
                               for a, value in zip(axis, np.atleast_1d(angle)))
            return self.pose_result(kind, result, "wrist tilted " + detail)
        return self._delivered(kind, self._drop_through(
            original, position, orientation, args, kwargs))

    # --- patched executor entry points ------------------------------------
    def _line(self, pos, ori=None, *args, **kwargs):
        kind = self.take_motion()
        original = self.saved["line"]
        if kind is None:
            return original(pos, ori, *args, **kwargs)
        return self._faulty(kind, original, pos, ori, args, kwargs)

    def _reach(self, pos, ori=None):
        kind = self.take_motion()
        original = self.saved["reach"]
        if kind is None:
            return original(pos, ori)
        return self._faulty(kind, original, pos, ori)

    def _follow(self, path, *args, **kwargs):
        kind = self.take_motion()
        original = self.saved["follow"]
        path = [np.asarray(q, float) for q in path]
        if kind is None:
            return original(path, *args, **kwargs)
        if kind == "grasp_loss":
            # RRT commonly returns just start and goal. Split the distance,
            # not the list length, so even that route drops halfway through.
            current = self.session.executor.scene.arm_qpos().copy()
            path = [current] + path[1:]
            first, rest = split_path(path)
            if not original(first, *args, **kwargs):
                self.missed.append("waypoint %d grasp loss (move stopped before release)"
                                   % (self.row + 1))
                return False
            if not self.release():
                return False
            moved = original(rest, *args, **kwargs)
            self._delivered(kind, "ok" if moved else UNSETTLED)
            return True
        # Resolve the original joint-route endpoint before the fault moves
        # the robot, so the requested waypoint is preserved.
        position, orientation = self.session.executor.scene.eef_frame(path[-1])
        # After a collision the old joint route starts in the wrong place.
        # Resume toward its original endpoint from the robot's current pose.
        limits = (kwargs.get("limits") or (args[2] if len(args) > 2 else None)) if kind == "collision" else None
        return self._faulty(kind, self.saved["line"], position, orientation,
                            args=limits or ()).startswith("ok")
