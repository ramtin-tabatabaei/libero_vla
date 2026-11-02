"""Timestamp measured task alerts without changing the VLA or simulator state."""
from dataclasses import asdict, dataclass

import numpy as np


EVENT_LABELS = dict(wrong_object="Wrong object picked up", collision="Scene collision",
                    off_plan="Far from waypoint plan", unfinished="Task not finished")


@dataclass
class FailureConfig:
    collision_force_n: float = 10.
    off_plan_metres: float = .30
    off_plan_seconds: float = 1.
    pickup_lift_metres: float = .02
    pickup_hold_seconds: float = .15
    expected_object: str = "automatic"
    collision_quiet_seconds: float = .1

    def validate(self):
        values = (self.collision_force_n, self.off_plan_metres, self.off_plan_seconds,
                  self.pickup_lift_metres, self.pickup_hold_seconds, self.collision_quiet_seconds)
        if not np.isfinite(values).all() or any(value <= 0 for value in values):
            raise ValueError("Failure force, distance, lift and confirmation times must be positive finite numbers.")
        return self


def goal_atoms(value):
    if not isinstance(value, (list, tuple)) or not value:
        return
    if isinstance(value[0], str) and all(isinstance(part, str) for part in value):
        yield value
    else:
        for part in value:
            yield from goal_atoms(part)


def task_intent(goals, chain, movable, override="automatic"):
    """Use task goal subjects for pickup identity, never destination arguments."""
    movable = set(movable)
    atoms = list(goal_atoms(goals))
    contacts = {name for goal in atoms for name in goal[1:]}
    contacts.update(w.reference for w in chain.waypoints if w.motion in ("line", "grip", "arc"))
    contacts.update(w.grasp for w in chain.waypoints if w.grasp)
    if override == "none":
        return set(), contacts, "No object should be picked up (manual)"
    if override != "automatic":
        if override not in movable:
            raise ValueError("The expected pickup object is not present in this scene: " + override)
        return {override}, contacts | {override}, "Manual selection"
    placement = {g[1] for g in atoms if len(g) >= 3 and g[0].lower() in ("in", "on") and g[1] in movable}
    if placement:
        return placement, contacts | placement, "Task goal subjects"
    if atoms and all(g[0].lower() in ("open", "close", "turnon", "turnoff") for g in atoms):
        return set(), contacts, "Fixture-only task; no movable-object pickup expected"
    sources = {w.grasp for w in chain.waypoints if w.grasp in movable}
    skill = getattr(chain, "execution_skill", {})
    def skill_sources(item):
        if item.get("source") in movable:
            sources.add(item["source"])
        for stage in item.get("stages", []):
            skill_sources(stage)
    skill_sources(skill)
    for w in chain.waypoints:
        if (w.gripper == "close" or w.gripper_after == "close") and w.reference in movable:
            sources.add(w.reference)
    if sources:
        return sources, contacts | sources, "Saved waypoint grasp targets"
    return None, contacts, "Cannot infer pickup identity; select an expected object in Settings"


def distance_to_route(position, route):
    """Distance to the whole polyline, including segments between checkpoints."""
    if len(route) < 2:
        return None
    points = np.asarray(route, dtype=float)
    start, delta = points[:-1], np.diff(points, axis=0)
    squared = np.einsum("ij,ij->i", delta, delta)
    along = np.clip(np.einsum("ij,ij->i", np.asarray(position)-start, delta)
                    / np.maximum(squared, 1e-15), 0., 1.)
    return float(np.linalg.norm(np.asarray(position)-(start+along[:, None]*delta), axis=1).min())


class FailureTimeline:
    """Retain event onsets and confirmations, grouping continuous alert samples."""
    def __init__(self, config, expected, expected_source, route, wall_seconds=None):
        self.config = config.validate()
        self.expected, self.expected_source = expected, expected_source
        self.route = [np.asarray(p).copy() for p in route]
        self.events = {}
        self.history = []
        self.pending_wrong = {}
        self.wrong_active = set()
        self.pending_off = None
        self.off_active = False
        self.collision_active = False
        self.last_collision_contact = None
        self.assessment = "running"
        self.wall_seconds = wall_seconds

    def add(self, kind, onset, confirmed, step, detail, **evidence):
        event = dict(kind=kind, event=EVENT_LABELS[kind],
            onset_sim_seconds=onset, confirmed_sim_seconds=float(confirmed),
            confirmed_wall_seconds=float(self.wall_seconds()) if self.wall_seconds is not None else None,
            control_step=int(step), detail=detail, evidence=evidence)
        self.events.setdefault(kind, event)
        self.history.append(event)

    def collision(self, seconds, step, force_n, robot_geom, scene_geom):
        if force_n >= self.config.collision_force_n:
            if not self.collision_active:
                self.add("collision", None if step == 0 else float(seconds), seconds, step,
                    f"{robot_geom} hit {scene_geom}: {force_n:.1f} N",
                    force_n=float(force_n), robot_geom=robot_geom, scene_geom=scene_geom,
                    already_present_at_start=step == 0)
            self.collision_active = True
            self.last_collision_contact = float(seconds)
        elif (self.collision_active and seconds-self.last_collision_contact + 1e-9
              >= self.config.collision_quiet_seconds):
            self.collision_active = False

    def update(self, seconds, step, position, grasped, lifts):
        if step == 0:
            return
        if self.expected is not None:
            wrong = {name for name in grasped if name not in self.expected
                     and lifts.get(name, 0.) >= self.config.pickup_lift_metres}
            self.pending_wrong = {name: time for name, time in self.pending_wrong.items() if name in wrong}
            self.wrong_active.intersection_update(wrong)
            for name in wrong:
                self.pending_wrong.setdefault(name, float(seconds))
            for name in sorted(wrong, key=lambda value: (self.pending_wrong[value], value)):
                onset = self.pending_wrong[name]
                if name not in self.wrong_active and seconds-onset + 1e-9 >= self.config.pickup_hold_seconds:
                    allowed = ", ".join(sorted(self.expected)) or "no object"
                    self.add("wrong_object", onset, seconds, step,
                        f"Lifted {name} by {100*lifts[name]:.1f} cm; expected {allowed}",
                        object=name, expected=sorted(self.expected), lift_metres=float(lifts[name]))
                    self.wrong_active.add(name)
        distance = distance_to_route(position, self.route)
        if distance is not None:
            if distance > self.config.off_plan_metres:
                if self.pending_off is None:
                    self.pending_off = float(seconds)
                if not self.off_active and seconds-self.pending_off + 1e-9 >= self.config.off_plan_seconds:
                    self.add("off_plan", self.pending_off, seconds, step,
                        f"Stayed over {100*self.config.off_plan_metres:g} cm from the plan for "
                        f"{self.config.off_plan_seconds:g} s; now {100*distance:.1f} cm away",
                        distance_metres=distance)
                    self.off_active = True
            else:
                self.pending_off = None
                self.off_active = False

    def finish(self, seconds, step, state, success, end_reason):
        self.assessment = ("interrupted; completion unassessed" if state == "stopped" else
                           "policy error; completion unassessed" if state == "error" else
                           "task succeeded" if success is True else
                           "step limit exhausted" if end_reason == "step_limit" else
                           "completion unassessed")
        if state == "finished" and success is False and end_reason == "step_limit" and step:
            self.add("unfinished", float(seconds), seconds, step,
                     f"Goal still false when the policy step limit ended ({step} observed actions)")

    def report(self):
        events = sorted(self.history, key=lambda e: (e["confirmed_sim_seconds"]
                        if e["onset_sim_seconds"] is None else e["onset_sim_seconds"]))
        return dict(events=events, expected_objects=None if self.expected is None else sorted(self.expected),
            expected_source=self.expected_source, config=asdict(self.config), assessment=self.assessment,
            off_plan_available=len(self.route) >= 2,
            timing="Simulation seconds since VLA start; onset and confirmation are separate. "
                   "A blank onset means contact was already present at start.",
            rules="Pickup requires bilateral grasp and lift above its starting height. "
                  "Collision is measured robot/scene force excluding intended finger contacts. "
                  "Off-plan is a sustained distance proxy, not proof of task failure. "
                  "Repeated samples of the same active alert are grouped into episodes.")


class SimulationFailureObserver:
    """Read grasps at control rate and contact forces at every physics step."""
    def __init__(self, session, chain, targets, start, config, control_step, wall_seconds=None):
        import mujoco
        self.mujoco = mujoco
        self.session, self.start, self.control_step = session, start, control_step
        self.domain, self.sim = session.env.env, session.env.sim
        self.world = session.world
        self.robot = self.domain.robots[0]
        self.initial_z = {name: float(self.world.pos(name)[2]) for name in self.domain.objects_dict}
        expected, permitted, source = task_intent(self.domain.parsed_problem.get("goal_state", []),
                                                chain, self.initial_z, config.expected_object)
        position, _ = session.gripper_pose()
        self.timeline = FailureTimeline(config, expected, source,
            [position] + [p for p, _ in targets] if targets else [], wall_seconds)
        m = self.world.m
        own = set(self.robot.robot_model.bodies) | set(self.robot.gripper.bodies)
        self.robot_geoms = {g for g in range(m.ngeom)
                            if self.sim.model.body_id2name(m.geom_bodyid[g]) in own}
        fingers = self.robot.gripper.important_geoms
        self.finger_geoms = self.geom_ids(fingers.get("left_finger", fingers["left_fingerpad"])
                                        + fingers.get("right_finger", fingers["right_fingerpad"]))
        self.permitted_geoms = set()
        for name in permitted:
            if name in self.domain.objects_dict:
                self.permitted_geoms.update(self.geom_ids(self.domain.objects_dict[name].contact_geoms))
            elif name in self.domain.object_sites_dict or name in self.domain.fixtures_dict:
                self.permitted_geoms.update(self.world._geoms_of(self.world.parent_body(name)))
            elif name and name in session.reference_poses:
                # Joint-frame waypoints name the moving fixture link.
                try:
                    body = self.world.joint_info(name)["body"]
                except (KeyError, ValueError):
                    continue
                descendants = {int(body)}
                for b in range(int(body)+1, m.nbody):
                    if int(m.body_parentid[b]) in descendants:
                        descendants.add(b)
                self.permitted_geoms.update(g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in descendants)
        self.force = np.zeros(6)
        self.installed = False

    def geom_ids(self, names):
        result = set()
        for name in names:
            index = self.mujoco.mj_name2id(self.world.m, self.mujoco.mjtObj.mjOBJ_GEOM, name)
            if index >= 0:
                result.add(int(index))
        return result

    def capture_collision(self, step=None):
        data = self.world.d
        largest, names = 0., ("", "")
        for i in range(data.ncon):
            contact = data.contact[i]
            first, second = int(contact.geom1), int(contact.geom2)
            if (first in self.robot_geoms) == (second in self.robot_geoms):
                continue
            mine, other = (first, second) if first in self.robot_geoms else (second, first)
            if mine in self.finger_geoms and other in self.permitted_geoms:
                continue
            self.mujoco.mj_contactForce(self.world.m, data, i, self.force)
            magnitude = float(np.linalg.norm(self.force[:3]))
            if magnitude > largest:
                largest = magnitude
                names = (self.sim.model.geom_id2name(mine) or str(mine),
                         self.sim.model.geom_id2name(other) or str(other))
        self.timeline.collision(float(data.time)-self.start,
            self.control_step() if step is None else step, largest, *names)

    def install(self):
        self.original_step = self.sim.step
        self.had_step = "step" in vars(self.sim)
        self.capture_collision(step=0)
        def step(*args, **kwargs):
            result = self.original_step(*args, **kwargs)
            self.capture_collision()
            return result
        self.sim.step = step
        self.installed = True

    def capture(self, seconds, step, position):
        grasped = [name for name, item in self.domain.objects_dict.items()
                   if self.domain._check_grasp(self.robot.gripper, item.contact_geoms)]
        lifts = {name: float(self.world.pos(name)[2])-self.initial_z[name] for name in grasped}
        self.timeline.update(seconds, step, position, grasped, lifts)

    def close(self):
        if self.installed:
            if self.had_step:
                self.sim.step = self.original_step
            else:
                del self.sim.step
            self.installed = False
