"""Passive pose coverage of saved waypoint targets during a free VLA rollout."""
import copy
import csv
import json
import time

import numpy as np

from waypoint_chain import rotation, vector


class PoseCoverage:
    """Track visits in any order and the contiguous prefix reached in plan order."""
    def __init__(self, labels, targets, position_tolerance=.05, orientation_degrees=None):
        if not np.isfinite(position_tolerance) or position_tolerance <= 0:
            raise ValueError("Position tolerance must be a positive finite number.")
        if orientation_degrees is not None and (
                not np.isfinite(orientation_degrees) or not 0 < orientation_degrees <= 180):
            raise ValueError("Wrist angle tolerance must be greater than 0 and at most 180 degrees.")
        if not labels or len(labels) != len(targets):
            raise ValueError("Load a waypoint plan with a target for every row.")
        self.labels = list(labels)
        self.targets = [(vector(p).copy(), rotation(r).copy()) for p, r in targets]
        self.position_tolerance = float(position_tolerance)
        self.orientation_degrees = orientation_degrees
        self.closest = [None] * len(labels)
        self.angle_at_closest = [None] * len(labels)
        self.first_visit = [None] * len(labels)
        self.ordered_visit = [None] * len(labels)
        self.at_start = [False] * len(labels)
        self.prefix = self.steps = 0
        self.seconds = 0.

    def update(self, position, orientation, seconds, step):
        position, orientation = vector(position), rotation(orientation)
        if (not np.isfinite(seconds) or seconds < self.seconds or seconds < 0
                or not isinstance(step, int) or step < self.steps or step < 0):
            raise ValueError("Coverage samples must have increasing simulation time and step numbers.")
        self.seconds, self.steps = float(seconds), step
        near = []
        for i, (target, basis) in enumerate(self.targets):
            distance = float(np.linalg.norm(position - target))
            angle = float(np.degrees(np.arccos(np.clip((np.trace(basis.T @ orientation)-1)/2, -1, 1))))
            if (self.closest[i] is None or distance < self.closest[i] or
                    (np.isclose(distance, self.closest[i], atol=1e-12, rtol=0) and
                     angle < self.angle_at_closest[i])):
                self.closest[i], self.angle_at_closest[i] = distance, angle
            matches = distance <= self.position_tolerance + 1e-12 and (
                self.orientation_degrees is None or angle <= self.orientation_degrees + 1e-9)
            near.append(matches)
            if step == 0:
                self.at_start[i] = matches
            elif matches and self.first_visit[i] is None:
                self.first_visit[i] = self.seconds
        # A target near the starting pose alone is not evidence of a VLA visit.
        if step:
            while self.prefix < len(near) and near[self.prefix]:
                self.ordered_visit[self.prefix] = self.seconds
                self.prefix += 1

    def rows(self, finished=False):
        result = []
        for i, (label, (position, basis)) in enumerate(zip(self.labels, self.targets)):
            status = ("In plan order" if self.ordered_visit[i] is not None else
                      "Visited (other order)" if self.first_visit[i] is not None else
                      "At start only" if self.at_start[i] else
                      "Not visited" if finished else "Pending")
            result.append(dict(waypoint=i+1, name=label, status=status,
                closest_position_cm=None if self.closest[i] is None else 100*self.closest[i],
                angle_at_closest_deg=self.angle_at_closest[i],
                first_visit_sim_seconds=self.first_visit[i],
                ordered_visit_sim_seconds=self.ordered_visit[i],
                near_at_start=self.at_start[i], target_position=position.tolist(),
                target_orientation=basis.tolist()))
        return result


def frozen_targets(session, chain):
    """Resolve the plan against the current scene; retain the editor's references."""
    names = ("reference_poses", "start_pose", "reference_carry", "reference_bottom",
             "reference_joints", "previewed")
    missing = object()
    saved = {name: getattr(session, name, missing) for name in names}
    try:
        session.refresh_references()
        return [(np.asarray(p).copy(), np.asarray(r).copy()) for p, r in session.preview(chain)]
    finally:
        for name, value in saved.items():
            if value is missing:
                session.__dict__.pop(name, None)
            else:
                setattr(session, name, value)


class RolloutCoverage:
    """Observe every real control step without changing actions or adding steps."""
    def __init__(self, session, chain, backend, position_tolerance=.05,
                 orientation_degrees=None, on_update=None, failure_config=None,
                 check_waypoints=True):
        self.session, self.chain = session, copy.deepcopy(chain)
        self.backend = backend
        self.position_tolerance, self.orientation_degrees = position_tolerance, orientation_degrees
        self.on_update = on_update or (lambda monitor: None)
        self.tracker = None
        self.failure_config, self.check_waypoints = failure_config, check_waypoints
        self.failures = None
        self.env_wrapped = False
        self.steps = 0
        self.seconds = 0.
        self.end_reason = None
        self.state = "preparing"
        self.task_success = None

    def __enter__(self):
        try:
            targets = frozen_targets(self.session, self.chain) if self.chain.waypoints else []
            labels = [getattr(w, "skill_role", "") or w.name for w in self.chain.waypoints]
            if labels and self.check_waypoints:
                self.tracker = PoseCoverage(labels, targets, self.position_tolerance, self.orientation_degrees)
            env = self.session.env
            self.start = float(env.sim.data.time)
            self.start_wall = time.monotonic()
            self.original_step = env.step
            self.had_step = "step" in env.__dict__
            if self.failure_config is not None:
                from waypoint_task_events import SimulationFailureObserver
                self.failures = SimulationFailureObserver(self.session, self.chain, targets, self.start,
                    self.failure_config, lambda: self.steps+1, lambda: time.monotonic()-self.start_wall)
                self.failures.install()
            self.state = "running"
            p, r = self.session.gripper_pose()
            if self.tracker is not None:
                self.tracker.update(p, r, 0., 0)
            self.on_update(self)

            def step(*args, **kwargs):
                result = self.original_step(*args, **kwargs)
                p, r = self.session.gripper_pose()
                self.steps += 1
                self.seconds = float(env.sim.data.time)-self.start
                if self.tracker is not None:
                    self.tracker.update(p, r, self.seconds, self.steps)
                if self.failures is not None:
                    self.failures.capture(self.seconds, self.steps, p)
                self.on_update(self)
                return result

            env.step = step
            self.env_wrapped = True
            return self
        except BaseException:
            self.restore_hooks()
            self.state = "error"
            if self.failures is not None:
                self.failures.timeline.finish(self.seconds, self.steps, self.state, None, None)
            self.on_update(self)
            raise

    def restore_hooks(self):
        if self.env_wrapped:
            if self.had_step:
                self.session.env.step = self.original_step
            else:
                del self.session.env.step
            self.env_wrapped = False
        if self.failures is not None:
            self.failures.close()

    def set_policy_result(self, result):
        self.task_success = bool(result["success"])
        self.end_reason = "task_success" if self.task_success else "step_limit"

    def __exit__(self, exc_type, exc, traceback):
        self.restore_hooks()
        self.state = "finished" if exc_type is None else "stopped" if exc_type.__name__ == "Abort" else "error"
        if self.failures is not None:
            self.failures.timeline.finish(self.seconds, self.steps, self.state, self.task_success, self.end_reason)
        self.on_update(self)

    def report(self):
        tracker = self.tracker
        return dict(suite=self.chain.suite, task_id=self.chain.task_id,
            language=self.chain.language, backend=self.backend, run_state=self.state,
            task_success=self.task_success, position_tolerance_cm=100*self.position_tolerance,
            orientation_tolerance_degrees=self.orientation_degrees,
            target_frame="World poses resolved from the saved plan and scene at rollout start",
            measurement="End-effector pose proximity; does not verify grasp, release or object motion",
            steps=self.steps, simulation_seconds=self.seconds, checking_waypoints=self.check_waypoints,
            termination_reason=self.end_reason,
            visited=sum(t is not None for t in tracker.first_visit) if tracker else 0,
            ordered_prefix=tracker.prefix if tracker else 0,
            total=len(self.chain.waypoints),
            rows=tracker.rows(self.state == "finished") if tracker else [],
            failure_timeline=self.failures.timeline.report() if self.failures is not None else None)

    def save_csv(self, path):
        report = self.report()
        fields = ("waypoint", "name", "status", "closest_position_cm", "angle_at_closest_deg",
                  "first_visit_sim_seconds", "ordered_visit_sim_seconds", "near_at_start",
                  "target_position", "target_orientation", "suite", "task_id", "backend",
                  "run_state", "task_success", "position_tolerance_cm", "orientation_tolerance_degrees")
        with open(path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for row in report["rows"]:
                writer.writerow(dict(row, **{k: report[k] for k in fields if k in report}))

    def save_failures_csv(self, path):
        report = self.report()
        fields = ("kind", "event", "onset_sim_seconds", "confirmed_sim_seconds", "confirmed_wall_seconds", "control_step",
                  "detail", "evidence", "suite", "task_id", "backend", "run_state", "task_success",
                  "expected_objects", "expected_source", "assessment", "config")
        timeline = report["failure_timeline"]
        with open(path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for event in timeline["events"] if timeline else []:
                writer.writerow(dict(event,
                    **{key: report[key] for key in ("suite", "task_id", "backend", "run_state", "task_success")},
                    **{key: timeline[key] for key in ("expected_objects", "expected_source", "assessment", "config")}))

    def save_json(self, path):
        with open(path, "w") as stream:
            json.dump(self.report(), stream, indent=2, allow_nan=False)
