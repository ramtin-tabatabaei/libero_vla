"""Simulation adapter for the waypoint editor; reuses the existing IK/RRT executor."""
import dataclasses
from contextlib import contextmanager
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from libero.libero import benchmark
import subgoal_planner as sp
from waypoint_chain import Chain, Waypoint, WORLD, PREVIOUS, resolve, relative_waypoint


class Session:
    def __init__(self, args):
        self.args = dataclasses.replace(args, render=False)
        self.env = None
        self.tick = lambda: None
        self.realtime = False
        self.playback_rate = 1.0
        self._last_step = 0.
        self.continue_on_failure = False
        self.last_run_failures = []
        self.running_target = None
        """The pose the executor was actually given for the row now running.

        A plan row says where to go; the executor does not always go exactly
        there. A grasp adjusted for clearance, or a contact refreshed from an
        object that has moved since the run began, is a different pose, and it
        is the one a viewer watching the arm needs to see.
        """
        self.injected_failure = ""
        """Set while a waypoint is carrying out a deliberate failure.

        A row told to miss its target is not measured against it, so the
        arrival checks report this instead of a tolerance they know is broken.
        """

    def load(self, suite, task_id, cameras=True):
        if self.env is not None:
            self.env.close()
            self.env = None
        self.args.task_suite_name, self.args.task_id = suite, task_id
        self.task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
        self.env = sp.build_env(self.args, self.task, camera_observations=cameras)
        self.attach()
        return Chain(suite, task_id, self.task.language)

    def attach(self, settle=True):
        self.world = sp.World(self.env)
        self.executor = sp.Executor(self.env, self.world, self.args)
        if settle:
            self.executor.hold(self.args.settle_steps)
        self.executor.on_step = self.motion_tick
        self.executor.planner.on_progress = lambda: self.tick()
        self.executor.prober.on_progress = lambda: self.tick()
        self.refresh_references()

    def motion_tick(self):
        if self.realtime:
            delay = self.env.env.control_timestep / self.playback_rate - (time.monotonic()-self._last_step)
            if delay > 0:
                time.sleep(delay)
            self._last_step = time.monotonic()
        self.tick()

    @contextmanager
    def waypoint_view(self):
        """Render the editor camera on demand while keeping policy images idle."""
        domain = self.env.env
        images = {name: (obs.is_enabled(), obs.is_active())
                  for name, obs in domain._observables.items()
                  if name.endswith("_image")}
        try:
            for name in images:
                domain.modify_observable(name, "enabled", False)
                domain.modify_observable(name, "active", False)
            yield
        finally:
            for name, (enabled, active) in images.items():
                domain.modify_observable(name, "enabled", enabled)
                domain.modify_observable(name, "active", active)
            if images:
                domain._get_observations(force_update=True)

    def reset(self):
        self.env.reset()
        self.attach()

    def run_policy(self, language, report=None, backend=None):
        from waypoint_policy import run

        return run(self, language, port=self.args.policy_port,
                   max_steps=self.args.policy_max_steps, report=report,
                   backend=backend or self.args.policy_backend)

    def poses(self):
        w = self.world
        names = list(w.domain.objects_dict) + list(w.domain.fixtures_dict) + list(w.domain.object_sites_dict)
        result = {}
        for name in names:
            if w.is_site(name):
                basis = w.sim.data.get_site_xmat(name).reshape(3, 3)
            else:
                basis = w.sim.data.body_xmat[w.domain.obj_body_id[name]].reshape(3, 3)
            result[name] = (w.pos(name).copy(), basis.copy())
        # Joint frames make generated handle targets object-relative too.
        for j in range(w.m.njnt):
            name = w.sim.model.joint_id2name(j)
            if name and not name.startswith(("robot", "gripper")):
                bid = w.m.jnt_bodyid[j]
                result[name] = (np.array(w.d.xpos[bid]).copy(), np.array(w.d.xmat[bid]).reshape(3, 3).copy())
        return result

    def gripper_pose(self):
        sid = self.executor.scene.site_id
        return self.world.eef(), self.env.sim.data.site_xmat[sid].reshape(3, 3).copy()

    def refresh_references(self):
        """Take one scene snapshot for editing; testing a target never moves its frame."""
        self.reference_poses = self.poses()
        self.start_pose = self.gripper_pose()
        self.reference_carry = self.executor.carry_offset.copy(), self.executor.carry_bottom
        self.reference_bottom = {name: self.world.pos(name)[2]-self.world.aabb(name)[0][2]
                                 for name in self.world.domain.objects_dict}
        self.reference_joints = {name: self.world.joint_info(name)["qpos"]
                                 for name in self.reference_poses
                                 if name not in self.world.domain.objects_dict
                                 and name not in self.world.domain.fixtures_dict
                                 and name not in self.world.domain.object_sites_dict}

    def image(self):
        return self.env.sim.render(width=480, height=360, camera_name="agentview")[::-1].copy()

    def preview(self, chain):
        poses, previous = self.reference_poses, self.start_pose
        result = []
        context = dict(preview=True)
        carry_offset, carry_bottom = self.reference_carry
        for w in chain.waypoints:
            if w.grasp and w.gripper == "close":
                carry_offset = poses[w.grasp][0] - previous[0]
                carry_bottom = self.reference_bottom[w.grasp]
            elif w.gripper == "open":
                carry_offset, carry_bottom = np.zeros(3), 0.
            if w.motion not in ("grip", "arc"):
                position, orientation = self.resolve_target(w, poses, previous, context)
                if w.seat:
                    position = position + [0, 0, carry_bottom]
                if w.carry:
                    position = position - carry_offset
                if w.hold_xy:
                    position[:2] = previous[0][:2]
                previous = position, orientation
            result.append(previous)
            if w.gripper_after == "open":
                carry_offset, carry_bottom = np.zeros(3), 0.
        return result

    def editor_target(self, chain, index, waypoint):
        """Use the same connected, fully resolved endpoint for the map and Move."""
        prefix = chain.waypoints[:index] if index is not None else chain.waypoints
        return self.preview(dataclasses.replace(chain, waypoints=prefix + [waypoint]))[-1]

    def test_waypoint(self, chain, index, waypoint, report):
        w = dataclasses.replace(waypoint)
        w.validate()
        final_grip = w.gripper_after if w.gripper_after != "keep" else w.gripper
        same_gripper = (final_grip == "keep" or
                        self.executor.gripper == (sp.GRIPPER_CLOSE if final_grip == "close" else sp.GRIPPER_OPEN))
        same_grasp = not w.grasp or self.executor.holding == w.grasp
        if w.motion == "arc":
            # The saved sweep is relative to the captured joint state, not each click.
            goal = self.reference_joints[w.joint] + w.sweep
            w.sweep = goal - self.world.joint_info(w.joint)["qpos"]
            if abs(w.sweep) < .005 and same_gripper and same_grasp:
                report(0, "ok (joint already at target)")
                return "Finished: joint already at target"
        elif w.motion != "grip":
            pos, ori = self.editor_target(chain, index, w)
            here, rotation = self.gripper_pose()
            angle = np.arccos(np.clip((np.trace(ori @ rotation.T)-1)/2, -1, 1))
            if (np.linalg.norm(here-pos) < .002 and angle < np.radians(1)
                    and same_gripper and same_grasp):
                report(0, "ok (already at target)")
                return "Finished: already at target"
            w = dataclasses.replace(w, reference=WORLD, offset=pos.tolist(), orientation=ori.tolist(),
                                    carry=False, seat=False, hold_xy=False)
        return self.run(dataclasses.replace(chain, waypoints=[w]), report)

    def resolve_target(self, waypoint, poses, previous, context):
        return resolve(waypoint, poses, previous)

    def update_run_frames(self, poses, waypoint, context):
        # Movable objects stay anchored to the run start; fixtures can articulate.
        for name, pose in self.poses().items():
            parent = self.world.parent_body(name) if self.world.is_site(name) else name
            if parent not in self.world.domain.objects_dict:
                poses[name] = pose

    def check_motion_result(self, waypoint, result, position, orientation):
        if self.injected_failure:
            return self.injected_failure
        gap = np.linalg.norm(self.world.eef() - position)
        if gap > self.args.reach_tolerance:
            missed = f"target missed by {gap * 100:.1f} cm"
            return f"failed: {missed}" if result.startswith("ok") else f"{result}; {missed}"
        if self.executor.holding:
            name = self.executor.holding
            domain = self.world.domain
            obj = domain.objects_dict.get(name)
            if obj is not None and not domain._check_grasp(
                    domain.robots[0].gripper, obj.contact_geoms):
                return f"failed: lost grasp on {name} during motion"
        return result

    @contextmanager
    def playback_policy(self):
        """Continue runs get one extra second of playback time without re-driving."""
        if not self.continue_on_failure:
            yield
            return
        ex = self.executor
        missing = object()
        settings = dict(arrival_grace_seconds=1.0, retry_motion=False)
        saved = {key: getattr(ex, key, missing) for key in settings}
        try:
            for key, value in settings.items():
                setattr(ex, key, value)
            yield
        finally:
            for key, value in saved.items():
                if value is missing:
                    delattr(ex, key)
                else:
                    setattr(ex, key, value)

    def run(self, chain, report):
        with self.playback_policy():
            return self._run_waypoints(chain, report)

    def _run_waypoints(self, chain, report):
        chain.validate()
        poses, previous = self.poses(), self.gripper_pose()
        context = {}
        self.last_run_failures = []
        for i, w in enumerate(chain.waypoints):
            self.tick()
            # A pose left over from the previous row must never be shown as
            # this one's, so it is cleared before the row is announced.
            self.running_target = None
            self.update_run_frames(poses, w, context)
            report(i, "Running")
            ex = self.executor
            failures = []
            ex._set_ignore(w.ignore)
            if w.gripper != "keep":
                ex.gripper = sp.GRIPPER_CLOSE if w.gripper == "close" else sp.GRIPPER_OPEN
                if w.motion != "grip":
                    grip_result = ex.run_subgoal(sp.Subgoal(
                        w.name, "grip", gripper=ex.gripper,
                        check_grasp=w.grasp if w.gripper == "close" else "", ignore=w.ignore))
                    if not grip_result.startswith("ok"):
                        failures.append(grip_result)
                        if not self.continue_on_failure:
                            self.last_run_failures.append(dict(row=i, result=grip_result))
                            report(i, grip_result)
                            return f"Stopped at {i + 1}: {grip_result}"
            if w.motion == "grip":
                result = ex.run_subgoal(sp.Subgoal(w.name, "grip", gripper=ex.gripper,
                    check_grasp=w.grasp if ex.gripper == sp.GRIPPER_CLOSE else ""))
            elif w.motion == "arc":
                result = ex.arc(w.joint, w.sweep, np.asarray(w.orientation))
                previous = self.gripper_pose()

            else:
                position, orientation = self.resolve_target(w, poses, previous, context)
                if w.seat:
                    position = position + [0, 0, ex.carry_bottom]
                if w.carry:
                    position = position - ex.carry_offset
                if w.hold_xy:
                    position[:2] = self.world.eef()[:2]
                self.running_target = (position, orientation)
                result = (ex.reach if w.motion == "reach" else ex.line)(position, orientation)
                result = self.check_motion_result(w, result, position, orientation)
                previous = (position, orientation)
            if not result.startswith("ok"):
                failures.append(result)
            if (result.startswith("ok") or self.continue_on_failure) and w.gripper_after != "keep":
                grip_result = ex.run_subgoal(sp.Subgoal(w.name, "grip", ignore=w.ignore,
                    check_grasp=w.grasp if w.gripper_after == "close" else "",
                    gripper=sp.GRIPPER_CLOSE if w.gripper_after == "close" else sp.GRIPPER_OPEN))
                if not grip_result.startswith("ok"):
                    failures.append(grip_result)
            if failures:
                result = "; ".join(failures)
                self.last_run_failures.append(dict(row=i, result=result))
            report(i, result)
            finished = getattr(self, 'on_waypoint_finished', None)
            if finished is not None:
                finished(i, result)
            if failures and not self.continue_on_failure:
                return f"Stopped at {i + 1}: {result}"
        return (f"Finished {len(chain.waypoints)} waypoints • "
                f"failed waypoints: {len(self.last_run_failures)} • task success: {self.env.check_success()}")

    def generate(self, chain):
        """Adapt symbolic task goals without baking object world positions into the file."""
        poses, previous = self.poses(), self.gripper_pose()
        supported = {"in", "on", "open", "close", "turnon", "turnoff"}
        goals = self.world.domain.parsed_problem["goal_state"]
        unsupported = [g[0] for g in goals if g[0].lower() not in supported]
        if unsupported:
            raise ValueError(f"No goal template for {unsupported}. Use manual waypoints or learn a pi0.5 draft.")
        result = []
        for step in sp.decompose(self.world, self.args):
            self.tick()
            for sg in (step.expand(self.executor.probe) if step.expand else step.subgoals):
                t = sg.target
                ori = sp.down_yaw(sg.yaw)
                kw = dict(motion=sg.kind, gripper="keep" if sg.gripper is None else "close" if sg.gripper > 0 else "open",
                          grasp=sg.check_grasp, ignore=list(sg.ignore), joint=sg.joint, sweep=sg.sweep)
                if t is None:
                    w = Waypoint(sg.name, sg.check_grasp or PREVIOUS, orientation=ori.tolist(), **kw)
                elif t.frame == "eef":
                    w = Waypoint(sg.name, PREVIOUS, t.offset.tolist(), ori.tolist(), **kw)
                else:
                    ref = WORLD if t.frame == "world" else t.name
                    clean = dataclasses.replace(t, carry=False, seat=False, hold_xy=False)
                    position = self.executor.resolve(clean)
                    w = relative_waypoint(sg.name, position, ori, ref, poses, previous,
                                          carry=t.carry, seat=t.seat, hold_xy=t.hold_xy, **kw)
                result.append(w)
        if not result:
            raise ValueError("No supported goal template for this task. Add waypoints manually or record pi0.5.")
        return dataclasses.replace(chain, waypoints=result, source="task goal template; review before running")

    def record_policy(self, chain, port=None, max_steps=None, report=None, archive_dir=None,
                      repeats=2, waypoint_count=None):
        """Record the requested runs and extract a plan from the best usable run."""
        from waypoint_demo import record
        from waypoint_library import choose_demonstration
        from waypoint_chain import MAX_WAYPOINTS
        if port is None:
            port = self.args.policy_port
        if not isinstance(repeats, int) or not 1 <= repeats <= 100:
            raise ValueError("Choose between 1 and 100 learning runs.")
        if waypoint_count is not None and (not isinstance(waypoint_count, int) or not 1 <= waypoint_count <= MAX_WAYPOINTS):
            raise ValueError("Choose between 1 and 9 waypoints, or Auto.")
        report = report or (lambda message: None)
        steps = max_steps or (800 if chain.suite == "libero_10" else 500)
        root = Path(archive_dir) if archive_dir is not None else Path(__file__).parent / "task_waypoints/_demonstrations/gui"
        folder = root / chain.suite / f"task_{chain.task_id:03d}" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        folder.mkdir(parents=True, exist_ok=True)
        demonstrations = []
        for i in range(repeats):
            report(f"pi0.5 demonstration {i+1}/{repeats} · connecting to the local policy…")
            self.tick()
            record(self, chain.language, port, steps,
                   progress=lambda step, total, run=i+1: report(f"pi0.5 demonstration {run}/{repeats} · step {step+1}/{total}"))
            demonstration = self.last_demonstration
            demonstrations.append(demonstration)
            (folder / f"rollout_{i+1}.json").write_text(json.dumps(demonstration) + "\n")
        report("Finding approach, grasp, lift, transfer and placement waypoints…")
        chain, selected = choose_demonstration(demonstrations, waypoint_count=waypoint_count)
        (folder / "selection.json").write_text(json.dumps(dict(selected=selected+1,
            rollout_success=[d["success"] for d in demonstrations], waypoints=len(chain.waypoints),
            requested_runs=repeats, requested_waypoints=waypoint_count,
            replay="not run", source=chain.source), indent=2) + "\n")
        self.last_learning_archive = folder
        return chain
