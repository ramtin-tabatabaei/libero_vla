"""Run the selected policy in the studio's current scene."""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np

from run_vla import MAX_STEPS
from serve_pi05_libero import metadata as pi05_metadata, REPLAN_STEPS as PI05_REPLAN_STEPS
from waypoint_demo import PolicyConnection, observation, switch_controller


class StableSuccess:
    """Require an uninterrupted successful goal and stationary scene poses."""
    def __init__(self, seconds, metres=.01, degrees=5.):
        if (not np.isfinite([seconds, metres, degrees]).all() or seconds < 0
                or metres <= 0 or degrees <= 0):
            raise ValueError("Stable-success duration and pose tolerances must be finite and valid")
        self.required, self.metres, self.radians = seconds, metres, np.radians(degrees)
        self.start = self.first_goal = None
        self.anchor = None
        self.duration = 0.

    def update(self, now, goal, poses):
        if not goal:
            self.start = self.anchor = None
            self.duration = 0.
            return False
        if self.first_goal is None:
            self.first_goal = float(now)
        moved = self.anchor is not None and (set(poses) != set(self.anchor) or any(
            np.linalg.norm(np.asarray(position)-self.anchor[name][0]) > self.metres or
            np.arccos(np.clip((np.trace(self.anchor[name][1].T @ orientation)-1)/2, -1., 1.))
                > self.radians for name, (position, orientation) in poses.items()))
        if self.anchor is None or moved:
            self.start = float(now)
            self.anchor = {name: (np.asarray(p).copy(), np.asarray(r).copy())
                           for name, (p, r) in poses.items()}
        self.duration = max(0., float(now)-self.start)
        return self.duration + 1e-8 >= self.required


def run(session, language, port=8002, max_steps=None, report=None, backend="pi05",
        stable_seconds=0., stability_metres=.01, stability_degrees=5.):
    """Execute policy actions without resetting the scene or producing demonstrations."""
    if session.env is None:
        raise ValueError("Load a task first.")
    if backend != "pi05":
        raise ValueError("Pi 0.5 is the only installed policy")
    if max_steps is None:
        max_steps = MAX_STEPS[session.args.task_suite_name]
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("Policy step limit must be a positive whole number")
    stable = StableSuccess(stable_seconds, stability_metres, stability_degrees)
    report = report or (lambda message: None)
    connection = PolicyConnection(port)
    expected = pi05_metadata()
    name, replan_steps = "Pi 0.5", PI05_REPLAN_STEPS
    pool = ThreadPoolExecutor(max_workers=1)
    switched = False
    gripper = session.executor.gripper

    def wait(future):
        while not future.done():
            session.tick()
            time.sleep(.01)
        session.tick()
        return future.result()

    try:
        report("Connecting to " + name + "…")
        metadata = wait(pool.submit(connection.connect))
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Expected {name} policy: "
                                 f"{key}={metadata.get(key)!r}, expected {value!r}")
        switched = True
        switch_controller(session.env, "OSC_POSE")
        obs = session.env.env._get_observations(force_update=True)
        goal = bool(session.env.check_success())
        policy_start = float(session.env.sim.data.time) if stable_seconds else None
        first_action = None
        success = (stable.update(policy_start, goal, session.poses()) if stable_seconds else goal)
        steps, actions, vla_steps, hold_steps = 0, [], 0, 0
        while steps < max_steps and not success:
            session.tick()
            if stable_seconds and goal:
                # Keep the achieved pose and gripper command while testing
                # whether the task and objects remain stable for the full window.
                action = np.zeros(7)
                action[-1] = gripper
                actions = []
                hold_steps += 1
            else:
                if not actions:
                    report(f"{name} · step {steps}/{max_steps}")
                    inputs = observation(obs, language)
                    output = wait(pool.submit(connection.infer, inputs))
                    chunk = np.asarray(output["actions"], dtype=float)
                    if (chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < replan_steps
                            or not np.isfinite(chunk).all()):
                        raise ValueError("Policy returned an invalid action chunk")
                    actions = list(chunk[:replan_steps])
                action = actions.pop(0)
                vla_steps += 1
            obs, _, _, _ = session.env.step(action)
            if stable_seconds and first_action is None:
                first_action = float(session.env.sim.data.time)
            # Retain the final gripper command when handing control back to the editor.
            gripper = 1.0 if action[-1] > 0 else -1.0
            steps += 1
            session.tick()
            goal = bool(session.env.check_success())
            success = (stable.update(float(session.env.sim.data.time), goal, session.poses())
                       if stable_seconds else goal)
        result = {"success": success, "steps": steps, "model_id": metadata["model_id"]}
        if stable_seconds:
            result.update(task_success=goal, stable_seconds=stable.duration,
                first_goal_sim_time=stable.first_goal, stable_start_sim_time=stable.start,
                policy_start_sim_time=policy_start, first_action_sim_time=first_action,
                policy_end_sim_time=float(session.env.sim.data.time),
                vla_action_steps=vla_steps, stability_hold_steps=hold_steps)
        return result
    finally:
        try:
            connection.close()
        finally:
            pool.shutdown(wait=False)
            if switched:
                switch_controller(session.env, "JOINT_POSITION")
                session.attach(settle=False)
                session.executor.gripper = gripper
