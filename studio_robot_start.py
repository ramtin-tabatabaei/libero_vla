"""Choose a collision-checked arm start without stepping or moving the scene."""
import numpy as np

from waypoint_rrt import Scene


START_VARIATIONS = {
    "small": (.12, .04, .08),
    "large": (.8, .20, .35),
}
"""Maximum joint offset (radians), minimum and maximum gripper offset (metres)."""


def randomize_robot_start(session, rng=None, *, variation="small"):
    """Vary the arm around its normal start; preserve objects, gripper and time."""
    if variation not in START_VARIATIONS:
        raise ValueError("Choose a small or large robot start variation.")
    if session.env is None:
        raise ValueError("Load a task first.")
    if session.executor.holding:
        raise ValueError("Release the held object or reset before choosing a robot start.")
    rng = np.random.default_rng() if rng is None else rng
    scene = Scene(session.env)
    sim = session.env.sim
    robot = session.env.env.robots[0]
    before = sim.get_state().flatten().copy()
    gripper = session.executor.gripper
    position = scene.eef_pos().copy()
    home = np.asarray(getattr(session, "normal_start_qpos", robot.init_qpos), dtype=float)
    joint_offset, min_distance, max_distance = START_VARIATIONS[variation]
    lower = np.maximum(scene.limits[:, 0] + .02, home - joint_offset)
    upper = np.minimum(scene.limits[:, 1] - .02, home + joint_offset)
    try:
        normal_position, _ = scene.eef_frame(home)
        for _ in range(128):
            session.tick()
            joints = rng.uniform(lower, upper)
            if not scene.valid(joints):
                continue
            candidate, _ = scene.eef_frame(joints)
            if not min_distance <= np.linalg.norm(candidate - normal_position) <= max_distance:
                continue
            if np.linalg.norm(candidate - position) < .04:
                continue
            sim.data.qpos[scene.qadr] = joints
            sim.data.qvel[robot._ref_joint_vel_indexes] = 0.
            sim.forward()
            robot.controller.update(force=True)
            robot.controller.reset_goal()
            session.attach(settle=False)
            session.executor.gripper = gripper
            session.env.env._get_observations(force_update=True)
            return candidate.copy()
        raise ValueError(f"Could not find a different collision-free {variation} robot start in this layout.")
    except BaseException:
        sim.set_state_from_flattened(before)
        sim.forward()
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        session.attach(settle=False)
        session.executor.gripper = gripper
        raise
