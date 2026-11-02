"""Choose a collision-checked arm start without stepping or moving the scene."""
import numpy as np

from waypoint_rrt import Scene


def randomize_robot_start(session, rng=None):
    """Vary the arm near its home pose; preserve objects, gripper and sim time."""
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
    home = np.asarray(robot.init_qpos, dtype=float)
    lower = np.maximum(scene.limits[:, 0] + .02, home - .4)
    upper = np.minimum(scene.limits[:, 1] - .02, home + .4)
    try:
        for _ in range(128):
            session.tick()
            joints = rng.uniform(lower, upper)
            if not scene.valid(joints):
                continue
            candidate, _ = scene.eef_frame(joints)
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
        raise ValueError("Could not find a different collision-free robot start in this layout.")
    except BaseException:
        sim.set_state_from_flattened(before)
        sim.forward()
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        session.attach(settle=False)
        session.executor.gripper = gripper
        raise
