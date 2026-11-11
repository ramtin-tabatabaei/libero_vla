"""Robot starts change only the arm and leave a usable waypoint controller."""
import numpy as np
import pytest

import subgoal_planner as sp
from studio_robot_start import START_VARIATIONS, randomize_robot_start
from waypoint_gmm_editor import GMMSession
from waypoint_rrt import Scene


@pytest.fixture(scope="module")
def session():
    session = GMMSession(sp.Args(task_suite_name="libero_spatial", task_id=0))
    session.load("libero_spatial", 0, cameras=False)
    yield session
    session.env.close()


@pytest.fixture(autouse=True)
def restore(session):
    session.restore_training_scene()
    session.tick = lambda: None
    yield
    session.tick = lambda: None
    session.restore_training_scene()


@pytest.mark.parametrize("variation", ["small", "large"])
def test_random_start_changes_arm_only_and_keeps_controller_usable(session, variation):
    sim = session.env.sim
    before = sim.get_state().flatten().copy()
    qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
    scene = Scene(session.env)
    position = scene.eef_pos().copy()
    gripper = session.executor.gripper
    normal = session.normal_start_qpos.copy()
    normal_position, _ = scene.eef_frame(normal)
    result = randomize_robot_start(session, np.random.default_rng(5), variation=variation)
    joint_offset, minimum, maximum = START_VARIATIONS[variation]
    assert minimum <= np.linalg.norm(result - normal_position) <= maximum
    assert np.max(np.abs(scene.arm_qpos() - normal)) <= joint_offset
    np.testing.assert_array_equal(session.normal_start_qpos, normal)
    assert np.linalg.norm(result - position) >= .04
    assert sim.data.time == before[0]
    other_qpos = np.setdiff1d(np.arange(len(qpos)), scene.qadr)
    arm_vel = session.env.env.robots[0]._ref_joint_vel_indexes
    other_qvel = np.setdiff1d(np.arange(len(qvel)), arm_vel)
    np.testing.assert_array_equal(sim.data.qpos[other_qpos], qpos[other_qpos])
    np.testing.assert_array_equal(sim.data.qvel[other_qvel], qvel[other_qvel])
    assert scene.valid(scene.arm_qpos())
    np.testing.assert_allclose(session.start_pose[0], result)
    np.testing.assert_allclose(session.executor.q_hold, scene.arm_qpos())
    np.testing.assert_allclose(session.env.env.robots[0].controller.goal_qpos, scene.arm_qpos())
    assert session.executor.gripper == gripper
    np.testing.assert_array_equal(session.training_state, before)
    session.executor.hold(3)
    assert np.linalg.norm(session.world.eef() - result) < .02


@pytest.mark.parametrize("variation", ["small", "large"])
def test_random_seed_reproduces_start_and_new_sample_changes_it(session, variation):
    a = randomize_robot_start(session, np.random.default_rng(12), variation=variation)
    session.restore_training_scene()
    b = randomize_robot_start(session, np.random.default_rng(12), variation=variation)
    np.testing.assert_array_equal(a, b)
    c = randomize_robot_start(session, np.random.default_rng(13), variation=variation)
    assert np.linalg.norm(c - b) >= .04


def test_small_change_stays_near_normal_after_a_large_change(session):
    scene = Scene(session.env)
    normal = session.normal_start_qpos.copy()
    normal_position, _ = scene.eef_frame(normal)
    large = randomize_robot_start(session, np.random.default_rng(5), variation="large")
    small = randomize_robot_start(session, np.random.default_rng(5), variation="small")
    assert np.linalg.norm(large - normal_position) >= .20
    assert .04 <= np.linalg.norm(small - normal_position) <= .08
    np.testing.assert_array_equal(session.normal_start_qpos, normal)


def test_unavailable_distance_and_invalid_variation_preserve_scene(session):
    before = session.env.sim.get_state().flatten().copy()
    class NoVariation:
        def uniform(self, lower, upper):
            return session.normal_start_qpos.copy()
    with pytest.raises(ValueError, match="collision-free large"):
        randomize_robot_start(session, NoVariation(), variation="large")
    np.testing.assert_array_equal(before, session.env.sim.get_state().flatten())
    with pytest.raises(ValueError, match="small or large"):
        randomize_robot_start(session, variation="unknown")
    np.testing.assert_array_equal(before, session.env.sim.get_state().flatten())


def test_no_valid_start_preserves_original_scene(session, monkeypatch):
    before = session.env.sim.get_state().flatten().copy()
    monkeypatch.setattr(Scene, "valid", lambda *args: False)
    with pytest.raises(ValueError, match="collision-free"):
        randomize_robot_start(session, np.random.default_rng(1))
    np.testing.assert_array_equal(before, session.env.sim.get_state().flatten())


def test_cancel_preserves_scene_and_held_object_is_rejected(session):
    before = session.env.sim.get_state().flatten().copy()
    def stop():
        raise sp.Abort()
    session.tick = stop
    with pytest.raises(sp.Abort):
        randomize_robot_start(session)
    np.testing.assert_array_equal(before, session.env.sim.get_state().flatten())
    session.executor.holding = "bowl"
    with pytest.raises(ValueError, match="Release"):
        randomize_robot_start(session)
    np.testing.assert_array_equal(before, session.env.sim.get_state().flatten())
