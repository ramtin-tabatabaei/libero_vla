"""Direct policy execution must preserve scene progress and skip learning."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import waypoint_policy as policy
from serve_pi05_libero import MODEL_ID, metadata as pi05_metadata


def session():
    env = Mock()
    obs = {"pixels": "current scene"}
    env.env._get_observations.return_value = obs
    env.step.return_value = (obs, 0, True, {})
    env.check_success.return_value = False
    return SimpleNamespace(env=env, args=SimpleNamespace(task_suite_name="libero_object", seed=0),
                           executor=SimpleNamespace(gripper=1.), tick=Mock(), attach=Mock(),
                           last_demonstration={"existing": "recording"})


@pytest.fixture
def connection(monkeypatch):
    conn = Mock()
    conn.connect.return_value = dict(pi05_metadata())
    actions = np.zeros((10, 7))
    actions[:, -1] = -1.
    conn.infer.return_value = {"actions": actions}
    monkeypatch.setattr(policy, "PolicyConnection", lambda port: conn)
    monkeypatch.setattr(policy, "observation", lambda obs, language: obs)
    monkeypatch.setattr(policy, "switch_controller", Mock())
    return conn


def test_live_run_replans_and_keeps_result_without_recording(connection):
    s = session()
    result = policy.run(s, "pick soup", max_steps=7)
    assert result == {"success": False, "steps": 7, "model_id": MODEL_ID}
    assert s.env.step.call_count == 7  # done=True alone is not task success
    assert connection.infer.call_count == 2  # execute five actions from each chunk
    assert s.last_demonstration == {"existing": "recording"}
    s.env.reset.assert_not_called()
    s.env.set_init_state.assert_not_called()
    s.env.sim.set_state_from_flattened.assert_not_called()
    s.attach.assert_called_once_with(settle=False)
    assert s.executor.gripper == -1.
    connection.close.assert_called_once()
    assert [call.args[1] for call in policy.switch_controller.call_args_list] == ["OSC_POSE", "JOINT_POSITION"]


def test_stops_on_real_task_success(connection):
    s = session()
    s.env.check_success.side_effect = [False, True]
    assert policy.run(s, "pick soup", max_steps=7)["steps"] == 1
    assert s.env.step.call_count == 1


def test_stable_success_holds_pose_for_four_seconds_then_stops(connection):
    s = session()
    data = SimpleNamespace(time=0.)
    s.env.sim.data = data
    s.poses = lambda: {'bowl': (np.zeros(3), np.eye(3))}
    def step(action):
        data.time += .05
        return {}, 0., False, {}
    s.env.step.side_effect = step
    s.env.check_success.side_effect = lambda: data.time >= .05
    result = policy.run(s, 'put bowl on plate', max_steps=100, stable_seconds=4.)
    assert result['success']
    assert result['steps'] == 81
    assert result['stable_seconds'] == pytest.approx(4.)
    assert result['vla_action_steps'] == 1
    assert result['stability_hold_steps'] == 80
    assert connection.infer.call_count == 1
    for call in s.env.step.call_args_list[1:]:
        np.testing.assert_array_equal(call.args[0], [0., 0., 0., 0., 0., 0., -1.])
    s.env.reset.assert_not_called()
    s.env.set_init_state.assert_not_called()


def test_transient_success_cannot_pass_stable_recovery(connection):
    s = session()
    data = SimpleNamespace(time=0.)
    s.env.sim.data = data
    s.poses = lambda: {'bowl': (np.zeros(3), np.eye(3))}
    def step(action):
        data.time += .05
        return {}, 0., False, {}
    s.env.step.side_effect = step
    s.env.check_success.side_effect = lambda: .05 <= data.time < .15
    result = policy.run(s, 'put bowl on plate', max_steps=100, stable_seconds=4.)
    assert not result['success']
    assert result['stable_seconds'] == 0.
    assert connection.infer.call_count > 1


def test_wrong_server_cannot_move_robot(connection):
    s = session()
    connection.connect.return_value["model_id"] = "another_policy"
    with pytest.raises(ValueError, match="Expected Pi 0.5"):
        policy.run(s, "pick soup", max_steps=3)
    s.env.step.assert_not_called()
    policy.switch_controller.assert_not_called()
    s.attach.assert_not_called()
    connection.close.assert_called_once()


def test_standard_pi05_uses_its_checkpoint_and_replans_without_logging(connection):
    s = session()
    connection.connect.return_value = pi05_metadata()
    result = policy.run(s, "pick soup", backend="pi05", port=8000, max_steps=7)
    assert result == {"success": False, "steps": 7, "model_id": "pi05_libero"}
    assert connection.infer.call_count == 2
    assert s.env.step.call_count == 7
    s.env.reset.assert_not_called()
    connection.close.assert_called_once()


@pytest.mark.parametrize("metadata", [
    {},
    {"model_id": "another_policy", "action_horizon": 10, "num_steps": 5,
     "discrete_state_input": False},
    dict(pi05_metadata(), training="PPO"),
    dict(pi05_metadata(), rl_finetuned=True),
])
def test_standard_pi05_rejects_rl_or_unidentified_policies_before_motion(connection, metadata):
    s = session()
    connection.connect.return_value = metadata
    with pytest.raises(ValueError, match="Expected Pi 0.5"):
        policy.run(s, "pick soup", backend="pi05", port=8000, max_steps=3)
    s.env.step.assert_not_called()
    policy.switch_controller.assert_not_called()
    s.attach.assert_not_called()
    connection.close.assert_called_once()


@pytest.mark.parametrize("actions", [np.zeros((3, 7)), np.zeros((10, 8)), np.full((10, 7), np.nan)])
def test_bad_actions_restore_controller_and_keep_scene(connection, actions):
    s = session()
    connection.infer.return_value = {"actions": actions}
    with pytest.raises(ValueError, match="invalid action"):
        policy.run(s, "pick soup", max_steps=3)
    s.env.step.assert_not_called()
    s.env.sim.set_state_from_flattened.assert_not_called()
    s.attach.assert_called_once_with(settle=False)
    assert s.executor.gripper == 1.
    assert policy.switch_controller.call_args.args[1] == "JOINT_POSITION"


def test_stop_leaves_the_robot_at_its_current_state(connection):
    s = session()

    class Stop(Exception):
        pass

    def tick():
        if s.env.step.call_count:
            raise Stop()
    s.tick.side_effect = tick
    with pytest.raises(Stop):
        policy.run(s, "pick soup", max_steps=7)
    assert s.env.step.call_count == 1
    s.env.sim.set_state_from_flattened.assert_not_called()
    s.attach.assert_called_once_with(settle=False)
    assert s.executor.gripper == -1.
    connection.close.assert_called_once()


def test_inference_error_restores_controller(connection):
    s = session()
    connection.infer.side_effect = RuntimeError("server disconnected")
    with pytest.raises(RuntimeError, match="disconnected"):
        policy.run(s, "pick soup", max_steps=3)
    s.attach.assert_called_once_with(settle=False)
    assert policy.switch_controller.call_args.args[1] == "JOINT_POSITION"
