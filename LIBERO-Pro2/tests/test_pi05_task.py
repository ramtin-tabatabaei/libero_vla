"""Check the OpenPI bridge without loading model weights or starting a simulator."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from openpi_client import image_tools
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_pi05_task", ROOT / "scripts/run_pi05_task.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def observation():
    return {
        "agentview_image": np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3),
        "robot0_eye_in_hand_image": np.full((12, 16, 3), 83, dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat_xyzw": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
        "instruction": "pick up the bowl",
    }


def test_preprocessing_matches_vendored_openpi_pipeline():
    # Execute the upstream conversion itself as the quaternion reference.
    source = ast.parse((ROOT / "openpi/examples/libero/main.py").read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_quat2axisangle")
    namespace = {"np": np, "math": math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "official_openpi", "exec"), namespace)
    reference = namespace["_quat2axisangle"]
    for quaternion in [[0, 0, 0, 1], [0, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], [0, 0, 0, 1.00001]]:
        original = np.asarray(quaternion, dtype=np.float32)
        before = original.copy()
        np.testing.assert_allclose(bridge.quat_to_axis_angle(original), reference(original.copy()))
        np.testing.assert_array_equal(original, before)
    obs = observation()
    converted = bridge.model_observation(obs)
    for source_key, target in [("agentview_image", "observation/image"),
                               ("robot0_eye_in_hand_image", "observation/wrist_image")]:
        expected = image_tools.convert_to_uint8(image_tools.resize_with_pad(
            np.ascontiguousarray(obs[source_key][::-1, ::-1]), 224, 224))
        np.testing.assert_array_equal(converted[target], expected)
    np.testing.assert_allclose(converted["observation/state"], [0.1, 0.2, 0.3, 0, 0, 0, 0.02, -0.02])
    assert converted["prompt"] == obs["instruction"]


class Policy:
    def __init__(self, chunk=None):
        self.calls = []
        self.chunk = np.arange(70, dtype=np.float32).reshape(10, 7) / 100 if chunk is None else chunk

    def reset(self):
        pass

    def infer(self, obs):
        self.calls.append(obs)
        return {"actions": self.chunk}


def test_replanning_discards_unused_chunk_tail_and_reset_clears_queue():
    policy = Policy()
    agent = bridge.OpenPiAgent(policy)
    obs = observation()
    agent.reset(SimpleNamespace(instruction=obs["instruction"]))
    for i in range(5):
        np.testing.assert_array_equal(agent.act(obs), policy.chunk[i])
    assert len(policy.calls) == 1
    np.testing.assert_array_equal(agent.act(obs), policy.chunk[0])
    assert len(policy.calls) == 2
    agent.reset(SimpleNamespace(instruction=obs["instruction"]))
    np.testing.assert_array_equal(agent.act(obs), policy.chunk[0])
    assert len(policy.calls) == 3


def test_runtime_instruction_change_invalidates_cached_actions():
    policy = Policy()
    agent = bridge.OpenPiAgent(policy)
    obs = observation()
    agent.reset(SimpleNamespace(instruction=obs["instruction"]))
    agent.act(obs)
    obs["instruction"] = "move the bowl to the plate"
    np.testing.assert_array_equal(agent.act(obs), policy.chunk[0])
    assert len(policy.calls) == 2
    assert policy.calls[-1]["prompt"] == obs["instruction"]


@pytest.mark.parametrize("chunk", [np.zeros((2, 7)), np.zeros((5, 8)), np.full((5, 7), np.nan)])
def test_invalid_chunks_fail_explicitly(chunk):
    agent = bridge.OpenPiAgent(Policy(chunk))
    with pytest.raises(ValueError):
        agent.act(observation())


def test_actions_respect_remote_controller_bounds():
    agent = bridge.OpenPiAgent(Policy(np.tile([-1.1, 0, 0, 0, 0, 0, 1.1], (5, 1))))
    np.testing.assert_array_equal(agent.act(observation()), [-1, 0, 0, 0, 0, 0, 1])


def mock_preview_gui(monkeypatch, key=-1, visible=1):
    import cv2
    shown = []
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: shown.append(frame.copy()))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: visible)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: None)
    return cv2, shown


def test_live_preview_displays_both_cameras_without_changing_observation(monkeypatch):
    cv2, shown = mock_preview_gui(monkeypatch)
    preview = bridge.LivePreview("test task")
    obs = observation()
    original = {key: value.copy() for key, value in obs.items() if isinstance(value, np.ndarray)}
    preview.write(obs)
    assert shown[0].shape == (584, 1024, 3)
    for index, name in enumerate(("agentview_image", "robot0_eye_in_hand_image")):
        expected = cv2.resize(np.ascontiguousarray(obs[name][::-1, ::-1, ::-1]), (512, 512))
        np.testing.assert_array_equal(shown[0][40:552, index * 512:(index + 1) * 512], expected)
    for key, value in original.items():
        np.testing.assert_array_equal(obs[key], value)
    preview.close()


@pytest.mark.parametrize("key,visible", [(ord("q"), 1), (27, 1), (-1, -1)])
def test_live_preview_stop_controls(monkeypatch, key, visible):
    mock_preview_gui(monkeypatch, key=key, visible=visible)
    preview = bridge.LivePreview("test task")
    with pytest.raises(bridge.PreviewStopped):
        preview.write(observation())
    preview.close()
