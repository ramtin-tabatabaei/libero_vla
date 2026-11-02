"""Every recorded control step must have both cameras with the correct phase."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from recovery_frame_images import FrameImages
from waypoint_recovery_batch import CompactRecorder


class Env:
    def __init__(self):
        self.data = SimpleNamespace(time=0., qpos=np.zeros(3), qvel=np.zeros(3))
        self.sim = SimpleNamespace(data=self.data, render=self.render)
        self.env = SimpleNamespace(control_timestep=.05)
        self.steps = 0

    def step(self, action):
        self.steps += 1
        self.data.time += .05
        self.data.qpos += 1
        return {'state': self.steps}

    def render(self, width, height, camera_name):
        image = np.full((height, width, 3), self.steps, dtype=np.uint8)
        image[0, :, 1] = 100 if camera_name == 'agentview' else 200
        return image

    def check_success(self):
        return False


def test_every_step_has_both_camera_images_and_a_controller_index(tmp_path):
    env = Env()
    session = SimpleNamespace(env=env, gripper_pose=lambda: (np.zeros(3), np.eye(3)))
    folder, images = tmp_path/'trial', tmp_path/'frame_drive/trial'
    recorder = CompactRecorder(session, folder, 1e6, image_folder=images)
    with recorder:
        assert env.step(np.zeros(8)) == {'state': 1}
        recorder.phase = 1
        env.step(np.zeros(7))
        env.step(np.zeros(7))
    assert env.steps == 3
    assert 'step' not in env.__dict__
    with (folder/'photos/frames.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8  # Initial pair, then two frames per step; none sampled out.
    assert len(list(images.rglob('*.png'))) == 8
    assert not list(images.rglob('*.tmp'))
    with np.load(str(folder/'trajectory.npz')) as trajectory:
        for step in range(3):
            frames = [r for r in rows if int(r['control_step']) == step]
            assert {r['camera'] for r in frames} == {'scene', 'wrist'}
            for frame in frames:
                assert float(frame['sim_time']) == trajectory['sim_time'][step]
                assert frame['controller'] == ('waypoints' if trajectory['phase'][step] == 0 else 'policy')
                pixels = np.asarray(Image.open(images/frame['image']))
                assert pixels.shape == (256, 256, 3)
                assert pixels[0, 0, 0] == step+1  # The frame is post-step, not stale.
                assert pixels[-1, 0, 1] == (100 if frame['camera'] == 'scene' else 200)
    capture = json.loads((images/'capture.json').read_text())
    assert capture['completed']
    assert capture['waypoint_images'] == 4
    assert capture['vla_images'] == 4


def test_interrupted_capture_is_closed_and_preserved_on_retry(tmp_path):
    env = Env()
    session = SimpleNamespace(env=env, gripper_pose=lambda: (np.zeros(3), np.eye(3)))
    folder, images = tmp_path/'trial', tmp_path/'drive/trial'
    with pytest.raises(RuntimeError, match='interrupted'):
        with CompactRecorder(session, folder, 1e6, image_folder=images):
            env.step(np.zeros(8))
            raise RuntimeError('interrupted')
    assert not json.loads((images/'capture.json').read_text())['completed']
    retry = FrameImages(session, folder, images)
    retry.start()
    retry.close(True)
    previous = list(images.parent.glob('trial.interrupted_*'))
    assert len(previous) == 1
    assert len(list(previous[0].rglob('*.png'))) == 4
    assert len(list(images.rglob('*.png'))) == 2
