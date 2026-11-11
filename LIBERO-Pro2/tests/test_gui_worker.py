import sys
from pathlib import Path
import queue
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import libero_gui as gui


def test_camera_queue_keeps_latest_frame_without_aliasing():
    frames, events, stop = queue.Queue(maxsize=1), queue.Queue(), threading.Event()
    observer = gui.FrameObserver(frames, events, stop)
    original = np.ones((3, 4, 3), dtype=np.uint8)
    obs = {"agentview_image": original, "robot0_eye_in_hand_image": original}
    observer.write(obs)
    original[:] = 7
    observer.write(obs)
    original[:] = 9
    pair = frames.get_nowait()
    assert frames.empty()
    assert np.all(pair[0] == 7)
    assert np.all(pair[1] == 7)
    stop.set()
    with pytest.raises(gui.EpisodeStopped):
        observer.write(obs)


class Socket:
    def __init__(self, stop=None):
        self.messages = [gui.msgpack_numpy.packb({}), None,
                         gui.msgpack_numpy.packb({"actions": np.ones((5, 7), dtype=np.float32)})]
        self.sent = []
        self.stop = stop
        self.closed = False

    def recv(self, timeout=None):
        value = self.messages.pop(0)
        if value is None:
            if self.stop:
                self.stop.set()
            raise TimeoutError()
        return value

    def send(self, value):
        self.sent.append(value)

    def close(self):
        self.closed = True


def test_policy_uses_official_numpy_messagepack_protocol(monkeypatch):
    socket = Socket()
    monkeypatch.setattr(gui, "connect", lambda *a, **k: socket)
    policy = gui.GuiPolicy("127.0.0.1", 8001, threading.Event())
    result = policy.infer({"prompt": "move the bowl", "observation/state": np.zeros(8)})
    assert result["actions"].shape == (5, 7)
    assert gui.msgpack_numpy.unpackb(socket.sent[0])["prompt"] == "move the bowl"


def test_stop_interrupts_waiting_for_policy_response(monkeypatch):
    stop = threading.Event()
    socket = Socket(stop)
    monkeypatch.setattr(gui, "connect", lambda *a, **k: socket)
    policy = gui.GuiPolicy("127.0.0.1", 8001, stop)
    with pytest.raises(gui.EpisodeStopped):
        policy.infer({"prompt": "move the bowl"})
