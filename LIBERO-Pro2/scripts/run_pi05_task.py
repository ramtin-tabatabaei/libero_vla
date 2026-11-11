#!/usr/bin/env python3
"""Run one LIBERO-Pro2 episode through an existing OpenPI pi05_libero server.

Use the simulator's Python environment with the local openpi-client installed.
Use --live to watch both cameras while the task runs. No images or videos are saved.
This is an interactive single-episode check;
the standard OpenPI websocket API does not reset the server's policy RNG.
"""

import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


class PreviewStopped(Exception):
    """The user closed the live window or pressed its stop key."""


class LivePreview:
    """Display the initial observation and each resulting environment frame."""

    def __init__(self, label):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("--live requires a desktop terminal with DISPLAY or WAYLAND_DISPLAY set")
        import cv2
        self.cv2 = cv2
        self.title = "LIBERO-Pro2 - Live task (Q or Esc to close)"
        self.label = label
        self.frames = 0
        self.frame = None
        self.last_display = 0.0
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 1024, 584)
        logging.info("Live camera window opened: %s", label)

    def stopped(self, delay=1):
        key = self.cv2.waitKey(delay) & 0xFF
        return key in (ord("q"), ord("Q"), 27) or self.cv2.getWindowProperty(
            self.title, self.cv2.WND_PROP_VISIBLE) < 1

    def write(self, observation):
        cv2 = self.cv2
        views = []
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            image = np.asarray(observation[key])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{key} must be an HWC uint8 RGB image")
            # Show the same camera orientation used by the policy, at display size.
            bgr = cv2.cvtColor(np.ascontiguousarray(image[::-1, ::-1]), cv2.COLOR_RGB2BGR)
            views.append(cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR))
        self.frame = np.zeros((584, 1024, 3), dtype=np.uint8)
        self.frame[40:552] = np.concatenate(views, axis=1)
        cv2.putText(self.frame, "Agent view", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(self.frame, "Wrist camera", (524, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        caption = f"{self.label} | step {self.frames} | Q / Esc: stop"
        cv2.putText(self.frame, caption, (12, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(self.title, self.frame)
        delay = max(1, int(1000 * (0.05 - (time.monotonic() - self.last_display))))
        if self.stopped(delay):
            raise PreviewStopped()
        self.last_display = time.monotonic()
        self.frames += 1

    def finish(self, success):
        if self.frame is None:
            return
        cv2 = self.cv2
        self.frame[552:] = 0
        text = ("SUCCESS" if success else "EPISODE FINISHED - goal not reached") + " | Q / Esc: close"
        color = (80, 230, 80) if success else (80, 160, 255)
        cv2.putText(self.frame, text, (12, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)
        cv2.imshow(self.title, self.frame)
        logging.info("Episode finished. The final frame stays open; press Q / Esc to close it.")
        while not self.stopped(50):
            pass

    def close(self):
        try:
            self.cv2.destroyWindow(self.title)
            self.cv2.waitKey(1)
        except self.cv2.error:
            pass


def quat_to_axis_angle(quaternion):
    """Match openpi/examples/libero/main.py::_quat2axisangle (xyzw input)."""
    quat = np.asarray(quaternion, dtype=np.float32).copy()
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError("Expected a finite xyzw quaternion with shape (4,)")
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return quat[:3] * 2.0 * math.acos(quat[3]) / denominator


def model_observation(observation):
    """Use the official LIBERO image rotation, resize, and eight-value state."""
    result = {"prompt": str(observation["instruction"])}
    for source, target in (("agentview_image", "observation/image"),
                           ("robot0_eye_in_hand_image", "observation/wrist_image")):
        image = np.asarray(observation[source])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{source} must be an HWC uint8 RGB image")
        image = np.ascontiguousarray(image[::-1, ::-1])
        result[target] = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    state = np.concatenate((observation["robot0_eef_pos"],
                            quat_to_axis_angle(observation["robot0_eef_quat_xyzw"]),
                            observation["robot0_gripper_qpos"]))
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Expected finite LIBERO state: position(3), axis-angle(3), gripper(2)")
    result["observation/state"] = state
    return result


class OpenPiAgent:
    def __init__(self, policy, replan_steps=5):
        if replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        self.policy = policy
        self.replan_steps = replan_steps
        self.actions = deque()
        self.instruction = ""

    def describe(self):
        from libero_remote.runner import AgentCapabilities
        return AgentCapabilities(agent_name="openpi-pi05-libero", agent_version="0.1.0",
                                 control_mode="relative", maximum_inference_seconds=900.0)

    def reset(self, episode):
        self.actions.clear()
        self.instruction = episode.instruction
        self.policy.reset()
        logging.info("Task: %s", self.instruction)

    def act(self, observation):
        instruction = str(observation["instruction"])
        if instruction != self.instruction:
            self.actions.clear()
            self.instruction = instruction
        if not self.actions:
            chunk = np.asarray(self.policy.infer(model_observation(observation))["actions"], dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < self.replan_steps:
                raise ValueError(f"Expected at least {self.replan_steps} actions of dimension 7; got {chunk.shape}")
            if not np.isfinite(chunk).all():
                raise ValueError("OpenPI returned non-finite actions")
            # The remote API enforces normalized controller bounds explicitly.
            self.actions.extend(np.clip(chunk[:self.replan_steps], -1.0, 1.0).copy())
        step = observation.get("step_index", 0)
        if step % 25 == 0:
            logging.info("Environment step %d", step)
        return self.actions.popleft()

    def close(self):
        self.actions.clear()
        # The vendored OpenPI client has no public close method.
        connection = getattr(self.policy, "_ws", None)
        if connection is not None:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=LIMITS, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--variant", default="base")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8001)
    parser.add_argument("--base-url", default=os.environ.get("LIBERO_REMOTE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true", help="Show agent and wrist cameras live; Q / Esc stops the episode")
    args = parser.parse_args()
    api_key = os.environ.get("LIBERO_REMOTE_API_KEY")
    if not api_key:
        parser.error("Load the simulator's client.env first (set -a; source client.env; set +a)")
    if args.task_id < 0 or args.init_state_id < 0 or args.replan_steps < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("IDs must be nonnegative; step limits must be positive")
    from libero_remote import RemoteLiberoEnv
    from libero_remote.runner import AgentEpisodeSpec, run_remote_agent_episode

    # Fail promptly if the server is absent rather than silently waiting forever.
    with socket.create_connection((args.policy_host, args.policy_port), timeout=5):
        pass
    policy = WebsocketClientPolicy(args.policy_host, args.policy_port)
    agent = OpenPiAgent(policy, args.replan_steps)
    spec = AgentEpisodeSpec(suite=args.suite, task_id=args.task_id, init_state_id=args.init_state_id,
                            seed=args.seed, max_steps=args.max_steps or LIMITS[args.suite],
                            control_mode="relative", variant=args.variant)
    with ExitStack() as cleanup:
        cleanup.callback(agent.close)
        preview = None
        if args.live:
            preview = LivePreview(f"{args.suite} / task {args.task_id} / {args.variant}")
            cleanup.callback(preview.close)
        try:
            with RemoteLiberoEnv(args.base_url, api_key=api_key, timeout=120.0) as env:
                result = run_remote_agent_episode(env, agent, spec, recorder=preview).to_dict()
                result["server_result"] = env.result()
        except PreviewStopped:
            logging.info("Episode stopped from the live viewer.")
            return
        result.update({"variant": args.variant, "replan_steps": args.replan_steps,
                       "policy_rng_reset": False, "purpose": "single-episode check",
                       "live_preview": args.live})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or ROOT / "results" / f"pi05_{args.suite}_{args.task_id}_{stamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        print(f"Result saved to {output}", flush=True)
        if preview is not None:
            preview.finish(result["success"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
