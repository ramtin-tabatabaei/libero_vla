"""A bounded pi0.5 rollout recorder using the existing local OpenPI wire format."""
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np


def switch_controller(env, name):
    from robosuite.controllers import load_controller_config
    from robosuite.utils.buffers import DeltaBuffer
    robot = env.env.robots[0]
    robot.controller_config = load_controller_config(default_controller=name)
    robot._load_controller()
    robot.controller.update_base_pose(robot.base_pos, robot.base_ori)
    robot.controller.reset_goal()
    robot.recent_actions = DeltaBuffer(dim=robot.action_dim)
    env.env._action_dim = sum(r.action_dim for r in env.env.robots)


def observation(obs, language):
    from openpi_client import image_tools
    from robosuite.utils.transform_utils import quat2axisangle
    def picture(key):
        img = np.ascontiguousarray(obs[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
    return {
        "observation/image": picture("agentview_image"),
        "observation/wrist_image": picture("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"].copy()), obs["robot0_gripper_qpos"])),
        "prompt": language,
    }


class PolicyConnection:
    """Finite connection/inference timeouts so Stop and closing the GUI remain usable."""
    def __init__(self, port):
        self.port, self.ws = port, None
        self.cancelled = False

    def connect(self):
        from websockets.sync.client import connect
        from openpi_client import msgpack_numpy
        self.ws = connect(f"ws://127.0.0.1:{self.port}", compression=None, max_size=None, open_timeout=10, close_timeout=1)
        if self.cancelled:
            self.ws.close()
            raise RuntimeError("Policy connection cancelled.")
        return msgpack_numpy.unpackb(self.ws.recv(timeout=15))

    def infer(self, obs):
        from openpi_client import msgpack_numpy
        self.ws.send(msgpack_numpy.packb(obs))
        response = self.ws.recv(timeout=120)
        if isinstance(response, str):
            raise RuntimeError(response)
        return msgpack_numpy.unpackb(response)

    def close(self):
        self.cancelled = True
        if self.ws is not None:
            self.ws.close()


def record(session, language, port=None, max_steps=300, progress=None):
    """Return sampled gripper poses; restore the scene even on cancellation/failure."""
    if port is None:
        port = session.args.policy_port
    samples = []
    state = session.env.sim.get_state().flatten().copy()
    initial_gripper = session.executor.gripper
    poses = session.poses()
    # Keep the complete measured demonstration as evidence, separately from the
    # small editable plan. Each sample records the pose BEFORE its grip command.
    demonstration = dict(version=1, policy="pi05_libero", language=language,
                         suite=session.args.task_suite_name, task_id=session.args.task_id,
                         initial_state=state.tolist(),
                         poses={name: [p.tolist(), r.tolist()] for name, (p, r) in poses.items()},
                         objects=list(session.world.domain.objects_dict),
                         fixtures=list(session.world.domain.fixtures_dict), samples=[], success=False)
    from waypoint_observation import episode_metadata, measured_sample
    demonstration.update(episode_metadata(session))
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    pool = ThreadPoolExecutor(max_workers=1)
    connection = PolicyConnection(port)
    def wait(future):
        while not future.done():
            session.tick()
            time.sleep(.01)
        return future.result()
    switched = False
    try:
        metadata = wait(pool.submit(connection.connect))
        demonstration["policy_metadata"] = str(metadata)
        switch_controller(session.env, "OSC_POSE")
        switched = True
        obs = session.env.env._get_observations()
        actions, last_grip = [], "close" if initial_gripper > 0 else "open"
        for step in range(max_steps):
            if progress is not None and step % 5 == 0:
                progress(step, max_steps)
            if not actions:
                output = wait(pool.submit(connection.infer, observation(obs, language)))
                chunk = np.asarray(output["actions"], dtype=float)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or not len(chunk) or not np.isfinite(chunk).all():
                    raise ValueError("Policy returned an invalid action chunk.")
                actions = list(chunk[:5])
            action = actions.pop(0)
            previous = demonstration["samples"][-1] if demonstration["samples"] else None
            demonstration["samples"].append(measured_sample(session, obs, action, step, previous))
            obs, _, _, _ = session.env.step(action)
            grip = "close" if action[-1] > 0 else "open"
            pos, ori = session.gripper_pose()
            if not samples or np.linalg.norm(pos-samples[-1][0]) >= .05 or grip != last_grip or step == max_steps-1:
                samples.append((pos, ori, grip, grip != last_grip))
            last_grip = grip
            session.tick()
            if session.env.check_success():
                if np.linalg.norm(pos-samples[-1][0]) > 1e-6:
                    samples.append((pos, ori, grip, False))
                break
        demonstration["terminal_sample"] = measured_sample(
            session, obs, action, len(demonstration["samples"]), demonstration["samples"][-1])
        demonstration["episode_length"] = len(demonstration["samples"])
        demonstration["end_position"] = pos.tolist()
        demonstration["end_orientation"] = ori.tolist()
        demonstration["success"] = bool(session.env.check_success())
        session.last_demonstration = demonstration
        return samples, demonstration["success"]
    finally:
        connection.close()
        pool.shutdown(wait=False)
        if switched:
            switch_controller(session.env, "JOINT_POSITION")
        session.env.sim.set_state_from_flattened(state)
        session.env.sim.forward()
        session.attach(settle=False)
        session.executor.gripper = initial_gripper
        session.env.env.robots[0].controller.reset_goal()


def main():
    """Collect a bounded number of attempts; retain failures separately for audit."""
    import argparse
    import json
    from pathlib import Path
    import subgoal_planner as sp
    from waypoint_session import Session
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--num-successes', type=int, default=5)
    parser.add_argument('--max-attempts', type=int, default=30)
    parser.add_argument('--max-steps', type=int, default=800)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-placement', action='store_true', default=True,
                        help='Use the original unshuffled scene for every VLA recording (default).')
    args = parser.parse_args()
    if args.max_steps < 3:
        parser.error('max-steps must be at least 3')
    if args.num_successes < 1 or args.max_attempts < args.num_successes:
        parser.error('Require 1 <= num-successes <= max-attempts')
    args.output.mkdir(parents=True, exist_ok=True)
    if list(args.output.glob('demo_*.json')):
        parser.error('Output already contains demonstrations; use a fresh directory')
    session = Session(sp.Args(seed=args.seed))
    successes, attempts = 0, []
    try:
        chain = session.load(args.suite, args.task_id)
        training_state = session.env.sim.get_state().flatten().copy()
        training_gripper = session.executor.gripper
        for attempt in range(args.max_attempts):
            session.env.set_init_state(training_state)
            session.attach(settle=False)
            session.executor.gripper = training_gripper
            record(session, chain.language, args.port, args.max_steps)
            demo = session.last_demonstration
            usable = demo['success'] and len(demo['samples']) >= 3 and not demo['samples'][0]['task_success']
            folder = args.output if usable else args.output/'failed'
            folder.mkdir(exist_ok=True)
            name = 'demo_%03d' % attempt
            state = demo.pop('initial_state')
            state_path = folder/(name+'.state.json')
            state_path.write_text(json.dumps(dict(initial_state=state)))
            demo['replay_state_file'] = state_path.name
            demo['seed'] = args.seed
            demo['placement_mode'] = 'fixed_unshuffled'
            demo['init_state_index'] = None
            (folder/(name+'.json')).write_text(json.dumps(demo))
            successes += int(usable)
            attempts.append(dict(attempt=attempt, success=demo['success'], usable=bool(usable), file=str(folder/(name+'.json'))))
            print('%d/%d successful demonstrations' % (successes, args.num_successes), flush=True)
            if successes >= args.num_successes:
                break
    finally:
        if session.env is not None:
            session.env.close()
        (args.output/'recording_report.json').write_text(json.dumps(dict(successes=successes, attempts=attempts), indent=2))
    if successes < args.num_successes:
        raise SystemExit('Attempt budget exhausted: collected %d successes' % successes)


if __name__ == '__main__':
    main()
