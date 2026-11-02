"""Run a LIBERO task in the MuJoCo viewer and take commands over a socket.

Control modes:

  --mode waypoint  (default)  JOINT_POSITION + IK + RRT-Connect.
                              You give coordinates; motion is collision-checked.
  --mode vla                  OSC_POSE, the action space pi0.5 was trained on.
                              You give natural language; the policy decides.
  --mode hybrid               RRT grasp followed by VLA in the same simulation.
                              Use `pick milk_1`, then `vla "put the milk in the basket"`.

    Terminal 1:  MUJOCO_GL=glx python robot_server.py --mode vla
    Terminal 2:  python robot_cli.py vla "pick up the milk and place it in the basket"

In vla or hybrid mode the openpi policy server is started automatically unless one is
already listening (--no-auto-policy to manage it yourself).

The sim cannot be shared between processes, so this process owns it. Commands
arrive on a socket, are queued, and are executed by the main loop -- MuJoCo and
the viewer are not thread-safe, so nothing touches the sim off-thread.
"""

import atexit
import collections
import dataclasses
import json
import math
import pathlib
import queue
import shlex
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from waypoint_rrt import CTRL_DELTA_MAX, GRIPPER_CLOSE, GRIPPER_OPEN, RRTConnect, Scene

HOST = "127.0.0.1"
OPENPI_DIR = pathlib.Path("/home/ramtin/openpi")
CKPT_ROOT = pathlib.Path.home() / ".cache/openpi/openpi-assets/checkpoints"

# OSC_POSE maps action [-1,1] -> +-0.05 m translation per step.
OSC_POS_SCALE = 0.05
VLA_RESOLUTION = 256  # what the pi0.5 LIBERO data was rendered at
VLA_RESIZE = 224  # what the policy expects

HELP = """commands:
  state                    object + eef poses
  forces                   finger actuator/contact forces and wrist wrench (JSON)
  success                  is the task goal satisfied?
  move X Y Z               move the gripper to a world XYZ
  moveto OBJ [dz]          move above object OBJ (default dz=0.12)
  lift DZ                  move straight up by DZ from where you are
  grasp / release          close / open the gripper
  pick OBJ                 RRT approach, grasp and lift; hold for next command [hybrid/waypoint]
  home                     return to the start configuration   [waypoint mode]
  run FILE.json            execute a waypoint file             [waypoint mode]
  vla "INSTRUCTION" [N]    let pi0.5 drive for N steps         [vla/hybrid mode]
  shuffle [SEED] [OBJ...]  permute where the objects sit
  reset                    reset the task
  help                     this text
  quit                     shut the server down"""


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_object"
    task_id: int = 0
    seed: int = 0
    port: int = 5599

    mode: str = "waypoint"
    """'waypoint' (RRT), 'vla' (pi0.5), or 'hybrid' (RRT grasp then pi0.5)."""

    viewer: bool = True
    """Open the MuJoCo window. Needs MUJOCO_GL=glx."""

    # --- vla mode ---
    policy: str = "pi05_libero"
    """pi05_libero or pi0_libero."""
    policy_port: int = 8000
    auto_policy: bool = True
    """Start the openpi policy server if nothing is listening on policy_port."""
    replan_steps: int = 5
    vla_max_steps: int = 400

    # --- waypoint mode ---
    rrt_max_iters: int = 4000
    rrt_step: float = 0.15
    collision_check_step: float = 0.05
    settle_steps: int = 10


def _port_open(port, host=HOST):
    s = socket.socket()
    s.settimeout(0.5)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils (xyzw)."""
    quat = np.asarray(quat, dtype=float).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class PolicyServerProcess:
    """Starts openpi's serve_policy.py and shuts it down with us."""

    def __init__(self, policy, port):
        ckpt = CKPT_ROOT / policy
        if not ckpt.is_dir():
            raise FileNotFoundError(f"no checkpoint at {ckpt}")
        log_path = pathlib.Path(__file__).parent / "policy_server.log"
        self.log = open(log_path, "w")
        print(f"starting policy server ({policy}); log: {log_path}")
        self.proc = subprocess.Popen(
            [
                str(OPENPI_DIR / ".venv/bin/python"),
                str(OPENPI_DIR / "scripts/serve_policy.py"),
                "--env", "LIBERO", "--port", str(port),
                "policy:checkpoint",
                "--policy.config", policy,
                "--policy.dir", str(ckpt),
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self.stop)

        print("  waiting for the checkpoint to load", end="", flush=True)
        for _ in range(180):
            if self.proc.poll() is not None:
                raise RuntimeError(f"policy server died on startup; see {log_path}")
            if _port_open(port):
                print(" ready")
                return
            print(".", end="", flush=True)
            time.sleep(2)
        raise TimeoutError(f"policy server never came up; see {log_path}")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class RobotServer:
    def __init__(self, args: Args):
        if args.mode not in ("waypoint", "vla", "hybrid"):
            raise ValueError("--mode must be 'waypoint', 'vla', or 'hybrid'")
        self.args = args
        self.vla_mode = args.mode == "vla"
        self.cmd_q: "queue.Queue" = queue.Queue()
        self.running = True

        # Load the policy BEFORE the sim. pi0.5 needs ~13 GB of host RAM to
        # restore, and on a 31 GB box with no swap, doing it second gets the
        # process OOM-killed mid-checkpoint.
        self.policy_proc = None
        self.policy = None
        if args.mode in ("vla", "hybrid"):
            self._connect_policy()

        suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
        self.task = suite.get_task(args.task_id)
        bddl = (
            pathlib.Path(get_libero_path("bddl_files"))
            / self.task.problem_folder
            / self.task.bddl_file
        )
        res = VLA_RESOLUTION if args.mode in ("vla", "hybrid") else 128
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=res,
            camera_widths=res,
            controller="OSC_POSE" if self.vla_mode else "JOINT_POSITION",
            # Idling between commands would otherwise burn through robosuite's
            # 1000-step horizon and terminate the episode.
            ignore_done=True,
            horizon=10_000_000,
        )
        self.env.seed(args.seed)
        self.obs = self.env.reset()

        self.action_dim = self.env.env.action_dim
        self.grip_idx = self.action_dim - 1
        self.gripper = GRIPPER_OPEN
        self.viewer = None

        self.scene = Scene(self.env, ignore_bodies=list(self.env.env.obj_of_interest))
        self.q_home = self.scene.arm_qpos()
        self.planner = RRTConnect(
            self.scene,
            SimpleNamespace(
                rrt_max_iters=args.rrt_max_iters,
                rrt_step=args.rrt_step,
                collision_check_step=args.collision_check_step,
                force_rrt=args.mode == "hybrid",
            ),
            np.random.default_rng(args.seed),
        )

    def _connect_policy(self):
        from openpi_client import websocket_client_policy

        if not _port_open(self.args.policy_port):
            if not self.args.auto_policy:
                raise RuntimeError(
                    f"nothing on :{self.args.policy_port} and --no-auto-policy was set"
                )
            self.policy_proc = PolicyServerProcess(self.args.policy, self.args.policy_port)
        else:
            print(f"reusing policy server already on :{self.args.policy_port}")
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            HOST, self.args.policy_port
        )

    # ------------------------------------------------------------- plumbing
    def _sync(self):
        if self.viewer is not None:
            self.viewer.sync()

    def _step(self, action):
        self.obs, _, _, _ = self.env.step(action)
        self._sync()

    def idle(self):
        a = np.zeros(self.action_dim)
        a[self.grip_idx] = self.gripper
        self._step(a)

    def _hold(self, steps):
        for _ in range(steps):
            self.idle()

    def gripper_forces(self):
        """Instantaneous simulated forces, sampled on the simulation thread."""
        import mujoco

        robot = self.env.env.robots[0]
        sim = self.env.sim
        model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        actuators = {
            name: float(data.actuator_force[sim.model.actuator_name2id(name)])
            for name in robot.gripper.actuators
        }
        contacts = {}
        for side in ("left_finger", "right_finger"):
            geoms = {sim.model.geom_name2id(name) for name in robot.gripper.important_geoms[side]}
            normal_force = 0.0
            for index in range(data.ncon):
                contact = data.contact[index]
                if (contact.geom1 in geoms) != (contact.geom2 in geoms):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, wrench)
                    normal_force += max(0.0, float(wrench[0]))
            contacts[side] = normal_force
        return {
            "sim_time_s": float(data.time),
            "finger_actuator_force_N": actuators,
            "finger_contact_normal_force_N": contacts,
            "wrist_force_sensor_frame_N": robot.ee_force.tolist(),
            "wrist_torque_sensor_frame_Nm": robot.ee_torque.tolist(),
        }

    # ------------------------------------------------------- motion: waypoint
    def _switch_controller(self, vla):
        """Replace only the controller, preserving the simulator and grasp."""
        if self.vla_mode == vla:
            return
        from robosuite.controllers import load_controller_config
        from robosuite.utils.buffers import DeltaBuffer

        robot = self.env.env.robots[0]
        robot.controller_config = load_controller_config(
            default_controller="OSC_POSE" if vla else "JOINT_POSITION"
        )
        robot._load_controller()
        robot.controller.update_base_pose(robot.base_pos, robot.base_ori)
        robot.controller.reset_goal()
        robot.recent_actions = DeltaBuffer(dim=robot.action_dim)
        self.env.env._action_dim = sum(r.action_dim for r in self.env.env.robots)
        self.action_dim = self.env.env.action_dim
        self.grip_idx = self.action_dim - 1
        self.vla_mode = vla

    def _drive_to_q(self, q_target, max_steps=300, tol=0.004):
        for _ in range(max_steps):
            err = q_target - self.scene.arm_qpos()
            if np.max(np.abs(err)) < tol:
                return True
            a = np.zeros(self.action_dim)
            a[:7] = np.clip(err / CTRL_DELTA_MAX, -1.0, 1.0)
            a[self.grip_idx] = self.gripper
            self._step(a)
        return False

    def _goto_xyz_rrt(self, target, target_ori=None):
        q_cur = self.scene.arm_qpos()
        q_goal, ik_err = self.scene.ik(target, q_init=q_cur, target_ori=target_ori)
        if ik_err > 0.05:
            return f"IK failed: closest reachable point is {ik_err*100:.1f}cm away"
        path = self.planner.plan(q_cur, q_goal)
        if path is None:
            return "no collision-free path to that pose"
        for q in path[1:]:
            if not self._drive_to_q(q):
                return "joint tracking failed; motion stopped"
        self._hold(self.args.settle_steps)
        err = np.linalg.norm(self.scene.eef_pos() - target)
        return f"moved ({len(path)} configs, eef error {err*100:.1f}cm)"

    def pick_object(self, name):
        if self.args.mode == "vla":
            return "pick requires --mode hybrid or --mode waypoint"
        objects = self.env.env.objects_dict
        if name not in objects:
            return f"unknown object {name!r}; known: {sorted(objects)}"
        self._switch_controller(False)
        # Permit intended gripper contact with the selected object only.
        self.scene = Scene(self.env, ignore_bodies=[name])
        self.planner.s = self.scene
        self.gripper = GRIPPER_OPEN
        self._hold(20)
        start = np.array(self.obs[f"{name}_pos"], dtype=float)
        # Keep the fingers pointing down instead of letting position-only IK
        # rotate the wrist while descending into the grasp.
        orientation = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        for label, dz in [("approach", 0.15), ("descend", 0.035)]:
            target = np.array(self.obs[f"{name}_pos"], dtype=float)
            target[2] += dz
            result = self._goto_xyz_rrt(target, target_ori=orientation)
            print(f"{label}: {result}", flush=True)
            if not result.startswith("moved") or np.linalg.norm(self.scene.eef_pos() - target) > 0.025:
                return f"pick stopped at {label}: {result}"
        self.gripper = GRIPPER_CLOSE
        self._hold(30)
        obj = objects[name]
        domain = self.env.env
        if not domain._check_grasp(domain.robots[0].gripper, obj.contact_geoms):
            return "pick failed: no two-finger contact; VLA has not been activated"
        target = self.scene.eef_pos()
        target[2] += 0.12
        result = self._goto_xyz_rrt(target, target_ori=orientation)
        lifted = float(self.obs[f"{name}_pos"][2] - start[2])
        if not result.startswith("moved") or lifted < 0.05 or not domain._check_grasp(domain.robots[0].gripper, obj.contact_geoms):
            return f"pick failed during lift ({lifted:.3f} m): {result}"
        return f"RRT grasp confirmed: holding {name}, lifted {lifted:.3f} m. Ready for vla command."

    # ------------------------------------------------------------ motion: osc
    def _goto_xyz_osc(self, target, max_steps=400, tol=0.005):
        """Cartesian servoing. OSC_POSE has no planner, so this is a straight line."""
        for _ in range(max_steps):
            err = np.asarray(target) - self.obs["robot0_eef_pos"]
            if np.linalg.norm(err) < tol:
                break
            a = np.zeros(self.action_dim)
            a[:3] = np.clip(err / OSC_POS_SCALE, -1.0, 1.0)
            a[self.grip_idx] = self.gripper
            self._step(a)
        self._hold(5)
        err = np.linalg.norm(np.asarray(target) - self.obs["robot0_eef_pos"])
        return f"moved (straight line, eef error {err*100:.1f}cm)"

    def goto_xyz(self, target):
        target = np.asarray(target, dtype=float)
        if self.vla_mode:
            return self._goto_xyz_osc(target)
        return self._goto_xyz_rrt(target)

    # --------------------------------------------------------------- shuffle
    def shuffle_objects(self, seed=None, names=None):
        """Permute the xy placements of the graspable objects among each other.

        Each object keeps its own z and orientation and only inherits another
        object's xy, so every placement was a valid one a moment ago and nothing
        ends up inside the table. 'reset' puts the original layout back.
        """
        sim = self.env.sim
        if names is None:
            # everything with a free joint except the goal container
            container = list(self.env.env.obj_of_interest[1:])
            names = [n for n in self.env.env.objects_dict if n not in container]
        names = sorted(names)
        adr = {}
        for n in names:
            jname = f"{n}_joint0"
            if jname not in sim.model.joint_names:
                known = sorted(self.env.env.objects_dict)
                return f"'{n}' has no free joint. known objects: {known}"
            jid = sim.model.joint_name2id(jname)
            adr[n] = (sim.model.jnt_qposadr[jid], sim.model.jnt_dofadr[jid])
        if len(names) < 2:
            return "need at least two objects to permute"

        rng = np.random.default_rng(seed)
        xy = np.array([sim.data.qpos[adr[n][0] : adr[n][0] + 2].copy() for n in names])
        while True:  # a derangement, so every object really moves
            perm = rng.permutation(len(names))
            if not np.any(perm == np.arange(len(names))):
                break

        for i, n in enumerate(names):
            q, v = adr[n]
            sim.data.qpos[q : q + 2] = xy[perm[i]]
            sim.data.qvel[v : v + 6] = 0.0  # drop whatever momentum they had
        sim.forward()
        self._hold(self.args.settle_steps * 5)  # let them settle onto the table

        return "shuffled:\n" + "\n".join(
            f"  {n:26s} -> {names[perm[i]]}'s spot" for i, n in enumerate(names)
        )

    # ------------------------------------------------------------------- vla
    def _vla_observation(self, instruction):
        from openpi_client import image_tools

        img = np.ascontiguousarray(self.obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(self.obs["robot0_eye_in_hand_image"][::-1, ::-1])
        return {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, VLA_RESIZE, VLA_RESIZE)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist, VLA_RESIZE, VLA_RESIZE)
            ),
            "observation/state": np.concatenate(
                (
                    self.obs["robot0_eef_pos"],
                    _quat2axisangle(self.obs["robot0_eef_quat"]),
                    self.obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(instruction),
        }

    def run_vla(self, instruction, max_steps):
        if self.policy is None:
            return "no policy connected -- restart with --mode hybrid or --mode vla"
        if max_steps < 1:
            return "max_steps must be positive"
        self._switch_controller(True)

        plan = collections.deque()
        t0 = time.time()
        for step in range(max_steps):
            if not plan:
                chunk = self.policy.infer(self._vla_observation(instruction))["actions"]
                plan.extend(chunk[: self.args.replan_steps])
            action = np.asarray(plan.popleft(), dtype=float)
            self._step(action.tolist())
            # Keep manual gripper commands consistent with what the policy did.
            self.gripper = float(action[self.grip_idx])
        # Benchmark success may differ from the instruction. Keep querying the
        # policy so its behavior after grasping / placement can be observed.
        return (
            f"ran {max_steps} steps ({time.time()-t0:.1f}s), step limit reached\n"
            f'  instruction: "{instruction}"'
        )

    # ------------------------------------------------------------- dispatch
    def handle(self, line: str) -> str:
        try:
            parts = shlex.split(line.strip())
        except ValueError as e:
            return f"could not parse command: {e}"
        if not parts:
            return ""
        cmd, rest = parts[0].lower(), parts[1:]

        try:
            if cmd in ("help", "?"):
                return HELP

            if cmd == "forces":
                return json.dumps(self.gripper_forces(), indent=2)

            if cmd == "state":
                out = [f"task: {self.task.language}", f"mode: {self.args.mode}",
                       f"controller: {'OSC_POSE (VLA)' if self.vla_mode else 'JOINT_POSITION (RRT)'}"]
                for k in sorted(self.obs):
                    if k.endswith("_pos") and not k.startswith("robot0") and "_to_robot0" not in k:
                        out.append(f"  {k[:-4]:26s} {np.round(self.obs[k],4).tolist()}")
                out.append(f"  {'robot0_eef':26s} {np.round(self.obs['robot0_eef_pos'],4).tolist()}")
                out.append(f"  gripper: {'closed' if self.gripper > 0 else 'open'}")
                return "\n".join(out)

            if cmd == "pick":
                if len(rest) != 1:
                    return "usage: pick OBJECT (for example: pick milk_1)"
                return self.pick_object(rest[0])

            if cmd == "success":
                return f"success = {self.env.check_success()}"

            if cmd == "vla":
                if not rest:
                    return 'usage: vla "INSTRUCTION" [max_steps]'
                steps = self.args.vla_max_steps
                if len(rest) > 1 and rest[-1].isdigit():
                    steps = int(rest[-1])
                    rest = rest[:-1]
                return self.run_vla(" ".join(rest), steps)

            if cmd == "shuffle":
                digits = [r for r in rest if r.lstrip("-").isdigit()]
                objs = [r for r in rest if r not in digits]
                return self.shuffle_objects(
                    int(digits[0]) if digits else None, objs or None
                )

            if cmd == "move":
                if len(rest) != 3:
                    return "usage: move X Y Z"
                return self.goto_xyz([float(v) for v in rest])

            if cmd == "moveto":
                if not rest:
                    return "usage: moveto OBJECT [dz]"
                key = f"{rest[0]}_pos"
                if key not in self.obs:
                    have = [k[:-4] for k in self.obs if k.endswith("_pos") and "_to_robot0" not in k]
                    return f"unknown object '{rest[0]}'. known: {have}"
                p = np.array(self.obs[key], dtype=float)
                p[2] += float(rest[1]) if len(rest) > 1 else 0.12
                return self.goto_xyz(p)

            if cmd == "lift":
                p = np.array(self.obs["robot0_eef_pos"], dtype=float)
                p[2] += float(rest[0]) if rest else 0.15
                return self.goto_xyz(p)

            if cmd == "grasp":
                self.gripper = GRIPPER_CLOSE
                self._hold(30)
                return "gripper closed"

            if cmd == "release":
                self.gripper = GRIPPER_OPEN
                self._hold(30)
                return "gripper open"

            if cmd == "home":
                if self.vla_mode:
                    return "home needs joint control; restart with --mode waypoint"
                path = self.planner.plan(self.scene.arm_qpos(), self.q_home)
                if path is None:
                    return "no path home"
                for q in path[1:]:
                    self._drive_to_q(q)
                self._hold(self.args.settle_steps)
                return "at home"

            if cmd == "run":
                if not rest:
                    return "usage: run FILE.json"
                wps = json.loads(pathlib.Path(rest[0]).read_text())
                out = []
                for wp in wps:
                    self.gripper = (
                        GRIPPER_CLOSE if wp.get("gripper", "open") == "close" else GRIPPER_OPEN
                    )
                    out.append(f"[{wp.get('name','?')}] " + self.goto_xyz(wp["pos"]))
                out.append(f"success = {self.env.check_success()}")
                return "\n".join(out)

            if cmd == "reset":
                self._switch_controller(self.args.mode == "vla")
                self.obs = self.env.reset()
                self.action_dim = self.env.env.action_dim
                self.grip_idx = self.action_dim - 1
                self.scene = Scene(self.env, ignore_bodies=list(self.env.env.obj_of_interest))
                self.planner.s = self.scene
                self.q_home = self.scene.arm_qpos()
                self.gripper = GRIPPER_OPEN
                self._sync()
                return "task reset"

            if cmd == "quit":
                self.running = False
                return "shutting down"

            return f"unknown command '{cmd}' -- try 'help'"

        except Exception as e:  # a bad command must not kill the server
            return f"error: {type(e).__name__}: {e}"

    # ------------------------------------------------------------ networking
    def serve_forever(self, port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, port))
        srv.listen(5)
        srv.settimeout(0.5)
        while self.running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()
        srv.close()

    def _client(self, conn):
        with conn:
            data = conn.recv(65536).decode("utf-8", "replace")
            if not data:
                return
            reply_q: "queue.Queue" = queue.Queue()
            self.cmd_q.put((data, reply_q))
            try:
                reply = reply_q.get(timeout=3600)
            except queue.Empty:
                reply = "timed out waiting for the sim"
            conn.sendall(reply.encode())

    # -------------------------------------------------------------- mainloop
    def run(self):
        threading.Thread(target=self.serve_forever, args=(self.args.port,), daemon=True).start()
        print(f"\ntask [{self.args.task_id}]: {self.task.language}")
        print(f"mode: {self.args.mode}  |  listening on {HOST}:{self.args.port}")
        if self.vla_mode:
            print(f'  python robot_cli.py vla "{self.task.language}"')
        else:
            print("  python robot_cli.py state")
        print()

        try:
            if self.args.viewer:
                import mujoco.viewer

                with mujoco.viewer.launch_passive(
                    self.env.sim.model._model, self.env.sim.data._data
                ) as v:
                    self.viewer = v
                    while self.running and v.is_running():
                        self._pump()
            else:
                while self.running:
                    self._pump()
        finally:
            if self.policy_proc:
                self.policy_proc.stop()
            self.env.close()

    def _pump(self):
        try:
            line, reply_q = self.cmd_q.get_nowait()
        except queue.Empty:
            self.idle()
            time.sleep(0.001)
            return
        print(f"> {line.strip()}")
        reply = self.handle(line)
        print(reply)
        reply_q.put(reply)


if __name__ == "__main__":
    RobotServer(tyro.cli(Args)).run()
