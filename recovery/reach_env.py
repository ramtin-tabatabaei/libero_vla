"""Task-conditioned PPO recovery on LIBERO with physical collision penalties.

Movable targets use contact, stable grasp and lift indicators. Articulated
objects use their BDDL open/close/on/off predicates, aiming at the moving
handle. Reach is a fallback or explicit comparison mode. This is a recovery
skill, not a complete placement or multi-object task planner.
"""
import dataclasses
import pathlib
from typing import Optional

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

import subgoal_planner as sp
from waypoint_rrt import GRIPPER_CLOSE, GRIPPER_OPEN

from recovery import scene as scene_tools
from recovery.reward import RewardConfig, alignment_error, step_reward

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
"""Every LIBERO suite, in the order build_waypoint_library.py lists them."""

ACTION_DIM = 7
OBSTACLE_COUNT = 4
OBS_DIM = 56 + 7 * OBSTACLE_COUNT
OBS_SCHEMA = "collision_geometry_v2"

ARM_SLICE = slice(1, 8)
"""Arm joints inside a flat init state. Verified identical on all 130 tasks,
whose flat state length varies from 45 to 123 -- robosuite lays the robot's
qpos down first, so the slice does not move when the scene does."""
GRIPPER_SLICE = slice(8, 10)

GRASP, ARTICULATE, REACH = "grasp", "articulate", "reach"


# ================================================================== task setup


@dataclasses.dataclass
class ReachConfig:
    horizon: int = 300
    """Control steps before truncation. A grasp needs the approach, the close
    and the lift, where a bare reach needed only the approach -- the converged
    reach policy finished in 26 steps, so 200 was ample for it and is not for
    this."""
    success_radius: float = 0.03
    """Grip site to grasp point, in metres. Matches subgoal_planner's
    `reach_tolerance`. Only decides success in `reach` mode; in the other two
    it just marks the approach as done for the shaping term."""
    lift_target: float = 0.05
    """How far the object must come up off its resting height. Matches
    run_vla.py's `drop_min_lift`, which is the height the repo already uses to
    decide that a grasp is real and not a nudge."""
    joint_fraction: float = 0.25
    """Share of an articulated joint's own travel that counts as having worked
    it. A fraction rather than an absolute, because a drawer slides 16 cm and a
    microwave door swings two radians."""
    success_hold: int = 5
    """Consecutive steps the finished condition must hold -- a quarter second
    at 20 Hz. Two was enough for a reach; a grasp needs longer, because the
    instant a lift begins is exactly when a bad grasp slips."""
    settle_steps: int = 10
    """The sim drops objects on reset; hold still until they land."""
    arm_noise: float = 0.15
    """Uniform radians added per arm joint at reset. A recovery controller is
    entered from wherever the VLA abandoned the arm, not from the benchmark's
    home pose, so the start distribution has to be wider than the benchmark's."""
    randomise_gripper: bool = True
    """Start half the episodes with the jaws shut, as they are after a drop."""
    grasp_depth: float = 0.02
    """How far below an object's top face to aim, for objects narrow enough to
    pinch. Passed straight to World.grasp_candidates."""
    mode: Optional[str] = None
    """Force one of grasp/articulate/reach. None picks per task from the scene."""
    contact_hold: int = 3
    """Consecutive steps of contact before it counts as a *stable* grasp rather
    than a touch. The repo's own grasp predicate latches on two; three, at
    20 Hz, is 150 ms of both pads staying shut."""

    # --- scene randomisation ---
    shuffle_objects: bool = True
    """Permute the graspable objects' xy placements among themselves at reset,
    so the target is somewhere different every episode and the policy cannot
    memorise one approach. A derangement, so every shuffled object really
    moves -- the same trick shuffle_eval.py uses to test the VLA."""
    shuffle_max_span: float = 0.15
    """Only objects smaller than this join the permutation. A basket or a plate
    is a surface other things stand on; dropping one where a 6 cm can was
    leaves it interpenetrating its neighbours."""

    reward: RewardConfig = dataclasses.field(default_factory=RewardConfig)

    def validate(self):
        if self.horizon < 1:
            raise ValueError("Horizon must be at least one control step.")
        if self.success_radius <= 0 or self.lift_target <= 0:
            raise ValueError("Success radius and lift target must be greater than zero.")
        if not 0 < self.joint_fraction <= 1:
            raise ValueError("Joint fraction must lie in (0, 1].")
        if self.success_hold < 1 or self.contact_hold < 1:
            raise ValueError("Hold counts must be at least one step.")
        if self.settle_steps < 0:
            raise ValueError("Settle steps must not be negative.")
        if self.arm_noise < 0:
            raise ValueError("Arm noise must not be negative.")
        if self.shuffle_max_span <= 0:
            raise ValueError("Shuffle span limit must be greater than zero.")
        if self.mode is not None and self.mode not in (GRASP, ARTICULATE, REACH):
            raise ValueError("Mode must be one of grasp, articulate, reach, or None.")
        self.reward.validate()


def task_list(suites=SUITES, suite=None, task_id=None):
    """Every (suite, task_id) pair the request names, in benchmark order."""
    names = [suite] if suite else list(suites)
    for name in names:
        if name not in SUITES:
            raise ValueError("Unknown suite: " + str(name) + ". Choose one of " + ", ".join(SUITES) + ".")
    out = []
    for name in names:
        bench = benchmark.get_benchmark_dict()[name]()
        ids = [task_id] if task_id is not None else range(bench.n_tasks)
        for i in ids:
            if not 0 <= i < bench.n_tasks:
                raise ValueError("Task %d is outside %s, which has %d tasks." % (i, name, bench.n_tasks))
            out.append((name, int(i)))
    return out


def build_env(suite_name, task_id, horizon, cameras=False, resolution=256, viewer=False):
    """A renderer-free ControlEnv -- no cameras, no GL, ~10 ms a step.

    Cameras are the entire cost of a LIBERO step and this policy never looks at
    one, so they are switched off rather than rendered and discarded. `ignore_done`
    keeps robosuite's own horizon from ending episodes underneath us; truncation
    is this module's job so that PPO can tell it apart from real termination.

    `hard_reset=False` is not just a speed knob. A hard reset rebuilds the whole
    MjSim, which leaves anything holding the old `sim`, `model` or `data` --
    `World` caches all three -- pointing at freed memory, and the next site
    lookup dies with "'MjSim' object has no attribute 'data'". A soft reset
    keeps the handles valid, and `set_init_state` restores the full flat state
    afterwards anyway, so nothing is lost by it.
    """
    bench = benchmark.get_benchmark_dict()[suite_name]()
    task = bench.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = ControlEnv(
        bddl_file_name=str(bddl),
        controller="OSC_POSE",
        use_camera_obs=cameras,
        has_renderer=viewer,
        has_offscreen_renderer=cameras,
        camera_names=["agentview"] if cameras else [],
        camera_heights=resolution,
        camera_widths=resolution,
        ignore_done=True,
        hard_reset=False,
        horizon=max(horizon * 4, 1000),
    )
    return env, task, np.asarray(bench.get_task_init_states(task_id), dtype=float)


# =============================================================== pure geometry


def grasp_point(world, name, depth):
    """Where on the target a gripper would actually go.

    `World.grasp_candidates` sorts by reach from the robot base and its first
    entry is the best grasp; for a region site the bounds collapse to a point
    and the offset is zero, so sites resolve to themselves.
    """
    return world.pos(name) + world.grasp_candidates(name, depth)[0][0]


def rotation_6d(matrix):
    """First two columns of a rotation, the continuous parameterisation.

    Quaternions double-cover and Euler angles wrap, and a network asked to
    regress either one has to learn around the discontinuity; the 6D form has
    none (Zhou et al., 2019).
    """
    return np.asarray(matrix, dtype=float)[:, :2].T.reshape(6)


def observation_vector(grip, grip_mat, target, span, joint_pos, joint_vel,
                       gripper_qpos, gripper_qvel, previous_action,
                       engaged=0.0, progress=0.0, align=0.0, contact_force=0.0, mode=GRASP, joint_remaining=0.0,
                       obstacles=None):
    """The 84 numbers the policy sees, all relative to the grip site.

    `engaged` and `progress` are what tell the policy which stage it is in --
    without them the value function cannot distinguish "hovering over the
    object" from "holding it", and those two states have very different
    returns. `align` and `contact_force` are the two terms the reward charges
    for that are otherwise invisible: a policy penalised for a misaligned wrist
    or for shoving the scenery, and unable to observe either, is being asked to
    guess.

    Kept free of the simulator so the packing can be tested without building a
    scene, and so the recovery runner can assemble it from a live rollout.
    """
    grip = np.asarray(grip, dtype=float)
    grip_mat = np.asarray(grip_mat, dtype=float).reshape(3, 3)
    target = np.asarray(target, dtype=float)
    delta = target - grip
    distance = float(np.linalg.norm(delta))
    joint_pos = np.asarray(joint_pos, dtype=float)
    if obstacles is None:
        obstacles = np.zeros((OBSTACLE_COUNT, 7))
    return np.concatenate([
        delta,                              # 3  world-frame vector to the goal
        grip_mat.T @ delta,                 # 3  the same vector the wrist feels
        [distance],                         # 1
        rotation_6d(grip_mat),              # 6  wrist attitude
        np.cos(joint_pos),                  # 7
        np.sin(joint_pos),                  # 7
        np.asarray(joint_vel, dtype=float), # 7
        np.asarray(gripper_qpos, dtype=float),   # 2
        np.asarray(gripper_qvel, dtype=float),   # 2
        np.asarray(span, dtype=float),      # 3  how big the target is
        np.asarray(previous_action, dtype=float),  # 7
        [float(engaged), float(progress)],  # 2  which stage the episode is in
        [float(align), float(contact_force)],  # 2  what the reward is charging for
        [float(mode == m) for m in (GRASP, ARTICULATE, REACH)],
        [float(joint_remaining)],
        np.asarray(obstacles, dtype=float).reshape(7 * OBSTACLE_COUNT),
    ]).astype(np.float32)


class ScriptedPick:
    """A four-stage top-down pick: above, down, close, lift.

    This is the floor PPO has to clear, and it exists because the obvious
    baseline is not one. Servoing straight at the grasp point -- which is what
    `RobotServer._goto_xyz_osc` does, and what an earlier version of this file
    used -- approaches on the diagonal into a point 2 cm inside the object, so
    the open fingers catch its side and bulldoze it: measured on the alphabet
    soup, the can slid 42 cm across the table while the grip site stalled 4.7 cm
    away and never got close enough to close. Going over the object first and
    coming straight down succeeds 6 times out of 6 in about 43 steps.

    Batched, because `evaluate.py` runs it across a vector of envs, each of
    which is at its own stage.
    """

    ABOVE, DESCEND, CLOSE = 0, 1, 2

    def __init__(self, num_envs=1, approach=0.10, above_tolerance=0.015,
                 descend_tolerance=0.012):
        self.approach = approach
        self.above_tolerance = above_tolerance
        self.descend_tolerance = descend_tolerance
        self.phase = np.zeros(int(num_envs), dtype=int)

    def reset(self, mask=None):
        """Send the named envs back to the start of the sequence."""
        if mask is None:
            self.phase[:] = self.ABOVE
        else:
            self.phase[np.asarray(mask, dtype=bool)] = self.ABOVE

    def act(self, obs):
        obs = np.atleast_2d(np.asarray(obs, dtype=float))
        actions = np.zeros((obs.shape[0], ACTION_DIM))
        delta = obs[:, :3]
        engaged = obs[:, 48] > 0.5
        above = delta + np.array([0.0, 0.0, self.approach])

        for i in range(obs.shape[0]):
            if engaged[i]:
                actions[i] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, GRIPPER_CLOSE]
                continue
            if self.phase[i] == self.ABOVE:
                actions[i, :3] = np.clip(above[i] / 0.05, -1.0, 1.0)
                actions[i, -1] = GRIPPER_OPEN
                if np.linalg.norm(above[i]) < self.above_tolerance:
                    self.phase[i] = self.DESCEND
            elif self.phase[i] == self.DESCEND:
                actions[i, :3] = np.clip(delta[i] / 0.05, -1.0, 1.0)
                actions[i, -1] = GRIPPER_OPEN
                if np.linalg.norm(delta[i]) < self.descend_tolerance:
                    self.phase[i] = self.CLOSE
            else:
                # Ease off the position command while closing, so the fingers
                # settle on the object instead of driving through it.
                actions[i, :3] = 0.2 * np.clip(delta[i] / 0.05, -1.0, 1.0)
                actions[i, -1] = GRIPPER_CLOSE
        return actions


# ============================================================ the environment


class ReachEnv:
    """One LIBERO scene as a pick-up MDP.

    Returns the five-tuple (obs, reward, terminated, truncated, info) rather
    than gym's older four-tuple on purpose: PPO must bootstrap the value
    function through a horizon cut-off and must not bootstrap through a real
    success, and collapsing the two into one `done` flag is the single easiest
    way to get a silently worse policy.
    """

    def __init__(self, suite_name, task_id, config=None, seed=0, cameras=False, viewer=False):
        self.config = config or ReachConfig()
        self.config.validate()
        self.suite_name = suite_name
        self.task_id = int(task_id)
        self.cameras = cameras
        self.viewer = viewer
        self.env, self.task, self.init_states = build_env(
            suite_name, task_id, self.config.horizon, cameras=cameras, viewer=viewer)
        self.env.seed(int(seed))
        self.env.reset()
        self.rng = np.random.default_rng(seed)
        self.world = sp.World(self.env)
        self.target_name = list(self.env.env.obj_of_interest)[0]
        self.language = self.task.language

        self._configure_task()

        model = self.env.sim.model
        self._joint_limits = np.array(
            [model.jnt_range[model.joint_name2id(name)] for name in
             ["robot0_joint%d" % i for i in range(1, 8)]], dtype=float)

        # Contacts with the target itself are what a grasp IS, so they are
        # exempted from the collision penalty; everything else the robot hits
        # is charged for.
        self.contacts = scene_tools.ContactMonitor(self.world)
        self.contacts.allow(self._allowed_geoms(), self._target_geoms())

        self.slots = scene_tools.free_joint_slots(self.env)
        self.shuffle_names = []
        if self.config.shuffle_objects:
            self.shuffle_names = scene_tools.shufflable(
                self.world, list(self.slots), self.config.shuffle_max_span)

        self.steps = 0
        self.streak = 0
        self.contact_streak = 0
        self.awarded = (False, False, False)
        self.previous_action = np.zeros(ACTION_DIM)
        self.previous_distance = 0.0
        self.initial_height = 0.0
        self.initial_qpos = 0.0
        self.last_force = 0.0

    # --- classification ---------------------------------------------------

    def _configure_task(self):
        self.goal_predicate = None
        self.goal_test = None
        self.joint_goal = None
        self.joint_direction = 1.0
        self.joint_threshold = 0.0
        tests = dict(open="is_open", close="is_close", turnon="turn_on", turnoff="turn_off")
        for goal in self.env.env.parsed_problem["goal_state"]:
            if len(goal) == 2 and goal[0].lower() in tests:
                if (goal[1] == self.target_name or (not self.world.is_site(self.target_name)
                        and self.world.parent_body(goal[1]) == self.target_name)):
                    self.target_name = goal[1]
                    self.goal_predicate = goal[0].lower()
                    self.goal_test = tests[self.goal_predicate]
                    break
        self.mode, self.joint_name = self._classify()
        scene_tools.enable_robot_collisions(self.world)

    def _classify(self):
        forced = self.config.mode
        joint = self.world.articulated_joint(self.target_name)
        if forced == REACH:
            return REACH, None
        if forced == ARTICULATE or (forced is None and joint is not None):
            if joint is None:
                raise ValueError("Target has no articulated joint: " + self.target_name)
            return ARTICULATE, joint
        if forced == GRASP or self.world.is_movable(self.target_name):
            if not self.world.is_movable(self.target_name):
                raise ValueError("Target is not graspable: " + self.target_name)
            return GRASP, None
        return REACH, None

    def _reset_measurements(self):
        self.initial_height = self._height() if self.mode == GRASP else 0.0
        self.initial_qpos = 0.0
        if self.mode == ARTICULATE:
            info = self.world.joint_info(self.joint_name)
            self.initial_qpos = info["qpos"]
            span = float(np.ptp(info["range"]))
            if self.goal_test:
                self.joint_goal = sp.target_qpos(self.world, self.target_name,
                                                self.joint_name, self.goal_test)
            else:
                self.joint_goal = max(info["range"], key=lambda q: abs(q - self.initial_qpos))
            self.joint_direction = float(np.sign(self.joint_goal - self.initial_qpos))
            self.joint_threshold = max(1e-6, min(self.config.joint_fraction * span,
                                               abs(self.joint_goal - self.initial_qpos)))
        self.contact_streak = 0
        self.contacts = scene_tools.ContactMonitor(self.world)
        self.contacts.allow(self._allowed_geoms(), self._target_geoms())
        self.contacts.install(self.env.sim, self.config.reward.force_cap)
        self.obstacles = scene_tools.CollisionObstacles(self.world, self.contacts, OBSTACLE_COUNT)

    # --- plumbing ---------------------------------------------------------

    def _target_geoms(self):
        """The target's own collision geoms, however the scene describes it."""
        if self.mode == ARTICULATE:
            return scene_tools.joint_geom_ids(self.world, self.joint_name)
        if self.world.is_movable(self.target_name):
            return scene_tools.geom_ids(
                self.world, self.env.env.objects_dict[self.target_name].contact_geoms)
        return scene_tools.body_geom_ids(self.world,
                                         self.world.parent_body(self.target_name))

    def _pad_geoms(self):
        """The two finger pads -- the only robot geoms allowed on the target."""
        important = self.env.env.robots[0].gripper.important_geoms
        names = list(important.get("left_fingerpad", [])) + list(
            important.get("right_fingerpad", []))
        return scene_tools.geom_ids(self.world, names)

    def _allowed_geoms(self):
        if self.mode == ARTICULATE:
            # Handles may be hooked or pushed by the fingers; the arm and palm
            # still incur collision costs, as do contacts with fixed panels.
            important = self.env.env.robots[0].gripper.important_geoms
            names = [g for key, values in important.items() if "finger" in key for g in values]
            return scene_tools.geom_ids(self.world, names)
        return self._pad_geoms()

    def _handle(self):
        info = self.world.joint_info(self.joint_name)
        travel = info["axis"] * self.joint_direction
        if not info["slide"]:
            radial = self.world.handle_grasp(self.joint_name)[0] - info["anchor"]
            travel = np.cross(info["axis"], radial) * self.joint_direction
        return self.world.handle_grasp(self.joint_name, travel)

    def _grasp_yaw(self):
        """The wrist yaw of the best grasp, for the alignment term."""
        if self.mode == ARTICULATE:
            return self._handle()[1]
        return self.world.grasp_candidates(self.target_name, self.config.grasp_depth)[0][1]

    def _sample_state(self):
        """A benchmark init state, with the arm and the objects both moved."""
        state = self.init_states[self.rng.integers(len(self.init_states))].copy()
        self.shuffle_mapping = {}
        if self.config.arm_noise > 0:
            noise = self.rng.uniform(-self.config.arm_noise, self.config.arm_noise, 7)
            state[ARM_SLICE] = np.clip(state[ARM_SLICE] + noise,
                                       self._joint_limits[:, 0], self._joint_limits[:, 1])
        if self.config.randomise_gripper and self.rng.random() < 0.5:
            state[GRIPPER_SLICE] = [0.0, 0.0]
        return state

    def _target(self):
        if self.mode == ARTICULATE:
            return self._handle()[0]
        return grasp_point(self.world, self.target_name, self.config.grasp_depth)

    def _height(self):
        body = self.env.env.obj_body_id[self.target_name]
        return float(self.env.sim.data.body_xpos[body][2])

    def _grasped(self):
        """The repo's canonical grasp predicate, as DropObject and StopOnGrasp use it."""
        domain = self.env.env
        obj = domain.objects_dict[self.target_name]
        return bool(domain._check_grasp(domain.robots[0].gripper, obj.contact_geoms))

    def _touching(self):
        return self.contacts.measure()[1]

    def _grip_mat(self):
        site = self.env.sim.model.site_name2id("gripper0_grip_site")
        return np.array(self.env.sim.data.site_xmat[site], dtype=float).reshape(3, 3)

    def _stage(self):
        """Everything the reward and the observation need, measured once.

        The three staged indicators mean different things per mode, because
        the modes are different problems: a drawer cannot be lifted and a can
        has no hinge. Contact and sustained contact carry over unchanged; only
        the final stage differs.
        """
        distance = float(np.linalg.norm(self._target() - self.world.eef()))
        touching = self._touching()
        engaged = self._grasped() if self.mode == GRASP else touching
        self.contact_streak = self.contact_streak + 1 if engaged else 0
        stable = self.contact_streak >= self.config.contact_hold
        joint_remaining = 0.0
        if self.mode == GRASP:
            lift = self._height() - self.initial_height
            progress = lift / self.config.lift_target
            final_stage = lift > self.config.lift_target
            finished = stable and final_stage
        elif self.mode == ARTICULATE:
            info = self.world.joint_info(self.joint_name)
            moved = self.joint_direction * (info["qpos"] - self.initial_qpos)
            progress = moved / self.joint_threshold
            joint_remaining = (self.joint_goal - info["qpos"]) / max(float(np.ptp(info["range"])), 1e-6)
            if self.goal_test:
                obj = self.env.env.get_object(self.world.parent_body(self.target_name))
                final_stage = bool(getattr(obj, self.goal_test)(info["qpos"]))
            else:
                final_stage = moved >= self.joint_threshold
            finished = final_stage
        else:
            touching = stable = False
            progress = 0.0
            final_stage = finished = distance < self.config.success_radius
        return dict(distance=distance, touching=bool(touching), stable=bool(stable),
                    progress=float(progress), finished=bool(finished),
                    engaged=bool(stable), final_stage=bool(final_stage),
                    joint_remaining=float(joint_remaining))

    def _observation(self, obs, stage, align, force):
        grip = self.world.eef()
        low, high = self.world.aabb(self.target_name)
        grip_mat = self._grip_mat()
        obstacles = self.obstacles.observe(grip, grip_mat)
        return observation_vector(
            grip, grip_mat, self._target(), np.asarray(high) - np.asarray(low),
            obs["robot0_joint_pos"], obs["robot0_joint_vel"],
            obs["robot0_gripper_qpos"], obs["robot0_gripper_qvel"],
            self.previous_action, engaged=stage["engaged"], progress=stage["progress"],
            # Scaled to roughly unit range so the running normaliser has
            # something sane to work with from the first batch.
            align=align / np.pi, contact_force=force / self.config.reward.force_cap,
            mode=self.mode, joint_remaining=stage["joint_remaining"], obstacles=obstacles)

    # --- the MDP ----------------------------------------------------------

    def reset(self):
        self.env.reset()
        # Cheap, and the one line that survives someone later turning hard
        # resets back on: World caches sim/model/data handles.
        self.world = sp.World(self.env)
        scene_tools.enable_robot_collisions(self.world)
        obs = self.env.set_init_state(self._sample_state())
        hold = np.zeros(ACTION_DIM)
        hold[-1] = (GRIPPER_CLOSE if self.config.randomise_gripper and self.rng.random() < 0.5
                    else GRIPPER_OPEN)
        for _ in range(self.config.settle_steps):
            obs, _, _, _ = self.env.step(hold)

        if self.config.shuffle_objects:
            # Benchmark states can start with objects intersecting the table.
            # Measure support heights only after the first settling pass.
            state = self.env.sim.get_state().flatten()
            self.shuffle_names = scene_tools.shufflable(
                self.world, list(self.slots), self.config.shuffle_max_span)
            shuffled, self.shuffle_mapping = scene_tools.shuffle_scene(
                self.world, state, self.slots, self.shuffle_names, self.rng)
            if self.shuffle_mapping:
                obs = self.env.set_init_state(shuffled)
                for _ in range(self.config.settle_steps):
                    obs, _, _, _ = self.env.step(hold)

        self._reset_measurements()

        self.steps = 0
        self.streak = 0
        self.contact_streak = 0
        self.awarded = (False, False, False)
        self.ever_engaged = False
        self.best_progress = 0.0
        self.collision_total = 0.0
        self.severe_steps = 0
        self.last_force = 0.0
        self.previous_action = np.zeros(ACTION_DIM)

        stage = self._stage()
        self.previous_distance = stage["distance"]
        align = alignment_error(self._grip_mat(), sp.down_yaw(self._grasp_yaw()))
        _, self.last_force = self.contacts.interval()
        return self._observation(obs, stage, align, self.last_force)

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(ACTION_DIM)
        if not np.isfinite(action).all():
            raise ValueError("Actions must be finite.")
        action = np.clip(action, -1.0, 1.0)
        self.contacts.clear()
        obs, _, _, _ = self.env.step(action)
        self.previous_action = action
        self.steps += 1

        stage = self._stage()
        self.streak = self.streak + 1 if stage["finished"] else 0
        done = self.streak >= self.config.success_hold
        # Whether the jaws ever closed on it, and how far it ever got, are the
        # diagnostics that separate "learning to approach" from "learning to
        # pick up". The final step's values cannot tell those apart: an episode
        # that grasped and dropped ends looking identical to one that never
        # touched the object.
        self.ever_engaged = self.ever_engaged or stage["engaged"]
        self.best_progress = max(self.best_progress, stage["progress"])

        forces, self.last_force = self.contacts.interval()
        self.severe_steps += int(self.last_force > self.config.reward.force_safe)
        self.collision_total += float(forces.sum())
        align = alignment_error(self._grip_mat(), sp.down_yaw(self._grasp_yaw()))

        reward, self.awarded, terms = step_reward(
            self.previous_distance, stage["distance"], self._grip_mat(),
            sp.down_yaw(self._grasp_yaw()),
            (stage["touching"], stage["stable"],
             done if self.config.reward.terminal_stage_bonus else stage["final_stage"]),
            self.awarded, forces, action, self.config.reward,
            contact_force=self.last_force, return_terms=True)
        self.previous_distance = stage["distance"]

        truncated = (not done) and self.steps >= self.config.horizon
        info = dict(distance=stage["distance"], success=bool(done), steps=self.steps,
                    engaged=stage["engaged"], progress=stage["progress"], mode=self.mode,
                    ever_engaged=self.ever_engaged, best_progress=self.best_progress,
                    align=align, contact_force=self.last_force,
                    collision_total=self.collision_total, severe_steps=self.severe_steps,
                    reward_terms=terms,
                    calibration=dict(
                        peak_force_n=self.last_force,
                        sum_bad_force_n=self.contacts.peak_bad_force_sum,
                        capped_bad_force_n=float(np.minimum(forces, self.config.reward.force_cap).sum()),
                        r_collision=terms["collision"], r_severe=terms["severe"],
                        r_collision_total=terms["collision"] + terms["severe"],
                        r_task=terms["progress"] + terms["alignment"] + terms["stages"],
                        severe_event=float(self.last_force > self.config.reward.force_safe)),
                    shuffle_mapping=self.shuffle_mapping,
                    goal_predicate=self.goal_predicate,
                    suite=self.suite_name, task_id=self.task_id)
        return (self._observation(obs, stage, align, self.last_force), reward,
                bool(done), bool(truncated), info)

    def render(self):
        """Draw one frame into the on-screen MuJoCo window.

        Needs MUJOCO_GL=glx: EGL is headless and cannot open a window, which is
        the same reason run.sh switches the backend on --render.
        """
        if not self.viewer:
            raise ValueError("Build the env with viewer=True before rendering to a window.")
        self.env.env.render()

    def frame(self):
        """The agentview image, turned upright.

        LIBERO renders the scene upside down -- run_vla.py flips both axes to
        match how the VLA's training data was rendered. A video only needs the
        vertical flip to look like the room it came from.
        """
        if not self.cameras:
            raise ValueError("Build the env with cameras=True before asking for a frame.")
        return np.ascontiguousarray(
            self.env.env._get_observations()["agentview_image"][::-1])

    def close(self):
        self.env.close()


# ============================================================== manual check


def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    args = parser.parse_args()

    env = ReachEnv(args.suite, args.task_id)
    print("task    :", env.language)
    print("target  :", env.target_name)
    print("mode    :", env.mode, "" if env.joint_name is None
          else "(joint %s, %.3f to count)" % (env.joint_name, env.joint_threshold))
    print("obs dim :", env.reset().shape)
    scripted = ScriptedPick()
    for episode in range(args.episodes):
        obs = env.reset()
        scripted.reset()
        total = 0.0
        while True:
            obs, reward, terminated, truncated, info = env.step(scripted.act(obs)[0])
            total += reward
            if terminated or truncated:
                break
        print("episode %d  steps %3d  distance %.3f  progress %.2f  return %7.2f  %s"
              % (episode, info["steps"], info["distance"], info["progress"], total,
                 "DONE" if info["success"] else "timed out"))
    env.close()


if __name__ == "__main__":
    _main()
