"""Evaluate Pi 0.5 on 40 tasks x 3 robot starts x 2 layouts x 10 trials.

    libero-vla-batch run --output experiments/pi05_40 --dry-run
    libero-vla-batch run --output experiments/pi05_40
    libero-vla-batch run --output experiments/pi05_40 --resume
    libero-vla-batch replay experiments/pi05_40/trials/<trial-id> --render

Start serve_pi05_libero.sh separately. The batch uses official benchmark initial
states, original task instructions and LIBERO's success predicate. Unsupported
scene perturbations are reported separately from policy failures. Failed/error
trials retain every recorded camera image, action, policy response and simulator
state. Source/configuration snapshots and compact results are kept for all runs.
"""
import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STARTS = ("normal", "small", "large")
LAYOUTS = ("original", "shuffled")
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
COMPLETE = {"success", "failed", "setup_unavailable", "invalid_start"}
SOURCE_FILES = ("pi05_benchmark.py", "studio_robot_start.py", "waypoint_demo.py",
                "waypoint_session.py", "waypoint_replay.py", "waypoint_rrt.py",
                "subgoal_planner.py", "serve_pi05_libero.py")


@dataclass
class Options:
    trials: int = 10
    seed: int = 7
    port: int = 8000
    suites: list = field(default_factory=lambda: list(SUITES))
    task_ids: list = field(default_factory=lambda: list(range(10)))
    starts: list = field(default_factory=lambda: list(STARTS))
    layouts: list = field(default_factory=lambda: list(LAYOUTS))
    resolution: int = 256
    settle_steps: int = 10
    max_steps: int | None = None
    replan_steps: int = 5
    image_every: int = 1
    save_video: bool = True
    keep_success_media: bool = False
    min_free_gib: float = 5.

    def validate(self):
        for values, allowed, name in ((self.suites, SUITES, "suites"),
                (self.task_ids, range(10), "task IDs"), (self.starts, STARTS, "robot starts"),
                (self.layouts, LAYOUTS, "layouts")):
            if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
                raise ValueError(f"Invalid or duplicate {name}: {values}")
        if not 1 <= self.trials <= 50:
            raise ValueError("Use 1–50 trials, each selecting a different official initial state")
        if (self.resolution < 32 or self.settle_steps < 0 or self.image_every < 1
                or not 1 <= self.replan_steps <= 10 or not 1 <= self.port <= 65535
                or (self.max_steps is not None and self.max_steps < 1)
                or not np.isfinite(self.min_free_gib) or self.min_free_gib < 0):
            raise ValueError("Invalid resolution, step count, port or disk reserve")


class SetupUnavailable(ValueError):
    pass


class DiskReserveError(RuntimeError):
    pass


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=json_array) + "\n")
    temporary.replace(path)


def json_array(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derived_seed(seed, suite, task, trial, stream):
    return int(np.random.SeedSequence([seed, SUITES.index(suite), task, trial, stream])
               .generate_state(1)[0])


def schedule(options):
    options.validate()
    return [dict(suite=suite, task_id=task, trial=trial, initial_state_index=trial,
                 robot_start=start, layout=layout,
                 model_seed=derived_seed(options.seed, suite, task, 0, 0),
                 environment_seed=derived_seed(options.seed, suite, task, trial, 0),
                 shuffle_seed=derived_seed(options.seed, suite, task, trial, 1),
                 robot_seed=derived_seed(options.seed, suite, task, trial, 2 if start == "small" else 3))
            for suite in options.suites for task in options.task_ids for trial in range(options.trials)
            for layout in options.layouts for start in options.starts]


def trial_id(spec):
    return (f"{spec['suite']}_t{spec['task_id']:03d}_r{spec['trial']:03d}"
            f"_{spec['robot_start']}_{spec['layout']}")


def check_disk(path, minimum_gib):
    free = shutil.disk_usage(path).free / 2**30
    if free < minimum_gib:
        raise DiskReserveError(f"Only {free:.2f} GiB free; stopping to preserve the {minimum_gib:g} GiB reserve")


def reset_environment(session, attempts=10):
    """Bound placement retries; preserve interrupts and unexpected reset errors."""
    from robosuite.utils.errors import RandomizationError
    for _ in range(attempts):
        try:
            return session.env.env.reset()
        except RandomizationError:
            continue
    raise SetupUnavailable(f"Scene placement failed after {attempts} reset attempts")


def make_session(suite, task_id, seed, resolution, render=False):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    import subgoal_planner as sp
    from waypoint_session import Session

    np.random.seed(seed)
    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    session = Session(sp.Args(task_suite_name=suite, task_id=task_id, seed=seed, settle_steps=0))
    session.task = task
    session.env = ControlEnv(bddl_file_name=str(bddl), controller="JOINT_POSITION",
        camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=resolution,
        camera_widths=resolution, use_camera_obs=True, has_renderer=render,
        has_offscreen_renderer=True, hard_reset=False, ignore_done=True, horizon=100000)
    try:
        session.env.seed(seed)
        reset_environment(session)
        session.attach(settle=False)
    except BaseException:
        session.env.close()
        raise
    return session


def restore(session, state, controller="JOINT_POSITION", integration=None):
    import mujoco
    from waypoint_demo import switch_controller
    sim = session.env.sim
    model, data = getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)
    mujoco.mj_resetData(model, data)
    session.env.set_init_state(np.asarray(state))
    if integration is not None:
        mujoco.mj_setState(model, data, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
        sim.forward()
    # Reset episode/controller bookkeeping as well as physics, without rebuilding the scene.
    domain = session.env.env
    domain.done, domain.timestep = False, 0
    if hasattr(domain, "cur_time"):
        domain.cur_time = float(session.env.sim.data.time)
    switch_controller(session.env, controller)
    session.attach(settle=False)
    session.executor.gripper = -1.


def integration_state(session):
    import mujoco
    sim = session.env.sim
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, specification), dtype=np.float64)
    mujoco.mj_getState(model, data, state, specification)
    return state


def save_state(path, session):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, state=session.env.sim.get_state().flatten(),
                                integration_state=integration_state(session))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def object_positions(session):
    return {name: session.world.pos(name).tolist() for name in session.world.domain.objects_dict}


def task_assets(session, output, suite, task_id):
    from libero.libero import get_libero_path
    folder = output / "assets" / suite / f"task_{task_id:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    xml = session.env.sim.model.get_xml()
    model = folder / "model.xml"
    if model.exists() and model.read_text() != xml:
        raise ValueError("Simulator model differs from the saved experiment: " + str(model))
    model.write_text(xml)
    bddl = Path(get_libero_path("bddl_files")) / session.task.problem_folder / session.task.bddl_file
    shutil.copy2(bddl, folder / "task.bddl")
    return dict(model_xml=str(model.relative_to(output)), model_sha256=digest(model),
                bddl=str((folder / "task.bddl").relative_to(output)), bddl_sha256=digest(bddl),
                instruction=session.task.language, task_name=session.task.name)


def prepare_layout(session, official, spec, options, output):
    """Cache each layout once so all three robot-start conditions share it."""
    from waypoint_replay import shuffle_objects, shuffle_positions

    base = output / "layouts" / spec["suite"] / f"task_{spec['task_id']:03d}" / f"trial_{spec['trial']:03d}"
    base.mkdir(parents=True, exist_ok=True)
    original_path = base / "original.npz"
    if not original_path.exists() or not (base / "original.json").exists():
        session.env.seed(spec["environment_seed"])
        # Keep the task's fixture geometry fixed. LIBERO resets can randomize
        # model.body_pos, which is not part of an official flat initial state.
        # Reset physics/controller bookkeeping through restore instead.
        restore(session, official, controller="OSC_POSE")
        for _ in range(options.settle_steps):
            session.env.step([0.] * 6 + [-1.])
        save_state(original_path, session)
        write_json(base / "original.json", dict(method="official_initial_state", **dict(spec, layout="original", robot_start="normal"),
            objects=object_positions(session), initial_state_sha256=hashlib.sha256(np.asarray(official).tobytes()).hexdigest()))
    with np.load(original_path, allow_pickle=False) as saved:
        original = saved["state"].copy()
        original_integration = saved["integration_state"].copy()
    restore(session, original, integration=original_integration)
    normal_qpos = session.executor.scene.arm_qpos()
    session.normal_start_qpos = normal_qpos.copy()
    path, info_path = base / (spec["layout"] + ".npz"), base / (spec["layout"] + ".json")
    unavailable = base / "shuffled.unavailable.json"
    if spec["layout"] == "shuffled" and unavailable.exists() and not path.exists():
        raise SetupUnavailable(json.loads(unavailable.read_text())["error"])
    if spec["layout"] == "shuffled" and (not path.exists() or not info_path.exists()):
        errors = []
        for attempt in range(8):
            seed = int(np.random.SeedSequence([spec["shuffle_seed"], attempt]).generate_state(1)[0])
            restore(session, original, integration=original_integration)
            try:
                method = "object_permutation"
                try:
                    mapping = shuffle_objects(session, seed=seed)
                except ValueError as exc:
                    errors.append(str(exc))
                    restore(session, original, integration=original_integration)
                    method = "source_relocation"
                    mapping = shuffle_positions(session, seed=seed)
                # Shuffling settles objects. Restore the original robot's joints,
                # fingers and velocities so layout alone does not alter its start.
                sim = session.env.sim
                state = sim.get_state().flatten().copy()
                nq = sim.model.nq
                robot = session.env.env.robots[0]
                joints = list(robot.robot_joints) + list(robot.gripper.joints)
                for name in joints:
                    joint = sim.model.joint_name2id(name)
                    q, v = int(sim.model.jnt_qposadr[joint]), int(sim.model.jnt_dofadr[joint])
                    state[1 + q] = original[1 + q]
                    state[1 + nq + v] = original[1 + nq + v]
                state[0] = original[0]
                restore(session, state)
                old = json.loads((base / "original.json").read_text())["objects"]
                moved = {name: float(np.linalg.norm(np.asarray(p)[:2] - np.asarray(old[name])[:2]))
                         for name, p in object_positions(session).items()}
                if max(moved.values(), default=0.) < .02:
                    raise ValueError("Shuffle did not move an object by at least 2 cm")
                if session.env.check_success():
                    raise ValueError("Shuffle already satisfies the task goal")
                save_state(path, session)
                write_json(info_path, dict(method=method, mapping=mapping, actual_seed=seed,
                    attempt=attempt, objects=object_positions(session), moved_xy_m=moved))
                break
            except ValueError as exc:
                errors.append(str(exc))
        else:
            restore(session, original, integration=original_integration)
            error = "No valid shuffled scene: " + "; ".join(errors[-3:])
            write_json(unavailable, dict(error=error, seed=spec["shuffle_seed"], attempts=errors))
            raise SetupUnavailable(error)
    with np.load(path, allow_pickle=False) as saved:
        restore(session, saved["state"], integration=saved["integration_state"])
    session.normal_start_qpos = normal_qpos.copy()
    info = json.loads(info_path.read_text())
    # Original metadata stores pairing information; expose only layout details here.
    info = {key: value for key, value in info.items() if key not in spec}
    info.update(state_file=str(path.relative_to(output)), state_file_sha256=digest(path))
    return info


class Recorder:
    """Stream evidence to disk; flush every transition, retaining partial errors."""
    def __init__(self, folder, session, options):
        import h5py
        self.folder, self.session, self.options = Path(folder), session, options
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.folder / "trajectory.h5", "x")
        self.file.attrs["schema"] = "pi05_40_trajectory_v1"
        self.file.attrs["camera_rotation_degrees"] = 180
        self.file.attrs["camera_keys"] = json.dumps(CAMERAS)
        self.file.attrs["integration_specification"] = "mjSTATE_INTEGRATION"
        self.index = (self.folder / "frames.jsonl").open("x", buffering=1)
        self.queries = (self.folder / "policy_queries.jsonl").open("x", buffering=1)
        self.video = None
        self.errors, self.observations, self.actions, self.last_image_step = [], 0, 0, None

    def append(self, name, value):
        value = np.asarray(value)
        if name not in self.file:
            self.file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                                     dtype=value.dtype, chunks=True, compression="gzip")
        data = self.file[name]
        data.resize(len(data) + 1, axis=0)
        data[-1] = value

    def action(self, action):
        self.append("actions", np.asarray(action, np.float32))
        self.actions += 1

    def observe(self, obs, goal, final=False):
        from waypoint_demo import observation
        inputs = observation(obs, self.session.task.language)
        step = self.observations
        self.append("states", self.session.env.sim.get_state().flatten())
        self.append("integration_states", integration_state(self.session))
        self.append("policy_states", np.asarray(inputs["observation/state"], np.float32))
        self.append("task_success", np.asarray(goal, dtype=bool))
        self.file.flush()
        self.observations += 1
        if step % self.options.image_every == 0 or final:
            self.images(obs, step)
        if step % 50 == 0:
            check_disk(self.folder, self.options.min_free_gib)

    def images(self, obs, step, phase="rollout"):
        if self.last_image_step == step and phase == "rollout":
            return
        from PIL import Image
        images, paths = [], {}
        for key in CAMERAS:
            image = np.ascontiguousarray(obs[key][::-1, ::-1])
            folder = self.folder / "frames" / key
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / (f"{step:06d}.png" if phase == "rollout" else f"{phase}.png")
            Image.fromarray(image).save(path)
            paths[key] = str(path.relative_to(self.folder))
            images.append(image)
        self.index.write(json.dumps(dict(step=step, phase=phase, sim_time=float(self.session.env.sim.data.time), images=paths)) + "\n")
        if phase == "rollout":
            self.last_image_step = step
        if self.options.save_video and phase == "rollout":
            try:
                import imageio.v2 as imageio
                if self.video is None:
                    fps = 1 / (self.session.env.env.control_timestep * self.options.image_every)
                    self.video = imageio.get_writer(self.folder / "rollout.mp4", fps=fps,
                                                    codec="libx264", macro_block_size=1)
                self.video.append_data(np.concatenate(images, axis=1))
            except Exception as exc:
                self.errors.append("Video: " + str(exc))
                self.options = Options(**dict(asdict(self.options), save_video=False))
                if self.video is not None:
                    try:
                        self.video.close()
                    except Exception:
                        pass
                    self.video = None

    def query(self, step, output):
        # Store full chunks, including predictions not executed by replanning.
        record = {key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in output.items()}
        self.queries.write(json.dumps(dict(step=step, response=record), allow_nan=False, default=json_array) + "\n")

    def close(self):
        self.file.close()
        self.index.close()
        self.queries.close()
        if self.video is not None:
            try:
                self.video.close()
            except Exception as exc:
                self.errors.append("Video close: " + str(exc))


def archive_incomplete(folder):
    if folder.exists():
        attempts = folder.parent / "_previous_attempts"
        attempts.mkdir(exist_ok=True)
        index = 1
        while (attempts / f"{folder.name}_{index:03d}").exists():
            index += 1
        folder.rename(attempts / f"{folder.name}_{index:03d}")


def run_case(session, official, spec, options, output, policy_metadata, assets, client_factory=None):
    from studio_robot_start import START_VARIATIONS, randomize_robot_start
    from waypoint_demo import PolicyConnection, observation, switch_controller
    client_factory = client_factory or PolicyConnection
    folder = output / "trials" / trial_id(spec)
    archive_incomplete(folder)
    folder.mkdir(parents=True)
    started, client, recorder = time.monotonic(), None, None
    result = dict(spec, trial_id=trial_id(spec), status="setup_error", success=None,
                  steps=0, scored=False)
    config = dict(schema="pi05_40_trial_v1", scenario=spec, options=asdict(options),
        experiment_root=str(output.resolve()),
        policy=policy_metadata, assets=assets, offset_ranges=START_VARIATIONS,
        action_contract="OSC_POSE, xyz + axis-angle deltas + gripper; replan every 5 actions",
        camera_keys=list(CAMERAS), camera_rotation_degrees=180,
        success_criterion="LIBERO task check_success(), first true after an action",
        layout=None, offset=None, prepared=False)
    write_json(folder / "config.json", config)
    interrupt, fatal = None, None
    try:
        config["layout"] = prepare_layout(session, official, spec, options, output)
        normal_position = session.executor.scene.eef_pos().copy()
        if spec["robot_start"] != "normal":
            try:
                randomize_robot_start(session, np.random.default_rng(spec["robot_seed"]),
                                      variation=spec["robot_start"])
            except ValueError as exc:
                raise SetupUnavailable(str(exc)) from exc
        actual = session.executor.scene.eef_pos()
        config["offset"] = dict(normal_position=normal_position.tolist(),
            position=actual.tolist(), distance_m=float(np.linalg.norm(actual - normal_position)),
            normal_joints=session.normal_start_qpos.tolist(), joints=session.executor.scene.arm_qpos().tolist())
        switch_controller(session.env, "OSC_POSE")
        config["prepared"] = True
        config["control_timestep"] = float(session.env.env.control_timestep)
        write_json(folder / "config.json", config)
        save_state(folder / "initial.npz", session)
        obs = session.env.env._get_observations(force_update=True)
        recorder = Recorder(folder, session, options)
        goal = bool(session.env.check_success())
        recorder.observe(obs, goal)
        if goal:
            result["status"] = "invalid_start"
            result["error"] = "Task goal already satisfied before policy execution"
        else:
            client = client_factory(options.port)
            metadata = client.connect()
            if metadata != policy_metadata:
                raise ValueError("Policy metadata changed since the batch started")
            actions = deque()
            limit = options.max_steps or MAX_STEPS[spec["suite"]]
            result.update(status="failed", success=False, scored=True)
            for step in range(limit):
                if not actions:
                    predicted = client.infer(observation(obs, session.task.language))
                    chunk = np.asarray(predicted["actions"], dtype=float)
                    if (chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) < options.replan_steps
                            or not np.isfinite(chunk).all()):
                        raise ValueError("Policy returned an invalid finite [horizon, 7] action chunk")
                    recorder.query(step, predicted)
                    actions.extend(chunk[:options.replan_steps])
                action = actions.popleft()
                obs, _, _, _ = session.env.step(action.tolist())
                recorder.action(action)
                result["steps"] = step + 1
                goal = bool(session.env.check_success())
                recorder.observe(obs, goal, final=goal or step + 1 == limit)
                if goal:
                    result.update(status="success", success=True)
                    break
    except SetupUnavailable as exc:
        result.update(status="setup_unavailable", error=str(exc), success=None, scored=False)
    except KeyboardInterrupt as exc:
        result.update(status="interrupted", success=None, scored=False, error="Interrupted by user")
        interrupt = exc
    except Exception as exc:
        result.update(status="error", success=None, scored=False,
                      error=f"{type(exc).__name__}: {exc}")
        (folder / "error.txt").write_text(traceback.format_exc())
        fatal = exc
    finally:
        write_json(folder / "config.json", config)
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        try:
            save_state(folder / "terminal.npz", session)
            terminal_obs = session.env.env._get_observations(force_update=True)
            if recorder is None:
                recorder = Recorder(folder, session, options)
                recorder.observe(terminal_obs, bool(session.env.check_success()), final=True)
            else:
                recorder.images(terminal_obs, result["steps"], phase="terminal")
        except Exception as exc:
            result.setdefault("artifact_errors", []).append("Terminal snapshot: " + str(exc))
        if recorder is not None:
            recorder.close()
            if recorder.errors:
                result.setdefault("artifact_errors", []).extend(recorder.errors)
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        result["media_retained"] = result["status"] != "success" or options.keep_success_media
        if not result["media_retained"]:
            shutil.rmtree(folder / "frames", ignore_errors=True)
            for name in ("rollout.mp4", "trajectory.h5", "frames.jsonl", "policy_queries.jsonl"):
                (folder / name).unlink(missing_ok=True)
        write_json(folder / "result.json", result)
    if interrupt is not None:
        raise interrupt
    if fatal is not None:
        raise RuntimeError(f"Stopped after saving error artifacts in {folder}: {fatal}") from fatal
    return result


def collect_results(output, cases):
    rows = {}
    for spec in cases:
        path = output / "trials" / trial_id(spec) / "result.json"
        if path.exists():
            rows[trial_id(spec)] = json.loads(path.read_text())
    return rows


def summarize(output, cases, cached_rows=None):
    import csv
    rows = list((collect_results(output, cases) if cached_rows is None else cached_rows).values())
    groups = {}
    for row in rows:
        key = "/".join((row["suite"], row["robot_start"], row["layout"]))
        group = groups.setdefault(key, dict(completed=0, evaluated=0, successes=0, policy_failures=0, unavailable=0, errors=0))
        group["completed"] += 1
        group["evaluated"] += int(row["scored"])
        group["successes"] += int(row["status"] == "success")
        group["policy_failures"] += int(row["status"] == "failed")
        group["unavailable"] += int(row["status"] in ("setup_unavailable", "invalid_start"))
        group["errors"] += int(row["status"] in ("error", "interrupted"))
    for group in groups.values():
        group["success_rate"] = group["successes"] / group["evaluated"] if group["evaluated"] else None
    evaluated = sum(bool(r["scored"]) for r in rows)
    successes = sum(r["status"] == "success" for r in rows)
    summary = dict(planned=len(cases), recorded=len(rows), finished=sum(r["status"] in COMPLETE for r in rows),
        evaluated=evaluated, successes=successes, success_rate=successes / evaluated if evaluated else None,
        statuses=dict(Counter(r["status"] for r in rows)), groups=groups)
    write_json(output / "summary.json", summary)
    columns = ["trial_id", "suite", "task_id", "trial", "initial_state_index", "robot_start", "layout",
               "status", "scored", "success", "steps", "wall_seconds", "media_retained", "error"]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary


def input_tasks(options):
    from libero.libero import benchmark, get_libero_path
    tasks = []
    for suite in options.suites:
        bench = benchmark.get_benchmark_dict()[suite]()
        if bench.n_tasks != 10:
            raise ValueError(f"Expected 10 tasks in {suite}, found {bench.n_tasks}")
        for task_id in options.task_ids:
            task = bench.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            initial = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            tasks.append(dict(suite=suite, task_id=task_id, instruction=task.language,
                              bddl_sha256=digest(bddl), initial_states_sha256=digest(initial)))
    return tasks


def environment_info():
    versions = {}
    for name in ("numpy", "torch", "mujoco", "robosuite", "h5py", "imageio", "openpi-client"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        versions=versions, git_commit=git("rev-parse", "HEAD"), git_status=git("status", "--short"),
        environment={name: os.environ.get(name) for name in ("MUJOCO_GL", "LIBERO_CONFIG_PATH", "CUDA_VISIBLE_DEVICES")})


def checkpoint_identity(policy):
    """Record local checkpoint inventory without copying or hashing multi-GB weights."""
    root = Path(policy.get("checkpoint", "/nonexistent-checkpoint"))
    if not root.is_dir():
        return dict(available_locally=False, path=str(root))
    return dict(available_locally=True, path=str(root.resolve()), files=[
        dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
             mtime_ns=path.stat().st_mtime_ns, sha256=digest(path) if path.stat().st_size < 1024**2 else None)
        for path in sorted(root.rglob("*")) if path.is_file()])


def run_batch(args):
    if args.dry_run:
        return _run_batch(args)
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another batch process is writing to this output directory") from exc
        return _run_batch(args)


def _run_batch(args):
    from libero.libero import benchmark
    from serve_pi05_libero import metadata as expected_metadata
    from waypoint_demo import PolicyConnection

    options = Options(**{name: getattr(args, name) for name in Options.__dataclass_fields__})
    cases = schedule(options)
    output = args.output.resolve()
    tasks = input_tasks(options)
    if args.dry_run:
        print(json.dumps(dict(runs=len(cases), tasks=len(tasks), options=asdict(options),
                              first=cases[0], last=cases[-1], output=str(output)), indent=2))
        return
    output.mkdir(parents=True, exist_ok=True)
    check_disk(output, options.min_free_gib)
    client = PolicyConnection(options.port)
    try:
        policy = client.connect()
    finally:
        client.close()
    if any(policy.get(key) != value for key, value in expected_metadata().items()):
        raise ValueError("Start this project's serve_pi05_libero.sh: the server must identify the standard Pi 0.5 LIBERO policy")
    manifest = dict(schema="pi05_40_batch_v1", options=asdict(options), tasks=tasks, policy=policy,
        checkpoint_inventory=checkpoint_identity(policy),
        sources={name: digest(ROOT / name) for name in SOURCE_FILES},
        policy_rng="Shared server PRNG is not reset per episode. Exact outputs and executed actions are saved for failures.",
        image_contract="Upright PNGs: raw LIBERO images rotated 180 degrees; policy inputs resized to 224 as in waypoint_demo.")
    path = output / "manifest.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text()) != manifest:
            raise ValueError("Existing output requires --resume with identical options, policy, code and task inputs")
    else:
        if args.resume or any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("Choose a fresh output directory, or resume an existing manifest")
        write_json(path, manifest)
        write_json(output / "environment.json", environment_info())
        write_json(output / "schedule.json", cases)
        (output / "source").mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(ROOT / name, output / "source" / name)
    rows = collect_results(output, cases)
    session, current_key, states, assets = None, None, None, None
    try:
        for index, spec in enumerate(cases):
            result_path = output / "trials" / trial_id(spec) / "result.json"
            if result_path.exists() and json.loads(result_path.read_text())["status"] in COMPLETE:
                continue
            check_disk(output, options.min_free_gib)
            key = spec["suite"], spec["task_id"]
            if key != current_key:
                if session is not None:
                    session.env.close()
                    session = None
                bench = benchmark.get_benchmark_dict()[key[0]]()
                states = np.asarray(bench.get_task_init_states(key[1]), dtype=float)
                if options.trials > len(states):
                    raise ValueError(f"{key} has only {len(states)} initial states")
                session = make_session(*key, spec["model_seed"], options.resolution)
                assets = task_assets(session, output, *key)
                current_key = key
            write_json(output / "progress.json", dict(index=index + 1, total=len(cases), trial_id=trial_id(spec), status="running"))
            print(f"[{index+1}/{len(cases)}] {trial_id(spec)}", flush=True)
            try:
                row = run_case(session, states[spec["initial_state_index"]], spec, options, output, policy, assets)
                print(f"  {row['status']} | steps={row['steps']} | {row['wall_seconds']:.1f}s", flush=True)
            finally:
                if result_path.exists():
                    rows[trial_id(spec)] = json.loads(result_path.read_text())
                summarize(output, cases, rows)
    finally:
        if session is not None:
            session.env.close()
        summary = summarize(output, cases, rows)
        write_json(output / "progress.json", dict(status="stopped" if summary["finished"] < len(cases) else "complete", **summary))
    print(json.dumps(summary, indent=2))


def replay(args):
    """Replay recorded states exactly, or re-execute saved actions and measure drift."""
    import h5py
    import mujoco
    trial = args.trial.resolve()
    config = json.loads((trial / "config.json").read_text())
    if not config["prepared"]:
        raise ValueError("This condition could not be prepared; inspect config.json and terminal.npz")
    root = next((parent for parent in trial.parents if (parent / "manifest.json").exists()),
                Path(config.get("experiment_root", trial.parent.parent)))
    spec, options = config["scenario"], config["options"]
    session = make_session(spec["suite"], spec["task_id"], spec.get("model_seed", spec["environment_seed"]), options["resolution"], args.render)
    writer = None
    try:
        xml_path = root / config["assets"]["model_xml"]
        if digest(xml_path) != config["assets"]["model_sha256"]:
            raise ValueError("Saved simulator XML was modified")
        session.env.reset_from_xml_string(xml_path.read_text())
        with np.load(trial / "initial.npz", allow_pickle=False) as saved:
            restore(session, saved["state"], controller="OSC_POSE", integration=saved["integration_state"])
        sim = session.env.sim
        model, data = getattr(sim.model, "_model", sim.model), getattr(sim.data, "_data", sim.data)
        if args.output:
            import imageio.v2 as imageio
            if args.output.exists():
                raise FileExistsError(args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.output, fps=1 / config["control_timestep"], codec="libx264", macro_block_size=1)
        maximum_drift, count = 0., 0
        with h5py.File(trial / "trajectory.h5", "r") as trajectory:
            length = len(trajectory["states"])
            if args.max_steps is not None:
                length = min(length, args.max_steps + 1)
            for index in range(length):
                if args.mode == "states" or index == 0:
                    mujoco.mj_setState(model, data, trajectory["integration_states"][index], mujoco.mjtState.mjSTATE_INTEGRATION)
                    sim.forward()
                    session.env._post_process()
                elif index - 1 < len(trajectory["actions"]):
                    session.env.step(trajectory["actions"][index - 1])
                drift = float(np.max(np.abs(sim.get_state().flatten() - trajectory["states"][index])))
                maximum_drift = max(maximum_drift, drift)
                if args.render:
                    session.env.env.render()
                if writer is not None:
                    obs = session.env.env._get_observations(force_update=True)
                    writer.append_data(np.concatenate([obs[key][::-1, ::-1] for key in CAMERAS], axis=1))
                if args.realtime:
                    time.sleep(config["control_timestep"])
                count += 1
        print(json.dumps(dict(mode=args.mode, observations=count, maximum_flat_state_error=maximum_drift,
                              final_task_success=bool(session.env.check_success())), indent=2))
    finally:
        if writer is not None:
            writer.close()
        session.env.close()


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run or resume the condition grid")
    defaults = Options()
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--trials", type=int, default=defaults.trials)
    run.add_argument("--seed", type=int, default=defaults.seed)
    run.add_argument("--port", type=int, default=defaults.port)
    for name in ("suites", "starts", "layouts"):
        run.add_argument("--" + name, nargs="+", default=getattr(defaults, name))
    run.add_argument("--task-ids", nargs="+", type=int, default=defaults.task_ids)
    for name in ("resolution", "settle_steps", "max_steps", "replan_steps", "image_every"):
        run.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    run.add_argument("--no-video", action="store_false", dest="save_video", default=True)
    run.add_argument("--keep-success-media", action="store_true")
    run.add_argument("--min-free-gib", type=float, default=defaults.min_free_gib)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true", help="Validate and print the plan without connecting to a policy or writing output")
    playback = commands.add_parser("replay", help="Replay a retained episode without a policy server")
    playback.add_argument("trial", type=Path)
    playback.add_argument("--mode", choices=["states", "actions"], default="states")
    playback.add_argument("--render", action="store_true")
    playback.add_argument("--realtime", action="store_true")
    playback.add_argument("--max-steps", type=int)
    playback.add_argument("--output", type=Path, help="Optional side-by-side camera video")
    return result


def main():
    args = parser().parse_args()
    if args.command == "replay" and args.max_steps is not None and args.max_steps < 0:
        raise ValueError("Replay step limit must be nonnegative")
    if args.command == "run":
        run_batch(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
