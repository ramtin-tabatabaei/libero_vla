# libero_vla

Research code for driving a Franka Panda in the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
benchmark with a pi0 / pi0.5 vision-language-action policy, and for the waypoint
planning, failure injection, and recovery work layered on top of it.

The repository covers four connected threads:

- **VLA rollouts.** Run LIBERO tasks with a pi0.5 policy served by
  [openpi](https://github.com/Physical-Intelligence/openpi) over a websocket,
  while this process owns the MuJoCo simulation and steps it.
- **Hybrid control.** Hand a task back and forth between a classical
  pose-IK / RRT-Connect planner and the VLA, without resetting the scene, the
  robot joints, or a held object.
- **Waypoint learning.** Learn per-task waypoint chains from recorded pi0.5
  rollouts, edit and replay them in a GUI, and measure coverage and failure
  modes.
- **Recovery.** Train controllers that recover from a failed grasp or drop — a
  continuous PPO controller, and a return-point selector that chooses which
  previously visited primitive to roll back to before returning control to the
  VLA.

## Prerequisites

This code expects several things that live outside the repository.

| Dependency | Expected location | Notes |
|---|---|---|
| LIBERO | `./LIBERO/` | `git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git` — developed against commit `8f1084e` |
| GalaxeaVLA | `./GalaxeaVLA/` | `git clone https://github.com/OpenGalaxea/GalaxeaVLA.git` — developed against commit `89f2322` |
| openpi | `~/openpi` | Serves the policy from its own JAX virtualenv |
| `libero` conda env | `~/miniconda3/envs/libero` | Runs the simulator and everything in this repo |
| pi0.5 checkpoint | `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero` | Also `pi0_libero` for the pi0 variant |

Several scripts hardcode these paths; `run.sh` is the place to look first if
your layout differs.

## Running

Start a rollout — this launches the policy server, waits for the checkpoint to
load, runs the rollout, and shuts the server down:

```bash
./run.sh                                   # libero_object, task 0, 3 trials, pi05
./run.sh --task-id 4 --num-trials-per-task 10
./run.sh --render --drop-object            # force a drop and watch pi05 recover
POLICY=pi0_libero ./run.sh                 # the pi0 checkpoint instead
```

Hybrid planner/VLA control, where the planner grasps and the VLA places:

```bash
bash run_hybrid_libero.sh                  # terminal 1: scene + policy server
python robot_cli.py pick milk_1            # terminal 2, in the libero env
python robot_cli.py vla "Put the milk in the basket." 250
```

The waypoint studio, for editing and replaying per-task waypoint chains:

```bash
MUJOCO_GL=glx ~/miniconda3/envs/libero/bin/python subgoal_gui.py
```

Recovery training and evaluation live in [recovery/](recovery/); see
[recovery/README.md](recovery/README.md).

## Layout

```
run_vla.py, run.sh            pi0/pi0.5 rollouts against LIBERO
robot_server.py, robot_cli.py hybrid planner/VLA control over a command socket
serve_pi05_*.py               openpi policy server wrappers
waypoint_*.py                 waypoint learning, replay, coverage, failures, recovery
subgoal_planner.py            subgoal decomposition
subgoal_gui.py, waypoint_ui.py, *_gui.py, *_editor.py
                              Tk/Qt studios for building and replaying plans
recovery/                     PPO recovery controller and return-point RL
tests/                        pytest suite
task_waypoints/               learned waypoint chains, per suite and task (JSON)
saved_plans/                  saved subgoal plans
demonstrations/               recorded demonstrations (JSON)
```

### Documentation

- [WAYPOINT_LEARNING.md](WAYPOINT_LEARNING.md) — waypoint learning pipeline
- [WAYPOINT_STUDIO.md](WAYPOINT_STUDIO.md) — the waypoint editing GUI
- [PI05_LIBERO.md](PI05_LIBERO.md) — hybrid RRT + VLA experiments
- [PI05_BASE_LIBERO.md](PI05_BASE_LIBERO.md) — base pi0.5 on LIBERO
- [PI05_40_EVALUATION.md](PI05_40_EVALUATION.md) — 2,400 episodes across 40 tasks, three robot starts, original/shuffled layouts, and ten trials
- [recovery/RETURN_POINT_RL.md](recovery/RETURN_POINT_RL.md) — return-point selector
- [recovery/RETURN_POINT_TRAINING.md](recovery/RETURN_POINT_TRAINING.md) — training it

## Not in this repository

Model weights, rollout output, and the vendored upstream repositories are
excluded — together they account for roughly 87 GB. Specifically:
`LIBERO/`, `GalaxeaVLA/`, `checkpoints/`, `waypoint_models/`, `experiments/`,
`recovery/runs/`, `graphify-out/`, `videos/`, `backups/`, and `*.log`. See
[.gitignore](.gitignore) for the full list.

## Tests

```bash
~/miniconda3/envs/libero/bin/python -m pytest tests/
```

## LIBERO-Pro2.0 and LIBERO-Max workspace

The [LIBERO-Pro2 workspace](LIBERO-Pro2/README.md) adds the Pro2 and Max task
GUI, a pi0.5 task bridge, download tools, and tests. Its scripts and environments
are kept in `LIBERO-Pro2/`; the existing waypoint and recovery workflows above
keep their original locations.

To continue this workspace on another computer:

```bash
git clone https://github.com/ramtin-tabatabaei/libero_vla.git
cd libero_vla/LIBERO-Pro2
```

Follow [LIBERO-Pro2/MIGRATION.md](LIBERO-Pro2/MIGRATION.md) to restore the pinned
OpenPI checkout, simulator release, model weights, and optional Max dependencies.
Credentials, model weights, datasets, and local environments are excluded from Git.
