# Pi 0.5: 40-task initial-state evaluation

The default schedule is **40 tasks × 3 robot starts × 2 object layouts × 10
trials = 2,400 episodes**. It uses `libero_spatial`, `libero_object`,
`libero_goal`, and `libero_10`; each suite contributes ten tasks.

The six conditions are normal/small/large robot starts, each with original and
shuffled object layouts. Small means a 4–8 cm gripper displacement from the
normal robot start; large means 20–35 cm. These are the same collision-checked
offsets available in the GUI. Original task instructions and success predicates
are used throughout.

Trial indices 0–9 select ten different **official LIBERO initial states**. Each
six-condition group shares the same official state, environment seed, and
shuffle seed. Layouts are generated once and cached. The object's placement is
identical across robot-start conditions within a layout. Failed permutations
fall back to a supported source-object relocation, recorded explicitly as
`source_relocation`. If neither operation can be prepared, the condition is
`setup_unavailable`; it is excluded from policy success rates. Starts whose goal
is already satisfied are also excluded.

Each task uses a fixed model-construction seed, including after resume. Official
states reset physics without re-randomizing fixture geometry in the model.
Robot offsets share a seed across layouts; collision rejection can select a
different valid arm pose in a shuffled scene. The actual joints and measured
displacement are recorded for every condition.

## Run on this machine

Start the standard policy server in terminal 1 and leave it running:

```bash
cd ~/libero_vla
bash serve_pi05_libero.sh
```

In terminal 2, inspect the full plan before running:

```bash
~/.local/bin/libero-vla-batch run --output experiments/pi05_40 --dry-run
~/.local/bin/libero-vla-batch run --output experiments/pi05_40
```

The local launcher sets the existing Python/LIBERO paths and defaults to EGL
headless rendering. `MUJOCO_GL=osmesa` selects software rendering instead. A
generic environment with this repository and its simulator dependencies can run
`python pi05_benchmark.py` with the same arguments.

From a clone on another computer, the `~/.local/bin/libero-vla-batch` launcher
is not included. Activate your LIBERO simulator environment, configure its
LIBERO asset paths, and run from the repository root:

```bash
python pi05_benchmark.py run --output experiments/pi05_40 --dry-run
python pi05_benchmark.py run --output experiments/pi05_40
```

Keep the policy server described above running during the evaluation. The
40-task runner uses port 8000 by default; run it separately from the Pro2
simulator service, which also defaults to port 8000. The saved experiment
outputs live under the ignored `experiments/` directory and need a separate
transfer if you want to resume an existing run on another computer.

Resume with the same command plus `--resume`. Completed episodes are skipped;
interrupted/error episodes are retried while their earlier attempts remain in
`trials/_previous_attempts/`. Configuration, task inputs, policy identity and code
must match the original manifest. Two processes cannot write the same output.

For a small real-policy smoke run:

```bash
~/.local/bin/libero-vla-batch run --output experiments/pi05_smoke \
  --suites libero_object --task-ids 0 --trials 1 --max-steps 20
```

Suite episode limits default to 220/280/300/520 executed policy actions. Policies
are queried every five actions. The server metadata must match this project's
standard supervised Pi 0.5 LIBERO contract.

## Saved evidence

`summary.json`, `results.csv`, and `progress.json` are updated after each episode.
`manifest.json`, `environment.json`, `source/`, `assets/` and `layouts/` preserve
the configuration, code, versions, task definitions, simulator XML and paired
initial states.

Every failed, unavailable, interrupted or errored episode retains its directory
under `trials/<trial-id>/`:

- `config.json` and `result.json`: condition, exact seeds, selected official
  initial-state index, instruction, actual shuffle and robot offset, policy
  metadata, result and errors.
- `initial.npz` and `terminal.npz`: flat simulator states and full MuJoCo
  integration states. An unavailable setup can have only a terminal state.
- `trajectory.h5`: every executed action, per-step flat and integration state,
  policy proprioception and task-success flag. It is flushed after every step.
- `frames/agentview_image/` and `frames/robot0_eye_in_hand_image/`: upright PNGs
  at every policy step, plus terminal snapshots. `frames.jsonl` records their
  step indices and simulation times. These images can later feed DINO.
- `policy_queries.jsonl`: complete predicted action chunks and server timing.
- `rollout.mp4`: side-by-side camera video when video encoding is available.
- `error.txt`: traceback for execution errors, when applicable.

Successful episodes retain their configuration, result and endpoint states;
their bulky media/trajectory are removed unless `--keep-success-media` is set.
`--image-every N` reduces image/video sampling, while state/action recording
stays per-step. A 5 GiB disk reserve stops the batch before it exhausts storage.
Recording/video problems are reported rather than silently discarded.

## Replay a failure without a policy server

```bash
~/.local/bin/libero-vla-batch replay \
  experiments/pi05_40/trials/libero_object_t000_r000_small_shuffled \
  --output /tmp/failed_trial.mp4
```

The default `--mode states` renders the recorded state sequence. `--render
--realtime` opens a live viewer; `--mode actions` executes the saved actions and
reports any state divergence. MuJoCo and the original model assets must remain
available. Videos and existing output files are never overwritten by replay.

The shared policy server's random-number generator is not reset per episode.
Repeated inference is therefore not promised to reproduce identical actions;
the saved outputs, actions, images and state sequence preserve what actually
happened in each failed episode.
