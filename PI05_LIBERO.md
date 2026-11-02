# pi0.5 LIBERO experiments

## RRT grasps milk, then VLA places it

Terminal 1:

```bash
bash run_hybrid_libero.sh
```

This opens an interactive scene and reuses the policy server on port 8000,
or starts pi05_libero if none is running. It stays running between commands.
There is no drop intervention in this runner.

Terminal 2 (in the `libero` conda environment):

```bash
python robot_cli.py pick milk_1
# Wait for "RRT grasp confirmed" before handing control to the policy:
python robot_cli.py vla "Put the milk in the basket." 250
```

`pick` uses pose IK and RRT-Connect for approach, descent and lift. It checks
two-finger contact and at least 5 cm of object lift before reporting success.
The robot keeps its gripper closed while waiting for the next command. `vla`
switches from joint control to the policy's OSC_POSE controller without
resetting the scene, robot joints, or held object. It uses fresh observations
and runs for the specified number of steps. Further `vla` prompts continue
from the resulting state. `pick` switches back to joint control if needed.

The default scene is the same alphabet-soup task used by `run.sh`, and includes
milk. Its built-in `success` command still measures the original soup goal;
the custom milk instruction does not change that goal. Grasp planning permits
contact with the selected object; it does not model the held object as an
attached collision shape. Failed planning or grasp verification stops `pick`
and leaves VLA inactive until explicitly commanded.

Other commands: `python robot_cli.py state`, `reset`, `release`, or `quit`.
Reset starts a fresh scene. A separately started policy server stays running
after quitting; a policy server started by the hybrid runner stops with it.

Read instantaneous gripper forces with `python robot_cli.py forces`:

- `finger_actuator_force_N`: signed force from each finger's position actuator.
- `finger_contact_normal_force_N`: sum of compressive normal contact forces
  per finger, including contacts with any object or scene geometry.
- `wrist_force_sensor_frame_N` and `wrist_torque_sensor_frame_Nm`: wrist
  force/torque sensor vectors, expressed in that sensor's local frame.

These are simulated measurements. Wrist force measures the transmitted load;
opposing finger squeeze forces can cancel there. The contact values sum
magnitudes, so do not add both fingers and treat the result as a net force.
Commands execute between motions: a `forces` request during `pick` or `vla`
waits for that command to finish. Restart an existing robot server after
updating the code to expose the new command.

## Keep the model loaded between runs

Start this once in terminal 1 and leave it running:

```bash
bash serve_pi05_libero.sh
```

Once it reports `server listening`, use terminal 2:

```bash
export USE_EXISTING_SERVER=1
bash run.sh --render --drop-object --drop-delay-steps 5 \
  --drop-open-steps 10 --num-trials-per-task 1
```

Repeat the task command as often as needed. The server keeps the model in
memory and retains compiled inference between runs. It continues using GPU
memory until you press Ctrl+C in terminal 1. `POLICY` selects the checkpoint
when starting the server; changing it on a client run does not switch the
running server's model. Both scripts accept `PORT` (default `8000`).

Checkpoints are already cached on disk. Reloading them at server startup is
not another download. The task's `1/1` progress bar measures trial completion.

## Start and stop the server for one run

If you exported the variable above, first run `unset USE_EXISTING_SERVER`.

```bash
POLICY=pi05_libero bash run.sh --render --drop-object \
  --drop-delay-steps 5 --drop-open-steps 10 \
  --num-trials-per-task 1
```

The runner starts the LIBERO-fine-tuned policy server and saves a video in
`videos/`. Once an object has two consecutive steps of gripper contact and
has lifted at least 3 cm from its initial height, the drop is armed. After
the requested delay, the Panda's gripper actuator target is set fully open
for the requested duration. The fingers move through normal physics. Arm
motion continues, then the policy regains gripper control.

`--drop-min-lift` sets the minimum lift in meters (default `0.03`; `0` uses
contact alone). Delay and opening duration are simulator control steps,
not seconds. A duration of 10 gives the object more time to fall than 5.

Look for `armed`, `open`, and `resume` in the terminal. If the episode ends
before opening, the runner prints `Drop did NOT trigger`. Normal task success
and episode limits still apply. An opening event confirms the intervention,
not necessarily a fall: the object can still land on or be supported by nearby
geometry. The intervention happens at most once per episode.
