# Task waypoint studio

Run using the same command:

```bash
MUJOCO_GL=glx /home/ramtin/miniconda3/envs/libero/bin/python subgoal_gui.py
```

The default task loads on startup. **Change task…** opens a searchable task
picker; the task list stays out of the main workspace. Each task plan has **at most
nine waypoints**, learned from a recorded pi0.5 rollout. A task without a saved
chain opens empty; it does not silently substitute a symbolic template.
Choose **LIBERO-90 · all 90 tasks** in the picker to access every training task.
The picker shows plan status and waypoint counts and can filter to plans that
passed replay. **Refresh build status** picks up newly saved plans. The ‹ / ›
buttons beside **Change task** switch within the current suite (Alt+Left/Right).
Ctrl+T opens the picker. Task changes preserve the save/discard/cancel flow.
Chains reopen from `task_waypoints/<suite>/task_<id>.json`. Each task has its own
chain. The left column contains your waypoints, the centre shows the robot, and
the right column edits the selected waypoint.

**Green buttons move the robot. Red stops it. Learning also moves the robot during the demonstrations.**

- **Save waypoint changes** saves the selected step. **Save new waypoint** adds a new one.
- **Save plan** (or Ctrl+S) saves the current edit and the whole plan for this task.
- **Move robot here** tests the target currently shown in the editor.
- **Run all waypoints** executes the saved sequence.
- **Reset robot & scene** returns the simulator to a starting scene.
- **Inject failures…** arms deliberate faults for the next full run.
- **Stop robot / Esc** interrupts execution.

Reorder, copy, and delete save automatically. Other edits show an unsaved indicator;
switching steps or closing the window asks whether to keep them. Save and Move
remain visible when the editor scrolls. **Learn from the VLA** sits at the top of
the right-hand column, above the waypoint editor, since both are about the plan
being edited; the left column is the plan itself and what runs it.
Opening a plan rejects files belonging to a different task.
Regeneration/import preserves the previous chain in a `.backup.json` file.
Task numbers in the picker start at 1; command-line task IDs and filenames start at 0.

## Define and connect targets

1. Click the object button under **Position relative to**. The searchable picker
   shows friendly names, with task targets first. Regions and joint frames are
   available through **Include regions and joints**. The chosen reference is
   marked in the camera. **Above object** puts a target 15 cm above its current
   top. **Use robot position** records the actual gripper's
   position and full orientation relative to the selected reference.
2. Adjust XYZ in **centimetres**, use the 1 cm buttons, or drag the orange target
   in the **Waypoint map** tab. The map updates as you type. Editing does not move
   the robot. **Wrist & placement options…** contains wrist rotation in degrees
   around the reference Z axis; captured roll and pitch are retained.
   **Vertical move · use previous X/Y** holds the preceding target's horizontal
   position. Editing X or Y turns this option off so your values take effect.
3. **+ Add a waypoint** starts a new draft. Press **Save new waypoint** to append
   it. Select a row and press **Save waypoint changes** to update it.
   **Copy** and the arrow buttons duplicate/reorder steps.
4. Use **From the previous waypoint** for connected relative motions, such as a lift.
   Its offsets use world axes. For the first node, the previous pose is the
   gripper's captured starting pose, not its position after each test click.
   Reordering changes these connections.
5. Set **Gripper before this move** to Open, Close, or Keep unchanged. A close/open
   command finishes before the robot moves. You can also add **Gripper only** steps.
   **Gripper on arrival** closes or opens at the endpoint. Learned plans use this
   to combine reaching and grasping in one waypoint, preserving contact timing.
   For grasp verification,
   choose the object being grasped as that row's reference. Subsequent placement
   nodes can enable **Position the held object at this target** in the wrist and
   placement options after a verified grasp. The camera also has buttons to open
   and close the gripper immediately.
6. **Move robot here** moves to the edited target. **Run all waypoints** executes rows in order
   from the robot's current state. Reset the scene first to repeat a full plan.
   **Stop robot / Esc** interrupts motion or planning. A failed target stops the chain.

While editing, reference poses are captured when the scene loads or resets.
The map and **Move robot here** use the same resolved endpoint, including placement
corrections. Repeated clicks do not accumulate offsets or chase an object being
carried. A robot already within 2 mm and 1 degree of the target does not move again
unless its gripper setting needs to change. To use an object's new current pose,
press **Update references from scene**; your local XYZ offsets remain unchanged.
**Run all waypoints** also captures fresh references before starting the sequence.

While a sequence runs, the axes drawn on the camera follow the waypoint the robot
is actually doing: the row is set by the executor's own start event for that row,
and the pose is the one the executor was given, published as `session.running_target`.
That is not always the pose in the file. A grasp adjusted for clearance, or a
contact refreshed from an object that has moved since the run began, can be
several centimetres away — 4.2 cm on a grasp row whose object had just been
shoved. Idle, the axes go back to following the editor's own target.

Object targets use full local poses: `world_position = object_position +
object_rotation @ local_offset`. Movable object poses are captured at run start,
so moving an object before a run moves its chain, while grasping it during a run
does not make its targets chase it. Fixtures, joint frames, and regions belonging
to fixtures are refreshed between nodes for drawer/door motion. World references
remain fixed. Connections are sequential commands; map arrows show target order,
not a planned collision-free trajectory. Arc endpoints and carrying corrections
in the preview are approximate (full-sequence execution measures the actual grasp).
Straight-line motions allow contact and are not
collision-checked; planned motions use the existing IK/RRT executor.
Motion uses gradual starts and stops with continuous Cartesian travel and wrist
rotation. Obstacle-avoidance paths retain their checked joint edges and stop at
corners. GUI playback is paced to simulation time and refreshes up to 30 times per second.

## Learning from pi0.5

**Learn waypoints · pi0.5 ×2** uses the local OpenPI server on port 8000.
Start it with `bash serve_pi05_libero.sh` if needed. The button records exactly
two rollouts from the same starting scene, each up to 500 simulation steps
(800 for the long LIBERO-10 tasks), and restores the scene afterward. The status
bar shows demonstration 1/2 or 2/2 and its step count. Stop/Esc cancels learning.
Inference runs in a worker; the simulator stays on the GUI thread.

For a simple pick/place motion it identifies **Approach → Reach & close gripper →
Lift → Transfer → Going down → Finish & open gripper** from measured contact and
vertical travel. Redundant phases are merged; substantial path bends and
multi-stage interactions can retain extra waypoints, with a hard maximum of nine.
It prefers a successful rollout, then fewer waypoints and shorter travel. It
selects one coherent route instead of averaging incompatible grasps. If neither
rollout succeeds, the result is an explicitly unverified draft for manual review.
Generated plans are saved automatically, with the previous plan backed up. Press
**Run all waypoints** to check the inferred plan yourself; learning does not claim
the compact plan has passed replay. All phases and offsets remain editable.

Both raw recordings and the selection result are stored under
`task_waypoints/_demonstrations/gui/<suite>/task_<id>/<timestamp>/`.

The extractor retains grasp/release events and the final pose, then selects the
most significant position and wrist changes until it reaches nine waypoints.
Grasp poses use measured finger contact: pi0.5 may start closing while it is still
descending. Every endpoint is a measured demonstration pose, expressed relative
to a scene object or fixture. There are no hidden motion waypoint lists inside
the nine editable steps. Brief finger-command pulses that do not change contact
are filtered. Excess genuine gripper transitions in successful demonstrations
are reported. Repeated attempts in a failed demonstration may be reduced to an
explicitly labelled limited draft, with omitted transitions recorded in its source.

The batch builder covers all 130 tasks across the five installed suites:

```bash
/home/ramtin/miniconda3/envs/libero/bin/python build_waypoint_library.py --resume --jobs 3
```

It records once per task, caches the full demonstration, compresses it, and checks
the compact plan by replay. Failed replays try alternative compression of the
same recording. Rerunning the builder reuses cached recordings. `--only-failed`
retries failed plans; it does not request another pi0.5 rollout.

- `task_waypoints/<suite>/task_<id>.json`: editable plan, at most nine steps.
- `task_waypoints/_demonstrations/<suite>/task_<id>.json`: measured poses, actions,
  finger contacts, reference poses, initial simulator state, and rollout outcome.
- `task_waypoints/_reports/<suite>/task_<id>.json`: rollout and replay outcomes,
  attempted compressions, and per-waypoint results.
- `task_waypoints/build_progress.json`: current batch progress.
- `task_waypoints/_backups/`: previous plans preserved before replacement.

The GUI labels learned plans as replay passed, failed, or needing replay. A
successful pi0.5 rollout does not guarantee a successful compressed replay.
Validation uses the recorded starting scene; it does not establish success under
arbitrary object rearrangements. Edits invalidate the saved replay label.

## Inject failures

**Inject failures…** arms at most one deliberate fault per waypoint. Only
**Run all waypoints** applies them; **Move robot here** never does, and nothing
armed here is written to the plan file. Loading another task, learning a new plan
or importing one disarms everything.

- **Collision** drives level into the middle of the body nearest that leg of the
  route, instead of around it. A sideways stroke past the object's centre is what
  makes contact: coming down from above leaves a narrow object standing between
  open fingers, touching nothing.
- **Transition deviation** ends the move the chosen distance to one side of the
  waypoint, across the direction of travel.
- **Orientation deviation** arrives at the waypoint with the wrist rotated by the
  chosen angle about its own approach axis. An angle the arm cannot track is
  halved once rather than left as a move that stays exactly on plan.
- **Grasp loss** opens the fingers wide half way through the move and leaves
  them open, so whatever was held falls and the rest of the move carries
  nothing. Every later row that only says “keep” inherits that open hand.

Distance and angle apply to every armed fault in the run. A fault fires once per
waypoint however many moves that waypoint makes, so a guarded move that probes,
plans and then falls back to a straight line still fails once.

Nothing is faked: the fault is really carried out, and a run recorded for a
failure detector has to contain the failure rather than a label for one. The
status line afterwards names every fault that actually fired.

A waypoint that has carried out its fault is **not** measured against the target
it was told to miss. Its learned position and orientation tolerance is turned off
for that row, along with the small corrective retry that would otherwise pull the
wrist quietly back onto plan, and the row reports `ok (injected …)` instead. The
sequence then carries on, which is the point: what the fault costs the waypoints
*after* it — a grasp that now misses a shoved object, a placement made with an
empty hand — is most of what there is to detect, and none of it happens if the
plan stops at the row that was broken on purpose. Every other waypoint in the run
is checked exactly as usual.

Two guards are outside that and can still stop a run at the injected row: an arm
that cannot reach a faulty pose at all reports the executor's own `unreachable`,
and the VLA skill executor's grasp check reports `target object lost during
transport` when a transport row finds the gripper empty.

## Code and checks

- `waypoint_chain.py`: versioned task data, validation and pose transforms.
- `waypoint_session.py`: scene lifecycle, template adaptation and execution.
- `waypoint_demo.py`: bounded policy connection, observation adapter and recorder.
- `waypoint_library.py`: extraction of at most nine measured pi0.5 key poses.
- `build_waypoint_library.py`: resumable recording, replay, and provenance reports.
- `subgoal_gui.py`: Tk editor, embedded camera and connected target map.
- `waypoint_ui.py`: layout, colours, readable labels and searchable pickers.
- `waypoint_failures.py`: deliberate faults carried out during a full replay.
- `waypoint_session.py` and `waypoint_gmm.py` read `session.injected_failure`,
  the flag that takes a deliberately broken row out of the arrival checks.
- `studio_fonts.py`: smooth-font support for the local conda Tk runtime.

The existing command-line planners remain available. The studio uses a versioned
chain format, separate from the old `example_subgoals.json` list format.

```bash
/home/ramtin/miniconda3/envs/libero/bin/python -m unittest test_waypoint_chain test_waypoint_library
NUMBA_CACHE_DIR=/tmp/libero_numba MUJOCO_GL=egl \
  /home/ramtin/miniconda3/envs/libero/bin/python -m unittest test_waypoint_motion
NUMBA_CACHE_DIR=/tmp/libero_numba MUJOCO_GL=egl xvfb-run -a \
  /home/ramtin/miniconda3/envs/libero/bin/python -m unittest test_waypoint_gui
NUMBA_CACHE_DIR=/tmp/libero_numba MUJOCO_GL=glx xvfb-run -a \
  /home/ramtin/miniconda3/envs/libero/bin/python -m unittest test_waypoint_failures
```
