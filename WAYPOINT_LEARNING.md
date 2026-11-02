Multi-demonstration waypoint learning
====================================

Option 2 learns **minimal, single-reference waypoints** from successful VLA
recordings. A pick-and-place plan has seven rows, one for each motion the
executor drives: move clear of where the arm stands, move across to above the
object, grasp, lift the object straight up, carry it across to above the goal,
lower it, release. Move and transport each appear twice because each is two
motions — a vertical one and a horizontal one — and folding them into a single
row hid half of what the robot did. A demonstrated articulated interaction adds move, move and push — stand off the
handle, close on it, push — before them when the drawer is opened to reach the
object and after them when it is closed again. A task solved by sliding an
object rather than carrying it is move, move and push per demonstrated stroke,
at most two strokes. Tasks with
several clean, demonstrated object transfers compose one five-row stage per
object, in VLA order. Each saved position is a
signed XYZ offset from exactly one
object, joint frame, or the preceding waypoint. The first move connects from
the robot's pose actually reached at runtime.

A vertical motion is always listed separately from the horizontal one that
follows it, because the two are not interchangeable. The lift has to leave the
object's support straight up: a diagonal departure drags the object sideways
across whatever it is standing on before it clears. The same holds in reverse
for the arm's first move, and for the stand-off before a handle. Each of those
pairs shares a reference and differs in one coordinate, so the list reads as
"up by this much, then across by that much".

Every free-space connector travels along **one axis of the frame it is stored
against**: the pre-grasp move shares the grasp row's other two coordinates, and
the loaded traverse shares the place row's. The frame axis used is whichever
lies closest to world vertical, so a rotated object frame still yields a single
changed number in the editor. Contact poses — grasp, place, and the end of a
push — stay exactly as measured; squaring a drawer pull onto one axis discards
the few millimetres of wrist descent that keep the open fingers behind the
handle, and loses the drawer part way out.

The offset shown in the editor is the point the plan actually aims at. Preview,
`Move robot here`, `Run all waypoints` and the saved file all resolve the same
stored offsets, and the clearance heights they encode come from one shared
formula rather than three similar ones. Because those heights depend on the
tallest object present, the plan's offsets are recomputed from the scene on
screen whenever the task is loaded, shuffled or restored. Rows that follow a
drawer push are stored against the drawer where the push leaves it, not where
it starts, because that is the frame the executor resolves them in.

The grasp target is relative to the source object. Its approach rotation is
the VLA wrist rotation at the first gripper-close command, expressed in the
source object's frame at that instant. The contact rotation is the VLA wrist
rotation at stable grasp, expressed in that same pre-contact source frame;
this avoids encoding rotations imparted to the object while the fingers close.
The transport rotation is the VLA wrist rotation at its highest carried point,
expressed in the destination's frame. A demonstrated drawer push follows the
VLA rotations measured at contact and at the end of the push.

Transport and place target the destination object or region. The executor
replans clearance against the live layout, checks the grasp and lift, and
corrects placement using the measured object pose in the gripper. It does not
call the VLA during replay.

**Free-space motion goes through the collision-free planner.** The learned plan
says *where* the arm has to be; how it travels between those poses is
RRT-Connect's job. Every edge is checked in a scene where a grasped object
moves with the wrist, and the planner returns a straight joint-space edge
wherever one is free, so an unobstructed move is not made longer for the sake
of planning it. A whole planned path runs as one trajectory and settles once at
the end; stopping at each of the planner's corners costs seconds and buys
nothing, since only the last corner is a target.

The planner reads which joints the arm has, where its end effector is and
which geometry belongs to it from the loaded model, so nothing assumes a joint
count, a naming convention or a particular robot; its search, smoothing and IK
constants are all fields on `Args` rather than literals in the code. Edge
checking is bounded by end-effector travel as well as joint travel: the joint
spacing alone does not bound how far the wrist moves, and at the old 0.05 rad
spacing it moved up to 29 mm between checks, far enough to step over an 18 mm
object. Sampling now also honours a 7.5 mm chord limit, which measured 10.6 mm
worst case on a curved edge, at roughly twice the collision checks per edge.

Two cases keep the straight Cartesian move first, with the planner behind it.
Carrying something, because a joint-space route swings the wrist and tips the
object out. Approaching a handle or a surface contact, because a joint-space
route into one presses the wrist against the fixture instead of arriving beside
it. In both the straight path is still walked, IK-solved and collision tested
before it is driven. During that
check the grasped object is carried with the wrist and counted as part of the
arm, so the planner steers *it* clear of the scene rather than only the
fingers; a free-joint object does not follow arm joint angles on its own, and
without this the check would clear a path straight through whatever the object
would have hit. The destination being placed into is excluded, since you cannot
plan into a basket that counts as an obstacle. Contact motions — the descent
onto an object, the lift, the placement, and the push strokes themselves — stay
straight and unchecked by design; a checked planner refuses every one of them.
Each executed move records how its clearance was established (`line`,
`planned`, `contact`, or `unchecked` where no reachable route existed and the
executor was left to report its own failure) in the replay report. The approach
to a drawer handle excludes the cabinet that carries it, for the same reason.

How fast each motion runs depends on what it is near. A free-space connector
runs at 0.6 rad/s peak joint speed, a move with something in the gripper at
0.2 rad/s, and the centimetres either side of a touch at 0.075 rad/s. A single
Cartesian stroke carries both: a descent onto an object crosses the air quickly
and slows for the last five centimetres, timed as one trajectory that settles
once. Those limits are `travel_*`, `carry_*` and `contact_*` in
`subgoal_planner.Args`. The carried limit is not a guess: at 0.3 rad/s a bowl
shifts in the fingers and lands 38 mm off its plate, and the run ends up
*slower* than at 0.2 for having to correct. A planned joint-space route is also
never used while carrying, because it swings the wrist as it goes; loaded, the
checked straight move keeps the wrist level and the planner is the fallback.

Clearance heights cover only what lies **under the path being flown**, found by
testing each body's footprint against the straight segment. Lifting over the
tallest object anywhere in the scene means climbing over a cabinet the arm
never crosses, which can put the traverse outside its reach for no gain. Within
that, the loaded traverse takes the highest pose the arm can actually hold:
the demonstrated carry height is a ceiling, the lowest height that still clears
the path is a floor, and the demonstrated carry rotation gives way to the
placement rotation if the arm cannot hold it over this layout. The preview
runs the same search against the same scene, so the offsets shown are the
offsets flown.

Arrival precision is 3 mm and 0.03 rad for a measured contact, and 20 mm and
0.15 rad for a free-space connector. A connector only has to get over the
scene; holding it to contact precision fails moves at the edge of the arm's
reach and buys nothing, because the following contact target and wrist rotation
are recomputed from the live object pose anyway. No source-to-destination frame or two-object offset
is fitted. New waypoints referenced to the preceding waypoint store their
rotation relative to that waypoint's wrist rotation; older saved plans without
the `orientation_relative_to_previous` flag retain their original world
rotation for compatibility.

A task whose only goal is articulation — opening a drawer, turning a knob — has
no grasp and no destination object, so the pick/place learner has nothing to
fit. It is learned as exactly two waypoints: move to the measured contact, then
push. LIBERO ends an episode the moment its predicate holds, so a policy that
sets an object down and stops is never recorded opening its fingers; the
settled terminal pose is accepted as the demonstrated placement, while an
episode that ends with the object in mid-air is not. Where a policy fumbled and
re-grasped, the hold that completed the placement is the one learned, and a
fumbled object picked straight back up counts as one transfer rather than two.

The older GMM/GMR fallback now also fits one semantic reference per interval
and keeps only boundary targets for move and transport. When no semantic skill
fits at all, the fallback plan is reduced to the closing boundary of each
primitive run within each interval, plus every pose that carries a gripper
event, so it stays an editable list rather than dozens of rows. Move and
transport boundaries reconstruct at local phase 1, where the connector
correction is already zero, so dropping their interior samples changes no
target. Unedited saved plans
with sufficient recorded motion evidence are rebuilt when loaded. Contact
loss lasting at most two samples is bridged when comparing VLA event sequences;
real gripper retries remain separate. Option 2 continues collecting successful
recordings until it has a compatible sequence, and can reuse prior successful
recordings only for the same task, instruction, and original object setup.
Archived drawer-only recordings without joint or contact measurements keep their
older plan; a fresh VLA recording is needed to learn their push boundary
reliably. Editing a learned waypoint
switches to manual replay so the entered target is used. Previous benchmark
results in [RESULTS.md](waypoint_models/shuffle_study_20260926/RESULTS.md)
measure an older executor and do not validate this simplified representation.

Forty-one tasks across `libero_object`, `libero_spatial`, `libero_goal`,
`libero_10` and `libero_90` were recorded, learned, edited and replayed through
the option-2 tab. Every one produced a minimal plan — 2, 4, 5, 7 or 10 rows —
the largest gap between a displayed offset and the point actually flown was
0.000000 mm, and every free-space connector moved along one frame axis. 120 of
146 layouts, original plus up to three unseen shuffles each, ended in LIBERO
task success. The per-task table, the failures and the scope are in
[`minimal_waypoints_20260928/RESULTS.md`](waypoint_models/minimal_waypoints_20260928/RESULTS.md).

LIBERO-90 task 38 demonstrates the multi-object case: the VLA first clears
`moka_pot_2` from the stove, then places `moka_pot_1`. The saved option-2
plan has ten single-reference rows. A demonstration that completes two
transfers is learned as two stages even when its first transfer alone would
also fit the single-object learner. It was learned from compatible,
unshuffled recordings in the same original setup and replayed successfully in
that setup and one pot-swapped scene. See
[`task38_signature_repair_20260927.json`](waypoint_models/task38_signature_repair_20260927.json)
for the exact results. This is a two-layout check, not a task-wide
robustness estimate.

The new pipeline runs alongside the existing nine-waypoint editor and
`waypoint_library.demonstration_plan` baseline. It records measured VLA motion,
partitions successful runs by ordered semantic events, aligns compatible runs,
and fits each interval in one semantic frame. The output is a mean trajectory,
conditional covariance, and the minimal waypoint list described above.
No recovery, PPO, behavior-tree, or VLA-training code is involved.

The TP-GMM change is backed up in `backups/before_tpgmm_20260924_122223/`
(53 files, `source.tar.gz` plus SHA-256 `manifest.json`). The earlier pre-change
backup is `backups/before_multidemo_20260924_104432/source.tar.gz`. Its adjacent
`manifest.json` contains SHA-256 checksums for the 57 backed-up files. The backup
covers project-level source/configuration and tests, including recovery source;
it excludes third-party repositories, checkpoints, recordings, and generated
outputs. Extract it into a separate directory to inspect or restore selected
files.

Workflow
--------

Use the existing environment; NumPy, SciPy and Matplotlib are sufficient. No
scikit-learn installation is required. Run from `/home/ramtin/libero_vla`:

```bash
export NUMBA_CACHE_DIR=/tmp/libero_numba
export MUJOCO_GL=egl
export OPENBLAS_NUM_THREADS=1
PYTHON=/home/ramtin/miniconda3/envs/libero/bin/python

# Requires your existing pi0.5 policy server on localhost:8000.
# Attempts are bounded; every attempt restores the same unshuffled scene.
$PYTHON -m waypoint_demo --suite libero_object --task-id 0 \
  --num-successes 5 --max-attempts 30 --port 8000 \
  --output demonstrations/task0/

$PYTHON -m waypoint_learn --method gmm --input demonstrations/task0/ \
  --output waypoint_models/task0/

$PYTHON -m waypoint_validate --input demonstrations/task0/ \
  --output waypoint_models/task0/validation.json

$PYTHON -m waypoint_visualize --model waypoint_models/task0/

# Both sparse waypoints and dense learned trajectory; original and shifted scene.
$PYTHON -m waypoint_replay --model waypoint_models/task0/ \
  --mode both --scene both --shift 0.02 0 0
```

The studio is one editor, with a task picker, waypoint list, live camera, map,
reference controls, XYZ/wrist editing, add/copy/delete/reorder,
save/import/export, Move robot here, Run all waypoints, Reset and Stop controls:

```bash
MUJOCO_GL=glx /home/ramtin/miniconda3/envs/libero/bin/python subgoal_gui.py
```

It learns from multiple successful recordings. For tasks outside the supported
object-relative pick/place path, it uses event-aware alignment and the
probabilistic GMM/GMR model. It defaults to five successful demonstrations and
an adaptive waypoint count. **Learn waypoints · VLA** records up to six attempts
per requested success, archives all attempts, fits the model, and populates the
editable waypoint list. A running local pi0.5 server is required to collect new
recordings; nothing else in the studio needs one.

**Save plan** (or Ctrl+S) writes the waypoints to `saved_plans/<suite>/task_NNN.json`
and says where in the status bar. That file is separate from the autosave the
editor keeps in `task_waypoints_gmm/` while you work: it is the copy you chose
to keep, and its presence is what the task picker reports. A task with one is
listed green and marked `saved` in the picker's Saved column, and **Only show
tasks I have saved** narrows the list to those.

Option 2 uses exactly **move, grasp, transport, place, release, push**.
The first five describe grasp-based manipulation. Push requires measured
contact and object translation, rotation, or joint displacement without a grasp.
A plan carries at most two pushes: one before the transfer and one after it for
an articulated fixture, or at most two strokes for a slid object.
Every new VLA recording starts from the task's original unshuffled scene,
even if you shuffled the preview first. The edited scene is restored afterward.

For move and transport, the first target connects to the actual end-effector
pose at interval entry (including the first interval). A smooth correction
fades to zero at the learned object-relative endpoint. Grasp/place targets
follow the named object's replay pose, so independently moved objects do not
pull the endpoint toward their old training positions.

Under the option-2 camera, press **Shuffle task**, then **Run all waypoints**.
Each shuffle starts from the original setup and rejects the previous shuffle,
so consecutive successful clicks produce different layouts. **Restore first
setup** returns the robot and objects to their exact state when this task was
loaded; learned waypoints and edits are kept. If another valid shuffle cannot
be found, the current scene is preserved and the status explains why.
**Plan tools → Shuffle objects for replay** runs the same action. Source objects exchange supported, non-overlapping
positions; goal containers, including parents of goal sites, stay fixed.
Saved edited plans are preserved. If a legacy plan is loaded, use
**Plan tools → Rebuild primitives from this model’s VLA recordings** or record
new demonstrations. Automatic model discovery prefers connected primitive
models over legacy models, then the most recently saved model.

To measure complete-task success over several reproducible shuffled layouts:

```bash
$PYTHON -m waypoint_replay --model waypoint_models/object0_primitives/ \
  --scene shuffled --shuffle-seeds 0 1 2 --mode waypoints \
  --output /tmp/primitive_shuffle_report.json
```

These are geometric adaptations of recorded VLA motion. Success on a few
layouts does not establish robustness for every task, obstruction, or placement;
check the per-layout `replay_success` and `task_success` results.

The editor also has **Plan tools → Learn from saved demonstrations…** and
**Open learned GMM/GMR model…**, which do not require a running VLA server. Select
the corresponding task first. Available saved models are picked up when loading
a task with no previously edited GMM plan. To try the existing example, select
**Goal tasks → Task 01: open the middle drawer of the cabinet**. Its saved
recordings are converted to minimal waypoints when loaded. **Plan tools → Restore model’s recorded scene** restores the
original example state before running it.

Option 2 saves
editable version-2 `gmm_editor_chain` files in `task_waypoints_gmm/`, retaining
pose covariance, tolerances, semantic phase and reference policy. Learned model
artifacts remain separate in `waypoint_models/`; new GUI fits and recordings go
under `waypoint_models/gui/`. The legacy format and its nine-waypoint limit are
unchanged. The editor uses the model's adaptive safety bound (100 by
default), and supports manually adding further waypoints up to that bound.

Editing a learned mean marks it as edited; demonstration covariance and tolerances
are retained as provenance, not re-estimated for the edited pose. Changing the
reference re-expresses positional covariance in the new frame. Manually added
waypoints use the existing controller's tolerances. Full runs preserve learned
interval anchors and apply learned arrival tolerances. A small arrival miss
(within twice the learned bounds, capped at 10 mm / 0.1 rad) gets up to two
same-target Cartesian corrections before stopping. Corrections keep the gripper
unchanged, require a stationary arrival, and never widen the learned bounds.
Failure messages include the measured errors and allowed limits. GUI and CLI
replay use the same checks. Move robot here uses the
same resolved target as the shared editor map. Future fixture positions cannot be
predicted by the static preview, so actual articulated interval frames are
captured at execution time.

This replaces the earlier read-only multi-demo replay tab. Dense trajectory replay
remains available through `waypoint_replay --mode trajectory`.

Watch the saved example in a simulator window from a desktop terminal:

```bash
MUJOCO_GL=glx NUMBA_CACHE_DIR=/tmp/libero_numba OPENBLAS_NUM_THREADS=1 \
/home/ramtin/miniconda3/envs/libero/bin/python -m waypoint_replay \
  --model waypoint_models/cached_goal0_smoke --scene original \
  --mode waypoints --render --output /tmp/waypoint_visible_replay.json
```

Add `--realtime` for control-timestep pacing, or use `--mode trajectory` for the
dense learned curve. The window closes after execution. Ctrl+C in the terminal
stops the replay. A desktop display is required for `--render`.

The recorder writes successes at the output root and unsuccessful/unusable
attempts in `failed/`. Episodes already solved at their first measurement do not
count toward the requested successes. Raw simulator initial states go into
`demo_NNN.state.json` sidecars, separate from model features. A bounded attempt
budget can finish with fewer successes and a nonzero exit status; inspect
`recording_report.json`. Existing GUI recording retains the inline initial state
for compatibility.

Inputs may be one JSON file or a directory tree of recordings. The learner needs
at least two compatible successful nontrivial episodes; five is the intended
starting point. Leave-one-out needs at least three. It checks task identity,
finite positions, and proper SO(3) rotations. Failures, trivial solved episodes,
and incompatible event modes are explicitly listed as excluded. Training and
validation never silently count these as useful demonstrations.

Configuration and ambiguous references
---------------------------------------

Both learning and validation accept `--config config.json` and
`--reference-map references.json`. Config keys are the fields of
`waypoint_learning.Config`; omitted fields retain defaults. For example:

```json
{
  "samples_per_interval": 24,
  "components": 4,
  "position_scale": 0.05,
  "orientation_scale": 0.35,
  "max_waypoints": 100
}
```

`max_waypoints` is a safety bound, not a requested output count. Extraction raises
an actionable error rather than silently deleting anchors when it exceeds the
bound. Increase it with `--max-waypoints 150` if the task requires more detail.

A reference override might be:

```json
{
  "target": "alphabet_soup_1",
  "destination": "basket_1_contain_region"
}
```

Keys can also be individual event keys such as `stable_grasp:alphabet_soup_1:1`.
An event-specific override takes priority over a role override. References must
exist in the demonstrations. `training_error.json` preserves errors and candidate
names when no unambiguous semantic reference can be determined.

Recording and legacy migration
------------------------------

`waypoint_observation.py` records at each control timestep:

- Control-step index and elapsed timestamp.
- Measured world EEF position and rotation matrix, with the legacy
  `position`/`orientation` aliases.
- Linear and angular velocity from consecutive executed poses; angular velocity
  is `Log(R_now R_previous.T) / dt` in world coordinates.
- Gripper command, measured joint positions and finite-difference velocities.
- Separate left/right pad contacts, two-pad grasp identities, and fixture/moving
  link contacts. Moving links are identified by their nonrobot hinge/slide joint.
- Object, fixture, site and joint-frame poses, reference candidates, articulation
  joint positions, evaluated BDDL goal predicates, and task-success state.

First-sample derivative fields are zero because no preceding observation exists.
Contacts are sampled at the control rate, not at every physics substep. A terminal
measurement is saved after the final executed action; ordinary samples retain the
old convention of measuring before their associated action command.

Episode metadata includes task name, suite, ID, instruction, success, episode
length, control timestep, initial poses, BDDL goals and articulated joints.
`action` is retained for audit, but learned positions and rotations come from
executed motion, never integrated raw predictions.

Version-1 recordings remain usable. Missing terminal measurements use the saved
endpoint. Two-pad `grasped` evidence supplies the earliest *known* contact; the
system does not fabricate an earlier first-finger contact, lift onset, joint
motion or predicate event. Missing goals can be migrated from the installed
LIBERO benchmark BDDL. This is the only optional simulator-package dependency in
legacy reference migration; new recordings contain their own semantics. Warnings
in the model identify incomplete legacy evidence.

Events and temporal alignment
-----------------------------

Events include start/end, close/open command transitions, finger/object contact,
stable grasp (three consecutive measurements by default), object lift (8 mm of
measured object rise from grasp onset), release/loss of grasp, fixture contact and
release, articulation motion (5 mm for slides or 0.005 rad for hinges), individual
goal-predicate achievement, and task success. Identity and occurrence number are
part of the event key, so repeated manipulations remain distinct.

Runs with the same ordered event signature form a compatible group. Training
uses the largest group and saves all group memberships and exclusions. It does
not stretch missing events or combine different retry sequences. This is a
conservative policy: noisy contacts and command pulses can require collecting
more compatible demonstrations. Raw event sequences remain inspectable.

Within each adjacent pair of anchors:

1. Transform poses into the semantic interval's reference frame.
2. Resample positions linearly and orientations with quaternion SLERP.
3. Choose a training medoid as the interval template.
4. Run monotone, endpoint-constrained DTW with a normalized band of 0.3.

The DTW cost is

```text
||p_i - p_j||² / position_scale²
+ angle(R_i.T R_j)² / orientation_scale²
+ gripper_weight * (g_i - g_j)²
```

Default scales are 5 cm, 0.35 rad and gripper weight 4. Alignment cannot cross an
anchor, so pre-grasp samples cannot be registered to a post-grasp interval.
Common global phase lies in `[0,1]`; interval durations follow the training mean
of normalized control-step event times. Simultaneous events retain separate
ordered boundaries with a tiny positive phase width. These are discrete ordering
markers, not evidence of elapsed physical time.

Frames and orientation mathematics
----------------------------------

For frame world pose `(R_o, p_o)`:

```text
p_rel = R_o.T (p_eef - p_o)
R_rel = R_o.T R_eef
p_world = p_o + R_o p_rel
R_world = R_o R_rel
```

References follow measured grasp identity and BDDL predicates. Approach/grasp use
the source object. The first measured lift interval ends in that source frame;
subsequent transport/placement uses its `in`/`on` destination region. Legacy data
without lift evidence switches after stable grasp and is flagged as incomplete.
Articulation prefers an observed moving-joint frame, then the semantic handle or
region. No nearest-object selection is used in this implementation. Multi-goal
ambiguity is surfaced for explicit overrides rather than guessed.

Movable-object frames are captured at episode start. This preserves actual lift
and transport displacement: expressing the gripper relative to an object that
moves with it would erase that motion. Articulated frames are captured at each
interval start. Replay uses the same policy with the current scene's poses,
preventing a target from continuously chasing its moving reference.

Orientations are rotation matrices on disk. At each aligned phase, SciPy's
sign-invariant rotation mean defines a nominal `R_bar`. Model orientation
residuals are right tangent vectors:

```text
r = Log(R_bar.T R_rel)
R_predicted = R_bar Exp(mu_r)
```

No Euler angles are used. This chart avoids the quaternion sign ambiguity and
handles demonstrations on opposite sides of ±π in absolute orientation. Residual
orientation spread over 2.6 radians is rejected as incompatible local modes.
Orientation covariance is in the local tangent coordinates of the nominal curve.

Probabilistic model
-------------------

The GMM fallback fits each semantic interval in **one** task frame: an object,
region, fixture, or articulated joint chosen from measured interaction and BDDL
goals. The local trajectory contains phase, XYZ position, SO(3) orientation
residual, velocity, and gripper state. Deterministic EM fits a Gaussian mixture;
GMR predicts the conditional curve and covariance at replay. Every waypoint
stores one local XYZ offset and one local orientation. The model keeps the
version-2 file schema for compatibility and adds
`reference_scheme: single_reference_minimal`.

Move and transport keep only their two interval boundaries. The first boundary
connects to the robot pose actually reached at the preceding waypoint. The
connection correction fades smoothly with local phase `u`: its positional weight
is `1 - 3u² + 2u³`, so the learned object-relative endpoint is unchanged. This
previous-waypoint connection is computed at replay; it is not another stored
reference frame or a second object offset. Other primitives keep contact and
event waypoints where their demonstrated motion requires them.

Older version-2 multi-frame models still load for inspection. Unedited option-2
plans with recordings are rebuilt in the one-reference format when loaded.

Waypoints and executable tolerances
----------------------------------

All semantic endpoints and frame boundaries are retained. Move and transport
keep only their boundaries. Other primitives use:

- Low position/orientation covariance as consistency evidence.
- Direction changes, translational turns and rotational change.
- Local maxima of the combined importance score.
- Pose reconstruction error relative to learned positional tolerances and a
  rotational error threshold.
- Measured approach samples just before contact/open events, labelled pre-grasp,
  pre-interaction or pre-release only when there is actual motion evidence.

Each waypoint stores phase, gripper command, event identity, selection reason,
one task-frame reference, one relative pose mean, covariance and tolerances.
Replay resolves that reference against the current scene. Move and transport
also use the reached previous waypoint as the start of their connector, without
storing another object offset. Interval endpoints may share one event; replay
deduplicates the event action.

For each 3×3 pose covariance, scalar tolerance is

```text
tolerance = clip(2 * sqrt(largest_eigenvalue(Sigma)), minimum, maximum)
```

Default bounds: 2–40 mm for position, 0.02–0.35 rad for orientation. These are
execution bounds derived from demonstration spread, not calibrated 95% confidence
intervals. Full covariance remains saved for anisotropic inspection. Tighter
regions also receive finer geometric waypoint reconstruction.

Replay checks both tolerances before executing an arrival gripper event. The
existing joint controller can fail its fixed joint settling criterion during
contact even while its measured Cartesian pose is acceptable. Only for that
specific settling failure, the adapter permits a pose inside both learned bounds
that remains stationary over five further control steps. It logs the original
executor result. Unreachable paths and other executor failures remain failures.

Three contact motions have their own acceptance rule, because a wrist loaded
against something cannot satisfy a free-space criterion:

- A **placement** that stops short is judged by the object, not the wrist: it
  is accepted when the carried object has come to rest within 20 mm of where
  the demonstration left it. Lowering a bowl into a drawer or onto a plate
  stops when the bowl reaches what holds it, which can be a centimetre or two
  from the pose recorded in the demonstration's own layout, and where the
  wrist ended is not what the task asks about. An object still moving, or at
  rest somewhere else, is a miss.
- A **drawer stroke** that fails only to settle is accepted while the measured
  joint is still advancing. A stalled joint is still a failure.
- A **push stroke on a free object** is accepted the same way, judged on the
  object's measured displacement.

A **knob or door** is swept about its own joint axis rather than along the
straight chord between the measured contact and finish, which slides off a
turning handle part way round; the wrist rotates with the sweep. When the
fingers never engage, the demonstrated chord seats them and the sweep resumes.
A prismatic drawer follows the measured stroke directly.

Replay and validation
---------------------

`waypoint_replay` reconstructs targets using current references and reuses the
existing IK/Cartesian executor. `--mode trajectory` follows the dense sampled GMR
curve; `--mode waypoints` follows the adaptive list. Both apply the discrete learned
gripper events, check learned tolerances, and separately report completion and
LIBERO task success. Dense replay currently settles at each sample; it is not a
continuous velocity/time-profile controller.

`--demo FILE` chooses the original initial-state sidecar/inline state. By default,
replay uses the first selected recording. The demonstration must match the model's
task. Exact simulator states are never included in the learned model features.

Shift testing supports horizontal movement of free objects only. It conservatively
checks AABB overlap and support, settles the object and verifies support height.
Use `--shift-object NAME --shift DX DY 0`. Fixed fixtures and ambiguous support are
reported as unsupported, not as successful tests. The reused Cartesian executor
is not collision checked; successful reconstruction is not a guarantee that a
new scene has a feasible path.

Leave-one-out trains every fold from the remaining successful usable recordings.
Templates, frames, nominal curves and mixture fitting use training data only.
The report includes mean/max relative-position and SO(3) errors, unwarped event
phase errors, gripper-event consistency, position ellipsoid coverage, and the
unchanged single-demonstration extractor baseline. Held-out DTW is observational
shape registration; it is not an online timing prediction or replay success test.
Timing error is reported separately using original episode fractions, so hard
alignment does not make timing error trivially zero. Incompatible folds are
reported, never silently omitted from the fold count.

Visualization saves `plots/trajectories.png`, `phase_diagnostics.png` and
`frame_intervals.png`: raw executed runs, aligned local trajectories, means,
principal covariance axes, selected waypoints, semantic anchors, and phase plots
of uncertainty, gripper probability and importance. Different reference frames
are explicitly labelled; the interval plot avoids connecting unrelated frames.

Files and migration
-------------------

Modified: `waypoint_demo.py` (version-2 recording and bounded recording CLI);
`subgoal_gui.py` and `waypoint_ui.py` (the editor, its layout and the
application lifecycle); `waypoint_session.py` (overridable frame and
arrival checks, with unchanged default behavior); `waypoint_rrt.py` (`Scene`
carries a grasped object with the wrist while sampling configurations, so a
free-joint object is not left behind by a collision check).

Added:

- `waypoint_gmm.py`: editable adaptive chain serialization and learned tolerances.
- `waypoint_gmm_editor.py`: GMM/GMR backend for the shared full editor.
- `test_waypoint_gmm.py`, `test_waypoint_learned_gui.py`: chain and full-editor tests.
- `waypoint_observation.py`: simulator measurement adapter.
- `waypoint_learning.py`: representation, events, semantics, DTW, EM/GMR,
  uncertainty and adaptive extraction.
- `waypoint_learn.py`, `waypoint_validate.py`, `waypoint_visualize.py`,
  `waypoint_replay.py`: command-line workflows.
- `test_waypoint_learning.py`, `test_waypoint_learning_sim.py`: numerical and
  headless simulator coverage.
- `WAYPOINT_LEARNING.md`: this guide.

Existing `waypoint_chain.py`, `waypoint_library.py`, and
`build_waypoint_library.py` retain their original behavior and nine-waypoint
format. Default session behavior is unchanged. New models use a separate
`model_type` and loader; they do not silently truncate into legacy chains. The
single-demonstration editor has been removed from the studio; `Panel` remains
as the shared base class and `waypoint_library.demonstration_plan` still
provides the legacy extractor for anything that wants it. The studio records,
fits, edits and executes multi-demonstration waypoints.

Tests and integration evidence
------------------------------

```bash
$PYTHON -m unittest test_waypoint_learning test_waypoint_learning_sim \
  test_waypoint_library test_waypoint_chain test_waypoint_motion \
  test_waypoint_skill test_waypoint_gmm test_waypoint_primitives -v

xvfb-run -a $PYTHON -m unittest test_waypoint_gui test_waypoint_learned_gui -v
```

The tests cover transform round trips, orientation correctness and ±π behavior,
DTW paths/endpoints, semantic anchors, numerical EM/GMR, covariance sensitivity,
covariance-derived tolerances, shifted target reconstruction, learned serialization,
invalid inputs, fixture events, measured recording/restoration, and both replay
modes in original and physically shifted simulator scenes.

A real-data integration model is saved at `waypoint_models/cached_goal0_smoke/`.
It uses five archived successful VLA drawer-opening runs for `libero_goal`, task 0.
Four already-solved one-step episodes and two unsuccessful recordings are excluded.
The model has 15 adaptive waypoints; five leave-one-out folds evaluated. Legacy
recordings have no fixture-contact or joint-motion telemetry, so this example
exercises the generic geometry model with start/success/end anchors, not complete
articulation-event learning. Fresh version-2 recordings provide the missing data.

Observed results for this small archived-data check:

| Check | Result |
| --- | --- |
| Numerical/simulator/legacy motion tests | 38 passed |
| Existing GUI tests | 12 passed |
| Mean held-out relative position error | Multi: 18.19 mm; single-demo baseline: 26.37 mm |
| Mean held-out orientation error | Multi: 0.0949 rad; baseline: 0.1098 rad |
| Held-out nominal 95% position ellipsoid coverage | 46.25%; covariance is undercalibrated on these five runs |
| Original-scene learned waypoint replay | Completed; LIBERO task success true |
| Original-scene dense trajectory replay | Completed; LIBERO task success true |
| Shifted drawer fixture | Unsupported: fixed fixture, not a free object |
| Shifted free-object simulator regression | Both replay modes reached targets; not a full shifted-task success test |

Reports, logs and figures are saved with the example model. Comparisons describe
DTW-registered trajectory shape on these recordings, not benchmark success rates.

The local VLA server was unavailable during implementation, so no new five-run VLA
collection was claimed. The archived drawer integration and simulator regressions
are limited checks, not benchmark-wide validation. Covariance with five samples
can be undercalibrated; inspect held-out coverage, errors and replay reports before
interpreting low variance as a reliable task constraint. Different geometric routes
within one event mode can still produce an infeasible mean. No collision-free
or cross-scene success guarantee is made.

API references used: SciPy's rotation/SLERP documentation
(https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
and the Gaussian-mixture overview
(https://scikit-learn.org/stable/modules/mixture.html).

The full-editor replacement passed 22 GUI/chain tests and 26 legacy/simulator
regressions (48 total). The archived 15-waypoint drawer model also completed the
LIBERO task through the editor's Run all waypoints control. Reports and a
screenshot are in `waypoint_models/_gui_editor_validation/`. Recording integration
uses mocked VLA output in tests; fitting uses the real GMM/GMR implementation.
No fresh VLA collection is claimed for this GUI change.
