# RL selection of previously visited recovery states

The action is **(recorded primitive/waypoint visit, percentage within that visit)**.
For example, the model can select `waypoint 2 / approach / 10%`, physically return
to that previously measured robot configuration, and give control to the fixed
Pi 0.5 VLA to grasp the object and finish the task.

This is one shared model across tasks. Task IDs, object names, waypoint numbers,
and primitive names are audit/display metadata, not neural inputs. The policy
does need the current goal: geometric goal-predicate tokens preserve which
source goes to which destination and whether a mechanism should open, close,
turn on, or turn off. Task independence means shared parameters and no per-task
policy, not success prediction without knowledge of the goal or VLA checkpoint.
Generalization to unseen tasks must be measured.

## What a return point means

`RecordVisits` records actual joint configurations, wrist poses, finger positions,
gripper commands, velocities, held objects, goal context, and scene geometry at
primitive starts and control steps. Candidates contain only samples strictly
before the measured failure. No target is interpolated or invented.

The percentage is elapsed simulation time through that recorded primitive.
`completed_duration` means the entire primitive was visited and completed before
failure. `visited_prefix_duration` means the primitive was interrupted: 10% is
10% through its already visited prefix, not a claim about the unexecuted suffix.
Phase bins choose the nearest actual sample; the output reports its actual phase,
not the requested bin. Use more bins for finer choices. The percentage is not
Cartesian distance, planned motion phase, or an offset into a future trajectory.

The live controller returns the robot's historical joint, wrist, and gripper
configuration within measured tolerances and settles at rest. It does not rewind
simulation time or restore old object positions or historical velocities. Scene
changes remain visible to the model and to the resumed VLA. A historical held
object must still be held; the controller rejects missing historical grasps or
incompatible held-object states.

## Questions answered by the model

| Question | Prediction and measurement |
| --- | --- |
| Reliability | `P(VLA task success within the rollout horizon | safe arrival, current scene, goal, return point, fixed VLA)` |
| Reachability | `P(reached the return point without configured recovery safety violations)` plus a current-scene RRT path check |
| Efficiency | Typical recovery time conditional on safe arrival; VLA completion time conditional on success; total time conditional on joint recovery/VLA success |

The set encoder consumes current and historical scene tokens and relative
candidate geometry. It supports variable numbers of return points and scene
objects. The LIBERO adapter uses privileged simulator measurements and Panda
proprioception; deployment on a real robot would require equivalent observations.
It currently retains at most 32 scene/goal tokens, prioritizing goals and nearby
geometry. This is limited scene coverage, not complete perception.

The model has six heads: a Q value, safe-arrival probability, conditional VLA
success probability, and three log-time regressions. Probabilities are learned
estimates, not guarantees. Log-time regression estimates a typical duration;
it does not establish an upper latency bound or an arithmetic expected duration.
Times use **simulation seconds**, including the stable-success hold. They exclude
planner wall time, network latency, and inference latency.

## RL formulation and training labels

There is one high-level decision per episode: select a return point, execute the
recovery controller, and run the frozen VLA until success, safety stop, or timeout.
This makes the selector a contextual bandit, the one-decision case of RL. It
optimizes end-to-end outcome rather than distance to a previous waypoint.
It does not train a new low-level motor controller or fine-tune the VLA.

```text
return = 10 * joint_success
         - 2 * unsuccessful_episode
         - 10 * any_recovery_or_VLA_safety_violation
         - 0.1 * (recovery_seconds + observed_VLA_rollout_seconds)
```

Weights are configurable. Q learns measured returns with a conservative
log-sum-exp regularizer over the available candidates, adapted from
[Conservative Q-Learning](https://arxiv.org/abs/2006.04779).
This small neural implementation does not establish that paper's theoretical
lower-bound guarantees for these finite robot datasets.

Safe arrival and VLA success use separate binary losses. If recovery fails,
there is no VLA label; it is not treated as a VLA failure. Timeouts are negative
success labels with censored completion times. Recovery-time regression uses
safely reached episodes; finish-time regression uses successful VLA continuations;
total-time regression uses jointly successful episodes. Observed timeout time
still enters the RL return, so failing quickly cannot earn a success bonus.

The trainer splits by **whole task**, keeping scenes, repeats, and counterfactual
returns for each task together. At least two tasks and successful training
continuations are required. `evaluation.json` reports held-out Brier scores,
duration errors, and Q errors. Ranking regret is reported only for fault decisions
with outcomes for every candidate; missing counterfactual outcomes are never
fabricated. Ranking regret is for unconstrained Q ranking, not the deployed
selector's safety/support/planner gates. Validation selects the checkpoint;
an additional untouched task set is needed for final generalization claims.

## Collection, training, and visualization

Run from `/home/ramtin/libero_vla` with the existing LIBERO interpreter. The
standard Pi 0.5 server must already be running on port 8000. Each named task
needs a saved learned plan. These commands create a new experiment, leaving
earlier timing studies intact.

```bash
export NUMBA_CACHE_DIR=/tmp/libero_return_numba
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/libero_return_mpl
RETURN_PY=/home/ramtin/miniconda3/envs/libero/bin/python

$RETURN_PY -m recovery.return_collect \
  --tasks libero_goal:1 libero_goal:2 libero_object:3 \
  --vla-id pi05_libero@standard_cached --port 8000 \
  --output experiments/return_point_training \
  --kinds transition orientation grasp_loss \
  --intensities low high --scenes original shuffled \
  --phases 0 0.1 0.25 0.5 0.75 1 --repeats 3

$RETURN_PY -m recovery.return_train \
  --data experiments/return_point_training/transitions.jsonl \
  --output recovery/runs/return_point_selector \
  --validation-tasks libero_object:3 --epochs 100

$RETURN_PY -m recovery.return_view \
  --checkpoint recovery/runs/return_point_selector/checkpoint.pt \
  --decision experiments/return_point_training/libero_goal_t001_original_transition_low/decision.json \
  --output experiments/return_point_training/selected_return.png
```

Collection restores the same fault world before **each offline counterfactual
experiment**, executes each return physically, then resumes the VLA. This reset
is for comparing actions in simulation; the live recovery path does not reset
the world. Repeated VLA trials can vary because its inference randomness is not
rewound. Separate server errors or interrupted trials stop collection rather
than becoming misleading failure labels. Use a fresh output directory to rerun.

The aggregate `experiments/pi05_failure_recovery_timed_paired_20261001/timing.csv`
cannot train this action space: it lacks dense pre-failure robot visits,
candidate choices, safe-return labels, and matched VLA continuations. The new
JSONL format includes those measurements and records the exact controller identity.
Pin `--vla-id` to the actual VLA checkpoint/version used; the collector checks
the server's Pi 0.5 metadata, but a human-supplied version tag does not itself
verify checkpoint content hashes. Mixed VLA/controller datasets are rejected.

`return_view` writes a 3D plot of recorded wrist positions and marks the selected
point, plus a JSON ranking with percentages, timestamps, probabilities, ETAs,
and rejection reasons. This offline view does not run a physical planner.

## Live handoff

Wrap the waypoint execution with `RecordVisits`. After the existing fault detector
has observed a failure and unwound waypoint control at a control boundary:

Enable physical robot collision meshes with
`recovery.scene.enable_robot_collisions(session.world)` **before recording the
episode**. The collector does this automatically. Live callers must use the same
collision settings as the training experiment for the force monitor and learned
probabilities to have matching semantics.

```python
from recovery.return_model import ReturnPointSelector
from recovery.return_sim import RecordVisits, recover_with_selector

selector = ReturnPointSelector('recovery/runs/return_point_selector/checkpoint.pt')
# recorder.history is the RecordVisits history from this same live episode.
report, outcome = recover_with_selector(
    session, recorder.history, measured_failure_sim_time, selector, chain.language,
    vla_id='pi05_libero@standard_cached', port=8000)
```

An acceptable return needs similar training support, estimated probabilities
above configured thresholds, compatible held-object state, and a current-scene
collision-checked joint path. No acceptable option means no automatic return or
VLA handoff. Default probability/support thresholds are provisional and must
be tuned using validation data; neighborhood counts are not confidence bounds.

The return controller includes carried objects in RRT checks, verifies measured
arrival, enforces a 20-second recovery limit and joint limits, and checks
robot/world contact forces at every physics substep. Defaults stop above 100 N
individual contact or 10 N summed undesired contact; permitted finger-pad contacts
are limited to goal/held objects. The same force monitor can stop the resumed VLA.
These thresholds are not calibrated hardware safety limits. Robot self-contact
and passive world/world contacts are outside this monitor; sampled RRT geometry
and force stops do not prove continuous collision avoidance or prevent the
first impact. A live unsafe or failed return never resumes the VLA.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $RETURN_PY -m pytest tests/test_return_point_rl.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $RETURN_PY -m pytest tests/test_return_point_sim.py -q
```

Tests exercise causal history, recorded phase selection, scene padding, candidate
permutation, censored/conditional training labels, held-out task splits, actual
checkpoint training/loading, unsupported-state rejection, safe handoff gating,
physical return without time rewind, and measured pre-failure history in LIBERO.
Synthetic unit-test checkpoints validate software behavior only; they are not
robot success-rate evidence.

The first measured two-task pilot and starter checkpoint are documented in
[PILOT_RESULTS.md](../experiments/return_point_rl_pilot_20261012/PILOT_RESULTS.md).
The default-gated pilot selector declines every return point in the inspected
decision; the checkpoint is a pipeline smoke test, not a validated recovery policy.

For the larger experiment with separate training, validation, and untouched
physical test tasks, see [RETURN_POINT_TRAINING.md](RETURN_POINT_TRAINING.md).
It preserves force stops during recovery and observes the resumed VLA's contacts
without aborting at the return controller's provisional threshold. Goal success
and success without force exceedances are reported separately. The live API
retains strict contact stops by default; a checkpoint's contact contract must match.
