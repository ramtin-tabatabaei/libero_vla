# PPO recovery for LIBERO

For a shared RL model that chooses **which previously visited primitive and
percentage to return to**, predicts safe arrival/VLA success/completion time,
and hands control back to the VLA, see [RETURN_POINT_RL.md](RETURN_POINT_RL.md).
That selector uses the existing physical planner and is trained separately from
the continuous PPO controller below.

This module trains a state-based Panda controller with the requested reward,
physical arm/object collisions, and randomized object placements. It recovers a
pick-up or works a specified mechanism; it does not complete the subsequent
placement or every stage of a multi-object LIBERO task.

## Reward

The defaults implement:

```text
r = P(d_previous) - P(d)
    - lambda_R g(d) e_R
    + 2 [first valid contact] + 5 [first stable grasp]
    + 10 [held task success]
    - lambda_c sum_i min(F_i, F_max)
    - lambda_s [F_contact > F_safe]
    - lambda_a sum_j action_j²
```

`P(d) = 10d - 2 exp(-10d)`, so `P'(d) = w(d) = 10 + 20 exp(-d / 0.10)`, and
`g(d) = 1 / (1 + exp((d - d_align) / tau))`. Defaults: alignment midpoint
`d_align=0.10 m`, transition temperature `tau=0.01 m`,
`lambda_R=0.5`, lift threshold 0.05 m, `lambda_c=0.02`, `F_max=50 N`,
`lambda_s=1`, `F_safe=100 N`, `lambda_a=0.01`. All are exposed through
`--env.reward.*` (lift through `--env.lift-target`). Rotation error is the
geodesic angle to the task's desired grasp pose. Stable grasp requires three
consecutive control samples with both pads engaged; success requires a held
lift for five samples.

Contact and grasp bonuses are **one-time per episode** by default
(`latch_bonuses=True`). Losing and regaining contact does not pay again. The
10-point final bonus waits for the configured success hold, then the episode
terminates. For grasp mode this requires a stable grasp above the lift threshold;
a brief lift or toss does not earn the completion bonus.

The alignment gate is smooth: at 9, 10 and 11 cm its values are approximately
0.731, 0.5 and 0.269 with the default temperature. Set
`--env.reward.align-temperature` to vary the transition width; it must be
positive. This initial 1 cm temperature is configurable, not a calibrated optimum.

Integrated progress is now the default (`integrated_progress=True`):

```text
r_progress = 10(d_previous - d)
             + 2(exp(-10d) - exp(-10d_previous))
```

It retains the same distance-dependent gain while making the **undiscounted**
progress sum around any closed distance loop zero, up to floating-point error.
The old endpoint-weighted product pays about +0.465 for 0.20 → 0.10 → 0.20 m;
the integrated term pays zero. This statement concerns the progress component,
not the entire reward or a claim of policy invariance under PPO's discount.

The former reward design can be reproduced with
`--env.reward.no-integrated-progress`, `--env.reward.no-latch-bonuses`,
`--env.reward.no-smooth-alignment`, and
`--env.reward.no-terminal-stage-bonus`. A time cost remains optional.
`info['reward_terms']` exposes each term.

### Collision calibration before scaling up

Collision coefficients remain `lambda_c=0.02` and `lambda_s=1` pending
measurement; they are not claimed to be calibrated. Every PPO rollout appends
a record to `calibration.jsonl`, also retained under `calibration` in
`history.json`. Statistics include **all control steps**, even when no episode
has finished, using raw rewards before normalization, clipping, or bootstrap
adjustments. Records contain overall and per-mode distributions:

| Metric | Definition |
| --- | --- |
| `peak_force_n` | Largest individual robot/world contact force in the control interval, including permitted contacts |
| `sum_bad_force_n` | Largest uncapped sum of undesired forces across physics substeps |
| `capped_bad_force_n` | Largest sum of per-contact capped undesired forces; the collision reward input |
| `r_collision` | Continuous collision reward, negative |
| `r_severe` | Severe collision reward, negative |
| `r_collision_total` | Continuous plus severe collision reward |
| `r_task` | Progress + alignment + stage bonuses, before effort/time/collision costs |
| `severe_event` | 0/1 threshold exceedance; its mean is the severe-contact step rate |

Each distribution includes count, mean, standard deviation, minimum, p05,
p50, p90, p95, p99 and maximum, plus nonzero rate and conditional nonzero
statistics so rare impacts remain visible. `peak_force_n` is an observed force;
it is different from the configured `force_cap` (`F_max` in the reward).
Raw and capped force-sum maxima may occur on different physics substeps.

Start with focused runs. Compare penalty tails and nonzero collision rewards
against task rewards, then compare held-out success, severe-contact frequency,
and contact/grasp rates across coefficient settings under the same seeds.
Choose magnitudes that reduce impacts without suppressing useful contact.
Distributions alone cannot establish that a coefficient changes policy behavior;
that requires these controlled training/evaluation comparisons. Collisions
continue to incur penalties without ending episodes automatically.

## Task taxonomy and contact rules

- **Grasp:** movable object; only target/finger-pad contact is permitted.
  Arm, palm, finger-back, table and distractor contacts incur penalties.
- **Articulate:** moving drawer/door/knob; use the BDDL open, close, turn-on or
  turn-off predicate and aim at its handle. Finger contact with the moving link
  is permitted for pushing/hooking. Fixed panels and arm contacts are penalized.
  The final 10-point bonus requires the goal predicate to hold for the
  configured success hold instead of requiring a lift.
  If no predicate is specified, signed joint travel is the fallback.
- **Reach:** explicit comparison/fallback; contact/grasp bonuses are disabled,
  and held arrival supplies the final bonus.

Robot collision meshes remain active; visual meshes stay noncolliding. Forces
use the norm of normal and tangential components, excluding torque. The monitor
samples every physics substep and uses the snapshot with the largest summed
capped undesired force in the control interval. Severe collision checks use the
maximum individual robot/world contact force across that interval, **including
permitted target contacts**. Robot self-contact and passive world/world contact
are outside this collision reward. No collision automatically ends an episode.

## Randomization and observation

Resets sample benchmark states, perturb arm joints, settle the scene, then
shuffle small free objects among compatible support heights. Object orientation
is preserved and resting height is adjusted to the destination. Conservative
AABB rejection checks other objects and fixtures; scenes without a feasible
permutation retain their original placement. A second settling pass precedes
the lift baseline. `info['shuffle_mapping']` records the actual swaps.
Use `--env.no-shuffle-objects` for an unshuffled evaluation.

The policy receives 84 values: the original 50 relative target, wrist, joint,
gripper, action and stage features; alignment and force; a three-value task mode
and signed joint-goal remainder; plus the nearest **four undesired collision
geometries**, regardless of whether their bodies are fixed or movable. Each
obstacle slot is:

```text
[R_eef.T @ (geometry_center - gripper_position), world_AABB_span, presence_mask]
```

Candidates include collision-enabled cabinet faces, drawers, microwave walls,
stove parts, tables/floors, and movable objects. Collision masks and explicit
contact pairs determine which geometries can contact the robot; visual-only
meshes and robot self-geometry are excluded. Target geometry remains visible
when arm/palm contact with it is undesired, despite permitted finger contact.

Rank uses distance to each world AABB, so large nearby surfaces are selected
by proximity to their bounds rather than their centers. Ties use geometry ID.
Bounds and wrist-frame positions update each observation. Unused slots are
zero-padded with mask zero. Infinite planes use the closest point on the plane
and a finite 2 m square surface patch for their span, avoiding unbounded inputs.
Four slots still limit local coverage; they do not describe the complete scene.
These are privileged simulator observations. Training and live deployment use
the same measurement code.

## Train and evaluate

Run from the repository root with the existing interpreter:

```bash
export NUMBA_CACHE_DIR=/tmp/libero_numba
export MUJOCO_GL=egl
PYTHON=/home/ramtin/miniconda3/envs/libero/bin/python

# Start with a focused policy and inspect calibration.jsonl before scaling up.
$PYTHON -m recovery.train --suite libero_object --task-id 0 \
  --run-name object0_geometry_v2

# Different reset seed, frozen normalization, collision and success metrics.
$PYTHON -m recovery.evaluate \
  --checkpoint recovery/runs/object0_geometry_v2/checkpoint.pt \
  --suite libero_object --task-id 0 --episodes 20 --num-envs 2 --seed 101

# After focused experiments support the reward/collision settings:
$PYTHON -m recovery.train --run-name task_geometry_v2
```

Evaluation restores saved environment settings. For a deliberate distribution
change, add `--no-use-checkpoint-env` and the desired `--env.*` overrides.
Use the same task selection and training configuration with `--resume` to
continue a run. The new observation schema is `collision_geometry_v2`.
**All earlier checkpoints require retraining**, including 84-input
`task_collision_v1` checkpoints: their obstacle slots represented movable
objects in world coordinates. Equal input width does not imply compatible
feature meanings. The loader rejects these checkpoints explicitly. Resume also
rejects mismatched reward settings.
The old run in `runs/grasp_object0` is preserved.

PPO uses tanh-squashed actions, separate termination/truncation flags, time-limit
bootstrapping, observation/reward normalization, and checkpointed optimizer
state. Checkpoints also store the environment config and observation schema.

```python
from recovery.policy import ReachPolicy
policy = ReachPolicy('recovery/runs/object0_geometry_v2/checkpoint.pt', env)
obs, outcome = policy.run(obs, max_steps=300)
```

## Validation

```bash
$PYTHON -m unittest test_recovery test_recovery_sim
```

Tests cover numerical reward terms and thresholds, contact exemptions, transient
physical arm impacts, deterministic shuffling, joint goal predicates, and
training/deployment observation parity, alongside PPO regressions.

`runs/reward_collision_smoke` is a short integration-test checkpoint, not a
validated recovery policy. `runs/reward_validation/task_audit.json` records
reset/step checks across the benchmark; these checks do not measure learned
success. Substantial training and held-out evaluation are still required before
calling a policy reliable.

`runs/research_calibration_smoke` tested the preceding observation schema.
`runs/geometry_progress_smoke` checks the current schema and integrated progress
through PPO and calibration logging. Neither is a policy-quality experiment.
