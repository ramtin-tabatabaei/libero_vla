# Multi-task training and physical test of the return-point selector

Run the bounded study from the project root:

```bash
/home/ramtin/miniconda3/envs/libero/bin/python -m recovery.return_study \
  --output experiments/return_point_training_test_20261012 \
  --max-candidates 8 --epochs 150 --test-repeats 2
```

The standard Pi 0.5 server must be running on local port 8000. The entry point
sets process-local EGL and cache settings; the small selector trains on CPU.
Use a fresh output directory to preserve previous measurements.

## Predeclared split and experiment

| Split | Tasks |
| --- | --- |
| Training | libero_goal:1, libero_goal:2, libero_object:3, libero_object:5 |
| Validation | libero_goal:4, libero_object:6 |
| Final physical test | libero_goal:9, libero_object:7 |

The study uses original and shuffled scenes, Low transition deviations, and High
grasp loss. It samples actual earlier visits near 10% and 75% of each primitive
segment, caps the set at eight chronologically distributed measured candidates,
and tries each training/validation candidate once from the same saved fault world.
Missing observed faults and already-completed tasks are reported separately.
There is no fabricated VLA outcome for a failed or unsafe return.

Pi 0.5 receives at most 400 control steps and must maintain task success for 0.5
simulation seconds. The network trains for 150 epochs with hidden width 128,
learning rate 0.001, and seed 17. Validation chooses the saved checkpoint using Q
error plus safe-arrival Brier score. The final test tasks are generated only after
that checkpoint is frozen, and the test evaluator rejects tasks that appeared in
training or validation.

Environment and planner randomness are seeded. The fault injector's random
direction and the VLA's inference randomness are not rewound. Recorded dense
history and saved fault states retain the actual experiment for comparison.
This is a small benchmark experiment, not a statistical certificate of generalization.

## What changed after the strict pilot

The earlier pilot stopped the resumed VLA at the return controller's provisional
force thresholds. A new audit recorded the offending geometry and forces. Its
four measured stops on libero_goal:1 were approximately 10.74–12.00 N summed
undesired palm/finger contact with the target bowl, with individual forces below
6 N. These were threshold stops, not 100 N impacts. The object-task audit had no
observed fault and is excluded from outcome counts.

For this simulation study, the recovery move still stops on excessive force,
joint-limit violation, timeout, or changed held-object state. The resumed VLA
uses `resume_contact_mode='observe'`: contacts and threshold exceedances are
recorded while the VLA finishes or reaches its horizon. This separates the
requested safe-return prediction from task-completion prediction for the frozen
VLA. It does not establish that the whole VLA trajectory is safe.

The deployment API and original collector retain `resume_contact_mode='stop'`
by default. The contract is stored in every decision/checkpoint and must match at
inference; datasets with different contact contracts cannot be mixed. Old pilot
results are preserved. New checkpoints predict success under the recorded
observation contract, so they are not interchangeable with strict-stop checkpoints.

`Outcome.success` means safe physical return followed by VLA goal completion
without an enforced stop. `Outcome.force_compliant_success` additionally requires
no VLA force-limit exceedance. Contact telemetry records peak individual force,
peak undesired sum, the first offending contact geometries, and the number of
physics steps over a threshold. Both success measures are reported.

## Final physical comparison

Each observed test fault is restored before every controller/repeat. Controller
order is shuffled with seed 2718; VLA inference noise is not identically replayed.
The default test repeats each condition twice and compares:

- **RL:** the frozen selector with its default probability/support gates and a
  current-scene path check. An abstention is counted in the total denominator.
- **resume_now:** give the VLA control directly at the fault state, without a return.
- **latest_reachable:** physically return to the most recently visited candidate
  for which the same recovery planner finds a path, then resume the VLA.

Metrics include selection coverage, total task success, success without force
exceedance, recovery safety outcomes, VLA force exceedances, and median total
simulation completion time among successful episodes. Immediate resumption has
no return-attempt/safe-arrival label. Wall durations are recorded separately.
Unobserved fault conditions are excluded and their coverage must be reported.

Artifacts under the study directory:

- `protocol.json`: settings, split, and running server metadata.
- `training/transitions.jsonl`: measured training and validation candidates/outcomes.
- `training/summary.json`: collected and excluded condition counts.
- `model/checkpoint.pt`, `history.json`, `evaluation.json`: saved training and
  validation metrics.
- `test_faults/`: untouched-task decisions, dense prior visits, and physical fault states.
- `physical_test/episodes.jsonl`, `summary.json`, `configuration.json`: fresh,
  matched physical comparisons, including individual return choices and rankings.
- `result.json`: final controller summary after the full study finishes.

The final results report is written after collection, training, and physical
testing complete. Intermediate counts are not final test results.
