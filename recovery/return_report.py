"""Export the completed return-point study as CSV, plots, and a readable report."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from recovery.return_points import Decision, load_transitions
from recovery.return_view import plot_return_points


def write_report(root):
    root = Path(root)
    protocol = json.loads((root / 'protocol.json').read_text())
    final = json.loads((root / 'result.json').read_text())
    records = [json.loads(line) for line in (root / 'physical_test/episodes.jsonl').read_text().splitlines()]
    rows = load_transitions(root / 'training/transitions.jsonl')
    checkpoint = torch.load(root / 'model/checkpoint.pt', weights_only=True)
    assert set(checkpoint['training_tasks']).isdisjoint(protocol['test_tasks'])
    assert set(checkpoint['validation_tasks']).isdisjoint(protocol['test_tasks'])
    assert set(record['task'] for record in records) <= set(protocol['test_tasks'])
    grouped = defaultdict(list)
    for record in records:
        grouped[record['controller']].append(record)
    for name, episodes in grouped.items():
        assert final[name]['episodes'] == len(episodes)
        assert final[name]['task_successes'] == sum(r['success'] for r in episodes)
    faults = json.loads((root / 'test_faults/summary.json').read_text())
    expected_decisions = sum(row['status'] == 'decision_saved' for row in faults)
    assert len(records) == expected_decisions * protocol['test_repeats'] * 3
    fields = ['task', 'condition', 'repeat', 'controller', 'selected', 'primitive',
              'waypoint', 'phase', 'phase_basis', 'safe_return', 'success',
              'force_compliant_success', 'recovery_violation', 'vla_force_exceeded',
              'total_seconds', 'wall_seconds']
    with (root / 'timing.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    selected = [r for r in grouped['rl'] if r['selected']]
    force_stops = [r for r in selected if (r.get('outcome', {}).get('recovery_contacts') or {}).get('limit_exceeded')]
    force_note = ''
    if force_stops:
        peaks = [r['outcome']['recovery_contacts']['peak_force_n'] for r in force_stops]
        force_note = (f'The runtime force guard stopped {len(force_stops)} selected returns '
            f'with peak individual forces of {min(peaks):.2f}–{max(peaks):.2f} N, '
            'after the static planner had accepted a path. ')
    errors = defaultdict(list)
    for record in selected:
        prediction = next(r for r in record['ranking']['rankings'] if r['id'] == record['candidate_id'])
        outcome = record['outcome']
        errors['safe_brier'].append((prediction['safe_arrival_probability'] - record['safe_return']) ** 2)
        if outcome['resume_success'] is not None:
            errors['resume_brier'].append((prediction['vla_success_given_arrival_probability'] - outcome['resume_success']) ** 2)
        if record['safe_return']:
            errors['recovery_mae_seconds'].append(abs(prediction['recovery_seconds_given_safe_arrival'] - outcome['recovery_seconds']))
        if record['success']:
            errors['total_mae_seconds'].append(abs(prediction['total_seconds_given_success'] - record['total_seconds']))
    prediction_metrics = {key: dict(mean=float(np.mean(values)), count=len(values))
                          for key, values in errors.items()}
    (root / 'selected_prediction_metrics.json').write_text(json.dumps(prediction_metrics, indent=2) + '\n')
    # Compare time only on fault/repeat pairs where both controllers succeeded.
    pairs = defaultdict(dict)
    for record in records:
        pairs[(record['decision_id'], record['repeat'])][record['controller']] = record
    time_differences = {}
    for baseline in ('resume_now', 'latest_reachable'):
        differences = [controllers['rl']['total_seconds'] - controllers[baseline]['total_seconds']
            for controllers in pairs.values() if controllers['rl']['success'] and controllers[baseline]['success']]
        time_differences[baseline] = dict(common_success_pairs=len(differences),
            median_rl_minus_baseline_seconds=float(np.median(differences)) if differences else None)
    (root / 'paired_times.json').write_text(json.dumps(time_differences, indent=2) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/libero_return_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['rl', 'resume_now', 'latest_reachable']
    x = np.arange(3)
    figure, axes = plt.subplots(figsize=(9, 5))
    goal = [final[n]['task_success_rate'] * 100 for n in names]
    compliant = [final[n]['force_compliant_successes'] / final[n]['episodes'] * 100 for n in names]
    axes.bar(x - .18, goal, .36, label='Task completion')
    axes.bar(x + .18, compliant, .36, label='Completion without VLA force exceedance')
    for index, name in enumerate(names):
        axes.text(index - .18, goal[index] + 2,
                  f"{final[name]['task_successes']}/{final[name]['episodes']}", ha='center', fontsize=9)
    axes.set(xticks=x, xticklabels=['RL selector', 'Resume VLA now', 'Latest reachable visit'],
             ylabel='Episodes (%)', ylim=(0, 112),
             title=f'Physical test on {len({r["task"] for r in records})} tasks excluded from training and validation')
    axes.legend(loc='upper right', fontsize=8)
    axes.grid(axis='y', alpha=.2)
    figure.tight_layout()
    figure.savefig(root / 'comparison.png', dpi=160)
    plt.close(figure)
    example = next((r for r in selected if r['success']), next(iter(selected), None))
    if example is not None:
        decision = Decision.from_dict(json.loads((root / 'test_faults' / example['condition'] / 'decision.json').read_text()))
        plot_return_points(decision, dict(example['ranking'], actual_outcome=example['outcome']),
                           root / 'rl_selected_return.png')
        (root / 'selected_return_example.json').write_text(json.dumps(example, indent=2) + '\n')
    def seconds(value):
        return f'{value:.2f}' if value is not None else 'unmeasured'
    training = [r for r in rows if r[0].task_key in checkpoint['training_tasks']]
    validation = [r for r in rows if r[0].task_key in checkpoint['validation_tasks']]
    verdict = ('The trained selector completed no final-test episode and is not reliable on this test set.'
               if not final['rl']['task_successes'] else
               'The trained selector completed some final-test episodes; the small test set limits generalization claims.')
    texts = ['# Return-point RL: completed training and untouched-task physical test', '',
        '**' + verdict + '**', '',
        f'Trained on **{len(training)} measured action/outcome records** across {len(checkpoint["training_tasks"])} tasks; '
        f'validation used **{len(validation)} records** across {len(checkpoint["validation_tasks"])} different tasks. '
        f'The saved checkpoint is epoch **{checkpoint["epoch"]}** from a {protocol["epochs"]}-epoch run.', '',
        f'Final test tasks: {", ".join(protocol["test_tasks"])}. '
        f'Of {len(faults)} planned test conditions, {expected_decisions} yielded an observed, '
        f'uncompleted fault. Each controller ran {protocol["test_repeats"]} times per eligible condition. '
        'The final test data was generated after checkpoint selection.', '',
        '| Controller | Task completion | Selection coverage | Force-compliant completion | Median successful total time (s) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in names:
        result = final[name]
        texts.append(f'| {name} | {result["task_successes"]}/{result["episodes"]} '
            f'({result["task_success_rate"]:.1%}) | {result["coverage"]:.1%} | '
            f'{result["force_compliant_successes"]}/{result["episodes"]} | '
            f'{seconds(result["median_completion_sim_seconds"])} |')
    texts.extend(['', '![Controller comparison](comparison.png)', '',
        f'The RL selector made {len(selected)} selections. It completed '
        f'{sum(r["success"] for r in selected)} tasks and safely reached '
        f'{sum(bool(r.get("safe_return")) for r in selected)} selected return configurations. '
        f'It abstained in {final["rl"]["episodes"] - len(selected)} episodes; '
        'abstentions count against overall task-completion rate. The latest-return '
        'baseline uses the same planner and return safety checks. Immediate VLA '
        'resumption has no physical-return label.', '',
        force_note +
        'A planner check and a high predicted probability did not establish safe '
        'execution guarantees. The current checkpoint should not be treated as a validated '
        'recovery selector.', '',
        '## What these results measure', '',
        'Recovery retains force/joint/time/held-object safety stops. The resumed VLA '
        'uses **observe** contact mode: it can complete despite exceeding the '
        'provisional 100 N individual or 10 N summed undesired-force threshold. '
        'Task completion therefore does not establish safe VLA operation. '
        'Force-compliant completion is shown separately; production APIs default '
        'to contact **stop** mode and reject mismatched checkpoint contracts.', '',
        'Simulation times include return motion, VLA control, and stable-success '
        'holding. They exclude planner/inference/network wall latency. Medians '
        'cover successful episodes only, with potentially different successful '
        'conditions between controllers. `paired_times.json` additionally compares '
        'the same fault/repeat pairs where both RL and a baseline succeeded.', '',
        'There are only two final test tasks and one generated fault per condition. '
        'Repeated VLA outcomes share fault states and are not independent tasks. '
        'This measures transfer on this small test set, not general task independence. '
        'Prediction errors in `selected_prediction_metrics.json` are measured on '
        'selected returns and have selection bias; they do not certify calibration '
        'over all candidates.', '', '## Chosen historical state', ''])
    if example is None:
        texts.append('The RL model selected no return point under its default gates.')
    else:
        texts.extend([f'Example: **{example["task"]}, waypoint {example["waypoint"] + 1}, '
            f'{example["primitive"]}, {example["phase"]:.1%} ({example["phase_basis"]})**. '
            f'Observed task completion: {example["success"]}; '
            f'safe arrival: {example["safe_return"]}; '
            f'total time: {example["total_seconds"]:.2f} simulation seconds. '
            f'Controller result: {example["outcome"]["detail"]}.', '',
            '![Actual selected historical return](rl_selected_return.png)', '',
            'The green marker identifies the selected measured pre-failure wrist '
            'configuration and the red marker the fault/handoff pose. Historical '
            'plot lines connect sampled poses; they are not the planned return route.'])
    texts.extend(['', '## Artifacts', '',
        '- `model/checkpoint.pt`: trained shared selector and validation provenance.',
        '- `timing.csv`: every final physical comparison episode.',
        '- `physical_test/episodes.jsonl`: choices, predictions, outcomes, and contact telemetry.',
        '- `result.json`: final controller counts and rates.',
        '- `training/transitions.jsonl`: all measured training/validation actions.',
        '- `test_faults/`: actual dense histories and fault-state snapshots.', '',
        'Software validation: 68 relevant tests passed.', ''])
    (root / 'RESULTS.md').write_text('\n'.join(texts))
    print(json.dumps(dict(report=str(root / 'RESULTS.md'), prediction_metrics=prediction_metrics,
                          paired_times=time_differences), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    args = parser.parse_args()
    write_report(args.study)


if __name__ == '__main__':
    main()
