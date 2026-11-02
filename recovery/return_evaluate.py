"""Execute a frozen selector and baselines from matched, held-out fault states."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from recovery.return_collect import restore_fault
from recovery.return_model import ReturnPointSelector
from recovery.return_points import Decision
from recovery.return_sim import (PhysicalReturnController, ReturnSafety, ReturnViolation,
                                 resume_from_return, watch_contacts)


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record['controller']].append(record)
    summary = {}
    for name, rows in groups.items():
        successful = [r for r in rows if r['success']]
        attempted = [r for r in rows if r.get('return_attempted')]
        summary[name] = dict(episodes=len(rows), selections=sum(r['selected'] for r in rows),
            coverage=sum(r['selected'] for r in rows) / len(rows),
            task_successes=len(successful), task_success_rate=len(successful) / len(rows),
            force_compliant_successes=sum(r['force_compliant_success'] for r in rows),
            recovery_attempts=len(attempted),
            safe_returns=sum(r.get('safe_return', False) for r in attempted),
            recovery_violations=sum(r.get('recovery_violation', False) for r in attempted),
            vla_force_exceedances=sum(r.get('vla_force_exceeded', False) for r in rows),
            median_completion_sim_seconds=float(np.median([r['total_seconds'] for r in successful]))
                if successful else None)
    return summary


def direct_continuation(session, decision, language, port, safety):
    from waypoint_policy import run
    start = float(session.env.sim.data.time)
    success, stopped = False, False
    with watch_contacts(session, safety, stop=decision.resume_contact_mode == 'stop') as contacts:
        try:
            result = run(session, language, port=port, backend='pi05',
                max_steps=decision.max_policy_steps, stable_seconds=decision.stable_seconds)
            success = bool(result['success'])
        except ReturnViolation:
            stopped = True
    return dict(success=success and not stopped,
        force_compliant_success=success and not contacts['limit_exceeded'],
        vla_force_exceeded=contacts['limit_exceeded'],
        total_seconds=float(session.env.sim.data.time) - start,
        contacts=contacts, selected=True, return_attempted=False, safe_return=None)


def evaluate_physical(checkpoint, experiment, output, repeats=2, port=8000, seed=2718,
                      controllers=('rl', 'resume_now', 'latest_reachable')):
    import subgoal_planner as sp
    from waypoint_gmm_editor import GMMSession
    from recovery.scene import enable_robot_collisions
    selector = ReturnPointSelector(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    excluded = set(saved['training_tasks']) | set(saved['validation_tasks'])
    paths = sorted(Path(experiment).glob('*/decision.json'))
    if not paths:
        raise ValueError('No held-out fault decisions found')
    decisions = [Decision.from_dict(json.loads(path.read_text())) for path in paths]
    if any(d.task_key in excluded for d in decisions):
        raise ValueError('Final test tasks overlap checkpoint training or validation tasks')
    if repeats < 1 or any(name not in ('rl', 'resume_now', 'latest_reachable') for name in controllers):
        raise ValueError('Invalid evaluation configuration')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'episodes.jsonl'
    if records_path.exists():
        raise FileExistsError('Preserve existing evaluation; choose a fresh output directory')
    (output / 'configuration.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),
        experiment=str(experiment), repeats=repeats, seed=seed, controllers=list(controllers),
        test_tasks=sorted({d.task_key for d in decisions}), excluded_tasks=sorted(excluded)), indent=2) + '\n')
    rng = np.random.default_rng(seed)
    session = GMMSession(sp.Args(seed=1000, policy_port=port))
    records = []
    try:
        for path, decision in zip(paths, decisions):
            suite, task = decision.task_key.split(':')
            chain = session.load(suite, int(task))
            enable_robot_collisions(session.world)
            fault_state = np.asarray(json.loads(path.with_name('failure_state.json').read_text())['state'])
            for repeat in range(repeats):
                for name in rng.permutation(list(controllers)):
                    restore_fault(session, fault_state, decision.current)
                    controller = PhysicalReturnController(session)
                    if controller.id != decision.controller_id:
                        raise ValueError('Physical controller has changed since collection')
                    started = time.monotonic()
                    base = dict(decision_id=decision.id, task=decision.task_key,
                                condition=path.parent.name, repeat=repeat, controller=str(name))
                    if name == 'resume_now':
                        result = direct_continuation(session, decision, chain.language, port, controller.safety)
                    else:
                        candidate, ranking = None, None
                        if name == 'rl':
                            ranking = selector.select(decision, feasible=controller.feasible)
                            candidate = next((c for c in decision.candidates if c.id == ranking['selected']), None)
                        else:
                            for point in sorted(decision.candidates, key=lambda c: c.state.time, reverse=True):
                                if controller.feasible(point):
                                    candidate = point
                                    break
                        if candidate is None:
                            result = dict(selected=False, success=False, force_compliant_success=False,
                                return_attempted=False, total_seconds=0., reason='no acceptable return point')
                        else:
                            outcome = resume_from_return(session, candidate, controller, chain.language,
                                port, decision.max_policy_steps, decision.stable_seconds,
                                resume_contact_mode=decision.resume_contact_mode)
                            result = dict(selected=True, return_attempted=True,
                                candidate_id=candidate.id, waypoint=candidate.waypoint,
                                primitive=candidate.primitive, phase=candidate.phase, phase_basis=candidate.phase_basis,
                                success=outcome.success, force_compliant_success=outcome.force_compliant_success,
                                safe_return=outcome.safe_reached, recovery_violation=outcome.recovery_violation,
                                vla_force_exceeded=(outcome.resume_contacts or {}).get('limit_exceeded', False),
                                total_seconds=outcome.recovery_seconds + outcome.rollout_seconds,
                                outcome=asdict(outcome))
                        if ranking is not None:
                            result['ranking'] = ranking
                    record = dict(base, **result, wall_seconds=time.monotonic() - started)
                    records.append(record)
                    with records_path.open('a') as stream:
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
                    (output / 'summary.json').write_text(json.dumps(summarize(records), indent=2) + '\n')
                    print(f"{decision.task_key} {path.parent.name} repeat {repeat + 1}: "
                          f"{name}: selected={record['selected']}, success={record['success']}, "
                          f"force-compliant={record['force_compliant_success']}", flush=True)
    finally:
        if session.env is not None:
            session.env.close()
    return summarize(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print(json.dumps(evaluate_physical(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
