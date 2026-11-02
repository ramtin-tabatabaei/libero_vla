"""Collect counterfactual returns from the SAME fault state, then resume Pi 0.5."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from recovery.return_points import Decision, write_transition
from recovery.return_sim import (PhysicalReturnController, RecordVisits, ReturnSafety,
                                 resume_from_return, snapshot)


def restore_fault(session, state, robot_state):
    """Offline experiment reset; live recovery itself never rewinds the world."""
    from waypoint_demo import switch_controller
    session.env.set_init_state(state.copy())
    switch_controller(session.env, 'JOINT_POSITION')
    session.attach(settle=False)
    session.executor.gripper = robot_state.gripper_command


def collect(session, chain, spec, output, vla_id, phases=(0., .1, .25, .5, .75, 1.),
            repeats=1, delay=0., max_waypoint_seconds=180., max_policy_steps=400,
            stable_seconds=1., port=8000, safety=None, decisions_only=False,
            resume_contact_mode='stop', max_candidates=None):
    from waypoint_recovery_benchmark import run_fault, trial_id, write_json
    if repeats < 1 or max_policy_steps < 1:
        raise ValueError('Repeat count and rollout step limit must be positive')
    output = Path(output)
    folder = output / trial_id(spec)
    if folder.exists():
        raise FileExistsError(f'{folder} already exists; use a new output directory')
    folder.mkdir(parents=True)
    # Enable physical robot collision meshes before visits and every return.
    from recovery.scene import enable_robot_collisions
    enable_robot_collisions(session.world)
    with RecordVisits(session) as recorder:
        fault = run_fault(session, chain, spec, delay, max_waypoint_seconds, folder,
                          capture=lambda *_: None)
    write_json(folder / 'fault.json', fault)
    if not fault['observed_failure']:
        return dict(status='failure_not_observed', rollouts=0)
    if fault.get('success_before_policy'):
        return dict(status='already_successful', rollouts=0)
    failure_time = fault['failure_event']['sim_time']
    points = recorder.history.candidates(failure_time, phases)
    if max_candidates is not None:
        if max_candidates < 2:
            raise ValueError('Candidate cap must retain at least two visits')
        if len(points) > max_candidates:
            ordered = sorted(points, key=lambda point: point.state.time)
            indices = np.linspace(0, len(ordered) - 1, max_candidates).round().astype(int)
            points = [ordered[index] for index in indices]
    if not points:
        return dict(status='no_pre_failure_visits', rollouts=0)
    current = snapshot(session)
    saved_state = session.env.sim.get_state().flatten().copy()
    controller = PhysicalReturnController(session, safety)
    identifier = trial_id(spec) + ':' + hashlib.sha256(saved_state.tobytes()).hexdigest()[:16]
    decision = Decision(identifier, f"{spec['suite']}:{spec['task_id']}",
                        vla_id, controller.id, failure_time, current, points,
                        max_policy_steps=max_policy_steps, stable_seconds=stable_seconds,
                        resume_contact_mode=resume_contact_mode)
    decision.validate()
    write_json(folder / 'decision.json', asdict(decision))
    write_json(folder / 'visited_history.json',
               [dict(segment, states=[asdict(s) for s in segment['states']])
                for segment in recorder.history.segments])
    write_json(folder / 'failure_state.json', dict(state=saved_state.tolist()))
    if decisions_only:
        return dict(status='decision_saved', candidates=len(points), rollouts=0)
    completed = 0
    try:
        for repeat in range(repeats):
            for candidate in points:
                restore_fault(session, saved_state, current)
                outcome = resume_from_return(session, candidate,
                    PhysicalReturnController(session, safety), chain.language,
                    port, max_policy_steps, stable_seconds, resume_contact_mode=resume_contact_mode)
                write_transition(output / 'transitions.jsonl', decision, candidate.id, outcome)
                write_json(folder / f'return_{candidate.id.replace(":", "_")}_{repeat}.json',
                    dict(candidate=asdict(candidate), outcome=asdict(outcome)))
                completed += 1
                print(f"{decision.id}: waypoint {candidate.waypoint + 1} "
                      f"{candidate.primitive} {candidate.phase:.0%} ({candidate.phase_basis}); "
                      f"safe arrival={outcome.safe_reached}, VLA success={outcome.resume_success}", flush=True)
    finally:
        restore_fault(session, saved_state, current)
    return dict(status='collected', candidates=len(points), rollouts=completed)


def main():
    import subgoal_planner as sp
    from waypoint_gmm import GMMChain
    from waypoint_gmm_editor import GMMSession
    from waypoint_recovery_benchmark import prepare_scene
    from waypoint_recovery_images import plan_path
    from waypoint_demo import PolicyConnection
    from serve_pi05_libero import metadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', required=True, help='Task keys, e.g. libero_goal:1 libero_object:3')
    parser.add_argument('--output', required=True)
    parser.add_argument('--vla-id', required=True, help='Pinned VLA identity, e.g. pi05_libero@standard_cached')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--kinds', nargs='+', default=['transition', 'orientation'])
    parser.add_argument('--intensities', nargs='+', choices=['low', 'high'], default=['low'])
    parser.add_argument('--scenes', nargs='+', choices=['original', 'shuffled'], default=['original', 'shuffled'])
    parser.add_argument('--phases', nargs='+', type=float, default=[0., .1, .25, .5, .75, 1.])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--delay', type=float, default=0.)
    parser.add_argument('--max-waypoint-seconds', type=float, default=180.)
    parser.add_argument('--max-policy-steps', type=int, default=400)
    parser.add_argument('--stable-seconds', type=float, default=1.)
    parser.add_argument('--decisions-only', action='store_true')
    parser.add_argument('--resume-contact-mode', choices=['stop', 'observe'], default='stop')
    args = parser.parse_args()
    from waypoint_failures import FAILURES
    if any(kind not in FAILURES for kind in args.kinds):
        parser.error('Unknown failure type')
    if not args.vla_id.startswith(metadata()['model_id'] + '@'):
        parser.error('--vla-id must pin the Pi 0.5 checkpoint as pi05_libero@<version>')
    # Fail quickly on server/metadata errors before generating a fault trajectory.
    connection = PolicyConnection(args.port)
    try:
        server = connection.connect()
        if any(server.get(key) != value for key, value in metadata().items()):
            raise ValueError('The running server does not match the fixed Pi 0.5 rollout contract')
    finally:
        connection.close()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error('Use a new output directory to preserve existing experiments')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'configuration.json').write_text(json.dumps(dict(vars(args), server=server), indent=2) + '\n')
    session = GMMSession(sp.Args(seed=args.seed, policy_port=args.port))
    summary = []
    try:
        for task_key in args.tasks:
            suite, task_text = task_key.split(':')
            task = int(task_text)
            path = plan_path(suite, task)
            if path is None:
                raise FileNotFoundError(f'No saved waypoint plan for {task_key}')
            session.load(suite, task)
            chain = GMMChain.load(path, suite, task)
            for scene in args.scenes:
                for kind in args.kinds:
                    for intensity in args.intensities:
                        spec = dict(suite=suite, task_id=task, scene=scene, kind=kind,
                                    intensity=intensity, seed=args.seed)
                        prepare_scene(session, chain, spec, output / 'layouts')
                        result = collect(session, chain, spec, output, args.vla_id, args.phases,
                            args.repeats, args.delay, args.max_waypoint_seconds, args.max_policy_steps,
                            args.stable_seconds, args.port, decisions_only=args.decisions_only,
                            resume_contact_mode=args.resume_contact_mode)
                        summary.append(dict(spec, **result))
                        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        if session.env is not None:
            session.env.close()


if __name__ == '__main__':
    main()
