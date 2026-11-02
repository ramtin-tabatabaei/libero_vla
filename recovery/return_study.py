"""Bounded multi-task collection, training, and untouched-task physical testing."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/libero_return_numba')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/libero_return_mpl')

import argparse
import json
from pathlib import Path


TRAIN_TASKS = ['libero_goal:1', 'libero_goal:2', 'libero_object:3', 'libero_object:5']
VALIDATION_TASKS = ['libero_goal:4', 'libero_object:6']
TEST_TASKS = ['libero_goal:9', 'libero_object:7']


def collect_group(tasks, root, args, decisions_only=False):
    import subgoal_planner as sp
    from waypoint_gmm import GMMChain
    from waypoint_gmm_editor import GMMSession
    from waypoint_recovery_benchmark import prepare_scene
    from waypoint_recovery_images import plan_path
    from recovery.return_collect import collect
    root.mkdir(parents=True, exist_ok=True)
    session = GMMSession(sp.Args(seed=args.seed, policy_port=args.port))
    summary = []
    try:
        for task_key in tasks:
            suite, task = task_key.split(':')
            task = int(task)
            session.load(suite, task)
            path = plan_path(suite, task)
            if path is None:
                raise FileNotFoundError(f'No plan for {task_key}')
            chain = GMMChain.load(path, suite, task)
            for scene in args.scenes:
                for kind, intensity in [('transition', 'low'), ('grasp_loss', 'high')]:
                    spec = dict(suite=suite, task_id=task, scene=scene, kind=kind,
                                intensity=intensity, seed=args.seed)
                    prepare_scene(session, chain, spec, root / 'layouts')
                    print(f'Collecting {task_key} {scene} {kind}/{intensity}', flush=True)
                    result = collect(session, chain, spec, root, 'pi05_libero@standard_cached',
                        phases=args.phases, repeats=args.repeats, port=args.port,
                        max_policy_steps=args.max_policy_steps, stable_seconds=.5,
                        decisions_only=decisions_only, resume_contact_mode='observe',
                        max_candidates=args.max_candidates)
                    summary.append(dict(spec, **result))
                    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    finally:
        if session.env is not None:
            session.env.close()
    return summary


def main():
    from waypoint_demo import PolicyConnection
    from serve_pi05_libero import metadata
    from recovery.return_points import load_transitions
    from recovery.return_train import train
    from recovery.return_evaluate import evaluate_physical
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--phases', type=float, nargs='+', default=[.1, .75])
    parser.add_argument('--scenes', nargs='+', choices=['original', 'shuffled'], default=['original', 'shuffled'])
    parser.add_argument('--max-candidates', type=int, default=8)
    parser.add_argument('--max-policy-steps', type=int, default=400)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--test-repeats', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--train-tasks', nargs='+', default=TRAIN_TASKS)
    parser.add_argument('--validation-tasks', nargs='+', default=VALIDATION_TASKS)
    parser.add_argument('--test-tasks', nargs='+', default=TEST_TASKS)
    args = parser.parse_args()
    groups = [set(args.train_tasks), set(args.validation_tasks), set(args.test_tasks)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        parser.error('Train/validation/final-test task groups must be disjoint')
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        parser.error('Choose a new study output directory')
    connection = PolicyConnection(args.port)
    try:
        server = connection.connect()
        if any(server.get(key) != value for key, value in metadata().items()):
            raise ValueError('Running VLA server differs from the fixed Pi 0.5 contract')
    finally:
        connection.close()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'protocol.json').write_text(json.dumps(dict(vars(args), server=server,
        recovery_contact_mode='stop', resume_contact_mode='observe',
        test_controllers=['rl', 'resume_now', 'latest_reachable']), indent=2) + '\n')
    collect_group(args.train_tasks + args.validation_tasks, root / 'training', args)
    rows = load_transitions(root / 'training' / 'transitions.jsonl')
    print(f'Training on {len(rows)} measured action/outcome records', flush=True)
    checkpoint = train(rows, root / 'model', epochs=args.epochs, hidden=128,
        learning_rate=1e-3, validation_tasks=args.validation_tasks, seed=17)
    # Final test faults and controllers are executed only after the model freezes.
    collect_group(args.test_tasks, root / 'test_faults', args, decisions_only=True)
    result = evaluate_physical(checkpoint, root / 'test_faults', root / 'physical_test',
                              repeats=args.test_repeats, port=args.port)
    (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
