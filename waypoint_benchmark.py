"""Reproducible fixed-scene VLA learning and held-out shuffle evaluation."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from waypoint_learning import Config, learn, load_model, save_model, write_json


def collect(folder, suite, task_id, successes=3, max_attempts=12):
    import subgoal_planner as sp
    from waypoint_demo import record
    from waypoint_gmm_editor import GMMSession
    folder.mkdir(parents=True, exist_ok=True)
    session = GMMSession(sp.Args())
    demos, attempts = [], []
    for path in sorted((folder/'demonstrations').glob('demo_*.json')):
        if path.name.endswith('.state.json'):
            continue
        demo = json.loads(path.read_text())
        demos.append(dict(demo, _path=str(path.resolve())))
    try:
        chain = session.load(suite, task_id, cameras=True)
        state = session.training_state.copy()
        if demos:
            from waypoint_replay import initial_state
            state = np.asarray(initial_state(demos[0], demos[0]['_path']))
            session.training_state = state
        model = None
        for attempt in range(max_attempts+1):
            if len(demos) >= successes:
                try:
                    model = learn(demos, Config(max_waypoints=250))
                    save_model(model, folder/'gmm')
                    break
                except ValueError as exc:
                    print('FIT', task_id, str(exc), flush=True)
            if attempt == max_attempts:
                break
            session.restore_training_scene()
            record(session, chain.language, max_steps=500)
            d = session.last_demonstration
            usable = d['success'] and len(d['samples']) >= 3 and not d['samples'][0]['task_success']
            target = folder/'demonstrations' if usable else folder/'failed'
            target.mkdir(parents=True, exist_ok=True)
            name = 'demo_'+datetime.now(timezone.utc).strftime('%H%M%S_%f')
            write_json(target/(name+'.state.json'), dict(initial_state=d.pop('initial_state')))
            d.update(replay_state_file=name+'.state.json', placement_mode='fixed_unshuffled')
            path = target/(name+'.json')
            write_json(path, d)
            attempts.append(dict(file=str(path), success=bool(usable), steps=len(d['samples'])))
            if usable:
                demos.append(dict(d, _path=str(path.resolve())))
            print('RECORD', task_id, len(demos), 'successes', len(d['samples']), 'steps', usable, flush=True)
            write_json(folder/'collection.json', dict(suite=suite, task_id=task_id, attempts=attempts, successes=len(demos)))
        if model is None:
            raise ValueError('Unable to fit compatible successful VLA demonstrations')
        return model
    finally:
        if session.env is not None:
            session.env.close()


def evaluate(folder, method, seeds, original=True, label='evaluation'):
    import subgoal_planner as sp
    from waypoint_gmm_editor import GMMSession
    from waypoint_gmm import chain_from_model
    from waypoint_replay import initial_state, shuffle_objects
    model = load_model(folder/'gmm')
    demo_path = Path(model['demonstration_files'][0])
    demo = json.loads(demo_path.read_text())
    state = initial_state(demo, demo_path)
    session = GMMSession(sp.Args())
    results = []
    if method != 'gmm':
        from waypoint_skill import learn_skill, run_skill
        paths = sorted((folder/'demonstrations').glob('demo_*.json'))
        demos = [dict(json.loads(p.read_text()), _path=str(p.resolve())) for p in paths if not p.name.endswith('.state.json')]
        skill = learn_skill(demos)
        write_json(folder/'skill.json', skill)
    try:
        session.load(model['suite'], model['task_id'], cameras=False)
        for seed in ([None] if original else []) + list(seeds):
            session.env.set_init_state(state)
            session.attach(settle=False)
            trace, mapping = [], None
            started = time.monotonic()
            entry = dict(method=method, suite=model['suite'], task_id=model['task_id'], seed=seed, scene='original' if seed is None else 'shuffled')
            try:
                if seed is not None:
                    mapping = shuffle_objects(session, seed=seed)
                poses = {n: [p.tolist(), r.tolist()] for n, (p,r) in session.poses().items() if n in session.world.domain.objects_dict}
                def report(i, message):
                    if message != 'Running':
                        trace.append(dict(index=i, result=message))
                if method == 'gmm':
                    outcome = session.run(chain_from_model(model), report)
                    completed = outcome.startswith('Finished')
                else:
                    out = run_skill(session, skill, report, mode=method)
                    outcome, completed = out['result'], out['completed']
                    entry['details'] = out
                task_success = bool(session.env.check_success())
                entry.update(result=outcome, completed=completed, task_success=task_success,
                             success=bool(completed and task_success), shuffle=mapping, initial_object_poses=poses)
            except Exception as exc:
                entry.update(result=type(exc).__name__+': '+str(exc), completed=False, task_success=bool(session.env.check_success()), success=False, shuffle=mapping)
            entry.update(seconds=round(time.monotonic()-started, 2), trace=trace)
            results.append(entry)
            write_json(folder/(method+'_'+label+'.json'), results)
            print('EVAL',model['task_id'],method,seed,entry['success'],entry['result'],flush=True)
    finally:
        if session.env is not None:
            session.env.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--suite', default='libero_object')
    p.add_argument('--task-id', type=int, required=True)
    p.add_argument('--stage', choices=['collect','evaluate'], required=True)
    p.add_argument('--method', default='gmm')
    p.add_argument('--seeds', nargs='*', type=int, default=[10,11])
    p.add_argument('--no-original', action='store_true')
    p.add_argument('--label', default='evaluation')
    a=p.parse_args();folder=a.root/a.suite/('task_%03d'%a.task_id)
    if a.stage=='collect':collect(folder,a.suite,a.task_id)
    else:evaluate(folder,a.method,a.seeds,not a.no_original,a.label)

if __name__=='__main__':main()
