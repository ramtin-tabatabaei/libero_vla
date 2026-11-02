"""Retry physically unrealized faults at other eligible rows, preserving attempts."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import subgoal_planner as sp
from waypoint_failures import allowed_intensities, waypoint_primitive
from waypoint_recovery_benchmark import ROOT, execute_trial, summarize, write_json


def alternative_rows(chain, result):
    tried = {result['injection_row']} | {a['injection_row'] for a in result.get('attempts', [])}
    return [i for i, w in enumerate(chain.waypoints) if i not in tried and
            result['intensity'] in allowed_intensities(result['kind'], waypoint_primitive(w), w.motion)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--trial-id', help='Restrict retries to one missing condition')
    args = parser.parse_args()
    manifest = json.loads((args.folder/'manifest.json').read_text())
    results = json.loads((args.folder/'results.json').read_text())
    if len(results) != len(manifest['scenarios']):
        parser.error('Finish the main batch before retrying; its result file must have one entry per scenario')
    for plan in manifest['tasks']:
        if hashlib.sha256(Path(plan['plan']).read_bytes()).hexdigest() != plan['sha256']:
            raise ValueError('A saved plan changed since the main batch')
    from waypoint_demo import PolicyConnection
    from waypoint_gmm import GMMChain
    import waypoint_gmm_editor as editor
    from waypoint_recovery_metrics import export_timing_csv
    client = PolicyConnection(manifest['port'])
    try:
        metadata = client.connect()
    finally:
        client.close()
    if metadata != manifest['policy']:
        raise ValueError('The policy metadata changed since the main batch')
    options = SimpleNamespace(port=manifest['port'], delay=manifest['delay_seconds'],
        max_policy_steps=manifest['max_policy_steps'],
        max_waypoint_seconds=manifest['max_waypoint_seconds'], no_video=not manifest['save_video'],
        layout_root=args.folder/'layouts')
    editor.MODEL_STORE = args.folder/'waypoint_archives'
    session = editor.GMMSession(sp.Args(policy_port=options.port, policy_max_steps=options.max_policy_steps))
    current = None
    try:
        for index, original in enumerate(results):
            if original['status'] != 'failure_not_observed' or args.trial_id and original['trial_id'] != args.trial_id:
                continue
            key = original['suite'], original['task_id']
            if key != current:
                session.load(*key, cameras=True)
                current = key
            chain = GMMChain.load(ROOT/'saved_plans'/key[0]/('task_%03d.json'%key[1]), *key)
            candidates = alternative_rows(chain, original)
            attempts = list(original.get('attempts', []))
            if not attempts:
                attempts.append(dict(injection_row=original['injection_row'], status=original['status'],
                                     path='trials/'+original['trial_id']))
            for row in candidates:
                spec = next(dict(s) for s in manifest['scenarios'] if all(
                    s.get(k)==original.get(k) for k in ('suite','task_id','scene','mode','kind','intensity')))
                spec['injection_row'] = row
                relative = Path('attempts')/original['trial_id']/('attempt_%02d'%(len(attempts)+1))
                folder = args.folder/relative
                started = time.monotonic()
                print('RETRY', original['trial_id'], 'row', row, flush=True)
                try:
                    entry = execute_trial(session, chain, spec, options, folder, manifest['prompt_window_seconds'])
                except Exception as exc:
                    entry = dict(spec, trial_id=original['trial_id'], success=False,
                                 status='error', error=type(exc).__name__+': '+str(exc))
                entry['wall_seconds'] = round(time.monotonic()-started, 3)
                write_json(folder/'result.json', entry)
                old_state = json.loads((args.folder/'trials'/original['trial_id']/'initial.state.json').read_text())
                new_state = json.loads((folder/'initial.state.json').read_text())
                if old_state != new_state:
                    raise ValueError('Retry initial state differs from the paired main-batch state')
                attempts.append(dict(injection_row=row, status=entry['status'], path=str(relative)))
                if entry.get('observed_failure'):
                    previous = args.folder/'trials'/original['trial_id']
                    backup = args.folder/'attempts'/original['trial_id']/'attempt_01'
                    previous.rename(backup)
                    attempts[0]['path'] = str(backup.relative_to(args.folder))
                    entry['attempts'] = attempts
                    entry['selected_attempt'] = len(attempts)
                    folder.rename(previous)
                    attempts[-1]['path'] = str(previous.relative_to(args.folder))
                    results[index] = entry
                    write_json(previous/'result.json', entry)
                else:
                    original['attempts'] = attempts
                    results[index] = original
                    write_json(args.folder/'trials'/original['trial_id']/'result.json', original)
                write_json(args.folder/'results.json', results)
                write_json(args.folder/'summary.json', summarize(results))
                export_timing_csv(args.folder, results)
                print('RETRY RESULT', original['trial_id'], entry['status'], flush=True)
                if entry.get('observed_failure'):
                    break
    finally:
        if session.env is not None:
            session.env.close()


if __name__ == '__main__':
    main()
