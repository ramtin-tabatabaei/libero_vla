"""Audit the fixed-scene ten-task study and write its results table."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from waypoint_learning import write_json, load_model, save_model
from waypoint_skill_editor import fit_editor_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('waypoint_models/shuffle_study_20260926'))
    a=p.parse_args();root=a.root
    development={m:[] for m in ['gmm','keyframes','feedback','constrained']}
    heldout={m:[] for m in ['gmm','constrained']}
    tasks=[]
    for i in range(10):
        folder=root/'libero_object'/('task_%03d'%i)
        model=load_model(folder/'gmm')
        paths=[p for p in sorted((folder/'demonstrations').glob('demo_*.json')) if not p.name.endswith('.state.json')]
        demos=[dict(json.loads(p.read_text()),_path=str(p.resolve())) for p in paths]
        states=[json.loads((p.parent/d['replay_state_file']).read_text())['initial_state'] for p,d in zip(paths,demos)]
        assert len(demos)>=3 and all(d['success'] and d['placement_mode']=='fixed_unshuffled' for d in demos)
        assert all(np.array_equal(states[0],s) for s in states[1:]),'Training layouts changed'
        task=dict(task_id=i,language=model['instruction'],demonstrations=len(demos),fixed_initial_state=True)
        for m in development:
            rows=json.loads((folder/(m+'_evaluation.json')).read_text())
            assert {r['seed'] for r in rows}=={None,10,11} and len(rows)==3
            development[m]+=rows
        for m in heldout:
            rows=json.loads((folder/(m+'_heldout.json')).read_text())
            assert {r['seed'] for r in rows}=={100,101,102,103,104} and len(rows)==5
            assert len({json.dumps(r['shuffle'],sort_keys=True) for r in rows})==5,'Duplicate shuffle layouts'
            heldout[m]+=rows
            task[m+'_heldout']=sum(r['success'] for r in rows)
        robust=fit_editor_model(demos)
        robust['replay_validation']=dict(study=str(root),development_successes=sum(r['success'] for r in json.loads((folder/'constrained_evaluation.json').read_text())),
            heldout_successes=task['constrained_heldout'],heldout_trials=5,seeds=[100,101,102,103,104])
        save_model(robust,folder/'robust')
        tasks.append(task)
    paired = {(r['task_id'],r['seed']):r for r in heldout['gmm']}
    for r in heldout['constrained']:
        other = paired[(r['task_id'],r['seed'])]
        assert r['shuffle'] == other['shuffle'], 'Methods saw different permutations'
        for name, pose in r['initial_object_poses'].items():
            np.testing.assert_allclose(pose[0], other['initial_object_poses'][name][0], atol=1e-9, rtol=0)
            np.testing.assert_allclose(pose[1], other['initial_object_poses'][name][1], atol=1e-9, rtol=0)
        assert r['details']['runtime_vla_calls'] == 0
    def counts(rows):return dict(successes=sum(r['success'] for r in rows),trials=len(rows))
    summary=dict(tasks=tasks,training_demonstrations=sum(t['demonstrations'] for t in tasks),
        development={m:counts(v) for m,v in development.items()},heldout={m:counts(v) for m,v in heldout.items()},
        scope='10 LIBERO-Object pick/place tasks; fixed goal basket; supported permutations preserve object heights and orientations; privileged simulator geometry; no runtime VLA',
        failures={m:[dict(task_id=r['task_id'],seed=r['seed'],result=r['result']) for r in rows if not r['success']] for m,rows in heldout.items()},
        selected_executor_sha256=hashlib.sha256(Path('waypoint_skill.py').read_bytes()).hexdigest())
    frozen=json.loads((root/'heldout_manifest.json').read_text())
    assert summary['selected_executor_sha256']==frozen['code_hashes']['waypoint_skill.py'],'Selected algorithm changed during held-out evaluation'
    write_json(root/'summary.json',summary)
    lines=['# Ten-task waypoint shuffle study','',
           'Training: **%d successful VLA demonstrations**, with exactly the same unshuffled initial simulator state within each task.'%summary['training_demonstrations'],'',
           'Development: original + seeds 10 and 11 for each task. Held-out evaluation: seeds 100–104 (five distinct valid permutations per task), used after the selected executor was frozen. No VLA calls during replay.','',
           '| Method | Development task successes | Unseen shuffled task successes |','| --- | ---: | ---: |']
    names=dict(gmm='GMM waypoint replay',keyframes='Object-relative demonstrated keyframes',feedback='Keyframes + measured grasp/placement correction',constrained='Collision/reachability constraints + grasp/placement feedback')
    for m,rows in development.items():
        h=heldout.get(m)
        lines.append('| %s | %d/%d | %s |'%(names[m],sum(r['success'] for r in rows),len(rows),('%d/%d'%(sum(r['success'] for r in h),len(h))) if h else 'Not evaluated'))
    lines+=['','| Task ID (0-based) | Task | Unshuffled VLA demos | GMM unseen shuffles | Selected method unseen shuffles |','| --- | --- | ---: | ---: | ---: |']
    for t in tasks:
        lines.append('| %d | %s | %d | %d/5 | %d/5 |'%(t['task_id'],t['language'],t['demonstrations'],t['gmm_heldout'],t['constrained_heldout']))
    lines+=['','Success requires execution completion **and** the LIBERO task predicate after release/settling. Merely reaching the wrist waypoints is not success. Failed trials stay in the denominator.','',
            'Each task directory contains its raw VLA recordings, baseline and robust models, per-trial object layouts, shuffle maps, traces and failures. `heldout_manifest.json` records the seeds and frozen executor hash; `summary.json` is machine-readable.','',
            'Scope: these are ten distinct object tasks, all single-object pick-and-place into a basket. This does not validate push, articulation, multi-object tasks, arbitrary translations/rotations, moving obstacles, or real-world perception. Replay uses privileged simulator object poses and collision geometry. Goal containers are fixed during these shuffles. Five successful permutations per task are evidence for this tested distribution, not a universal guarantee.','']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(development=summary['development'],heldout=summary['heldout']),indent=2))

if __name__=='__main__':main()
