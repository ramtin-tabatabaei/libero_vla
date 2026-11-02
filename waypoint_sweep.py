"""Check option 2 on every task that has recordings, through the editor itself.

For each task: load it on the VLA tab, learn from its saved demonstrations,
then verify three things before running anything —

  * the plan is minimal (how many rows, and of which primitives),
  * every offset the editor displays resolves to the point the editor previews
    and `Move robot here` would drive to, and
  * consecutive rows sharing a reference differ in one coordinate.

Then replay it on the original layout and on shuffled ones, reporting LIBERO
task success, not merely reaching the waypoints.

    NUMBA_CACHE_DIR=/tmp/libero_numba OPENBLAS_NUM_THREADS=1 \
    xvfb-run -a python waypoint_sweep.py --output /tmp/sweep.json \
      --seeds 100 101 102 --tasks libero_object:0 libero_object:1

Omit --tasks to sweep every task with recordings. Plans are written to
SWEEP_STORE (default /tmp/gui_sweep_plans) so the real task_waypoints_gmm/
store is left alone. Running more than four of these at once exhausts memory
on a 32 GB machine; each holds two LIBERO environments.
"""

import argparse, copy, dataclasses, glob, json, os, sys, traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np


def recordings():
    tasks = defaultdict(list)
    patterns = ['waypoint_models/**/demo_*.json', 'demonstrations/**/demo_*.json',
                'task_waypoints/_demonstrations/**/rollout_*.json']
    for pattern in patterns:
        for p in glob.glob(str(Path(__file__).resolve().parent/pattern), recursive=True):
            if p.endswith('.state.json') or '/failed/' in p:
                continue
            try:
                d = json.loads(Path(p).read_text())
            except Exception:
                continue
            if (not d.get('success') or not d.get('samples')
                    or d.get('placement_mode') not in (None, 'fixed_unshuffled')):
                continue
            d['_path'] = str(Path(p).resolve())
            tasks[(d.get('suite'), d.get('task_id'))].append(d)
    for demos in tasks.values():
        demos.sort(key=lambda d: d['_path'])
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seeds', nargs='*', type=int, default=[100, 101, 102])
    ap.add_argument('--tasks', nargs='*', default=[])
    args = ap.parse_args()

    import tkinter as tk
    from unittest.mock import patch
    import subgoal_gui as gui
    import waypoint_gmm_editor as gmm
    import subgoal_planner as sp
    from waypoint_chain import resolve
    from waypoint_learning import Config

    store = Path(os.environ.get('SWEEP_STORE', '/tmp/gui_sweep_plans'))
    gmm.GMM_STORE = store
    errors = []
    patch.object(gui.messagebox, 'showerror',
                 side_effect=lambda title, msg, **kw: errors.append(str(msg))).start()
    root = tk.Tk()
    root.report_callback_exception = lambda t, v, tb: errors.append(str(v))
    app = gui.StudioApplication(root, sp.Args(task_suite_name='libero_object', task_id=0))
    root.update()
    panel, = app.panels
    panel.session.realtime = False

    tasks = recordings()
    report = []
    for key in sorted(tasks, key=lambda k: (str(k[0]), k[1])):
        suite, task_id = key
        label = f'{suite}:{task_id}'
        if args.tasks and label not in args.tasks:
            continue
        row = dict(task=label, demos=len(tasks[key]))
        errors.clear()
        try:
            for path in (store/suite/f'task_{task_id:03d}.json',):
                if path.exists():
                    path.unlink()
            panel.suite.set(suite)
            panel.task_id = task_id
            panel.load_task()
            root.update()
            row['instruction'] = panel.chain.language
            # Plan tools -> Learn from saved demonstrations, without the dialog.
            panel.replace_chain(lambda chain: panel.fit(
                tasks[key], panel.model_folder()))
            root.update()
            panel.resync_plan()
            root.update()
            chain = panel.chain
            row.update(waypoints=len(chain.waypoints),
                       skill=bool(chain.execution_skill),
                       kind=(chain.execution_skill or {}).get('kind'),
                       roles=[w.skill_role or (w.learned or {}).get('primitive')
                              for w in chain.waypoints],
                       source=chain.source)
            # What the offset box shows must be where Move robot here goes.
            gaps, axes = [], []
            poses = panel.editor_poses()
            previous = panel.session.start_pose
            for i, w in enumerate(chain.waypoints):
                target = panel.session.editor_target(chain, i, w)
                gaps.append(float(np.linalg.norm(np.asarray(resolve(w, poses, previous)[0])
                                                 - target[0])))
                previous = target
            for a, b in zip(chain.waypoints, chain.waypoints[1:]):
                if a.reference == b.reference:
                    delta = np.abs(np.asarray(b.offset)-a.offset)
                    axes.append(dict(rows=[a.name, b.name],
                                     axes=int((delta > 1e-6).sum()),
                                     mm=[round(float(v)*1000, 2) for v in delta]))
            row['offset_gap_mm'] = round(max(gaps)*1000, 6) if gaps else 0.
            row['axis_changes'] = axes
            runs = []
            for seed in [None]+list(args.seeds):
                if seed is None:
                    panel.restore_first_setup()
                else:
                    try:
                        panel.session.shuffle_scene(seed=seed)
                    except ValueError as exc:
                        runs.append(dict(seed=seed, skipped=str(exc)[:80]))
                        continue
                root.update()
                panel.resync_plan()
                status = []
                panel.session.run(panel.chain, lambda i, m: status.append((i, m)))
                runs.append(dict(seed=seed,
                                 success=bool(panel.session.env.check_success()),
                                 last=status[-1][1] if status else None))
            row['runs'] = runs
        except Exception as exc:
            row['error'] = type(exc).__name__+': '+str(exc)
            row['traceback'] = traceback.format_exc()[-900:]
        row['gui_errors'] = list(errors)
        report.append(row)
        print(json.dumps({k: v for k, v in row.items()
                          if k not in ('roles', 'traceback', 'axis_changes')}, default=str), flush=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
    root.destroy()


main()
