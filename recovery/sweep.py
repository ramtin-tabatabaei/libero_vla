"""Fine-tune the shared reach policy on every task, one subprocess at a time.

The multi-task policy from `train.py` is the thing to use; this is for the tail
of scenes it does not solve. Each task gets a short run warm-started from the
shared checkpoint, and only the ones that actually improve are kept, so the
sweep can be re-run against a better shared policy without losing ground.

Runs are sequential by default, not parallel. A single training run already
holds `--ppo.num-envs` worker processes at about a gigabyte each, and this box
has eight physical cores behind its sixteen threads -- two concurrent runs
oversubscribe it and both go slower than one would have.

    # every task the shared policy has not solved
    /home/ramtin/miniconda3/envs/libero/bin/python -m recovery.sweep \
      --shared recovery/runs/all_tasks/checkpoint.pt --resume

    # one suite, and re-do the ones that failed last time
    ... -m recovery.sweep --suite libero_90 --only-failed
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from build_waypoint_library import write_json
from recovery.reach_env import SUITES

ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPORTS = RUNS / "_reports"


def catalog(suite=None):
    """Every (suite, task_id) in benchmark order, as plain dicts."""
    from libero.libero import benchmark

    tasks = []
    for name in ([suite] if suite else SUITES):
        bench = benchmark.get_benchmark_dict()[name]()
        for i in range(bench.n_tasks):
            tasks.append(dict(suite=name, task_id=i,
                              language=bench.get_task(i).language,
                              file="%s/task_%03d.json" % (name, i)))
    return tasks


def report_for(task):
    return REPORTS / task["file"]


def needed(task, only_failed, threshold):
    """Whether this task still wants a run."""
    path = report_for(task)
    if not path.exists():
        return True
    report = json.loads(path.read_text())
    if only_failed:
        return not report.get("ok", False)
    return report.get("success_rate", 0.0) < threshold


def run_one(task, options):
    """Train one task in its own process and return its report."""
    name = "%s_task%03d" % (task["suite"], task["task_id"])
    log_path = RUNS / "_logs" / (task["file"].replace(".json", ".log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", "recovery.train",
               "--suite", task["suite"], "--task-id", str(task["task_id"]),
               "--run-name", name,
               "--ppo.total-timesteps", str(options.timesteps),
               "--ppo.num-envs", str(options.num_envs),
               "--no-snapshot"]
    if options.shared:
        command += ["--init-from", options.shared]

    started = time.time()
    try:
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=options.budget, cwd=str(ROOT.parent))
        if completed.returncode:
            raise RuntimeError("trainer exited %d; see %s" % (completed.returncode, log_path))
        history = json.loads((RUNS / name / "history.json").read_text())
        finished = [h for h in history if h["episodes"]]
        recent = finished[-5:] if finished else []
        rate = sum(h["success_rate"] for h in recent) / len(recent) if recent else 0.0
        report = dict(suite=task["suite"], task_id=task["task_id"], ok=True,
                      success_rate=rate,
                      mean_distance=recent[-1]["mean_distance"] if recent else float("nan"),
                      steps=finished[-1]["global_step"] if finished else 0,
                      seconds=round(time.time() - started, 1),
                      checkpoint=str(RUNS / name / "checkpoint.pt"))
    except Exception as exc:
        report = dict(suite=task["suite"], task_id=task["task_id"], ok=False,
                      success_rate=0.0, error=str(exc),
                      seconds=round(time.time() - started, 1))
    write_json(report_for(task), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--shared", help="Checkpoint to warm-start each task from")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--budget", type=int, default=3600, help="Seconds per task")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Success rate at or above which a task is considered done")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--progress-file", default="sweep_progress.json")
    options = parser.parse_args()

    tasks = catalog(options.suite)
    write_json(RUNS / "catalog.json", tasks)
    if options.resume or options.only_failed:
        tasks = [t for t in tasks if needed(t, options.only_failed, options.threshold)]
    if options.limit:
        tasks = tasks[:options.limit]

    print("%d tasks to run" % len(tasks))
    results = []
    for task in tasks:
        report = run_one(task, options)
        results.append(report)
        print("[%d/%d] %s:%d  success %5.1f%%  %.0fs%s"
              % (len(results), len(tasks), report["suite"], report["task_id"],
                 100 * report["success_rate"], report["seconds"],
                 "" if report["ok"] else "  FAILED: " + report.get("error", "")[:60]),
              flush=True)
        write_json(RUNS / options.progress_file,
                   dict(total=len(tasks), completed=len(results),
                        solved=sum(r["success_rate"] >= options.threshold for r in results),
                        results=results))

    if results:
        solved = sum(r["success_rate"] >= options.threshold for r in results)
        print("\n%d/%d tasks at or above %.0f%% success"
              % (solved, len(results), 100 * options.threshold))


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/libero_numba")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
