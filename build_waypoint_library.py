"""Record one pi0.5 rollout per task and build plans of at most nine waypoints.

Raw demonstrations are cached so replay/compression can be retried without
another policy rollout. Reports distinguish rollout success from replay success.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
STORE = ROOT / "task_waypoints"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def catalog():
    from libero.libero import benchmark
    result = []
    for name in SUITES:
        suite = benchmark.get_benchmark_dict()[name]()
        for i in range(suite.n_tasks):
            t = suite.get_task(i)
            result.append(dict(suite=name, task_id=i, language=t.language,
                               task_name=t.name, file=f"{name}/task_{i:03d}.json"))
    return result


def worker(suite, task_id, budget, record_only=False):
    import subgoal_planner as sp
    from waypoint_demo import record
    from waypoint_library import demonstration_plan
    from waypoint_session import Session

    relative = Path(suite) / f"task_{task_id:03d}.json"
    path = STORE / relative
    demo_path = STORE / "_demonstrations" / relative
    report_path = STORE / "_reports" / relative
    report = dict(suite=suite, task_id=task_id, replay_success=False, attempts=[],
                  mode="demonstration", max_waypoints=9, seed=0, validation="not_run")
    started = time.monotonic()
    session = Session(sp.Args(rrt_max_iters=1600, probe_iters=200))
    def tick():
        if time.monotonic()-started > budget:
            raise TimeoutError("Task recording/replay time budget reached")
    session.tick = tick
    try:
        cached = demo_path.exists()
        original = session.load(suite, task_id, cameras=not cached)
        if cached:
            demo = json.loads(demo_path.read_text())
        else:
            record(session, original.language, max_steps=800 if suite == "libero_10" else 500)
            demo = session.last_demonstration
            write_json(demo_path, demo)
        report.update(language=original.language, rollout_success=demo["success"],
                      demonstration=str(demo_path.relative_to(ROOT)), recorded_steps=len(demo["samples"]),
                      reused_demonstration=cached)
        print(suite, task_id, "pi0.5", len(demo["samples"]), "steps, success:", demo["success"], flush=True)
        if record_only:
            report["validation"] = "recorded"
            return report
        for name in session.env.env.observation_names:
            if name.endswith("_image"):
                session.env.env.modify_observable(name, "active", False)
        selected = demonstration_plan(demo)
        variants = [(.015, 0), (.035, 0), (.007, 0), (.015, 3), (.015, 6), (.06, 0)]
        if not demo["success"]:
            variants = variants[:2]
        best_score = -1
        for tolerance, contact_delay in variants:
            tick()
            chain = demonstration_plan(demo, position_tolerance=tolerance, contact_delay=contact_delay)
            session.env.set_init_state(demo["initial_state"])
            session.attach(settle=False)
            trace = []
            before = time.monotonic()
            try:
                outcome = session.run(chain, lambda i, result: trace.append(dict(index=i, result=result)))
                success = bool(session.env.check_success()) and outcome.startswith("Finished")
            except Exception as exc:
                outcome, success = f"{type(exc).__name__}: {exc}", False
            report["attempts"].append(dict(waypoints=len(chain.waypoints), tolerance=tolerance, contact_delay=contact_delay,
                                            result=outcome, success=success,
                                            seconds=round(time.monotonic()-before, 2), trace=trace))
            print(suite, task_id, len(chain.waypoints), "waypoints:", outcome, flush=True)
            score = sum(t["result"].startswith("ok") for t in trace)/max(len(chain.waypoints), 1)
            if score > best_score:
                selected, best_score = chain, score
            if success:
                selected = chain
                report["replay_success"] = True
                break
        report["validation"] = "replay_passed" if report["replay_success"] else "replay_failed"
        report["waypoints"] = len(selected.waypoints)
        selected.source = selected.source.replace("replay unverified", "replay " +
                             ("passed (seed 0)" if report["replay_success"] else "FAILED — review required"))
        assert 0 < len(selected.waypoints) < 10
        report["source"] = selected.source
        selected.save(path)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["validation"] = "build_failed"
        print(traceback.format_exc(), flush=True)
    finally:
        report["seconds"] = round(time.monotonic()-started, 2)
        report["checked_at"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        if session.env is not None:
            session.env.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("SUITE", "ID"))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--record-only", action="store_true")
    parser.add_argument("--cached-only", action="store_true", help="Only revisit completed tasks with cached demonstrations")
    parser.add_argument("--progress-file", default="build_progress.json")
    parser.add_argument("--suite", choices=SUITES)
    options = parser.parse_args()
    if options.worker:
        worker(options.worker[0], int(options.worker[1]), options.budget, options.record_only)
        return
    tasks = catalog()
    write_json(STORE / "catalog.json", tasks)
    if options.suite:
        tasks = [t for t in tasks if t["suite"] == options.suite]
    if options.cached_only:
        tasks = [t for t in tasks if (STORE / "_demonstrations" / t["file"]).exists()
                 and (STORE / "_reports" / t["file"]).exists()]
    if options.resume or options.only_failed:
        def needed(task):
            p = STORE / "_reports" / task["file"]
            if not p.exists():
                return True
            r = json.loads(p.read_text())
            valid = r.get("mode") == "demonstration" and 0 < r.get("waypoints", 100) < 10
            return not (valid and (r.get("replay_success") if options.only_failed else
                                  r.get("validation") in ("replay_passed", "replay_failed")))
        tasks = [t for t in tasks if needed(t)]
    if options.limit:
        tasks = tasks[:options.limit]
    backup = STORE / "_backups" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for task in tasks:
        for base, target in ((STORE, backup), (STORE / "_reports", backup / "reports")):
            path = base / task["file"]
            if path.exists():
                destination = target / task["file"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
    def run(task):
        log_path = STORE / "_logs" / task["file"].replace(".json", ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, __file__, "--worker", task["suite"], str(task["task_id"]),
               "--budget", str(options.budget)]
        if options.record_only:
            cmd.append("--record-only")
        try:
            with log_path.open("w") as log:
                completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=options.budget+30, cwd=str(ROOT))
            if completed.returncode:
                raise RuntimeError(f"Worker exited {completed.returncode}; see {log_path}")
            return json.loads((STORE / "_reports" / task["file"]).read_text())
        except Exception as exc:
            result = dict(suite=task["suite"], task_id=task["task_id"], mode="demonstration",
                          replay_success=False, validation="worker_failed", error=str(exc))
            write_json(STORE / "_reports" / task["file"], result)
            return result
    results = []
    def progress():
        write_json(STORE / options.progress_file, dict(total=len(tasks), completed=len(results),
                   passed=sum(r.get("replay_success", False) for r in results), mode="demonstration",
                   max_waypoints=9, results=results))
    progress()
    with ThreadPoolExecutor(max_workers=options.jobs) as pool:
        futures = {pool.submit(run, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(tasks)}] {result['suite']}:{result['task_id']} "
                  f"{result.get('waypoints', '?')} waypoints | {result.get('validation')}", flush=True)
            progress()


if __name__ == "__main__":
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/libero_numba")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/libero_matplotlib")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    main()
