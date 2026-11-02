"""Simulation workers for the compact benchmark; the parent alone writes CSV."""
from contextlib import contextmanager
import multiprocessing
import os
from queue import Empty
import signal
import sys
import time


def task_batches(specs):
    batches = {}
    for spec in specs:
        batches.setdefault((spec['suite'], spec['task_id']), []).append(spec)
    return list(batches.values())


def assign_workers(specs, workers):
    """Keep every layout and handoff condition for a task in one process."""
    batches = task_batches(specs)
    buckets = [[] for _ in range(min(workers, len(batches)))]
    for index, batch in enumerate(batches):
        buckets[index % len(buckets)].extend(batch)
    return buckets


@contextmanager
def quiet_start():
    """Workers inherit quiet streams, including library startup notices."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with open(os.devnull, 'w') as stream:
            for fd in (1, 2):
                os.dup2(stream.fileno(), fd)
            yield
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def run_worker(worker, args, specs, paths, plan_hashes, digest, completed,
               state_hashes, events, stop):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from waypoint_recovery_images import iter_entries
        def start(worker, trial):
            events.put(('start', worker, trial))
        for entry in iter_entries(args, specs, paths, plan_hashes, digest, completed,
                                  state_hashes, worker=worker, on_start=start, stop_event=stop):
            events.put(('result', worker, entry))
    except KeyboardInterrupt:
        pass
    except BaseException as exc:
        events.put(('fatal', worker, '%s: %s' % (type(exc).__name__, exc)))
    finally:
        events.put(('done', worker, None))


def parallel_entries(args, specs, paths, plan_hashes, digest, completed,
                     state_hashes, on_start=None):
    """Yield each completed result while other independent simulations continue."""
    buckets = assign_workers(specs, args.workers)
    if not buckets:
        return
    context = multiprocessing.get_context('spawn')
    events = context.Queue()
    stop = context.Event()
    processes = []
    finished = set()
    try:
        for worker, batch in enumerate(buckets, 1):
            process = context.Process(target=run_worker,
                args=(worker, args, batch, paths, plan_hashes, digest,
                      completed, state_hashes, events, stop), name='recovery-worker-%d' % worker)
            with quiet_start():
                process.start()
            processes.append(process)
        while len(finished) < len(processes):
            try:
                kind, worker, data = events.get(timeout=.25)
            except Empty:
                for worker, process in enumerate(processes, 1):
                    if worker not in finished and process.exitcode is not None:
                        raise RuntimeError('Worker %d exited unexpectedly (exit code %s)' %
                                           (worker, process.exitcode))
                continue
            if kind == 'start' and on_start is not None:
                on_start(worker, data)
            elif kind == 'result':
                yield data
            elif kind == 'fatal':
                raise RuntimeError('Worker %d failed: %s' % (worker, data))
            elif kind == 'done':
                finished.add(worker)
    finally:
        stop.set()
        deadline = time.monotonic()+5.
        for process in processes:
            process.join(timeout=max(0., deadline-time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=1.)
        events.close()
        events.join_thread()
