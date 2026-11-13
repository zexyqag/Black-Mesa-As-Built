"""The app's one heavy job at a time (evaluating a map again, working register entries out of a game, copying games in), each in
a process of its own, so the app stays quick while it runs and the pages can ask how it goes."""
import multiprocessing
import multiprocessing.connection
import os
import threading
import time

_lock = threading.Lock()
_job = {}  # the last job: task, says (what it is doing, for the pages), started, ended, result, pool


def with_the_app():
    """Where a process the app starts begins: it ends when the app's process does, however that ends (bmab.app ends at once),
    rather than go on with no one to tell."""
    parent = multiprocessing.parent_process()
    threading.Thread(target=lambda: (multiprocessing.connection.wait([parent.sentinel]), os._exit(0)), daemon=True).start()


def start(task, says, work, job):
    """Run work(job) in a process of its own, as the one job at a time; refused while another is running."""
    with _lock:
        if _job.get("pool") and not _job["result"].ready():
            raise ValueError(f"busy {_job['says']}")
        if _job.get("pool"):
            _job["pool"].join()
        pool = multiprocessing.get_context("spawn").Pool(1, with_the_app, maxtasksperchild=1)
        end = lambda _: _job.update(ended=time.time())  # noqa: E731
        result = pool.apply_async(work, [job], callback=end, error_callback=end)
        pool.close()
        _job.clear()
        _job.update(task=task, says=says, started=time.time(), result=result, pool=pool)


def running():
    """What the job running now is doing (its says), or None."""
    job = dict(_job)
    return job["says"] if job and not job["result"].ready() else None


def state(task):
    """How the job for task goes: ({state: none, running, failed or done; seconds; busy (none: what the job doing something else
    is doing); why (failed)}, its result when done."""
    job = dict(_job)
    if not job or job["task"] != task:
        return {"state": "none", "busy": running()}, None
    result = job["result"]
    seconds = round(job.get("ended", time.time()) - job["started"])
    if not result.ready():
        return {"state": "running", "seconds": seconds}, None
    try:
        rows = result.get()
    except Exception as e:  # what went wrong in the other process, for the page to say
        return {"state": "failed", "seconds": seconds, "why": f"{type(e).__name__}: {e}"}, None
    if rows is None:
        return {"state": "failed", "seconds": seconds, "why": "it needed more memory than is free"}, None
    return {"state": "done", "seconds": seconds}, rows


def forget(task):
    with _lock:
        if _job.get("task") == task and _job["result"].ready():
            _job.pop("pool").join()
            _job.clear()
