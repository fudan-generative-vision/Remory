"""AutomationBench shared-filesystem queue. Flocks own cells; leases do not."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from shared_runtime.storage import atomic_json, lock, read_json, safe_name, utcnow
from . import runner
from .runtime import official_python, interpreter_environment
from .tool_runtime import JOBBENCH_TOOLS, prepare_tools

DEFAULT_JOB = "automationbench-native-compact-residual-c2320-xhigh-v1"
DEFAULT_SHUFFLE_SEED = 20260910
DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")
TERMINAL = {"completed", "step_limit", "timeout", "errored"}
RETRYABLE = {"interrupted", "not_started"}


def catalog(root):
    raw = subprocess.check_output([str(official_python(root)),
        str(Path(__file__).with_name("official_session.py")), "--catalog"],
        env=interpreter_environment(), cwd="/tmp", timeout=240, text=True)
    tasks = json.loads(raw)
    if Counter(t["domain"] for t in tasks) != Counter({d: 100 for d in DOMAINS}):
        raise RuntimeError("Pinned public dataset must contain 100 tasks in each of six domains")
    if len({safe_name(t["task_id"]) for t in tasks}) != 600:
        raise RuntimeError("Public catalog must contain 600 unique task IDs")
    return tasks


def default_config(checkpoint=runner.DEFAULT_RESIDUAL):
    return {"mode": "native_compact", "compressor": "residual", "checkpoint": checkpoint,
        "working_memory_turns": None, "reasoning_effort": "xhigh",
        "serving_eviction_policy": "checkpoint_1024_v1",
        "max_steps": 0, "timeout_seconds": 3600.0, "official_commit": runner.REVISION,
        "attempts_per_task": 1, "toolset": "api", "tool_profile": JOBBENCH_TOOLS}


def shuffled_tasks(tasks, seed):
    """Seeded pseudorandom permutation, independent of Python/catalog ordering.

    The manifest persists the resulting order. Repeated worker launches produce
    the same contract rather than shuffling the remaining queue on every node.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("shuffle seed must be an unsigned 64-bit integer")
    return sorted(tasks, key=lambda t: (
        hashlib.sha256(f"{seed}\0{t['task_id']}".encode("utf-8")).digest(), t["task_id"]))


def load_pool(job):
    manifest = read_json(job / "pool.json")
    if manifest.get("schema") != "automationbench_pool_v1":
        raise RuntimeError("Unsupported AutomationBench pool schema")
    return manifest


def execution_path(cell, state):
    number = state.get("execution", 0)
    return cell / "executions" / f"{number:04d}" if type(number) is int and number > 0 else None




def initialize(root, job_name, config, tasks, *, pilot=None):
    from .reasoning_policy import validate_profile
    validate_profile(config)
    if pilot is not None:
        raise ValueError("Importing scores from other harness profiles is not supported")
    job = root / "jobs" / safe_name(job_name)
    manifest = {"schema": "automationbench_pool_v1", "config": config, "tasks": tasks}
    with lock(job / "init.lock"):
        if (job / "pool.json").exists():
            if load_pool(job) != manifest:
                raise RuntimeError("Pool contract changed; use a new --job-name. Existing results preserved.")
            return job
        for task in tasks:
            cell = job / "cells" / safe_name(task["task_id"])
            if not (cell / "state.json").exists():
                atomic_json(cell / "state.json", {"task_id": task["task_id"], "domain": task["domain"],
                    "attempt": 1, "execution": 0, "status": "pending"})
        # pool.json is the commit marker. A worker cannot see a partial pool.
        atomic_json(job / "config.json", {**config, "task_ids": [t["task_id"] for t in tasks]})
        atomic_json(job / "pool.json", manifest)
    return job


def status(job):
    manifest = load_pool(job)
    counts = Counter(read_json(job / "cells" / t["task_id"] / "state.json")["status"] for t in manifest["tasks"])
    return {"job_name": job.name, "total": len(manifest["tasks"]), "counts": dict(counts)}


def reconcile(cell, state):
    """Only call while holding cell ownership; no heartbeat-based stealing."""
    path = execution_path(cell, state)
    if state["status"] != "running":
        return state
    result = read_json(path / "result.json") if path and (path / "result.json").exists() else None
    if result and result.get("status") in TERMINAL:
        state.update(status=result["status"], finished_at=result.get("finished_at"))
    else:
        if path and not result:
            atomic_json(path / "result.json", {"task_id": state["task_id"], "status": "interrupted",
                "reward": None, "included_in_benchmark_score": False,
                "error": "Previous worker released ownership without a terminal result", "finished_at": utcnow()})
        state.update(status="pending", last_interruption_at=utcnow())
    atomic_json(cell / "state.json", state)
    return state


def model_busy(*, shared=False):
    # Same machine-wide inference lease as JobBench/TB. Never hold this probe
    # across backend startup: the backend acquires the lease itself.
    from terminal_bench_eval.codex_r2eact_gateway import DEFAULT_PROCESS_LOCK
    with lock(DEFAULT_PROCESS_LOCK, blocking=False, shared=shared) as owner:
        return owner is None


def run_request(path):
    request = read_json(path)
    task, config = request["task"], request["config"]
    execution = path.parent
    # Each task starts in a fresh Python process, including tasks launched by
    # workers that cached the earlier pool config. Refresh only the mutable
    # step budget at this boundary; never change an in-flight task's metadata.
    if execution.parent.name == "executions" and execution.parents[2].name == "cells":
        job = execution.parents[3]
        with lock(job / "init.lock"):
            manifest = load_pool(job)
            if task not in manifest["tasks"]:
                raise RuntimeError("Request task is not in its owning pool")
            max_steps = manifest["config"]["max_steps"]
            if type(max_steps) is not int or max_steps < 0:
                raise ValueError("Pool max_steps must be a nonnegative integer")
            if config.get("max_steps") != max_steps:
                request["worker_max_steps"] = config.get("max_steps")
                config = {**config, "max_steps": max_steps}
                request["config"] = config
                atomic_json(path, request)
    args = SimpleNamespace(**{**config, **task, "runtime_root": Path(request["runtime_root"]),
                              "sglang_url": request["sglang_url"],
                              "concurrency": request.get("concurrency", 1)})
    def interrupted(signum, frame):
        raise InterruptedError(f"Interrupted before/during task initialization: signal {signum}")
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        return runner.run_task(args, task["task_id"], execution)
    except (Exception, KeyboardInterrupt) as exc:
        state = ("not_started" if isinstance(exc, runner.EndpointBusyError) else "interrupted"
                 if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "errored")
        result = {"task_id": task["task_id"], "domain": task["domain"], "status": state,
            "reward": None, "included_in_benchmark_score": False, "finished_at": utcnow(),
            "error": f"{type(exc).__name__}: {exc}"}
        atomic_json(execution / "result.json", result)
        return result
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def worker(job, *, worker_id, device_label, sglang_url, idle_exit_seconds=120, max_tasks=None,
           concurrency=1):
    manifest = load_pool(job)
    root = job.parents[1]
    # Preserve the original pool/finished results, but stamp the effective tool
    # ABI in every fresh execution request. In-flight old workers remain legacy
    # until restarted; their request files and metadata tell the truth.
    config = {**manifest["config"], "tool_profile": JOBBENCH_TOOLS}
    from .reasoning_policy import validate_profile
    validate_profile(config)
    from shared_runtime.discrete_concurrency import summary_concurrency
    summary_concurrency(SimpleNamespace(**config, concurrency=concurrency))
    # Even one worker may coexist with other benchmark pools. Both supported
    # profiles use the same attested, request-isolated paired backend.
    shared = True
    runner.check_source(root)
    prepare_tools(JOBBENCH_TOOLS)
    print(f"[automationbench] tools={JOBBENCH_TOOLS}; tool runtime preflight passed", flush=True)
    stop = threading.Event()
    child = None
    def interrupted(signum, frame):
        stop.set()
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    identity = hashlib.sha256(f"{socket.gethostname()}:{sglang_url}".encode()).hexdigest()[:20]
    done, idle_since = 0, time.monotonic()
    try:
        with contextlib.ExitStack() as leases:
            device = leases.enter_context(lock(Path(f"/tmp/automationbench-device-{identity}.lock"),
                blocking=False, shared=shared))
            if device is None:
                raise RuntimeError("An AutomationBench worker already owns this node's endpoint")
            lease_fds = [device.fileno()]
            if shared:
                slot_id = hashlib.sha256(worker_id.encode()).hexdigest()[:20]
                slot = leases.enter_context(lock(Path(f"/tmp/automationbench-slot-{identity}-{slot_id}.lock"), blocking=False))
                if slot is None:
                    raise RuntimeError(f"An AutomationBench worker already owns slot {worker_id}")
                lease_fds.append(slot.fileno())
            while not stop.is_set():
                busy = model_busy(shared=True) if shared else model_busy()
                if busy:
                    idle_since = time.monotonic()
                    stop.wait(5)
                    continue
                claimed = False
                for task in manifest["tasks"]:
                    if stop.is_set():
                        break
                    cell = job / "cells" / task["task_id"]
                    # Fast path only. Every decision is rechecked under lock.
                    if read_json(cell / "state.json")["status"] in TERMINAL:
                        continue
                    with lock(cell / "owner.lock", blocking=False) as owner:
                        if owner is None:
                            continue
                        state = reconcile(cell, read_json(cell / "state.json"))
                        if state["status"] in TERMINAL:
                            continue
                        try:
                            runner.ensure_endpoint_available(SimpleNamespace(**config,
                                sglang_url=sglang_url, concurrency=concurrency))
                        except runner.EndpointBusyError:
                            idle_since = time.monotonic()
                            break
                        if stop.is_set():
                            break
                        number = state.get("execution", 0) + 1
                        execution = cell / "executions" / f"{number:04d}"
                        # Never overwrite orphan artifacts after a crash before
                        # state publication; execution numbers are not attempts.
                        while execution.exists():
                            number += 1
                            execution = cell / "executions" / f"{number:04d}"
                        execution.mkdir(parents=True, mode=0o700)
                        request = execution / "request.json"
                        atomic_json(request, {"config": config, "task": task, "runtime_root": str(root),
                                              "sglang_url": sglang_url, "concurrency": concurrency})
                        state = {"task_id": task["task_id"], "domain": task["domain"], "attempt": 1,
                            "execution": number, "status": "running", "worker_id": worker_id,
                            "device_label": device_label, "hostname": socket.gethostname(),
                            "started_at": utcnow(), "heartbeat_at": utcnow()}
                        atomic_json(cell / "state.json", state)
                        print(f"[automationbench] claimed {task['task_id']} execution={number}", flush=True)
                        claimed = True
                        try:
                            with (execution / "runner.log").open("w") as log:
                                child = subprocess.Popen([sys.executable, "-m", "automation_bench_eval.pool",
                                    "--request", str(request)], cwd=runner.PROJECT, stdout=log, stderr=log,
                                    start_new_session=True, pass_fds=(owner.fileno(), *lease_fds))
                                deadline = None
                                while child.poll() is None:
                                    if stop.is_set():
                                        if deadline is None:
                                            interrupted(signal.SIGTERM, None)
                                            deadline = time.monotonic() + 45
                                        elif time.monotonic() >= deadline:
                                            with contextlib.suppress(ProcessLookupError):
                                                os.killpg(child.pid, signal.SIGKILL)
                                    state["heartbeat_at"] = utcnow()
                                    atomic_json(cell / "state.json", state)
                                    try:
                                        child.wait(timeout=2 if stop.is_set() else 5)
                                    except subprocess.TimeoutExpired:
                                        pass
                        finally:
                            # Do not release either ownership lock over a live child.
                            if child is not None and child.poll() is None:
                                with contextlib.suppress(ProcessLookupError):
                                    os.killpg(child.pid, signal.SIGTERM)
                                try:
                                    child.wait(timeout=45)
                                except subprocess.TimeoutExpired:
                                    with contextlib.suppress(ProcessLookupError):
                                        os.killpg(child.pid, signal.SIGKILL)
                                    child.wait()
                            code = child.returncode if child else None
                            child = None
                        if not (execution / "result.json").exists():
                            atomic_json(execution / "result.json", {"task_id": task["task_id"],
                                "status": "interrupted" if stop.is_set() else "errored", "reward": None,
                                "included_in_benchmark_score": False, "finished_at": utcnow(),
                                "error": f"Runner exited without a result (exit={code}); see runner.log"})
                        result = read_json(execution / "result.json")
                        state.update(status="pending" if result["status"] in RETRYABLE else result["status"],
                                     finished_at=result.get("finished_at"), heartbeat_at=utcnow())
                        atomic_json(cell / "state.json", state)
                        print(f"[automationbench] {task['task_id']}: {state['status']}", flush=True)
                        if result["status"] not in RETRYABLE:
                            done += 1
                        if result["status"] == "errored":
                            print(f"[automationbench] execution error: {result.get('error')}; remaining tasks stay pending", flush=True)
                            stop.set()
                        if max_tasks is not None and done >= max_tasks:
                            stop.set()
                        idle_since = time.monotonic()
                        break
                if not claimed and not stop.is_set():
                    if deferred:
                        # Do not idle-exit while waiting for remote xhigh results.
                        # Other unrelated pending tasks above remain runnable.
                        idle_since = time.monotonic()
                    if time.monotonic() - idle_since >= idle_exit_seconds:
                        break
                    stop.wait(min(5, idle_exit_seconds))
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    return status(job)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", nargs="?", choices=["worker", "status"])
    p.add_argument("--request", type=Path)
    p.add_argument("--runtime-root", type=Path, default=runner.DEFAULT_RUNTIME)
    p.add_argument("--job-name", default=DEFAULT_JOB)
    p.add_argument("--worker-id", default=socket.gethostname())
    p.add_argument("--device-label", default="platform-assigned")
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument("--idle-exit-seconds", type=float, default=120)
    p.add_argument("--max-tasks", type=int)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--offline-dataset", action="store_true")
    a = p.parse_args()
    if a.request:
        print(json.dumps(run_request(a.request), ensure_ascii=False), flush=True)
        return
    if not a.action or not 0 < a.idle_exit_seconds < float("inf") or (a.max_tasks is not None and a.max_tasks < 1):
        p.error("action and positive finite idle/max-tasks budgets required")
    job = a.runtime_root.resolve() / "jobs" / safe_name(a.job_name)
    output = status(job) if a.action == "status" else worker(job, worker_id=a.worker_id,
        device_label=a.device_label, sglang_url=a.sglang_url, concurrency=a.concurrency,
        idle_exit_seconds=a.idle_exit_seconds, max_tasks=a.max_tasks)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
