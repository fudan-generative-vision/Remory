"""Shared-filesystem JobBench pool, one immutable attempt per task.

Every claim holds a flock until its runner exits. A dead heartbeat never
overrides a live owner. Interrupted executions are archived and retryable;
committed completed/timeout/error results are never rerun implicitly.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .config import DEFAULT_RUNTIME_ROOT, DEFAULT_TASK_TIMEOUT_SECONDS, JobBenchConfig
from .dataset import discover_tasks
from shared_runtime.storage import atomic_json, lock, read_json, safe_name, utcnow
from .serving_recovery import TEMPORARY_SERVING_EXIT, recovered_state, serving_failure, supervisor_failure, transient_transport


TERMINAL = {"completed", "timeout", "errored"}


def initialize(root, name, config, task_ids=None):
    job = Path(root) / "jobs" / safe_name(name)
    snapshot = discover_tasks(config.dataset_root, split=config.split)
    selected = [t for t in snapshot.tasks if not task_ids or t.task_id in task_ids]
    if task_ids and set(task_ids) != {t.task_id for t in selected}:
        raise ValueError("unknown selected task id")
    tasks = []
    for task in selected:
        record = asdict(task)
        record["task_folder"] = str(task.task_folder)
        tasks.append(record)
    manifest = {"schema": "jobbench_pool_v1", "job_name": name, "config": config.to_json(),
        "contract_hash": config.contract_hash, "dataset_sha256": snapshot.snapshot_sha256, "tasks": tasks}
    with lock(job / "init.lock"):
        if (job / "pool.json").exists():
            if read_json(job / "pool.json") != manifest:
                raise RuntimeError("existing JobBench pool contract/selection differs; choose a new job name")
        else:
            # Commit the marker last: a partial initialization is safe to repeat.
            for task in tasks:
                atomic_json(job / "cells" / task["task_id"] / "state.json", {"status": "pending", "execution": 0})
            atomic_json(job / "pool.json", manifest)
    return job


def status(job):
    manifest = read_json(job / "pool.json")
    return {"job_name": manifest["job_name"], "total": len(manifest["tasks"]),
        "counts": dict(Counter(read_json(job / "cells" / t["task_id"] / "state.json")["status"] for t in manifest["tasks"]))}


def worker(job, *, worker_id, device_label, sglang_url, idle_exit_seconds=120, max_tasks=0, preflight=True,
           runner_module="job_bench_eval.runner", config_class=JobBenchConfig):
    manifest = read_json(job / "pool.json")
    config = config_class.from_json(manifest["config"])
    if preflight:
        subprocess.run([sys.executable, "-m", "job_bench_eval.runner", "--preflight"], check=True)
    if config.harness_mode == "native_compact":
        from shared_runtime.residual_preflight import assert_paired_server, get_server_info, RESIDUAL_CONCURRENCY_CAPABILITY
        try:
            assert_paired_server(config, sglang_url)
            info = get_server_info(sglang_url)
        except Exception as exc:
            if not transient_transport(exc):
                raise
            print(f"[jobbench] serving admission unavailable: {exc}; no task claimed", flush=True)
            return TEMPORARY_SERVING_EXIT
    concurrent = (RESIDUAL_CONCURRENCY_CAPABILITY in
                  (info.get("request_capabilities") or []))
    stop = False
    child = None
    def interrupt(signum, frame):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
    previous = {s: signal.signal(s, interrupt) for s in (signal.SIGINT, signal.SIGTERM)}
    idle_since = time.monotonic()
    count = 0
    last_model_notice = 0.0
    key = hashlib.sha256(f"{socket.gethostname()}:{sglang_url}".encode()).hexdigest()[:20]
    try:
        # Both paired clients share only an attested, request-isolated endpoint.
        # Legacy runners keep exclusive ownership, including across new slots.
        with contextlib.ExitStack() as ownership:
            device_lock = ownership.enter_context(lock(
                Path("/tmp") / f"jobbench-device-{key}.lock", blocking=False, shared=concurrent))
            if device_lock is None:
                raise RuntimeError("another JobBench worker already owns this local endpoint")
            slot_key = hashlib.sha256(f"{key}:{worker_id}".encode()).hexdigest()[:24]
            slot_lock = ownership.enter_context(lock(Path("/tmp") / f"jobbench-slot-{slot_key}.lock", blocking=False))
            if slot_lock is None:
                raise RuntimeError("another worker already owns this endpoint/worker-id slot")
            while not stop:
                # An existing TB/BrowseComp client may own this endpoint. Do not
                # start a task's wall-clock budget while waiting for that client.
                with lock(Path("/tmp/r2eact-terminal-agent-sglang.lock"), blocking=False, shared=concurrent) as model_slot:
                    model_busy = model_slot is None
                claimed = False
                waiting_model = False
                cooling_down = False
                for task in manifest["tasks"]:
                    if stop:
                        break
                    cell = job / "cells" / task["task_id"]
                    with lock(cell / "owner.lock", blocking=False) as owner:
                        if owner is None:
                            continue
                        state = read_json(cell / "state.json")
                        if state["status"] in TERMINAL:
                            continue
                        if (state.get("retry_after_epoch") or 0) > time.time():
                            cooling_down = True
                            continue
                        # Reconcile commit-before-state-update crashes without rerunning.
                        old_execution = cell / "executions" / f"{state['execution']:04d}"
                        if (old_execution / "result.json").exists():
                            result = read_json(old_execution / "result.json")
                            if result.get("status") in TERMINAL:
                                atomic_json(cell / "state.json", {**state, "status": result["status"], "result": str(old_execution / "result.json")})
                                continue
                            if result.get("infrastructure_recovery") == "retry_pending":
                                state.update({k: result[k] for k in ("infrastructure_failures", "retry_after_epoch", "infrastructure_recovery")})
                                if (state.get("retry_after_epoch") or 0) > time.time():
                                    atomic_json(cell / "state.json", {**state, "status": "pending"})
                                    cooling_down = True
                                    continue
                        if model_busy:
                            waiting_model = True
                            break
                        if config.harness_mode == "native_compact":
                            # A shared endpoint may have been replaced while idle.
                            # Reject identity drift before any task/attempt is claimed.
                            try:
                                assert_paired_server(config, sglang_url)
                            except Exception as exc:
                                if not transient_transport(exc):
                                    raise
                                print(f"[jobbench] SGLang unavailable before claim: {exc}", flush=True)
                                return TEMPORARY_SERVING_EXIT
                        claimed = True
                        index = state["execution"] + 1
                        execution = cell / "executions" / f"{index:04d}"
                        execution.mkdir(parents=True, exist_ok=False, mode=0o700)
                        request = {"task": task, "config": config.to_json(), "execution": str(execution), "sglang_url": sglang_url}
                        atomic_json(execution / "request.json", request)
                        state = {"infrastructure_failures": state.get("infrastructure_failures", 0),
                            "status": "running", "execution": index, "attempt": 1,
                            "worker": worker_id, "device": device_label, "host": socket.gethostname(), "started_at": utcnow(), "heartbeat": utcnow()}
                        atomic_json(cell / "state.json", state)
                        print(f"[jobbench] claimed {task['task_id']} execution={index}", flush=True)
                        with (execution / "runner.log").open("ab", buffering=0) as log:
                            child = subprocess.Popen([sys.executable, "-m", runner_module, "--request", str(execution / "request.json")],
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                pass_fds=(owner.fileno(), device_lock.fileno(), slot_lock.fileno()))
                            stop_deadline = None
                            while child.poll() is None:
                                if stop:
                                    if stop_deadline is None:
                                        stop_deadline = time.monotonic() + 45
                                    if time.monotonic() > stop_deadline:
                                        with contextlib.suppress(ProcessLookupError):
                                            os.killpg(child.pid, signal.SIGKILL)
                                state["heartbeat"] = utcnow()
                                atomic_json(cell / "state.json", state)
                                try:
                                    child.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    pass
                            runner_exit_code = child.returncode
                            child = None
                        if (execution / "result.json").is_file():
                            result = read_json(execution / "result.json")
                        else:
                            result = {"status": "interrupted" if stop else "errored", "exit_code": runner_exit_code,
                                "error": "runner exited without a result; see runner.log", "reward": None}
                            atomic_json(execution / "result.json", result)
                        supervisor_cause = supervisor_failure() if stop else None
                        failure = (supervisor_cause or serving_failure(execution)) if result["status"] in {"errored", "interrupted"} else None
                        if result["status"] == "errored" and failure is None and not stop and config.harness_mode == "native_compact":
                            try:
                                get_server_info(sglang_url, total_timeout=2, request_timeout=1, retry_backoff=0.1)
                            except Exception as exc:
                                if transient_transport(exc):
                                    failure = {"kind": "sglang_unavailable", "source": "post_error_server_probe",
                                               "message": str(exc)[:2000]}
                        if failure is not None and (not stop or supervisor_cause is not None):
                            state, result = recovered_state(state, result, failure, now=time.time())
                            # Result first, as for ordinary terminal commits. All
                            # earlier executions and local workspaces stay intact.
                            atomic_json(execution / "result.json", result)
                            state.update(result=str(execution / "result.json"), finished_at=utcnow())
                            atomic_json(cell / "state.json", state)
                            print(f"[jobbench] {task['task_id']}: {state['infrastructure_recovery']}; "
                                  "requesting serving recovery (75)", flush=True)
                            return TEMPORARY_SERVING_EXIT
                        state.update(status="pending" if result["status"] == "interrupted" else result["status"],
                            result=str(execution / "result.json"), finished_at=utcnow())
                        atomic_json(cell / "state.json", state)
                        print(f"[jobbench] {task['task_id']}: {state['status']}", flush=True)
                        if state["status"] == "errored":
                            print("[jobbench] stopping this worker after an execution error; remaining tasks stay pending", flush=True)
                            stop = True
                        count += 1
                        if max_tasks and count >= max_tasks:
                            stop = True
                    if claimed:
                        break
                if waiting_model:
                    if time.monotonic() - last_model_notice > 30:
                        print("[jobbench] local model is owned by another harness; waiting without claiming a task", flush=True)
                        last_model_notice = time.monotonic()
                    idle_since = time.monotonic()
                    time.sleep(2)
                elif claimed or cooling_down:
                    idle_since = time.monotonic()
                    if not claimed:
                        time.sleep(2)
                elif time.monotonic() - idle_since >= idle_exit_seconds:
                    break
                else:
                    time.sleep(2)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["init", "worker", "status"])
    p.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    p.add_argument("--job-name", required=True)
    p.add_argument("--mode", choices=["native_compact"], default="native_compact")
    p.add_argument("--split", choices=["main", "easy"], default="main")
    p.add_argument("--timeout-seconds", type=float, default=DEFAULT_TASK_TIMEOUT_SECONDS)
    p.add_argument("--working-memory-turns", type=int, default=12)
    p.add_argument("--serving-eviction-policy", default="checkpoint_1024_v1")
    p.add_argument("--native-compaction-text-summary", action="store_true", default=True)
    p.add_argument("--residual-compaction", action="store_true", default=True)
    p.add_argument("--residual-checkpoint", default="")
    p.add_argument("--task-id", action="append")
    p.add_argument("--worker-id", default=f"{socket.gethostname()}-{os.getpid()}")
    p.add_argument("--device-label", default=os.environ.get("CUDA_VISIBLE_DEVICES", "platform-assigned"))
    p.add_argument("--sglang-url", default=os.environ.get("TERMINAL_BENCH_POOL_SGLANG_URL", "http://127.0.0.1:30000"))
    p.add_argument("--idle-exit-seconds", type=float, default=120)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--offline-dataset", action="store_true", help="accepted for parity; worker always uses the committed offline source")
    a = p.parse_args()
    worker_code = 0
    root = a.runtime_root.resolve()
    job = root / "jobs" / safe_name(a.job_name)
    if a.action == "init":
        config = JobBenchConfig(runtime_root=str(root), dataset_root=str(root / "source/dataset"),
            split=a.split, timeout_seconds=a.timeout_seconds, harness_mode=a.mode,
            native_compaction_text_summary=a.native_compaction_text_summary,
            compressor_enabled=True,
            residual_compaction=a.residual_compaction, residual_checkpoint=a.residual_checkpoint,
            harness_name="Codex CLI / native compact + residual / xhigh",
            working_memory_turns=a.working_memory_turns, serving_eviction_policy=a.serving_eviction_policy).validate()
        initialize(root, a.job_name, config, a.task_id)
    elif a.action == "worker":
        worker_code = worker(job, worker_id=a.worker_id, device_label=a.device_label, sglang_url=a.sglang_url,
            idle_exit_seconds=a.idle_exit_seconds, max_tasks=a.max_tasks)
    print(json.dumps(status(job), ensure_ascii=False, indent=2))
    if worker_code:
        raise SystemExit(worker_code)


if __name__ == "__main__":
    main()
