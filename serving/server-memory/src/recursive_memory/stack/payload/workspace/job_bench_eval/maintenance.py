"""Explicit pool maintenance; live runners keep their already-pinned budget."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
import uuid

from .config import JobBenchConfig
from shared_runtime.storage import atomic_json, lock, read_json, safe_name, utcnow

POLICY_FILE = "execution-policy.json"
POLICY_SCHEMA = "jobbench_execution_policy_v1"


def set_timeout(job: Path, timeout_seconds: float, *, reason: str | None = None) -> dict:
    """Set the next-execution budget without touching cells or frozen inputs."""
    job = job.resolve()
    with lock(job / "init.lock"):
        manifest = read_json(job / "pool.json")
        config = JobBenchConfig.from_json(manifest["config"])
        if config.contract_hash != manifest["contract_hash"]:
            raise ValueError("pool contract hash does not match its config")
        updated = replace(config, timeout_seconds=timeout_seconds).validate()
        path = job / POLICY_FILE
        if path.is_symlink():
            raise ValueError("execution policy must not be a symlink")
        previous = read_json(path) if path.exists() else None
        if previous is not None:
            if (previous.get("schema") != POLICY_SCHEMA
                    or previous.get("job_name") != manifest["job_name"]
                    or previous.get("base_contract_hash") != manifest["contract_hash"]):
                raise ValueError("execution policy does not match the pinned pool")
            if previous.get("timeout_seconds") == updated.timeout_seconds:
                return dict(job_name=manifest["job_name"], timeout_seconds=updated.timeout_seconds,
                            changed=False)
        policy = dict(schema=POLICY_SCHEMA, job_name=manifest["job_name"],
                      base_contract_hash=manifest["contract_hash"],
                      timeout_seconds=updated.timeout_seconds, updated_at=utcnow(),
                      applies_to="executions_started_after_this_commit")
        stamp = utcnow().replace(":", "").replace("+", "_") + "-" + uuid.uuid4().hex[:8]
        audit = job / "maintenance" / (stamp + "-timeout-policy")
        atomic_json(audit / "request.json", dict(action="set_timeout", policy=policy,
                                                previous_policy=previous, reason=reason))
        atomic_json(path, policy)
        report = dict(job_name=manifest["job_name"], timeout_seconds=updated.timeout_seconds,
                      changed=True, completed_at=utcnow())
        atomic_json(audit / "completed.json", report)
        return report


def prepare_memory_request(request_path: Path) -> dict:
    """Pin a fresh execution's memory independently of frozen pool/budget policy."""
    from evaluation_memory import requested_memory_profile, memory_profile_receipt, memory_profile_values

    request_path = request_path.resolve()
    request = read_json(request_path)
    if "memory_policy" in request:
        return request
    config = JobBenchConfig.from_json(request["config"])
    profile = requested_memory_profile(native_compaction=config.harness_mode == "native_compact")
    if profile is None:
        return request
    execution = Path(request["execution"]).resolve()
    if request_path.parent != execution or execution.parent.name != "executions":
        raise ValueError("request must belong to its cell execution")
    if (execution / "result.json").exists():
        raise ValueError("cannot change memory policy of an already finished execution")
    updated = replace(config, **memory_profile_values(profile)).validate()
    request.update(config=updated.to_json(), memory_policy=memory_profile_receipt(profile),
                   original_memory_config={"working_memory_turns": config.working_memory_turns,
                                           "serving_eviction_policy": config.serving_eviction_policy})
    atomic_json(request_path, request)
    print(f"[jobbench] execution memory: WM={updated.working_memory_turns}, {updated.serving_eviction_policy}", flush=True)
    return request


def prepare_request(request_path: Path) -> dict:
    """Called by each fresh runner, including runners launched by old workers."""
    request_path = request_path.resolve()
    request = read_json(request_path)
    execution = Path(request["execution"]).resolve()
    if request_path.parent != execution or execution.parent.name != "executions":
        raise ValueError("request must belong to its cell execution")
    job = execution.parents[3]
    policy_path = job / POLICY_FILE
    # A previously prepared execution remains pinned even if the policy changes.
    if "execution_policy" in request or "memory_policy" in request or not policy_path.exists():
        return request
    config = JobBenchConfig.from_json(request["config"])
    manifest = read_json(job / "pool.json")
    policy = read_json(policy_path)
    if (policy.get("schema") != POLICY_SCHEMA
            or policy.get("job_name") != manifest["job_name"]
            or policy.get("base_contract_hash") != manifest["contract_hash"]
            or config.contract_hash != manifest["contract_hash"]):
        raise ValueError("execution policy does not match the pinned pool")
    updated = replace(config, timeout_seconds=policy["timeout_seconds"]).validate()
    request.update(config=updated.to_json(), execution_policy=policy,
                   original_timeout_seconds=config.timeout_seconds)
    atomic_json(request_path, request)
    print(f"[jobbench] per-task budget: {updated.timeout_seconds:g}s "
          f"(pool default {config.timeout_seconds:g}s; execution policy)", flush=True)
    return request


def requeue(job: Path, task_ids: list[str], *, timeout_seconds: float | None = None,
            include_completed: bool = False, reason: str | None = None,
            config_class=JobBenchConfig) -> dict:
    """Archive explicit terminal executions; completed tasks require opt-in.

    Omitting a timeout preserves the current execution policy byte for byte.
    This supports checkpoint replacement without changing worker launchers or
    discarding recoverable prior outputs and judgments.
    """
    job = job.resolve()
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("provide distinct task IDs")
    with ExitStack() as stack:
        stack.enter_context(lock(job / "init.lock"))
        manifest = read_json(job / "pool.json")
        config = config_class.from_json(manifest["config"])
        updated = (replace(config, timeout_seconds=timeout_seconds).validate()
                   if timeout_seconds is not None else None)
        old_policy = read_json(job / POLICY_FILE) if (job / POLICY_FILE).exists() else None
        allowed_statuses = {"timeout", "errored"}
        if include_completed:
            allowed_statuses.add("completed")
        allowed = {t["task_id"] for t in manifest["tasks"]}
        targets = []
        for task_id in sorted(task_ids):
            if safe_name(task_id) not in allowed:
                raise ValueError(f"unknown task: {task_id}")
            cell = job / "cells" / task_id
            owner = stack.enter_context(lock(cell / "owner.lock", blocking=False))
            if owner is None:
                raise RuntimeError(f"task currently owned by a worker/judge: {task_id}")
            state = read_json(cell / "state.json")
            if state.get("status") not in allowed_statuses:
                raise ValueError(f"refusing to requeue {task_id}: {state.get('status')}")
            execution = cell / "executions" / f"{state['execution']:04d}"
            if execution.is_symlink() or not execution.is_dir():
                raise ValueError(f"missing or unsafe execution: {task_id}")
            result = read_json(execution / "result.json")
            if result.get("status") != state["status"]:
                raise ValueError(f"state/result disagreement: {task_id}")
            targets.append((task_id, cell, state, execution))

        stamp = utcnow().replace(":", "").replace("+", "_") + "-" + uuid.uuid4().hex[:8]
        audit = job / "maintenance" / stamp
        policy = (dict(schema=POLICY_SCHEMA, job_name=manifest["job_name"],
                       base_contract_hash=manifest["contract_hash"],
                       timeout_seconds=updated.timeout_seconds, updated_at=utcnow(),
                       applies_to="executions_started_after_this_commit")
                  if updated is not None else old_policy)
        atomic_json(audit / "request.json", dict(task_ids=task_ids, policy=policy,
                                                previous_policy=old_policy,
                                                include_completed=include_completed,
                                                reason=reason))
        if updated is not None:
            atomic_json(job / POLICY_FILE, policy)
        archived = []
        for task_id, cell, state, execution in targets:
            archive = audit / task_id
            atomic_json(archive / "state.json", state)
            destination = archive / execution.name
            execution.rename(destination)
            # Retain the execution counter. The original result must be moved:
            # old workers otherwise reconcile it back to terminal before claiming.
            atomic_json(cell / "state.json", dict(status="pending", execution=state["execution"],
                requeued_at=utcnow(), previous_execution_archive=str(destination)))
            archived.append(dict(task_id=task_id, archive=str(destination)))
        effective_timeout = policy["timeout_seconds"] if policy else config.timeout_seconds
        report = dict(requeued=archived, timeout_seconds=effective_timeout,
                      completed_at=utcnow())
        atomic_json(audit / "completed.json", report)
        return report


def main():
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--task-id", action="append")
    p.add_argument("--set-timeout-only", action="store_true",
                   help="update the next-execution budget without requeuing any task")
    p.add_argument("--timeout-seconds", type=float, help="omit to preserve the current budget policy")
    p.add_argument("--include-completed", action="store_true",
                   help="explicitly permit requeuing completed samples (e.g. checkpoint replacement)")
    p.add_argument("--reason", help="maintenance audit explanation")
    args = p.parse_args()
    if args.set_timeout_only:
        if args.task_id or args.include_completed or args.timeout_seconds is None:
            p.error("--set-timeout-only requires --timeout-seconds and no task selection")
        print(json.dumps(set_timeout(args.job, args.timeout_seconds, reason=args.reason), indent=2))
        return
    if not args.task_id:
        p.error("requeue requires at least one --task-id")
    print(json.dumps(requeue(args.job, args.task_id, timeout_seconds=args.timeout_seconds,
                             include_completed=args.include_completed, reason=args.reason), indent=2))


if __name__ == "__main__":
    main()
