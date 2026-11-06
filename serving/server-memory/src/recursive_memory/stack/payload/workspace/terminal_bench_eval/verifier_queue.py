"""A CPU-only verifier recovery queue; never claims a new model attempt."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

from terminal_bench_eval.verifier_checkpoint import (
    DIRECTORY, PENDING_SCHEMA, REPLAY_AGENT, atomic_json, digest, now, pending_record,
    preparation_failure,
)
from terminal_bench_eval.verifier_bootstrap_replay import validate_completed_ctrf

COMMIT_SCHEMA = "tb_snapshot_verifier_commit_v1"


class VerifierReplayPreparationError(RuntimeError):
    """Dependency bootstrap failed; no official assertion was evaluated."""


def replay_preparation_failure(replay):
    if replay is None:
        return None
    logs = Path(replay) / "verifier"
    stdout = logs / "test-stdout.txt"
    if not stdout.is_file() or stdout.is_symlink():
        return None
    return preparation_failure(stdout.read_bytes(), report_exists=(logs / "ctrf.json").exists())


def _pool():
    from terminal_bench_eval import terminal_bench_pool
    return terminal_bench_pool


def trial_pending(spec, cell, result):
    relative = result.get("trial_path")
    if not relative: return None
    trial = spec.paths.root / relative
    trial.resolve().relative_to(spec.cell_paths(cell).harbor_job.resolve())
    value = pending_record(trial)
    if value is not None:
        task = value.get("task") or {}
        if task.get("ref") != cell.task_ref or task.get("source") != cell.source:
            raise RuntimeError("Pending verification task identity mismatch")
        return trial, value
    return None


def validate_recovery_reward(spec, cell, result):
    """Revalidate the durable no-model recovery proof whenever scoring."""
    p = _pool()
    info = result.get("verification_recovery") or {}
    receipt_path = spec.paths.root / info["receipt_path"]
    expected = spec.cell_paths(cell).attempts_dir / f"execution-{result['execution_attempt']:04d}"
    receipt_path.resolve().relative_to(expected.resolve())
    if receipt_path.is_symlink() or digest(receipt_path) != info["receipt_sha256"]:
        raise p.PoolError("Snapshot verifier receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("schema") != COMMIT_SCHEMA or receipt.get("cell_id") != cell.cell_id
            or receipt.get("task_ref") != cell.task_ref or receipt.get("model_called") is not False
            or receipt.get("execution_attempt") != result["execution_attempt"]
            or receipt.get("official_attempt_index") != cell.official_attempt_index):
        raise p.PoolError("Snapshot verifier receipt identity mismatch")
    evidence = {}
    for name, value in receipt["evidence"].items():
        source = receipt_path.parent / value["file"]
        source.resolve().relative_to(receipt_path.parent.resolve())
        if source.is_symlink() or digest(source) != value["sha256"]:
            raise p.PoolError("Snapshot verifier evidence changed")
        evidence[name] = source.read_bytes()
    replay = json.loads(evidence["replay-result"])
    original = json.loads(evidence["original-trial-result"])
    runner = json.loads(evidence["runner"])
    backup = json.loads(evidence["checkpoint"])
    config = replay["config"]
    from harbor.models.trial.config import TrialConfig
    config_matches = (TrialConfig.model_validate(json.loads(evidence["replay-config"])).model_dump(mode="json")
                      == TrialConfig.model_validate(config).model_dump(mode="json"))
    lock = json.loads(evidence["replay-lock"])
    pending = json.loads(evidence["pending"])
    original_pool = json.loads(evidence["original-result"])
    reward = replay["verifier_result"]["rewards"]["reward"]
    if (replay.get("exception_info") is not None or config["agent"].get("import_path") != REPLAY_AGENT
            or config["task"].get("ref") != cell.task_ref or config["task"].get("source") != cell.source
            or replay["task_checksum"] != original["task_checksum"]
            or runner.get("model_called") is not False
            or runner.get("archive_sha256") != backup.get("archive_sha256")
            or runner.get("task") != backup.get("task")
            or backup["task"].get("ref") != cell.task_ref or backup["task"].get("source") != cell.source
            or lock["task"].get("digest") != cell.task_ref or lock["task"].get("source") != cell.source
            or not config_matches
            or pending.get("schema") != PENDING_SCHEMA or pending.get("model_called") is not False
            or pending.get("checkpoint_sha256") != receipt["evidence"]["checkpoint"]["sha256"]
            or original_pool.get("cell_id") != cell.cell_id
            or original_pool.get("execution_attempt") != result["execution_attempt"]
            or backup.get("auto_restore") is not True
            or result.get("rewards") != {"reward": reward} or receipt.get("reward") != reward):
        raise p.PoolError("Snapshot verifier replay provenance mismatch")
    env_kwargs = config["environment"]["kwargs"]
    if env_kwargs.get("verifier_checkpoint_sha256") != receipt["evidence"]["checkpoint"]["sha256"]:
        raise p.PoolError("Replay did not use the pinned checkpoint")
    validate_completed_ctrf(evidence["ctrf"], reward)
    return float(reward)


def commit_recovery(spec, claim, original, trial, replay, pending):
    p = _pool()
    replay_result = json.loads((replay / "result.json").read_text())
    reward = p._result_rewards(replay_result).get("reward")
    if replay_result.get("exception_info") is not None or reward not in (0.0, 1.0):
        raise RuntimeError("Verifier recovery did not produce a clean binary result")
    if replay_preparation_failure(replay) is not None:
        raise VerifierReplayPreparationError("Verifier dependencies failed to download; tests never started")
    if not (replay / "verifier/ctrf.json").is_file():
        raise RuntimeError("Verifier recovery has no completed CTRF report; reward is not accepted")
    validate_completed_ctrf((replay / "verifier/ctrf.json").read_bytes(), reward)
    if replay_result["task_checksum"] != json.loads((trial / "result.json").read_text())["task_checksum"]:
        raise RuntimeError("Verifier recovery changed the frozen task")
    destination = replay / "commit"
    destination.mkdir(mode=0o700, exist_ok=True)
    sources = {"original-result": claim.paths.result, "original-trial-result": trial / "result.json",
               "pending": trial / DIRECTORY / "pending.json", "checkpoint": trial / DIRECTORY / "checkpoint.json",
               "replay-result": replay / "result.json", "replay-config": replay / "config.json",
               "replay-lock": replay / "lock.json", "runner": replay / "agent/checkpoint-replay.json",
               "ctrf": replay / "verifier/ctrf.json", "test-stdout": replay / "verifier/test-stdout.txt"}
    evidence = {}
    for name, source in sources.items():
        target = destination / (name + ".evidence")
        if target.exists() and digest(target) != digest(source):
            raise RuntimeError("Recovery evidence destination already has different contents")
        if not target.exists():
            with source.open("rb") as src, target.open("xb") as dst:
                os.chmod(target, 0o600)
                shutil.copyfileobj(src, dst); dst.flush(); os.fsync(dst.fileno())
        evidence[name] = {"file": target.name, "sha256": digest(target)}
    receipt = {"schema": COMMIT_SCHEMA, "cell_id": claim.cell.cell_id, "task_ref": claim.cell.task_ref,
               "execution_attempt": claim.execution_attempt, "official_attempt_index": claim.cell.official_attempt_index,
               "model_called": False, "reward": reward, "evidence": evidence, "committed_at": now()}
    receipt_path = destination / "receipt.json"
    atomic_json(receipt_path, receipt)
    result = {**original, "outcome": "passed" if reward else "failed", "rewards": {"reward": reward},
              "exception_info": None, "rejected_verifier_rewards": None,
              "verification_recovery": {"receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                                        "receipt_sha256": digest(receipt_path)}, "committed_at": now()}
    validate_recovery_reward(spec, claim.cell, result)
    p._commit_cell_result(spec, claim, result)
    atomic_json(trial / DIRECTORY / "pending.json", {**pending, "status": "completed", "completed_at": now(),
                                                    "reward": reward})
    return reward


async def recover_one(spec, cell, *, maximum_attempts=3):
    p = _pool(); paths = spec.cell_paths(cell)
    lock = p._open_lock(paths.claim_lock, blocking=False)
    if lock is None: return None
    try:
        from terminal_bench_eval.agent_timeout_policy import commit_existing, validate_reward
        if commit_existing(spec, cell, lock):
            return "agent_timeout_reward=0.0"
        result = p._cell_result(spec, cell)
        if result is None or result.get("verification_recovery"): return None
        if result.get("agent_timeout_policy") is not None:
            if validate_reward(spec, cell, result) != 0.0:
                raise p.PoolError("invalid agent timeout policy receipt")
            return None
        item = trial_pending(spec, cell, result)
        if item is None: return None
        trial, pending = item
        claim = p.ClaimedCell(cell=cell, paths=paths, descriptor=lock,
                              execution_attempt=result["execution_attempt"], dispatch=p._read_json(paths.dispatch))
        if result.get("outcome") != "verification_pending":
            # Mixed-version pool parents can project the new typed exception
            # as errored. Reconcile it without claiming/resampling the agent.
            atomic_json(trial / DIRECTORY / "pre-pending-pool-result.json", result)
            result = {**result, "outcome": "verification_pending", "rewards": {}}
            p._commit_cell_result(spec, claim, result)
        queue = trial / DIRECTORY / "queue.json"
        state = json.loads(queue.read_text()) if queue.exists() else {"attempts": 0}
        previous = Path(state["replay"]) if state.get("replay") else None
        finished = json.loads((previous / "result.json").read_text()) if previous is not None and (previous / "result.json").exists() else None
        if (finished is not None and finished.get("exception_info") is None
                and p._result_rewards(finished).get("reward") in (0.0, 1.0)
                and (previous / "verifier/ctrf.json").exists()):
            # Finish an interrupted commit before considering another verifier invocation.
            reward = commit_recovery(spec, claim, result, trial, previous, pending)
            atomic_json(queue, {**state, "status": "completed", "reward": reward, "updated_at": now()})
            return f"reward={reward}"
        if not pending.get("auto_restore"):
            atomic_json(queue, {**state, "status": "manual_restore_required", "updated_at": now()})
            return "manual_restore_required"
        # A network/bootstrap false-zero is not a completed verifier attempt.
        # Continue bounded, backoff-spaced recovery from the immutable snapshot
        # even after three such failures. Never retry actual test assertions.
        preparation = replay_preparation_failure(previous)
        if state["attempts"] >= maximum_attempts and preparation is None:
            return "retry_exhausted_pending"
        if state.get("retry_after", 0) > time.time(): return None
        marker = trial / DIRECTORY / "checkpoint.json"
        if digest(marker) != pending["checkpoint_sha256"]: raise RuntimeError("Checkpoint metadata changed")
        backup = json.loads(marker.read_text())
        config = deepcopy(backup["trial_config"])
        attempt_dir = paths.attempts_dir / f"execution-{result['execution_attempt']:04d}"
        replay = attempt_dir / f"snapshot-verifier-{state['attempts'] + 1:04d}"
        config.pop("job_id", None)
        config.update(trial_name=replay.name, trials_dir=str(replay.parent))
        config["agent"] = {"import_path": REPLAY_AGENT, "override_timeout_sec": 120}
        config["environment"]["delete"] = False
        config["environment"].setdefault("kwargs", {}).update(
            verifier_checkpoint_path=str(marker), verifier_checkpoint_sha256=pending["checkpoint_sha256"],
            verifier_checkpoint_task=backup["task"], keep_containers=True)
        # Keep official verifier user/env/time budget. Never enable a model.
        state.update(attempts=state["attempts"] + 1, status="running", updated_at=now(), replay=str(replay))
        atomic_json(queue, state)
        from harbor.models.trial.config import TrialConfig
        from harbor.trial.trial import Trial
        await asyncio.to_thread(p._prewarm_task_cache, dict(cell.task_config))
        instance = await Trial.create(TrialConfig.model_validate(config))
        await instance.run()
        claim = p.ClaimedCell(cell=cell, paths=paths, descriptor=lock,
                              execution_attempt=result["execution_attempt"], dispatch=p._read_json(paths.dispatch))
        reward = commit_recovery(spec, claim, result, trial, replay, pending)
        atomic_json(queue, {**state, "status": "completed", "reward": reward, "updated_at": now()})
        return f"reward={reward}"
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        if 'queue' in locals() and 'state' in locals():
            failure = replay_preparation_failure(Path(state["replay"])) if state.get("replay") else None
            atomic_json(queue, {**state, "status": "pending", "last_error": type(exc).__name__,
                               "last_error_message": str(exc)[:2000], "preparation_failure": failure,
                               "updated_at": now(),
                               "retry_after": time.time() + min(3600, 60 * 2 ** min(state.get("attempts", 0), 6))})
        return f"pending: {type(exc).__name__}: {str(exc)[:180]}"
    finally:
        p._close_lock(lock)


def requeue_legacy_bootstrap(spec, ids):
    """Explicit full rerun only when no viable original snapshot exists."""
    p = _pool(); locked = []; candidates = []
    if not ids or len(ids) != len(set(ids)): raise p.PoolError("Explicit unique cell IDs required")
    if set(ids) - {c.cell_id for c in spec.cells}: raise p.PoolError("Unknown cell")
    try:
        for cell in spec.cells:
            if cell.cell_id not in ids: continue
            paths = spec.cell_paths(cell); fd = p._open_lock(paths.claim_lock, blocking=False)
            if fd is None: raise p.PoolError("Cell currently owned")
            locked.append(fd)
            result = p._cell_result(spec, cell)
            if result is None: continue  # Already requeued or claimed: no mutation.
            candidate = p._committed_verifier_timeout_candidate(
                spec, cell, descriptor=fd, expected_execution=result["execution_attempt"], allow_bootstrap=True)
            if pending_record(candidate["trial_path"]) is not None:
                raise p.PoolError("Use verifier queue instead of resampling a checkpointed task")
            archive = paths.attempts_dir / f"execution-{result['execution_attempt']:04d}" / "bootstrap-full-rerun"
            harbor_archive = paths.attempts_dir / f"execution-{result['execution_attempt']:04d}" / "harbor-job-committed-verifier-timeout"
            if archive.exists() or harbor_archive.exists(): raise p.PoolError("Repair archive already exists")
            candidates.append(candidate)
        for candidate in candidates:
            claim = candidate["claim"]; paths = claim.paths
            archive = paths.attempts_dir / f"execution-{claim.execution_attempt:04d}" / "bootstrap-full-rerun"
            if archive.exists(): raise p.PoolError("Repair archive already exists")
            archive.mkdir(parents=True, mode=0o700)
            for name, raw in candidate["raw"].items():
                target = archive / name
                with target.open("xb") as handle:
                    os.chmod(target, 0o600); handle.write(raw); handle.flush(); os.fsync(handle.fileno())
            harbor = p._archive_verifier_timeout_harbor_job(claim, trial_path=candidate["trial_path"], failure=candidate["failure"])
            dispatch = {**candidate["dispatch"], "status": "queued", "updated_at": now(),
                        "last_verifier_bootstrap_archive": str(archive.relative_to(spec.paths.root))}
            for name in ("claim", "outcome", "harbor_job_path", "retry_not_before"): dispatch.pop(name, None)
            state = {**candidate["state"], "status": "queued", "updated_at": now()}
            for name in ("outcome", "finished_at"): state.pop(name, None)
            p._atomic_write_json(paths.dispatch, dispatch); p._atomic_write_json(paths.state, state)
            # Original result was copied durably; move the terminal marker last.
            paths.result.replace(archive / "retired-terminal-result.json")
            p._fsync_directory(paths.root); p._fsync_directory(archive)
            atomic_json(archive / "requeue.json", {"cell_id": claim.cell.cell_id, "reason": candidate["failure"],
                                                  "old_harbor_archive": str(harbor), "full_agent_rerun": True,
                                                  "official_attempt_index": claim.cell.official_attempt_index,
                                                  "requeued_at": now()})
        return [c["claim"].cell.cell_id for c in candidates]
    finally:
        for fd in reversed(locked): p._close_lock(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--jobs-root", type=Path, default=Path(__file__).resolve().parents[1] / "eval/terminal_bench/jobs")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--requeue-bootstrap", action="append", default=[])
    args = parser.parse_args()
    if args.interval < 1: parser.error("interval must be positive")
    from terminal_bench_eval.harbor_pinned_task_cache import install_pinned_package_resolution
    install_pinned_package_resolution()
    p = _pool(); spec = p.load_pool_spec(job_name=args.job_name, jobs_root=args.jobs_root)
    if args.requeue_bootstrap:
        print(json.dumps({"requeued": requeue_legacy_bootstrap(spec, args.requeue_bootstrap),
                          "state": p.aggregate_pool(spec)})); return
    lock = p._open_lock(spec.paths.root / "verifier-queue.lock", blocking=False)
    if lock is None: raise SystemExit("A verifier queue worker already owns this pool")
    print(json.dumps({"job_name": args.job_name, "mode": "verifier-only", "model_called": False,
                      "interval_seconds": args.interval}), flush=True)
    heartbeat = spec.paths.root / "verifier-queue-worker.json"
    try:
        while True:
            atomic_json(heartbeat, {"pid": os.getpid(), "host": os.uname().nodename,
                                   "status": "polling", "updated_at": now(), "model_called": False})
            for cell in spec.cells:
                status = asyncio.run(recover_one(spec, cell))
                if status: print(json.dumps({"task": cell.task_name, "verification": status}), flush=True)
            p.aggregate_pool(spec)
            if args.once: return
            time.sleep(args.interval)
    finally:
        try:
            atomic_json(heartbeat, {"pid": os.getpid(), "host": os.uname().nodename,
                                   "status": "stopped", "updated_at": now(), "model_called": False})
        finally:
            p._close_lock(lock)


if __name__ == "__main__": main()
