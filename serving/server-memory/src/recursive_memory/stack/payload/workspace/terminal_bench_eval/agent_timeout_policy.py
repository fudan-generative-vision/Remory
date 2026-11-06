"""Explicit per-pool policy: a typed Harbor agent timeout ends the attempt at zero.

The operator policy is separate from verifier output. Original Harbor artifacts
are immutable; aggregate and monitor gates revalidate the timeout evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

POLICY = "agent_timeout_reward_zero_no_retry_v1"
FILENAME = "agent-timeout-policy.json"
AUTHORITY = "terminal_bench_pool._scoreable_cell_reward+agent_timeout_policy.validate_reward"


def _pool():
    from terminal_bench_eval import terminal_bench_pool
    return terminal_bench_pool


def load_policy(spec):
    p = _pool()
    path = spec.paths.root / FILENAME
    if not path.exists():
        return None
    raw = p._read_regular_bytes(path, max_bytes=64 * 1024)
    value = json.loads(raw)
    if (value.get("schema") != "terminal_bench_agent_timeout_policy_v1"
            or value.get("policy") != POLICY or value.get("job_name") != spec.paths.root.name
            or value.get("config_contract_hash") != spec.contract_hash
            or value.get("reward") != 0 or isinstance(value.get("reward"), bool)
            or value.get("retry_agent") is not False):
        raise p.PoolError("invalid agent timeout scoring policy")
    return {**value, "sha256": hashlib.sha256(raw).hexdigest()}


def applies(spec, trial):
    exception = trial.get("exception_info")
    return (isinstance(exception, Mapping) and exception.get("exception_type") == "AgentTimeoutError"
            and load_policy(spec) is not None)


def proof(spec, claim, trial_path, trial):
    p = _pool()
    policy = load_policy(spec)
    if policy is None or not applies(spec, trial) or not trial.get("finished_at"):
        raise p.PoolError("no completed typed agent timeout for this policy")
    if Path(trial_path).parent != claim.paths.harbor_job or Path(trial_path).is_symlink():
        raise p.PoolError("timeout trial is outside its cell")
    p._validate_harbor_trial(spec, claim, trial)
    raw = p._read_regular_bytes(Path(trial_path) / "result.json", max_bytes=p.MAX_ARTIFACT_BYTES)
    saved = json.loads(raw)
    if saved.get("exception_info") != trial.get("exception_info") or not saved.get("finished_at"):
        raise p.PoolError("timeout evidence differs from the recorded trial")
    return {"policy": POLICY, "policy_sha256": policy["sha256"],
            "cell_id": claim.cell.cell_id, "official_attempt_index": claim.cell.official_attempt_index,
            "execution_attempt": claim.execution_attempt,
            "trial_path": str(Path(trial_path).relative_to(spec.paths.root)),
            "trial_result_sha256": hashlib.sha256(raw).hexdigest(),
            "exception_sha256": p._digest(trial["exception_info"]), "reward": 0.0, "retry_agent": False}


def project(spec, claim, trial_path, trial, result):
    if not applies(spec, trial):
        return result
    return {**result, "status": "completed", "outcome": "failed", "rewards": {"reward": 0.0},
            "rejected_verifier_rewards": None, "agent_timeout_policy": proof(spec, claim, trial_path, trial),
            "timeout_verifier_rewards": _pool()._result_rewards(trial)}


def validate_reward(spec, cell, result):
    p = _pool()
    try:
        if (result.get("status") != "completed" or result.get("outcome") != "failed"
                or result.get("rewards") != {"reward": 0.0} or result.get("rejected_verifier_rewards")):
            return None
        claim = p.ClaimedCell(cell, spec.cell_paths(cell), -1,
                             p._valid_execution(result.get("execution_attempt")), {})
        terminal = p._terminal_harbor_trial(spec, claim)
        if terminal is None:
            return None
        trial_path, trial = terminal
        if (result.get("trial_path") != str(trial_path.relative_to(spec.paths.root))
                or result.get("trial_name") != trial.get("trial_name")
                or p._digest(result.get("exception_info")) != p._digest(trial.get("exception_info"))):
            return None
        expected = proof(spec, claim, trial_path, trial)
        return 0.0 if p._digest(result.get("agent_timeout_policy")) == p._digest(expected) else None
    except (OSError, ValueError, KeyError, TypeError, p.PoolError):
        return None


def commit_existing(spec, cell, descriptor, *, archive=True):
    """Under the cell lock, finish a legacy timeout without another execution."""
    p = _pool()
    if load_policy(spec) is None:
        return False
    paths = spec.cell_paths(cell)
    dispatch = p._read_json(paths.dispatch) if paths.dispatch.exists() else {}
    result = p._cell_result(spec, cell)
    if result and not applies(spec, result):
        return False
    if result and validate_reward(spec, cell, result) == 0.0:
        return False
    execution = p._valid_execution((result or dispatch).get("execution_attempt"))
    if execution < 1:
        return False
    claim = p.ClaimedCell(cell, paths, descriptor, execution, dispatch)
    terminal = p._terminal_harbor_trial(spec, claim)
    if terminal is None or not applies(spec, terminal[1]):
        return False
    trial_path, trial = terminal
    updated = p._cell_result_from_trial(spec, claim, trial_path, trial,
        worker_id=(result or {}).get("worker_id", "agent-timeout-policy"),
        device_label=(result or {}).get("device", "existing-execution"))
    if result:
        # Preserve original provenance and timestamps, especially inherited results.
        updated = project(spec, claim, trial_path, trial, result)
    if archive:
        destination = paths.attempts_dir / f"execution-{execution:04d}" / POLICY
        p._ensure_dir(destination)
        for path in (paths.result, paths.state, paths.dispatch):
            if path.exists() and not (destination / path.name).exists():
                p._atomic_write_bytes(destination / path.name, path.read_bytes())
        p._atomic_write_json(destination / "policy-receipt.json", updated["agent_timeout_policy"])
    p._commit_cell_result(spec, claim, updated)
    p._best_effort_event(spec.paths.root, {"event": "agent_timeout_scored_zero", "cell_id": cell.cell_id,
                                           "execution_attempt": execution, "policy": POLICY,
                                           "recorded_at": p._utc_now()})
    return True
