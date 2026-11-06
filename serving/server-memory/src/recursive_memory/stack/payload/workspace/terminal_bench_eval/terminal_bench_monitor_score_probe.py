"""Read-only authoritative score probe for the lightweight pool monitor.

The Streamlit monitor intentionally runs in an environment without Harbor, so
it cannot import :mod:`terminal_bench_pool` directly.  This small subprocess
entry point runs under the pool's Harbor environment and exposes only the
strict, read-only reconciliation check needed by the monitor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from terminal_bench_eval import terminal_bench_pool as pool


SCORE_PROBE_SCHEMA = "terminal_bench_monitor_cell_score_probe_v1"
SCORE_AUTHORITY = (
    "terminal_bench_pool._scoreable_cell_reward+"
    "_validate_verified_agent_sigterm_reconciliation"
)
AGENT_TIMEOUT_SCORE_AUTHORITY = (
    "terminal_bench_pool._scoreable_cell_reward+_verified_reward_after_agent_timeout"
)


def inspect_reconciled_cell_score(
    *,
    job_name: str,
    jobs_root: Path,
    cell_id: str,
    expected_result_sha256: str,
) -> dict[str, Any]:
    """Return one strictly revalidated reconciled reward without writing state."""

    if re.fullmatch(r"[0-9a-f]{64}", expected_result_sha256) is None:
        raise pool.PoolError("expected result digest must be lowercase SHA-256")
    spec = pool.load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    matches = [cell for cell in spec.cells if cell.cell_id == cell_id]
    if len(matches) != 1:
        raise pool.PoolError(f"unknown or ambiguous cell id: {cell_id}")
    cell = matches[0]
    result_path = spec.cell_paths(cell).result
    before = pool._read_regular_bytes(result_path, max_bytes=pool.MAX_ARTIFACT_BYTES)
    observed_sha256 = hashlib.sha256(before).hexdigest()
    if observed_sha256 != expected_result_sha256:
        raise pool.PoolError("cell result changed before score attestation")

    result = pool._cell_result(spec, cell)
    if result is None:
        raise pool.PoolError(f"cell is not terminal: {cell_id}")
    if isinstance(result.get("agent_timeout_policy"), Mapping):
        from terminal_bench_eval.agent_timeout_policy import AUTHORITY
        authority = AUTHORITY
    elif isinstance(result.get("attribution_reconciliation"), Mapping):
        pool._validate_verified_agent_sigterm_reconciliation(spec, cell, result)
        authority = SCORE_AUTHORITY
    elif (
        isinstance(result.get("verified_agent_timeout_warning"), Mapping)
        and result["verified_agent_timeout_warning"].get("policy")
        == pool._VERIFIED_AGENT_TIMEOUT_REWARD_POLICY
        and isinstance(result.get("exception_info"), Mapping)
        and result["exception_info"].get("exception_type") == "AgentTimeoutError"
    ):
        # The pool gate below reopens the exact Harbor trial, validates phase
        # ordering/runtime evidence and compares this warning to its own proof.
        # A timeout label alone (or a missing verifier) never implies reward=0.
        authority = AGENT_TIMEOUT_SCORE_AUTHORITY
    else:
        raise pool.PoolError("cell has no attribution reconciliation or verified agent timeout")
    reward = pool._scoreable_cell_reward(spec, cell, result)
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or not math.isfinite(float(reward))
    ):
        raise pool.PoolError("authoritative pool attribution rejected the reward")

    after = pool._read_regular_bytes(result_path, max_bytes=pool.MAX_ARTIFACT_BYTES)
    if hashlib.sha256(after).hexdigest() != expected_result_sha256:
        raise pool.PoolError("cell result changed during score attestation")
    exception = result.get("exception_info")
    exception_type = (
        str(exception.get("exception_type") or "")
        if isinstance(exception, Mapping)
        else ""
    )
    return {
        "schema": SCORE_PROBE_SCHEMA,
        "authority": authority,
        "job_name": spec.paths.root.name,
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "official_attempt_index": cell.official_attempt_index,
        "result_sha256": expected_result_sha256,
        "reward": float(reward),
        "agent_timeout_policy_sha256": pool._digest(result.get("agent_timeout_policy")),
        "exception_type": exception_type,
        "verified_agent_exit_warning_sha256": pool._digest(
            result.get("verified_agent_exit_warning")
        ),
        "verified_agent_timeout_warning_sha256": pool._digest(
            result.get("verified_agent_timeout_warning")
        ),
        "attribution_reconciliation_sha256": pool._digest(
            result.get("attribution_reconciliation")
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--jobs-root", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = inspect_reconciled_cell_score(
            job_name=args.job_name,
            jobs_root=args.jobs_root,
            cell_id=args.cell_id,
            expected_result_sha256=args.expected_result_sha256,
        )
    except (OSError, ValueError, pool.PoolError) as exc:
        print(f"terminal-bench monitor score probe error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
