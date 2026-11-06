"""Strict evidence gates for repairing verifier bootstrap false-zero results.

This does not requeue tasks or reinterpret real test failures. Callers must
also validate frozen task identity, ownership, and the absence of a CTRF file.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

POLICY = "verifier_missing_uvx_after_network_bootstrap_false_zero_v1"
RUNTIME_DOWNLOAD_POLICY = "verifier_runtime_download_after_network_bootstrap_false_zero_v1"
MISSING_UV_POLICY = "verifier_missing_uv_after_network_bootstrap_false_zero_v1"
NESTED_DOWNLOAD_POLICY = "verifier_cpython_nested_download_failure_false_zero_v1"
POLICIES = (POLICY, RUNTIME_DOWNLOAD_POLICY, MISSING_UV_POLICY, NESTED_DOWNLOAD_POLICY)
LOG_KEY = "verifier-bootstrap-stdout.txt"


def bootstrap_failure(result: Mapping[str, Any], trial: Mapping[str, Any],
                      log: bytes) -> dict[str, Any] | None:
    """Recognize transport failures that provably prevent the test runner starting."""
    verifier = trial.get("verifier_result")
    if (result.get("outcome") != "failed" or result.get("rewards") != {"reward": 0.0}
            or result.get("exception_info") is not None
            or trial.get("exception_info") is not None
            or not isinstance(verifier, Mapping)
            or verifier.get("rewards") != {"reward": 0.0}):
        return None
    text = log.decode("utf-8", errors="replace")
    if ("E: Failed to fetch " not in text
            or not any(s in text for s in ("502  Bad Gateway", "Connection timed out",
                                          "Connection reset", "Could not resolve"))
            or any(s in text for s in ("test session starts", "collected ",
                                       "short test summary info", "FAILED /tests/"))):
        return None
    if re.search(r"(?m)^/tests/test\.sh: line \d+: uvx: command not found$", text):
        policy = POLICY
    elif re.search(r"(?m)^/tests/test\.sh: line \d+: uv: command not found$", text):
        policy = MISSING_UV_POLICY
    elif (re.search(r"(?m)^/tests/test\.sh: line \d+: curl: command not found$", text)
          and re.search(r"(?m)^error: Failed to download https://github\.com/astral-sh/"
                        r"python-build-standalone/releases/download/[^\s]+/cpython-[^\s]+\.tar\.gz$", text)
          and "Caused by: Request failed after 3 retries" in text
          and "Caused by: error sending request for url" in text
          and any(s in text for s in ("Connection reset by peer", "operation timed out"))):
        policy = RUNTIME_DOWNLOAD_POLICY
    elif ("error: Request failed after 3 retries" in text
          and re.search(r"(?m)^\s+Caused by: Failed to download https://github\.com/astral-sh/"
                        r"python-build-standalone/releases/download/[^\s]+/cpython-[^\s]+\.tar\.gz$", text)
          and "Caused by: error sending request for url" in text
          and "Connection reset by peer" in text):
        policy = NESTED_DOWNLOAD_POLICY
    else:
        return None
    return {"policy": policy, "cause": "official_test_runner_never_started",
            "stdout_sha256": hashlib.sha256(log).hexdigest(),
            "original_reward": 0.0}


def validate_completed_ctrf(payload: bytes, reward: float) -> None:
    """A reward file alone cannot prove bootstrap or pytest actually succeeded."""
    try:
        results = json.loads(payload)["results"]
        summary = results["summary"]
        tests = results["tests"]
        n, passed, failed = (summary[k] for k in ("tests", "passed", "failed"))
        if any(type(x) is not int or x < 0 for x in (n, passed, failed)):
            raise ValueError("invalid CTRF counters")
        if (n < 1 or passed + failed != n or not isinstance(tests, list)
                or len(tests) != n
                or any(summary.get(k, 0) != 0 for k in ("skipped", "pending", "other"))):
            raise ValueError("CTRF is incomplete or has non-test outcomes")
        statuses = [t["status"] for t in tests]
        if (statuses.count("passed") != passed or statuses.count("failed") != failed
                or reward != (1.0 if failed == 0 else 0.0)):
            raise ValueError("CTRF and reward disagree")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid completed CTRF report") from exc
