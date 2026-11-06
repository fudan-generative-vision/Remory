"""Explicit execution-time memory profiles, separate from historical pool identity."""
from __future__ import annotations

import os
from typing import Any, Mapping


MEMORY_PROFILE_ENV = "R2EACT_EVAL_MEMORY_PROFILE"
WM12_ACC8192 = "wm12_acc8192"
WORKING_MEMORY_TURNS = 12
SERVING_EVICTION_POLICY = "qwen38_serving_8192_accumulation_canary_v1"
ACCUMULATION_TOKENS = 8192
CODEX_AGENT = "terminal_bench_eval.codex_r2eact_agent:CodexR2EActAgent"


def requested_memory_profile(*, native_compaction: bool = False) -> str | None:
    # Native compact, including summary-only, must ignore an inherited r2eAct pin.
    if native_compaction:
        return None
    profile = os.environ.get(MEMORY_PROFILE_ENV) or None
    if profile is not None:
        memory_profile_values(profile)
    return profile


def memory_profile_values(profile: str) -> dict[str, Any]:
    if profile != WM12_ACC8192:
        raise ValueError(f"unsupported evaluation memory profile: {profile!r}")
    return {"working_memory_turns": WORKING_MEMORY_TURNS,
            "serving_eviction_policy": SERVING_EVICTION_POLICY}


def codex_r2eact_agent(agent: Mapping[str, Any]) -> bool:
    return (agent.get("import_path") == CODEX_AGENT
            and not agent.get("kwargs", {}).get("native_compaction", False))


def memory_profile_receipt(profile: str) -> dict[str, Any]:
    return {"profile": profile, **memory_profile_values(profile),
            "accumulation_target_tokens": ACCUMULATION_TOKENS,
            "applies_to": "this_execution_only; frozen_pool_and_prior_results_unchanged"}
