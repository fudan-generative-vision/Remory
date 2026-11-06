"""Shared inference helpers; benchmark/training entry points are not distributed."""

from __future__ import annotations


from dataclasses import dataclass


from pathlib import Path


from typing import Any, Callable, Mapping, Optional, Sequence


from context_compression.grepseek_rollout import (
    GrepSeekRolloutModelGenerator,
    HeldoutEpisodeStart,
    load_heldout_episode_starts,
    run_rollout_episode,
)


from context_compression.grepseek_serving_contract import (
    CAUSAL_OBSERVATION_CACHE_V1,
    LEGACY_CHECKPOINT_SCHEMA,
    QWEN36_POST_COMPRESSION_GENERATION_POLICIES,
    QWEN36_POST_COMPRESSION_GENERATION_POLICY_CHECKPOINT_CAP,
    QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL,
    QWEN36_ROUND_RECURSIVE_V1,
    QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
    QWEN36_TOKEN_CLOCK_V1,
    RECURSIVE_HISTORY_V2,
    TOKEN_BUDGET_WORKING_MEMORY_WINDOW_POLICY,
    TOKEN_TRIGGERED_WORKING_MEMORY_POLICY,
    resolve_grepseek_checkpoint_serving_contract,
    validate_working_memory_token_budget,
)


from context_compression.grepseek_tool import (
    GrepSeekCorpusToolRunner,
    SHELL_CORPUS_MODES,
    SHELL_CORPUS_REACT_MEMORY,
    SHELL_CORPUS_WIKI18,
    append_browser_system_prompt,
    append_web_search_system_prompt,
    preflight_corpus_tool,
    react_memory_shell_system_prompt,
)


USER_QUERY_PREFIX = "Query: "


@dataclass(frozen=True)
class GrepSeekLiveBundle:
    """Long-lived, shareable resources for one live-demo checkpoint."""

    checkpoint: Path
    raw_parquet: Optional[Path]
    prepared_data: Optional[Path]
    system_prompt: str
    system_prompt_source_episode_id: str
    generator: GrepSeekRolloutModelGenerator
    tool_runner: GrepSeekCorpusToolRunner
    corpus_preflight: Mapping[str, Any]
    lock: Any
    config: Mapping[str, Any]
    target_model: str
    context_organization: str = CAUSAL_OBSERVATION_CACHE_V1
    checkpoint_context_organization: str = CAUSAL_OBSERVATION_CACHE_V1
    checkpoint_schema: str = LEGACY_CHECKPOINT_SCHEMA
    compression_ratio: Optional[int] = None
    recursive_working_memory_enabled: bool = False
    recursive_working_memory_policy: Optional[str] = None
    backend: str = "huggingface"
    sglang_url: Optional[str] = None
    web_search_enabled: bool = False
    browser_enabled: bool = False


def _load_real_system_prompt(
    raw_parquet: Path,
    split_config: Mapping[str, Any],
) -> tuple[str, str]:
    """Load only a held-out trajectory's initial system/user pair."""

    starts = load_heldout_episode_starts(
        raw_parquet,
        validation_fraction=float(split_config["validation_fraction"]),
        split_seed=int(split_config["seed"]),
        max_samples=1,
    )
    if len(starts) != 1:
        raise ValueError("expected exactly one raw trajectory for the system prompt")
    start = starts[0]
    system_message, user_message = start.messages
    system_prompt = str(system_message["content"])
    if not system_prompt:
        raise ValueError("raw GrepSeek system prompt must not be empty")
    if not str(user_message["content"]).startswith(USER_QUERY_PREFIX.rstrip()):
        raise ValueError("raw GrepSeek user message must use the 'Query:' contract")
    return system_prompt, str(start.episode_id)
