"""SGLang-backed turn generators and loader for the GrepSeek live demo.

The Streamlit process and SSE gateway remain CPU-only.  Checkpoint metadata
selects one of three fail-closed native context organizations: the legacy causal
cache compresses older tool observations independently, while
recursive-history-v2 replaces one episode memory after every completed
assistant/tool pair using ``P + M_prev + incremental_suffix``.  Recursive
checkpoints may additionally opt into an explicitly non-checkpoint-native
working-memory ablation.  Checkpoint-native working-memory-v3 uses the trained
``P + M_prev + semantic_W`` update and ``P + M_new + raw_W + G5`` recovery at
r16.  All paths share the same serialized SGLang client and raw-history
comparison arm.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from transformers import AutoTokenizer

from context_compression.evaluate_grepseek import (
    _same_path_or_id,
    read_checkpoint_config,
)
from context_compression.grepseek_data import (
    _apply_chat_ids,
    _canonical_tool_content,
    _decode,
    _encode,
    _render_chat,
    parse_tool_response,
)
from context_compression.grepseek_generation import (
    EOS_TOKEN_ID,
    THINK_CLOSE_TOKEN_ID,
    THINK_OPEN_TOKEN_ID,
    TOOL_OPEN_TOKEN_ID,
    parse_grepseek_continuation,
    parse_qwen36_continuation,
    trim_at_eos,
)
from context_compression.grepseek_live import (
    GrepSeekLiveBundle,
    _load_real_system_prompt,
)
from context_compression.grepseek_rollout import (
    ASSISTANT_CONTENT_PREFIX,
    RolloutEventSink,
    RolloutGeneration,
    _emit_rollout_event,
    _validate_rollout_history,
    build_rollout_feature,
)
from context_compression.grepseek_recursive_data import (
    ASSISTANT_OPENING_IDS,
    ASSISTANT_OPENING_TEXT,
    GENERATION_BOUNDARY_IDS,
    GENERATION_BOUNDARY_TEXT,
    USER_TURN_END_TEXT,
    _canonicalize_tools,
    _rendered_suffix,
)
from context_compression.qwen36_recursive_data import (
    QWEN36_ASSISTANT_OPENING_IDS,
    QWEN36_ASSISTANT_OPENING_TEXT,
    QWEN36_ENABLE_THINKING,
)
from context_compression.grepseek_serving_contract import (
    CAUSAL_OBSERVATION_CACHE_V1,
    LATEST_TOOL_EXCHANGE_RAW_V1,
    LEGACY_CHECKPOINT_SCHEMA,
    ONE_ROUND_LAG_RECURSIVE_WORKING_MEMORY_V3,
    QWEN36_POST_COMPRESSION_GENERATION_POLICIES,
    QWEN36_POST_COMPRESSION_GENERATION_POLICY_CHECKPOINT_CAP,
    QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL,
    QWEN36_ROUND_RECURSIVE_SOURCE_CONTRACT,
    QWEN36_ROUND_RECURSIVE_STATE_POLICY,
    QWEN36_ROUND_EXPERIMENTAL_GENERATION_BUDGET_POLICY,
    QWEN36_ROUND_EXPERIMENTAL_TOOL_RESULT_FIT_POLICY,
    QWEN36_ROUND_GENERATION_RESERVE_TOKENS,
    QWEN36_ROUND_DYNAMIC_GENERATION_BUDGET_POLICY,
    QWEN36_ROUND_FIXED_GENERATION_BUDGET_POLICY,
    QWEN36_ROUND_RECURSIVE_V1,
    QWEN36_ROUND_TOOL_RESULT_FIT_POLICY,
    QWEN36_ROUND_WORKING_MEMORY_POLICY,
    QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
    QWEN36_TOKEN_CLOCK_V1,
    QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
    RECURSIVE_HISTORY_V2,
    RECURSIVE_HISTORY_WITH_RAW_WORKING_MEMORY_ABLATION_V1,
    RECURSIVE_WORKING_MEMORY_V3,
    REPLACE_PREVIOUS_MEMORY_KEEP_LATEST_RAW_WORKING_MEMORY_V3,
    TOKEN_BUDGET_WORKING_MEMORY_WINDOW_POLICY,
    TOKEN_TRIGGERED_WORKING_MEMORY_POLICY,
    WORKING_MEMORY_V3_TOOL_RESULT_FIT_POLICY,
    WORKING_MEMORY_CHECKPOINT_SCHEMAS,
    WORKING_MEMORY_V3_SOURCE_CONTRACT,
    WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS,
    GrepSeekCheckpointServingContract,
    ResolvedGrepSeekLiveData,
    resolve_grepseek_checkpoint_serving_contract,
    resolve_grepseek_live_data_paths,
    validate_working_memory_token_budget,
)
from context_compression.grepseek_sglang_client import (
    GrepSeekSGLangClient,
    MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1,
    MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1,
    SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1,
    SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1,
    SGLangGeneration,
    TPGPUMemoryHandle,
)
from context_compression.grepseek_tool import (
    GrepSeekCorpusToolRunner,
    ToolResult,
    preflight_corpus_tool,
    react_memory_shell_system_prompt,
)
from context_compression.rollout_grepseek import _assert_token_contract


MAX_WORKING_MEMORY_TURNS = 32
TOOL_CALL_TURN_GUIDANCE_POLICY = "dynamic_user_suffix_excluded_from_compression_v1"
QWEN36_TOOL_RESPONSE_TURN_GUIDANCE_POLICY = (
    "dynamic_native_tool_response_suffix_in_history_v1"
)
_WORKING_MEMORY_V3_TOOL_TRUNCATION_MARKER = (
    "\n[tool output truncated to fit recursive semantic-memory capacity]"
)
_QWEN36_TOOL_NAMES = frozenset({"shell", "web_search", "browser"})
QWEN36_GENERATION_SAMPLING_POLICY = "qwen38_evaluation_sampled_t1_p095_k20_v1"
QWEN38_GENERATION_SAMPLING_POLICY = "qwen38_evaluation_sampled_t1_p095_k20_v1"
QWEN36_GENERATION_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
QWEN36_OVERSIZED_COMPLETE_ROUND_EVICTION_POLICY = (
    "whole_complete_round_above_serving_pool_per_wm_slot_" "latest_successor_grace_v2"
)
CHECKPOINT_1024_SERVING_EVICTION_POLICY = "checkpoint_1024_v1"
QWEN38_8192_ACCUMULATION_SERVING_EVICTION_POLICY = (
    "qwen38_serving_8192_accumulation_canary_v1"
)
QWEN38_SERVING_EVICTION_ACCUMULATION_TOKENS = 8_192
CODEX_NATIVE_COMPACTION_POLICY = "codex_remote_compaction_v2_compressor_v1"
_CODEX_NATIVE_COMPACTION_FORCED_FLUSH_REASON = (
    "codex_native_compaction_target"
)
# Incremental history rendering is deliberately pinned to the audited local
# Qwen3.8 template.  A different template stays on the legacy full-history
# oracle even when the feature is explicitly enabled.
QWEN38_INCREMENTAL_HISTORY_CHAT_TEMPLATE_SHA256 = (
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
)
QWEN38_INCREMENTAL_HISTORY_RECEIPT_POLICY = "qwen38_append_only_native_chat_receipt_v1"
NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER = "legacy_full_render_v1"
NATIVE_HISTORY_POLICY_APPEND_ONLY_TOKEN_EXACT = "append_only_token_exact_v1"
R2EACT_RAW_SUFFIX_IMAGE_POLICY = "request_local_raw_suffix_images_v1"
R2EACT_RAW_SUFFIX_IMAGE_CONTEXT_BUDGET_POLICY = (
    "qwen38_max_16384_visual_tokens_per_image_conservative_reserve_v1"
)
# Audited against Qwen3.8-27B/preprocessor_config.json and Qwen3VLProcessor:
# longest_edge is a pixel-area ceiling, while one LLM visual token covers one
# (patch_size * merge_size)^2 region.  A still image has grid_t == 1.
QWEN38_RAW_SUFFIX_IMAGE_MAX_PIXELS = 16_777_216
QWEN38_RAW_SUFFIX_IMAGE_PATCH_SIZE = 16
QWEN38_RAW_SUFFIX_IMAGE_MERGE_SIZE = 2
QWEN38_RAW_SUFFIX_IMAGE_MAX_VISUAL_TOKENS_PER_IMAGE = (
    QWEN38_RAW_SUFFIX_IMAGE_MAX_PIXELS
    // (
        QWEN38_RAW_SUFFIX_IMAGE_PATCH_SIZE
        * QWEN38_RAW_SUFFIX_IMAGE_MERGE_SIZE
    )
    ** 2
)
# The unprocessed prompt already contains one <|image_pad|> per image, so the
# exact maximum expansion delta is 16,383.  Reserving the full visual-token
# ceiling leaves one additional context token of headroom per image.
QWEN38_RAW_SUFFIX_IMAGE_CONTEXT_RESERVE_TOKENS_PER_IMAGE = (
    QWEN38_RAW_SUFFIX_IMAGE_MAX_VISUAL_TOKENS_PER_IMAGE
)
_NATIVE_HISTORY_POLICIES = frozenset(
    {
        NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER,
        NATIVE_HISTORY_POLICY_APPEND_ONLY_TOKEN_EXACT,
    }
)
_TOKEN_CLOCK_SERVING_EVICTION_POLICIES = frozenset(
    {
        CHECKPOINT_1024_SERVING_EVICTION_POLICY,
        QWEN38_8192_ACCUMULATION_SERVING_EVICTION_POLICY,
    }
)


def _token_clock_sampling_telemetry(policy: str) -> dict[str, Any]:
    return {
        "generation_sampling_policy": policy,
        **{
            f"generation_{key}": value
            for key, value in QWEN36_GENERATION_SAMPLING.items()
        },
    }


def _qwen36_sampling_telemetry() -> dict[str, Any]:
    """Retain the historical helper for Qwen3.6 callers and tests."""

    return _token_clock_sampling_telemetry(QWEN36_GENERATION_SAMPLING_POLICY)


@dataclass(frozen=True)
class _TokenClockCheckpointRuntime:
    target_family: str
    display_name: str
    max_encoder_tokens: int
    generation_sampling_policy: str
    telemetry_mode: str
    source_episode_id: str
    checkpoint_cap_policy: str
    dynamic_generation_budget_policy: str
    fixed_generation_budget_policy: str


@dataclass(frozen=True)
class _RoundEvictionEpisodeSnapshot:
    """Immutable inputs shared by one eviction projection and its executor."""

    serving_eviction_policy: str
    raw_prefix_tokens: int
    memory_tokens: int
    consumed_history_tokens: int
    consumed_round_count: int
    round_end_tokens: tuple[int, ...]
    completed_round_count: int


@dataclass(frozen=True)
class _RoundEvictionTransitionSnapshot:
    """One contiguous recursive update selected by the pure planner."""

    history_start: int
    history_end: int
    raw_chunk_tokens: int
    memory_tokens_before: int
    memory_tokens_after: int
    source_encoder_tokens: int


_TOKEN_CLOCK_RUNTIME_BY_SCHEMA = {
    QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA: _TokenClockCheckpointRuntime(
        target_family="qwen36",
        display_name="Qwen3.6",
        max_encoder_tokens=8_192,
        generation_sampling_policy=QWEN36_GENERATION_SAMPLING_POLICY,
        telemetry_mode="qwen36_round_recursive",
        source_episode_id="qwen36-token-clock-serving-v1",
        checkpoint_cap_policy=QWEN36_POST_COMPRESSION_GENERATION_POLICY_CHECKPOINT_CAP,
        dynamic_generation_budget_policy=QWEN36_ROUND_DYNAMIC_GENERATION_BUDGET_POLICY,
        fixed_generation_budget_policy=QWEN36_ROUND_FIXED_GENERATION_BUDGET_POLICY,
    ),
    QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA: _TokenClockCheckpointRuntime(
        target_family="qwen38",
        display_name="Qwen3.8",
        max_encoder_tokens=12_288,
        generation_sampling_policy=QWEN38_GENERATION_SAMPLING_POLICY,
        telemetry_mode="r2eact_round_recursive",
        source_episode_id="qwen38-r2eact-token-clock-serving-v1",
        checkpoint_cap_policy="checkpoint_12288_cap",
        dynamic_generation_budget_policy=(
            "remaining_context_before_first_compression_then_checkpoint_12288_cap"
        ),
        fixed_generation_budget_policy=(
            "configured_cap_before_first_compression_then_checkpoint_12288_cap"
        ),
    ),
}


def _token_clock_runtime_for_schema(
    checkpoint_schema: str,
) -> _TokenClockCheckpointRuntime:
    try:
        return _TOKEN_CLOCK_RUNTIME_BY_SCHEMA[str(checkpoint_schema)]
    except KeyError as exc:
        raise ValueError(
            "token-clock runtime requires a verified Qwen3.6 or Qwen3.8 "
            f"deployment schema, got {checkpoint_schema!r}"
        ) from exc


def _qwen36_tool_schemas(
    tool_names: Sequence[str], *, model_display_name: str = "Qwen3.6"
) -> list[dict[str, Any]]:
    """Return the native Qwen3.5+ function schemas for enabled live tools."""

    names = tuple(str(name) for name in tool_names)
    if len(set(names)) != len(names) or set(names) - _QWEN36_TOOL_NAMES:
        raise ValueError(
            f"{model_display_name} tools must be a unique supported sequence"
        )
    definitions: dict[str, dict[str, Any]] = {
        "shell": {
            "name": "shell",
            "description": (
                "Act as a hippocampus-like exact-recall function over "
                "memory.jsonl, the read-only episodic archive containing the "
                "complete raw ReAct exchanges affected by context compression. "
                "Use one safe rg/grep/find/sed/awk pipeline; this memory recalls "
                "prior evidence but does not provide new external facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "One read-only rg/grep/find/sed/awk pipeline that "
                            "names memory.jsonl exactly once."
                        ),
                    }
                },
                "required": ["command"],
            },
        },
        "web_search": {
            "name": "web_search",
            "description": "Search the public web and return compact snippets and URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    "gl": {"type": "string"},
                    "hl": {"type": "string"},
                    "timeout": {"type": "number", "minimum": 1, "maximum": 60},
                },
                "required": ["query"],
            },
        },
        "browser": {
            "name": "browser",
            "description": "Open a public HTTP(S) page and return readable page text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "timeout": {"type": "number", "minimum": 1, "maximum": 60},
                    "max_length": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32000,
                    },
                },
                "required": ["url"],
            },
        },
    }
    return [{"type": "function", "function": definitions[name]} for name in names]


def _apply_qwen36_chat_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    tool_names: Sequence[str],
    add_generation_prompt: bool,
    model_display_name: str = "Qwen3.6",
) -> list[int]:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": bool(add_generation_prompt),
        # Keep Qwen's native reasoning prefix enabled.  GrepSeek's official
        # agent contract is think -> tool_call/tool_response -> answer; the
        # step-1750 compressor data is OOD for that loop and must not silently
        # redefine the target model's generation protocol.
        "enable_thinking": QWEN36_ENABLE_THINKING,
        # Qwen3.6 recommends retaining historical reasoning for agent tasks.
        # This is redundant for the common one-user ReAct chain, but makes
        # the native template contract explicit and stable.
        "preserve_thinking": True,
        "return_dict": False,
    }
    schemas = _qwen36_tool_schemas(tool_names, model_display_name=model_display_name)
    if schemas:
        kwargs["tools"] = schemas
    result = tokenizer.apply_chat_template(
        [dict(message) for message in messages], **kwargs
    )
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes)):
        raise TypeError(f"{model_display_name} chat template must return token ids")
    return [int(value) for value in result]


def _observation_role_counts(
    messages: Sequence[Mapping[str, str]],
) -> tuple[int, int, int]:
    """Return completed rounds, tool observations, and user-feedback rounds."""

    roles = [str(message.get("role", "")) for message in messages]
    if len(roles) < 2 or roles[:2] != ["system", "user"]:
        raise ValueError("observation counting requires initial system/user messages")
    history_roles = roles[2:]
    if len(history_roles) % 2:
        raise ValueError("observation counting requires complete round pairs")
    tool_observations = 0
    user_feedback_observations = 0
    for offset in range(0, len(history_roles), 2):
        assistant_role, observation_role = history_roles[offset : offset + 2]
        if assistant_role != "assistant" or observation_role not in {"tool", "user"}:
            raise ValueError(
                "observation counting requires assistant/(tool or user-feedback) pairs"
            )
        if observation_role == "tool":
            tool_observations += 1
        else:
            user_feedback_observations += 1
    return len(history_roles) // 2, tool_observations, user_feedback_observations


def _tool_call_turn_guidance(
    *,
    max_turns: Optional[int],
    round_index: int,
    current_tool_calls: Optional[int] = None,
) -> Optional[str]:
    """Return the request-local user suffix for this generation round."""

    if max_turns is None:
        return None
    if (
        isinstance(max_turns, bool)
        or not isinstance(max_turns, int)
        or not 1 <= max_turns <= MAX_WORKING_MEMORY_TURNS
    ):
        raise ValueError(
            f"max_turns must be between 1 and {MAX_WORKING_MEMORY_TURNS} or None"
        )
    if (
        isinstance(round_index, bool)
        or not isinstance(round_index, int)
        or not 1 <= round_index <= max_turns
    ):
        raise ValueError("round_index must be between 1 and max_turns")
    max_tool_calls = max_turns - 1
    if current_tool_calls is None:
        current_tool_calls = round_index - 1
    if (
        isinstance(current_tool_calls, bool)
        or not isinstance(current_tool_calls, int)
        or not 0 <= current_tool_calls <= round_index - 1
    ):
        raise ValueError("current_tool_calls must fit the completed round history")
    action = (
        "continue"
        if round_index < max_turns and current_tool_calls < max_tool_calls
        else "answer"
    )
    return (
        "You are expected to give the final answer within "
        f"{max_tool_calls} turns of tool_calls, now you have called tools for "
        f"{current_tool_calls} times, please {action}"
    )


def _messages_with_tool_call_turn_guidance(
    messages: Sequence[Mapping[str, str]],
    *,
    max_turns: Optional[int],
    round_index: int,
    current_tool_calls: Optional[int] = None,
) -> tuple[list[dict[str, str]], Optional[str]]:
    """Append dynamic guidance to the initial user content without mutation."""

    copied = [
        {
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
        }
        for message in messages
    ]
    if current_tool_calls is None and len(copied) > 2:
        _completed_rounds, current_tool_calls, _user_feedback_rounds = (
            _observation_role_counts(copied)
        )
    guidance = _tool_call_turn_guidance(
        max_turns=max_turns,
        round_index=round_index,
        current_tool_calls=current_tool_calls,
    )
    if guidance is None:
        return copied, None
    if len(copied) < 2 or copied[0]["role"] != "system" or copied[1]["role"] != "user":
        raise ValueError("turn guidance requires initial system/user messages")
    copied[1]["content"] = f"{copied[1]['content'].rstrip()}\n\n{guidance}"
    return copied, guidance


def _qwen36_messages_with_tool_response_turn_guidance(
    messages: Sequence[Mapping[str, str]],
    *,
    max_turns: Optional[int],
    round_index: int,
    model_display_name: str = "Qwen3.6",
) -> tuple[list[dict[str, str]], Optional[str]]:
    """Put the dynamic budget inside each native tool-response envelope.

    Qwen3.6's official template wraps every ``role=tool`` message as
    ``<tool_response>CONTENT</tool_response>``.  Appending the guidance to the
    message content therefore places it immediately before the closing tag
    without modifying persistent system/user prefix P.  Historical tool
    messages are annotated from their own ordinal so the rendered token
    history remains append-only across rounds.
    """

    copied = [
        {
            "role": str(message.get("role", "")),
            "content": str(message.get("content", "")),
        }
        for message in messages
    ]
    if max_turns is None:
        return copied, None
    if len(copied) < 2 or [message["role"] for message in copied[:2]] != [
        "system",
        "user",
    ]:
        raise ValueError(
            f"{model_display_name} turn guidance requires initial system/user messages"
        )
    history = copied[2:]
    if len(history) % 2:
        raise ValueError(
            f"{model_display_name} turn guidance requires complete "
            "assistant/observation pairs"
        )
    completed_rounds = 0
    current_tool_calls = 0
    latest_guidance: Optional[str] = None
    for offset in range(0, len(history), 2):
        assistant, observation = history[offset : offset + 2]
        if assistant["role"] != "assistant" or observation["role"] not in {
            "tool",
            "user",
        }:
            raise ValueError(
                f"{model_display_name} turn guidance requires atomic "
                "assistant/(tool or user-feedback) groups"
            )
        completed_rounds += 1
        if observation["role"] == "tool":
            current_tool_calls += 1
            guidance = _tool_call_turn_guidance(
                max_turns=max_turns,
                round_index=completed_rounds + 1,
                current_tool_calls=current_tool_calls,
            )
            assert guidance is not None
            observation["content"] = (
                f"{observation['content'].rstrip()}\n\n{guidance}"
            )
            latest_guidance = guidance
        else:
            latest_guidance = None
    if completed_rounds != round_index - 1:
        raise ValueError(
            f"{model_display_name} turn guidance requires exactly one completed "
            "observation before each subsequent generation round"
        )
    return copied, latest_guidance


def _validate_server_input_length(
    *,
    token_count: int,
    server_max_req_input_len: Optional[int],
    label: str,
) -> None:
    if server_max_req_input_len is None:
        return
    input_threshold = int(server_max_req_input_len)
    if input_threshold <= 1:
        raise ValueError("server_max_req_input_len must be greater than one")
    if int(token_count) >= input_threshold:
        raise ValueError(
            f"{label} input context is exhausted: "
            f"prefix_tokens={int(token_count)}, "
            f"maximum_accepted_input_tokens={input_threshold - 1}"
        )


def _generation_token_budget(
    *,
    prefix_token_count: int,
    configured_max_new_tokens: int,
    server_context_length: Optional[int],
    server_max_total_num_tokens: Optional[int],
    server_max_req_input_len: Optional[int],
    use_remaining_context: bool,
    remaining_context_output_ceiling_tokens: Optional[int] = None,
) -> int:
    """Return a request budget that SGLang will accept without clamping.

    SGLang rejects ``input + max_new >= context_length`` and its scheduler
    reserves one further token below ``min(context_length, token_pool)``.
    Therefore a request that uses the whole remaining serving window must
    leave two tokens of headroom.
    """

    if isinstance(prefix_token_count, bool) or int(prefix_token_count) <= 0:
        raise ValueError("generation prefix must contain at least one token")
    if (
        isinstance(configured_max_new_tokens, bool)
        or int(configured_max_new_tokens) <= 0
    ):
        raise ValueError("configured_max_new_tokens must be positive")
    if remaining_context_output_ceiling_tokens is not None and (
        isinstance(remaining_context_output_ceiling_tokens, bool)
        or int(remaining_context_output_ceiling_tokens) <= 0
    ):
        raise ValueError(
            "remaining_context_output_ceiling_tokens must be positive or None"
        )
    prefix_tokens = int(prefix_token_count)
    _validate_server_input_length(
        token_count=prefix_tokens,
        server_max_req_input_len=server_max_req_input_len,
        label="generation",
    )

    limits = [
        int(value)
        for value in (server_context_length, server_max_total_num_tokens)
        if value is not None
    ]
    if not limits:
        if use_remaining_context:
            raise ValueError(
                "remaining-context generation requires SGLang serving limits"
            )
        return int(configured_max_new_tokens)
    if any(value <= 2 for value in limits):
        raise ValueError("SGLang serving token limits must be greater than two")

    remaining = min(limits) - prefix_tokens - 2
    if remaining <= 0:
        raise ValueError(
            "generation context is exhausted: "
            f"prefix_tokens={prefix_tokens}, total_token_limit={min(limits)}"
        )
    if use_remaining_context:
        if remaining_context_output_ceiling_tokens is None:
            return remaining
        return min(int(remaining_context_output_ceiling_tokens), remaining)
    return min(int(configured_max_new_tokens), remaining)


def _working_memory_v3_generation_token_budget(
    *,
    prefix_token_count: int,
    server_prefix_token_count: Optional[int] = None,
    max_encoder_tokens: int,
    configured_max_new_tokens: int,
    server_context_length: Optional[int],
    server_max_total_num_tokens: Optional[int],
    server_max_req_input_len: Optional[int],
    use_remaining_context: bool,
) -> int:
    """Clamp generation to both SGLang and the v3 trained recovery cap."""

    server_budget = _generation_token_budget(
        prefix_token_count=(
            prefix_token_count
            if server_prefix_token_count is None
            else server_prefix_token_count
        ),
        configured_max_new_tokens=configured_max_new_tokens,
        server_context_length=server_context_length,
        server_max_total_num_tokens=server_max_total_num_tokens,
        server_max_req_input_len=server_max_req_input_len,
        use_remaining_context=use_remaining_context,
    )
    trained_budget = int(max_encoder_tokens) - int(prefix_token_count)
    if trained_budget <= 0:
        raise ValueError(
            "working-memory-v3 recovery prefix exhausts the trained encoder cap: "
            f"prefix_tokens={int(prefix_token_count)}, "
            f"max_encoder_tokens={int(max_encoder_tokens)}"
        )
    return min(server_budget, trained_budget)


def _server_positive_integer(server_info: Mapping[str, Any], name: str) -> int:
    value = server_info.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(
            f"SGLang /server_info is missing a positive integer {name!r}"
        )
    return int(value)


def _finite(value: float) -> Optional[float]:
    result = float(value)
    return result if math.isfinite(result) else None


def _causal_observation_encode_input(
    source_prompt_ids: Sequence[int],
    evidence_span: Sequence[int],
    *,
    max_encoder_tokens: Optional[int],
) -> tuple[list[int], list[int], int]:
    """End at one evidence span and safely left-crop its causal prefix."""

    prompt = [int(value) for value in source_prompt_ids]
    if len(evidence_span) != 2:
        raise ValueError("each source evidence span must be [start, end]")
    start, end = (int(evidence_span[0]), int(evidence_span[1]))
    if start < 0 or end <= start or end > len(prompt):
        raise ValueError("source evidence span lies outside source prompt")
    causal_prompt = prompt[:end]
    if max_encoder_tokens is None or len(causal_prompt) <= max_encoder_tokens:
        return causal_prompt, [start, end], 0
    drop = len(causal_prompt) - int(max_encoder_tokens)
    if drop > start:
        raise ValueError(
            "one tool observation exceeds max_encoder_tokens after causal cropping"
        )
    return causal_prompt[drop:], [start - drop, end - drop], drop


def _pack_cached_memory_prefix(
    discrete_segments: Sequence[Sequence[int]],
    slot_counts: Sequence[int],
    *,
    placeholder_token_id: int,
) -> tuple[list[int], list[int]]:
    if len(discrete_segments) != len(slot_counts) + 1:
        raise ValueError("discrete/memory interleave arity mismatch")
    prefix = [int(value) for value in discrete_segments[0]]
    memory_positions: list[int] = []
    for slot_count, discrete in zip(slot_counts, discrete_segments[1:]):
        count = int(slot_count)
        if count <= 0:
            raise ValueError("every cached observation needs at least one memory slot")
        memory_positions.extend(range(len(prefix), len(prefix) + count))
        prefix.extend([int(placeholder_token_id)] * count)
        prefix.extend(int(value) for value in discrete)
    if not prefix:
        raise ValueError("generation prefix must not be empty")
    return prefix, memory_positions


def _observation_signature(
    encoder_input_ids: Sequence[int],
    evidence_span: Sequence[int],
    source_input_ids: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    for sequence in (encoder_input_ids, evidence_span, source_input_ids):
        digest.update(len(sequence).to_bytes(8, "little", signed=False))
        for value in sequence:
            integer = int(value)
            if integer < 0:
                raise ValueError("observation cache signatures require nonnegative ids")
            digest.update(integer.to_bytes(8, "little", signed=False))
    return digest.hexdigest()


@dataclass(frozen=True)
class _CachedObservationMemory:
    signature: str
    embeddings: torch.Tensor
    source_token_length: int
    source_encoder_tokens: int
    history_tokens_dropped: int


@dataclass(frozen=True)
class _RecursiveTurnInput:
    raw_prefix_input_ids: tuple[int, ...]
    incremental_suffix_input_ids: Optional[tuple[int, ...]]
    transition_index: Optional[int]


@dataclass(frozen=True)
class _WorkingMemoryV3TurnInput:
    """Exact live reconstruction of v3 P/raw-W/semantic-W arrays."""

    raw_prefix_input_ids: tuple[int, ...]
    latest_raw_working_memory_input_ids: Optional[tuple[int, ...]]
    latest_semantic_working_memory_input_ids: Optional[tuple[int, ...]]
    working_memory_index: Optional[int]
    generation_raw_prefix_input_ids: Optional[tuple[int, ...]] = None
    generation_user_suffix_tokens: int = 0


@dataclass(frozen=True)
class _Qwen38IncrementalHistoryReceipt:
    """Episode-local proof receipt for append-only Qwen3.8 chat history.

    Messages are stored as immutable ``(role, content)`` pairs so a caller
    cannot mutate an already accepted prefix through a shared dictionary.
    The raw-message digests protect the pre-canonicalization trajectory,
    while the canonical/guided pairs are the exact values whose isolated
    native renders produced ``history_input_ids``.
    """

    episode_id: str
    template_sha256: str
    checkpoint_schema: str
    tool_names: tuple[str, ...]
    max_turns: Optional[int]
    raw_message_digests: tuple[str, ...]
    canonical_messages: tuple[tuple[str, str], ...]
    guided_messages: tuple[tuple[str, str], ...]
    raw_prefix_input_ids: tuple[int, ...]
    history_input_ids: tuple[int, ...]
    message_end_tokens: tuple[int, ...]
    latest_guidance: Optional[str]


@dataclass(frozen=True)
class _TokenClockHistoryRender:
    canonical_messages: tuple[tuple[str, str], ...]
    guided_messages: tuple[tuple[str, str], ...]
    guidance: Optional[str]
    raw_prefix_input_ids: tuple[int, ...]
    full_discrete_input_ids: tuple[int, ...]
    history_input_ids: tuple[int, ...]
    message_end_tokens: tuple[int, ...]
    staged_receipt: Optional[_Qwen38IncrementalHistoryReceipt]
    render_policy: str
    appended_message_count: int


@dataclass(frozen=True)
class R2EActRawSuffixImage:
    """One image attached to a completed round that is still raw.

    ``completed_round_index`` is one-based: an image returned by the first
    completed assistant/tool exchange belongs to round 1 and can first be sent
    while generating round 2. The gateway supplies the complete append-only
    attachment list on every call. The generator admits only attachments whose
    owner round remains to the right of the compression cursor.

    ``image_data`` is the native SGLang ``/generate`` image value (normally a
    bounded local/data URL materialized by the gateway). It is never sent to
    the compressor or copied into compression telemetry.
    """

    completed_round_index: int
    image_data: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.completed_round_index, bool)
            or not isinstance(self.completed_round_index, int)
            or self.completed_round_index < 1
        ):
            raise ValueError("completed_round_index must be a positive integer")
        if not isinstance(self.image_data, str) or not self.image_data:
            raise ValueError("image_data must be a nonempty string")


def _select_raw_suffix_images(
    images: Sequence[R2EActRawSuffixImage | Mapping[str, Any]],
    *,
    consumed_round_count: int,
    completed_round_count: int,
) -> tuple[R2EActRawSuffixImage, ...]:
    """Validate the attachment clock and retain only uncompressed images."""

    if isinstance(images, (str, bytes)) or not isinstance(images, Sequence):
        raise TypeError("raw_suffix_images must be a sequence")
    if not 0 <= consumed_round_count <= completed_round_count:
        raise ValueError("invalid compressed/completed round frontier")
    selected: list[R2EActRawSuffixImage] = []
    previous_round = 0
    for index, item in enumerate(images):
        if isinstance(item, Mapping):
            try:
                item = R2EActRawSuffixImage(
                    completed_round_index=item["completed_round_index"],
                    image_data=item["image_data"],
                )
            except KeyError as exc:
                raise ValueError(
                    f"raw_suffix_images[{index}] is missing {exc.args[0]!r}"
                ) from exc
        if not isinstance(item, R2EActRawSuffixImage):
            raise TypeError(
                f"raw_suffix_images[{index}] must be R2EActRawSuffixImage or a mapping"
            )
        owner = item.completed_round_index
        if owner < previous_round:
            raise ValueError("raw_suffix_images must be ordered by completed round")
        if owner > completed_round_count:
            raise ValueError(
                "raw_suffix_images cannot refer to an incomplete or future round"
            )
        previous_round = owner
        if owner > consumed_round_count:
            selected.append(item)
    return tuple(selected)


def _raw_suffix_image_context_reserve_tokens(image_count: int) -> int:
    """Return the conservative server-context reserve for Qwen3.8 images."""

    if (
        isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or image_count < 0
    ):
        raise ValueError("image_count must be a non-negative integer")
    return (
        image_count * QWEN38_RAW_SUFFIX_IMAGE_CONTEXT_RESERVE_TOKENS_PER_IMAGE
    )


def _immutable_message_pairs(
    messages: Sequence[Mapping[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(message["role"]), str(message["content"])) for message in messages
    )


def _message_pair_dicts(
    messages: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    return [{"role": role, "content": content} for role, content in messages]


def _raw_message_digest(message: Mapping[str, str]) -> str:
    """Hash the exact normalized input consumed by tool canonicalization."""

    role = str(message["role"])
    content = str(message["content"])
    digest = hashlib.sha256()
    for value in (role, content):
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _build_recursive_turn_input(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    episode_id: str,
    round_index: int,
) -> _RecursiveTurnInput:
    """Reconstruct the exact recursive-v2 ``P`` and newest history suffix."""

    history = _validate_rollout_history(str(episode_id), messages)
    expected_round = len(history) // 2
    if int(round_index) != expected_round:
        raise ValueError(
            "recursive rollout round/history mismatch: "
            f"round_index={int(round_index)}, expected={expected_round}"
        )
    canonical, _released_lengths, _canonical_lengths = _canonicalize_tools(
        str(episode_id), history
    )
    initial_prompt_ids = _apply_chat_ids(
        tokenizer, canonical[:2], add_generation_prompt=True
    )
    boundary = [int(value) for value in GENERATION_BOUNDARY_IDS]
    if (
        len(initial_prompt_ids) <= len(boundary)
        or initial_prompt_ids[-len(boundary) :] != boundary
    ):
        raise ValueError(
            "recursive generation prompt does not end in the native G7 boundary"
        )
    raw_prefix = tuple(int(value) for value in initial_prompt_ids[: -len(boundary)])
    if len(canonical) == 2:
        return _RecursiveTurnInput(
            raw_prefix_input_ids=raw_prefix,
            incremental_suffix_input_ids=None,
            transition_index=None,
        )

    # The newest completed assistant/tool pair is independently rendered as a
    # text suffix after its real native generation prompt, exactly as in the
    # recursive-v2 materializer.  Earlier pairs are represented only by M_prev.
    _, completed_suffix = _rendered_suffix(tokenizer, canonical[:-2], canonical)
    suffix = tuple(boundary + _encode(tokenizer, completed_suffix))
    if len(suffix) <= len(boundary) or suffix[: len(boundary)] != tuple(boundary):
        raise ValueError("recursive incremental suffix lost its native G7 boundary")
    return _RecursiveTurnInput(
        raw_prefix_input_ids=raw_prefix,
        incremental_suffix_input_ids=suffix,
        transition_index=int(round_index) - 1,
    )


def _build_working_memory_v3_turn_input(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    episode_id: str,
    round_index: int,
    max_turns: Optional[int] = None,
) -> _WorkingMemoryV3TurnInput:
    """Reconstruct the checkpoint-native v3 P, newest raw W, and semantic W."""

    history = _validate_rollout_history(str(episode_id), messages)
    expected_round = len(history) // 2
    if int(round_index) != expected_round:
        raise ValueError(
            "working-memory-v3 rollout round/history mismatch: "
            f"round_index={int(round_index)}, expected={expected_round}"
        )
    canonical, _released_lengths, _canonical_lengths = _canonicalize_tools(
        str(episode_id), history
    )

    initial_messages = canonical[:2]
    raw_prefix_text = _render_chat(
        tokenizer, initial_messages, add_generation_prompt=False
    )
    raw_prefix_ids = _apply_chat_ids(
        tokenizer, initial_messages, add_generation_prompt=False
    )
    initial_prompt_text = _render_chat(
        tokenizer, initial_messages, add_generation_prompt=True
    )
    initial_prompt_ids = _apply_chat_ids(
        tokenizer, initial_messages, add_generation_prompt=True
    )
    assistant_opening = [int(value) for value in ASSISTANT_OPENING_IDS]
    if not raw_prefix_text.endswith(USER_TURN_END_TEXT):
        raise ValueError(
            "working-memory-v3 P does not include the complete user turn close"
        )
    if initial_prompt_text != raw_prefix_text + ASSISTANT_OPENING_TEXT:
        raise ValueError(
            "working-memory-v3 initial prompt is not exact P followed by G5"
        )
    if initial_prompt_ids != raw_prefix_ids + assistant_opening:
        raise ValueError(
            "working-memory-v3 initial prompt ids are not exact P followed by G5"
        )
    if _encode(tokenizer, raw_prefix_text) != raw_prefix_ids:
        raise ValueError("working-memory-v3 P is not independently tokenization-stable")
    raw_prefix = tuple(int(value) for value in raw_prefix_ids)
    if not raw_prefix:
        raise ValueError("working-memory-v3 persistent P is empty")
    generation_initial_messages, guidance = _messages_with_tool_call_turn_guidance(
        initial_messages,
        max_turns=max_turns,
        round_index=round_index,
    )
    generation_prefix_ids = tuple(
        _apply_chat_ids(
            tokenizer,
            generation_initial_messages,
            add_generation_prompt=False,
        )
    )
    generation_prompt_ids = _apply_chat_ids(
        tokenizer,
        generation_initial_messages,
        add_generation_prompt=True,
    )
    if generation_prompt_ids != list(generation_prefix_ids) + assistant_opening:
        raise ValueError(
            "working-memory-v3 guided generation prefix lost the native G5 boundary"
        )
    generation_user_suffix_tokens = len(generation_prefix_ids) - len(raw_prefix)
    if guidance is None:
        if generation_prefix_ids != raw_prefix or generation_user_suffix_tokens != 0:
            raise ValueError("unguided working-memory-v3 generation prefix drifted")
    elif generation_user_suffix_tokens <= 0:
        raise ValueError("working-memory-v3 user guidance tokenized empty")
    if len(canonical) == 2:
        return _WorkingMemoryV3TurnInput(
            raw_prefix,
            None,
            None,
            None,
            generation_prefix_ids,
            generation_user_suffix_tokens,
        )

    assistant_message = canonical[-2]
    tool_message = canonical[-1]
    if (
        assistant_message.get("role") != "assistant"
        or tool_message.get("role") != "tool"
    ):
        raise ValueError(
            "working-memory-v3 newest completed exchange must be assistant/tool"
        )
    assistant_semantic_ids = _encode(
        tokenizer, str(assistant_message.get("content", ""))
    )
    tool_semantic_ids = _encode(tokenizer, str(tool_message.get("content", "")))
    if not assistant_semantic_ids or not tool_semantic_ids:
        raise ValueError(
            "working-memory-v3 assistant and tool semantic content must tokenize nonempty"
        )
    semantic = tuple(
        [int(value) for value in assistant_semantic_ids]
        + [int(value) for value in tool_semantic_ids]
    )

    prior_history = canonical[:-2]
    _, completed_suffix = _rendered_suffix(tokenizer, prior_history, canonical)
    raw_working_memory_text = ASSISTANT_OPENING_TEXT + completed_suffix
    raw_working_memory = tuple(_encode(tokenizer, raw_working_memory_text))
    if (
        len(raw_working_memory) <= len(assistant_opening)
        or list(raw_working_memory[: len(assistant_opening)]) != assistant_opening
    ):
        raise ValueError("working-memory-v3 raw W lost its native G5 opening")
    prior_text = _render_chat(tokenizer, prior_history, add_generation_prompt=False)
    complete_text = _render_chat(tokenizer, canonical, add_generation_prompt=False)
    if complete_text != prior_text + raw_working_memory_text:
        raise ValueError(
            "working-memory-v3 newest raw W does not reproduce the native chat span"
        )
    return _WorkingMemoryV3TurnInput(
        raw_prefix_input_ids=raw_prefix,
        latest_raw_working_memory_input_ids=raw_working_memory,
        latest_semantic_working_memory_input_ids=semantic,
        working_memory_index=int(round_index) - 1,
        generation_raw_prefix_input_ids=generation_prefix_ids,
        generation_user_suffix_tokens=generation_user_suffix_tokens,
    )


def _finish_generation(
    *,
    tokenizer: Any,
    result: SGLangGeneration,
    eos_token_id: int,
    compression: Mapping[str, Any],
    qwen36_native_tools: bool = False,
    qwen_native_family: Optional[str] = None,
) -> RolloutGeneration:
    if qwen_native_family is not None and qwen_native_family not in {
        "qwen36",
        "qwen38",
    }:
        raise ValueError("qwen_native_family must be qwen36, qwen38, or None")
    qwen_native_tools = qwen36_native_tools or qwen_native_family is not None
    qwen_telemetry_family = qwen_native_family or "qwen36"
    raw_ids = [int(token_id) for token_id in result.token_ids]
    token_ids, eos_reached = trim_at_eos(raw_ids, eos_token_id=eos_token_id)
    finished = bool(result.finished_on_eos) and eos_reached
    body_ids = token_ids[:-1] if eos_reached else token_ids
    generated_text = _decode(tokenizer, token_ids)
    decoded_body = _decode(tokenizer, body_ids)
    assistant_content = ASSISTANT_CONTENT_PREFIX + decoded_body
    generated_qwen_think_open = bool(
        qwen_native_tools and body_ids and body_ids[0] == THINK_OPEN_TOKEN_ID
    )
    qwen_tool_open_without_think_close = bool(
        qwen_native_tools
        and THINK_CLOSE_TOKEN_ID not in body_ids
        and TOOL_OPEN_TOKEN_ID in body_ids
    )
    if qwen_native_tools and (
        generated_qwen_think_open or THINK_CLOSE_TOKEN_ID not in body_ids
    ):
        # The official Qwen reasoning parser strips one optional generated
        # opening.  Do not prepend the prompt's opening a second time when the
        # sampled continuation already contains it.  A completed native
        # message without any explicit think block is also retained verbatim.
        assistant_content = decoded_body
    parser = (
        parse_qwen36_continuation if qwen_native_tools else parse_grepseek_continuation
    )
    parsed = parser(token_ids, tokenizer, eos_token_id=eos_token_id)
    finish_kind = "unknown"
    if isinstance(result.finish_reason, str):
        finish_kind = result.finish_reason.strip().lower() or "unknown"
    elif isinstance(result.finish_reason, Mapping):
        for key in ("type", "reason", "finish_reason"):
            candidate = result.finish_reason.get(key)
            if isinstance(candidate, str) and candidate.strip():
                finish_kind = candidate.strip().lower()
                break
    generation_guard = result.meta_info.get("generation_degeneration_guard")
    guard_telemetry = (
        dict(generation_guard) if isinstance(generation_guard, Mapping) else None
    )
    return RolloutGeneration(
        token_ids=tuple(token_ids),
        generated_text=generated_text,
        assistant_content=assistant_content,
        parsed=parsed,
        finished_on_eos=finished,
        hit_generation_cap=(
            not finished
            and finish_kind in {"length", "max_tokens", "max_new_tokens", "token_limit"}
        ),
        compression={
            **dict(compression),
            **(
                {"generation_degeneration_guard": guard_telemetry}
                if guard_telemetry is not None
                else {}
            ),
            **(
                {
                    f"{qwen_telemetry_family}_generated_think_open_normalized": (
                        generated_qwen_think_open
                    ),
                    f"{qwen_telemetry_family}_tool_open_without_think_close_normalized": (
                        qwen_tool_open_without_think_close
                    ),
                }
                if qwen_native_tools
                else {}
            ),
        },
        finish_reason=result.finish_reason,
    )


class GrepSeekSGLangRolloutGenerator:
    """Fail-closed batch-one SGLang implementation of ``RolloutTurnGenerator``."""

    backend = "sglang"
    device = "remote-sglang"
    collect_diagnostics = True
    context_organization = CAUSAL_OBSERVATION_CACHE_V1
    checkpoint_context_organization = CAUSAL_OBSERVATION_CACHE_V1
    recursive_working_memory_enabled = False
    recursive_working_memory_policy = None

    def __init__(
        self,
        *,
        tokenizer: Any,
        client: GrepSeekSGLangClient,
        compression_ratio: int,
        target_hidden_size: int,
        max_encoder_tokens: Optional[int],
        pad_to_multiple_of: Optional[int],
        max_new_tokens: int,
        eos_token_id: int = EOS_TOKEN_ID,
        server_context_length: Optional[int] = None,
        server_max_total_num_tokens: Optional[int] = None,
        server_max_req_input_len: Optional[int] = None,
        use_remaining_context_for_generation: bool = False,
        checkpoint_schema: str = LEGACY_CHECKPOINT_SCHEMA,
    ) -> None:
        if isinstance(compression_ratio, bool) or int(compression_ratio) <= 0:
            raise ValueError("compression_ratio must be positive")
        if isinstance(target_hidden_size, bool) or int(target_hidden_size) <= 0:
            raise ValueError("target_hidden_size must be positive")
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if max_encoder_tokens is not None and (
            isinstance(max_encoder_tokens, bool) or int(max_encoder_tokens) <= 0
        ):
            raise ValueError("max_encoder_tokens must be positive or None")
        if pad_to_multiple_of is not None and (
            isinstance(pad_to_multiple_of, bool) or int(pad_to_multiple_of) <= 0
        ):
            raise ValueError("pad_to_multiple_of must be positive or None")
        self.tokenizer = tokenizer
        self.client = client
        self.compression_ratio = int(compression_ratio)
        self.checkpoint_schema = str(checkpoint_schema)
        self.target_hidden_size = int(target_hidden_size)
        self.max_encoder_tokens = (
            int(max_encoder_tokens) if max_encoder_tokens is not None else None
        )
        self.pad_to_multiple_of = (
            int(pad_to_multiple_of) if pad_to_multiple_of is not None else None
        )
        self.max_new_tokens = int(max_new_tokens)
        self.eos_token_id = int(eos_token_id)
        self.server_context_length = (
            int(server_context_length) if server_context_length is not None else None
        )
        self.server_max_total_num_tokens = (
            int(server_max_total_num_tokens)
            if server_max_total_num_tokens is not None
            else None
        )
        self.server_max_req_input_len = (
            int(server_max_req_input_len)
            if server_max_req_input_len is not None
            else None
        )
        self.use_remaining_context_for_generation = bool(
            use_remaining_context_for_generation
        )
        self._observation_cache_lock = threading.RLock()
        self._observation_cache_episode_id: Optional[str] = None
        self._observation_memory_cache: dict[int, _CachedObservationMemory] = {}

    def _finish(
        self,
        result: SGLangGeneration,
        compression: Mapping[str, Any],
    ) -> RolloutGeneration:
        return _finish_generation(
            tokenizer=self.tokenizer,
            result=result,
            eos_token_id=self.eos_token_id,
            compression=compression,
        )

    def clear_observation_cache(self) -> None:
        """Drop all per-rollout CPU memory slots."""

        with self._observation_cache_lock:
            self._observation_cache_episode_id = None
            self._observation_memory_cache.clear()

    def _activate_observation_cache(self, episode_id: str, round_index: int) -> None:
        current = str(episode_id)
        if self._observation_cache_episode_id != current or int(round_index) == 1:
            self._observation_memory_cache.clear()
            self._observation_cache_episode_id = current

    @torch.inference_mode()
    def generate_next(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink] = None,
    ) -> RolloutGeneration:
        with self._observation_cache_lock:
            return self._generate_next_locked(
                messages,
                episode_id=episode_id,
                round_index=round_index,
                event_sink=event_sink,
            )

    def _generate_next_locked(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink] = None,
    ) -> RolloutGeneration:
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._activate_observation_cache(episode_id, round_index)
        prompt_text = _render_chat(self.tokenizer, messages, add_generation_prompt=True)
        if not prompt_text.endswith(ASSISTANT_CONTENT_PREFIX):
            raise ValueError(
                "GrepSeek generation prompt no longer ends with '<think>\\n'"
            )
        feature = build_rollout_feature(
            messages,
            self.tokenizer,
            episode_id=episode_id,
            round_index=round_index,
        )
        source_count = int(feature["num_source_segments"])
        source_tokens = int(feature["rollout_source_token_count"])
        latest_raw_observation_round_index = feature.get(
            "latest_raw_observation_round_index"
        )
        latest_raw_observation_token_count = int(
            feature.get("latest_raw_observation_token_count", 0)
        )
        source_observation_round_indices = [
            int(value) for value in feature.get("source_observation_round_indices", [])
        ]
        if len(source_observation_round_indices) != source_count:
            raise ValueError(
                "rollout observation-round mapping disagrees with source count"
            )
        compression_request_seconds = 0.0
        cache_hit_rounds: list[int] = []
        cache_miss_rounds: list[int] = []
        source_encoder_token_lengths: list[int] = []
        observation_encoder_token_lengths: list[int] = []
        observation_history_tokens_dropped: list[int] = []
        new_memory_slots = 0
        reused_memory_slots = 0

        active_source_rounds = set(source_observation_round_indices)
        for cached_round in tuple(self._observation_memory_cache):
            if cached_round not in active_source_rounds:
                del self._observation_memory_cache[cached_round]

        if source_count:
            raw_sources = [
                [int(value) for value in source]
                for source in feature["source_segments_input_ids"]
            ]
            discrete_segments = [
                [int(value) for value in segment]
                for segment in feature["discrete_segments_input_ids"]
            ]
            source_prompt_ids = [
                int(value) for value in feature["source_prompt_input_ids"]
            ]
            evidence_spans = [
                [int(value) for value in span]
                for span in feature["source_prompt_evidence_spans"]
            ]
            if not (
                len(raw_sources)
                == len(evidence_spans)
                == len(source_observation_round_indices)
                == source_count
            ):
                raise ValueError("rollout source/cache contract arity mismatch")
            if len(discrete_segments) != source_count + 1:
                raise ValueError("rollout discrete/source interleave arity mismatch")
            if source_observation_round_indices != sorted(
                set(source_observation_round_indices)
            ):
                raise ValueError(
                    "source observation rounds must be unique and increasing"
                )
            source_lengths = [len(source) for source in raw_sources]
            if any(length <= 0 for length in source_lengths):
                raise ValueError("cached tool observations must be nonempty")
            if sum(source_lengths) != source_tokens:
                raise ValueError("rollout source token total is inconsistent")
            slot_counts = [
                int(math.ceil(length / self.compression_ratio))
                for length in source_lengths
            ]

            memory_segments: list[torch.Tensor] = []
            for segment_index, (
                observation_round,
                source_ids,
                evidence_span,
                slot_count,
            ) in enumerate(
                zip(
                    source_observation_round_indices,
                    raw_sources,
                    evidence_spans,
                    slot_counts,
                )
            ):
                encoder_ids, encoder_span, history_dropped = (
                    _causal_observation_encode_input(
                        source_prompt_ids,
                        evidence_span,
                        max_encoder_tokens=self.max_encoder_tokens,
                    )
                )
                _validate_server_input_length(
                    token_count=len(encoder_ids),
                    server_max_req_input_len=self.server_max_req_input_len,
                    label=f"observation {observation_round} source encoder",
                )
                signature = _observation_signature(
                    encoder_ids, encoder_span, source_ids
                )
                cached = self._observation_memory_cache.get(observation_round)
                if cached is not None and cached.signature != signature:
                    del self._observation_memory_cache[observation_round]
                    cached = None
                if cached is None:
                    compression_started = time.perf_counter()
                    encoded = self.client.encode_memory(
                        encoder_ids,
                        evidence_spans=[encoder_span],
                        source_token_lengths=[len(source_ids)],
                        expected_slot_count=slot_count,
                    )
                    compression_request_seconds += (
                        time.perf_counter() - compression_started
                    )
                    if encoded.shape != (slot_count, self.target_hidden_size):
                        raise RuntimeError(
                            "SGLang live observation memory shape disagrees with "
                            "checkpoint contract: "
                            f"segment={segment_index}, "
                            f"expected={(slot_count, self.target_hidden_size)}, "
                            f"observed={tuple(encoded.shape)}"
                        )
                    if not torch.is_floating_point(encoded) or not bool(
                        torch.isfinite(encoded).all().item()
                    ):
                        raise RuntimeError(
                            "SGLang live observation memory must be finite floating point"
                        )
                    cached = _CachedObservationMemory(
                        signature=signature,
                        embeddings=encoded.detach().cpu().clone(),
                        source_token_length=len(source_ids),
                        source_encoder_tokens=len(encoder_ids),
                        history_tokens_dropped=history_dropped,
                    )
                    self._observation_memory_cache[observation_round] = cached
                    cache_miss_rounds.append(observation_round)
                    source_encoder_token_lengths.append(len(encoder_ids))
                    new_memory_slots += slot_count
                else:
                    cache_hit_rounds.append(observation_round)
                    reused_memory_slots += slot_count
                if cached.source_token_length != len(
                    source_ids
                ) or cached.embeddings.shape != (slot_count, self.target_hidden_size):
                    raise RuntimeError("observation memory cache entry is inconsistent")
                observation_encoder_token_lengths.append(cached.source_encoder_tokens)
                observation_history_tokens_dropped.append(cached.history_tokens_dropped)
                memory_segments.append(cached.embeddings)

            memory = torch.cat(memory_segments, dim=0)
            memory_slots = int(memory.shape[0])
            expected_slots = sum(slot_counts)
            if memory_slots != expected_slots:
                raise RuntimeError("packed cached memory slot count is inconsistent")
            prefix_ids, memory_positions = _pack_cached_memory_prefix(
                discrete_segments,
                slot_counts,
                placeholder_token_id=int(self.tokenizer.pad_token_id),
            )
            if len(memory_positions) != memory_slots:
                raise RuntimeError("recovery positions disagree with cached memory")
            generation_max_new_tokens = _generation_token_budget(
                prefix_token_count=len(prefix_ids),
                configured_max_new_tokens=self.max_new_tokens,
                server_context_length=self.server_context_length,
                server_max_total_num_tokens=self.server_max_total_num_tokens,
                server_max_req_input_len=self.server_max_req_input_len,
                use_remaining_context=self.use_remaining_context_for_generation,
            )
            valid_memory = memory.detach().float()
            source_encoder_tokens = sum(source_encoder_token_lengths)
            student_prefix_tokens = len(prefix_ids)
            memory_shape: Optional[list[int]] = [int(size) for size in memory.shape]
            memory_dtype: Optional[str] = str(memory.dtype).removeprefix("torch.")
            memory_rms = _finite(valid_memory.square().mean().sqrt().item())
            memory_mean_l2 = _finite(valid_memory.norm(dim=-1).mean().item())
        else:
            prefix_ids = [
                int(token_id) for token_id in feature["discrete_segments_input_ids"][0]
            ]
            if not prefix_ids:
                raise ValueError("discrete first-turn prefix must not be empty")
            memory_positions = []
            memory = None
            source_lengths = []
            slot_counts = []
            memory_slots = 0
            source_encoder_tokens = 0
            student_prefix_tokens = len(prefix_ids)
            memory_shape = None
            memory_dtype = None
            memory_rms = None
            memory_mean_l2 = None
            observation_encoder_token_lengths = []
            observation_history_tokens_dropped = []
            generation_max_new_tokens = _generation_token_budget(
                prefix_token_count=len(prefix_ids),
                configured_max_new_tokens=self.max_new_tokens,
                server_context_length=self.server_context_length,
                server_max_total_num_tokens=self.server_max_total_num_tokens,
                server_max_req_input_len=self.server_max_req_input_len,
                use_remaining_context=self.use_remaining_context_for_generation,
            )
        compression: dict[str, Any] = {
            "backend": self.backend,
            "context_organization": self.context_organization,
            "checkpoint_context_organization": (self.checkpoint_context_organization),
            "checkpoint_schema": self.checkpoint_schema,
            "compression_ratio": self.compression_ratio,
            "recursive_working_memory_enabled": (self.recursive_working_memory_enabled),
            "recursive_working_memory_policy": (self.recursive_working_memory_policy),
            "source_segments": source_count,
            "source_tokens": source_tokens,
            "memory_slots": memory_slots,
            "effective_compression_ratio": (
                source_tokens / memory_slots if memory_slots else None
            ),
            "source_encoder_tokens": source_encoder_tokens,
            "source_encoder_token_lengths": source_encoder_token_lengths,
            "student_prefix_tokens": student_prefix_tokens,
            "latest_raw_observation_round_index": latest_raw_observation_round_index,
            "latest_raw_observation_token_count": latest_raw_observation_token_count,
            "observation_compression_policy": feature.get(
                "observation_compression_policy"
            ),
            "observation_count": int(feature.get("observation_count", 0)),
            "source_token_lengths": source_lengths,
            "source_observation_round_indices": source_observation_round_indices,
            "observation_encoder_token_lengths": (observation_encoder_token_lengths),
            "observation_history_tokens_dropped": (observation_history_tokens_dropped),
            "slot_counts": slot_counts,
            "observation_memory_cache_policy": "causal_prefix_per_observation_v1",
            "observation_memory_cache_hits": len(cache_hit_rounds),
            "observation_memory_cache_misses": len(cache_miss_rounds),
            "reused_observation_round_indices": cache_hit_rounds,
            "newly_compressed_observation_round_indices": cache_miss_rounds,
            "reused_memory_slots": reused_memory_slots,
            "new_memory_slots": new_memory_slots,
            "observation_memory_cache_entries": len(self._observation_memory_cache),
            "memory_tensor_shape": memory_shape,
            "valid_memory_shape": [memory_slots, self.target_hidden_size],
            "memory_dtype": memory_dtype,
            "valid_memory_rms": memory_rms,
            "valid_memory_mean_l2": memory_mean_l2,
            # SGLang performs source target-forward and compressor-forward in
            # one remote request, so reporting separate timings would be fake.
            "source_encode_seconds": None,
            "compressor_seconds": None,
            "source_compression_request_seconds": _finite(compression_request_seconds),
            "target_generation_seconds": None,
            "server_context_length": self.server_context_length,
            "server_max_total_num_tokens": self.server_max_total_num_tokens,
            "server_max_req_input_len": self.server_max_req_input_len,
            "generation_max_new_tokens": generation_max_new_tokens,
            "generation_budget_policy": (
                "remaining_context_after_input"
                if self.use_remaining_context_for_generation
                else "configured_cap_clamped_to_remaining_context"
            ),
        }
        _emit_rollout_event(
            event_sink,
            event="compression_finished",
            episode_id=episode_id,
            round_index=round_index,
            payload={"compression": dict(compression)},
        )

        generation_started = time.perf_counter()
        if source_count:
            assert memory is not None
            result = self.client.generate_with_memory(
                prefix_ids,
                memory_positions=memory_positions,
                memory_embeddings=memory,
                max_new_tokens=generation_max_new_tokens,
            )
        else:
            result = self.client.generate_discrete(
                prefix_ids, max_new_tokens=generation_max_new_tokens
            )
        compression["target_generation_seconds"] = _finite(
            time.perf_counter() - generation_started
        )
        return self._finish(result, compression)


class GrepSeekSGLangRecursiveRolloutGenerator:
    """SGLang implementation of the recursive-history-v2 serving ABI."""

    backend = "sglang"
    device = "remote-sglang"
    collect_diagnostics = True
    context_organization = RECURSIVE_HISTORY_V2
    checkpoint_context_organization = RECURSIVE_HISTORY_V2
    recursive_working_memory_enabled = False
    recursive_working_memory_policy = None
    recursive_state_policy = "replace_previous_memory_each_transition_v1"

    def __init__(
        self,
        *,
        tokenizer: Any,
        client: GrepSeekSGLangClient,
        compression_ratio: int,
        target_hidden_size: int,
        max_encoder_tokens: Optional[int],
        max_new_tokens: int,
        checkpoint_schema: str,
        eos_token_id: int = EOS_TOKEN_ID,
        server_context_length: Optional[int] = None,
        server_max_total_num_tokens: Optional[int] = None,
        server_max_req_input_len: Optional[int] = None,
        use_remaining_context_for_generation: bool = False,
    ) -> None:
        if int(compression_ratio) != 128:
            raise ValueError("recursive_history_v2 requires compression_ratio=128")
        if isinstance(target_hidden_size, bool) or int(target_hidden_size) <= 0:
            raise ValueError("target_hidden_size must be positive")
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if max_encoder_tokens is None or (
            isinstance(max_encoder_tokens, bool) or int(max_encoder_tokens) <= 0
        ):
            raise ValueError(
                "recursive_history_v2 requires a positive max_encoder_tokens cap"
            )
        if not str(checkpoint_schema):
            raise ValueError("checkpoint_schema must be nonempty")
        self.tokenizer = tokenizer
        self.client = client
        self.compression_ratio = int(compression_ratio)
        self.target_hidden_size = int(target_hidden_size)
        self.max_encoder_tokens = int(max_encoder_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.checkpoint_schema = str(checkpoint_schema)
        self.eos_token_id = int(eos_token_id)
        self.server_context_length = (
            int(server_context_length) if server_context_length is not None else None
        )
        self.server_max_total_num_tokens = (
            int(server_max_total_num_tokens)
            if server_max_total_num_tokens is not None
            else None
        )
        self.server_max_req_input_len = (
            int(server_max_req_input_len)
            if server_max_req_input_len is not None
            else None
        )
        self.use_remaining_context_for_generation = bool(
            use_remaining_context_for_generation
        )
        self._state_lock = threading.RLock()
        self._episode_id: Optional[str] = None
        self._raw_prefix_input_ids: Optional[tuple[int, ...]] = None
        self._memory: Optional[torch.Tensor] = None
        self._state_transition_index: Optional[int] = None

    def _finish(
        self,
        result: SGLangGeneration,
        compression: Mapping[str, Any],
    ) -> RolloutGeneration:
        return _finish_generation(
            tokenizer=self.tokenizer,
            result=result,
            eos_token_id=self.eos_token_id,
            compression=compression,
        )

    def clear_recursive_state(self) -> None:
        with self._state_lock:
            self._episode_id = None
            self._raw_prefix_input_ids = None
            self._memory = None
            self._state_transition_index = None

    def _activate_episode(self, episode_id: str, round_index: int) -> None:
        current = str(episode_id)
        if self._episode_id != current or int(round_index) == 1:
            self._episode_id = current
            self._raw_prefix_input_ids = None
            self._memory = None
            self._state_transition_index = None

    @torch.inference_mode()
    def generate_next(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink] = None,
    ) -> RolloutGeneration:
        with self._state_lock:
            return self._generate_next_locked(
                messages,
                episode_id=episode_id,
                round_index=round_index,
                event_sink=event_sink,
            )

    def _generate_next_locked(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink],
    ) -> RolloutGeneration:
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._activate_episode(episode_id, round_index)
        prompt_text = _render_chat(self.tokenizer, messages, add_generation_prompt=True)
        if not prompt_text.endswith(ASSISTANT_CONTENT_PREFIX):
            raise ValueError(
                "GrepSeek generation prompt no longer ends with '<think>\\n'"
            )
        turn = _build_recursive_turn_input(
            messages,
            self.tokenizer,
            episode_id=episode_id,
            round_index=round_index,
        )
        if self._raw_prefix_input_ids is None:
            self._raw_prefix_input_ids = turn.raw_prefix_input_ids
        elif self._raw_prefix_input_ids != turn.raw_prefix_input_ids:
            raise RuntimeError(
                "recursive raw system/user prefix changed within episode"
            )
        raw_prefix = list(turn.raw_prefix_input_ids)
        boundary = [int(value) for value in GENERATION_BOUNDARY_IDS]
        raw_prefix_tokens = len(raw_prefix)
        transition_index = turn.transition_index
        compression_started: Optional[float] = None

        if transition_index is None:
            if self._memory is not None or self._state_transition_index is not None:
                raise RuntimeError("recursive first turn unexpectedly retained memory")
            previous_memory_tokens = 0
            discrete_suffix_tokens = 0
            physical_source_tokens = 0
            expected_memory_slots = 0
            memory_slots = 0
            source_encoder_tokens = 0
            prefix_ids = raw_prefix + boundary
            memory_positions: list[int] = []
            memory: Optional[torch.Tensor] = None
            source_compression_seconds = 0.0
            state_replaced = False
        else:
            if turn.incremental_suffix_input_ids is None:
                raise AssertionError("recursive transition has no incremental suffix")
            if transition_index == 1:
                if self._memory is not None or self._state_transition_index is not None:
                    raise RuntimeError("recursive transition 1 was already applied")
                previous_memory = None
            else:
                if (
                    self._memory is None
                    or self._state_transition_index != transition_index - 1
                ):
                    raise RuntimeError(
                        "recursive transition requires the immediately previous memory state"
                    )
                previous_memory = self._memory
            previous_memory_tokens = (
                0 if previous_memory is None else int(previous_memory.shape[0])
            )
            suffix = [int(value) for value in turn.incremental_suffix_input_ids]
            discrete_suffix_tokens = len(suffix)
            physical_source_tokens = previous_memory_tokens + discrete_suffix_tokens
            expected_memory_slots = int(
                math.ceil(physical_source_tokens / self.compression_ratio)
            )
            encoder_ids = list(raw_prefix)
            previous_positions: Optional[list[int]] = None
            if previous_memory is not None:
                previous_positions = list(
                    range(raw_prefix_tokens, raw_prefix_tokens + previous_memory_tokens)
                )
                encoder_ids.extend(
                    [int(self.tokenizer.pad_token_id)] * previous_memory_tokens
                )
            encoder_ids.extend(suffix)
            source_encoder_tokens = len(encoder_ids)
            if source_encoder_tokens > self.max_encoder_tokens:
                raise ValueError(
                    "recursive mixed source exceeds the trained encoder cap: "
                    f"P={raw_prefix_tokens}, M_prev={previous_memory_tokens}, "
                    f"suffix={discrete_suffix_tokens}, total={source_encoder_tokens}, "
                    f"max_encoder_tokens={self.max_encoder_tokens}; refusing to crop"
                )
            _validate_server_input_length(
                token_count=source_encoder_tokens,
                server_max_req_input_len=self.server_max_req_input_len,
                label=f"recursive transition {transition_index} source encoder",
            )
            compression_started = time.perf_counter()
            memory = self.client.encode_memory(
                encoder_ids,
                evidence_spans=[[raw_prefix_tokens, source_encoder_tokens]],
                source_token_lengths=[physical_source_tokens],
                expected_slot_count=expected_memory_slots,
                recursive=True,
                memory_positions=previous_positions,
                memory_embeddings=previous_memory,
            )
            source_compression_seconds = time.perf_counter() - compression_started
            if memory.shape != (expected_memory_slots, self.target_hidden_size):
                raise RuntimeError(
                    "SGLang recursive memory shape disagrees with checkpoint contract: "
                    f"expected={(expected_memory_slots, self.target_hidden_size)}, "
                    f"observed={tuple(memory.shape)}"
                )
            if not torch.is_floating_point(memory) or not bool(
                torch.isfinite(memory).all().item()
            ):
                raise RuntimeError(
                    "SGLang recursive memory must be finite floating point"
                )
            memory = memory.detach().cpu().clone()
            memory_slots = int(memory.shape[0])
            prefix_ids = list(raw_prefix)
            memory_positions = list(
                range(raw_prefix_tokens, raw_prefix_tokens + memory_slots)
            )
            prefix_ids.extend([int(self.tokenizer.pad_token_id)] * memory_slots)
            prefix_ids.extend(boundary)
            state_replaced = True

        generation_max_new_tokens = _generation_token_budget(
            prefix_token_count=len(prefix_ids),
            configured_max_new_tokens=self.max_new_tokens,
            server_context_length=self.server_context_length,
            server_max_total_num_tokens=self.server_max_total_num_tokens,
            server_max_req_input_len=self.server_max_req_input_len,
            use_remaining_context=self.use_remaining_context_for_generation,
        )
        valid_memory = memory.detach().float() if memory is not None else None
        compression: dict[str, Any] = {
            "backend": self.backend,
            "context_organization": self.context_organization,
            "checkpoint_context_organization": (self.checkpoint_context_organization),
            "checkpoint_schema": self.checkpoint_schema,
            "compression_ratio": self.compression_ratio,
            "recursive_working_memory_enabled": (self.recursive_working_memory_enabled),
            "recursive_working_memory_policy": (self.recursive_working_memory_policy),
            "transition_index": transition_index,
            "raw_prefix_tokens": raw_prefix_tokens,
            "previous_memory_tokens": previous_memory_tokens,
            "discrete_suffix_tokens": discrete_suffix_tokens,
            "physical_source_tokens": physical_source_tokens,
            "expected_memory_slots": expected_memory_slots,
            "memory_slots": memory_slots,
            "state_replaced": state_replaced,
            "student_prefix_tokens": len(prefix_ids),
            "generation_boundary_tokens": len(boundary),
            "recursive_state_policy": self.recursive_state_policy,
            "source_segments": 0 if transition_index is None else 1,
            "source_tokens": physical_source_tokens,
            "effective_compression_ratio": (
                physical_source_tokens / memory_slots if memory_slots else None
            ),
            "source_encoder_tokens": source_encoder_tokens,
            "source_encoder_token_lengths": (
                [] if transition_index is None else [source_encoder_tokens]
            ),
            "source_token_lengths": (
                [] if transition_index is None else [physical_source_tokens]
            ),
            "slot_counts": [] if transition_index is None else [memory_slots],
            "memory_tensor_shape": (
                None if memory is None else [int(size) for size in memory.shape]
            ),
            "valid_memory_shape": [memory_slots, self.target_hidden_size],
            "memory_dtype": (
                None if memory is None else str(memory.dtype).removeprefix("torch.")
            ),
            "valid_memory_rms": (
                None
                if valid_memory is None
                else _finite(valid_memory.square().mean().sqrt().item())
            ),
            "valid_memory_mean_l2": (
                None
                if valid_memory is None
                else _finite(valid_memory.norm(dim=-1).mean().item())
            ),
            "source_encode_seconds": None,
            "compressor_seconds": None,
            "source_compression_request_seconds": _finite(source_compression_seconds),
            "target_generation_seconds": None,
            "max_encoder_tokens": self.max_encoder_tokens,
            "server_context_length": self.server_context_length,
            "server_max_total_num_tokens": self.server_max_total_num_tokens,
            "server_max_req_input_len": self.server_max_req_input_len,
            "generation_max_new_tokens": generation_max_new_tokens,
            "generation_budget_policy": (
                "remaining_context_after_input"
                if self.use_remaining_context_for_generation
                else "configured_cap_clamped_to_remaining_context"
            ),
        }
        _emit_rollout_event(
            event_sink,
            event="compression_finished",
            episode_id=episode_id,
            round_index=round_index,
            payload={"compression": dict(compression)},
        )

        generation_started = time.perf_counter()
        if memory is None:
            result = self.client.generate_discrete(
                prefix_ids, max_new_tokens=generation_max_new_tokens
            )
        else:
            result = self.client.generate_with_memory(
                prefix_ids,
                memory_positions=memory_positions,
                memory_embeddings=memory,
                max_new_tokens=generation_max_new_tokens,
            )
        compression["target_generation_seconds"] = _finite(
            time.perf_counter() - generation_started
        )
        if transition_index is not None:
            assert memory is not None
            self._memory = memory
            self._state_transition_index = transition_index
        return self._finish(result, compression)


class GrepSeekSGLangRecursiveWorkingMemoryRolloutGenerator(
    GrepSeekSGLangRecursiveRolloutGenerator
):
    """Experimental recursive inference with the newest tool exchange kept raw.

    The checkpoint remains native ``recursive_history_v2``.  This inference-only
    ablation delays each recurrent transition by one turn: ``M_k`` summarizes
    history only through ``W_k``, while the newest ``W_{k+1}`` is passed raw to
    recovery.  Consequently the raw working memory is never also present inside
    the memory tensor used by the same generation request.
    """

    context_organization = RECURSIVE_HISTORY_WITH_RAW_WORKING_MEMORY_ABLATION_V1
    checkpoint_context_organization = RECURSIVE_HISTORY_V2
    recursive_working_memory_enabled = True
    recursive_working_memory_policy = LATEST_TOOL_EXCHANGE_RAW_V1
    recursive_state_policy = "replace_previous_memory_keep_latest_tool_exchange_raw_v1"

    def __init__(self, **kwargs: Any) -> None:
        max_encoder_tokens = kwargs.get("max_encoder_tokens")
        if isinstance(max_encoder_tokens, bool) or max_encoder_tokens != 8192:
            raise ValueError(
                "recursive working-memory ablation requires " "max_encoder_tokens=8192"
            )
        super().__init__(**kwargs)
        self._latest_raw_working_memory: Optional[tuple[int, ...]] = None
        self._latest_raw_working_memory_index: Optional[int] = None

    def clear_recursive_state(self) -> None:
        with self._state_lock:
            super().clear_recursive_state()
            self._latest_raw_working_memory = None
            self._latest_raw_working_memory_index = None

    def _activate_episode(self, episode_id: str, round_index: int) -> None:
        current = str(episode_id)
        reset = self._episode_id != current or int(round_index) == 1
        super()._activate_episode(current, round_index)
        if reset:
            self._latest_raw_working_memory = None
            self._latest_raw_working_memory_index = None

    def _generate_next_locked(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink],
    ) -> RolloutGeneration:
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._activate_episode(episode_id, round_index)
        prompt_text = _render_chat(self.tokenizer, messages, add_generation_prompt=True)
        if not prompt_text.endswith(ASSISTANT_CONTENT_PREFIX):
            raise ValueError(
                "GrepSeek generation prompt no longer ends with '<think>\\n'"
            )
        turn = _build_recursive_turn_input(
            messages,
            self.tokenizer,
            episode_id=episode_id,
            round_index=round_index,
        )
        if self._raw_prefix_input_ids is None:
            self._raw_prefix_input_ids = turn.raw_prefix_input_ids
        elif self._raw_prefix_input_ids != turn.raw_prefix_input_ids:
            raise RuntimeError(
                "recursive raw system/user prefix changed within episode"
            )

        raw_prefix = list(turn.raw_prefix_input_ids)
        boundary = [int(value) for value in GENERATION_BOUNDARY_IDS]
        raw_prefix_tokens = len(raw_prefix)
        current_working_memory = (
            None
            if turn.incremental_suffix_input_ids is None
            else tuple(int(value) for value in turn.incremental_suffix_input_ids)
        )
        current_working_memory_index = turn.transition_index
        if current_working_memory is not None and (
            len(current_working_memory) <= len(boundary)
            or list(current_working_memory[: len(boundary)]) != boundary
        ):
            raise ValueError("raw working memory lost its native G7 boundary")

        transition_index: Optional[int] = None
        compressed_working_memory_index: Optional[int] = None
        compressed_working_memory_tokens = 0
        previous_memory_tokens = 0
        discrete_suffix_tokens = 0
        physical_source_tokens = 0
        expected_memory_slots = 0
        memory_slots = 0
        source_encoder_tokens = 0
        source_compression_seconds = 0.0
        memory_positions: list[int] = []
        memory: Optional[torch.Tensor] = None
        state_replaced = False

        if current_working_memory_index is None:
            if int(round_index) != 1 or current_working_memory is not None:
                raise RuntimeError(
                    "working-memory first turn has an inconsistent raw suffix"
                )
            if (
                self._memory is not None
                or self._state_transition_index is not None
                or self._latest_raw_working_memory is not None
                or self._latest_raw_working_memory_index is not None
            ):
                raise RuntimeError(
                    "working-memory first turn unexpectedly retained episode state"
                )
            prefix_ids = raw_prefix + boundary
        elif int(round_index) == 2:
            if current_working_memory_index != 1 or current_working_memory is None:
                raise RuntimeError("working-memory round 2 requires exactly raw W1")
            if (
                self._memory is not None
                or self._state_transition_index is not None
                or self._latest_raw_working_memory is not None
                or self._latest_raw_working_memory_index is not None
            ):
                raise RuntimeError(
                    "working-memory round 2 unexpectedly retained prior state"
                )
            # W1 is deliberately not compressed.  It remains the sole raw
            # working-memory exchange in the recovery prefix.
            prefix_ids = raw_prefix + list(current_working_memory) + boundary
        else:
            if current_working_memory is None:
                raise AssertionError("working-memory transition has no current W")
            expected_current_index = int(round_index) - 1
            if current_working_memory_index != expected_current_index:
                raise RuntimeError(
                    "working-memory current suffix is not the newest completed "
                    "assistant/tool exchange"
                )
            transition_index = expected_current_index - 1
            if (
                self._latest_raw_working_memory is None
                or self._latest_raw_working_memory_index != transition_index
            ):
                raise RuntimeError(
                    "working-memory transition requires the immediately previous "
                    "raw tool exchange"
                )
            previous_memory: Optional[torch.Tensor]
            if transition_index == 1:
                if self._memory is not None or self._state_transition_index is not None:
                    raise RuntimeError(
                        "working-memory transition 1 unexpectedly retained memory"
                    )
                previous_memory = None
            else:
                if (
                    self._memory is None
                    or self._state_transition_index != transition_index - 1
                ):
                    raise RuntimeError(
                        "working-memory transition requires the immediately "
                        "previous recursive memory state"
                    )
                previous_memory = self._memory

            previous_memory_tokens = (
                0 if previous_memory is None else int(previous_memory.shape[0])
            )
            previous_raw = list(self._latest_raw_working_memory)
            compressed_working_memory_index = transition_index
            compressed_working_memory_tokens = len(previous_raw)
            discrete_suffix_tokens = compressed_working_memory_tokens
            physical_source_tokens = (
                previous_memory_tokens + compressed_working_memory_tokens
            )
            expected_memory_slots = int(
                math.ceil(physical_source_tokens / self.compression_ratio)
            )
            encoder_ids = list(raw_prefix)
            previous_positions: Optional[list[int]] = None
            if previous_memory is not None:
                previous_positions = list(
                    range(
                        raw_prefix_tokens,
                        raw_prefix_tokens + previous_memory_tokens,
                    )
                )
                encoder_ids.extend(
                    [int(self.tokenizer.pad_token_id)] * previous_memory_tokens
                )
            encoder_ids.extend(previous_raw)
            source_encoder_tokens = len(encoder_ids)
            if source_encoder_tokens > self.max_encoder_tokens:
                raise ValueError(
                    "recursive working-memory source exceeds the trained encoder "
                    "cap: "
                    f"P={raw_prefix_tokens}, M_prev={previous_memory_tokens}, "
                    f"W_prev={compressed_working_memory_tokens}, "
                    f"total={source_encoder_tokens}, "
                    f"max_encoder_tokens={self.max_encoder_tokens}; refusing to crop"
                )
            _validate_server_input_length(
                token_count=source_encoder_tokens,
                server_max_req_input_len=self.server_max_req_input_len,
                label=(
                    f"recursive working-memory transition {transition_index} "
                    "source encoder"
                ),
            )
            compression_started = time.perf_counter()
            memory = self.client.encode_memory(
                encoder_ids,
                evidence_spans=[[raw_prefix_tokens, source_encoder_tokens]],
                source_token_lengths=[physical_source_tokens],
                expected_slot_count=expected_memory_slots,
                recursive=True,
                memory_positions=previous_positions,
                memory_embeddings=previous_memory,
            )
            source_compression_seconds = time.perf_counter() - compression_started
            if memory.shape != (expected_memory_slots, self.target_hidden_size):
                raise RuntimeError(
                    "SGLang recursive working memory shape disagrees with "
                    "checkpoint contract: "
                    f"expected={(expected_memory_slots, self.target_hidden_size)}, "
                    f"observed={tuple(memory.shape)}"
                )
            if not torch.is_floating_point(memory) or not bool(
                torch.isfinite(memory).all().item()
            ):
                raise RuntimeError(
                    "SGLang recursive working memory must be finite floating point"
                )
            memory = memory.detach().cpu().clone()
            memory_slots = int(memory.shape[0])
            prefix_ids = list(raw_prefix)
            memory_positions = list(
                range(raw_prefix_tokens, raw_prefix_tokens + memory_slots)
            )
            prefix_ids.extend([int(self.tokenizer.pad_token_id)] * memory_slots)
            # The current W is raw and appears exactly once after the new M.
            prefix_ids.extend(current_working_memory)
            prefix_ids.extend(boundary)
            state_replaced = True

        generation_max_new_tokens = _generation_token_budget(
            prefix_token_count=len(prefix_ids),
            configured_max_new_tokens=self.max_new_tokens,
            server_context_length=self.server_context_length,
            server_max_total_num_tokens=self.server_max_total_num_tokens,
            server_max_req_input_len=self.server_max_req_input_len,
            use_remaining_context=self.use_remaining_context_for_generation,
        )
        valid_memory = memory.detach().float() if memory is not None else None
        latest_raw_tokens = (
            0 if current_working_memory is None else len(current_working_memory)
        )
        compression: dict[str, Any] = {
            "backend": self.backend,
            "context_organization": self.context_organization,
            "checkpoint_context_organization": (self.checkpoint_context_organization),
            "checkpoint_schema": self.checkpoint_schema,
            "compression_ratio": self.compression_ratio,
            "recursive_working_memory_enabled": (self.recursive_working_memory_enabled),
            "recursive_working_memory_policy": (self.recursive_working_memory_policy),
            "transition_index": transition_index,
            "raw_prefix_tokens": raw_prefix_tokens,
            "previous_memory_tokens": previous_memory_tokens,
            "discrete_suffix_tokens": discrete_suffix_tokens,
            "physical_source_tokens": physical_source_tokens,
            "expected_memory_slots": expected_memory_slots,
            "memory_slots": memory_slots,
            "state_replaced": state_replaced,
            "student_prefix_tokens": len(prefix_ids),
            "generation_boundary_tokens": len(boundary),
            "recursive_state_policy": self.recursive_state_policy,
            "compressed_working_memory_index": (compressed_working_memory_index),
            "compressed_working_memory_tokens": (compressed_working_memory_tokens),
            "latest_raw_working_memory_index": current_working_memory_index,
            "latest_raw_working_memory_tokens": latest_raw_tokens,
            "source_segments": 0 if transition_index is None else 1,
            "source_tokens": physical_source_tokens,
            "effective_compression_ratio": (
                physical_source_tokens / memory_slots if memory_slots else None
            ),
            "source_encoder_tokens": source_encoder_tokens,
            "source_encoder_token_lengths": (
                [] if transition_index is None else [source_encoder_tokens]
            ),
            "source_token_lengths": (
                [] if transition_index is None else [physical_source_tokens]
            ),
            "slot_counts": [] if transition_index is None else [memory_slots],
            "memory_tensor_shape": (
                None if memory is None else [int(size) for size in memory.shape]
            ),
            "valid_memory_shape": [memory_slots, self.target_hidden_size],
            "memory_dtype": (
                None if memory is None else str(memory.dtype).removeprefix("torch.")
            ),
            "valid_memory_rms": (
                None
                if valid_memory is None
                else _finite(valid_memory.square().mean().sqrt().item())
            ),
            "valid_memory_mean_l2": (
                None
                if valid_memory is None
                else _finite(valid_memory.norm(dim=-1).mean().item())
            ),
            "source_encode_seconds": None,
            "compressor_seconds": None,
            "source_compression_request_seconds": _finite(source_compression_seconds),
            "target_generation_seconds": None,
            "max_encoder_tokens": self.max_encoder_tokens,
            "server_context_length": self.server_context_length,
            "server_max_total_num_tokens": self.server_max_total_num_tokens,
            "server_max_req_input_len": self.server_max_req_input_len,
            "generation_max_new_tokens": generation_max_new_tokens,
            "generation_budget_policy": (
                "remaining_context_after_input"
                if self.use_remaining_context_for_generation
                else "configured_cap_clamped_to_remaining_context"
            ),
        }
        _emit_rollout_event(
            event_sink,
            event="compression_finished",
            episode_id=episode_id,
            round_index=round_index,
            payload={"compression": dict(compression)},
        )

        generation_started = time.perf_counter()
        if memory is None:
            result = self.client.generate_discrete(
                prefix_ids, max_new_tokens=generation_max_new_tokens
            )
        else:
            result = self.client.generate_with_memory(
                prefix_ids,
                memory_positions=memory_positions,
                memory_embeddings=memory,
                max_new_tokens=generation_max_new_tokens,
            )
        compression["target_generation_seconds"] = _finite(
            time.perf_counter() - generation_started
        )

        # Finish all local decoding/validation before committing state.  Both a
        # failed SGLang request and a local finish failure can therefore be
        # retried without skipping or partially applying a transition.
        finished = self._finish(result, compression)
        if transition_index is not None:
            assert memory is not None
            self._memory = memory
            self._state_transition_index = transition_index
        if current_working_memory_index is not None:
            assert current_working_memory is not None
            self._latest_raw_working_memory = current_working_memory
            self._latest_raw_working_memory_index = current_working_memory_index
        return finished


class GrepSeekSGLangWorkingMemoryV3RolloutGenerator(
    GrepSeekSGLangRecursiveRolloutGenerator
):
    """Checkpoint-native r16 P/M/semantic-W/raw-W/G5 serving recurrence."""

    context_organization = RECURSIVE_WORKING_MEMORY_V3
    checkpoint_context_organization = RECURSIVE_WORKING_MEMORY_V3
    recursive_working_memory_enabled = True
    recursive_working_memory_policy = ONE_ROUND_LAG_RECURSIVE_WORKING_MEMORY_V3
    recursive_state_policy = REPLACE_PREVIOUS_MEMORY_KEEP_LATEST_RAW_WORKING_MEMORY_V3
    recursive_source_contract = WORKING_MEMORY_V3_SOURCE_CONTRACT

    def __init__(
        self,
        *,
        tokenizer: Any,
        client: GrepSeekSGLangClient,
        compression_ratio: int,
        target_hidden_size: int,
        max_encoder_tokens: Optional[int],
        max_new_tokens: int,
        checkpoint_schema: str,
        eos_token_id: int = EOS_TOKEN_ID,
        server_context_length: Optional[int] = None,
        server_max_total_num_tokens: Optional[int] = None,
        server_max_req_input_len: Optional[int] = None,
        use_remaining_context_for_generation: bool = False,
    ) -> None:
        if int(compression_ratio) != 16:
            raise ValueError(
                "recursive_working_memory_v3 requires compression_ratio=16"
            )
        if isinstance(target_hidden_size, bool) or int(target_hidden_size) <= 0:
            raise ValueError("target_hidden_size must be positive")
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if isinstance(max_encoder_tokens, bool) or max_encoder_tokens != 8192:
            raise ValueError(
                "recursive_working_memory_v3 requires max_encoder_tokens=8192"
            )
        if str(checkpoint_schema) not in WORKING_MEMORY_CHECKPOINT_SCHEMAS:
            raise ValueError(
                "recursive_working_memory_v3 requires a verified checkpoint-native schema"
            )
        self.tokenizer = tokenizer
        self.client = client
        self.compression_ratio = 16
        self.target_hidden_size = int(target_hidden_size)
        self.max_encoder_tokens = 8192
        self.max_new_tokens = int(max_new_tokens)
        self.checkpoint_schema = str(checkpoint_schema)
        self.eos_token_id = int(eos_token_id)
        self.server_context_length = (
            int(server_context_length) if server_context_length is not None else None
        )
        self.server_max_total_num_tokens = (
            int(server_max_total_num_tokens)
            if server_max_total_num_tokens is not None
            else None
        )
        self.server_max_req_input_len = (
            int(server_max_req_input_len)
            if server_max_req_input_len is not None
            else None
        )
        self.use_remaining_context_for_generation = bool(
            use_remaining_context_for_generation
        )
        self._state_lock = threading.RLock()
        self._episode_id: Optional[str] = None
        self._raw_prefix_input_ids: Optional[tuple[int, ...]] = None
        self._memory: Optional[torch.Tensor] = None
        self._state_transition_index: Optional[int] = None
        self._latest_raw_working_memory: Optional[tuple[int, ...]] = None
        self._latest_semantic_working_memory: Optional[tuple[int, ...]] = None
        self._latest_working_memory_index: Optional[int] = None
        self._working_memory_turns = 1
        self._working_memory_tokens: Optional[int] = None
        self._compression_trigger_tokens: Optional[int] = None
        self._max_turns: Optional[int] = None
        self._working_memory_window: list[
            tuple[int, tuple[int, ...], tuple[int, ...]]
        ] = []

    def configure_working_memory_turns(self, value: int) -> None:
        """Select the request-local raw-W window before round one starts."""

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_WORKING_MEMORY_TURNS
        ):
            raise ValueError(
                "working_memory_turns must be between 1 and "
                f"{MAX_WORKING_MEMORY_TURNS}"
            )
        with self._state_lock:
            self._working_memory_turns = int(value)
            self._working_memory_tokens = None
            self._compression_trigger_tokens = None

    def configure_working_memory_tokens(
        self, *, working_memory_tokens: int, compression_trigger_tokens: int
    ) -> None:
        """Configure protected raw-W and left-context compaction for a request.

        The newest complete exchanges within ``working_memory_tokens`` form a
        protected raw suffix R.  Only ``M + L`` (the unprotected raw prefix)
        participates in ``compression_trigger_tokens``.
        """

        working_memory_tokens, compression_trigger_tokens = (
            validate_working_memory_token_budget(
                working_memory_tokens, compression_trigger_tokens
            )
        )
        with self._state_lock:
            self._working_memory_tokens = int(working_memory_tokens)
            self._compression_trigger_tokens = int(compression_trigger_tokens)

    def configure_max_turns(self, value: Optional[int]) -> None:
        """Configure dynamic generation-only guidance for this request."""

        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_WORKING_MEMORY_TURNS
        ):
            raise ValueError(
                f"max_turns must be between 1 and {MAX_WORKING_MEMORY_TURNS} or None"
            )
        with self._state_lock:
            self._max_turns = value

    def prepare_tool_result_for_next_round(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        tool_result: ToolResult,
        episode_id: str,
        round_index: int,
    ) -> tuple[ToolResult, dict[str, Any]]:
        """Fit one new W to its future recursive semantic-source budget.

        Triggering compression after a tool result is too late when that one
        indivisible exchange is itself wider than ``8192 - P - M_prev``.  This
        admission step runs before the released tool payload is appended to
        history.  It preserves as much stdout as possible, then stderr, and
        verifies the exact native/canonical tokenization used by training.
        """

        if not isinstance(tool_result, ToolResult):
            raise TypeError("tool_result must be ToolResult")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index <= 0
        ):
            raise ValueError("round_index must be positive")

        with self._state_lock:
            if self._episode_id != str(episode_id):
                raise RuntimeError(
                    "working-memory-v3 tool-result admission has no active episode"
                )
            if self._raw_prefix_input_ids is None:
                raise RuntimeError(
                    "working-memory-v3 tool-result admission has no persistent P"
                )
            if not messages or str(messages[-1].get("role", "")) != "assistant":
                raise ValueError(
                    "working-memory-v3 tool-result admission requires the current "
                    "assistant message"
                )

            current_memory_tokens = (
                0 if self._memory is None else int(self._memory.shape[0])
            )
            # Before this new W can be evicted, every currently retained older
            # exchange must be consumed in FIFO order.  Project that exact slot
            # recurrence so the admission check remains safe for windows > 1.
            projected_previous_memory_tokens = current_memory_tokens
            for _index, _raw, semantic in self._working_memory_window:
                projected_previous_memory_tokens = math.ceil(
                    (projected_previous_memory_tokens + len(semantic))
                    / self.compression_ratio
                )
            raw_prefix_tokens = len(self._raw_prefix_input_ids)
            semantic_capacity = (
                self.max_encoder_tokens
                - raw_prefix_tokens
                - projected_previous_memory_tokens
            )
            if semantic_capacity <= 0:
                raise ValueError(
                    "working-memory-v3 has no capacity for a new semantic W: "
                    f"P={raw_prefix_tokens}, "
                    f"projected_M_prev={projected_previous_memory_tokens}, "
                    f"max_encoder_tokens={self.max_encoder_tokens}"
                )

            def build_turn(candidate: ToolResult) -> _WorkingMemoryV3TurnInput:
                candidate_messages = [
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                    }
                    for message in messages
                ]
                candidate_messages.append(
                    {"role": "tool", "content": candidate.to_payload()}
                )
                turn = _build_working_memory_v3_turn_input(
                    candidate_messages,
                    self.tokenizer,
                    episode_id=str(episode_id),
                    round_index=int(round_index) + 1,
                    max_turns=self._max_turns,
                )
                if turn.raw_prefix_input_ids != self._raw_prefix_input_ids:
                    raise RuntimeError(
                        "working-memory-v3 tool-result admission changed P"
                    )
                if turn.working_memory_index != int(round_index):
                    raise RuntimeError(
                        "working-memory-v3 tool-result admission index drifted"
                    )
                if turn.latest_semantic_working_memory_input_ids is None:
                    raise RuntimeError(
                        "working-memory-v3 tool-result admission lost semantic W"
                    )
                return turn

            def replace_streams(stdout: str, stderr: str) -> ToolResult:
                return replace(
                    tool_result,
                    stdout=stdout,
                    stderr=stderr,
                    truncated=True,
                    information_lines=[
                        line for line in stdout.splitlines() if line.strip()
                    ],
                )

            full_turn = build_turn(tool_result)
            full_semantic = full_turn.latest_semantic_working_memory_input_ids
            assert full_semantic is not None
            semantic_tokens_before = len(full_semantic)
            raw_tokens_before = len(full_turn.latest_raw_working_memory_input_ids or ())
            generation_prefix_tokens = len(
                full_turn.generation_raw_prefix_input_ids
                or full_turn.raw_prefix_input_ids
            )
            protected_raw_capacity: Optional[int] = None
            if self._working_memory_tokens is not None:
                protected_raw_capacity = min(
                    self._working_memory_tokens,
                    self.max_encoder_tokens
                    - generation_prefix_tokens
                    - projected_previous_memory_tokens
                    - len(ASSISTANT_OPENING_IDS)
                    - WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS,
                )
                if protected_raw_capacity <= 0:
                    raise ValueError(
                        "working-memory-v3 has no capacity for the protected "
                        "latest raw W: "
                        f"generation_P={generation_prefix_tokens}, "
                        f"projected_M={projected_previous_memory_tokens}, "
                        f"configured_working_memory_tokens="
                        f"{self._working_memory_tokens}, "
                        f"generation_reserve="
                        f"{WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS}"
                    )

            stdout_ids = _encode(self.tokenizer, tool_result.stdout)
            stderr_ids = _encode(self.tokenizer, tool_result.stderr)

            def cropped_text(original: str, token_ids: Sequence[int], keep: int) -> str:
                if keep >= len(token_ids):
                    return original
                prefix = _decode(self.tokenizer, token_ids[:keep]).rstrip()
                return prefix + _WORKING_MEMORY_V3_TOOL_TRUNCATION_MARKER

            def measured_candidate(
                stdout_keep: int,
                stderr_keep: int,
                *,
                marker: bool = True,
            ) -> tuple[ToolResult, _WorkingMemoryV3TurnInput]:
                if marker:
                    stdout = cropped_text(tool_result.stdout, stdout_ids, stdout_keep)
                    stderr = cropped_text(tool_result.stderr, stderr_ids, stderr_keep)
                else:
                    stdout = (
                        tool_result.stdout
                        if stdout_keep >= len(stdout_ids)
                        else _decode(self.tokenizer, stdout_ids[:stdout_keep])
                    )
                    stderr = (
                        tool_result.stderr
                        if stderr_keep >= len(stderr_ids)
                        else _decode(self.tokenizer, stderr_ids[:stderr_keep])
                    )
                candidate = replace_streams(stdout, stderr)
                return candidate, build_turn(candidate)

            def fits(turn: _WorkingMemoryV3TurnInput) -> bool:
                semantic = turn.latest_semantic_working_memory_input_ids
                raw = turn.latest_raw_working_memory_input_ids
                return (
                    semantic is not None
                    and raw is not None
                    and len(semantic) <= semantic_capacity
                    and (
                        protected_raw_capacity is None
                        or len(raw) <= protected_raw_capacity
                    )
                )

            selected = tool_result
            selected_turn = full_turn
            if semantic_tokens_before > semantic_capacity or (
                protected_raw_capacity is not None
                and raw_tokens_before > protected_raw_capacity
            ):
                # Preserve primary evidence (stdout) ahead of diagnostics
                # (stderr).  Search exact tokenized candidates; no approximate
                # character/byte multiplier participates in admission.
                zero_stderr = 0
                stdout_full_candidate, stdout_full_turn = measured_candidate(
                    len(stdout_ids), zero_stderr
                )
                if fits(stdout_full_turn):
                    low, high = 0, len(stderr_ids)
                    selected, selected_turn = (
                        stdout_full_candidate,
                        stdout_full_turn,
                    )
                    while low <= high:
                        middle = (low + high) // 2
                        candidate, turn = measured_candidate(len(stdout_ids), middle)
                        if fits(turn):
                            selected, selected_turn = candidate, turn
                            low = middle + 1
                        else:
                            high = middle - 1
                else:
                    low, high = 0, len(stdout_ids)
                    selected = replace_streams("", "")
                    selected_turn = build_turn(selected)
                    while low <= high:
                        middle = (low + high) // 2
                        candidate, turn = measured_candidate(middle, zero_stderr)
                        if fits(turn):
                            selected, selected_turn = candidate, turn
                            low = middle + 1
                        else:
                            high = middle - 1

                if not fits(selected_turn):
                    # A marker is useful but not allowed to turn a recoverable
                    # large observation into another hard failure.
                    selected, selected_turn = measured_candidate(0, 0, marker=False)
                if not fits(selected_turn):
                    semantic = selected_turn.latest_semantic_working_memory_input_ids
                    raise ValueError(
                        "working-memory-v3 assistant/tool metadata cannot fit "
                        "the protected raw/semantic budgets even with empty "
                        "stdout/stderr: "
                        f"P={raw_prefix_tokens}, "
                        f"projected_M_prev={projected_previous_memory_tokens}, "
                        f"semantic_W={0 if semantic is None else len(semantic)}, "
                        f"semantic_capacity={semantic_capacity}, "
                        f"protected_raw_capacity={protected_raw_capacity}"
                    )

            selected_semantic = selected_turn.latest_semantic_working_memory_input_ids
            assert selected_semantic is not None
            semantic_tokens_after = len(selected_semantic)
            if semantic_tokens_after > semantic_capacity:
                raise AssertionError(
                    "working-memory-v3 tool-result admission exceeded its budget"
                )
            return selected, {
                "policy": WORKING_MEMORY_V3_TOOL_RESULT_FIT_POLICY,
                "applied": selected != tool_result,
                "raw_prefix_tokens": raw_prefix_tokens,
                "current_memory_tokens": current_memory_tokens,
                "projected_previous_memory_tokens": (projected_previous_memory_tokens),
                "max_encoder_tokens": self.max_encoder_tokens,
                "semantic_capacity_tokens": semantic_capacity,
                "protected_raw_capacity_tokens": protected_raw_capacity,
                "generation_reserve_tokens": (
                    WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS
                ),
                "semantic_tokens_before": semantic_tokens_before,
                "semantic_tokens_after": semantic_tokens_after,
                "raw_working_memory_tokens_before": raw_tokens_before,
                "raw_working_memory_tokens_after": len(
                    selected_turn.latest_raw_working_memory_input_ids or ()
                ),
                "stdout_tokens_before": len(stdout_ids),
                "stdout_tokens_after": len(_encode(self.tokenizer, selected.stdout)),
                "stderr_tokens_before": len(stderr_ids),
                "stderr_tokens_after": len(_encode(self.tokenizer, selected.stderr)),
            }

    def clear_recursive_state(self) -> None:
        with self._state_lock:
            self._episode_id = None
            self._raw_prefix_input_ids = None
            self._memory = None
            self._state_transition_index = None
            self._latest_raw_working_memory = None
            self._latest_semantic_working_memory = None
            self._latest_working_memory_index = None
            self._working_memory_window = []

    def _activate_episode(self, episode_id: str, round_index: int) -> None:
        current = str(episode_id)
        if self._episode_id != current or int(round_index) == 1:
            self._episode_id = current
            self._raw_prefix_input_ids = None
            self._memory = None
            self._state_transition_index = None
            self._latest_raw_working_memory = None
            self._latest_semantic_working_memory = None
            self._latest_working_memory_index = None
            self._working_memory_window = []

    def _generate_next_locked(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink],
    ) -> RolloutGeneration:
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._activate_episode(episode_id, round_index)
        prompt_text = _render_chat(self.tokenizer, messages, add_generation_prompt=True)
        if not prompt_text.endswith(ASSISTANT_CONTENT_PREFIX):
            raise ValueError(
                "GrepSeek generation prompt no longer ends with '<think>\\n'"
            )
        turn = _build_working_memory_v3_turn_input(
            messages,
            self.tokenizer,
            episode_id=episode_id,
            round_index=round_index,
            max_turns=self._max_turns,
        )
        if self._raw_prefix_input_ids is None:
            self._raw_prefix_input_ids = turn.raw_prefix_input_ids
        elif self._raw_prefix_input_ids != turn.raw_prefix_input_ids:
            raise RuntimeError(
                "working-memory-v3 raw system/user prefix changed within episode"
            )

        raw_prefix = list(turn.raw_prefix_input_ids)
        raw_prefix_tokens = len(raw_prefix)
        generation_raw_prefix = list(
            turn.generation_raw_prefix_input_ids or turn.raw_prefix_input_ids
        )
        generation_raw_prefix_tokens = len(generation_raw_prefix)
        generation_user_suffix_tokens = int(turn.generation_user_suffix_tokens)
        if generation_raw_prefix_tokens != (
            raw_prefix_tokens + generation_user_suffix_tokens
        ):
            raise RuntimeError(
                "working-memory-v3 generation/user-prefix token receipt drifted"
            )
        boundary = [int(value) for value in ASSISTANT_OPENING_IDS]
        current_index = turn.working_memory_index
        current_raw = turn.latest_raw_working_memory_input_ids
        current_semantic = turn.latest_semantic_working_memory_input_ids
        if (current_raw is None) != (current_semantic is None):
            raise RuntimeError("working-memory-v3 current raw/semantic W arity drifted")

        transition_indices: list[int] = []
        compressed_semantic_token_lengths: list[int] = []
        source_encoder_token_lengths: list[int] = []
        source_token_lengths: list[int] = []
        slot_counts: list[int] = []
        transition_index: Optional[int] = None
        compressed_semantic_index: Optional[int] = None
        previous_memory_tokens = 0
        semantic_working_memory_tokens = 0
        physical_source_tokens = 0
        expected_memory_slots = 0
        memory_slots = 0
        source_encoder_tokens = 0
        source_compression_seconds = 0.0
        memory_positions: list[int] = []
        memory: Optional[torch.Tensor] = self._memory
        memory_slots = 0 if memory is None else int(memory.shape[0])
        memory_slots_before_trigger = memory_slots
        candidate_state_transition_index = self._state_transition_index
        state_replaced = False
        token_trigger_mode = (
            self._working_memory_tokens is not None
            and self._compression_trigger_tokens is not None
        )
        compression_trigger_reasons: list[str] = []
        context_trigger_tokens_before = memory_slots
        context_trigger_tokens_after = memory_slots
        left_context_tokens_before = memory_slots
        left_context_tokens_after = memory_slots
        effective_working_memory_tokens: Optional[int] = None
        effective_compression_trigger_tokens: Optional[int] = None
        protected_raw_tokens_before = 0
        protected_raw_tokens_after = 0
        protected_raw_count_before = 0
        protected_raw_count_after = 0
        unprotected_raw_tokens_before = 0
        unprotected_raw_tokens_after = 0

        if token_trigger_mode:
            assert self._working_memory_tokens is not None
            assert self._compression_trigger_tokens is not None
            available_context = (
                self.max_encoder_tokens
                - generation_raw_prefix_tokens
                - len(boundary)
                - WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS
            )
            if available_context <= 1:
                raise ValueError(
                    "working-memory-v3 persistent P leaves no room for protected "
                    "working memory plus a compaction trigger: "
                    f"generation_P={generation_raw_prefix_tokens}, "
                    f"boundary={len(boundary)}, "
                    f"generation_reserve="
                    f"{WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS}, "
                    f"max_encoder_tokens={self.max_encoder_tokens}"
                )
            effective_working_memory_tokens = min(
                self._working_memory_tokens, available_context - 1
            )
            effective_compression_trigger_tokens = min(
                self._compression_trigger_tokens,
                available_context - effective_working_memory_tokens,
            )
            if (
                effective_working_memory_tokens <= 0
                or effective_compression_trigger_tokens <= 0
            ):
                raise ValueError(
                    "working-memory-v3 cannot derive positive protected-W and "
                    "left-context budgets from the trained recovery cap"
                )

        candidate_window = list(self._working_memory_window)
        if current_index is None:
            if int(round_index) != 1 or current_raw is not None:
                raise RuntimeError("working-memory-v3 first turn has inconsistent W")
            if candidate_window or any(
                value is not None
                for value in (
                    self._memory,
                    self._state_transition_index,
                    self._latest_raw_working_memory,
                    self._latest_semantic_working_memory,
                    self._latest_working_memory_index,
                )
            ):
                raise RuntimeError(
                    "working-memory-v3 first turn unexpectedly retained episode state"
                )
            prefix_ids = generation_raw_prefix + boundary
        else:
            expected_current_index = int(round_index) - 1
            if (
                current_index != expected_current_index
                or current_raw is None
                or current_semantic is None
            ):
                raise RuntimeError(
                    "working-memory-v3 current W is not the newest completed exchange"
                )
            expected_retained_indices = list(
                range(
                    (
                        1
                        if candidate_state_transition_index is None
                        else candidate_state_transition_index + 1
                    ),
                    current_index,
                )
            )
            if [entry[0] for entry in candidate_window] != expected_retained_indices:
                raise RuntimeError(
                    "working-memory-v3 raw working-memory window is not contiguous"
                )
            candidate_window.append(
                (current_index, tuple(current_raw), tuple(current_semantic))
            )
            raw_tokens_before = sum(len(entry[1]) for entry in candidate_window)
            hard_prefix_tokens_before = (
                generation_raw_prefix_tokens
                + memory_slots
                + raw_tokens_before
                + len(boundary)
            )
            if token_trigger_mode:
                assert self._working_memory_tokens is not None
                assert effective_working_memory_tokens is not None
                assert effective_compression_trigger_tokens is not None
                retained_tokens = 0
                protected_raw_count_before = 0
                for _index, raw_working_memory, _semantic in reversed(candidate_window):
                    width = len(raw_working_memory)
                    if retained_tokens + width > effective_working_memory_tokens:
                        break
                    retained_tokens += width
                    protected_raw_count_before += 1
                if protected_raw_count_before == 0:
                    raise ValueError(
                        "working-memory-v3 newest complete exchange exceeds the "
                        "effective protected raw-W budget despite admission: "
                        f"newest_W={len(candidate_window[-1][1])}, "
                        f"effective_working_memory_tokens="
                        f"{effective_working_memory_tokens}"
                    )
                protected_raw_tokens_before = retained_tokens
                unprotected_count = len(candidate_window) - protected_raw_count_before
                unprotected_raw_tokens_before = sum(
                    len(entry[1]) for entry in candidate_window[:unprotected_count]
                )
                left_context_tokens_before = (
                    memory_slots + unprotected_raw_tokens_before
                )
                context_trigger_tokens_before = left_context_tokens_before
                if (
                    unprotected_count > 0
                    and left_context_tokens_before
                    >= effective_compression_trigger_tokens
                ):
                    compression_trigger_reasons.append(
                        "m_plus_unprotected_raw_working_memory"
                    )
                if (
                    unprotected_count > 0
                    and hard_prefix_tokens_before
                    + WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS
                    > self.max_encoder_tokens
                ):
                    compression_trigger_reasons.append("checkpoint_recovery_headroom")
                # Scheme 2: when compaction fires, fold the complete left raw
                # prefix L into M in trained one-exchange recurrence steps.
                # The protected right suffix R is never eligible for eviction.
                initial_evictions = (
                    unprotected_count if compression_trigger_reasons else 0
                )
            else:
                initial_evictions = max(
                    0, len(candidate_window) - self._working_memory_turns
                )
                if initial_evictions:
                    compression_trigger_reasons.append("working_memory_turns")

            while candidate_window and len(transition_indices) < initial_evictions:
                evicted = candidate_window.pop(0)
                transition_index = int(evicted[0])
                expected_transition_index = (
                    1
                    if candidate_state_transition_index is None
                    else candidate_state_transition_index + 1
                )
                if transition_index != expected_transition_index:
                    raise RuntimeError(
                        "working-memory-v3 update lost contiguous recurrence: "
                        f"expected={expected_transition_index}, "
                        f"observed={transition_index}"
                    )
                compressed_semantic_index = transition_index
                previous_memory = memory
                previous_memory_tokens = (
                    0 if previous_memory is None else int(previous_memory.shape[0])
                )
                semantic_source = list(evicted[2])
                semantic_working_memory_tokens = len(semantic_source)
                physical_source_tokens = (
                    previous_memory_tokens + semantic_working_memory_tokens
                )
                expected_memory_slots = math.ceil(
                    physical_source_tokens / self.compression_ratio
                )
                encoder_ids = list(raw_prefix)
                previous_positions: Optional[list[int]] = None
                if previous_memory is not None:
                    previous_positions = list(
                        range(
                            raw_prefix_tokens,
                            raw_prefix_tokens + previous_memory_tokens,
                        )
                    )
                    encoder_ids.extend(
                        [int(self.tokenizer.pad_token_id)] * previous_memory_tokens
                    )
                encoder_ids.extend(semantic_source)
                source_encoder_tokens = len(encoder_ids)
                if source_encoder_tokens > self.max_encoder_tokens:
                    raise ValueError(
                        "working-memory-v3 semantic source exceeds the trained encoder cap: "
                        f"P={raw_prefix_tokens}, M_prev={previous_memory_tokens}, "
                        f"semantic_W={semantic_working_memory_tokens}, "
                        f"total={source_encoder_tokens}, "
                        f"max_encoder_tokens={self.max_encoder_tokens}; refusing to crop"
                    )
                _validate_server_input_length(
                    token_count=source_encoder_tokens,
                    server_max_req_input_len=self.server_max_req_input_len,
                    label=(
                        f"working-memory-v3 update {transition_index} source encoder"
                    ),
                )
                compression_started = time.perf_counter()
                memory = self.client.encode_memory(
                    encoder_ids,
                    evidence_spans=[[raw_prefix_tokens, source_encoder_tokens]],
                    source_token_lengths=[physical_source_tokens],
                    expected_slot_count=expected_memory_slots,
                    recursive=True,
                    recursive_source_contract=WORKING_MEMORY_V3_SOURCE_CONTRACT,
                    memory_positions=previous_positions,
                    memory_embeddings=previous_memory,
                )
                source_compression_seconds = (
                    source_compression_seconds
                    + time.perf_counter()
                    - compression_started
                )
                if memory.shape != (expected_memory_slots, self.target_hidden_size):
                    raise RuntimeError(
                        "SGLang working-memory-v3 shape disagrees with checkpoint "
                        f"contract: expected={(expected_memory_slots, self.target_hidden_size)}, "
                        f"observed={tuple(memory.shape)}"
                    )
                if not torch.is_floating_point(memory) or not bool(
                    torch.isfinite(memory).all().item()
                ):
                    raise RuntimeError(
                        "SGLang working-memory-v3 memory must be finite floating point"
                    )
                memory = memory.detach().cpu().clone()
                memory_slots = int(memory.shape[0])
                state_replaced = True
                candidate_state_transition_index = transition_index
                transition_indices.append(transition_index)
                compressed_semantic_token_lengths.append(semantic_working_memory_tokens)
                source_encoder_token_lengths.append(source_encoder_tokens)
                source_token_lengths.append(physical_source_tokens)
                slot_counts.append(memory_slots)

            if token_trigger_mode:
                assert effective_working_memory_tokens is not None
                retained_tokens = 0
                protected_raw_count_after = 0
                for _index, raw_working_memory, _semantic in reversed(candidate_window):
                    width = len(raw_working_memory)
                    if retained_tokens + width > effective_working_memory_tokens:
                        break
                    retained_tokens += width
                    protected_raw_count_after += 1
                protected_raw_tokens_after = retained_tokens
                unprotected_after_count = (
                    len(candidate_window) - protected_raw_count_after
                )
                unprotected_raw_tokens_after = sum(
                    len(entry[1])
                    for entry in candidate_window[:unprotected_after_count]
                )
                left_context_tokens_after = memory_slots + unprotected_raw_tokens_after
                context_trigger_tokens_after = left_context_tokens_after
                hard_prefix_after = (
                    generation_raw_prefix_tokens
                    + memory_slots
                    + sum(len(entry[1]) for entry in candidate_window)
                    + len(boundary)
                )
                if (
                    hard_prefix_after + WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS
                    > self.max_encoder_tokens
                ):
                    raise ValueError(
                        "working-memory-v3 protected right working memory cannot "
                        "fit the trained recovery cap without violating scheme 2: "
                        f"prefix={hard_prefix_after}, reserve="
                        f"{WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS}, "
                        f"max_encoder_tokens={self.max_encoder_tokens}"
                    )

            prefix_ids = list(generation_raw_prefix)
            if memory is not None:
                memory_positions = list(
                    range(
                        generation_raw_prefix_tokens,
                        generation_raw_prefix_tokens + memory_slots,
                    )
                )
                prefix_ids.extend([int(self.tokenizer.pad_token_id)] * memory_slots)
            for _index, raw_working_memory, _semantic in candidate_window:
                prefix_ids.extend(raw_working_memory)
            prefix_ids.extend(boundary)

        generation_max_new_tokens = _working_memory_v3_generation_token_budget(
            prefix_token_count=len(prefix_ids),
            max_encoder_tokens=self.max_encoder_tokens,
            configured_max_new_tokens=self.max_new_tokens,
            server_context_length=self.server_context_length,
            server_max_total_num_tokens=self.server_max_total_num_tokens,
            server_max_req_input_len=self.server_max_req_input_len,
            use_remaining_context=self.use_remaining_context_for_generation,
        )
        latest_raw_tokens = 0 if current_raw is None else len(current_raw)
        latest_semantic_tokens = (
            0 if current_semantic is None else len(current_semantic)
        )
        raw_window_tokens = sum(len(entry[1]) for entry in candidate_window)
        raw_window_indices = [int(entry[0]) for entry in candidate_window]
        compressed_source_tokens = sum(source_token_lengths)
        retained_latest_raw_index = (
            None if not candidate_window else int(candidate_window[-1][0])
        )
        retained_latest_raw_tokens = (
            0 if not candidate_window else len(candidate_window[-1][1])
        )
        valid_memory = memory.detach().float() if memory is not None else None
        compression: dict[str, Any] = {
            "backend": self.backend,
            "context_organization": self.context_organization,
            "checkpoint_context_organization": self.checkpoint_context_organization,
            "checkpoint_schema": self.checkpoint_schema,
            "compression_ratio": self.compression_ratio,
            "recursive_working_memory_enabled": True,
            "recursive_working_memory_policy": self.recursive_working_memory_policy,
            "recursive_state_policy": self.recursive_state_policy,
            "recursive_source_contract": WORKING_MEMORY_V3_SOURCE_CONTRACT,
            "working_memory_turns": (
                None if token_trigger_mode else self._working_memory_turns
            ),
            "working_memory_turns_checkpoint_native": (
                False if token_trigger_mode else self._working_memory_turns == 1
            ),
            "working_memory_tokens": self._working_memory_tokens,
            "compression_trigger_tokens": self._compression_trigger_tokens,
            "effective_compression_trigger_tokens": (
                effective_compression_trigger_tokens
            ),
            "generation_reserve_tokens": (
                WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS
                if token_trigger_mode
                else None
            ),
            "working_memory_token_trigger_enabled": token_trigger_mode,
            "working_memory_trigger_policy": (
                TOKEN_TRIGGERED_WORKING_MEMORY_POLICY if token_trigger_mode else None
            ),
            "working_memory_window_policy": (
                TOKEN_BUDGET_WORKING_MEMORY_WINDOW_POLICY
                if token_trigger_mode
                else (
                    "checkpoint_native_one_round_lag_v3"
                    if self._working_memory_turns == 1
                    else "serving_time_latest_n_raw_exchanges_v1"
                )
            ),
            "compression_triggered": bool(transition_indices),
            "compression_trigger_reason": (
                "+".join(compression_trigger_reasons)
                if compression_trigger_reasons
                else None
            ),
            "compression_trigger_reasons": list(compression_trigger_reasons),
            "context_trigger_tokens_before": context_trigger_tokens_before,
            "context_trigger_tokens_after": context_trigger_tokens_after,
            "left_context_tokens_before": left_context_tokens_before,
            "left_context_tokens_after": left_context_tokens_after,
            "effective_working_memory_tokens": (effective_working_memory_tokens),
            "protected_raw_working_memory_count_before": (protected_raw_count_before),
            "protected_raw_working_memory_tokens_before": (protected_raw_tokens_before),
            "protected_raw_working_memory_count_after": (protected_raw_count_after),
            "protected_raw_working_memory_tokens_after": (protected_raw_tokens_after),
            "unprotected_raw_working_memory_tokens_before": (
                unprotected_raw_tokens_before
            ),
            "unprotected_raw_working_memory_tokens_after": (
                unprotected_raw_tokens_after
            ),
            "compressed_transition_count": len(transition_indices),
            "transition_indices": list(transition_indices),
            "compressed_semantic_working_memory_token_lengths": list(
                compressed_semantic_token_lengths
            ),
            "tool_call_turn_guidance_policy": (
                TOOL_CALL_TURN_GUIDANCE_POLICY if self._max_turns is not None else None
            ),
            "tool_call_turn_guidance_max_turns": self._max_turns,
            "tool_call_turn_guidance_max_tool_calls": (
                None if self._max_turns is None else self._max_turns - 1
            ),
            "tool_call_turn_guidance_current_tool_calls": (
                None if self._max_turns is None else int(round_index) - 1
            ),
            "tool_call_turn_guidance_action": (
                None
                if self._max_turns is None
                else (
                    "continue"
                    if int(round_index) - 1 < self._max_turns - 1
                    else "answer"
                )
            ),
            "generation_user_suffix_tokens": generation_user_suffix_tokens,
            "raw_working_memory_window_count": len(candidate_window),
            "raw_working_memory_window_tokens": raw_window_tokens,
            "raw_working_memory_window_indices": raw_window_indices,
            "transition_index": transition_index,
            "state_round_index": candidate_state_transition_index,
            "raw_prefix_tokens": raw_prefix_tokens,
            "previous_memory_tokens": previous_memory_tokens,
            "previous_memory_slots": previous_memory_tokens,
            "memory_slots_before_trigger": memory_slots_before_trigger,
            "compressed_semantic_working_memory_index": compressed_semantic_index,
            "compressed_semantic_working_memory_tokens": (
                semantic_working_memory_tokens
            ),
            "semantic_working_memory_tokens": semantic_working_memory_tokens,
            "discrete_suffix_tokens": semantic_working_memory_tokens,
            "physical_source_tokens": physical_source_tokens,
            "expected_memory_slots": expected_memory_slots,
            "memory_slots": memory_slots,
            "state_replaced": state_replaced,
            "latest_raw_working_memory_index": retained_latest_raw_index,
            "latest_raw_working_memory_tokens": retained_latest_raw_tokens,
            "current_working_memory_index": current_index,
            "current_raw_working_memory_tokens": latest_raw_tokens,
            "latest_semantic_working_memory_index": current_index,
            "latest_semantic_working_memory_tokens": latest_semantic_tokens,
            "generation_boundary_tokens": len(boundary),
            "student_prefix_tokens": len(prefix_ids),
            "source_segments": len(transition_indices),
            "source_tokens": compressed_source_tokens,
            "effective_compression_ratio": (
                physical_source_tokens / memory_slots if memory_slots else None
            ),
            "source_encoder_tokens": source_encoder_tokens,
            "source_encoder_token_lengths": list(source_encoder_token_lengths),
            "source_token_lengths": list(source_token_lengths),
            "slot_counts": list(slot_counts),
            "memory_tensor_shape": (
                None if memory is None else [int(size) for size in memory.shape]
            ),
            "valid_memory_shape": [memory_slots, self.target_hidden_size],
            "memory_dtype": (
                None if memory is None else str(memory.dtype).removeprefix("torch.")
            ),
            "valid_memory_rms": (
                None
                if valid_memory is None
                else _finite(valid_memory.square().mean().sqrt().item())
            ),
            "valid_memory_mean_l2": (
                None
                if valid_memory is None
                else _finite(valid_memory.norm(dim=-1).mean().item())
            ),
            "source_encode_seconds": None,
            "compressor_seconds": None,
            "source_compression_request_seconds": _finite(source_compression_seconds),
            "target_generation_seconds": None,
            "max_encoder_tokens": self.max_encoder_tokens,
            "server_context_length": self.server_context_length,
            "server_max_total_num_tokens": self.server_max_total_num_tokens,
            "server_max_req_input_len": self.server_max_req_input_len,
            "generation_max_new_tokens": generation_max_new_tokens,
            "generation_budget_policy": (
                "checkpoint_encoder_cap_and_remaining_server_context"
                if self.use_remaining_context_for_generation
                else "configured_cap_clamped_to_checkpoint_and_server_context"
            ),
        }
        _emit_rollout_event(
            event_sink,
            event="compression_finished",
            episode_id=episode_id,
            round_index=round_index,
            payload={"compression": dict(compression)},
        )

        generation_started = time.perf_counter()
        if memory is None:
            result = self.client.generate_discrete(
                prefix_ids, max_new_tokens=generation_max_new_tokens
            )
        else:
            result = self.client.generate_with_memory(
                prefix_ids,
                memory_positions=memory_positions,
                memory_embeddings=memory,
                max_new_tokens=generation_max_new_tokens,
            )
        compression["target_generation_seconds"] = _finite(
            time.perf_counter() - generation_started
        )

        # Decode/validate before committing M/W so the same round is retryable.
        finished = self._finish(result, compression)
        if transition_indices:
            assert memory is not None
            self._memory = memory
            self._state_transition_index = candidate_state_transition_index
        self._working_memory_window = list(candidate_window)
        if current_index is not None:
            assert current_raw is not None and current_semantic is not None
            self._latest_raw_working_memory = tuple(current_raw)
            self._latest_semantic_working_memory = tuple(current_semantic)
            self._latest_working_memory_index = current_index
        return finished


class Qwen36SGLangTokenClockRolloutGenerator:
    """Round-triggered recursive serving for verified 27B token-clock weights.

    The checkpoint itself was exported with a 1,024-token training clock.  The
    live runtime deliberately exposes a separate serving ABI: compression may
    run only after a complete native ``</tool_response>`` boundary, and the
    newest ``k`` complete ReAct rounds remain discrete working memory.  An
    evicted round is folded left-to-right in checkpoint-sized chunks so one
    unusually large observation cannot exceed the checkpoint's trained source
    cap. Telemetry keeps checkpoint identity and runtime organization distinct.
    """

    backend = "sglang"
    device = "remote-sglang"
    collect_diagnostics = True
    context_organization = QWEN36_ROUND_RECURSIVE_V1
    checkpoint_context_organization = QWEN36_TOKEN_CLOCK_V1
    recursive_working_memory_enabled = True
    recursive_working_memory_policy = QWEN36_ROUND_WORKING_MEMORY_POLICY
    recursive_state_policy = QWEN36_ROUND_RECURSIVE_STATE_POLICY
    recursive_source_contract = QWEN36_ROUND_RECURSIVE_SOURCE_CONTRACT
    enable_thinking = QWEN36_ENABLE_THINKING
    generation_sampling_policy = QWEN36_GENERATION_SAMPLING_POLICY
    generation_sampling = QWEN36_GENERATION_SAMPLING
    oversized_complete_round_eviction_policy = (
        QWEN36_OVERSIZED_COMPLETE_ROUND_EVICTION_POLICY
    )
    eviction_block_tokens = 1024
    raw_working_memory_tokens = 1024
    checkpoint_eviction_block_tokens = 1024
    checkpoint_raw_working_memory_tokens = 1024
    checkpoint_compression_trigger_tokens = 2048
    checkpoint_left_context_trigger_tokens = 1024

    def __init__(
        self,
        *,
        tokenizer: Any,
        client: GrepSeekSGLangClient,
        compression_ratio: int,
        target_hidden_size: int,
        max_encoder_tokens: Optional[int],
        max_new_tokens: int,
        checkpoint_schema: str,
        eos_token_id: int = EOS_TOKEN_ID,
        server_context_length: Optional[int] = None,
        server_max_total_num_tokens: Optional[int] = None,
        server_max_req_input_len: Optional[int] = None,
        use_remaining_context_for_generation: bool = False,
        remaining_context_output_ceiling_tokens: Optional[int] = None,
    ) -> None:
        runtime_contract = _token_clock_runtime_for_schema(checkpoint_schema)
        if compression_ratio != 16 or target_hidden_size != 5120:
            raise ValueError(
                f"{runtime_contract.display_name} token-clock runtime requires "
                "r16 and hidden size 5120"
            )
        if (
            isinstance(max_encoder_tokens, bool)
            or max_encoder_tokens != runtime_contract.max_encoder_tokens
        ):
            raise ValueError(
                f"{runtime_contract.display_name} token-clock runtime requires "
                "max_encoder_tokens="
                f"{runtime_contract.max_encoder_tokens}"
            )
        if isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if remaining_context_output_ceiling_tokens is not None and (
            isinstance(remaining_context_output_ceiling_tokens, bool)
            or int(remaining_context_output_ceiling_tokens) <= 0
        ):
            raise ValueError(
                "remaining_context_output_ceiling_tokens must be positive or None"
            )
        self.tokenizer = tokenizer
        self.client = client
        self.target_family = runtime_contract.target_family
        self.model_display_name = runtime_contract.display_name
        self.generation_sampling_policy = runtime_contract.generation_sampling_policy
        self.generation_sampling = QWEN36_GENERATION_SAMPLING
        self.telemetry_mode = runtime_contract.telemetry_mode
        self.source_episode_id = runtime_contract.source_episode_id
        self.checkpoint_cap_policy = runtime_contract.checkpoint_cap_policy
        self.dynamic_generation_budget_policy = (
            runtime_contract.dynamic_generation_budget_policy
        )
        self.fixed_generation_budget_policy = (
            runtime_contract.fixed_generation_budget_policy
        )
        self.compression_ratio = 16
        self.target_hidden_size = 5120
        self.max_encoder_tokens = runtime_contract.max_encoder_tokens
        self.max_new_tokens = int(max_new_tokens)
        self.checkpoint_schema = str(checkpoint_schema)
        self.eos_token_id = int(eos_token_id)
        self.server_context_length = (
            int(server_context_length) if server_context_length is not None else None
        )
        self.server_max_total_num_tokens = (
            int(server_max_total_num_tokens)
            if server_max_total_num_tokens is not None
            else None
        )
        self.server_max_req_input_len = (
            int(server_max_req_input_len)
            if server_max_req_input_len is not None
            else None
        )
        self.use_remaining_context_for_generation = bool(
            use_remaining_context_for_generation
        )
        self.remaining_context_output_ceiling_tokens = (
            int(remaining_context_output_ceiling_tokens)
            if remaining_context_output_ceiling_tokens is not None
            else None
        )
        self._state_lock = threading.RLock()
        self._episode_id: Optional[str] = None
        self._raw_prefix_input_ids: Optional[tuple[int, ...]] = None
        self._known_history_input_ids: tuple[int, ...] = ()
        self._known_history_message_end_tokens: tuple[int, ...] = ()
        # Opt-in only: currently running pools may fresh-import this module for
        # later samples, so changing the code must not silently change their
        # tokenization path.  The configured value is snapshotted per episode.
        self._native_history_policy = NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER
        self._episode_native_history_policy = NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER
        self._incremental_history_receipt_enabled = False
        self._episode_incremental_history_receipt_enabled = False
        self._incremental_history_receipt: Optional[
            _Qwen38IncrementalHistoryReceipt
        ] = None
        self._incremental_history_contract_identity: Optional[
            tuple[str, tuple[str, ...]]
        ] = None
        self._consumed_history_tokens = 0
        self._consumed_round_count = 0
        self._memory: Optional[torch.Tensor] = None
        self._memory_handle: Optional[TPGPUMemoryHandle] = None
        self._memory_transport_policy = (
            MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1
        )
        self._episode_memory_transport_policy = (
            MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1
        )
        self._state_transition_index: Optional[int] = None
        self._capacity_pressure_consumed_round_count = 0
        self._capacity_pressure_evicted_round_indices: set[int] = set()
        self._oversized_complete_round_consumed_round_count = 0
        self._oversized_complete_round_evicted_round_indices: set[int] = set()
        self._serving_eviction_policy = CHECKPOINT_1024_SERVING_EVICTION_POLICY
        self._episode_serving_eviction_policy: Optional[str] = None
        self._codex_native_compaction_enabled = False
        self._episode_codex_native_compaction_enabled = False
        self._codex_native_compaction_target_completed_rounds = 0
        self._max_turns: Optional[int] = None
        self._tool_names: tuple[str, ...] = ()
        self._working_memory_turns = 1
        self._post_compression_generation_policy = self.checkpoint_cap_policy
        self._capacity_pressure_eviction_enabled = False
        self._capacity_pressure_eviction_reserve_tokens = (
            QWEN36_ROUND_GENERATION_RESERVE_TOKENS
        )
        self._oversized_complete_round_eviction_enabled = False
        self.eviction_block_tokens = self.checkpoint_eviction_block_tokens
        self.raw_working_memory_tokens = self.checkpoint_raw_working_memory_tokens
        self.compression_trigger_tokens = self.checkpoint_left_context_trigger_tokens

    def configure_memory_transport_policy(self, policy: str) -> None:
        """Configure the versioned memory transport for the next episode."""

        if policy not in {
            MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1,
            MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1,
        }:
            raise ValueError(f"unsupported memory transport policy: {policy!r}")
        with self._state_lock:
            if self._episode_id is not None:
                raise RuntimeError(
                    "memory transport policy cannot change during an active episode"
                )
            self.client.configure_memory_transport_policy(policy)
            self._memory_transport_policy = policy

    def attest_memory_transport_policy(
        self,
        server_info: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        """Attest and return the receipt persisted by a versioned harness."""

        return self.client.attest_memory_transport_policy(server_info)

    def _release_gpu_memory_handle(
        self,
        handle: TPGPUMemoryHandle,
        *,
        reason: str,
    ) -> None:
        descriptor_digest = str(handle.descriptor["descriptor_digest"])
        operation_id = "r2rel_" + hashlib.sha256(
            f"{descriptor_digest}\0{reason}".encode("utf-8")
        ).hexdigest()
        self.client.release_memory_handle(
            handle,
            operation_id=operation_id,
        )

    def _cleanup_gpu_candidate_after_failure(
        self,
        candidate: Optional[TPGPUMemoryHandle],
        previous: Optional[TPGPUMemoryHandle],
        error: BaseException,
    ) -> None:
        if candidate is None or (
            previous is not None and candidate.handle_id == previous.handle_id
        ):
            return
        try:
            self._release_gpu_memory_handle(
                candidate,
                reason="candidate_generation_failure",
            )
        except BaseException as release_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    "candidate GPU memory cleanup also failed: "
                    f"{release_error}"
                )

    def configure_serving_eviction_policy(self, policy: str) -> None:
        """Configure a serving-only eviction policy for the next episode.

        The verified checkpoint clock remains 1,024 tokens.  The 8,192-token
        policy is an explicitly Qwen3.8-only serving canary: eligible complete
        rounds accumulate raw until the threshold is reached, then their whole
        contiguous span is folded using capacity-balanced recursive updates.
        An active episode keeps the policy snapshot it started with.
        """

        if not isinstance(policy, str) or policy not in (
            _TOKEN_CLOCK_SERVING_EVICTION_POLICIES
        ):
            raise ValueError(f"unsupported serving eviction policy: {policy!r}")
        if (
            policy == QWEN38_8192_ACCUMULATION_SERVING_EVICTION_POLICY
            and self.target_family != "qwen38"
        ):
            raise ValueError(
                "the 8192-token accumulation serving policy requires Qwen3.8"
            )
        with self._state_lock:
            self._serving_eviction_policy = policy

    @property
    def serving_eviction_policy(self) -> str:
        """Return the configured policy; active episodes retain a private snapshot."""

        with self._state_lock:
            return self._serving_eviction_policy

    def _effective_serving_eviction_policy(self) -> str:
        return self._episode_serving_eviction_policy or self._serving_eviction_policy

    def configure_codex_native_compaction(self, enabled: bool) -> None:
        """Let an external Codex checkpoint own the next episode's frontier."""

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        with self._state_lock:
            if self._episode_id is not None:
                raise RuntimeError(
                    "Codex native compaction mode cannot change during an active episode"
                )
            self._codex_native_compaction_enabled = enabled

    def stage_codex_native_compaction_target(
        self, target_completed_rounds: int
    ) -> None:
        """Stage one monotonic complete-round frontier for the active episode."""

        if (
            isinstance(target_completed_rounds, bool)
            or not isinstance(target_completed_rounds, int)
            or target_completed_rounds < 0
        ):
            raise ValueError("target_completed_rounds must be a nonnegative integer")
        with self._state_lock:
            if (
                self._episode_id is None
                or not self._episode_codex_native_compaction_enabled
            ):
                raise RuntimeError(
                    "Codex native compaction target requires an active native episode"
                )
            current_target = self._codex_native_compaction_target_completed_rounds
            if target_completed_rounds == current_target:
                return
            if target_completed_rounds < current_target:
                raise ValueError(
                    "Codex native compaction target must advance monotonically"
                )
            if target_completed_rounds < self._consumed_round_count:
                raise ValueError(
                    "Codex native compaction target precedes the committed cursor"
                )
            self._codex_native_compaction_target_completed_rounds = (
                target_completed_rounds
            )

    def stage_codex_native_summary(self, input_ids: Sequence[int]) -> None:
        """Replace a decode-only summary; ordinary admission enforces capacity."""
        ids = tuple(input_ids)
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("native summary must contain nonnegative integer token IDs")
        with self._state_lock:
            if not self._episode_codex_native_compaction_enabled:
                raise RuntimeError("native summary requires an active native episode")
            self._codex_native_summary_input_ids = ids

    def configure_max_turns(self, value: Optional[int]) -> None:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_WORKING_MEMORY_TURNS
        ):
            raise ValueError(
                f"max_turns must be between 1 and {MAX_WORKING_MEMORY_TURNS} or None"
            )
        with self._state_lock:
            self._max_turns = value

    def configure_tools(self, tool_names: Sequence[str]) -> None:
        names = tuple(str(name) for name in tool_names)
        _qwen36_tool_schemas(names, model_display_name=self.model_display_name)
        with self._state_lock:
            self._tool_names = names

    def _incremental_history_template_sha256(self) -> Optional[str]:
        template = getattr(self.tokenizer, "chat_template", None)
        if not isinstance(template, str) or not template:
            return None
        return hashlib.sha256(template.encode("utf-8")).hexdigest()

    def configure_native_history_policy(self, policy: str) -> None:
        """Select a versioned native-history renderer for future episodes.

        Legacy full-history rendering remains the default.  Enabling is
        fail-closed on the exact checkpoint schema and chat-template hash; the
        setting is snapshotted when an episode starts so a live episode cannot
        switch serializer halfway through its token clock.
        """

        if not isinstance(policy, str) or policy not in _NATIVE_HISTORY_POLICIES:
            raise ValueError(f"unsupported native history policy: {policy!r}")
        enabled = policy == NATIVE_HISTORY_POLICY_APPEND_ONLY_TOKEN_EXACT
        if enabled:
            if (
                self.target_family != "qwen38"
                or self.checkpoint_schema != QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA
            ):
                raise ValueError(
                    "incremental history receipts require the exact Qwen3.8 "
                    "r^2eAct token-clock checkpoint schema"
                )
            observed_hash = self._incremental_history_template_sha256()
            if observed_hash != QWEN38_INCREMENTAL_HISTORY_CHAT_TEMPLATE_SHA256:
                raise ValueError(
                    "incremental history receipts require the audited Qwen3.8 "
                    "chat template: expected_sha256="
                    f"{QWEN38_INCREMENTAL_HISTORY_CHAT_TEMPLATE_SHA256}, "
                    f"observed_sha256={observed_hash}"
                )
        with self._state_lock:
            self._native_history_policy = policy
            self._incremental_history_receipt_enabled = enabled

    def configure_incremental_history_receipt(self, enabled: bool) -> None:
        """Compatibility adapter for the versioned native-history policy."""

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        self.configure_native_history_policy(
            NATIVE_HISTORY_POLICY_APPEND_ONLY_TOKEN_EXACT
            if enabled
            else NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER
        )

    @property
    def native_history_policy(self) -> str:
        """Return the configured next-episode history policy."""

        with self._state_lock:
            return self._native_history_policy

    @property
    def incremental_history_receipt_enabled(self) -> bool:
        """Return the configured next-episode value, not the active snapshot."""

        with self._state_lock:
            return self._incremental_history_receipt_enabled

    def _apply_native_chat_ids(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tool_names: Sequence[str],
        add_generation_prompt: bool,
    ) -> list[int]:
        optional_family: dict[str, Any] = {}
        # Keep Qwen3.6 monkeypatch/call receipts byte-for-byte compatible while
        # giving Qwen3.8 failures their truthful model-family identity.
        if self.target_family == "qwen38":
            optional_family["model_display_name"] = self.model_display_name
        return _apply_qwen36_chat_ids(
            self.tokenizer,
            messages,
            tool_names=tool_names,
            add_generation_prompt=add_generation_prompt,
            **optional_family,
        )

    def _messages_with_native_tool_response_guidance(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_turns: Optional[int],
        round_index: int,
    ) -> tuple[list[dict[str, str]], Optional[str]]:
        optional_family: dict[str, Any] = {}
        if self.target_family == "qwen38":
            optional_family["model_display_name"] = self.model_display_name
        return _qwen36_messages_with_tool_response_turn_guidance(
            messages,
            max_turns=max_turns,
            round_index=round_index,
            **optional_family,
        )

    def _verify_incremental_history_append_contract(
        self,
        *,
        template_sha256: str,
    ) -> None:
        """Probe the pinned template's observation-pair identity once.

        The production receipt never hand-renders Qwen markup.  Instead it
        renders complete assistant/observation pairs through the native
        template and subtracts the same native system/user prefix.  The probe
        covers both a tool observation and a controller user-feedback
        observation, and proves that their isolated suffixes equal the native
        combined render while the generation suffix remains exactly G5.
        """

        identity = (str(template_sha256), tuple(self._tool_names))
        if self._incremental_history_contract_identity == identity:
            return
        initial = [
            {"role": "system", "content": "incremental receipt system probe"},
            {"role": "user", "content": "incremental receipt user probe"},
        ]
        pair_one = [
            {"role": "assistant", "content": "receipt assistant alpha"},
            {"role": "tool", "content": "receipt tool alpha"},
        ]
        pair_two = [
            {"role": "assistant", "content": "receipt assistant beta"},
            {"role": "user", "content": "receipt user feedback beta"},
        ]
        prefix = self._apply_native_chat_ids(
            initial,
            tool_names=self._tool_names,
            add_generation_prompt=False,
        )
        isolated_suffixes: list[int] = []
        for pair in (pair_one, pair_two):
            isolated = self._apply_native_chat_ids(
                initial + pair,
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
            assistant_only = self._apply_native_chat_ids(
                initial + pair[:1],
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
            if isolated[: len(prefix)] != prefix:
                raise RuntimeError(
                    f"{self.model_display_name} incremental receipt probe changed P"
                )
            if (
                assistant_only[: len(prefix)] != prefix
                or isolated[: len(assistant_only)] != assistant_only
            ):
                raise RuntimeError(
                    f"{self.model_display_name} incremental receipt probe has an "
                    "unstable assistant boundary"
                )
            isolated_suffixes.extend(isolated[len(prefix) :])
        combined = self._apply_native_chat_ids(
            initial + pair_one + pair_two,
            tool_names=self._tool_names,
            add_generation_prompt=False,
        )
        expected = prefix + isolated_suffixes
        if combined != expected:
            raise RuntimeError(
                f"{self.model_display_name} incremental receipt probe is not "
                "pair-concatenation stable"
            )
        prompt = self._apply_native_chat_ids(
            initial + pair_one + pair_two,
            tool_names=self._tool_names,
            add_generation_prompt=True,
        )
        if prompt != combined + list(QWEN36_ASSISTANT_OPENING_IDS):
            raise RuntimeError(
                f"{self.model_display_name} incremental receipt probe changed G5"
            )
        self._incremental_history_contract_identity = identity

    def _legacy_token_clock_history_render(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        template_sha256: Optional[str],
    ) -> _TokenClockHistoryRender:
        canonical, _released_lengths, _canonical_lengths = _canonicalize_tools(
            str(episode_id), messages
        )
        if len(canonical) < 2 or [item.get("role") for item in canonical[:2]] != [
            "system",
            "user",
        ]:
            raise ValueError(
                f"{self.model_display_name} token clock requires initial system/user messages"
            )
        guided, guidance = self._messages_with_native_tool_response_guidance(
            canonical,
            max_turns=self._max_turns,
            round_index=round_index,
        )
        raw_prefix = tuple(
            self._apply_native_chat_ids(
                guided[:2],
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
        )
        full_discrete = tuple(
            self._apply_native_chat_ids(
                guided,
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
        )
        if not raw_prefix or full_discrete[: len(raw_prefix)] != raw_prefix:
            raise ValueError(
                f"{self.model_display_name} native chat history does not preserve P"
            )
        history = full_discrete[len(raw_prefix) :]
        if self._raw_prefix_input_ids is not None and (
            self._raw_prefix_input_ids != raw_prefix
        ):
            raise RuntimeError(
                f"{self.model_display_name} token-clock P changed within the episode"
            )
        if history[: len(self._known_history_input_ids)] != (
            self._known_history_input_ids
        ):
            raise RuntimeError(
                f"{self.model_display_name} token-clock history is not append-only"
            )
        message_end_tokens = list(self._known_history_message_end_tokens)
        history_message_count = len(guided) - 2
        if len(message_end_tokens) > history_message_count:
            raise RuntimeError(
                f"{self.model_display_name} message-boundary receipt exceeds history"
            )
        appended_message_count = history_message_count - len(message_end_tokens)
        for message_offset in range(len(message_end_tokens), history_message_count):
            message_index = message_offset + 2
            rendered_prefix = self._apply_native_chat_ids(
                guided[: message_index + 1],
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
            if rendered_prefix != list(full_discrete[: len(rendered_prefix)]):
                raise RuntimeError(
                    f"{self.model_display_name} message prefix is not stable under native template"
                )
            message_end = len(rendered_prefix) - len(raw_prefix)
            previous_end = message_end_tokens[-1] if message_end_tokens else 0
            if message_end <= previous_end:
                raise RuntimeError(
                    f"{self.model_display_name} message token interval is empty"
                )
            message_end_tokens.append(message_end)
        if message_end_tokens and message_end_tokens[-1] != len(history):
            raise RuntimeError(
                f"{self.model_display_name} message boundaries do not cover history"
            )

        staged_receipt: Optional[_Qwen38IncrementalHistoryReceipt] = None
        render_policy = "legacy_full_history_v1"
        if self._episode_incremental_history_receipt_enabled:
            if template_sha256 != QWEN38_INCREMENTAL_HISTORY_CHAT_TEMPLATE_SHA256:
                raise RuntimeError(
                    f"{self.model_display_name} incremental history template identity "
                    "changed after episode activation"
                )
            assert template_sha256 is not None
            self._verify_incremental_history_append_contract(
                template_sha256=template_sha256
            )
            staged_receipt = _Qwen38IncrementalHistoryReceipt(
                episode_id=str(episode_id),
                template_sha256=template_sha256,
                checkpoint_schema=self.checkpoint_schema,
                tool_names=tuple(self._tool_names),
                max_turns=self._max_turns,
                raw_message_digests=tuple(
                    _raw_message_digest(message) for message in messages
                ),
                canonical_messages=_immutable_message_pairs(canonical),
                guided_messages=_immutable_message_pairs(guided),
                raw_prefix_input_ids=raw_prefix,
                history_input_ids=history,
                message_end_tokens=tuple(message_end_tokens),
                latest_guidance=guidance,
            )
            render_policy = (
                f"{QWEN38_INCREMENTAL_HISTORY_RECEIPT_POLICY}_bootstrap_oracle"
            )
        return _TokenClockHistoryRender(
            canonical_messages=_immutable_message_pairs(canonical),
            guided_messages=_immutable_message_pairs(guided),
            guidance=guidance,
            raw_prefix_input_ids=raw_prefix,
            full_discrete_input_ids=full_discrete,
            history_input_ids=history,
            message_end_tokens=tuple(message_end_tokens),
            staged_receipt=staged_receipt,
            render_policy=render_policy,
            appended_message_count=appended_message_count,
        )

    def _incremental_token_clock_history_render(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        template_sha256: str,
    ) -> _TokenClockHistoryRender:
        receipt = self._incremental_history_receipt
        if receipt is None:
            raise RuntimeError("incremental history receipt is not initialized")
        expected_identity = (
            str(episode_id),
            str(template_sha256),
            self.checkpoint_schema,
            tuple(self._tool_names),
            self._max_turns,
        )
        observed_identity = (
            receipt.episode_id,
            receipt.template_sha256,
            receipt.checkpoint_schema,
            receipt.tool_names,
            receipt.max_turns,
        )
        if observed_identity != expected_identity:
            raise RuntimeError(
                f"{self.model_display_name} incremental history receipt identity changed"
            )
        current_digests = tuple(_raw_message_digest(message) for message in messages)
        known_count = len(receipt.raw_message_digests)
        if len(current_digests) < known_count or current_digests[:known_count] != (
            receipt.raw_message_digests
        ):
            raise RuntimeError(
                f"{self.model_display_name} token-clock raw history is not append-only"
            )
        appended = messages[known_count:]
        if len(appended) % 2:
            raise ValueError(
                f"{self.model_display_name} incremental history requires complete "
                "assistant/observation pairs"
            )

        canonical_tail: list[dict[str, str]] = []
        for tail_offset, message in enumerate(appended):
            absolute_index = known_count + tail_offset
            normalized = {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            if normalized["role"] == "tool":
                payload = parse_tool_response(
                    normalized["content"],
                    episode_id=str(episode_id),
                    message_index=absolute_index,
                )
                normalized["content"] = _canonical_tool_content(
                    payload, str(payload["stdout"])
                )
            canonical_tail.append(normalized)
        for offset in range(0, len(canonical_tail), 2):
            pair_roles = [
                item["role"] for item in canonical_tail[offset : offset + 2]
            ]
            if (
                len(pair_roles) != 2
                or pair_roles[0] != "assistant"
                or pair_roles[1] not in {"tool", "user"}
            ):
                raise ValueError(
                    f"{self.model_display_name} incremental history requires atomic "
                    "assistant/(tool or user-feedback) groups"
                )

        guided_tail = [dict(message) for message in canonical_tail]
        receipt_messages = _message_pair_dicts(receipt.canonical_messages)
        (
            prior_completed_rounds,
            current_tool_calls,
            _prior_user_feedback_rounds,
        ) = _observation_role_counts(receipt_messages)
        latest_guidance = receipt.latest_guidance
        if self._max_turns is not None:
            for offset in range(0, len(guided_tail), 2):
                observation = guided_tail[offset + 1]
                prior_completed_rounds += 1
                if observation["role"] == "tool":
                    current_tool_calls += 1
                    guidance = _tool_call_turn_guidance(
                        max_turns=self._max_turns,
                        round_index=prior_completed_rounds + 1,
                        current_tool_calls=current_tool_calls,
                    )
                    assert guidance is not None
                    observation["content"] = (
                        f"{observation['content'].rstrip()}\n\n{guidance}"
                    )
                    latest_guidance = guidance
                else:
                    latest_guidance = None
            if prior_completed_rounds != int(round_index) - 1:
                raise ValueError(
                    f"{self.model_display_name} turn guidance requires exactly one "
                    "completed observation before each subsequent generation round"
                )
        else:
            latest_guidance = None

        raw_prefix = receipt.raw_prefix_input_ids
        history = list(receipt.history_input_ids)
        message_end_tokens = list(receipt.message_end_tokens)
        initial = _message_pair_dicts(receipt.guided_messages[:2])
        boundary = list(QWEN36_ASSISTANT_OPENING_IDS)
        for offset in range(0, len(guided_tail), 2):
            pair = guided_tail[offset : offset + 2]
            pair_discrete = self._apply_native_chat_ids(
                initial + pair,
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
            assistant_discrete = self._apply_native_chat_ids(
                initial + pair[:1],
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
            pair_prompt = self._apply_native_chat_ids(
                initial + pair,
                tool_names=self._tool_names,
                add_generation_prompt=True,
            )
            if pair_discrete[: len(raw_prefix)] != list(raw_prefix):
                raise RuntimeError(
                    f"{self.model_display_name} incremental pair changed P"
                )
            if assistant_discrete[: len(raw_prefix)] != list(raw_prefix) or (
                pair_discrete[: len(assistant_discrete)] != assistant_discrete
            ):
                raise RuntimeError(
                    f"{self.model_display_name} incremental assistant boundary is unstable"
                )
            if pair_prompt != pair_discrete + boundary:
                raise RuntimeError(
                    f"{self.model_display_name} incremental generation prompt changed G5"
                )
            previous_end = message_end_tokens[-1] if message_end_tokens else 0
            assistant_end = previous_end + len(assistant_discrete) - len(raw_prefix)
            pair_end = previous_end + len(pair_discrete) - len(raw_prefix)
            if not previous_end < assistant_end < pair_end:
                raise RuntimeError(
                    f"{self.model_display_name} incremental message interval is empty"
                )
            history.extend(pair_discrete[len(raw_prefix) :])
            message_end_tokens.extend((assistant_end, pair_end))

        canonical_pairs = receipt.canonical_messages + _immutable_message_pairs(
            canonical_tail
        )
        guided_pairs = receipt.guided_messages + _immutable_message_pairs(guided_tail)
        history_tuple = tuple(history)
        full_discrete = raw_prefix + history_tuple
        staged_receipt = _Qwen38IncrementalHistoryReceipt(
            episode_id=str(episode_id),
            template_sha256=str(template_sha256),
            checkpoint_schema=self.checkpoint_schema,
            tool_names=tuple(self._tool_names),
            max_turns=self._max_turns,
            raw_message_digests=current_digests,
            canonical_messages=canonical_pairs,
            guided_messages=guided_pairs,
            raw_prefix_input_ids=raw_prefix,
            history_input_ids=history_tuple,
            message_end_tokens=tuple(message_end_tokens),
            latest_guidance=latest_guidance,
        )
        return _TokenClockHistoryRender(
            canonical_messages=canonical_pairs,
            guided_messages=guided_pairs,
            guidance=latest_guidance,
            raw_prefix_input_ids=raw_prefix,
            full_discrete_input_ids=full_discrete,
            history_input_ids=history_tuple,
            message_end_tokens=tuple(message_end_tokens),
            staged_receipt=staged_receipt,
            render_policy=QWEN38_INCREMENTAL_HISTORY_RECEIPT_POLICY,
            appended_message_count=len(appended),
        )

    def _render_token_clock_history(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
    ) -> _TokenClockHistoryRender:
        template_sha256 = self._incremental_history_template_sha256()
        if not self._episode_incremental_history_receipt_enabled:
            return self._legacy_token_clock_history_render(
                messages,
                episode_id=episode_id,
                round_index=round_index,
                template_sha256=template_sha256,
            )
        if template_sha256 != QWEN38_INCREMENTAL_HISTORY_CHAT_TEMPLATE_SHA256:
            raise RuntimeError(
                f"{self.model_display_name} incremental history template identity "
                "changed after episode activation"
            )
        self._verify_incremental_history_append_contract(
            template_sha256=template_sha256
        )
        if self._incremental_history_receipt is None:
            return self._legacy_token_clock_history_render(
                messages,
                episode_id=episode_id,
                round_index=round_index,
                template_sha256=template_sha256,
            )
        return self._incremental_token_clock_history_render(
            messages,
            episode_id=episode_id,
            round_index=round_index,
            template_sha256=template_sha256,
        )

    def render_initial_prompt(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tool_names: Sequence[str],
    ) -> dict[str, Any]:
        """Render system+user with the checkpoint's exact Qwen tokenizer."""

        names = tuple(str(name) for name in tool_names)
        input_ids = self._apply_native_chat_ids(
            messages,
            tool_names=names,
            add_generation_prompt=False,
        )
        return {
            "rendered_prompt": _decode(self.tokenizer, input_ids),
            "token_count": len(input_ids),
            "add_generation_prompt": False,
            "enable_thinking": self.enable_thinking,
            "preserve_thinking": True,
        }

    def configure_working_memory_turns(self, value: int) -> None:
        """Keep the newest ``value`` complete assistant/tool rounds raw."""

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_WORKING_MEMORY_TURNS
        ):
            raise ValueError(
                "working_memory_turns must be between 1 and "
                f"{MAX_WORKING_MEMORY_TURNS}"
            )
        with self._state_lock:
            self._working_memory_turns = int(value)

    def configure_oversized_complete_round_eviction(
        self,
        *,
        enabled: bool,
    ) -> None:
        """Opt in to folding anomalously large complete raw rounds early.

        The threshold is serving-derived rather than a generation cap:
        ``floor(total_token_ceiling / (working_memory_turns + 1))``.  When a
        nonlatest raw complete round exceeds it, recursive consumption advances
        left-to-right through that whole round.  A newly completed latest round
        gets one successor-visible raw generation whenever capacity can admit
        it; capacity pressure still folds it if it cannot fit after older rounds
        are gone.  Target generation continues to use the full remaining
        authoritative SGLang budget.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if enabled:
            self._oversized_complete_round_threshold()
        with self._state_lock:
            self._oversized_complete_round_eviction_enabled = enabled

    def _authoritative_sglang_total_token_ceiling(self) -> int:
        missing_limits = [
            name
            for name, limit in (
                ("server_context_length", self.server_context_length),
                ("server_max_total_num_tokens", self.server_max_total_num_tokens),
            )
            if limit is None
        ]
        if missing_limits:
            raise ValueError(
                "oversized complete-round eviction requires authoritative "
                "SGLang limits; missing " + ", ".join(missing_limits)
            )
        assert self.server_context_length is not None
        assert self.server_max_total_num_tokens is not None
        total_token_ceiling = min(
            self.server_context_length,
            self.server_max_total_num_tokens,
        )
        if total_token_ceiling <= 0:
            raise ValueError("authoritative SGLang token limits must be positive")
        return total_token_ceiling

    def _oversized_complete_round_threshold(self) -> tuple[int, int]:
        total_token_ceiling = self._authoritative_sglang_total_token_ceiling()
        threshold_tokens = total_token_ceiling // (self._working_memory_turns + 1)
        if threshold_tokens <= 0:
            raise ValueError(
                "authoritative SGLang token pool is too small for oversized "
                "complete-round eviction"
            )
        return total_token_ceiling, threshold_tokens

    def configure_capacity_pressure_eviction(
        self,
        *,
        enabled: bool,
        reserve_tokens: int = QWEN36_ROUND_GENERATION_RESERVE_TOKENS,
    ) -> None:
        """Allow authoritative SGLang capacity to shrink the latest-k window.

        This serving-only policy is deliberately opt-in.  It never splits a
        completed ReAct round in the recovery prefix: pressure evicts the
        oldest remaining complete round, whose encoder updates retain the
        checkpoint-native 1,024-token block size.
        """

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a bool")
        if (
            isinstance(reserve_tokens, bool)
            or not isinstance(reserve_tokens, int)
            or reserve_tokens <= 0
        ):
            raise ValueError("reserve_tokens must be a positive integer")
        if enabled:
            self._authoritative_sglang_prefix_ceiling(
                reserve_tokens=int(reserve_tokens)
            )
        with self._state_lock:
            self._capacity_pressure_eviction_enabled = enabled
            self._capacity_pressure_eviction_reserve_tokens = int(reserve_tokens)

    def _authoritative_sglang_prefix_ceiling(
        self,
        *,
        reserve_tokens: int,
    ) -> tuple[int, int]:
        """Return ``(total-token ceiling, accepted recovery-prefix ceiling)``."""

        missing_limits = [
            name
            for name, limit in (
                ("server_context_length", self.server_context_length),
                ("server_max_total_num_tokens", self.server_max_total_num_tokens),
                ("server_max_req_input_len", self.server_max_req_input_len),
            )
            if limit is None
        ]
        if missing_limits:
            raise ValueError(
                "capacity-pressure eviction requires authoritative SGLang "
                "limits; missing " + ", ".join(missing_limits)
            )
        assert self.server_context_length is not None
        assert self.server_max_total_num_tokens is not None
        assert self.server_max_req_input_len is not None
        if self.server_context_length <= 2 or self.server_max_total_num_tokens <= 2:
            raise ValueError("SGLang serving token limits must be greater than two")
        if self.server_max_req_input_len <= 1:
            raise ValueError("server_max_req_input_len must be greater than one")
        total_token_ceiling = min(
            self.server_context_length,
            self.server_max_total_num_tokens,
        )
        maximum_prefix_tokens = min(
            total_token_ceiling - int(reserve_tokens) - 2,
            self.server_max_req_input_len - 1,
        )
        if maximum_prefix_tokens <= 0:
            raise ValueError(
                "SGLang serving context cannot reserve capacity-pressure output: "
                f"total_token_ceiling={total_token_ceiling}, "
                f"reserve_tokens={int(reserve_tokens)}, "
                f"server_max_req_input_len={self.server_max_req_input_len}"
            )
        return total_token_ceiling, maximum_prefix_tokens

    def _project_accumulated_round_span(
        self,
        snapshot: _RoundEvictionEpisodeSnapshot,
        *,
        target_consumed_rounds: int,
    ) -> tuple[_RoundEvictionTransitionSnapshot, ...]:
        """Balance one whole-round span against the recursive encoder capacity.

        The accumulation target controls when a flush starts, not a hard chunk
        width.  Once selected, the entire contiguous span is divided into the
        minimum number of near-equal updates that fit ``P + M_prev + B``.  This
        avoids a trained-clock-sized full block followed by a tiny tail while
        keeping the persistent cursor on a complete round boundary.
        """

        target_rounds = int(target_consumed_rounds)
        if (
            not snapshot.consumed_round_count
            <= target_rounds
            <= (snapshot.completed_round_count)
        ):
            raise RuntimeError(
                f"{self.model_display_name} accumulation target is outside history"
            )
        target_history_tokens = (
            snapshot.consumed_history_tokens
            if target_rounds == snapshot.consumed_round_count
            else snapshot.round_end_tokens[target_rounds - 1]
        )
        raw_tokens = target_history_tokens - snapshot.consumed_history_tokens
        if raw_tokens < 0:
            raise RuntimeError(
                f"{self.model_display_name} accumulation target precedes its cursor"
            )
        if raw_tokens == 0:
            return ()
        initial_capacity = (
            self.max_encoder_tokens
            - snapshot.raw_prefix_tokens
            - snapshot.memory_tokens
        )
        if initial_capacity <= 0:
            raise ValueError(
                f"{self.model_display_name} accumulated recursive source has no capacity"
            )

        minimum_chunk_count = max(1, math.ceil(raw_tokens / initial_capacity))
        for chunk_count in range(minimum_chunk_count, raw_tokens + 1):
            base, remainder = divmod(raw_tokens, chunk_count)
            chunk_lengths = [
                base + (1 if index < remainder else 0) for index in range(chunk_count)
            ]
            memory_tokens = snapshot.memory_tokens
            history_cursor = snapshot.consumed_history_tokens
            transitions: list[_RoundEvictionTransitionSnapshot] = []
            fits = True
            for chunk_tokens in chunk_lengths:
                source_capacity = (
                    self.max_encoder_tokens - snapshot.raw_prefix_tokens - memory_tokens
                )
                if chunk_tokens <= 0 or chunk_tokens > source_capacity:
                    fits = False
                    break
                next_memory_tokens = math.ceil(
                    (memory_tokens + chunk_tokens) / self.compression_ratio
                )
                transitions.append(
                    _RoundEvictionTransitionSnapshot(
                        history_start=history_cursor,
                        history_end=history_cursor + chunk_tokens,
                        raw_chunk_tokens=chunk_tokens,
                        memory_tokens_before=memory_tokens,
                        memory_tokens_after=next_memory_tokens,
                        source_encoder_tokens=(
                            snapshot.raw_prefix_tokens + memory_tokens + chunk_tokens
                        ),
                    )
                )
                history_cursor += chunk_tokens
                memory_tokens = next_memory_tokens
            if fits and history_cursor == target_history_tokens:
                return tuple(transitions)
        raise ValueError(
            f"{self.model_display_name} cannot partition the accumulated recursive source"
        )

    def _plan_round_evictions(
        self,
        *,
        raw_prefix_tokens: int,
        history_tokens: int,
        round_end_tokens: Sequence[int],
        completed_round_count: int,
        generation_boundary_tokens: int,
    ) -> dict[str, Any]:
        """Plan policy-driven whole-round evictions without mutating state."""

        ends = tuple(int(value) for value in round_end_tokens)
        if len(ends) != int(completed_round_count):
            raise RuntimeError(
                f"{self.model_display_name} round-eviction plan has incomplete boundaries"
            )
        if ends and ends[-1] != int(history_tokens):
            raise RuntimeError(
                f"{self.model_display_name} round-eviction plan does not cover history"
            )
        if any(
            end <= (0 if index == 0 else ends[index - 1])
            for index, end in enumerate(ends)
        ):
            raise RuntimeError(
                f"{self.model_display_name} round-eviction boundaries are not increasing"
            )
        if not 0 <= self._consumed_round_count <= int(completed_round_count):
            raise RuntimeError(
                f"{self.model_display_name} consumed round count exceeds history"
            )
        expected_consumed = (
            0
            if self._consumed_round_count == 0
            else ends[self._consumed_round_count - 1]
        )
        if self._consumed_history_tokens != expected_consumed:
            raise RuntimeError(
                f"{self.model_display_name} compressed cursor is not on a complete round boundary"
            )

        codex_native_compaction_enabled = (
            self._episode_codex_native_compaction_enabled
        )
        codex_native_compaction_target = (
            self._codex_native_compaction_target_completed_rounds
        )
        if (
            codex_native_compaction_enabled
            and codex_native_compaction_target > int(completed_round_count)
        ):
            raise ValueError(
                "Codex native compaction target exceeds completed history: "
                f"target={codex_native_compaction_target}, "
                f"completed_rounds={int(completed_round_count)}"
            )
        nominal_target = (
            codex_native_compaction_target
            if codex_native_compaction_enabled
            else max(0, int(completed_round_count) - self._working_memory_turns)
        )
        if self._consumed_round_count > nominal_target:
            prior_policy_covers_cursor = self._consumed_round_count <= max(
                self._capacity_pressure_consumed_round_count,
                self._oversized_complete_round_consumed_round_count,
            )
            if not prior_policy_covers_cursor:
                if codex_native_compaction_enabled:
                    raise RuntimeError(
                        "Codex native compaction target precedes the committed cursor"
                    )
                raise RuntimeError(
                    "working_memory_turns changed after the episode started"
                )

        consumed = self._consumed_history_tokens
        consumed_rounds = self._consumed_round_count
        if self._memory is not None and self._memory_handle is not None:
            raise RuntimeError("CPU memory and GPU memory handle are both active")
        memory_tokens = (
            int(self._memory.shape[0])
            if self._memory is not None
            else (
                0
                if self._memory_handle is None
                else self._memory_handle.shape[0]
            )
        )
        serving_eviction_policy = self._effective_serving_eviction_policy()
        accumulation_enabled = serving_eviction_policy == (
            QWEN38_8192_ACCUMULATION_SERVING_EVICTION_POLICY
        )
        episode_snapshot = _RoundEvictionEpisodeSnapshot(
            serving_eviction_policy=serving_eviction_policy,
            raw_prefix_tokens=int(raw_prefix_tokens),
            memory_tokens=memory_tokens,
            consumed_history_tokens=consumed,
            consumed_round_count=consumed_rounds,
            round_end_tokens=ends,
            completed_round_count=int(completed_round_count),
        )
        accumulation_projection: tuple[_RoundEvictionTransitionSnapshot, ...] = ()

        def advance_one_round() -> None:
            nonlocal consumed, consumed_rounds, memory_tokens, accumulation_projection
            evicted_round = consumed_rounds
            round_start = 0 if evicted_round == 0 else ends[evicted_round - 1]
            round_end = ends[evicted_round]
            if consumed != round_start or round_end <= round_start:
                raise RuntimeError(
                    f"{self.model_display_name} round-eviction plan lost a complete round boundary"
                )
            if accumulation_enabled:
                accumulation_projection = self._project_accumulated_round_span(
                    episode_snapshot,
                    target_consumed_rounds=consumed_rounds + 1,
                )
                if not accumulation_projection:
                    raise RuntimeError(
                        f"{self.model_display_name} accumulation projection is empty"
                    )
                consumed_rounds += 1
                consumed = ends[consumed_rounds - 1]
                memory_tokens = accumulation_projection[-1].memory_tokens_after
                return
            while consumed < round_end:
                source_capacity = (
                    self.max_encoder_tokens - int(raw_prefix_tokens) - memory_tokens
                )
                if source_capacity <= 0:
                    raise ValueError(
                        f"{self.model_display_name} projected recursive source has no remaining capacity"
                    )
                chunk_tokens = min(
                    round_end - consumed,
                    self.checkpoint_eviction_block_tokens,
                    source_capacity,
                )
                memory_tokens = math.ceil(
                    (memory_tokens + chunk_tokens) / self.compression_ratio
                )
                consumed += chunk_tokens
            consumed_rounds += 1

        accumulated_eligible_tokens_before = (
            0
            if nominal_target <= consumed_rounds
            else ends[nominal_target - 1] - consumed
        )
        accumulation_threshold_reached = bool(
            accumulation_enabled
            and accumulated_eligible_tokens_before
            >= QWEN38_SERVING_EVICTION_ACCUMULATION_TOKENS
        )
        codex_native_target_pending = bool(
            codex_native_compaction_enabled and nominal_target > consumed_rounds
        )
        baseline_target = max(nominal_target, consumed_rounds)
        if (
            accumulation_enabled
            and not accumulation_threshold_reached
            and not codex_native_target_pending
        ):
            baseline_target = consumed_rounds
        runtime_policy = getattr(self, "_runtime_window_policy", None)
        if runtime_policy is not None and accumulation_enabled and not codex_native_target_pending:
            baseline_target, _ = runtime_policy.accumulation_target(
                ends, consumed_rounds, nominal_target,
                QWEN38_SERVING_EVICTION_ACCUMULATION_TOKENS)
        # Default checkpoint behavior measures forced evictions relative to the
        # nominal latest-k window, including irreversible prior pressure.  The
        # accumulation canary additionally treats a threshold-deferred nominal
        # round as forced when capacity/oversized policy consumes it now.
        policy_effective_floor = min(nominal_target, baseline_target)
        while consumed_rounds < baseline_target:
            advance_one_round()
        codex_native_target_consumed_rounds = consumed_rounds
        codex_native_target_source_tokens = (
            consumed - episode_snapshot.consumed_history_tokens
            if codex_native_target_pending
            else 0
        )
        codex_native_target_applied = bool(
            codex_native_target_pending
            and consumed_rounds > episode_snapshot.consumed_round_count
        )

        def projected_prefix_tokens() -> int:
            return (
                int(raw_prefix_tokens)
                + memory_tokens
                + int(history_tokens)
                - consumed
                + int(generation_boundary_tokens)
            )

        prefix_before_oversized_round_eviction = projected_prefix_tokens()
        oversized_total_token_ceiling: Optional[int] = None
        oversized_round_threshold_tokens: Optional[int] = None
        oversized_trigger_indices: list[int] = []
        oversized_trigger_token_lengths: list[int] = []
        oversized_evicted_this_round = 0
        oversized_evicted_indices_this_round: list[int] = []
        oversized_reason: Optional[str] = None
        latest_oversized_round_index: Optional[int] = None
        if self._oversized_complete_round_eviction_enabled:
            (
                oversized_total_token_ceiling,
                oversized_round_threshold_tokens,
            ) = self._oversized_complete_round_threshold()
            for round_zero_based in range(consumed_rounds, int(completed_round_count)):
                round_start = 0 if round_zero_based == 0 else ends[round_zero_based - 1]
                round_tokens = ends[round_zero_based] - round_start
                if round_tokens > oversized_round_threshold_tokens:
                    oversized_trigger_indices.append(round_zero_based + 1)
                    oversized_trigger_token_lengths.append(round_tokens)
            if oversized_trigger_indices:
                if runtime_policy is not None:
                    oversized_trigger_indices, _ = runtime_policy.oversized_rounds(
                        ends, consumed_rounds, oversized_total_token_ceiling, self._working_memory_turns)
                latest_round_index = int(completed_round_count)
                if oversized_trigger_indices[-1] == latest_round_index:
                    latest_oversized_round_index = latest_round_index
                nonlatest_trigger_indices = [
                    round_number
                    for round_number in oversized_trigger_indices
                    if round_number < latest_round_index
                ]
                if nonlatest_trigger_indices:
                    # The cursor is monotonic.  Reaching an oversized raw round
                    # necessarily folds every older raw round first, and no
                    # round is removed until its complete boundary is reached.
                    # The newest completed round is deliberately excluded here:
                    # capacity pressure may first evict older rounds and keeps
                    # that newest evidence raw for its successor whenever it fits.
                    oversized_target = nonlatest_trigger_indices[-1]
                    while consumed_rounds < oversized_target:
                        advance_one_round()
                        oversized_evicted_this_round += 1
                        oversized_evicted_indices_this_round.append(consumed_rounds)
                    oversized_reason = (
                        "raw_complete_round_exceeds_authoritative_per_round_"
                        "token_threshold"
                    )

        oversized_stage_consumed_rounds = consumed_rounds
        prefix_after_oversized_round_eviction = projected_prefix_tokens()
        projected_oversized_evicted_indices = set(
            self._oversized_complete_round_evicted_round_indices
        )
        projected_oversized_evicted_indices.update(oversized_evicted_indices_this_round)
        effective_oversized_evicted_indices = sorted(
            round_index
            for round_index in projected_oversized_evicted_indices
            if policy_effective_floor < round_index <= consumed_rounds
        )
        oversized_eviction_effective = bool(effective_oversized_evicted_indices)
        if oversized_reason is None and oversized_eviction_effective:
            oversized_reason = "prior_oversized_complete_round_eviction_is_irreversible"

        prefix_before_capacity_pressure = projected_prefix_tokens()
        total_token_ceiling: Optional[int] = None
        maximum_prefix_tokens: Optional[int] = None
        evicted_this_round = 0
        capacity_evicted_indices_this_round: list[int] = []
        reason: Optional[str] = None
        if self._capacity_pressure_eviction_enabled:
            total_token_ceiling, maximum_prefix_tokens = (
                self._authoritative_sglang_prefix_ceiling(
                    reserve_tokens=self._capacity_pressure_eviction_reserve_tokens
                )
            )
            def image_budget_tokens():
                callback = getattr(self, "_runtime_visual_budget", None)
                return callback(consumed_rounds, int(completed_round_count)) if callback else 0

            while (
                projected_prefix_tokens() + image_budget_tokens() > maximum_prefix_tokens
                and consumed_rounds < int(completed_round_count)
            ):
                advance_one_round()
                evicted_this_round += 1
                capacity_evicted_indices_this_round.append(consumed_rounds)
            if projected_prefix_tokens() + image_budget_tokens() > maximum_prefix_tokens:
                raise ValueError(
                    f"{self.model_display_name} recovery prefix exceeds authoritative SGLang "
                    "capacity even after evicting every complete round: "
                    f"prefix_tokens={projected_prefix_tokens()}, "
                    f"maximum_prefix_tokens={maximum_prefix_tokens}, "
                    "capacity_pressure_reserve_tokens="
                    f"{self._capacity_pressure_eviction_reserve_tokens}"
                )
            if evicted_this_round:
                reason = (
                    "projected_recovery_prefix_plus_reserve_exceeds_"
                    "authoritative_sglang_admission_ceiling"
                )

        projected_capacity_evicted_indices = set(
            self._capacity_pressure_evicted_round_indices
        )
        projected_capacity_evicted_indices.update(capacity_evicted_indices_this_round)
        effective_capacity_evicted_indices = sorted(
            round_index
            for round_index in projected_capacity_evicted_indices
            if policy_effective_floor < round_index <= consumed_rounds
        )
        capacity_eviction_effective = bool(effective_capacity_evicted_indices)
        if reason is None and capacity_eviction_effective:
            reason = "prior_capacity_pressure_eviction_is_irreversible"

        latest_oversized_round_evicted_by_capacity = (
            latest_oversized_round_index in capacity_evicted_indices_this_round
            if latest_oversized_round_index is not None
            else False
        )
        latest_oversized_round_grace_effective = (
            latest_oversized_round_index is not None
            and consumed_rounds < latest_oversized_round_index
        )
        latest_oversized_round_grace_reason: Optional[str] = None
        if latest_oversized_round_grace_effective:
            latest_oversized_round_grace_reason = (
                "latest_oversized_complete_round_retained_for_successor_generation"
            )
        elif latest_oversized_round_evicted_by_capacity:
            latest_oversized_round_grace_reason = (
                "latest_oversized_complete_round_cannot_fit_authoritative_capacity"
            )

        effective_raw_rounds = int(completed_round_count) - consumed_rounds
        nominal_raw_rounds = int(completed_round_count) - nominal_target
        accumulated_source_tokens = consumed - episode_snapshot.consumed_history_tokens
        codex_native_forced_subthreshold_flush = bool(
            accumulation_enabled
            and codex_native_target_applied
            and 0
            < codex_native_target_source_tokens
            < QWEN38_SERVING_EVICTION_ACCUMULATION_TOKENS
        )
        forced_subthreshold_flush = codex_native_forced_subthreshold_flush or bool(
            accumulation_enabled
            and 0
            < accumulated_source_tokens
            < QWEN38_SERVING_EVICTION_ACCUMULATION_TOKENS
        )
        forced_flush_reasons: list[str] = []
        if codex_native_forced_subthreshold_flush:
            forced_flush_reasons.append(
                _CODEX_NATIVE_COMPACTION_FORCED_FLUSH_REASON
            )
        if forced_subthreshold_flush and oversized_evicted_this_round:
            forced_flush_reasons.append("oversized_complete_round_eviction")
        if forced_subthreshold_flush and evicted_this_round:
            forced_flush_reasons.append("capacity_pressure_eviction")
        staged_eligible_tokens_after = (
            0
            if nominal_target <= consumed_rounds
            else ends[nominal_target - 1] - consumed
        )
        effective_episode_snapshot = (
            {
                "serving_eviction_policy": serving_eviction_policy,
                "consumed_history_tokens_before": (
                    episode_snapshot.consumed_history_tokens
                ),
                "consumed_round_count_before": episode_snapshot.consumed_round_count,
                "target_consumed_history_tokens": consumed,
                "target_consumed_round_count": consumed_rounds,
                "memory_tokens_before": episode_snapshot.memory_tokens,
                "memory_tokens_after": memory_tokens,
                "transitions": [
                    {
                        "history_start": transition.history_start,
                        "history_end": transition.history_end,
                        "raw_chunk_tokens": transition.raw_chunk_tokens,
                        "memory_tokens_before": transition.memory_tokens_before,
                        "memory_tokens_after": transition.memory_tokens_after,
                        "source_encoder_tokens": transition.source_encoder_tokens,
                    }
                    for transition in accumulation_projection
                ],
            }
            if accumulation_enabled
            else None
        )
        return {
            "prefix_tokens": projected_prefix_tokens(),
            "image_budget_tokens": (getattr(self, "_runtime_visual_budget", lambda *_: 0)(
                consumed_rounds, int(completed_round_count))),
            "prefix_tokens_before_capacity_pressure": (prefix_before_capacity_pressure),
            "prefix_tokens_before_oversized_complete_round_eviction": (
                prefix_before_oversized_round_eviction
            ),
            "prefix_tokens_after_oversized_complete_round_eviction": (
                prefix_after_oversized_round_eviction
            ),
            "projected_memory_tokens": memory_tokens,
            "projected_raw_working_memory_tokens": (int(history_tokens) - consumed),
            "projected_consumed_history_tokens": consumed,
            "projected_consumed_rounds": consumed_rounds,
            "nominal_target_consumed_rounds": nominal_target,
            "effective_target_consumed_rounds": consumed_rounds,
            "nominal_raw_working_memory_window_count": nominal_raw_rounds,
            "effective_raw_working_memory_window_count": effective_raw_rounds,
            "serving_eviction_policy": serving_eviction_policy,
            **(
                {
                    "codex_native_compaction_enabled": True,
                    "codex_native_compaction_policy": (
                        CODEX_NATIVE_COMPACTION_POLICY
                    ),
                    "codex_native_compaction_staged_target_completed_rounds": (
                        codex_native_compaction_target
                    ),
                    "codex_native_compaction_target_pending_before_plan": (
                        codex_native_target_pending
                    ),
                    "codex_native_compaction_target_applied_this_round": (
                        codex_native_target_applied
                    ),
                    "codex_native_compaction_target_consumed_rounds": (
                        codex_native_target_consumed_rounds
                    ),
                    "codex_native_compaction_source_tokens_this_round": (
                        codex_native_target_source_tokens
                    ),
                    "codex_native_compaction_forced_subthreshold_flush": (
                        codex_native_forced_subthreshold_flush
                    ),
                    "codex_native_compaction_forced_flush_reason": (
                        _CODEX_NATIVE_COMPACTION_FORCED_FLUSH_REASON
                        if codex_native_forced_subthreshold_flush
                        else None
                    ),
                }
                if codex_native_compaction_enabled
                else {}
            ),
            "serving_eviction_accumulation_target_tokens": (
                QWEN38_SERVING_EVICTION_ACCUMULATION_TOKENS
                if accumulation_enabled
                else None
            ),
            "serving_eviction_accumulated_eligible_tokens_before": (
                accumulated_eligible_tokens_before if accumulation_enabled else None
            ),
            "serving_eviction_accumulation_threshold_reached": (
                accumulation_threshold_reached if accumulation_enabled else False
            ),
            "serving_eviction_accumulation_flush_triggered": bool(
                accumulation_enabled and accumulated_source_tokens > 0
            ),
            "serving_eviction_forced_subthreshold_flush": forced_subthreshold_flush,
            "serving_eviction_forced_flush_reasons": forced_flush_reasons,
            "serving_eviction_accumulated_source_tokens": (
                accumulated_source_tokens if accumulation_enabled else None
            ),
            "serving_eviction_staged_eligible_tokens_after": (
                staged_eligible_tokens_after if accumulation_enabled else None
            ),
            "serving_eviction_balanced_chunk_token_lengths": (
                [transition.raw_chunk_tokens for transition in accumulation_projection]
                if accumulation_enabled
                else []
            ),
            "effective_episode_snapshot": effective_episode_snapshot,
            "oversized_complete_round_eviction_configured": (
                self._oversized_complete_round_eviction_enabled
            ),
            "oversized_complete_round_eviction_effective": (
                oversized_eviction_effective
            ),
            "oversized_complete_round_eviction_applied_this_round": bool(
                oversized_evicted_this_round
            ),
            "oversized_complete_round_evicted_round_count": max(
                0, len(effective_oversized_evicted_indices)
            ),
            "oversized_complete_round_evicted_round_count_this_round": (
                oversized_evicted_this_round
            ),
            "oversized_complete_round_eviction_reason": oversized_reason,
            "oversized_complete_round_evicted_round_indices": (
                effective_oversized_evicted_indices
            ),
            "oversized_complete_round_evicted_round_indices_this_round": (
                oversized_evicted_indices_this_round
            ),
            "oversized_complete_round_authoritative_sglang_token_ceiling": (
                oversized_total_token_ceiling
            ),
            "oversized_complete_round_threshold_tokens": (
                oversized_round_threshold_tokens
            ),
            "oversized_complete_round_trigger_indices": oversized_trigger_indices,
            "oversized_complete_round_trigger_token_lengths": (
                oversized_trigger_token_lengths
            ),
            "oversized_complete_round_latest_grace_considered": (
                latest_oversized_round_index is not None
            ),
            "oversized_complete_round_latest_grace_effective": (
                latest_oversized_round_grace_effective
            ),
            "oversized_complete_round_latest_grace_round_index": (
                latest_oversized_round_index
            ),
            "oversized_complete_round_latest_grace_reason": (
                latest_oversized_round_grace_reason
            ),
            "oversized_complete_round_latest_evicted_by_capacity_pressure": (
                latest_oversized_round_evicted_by_capacity
            ),
            "oversized_complete_round_latest_grace_maximum_prefix_tokens": (
                maximum_prefix_tokens
            ),
            "oversized_complete_round_stage_consumed_round_count": (
                oversized_stage_consumed_rounds
            ),
            "capacity_pressure_eviction_configured": (
                self._capacity_pressure_eviction_enabled
            ),
            "capacity_pressure_eviction_effective": capacity_eviction_effective,
            "capacity_pressure_eviction_applied_this_round": bool(evicted_this_round),
            "capacity_pressure_evicted_round_count": max(
                0, len(effective_capacity_evicted_indices)
            ),
            "capacity_pressure_evicted_round_count_this_round": (evicted_this_round),
            "capacity_pressure_eviction_reason": reason,
            "capacity_pressure_evicted_round_indices": (
                effective_capacity_evicted_indices
            ),
            "capacity_pressure_evicted_round_indices_this_round": (
                capacity_evicted_indices_this_round
            ),
            "capacity_pressure_reserve_tokens": (
                self._capacity_pressure_eviction_reserve_tokens
            ),
            "capacity_pressure_authoritative_sglang_token_ceiling": (
                total_token_ceiling
            ),
            "capacity_pressure_maximum_prefix_tokens": maximum_prefix_tokens,
        }

    def configure_post_compression_generation_policy(self, value: str) -> None:
        """Select the request-local recovery budget without changing encode."""

        supported_policies = {
            QWEN36_POST_COMPRESSION_GENERATION_POLICY_CHECKPOINT_CAP,
            self.checkpoint_cap_policy,
        }
        # Both verified token-clock runtimes use the same family-neutral
        # remaining-SGLang-window recovery path below.  Qwen3.8 was initially
        # shipped with only its 12,288-token checkpoint cap exposed, even
        # though the serving implementation and authoritative server-limit
        # validation already support the explicit remaining-window policy.
        if self.target_family in {"qwen36", "qwen38"}:
            supported_policies.add(
                QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL
            )
        if not isinstance(value, str) or value not in supported_policies:
            raise ValueError(
                f"unsupported {self.model_display_name} post-compression generation policy: "
                f"{value!r}"
            )
        if value == (QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL):
            if not self.use_remaining_context_for_generation:
                raise ValueError(
                    "experimental post-compression generation requires "
                    "remaining-context generation"
                )
            missing_limits = [
                name
                for name, limit in (
                    ("server_context_length", self.server_context_length),
                    (
                        "server_max_total_num_tokens",
                        self.server_max_total_num_tokens,
                    ),
                    ("server_max_req_input_len", self.server_max_req_input_len),
                )
                if limit is None
            ]
            if missing_limits:
                raise ValueError(
                    "experimental post-compression generation requires "
                    "authoritative SGLang limits; missing " + ", ".join(missing_limits)
                )
        normalized = (
            self.checkpoint_cap_policy
            if value == QWEN36_POST_COMPRESSION_GENERATION_POLICY_CHECKPOINT_CAP
            else value
        )
        with self._state_lock:
            self._post_compression_generation_policy = normalized

    @property
    def post_compression_generation_policy(self) -> str:
        """Return the schema-normalized request policy currently in effect."""

        with self._state_lock:
            return self._post_compression_generation_policy

    def _project_next_round_prefix(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        tool_result: ToolResult,
        episode_id: str,
        round_index: int,
    ) -> dict[str, Any]:
        """Measure the next round after whole-round eviction, without mutation."""

        candidate_messages = [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in messages
        ]
        candidate_messages.append({"role": "tool", "content": tool_result.to_payload()})
        history_render = self._render_token_clock_history(
            candidate_messages,
            episode_id=str(episode_id),
            round_index=int(round_index) + 1,
        )
        guided = _message_pair_dicts(history_render.guided_messages)
        raw_prefix = history_render.raw_prefix_input_ids
        full_discrete = history_render.full_discrete_input_ids
        if full_discrete[: len(raw_prefix)] != raw_prefix:
            raise RuntimeError(
                f"{self.model_display_name} tool-result admission changed P"
            )
        if self._raw_prefix_input_ids != raw_prefix:
            raise RuntimeError(
                f"{self.model_display_name} tool-result admission disagrees with persistent P"
            )
        history = history_render.history_input_ids
        if (
            history[: len(self._known_history_input_ids)]
            != self._known_history_input_ids
        ):
            raise RuntimeError(
                f"{self.model_display_name} tool-result admission changed prior history"
            )
        roles = [str(message.get("role", "")) for message in guided[2:]]
        if len(roles) != int(round_index) * 2:
            raise ValueError(
                f"{self.model_display_name} tool-result admission has the wrong round count"
            )
        for offset in range(0, len(roles), 2):
            pair_roles = roles[offset : offset + 2]
            if (
                len(pair_roles) != 2
                or pair_roles[0] != "assistant"
                or pair_roles[1] not in {"tool", "user"}
            ):
                raise ValueError(
                    f"{self.model_display_name} tool-result admission requires atomic "
                    "assistant/observation rounds"
                )

        round_end_tokens = [
            int(history_render.message_end_tokens[(completed_round * 2) - 1])
            for completed_round in range(1, int(round_index) + 1)
        ]
        if not round_end_tokens or round_end_tokens[-1] != len(history):
            raise RuntimeError(
                f"{self.model_display_name} tool-result admission lost the latest round"
            )

        plan = self._plan_round_evictions(
            raw_prefix_tokens=len(raw_prefix),
            history_tokens=len(history),
            round_end_tokens=round_end_tokens,
            completed_round_count=int(round_index),
            generation_boundary_tokens=len(QWEN36_ASSISTANT_OPENING_IDS),
        )
        return {"raw_prefix_tokens": len(raw_prefix), **plan,
                "prefix_tokens": plan["prefix_tokens"] + plan.get("image_budget_tokens", 0)}

    def prepare_tool_result_for_next_round(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        tool_result: ToolResult,
        episode_id: str,
        round_index: int,
    ) -> tuple[ToolResult, dict[str, Any]]:
        """Crop released tool streams so the next raw-round prefix has headroom."""

        if not isinstance(tool_result, ToolResult):
            raise TypeError("tool_result must be ToolResult")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index <= 0
        ):
            raise ValueError("round_index must be positive")
        if not messages or str(messages[-1].get("role", "")) != "assistant":
            raise ValueError(
                f"{self.model_display_name} tool-result admission requires the current assistant message"
            )
        with self._state_lock:
            if (
                self._episode_id != str(episode_id)
                or self._raw_prefix_input_ids is None
            ):
                raise RuntimeError(
                    f"{self.model_display_name} tool-result admission has no active episode"
                )

            def replace_streams(stdout: str, stderr: str) -> ToolResult:
                return replace(
                    tool_result,
                    stdout=stdout,
                    stderr=stderr,
                    truncated=True,
                    information_lines=[
                        line for line in stdout.splitlines() if line.strip()
                    ],
                )

            stdout_ids = _encode(self.tokenizer, tool_result.stdout)
            stderr_ids = _encode(self.tokenizer, tool_result.stderr)

            def crop_text(original: str, ids: Sequence[int], keep: int) -> str:
                if keep >= len(ids):
                    return original
                prefix = _decode(self.tokenizer, ids[:keep]).rstrip()
                return prefix + _WORKING_MEMORY_V3_TOOL_TRUNCATION_MARKER

            def candidate(
                stdout_keep: int,
                stderr_keep: int,
                *,
                marker: bool = True,
            ) -> tuple[ToolResult, dict[str, Any]]:
                if marker:
                    stdout = crop_text(tool_result.stdout, stdout_ids, stdout_keep)
                    stderr = crop_text(tool_result.stderr, stderr_ids, stderr_keep)
                else:
                    stdout = _decode(self.tokenizer, stdout_ids[:stdout_keep])
                    stderr = _decode(self.tokenizer, stderr_ids[:stderr_keep])
                prepared = replace_streams(stdout, stderr)
                return prepared, self._project_next_round_prefix(
                    messages=messages,
                    tool_result=prepared,
                    episode_id=episode_id,
                    round_index=round_index,
                )

            full_projection = self._project_next_round_prefix(
                messages=messages,
                tool_result=tool_result,
                episode_id=episode_id,
                round_index=round_index,
            )
            projected_soft_memory_active = (
                int(full_projection["projected_memory_tokens"]) > 0
            )
            experimental_sglang_recovery = self._post_compression_generation_policy == (
                QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL
            )
            generation_reserve_tokens = (
                self._capacity_pressure_eviction_reserve_tokens
                if self._capacity_pressure_eviction_enabled
                else QWEN36_ROUND_GENERATION_RESERVE_TOKENS
            )
            if projected_soft_memory_active and not experimental_sglang_recovery:
                maximum_prefix_tokens = (
                    self.max_encoder_tokens - generation_reserve_tokens
                )
                admission_budget_policy = (
                    "checkpoint_recovery_headroom_after_soft_memory"
                )
            else:
                serving_limits = [
                    int(value)
                    for value in (
                        self.server_context_length,
                        self.server_max_total_num_tokens,
                    )
                    if value is not None
                ]
                if serving_limits:
                    maximum_prefix_tokens = (
                        min(serving_limits) - generation_reserve_tokens - 2
                    )
                    admission_budget_policy = (
                        "server_generation_headroom_after_soft_memory_experimental"
                        if projected_soft_memory_active
                        else "server_generation_headroom_before_first_compression"
                    )
                else:
                    # Fixed-cap unit/local callers may not have authoritative
                    # SGLang limits.  Retain the old conservative bound rather
                    # than claiming unavailable serving headroom.
                    maximum_prefix_tokens = (
                        self.max_encoder_tokens - generation_reserve_tokens
                    )
                    admission_budget_policy = (
                        "checkpoint_fallback_without_server_limits"
                    )
                if self.server_max_req_input_len is not None:
                    maximum_prefix_tokens = min(
                        maximum_prefix_tokens,
                        int(self.server_max_req_input_len) - 1,
                    )
                if maximum_prefix_tokens <= 0:
                    raise ValueError(
                        f"{self.model_display_name} serving context cannot reserve the next "
                        "generation: "
                        f"maximum_prefix_tokens={maximum_prefix_tokens}, "
                        "generation_reserve_tokens="
                        f"{generation_reserve_tokens}"
                    )
            selected = tool_result
            selected_projection = full_projection
            runtime_admission = getattr(self, "_runtime_admit_tool_output", None)
            if runtime_admission is not None:
                def fits_runtime_candidate(value):
                    projected = self._project_next_round_prefix(messages=messages,
                        tool_result=value, episode_id=episode_id, round_index=round_index)
                    return projected["prefix_tokens"] <= maximum_prefix_tokens
                selected = runtime_admission(tool_result, tokenizer=self.tokenizer,
                    fits=fits_runtime_candidate, marker=_WORKING_MEMORY_V3_TOOL_TRUNCATION_MARKER)
                selected_projection = self._project_next_round_prefix(messages=messages,
                    tool_result=selected, episode_id=episode_id, round_index=round_index)
            elif full_projection["prefix_tokens"] > maximum_prefix_tokens:
                stdout_only, stdout_only_projection = candidate(len(stdout_ids), 0)
                if stdout_only_projection["prefix_tokens"] <= maximum_prefix_tokens:
                    selected, selected_projection = stdout_only, stdout_only_projection
                    low, high = 0, len(stderr_ids)
                    while low <= high:
                        middle = (low + high) // 2
                        measured, projection = candidate(len(stdout_ids), middle)
                        if projection["prefix_tokens"] <= maximum_prefix_tokens:
                            selected, selected_projection = measured, projection
                            low = middle + 1
                        else:
                            high = middle - 1
                else:
                    low, high = 0, len(stdout_ids)
                    selected, selected_projection = candidate(0, 0, marker=False)
                    while low <= high:
                        middle = (low + high) // 2
                        measured, projection = candidate(middle, 0)
                        if projection["prefix_tokens"] <= maximum_prefix_tokens:
                            selected, selected_projection = measured, projection
                            low = middle + 1
                        else:
                            high = middle - 1
                if selected_projection["prefix_tokens"] > maximum_prefix_tokens:
                    selected, selected_projection = candidate(0, 0, marker=False)
                if selected_projection["prefix_tokens"] > maximum_prefix_tokens:
                    raise ValueError(
                        f"{self.model_display_name} assistant/tool metadata cannot fit the next "
                        "round even with empty stdout/stderr: "
                        f"prefix_tokens={selected_projection['prefix_tokens']}, "
                        f"maximum_prefix_tokens={maximum_prefix_tokens}"
                    )

            return selected, {
                "policy": (
                    QWEN36_ROUND_EXPERIMENTAL_TOOL_RESULT_FIT_POLICY
                    if experimental_sglang_recovery
                    else QWEN36_ROUND_TOOL_RESULT_FIT_POLICY
                ),
                "applied": selected != tool_result,
                "admission_budget_policy": admission_budget_policy,
                "post_compression_generation_policy": (
                    self._post_compression_generation_policy
                ),
                "untrained_long_generation_enabled": experimental_sglang_recovery,
                "projected_soft_memory_active": projected_soft_memory_active,
                "maximum_prefix_tokens": maximum_prefix_tokens,
                "max_encoder_tokens": self.max_encoder_tokens,
                "generation_reserve_tokens": generation_reserve_tokens,
                "serving_eviction_policy": selected_projection[
                    "serving_eviction_policy"
                ],
                "serving_eviction_accumulation_target_tokens": selected_projection[
                    "serving_eviction_accumulation_target_tokens"
                ],
                "serving_eviction_accumulation_threshold_reached": (
                    selected_projection[
                        "serving_eviction_accumulation_threshold_reached"
                    ]
                ),
                "serving_eviction_forced_subthreshold_flush": selected_projection[
                    "serving_eviction_forced_subthreshold_flush"
                ],
                "serving_eviction_balanced_chunk_token_lengths": (
                    selected_projection["serving_eviction_balanced_chunk_token_lengths"]
                ),
                "prefix_tokens_before": full_projection["prefix_tokens"],
                "prefix_tokens_after": selected_projection["prefix_tokens"],
                "projected_memory_tokens": selected_projection[
                    "projected_memory_tokens"
                ],
                "projected_raw_working_memory_tokens": selected_projection[
                    "projected_raw_working_memory_tokens"
                ],
                "projected_consumed_history_tokens": selected_projection[
                    "projected_consumed_history_tokens"
                ],
                "projected_consumed_rounds": selected_projection[
                    "projected_consumed_rounds"
                ],
                "nominal_raw_working_memory_window_count": selected_projection[
                    "nominal_raw_working_memory_window_count"
                ],
                "effective_raw_working_memory_window_count": selected_projection[
                    "effective_raw_working_memory_window_count"
                ],
                "oversized_complete_round_eviction_configured": (
                    selected_projection["oversized_complete_round_eviction_configured"]
                ),
                "oversized_complete_round_eviction_effective": (
                    selected_projection["oversized_complete_round_eviction_effective"]
                ),
                "oversized_complete_round_eviction_applied_this_round": (
                    selected_projection[
                        "oversized_complete_round_eviction_applied_this_round"
                    ]
                ),
                "oversized_complete_round_evicted_round_count": (
                    selected_projection["oversized_complete_round_evicted_round_count"]
                ),
                "oversized_complete_round_evicted_round_count_this_round": (
                    selected_projection[
                        "oversized_complete_round_evicted_round_count_this_round"
                    ]
                ),
                "oversized_complete_round_eviction_reason": selected_projection[
                    "oversized_complete_round_eviction_reason"
                ],
                "oversized_complete_round_evicted_round_indices": (
                    selected_projection[
                        "oversized_complete_round_evicted_round_indices"
                    ]
                ),
                "oversized_complete_round_evicted_round_indices_this_round": (
                    selected_projection[
                        "oversized_complete_round_evicted_round_indices_this_round"
                    ]
                ),
                "oversized_complete_round_authoritative_sglang_token_ceiling": (
                    selected_projection[
                        "oversized_complete_round_authoritative_sglang_token_ceiling"
                    ]
                ),
                "oversized_complete_round_threshold_tokens": selected_projection[
                    "oversized_complete_round_threshold_tokens"
                ],
                "oversized_complete_round_trigger_indices": selected_projection[
                    "oversized_complete_round_trigger_indices"
                ],
                "oversized_complete_round_trigger_token_lengths": (
                    selected_projection[
                        "oversized_complete_round_trigger_token_lengths"
                    ]
                ),
                "oversized_complete_round_latest_grace_considered": (
                    selected_projection[
                        "oversized_complete_round_latest_grace_considered"
                    ]
                ),
                "oversized_complete_round_latest_grace_effective": (
                    selected_projection[
                        "oversized_complete_round_latest_grace_effective"
                    ]
                ),
                "oversized_complete_round_latest_grace_round_index": (
                    selected_projection[
                        "oversized_complete_round_latest_grace_round_index"
                    ]
                ),
                "oversized_complete_round_latest_grace_reason": (
                    selected_projection["oversized_complete_round_latest_grace_reason"]
                ),
                "oversized_complete_round_latest_evicted_by_capacity_pressure": (
                    selected_projection[
                        "oversized_complete_round_latest_evicted_by_capacity_pressure"
                    ]
                ),
                "oversized_complete_round_latest_grace_maximum_prefix_tokens": (
                    selected_projection[
                        "oversized_complete_round_latest_grace_maximum_prefix_tokens"
                    ]
                ),
                "prefix_tokens_before_oversized_complete_round_eviction": (
                    selected_projection[
                        "prefix_tokens_before_oversized_complete_round_eviction"
                    ]
                ),
                "prefix_tokens_after_oversized_complete_round_eviction": (
                    selected_projection[
                        "prefix_tokens_after_oversized_complete_round_eviction"
                    ]
                ),
                "capacity_pressure_eviction_configured": selected_projection[
                    "capacity_pressure_eviction_configured"
                ],
                "capacity_pressure_eviction_effective": selected_projection[
                    "capacity_pressure_eviction_effective"
                ],
                "capacity_pressure_eviction_applied_this_round": (
                    selected_projection["capacity_pressure_eviction_applied_this_round"]
                ),
                "capacity_pressure_evicted_round_count": selected_projection[
                    "capacity_pressure_evicted_round_count"
                ],
                "capacity_pressure_evicted_round_count_this_round": (
                    selected_projection[
                        "capacity_pressure_evicted_round_count_this_round"
                    ]
                ),
                "capacity_pressure_eviction_reason": selected_projection[
                    "capacity_pressure_eviction_reason"
                ],
                "capacity_pressure_evicted_round_indices": selected_projection[
                    "capacity_pressure_evicted_round_indices"
                ],
                "capacity_pressure_evicted_round_indices_this_round": (
                    selected_projection[
                        "capacity_pressure_evicted_round_indices_this_round"
                    ]
                ),
                "capacity_pressure_reserve_tokens": selected_projection[
                    "capacity_pressure_reserve_tokens"
                ],
                "capacity_pressure_authoritative_sglang_token_ceiling": (
                    selected_projection[
                        "capacity_pressure_authoritative_sglang_token_ceiling"
                    ]
                ),
                "capacity_pressure_maximum_prefix_tokens": selected_projection[
                    "capacity_pressure_maximum_prefix_tokens"
                ],
                "prefix_tokens_before_capacity_pressure": selected_projection[
                    "prefix_tokens_before_capacity_pressure"
                ],
                "stdout_tokens_before": len(stdout_ids),
                "stdout_tokens_after": len(_encode(self.tokenizer, selected.stdout)),
                "stderr_tokens_before": len(stderr_ids),
                "stderr_tokens_after": len(_encode(self.tokenizer, selected.stderr)),
            }

    def clear_recursive_state(self) -> None:
        with self._state_lock:
            if self._memory_handle is not None:
                self._release_gpu_memory_handle(
                    self._memory_handle,
                    reason="clear_recursive_state",
                )
            self._episode_id = None
            self._raw_prefix_input_ids = None
            self._known_history_input_ids = ()
            self._known_history_message_end_tokens = ()
            self._incremental_history_receipt = None
            self._episode_incremental_history_receipt_enabled = False
            self._episode_native_history_policy = (
                NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER
            )
            self._consumed_history_tokens = 0
            self._consumed_round_count = 0
            self._memory = None
            self._memory_handle = None
            self._episode_memory_transport_policy = (
                MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1
            )
            self._state_transition_index = None
            self._capacity_pressure_consumed_round_count = 0
            self._capacity_pressure_evicted_round_indices = set()
            self._oversized_complete_round_consumed_round_count = 0
            self._oversized_complete_round_evicted_round_indices = set()
            self._episode_serving_eviction_policy = None
            self._episode_codex_native_compaction_enabled = False
            self._codex_native_compaction_target_completed_rounds = 0

    def _activate_episode(self, episode_id: str, round_index: int) -> None:
        if self._episode_id != str(episode_id) or round_index == 1:
            if self._memory_handle is not None:
                self._release_gpu_memory_handle(
                    self._memory_handle,
                    reason="episode_reset",
                )
            self._episode_id = str(episode_id)
            self._raw_prefix_input_ids = None
            self._known_history_input_ids = ()
            self._known_history_message_end_tokens = ()
            self._incremental_history_receipt = None
            self._episode_incremental_history_receipt_enabled = (
                self._incremental_history_receipt_enabled
            )
            self._episode_native_history_policy = self._native_history_policy
            self._consumed_history_tokens = 0
            self._consumed_round_count = 0
            self._memory = None
            self._memory_handle = None
            self._episode_memory_transport_policy = self._memory_transport_policy
            self._state_transition_index = None
            self._capacity_pressure_consumed_round_count = 0
            self._capacity_pressure_evicted_round_indices = set()
            self._oversized_complete_round_consumed_round_count = 0
            self._oversized_complete_round_evicted_round_indices = set()
            self._episode_serving_eviction_policy = self._serving_eviction_policy
            self._episode_codex_native_compaction_enabled = (
                self._codex_native_compaction_enabled
            )
            self._codex_native_compaction_target_completed_rounds = 0
            self._codex_native_summary_input_ids = ()

    @torch.inference_mode()
    def generate_next(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink] = None,
        raw_suffix_images: Sequence[R2EActRawSuffixImage | Mapping[str, Any]] = (),
    ) -> RolloutGeneration:
        with self._state_lock:
            return self._generate_next_locked(
                messages,
                episode_id=episode_id,
                round_index=round_index,
                event_sink=event_sink,
                raw_suffix_images=raw_suffix_images,
            )

    def _generate_next_locked(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink],
        raw_suffix_images: Sequence[R2EActRawSuffixImage | Mapping[str, Any]],
    ) -> RolloutGeneration:
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._activate_episode(episode_id, round_index)
        history_render = self._render_token_clock_history(
            messages,
            episode_id=str(episode_id),
            round_index=int(round_index),
        )
        guided_canonical = _message_pair_dicts(history_render.guided_messages)
        self._runtime_initial = guided_canonical[:2]
        self._runtime_images = tuple(raw_suffix_images)
        guidance = history_render.guidance
        raw_prefix = history_render.raw_prefix_input_ids
        full_discrete = history_render.full_discrete_input_ids
        history = history_render.history_input_ids
        if self._raw_prefix_input_ids is None:
            self._raw_prefix_input_ids = raw_prefix
        elif self._raw_prefix_input_ids != raw_prefix:
            raise RuntimeError(
                f"{self.model_display_name} token-clock P changed within the episode"
            )
        if (
            history[: len(self._known_history_input_ids)]
            != self._known_history_input_ids
        ):
            raise RuntimeError(
                f"{self.model_display_name} token-clock history is not append-only"
            )
        if self._consumed_history_tokens > len(history):
            raise RuntimeError(
                f"{self.model_display_name} token-clock consumed cursor exceeds history"
            )
        history_message_count = len(guided_canonical) - 2
        message_end_tokens = list(history_render.message_end_tokens)
        if len(message_end_tokens) > history_message_count:
            raise RuntimeError(
                f"{self.model_display_name} message-boundary receipt exceeds history"
            )
        if message_end_tokens and message_end_tokens[-1] != len(history):
            raise RuntimeError(
                f"{self.model_display_name} message boundaries do not cover history"
            )

        # Keep persistent P byte/token exact.  Dynamic turn guidance is placed
        # in native tool-response contents above, never in system/user P.
        guided_initial = [dict(message) for message in guided_canonical[:2]]
        if history_render.render_policy == QWEN38_INCREMENTAL_HISTORY_RECEIPT_POLICY:
            guided_prefix = list(raw_prefix)
        else:
            guided_prefix = self._apply_native_chat_ids(
                guided_initial,
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
        boundary = list(QWEN36_ASSISTANT_OPENING_IDS)
        if _encode(self.tokenizer, QWEN36_ASSISTANT_OPENING_TEXT) != boundary:
            raise ValueError(
                f"{self.model_display_name} tokenizer disagrees with its thinking assistant opening"
            )
        if history_render.render_policy == QWEN38_INCREMENTAL_HISTORY_RECEIPT_POLICY:
            native_prompt = list(full_discrete) + boundary
        else:
            native_prompt = self._apply_native_chat_ids(
                guided_canonical,
                tool_names=self._tool_names,
                add_generation_prompt=True,
            )
        if native_prompt != list(full_discrete) + boundary:
            raise ValueError(
                f"{self.model_display_name} native generation prompt does not equal history plus "
                "the thinking assistant opening"
            )

        roles = [str(message.get("role", "")) for message in guided_canonical[2:]]
        if len(roles) % 2:
            raise ValueError(
                f"{self.model_display_name} round-trigger history must end at a complete "
                "</tool_response> boundary"
            )
        for offset in range(0, len(roles), 2):
            pair_roles = roles[offset : offset + 2]
            if (
                len(pair_roles) != 2
                or pair_roles[0] != "assistant"
                or pair_roles[1] not in {"tool", "user"}
            ):
                raise ValueError(
                    f"{self.model_display_name} round-trigger history requires one atomic "
                    "assistant/(tool or user-feedback) group per ReAct round"
                )
        completed_round_count = len(roles) // 2
        if completed_round_count != int(round_index) - 1:
            raise ValueError(
                f"{self.model_display_name} round/history mismatch: "
                f"round_index={int(round_index)}, "
                f"completed_rounds={completed_round_count}"
            )
        (
            counted_completed_rounds,
            current_tool_calls,
            current_user_feedback_rounds,
        ) = _observation_role_counts(guided_canonical)
        if counted_completed_rounds != completed_round_count:
            raise RuntimeError(
                f"{self.model_display_name} observation-role count drifted"
            )
        round_end_tokens = [
            int(message_end_tokens[(round_number * 2) + 1])
            for round_number in range(completed_round_count)
        ]
        expected_consumed = (
            0
            if self._consumed_round_count == 0
            else round_end_tokens[self._consumed_round_count - 1]
        )
        if self._consumed_history_tokens != expected_consumed:
            raise RuntimeError(
                f"{self.model_display_name} compressed cursor is not on a </tool_response> "
                "round boundary"
            )

        memory = self._memory
        memory_handle = self._memory_handle
        gpu_memory_transport = self._episode_memory_transport_policy == (
            MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1
        )
        if memory is not None and memory_handle is not None:
            raise RuntimeError(
                f"{self.model_display_name} has both CPU memory and a GPU handle"
            )
        if gpu_memory_transport and memory is not None:
            raise RuntimeError(
                f"{self.model_display_name} GPU transport cannot consume legacy CPU memory"
            )
        if not gpu_memory_transport and memory_handle is not None:
            raise RuntimeError(
                f"{self.model_display_name} legacy transport cannot consume a GPU handle"
            )
        consumed = self._consumed_history_tokens
        consumed_round_count = self._consumed_round_count
        transition_index = self._state_transition_index
        planned_memory_tokens = (
            int(memory.shape[0])
            if memory is not None
            else (0 if memory_handle is None else memory_handle.shape[0])
        )
        previous_memory_tokens = planned_memory_tokens
        compression_seconds = 0.0
        transitions_applied = 0
        compressed_round_indices: list[int] = []
        source_token_lengths: list[int] = []
        source_encoder_token_lengths: list[int] = []
        slot_counts: list[int] = []
        last_physical_source_tokens = 0
        last_source_encoder_tokens = 0
        pending_gpu_transitions: list[dict[str, Any]] = []
        eviction_plan = self._plan_round_evictions(
            raw_prefix_tokens=len(raw_prefix),
            history_tokens=len(history),
            round_end_tokens=round_end_tokens,
            completed_round_count=completed_round_count,
            generation_boundary_tokens=(
                len(boundary) + len(getattr(self, "_codex_native_summary_input_ids", ()))
            ),
        )
        target_consumed_round_count = int(
            eviction_plan["effective_target_consumed_rounds"]
        )

        def encode_contiguous_block(
            block_tokens: int,
            *,
            planned_transition: Optional[Mapping[str, Any]] = None,
        ) -> None:
            nonlocal memory, consumed, transition_index, compression_seconds
            nonlocal transitions_applied, last_physical_source_tokens
            nonlocal last_source_encoder_tokens, planned_memory_tokens

            current_memory_tokens = planned_memory_tokens
            source_capacity = (
                self.max_encoder_tokens - len(raw_prefix) - current_memory_tokens
            )
            if source_capacity <= 0 or not 0 < block_tokens <= source_capacity:
                raise ValueError(
                    f"{self.model_display_name} round-trigger source has no capacity for the "
                    "next completed-round chunk: "
                    f"P={len(raw_prefix)}, M_prev={current_memory_tokens}, "
                    f"block={block_tokens}, max_encoder_tokens={self.max_encoder_tokens}"
                )
            block = list(history[consumed : consumed + block_tokens])
            if len(block) != block_tokens or not block:
                raise RuntimeError(
                    f"{self.model_display_name} round-trigger chunk partition is inconsistent"
                )
            source_ids = list(raw_prefix)
            memory_positions: Optional[list[int]] = None
            if current_memory_tokens:
                memory_positions = list(
                    range(
                        len(raw_prefix),
                        len(raw_prefix) + current_memory_tokens,
                    )
                )
                source_ids.extend(
                    [int(self.tokenizer.pad_token_id)] * current_memory_tokens
                )
            source_ids.extend(block)
            physical_source_tokens = current_memory_tokens + len(block)
            source_encoder_tokens = len(source_ids)
            expected_slots = math.ceil(physical_source_tokens / self.compression_ratio)
            if planned_transition is not None:
                expected_plan = {
                    "history_start": consumed,
                    "history_end": consumed + block_tokens,
                    "raw_chunk_tokens": block_tokens,
                    "memory_tokens_before": current_memory_tokens,
                    "memory_tokens_after": expected_slots,
                    "source_encoder_tokens": source_encoder_tokens,
                }
                if any(
                    int(planned_transition.get(key, -1)) != value
                    for key, value in expected_plan.items()
                ):
                    raise RuntimeError(
                        f"{self.model_display_name} accumulated executor diverged from its episode snapshot"
                    )
            if gpu_memory_transport:
                pending_gpu_transitions.append(
                    {
                        "transition_index": len(pending_gpu_transitions),
                        "source_input_ids": source_ids,
                        "evidence_spans": [
                            [len(raw_prefix), source_encoder_tokens]
                        ],
                        "source_token_lengths": [physical_source_tokens],
                        "expected_slot_count": expected_slots,
                        # qwen36_token_clock_v1 has no hook-wire source
                        # discriminator.  The generator's round-recursive
                        # contract remains public telemetry/preflight metadata,
                        # but sending that label in encode custom params is an
                        # ABI violation (legacy CPU transitions omit it too).
                        "recursive_source_contract": None,
                    }
                )
            else:
                started = time.perf_counter()
                next_memory = self.client.encode_memory(
                    source_ids,
                    evidence_spans=[[len(raw_prefix), source_encoder_tokens]],
                    source_token_lengths=[physical_source_tokens],
                    expected_slot_count=expected_slots,
                    recursive=True,
                    memory_positions=memory_positions,
                    memory_embeddings=memory,
                    expected_hidden_size=self.target_hidden_size,
                )
                compression_seconds += time.perf_counter() - started
                if next_memory.shape != (
                    expected_slots,
                    self.target_hidden_size,
                ):
                    raise RuntimeError(
                        f"{self.model_display_name} compressor returned an invalid memory shape"
                    )
                memory = next_memory.detach().cpu().clone()
            planned_memory_tokens = expected_slots
            consumed += block_tokens
            transition_index = 0 if transition_index is None else transition_index + 1
            transitions_applied += 1
            last_physical_source_tokens = physical_source_tokens
            last_source_encoder_tokens = source_encoder_tokens
            source_token_lengths.append(physical_source_tokens)
            source_encoder_token_lengths.append(source_encoder_tokens)
            slot_counts.append(expected_slots)

        effective_policy = str(eviction_plan["serving_eviction_policy"])
        if effective_policy == QWEN38_8192_ACCUMULATION_SERVING_EVICTION_POLICY:
            snapshot = eviction_plan.get("effective_episode_snapshot")
            if not isinstance(snapshot, Mapping):
                raise RuntimeError(
                    f"{self.model_display_name} accumulation plan has no episode snapshot"
                )
            planned_transitions = snapshot.get("transitions")
            if not isinstance(planned_transitions, list):
                raise RuntimeError(
                    f"{self.model_display_name} accumulation plan has no transition list"
                )
            initial_consumed_round_count = consumed_round_count
            for planned_transition in planned_transitions:
                if not isinstance(planned_transition, Mapping):
                    raise RuntimeError(
                        f"{self.model_display_name} accumulation transition is invalid"
                    )
                encode_contiguous_block(
                    int(planned_transition.get("raw_chunk_tokens", 0)),
                    planned_transition=planned_transition,
                )
            consumed_round_count = target_consumed_round_count
            compressed_round_indices.extend(
                range(initial_consumed_round_count + 1, consumed_round_count + 1)
            )
        else:
            # Preserve the checkpoint-native per-round 1,024-token execution
            # byte-for-byte unless the explicit Qwen3.8 canary is selected.
            for evicted_round_zero_based in range(
                consumed_round_count, target_consumed_round_count
            ):
                round_start = (
                    0
                    if evicted_round_zero_based == 0
                    else round_end_tokens[evicted_round_zero_based - 1]
                )
                round_end = round_end_tokens[evicted_round_zero_based]
                if consumed != round_start or round_end <= round_start:
                    raise RuntimeError(
                        f"{self.model_display_name} round-trigger cursor lost a complete round boundary"
                )
                while consumed < round_end:
                    current_memory_tokens = planned_memory_tokens
                    source_capacity = (
                        self.max_encoder_tokens
                        - len(raw_prefix)
                        - current_memory_tokens
                    )
                    block_tokens = min(
                        round_end - consumed,
                        self.checkpoint_eviction_block_tokens,
                        source_capacity,
                    )
                    encode_contiguous_block(block_tokens)
                consumed_round_count = evicted_round_zero_based + 1
                compressed_round_indices.append(consumed_round_count)

        def materialize_pending_gpu_memory() -> None:
            nonlocal memory_handle, compression_seconds
            if not pending_gpu_transitions:
                return
            operation_seed = (
                f"{episode_id}\0{int(round_index)}\0"
                f"{self._state_transition_index}\0"
                f"{self._consumed_history_tokens}\0{consumed}"
            )
            operation_id = "r2mem_" + hashlib.sha256(
                operation_seed.encode("utf-8")
            ).hexdigest()
            started = time.perf_counter()
            if len(pending_gpu_transitions) == 1:
                transition = pending_gpu_transitions[0]
                evidence_start = int(transition["evidence_spans"][0][0])
                parent_slots = (
                    0 if memory_handle is None else memory_handle.shape[0]
                )
                memory_positions = (
                    None
                    if memory_handle is None
                    else list(
                        range(evidence_start, evidence_start + parent_slots)
                    )
                )
                memory_handle = self.client.encode_memory_handle(
                    transition["source_input_ids"],
                    evidence_spans=transition["evidence_spans"],
                    source_token_lengths=transition["source_token_lengths"],
                    operation_id=operation_id,
                    expected_slot_count=transition["expected_slot_count"],
                    recursive_source_contract=transition[
                        "recursive_source_contract"
                    ],
                    memory_positions=memory_positions,
                    parent_memory_handle=memory_handle,
                    expected_hidden_size=self.target_hidden_size,
                )
            else:
                chain_result = self.client.encode_recursive_memory_chain(
                    pending_gpu_transitions,
                    operation_id=operation_id,
                    compression_ratio=self.compression_ratio,
                    expected_hidden_size=self.target_hidden_size,
                    parent_memory_handle=memory_handle,
                )
                memory_handle = chain_result.final_memory_handle
            compression_seconds += time.perf_counter() - started
            if memory_handle.shape != (
                planned_memory_tokens,
                self.target_hidden_size,
            ):
                raise RuntimeError(
                    f"{self.model_display_name} compressor returned an invalid GPU memory handle shape"
                )

        raw_suffix = list(history[consumed:])
        memory_tokens = planned_memory_tokens
        has_memory = memory_tokens > 0
        prefix_ids = list(guided_prefix)
        generation_memory_positions: list[int] = []
        if has_memory:
            generation_memory_positions = list(
                range(len(guided_prefix), len(guided_prefix) + memory_tokens)
            )
            prefix_ids.extend([int(self.tokenizer.pad_token_id)] * memory_tokens)
        prefix_ids.extend(getattr(self, "_codex_native_summary_input_ids", ()))
        prefix_ids.extend(raw_suffix)
        retained_images = _select_raw_suffix_images(
            raw_suffix_images,
            consumed_round_count=consumed_round_count,
            completed_round_count=completed_round_count,
        )
        image_suffix_ids: list[int] = []
        if retained_images:
            if self.target_family != "qwen38":
                raise ValueError("raw-suffix images currently require Qwen3.8")
            missing_multimodal_limits = [
                name
                for name, value in (
                    ("server_context_length", self.server_context_length),
                    (
                        "server_max_total_num_tokens",
                        self.server_max_total_num_tokens,
                    ),
                    ("server_max_req_input_len", self.server_max_req_input_len),
                )
                if value is None
            ]
            if missing_multimodal_limits:
                raise ValueError(
                    "raw-suffix image context budgeting requires authoritative "
                    "SGLang limits; missing " + ", ".join(missing_multimodal_limits)
                )
            visual_content: list[dict[str, str]] = []
            for image_index, image in enumerate(retained_images, start=1):
                visual_content.extend(
                    (
                        {
                            "type": "text",
                            "text": (
                                "Retained visual tool observation from completed "
                                f"round {image.completed_round_index}, image "
                                f"{image_index}:"
                            ),
                        },
                        {
                            "type": "image",
                            # The tokenizer only renders one native placeholder;
                            # bytes stay on SGLang's image_data wire field.
                            "image": f"request-local://image/{image_index}",
                        },
                    )
                )
            initial_ids = self._apply_native_chat_ids(
                guided_initial,
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
            if initial_ids != list(guided_prefix):
                raise RuntimeError(
                    f"{self.model_display_name} multimodal suffix changed persistent P"
                )
            visual_ids = self._apply_native_chat_ids(
                guided_initial + [{"role": "user", "content": visual_content}],
                tool_names=self._tool_names,
                add_generation_prompt=False,
            )
            if visual_ids[: len(initial_ids)] != initial_ids:
                raise RuntimeError(
                    f"{self.model_display_name} multimodal user turn is not suffix-stable"
                )
            image_suffix_ids = visual_ids[len(initial_ids) :]
            image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            if (
                isinstance(image_pad_id, bool)
                or not isinstance(image_pad_id, int)
                or image_suffix_ids.count(image_pad_id) != len(retained_images)
            ):
                raise RuntimeError(
                    f"{self.model_display_name} multimodal suffix did not render exactly "
                    "one image placeholder per attachment"
                )
            # Appending the request-local visual turn *after* every memory slot
            # is the key coordinate invariant: server-side placeholder expansion
            # cannot shift the already-declared soft-memory positions.
            prefix_ids.extend(image_suffix_ids)
        prefix_ids.extend(boundary)
        raw_suffix_image_context_reserve_tokens = (
            self._image_token_budget.reserve(retained_images)
            if getattr(self, "_image_token_budget", None) is not None
            else _raw_suffix_image_context_reserve_tokens(len(retained_images))
        )
        # ``prefix_ids`` contains one unexpanded image-pad token per image.
        # SGLang validates context length only after Qwen3VLProcessor replaces
        # each placeholder with its image grid.  Budget against the audited
        # maximum expansion here; the full 16,384-token reserve (rather than
        # the exact 16,383-token delta) deliberately leaves one extra token of
        # headroom per retained image.
        server_budget_prefix_tokens = (
            len(prefix_ids) + raw_suffix_image_context_reserve_tokens
        )
        if consumed != int(eviction_plan["projected_consumed_history_tokens"]):
            raise RuntimeError(
                f"{self.model_display_name} round eviction diverged from its projected cursor"
            )
        if memory_tokens != int(eviction_plan["projected_memory_tokens"]):
            raise RuntimeError(
                f"{self.model_display_name} round eviction diverged from its projected memory size"
            )
        if len(prefix_ids) - len(image_suffix_ids) != int(
            eviction_plan["prefix_tokens"]
        ):
            raise RuntimeError(
                f"{self.model_display_name} recovery prefix diverged from its eviction projection"
            )
        experimental_sglang_recovery = self._post_compression_generation_policy == (
            QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL
        )
        checkpoint_recovery_cap_enforced = (
            has_memory and not experimental_sglang_recovery
        )
        if checkpoint_recovery_cap_enforced and len(prefix_ids) >= (
            self.max_encoder_tokens
        ):
            raise ValueError(
                f"{self.model_display_name} round-trigger recovery prefix exhausts "
                f"the trained {self.max_encoder_tokens}-token cap while preserving "
                "the requested raw rounds: "
                f"prefix_tokens={len(prefix_ids)}"
            )
        # Before the first compression this arm is still a pure discrete target
        # prefill, so the compressor checkpoint's training envelope
        # does not apply.  Once soft memory is present, the default preserves
        # the trained prefix+continuation envelope fail-closed.  The explicit
        # Streamlit canary instead uses the authoritative SGLang window; encode
        # still follows the checkpoint's unchanged source-cap/1,024-block ABI.
        if not has_memory or experimental_sglang_recovery:
            generation_max_new_tokens = _generation_token_budget(
                prefix_token_count=server_budget_prefix_tokens,
                configured_max_new_tokens=self.max_new_tokens,
                server_context_length=self.server_context_length,
                server_max_total_num_tokens=self.server_max_total_num_tokens,
                server_max_req_input_len=self.server_max_req_input_len,
                use_remaining_context=self.use_remaining_context_for_generation,
                remaining_context_output_ceiling_tokens=(
                    self.remaining_context_output_ceiling_tokens
                ),
            )
            if has_memory:
                generation_budget_policy = (
                    "sglang_remaining_context_after_soft_memory_experimental"
                    + (
                        "_capped_to_output_ceiling"
                        if self.remaining_context_output_ceiling_tokens is not None
                        else ""
                    )
                )
            else:
                generation_budget_policy = (
                    (
                        "remaining_context_after_input_capped_to_output_ceiling"
                        if self.remaining_context_output_ceiling_tokens is not None
                        else "remaining_context_after_input"
                    )
                    if self.use_remaining_context_for_generation
                    else "configured_cap_clamped_to_remaining_context"
                )
        else:
            generation_max_new_tokens = _working_memory_v3_generation_token_budget(
                prefix_token_count=len(prefix_ids),
                server_prefix_token_count=server_budget_prefix_tokens,
                max_encoder_tokens=self.max_encoder_tokens,
                configured_max_new_tokens=self.max_new_tokens,
                server_context_length=self.server_context_length,
                server_max_total_num_tokens=self.server_max_total_num_tokens,
                server_max_req_input_len=self.server_max_req_input_len,
                use_remaining_context=self.use_remaining_context_for_generation,
            )
            generation_budget_policy = (
                f"checkpoint_{self.max_encoder_tokens}_cap_and_remaining_server_context"
                if self.use_remaining_context_for_generation
                else "configured_cap_clamped_to_checkpoint_and_server_context"
            )
        compressed_message_indices = list(range(2, 2 + (consumed_round_count * 2)))
        archived_message_indices = list(compressed_message_indices)
        raw_round_indices = list(
            range(consumed_round_count + 1, completed_round_count + 1)
        )
        compression: dict[str, Any] = {
            "backend": self.backend,
            "mode": self.telemetry_mode,
            "context_organization": self.context_organization,
            "checkpoint_context_organization": self.checkpoint_context_organization,
            "checkpoint_schema": self.checkpoint_schema,
            "target_family": self.target_family,
            "native_history_policy": self._episode_native_history_policy,
            "native_history_render_policy": history_render.render_policy,
            "native_history_incremental_receipt_active": (
                history_render.render_policy
                == QWEN38_INCREMENTAL_HISTORY_RECEIPT_POLICY
            ),
            "native_history_appended_message_count": (
                history_render.appended_message_count
            ),
            "native_history_template_sha256": (
                history_render.staged_receipt.template_sha256
                if history_render.staged_receipt is not None
                else None
            ),
            "compression_ratio": self.compression_ratio,
            **(
                {
                    "checkpoint_training_stage": "epoch_1_one_shot_only",
                    "recursive_serving_status": (
                        "exploratory_out_of_training_distribution"
                    ),
                }
                if self.target_family == "qwen36"
                else {}
            ),
            "recursive_working_memory_enabled": True,
            "recursive_working_memory_policy": self.recursive_working_memory_policy,
            "recursive_state_policy": self.recursive_state_policy,
            "recursive_source_contract": self.recursive_source_contract,
            "memory_transport_policy": self._episode_memory_transport_policy,
            "memory_transport_gpu_handle_active": (
                gpu_memory_transport and has_memory
            ),
            "serving_eviction_policy": eviction_plan["serving_eviction_policy"],
            **(
                {
                    "codex_native_compaction_enabled": True,
                    "codex_native_compaction_policy": eviction_plan[
                        "codex_native_compaction_policy"
                    ],
                    "round_working_memory_policy_active": False,
                    "codex_native_compaction_staged_target_completed_rounds": (
                        eviction_plan[
                            "codex_native_compaction_staged_target_completed_rounds"
                        ]
                    ),
                    "codex_native_compaction_target_pending_before_plan": (
                        eviction_plan[
                            "codex_native_compaction_target_pending_before_plan"
                        ]
                    ),
                    "codex_native_compaction_target_applied_this_round": (
                        eviction_plan[
                            "codex_native_compaction_target_applied_this_round"
                        ]
                    ),
                    "codex_native_compaction_target_consumed_rounds": (
                        eviction_plan[
                            "codex_native_compaction_target_consumed_rounds"
                        ]
                    ),
                    "codex_native_compaction_source_tokens_this_round": (
                        eviction_plan[
                            "codex_native_compaction_source_tokens_this_round"
                        ]
                    ),
                    "codex_native_compaction_forced_subthreshold_flush": (
                        eviction_plan[
                            "codex_native_compaction_forced_subthreshold_flush"
                        ]
                    ),
                    "codex_native_compaction_forced_flush_reason": (
                        eviction_plan[
                            "codex_native_compaction_forced_flush_reason"
                        ]
                    ),
                }
                if self._episode_codex_native_compaction_enabled
                else {}
            ),
            "serving_eviction_accumulation_target_tokens": eviction_plan[
                "serving_eviction_accumulation_target_tokens"
            ],
            "serving_eviction_accumulated_eligible_tokens_before": eviction_plan[
                "serving_eviction_accumulated_eligible_tokens_before"
            ],
            "serving_eviction_accumulation_threshold_reached": eviction_plan[
                "serving_eviction_accumulation_threshold_reached"
            ],
            "serving_eviction_accumulation_flush_triggered": eviction_plan[
                "serving_eviction_accumulation_flush_triggered"
            ],
            "serving_eviction_forced_subthreshold_flush": eviction_plan[
                "serving_eviction_forced_subthreshold_flush"
            ],
            "serving_eviction_forced_flush_reasons": eviction_plan[
                "serving_eviction_forced_flush_reasons"
            ],
            "serving_eviction_accumulated_source_tokens": eviction_plan[
                "serving_eviction_accumulated_source_tokens"
            ],
            "serving_eviction_staged_eligible_tokens_after": eviction_plan[
                "serving_eviction_staged_eligible_tokens_after"
            ],
            "serving_eviction_balanced_chunk_token_lengths": eviction_plan[
                "serving_eviction_balanced_chunk_token_lengths"
            ],
            "enable_thinking": self.enable_thinking,
            **_token_clock_sampling_telemetry(self.generation_sampling_policy),
            "transition_index": transition_index,
            "transitions_applied_this_round": transitions_applied,
            "raw_prefix_tokens": len(raw_prefix),
            "previous_memory_tokens": previous_memory_tokens,
            "physical_source_tokens": last_physical_source_tokens,
            "source_encoder_tokens": last_source_encoder_tokens,
            "memory_slots": memory_tokens,
            "history_tokens": len(history),
            "consumed_history_tokens": consumed,
            "consumed_round_count": consumed_round_count,
            "completed_round_count": completed_round_count,
            "compressed_message_indices": compressed_message_indices,
            "compressed_react_round_indices": compressed_round_indices,
            "archived_message_indices": archived_message_indices,
            "memory_archive_policy": ("full_raw_rounds_after_recursive_compression_v1"),
            "memory_archive_filename": "memory.jsonl",
            "latest_raw_working_memory_tokens": len(raw_suffix),
            "raw_working_memory_window_count": len(raw_round_indices),
            "raw_working_memory_window_tokens": len(raw_suffix),
            "raw_working_memory_window_indices": raw_round_indices,
            "raw_suffix_image_policy": (
                R2EACT_RAW_SUFFIX_IMAGE_POLICY if retained_images else None
            ),
            "raw_suffix_image_count": len(retained_images),
            "raw_suffix_image_round_indices": [
                image.completed_round_index for image in retained_images
            ],
            "raw_suffix_image_placeholder_tokens": len(image_suffix_ids),
            "working_memory_turns": (
                None
                if self._episode_codex_native_compaction_enabled
                else self._working_memory_turns
            ),
            "configured_working_memory_turns": (
                None
                if self._episode_codex_native_compaction_enabled
                else self._working_memory_turns
            ),
            "effective_working_memory_turns": eviction_plan[
                "effective_raw_working_memory_window_count"
            ],
            "nominal_raw_working_memory_window_count": eviction_plan[
                "nominal_raw_working_memory_window_count"
            ],
            "effective_raw_working_memory_window_count": eviction_plan[
                "effective_raw_working_memory_window_count"
            ],
            "oversized_complete_round_eviction_configured": eviction_plan[
                "oversized_complete_round_eviction_configured"
            ],
            "oversized_complete_round_eviction_effective": eviction_plan[
                "oversized_complete_round_eviction_effective"
            ],
            "oversized_complete_round_eviction_applied_this_round": eviction_plan[
                "oversized_complete_round_eviction_applied_this_round"
            ],
            "oversized_complete_round_evicted_round_count": eviction_plan[
                "oversized_complete_round_evicted_round_count"
            ],
            "oversized_complete_round_evicted_round_count_this_round": (
                eviction_plan["oversized_complete_round_evicted_round_count_this_round"]
            ),
            "oversized_complete_round_eviction_reason": eviction_plan[
                "oversized_complete_round_eviction_reason"
            ],
            "oversized_complete_round_evicted_round_indices": eviction_plan[
                "oversized_complete_round_evicted_round_indices"
            ],
            "oversized_complete_round_evicted_round_indices_this_round": (
                eviction_plan[
                    "oversized_complete_round_evicted_round_indices_this_round"
                ]
            ),
            "oversized_complete_round_authoritative_sglang_token_ceiling": (
                eviction_plan[
                    "oversized_complete_round_authoritative_sglang_token_ceiling"
                ]
            ),
            "oversized_complete_round_threshold_tokens": eviction_plan[
                "oversized_complete_round_threshold_tokens"
            ],
            "oversized_complete_round_trigger_indices": eviction_plan[
                "oversized_complete_round_trigger_indices"
            ],
            "oversized_complete_round_trigger_token_lengths": eviction_plan[
                "oversized_complete_round_trigger_token_lengths"
            ],
            "oversized_complete_round_latest_grace_considered": eviction_plan[
                "oversized_complete_round_latest_grace_considered"
            ],
            "oversized_complete_round_latest_grace_effective": eviction_plan[
                "oversized_complete_round_latest_grace_effective"
            ],
            "oversized_complete_round_latest_grace_round_index": eviction_plan[
                "oversized_complete_round_latest_grace_round_index"
            ],
            "oversized_complete_round_latest_grace_reason": eviction_plan[
                "oversized_complete_round_latest_grace_reason"
            ],
            "oversized_complete_round_latest_evicted_by_capacity_pressure": (
                eviction_plan[
                    "oversized_complete_round_latest_evicted_by_capacity_pressure"
                ]
            ),
            "oversized_complete_round_latest_grace_maximum_prefix_tokens": (
                eviction_plan[
                    "oversized_complete_round_latest_grace_maximum_prefix_tokens"
                ]
            ),
            "prefix_tokens_before_oversized_complete_round_eviction": (
                eviction_plan["prefix_tokens_before_oversized_complete_round_eviction"]
            ),
            "prefix_tokens_after_oversized_complete_round_eviction": (
                eviction_plan["prefix_tokens_after_oversized_complete_round_eviction"]
            ),
            "capacity_pressure_eviction_configured": eviction_plan[
                "capacity_pressure_eviction_configured"
            ],
            "capacity_pressure_eviction_effective": eviction_plan[
                "capacity_pressure_eviction_effective"
            ],
            "capacity_pressure_eviction_applied_this_round": eviction_plan[
                "capacity_pressure_eviction_applied_this_round"
            ],
            "capacity_pressure_evicted_round_count": eviction_plan[
                "capacity_pressure_evicted_round_count"
            ],
            "capacity_pressure_evicted_round_count_this_round": eviction_plan[
                "capacity_pressure_evicted_round_count_this_round"
            ],
            "capacity_pressure_eviction_reason": eviction_plan[
                "capacity_pressure_eviction_reason"
            ],
            "capacity_pressure_evicted_round_indices": eviction_plan[
                "capacity_pressure_evicted_round_indices"
            ],
            "capacity_pressure_evicted_round_indices_this_round": eviction_plan[
                "capacity_pressure_evicted_round_indices_this_round"
            ],
            "capacity_pressure_reserve_tokens": eviction_plan[
                "capacity_pressure_reserve_tokens"
            ],
            "capacity_pressure_authoritative_sglang_token_ceiling": (
                eviction_plan["capacity_pressure_authoritative_sglang_token_ceiling"]
            ),
            "capacity_pressure_maximum_prefix_tokens": eviction_plan[
                "capacity_pressure_maximum_prefix_tokens"
            ],
            "prefix_tokens_before_capacity_pressure": eviction_plan[
                "prefix_tokens_before_capacity_pressure"
            ],
            "working_memory_turns_checkpoint_native": False,
            "generation_reserve_tokens": (
                self._capacity_pressure_eviction_reserve_tokens
                if self._capacity_pressure_eviction_enabled
                else QWEN36_ROUND_GENERATION_RESERVE_TOKENS
            ),
            "tool_result_admission_policy": (
                QWEN36_ROUND_EXPERIMENTAL_TOOL_RESULT_FIT_POLICY
                if experimental_sglang_recovery
                else QWEN36_ROUND_TOOL_RESULT_FIT_POLICY
            ),
            "eviction_block_tokens": self.eviction_block_tokens,
            "checkpoint_eviction_block_tokens": (self.checkpoint_eviction_block_tokens),
            "configured_raw_working_memory_tokens": self.raw_working_memory_tokens,
            "configured_compression_trigger_tokens": (self.compression_trigger_tokens),
            **(
                {
                    "token_clock_override_enabled": True,
                    "token_clock_override_policy": (
                        "round_trigger_serving_ablation_v1"
                    ),
                }
                if self.target_family == "qwen36"
                else {}
            ),
            "working_memory_tokens": None,
            "compression_trigger_tokens": None,
            "effective_working_memory_tokens": None,
            "effective_compression_trigger_tokens": None,
            "working_memory_token_trigger_enabled": False,
            "working_memory_trigger_policy": (
                CODEX_NATIVE_COMPACTION_POLICY
                if self._episode_codex_native_compaction_enabled
                else QWEN36_ROUND_WORKING_MEMORY_POLICY
            ),
            "working_memory_window_policy": (
                "codex_native_compaction_frontier_v1"
                if self._episode_codex_native_compaction_enabled
                else "latest_k_complete_react_rounds_v1"
            ),
            "round_trigger_boundary": "complete_assistant_observation_pair",
            "compression_source_policy": (
                "full_native_assistant_observation_history_blocks_v1"
            ),
            "completed_observation_round_count": completed_round_count,
            "tool_observation_count": current_tool_calls,
            "user_feedback_observation_count": current_user_feedback_rounds,
            "compression_triggered": transitions_applied > 0,
            "compression_trigger_reason": (
                _CODEX_NATIVE_COMPACTION_FORCED_FLUSH_REASON
                if eviction_plan.get(
                    "codex_native_compaction_target_applied_this_round", False
                )
                else "completed_assistant_observation_round_boundary"
                if compressed_round_indices
                else None
            ),
            "compression_trigger_reasons": (
                [_CODEX_NATIVE_COMPACTION_FORCED_FLUSH_REASON]
                if eviction_plan.get(
                    "codex_native_compaction_target_applied_this_round", False
                )
                else ["completed_assistant_observation_round_boundary"]
                if compressed_round_indices
                else []
            ),
            "context_trigger_tokens_before": None,
            "context_trigger_tokens_after": None,
            "left_context_tokens_before": None,
            "left_context_tokens_after": None,
            "protected_raw_working_memory_tokens_before": None,
            "protected_raw_working_memory_tokens_after": len(raw_suffix),
            "unprotected_raw_working_memory_tokens_before": None,
            "unprotected_raw_working_memory_tokens_after": 0,
            "compressed_transition_count": transitions_applied,
            "compressed_round_count_this_generation": len(compressed_round_indices),
            "source_token_lengths": source_token_lengths,
            "source_encoder_token_lengths": source_encoder_token_lengths,
            "slot_counts": slot_counts,
            "student_prefix_tokens": len(prefix_ids),
            "generation_boundary_tokens": len(boundary),
            "generation_user_suffix_tokens": len(guided_prefix) - len(raw_prefix),
            "tool_call_turn_guidance_policy": (
                QWEN36_TOOL_RESPONSE_TURN_GUIDANCE_POLICY
                if self._max_turns is not None
                else None
            ),
            "tool_call_turn_guidance_placement": (
                "before_each_native_tool_response_close"
                if self._max_turns is not None
                else None
            ),
            "tool_call_turn_guidance_max_turns": self._max_turns,
            "tool_call_turn_guidance_max_tool_calls": (
                None if self._max_turns is None else self._max_turns - 1
            ),
            "tool_call_turn_guidance_current_tool_calls": (
                None if self._max_turns is None else current_tool_calls
            ),
            "tool_call_turn_guidance_action": (
                None
                if self._max_turns is None
                else (
                    "continue"
                    if int(round_index) < self._max_turns
                    and current_tool_calls < self._max_turns - 1
                    else "answer"
                )
            ),
            "source_compression_request_seconds": _finite(compression_seconds),
            "target_generation_seconds": None,
            "max_encoder_tokens": self.max_encoder_tokens,
            "server_context_length": self.server_context_length,
            "server_max_total_num_tokens": self.server_max_total_num_tokens,
            "server_max_req_input_len": self.server_max_req_input_len,
            "configured_max_new_tokens": self.max_new_tokens,
            "remaining_context_output_ceiling_tokens": (
                self.remaining_context_output_ceiling_tokens
            ),
            "generation_max_new_tokens": generation_max_new_tokens,
            "soft_memory_active": has_memory,
            "post_compression_generation_policy": (
                self._post_compression_generation_policy
            ),
            "trained_recovery_cap_tokens": self.max_encoder_tokens,
            "checkpoint_recovery_cap_enforced": (checkpoint_recovery_cap_enforced),
            "checkpoint_recovery_prefix_cap_enforced": (
                checkpoint_recovery_cap_enforced
            ),
            "checkpoint_recovery_generation_cap_enforced": (
                checkpoint_recovery_cap_enforced
            ),
            "checkpoint_prefix_plus_generation_cap_enforced": (
                checkpoint_recovery_cap_enforced
            ),
            "untrained_long_generation_enabled": experimental_sglang_recovery,
            "untrained_long_generation_active": (
                has_memory and experimental_sglang_recovery
            ),
            "generation_budget_policy": generation_budget_policy,
            "generation_budget_contract_policy": (
                (
                    "remaining_sglang_context_with_output_ceiling_v1"
                    if self.remaining_context_output_ceiling_tokens is not None
                    else QWEN36_ROUND_EXPERIMENTAL_GENERATION_BUDGET_POLICY
                )
                if experimental_sglang_recovery
                else (
                    (
                        "remaining_sglang_context_with_output_ceiling_v1"
                        if self.remaining_context_output_ceiling_tokens is not None
                        else self.dynamic_generation_budget_policy
                    )
                    if self.use_remaining_context_for_generation
                    else self.fixed_generation_budget_policy
                )
            ),
        }
        if retained_images:
            compression.update(
                {
                    "raw_suffix_image_context_budget_policy": (
                        R2EACT_RAW_SUFFIX_IMAGE_CONTEXT_BUDGET_POLICY
                    ),
                    "raw_suffix_image_max_pixels_per_image": (
                        QWEN38_RAW_SUFFIX_IMAGE_MAX_PIXELS
                    ),
                    "raw_suffix_image_max_visual_tokens_per_image": (
                        QWEN38_RAW_SUFFIX_IMAGE_MAX_VISUAL_TOKENS_PER_IMAGE
                    ),
                    "raw_suffix_image_context_reserve_tokens_per_image": (
                        QWEN38_RAW_SUFFIX_IMAGE_CONTEXT_RESERVE_TOKENS_PER_IMAGE
                    ),
                    "raw_suffix_image_context_reserve_tokens": (
                        raw_suffix_image_context_reserve_tokens
                    ),
                    "generation_server_budget_prefix_tokens": (
                        server_budget_prefix_tokens
                    ),
                }
            )
            if getattr(self, "_image_token_budget", None) is not None:
                estimate = self._image_token_budget.estimate(retained_images).to_dict()
                compression.update(image_token_budget=estimate,
                    raw_suffix_image_context_budget_policy=estimate["policy"],
                    raw_suffix_image_context_reserve_tokens_per_image=None)
        previous_persisted_handle = self._memory_handle
        generation_multimodal_kwargs: dict[str, Any] = {}
        if retained_images:
            generation_multimodal_kwargs["image_data"] = [
                image.image_data for image in retained_images
            ]
        try:
            materialize_pending_gpu_memory()
            compression["source_compression_request_seconds"] = _finite(
                compression_seconds
            )
            _emit_rollout_event(
                event_sink,
                event="compression_finished",
                episode_id=episode_id,
                round_index=round_index,
                payload={"compression": dict(compression)},
            )
            started = time.perf_counter()
            if not has_memory:
                result = self.client.generate_discrete(
                    prefix_ids,
                    max_new_tokens=generation_max_new_tokens,
                    sampling_overrides=self.generation_sampling,
                    **generation_multimodal_kwargs,
                )
            elif memory_handle is not None:
                result = self.client.generate_with_memory_handle(
                    prefix_ids,
                    memory_positions=generation_memory_positions,
                    memory_handle=memory_handle,
                    max_new_tokens=generation_max_new_tokens,
                    expected_hidden_size=self.target_hidden_size,
                    sampling_overrides=self.generation_sampling,
                    **generation_multimodal_kwargs,
                )
            else:
                assert memory is not None
                result = self.client.generate_with_memory(
                    prefix_ids,
                    memory_positions=generation_memory_positions,
                    memory_embeddings=memory,
                    max_new_tokens=generation_max_new_tokens,
                    expected_hidden_size=self.target_hidden_size,
                    sampling_overrides=self.generation_sampling,
                    **generation_multimodal_kwargs,
                )
            compression["target_generation_seconds"] = _finite(
                time.perf_counter() - started
            )
            finished = _finish_generation(
                tokenizer=self.tokenizer,
                result=result,
                eos_token_id=self.eos_token_id,
                compression=compression,
                qwen_native_family=self.target_family,
            )
        except BaseException as error:
            self._cleanup_gpu_candidate_after_failure(
                memory_handle,
                previous_persisted_handle,
                error,
            )
            raise

        candidate_replaces_parent = memory_handle is not None and (
            previous_persisted_handle is None
            or memory_handle.handle_id != previous_persisted_handle.handle_id
        )
        if candidate_replaces_parent and previous_persisted_handle is not None:
            try:
                self._release_gpu_memory_handle(
                    previous_persisted_handle,
                    reason="parent_replaced_after_recovery",
                )
            except BaseException as error:
                assert memory_handle is not None
                try:
                    self._release_gpu_memory_handle(
                        memory_handle,
                        reason="candidate_parent_release_rollback",
                    )
                except BaseException as release_error:
                    add_note = getattr(error, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "candidate rollback cleanup also failed: "
                            f"{release_error}"
                        )
                raise
        self._known_history_input_ids = history
        self._known_history_message_end_tokens = tuple(message_end_tokens)
        self._incremental_history_receipt = history_render.staged_receipt
        self._consumed_history_tokens = consumed
        self._consumed_round_count = consumed_round_count
        self._memory = memory
        self._memory_handle = memory_handle
        self._state_transition_index = transition_index
        if eviction_plan["oversized_complete_round_eviction_applied_this_round"]:
            self._oversized_complete_round_consumed_round_count = max(
                self._oversized_complete_round_consumed_round_count,
                int(
                    eviction_plan["oversized_complete_round_stage_consumed_round_count"]
                ),
            )
            self._oversized_complete_round_evicted_round_indices.update(
                int(round_number)
                for round_number in eviction_plan[
                    "oversized_complete_round_evicted_round_indices_this_round"
                ]
            )
        if eviction_plan["capacity_pressure_eviction_applied_this_round"]:
            self._capacity_pressure_consumed_round_count = max(
                self._capacity_pressure_consumed_round_count,
                consumed_round_count,
            )
            self._capacity_pressure_evicted_round_indices.update(
                int(round_number)
                for round_number in eviction_plan[
                    "capacity_pressure_evicted_round_indices_this_round"
                ]
            )
        return finished


class GrepSeekSGLangRawRolloutGenerator:
    """Generate from the complete discrete ReAct history without compression."""

    backend = "sglang-raw-history"
    device = "remote-sglang"
    collect_diagnostics = True

    def __init__(
        self,
        *,
        tokenizer: Any,
        client: GrepSeekSGLangClient,
        max_new_tokens: int,
        eos_token_id: int = EOS_TOKEN_ID,
        server_context_length: Optional[int] = None,
        server_max_total_num_tokens: Optional[int] = None,
        server_max_req_input_len: Optional[int] = None,
        use_remaining_context_for_generation: bool = False,
        remaining_context_output_ceiling_tokens: Optional[int] = None,
        context_organization: str = CAUSAL_OBSERVATION_CACHE_V1,
        checkpoint_context_organization: str = CAUSAL_OBSERVATION_CACHE_V1,
        checkpoint_schema: str = LEGACY_CHECKPOINT_SCHEMA,
        compression_ratio: Optional[int] = None,
        recursive_working_memory_enabled: bool = False,
        recursive_working_memory_policy: Optional[str] = None,
    ) -> None:
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if remaining_context_output_ceiling_tokens is not None and (
            isinstance(remaining_context_output_ceiling_tokens, bool)
            or int(remaining_context_output_ceiling_tokens) <= 0
        ):
            raise ValueError(
                "remaining_context_output_ceiling_tokens must be positive or None"
            )
        self.tokenizer = tokenizer
        self.client = client
        self.max_new_tokens = int(max_new_tokens)
        self.eos_token_id = int(eos_token_id)
        self.server_context_length = (
            int(server_context_length) if server_context_length is not None else None
        )
        self.server_max_total_num_tokens = (
            int(server_max_total_num_tokens)
            if server_max_total_num_tokens is not None
            else None
        )
        self.server_max_req_input_len = (
            int(server_max_req_input_len)
            if server_max_req_input_len is not None
            else None
        )
        self.use_remaining_context_for_generation = bool(
            use_remaining_context_for_generation
        )
        self.remaining_context_output_ceiling_tokens = (
            int(remaining_context_output_ceiling_tokens)
            if remaining_context_output_ceiling_tokens is not None
            else None
        )
        self.context_organization = str(context_organization)
        self.checkpoint_context_organization = str(checkpoint_context_organization)
        self.checkpoint_schema = str(checkpoint_schema)
        self._token_clock_runtime = (
            _token_clock_runtime_for_schema(self.checkpoint_schema)
            if self.checkpoint_context_organization == QWEN36_TOKEN_CLOCK_V1
            else None
        )
        self.compression_ratio = (
            int(compression_ratio) if compression_ratio is not None else None
        )
        if not isinstance(recursive_working_memory_enabled, bool):
            raise TypeError("recursive_working_memory_enabled must be boolean")
        self.recursive_working_memory_enabled = recursive_working_memory_enabled
        self.recursive_working_memory_policy = (
            str(recursive_working_memory_policy)
            if recursive_working_memory_policy is not None
            else None
        )
        self._max_turns: Optional[int] = None
        self._tool_names: tuple[str, ...] = ()

    @property
    def _qwen36_native(self) -> bool:
        """Use shared Qwen3.6/3.8 framing based on checkpoint provenance."""

        return self._token_clock_runtime is not None

    @property
    def _qwen_native_family(self) -> Optional[str]:
        runtime = self._token_clock_runtime
        return None if runtime is None else runtime.target_family

    @property
    def _qwen_model_display_name(self) -> str:
        runtime = self._token_clock_runtime
        return "Qwen" if runtime is None else runtime.display_name

    def _apply_native_chat_ids(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        optional_family: dict[str, Any] = {}
        if self._qwen_native_family == "qwen38":
            optional_family["model_display_name"] = self._qwen_model_display_name
        return _apply_qwen36_chat_ids(
            self.tokenizer,
            messages,
            tool_names=self._tool_names,
            add_generation_prompt=add_generation_prompt,
            **optional_family,
        )

    def configure_max_turns(self, value: Optional[int]) -> None:
        """Configure dynamic user guidance for the next serialized request."""

        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_WORKING_MEMORY_TURNS
        ):
            raise ValueError(
                f"max_turns must be between 1 and {MAX_WORKING_MEMORY_TURNS} or None"
            )
        self._max_turns = value

    def configure_tools(self, tool_names: Sequence[str]) -> None:
        names = tuple(str(name) for name in tool_names)
        if self._qwen36_native:
            _qwen36_tool_schemas(
                names, model_display_name=self._qwen_model_display_name
            )
        self._tool_names = names

    @torch.inference_mode()
    def generate_next(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink] = None,
    ) -> RolloutGeneration:
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        canonical_messages: list[dict[str, str]] = []
        source_segments = 0
        source_tokens = 0
        empty_stdout_segments = 0
        source_token_lengths: list[int] = []
        for message_index, message in enumerate(messages):
            if message.get("role") != "tool":
                canonical_messages.append(
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                    }
                )
                continue
            payload = parse_tool_response(
                str(message.get("content", "")),
                episode_id=episode_id,
                message_index=message_index,
            )
            stdout = str(payload["stdout"])
            canonical_messages.append(
                {
                    "role": "tool",
                    "content": _canonical_tool_content(payload, stdout),
                }
            )
            token_count = len(_encode(self.tokenizer, stdout))
            if token_count:
                source_segments += 1
                source_tokens += token_count
                source_token_lengths.append(token_count)
            else:
                empty_stdout_segments += 1

        (
            completed_observation_rounds,
            current_tool_calls,
            current_user_feedback_rounds,
        ) = _observation_role_counts(canonical_messages)

        if self._qwen36_native:
            generation_messages, guidance = (
                _qwen36_messages_with_tool_response_turn_guidance(
                    canonical_messages,
                    max_turns=self._max_turns,
                    round_index=round_index,
                    **(
                        {"model_display_name": self._qwen_model_display_name}
                        if self._qwen_native_family == "qwen38"
                        else {}
                    ),
                )
            )
            guided_initial = generation_messages[:2]
        else:
            guided_messages, guidance = _messages_with_tool_call_turn_guidance(
                canonical_messages,
                max_turns=self._max_turns,
                round_index=round_index,
            )
            guided_initial = guided_messages[:2]
            generation_messages = guided_initial + canonical_messages[2:]
        if self._qwen36_native:
            original_prefix_tokens = len(
                self._apply_native_chat_ids(
                    canonical_messages,
                    add_generation_prompt=True,
                )
            )
            prefix_ids = self._apply_native_chat_ids(
                generation_messages,
                add_generation_prompt=True,
            )
            if prefix_ids[-len(QWEN36_ASSISTANT_OPENING_IDS) :] != list(
                QWEN36_ASSISTANT_OPENING_IDS
            ):
                raise ValueError(
                    f"{self._qwen_model_display_name} generation prompt lost its native thinking opening"
                )
        else:
            original_prefix_tokens = len(
                _apply_chat_ids(
                    self.tokenizer,
                    canonical_messages,
                    add_generation_prompt=True,
                )
            )
            prompt_text = _render_chat(
                self.tokenizer, generation_messages, add_generation_prompt=True
            )
            if not prompt_text.endswith(ASSISTANT_CONTENT_PREFIX):
                raise ValueError(
                    "GrepSeek generation prompt no longer ends with '<think>\\n'"
                )
            prefix_ids = _apply_chat_ids(
                self.tokenizer, generation_messages, add_generation_prompt=True
            )
        if not prefix_ids:
            raise ValueError("raw-history generation prefix must not be empty")

        generation_max_new_tokens = _generation_token_budget(
            prefix_token_count=len(prefix_ids),
            configured_max_new_tokens=self.max_new_tokens,
            server_context_length=self.server_context_length,
            server_max_total_num_tokens=self.server_max_total_num_tokens,
            server_max_req_input_len=self.server_max_req_input_len,
            use_remaining_context=self.use_remaining_context_for_generation,
            remaining_context_output_ceiling_tokens=(
                self.remaining_context_output_ceiling_tokens
            ),
        )
        compression: dict[str, Any] = {
            "backend": self.backend,
            "mode": "raw_history",
            "context_organization": self.context_organization,
            "checkpoint_context_organization": (self.checkpoint_context_organization),
            "checkpoint_schema": self.checkpoint_schema,
            "compression_ratio": self.compression_ratio,
            "recursive_working_memory_enabled": (self.recursive_working_memory_enabled),
            "recursive_working_memory_policy": (self.recursive_working_memory_policy),
            **(
                {
                    "enable_thinking": QWEN36_ENABLE_THINKING,
                    "target_family": self._qwen_native_family,
                    **_token_clock_sampling_telemetry(
                        self._token_clock_runtime.generation_sampling_policy
                    ),
                    "compressed_message_indices": [],
                    "compressed_react_round_indices": [],
                    "archived_message_indices": [],
                    "memory_archive_policy": (
                        "no_archive_without_context_compression_v1"
                    ),
                    "memory_archive_filename": "memory.jsonl",
                }
                if self._qwen36_native
                else {}
            ),
            "source_segments": source_segments,
            "source_tokens": source_tokens,
            "empty_stdout_segments": empty_stdout_segments,
            "source_token_lengths": source_token_lengths,
            "completed_observation_round_count": completed_observation_rounds,
            "tool_observation_count": current_tool_calls,
            "user_feedback_observation_count": current_user_feedback_rounds,
            "memory_slots": 0,
            "effective_compression_ratio": None,
            "source_encoder_tokens": 0,
            "student_prefix_tokens": len(prefix_ids),
            "tool_call_turn_guidance_policy": (
                (
                    QWEN36_TOOL_RESPONSE_TURN_GUIDANCE_POLICY
                    if self._qwen36_native
                    else TOOL_CALL_TURN_GUIDANCE_POLICY
                )
                if self._max_turns is not None
                else None
            ),
            "tool_call_turn_guidance_placement": (
                (
                    "before_each_native_tool_response_close"
                    if self._qwen36_native
                    else "initial_user_suffix"
                )
                if self._max_turns is not None
                else None
            ),
            "tool_call_turn_guidance_max_turns": self._max_turns,
            "tool_call_turn_guidance_max_tool_calls": (
                None if self._max_turns is None else self._max_turns - 1
            ),
            "tool_call_turn_guidance_current_tool_calls": (
                None if self._max_turns is None else current_tool_calls
            ),
            "tool_call_turn_guidance_action": (
                None
                if self._max_turns is None
                else (
                    "continue"
                    if int(round_index) < self._max_turns
                    and current_tool_calls < self._max_turns - 1
                    else "answer"
                )
            ),
            "generation_user_suffix_tokens": len(prefix_ids) - original_prefix_tokens,
            "source_encode_seconds": None,
            "compressor_seconds": None,
            "source_compression_request_seconds": None,
            "target_generation_seconds": None,
            "server_context_length": self.server_context_length,
            "server_max_total_num_tokens": self.server_max_total_num_tokens,
            "server_max_req_input_len": self.server_max_req_input_len,
            "configured_max_new_tokens": self.max_new_tokens,
            "remaining_context_output_ceiling_tokens": (
                self.remaining_context_output_ceiling_tokens
            ),
            "generation_max_new_tokens": generation_max_new_tokens,
            "generation_budget_policy": (
                (
                    "remaining_context_after_input_capped_to_output_ceiling"
                    if self.remaining_context_output_ceiling_tokens is not None
                    else "remaining_context_after_input"
                )
                if self.use_remaining_context_for_generation
                else "configured_cap_clamped_to_remaining_context"
            ),
            "generation_budget_contract_policy": (
                "remaining_sglang_context_with_output_ceiling_v1"
                if (
                    self.use_remaining_context_for_generation
                    and self.remaining_context_output_ceiling_tokens is not None
                )
                else None
            ),
        }
        _emit_rollout_event(
            event_sink,
            event="compression_finished",
            episode_id=episode_id,
            round_index=round_index,
            payload={"compression": dict(compression)},
        )
        generation_started = time.perf_counter()
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": generation_max_new_tokens
        }
        if self._qwen36_native:
            generation_kwargs["sampling_overrides"] = QWEN36_GENERATION_SAMPLING
        result = self.client.generate_discrete(prefix_ids, **generation_kwargs)
        compression["target_generation_seconds"] = _finite(
            time.perf_counter() - generation_started
        )
        return _finish_generation(
            tokenizer=self.tokenizer,
            result=result,
            eos_token_id=self.eos_token_id,
            compression=compression,
            qwen_native_family=self._qwen_native_family,
        )


def make_sglang_raw_live_bundle(
    bundle: GrepSeekLiveBundle,
) -> GrepSeekLiveBundle:
    """Share one SGLang client/tool runtime with a raw-history generator."""

    compressed = bundle.generator
    if not isinstance(
        compressed,
        (
            GrepSeekSGLangRolloutGenerator,
            GrepSeekSGLangRecursiveRolloutGenerator,
            GrepSeekSGLangRecursiveWorkingMemoryRolloutGenerator,
            GrepSeekSGLangWorkingMemoryV3RolloutGenerator,
            Qwen36SGLangTokenClockRolloutGenerator,
        ),
    ):
        raise TypeError("raw live bundle requires an SGLang compressor generator")
    raw_generator = GrepSeekSGLangRawRolloutGenerator(
        tokenizer=compressed.tokenizer,
        client=compressed.client,
        max_new_tokens=compressed.max_new_tokens,
        eos_token_id=compressed.eos_token_id,
        server_context_length=compressed.server_context_length,
        server_max_total_num_tokens=compressed.server_max_total_num_tokens,
        server_max_req_input_len=compressed.server_max_req_input_len,
        use_remaining_context_for_generation=(
            compressed.use_remaining_context_for_generation
        ),
        remaining_context_output_ceiling_tokens=getattr(
            compressed, "remaining_context_output_ceiling_tokens", None
        ),
        context_organization=compressed.context_organization,
        checkpoint_context_organization=(compressed.checkpoint_context_organization),
        checkpoint_schema=compressed.checkpoint_schema,
        compression_ratio=compressed.compression_ratio,
        recursive_working_memory_enabled=(compressed.recursive_working_memory_enabled),
        recursive_working_memory_policy=(compressed.recursive_working_memory_policy),
    )
    return replace(bundle, generator=raw_generator)


def load_sglang_live_bundle(
    checkpoint: str | Path,
    *,
    sglang_url: str,
    corpus_dir: str | Path,
    max_new_tokens: int = 512,
    tool_timeout: float = 180.0,
    tool_max_output_tokens: Optional[int] = 2_048,
    request_timeout: float = 1_800.0,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    target_model: Optional[str] = None,
    prepared_data: Optional[str | Path] = None,
    raw_parquet: Optional[str | Path] = None,
    use_remaining_context_for_generation: bool = False,
    remaining_context_output_ceiling_tokens: Optional[int] = None,
    recursive_working_memory: bool = False,
    native_history_policy: str = NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER,
    sglang_cache_policy: str = SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1,
) -> GrepSeekLiveBundle:
    """Build a CPU-only live bundle whose model work is owned by SGLang."""

    if not isinstance(recursive_working_memory, bool):
        raise TypeError("recursive_working_memory must be boolean")
    if remaining_context_output_ceiling_tokens is not None and (
        isinstance(remaining_context_output_ceiling_tokens, bool)
        or int(remaining_context_output_ceiling_tokens) <= 0
    ):
        raise ValueError(
            "remaining_context_output_ceiling_tokens must be positive or None"
        )
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config = read_checkpoint_config(checkpoint_path)
    serving_contract = resolve_grepseek_checkpoint_serving_contract(config)
    token_clock_runtime = (
        _token_clock_runtime_for_schema(serving_contract.checkpoint_schema)
        if serving_contract.context_organization == QWEN36_TOKEN_CLOCK_V1
        else None
    )
    if (
        serving_contract.context_organization != QWEN36_TOKEN_CLOCK_V1
        and native_history_policy != NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER
    ):
        raise ValueError(
            "non-token-clock checkpoints require native_history_policy="
            f"{NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER!r}"
        )
    if recursive_working_memory and (
        serving_contract.context_organization != RECURSIVE_HISTORY_V2
    ):
        raise ValueError(
            "--recursive-working-memory requires a verified recursive_history_v2 "
            "checkpoint and is an inference-only ablation; checkpoint-native "
            "recursive_working_memory_v3 is selected automatically with the "
            "flag disabled"
        )
    saved_args = config["args"]
    target_path = str(target_model or saved_args["target_model"])
    if target_model and not _same_path_or_id(
        target_path, str(saved_args["target_model"])
    ):
        raise ValueError("target_model differs from checkpoint config")

    if serving_contract.context_organization == QWEN36_TOKEN_CLOCK_V1:
        if prepared_data is not None or raw_parquet is not None:
            raise ValueError(
                f"{token_clock_runtime.display_name} token-clock serving does not "
                "accept training-data overrides"
            )
        # Serving exports (including multimodal repeat checkpoints) need not
        # retain their training-data path. Token-clock inference uses the
        # fixed ReAct prompt below; this field is provenance, not an input.
        training_data_path = saved_args.get("data")
        resolved_data = ResolvedGrepSeekLiveData(
            prepared_data=None,
            raw_parquet=(
                Path(str(training_data_path)).expanduser().resolve()
                if training_data_path
                else None
            ),
            split_config={"validation_fraction": 0.0, "seed": 42},
        )
        system_prompt = react_memory_shell_system_prompt()
        source_episode_id = token_clock_runtime.source_episode_id
    else:
        resolved_data = resolve_grepseek_live_data_paths(
            saved_args,
            prepared_override=prepared_data,
            raw_override=raw_parquet,
        )
        system_prompt, source_episode_id = _load_real_system_prompt(
            resolved_data.raw_parquet, resolved_data.split_config
        )
    tokenizer = AutoTokenizer.from_pretrained(
        target_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _assert_token_contract(tokenizer)
    if serving_contract.context_organization == RECURSIVE_HISTORY_V2 and tuple(
        _encode(tokenizer, GENERATION_BOUNDARY_TEXT)
    ) != tuple(GENERATION_BOUNDARY_IDS):
        raise ValueError("tokenizer disagrees with the recursive-v2 native G7 boundary")
    if serving_contract.context_organization == RECURSIVE_WORKING_MEMORY_V3:
        if tuple(_encode(tokenizer, ASSISTANT_OPENING_TEXT)) != tuple(
            ASSISTANT_OPENING_IDS
        ):
            raise ValueError(
                "tokenizer disagrees with the working-memory-v3 native G5 opening"
            )
        if tuple(_encode(tokenizer, USER_TURN_END_TEXT)) != tuple(
            GENERATION_BOUNDARY_IDS[:2]
        ):
            raise ValueError(
                "tokenizer disagrees with the working-memory-v3 user turn close"
            )
    elif serving_contract.context_organization == QWEN36_TOKEN_CLOCK_V1:
        if tuple(_encode(tokenizer, QWEN36_ASSISTANT_OPENING_TEXT)) != tuple(
            QWEN36_ASSISTANT_OPENING_IDS
        ):
            raise ValueError(
                "tokenizer disagrees with the "
                f"{token_clock_runtime.display_name} thinking assistant opening"
            )

    # Keep the safe corpus preflight ahead of any inference-server request.
    tool_runner = GrepSeekCorpusToolRunner(
        corpus_dir=corpus_dir,
        tokenizer=tokenizer,
        timeout=tool_timeout,
        max_output_tokens=tool_max_output_tokens,
    )
    corpus_preflight = preflight_corpus_tool(tool_runner)
    web_search_enabled = bool(getattr(tool_runner, "web_search_enabled", False))
    browser_enabled = bool(getattr(tool_runner, "browser_enabled", False))
    client_kwargs: dict[str, Any] = {
        "request_timeout": request_timeout,
        "eos_token_id": EOS_TOKEN_ID,
    }
    if sglang_cache_policy != SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1:
        client_kwargs["sglang_cache_policy"] = sglang_cache_policy
    client = GrepSeekSGLangClient(sglang_url, **client_kwargs)
    client.health()
    server_info = client.server_info()
    if (
        sglang_cache_policy
        == SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1
    ):
        client.attest_sglang_cache_policy(server_info)
    server_context_length = _server_positive_integer(server_info, "context_length")
    server_max_total_num_tokens = _server_positive_integer(
        server_info, "max_total_num_tokens"
    )
    server_max_req_input_len = _server_positive_integer(
        server_info, "max_req_input_len"
    )

    common_generator_kwargs = {
        "tokenizer": tokenizer,
        "client": client,
        "compression_ratio": serving_contract.compression_ratio,
        "target_hidden_size": int(config["target_hidden_size"]),
        "max_encoder_tokens": saved_args.get("max_encoder_tokens"),
        "max_new_tokens": max_new_tokens,
        "checkpoint_schema": serving_contract.checkpoint_schema,
        "eos_token_id": EOS_TOKEN_ID,
        "server_context_length": server_context_length,
        "server_max_total_num_tokens": server_max_total_num_tokens,
        "server_max_req_input_len": server_max_req_input_len,
        "use_remaining_context_for_generation": use_remaining_context_for_generation,
    }
    if serving_contract.context_organization == RECURSIVE_HISTORY_V2:
        recursive_generator_type = (
            GrepSeekSGLangRecursiveWorkingMemoryRolloutGenerator
            if recursive_working_memory
            else GrepSeekSGLangRecursiveRolloutGenerator
        )
        generator = recursive_generator_type(**common_generator_kwargs)
    elif serving_contract.context_organization == RECURSIVE_WORKING_MEMORY_V3:
        generator = GrepSeekSGLangWorkingMemoryV3RolloutGenerator(
            **common_generator_kwargs
        )
    elif serving_contract.context_organization == QWEN36_TOKEN_CLOCK_V1:
        generator = Qwen36SGLangTokenClockRolloutGenerator(
            **common_generator_kwargs,
            remaining_context_output_ceiling_tokens=(
                remaining_context_output_ceiling_tokens
            ),
        )
        if native_history_policy != NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER:
            generator.configure_native_history_policy(native_history_policy)
    else:
        generator = GrepSeekSGLangRolloutGenerator(
            **common_generator_kwargs,
            pad_to_multiple_of=saved_args.get("pad_to_multiple_of"),
        )
    runtime_context_organization = (
        RECURSIVE_HISTORY_WITH_RAW_WORKING_MEMORY_ABLATION_V1
        if recursive_working_memory
        else str(
            getattr(
                generator, "context_organization", serving_contract.context_organization
            )
        )
    )
    working_memory_policy = (
        LATEST_TOOL_EXCHANGE_RAW_V1
        if recursive_working_memory
        else (
            ONE_ROUND_LAG_RECURSIVE_WORKING_MEMORY_V3
            if serving_contract.context_organization == RECURSIVE_WORKING_MEMORY_V3
            else (
                QWEN36_ROUND_WORKING_MEMORY_POLICY
                if serving_contract.context_organization == QWEN36_TOKEN_CLOCK_V1
                else None
            )
        )
    )
    return GrepSeekLiveBundle(
        checkpoint=checkpoint_path,
        raw_parquet=resolved_data.raw_parquet,
        prepared_data=resolved_data.prepared_data,
        system_prompt=system_prompt,
        system_prompt_source_episode_id=source_episode_id,
        generator=generator,
        tool_runner=tool_runner,
        corpus_preflight=corpus_preflight,
        lock=threading.Lock(),
        config=config,
        target_model=target_path,
        context_organization=runtime_context_organization,
        checkpoint_context_organization=serving_contract.context_organization,
        checkpoint_schema=serving_contract.checkpoint_schema,
        compression_ratio=serving_contract.compression_ratio,
        recursive_working_memory_enabled=(
            recursive_working_memory
            or serving_contract.context_organization
            in {RECURSIVE_WORKING_MEMORY_V3, QWEN36_TOKEN_CLOCK_V1}
        ),
        recursive_working_memory_policy=working_memory_policy,
        backend="sglang",
        sglang_url=client.base_url,
        web_search_enabled=web_search_enabled,
        browser_enabled=browser_enabled,
    )


__all__ = [
    "CAUSAL_OBSERVATION_CACHE_V1",
    "CHECKPOINT_1024_SERVING_EVICTION_POLICY",
    "CODEX_NATIVE_COMPACTION_POLICY",
    "LATEST_TOOL_EXCHANGE_RAW_V1",
    "NATIVE_HISTORY_POLICY_APPEND_ONLY_TOKEN_EXACT",
    "NATIVE_HISTORY_POLICY_LEGACY_FULL_RENDER",
    "QWEN36_GENERATION_SAMPLING",
    "QWEN36_GENERATION_SAMPLING_POLICY",
    "QWEN36_OVERSIZED_COMPLETE_ROUND_EVICTION_POLICY",
    "QWEN36_ROUND_RECURSIVE_V1",
    "QWEN36_ROUND_WORKING_MEMORY_POLICY",
    "QWEN36_TOKEN_CLOCK_V1",
    "QWEN38_8192_ACCUMULATION_SERVING_EVICTION_POLICY",
    "QWEN38_SERVING_EVICTION_ACCUMULATION_TOKENS",
    "R2EACT_RAW_SUFFIX_IMAGE_POLICY",
    "RECURSIVE_HISTORY_V2",
    "RECURSIVE_HISTORY_WITH_RAW_WORKING_MEMORY_ABLATION_V1",
    "RECURSIVE_WORKING_MEMORY_V3",
    "SGLANG_CACHE_POLICY_IDLE_FLUSH_UNIQUE_KEY_V1",
    "SGLANG_CACHE_POLICY_ISOLATED_SOFT_NO_FLUSH_DISCRETE_PREFIX_V1",
    "GrepSeekCheckpointServingContract",
    "R2EActRawSuffixImage",
    "GrepSeekSGLangRawRolloutGenerator",
    "GrepSeekSGLangRecursiveRolloutGenerator",
    "GrepSeekSGLangRecursiveWorkingMemoryRolloutGenerator",
    "GrepSeekSGLangWorkingMemoryV3RolloutGenerator",
    "GrepSeekSGLangRolloutGenerator",
    "Qwen36SGLangTokenClockRolloutGenerator",
    "ResolvedGrepSeekLiveData",
    "load_sglang_live_bundle",
    "make_sglang_raw_live_bundle",
    "resolve_grepseek_checkpoint_serving_contract",
    "resolve_grepseek_live_data_paths",
]
