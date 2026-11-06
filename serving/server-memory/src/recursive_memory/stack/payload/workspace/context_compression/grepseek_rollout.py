"""Leak-free multi-turn GrepSeek rollouts with compressed tool evidence.

This module is deliberately independent of the next-turn evaluation CLI.  A
rollout starts with only the held-out episode's system and user messages.  It
then alternates generated assistant turns and results from the read-only corpus
runner.  Before every later assistant turn, eligible older nonempty tool
``stdout`` values are encoded by the frozen target and replaced by trained
DFlash soft memories.  The newest observation always remains discrete for the
next action, using the same canonical JSON and interleaving contract as training.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Sequence

import torch

from context_compression.grepseek_data import (
    GrepSeekCompressionCollator,
    _apply_chat_ids,
    _canonical_tool_content,
    _choose_soft_marker,
    _decode,
    _encode,
    _render_chat,
    _rendered_ids_and_offsets,
    _split_chat_at_markers,
    _validate_messages,
    assign_episode_split,
    iter_grepseek_rows,
    parse_tool_response,
)
from context_compression.grepseek_generation import (
    EOS_TOKEN_ID,
    ParsedGrepSeekContinuation,
    ParsedGrepSeekToolCall,
    build_discrete_generation_prefix,
    build_interleaved_generation_prefix,
    normalize_answer,
    normalized_answer_token_f1,
    parse_grepseek_continuation,
    trim_at_eos,
)
from context_compression.grepseek_modeling import (
    GrepSeekDFlashCompressor,
    GrepSeekFrozenTargetRuntime,
)
from context_compression.grepseek_tool import (
    MAX_CONSECUTIVE_RECOVERABLE_TOOL_SCHEMA_ERRORS,
    ToolResult,
)
from context_compression.frozen_target import (
    _flatten_encoded_evidence,
    _validate_memory_contract,
)


ASSISTANT_CONTENT_PREFIX = "<think>\n"
ROLLOUT_SCHEMA = "grepseek_compressed_multiturn_v1"
RolloutEventSink = Callable[[Mapping[str, Any]], None]
MAX_RECOVERABLE_TOOL_SCHEMA_ERROR_CHARS = 512


def _sanitize_tool_schema_error(value: Any) -> Optional[str]:
    """Return bounded single-line parser feedback safe for model context."""

    if not isinstance(value, str):
        return None
    without_controls = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in value
    )
    normalized = " ".join(without_controls.split())
    if not normalized:
        return None
    if len(normalized) > MAX_RECOVERABLE_TOOL_SCHEMA_ERROR_CHARS:
        normalized = (
            normalized[: MAX_RECOVERABLE_TOOL_SCHEMA_ERROR_CHARS - 3].rstrip() + "..."
        )
    return normalized


def _recoverable_tool_schema_result(
    parsed: ParsedGrepSeekContinuation | ParsedGrepSeekToolCall,
) -> Optional[ToolResult]:
    """Build a local correction observation for a known tool's bad arguments.

    This path never executes the requested tool.  It is deliberately limited
    to valid JSON for a known tool with a parser-produced, public-schema error.
    Malformed syntax and other public-schema failures are handled separately by
    bounded, non-executing retries.
    """

    payload = parsed.tool_payload
    schema_error = _sanitize_tool_schema_error(parsed.tool_schema_error)
    if (
        getattr(parsed, "kind", "tool_call") != "tool_call"
        or not parsed.tool_json_valid
        or parsed.tool_schema_valid
        or not isinstance(payload, dict)
        or schema_error is None
    ):
        return None
    tool_name = payload.get("name")
    if tool_name not in {"shell", "web_search", "browser"}:
        return None
    return ToolResult(
        command=f"{tool_name} <invalid arguments>",
        stdout="",
        stderr=(
            f"validation error: {schema_error}. Correct the arguments and try again, "
            "or answer from evidence already collected."
        ),
        exit_code=-2,
        timed_out=False,
        truncated=False,
        information_lines=[],
    )


def _recoverable_malformed_tool_result(
    parsed: ParsedGrepSeekContinuation | ParsedGrepSeekToolCall,
) -> Optional[ToolResult]:
    """Return a bounded correction observation for malformed tool syntax.

    The malformed payload is never repaired or executed.  A retry is allowed
    only when the parser can still prove that the assistant attempted a tool
    call.  The existing consecutive-error budget then prevents an unbounded
    malformed-generation loop.
    """

    if (
        getattr(parsed, "kind", "tool_call") != "tool_call"
        or not parsed.errors
    ):
        return None
    public_errors = {
        "invalid_tool_json",
        "invalid_native_tool_xml",
        "missing_think_close",
        "unexpected_generated_think_open",
        "missing_tool_close",
        "tokens_after_tool_close",
    }
    errors = tuple(
        str(error).strip()
        for error in parsed.errors
        if str(error).strip() in public_errors
    )
    if not errors:
        return None
    return ToolResult(
        command="tool_call <malformed syntax>",
        stdout="",
        stderr=(
            "validation error: malformed tool-call syntax "
            f"({', '.join(errors)}). Emit complete tool calls using "
            "the advertised schema, or answer from evidence already collected."
        ),
        exit_code=-2,
        timed_out=False,
        truncated=False,
        information_lines=[],
    )


def _recoverable_tool_validation_result(
    parsed: ParsedGrepSeekContinuation | ParsedGrepSeekToolCall,
) -> tuple[Optional[ToolResult], bool]:
    """Return a non-executing correction for any recognizable bad tool call.

    The second return value identifies malformed syntax for rollout telemetry.
    No model-produced payload is ever repaired and sent to a tool runner.
    """

    result = _recoverable_tool_schema_result(parsed)
    if result is not None:
        return result, False
    result = _recoverable_malformed_tool_result(parsed)
    if result is not None:
        return result, True
    if (
        getattr(parsed, "kind", "tool_call") != "tool_call"
        or parsed.tool_schema_valid
    ):
        return None, False
    payload = parsed.tool_payload
    tool_name = payload.get("name") if isinstance(payload, dict) else None
    safe_name = (
        str(tool_name)
        if isinstance(tool_name, str) and tool_name in {"shell", "web_search", "browser"}
        else "tool_call"
    )
    public_error_names = {
        "tool_payload_not_object",
        "tool_name_not_shell",
        "tool_arguments_not_object",
        "missing_tool_command",
        "invalid_web_search_arguments",
        "invalid_browser_arguments",
    }
    errors = [
        str(error)
        for error in parsed.errors
        if str(error) in public_error_names
    ]
    detail = _sanitize_tool_schema_error(parsed.tool_schema_error)
    if detail is None:
        detail = ", ".join(errors) if errors else "invalid tool schema"
    return (
        ToolResult(
            command=f"{safe_name} <invalid schema>",
            stdout="",
            stderr=(
                f"validation error: {detail}. Use one of the advertised tools "
                "with valid arguments, or answer from evidence already collected."
            ),
            exit_code=-2,
            timed_out=False,
            truncated=False,
            information_lines=[],
        ),
        False,
    )


def _tool_record(
    *,
    command: str,
    result: ToolResult,
    executed: bool,
    call_index: Optional[int] = None,
    schema_repair: bool = False,
    malformed_tool_retry: bool = False,
    schema_error: Optional[str] = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "command": command,
        "result": result.to_dict(),
        "executed": bool(executed),
        "validation_failed": result.exit_code == -2,
        "token_truncation_failed": result.stdout
        == "[tool output unavailable: token truncation failed]",
        "timed_out": result.timed_out,
        "truncated": result.truncated,
        "nonzero_exit": result.exit_code != 0,
    }
    if call_index is not None:
        record["call_index"] = int(call_index)
    if schema_repair:
        record.update(
            {
                "schema_repair": True,
                "malformed_tool_retry": bool(malformed_tool_retry),
                "schema_error": schema_error,
            }
        )
    return record


def _aggregate_tool_call_results(
    call_records: Sequence[Mapping[str, Any]],
) -> ToolResult:
    """Encode several observations in one released seven-key tool response.

    Rollout/live history requires exactly one assistant/tool pair per generation
    round.  The nested JSON is therefore evidence text in ``stdout`` rather
    than a new top-level response schema; existing history parsers and recursive
    context builders continue to see one canonical :class:`ToolResult`.
    """

    if len(call_records) <= 1:
        raise ValueError("tool-result aggregation requires at least two calls")
    results = []
    for ordinal, record in enumerate(call_records, start=1):
        result = record.get("result")
        if not isinstance(result, Mapping):
            raise TypeError("batched tool record result must be an object")
        # ``information_lines`` is a legacy duplicate of stdout.  Keep the
        # full call record for trajectory auditing, but never nest that second
        # copy inside the model-visible batch JSON: doing so recursively
        # doubles quotes/backslashes and evidence tokens at every archive
        # serialization boundary.
        nested_result = dict(result)
        nested_result["information_lines"] = []
        results.append(
            {
                "call_index": int(record.get("call_index", ordinal)),
                "executed": bool(record.get("executed", False)),
                "validation_failed": bool(record.get("validation_failed", False)),
                "command": str(record.get("command", "")),
                "result": nested_result,
            }
        )
    validation_only = all(not item["executed"] for item in results)
    exit_codes = [int(item["result"].get("exit_code", 1)) for item in results]
    payload = {
        "schema": "grepseek_tool_batch_v1",
        "tool_call_count": len(results),
        "results": results,
    }
    stdout = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ToolResult(
        command=f"tool_batch[{len(results)}]",
        stdout=stdout,
        stderr="",
        exit_code=(
            -2
            if validation_only
            else (0 if all(exit_code == 0 for exit_code in exit_codes) else 1)
        ),
        timed_out=any(bool(item["result"].get("timed_out")) for item in results),
        truncated=any(bool(item["result"].get("truncated")) for item in results),
        information_lines=[line for line in stdout.splitlines() if line.strip()],
    )


def _finish_reason_kind(value: Any) -> str:
    if isinstance(value, str):
        result = value.strip().lower()
        return result or "unknown"
    if isinstance(value, Mapping):
        for key in ("type", "reason", "finish_reason"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().lower()
    return "unknown"


def _finite_diagnostic(value: float) -> Optional[float]:
    result = float(value)
    return result if math.isfinite(result) else None


def _emit_rollout_event(
    event_sink: Optional[RolloutEventSink],
    *,
    event: str,
    episode_id: str,
    round_index: int,
    payload: Optional[Mapping[str, Any]] = None,
) -> None:
    """Send one request-local, JSON-safe rollout event when requested."""

    if event_sink is None:
        return
    event_sink(
        {
            "event": str(event),
            "episode_id": str(episode_id),
            "round_index": int(round_index),
            "payload": dict(payload or {}),
        }
    )


@dataclass(frozen=True)
class HeldoutEpisodeStart:
    """The only gold data made visible to a rollout."""

    episode_id: str
    messages: tuple[dict[str, str], dict[str, str]]

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "messages": [dict(message) for message in self.messages],
        }


@dataclass(frozen=True)
class HiddenAnswerReference:
    """A gold label loaded only after generation has completely finished."""

    episode_id: str
    answer_text: str
    final_assistant_content: str

    def to_json(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "normalized_answer": normalize_answer(self.answer_text),
            "final_assistant_content": self.final_assistant_content,
        }


def heldout_start_from_raw_episode(row: Mapping[str, Any]) -> HeldoutEpisodeStart:
    """Discard every gold message after the initial system/user pair."""

    episode_id, messages = _validate_messages(row)
    return HeldoutEpisodeStart(
        episode_id=episode_id,
        messages=(dict(messages[0]), dict(messages[1])),
    )


def select_heldout_episode_starts(
    rows: Iterable[Mapping[str, Any]],
    *,
    validation_fraction: float,
    split_seed: int,
    max_samples: int,
    episode_ids: Optional[Sequence[str]] = None,
) -> list[HeldoutEpisodeStart]:
    """Select validation episodes while retaining no gold future messages."""

    if isinstance(max_samples, bool) or int(max_samples) <= 0:
        raise ValueError("max_samples must be a positive integer")
    requested_order = None
    requested = None
    if episode_ids:
        requested_order = [str(episode_id) for episode_id in episode_ids]
        if len(set(requested_order)) != len(requested_order):
            raise ValueError("episode_ids must be unique")
        requested = set(requested_order)
    selected: dict[str, HeldoutEpisodeStart] = {}
    for row in rows:
        start = heldout_start_from_raw_episode(row)
        if requested is not None and start.episode_id not in requested:
            continue
        if (
            assign_episode_split(
                start.episode_id,
                validation_fraction=validation_fraction,
                seed=split_seed,
            )
            != "validation"
        ):
            if requested is not None:
                raise ValueError(
                    f"requested episode {start.episode_id!r} is not held-out validation"
                )
            continue
        if start.episode_id in selected:
            raise ValueError(f"duplicate raw episode id: {start.episode_id!r}")
        selected[start.episode_id] = start
        if requested is None and len(selected) >= int(max_samples):
            break
    if requested_order is not None:
        missing = [
            episode_id for episode_id in requested_order if episode_id not in selected
        ]
        if missing:
            raise KeyError(f"requested held-out episodes were not found: {missing[:5]}")
        ordered = [selected[episode_id] for episode_id in requested_order]
        return ordered[: int(max_samples)]
    if not selected:
        raise ValueError("no held-out validation episodes were selected")
    return list(selected.values())


def load_heldout_episode_starts(
    raw_parquet: str | Path,
    *,
    validation_fraction: float,
    split_seed: int,
    max_samples: int,
    episode_ids: Optional[Sequence[str]] = None,
) -> list[HeldoutEpisodeStart]:
    return select_heldout_episode_starts(
        iter_grepseek_rows(raw_parquet),
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        max_samples=max_samples,
        episode_ids=episode_ids,
    )


_FINAL_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def hidden_answer_reference_from_raw_episode(
    row: Mapping[str, Any],
) -> HiddenAnswerReference:
    """Strictly extract one terminal gold answer from a raw trajectory."""

    episode_id, messages = _validate_messages(row)
    final_message = messages[-1]
    if final_message["role"] != "assistant":
        raise ValueError(
            f"episode {episode_id!r}: final gold message must be assistant"
        )
    content = final_message["content"]
    if content.count("<answer>") != 1 or content.count("</answer>") != 1:
        raise ValueError(
            f"episode {episode_id!r}: final gold assistant must contain exactly "
            "one <answer>...</answer>"
        )
    matches = list(_FINAL_ANSWER_PATTERN.finditer(content))
    if len(matches) != 1:
        raise ValueError(
            f"episode {episode_id!r}: final gold answer tags are malformed"
        )
    match = matches[0]
    if content[match.end() :].strip():
        raise ValueError(
            f"episode {episode_id!r}: content follows final gold </answer>"
        )
    answer_text = match.group(1).strip()
    if not answer_text:
        raise ValueError(f"episode {episode_id!r}: final gold answer is empty")
    return HiddenAnswerReference(
        episode_id=episode_id,
        answer_text=answer_text,
        final_assistant_content=content,
    )


def collect_hidden_answer_references(
    rows: Iterable[Mapping[str, Any]],
    episode_ids: Sequence[str],
) -> dict[str, HiddenAnswerReference]:
    """Collect exactly one strict reference for every requested episode."""

    requested = {str(episode_id) for episode_id in episode_ids}
    if not requested:
        raise ValueError("at least one hidden-label episode id is required")
    if len(requested) != len(episode_ids):
        raise ValueError("hidden-label episode ids must be unique")
    references: dict[str, HiddenAnswerReference] = {}
    for row in rows:
        episode_id = str(row.get("id"))
        if episode_id not in requested:
            continue
        if episode_id in references:
            raise ValueError(f"duplicate hidden-label raw episode: {episode_id!r}")
        references[episode_id] = hidden_answer_reference_from_raw_episode(row)
    missing = sorted(requested - set(references))
    if missing:
        raise KeyError(f"hidden answer references were not found: {missing[:5]}")
    return references


def load_hidden_answer_references(
    raw_parquet: str | Path,
    episode_ids: Sequence[str],
) -> dict[str, HiddenAnswerReference]:
    """Second-pass raw-data read; call only after all rollout generation."""

    return collect_hidden_answer_references(
        iter_grepseek_rows(raw_parquet), episode_ids
    )


def evaluate_hidden_answers(
    records: Sequence[Mapping[str, Any]],
    references: Mapping[str, HiddenAnswerReference],
) -> list[dict[str, Any]]:
    """Attach post-hoc answer labels; unanswered episodes receive zero."""

    record_ids = [str(record["episode_id"]) for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("rollout records contain duplicate episode ids")
    if set(record_ids) != set(references):
        raise ValueError(
            "rollout records and hidden references have different episodes"
        )
    evaluated: list[dict[str, Any]] = []
    for record in records:
        if record.get("gold_future_messages_used") is not False:
            raise ValueError(
                "cannot score a rollout that may have seen gold future data"
            )
        episode_id = str(record["episode_id"])
        reference = references[episode_id]
        predicted = record.get("answer_text")
        answered = (
            record.get("stop_reason") == "answered"
            and isinstance(predicted, str)
            and bool(predicted.strip())
        )
        if record.get("stop_reason") == "answered" and not answered:
            raise ValueError(f"answered episode {episode_id!r} has no answer text")
        normalized_exact = bool(
            answered
            and normalize_answer(predicted) == normalize_answer(reference.answer_text)
        )
        token_f1 = (
            normalized_answer_token_f1(predicted, reference.answer_text)
            if answered
            else 0.0
        )
        scored = copy.deepcopy(dict(record))
        scored["posthoc_hidden_reference"] = reference.to_json()
        scored["posthoc_answer_comparison"] = {
            "answered": answered,
            "normalized_exact": normalized_exact,
            "token_f1": token_f1,
            "unconditional_normalized_exact": normalized_exact,
            "unconditional_token_f1": token_f1,
        }
        evaluated.append(scored)
    return evaluated


def _validate_rollout_history(
    episode_id: str, messages: Sequence[Mapping[str, str]]
) -> list[dict[str, str]]:
    _, normalized = _validate_messages(
        {"id": str(episode_id), "messages": list(messages)}
    )
    if len(normalized) < 2 or len(normalized) % 2:
        raise ValueError(
            "rollout history must end immediately before an assistant turn"
        )
    for index in range(2, len(normalized)):
        expected = {"assistant"} if index % 2 == 0 else {"tool", "user"}
        if normalized[index]["role"] not in expected:
            raise ValueError(
                f"rollout history role {index} must be one of "
                f"{sorted(expected)!r}, got "
                f"{normalized[index]['role']!r}"
            )
    return normalized


def _rollout_source_prompt_and_spans(
    tokenizer: Any,
    history: Sequence[Mapping[str, str]],
    parsed_tools: Mapping[int, Mapping[str, Any]],
    source_tool_indices: Sequence[int],
    source_texts: Sequence[str],
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Locate only nonempty stdout spans; empty stdout remains discrete."""

    if len(source_tool_indices) != len(source_texts):
        raise ValueError("rollout source index/text arity mismatch")
    marker_by_index: dict[int, str] = {}
    all_content = "\0".join(message["content"] for message in history)
    for ordinal, tool_index in enumerate(source_tool_indices):
        marker = f"DFLASH_ROLLOUT_BOUNDARY_{ordinal:06d}_DO_NOT_TOKENIZE"
        while marker in all_content:
            marker += "_X"
        marker_by_index[tool_index] = marker

    marked_history: list[dict[str, str]] = []
    raw_history: list[dict[str, str]] = []
    for index, message in enumerate(history):
        if message["role"] != "tool":
            marked_history.append(dict(message))
            raw_history.append(dict(message))
            continue
        payload = parsed_tools[index]
        stdout = str(payload["stdout"])
        raw_history.append(
            {"role": "tool", "content": _canonical_tool_content(payload, stdout)}
        )
        marked_history.append(
            {
                "role": "tool",
                "content": _canonical_tool_content(
                    payload,
                    marker_by_index[index] if index in marker_by_index else stdout,
                ),
            }
        )

    raw_render = _render_chat(tokenizer, raw_history, add_generation_prompt=True)
    marked_render = _render_chat(tokenizer, marked_history, add_generation_prompt=True)
    static_parts: list[str] = []
    cursor = 0
    for tool_index in source_tool_indices:
        marker = marker_by_index[tool_index]
        start = marked_render.find(marker, cursor)
        if start < 0 or marked_render.find(marker, start + len(marker)) >= 0:
            raise ValueError("rollout source boundary marker was not rendered once")
        static_parts.append(marked_render[cursor:start])
        cursor = start + len(marker)
    static_parts.append(marked_render[cursor:])

    rebuilt: list[str] = []
    stdout_char_spans: list[list[int]] = []
    raw_cursor = 0
    for ordinal, source_text in enumerate(source_texts):
        static = static_parts[ordinal]
        escaped = json.dumps(str(source_text), ensure_ascii=False)[1:-1]
        rebuilt.extend((static, escaped))
        raw_cursor += len(static)
        stdout_char_spans.append([raw_cursor, raw_cursor + len(escaped)])
        raw_cursor += len(escaped)
    rebuilt.append(static_parts[-1])
    if "".join(rebuilt) != raw_render:
        raise ValueError(
            "rollout canonical source render is not exact marker substitution"
        )

    source_prompt_ids, offsets = _rendered_ids_and_offsets(tokenizer, raw_render)
    if source_prompt_ids != _apply_chat_ids(
        tokenizer, raw_history, add_generation_prompt=True
    ):
        raise ValueError("rollout source render differs from native chat ids")
    evidence_spans: list[list[int]] = []
    for char_start, char_end in stdout_char_spans:
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > char_start and token_start < char_end
        ]
        if not overlapping:
            raise ValueError("nonempty rollout stdout has no source-prompt token")
        start, end = overlapping[0], overlapping[-1] + 1
        if overlapping != list(range(start, end)):
            raise ValueError("rollout stdout token span is not contiguous")
        evidence_spans.append([start, end])
    return source_prompt_ids, evidence_spans, stdout_char_spans


def build_rollout_feature(
    messages: Sequence[Mapping[str, str]],
    tokenizer: Any,
    *,
    episode_id: str,
    round_index: int,
) -> dict[str, Any]:
    """Build one next-turn feature solely from generated rollout history."""

    history = _validate_rollout_history(episode_id, messages)
    parsed_tools: dict[int, dict[str, Any]] = {}
    source_tool_indices: list[int] = []
    source_observation_round_indices: list[int] = []
    sources: list[list[int]] = []
    source_texts: list[str] = []
    tool_observations: list[tuple[int, int, dict[str, Any], list[int]]] = []
    user_feedback_observation_count = 0
    for index, message in enumerate(history):
        if index >= 2 and message["role"] == "user":
            user_feedback_observation_count += 1
        if message["role"] != "tool":
            continue
        if index < 3 or index % 2 != 1 or history[index - 1]["role"] != "assistant":
            raise ValueError("tool observation is not paired with an assistant round")
        observation_round_index = ((index - 2) // 2) + 1
        payload = parse_tool_response(
            message["content"], episode_id=episode_id, message_index=index
        )
        parsed_tools[index] = payload
        source_ids = _encode(tokenizer, payload["stdout"])
        tool_observations.append((index, observation_round_index, payload, source_ids))

    # The newest observation remains discrete for action_{k+1}.  Once that
    # action and its next observation exist, the former observation is no
    # longer latest and may be compressed in the prefix before action_{k+2}.
    # This is the only causal policy that guarantees a final answer always sees
    # its immediately preceding tool output verbatim without knowing the future.
    latest_tool_index = tool_observations[-1][0] if tool_observations else None
    for index, source_round, payload, source_ids in tool_observations:
        if source_ids and index != latest_tool_index:
            source_tool_indices.append(index)
            source_observation_round_indices.append(source_round)
            sources.append(source_ids)
            source_texts.append(str(payload["stdout"]))

    marker = _choose_soft_marker(tokenizer)
    student_history: list[dict[str, str]] = []
    for index, message in enumerate(history):
        if message["role"] != "tool":
            student_history.append(dict(message))
            continue
        payload = parsed_tools[index]
        student_history.append(
            {
                "role": "tool",
                "content": _canonical_tool_content(
                    payload,
                    marker if index in source_tool_indices else payload["stdout"],
                ),
            }
        )
    discrete_segments = _split_chat_at_markers(
        tokenizer,
        student_history,
        marker=marker,
        expected_markers=len(sources),
        add_generation_prompt=True,
        target_message=None,
    )

    if sources:
        source_prompt_ids, evidence_spans, evidence_char_spans = (
            _rollout_source_prompt_and_spans(
                tokenizer,
                history,
                parsed_tools,
                source_tool_indices,
                source_texts,
            )
        )
    else:
        source_prompt_ids = _apply_chat_ids(
            tokenizer, student_history, add_generation_prompt=True
        )
        evidence_spans = []
        evidence_char_spans = []
    if len(discrete_segments) != len(sources) + 1:
        raise AssertionError("rollout discrete/source interleave arity mismatch")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("rollout tokenizer must define eos_token_id")
    return {
        "id": f"{episode_id}:rollout-{int(round_index):03d}",
        "episode_id": str(episode_id),
        "num_source_segments": len(sources),
        "discrete_segments_input_ids": discrete_segments,
        "source_segments_input_ids": sources,
        "source_prompt_input_ids": source_prompt_ids,
        "source_prompt_evidence_spans": evidence_spans,
        "source_prompt_stdout_char_spans": evidence_char_spans,
        # A constant dummy target lets the existing training collator build its
        # exact scatter ABI.  It is immediately removed before generation and
        # contains no episode-dependent or gold-future information.
        "target_input_ids": [int(eos_token_id)],
        "rollout_history_message_count": len(history),
        "rollout_source_token_count": sum(len(source) for source in sources),
        "source_observation_round_indices": source_observation_round_indices,
        "observation_compression_policy": "causal_keep_latest_raw",
        "completed_observation_round_count": (len(history) - 2) // 2,
        "observation_count": len(tool_observations),
        "tool_observation_count": len(tool_observations),
        "user_feedback_observation_count": user_feedback_observation_count,
        "latest_raw_observation_round_index": (
            tool_observations[-1][1] if tool_observations else None
        ),
        "latest_raw_observation_token_count": (
            len(tool_observations[-1][3]) if tool_observations else 0
        ),
    }


@dataclass(frozen=True)
class RolloutGeneration:
    token_ids: tuple[int, ...]
    generated_text: str
    assistant_content: str
    parsed: ParsedGrepSeekContinuation
    finished_on_eos: bool
    hit_generation_cap: bool
    compression: Mapping[str, Any]
    finish_reason: Any = None

    def to_json(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "token_count": len(self.token_ids),
            "generated_text": self.generated_text,
            "assistant_content": self.assistant_content,
            "parsed": self.parsed.to_json(),
            "finished_on_eos": self.finished_on_eos,
            "hit_generation_cap": self.hit_generation_cap,
            "finish_reason": _finish_reason_kind(self.finish_reason),
            "compression": dict(self.compression),
        }


class RolloutTurnGenerator(Protocol):
    def generate_next(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Optional[RolloutEventSink] = None,
    ) -> RolloutGeneration: ...


class RolloutToolRunner(Protocol):
    def execute(self, command: str) -> ToolResult: ...


def _prepare_tool_result_for_next_round(
    generator: RolloutTurnGenerator,
    *,
    messages: Sequence[Mapping[str, str]],
    tool_result: ToolResult,
    episode_id: str,
    round_index: int,
) -> tuple[ToolResult, Optional[dict[str, Any]]]:
    """Apply an optional generator-specific admission policy to a tool result.

    The checkpoint-native working-memory generator uses this hook to ensure a
    newly completed assistant/tool exchange can be consumed by the *next*
    recursive encoder update.  Other generators retain the released seven-key
    tool-result contract byte-for-byte.
    """

    hook = getattr(generator, "prepare_tool_result_for_next_round", None)
    if hook is None:
        return tool_result, None
    if not callable(hook):
        raise TypeError(
            "generator prepare_tool_result_for_next_round must be callable"
        )
    prepared = hook(
        messages=tuple(copy.deepcopy(messages)),
        tool_result=tool_result,
        episode_id=str(episode_id),
        round_index=int(round_index),
    )
    telemetry: Optional[Mapping[str, Any]] = None
    if isinstance(prepared, tuple):
        if len(prepared) != 2:
            raise TypeError(
                "tool-result preparation must return ToolResult or "
                "(ToolResult, telemetry)"
            )
        prepared, telemetry = prepared
    if not isinstance(prepared, ToolResult):
        raise TypeError("tool-result preparation must return ToolResult")
    if telemetry is not None and not isinstance(telemetry, Mapping):
        raise TypeError("tool-result preparation telemetry must be an object")
    return prepared, None if telemetry is None else dict(telemetry)


def _prepare_answer_for_next_round(
    generator: RolloutTurnGenerator,
    *,
    messages: Sequence[Mapping[str, str]],
    answer_text: str,
    episode_id: str,
    round_index: int,
) -> Optional[tuple[str, dict[str, Any]]]:
    """Let a generator defer one candidate answer with a real user turn.

    This optional hook is absent from ordinary generators, preserving the
    released terminal-answer behavior.  A non-``None`` return must be either a
    nonempty user-message string or ``(user_message, telemetry)``.  The
    rollout records the candidate generation, appends assistant then user, and
    continues the same episode without exposing any hidden reference.
    """

    hook = getattr(generator, "prepare_answer_for_next_round", None)
    if hook is None:
        return None
    if not callable(hook):
        raise TypeError("generator prepare_answer_for_next_round must be callable")
    prepared = hook(
        messages=tuple(copy.deepcopy(messages)),
        answer_text=str(answer_text),
        episode_id=str(episode_id),
        round_index=int(round_index),
    )
    if prepared is None:
        return None
    telemetry: Optional[Mapping[str, Any]] = None
    if isinstance(prepared, tuple):
        if len(prepared) != 2:
            raise TypeError(
                "answer preparation must return a user message, None, or "
                "(user message, telemetry)"
            )
        prepared, telemetry = prepared
    if not isinstance(prepared, str) or not prepared.strip():
        raise TypeError("answer preparation user message must be nonempty")
    if telemetry is not None and not isinstance(telemetry, Mapping):
        raise TypeError("answer preparation telemetry must be an object")
    return prepared.strip(), {} if telemetry is None else dict(telemetry)


def _run_rollout_episode_core(
    start: HeldoutEpisodeStart,
    *,
    generator: RolloutTurnGenerator,
    tool_runner: RolloutToolRunner,
    max_rounds: Optional[int],
    max_consecutive_recoverable_tool_schema_errors: Optional[int] = (
        MAX_CONSECUTIVE_RECOVERABLE_TOOL_SCHEMA_ERRORS
    ),
    event_sink: Optional[RolloutEventSink] = None,
    allow_tool_calls: bool = True,
) -> dict[str, Any]:
    """Run one sequential tool-use episode with explicit terminal reasons."""

    if max_rounds is not None and (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds <= 0
    ):
        raise ValueError("max_rounds must be a positive integer or None")
    if max_consecutive_recoverable_tool_schema_errors is not None and (
        isinstance(max_consecutive_recoverable_tool_schema_errors, bool)
        or not isinstance(max_consecutive_recoverable_tool_schema_errors, int)
        or max_consecutive_recoverable_tool_schema_errors <= 0
    ):
        raise ValueError(
            "max_consecutive_recoverable_tool_schema_errors must be a positive "
            "integer or None"
        )
    if event_sink is not None and not callable(event_sink):
        raise TypeError("event_sink must be callable or None")
    if not isinstance(allow_tool_calls, bool):
        raise TypeError("allow_tool_calls must be boolean")
    messages = [dict(message) for message in start.messages]
    rounds: list[dict[str, Any]] = []
    stop_reason: Optional[str] = None
    answer_text: Optional[str] = None
    consecutive_recoverable_tool_schema_errors = 0
    started = time.monotonic()

    round_indices: Iterable[int]
    if max_rounds is None:
        # Interactive context-limited mode.  Each generation still fails
        # closed once its SGLang input no longer fits the configured context.
        round_indices = itertools.count(1)
    else:
        round_indices = range(1, max_rounds + 1)
    for round_index in round_indices:
        round_record: dict[str, Any] = {"round_index": round_index}
        _emit_rollout_event(
            event_sink,
            event="generation_started",
            episode_id=start.episode_id,
            round_index=round_index,
            payload={"message_count": len(messages)},
        )
        try:
            generation_messages = tuple(copy.deepcopy(messages))
            if event_sink is None:
                # Preserve compatibility with pre-event generators when the
                # caller does not request instrumentation.
                generated = generator.generate_next(
                    generation_messages,
                    episode_id=start.episode_id,
                    round_index=round_index,
                )
            else:
                generated = generator.generate_next(
                    generation_messages,
                    episode_id=start.episode_id,
                    round_index=round_index,
                    event_sink=event_sink,
                )
        except Exception as exc:  # keep one bad episode from aborting a smoke set
            round_record["generation_exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            rounds.append(round_record)
            stop_reason = "generation_exception"
            break
        round_record["generation"] = generated.to_json()
        _emit_rollout_event(
            event_sink,
            event="generation_finished",
            episode_id=start.episode_id,
            round_index=round_index,
            payload={"generation": round_record["generation"]},
        )
        parsed = generated.parsed
        if not generated.finished_on_eos or not parsed.eos_reached:
            rounds.append(round_record)
            finish_kind = _finish_reason_kind(generated.finish_reason)
            if finish_kind in {"abort", "aborted", "cancel", "cancelled"}:
                stop_reason = "generation_aborted"
            elif generated.hit_generation_cap:
                stop_reason = "generation_cap_without_eos"
            else:
                stop_reason = "generation_incomplete_without_eos"
            break
        if parsed.kind == "tool_call" and parsed.tool_calls:
            if not allow_tool_calls:
                messages.append(
                    {"role": "assistant", "content": generated.assistant_content}
                )
                rounds.append(round_record)
                stop_reason = "tool_call_disallowed"
                break

            messages.append(
                {"role": "assistant", "content": generated.assistant_content}
            )
            valid_calls = [
                call
                for call in parsed.tool_calls
                if call.tool_schema_valid and bool(call.tool_command)
            ]
            batch_command = f"tool_batch[{len(parsed.tool_calls)}]"
            _emit_rollout_event(
                event_sink,
                event="tool_started",
                episode_id=start.episode_id,
                round_index=round_index,
                payload={
                    "command": batch_command,
                    "tool_call_count": len(parsed.tool_calls),
                },
            )
            call_records: list[dict[str, Any]] = []
            try:
                if valid_calls:
                    update_compressed_memory = getattr(
                        tool_runner, "update_compressed_episode_memory", None
                    )
                    if update_compressed_memory is not None:
                        if not callable(update_compressed_memory):
                            raise TypeError(
                                "tool runner update_compressed_episode_memory "
                                "must be callable"
                            )
                        update_compressed_memory(
                            episode_id=start.episode_id,
                            messages=tuple(copy.deepcopy(messages)),
                            round_index=round_index,
                            compression=copy.deepcopy(generated.compression),
                        )
                    else:
                        update_episode_memory = getattr(
                            tool_runner, "update_episode_memory", None
                        )
                        if update_episode_memory is not None and not callable(
                            update_episode_memory
                        ):
                            raise TypeError(
                                "tool runner update_episode_memory must be callable"
                            )
                        if update_episode_memory is not None:
                            update_episode_memory(
                                episode_id=start.episode_id,
                                messages=tuple(copy.deepcopy(messages)),
                                round_index=round_index,
                            )

                for call in parsed.tool_calls:
                    if call.tool_schema_valid and call.tool_command:
                        result = tool_runner.execute(call.tool_command)
                        if not isinstance(result, ToolResult):
                            raise TypeError(
                                "tool runner execute() must return ToolResult"
                            )
                        call_records.append(
                            _tool_record(
                                command=call.tool_command,
                                result=result,
                                executed=True,
                                call_index=call.call_index,
                            )
                        )
                        continue
                    validation_result, malformed_retry = (
                        _recoverable_tool_validation_result(call)
                    )
                    if validation_result is None:
                        raise ValueError(
                            f"tool call {call.call_index} has no safe validation "
                            "observation"
                        )
                    call_records.append(
                        _tool_record(
                            command=validation_result.command,
                            result=validation_result,
                            executed=False,
                            call_index=call.call_index,
                            schema_repair=True,
                            malformed_tool_retry=malformed_retry,
                            schema_error=call.tool_schema_error,
                        )
                    )
                tool_result = _aggregate_tool_call_results(call_records)
                tool_context_fit = None
                if max_rounds is None or round_index < max_rounds:
                    tool_result, tool_context_fit = (
                        _prepare_tool_result_for_next_round(
                            generator,
                            messages=messages,
                            tool_result=tool_result,
                            episode_id=start.episode_id,
                            round_index=round_index,
                        )
                    )
            except Exception as exc:
                round_record["tool_calls"] = call_records
                round_record["tool_exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                _emit_rollout_event(
                    event_sink,
                    event="tool_finished",
                    episode_id=start.episode_id,
                    round_index=round_index,
                    payload={
                        "command": batch_command,
                        "exception": dict(round_record["tool_exception"]),
                    },
                )
                rounds.append(round_record)
                stop_reason = "tool_runner_exception"
                break

            messages.append({"role": "tool", "content": tool_result.to_payload()})
            executed_count = sum(
                bool(call_record["executed"]) for call_record in call_records
            )
            round_record["tool_calls"] = call_records
            round_record["tool"] = _tool_record(
                command=batch_command,
                result=tool_result,
                executed=executed_count > 0,
            )
            round_record["tool"].update(
                {
                    "batch": True,
                    "tool_call_count": len(call_records),
                    "executed_tool_call_count": executed_count,
                    "nonexecuted_tool_call_count": len(call_records)
                    - executed_count,
                    "validation_failed": any(
                        bool(call_record["validation_failed"])
                        for call_record in call_records
                    ),
                }
            )
            if tool_context_fit is not None:
                round_record["tool"]["context_fit"] = tool_context_fit
            _emit_rollout_event(
                event_sink,
                event="tool_finished",
                episode_id=start.episode_id,
                round_index=round_index,
                payload={"tool": dict(round_record["tool"])},
            )
            rounds.append(round_record)
            if executed_count:
                consecutive_recoverable_tool_schema_errors = 0
            else:
                consecutive_recoverable_tool_schema_errors += 1
                if (
                    max_consecutive_recoverable_tool_schema_errors is not None
                    and consecutive_recoverable_tool_schema_errors
                    >= max_consecutive_recoverable_tool_schema_errors
                ):
                    stop_reason = "invalid_tool_schema"
                    break
            if max_rounds is not None and round_index == max_rounds:
                stop_reason = "max_rounds_after_tool"
                break
            _emit_rollout_event(
                event_sink,
                event="next_round",
                episode_id=start.episode_id,
                round_index=round_index,
                payload={"next_round_index": round_index + 1},
            )
            continue
        if parsed.kind == "tool_call" and (
            not parsed.tool_schema_valid or not parsed.structure_valid
        ):
            validation_result, malformed_retry = (
                _recoverable_tool_validation_result(parsed)
            )
            if validation_result is not None:
                next_error_count = consecutive_recoverable_tool_schema_errors + 1
                error_budget_allows_next_round = (
                    max_consecutive_recoverable_tool_schema_errors is None
                    or next_error_count < max_consecutive_recoverable_tool_schema_errors
                )
                round_budget_allows_next_round = (
                    max_rounds is None or round_index < max_rounds
                )
                will_continue = (
                    error_budget_allows_next_round and round_budget_allows_next_round
                )
                messages.append(
                    {"role": "assistant", "content": generated.assistant_content}
                )
                _emit_rollout_event(
                    event_sink,
                    event="tool_started",
                    episode_id=start.episode_id,
                    round_index=round_index,
                    payload={"command": validation_result.command},
                )
                tool_context_fit = None
                try:
                    if will_continue:
                        validation_result, tool_context_fit = (
                            _prepare_tool_result_for_next_round(
                                generator,
                                messages=messages,
                                tool_result=validation_result,
                                episode_id=start.episode_id,
                                round_index=round_index,
                            )
                        )
                except Exception as exc:
                    round_record["tool_exception"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    _emit_rollout_event(
                        event_sink,
                        event="tool_finished",
                        episode_id=start.episode_id,
                        round_index=round_index,
                        payload={
                            "command": validation_result.command,
                            "exception": dict(round_record["tool_exception"]),
                        },
                    )
                    rounds.append(round_record)
                    stop_reason = "tool_runner_exception"
                    break
                messages.append(
                    {"role": "tool", "content": validation_result.to_payload()}
                )
                consecutive_recoverable_tool_schema_errors = next_error_count
                round_record["tool"] = {
                    "command": validation_result.command,
                    "result": validation_result.to_dict(),
                    "validation_failed": True,
                    "token_truncation_failed": False,
                    "timed_out": False,
                    "truncated": False,
                    "nonzero_exit": True,
                    "schema_repair": True,
                    "malformed_tool_retry": malformed_retry,
                    "schema_error": parsed.tool_schema_error,
                }
                if tool_context_fit is not None:
                    round_record["tool"]["context_fit"] = tool_context_fit
                _emit_rollout_event(
                    event_sink,
                    event="tool_finished",
                    episode_id=start.episode_id,
                    round_index=round_index,
                    payload={"tool": dict(round_record["tool"])},
                )
                rounds.append(round_record)
                if (
                    max_consecutive_recoverable_tool_schema_errors is not None
                    and consecutive_recoverable_tool_schema_errors
                    >= max_consecutive_recoverable_tool_schema_errors
                ):
                    stop_reason = "invalid_tool_schema"
                    break
                if max_rounds is not None and round_index == max_rounds:
                    stop_reason = "max_rounds_after_tool"
                    break
                _emit_rollout_event(
                    event_sink,
                    event="next_round",
                    episode_id=start.episode_id,
                    round_index=round_index,
                    payload={"next_round_index": round_index + 1},
                )
                continue
            rounds.append(round_record)
            stop_reason = "invalid_tool_schema"
            break
        if not parsed.structure_valid or not parsed.complete_valid:
            rounds.append(round_record)
            stop_reason = "invalid_generation_structure"
            break
        if parsed.kind == "answer":
            messages.append(
                {"role": "assistant", "content": generated.assistant_content}
            )
            reconsideration = _prepare_answer_for_next_round(
                generator,
                messages=messages,
                answer_text=str(parsed.answer_text or ""),
                episode_id=start.episode_id,
                round_index=round_index,
            )
            if reconsideration is not None:
                user_message, telemetry = reconsideration
                messages.append({"role": "user", "content": user_message})
                round_record["answer_reconsideration"] = {
                    "user_message": user_message,
                    "telemetry": telemetry,
                }
                rounds.append(round_record)
                consecutive_recoverable_tool_schema_errors = 0
                _emit_rollout_event(
                    event_sink,
                    event="answer_reconsideration",
                    episode_id=start.episode_id,
                    round_index=round_index,
                    payload={
                        "user_message": user_message,
                        "telemetry": telemetry,
                    },
                )
                if max_rounds is not None and round_index == max_rounds:
                    stop_reason = "max_rounds_after_answer_reconsideration"
                    break
                _emit_rollout_event(
                    event_sink,
                    event="next_round",
                    episode_id=start.episode_id,
                    round_index=round_index,
                    payload={"next_round_index": round_index + 1},
                )
                continue
            answer_text = parsed.answer_text
            rounds.append(round_record)
            stop_reason = "answered"
            _emit_rollout_event(
                event_sink,
                event="final_answer",
                episode_id=start.episode_id,
                round_index=round_index,
                payload={"answer_text": answer_text},
            )
            break
        if parsed.kind == "tool_call" and not allow_tool_calls:
            messages.append(
                {"role": "assistant", "content": generated.assistant_content}
            )
            rounds.append(round_record)
            stop_reason = "tool_call_disallowed"
            break
        if parsed.kind != "tool_call" or not parsed.tool_command:
            rounds.append(round_record)
            stop_reason = "invalid_generation_kind"
            break

        consecutive_recoverable_tool_schema_errors = 0
        messages.append({"role": "assistant", "content": generated.assistant_content})
        _emit_rollout_event(
            event_sink,
            event="tool_started",
            episode_id=start.episode_id,
            round_index=round_index,
            payload={"command": parsed.tool_command},
        )
        try:
            update_compressed_memory = getattr(
                tool_runner, "update_compressed_episode_memory", None
            )
            if update_compressed_memory is not None:
                if not callable(update_compressed_memory):
                    raise TypeError(
                        "tool runner update_compressed_episode_memory must be callable"
                    )
                update_compressed_memory(
                    episode_id=start.episode_id,
                    messages=tuple(copy.deepcopy(messages)),
                    round_index=round_index,
                    compression=copy.deepcopy(generated.compression),
                )
            else:
                update_episode_memory = getattr(
                    tool_runner, "update_episode_memory", None
                )
                if update_episode_memory is not None and not callable(
                    update_episode_memory
                ):
                    raise TypeError(
                        "tool runner update_episode_memory must be callable"
                    )
                if update_episode_memory is not None:
                    update_episode_memory(
                        episode_id=start.episode_id,
                        messages=tuple(copy.deepcopy(messages)),
                        round_index=round_index,
                    )
            tool_result = tool_runner.execute(parsed.tool_command)
            if not isinstance(tool_result, ToolResult):
                raise TypeError("tool runner execute() must return ToolResult")
            tool_context_fit = None
            if max_rounds is None or round_index < max_rounds:
                tool_result, tool_context_fit = _prepare_tool_result_for_next_round(
                    generator,
                    messages=messages,
                    tool_result=tool_result,
                    episode_id=start.episode_id,
                    round_index=round_index,
                )
        except Exception as exc:
            round_record["tool_exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _emit_rollout_event(
                event_sink,
                event="tool_finished",
                episode_id=start.episode_id,
                round_index=round_index,
                payload={
                    "command": parsed.tool_command,
                    "exception": dict(round_record["tool_exception"]),
                },
            )
            rounds.append(round_record)
            stop_reason = "tool_runner_exception"
            break
        tool_payload = tool_result.to_payload()
        messages.append({"role": "tool", "content": tool_payload})
        round_record["tool"] = {
            "command": parsed.tool_command,
            "result": tool_result.to_dict(),
            "validation_failed": tool_result.exit_code == -2,
            "token_truncation_failed": tool_result.stdout
            == "[tool output unavailable: token truncation failed]",
            "timed_out": tool_result.timed_out,
            "truncated": tool_result.truncated,
            "nonzero_exit": tool_result.exit_code != 0,
        }
        if tool_context_fit is not None:
            round_record["tool"]["context_fit"] = tool_context_fit
        _emit_rollout_event(
            event_sink,
            event="tool_finished",
            episode_id=start.episode_id,
            round_index=round_index,
            payload={"tool": dict(round_record["tool"])},
        )
        rounds.append(round_record)
        if max_rounds is not None and round_index == max_rounds:
            stop_reason = "max_rounds_after_tool"
            break
        _emit_rollout_event(
            event_sink,
            event="next_round",
            episode_id=start.episode_id,
            round_index=round_index,
            payload={"next_round_index": round_index + 1},
        )

    if stop_reason is None:
        raise AssertionError("rollout loop ended without a terminal reason")
    record = {
        "schema": ROLLOUT_SCHEMA,
        "episode_id": start.episode_id,
        "heldout_initial_messages": [dict(message) for message in start.messages],
        "gold_future_messages_used": False,
        "rounds": rounds,
        "round_count": len(rounds),
        "stop_reason": stop_reason,
        "success": stop_reason == "answered",
        "answer_text": answer_text,
        "final_messages": messages,
        "elapsed_seconds": time.monotonic() - started,
    }
    _emit_rollout_event(
        event_sink,
        event="stopped",
        episode_id=start.episode_id,
        round_index=int(rounds[-1]["round_index"]),
        payload={
            "stop_reason": stop_reason,
            "success": stop_reason == "answered",
            "answer_text": answer_text,
            "round_count": len(rounds),
        },
    )
    return record


def run_rollout_episode(
    start: HeldoutEpisodeStart,
    *,
    generator: RolloutTurnGenerator,
    tool_runner: RolloutToolRunner,
    max_rounds: Optional[int],
    max_consecutive_recoverable_tool_schema_errors: Optional[int] = (
        MAX_CONSECUTIVE_RECOVERABLE_TOOL_SCHEMA_ERRORS
    ),
    event_sink: Optional[RolloutEventSink] = None,
    allow_tool_calls: bool = True,
) -> dict[str, Any]:
    """Run an episode with optional tool-runner lifecycle isolation."""

    begin_episode = getattr(tool_runner, "begin_episode", None)
    end_episode = getattr(tool_runner, "end_episode", None)
    if (begin_episode is None) != (end_episode is None):
        raise TypeError(
            "tool runner must implement both begin_episode and end_episode"
        )
    if begin_episode is not None and not callable(begin_episode):
        raise TypeError("tool runner begin_episode must be callable")
    if end_episode is not None and not callable(end_episode):
        raise TypeError("tool runner end_episode must be callable")
    if begin_episode is not None:
        begin_episode(start.episode_id)
    try:
        return _run_rollout_episode_core(
            start,
            generator=generator,
            tool_runner=tool_runner,
            max_rounds=max_rounds,
            max_consecutive_recoverable_tool_schema_errors=(
                max_consecutive_recoverable_tool_schema_errors
            ),
            event_sink=event_sink,
            allow_tool_calls=allow_tool_calls,
        )
    finally:
        if end_episode is not None:
            end_episode(start.episode_id)


class GrepSeekRolloutModelGenerator:
    """Batch-one trained-arm generator for the rollout state machine."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        runtime: GrepSeekFrozenTargetRuntime,
        compressor: GrepSeekDFlashCompressor,
        compression_ratio: int,
        max_encoder_tokens: Optional[int],
        pad_to_multiple_of: Optional[int],
        device: torch.device,
        max_new_tokens: int,
        eos_token_id: int = EOS_TOKEN_ID,
        collect_diagnostics: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.compressor = compressor
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.eos_token_id = int(eos_token_id)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.collator = GrepSeekCompressionCollator(
            tokenizer,
            compression_ratio=int(compression_ratio),
            max_encoder_tokens=max_encoder_tokens,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def _move(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

    def _sync_diagnostics(self) -> None:
        """Synchronize only for explicitly requested wall-clock diagnostics."""

        if self.collect_diagnostics and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

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
        source_observation_round_indices = [
            int(value)
            for value in feature.get("source_observation_round_indices", [])
        ]
        if len(source_observation_round_indices) != source_count:
            raise ValueError(
                "rollout observation-round mapping disagrees with source count"
            )
        source_encode_seconds = 0.0
        compressor_seconds = 0.0
        if source_count:
            batch = self._move(self.collator([feature]))
            self._sync_diagnostics()
            source_encode_started = time.perf_counter()
            source_features = self.runtime.encode_source(
                batch["source_input_ids"], batch["source_attention_mask"]
            )
            self._sync_diagnostics()
            if self.collect_diagnostics:
                source_encode_seconds = time.perf_counter() - source_encode_started
            flattened = _flatten_encoded_evidence(source_features, batch)
            self._sync_diagnostics()
            compressor_started = time.perf_counter()
            compressed = self.compressor(
                flattened.features,
                flattened.attention_mask,
                source_token_lengths=batch["source_lengths"],
            )
            self._sync_diagnostics()
            if self.collect_diagnostics:
                compressor_seconds = time.perf_counter() - compressor_started
            _validate_memory_contract(compressed, batch)
            prefix = build_interleaved_generation_prefix(
                batch, pad_token_id=int(self.tokenizer.pad_token_id)
            )
            memory_slots = int(compressed.slot_counts.sum().item())
            source_encoder_tokens = int(batch["source_attention_mask"].sum().item())
            student_prefix_tokens = int(prefix.attention_mask.sum().item())
        else:
            prompt_ids = feature["discrete_segments_input_ids"][0]
            input_ids, attention_mask = build_discrete_generation_prefix(
                [prompt_ids],
                pad_token_id=int(self.tokenizer.pad_token_id),
                device=self.device,
            )
            memory_slots = 0
            source_encoder_tokens = 0
            student_prefix_tokens = len(prompt_ids)

        compression: dict[str, Any] = {
            "source_segments": source_count,
            "source_tokens": source_tokens,
            "memory_slots": memory_slots,
            "effective_compression_ratio": (
                source_tokens / memory_slots if memory_slots else None
            ),
            "source_encoder_tokens": source_encoder_tokens,
            "student_prefix_tokens": student_prefix_tokens,
            "latest_raw_observation_round_index": feature.get(
                "latest_raw_observation_round_index"
            ),
            "latest_raw_observation_token_count": int(
                feature.get("latest_raw_observation_token_count", 0)
            ),
            "observation_compression_policy": feature.get(
                "observation_compression_policy"
            ),
            "observation_count": int(feature.get("observation_count", 0)),
            "source_observation_round_indices": source_observation_round_indices,
        }
        if self.collect_diagnostics:
            if source_count:
                valid_memory = compressed.embeddings[compressed.attention_mask.bool()]
                valid_memory_float = valid_memory.detach().float()
                valid_memory_rms = _finite_diagnostic(
                    valid_memory_float.square().mean().sqrt().item()
                )
                valid_memory_mean_l2 = _finite_diagnostic(
                    valid_memory_float.norm(dim=-1).mean().item()
                )
                memory_tensor_shape: Optional[list[int]] = [
                    int(size) for size in compressed.embeddings.shape
                ]
                valid_memory_shape = [int(size) for size in valid_memory.shape]
                memory_dtype: Optional[str] = str(
                    compressed.embeddings.dtype
                ).removeprefix("torch.")
                source_token_lengths = [
                    int(value) for value in batch["source_lengths"].tolist()
                ]
                slot_counts = [int(value) for value in compressed.slot_counts.tolist()]
            else:
                hidden_size = int(self.runtime.text_config.hidden_size)
                source_token_lengths = []
                slot_counts = []
                memory_tensor_shape = None
                valid_memory_shape = [0, hidden_size]
                memory_dtype = None
                valid_memory_rms = None
                valid_memory_mean_l2 = None
            compression.update(
                {
                    "source_token_lengths": source_token_lengths,
                    "slot_counts": slot_counts,
                    "memory_tensor_shape": memory_tensor_shape,
                    "valid_memory_shape": valid_memory_shape,
                    "memory_dtype": memory_dtype,
                    "valid_memory_rms": valid_memory_rms,
                    "valid_memory_mean_l2": valid_memory_mean_l2,
                    "source_encode_seconds": _finite_diagnostic(source_encode_seconds),
                    "compressor_seconds": _finite_diagnostic(compressor_seconds),
                    # Populated after the target finishes; the preceding
                    # compression event correctly observes this as not-yet-run.
                    "target_generation_seconds": None,
                }
            )

        _emit_rollout_event(
            event_sink,
            event="compression_finished",
            episode_id=episode_id,
            round_index=round_index,
            payload={"compression": dict(compression)},
        )

        self._sync_diagnostics()
        target_generation_started = time.perf_counter()
        if source_count:
            output = self.runtime.greedy_generate_interleaved(
                compressed.embeddings,
                compressed.attention_mask,
                prefix.input_ids,
                prefix.attention_mask,
                prefix.memory_positions,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.eos_token_id,
            )
        else:
            output = self.runtime.greedy_generate_prompt(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=self.eos_token_id,
            )
        self._sync_diagnostics()
        if self.collect_diagnostics:
            compression["target_generation_seconds"] = _finite_diagnostic(
                time.perf_counter() - target_generation_started
            )

        length = int(output.lengths[0].item())
        raw_ids = [int(token_id) for token_id in output.token_ids[0, :length].tolist()]
        token_ids, eos_reached = trim_at_eos(raw_ids, eos_token_id=self.eos_token_id)
        finished = bool(output.finished[0].item()) and eos_reached
        body_ids = token_ids[:-1] if eos_reached else token_ids
        generated_text = _decode(self.tokenizer, token_ids)
        assistant_content = ASSISTANT_CONTENT_PREFIX + _decode(self.tokenizer, body_ids)
        parsed = parse_grepseek_continuation(
            token_ids, self.tokenizer, eos_token_id=self.eos_token_id
        )
        return RolloutGeneration(
            token_ids=tuple(token_ids),
            generated_text=generated_text,
            assistant_content=assistant_content,
            parsed=parsed,
            finished_on_eos=finished,
            hit_generation_cap=not finished,
            compression=compression,
        )


def summarize_rollouts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero rollout records")
    comparisons = [record.get("posthoc_answer_comparison") for record in records]
    if any(comparison is None for comparison in comparisons):
        raise ValueError(
            "rollout summary requires post-hoc hidden-answer evaluation for every episode"
        )
    answered_comparisons = [
        comparison for comparison in comparisons if bool(comparison["answered"])
    ]
    unconditional_em = sum(
        bool(comparison["unconditional_normalized_exact"]) for comparison in comparisons
    ) / len(records)
    unconditional_f1 = sum(
        float(comparison["unconditional_token_f1"]) for comparison in comparisons
    ) / len(records)
    stops = Counter(str(record["stop_reason"]) for record in records)
    if len(answered_comparisons) != stops.get("answered", 0):
        raise ValueError("post-hoc answered flags disagree with rollout stop reasons")
    rounds = [round_record for record in records for round_record in record["rounds"]]
    generations = [
        round_record["generation"]
        for round_record in rounds
        if "generation" in round_record
    ]
    tools = [round_record["tool"] for round_record in rounds if "tool" in round_record]
    compression = [generation["compression"] for generation in generations]
    source_tokens = sum(int(item["source_tokens"]) for item in compression)
    memory_slots = sum(int(item["memory_slots"]) for item in compression)
    return {
        "episodes": len(records),
        "answered": stops.get("answered", 0),
        "answer_rate": stops.get("answered", 0) / len(records),
        "unconditional_answer_em": unconditional_em,
        "unconditional_answer_token_f1": unconditional_f1,
        "conditional_answer_em": (
            sum(bool(item["normalized_exact"]) for item in answered_comparisons)
            / len(answered_comparisons)
            if answered_comparisons
            else None
        ),
        "conditional_answer_token_f1": (
            sum(float(item["token_f1"]) for item in answered_comparisons)
            / len(answered_comparisons)
            if answered_comparisons
            else None
        ),
        "stop_reason_counts": dict(sorted(stops.items())),
        "rounds": len(rounds),
        "mean_rounds": len(rounds) / len(records),
        "generations": len(generations),
        "generation_eos_rate": (
            sum(bool(item["finished_on_eos"]) for item in generations)
            / max(len(generations), 1)
        ),
        "generation_structure_valid_rate": (
            sum(bool(item["parsed"]["structure_valid"]) for item in generations)
            / max(len(generations), 1)
        ),
        "generation_cap_count": sum(
            bool(item["hit_generation_cap"]) for item in generations
        ),
        "tool_calls_executed": sum(
            int(item.get("executed_tool_call_count", 1)) for item in tools
        ),
        "tool_validation_failures": sum(
            bool(item["validation_failed"]) for item in tools
        ),
        "tool_token_truncation_failures": sum(
            bool(item["token_truncation_failed"]) for item in tools
        ),
        "tool_timeouts": sum(bool(item["timed_out"]) for item in tools),
        "tool_truncations": sum(bool(item["truncated"]) for item in tools),
        "compressed_generation_rounds": sum(
            int(item["source_segments"]) > 0 for item in compression
        ),
        "source_segments": sum(int(item["source_segments"]) for item in compression),
        "source_tokens": source_tokens,
        "memory_slots": memory_slots,
        "effective_compression_ratio": (
            source_tokens / memory_slots if memory_slots else None
        ),
    }


__all__ = [
    "ASSISTANT_CONTENT_PREFIX",
    "GrepSeekRolloutModelGenerator",
    "HeldoutEpisodeStart",
    "HiddenAnswerReference",
    "ROLLOUT_SCHEMA",
    "RolloutEventSink",
    "RolloutGeneration",
    "build_rollout_feature",
    "collect_hidden_answer_references",
    "evaluate_hidden_answers",
    "heldout_start_from_raw_episode",
    "hidden_answer_reference_from_raw_episode",
    "load_hidden_answer_references",
    "load_heldout_episode_starts",
    "run_rollout_episode",
    "select_heldout_episode_starts",
    "summarize_rollouts",
]
