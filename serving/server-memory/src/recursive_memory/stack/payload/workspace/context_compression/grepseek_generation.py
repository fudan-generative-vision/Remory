"""Free-running GrepSeek prompt construction, parsing, and metrics.

Unlike teacher forcing, generation must never use a prefix whose token boundary
was chosen with knowledge of the gold assistant message.  This module rebuilds
student and oracle prompts from the raw history with
``add_generation_prompt=True`` and derives the reference from the rendered
textual suffix after that prompt.
"""

from __future__ import annotations

import json
import math
import random
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Optional, Sequence

import torch

from context_compression.grepseek_data import (
    _apply_chat_ids,
    _canonical_tool_content,
    _choose_soft_marker,
    _decode,
    _encode,
    _render_chat,
    _source_prompt_and_spans,
    _split_chat_at_markers,
    _validate_messages,
    iter_grepseek_rows,
    parse_tool_response,
)
from context_compression.grepseek_tool import (
    ToolError,
    canonical_browser_command,
    canonical_web_search_command,
)

EOS_TOKEN_ID = 248046
THINK_OPEN_TOKEN_ID = 248068
THINK_CLOSE_TOKEN_ID = 248069
TOOL_OPEN_TOKEN_ID = 248058
TOOL_CLOSE_TOKEN_ID = 248059
ANSWER_OPEN_TOKEN_IDS = (27, 8944, 29)
ANSWER_CLOSE_TOKEN_IDS = (510, 8944, 29)


def load_raw_episode_subset(
    parquet_path: str | Path,
    episode_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Load exactly the requested raw trajectories from the released parquet."""

    requested = {str(episode_id) for episode_id in episode_ids}
    if not requested:
        raise ValueError("at least one raw episode id is required")
    result: dict[str, dict[str, Any]] = {}
    for row in iter_grepseek_rows(parquet_path):
        episode_id = str(row["id"])
        if episode_id not in requested:
            continue
        if episode_id in result:
            raise ValueError(f"duplicate raw episode id: {episode_id!r}")
        result[episode_id] = dict(row)
    missing = sorted(requested - set(result))
    if missing:
        raise KeyError(f"raw parquet is missing requested episodes: {missing[:5]}")
    return result


def _dynamic_reference_ids(
    tokenizer: Any,
    history: Sequence[Mapping[str, str]],
    target_message: Mapping[str, str],
) -> list[int]:
    prompt = _render_chat(tokenizer, history, add_generation_prompt=True)
    full = _render_chat(
        tokenizer, list(history) + [dict(target_message)], add_generation_prompt=False
    )
    if not full.startswith(prompt):
        raise ValueError("native full chat text does not start with generation prompt")
    suffix = full[len(prompt) :]
    result = _encode(tokenizer, suffix)
    if not result:
        raise ValueError("dynamic free-run reference is empty")
    return result


def build_free_run_feature(
    prepared: Mapping[str, Any],
    raw_episode: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """Rebuild one leak-free generation feature from its raw gold history."""

    episode_id, messages = _validate_messages(raw_episode)
    if episode_id != str(prepared["episode_id"]):
        raise ValueError("prepared and raw episode ids differ")
    assistant_index = int(prepared["assistant_message_index"])
    if assistant_index <= 0 or assistant_index >= len(messages):
        raise ValueError("prepared assistant index lies outside raw trajectory")
    target_message = messages[assistant_index]
    if target_message["role"] != "assistant":
        raise ValueError("prepared target index is not an assistant message")
    history = messages[:assistant_index]
    history_tool_indices = [
        index for index, message in enumerate(history) if message["role"] == "tool"
    ]
    sources = [
        [int(token_id) for token_id in source]
        for source in prepared["source_segments_input_ids"]
    ]
    if len(history_tool_indices) != len(sources):
        raise ValueError("raw history and prepared source counts differ")
    source_texts = [_decode(tokenizer, source) for source in sources]
    parsed_tools = {
        index: parse_tool_response(
            history[index]["content"],
            episode_id=episode_id,
            message_index=index,
        )
        for index in history_tool_indices
    }
    marker = _choose_soft_marker(tokenizer)
    student_history: list[dict[str, str]] = []
    oracle_history: list[dict[str, str]] = []
    source_ordinal = 0
    for index, message in enumerate(history):
        if message["role"] != "tool":
            student_history.append(dict(message))
            oracle_history.append(dict(message))
            continue
        payload = parsed_tools[index]
        student_history.append(
            {
                "role": "tool",
                "content": _canonical_tool_content(payload, marker),
            }
        )
        oracle_history.append(
            {
                "role": "tool",
                "content": _canonical_tool_content(
                    payload, source_texts[source_ordinal]
                ),
            }
        )
        source_ordinal += 1

    discrete_segments = _split_chat_at_markers(
        tokenizer,
        student_history,
        marker=marker,
        expected_markers=len(history_tool_indices),
        add_generation_prompt=True,
        target_message=None,
    )
    oracle_prompt_ids = _apply_chat_ids(
        tokenizer, oracle_history, add_generation_prompt=True
    )
    student_reference_ids = _dynamic_reference_ids(
        tokenizer, student_history, target_message
    )
    oracle_reference_ids = _dynamic_reference_ids(
        tokenizer, oracle_history, target_message
    )
    if student_reference_ids != oracle_reference_ids:
        raise ValueError("student and oracle dynamic reference tokenizations differ")

    # Source encoding was already constructed dynamically during preparation,
    # but rebuild it here to prove raw-row alignment and canonical stdout use.
    source_prompt_ids, evidence_spans, _ = _source_prompt_and_spans(
        tokenizer,
        history,
        parsed_tools,
        history_tool_indices,
        source_texts,
    )
    if source_prompt_ids != [int(x) for x in prepared["source_prompt_input_ids"]]:
        raise ValueError(
            "prepared source prompt differs from dynamic canonical history"
        )
    prepared_spans = [
        [int(span[0]), int(span[1])]
        for span in prepared["source_prompt_evidence_spans"]
    ]
    if evidence_spans != prepared_spans:
        raise ValueError(
            "prepared evidence spans differ from dynamic canonical history"
        )

    result = dict(prepared)
    result["discrete_segments_input_ids"] = discrete_segments
    result["target_input_ids"] = student_reference_ids
    result["oracle_prompt_input_ids"] = oracle_prompt_ids
    result["free_run_reference_raw_assistant_content"] = str(target_message["content"])
    result["free_run_dynamic_prefix_changed"] = discrete_segments != [
        [int(token_id) for token_id in segment]
        for segment in prepared["discrete_segments_input_ids"]
    ]
    result["free_run_dynamic_reference_changed"] = student_reference_ids != [
        int(token_id) for token_id in prepared["target_input_ids"]
    ]
    return result


class InterleavedGenerationPrefix(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    memory_positions: torch.Tensor


def build_interleaved_generation_prefix(
    batch: Mapping[str, Any], *, pad_token_id: int
) -> InterleavedGenerationPrefix:
    """Remove gold targets and repad dynamic student prefixes on the left."""

    input_ids = batch["input_ids"]
    if input_ids.ndim != 2:
        raise ValueError("collated input_ids must be rank two")
    left_padding = batch["left_pad_lengths"].long()
    target_starts = batch["target_start_positions"].long()
    if (
        left_padding.shape != target_starts.shape
        or left_padding.numel() != input_ids.shape[0]
    ):
        raise ValueError("prefix boundary metadata has an invalid shape")
    prefixes: list[torch.Tensor] = []
    for sample_index in range(input_ids.shape[0]):
        start = int(left_padding[sample_index])
        stop = int(target_starts[sample_index])
        if not 0 <= start < stop <= input_ids.shape[1]:
            raise ValueError("target boundary lies outside collated recovery sequence")
        prefixes.append(input_ids[sample_index, start:stop])
    width = max(int(prefix.numel()) for prefix in prefixes)
    generated_ids = torch.full(
        (len(prefixes), width),
        int(pad_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    attention_mask = torch.zeros(
        (len(prefixes), width), dtype=torch.bool, device=input_ids.device
    )
    new_left_padding: list[int] = []
    for sample_index, prefix in enumerate(prefixes):
        left = width - int(prefix.numel())
        new_left_padding.append(left)
        generated_ids[sample_index, left:] = prefix
        attention_mask[sample_index, left:] = True

    memory_positions = batch["memory_positions"].long()
    new_positions = torch.empty_like(memory_positions)
    for row_index, (sample_index, old_column) in enumerate(memory_positions.tolist()):
        local_column = int(old_column) - int(left_padding[sample_index])
        if not 0 <= local_column < prefixes[sample_index].numel():
            raise ValueError("memory position lies outside dynamic student prefix")
        new_positions[row_index, 0] = sample_index
        new_positions[row_index, 1] = new_left_padding[sample_index] + local_column
    if not bool(attention_mask[:, -1].all().item()):
        raise AssertionError("dynamic generation prefix must end in a valid token")
    if memory_positions.numel() and not bool(
        attention_mask[new_positions[:, 0], new_positions[:, 1]].all().item()
    ):
        raise AssertionError("recomputed memory position is not attended")
    return InterleavedGenerationPrefix(
        input_ids=generated_ids,
        attention_mask=attention_mask,
        memory_positions=new_positions,
    )


def build_discrete_generation_prefix(
    sequences: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad canonical oracle prompts, keeping every final token valid."""

    if not sequences or any(len(sequence) <= 0 for sequence in sequences):
        raise ValueError("oracle generation prompts must be nonempty")
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), width), dtype=torch.bool, device=device
    )
    for index, sequence in enumerate(sequences):
        values = torch.tensor(list(sequence), dtype=torch.long, device=device)
        input_ids[index, width - values.numel() :] = values
        attention_mask[index, width - values.numel() :] = True
    return input_ids, attention_mask


def trim_at_eos(
    token_ids: Sequence[int], *, eos_token_id: int = EOS_TOKEN_ID
) -> tuple[list[int], bool]:
    result = [int(token_id) for token_id in token_ids]
    if int(eos_token_id) not in result:
        return result, False
    return result[: result.index(int(eos_token_id)) + 1], True


def _find_pattern(
    values: Sequence[int], pattern: Sequence[int], start: int = 0
) -> Optional[int]:
    pattern = tuple(int(value) for value in pattern)
    for index in range(max(0, int(start)), len(values) - len(pattern) + 1):
        if tuple(values[index : index + len(pattern)]) == pattern:
            return index
    return None


def _decode_clean(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return _decode(tokenizer, token_ids)


def _strip_whitespace_tokens(tokenizer: Any, values: Sequence[int]) -> list[int]:
    result = [int(value) for value in values]

    def whitespace(token_id: int) -> bool:
        text = _decode_clean(tokenizer, [token_id])
        return bool(text) and text.isspace()

    while result and whitespace(result[0]):
        result.pop(0)
    while result and whitespace(result[-1]):
        result.pop()
    return result


@dataclass(frozen=True)
class ParsedGrepSeekToolCall:
    """One normalized tool call inside an assistant continuation.

    The released GrepSeek checkpoints emit exactly one JSON call and continue
    to use the scalar fields on :class:`ParsedGrepSeekContinuation`.  Qwen3.6's
    native XML protocol may emit several complete ``tool_call`` blocks in one
    assistant message, so only that multi-call path populates this structure.
    """

    call_index: int
    tool_payload: Optional[dict[str, Any]]
    tool_json_valid: bool
    tool_schema_valid: bool
    tool_command: Optional[str]
    errors: tuple[str, ...]
    tool_schema_error: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "call_index": self.call_index,
            "tool_payload": self.tool_payload,
            "tool_json_valid": self.tool_json_valid,
            "tool_schema_valid": self.tool_schema_valid,
            "tool_command": self.tool_command,
            "errors": list(self.errors),
            "tool_schema_error": self.tool_schema_error,
        }


@dataclass(frozen=True)
class ParsedGrepSeekContinuation:
    kind: str
    eos_reached: bool
    structure_valid: bool
    complete_valid: bool
    think_text: str
    think_nonempty: bool
    tool_payload: Optional[dict[str, Any]]
    tool_json_valid: bool
    tool_schema_valid: bool
    tool_command: Optional[str]
    answer_text: Optional[str]
    errors: tuple[str, ...]
    # A safe, model-facing diagnostic is retained only when the model emitted
    # valid JSON for a known tool but its arguments violated that tool's
    # public schema.  Malformed syntax and unknown tools use public error codes
    # instead of retaining model-controlled diagnostic text.
    tool_schema_error: Optional[str] = None
    # Kept empty for every legacy/single-call continuation so existing JSON
    # records remain byte-compatible.  Native Qwen multi-call continuations
    # populate this tuple in source order.
    tool_calls: tuple[ParsedGrepSeekToolCall, ...] = ()
    # Qwen3.6 permits natural-language assistant content after ``</think>`` and
    # before the first native tool call.  Keep it distinct from hidden
    # reasoning; omit it from old/single-call JSON when absent.
    tool_preamble_text: str = ""

    def to_json(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "eos_reached": self.eos_reached,
            "structure_valid": self.structure_valid,
            "complete_valid": self.complete_valid,
            "think_text": self.think_text,
            "think_nonempty": self.think_nonempty,
            "tool_payload": self.tool_payload,
            "tool_json_valid": self.tool_json_valid,
            "tool_schema_valid": self.tool_schema_valid,
            "tool_command": self.tool_command,
            "answer_text": self.answer_text,
            "errors": list(self.errors),
            "tool_schema_error": self.tool_schema_error,
        }
        if self.tool_calls:
            result["tool_calls"] = [call.to_json() for call in self.tool_calls]
        if self.tool_preamble_text:
            result["tool_preamble_text"] = self.tool_preamble_text
        return result


def parse_grepseek_continuation(
    token_ids: Sequence[int],
    tokenizer: Any,
    *,
    eos_token_id: int = EOS_TOKEN_ID,
) -> ParsedGrepSeekContinuation:
    """Parse generated reasoning followed by one tool call or one answer."""

    trimmed, eos_reached = trim_at_eos(token_ids, eos_token_id=eos_token_id)
    body = trimmed[:-1] if eos_reached else trimmed
    errors: list[str] = []
    think_close = _find_pattern(body, (THINK_CLOSE_TOKEN_ID,))
    if think_close is None:
        errors.append("missing_think_close")
        think_ids: list[int] = []
        tail = body
    else:
        think_ids = body[:think_close]
        tail = body[think_close + 1 :]
    if THINK_OPEN_TOKEN_ID in think_ids:
        errors.append("unexpected_generated_think_open")
    think_text = _decode_clean(tokenizer, think_ids).strip()
    tail = _strip_whitespace_tokens(tokenizer, tail)

    kind = "invalid"
    tool_payload: Optional[dict[str, Any]] = None
    tool_json_valid = False
    tool_schema_valid = False
    tool_schema_error: Optional[str] = None
    tool_command: Optional[str] = None
    answer_text: Optional[str] = None
    if tail and tail[0] == TOOL_OPEN_TOKEN_ID:
        kind = "tool_call"
        closing = _find_pattern(tail, (TOOL_CLOSE_TOKEN_ID,), 1)
        if closing is None:
            errors.append("missing_tool_close")
        else:
            payload_ids = _strip_whitespace_tokens(tokenizer, tail[1:closing])
            trailing = _strip_whitespace_tokens(tokenizer, tail[closing + 1 :])
            if trailing:
                errors.append("tokens_after_tool_close")
            payload_text = _decode_clean(tokenizer, payload_ids)
            try:
                parsed = json.loads(payload_text)
            except (json.JSONDecodeError, TypeError):
                errors.append("invalid_tool_json")
            else:
                tool_json_valid = True
                if not isinstance(parsed, dict):
                    errors.append("tool_payload_not_object")
                else:
                    tool_payload = parsed
                    tool_name = parsed.get("name")
                    if tool_name not in {"shell", "web_search", "browser"}:
                        errors.append("tool_name_not_shell")
                    arguments = parsed.get("arguments")
                    if not isinstance(arguments, dict):
                        errors.append("tool_arguments_not_object")
                    elif tool_name == "shell":
                        command = arguments.get("command")
                        if isinstance(command, str) and command.strip():
                            tool_command = command
                        else:
                            errors.append("missing_tool_command")
                            tool_schema_error = (
                                "shell command must be a nonempty string"
                            )
                        tool_schema_valid = (
                            isinstance(command, str) and bool(command.strip())
                        )
                    elif tool_name == "web_search":
                        if set(parsed) != {"name", "arguments"}:
                            errors.append("invalid_web_search_arguments")
                            tool_schema_error = (
                                "web_search tool call must contain exactly "
                                "'name' and 'arguments'"
                            )
                        else:
                            try:
                                tool_command = canonical_web_search_command(arguments)
                            except ToolError as exc:
                                errors.append("invalid_web_search_arguments")
                                tool_schema_error = str(exc)
                            else:
                                tool_schema_valid = True
                    elif tool_name == "browser":
                        if set(parsed) != {"name", "arguments"}:
                            errors.append("invalid_browser_arguments")
                            tool_schema_error = (
                                "browser tool call must contain exactly "
                                "'name' and 'arguments'"
                            )
                        else:
                            try:
                                tool_command = canonical_browser_command(arguments)
                            except ToolError as exc:
                                errors.append("invalid_browser_arguments")
                                tool_schema_error = str(exc)
                            else:
                                tool_schema_valid = True
    elif tail and tuple(tail[: len(ANSWER_OPEN_TOKEN_IDS)]) == ANSWER_OPEN_TOKEN_IDS:
        kind = "answer"
        content_start = len(ANSWER_OPEN_TOKEN_IDS)
        closing = _find_pattern(tail, ANSWER_CLOSE_TOKEN_IDS, content_start)
        if closing is None:
            errors.append("missing_answer_close")
        else:
            answer_ids = _strip_whitespace_tokens(
                tokenizer, tail[content_start:closing]
            )
            answer_text = _decode_clean(tokenizer, answer_ids).strip()
            trailing = _strip_whitespace_tokens(
                tokenizer, tail[closing + len(ANSWER_CLOSE_TOKEN_IDS) :]
            )
            if trailing:
                errors.append("tokens_after_answer_close")
            if not answer_text:
                errors.append("empty_answer")
    else:
        errors.append("missing_tool_or_answer")
    if not eos_reached:
        errors.append("missing_eos")
    structure_errors = [error for error in errors if error != "missing_eos"]
    structure_valid = not structure_errors and kind in {"tool_call", "answer"}
    return ParsedGrepSeekContinuation(
        kind=kind,
        eos_reached=eos_reached,
        structure_valid=structure_valid,
        complete_valid=structure_valid and eos_reached,
        think_text=think_text,
        think_nonempty=bool(think_text),
        tool_payload=tool_payload,
        tool_json_valid=tool_json_valid,
        tool_schema_valid=tool_schema_valid,
        tool_command=tool_command,
        answer_text=answer_text,
        errors=tuple(errors),
        tool_schema_error=tool_schema_error,
    )


_QWEN_NATIVE_FUNCTION = re.compile(
    r"\A\s*<function=([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</function>\s*\Z",
    re.DOTALL,
)
_QWEN_NATIVE_PARAMETER = re.compile(
    r"\s*<parameter=([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _qwen_native_tool_payload(payload_text: str) -> Optional[dict[str, Any]]:
    """Normalize Qwen3.5+ native function XML into the public tool ABI."""

    # Some Qwen3.6 generations insert the role-opening special token directly
    # before ``function=`` inside an otherwise canonical tool_call.  Accept
    # only this exact, leading variant; all other extra tokens still fail.
    payload_text = re.sub(
        r"\A\s*<\|im_start\|>function=", "<function=", payload_text, count=1
    )
    function = _QWEN_NATIVE_FUNCTION.fullmatch(payload_text)
    if function is None:
        return None
    name, parameter_text = function.groups()
    arguments: dict[str, Any] = {}
    cursor = 0
    for match in _QWEN_NATIVE_PARAMETER.finditer(parameter_text):
        if parameter_text[cursor : match.start()].strip():
            return None
        key, raw_value = match.groups()
        if key in arguments:
            return None
        value: Any = raw_value.strip()
        if key in {"top_k", "max_length"}:
            if re.fullmatch(r"[+-]?\d+", value) is None:
                return None
            value = int(value)
        elif key == "timeout":
            try:
                value = float(value)
            except ValueError:
                return None
        arguments[key] = value
        cursor = match.end()
    if parameter_text[cursor:].strip() or not arguments:
        return None
    return {"name": name, "arguments": arguments}


def _qwen_native_tool_blocks(
    tail: Sequence[int], tokenizer: Any
) -> Optional[list[str]]:
    """Return an exact sequence of complete native tool-call payloads.

    Arbitrary text between or after calls is rejected.  This is deliberately
    token-boundary based because ``<tool_call>`` and ``</tool_call>`` are Qwen
    special tokens, while the function/parameter XML inside them is ordinary
    text.
    """

    remaining = _strip_whitespace_tokens(tokenizer, tail)
    payloads: list[str] = []
    while remaining:
        if remaining[0] != TOOL_OPEN_TOKEN_ID:
            return None
        closing = _find_pattern(remaining, (TOOL_CLOSE_TOKEN_ID,), 1)
        if closing is None:
            return None
        payload_ids = _strip_whitespace_tokens(tokenizer, remaining[1:closing])
        payloads.append(_decode_clean(tokenizer, payload_ids))
        remaining = _strip_whitespace_tokens(tokenizer, remaining[closing + 1 :])
    return payloads or None


def _normalized_qwen_tool_call(
    payload_text: str,
    tokenizer: Any,
    *,
    call_index: int,
    eos_token_id: int,
) -> ParsedGrepSeekToolCall:
    """Parse one native XML call through the released public tool schema."""

    payload = _qwen_native_tool_payload(payload_text)
    if payload is None:
        return ParsedGrepSeekToolCall(
            call_index=call_index,
            tool_payload=None,
            tool_json_valid=False,
            tool_schema_valid=False,
            tool_command=None,
            errors=("invalid_native_tool_xml",),
            tool_schema_error=(
                "native tool XML must contain one function with unique, "
                "well-formed parameter elements"
            ),
        )
    canonical_ids = [
        THINK_CLOSE_TOKEN_ID,
        TOOL_OPEN_TOKEN_ID,
        *_encode(
            tokenizer,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ),
        TOOL_CLOSE_TOKEN_ID,
        int(eos_token_id),
    ]
    parsed = parse_grepseek_continuation(
        canonical_ids, tokenizer, eos_token_id=eos_token_id
    )
    return ParsedGrepSeekToolCall(
        call_index=call_index,
        tool_payload=parsed.tool_payload,
        tool_json_valid=parsed.tool_json_valid,
        tool_schema_valid=parsed.tool_schema_valid,
        tool_command=parsed.tool_command,
        errors=parsed.errors,
        tool_schema_error=parsed.tool_schema_error,
    )


def parse_qwen36_continuation(
    token_ids: Sequence[int],
    tokenizer: Any,
    *,
    eos_token_id: int = EOS_TOKEN_ID,
) -> ParsedGrepSeekContinuation:
    """Parse Qwen3.5+ native tools while retaining the GrepSeek result ABI.

    Qwen's current chat template emits ``<function=...><parameter=...>`` inside
    the same special ``<tool_call>`` tokens used by the old JSON protocol, and
    permits natural-language content before one or more complete call blocks in
    one assistant turn.  It also specifies an ordinary free-text final answer
    when no function is called.  Normalize native calls and answers here; the
    legacy parser remains unchanged for every existing GrepSeek checkpoint.
    """

    # Qwen's official reasoning parser accepts and strips one optional
    # generated ``<think>`` opening.  The chat template already ends in that
    # opening for thinking mode, but sampled generations can still repeat it.
    # Normalize only this exact leading special token for parsing; callers
    # retain the untouched output token IDs for auditability.
    trimmed_original, original_eos_reached = trim_at_eos(
        token_ids, eos_token_id=eos_token_id
    )
    original_body = (
        trimmed_original[:-1] if original_eos_reached else trimmed_original
    )
    generated_think_open = bool(
        original_body and original_body[0] == THINK_OPEN_TOKEN_ID
    )
    normalized_token_ids = list(original_body[1:] if generated_think_open else original_body)
    if original_eos_reached:
        normalized_token_ids.append(int(eos_token_id))

    legacy = parse_grepseek_continuation(
        normalized_token_ids, tokenizer, eos_token_id=eos_token_id
    )
    if legacy.kind in {"tool_call", "answer"} and legacy.structure_valid:
        return legacy

    trimmed, eos_reached = trim_at_eos(
        normalized_token_ids, eos_token_id=eos_token_id
    )
    body = trimmed[:-1] if eos_reached else trimmed
    think_close = _find_pattern(body, (THINK_CLOSE_TOKEN_ID,))
    implicit_tool_boundary: Optional[int] = None
    if think_close is None:
        implicit_tool_boundary = _find_pattern(body, (TOOL_OPEN_TOKEN_ID,))
        if generated_think_open and implicit_tool_boundary is None:
            # An explicit generated opening without its close is unfinished
            # reasoning, not a direct no-thinking answer.
            return legacy
        if implicit_tool_boundary is not None:
            # Qwen occasionally emits ordinary reasoning text followed by the
            # native tool special token without either think delimiter.  The
            # tool token is an unambiguous boundary; retain the preceding text
            # as reasoning and continue to validate the native tool payload
            # strictly below.
            tail = _strip_whitespace_tokens(
                tokenizer, body[implicit_tool_boundary:]
            )
            think_text = _decode_clean(
                tokenizer, body[:implicit_tool_boundary]
            ).strip()
        else:
            # ``enable_thinking=False`` puts the empty <think></think> block in
            # the prompt itself, so the generated continuation starts directly
            # with a native tool call or final answer.
            tail = _strip_whitespace_tokens(tokenizer, body)
            think_text = ""
    else:
        tail = _strip_whitespace_tokens(tokenizer, body[think_close + 1 :])
        think_text = _decode_clean(tokenizer, body[:think_close]).strip()
    if not tail:
        return legacy

    # The official template permits optional natural-language assistant
    # content before the first native tool call, including immediately after a
    # completed </think> block.  Keep that content semantically separate from
    # reasoning and require the remainder to be an exact sequence of calls.
    tool_preamble_text = ""
    tool_boundary = _find_pattern(tail, (TOOL_OPEN_TOKEN_ID,))
    native_tool_tail = tail
    if tool_boundary is not None:
        preamble_ids = _strip_whitespace_tokens(tokenizer, tail[:tool_boundary])
        if TOOL_CLOSE_TOKEN_ID in preamble_ids:
            return legacy
        tool_preamble_text = _decode_clean(tokenizer, preamble_ids).strip()
        native_tool_tail = _strip_whitespace_tokens(tokenizer, tail[tool_boundary:])

    if native_tool_tail[0] == TOOL_OPEN_TOKEN_ID:
        native_blocks = _qwen_native_tool_blocks(native_tool_tail, tokenizer)
        if native_blocks is None:
            return legacy
        if len(native_blocks) > 1:
            tool_calls = tuple(
                _normalized_qwen_tool_call(
                    payload_text,
                    tokenizer,
                    call_index=call_index,
                    eos_token_id=eos_token_id,
                )
                for call_index, payload_text in enumerate(native_blocks, start=1)
            )
            call_errors = tuple(
                f"tool_call[{call.call_index}].{error}"
                for call in tool_calls
                for error in call.errors
            )
            errors = call_errors + (() if eos_reached else ("missing_eos",))
            first = tool_calls[0]
            first_schema_error = next(
                (
                    call.tool_schema_error
                    for call in tool_calls
                    if call.tool_schema_error
                ),
                None,
            )
            structure_valid = not call_errors
            return ParsedGrepSeekContinuation(
                kind="tool_call",
                eos_reached=eos_reached,
                structure_valid=structure_valid,
                complete_valid=structure_valid and eos_reached,
                think_text=think_text,
                think_nonempty=bool(think_text),
                # Scalar fields retain first-call compatibility.  Aggregate
                # validity is fail-closed across the complete native batch.
                tool_payload=first.tool_payload,
                tool_json_valid=all(
                    call.tool_json_valid for call in tool_calls
                ),
                tool_schema_valid=all(
                    call.tool_schema_valid for call in tool_calls
                ),
                tool_command=first.tool_command,
                answer_text=None,
                errors=errors,
                tool_schema_error=first_schema_error,
                tool_calls=tool_calls,
                tool_preamble_text=tool_preamble_text,
            )

        payload = _qwen_native_tool_payload(native_blocks[0])
        if payload is None:
            return legacy
        canonical_ids = [
            *(
                body[: think_close + 1]
                if think_close is not None
                else [
                    *(
                        body[:implicit_tool_boundary]
                        if implicit_tool_boundary is not None
                        else []
                    ),
                    THINK_CLOSE_TOKEN_ID,
                ]
            ),
            TOOL_OPEN_TOKEN_ID,
            *_encode(
                tokenizer,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
            TOOL_CLOSE_TOKEN_ID,
        ]
        if eos_reached:
            canonical_ids.append(int(eos_token_id))
        parsed = parse_grepseek_continuation(
            canonical_ids, tokenizer, eos_token_id=eos_token_id
        )
        return replace(parsed, tool_preamble_text=tool_preamble_text)

    # Native Qwen tool prompting asks for a normal answer when no function is
    # needed.  Do not reinterpret malformed tool XML as an answer.
    if TOOL_OPEN_TOKEN_ID in tail or TOOL_CLOSE_TOKEN_ID in tail:
        return legacy
    answer_text = _decode_clean(tokenizer, tail).strip()
    if not answer_text:
        return legacy
    errors = () if eos_reached else ("missing_eos",)
    return ParsedGrepSeekContinuation(
        kind="answer",
        eos_reached=eos_reached,
        structure_valid=True,
        complete_valid=eos_reached,
        think_text=think_text,
        think_nonempty=bool(think_text),
        tool_payload=None,
        tool_json_valid=False,
        tool_schema_valid=False,
        tool_command=None,
        answer_text=answer_text,
        errors=errors,
    )


def longest_common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    result = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        result += 1
    return result


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    lowered = str(text).lower()
    no_punctuation = "".join(
        character
        for character in lowered
        if character not in string.punctuation
        and not unicodedata.category(character).startswith("P")
    )
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def normalized_answer_token_f1(
    prediction: Optional[str], reference: Optional[str]
) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


@dataclass
class FreeRunMetricAccumulator:
    samples: int = 0
    eos_reached: int = 0
    structure_valid: int = 0
    complete_valid: int = 0
    nonempty_think: int = 0
    truncated: int = 0
    continuation_exact: int = 0
    kind_correct: int = 0
    lcp_tokens: int = 0
    reference_tokens: int = 0
    tool_references: int = 0
    tool_json_valid: int = 0
    tool_schema_valid: int = 0
    tool_command_exact: int = 0
    answer_references: int = 0
    answer_covered: int = 0
    answer_normalized_exact: int = 0
    answer_token_f1_sum: float = 0.0

    def update(
        self,
        generated_ids: Sequence[int],
        reference_ids: Sequence[int],
        generated: ParsedGrepSeekContinuation,
        reference: ParsedGrepSeekContinuation,
    ) -> dict[str, Any]:
        generated_trimmed, _ = trim_at_eos(generated_ids)
        reference_trimmed, _ = trim_at_eos(reference_ids)
        lcp = longest_common_prefix_length(generated_trimmed, reference_trimmed)
        continuation_exact = generated_trimmed == reference_trimmed
        self.samples += 1
        self.eos_reached += int(generated.eos_reached)
        self.structure_valid += int(generated.structure_valid)
        self.complete_valid += int(generated.complete_valid)
        self.nonempty_think += int(generated.think_nonempty)
        self.truncated += int(not generated.eos_reached)
        self.continuation_exact += int(continuation_exact)
        self.kind_correct += int(generated.kind == reference.kind)
        self.lcp_tokens += lcp
        self.reference_tokens += len(reference_trimmed)
        if reference.kind == "tool_call":
            self.tool_references += 1
            self.tool_json_valid += int(generated.tool_json_valid)
            self.tool_schema_valid += int(generated.tool_schema_valid)
            self.tool_command_exact += int(
                generated.tool_command is not None
                and generated.tool_command == reference.tool_command
            )
        if reference.kind == "answer":
            self.answer_references += 1
            covered = generated.kind == "answer" and generated.answer_text is not None
            self.answer_covered += int(covered)
            answer_exact_for_accumulator = covered and normalize_answer(
                generated.answer_text
            ) == normalize_answer(reference.answer_text)
            self.answer_normalized_exact += int(answer_exact_for_accumulator)
            self.answer_token_f1_sum += (
                normalized_answer_token_f1(
                    generated.answer_text if covered else None,
                    reference.answer_text,
                )
                if covered
                else 0.0
            )
        answer_covered = (
            reference.kind == "answer"
            and generated.kind == "answer"
            and generated.answer_text is not None
        )
        answer_exact = answer_covered and normalize_answer(
            generated.answer_text
        ) == normalize_answer(reference.answer_text)
        answer_f1 = (
            normalized_answer_token_f1(generated.answer_text, reference.answer_text)
            if answer_covered
            else 0.0
        )
        return {
            "continuation_exact": continuation_exact,
            "kind_correct": generated.kind == reference.kind,
            "lcp_tokens": lcp,
            "lcp_fraction": lcp / max(len(reference_trimmed), 1),
            "reference_token_count": len(reference_trimmed),
            "answer_reference": reference.kind == "answer",
            "answer_covered": answer_covered,
            "answer_normalized_exact": answer_exact,
            "answer_token_f1": answer_f1,
            "tool_reference": reference.kind == "tool_call",
            "tool_json_valid": generated.tool_json_valid,
            "tool_schema_valid": generated.tool_schema_valid,
            "tool_command_exact": (
                generated.tool_command is not None
                and generated.tool_command == reference.tool_command
            ),
        }

    def result(self) -> dict[str, Any]:
        if self.samples <= 0:
            raise RuntimeError("free-run metric has no samples")
        return {
            "samples": self.samples,
            "eos_rate": self.eos_reached / self.samples,
            "structure_valid_rate": self.structure_valid / self.samples,
            "complete_valid_rate": self.complete_valid / self.samples,
            "nonempty_think_rate": self.nonempty_think / self.samples,
            "truncated_rate": self.truncated / self.samples,
            "continuation_exact_rate": self.continuation_exact / self.samples,
            "kind_accuracy": self.kind_correct / self.samples,
            "token_lcp_fraction": self.lcp_tokens / max(self.reference_tokens, 1),
            "mean_lcp_tokens": self.lcp_tokens / self.samples,
            "reference_token_count": self.reference_tokens,
            "tool": {
                "reference_samples": self.tool_references,
                "json_valid_rate": (
                    self.tool_json_valid / self.tool_references
                    if self.tool_references
                    else None
                ),
                "schema_valid_rate": (
                    self.tool_schema_valid / self.tool_references
                    if self.tool_references
                    else None
                ),
                "command_exact_rate": (
                    self.tool_command_exact / self.tool_references
                    if self.tool_references
                    else None
                ),
            },
            "answer": {
                "reference_samples": self.answer_references,
                "covered_samples": self.answer_covered,
                "coverage": (
                    self.answer_covered / self.answer_references
                    if self.answer_references
                    else None
                ),
                "unconditional_normalized_exact_rate": (
                    self.answer_normalized_exact / self.answer_references
                    if self.answer_references
                    else None
                ),
                "conditional_normalized_exact_rate": (
                    self.answer_normalized_exact / self.answer_covered
                    if self.answer_covered
                    else None
                ),
                "unconditional_token_f1": (
                    self.answer_token_f1_sum / self.answer_references
                    if self.answer_references
                    else None
                ),
                "conditional_token_f1": (
                    self.answer_token_f1_sum / self.answer_covered
                    if self.answer_covered
                    else None
                ),
            },
        }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid bootstrap quantile input")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_episode_bootstrap_free_run(
    records: Sequence[Mapping[str, Any]],
    *,
    arm_a_condition: str = "trained_on_shuffled_coverage",
    arm_b_condition: str = "shuffled",
    direction: str = "arm_b_minus_arm_a",
    replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired episode-macro bootstrap for any two generation conditions."""

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    if arm_a_condition == arm_b_condition:
        raise ValueError("paired free-run arms must be different")
    if direction not in {"arm_b_minus_arm_a", "arm_a_minus_arm_b"}:
        raise ValueError("direction must be 'arm_b_minus_arm_a' or 'arm_a_minus_arm_b'")
    by_condition: dict[str, dict[str, Mapping[str, Any]]] = {
        arm_a_condition: {},
        arm_b_condition: {},
    }
    for record in records:
        condition = str(record.get("condition"))
        if condition not in by_condition:
            continue
        sample_id = str(record["sample_id"])
        if sample_id in by_condition[condition]:
            raise ValueError(
                f"duplicate paired generation record: {condition}/{sample_id}"
            )
        by_condition[condition][sample_id] = record
    arm_a = by_condition[arm_a_condition]
    arm_b = by_condition[arm_b_condition]
    if set(arm_a) != set(arm_b) or not arm_a:
        raise ValueError("paired free-run arms have different or empty sample sets")

    def outcome(record: Mapping[str, Any], name: str) -> tuple[float, int]:
        generated = record["generated"]
        comparison = record["comparison"]
        if name == "eos_rate":
            return float(bool(generated["eos_reached"])), 1
        if name == "structure_valid_rate":
            return float(bool(generated["structure_valid"])), 1
        if name == "complete_valid_rate":
            return float(bool(generated["complete_valid"])), 1
        if name == "continuation_exact_rate":
            return float(bool(comparison["continuation_exact"])), 1
        if name == "kind_accuracy":
            return float(bool(comparison["kind_correct"])), 1
        if name == "token_lcp_fraction":
            return float(comparison["lcp_tokens"]), int(
                comparison["reference_token_count"]
            )
        if name == "answer_unconditional_em":
            return (
                float(bool(comparison["answer_normalized_exact"])),
                int(bool(comparison["answer_reference"])),
            )
        if name == "answer_unconditional_token_f1":
            return (
                float(comparison["answer_token_f1"]),
                int(bool(comparison["answer_reference"])),
            )
        if name == "answer_coverage":
            applicable = bool(comparison["answer_reference"])
            return (
                float(applicable and bool(comparison["answer_covered"])),
                int(applicable),
            )
        if name == "tool_json_valid_rate":
            applicable = bool(comparison["tool_reference"])
            return (
                float(applicable and bool(comparison["tool_json_valid"])),
                int(applicable),
            )
        if name == "tool_schema_valid_rate":
            applicable = bool(comparison["tool_reference"])
            return (
                float(applicable and bool(comparison["tool_schema_valid"])),
                int(applicable),
            )
        if name == "tool_command_exact_rate":
            applicable = bool(comparison["tool_reference"])
            return (
                float(applicable and bool(comparison["tool_command_exact"])),
                int(applicable),
            )
        raise KeyError(name)

    metric_names = (
        "eos_rate",
        "structure_valid_rate",
        "complete_valid_rate",
        "continuation_exact_rate",
        "kind_accuracy",
        "token_lcp_fraction",
        "answer_unconditional_em",
        "answer_unconditional_token_f1",
        "answer_coverage",
        "tool_json_valid_rate",
        "tool_schema_valid_rate",
        "tool_command_exact_rate",
    )
    episode_rows: dict[str, dict[str, list[float]]] = {}
    for sample_id in sorted(arm_a):
        arm_a_record = arm_a[sample_id]
        arm_b_record = arm_b[sample_id]
        if str(arm_a_record["episode_id"]) != str(arm_b_record["episode_id"]):
            raise ValueError(f"paired sample {sample_id!r} changed episode")
        episode_id = str(arm_a_record["episode_id"])
        row = episode_rows.setdefault(
            episode_id,
            {name: [0.0, 0.0, 0.0, 0.0] for name in metric_names},
        )
        for name in metric_names:
            arm_a_num, arm_a_den = outcome(arm_a_record, name)
            arm_b_num, arm_b_den = outcome(arm_b_record, name)
            if arm_a_den != arm_b_den:
                raise ValueError(f"paired denominator changed for {sample_id}/{name}")
            row[name][0] += arm_a_num
            row[name][1] += arm_a_den
            row[name][2] += arm_b_num
            row[name][3] += arm_b_den
    episodes = [episode_rows[key] for key in sorted(episode_rows)]
    generator = random.Random(int(seed))
    bootstrap_indices = [
        [generator.randrange(len(episodes)) for _ in range(len(episodes))]
        for _ in range(int(replicates))
    ]
    metric_results: dict[str, Any] = {}
    for name in metric_names:
        episode_ratios: list[Optional[tuple[float, float]]] = []
        total_observation_denominator = 0.0
        for episode in episodes:
            arm_a_num, arm_a_den, arm_b_num, arm_b_den = episode[name]
            if arm_a_den != arm_b_den:
                raise RuntimeError(f"episode denominator changed for {name}")
            if arm_a_den <= 0:
                episode_ratios.append(None)
                continue
            episode_ratios.append((arm_a_num / arm_a_den, arm_b_num / arm_b_den))
            total_observation_denominator += arm_a_den
        eligible_ratios = [ratio for ratio in episode_ratios if ratio is not None]
        if not eligible_ratios:
            metric_results[name] = None
            continue
        arm_a_estimate = sum(ratio[0] for ratio in eligible_ratios) / len(
            eligible_ratios
        )
        arm_b_estimate = sum(ratio[1] for ratio in eligible_ratios) / len(
            eligible_ratios
        )
        deltas: list[float] = []
        for indices in bootstrap_indices:
            sampled_ratios = [
                episode_ratios[index]
                for index in indices
                if episode_ratios[index] is not None
            ]
            if not sampled_ratios:
                continue
            sign = 1.0 if direction == "arm_b_minus_arm_a" else -1.0
            deltas.append(
                sign
                * sum(ratio[1] - ratio[0] for ratio in sampled_ratios)
                / len(sampled_ratios)
            )
        if not deltas:
            raise RuntimeError(f"bootstrap produced no valid draws for {name}")
        signed_delta = (
            arm_b_estimate - arm_a_estimate
            if direction == "arm_b_minus_arm_a"
            else arm_a_estimate - arm_b_estimate
        )
        metric_results[name] = {
            "estimates": {
                arm_a_condition: arm_a_estimate,
                arm_b_condition: arm_b_estimate,
            },
            "delta": signed_delta,
            "delta_ci95": [_quantile(deltas, 0.025), _quantile(deltas, 0.975)],
            "effective_bootstrap_replicates": len(deltas),
            "estimator": "episode_macro",
            "eligible_episode_count": len(eligible_ratios),
            "observation_denominator": int(total_observation_denominator),
        }
    return {
        "arm_a_condition": arm_a_condition,
        "arm_b_condition": arm_b_condition,
        "direction": (
            f"{arm_b_condition}_minus_{arm_a_condition}"
            if direction == "arm_b_minus_arm_a"
            else f"{arm_a_condition}_minus_{arm_b_condition}"
        ),
        "bootstrap_method": "paired_percentile_episode_cluster",
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "episode_count": len(episodes),
        "sample_count": len(arm_a),
        "metrics": metric_results,
    }


__all__ = [
    "ANSWER_CLOSE_TOKEN_IDS",
    "ANSWER_OPEN_TOKEN_IDS",
    "EOS_TOKEN_ID",
    "FreeRunMetricAccumulator",
    "InterleavedGenerationPrefix",
    "ParsedGrepSeekContinuation",
    "ParsedGrepSeekToolCall",
    "build_discrete_generation_prefix",
    "build_free_run_feature",
    "build_interleaved_generation_prefix",
    "load_raw_episode_subset",
    "longest_common_prefix_length",
    "normalize_answer",
    "normalized_answer_token_f1",
    "parse_grepseek_continuation",
    "paired_episode_bootstrap_free_run",
    "trim_at_eos",
]
