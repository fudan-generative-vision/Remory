"""Qwen3.6 A3B tool-calling rows for recursive compressor training.

The Hugging Face parquet stores a sampled completion as its final assistant
message.  Earlier assistant messages are prompt context, not independent SFT
targets.  This module therefore keeps one target per row and builds a
one-tool-batch-lag eviction plan:

* native tools plus the complete initial system/user turn form persistent P;
* with at least two completed tool-result batches, history through the
  penultimate batch is evictable;
* the evictable prefix is partitioned into consecutive chunks of at most 1024
  tokens; and
* every chunk is distilled against the same online-teacher positions from the
  final A3B completion.  Recurrence state is expected to detach between these
  events in the trainer.

The chosen completion is tokenized as it would be generated: native prompt
IDs through G5 are concatenated with an independently encoded textual
continuation.  It is not sliced from a jointly tokenized completed chat,
because a newline BPE merge at the prompt/continuation seam can otherwise
change the generation prefix IDs.

Rows with zero or one completed tool batch intentionally produce no events.
They never fall back to the unrelated fixed token-clock contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EVICTION_BLOCK_TOKENS = 1024
COMPRESSION_RATIO = 16
TARGET_TOKEN_CAP = 256
QWEN36_ENABLE_THINKING = True
QWEN36_PRESERVE_THINKING = True
QWEN36_TOOL_RESULT_BATCH_CLOSING_IDS = (198, 248067, 248046, 198)
QWEN36_GENERATION_PROMPT_IDS = (248045, 74455, 198, 248068, 198)
# Keep the familiar name used by the non-agentic Qwen3.6 data path.
QWEN36_ASSISTANT_OPENING_IDS = QWEN36_GENERATION_PROMPT_IDS


@dataclass(frozen=True)
class Qwen36AgenticRow:
    """Strict normalized form of one parquet row."""

    example_id: str
    prompt_group_id: str
    tools: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Qwen36ToolBatch:
    """One maximal consecutive run of native tool-result messages.

    Token offsets are history-relative and half-open.  ``message_stop`` is
    likewise exclusive.  ``next_assistant_start`` points at the complete
    native assistant message opening.  (Historical assistants with empty
    reasoning can merge the opening newline differently from generation G5.)
    """

    batch_index: int
    message_start: int
    message_stop: int
    history_start: int
    history_stop: int
    next_assistant_start: int
    next_assistant_stop: int
    tool_call_ids: tuple[str, ...]


@dataclass(frozen=True)
class AgenticCompressionEvent:
    """One chronological recursive state update.

    All offsets are history-relative and half-open.  The compressor consumes
    the previous detached memory plus ``[block_start:block_stop]``.  The raw
    recovery/teacher suffix is ``[working_start:working_stop]`` and already
    ends with the final generation prompt.  KL positions are
    ``[target_start:target_stop]`` from the frozen online teacher.
    """

    event_index: int
    batch_index: int
    chunk_index: int
    block_start: int
    block_stop: int
    working_start: int
    working_stop: int
    target_start: int
    target_stop: int
    is_serving_state: bool


@dataclass(frozen=True)
class Qwen36AgenticExample:
    """Rendered A3B row and its complete recursive training plan."""

    example_id: str
    prompt_group_id: str
    raw_prefix_input_ids: tuple[int, ...]
    history_input_ids: tuple[int, ...]
    tool_batches: tuple[Qwen36ToolBatch, ...]
    events: tuple[AgenticCompressionEvent, ...]
    final_target_start: int
    final_target_stop: int


def _plain_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a plain int")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _plain_positive_int(name: str, value: Any) -> int:
    value = _plain_non_negative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def slot_count_for_physical_tokens(
    physical_tokens: int,
    *,
    compression_ratio: int = COMPRESSION_RATIO,
    max_slots: int | None = None,
) -> int:
    """Return the variable r:N slot count, optionally capped."""

    physical_tokens = _plain_non_negative_int("physical_tokens", physical_tokens)
    compression_ratio = _plain_positive_int("compression_ratio", compression_ratio)
    if max_slots is not None:
        max_slots = _plain_positive_int("max_slots", max_slots)
    slots = math.ceil(physical_tokens / compression_ratio)
    return slots if max_slots is None else min(slots, max_slots)


def recurrent_slot_count(
    previous_slots: int,
    block_tokens: int,
    *,
    compression_ratio: int = COMPRESSION_RATIO,
    max_slots: int | None = None,
) -> int:
    """Count slots after compressing previous memory plus one raw chunk."""

    previous_slots = _plain_non_negative_int("previous_slots", previous_slots)
    block_tokens = _plain_non_negative_int("block_tokens", block_tokens)
    if previous_slots + block_tokens == 0:
        return 0
    return slot_count_for_physical_tokens(
        previous_slots + block_tokens,
        compression_ratio=compression_ratio,
        max_slots=max_slots,
    )


def cap_target_range(
    start: int,
    assistant_stop: int,
    *,
    cap: int = TARGET_TOKEN_CAP,
) -> tuple[int, int]:
    """Cap a non-empty target without crossing its assistant message end."""

    start = _plain_non_negative_int("start", start)
    assistant_stop = _plain_non_negative_int("assistant_stop", assistant_stop)
    cap = _plain_positive_int("cap", cap)
    if assistant_stop <= start:
        raise ValueError("assistant_stop must follow target start")
    return start, min(assistant_stop, start + cap)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _strict_json(text: Any, *, label: str) -> Any:
    if not isinstance(text, str):
        raise ValueError(f"{label} must be a JSON string")
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON") from error


def _copy_json(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_copy_json(item, label=label) for item in value]
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string object key")
            if key in copied:
                raise ValueError(f"{label} contains a duplicate key")
            copied[key] = _copy_json(item, label=label)
        return copied
    raise ValueError(f"{label} contains a non-JSON value")


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_tool_call(
    raw: Any,
    *,
    message_index: int,
    call_index: int,
) -> dict[str, Any]:
    label = f"messages[{message_index}].tool_calls[{call_index}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(raw) != {"id", "type", "function"}:
        raise ValueError(f"{label} must contain exactly id/type/function")
    call_id = raw.get("id")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"{label}.id must be a non-empty string")
    if raw.get("type") != "function":
        raise ValueError(f"{label}.type must be 'function'")
    function = raw.get("function")
    if not isinstance(function, Mapping) or set(function) != {"name", "arguments"}:
        raise ValueError(f"{label}.function must contain exactly name/arguments")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.function.name must be a non-empty string")
    arguments = _strict_json(
        function.get("arguments"), label=f"{label}.function.arguments"
    )
    if not isinstance(arguments, dict):
        raise ValueError(f"{label}.function.arguments must decode to an object")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _normalize_tools(raw: Any) -> tuple[dict[str, Any], ...]:
    decoded = _strict_json(raw, label="row.tools")
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("row.tools must decode to a non-empty array")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(decoded):
        label = f"row.tools[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} must be an object")
        if item.get("type") != "function" or not isinstance(
            item.get("function"), Mapping
        ):
            raise ValueError(f"{label} must be a function tool")
        function = item["function"]
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.function.name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate declared tool name: {name!r}")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{label}.function.parameters must be an object")
        names.add(name)
        tools.append(_copy_json(item, label=label))
    return tuple(tools)


def normalize_qwen36_agentic_row(row: Mapping[str, Any]) -> Qwen36AgenticRow:
    """Normalize one raw Hugging Face parquet row and enforce call pairing."""

    if not isinstance(row, Mapping):
        raise ValueError("row must be an object")
    if set(row) != {"tools", "messages"}:
        raise ValueError("row must contain exactly tools and messages")
    tools = _normalize_tools(row.get("tools"))
    tool_names = {str(tool["function"]["name"]) for tool in tools}
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(
        raw_messages, (str, bytes)
    ):
        raise ValueError("row.messages must be a sequence")
    if len(raw_messages) < 2:
        raise ValueError("row.messages must contain a prompt and chosen assistant")

    messages: list[dict[str, Any]] = []
    allowed_message_keys = {
        "role",
        "content",
        "reasoning",
        "tool_calls",
        "tool_call_id",
    }
    seen_call_ids: set[str] = set()
    for message_index, raw_message in enumerate(raw_messages):
        label = f"messages[{message_index}]"
        if not isinstance(raw_message, Mapping):
            raise ValueError(f"{label} must be an object")
        if not set(raw_message).issubset(allowed_message_keys):
            raise ValueError(f"{label} contains unsupported fields")
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{label}.role is unsupported")
        if not isinstance(content, str):
            raise ValueError(f"{label}.content must be a string")
        reasoning = raw_message.get("reasoning")
        raw_calls = raw_message.get("tool_calls")
        tool_call_id = raw_message.get("tool_call_id")
        normalized: dict[str, Any] = {"role": role, "content": content}

        if role == "assistant":
            if reasoning is not None:
                if not isinstance(reasoning, str):
                    raise ValueError(f"{label}.reasoning must be a string or null")
                normalized["reasoning_content"] = reasoning
            if tool_call_id is not None:
                raise ValueError(f"{label}.tool_call_id is only valid for tools")
            if raw_calls is not None:
                if not isinstance(raw_calls, Sequence) or isinstance(
                    raw_calls, (str, bytes)
                ):
                    raise ValueError(f"{label}.tool_calls must be an array or null")
                if not raw_calls:
                    raise ValueError(f"{label}.tool_calls cannot be an empty array")
                calls = [
                    _normalize_tool_call(
                        call,
                        message_index=message_index,
                        call_index=call_index,
                    )
                    for call_index, call in enumerate(raw_calls)
                ]
                for call in calls:
                    call_id = str(call["id"])
                    if call_id in seen_call_ids:
                        raise ValueError(f"duplicate tool call id: {call_id!r}")
                    seen_call_ids.add(call_id)
                    if call["function"]["name"] not in tool_names:
                        raise ValueError(
                            f"tool call names an undeclared function: "
                            f"{call['function']['name']!r}"
                        )
                normalized["tool_calls"] = calls
        elif role == "tool":
            if reasoning is not None or raw_calls is not None:
                raise ValueError(f"{label} cannot contain reasoning/tool_calls")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise ValueError(f"{label}.tool_call_id must be a non-empty string")
            normalized["tool_call_id"] = tool_call_id
        else:
            if reasoning is not None or raw_calls is not None or tool_call_id is not None:
                raise ValueError(
                    f"{label} cannot contain reasoning/tool_calls/tool_call_id"
                )
        messages.append(normalized)

    initial_user = 1 if messages[0]["role"] == "system" else 0
    if messages[initial_user]["role"] != "user":
        raise ValueError("messages must begin with user or system+user")
    if initial_user + 1 >= len(messages) or messages[initial_user + 1]["role"] != (
        "assistant"
    ):
        raise ValueError("the complete initial user turn must precede history")
    if any(message["role"] == "system" for message in messages[1:]):
        raise ValueError("system is only allowed as the first message")
    if messages[-1]["role"] != "assistant":
        raise ValueError("the final message must be the chosen assistant completion")
    final = messages[-1]
    if not (
        final["content"].strip()
        or str(final.get("reasoning_content", "")).strip()
        or final.get("tool_calls")
    ):
        raise ValueError("the chosen assistant completion is empty")

    # Every context assistant call must be followed by exactly one maximal
    # result batch containing the same unique IDs.  The final chosen assistant
    # is sampled output and intentionally has no result yet.
    index = initial_user + 1
    while index < len(messages) - 1:
        message = messages[index]
        role = message["role"]
        if role == "assistant":
            calls = message.get("tool_calls")
            if calls:
                stop = index + 1
                while stop < len(messages) - 1 and messages[stop]["role"] == "tool":
                    stop += 1
                expected_ids = [str(call["id"]) for call in calls]
                observed_ids = [
                    str(tool_message["tool_call_id"])
                    for tool_message in messages[index + 1 : stop]
                ]
                if len(expected_ids) != len(observed_ids) or set(expected_ids) != set(
                    observed_ids
                ):
                    raise ValueError(
                        f"messages[{index}]: tool_call_id batch does not exactly pair with "
                        "the preceding assistant calls"
                    )
                if len(set(observed_ids)) != len(observed_ids):
                    raise ValueError("tool result batch contains duplicate IDs")
                index = stop
                continue
            if index + 1 < len(messages) - 1 and messages[index + 1]["role"] not in {
                "user",
                "assistant",
            }:
                raise ValueError("an assistant without calls cannot precede a tool")
        elif role == "tool":
            raise ValueError("tool result has no immediately preceding call batch")
        elif role == "user":
            if index + 1 >= len(messages) or messages[index + 1]["role"] != "assistant":
                raise ValueError("every user message must be followed by an assistant")
        index += 1

    normalized_tools = [dict(tool) for tool in tools]
    prompt_material = {"tools": normalized_tools, "messages": messages[:-1]}
    full_material = {"tools": normalized_tools, "messages": messages}
    return Qwen36AgenticRow(
        example_id=_canonical_digest(full_material),
        prompt_group_id=_canonical_digest(prompt_material),
        tools=tools,
        messages=tuple(messages),
    )


def _as_token_ids(value: Any, *, label: str) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must return token ids")
    if value and isinstance(value[0], Sequence) and not isinstance(
        value[0], (str, bytes)
    ):
        if len(value) != 1:
            raise ValueError(f"{label} unexpectedly returned a batch")
        value = value[0]
    tokens: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError(f"{label} returned a non-integer token")
        tokens.append(token)
    if not tokens:
        raise ValueError(f"{label} returned no token ids")
    return tuple(tokens)


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]],
    add_generation_prompt: bool,
) -> tuple[int, ...]:
    value = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tools=[dict(tool) for tool in tools],
        tokenize=True,
        add_generation_prompt=bool(add_generation_prompt),
        enable_thinking=QWEN36_ENABLE_THINKING,
        preserve_thinking=QWEN36_PRESERVE_THINKING,
        return_dict=False,
    )
    return _as_token_ids(value, label="Qwen3.6 apply_chat_template")


def _apply_chat_text(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]],
    add_generation_prompt: bool,
) -> str:
    value = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tools=[dict(tool) for tool in tools],
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
        enable_thinking=QWEN36_ENABLE_THINKING,
        preserve_thinking=QWEN36_PRESERVE_THINKING,
        return_dict=False,
    )
    if not isinstance(value, str):
        raise TypeError("Qwen3.6 textual chat template must return a string")
    if not value:
        raise ValueError("Qwen3.6 textual chat template returned an empty string")
    return value


def _encode_continuation(tokenizer: Any, text: str) -> tuple[int, ...]:
    value = tokenizer.encode(text, add_special_tokens=False)
    return _as_token_ids(value, label="Qwen3.6 continuation encode")


def _history_stop_after_message(
    row: Qwen36AgenticRow,
    tokenizer: Any,
    *,
    message_stop: int,
    raw_prefix: Sequence[int],
    complete: Sequence[int],
) -> int:
    rendered = _apply_chat_template(
        tokenizer,
        row.messages[:message_stop],
        tools=row.tools,
        add_generation_prompt=False,
    )
    if tuple(complete[: len(rendered)]) != rendered:
        raise ValueError("native message rendering is not an append-only prefix")
    if tuple(rendered[: len(raw_prefix)]) != tuple(raw_prefix):
        raise ValueError("native message rendering changed persistent P")
    return len(rendered) - len(raw_prefix)


def find_tool_batches(
    row: Qwen36AgenticRow,
    tokenizer: Any,
    *,
    raw_prefix_input_ids: Sequence[int],
    history_input_ids: Sequence[int],
    final_prompt_stop: int | None = None,
) -> tuple[Qwen36ToolBatch, ...]:
    """Locate every maximal context tool-result batch in native token space."""

    prefix = tuple(int(token) for token in raw_prefix_input_ids)
    history = tuple(int(token) for token in history_input_ids)
    complete = (*prefix, *history)
    messages = row.messages
    if final_prompt_stop is None:
        final_prompt_stop = len(history)
    final_prompt_stop = _plain_non_negative_int(
        "final_prompt_stop", final_prompt_stop
    )
    if final_prompt_stop > len(history):
        raise ValueError("final_prompt_stop exceeds history")
    batches: list[Qwen36ToolBatch] = []
    index = 0
    while index < len(messages) - 1:
        if messages[index]["role"] != "tool":
            index += 1
            continue
        start = index
        while index < len(messages) - 1 and messages[index]["role"] == "tool":
            index += 1
        stop = index
        history_start = _history_stop_after_message(
            row,
            tokenizer,
            message_stop=start,
            raw_prefix=prefix,
            complete=complete,
        )
        history_stop = _history_stop_after_message(
            row,
            tokenizer,
            message_stop=stop,
            raw_prefix=prefix,
            complete=complete,
        )
        closing = QWEN36_TOOL_RESULT_BATCH_CLOSING_IDS
        if tuple(history[max(0, history_stop - len(closing)) : history_stop]) != closing:
            raise ValueError(
                "native tool-result batch does not end in the verified Qwen3.6 IDs"
            )

        next_assistant_index = stop
        while (
            next_assistant_index < len(messages)
            and messages[next_assistant_index]["role"] != "assistant"
        ):
            next_assistant_index += 1
        if next_assistant_index >= len(messages):
            raise ValueError("tool-result batch has no later assistant message")
        if next_assistant_index == len(messages) - 1:
            assistant_message_start = final_prompt_stop - len(
                QWEN36_GENERATION_PROMPT_IDS
            )
            assistant_stop = len(history)
            if tuple(history[assistant_message_start:final_prompt_stop]) != (
                QWEN36_GENERATION_PROMPT_IDS
            ):
                raise ValueError(
                    "segmented final assistant does not retain verified G5 IDs"
                )
        else:
            assistant_message_start = _history_stop_after_message(
                row,
                tokenizer,
                message_stop=next_assistant_index,
                raw_prefix=prefix,
                complete=complete,
            )
            assistant_stop = _history_stop_after_message(
                row,
                tokenizer,
                message_stop=next_assistant_index + 1,
                raw_prefix=prefix,
                complete=complete,
            )
        if assistant_stop <= assistant_message_start:
            raise ValueError("native assistant continuation is empty")
        batches.append(
            Qwen36ToolBatch(
                batch_index=len(batches),
                message_start=start,
                message_stop=stop,
                history_start=history_start,
                history_stop=history_stop,
                next_assistant_start=assistant_message_start,
                next_assistant_stop=assistant_stop,
                tool_call_ids=tuple(
                    str(message["tool_call_id"]) for message in messages[start:stop]
                ),
            )
        )
    return tuple(batches)


def agentic_compression_events(
    history_tokens: int,
    tool_batches: Sequence[Qwen36ToolBatch],
    *,
    final_prompt_stop: int,
    final_assistant_stop: int,
    block_tokens: int = EVICTION_BLOCK_TOKENS,
    target_cap: int = TARGET_TOKEN_CAP,
) -> tuple[AgenticCompressionEvent, ...]:
    """Build the contiguous one-batch-lag recurrence for one chosen target."""

    history_tokens = _plain_non_negative_int("history_tokens", history_tokens)
    final_prompt_stop = _plain_non_negative_int(
        "final_prompt_stop", final_prompt_stop
    )
    final_assistant_stop = _plain_non_negative_int(
        "final_assistant_stop", final_assistant_stop
    )
    block_tokens = _plain_positive_int("block_tokens", block_tokens)
    target_cap = _plain_positive_int("target_cap", target_cap)
    if final_assistant_stop > history_tokens:
        raise ValueError("final assistant stop exceeds history")
    target_start, target_stop = cap_target_range(
        final_prompt_stop,
        final_assistant_stop,
        cap=target_cap,
    )
    if len(tool_batches) < 2:
        return ()
    previous_stop = -1
    for expected_index, batch in enumerate(tool_batches):
        if batch.batch_index != expected_index:
            raise ValueError("tool batch indices must be contiguous")
        if batch.history_stop <= max(previous_stop, batch.history_start):
            raise ValueError("tool batch token ranges must be ordered and non-empty")
        if batch.history_stop > final_prompt_stop:
            raise ValueError("tool batch extends beyond final generation prompt")
        previous_stop = batch.history_stop

    eviction_stop = tool_batches[-2].history_stop
    if not 0 < eviction_stop < final_prompt_stop:
        raise ValueError("one-batch-lag eviction boundary is invalid")
    events: list[AgenticCompressionEvent] = []
    cursor = 0
    # A physical update must never straddle a native tool-batch closure.  This
    # mirrors serving, which makes each completed old batch independently
    # eligible, while event_index still provides one globally contiguous
    # recurrence cursor.
    for batch in tool_batches[:-1]:
        segment_stop = batch.history_stop
        chunk_index = 0
        while cursor < segment_stop:
            block_stop = min(cursor + block_tokens, segment_stop)
            events.append(
                AgenticCompressionEvent(
                    event_index=len(events),
                    batch_index=batch.batch_index,
                    chunk_index=chunk_index,
                    block_start=cursor,
                    block_stop=block_stop,
                    working_start=block_stop,
                    working_stop=final_prompt_stop,
                    target_start=target_start,
                    target_stop=target_stop,
                    is_serving_state=block_stop == eviction_stop,
                )
            )
            cursor = block_stop
            chunk_index += 1

    if not events or events[0].block_start != 0:
        raise RuntimeError("agentic recurrence did not start at history zero")
    for previous, current in zip(events, events[1:]):
        if previous.block_stop != current.block_start:
            raise RuntimeError("agentic recurrence block cursor is not contiguous")
        if previous.is_serving_state:
            raise RuntimeError("only the final event may be a serving state")
    if events[-1].block_stop != eviction_stop or not events[-1].is_serving_state:
        raise RuntimeError("agentic recurrence did not reach its serving boundary")
    return tuple(events)


def render_qwen36_agentic_example(
    row: Mapping[str, Any] | Qwen36AgenticRow,
    tokenizer: Any,
    *,
    block_tokens: int = EVICTION_BLOCK_TOKENS,
    target_cap: int = TARGET_TOKEN_CAP,
) -> Qwen36AgenticExample:
    """Normalize, natively render, and plan one A3B chosen completion."""

    normalized = (
        row if isinstance(row, Qwen36AgenticRow) else normalize_qwen36_agentic_row(row)
    )
    first_assistant = next(
        index
        for index, message in enumerate(normalized.messages)
        if message["role"] == "assistant"
    )
    prefix_messages = normalized.messages[:first_assistant]
    prefix_ids = _apply_chat_template(
        tokenizer,
        prefix_messages,
        tools=normalized.tools,
        add_generation_prompt=False,
    )
    final_prompt_ids = _apply_chat_template(
        tokenizer,
        normalized.messages[:-1],
        tools=normalized.tools,
        add_generation_prompt=True,
    )
    if final_prompt_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("native final generation prompt does not preserve P exactly")
    if final_prompt_ids[-len(QWEN36_GENERATION_PROMPT_IDS) :] != (
        QWEN36_GENERATION_PROMPT_IDS
    ):
        raise ValueError("native final generation prompt does not end in verified G5")

    prompt_text = _apply_chat_text(
        tokenizer,
        normalized.messages[:-1],
        tools=normalized.tools,
        add_generation_prompt=True,
    )
    full_text = _apply_chat_text(
        tokenizer,
        normalized.messages,
        tools=normalized.tools,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "native completed-chat text does not preserve its generation prompt"
        )
    continuation_text = full_text[len(prompt_text) :]
    if not continuation_text:
        raise ValueError("chosen assistant textual continuation is empty")
    continuation_ids = _encode_continuation(tokenizer, continuation_text)
    history_ids = (
        *final_prompt_ids[len(prefix_ids) :],
        *continuation_ids,
    )
    final_prompt_stop = len(final_prompt_ids) - len(prefix_ids)
    final_target_start, final_target_stop = cap_target_range(
        final_prompt_stop,
        len(history_ids),
        cap=target_cap,
    )

    batches = find_tool_batches(
        normalized,
        tokenizer,
        raw_prefix_input_ids=prefix_ids,
        history_input_ids=history_ids,
        final_prompt_stop=final_prompt_stop,
    )
    events = agentic_compression_events(
        len(history_ids),
        batches,
        final_prompt_stop=final_target_start,
        final_assistant_stop=len(history_ids),
        block_tokens=block_tokens,
        target_cap=target_cap,
    )
    return Qwen36AgenticExample(
        example_id=normalized.example_id,
        prompt_group_id=normalized.prompt_group_id,
        raw_prefix_input_ids=prefix_ids,
        history_input_ids=history_ids,
        tool_batches=batches,
        events=events,
        final_target_start=final_target_start,
        final_target_stop=final_target_stop,
    )


__all__ = [
    "AgenticCompressionEvent",
    "COMPRESSION_RATIO",
    "EVICTION_BLOCK_TOKENS",
    "QWEN36_ASSISTANT_OPENING_IDS",
    "QWEN36_ENABLE_THINKING",
    "QWEN36_GENERATION_PROMPT_IDS",
    "QWEN36_PRESERVE_THINKING",
    "QWEN36_TOOL_RESULT_BATCH_CLOSING_IDS",
    "Qwen36AgenticExample",
    "Qwen36AgenticRow",
    "Qwen36ToolBatch",
    "TARGET_TOKEN_CAP",
    "agentic_compression_events",
    "cap_target_range",
    "find_tool_batches",
    "normalize_qwen36_agentic_row",
    "recurrent_slot_count",
    "render_qwen36_agentic_example",
    "slot_count_for_physical_tokens",
]
