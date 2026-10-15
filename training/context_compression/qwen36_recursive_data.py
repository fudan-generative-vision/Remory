"""Token-clock data contract for Qwen3.6 recursive context compression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


EVICTION_BLOCK_TOKENS = 1024
RAW_WORKING_MEMORY_TOKENS = 1024
COMPRESSION_RATIO = 16
QWEN36_ENABLE_THINKING = True
QWEN36_ASSISTANT_OPENING_TEXT = "<|im_start|>assistant\n<think>\n"
QWEN36_ASSISTANT_OPENING_IDS = (248045, 74455, 198, 248068, 198)


@dataclass(frozen=True)
class Qwen36TokenClockExample:
    example_id: str
    raw_prefix_input_ids: tuple[int, ...]
    history_input_ids: tuple[int, ...]


@dataclass(frozen=True)
class TokenClockEvent:
    event_index: int
    block_start: int
    block_stop: int
    working_memory_start: int
    working_memory_stop: int
    target_start: int
    target_stop: int


def recurrent_slot_counts(
    event_count: int,
    *,
    block_tokens: int = EVICTION_BLOCK_TOKENS,
    compression_ratio: int = COMPRESSION_RATIO,
) -> tuple[int, ...]:
    if isinstance(event_count, bool) or not isinstance(event_count, int):
        raise TypeError("event_count must be a plain int")
    if event_count < 0:
        raise ValueError("event_count must be non-negative")
    if block_tokens <= 0 or compression_ratio <= 0:
        raise ValueError("block_tokens and compression_ratio must be positive")
    counts: list[int] = []
    previous = 0
    for _ in range(event_count):
        previous = (previous + block_tokens + compression_ratio - 1) // compression_ratio
        counts.append(previous)
    return tuple(counts)


def token_clock_events(
    history_tokens: int,
    *,
    target_tokens: int,
    block_tokens: int = EVICTION_BLOCK_TOKENS,
    working_memory_tokens: int = RAW_WORKING_MEMORY_TOKENS,
) -> tuple[TokenClockEvent, ...]:
    """Return every legal chronological 1024-1024 compression event.

    An event is legal only when its eviction block and raw working-memory
    suffix are complete and at least one shared teacher-forcing token follows.
    The returned target range is capped but never changes the trigger cadence.
    """

    for name, value in (
        ("history_tokens", history_tokens),
        ("target_tokens", target_tokens),
        ("block_tokens", block_tokens),
        ("working_memory_tokens", working_memory_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a plain int")
    if history_tokens < 0:
        raise ValueError("history_tokens must be non-negative")
    if target_tokens <= 0 or block_tokens <= 0 or working_memory_tokens <= 0:
        raise ValueError("target/block/working-memory token counts must be positive")

    events: list[TokenClockEvent] = []
    event_index = 0
    while True:
        block_start = event_index * block_tokens
        block_stop = block_start + block_tokens
        working_stop = block_stop + working_memory_tokens
        if working_stop >= history_tokens:
            break
        events.append(
            TokenClockEvent(
                event_index=event_index,
                block_start=block_start,
                block_stop=block_stop,
                working_memory_start=block_stop,
                working_memory_stop=working_stop,
                target_start=working_stop,
                target_stop=min(history_tokens, working_stop + target_tokens),
            )
        )
        event_index += 1
    return tuple(events)


def build_qwen36_token_clock_example(
    row: Mapping[str, Any],
    tokenizer: Any,
) -> Qwen36TokenClockExample:
    """Render one successful system/user→assistant row with the native template."""

    conversations = row.get("conversations")
    if not isinstance(conversations, Sequence) or isinstance(conversations, (str, bytes)):
        raise ValueError("row.conversations must be a message sequence")
    messages: list[dict[str, Any]] = []
    for message in conversations:
        if not isinstance(message, Mapping):
            raise ValueError("every conversation message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("messages must contain supported roles and string content")
        normalized = {"role": str(role), "content": content}
        reasoning_content = message.get("reasoning_content")
        if reasoning_content is not None:
            if role != "assistant" or not isinstance(reasoning_content, str):
                raise ValueError(
                    "reasoning_content must be a string on the final assistant message"
                )
            # Qwen3.8's native chat template consumes this field separately from
            # the visible final answer.  Retaining it here is essential: folding
            # it into content produces a second, empty <think> envelope, while
            # dropping it silently removes the regenerated reasoning trajectory.
            normalized["reasoning_content"] = reasoning_content
        messages.append(normalized)
    assistant_indices = [
        i for i, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if not assistant_indices or assistant_indices[-1] != len(messages) - 1:
        raise ValueError("history must end with an assistant message")
    first_assistant_index = assistant_indices[0]
    prefix_messages = messages[:first_assistant_index]
    prefix_roles = [message["role"] for message in prefix_messages]
    if prefix_roles not in (["user"], ["system", "user"]):
        raise ValueError("prefix must be exactly user or system+user")
    history_roles = [message["role"] for message in messages[first_assistant_index:]]
    expected_history_roles = [
        "assistant" if index % 2 == 0 else "user"
        for index in range(len(history_roles))
    ]
    if history_roles != expected_history_roles:
        raise ValueError("history must alternate assistant/user and end with assistant")

    prefix_ids = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=QWEN36_ENABLE_THINKING,
        return_dict=False,
    )
    complete_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=QWEN36_ENABLE_THINKING,
        return_dict=False,
    )
    prefix_ids = [int(value) for value in prefix_ids]
    complete_ids = [int(value) for value in complete_ids]
    if not prefix_ids or complete_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("native complete rendering does not preserve the prefix exactly")
    history_ids = complete_ids[len(prefix_ids) :]
    if not history_ids:
        raise ValueError("assistant history is empty")
    return Qwen36TokenClockExample(
        example_id=str(row.get("id", "")),
        raw_prefix_input_ids=tuple(prefix_ids),
        history_input_ids=tuple(history_ids),
    )


__all__ = [
    "COMPRESSION_RATIO",
    "EVICTION_BLOCK_TOKENS",
    "QWEN36_ASSISTANT_OPENING_IDS",
    "QWEN36_ASSISTANT_OPENING_TEXT",
    "QWEN36_ENABLE_THINKING",
    "Qwen36TokenClockExample",
    "RAW_WORKING_MEMORY_TOKENS",
    "TokenClockEvent",
    "build_qwen36_token_clock_example",
    "recurrent_slot_counts",
    "token_clock_events",
]
