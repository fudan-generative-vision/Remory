"""Episode-level data ABIs for recursive GrepSeek history compression.

The materialized representation is deliberately independent of a compression
ratio.  One row contains the complete ordered recurrence for one episode:

* ``raw_prefix_input_ids`` is the persistent native system/question prefix;
  appending the exact seven-token generation boundary reconstructs the native
  initial generation prompt;
* every entry in ``history_suffix_input_ids`` is that fixed generation
  boundary followed by one assistant continuation and its canonical tool
  message, with no system/question duplication; and
* ``target_ids[i]`` is the text suffix after the real native generation prompt
  for the assistant following round ``i``.

At training time the raw prefix conditions every recurrence but is never
compressed.  Only the previous continuous memory plus the current history
suffix is passed through the recursive compressor.

Tool payloads are parsed with the released seven-key contract and rendered
with ``information_lines=[]`` so stdout never occurs twice.  Targets are
derived at the rendered-text boundary, then tokenized independently.  This is
intentional: a small number of native trajectories have a newline BPE merge at
the prompt/continuation boundary, so subtracting token-list lengths is not an
exact text-suffix operation.

The module also exposes a separate working-memory-v3 ABI.  It deliberately
does not reinterpret v2 rows:

* ``raw_prefix_input_ids`` contains the complete native system and user turns,
  including the user's ``<|im_end|>\n``;
* ``raw_working_memory_input_ids[i]`` is the newest assistant/tool exchange
  with its complete native role and tool template, kept discrete for the next
  target;
* ``recursive_semantic_input_ids[i]`` contains only assistant content followed
  by canonical tool content, with all intermediate chat/tool-role template
  text removed;
  and
* recurrence is delayed by one target: target 0 sees raw ``W1`` and no memory,
  while target ``i`` first folds semantic ``W_i`` into the previous memory and
  keeps raw ``W_{i+1}`` beside the resulting memory.

Both formats remain ratio-free.  In particular, the v3 data ABI does not bake
the intended 1/16 compressor ratio into materialized rows.
"""

from __future__ import annotations

import bisect
import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from context_compression.grepseek_data import (
    DEFAULT_VALIDATION_FRACTION,
    _apply_chat_ids,
    _canonical_tool_content,
    _render_chat,
    _validate_messages,
    assign_episode_split,
    iter_grepseek_rows,
    parse_tool_response,
)


FORMAT_VERSION = 2
SCHEMA_NAME = "grepseek_recursive_v2"
WORKING_MEMORY_FORMAT_VERSION = 3
WORKING_MEMORY_SCHEMA_NAME = "grepseek_recursive_working_memory_v3"
GENERATION_BOUNDARY_TEXT = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n"
)
GENERATION_BOUNDARY_IDS: Tuple[int, ...] = (
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    198,
)
ASSISTANT_OPENING_TEXT = "<|im_start|>assistant\n<think>\n"
ASSISTANT_OPENING_IDS: Tuple[int, ...] = (
    248045,
    74455,
    198,
    248068,
    198,
)
USER_TURN_END_TEXT = "<|im_end|>\n"


@dataclass(frozen=True)
class RecursivePreparationConfig:
    """Deterministic episode split controls for ratio-free recursive data."""

    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")


def _encode(tokenizer: Any, text: str) -> List[int]:
    value = tokenizer.encode(str(text), add_special_tokens=False)
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("tokenizer.encode result has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("tokenizer.encode unexpectedly returned a batch")
        value = value[0]
    return [int(token_id) for token_id in value]


def _native_render(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> Tuple[str, List[int]]:
    text = _render_chat(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    native_ids = _apply_chat_ids(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
    )
    encoded_ids = _encode(tokenizer, text)
    if encoded_ids != native_ids:
        raise ValueError(
            "tokenizing rendered native chat does not reproduce "
            "apply_chat_template ids"
        )
    return text, native_ids


def _template_sha256(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str) and template:
        material = template
    elif isinstance(template, Mapping) and template:
        material = json.dumps(
            dict(template), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    else:
        raise ValueError(
            "recursive GrepSeek preparation requires tokenizer.chat_template "
            "for an auditable template hash"
        )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _validate_episode(
    row: Mapping[str, Any],
) -> Tuple[str, List[Dict[str, str]]]:
    episode_id, messages = _validate_messages(row)
    if len(messages) < 5 or len(messages) % 2 == 0:
        raise ValueError(
            f"episode {episode_id!r}: expected system, user, one or more "
            "assistant/tool rounds, and a final assistant"
        )
    for index in range(2, len(messages) - 1):
        expected = "assistant" if index % 2 == 0 else "tool"
        if messages[index]["role"] != expected:
            raise ValueError(
                f"episode {episode_id!r} message {index}: expected role "
                f"{expected!r}, got {messages[index]['role']!r}"
            )
    if messages[-1]["role"] != "assistant":
        raise ValueError(
            f"episode {episode_id!r}: final message must be assistant"
        )
    return episode_id, messages


def _canonicalize_tools(
    episode_id: str,
    messages: Sequence[Mapping[str, str]],
) -> Tuple[List[Dict[str, str]], List[int], List[int]]:
    canonical: List[Dict[str, str]] = []
    released_lengths: List[int] = []
    canonical_lengths: List[int] = []
    for index, message in enumerate(messages):
        normalized = {"role": str(message["role"]), "content": str(message["content"])}
        if normalized["role"] == "tool":
            payload = parse_tool_response(
                normalized["content"],
                episode_id=episode_id,
                message_index=index,
            )
            released_lengths.append(len(normalized["content"]))
            normalized["content"] = _canonical_tool_content(
                payload, str(payload["stdout"])
            )
            canonical_lengths.append(len(normalized["content"]))
        canonical.append(normalized)
    return canonical, released_lengths, canonical_lengths


def _rendered_suffix(
    tokenizer: Any,
    prefix_messages: Sequence[Mapping[str, str]],
    complete_messages: Sequence[Mapping[str, str]],
) -> Tuple[str, str]:
    prompt_text, _ = _native_render(
        tokenizer,
        prefix_messages,
        add_generation_prompt=True,
    )
    full_text, _ = _native_render(
        tokenizer,
        complete_messages,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "native completed chat text does not start with its generation "
            "prompt text"
        )
    suffix = full_text[len(prompt_text) :]
    if not suffix:
        raise ValueError("native rendered continuation suffix is empty")
    return prompt_text, suffix


def _token_contract(generation_boundary_ids: Sequence[int]) -> Dict[str, Any]:
    return {
        "name": "grepseek_recursive_native_text_suffix_v2",
        "generation_boundary_text": GENERATION_BOUNDARY_TEXT,
        "generation_boundary_ids": [int(value) for value in generation_boundary_ids],
        "raw_prefix": (
            "native_system_user_generation_prompt_without_trailing_exact_"
            "generation_boundary"
        ),
        "history_suffix": (
            "generation_boundary_ids_then_independently_encoded_native_"
            "assistant_and_tool_text_suffix"
        ),
        "recurrence_conditioning": (
            "raw_prefix_persists_discrete_and_conditions_every_round;compress_"
            "only_previous_memory_plus_current_history_suffix"
        ),
        "target": (
            "independently_encode_full_native_render_text_suffix_after_real_"
            "generation_prompt_text"
        ),
        "tool_canonicalization": (
            "parse_seven_key_payload_then_render_stdout_once_with_"
            "information_lines_empty"
        ),
        "compression": "chosen_at_training_time_not_materialized",
    }


def _working_memory_token_contract(
    assistant_opening_ids: Sequence[int],
) -> Dict[str, Any]:
    """Return the explicit, ratio-free working-memory-v3 token contract."""

    return {
        "name": "grepseek_recursive_working_memory_native_v3",
        "assistant_opening_text": ASSISTANT_OPENING_TEXT,
        "assistant_opening_ids": [int(value) for value in assistant_opening_ids],
        "raw_prefix": (
            "complete_native_system_user_turns_through_user_im_end_newline"
        ),
        "recursive_semantic": (
            "independently_encoded_assistant_content_then_canonical_tool_content;"
            "no_intermediate_chat_role_or_tool_wrapper_template"
        ),
        "raw_working_memory": (
            "independently_encoded_complete_native_assistant_tool_exchange_"
            "starting_with_assistant_opening"
        ),
        "one_round_lag": (
            "target_0_uses_raw_W1_without_memory;target_i_first_folds_"
            "semantic_Wi_into_Mprev_then_uses_raw_Wi+1"
        ),
        "recovery": (
            "raw_prefix_then_optional_memory_then_latest_raw_working_memory_"
            "then_assistant_opening"
        ),
        "recurrence_conditioning": (
            "compress_only_previous_memory_plus_recursive_semantic;"
            "raw_prefix_and_raw_working_memory_are_not_compressor_source"
        ),
        "target": (
            "independently_encode_full_native_render_text_suffix_after_real_"
            "generation_prompt_text"
        ),
        "tool_canonicalization": (
            "raw_working_memory_parses_seven_key_payload_then_renders_stdout_"
            "once_with_information_lines_empty;recursive_semantic_uses_"
            "canonical_tool_content_without_native_tool_wrapper"
        ),
        "compression": "chosen_at_training_time_not_materialized",
    }


def build_grepseek_recursive_episode(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    config: Optional[RecursivePreparationConfig] = None,
    tokenizer_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one strict, future-free recursive record from one raw episode."""

    cfg = config or RecursivePreparationConfig()
    episode_id, released_messages = _validate_episode(row)
    messages, released_tool_lengths, canonical_tool_lengths = _canonicalize_tools(
        episode_id, released_messages
    )
    generation_boundary_ids = _encode(tokenizer, GENERATION_BOUNDARY_TEXT)
    if tuple(generation_boundary_ids) != GENERATION_BOUNDARY_IDS:
        raise ValueError(
            "tokenizer disagrees with the GrepSeek native seven-token "
            f"generation boundary: expected={list(GENERATION_BOUNDARY_IDS)}, "
            f"got={generation_boundary_ids}"
        )

    initial_prompt_text, initial_prompt_ids = _native_render(
        tokenizer,
        messages[:2],
        add_generation_prompt=True,
    )
    if not initial_prompt_text.endswith(GENERATION_BOUNDARY_TEXT):
        raise ValueError(
            "native initial generation prompt text does not end in the exact "
            "seven-token generation boundary"
        )
    if initial_prompt_ids[-len(GENERATION_BOUNDARY_IDS) :] != list(
        GENERATION_BOUNDARY_IDS
    ):
        raise ValueError(
            "native initial generation prompt ids do not end in the exact "
            "seven-token generation boundary"
        )
    raw_prefix_text = initial_prompt_text[: -len(GENERATION_BOUNDARY_TEXT)]
    raw_prefix_input_ids = initial_prompt_ids[: -len(GENERATION_BOUNDARY_IDS)]
    if not raw_prefix_input_ids:
        raise ValueError("persistent raw system/question prefix is empty")
    if _encode(tokenizer, raw_prefix_text) != raw_prefix_input_ids:
        raise ValueError(
            "native raw prefix text does not independently reproduce its ids"
        )

    round_count = (len(messages) - 3) // 2

    history_suffix_input_ids: List[List[int]] = []
    target_ids: List[List[int]] = []
    ordered_rounds: List[Dict[str, Any]] = []
    history_suffix_text_lengths: List[int] = []
    target_text_lengths: List[int] = []

    for zero_based_round in range(round_count):
        assistant_index = 2 + 2 * zero_based_round
        tool_index = assistant_index + 1
        target_index = tool_index + 1
        history_through_tool = messages[: tool_index + 1]

        _, target_suffix = _rendered_suffix(
            tokenizer,
            history_through_tool,
            messages[: target_index + 1],
        )
        current_target_ids = _encode(tokenizer, target_suffix)
        if not current_target_ids:
            raise ValueError(
                f"episode {episode_id!r} round {zero_based_round + 1}: "
                "target tokenizes empty"
            )
        target_ids.append(current_target_ids)
        target_text_lengths.append(len(target_suffix))

        prior_history = messages[:assistant_index]
        _, completed_suffix = _rendered_suffix(
            tokenizer,
            prior_history,
            history_through_tool,
        )
        completed_suffix_ids = _encode(tokenizer, completed_suffix)
        current_history_suffix_ids = generation_boundary_ids + completed_suffix_ids
        if current_history_suffix_ids[: len(GENERATION_BOUNDARY_IDS)] != list(
            GENERATION_BOUNDARY_IDS
        ):
            raise AssertionError("history suffix lost its native boundary")
        history_suffix_input_ids.append(current_history_suffix_ids)
        history_suffix_text_lengths.append(
            len(GENERATION_BOUNDARY_TEXT) + len(completed_suffix)
        )

        ordered_rounds.append(
            {
                "round_index": zero_based_round + 1,
                "assistant_message_index": assistant_index,
                "tool_message_index": tool_index,
                "target_assistant_message_index": target_index,
                "history_suffix_array_index": zero_based_round,
                "history_suffix_token_length": len(current_history_suffix_ids),
                "target_array_index": zero_based_round,
                "target_token_length": len(current_target_ids),
            }
        )

    resolved_tokenizer_name = str(
        tokenizer_name
        or getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    )
    return {
        "format_version": FORMAT_VERSION,
        "schema": SCHEMA_NAME,
        "episode_id": episode_id,
        "split": assign_episode_split(
            episode_id,
            validation_fraction=cfg.validation_fraction,
            seed=cfg.seed,
        ),
        "round_count": round_count,
        "ordered_rounds": ordered_rounds,
        "raw_prefix_input_ids": raw_prefix_input_ids,
        "history_suffix_input_ids": history_suffix_input_ids,
        "target_ids": target_ids,
        "raw_lengths": {
            "raw_prefix_tokens": len(raw_prefix_input_ids),
            "raw_prefix_characters": len(raw_prefix_text),
            "history_suffix_tokens": [
                len(source) for source in history_suffix_input_ids
            ],
            "history_suffix_characters": history_suffix_text_lengths,
            "target_tokens": [len(target) for target in target_ids],
            "target_characters": target_text_lengths,
            "assistant_content_characters": [
                len(messages[index]["content"])
                for index in range(2, len(messages), 2)
            ],
            "released_tool_content_characters": released_tool_lengths,
            "canonical_tool_content_characters": canonical_tool_lengths,
        },
        "chat_template_sha256": _template_sha256(tokenizer),
        "tokenizer_name": resolved_tokenizer_name,
        "token_contract": _token_contract(generation_boundary_ids),
    }


def build_grepseek_recursive_working_memory_episode(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    config: Optional[RecursivePreparationConfig] = None,
    tokenizer_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one strict working-memory-v3 episode record.

    Arrays use completed tool-round order.  For target array index ``i`` the
    raw working-memory array index is also ``i``.  Index 0 has no recurrent
    memory.  For ``i >= 1``, ``memory_transition_semantic_array_index`` is
    ``i - 1``: that older semantic exchange is folded into ``M_prev`` before
    the target sees the current raw exchange.  This makes the one-round lag an
    explicit data invariant rather than trainer-side convention.
    """

    cfg = config or RecursivePreparationConfig()
    episode_id, released_messages = _validate_episode(row)
    messages, released_tool_lengths, canonical_tool_lengths = _canonicalize_tools(
        episode_id, released_messages
    )

    assistant_opening_ids = _encode(tokenizer, ASSISTANT_OPENING_TEXT)
    if tuple(assistant_opening_ids) != ASSISTANT_OPENING_IDS:
        raise ValueError(
            "tokenizer disagrees with the GrepSeek native five-token "
            f"assistant opening: expected={list(ASSISTANT_OPENING_IDS)}, "
            f"got={assistant_opening_ids}"
        )

    raw_prefix_text, raw_prefix_input_ids = _native_render(
        tokenizer,
        messages[:2],
        add_generation_prompt=False,
    )
    initial_prompt_text, initial_prompt_ids = _native_render(
        tokenizer,
        messages[:2],
        add_generation_prompt=True,
    )
    if not raw_prefix_text.endswith(USER_TURN_END_TEXT):
        raise ValueError(
            "native system/user prefix does not end with user <|im_end|> newline"
        )
    if initial_prompt_text != raw_prefix_text + ASSISTANT_OPENING_TEXT:
        raise ValueError(
            "native initial prompt is not the complete system/user prefix "
            "followed by the exact assistant opening"
        )
    if initial_prompt_ids != raw_prefix_input_ids + assistant_opening_ids:
        raise ValueError(
            "native initial prompt ids are not raw_prefix_input_ids followed "
            "by assistant_opening_ids"
        )
    if not raw_prefix_input_ids:
        raise ValueError("persistent complete system/user prefix is empty")
    if _encode(tokenizer, raw_prefix_text) != raw_prefix_input_ids:
        raise ValueError(
            "native complete system/user prefix text does not independently "
            "reproduce its ids"
        )

    round_count = (len(messages) - 3) // 2
    recursive_semantic_input_ids: List[List[int]] = []
    raw_working_memory_input_ids: List[List[int]] = []
    target_ids: List[List[int]] = []
    ordered_rounds: List[Dict[str, Any]] = []
    semantic_text_lengths: List[int] = []
    assistant_semantic_token_lengths: List[int] = []
    tool_semantic_token_lengths: List[int] = []
    raw_working_memory_text_lengths: List[int] = []
    target_text_lengths: List[int] = []

    for zero_based_round in range(round_count):
        assistant_index = 2 + 2 * zero_based_round
        tool_index = assistant_index + 1
        target_index = tool_index + 1
        history_through_tool = messages[: tool_index + 1]

        assistant_semantic_text = str(messages[assistant_index]["content"])
        tool_semantic_text = str(messages[tool_index]["content"])
        assistant_semantic_ids = _encode(tokenizer, assistant_semantic_text)
        tool_semantic_ids = _encode(tokenizer, tool_semantic_text)
        if not assistant_semantic_ids:
            raise ValueError(
                f"episode {episode_id!r} round {zero_based_round + 1}: "
                "assistant semantic content tokenizes empty"
            )
        if not tool_semantic_ids:
            raise ValueError(
                f"episode {episode_id!r} round {zero_based_round + 1}: "
                "canonical tool semantic content tokenizes empty"
            )
        semantic_ids = assistant_semantic_ids + tool_semantic_ids
        tool_semantic_start = len(assistant_semantic_ids)
        if semantic_ids[:tool_semantic_start] != assistant_semantic_ids:
            raise AssertionError("assistant semantic concatenation boundary drifted")
        if semantic_ids[tool_semantic_start:] != tool_semantic_ids:
            raise AssertionError("tool semantic concatenation boundary drifted")
        recursive_semantic_input_ids.append(semantic_ids)
        assistant_semantic_token_lengths.append(len(assistant_semantic_ids))
        tool_semantic_token_lengths.append(len(tool_semantic_ids))
        semantic_text_lengths.append(
            len(assistant_semantic_text) + len(tool_semantic_text)
        )

        prior_history = messages[:assistant_index]
        _, completed_suffix = _rendered_suffix(
            tokenizer,
            prior_history,
            history_through_tool,
        )
        raw_working_memory_text = ASSISTANT_OPENING_TEXT + completed_suffix
        current_raw_working_memory_ids = _encode(
            tokenizer, raw_working_memory_text
        )
        if not current_raw_working_memory_ids:
            raise ValueError(
                f"episode {episode_id!r} round {zero_based_round + 1}: "
                "raw working memory tokenizes empty"
            )
        # Verify the complete native assistant/tool span at the rendered-text
        # boundary.  We intentionally encode the whole span at once so native
        # newline BPE merges inside W are preserved.
        prior_text, _ = _native_render(
            tokenizer,
            prior_history,
            add_generation_prompt=False,
        )
        complete_text, _ = _native_render(
            tokenizer,
            history_through_tool,
            add_generation_prompt=False,
        )
        if complete_text != prior_text + raw_working_memory_text:
            raise ValueError(
                "native completed chat does not decompose into prior chat "
                "plus one complete raw working-memory exchange"
            )
        raw_working_memory_input_ids.append(current_raw_working_memory_ids)
        raw_working_memory_text_lengths.append(len(raw_working_memory_text))

        _, target_suffix = _rendered_suffix(
            tokenizer,
            history_through_tool,
            messages[: target_index + 1],
        )
        current_target_ids = _encode(tokenizer, target_suffix)
        if not current_target_ids:
            raise ValueError(
                f"episode {episode_id!r} round {zero_based_round + 1}: "
                "target tokenizes empty"
            )
        target_ids.append(current_target_ids)
        target_text_lengths.append(len(target_suffix))

        memory_source_index = zero_based_round - 1
        ordered_rounds.append(
            {
                "round_index": zero_based_round + 1,
                "assistant_message_index": assistant_index,
                "tool_message_index": tool_index,
                "target_assistant_message_index": target_index,
                "target_generation_round": zero_based_round + 2,
                "recursive_semantic_array_index": zero_based_round,
                "recursive_semantic_token_length": len(semantic_ids),
                "assistant_semantic_tokens": len(assistant_semantic_ids),
                "tool_semantic_start": tool_semantic_start,
                "tool_semantic_tokens": len(tool_semantic_ids),
                "memory_transition_semantic_array_index": memory_source_index,
                "memory_summarizes_through_round": zero_based_round,
                "raw_working_memory_array_index": zero_based_round,
                "raw_working_memory_token_length": len(
                    current_raw_working_memory_ids
                ),
                "target_array_index": zero_based_round,
                "target_token_length": len(current_target_ids),
            }
        )

    resolved_tokenizer_name = str(
        tokenizer_name
        or getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    )
    return {
        "format_version": WORKING_MEMORY_FORMAT_VERSION,
        "schema": WORKING_MEMORY_SCHEMA_NAME,
        "episode_id": episode_id,
        "split": assign_episode_split(
            episode_id,
            validation_fraction=cfg.validation_fraction,
            seed=cfg.seed,
        ),
        "round_count": round_count,
        "ordered_rounds": ordered_rounds,
        "raw_prefix_input_ids": raw_prefix_input_ids,
        "assistant_opening_ids": assistant_opening_ids,
        "recursive_semantic_input_ids": recursive_semantic_input_ids,
        "raw_working_memory_input_ids": raw_working_memory_input_ids,
        "target_ids": target_ids,
        "raw_lengths": {
            "raw_prefix_tokens": len(raw_prefix_input_ids),
            "raw_prefix_characters": len(raw_prefix_text),
            "recursive_semantic_tokens": [
                len(source) for source in recursive_semantic_input_ids
            ],
            "recursive_semantic_characters": semantic_text_lengths,
            "assistant_semantic_tokens": assistant_semantic_token_lengths,
            "tool_semantic_tokens": tool_semantic_token_lengths,
            "raw_working_memory_tokens": [
                len(source) for source in raw_working_memory_input_ids
            ],
            "raw_working_memory_characters": raw_working_memory_text_lengths,
            "target_tokens": [len(target) for target in target_ids],
            "target_characters": target_text_lengths,
            "assistant_content_characters": [
                len(messages[index]["content"])
                for index in range(2, len(messages), 2)
            ],
            "released_tool_content_characters": released_tool_lengths,
            "canonical_tool_content_characters": canonical_tool_lengths,
        },
        "chat_template_sha256": _template_sha256(tokenizer),
        "tokenizer_name": resolved_tokenizer_name,
        "token_contract": _working_memory_token_contract(assistant_opening_ids),
    }


def _arrow_schema(pa: Any) -> Any:
    round_type = pa.struct(
        [
            ("round_index", pa.int32()),
            ("assistant_message_index", pa.int32()),
            ("tool_message_index", pa.int32()),
            ("target_assistant_message_index", pa.int32()),
            ("history_suffix_array_index", pa.int32()),
            ("history_suffix_token_length", pa.int64()),
            ("target_array_index", pa.int32()),
            ("target_token_length", pa.int64()),
        ]
    )
    raw_lengths_type = pa.struct(
        [
            ("raw_prefix_tokens", pa.int64()),
            ("raw_prefix_characters", pa.int64()),
            ("history_suffix_tokens", pa.list_(pa.int64())),
            ("history_suffix_characters", pa.list_(pa.int64())),
            ("target_tokens", pa.list_(pa.int64())),
            ("target_characters", pa.list_(pa.int64())),
            ("assistant_content_characters", pa.list_(pa.int64())),
            ("released_tool_content_characters", pa.list_(pa.int64())),
            ("canonical_tool_content_characters", pa.list_(pa.int64())),
        ]
    )
    token_contract_type = pa.struct(
        [
            ("name", pa.string()),
            ("generation_boundary_text", pa.string()),
            ("generation_boundary_ids", pa.list_(pa.int64())),
            ("raw_prefix", pa.string()),
            ("history_suffix", pa.string()),
            ("recurrence_conditioning", pa.string()),
            ("target", pa.string()),
            ("tool_canonicalization", pa.string()),
            ("compression", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("format_version", pa.int32()),
            ("schema", pa.string()),
            ("episode_id", pa.string()),
            ("split", pa.string()),
            ("round_count", pa.int32()),
            ("ordered_rounds", pa.list_(round_type)),
            ("raw_prefix_input_ids", pa.list_(pa.int64())),
            ("history_suffix_input_ids", pa.list_(pa.list_(pa.int64()))),
            ("target_ids", pa.list_(pa.list_(pa.int64()))),
            ("raw_lengths", raw_lengths_type),
            ("chat_template_sha256", pa.string()),
            ("tokenizer_name", pa.string()),
            ("token_contract", token_contract_type),
        ]
    )


def _working_memory_arrow_schema(pa: Any) -> Any:
    round_type = pa.struct(
        [
            ("round_index", pa.int32()),
            ("assistant_message_index", pa.int32()),
            ("tool_message_index", pa.int32()),
            ("target_assistant_message_index", pa.int32()),
            ("target_generation_round", pa.int32()),
            ("recursive_semantic_array_index", pa.int32()),
            ("recursive_semantic_token_length", pa.int64()),
            ("assistant_semantic_tokens", pa.int64()),
            ("tool_semantic_start", pa.int64()),
            ("tool_semantic_tokens", pa.int64()),
            ("memory_transition_semantic_array_index", pa.int32()),
            ("memory_summarizes_through_round", pa.int32()),
            ("raw_working_memory_array_index", pa.int32()),
            ("raw_working_memory_token_length", pa.int64()),
            ("target_array_index", pa.int32()),
            ("target_token_length", pa.int64()),
        ]
    )
    raw_lengths_type = pa.struct(
        [
            ("raw_prefix_tokens", pa.int64()),
            ("raw_prefix_characters", pa.int64()),
            ("recursive_semantic_tokens", pa.list_(pa.int64())),
            ("recursive_semantic_characters", pa.list_(pa.int64())),
            ("assistant_semantic_tokens", pa.list_(pa.int64())),
            ("tool_semantic_tokens", pa.list_(pa.int64())),
            ("raw_working_memory_tokens", pa.list_(pa.int64())),
            ("raw_working_memory_characters", pa.list_(pa.int64())),
            ("target_tokens", pa.list_(pa.int64())),
            ("target_characters", pa.list_(pa.int64())),
            ("assistant_content_characters", pa.list_(pa.int64())),
            ("released_tool_content_characters", pa.list_(pa.int64())),
            ("canonical_tool_content_characters", pa.list_(pa.int64())),
        ]
    )
    token_contract_type = pa.struct(
        [
            ("name", pa.string()),
            ("assistant_opening_text", pa.string()),
            ("assistant_opening_ids", pa.list_(pa.int64())),
            ("raw_prefix", pa.string()),
            ("recursive_semantic", pa.string()),
            ("raw_working_memory", pa.string()),
            ("one_round_lag", pa.string()),
            ("recovery", pa.string()),
            ("recurrence_conditioning", pa.string()),
            ("target", pa.string()),
            ("tool_canonicalization", pa.string()),
            ("compression", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("format_version", pa.int32()),
            ("schema", pa.string()),
            ("episode_id", pa.string()),
            ("split", pa.string()),
            ("round_count", pa.int32()),
            ("ordered_rounds", pa.list_(round_type)),
            ("raw_prefix_input_ids", pa.list_(pa.int64())),
            ("assistant_opening_ids", pa.list_(pa.int64())),
            ("recursive_semantic_input_ids", pa.list_(pa.list_(pa.int64()))),
            ("raw_working_memory_input_ids", pa.list_(pa.list_(pa.int64()))),
            ("target_ids", pa.list_(pa.list_(pa.int64()))),
            ("raw_lengths", raw_lengths_type),
            ("chat_template_sha256", pa.string()),
            ("tokenizer_name", pa.string()),
            ("token_contract", token_contract_type),
        ]
    )


def materialize_grepseek_recursive(
    parquet_path: str | Path,
    output_dir: str | Path,
    tokenizer: Any,
    *,
    config: Optional[RecursivePreparationConfig] = None,
    tokenizer_name: Optional[str] = None,
    shard_size: int = 1024,
    batch_size: int = 256,
    max_episodes: Optional[int] = None,
    compression: str = "zstd",
    overwrite: bool = False,
    expected_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Write one recursive-v2 parquet row per raw GrepSeek episode."""

    if shard_size <= 0 or batch_size <= 0:
        raise ValueError("shard_size and batch_size must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive or None")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to materialize recursive GrepSeek data"
        ) from exc

    cfg = config or RecursivePreparationConfig()
    source_path = Path(parquet_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"input parquet does not exist: {source_path}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    known_outputs = sorted(destination.glob("train-*.parquet")) + sorted(
        destination.glob("validation-*.parquet")
    )
    if manifest_path.exists():
        known_outputs.append(manifest_path)
    if known_outputs and not overwrite:
        raise FileExistsError(
            f"prepared outputs already exist in {destination}; use overwrite=True"
        )
    if overwrite:
        for path in known_outputs:
            path.unlink()

    schema = _arrow_schema(pa)
    buffers: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": []}
    counts = {"train": 0, "validation": 0}
    shard_indices = {"train": 0, "validation": 0}
    files: Dict[str, List[str]] = {"train": [], "validation": []}

    def flush(split: str) -> None:
        records = buffers[split]
        if not records:
            return
        filename = f"{split}-{shard_indices[split]:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(records, schema=schema),
            destination / filename,
            compression=compression,
        )
        files[split].append(filename)
        shard_indices[split] += 1
        buffers[split] = []

    stats = {
        "episodes": 0,
        "rounds": 0,
        "raw_prefix_tokens": 0,
        "history_suffix_tokens": 0,
        "target_tokens": 0,
        "released_tool_content_characters": 0,
        "canonical_tool_content_characters": 0,
    }
    template_hash: Optional[str] = None
    for row in iter_grepseek_rows(source_path, batch_size=batch_size):
        record = build_grepseek_recursive_episode(
            row,
            tokenizer,
            config=cfg,
            tokenizer_name=tokenizer_name,
        )
        if template_hash is None:
            template_hash = str(record["chat_template_sha256"])
        elif record["chat_template_sha256"] != template_hash:
            raise ValueError("chat template hash changed during materialization")
        split = str(record["split"])
        buffers[split].append(record)
        counts[split] += 1
        stats["episodes"] += 1
        stats["rounds"] += int(record["round_count"])
        lengths = record["raw_lengths"]
        stats["raw_prefix_tokens"] += int(lengths["raw_prefix_tokens"])
        stats["history_suffix_tokens"] += sum(
            int(value) for value in lengths["history_suffix_tokens"]
        )
        stats["target_tokens"] += sum(
            int(value) for value in lengths["target_tokens"]
        )
        stats["released_tool_content_characters"] += sum(
            int(value) for value in lengths["released_tool_content_characters"]
        )
        stats["canonical_tool_content_characters"] += sum(
            int(value) for value in lengths["canonical_tool_content_characters"]
        )
        if len(buffers[split]) >= shard_size:
            flush(split)
        if max_episodes is not None and stats["episodes"] >= max_episodes:
            break
    flush("train")
    flush("validation")

    if expected_episodes is not None and stats["episodes"] != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes, materialized "
            f"{stats['episodes']}"
        )
    resolved_tokenizer_name = str(
        tokenizer_name
        or getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    )
    manifest: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "schema": SCHEMA_NAME,
        "dataset": "alireza7/GrepSeek-ColdStart-SFT-10k",
        "source_parquet": str(source_path.resolve()),
        "tokenizer_name": resolved_tokenizer_name,
        "chat_template_sha256": template_hash,
        "config": asdict(cfg),
        "counts": counts,
        "stats": stats,
        "files": files,
        "contracts": _token_contract(GENERATION_BOUNDARY_IDS),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize_grepseek_recursive_working_memory(
    parquet_path: str | Path,
    output_dir: str | Path,
    tokenizer: Any,
    *,
    config: Optional[RecursivePreparationConfig] = None,
    tokenizer_name: Optional[str] = None,
    shard_size: int = 1024,
    batch_size: int = 256,
    max_episodes: Optional[int] = None,
    compression: str = "zstd",
    overwrite: bool = False,
    expected_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """Write one working-memory-v3 parquet row per raw GrepSeek episode."""

    if shard_size <= 0 or batch_size <= 0:
        raise ValueError("shard_size and batch_size must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive or None")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required to materialize recursive GrepSeek data"
        ) from exc

    cfg = config or RecursivePreparationConfig()
    source_path = Path(parquet_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"input parquet does not exist: {source_path}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    known_outputs = sorted(destination.glob("train-*.parquet")) + sorted(
        destination.glob("validation-*.parquet")
    )
    if manifest_path.exists():
        known_outputs.append(manifest_path)
    if known_outputs and not overwrite:
        raise FileExistsError(
            f"prepared outputs already exist in {destination}; use overwrite=True"
        )
    if overwrite:
        for path in known_outputs:
            path.unlink()

    schema = _working_memory_arrow_schema(pa)
    buffers: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": []}
    counts = {"train": 0, "validation": 0}
    shard_indices = {"train": 0, "validation": 0}
    files: Dict[str, List[str]] = {"train": [], "validation": []}

    def flush(split: str) -> None:
        records = buffers[split]
        if not records:
            return
        filename = f"{split}-{shard_indices[split]:05d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(records, schema=schema),
            destination / filename,
            compression=compression,
        )
        files[split].append(filename)
        shard_indices[split] += 1
        buffers[split] = []

    stats = {
        "episodes": 0,
        "rounds": 0,
        "raw_prefix_tokens": 0,
        "recursive_semantic_tokens": 0,
        "raw_working_memory_tokens": 0,
        "target_tokens": 0,
        "released_tool_content_characters": 0,
        "canonical_tool_content_characters": 0,
    }
    template_hash: Optional[str] = None
    for row in iter_grepseek_rows(source_path, batch_size=batch_size):
        record = build_grepseek_recursive_working_memory_episode(
            row,
            tokenizer,
            config=cfg,
            tokenizer_name=tokenizer_name,
        )
        if template_hash is None:
            template_hash = str(record["chat_template_sha256"])
        elif record["chat_template_sha256"] != template_hash:
            raise ValueError("chat template hash changed during materialization")
        split = str(record["split"])
        buffers[split].append(record)
        counts[split] += 1
        stats["episodes"] += 1
        stats["rounds"] += int(record["round_count"])
        lengths = record["raw_lengths"]
        stats["raw_prefix_tokens"] += int(lengths["raw_prefix_tokens"])
        stats["recursive_semantic_tokens"] += sum(
            int(value) for value in lengths["recursive_semantic_tokens"]
        )
        stats["raw_working_memory_tokens"] += sum(
            int(value) for value in lengths["raw_working_memory_tokens"]
        )
        stats["target_tokens"] += sum(
            int(value) for value in lengths["target_tokens"]
        )
        stats["released_tool_content_characters"] += sum(
            int(value) for value in lengths["released_tool_content_characters"]
        )
        stats["canonical_tool_content_characters"] += sum(
            int(value) for value in lengths["canonical_tool_content_characters"]
        )
        if len(buffers[split]) >= shard_size:
            flush(split)
        if max_episodes is not None and stats["episodes"] >= max_episodes:
            break
    flush("train")
    flush("validation")

    if expected_episodes is not None and stats["episodes"] != expected_episodes:
        raise ValueError(
            f"expected {expected_episodes} episodes, materialized "
            f"{stats['episodes']}"
        )
    resolved_tokenizer_name = str(
        tokenizer_name
        or getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    )
    manifest: Dict[str, Any] = {
        "format_version": WORKING_MEMORY_FORMAT_VERSION,
        "schema": WORKING_MEMORY_SCHEMA_NAME,
        "dataset": "alireza7/GrepSeek-ColdStart-SFT-10k",
        "source_parquet": str(source_path.resolve()),
        "tokenizer_name": resolved_tokenizer_name,
        "chat_template_sha256": template_hash,
        "config": asdict(cfg),
        "counts": counts,
        "stats": stats,
        "files": files,
        "contracts": _working_memory_token_contract(ASSISTANT_OPENING_IDS),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


class RecursivePreparedDataset:
    """Lazy row-group reader for recursive-v2 episode parquet."""

    prepared_format_version = FORMAT_VERSION
    prepared_schema_name = SCHEMA_NAME

    def __init__(
        self,
        path: str | Path,
        split: Optional[str] = None,
        *,
        row_group_cache_size: int = 2,
    ) -> None:
        if split not in {None, "train", "validation"}:
            raise ValueError("split must be train, validation, or None")
        if row_group_cache_size <= 0:
            raise ValueError("row_group_cache_size must be positive")
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyarrow is required to load recursive GrepSeek data"
            ) from exc

        self._pq = pq
        self.path = Path(path)
        self.split = split
        self.manifest: Optional[Dict[str, Any]] = None
        if self.path.is_dir():
            manifest_path = self.path / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"prepared manifest does not exist: {manifest_path}"
                )
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                self.manifest.get("format_version") != self.prepared_format_version
                or self.manifest.get("schema") != self.prepared_schema_name
            ):
                raise ValueError("unsupported recursive GrepSeek prepared format")
            splits = [split] if split is not None else ["train", "validation"]
            files = [
                self.path / filename
                for split_name in splits
                for filename in self.manifest["files"].get(split_name, [])
            ]
        elif self.path.is_file():
            if split is not None:
                raise ValueError(
                    "split filtering requires a prepared directory with manifest"
                )
            files = [self.path]
        else:
            raise FileNotFoundError(f"prepared data does not exist: {self.path}")
        if not files:
            raise ValueError(f"prepared data contains no files for split={split!r}")

        self.files = [Path(file) for file in files]
        self._row_groups: List[Tuple[Path, int]] = []
        self._cumulative_rows: List[int] = []
        total = 0
        for file in self.files:
            parquet_file = pq.ParquetFile(file)
            for row_group in range(parquet_file.num_row_groups):
                total += parquet_file.metadata.row_group(row_group).num_rows
                self._row_groups.append((file, row_group))
                self._cumulative_rows.append(total)
        self._length = total
        self._cache_size = int(row_group_cache_size)
        self._cache: OrderedDict[Tuple[str, int], List[Dict[str, Any]]] = OrderedDict()
        self._round_counts: Optional[Tuple[int, ...]] = None

    def __len__(self) -> int:
        return self._length

    @property
    def round_counts(self) -> Tuple[int, ...]:
        """Return cached episode round counts without decoding nested ID lists."""

        if self._round_counts is not None:
            return self._round_counts
        values: List[int] = []
        for file, row_group in self._row_groups:
            table = self._pq.ParquetFile(file).read_row_group(
                row_group, columns=["round_count"]
            )
            values.extend(int(value) for value in table["round_count"].to_pylist())
        if len(values) != self._length:
            raise ValueError(
                "round_count column length disagrees with prepared dataset length"
            )
        if any(value <= 0 for value in values):
            raise ValueError("prepared episode round_count must be positive")
        self._round_counts = tuple(values)
        return self._round_counts

    def _read_group(self, file: Path, row_group: int) -> List[Dict[str, Any]]:
        key = (str(file), row_group)
        if key in self._cache:
            rows = self._cache.pop(key)
            self._cache[key] = rows
            return rows
        rows = self._pq.ParquetFile(file).read_row_group(row_group).to_pylist()
        self._cache[key] = rows
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return rows

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        group_index = bisect.bisect_right(self._cumulative_rows, index)
        group_start = 0 if group_index == 0 else self._cumulative_rows[group_index - 1]
        file, row_group = self._row_groups[group_index]
        return self._read_group(file, row_group)[index - group_start]


class RecursiveWorkingMemoryPreparedDataset(RecursivePreparedDataset):
    """Lazy row-group reader for recursive working-memory-v3 parquet."""

    prepared_format_version = WORKING_MEMORY_FORMAT_VERSION
    prepared_schema_name = WORKING_MEMORY_SCHEMA_NAME


__all__ = [
    "ASSISTANT_OPENING_IDS",
    "ASSISTANT_OPENING_TEXT",
    "FORMAT_VERSION",
    "GENERATION_BOUNDARY_IDS",
    "GENERATION_BOUNDARY_TEXT",
    "RecursivePreparationConfig",
    "RecursivePreparedDataset",
    "RecursiveWorkingMemoryPreparedDataset",
    "SCHEMA_NAME",
    "USER_TURN_END_TEXT",
    "WORKING_MEMORY_FORMAT_VERSION",
    "WORKING_MEMORY_SCHEMA_NAME",
    "build_grepseek_recursive_episode",
    "build_grepseek_recursive_working_memory_episode",
    "materialize_grepseek_recursive",
    "materialize_grepseek_recursive_working_memory",
]
