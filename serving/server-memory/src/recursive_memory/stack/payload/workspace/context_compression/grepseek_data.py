"""GrepSeek trajectory materialization for query-conditioned compression.

Only tool-response ``stdout`` is compressible evidence.  Every other token in
the trajectory remains discrete, including the query, previous assistant
reasoning/tool calls, tool JSON keys, command, and execution status.  The
duplicated ``information_lines`` field is validated against ``stdout`` and is
then replaced by ``[]`` in the student history.

The prepared representation is deliberately model-agnostic.  For a sample
with ``S`` tool responses, ``discrete_segments_input_ids`` has ``S + 1``
elements and describes this exact interleave::

    discrete[0], MEMORY(source[0]), ..., MEMORY(source[S-1]), discrete[S]

The collator expands each MEMORY segment to ``ceil(evidence_tokens / ratio)``
valid placeholder positions.  It also flattens query-conditioned source
encoder sequences in the same sample/segment order and returns the metadata
needed to scatter compressor outputs back into the left-padded student input.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


TOOL_RESPONSE_KEYS: Tuple[str, ...] = (
    "command",
    "stdout",
    "stderr",
    "exit_code",
    "timed_out",
    "truncated",
    "information_lines",
)
DEFAULT_VALIDATION_FRACTION = 0.02
DEFAULT_COMPRESSION_RATIO = 16
FORMAT_VERSION = 1


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _stable_split_fraction(episode_id: str, seed: int) -> float:
    digest = hashlib.sha256(
        f"grepseek-episode-split\0{int(seed)}\0{episode_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_episode_split(
    episode_id: str,
    *,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    seed: int = 42,
) -> str:
    """Assign every sample from an episode to one deterministic split."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    return (
        "validation"
        if _stable_split_fraction(str(episode_id), int(seed)) < validation_fraction
        else "train"
    )


def parse_tool_response(
    content: str,
    *,
    episode_id: str = "<unknown>",
    message_index: int = -1,
) -> Dict[str, Any]:
    """Strictly parse the seven-key GrepSeek tool-response JSON contract."""

    where = f"episode {episode_id!r} message {message_index}"
    if not isinstance(content, str):
        raise ValueError(f"{where}: tool content must be a JSON string")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: invalid tool-response JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{where}: tool response must decode to an object")
    if set(payload) != set(TOOL_RESPONSE_KEYS) or len(payload) != len(
        TOOL_RESPONSE_KEYS
    ):
        missing = sorted(set(TOOL_RESPONSE_KEYS) - set(payload))
        extra = sorted(set(payload) - set(TOOL_RESPONSE_KEYS))
        raise ValueError(
            f"{where}: tool response must have exactly the seven expected keys; "
            f"missing={missing}, extra={extra}"
        )
    if not isinstance(payload["command"], str):
        raise ValueError(f"{where}: command must be a string")
    if not isinstance(payload["stdout"], str):
        raise ValueError(f"{where}: stdout must be a string")
    if not isinstance(payload["stderr"], str):
        raise ValueError(f"{where}: stderr must be a string")
    if isinstance(payload["exit_code"], bool) or not isinstance(
        payload["exit_code"], int
    ):
        raise ValueError(f"{where}: exit_code must be an integer")
    if not isinstance(payload["timed_out"], bool):
        raise ValueError(f"{where}: timed_out must be boolean")
    if not isinstance(payload["truncated"], bool):
        raise ValueError(f"{where}: truncated must be boolean")
    information_lines = payload["information_lines"]
    if not isinstance(information_lines, list) or not all(
        isinstance(line, str) for line in information_lines
    ):
        raise ValueError(f"{where}: information_lines must be a list of strings")
    expected_lines = [line for line in payload["stdout"].splitlines() if line.strip()]
    if information_lines != expected_lines:
        raise ValueError(
            f"{where}: information_lines must exactly equal the nonempty stdout lines"
        )
    return {key: payload[key] for key in TOOL_RESPONSE_KEYS}


def _canonical_tool_content(payload: Mapping[str, Any], stdout: str) -> str:
    canonical = {key: payload[key] for key in TOOL_RESPONSE_KEYS}
    canonical["stdout"] = str(stdout)
    # This field duplicates stdout in every released ColdStart trajectory.  It
    # must never leak a second, uncompressed copy into the student context.
    canonical["information_lines"] = []
    return json.dumps(canonical, ensure_ascii=False)


def _coerce_template_ids(value: Any) -> List[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("chat-template result has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template unexpectedly returned a batch")
        value = value[0]
    return [int(token_id) for token_id in value]


def _apply_chat_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> List[int]:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise ValueError("GrepSeek preparation requires tokenizer.apply_chat_template")
    return _coerce_template_ids(
        apply_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )


def _assistant_continuation_ids(
    tokenizer: Any,
    history: Sequence[Mapping[str, str]],
    target_message: Mapping[str, str],
) -> Tuple[List[int], List[int], List[int]]:
    """Return native ``(prompt, full, continuation)`` with overlap handling.

    Qwen3.5's generation prompt ends in ``<think>\n``.  Most stored assistant
    messages start with ``<think>\n...`` and therefore have the prompt as an
    exact prefix.  A small number start ``<think>\n\n...``; the full native
    template then tokenizes the boundary newline jointly/differently.  Taking
    the longest common prefix and supervising the complete differing suffix
    preserves every target token without pretending an inexact token prefix.
    """

    prompt_ids = _apply_chat_ids(tokenizer, history, add_generation_prompt=True)
    full_ids = _apply_chat_ids(
        tokenizer,
        list(history) + [dict(target_message)],
        add_generation_prompt=False,
    )
    common = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids):
        if prompt_token != full_token:
            break
        common += 1
    if common == 0:
        raise ValueError("native full chat and generation prompt have no common prefix")
    # Use the exact full-chat prefix so concatenating prompt/target reconstructs
    # the native full serialization bit-for-bit, including boundary BPE merges.
    exact_prompt_ids = full_ids[:common]
    continuation_ids = full_ids[common:]
    if not continuation_ids:
        raise ValueError("native assistant continuation is empty")
    return exact_prompt_ids, full_ids, continuation_ids


def _render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str):
        raise ValueError(
            "tokenizer.apply_chat_template(tokenize=False) returned non-text"
        )
    return rendered


def _encode(tokenizer: Any, text: str) -> List[int]:
    value = tokenizer.encode(str(text), add_special_tokens=False)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(token_id) for token_id in value]


def _resolve_pad_token_id(tokenizer: Any) -> int:
    value = getattr(tokenizer, "pad_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    return int(value)


def _choose_soft_marker(tokenizer: Any) -> str:
    """Choose one existing special token that is exactly one tokenizer token."""

    candidates: List[str] = []
    for attribute in ("image_token", "video_token", "audio_token"):
        token = getattr(tokenizer, attribute, None)
        if isinstance(token, str):
            candidates.append(token)
    candidates.extend(
        str(token)
        for token in _as_list(getattr(tokenizer, "additional_special_tokens", []))
        if isinstance(token, str)
    )
    for attribute in ("eos_token", "bos_token", "pad_token"):
        token = getattr(tokenizer, attribute, None)
        if isinstance(token, str):
            candidates.append(token)
    for marker in dict.fromkeys(candidates):
        # Role delimiters occur naturally throughout the rendered template and
        # would make accidental marker collisions needlessly likely.
        if marker in {"<|im_start|>", "<|im_end|>"}:
            continue
        if len(_encode(tokenizer, marker)) == 1:
            return marker
    raise ValueError(
        "target tokenizer has no one-token special marker suitable for exact "
        "soft-memory span extraction"
    )


def _rendered_ids_and_offsets(
    tokenizer: Any,
    rendered: str,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError) as exc:
        raise ValueError(
            "GrepSeek preparation requires a fast tokenizer with offset mappings"
        ) from exc
    if isinstance(encoded, Mapping):
        ids_value = encoded.get("input_ids")
        offsets_value = encoded.get("offset_mapping")
    else:
        ids_value = getattr(encoded, "input_ids", None)
        offsets_value = getattr(encoded, "offset_mapping", None)
    if ids_value is None or offsets_value is None:
        raise ValueError("tokenizer did not return input_ids and offset_mapping")
    ids = _coerce_template_ids(ids_value)
    if hasattr(offsets_value, "tolist"):
        offsets_value = offsets_value.tolist()
    if (
        offsets_value
        and isinstance(offsets_value[0], list)
        and offsets_value[0]
        and isinstance(offsets_value[0][0], (list, tuple))
    ):
        if len(offsets_value) != 1:
            raise ValueError("offset tokenizer unexpectedly returned a batch")
        offsets_value = offsets_value[0]
    offsets = [(int(pair[0]), int(pair[1])) for pair in offsets_value]
    if len(ids) != len(offsets):
        raise ValueError("token ids and offset mappings have different lengths")
    return ids, offsets


def _marker_char_spans(rendered: str, marker: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    while True:
        start = rendered.find(marker, cursor)
        if start < 0:
            break
        end = start + len(marker)
        spans.append((start, end))
        cursor = end
    return spans


def _split_chat_at_markers(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    marker: str,
    expected_markers: int,
    add_generation_prompt: bool,
    target_message: Optional[Mapping[str, str]] = None,
) -> List[List[int]]:
    """Native-chat tokenize and remove only exact special-marker tokens."""

    if target_message is None:
        rendered_messages = list(messages)
        rendered = _render_chat(
            tokenizer, rendered_messages, add_generation_prompt=add_generation_prompt
        )
        native_ids = _apply_chat_ids(
            tokenizer, rendered_messages, add_generation_prompt=add_generation_prompt
        )
        prefix_width: Optional[int] = None
    else:
        if not add_generation_prompt:
            raise ValueError("target_message boundary mode requires generation prompt")
        rendered_messages = list(messages) + [dict(target_message)]
        rendered = _render_chat(
            tokenizer, rendered_messages, add_generation_prompt=False
        )
        exact_prompt, native_ids, _ = _assistant_continuation_ids(
            tokenizer, messages, target_message
        )
        prefix_width = len(exact_prompt)
    ids, offsets = _rendered_ids_and_offsets(tokenizer, rendered)
    if ids != native_ids:
        raise ValueError(
            "tokenizing rendered native chat does not reproduce apply_chat_template ids"
        )
    char_spans = _marker_char_spans(rendered, marker)
    if len(char_spans) != expected_markers:
        raise ValueError(
            f"expected {expected_markers} soft markers in rendered chat, found "
            f"{len(char_spans)}"
        )
    marker_token_id = _encode(tokenizer, marker)[0]
    marker_indices: List[int] = []
    for span in char_spans:
        exact = [index for index, offset in enumerate(offsets) if offset == span]
        if len(exact) != 1 or ids[exact[0]] != marker_token_id:
            raise ValueError(
                "soft marker does not align to exactly one special token; refusing "
                "an inexact evidence boundary"
            )
        marker_indices.append(exact[0])
    if marker_indices != sorted(set(marker_indices)):
        raise ValueError("soft marker token positions are not strictly increasing")

    if prefix_width is not None:
        if any(index >= prefix_width for index in marker_indices):
            raise ValueError("soft marker unexpectedly lies inside target continuation")
        ids = ids[:prefix_width]
    segments: List[List[int]] = []
    start = 0
    for marker_index in marker_indices:
        segments.append(ids[start:marker_index])
        start = marker_index + 1
    segments.append(ids[start:])
    return segments


@dataclass(frozen=True)
class GrepSeekPreparationConfig:
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    seed: int = 42
    max_source_tokens: Optional[int] = None
    max_continuation_tokens: Optional[int] = None
    max_encoder_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        for name in (
            "max_source_tokens",
            "max_continuation_tokens",
            "max_encoder_tokens",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ValueError(f"{name} must be positive or None")


def _validate_messages(row: Mapping[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    if set(row) != {"id", "messages"}:
        raise ValueError(
            f"raw GrepSeek row must have exactly id/messages, got {sorted(row)}"
        )
    episode_id = row.get("id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("GrepSeek row id must be a nonempty string")
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(f"episode {episode_id!r}: messages must be a nonempty list")
    messages: List[Dict[str, str]] = []
    allowed_roles = {"system", "user", "assistant", "tool"}
    for index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(
                f"episode {episode_id!r} message {index}: expected role/content"
            )
        role, content = message.get("role"), message.get("content")
        if role not in allowed_roles or not isinstance(content, str):
            raise ValueError(
                f"episode {episode_id!r} message {index}: invalid role/content"
            )
        messages.append({"role": str(role), "content": content})
    if messages[0]["role"] != "system":
        raise ValueError(f"episode {episode_id!r}: first message must be system")
    if len(messages) < 2 or messages[1]["role"] != "user":
        raise ValueError(f"episode {episode_id!r}: second message must be user")
    return episode_id, messages


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))


def _source_prompt_and_spans(
    tokenizer: Any,
    history: Sequence[Mapping[str, str]],
    parsed_tools: Mapping[int, Mapping[str, Any]],
    history_tool_indices: Sequence[int],
    source_texts: Sequence[str],
) -> Tuple[List[int], List[List[int]], List[List[int]]]:
    """Render canonical raw history once and locate every stdout exactly.

    ``information_lines`` is empty in both renderings.  Unique textual markers
    are used only to prove the character boundaries; they never enter the
    returned source prompt.  Token spans include a boundary token when the BPE
    tokenizer merges it with evidence, which is the only exact representation
    possible without changing the model's native chat serialization.
    """

    if len(history_tool_indices) != len(source_texts):
        raise ValueError("tool/source arity mismatch while rendering source prompt")
    marker_by_index: Dict[int, str] = {}
    for ordinal, tool_index in enumerate(history_tool_indices):
        marker = f"DFLASH_EVIDENCE_BOUNDARY_{ordinal:06d}_DO_NOT_TOKENIZE"
        if any(marker in message["content"] for message in history):
            raise ValueError("generated source marker collides with trajectory text")
        marker_by_index[tool_index] = marker

    marked_history: List[Dict[str, str]] = []
    raw_history: List[Dict[str, str]] = []
    source_by_index = dict(zip(history_tool_indices, source_texts))
    for index, message in enumerate(history):
        if message["role"] != "tool":
            marked_history.append(dict(message))
            raw_history.append(dict(message))
            continue
        marked_history.append(
            {
                "role": "tool",
                "content": _canonical_tool_content(
                    parsed_tools[index], marker_by_index[index]
                ),
            }
        )
        raw_history.append(
            {
                "role": "tool",
                "content": _canonical_tool_content(
                    parsed_tools[index], source_by_index[index]
                ),
            }
        )

    marked_render = _render_chat(tokenizer, marked_history, add_generation_prompt=True)
    raw_render = _render_chat(tokenizer, raw_history, add_generation_prompt=True)
    static_parts: List[str] = []
    cursor = 0
    for tool_index in history_tool_indices:
        marker = marker_by_index[tool_index]
        start = marked_render.find(marker, cursor)
        if start < 0 or marked_render.find(marker, start + len(marker)) >= 0:
            raise ValueError("source boundary marker was not rendered exactly once")
        static_parts.append(marked_render[cursor:start])
        cursor = start + len(marker)
    static_parts.append(marked_render[cursor:])

    stdout_char_spans: List[List[int]] = []
    rebuilt_parts: List[str] = []
    raw_cursor = 0
    for ordinal, source_text in enumerate(source_texts):
        static = static_parts[ordinal]
        escaped_stdout = json.dumps(str(source_text), ensure_ascii=False)[1:-1]
        rebuilt_parts.extend((static, escaped_stdout))
        raw_cursor += len(static)
        stdout_char_spans.append([raw_cursor, raw_cursor + len(escaped_stdout)])
        raw_cursor += len(escaped_stdout)
    rebuilt_parts.append(static_parts[-1])
    if "".join(rebuilt_parts) != raw_render:
        raise ValueError(
            "canonical raw source rendering is not an exact marker substitution"
        )

    source_prompt_ids, offsets = _rendered_ids_and_offsets(tokenizer, raw_render)
    native_ids = _apply_chat_ids(tokenizer, raw_history, add_generation_prompt=True)
    if source_prompt_ids != native_ids:
        raise ValueError("canonical raw source prompt differs from native chat ids")
    evidence_spans: List[List[int]] = []
    for char_start, char_end in stdout_char_spans:
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > char_start and token_start < char_end
        ]
        if not overlapping:
            raise ValueError(
                "stdout has no overlapping token in canonical source prompt"
            )
        start, end = overlapping[0], overlapping[-1] + 1
        if overlapping != list(range(start, end)):
            raise ValueError("stdout token span is unexpectedly non-contiguous")
        evidence_spans.append([start, end])
    return source_prompt_ids, evidence_spans, stdout_char_spans


def _left_crop_source_prompt(
    input_ids: Sequence[int],
    evidence_spans: Sequence[Sequence[int]],
    max_encoder_tokens: Optional[int],
) -> Tuple[List[int], List[List[int]], int]:
    """Optionally crop only the prefix before the first evidence span."""

    ids = [int(x) for x in input_ids]
    spans = [[int(span[0]), int(span[1])] for span in evidence_spans]
    if max_encoder_tokens is None or len(ids) <= max_encoder_tokens:
        return ids, spans, 0
    drop = len(ids) - max_encoder_tokens
    first_evidence_start = min(span[0] for span in spans)
    if drop > first_evidence_start:
        raise ValueError(
            "max_encoder_tokens would remove source evidence; increase it or use "
            "max_source_tokens explicitly"
        )
    return ids[drop:], [[start - drop, end - drop] for start, end in spans], drop


def build_grepseek_examples(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    config: Optional[GrepSeekPreparationConfig] = None,
    tokenizer_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build every post-tool next-assistant training sample in one episode."""

    cfg = config or GrepSeekPreparationConfig()
    episode_id, messages = _validate_messages(row)
    marker = _choose_soft_marker(tokenizer)
    parsed_tools: Dict[int, Dict[str, Any]] = {}
    for index, message in enumerate(messages):
        if message["role"] == "tool":
            parsed_tools[index] = parse_tool_response(
                message["content"],
                episode_id=episode_id,
                message_index=index,
            )
            if index + 1 >= len(messages) or messages[index + 1]["role"] != "assistant":
                raise ValueError(
                    f"episode {episode_id!r}: tool message {index} is not followed by "
                    "an assistant continuation"
                )

    split = assign_episode_split(
        episode_id,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
    )
    records: List[Dict[str, Any]] = []
    for assistant_index, target_message in enumerate(messages):
        if target_message["role"] != "assistant" or assistant_index == 0:
            continue
        if messages[assistant_index - 1]["role"] != "tool":
            continue
        history = messages[:assistant_index]
        history_tool_indices = [
            index
            for index in range(assistant_index)
            if messages[index]["role"] == "tool"
        ]
        if not history_tool_indices:
            raise AssertionError("post-tool assistant sample has no tool history")

        teacher_prompt_ids, teacher_full_ids, original_target_ids = (
            _assistant_continuation_ids(
                tokenizer,
                history,
                target_message,
            )
        )
        target_ids = original_target_ids
        target_truncated = False
        if (
            cfg.max_continuation_tokens is not None
            and len(target_ids) > cfg.max_continuation_tokens
        ):
            target_ids = target_ids[: cfg.max_continuation_tokens]
            target_truncated = True

        student_history: List[Dict[str, str]] = []
        for index, message in enumerate(history):
            if message["role"] != "tool":
                student_history.append(dict(message))
                continue
            student_history.append(
                {
                    "role": "tool",
                    "content": _canonical_tool_content(parsed_tools[index], marker),
                }
            )
        discrete_segments = _split_chat_at_markers(
            tokenizer,
            student_history,
            marker=marker,
            expected_markers=len(history_tool_indices),
            add_generation_prompt=True,
            target_message=target_message,
        )
        _, _, student_target_ids = _assistant_continuation_ids(
            tokenizer, student_history, target_message
        )
        if student_target_ids != original_target_ids:
            raise ValueError(
                f"episode {episode_id!r} assistant {assistant_index}: compressing "
                "history changed native target-boundary tokenization"
            )

        sources: List[List[int]] = []
        source_texts: List[str] = []
        original_source_lengths: List[int] = []
        source_truncated: List[bool] = []
        stdout_sha256: List[str] = []

        for tool_index in history_tool_indices:
            payload = parsed_tools[tool_index]
            original_source = _encode(tokenizer, payload["stdout"])
            if not original_source:
                raise ValueError(
                    f"episode {episode_id!r} tool {tool_index}: stdout tokenizes empty"
                )
            source = original_source
            source_text = payload["stdout"]
            was_truncated = False
            if (
                cfg.max_source_tokens is not None
                and len(source) > cfg.max_source_tokens
            ):
                source = source[: cfg.max_source_tokens]
                source_text = _decode(tokenizer, source)
                source = _encode(tokenizer, source_text)
                if len(source) > cfg.max_source_tokens:
                    # Fast BPE tokenizers should round-trip exactly.  This
                    # closes the door on a custom tokenizer that expands on
                    # decode/encode and would silently violate the limit.
                    source = source[: cfg.max_source_tokens]
                    source_text = _decode(tokenizer, source)
                    source = _encode(tokenizer, source_text)
                if len(source) > cfg.max_source_tokens:
                    raise ValueError("tokenizer cannot stably truncate stdout text")
                was_truncated = True

            sources.append(source)
            source_texts.append(source_text)
            original_source_lengths.append(len(original_source))
            source_truncated.append(was_truncated)
            stdout_sha256.append(
                hashlib.sha256(payload["stdout"].encode("utf-8")).hexdigest()
            )

        source_prompt_ids, evidence_spans, evidence_char_spans = (
            _source_prompt_and_spans(
                tokenizer,
                history,
                parsed_tools,
                history_tool_indices,
                source_texts,
            )
        )
        source_prompt_ids, evidence_spans, encoder_history_dropped = (
            _left_crop_source_prompt(
                source_prompt_ids, evidence_spans, cfg.max_encoder_tokens
            )
        )

        if len(discrete_segments) != len(sources) + 1:
            raise AssertionError("student discrete/source segment arity mismatch")
        return_name = str(
            tokenizer_name
            or getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
        )
        records.append(
            {
                "id": f"{episode_id}:assistant-{assistant_index:03d}",
                "episode_id": episode_id,
                "assistant_message_index": assistant_index,
                "split": split,
                "num_source_segments": len(sources),
                "discrete_segments_input_ids": discrete_segments,
                "source_segments_input_ids": sources,
                "source_prompt_input_ids": source_prompt_ids,
                "source_prompt_evidence_spans": evidence_spans,
                "source_prompt_stdout_char_spans": evidence_char_spans,
                "source_original_token_lengths": original_source_lengths,
                "source_token_lengths": [len(source) for source in sources],
                "source_was_truncated": source_truncated,
                "source_stdout_sha256": stdout_sha256,
                "encoder_history_tokens_dropped": encoder_history_dropped,
                "encoder_evidence_tokens_dropped": 0,
                "teacher_prompt_input_ids": teacher_prompt_ids,
                "teacher_prompt_length": len(teacher_prompt_ids),
                "target_input_ids": target_ids,
                "target_original_length": len(original_target_ids),
                "target_was_truncated": target_truncated,
                "student_discrete_token_count": sum(
                    len(segment) for segment in discrete_segments
                ),
                "tokenizer_name": return_name,
                "construction_seed": cfg.seed,
            }
        )
    if len(records) != len(parsed_tools):
        raise ValueError(
            f"episode {episode_id!r}: found {len(parsed_tools)} tool responses but "
            f"constructed {len(records)} post-tool assistant samples"
        )
    return records


def iter_grepseek_rows(
    parquet_path: str | Path,
    *,
    batch_size: int = 256,
) -> Iterator[Dict[str, Any]]:
    """Stream the strict local ``id/messages`` ColdStart parquet schema."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to read GrepSeek parquet") from exc
    parquet_file = pq.ParquetFile(str(parquet_path))
    schema_names = set(parquet_file.schema_arrow.names)
    if schema_names != {"id", "messages"}:
        raise ValueError(
            f"raw GrepSeek parquet must have exactly id/messages, got {schema_names}"
        )
    for batch in parquet_file.iter_batches(
        batch_size=batch_size, columns=["id", "messages"]
    ):
        yield from batch.to_pylist()


def materialize_grepseek(
    parquet_path: str | Path,
    output_dir: str | Path,
    tokenizer: Any,
    *,
    config: Optional[GrepSeekPreparationConfig] = None,
    tokenizer_name: Optional[str] = None,
    shard_size: int = 1024,
    batch_size: int = 256,
    max_episodes: Optional[int] = None,
    compression: str = "zstd",
    overwrite: bool = False,
    expected_examples: Optional[int] = None,
) -> Dict[str, Any]:
    """Materialize deterministic split parquet shards and an audit manifest."""

    if shard_size <= 0 or batch_size <= 0:
        raise ValueError("shard_size and batch_size must be positive")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive or None")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required to materialize GrepSeek data") from exc

    cfg = config or GrepSeekPreparationConfig()
    source_path = Path(parquet_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"input parquet does not exist: {source_path}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    known_outputs = sorted(destination.glob("train-*.parquet")) + sorted(
        destination.glob("validation-*.parquet")
    )
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        known_outputs.append(manifest_path)
    if known_outputs and not overwrite:
        raise FileExistsError(
            f"prepared outputs already exist in {destination}; use overwrite=True"
        )
    if overwrite:
        for path in known_outputs:
            path.unlink()

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
            pa.Table.from_pylist(records),
            destination / filename,
            compression=compression,
        )
        files[split].append(filename)
        shard_indices[split] += 1
        buffers[split] = []

    stats: Dict[str, int] = {
        "episodes": 0,
        "tool_responses": 0,
        "examples": 0,
        "source_segment_occurrences": 0,
        "source_segments_truncated": 0,
        "source_tokens_dropped": 0,
        "encoder_history_tokens_dropped": 0,
        "encoder_evidence_tokens_dropped": 0,
        "continuations_truncated": 0,
        "continuation_tokens_dropped": 0,
    }
    for row in iter_grepseek_rows(source_path, batch_size=batch_size):
        records = build_grepseek_examples(
            row, tokenizer, config=cfg, tokenizer_name=tokenizer_name
        )
        stats["episodes"] += 1
        stats["tool_responses"] += len(records)
        for record in records:
            split = record["split"]
            buffers[split].append(record)
            counts[split] += 1
            stats["examples"] += 1
            stats["source_segment_occurrences"] += record["num_source_segments"]
            stats["source_segments_truncated"] += sum(
                bool(value) for value in record["source_was_truncated"]
            )
            stats["source_tokens_dropped"] += sum(
                original - kept
                for original, kept in zip(
                    record["source_original_token_lengths"],
                    record["source_token_lengths"],
                )
            )
            stats["encoder_history_tokens_dropped"] += int(
                record["encoder_history_tokens_dropped"]
            )
            stats["encoder_evidence_tokens_dropped"] += int(
                record["encoder_evidence_tokens_dropped"]
            )
            stats["continuations_truncated"] += int(record["target_was_truncated"])
            stats["continuation_tokens_dropped"] += record[
                "target_original_length"
            ] - len(record["target_input_ids"])
            if len(buffers[split]) >= shard_size:
                flush(split)
        if max_episodes is not None and stats["episodes"] >= max_episodes:
            break
    flush("train")
    flush("validation")

    if expected_examples is not None and stats["examples"] != expected_examples:
        raise ValueError(
            f"expected {expected_examples} examples, materialized {stats['examples']}"
        )
    resolved_tokenizer_name = str(
        tokenizer_name
        or getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    )
    manifest: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "dataset": "alireza7/GrepSeek-ColdStart-SFT-10k",
        "source_parquet": str(source_path.resolve()),
        "tokenizer_name": resolved_tokenizer_name,
        "config": asdict(cfg),
        "counts": counts,
        "stats": stats,
        "files": files,
        "contracts": {
            "canonical_evidence": "tool_response.stdout",
            "information_lines": "validated_equal_to_nonempty_stdout_lines_then_removed",
            "target": "native_chat_full_minus_native_generation_prompt",
            "split_unit": "episode_id",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


class GrepSeekPreparedDataset(Dataset):
    """Lazy row-group reader for a prepared directory or parquet shard."""

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
            raise RuntimeError("pyarrow is required to load prepared data") from exc
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
            if self.manifest.get("format_version") != FORMAT_VERSION:
                raise ValueError("unsupported GrepSeek prepared format version")
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

    def __len__(self) -> int:
        return self._length

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


def load_grepseek_prepared(
    path: str | Path,
    split: Optional[str] = None,
) -> GrepSeekPreparedDataset:
    return GrepSeekPreparedDataset(path, split=split)


def _pad_right(
    sequences: Sequence[Sequence[int]],
    *,
    pad_value: int,
) -> torch.Tensor:
    if not sequences:
        raise ValueError("cannot pad an empty sequence collection")
    width = max(len(sequence) for sequence in sequences)
    if width <= 0:
        raise ValueError("cannot pad zero-width sequences")
    return torch.tensor(
        [
            [int(x) for x in sequence] + [int(pad_value)] * (width - len(sequence))
            for sequence in sequences
        ],
        dtype=torch.long,
    )


class GrepSeekCompressionCollator:
    """Collate ragged GrepSeek memories into scatter-ready tensors.

    ``memory_positions`` is ordered exactly like flattened source segments;
    within segment ``j`` its ``K_j`` positions are consecutive.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        *,
        pad_token_id: Optional[int] = None,
        compression_ratio: int = DEFAULT_COMPRESSION_RATIO,
        label_pad_token_id: int = -100,
        max_encoder_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        if pad_token_id is None:
            if tokenizer is None:
                raise ValueError("provide tokenizer or pad_token_id")
            pad_token_id = _resolve_pad_token_id(tokenizer)
        if isinstance(compression_ratio, bool) or compression_ratio <= 0:
            raise ValueError("compression_ratio must be positive")
        if max_encoder_tokens is not None and max_encoder_tokens <= 0:
            raise ValueError("max_encoder_tokens must be positive or None")
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive or None")
        self.pad_token_id = int(pad_token_id)
        self.compression_ratio = int(compression_ratio)
        self.label_pad_token_id = int(label_pad_token_id)
        self.max_encoder_tokens = max_encoder_tokens
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("cannot collate an empty batch")

        source_prompts: List[List[int]] = []
        source_prompt_masks: List[List[int]] = []
        evidence_spans_by_sample: List[List[List[int]]] = []
        source_lengths: List[int] = []
        source_span_lengths: List[int] = []
        source_sample_indices: List[int] = []
        source_segment_indices: List[int] = []
        memory_lengths: List[int] = []
        per_sample_memory_starts: List[List[Tuple[int, int]]] = []
        combined_unpadded: List[List[int]] = []
        labels_unpadded: List[List[int]] = []
        prefix_lengths: List[int] = []
        target_lengths: List[int] = []
        num_source_segments: List[int] = []
        runtime_history_dropped = 0
        runtime_evidence_dropped = 0

        for sample_index, feature in enumerate(features):
            discrete = [
                [int(x) for x in _as_list(segment)]
                for segment in _as_list(feature["discrete_segments_input_ids"])
            ]
            raw_sources = [
                [int(x) for x in _as_list(segment)]
                for segment in _as_list(feature["source_segments_input_ids"])
            ]
            source_prompt = [
                int(x) for x in _as_list(feature["source_prompt_input_ids"])
            ]
            evidence_spans = [
                [int(x) for x in _as_list(span)]
                for span in _as_list(feature["source_prompt_evidence_spans"])
            ]
            if not raw_sources or len(discrete) != len(raw_sources) + 1:
                raise ValueError(
                    "each sample needs S sources and S+1 discrete segments"
                )
            if len(evidence_spans) != len(raw_sources):
                raise ValueError("source prompt/evidence segment arity mismatch")
            source_prompt, evidence_spans, history_drop = _left_crop_source_prompt(
                source_prompt, evidence_spans, self.max_encoder_tokens
            )
            runtime_history_dropped += history_drop
            previous_end = -1
            prompt_mask = [0] * len(source_prompt)
            for segment_index, (raw, span) in enumerate(
                zip(raw_sources, evidence_spans)
            ):
                if len(span) != 2:
                    raise ValueError("each source evidence span must be [start, end]")
                start, end = span
                if start < 0 or end <= start or end > len(source_prompt):
                    raise ValueError("source evidence span lies outside source prompt")
                if start < previous_end:
                    raise ValueError(
                        "source evidence spans must be ordered and disjoint"
                    )
                previous_end = end
                for position in range(start, end):
                    prompt_mask[position] = 1
                evidence_length = len(raw)
                if evidence_length <= 0:
                    raise ValueError("source evidence cannot be empty")
                source_lengths.append(evidence_length)
                source_span_lengths.append(end - start)
                source_sample_indices.append(sample_index)
                source_segment_indices.append(segment_index)
                memory_lengths.append(
                    int(math.ceil(evidence_length / self.compression_ratio))
                )
            source_prompts.append(source_prompt)
            source_prompt_masks.append(prompt_mask)
            evidence_spans_by_sample.append(evidence_spans)

            sequence = list(discrete[0])
            sample_memory_starts: List[Tuple[int, int]] = []
            flat_base = len(memory_lengths) - len(raw_sources)
            for segment_index in range(len(raw_sources)):
                memory_length = memory_lengths[flat_base + segment_index]
                start = len(sequence)
                sequence.extend([self.pad_token_id] * memory_length)
                sample_memory_starts.append((start, memory_length))
                sequence.extend(discrete[segment_index + 1])
            prefix_length = len(sequence)
            target = [int(x) for x in _as_list(feature["target_input_ids"])]
            if not target:
                raise ValueError("target_input_ids cannot be empty")
            sequence.extend(target)
            labels = [self.label_pad_token_id] * prefix_length + target

            combined_unpadded.append(sequence)
            labels_unpadded.append(labels)
            per_sample_memory_starts.append(sample_memory_starts)
            prefix_lengths.append(prefix_length)
            target_lengths.append(len(target))
            num_source_segments.append(len(raw_sources))

        total_width = max(len(sequence) for sequence in combined_unpadded)
        if self.pad_to_multiple_of is not None:
            total_width = int(
                math.ceil(total_width / self.pad_to_multiple_of)
                * self.pad_to_multiple_of
            )
        left_pad_lengths = [
            total_width - len(sequence) for sequence in combined_unpadded
        ]
        input_ids = torch.tensor(
            [
                [self.pad_token_id] * left_pad + sequence
                for left_pad, sequence in zip(left_pad_lengths, combined_unpadded)
            ],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [
                [0] * left_pad + [1] * len(sequence)
                for left_pad, sequence in zip(left_pad_lengths, combined_unpadded)
            ],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [
                [self.label_pad_token_id] * left_pad + sequence_labels
                for left_pad, sequence_labels in zip(left_pad_lengths, labels_unpadded)
            ],
            dtype=torch.long,
        )

        memory_positions_list: List[List[int]] = []
        segment_memory_offsets = [0]
        for sample_index, starts in enumerate(per_sample_memory_starts):
            left_pad = left_pad_lengths[sample_index]
            for start, length in starts:
                memory_positions_list.extend(
                    [sample_index, left_pad + position]
                    for position in range(start, start + length)
                )
                segment_memory_offsets.append(segment_memory_offsets[-1] + length)
        memory_positions = torch.tensor(memory_positions_list, dtype=torch.long)
        if memory_positions.ndim != 2 or memory_positions.shape[1] != 2:
            raise AssertionError("memory_positions must have shape [Nmem, 2]")

        source_input_ids = _pad_right(source_prompts, pad_value=self.pad_token_id)
        source_attention_mask = _pad_right(
            [[1] * len(ids) for ids in source_prompts], pad_value=0
        )
        source_evidence_mask = _pad_right(source_prompt_masks, pad_value=0)
        source_evidence_positions_list: List[List[int]] = []
        source_evidence_offsets = [0]
        for sample_index, spans in enumerate(evidence_spans_by_sample):
            for start, end in spans:
                source_evidence_positions_list.extend(
                    [sample_index, position] for position in range(start, end)
                )
                source_evidence_offsets.append(
                    source_evidence_offsets[-1] + end - start
                )
        source_evidence_positions = torch.tensor(
            source_evidence_positions_list, dtype=torch.long
        )
        if (
            source_evidence_positions.ndim != 2
            or source_evidence_positions.shape[1] != 2
        ):
            raise AssertionError("source_evidence_positions must have shape [Ne, 2]")
        target_input_ids = _pad_right(
            [[int(x) for x in _as_list(f["target_input_ids"])] for f in features],
            pad_value=self.pad_token_id,
        )
        target_attention_mask = _pad_right(
            [[1] * length for length in target_lengths], pad_value=0
        )

        return {
            "input_ids": input_ids,
            "student_input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "source_input_ids": source_input_ids,
            "source_attention_mask": source_attention_mask,
            "source_evidence_mask": source_evidence_mask,
            "source_evidence_positions": source_evidence_positions,
            "source_evidence_offsets": torch.tensor(
                source_evidence_offsets, dtype=torch.long
            ),
            "source_lengths": torch.tensor(source_lengths, dtype=torch.long),
            "source_encoder_lengths": torch.tensor(
                [len(ids) for ids in source_prompts], dtype=torch.long
            ),
            "source_span_lengths": torch.tensor(source_span_lengths, dtype=torch.long),
            "source_sample_indices": torch.tensor(
                source_sample_indices, dtype=torch.long
            ),
            "source_segment_indices": torch.tensor(
                source_segment_indices, dtype=torch.long
            ),
            "memory_lengths": torch.tensor(memory_lengths, dtype=torch.long),
            "memory_positions": memory_positions,
            "segment_memory_offsets": torch.tensor(
                segment_memory_offsets, dtype=torch.long
            ),
            "prefix_lengths": torch.tensor(prefix_lengths, dtype=torch.long),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "left_pad_lengths": torch.tensor(left_pad_lengths, dtype=torch.long),
            "target_start_positions": torch.tensor(
                [
                    left_pad + prefix
                    for left_pad, prefix in zip(left_pad_lengths, prefix_lengths)
                ],
                dtype=torch.long,
            ),
            "num_source_segments": torch.tensor(num_source_segments, dtype=torch.long),
            "target_input_ids": target_input_ids,
            "target_attention_mask": target_attention_mask,
            "sample_ids": [str(feature["id"]) for feature in features],
            "episode_ids": [str(feature["episode_id"]) for feature in features],
            "runtime_encoder_history_tokens_dropped": runtime_history_dropped,
            "runtime_encoder_evidence_tokens_dropped": runtime_evidence_dropped,
        }


__all__ = [
    "DEFAULT_COMPRESSION_RATIO",
    "DEFAULT_VALIDATION_FRACTION",
    "FORMAT_VERSION",
    "GrepSeekCompressionCollator",
    "GrepSeekPreparationConfig",
    "GrepSeekPreparedDataset",
    "TOOL_RESPONSE_KEYS",
    "assign_episode_split",
    "build_grepseek_examples",
    "iter_grepseek_rows",
    "load_grepseek_prepared",
    "materialize_grepseek",
    "parse_tool_response",
]
