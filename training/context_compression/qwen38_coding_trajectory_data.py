"""Frozen Qwen3.8 coding trajectories for serving-aligned WM=4 training.

This module is deliberately CPU-only and independent of the trainer.  It
provides two contracts:

* an append-safe JSONL snapshot manifest which hashes and later reads only the
  newline-complete byte prefix observed at freeze time; and
* an action adapter which turns an accepted SpecForge trajectory into one
  final serving-state example for every assistant target after at least five
  completed tool rounds.  The oldest rounds are folded left-to-right while
  the newest four remain discrete working memory.

Parallel tool calls are normalized exactly like the serving rollout: their
individual observations are represented by one ``grepseek_tool_batch_v1``
payload inside one released seven-field tool response.  Consequently every
completed history round is one atomic assistant/tool pair.

The manifest stores only receipts, counts, token hashes, and action
descriptors.  It never stores prompts, reasoning, tool output, or source code.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _datetime
import glob
import hashlib
import json
import math
import os
import sys
import types
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DECONTAMINATION_MODULE_SHA256 = (
    "cbd08613e73116e706e313d3e65c5a0288f6e15ef398818111739a7d5df87d25"
)
DECONTAMINATION_MODULE_SIZE_BYTES = 8914


def _load_decontamination_index_class() -> type:
    """Load only the pinned stdlib-only scanner from verified bytes in memory."""

    module_name = "_qwen38_coding_decontamination"
    module = sys.modules.get(module_name)
    if module is None:
        path = (
            Path(__file__).resolve().parents[1]
            / "specforge"
            / "data"
            / "decontamination.py"
        )
        path_before = os.lstat(path)
        if not path.is_file() or path_before.st_size != DECONTAMINATION_MODULE_SIZE_BYTES:
            raise ImportError(f"invalid decontamination scanner: {path}")
        descriptor = os.open(
            path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            before = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size) != (
                path_before.st_dev,
                path_before.st_ino,
                path_before.st_size,
            ):
                raise ImportError(f"decontamination scanner identity differs: {path}")
            payload = b"".join(
                iter(lambda: os.read(descriptor, 1024 * 1024), b"")
            )
            after = os.fstat(descriptor)
            path_after = os.lstat(path)
            if (
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (path_before.st_dev, path_before.st_ino, path_before.st_size, path_before.st_mtime_ns)
                != (path_after.st_dev, path_after.st_ino, path_after.st_size, path_after.st_mtime_ns)
            ):
                raise ImportError(f"decontamination scanner changed while read: {path}")
        finally:
            os.close(descriptor)
        if hashlib.sha256(payload).hexdigest() != DECONTAMINATION_MODULE_SHA256:
            raise ImportError(f"decontamination scanner SHA256 differs: {path}")
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        sys.modules[module_name] = module
        exec(compile(payload, str(path), "exec"), module.__dict__, module.__dict__)
    return module.DecontaminationIndex


DecontaminationIndex = _load_decontamination_index_class()


MANIFEST_SCHEMA = "qwen38_coding_trajectory_snapshot_v1"
MANIFEST_VERSION = 1
ACCEPTED_SHARD_PREFIX = "qwen38_27b_trajectories.part-"
CANARY_SHARD_PREFIX = "qwen38_27b_canary_trajectories.part-"
PARALLEL_NORMALIZATION = "grepseek_tool_batch_v1"
WORKING_MEMORY_ROUNDS = 4
TARGET_TOKEN_CAP = 256
EVICTION_CHUNK_TOKENS = 1024
COMPRESSION_RATIO = 16
TRAINING_EPOCHS = 10
STUDENT_SEQUENCE_CAP = 12_288
RAW_TEACHER_SEQUENCE_CAP = 12_288
CACHED_PREFILL_CAP = 12_288
TEACHER_EXECUTION_MODE = "monolithic_only"
REPORT_SEQUENCE_CAPS = (12_288, 24_576, 49_152)
QWEN38_ENABLE_THINKING = True
QWEN38_PRESERVE_THINKING = True
QWEN38_GENERATION_BOUNDARY_IDS = (248045, 74455, 198, 248068, 198)
TRAJECTORY_WEIGHTING = "one_action_per_trajectory_progressive_monotone_depth_quantiles"
CANARY_SELECTOR_CONTRACT = (
    "primary_student_cap_gap_ascending_then_teacher_cap_gap_ascending_then_"
    "transition_chunks_descending_then_completed_rounds_descending_then_"
    "descriptor_id__secondary_transition_chunks_descending_then_completed_"
    "rounds_descending_then_teacher_cap_gap_ascending_then_student_cap_gap_"
    "ascending_then_descriptor_id"
)


@dataclass(frozen=True)
class FrozenFileReceipt:
    """Immutable receipt for one newline-complete JSONL prefix."""

    path: str
    size_bytes: int
    complete_bytes: int
    complete_line_count: int
    complete_sha256: str


@dataclass(frozen=True)
class FrozenRowReceipt:
    """Cursor for one row inside a frozen file prefix."""

    file_index: int
    path: str
    line_number: int
    byte_start: int
    byte_length: int


@dataclass(frozen=True)
class FrozenTrajectoryRow:
    """One first-seen trajectory and its deterministic global order."""

    trajectory_order: int
    trajectory_id: str
    receipt: FrozenRowReceipt
    row: dict[str, Any]


@dataclass(frozen=True)
class ServingEvictionChunk:
    """One physical recurrence update wholly inside one evicted round."""

    round_index: int
    chunk_index: int
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class ServingEvictionFold:
    """All left-to-right physical chunks belonging to one complete round."""

    round_index: int
    input_ids: tuple[int, ...]
    chunks: tuple[ServingEvictionChunk, ...]


@dataclass(frozen=True)
class NormalizedServingRound:
    """One atomic assistant plus canonical single tool-observation pair."""

    round_index: int
    assistant_message_index: int
    source_tool_message_indices: tuple[int, ...]
    tool_call_count: int
    parallel_aggregated: bool
    messages: tuple[dict[str, Any], dict[str, Any]]


@dataclass(frozen=True)
class NormalizedCodingTrajectory:
    """Accepted trajectory prefix ending at its first terminal assistant."""

    trajectory_id: str
    tools: tuple[dict[str, Any], ...]
    prefix_messages: tuple[dict[str, Any], dict[str, Any]]
    rounds: tuple[NormalizedServingRound, ...]
    assistant_targets: tuple[tuple[int, dict[str, Any], int], ...]


@dataclass(frozen=True)
class ServingActionExample:
    """One WM=4 action with the final serving state made explicit.

    ``raw_teacher_history_input_ids`` excludes persistent P and the generation
    opening.  ``target_input_ids`` starts after that opening, so its loss mask
    never supervises the assistant header.  Every eviction fold is a complete
    old round; its chunks are at most 1024 tokens and never cross a round.
    """

    descriptor_id: str
    trajectory_id: str
    source_receipt: FrozenRowReceipt | None
    target_message_index: int
    completed_rounds: int
    evicted_rounds: int
    transition_chunks: int
    is_final_serving_state: bool
    raw_prefix_input_ids: tuple[int, ...]
    eviction_round_input_ids: tuple[tuple[int, ...], ...]
    eviction_folds: tuple[ServingEvictionFold, ...]
    working_memory_round_input_ids: tuple[tuple[int, ...], ...]
    working_memory_input_ids: tuple[int, ...]
    raw_teacher_history_input_ids: tuple[int, ...]
    generation_boundary_input_ids: tuple[int, ...]
    target_input_ids: tuple[int, ...]
    target_loss_mask: tuple[bool, ...]
    soft_memory_tokens: int
    teacher_prefix_tokens: int
    teacher_sequence_tokens: int
    cached_prefill_tokens: int
    student_prefix_tokens: int
    student_sequence_tokens: int
    token_hashes: dict[str, str]

    @property
    def example_id(self) -> str:
        """Trainer-facing alias; an action descriptor is one example."""

        return self.descriptor_id

    @property
    def max_recurrent_source_sequence_tokens(self) -> int:
        """Largest ``P + previous soft slots + one physical fold chunk``."""

        slots = 0
        maximum = len(self.raw_prefix_input_ids)
        for fold in self.eviction_folds:
            for chunk in fold.chunks:
                maximum = max(
                    maximum,
                    len(self.raw_prefix_input_ids) + slots + len(chunk.input_ids),
                )
                slots = math.ceil((slots + len(chunk.input_ids)) / COMPRESSION_RATIO)
        return maximum

    def descriptor(self) -> dict[str, Any]:
        receipt = (
            dataclasses.asdict(self.source_receipt)
            if self.source_receipt is not None
            else None
        )
        result = {
            "descriptor_id": self.descriptor_id,
            "trajectory_id": self.trajectory_id,
            "source_receipt": receipt,
            "target_message_index": self.target_message_index,
            "completed_rounds": self.completed_rounds,
            "evicted_rounds": self.evicted_rounds,
            "transition_chunks": self.transition_chunks,
            "is_final_serving_state": self.is_final_serving_state,
            "prefix_tokens": len(self.raw_prefix_input_ids),
            "working_memory_tokens": len(self.working_memory_input_ids),
            "generation_boundary_tokens": len(self.generation_boundary_input_ids),
            "target_tokens": len(self.target_input_ids),
            "soft_memory_tokens": self.soft_memory_tokens,
            "teacher_prefix_tokens": self.teacher_prefix_tokens,
            "teacher_sequence_tokens": self.teacher_sequence_tokens,
            "cached_prefill_tokens": self.cached_prefill_tokens,
            "student_prefix_tokens": self.student_prefix_tokens,
            "student_sequence_tokens": self.student_sequence_tokens,
            "max_recurrent_source_sequence_tokens": (
                self.max_recurrent_source_sequence_tokens
            ),
            "token_hashes": dict(self.token_hashes),
        }
        result["descriptor_sha256"] = _canonical_sha(result)
        return result


def action_manifest_descriptor(
    example: ServingActionExample, *, trajectory_order: int
) -> dict[str, Any]:
    """Return the fully signed descriptor representation stored in manifests."""

    if (
        isinstance(trajectory_order, bool)
        or not isinstance(trajectory_order, int)
        or trajectory_order < 0
    ):
        raise ValueError("trajectory_order must be a nonnegative integer")
    descriptor = example.descriptor()
    descriptor["trajectory_order"] = trajectory_order
    return _resign_descriptor(descriptor)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _token_sha(input_ids: Sequence[int]) -> str:
    return _canonical_sha([int(token) for token in input_ids])


def _resign_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("descriptor_sha256", None)
    result["descriptor_sha256"] = _canonical_sha(result)
    return result


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
                raise ValueError(f"{label} contains a non-string key")
            copied[key] = _copy_json(item, label=label)
        return copied
    raise ValueError(f"{label} contains a non-JSON value")


def _as_token_ids(
    value: Any, *, label: str, allow_empty: bool = False
) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must return token ids")
    if (
        value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        if len(value) != 1:
            raise ValueError(f"{label} unexpectedly returned a batch")
        value = value[0]
    result: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int):
            raise TypeError(f"{label} returned a non-integer token")
        result.append(int(token))
    if not result and not allow_empty:
        raise ValueError(f"{label} returned no token ids")
    return tuple(result)


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]],
    tokenize: bool,
    add_generation_prompt: bool,
) -> Any:
    value = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tools=[dict(tool) for tool in tools],
        tokenize=bool(tokenize),
        add_generation_prompt=bool(add_generation_prompt),
        enable_thinking=QWEN38_ENABLE_THINKING,
        preserve_thinking=QWEN38_PRESERVE_THINKING,
        return_dict=False,
    )
    if tokenize:
        return _as_token_ids(value, label="Qwen3.8 apply_chat_template")
    if not isinstance(value, str) or not value:
        raise TypeError("Qwen3.8 textual chat template must return nonempty text")
    return value


def _encode_continuation(tokenizer: Any, text: str) -> tuple[int, ...]:
    return _as_token_ids(
        tokenizer.encode(text, add_special_tokens=False),
        label="Qwen3.8 continuation encode",
    )


def _normalize_tool_call(raw: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    call_id = raw.get("id")
    function = raw.get("function")
    if not isinstance(call_id, str) or not call_id:
        raise ValueError(f"{label}.id must be nonempty")
    if raw.get("type") != "function" or not isinstance(function, Mapping):
        raise ValueError(f"{label} must be a function call")
    name = function.get("name")
    arguments = function.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.function.name must be nonempty")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}.function.arguments is invalid JSON") from error
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{label}.function.arguments must be an object")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _copy_json(arguments, label=f"{label}.arguments"),
        },
    }


def _normalize_message(raw: Any, *, message_index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"messages[{message_index}] must be an object")
    role = raw.get("role")
    content = raw.get("content")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"messages[{message_index}].role is unsupported")
    if not isinstance(content, str):
        raise ValueError(f"messages[{message_index}].content must be a string")
    result: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        reasoning = raw.get("reasoning_content")
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise ValueError("assistant reasoning_content must be a string")
            result["reasoning_content"] = reasoning
        calls = raw.get("tool_calls")
        if calls is not None:
            if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
                raise ValueError("assistant tool_calls must be an array")
            if calls:
                result["tool_calls"] = [
                    _normalize_tool_call(
                        call,
                        label=f"messages[{message_index}].tool_calls[{call_index}]",
                    )
                    for call_index, call in enumerate(calls)
                ]
    elif role == "tool":
        call_id = raw.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool message must have a nonempty tool_call_id")
        result["tool_call_id"] = call_id
        if isinstance(raw.get("name"), str) and raw.get("name"):
            result["name"] = str(raw["name"])
    return result


def _result_exit_code(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0
    for key in ("exit_code", "returncode"):
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return int(candidate)
    return 0 if value.get("ok", True) else 1


def _canonical_source_result(
    call: Mapping[str, Any], tool_message: Mapping[str, Any]
) -> dict[str, Any]:
    """Project a SpecForge tool result onto the serving seven-field ABI."""

    content = str(tool_message.get("content") or "")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        decoded = None
    function = call["function"]
    fallback_command = str(function["name"])
    arguments = function.get("arguments") or {}
    if arguments:
        fallback_command += " " + json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    command = (
        str(decoded.get("command"))
        if isinstance(decoded, Mapping) and isinstance(decoded.get("command"), str)
        else fallback_command
    )
    stdout = content
    stderr = ""
    if isinstance(decoded, Mapping):
        if isinstance(decoded.get("stdout"), str):
            stdout = str(decoded["stdout"])
        elif isinstance(decoded.get("content"), str):
            stdout = str(decoded["content"])
        if isinstance(decoded.get("stderr"), str):
            stderr = str(decoded["stderr"])
        elif isinstance(decoded.get("error"), str):
            stderr = str(decoded["error"])
    return {
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": _result_exit_code(decoded),
        "timed_out": (
            bool(decoded.get("timed_out", False))
            if isinstance(decoded, Mapping)
            else False
        ),
        "truncated": (
            bool(decoded.get("truncated", False))
            if isinstance(decoded, Mapping)
            else False
        ),
        "information_lines": [],
    }


def canonical_aggregate_tool_results(
    calls: Sequence[Mapping[str, Any]],
    tool_messages: Sequence[Mapping[str, Any]],
) -> str:
    """Return the exact single tool-role payload used by serving.

    A single result is projected directly onto the released seven-field
    contract.  Multiple results use the same nested ``grepseek_tool_batch_v1``
    structure and outer :class:`ToolResult` shape as
    ``grepseek_rollout._aggregate_tool_call_results``.
    """

    if not calls or len(calls) != len(tool_messages):
        raise ValueError("calls and tool messages must be nonempty and equal length")
    by_id = {str(message["tool_call_id"]): message for message in tool_messages}
    if len(by_id) != len(tool_messages):
        raise ValueError("tool result batch contains duplicate IDs")
    ordered_messages = []
    for call in calls:
        call_id = str(call["id"])
        if call_id not in by_id:
            raise ValueError("tool result IDs do not match assistant calls")
        ordered_messages.append(by_id[call_id])
    results = [
        _canonical_source_result(call, message)
        for call, message in zip(calls, ordered_messages, strict=True)
    ]
    if len(results) == 1:
        return json.dumps(results[0], ensure_ascii=False)

    nested = []
    for call_index, result in enumerate(results, start=1):
        nested_result = dict(result)
        nested_result["information_lines"] = []
        nested.append(
            {
                "call_index": call_index,
                "executed": True,
                "validation_failed": result["exit_code"] == -2,
                "command": result["command"],
                "result": nested_result,
            }
        )
    batch_payload = {
        "schema": PARALLEL_NORMALIZATION,
        "tool_call_count": len(results),
        "results": nested,
    }
    stdout = json.dumps(
        batch_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    outer = {
        "command": f"tool_batch[{len(results)}]",
        "stdout": stdout,
        "stderr": "",
        "exit_code": 0 if all(item["exit_code"] == 0 for item in results) else 1,
        "timed_out": any(bool(item["timed_out"]) for item in results),
        "truncated": any(bool(item["truncated"]) for item in results),
        # Model-visible serving releases stdout exactly once.  The legacy
        # information_lines mirror is audit-only and must stay empty here.
        "information_lines": [],
    }
    return json.dumps(outer, ensure_ascii=False)


def normalize_coding_trajectory(row: Mapping[str, Any]) -> NormalizedCodingTrajectory:
    """Normalize one accepted row into atomic serving tool rounds.

    The usable serving episode stops at its first assistant without tool calls.
    Later user follow-ups belong to a new conversational episode and are not
    silently folded into the original tool trajectory.
    """

    if not isinstance(row, Mapping):
        raise ValueError("trajectory row must be an object")
    trajectory_id = row.get("id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("trajectory id must be nonempty")
    validation = row.get("validation")
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ValueError("only validation.valid accepted trajectories are supported")
    generation_start = row.get("generation_start_message_index")
    if generation_start != 2:
        raise ValueError("generation_start_message_index must equal 2")
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise ValueError("trajectory messages must be an array")
    messages = tuple(
        _normalize_message(message, message_index=index)
        for index, message in enumerate(raw_messages)
    )
    if len(messages) < 3 or [message["role"] for message in messages[:2]] != [
        "system",
        "user",
    ]:
        raise ValueError("trajectory prefix must be exactly system/user")
    raw_tools = row.get("tools") or []
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        raise ValueError("trajectory tools must be an array")
    tools = tuple(_copy_json(tool, label="tools") for tool in raw_tools)

    rounds: list[NormalizedServingRound] = []
    targets: list[tuple[int, dict[str, Any], int]] = []
    seen_call_ids: set[str] = set()
    index = generation_start
    while index < len(messages):
        assistant = messages[index]
        if assistant["role"] != "assistant":
            raise ValueError(
                "serving trajectory must contain an assistant at every action boundary"
            )
        calls = assistant.get("tool_calls") or []
        if not (
            assistant["content"].strip()
            or str(assistant.get("reasoning_content") or "").strip()
            or calls
        ):
            raise ValueError("assistant target is empty")
        targets.append((index, dict(assistant), len(rounds)))
        if not calls:
            break
        call_ids = [str(call["id"]) for call in calls]
        if len(set(call_ids)) != len(call_ids) or seen_call_ids.intersection(call_ids):
            raise ValueError("tool call IDs must be unique across the episode")
        seen_call_ids.update(call_ids)
        stop = index + 1
        result_messages: list[dict[str, Any]] = []
        result_indices: list[int] = []
        while stop < len(messages) and messages[stop]["role"] == "tool":
            result_messages.append(dict(messages[stop]))
            result_indices.append(stop)
            stop += 1
        observed_ids = [str(message["tool_call_id"]) for message in result_messages]
        if Counter(call_ids) != Counter(observed_ids):
            raise ValueError(
                "assistant calls must have one complete matching result batch"
            )
        canonical_tool = {
            "role": "tool",
            "content": canonical_aggregate_tool_results(calls, result_messages),
        }
        rounds.append(
            NormalizedServingRound(
                round_index=len(rounds),
                assistant_message_index=index,
                source_tool_message_indices=tuple(result_indices),
                tool_call_count=len(calls),
                parallel_aggregated=len(calls) > 1,
                messages=(dict(assistant), canonical_tool),
            )
        )
        index = stop
    return NormalizedCodingTrajectory(
        trajectory_id=trajectory_id,
        tools=tools,
        prefix_messages=(dict(messages[0]), dict(messages[1])),
        rounds=tuple(rounds),
        assistant_targets=tuple(targets),
    )


def _recurrent_slots(chunks: Iterable[ServingEvictionChunk]) -> int:
    slots = 0
    for chunk in chunks:
        physical = slots + len(chunk.input_ids)
        slots = math.ceil(physical / COMPRESSION_RATIO)
    return slots


def _eviction_folds(
    round_input_ids: Sequence[Sequence[int]],
) -> tuple[ServingEvictionFold, ...]:
    folds: list[ServingEvictionFold] = []
    for round_index, raw_ids in enumerate(round_input_ids):
        ids = tuple(int(token) for token in raw_ids)
        if not ids:
            raise ValueError("a complete serving round rendered no tokens")
        chunks = tuple(
            ServingEvictionChunk(
                round_index=round_index,
                chunk_index=chunk_index,
                input_ids=ids[start : start + EVICTION_CHUNK_TOKENS],
            )
            for chunk_index, start in enumerate(
                range(0, len(ids), EVICTION_CHUNK_TOKENS)
            )
        )
        folds.append(
            ServingEvictionFold(
                round_index=round_index,
                input_ids=ids,
                chunks=chunks,
            )
        )
    return tuple(folds)


def iter_wm4_action_examples(
    row: Mapping[str, Any] | NormalizedCodingTrajectory,
    tokenizer: Any,
    *,
    source_receipt: FrozenRowReceipt | None = None,
    working_memory_rounds: int = WORKING_MEMORY_ROUNDS,
    target_cap: int = TARGET_TOKEN_CAP,
    student_cap: int | None = None,
) -> tuple[ServingActionExample, ...]:
    """Return every serving-aligned action with nonempty soft memory.

    ``student_cap=None`` freezes all eligible events.  Passing a cap filters
    whole examples only; no round, boundary, or target is truncated except for
    the explicit assistant-only ``target_cap``.
    """

    if working_memory_rounds != WORKING_MEMORY_ROUNDS:
        raise ValueError("the Qwen3.8 coding contract requires WM=4")
    if target_cap != TARGET_TOKEN_CAP:
        raise ValueError("the Qwen3.8 coding contract requires target_cap=256")
    if student_cap is not None and (
        isinstance(student_cap, bool)
        or not isinstance(student_cap, int)
        or student_cap <= 0
    ):
        raise ValueError("student_cap must be a positive integer or None")
    episode = (
        row
        if isinstance(row, NormalizedCodingTrajectory)
        else normalize_coding_trajectory(row)
    )
    if not any(
        completed_rounds > working_memory_rounds
        for _, _, completed_rounds in episode.assistant_targets
    ):
        return ()
    prefix_ids = _apply_chat_template(
        tokenizer,
        episode.prefix_messages,
        tools=episode.tools,
        tokenize=True,
        add_generation_prompt=False,
    )
    prefix_prompt_ids = _apply_chat_template(
        tokenizer,
        episode.prefix_messages,
        tools=episode.tools,
        tokenize=True,
        add_generation_prompt=True,
    )
    if prefix_prompt_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("native generation prompt does not preserve P")
    boundary_ids = prefix_prompt_ids[len(prefix_ids) :]
    if boundary_ids != QWEN38_GENERATION_BOUNDARY_IDS:
        raise ValueError("native Qwen3.8 generation boundary IDs changed")
    prefix_prompt_text = _apply_chat_template(
        tokenizer,
        episode.prefix_messages,
        tools=episode.tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    # With preserve_thinking=True the verified Qwen3.8 template renders every
    # completed assistant/tool pair append-stably.  Isolating each round avoids
    # quadratic full-history tokenization while the deepest-history checks
    # below still fail closed if the native template ever changes that rule.
    round_ids: list[tuple[int, ...]] = []
    for completed_round in episode.rounds:
        rendered = _apply_chat_template(
            tokenizer,
            [*episode.prefix_messages, *completed_round.messages],
            tools=episode.tools,
            tokenize=True,
            add_generation_prompt=False,
        )
        if rendered[: len(prefix_ids)] != prefix_ids:
            raise ValueError("native complete round rendering changed P")
        delta = rendered[len(prefix_ids) :]
        if not delta:
            raise ValueError("native complete round rendered no tokens")
        round_ids.append(delta)

    full_history_messages: list[dict[str, Any]] = [
        dict(message) for message in episode.prefix_messages
    ]
    for completed_round in episode.rounds:
        full_history_messages.extend(
            dict(message) for message in completed_round.messages
        )
    deepest_ids = _apply_chat_template(
        tokenizer,
        full_history_messages,
        tools=episode.tools,
        tokenize=True,
        add_generation_prompt=False,
    )
    expected_deepest_ids = (
        *prefix_ids,
        *(token for ids in round_ids for token in ids),
    )
    if deepest_ids != expected_deepest_ids:
        raise ValueError("native complete round rendering is not append-stable")
    deepest_prompt_ids = _apply_chat_template(
        tokenizer,
        full_history_messages,
        tools=episode.tools,
        tokenize=True,
        add_generation_prompt=True,
    )
    if deepest_prompt_ids != (*deepest_ids, *boundary_ids):
        raise ValueError("native generation boundary changed with history depth")

    examples: list[ServingActionExample] = []
    for (
        target_message_index,
        target_message,
        completed_rounds,
    ) in episode.assistant_targets:
        if completed_rounds > len(round_ids):
            raise RuntimeError("assistant target depth exceeds rendered rounds")
        if completed_rounds <= working_memory_rounds:
            continue
        completed_text = _apply_chat_template(
            tokenizer,
            [*episode.prefix_messages, target_message],
            tools=episode.tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not completed_text.startswith(prefix_prompt_text):
            raise ValueError("assistant completion text does not preserve its prompt")
        continuation_text = completed_text[len(prefix_prompt_text) :]
        target_ids = _encode_continuation(tokenizer, continuation_text)[:target_cap]
        if not target_ids:
            raise ValueError("assistant target rendered no continuation tokens")

        completed_round_ids = round_ids[:completed_rounds]
        evicted = completed_rounds - working_memory_rounds
        eviction_rounds = tuple(completed_round_ids[:evicted])
        working_rounds = tuple(completed_round_ids[evicted:])
        folds = _eviction_folds(eviction_rounds)
        chunks = tuple(chunk for fold in folds for chunk in fold.chunks)
        soft_tokens = _recurrent_slots(chunks)
        working_ids = tuple(token for ids in working_rounds for token in ids)
        teacher_history = tuple(token for ids in completed_round_ids for token in ids)
        teacher_prefix_tokens = (
            len(prefix_ids) + len(teacher_history) + len(boundary_ids)
        )
        teacher_sequence_tokens = teacher_prefix_tokens + len(target_ids)
        student_prefix_tokens = (
            len(prefix_ids) + soft_tokens + len(working_ids) + len(boundary_ids)
        )
        student_sequence_tokens = student_prefix_tokens + len(target_ids)
        token_hashes = {
            "raw_prefix_sha256": _token_sha(prefix_ids),
            "eviction_rounds_sha256": _canonical_sha(
                [_token_sha(ids) for ids in eviction_rounds]
            ),
            "working_memory_sha256": _token_sha(working_ids),
            "raw_teacher_history_sha256": _token_sha(teacher_history),
            "generation_boundary_sha256": _token_sha(boundary_ids),
            "teacher_prefix_sha256": _token_sha(
                (*prefix_ids, *teacher_history, *boundary_ids)
            ),
            "student_layout_sha256": _canonical_sha(
                {
                    "raw_prefix": _token_sha(prefix_ids),
                    "soft_memory_tokens": soft_tokens,
                    "working_memory": _token_sha(working_ids),
                    "generation_boundary": _token_sha(boundary_ids),
                }
            ),
            "target_sha256": _token_sha(target_ids),
        }
        identity = {
            "trajectory_id": episode.trajectory_id,
            "target_message_index": target_message_index,
            "completed_rounds": completed_rounds,
            "token_hashes": token_hashes,
        }
        example = ServingActionExample(
            descriptor_id=_canonical_sha(identity),
            trajectory_id=episode.trajectory_id,
            source_receipt=source_receipt,
            target_message_index=target_message_index,
            completed_rounds=completed_rounds,
            evicted_rounds=evicted,
            transition_chunks=len(chunks),
            is_final_serving_state=True,
            raw_prefix_input_ids=prefix_ids,
            eviction_round_input_ids=eviction_rounds,
            eviction_folds=folds,
            working_memory_round_input_ids=working_rounds,
            working_memory_input_ids=working_ids,
            raw_teacher_history_input_ids=teacher_history,
            generation_boundary_input_ids=boundary_ids,
            target_input_ids=target_ids,
            target_loss_mask=(True,) * len(target_ids),
            soft_memory_tokens=soft_tokens,
            teacher_prefix_tokens=teacher_prefix_tokens,
            teacher_sequence_tokens=teacher_sequence_tokens,
            cached_prefill_tokens=teacher_prefix_tokens,
            student_prefix_tokens=student_prefix_tokens,
            student_sequence_tokens=student_sequence_tokens,
            token_hashes=token_hashes,
        )
        if student_cap is None or example.student_sequence_tokens <= student_cap:
            examples.append(example)
    return tuple(examples)


def select_action_for_epoch(
    examples: Sequence[ServingActionExample],
    *,
    epoch: int,
    seed: int = 0,
    epochs: int = TRAINING_EPOCHS,
) -> ServingActionExample:
    """Choose one monotone depth-quantile target for this trajectory/epoch."""

    if not examples:
        raise ValueError("cannot select from zero action examples")
    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or epochs < 2
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 0 <= epoch < epochs
    ):
        raise ValueError("epoch is outside the configured training schedule")
    trajectory_ids = {item.trajectory_id for item in examples}
    if len(trajectory_ids) != 1:
        raise ValueError("epoch selection requires one trajectory at a time")
    trajectory_id = next(iter(trajectory_ids))
    # The seed is deliberately permitted only as the final impossible-tie
    # breaker.  It can never alter completed-round depth or chronology.
    ordered = sorted(
        examples,
        key=lambda item: (
            item.completed_rounds,
            item.target_message_index,
            hashlib.sha256(
                f"{seed}:{trajectory_id}:{item.descriptor_id}".encode("utf-8")
            ).hexdigest(),
            item.descriptor_id,
        ),
    )
    quantile_index = (epoch * (len(ordered) - 1)) // (epochs - 1)
    return ordered[quantile_index]


def _complete_prefix_bytes(path: Path, *, observed_size: int) -> int:
    if observed_size == 0:
        return 0
    with path.open("rb") as handle:
        if observed_size > 0:
            handle.seek(observed_size - 1)
            if handle.read(1) == b"\n":
                return observed_size
        cursor = observed_size
        while cursor:
            take = min(cursor, 1024 * 1024)
            cursor -= take
            handle.seek(cursor)
            chunk = handle.read(take)
            index = chunk.rfind(b"\n")
            if index >= 0:
                return cursor + index + 1
    return 0


def _hash_file_prefix(path: Path, limit: int) -> str:
    digest = hashlib.sha256()
    remaining = int(limit)
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError(f"{path} became shorter than its frozen prefix")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def freeze_file_receipts(paths: Sequence[str | Path]) -> tuple[FrozenFileReceipt, ...]:
    """Capture append-safe receipts without trusting a live final line."""

    resolved = [Path(path).expanduser().resolve() for path in paths]
    if not resolved:
        raise ValueError("at least one accepted JSONL shard is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("snapshot shard paths must be unique")
    receipts: list[FrozenFileReceipt] = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        accepted_name = path.name.startswith(ACCEPTED_SHARD_PREFIX) or (
            path.name.startswith(CANARY_SHARD_PREFIX)
        )
        if not (
            accepted_name
            and path.name.endswith(".jsonl")
            and "partial" not in path.name
            and ".pre_" not in path.name
        ):
            raise ValueError(f"non-accepted shard is forbidden: {path.name}")
        observed_size = path.stat().st_size
        complete_bytes = _complete_prefix_bytes(path, observed_size=observed_size)
        line_count = 0
        remaining = complete_bytes
        with path.open("rb") as handle:
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(f"{path} shrank during snapshot")
                line_count += chunk.count(b"\n")
                remaining -= len(chunk)
        receipts.append(
            FrozenFileReceipt(
                path=str(path),
                size_bytes=observed_size,
                complete_bytes=complete_bytes,
                complete_line_count=line_count,
                complete_sha256=_hash_file_prefix(path, complete_bytes),
            )
        )
    return tuple(receipts)


def _receipt_from_mapping(value: Mapping[str, Any]) -> FrozenFileReceipt:
    return FrozenFileReceipt(
        path=str(value["path"]),
        size_bytes=int(value["size_bytes"]),
        complete_bytes=int(value["complete_bytes"]),
        complete_line_count=int(value["complete_line_count"]),
        complete_sha256=str(value["complete_sha256"]),
    )


def _iter_receipt_rows(
    receipts: Sequence[FrozenFileReceipt], *, verify_files: bool
) -> Iterator[tuple[dict[str, Any], FrozenRowReceipt]]:
    for file_index, receipt in enumerate(receipts):
        path = Path(receipt.path)
        if verify_files and _hash_file_prefix(path, receipt.complete_bytes) != (
            receipt.complete_sha256
        ):
            raise ValueError(f"frozen JSONL prefix hash mismatch: {path}")
        with path.open("rb") as handle:
            line_number = 0
            while handle.tell() < receipt.complete_bytes:
                byte_start = handle.tell()
                remaining = receipt.complete_bytes - byte_start
                line = handle.readline(remaining)
                if not line.endswith(b"\n"):
                    raise ValueError(
                        "frozen complete prefix contains an incomplete line"
                    )
                line_number += 1
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSON in frozen shard {path}:{line_number}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError("frozen JSONL row must be an object")
                yield row, FrozenRowReceipt(
                    file_index=file_index,
                    path=str(path),
                    line_number=line_number,
                    byte_start=byte_start,
                    byte_length=len(line),
                )
            if line_number != receipt.complete_line_count:
                raise ValueError("frozen line count does not match its receipt")


def iter_frozen_trajectory_rows(
    manifest: Mapping[str, Any] | str | Path,
    *,
    verify_files: bool = True,
    eligible_only: bool = False,
) -> Iterator[FrozenTrajectoryRow]:
    """Yield first-seen rows in manifest order with stable source cursors."""

    payload = (
        load_frozen_trajectory_manifest(manifest, verify_files=verify_files)
        if isinstance(manifest, (str, Path))
        else dict(manifest)
    )
    _validate_manifest_contract(payload)
    if verify_files and not isinstance(manifest, (str, Path)):
        _verify_external_receipts(payload)
    receipts = tuple(_receipt_from_mapping(item) for item in payload["files"])
    eligible_ids = (
        {str(item["trajectory_id"]) for item in payload["action_descriptors"]}
        if eligible_only
        else None
    )
    seen: set[str] = set()
    trajectory_order = 0
    for row, receipt in _iter_receipt_rows(receipts, verify_files=verify_files):
        trajectory_id = row.get("id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            continue
        if trajectory_id in seen:
            continue
        seen.add(trajectory_id)
        if eligible_ids is None or trajectory_id in eligible_ids:
            yield FrozenTrajectoryRow(
                trajectory_order=trajectory_order,
                trajectory_id=trajectory_id,
                receipt=receipt,
                row=row,
            )
        trajectory_order += 1


def iter_frozen_action_examples(
    manifest: Mapping[str, Any] | str | Path,
    tokenizer: Any,
    *,
    expected_sha256: str | None = None,
    verify_files: bool = True,
    student_cap: int = STUDENT_SEQUENCE_CAP,
    raw_teacher_cap: int = RAW_TEACHER_SEQUENCE_CAP,
) -> Iterator[ServingActionExample]:
    """Rebuild exactly the pinned, cap-safe actions in descriptor order.

    This is the trainer-facing load path.  Source rows are recovered only from
    their frozen newline-complete prefixes, and each native render is checked
    against the text-free token/count receipt before it is released.
    """

    payload = (
        load_frozen_trajectory_manifest(
            manifest,
            expected_sha256=expected_sha256,
            verify_files=verify_files,
        )
        if isinstance(manifest, (str, Path))
        else dict(manifest)
    )
    _validate_manifest_contract(payload)
    if expected_sha256 is not None and payload["canonical_sha256"] != expected_sha256:
        raise ValueError("frozen manifest does not match expected canonical SHA256")
    if verify_files and not isinstance(manifest, (str, Path)):
        _verify_external_receipts(payload)
    if student_cap != STUDENT_SEQUENCE_CAP:
        raise ValueError("trainer load requires the locked student cap 12288")
    if raw_teacher_cap != RAW_TEACHER_SEQUENCE_CAP:
        raise ValueError("trainer load requires the locked raw teacher cap 12288")

    descriptors_by_trajectory: dict[str, list[Mapping[str, Any]]] = {}
    for descriptor in payload["action_descriptors"]:
        if (
            int(descriptor["student_sequence_tokens"]) > student_cap
            or int(descriptor["teacher_sequence_tokens"]) > raw_teacher_cap
            or int(descriptor["max_recurrent_source_sequence_tokens"]) > student_cap
        ):
            continue
        descriptors_by_trajectory.setdefault(
            str(descriptor["trajectory_id"]), []
        ).append(descriptor)

    rebuilt: dict[str, ServingActionExample] = {}
    for frozen in iter_frozen_trajectory_rows(
        payload,
        verify_files=False,
        eligible_only=True,
    ):
        pinned = descriptors_by_trajectory.get(frozen.trajectory_id)
        if not pinned:
            continue
        examples = iter_wm4_action_examples(
            frozen.row,
            tokenizer,
            source_receipt=frozen.receipt,
            student_cap=None,
        )
        local = {example.descriptor_id: example for example in examples}
        for descriptor in pinned:
            descriptor_id = str(descriptor["descriptor_id"])
            example = local.get(descriptor_id)
            if example is None:
                raise ValueError(f"pinned action was not rebuilt: {descriptor_id}")
            actual = action_manifest_descriptor(
                example,
                trajectory_order=int(descriptor["trajectory_order"]),
            )
            if actual != dict(descriptor):
                raise ValueError(f"native action receipt mismatch: {descriptor_id}")
            rebuilt[descriptor_id] = example

    for descriptor in payload["action_descriptors"]:
        descriptor_id = str(descriptor["descriptor_id"])
        if (
            int(descriptor["student_sequence_tokens"]) > student_cap
            or int(descriptor["teacher_sequence_tokens"]) > raw_teacher_cap
            or int(descriptor["max_recurrent_source_sequence_tokens"]) > student_cap
        ):
            continue
        example = rebuilt.get(descriptor_id)
        if example is None:
            raise ValueError(f"cap-safe action row was not found: {descriptor_id}")
        yield example


def iter_epoch_selected_action_examples(
    manifest: Mapping[str, Any] | str | Path,
    tokenizer: Any,
    *,
    epoch: int,
    seed: int = 0,
    expected_sha256: str | None = None,
    verify_files: bool = True,
) -> Iterator[ServingActionExample]:
    """Yield one deterministic depth-rotating action per trajectory."""

    current_id: str | None = None
    group: list[ServingActionExample] = []
    for example in iter_frozen_action_examples(
        manifest,
        tokenizer,
        expected_sha256=expected_sha256,
        verify_files=verify_files,
    ):
        if current_id is not None and example.trajectory_id != current_id:
            yield select_action_for_epoch(group, epoch=epoch, seed=seed)
            group = []
        current_id = example.trajectory_id
        group.append(example)
    if group:
        yield select_action_for_epoch(group, epoch=epoch, seed=seed)


def _screening_record(row: Mapping[str, Any]) -> dict[str, str]:
    metadata = row.get("source_metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    messages = row.get("messages") or []
    prompt_parts = []
    for message in messages if isinstance(messages, Sequence) else []:
        if (
            isinstance(message, Mapping)
            and message.get("role") in {"system", "user"}
            and isinstance(message.get("content"), str)
        ):
            prompt_parts.append(str(message["content"]))
    attachment = metadata.get("attachment_context")
    if isinstance(attachment, str) and attachment:
        prompt_parts.append(attachment)
    return {
        "source": str(metadata.get("source") or ""),
        "source_id": str(metadata.get("source_id") or ""),
        "repo": str(metadata.get("resolved_repo") or metadata.get("repo") or ""),
        "prompt": "\n\n".join(prompt_parts),
    }


def _validate_decontamination_payload(payload: Mapping[str, Any]) -> None:
    """Require the public content-free index and its critical coding blocks."""

    if payload.get("contains_benchmark_content") is not False:
        raise ValueError("decontamination index must be content-free")
    policy = payload.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("store_benchmark_content") is not False
    ):
        raise ValueError("decontamination policy must forbid benchmark content")
    markers = "\n".join(
        str(value).casefold()
        for key in ("blocked_source_markers", "blocked_text_markers")
        for value in (payload.get(key) or [])
    ).replace("_", "-")
    benchmark_names = "\n".join(
        str(item.get("name") or "").casefold().replace("_", "-")
        for item in (payload.get("benchmarks") or [])
        if isinstance(item, Mapping)
    )
    combined = f"{markers}\n{benchmark_names}"
    if "terminal-bench" not in combined or "swe-bench" not in combined:
        raise ValueError(
            "decontamination index lacks Terminal-Bench or SWE-bench blocks"
        )
    swe_repositories = [
        repository
        for item in (payload.get("benchmarks") or [])
        if isinstance(item, Mapping)
        and "swe-bench" in str(item.get("name") or "").casefold().replace("_", "-")
        for repository in (item.get("repositories") or [])
        if str(repository).strip()
    ]
    if not swe_repositories:
        raise ValueError("decontamination index lacks blocked SWE-bench repositories")


def _percentiles(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            name: None
            for name in ("min", "p50", "p75", "p90", "p95", "p99", "max", "mean")
        }
    ordered = sorted(int(value) for value in values)

    def rank(p: float) -> int:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "p50": rank(0.50),
        "p75": rank(0.75),
        "p90": rank(0.90),
        "p95": rank(0.95),
        "p99": rank(0.99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 3),
    }


def _cap_coverage(descriptors: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for cap in REPORT_SEQUENCE_CAPS:
        teacher = sum(
            int(item["teacher_sequence_tokens"]) <= cap for item in descriptors
        )
        student = sum(
            int(item["student_sequence_tokens"]) <= cap for item in descriptors
        )
        both = sum(
            int(item["teacher_sequence_tokens"]) <= cap
            and int(item["student_sequence_tokens"]) <= cap
            for item in descriptors
        )
        matrix[str(cap)] = {"teacher": teacher, "student": student, "both": both}
    final_student_usable = sum(
        int(item["student_sequence_tokens"]) <= STUDENT_SEQUENCE_CAP
        for item in descriptors
    )
    recurrent_source_usable = sum(
        int(item["max_recurrent_source_sequence_tokens"]) <= STUDENT_SEQUENCE_CAP
        for item in descriptors
    )
    student_usable = sum(
        int(item["student_sequence_tokens"]) <= STUDENT_SEQUENCE_CAP
        and int(item["max_recurrent_source_sequence_tokens"]) <= STUDENT_SEQUENCE_CAP
        for item in descriptors
    )
    trajectory_ids = {str(item["trajectory_id"]) for item in descriptors}
    cap_safe_trajectory_ids = {
        str(item["trajectory_id"])
        for item in descriptors
        if int(item["student_sequence_tokens"]) <= STUDENT_SEQUENCE_CAP
        and int(item["max_recurrent_source_sequence_tokens"]) <= STUDENT_SEQUENCE_CAP
        and int(item["teacher_sequence_tokens"]) <= RAW_TEACHER_SEQUENCE_CAP
    }
    locked = {
        "student_cap": STUDENT_SEQUENCE_CAP,
        "raw_teacher_cap": RAW_TEACHER_SEQUENCE_CAP,
        "cached_prefill_cap": CACHED_PREFILL_CAP,
        "teacher_execution_mode": TEACHER_EXECUTION_MODE,
        "cap_safe_trajectories": len(cap_safe_trajectory_ids),
        "trajectories_without_cap_safe_action": len(trajectory_ids)
        - len(cap_safe_trajectory_ids),
        "student_usable": student_usable,
        "student_skipped": len(descriptors) - student_usable,
        "final_student_usable": final_student_usable,
        "final_student_skipped": len(descriptors) - final_student_usable,
        "recurrent_source_usable": recurrent_source_usable,
        "recurrent_source_skipped": len(descriptors) - recurrent_source_usable,
        "raw_teacher_usable": sum(
            int(item["teacher_sequence_tokens"]) <= RAW_TEACHER_SEQUENCE_CAP
            for item in descriptors
        ),
        "cached_prefill_usable": sum(
            int(item["cached_prefill_tokens"]) <= CACHED_PREFILL_CAP
            for item in descriptors
        ),
    }
    return {"report_matrix": matrix, "locked_geometry": locked}


def select_worst_case_probes(
    descriptors: Sequence[Mapping[str, Any]],
    *,
    student_cap: int = STUDENT_SEQUENCE_CAP,
) -> dict[str, Any]:
    """Select stable cap-adjacent high-depth A/B/C canary descriptors."""

    cap_safe = [
        dict(item)
        for item in descriptors
        if int(item["student_sequence_tokens"]) <= student_cap
        and int(item["max_recurrent_source_sequence_tokens"]) <= student_cap
        and int(item["teacher_sequence_tokens"]) <= RAW_TEACHER_SEQUENCE_CAP
    ]
    epochs_by_descriptor = _progressive_epochs_by_descriptor(cap_safe, seed=0)
    eligible = [
        item for item in cap_safe if str(item["descriptor_id"]) in epochs_by_descriptor
    ]
    if len({str(item["trajectory_id"]) for item in eligible}) < 2:
        raise ValueError(
            "worst-case canary selection requires two cap-safe trajectories"
        )
    primary_ordered = sorted(
        eligible,
        key=lambda item: (
            student_cap - int(item["student_sequence_tokens"]),
            RAW_TEACHER_SEQUENCE_CAP - int(item["teacher_sequence_tokens"]),
            -int(item["transition_chunks"]),
            -int(item["completed_rounds"]),
            str(item["descriptor_id"]),
        ),
    )
    primary = primary_ordered[0]
    secondary_candidates = [
        item
        for item in eligible
        if str(item["trajectory_id"]) != str(primary["trajectory_id"])
    ]
    secondary = min(
        secondary_candidates,
        key=lambda item: (
            -int(item["transition_chunks"]),
            -int(item["completed_rounds"]),
            RAW_TEACHER_SEQUENCE_CAP - int(item["teacher_sequence_tokens"]),
            student_cap - int(item["student_sequence_tokens"]),
            str(item["descriptor_id"]),
        ),
    )
    if int(secondary["transition_chunks"]) < 2 or int(secondary["evicted_rounds"]) < 2:
        raise ValueError("secondary canary probe does not exercise recurrent memory")
    result = {
        "selector_contract": CANARY_SELECTOR_CONTRACT,
        "student_cap": student_cap,
        "training_selector": "progressive_monotone_depth_quantiles",
        "training_epochs": TRAINING_EPOCHS,
        "primary_descriptor_id": primary["descriptor_id"],
        "secondary_descriptor_id": secondary["descriptor_id"],
        "primary_selected_epochs": epochs_by_descriptor[str(primary["descriptor_id"])],
        "secondary_selected_epochs": epochs_by_descriptor[
            str(secondary["descriptor_id"])
        ],
        "schedule": [
            primary["descriptor_id"],
            primary["descriptor_id"],
            secondary["descriptor_id"],
        ],
        "primary": primary,
        "secondary": secondary,
    }
    result["selection_sha256"] = _canonical_sha(result)
    return result


def _progressive_epochs_by_descriptor(
    descriptors: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, list[int]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for item in descriptors:
        groups.setdefault(str(item["trajectory_id"]), []).append(item)
    selected: dict[str, list[int]] = {}
    for trajectory_id, values in groups.items():
        values.sort(
            key=lambda item: (
                int(item["completed_rounds"]),
                int(item["target_message_index"]),
                hashlib.sha256(
                    f"{seed}:{trajectory_id}:{item['descriptor_id']}".encode("utf-8")
                ).hexdigest(),
                str(item["descriptor_id"]),
            )
        )
        for epoch in range(TRAINING_EPOCHS):
            index = (epoch * (len(values) - 1)) // (TRAINING_EPOCHS - 1)
            selected.setdefault(str(values[index]["descriptor_id"]), []).append(epoch)
    return selected


def build_canary_schedule_receipt(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Pin A/B to primary and C to secondary without changing depth by seed."""

    primary_epochs = [int(value) for value in selection["primary_selected_epochs"]]
    secondary_epochs = [int(value) for value in selection["secondary_selected_epochs"]]
    if not primary_epochs or not secondary_epochs:
        raise ValueError("canary probes are not reachable by the epoch selector")
    receipt: dict[str, Any] = {
        "schema": "qwen38_wm4_progressive_canary_schedule_v1",
        "seed": 0,
        "row_schedule": [0, 0, 1],
        "epoch_schedule": [
            primary_epochs[0],
            primary_epochs[0],
            secondary_epochs[0],
        ],
        "descriptor_schedule": list(selection["schedule"]),
        "primary_trajectory_id": selection["primary"]["trajectory_id"],
        "secondary_trajectory_id": selection["secondary"]["trajectory_id"],
        "primary_descriptor_id": selection["primary_descriptor_id"],
        "secondary_descriptor_id": selection["secondary_descriptor_id"],
        "selector": "progressive_monotone_depth_quantiles",
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    return receipt


def _tokenizer_receipt(tokenizer_path: str | Path | None) -> dict[str, Any]:
    if tokenizer_path is None:
        result: dict[str, Any] = {"path": None, "files": []}
        result["receipt_sha256"] = _canonical_sha(result)
        return result
    root = Path(tokenizer_path).expanduser().resolve()
    names = ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja")
    files = []
    for name in names:
        path = root / name
        if path.is_file():
            files.append(
                {
                    "name": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _hash_file_prefix(path, path.stat().st_size),
                }
            )
    if not files:
        raise FileNotFoundError("tokenizer receipt has no tokenizer/template files")
    result = {"path": str(root), "files": files}
    result["receipt_sha256"] = _canonical_sha(result)
    return result


def build_frozen_trajectory_manifest(
    paths: Sequence[str | Path],
    tokenizer: Any,
    *,
    decontamination_index_path: str | Path,
    tokenizer_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze accepted shards and describe every decontaminated WM=4 action."""

    receipts = freeze_file_receipts(paths)
    decontamination_path = Path(decontamination_index_path).expanduser().resolve()
    if not decontamination_path.is_file():
        raise FileNotFoundError(decontamination_path)
    decontamination_payload = json.loads(decontamination_path.read_text("utf-8"))
    _validate_decontamination_payload(decontamination_payload)
    index = DecontaminationIndex.from_path(decontamination_path)

    counters: Counter[str] = Counter()
    decontamination_reasons: Counter[str] = Counter()
    excluded_ids: list[str] = []
    unique_ids: list[str] = []
    descriptors: list[dict[str, Any]] = []
    seen: set[str] = set()
    trajectory_orders: dict[str, int] = {}

    for row, receipt in _iter_receipt_rows(receipts, verify_files=True):
        counters["complete_rows"] += 1
        trajectory_id = row.get("id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            counters["missing_id_rows"] += 1
            continue
        if trajectory_id in seen:
            counters["duplicate_id_rows"] += 1
            continue
        seen.add(trajectory_id)
        trajectory_orders[trajectory_id] = len(unique_ids)
        unique_ids.append(trajectory_id)
        validation = row.get("validation")
        if not isinstance(validation, Mapping) or validation.get("valid") is not True:
            counters["nonaccepted_rows"] += 1
            continue
        counters["accepted_unique_rows"] += 1
        matches = index.screen(_screening_record(row))
        if matches:
            counters["decontamination_excluded_rows"] += 1
            excluded_ids.append(trajectory_id)
            for match in matches:
                decontamination_reasons[match.reason] += 1
            continue
        try:
            examples = iter_wm4_action_examples(
                row,
                tokenizer,
                source_receipt=receipt,
                student_cap=None,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            counters[f"adapter_rejected:{type(error).__name__}"] += 1
            continue
        if not examples:
            counters["no_wm4_action_rows"] += 1
            continue
        counters["eligible_trajectories"] += 1
        counters["eligible_actions"] += len(examples)
        for example in examples:
            descriptors.append(
                action_manifest_descriptor(
                    example,
                    trajectory_order=trajectory_orders[trajectory_id],
                )
            )

    descriptors.sort(
        key=lambda item: (
            int(item["trajectory_order"]),
            int(item["completed_rounds"]),
            int(item["target_message_index"]),
            str(item["descriptor_id"]),
        )
    )
    token_statistics = {
        name: _percentiles([int(item[name]) for item in descriptors])
        for name in (
            "prefix_tokens",
            "working_memory_tokens",
            "generation_boundary_tokens",
            "target_tokens",
            "soft_memory_tokens",
            "teacher_prefix_tokens",
            "teacher_sequence_tokens",
            "cached_prefill_tokens",
            "student_prefix_tokens",
            "student_sequence_tokens",
            "max_recurrent_source_sequence_tokens",
            "completed_rounds",
            "evicted_rounds",
            "transition_chunks",
        )
    }
    cap_coverage = _cap_coverage(descriptors)
    eligible_trajectory_count = counters["eligible_trajectories"]
    canary = (
        select_worst_case_probes(descriptors)
        if len({str(item["trajectory_id"]) for item in descriptors}) >= 2
        else None
    )
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "version": MANIFEST_VERSION,
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "dataset_kind": "accepted",
        "manifest_kind": "formal",
        "accepted_only": True,
        "row_order": "ordered_files_then_complete_jsonl_lines_first_id_wins",
        "files": [dataclasses.asdict(receipt) for receipt in receipts],
        "source_complete_bytes": sum(item.complete_bytes for item in receipts),
        "source_complete_rows": counters["complete_rows"],
        "unique_id_count": len(unique_ids),
        "unique_ids_order_sha256": _canonical_sha(unique_ids),
        "duplicate_id_rows": counters["duplicate_id_rows"],
        "trajectory_count": eligible_trajectory_count,
        "eligible_trajectory_count": eligible_trajectory_count,
        "decontamination": {
            "path": str(decontamination_path),
            "sha256": _hash_file_prefix(
                decontamination_path, decontamination_path.stat().st_size
            ),
            "excluded_count": counters["decontamination_excluded_rows"],
            "excluded_ids_sha256": _canonical_sha(sorted(excluded_ids)),
            "reason_counts": dict(sorted(decontamination_reasons.items())),
        },
        "tokenizer": _tokenizer_receipt(tokenizer_path),
        "adapter_contract": {
            "working_memory_rounds": WORKING_MEMORY_ROUNDS,
            "parallel_tool_normalization": PARALLEL_NORMALIZATION,
            "target_token_cap": TARGET_TOKEN_CAP,
            "eviction_chunk_tokens": EVICTION_CHUNK_TOKENS,
            "compression_ratio": COMPRESSION_RATIO,
            "training_epochs": TRAINING_EPOCHS,
            "trajectory_weighting": TRAJECTORY_WEIGHTING,
            "epoch_selector": "progressive_monotone_depth_quantiles",
            "generation_boundary_input_ids": list(QWEN38_GENERATION_BOUNDARY_IDS),
            "student_sequence_cap": STUDENT_SEQUENCE_CAP,
            "raw_teacher_sequence_cap": RAW_TEACHER_SEQUENCE_CAP,
            "cached_prefill_cap": CACHED_PREFILL_CAP,
            "teacher_execution_mode": TEACHER_EXECUTION_MODE,
            "round_boundary": "one_assistant_plus_one_canonical_tool_observation",
            "event_state": "final_serving_state_for_each_target",
        },
        "counts": dict(sorted(counters.items())),
        "action_descriptor_count": len(descriptors),
        "action_descriptors": descriptors,
        "token_statistics": token_statistics,
        "cap_coverage": cap_coverage,
        "canary_probe_selection": canary,
    }
    manifest["canonical_sha256"] = _canonical_sha(manifest)
    return manifest


def _validate_manifest_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("version") != (
        MANIFEST_VERSION
    ):
        raise ValueError("unsupported frozen trajectory manifest schema")
    if (
        payload.get("dataset_kind") != "accepted"
        or payload.get("accepted_only") is not True
    ):
        raise ValueError("frozen trajectory manifest is not accepted-only")
    if payload.get("manifest_kind") not in {"formal", "canary"}:
        raise ValueError("unsupported frozen trajectory manifest kind")
    if payload.get("row_order") != (
        "ordered_files_then_complete_jsonl_lines_first_id_wins"
    ):
        raise ValueError("unsupported frozen trajectory row order")
    contract = payload.get("adapter_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("manifest has no adapter contract")
    expected = {
        "working_memory_rounds": WORKING_MEMORY_ROUNDS,
        "parallel_tool_normalization": PARALLEL_NORMALIZATION,
        "target_token_cap": TARGET_TOKEN_CAP,
        "eviction_chunk_tokens": EVICTION_CHUNK_TOKENS,
        "compression_ratio": COMPRESSION_RATIO,
        "training_epochs": TRAINING_EPOCHS,
        "trajectory_weighting": TRAJECTORY_WEIGHTING,
        "epoch_selector": "progressive_monotone_depth_quantiles",
        "generation_boundary_input_ids": list(QWEN38_GENERATION_BOUNDARY_IDS),
        "student_sequence_cap": STUDENT_SEQUENCE_CAP,
        "raw_teacher_sequence_cap": RAW_TEACHER_SEQUENCE_CAP,
        "cached_prefill_cap": CACHED_PREFILL_CAP,
        "teacher_execution_mode": TEACHER_EXECUTION_MODE,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"manifest adapter contract mismatch: {key}")

    files = payload.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or not files:
        raise ValueError("manifest files must be a nonempty array")
    file_receipts = []
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("manifest file receipt must be an object")
        receipt = _receipt_from_mapping(item)
        if (
            receipt.complete_bytes < 0
            or receipt.size_bytes < receipt.complete_bytes
            or receipt.complete_line_count < 0
            or len(receipt.complete_sha256) != 64
        ):
            raise ValueError("invalid frozen file receipt")
        file_receipts.append(receipt)

    descriptors = payload.get("action_descriptors")
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, (str, bytes)):
        raise ValueError("manifest action_descriptors must be an array")
    if int(payload.get("action_descriptor_count", -1)) != len(descriptors):
        raise ValueError("manifest action descriptor count mismatch")
    descriptor_ids: set[str] = set()
    trajectory_ids: set[str] = set()
    for item in descriptors:
        if not isinstance(item, Mapping):
            raise ValueError("manifest action descriptor must be an object")
        descriptor_id = item.get("descriptor_id")
        trajectory_id = item.get("trajectory_id")
        if not isinstance(descriptor_id, str) or len(descriptor_id) != 64:
            raise ValueError("invalid action descriptor ID")
        if descriptor_id in descriptor_ids:
            raise ValueError("duplicate action descriptor ID")
        descriptor_ids.add(descriptor_id)
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("invalid descriptor trajectory ID")
        trajectory_ids.add(trajectory_id)
        if item.get("is_final_serving_state") is not True:
            raise ValueError("manifest contains a non-final serving-state action")
        completed = int(item.get("completed_rounds", -1))
        if (
            completed <= WORKING_MEMORY_ROUNDS
            or int(item.get("evicted_rounds", -1)) != completed - WORKING_MEMORY_ROUNDS
        ):
            raise ValueError("manifest descriptor has invalid WM=4 depth")
        target_tokens = int(item.get("target_tokens", -1))
        if not 0 < target_tokens <= TARGET_TOKEN_CAP:
            raise ValueError("manifest descriptor has invalid target length")
        if int(item.get("max_recurrent_source_sequence_tokens", -1)) <= 0:
            raise ValueError("manifest descriptor has invalid recurrent source length")
        supplied_descriptor_sha = item.get("descriptor_sha256")
        unsigned_descriptor = dict(item)
        unsigned_descriptor.pop("descriptor_sha256", None)
        if supplied_descriptor_sha != _canonical_sha(unsigned_descriptor):
            raise ValueError("manifest action descriptor SHA256 mismatch")
        source_receipt = item.get("source_receipt")
        if not isinstance(source_receipt, Mapping):
            raise ValueError("manifest descriptor has no source row receipt")
        file_index = int(source_receipt.get("file_index", -1))
        if not 0 <= file_index < len(file_receipts):
            raise ValueError("manifest descriptor file cursor is out of range")
        if str(source_receipt.get("path")) != file_receipts[file_index].path:
            raise ValueError("manifest descriptor path disagrees with file cursor")
        start = int(source_receipt.get("byte_start", -1))
        length = int(source_receipt.get("byte_length", -1))
        if (
            start < 0
            or length <= 0
            or (start + length > file_receipts[file_index].complete_bytes)
        ):
            raise ValueError("manifest descriptor byte cursor is out of range")
    trajectory_count = int(payload.get("trajectory_count", -1))
    if trajectory_count != len(trajectory_ids):
        raise ValueError("manifest trajectory count mismatch")
    if int(payload.get("eligible_trajectory_count", -1)) != trajectory_count:
        raise ValueError("manifest eligible trajectory count mismatch")

    decontamination = payload.get("decontamination")
    if not isinstance(decontamination, Mapping):
        raise ValueError("manifest has no decontamination receipt")
    if (
        not isinstance(decontamination.get("path"), str)
        or len(str(decontamination.get("sha256", ""))) != 64
    ):
        raise ValueError("manifest has an invalid decontamination receipt")
    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise ValueError("manifest has no tokenizer receipt")
    tokenizer_unsigned = dict(tokenizer)
    tokenizer_sha = tokenizer_unsigned.pop("receipt_sha256", None)
    if tokenizer_sha != _canonical_sha(tokenizer_unsigned):
        raise ValueError("manifest tokenizer receipt SHA256 mismatch")

    selection = payload.get("canary_probe_selection")
    if selection is not None:
        if not isinstance(selection, Mapping):
            raise ValueError("manifest canary selection must be an object")
        unsigned_selection = dict(selection)
        selection_sha = unsigned_selection.pop("selection_sha256", None)
        if selection_sha != _canonical_sha(unsigned_selection):
            raise ValueError("manifest canary selection SHA256 mismatch")
        primary_id = selection.get("primary_descriptor_id")
        secondary_id = selection.get("secondary_descriptor_id")
        if primary_id not in descriptor_ids or secondary_id not in descriptor_ids:
            raise ValueError("manifest canary descriptor is absent")
        if selection.get("schedule") != [primary_id, primary_id, secondary_id]:
            raise ValueError("manifest canary schedule must be A/A/C")
        if (
            selection["primary"]["trajectory_id"]
            == selection["secondary"]["trajectory_id"]
        ):
            raise ValueError("manifest canary probes must use two trajectories")
    schedule_selection = payload.get("canary_schedule_selection")
    if payload.get("manifest_kind") == "canary":
        if not isinstance(schedule_selection, Mapping):
            raise ValueError("canary manifest has no progressive schedule receipt")
        unsigned_schedule = dict(schedule_selection)
        schedule_sha = unsigned_schedule.pop("receipt_sha256", None)
        if schedule_sha != _canonical_sha(unsigned_schedule):
            raise ValueError("canary progressive schedule receipt SHA256 mismatch")
        if schedule_selection.get("selector") != (
            "progressive_monotone_depth_quantiles"
        ):
            raise ValueError("canary schedule uses the wrong epoch selector")
        epochs = schedule_selection.get("epoch_schedule")
        if (
            not isinstance(epochs, list)
            or len(epochs) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < TRAINING_EPOCHS
                for value in epochs
            )
        ):
            raise ValueError("canary progressive epoch schedule is invalid")
        if selection is None or (
            schedule_selection.get("primary_descriptor_id")
            != selection.get("primary_descriptor_id")
            or schedule_selection.get("secondary_descriptor_id")
            != selection.get("secondary_descriptor_id")
            or schedule_selection.get("descriptor_schedule")
            != selection.get("schedule")
            or schedule_selection.get("row_schedule") != [0, 0, 1]
        ):
            raise ValueError("canary progressive schedule disagrees with probes")

    supplied = payload.get("canonical_sha256")
    if not isinstance(supplied, str):
        raise ValueError("manifest has no canonical SHA256")
    unsigned = dict(payload)
    del unsigned["canonical_sha256"]
    if _canonical_sha(unsigned) != supplied:
        raise ValueError("manifest canonical SHA256 mismatch")


def _verify_external_receipts(payload: Mapping[str, Any]) -> None:
    """Fail closed if any source, decontamination, or tokenizer pin moved."""

    for item in payload["files"]:
        receipt = _receipt_from_mapping(item)
        path = Path(receipt.path)
        if _hash_file_prefix(path, receipt.complete_bytes) != receipt.complete_sha256:
            raise ValueError(f"frozen JSONL prefix hash mismatch: {receipt.path}")

    decontamination = payload["decontamination"]
    index_path = Path(str(decontamination["path"]))
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index_size = index_path.stat().st_size
    if _hash_file_prefix(index_path, index_size) != decontamination["sha256"]:
        raise ValueError("decontamination index SHA256 mismatch")

    tokenizer = payload["tokenizer"]
    tokenizer_path = tokenizer.get("path")
    if tokenizer_path is not None:
        root = Path(str(tokenizer_path))
        for item in tokenizer.get("files", []):
            path = root / str(item["name"])
            expected_size = int(item["size_bytes"])
            if path.stat().st_size != expected_size or _hash_file_prefix(
                path, expected_size
            ) != str(item["sha256"]):
                raise ValueError(f"tokenizer receipt mismatch: {path}")


def load_frozen_trajectory_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Load, self-verify, and optionally re-hash every frozen file prefix."""

    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frozen trajectory manifest must be an object")
    _validate_manifest_contract(payload)
    if expected_sha256 is not None and payload["canonical_sha256"] != expected_sha256:
        raise ValueError("frozen manifest does not match expected canonical SHA256")
    if verify_files:
        _verify_external_receipts(payload)
    return payload


def write_frozen_trajectory_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    """Atomically publish a validated manifest."""

    _validate_manifest_contract(payload)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def materialize_canary_manifest(
    source_manifest: Mapping[str, Any] | str | Path,
    *,
    output_data_path: str | Path,
    output_manifest_path: str | Path,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish a standalone two-row A/A/C canary snapshot.

    The data file contains the primary trajectory on row 0 and the secondary
    trajectory on row 1.  It pins every descriptor required by trainer replay
    plus explicit progressive-selector epochs for the two worst-case probes.
    Thus A and B resolve to row 0 while resume probe C resolves to row 1; no
    ordering in the formal training manifest is consulted at canary time.
    """

    payload = (
        load_frozen_trajectory_manifest(
            source_manifest,
            expected_sha256=expected_sha256,
            verify_files=True,
        )
        if isinstance(source_manifest, (str, Path))
        else dict(source_manifest)
    )
    _validate_manifest_contract(payload)
    if expected_sha256 is not None and payload["canonical_sha256"] != expected_sha256:
        raise ValueError("frozen manifest does not match expected canonical SHA256")
    if not isinstance(source_manifest, (str, Path)):
        _verify_external_receipts(payload)
    selection = payload.get("canary_probe_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("formal manifest has no two-trajectory canary selection")
    probe_source_descriptors = [
        dict(selection["primary"]),
        dict(selection["secondary"]),
    ]
    trajectory_ids = [
        str(descriptor["trajectory_id"]) for descriptor in probe_source_descriptors
    ]
    if len(set(trajectory_ids)) != 2:
        raise ValueError("canary materialization requires two distinct trajectories")

    selected_rows: dict[str, dict[str, Any]] = {}
    for frozen in iter_frozen_trajectory_rows(
        payload,
        verify_files=False,
        eligible_only=True,
    ):
        if frozen.trajectory_id in trajectory_ids:
            selected_rows[frozen.trajectory_id] = frozen.row
    if set(selected_rows) != set(trajectory_ids):
        raise ValueError("selected canary trajectory row was not found")

    data_path = Path(output_data_path).expanduser().resolve()
    manifest_path = Path(output_manifest_path).expanduser().resolve()
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError("canary output paths must not already exist")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_name(f".{data_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            for trajectory_id in trajectory_ids:
                # Native Qwen tool rendering preserves mapping insertion order.
                # Re-serializing a source row with sorted keys changes the prompt
                # even though the decoded JSON value remains equal.
                handle.write(
                    json.dumps(
                        selected_rows[trajectory_id],
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=False,
                    ).encode("utf-8")
                )
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, data_path)

        receipts = freeze_file_receipts([data_path])
        row_receipts: dict[str, FrozenRowReceipt] = {}
        for row, receipt in _iter_receipt_rows(receipts, verify_files=True):
            trajectory_id = str(row.get("id") or "")
            if trajectory_id in row_receipts:
                raise ValueError("materialized canary contains a duplicate ID")
            row_receipts[trajectory_id] = receipt

        descriptors: list[dict[str, Any]] = []
        for order, trajectory_id in enumerate(trajectory_ids):
            source_descriptors = [
                item
                for item in payload["action_descriptors"]
                if str(item["trajectory_id"]) == trajectory_id
            ]
            if not source_descriptors:
                raise ValueError("selected canary trajectory has no descriptors")
            for source_descriptor in source_descriptors:
                descriptor = dict(source_descriptor)
                descriptor["trajectory_order"] = order
                descriptor["source_receipt"] = dataclasses.asdict(
                    row_receipts[trajectory_id]
                )
                descriptors.append(_resign_descriptor(descriptor))

        token_statistics = {
            name: _percentiles([int(item[name]) for item in descriptors])
            for name in (
                "prefix_tokens",
                "working_memory_tokens",
                "generation_boundary_tokens",
                "target_tokens",
                "soft_memory_tokens",
                "teacher_prefix_tokens",
                "teacher_sequence_tokens",
                "cached_prefill_tokens",
                "student_prefix_tokens",
                "student_sequence_tokens",
                "max_recurrent_source_sequence_tokens",
                "completed_rounds",
                "evicted_rounds",
                "transition_chunks",
            )
        }
        canary_selection = select_worst_case_probes(descriptors)
        canary_schedule = build_canary_schedule_receipt(canary_selection)
        canary: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "version": MANIFEST_VERSION,
            "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "dataset_kind": "accepted",
            "manifest_kind": "canary",
            "accepted_only": True,
            "row_order": "ordered_files_then_complete_jsonl_lines_first_id_wins",
            "files": [dataclasses.asdict(receipt) for receipt in receipts],
            "source_complete_bytes": sum(
                receipt.complete_bytes for receipt in receipts
            ),
            "source_complete_rows": 2,
            "unique_id_count": 2,
            "unique_ids_order_sha256": _canonical_sha(trajectory_ids),
            "duplicate_id_rows": 0,
            "trajectory_count": 2,
            "eligible_trajectory_count": 2,
            "decontamination": {
                "path": payload["decontamination"]["path"],
                "sha256": payload["decontamination"]["sha256"],
                "excluded_count": 0,
                "excluded_ids_sha256": _canonical_sha([]),
                "reason_counts": {},
            },
            "tokenizer": dict(payload["tokenizer"]),
            "adapter_contract": dict(payload["adapter_contract"]),
            "counts": {
                "accepted_unique_rows": 2,
                "complete_rows": 2,
                "eligible_actions": len(descriptors),
                "eligible_trajectories": 2,
            },
            "action_descriptor_count": len(descriptors),
            "action_descriptors": descriptors,
            "token_statistics": token_statistics,
            "cap_coverage": _cap_coverage(descriptors),
            "canary_probe_selection": canary_selection,
            "canary_schedule_selection": canary_schedule,
            "derived_from": {
                "formal_manifest_canonical_sha256": payload["canonical_sha256"],
                "formal_selection_sha256": selection["selection_sha256"],
                "row_schedule": canary_schedule["row_schedule"],
                "epoch_schedule": canary_schedule["epoch_schedule"],
                "descriptor_schedule": canary_schedule["descriptor_schedule"],
            },
        }
        canary["canonical_sha256"] = _canonical_sha(canary)
        write_frozen_trajectory_manifest(manifest_path, canary)
        return canary
    except Exception:
        if temporary.exists():
            temporary.unlink()
        if data_path.exists() and not manifest_path.exists():
            data_path.unlink()
        raise


def _expand_inputs(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches:
            raise FileNotFoundError(f"input glob matched no files: {pattern}")
        paths.extend(matches)
    return sorted(set(path.resolve() for path in paths))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--input-glob", action="append", required=True)
    freeze.add_argument("--tokenizer", type=Path, required=True)
    freeze.add_argument("--decontamination-index", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--expected-sha256")
    describe = commands.add_parser("describe-canary")
    describe.add_argument("--manifest", type=Path, required=True)
    describe.add_argument("--expected-sha256")
    materialize = commands.add_parser("materialize-canary")
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--expected-sha256")
    materialize.add_argument("--output-data", type=Path, required=True)
    materialize.add_argument("--output-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        from transformers import AutoTokenizer

        tokenizer_path = args.tokenizer.expanduser().resolve()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        payload = build_frozen_trajectory_manifest(
            _expand_inputs(args.input_glob),
            tokenizer,
            decontamination_index_path=args.decontamination_index,
            tokenizer_path=tokenizer_path,
        )
        write_frozen_trajectory_manifest(args.output, payload)
        print(
            json.dumps(
                {
                    "manifest": str(args.output.resolve()),
                    "canonical_sha256": payload["canonical_sha256"],
                    "eligible_trajectories": payload["counts"].get(
                        "eligible_trajectories", 0
                    ),
                    "eligible_actions": payload["action_descriptor_count"],
                    "cap_coverage": payload["cap_coverage"],
                    "canary_probe_selection": payload["canary_probe_selection"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "materialize-canary":
        payload = materialize_canary_manifest(
            args.manifest,
            output_data_path=args.output_data,
            output_manifest_path=args.output_manifest,
            expected_sha256=args.expected_sha256,
        )
        print(
            json.dumps(
                {
                    "manifest": str(args.output_manifest.resolve()),
                    "data": str(args.output_data.resolve()),
                    "canonical_sha256": payload["canonical_sha256"],
                    "row_schedule": payload["derived_from"]["row_schedule"],
                    "epoch_schedule": payload["derived_from"]["epoch_schedule"],
                    "canary_seed": payload["canary_schedule_selection"]["seed"],
                    "descriptor_schedule": payload["derived_from"][
                        "descriptor_schedule"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = load_frozen_trajectory_manifest(
        args.manifest,
        expected_sha256=args.expected_sha256,
        verify_files=True,
    )
    if args.command == "describe-canary":
        print(json.dumps(payload["canary_probe_selection"], indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest.resolve()),
                    "canonical_sha256": payload["canonical_sha256"],
                    "files": len(payload["files"]),
                    "complete_rows": payload["source_complete_rows"],
                    "action_descriptors": payload["action_descriptor_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


__all__ = [
    "CANARY_SELECTOR_CONTRACT",
    "CACHED_PREFILL_CAP",
    "CANARY_SHARD_PREFIX",
    "COMPRESSION_RATIO",
    "EVICTION_CHUNK_TOKENS",
    "FrozenFileReceipt",
    "FrozenRowReceipt",
    "FrozenTrajectoryRow",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "NormalizedCodingTrajectory",
    "NormalizedServingRound",
    "PARALLEL_NORMALIZATION",
    "QWEN38_GENERATION_BOUNDARY_IDS",
    "RAW_TEACHER_SEQUENCE_CAP",
    "STUDENT_SEQUENCE_CAP",
    "ServingActionExample",
    "ServingEvictionChunk",
    "ServingEvictionFold",
    "TARGET_TOKEN_CAP",
    "TEACHER_EXECUTION_MODE",
    "TRAINING_EPOCHS",
    "TRAJECTORY_WEIGHTING",
    "WORKING_MEMORY_ROUNDS",
    "action_manifest_descriptor",
    "build_frozen_trajectory_manifest",
    "build_canary_schedule_receipt",
    "canonical_aggregate_tool_results",
    "freeze_file_receipts",
    "iter_frozen_trajectory_rows",
    "iter_frozen_action_examples",
    "iter_epoch_selected_action_examples",
    "iter_wm4_action_examples",
    "load_frozen_trajectory_manifest",
    "main",
    "materialize_canary_manifest",
    "normalize_coding_trajectory",
    "select_action_for_epoch",
    "select_worst_case_probes",
    "write_frozen_trajectory_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
