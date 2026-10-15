"""Strict MindForge GLM-5.2 trajectories for frontier-token CE.

The released MindForge file is a collection of complete ms-swift sessions,
not independent chat examples.  In that format an assistant response and its
following ``tool_call`` records are separate supervised roles.  This adapter
merges each such run into one native assistant message and emits one causal
decision event per assistant turn.  A decision sees only messages that existed
before it; the corresponding tool observations enter the context of later
decisions and are always masked from CE.

The release appends ``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` as harness
control.  That final call is deliberately removed from the native target while
the adjacent natural-language assistant response remains supervised.  Tool
observations containing ProgramBench's context-submit reminder are preserved
byte-for-byte by default.  A caller may provide an explicit canonicalizer, but
the transformation is then recorded in observation audit metadata.

Rendering follows autoregressive generation rather than slicing a jointly
tokenized completed chat: the native generation prompt is tokenized first and
the textual assistant continuation is encoded independently.  Every prompt
position receives ``IGNORE_INDEX`` and only frontier assistant tokens receive
CE labels.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence


MINDFORGE_DATASET_REPO = "centre-for-swe/MindForge-27B-Training-Trajectories"
MINDFORGE_DATASET_REVISION = "110407739408831dd2b093f16e3beb7ad6f54100"
MINDFORGE_DATASET_FILE = "frontier.surgical.messages_only.jsonl"
MINDFORGE_DATASET_FILE_SHA256 = (
    "d9cadedb9608a2abdd03045eccce8ca5765483f663d7ad1832ed8af703831459"
)
MINDFORGE_DATASET_FILE_BYTES = 678_427_834
MINDFORGE_DATASET_TRAJECTORIES = 1_001

MINDFORGE_CONTRACT = "mindforge-glm52-causal-frontier-ce-v2"
MINDFORGE_WEIGHTING_SCHEME = "unit-trajectory-uniform-cutpoint-v1"
MINDFORGE_MAX_SEQUENCE_TOKENS = 262_144
MINDFORGE_MAX_MESSAGES = 2_048
MINDFORGE_MAX_DECISIONS = 512
MINDFORGE_MAX_ROW_BYTES = 16 * 1024 * 1024
MINDFORGE_MAX_MESSAGE_CHARS = 8 * 1024 * 1024
MINDFORGE_MAX_TOTAL_CONTENT_CHARS = 16 * 1024 * 1024
MINDFORGE_MAX_TOOL_SCHEMA_CHARS = 1024 * 1024
MINDFORGE_HASH_CHUNK_BYTES = 8 * 1024 * 1024
IGNORE_INDEX = -100
QWEN38_ENABLE_THINKING = True
QWEN38_PRESERVE_THINKING = True

MINDFORGE_SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MINDFORGE_SUBMIT_COMMANDS = frozenset(
    {
        "cd /workspace\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        'cd /workspace && echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"',
        "cd /workspace && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        (
            "cd /workspace && rm -f fix_fnv.py fixloc.py && echo "
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        ),
        'cd /workspace; echo "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"',
        "cd /workspace; echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        MINDFORGE_SUBMIT_COMMAND,
        "rm -f /workspace/executable && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    }
)
MINDFORGE_CONTEXT_SUBMIT_REMINDER = (
    '<IMPORTANT id="programbench-context-submit-reminder">'
)
MINDFORGE_OUTPUT_LIMIT_FEEDBACK = (
    "Your previous response reached the output token limit "
    "(finish_reason=tool_calls) before you produced a tool call, so it was cut "
    "off. Respond more concisely and finish with exactly one bash tool call. If "
    "you need to think more, do so briefly."
)
MINDFORGE_OUTPUT_LIMIT_FEEDBACK_SEPARATOR = "\n\n"

# These are the only release defects canonicalized by this adapter.  The tuple
# key is (1-based line, 0-based message index, raw-payload SHA, content SHA).
MINDFORGE_TOOL_ARGUMENT_ANOMALIES = {
    (
        280,
        319,
        "3ce2bafd8a37f03409422a4872098158ddf06e30928464db47db84953dc02d57",
        "34d6dd0192456e42a95793c8b5c8e941316a4267614c68d4f4f9920458c50ed7",
    ): ("count", "description"),
    (
        934,
        486,
        "ec2440414399f7d27ff02c16842caf5d896426a3b89926de0e5c021caf610f5f",
        "fd4b610692e4d8c6779c605f0abc3ab4b01b41b784ace55220721560a2ab97dd",
    ): ("content",),
}
MINDFORGE_THINK_ANOMALY = (
    97,
    275,
    "00b1db76b082d0d535b08ecda97c00c629dd9c526137c23c0d7f90f1933b9cb3",
    "72b93d9c255e3b11cd29e4a8cc9d2ec85872e091b1ad1f52bc75269d12262133",
)
MINDFORGE_DUPLICATE_THINK_CLOSE_SUFFIX = "\n\n</think>"


ToolObservationCanonicalizer = Callable[[str], str]


class MindForgeEventTooLongError(ValueError):
    """One causal raw-oracle event exceeds the configured context bound."""


@dataclass(frozen=True)
class MindForgeEventWeight:
    """Sampling metadata whose events sum to one unit per trajectory."""

    scheme: str
    trajectory_weight: float
    cutpoint_weight: float
    event_weight: float
    cutpoint_index: int
    cutpoint_count: int


@dataclass(frozen=True)
class MindForgeObservationAudit:
    """Provenance for one exact or explicitly canonicalized tool result."""

    canonical_message_index: int
    tool_call_id: str
    raw_sha256: str
    canonical_sha256: str
    has_context_submit_reminder: bool
    has_generation_output_limit_feedback: bool
    generation_output_limit_feedback_count: int
    was_canonicalized: bool


@dataclass(frozen=True)
class MindForgeToolCallAudit:
    """One exact provider-side argument cleanup from the pinned receipt."""

    source_message_index: int
    tool_call_id: str
    dropped_argument_keys: tuple[str, ...]
    raw_content_sha256: str
    canonical_call_sha256: str


@dataclass(frozen=True)
class MindForgeReasoningAudit:
    """One exact malformed think envelope repaired from the pinned receipt."""

    source_message_index: int
    kind: str
    raw_content_sha256: str
    canonical_content_sha256: str


@dataclass(frozen=True)
class MindForgeHarnessControl:
    """One source call excluded because it belongs to the rollout harness."""

    source_message_index: int
    name: str
    arguments: dict[str, Any]
    source_sha256: str


@dataclass(frozen=True)
class MindForgeDecisionEvent:
    """One causal next-assistant target from a complete trajectory."""

    event_id: str
    source_trajectory_sha256: str
    context_sha256: str
    target_sha256: str
    tools_sha256: str
    cutpoint_index: int
    tools: tuple[dict[str, Any], ...]
    context_messages: tuple[dict[str, Any], ...]
    target_message: dict[str, Any]
    context_submit_reminder_count: int
    generation_output_limit_feedback_count: int
    is_generation_output_limit_recovery: bool
    tool_argument_cleanup_count: int
    duplicate_think_close_cleanup_count: int
    anomaly_flags: tuple[str, ...]
    weight: MindForgeEventWeight


@dataclass(frozen=True)
class MindForgeTrajectory:
    """Strict normalized trajectory plus its causal decision cutpoints."""

    source_trajectory_sha256: str
    source_line_number: int | None
    raw_jsonl_line_sha256: str | None
    raw_json_payload_sha256: str | None
    normalized_trajectory_sha256: str
    tools_sha256: str
    tools: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    events: tuple[MindForgeDecisionEvent, ...]
    observations: tuple[MindForgeObservationAudit, ...]
    tool_call_audits: tuple[MindForgeToolCallAudit, ...]
    reasoning_audits: tuple[MindForgeReasoningAudit, ...]
    harness_controls: tuple[MindForgeHarnessControl, ...]
    dropped_empty_terminal_assistant: bool
    generation_output_limit_feedback_count: int
    anomaly_flags: tuple[str, ...]


@dataclass(frozen=True)
class MindForgeRenderedEvent:
    """One Qwen3.8-native teacher-forcing sequence and its exact CE mask."""

    event_id: str
    source_trajectory_sha256: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    target_tokens: int
    target_start: int
    target_stop: int
    rendered_sha256: str
    weight: MindForgeEventWeight


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_text(text: Any, *, label: str, max_chars: int) -> Any:
    if not isinstance(text, str):
        raise ValueError(f"{label} must be a JSON string")
    if len(text) > max_chars:
        raise ValueError(f"{label} exceeds its character bound")
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
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
                raise ValueError(f"{label} contains a duplicate object key")
            copied[key] = _copy_json(item, label=label)
        return copied
    raise ValueError(f"{label} contains a non-JSON value")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _optional_sha256(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest or None")
    return value


def _context_chain_start(tools_sha256: str) -> str:
    return _digest(
        {
            "contract": MINDFORGE_CONTRACT,
            "kind": "causal-context-chain-v1",
            "tools_sha256": tools_sha256,
        }
    )


def _context_chain_append(previous: str, message: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(message)
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous))
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _context_chain_for(tools_sha256: str, messages: Sequence[Mapping[str, Any]]) -> str:
    result = _context_chain_start(tools_sha256)
    for message in messages:
        result = _context_chain_append(result, message)
    return result


def _plain_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a plain int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _normalize_tools(raw: Any) -> tuple[dict[str, Any], ...]:
    decoded = _strict_json_text(
        raw,
        label="row.tools",
        max_chars=MINDFORGE_MAX_TOOL_SCHEMA_CHARS,
    )
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("row.tools must decode to a non-empty array")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(decoded):
        label = f"row.tools[{index}]"
        if not isinstance(item, Mapping) or set(item) != {"type", "function"}:
            raise ValueError(f"{label} must contain exactly type/function")
        if item.get("type") != "function":
            raise ValueError(f"{label}.type must be 'function'")
        function = item.get("function")
        if not isinstance(function, Mapping):
            raise ValueError(f"{label}.function must be an object")
        allowed = {"name", "description", "parameters", "strict"}
        if not set(function).issubset(allowed):
            raise ValueError(f"{label}.function contains unsupported fields")
        if not {"name", "description", "parameters"}.issubset(function):
            raise ValueError(
                f"{label}.function must explicitly define name/description/parameters"
            )
        name = function.get("name")
        description = function.get("description")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}.function.name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate declared tool name: {name!r}")
        if not isinstance(description, str):
            raise ValueError(f"{label}.function.description must be a string")
        if not isinstance(parameters, Mapping) or parameters.get("type") != "object":
            raise ValueError(
                f"{label}.function.parameters must be an object JSON schema"
            )
        if "strict" in function and not isinstance(function["strict"], bool):
            raise ValueError(f"{label}.function.strict must be boolean")
        names.add(name)
        tools.append(_copy_json(item, label=label))
    return tuple(tools)


def _source_messages(raw: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("row.messages must be an array")
    if len(raw) < 3 or len(raw) > MINDFORGE_MAX_MESSAGES:
        raise ValueError("row.messages is outside the supported message bound")
    result: list[dict[str, str]] = []
    total_chars = 0
    for index, message in enumerate(raw):
        label = f"row.messages[{index}]"
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise ValueError(f"{label} must contain exactly role/content")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool_call", "tool"}:
            raise ValueError(f"{label}.role is unsupported")
        if not isinstance(content, str):
            raise ValueError(f"{label}.content must be a string")
        if len(content) > MINDFORGE_MAX_MESSAGE_CHARS:
            raise ValueError(f"{label}.content exceeds its character bound")
        total_chars += len(content)
        if total_chars > MINDFORGE_MAX_TOTAL_CONTENT_CHARS:
            raise ValueError("row message content exceeds its aggregate bound")
        result.append({"role": role, "content": content})
    if [result[0]["role"], result[1]["role"]] != ["system", "user"]:
        raise ValueError("MindForge rows must begin with exactly system then user")
    return tuple(result)


def _split_reasoning(content: str, *, label: str) -> tuple[str | None, str]:
    has_open = "<think>" in content
    has_close = "</think>" in content
    if not has_open and not has_close:
        return None, content
    if has_open != has_close:
        raise ValueError(f"{label} contains an unpaired think tag")
    if not content.startswith("<think>"):
        raise ValueError(f"{label} think envelope must start at character zero")
    if content.count("<think>") != 1 or content.count("</think>") != 1:
        raise ValueError(f"{label} must contain exactly one think envelope")
    reasoning, separator, visible = content[len("<think>") :].partition("</think>")
    if not separator:
        raise ValueError(f"{label} contains an unterminated think envelope")
    if visible == "":
        return reasoning, ""
    if not visible.startswith("\n\n"):
        raise ValueError(f"{label} think envelope must use a blank-line separator")
    return reasoning, visible[2:]


def _normalize_tool_call(
    source: Mapping[str, str],
    *,
    source_index: int,
    decision_index: int,
    call_index: int,
    declared_names: set[str],
    source_line_number: int | None,
    raw_json_payload_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], MindForgeToolCallAudit | None]:
    decoded = _strict_json_text(
        source["content"],
        label=f"row.messages[{source_index}].content",
        max_chars=MINDFORGE_MAX_MESSAGE_CHARS,
    )
    if not isinstance(decoded, Mapping) or set(decoded) != {"name", "arguments"}:
        raise ValueError(
            f"row.messages[{source_index}] tool call must contain exactly name/arguments"
        )
    name = decoded.get("name")
    arguments = decoded.get("arguments")
    if not isinstance(name, str) or not name:
        raise ValueError(f"row.messages[{source_index}] tool name is empty")
    if name not in declared_names:
        raise ValueError(f"row.messages[{source_index}] calls undeclared tool {name!r}")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"row.messages[{source_index}] arguments must be an object")
    raw_content_sha256 = hashlib.sha256(source["content"].encode("utf-8")).hexdigest()
    argument_keys = set(arguments)
    extra_keys = argument_keys - {"command"}
    audit: MindForgeToolCallAudit | None = None
    if extra_keys:
        fingerprint = (
            source_line_number,
            source_index,
            raw_json_payload_sha256,
            raw_content_sha256,
        )
        allowed = MINDFORGE_TOOL_ARGUMENT_ANOMALIES.get(fingerprint)
        if allowed is None or extra_keys != set(allowed):
            raise ValueError(
                f"row.messages[{source_index}] arguments exceed the declared schema"
            )
    if argument_keys - extra_keys != {"command"}:
        raise ValueError(f"row.messages[{source_index}] arguments must contain command")
    command = arguments.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError(f"row.messages[{source_index}] command must be non-empty")
    normalized_arguments = {"command": command}
    call_material = {"name": name, "arguments": normalized_arguments}
    call_id = (
        f"mindforge_{decision_index:04d}_{call_index:03d}_"
        f"{_digest(call_material)[:16]}"
    )
    native = {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": normalized_arguments},
    }
    if extra_keys:
        audit = MindForgeToolCallAudit(
            source_message_index=source_index,
            tool_call_id=call_id,
            dropped_argument_keys=tuple(sorted(extra_keys)),
            raw_content_sha256=raw_content_sha256,
            canonical_call_sha256=_digest(call_material),
        )
    return native, call_material, audit


def _is_submit_control(call_material: Mapping[str, Any]) -> bool:
    arguments = call_material.get("arguments")
    return (
        call_material.get("name") == "bash"
        and isinstance(arguments, Mapping)
        and set(arguments) == {"command"}
        and arguments.get("command") in MINDFORGE_SUBMIT_COMMANDS
    )


def _native_assistant(
    source: Mapping[str, str],
    *,
    source_index: int,
    tool_calls: Sequence[Mapping[str, Any]],
    source_line_number: int | None,
    raw_json_payload_sha256: str | None,
    allow_empty: bool = False,
) -> tuple[dict[str, Any], MindForgeReasoningAudit | None]:
    source_content = source["content"]
    raw_content_sha256 = hashlib.sha256(source_content.encode("utf-8")).hexdigest()
    reasoning_audit: MindForgeReasoningAudit | None = None
    fingerprint = (
        source_line_number,
        source_index,
        raw_json_payload_sha256,
        raw_content_sha256,
    )
    if fingerprint == MINDFORGE_THINK_ANOMALY:
        if (
            source_content.count("<think>") != 1
            or source_content.count("</think>") != 2
            or not source_content.endswith(MINDFORGE_DUPLICATE_THINK_CLOSE_SUFFIX)
        ):
            raise ValueError("known duplicate-think anomaly no longer matches receipt")
        canonical_content = source_content[
            : -len(MINDFORGE_DUPLICATE_THINK_CLOSE_SUFFIX)
        ]
        reasoning_audit = MindForgeReasoningAudit(
            source_message_index=source_index,
            kind="duplicate_think_close_cleaned",
            raw_content_sha256=raw_content_sha256,
            canonical_content_sha256=hashlib.sha256(
                canonical_content.encode("utf-8")
            ).hexdigest(),
        )
        source_content = canonical_content
    reasoning, content = _split_reasoning(
        source_content, label=f"row.messages[{source_index}].content"
    )
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            _copy_json(call, label=f"row.messages[{source_index}].tool_calls")
            for call in tool_calls
        ]
    if (
        not allow_empty
        and not content.strip()
        and not str(reasoning or "").strip()
        and not tool_calls
    ):
        raise ValueError(f"row.messages[{source_index}] yields an empty assistant")
    return message, reasoning_audit


def parse_mindforge_trajectory(
    row: Mapping[str, Any],
    *,
    trajectory_weight: float = 1.0,
    tool_observation_canonicalizer: ToolObservationCanonicalizer | None = None,
    source_line_number: int | None = None,
    raw_jsonl_line_sha256: str | None = None,
    raw_json_payload_sha256: str | None = None,
) -> MindForgeTrajectory:
    """Parse one pinned-format row into leak-free assistant cutpoints.

    The optional observation canonicalizer is intentionally explicit.  With
    the default ``None``, every tool observation is retained exactly.
    """

    if not isinstance(row, Mapping) or set(row) != {"messages", "tools"}:
        raise ValueError("row must contain exactly messages/tools")
    if source_line_number is not None:
        source_line_number = _plain_positive_int(
            "source_line_number", source_line_number
        )
    raw_jsonl_line_sha256 = _optional_sha256(
        "raw_jsonl_line_sha256", raw_jsonl_line_sha256
    )
    raw_json_payload_sha256 = _optional_sha256(
        "raw_json_payload_sha256", raw_json_payload_sha256
    )
    trajectory_weight = _positive_finite_float("trajectory_weight", trajectory_weight)
    source_material = _copy_json(row, label="row")
    source_trajectory_sha256 = _digest(source_material)
    tools = _normalize_tools(row.get("tools"))
    tools_list = [_copy_json(tool, label="tools") for tool in tools]
    tools_sha256 = _digest(tools_list)
    declared_names = {str(tool["function"]["name"]) for tool in tools}
    source_messages = _source_messages(row.get("messages"))

    canonical: list[dict[str, Any]] = [
        dict(source_messages[0]),
        dict(source_messages[1]),
    ]
    context_sha256 = _context_chain_start(tools_sha256)
    for message in canonical:
        context_sha256 = _context_chain_append(context_sha256, message)
    raw_events: list[
        tuple[
            tuple[dict[str, Any], ...],
            dict[str, Any],
            str,
            int,
            int,
            bool,
            int,
            int,
        ]
    ] = []
    observations: list[MindForgeObservationAudit] = []
    tool_call_audits: list[MindForgeToolCallAudit] = []
    reasoning_audits: list[MindForgeReasoningAudit] = []
    controls: list[MindForgeHarnessControl] = []
    source_index = 2
    decision_index = 0
    reminder_count = 0
    output_limit_feedback_count = 0
    preceding_output_limit_feedback_count = 0
    dropped_empty_terminal_assistant = False

    while source_index < len(source_messages):
        if decision_index >= MINDFORGE_MAX_DECISIONS:
            raise ValueError("row exceeds the supported assistant decision bound")
        assistant_index = source_index
        if source_messages[source_index]["role"] != "assistant":
            raise ValueError(
                f"row.messages[{source_index}] must begin an assistant decision"
            )
        source_index += 1

        native_calls: list[dict[str, Any]] = []
        call_materials: list[dict[str, Any]] = []
        call_source_indices: list[int] = []
        decision_tool_call_audits: list[MindForgeToolCallAudit] = []
        while (
            source_index < len(source_messages)
            and source_messages[source_index]["role"] == "tool_call"
        ):
            native_call, call_material, call_audit = _normalize_tool_call(
                source_messages[source_index],
                source_index=source_index,
                decision_index=decision_index,
                call_index=len(native_calls),
                declared_names=declared_names,
                source_line_number=source_line_number,
                raw_json_payload_sha256=raw_json_payload_sha256,
            )
            native_calls.append(native_call)
            call_materials.append(call_material)
            call_source_indices.append(source_index)
            if call_audit is not None:
                decision_tool_call_audits.append(call_audit)
            source_index += 1

        # The release's terminal submit call is rollout plumbing, not a teacher
        # action.  It is valid only as a singleton final call batch.
        if call_materials and any(_is_submit_control(call) for call in call_materials):
            if (
                len(call_materials) != 1
                or source_index != len(source_messages)
                or not _is_submit_control(call_materials[0])
            ):
                raise ValueError(
                    "submit harness control must be the singleton final call"
                )
            controls.append(
                MindForgeHarnessControl(
                    source_message_index=call_source_indices[0],
                    name=str(call_materials[0]["name"]),
                    arguments=_copy_json(
                        call_materials[0]["arguments"], label="submit arguments"
                    ),
                    source_sha256=_digest(call_materials[0]),
                )
            )
            native_calls = []
            call_materials = []
            decision_tool_call_audits = []

        target, reasoning_audit = _native_assistant(
            source_messages[assistant_index],
            source_index=assistant_index,
            tool_calls=native_calls,
            source_line_number=source_line_number,
            raw_json_payload_sha256=raw_json_payload_sha256,
            allow_empty=bool(controls),
        )
        target_is_empty = not (
            str(target.get("content", "")).strip()
            or str(target.get("reasoning_content", "")).strip()
            or target.get("tool_calls")
        )
        if controls and target_is_empty:
            if reasoning_audit is not None:
                raise ValueError("empty harness assistant unexpectedly needed cleanup")
            dropped_empty_terminal_assistant = True
            break
        event_context = tuple(canonical)
        decision_reasoning_cleanup_count = int(reasoning_audit is not None)
        if reasoning_audit is not None:
            reasoning_audits.append(reasoning_audit)
        tool_call_audits.extend(decision_tool_call_audits)
        raw_events.append(
            (
                event_context,
                _copy_json(target, label="target"),
                context_sha256,
                reminder_count,
                output_limit_feedback_count,
                preceding_output_limit_feedback_count > 0,
                len(decision_tool_call_audits),
                decision_reasoning_cleanup_count,
            )
        )
        canonical_target = _copy_json(target, label="canonical assistant")
        canonical.append(canonical_target)
        context_sha256 = _context_chain_append(context_sha256, canonical_target)
        decision_index += 1

        if controls:
            break
        if not native_calls:
            if source_index != len(source_messages):
                raise ValueError(
                    "a content-only assistant is allowed only at trajectory end"
                )
            break

        tool_start = source_index
        while (
            source_index < len(source_messages)
            and source_messages[source_index]["role"] == "tool"
        ):
            source_index += 1
        tool_sources = list(source_messages[tool_start:source_index])
        output_limit_feedback: list[str] = []
        while (
            len(tool_sources) > len(native_calls)
            and tool_sources[-1]["content"] == MINDFORGE_OUTPUT_LIMIT_FEEDBACK
        ):
            output_limit_feedback.append(tool_sources.pop()["content"])
        output_limit_feedback.reverse()
        if len(tool_sources) != len(native_calls):
            raise ValueError(
                f"assistant decision {decision_index - 1} has "
                f"{len(native_calls)} calls but {len(tool_sources)} observations"
            )
        for tool_offset, (source_tool, native_call) in enumerate(
            zip(tool_sources, native_calls, strict=True)
        ):
            raw_content = source_tool["content"]
            has_output_limit_feedback = (
                bool(output_limit_feedback) and tool_offset == len(tool_sources) - 1
            )
            if has_output_limit_feedback:
                raw_content += MINDFORGE_OUTPUT_LIMIT_FEEDBACK_SEPARATOR + (
                    MINDFORGE_OUTPUT_LIMIT_FEEDBACK_SEPARATOR.join(
                        output_limit_feedback
                    )
                )
            canonical_content = raw_content
            if tool_observation_canonicalizer is not None:
                canonical_content = tool_observation_canonicalizer(raw_content)
                if not isinstance(canonical_content, str):
                    raise ValueError(
                        "tool observation canonicalizer must return a string"
                    )
                if len(canonical_content) > MINDFORGE_MAX_MESSAGE_CHARS:
                    raise ValueError("canonicalized tool observation exceeds its bound")
            has_reminder = MINDFORGE_CONTEXT_SUBMIT_REMINDER in raw_content
            reminder_count += int(has_reminder)
            feedback_count = (
                len(output_limit_feedback) if has_output_limit_feedback else 0
            )
            output_limit_feedback_count += feedback_count
            tool_message = {
                "role": "tool",
                "name": native_call["function"]["name"],
                "tool_call_id": native_call["id"],
                "content": canonical_content,
            }
            canonical.append(tool_message)
            context_sha256 = _context_chain_append(context_sha256, tool_message)
            observations.append(
                MindForgeObservationAudit(
                    canonical_message_index=len(canonical) - 1,
                    tool_call_id=str(native_call["id"]),
                    raw_sha256=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
                    canonical_sha256=hashlib.sha256(
                        canonical_content.encode("utf-8")
                    ).hexdigest(),
                    has_context_submit_reminder=has_reminder,
                    has_generation_output_limit_feedback=has_output_limit_feedback,
                    generation_output_limit_feedback_count=feedback_count,
                    was_canonicalized=canonical_content != raw_content,
                )
            )
        preceding_output_limit_feedback_count = len(output_limit_feedback)

    if source_index != len(source_messages):
        raise ValueError("source message grammar was not consumed exactly")
    if not raw_events:
        raise ValueError("trajectory contains no assistant decisions")
    if len(controls) != 1:
        raise ValueError("pinned MindForge rows must end in one submit harness control")

    normalized_material = {
        "contract": MINDFORGE_CONTRACT,
        "tools": tools_list,
        "messages": canonical,
    }
    normalized_trajectory_sha256 = _digest(normalized_material)
    event_count = len(raw_events)
    cutpoint_weight = 1.0 / event_count
    events: list[MindForgeDecisionEvent] = []
    seen_event_ids: set[str] = set()
    for cutpoint_index, (
        context,
        target,
        causal_context_sha256,
        causal_reminders,
        causal_output_limit_feedback,
        is_output_limit_recovery,
        tool_argument_cleanup_count,
        duplicate_think_close_cleanup_count,
    ) in enumerate(raw_events):
        target_sha256 = _digest(target)
        event_anomaly_flags: list[str] = []
        if is_output_limit_recovery:
            event_anomaly_flags.append("generation_output_limit_recovery")
        if tool_argument_cleanup_count:
            event_anomaly_flags.append("tool_arguments_outside_declared_schema")
        if duplicate_think_close_cleanup_count:
            event_anomaly_flags.append("duplicate_think_close_cleaned")
        event_id = _digest(
            {
                "contract": MINDFORGE_CONTRACT,
                "cutpoint_index": cutpoint_index,
                "tools_sha256": tools_sha256,
                "context_sha256": causal_context_sha256,
                "target_sha256": target_sha256,
            }
        )
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate causal event: {event_id}")
        seen_event_ids.add(event_id)
        weight = MindForgeEventWeight(
            scheme=MINDFORGE_WEIGHTING_SCHEME,
            trajectory_weight=trajectory_weight,
            cutpoint_weight=cutpoint_weight,
            event_weight=trajectory_weight * cutpoint_weight,
            cutpoint_index=cutpoint_index,
            cutpoint_count=event_count,
        )
        events.append(
            MindForgeDecisionEvent(
                event_id=event_id,
                source_trajectory_sha256=source_trajectory_sha256,
                context_sha256=causal_context_sha256,
                target_sha256=target_sha256,
                tools_sha256=tools_sha256,
                cutpoint_index=cutpoint_index,
                tools=tuple(_copy_json(tool, label="event tools") for tool in tools),
                context_messages=context,
                target_message=target,
                context_submit_reminder_count=causal_reminders,
                generation_output_limit_feedback_count=(causal_output_limit_feedback),
                is_generation_output_limit_recovery=is_output_limit_recovery,
                tool_argument_cleanup_count=tool_argument_cleanup_count,
                duplicate_think_close_cleanup_count=(
                    duplicate_think_close_cleanup_count
                ),
                anomaly_flags=tuple(event_anomaly_flags),
                weight=weight,
            )
        )

    return MindForgeTrajectory(
        source_trajectory_sha256=source_trajectory_sha256,
        source_line_number=source_line_number,
        raw_jsonl_line_sha256=raw_jsonl_line_sha256,
        raw_json_payload_sha256=raw_json_payload_sha256,
        normalized_trajectory_sha256=normalized_trajectory_sha256,
        tools_sha256=tools_sha256,
        tools=tools,
        messages=tuple(_copy_json(message, label="messages") for message in canonical),
        events=tuple(events),
        observations=tuple(observations),
        tool_call_audits=tuple(tool_call_audits),
        reasoning_audits=tuple(reasoning_audits),
        harness_controls=tuple(controls),
        dropped_empty_terminal_assistant=dropped_empty_terminal_assistant,
        generation_output_limit_feedback_count=output_limit_feedback_count,
        anomaly_flags=tuple(
            name
            for name, present in (
                (
                    "generation_output_limit_feedback",
                    output_limit_feedback_count > 0,
                ),
                (
                    "tool_arguments_outside_declared_schema",
                    bool(tool_call_audits),
                ),
                ("duplicate_think_close_cleaned", bool(reasoning_audits)),
            )
            if present
        ),
    )


def _as_token_ids(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"{label} mapping has no input_ids")
        value = value["input_ids"]
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
    tokens: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise TypeError(f"{label} returned an invalid token id")
        tokens.append(token)
    if not tokens:
        raise ValueError(f"{label} returned no token ids")
    return tuple(tokens)


def _chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]],
    tokenize: bool,
    add_generation_prompt: bool,
) -> Any:
    return tokenizer.apply_chat_template(
        [_copy_json(message, label="chat message") for message in messages],
        tools=[_copy_json(tool, label="chat tool") for tool in tools],
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=QWEN38_ENABLE_THINKING,
        preserve_thinking=QWEN38_PRESERVE_THINKING,
        return_dict=False,
    )


def render_mindforge_decision_event(
    event: MindForgeDecisionEvent,
    tokenizer: Any,
    *,
    max_sequence_tokens: int = MINDFORGE_MAX_SEQUENCE_TOKENS,
) -> MindForgeRenderedEvent:
    """Render one cutpoint with context-only masking and frontier-token CE."""

    if not isinstance(event, MindForgeDecisionEvent):
        raise TypeError("event must be a MindForgeDecisionEvent")
    max_sequence_tokens = _plain_positive_int(
        "max_sequence_tokens", max_sequence_tokens
    )
    if max_sequence_tokens > MINDFORGE_MAX_SEQUENCE_TOKENS:
        raise ValueError("max_sequence_tokens exceeds the verified Qwen3.8 bound")
    tools_sha256 = _digest(list(event.tools))
    context_sha256 = _context_chain_for(tools_sha256, event.context_messages)
    target_sha256 = _digest(event.target_message)
    expected_event_id = _digest(
        {
            "contract": MINDFORGE_CONTRACT,
            "cutpoint_index": event.cutpoint_index,
            "tools_sha256": tools_sha256,
            "context_sha256": context_sha256,
            "target_sha256": target_sha256,
        }
    )
    if (
        tools_sha256 != event.tools_sha256
        or context_sha256 != event.context_sha256
        or target_sha256 != event.target_sha256
        or expected_event_id != event.event_id
    ):
        raise ValueError("MindForge event content does not match its causal hashes")

    prompt_ids = _as_token_ids(
        _chat_template(
            tokenizer,
            event.context_messages,
            tools=event.tools,
            tokenize=True,
            add_generation_prompt=True,
        ),
        label="Qwen3.8 generation prompt",
    )
    prompt_text = _chat_template(
        tokenizer,
        event.context_messages,
        tools=event.tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = _chat_template(
        tokenizer,
        (*event.context_messages, event.target_message),
        tools=event.tools,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("Qwen3.8 textual generation prompt is empty")
    if not isinstance(full_text, str) or not full_text.startswith(prompt_text):
        raise ValueError("Qwen3.8 completed chat is not prompt-prefix-stable")
    continuation_text = full_text[len(prompt_text) :]
    if not continuation_text:
        raise ValueError("Qwen3.8 assistant continuation is empty")
    target_ids = _as_token_ids(
        tokenizer.encode(continuation_text, add_special_tokens=False),
        label="Qwen3.8 assistant continuation",
    )
    total_tokens = len(prompt_ids) + len(target_ids)
    if total_tokens > max_sequence_tokens:
        raise MindForgeEventTooLongError(
            f"MindForge event has {total_tokens} tokens, exceeding "
            f"max_sequence_tokens={max_sequence_tokens}"
        )
    input_ids = (*prompt_ids, *target_ids)
    labels = (*(IGNORE_INDEX for _ in prompt_ids), *target_ids)
    if len(input_ids) != len(labels) or not target_ids:
        raise RuntimeError("frontier-token labels are malformed")
    rendered_sha256 = _digest(
        {"contract": MINDFORGE_CONTRACT, "input_ids": input_ids, "labels": labels}
    )
    return MindForgeRenderedEvent(
        event_id=event.event_id,
        source_trajectory_sha256=event.source_trajectory_sha256,
        input_ids=input_ids,
        labels=labels,
        prompt_tokens=len(prompt_ids),
        target_tokens=len(target_ids),
        target_start=len(prompt_ids),
        target_stop=total_tokens,
        rendered_sha256=rendered_sha256,
        weight=event.weight,
    )


def verify_mindforge_source_file(
    path: str | Path,
    *,
    expected_bytes: int = MINDFORGE_DATASET_FILE_BYTES,
    expected_sha256: str = MINDFORGE_DATASET_FILE_SHA256,
) -> str:
    """Verify a local source against the pinned immutable Hub object."""

    expected_bytes = _plain_positive_int("expected_bytes", expected_bytes)
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    source = Path(path)
    stat = source.stat()
    if not source.is_file() or stat.st_size != expected_bytes:
        raise ValueError(
            f"MindForge source size mismatch: got {stat.st_size}, "
            f"expected {expected_bytes}"
        )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(MINDFORGE_HASH_CHUNK_BYTES):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise ValueError(
            f"MindForge source SHA-256 mismatch: got {observed}, "
            f"expected {expected_sha256}"
        )
    return observed


def parse_mindforge_jsonl_line(
    raw_line: bytes,
    *,
    line_number: int,
    trajectory_weight: float = 1.0,
    tool_observation_canonicalizer: ToolObservationCanonicalizer | None = None,
) -> MindForgeTrajectory:
    """Strictly decode one bounded UTF-8 JSONL row."""

    line_number = _plain_positive_int("line_number", line_number)
    if not isinstance(raw_line, bytes):
        raise TypeError("raw_line must be bytes")
    if not raw_line or len(raw_line) > MINDFORGE_MAX_ROW_BYTES:
        raise ValueError(f"line {line_number} is empty or exceeds its byte bound")
    if not raw_line.endswith(b"\n"):
        raise ValueError(f"line {line_number} is not newline terminated")
    payload = raw_line[:-1]
    if payload.endswith(b"\r"):
        raise ValueError(f"line {line_number} uses unsupported CRLF framing")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"line {line_number} is not valid UTF-8") from error
    row = _strict_json_text(
        text,
        label=f"line {line_number}",
        max_chars=MINDFORGE_MAX_ROW_BYTES,
    )
    if not isinstance(row, Mapping):
        raise ValueError(f"line {line_number} must decode to an object")
    return parse_mindforge_trajectory(
        row,
        trajectory_weight=trajectory_weight,
        tool_observation_canonicalizer=tool_observation_canonicalizer,
        source_line_number=line_number,
        raw_jsonl_line_sha256=hashlib.sha256(raw_line).hexdigest(),
        raw_json_payload_sha256=hashlib.sha256(payload).hexdigest(),
    )


def iter_mindforge_jsonl(
    path: str | Path,
    *,
    verify_source: bool = True,
    expected_bytes: int = MINDFORGE_DATASET_FILE_BYTES,
    expected_sha256: str = MINDFORGE_DATASET_FILE_SHA256,
    expected_trajectories: int = MINDFORGE_DATASET_TRAJECTORIES,
    trajectory_weight: float = 1.0,
    tool_observation_canonicalizer: ToolObservationCanonicalizer | None = None,
) -> Iterator[MindForgeTrajectory]:
    """Stream a verified JSONL while rejecting duplicate source rows.

    Equal causal prefixes in two different trajectories are legitimate branch
    points and retain the same event ID.  Only duplicates within one parsed
    trajectory (checked by :func:`parse_mindforge_trajectory`) or an exact
    duplicate source row fail closed here.
    """

    expected_trajectories = _plain_positive_int(
        "expected_trajectories", expected_trajectories
    )
    source = Path(path)
    if verify_source:
        verify_mindforge_source_file(
            source,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
    seen_trajectories: set[str] = set()
    seen_raw_lines: set[str] = set()
    line_count = 0
    with source.open("rb") as handle:
        while raw_line := handle.readline(MINDFORGE_MAX_ROW_BYTES + 1):
            line_count += 1
            trajectory = parse_mindforge_jsonl_line(
                raw_line,
                line_number=line_count,
                trajectory_weight=trajectory_weight,
                tool_observation_canonicalizer=tool_observation_canonicalizer,
            )
            raw_line_sha256 = trajectory.raw_jsonl_line_sha256
            if raw_line_sha256 is None:
                raise RuntimeError("JSONL parser did not retain raw line identity")
            if raw_line_sha256 in seen_raw_lines:
                raise ValueError(
                    f"duplicate raw JSONL line on line {line_count}: "
                    f"{raw_line_sha256}"
                )
            if trajectory.source_trajectory_sha256 in seen_trajectories:
                raise ValueError(
                    f"duplicate source trajectory on line {line_count}: "
                    f"{trajectory.source_trajectory_sha256}"
                )
            seen_trajectories.add(trajectory.source_trajectory_sha256)
            seen_raw_lines.add(raw_line_sha256)
            yield trajectory
    if line_count != expected_trajectories:
        raise ValueError(
            f"MindForge trajectory count mismatch: got {line_count}, "
            f"expected {expected_trajectories}"
        )


__all__ = [
    "IGNORE_INDEX",
    "MINDFORGE_CONTEXT_SUBMIT_REMINDER",
    "MINDFORGE_CONTRACT",
    "MINDFORGE_DATASET_FILE",
    "MINDFORGE_DATASET_FILE_BYTES",
    "MINDFORGE_DATASET_FILE_SHA256",
    "MINDFORGE_DATASET_REPO",
    "MINDFORGE_DATASET_REVISION",
    "MINDFORGE_DATASET_TRAJECTORIES",
    "MINDFORGE_MAX_SEQUENCE_TOKENS",
    "MINDFORGE_OUTPUT_LIMIT_FEEDBACK",
    "MINDFORGE_OUTPUT_LIMIT_FEEDBACK_SEPARATOR",
    "MINDFORGE_SUBMIT_COMMAND",
    "MINDFORGE_SUBMIT_COMMANDS",
    "MINDFORGE_WEIGHTING_SCHEME",
    "MindForgeDecisionEvent",
    "MindForgeEventTooLongError",
    "MindForgeEventWeight",
    "MindForgeHarnessControl",
    "MindForgeObservationAudit",
    "MindForgeReasoningAudit",
    "MindForgeRenderedEvent",
    "MindForgeTrajectory",
    "MindForgeToolCallAudit",
    "ToolObservationCanonicalizer",
    "iter_mindforge_jsonl",
    "parse_mindforge_jsonl_line",
    "parse_mindforge_trajectory",
    "render_mindforge_decision_event",
    "verify_mindforge_source_file",
]
