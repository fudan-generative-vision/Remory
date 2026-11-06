"""Terminus-2 with the r^2eAct 0.5.0 context policy.

This module is deliberately additive.  The released 0.4.12 implementation in
``terminus_r2eact_agent`` remains the compatibility baseline, while this module
selects a static system-only memory contract, eagerly persists compressor
events, and binds the rollout client to an explicitly attested Qwen runtime.

The Terminus control and terminal data planes are inherited unchanged: Harbor's
JSON parser, parser-feedback loop, direct verified tmux transport, controller
memory query routing, and two-step completion confirmation remain authoritative.
"""

from __future__ import annotations

import copy
import contextlib
import dataclasses
import hashlib
import json
import stat
import threading
from pathlib import Path
from typing import Any, Mapping, Sequence

from harbor.llms.base import BaseLLM

from terminal_bench_eval.r2eact_agent import (
    R2EACT_MEMORY_FILENAME,
    R2EACT_MEMORY_VIRTUAL_PATH,
)
from terminal_bench_eval.terminus_r2eact_agent import (
    TERMINUS_R2EACT_EVENTS_FILENAME,
    TERMINUS_R2EACT_NAME,
    TERMINUS_R2EACT_PROGRESS_GUARD_POLICY,
    TerminusR2EActAgent,
    TerminusR2EActLLM,
)


TERMINUS_R2EACT_V050_VERSION = "0.5.0"
TERMINUS_R2EACT_V050_NAME = TERMINUS_R2EACT_NAME
TERMINUS_R2EACT_V050_PROMPT_ABI = (
    "terminus2_json_qwen_static_memory_system_projection_v1"
)
TERMINUS_R2EACT_V050_CONTEXT_ABI = (
    "qwen_round_recursive_event_durable_conditional_guard_v1"
)
TERMINUS_R2EACT_V050_ROUND_REMINDER_POLICY = (
    "initial_system_only_no_successor_round_reminder_v1"
)
TERMINUS_R2EACT_V050_PROGRESS_GUARD_POLICY = (
    "terminal_artifact_repeat_and_stagnation_conditional_cue_v1"
)
TERMINUS_R2EACT_V050_MEMORY_SCHEMA = "contiguous_complete_assistant_tool_pairs_v1"
TERMINUS_R2EACT_V050_MEMORY_REDACTION_POLICY = (
    "omit_stdout_information_lines_preserve_status_v1"
)
TERMINUS_R2EACT_V050_COMPRESSION_EVENT_POLICY = (
    "pre_generation_messages_before_event_forward_hash_deduplicated_v1"
)
TERMINUS_R2EACT_V050_GENERATION_GUARD_POLICY = (
    "exact_periodic_tail_force_eos_same_sampling_retry_v1"
)
TERMINUS_R2EACT_V050_GENERATION_GUARD_SERVER_ABI = (
    "specforge_exact_periodic_guard_global_class_v1"
)
TERMINUS_R2EACT_V050_GENERATION_GUARD_CHECK_INTERVAL_TOKENS = 256
TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS = 1_024
TERMINUS_R2EACT_V050_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS = 256
TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_REPEAT_COUNT = 8
TERMINUS_R2EACT_V050_GENERATION_GUARD_MAX_RETRIES = 1


class RetryableSGLangInfrastructureError(RuntimeError):
    """A generation-side SGLang transport failure, not a model outcome."""


_RETRYABLE_SGLANG_TRANSPORT_EXCEPTION_TYPES = frozenset(
    {
        "BadStatusLine",
        "BrokenPipeError",
        "ConnectionAbortedError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "EOFError",
        "HTTPError",
        "IncompleteRead",
        "JSONDecodeError",
        "RemoteDisconnected",
        "ResponseNotReady",
        "TimeoutError",
        "URLError",
    }
)
_RETRYABLE_SGLANG_RUNTIME_MESSAGE_PREFIXES = (
    "sglang /generate request failed:",
    "sglang /generate returned http ",
    "sglang /generate returned a non-object response",
    "sglang /generate failed:",
    "sglang cache flush request failed:",
    "sglang cache flush returned http ",
    "sglang encode response has no ",
    "sglang encode returned ",
    "sglang health request failed:",
    "sglang health returned http ",
    "sglang server info request failed:",
    "sglang server info returned http ",
)


def _is_retryable_sglang_infrastructure_error(exc: BaseException) -> bool:
    """Classify only transport/HTTP failures at the serving boundary."""

    pending: list[BaseException] = [exc]
    observed: set[int] = set()
    while pending:
        current = pending.pop(0)
        identity = id(current)
        if identity in observed:
            continue
        observed.add(identity)
        exception_type = type(current).__name__
        message = str(current).strip().lower()
        if exception_type == "RetryableSGLangInfrastructureError":
            return True
        if exception_type in _RETRYABLE_SGLANG_TRANSPORT_EXCEPTION_TYPES:
            return True
        if exception_type == "RuntimeError" and message.startswith(
            _RETRYABLE_SGLANG_RUNTIME_MESSAGE_PREFIXES
        ):
            return True
        cause = current.__cause__
        context = current.__context__
        if isinstance(cause, BaseException):
            pending.append(cause)
        if isinstance(context, BaseException) and context is not cause:
            pending.append(context)
    return False

TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL = "r2eact-qwen38-27b-compressor-r16-step1000"
TERMINUS_R2EACT_V050_DEFAULT_MODEL_PATH = Path(
    "/inspire/ssd/project/sais-bio/public/chenbaoyou/pretrained_models/"
    "Qwen/Qwen3.8-27B"
)
TERMINUS_R2EACT_V050_DEFAULT_TARGET_FAMILY = "qwen38"
TERMINUS_R2EACT_V050_DEFAULT_CHECKPOINT_SCHEMA = (
    "qwen38_27b_r2eact_token_clock_serving_v1"
)

_PROGRESS_GUARD_MARKER = "[r^2eAct progress guard]"
_POST_DECODE_COMPRESSION_TELEMETRY_KEYS = frozenset(
    {
        # These fields do not describe the input-side archive transition.  They
        # are populated only after target decoding, so the synchronous
        # compression_finished event and the post-return safety commit are
        # intentionally allowed to disagree on them.
        "generation_degeneration_guard",
        "qwen38_generated_think_open_normalized",
        "qwen38_tool_open_without_think_close_normalized",
        "target_generation_seconds",
        # V0.5.3 may project a completed, non-action answer after target
        # decoding.  This receipt describes only that output-side admission;
        # it must not make the already-committed input-side compression event
        # appear inconsistent with the generation-return fallback.
        "oversized_answer_recovery",
    }
)
_R2EACT_FORWARD_HOOK_FACTORY = (
    "context_compression.sglang_compressor_hook:make_r2eact_compressor_hook"
)
_PROGRESS_STATE_EXACT_NAMES = frozenset(
    {
        "_no_concrete_progress_rounds",
        "_inspection_history",
        "_last_stagnation_cue_bucket",
    }
)


def _required_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _terminal_system_prompt_v050(working_memory_turns: int) -> str:
    """Render the single, stable controller policy visible to the model."""

    return f"""
r^2eAct recursive-memory policy:
- Follow the Terminal/JSON response contract in the task prompt. You may inspect,
  install, edit, build, test, and run whatever the task reasonably requires.
- The newest {working_memory_turns} completed interaction rounds normally remain
  as raw working memory. Older complete rounds are recursively
  folded from left to right. Capacity pressure or an anomalously large complete
  round may advance that whole-round cursor so the next request fits.
- Exact assistant actions and terminal evidence that leave raw context are preserved
  in this episode's private, controller-managed, read-only archive at
  {R2EACT_MEMORY_VIRTUAL_PATH}. The archive can be empty before compression.
- When an exact old command, path, delimiter, error, or result matters, retrieve it
  with one dedicated read-only shell command or pipeline that names that path
  exactly once, for example `grep -n -i -F -- 'NEEDLE'
  {R2EACT_MEMORY_VIRTUAL_PATH} | head -n 8`. Do not combine archive retrieval with
  unrelated task commands. Retrieval output is transient recall and is not copied
  recursively back into the archive.
- There is no fixed round limit. The external trial timeout is the complete
  budget. A `[r^2eAct progress guard]` observation appears only after repeated or
  stagnant terminal work; treat it as a controller cue to use known evidence,
  retrieve one precise old fact, or make/test a concrete task change.
- Keep each JSON response's analysis and plan concise. Prefer bounded inspection
  when output can be large. Terminus's tmux transport cannot observe a command's
  exit status: projected `exit_code: -1` means unavailable, not failure; explicit
  timeout state remains authoritative.
""".strip()


# Public compatibility spelling used by launchers/tests that render the prompt
# before constructing the LLM.  It intentionally resolves only to the v0.5.0
# renderer; the frozen module retains its own independent function and ABI.
def _memory_system_prompt(working_memory_turns: int) -> str:
    return _terminal_system_prompt_v050(working_memory_turns)


def _durable_memory_metadata(path: Path) -> dict[str, Any]:
    """Describe a safe durable JSONL archive using a strict round prefix.

    Individual malformed or future rows remain visible in audit counters but
    cannot advance the retrievable prefix.  A duplicate role invalidates its
    round, and a symlink/non-regular path is never exposed as queryable.
    """

    path = Path(path)
    payload = b""
    read_error: str | None = None
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise OSError("memory archive is not a safe regular file")
        payload = path.read_bytes()
    except FileNotFoundError:
        payload = b""
    except OSError as exc:
        read_error = f"{type(exc).__name__}: {exc}"

    records = [line for line in payload.splitlines() if line.strip()]
    parsed_records = 0
    max_round = 0
    roles_by_round: dict[int, set[str]] = {}
    invalid_rounds: set[int] = set()
    duplicate_role_records = 0
    for line in records:
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            continue
        if not isinstance(value, Mapping):
            continue
        parsed_records += 1
        round_value = value.get("round")
        role = value.get("role")
        if (
            not isinstance(round_value, int)
            or isinstance(round_value, bool)
            or round_value <= 0
            or role not in {"assistant", "tool"}
        ):
            continue
        max_round = max(max_round, round_value)
        roles = roles_by_round.setdefault(round_value, set())
        if str(role) in roles:
            duplicate_role_records += 1
            invalid_rounds.add(round_value)
            continue
        roles.add(str(role))

    prefix_end = 0
    while prefix_end + 1 not in invalid_rounds and roles_by_round.get(
        prefix_end + 1
    ) == {"assistant", "tool"}:
        prefix_end += 1
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "records": len(records),
        "parsed_records": parsed_records,
        "max_round": max_round,
        "complete_round_prefix_end": prefix_end,
        "complete_round_prefix_roles": ["assistant", "tool"],
        "duplicate_role_records": duplicate_role_records,
        "query_available": read_error is None,
        "snapshot_read_error": read_error,
    }


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compression_event_payload(
    event: Mapping[str, Any],
    *,
    episode_id: str,
    round_index: int,
) -> Mapping[str, Any] | None:
    if str(event.get("event", "")) != "compression_finished":
        return None
    if str(event.get("episode_id", "")) != str(episode_id):
        raise RuntimeError("compression event episode does not match generation")
    observed_round = event.get("round_index")
    if (
        isinstance(observed_round, bool)
        or not isinstance(observed_round, int)
        or observed_round != round_index
    ):
        raise RuntimeError("compression event round does not match generation")
    payload = event.get("payload")
    if not isinstance(payload, Mapping):
        raise RuntimeError("compression_finished event has no payload object")
    compression = payload.get("compression")
    if not isinstance(compression, Mapping):
        raise RuntimeError("compression_finished event has no compression object")
    return compression


@dataclasses.dataclass(frozen=True)
class _PendingProgressTransaction:
    """One rendered terminal observation awaiting native admission."""

    agent: Any
    pre_state: Mapping[str, Any]
    post_state: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    round_index: int
    observation_sha256: str
    guard_prefix: str | None


def _tool_result_with_stdout(tool_result: Any, stdout: str) -> Any:
    """Clone the released immutable ToolResult with a controller prefix.

    The redundant information_lines field is part of the validated wire
    schema.  Keep it synchronized even though SpecForge removes the duplicate
    before Qwen tokenization.
    """

    if not dataclasses.is_dataclass(tool_result) or isinstance(tool_result, type):
        raise TypeError("tool-result admission requires a dataclass result")
    field_names = {field.name for field in dataclasses.fields(tool_result)}
    if not {"stdout", "information_lines"}.issubset(field_names):
        raise TypeError("tool-result admission result has an incompatible schema")
    return dataclasses.replace(
        tool_result,
        stdout=stdout,
        information_lines=[line for line in stdout.splitlines() if line.strip()],
    )


class _MemoryPersistingGenerator:
    """Forwarding generator that makes compression memory durable first."""

    def __init__(self, generator: Any, owner: "TerminusR2EActLLMV050") -> None:
        self._generator = generator
        self._owner = owner

    @property
    def wrapped_generator(self) -> Any:
        return self._generator

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)

    def prepare_tool_result_for_next_round(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        tool_result: Any,
        episode_id: str,
        round_index: int,
    ) -> Any:
        prepare = getattr(self._generator, "prepare_tool_result_for_next_round", None)
        if not callable(prepare):
            raise RuntimeError("wrapped generator has no tool-result admission hook")

        original_stdout = getattr(tool_result, "stdout", None)
        if not isinstance(original_stdout, str):
            raise TypeError("tool-result admission stdout must be a string")
        admission_stdout, guard_prefix = (
            self._owner._prepare_progress_transaction_observation(
                round_index=round_index,
                observation=original_stdout,
            )
        )
        if admission_stdout != original_stdout:
            tool_result = _tool_result_with_stdout(tool_result, admission_stdout)
        self._owner._validate_progress_transaction_observation(
            round_index=round_index,
            observation=admission_stdout,
        )

        value = prepare(
            messages=messages,
            tool_result=tool_result,
            episode_id=episode_id,
            round_index=round_index,
        )
        prepared = value
        telemetry: Any = None
        if isinstance(value, tuple):
            if len(value) != 2:
                raise TypeError(
                    "tool-result admission must return ToolResult or "
                    "(ToolResult, telemetry)"
                )
            prepared, telemetry = value
        if telemetry is not None and not isinstance(telemetry, Mapping):
            raise TypeError("tool-result admission telemetry must be an object")
        if not isinstance(prepared, type(tool_result)):
            raise TypeError("tool-result admission returned an invalid result")
        if guard_prefix is not None:
            prepared_stdout = getattr(prepared, "stdout", None)
            if not isinstance(prepared_stdout, str) or not prepared_stdout.startswith(
                guard_prefix
            ):
                raise RuntimeError(
                    "tool-result admission cropped the r^2eAct progress guard; "
                    "refusing an unauditable generation"
                )
        self._owner._commit_progress_transaction(
            round_index=round_index,
            observation=admission_stdout,
        )
        return value

    def generate_next(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        episode_id: str,
        round_index: int,
        event_sink: Any = None,
    ) -> Any:
        generation_messages = tuple(dict(message) for message in messages)

        def sink(event: Mapping[str, Any]) -> None:
            payload = dict(event)
            compression = _compression_event_payload(
                payload,
                episode_id=episode_id,
                round_index=round_index,
            )
            if compression is not None:
                # The event describes the input-side recursive transition.  Make
                # its exact pre-generation transcript durable before observers
                # can see the event or the target decode can be cancelled.
                self._owner._commit_archive_from_compression_event(
                    messages=generation_messages,
                    compression=compression,
                    round_index=round_index,
                )
            if event_sink is not None:
                event_sink(payload)

        try:
            return self._generator.generate_next(
                generation_messages,
                episode_id=episode_id,
                round_index=round_index,
                event_sink=sink,
            )
        except BaseException as exc:
            telemetry = getattr(exc, "telemetry", None)
            if type(exc).__name__ == "DegenerateGenerationError":
                audit_event = {
                    "event": "generation_degeneration_guard_exhausted",
                    "episode_id": str(episode_id),
                    "round_index": int(round_index),
                    "payload": {
                        "exception_type": type(exc).__name__,
                        "telemetry": (
                            dict(telemetry) if isinstance(telemetry, Mapping) else None
                        ),
                    },
                }
                # Audit must not replace the typed generation failure as the
                # causal exception.
                with contextlib.suppress(BaseException):
                    if event_sink is not None:
                        event_sink(audit_event)
                    else:
                        self._owner._event_sink(audit_event)
            raise


class TerminusR2EActLLMV050(TerminusR2EActLLM):
    """0.5.0 LLM policy over the frozen 0.4.12 Terminus projection."""

    def _expected_static_system_prompt(self) -> str:
        """Return the version-specific immutable system prompt.

        Later additive harness versions may override this renderer without
        weakening v0.5.0's exact-prompt validation.
        """

        return _memory_system_prompt(self.working_memory_turns)

    def __init__(
        self,
        *,
        expected_served_model: str = TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
        expected_model_path: str | Path = TERMINUS_R2EACT_V050_DEFAULT_MODEL_PATH,
        expected_target_family: str = TERMINUS_R2EACT_V050_DEFAULT_TARGET_FAMILY,
        expected_checkpoint_schema: str = (
            TERMINUS_R2EACT_V050_DEFAULT_CHECKPOINT_SCHEMA
        ),
        sync_token_ids_across_tp: bool = True,
        **kwargs: Any,
    ) -> None:
        self.expected_served_model = _required_nonempty_string(
            expected_served_model, name="expected_served_model"
        )
        self.expected_model_path = Path(expected_model_path).expanduser().resolve()
        self.expected_target_family = _required_nonempty_string(
            expected_target_family, name="expected_target_family"
        )
        self.expected_checkpoint_schema = _required_nonempty_string(
            expected_checkpoint_schema, name="expected_checkpoint_schema"
        )
        if sync_token_ids_across_tp is not True:
            raise ValueError("sync_token_ids_across_tp must be true")
        self.sync_token_ids_across_tp = True
        self._v050_generation_guard_configured = False
        self._v050_bundle_attested = False
        self._v050_runtime_contract: dict[str, Any] | None = None
        self._v050_archive_commit_keys: dict[str, str] = {}
        self._v050_archive_commit_counts = {
            "compression_event_commits": 0,
            "generation_commit_fallbacks": 0,
            "deduplicated_generation_commits": 0,
            "failed_commits": 0,
        }
        self._v050_archive_commit_lock = threading.RLock()
        self._v050_progress_transaction_lock = threading.RLock()
        self._v050_pending_progress_transaction: _PendingProgressTransaction | None = (
            None
        )
        self._v050_progress_render_events: list[Mapping[str, Any]] | None = None
        self._v050_progress_transaction_counts = {
            "staged": 0,
            "committed": 0,
            "committed_events": 0,
        }
        super().__init__(**kwargs)
        expected_system_prompt = self._expected_static_system_prompt()
        if self.system_prompt != expected_system_prompt:
            raise ValueError(
                "v0.5.0 requires its exact static system-only memory prompt"
            )

    def begin_run(self) -> None:
        super().begin_run()
        with self._v050_archive_commit_lock:
            self._v050_archive_commit_keys.clear()
            for key in self._v050_archive_commit_counts:
                self._v050_archive_commit_counts[key] = 0
        with self._v050_progress_transaction_lock:
            self._v050_pending_progress_transaction = None
            self._v050_progress_render_events = None
            for key in self._v050_progress_transaction_counts:
                self._v050_progress_transaction_counts[key] = 0

    def _generate_sync(self, prompt: str) -> Any:
        """Give Harbor a structural marker for serving transport failures."""

        try:
            return super()._generate_sync(prompt)
        except Exception as exc:
            if _is_retryable_sglang_infrastructure_error(exc):
                raise RetryableSGLangInfrastructureError(str(exc)) from exc
            raise

    def _round_reminder(self, round_index: int) -> tuple[None, None]:
        del round_index
        return None, None

    def _begin_progress_guard_render(self) -> None:
        with self._v050_progress_transaction_lock:
            if self._v050_pending_progress_transaction is not None:
                raise RuntimeError(
                    "a progress-state transaction is already awaiting admission"
                )
            if self._v050_progress_render_events is not None:
                raise RuntimeError("progress-guard rendering is already active")
            self._v050_progress_render_events = []

    def _finish_progress_guard_render(self) -> tuple[Mapping[str, Any], ...]:
        with self._v050_progress_transaction_lock:
            events = self._v050_progress_render_events
            if events is None:
                raise RuntimeError("progress-guard rendering is not active")
            self._v050_progress_render_events = None
            return tuple(copy.deepcopy(event) for event in events)

    def _abort_progress_guard_render(self) -> None:
        with self._v050_progress_transaction_lock:
            self._v050_progress_render_events = None

    def note_progress_guard(
        self,
        *,
        trigger: str,
        round_index: int,
        payload: Mapping[str, Any],
    ) -> None:
        event = {
            "event": "terminus_progress_guard",
            "episode_id": self._episode_id,
            "round_index": int(round_index),
            "payload": {"trigger": str(trigger), **dict(payload)},
        }
        with self._v050_progress_transaction_lock:
            render_events = self._v050_progress_render_events
            if render_events is not None:
                render_events.append(copy.deepcopy(event))
                return
        # Preserve the inherited API for an explicit caller outside the
        # classifier render phase. Agent-generated cues always take the staged
        # branch above.
        super().note_progress_guard(
            trigger=trigger,
            round_index=round_index,
            payload=payload,
        )

    def _stage_progress_transaction(
        self,
        *,
        agent: Any,
        pre_state: Mapping[str, Any],
        post_state: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        round_index: int,
        observation: str,
        guard_prefix: str | None,
    ) -> None:
        if not isinstance(observation, str):
            raise TypeError("progress transaction observation must be a string")
        normalized_round = int(round_index)
        if normalized_round <= 0:
            raise ValueError("progress transaction round must be positive")
        if guard_prefix is not None:
            if not isinstance(guard_prefix, str) or not guard_prefix.startswith(
                _PROGRESS_GUARD_MARKER
            ):
                raise ValueError("progress transaction guard prefix is invalid")
            if not observation.startswith(guard_prefix):
                raise ValueError(
                    "progress transaction observation does not contain its guard prefix"
                )
        if bool(events) != bool(guard_prefix):
            raise ValueError(
                "progress transaction event and guard-prefix state disagree"
            )
        transaction = _PendingProgressTransaction(
            agent=agent,
            pre_state=copy.deepcopy(dict(pre_state)),
            post_state=copy.deepcopy(dict(post_state)),
            events=tuple(copy.deepcopy(dict(event)) for event in events),
            round_index=normalized_round,
            observation_sha256=hashlib.sha256(observation.encode("utf-8")).hexdigest(),
            guard_prefix=guard_prefix,
        )
        with self._v050_progress_transaction_lock:
            if self._v050_progress_render_events is not None:
                raise RuntimeError("cannot stage an active progress-guard render")
            if self._v050_pending_progress_transaction is not None:
                raise RuntimeError(
                    "a progress-state transaction is already awaiting admission"
                )
            self._v050_pending_progress_transaction = transaction
            self._v050_progress_transaction_counts["staged"] += 1

    @staticmethod
    def _move_guard_to_observation_prefix(observation: str, guard: str) -> str:
        """Place one staged guard outside Harbor's warning/confirmation wrapper."""

        if observation.startswith(guard):
            return observation
        marker_at = observation.find(guard)
        body = observation
        if marker_at >= 0:
            before = observation[:marker_at]
            after = observation[marker_at + len(guard) :]
            if after.startswith("\r\n\r\n"):
                after = after[4:]
            elif after.startswith("\n\n"):
                after = after[2:]
            body = before + after
        return guard if not body else guard + "\n\n" + body

    def _prepare_progress_transaction_observation(
        self,
        *,
        round_index: int,
        observation: str,
    ) -> tuple[str, str | None]:
        """Bind Harbor's final observation wrapper to the staged round.

        Harbor may add parser-warning or completion-confirmation text after the
        terminal observation was classified.  The pending singleton and round
        index are the transaction identity; the raw digest remains audit data,
        rather than an invalid byte-equality requirement across those wrappers.
        """

        if not isinstance(observation, str):
            raise TypeError("progress transaction observation must be a string")
        with self._v050_progress_transaction_lock:
            transaction = self._v050_pending_progress_transaction
            if transaction is None:
                return observation, None
            if transaction.round_index != int(round_index):
                raise RuntimeError(
                    "pending progress-state transaction does not match this round"
                )
            guard = transaction.guard_prefix
        if guard is None:
            return observation, None
        return self._move_guard_to_observation_prefix(observation, guard), guard

    def _validate_progress_transaction_observation(
        self,
        *,
        round_index: int,
        observation: str,
    ) -> None:
        with self._v050_progress_transaction_lock:
            transaction = self._v050_pending_progress_transaction
            if transaction is None:
                return
            if transaction.round_index != int(round_index):
                raise RuntimeError(
                    "pending progress-state transaction does not match this round"
                )
            if transaction.guard_prefix is not None and not observation.startswith(
                transaction.guard_prefix
            ):
                raise RuntimeError(
                    "pending progress-state guard is not the observation prefix"
                )

    def _commit_progress_transaction(
        self,
        *,
        round_index: int,
        observation: str,
    ) -> None:
        events: tuple[Mapping[str, Any], ...] = ()
        with self._v050_progress_transaction_lock:
            transaction = self._v050_pending_progress_transaction
            if transaction is None:
                return
            if transaction.round_index != int(round_index):
                raise RuntimeError(
                    "pending progress-state transaction does not match this round"
                )
            if transaction.guard_prefix is not None and not observation.startswith(
                transaction.guard_prefix
            ):
                raise RuntimeError(
                    "pending progress-state guard is not the observation prefix"
                )
            transaction.agent._commit_progress_state_transaction(
                expected_pre=transaction.pre_state,
                post_state=transaction.post_state,
            )
            events = transaction.events
            self._v050_pending_progress_transaction = None
            self._v050_progress_transaction_counts["committed"] += 1
            self._v050_progress_transaction_counts["committed_events"] += len(events)
        for event in events:
            self._event_sink(event)

    def _configure_request_cancellation(self, generator: Any) -> None:
        super()._configure_request_cancellation(generator)
        if self._v050_generation_guard_configured:
            return
        client = getattr(generator, "client", None)
        configure = getattr(client, "configure_generation_degeneration_guard", None)
        if not callable(configure):
            raise RuntimeError(
                "SpecForge lacks the required generation degeneration guard"
            )
        configure(
            True,
            check_interval_tokens=(
                TERMINUS_R2EACT_V050_GENERATION_GUARD_CHECK_INTERVAL_TOKENS
            ),
            minimum_periodic_span_tokens=(
                TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS
            ),
            maximum_period_tokens=(
                TERMINUS_R2EACT_V050_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS
            ),
            minimum_repeat_count=(
                TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_REPEAT_COUNT
            ),
            max_retries=TERMINUS_R2EACT_V050_GENERATION_GUARD_MAX_RETRIES,
        )
        expected = {
            "enabled": True,
            "check_interval_tokens": (
                TERMINUS_R2EACT_V050_GENERATION_GUARD_CHECK_INTERVAL_TOKENS
            ),
            "minimum_periodic_span_tokens": (
                TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS
            ),
            "maximum_period_tokens": (
                TERMINUS_R2EACT_V050_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS
            ),
            "minimum_repeat_count": (
                TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_REPEAT_COUNT
            ),
            "max_retries": TERMINUS_R2EACT_V050_GENERATION_GUARD_MAX_RETRIES,
        }
        # The released client retains private fields; small test doubles and
        # future clients may instead expose the effective public config.  If
        # either representation is present, require it to match exactly.
        public_observed = getattr(client, "generation_guard_config", None)
        private_names = {
            "enabled": "_generation_guard_enabled",
            "check_interval_tokens": "_generation_guard_check_interval_tokens",
            "minimum_periodic_span_tokens": (
                "_generation_guard_minimum_periodic_span_tokens"
            ),
            "maximum_period_tokens": "_generation_guard_maximum_period_tokens",
            "minimum_repeat_count": "_generation_guard_minimum_repeat_count",
            "max_retries": "_generation_guard_max_retries",
        }
        has_all_private = all(hasattr(client, name) for name in private_names.values())
        private_observed = (
            {
                key: getattr(client, attribute)
                for key, attribute in private_names.items()
            }
            if has_all_private
            else None
        )
        if isinstance(public_observed, Mapping):
            observed: Mapping[str, Any] | None = public_observed
        else:
            observed = private_observed
        if observed is not None and dict(observed) != expected:
            raise RuntimeError(
                "SpecForge generation degeneration guard did not retain its "
                "configured contract"
            )
        self._v050_generation_guard_configured = True

    @staticmethod
    def _underlying_generator(generator: Any) -> Any:
        if isinstance(generator, _MemoryPersistingGenerator):
            return generator.wrapped_generator
        return generator

    def _attest_bundle(self, bundle: Any) -> None:
        generator = self._underlying_generator(bundle.generator)
        observed_family = getattr(generator, "target_family", None)
        observed_schema = getattr(generator, "checkpoint_schema", None)
        bundle_schema = getattr(bundle, "checkpoint_schema", None)
        if observed_family != self.expected_target_family:
            raise RuntimeError(
                "r^2eAct target-family mismatch: expected "
                f"{self.expected_target_family!r}, observed {observed_family!r}"
            )
        if (
            observed_schema != self.expected_checkpoint_schema
            or bundle_schema != self.expected_checkpoint_schema
        ):
            raise RuntimeError(
                "r^2eAct checkpoint-schema mismatch: expected "
                f"{self.expected_checkpoint_schema!r}, observed generator="
                f"{observed_schema!r}, bundle={bundle_schema!r}"
            )

        bundle_target = Path(str(getattr(bundle, "target_model", ""))).expanduser()
        try:
            bundle_target = bundle_target.resolve()
        except OSError as exc:
            raise RuntimeError("could not resolve bundle target model") from exc
        if bundle_target != self.expected_model_path:
            raise RuntimeError(
                "r^2eAct bundle target-model mismatch: expected "
                f"{self.expected_model_path}, observed {bundle_target}"
            )

        client = getattr(generator, "client", None)
        server_info_method = getattr(client, "server_info", None)
        if not callable(server_info_method):
            raise RuntimeError("r^2eAct SGLang client has no server_info method")
        server_info = server_info_method()
        if not isinstance(server_info, Mapping):
            raise RuntimeError("SGLang server_info must be an object")
        served_model = server_info.get("served_model_name")
        if served_model != self.expected_served_model:
            raise RuntimeError(
                "SGLang served-model mismatch: expected "
                f"{self.expected_served_model!r}, observed {served_model!r}"
            )
        server_model_path = server_info.get("model_path")
        if not isinstance(server_model_path, str) or not server_model_path.strip():
            raise RuntimeError("SGLang server_info has no model_path")
        if Path(server_model_path).expanduser().resolve() != self.expected_model_path:
            raise RuntimeError(
                "SGLang model-path mismatch: expected "
                f"{self.expected_model_path}, observed {server_model_path}"
            )

        raw_hooks = server_info.get("forward_hooks")
        if not isinstance(raw_hooks, list):
            raise RuntimeError("SGLang server_info has no forward_hooks list")
        matching_hooks = [
            value
            for value in raw_hooks
            if isinstance(value, Mapping)
            and value.get("hook_factory") == _R2EACT_FORWARD_HOOK_FACTORY
        ]
        if len(matching_hooks) != 1:
            raise RuntimeError(
                "SGLang must expose exactly one r^2eAct compressor forward hook"
            )
        hook_config = matching_hooks[0].get("config")
        if not isinstance(hook_config, Mapping):
            raise RuntimeError("SGLang r^2eAct forward-hook config is missing")
        if hook_config.get("generation_degeneration_guard_server_abi") != (
            TERMINUS_R2EACT_V050_GENERATION_GUARD_SERVER_ABI
        ):
            raise RuntimeError("SGLang degeneration-guard ABI mismatch")
        if hook_config.get("sync_token_ids_across_tp") is not True:
            raise RuntimeError("SGLang TP token synchronization is not enabled")
        if self.checkpoint is None:
            raise RuntimeError("r^2eAct checkpoint path was not configured")
        hook_checkpoint = hook_config.get("checkpoint")
        if not isinstance(hook_checkpoint, str) or not hook_checkpoint.strip():
            raise RuntimeError("SGLang r^2eAct hook has no checkpoint path")
        expected_checkpoint = self.checkpoint.expanduser().resolve()
        if Path(hook_checkpoint).expanduser().resolve() != expected_checkpoint:
            raise RuntimeError(
                "SGLang compressor checkpoint mismatch: expected "
                f"{expected_checkpoint}, observed {hook_checkpoint}"
            )

        self._v050_runtime_contract = {
            "served_model": self.expected_served_model,
            "model_path": str(self.expected_model_path),
            "target_family": self.expected_target_family,
            "checkpoint_schema": self.expected_checkpoint_schema,
            "checkpoint": str(expected_checkpoint),
            "forward_hook_factory": _R2EACT_FORWARD_HOOK_FACTORY,
            "generation_degeneration_guard_server_abi": (
                TERMINUS_R2EACT_V050_GENERATION_GUARD_SERVER_ABI
            ),
            "sync_token_ids_across_tp": True,
        }
        self._v050_bundle_attested = True

    def _load_bundle(self) -> Any:
        # A pre-populated bundle is an explicit dependency-injection seam used
        # by deterministic unit tests. Production begins with ``None`` and is
        # always attested on the first bundle returned by the real loader.
        preloaded = self._bundle
        dependency_injected = (
            preloaded is not None
            and self._memory_runner is not None
            and self._tool_result_type is not None
            and not all(
                hasattr(preloaded, name)
                for name in ("checkpoint_schema", "target_model", "tool_runner")
            )
        )
        bundle = super()._load_bundle()
        if not dependency_injected and not self._v050_bundle_attested:
            self._attest_bundle(bundle)
        if not isinstance(bundle.generator, _MemoryPersistingGenerator):
            wrapped = _MemoryPersistingGenerator(bundle.generator, self)
            if dataclasses.is_dataclass(bundle):
                bundle = dataclasses.replace(bundle, generator=wrapped)
            else:
                setattr(bundle, "generator", wrapped)
            self._bundle = bundle
        return bundle

    def _archive_commit_key(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        round_index: int,
    ) -> str:
        return _canonical_digest(
            {
                "round_index": int(round_index),
                "messages": [dict(message) for message in messages],
            }
        )

    @staticmethod
    def _archive_transition_digest(compression: Mapping[str, Any]) -> str:
        return _canonical_digest(
            {
                key: value
                for key, value in compression.items()
                if key not in _POST_DECODE_COMPRESSION_TELEMETRY_KEYS
            }
        )

    def _commit_archive_once(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        compression: Mapping[str, Any],
        round_index: int,
        source: str,
    ) -> None:
        key = self._archive_commit_key(
            messages=messages,
            round_index=round_index,
        )
        transition_digest = self._archive_transition_digest(compression)
        with self._v050_archive_commit_lock:
            prior_transition = self._v050_archive_commit_keys.get(key)
            if prior_transition is not None:
                if prior_transition != transition_digest:
                    raise RuntimeError(
                        "compression archive transition changed between the "
                        "event-first commit and its generation fallback"
                    )
                if source == "generation_commit":
                    self._v050_archive_commit_counts[
                        "deduplicated_generation_commits"
                    ] += 1
                return
            before_revision = self._memory_revision
            super()._commit_archive(
                messages=messages,
                compression=compression,
                round_index=round_index,
            )
            if self._memory_revision <= before_revision:
                self._v050_archive_commit_counts["failed_commits"] += 1
                if source == "compression_finished":
                    raise RuntimeError(
                        "compression memory was not made durable before target "
                        "decode; refusing to forward the compression event"
                    )
                return
            self._v050_archive_commit_keys[key] = transition_digest
            counter = (
                "compression_event_commits"
                if source == "compression_finished"
                else "generation_commit_fallbacks"
            )
            self._v050_archive_commit_counts[counter] += 1

    def _commit_archive_from_compression_event(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        compression: Mapping[str, Any],
        round_index: int,
    ) -> None:
        self._commit_archive_once(
            messages=messages,
            compression=compression,
            round_index=round_index,
            source="compression_finished",
        )

    def _commit_archive(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        compression: Mapping[str, Any],
        round_index: int,
    ) -> None:
        self._commit_archive_once(
            messages=messages,
            compression=compression,
            round_index=round_index,
            source="generation_commit",
        )

    @staticmethod
    def _strict_memory_projection(payload: bytes) -> dict[str, Any]:
        lines = [line for line in payload.splitlines() if line.strip()]
        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(lines, 1):
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"memory record {line_number} is not valid UTF-8 JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"memory record {line_number} is not an object")
            records.append(value)

        for pair_index in range(0, len(records), 2):
            if pair_index + 1 >= len(records):
                raise ValueError("memory archive ends with an incomplete round")
            round_index = pair_index // 2 + 1
            for offset, expected_role in enumerate(("assistant", "tool")):
                record = records[pair_index + offset]
                where = f"memory round {round_index} {expected_role}"
                if record.get("round") != round_index:
                    raise ValueError(f"{where} has a non-contiguous round number")
                if record.get("role") != expected_role:
                    raise ValueError(f"{where} has the wrong role")
                expected_id = f"round-{round_index:04d}-{expected_role}"
                if record.get("id") != expected_id:
                    raise ValueError(f"{where} has the wrong record id")
                expected_message_index = 2 * round_index + offset
                if record.get("message_index") != expected_message_index:
                    raise ValueError(f"{where} has the wrong message index")
                if not isinstance(record.get("content"), str):
                    raise ValueError(f"{where} content is not a string")
                if not isinstance(record.get("compression_overlap"), bool):
                    raise ValueError(f"{where} compression_overlap is not boolean")
        complete_rounds = len(records) // 2
        return {
            "records": len(records),
            "parsed_records": len(records),
            "max_round": complete_rounds,
            "complete_round_prefix_end": complete_rounds,
        }

    def _persist_memory(self, *, allow_empty: bool = True) -> None:
        super()._persist_memory(allow_empty=allow_empty)
        target = self.logs_dir / R2EACT_MEMORY_FILENAME
        if target.is_file():
            target.chmod(0o600)
        payload = target.read_bytes() if target.is_file() else b""
        try:
            self._strict_memory_projection(payload)
        except ValueError as exc:
            self._memory_query_error = (
                "memory archive failed strict complete-round validation"
            )
            raise RuntimeError(self._memory_query_error) from exc

    def _memory_metadata(self) -> dict[str, Any]:
        path = self.logs_dir / R2EACT_MEMORY_FILENAME
        read_error: str | None = None
        validation_error: str | None = None
        try:
            payload = path.read_bytes() if path.is_file() else b""
        except OSError as exc:
            payload = b""
            read_error = f"{type(exc).__name__}: {exc}"
        try:
            projection = self._strict_memory_projection(payload)
        except ValueError as exc:
            projection = {
                "records": len([line for line in payload.splitlines() if line.strip()]),
                "parsed_records": 0,
                "max_round": 0,
                "complete_round_prefix_end": 0,
            }
            validation_error = f"{type(exc).__name__}: {exc}"
        query_error = self._memory_query_error
        return {
            "schema": TERMINUS_R2EACT_V050_MEMORY_SCHEMA,
            "path": str(path),
            "virtual_path": R2EACT_MEMORY_VIRTUAL_PATH,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            **projection,
            "complete_round_prefix_roles": ["assistant", "tool"],
            "revision": self._memory_revision,
            "query_available": (
                query_error is None and read_error is None and validation_error is None
            ),
            "query_error": query_error,
            "snapshot_read_error": read_error,
            "strict_validation_error": validation_error,
            "memory_query_result_archive_policy": (
                TERMINUS_R2EACT_V050_MEMORY_REDACTION_POLICY
            ),
        }

    def _response_extra(
        self,
        *,
        compression: Mapping[str, Any],
        round_index: int,
        tool_result_admission: Mapping[str, Any] | None,
        tool_observation_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        value = super()._response_extra(
            compression=compression,
            round_index=round_index,
            tool_result_admission=tool_result_admission,
            tool_observation_projection=tool_observation_projection,
        )
        value["r2eact_prompt_abi"] = TERMINUS_R2EACT_V050_PROMPT_ABI
        value["r2eact_context_abi"] = TERMINUS_R2EACT_V050_CONTEXT_ABI
        value["r2eact_successor_round_reminder"] = False
        return value

    def metadata(self) -> dict[str, Any]:
        value = super().metadata()
        value.update(
            {
                "agent": TERMINUS_R2EACT_V050_NAME,
                "agent_version": TERMINUS_R2EACT_V050_VERSION,
                "prompt_abi": TERMINUS_R2EACT_V050_PROMPT_ABI,
                "context_abi": TERMINUS_R2EACT_V050_CONTEXT_ABI,
                "system_prompt_delivery": "initial_system_message_only",
                "runtime_contract": (
                    {
                        "attested": self._v050_bundle_attested,
                        "expected_served_model": self.expected_served_model,
                        "expected_model_path": str(self.expected_model_path),
                        "expected_target_family": self.expected_target_family,
                        "expected_checkpoint_schema": (self.expected_checkpoint_schema),
                        "sync_token_ids_across_tp": (self.sync_token_ids_across_tp),
                        "observed": (
                            None
                            if self._v050_runtime_contract is None
                            else dict(self._v050_runtime_contract)
                        ),
                    }
                ),
                "generation_degeneration_guard": {
                    "configured": self._v050_generation_guard_configured,
                    "enabled": True,
                    "policy": TERMINUS_R2EACT_V050_GENERATION_GUARD_POLICY,
                    "server_abi": (TERMINUS_R2EACT_V050_GENERATION_GUARD_SERVER_ABI),
                    "check_interval_tokens": (
                        TERMINUS_R2EACT_V050_GENERATION_GUARD_CHECK_INTERVAL_TOKENS
                    ),
                    "minimum_periodic_span_tokens": (
                        TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS
                    ),
                    "maximum_period_tokens": (
                        TERMINUS_R2EACT_V050_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS
                    ),
                    "minimum_repeat_count": (
                        TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_REPEAT_COUNT
                    ),
                    "max_retries": (TERMINUS_R2EACT_V050_GENERATION_GUARD_MAX_RETRIES),
                    "sampling_unchanged_across_retries": True,
                    "sampling_policy": "unchanged_between_attempts",
                    "degenerate_attempt_admission": "discard_entire_attempt",
                },
                "round_reminder": {
                    "policy": TERMINUS_R2EACT_V050_ROUND_REMINDER_POLICY,
                    "initial_round_delivery": "static_system_policy",
                    "successor_delivery_scope": ("conditional_progress_guard_only"),
                    "model_visible_successor_round_reminder": False,
                    "configured_max_turns": self.configured_max_turns,
                },
                "memory_query_result_archive_policy": (
                    TERMINUS_R2EACT_V050_MEMORY_REDACTION_POLICY
                ),
                "compression_event_memory_persistence": {
                    "policy": TERMINUS_R2EACT_V050_COMPRESSION_EVENT_POLICY,
                    **dict(self._v050_archive_commit_counts),
                },
                "progress_state_transaction": {
                    "policy": ("render_restore_then_native_admission_commit_v1"),
                    **dict(self._v050_progress_transaction_counts),
                    "pending": (self._v050_pending_progress_transaction is not None),
                    "render_active": self._v050_progress_render_events is not None,
                },
            }
        )
        return value

    def close_sync(self, *, reason: str) -> None:
        try:
            super().close_sync(reason=reason)
        finally:
            with self._v050_progress_transaction_lock:
                self._v050_pending_progress_transaction = None
                self._v050_progress_render_events = None


class TerminusR2EActAgentV050(TerminusR2EActAgent):
    """Terminus-2 agent selecting the isolated r^2eAct 0.5.0 LLM policy."""

    @staticmethod
    def name() -> str:
        return TERMINUS_R2EACT_V050_NAME

    def version(self) -> str:
        return TERMINUS_R2EACT_V050_VERSION

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
        *,
        expected_served_model: str = TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL,
        expected_model_path: str | Path = (TERMINUS_R2EACT_V050_DEFAULT_MODEL_PATH),
        expected_target_family: str = (TERMINUS_R2EACT_V050_DEFAULT_TARGET_FAMILY),
        expected_checkpoint_schema: str = (
            TERMINUS_R2EACT_V050_DEFAULT_CHECKPOINT_SCHEMA
        ),
        sync_token_ids_across_tp: bool = True,
        **kwargs: Any,
    ) -> None:
        normalized_served_model = _required_nonempty_string(
            expected_served_model, name="expected_served_model"
        )
        resolved_model_name = model_name or normalized_served_model
        if resolved_model_name != normalized_served_model:
            raise ValueError(
                "model_name must exactly match expected_served_model for the "
                "attested v0.5.0 harness"
            )
        self._v050_expected_served_model = normalized_served_model
        self._v050_expected_model_path = (
            Path(expected_model_path).expanduser().resolve()
        )
        self._v050_expected_target_family = _required_nonempty_string(
            expected_target_family, name="expected_target_family"
        )
        self._v050_expected_checkpoint_schema = _required_nonempty_string(
            expected_checkpoint_schema, name="expected_checkpoint_schema"
        )
        if sync_token_ids_across_tp is not True:
            raise ValueError("sync_token_ids_across_tp must be true")
        self._v050_sync_token_ids_across_tp = True
        super().__init__(
            logs_dir=logs_dir,
            model_name=resolved_model_name,
            **kwargs,
        )

    def _init_llm(self, *args: Any, **kwargs: Any) -> BaseLLM:
        del args, kwargs
        bootstrap = dict(self._r2eact_bootstrap)
        bootstrap["system_prompt"] = _terminal_system_prompt_v050(
            int(bootstrap["working_memory_turns"])
        )
        bootstrap.update(
            {
                "expected_served_model": self._v050_expected_served_model,
                "expected_model_path": self._v050_expected_model_path,
                "expected_target_family": self._v050_expected_target_family,
                "expected_checkpoint_schema": (self._v050_expected_checkpoint_schema),
                "sync_token_ids_across_tp": (self._v050_sync_token_ids_across_tp),
            }
        )
        # Keep the inherited setup/receipt paths aligned with the actual LLM.
        self._r2eact_bootstrap = bootstrap
        return TerminusR2EActLLMV050(**bootstrap)

    def _progress_guard_snapshot(self) -> dict[str, Any]:
        value = super()._progress_guard_snapshot()
        if value.get("policy") != TERMINUS_R2EACT_PROGRESS_GUARD_POLICY:
            raise RuntimeError("legacy Terminus progress-guard policy changed")
        value.update(
            {
                "policy": TERMINUS_R2EACT_V050_PROGRESS_GUARD_POLICY,
                "delivery": "conditional_terminal_observation_only",
                "successor_round_reminder_dependency": False,
            }
        )
        return value

    def _progress_state_field_names(self) -> set[str]:
        return {
            name
            for name in vars(self)
            if name.startswith("_progress_") or name in _PROGRESS_STATE_EXACT_NAMES
        }

    def _capture_progress_state(self) -> dict[str, Any]:
        return {
            name: copy.deepcopy(getattr(self, name))
            for name in self._progress_state_field_names()
        }

    def _restore_progress_state(self, state: Mapping[str, Any]) -> None:
        expected_names = set(state)
        for name in self._progress_state_field_names() - expected_names:
            delattr(self, name)
        for name, value in state.items():
            setattr(self, name, copy.deepcopy(value))

    def _commit_progress_state_transaction(
        self,
        *,
        expected_pre: Mapping[str, Any],
        post_state: Mapping[str, Any],
    ) -> None:
        if self._capture_progress_state() != dict(expected_pre):
            raise RuntimeError(
                "agent progress state changed while native admission was pending"
            )
        self._restore_progress_state(post_state)

    def _apply_progress_guard(
        self,
        *,
        commands: Sequence[Any],
        output: str,
    ) -> str:
        """Keep a conditional guard at the admitted observation's front.

        The frozen policy computes the trigger and audit event.  V050 changes
        only its wire placement: the controller cue is the prefix protected by
        native context admission, while the exact terminal evidence remains at
        the suffix retained by Terminus's bounded-output projection.
        """

        pre_state = self._capture_progress_state()
        self._r2eact_llm._begin_progress_guard_render()
        try:
            try:
                rendered = super()._apply_progress_guard(
                    commands=commands,
                    output=output,
                )
                post_state = self._capture_progress_state()
                events = self._r2eact_llm._finish_progress_guard_render()
            except BaseException:
                self._r2eact_llm._abort_progress_guard_render()
                raise
        finally:
            self._restore_progress_state(pre_state)
        if events:
            marker_at = rendered.rfind(_PROGRESS_GUARD_MARKER)
            if marker_at < 0:
                raise RuntimeError(
                    "progress-guard event was staged without its model-visible cue"
                )
            guard = rendered[marker_at:].rstrip()
            terminal_output = str(output).rstrip()
            rendered = self._limit_output_length(
                guard + "\n\n" + terminal_output + ("\n" if terminal_output else "")
            )
        self._r2eact_llm._stage_progress_transaction(
            agent=self,
            pre_state=pre_state,
            post_state=post_state,
            events=events,
            round_index=max(1, int(self._n_episodes)),
            observation=rendered,
            guard_prefix=guard if events else None,
        )
        return rendered


__all__ = [
    "TERMINUS_R2EACT_V050_COMPRESSION_EVENT_POLICY",
    "TERMINUS_R2EACT_V050_CONTEXT_ABI",
    "TERMINUS_R2EACT_V050_DEFAULT_CHECKPOINT_SCHEMA",
    "TERMINUS_R2EACT_V050_DEFAULT_MODEL_PATH",
    "TERMINUS_R2EACT_V050_DEFAULT_SERVED_MODEL",
    "TERMINUS_R2EACT_V050_DEFAULT_TARGET_FAMILY",
    "TERMINUS_R2EACT_V050_GENERATION_GUARD_CHECK_INTERVAL_TOKENS",
    "TERMINUS_R2EACT_V050_GENERATION_GUARD_MAXIMUM_PERIOD_TOKENS",
    "TERMINUS_R2EACT_V050_GENERATION_GUARD_MAX_RETRIES",
    "TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_PERIODIC_SPAN_TOKENS",
    "TERMINUS_R2EACT_V050_GENERATION_GUARD_MINIMUM_REPEAT_COUNT",
    "TERMINUS_R2EACT_V050_GENERATION_GUARD_POLICY",
    "TERMINUS_R2EACT_V050_GENERATION_GUARD_SERVER_ABI",
    "TERMINUS_R2EACT_V050_MEMORY_REDACTION_POLICY",
    "TERMINUS_R2EACT_V050_MEMORY_SCHEMA",
    "TERMINUS_R2EACT_V050_NAME",
    "TERMINUS_R2EACT_V050_PROGRESS_GUARD_POLICY",
    "TERMINUS_R2EACT_V050_PROMPT_ABI",
    "TERMINUS_R2EACT_V050_ROUND_REMINDER_POLICY",
    "TERMINUS_R2EACT_V050_VERSION",
    "TerminusR2EActAgentV050",
    "TerminusR2EActLLMV050",
    "_durable_memory_metadata",
    "_memory_system_prompt",
    "_MemoryPersistingGenerator",
    "_terminal_system_prompt_v050",
]
