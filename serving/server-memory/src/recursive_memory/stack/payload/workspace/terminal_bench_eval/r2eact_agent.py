"""Harbor agent that evaluates r^2eAct on terminal tasks.

The existing live gateway owns its own read-only Wikipedia tool runner, so it
cannot be used as a Terminal-Bench agent.  This module deliberately bypasses
that gateway while reusing the same CPU-only SGLang compressor client and
rollout state machine.  Model encode/recover requests still go to the custom
``specforge_context_compressor`` hook loaded by SGLang; shell calls are routed
to the task environment supplied by Harbor.

This adapter is intentionally narrow.  It supports the Qwen3.6 token-clock
checkpoint currently deployed with SpecForge and Harbor 0.21.x.  Both projects'
private Python APIs are checked fail-closed before a trial starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import importlib.metadata
import json
import logging
import math
import os
import re
import shlex
import sys
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence


# Keeping the protocol helpers importable without Harbor makes it possible to
# unit-test the adapter in the model-serving environment.  A real trial always
# imports the actual BaseAgent; missing dependencies in an installed Harbor are
# not hidden by this fallback.
try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext

    _HARBOR_IMPORT_ERROR: BaseException | None = None
except (ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
    missing_name = getattr(exc, "name", None)
    if missing_name not in {None, "harbor"}:
        raise
    _HARBOR_IMPORT_ERROR = exc

    class BaseAgent:  # type: ignore[no-redef]
        """Test-only stand-in used when Harbor is not installed."""

        session_id: str | None = None

        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            *args: Any,
            extra_env: dict[str, str] | None = None,
            **kwargs: Any,
        ) -> None:
            del args, kwargs
            self.logs_dir = Path(logs_dir)
            self.model_name = model_name
            self._extra_env = dict(extra_env or {})
            self.logger = logging.getLogger(__name__)

        def _get_env(self, key: str, *alternatives: str) -> str | None:
            for source in (self._extra_env, os.environ):
                for name in (key, *alternatives):
                    if name in source:
                        return source[name]
            return None

    BaseEnvironment = Any  # type: ignore[misc,assignment]
    AgentContext = Any  # type: ignore[misc,assignment]


DEFAULT_SPECFORGE_ROOT = Path(
    "/inspire/ssd/project/sais-bio/public/chenbaoyou/workspace/SpecForge"
)
DEFAULT_CHECKPOINT_RELATIVE = Path(
    "outputs/qwen36-27b-recursive-r16-step1750-serving-canary-20260820-v1"
)
DEFAULT_CORPUS_RELATIVE = Path("data/wiki-18-corpus")
DEFAULT_SGLANG_URL = "http://127.0.0.1:30000"
DEFAULT_PROCESS_LOCK = Path("/tmp/r2eact-terminal-agent-sglang.lock")
R2EACT_SESSION_PREFIX = "r2eact-"
R2EACT_START_MARKER_PREFIX = "__R2EACT_TERMINAL_START_"
R2EACT_EXIT_MARKER_PREFIX = "__R2EACT_TERMINAL_EXIT_"
R2EACT_WAIT_CHANNEL_PREFIX = "r2eact-terminal-"
R2EACT_PANE_LOG_PATH = "/logs/agent/r2eact.pane"
R2EACT_EVENTS_FILENAME = "r2eact-events.jsonl"
R2EACT_ROLLOUT_FILENAME = "r2eact-rollout.json"
R2EACT_MEMORY_FILENAME = "memory.jsonl"
R2EACT_MEMORY_VIRTUAL_PATH = "/__r2eact__/memory.jsonl"
R2EACT_CAPTURE_MAX_BYTES = 1_048_576
R2EACT_CAPTURE_SCAN_BYTES = 262_144
R2EACT_CAPTURE_FLUSH_TIMEOUT_SECONDS = 5.0
R2EACT_PROCESS_LOCK_POLL_SECONDS = 0.1
R2EACT_ADAPTER_VERSION = "0.4.1"
QWEN36_TB20_PROTOCOL_ID = "qwen36_terminal_bench_2_0_model_card_v1"
QWEN36_TB20_CONTEXT_TOKENS = 262_144
QWEN36_TB20_MAX_NEW_TOKENS = 81_920
QWEN36_TB20_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
R2EACT_POST_COMPRESSION_GENERATION_POLICY = (
    "sglang_remaining_context_experimental"
)

# Interactive pagers break the completion receipt used by TmuxSession: the
# receipt is appended after the command, so it is never reached while tools
# such as ``git log`` are waiting inside ``less``.  Terminal agents conventionally
# disable pagers for exactly this reason.  User-provided agent environment
# values still take precedence over these safe defaults.
TERMINAL_SESSION_ENV = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "SYSTEMD_PAGER": "cat",
    "MANPAGER": "cat",
}

_PATCH_LOCK = threading.Lock()
_SENTINEL_PATTERN_TEMPLATE = r"(?:^|\n){marker}:(-?\d+)(?=\r?(?:\n|$))"
_ANSI_ESCAPE_PATTERN = re.compile(
    r"\x1B(?:\][^\x07]*(?:\x07|\x1b\\)|[@-_][0-?]*[ -/]*[@-~])"
)


class _RolloutCancelled(RuntimeError):
    """Cooperative stop raised inside the synchronous rollout worker."""


@contextlib.asynccontextmanager
async def _exclusive_process_lease(
    path: Path,
    *,
    poll_seconds: float = R2EACT_PROCESS_LOCK_POLL_SECONDS,
):
    """Acquire a process flock without leaving a cancelled waiter thread.

    ``flock(LOCK_EX)`` inside ``asyncio.to_thread`` cannot be cancelled.  Poll
    non-blockingly on the event-loop side and only start the rollout worker
    after the lease is held.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    acquired = False
    try:
        while not acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                await asyncio.sleep(poll_seconds)
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def terminal_tool_schemas(tool_names: Sequence[str]) -> list[dict[str, Any]]:
    """Return the native Qwen function schema used for Terminal-Bench.

    The Terminal-Bench variant combines its persistent task terminal with the
    same controller-managed episodic archive used by the Streamlit compressed
    arm.  The archive has a reserved virtual path so a task's own
    ``memory.jsonl`` remains an ordinary task file.
    """

    names = tuple(str(name) for name in tool_names)
    if names != ("shell",):
        raise ValueError("the terminal adapter supports exactly one 'shell' tool")
    return [
        {
            "type": "function",
            "function": {
                "name": "shell",
                "description": (
                    "Execute one shell command in the persistent Linux terminal "
                    "for the current benchmark task. Commands may inspect or "
                    "modify files, run tests and builds, and invoke installed "
                    "programs. A controller-managed, read-only episodic archive "
                    f"named memory.jsonl is available at {R2EACT_MEMORY_VIRTUAL_PATH}. "
                    "It contains the complete raw assistant/tool exchanges affected "
                    "by context compression, preserves older evidence for exact "
                    "recall, and may be empty before compression first occurs. Query "
                    "that virtual path in a dedicated read-only rg/grep/find/sed/awk "
                    "pipeline; do not combine an archive lookup with task commands. "
                    "A memory.jsonl elsewhere in the task is an ordinary task file. "
                    "Return stdout, stderr, exit status, and timeout information."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The complete shell command to execute. Multiline "
                                "commands and heredocs are allowed for task work. "
                                f"To recall compressed history, use a separate "
                                f"read-only pipeline naming {R2EACT_MEMORY_VIRTUAL_PATH} "
                                "exactly once."
                            ),
                        }
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _as_bool(value: bool | str | int, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be boolean")


def _positive_int(value: int | str, *, name: str, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or (maximum is not None and result > maximum):
        suffix = f" no greater than {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be a positive integer{suffix}")
    return result


def _optional_positive_int(
    value: int | str | None,
    *,
    name: str,
) -> int | None:
    """Parse a positive integer, treating zero/None as an unlimited setting."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer or None")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer or None") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return None if result == 0 else result


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive") from exc
    if not result > 0:
        raise ValueError(f"{name} must be positive")
    return result


def _serving_capacity_metadata(
    generator: Any,
    *,
    requested_max_new_tokens: int,
) -> dict[str, Any]:
    """Describe the authoritative SGLang window without treating it as model ctx."""

    def optional_positive_attribute(name: str) -> int | None:
        value = getattr(generator, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return None
        return int(value)

    context_length = optional_positive_attribute("server_context_length")
    token_pool = optional_positive_attribute("server_max_total_num_tokens")
    maximum_input = optional_positive_attribute("server_max_req_input_len")
    total_limits = [
        value for value in (context_length, token_pool) if value is not None
    ]
    empty_prefix_capacity = (
        max(0, min(total_limits) - 2) if total_limits else None
    )
    capacity_satisfied = (
        None
        if empty_prefix_capacity is None
        else empty_prefix_capacity >= int(requested_max_new_tokens)
    )
    shortfall = (
        None
        if empty_prefix_capacity is None
        else max(0, int(requested_max_new_tokens) - empty_prefix_capacity)
    )
    return {
        "server_context_length": context_length,
        "server_max_total_num_tokens": token_pool,
        "server_max_req_input_len": maximum_input,
        "official_requested_max_new_tokens": int(requested_max_new_tokens),
        "empty_prefix_effective_max_new_tokens": (
            None
            if empty_prefix_capacity is None
            else min(int(requested_max_new_tokens), empty_prefix_capacity)
        ),
        "official_max_tokens_capacity_satisfied": capacity_satisfied,
        "official_max_tokens_capacity_shortfall_tokens": shortfall,
        "local_token_pool_is_harness_runtime": True,
    }


def _pane_delta(before: str, after: str) -> str:
    """Return the new suffix of a tmux capture without assuming line alignment."""

    limit = min(len(before), len(after))
    common = 0
    while common < limit and before[common] == after[common]:
        common += 1
    return after[common:]


def _extract_exit_sentinel(output: str, marker: str) -> tuple[str, int | None]:
    """Remove the last private exit marker and return its integer status."""

    pattern = re.compile(
        _SENTINEL_PATTERN_TEMPLATE.format(marker=re.escape(marker)), re.MULTILINE
    )
    matches = list(pattern.finditer(output.replace("\r\n", "\n")))
    if not matches:
        return output, None
    normalized = output.replace("\r\n", "\n")
    match = matches[-1]
    cleaned = normalized[: match.start()] + normalized[match.end() :]
    return cleaned.strip("\n"), int(match.group(1))


def _is_virtual_memory_query(command: str) -> bool:
    """Identify the reserved archive path without hijacking task-local files."""

    if not isinstance(command, str) or R2EACT_MEMORY_VIRTUAL_PATH not in command:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return R2EACT_MEMORY_VIRTUAL_PATH in tuple(lexer)
    except ValueError:
        # Route malformed quoting that names the reserved path to the safe
        # archive runner, which returns its normal validation observation.
        return True


def compression_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize receipts that prove whether encode/recover actually ran."""

    receipts: list[Mapping[str, Any]] = []
    output_tokens = 0
    input_tokens = 0
    generation_budgets: list[int] = []
    generation_budget_policies: set[str] = set()
    generation_budget_contracts: set[str] = set()
    post_compression_generation_policies: set[str] = set()
    tool_result_admission_policies: set[str] = set()
    untrained_long_generation_enabled_rounds = 0
    untrained_long_generation_active_rounds = 0
    checkpoint_recovery_cap_enforced_rounds = 0
    rounds_value = record.get("rounds", [])
    rounds = rounds_value if isinstance(rounds_value, list) else []
    for round_record in rounds:
        if not isinstance(round_record, Mapping):
            continue
        generation = round_record.get("generation")
        if not isinstance(generation, Mapping):
            continue
        token_count = generation.get("token_count")
        if isinstance(token_count, int) and not isinstance(token_count, bool):
            output_tokens += token_count
        compression = generation.get("compression")
        if isinstance(compression, Mapping):
            receipts.append(compression)
            prefix_tokens = compression.get("student_prefix_tokens")
            if isinstance(prefix_tokens, int) and not isinstance(prefix_tokens, bool):
                input_tokens += prefix_tokens
            generation_budget = compression.get("generation_max_new_tokens")
            if isinstance(generation_budget, int) and not isinstance(
                generation_budget, bool
            ):
                generation_budgets.append(generation_budget)
            budget_policy = compression.get("generation_budget_policy")
            if isinstance(budget_policy, str) and budget_policy:
                generation_budget_policies.add(budget_policy)
            budget_contract = compression.get("generation_budget_contract_policy")
            if isinstance(budget_contract, str) and budget_contract:
                generation_budget_contracts.add(budget_contract)
            post_compression_policy = compression.get(
                "post_compression_generation_policy"
            )
            if isinstance(post_compression_policy, str) and post_compression_policy:
                post_compression_generation_policies.add(post_compression_policy)
            admission_policy = compression.get("tool_result_admission_policy")
            if isinstance(admission_policy, str) and admission_policy:
                tool_result_admission_policies.add(admission_policy)
            untrained_long_generation_enabled_rounds += int(
                compression.get("untrained_long_generation_enabled") is True
            )
            untrained_long_generation_active_rounds += int(
                compression.get("untrained_long_generation_active") is True
            )
            checkpoint_recovery_cap_enforced_rounds += int(
                compression.get("checkpoint_recovery_cap_enforced") is True
            )

    def integer(item: Mapping[str, Any], key: str) -> int:
        value = item.get(key, 0)
        return (
            int(value) if isinstance(value, int) and not isinstance(value, bool) else 0
        )

    transition_count = sum(
        integer(item, "transitions_applied_this_round")
        or integer(item, "compressed_transition_count")
        for item in receipts
    )
    encode_rounds = sum(
        (integer(item, "transitions_applied_this_round") > 0)
        or bool(item.get("compression_triggered", False))
        for item in receipts
    )
    recover_rounds = sum(
        bool(item.get("soft_memory_active", False)) or integer(item, "memory_slots") > 0
        for item in receipts
    )
    return {
        "generation_rounds": len(receipts),
        "encode_rounds": int(encode_rounds),
        "recover_rounds": int(recover_rounds),
        "transition_count": int(transition_count),
        "compressor_observed": bool(encode_rounds and recover_rounds),
        "approximate_input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "generation_max_new_tokens_by_round": generation_budgets,
        "generation_budget_policies": sorted(generation_budget_policies),
        "generation_budget_contracts": sorted(generation_budget_contracts),
        "post_compression_generation_policies": sorted(
            post_compression_generation_policies
        ),
        "tool_result_admission_policies": sorted(
            tool_result_admission_policies
        ),
        "untrained_long_generation_enabled_rounds": (
            untrained_long_generation_enabled_rounds
        ),
        "untrained_long_generation_active_rounds": (
            untrained_long_generation_active_rounds
        ),
        "checkpoint_recovery_cap_enforced_rounds": (
            checkpoint_recovery_cap_enforced_rounds
        ),
    }


@contextlib.contextmanager
def _terminal_specforge_contract():
    """Inject only the Terminal schema while preserving Qwen's native prompt."""

    import context_compression.r2eact_live as live_module
    import context_compression.r2eact_sglang_live as sglang_live_module

    # The canonical facades deliberately expose only public r^2eAct APIs. Their
    # callable aliases still execute in the implementation modules, so patch the
    # owning modules while using the facade entry points for load/run below.
    live_contract_module = getattr(live_module, "_legacy", live_module)
    sglang_contract_module = getattr(sglang_live_module, "_legacy", sglang_live_module)
    required = (
        hasattr(sglang_contract_module, "_qwen36_tool_schemas"),
        hasattr(live_contract_module, "QWEN36_NATIVE_AGENT_SYSTEM_PROMPT"),
        hasattr(sglang_live_module, "load_sglang_live_bundle"),
        hasattr(
            sglang_live_module,
            "QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL",
        ),
        hasattr(live_module, "run_live_question"),
    )
    if not all(required):
        raise RuntimeError(
            "SpecForge live APIs changed; refusing to run with an unknown prompt ABI"
        )
    advertised_policy = getattr(
        sglang_live_module,
        "QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL",
        None,
    )
    if advertised_policy != R2EACT_POST_COMPRESSION_GENERATION_POLICY:
        raise RuntimeError(
            "SpecForge changed the full-SGLang post-compression policy wire value; "
            "refusing to run an ambiguous Terminal-Bench contract"
        )
    if live_contract_module.QWEN36_NATIVE_AGENT_SYSTEM_PROMPT != "":
        raise RuntimeError(
            "the Qwen native caller system prompt is no longer empty; refusing "
            "to silently change the Terminal-Bench prompt contract"
        )
    old_schema_builder = sglang_contract_module._qwen36_tool_schemas
    sglang_contract_module._qwen36_tool_schemas = terminal_tool_schemas
    try:
        yield live_module, sglang_live_module
    finally:
        sglang_contract_module._qwen36_tool_schemas = old_schema_builder


class HarborTerminalToolRunner:
    """Synchronous r^2eAct tool protocol backed by Harbor's async tmux shell."""

    web_search_enabled = False
    browser_enabled = False

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        session: Any,
        tool_result_type: type,
        tokenizer: Any,
        timeout_seconds: float,
        max_output_tokens: int | None,
        cancel_requested: threading.Event | None = None,
        memory_archive_path: str | Path | None = None,
    ) -> None:
        self._loop = loop
        self._session = session
        self._tool_result_type = tool_result_type
        self._tokenizer = tokenizer
        self.timeout = float(timeout_seconds)
        self.max_output_tokens = (
            None if max_output_tokens is None else int(max_output_tokens)
        )
        self.corpus_dir = Path("/")
        self._episode_id: str | None = None
        self._memory_runner: Any | None = None
        self._cancel_requested = cancel_requested or threading.Event()
        self._memory_archive_path = (
            None
            if memory_archive_path is None
            else Path(memory_archive_path).expanduser().resolve()
        )

    @property
    def memory_archive_path(self) -> Path | None:
        """Return the durable agent-artifact path, when persistence is enabled."""

        return self._memory_archive_path

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise _RolloutCancelled("Terminal-Bench rollout cancellation requested")

    def attach_memory_runner(self, runner: Any) -> None:
        """Attach SpecForge's native per-episode memory archive sidecar."""

        if self._episode_id is not None:
            raise RuntimeError("cannot attach memory during a terminal episode")
        if self._memory_runner is not None:
            raise RuntimeError("a terminal memory runner is already attached")
        required = (
            "begin_episode",
            "end_episode",
            "update_episode_memory",
            "update_compressed_episode_memory",
            "execute",
        )
        missing = [
            name for name in required if not callable(getattr(runner, name, None))
        ]
        if missing:
            raise TypeError(f"memory runner is missing callable methods: {missing}")
        self._memory_runner = runner

    def restricted(
        self,
        tools: Sequence[str],
        *,
        shell_corpus: str = "react_memory",
    ) -> "HarborTerminalToolRunner":
        if tuple(str(name) for name in tools) != ("shell",):
            raise ValueError("the Harbor terminal runner only exposes shell")
        if shell_corpus != "react_memory":
            raise ValueError("the Harbor terminal runner requires react_memory mode")
        if self._memory_runner is None:
            raise RuntimeError("the Terminal memory archive sidecar is not attached")
        return self

    def begin_episode(self, episode_id: str) -> None:
        self._check_cancelled()
        if self._episode_id is not None:
            raise RuntimeError("a terminal episode is already active")
        if self._memory_runner is None:
            raise RuntimeError("the Terminal memory archive sidecar is not attached")
        normalized = str(episode_id)
        self._memory_runner.begin_episode(normalized)
        self._episode_id = normalized

    def _persist_memory_archive(self) -> None:
        """Copy the live SpecForge archive before its episode tempdir is removed.

        SpecForge exposes the active corpus directory on its restricted runner.
        Reading that file directly preserves the exact UTF-8 JSONL bytes and
        avoids passing the artifact through shell output/token truncation.
        Episodes that finish before the archive is initialized deliberately
        persist an empty ``memory.jsonl``.
        """

        target = self._memory_archive_path
        if target is None:
            return
        if self._memory_runner is None:
            raise RuntimeError("the Terminal memory archive sidecar is not attached")
        corpus_dir = getattr(self._memory_runner, "corpus_dir", None)
        if corpus_dir is None:
            raise RuntimeError(
                "the Terminal memory archive sidecar does not expose corpus_dir"
            )
        source = Path(corpus_dir) / R2EACT_MEMORY_FILENAME
        payload = source.read_bytes() if source.is_file() else b""

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def end_episode(self, episode_id: str) -> None:
        normalized = str(episode_id)
        if self._episode_id != normalized:
            raise RuntimeError("terminal episode lifecycle mismatch")
        try:
            # The SpecForge sidecar deletes its TemporaryDirectory in
            # end_episode(), so the durable copy must happen first.  The
            # finally block still tears down the sidecar if persistence fails.
            try:
                self._persist_memory_archive()
            finally:
                self._memory_runner.end_episode(normalized)
        finally:
            self._episode_id = None

    def update_episode_memory(
        self,
        *,
        episode_id: str,
        messages: Sequence[Mapping[str, str]],
        round_index: int,
    ) -> None:
        self._check_cancelled()
        if self._episode_id != str(episode_id):
            raise RuntimeError("terminal episode lifecycle mismatch")
        self._memory_runner.update_episode_memory(
            episode_id=episode_id,
            messages=messages,
            round_index=round_index,
        )

    def update_compressed_episode_memory(
        self,
        *,
        episode_id: str,
        messages: Sequence[Mapping[str, str]],
        round_index: int,
        compression: Mapping[str, Any],
    ) -> None:
        self._check_cancelled()
        if self._episode_id != str(episode_id):
            raise RuntimeError("terminal episode lifecycle mismatch")
        self._memory_runner.update_compressed_episode_memory(
            episode_id=episode_id,
            messages=messages,
            round_index=round_index,
            compression=compression,
        )

    def _await(self, coroutine: Any, *, extra_seconds: float = 180.0) -> Any:
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=self.timeout + extra_seconds)
        except TimeoutError:
            # Do not leave an event-loop coroutine manipulating the pane after
            # the synchronous caller has already treated the bridge as failed.
            future.cancel()
            raise

    def _transport(self) -> tuple[Any, str, Any, str]:
        environment = getattr(self._session, "environment", None)
        pane_id = str(getattr(self._session, "_session_name", ""))
        user = getattr(self._session, "_user", None)
        log_path = str(getattr(self._session, "_logging_path", R2EACT_PANE_LOG_PATH))
        if environment is None or not re.fullmatch(r"%\d+", pane_id):
            raise RuntimeError(
                "Harbor tmux transport is not pinned to a stable pane id"
            )
        if not log_path.startswith("/"):
            raise RuntimeError("Harbor tmux pane log path must be absolute")
        return environment, pane_id, user, log_path

    async def _transport_exec(
        self,
        command: str,
        *,
        timeout_seconds: float = 30.0,
    ) -> Any:
        environment, _, user, _ = self._transport()
        return await environment.exec(
            command,
            timeout_sec=max(1, int(math.ceil(timeout_seconds))),
            user=user,
        )

    async def _pane_log_size(self) -> int:
        _, _, _, log_path = self._transport()
        result = await self._transport_exec(
            "wc -c < " + shlex.quote(log_path),
        )
        if result.return_code != 0:
            raise RuntimeError(
                "failed to inspect the tmux pane log: "
                + str(result.stderr or "unknown error")
            )
        try:
            size = int(str(result.stdout).strip())
        except (TypeError, ValueError) as exc:
            raise RuntimeError("tmux pane log returned an invalid byte size") from exc
        if size < 0:
            raise RuntimeError("tmux pane log returned a negative byte size")
        return size

    async def _read_pane_log(self, start: int, count: int) -> str:
        if start < 0 or count < 0:
            raise ValueError("pane log byte range must be non-negative")
        if count == 0:
            return ""
        _, _, _, log_path = self._transport()
        command = (
            "dd if="
            + shlex.quote(log_path)
            + f" bs=1 skip={start} count={count} 2>/dev/null"
        )
        result = await self._transport_exec(command)
        if result.return_code != 0:
            raise RuntimeError(
                "failed to read the tmux pane log: "
                + str(result.stderr or "unknown error")
            )
        return result.stdout or ""

    async def _wait_for_pane_log_marker(
        self,
        *,
        start: int,
        marker: str,
    ) -> tuple[int, int, bool]:
        """Wait until pipe-pane flushes a marker and return its byte window."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + R2EACT_CAPTURE_FLUSH_TIMEOUT_SECONDS
        effective_start = start
        scan_cursor = start
        overlap = ""
        reset = False
        receipt_pattern = re.compile(re.escape(marker) + r":-?\d+")
        while True:
            size = await self._pane_log_size()
            if size < effective_start:
                # The task truncated/replaced the reserved log. Read the new
                # file, but preserve an explicit truncation receipt.
                effective_start = 0
                scan_cursor = 0
                overlap = ""
                reset = True
            if size < scan_cursor:
                scan_cursor = effective_start
                overlap = ""
                reset = True
            while scan_cursor < size:
                count = min(R2EACT_CAPTURE_SCAN_BYTES, size - scan_cursor)
                chunk = await self._read_pane_log(scan_cursor, count)
                combined = overlap + chunk
                receipt = receipt_pattern.search(combined)
                if receipt is not None:
                    overlap_bytes = len(overlap.encode("utf-8"))
                    receipt_bytes = len(combined[: receipt.end()].encode("utf-8"))
                    receipt_end = scan_cursor - overlap_bytes + receipt_bytes
                    return effective_start, receipt_end, reset
                overlap = combined[-(len(marker) + 32) :]
                scan_cursor += count
            if loop.time() >= deadline:
                raise RuntimeError(
                    "terminal completion marker was not flushed to the pane log"
                )
            await asyncio.sleep(0.05)

    async def _read_capture_window(
        self,
        *,
        start: int,
        end: int,
        already_truncated: bool = False,
    ) -> tuple[str, bool]:
        if end < start:
            raise ValueError("pane log capture end precedes its start")
        total = end - start
        if total <= R2EACT_CAPTURE_MAX_BYTES:
            return await self._read_pane_log(start, total), already_truncated

        marker = (
            "\n[terminal byte capture truncated by Harbor adapter: " f"{total} bytes]\n"
        )
        available = max(2, R2EACT_CAPTURE_MAX_BYTES - len(marker.encode("utf-8")))
        head_bytes = available // 2
        tail_bytes = available - head_bytes
        head = await self._read_pane_log(start, head_bytes)
        tail = await self._read_pane_log(end - tail_bytes, tail_bytes)
        return head + marker + tail, True

    @staticmethod
    def _normalize_terminal_capture(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = _ANSI_ESCAPE_PATTERN.sub("", normalized)
        return normalized.replace("\x00", "")

    def _completed_output(
        self,
        raw: str,
        *,
        start_marker: str,
        exit_marker: str,
    ) -> tuple[str, int]:
        normalized = self._normalize_terminal_capture(raw)
        start_index = normalized.rfind(start_marker)
        if start_index < 0:
            raise RuntimeError("terminal command completed without a start receipt")
        payload = normalized[start_index + len(start_marker) :].lstrip("\n")
        pattern = re.compile(
            _SENTINEL_PATTERN_TEMPLATE.format(marker=re.escape(exit_marker)),
            re.MULTILINE,
        )
        matches = list(pattern.finditer(payload))
        if not matches:
            raise RuntimeError("terminal command completed without an exit receipt")
        match = matches[-1]
        return payload[: match.start()].strip("\n"), int(match.group(1))

    def _timed_out_output(self, raw: str, *, start_marker: str) -> str:
        normalized = self._normalize_terminal_capture(raw)
        start_index = normalized.rfind(start_marker)
        if start_index < 0:
            return normalized.strip("\n")
        return normalized[start_index + len(start_marker) :].strip("\n")

    async def _wait_channel_once(self, channel: str, seconds: float) -> Any:
        duration = max(0.05, float(seconds))
        command = f"timeout {duration:.3f}s tmux wait-for " + shlex.quote(channel)
        return await self._transport_exec(
            command,
            timeout_seconds=duration + 5.0,
        )

    async def _wait_for_command(self, channel: str) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        while True:
            self._check_cancelled()
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            result = await self._wait_channel_once(channel, min(1.0, remaining))
            if result.return_code == 0:
                return True
            if result.return_code not in {124, 137}:
                raise RuntimeError(
                    "tmux command wait failed: "
                    + str(result.stderr or f"exit {result.return_code}")
                )

    async def _initialize_respawned_shell(self) -> None:
        channel = R2EACT_WAIT_CHANNEL_PREFIX + uuid.uuid4().hex
        await self._session.send_keys(
            [f"stty -echo; tmux wait-for -S {channel}", "Enter"],
            block=False,
            min_timeout_sec=0.0,
        )
        result = await self._wait_channel_once(channel, 10.0)
        if result.return_code != 0:
            raise RuntimeError("failed to initialize the respawned tmux shell")

    async def _respawn_pane(self) -> None:
        """Hard-reset a timed-out pane so late commands cannot poison it."""

        _, pane_id, _, log_path = self._transport()
        path_result = await self._transport_exec(
            "tmux display-message -p -t "
            + shlex.quote(pane_id)
            + " '#{pane_current_path}'"
        )
        current_path = str(path_result.stdout or "").strip()
        if path_result.return_code != 0 or not current_path.startswith("/"):
            current_path = "/"
        respawn = await self._transport_exec(
            "tmux respawn-pane -k -t "
            + shlex.quote(pane_id)
            + " -c "
            + shlex.quote(current_path)
            + " "
            + shlex.quote("bash --login")
        )
        if respawn.return_code != 0:
            raise RuntimeError(
                "failed to recover the timed-out tmux pane: "
                + str(respawn.stderr or "unknown error")
            )
        pipe_command = "cat >> " + shlex.quote(log_path)
        pipe_result = await self._transport_exec(
            "tmux pipe-pane -t "
            + shlex.quote(pane_id)
            + " "
            + shlex.quote(pipe_command)
        )
        if pipe_result.return_code != 0:
            raise RuntimeError("failed to restore tmux pane logging after timeout")
        await self._initialize_respawned_shell()

    async def _execute_in_tmux(
        self,
        command: str,
    ) -> tuple[str, int, bool, bool]:
        self._check_cancelled()
        capture_start = await self._pane_log_size()
        nonce = uuid.uuid4().hex
        start_marker = R2EACT_START_MARKER_PREFIX + nonce + "__"
        exit_marker = R2EACT_EXIT_MARKER_PREFIX + nonce + "__"
        channel = R2EACT_WAIT_CHANNEL_PREFIX + nonce
        wrapped = (
            "{\n"
            + f"printf '\\n{start_marker}\\n'\n"
            + command
            + "\n__r2eact_terminal_rc=$?\n"
            + f"printf '\\n{exit_marker}:%s\\n' \"$__r2eact_terminal_rc\"\n"
            + "tmux wait-for -S "
            + shlex.quote(channel)
            + "\n}"
        )
        await self._session.send_keys(
            [wrapped, "Enter"],
            block=False,
            min_timeout_sec=0.0,
        )
        try:
            completed = await self._wait_for_command(channel)
        except _RolloutCancelled:
            await self._respawn_pane()
            raise

        if completed:
            effective_start, capture_end, reset = await self._wait_for_pane_log_marker(
                start=capture_start,
                marker=exit_marker,
            )
            raw, byte_truncated = await self._read_capture_window(
                start=effective_start,
                end=capture_end,
                already_truncated=reset,
            )
            stdout, exit_code = self._completed_output(
                raw,
                start_marker=start_marker,
                exit_marker=exit_marker,
            )
            return stdout, exit_code, False, byte_truncated

        capture_end = await self._pane_log_size()
        effective_start = capture_start
        reset = False
        if capture_end < effective_start:
            effective_start = 0
            reset = True
        raw, byte_truncated = await self._read_capture_window(
            start=effective_start,
            end=capture_end,
            already_truncated=reset,
        )
        stdout = self._timed_out_output(raw, start_marker=start_marker)
        await self._respawn_pane()
        return stdout, -1, True, byte_truncated

    def _truncate(self, text: str) -> tuple[str, bool]:
        if self.max_output_tokens is None:
            return text, False
        token_ids = list(self._tokenizer.encode(text, add_special_tokens=False))
        if len(token_ids) <= self.max_output_tokens:
            return text, False
        marker = "\n[terminal output truncated by Harbor adapter]\n"
        marker_ids = list(self._tokenizer.encode(marker, add_special_tokens=False))
        available = max(2, self.max_output_tokens - len(marker_ids))
        head = available // 2
        tail = available - head
        kept = token_ids[:head] + marker_ids + token_ids[-tail:]
        return self._tokenizer.decode(kept, skip_special_tokens=False), True

    def _result(
        self,
        *,
        command: str,
        stdout: str,
        stderr: str,
        exit_code: int,
        timed_out: bool,
        truncated: bool,
    ) -> Any:
        return self._tool_result_type(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=int(exit_code),
            timed_out=bool(timed_out),
            truncated=bool(truncated),
            information_lines=[line for line in stdout.splitlines() if line.strip()],
        )

    def execute(self, command: str) -> Any:
        self._check_cancelled()
        if self._episode_id is None:
            raise RuntimeError("shell execution attempted outside an episode")
        if not isinstance(command, str) or not command.strip():
            return self._result(
                command=str(command),
                stdout="",
                stderr="validation error: shell command must be a nonempty string",
                exit_code=-2,
                timed_out=False,
                truncated=False,
            )
        if _is_virtual_memory_query(command):
            translated = command.replace(
                R2EACT_MEMORY_VIRTUAL_PATH,
                R2EACT_MEMORY_FILENAME,
            )
            result = self._memory_runner.execute(translated)
            return replace(result, command=command)
        stdout, exit_code, timed_out, byte_truncated = self._await(
            self._execute_in_tmux(command)
        )
        stdout, token_truncated = self._truncate(stdout)
        return self._result(
            command=command,
            stdout=stdout,
            stderr=(
                f"command timed out after {self.timeout:g} seconds" if timed_out else ""
            ),
            exit_code=exit_code,
            timed_out=timed_out,
            truncated=byte_truncated or token_truncated,
        )


class R2EActAgent(BaseAgent):
    """Harbor 0.21 agent for recursive-ReAct Terminal-Bench rollouts."""

    SUPPORTS_ATIF = False

    @staticmethod
    def name() -> str:
        return "r2eact"

    def version(self) -> str:
        return R2EACT_ADAPTER_VERSION

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        specforge_root: str | Path | None = None,
        checkpoint: str | Path | None = None,
        corpus_dir: str | Path | None = None,
        sglang_url: str = DEFAULT_SGLANG_URL,
        max_rounds: int | str | None = None,
        max_new_tokens: int | str = QWEN36_TB20_MAX_NEW_TOKENS,
        remaining_context_output_ceiling_tokens: int | str | None = (
            QWEN36_TB20_MAX_NEW_TOKENS
        ),
        working_memory_turns: int | str = 3,
        command_timeout: float | str = 180.0,
        request_timeout: float | str = 1_800.0,
        max_tool_output_tokens: int | str | None = 0,
        use_remaining_context_for_generation: bool | str | int = True,
        post_compression_generation_policy: str = (
            R2EACT_POST_COMPRESSION_GENERATION_POLICY
        ),
        process_lock: str | Path = DEFAULT_PROCESS_LOCK,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            logs_dir=logs_dir,
            model_name=model_name,
            extra_env=extra_env,
            **kwargs,
        )
        self._specforge_root_value = specforge_root
        self._checkpoint_value = checkpoint
        self._corpus_dir_value = corpus_dir
        self._sglang_url = str(sglang_url).rstrip("/")
        self._max_rounds = _optional_positive_int(max_rounds, name="max_rounds")
        self._max_new_tokens = _positive_int(max_new_tokens, name="max_new_tokens")
        self._remaining_context_output_ceiling_tokens = _optional_positive_int(
            remaining_context_output_ceiling_tokens,
            name="remaining_context_output_ceiling_tokens",
        )
        self._working_memory_turns = _positive_int(
            working_memory_turns, name="working_memory_turns", maximum=32
        )
        self._command_timeout = _positive_float(command_timeout, name="command_timeout")
        self._request_timeout = _positive_float(request_timeout, name="request_timeout")
        self._max_tool_output_tokens = _optional_positive_int(
            max_tool_output_tokens, name="max_tool_output_tokens"
        )
        self._use_remaining_context = _as_bool(
            use_remaining_context_for_generation,
            name="use_remaining_context_for_generation",
        )
        if not self._use_remaining_context:
            raise ValueError(
                "the Terminal-Bench full-SGLang post-compression policy requires "
                "use_remaining_context_for_generation=true"
            )
        if (
            not isinstance(post_compression_generation_policy, str)
            or post_compression_generation_policy
            != R2EACT_POST_COMPRESSION_GENERATION_POLICY
        ):
            raise ValueError(
                "post_compression_generation_policy must be "
                f"{R2EACT_POST_COMPRESSION_GENERATION_POLICY!r} for the "
                "Terminal-Bench 0.4.1 evaluation contract"
            )
        self._post_compression_generation_policy = (
            post_compression_generation_policy
        )
        self._process_lock = Path(process_lock).expanduser().resolve()
        self._session: Any | None = None
        self._tmux_session_name: str | None = None
        self._tmux_pane_id: str | None = None

    def _path_setting(
        self,
        value: str | Path | None,
        env_name: str,
        default: Path,
        *legacy_env_names: str,
    ) -> Path:
        raw = value
        if raw is None:
            # Canonical variable names win across all environment sources. Only
            # consult a legacy alias when the canonical name is entirely absent.
            for candidate in (env_name, *legacy_env_names):
                raw = self._get_env(candidate)
                if raw is not None:
                    break
        path = Path(raw) if raw is not None else default
        return path.expanduser().resolve()

    def _resolve_paths(self) -> tuple[Path, Path, Path]:
        root = self._path_setting(
            self._specforge_root_value,
            "SPECFORGE_ROOT",
            DEFAULT_SPECFORGE_ROOT,
        )
        checkpoint = self._path_setting(
            self._checkpoint_value,
            "R2EACT_CHECKPOINT",
            root / DEFAULT_CHECKPOINT_RELATIVE,
            "GREPSEEK_CHECKPOINT",
        )
        corpus = self._path_setting(
            self._corpus_dir_value,
            "R2EACT_CORPUS_DIR",
            root / DEFAULT_CORPUS_RELATIVE,
            "GREPSEEK_CORPUS_DIR",
        )
        if not (root / "context_compression" / "r2eact_sglang_live.py").is_file():
            raise FileNotFoundError(f"invalid SpecForge root: {root}")
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"compressor checkpoint not found: {checkpoint}")
        if not corpus.is_dir():
            raise FileNotFoundError(
                f"r^2eAct corpus preflight directory not found: {corpus}"
            )
        return root, checkpoint, corpus

    async def setup(self, environment: BaseEnvironment) -> None:
        if _HARBOR_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Harbor is required to run R2EActAgent"
            ) from _HARBOR_IMPORT_ERROR
        from harbor.agents.terminus_2.tmux_session import TmuxSession

        session_name = R2EACT_SESSION_PREFIX + uuid.uuid4().hex[:12]
        session_env = dict(TERMINAL_SESSION_ENV)
        session_env.update(self._extra_env)
        self._session = TmuxSession(
            session_name=session_name,
            environment=environment,
            logging_path=R2EACT_PANE_LOG_PATH,
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            pane_width=200,
            pane_height=60,
            extra_env=session_env,
            user=environment.default_user,
        )
        self._tmux_session_name = session_name
        try:
            await self._session.start()
            if getattr(self._session, "_session_name", None) != session_name:
                raise RuntimeError(
                    "Harbor 0.21 tmux private ABI changed before pane pinning"
                )
            pane_result = await environment.exec(
                "tmux display-message -p -t "
                + shlex.quote(session_name)
                + " '#{pane_id}'",
                timeout_sec=10,
                user=environment.default_user,
            )
            pane_id = str(pane_result.stdout or "").strip()
            if pane_result.return_code != 0 or not re.fullmatch(r"%\d+", pane_id):
                raise RuntimeError(
                    "failed to resolve the stable initial Harbor tmux pane id"
                )
            self._tmux_pane_id = pane_id
            # Harbor 0.21 consistently uses this field as the target for
            # send-keys/paste operations. Pinning it prevents task-issued
            # split/select-window commands from moving the adapter elsewhere.
            self._session._session_name = pane_id

            # Avoid Harbor's process-global `tmux wait done` completion channel.
            # A unique channel cannot leak a late signal into a later command.
            channel = R2EACT_WAIT_CHANNEL_PREFIX + uuid.uuid4().hex
            await self._session.send_keys(
                [f"stty -echo; tmux wait-for -S {channel}", "Enter"],
                block=False,
                min_timeout_sec=0.0,
            )
            init_result = await environment.exec(
                "timeout 10s tmux wait-for " + shlex.quote(channel),
                timeout_sec=15,
                user=environment.default_user,
            )
            if init_result.return_code != 0:
                raise RuntimeError("failed to initialize the pinned tmux pane")
        except BaseException:
            await self._cleanup_session_safely(environment)
            raise

    def _load_and_run(
        self,
        *,
        instruction: str,
        runner: HarborTerminalToolRunner,
        checkpoint: Path,
        corpus: Path,
        events_path: Path,
        cancel_requested: threading.Event | None = None,
    ) -> dict[str, Any]:
        from context_compression.r2eact_tool import ToolResult

        cancellation = cancel_requested or threading.Event()
        if cancellation.is_set():
            raise _RolloutCancelled("rollout cancelled before bundle loading")

        with _PATCH_LOCK, _terminal_specforge_contract() as (
            live_module,
            sglang_live_module,
        ):
            bundle = sglang_live_module.load_sglang_live_bundle(
                checkpoint,
                sglang_url=self._sglang_url,
                corpus_dir=corpus,
                max_new_tokens=self._max_new_tokens,
                tool_timeout=self._command_timeout,
                tool_max_output_tokens=self._max_tool_output_tokens,
                request_timeout=self._request_timeout,
                local_files_only=True,
                use_remaining_context_for_generation=self._use_remaining_context,
                remaining_context_output_ceiling_tokens=(
                    self._remaining_context_output_ceiling_tokens
                ),
            )
            if cancellation.is_set():
                raise _RolloutCancelled("rollout cancelled after bundle loading")
            if bundle.checkpoint_context_organization != (
                sglang_live_module.QWEN36_TOKEN_CLOCK_V1
            ):
                raise RuntimeError(
                    "this adapter only supports the deployed Qwen3.6 token-clock checkpoint"
                )
            if dict(sglang_live_module.QWEN36_GENERATION_SAMPLING) != (
                QWEN36_TB20_SAMPLING
            ):
                raise RuntimeError(
                    "SpecForge Qwen3.6 sampling no longer matches the pinned "
                    "Terminal-Bench 2.0 protocol"
                )
            serving_capacity = _serving_capacity_metadata(
                bundle.generator,
                requested_max_new_tokens=(
                    self._remaining_context_output_ceiling_tokens
                    or self._max_new_tokens
                ),
            )
            runner._tool_result_type = ToolResult
            runner._tokenizer = bundle.generator.tokenizer
            memory_runner = bundle.tool_runner.restricted(
                ("shell",),
                shell_corpus="react_memory",
            )
            runner.attach_memory_runner(memory_runner)
            bundle = replace(
                bundle,
                tool_runner=runner,
                corpus_preflight={
                    "runner": type(runner).__name__,
                    "execution_target": "harbor_task_environment",
                    "unrestricted_terminal": True,
                    "memory_archive": "specforge_react_memory_sidecar",
                    "memory_archive_virtual_path": R2EACT_MEMORY_VIRTUAL_PATH,
                },
                web_search_enabled=False,
                browser_enabled=False,
            )

            def event_sink(event: Mapping[str, Any]) -> None:
                if cancellation.is_set():
                    raise _RolloutCancelled("rollout cancelled at an event boundary")
                with events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")

            if cancellation.is_set():
                raise _RolloutCancelled("rollout cancelled before generation")
            post_compression_generation_policy = (
                self._post_compression_generation_policy
            )
            record = live_module.run_live_question(
                bundle,
                instruction.strip(),
                max_rounds=self._max_rounds,
                event_sink=event_sink,
                tools=("shell",),
                working_memory_turns=self._working_memory_turns,
                shell_corpus="react_memory",
                post_compression_generation_policy=(
                    post_compression_generation_policy
                ),
            )
            if cancellation.is_set():
                raise _RolloutCancelled("rollout cancelled after generation")
            # Upstream metadata describes its built-in corpus runner. Correct
            # those fields so a custom-harness score cannot be mistaken for a
            # Wiki18 rollout after we injected the terminal contract.
            metadata = dict(record.get("demo_metadata", {}))
            metadata.update(
                {
                    "mode": "harbor_terminal_bench",
                    "system_prompt_source": "qwen36_official_chat_template",
                    "system_prompt_source_episode_id": None,
                    "system_prompt_reused_verbatim": True,
                    "caller_system_prompt": "",
                    "user_prompt_source": "harbor_instruction_via_query_template",
                    "tool_schema_source": (
                        "harbor_terminal_plus_virtual_react_memory_v1"
                    ),
                    "shell_corpus": "react_memory",
                    "shell_execution_target": "harbor_task_environment",
                    "tool_runner": type(runner).__name__,
                    "memory_archive_filename": R2EACT_MEMORY_FILENAME,
                    "memory_archive_virtual_path": R2EACT_MEMORY_VIRTUAL_PATH,
                    "memory_archive_persistence": (
                        "agent_logs_atomic_snapshot_before_episode_cleanup_v1"
                    ),
                    # The public r2eact_tool facade aliases a legacy class for
                    # ABI compatibility; expose only its canonical role here.
                    "memory_archive_runner": "R2EActToolModeRunner",
                    "terminal_output_capture": "tmux_pipe_pane_incremental_v1",
                    "terminal_output_normalization": ("cr_to_lf_strip_ansi_nul_v1"),
                    "terminal_capture_max_bytes": R2EACT_CAPTURE_MAX_BYTES,
                    "terminal_completion_channel": "per_command_uuid_v1",
                    "terminal_timeout_recovery": (
                        "respawn_pinned_pane_preserve_cwd_v1"
                    ),
                    "evaluation_post_compression_generation_policy": (
                        post_compression_generation_policy
                    ),
                    "evaluation_protocol_target": QWEN36_TB20_PROTOCOL_ID,
                    "official_context_tokens": QWEN36_TB20_CONTEXT_TOKENS,
                    "official_max_new_tokens": QWEN36_TB20_MAX_NEW_TOKENS,
                    "remaining_context_output_ceiling_tokens": (
                        self._remaining_context_output_ceiling_tokens
                    ),
                    "generation_sampling": dict(QWEN36_TB20_SAMPLING),
                    "serving_capacity": serving_capacity,
                }
            )
            record["demo_metadata"] = metadata
            return record

    async def _kill_session(self, environment: BaseEnvironment) -> None:
        session = self._session
        session_name = self._tmux_session_name
        if session is None:
            return
        try:
            if session_name:
                result = await environment.exec(
                    "tmux kill-session -t " + shlex.quote(session_name),
                    timeout_sec=10,
                    user=environment.default_user,
                )
                if result.return_code != 0:
                    self.logger.debug(
                        "tmux session cleanup returned %s: %s",
                        result.return_code,
                        result.stderr,
                    )
        except asyncio.CancelledError:
            # Preserve the handle so a shielded caller can retry/drain cleanup.
            raise
        except Exception:
            self.logger.debug("failed to clean up tmux session", exc_info=True)
        self._session = None
        self._tmux_session_name = None
        self._tmux_pane_id = None

    async def _cleanup_session_safely(self, environment: BaseEnvironment) -> None:
        """Finish tmux cleanup even if the owning Harbor task is cancelled again."""

        cleanup = asyncio.create_task(self._kill_session(environment))
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
                continue
        # Surface cleanup errors before re-propagating any cancellation.
        await cleanup
        if cancelled:
            raise asyncio.CancelledError

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self._session is None:
            raise RuntimeError("setup() did not create the terminal session")
        cancel_requested = threading.Event()
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            events_path = self.logs_dir / R2EACT_EVENTS_FILENAME
            record_path = self.logs_dir / R2EACT_ROLLOUT_FILENAME
            memory_path = self.logs_dir / R2EACT_MEMORY_FILENAME
            events_path.unlink(missing_ok=True)
            # Leave a deterministic artifact even if preflight, loading,
            # generation, or cancellation happens before SpecForge begins an
            # episode.  A live sidecar snapshot atomically replaces it during
            # end_episode().
            memory_path.write_bytes(b"")

            root, checkpoint, corpus = self._resolve_paths()
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)

            loop = asyncio.get_running_loop()
            runner = HarborTerminalToolRunner(
                loop=loop,
                session=self._session,
                tool_result_type=object,  # replaced after SpecForge imports
                tokenizer=None,
                timeout_seconds=self._command_timeout,
                max_output_tokens=self._max_tool_output_tokens,
                cancel_requested=cancel_requested,
                memory_archive_path=memory_path,
            )

            def blocking_rollout() -> dict[str, Any]:
                return self._load_and_run(
                    instruction=instruction,
                    runner=runner,
                    checkpoint=checkpoint,
                    corpus=corpus,
                    events_path=events_path,
                    cancel_requested=cancel_requested,
                )

            async with _exclusive_process_lease(self._process_lock):
                worker = asyncio.create_task(asyncio.to_thread(blocking_rollout))
                try:
                    record = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # Python cannot safely kill a worker thread. Ask the rollout
                    # to stop at its next event/tool boundary and drain it while
                    # the process lease and task environment are still alive.
                    cancel_requested.set()
                    while not worker.done():
                        try:
                            await asyncio.shield(worker)
                        except asyncio.CancelledError:
                            cancel_requested.set()
                            continue
                        except BaseException:
                            break
                    raise
            record_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summary = compression_summary(record)
            context.n_input_tokens = summary["approximate_input_tokens"]
            context.n_output_tokens = summary["output_tokens"]
            context.n_cache_tokens = 0
            context.cost_usd = 0.0
            context.metadata = {
                "adapter": self.name(),
                "adapter_version": self.version(),
                "checkpoint": str(checkpoint),
                "sglang_url": self._sglang_url,
                "stop_reason": record.get("stop_reason"),
                "round_count": record.get("round_count"),
                "rollout_answered": record.get("stop_reason") == "answered",
                "max_rounds": self._max_rounds,
                "round_limit_policy": (
                    "trial_timeout_only"
                    if self._max_rounds is None
                    else "configured_max_rounds"
                ),
                "configured_max_new_tokens": self._max_new_tokens,
                "remaining_context_output_ceiling_tokens": (
                    self._remaining_context_output_ceiling_tokens
                ),
                "working_memory_turns": self._working_memory_turns,
                "use_remaining_context_for_generation": self._use_remaining_context,
                "evaluation_post_compression_generation_policy": (
                    self._post_compression_generation_policy
                ),
                "max_tool_output_tokens": self._max_tool_output_tokens,
                "system_prompt_source": "qwen36_official_chat_template",
                "memory_archive_filename": R2EACT_MEMORY_FILENAME,
                "memory_archive_virtual_path": R2EACT_MEMORY_VIRTUAL_PATH,
                "memory_archive_path": str(memory_path),
                "memory_archive_persisted": memory_path.is_file(),
                "memory_archive_bytes": memory_path.stat().st_size,
                "terminal_output_capture": "tmux_pipe_pane_incremental_v1",
                "terminal_output_normalization": "cr_to_lf_strip_ansi_nul_v1",
                "terminal_capture_max_bytes": R2EACT_CAPTURE_MAX_BYTES,
                "evaluation_protocol_target": QWEN36_TB20_PROTOCOL_ID,
                "official_context_tokens": QWEN36_TB20_CONTEXT_TOKENS,
                "official_max_new_tokens": QWEN36_TB20_MAX_NEW_TOKENS,
                "generation_sampling": dict(QWEN36_TB20_SAMPLING),
                "serving_capacity": dict(
                    record.get("demo_metadata", {}).get("serving_capacity", {})
                ),
                "compression": summary,
                "record_path": str(record_path),
                "events_path": str(events_path),
            }
        finally:
            cancel_requested.set()
            await self._cleanup_session_safely(environment)


__all__ = [
    "R2EActAgent",
    "HarborTerminalToolRunner",
    "compression_summary",
    "terminal_tool_schemas",
]
