"""Terminus-2 control plane with the r^2eAct 0.4.x context plane.

This adapter intentionally keeps Harbor's Terminus-2 loop, JSON parser,
verbatim terminal interaction, completion confirmation, and ATIF trajectory
writer.  The LLM boundary projects completed Terminus rounds to Qwen's native
``assistant/tool`` transcript and passes them through SpecForge's recursive
SGLang compressor.  The terminal data plane is a direct, verified tmux variant
that excludes an inline recorder from the task-shell PTY path.

The projection is a serving ABI, not a claim that this is byte-identical to a
stock Terminus-2 chat-completions run.  Its identifier is persisted in trial
metadata so strict Terminus baselines remain distinguishable.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from harbor.agents.terminus_2.terminus_2 import Command, Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM, LLMResponse
from harbor.models.agent.context import AgentContext
from harbor.models.metric import UsageInfo
from harbor.models.trial.paths import EnvironmentPaths

from terminal_bench_eval.r2eact_agent import (
    DEFAULT_CHECKPOINT_RELATIVE,
    DEFAULT_CORPUS_RELATIVE,
    DEFAULT_PROCESS_LOCK,
    DEFAULT_SGLANG_URL,
    DEFAULT_SPECFORGE_ROOT,
    QWEN36_TB20_CONTEXT_TOKENS,
    QWEN36_TB20_MAX_NEW_TOKENS,
    QWEN36_TB20_SAMPLING,
    R2EACT_MEMORY_FILENAME,
    R2EACT_MEMORY_VIRTUAL_PATH,
    R2EACT_POST_COMPRESSION_GENERATION_POLICY,
    _as_bool,
    _exclusive_process_lease,
    _is_virtual_memory_query,
    _optional_positive_int,
    _positive_float,
    _positive_int,
    _serving_capacity_metadata,
)


TERMINUS_R2EACT_VERSION = "0.4.12"
TERMINUS_R2EACT_NAME = "r2eact-terminus-2"
TERMINUS_R2EACT_PROMPT_ABI = "terminus2_json_qwen_memory_system_projection_v8"
TERMINUS_R2EACT_CONTEXT_ABI = (
    "qwen36_round_recursive_oversized_round_capacity_tool_admission_v6"
)
TERMINUS_R2EACT_EVENTS_FILENAME = "r2eact-terminus-2-events.jsonl"
TERMINUS_R2EACT_RECEIPTS_FILENAME = "r2eact-terminus-2-receipts.json"
TERMINUS_TERMINAL_TRANSPORT_ABI = "direct_tmux_verified_paste_v1"
TERMINUS_TERMINAL_TRANSPORT_POLICY = (
    "direct_tmux_pipe_pane_no_inline_recorder_verified_oversized_enqueue_v1"
)
TERMINUS_TERMINAL_TRANSPORT_RECEIPTS_FILENAME = (
    "terminus-terminal-transport-events.jsonl"
)
TERMINUS_TERMINAL_TRANSPORT_PROBE_KEYSTROKE_BYTES = 100_000
TERMINUS_TERMINAL_TRANSPORT_MAX_PASTE_BYTES = 4 * 1024 * 1024
TERMINUS_R2EACT_MEMORY_RESERVE_TOKENS = 1_024
TERMINUS_R2EACT_PROGRESS_GUARD_POLICY = (
    "latched_repeat_and_explicit_concrete_progress_cue_v2"
)
TERMINUS_R2EACT_ROUND_REMINDER_POLICY = (
    "every_round_contiguous_complete_archive_prefix_grep_retrieval_v1"
)
TERMINUS_R2EACT_DEFAULT_STAGNATION_CUE_ROUNDS = 6
TERMINUS_R2EACT_DEFAULT_EXACT_REPEAT_CUE_COUNT = 3
TERMINUS_R2EACT_DEFAULT_WORKING_MEMORY_TURNS = 8
TERMINUS_R2EACT_OVERSIZED_ROUND_EVICTION_POLICY = (
    "whole_complete_round_above_serving_pool_per_wm_slot_"
    "latest_successor_grace_v2"
)
TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY = "bounded_command_projection_v2"
TERMINUS_TOOL_OBSERVATION_INFORMATION_LINES_POLICY = (
    "released_mirror_then_specforge_canonical_empty_v1"
)
TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING = (
    "sha256_len64be_prefixed_utf8_keystrokes_v1"
)
TERMINUS_TOOL_OBSERVATION_EXACT_MAX_BYTES = 1_024
# Compatibility export for v0.4.6/v0.4.7 tests and receipt readers.
TERMINUS_TOOL_OBSERVATION_EXACT_SINGLE_MAX_BYTES = (
    TERMINUS_TOOL_OBSERVATION_EXACT_MAX_BYTES
)
TERMINUS_TOOL_OBSERVATION_MANIFEST_FIRST_LINE_MAX_BYTES = 120
TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES = 16
TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_BYTES = 4_096

def _memory_system_prompt(working_memory_turns: int) -> str:
    """Render the memory contract from the effective raw-round configuration."""

    return f"""
r^2eAct recursive-memory policy:
- The newest {working_memory_turns} completed interaction rounds are normally kept as raw
  working memory. Older rounds are recursively folded from left to right. An anomalously
  large non-latest complete round or capacity pressure may advance that
  whole-round cursor earlier so the next request fits. The newest oversized
  round remains raw for one successor generation whenever it fits after older
  rounds are folded.
- A controller-managed, read-only episodic archive is available at
  {R2EACT_MEMORY_VIRTUAL_PATH}. It contains the exact assistant/terminal rounds
  that have left active raw context and may be empty before compression occurs.
  Older interaction history is not lost: when a needed fact is no longer in raw
  working memory, search for it in this archive.
- Recall old evidence with one dedicated read-only command, for example
  `grep -n -i -F 'phrase' {R2EACT_MEMORY_VIRTUAL_PATH} | head -n 8`. Do not combine
  an archive lookup with task commands. A bare `memory.jsonl`, or one elsewhere
  in the task, is an ordinary task file.
- Treat `[r^2eAct round reminder]` and `[r^2eAct progress guard]` as controller
  instructions. The round reminder is emitted on every round without a
  round-number cutoff and reports the exact contiguous archived prefix that is
  already durable. Before repeating an inspection or re-deriving an older
  fact, make one dedicated archive lookup. Otherwise make or test a concrete
  task change. Do not continue broad inspection after the evidence needed to
  act is available.
- Keep each JSON response's analysis and plan concise. For ordinary task files,
  prefer bounded reads with `grep`, `sed -n`, `od`, or `readelf` (and `head` or
  `tail` where useful). Narrow a truncated query instead of re-reading a whole
  file. Do not
  join independent diagnostics with `||`, because later commands then run only
  when an earlier one fails.
- Terminus's tmux transport does not expose per-command exit status. In the
  projected tool-response JSON, `exit_code: -1` means "unavailable", not
  failure; timeout status remains explicit in `timed_out`.
""".strip()


# Compatibility export for tests and callers that inspect the default contract.
_MEMORY_SYSTEM_PROMPT = _memory_system_prompt(
    TERMINUS_R2EACT_DEFAULT_WORKING_MEMORY_TURNS
)

_INSPECTION_COMMANDS = frozenset(
    {
        "awk",
        "basename",
        "cat",
        "cd",
        "command",
        "cut",
        "df",
        "dirname",
        "du",
        "echo",
        "env",
        "file",
        "find",
        "git",
        "grep",
        "head",
        "id",
        "ls",
        "lsof",
        "md5sum",
        "nm",
        "objdump",
        "od",
        "pgrep",
        "printenv",
        "printf",
        "ps",
        "pwd",
        "readelf",
        "realpath",
        "rg",
        "sed",
        "sha256sum",
        "stat",
        "strings",
        "tail",
        "type",
        "uname",
        "wc",
        "which",
        "whoami",
        "xxd",
    }
)
_READ_ONLY_GIT_SUBCOMMANDS = frozenset(
    {"diff", "log", "rev-parse", "show", "status"}
)
_ALWAYS_CONCRETE_PROGRESS_COMMANDS = frozenset(
    {"apply_patch"}
)
_FILESYSTEM_MUTATION_COMMANDS = frozenset(
    {
        "chmod",
        "chown",
        "chgrp",
        "cp",
        "install",
        "ln",
        "mkdir",
        "mv",
        "rm",
        "rmdir",
        "rsync",
        "touch",
        "truncate",
    }
)
_GIT_MUTATION_SUBCOMMANDS = frozenset(
    {
        "add",
        "apply",
        "checkout",
        "cherry-pick",
        "clean",
        "commit",
        "merge",
        "mv",
        "rebase",
        "reset",
        "restore",
        "rm",
        "switch",
    }
)


def _atomic_write(path: Path, payload: bytes) -> None:
    """Durably replace one controller-owned artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _terminal_transport_metadata() -> dict[str, Any]:
    """Describe the shared compressed/raw Terminus terminal data plane."""

    return {
        "abi": TERMINUS_TERMINAL_TRANSPORT_ABI,
        "policy": TERMINUS_TERMINAL_TRANSPORT_POLICY,
        "inline_terminal_recorder": "disabled_invariant",
        "terminal_output_artifact": "terminus_2.pane",
        "transport_receipts_artifact": (
            TERMINUS_TERMINAL_TRANSPORT_RECEIPTS_FILENAME
        ),
        "small_and_control_keys": "verbatim_tmux_send_keys",
        "oversized_literal_keys": {
            "staging": "guest_base64_file",
            "stage_verification": "controller_download_bytes_and_sha256",
            "tmux_buffer_verification": "save_buffer_download_bytes_and_sha256",
            "pane_preconditions": ["pane_dead=0", "pane_input_off=0"],
            "acceptance": "tmux_paste_command_return_code_zero",
            "delivery_scope": "tmux_pane_enqueue_not_target_read",
            "fail_closed": True,
            "maximum_source_bytes": TERMINUS_TERMINAL_TRANSPORT_MAX_PASTE_BYTES,
        },
        "per_environment_probe": {
            "enabled": True,
            "session_scope": "disposable_no_profile_no_history_tmux_session",
            "keystroke_bytes": (
                TERMINUS_TERMINAL_TRANSPORT_PROBE_KEYSTROKE_BYTES
            ),
            "verifies": [
                "heredoc_body_bytes_and_sha256",
                "post_payload_command_execution",
                "pane_remains_input_enabled",
            ],
        },
        "nonblocking_settle_time": "minimum_duration_after_tmux_enqueue",
        "shared_with_raw_history_control": True,
    }


class VerifiedTmuxSession(TmuxSession):
    """Direct tmux session with fail-closed oversized-input verification.

    Harbor's optional ``asciinema rec --stdin`` mode inserts a synchronous
    bidirectional PTY relay between tmux and the task shell.  Older recorder
    versions can deadlock under a large paste while the shell echoes output.
    This session deliberately keeps recording out of the input data plane and
    uses tmux's existing ``pipe-pane`` transcript instead.

    Large literal keys retain Terminus's verbatim semantics.  They are staged,
    downloaded and hashed, loaded into a named tmux buffer, saved back out and
    hashed again, and pasted only while the pane is alive and input-enabled.
    A successful receipt means the complete buffer was accepted for tmux pane
    enqueue; it does not make the stronger, generally unobservable claim that
    the foreground target process has already read every byte.
    """

    _PROBE_TIMEOUT_SECONDS = 30.0
    _PROBE_POLL_SECONDS = 0.05

    def __init__(
        self,
        *args: Any,
        transport_receipts_path: Path,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("local_asciinema_recording_path") is not None:
            raise ValueError("inline terminal recording is disabled by transport policy")
        if kwargs.get("remote_asciinema_recording_path") is not None:
            raise ValueError("inline terminal recording is disabled by transport policy")
        super().__init__(*args, **kwargs)
        self._transport_receipts_path = Path(transport_receipts_path)

    def _append_transport_event(self, event: Mapping[str, Any]) -> None:
        payload = {
            "schema": "terminus_terminal_transport_event_v1",
            "timestamp_ns": time.time_ns(),
            "transport_abi": TERMINUS_TERMINAL_TRANSPORT_ABI,
            **dict(event),
        }
        self._transport_receipts_path.parent.mkdir(parents=True, exist_ok=True)
        with self._transport_receipts_path.open("ab") as handle:
            handle.write(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
            )
            handle.flush()
            os.fsync(handle.fileno())

    async def _checked_environment_exec(self, command: str, *, action: str) -> Any:
        result = await self.environment.exec(command=command, user=self._user)
        if result.return_code != 0:
            raise self._send_keys_error(action, command, result)
        return result

    async def _download_exact_remote_bytes(self, remote_path: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="terminus-transport-verify-") as root:
            local_path = Path(root) / "payload"
            await self.environment.download_file(
                source_path=remote_path,
                target_path=local_path,
            )
            return local_path.read_bytes()

    async def _pane_input_state(self) -> tuple[int, int]:
        format_value = "#{pane_dead} #{pane_input_off}"
        command = (
            "tmux display-message -p -t "
            f"{shlex.quote(self._session_name)} {shlex.quote(format_value)}"
        )
        result = await self._checked_environment_exec(
            command, action="oversized-pane-precheck"
        )
        fields = (result.stdout or "").strip().split()
        if len(fields) != 2 or any(value not in {"0", "1"} for value in fields):
            raise RuntimeError(
                f"{self.environment.session_id}: invalid tmux pane state: "
                f"{result.stdout!r}"
            )
        return int(fields[0]), int(fields[1])

    async def _paste_key(self, key: str, action: str) -> None:
        if not key:
            return
        source = key.encode("utf-8")
        expected_sha256 = hashlib.sha256(source).hexdigest()
        receipt: dict[str, Any] = {
            "event": "oversized_literal_enqueue",
            "action": action,
            "source_bytes": len(source),
            "source_sha256": expected_sha256,
            "stage_verified": False,
            "tmux_buffer_verified": False,
            "paste_command_accepted": False,
            "delivery_scope": "tmux_pane_enqueue_not_target_read",
        }
        if len(source) > TERMINUS_TERMINAL_TRANSPORT_MAX_PASTE_BYTES:
            receipt.update(
                status="rejected",
                error=(
                    "oversized literal exceeds fail-closed transport ceiling: "
                    f"{len(source)} > {TERMINUS_TERMINAL_TRANSPORT_MAX_PASTE_BYTES}"
                ),
            )
            self._append_transport_event(receipt)
            raise RuntimeError(str(receipt["error"]))

        token = uuid.uuid4().hex
        remote_stage = f"/tmp/.terminus-tmux-stage-{token}"
        remote_buffer_copy = f"/tmp/.terminus-tmux-buffer-{token}"
        quoted_stage = shlex.quote(remote_stage)
        quoted_buffer_copy = shlex.quote(remote_buffer_copy)
        buffer_name = f"terminus-verified-{token}"
        payload = base64.b64encode(source).decode("ascii")
        buffer_loaded = False

        try:
            for offset in range(0, len(payload), self._PASTE_BASE64_CHUNK_LEN):
                chunk = payload[offset : offset + self._PASTE_BASE64_CHUNK_LEN]
                redirect = ">>" if offset else ">"
                await self._checked_environment_exec(
                    f"printf %s {chunk} | base64 -d {redirect} {quoted_stage}",
                    action="oversized-stage-write",
                )

            staged = await self._download_exact_remote_bytes(remote_stage)
            receipt["stage_bytes"] = len(staged)
            receipt["stage_sha256"] = hashlib.sha256(staged).hexdigest()
            if staged != source:
                raise RuntimeError(
                    f"{self.environment.session_id}: oversized stage integrity "
                    f"mismatch: expected {len(source)} bytes/{expected_sha256}, got "
                    f"{len(staged)} bytes/{receipt['stage_sha256']}"
                )
            receipt["stage_verified"] = True

            pane_dead, pane_input_off = await self._pane_input_state()
            receipt["pane_dead"] = pane_dead
            receipt["pane_input_off"] = pane_input_off
            if pane_dead or pane_input_off:
                raise RuntimeError(
                    f"{self.environment.session_id}: refusing oversized paste into "
                    f"pane_dead={pane_dead}, pane_input_off={pane_input_off}"
                )

            await self._checked_environment_exec(
                f"tmux load-buffer -b {buffer_name} {quoted_stage}",
                action="oversized-buffer-load",
            )
            buffer_loaded = True
            await self._checked_environment_exec(
                f"tmux save-buffer -b {buffer_name} {quoted_buffer_copy}",
                action="oversized-buffer-save",
            )
            buffered = await self._download_exact_remote_bytes(remote_buffer_copy)
            receipt["tmux_buffer_bytes"] = len(buffered)
            receipt["tmux_buffer_sha256"] = hashlib.sha256(buffered).hexdigest()
            if buffered != source:
                raise RuntimeError(
                    f"{self.environment.session_id}: tmux buffer integrity mismatch: "
                    f"expected {len(source)} bytes/{expected_sha256}, got "
                    f"{len(buffered)} bytes/{receipt['tmux_buffer_sha256']}"
                )
            receipt["tmux_buffer_verified"] = True

            await self._checked_environment_exec(
                "tmux paste-buffer -d "
                f"-b {buffer_name} -t {shlex.quote(self._session_name)}",
                action=action,
            )
            buffer_loaded = False
            receipt["paste_command_accepted"] = True
            receipt["status"] = "accepted"
            self._append_transport_event(receipt)
        except BaseException as exc:
            receipt["status"] = "failed"
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            self._append_transport_event(receipt)
            raise
        finally:
            if buffer_loaded:
                with contextlib.suppress(BaseException):
                    await self.environment.exec(
                        command=f"tmux delete-buffer -b {buffer_name}",
                        user=self._user,
                    )
            with contextlib.suppress(BaseException):
                await self.environment.exec(
                    command=f"rm -f {quoted_stage} {quoted_buffer_copy}",
                    user=self._user,
                )

    async def _send_non_blocking_keys(
        self,
        keys: list[str],
        min_timeout_sec: float,
    ) -> None:
        """Apply Terminus's observation delay after the complete enqueue.

        Base Harbor starts this clock before key delivery.  That is equivalent
        for small ``send-keys`` calls, but verified staging and two downloads
        can otherwise consume the entire requested settle interval before an
        oversized literal reaches the pane.
        """

        await self._send_keys_to_session(keys, action="non-blocking")
        if min_timeout_sec > 0.0:
            await asyncio.sleep(min_timeout_sec)

    @staticmethod
    def _probe_body(length: int) -> bytes:
        pattern = b"// direct tmux transport probe 0123456789abcdef\n"
        body = (pattern * ((length // len(pattern)) + 1))[:length]
        if not body.endswith(b"\n"):
            body = body[:-1] + b"\n"
        return body

    async def _wait_for_probe_ack(self, path: str, token: str) -> None:
        deadline = time.monotonic() + self._PROBE_TIMEOUT_SECONDS
        command = f"cat {shlex.quote(path)} 2>/dev/null"
        while time.monotonic() < deadline:
            result = await self.environment.exec(command=command, user=self._user)
            if result.return_code == 0 and (result.stdout or "") == token:
                return
            await asyncio.sleep(self._PROBE_POLL_SECONDS)
        raise TimeoutError(
            f"{self.environment.session_id}: direct tmux transport probe timed out"
        )

    async def _run_transport_probe(self, probe_id: str | None = None) -> None:
        token = probe_id or uuid.uuid4().hex
        delimiter = f"R2EACT_TRANSPORT_{token}"
        body_path = f"/tmp/.terminus-transport-probe-body-{token}"
        ack_path = f"/tmp/.terminus-transport-probe-ack-{token}"
        after_path = f"/tmp/.terminus-transport-probe-after-{token}"
        prefix = f"cat > {shlex.quote(body_path)} <<'{delimiter}'\n"
        suffix = (
            f"{delimiter}\n"
            f"printf %s {token} > {shlex.quote(ack_path)}\n"
        )
        body_length = (
            TERMINUS_TERMINAL_TRANSPORT_PROBE_KEYSTROKE_BYTES
            - len(prefix.encode("utf-8"))
            - len(suffix.encode("utf-8"))
        )
        if body_length <= 0:
            raise RuntimeError("invalid terminal transport probe size")
        body = self._probe_body(body_length)
        keystrokes = prefix.encode("utf-8") + body + suffix.encode("utf-8")
        if len(keystrokes) != TERMINUS_TERMINAL_TRANSPORT_PROBE_KEYSTROKE_BYTES:
            raise RuntimeError("terminal transport probe construction drifted")
        body_sha256 = hashlib.sha256(body).hexdigest()
        receipt: dict[str, Any] = {
            "event": "environment_transport_probe",
            "probe_id": token,
            "keystroke_bytes": len(keystrokes),
            "keystroke_sha256": hashlib.sha256(keystrokes).hexdigest(),
            "expected_body_bytes": len(body),
            "expected_body_sha256": body_sha256,
            "status": "running",
        }
        cleanup = "rm -f " + " ".join(
            shlex.quote(path) for path in (body_path, ack_path, after_path)
        )
        try:
            await self.send_keys(keystrokes.decode("utf-8"), min_timeout_sec=0.0)
            await self._wait_for_probe_ack(ack_path, token)
            observed_body = await self._download_exact_remote_bytes(body_path)
            if observed_body != body:
                raise RuntimeError(
                    f"{self.environment.session_id}: transport probe body mismatch: "
                    f"expected {len(body)} bytes/{body_sha256}, got "
                    f"{len(observed_body)} bytes/"
                    f"{hashlib.sha256(observed_body).hexdigest()}"
                )

            await self.send_keys(
                f"printf %s {token} > {shlex.quote(after_path)}\n",
                min_timeout_sec=0.0,
            )
            await self._wait_for_probe_ack(after_path, token)
            pane_dead, pane_input_off = await self._pane_input_state()
            if pane_dead or pane_input_off:
                raise RuntimeError(
                    f"{self.environment.session_id}: transport probe left pane in "
                    f"pane_dead={pane_dead}, pane_input_off={pane_input_off}"
                )
            receipt.update(
                status="passed",
                observed_body_bytes=len(observed_body),
                observed_body_sha256=hashlib.sha256(observed_body).hexdigest(),
                post_payload_command_acknowledged=True,
                pane_dead=pane_dead,
                pane_input_off=pane_input_off,
            )
            self._append_transport_event(receipt)
        except BaseException as exc:
            receipt["status"] = "failed"
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            self._append_transport_event(receipt)
            raise
        finally:
            with contextlib.suppress(BaseException):
                await self.environment.exec(command=cleanup, user=self._user)

    async def _run_disposable_transport_probe(self) -> None:
        """Probe a throwaway shell without touching task history or profiles."""

        token = uuid.uuid4().hex
        task_session_name = self._session_name
        probe_session_name = f"{task_session_name}-probe-{token[:12]}"
        probe_log = f"/tmp/.terminus-transport-probe-pane-{token}"
        target = shlex.quote(probe_session_name)
        shell_command = (
            "env HISTFILE=/dev/null HISTSIZE=0 HISTFILESIZE=0 "
            "bash --noprofile --norc"
        )
        pipe_command = f"cat > {shlex.quote(probe_log)}"
        start_command = (
            "export TERM=xterm-256color && export SHELL=/bin/bash && "
            f"tmux new-session -x {self._pane_width} -y {self._pane_height} "
            f"-d -s {target} {shlex.quote(shell_command)} \\; "
            f"pipe-pane -t {target} {shlex.quote(pipe_command)}"
        )

        await self._checked_environment_exec(
            start_command,
            action="transport-probe-session-start",
        )
        probe_error: BaseException | None = None
        self._session_name = probe_session_name
        try:
            await self._run_transport_probe(token)
        except BaseException as exc:
            probe_error = exc
        finally:
            self._session_name = task_session_name
        kill_result = await self.environment.exec(
            command=f"tmux kill-session -t {target}",
            user=self._user,
        )
        alive_result = await self.environment.exec(
            command=f"tmux has-session -t {target}",
            user=self._user,
        )
        remove_result = await self.environment.exec(
            command=f"rm -f {shlex.quote(probe_log)}",
            user=self._user,
        )
        disposed = alive_result.return_code != 0 and remove_result.return_code == 0
        self._append_transport_event(
            {
                "event": "transport_probe_session_disposal",
                "probe_id": token,
                "status": "passed" if disposed else "failed",
                "kill_return_code": kill_result.return_code,
                "session_absent": alive_result.return_code != 0,
                "probe_log_removed": remove_result.return_code == 0,
            }
        )
        if not disposed:
            raise RuntimeError(
                f"{self.environment.session_id}: disposable transport probe "
                "session cleanup failed"
            )
        if probe_error is not None:
            raise probe_error

    async def start(self) -> None:
        _atomic_write(self._transport_receipts_path, b"")
        self._append_transport_event(
            {
                "event": "transport_policy",
                "status": "configured",
                "policy": _terminal_transport_metadata(),
            }
        )
        await self._attempt_tmux_installation()
        await self._run_disposable_transport_probe()
        await super().start()
        await self.send_keys(["clear", "Enter"], min_timeout_sec=0.1)


def _one_terminal_line(command: str) -> str:
    """Remove exactly one Terminus execution newline for safe memory lookup."""

    value = str(command)
    if value.endswith("\r\n"):
        value = value[:-2]
    elif value.endswith("\n"):
        value = value[:-1]
    if "\n" in value or "\r" in value:
        raise ValueError("a memory archive lookup must be one command line")
    return value


def _inspection_command_signature(command: str) -> str | None:
    """Return a bounded read-only command signature, conservatively.

    This is intentionally not a general shell safety parser.  A false negative
    merely suppresses an exact-repeat hint.  Stagnation accounting is separate
    and therefore does not mistake an unknown diagnostic for concrete progress.
    """

    value = str(command).strip()
    if not value or "\n" in value or "\r" in value:
        return None
    try:
        lexer = shlex.shlex(value, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        raw_tokens = list(lexer)
    except ValueError:
        return None

    tokens: list[str] = []
    index = 0
    while index < len(raw_tokens):
        # Reads and diagnostic caches under /tmp do not alter a task artifact.
        # Strip those redirections solely for classifying the surrounding
        # command; the returned signature still preserves the original text.
        operator_index = index
        if (
            raw_tokens[index].isdigit()
            and index + 1 < len(raw_tokens)
            and raw_tokens[index + 1] in {"<", ">", ">>", ">&"}
        ):
            operator_index += 1
        operator = raw_tokens[operator_index]
        if operator in {"<", ">", ">>", ">&", "&>"}:
            target_index = operator_index + 1
            if target_index >= len(raw_tokens):
                return None
            target = raw_tokens[target_index]
            if operator in {">", ">>", "&>"} and not (
                _is_diagnostic_temporary_path(target)
            ):
                return None
            if operator == ">&" and target not in {"1", "2"} and not (
                _is_diagnostic_temporary_path(target)
            ):
                return None
            index = target_index + 1
            continue
        if any(character in raw_tokens[index] for character in "<>"):
            return None
        tokens.append(raw_tokens[index])
        index += 1

    separators = {"|", "||", "&&", ";", "&"}
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in separators:
            if not segments[-1]:
                return None
            segments.append([])
        else:
            segments[-1].append(token)
    if not segments or any(not segment for segment in segments):
        return None

    for segment in segments:
        command_index = 0
        while command_index < len(segment) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*", segment[command_index]
        ):
            command_index += 1
        if command_index >= len(segment):
            return None
        executable = os.path.basename(segment[command_index])
        for cross_tool in ("nm", "objdump", "readelf", "strings"):
            if executable.endswith(f"-{cross_tool}"):
                executable = cross_tool
                break
        arguments = segment[command_index + 1 :]
        if executable not in _INSPECTION_COMMANDS:
            return None
        if executable == "sed" and any(
            argument == "-i" or argument.startswith("-i") for argument in arguments
        ):
            return None
        if executable == "find" and any(
            argument in {"-delete", "-exec", "-execdir", "-ok", "-okdir"}
            for argument in arguments
        ):
            return None
        if executable == "git":
            subcommand = next(
                (argument for argument in arguments if not argument.startswith("-")),
                None,
            )
            if subcommand not in _READ_ONLY_GIT_SUBCOMMANDS:
                return None
        if executable == "command" and not (
            arguments and arguments[0] in {"-v", "-V"}
        ):
            return None
    return re.sub(r"\s+", " ", value)


def _is_diagnostic_temporary_path(value: str) -> bool:
    candidate = str(value).strip("'\"")
    return (
        candidate in {"/dev/null", "/tmp", "/var/tmp", "&1", "&2"}
        or candidate.startswith("/tmp/")
        or candidate.startswith("/var/tmp/")
    )


def _command_has_concrete_progress(command: str) -> bool:
    """Recognize an explicit task edit, build, or execution.

    The progress cue is advisory and never blocks a command, so an unknown
    command is not credited as progress merely because its effects are hard to
    parse.  This avoids resetting the stagnation clock for read-only Python,
    objdump, or temporary diagnostic-cache commands while preserving clear
    edits/builds/tests as progress.
    """

    value = str(command).strip()
    if not value:
        return False

    # Python/JavaScript snippets and here-documents can hide a file write from
    # shell tokenization. Credit only a statically visible non-temporary target.
    embedded_write_patterns = (
        r"\bopen\s*\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"][wax+]",
        r"\bPath\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\."
        r"(?:chmod|mkdir|rename|replace|rmdir|touch|unlink|write_bytes|write_text)\s*\(",
        r"\b(?:writeFile|writeFileSync|appendFile|appendFileSync)\s*\(\s*"
        r"['\"]([^'\"]+)['\"]",
    )
    for pattern in embedded_write_patterns:
        for match in re.finditer(pattern, value):
            if not _is_diagnostic_temporary_path(match.group(1)):
                return True

    shell_line = value.splitlines()[0]
    try:
        lexer = shlex.shlex(shell_line, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return False
    if not tokens:
        return False

    # Output redirected into the task tree (or an unresolved/relative target)
    # is concrete progress. Diagnostic artifacts under /tmp remain inspection.
    for index, token in enumerate(tokens):
        if token not in {">", ">>", "&>"}:
            continue
        if index + 1 >= len(tokens):
            return True
        target = tokens[index + 1]
        if target == "&" and index + 2 < len(tokens):
            target = "&" + tokens[index + 2]
        if not _is_diagnostic_temporary_path(target):
            return True

    separators = {"|", "||", "&&", ";", "&"}
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in separators:
            if segments[-1]:
                segments.append([])
        elif token not in {"<", "<<", ">", ">>"}:
            segments[-1].append(token)

    for segment in segments:
        if not segment:
            continue
        command_index = 0
        while command_index < len(segment) and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*=.*", segment[command_index]
        ):
            command_index += 1
        if command_index >= len(segment):
            continue
        executable = os.path.basename(segment[command_index])
        arguments = segment[command_index + 1 :]
        operands = [argument for argument in arguments if not argument.startswith("-")]
        if any(
            argument == "--version" or argument.startswith("--help")
            for argument in arguments
        ):
            continue
        if executable in _ALWAYS_CONCRETE_PROGRESS_COMMANDS:
            return True
        if segment[command_index].startswith(("./", "/app/")):
            return True
        if executable in _FILESYSTEM_MUTATION_COMMANDS:
            paths = [
                argument
                for argument in operands
                if not re.fullmatch(r"[0-7]{3,4}", argument)
            ]
            if executable in {"cp", "install", "rsync"}:
                return bool(paths) and not _is_diagnostic_temporary_path(paths[-1])
            if executable == "ln":
                return bool(paths) and (
                    len(paths) == 1
                    or not _is_diagnostic_temporary_path(paths[-1])
                )
            if paths and any(
                not _is_diagnostic_temporary_path(path) for path in paths
            ):
                return True
        if executable in {"npm", "pnpm", "yarn"}:
            subcommand = next(
                (argument for argument in arguments if not argument.startswith("-")),
                None,
            )
            if subcommand in {
                "add",
                "build",
                "ci",
                "exec",
                "install",
                "run",
                "start",
                "test",
                "uninstall",
                "update",
            }:
                return True
        if executable == "npx" and operands:
            return True
        if executable == "git":
            subcommand = next(
                (argument for argument in arguments if not argument.startswith("-")),
                None,
            )
            if subcommand in _GIT_MUTATION_SUBCOMMANDS:
                return True
        if executable == "sed" and any(
            argument == "-i" or argument.startswith("-i") for argument in arguments
        ):
            return True
        if executable == "find":
            if "-delete" in arguments and any(
                not _is_diagnostic_temporary_path(path)
                for path in operands
                if not path.startswith("{")
            ):
                return True
            for operator in ("-exec", "-execdir", "-ok", "-okdir"):
                if operator not in arguments:
                    continue
                nested_index = arguments.index(operator) + 1
                if nested_index < len(arguments):
                    nested = os.path.basename(arguments[nested_index])
                    if nested in _FILESYSTEM_MUTATION_COMMANDS | {
                        "install",
                        "patch",
                    }:
                        return True
        if executable in {"python", "python3", "bash", "sh"} and any(
            not argument.startswith("-")
            and argument.endswith((".py", ".sh"))
            and not _is_diagnostic_temporary_path(argument)
            for argument in arguments
        ):
            return True
        if executable == "node":
            if any(
                argument in {"-e", "-p", "--eval", "--print"}
                for argument in arguments
            ):
                continue
            script = next(
                (argument for argument in arguments if not argument.startswith("-")),
                None,
            )
            if script is not None and not _is_diagnostic_temporary_path(script):
                return True
        if executable in {"qemu-mips", "qemu-mipsel"}:
            target = next(
                (argument for argument in arguments if not argument.startswith("-")),
                None,
            )
            if target is not None:
                return True
        if executable in {"gcc", "g++", "cc", "clang", "clang++", "ld"}:
            if "-E" in arguments and "-o" not in arguments:
                continue
            if any(
                argument.startswith("-dump") for argument in arguments
            ):
                continue
            if operands:
                return True
        if executable == "cmake":
            if arguments and (
                any(
                    argument in {"--build", "--install", "--workflow"}
                    for argument in arguments
                )
                or "-S" in arguments
                or bool(operands)
            ):
                return True
        if executable == "meson":
            subcommand = next(
                (argument for argument in arguments if not argument.startswith("-")),
                None,
            )
            if subcommand in {
                "compile",
                "dist",
                "install",
                "rewrite",
                "setup",
                "test",
            }:
                return True
        if executable == "make":
            if any(
                argument
                in {"-n", "--dry-run", "--just-print", "-q", "--question"}
                for argument in arguments
            ) or operands == ["help"]:
                continue
            return True
        if executable == "ninja":
            if any(argument in {"-n", "--dry-run"} for argument in arguments):
                continue
            if "-t" in arguments:
                tool_index = arguments.index("-t") + 1
                tool = arguments[tool_index] if tool_index < len(arguments) else None
                if tool != "clean":
                    continue
            return True
        if executable == "cargo":
            subcommand = next(
                (argument for argument in arguments if not argument.startswith("-")),
                None,
            )
            if subcommand in {"build", "check", "fmt", "run", "test"}:
                return True
        if executable == "go":
            subcommand = next(
                (argument for argument in arguments if not argument.startswith("-")),
                None,
            )
            if subcommand in {"build", "fmt", "generate", "run", "test"}:
                return True
        if executable == "ar":
            operation = next(
                (argument for argument in arguments if not argument.startswith("-")),
                "",
            ).lstrip("-")
            if any(flag in operation for flag in "rdmqx"):
                return True
        if executable == "pytest":
            if "--collect-only" not in arguments:
                return True
        if executable == "patch":
            if "--dry-run" not in arguments:
                return True
        if executable == "dd":
            target = next(
                (argument[3:] for argument in arguments if argument.startswith("of=")),
                None,
            )
            if target is not None and not _is_diagnostic_temporary_path(target):
                return True
        if executable in {"curl", "wget"}:
            for index, argument in enumerate(arguments[:-1]):
                if argument in {"-o", "--output"} and not (
                    _is_diagnostic_temporary_path(arguments[index + 1])
                ):
                    return True
    return False


def _finish_reason_kind(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower() or "unknown"
    if isinstance(value, Mapping):
        for name in ("type", "reason", "finish_reason"):
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip().lower()
    return "unknown"


def _utf8_prefix(value: str, max_bytes: int) -> str:
    """Return the longest valid UTF-8 prefix within ``max_bytes``."""

    encoded = str(value).encode("utf-8")
    if len(encoded) <= max_bytes:
        return str(value)
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _command_batch_identity(
    commands: Sequence[Mapping[str, Any]],
) -> tuple[list[bytes], int, str]:
    """Hash the full ordered command batch without re-embedding its contents."""

    encoded_commands = [
        str(command.get("keystrokes", "")).encode("utf-8") for command in commands
    ]
    digest = hashlib.sha256()
    for encoded in encoded_commands:
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return encoded_commands, sum(map(len, encoded_commands)), digest.hexdigest()


def _command_manifest_entry(index: int, encoded: bytes) -> dict[str, Any]:
    command = encoded.decode("utf-8")
    lines = command.splitlines()
    first_line = lines[0] if lines else ""
    return {
        "index": int(index) + 1,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "first_line": _utf8_prefix(
            first_line,
            TERMINUS_TOOL_OBSERVATION_MANIFEST_FIRST_LINE_MAX_BYTES,
        ),
    }


def _bounded_command_projection(
    commands: Sequence[Mapping[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    """Project execution provenance without echoing an unbounded command body.

    A compact command projection up to 1,024 bytes remains exact: a single
    command retains the historical Terminus spelling (apart from its execution
    newline), while a batch uses compact JSON.  Larger projections use a
    deterministic manifest.  The complete keystrokes remain in the prior
    assistant message and in the commands sent to the terminal.
    """

    limits = {
        "exact_projection_max_bytes": TERMINUS_TOOL_OBSERVATION_EXACT_MAX_BYTES,
        "manifest_first_line_max_bytes": (
            TERMINUS_TOOL_OBSERVATION_MANIFEST_FIRST_LINE_MAX_BYTES
        ),
        "manifest_max_entries": TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES,
        "manifest_max_bytes": TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_BYTES,
    }
    if commands is None:
        label = "<terminus-feedback>"
        return label, {
            "policy": TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY,
            "mode": "feedback",
            "applied": True,
            "batch_hash_encoding": (
                TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING
            ),
            "command_count": None,
            "source_command_bytes": 0,
            "source_keystroke_bytes": 0,
            "rendered_command_bytes": len(label.encode("utf-8")),
            "manifest_entries": 0,
            "omitted_commands": 0,
            "limits": limits,
        }

    normalized = [dict(command) for command in commands]
    encoded_commands, batch_bytes, batch_sha256 = _command_batch_identity(normalized)
    command_count = len(normalized)
    if command_count == 0:
        label = "<terminus-no-command-observation>"
        return label, {
            "policy": TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY,
            "mode": "no_commands",
            "applied": True,
            "batch_hash_encoding": (
                TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING
            ),
            "batch_sha256": batch_sha256,
            "command_count": 0,
            "source_command_bytes": 0,
            "source_keystroke_bytes": 0,
            "rendered_command_bytes": len(label.encode("utf-8")),
            "manifest_entries": 0,
            "omitted_commands": 0,
            "limits": limits,
        }

    if command_count == 1:
        exact_label = encoded_commands[0].decode("utf-8").rstrip("\r\n")
        exact_mode = "exact_single"
    else:
        exact_label = json.dumps(
            normalized,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        exact_mode = "exact_batch_v1"
    exact_bytes = len(exact_label.encode("utf-8"))
    if exact_bytes <= TERMINUS_TOOL_OBSERVATION_EXACT_MAX_BYTES:
        label = exact_label
        return label, {
            "policy": TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY,
            "mode": exact_mode,
            "applied": True,
            "batch_hash_encoding": (
                TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING
            ),
            "batch_sha256": batch_sha256,
            "command_count": command_count,
            "source_command_bytes": exact_bytes,
            "source_keystroke_bytes": batch_bytes,
            "rendered_command_bytes": len(label.encode("utf-8")),
            "manifest_entries": 0,
            "omitted_commands": 0,
            "limits": limits,
        }

    if command_count <= TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES:
        selected_indices = list(range(command_count))
    else:
        left_count = TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES // 2
        right_count = TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES - left_count
        selected_indices = [
            *range(left_count),
            *range(command_count - right_count, command_count),
        ]
    entries = [
        _command_manifest_entry(index, encoded_commands[index])
        for index in selected_indices
    ]

    def render_manifest() -> str:
        manifest = {
            "schema": "terminus_command_manifest_v1",
            "batch": {
                "sha256": batch_sha256,
                "bytes": batch_bytes,
                "count": command_count,
                "omitted": command_count - len(entries),
            },
            "entries": entries,
        }
        return json.dumps(
            manifest,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    label = render_manifest()
    while (
        len(label.encode("utf-8"))
        > TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_BYTES
        and entries
    ):
        # Preserve both ends of a batch as long as possible.  Very heavily
        # escaped first lines can otherwise make 16 entries exceed 4096 bytes.
        removal_index = len(entries) // 2
        if len(entries) > 1 and removal_index == len(entries) - 1:
            removal_index -= 1
        del entries[removal_index]
        label = render_manifest()
    if len(label.encode("utf-8")) > TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_BYTES:
        raise RuntimeError("bounded Terminus command manifest exceeded its ABI limit")

    return label, {
        "policy": TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY,
        "mode": "manifest_v1",
        "applied": True,
        "batch_hash_encoding": TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING,
        "batch_sha256": batch_sha256,
        "command_count": command_count,
        "source_command_bytes": exact_bytes,
        "source_keystroke_bytes": batch_bytes,
        "rendered_command_bytes": len(label.encode("utf-8")),
        "manifest_entries": len(entries),
        "omitted_commands": command_count - len(entries),
        "limits": limits,
    }


def _initial_tool_observation_projection() -> dict[str, Any]:
    return {
        "policy": TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY,
        "information_lines_policy": (
            TERMINUS_TOOL_OBSERVATION_INFORMATION_LINES_POLICY
        ),
        "mode": "initial_prompt",
        "applied": False,
        "command_count": 0,
        "source_command_bytes": 0,
        "source_keystroke_bytes": 0,
        "rendered_command_bytes": 0,
        "manifest_entries": 0,
        "omitted_commands": 0,
        "information_lines_omitted": 0,
        "information_line_bytes_omitted": 0,
        "wire_information_lines_count": 0,
        "wire_information_line_bytes": 0,
        "model_information_lines_count": 0,
        "canonicalization_stage": "specforge_before_qwen_tokenization",
    }


def _new_tool_observation_projection_totals() -> dict[str, int]:
    return {
        "committed_observations": 0,
        "exact_single_observations": 0,
        "exact_batch_observations": 0,
        "manifest_observations": 0,
        "feedback_observations": 0,
        "no_command_observations": 0,
        "source_command_bytes": 0,
        "source_keystroke_bytes": 0,
        "rendered_command_bytes": 0,
        "command_bytes_avoided": 0,
        "command_projection_overhead_bytes": 0,
        "manifest_entries": 0,
        "omitted_commands": 0,
        "information_lines_omitted": 0,
        "information_line_bytes_omitted": 0,
        "wire_information_lines_count": 0,
        "wire_information_line_bytes": 0,
        "model_information_lines_count": 0,
    }


@dataclass(frozen=True)
class _PendingTerminusGeneration:
    """One generated response that is not durable until its await succeeds."""

    candidate_messages: tuple[dict[str, str], ...]
    generation_messages: tuple[dict[str, str], ...]
    assistant_content: str
    round_index: int
    had_prior_round: bool
    compression: dict[str, Any]
    receipt: dict[str, Any]
    response: LLMResponse
    tool_observation_projection: dict[str, Any]


@dataclass
class _InspectionRecord:
    epoch: int
    rounds: list[int]
    output_sha256: str
    last_cued_occurrence: int = 0


class TerminusR2EActLLM(BaseLLM):
    """Stateful Harbor LLM backend backed by SpecForge encode/recover calls."""

    def __init__(
        self,
        *,
        logs_dir: Path,
        model_name: str,
        sglang_url: str,
        max_new_tokens: int,
        remaining_context_output_ceiling_tokens: int | None,
        working_memory_turns: int,
        request_timeout: float,
        use_remaining_context_for_generation: bool,
        post_compression_generation_policy: str,
        capacity_pressure_reserve_tokens: int,
        oversized_complete_round_eviction: bool,
        collect_rollout_details: bool,
        system_prompt: str,
        configured_max_turns: int | None,
    ) -> None:
        super().__init__()
        self.logs_dir = Path(logs_dir)
        self.model_name = str(model_name)
        self.sglang_url = str(sglang_url).rstrip("/")
        self.max_new_tokens = int(max_new_tokens)
        self.remaining_context_output_ceiling_tokens = (
            None
            if remaining_context_output_ceiling_tokens is None
            else int(remaining_context_output_ceiling_tokens)
        )
        self.working_memory_turns = int(working_memory_turns)
        self.request_timeout = float(request_timeout)
        self.use_remaining_context_for_generation = bool(
            use_remaining_context_for_generation
        )
        self.post_compression_generation_policy = str(
            post_compression_generation_policy
        )
        self.capacity_pressure_reserve_tokens = int(capacity_pressure_reserve_tokens)
        self.oversized_complete_round_eviction = bool(
            oversized_complete_round_eviction
        )
        self.collect_rollout_details = bool(collect_rollout_details)
        self.system_prompt = str(system_prompt)
        self.configured_max_turns = (
            None if configured_max_turns is None else int(configured_max_turns)
        )

        self.specforge_root: Path | None = None
        self.checkpoint: Path | None = None
        self.corpus_dir: Path | None = None
        self._bundle: Any | None = None
        self._memory_runner: Any | None = None
        self._tool_result_type: type[Any] | None = None
        self._episode_id: str | None = None
        self._messages: list[dict[str, str]] = []
        self._round_index = 1
        self._pending_commands: list[dict[str, Any]] | None = None
        self._pending_timeout = False
        self._receipts: list[dict[str, Any]] = []
        self._memory_revision = 0
        self._memory_lock = threading.RLock()
        self._artifact_errors: list[dict[str, str]] = []
        self._memory_query_error: str | None = None
        self._memory_lookup_count = 0
        self._memory_lookup_failure_count = 0
        self._last_memory_lookup_round: int | None = None
        self._last_memory_lookup_exit_code: int | None = None
        self._progress_guard_metadata: dict[str, Any] = {}
        self._tool_observation_projection_totals = (
            _new_tool_observation_projection_totals()
        )
        self._request_cancellation_configured = False
        self._cancel_requested = threading.Event()
        self._active_worker: asyncio.Task[Any] | None = None
        self._closed = True

    @property
    def memory_runner(self) -> Any | None:
        return self._memory_runner

    @property
    def memory_query_error(self) -> str | None:
        return self._memory_query_error

    @property
    def messages(self) -> tuple[dict[str, str], ...]:
        return tuple(dict(message) for message in self._messages)

    @property
    def receipts(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(receipt) for receipt in self._receipts)

    def configure_paths(
        self,
        *,
        specforge_root: Path,
        checkpoint: Path,
        corpus_dir: Path,
    ) -> None:
        self.specforge_root = Path(specforge_root)
        self.checkpoint = Path(checkpoint)
        self.corpus_dir = Path(corpus_dir)

    def begin_run(self) -> None:
        if not self._closed:
            raise RuntimeError("a Terminus r^2eAct episode is already active")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.logs_dir / R2EACT_MEMORY_FILENAME, b"")
        _atomic_write(self.logs_dir / TERMINUS_R2EACT_EVENTS_FILENAME, b"")
        self._episode_id = f"terminus-r2eact-{uuid.uuid4().hex}"
        self._messages = []
        self._round_index = 1
        self._pending_commands = None
        self._pending_timeout = False
        self._receipts = []
        self._memory_revision = 0
        self._artifact_errors = []
        self._memory_query_error = None
        self._memory_lookup_count = 0
        self._memory_lookup_failure_count = 0
        self._last_memory_lookup_round = None
        self._last_memory_lookup_exit_code = None
        self._progress_guard_metadata = {}
        self._tool_observation_projection_totals = (
            _new_tool_observation_projection_totals()
        )
        self._cancel_requested.clear()
        self._closed = False

    def set_progress_guard_metadata(self, value: Mapping[str, Any]) -> None:
        self._progress_guard_metadata = dict(value)

    def note_memory_lookup(
        self,
        *,
        query: str,
        exit_code: int | None,
        succeeded: bool,
        error: str | None = None,
    ) -> None:
        """Persist bounded telemetry for the controller-routed archive action."""

        action_round = max(1, self._round_index - 1)
        self._memory_lookup_count += 1
        if not succeeded:
            self._memory_lookup_failure_count += 1
        self._last_memory_lookup_round = action_round
        self._last_memory_lookup_exit_code = (
            None if exit_code is None else int(exit_code)
        )
        self._event_sink(
            {
                "event": "terminus_memory_lookup",
                "episode_id": self._episode_id,
                "round_index": action_round,
                "payload": {
                    "query": str(query),
                    "query_sha256": hashlib.sha256(
                        str(query).encode("utf-8")
                    ).hexdigest(),
                    "exit_code": self._last_memory_lookup_exit_code,
                    "succeeded": bool(succeeded),
                    "error": None if error is None else str(error),
                },
            }
        )

    def note_progress_guard(
        self,
        *,
        trigger: str,
        round_index: int,
        payload: Mapping[str, Any],
    ) -> None:
        self._event_sink(
            {
                "event": "terminus_progress_guard",
                "episode_id": self._episode_id,
                "round_index": int(round_index),
                "payload": {"trigger": str(trigger), **dict(payload)},
            }
        )

    def note_terminal_execution(
        self,
        commands: Sequence[Command],
        *,
        timed_out: bool,
    ) -> None:
        self._pending_commands = [
            {
                "keystrokes": str(command.keystrokes),
                "duration_sec": float(command.duration_sec),
            }
            for command in commands
        ]
        self._pending_timeout = bool(timed_out)

    def get_model_context_limit(self) -> int:
        return QWEN36_TB20_CONTEXT_TOKENS

    def get_model_output_limit(self) -> int | None:
        return self.max_new_tokens

    def _record_artifact_error(self, stage: str, exc: BaseException) -> None:
        """Remember sidecar failures without invalidating an LLM response."""

        with self._memory_lock:
            if len(self._artifact_errors) >= 64:
                return
            self._artifact_errors.append(
                {
                    "stage": str(stage),
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    def _event_sink(self, event: Mapping[str, Any]) -> None:
        if self._cancel_requested.is_set():
            raise asyncio.CancelledError("r^2eAct generation cancelled")
        event_path = self.logs_dir / TERMINUS_R2EACT_EVENTS_FILENAME
        try:
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(event), ensure_ascii=False) + "\n")
        except Exception as exc:
            # The compressor may already have advanced its recursive state when
            # it emits an event. Raising here would make Terminus retry a model
            # response whose internal state was committed, creating a ghost
            # round. Sidecar telemetry is therefore explicitly best effort.
            self._record_artifact_error("event_append", exc)

    def _load_bundle(self) -> Any:
        if self._bundle is not None:
            if self.oversized_complete_round_eviction and getattr(
                self._bundle.generator,
                "oversized_complete_round_eviction_policy",
                None,
            ) != TERMINUS_R2EACT_OVERSIZED_ROUND_EVICTION_POLICY:
                raise RuntimeError(
                    "SpecForge lacks the pinned latest-successor grace policy; "
                    "refusing an ambiguous oversized-round runtime"
                )
            if self.oversized_complete_round_eviction and not callable(
                getattr(
                    self._bundle.generator,
                    "configure_oversized_complete_round_eviction",
                    None,
                )
            ):
                raise RuntimeError(
                    "SpecForge lacks oversized complete-round eviction; refusing "
                    "an unbounded raw working-memory round"
                )
            if not callable(
                getattr(
                    self._bundle.generator,
                    "prepare_tool_result_for_next_round",
                    None,
                )
            ):
                raise RuntimeError(
                    "SpecForge lacks Qwen3.6 tool-result admission; refusing an "
                    "incomplete r^2eAct rollout ABI"
                )
            self._configure_request_cancellation(self._bundle.generator)
            if self._memory_runner is None:
                if self._episode_id is None:
                    raise RuntimeError("Terminus r^2eAct episode was not initialized")
                self._memory_runner = self._bundle.tool_runner.restricted(
                    ("shell",), shell_corpus="react_memory"
                )
                self._memory_runner.begin_episode(self._episode_id)
            return self._bundle
        if (
            self.specforge_root is None
            or self.checkpoint is None
            or self.corpus_dir is None
        ):
            raise RuntimeError("Terminus r^2eAct paths were not configured")
        root_text = str(self.specforge_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        import context_compression.r2eact_sglang_live as sglang_live

        bundle = sglang_live.load_sglang_live_bundle(
            self.checkpoint,
            sglang_url=self.sglang_url,
            corpus_dir=self.corpus_dir,
            max_new_tokens=self.max_new_tokens,
            request_timeout=self.request_timeout,
            local_files_only=True,
            use_remaining_context_for_generation=(
                self.use_remaining_context_for_generation
            ),
            remaining_context_output_ceiling_tokens=(
                self.remaining_context_output_ceiling_tokens
            ),
        )
        if bundle.checkpoint_context_organization != sglang_live.QWEN36_TOKEN_CLOCK_V1:
            raise RuntimeError(
                "Terminus r^2eAct requires the deployed Qwen3.6 token-clock checkpoint"
            )
        if dict(sglang_live.QWEN36_GENERATION_SAMPLING) != QWEN36_TB20_SAMPLING:
            raise RuntimeError(
                "SpecForge Qwen3.6 sampling changed from the pinned policy"
            )

        generator = bundle.generator
        if self.oversized_complete_round_eviction and getattr(
            generator,
            "oversized_complete_round_eviction_policy",
            None,
        ) != TERMINUS_R2EACT_OVERSIZED_ROUND_EVICTION_POLICY:
            raise RuntimeError(
                "SpecForge lacks the pinned latest-successor grace policy; "
                "refusing an ambiguous oversized-round runtime"
            )
        configure_pressure = getattr(
            generator, "configure_capacity_pressure_eviction", None
        )
        if not callable(configure_pressure):
            raise RuntimeError(
                "SpecForge lacks capacity-pressure eviction; refusing to restore "
                "the absolute raw-working-memory overflow bug"
            )
        configure_oversized_round = getattr(
            generator,
            "configure_oversized_complete_round_eviction",
            None,
        )
        if self.oversized_complete_round_eviction and not callable(
            configure_oversized_round
        ):
            raise RuntimeError(
                "SpecForge lacks oversized complete-round eviction; refusing "
                "an unbounded raw working-memory round"
            )
        generator.configure_working_memory_turns(self.working_memory_turns)
        if callable(configure_oversized_round):
            configure_oversized_round(
                enabled=self.oversized_complete_round_eviction
            )
        generator.configure_post_compression_generation_policy(
            self.post_compression_generation_policy
        )
        generator.configure_max_turns(None)
        generator.configure_tools(())
        prepare_tool_result = getattr(
            generator, "prepare_tool_result_for_next_round", None
        )
        if not callable(prepare_tool_result):
            raise RuntimeError(
                "SpecForge lacks Qwen3.6 tool-result admission; refusing an "
                "incomplete r^2eAct rollout ABI"
            )
        configure_pressure(
            enabled=True,
            reserve_tokens=self.capacity_pressure_reserve_tokens,
        )
        self._configure_request_cancellation(generator)

        from context_compression.r2eact_tool import ToolResult

        memory_runner = bundle.tool_runner.restricted(
            ("shell",), shell_corpus="react_memory"
        )
        if self._episode_id is None:
            raise RuntimeError("Terminus r^2eAct episode was not initialized")
        memory_runner.begin_episode(self._episode_id)
        self._bundle = bundle
        self._memory_runner = memory_runner
        self._tool_result_type = ToolResult
        return bundle

    def _configure_request_cancellation(self, generator: Any) -> None:
        """Require and enable the hybrid's SGLang cancellation control plane."""

        if self._request_cancellation_configured:
            return
        client = getattr(generator, "client", None)
        configure = getattr(client, "configure_request_cancellation", None)
        abort_active = getattr(client, "abort_active_requests", None)
        if not callable(configure) or not callable(abort_active):
            raise RuntimeError(
                "SpecForge lacks abortable SGLang request IDs; refusing an "
                "unbounded Terminus cancellation path"
            )
        configure(enabled=True)
        self._request_cancellation_configured = True

    def _round_reminder(
        self, round_index: int
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Render the transactional cue for the immediately following response.

        The durable archive, rather than compressor bookkeeping, is the source
        of truth.  Requiring complete assistant/tool pairs and a contiguous
        prefix prevents a corrupt, partial, or future record from overstating
        what the model can retrieve.
        """

        current_round = max(1, int(round_index))
        memory = self._memory_metadata()
        prefix_end = memory.get("complete_round_prefix_end", 0)
        if not isinstance(prefix_end, int) or isinstance(prefix_end, bool):
            prefix_end = 0
        prefix_end = min(max(0, prefix_end), current_round - 1)

        if prefix_end > 0:
            archive_status = (
                f"Completed prior rounds 1-{prefix_end} have been compressed into "
                f"and preserved exactly in {R2EACT_MEMORY_VIRTUAL_PATH}."
            )
        else:
            archive_status = (
                "No complete prior round is archived yet; "
                f"{R2EACT_MEMORY_VIRTUAL_PATH} may be empty."
            )

        query_available = bool(memory.get("query_available", False))
        if query_available:
            retrieval = (
                "For an exact prior command, delimiter, path, error, or result, "
                "retrieve instead of guessing with one dedicated read-only action: "
                f"`grep -n -i -F -- 'NEEDLE' {R2EACT_MEMORY_VIRTUAL_PATH} | "
                "head -n 8` (or one narrow `sed -n` read). The archive lookup must "
                "be the only command in that action."
            )
        else:
            retrieval = (
                "Controller archive retrieval is currently unavailable; do not "
                "pretend that a lookup succeeded."
            )

        if self.configured_max_turns is None:
            limit_status = "No model-round limit is configured"
        else:
            limit_status = (
                f"A {self.configured_max_turns}-round model limit is configured"
            )
        text = (
            f"[r^2eAct round reminder] Current round: {current_round}. "
            "This marker applies only to the immediately following response and "
            "is emitted on every reached round with no reminder cutoff. "
            f"{limit_status}; task completion or the external trial timeout may "
            f"still stop execution. {archive_status} {retrieval}"
        )
        payload = {
            "policy": TERMINUS_R2EACT_ROUND_REMINDER_POLICY,
            "round_index": current_round,
            "reminder_round_cutoff": None,
            "configured_max_turns": self.configured_max_turns,
            "archive_prefix_start": 1 if prefix_end > 0 else None,
            "archive_prefix_end": prefix_end,
            "archive_records": int(memory.get("records", 0)),
            "archive_revision": int(memory.get("revision", 0)),
            "archive_query_available": query_available,
            "archive_virtual_path": R2EACT_MEMORY_VIRTUAL_PATH,
            "retrieval_action": "dedicated_grep_or_narrow_sed",
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        return text, payload

    def _tool_observation_fields(
        self,
        prompt: str,
        *,
        controller_cue: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        command_label, projection = _bounded_command_projection(
            self._pending_commands
        )
        rendered_prompt = str(prompt)
        if controller_cue:
            rendered_prompt = (
                str(controller_cue).rstrip() + "\n\n" + rendered_prompt
            )
        information_lines = [
            line for line in rendered_prompt.splitlines() if line.strip()
        ]
        projection.update(
            {
                "information_lines_policy": (
                    TERMINUS_TOOL_OBSERVATION_INFORMATION_LINES_POLICY
                ),
                "information_lines_omitted": len(information_lines),
                "information_line_bytes_omitted": sum(
                    len(line.encode("utf-8")) for line in information_lines
                ),
                "wire_information_lines_count": len(information_lines),
                "wire_information_line_bytes": sum(
                    len(line.encode("utf-8")) for line in information_lines
                ),
                "model_information_lines_count": 0,
                "canonicalization_stage": (
                    "specforge_before_qwen_tokenization"
                ),
            }
        )
        return {
            "command": command_label,
            "stdout": rendered_prompt,
            "stderr": (
                "Terminus command duration elapsed" if self._pending_timeout else ""
            ),
            # Tmux does not expose the command's exit status. -1 is an explicit
            # unavailable sentinel; it must not be confused with success.
            "exit_code": (124 if self._pending_timeout else -1),
            "timed_out": self._pending_timeout,
            "truncated": ("[... output limited to 10000 bytes;" in rendered_prompt),
            # The released ToolResult schema requires this exact redundant
            # copy.  SpecForge validates it first, then its canonical Qwen
            # projection clears the field before tokenization, so it does not
            # consume model context.
            "information_lines": information_lines,
        }, projection

    def _commit_tool_observation_projection(
        self, projection: Mapping[str, Any]
    ) -> None:
        """Aggregate only projections belonging to committed generations."""

        if not bool(projection.get("applied")):
            return
        totals = self._tool_observation_projection_totals
        totals["committed_observations"] += 1
        mode_key = {
            "exact_single": "exact_single_observations",
            "exact_batch_v1": "exact_batch_observations",
            "manifest_v1": "manifest_observations",
            "feedback": "feedback_observations",
            "no_commands": "no_command_observations",
        }.get(str(projection.get("mode")))
        if mode_key is not None:
            totals[mode_key] += 1
        for key in (
            "source_command_bytes",
            "source_keystroke_bytes",
            "rendered_command_bytes",
            "manifest_entries",
            "omitted_commands",
            "information_lines_omitted",
            "information_line_bytes_omitted",
            "wire_information_lines_count",
            "wire_information_line_bytes",
            "model_information_lines_count",
        ):
            value = projection.get(key, 0)
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] += max(0, value)
        source_bytes = projection.get("source_command_bytes", 0)
        rendered_bytes = projection.get("rendered_command_bytes", 0)
        if not isinstance(source_bytes, int) or isinstance(source_bytes, bool):
            source_bytes = 0
        if not isinstance(rendered_bytes, int) or isinstance(rendered_bytes, bool):
            rendered_bytes = 0
        totals["command_bytes_avoided"] += max(0, source_bytes - rendered_bytes)
        totals["command_projection_overhead_bytes"] += max(
            0, rendered_bytes - source_bytes
        )

    def _tool_observation_projection_metadata(self) -> dict[str, Any]:
        return {
            "policy": TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY,
            "information_lines_policy": (
                TERMINUS_TOOL_OBSERVATION_INFORMATION_LINES_POLICY
            ),
            "batch_hash_encoding": TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING,
            "exact_projection_max_bytes": (
                TERMINUS_TOOL_OBSERVATION_EXACT_MAX_BYTES
            ),
            "manifest_first_line_max_bytes": (
                TERMINUS_TOOL_OBSERVATION_MANIFEST_FIRST_LINE_MAX_BYTES
            ),
            "manifest_max_entries": TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES,
            "manifest_max_bytes": TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_BYTES,
            "totals": dict(self._tool_observation_projection_totals),
        }

    def _consume_pending_observation(self) -> None:
        self._pending_commands = None
        self._pending_timeout = False

    def _prepare_tool_result_for_next_round(
        self,
        *,
        generator: Any,
        messages: Sequence[Mapping[str, str]],
        tool_result: Any,
        episode_id: str,
        round_index: int,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Apply the compressed arm's context-admission policy.

        The raw-history control overrides this narrow hook with an identity
        projection.  Keeping the rest of ``_generate_sync`` shared makes the
        Terminus/Qwen transcript and transactional commit path identical
        between the two arms.
        """

        prepare = getattr(generator, "prepare_tool_result_for_next_round", None)
        if not callable(prepare):
            raise RuntimeError("Qwen3.6 tool-result admission is unavailable")
        prepared = prepare(
            messages=tuple(dict(message) for message in messages),
            tool_result=tool_result,
            episode_id=episode_id,
            round_index=round_index,
        )
        telemetry: Mapping[str, Any] | None = None
        if isinstance(prepared, tuple):
            if len(prepared) != 2:
                raise TypeError(
                    "tool-result admission must return ToolResult or "
                    "(ToolResult, telemetry)"
                )
            prepared, telemetry = prepared
        if telemetry is not None and not isinstance(telemetry, Mapping):
            raise TypeError("tool-result admission telemetry must be an object")
        return prepared, None if telemetry is None else dict(telemetry)

    def _persist_memory(self, *, allow_empty: bool = True) -> None:
        target = self.logs_dir / R2EACT_MEMORY_FILENAME
        runner = self._memory_runner
        if runner is None:
            if allow_empty and not target.exists():
                _atomic_write(target, b"")
            return
        source = Path(runner.corpus_dir) / R2EACT_MEMORY_FILENAME
        if source.is_file():
            payload = source.read_bytes()
        elif allow_empty and (not target.exists() or target.stat().st_size == 0):
            payload = b""
        else:
            return
        _atomic_write(target, payload)

    def _commit_archive(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        compression: Mapping[str, Any],
        round_index: int,
    ) -> None:
        if self._memory_runner is None or self._episode_id is None:
            exc = RuntimeError("Terminus r^2eAct memory sidecar is unavailable")
            self._memory_query_error = str(exc)
            self._record_artifact_error("memory_update", exc)
            return
        # This happens immediately after the compression receipt, including on
        # final-answer/parser-error paths.  It therefore does not depend on a
        # later valid tool call to make the archive durable.
        try:
            self._memory_runner.update_compressed_episode_memory(
                episode_id=self._episode_id,
                messages=messages,
                round_index=round_index,
                compression=compression,
            )
        except Exception as exc:
            self._memory_query_error = (
                "memory archive update failed; refusing stale recall"
            )
            self._record_artifact_error("memory_update", exc)
            return
        try:
            with self._memory_lock:
                self._persist_memory()
                self._memory_revision += 1
        except Exception as exc:
            self._record_artifact_error("memory_snapshot", exc)

    @staticmethod
    def _response_parts(generation: Any) -> tuple[str, str | None]:
        parsed = generation.parsed
        answer = getattr(parsed, "answer_text", None)
        reasoning = getattr(parsed, "think_text", None)
        if isinstance(answer, str):
            content = answer
        else:
            # Preserve malformed/capped text so Terminus's own tolerant parser
            # and parser-feedback loop remain authoritative.
            content = str(generation.assistant_content)
            close = content.find("</think>")
            if close >= 0:
                if not isinstance(reasoning, str) or not reasoning.strip():
                    opening = content.find("<think>")
                    reasoning = content[
                        opening + len("<think>") if opening >= 0 else 0 : close
                    ].strip()
                content = content[close + len("</think>") :].strip()
        normalized_reasoning = (
            reasoning.strip()
            if isinstance(reasoning, str) and reasoning.strip()
            else None
        )
        return content, normalized_reasoning

    def _response_extra(
        self,
        *,
        compression: Mapping[str, Any],
        round_index: int,
        tool_result_admission: Mapping[str, Any] | None,
        tool_observation_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build arm-specific Harbor response metadata."""

        return {
            "r2eact_compression": dict(compression),
            "r2eact_round_index": int(round_index),
            "r2eact_prompt_abi": TERMINUS_R2EACT_PROMPT_ABI,
            "r2eact_input_tool_context_fit": (
                None
                if tool_result_admission is None
                else dict(tool_result_admission)
            ),
            "r2eact_tool_observation_projection": dict(
                tool_observation_projection
            ),
        }

    def _generate_sync(self, prompt: str) -> _PendingTerminusGeneration:
        if self._cancel_requested.is_set():
            raise asyncio.CancelledError("r^2eAct generation cancelled")
        bundle = self._load_bundle()
        if self._episode_id is None:
            raise RuntimeError("Terminus r^2eAct episode is unavailable")

        round_index = self._round_index
        round_reminder, round_reminder_metadata = self._round_reminder(round_index)
        has_prior_round = bool(self._messages)
        if not has_prior_round:
            # The compressed arm prefixes its first transactional round cue.
            # The raw-history control disables that cue and therefore keeps the
            # complete stock Terminus prompt verbatim.
            initial_prompt = str(prompt)
            if round_reminder:
                initial_prompt = (
                    round_reminder.rstrip() + "\n\n" + initial_prompt
                )
                if round_reminder_metadata is not None:
                    round_reminder_metadata["delivery"] = (
                        "initial_user_prefix_exact"
                    )
            candidate_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": initial_prompt},
            ]
        else:
            if self._messages[-1].get("role") != "assistant":
                raise RuntimeError("Terminus r^2eAct transcript is not append-only")
            candidate_messages = [*self._messages]

        tool_result_admission: dict[str, Any] | None = None
        tool_observation_projection = _initial_tool_observation_projection()
        started = time.perf_counter()
        with bundle.lock:
            if has_prior_round:
                fields, tool_observation_projection = self._tool_observation_fields(
                    prompt,
                    controller_cue=round_reminder,
                )
                tool_result_type = self._tool_result_type
                if tool_result_type is None:
                    from context_compression.r2eact_tool import ToolResult

                    tool_result_type = ToolResult
                    self._tool_result_type = ToolResult
                prepared, tool_result_admission = (
                    self._prepare_tool_result_for_next_round(
                        generator=bundle.generator,
                        messages=self._messages,
                        tool_result=tool_result_type(**fields),
                        episode_id=self._episode_id,
                        round_index=round_index - 1,
                    )
                )
                if not isinstance(prepared, tool_result_type):
                    raise TypeError("tool-result admission returned an invalid result")
                if round_reminder:
                    prepared_stdout = getattr(prepared, "stdout", None)
                    if not isinstance(prepared_stdout, str) or not (
                        prepared_stdout.startswith(round_reminder)
                    ):
                        raise RuntimeError(
                            "tool-result admission cropped the r^2eAct round "
                            "reminder; refusing an unauditable generation"
                        )
                    if round_reminder_metadata is not None:
                        round_reminder_metadata["delivery"] = (
                            "admitted_tool_stdout_prefix_exact"
                        )
                tool_payload = prepared.to_payload()
                candidate_messages.append({"role": "tool", "content": tool_payload})

            # Do not mutate the durable ledger until generation returns. Harbor
            # may retry a transient pre-response exception with the same prompt;
            # a half-appended observation would break the compressor transcript.
            generation_messages = tuple(dict(message) for message in candidate_messages)
            generation = bundle.generator.generate_next(
                generation_messages,
                episode_id=self._episode_id,
                round_index=round_index,
                event_sink=self._event_sink,
            )
            if self._cancel_requested.is_set():
                raise asyncio.CancelledError("r^2eAct generation cancelled")
        elapsed = time.perf_counter() - started
        compression = dict(generation.compression)
        receipt = {
            "round_index": round_index,
            "elapsed_seconds": elapsed,
            "token_count": len(generation.token_ids),
            "finished_on_eos": bool(generation.finished_on_eos),
            "hit_generation_cap": bool(generation.hit_generation_cap),
            "finish_reason": _finish_reason_kind(generation.finish_reason),
            "input_tool_context_fit": tool_result_admission,
            "tool_observation_projection": dict(tool_observation_projection),
            "compression": compression,
        }
        if round_reminder_metadata is not None:
            receipt["round_reminder"] = dict(round_reminder_metadata)
        content, reasoning = self._response_parts(generation)
        prompt_tokens = compression.get("student_prefix_tokens", 0)
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
            prompt_tokens = 0
        response_extra = self._response_extra(
            compression=compression,
            round_index=round_index,
            tool_result_admission=tool_result_admission,
            tool_observation_projection=tool_observation_projection,
        )
        if round_reminder_metadata is not None:
            response_extra["r2eact_round_reminder"] = dict(
                round_reminder_metadata
            )
        response = LLMResponse(
            content=content,
            reasoning_content=reasoning,
            model_name=self.model_name,
            usage=UsageInfo(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=len(generation.token_ids),
                cache_tokens=0,
                cost_usd=0.0,
            ),
            completion_token_ids=(
                list(generation.token_ids) if self.collect_rollout_details else None
            ),
            extra=response_extra,
        )
        return _PendingTerminusGeneration(
            candidate_messages=tuple(dict(message) for message in candidate_messages),
            generation_messages=generation_messages,
            assistant_content=str(generation.assistant_content),
            round_index=round_index,
            had_prior_round=has_prior_round,
            compression=compression,
            receipt=receipt,
            response=response,
            tool_observation_projection=dict(tool_observation_projection),
        )

    def _commit_generation(self, pending: _PendingTerminusGeneration) -> LLMResponse:
        """Commit only a candidate whose shielded await reached the caller."""

        if self._cancel_requested.is_set():
            raise asyncio.CancelledError("r^2eAct generation cancelled")
        if pending.round_index != self._round_index:
            raise RuntimeError("Terminus r^2eAct generation committed out of order")
        self._commit_archive(
            messages=pending.generation_messages,
            compression=pending.compression,
            round_index=pending.round_index,
        )
        self._messages = [
            *(dict(message) for message in pending.candidate_messages),
            {"role": "assistant", "content": pending.assistant_content},
        ]
        if pending.had_prior_round:
            self._consume_pending_observation()
        self._commit_tool_observation_projection(
            pending.tool_observation_projection
        )
        self._round_index += 1
        self._receipts.append(dict(pending.receipt))
        self._event_sink(
            {
                "event": "terminus_llm_response",
                "episode_id": self._episode_id,
                "round_index": pending.round_index,
                "payload": pending.receipt,
            }
        )
        return pending.response

    async def call(self, prompt: str, **kwargs: Any) -> LLMResponse:
        del kwargs  # Terminus Chat history is intentionally projected from our ledger.
        if self._closed:
            raise RuntimeError("Terminus r^2eAct run has not started")
        worker = asyncio.create_task(asyncio.to_thread(self._generate_sync, prompt))
        self._active_worker = worker
        try:
            pending = await asyncio.shield(worker)
            return self._commit_generation(pending)
        except asyncio.CancelledError:
            self._cancel_requested.set()
            client = getattr(getattr(self._bundle, "generator", None), "client", None)
            abort_active = getattr(client, "abort_active_requests", None)
            if callable(abort_active):
                # A cancellation can land during the client's idle-cache
                # flush, before `/generate` has registered its rid. Poll until
                # every new request becomes abortable or the worker exits; one
                # early empty snapshot must not permit a later encode/recover
                # or target generation to start without another abort attempt.
                abort_failures = 0
                while not worker.done():
                    abort_worker = asyncio.create_task(asyncio.to_thread(abort_active))
                    while True:
                        try:
                            aborted = await asyncio.shield(abort_worker)
                            break
                        except asyncio.CancelledError:
                            # Harbor may repeat cancellation while the abort
                            # control request is in flight. Keep collecting it
                            # instead of detaching another background thread.
                            continue
                        except Exception as exc:
                            self._record_artifact_error("sglang_abort_request", exc)
                            aborted = ()
                            abort_failures += 1
                            break
                    if aborted:
                        abort_failures = 0
                    wait_seconds = (
                        0.25 * (2 ** min(abort_failures - 1, 3))
                        if abort_failures
                        else 0.05
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(worker), timeout=wait_seconds
                        )
                    except TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    # Aborting an in-flight HTTP request commonly makes the
                    # generation worker finish with a transport exception.
                    # Harbor's cancellation remains the authoritative cause.
                    break
            with contextlib.suppress(BaseException):
                await worker
            raise
        finally:
            self._active_worker = None

    def _memory_metadata(self) -> dict[str, Any]:
        path = self.logs_dir / R2EACT_MEMORY_FILENAME
        read_error: str | None = None
        try:
            payload = path.read_bytes() if path.is_file() else b""
        except OSError as exc:
            payload = b""
            read_error = f"{type(exc).__name__}: {exc}"
        records = [line for line in payload.splitlines() if line.strip()]
        max_round = 0
        parsed_records = 0
        roles_by_round: dict[int, set[str]] = {}
        for line in records:
            with contextlib.suppress(Exception):
                value = json.loads(line)
                if isinstance(value, Mapping):
                    parsed_records += 1
                    round_value = value.get("round")
                    if (
                        isinstance(round_value, int)
                        and not isinstance(round_value, bool)
                        and round_value > 0
                    ):
                        max_round = max(max_round, round_value)
                        role = value.get("role")
                        if isinstance(role, str):
                            roles_by_round.setdefault(round_value, set()).add(role)
        complete_round_prefix_end = 0
        while {"assistant", "tool"}.issubset(
            roles_by_round.get(complete_round_prefix_end + 1, set())
        ):
            complete_round_prefix_end += 1
        return {
            "path": str(path),
            "virtual_path": R2EACT_MEMORY_VIRTUAL_PATH,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "records": len(records),
            "parsed_records": parsed_records,
            "max_round": max_round,
            "complete_round_prefix_end": complete_round_prefix_end,
            "complete_round_prefix_roles": ["assistant", "tool"],
            "revision": self._memory_revision,
            "query_available": (
                self._memory_query_error is None and read_error is None
            ),
            "query_error": self._memory_query_error,
            "snapshot_read_error": read_error,
        }

    def metadata(self) -> dict[str, Any]:
        latest_round_reminder: dict[str, Any] | None = None
        if self._receipts:
            candidate = self._receipts[-1].get("round_reminder")
            if isinstance(candidate, Mapping):
                latest_round_reminder = dict(candidate)
        serving = (
            None
            if self._bundle is None
            else _serving_capacity_metadata(
                self._bundle.generator,
                requested_max_new_tokens=(
                    self.remaining_context_output_ceiling_tokens or self.max_new_tokens
                ),
            )
        )
        return {
            "agent": TERMINUS_R2EACT_NAME,
            "agent_version": TERMINUS_R2EACT_VERSION,
            "terminus_base_version": "2.0.0",
            "prompt_abi": TERMINUS_R2EACT_PROMPT_ABI,
            "context_abi": TERMINUS_R2EACT_CONTEXT_ABI,
            "terminal_transport": _terminal_transport_metadata(),
            "system_prompt_sha256": hashlib.sha256(
                self.system_prompt.encode("utf-8")
            ).hexdigest(),
            "working_memory_turns_configured": self.working_memory_turns,
            "working_memory_turns_semantics": "maximum_not_absolute",
            "capacity_pressure_reserve_tokens": (self.capacity_pressure_reserve_tokens),
            "oversized_complete_round_eviction": {
                "enabled": self.oversized_complete_round_eviction,
                "policy": TERMINUS_R2EACT_OVERSIZED_ROUND_EVICTION_POLICY,
                "threshold_formula": (
                    "floor(min(server_context_length,server_max_total_num_tokens)"
                    "/(working_memory_turns+1))"
                ),
                "whole_rounds_only": True,
                "eviction_order": "left_to_right",
                "latest_round_successor_grace": True,
                "latest_round_capacity_fallback": (
                    "evict_older_first_then_latest_only_if_still_required"
                ),
                "encoder_block_tokens": 1_024,
                "changes_generation_budget": False,
            },
            "post_compression_generation_policy": (
                self.post_compression_generation_policy
            ),
            "generation_sampling": dict(QWEN36_TB20_SAMPLING),
            "collect_rollout_details": self.collect_rollout_details,
            "terminus_tool_exit_status_policy": "unavailable_sentinel_minus_one",
            "tool_result_admission": "qwen36_prepare_tool_result_for_next_round",
            "tool_observation_projection": (
                self._tool_observation_projection_metadata()
            ),
            "cancellation_policy": ("sglang_abort_each_new_request_until_worker_drain"),
            "request_cancellation_configured": (self._request_cancellation_configured),
            "serving_capacity": serving,
            "rounds": len(self._receipts),
            "memory": self._memory_metadata(),
            "memory_lookup": {
                "count": self._memory_lookup_count,
                "failure_count": self._memory_lookup_failure_count,
                "last_round": self._last_memory_lookup_round,
                "last_exit_code": self._last_memory_lookup_exit_code,
            },
            "round_reminder": {
                "policy": TERMINUS_R2EACT_ROUND_REMINDER_POLICY,
                "emission": "every_reached_model_round",
                "reminder_round_cutoff": None,
                "configured_max_turns": self.configured_max_turns,
                "committed_count": len(self._receipts),
                "latest": latest_round_reminder,
            },
            "progress_guard": dict(self._progress_guard_metadata),
            "artifact_errors": [dict(item) for item in self._artifact_errors],
        }

    def close_sync(self, *, reason: str) -> None:
        if self._closed:
            return
        try:
            try:
                with self._memory_lock:
                    self._persist_memory(allow_empty=False)
            except Exception as exc:
                self._record_artifact_error("memory_final_snapshot", exc)
            if self._memory_runner is not None and self._episode_id is not None:
                try:
                    self._memory_runner.end_episode(self._episode_id)
                except Exception as exc:
                    self._record_artifact_error("memory_end_episode", exc)
            receipts_path = self.logs_dir / TERMINUS_R2EACT_RECEIPTS_FILENAME
            try:
                _atomic_write(
                    receipts_path,
                    json.dumps(
                        {
                            "schema": "terminus_r2eact_receipts_v1",
                            "finalization_reason": str(reason),
                            "metadata": self.metadata(),
                            "receipts": self._receipts,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ).encode("utf-8"),
                )
            except Exception as exc:
                self._record_artifact_error("receipt_snapshot", exc)
        finally:
            self._memory_runner = None
            self._episode_id = None
            self._messages = []
            self._pending_commands = None
            self._closed = True


class TerminusR2EActAgent(Terminus2):
    """Harbor Terminus-2 harness with r^2eAct recursive context management."""

    SUPPORTS_ATIF = True
    INJECT_MEMORY_PROMPT = True
    CONTEXT_METADATA_KEY = "r2eact"

    @staticmethod
    def name() -> str:
        return TERMINUS_R2EACT_NAME

    def version(self) -> str:
        return TERMINUS_R2EACT_VERSION

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        specforge_root: str | Path | None = None,
        checkpoint: str | Path | None = None,
        corpus_dir: str | Path | None = None,
        sglang_url: str = DEFAULT_SGLANG_URL,
        max_turns: int | None = None,
        max_new_tokens: int | str = QWEN36_TB20_MAX_NEW_TOKENS,
        remaining_context_output_ceiling_tokens: int | str | None = (
            QWEN36_TB20_MAX_NEW_TOKENS
        ),
        working_memory_turns: int | str = (
            TERMINUS_R2EACT_DEFAULT_WORKING_MEMORY_TURNS
        ),
        request_timeout: float | str = 1_800.0,
        use_remaining_context_for_generation: bool | str | int = True,
        post_compression_generation_policy: str = (
            R2EACT_POST_COMPRESSION_GENERATION_POLICY
        ),
        capacity_pressure_reserve_tokens: int | str = (
            TERMINUS_R2EACT_MEMORY_RESERVE_TOKENS
        ),
        oversized_complete_round_eviction: bool | str | int = True,
        stagnation_cue_rounds: int | str = (
            TERMINUS_R2EACT_DEFAULT_STAGNATION_CUE_ROUNDS
        ),
        exact_repeat_cue_count: int | str = (
            TERMINUS_R2EACT_DEFAULT_EXACT_REPEAT_CUE_COUNT
        ),
        process_lock: str | Path = DEFAULT_PROCESS_LOCK,
        parser_name: str = "json",
        enable_summarize: bool = False,
        temperature: float | None = None,
        collect_rollout_details: bool | str | int = False,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        if parser_name != "json":
            raise ValueError("Terminus r^2eAct currently supports the JSON parser only")
        if enable_summarize:
            raise ValueError(
                "Terminus summarization must remain disabled because r^2eAct owns context"
            )
        if temperature not in (None, 1.0):
            raise ValueError("Qwen3.6 evaluation sampling fixes temperature at 1.0")
        if kwargs.get("record_terminal_session") not in (None, False):
            raise ValueError(
                "inline terminal recording is disabled: it must not sit between "
                "tmux and the task shell; use terminus_2.pane and transport "
                "receipts instead"
            )
        kwargs["record_terminal_session"] = False
        trajectory_config = kwargs.get("trajectory_config")
        if (
            isinstance(trajectory_config, Mapping)
            and trajectory_config.get("raw_content") is True
        ):
            raise ValueError(
                "Terminus r^2eAct requires structured ATIF tool calls; "
                "trajectory raw_content must remain disabled"
            )
        parsed_max_new = _positive_int(max_new_tokens, name="max_new_tokens")
        parsed_output_ceiling = _optional_positive_int(
            remaining_context_output_ceiling_tokens,
            name="remaining_context_output_ceiling_tokens",
        )
        parsed_working_memory = _positive_int(
            working_memory_turns, name="working_memory_turns", maximum=32
        )
        parsed_request_timeout = _positive_float(
            request_timeout, name="request_timeout"
        )
        parsed_remaining = _as_bool(
            use_remaining_context_for_generation,
            name="use_remaining_context_for_generation",
        )
        parsed_collect_rollout_details = _as_bool(
            collect_rollout_details,
            name="collect_rollout_details",
        )
        if not parsed_remaining:
            raise ValueError(
                "the full-SGLang recovery policy requires remaining-context generation"
            )
        if post_compression_generation_policy != (
            R2EACT_POST_COMPRESSION_GENERATION_POLICY
        ):
            raise ValueError(
                "Terminus r^2eAct requires the full-SGLang post-compression policy"
            )
        parsed_reserve = _positive_int(
            capacity_pressure_reserve_tokens,
            name="capacity_pressure_reserve_tokens",
        )
        parsed_oversized_round_eviction = _as_bool(
            oversized_complete_round_eviction,
            name="oversized_complete_round_eviction",
        )
        self._stagnation_cue_rounds = _positive_int(
            stagnation_cue_rounds,
            name="stagnation_cue_rounds",
        )
        self._exact_repeat_cue_count = _positive_int(
            exact_repeat_cue_count,
            name="exact_repeat_cue_count",
            maximum=32,
        )
        if self._exact_repeat_cue_count < 2:
            raise ValueError("exact_repeat_cue_count must be at least two")
        resolved_model_name = model_name or "qwen36-27b-compressor-r16-step1750"
        self._r2eact_bootstrap = {
            "logs_dir": Path(logs_dir),
            "model_name": resolved_model_name,
            "sglang_url": sglang_url,
            "max_new_tokens": parsed_max_new,
            "remaining_context_output_ceiling_tokens": parsed_output_ceiling,
            "working_memory_turns": parsed_working_memory,
            "request_timeout": parsed_request_timeout,
            "use_remaining_context_for_generation": parsed_remaining,
            "post_compression_generation_policy": (post_compression_generation_policy),
            "capacity_pressure_reserve_tokens": parsed_reserve,
            "oversized_complete_round_eviction": (
                parsed_oversized_round_eviction
            ),
            "collect_rollout_details": parsed_collect_rollout_details,
            "configured_max_turns": max_turns,
            "system_prompt": (
                _memory_system_prompt(parsed_working_memory)
                if self.INJECT_MEMORY_PROMPT
                else ""
            ),
        }
        self._specforge_root_value = specforge_root
        self._checkpoint_value = checkpoint
        self._corpus_dir_value = corpus_dir
        self._process_lock = Path(process_lock).expanduser().resolve()
        super().__init__(
            logs_dir=Path(logs_dir),
            model_name=resolved_model_name,
            max_turns=max_turns,
            parser_name=parser_name,
            temperature=temperature,
            collect_rollout_details=parsed_collect_rollout_details,
            enable_summarize=False,
            proactive_summarization_threshold=0,
            suppress_max_turns_warning=True,
            extra_env=extra_env,
            **kwargs,
        )
        if not isinstance(self._llm, TerminusR2EActLLM):
            raise RuntimeError("Terminus installed an unexpected LLM backend")
        self._r2eact_llm = self._llm
        # Preserve the official Terminus JSON prompt body. The compressed arm
        # prefixes its per-round controller marker at the LLM boundary; the raw
        # control disables it and remains verbatim.
        marker = "\nTask Description:\n"
        if marker not in self._prompt_template:
            raise RuntimeError("Harbor Terminus-2 prompt template changed")
        self._reset_progress_guard_state()

    def _init_llm(self, *args: Any, **kwargs: Any) -> BaseLLM:
        del args, kwargs
        return TerminusR2EActLLM(**self._r2eact_bootstrap)

    async def setup(self, environment: BaseEnvironment) -> None:
        """Start a direct, verified tmux data plane with no inline PTY relay."""

        self._session = VerifiedTmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2.pane",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
            pane_width=self._tmux_pane_width,
            pane_height=self._tmux_pane_height,
            extra_env=self._extra_env,
            user=environment.default_user,
            transport_receipts_path=(
                Path(self._r2eact_bootstrap["logs_dir"])
                / TERMINUS_TERMINAL_TRANSPORT_RECEIPTS_FILENAME
            ),
        )
        await self._session.start()

    def _reset_progress_guard_state(self) -> None:
        self._progress_epoch = 0
        self._no_concrete_progress_rounds = 0
        self._inspection_history: dict[str, _InspectionRecord] = {}
        self._progress_guard_cues = 0
        self._progress_guard_exact_repeat_cues = 0
        self._progress_guard_stagnation_cues = 0
        self._progress_guard_memory_batch_rejections = 0
        self._progress_guard_max_no_concrete_progress_rounds = 0
        self._progress_guard_concrete_progress_rounds = 0
        self._progress_guard_concrete_progress_commands = 0
        self._progress_guard_known_inspection_commands = 0
        self._progress_guard_unknown_commands = 0
        self._last_stagnation_cue_bucket = 0

    def _reset_per_run_state(self) -> None:
        super()._reset_per_run_state()
        self._reset_progress_guard_state()

    def _progress_guard_snapshot(self) -> dict[str, Any]:
        current = self._no_concrete_progress_rounds
        maximum = self._progress_guard_max_no_concrete_progress_rounds
        return {
            "policy": TERMINUS_R2EACT_PROGRESS_GUARD_POLICY,
            "exact_repeat_cue_count": self._exact_repeat_cue_count,
            "stagnation_cue_rounds": self._stagnation_cue_rounds,
            "cues": self._progress_guard_cues,
            "exact_repeat_cues": self._progress_guard_exact_repeat_cues,
            "stagnation_cues": self._progress_guard_stagnation_cues,
            "memory_batch_rejections": (
                self._progress_guard_memory_batch_rejections
            ),
            "max_consecutive_no_concrete_progress_rounds": maximum,
            "current_consecutive_no_concrete_progress_rounds": current,
            # Compatibility aliases for 0.4.4 receipts.  In v2 these counters
            # include unknown commands as well as proven read-only inspection.
            "max_consecutive_inspection_only_rounds": maximum,
            "current_consecutive_inspection_only_rounds": current,
            "deprecated_telemetry_aliases": {
                "max_consecutive_inspection_only_rounds": (
                    "max_consecutive_no_concrete_progress_rounds"
                ),
                "current_consecutive_inspection_only_rounds": (
                    "current_consecutive_no_concrete_progress_rounds"
                ),
            },
            "stagnation_semantics": (
                "consecutive_rounds_without_explicit_task_artifact_edit_"
                "build_test_or_task_run"
            ),
            "concrete_progress_rounds": self._progress_guard_concrete_progress_rounds,
            "concrete_progress_commands": (
                self._progress_guard_concrete_progress_commands
            ),
            "known_inspection_commands": (
                self._progress_guard_known_inspection_commands
            ),
            "unknown_commands_without_progress_credit": (
                self._progress_guard_unknown_commands
            ),
            "stagnation_cue_bucket": self._last_stagnation_cue_bucket,
            "memory_lookup_resets_stagnation": False,
            "tracked_exact_inspection_commands": len(self._inspection_history),
            "blocking": False,
            "round_limit": None,
        }

    def _apply_progress_guard(
        self,
        *,
        commands: Sequence[Command],
        output: str,
    ) -> str:
        requested = [command for command in commands if command.keystrokes.strip()]
        if not requested:
            return output
        concrete_flags = [
            _command_has_concrete_progress(command.keystrokes)
            for command in requested
        ]
        if any(concrete_flags):
            self._progress_epoch += 1
            self._no_concrete_progress_rounds = 0
            self._last_stagnation_cue_bucket = 0
            self._progress_guard_concrete_progress_rounds += 1
            self._progress_guard_concrete_progress_commands += sum(concrete_flags)
            return output

        signatures = [
            _inspection_command_signature(command.keystrokes)
            for command in requested
        ]
        known_signature_count = sum(value is not None for value in signatures)
        unknown_count = len(signatures) - known_signature_count
        self._progress_guard_known_inspection_commands += known_signature_count
        self._progress_guard_unknown_commands += unknown_count
        self._no_concrete_progress_rounds += 1
        self._progress_guard_max_no_concrete_progress_rounds = max(
            self._progress_guard_max_no_concrete_progress_rounds,
            self._no_concrete_progress_rounds,
        )
        round_index = max(1, int(self._n_episodes))
        output_digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
        repeated: list[tuple[str, tuple[int, ...]]] = []
        for signature in dict.fromkeys(
            str(value) for value in signatures if value is not None
        ):
            record = self._inspection_history.get(signature)
            if record is None or record.epoch != self._progress_epoch:
                record = _InspectionRecord(
                    epoch=self._progress_epoch,
                    rounds=[],
                    output_sha256=output_digest,
                )
                self._inspection_history[signature] = record
            occurrence = len(record.rounds) + 1
            if (
                occurrence % self._exact_repeat_cue_count == 0
                and occurrence > record.last_cued_occurrence
            ):
                repeated.append((signature, tuple(record.rounds)))
                record.last_cued_occurrence = occurrence
            record.rounds.append(round_index)
            record.output_sha256 = output_digest

        hint_lines: list[str] = []
        triggers: list[str] = []
        if repeated:
            self._progress_guard_exact_repeat_cues += 1
            triggers.append("exact_inspection_repeat")
            rendered_repeats = "; ".join(
                f"`{signature[:200]}` previously ran in rounds "
                + ",".join(str(value) for value in prior_rounds[-4:])
                for signature, prior_rounds in repeated[:3]
            )
            hint_lines.append(rendered_repeats)
        stagnation_bucket = (
            self._no_concrete_progress_rounds // self._stagnation_cue_rounds
        )
        if (
            self._no_concrete_progress_rounds % self._stagnation_cue_rounds == 0
            and stagnation_bucket > self._last_stagnation_cue_bucket
        ):
            self._last_stagnation_cue_bucket = stagnation_bucket
            self._progress_guard_stagnation_cues += 1
            triggers.append("consecutive_no_concrete_progress_rounds")
            hint_lines.append(
                f"{self._no_concrete_progress_rounds} consecutive rounds contained "
                "no explicit task-artifact edit, build/test, or task execution."
            )
        if not hint_lines:
            return output

        self._progress_guard_cues += 1
        hint = (
            "[r^2eAct progress guard] "
            + " ".join(hint_lines)
            + " Before another repeated/broad inspection, use one dedicated "
            f"archive query against {R2EACT_MEMORY_VIRTUAL_PATH}; otherwise "
            "make or test a concrete task change now. Keep the next analysis concise."
        )
        self._r2eact_llm.note_progress_guard(
            trigger="+".join(triggers),
            round_index=round_index,
            payload={
                "no_concrete_progress_rounds": self._no_concrete_progress_rounds,
                # Compatibility alias; see the snapshot's alias map.
                "inspection_only_rounds": self._no_concrete_progress_rounds,
                "known_inspection_command_count": known_signature_count,
                "unknown_command_count": unknown_count,
                "concrete_progress_detected": False,
                "repeat_count": len(repeated),
                "stagnation_cue_bucket": self._last_stagnation_cue_bucket,
                "output_sha256": output_digest,
            },
        )
        return self._limit_output_length(output.rstrip() + "\n\n" + hint + "\n")

    def _path_setting(
        self,
        value: str | Path | None,
        env_name: str,
        default: Path,
        *legacy_env_names: str,
    ) -> Path:
        raw = value
        if raw is None:
            for candidate in (env_name, *legacy_env_names):
                raw = self._get_env(candidate)
                if raw is not None:
                    break
        return Path(raw if raw is not None else default).expanduser().resolve()

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

    async def _execute_commands(
        self,
        commands: list[Command],
        session: TmuxSession,
    ) -> tuple[bool, str]:
        memory_commands = [
            command
            for command in commands
            if _is_virtual_memory_query(command.keystrokes)
        ]
        if memory_commands:
            if len(commands) != 1:
                observation = (
                    "ERROR: a memory archive lookup must be the only command in "
                    "its Terminus action; no commands were executed"
                )
                self._progress_guard_memory_batch_rejections += 1
                self._r2eact_llm.note_memory_lookup(
                    query=str(memory_commands[0].keystrokes).rstrip("\r\n"),
                    exit_code=None,
                    succeeded=False,
                    error="mixed memory/task command batch rejected",
                )
                self._r2eact_llm.note_terminal_execution(commands, timed_out=False)
                return False, observation
            if self._r2eact_llm.memory_query_error is not None:
                observation = f"ERROR: {self._r2eact_llm.memory_query_error}"
                self._r2eact_llm.note_memory_lookup(
                    query=str(commands[0].keystrokes).rstrip("\r\n"),
                    exit_code=None,
                    succeeded=False,
                    error=self._r2eact_llm.memory_query_error,
                )
                self._r2eact_llm.note_terminal_execution(commands, timed_out=False)
                return False, observation
            runner = self._r2eact_llm.memory_runner
            if runner is None:
                observation = "ERROR: r^2eAct memory archive is not initialized"
                self._r2eact_llm.note_memory_lookup(
                    query=str(commands[0].keystrokes).rstrip("\r\n"),
                    exit_code=None,
                    succeeded=False,
                    error="memory archive is not initialized",
                )
                self._r2eact_llm.note_terminal_execution(commands, timed_out=False)
                return False, observation
            query = str(commands[0].keystrokes).rstrip("\r\n")
            lookup_exit_code: int | None = None
            lookup_succeeded = False
            lookup_error: str | None = None
            try:
                query = _one_terminal_line(commands[0].keystrokes)
            except ValueError as exc:
                observation = f"ERROR: {exc}; no command was executed"
                lookup_error = str(exc)
            else:
                translated = query.replace(
                    R2EACT_MEMORY_VIRTUAL_PATH, R2EACT_MEMORY_FILENAME
                )
                try:
                    result = runner.execute(translated)
                except Exception as exc:
                    self._r2eact_llm._record_artifact_error("memory_query", exc)
                    observation = (
                        "ERROR: memory archive lookup failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    lookup_error = f"{type(exc).__name__}: {exc}"
                else:
                    parts = []
                    if result.stdout:
                        parts.append(result.stdout)
                    if result.stderr:
                        parts.append(f"[stderr] {result.stderr}")
                    parts.append(f"[memory lookup exit_code={result.exit_code}]")
                    observation = "\n".join(parts)
                    lookup_exit_code = int(result.exit_code)
                    lookup_succeeded = result.exit_code == 0
                    if not lookup_succeeded:
                        lookup_error = f"memory lookup exit_code={result.exit_code}"
            self._r2eact_llm.note_memory_lookup(
                query=query,
                exit_code=lookup_exit_code,
                succeeded=lookup_succeeded,
                error=lookup_error,
            )
            # Archive recall is useful, but it is not a task-artifact edit,
            # build/test, or task execution.  Preserve (and do not increment)
            # the no-concrete-progress clock across the controller-only lookup.
            self._r2eact_llm.note_terminal_execution(commands, timed_out=False)
            return False, self._limit_output_length(observation)

        timed_out, output = await super()._execute_commands(commands, session)
        self._r2eact_llm.note_terminal_execution(commands, timed_out=timed_out)
        return timed_out, self._apply_progress_guard(
            commands=commands,
            output=output,
        )

    def _completion_was_confirmed(self) -> bool:
        """Return true only for Terminus's two consecutive completion marks."""

        completion_marks: list[bool] = []
        for step in self._trajectory_steps:
            if str(getattr(step, "source", "")) != "agent":
                continue
            tool_calls = getattr(step, "tool_calls", None) or ()
            completion_marks.append(
                any(
                    getattr(call, "function_name", None) == "mark_task_complete"
                    for call in tool_calls
                )
            )
        return len(completion_marks) >= 2 and completion_marks[-2:] == [True, True]

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        root, checkpoint, corpus = self._resolve_paths()
        self._r2eact_llm.configure_paths(
            specforge_root=root,
            checkpoint=checkpoint,
            corpus_dir=corpus,
        )
        self._r2eact_llm.begin_run()
        finalization_reason = "run_returned"
        try:
            async with _exclusive_process_lease(self._process_lock):
                await super().run(instruction, environment, context)
            finalization_reason = (
                "task_complete"
                if self._completion_was_confirmed()
                else "agent_loop_ended"
            )
        except asyncio.CancelledError:
            finalization_reason = "cancelled"
            raise
        except BaseException:
            finalization_reason = "error"
            raise
        finally:
            self._r2eact_llm.set_progress_guard_metadata(
                self._progress_guard_snapshot()
            )
            close_task = asyncio.create_task(
                asyncio.to_thread(
                    self._r2eact_llm.close_sync, reason=finalization_reason
                )
            )
            cancelled = False
            while not close_task.done():
                try:
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    cancelled = True
                    continue
            await close_task
            metadata = dict(context.metadata or {})
            metadata[self.CONTEXT_METADATA_KEY] = self._r2eact_llm.metadata()
            metadata[self.CONTEXT_METADATA_KEY]["finalization_reason"] = (
                finalization_reason
            )
            context.metadata = metadata
            if cancelled:
                raise asyncio.CancelledError


__all__ = [
    "TERMINUS_R2EACT_CONTEXT_ABI",
    "TERMINUS_R2EACT_DEFAULT_EXACT_REPEAT_CUE_COUNT",
    "TERMINUS_R2EACT_DEFAULT_STAGNATION_CUE_ROUNDS",
    "TERMINUS_R2EACT_DEFAULT_WORKING_MEMORY_TURNS",
    "TERMINUS_R2EACT_EVENTS_FILENAME",
    "TERMINUS_R2EACT_NAME",
    "TERMINUS_R2EACT_OVERSIZED_ROUND_EVICTION_POLICY",
    "TERMINUS_R2EACT_PROGRESS_GUARD_POLICY",
    "TERMINUS_R2EACT_PROMPT_ABI",
    "TERMINUS_R2EACT_RECEIPTS_FILENAME",
    "TERMINUS_R2EACT_ROUND_REMINDER_POLICY",
    "TERMINUS_R2EACT_VERSION",
    "TERMINUS_TERMINAL_TRANSPORT_ABI",
    "TERMINUS_TERMINAL_TRANSPORT_MAX_PASTE_BYTES",
    "TERMINUS_TERMINAL_TRANSPORT_POLICY",
    "TERMINUS_TERMINAL_TRANSPORT_PROBE_KEYSTROKE_BYTES",
    "TERMINUS_TERMINAL_TRANSPORT_RECEIPTS_FILENAME",
    "TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING",
    "TERMINUS_TOOL_OBSERVATION_EXACT_MAX_BYTES",
    "TERMINUS_TOOL_OBSERVATION_EXACT_SINGLE_MAX_BYTES",
    "TERMINUS_TOOL_OBSERVATION_INFORMATION_LINES_POLICY",
    "TERMINUS_TOOL_OBSERVATION_MANIFEST_FIRST_LINE_MAX_BYTES",
    "TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_BYTES",
    "TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES",
    "TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY",
    "TerminusR2EActAgent",
    "TerminusR2EActLLM",
    "VerifiedTmuxSession",
]
