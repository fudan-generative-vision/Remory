"""Safe corpus and Serper tool runner for GrepSeek free-running rollouts.

The released GrepSeek trajectories expose a small shell-like search tool.  A
rollout must reproduce its response contract, but model-generated commands are
untrusted input.  This module therefore accepts one pipeline of explicitly
whitelisted read-only programs and executes the stages directly, without a
shell.  Redirection, chaining, command substitution, and program-specific
write/execute flags are rejected before a subprocess is started.

The returned JSON has exactly the seven keys used by the released trajectories::

    {command, stdout, stderr, exit_code, timed_out, truncated,
     information_lines}

``corpus.jsonl`` is the logical name shown to the model.  It is rewritten to
``wiki_corpus.jsonl`` only in the executed argv; the reported command remains
unchanged.
"""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlsplit

from context_compression.grepseek_serving_contract import (
    validate_working_memory_token_budget,
)


ALLOWED_COMMANDS = frozenset(
    {
        "rg",
        "grep",
        "find",
        "sed",
        "awk",
        "head",
        "tail",
        "cat",
        "ls",
        "wc",
        "sort",
        "cut",
        "uniq",
        "tr",
    }
)
TOOL_RESPONSE_KEYS = (
    "command",
    "stdout",
    "stderr",
    "exit_code",
    "timed_out",
    "truncated",
    "information_lines",
)
CORPUS_FILENAME_LOGICAL = "corpus.jsonl"
CORPUS_FILENAME_ACTUAL = "wiki_corpus.jsonl"
REACT_MEMORY_FILENAME_LOGICAL = "memory.jsonl"
REACT_MEMORY_FILENAME_ACTUAL = "memory.jsonl"
REACT_MEMORY_CANONICAL_SHELL_PATH = "/__r2eact__/memory.jsonl"
REACT_MEMORY_QUERY_RESULT_ARCHIVE_POLICY = (
    "omit_stdout_information_lines_preserve_status_v1"
)
REACT_MEMORY_QUERY_RESULT_ARCHIVE_STUB = (
    "[memory query result omitted from recursive archive]"
)
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_WEB_SEARCH_TOP_K = 5
DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = 15.0
DEFAULT_WEB_SEARCH_MAX_CALLS = 2
DEFAULT_SERPER_ENDPOINT = "https://google.serper.dev/search"
WEB_SEARCH_COMMAND_PREFIX = "web_search "
DEFAULT_BROWSER_TIMEOUT_SECONDS = 15.0
DEFAULT_BROWSER_MAX_CALLS = 8
DEFAULT_BROWSER_MAX_LENGTH = 8_000
DEFAULT_SERPER_SCRAPE_ENDPOINT = "https://scrape.serper.dev"
BROWSER_IMPLEMENTATION = "lapha_v3_browser_compatible"
BROWSER_COMMAND_PREFIX = "browser "
TOOL_SCHEMA_RECOVERY_POLICY = "one_local_validation_observation_then_fail_v1"
MAX_CONSECUTIVE_RECOVERABLE_TOOL_SCHEMA_ERRORS = 2
MAX_COMMAND_CHARS = 2000
MAX_PIPELINE_STAGES = 16
CORPUS_PREFLIGHT_COMMAND = 'rg -m 1 -o -e \'"id": "[^"]+"\' corpus.jsonl'
_CORPUS_PREFLIGHT_STDOUT = re.compile(r'\A"id": "([^"\r\n]+)"\n\Z')
_MIN_CAPTURE_BYTES = 1 << 20
_MAX_STDERR_BYTES = 1 << 16
_WEB_SEARCH_QUERY_MAX_CHARS = 500
_WEB_SEARCH_TOP_K_MAX = 10
_WEB_SEARCH_TIMEOUT_MAX_SECONDS = 60.0
_WEB_SEARCH_LOCALE = re.compile(r"\A[A-Za-z]{2}\Z")
_BROWSER_URL_MAX_CHARS = 2_000
_BROWSER_MAX_LENGTH_LIMIT = 32_000
SHELL_CORPUS_WIKI18 = "wiki18"
SHELL_CORPUS_REACT_MEMORY = "react_memory"
SHELL_CORPUS_MODES = frozenset({SHELL_CORPUS_WIKI18, SHELL_CORPUS_REACT_MEMORY})
# The episode mirror contains raw archived exchanges, including exact tool
# responses, and can legitimately grow well beyond 32 MiB on long-running
# Terminal-Bench tasks.  Keep a finite guard against accidental/untrusted
# allocation, but make it large enough for those trajectories.  The limit is
# intentionally expressed once so the validation message cannot drift from
# the enforced byte ceiling.
_REACT_MEMORY_CORPUS_MAX_BYTES = 256 << 20

_REACT_MEMORY_POLICY_MARKER = "Hippocampus memory-search policy:"
_REACT_MEMORY_SHELL_SYSTEM_PROMPT = """You are a research agent with
{memory_architecture}, plus a read-only hippocampus lookup exposed as `shell`.

Hippocampus memory-search policy: `memory.jsonl` is an episodic archive of the
complete ReAct exchanges whose messages have been affected by context
compression. It preserves their raw assistant reasoning/tool calls and tool
responses even after active context retains only compressed soft memory.
This archive is not Wikipedia and does not provide new external facts; use rg/grep through
`shell` as a hippocampus-like index to recall older evidence, contradictions,
and repeated actions. Each line is JSON with stable `id`, `round`, `role`, and
`content` fields.

Useful command patterns:
- Recall a phrase: `rg -n -i -F "phrase" memory.jsonl | head -n 8`
- Recall prior tool evidence: `rg -n -F '"role":"tool"' memory.jsonl | tail -n 8`
- Find repeated plans/actions: `rg -n -i "keyword1|keyword2" memory.jsonl | head -n 12`

Allowed shell tools: rg, grep, find, sed, awk, head, tail, cat, ls, wc, sort,
cut, uniq, tr. You MAY pipe with |. You MAY NOT use redirection (>, <), command
chaining (; && ||), or command substitution ($(...), `...`).

For every step, write a short paragraph of reasoning in plain prose (2-5
sentences) describing what you learned, what is still missing, and the next
action. Then output exactly one of:

<tool_call>
{{"name": "shell", "arguments": {{"command": "your single-pipeline memory query, no newlines"}}}}
</tool_call>

OR (only when you have enough information to answer):

<answer>
the final answer (concise -- typically a short noun phrase, name, or date)
</answer>

Always reason first, then exactly one action block. Do not skip the reasoning."""

_WEB_SEARCH_POLICY_MARKER = "Live web-search policy:"
_WEB_SEARCH_TOOL_BLOCK = """<tool_call>
{{"name": "web_search", "arguments": {{"query": "your concise search query", "top_k": 5}}}}
</tool_call>"""
_WEB_SEARCH_GUIDANCE = """Live web-search policy: web_search is enabled for
current information or when the local Wikipedia corpus is insufficient.
{call_budget} Its result contains compact
snippets and source URLs. `top_k` must be an integer from 1 to 10 (default 5).
Optional arguments are `gl` (two-letter country code), `hl` (two-letter
language code), and `timeout` (1-60 seconds). Never put API keys, endpoints,
shell syntax, or URLs containing credentials in tool arguments."""
WEB_SEARCH_SYSTEM_PROMPT_APPENDIX = """{guidance}

{web_tool_block}

The existing `shell` tool remains available for corpus searches. Emit exactly
one enabled tool call or one final answer after each reasoning paragraph."""

_BROWSER_POLICY_MARKER = "LaPha browser policy:"
_BROWSER_TOOL_BLOCK = """<tool_call>
{{"name": "browser", "arguments": {{"url": "https://source.example/page"}}}}
</tool_call>"""
_BROWSER_GUIDANCE = """LaPha browser policy: browser opens a public HTTP(S)
URL through the LaPha Serper Scrape adapter and returns cleaned source-page
text. Use it to verify evidence from URLs returned by web_search. A rollout may
call it at most {max_calls} times. Optional arguments are `timeout` (1-60
seconds) and `max_length` (1-{max_length_limit} characters). Never submit API
keys, credential-bearing URLs, private-network URLs, or local file paths."""
_WEB_BROWSER_SEQUENCE_GUIDANCE = """When web_search is also enabled, treat its
snippets as discovery rather than final evidence. After the first successful
web_search result containing a public URL, the next tool call must be browser
on one returned URL before answering. Never repeat a tool call after its
call-budget-exceeded error; use the other enabled tool or answer from evidence
already collected."""
_BROWSER_ONLY_SYSTEM_PROMPT = """You are a research agent that opens public
source URLs to answer questions. The local shell corpus and live web search are
disabled for this rollout, so browser can only open URLs present in the
question or prior context.

{guidance}

For every step, write a short paragraph of reasoning in plain prose (2-5
sentences) describing what you learned and what is still missing. Then output
exactly one of:

{browser_tool_block}

OR (only when you have enough information to answer):

<answer>
the final answer (concise -- typically a short noun phrase, name, or date)
</answer>

Always reason first, then exactly one action block. Do not skip the reasoning."""

_GREPSEEK_ACTION_START = "Then output exactly one of:\n\n"
_GREPSEEK_ANSWER_START = "\n\nOR (only when you have enough information to answer):"
_WEB_SEARCH_ONLY_SYSTEM_PROMPT = """You are a research agent that uses live web
search to answer questions. Search when evidence is needed, especially for
current information, and base the answer on the returned snippets and sources.
The `shell` corpus tool is disabled for this rollout.

{guidance}

For every step, write a short paragraph of reasoning in plain prose (2-5
sentences) describing what you learned and what is still missing. Then output
exactly one of:

{web_tool_block}

OR (only when you have enough information to answer):

<answer>
the final answer (concise -- typically a short noun phrase, name, or date)
</answer>

Always reason first, then exactly one action block. Do not skip the reasoning."""


class ToolError(ValueError):
    """Raised when an untrusted command violates the tool policy."""


@dataclass(frozen=True)
class ToolResult:
    """One GrepSeek tool response with the released seven-field contract."""

    command: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool
    information_lines: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            "information_lines": list(self.information_lines),
        }

    def to_payload(self) -> str:
        """Serialize the exact tool-role JSON payload used in trajectories."""

        return json.dumps(self.to_dict(), ensure_ascii=False)


def canonical_web_search_command(arguments: Mapping[str, Any]) -> str:
    """Validate a public tool payload and return its private runner command.

    The serialized command is an internal bridge for the existing GrepSeek
    rollout ABI.  In particular, endpoint, API key, call budget, and output
    limits are deliberately absent from the model-controlled schema.
    """

    if not isinstance(arguments, Mapping):
        raise ToolError("web_search arguments must be an object")
    allowed = {"query", "top_k", "timeout", "gl", "hl"}
    unknown = sorted(str(key) for key in set(arguments) - allowed)
    if unknown:
        raise ToolError(
            "web_search contains unsupported arguments: " + ", ".join(unknown)
        )
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("web_search query must be a nonempty string")
    query = query.strip()
    if len(query) > _WEB_SEARCH_QUERY_MAX_CHARS:
        raise ToolError(
            f"web_search query exceeds {_WEB_SEARCH_QUERY_MAX_CHARS} characters"
        )

    normalized: dict[str, Any] = {"query": query}
    top_k = arguments.get("top_k")
    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ToolError("web_search top_k must be an integer")
        if not 1 <= top_k <= _WEB_SEARCH_TOP_K_MAX:
            raise ToolError(
                f"web_search top_k must be between 1 and {_WEB_SEARCH_TOP_K_MAX}"
            )
        normalized["top_k"] = int(top_k)

    timeout = arguments.get("timeout")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ToolError("web_search timeout must be numeric")
        timeout = float(timeout)
        if (
            not math.isfinite(timeout)
            or not 1 <= timeout <= _WEB_SEARCH_TIMEOUT_MAX_SECONDS
        ):
            raise ToolError(
                "web_search timeout must be between 1 and "
                f"{int(_WEB_SEARCH_TIMEOUT_MAX_SECONDS)} seconds"
            )
        normalized["timeout"] = timeout

    for name in ("gl", "hl"):
        value = arguments.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or _WEB_SEARCH_LOCALE.fullmatch(value) is None:
            raise ToolError(f"web_search {name} must be a two-letter code")
        normalized[name] = value.lower()
    return WEB_SEARCH_COMMAND_PREFIX + json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def parse_web_search_command(command: str) -> Optional[dict[str, Any]]:
    """Return validated arguments for an internal web-search command."""

    if not isinstance(command, str) or not command.startswith(
        WEB_SEARCH_COMMAND_PREFIX
    ):
        return None
    if "\x00" in command or "\n" in command or "\r" in command:
        raise ToolError("web_search command contains forbidden control characters")
    if len(command) > MAX_COMMAND_CHARS:
        raise ToolError(f"command exceeds {MAX_COMMAND_CHARS} characters")
    raw = command[len(WEB_SEARCH_COMMAND_PREFIX) :]
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError("web_search command payload is not valid JSON") from exc
    canonical = canonical_web_search_command(arguments)
    if canonical != command:
        raise ToolError("web_search command is not in canonical form")
    return dict(arguments)


def append_web_search_system_prompt(
    system_prompt: str, *, max_calls: Optional[int], shell_enabled: bool = True
) -> str:
    """Extend a released GrepSeek prompt for the selected live tool set."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be nonempty")
    if max_calls is not None and (
        isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls <= 0
    ):
        raise ValueError("max_calls must be a positive integer or None")
    if _WEB_SEARCH_POLICY_MARKER in system_prompt:
        return system_prompt

    call_budget = (
        "There is no per-rollout call-count limit; stop searching once enough "
        "evidence is available."
        if max_calls is None
        else f"A rollout may call it at most {max_calls} times."
    )
    guidance = _WEB_SEARCH_GUIDANCE.format(call_budget=call_budget)
    if _REACT_MEMORY_POLICY_MARKER in system_prompt:
        guidance = guidance.replace(
            "the local Wikipedia corpus is insufficient",
            "the episode-memory shell corpus cannot supply new external evidence",
        )
    web_tool_block = _WEB_SEARCH_TOOL_BLOCK.format()
    if not shell_enabled:
        return _WEB_SEARCH_ONLY_SYSTEM_PROMPT.format(
            guidance=guidance,
            web_tool_block=web_tool_block,
        )

    # The released GrepSeek prompt's original action list says shell-or-answer.
    # Insert web_search into that exact list so the final prompt has one
    # coherent action contract instead of two contradictory contracts.
    if _GREPSEEK_ACTION_START in system_prompt:
        prefix, actions = system_prompt.split(_GREPSEEK_ACTION_START, 1)
        if _GREPSEEK_ANSWER_START not in actions:
            raise ValueError("system prompt has an unrecognized answer action contract")
        shell_action, answer_action = actions.split(_GREPSEEK_ANSWER_START, 1)
        return (
            prefix.rstrip()
            + "\n\n"
            + guidance
            + "\n\n"
            + _GREPSEEK_ACTION_START
            + shell_action.rstrip()
            + "\n\nOR:\n\n"
            + web_tool_block
            + _GREPSEEK_ANSWER_START
            + answer_action
        )

    # Generic callers may not use the released GrepSeek shell prompt. In that
    # case the appended policy is still self-contained and non-exclusive.
    appendix = WEB_SEARCH_SYSTEM_PROMPT_APPENDIX.format(
        guidance=guidance,
        web_tool_block=web_tool_block,
    )
    return system_prompt.rstrip() + "\n\n" + appendix


def canonical_browser_command(arguments: Mapping[str, Any]) -> str:
    """Validate LaPha's public browser schema and build its private command."""

    if not isinstance(arguments, Mapping):
        raise ToolError("browser arguments must be an object")
    allowed = {"url", "timeout", "max_length"}
    unknown = sorted(str(key) for key in set(arguments) - allowed)
    if unknown:
        raise ToolError("browser contains unsupported arguments: " + ", ".join(unknown))

    raw_url = arguments.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ToolError("browser url must be a nonempty string")
    url = raw_url.strip()
    if len(url) > _BROWSER_URL_MAX_CHARS:
        raise ToolError(f"browser url exceeds {_BROWSER_URL_MAX_CHARS} characters")
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ToolError("browser url contains forbidden control characters")
    supplied = urlsplit(url)
    if supplied.scheme and supplied.scheme not in {"http", "https"}:
        raise ToolError("browser url must use HTTP or HTTPS")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        raise ToolError("browser url must be a public HTTP(S) URL without credentials")
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname or hostname == "localhost" or hostname.endswith(".local"):
        raise ToolError("browser url must not target a local host")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise ToolError("browser url must not target a private or reserved address")

    normalized: dict[str, Any] = {"url": url}
    timeout = arguments.get("timeout")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ToolError("browser timeout must be numeric")
        timeout = float(timeout)
        if (
            not math.isfinite(timeout)
            or not 1 <= timeout <= _WEB_SEARCH_TIMEOUT_MAX_SECONDS
        ):
            raise ToolError(
                "browser timeout must be between 1 and "
                f"{int(_WEB_SEARCH_TIMEOUT_MAX_SECONDS)} seconds"
            )
        normalized["timeout"] = timeout

    max_length = arguments.get("max_length")
    if max_length is not None:
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise ToolError("browser max_length must be an integer")
        if not 1 <= max_length <= _BROWSER_MAX_LENGTH_LIMIT:
            raise ToolError(
                "browser max_length must be between 1 and "
                f"{_BROWSER_MAX_LENGTH_LIMIT} characters"
            )
        normalized["max_length"] = int(max_length)

    return BROWSER_COMMAND_PREFIX + json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def parse_browser_command(command: str) -> Optional[dict[str, Any]]:
    """Return validated arguments for an internal LaPha-browser command."""

    if not isinstance(command, str) or not command.startswith(BROWSER_COMMAND_PREFIX):
        return None
    if "\x00" in command or "\n" in command or "\r" in command:
        raise ToolError("browser command contains forbidden control characters")
    if len(command) > MAX_COMMAND_CHARS:
        raise ToolError(f"command exceeds {MAX_COMMAND_CHARS} characters")
    raw = command[len(BROWSER_COMMAND_PREFIX) :]
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError("browser command payload is not valid JSON") from exc
    canonical = canonical_browser_command(arguments)
    if canonical != command:
        raise ToolError("browser command is not in canonical form")
    return dict(arguments)


def append_browser_system_prompt(
    system_prompt: str,
    *,
    max_calls: int,
    other_tool_enabled: bool = True,
) -> str:
    """Add LaPha browser instructions to a GrepSeek live-tool prompt."""

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be nonempty")
    if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls <= 0:
        raise ValueError("max_calls must be a positive integer")
    if _BROWSER_POLICY_MARKER in system_prompt:
        return system_prompt
    guidance = _BROWSER_GUIDANCE.format(
        max_calls=max_calls,
        max_length_limit=_BROWSER_MAX_LENGTH_LIMIT,
    )
    if other_tool_enabled:
        guidance += "\n\n" + _WEB_BROWSER_SEQUENCE_GUIDANCE
    browser_tool_block = _BROWSER_TOOL_BLOCK.format()
    if not other_tool_enabled:
        return _BROWSER_ONLY_SYSTEM_PROMPT.format(
            guidance=guidance,
            browser_tool_block=browser_tool_block,
        )
    if _GREPSEEK_ACTION_START in system_prompt:
        prefix, actions = system_prompt.split(_GREPSEEK_ACTION_START, 1)
        if _GREPSEEK_ANSWER_START not in actions:
            raise ValueError("system prompt has an unrecognized answer action contract")
        existing_actions, answer_action = actions.split(_GREPSEEK_ANSWER_START, 1)
        return (
            prefix.rstrip()
            + "\n\n"
            + guidance
            + "\n\n"
            + _GREPSEEK_ACTION_START
            + existing_actions.rstrip()
            + "\n\nOR:\n\n"
            + browser_tool_block
            + _GREPSEEK_ANSWER_START
            + answer_action
        )
    return system_prompt.rstrip() + "\n\n" + guidance + "\n\n" + browser_tool_block


def react_memory_shell_system_prompt(
    *,
    working_memory_turns: Optional[int] = None,
    working_memory_tokens: Optional[int] = None,
    compression_trigger_tokens: Optional[int] = None,
) -> str:
    """Return the coherent shell prompt for the episode-memory corpus.

    The released prompt describes Wiki-18 and therefore cannot be patched with
    a short appendix without leaving contradictory instructions.  This prompt
    preserves the exact GrepSeek action/answer envelope while replacing the
    shell's data-source contract in full.
    """

    token_mode = (
        working_memory_tokens is not None or compression_trigger_tokens is not None
    )
    if token_mode:
        if working_memory_turns is not None:
            raise ValueError(
                "working-memory turns and token budgets are mutually exclusive"
            )
        if working_memory_tokens is None or compression_trigger_tokens is None:
            raise ValueError(
                "working_memory_tokens and compression_trigger_tokens must be "
                "provided together"
            )
        working_memory_tokens, compression_trigger_tokens = (
            validate_working_memory_token_budget(
                working_memory_tokens, compression_trigger_tokens
            )
        )
        architecture = (
            "three memory layers: long-term semantic memory represented by "
            "compressed soft tokens plus a protected right-side raw working "
            "memory of the newest complete ReAct exchanges, budgeted at "
            f"{working_memory_tokens} tokens. Older raw exchanges form an "
            "unprotected left staging area. Only semantic memory M plus that "
            "left staging area counts toward the compression trigger of "
            f"{compression_trigger_tokens} tokens; when triggered, the left "
            "staging area is recursively folded into M while the protected "
            "right-side working memory remains raw"
        )
    elif working_memory_turns is None:
        architecture = "the complete discrete ReAct history kept raw"
    else:
        if (
            isinstance(working_memory_turns, bool)
            or not isinstance(working_memory_turns, int)
            or not 1 <= working_memory_turns <= 32
        ):
            raise ValueError("working_memory_turns must be between 1 and 32")
        architecture = (
            "three memory layers: long-term semantic memory represented by "
            "compressed soft tokens and a raw working-memory window containing "
            f"the latest {working_memory_turns} completed ReAct exchange(s)"
        )
    return _REACT_MEMORY_SHELL_SYSTEM_PROMPT.format(memory_architecture=architecture)


def _split_pipeline(command: str) -> list[str]:
    """Split only on unquoted pipes and reject shell control syntax."""

    stages: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if quote == "'":
            if char == "'":
                quote = None
            index += 1
            continue
        if char == "\\":
            escaped = True
            index += 1
            continue
        if quote == '"':
            if char == '"':
                quote = None
            elif char == "`" or command.startswith("$(", index):
                raise ToolError("command substitution is not allowed")
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "`" or command.startswith("$(", index):
            raise ToolError("command substitution is not allowed")
        if command.startswith("&&", index) or command.startswith("||", index):
            raise ToolError("command chaining is not allowed")
        if char == "|":
            segment = command[start:index].strip()
            if not segment:
                raise ToolError("empty pipeline segment")
            stages.append(segment)
            start = index + 1
        elif char in {";", "&", ">", "<"}:
            raise ToolError(
                f"shell control/redirection character {char!r} is not allowed"
            )
        elif char in {"(", ")"}:
            raise ToolError("unquoted shell grouping is not allowed")
        index += 1
    if quote is not None:
        raise ToolError("unterminated quoted string")
    if escaped:
        raise ToolError("trailing escape is not allowed")
    final = command[start:].strip()
    if not final:
        raise ToolError("empty pipeline segment")
    stages.append(final)
    return stages


def _long_option(token: str) -> str:
    return token.split("=", 1)[0] if token.startswith("--") else ""


def _short_option_is_present(
    args: Sequence[str],
    target: str,
    *,
    value_options: frozenset[str],
) -> bool:
    """Recognize a short option even when it is inside an option cluster.

    Once an option that consumes a value is reached, the rest of that token is
    its attached value rather than more option letters.  This distinction keeps
    a harmless pattern such as ``-efoo`` from being mistaken for ``-f FILE``,
    while still catching external-file forms such as ``-vf/etc/passwd``.
    """

    if len(target) != 1:
        raise ValueError("target must be one short-option letter")
    for token in args:
        if token == "--":
            break
        if not token.startswith("-") or token.startswith("--") or token == "-":
            continue
        for flag in token[1:]:
            if flag == target:
                return True
            if flag in value_options:
                break
    return False


def _validate_rg(args: Sequence[str]) -> None:
    for token in args:
        option = _long_option(token)
        if option == "--files":
            raise ToolError("rg --files is not a corpus-content query")
        if option in {"--pre", "--hostname-bin"}:
            raise ToolError(f"rg option {option!r} can execute another program")
        if option in {"--file", "--ignore-file"}:
            raise ToolError(f"rg option {option!r} reads outside the corpus input")
    if _short_option_is_present(
        args,
        "f",
        value_options=frozenset("ABCDEdefgjMmrtT"),
    ):
        raise ToolError("rg -f reads patterns from another file")


def _validate_grep(args: Sequence[str]) -> None:
    for token in args:
        option = _long_option(token)
        external_file_option = _matches_long_prefix(
            option, "--file", len("--f")
        ) or _matches_long_prefix(
            option,
            "--exclude-from",
            len("--exclude-f"),
        )
        if external_file_option:
            raise ToolError(f"grep option {option!r} reads outside the corpus input")
    if _short_option_is_present(
        args,
        "f",
        value_options=frozenset("ABCDdefm"),
    ):
        raise ToolError("grep -f reads patterns from another file")


def _validate_find(args: Sequence[str]) -> None:
    write_or_exec_actions = {
        "-delete",
        "-exec",
        "-execdir",
        "-ok",
        "-okdir",
        "-fprintf",
        "-fprint",
        "-fprint0",
        "-fls",
        "-files0-from",
    }
    for token in args:
        reads_reference_path = token in {
            "-anewer",
            "-cnewer",
            "-samefile",
        } or token.startswith("-newer")
        if reads_reference_path:
            raise ToolError(f"find action {token!r} reads outside the corpus input")
        if token in write_or_exec_actions:
            raise ToolError(f"find action {token!r} is not read-only")


def _sed_substitution_side_effect(script: str) -> bool:
    """Recognize ``s///e`` and ``s///w file`` forms.

    GNU sed sandbox mode is also injected at execution time.  This parser is a
    validation-time diagnostic and defense in depth, not the sole boundary.
    """

    for start, char in enumerate(script):
        if char != "s" or start + 1 >= len(script):
            continue
        delimiter = script[start + 1]
        if delimiter.isspace() or delimiter.isalnum() or delimiter == "\\":
            continue
        cursor = start + 2
        complete = True
        for _ in range(2):
            escaped = False
            while cursor < len(script):
                current = script[cursor]
                cursor += 1
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == delimiter:
                    break
            else:
                complete = False
                break
        if not complete:
            continue
        flags = script[cursor:]
        flags = re.split(r"[;\n}]", flags, maxsplit=1)[0]
        if re.search(r"(?:^|[0-9gIpM]*)e(?:[0-9gIpM]*|\s|$)", flags):
            return True
        if re.search(r"(?:^|[0-9gIpMe]*)w(?:\s|$)", flags):
            return True
    return False


_SED_DIRECT_SIDE_EFFECT = re.compile(
    r"(?:^|[;{}\n])\s*(?:(?:\d+|\$|/(?:\\.|[^/])*/)(?:\s*,\s*"
    r"(?:\d+|\$|/(?:\\.|[^/])*/))?\s*)?"
    r"[!\s]*(?:e|r|R|w|W)(?:\s|$)"
)


def _validate_sed(args: Sequence[str]) -> None:
    scripts: list[str] = []
    explicit_expression = False
    options = True
    index = 0
    while index < len(args):
        token = args[index]
        if options and token == "--":
            options = False
            index += 1
            continue
        if options and token.startswith("--"):
            option, separator, value = token.partition("=")
            if _matches_long_prefix(option, "--in-place", len("--in")):
                raise ToolError("sed --in-place is not read-only")
            if _matches_long_prefix(option, "--file", len("--fil")):
                raise ToolError("sed program files are not allowed")
            if option == "--expression":
                explicit_expression = True
                if not separator:
                    index += 1
                    if index >= len(args):
                        raise ToolError("sed --expression requires a script")
                    value = args[index]
                scripts.append(value)
            elif option == "--line-length" and not separator:
                index += 1
                if index >= len(args):
                    raise ToolError("sed --line-length requires a value")
            index += 1
            continue
        if options and token.startswith("-") and token != "-":
            cluster = token[1:]
            position = 0
            while position < len(cluster):
                flag = cluster[position]
                if flag == "i":
                    raise ToolError("sed -i is not read-only")
                if flag == "f":
                    raise ToolError("sed program files are not allowed")
                if flag == "e":
                    explicit_expression = True
                    attached = cluster[position + 1 :]
                    if attached:
                        scripts.append(attached)
                    else:
                        index += 1
                        if index >= len(args):
                            raise ToolError("sed -e requires a script")
                        scripts.append(args[index])
                    break
                if flag == "l":
                    if not cluster[position + 1 :]:
                        index += 1
                        if index >= len(args):
                            raise ToolError("sed -l requires a value")
                    break
                position += 1
            index += 1
            continue
        if not explicit_expression and not scripts:
            scripts.append(token)
        index += 1
    for script in scripts:
        if _SED_DIRECT_SIDE_EFFECT.search(script) or _sed_substitution_side_effect(
            script
        ):
            raise ToolError("sed script contains external I/O or execution")


def _awk_programs(args: Sequence[str]) -> list[str]:
    programs: list[str] = []
    index = 0
    options = True
    while index < len(args):
        token = args[index]
        if options and token == "--":
            options = False
            index += 1
            continue
        option = _long_option(token)
        dangerous_short = any(
            token == flag or token.startswith(flag) for flag in ("-f", "-E", "-i", "-l")
        )
        dangerous_long = any(
            _matches_long_prefix(option, name, minimum)
            for name, minimum in (
                ("--file", len("--fil")),
                ("--exec", len("--ex")),
                ("--include", len("--inc")),
                ("--load", len("--lo")),
            )
        )
        if options and (dangerous_short or dangerous_long):
            raise ToolError(f"awk option {token!r} can load or execute external code")
        if options and (token == "-W" or token.startswith("-W")):
            value = token[2:] if token != "-W" else ""
            if not value:
                index += 1
                if index >= len(args):
                    raise ToolError("awk -W requires a value")
                value = args[index]
            if value.lstrip("=").lower().startswith("exec"):
                raise ToolError("awk -W exec is not allowed")
            index += 1
            continue
        if options and token in {"-F", "-v", "--assign"}:
            index += 2
            if index > len(args):
                raise ToolError(f"awk option {token!r} requires a value")
            continue
        if options and (
            (token.startswith("-F") and token != "-F")
            or (token.startswith("-v") and token != "-v")
            or option == "--assign"
        ):
            index += 1
            continue
        if options and (token == "-e" or option == "--source"):
            if "=" in token:
                programs.append(token.split("=", 1)[1])
            elif token != "-e" and token.startswith("--source="):
                programs.append(token.split("=", 1)[1])
            else:
                index += 1
                if index >= len(args):
                    raise ToolError(f"awk option {token!r} requires a program")
                programs.append(args[index])
            index += 1
            continue
        if options and token.startswith("-") and token != "-":
            index += 1
            continue
        if not programs:
            programs.append(token)
        break
    return programs


_AWK_REDIRECTION = re.compile(r"\b(?:print|printf)\b[^;\n}]*(?:>>|>)")


def _validate_awk(args: Sequence[str]) -> None:
    for program in _awk_programs(args):
        if re.search(r"\bsystem\s*\(", program):
            raise ToolError("awk system() is not allowed")
        if re.search(r"\b(?:getline|ARGV)\b", program):
            raise ToolError("awk external input access is not allowed")
        if "|" in program:
            raise ToolError("awk pipe-to-command syntax is not allowed")
        if _AWK_REDIRECTION.search(program):
            raise ToolError("awk output redirection is not allowed")
        if re.search(r"@\s*(?:include|load)\b", program):
            raise ToolError("awk external program loading is not allowed")


def _matches_long_prefix(option: str, dangerous: str, minimum: int) -> bool:
    return bool(option) and len(option) >= minimum and dangerous.startswith(option)


def _validate_sort(args: Sequence[str]) -> None:
    options = True
    for token in args:
        if options and token == "--":
            options = False
            continue
        if not options:
            continue
        option = _long_option(token)
        if any(
            _matches_long_prefix(option, dangerous, minimum)
            for dangerous, minimum in (
                ("--output", len("--out")),
                ("--temporary-directory", len("--temp")),
                ("--compress-program", len("--compress")),
                ("--random-source", len("--random")),
            )
        ):
            raise ToolError(f"sort option {option!r} can write or execute a program")
        if token.startswith("-") and not token.startswith("--") and token != "-":
            cluster = token[1:]
            if "o" in cluster or "T" in cluster:
                raise ToolError(f"sort option {token!r} is not read-only")


def _validate_uniq(args: Sequence[str]) -> None:
    """Reject uniq's optional OUTPUT positional operand."""

    operands: list[str] = []
    options = True
    index = 0
    while index < len(args):
        token = args[index]
        if options and token == "--":
            options = False
            index += 1
            continue
        if options and token.startswith("--"):
            option = _long_option(token)
            if (
                option in {"--skip-fields", "--skip-chars", "--check-chars"}
                and "=" not in token
            ):
                index += 1
            index += 1
            continue
        if options and token.startswith("-") and token != "-":
            cluster = token[1:]
            for position, flag in enumerate(cluster):
                if flag in {"f", "s", "w"}:
                    if not cluster[position + 1 :]:
                        index += 1
                    break
            index += 1
            continue
        operands.append(token)
        index += 1
    if len(operands) > 1:
        raise ToolError("uniq OUTPUT operand is not allowed")


def _validate_wc(args: Sequence[str]) -> None:
    if any(_long_option(token) == "--files0-from" for token in args):
        raise ToolError("wc --files0-from reads additional paths")


_COMMAND_VALIDATORS = {
    "rg": _validate_rg,
    "grep": _validate_grep,
    "find": _validate_find,
    "sed": _validate_sed,
    "awk": _validate_awk,
    "sort": _validate_sort,
    "uniq": _validate_uniq,
    "wc": _validate_wc,
}


_POSITIONAL_OPTION_VALUES: dict[str, tuple[set[str], set[str]]] = {
    "rg": (
        set("ABCDEdefgjMmrtT"),
        {
            "--after-context",
            "--before-context",
            "--colors",
            "--context",
            "--context-separator",
            "--dfa-size-limit",
            "--encoding",
            "--engine",
            "--field-context-separator",
            "--field-match-separator",
            "--glob",
            "--hostname-bin",
            "--hyperlink-format",
            "--iglob",
            "--ignore-file",
            "--max-columns",
            "--max-count",
            "--max-depth",
            "--path-separator",
            "--pre",
            "--pre-glob",
            "--regex-size-limit",
            "--regexp",
            "--replace",
            "--sort",
            "--sortr",
            "--threads",
            "--type",
            "--type-not",
        },
    ),
    "grep": (
        set("ABCefm"),
        {
            "--after-context",
            "--before-context",
            "--binary-files",
            "--context",
            "--devices",
            "--directories",
            "--exclude",
            "--exclude-dir",
            "--exclude-from",
            "--file",
            "--group-separator",
            "--include",
            "--label",
            "--max-count",
            "--regexp",
        },
    ),
    "head": (set("cn"), {"--bytes", "--lines"}),
    "tail": (
        set("cns"),
        {"--bytes", "--lines", "--max-unchanged-stats", "--pid", "--sleep-interval"},
    ),
    "sed": (set("efl"), {"--expression", "--file", "--line-length"}),
    "awk": (set("Ffv"), {"--assign", "--file", "--source"}),
    "sort": (
        set("koST"),
        {
            "--batch-size",
            "--buffer-size",
            "--compress-program",
            "--field-separator",
            "--key",
            "--output",
            "--parallel",
            "--random-source",
            "--temporary-directory",
        },
    ),
    "cut": (
        set("bcdf"),
        {"--bytes", "--characters", "--delimiter", "--fields", "--output-delimiter"},
    ),
    "uniq": (
        set("fsw"),
        {"--check-chars", "--skip-chars", "--skip-fields"},
    ),
    "ls": (
        set("ITw"),
        {
            "--block-size",
            "--color",
            "--format",
            "--hide",
            "--hyperlink",
            "--ignore",
            "--indicator-style",
            "--quoting-style",
            "--sort",
            "--tabsize",
            "--time",
            "--time-style",
            "--width",
        },
    ),
}


def _positionals(program: str, args: Sequence[str]) -> list[str]:
    """Conservatively identify operands that a utility may treat as files."""

    short_values, long_values = _POSITIONAL_OPTION_VALUES.get(program, (set(), set()))
    result: list[str] = []
    options = True
    index = 0
    while index < len(args):
        token = args[index]
        if options and token == "--":
            options = False
            index += 1
            continue
        if options and token.startswith("--"):
            option = _long_option(token)
            if option in long_values and "=" not in token:
                index += 1
                if index >= len(args):
                    raise ToolError(f"option {option!r} requires a value")
            index += 1
            continue
        if options and token.startswith("-") and token != "-":
            cluster = token[1:]
            for position, flag in enumerate(cluster):
                if flag in short_values:
                    if not cluster[position + 1 :]:
                        index += 1
                        if index >= len(args):
                            raise ToolError(f"option -{flag} requires a value")
                    break
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def _has_explicit_pattern(args: Sequence[str]) -> bool:
    for token in args:
        if _long_option(token) == "--regexp":
            return True
        if token == "-e" or (token.startswith("-e") and token != "-"):
            return True
    return False


def _validate_corpus_pipeline(stages: Sequence[Sequence[str]]) -> None:
    """Force a single first-stage corpus source and stdin-only later stages."""

    occurrences = sum(
        token == CORPUS_FILENAME_LOGICAL for stage in stages for token in stage[1:]
    )
    if occurrences != 1:
        raise ToolError(f"pipeline must name {CORPUS_FILENAME_LOGICAL!r} exactly once")
    for stage_index, stage in enumerate(stages):
        program, args = stage[0], stage[1:]
        first = stage_index == 0
        if program in {"rg", "grep"}:
            operands = _positionals(program, args)
            expected = (
                [CORPUS_FILENAME_LOGICAL] if _has_explicit_pattern(args) else None
            )
            if first:
                valid = (
                    operands == expected
                    if expected is not None
                    else len(operands) == 2 and operands[-1] == CORPUS_FILENAME_LOGICAL
                )
            else:
                valid = operands == ([] if expected is not None else operands[:1]) and (
                    expected is not None or len(operands) == 1
                )
            if not valid:
                raise ToolError(
                    f"{program} may read only the first-stage logical corpus or pipeline stdin"
                )
            continue
        if program in {"cat", "head", "tail", "wc", "sort", "cut", "uniq"}:
            operands = _positionals(program, args)
            expected = [CORPUS_FILENAME_LOGICAL] if first else []
            if operands != expected:
                raise ToolError(
                    f"{program} may read only the first-stage logical corpus or pipeline stdin"
                )
            continue
        if program in {"sed", "awk"}:
            operands = _positionals(program, args)
            explicit_program = any(
                token == "-e"
                or token.startswith("-e")
                or _long_option(token) in {"--expression", "--source"}
                for token in args
            )
            if explicit_program:
                expected = [CORPUS_FILENAME_LOGICAL] if first else []
                valid = operands == expected
            else:
                valid = len(operands) == (2 if first else 1) and (
                    not first or operands[-1] == CORPUS_FILENAME_LOGICAL
                )
            if not valid:
                raise ToolError(
                    f"{program} may read only the first-stage logical corpus or pipeline stdin"
                )
            continue
        if program == "tr":
            if first or not 1 <= len(_positionals(program, args)) <= 2:
                raise ToolError("tr may only transform pipeline stdin")
            continue
        if program == "ls":
            if not first or _positionals(program, args) != [CORPUS_FILENAME_LOGICAL]:
                raise ToolError(
                    "ls may inspect only the logical corpus in the first stage"
                )
            continue
        if program == "find":
            paths: list[str] = []
            for token in args:
                if token.startswith("-") or token in {"!", "(", ")"}:
                    break
                paths.append(token)
            if not first or paths != [CORPUS_FILENAME_LOGICAL]:
                raise ToolError(
                    "find may inspect only the logical corpus in the first stage"
                )
            continue
        raise AssertionError(f"missing corpus I/O policy for {program!r}")


def _parse_and_validate_pipeline(command: str) -> list[list[str]]:
    if not isinstance(command, str):
        raise ToolError("command must be a string")
    if not command.strip():
        raise ToolError("empty command")
    if "\x00" in command:
        raise ToolError("NUL bytes are not allowed")
    if "\n" in command or "\r" in command:
        raise ToolError("newlines are not allowed")
    if len(command) > MAX_COMMAND_CHARS:
        raise ToolError(f"command exceeds {MAX_COMMAND_CHARS} characters")
    raw_stages = _split_pipeline(command)
    if len(raw_stages) > MAX_PIPELINE_STAGES:
        raise ToolError(f"pipeline exceeds {MAX_PIPELINE_STAGES} stages")

    stages: list[list[str]] = []
    for raw_stage in raw_stages:
        try:
            argv = shlex.split(raw_stage, posix=True, comments=False)
        except ValueError as exc:
            raise ToolError(f"could not parse pipeline stage: {exc}") from exc
        if not argv:
            raise ToolError("empty pipeline segment")
        program = argv[0]
        if program not in ALLOWED_COMMANDS:
            raise ToolError(f"command {program!r} is not in the read-only whitelist")
        validator = _COMMAND_VALIDATORS.get(program)
        if validator is not None:
            validator(argv[1:])
        stages.append(argv)
    _validate_corpus_pipeline(stages)
    return stages


def validate_pipeline(command: str) -> None:
    """Raise :class:`ToolError` unless ``command`` is one safe pipeline."""

    _parse_and_validate_pipeline(command)


def _strict_tool_result_payload(value: Any) -> Optional[dict[str, Any]]:
    """Copy one seven-field tool result, or fail closed with ``None``.

    Archive redaction must never guess at an unfamiliar observation shape.  In
    particular, accepting a merely JSON-looking object here could erase an
    ordinary tool's evidence.  The live rollout may canonicalize
    ``information_lines`` to ``[]``, so this validates types rather than
    requiring those lines to duplicate stdout.
    """

    if not isinstance(value, Mapping):
        return None
    if set(value) != set(TOOL_RESPONSE_KEYS) or len(value) != len(TOOL_RESPONSE_KEYS):
        return None
    if not all(isinstance(value[key], str) for key in ("command", "stdout", "stderr")):
        return None
    if isinstance(value["exit_code"], bool) or not isinstance(value["exit_code"], int):
        return None
    if not isinstance(value["timed_out"], bool) or not isinstance(
        value["truncated"], bool
    ):
        return None
    information_lines = value["information_lines"]
    if not isinstance(information_lines, list) or not all(
        isinstance(line, str) for line in information_lines
    ):
        return None
    return {key: value[key] for key in TOOL_RESPONSE_KEYS}


def _is_strict_react_memory_query_command(command: Any) -> bool:
    """Recognize a safe shell query whose corpus operand is ReAct memory.

    ``command`` may be the legacy raw pipeline or BrowseComp's canonical tool
    JSON.  The latter is accepted only with exact ``name``/``arguments`` and
    ``arguments.command`` keys.  Detection then shares the normal shell parser
    and corpus policy: we replace exactly one parsed argv token -- either the
    public sidecar name or BrowseComp's canonical mounted path -- and accept it
    only when the resulting command is a valid single-corpus read pipeline.
    Substrings and other paths such as ``/workspace/memory.jsonl`` are not
    aliases.
    """

    if not isinstance(command, str) or not command.strip():
        return False
    shell_command = command
    try:
        canonical = json.loads(command)
    except json.JSONDecodeError:
        canonical = None
    if canonical is not None:
        if (
            not isinstance(canonical, Mapping)
            or set(canonical) != {"name", "arguments"}
            or canonical.get("name") != "shell"
            or not isinstance(canonical.get("arguments"), Mapping)
            or set(canonical["arguments"]) != {"command"}
            or not isinstance(canonical["arguments"].get("command"), str)
        ):
            return False
        shell_command = canonical["arguments"]["command"]

    if "\x00" in shell_command or "\n" in shell_command or "\r" in shell_command:
        return False
    if len(shell_command) > MAX_COMMAND_CHARS:
        return False
    try:
        raw_stages = _split_pipeline(shell_command)
        if len(raw_stages) > MAX_PIPELINE_STAGES:
            return False
        stages = [
            shlex.split(stage, posix=True, comments=False) for stage in raw_stages
        ]
    except (ToolError, ValueError):
        return False
    if any(not stage for stage in stages):
        return False

    aliases = {
        REACT_MEMORY_FILENAME_LOGICAL,
        REACT_MEMORY_CANONICAL_SHELL_PATH,
    }
    candidates = [
        (stage_index, token_index)
        for stage_index, stage in enumerate(stages)
        for token_index, token in enumerate(stage)
        if token in aliases
    ]
    valid_candidates = 0
    for stage_index, token_index in candidates:
        translated = [list(stage) for stage in stages]
        translated[stage_index][token_index] = CORPUS_FILENAME_LOGICAL
        normalized = " | ".join(shlex.join(stage) for stage in translated)
        try:
            validate_pipeline(normalized)
        except (ToolError, ValueError):
            continue
        valid_candidates += 1
        if valid_candidates > 1:
            # Ambiguous source identification is not safe to redact.
            return False
    return valid_candidates == 1


def _redacted_memory_query_result(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    redacted = {key: payload[key] for key in TOOL_RESPONSE_KEYS}
    redacted["stdout"] = REACT_MEMORY_QUERY_RESULT_ARCHIVE_STUB
    redacted["information_lines"] = [REACT_MEMORY_QUERY_RESULT_ARCHIVE_STUB]
    return redacted


def _redact_react_memory_query_tool_content(content: str) -> str:
    """Return an archive-only copy with recursive memory evidence omitted.

    The caller retains the original message for active context and trajectory.
    For a native tool batch, only valid memory-query subresults are redacted;
    all other result payloads and execution status fields survive unchanged.
    Any unfamiliar or inconsistent structure is returned byte-for-byte.
    """

    if not isinstance(content, str):
        return content
    try:
        decoded = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content
    outer = _strict_tool_result_payload(decoded)
    if outer is None:
        return content

    if _is_strict_react_memory_query_command(outer["command"]):
        redacted = _redacted_memory_query_result(outer)
    else:
        batch_command_match = re.fullmatch(
            r"tool_batch\[([1-9][0-9]*)\]", outer["command"]
        )
        if batch_command_match is None:
            return content
        batch_prefix = ""
        batch_text = outer["stdout"]
        try:
            batch = json.loads(batch_text)
        except (json.JSONDecodeError, TypeError):
            prefix, separator, suffix = batch_text.rpartition("\n\n")
            if not separator:
                return content
            try:
                batch = json.loads(suffix)
            except (json.JSONDecodeError, TypeError):
                return content
            batch_prefix = prefix + separator
        if (
            not isinstance(batch, Mapping)
            or set(batch) != {"schema", "tool_call_count", "results"}
            or batch.get("schema") != "grepseek_tool_batch_v1"
            or isinstance(batch.get("tool_call_count"), bool)
            or not isinstance(batch.get("tool_call_count"), int)
            or batch["tool_call_count"] < 2
            or not isinstance(batch.get("results"), list)
            or len(batch["results"]) != batch["tool_call_count"]
            or int(batch_command_match.group(1)) != batch["tool_call_count"]
        ):
            return content

        copied_results: list[dict[str, Any]] = []
        changed = False
        expected_item_keys = {
            "call_index",
            "executed",
            "validation_failed",
            "command",
            "result",
        }
        for item in batch["results"]:
            if (
                not isinstance(item, Mapping)
                or set(item) != expected_item_keys
                or isinstance(item.get("call_index"), bool)
                or not isinstance(item.get("call_index"), int)
                or not isinstance(item.get("executed"), bool)
                or not isinstance(item.get("validation_failed"), bool)
                or not isinstance(item.get("command"), str)
            ):
                return content
            nested = _strict_tool_result_payload(item.get("result"))
            if nested is None or nested["command"] != item["command"]:
                return content
            copied_item = dict(item)
            if _is_strict_react_memory_query_command(item["command"]):
                nested = _redacted_memory_query_result(nested)
                changed = True
            copied_item["result"] = nested
            copied_results.append(copied_item)
        if not changed:
            return content
        copied_batch = dict(batch)
        copied_batch["results"] = copied_results
        batch_stdout = batch_prefix + json.dumps(
            copied_batch,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        redacted = dict(outer)
        redacted["stdout"] = batch_stdout
        redacted["information_lines"] = [
            line for line in batch_stdout.splitlines() if line.strip()
        ]

    return json.dumps(
        redacted,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def redact_react_memory_query_results_for_archive(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a detached archive view without recursive lookup output.

    Only a tool message immediately paired with an assistant message is a
    candidate.  Every input mapping is copied, including messages that remain
    byte-for-byte unchanged, so callers cannot accidentally mutate active
    model context or the persisted raw trajectory.
    """

    archived: list[dict[str, Any]] = []
    previous_role: Optional[str] = None
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("ReAct archive messages must be mappings")
        copied = dict(message)
        role_value = message.get("role")
        role = role_value if isinstance(role_value, str) else None
        content = message.get("content")
        if role == "tool" and previous_role == "assistant" and isinstance(content, str):
            copied["content"] = _redact_react_memory_query_tool_content(content)
        archived.append(copied)
        previous_role = role
    return archived


def _react_memory_tool_record(
    content: str,
    *,
    record_round: int,
    message_index: Optional[int] = None,
    compression_overlap: Optional[bool] = None,
) -> dict[str, Any]:
    """Flatten one tool-role payload into its searchable evidence record.

    Trajectory artifacts retain the released seven-field ToolResult verbatim.
    The private shell archive needs only its evidence and execution status;
    embedding the complete ToolResult as a JSON string inside JSONL adds a
    redundant serialization layer and recursively multiplies backslashes.
    """

    try:
        tool_payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("tool memory is not valid JSON") from exc
    if not isinstance(tool_payload, Mapping):
        raise ValueError("tool memory must be a JSON object")
    stdout = tool_payload.get("stdout", "")
    stderr = tool_payload.get("stderr", "")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        raise ValueError("tool memory stdout/stderr must be strings")
    evidence = stdout
    if stderr:
        evidence = (
            f"{evidence}\n[stderr] {stderr}"
            if evidence
            else f"[stderr] {stderr}"
        )
    record: dict[str, Any] = {
        "id": f"round-{record_round:04d}-tool",
        "round": record_round,
        "role": "tool",
        "content": evidence,
        "command": str(tool_payload.get("command", "")),
        "exit_code": tool_payload.get("exit_code"),
        "timed_out": tool_payload.get("timed_out"),
        "truncated": tool_payload.get("truncated"),
    }
    if message_index is not None:
        record["message_index"] = int(message_index)
    if compression_overlap is not None:
        record["compression_overlap"] = bool(compression_overlap)
    return record


def truncate_tokens(
    text: str, max_tokens: Optional[int], tokenizer: Any
) -> tuple[str, bool]:
    """Keep the first ``max_tokens`` or disable token truncation with ``None``."""

    if max_tokens is None:
        return text, False
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    if not text:
        return text, False
    encoded = tokenizer(text, add_special_tokens=False)
    try:
        token_ids = encoded["input_ids"]
    except (KeyError, TypeError) as exc:
        raise TypeError("tokenizer output must contain input_ids") from exc
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise TypeError("tokenizer unexpectedly returned a batch")
        token_ids = token_ids[0]
    if len(token_ids) <= max_tokens:
        return text, False
    truncated = tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=False)
    if not isinstance(truncated, str):
        raise TypeError("tokenizer.decode must return a string")
    return truncated, True


def _tool_environment() -> dict[str, str]:
    # Do not expose launcher credentials/configuration through awk ENVIRON or
    # utility-specific environment hooks.  All programs are invoked by their
    # already-resolved absolute path.
    return {"LANG": "C", "LC_ALL": "C", "PATH": os.defpath}


def _execution_argv(
    stages: Sequence[Sequence[str]],
    *,
    actual_corpus_filename: str = CORPUS_FILENAME_ACTUAL,
) -> list[list[str]]:
    if (
        not isinstance(actual_corpus_filename, str)
        or not actual_corpus_filename
        or Path(actual_corpus_filename).name != actual_corpus_filename
    ):
        raise ValueError("actual_corpus_filename must be one basename")
    result: list[list[str]] = []
    for stage in stages:
        program = stage[0]
        executable = shutil.which(program)
        if executable is None:
            raise FileNotFoundError(f"whitelisted command {program!r} is not installed")
        args = [
            token.replace(CORPUS_FILENAME_LOGICAL, actual_corpus_filename)
            for token in stage[1:]
        ]
        if program == "rg":
            for flag in reversed(("--mmap", "--no-config")):
                if flag not in args:
                    args.insert(0, flag)
        elif program == "sed" and "--sandbox" not in args:
            args.insert(0, "--sandbox")
        result.append([executable, *args])
    return result


def _kill_processes(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                pass
    for process in processes:
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass


def _append_bounded(buffer: bytearray, chunk: bytes, limit: int) -> bool:
    remaining = max(0, limit - len(buffer))
    if remaining:
        buffer.extend(chunk[:remaining])
    return len(chunk) > remaining


def _collect_pipeline(
    processes: Sequence[subprocess.Popen[bytes]],
    stderr_fd: int,
    *,
    timeout: float,
    stdout_limit: int,
) -> tuple[bytes, bytes, int, bool, bool]:
    stdout_stream = processes[-1].stdout
    if stdout_stream is None:
        raise AssertionError("last pipeline stage has no stdout pipe")
    stdout_fd = stdout_stream.fileno()
    os.set_blocking(stdout_fd, False)
    os.set_blocking(stderr_fd, False)
    selector = selectors.DefaultSelector()
    selector.register(stdout_fd, selectors.EVENT_READ, "stdout")
    selector.register(stderr_fd, selectors.EVENT_READ, "stderr")
    stdout = bytearray()
    stderr = bytearray()
    stdout_clipped = False
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            events = selector.select(remaining)
            if not events:
                timed_out = True
                break
            for key, _ in events:
                try:
                    chunk = os.read(key.fd, 1 << 16)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                if key.data == "stdout":
                    stdout_clipped = (
                        _append_bounded(stdout, chunk, stdout_limit) or stdout_clipped
                    )
                else:
                    _append_bounded(stderr, chunk, _MAX_STDERR_BYTES)
        if not timed_out:
            for process in processes:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    break
    finally:
        selector.close()
    if timed_out:
        _kill_processes(processes)
        return b"", b"", -1, True, False
    return (
        bytes(stdout),
        bytes(stderr),
        int(processes[-1].returncode),
        False,
        stdout_clipped,
    )


def _execute_pipeline(
    stages: Sequence[Sequence[str]],
    *,
    cwd: Path,
    timeout: float,
    stdout_limit: int,
    actual_corpus_filename: str = CORPUS_FILENAME_ACTUAL,
) -> tuple[str, str, int, bool, bool]:
    processes: list[subprocess.Popen[bytes]] = []
    stderr_read, stderr_write = os.pipe()
    previous_stdout = None
    try:
        for argv in _execution_argv(
            stages, actual_corpus_filename=actual_corpus_filename
        ):
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=_tool_environment(),
                stdin=previous_stdout,
                stdout=subprocess.PIPE,
                stderr=stderr_write,
                start_new_session=True,
            )
            if previous_stdout is not None:
                previous_stdout.close()
            previous_stdout = process.stdout
            processes.append(process)
        os.close(stderr_write)
        stderr_write = -1
        raw_stdout, raw_stderr, exit_code, timed_out, clipped = _collect_pipeline(
            processes,
            stderr_read,
            timeout=timeout,
            stdout_limit=stdout_limit,
        )
        os.close(stderr_read)
        stderr_read = -1
        return (
            raw_stdout.decode("utf-8", errors="replace"),
            raw_stderr.decode("utf-8", errors="replace"),
            exit_code,
            timed_out,
            clipped,
        )
    except OSError as exc:
        _kill_processes(processes)
        return "", f"execution error: {exc}", 127, False, False
    finally:
        if previous_stdout is not None and not previous_stdout.closed:
            previous_stdout.close()
        if stderr_write >= 0:
            os.close(stderr_write)
        if stderr_read >= 0:
            os.close(stderr_read)


def _validation_result(command: Any, error: ToolError) -> ToolResult:
    return ToolResult(
        command=command if isinstance(command, str) else str(command),
        stdout="",
        stderr=f"validation error: {error}",
        exit_code=-2,
        timed_out=False,
        truncated=False,
        information_lines=[],
    )


def run_tool(
    command: str,
    *,
    corpus_dir: str | os.PathLike[str],
    tokenizer: Any,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
    actual_corpus_filename: str = CORPUS_FILENAME_ACTUAL,
) -> ToolResult:
    """Validate and execute one corpus pipeline, returning a seven-key result."""

    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number")
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite number")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer or None")
    if tokenizer is None:
        raise ValueError("a tokenizer is required for output truncation")
    root = Path(corpus_dir).expanduser().resolve()
    if (
        not isinstance(actual_corpus_filename, str)
        or not actual_corpus_filename
        or Path(actual_corpus_filename).name != actual_corpus_filename
    ):
        raise ValueError("actual_corpus_filename must be one basename")
    corpus_file = root / actual_corpus_filename
    if not root.is_dir() or not corpus_file.is_file():
        raise ValueError(
            f"corpus directory must contain {actual_corpus_filename!r}: {root}"
        )
    try:
        stages = _parse_and_validate_pipeline(command)
    except ToolError as exc:
        return _validation_result(command, exc)

    # ``None`` removes the tokenizer-level cap, not the process-memory safety
    # boundary. A model-generated ``cat corpus.jsonl`` must never buffer the
    # entire multi-gigabyte corpus in the gateway.
    stdout_limit = (
        _MIN_CAPTURE_BYTES
        if max_tokens is None
        else max(_MIN_CAPTURE_BYTES, max_tokens * 256)
    )
    stdout, stderr, exit_code, timed_out, capture_clipped = _execute_pipeline(
        stages,
        cwd=root,
        timeout=timeout,
        stdout_limit=stdout_limit,
        actual_corpus_filename=actual_corpus_filename,
    )
    if timed_out:
        # Match the upstream trajectory spelling (for example ``60.0s``), so
        # a timeout does not create an avoidable distribution-only byte drift.
        stdout = f"[command timed out after {timeout}s]"
        stderr = ""
    truncation_failed = False
    try:
        stdout, token_truncated = truncate_tokens(stdout, max_tokens, tokenizer)
    except Exception as exc:
        # A tokenizer adapter failure must not abort an entire multi-turn
        # rollout after the safe subprocess has already completed.  Never pass
        # the unbounded captured output through on this fallback path.
        stdout = "[tool output unavailable: token truncation failed]"
        diagnostic = f"token truncation error: {type(exc).__name__}: {exc}"
        stderr = f"{stderr.rstrip()}\n{diagnostic}".lstrip()
        token_truncated = True
        truncation_failed = True
    truncated = capture_clipped or token_truncated
    if truncated and not truncation_failed:
        if max_tokens is None:
            stdout += (
                "\n[... output truncated at the "
                f"{stdout_limit}-byte capture safety limit]"
            )
        else:
            stdout += f"\n[... output truncated at {max_tokens} tokens]"
    information_lines = [line for line in stdout.splitlines() if line.strip()]
    return ToolResult(
        command=command,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        truncated=truncated,
        information_lines=information_lines,
    )


def _compact_serper_item(
    item: Mapping[str, Any], *, rank: int, kind: str
) -> dict[str, Any]:
    title = str(
        item.get("title") or item.get("question") or item.get("name") or kind
    ).strip()
    link = str(item.get("link") or item.get("website") or item.get("url") or "").strip()
    snippet = str(
        item.get("snippet") or item.get("answer") or item.get("description") or ""
    ).strip()
    result: dict[str, Any] = {
        "rank": int(rank),
        "type": kind,
        "title": title[:500],
        "link": link[:2_000],
        "snippet": snippet[:900],
    }
    source = str(item.get("source") or item.get("displayedLink") or "").strip()
    date = str(item.get("date") or "").strip()
    if source:
        result["source"] = source[:300]
    if date:
        result["date"] = date[:100]
    return result


def _normalize_serper_results(
    payload: Mapping[str, Any], *, top_k: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    answer_box = payload.get("answerBox")
    if isinstance(answer_box, Mapping):
        parts: list[str] = []
        for key in ("answer", "snippet", "snippetHighlighted", "title"):
            value = answer_box.get(key)
            if isinstance(value, list):
                parts.extend(str(item) for item in value if str(item).strip())
            elif value not in (None, "", [], {}):
                parts.append(str(value))
        if parts:
            results.append(
                {
                    "rank": 0,
                    "type": "answerBox",
                    "title": str(answer_box.get("title") or "Answer box")[:500],
                    "link": str(answer_box.get("link") or "")[:2_000],
                    "snippet": "\n".join(parts)[:900],
                }
            )

    knowledge_graph = payload.get("knowledgeGraph")
    if isinstance(knowledge_graph, Mapping):
        results.append(
            {
                "rank": 0,
                "type": "knowledgeGraph",
                "title": str(knowledge_graph.get("title") or "Knowledge graph")[:500],
                "link": str(
                    knowledge_graph.get("website")
                    or knowledge_graph.get("descriptionLink")
                    or ""
                )[:2_000],
                "snippet": str(
                    knowledge_graph.get("description")
                    or knowledge_graph.get("type")
                    or ""
                )[:900],
            }
        )

    ordinary_count = 0
    for key, kind in (
        ("organic", "organic"),
        ("news", "news"),
        ("peopleAlsoAsk", "peopleAlsoAsk"),
    ):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            ordinary_count += 1
            rank = item.get("position")
            if isinstance(rank, bool) or not isinstance(rank, int):
                rank = ordinary_count
            results.append(_compact_serper_item(item, rank=rank, kind=kind))
            if ordinary_count >= top_k:
                break
        if ordinary_count >= top_k:
            break

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for result in results:
        identity = (
            str(result.get("link", "")),
            str(result.get("title", "")),
            str(result.get("snippet", "")),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(result)
    return deduplicated[: top_k + 2]


def _serialize_web_search_output(
    *,
    query: str,
    results: Sequence[Mapping[str, Any]],
    max_tokens: Optional[int],
    tokenizer: Any,
) -> tuple[str, bool]:
    """Return bounded, always-valid JSON by dropping whole result objects."""

    def payload_for(
        kept_results: Sequence[Mapping[str, Any]], *, truncated: bool = False
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"query": query, "results": list(kept_results)}
        if truncated:
            payload["truncated"] = True
        for result in kept_results:
            link = result.get("link")
            if isinstance(link, str) and link.startswith(("http://", "https://")):
                payload["next_action_required"] = {
                    "name": "browser",
                    "arguments": {"url": link},
                    "reason": "verify the search evidence before answering",
                }
                break
        return payload

    kept = [dict(item) for item in results]
    payload = payload_for(kept)
    stdout = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    _prefix, exceeds_limit = truncate_tokens(stdout, max_tokens, tokenizer)
    if not exceeds_limit:
        return stdout, False

    while kept:
        kept.pop()
        payload = payload_for(kept, truncated=True)
        stdout = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        _prefix, exceeds_limit = truncate_tokens(stdout, max_tokens, tokenizer)
        if not exceeds_limit:
            return stdout, True

    # A very long query can itself exceed an unusually small output budget.
    # Omit it rather than return a syntactically broken JSON prefix.
    payload = {"results": [], "truncated": True}
    stdout = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    _prefix, exceeds_limit = truncate_tokens(stdout, max_tokens, tokenizer)
    if exceeds_limit:
        raise ValueError("max_output_tokens is too small for valid web-search JSON")
    return stdout, True


def _browser_value_to_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            part for item in value if (part := _browser_value_to_text(item))
        )
    if isinstance(value, Mapping):
        for name in (
            "text",
            "content",
            "markdown",
            "description",
            "articleBody",
            "body",
        ):
            nested = _browser_value_to_text(value.get(name))
            if nested:
                return nested
        return json.dumps(dict(value), ensure_ascii=False)
    return str(value)


def _first_browser_value(payload: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        value = payload.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def _normalize_serper_scrape(
    payload: Mapping[str, Any], *, url: str, max_chars: int
) -> dict[str, Any]:
    """Mirror LaPha's compact Serper-Scrape page contract."""

    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    title = _first_browser_value(payload, ("title", "name")) or _first_browser_value(
        metadata, ("title", "og:title", "twitter:title")
    )
    final_url = _first_browser_value(
        payload, ("url", "finalUrl", "sourceUrl")
    ) or _first_browser_value(metadata, ("url", "og:url", "canonical"))
    preferred_text = _first_browser_value(
        payload, ("markdown", "markdownContent", "contentMarkdown")
    ) or _first_browser_value(
        payload, ("text", "content", "plainText", "body", "articleBody")
    )
    if preferred_text is None:
        preferred_text = payload
    text = _browser_value_to_text(preferred_text)
    page: dict[str, Any] = {
        "title": str(title or ""),
        "url": str(final_url or url),
        "text": text[:max_chars],
        "source": "serper_scrape",
    }
    if len(text) > max_chars:
        page["character_truncated"] = True
    description = _first_browser_value(
        payload, ("description", "snippet")
    ) or _first_browser_value(metadata, ("description", "og:description"))
    if description:
        page["description"] = _browser_value_to_text(description)[:600]
    jsonld = _first_browser_value(payload, ("jsonld", "jsonLd", "json_ld", "jsonLD"))
    if jsonld not in (None, "", [], {}):
        page["jsonld"] = _browser_value_to_text(jsonld)[:1_200]
    return page


def _serialize_browser_output(
    page: Mapping[str, Any], *, max_tokens: Optional[int], tokenizer: Any
) -> tuple[str, bool]:
    """Keep LaPha page output valid JSON while respecting the tool token cap."""

    payload = dict(page)

    def serialize(candidate: Mapping[str, Any]) -> tuple[str, bool]:
        text = json.dumps(dict(candidate), ensure_ascii=False, separators=(",", ":"))
        _prefix, exceeds = truncate_tokens(text, max_tokens, tokenizer)
        return text, exceeds

    stdout, exceeds_limit = serialize(payload)
    if not exceeds_limit:
        return stdout, bool(payload.get("character_truncated", False))

    payload["token_truncated"] = True
    # Preserve the page identity and evidence text ahead of transport metadata.
    # This also leaves enough room for useful text under unusually small test
    # budgets while keeping the serialized value valid JSON.
    for optional in (
        "jsonld",
        "description",
        "include_markdown",
        "content_type",
        "status_code",
        "character_truncated",
    ):
        payload.pop(optional, None)
    original_text = str(payload.get("text", ""))
    low = 0
    high = len(original_text)
    best: Optional[str] = None
    while low <= high:
        midpoint = (low + high) // 2
        payload["text"] = original_text[:midpoint]
        candidate, too_large = serialize(payload)
        if too_large:
            high = midpoint - 1
        else:
            best = candidate
            low = midpoint + 1
    if best is not None:
        return best, True

    minimal = {
        "title": str(page.get("title", "")),
        "url": str(page.get("url", "")),
        "source": "serper_scrape",
        "text": "",
        "token_truncated": True,
    }
    stdout, exceeds_limit = serialize(minimal)
    if exceeds_limit:
        raise ValueError("max_output_tokens is too small for valid browser JSON")
    return stdout, True


def _positive_env_integer(name: str, default: int, *, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def _non_negative_env_integer(name: str, default: int, *, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return value


def _positive_env_float(name: str, default: float, *, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum:g}")
    return value


class GrepSeekCorpusToolRunner:
    """Configured, reusable JSON runner for later compressor rollouts."""

    def __init__(
        self,
        *,
        corpus_dir: str | os.PathLike[str],
        tokenizer: Any,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
        web_search_api_key: Optional[str] = None,
        web_search_endpoint: Optional[str] = None,
        web_search_top_k: Optional[int] = None,
        web_search_timeout: Optional[float] = None,
        web_search_max_calls: Optional[int] = None,
        browser_endpoint: Optional[str] = None,
        browser_timeout: Optional[float] = None,
        browser_max_calls: Optional[int] = None,
        browser_max_length: Optional[int] = None,
    ) -> None:
        # Fail fast before a rollout begins.
        root = Path(corpus_dir).expanduser().resolve()
        if not root.is_dir() or not (root / CORPUS_FILENAME_ACTUAL).is_file():
            raise ValueError(
                f"corpus directory must contain {CORPUS_FILENAME_ACTUAL!r}: {root}"
            )
        if tokenizer is None:
            raise ValueError("a tokenizer is required for output truncation")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a positive finite number")
        if not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("timeout must be a positive finite number")
        if max_output_tokens is not None and (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be a positive integer or None")
        self.corpus_dir = root
        self.tokenizer = tokenizer
        self.timeout = float(timeout)
        self.max_output_tokens = max_output_tokens
        self.max_capture_bytes = (
            _MIN_CAPTURE_BYTES
            if max_output_tokens is None
            else max(_MIN_CAPTURE_BYTES, max_output_tokens * 256)
        )
        self._web_search_api_key = str(
            web_search_api_key
            if web_search_api_key is not None
            else os.getenv("SERPER_API_KEY", "")
        ).strip()
        if (
            self._web_search_api_key
            and re.fullmatch(r"[A-Za-z0-9._-]{16,256}", self._web_search_api_key)
            is None
        ):
            raise ValueError("SERPER_API_KEY has an invalid format")
        self.web_search_enabled = bool(self._web_search_api_key)
        endpoint = str(
            web_search_endpoint
            or os.getenv("SERPER_ENDPOINT")
            or DEFAULT_SERPER_ENDPOINT
        ).strip()
        parsed_endpoint = urlsplit(endpoint)
        if (
            parsed_endpoint.scheme != "https"
            or parsed_endpoint.netloc != "google.serper.dev"
            or parsed_endpoint.path != "/search"
            or parsed_endpoint.username is not None
            or parsed_endpoint.password is not None
            or parsed_endpoint.query
            or parsed_endpoint.fragment
        ):
            raise ValueError(
                "web_search_endpoint must be exactly the HTTPS Serper search API"
            )
        self.web_search_endpoint = endpoint
        if web_search_top_k is None:
            web_search_top_k = _positive_env_integer(
                "WEB_SEARCH_TOPK",
                DEFAULT_WEB_SEARCH_TOP_K,
                maximum=_WEB_SEARCH_TOP_K_MAX,
            )
        if (
            isinstance(web_search_top_k, bool)
            or not isinstance(web_search_top_k, int)
            or not 1 <= web_search_top_k <= _WEB_SEARCH_TOP_K_MAX
        ):
            raise ValueError(
                f"web_search_top_k must be between 1 and {_WEB_SEARCH_TOP_K_MAX}"
            )
        self.web_search_top_k = int(web_search_top_k)
        if web_search_timeout is None:
            web_search_timeout = _positive_env_float(
                "WEB_SEARCH_TIMEOUT",
                DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
                maximum=_WEB_SEARCH_TIMEOUT_MAX_SECONDS,
            )
        if (
            isinstance(web_search_timeout, bool)
            or not isinstance(web_search_timeout, (int, float))
            or not math.isfinite(float(web_search_timeout))
            or not 1 <= float(web_search_timeout) <= _WEB_SEARCH_TIMEOUT_MAX_SECONDS
        ):
            raise ValueError(
                "web_search_timeout must be between 1 and "
                f"{int(_WEB_SEARCH_TIMEOUT_MAX_SECONDS)} seconds"
            )
        self.web_search_timeout = float(web_search_timeout)
        if web_search_max_calls is None:
            web_search_max_calls = _non_negative_env_integer(
                "WEB_SEARCH_MAX_CALLS",
                DEFAULT_WEB_SEARCH_MAX_CALLS,
                maximum=16,
            )
        if (
            isinstance(web_search_max_calls, bool)
            or not isinstance(web_search_max_calls, int)
            or not 0 <= web_search_max_calls <= 16
        ):
            raise ValueError("web_search_max_calls must be between 0 and 16")
        self.web_search_max_calls = (
            None if web_search_max_calls == 0 else int(web_search_max_calls)
        )
        self._web_search_lock = threading.Lock()
        self._web_search_episode_id: Optional[str] = None
        self._web_search_call_count = 0

        browser_endpoint = str(
            browser_endpoint
            or os.getenv("SERPER_SCRAPE_ENDPOINT")
            or os.getenv("SERPER_SCRAPE_URL")
            or DEFAULT_SERPER_SCRAPE_ENDPOINT
        ).strip()
        parsed_browser_endpoint = urlsplit(browser_endpoint)
        if (
            parsed_browser_endpoint.scheme != "https"
            or parsed_browser_endpoint.netloc != "scrape.serper.dev"
            or parsed_browser_endpoint.path not in {"", "/"}
            or parsed_browser_endpoint.username is not None
            or parsed_browser_endpoint.password is not None
            or parsed_browser_endpoint.query
            or parsed_browser_endpoint.fragment
        ):
            raise ValueError(
                "browser_endpoint must be exactly the HTTPS Serper Scrape API"
            )
        self.browser_endpoint = DEFAULT_SERPER_SCRAPE_ENDPOINT
        if browser_timeout is None:
            browser_timeout = _positive_env_float(
                "BROWSER_TIMEOUT",
                DEFAULT_BROWSER_TIMEOUT_SECONDS,
                maximum=_WEB_SEARCH_TIMEOUT_MAX_SECONDS,
            )
        if (
            isinstance(browser_timeout, bool)
            or not isinstance(browser_timeout, (int, float))
            or not math.isfinite(float(browser_timeout))
            or not 1 <= float(browser_timeout) <= _WEB_SEARCH_TIMEOUT_MAX_SECONDS
        ):
            raise ValueError(
                "browser_timeout must be between 1 and "
                f"{int(_WEB_SEARCH_TIMEOUT_MAX_SECONDS)} seconds"
            )
        self.browser_timeout = float(browser_timeout)
        if browser_max_calls is None:
            browser_max_calls = _positive_env_integer(
                "BROWSER_MAX_CALLS", DEFAULT_BROWSER_MAX_CALLS, maximum=16
            )
        if (
            isinstance(browser_max_calls, bool)
            or not isinstance(browser_max_calls, int)
            or not 1 <= browser_max_calls <= 16
        ):
            raise ValueError("browser_max_calls must be between 1 and 16")
        self.browser_max_calls = int(browser_max_calls)
        if browser_max_length is None:
            browser_max_length = _positive_env_integer(
                "BROWSER_MAX_LENGTH",
                DEFAULT_BROWSER_MAX_LENGTH,
                maximum=_BROWSER_MAX_LENGTH_LIMIT,
            )
        if (
            isinstance(browser_max_length, bool)
            or not isinstance(browser_max_length, int)
            or not 1 <= browser_max_length <= _BROWSER_MAX_LENGTH_LIMIT
        ):
            raise ValueError(
                "browser_max_length must be between 1 and "
                f"{_BROWSER_MAX_LENGTH_LIMIT}"
            )
        self.browser_max_length = int(browser_max_length)
        self.browser_enabled = bool(self._web_search_api_key)
        self.browser_implementation = BROWSER_IMPLEMENTATION
        self._browser_lock = threading.Lock()
        self._browser_episode_id: Optional[str] = None
        self._browser_call_count = 0

    def begin_episode(self, episode_id: str) -> None:
        """Reset the paid-call budget for one serialized rollout."""

        with self._web_search_lock:
            self._web_search_episode_id = str(episode_id)
            self._web_search_call_count = 0
        with self._browser_lock:
            self._browser_episode_id = str(episode_id)
            self._browser_call_count = 0

    def end_episode(self, episode_id: str) -> None:
        """Forget per-episode accounting without retaining user queries."""

        with self._web_search_lock:
            if self._web_search_episode_id == str(episode_id):
                self._web_search_episode_id = None
                self._web_search_call_count = 0
        with self._browser_lock:
            if self._browser_episode_id == str(episode_id):
                self._browser_episode_id = None
                self._browser_call_count = 0

    def _execute_web_search(
        self, command: str, arguments: Mapping[str, Any]
    ) -> ToolResult:
        try:
            import requests
        except ImportError as exc:
            return ToolResult(
                command=command,
                stdout="",
                stderr=f"web_search unavailable: {exc}",
                exit_code=127,
                timed_out=False,
                truncated=False,
                information_lines=[],
            )
        if not self.web_search_enabled:
            return ToolResult(
                command=command,
                stdout="",
                stderr="web_search unavailable: SERPER_API_KEY is not set",
                exit_code=-2,
                timed_out=False,
                truncated=False,
                information_lines=[],
            )
        with self._web_search_lock:
            if (
                self.web_search_max_calls is not None
                and self._web_search_call_count >= self.web_search_max_calls
            ):
                return ToolResult(
                    command=command,
                    stdout="",
                    stderr=(
                        "web_search call budget exceeded: "
                        f"{self._web_search_call_count}/{self.web_search_max_calls}"
                    ),
                    exit_code=-2,
                    timed_out=False,
                    truncated=False,
                    information_lines=[],
                )
            self._web_search_call_count += 1

        top_k = int(arguments.get("top_k", self.web_search_top_k))
        requested_timeout = float(arguments.get("timeout", self.web_search_timeout))
        timeout = min(requested_timeout, self.web_search_timeout)
        payload: dict[str, Any] = {"q": str(arguments["query"]), "num": top_k}
        for name in ("gl", "hl"):
            if name in arguments:
                payload[name] = str(arguments[name])
        headers = {
            "X-API-KEY": self._web_search_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = requests.post(
                self.web_search_endpoint,
                headers=headers,
                json=payload,
                timeout=timeout,
                allow_redirects=False,
            )
            status_code = int(getattr(response, "status_code", 200))
            if 300 <= status_code < 400:
                raise RuntimeError("Serper returned an unexpected redirect")
            response.raise_for_status()
            content_type = str(response.headers.get("content-type", "")).lower()
            if "json" not in content_type and not response.text.lstrip().startswith(
                ("{", "[")
            ):
                raise RuntimeError("Serper returned a non-JSON response")
            raw = response.json()
            if not isinstance(raw, Mapping):
                raise RuntimeError("Serper returned a non-object JSON response")
            explicit_error = raw.get("error") or raw.get("message")
            if explicit_error and not any(
                raw.get(name)
                for name in (
                    "answerBox",
                    "knowledgeGraph",
                    "organic",
                    "news",
                    "peopleAlsoAsk",
                )
            ):
                raise RuntimeError(f"Serper error: {str(explicit_error)[:300]}")
            results = _normalize_serper_results(raw, top_k=top_k)
            try:
                stdout, truncated = _serialize_web_search_output(
                    query=str(arguments["query"]),
                    results=results,
                    max_tokens=self.max_output_tokens,
                    tokenizer=self.tokenizer,
                )
            except Exception as exc:
                return ToolResult(
                    command=command,
                    stdout="",
                    stderr=f"token truncation error: {type(exc).__name__}: {exc}",
                    exit_code=1,
                    timed_out=False,
                    truncated=True,
                    information_lines=[],
                )
            return ToolResult(
                command=command,
                stdout=stdout,
                stderr="",
                exit_code=0,
                timed_out=False,
                truncated=truncated,
                information_lines=[
                    line for line in stdout.splitlines() if line.strip()
                ],
            )
        except requests.Timeout:
            message = f"[web_search timed out after {timeout}s]"
            return ToolResult(
                command=command,
                stdout=message,
                stderr="",
                exit_code=-1,
                timed_out=True,
                truncated=False,
                information_lines=[message],
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            diagnostic = f"{type(exc).__name__}: {exc}"
            if self._web_search_api_key:
                diagnostic = diagnostic.replace(self._web_search_api_key, "[REDACTED]")
            return ToolResult(
                command=command,
                stdout="",
                stderr=f"web_search error: {diagnostic}",
                exit_code=1,
                timed_out=False,
                truncated=False,
                information_lines=[],
            )

    def _execute_browser(
        self, command: str, arguments: Mapping[str, Any]
    ) -> ToolResult:
        """Run LaPha-compatible URL browsing through a locked Serper adapter."""

        try:
            import requests
        except ImportError as exc:
            return ToolResult(
                command=command,
                stdout="",
                stderr=f"browser unavailable: {exc}",
                exit_code=127,
                timed_out=False,
                truncated=False,
                information_lines=[],
            )
        if not self.browser_enabled:
            return ToolResult(
                command=command,
                stdout="",
                stderr="browser unavailable: SERPER_API_KEY is not set",
                exit_code=-2,
                timed_out=False,
                truncated=False,
                information_lines=[],
            )
        with self._browser_lock:
            if self._browser_call_count >= self.browser_max_calls:
                return ToolResult(
                    command=command,
                    stdout="",
                    stderr=(
                        "browser call budget exceeded: "
                        f"{self._browser_call_count}/{self.browser_max_calls}"
                    ),
                    exit_code=-2,
                    timed_out=False,
                    truncated=False,
                    information_lines=[],
                )
            self._browser_call_count += 1

        requested_timeout = float(arguments.get("timeout", self.browser_timeout))
        timeout = min(requested_timeout, self.browser_timeout)
        requested_length = int(arguments.get("max_length", self.browser_max_length))
        max_length = min(requested_length, self.browser_max_length)
        headers = {
            "X-API-KEY": self._web_search_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = requests.post(
                self.browser_endpoint,
                headers=headers,
                json={
                    "url": str(arguments["url"]),
                    "includeMarkdown": True,
                },
                timeout=timeout,
                allow_redirects=False,
            )
            status_code = int(getattr(response, "status_code", 200))
            if 300 <= status_code < 400:
                raise RuntimeError("Serper Scrape returned an unexpected redirect")
            response.raise_for_status()
            content_type = str(response.headers.get("content-type", "")).lower()
            if "json" in content_type or response.text.lstrip().startswith(("{", "[")):
                raw = response.json()
                if isinstance(raw, Mapping):
                    payload: Mapping[str, Any] = raw
                else:
                    payload = {"text": json.dumps(raw, ensure_ascii=False)}
            else:
                payload = {"text": response.text}
            explicit_error = payload.get("error") or payload.get("message")
            if explicit_error and not any(
                payload.get(name)
                for name in (
                    "text",
                    "content",
                    "markdown",
                    "metadata",
                    "jsonld",
                    "jsonLd",
                )
            ):
                raise RuntimeError(f"Serper Scrape error: {str(explicit_error)[:300]}")
            page = _normalize_serper_scrape(
                payload,
                url=str(arguments["url"]),
                max_chars=max_length,
            )
            page["include_markdown"] = True
            page["content_type"] = "application/json"
            page["status_code"] = status_code
            stdout, truncated = _serialize_browser_output(
                page,
                max_tokens=self.max_output_tokens,
                tokenizer=self.tokenizer,
            )
            return ToolResult(
                command=command,
                stdout=stdout,
                stderr="",
                exit_code=0,
                timed_out=False,
                truncated=truncated,
                information_lines=[stdout] if stdout else [],
            )
        except requests.Timeout:
            message = f"[browser timed out after {timeout}s]"
            return ToolResult(
                command=command,
                stdout=message,
                stderr="",
                exit_code=-1,
                timed_out=True,
                truncated=False,
                information_lines=[message],
            )
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            diagnostic = f"{type(exc).__name__}: {exc}"
            if self._web_search_api_key:
                diagnostic = diagnostic.replace(self._web_search_api_key, "[REDACTED]")
            return ToolResult(
                command=command,
                stdout="",
                stderr=f"browser error: {diagnostic}",
                exit_code=1,
                timed_out=False,
                truncated=False,
                information_lines=[],
            )

    def execute(self, command: str) -> ToolResult:
        try:
            browser_arguments = parse_browser_command(command)
        except ToolError as exc:
            return _validation_result(command, exc)
        if browser_arguments is not None:
            return self._execute_browser(command, browser_arguments)
        try:
            web_search_arguments = parse_web_search_command(command)
        except ToolError as exc:
            return _validation_result(command, exc)
        if web_search_arguments is not None:
            return self._execute_web_search(command, web_search_arguments)
        return run_tool(
            command,
            corpus_dir=self.corpus_dir,
            tokenizer=self.tokenizer,
            timeout=self.timeout,
            max_tokens=self.max_output_tokens,
        )

    def restricted(
        self,
        tools: Sequence[str],
        *,
        shell_corpus: str = SHELL_CORPUS_WIKI18,
    ) -> "GrepSeekToolModeRunner":
        return GrepSeekToolModeRunner(
            self,
            tools=tools,
            shell_corpus=shell_corpus,
        )

    def run(self, command: str) -> str:
        """Return the trajectory-compatible seven-key JSON string."""

        return self.execute(command).to_payload()

    run_json = run

    def __call__(self, command: str) -> str:
        return self.run(command)


class GrepSeekToolModeRunner:
    """Per-request view that permits the user-selected tool set."""

    def __init__(
        self,
        runner: GrepSeekCorpusToolRunner,
        *,
        tools: Sequence[str],
        shell_corpus: str = SHELL_CORPUS_WIKI18,
    ) -> None:
        selected = tuple(str(name) for name in tools)
        if not selected:
            raise ValueError("at least one tool must be selected")
        if len(set(selected)) != len(selected):
            raise ValueError("selected tools must be unique")
        unsupported = sorted(set(selected) - {"shell", "web_search", "browser"})
        if unsupported:
            raise ValueError(f"unsupported tools: {unsupported}")
        if "web_search" in selected and not runner.web_search_enabled:
            raise RuntimeError("web_search is unavailable because no API key is set")
        if "browser" in selected and not runner.browser_enabled:
            raise RuntimeError("browser is unavailable because no API key is set")
        corpus_mode = str(shell_corpus)
        if corpus_mode not in SHELL_CORPUS_MODES:
            raise ValueError(f"unsupported shell corpus mode: {corpus_mode!r}")
        self.runner = runner
        self.tools = frozenset(selected)
        self.shell_corpus = corpus_mode
        self.corpus_dir = runner.corpus_dir
        self._episode_id: Optional[str] = None
        self._memory_tempdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._memory_ready = False

    def begin_episode(self, episode_id: str) -> None:
        normalized = str(episode_id)
        self.runner.begin_episode(normalized)
        try:
            self._episode_id = normalized
            self._memory_ready = False
            if self.shell_corpus == SHELL_CORPUS_REACT_MEMORY:
                self._memory_tempdir = tempfile.TemporaryDirectory(
                    prefix="grepseek-react-memory-"
                )
                self.corpus_dir = Path(self._memory_tempdir.name).resolve()
        except Exception:
            self.runner.end_episode(normalized)
            raise

    def end_episode(self, episode_id: str) -> None:
        normalized = str(episode_id)
        try:
            if self._episode_id == normalized and self._memory_tempdir is not None:
                self._memory_tempdir.cleanup()
        finally:
            self._memory_tempdir = None
            self._memory_ready = False
            self._episode_id = None
            self.corpus_dir = self.runner.corpus_dir
            self.runner.end_episode(normalized)

    def update_episode_memory(
        self,
        *,
        episode_id: str,
        messages: Sequence[Mapping[str, str]],
        round_index: int,
    ) -> None:
        """Atomically rebuild the grep corpus from the current ReAct chain."""

        if self.shell_corpus != SHELL_CORPUS_REACT_MEMORY:
            return
        if str(episode_id) != self._episode_id or self._memory_tempdir is None:
            raise RuntimeError("ReAct memory corpus has no active episode")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index <= 0
        ):
            raise ValueError("round_index must be a positive integer")
        records: list[dict[str, Any]] = []
        assistant_round = 0
        archive_messages = redact_react_memory_query_results_for_archive(messages)
        for message_index, message in enumerate(archive_messages):
            if not isinstance(message, Mapping):
                raise TypeError("ReAct memory messages must be mappings")
            role = str(message.get("role", ""))
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError("ReAct memory message content must be a string")
            if role == "system":
                # The system contract is stable P, not episodic evidence, and
                # may contain deployment details that should not be searchable.
                continue
            if role == "user":
                if message_index == 1 and assistant_round == 0:
                    record_round = 0
                else:
                    if assistant_round <= 0:
                        raise ValueError(
                            "user-feedback memory appeared before an assistant turn"
                        )
                    record_round = assistant_round
            elif role == "assistant":
                assistant_round += 1
                record_round = assistant_round
            elif role == "tool":
                if assistant_round <= 0:
                    raise ValueError("tool memory appeared before an assistant turn")
                record_round = assistant_round
                records.append(
                    _react_memory_tool_record(
                        content,
                        record_round=record_round,
                    )
                )
                continue
            else:
                raise ValueError(f"unsupported ReAct memory role: {role!r}")
            records.append(
                {
                    "id": (
                        "round-0000-user"
                        if role == "user" and record_round == 0
                        else (
                            f"round-{record_round:04d}-user-feedback"
                            if role == "user"
                            else f"round-{record_round:04d}-assistant"
                        )
                    ),
                    "round": record_round,
                    "role": role,
                    "content": content,
                    "message_index": message_index,
                }
            )
        self._write_episode_memory_records(
            records, filename=REACT_MEMORY_FILENAME_ACTUAL
        )

    def update_compressed_episode_memory(
        self,
        *,
        episode_id: str,
        messages: Sequence[Mapping[str, str]],
        round_index: int,
        compression: Mapping[str, Any],
    ) -> None:
        """Archive complete raw rounds touched by Qwen token compression.

        ``compressed_message_indices`` identifies the exact serialized
        messages intersecting the compressed token prefix.
        ``archived_message_indices`` expands those intersections to complete
        assistant/observation exchanges, so hippocampal lookup can recover each
        assistant action plus either the exact tool stdout/stderr evidence or
        a controller user-feedback message even when a token block cuts
        through a message boundary.  The released raw ToolResult envelope
        remains in trajectory artifacts, not nested again inside this
        searchable JSONL projection.
        """

        if self.shell_corpus != SHELL_CORPUS_REACT_MEMORY:
            return
        if not isinstance(compression, Mapping):
            raise TypeError("compression receipt must be a mapping")
        archived_value = compression.get("archived_message_indices")
        if archived_value is None:
            # Older context organizations do not expose message-level token
            # receipts; preserve their existing all-chain memory behavior.
            self.update_episode_memory(
                episode_id=episode_id,
                messages=messages,
                round_index=round_index,
            )
            return
        if str(episode_id) != self._episode_id or self._memory_tempdir is None:
            raise RuntimeError("ReAct memory corpus has no active episode")
        if not isinstance(archived_value, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in archived_value
        ):
            raise ValueError("archived_message_indices must be a list of indices >= 2")
        archived_indices = set(archived_value)
        if len(archived_indices) != len(archived_value):
            raise ValueError("archived_message_indices must be unique")
        compressed_value = compression.get("compressed_message_indices", [])
        if not isinstance(compressed_value, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in compressed_value
        ):
            raise ValueError(
                "compressed_message_indices must be a list of indices >= 2"
            )
        compressed_indices = set(compressed_value)
        if not compressed_indices.issubset(archived_indices):
            raise ValueError("compressed messages must be included in the archive")

        records: list[dict[str, Any]] = []
        assistant_round = 0
        seen_indices: set[int] = set()
        archive_messages = redact_react_memory_query_results_for_archive(messages)
        for message_index, message in enumerate(archive_messages):
            if not isinstance(message, Mapping):
                raise TypeError("ReAct memory messages must be mappings")
            role = str(message.get("role", ""))
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError("ReAct memory message content must be a string")
            if role == "assistant":
                assistant_round += 1
                record_round = assistant_round
            elif role == "tool":
                if assistant_round <= 0:
                    raise ValueError("tool memory appeared before an assistant turn")
                record_round = assistant_round
            elif role == "user":
                if message_index == 1 and assistant_round == 0:
                    continue
                if assistant_round <= 0:
                    raise ValueError(
                        "user-feedback memory appeared before an assistant turn"
                    )
                record_round = assistant_round
            elif role == "system":
                continue
            else:
                raise ValueError(f"unsupported ReAct memory role: {role!r}")
            if message_index not in archived_indices:
                continue
            seen_indices.add(message_index)
            if role == "tool":
                records.append(
                    _react_memory_tool_record(
                        content,
                        record_round=record_round,
                        message_index=message_index,
                        compression_overlap=message_index in compressed_indices,
                    )
                )
            else:
                record_role = (
                    "user-feedback" if role == "user" else role
                )
                records.append(
                    {
                        "id": f"round-{record_round:04d}-{record_role}",
                        "round": record_round,
                        "role": role,
                        "content": content,
                        "message_index": message_index,
                        "compression_overlap": message_index in compressed_indices,
                    }
                )
        if seen_indices != archived_indices:
            missing = sorted(archived_indices - seen_indices)
            raise ValueError(
                f"archived messages are absent from the active chain: {missing[:8]}"
            )
        self._write_episode_memory_records(
            records, filename=REACT_MEMORY_FILENAME_ACTUAL
        )

    def _write_episode_memory_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        filename: str,
    ) -> None:
        if self._memory_tempdir is None:
            raise RuntimeError("ReAct memory corpus has no active episode")
        if Path(filename).name != filename:
            raise ValueError("memory filename must be one basename")
        serialized = "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ).encode("utf-8")
        if len(serialized) > _REACT_MEMORY_CORPUS_MAX_BYTES:
            limit_mib = _REACT_MEMORY_CORPUS_MAX_BYTES >> 20
            raise ValueError(
                "ReAct memory corpus exceeds the bounded "
                f"{limit_mib} MiB episode limit"
            )
        target = self.corpus_dir / filename
        temporary = self.corpus_dir / f".{filename}.tmp"
        with temporary.open("wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(self.corpus_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._memory_ready = True

    def execute(self, command: str) -> ToolResult:
        is_web_search = isinstance(command, str) and command.startswith(
            WEB_SEARCH_COMMAND_PREFIX
        )
        is_browser = isinstance(command, str) and command.startswith(
            BROWSER_COMMAND_PREFIX
        )
        if is_web_search and "web_search" not in self.tools:
            return _validation_result(
                command,
                ToolError("web_search is disabled for this rollout"),
            )
        if is_browser and "browser" not in self.tools:
            return _validation_result(
                command,
                ToolError("browser is disabled for this rollout"),
            )
        if not is_web_search and not is_browser and "shell" not in self.tools:
            return _validation_result(
                command,
                ToolError("the shell corpus tool is disabled for this rollout"),
            )
        if (
            not is_web_search
            and not is_browser
            and self.shell_corpus == SHELL_CORPUS_REACT_MEMORY
        ):
            if not self._memory_ready:
                return _validation_result(
                    command,
                    ToolError("the ReAct memory corpus is not initialized"),
                )
            if CORPUS_FILENAME_LOGICAL in command:
                return _validation_result(
                    command,
                    ToolError(
                        "the hippocampus corpus is memory.jsonl, not corpus.jsonl"
                    ),
                )
            if command.count(REACT_MEMORY_FILENAME_LOGICAL) != 1:
                return _validation_result(
                    command,
                    ToolError("pipeline must name 'memory.jsonl' exactly once"),
                )
            translated_command = command.replace(
                REACT_MEMORY_FILENAME_LOGICAL, CORPUS_FILENAME_LOGICAL
            )
            result = run_tool(
                translated_command,
                corpus_dir=self.corpus_dir,
                tokenizer=self.runner.tokenizer,
                timeout=self.runner.timeout,
                max_tokens=self.runner.max_output_tokens,
                actual_corpus_filename=REACT_MEMORY_FILENAME_ACTUAL,
            )
            return replace(result, command=command)
        return self.runner.execute(command)


def preflight_corpus_tool(runner: GrepSeekCorpusToolRunner) -> dict[str, Any]:
    """Execute one bounded real-corpus ``rg`` query and fail closed.

    Construction of :class:`GrepSeekCorpusToolRunner` verifies paths and
    numeric arguments, but deliberately does not spawn a process.  This probe
    additionally proves that the whitelisted executable, actual corpus file,
    tokenizer truncation adapter, and seven-field result contract all work
    before an expensive model is loaded.
    """

    started = time.monotonic()
    result = runner.execute(CORPUS_PREFLIGHT_COMMAND)
    elapsed_seconds = time.monotonic() - started
    match = _CORPUS_PREFLIGHT_STDOUT.fullmatch(result.stdout)
    expected_information_lines = (
        [result.stdout.removesuffix("\n")] if match is not None else None
    )
    failures: list[str] = []
    if result.command != CORPUS_PREFLIGHT_COMMAND:
        failures.append("result command differs from fixed preflight command")
    if result.exit_code != 0:
        failures.append(f"exit_code={result.exit_code}")
    if result.stderr:
        failures.append(f"stderr={result.stderr[:512]!r}")
    if result.timed_out:
        failures.append("timed_out=true")
    if result.truncated:
        failures.append("truncated=true")
    if match is None:
        failures.append(f"unexpected stdout={result.stdout[:512]!r}")
    if (
        expected_information_lines is None
        or result.information_lines != expected_information_lines
    ):
        failures.append("information_lines disagree with the single matched id")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        failures.append(f"invalid elapsed_seconds={elapsed_seconds!r}")
    if failures:
        raise RuntimeError("GrepSeek corpus preflight failed: " + "; ".join(failures))

    if match is None:  # narrowed by the failure branch above
        raise AssertionError("successful corpus preflight has no id match")
    metadata: dict[str, Any] = {
        "status": "ok",
        "runner": type(runner).__name__,
        "command": CORPUS_PREFLIGHT_COMMAND,
        "corpus_dir": str(runner.corpus_dir),
        "corpus_file": str(runner.corpus_dir / CORPUS_FILENAME_ACTUAL),
        "matched_id": match.group(1),
        "stdout": result.stdout,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
        "information_lines": list(result.information_lines),
        "elapsed_seconds": elapsed_seconds,
    }
    # Keep this health metadata directly JSON serializable for manifests and
    # demo records.  ``allow_nan=False`` also guards future timing changes.
    json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    return metadata


__all__ = [
    "ALLOWED_COMMANDS",
    "CORPUS_FILENAME_ACTUAL",
    "CORPUS_FILENAME_LOGICAL",
    "CORPUS_PREFLIGHT_COMMAND",
    "DEFAULT_BROWSER_MAX_CALLS",
    "DEFAULT_BROWSER_MAX_LENGTH",
    "DEFAULT_BROWSER_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WEB_SEARCH_MAX_CALLS",
    "DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS",
    "DEFAULT_WEB_SEARCH_TOP_K",
    "GrepSeekCorpusToolRunner",
    "GrepSeekToolModeRunner",
    "REACT_MEMORY_CANONICAL_SHELL_PATH",
    "REACT_MEMORY_QUERY_RESULT_ARCHIVE_POLICY",
    "REACT_MEMORY_QUERY_RESULT_ARCHIVE_STUB",
    "SHELL_CORPUS_MODES",
    "SHELL_CORPUS_REACT_MEMORY",
    "SHELL_CORPUS_WIKI18",
    "TOOL_RESPONSE_KEYS",
    "ToolError",
    "ToolResult",
    "append_browser_system_prompt",
    "append_web_search_system_prompt",
    "canonical_browser_command",
    "canonical_web_search_command",
    "parse_browser_command",
    "parse_web_search_command",
    "react_memory_shell_system_prompt",
    "run_tool",
    "preflight_corpus_tool",
    "redact_react_memory_query_results_for_archive",
    "truncate_tokens",
    "validate_pipeline",
]
