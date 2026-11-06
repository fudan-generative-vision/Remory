"""Read-only Streamlit monitor for Harbor Terminal-Bench jobs.

This mirrors the useful BrowseComp monitor contract while preserving
Terminal-Bench semantics: the answer is environment state, so completed trials
show the final agent message and verifier artifacts rather than a made-up text
ground truth. Evaluation artifacts are never modified and the monitor creates
no authentication sidecar.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shlex
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from terminal_bench_eval.codex_image_store import IMAGE_STORE_DIRNAME, MAX_IMAGE_BYTES
from terminal_bench_eval.codex_r2eact_artifacts import (
    CONTROLLER_ARTIFACT_DIRNAME,
    CONTROLLER_ARTIFACT_FILENAMES,
    GATEWAY_EVENTS_FILENAME,
    GATEWAY_IMAGE_RECEIPTS_FILENAME,
    GATEWAY_RECEIPT_FILENAMES,
    GATEWAY_RECEIPT_SUMMARY_SCHEMA,
    GATEWAY_RECEIPTS_JSONL_FILENAME,
    GATEWAY_RECEIPTS_SUMMARY_FILENAME,
    GATEWAY_ROUNDS_FILENAME,
    GATEWAY_TOOL_RESPONSES_FILENAME,
    GATEWAY_WEB_SEARCH_RECEIPTS_FILENAME,
    MEMORY_FILENAME,
)
from terminal_bench_eval.codex_r2eact_receipts import (
    GatewayReceiptPersistenceError,
    decode_receipt_record,
)
from terminal_bench_eval.harbor_unbounded_agent import (
    AGENT_TIME_BUDGET_RECEIPT_NAME,
    AGENT_TIME_BUDGET_RECEIPT_SCHEMA,
    TB21_DATASET_SOURCE,
    TB21_OFFICIAL_AGENT_TIMEOUT_POLICY,
    TB21_LEGACY_8H_AGENT_TIMEOUT_POLICY,
    TB21_PER_TASK_AGENT_TIMEOUT_SECONDS,
    TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
    TB21_UNIFORM_AGENT_TIMEOUT_SOURCE,
    UNBOUNDED_AGENT_TIMEOUT_POLICY,
    agent_import_uses_unbounded_timeout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = Path(
    os.environ.get(
        "R2EACT_TB_MONITOR_JOBS_ROOT",
        str(PROJECT_ROOT / "eval" / "terminal_bench" / "jobs"),
    )
).expanduser()
PREFERRED_JOB_DIR_RAW = os.environ.get("R2EACT_TB_MONITOR_JOB_DIR", "").strip()
MINIMUM_MONITOR_REFRESH_SECONDS = 30.0


def _effective_refresh_seconds(configured_seconds: Any) -> float:
    """Keep trajectory rendering readable while still updating live state."""

    try:
        configured = float(configured_seconds)
    except (TypeError, ValueError):
        return MINIMUM_MONITOR_REFRESH_SECONDS
    if not math.isfinite(configured):
        return MINIMUM_MONITOR_REFRESH_SECONDS
    return max(MINIMUM_MONITOR_REFRESH_SECONDS, configured)


REFRESH_SECONDS = _effective_refresh_seconds(
    os.environ.get("R2EACT_TB_MONITOR_REFRESH_SECONDS", "30")
)
SGLANG_LOAD_URL = os.environ.get(
    "R2EACT_TB_MONITOR_SGLANG_LOAD_URL", "http://127.0.0.1:30000/get_load"
).strip()
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_JSONL_BYTES = 16 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000
MAX_LIVE_ROUNDS = 8
MAX_STEP_MESSAGE_CHARS = 48_000
MAX_STEP_REASONING_CHARS = 48_000
MAX_STEP_OUTPUT_CHARS = 96_000
MAX_JSON_CACHE_ENTRIES = 24
MAX_JSON_CACHE_SOURCE_BYTES = 16 * 1024 * 1024
MAX_TRIAL_PROJECTION_CACHE_ENTRIES = 2_048
MAX_INCREMENTAL_JSONL_CACHE_ENTRIES = 256
MAX_JSONL_HEAD_SCAN_BYTES = 1024 * 1024
POOL_SCHEMA = "terminal_bench_shared_pool_v1"
POOL_CELL_RESULT_SCHEMA = "terminal_bench_pool_cell_result_v1"
POOL_STATE_SCHEMA = "terminal_bench_pool_state_v1"
POOL_WORKER_EXCEPTION_SCHEMA = "terminal_bench_pool_worker_exception_v1"
POOL_EXECUTION_SCHEMA = "terminal_bench_pool_execution_v1"
POOL_SCORE_PROBE_SCHEMA = "terminal_bench_monitor_cell_score_probe_v1"
POOL_SCORE_AUTHORITY = (
    "terminal_bench_pool._scoreable_cell_reward+"
    "_validate_verified_agent_sigterm_reconciliation"
)
POOL_AGENT_TIMEOUT_SCORE_AUTHORITY = (
    "terminal_bench_pool._scoreable_cell_reward+_verified_reward_after_agent_timeout"
)
LOGICAL_POOL_SCHEMA = "terminal_bench_logical_pool_v1"
POOL_DISPLAY_SCHEMA = "terminal_bench_pool_display_v1"
ATTEMPT_SELECTION_SCHEMA = "terminal_bench_pool_attempt_selection_v1"
QUEUED_ATTEMPT_RETIREMENT_POLICY = "keep_single_official_attempt_v1"
TASK_SELECTION_SCHEMA = "terminal_bench_pool_task_selection_v1"
QUEUED_TASK_RETIREMENT_POLICY = "retire_selected_queued_tasks_v1"
LOGICAL_POOL_FILENAME = "logical-pool.json"
POOL_DISPLAY_FILENAME = "pool-display.json"
MAX_POOL_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_CODEX_LOG_BYTES = 64 * 1024 * 1024
POOL_SCORE_PROBE_TIMEOUT_SECONDS = 30.0
POOL_SCORE_PROBE_PYTHON = Path(
    os.environ.get(
        "R2EACT_TB_MONITOR_POOL_PYTHON",
        str(
            PROJECT_ROOT
            / "eval"
            / "terminal_bench"
            / "compressor_venv"
            / "bin"
            / "python"
        ),
    )
).expanduser()
PROOT_CACHE_ROOT = Path(
    os.environ.get(
        "R2EACT_TB_MONITOR_PROOT_CACHE_ROOT",
        os.environ.get("HARBOR_PROOT_CACHE", "/tmp/harbor-proot-cache"),
    )
).expanduser()

EVENT_FILENAMES = (
    GATEWAY_EVENTS_FILENAME,
    "r2eact-terminus-2-events.jsonl",
    "r2eact-events.jsonl",
    "terminus-2-qwen-raw-events.jsonl",
)
RECEIPT_FILENAMES = (
    *GATEWAY_RECEIPT_FILENAMES,
    "r2eact-terminus-2-receipts.json",
    "r2eact-receipts.json",
)
TERMINAL_PANE_FILENAMES = ("terminus_2.pane", "terminus-2.pane")
CODEX_ROUND_TRACE_FILENAME = GATEWAY_ROUNDS_FILENAME
CODEX_ROUND_TRACE_SCHEMA = "codex_r2eact_round_trace_v1"
CODEX_TOOL_RESPONSE_TRACE_FILENAME = GATEWAY_TOOL_RESPONSES_FILENAME
CODEX_TOOL_RESPONSE_TRACE_SCHEMA_V1 = "codex_r2eact_tool_response_trace_v1"
CODEX_TOOL_RESPONSE_TRACE_SCHEMA = "codex_r2eact_tool_response_trace_v2"


class MonitorArtifactError(RuntimeError):
    """A monitor artifact violated containment or size constraints."""


_JSON_CACHE: OrderedDict[str, tuple[tuple[int, int], dict[str, Any], int]] = (
    OrderedDict()
)
_JSON_CACHE_SOURCE_BYTES = 0
_JSON_CACHE_LOCK = threading.Lock()
_TRIAL_PROJECTION_CACHE: dict[
    str, tuple[tuple[tuple[int, int], ...], dict[str, Any]]
] = {}
_INCREMENTAL_JSONL_CACHE: OrderedDict[tuple[str, str], dict[str, Any]] = (
    OrderedDict()
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _as_number(value: Any) -> float | int | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _as_finite_number(value: Any) -> float | int | None:
    number = _as_number(value)
    if number is None or not math.isfinite(float(number)):
        return None
    return number


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    total = int(seconds)
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_count(value: Any) -> str:
    number = _as_number(value)
    return "—" if number is None else f"{int(number):,}"


def _format_cumulative_count(value: Any, *, exact: bool) -> str:
    rendered = _format_count(value)
    if rendered == "—" or exact:
        return rendered
    return f"≥{rendered}"


def _elapsed(started_at: Any, finished_at: Any = None) -> float | None:
    start = _parse_time(started_at)
    if start is None:
        return None
    end = _parse_time(finished_at) or datetime.now(UTC)
    return max(0.0, (end - start).total_seconds())


def _truncate(value: Any, limit: int, *, keep_tail: bool = False) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    marker = "\n[monitor display truncated]\n"
    if keep_tail:
        return marker + text[-max(0, limit - len(marker)) :]
    return text[: max(0, limit - len(marker))] + marker


def _without_token_ids(value: Any) -> Any:
    """Recursively remove bulky token-id arrays from display projections."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_token_ids(item)
            for key, item in value.items()
            if str(key) not in {"token_ids", "completion_token_ids", "input_token_ids"}
            and not str(key).endswith("_token_ids")
        }
    if isinstance(value, list):
        return [_without_token_ids(item) for item in value]
    return value


def _assert_within(path: Path, root: Path, *, kind: str) -> Path:
    resolved_root = Path(root).expanduser().resolve(strict=True)
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise MonitorArtifactError(f"{kind} must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise MonitorArtifactError(f"unable to resolve {kind}") from exc
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise MonitorArtifactError(f"{kind} escapes jobs root")
    return resolved


def _has_symlink_component(path: Path) -> bool:
    """Reject a regular-looking artifact reached through a symlinked directory."""

    candidate = Path(path).absolute()
    for component in (candidate, *candidate.parents):
        try:
            if component.is_symlink():
                return True
        except OSError:
            return True
    return False


def _read_json(
    path: Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
    retries: int = 3,
    use_last_good: bool = True,
) -> Mapping[str, Any] | None:
    """Read a bounded mapping, tolerating Harbor's in-place JSON rewrites."""

    global _JSON_CACHE_SOURCE_BYTES
    path = Path(path)
    if _has_symlink_component(path):
        return None
    cache_key = str(path.absolute())
    for attempt in range(max(1, retries)):
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                return None
            signature = (info.st_mtime_ns, info.st_size)
            with _JSON_CACHE_LOCK:
                cached = _JSON_CACHE.get(cache_key)
                if cached is not None and cached[0] == signature:
                    _JSON_CACHE.move_to_end(cache_key)
                    return dict(cached[1])
            raw = path.read_bytes()
            if len(raw) > max_bytes:
                return None
            value = json.loads(raw)
            if not isinstance(value, Mapping):
                return None
            projected = dict(value)
            with _JSON_CACHE_LOCK:
                previous = _JSON_CACHE.pop(cache_key, None)
                if previous is not None:
                    _JSON_CACHE_SOURCE_BYTES -= previous[2]
                if len(raw) <= MAX_JSON_CACHE_SOURCE_BYTES:
                    _JSON_CACHE[cache_key] = (signature, projected, len(raw))
                    _JSON_CACHE_SOURCE_BYTES += len(raw)
                while (
                    len(_JSON_CACHE) > MAX_JSON_CACHE_ENTRIES
                    or _JSON_CACHE_SOURCE_BYTES > MAX_JSON_CACHE_SOURCE_BYTES
                ):
                    _, removed = _JSON_CACHE.popitem(last=False)
                    _JSON_CACHE_SOURCE_BYTES -= removed[2]
            return dict(projected)
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return None
        except json.JSONDecodeError:
            if attempt + 1 < max(1, retries):
                time.sleep(0.02)
                continue
    if use_last_good:
        with _JSON_CACHE_LOCK:
            cached = _JSON_CACHE.get(cache_key)
            if cached is not None:
                _JSON_CACHE.move_to_end(cache_key)
                return dict(cached[1])
    return None


def _read_json_value(
    path: Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
    retries: int = 3,
) -> Mapping[str, Any] | list[Any] | None:
    """Read one bounded JSON mapping/list without weakening `_read_json`.

    Harbor's ordinary artifacts are mappings, while the Codex r²eAct gateway
    intentionally persists its per-round receipts as a bare array.  Keeping
    this reader separate preserves the mapping-only contract relied on by pool
    state parsing.
    """

    path = Path(path)
    if _has_symlink_component(path):
        return None
    for attempt in range(max(1, retries)):
        try:
            info = path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                return None
            raw = path.read_bytes()
            if len(raw) > max_bytes:
                return None
            value = json.loads(raw)
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, list):
                return value
            return None
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            return None
        except json.JSONDecodeError:
            if attempt + 1 < max(1, retries):
                time.sleep(0.02)
                continue
    return None


def _complete_jsonl(
    path: Path, *, tail_bytes: int | None = None
) -> Iterable[dict[str, Any]]:
    path = Path(path)
    if _has_symlink_component(path):
        return
    try:
        handle = path.open("rb")
    except OSError:
        return
    with handle:
        if tail_bytes is not None:
            try:
                size = path.stat().st_size
            except OSError:
                return
            if size > tail_bytes:
                handle.seek(size - tail_bytes)
                handle.readline()
        for raw in handle:
            if not raw.endswith(b"\n"):
                continue
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, Mapping):
                yield dict(value)


def _first_jsonl_number(path: Path, key: str) -> float | int | None:
    """Read an early scalar without mistaking a bounded tail for file start."""

    path = Path(path)
    if _has_symlink_component(path):
        return None
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):
            return None
        consumed = 0
        with path.open("rb") as handle:
            while consumed < MAX_JSONL_HEAD_SCAN_BYTES:
                raw = handle.readline(MAX_JSONL_HEAD_SCAN_BYTES - consumed + 1)
                if not raw:
                    break
                consumed += len(raw)
                if consumed > MAX_JSONL_HEAD_SCAN_BYTES:
                    break
                if not raw.endswith(b"\n"):
                    continue
                try:
                    value = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, Mapping):
                    number = _as_number(value.get(key))
                    if number is not None:
                        return number
    except OSError:
        return None
    return None


def _new_incremental_jsonl_state(
    *, kind: str, device: int, inode: int
) -> dict[str, Any]:
    return {
        "kind": kind,
        "device": device,
        "inode": inode,
        "offset": 0,
        "mtime_ns": 0,
        "invalid_records": 0,
        "partial_record": False,
        "memory_records": 0,
        "memory_rounds": set(),
        "memory_max_round": 0,
        "round_metrics": {},
        "receipt_records": 0,
        "receipt_previous_digest": None,
        "receipt_latest_round": 0,
        "receipt_input_tokens": 0,
        "receipt_output_tokens": 0,
        "receipt_tool_calls": 0,
        "receipt_all_input_tokens": True,
        "receipt_all_output_tokens": True,
        "receipt_all_tool_calls": True,
        "receipt_latest": {},
    }


def _update_incremental_jsonl_state(path: Path, *, kind: str) -> dict[str, Any] | None:
    """Incrementally reduce an append-only controller JSONL artifact.

    Live gateway artifacts can grow far beyond the bounded transcript-rendering
    window.  This reducer retains only counters (and one tiny tuple per round),
    resumes at the last complete newline, and restarts if the file is replaced or
    truncated.  It therefore keeps cumulative telemetry exact without loading a
    multi-hundred-megabyte artifact on every Streamlit refresh.
    """

    path = Path(path)
    if kind not in {"memory", "round_trace", "receipt"} or _has_symlink_component(
        path
    ):
        return None
    try:
        info = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode):
        return None

    cache_key = (kind, str(path.absolute()))
    with _JSON_CACHE_LOCK:
        state = _INCREMENTAL_JSONL_CACHE.get(cache_key)
        reset = (
            state is None
            or state.get("device") != info.st_dev
            or state.get("inode") != info.st_ino
            or info.st_size < _as_int(state.get("offset"))
            or (
                info.st_size == _as_int(state.get("offset"))
                and info.st_mtime_ns != _as_int(state.get("mtime_ns"))
            )
        )
        if reset:
            state = _new_incremental_jsonl_state(
                kind=kind, device=info.st_dev, inode=info.st_ino
            )
        elif (
            info.st_size == _as_int(state.get("offset"))
            and info.st_mtime_ns == _as_int(state.get("mtime_ns"))
        ):
            _INCREMENTAL_JSONL_CACHE.move_to_end(cache_key)
            return dict(state)

        state["partial_record"] = False
        try:
            with path.open("rb") as handle:
                handle.seek(_as_int(state.get("offset")))
                while True:
                    line_start = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    if not raw.endswith(b"\n"):
                        state["partial_record"] = True
                        break
                    state["offset"] = handle.tell()
                    try:
                        value = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        state["invalid_records"] = (
                            _as_int(state.get("invalid_records")) + 1
                        )
                        continue
                    if not isinstance(value, Mapping):
                        state["invalid_records"] = (
                            _as_int(state.get("invalid_records")) + 1
                        )
                        continue
                    if kind == "memory":
                        state["memory_records"] = (
                            _as_int(state.get("memory_records")) + 1
                        )
                        round_index = _as_int(value.get("round"))
                        if round_index > 0:
                            state["memory_rounds"].add(round_index)
                            state["memory_max_round"] = max(
                                _as_int(state.get("memory_max_round")), round_index
                            )
                        continue

                    if kind == "receipt":
                        try:
                            receipt, digest = decode_receipt_record(
                                value,
                                expected_index=(
                                    _as_int(state.get("receipt_records")) + 1
                                ),
                                expected_previous_sha256=state.get(
                                    "receipt_previous_digest"
                                ),
                            )
                        except GatewayReceiptPersistenceError:
                            # A canonical journal is hash chained. Do not skip a
                            # bad record and accidentally bless its suffix.
                            state["invalid_records"] = (
                                _as_int(state.get("invalid_records")) + 1
                            )
                            state["offset"] = line_start
                            state["partial_record"] = True
                            break
                        state["receipt_records"] = (
                            _as_int(state.get("receipt_records")) + 1
                        )
                        state["receipt_previous_digest"] = digest
                        round_index = _as_int(receipt.get("round_index"))
                        if round_index >= _as_int(
                            state.get("receipt_latest_round")
                        ):
                            state["receipt_latest_round"] = round_index
                            state["receipt_latest"] = dict(receipt)
                        input_tokens = _as_number(receipt.get("input_tokens"))
                        if input_tokens is None:
                            state["receipt_all_input_tokens"] = False
                        else:
                            state["receipt_input_tokens"] = _as_int(
                                state.get("receipt_input_tokens")
                            ) + int(input_tokens)
                        output_tokens = _as_number(receipt.get("output_tokens"))
                        if output_tokens is None:
                            output_tokens = _as_number(receipt.get("token_count"))
                        if output_tokens is None:
                            state["receipt_all_output_tokens"] = False
                        else:
                            state["receipt_output_tokens"] = _as_int(
                                state.get("receipt_output_tokens")
                            ) + int(output_tokens)
                        tool_calls = receipt.get("tool_calls")
                        if not isinstance(tool_calls, list):
                            state["receipt_all_tool_calls"] = False
                        else:
                            state["receipt_tool_calls"] = _as_int(
                                state.get("receipt_tool_calls")
                            ) + len(tool_calls)
                        continue

                    record = _validated_round_trace_record(value)
                    if record is None:
                        state["invalid_records"] = (
                            _as_int(state.get("invalid_records")) + 1
                        )
                        continue
                    round_index = _as_int(record.get("round_index"))
                    state["round_metrics"][round_index] = (
                        _as_number(record.get("input_tokens")),
                        _as_number(record.get("output_tokens")),
                        len(record.get("tool_calls") or []),
                        _as_number(record.get("observed_at")),
                    )
                final_info = os.fstat(handle.fileno())
        except OSError:
            return None

        state["mtime_ns"] = final_info.st_mtime_ns
        state["observed_size"] = final_info.st_size
        _INCREMENTAL_JSONL_CACHE[cache_key] = state
        _INCREMENTAL_JSONL_CACHE.move_to_end(cache_key)
        while len(_INCREMENTAL_JSONL_CACHE) > MAX_INCREMENTAL_JSONL_CACHE_ENTRIES:
            _INCREMENTAL_JSONL_CACHE.popitem(last=False)
        return dict(state)


def _bounded_text(
    path: Path, *, max_chars: int = MAX_TEXT_CHARS, tail: bool = False
) -> str:
    path = Path(path)
    if _has_symlink_component(path):
        return ""
    try:
        with path.open("rb") as handle:
            if tail:
                size = path.stat().st_size
                byte_limit = max_chars * 4
                if size > byte_limit:
                    handle.seek(size - byte_limit)
                    handle.readline()
            raw = handle.read(max_chars * 4 + 1)
    except OSError:
        return ""
    return _truncate(raw.decode("utf-8", errors="replace"), max_chars, keep_tail=tail)


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path:
    candidates = [
        directory / CONTROLLER_ARTIFACT_DIRNAME / name
        for name in names
        if name in CONTROLLER_ARTIFACT_FILENAMES
    ]
    candidates.extend(directory / name for name in names)
    for candidate in candidates:
        if candidate.is_file() and not _has_symlink_component(candidate):
            return candidate
    return directory / names[0]


def _controller_artifact_path(directory: Path, filename: str) -> Path:
    """Prefer isolated controller state while retaining legacy readability."""

    if filename not in CONTROLLER_ARTIFACT_FILENAMES:
        raise ValueError(f"not a controller artifact filename: {filename}")
    isolated = directory / CONTROLLER_ARTIFACT_DIRNAME / filename
    if isolated.is_file() and not _has_symlink_component(isolated):
        return isolated
    return directory / filename


def _discover_jobs(jobs_root: Path) -> list[Path]:
    try:
        root = Path(jobs_root).expanduser().resolve(strict=True)
        candidates = list(root.iterdir())
    except OSError:
        return []
    jobs: list[Path] = []
    for path in candidates:
        if path.is_symlink() or not path.is_dir():
            continue
        if not ((path / "config.json").is_file() or (path / "result.json").is_file()):
            continue
        try:
            jobs.append(_assert_within(path, root, kind="job directory"))
        except MonitorArtifactError:
            continue
    # A pool directory's mtime is not its creation time: remote workers update
    # heartbeats, cells, and state beneath it for the lifetime of the run.  If
    # directory mtime decides the selector order, an older active pool can stay
    # ahead of a newly initialized release forever.  Pool initialization is an
    # immutable identity receipt and therefore the stable ordering authority.
    return sorted(jobs, key=_job_discovery_sort_key, reverse=True)


def _job_discovery_sort_key(job_dir: Path) -> tuple[int, str]:
    marker = _pool_marker(job_dir)
    if marker is not None:
        initialized_at = _parse_time(marker.get("initialized_at"))
        if initialized_at is not None:
            return int(initialized_at.timestamp() * 1_000_000_000), job_dir.name
    identity_timestamps: list[int] = []
    for filename in ("pool.json", "config.json", "result.json"):
        try:
            identity_timestamps.append((job_dir / filename).stat().st_mtime_ns)
        except OSError:
            continue
    return max(identity_timestamps, default=0), job_dir.name


def _pool_release_label(job_dir: Path) -> str:
    """Return a compact release label without trusting the job name."""

    config = _read_json(Path(job_dir) / "config.json") or {}
    datasets = config.get("datasets")
    source = ""
    if isinstance(datasets, list) and datasets and isinstance(datasets[0], Mapping):
        source = str(datasets[0].get("name") or "")
    known_releases = {
        "terminal-bench/terminal-bench": "Terminal-Bench 4.0",
        "terminal-bench/terminal-bench-2-1": "Terminal-Bench 2.1",
        "terminal-bench/terminal-bench-2": "Terminal-Bench 2.0",
        "terminal-bench-science/terminal-bench-science": "Terminal-Bench Science",
    }
    return known_releases.get(source, source or "Terminal-Bench")


def _pool_display_name(job_dir: Path, marker: Mapping[str, Any]) -> str:
    """Return a monitor-only alias bound to one immutable pool contract."""

    payload = _read_json(
        Path(job_dir) / POOL_DISPLAY_FILENAME,
        max_bytes=MAX_POOL_ARTIFACT_BYTES,
    )
    if payload is None:
        return Path(job_dir).name
    display_name = payload.get("display_job_name")
    if (
        payload.get("schema") != POOL_DISPLAY_SCHEMA
        or payload.get("canonical_job_name") != Path(job_dir).name
        or payload.get("config_contract_hash") != marker.get("config_contract_hash")
        or not isinstance(display_name, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}", display_name) is None
    ):
        return Path(job_dir).name
    return display_name


def _job_selector_label(job_dir: Path) -> str:
    marker = _pool_marker(job_dir)
    if marker is None:
        return job_dir.name
    task_count = _as_int(marker.get("task_count"))
    task_selection = _pool_task_selection(job_dir, marker)
    if task_selection is not None:
        task_count -= len(task_selection["excluded_task_names"])
    cell_count = _as_int(marker.get("cell_count"))
    attempts = _as_int(marker.get("n_attempts"))
    if _pool_attempt_selection(job_dir, marker) is not None:
        attempts = 1
    details: list[str] = []
    if task_count > 0:
        details.append(f"{task_count:,} tasks")
    if attempts > 0:
        details.append(f"{attempts} attempt{'s' if attempts != 1 else ''}/task")
    elif cell_count > 0 and task_count > 0 and cell_count % task_count == 0:
        inferred_attempts = cell_count // task_count
        details.append(
            f"{inferred_attempts} attempt{'s' if inferred_attempts != 1 else ''}/task"
        )
    suffix = " · ".join(details)
    if suffix:
        suffix = f" · {suffix}"
    return (
        f"{_pool_release_label(job_dir)}{suffix} · "
        f"{_pool_display_name(job_dir, marker)}"
    )


def _logical_pool_members(jobs_root: Path, canonical_job: Path) -> list[dict[str, Any]]:
    """Resolve explicitly linked retry jobs for one canonical pool.

    The sidecar is deliberately monitor-only: it gives a live, auditable union
    while an already-running Harbor retry job owns its directory.  It never
    rewrites pool cells or promotes a retry result into the official result.
    """

    payload = _read_json(
        Path(canonical_job) / LOGICAL_POOL_FILENAME,
        max_bytes=MAX_POOL_ARTIFACT_BYTES,
    )
    if (
        payload is None
        or payload.get("schema") != LOGICAL_POOL_SCHEMA
        or payload.get("canonical_job") != Path(canonical_job).name
    ):
        return []
    raw_members = payload.get("members")
    if not isinstance(raw_members, list):
        return []
    try:
        root = Path(jobs_root).expanduser().resolve(strict=True)
        canonical = _assert_within(canonical_job, root, kind="canonical pool")
    except (OSError, MonitorArtifactError):
        return []
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_members:
        if not isinstance(raw, Mapping):
            continue
        name = raw.get("job_name")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}", name) is None
            or name in seen
        ):
            continue
        try:
            path = _assert_within(root / name, root, kind="logical pool member")
        except MonitorArtifactError:
            continue
        if path == canonical or not path.is_dir() or path.is_symlink():
            continue
        if not ((path / "config.json").is_file() or (path / "result.json").is_file()):
            continue
        seen.add(name)
        members.append(
            {
                "job_name": name,
                "path": path,
                "role": str(raw.get("role") or "retry"),
                "harness": str(raw.get("harness") or "linked harness"),
                "priority": _as_int(raw.get("priority")),
            }
        )
    return sorted(members, key=lambda row: (row["priority"], row["job_name"]))


def _visible_jobs(jobs_root: Path, jobs: list[Path]) -> list[Path]:
    """Apply monitor-only visibility, then collapse linked retry members."""

    eligible: list[Path] = []
    for job in jobs:
        marker = _pool_marker(job)
        display = _read_json(job / POOL_DISPLAY_FILENAME, max_bytes=MAX_POOL_ARTIFACT_BYTES)
        if (marker is not None and display is not None
            and display.get("schema") == POOL_DISPLAY_SCHEMA
            and display.get("canonical_job_name") == job.name
            and marker.get("config_contract_hash")
            and display.get("config_contract_hash") == marker["config_contract_hash"]
            and display.get("visible") is False):
            continue
        eligible.append(job)
    hidden: set[Path] = set()
    for job in eligible:
        if _pool_marker(job) is None:
            continue
        hidden.update(
            Path(member["path"]) for member in _logical_pool_members(jobs_root, job)
        )
    visible = [job for job in eligible if job not in hidden]
    # Circular logical links may fall back, but explicit hides never reappear.
    return visible or eligible


def _pool_marker(job_dir: Path) -> Mapping[str, Any] | None:
    """Return a validated distributed-pool marker, if this is a pool job."""

    marker = _read_json(Path(job_dir) / "pool.json", max_bytes=MAX_POOL_ARTIFACT_BYTES)
    if marker is None or marker.get("schema") != POOL_SCHEMA:
        return None
    return marker


def _pool_cells(marker: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_cells = marker.get("cells")
    if not isinstance(raw_cells, list):
        return []
    cells: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            continue
        cell_id = raw.get("cell_id")
        task_name = raw.get("task_name")
        if (
            not isinstance(cell_id, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", cell_id)
            or cell_id in seen
            or not isinstance(task_name, str)
            or not task_name
        ):
            continue
        seen.add(cell_id)
        attempt = raw.get("official_attempt_index", raw.get("official_attempt"))
        cells.append(
            {
                **dict(raw),
                "cell_id": cell_id,
                "task_name": task_name,
                "official_attempt_index": _as_int(attempt),
            }
        )
    return cells


def _pool_contained_path(root: Path, value: Any, *, kind: str) -> Path | None:
    if not isinstance(value, (str, os.PathLike)):
        return None
    rendered = os.fspath(value)
    if not isinstance(rendered, str) or not rendered or "\x00" in rendered:
        return None
    raw = Path(rendered).expanduser()
    candidate = raw if raw.is_absolute() else Path(root) / raw
    try:
        return _assert_within(candidate, root, kind=kind)
    except MonitorArtifactError:
        return None


def _canonical_digest(value: Any) -> str | None:
    try:
        payload = json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    except (TypeError, UnicodeError, ValueError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _regular_file_sha256(path: Path, *, max_bytes: int) -> str | None:
    path = Path(path)
    if _has_symlink_component(path):
        return None
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            return None
        payload = path.read_bytes()
    except OSError:
        return None
    if len(payload) > max_bytes or len(payload) != info.st_size:
        return None
    return hashlib.sha256(payload).hexdigest()


def _pool_score_validator_signature() -> tuple[tuple[int, int], ...]:
    paths = (
        PROJECT_ROOT / "terminal_bench_eval" / "terminal_bench_pool.py",
        PROJECT_ROOT / "terminal_bench_eval" / "terminal_bench_monitor_score_probe.py",
    )
    signature: list[tuple[int, int]] = []
    for path in paths:
        try:
            info = path.stat()
        except OSError:
            return ()
        if not stat.S_ISREG(info.st_mode) or _has_symlink_component(path):
            return ()
        signature.append((info.st_mtime_ns, info.st_size))
    return tuple(signature)


def _pool_authoritative_reconciled_reward(
    job_dir: Path,
    cell_root: Path,
    result: Mapping[str, Any],
) -> float | None:
    """Ask the Harbor runtime to attest a reconciliation or verified timeout.

    The monitor's Python environment intentionally has no Harbor dependency.
    A bounded read-only subprocess therefore calls the exact pool reward gate
    and the durable reconciliation-receipt validator.  Any missing dependency,
    timeout, digest race, malformed output, or validation error fails closed.
    """

    reconciled = isinstance(result.get("attribution_reconciliation"), Mapping)
    timeout_zero = isinstance(result.get("agent_timeout_policy"), Mapping)
    verified_timeout = (
        isinstance(result.get("verified_agent_timeout_warning"), Mapping)
        and _as_mapping(result.get("exception_info")).get("exception_type")
        == "AgentTimeoutError"
    )
    if not reconciled and not verified_timeout and not timeout_zero:
        return None
    expected_authority = (
        POOL_SCORE_AUTHORITY if reconciled else POOL_AGENT_TIMEOUT_SCORE_AUTHORITY
    )
    if timeout_zero:
        from terminal_bench_eval.agent_timeout_policy import AUTHORITY
        expected_authority = AUTHORITY
    result_sha256 = _regular_file_sha256(
        Path(cell_root) / "result.json", max_bytes=MAX_POOL_ARTIFACT_BYTES
    )
    validator_signature = _pool_score_validator_signature()
    cell_id = str(result.get("cell_id") or "")
    if result_sha256 is None or not validator_signature or not cell_id:
        return None

    python = POOL_SCORE_PROBE_PYTHON.absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        return None
    command = [
        str(python),
        "-m",
        "terminal_bench_eval.terminal_bench_monitor_score_probe",
        "--job-name",
        Path(job_dir).name,
        "--jobs-root",
        str(Path(job_dir).parent),
        "--cell-id",
        cell_id,
        "--expected-result-sha256",
        result_sha256,
    ]
    probe_env = dict(os.environ)
    probe_env["PYTHONNOUSERSITE"] = "1"
    probe_env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=probe_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=POOL_SCORE_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if (
        completed.returncode != 0
        or not completed.stdout
        or len(completed.stdout) > 64 * 1024
    ):
        return None
    try:
        report = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(report, Mapping):
        return None
    reward = _as_finite_number(report.get("reward"))
    projected_reward = _as_finite_number(
        _as_mapping(result.get("rewards")).get("reward")
    )
    exception = _as_mapping(result.get("exception_info"))
    if (
        report.get("schema") != POOL_SCORE_PROBE_SCHEMA
        or report.get("authority") != expected_authority
        or report.get("job_name") != Path(job_dir).name
        or report.get("cell_id") != cell_id
        or report.get("task_name") != result.get("task_name")
        or report.get("official_attempt_index") != result.get("official_attempt_index")
        or report.get("result_sha256") != result_sha256
        or report.get("exception_type") != str(exception.get("exception_type") or "")
        or (timeout_zero and report.get("agent_timeout_policy_sha256")
            != _canonical_digest(result.get("agent_timeout_policy")))
        or report.get("verified_agent_exit_warning_sha256")
        != _canonical_digest(result.get("verified_agent_exit_warning"))
        or report.get("attribution_reconciliation_sha256")
        != _canonical_digest(result.get("attribution_reconciliation"))
        or (
            not reconciled
            and report.get("verified_agent_timeout_warning_sha256")
            != _canonical_digest(result.get("verified_agent_timeout_warning"))
        )
        or reward is None
        or projected_reward is None
        or float(reward) != float(projected_reward)
    ):
        return None
    return float(reward)


def _pool_reward(result: Mapping[str, Any]) -> float | int | None:
    if result.get("schema") == POOL_CELL_RESULT_SCHEMA:
        rewards = _as_mapping(result.get("rewards"))
        reward = _as_finite_number(rewards.get("reward"))
        if reward is None:
            return None
        expected_outcome = "passed" if float(reward) > 0 else "failed"
        exception = result.get("exception_info")
        has_exception = exception is not None and (
            not isinstance(exception, Mapping) or bool(exception)
        )
        rejected = result.get("rejected_verifier_rewards")
        has_rejected = rejected is not None and (
            not isinstance(rejected, Mapping) or bool(rejected)
        )
        if (
            result.get("status") != "completed"
            or result.get("outcome") != expected_outcome
            or has_rejected
            or has_exception
        ):
            return None
        return reward

    rewards = result.get("rewards")
    if not isinstance(rewards, Mapping):
        rewards = _as_mapping(_as_mapping(result.get("verifier_result")).get("rewards"))
    reward = _as_finite_number(rewards.get("reward"))
    return reward if reward is not None else _as_finite_number(result.get("reward"))


def _pool_cleanup_reward_candidate(
    result: Mapping[str, Any],
) -> float | int | None:
    """Return a cleanup-only candidate; formal pool state must still attest it."""

    if result.get("schema") != POOL_CELL_RESULT_SCHEMA:
        return None
    reward = _as_finite_number(_as_mapping(result.get("rewards")).get("reward"))
    warning = _as_mapping(result.get("infrastructure_cleanup_warning"))
    exception = _as_mapping(result.get("exception_info"))
    rejected = result.get("rejected_verifier_rewards")
    has_rejected = rejected is not None and (
        not isinstance(rejected, Mapping) or bool(rejected)
    )
    evidence = warning.get("evidence")
    if (
        reward is None
        or result.get("status") != "completed"
        or result.get("outcome") != ("passed" if float(reward) > 0 else "failed")
        or has_rejected
        or not exception
        or warning.get("policy") != "preserve_verified_reward_without_resampling_v1"
        or not isinstance(evidence, list)
        or not evidence
    ):
        return None
    return reward


def _pool_attempt_selection(
    job_dir: Path, marker: Mapping[str, Any]
) -> dict[str, Any] | None:
    selection = _read_json(
        Path(job_dir) / "attempt-selection.json", max_bytes=1024 * 1024
    )
    if selection is None:
        return None
    keep = selection.get("keep_official_attempt_index")
    configured = _as_int(marker.get("n_attempts"), -1)
    if (
        selection.get("schema") != ATTEMPT_SELECTION_SCHEMA
        or selection.get("policy") != QUEUED_ATTEMPT_RETIREMENT_POLICY
        or selection.get("job_name") != Path(job_dir).name
        or selection.get("config_contract_hash") != marker.get("config_contract_hash")
        or selection.get("selection_sha256") != marker.get("selection_sha256")
        or keep != 1
        or configured < 1
        or selection.get("excluded_official_attempt_indices")
        != list(range(2, configured + 1))
        or _parse_time(selection.get("activated_at")) is None
    ):
        return None
    return selection


def _apply_pool_attempt_selection(
    rows: list[dict[str, Any]], selection: Mapping[str, Any] | None
) -> None:
    if selection is None:
        return
    keep = _as_int(selection.get("keep_official_attempt_index"), -1)
    for row in rows:
        if _as_int(row.get("official_attempt"), -2) == keep:
            continue
        row["score_state"] = "excluded"
        row["unscoreable_reason"] = ""
        row["unscoreable_reason_code"] = ""
        row["unscoreable_evidence"] = "attempt-selection.json"


def _pool_task_selection(
    job_dir: Path, marker: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a strictly bound additive task-level scoring selection."""

    selection = _read_json(Path(job_dir) / "task-selection.json", max_bytes=1024 * 1024)
    if selection is None:
        return None
    excluded = selection.get("excluded_task_names")
    known_tasks = {cell["task_name"] for cell in _pool_cells(marker)}
    reason = selection.get("reason")
    if (
        selection.get("schema") != TASK_SELECTION_SCHEMA
        or selection.get("job_name") != Path(job_dir).name
        or selection.get("config_contract_hash") != marker.get("config_contract_hash")
        or selection.get("selection_sha256") != marker.get("selection_sha256")
        or selection.get("policy") != QUEUED_TASK_RETIREMENT_POLICY
        or not isinstance(excluded, list)
        or not excluded
        or any(not isinstance(name, str) or not name for name in excluded)
        or excluded != sorted(set(excluded))
        or not set(excluded) < known_tasks
        or not isinstance(reason, str)
        or not reason.strip()
        or "\n" in reason
        or "\r" in reason
        or len(reason) > 512
        or selection.get("accuracy_denominator") != "score_eligible_cells_only"
        or selection.get("immutable_pool_marker_changed") is not False
        or selection.get("immutable_pool_config_changed") is not False
        or _parse_time(selection.get("activated_at")) is None
    ):
        return None
    return selection


def _apply_pool_task_selection(
    rows: list[dict[str, Any]], selection: Mapping[str, Any] | None
) -> None:
    if selection is None:
        return
    excluded = set(selection.get("excluded_task_names") or ())
    for row in rows:
        if row.get("task") not in excluded:
            continue
        row["score_state"] = "excluded"
        row["unscoreable_reason"] = ""
        row["unscoreable_reason_code"] = ""
        row["unscoreable_evidence"] = "task-selection.json"


def _reconcile_formal_pool_state(
    rows: list[dict[str, Any]],
    *,
    state: Mapping[str, Any],
    marker: Mapping[str, Any],
    job_dir: Path,
) -> bool:
    """Validate aggregate identity and attest any cleanup-only row rewards.

    The monitor cannot independently replay Harbor's cleanup attribution gate.
    A cleanup candidate is therefore promoted only when the backend-authored
    state exactly accounts for every current terminal row and for the combined
    candidate count and mean. Any ambiguity remains conservatively unscoreable.
    """

    total = len(rows)
    terminal = [row for row in rows if row.get("terminal") is True]
    excluded = [row for row in rows if row.get("score_state") == "excluded"]
    eligible = [row for row in rows if row.get("score_state") != "excluded"]
    eligible_terminal = [
        row for row in terminal if row.get("score_state") != "excluded"
    ]
    direct_rewards = [
        float(row["reward"])
        for row in terminal
        if row.get("score_state") == "scored"
        and _as_finite_number(row.get("reward")) is not None
    ]
    candidates = [
        row
        for row in terminal
        if row.get("score_state") != "excluded"
        if _as_finite_number(row.get("cleanup_reward_candidate")) is not None
    ]
    candidate_rewards = [float(row["cleanup_reward_candidate"]) for row in candidates]
    expected_rewards = [*direct_rewards, *candidate_rewards]
    expected_mean = (
        math.fsum(expected_rewards) / len(expected_rewards)
        if expected_rewards
        else None
    )
    state_total = _as_int(state.get("cell_count"), -1)
    state_terminal = _as_int(state.get("terminal_count"), -1)
    state_scored = _as_int(state.get("observed_reward_count"), -1)
    state_artifact_errors = _as_int(state.get("artifact_error_count"), -1)
    state_eligible = _as_int(state.get("score_eligible_cell_count"), total)
    state_eligible_terminal = _as_int(
        state.get("score_eligible_terminal_count"), len(terminal)
    )
    state_excluded = _as_int(state.get("excluded_cell_count"), 0)
    observed_raw = state.get("observed_mean_reward")
    all_slot_raw = state.get("all_slot_mean_reward")
    observed = _as_finite_number(observed_raw)
    all_slot = _as_finite_number(all_slot_raw)
    marker_contract = marker.get("config_contract_hash")
    if (
        state.get("schema") != POOL_STATE_SCHEMA
        or state.get("job_name") != job_dir.name
        or not isinstance(marker_contract, str)
        or not marker_contract
        or state.get("config_contract_hash") != marker_contract
        or state_total != total
        or state_terminal != len(terminal)
        or state_eligible != len(eligible)
        or state_eligible_terminal != len(eligible_terminal)
        or state_excluded != len(excluded)
        or state_eligible + state_excluded != state_total
        or state_scored != len(expected_rewards)
        or state_artifact_errors
        != sum(row.get("status") == "artifact error" for row in rows)
        or not (0 <= state_scored <= state_eligible_terminal <= state_eligible)
        or not (state_eligible_terminal <= state_terminal <= state_total)
        or not (0 <= state_artifact_errors <= state_total - state_terminal)
    ):
        return False
    if state_scored == 0:
        if observed_raw is not None or expected_mean is not None:
            return False
    elif (
        observed is None
        or expected_mean is None
        or not math.isclose(float(observed), expected_mean, rel_tol=0.0, abs_tol=1e-12)
    ):
        return False
    if state_scored < state_eligible:
        if all_slot_raw is not None:
            return False
    elif state_eligible > 0 and (
        all_slot is None
        or observed is None
        or not math.isclose(
            float(all_slot), float(observed), rel_tol=0.0, abs_tol=1e-12
        )
    ):
        return False
    elif state_eligible == 0 and all_slot_raw is not None:
        return False

    for row in candidates:
        row["reward"] = row["cleanup_reward_candidate"]
        row["score_state"] = "scored"
        row["score_evidence"] = "formal state + cleanup attribution receipt"
        row["unscoreable_reason"] = ""
        row["unscoreable_reason_code"] = ""
        row["unscoreable_evidence"] = "formal state + cleanup attribution receipt"
    return True


def _pool_exception_record(
    cell_root: Path, result: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return only the worker exception belonging to this terminal result."""

    record = _read_json(
        Path(cell_root) / "exception.json", max_bytes=MAX_POOL_ARTIFACT_BYTES
    )
    if (
        record is None
        or record.get("schema") != POOL_WORKER_EXCEPTION_SCHEMA
        or record.get("cell_id") != result.get("cell_id")
        or record.get("task_name") != result.get("task_name")
        or _as_int(record.get("official_attempt_index"), -1)
        != _as_int(result.get("official_attempt_index"), -2)
        or _as_int(record.get("execution_attempt"), -1)
        != _as_int(result.get("execution_attempt"), -2)
    ):
        return {}
    return record


def _pool_exception_signature(value: Mapping[str, Any]) -> tuple[str, str]:
    nested = _as_mapping(value.get("exception_info"))
    exception_type = str(
        nested.get("exception_type")
        or nested.get("type")
        or value.get("exception_type")
        or ""
    )
    fields = (
        nested.get("exception_message"),
        nested.get("message"),
        nested.get("exception_traceback"),
        nested.get("traceback"),
        value.get("message"),
        value.get("traceback"),
    )
    # The actionable exception is at the tail of Harbor/Codex payloads.  A
    # bounded projection also avoids scanning a model's full terminal output.
    evidence = "\n".join(str(field)[-65_536:] for field in fields if field)
    return exception_type, evidence


def _is_registry_connection_failure(evidence: str) -> bool:
    lowered = evidence.lower()
    return "connecterror" in lowered and any(
        marker in lowered
        for marker in (
            "registrydb",
            "resolve_task_version",
            "download_tasks",
            "harbor/tasks/client.py",
        )
    )


def _is_gateway_thread_mismatch(evidence: str) -> bool:
    lowered = evidence.lower()
    return "gateway received a different codex thread" in lowered and (
        "409 conflict" in lowered or "/v1/responses" in lowered
    )


def _pool_execution_log_signature(cell_root: Path, result: Mapping[str, Any]) -> str:
    """Read a bounded log only from the execution committed by the result."""

    execution_attempt = _as_int(result.get("execution_attempt"))
    if execution_attempt < 1:
        return ""
    attempt_root = Path(cell_root) / "attempts" / f"execution-{execution_attempt:04d}"
    run = _read_json(attempt_root / "run.json", max_bytes=MAX_POOL_ARTIFACT_BYTES)
    if (
        run is None
        or run.get("schema") != POOL_EXECUTION_SCHEMA
        or run.get("cell_id") != result.get("cell_id")
        or _as_int(run.get("execution_attempt"), -1) != execution_attempt
        or run.get("status") not in {None, "exited"}
        or _as_int(run.get("return_code")) == 0
    ):
        return ""
    return _bounded_text(attempt_root / "harbor.log", max_chars=262_144, tail=True)


def _pool_score_projection(
    cell_root: Path,
    result: Mapping[str, Any],
    *,
    reward: float | int | None,
) -> dict[str, str]:
    """Classify a terminal reward without turning infrastructure loss into zero."""

    if reward is not None:
        return {
            "score_state": "scored",
            "unscoreable_reason": "",
            "unscoreable_reason_code": "",
            "unscoreable_evidence": "",
        }

    if (
        result.get("status") == "completed"
        and result.get("outcome") == "retired"
        and result.get("rewards") == {}
    ):
        return {
            "score_state": "excluded",
            "unscoreable_reason": "",
            "unscoreable_reason_code": "",
            "unscoreable_evidence": "attempt-selection.json",
        }

    result_type, result_evidence = _pool_exception_signature(result)
    worker = _pool_exception_record(cell_root, result)
    worker_type, worker_evidence = _pool_exception_signature(worker)

    if result_type == "VerifierTimeoutError" or worker_type == "VerifierTimeoutError":
        return {
            "score_state": "unscoreable",
            "unscoreable_reason": "verifier timeout",
            "unscoreable_reason_code": "verifier_timeout",
            "unscoreable_evidence": (
                "result.exception_info"
                if result_type == "VerifierTimeoutError"
                else "exception.json"
            ),
        }

    if result_type == "NetworkConnectionError" and _is_gateway_thread_mismatch(
        result_evidence
    ):
        return {
            "score_state": "unscoreable",
            "unscoreable_reason": "Codex gateway thread mismatch",
            "unscoreable_reason_code": "gateway_thread_mismatch",
            "unscoreable_evidence": "result.exception_info",
        }

    if _is_registry_connection_failure(worker_evidence):
        return {
            "score_state": "unscoreable",
            "unscoreable_reason": "Harbor registry connection failure",
            "unscoreable_reason_code": "registry_connect_error",
            "unscoreable_evidence": "exception.json",
        }

    execution_log = ""
    if result.get("status") == "pool_error":
        execution_log = _pool_execution_log_signature(cell_root, result)
    if _is_registry_connection_failure(execution_log):
        return {
            "score_state": "unscoreable",
            "unscoreable_reason": "Harbor registry connection failure",
            "unscoreable_reason_code": "registry_connect_error",
            "unscoreable_evidence": "harbor.log",
        }

    rejected = result.get("rejected_verifier_rewards")
    outcome = str(result.get("outcome") or "")
    if isinstance(rejected, Mapping) and rejected:
        reason = "execution error; verifier reward rejected"
        reason_code = "verifier_reward_rejected"
    elif outcome.endswith("_infrastructure_retry_exhausted"):
        reason = "infrastructure retries exhausted"
        reason_code = "infrastructure_retry_exhausted"
    elif result.get("status") == "pool_error":
        reason = "pool / Harbor execution error"
        reason_code = "pool_harbor_execution_error"
    elif outcome == "completed_without_reward":
        reason = "no accepted primary reward"
        reason_code = "no_accepted_primary_reward"
    elif outcome == "errored" or result_type or worker_type:
        reason = "execution error"
        reason_code = "execution_error"
    else:
        reason = "unknown unscoreable terminal result"
        reason_code = "unknown_unscoreable"
    return {
        "score_state": "unscoreable",
        "unscoreable_reason": reason,
        "unscoreable_reason_code": reason_code,
        "unscoreable_evidence": (
            "result.exception_info"
            if result_type
            else "exception.json" if worker_type else "result.json"
        ),
    }


def _pool_execution_time_budget(
    job_dir: Path,
    cell_root: Path,
    execution_attempt: int,
    *,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    recorded = _as_mapping(_as_mapping(result).get("agent_time_budget"))
    if execution_attempt < 1:
        return dict(recorded)
    attempt = cell_root / "attempts" / f"execution-{execution_attempt:04d}"
    receipt = _read_json(
        attempt / AGENT_TIME_BUDGET_RECEIPT_NAME,
        max_bytes=MAX_POOL_ARTIFACT_BYTES,
    )
    if (
        receipt is not None
        and receipt.get("schema") == AGENT_TIME_BUDGET_RECEIPT_SCHEMA
        and receipt.get("policy")
        in {
            UNBOUNDED_AGENT_TIMEOUT_POLICY,
            TB21_OFFICIAL_AGENT_TIMEOUT_POLICY,
            TB21_LEGACY_8H_AGENT_TIMEOUT_POLICY,
            TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
        }
    ):
        # The per-execution receipt is written by the fresh Harbor child.
        # Long-lived pool workers may still project their older default after
        # a policy update; that must not overwrite the budget actually used.
        return dict(receipt)
    if recorded:
        return dict(recorded)
    run = _read_json(attempt / "run.json", max_bytes=MAX_POOL_ARTIFACT_BYTES) or {}
    run_budget = _as_mapping(run.get("agent_time_budget"))
    if run_budget:
        return dict(run_budget)
    command = run.get("command")
    if isinstance(command, list) and any(
        str(part) == "terminal_bench_eval.harbor_unbounded_agent" for part in command
    ):
        policy = _safe_job_contract(job_dir).get("time_budget_policy")
        if policy == TB21_UNIFORM_AGENT_TIMEOUT_POLICY:
            return {
                "policy": TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
                "effective_total_agent_timeout_seconds": (
                    TB21_PER_TASK_AGENT_TIMEOUT_SECONDS
                ),
                "effective_timeout_source": TB21_UNIFORM_AGENT_TIMEOUT_SOURCE,
            }
        if policy == TB21_OFFICIAL_AGENT_TIMEOUT_POLICY:
            return {
                "policy": TB21_OFFICIAL_AGENT_TIMEOUT_POLICY,
                "effective_total_agent_timeout_seconds": None,
                "effective_timeout_source": "task.config.agent.timeout_sec",
            }
        return {
            "policy": UNBOUNDED_AGENT_TIMEOUT_POLICY,
            "effective_total_agent_timeout_seconds": None,
            "effective_timeout_source": "wrapper_command",
        }
    configured = _safe_job_contract(job_dir).get("configured_timeout_seconds")
    return {
        "policy": (
            "harbor_configured_timeout"
            if configured is not None
            else "harbor_task_timeout_fallback"
        ),
        "configured_harbor_timeout_seconds": configured,
        "effective_total_agent_timeout_seconds": configured,
        "effective_timeout_source": "legacy_harbor_command",
    }


def _execution_time_budget_label(budget: Mapping[str, Any]) -> str:
    if budget.get("policy") == UNBOUNDED_AGENT_TIMEOUT_POLICY:
        return "unlimited"
    seconds = _as_number(budget.get("effective_total_agent_timeout_seconds"))
    if seconds is not None:
        return _format_duration(float(seconds))
    if budget.get("policy") in {
        "harbor_task_timeout_fallback",
        TB21_OFFICIAL_AGENT_TIMEOUT_POLICY,
    }:
        return "task default"
    return "—"


def _pool_cell_projection(
    job_dir: Path,
    cell: Mapping[str, Any],
    *,
    now: datetime | None = None,
    expected_contract_hash: str | None = None,
) -> dict[str, Any]:
    """Project one compact cell without reading its large Harbor artifacts."""

    now = now or datetime.now(UTC)
    cell_id = str(cell["cell_id"])
    cell_root = Path(job_dir) / "cells" / cell_id
    expected_task = str(cell["task_name"])
    expected_attempt = _as_int(cell.get("official_attempt_index"))
    base = {
        "cell_id": cell_id,
        "task": expected_task,
        "task_ref": str(cell.get("task_ref") or cell.get("ref") or ""),
        "source": str(cell.get("source") or ""),
        "official_attempt": expected_attempt,
        "cell_root": cell_root,
        "terminal": False,
        "status": "queued",
        "outcome": "queued",
        "reward": None,
        "cleanup_reward_candidate": None,
        "score_evidence": "",
        "score_state": "pending",
        "unscoreable_reason": "",
        "unscoreable_reason_code": "",
        "unscoreable_evidence": "",
        "worker": "",
        "device": "",
        "heartbeat_at": None,
        "lease_expires_at": None,
        "execution_attempt": 0,
        "harbor_job_path": None,
        "trial_path": None,
        "exception": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "started_at": None,
        "finished_at": None,
        "agent_time_budget": {},
    }
    if cell_root.is_symlink() or not cell_root.is_dir():
        return {**base, "status": "artifact error", "outcome": "missing cell"}
    try:
        _assert_within(cell_root, job_dir, kind="pool cell directory")
    except MonitorArtifactError:
        return {**base, "status": "artifact error", "outcome": "unsafe cell"}

    result_path = cell_root / "result.json"
    result = _read_json(result_path, max_bytes=MAX_POOL_ARTIFACT_BYTES)
    if result is not None:
        observed_task = result.get("task_name")
        observed_attempt = result.get("official_attempt_index")
        if (
            result.get("schema") != POOL_CELL_RESULT_SCHEMA
            or result.get("cell_id") != cell_id
            or observed_task != expected_task
            or result.get("task_ref") != cell.get("task_ref")
            or _as_int(observed_attempt, -1) != expected_attempt
            or (
                expected_contract_hash
                and result.get("config_contract_hash") != expected_contract_hash
            )
            or result.get("status") not in {"completed", "pool_error"}
        ):
            return {
                **base,
                "status": "artifact error",
                "outcome": "terminal contract mismatch",
            }
        tokens = _as_mapping(result.get("tokens"))
        exception = _as_mapping(result.get("exception_info"))
        exception_label = str(
            result.get("exception_type")
            or exception.get("exception_type")
            or exception.get("type")
            or ""
        )
        reward = _pool_reward(result)
        score_evidence = "result.json" if reward is not None else ""
        if reward is None:
            reward = _pool_authoritative_reconciled_reward(job_dir, cell_root, result)
            if reward is not None:
                score_evidence = (
                    ("agent timeout zero/no-retry policy + "
                     if isinstance(result.get("agent_timeout_policy"), Mapping)
                     else "verified agent SIGTERM reconciliation receipt + "
                     if isinstance(result.get("attribution_reconciliation"), Mapping)
                     else "verified agent timeout receipt + ")
                    + "live Harbor verifier artifacts"
                )
        score = _pool_score_projection(cell_root, result, reward=reward)
        awaiting_verifier = result.get("outcome") == "verification_pending"
        if awaiting_verifier:
            score = {"score_state": "pending", "unscoreable_reason": "",
                     "unscoreable_reason_code": "", "unscoreable_evidence": ""}
            reward = None
        return {
            **base,
            **score,
            "terminal": not awaiting_verifier,
            "status": "verification_pending" if awaiting_verifier else str(result.get("status") or "completed"),
            "outcome": str(
                result.get("outcome") or result.get("status") or "completed"
            ),
            "reward": reward,
            "cleanup_reward_candidate": _pool_cleanup_reward_candidate(result),
            "score_evidence": score_evidence,
            "worker": str(result.get("worker_id") or ""),
            "device": str(result.get("device") or ""),
            "execution_attempt": _as_int(
                result.get("execution_attempt", result.get("attempt"))
            ),
            "harbor_job_path": result.get("harbor_job_path"),
            "trial_path": result.get("trial_path"),
            "exception": exception_label,
            "input_tokens": _as_int(
                result.get("n_input_tokens"), _as_int(tokens.get("input"))
            ),
            "output_tokens": _as_int(
                result.get("n_output_tokens"), _as_int(tokens.get("output"))
            ),
            "started_at": result.get("started_at"),
            "finished_at": result.get("finished_at") or result.get("committed_at"),
            "agent_time_budget": _pool_execution_time_budget(
                job_dir,
                cell_root,
                _as_int(result.get("execution_attempt", result.get("attempt"))),
                result=result,
            ),
        }
    if result_path.exists():
        return {
            **base,
            "status": "artifact error",
            "outcome": "invalid terminal result artifact",
        }

    dispatch = _read_json(
        cell_root / "dispatch.json", max_bytes=MAX_POOL_ARTIFACT_BYTES
    )
    if dispatch is None:
        return base
    claim = _as_mapping(dispatch.get("claim"))
    status = str(dispatch.get("status") or "queued")
    lease_value = claim.get("lease_expires_at") or dispatch.get("lease_expires_at")
    lease = _parse_time(lease_value)
    if status == "running" and (lease is None or lease <= now):
        status = "stale"
    elif status != "running":
        status = "queued"
    harbor_path = dispatch.get("harbor_job_path") or claim.get("harbor_job_path")
    execution_attempt = _as_int(
        dispatch.get("execution_attempt", dispatch.get("attempt"))
    )
    return {
        **base,
        "status": status,
        "outcome": status,
        "worker": str(claim.get("worker_id") or dispatch.get("worker_id") or ""),
        "device": str(claim.get("device") or dispatch.get("device") or ""),
        "heartbeat_at": claim.get("heartbeat_at") or dispatch.get("heartbeat_at"),
        "lease_expires_at": lease_value,
        "execution_attempt": execution_attempt,
        "harbor_job_path": harbor_path,
        "started_at": claim.get("claimed_at") or dispatch.get("started_at"),
        "agent_time_budget": _pool_execution_time_budget(
            job_dir, cell_root, execution_attempt
        ),
    }


def _pool_worker_projections(job_dir: Path) -> list[dict[str, Any]]:
    workers_dir = Path(job_dir) / "workers"
    if workers_dir.is_symlink() or not workers_dir.is_dir():
        return []
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for path in sorted(workers_dir.glob("*.json")):
        if path.is_symlink():
            continue
        worker = _read_json(path, max_bytes=MAX_POOL_ARTIFACT_BYTES)
        if worker is None:
            continue
        lease = _parse_time(worker.get("lease_expires_at"))
        status = str(worker.get("status") or "unknown")
        if status in {"starting", "running", "idle"} and (
            lease is None or lease <= now
        ):
            status = "stale"
        rows.append(
            {
                "worker": str(worker.get("worker_id") or path.stem),
                "status": status,
                "device": str(worker.get("device") or "—"),
                "cell": str(worker.get("cell_id") or worker.get("sample_id") or "—"),
                "task": str(worker.get("task_name") or "—"),
                "attempt": _as_int(worker.get("official_attempt_index")),
                "execution": _as_int(
                    worker.get("execution_attempt", worker.get("attempt"))
                ),
                "heartbeat": str(worker.get("heartbeat_at") or "—"),
                "lease": str(worker.get("lease_expires_at") or "—"),
            }
        )
    return rows


def _pool_trial_directory(job_dir: Path, row: Mapping[str, Any]) -> Path | None:
    """Resolve a selected cell's Harbor trial lazily and fail closed."""

    cell_id = str(row.get("cell_id") or "")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}", cell_id) is None:
        return None
    expected_cell_root = Path(job_dir) / "cells" / cell_id
    if expected_cell_root.is_symlink() or not expected_cell_root.is_dir():
        return None
    try:
        expected_cell_root = _assert_within(
            expected_cell_root, job_dir, kind="pool cell directory"
        )
    except MonitorArtifactError:
        return None
    expected_harbor_job = expected_cell_root / "harbor-job"
    if expected_harbor_job.is_symlink():
        return None

    trial = _pool_contained_path(
        job_dir, row.get("trial_path"), kind="pool Harbor trial directory"
    )
    if trial is not None and trial.is_dir() and expected_harbor_job.is_dir():
        try:
            if trial.parent == expected_harbor_job.resolve(strict=True):
                return trial
        except OSError:
            return None
    harbor_job = _pool_contained_path(
        job_dir, row.get("harbor_job_path"), kind="pool Harbor job directory"
    )
    if harbor_job is None or not harbor_job.is_dir():
        return None
    try:
        if harbor_job != expected_harbor_job.resolve(strict=True):
            return None
    except OSError:
        return None
    trials = _trial_directories(harbor_job)
    if not trials:
        return None
    terminal = [path for path in trials if _read_trial_result(path) is not None]
    return terminal[-1] if terminal else trials[-1]


def _enrich_pool_cell_projection(
    job_dir: Path,
    row: Mapping[str, Any],
    *,
    canonical_harness: str,
) -> dict[str, Any]:
    """Join one compact pool cell with its bounded Harbor trial projection.

    The pool result/dispatch is authoritative for queue ownership and reward,
    but it intentionally omits live rounds, memory, and harness telemetry.
    Those fields already exist in the cell-owned Harbor trial.  Joining them
    here keeps the monitor read-only while avoiding the old composite-pool
    ``effective_*`` schema leaking into a canonical Codex-only pool.
    """

    enriched = {
        **dict(row),
        "harness": str(row.get("harness") or canonical_harness or "—"),
        "rounds": row.get("rounds"),
        "observed_round": row.get("observed_round"),
        "inflight_round": row.get("inflight_round"),
        "memory_records": row.get("memory_records"),
        "duration": row.get("duration") or "—",
        "duration_seconds": row.get("duration_seconds"),
    }
    trial = _pool_trial_directory(job_dir, enriched)
    if trial is None:
        return enriched
    projection = _trial_projection(trial)
    enriched["monitor_trial_path"] = trial
    for key in (
        "rounds",
        "observed_round",
        "inflight_round",
        "input_tokens",
        "output_tokens",
        "memory_records",
        "duration",
        "duration_seconds",
        "agent",
        "version",
        "is_codex",
    ):
        value = projection.get(key)
        if value is not None and value != "":
            enriched[key] = value
    projected_harness = str(projection.get("harness") or "")
    if projected_harness:
        enriched["harness"] = projected_harness
    return enriched


def _trial_directories(job_dir: Path) -> list[Path]:
    try:
        root = Path(job_dir).resolve(strict=True)
        candidates = list(root.iterdir())
    except OSError:
        return []
    trials: list[Path] = []
    for path in candidates:
        if path.is_symlink() or not path.is_dir() or path.name.startswith("."):
            continue
        if not ((path / "config.json").is_file() or (path / "result.json").is_file()):
            continue
        try:
            trials.append(_assert_within(path, root, kind="trial directory"))
        except MonitorArtifactError:
            continue
    return sorted(trials, key=lambda path: path.name)


def _latest_mtime(paths: Iterable[Path]) -> datetime | None:
    timestamps: list[float] = []
    for path in paths:
        if _has_symlink_component(path):
            continue
        try:
            timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    if not timestamps:
        return None
    return datetime.fromtimestamp(max(timestamps), tz=UTC)


def _artifact_paths(trial: Path) -> list[Path]:
    agent = trial / "agent"
    return [
        trial / "config.json",
        trial / "trial.log",
        trial / "result.json",
        agent / "trajectory.json",
        agent / "codex.txt",
        *[
            agent / CONTROLLER_ARTIFACT_DIRNAME / name
            for name in sorted(CONTROLLER_ARTIFACT_FILENAMES)
        ],
        agent / MEMORY_FILENAME,
        agent / CODEX_ROUND_TRACE_FILENAME,
        agent / CODEX_TOOL_RESPONSE_TRACE_FILENAME,
        agent / GATEWAY_IMAGE_RECEIPTS_FILENAME,
        agent / GATEWAY_WEB_SEARCH_RECEIPTS_FILENAME,
        *[agent / name for name in EVENT_FILENAMES],
        *[agent / name for name in RECEIPT_FILENAMES],
        *[agent / name for name in TERMINAL_PANE_FILENAMES],
    ]


def _event_summary(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    latest_round = 0
    latest_committed_round = 0
    latest_compression: dict[str, Any] = {}
    latest_generation: dict[str, Any] = {}
    response_outcomes: Counter[str] = Counter()
    lookups = lookup_successes = lookup_failures = 0
    latest_lookup: dict[str, Any] = {}
    guard_count = 0
    latest_guard: dict[str, Any] = {}
    validation_recoveries = 0
    latest_validation_recovery: dict[str, Any] = {}
    first_observed_at = _first_jsonl_number(path, "observed_at")
    first_observed_at = (
        float(first_observed_at) if first_observed_at is not None else None
    )
    last_observed_at: float | None = None
    try:
        truncated = path.stat().st_size > MAX_JSONL_BYTES
    except OSError:
        truncated = False
    for event in _complete_jsonl(path, tail_bytes=MAX_JSONL_BYTES):
        name = str(event.get("event") or "unknown")
        counts[name] += 1
        event_round = _as_int(event.get("round_index"))
        latest_round = max(latest_round, event_round)
        observed_at = _as_number(event.get("observed_at"))
        if observed_at is not None:
            observed = float(observed_at)
            if first_observed_at is None:
                first_observed_at = observed
            last_observed_at = (
                observed
                if last_observed_at is None
                else max(last_observed_at, observed)
            )
        payload = _as_mapping(event.get("payload"))
        compression = payload.get("compression")
        if name == "compression_finished" and isinstance(compression, Mapping):
            latest_compression = dict(compression)
        elif name == "terminus_llm_response":
            latest_committed_round = max(
                latest_committed_round,
                event_round,
                _as_int(payload.get("round_index")),
            )
            latest_generation = {
                key: payload.get(key)
                for key in (
                    "round_index",
                    "elapsed_seconds",
                    "token_count",
                    "finished_on_eos",
                    "hit_generation_cap",
                    "finish_reason",
                )
            }
            if isinstance(compression, Mapping):
                latest_compression = dict(compression)
        elif name == "response_committed":
            latest_committed_round = max(latest_committed_round, event_round)
            outcome = str(event.get("outcome") or "unknown")
            response_outcomes[outcome] += 1
            top_level_compression = event.get("compression")
            if isinstance(top_level_compression, Mapping):
                latest_compression = dict(top_level_compression)
            latest_generation = {
                "round_index": event_round,
                "outcome": outcome,
                "call_ids": list(event.get("call_ids") or []),
            }
        elif name == "terminus_memory_lookup":
            lookups += 1
            latest_lookup = dict(payload)
            if payload.get("succeeded") is True:
                lookup_successes += 1
            else:
                lookup_failures += 1
        elif name == "terminus_progress_guard":
            guard_count += 1
            latest_guard = dict(payload)
        elif name == "model_tool_validation_recovered":
            validation_recoveries += 1
            latest_validation_recovery = dict(payload or event)
    return {
        "counts": dict(counts),
        "latest_round": latest_round,
        "latest_committed_round": latest_committed_round,
        "latest_compression": latest_compression,
        "latest_generation": latest_generation,
        "response_outcomes": dict(response_outcomes),
        "lookups": lookups,
        "lookup_successes": lookup_successes,
        "lookup_failures": lookup_failures,
        "latest_lookup": latest_lookup,
        "guard_count": guard_count,
        "latest_guard": latest_guard,
        "validation_recoveries": validation_recoveries,
        "latest_validation_recovery": latest_validation_recovery,
        "first_observed_at": first_observed_at,
        "last_observed_at": last_observed_at,
        "malformed_lines": 0,
        "truncated": truncated,
    }


def _receipt_records(path: Path) -> list[dict[str, Any]]:
    """Normalize canonical JSONL and historical mapping/array receipts.

    The canonical journal can be very large.  Consumers of this function only
    need recent display/reconstruction data, so validate a bounded suffix here;
    cumulative telemetry is handled separately by `_receipt_summary`.
    """

    path = Path(path)
    if path.name == GATEWAY_RECEIPTS_JSONL_FILENAME:
        records: list[dict[str, Any]] = []
        previous_digest: str | None = None
        previous_index: int | None = None
        try:
            for raw in _complete_jsonl(path, tail_bytes=MAX_JSONL_BYTES):
                if previous_index is None:
                    receipt, digest = decode_receipt_record(raw)
                else:
                    receipt, digest = decode_receipt_record(
                        raw,
                        expected_index=previous_index + 1,
                        expected_previous_sha256=previous_digest,
                    )
                previous_index = _as_int(raw.get("record_index"))
                previous_digest = digest
                records.append(receipt)
        except GatewayReceiptPersistenceError:
            return []
        return records

    value = _read_json_value(path)
    raw = value.get("receipts") if isinstance(value, Mapping) else value
    if not isinstance(raw, list):
        return []
    return [dict(record) for record in raw if isinstance(record, Mapping)]


def _receipt_latest_generation(latest: Mapping[str, Any]) -> dict[str, Any]:
    if not latest:
        return {}
    latest_output = _as_number(latest.get("output_tokens"))
    if latest_output is None:
        latest_output = _as_number(latest.get("token_count"))
    return {
        "round_index": _as_int(latest.get("round_index")),
        "elapsed_seconds": latest.get("elapsed_seconds"),
        "input_tokens": latest.get("input_tokens"),
        "token_count": latest_output,
        "finished_on_eos": latest.get("finished_on_eos"),
        "hit_generation_cap": latest.get("hit_generation_cap"),
        "complete_valid": latest.get("complete_valid"),
        "finish_reason": latest.get("finish_reason"),
        "tool_calls": latest.get("tool_calls"),
    }


def _canonical_receipt_sidecar(path: Path) -> dict[str, Any] | None:
    """Read a size-bound atomic journal projection without scanning the journal."""

    summary_path = path.with_name(GATEWAY_RECEIPTS_SUMMARY_FILENAME)
    for attempt in range(3):
        try:
            before = path.stat()
        except OSError:
            return None
        if not stat.S_ISREG(before.st_mode) or _has_symlink_component(path):
            return None
        summary = _read_json(summary_path, use_last_good=False)
        try:
            after = path.stat()
        except OSError:
            return None
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            if attempt < 2:
                time.sleep(0.02)
                continue
            return None
        if not isinstance(summary, Mapping):
            return None
        records = summary.get("records")
        latest_round = summary.get("latest_round")
        source_size = summary.get("source_size_bytes")
        last_digest = summary.get("last_record_sha256")
        latest_generation = summary.get("latest_generation")
        latest_compression = summary.get("latest_compression")
        input_tokens = summary.get("input_tokens")
        output_tokens = summary.get("output_tokens")
        valid_optional_total = lambda value: value is None or (
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
        )
        if (
            summary.get("schema") == GATEWAY_RECEIPT_SUMMARY_SCHEMA
            and summary.get("source_filename") == GATEWAY_RECEIPTS_JSONL_FILENAME
            and isinstance(source_size, int)
            and not isinstance(source_size, bool)
            and source_size == after.st_size
            and isinstance(records, int)
            and not isinstance(records, bool)
            and records >= 0
            and isinstance(latest_round, int)
            and not isinstance(latest_round, bool)
            and latest_round >= 0
            and valid_optional_total(input_tokens)
            and valid_optional_total(output_tokens)
            and (
                (records == 0 and last_digest is None)
                or (
                    records > 0
                    and isinstance(last_digest, str)
                    and re.fullmatch(r"[0-9a-f]{64}", last_digest) is not None
                )
            )
            and isinstance(latest_generation, Mapping)
            and isinstance(latest_compression, Mapping)
        ):
            return {
                "records": records,
                "latest_round": latest_round,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                # The v1 sidecar intentionally does not duplicate the cumulative
                # tool count; the round-trace reducer supplies that independently.
                "tool_calls": None,
                "complete": True,
                "input_tokens_exact": input_tokens is not None,
                "output_tokens_exact": output_tokens is not None,
                "tool_calls_exact": False,
                "latest_generation": dict(latest_generation),
                "latest_compression": dict(latest_compression),
                "source": "canonical_summary",
                "source_size_bytes": source_size,
            }
        return None
    return None


def _canonical_receipt_incremental_summary(path: Path) -> dict[str, Any]:
    """Validate and reduce a canonical journal when its sidecar is unavailable."""

    state = _update_incremental_jsonl_state(path, kind="receipt")
    if state is None:
        return {}
    try:
        size = path.stat().st_size
    except OSError:
        size = _as_int(state.get("observed_size"))
    observed_size = max(size, _as_int(state.get("observed_size")))
    complete = bool(
        not state.get("partial_record")
        and _as_int(state.get("invalid_records")) == 0
        and _as_int(state.get("offset")) == observed_size
    )
    records = _as_int(state.get("receipt_records"))
    latest = _as_mapping(state.get("receipt_latest"))
    all_input = bool(state.get("receipt_all_input_tokens"))
    all_output = bool(state.get("receipt_all_output_tokens"))
    all_tools = bool(state.get("receipt_all_tool_calls"))
    compression = latest.get("compression")
    return {
        "records": records,
        "latest_round": _as_int(state.get("receipt_latest_round")),
        "input_tokens": (
            _as_int(state.get("receipt_input_tokens")) if records and all_input else None
        ),
        "output_tokens": (
            _as_int(state.get("receipt_output_tokens"))
            if records and all_output
            else None
        ),
        "tool_calls": (
            _as_int(state.get("receipt_tool_calls")) if records and all_tools else None
        ),
        "complete": complete,
        "input_tokens_exact": complete and records > 0 and all_input,
        "output_tokens_exact": complete and records > 0 and all_output,
        "tool_calls_exact": complete and records > 0 and all_tools,
        "latest_generation": _receipt_latest_generation(latest),
        "latest_compression": (
            dict(compression) if isinstance(compression, Mapping) else {}
        ),
        "source": "canonical_incremental_scan",
        "source_size_bytes": observed_size,
    }


def _receipt_summary(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.name == GATEWAY_RECEIPTS_JSONL_FILENAME:
        return _canonical_receipt_sidecar(
            path
        ) or _canonical_receipt_incremental_summary(path)

    records = _receipt_records(path)
    latest: Mapping[str, Any] = {}
    latest_round = 0
    input_total = 0
    output_total = 0
    has_input = False
    has_output = False
    tool_calls = 0
    round_indices: set[int] = set()
    for record in records:
        round_index = _as_int(record.get("round_index"))
        if round_index > 0:
            round_indices.add(round_index)
        if round_index >= latest_round:
            latest_round = round_index
            latest = record
        input_tokens = _as_number(record.get("input_tokens"))
        if input_tokens is not None:
            input_total += int(input_tokens)
            has_input = True
        output_tokens = _as_number(record.get("output_tokens"))
        if output_tokens is None:
            output_tokens = _as_number(record.get("token_count"))
        if output_tokens is not None:
            output_total += int(output_tokens)
            has_output = True
        raw_tool_calls = record.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            tool_calls += len(raw_tool_calls)
    latest_generation = _receipt_latest_generation(latest)
    compression = latest.get("compression") if latest else None
    complete = bool(
        records
        and latest_round > 0
        and round_indices == set(range(1, latest_round + 1))
        and has_input
        and has_output
    )
    return {
        "records": len(records),
        "latest_round": latest_round,
        "input_tokens": input_total if has_input else None,
        "output_tokens": output_total if has_output else None,
        "tool_calls": tool_calls if records else None,
        "complete": complete,
        "input_tokens_exact": complete and has_input,
        "output_tokens_exact": complete and has_output,
        "tool_calls_exact": complete,
        "latest_generation": latest_generation,
        "latest_compression": (
            dict(compression) if isinstance(compression, Mapping) else {}
        ),
    }


def _memory_summary(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError:
        return {
            "bytes": 0,
            "records": 0,
            "rounds": 0,
            "max_round": 0,
            "truncated": False,
            "records_exact": True,
        }
    state = _update_incremental_jsonl_state(path, kind="memory")
    if state is None:
        return {
            "bytes": size,
            "records": 0,
            "rounds": 0,
            "max_round": 0,
            "truncated": size > MAX_JSONL_BYTES,
            "records_exact": False,
        }
    observed_size = max(size, _as_int(state.get("observed_size")))
    truncated = observed_size > MAX_JSONL_BYTES
    rounds = state.get("memory_rounds")
    rounds = rounds if isinstance(rounds, set) else set()
    records_exact = (
        not bool(state.get("partial_record"))
        and _as_int(state.get("invalid_records")) == 0
        and _as_int(state.get("offset")) == observed_size
    )
    return {
        "bytes": observed_size,
        "records": _as_int(state.get("memory_records")),
        "rounds": len(rounds),
        "max_round": _as_int(state.get("memory_max_round")),
        "truncated": truncated,
        "records_exact": records_exact,
    }


def _trajectory_payload(path: Path) -> Mapping[str, Any] | None:
    return _read_json(path)


_TASK_PATTERN = re.compile(
    r"(?:^|\n)Task Description:\s*\n(?P<task>.*?)(?=\n\nCurrent terminal (?:state|output):|\Z)",
    flags=re.DOTALL,
)


def _task_description(trajectory: Mapping[str, Any] | None, fallback: str = "") -> str:
    steps = _as_mapping(trajectory).get("steps")
    fallback_messages: list[str] = []
    if isinstance(steps, list):
        for raw in steps:
            if not isinstance(raw, Mapping) or raw.get("source") not in {None, "user"}:
                continue
            message = raw.get("message")
            if not isinstance(message, str):
                continue
            match = _TASK_PATTERN.search(message)
            if match:
                return _truncate(match.group("task").strip(), 64_000)
            stripped = message.strip()
            if (
                stripped
                and not stripped.startswith("<environment_context>")
                and not stripped.startswith("<skills_instructions>")
            ):
                fallback_messages.append(stripped)
    if fallback_messages:
        # Codex ATIF records its benchmark instruction as a plain user turn,
        # after the synthetic environment-context turn.
        return _truncate(fallback_messages[-1], 64_000)
    return fallback


def _codex_task_from_trial_log(path: Path) -> str:
    """Recover the active Codex instruction from Harbor's quoted launch log."""

    text = _bounded_text(path, max_chars=2_000_000)
    for match in re.finditer(
        r"Running command: (?P<command>.*?)(?=\nCommand outputs captured(?:\n|\Z)|\Z)",
        text,
        flags=re.DOTALL,
    ):
        command = match.group("command")
        if "codex exec" not in command or " -- " not in command:
            continue
        try:
            argv = shlex.split(command)
        except ValueError:
            continue
        try:
            separator = len(argv) - 1 - argv[::-1].index("--")
        except ValueError:
            continue
        if separator + 1 < len(argv):
            return _truncate(argv[separator + 1].strip(), 64_000)
    return ""


def _tool_command(function_name: str, arguments: Mapping[str, Any]) -> str:
    for key in ("keystrokes", "cmd", "command"):
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value.rstrip("\r\n")
    if function_name == "write_stdin":
        chars = arguments.get("chars")
        if isinstance(chars, str):
            session = arguments.get("session_id")
            prefix = (
                f"[write_stdin session={session}]"
                if session is not None
                else "[write_stdin]"
            )
            return f"{prefix}\n{chars}".rstrip("\r\n")
    return ""


def _split_think_content(content: str) -> tuple[str, str]:
    """Split Qwen ``<think>`` content without inventing missing visible text."""

    if not content.startswith("<think>"):
        return "", content
    body = content[len("<think>") :]
    closing = body.find("</think>")
    if closing < 0:
        return body.strip("\r\n"), ""
    reasoning = body[:closing].strip("\r\n")
    visible = body[closing + len("</think>") :].lstrip("\r\n")
    return reasoning, visible


def _canonical_json_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _project_tool_responses(raw_responses: Any) -> list[dict[str, Any]]:
    """Keep structured tool responses exact for rendering and source merging."""

    projected: list[dict[str, Any]] = []
    if not isinstance(raw_responses, list):
        return projected
    for raw in raw_responses:
        response = _as_mapping(raw)
        output = response.get("output")
        if not isinstance(output, str):
            continue
        item: dict[str, Any] = {
            "call_id": str(response.get("call_id") or ""),
            "function_name": str(response.get("function_name") or "tool"),
            "output": output,
        }
        if response.get("model_context_exact") is True:
            item["model_context_exact"] = True
        projected.append(item)
    return projected


def _merge_tool_responses(
    current: Any,
    incoming: Any,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[str, str, str], int] = {}
    for response in [
        *_project_tool_responses(current),
        *_project_tool_responses(incoming),
    ]:
        output = response["output"]
        identity = (
            response["call_id"],
            response["function_name"],
            hashlib.sha256(output.encode("utf-8")).hexdigest(),
        )
        previous_index = positions.get(identity)
        if previous_index is not None:
            if (
                response.get("model_context_exact") is True
                and merged[previous_index].get("model_context_exact") is not True
            ):
                merged[previous_index] = response
            continue
        positions[identity] = len(merged)
        merged.append(response)
    return merged


def _tool_response_source_priority(source: Any) -> int:
    """Rank persisted response projections without overstating exactness."""

    return {
        "trajectory": 10,
        "codex_runtime": 10,
        "private_tool_response_trace_v1": 20,
        "memory_archive": 30,
        "private_tool_response_trace_v2": 40,
    }.get(str(source or ""), 0)


def _tool_response_authority(source: Any, responses: Any) -> tuple[int, int]:
    projected = _project_tool_responses(responses)
    return (
        int(any(item.get("model_context_exact") is True for item in projected)),
        _tool_response_source_priority(source),
    )


def _joined_tool_response_output(responses: Any) -> str:
    projected = _project_tool_responses(responses)
    if len(projected) == 1:
        return projected[0]["output"]
    chunks: list[str] = []
    for index, response in enumerate(projected, 1):
        label = response["function_name"] or "tool"
        call_id = response["call_id"]
        suffix = f" call_id={call_id}" if call_id else ""
        chunks.append(f"[{label} response {index}{suffix}]\n{response['output']}")
    return "\n\n".join(chunks)


def _tool_response_trace_records(path: Path) -> list[dict[str, Any]]:
    """Read hash-checked v1 receipts and v2 model-context content by round."""

    by_round: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for raw in _complete_jsonl(path, tail_bytes=MAX_JSONL_BYTES):
        schema = raw.get("schema")
        if schema not in {
            CODEX_TOOL_RESPONSE_TRACE_SCHEMA_V1,
            CODEX_TOOL_RESPONSE_TRACE_SCHEMA,
        }:
            continue
        round_index = _as_int(raw.get("round_index"))
        episode_id = raw.get("episode_id")
        receipt_digest = raw.get("receipt_sha256")
        raw_responses = raw.get("responses")
        if (
            round_index <= 0
            or not isinstance(episode_id, str)
            or not episode_id
            or not isinstance(receipt_digest, str)
            or not isinstance(raw_responses, list)
            or not raw_responses
        ):
            continue
        responses: list[dict[str, Any]] = []
        valid = True
        for raw_response in raw_responses:
            response = _as_mapping(raw_response)
            call_id = response.get("call_id")
            function_name = response.get("function_name")
            output = response.get("output")
            output_digest = response.get("output_sha256")
            if (
                not isinstance(call_id, str)
                or not call_id
                or not isinstance(function_name, str)
                or not function_name
                or not isinstance(output, str)
                or not isinstance(output_digest, str)
                or hashlib.sha256(output.encode("utf-8")).hexdigest() != output_digest
            ):
                valid = False
                break
            responses.append(
                {
                    "call_id": call_id,
                    "function_name": function_name,
                    "output": output,
                    "output_sha256": output_digest,
                }
            )
        body: dict[str, Any] = {
            "schema": schema,
            "episode_id": episode_id,
            "round_index": round_index,
            "responses": responses,
        }
        if schema == CODEX_TOOL_RESPONSE_TRACE_SCHEMA:
            model_context_content = raw.get("model_context_content")
            model_context_digest = raw.get("model_context_content_sha256")
            if (
                not isinstance(model_context_content, str)
                or not isinstance(model_context_digest, str)
                or hashlib.sha256(model_context_content.encode("utf-8")).hexdigest()
                != model_context_digest
            ):
                continue
            body.update(
                {
                    "model_context_content": model_context_content,
                    "model_context_content_sha256": model_context_digest,
                }
            )
        if not valid or _canonical_json_digest(body) != receipt_digest:
            continue
        record = {
            **body,
            "receipt_sha256": receipt_digest,
            "observed_at": raw.get("observed_at"),
        }
        previous = by_round.get(round_index)
        if (
            previous is not None
            and previous.get("schema") == CODEX_TOOL_RESPONSE_TRACE_SCHEMA
            and schema == CODEX_TOOL_RESPONSE_TRACE_SCHEMA_V1
        ):
            continue
        by_round[round_index] = record
    return [by_round[index] for index in sorted(by_round)]


def _tool_response_trace_steps(path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for record in _tool_response_trace_records(path):
        round_index = _as_int(record.get("round_index"))
        schema = record.get("schema")
        if schema == CODEX_TOOL_RESPONSE_TRACE_SCHEMA:
            responses = [
                {
                    "call_id": "",
                    "function_name": "model_context",
                    "output": record["model_context_content"],
                    "model_context_exact": True,
                }
            ]
            response_source = "private_tool_response_trace_v2"
            display_label = f"Round {round_index} (exact private tool response)"
        else:
            responses = _project_tool_responses(record.get("responses"))
            response_source = "private_tool_response_trace_v1"
            display_label = f"Round {round_index} (legacy private tool receipt)"
        observed_at = _as_number(record.get("observed_at"))
        timestamp = ""
        if observed_at is not None:
            try:
                timestamp = datetime.fromtimestamp(
                    float(observed_at), tz=UTC
                ).isoformat()
            except (OverflowError, OSError, ValueError):
                pass
        steps.append(
            {
                "step_id": round_index,
                "round_index": round_index,
                "display_label": display_label,
                "timestamp": timestamp,
                "message": "",
                "reasoning": "",
                "tool_calls": [],
                "tool_responses": responses,
                "tool_response_source": response_source,
                "commands": [],
                # Structured responses are the only rendering source.  Do not
                # create a second, truncated/concatenated copy in step.output.
                "output": "",
                "private_tool_response_trace": True,
            }
        )
    return steps


def _validated_round_trace_record(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate one private Codex round trace before using its telemetry."""

    if raw.get("schema") != CODEX_ROUND_TRACE_SCHEMA:
        return None
    round_index = _as_int(raw.get("round_index"))
    episode_id = raw.get("episode_id")
    assistant_content = raw.get("assistant_content")
    reasoning = raw.get("reasoning")
    visible_text = raw.get("visible_text")
    digest = raw.get("sha256")
    tool_calls = raw.get("tool_calls")
    if (
        round_index <= 0
        or not isinstance(episode_id, str)
        or not episode_id
        or not isinstance(assistant_content, str)
        or not isinstance(reasoning, str)
        or not isinstance(visible_text, str)
        or not isinstance(digest, str)
        or not isinstance(tool_calls, list)
    ):
        return None
    expected = hashlib.sha256(assistant_content.encode("utf-8")).hexdigest()
    if digest != expected:
        return None
    expected_reasoning, expected_visible = _split_think_content(assistant_content)
    if reasoning != expected_reasoning or visible_text != expected_visible:
        return None
    return {
        **dict(raw),
        "round_index": round_index,
        "assistant_content": assistant_content,
        "reasoning": reasoning,
        "visible_text": visible_text,
        "tool_calls": list(tool_calls),
    }


def _round_trace_records(path: Path) -> list[dict[str, Any]]:
    """Read committed private Codex round traces, failing soft per JSONL line."""

    by_round: OrderedDict[int, dict[str, Any]] = OrderedDict()
    for raw in _complete_jsonl(path, tail_bytes=MAX_JSONL_BYTES):
        record = _validated_round_trace_record(raw)
        if record is None:
            continue
        round_index = _as_int(record.get("round_index"))
        by_round[round_index] = record
    return [by_round[index] for index in sorted(by_round)]


def _round_trace_cumulative_summary(path: Path) -> dict[str, Any]:
    """Return exact cumulative counters from an arbitrarily large round trace."""

    state = _update_incremental_jsonl_state(path, kind="round_trace")
    metrics = state.get("round_metrics") if state is not None else None
    if not isinstance(metrics, dict) or not metrics:
        return {
            "records": 0,
            "latest_round": 0,
            "input_tokens": None,
            "output_tokens": None,
            "tool_calls": None,
            "first_observed_at": None,
            "last_observed_at": None,
            "complete": False,
        }

    round_indices = sorted(
        index for index in metrics if isinstance(index, int) and index > 0
    )
    values = [metrics[index] for index in round_indices]
    input_values = [value[0] for value in values]
    output_values = [value[1] for value in values]
    observed_values = [value[3] for value in values if value[3] is not None]
    contiguous = round_indices == list(range(1, round_indices[-1] + 1))
    complete = (
        contiguous
        and not bool(state.get("partial_record"))
        and _as_int(state.get("invalid_records")) == 0
        and _as_int(state.get("offset")) == _as_int(state.get("observed_size"))
        and all(value is not None for value in input_values)
        and all(value is not None for value in output_values)
    )
    return {
        "records": len(round_indices),
        "latest_round": round_indices[-1],
        "input_tokens": (
            sum(int(value) for value in input_values if value is not None)
            if any(value is not None for value in input_values)
            else None
        ),
        "output_tokens": (
            sum(int(value) for value in output_values if value is not None)
            if any(value is not None for value in output_values)
            else None
        ),
        "tool_calls": sum(int(value[2]) for value in values),
        "first_observed_at": min(observed_values) if observed_values else None,
        "last_observed_at": max(observed_values) if observed_values else None,
        "complete": complete,
    }


def _trace_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    if not isinstance(raw_calls, list):
        return projected
    for raw in raw_calls:
        call = _as_mapping(raw)
        function = _as_mapping(call.get("function"))
        name = str(
            call.get("function_name")
            or call.get("name")
            or function.get("name")
            or "tool"
        )
        arguments: Any = call.get("arguments", function.get("arguments"))
        if isinstance(arguments, str):
            try:
                decoded = json.loads(arguments)
            except json.JSONDecodeError:
                decoded = {"raw": arguments}
            arguments = decoded
        projected.append(
            {
                "call_id": str(call.get("call_id") or call.get("id") or ""),
                "function_name": name,
                "arguments": _without_token_ids(_as_mapping(arguments)),
            }
        )
    return projected


def _round_trace_steps(path: Path) -> list[dict[str, Any]]:
    """Project the private gateway artifact into one monitor card per round."""

    steps: list[dict[str, Any]] = []
    for record in _round_trace_records(path):
        round_index = _as_int(record.get("round_index"))
        calls = _trace_tool_calls(record.get("tool_calls"))
        observed_at = _as_number(record.get("observed_at"))
        timestamp = str(record.get("observed_at") or "")
        if observed_at is not None:
            try:
                timestamp = datetime.fromtimestamp(
                    float(observed_at), tz=UTC
                ).isoformat()
            except (OverflowError, OSError, ValueError):
                timestamp = ""
        steps.append(
            {
                "step_id": round_index,
                "round_index": round_index,
                "display_label": f"Round {round_index} (private gateway trace)",
                "timestamp": timestamp,
                "message": _truncate(
                    record.get("visible_text"), MAX_STEP_MESSAGE_CHARS
                ),
                "reasoning": _truncate(
                    record.get("reasoning"), MAX_STEP_REASONING_CHARS
                ),
                "reasoning_source": "private_round_trace",
                "tool_calls": calls,
                "tool_responses": [],
                "commands": [
                    command
                    for call in calls
                    if (
                        command := _tool_command(
                            str(call.get("function_name") or "tool"),
                            _as_mapping(call.get("arguments")),
                        )
                    )
                ],
                "output": "",
                "prompt_tokens": _as_int(record.get("input_tokens")),
                "completion_tokens": _as_int(record.get("output_tokens")),
                "finish_reason": str(record.get("finish_reason") or ""),
                "private_round_trace": True,
            }
        )
    return steps


def _merge_round_steps(
    steps: list[dict[str, Any]],
    round_steps: list[dict[str, Any]],
    *,
    reasoning_only: bool = False,
) -> list[dict[str, Any]]:
    """Merge an authoritative round-indexed source into ATIF/Codex cards."""

    merged = [dict(step) for step in steps]
    positions: dict[int, list[int]] = {}
    for index, step in enumerate(merged):
        round_index = _as_int(step.get("round_index"))
        if round_index > 0:
            positions.setdefault(round_index, []).append(index)
    for round_step in round_steps:
        round_index = _as_int(round_step.get("round_index"))
        if round_index <= 0:
            continue
        indices = positions.get(round_index)
        if not indices:
            if not reasoning_only:
                positions[round_index] = [len(merged)]
                merged.append(dict(round_step))
            continue
        target_index = indices[-1]
        target = dict(merged[target_index])
        reasoning = str(round_step.get("reasoning") or "")
        if reasoning and (not str(target.get("reasoning") or "") or not reasoning_only):
            target["reasoning"] = reasoning
            target["reasoning_source"] = round_step.get("reasoning_source")
            if round_step.get("archived_reasoning"):
                target["archived_reasoning"] = True

        # Response provenance is independent from whether memory.jsonl is being
        # used only to backfill archived reasoning.  In particular, its semantic
        # role=tool content is a better fallback than a v1 raw receipt, while a
        # hash-checked v2 model_context_content remains the sole exact authority.
        current_responses = _project_tool_responses(target.get("tool_responses"))
        incoming_responses = _project_tool_responses(round_step.get("tool_responses"))
        accept_incoming_source = False
        if incoming_responses:
            current_source = str(target.get("tool_response_source") or "")
            incoming_source = str(round_step.get("tool_response_source") or "")
            if not current_responses:
                responses = incoming_responses
                accept_incoming_source = True
            else:
                current_authority = _tool_response_authority(
                    current_source, current_responses
                )
                incoming_authority = _tool_response_authority(
                    incoming_source, incoming_responses
                )
                if incoming_authority > current_authority:
                    responses = incoming_responses
                    accept_incoming_source = True
                elif incoming_authority == current_authority:
                    responses = _merge_tool_responses(
                        current_responses, incoming_responses
                    )
                    accept_incoming_source = bool(incoming_source)
                else:
                    responses = current_responses
                    accept_incoming_source = False
            target["tool_responses"] = responses
            if accept_incoming_source and incoming_source:
                target["tool_response_source"] = incoming_source

        if not reasoning_only:
            visible = str(round_step.get("message") or "")
            if visible:
                target["message"] = visible
            if not target.get("tool_calls") and round_step.get("tool_calls"):
                target["tool_calls"] = list(round_step["tool_calls"])
                target["commands"] = list(round_step.get("commands") or [])
            incoming_output = str(round_step.get("output") or "")
            if incoming_output and (
                accept_incoming_source
                or (not incoming_responses and not str(target.get("output") or ""))
            ):
                target["output"] = incoming_output
            for key in ("prompt_tokens", "completion_tokens"):
                if _as_number(round_step.get(key)) is not None:
                    target[key] = round_step[key]
            if round_step.get("private_round_trace") is True:
                target["private_round_trace"] = True
            if round_step.get("private_tool_response_trace") is True:
                target["private_tool_response_trace"] = True
        merged[target_index] = target
    return merged


def _sort_round_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order persisted model rounds chronologically while preserving ties.

    Different bounded artifacts can begin at different historical frontiers.
    Merge appends a round that is absent from the current base, so list position
    alone is not chronological once (for example) a full tool-response trace is
    joined to a truncated memory tail.  Unanchored live runtime items deliberately
    stay after all persisted model rounds.
    """

    indexed = list(enumerate(steps))
    indexed.sort(
        key=lambda item: (
            _as_int(item[1].get("round_index")) <= 0,
            (
                _as_int(item[1].get("round_index"))
                if _as_int(item[1].get("round_index")) > 0
                else 0
            ),
            item[0],
        )
    )
    return [dict(step) for _, step in indexed]


def _latest_archived_reasoning_fallback(
    display_steps: list[dict[str, Any]],
    archived_steps: list[dict[str, Any]],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Expose old archived reasoning only when the visible WM tail has none."""

    if any(str(step.get("reasoning") or "") for step in display_steps):
        return []
    candidates = [
        dict(step)
        for step in archived_steps
        if str(step.get("reasoning") or "") and step.get("archived_reasoning") is True
    ]
    return candidates[-max(1, limit) :]


def _trajectory_steps(path: Path, *, limit: int | None = 10) -> list[dict[str, Any]]:
    trajectory = _trajectory_payload(path)
    raw_steps = _as_mapping(trajectory).get("steps")
    if not isinstance(raw_steps, list):
        return []
    agent_steps = [
        (round_index, raw)
        for round_index, raw in enumerate(
            (
                raw
                for raw in raw_steps
                if isinstance(raw, Mapping)
                and raw.get("source") not in {"user", "system"}
            ),
            1,
        )
    ]
    if limit is not None:
        agent_steps = agent_steps[-max(1, limit) :]
    steps: list[dict[str, Any]] = []
    for round_index, raw in agent_steps:
        calls: list[dict[str, Any]] = []
        calls_by_id: dict[str, str] = {}
        tool_calls = raw.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                tool = _as_mapping(call)
                call_id = str(tool.get("tool_call_id") or tool.get("call_id") or "")
                function_name = str(
                    tool.get("function_name") or tool.get("name") or "tool"
                )
                if call_id:
                    calls_by_id[call_id] = function_name
                calls.append(
                    {
                        "call_id": call_id,
                        "function_name": function_name,
                        "arguments": _without_token_ids(
                            _as_mapping(tool.get("arguments"))
                        ),
                    }
                )
        tool_responses: list[dict[str, Any]] = []
        results = _as_mapping(raw.get("observation")).get("results")
        if isinstance(results, list):
            for result in results:
                result_mapping = _as_mapping(result)
                content = result_mapping.get("content")
                if not isinstance(content, str):
                    continue
                call_id = str(result_mapping.get("source_call_id") or "")
                tool_responses.append(
                    {
                        "call_id": call_id,
                        "function_name": calls_by_id.get(call_id, "tool"),
                        "output": content,
                    }
                )
        steps.append(
            {
                "step_id": _as_int(raw.get("step_id")),
                "round_index": round_index,
                "timestamp": str(raw.get("timestamp") or ""),
                "message": _truncate(raw.get("message"), MAX_STEP_MESSAGE_CHARS),
                "reasoning": _truncate(
                    raw.get("reasoning_content"), MAX_STEP_REASONING_CHARS
                ),
                "tool_calls": calls,
                "tool_responses": tool_responses,
                "tool_response_source": "trajectory",
                "commands": [
                    command
                    for call in calls
                    if (
                        command := _tool_command(
                            str(call.get("function_name") or "tool"),
                            _as_mapping(call.get("arguments")),
                        )
                    )
                ],
                "output": _truncate(
                    _joined_tool_response_output(tool_responses),
                    MAX_STEP_OUTPUT_CHARS,
                    keep_tail=True,
                ),
                "prompt_tokens": _as_int(
                    _as_mapping(raw.get("metrics")).get("prompt_tokens")
                ),
                "completion_tokens": _as_int(
                    _as_mapping(raw.get("metrics")).get("completion_tokens")
                ),
            }
        )
    return steps


def _memory_steps(
    path: Path,
    *,
    receipt_path: Path | None = None,
    limit: int | None = 10,
) -> list[dict[str, Any]]:
    """Reconstruct archived rounds when an interrupted Codex log was lost.

    This is intentionally a fallback.  `memory.jsonl` contains only rounds that
    crossed the compression/archive boundary, so its cards are labelled as a
    reconstruction and never imply that the uncompressed tail was recovered.
    """

    grouped: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
    for record in _complete_jsonl(path, tail_bytes=MAX_JSONL_BYTES):
        round_index = _as_int(record.get("round"))
        if round_index <= 0:
            continue
        grouped.setdefault(round_index, []).append(record)
    receipts = {
        _as_int(record.get("round_index")): record
        for record in (_receipt_records(receipt_path) if receipt_path else [])
        if _as_int(record.get("round_index")) > 0
    }
    projected: list[dict[str, Any]] = []
    for round_index, records in grouped.items():
        messages: list[str] = []
        reasoning_parts: list[str] = []
        outputs: list[str] = []
        calls: list[dict[str, Any]] = []
        tool_responses: list[dict[str, Any]] = []
        commands: list[str] = []
        for record in records:
            role = str(record.get("role") or "")
            content = record.get("content")
            if role == "assistant" and isinstance(content, str) and content:
                reasoning, visible = _split_think_content(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if visible:
                    messages.append(visible)
            elif role == "tool":
                command = record.get("command")
                command = command if isinstance(command, str) else ""
                call_id = ""
                function_name = "exec_command"
                if isinstance(content, str):
                    call_match = re.match(
                        r"^\[(?P<name>[A-Za-z0-9_]+) call_id=(?P<id>[^\]]+)\]",
                        content,
                    )
                    if call_match:
                        function_name = call_match.group("name")
                        call_id = call_match.group("id")
                arguments: dict[str, Any] = {}
                if command:
                    arguments["keystrokes"] = command
                    commands.append(command.rstrip("\r\n"))
                for key in ("exit_code", "timed_out", "truncated"):
                    if record.get(key) is not None:
                        arguments[key] = record.get(key)
                calls.append(
                    {
                        "call_id": call_id,
                        "function_name": function_name,
                        "arguments": arguments,
                    }
                )
                if isinstance(content, str):
                    outputs.append(content)
                    tool_responses.append(
                        {
                            "call_id": call_id,
                            "function_name": function_name,
                            "output": content,
                        }
                    )
        receipt = receipts.get(round_index, {})
        completion = _as_number(receipt.get("output_tokens"))
        if completion is None:
            completion = _as_number(receipt.get("token_count"))
        projected.append(
            {
                "step_id": round_index,
                "round_index": round_index,
                "display_label": (
                    f"Archived round {round_index} "
                    "(memory.jsonl; not current working memory)"
                ),
                "timestamp": "",
                "message": _truncate(
                    "\n\n".join(messages), MAX_STEP_MESSAGE_CHARS, keep_tail=True
                ),
                "reasoning": _truncate(
                    "\n\n".join(reasoning_parts),
                    MAX_STEP_REASONING_CHARS,
                    keep_tail=True,
                ),
                "reasoning_source": "memory_archive",
                "archived_reasoning": bool(reasoning_parts),
                "tool_calls": calls,
                "tool_responses": tool_responses,
                "tool_response_source": "memory_archive",
                "commands": commands,
                "output": _truncate(
                    "\n\n".join(outputs), MAX_STEP_OUTPUT_CHARS, keep_tail=True
                ),
                "prompt_tokens": _as_int(receipt.get("input_tokens")),
                "completion_tokens": int(completion or 0),
                "reconstructed_from_memory": True,
            }
        )
    if limit is not None:
        projected = projected[-max(1, limit) :]
    return projected


def _codex_log_path(trial: Path) -> Path | None:
    """Find a bounded Codex JSONL log, failing closed outside the trial/cache."""

    trial = Path(trial)
    persisted = trial / "agent" / "codex.txt"
    if not _has_symlink_component(persisted):
        try:
            info = persisted.stat()
            if stat.S_ISREG(info.st_mode) and info.st_size <= MAX_CODEX_LOG_BYTES:
                return persisted
        except OSError:
            pass

    # Active PRoot trials have not copied include_logs back to agent/ yet.  The
    # utility receipt is the trial-owned binding to the random cache run.
    expected_session = f"{trial.name}__env"
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", expected_session):
        return None
    receipt = _read_json(
        trial / f"proot-guest-utilities-{expected_session}.json",
        max_bytes=MAX_POOL_ARTIFACT_BYTES,
    )
    if receipt is None or receipt.get("session_id") != expected_session:
        return None

    cache_root = Path(PROOT_CACHE_ROOT)
    runs_root = cache_root / "runs"
    if _has_symlink_component(cache_root) or _has_symlink_component(runs_root):
        return None
    try:
        resolved_cache = cache_root.resolve(strict=True)
        resolved_runs = _assert_within(runs_root, resolved_cache, kind="PRoot runs")
        candidates = list(resolved_runs.iterdir())
    except (OSError, MonitorArtifactError):
        return None

    bundle_pattern = re.compile(re.escape(expected_session) + r"-[0-9a-f]{8}")
    matches: list[Path] = []
    for bundle in candidates:
        if (
            not bundle_pattern.fullmatch(bundle.name)
            or bundle.is_symlink()
            or not bundle.is_dir()
            or _has_symlink_component(bundle)
        ):
            continue
        candidate = bundle / "rootfs" / "logs" / "agent" / "codex.txt"
        if _has_symlink_component(candidate):
            continue
        try:
            resolved = _assert_within(candidate, resolved_runs, kind="Codex live log")
            info = resolved.stat()
        except (OSError, MonitorArtifactError):
            continue
        if stat.S_ISREG(info.st_mode) and info.st_size <= MAX_CODEX_LOG_BYTES:
            matches.append(resolved)
    # A stale duplicate must never be guessed as the active trial.
    return matches[0] if len(matches) == 1 else None


def _codex_steps(path: Path, *, limit: int | None = 10) -> list[dict[str, Any]]:
    """Project Codex CLI JSONL command/message items into monitor step cards."""

    path = Path(path)
    if _has_symlink_component(path):
        return []
    try:
        info = path.stat()
    except OSError:
        return []
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CODEX_LOG_BYTES:
        return []

    items: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for event in _complete_jsonl(path, tail_bytes=MAX_JSONL_BYTES):
        event_type = event.get("type")
        if event_type not in {"item.started", "item.completed"}:
            continue
        item = _as_mapping(event.get("item"))
        item_id = item.get("id")
        item_type = item.get("type")
        if not isinstance(item_id, str) or item_type not in {
            "command_execution",
            "agent_message",
            "error",
        }:
            continue
        items[item_id] = dict(item)

    projected: list[dict[str, Any]] = []
    for ordinal, (item_id, item) in enumerate(items.items(), 1):
        suffix = re.fullmatch(r"item_(\d+)", item_id)
        step_id = int(suffix.group(1)) if suffix else ordinal
        item_type = item.get("type")
        command = item.get("command") if item_type == "command_execution" else ""
        command = command if isinstance(command, str) else ""
        output = item.get("aggregated_output")
        output = output if isinstance(output, str) else ""
        message = (
            item.get("text")
            if item_type == "agent_message"
            else item.get("message") if item_type == "error" else ""
        )
        message = message if isinstance(message, str) else ""
        status = str(item.get("status") or "")
        exit_code = item.get("exit_code")
        arguments: dict[str, Any] = {"keystrokes": command} if command else {}
        if status:
            arguments["status"] = status
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            arguments["exit_code"] = exit_code
        projected.append(
            {
                "step_id": step_id,
                # A single gateway response can fan out into multiple Codex CLI
                # command items. The runtime log therefore cannot name model rounds
                # by ordinal; exact command-sequence reconciliation does that later.
                "round_index": 0,
                "runtime_item_type": str(item_type or ""),
                "display_label": (
                    f"Codex tool item {step_id}"
                    if item_type == "command_execution"
                    else (
                        f"Codex runtime error {step_id}"
                        if item_type == "error"
                        else f"Codex message {step_id}"
                    )
                ),
                "timestamp": "",
                "message": _truncate(
                    message
                    or (
                        f"Command {status or 'recorded'}"
                        + (f" (exit {exit_code})" if isinstance(exit_code, int) else "")
                    ),
                    MAX_STEP_MESSAGE_CHARS,
                ),
                "reasoning": "",
                "tool_calls": (
                    [
                        {
                            "call_id": item_id,
                            "function_name": "exec_command",
                            "arguments": arguments,
                        }
                    ]
                    if command
                    else []
                ),
                "tool_responses": (
                    [
                        {
                            "call_id": item_id,
                            "function_name": "exec_command",
                            "output": output,
                        }
                    ]
                    if item_type == "command_execution"
                    else []
                ),
                "tool_response_source": "codex_runtime",
                "commands": [command] if command else [],
                "output": _truncate(output, MAX_STEP_OUTPUT_CHARS, keep_tail=True),
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        )
    if limit is not None:
        projected = projected[-max(1, limit) :]
    return projected


def _unwrap_codex_shell_command(command: str) -> str:
    """Recover the command body from Codex's ``/bin/bash -lc`` wrapper."""

    try:
        argv = shlex.split(command)
    except ValueError:
        return command.strip()
    if (
        len(argv) == 3
        and argv[0] in {"bash", "/bin/bash", "sh", "/bin/sh"}
        and argv[1] in {"-c", "-lc"}
    ):
        return argv[2].strip()
    return command.strip()


def _merge_codex_runtime_steps(
    round_steps: list[dict[str, Any]],
    runtime_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach Codex runtime output to exact persisted command sequences.

    Runtime item ordinals are not model rounds: Codex may split one shell call
    into several command items, and one model response may contain several tool
    calls.  Persisted trace/memory commands are therefore the only round anchors.
    Duplicate commands are consumed in order. Unmatched live-tail items remain
    visible but deliberately carry no invented round number or reasoning.
    """

    if not round_steps:
        return [dict(step) for step in runtime_steps]
    merged = [dict(step) for step in round_steps]
    round_positions = {
        _as_int(step.get("round_index")): index
        for index, step in enumerate(merged)
        if _as_int(step.get("round_index")) > 0
    }
    anchors: list[dict[str, Any]] = []
    for step in merged:
        round_index = _as_int(step.get("round_index"))
        if round_index <= 0:
            continue
        for command in step.get("commands") or []:
            if isinstance(command, str) and command:
                anchors.append(
                    {"round_index": round_index, "command": command, "cursor": 0}
                )
    anchor_cursor = 0
    unmatched: list[tuple[int, dict[str, Any]]] = []
    last_matched_runtime_index = -1
    for runtime_index, runtime in enumerate(runtime_steps):
        runtime_type = str(runtime.get("runtime_item_type") or "")
        commands = [
            command
            for command in (runtime.get("commands") or [])
            if isinstance(command, str) and command
        ]
        matched_round = 0
        if commands:
            command = _unwrap_codex_shell_command(commands[0])
            for index in range(anchor_cursor, len(anchors)):
                anchor = anchors[index]
                persisted = str(anchor["command"])
                start = _as_int(anchor.get("cursor"))
                position = persisted.find(command, start)
                if position < 0:
                    continue
                matched_round = _as_int(anchor.get("round_index"))
                end = position + len(command)
                anchor["cursor"] = end
                remaining = persisted[end:].strip().lstrip(";&|").strip()
                anchor_cursor = index + 1 if not remaining else index
                break
        elif runtime_type == "agent_message":
            message = str(runtime.get("message") or "")
            if message:
                for step in merged:
                    if message == str(step.get("message") or ""):
                        matched_round = _as_int(step.get("round_index"))
                        break
        if matched_round > 0 and matched_round in round_positions:
            target_index = round_positions[matched_round]
            target = dict(merged[target_index])
            runtime_output = str(runtime.get("output") or "")
            response_source = str(target.get("tool_response_source") or "")
            runtime_may_supply_output = _tool_response_authority(
                response_source, target.get("tool_responses")
            ) <= _tool_response_authority(
                "codex_runtime", runtime.get("tool_responses")
            )
            if (
                runtime_output
                and runtime_may_supply_output
                and not str(target.get("output") or "")
            ):
                target["output"] = runtime_output
            elif (
                runtime_output
                and runtime_may_supply_output
                and target.get("private_round_trace") is True
                and runtime_output not in str(target.get("output") or "")
            ):
                target["output"] = (
                    str(target.get("output") or "") + "\n\n" + runtime_output
                )
            if response_source in {"", "codex_runtime"}:
                responses = _merge_tool_responses(
                    target.get("tool_responses"),
                    runtime.get("tool_responses"),
                )
                if responses:
                    target["tool_responses"] = responses
                    target["tool_response_source"] = "codex_runtime"
            target["runtime_item_count"] = _as_int(target.get("runtime_item_count")) + 1
            merged[target_index] = target
            last_matched_runtime_index = runtime_index
            continue
        if runtime_type == "error":
            continue
        unanchored = dict(runtime)
        unanchored["round_index"] = 0
        unanchored["reasoning"] = ""
        unanchored["reasoning_source"] = "unavailable_unarchived_runtime_item"
        unanchored["display_label"] = (
            str(runtime.get("display_label") or "Codex runtime item")
            + " (unmatched to persisted model round)"
        )
        unmatched.append((runtime_index, unanchored))
    # An unmatched item before a later exact anchor is an old reconciliation gap,
    # not the live WM tail. Keeping it at the end would misorder the trajectory.
    tail = [step for index, step in unmatched if index > last_matched_runtime_index]
    return [*merged, *tail]


def _read_trial_result(trial: Path) -> Mapping[str, Any] | None:
    result = _read_json(trial / "result.json")
    config = _read_json(trial / "config.json") or {}
    expected_task = _as_mapping(config.get("task")).get("name")
    if result is None:
        return None
    if (
        result.get("trial_name") != trial.name
        or not isinstance(expected_task, str)
        or not expected_task
        or result.get("task_name") != expected_task
        or _parse_time(result.get("finished_at")) is None
    ):
        return None
    return result


def _reward(result: Mapping[str, Any]) -> float | int | None:
    rewards = _as_mapping(_as_mapping(result.get("verifier_result")).get("rewards"))
    candidate = rewards.get("reward")
    primary = _as_finite_number(candidate)
    if primary is not None:
        return primary
    for value in rewards.values():
        auxiliary = _as_finite_number(value)
        if auxiliary is not None:
            return auxiliary
    return None


def _primary_verifier_reward(result: Mapping[str, Any]) -> float | int | None:
    rewards = _as_mapping(_as_mapping(result.get("verifier_result")).get("rewards"))
    return _as_finite_number(rewards.get("reward"))


def _harness_label(import_path: str, agent_name: str = "") -> tuple[str, bool]:
    normalized_import = str(import_path or "")
    normalized_agent = str(agent_name or "")
    lowered_import = normalized_import.lower()
    lowered_agent = normalized_agent.lower()
    is_codex = "codex" in lowered_import or lowered_agent == "codex"
    if is_codex:
        if "codexnativecompactr2eactagent" in lowered_import:
            return "Codex CLI + native compact", True
        return "Codex CLI + r²eAct", True
    if "terminus" in lowered_import:
        return "Terminus-2 + r²eAct", False
    return normalized_agent or "—", False


def _trial_status(result: Mapping[str, Any] | None, *, result_exists: bool) -> str:
    if result is None:
        return "committing" if result_exists else "incomplete"
    exception_type = str(
        _as_mapping(result.get("exception_info")).get("exception_type") or ""
    )
    if exception_type:
        return "cancelled" if "cancel" in exception_type.lower() else "error"
    reward = _reward(result)
    if reward is None:
        return "completed"
    return "passed" if float(reward) > 0 else "failed"


def _first_int(*values: Any) -> int | None:
    for value in values:
        number = _as_number(value)
        if number is not None:
            return int(number)
    return None


def _artifact_signature(paths: Iterable[Path]) -> tuple[tuple[int, int], ...]:
    signature: list[tuple[int, int]] = []
    for path in paths:
        if _has_symlink_component(path):
            signature.append((0, 0))
            continue
        try:
            info = path.stat()
        except OSError:
            signature.append((0, 0))
            continue
        signature.append((info.st_mtime_ns, info.st_size))
    return tuple(signature)


def _trial_projection(trial: Path) -> dict[str, Any]:
    trial = Path(trial)
    result_path = trial / "result.json"
    result_exists = result_path.is_file() and not _has_symlink_component(result_path)
    projection_signature = _artifact_signature(_artifact_paths(trial))
    if result_exists:
        with _JSON_CACHE_LOCK:
            cached_projection = _TRIAL_PROJECTION_CACHE.get(str(trial.absolute()))
        if (
            cached_projection is not None
            and cached_projection[0] == projection_signature
        ):
            return dict(cached_projection[1])
    config = _read_json(trial / "config.json") or {}
    result = _read_trial_result(trial)
    task_config = _as_mapping(config.get("task"))
    task_name = str(task_config.get("name") or trial.name.split("__", 1)[0])
    agent_config = _as_mapping(config.get("agent"))
    import_path = str(agent_config.get("import_path") or "")
    event_path = _first_existing(trial / "agent", EVENT_FILENAMES)
    receipt_path = _first_existing(trial / "agent", RECEIPT_FILENAMES)
    round_trace_path = _controller_artifact_path(
        trial / "agent", CODEX_ROUND_TRACE_FILENAME
    )
    events = _event_summary(event_path)
    receipts = _receipt_summary(receipt_path)
    round_metrics = _round_trace_cumulative_summary(round_trace_path)
    memory_summary = _memory_summary(
        _controller_artifact_path(trial / "agent", MEMORY_FILENAME)
    )
    trajectory = _trajectory_payload(trial / "agent" / "trajectory.json")
    final_metrics = _as_mapping(_as_mapping(trajectory).get("final_metrics"))

    agent_result = _as_mapping(_as_mapping(result).get("agent_result"))
    r2eact = _as_mapping(_as_mapping(agent_result.get("metadata")).get("r2eact"))
    recorded_memory = _as_mapping(r2eact.get("memory"))
    exception = _as_mapping(result.get("exception_info")) if result else {}
    agent_info = _as_mapping(result.get("agent_info")) if result else {}
    committed_round = max(
        _as_int(receipts.get("latest_round")),
        _as_int(events.get("latest_committed_round")),
        _as_int(round_metrics.get("latest_round")),
    )
    observed_round = max(committed_round, _as_int(events.get("latest_round")))
    rounds = _first_int(r2eact.get("rounds"))
    if not rounds:
        rounds = committed_round or None
    memory_records = _first_int(recorded_memory.get("records"))
    memory_path = _controller_artifact_path(trial / "agent", MEMORY_FILENAME)
    if not memory_records and memory_path.is_file():
        memory_records = _as_int(memory_summary.get("records"))
    terminal_input_tokens = _first_int(
        agent_result.get("n_input_tokens"),
        final_metrics.get("total_prompt_tokens"),
    )
    terminal_output_tokens = _first_int(
        agent_result.get("n_output_tokens"),
        final_metrics.get("total_completion_tokens"),
    )
    receipt_covers_committed = bool(receipts.get("complete")) and _as_int(
        receipts.get("latest_round")
    ) >= committed_round
    receipt_input_exact = receipt_covers_committed and bool(
        receipts.get("input_tokens_exact", receipts.get("complete"))
    )
    receipt_output_exact = receipt_covers_committed and bool(
        receipts.get("output_tokens_exact", receipts.get("complete"))
    )
    receipt_tools_exact = receipt_covers_committed and bool(
        receipts.get("tool_calls_exact", receipts.get("complete"))
    )
    round_metrics_exact = bool(round_metrics.get("complete")) and _as_int(
        round_metrics.get("latest_round")
    ) >= committed_round
    input_tokens = _first_int(
        terminal_input_tokens,
        receipts.get("input_tokens") if receipt_input_exact else None,
        round_metrics.get("input_tokens") if round_metrics_exact else None,
        receipts.get("input_tokens"),
        round_metrics.get("input_tokens"),
    )
    output_tokens = _first_int(
        terminal_output_tokens,
        receipts.get("output_tokens") if receipt_output_exact else None,
        round_metrics.get("output_tokens") if round_metrics_exact else None,
        receipts.get("output_tokens"),
        round_metrics.get("output_tokens"),
    )
    tool_calls = _first_int(
        receipts.get("tool_calls") if receipt_tools_exact else None,
        round_metrics.get("tool_calls") if round_metrics_exact else None,
        receipts.get("tool_calls"),
        round_metrics.get("tool_calls"),
    )
    input_tokens_exact = terminal_input_tokens is not None or receipt_input_exact or (
        round_metrics_exact and _as_number(round_metrics.get("input_tokens")) is not None
    )
    output_tokens_exact = (
        terminal_output_tokens is not None
        or receipt_output_exact
        or (
            round_metrics_exact
            and _as_number(round_metrics.get("output_tokens")) is not None
        )
    )
    tool_calls_exact = receipt_tools_exact or (
        round_metrics_exact and _as_number(round_metrics.get("tool_calls")) is not None
    )
    cumulative_metrics_exact = bool(
        input_tokens_exact and output_tokens_exact and tool_calls_exact
    )
    duration = (
        _elapsed(result.get("started_at"), result.get("finished_at"))
        if result is not None
        else None
    )
    if duration is None and _as_number(events.get("first_observed_at")) is not None:
        started = datetime.fromtimestamp(float(events["first_observed_at"]), tz=UTC)
        duration = max(0.0, (datetime.now(UTC) - started).total_seconds())
    if duration is None and _as_number(round_metrics.get("first_observed_at")) is not None:
        started = datetime.fromtimestamp(
            float(round_metrics["first_observed_at"]), tz=UTC
        )
        duration = max(0.0, (datetime.now(UTC) - started).total_seconds())
    agent_name = str(
        agent_info.get("name")
        or r2eact.get("agent")
        or (import_path.split(":")[-1] if import_path else "")
    )
    harness, is_codex = _harness_label(import_path, agent_name)
    latest_generation = _as_mapping(receipts.get("latest_generation")) or _as_mapping(
        events.get("latest_generation")
    )
    latest_compression = _as_mapping(receipts.get("latest_compression")) or _as_mapping(
        events.get("latest_compression")
    )
    projection = {
        "trial": str(_as_mapping(result).get("trial_name") or trial.name),
        "task": str(_as_mapping(result).get("task_name") or task_name),
        "task_ref": str(task_config.get("ref") or ""),
        "source": str(task_config.get("source") or ""),
        "status": _trial_status(result, result_exists=result_exists),
        "terminal": result is not None,
        "reward": _reward(result) if result is not None else None,
        "primary_reward": (
            _primary_verifier_reward(result) if result is not None else None
        ),
        "has_exception": bool(exception),
        "exception": str(exception.get("exception_type") or ""),
        "exception_message": _truncate(exception.get("exception_message"), 1_000),
        "duration": _format_duration(duration),
        "duration_seconds": duration,
        "rounds": rounds,
        "observed_round": observed_round or None,
        "inflight_round": observed_round if observed_round > committed_round else None,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_calls": tool_calls,
        "input_tokens_exact": input_tokens_exact,
        "output_tokens_exact": output_tokens_exact,
        "tool_calls_exact": tool_calls_exact,
        "cumulative_metrics_exact": cumulative_metrics_exact,
        "cumulative_metrics_round": max(
            _as_int(receipts.get("latest_round"))
            if (
                receipt_input_exact
                or receipt_output_exact
                or receipt_tools_exact
            )
            else 0,
            _as_int(round_metrics.get("latest_round")),
        ),
        "memory_records": memory_records,
        "memory_rounds": _as_int(memory_summary.get("rounds")),
        "agent": agent_name,
        "version": str(agent_info.get("version") or r2eact.get("agent_version") or ""),
        "harness": harness,
        "is_codex": is_codex,
        "latest_generation": dict(latest_generation),
        "latest_compression": dict(latest_compression),
        "receipt_records": _as_int(receipts.get("records")),
        "validation_recoveries": _as_int(events.get("validation_recoveries")),
        "response_outcomes": dict(_as_mapping(events.get("response_outcomes"))),
        "memory_lookups": _as_int(events.get("lookups")),
        "memory_lookup_successes": _as_int(events.get("lookup_successes")),
        "memory_lookup_failures": _as_int(events.get("lookup_failures")),
        "progress_guards": _as_int(events.get("guard_count")),
    }
    if result is not None:
        with _JSON_CACHE_LOCK:
            if len(_TRIAL_PROJECTION_CACHE) >= MAX_TRIAL_PROJECTION_CACHE_ENTRIES:
                _TRIAL_PROJECTION_CACHE.pop(next(iter(_TRIAL_PROJECTION_CACHE)))
            _TRIAL_PROJECTION_CACHE[str(trial.absolute())] = (
                projection_signature,
                dict(projection),
            )
    return projection


def _trial_result_summary(path: Path) -> dict[str, Any] | None:
    projection = _trial_projection(path)
    return projection if projection["terminal"] else None


def _partition_trials(
    trials: list[Path], *, reported_running: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    terminal: list[dict[str, Any]] = []
    incomplete: list[tuple[datetime | None, dict[str, Any]]] = []
    for trial in trials:
        row = _trial_projection(trial)
        row["path"] = trial
        if row["terminal"]:
            terminal.append(row)
        else:
            incomplete.append((_latest_mtime(_artifact_paths(trial)), row))
    incomplete.sort(
        key=lambda item: item[0] or datetime.min.replace(tzinfo=UTC), reverse=True
    )
    active_count = min(max(0, reported_running), len(incomplete))
    active = [row for _, row in incomplete[:active_count]]
    leftovers = [row for _, row in incomplete[active_count:]]
    for row in active:
        row["status"] = "running (artifact heuristic)"
    terminal.sort(key=lambda row: str(row["trial"]))
    return terminal, active, leftovers


def _configured_task_names(job_dir: Path) -> list[str]:
    config = _read_json(Path(job_dir) / "config.json") or {}
    names: list[str] = []
    datasets = config.get("datasets")
    if isinstance(datasets, list):
        for dataset in datasets:
            raw_names = _as_mapping(dataset).get("task_names")
            if not isinstance(raw_names, list):
                continue
            names.extend(str(name) for name in raw_names if isinstance(name, str))
    tasks = config.get("tasks")
    if isinstance(tasks, list):
        names.extend(
            str(_as_mapping(task).get("name"))
            for task in tasks
            if isinstance(_as_mapping(task).get("name"), str)
        )
    return list(dict.fromkeys(names))


def _logical_member_attempts(member: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project one linked Harbor job into per-task retry attempts."""

    job_dir = Path(member["path"])
    progress = _job_progress(_read_json(job_dir / "result.json"))
    terminal, active, incomplete = _partition_trials(
        _trial_directories(job_dir), reported_running=progress["running"]
    )
    attempts: list[dict[str, Any]] = []
    observed_tasks: set[str] = set()
    for state, rows in (
        ("terminal", terminal),
        ("running", active),
        ("interrupted", incomplete),
    ):
        for row in rows:
            observed_tasks.add(str(row["task"]))
            retry_state = row["status"] if state == "terminal" else state
            if (
                state == "terminal"
                and not str(row.get("exception") or "")
                and _as_finite_number(row.get("primary_reward")) is None
            ):
                retry_state = "unscoreable"
            attempts.append(
                {
                    **row,
                    "member_job": str(member["job_name"]),
                    "member_role": str(member["role"]),
                    "member_harness": str(member["harness"]),
                    "member_priority": _as_int(member.get("priority")),
                    "retry_state": retry_state,
                }
            )
    for task_name in _configured_task_names(job_dir):
        if task_name in observed_tasks:
            continue
        attempts.append(
            {
                "task": task_name,
                "trial": "",
                "path": None,
                "task_ref": "",
                "source": "",
                "terminal": False,
                "reward": None,
                "primary_reward": None,
                "exception": "",
                "status": "pending",
                "harness": str(member["harness"]),
                "member_job": str(member["job_name"]),
                "member_role": str(member["role"]),
                "member_harness": str(member["harness"]),
                "member_priority": _as_int(member.get("priority")),
                "retry_state": "pending",
            }
        )
    return attempts


def _apply_logical_pool_members(
    rows: list[dict[str, Any]], members: list[dict[str, Any]]
) -> dict[str, Any]:
    """Overlay linked retry attempts without mutating canonical pool artifacts."""

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    for row in rows:
        canonical_harness = str(row.get("harness") or "Terminus-2 + r²eAct")
        row["baseline_reward"] = row.get("reward")
        row["baseline_score_state"] = row.get("score_state")
        row["baseline_unscoreable_reason"] = row.get("unscoreable_reason")
        row["baseline_unscoreable_reason_code"] = row.get("unscoreable_reason_code")
        row["baseline_status"] = row.get("status")
        row["baseline_harness"] = canonical_harness
        row["effective_reward"] = row.get("reward")
        row["effective_harness"] = row["baseline_harness"]
        row["effective_rounds"] = row.get("rounds")
        row["effective_input_tokens"] = row.get("input_tokens")
        row["effective_output_tokens"] = row.get("output_tokens")
        row["effective_memory_records"] = row.get("memory_records")
        row["retry_state"] = "—"
        row["retry_harness"] = ""
        row["retry_job"] = ""
        row["logical_history"] = []
        baseline_trial = row.get("trial_path")
        if baseline_trial or row.get("harbor_job_path"):
            row["logical_history"].append(
                {
                    "label": f"baseline · {canonical_harness}",
                    "path": None,
                    "kind": "baseline",
                    "status": row.get("outcome"),
                    "reward": row.get("reward"),
                }
            )

    attempts: list[dict[str, Any]] = []
    ignored: list[dict[str, str]] = []
    for member in members:
        attempts.extend(_logical_member_attempts(member))
    attempts.sort(
        key=lambda row: (
            _as_int(row.get("member_priority")),
            str(row.get("member_job") or ""),
            str(row.get("trial") or ""),
        )
    )
    terminal_attempt_counts = Counter(
        (str(attempt.get("member_job") or ""), str(attempt.get("task") or ""))
        for attempt in attempts
        if attempt.get("terminal") is True
    )
    linked_running: list[dict[str, Any]] = []
    retry_counts: Counter[str] = Counter()
    for attempt in attempts:
        task_name = str(attempt.get("task") or "")
        member_task = (str(attempt.get("member_job") or ""), task_name)
        if (
            attempt.get("terminal") is True
            and terminal_attempt_counts[member_task] != 1
        ):
            ignored.append(
                {
                    "task": task_name,
                    "reason": "multiple terminal retries in one linked job",
                }
            )
            continue
        matching_cells = by_task.get(task_name, [])
        if not matching_cells:
            ignored.append(
                {"task": task_name, "reason": "task is not in canonical pool"}
            )
            continue
        if len(matching_cells) != 1:
            ignored.append(
                {
                    "task": task_name,
                    "reason": "official attempt is ambiguous in canonical pool",
                }
            )
            continue
        cell = matching_cells[0]
        task_ref = str(attempt.get("task_ref") or "")
        expected_task_ref = str(cell.get("task_ref") or "")
        source = str(attempt.get("source") or "")
        expected_source = str(_as_mapping(cell).get("source") or "")
        if not expected_task_ref or task_ref != expected_task_ref:
            ignored.append({"task": task_name, "reason": "task ref mismatch"})
            continue
        if not expected_source or source != expected_source:
            ignored.append({"task": task_name, "reason": "task source mismatch"})
            continue
        retry_state = str(attempt.get("retry_state") or "unknown")
        retry_counts[retry_state] += 1
        cell["retry_state"] = retry_state
        cell["retry_harness"] = str(attempt.get("member_harness") or "")
        cell["retry_job"] = str(attempt.get("member_job") or "")
        trial_path = attempt.get("path")
        cell["logical_history"].append(
            {
                "label": (
                    f"{retry_state} · {attempt.get('member_harness') or 'linked harness'}"
                    + (f" · {attempt.get('trial')}" if attempt.get("trial") else "")
                ),
                "path": trial_path,
                "kind": "linked",
                "status": attempt.get("status"),
                "reward": attempt.get("primary_reward"),
            }
        )
        primary_reward = _as_finite_number(attempt.get("primary_reward"))
        valid_terminal = (
            attempt.get("terminal") is True
            and not str(attempt.get("exception") or "")
            and attempt.get("has_exception") is not True
            and primary_reward is not None
        )
        if valid_terminal:
            cell["effective_reward"] = primary_reward
            cell["reward"] = primary_reward
            cell["score_state"] = "scored"
            cell["score_evidence"] = "linked terminal retry result"
            cell["unscoreable_reason"] = ""
            cell["unscoreable_reason_code"] = ""
            cell["unscoreable_evidence"] = ""
            effective_harness = str(
                attempt.get("member_harness")
                or attempt.get("harness")
                or "linked harness"
            )
            cell["effective_harness"] = effective_harness
            cell["harness"] = effective_harness
            cell["effective_rounds"] = attempt.get("rounds")
            cell["rounds"] = attempt.get("rounds")
            cell["effective_input_tokens"] = attempt.get("input_tokens")
            cell["input_tokens"] = attempt.get("input_tokens")
            cell["effective_output_tokens"] = attempt.get("output_tokens")
            cell["output_tokens"] = attempt.get("output_tokens")
            cell["effective_memory_records"] = attempt.get("memory_records")
            cell["memory_records"] = attempt.get("memory_records")
            cell["effective_trial_path"] = trial_path
        if retry_state == "running" and isinstance(trial_path, Path):
            linked_running.append(
                {
                    **cell,
                    "status": "retry running",
                    "outcome": "retry running",
                    "terminal": False,
                    "monitor_trial_path": trial_path,
                    "worker": str(attempt.get("member_job") or "linked job"),
                    "device": "—",
                    "execution_attempt": len(cell["logical_history"]),
                    "heartbeat_at": None,
                    "lease_expires_at": None,
                    "agent_time_budget": {},
                    "input_tokens": attempt.get("input_tokens"),
                    "output_tokens": attempt.get("output_tokens"),
                    "effective_rounds": attempt.get("rounds"),
                    "effective_memory_records": attempt.get("memory_records"),
                    "retry_harness": str(attempt.get("member_harness") or ""),
                }
            )
    return {
        "attempts": attempts,
        "running": linked_running,
        "retry_counts": dict(retry_counts),
        "ignored": ignored,
    }


def _job_progress(result: Mapping[str, Any] | None) -> dict[str, Any]:
    result = result or {}
    stats = _as_mapping(result.get("stats"))
    total = _as_int(result.get("n_total_trials"))
    if total <= 0:
        total = sum(
            _as_int(stats.get(key))
            for key in (
                "n_completed_trials",
                "n_running_trials",
                "n_pending_trials",
                "n_cancelled_trials",
            )
        )
    mean: float | None = None
    evals = stats.get("evals")
    if isinstance(evals, Mapping):
        for evaluation in evals.values():
            metrics = _as_mapping(evaluation).get("metrics")
            if isinstance(metrics, list) and metrics:
                candidate = _as_mapping(metrics[0]).get("mean")
                if _as_number(candidate) is not None:
                    mean = float(candidate)
                    break
    return {
        "total": total,
        "completed": _as_int(stats.get("n_completed_trials")),
        "running": _as_int(stats.get("n_running_trials")),
        "pending": _as_int(stats.get("n_pending_trials")),
        "errored": _as_int(stats.get("n_errored_trials")),
        "cancelled": _as_int(stats.get("n_cancelled_trials")),
        "input_tokens": _as_int(stats.get("n_input_tokens")),
        "output_tokens": _as_int(stats.get("n_output_tokens")),
        "mean": mean,
        "started_at": result.get("started_at"),
        "updated_at": result.get("updated_at"),
        "finished_at": result.get("finished_at"),
    }


def _parse_sglang_load(value: Any) -> dict[str, int] | None:
    if isinstance(value, Mapping):
        nested = value.get("loads") or value.get("data")
        value = nested if isinstance(nested, list) else [value]
    if not isinstance(value, list) or not value:
        return None
    records = [row for row in value if isinstance(row, Mapping)]
    if not records:
        return None
    total = sum(_as_int(row.get("num_reqs")) for row in records)
    waiting = sum(_as_int(row.get("num_waiting_reqs")) for row in records)
    return {
        "active_requests": max(0, total - waiting),
        "waiting_requests": waiting,
        "live_tokens": sum(_as_int(row.get("num_tokens")) for row in records),
    }


def _read_sglang_load(url: str) -> dict[str, int] | None:
    if not url:
        return None
    endpoint = (
        url if url.rstrip("/").endswith("/get_load") else f"{url.rstrip('/')}/get_load"
    )
    try:
        with urllib.request.urlopen(endpoint, timeout=1.0) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            return None
        return _parse_sglang_load(json.loads(raw))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _safe_job_contract(job_dir: Path) -> dict[str, Any]:
    config = _read_json(job_dir / "config.json") or {}
    agents = config.get("agents")
    agent = _as_mapping(agents[0]) if isinstance(agents, list) and agents else {}
    kwargs = _as_mapping(agent.get("kwargs"))
    datasets = config.get("datasets")
    dataset = (
        _as_mapping(datasets[0]) if isinstance(datasets, list) and datasets else {}
    )
    import_path = str(agent.get("import_path") or "")
    harness, is_codex = _harness_label(
        import_path,
        (import_path or "—").split(":")[-1],
    )
    unbounded = agent_import_uses_unbounded_timeout(import_path)
    is_tb21 = str(dataset.get("name") or "") == TB21_DATASET_SOURCE
    configured_timeout = _as_number(agent.get("override_timeout_sec"))
    if unbounded and is_tb21:
        time_budget_policy = TB21_UNIFORM_AGENT_TIMEOUT_POLICY
        timeout_seconds = float(TB21_PER_TASK_AGENT_TIMEOUT_SECONDS)
    elif unbounded:
        time_budget_policy = UNBOUNDED_AGENT_TIMEOUT_POLICY
        timeout_seconds = None
    elif configured_timeout is None:
        time_budget_policy = "harbor_task_timeout_fallback"
        timeout_seconds = None
    else:
        time_budget_policy = "harbor_configured_timeout"
        timeout_seconds = configured_timeout
    return {
        "dataset": str(dataset.get("name") or "—"),
        "agent": (import_path or "—").split(":")[-1],
        "import_path": import_path,
        "harness": harness,
        "is_codex": is_codex,
        "model": str(agent.get("model_name") or "—"),
        "attempts": _as_int(config.get("n_attempts")),
        "concurrency": _as_int(config.get("n_concurrent_trials")),
        "timeout_seconds": timeout_seconds,
        "configured_timeout_seconds": configured_timeout,
        "time_budget_policy": time_budget_policy,
        "working_memory_turns": _as_int(kwargs.get("working_memory_turns")),
        "max_new_tokens": _as_int(kwargs.get("max_new_tokens")),
    }


def _time_budget_label(contract: Mapping[str, Any]) -> str:
    if contract.get("time_budget_policy") == UNBOUNDED_AGENT_TIMEOUT_POLICY:
        return "unlimited"
    if contract.get("time_budget_policy") in {
        TB21_OFFICIAL_AGENT_TIMEOUT_POLICY,
        "harbor_task_timeout_fallback",
    }:
        return "task default"
    timeout = _as_number(contract.get("timeout_seconds"))
    return _format_duration(float(timeout)) if timeout is not None else "—"


def _phase_duration(result: Mapping[str, Any], field: str) -> float | None:
    phase = _as_mapping(result.get(field))
    return _elapsed(phase.get("started_at"), phase.get("finished_at"))


def _image_previews(
    agent_dir: Path, rounds: set[int]
) -> dict[int, list[dict[str, Any]]]:
    """Load exact admitted pixels, not mutable guest paths or remote URLs.

    Match AB's hash-verified preview contract. TB snapshots live beside their
    receipts in the private controller directory (or the legacy agent dir),
    so completed and remote-worker trials need no surviving local rootfs.
    """
    if not rounds:
        return {}
    from PIL import Image

    receipt_path = _controller_artifact_path(agent_dir, GATEWAY_IMAGE_RECEIPTS_FILENAME)
    result: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, str, str]] = set()
    for receipt in _complete_jsonl(receipt_path, tail_bytes=MAX_JSONL_BYTES):
        index, digest, call_id = (
            receipt.get("round_index"),
            receipt.get("sha256"),
            receipt.get("call_id"),
        )
        if (
            receipt.get("schema") != "codex_r2eact_image_receipt_v1"
            or receipt.get("function_name") != "view_image"
            or type(index) is not int
            or index not in rounds
            or not isinstance(call_id, str)
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            continue
        identity = (index, call_id, digest)
        if identity in seen:
            continue
        seen.add(identity)
        preview: dict[str, Any] = {
            "path": str(receipt.get("path") or ""),
            "sha256": digest,
        }
        result.setdefault(index, []).append(preview)
        path = receipt_path.parent / IMAGE_STORE_DIRNAME / digest
        if _has_symlink_component(path):
            continue
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_IMAGE_BYTES:
                    continue
                data = stream.read(MAX_IMAGE_BYTES + 1)
            if (
                len(data) > MAX_IMAGE_BYTES
                or hashlib.sha256(data).hexdigest() != digest
            ):
                continue
            with Image.open(io.BytesIO(data)) as img:
                if (
                    img.format not in {"PNG", "JPEG", "WEBP", "GIF"}
                    or img.width * img.height > 25_000_000
                ):
                    continue
                width = min(img.width, 900)
                img.verify()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            continue
        preview.update(data=data, width=width)
    return result


def _render_step(st: Any, step: Mapping[str, Any], *, expanded: bool) -> None:
    title = str(step.get("display_label") or f"Round {step.get('step_id', 0)}")
    timestamp = _parse_time(step.get("timestamp"))
    if timestamp:
        title += f" · {timestamp:%H:%M:%S} UTC"
    with st.expander(title, expanded=expanded):
        message = str(step.get("message") or "")
        if message:
            st.markdown("**Agent message**")
            st.code(message, language=None, wrap_lines=True)
        if step.get("reasoning_source") == "unavailable_unarchived_runtime_item":
            st.caption(
                "Reasoning and the exact model-round number are unavailable for "
                "this unarchived working-memory runtime item; the monitor does not "
                "infer either value."
            )
        reasoning = str(step.get("reasoning") or "")
        if reasoning:
            reasoning_label = (
                "Archived reasoning (memory.jsonl; not current WM)"
                if step.get("archived_reasoning") is True
                else "Reasoning"
            )
            # Streamlit does not allow an expander inside this round expander.
            st.markdown(f"**{reasoning_label}**")
            st.code(reasoning, language=None, wrap_lines=True)
        calls = step.get("tool_calls")
        if isinstance(calls, list):
            for index, call in enumerate(calls, 1):
                tool = _as_mapping(call)
                st.markdown(
                    f"**Tool {index}: `{tool.get('function_name') or 'tool'}`**"
                )
                arguments = _as_mapping(tool.get("arguments"))
                function_name = str(tool.get("function_name") or "tool")
                command = _tool_command(function_name, arguments)
                if command:
                    st.code(command, language="bash", wrap_lines=True)
                    remainder = {
                        key: value
                        for key, value in arguments.items()
                        if key not in {"keystrokes", "cmd", "command"}
                        and not (function_name == "write_stdin" and key == "chars")
                    }
                    if remainder:
                        st.json(remainder, expanded=False)
                elif arguments:
                    st.json(arguments, expanded=False)
        responses = _project_tool_responses(step.get("tool_responses"))
        response_source = str(step.get("tool_response_source") or "")
        if responses and response_source == "private_tool_response_trace_v1":
            st.caption(
                "Legacy v1 tool receipt: exact model-context payload unavailable."
            )
        elif responses and response_source == "memory_archive":
            st.caption(
                "memory.jsonl stores semantic tool content; exact model-context "
                "payload unavailable."
            )
        for index, response in enumerate(responses, 1):
            response_label = (
                "Exact model-context tool response"
                if response.get("model_context_exact") is True
                else "Tool response"
            )
            if len(responses) > 1:
                response_label += f" {index}"
            function_name = response.get("function_name") or "tool"
            st.markdown(f"**{response_label}: `{function_name}`**")
            st.code(response["output"], language=None, wrap_lines=True)
        for preview in step.get("image_previews", []):
            st.markdown("**view_image**")
            st.caption(str(preview.get("path") or ""))
            if preview.get("data"):
                st.image(preview["data"], width=preview["width"])
            else:
                st.caption("当时的图片快照不可用；不会用内容已改变的文件替代。")
        st.caption(
            f"prompt {_as_int(step.get('prompt_tokens')):,} · "
            f"completion {_as_int(step.get('completion_tokens')):,} tokens"
        )


def _render_verifier(st: Any, trial: Path, result: Mapping[str, Any]) -> None:
    st.markdown("#### Verifier (terminal result committed)")
    verifier_result = _as_mapping(result.get("verifier_result"))
    exception = _as_mapping(result.get("exception_info"))
    columns = st.columns(5)
    reward = _reward(result)
    columns[0].metric("reward", "—" if reward is None else str(reward))
    columns[1].metric(
        "environment", _format_duration(_phase_duration(result, "environment_setup"))
    )
    columns[2].metric(
        "agent setup", _format_duration(_phase_duration(result, "agent_setup"))
    )
    columns[3].metric(
        "agent", _format_duration(_phase_duration(result, "agent_execution"))
    )
    columns[4].metric("verifier", _format_duration(_phase_duration(result, "verifier")))
    if verifier_result:
        st.json(_without_token_ids(verifier_result), expanded=False)
    if exception:
        st.error(
            f"{exception.get('exception_type') or 'Exception'}: "
            f"{_truncate(exception.get('exception_message'), 8_000)}"
        )
    ctrf = _read_json(trial / "verifier" / "ctrf.json", max_bytes=8 * 1024 * 1024)
    if ctrf:
        with st.expander("CTRf verifier report", expanded=False):
            st.json(_without_token_ids(ctrf), expanded=False)
    stdout = _bounded_text(
        trial / "verifier" / "test-stdout.txt", max_chars=200_000, tail=True
    )
    if stdout:
        with st.expander("Verifier stdout", expanded=False):
            st.code(stdout, language="text", wrap_lines=True)


def _render_artifacts(
    st: Any,
    trial: Path,
    steps: list[dict[str, Any]],
    *,
    terminal: bool,
    is_codex: bool,
) -> None:
    st.markdown("#### Persisted artifacts")
    key = re.sub(r"[^A-Za-z0-9_-]", "_", trial.name)
    columns = st.columns(4)
    show_memory = columns[0].checkbox("Show memory.jsonl", key=f"tb_memory_{key}")
    runtime_label = "Show Codex CLI log" if is_codex else "Show terminal pane"
    show_runtime = columns[1].checkbox(runtime_label, key=f"tb_pane_{key}")
    show_events = columns[2].checkbox("Show event journal", key=f"tb_events_{key}")
    show_receipts = columns[3].checkbox("Show receipts", key=f"tb_receipts_{key}")
    agent = trial / "agent"
    if show_memory:
        memory = _bounded_text(
            _controller_artifact_path(agent, MEMORY_FILENAME),
            max_chars=MAX_TEXT_CHARS,
            tail=True,
        )
        st.code(memory or "[empty]", language="json", wrap_lines=True)
    if show_runtime:
        runtime_path = (
            _codex_log_path(trial)
            if is_codex
            else _first_existing(agent, TERMINAL_PANE_FILENAMES)
        )
        if runtime_path is None:
            st.info(
                "Codex CLI log was not recovered after interruption; archived "
                "memory rounds are used for the trajectory fallback."
            )
        else:
            st.code(
                _bounded_text(runtime_path, max_chars=500_000, tail=True) or "[empty]",
                language="text",
                wrap_lines=True,
            )
    if show_events:
        event_path = _first_existing(agent, EVENT_FILENAMES)
        events = list(_complete_jsonl(event_path, tail_bytes=MAX_JSONL_BYTES))
        st.json(_without_token_ids(events), expanded=False)
    if show_receipts:
        receipt_path = _first_existing(agent, RECEIPT_FILENAMES)
        receipts = (
            {
                "cumulative_summary": _receipt_summary(receipt_path),
                "recent_receipts": _receipt_records(receipt_path),
                "note": "recent_receipts is a bounded validated tail",
            }
            if receipt_path.name == GATEWAY_RECEIPTS_JSONL_FILENAME
            else _read_json_value(receipt_path)
        )
        st.json(_without_token_ids(receipts or {}), expanded=False)
    projection = _trial_projection(trial)
    payload = json.dumps(
        {
            "trial": trial.name,
            "terminal": terminal,
            "artifact_projection": projection,
            "steps": steps,
        },
        ensure_ascii=False,
        indent=2,
    ).encode()
    st.download_button(
        "Download trajectory audit projection (token IDs omitted)",
        data=payload,
        file_name=f"{trial.name}-trajectory-audit.json",
        mime="application/json",
        key=f"tb_download_{key}_{terminal}",
    )


def _render_trial(
    st: Any, trial: Path, *, terminal: bool, full_history: bool, status: str
) -> None:
    trajectory = _trajectory_payload(trial / "agent" / "trajectory.json")
    config = _read_json(trial / "config.json") or {}
    result = _read_trial_result(trial) if terminal else None
    projection = _trial_projection(trial)
    fallback_task = str(
        _as_mapping(result).get("task_name")
        or _as_mapping(config.get("task")).get("name")
        or trial.name.split("__", 1)[0]
    )
    task = _task_description(trajectory, fallback=fallback_task)
    if task == fallback_task and projection.get("is_codex"):
        task = _codex_task_from_trial_log(trial / "trial.log") or task
    agent_dir = trial / "agent"
    steps = _trajectory_steps(agent_dir / "trajectory.json", limit=None)
    receipt_path = _first_existing(agent_dir, RECEIPT_FILENAMES)
    trace_steps = _round_trace_steps(
        _controller_artifact_path(agent_dir, CODEX_ROUND_TRACE_FILENAME)
    )
    tool_response_steps = _tool_response_trace_steps(
        _controller_artifact_path(agent_dir, CODEX_TOOL_RESPONSE_TRACE_FILENAME)
    )
    archived_steps = _memory_steps(
        _controller_artifact_path(agent_dir, MEMORY_FILENAME),
        receipt_path=receipt_path,
        limit=None,
    )
    codex_log = _codex_log_path(trial) if projection.get("is_codex") else None
    runtime_steps = _codex_steps(codex_log, limit=None) if codex_log is not None else []
    if steps:
        if trace_steps:
            steps = _merge_round_steps(steps, trace_steps)
        if archived_steps:
            steps = _merge_round_steps(steps, archived_steps, reasoning_only=True)
    else:
        if trace_steps:
            steps = _merge_round_steps(archived_steps, trace_steps)
        elif archived_steps:
            steps = archived_steps
        else:
            steps = []
    if tool_response_steps:
        steps = _merge_round_steps(steps, tool_response_steps)
    steps = _sort_round_steps(steps)
    if runtime_steps:
        steps = _merge_codex_runtime_steps(steps, runtime_steps)
    display_steps = steps if full_history else steps[-MAX_LIVE_ROUNDS:]
    archived_reasoning_fallback = _latest_archived_reasoning_fallback(
        display_steps, archived_steps
    )
    image_previews = _image_previews(
        agent_dir,
        {
            step["round_index"]
            for step in [*display_steps, *archived_reasoning_fallback]
            if type(step.get("round_index")) is int and step["round_index"] > 0
        },
    )
    event_path = _first_existing(trial / "agent", EVENT_FILENAMES)
    events = _event_summary(event_path)
    memory = _memory_summary(
        _controller_artifact_path(trial / "agent", MEMORY_FILENAME)
    )
    input_tokens = projection.get("input_tokens")
    output_tokens = projection.get("output_tokens")
    cumulative_metrics_exact = bool(projection.get("cumulative_metrics_exact"))
    input_tokens_exact = bool(
        projection.get("input_tokens_exact", cumulative_metrics_exact)
    )
    output_tokens_exact = bool(
        projection.get("output_tokens_exact", cumulative_metrics_exact)
    )
    tool_calls_exact = bool(
        projection.get("tool_calls_exact", cumulative_metrics_exact)
    )
    elapsed_label = str(projection.get("duration") or "—")

    st.markdown(f"### {trial.name}")
    st.markdown("#### Task")
    st.code(task, language=None, wrap_lines=True)
    columns = st.columns(8)
    columns[0].metric("status", status)
    committed_round = projection.get("rounds")
    inflight_round = projection.get("inflight_round")
    round_label = _format_count(committed_round)
    if _as_number(inflight_round) is not None:
        round_label += f" · {int(inflight_round):,} generating"
    columns[1].metric("committed round", round_label)
    columns[2].metric(
        "input tokens",
        _format_cumulative_count(input_tokens, exact=input_tokens_exact),
    )
    columns[3].metric(
        "output tokens",
        _format_cumulative_count(output_tokens, exact=output_tokens_exact),
    )
    projected_tool_calls = projection.get("tool_calls")
    if _as_number(projected_tool_calls) is None:
        projected_tool_calls = sum(len(step["tool_calls"]) for step in steps)
    columns[4].metric(
        "tools",
        _format_cumulative_count(projected_tool_calls, exact=tool_calls_exact),
    )
    columns[5].metric(
        "memory records",
        _format_cumulative_count(
            memory["records"], exact=bool(memory.get("records_exact"))
        ),
    )
    columns[6].metric("memory", f"{memory['bytes'] / 1024:.1f} KiB")
    columns[7].metric("elapsed", elapsed_label)
    if not projection.get("is_codex") and (
        _as_int(projection.get("memory_lookups"))
        or _as_int(projection.get("progress_guards"))
    ):
        st.caption(
            f"memory lookups: {_as_int(projection.get('memory_lookups')):,} "
            f"({_as_int(projection.get('memory_lookup_successes')):,} ok / "
            f"{_as_int(projection.get('memory_lookup_failures')):,} failed) · "
            f"progress guards: {_as_int(projection.get('progress_guards')):,}"
        )
    if events.get("truncated"):
        st.info("Event telemetry is based on the bounded 16 MiB committed tail.")
    if result is not None:
        _render_verifier(st, trial, result)
    st.markdown("#### Trajectory")
    unanchored_tail_count = sum(
        step.get("reasoning_source") == "unavailable_unarchived_runtime_item"
        for step in display_steps
    )
    if unanchored_tail_count:
        st.info(
            f"{unanchored_tail_count} recent Codex runtime items belong to the "
            "unarchived working-memory tail and could not be anchored to a persisted "
            "model round. Their commands/output are shown, but their round numbers "
            "and reasoning are intentionally left unavailable."
        )
    if archived_reasoning_fallback:
        agent_config = _as_mapping(config.get("agent"))
        agent_import_path = str(agent_config.get("import_path") or "")
        native_compact = "CodexNativeCompactR2EActAgent" in agent_import_path
        working_memory_turns = _as_int(
            _as_mapping(agent_config.get("kwargs")).get("working_memory_turns")
        )
        latest_archived_round = max(
            _as_int(step.get("round_index")) for step in archived_reasoning_fallback
        )
        latest_observed_round = max(
            _as_int(projection.get("observed_round")),
            max((_as_int(step.get("round_index")) for step in steps), default=0),
        )
        observed_lag = max(0, latest_observed_round - latest_archived_round)
        archive_policy = (
            " Codex native compaction owns the archive frontier, so there is no "
            "fixed working-memory-round window."
            if native_compact
            else (
                f" The configured working-memory window is "
                f"{working_memory_turns or 12} rounds, and pending compression can "
                "make the archive lag larger."
            )
        )
        st.info(
            "The current unarchived working-memory tail has no persisted reasoning. "
            f"Showing the latest available archived reasoning through round "
            f"{latest_archived_round}; it is currently {observed_lag} rounds behind "
            f"the latest displayed round.{archive_policy}"
        )
        for step in reversed(archived_reasoning_fallback):
            _render_step(
                st,
                {
                    **step,
                    "image_previews": image_previews.get(step.get("round_index"), []),
                },
                expanded=False,
            )
    if not display_steps:
        if _as_number(committed_round) is not None:
            st.info(
                f"{int(committed_round):,} gateway rounds are committed, but no "
                "recoverable transcript artifact is available."
            )
        else:
            st.info(
                "No committed agent round yet; a generation may still be in flight."
            )
    for index, step in enumerate(reversed(display_steps)):
        _render_step(
            st,
            {**step, "image_previews": image_previews.get(step.get("round_index"), [])},
            expanded=index == 0,
        )
    _render_artifacts(
        st,
        trial,
        steps,
        terminal=terminal,
        is_codex=bool(projection.get("is_codex")),
    )


def _render_pool(
    st: Any,
    *,
    jobs_root: Path,
    job_dir: Path,
    marker: Mapping[str, Any],
) -> None:
    """Render compact distributed state and lazily open one Harbor trial."""

    refreshed_at = datetime.now(UTC)
    state = _read_json(job_dir / "state.json", max_bytes=MAX_POOL_ARTIFACT_BYTES) or {}
    cells = _pool_cells(marker)
    contract = _safe_job_contract(job_dir)
    canonical_harness = str(contract.get("harness") or "—")
    rows = [
        _enrich_pool_cell_projection(
            job_dir,
            _pool_cell_projection(
                job_dir,
                cell,
                now=refreshed_at,
                expected_contract_hash=str(marker.get("config_contract_hash") or ""),
            ),
            canonical_harness=canonical_harness,
        )
        for cell in cells
    ]
    attempt_selection = _pool_attempt_selection(job_dir, marker)
    _apply_pool_attempt_selection(rows, attempt_selection)
    task_selection = _pool_task_selection(job_dir, marker)
    _apply_pool_task_selection(rows, task_selection)
    formal_state_valid = _reconcile_formal_pool_state(
        rows,
        state=state,
        marker=marker,
        job_dir=job_dir,
    )
    # Keep the canonical artifacts and full row set above for receipt
    # reconciliation, but do not surface retired/non-scoring attempts in the
    # live monitor.  attempt-selection.json remains the auditable source of
    # truth on disk; the UI is intentionally a projection of the score-eligible
    # pool only.
    visible_rows = [row for row in rows if row.get("score_state") != "excluded"]
    canonical_unscoreable = [
        row for row in visible_rows if row.get("score_state") == "unscoreable"
    ]
    members = _logical_pool_members(jobs_root, job_dir)
    logical = (
        _apply_logical_pool_members(visible_rows, members)
        if members
        else {
            "attempts": [],
            "running": [],
            "retry_counts": {},
            "ignored": [],
        }
    )
    terminal = [row for row in visible_rows if row["terminal"]]
    canonical_running = [row for row in visible_rows if row["status"] == "running"]
    running = [
        *canonical_running,
        *logical["running"],
    ]
    stale = [row for row in visible_rows if row["status"] == "stale"]
    artifact_errors = [row for row in visible_rows if row["status"] == "artifact error"]
    queued = [
        row
        for row in visible_rows
        if not row["terminal"]
        and row["status"] not in {"running", "stale", "artifact error", "verification_pending"}
    ]
    workers = _pool_worker_projections(job_dir)
    active_workers = [
        worker
        for worker in workers
        if worker["status"] in {"starting", "running", "idle"}
    ]
    row_rewards = [
        float(row["reward"])
        for row in terminal
        if row.get("score_state") == "scored"
        and _as_number(row.get("reward")) is not None
    ]
    unscoreable_counts = Counter(
        str(row.get("unscoreable_reason") or "unknown") for row in canonical_unscoreable
    )
    row_scored_count = sum(row.get("score_state") == "scored" for row in terminal)
    row_unscoreable_count = sum(
        row.get("score_state") == "unscoreable" for row in terminal
    )
    state_all_slot_mean_raw = state.get("all_slot_mean_reward")
    state_all_slot_mean = _as_finite_number(state_all_slot_mean_raw)
    # Cell results are the terminal score authority.  Long-lived workers can
    # keep rewriting state.json with code loaded before a reconciliation, so
    # live counts, coverage, and observed mean must always come from the rows
    # revalidated by this monitor.  A matching formal state may attest cleanup
    # candidates above and expose the all-slot mean, but never overrides rows.
    metric_total = len(visible_rows)
    terminal_count = len(terminal)
    scored_count = row_scored_count
    unscoreable_count = row_unscoreable_count
    visible_artifact_errors = sum(
        row["status"] == "artifact error" for row in visible_rows
    )
    pending_count = len(visible_rows) - len(terminal) - visible_artifact_errors
    observed_mean = sum(row_rewards) / len(row_rewards) if row_rewards else None
    formal_all_slot_mean = state_all_slot_mean if formal_state_valid else None
    if observed_mean is not None and not math.isfinite(float(observed_mean)):
        observed_mean = None
    if formal_all_slot_mean is not None and not math.isfinite(
        float(formal_all_slot_mean)
    ):
        formal_all_slot_mean = None
    st.subheader(_pool_display_name(job_dir, marker))
    if members:
        st.caption(
            f"logical composite view: canonical {canonical_harness} pool + "
            + ", ".join(
                f"{member['job_name']} ({member['harness']}, {member['role']})"
                for member in members
            )
            + ". Linked retries retain their provenance in execution history; "
            "Terminal cells shows one resolved reward per cell."
        )
    pulse = st.columns(6 if members else 5)
    pulse[0].metric("refresh", refreshed_at.strftime("%H:%M:%S UTC"))
    pulse[1].metric("pool", str(state.get("status") or "running"))
    pulse[2].metric("active workers", f"{len(active_workers):,}")
    pulse[3].metric("canonical running cells", f"{len(canonical_running):,}")
    pulse[4].metric("stale leases", f"{len(stale):,}")
    if members:
        pulse[5].metric("linked retries running", f"{len(logical['running']):,}")
    st.progress(
        min(
            max(terminal_count / metric_total if metric_total else 0.0, 0.0),
            1.0,
        )
    )
    metrics = st.columns(8)
    metrics[0].metric("terminal", f"{terminal_count:,}/{metric_total:,}")
    metrics[1].metric("scored", f"{scored_count:,}")
    metrics[2].metric("unscoreable", f"{unscoreable_count:,}")
    metrics[3].metric("pending", f"{pending_count:,}")
    metrics[4].metric("canonical running", f"{len(canonical_running):,}")
    metrics[5].metric("stale", f"{len(stale):,}")
    metrics[6].metric("artifact errors", f"{len(artifact_errors):,}")
    metrics[7].metric(
        "elapsed",
        _format_duration(
            _elapsed(
                state.get("started_at") or marker.get("initialized_at"),
                state.get("finished_at"),
            )
        ),
    )
    score_metrics = st.columns(3)
    score_metrics[0].metric(
        "observed mean reward",
        "—" if observed_mean is None else f"{observed_mean:.4f}",
    )
    score_metrics[1].metric(
        "formal all-slot mean reward",
        "—" if formal_all_slot_mean is None else f"{formal_all_slot_mean:.4f}",
    )
    score_metrics[2].metric(
        "score coverage",
        f"{scored_count:,}/{metric_total:,}" if metric_total else "0/0",
    )
    if formal_state_valid:
        st.caption(
            "Live counts, coverage, and observed mean are revalidated from each "
            "cell result. The matching pool state receipt attests cleanup-only "
            "candidates and the formal all-slot mean. Formal all-slot mean remains "
            "— until every displayed slot is scoreable; pending or unscoreable "
            "cells are never model zero."
        )
    else:
        st.caption(
            "The formal pool state receipt is unavailable, stale, or internally "
            "inconsistent. Live counts, coverage, and observed mean are revalidated "
            "from each cell result; formal all-slot mean is withheld."
        )
    if unscoreable_counts:
        st.caption(
            "Canonical unscoreable breakdown: "
            + " · ".join(
                f"{reason}: {count:,}"
                for reason, count in sorted(
                    unscoreable_counts.items(), key=lambda item: (-item[1], item[0])
                )
            )
        )

    st.divider()
    verification_pending = [r for r in visible_rows if r["status"] == "verification_pending"]
    if verification_pending:
        st.subheader("Pending verification")
        st.caption("Agent execution is finished. These tasks await verifier-only recovery and have no reward yet.")
        st.dataframe([{"task": r["task"], "attempt": r["official_attempt"],
                       "execution": r["execution_attempt"], "status": "verification_pending"}
                      for r in verification_pending], use_container_width=True, hide_index=True)
    st.subheader("Terminal cells")
    if terminal:
        st.dataframe(
            [
                {
                    "task": row["task"],
                    "official attempt": row["official_attempt"],
                    "status": row["status"],
                    "outcome": row["outcome"],
                    "reward": row["reward"],
                    "score state": row["score_state"],
                    "score evidence": row.get("score_evidence") or "—",
                    "unscoreable reason": row["unscoreable_reason"] or "—",
                    "reason evidence": row["unscoreable_evidence"] or "—",
                    "harness": row.get("harness") or canonical_harness,
                    "exception": row["exception"] or "—",
                    "rounds": _format_count(row.get("rounds")),
                    "input tokens": _format_count(row.get("input_tokens")),
                    "output tokens": _format_count(row.get("output_tokens")),
                    "memory records": _format_count(row.get("memory_records")),
                    "execution": row["execution_attempt"],
                    "time budget": _execution_time_budget_label(
                        _as_mapping(row.get("agent_time_budget"))
                    ),
                    "elapsed": row.get("duration") or "—",
                    "finished": row["finished_at"] or "—",
                }
                for row in terminal
            ],
            width="stretch",
            hide_index=True,
        )
        completed_id = st.selectbox(
            "Completed cell history",
            [str(row["cell_id"]) for row in reversed(terminal)],
            format_func=lambda value: next(
                (
                    f"{row['task']} · attempt {row['official_attempt']}"
                    for row in terminal
                    if row["cell_id"] == value
                ),
                value,
            ),
            key=f"tb_pool_completed_{job_dir.name}",
        )
        if st.checkbox(
            "Show selected completed cell full history",
            value=False,
            key=f"tb_pool_completed_history_{job_dir.name}",
        ):
            selected = next(row for row in terminal if row["cell_id"] == completed_id)
            history = list(selected.get("logical_history") or [])
            history_options: list[tuple[str, Path | None]] = []
            for entry in history:
                path = entry.get("path")
                if entry.get("kind") == "baseline":
                    path = _pool_trial_directory(job_dir, selected)
                history_options.append((str(entry.get("label") or "attempt"), path))
            if not history_options:
                history_options.append(
                    (
                        f"canonical · {selected.get('harness') or canonical_harness}",
                        _pool_trial_directory(job_dir, selected),
                    )
                )
            chosen_history = next(
                (
                    index
                    for index in range(len(history_options) - 1, -1, -1)
                    if history_options[index][1] is not None
                ),
                0,
            )
            if len(history_options) > 1:
                chosen_label = st.selectbox(
                    "Harness execution",
                    [label for label, _ in history_options],
                    index=chosen_history,
                    key=f"tb_pool_execution_history_{job_dir.name}_{completed_id}",
                )
                chosen_history = next(
                    index
                    for index, (label, _) in enumerate(history_options)
                    if label == chosen_label
                )
            trial = history_options[chosen_history][1]
            if trial is None:
                st.warning("This terminal cell has no safe Harbor trial artifact path.")
            else:
                _render_trial(
                    st,
                    trial,
                    terminal=_read_trial_result(trial) is not None,
                    full_history=True,
                    status=str(_trial_projection(trial).get("status") or "unknown"),
                )
    else:
        st.caption("No terminal cell result commit yet.")

    st.divider()
    st.subheader("Running cells")
    if running:
        st.dataframe(
            [
                {
                    "task": row["task"],
                    "attempt": row["official_attempt"],
                    "worker": row["worker"] or "—",
                    "device": row["device"] or "—",
                    "execution": row["execution_attempt"],
                    "round": _format_count(row.get("rounds")),
                    "observed round": _format_count(row.get("observed_round")),
                    "input tokens": _format_count(row.get("input_tokens")),
                    "output tokens": _format_count(row.get("output_tokens")),
                    "memory records": _format_count(row.get("memory_records")),
                    "harness": row.get("harness") or canonical_harness,
                    "elapsed": row.get("duration") or "—",
                    "time budget": _execution_time_budget_label(
                        _as_mapping(row.get("agent_time_budget"))
                    ),
                    "heartbeat": row["heartbeat_at"] or "—",
                    "lease": row["lease_expires_at"] or "—",
                }
                for row in running
            ],
            width="stretch",
            hide_index=True,
        )
        running_options = [str(row["cell_id"]) for row in running]
        active_state_key = f"tb_pool_active_{job_dir.name}"
        if st.session_state.get(active_state_key) not in running_options:
            st.session_state.pop(active_state_key, None)
        running_id = st.selectbox(
            "Active cell",
            running_options,
            format_func=lambda value: next(
                (
                    f"{row['task']} · attempt {row['official_attempt']}"
                    for row in running
                    if str(row["cell_id"]) == value
                ),
                value,
            ),
            key=active_state_key,
        )
        selected = next(row for row in running if str(row["cell_id"]) == running_id)
        trial = selected.get("monitor_trial_path") or _pool_trial_directory(
            job_dir, selected
        )
        if trial is None:
            st.info(
                "The cell is claimed, but Harbor has not committed a safe trial "
                "directory yet; generation or environment setup may still be in flight."
            )
        else:
            _render_trial(
                st,
                trial,
                terminal=False,
                full_history=False,
                status="running",
            )
    else:
        st.caption("No cell currently has a live lease.")

    if stale or artifact_errors or queued:
        with st.expander(
            "Queue diagnostics "
            f"({len(queued)} queued · {len(stale)} stale · "
            f"{len(artifact_errors)} artifact errors)",
            expanded=bool(artifact_errors),
        ):
            st.dataframe(
                [
                    {
                        "task": row["task"],
                        "attempt": row["official_attempt"],
                        "state": row["status"],
                        "worker": row["worker"] or "—",
                        "detail": row["outcome"],
                    }
                    for row in [*artifact_errors, *stale, *queued]
                ],
                width="stretch",
                hide_index=True,
            )
    if logical["attempts"]:
        with st.expander(
            f"Linked harness retry attempts ({len(logical['attempts'])})",
            expanded=False,
        ):
            st.dataframe(
                [
                    {
                        "task": attempt.get("task"),
                        "state": attempt.get("retry_state"),
                        "reward": attempt.get("primary_reward"),
                        "exception": attempt.get("exception") or "—",
                        "harness": attempt.get("member_harness"),
                        "job": attempt.get("member_job"),
                        "trial": attempt.get("trial") or "—",
                    }
                    for attempt in logical["attempts"]
                ],
                width="stretch",
                hide_index=True,
            )
            if logical["ignored"]:
                st.warning(
                    "Some linked attempts failed canonical task/ref validation and "
                    "were omitted from effective rewards."
                )
                st.dataframe(logical["ignored"], width="stretch", hide_index=True)
    if workers:
        with st.expander(f"Worker heartbeats ({len(workers)})", expanded=False):
            st.dataframe(workers, width="stretch", hide_index=True)


def _render_job(st: Any, *, jobs_root: Path, job_dir: Path) -> None:
    marker = _pool_marker(job_dir)
    if marker is not None:
        _render_pool(st, jobs_root=jobs_root, job_dir=job_dir, marker=marker)
        return
    refreshed_at = datetime.now(UTC)
    job_result = _read_json(job_dir / "result.json")
    progress = _job_progress(job_result)
    contract = _safe_job_contract(job_dir)
    completed = progress["completed"]
    total = progress["total"]

    st.subheader(job_dir.name)
    st.caption(
        f"{contract['dataset']} · {contract['agent']} · {contract['model']} · "
        f"attempts {contract['attempts']} · concurrency {contract['concurrency']} · "
        f"time budget {_time_budget_label(contract)}"
    )
    load = _read_sglang_load(SGLANG_LOAD_URL)
    job_state = (
        "finished"
        if progress["finished_at"]
        else "active" if progress["running"] > 0 else "unfinished / inactive"
    )
    pulse = st.columns(5)
    pulse[0].metric("refresh", refreshed_at.strftime("%H:%M:%S UTC"))
    pulse[1].metric("job", job_state)
    pulse[2].metric(
        "local SGLang active",
        "—" if load is None else f"{load['active_requests']:,}",
    )
    pulse[3].metric(
        "local SGLang waiting",
        "—" if load is None else f"{load['waiting_requests']:,}",
    )
    pulse[4].metric(
        "local SGLang tokens", "—" if load is None else f"{load['live_tokens']:,}"
    )
    st.caption(
        "SGLang load is local/shared only. Zero tokens means idle at this refresh; "
        "it is not a trial-status signal."
    )
    st.progress(min(max(completed / total if total else 0.0, 0.0), 1.0))
    metrics = st.columns(8)
    metrics[0].metric("completed", f"{completed:,}/{total:,}")
    metrics[1].metric("Harbor running", f"{progress['running']:,}")
    metrics[2].metric("pending", f"{progress['pending']:,}")
    metrics[3].metric("errors", f"{progress['errored']:,}")
    metrics[4].metric("cancelled", f"{progress['cancelled']:,}")
    metrics[5].metric(
        "mean reward",
        "—" if progress["mean"] is None else f"{progress['mean']:.4f}",
    )
    metrics[6].metric("input tokens", f"{progress['input_tokens']:,}")
    metrics[7].metric(
        "elapsed",
        _format_duration(_elapsed(progress["started_at"], progress["finished_at"])),
    )

    trials = _trial_directories(job_dir)
    terminal, active, incomplete = _partition_trials(
        trials, reported_running=progress["running"]
    )
    st.divider()
    st.subheader("Terminal trials")
    if terminal:
        st.dataframe(
            [
                {
                    "task": row["task"],
                    "trial": row["trial"],
                    "status": row["status"],
                    "reward": row["reward"],
                    "exception": row["exception"] or "—",
                    "rounds": _format_count(row["rounds"]),
                    "input tokens": _format_count(row["input_tokens"]),
                    "output tokens": _format_count(row["output_tokens"]),
                    "memory records": _format_count(row["memory_records"]),
                    "elapsed": row["duration"],
                    "harness": row["harness"] or "—",
                    "agent": f"{row['agent']} {row['version']}".strip() or "—",
                }
                for row in terminal
            ],
            width="stretch",
            hide_index=True,
        )
        selected_name = st.selectbox(
            "Completed trial history",
            [str(row["trial"]) for row in reversed(terminal)],
            key=f"tb_completed_{job_dir.name}",
        )
        if st.checkbox(
            "Show selected completed trial full history",
            value=False,
            key=f"tb_completed_history_{job_dir.name}",
        ):
            selected = next(row for row in terminal if row["trial"] == selected_name)
            _render_trial(
                st,
                selected["path"],
                terminal=True,
                full_history=True,
                status=str(selected["status"]),
            )
    else:
        st.caption("No fully parsed terminal result commit yet.")

    st.divider()
    st.subheader("Running trials")
    if active:
        active_rows: list[dict[str, Any]] = []
        for row in active:
            trial = Path(row["path"])
            codex_log = _codex_log_path(trial) if row.get("is_codex") else None
            artifact_paths = _artifact_paths(trial)
            if codex_log is not None:
                artifact_paths.append(codex_log)
            latest = _latest_mtime(artifact_paths)
            age = (refreshed_at - latest).total_seconds() if latest else None
            active_rows.append(
                {
                    "task": row["task"],
                    "trial": row["trial"],
                    "status": row["status"],
                    "committed round": _format_count(row["rounds"]),
                    "observed round": _format_count(row["observed_round"]),
                    "input tokens": _format_count(row["input_tokens"]),
                    "output tokens": _format_count(row["output_tokens"]),
                    "memory records": _format_count(row["memory_records"]),
                    "elapsed": row["duration"],
                    "harness": row["harness"] or "—",
                    "artifact progress age": _format_duration(age),
                }
            )
        st.dataframe(active_rows, width="stretch", hide_index=True)
        st.caption(
            "Harbor supplies only the running count, not a per-trial heartbeat. "
            "Rows are matched to newest incomplete artifacts; age is observational only."
        )
        follow = st.checkbox(
            "Follow running trial", value=True, key=f"tb_follow_{job_dir.name}"
        )
        if follow:
            current = active[0]
        else:
            chosen = st.selectbox(
                "Active trial",
                [str(row["trial"]) for row in active],
                key=f"tb_active_{job_dir.name}",
            )
            current = next(row for row in active if row["trial"] == chosen)
        _render_trial(
            st,
            Path(current["path"]),
            terminal=False,
            full_history=False,
            status=str(current["status"]),
        )
    else:
        st.caption("Harbor currently reports no running trial.")
    if incomplete:
        with st.expander(
            f"Interrupted/incomplete artifact directories ({len(incomplete)})",
            expanded=False,
        ):
            st.caption(
                "These directories have no complete result commit and are not called "
                "active because Harbor's reported running count is already exhausted."
            )
            st.dataframe(
                [
                    {
                        "task": row["task"],
                        "trial": row["trial"],
                        "state": row["status"],
                    }
                    for row in incomplete
                ],
                width="stretch",
                hide_index=True,
            )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Terminal-Bench · r²eAct live monitor", layout="wide")
    st.title("Terminal-Bench · r²eAct live monitor")
    st.caption(
        "Committed artifacts only. Terminal-Bench grades environment state; verifier "
        "details appear only after a complete trial result commit."
    )
    jobs_root = DEFAULT_JOBS_ROOT.expanduser().resolve()
    if not jobs_root.is_dir():
        st.error(f"Jobs directory does not exist: {jobs_root}")
        return
    jobs = _visible_jobs(jobs_root, _discover_jobs(jobs_root))
    if not jobs:
        st.info(f"No Terminal-Bench jobs under {jobs_root}")
        return
    preferred_name = Path(PREFERRED_JOB_DIR_RAW).name if PREFERRED_JOB_DIR_RAW else ""
    names = [path.name for path in jobs]
    selected_index = names.index(preferred_name) if preferred_name in names else 0
    selected_name = st.selectbox(
        "Benchmark pool / historical job",
        names,
        index=selected_index,
        format_func=lambda name: _job_selector_label(
            next(path for path in jobs if path.name == name)
        ),
        key="tb_monitor_selected_job",
    )
    job_dir = next(path for path in jobs if path.name == selected_name)
    render = lambda: _render_job(st, jobs_root=jobs_root, job_dir=job_dir)
    fragment = getattr(st, "fragment", None)
    if callable(fragment):
        fragment(run_every=REFRESH_SECONDS)(render)()
    else:  # pragma: no cover - old Streamlit compatibility
        render()
    st.caption(f"monitor PID {os.getpid()}")


if __name__ == "__main__":
    main()
