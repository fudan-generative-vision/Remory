"""Classify serving failures from controller receipts, never model/tool text.

Recovery preserves each execution's artifacts. It does not manufacture a model
answer, reward, or claim that a fresh execution is a full session resume.
"""
from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import socket
import ssl
from urllib.error import HTTPError, URLError

TEMPORARY_SERVING_EXIT = 75
MAX_TASK_RECOVERIES = 3


def supervisor_failure():
    """Controller-only marker distinguishes serving restart from human Ctrl+C."""
    path = os.environ.get("TERMINAL_BENCH_POOL_RECOVERY_SIGNAL_FILE")
    if not path:
        return None
    try:
        with Path(path).open() as handle:
            reason = handle.read(128).strip()
    except FileNotFoundError:
        return None
    if reason == "sglang_unavailable":
        return {"kind": "sglang_unavailable", "source": "local_supervisor",
                "message": "serving process exited or repeatedly refused connections"}
    return None


def transient_transport(exc):
    if isinstance(exc, HTTPError):
        return exc.code in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, URLError):
        return transient_transport(exc.reason)
    if isinstance(exc, ssl.SSLError):
        return False
    return isinstance(exc, (ConnectionError, TimeoutError, socket.gaierror,
                            http.client.RemoteDisconnected, http.client.IncompleteRead))


def serving_failure(execution):
    """Only the last gateway failure can explain the terminal agent exit."""
    path = Path(execution) / "gateway/codex-r2eact-gateway-events.jsonl"
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            offset = max(0, handle.tell() - 512 * 1024)
            handle.seek(offset)
            if offset:
                handle.readline()
            lines = handle.read().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            event = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        # An earlier transient error followed by a successful response cannot
        # justify replaying a later, unrelated task failure.
        if event.get("event") == "response_committed":
            return None
        if event.get("event") != "gateway_request_failed":
            continue
        detail = str(event.get("error_message", "")) + "\n" + str(event.get("traceback", ""))
        transport_markers = ("ConnectionRefusedError", "ConnectionResetError", "RemoteDisconnected",
                             "IncompleteRead", "CUDA out of memory", "torch.OutOfMemoryError",
                             "SGLang stream interrupted", "SGLang stream closed", "SGLang /abort_request failed",
                             "SGLang stream made no token progress", "SGLang health request timed out")
        # Require a serving-client stack, not arbitrary tool/network failures.
        serving_stack = any(s in detail for s in ("sglang_streaming.py", "sglang_client.py"))
        if serving_stack and any(s in detail for s in transport_markers):
            return {"kind": "sglang_unavailable", "error_type": event.get("error_type"),
                    "message": str(event.get("error_message", ""))[:2000],
                    "observed_at": event.get("observed_at"), "source": str(path)}
        return None
    return None


def recovered_state(state, result, failure, *, now, cooldown=60, limit=MAX_TASK_RECOVERIES):
    count = int(state.get("infrastructure_failures", 0)) + 1
    quarantined = count >= limit
    updated_result = {**result, "status": "interrupted" if not quarantined else "errored",
        "reward": None, "infrastructure_failure": failure,
        "infrastructure_recovery": "quarantined" if quarantined else "retry_pending",
        "session_resume": False, "infrastructure_failures": count,
        "retry_after_epoch": now + cooldown if not quarantined else None}
    updated_state = {**state, "status": "errored" if quarantined else "pending",
        "infrastructure_failures": count, "infrastructure_recovery": updated_result["infrastructure_recovery"],
        "retry_after_epoch": now + cooldown if not quarantined else None}
    return updated_state, updated_result
