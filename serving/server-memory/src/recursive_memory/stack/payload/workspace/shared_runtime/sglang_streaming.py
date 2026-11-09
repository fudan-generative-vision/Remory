"""Stream decode tokens without changing the residual encoder or sampling.

Mixed into the pinned SGLang client: its request IDs, cache attestation,
per-request aborts and cancellation lifecycle remain authoritative. Zero-output
hidden-state/encoding requests still use the original non-streaming transport.
"""
from __future__ import annotations

from contextlib import closing
import http.client
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import threading
import time
import urllib.error
import urllib.request


POLICY = "sglang_sse_decode_private_delta_archive_v1"
QUEUE_WAIT_POLICY = "same_request_queue_progress_before_first_token_v1"
_MAX_EVENT_BYTES = 8 * 1024 * 1024


class StreamProtocolError(RuntimeError):
    """An invalid stream must not become an executable or final response."""


class StreamInterrupted(RuntimeError):
    """The transport ended without a terminal generation receipt."""


def _sse_events(response):
    data = []
    size = 0
    while True:
        line = response.readline(_MAX_EVENT_BYTES + 1)
        if not line:
            # Do not parse an unterminated event, even if it happens to be JSON.
            raise StreamInterrupted("SGLang stream closed before a terminal receipt")
        size += len(line)
        if size > _MAX_EVENT_BYTES:
            raise StreamProtocolError("SGLang SSE event exceeds the wire-size limit")
        line = line.rstrip(b"\r\n")
        if not line:
            if data:
                yield b"\n".join(data)
            else:
                yield None  # heartbeats do not extend the no-token deadline
            data, size = [], 0
        elif line.startswith(b"data:"):
            data.append(line[5:].removeprefix(b" "))


def _interruptible_events(response, *, poll_seconds):
    """Read the same SSE connection while its consumer can check queue progress.

    A socket timeout makes HTTPResponse's buffered reader unusable. Do not
    time out and reopen/re-submit a generation just because it is still queued.
    The consumer owns the idle deadline and cancellation; shutdown wakes the
    bounded reader thread on every exit (including a terminal receipt).
    """
    sock = response.fp.raw._sock
    sock.settimeout(None)
    items = queue.Queue(maxsize=1)
    stopping = threading.Event()

    def put(value):
        while not stopping.is_set():
            try:
                items.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def read():
        try:
            for event in _sse_events(response):
                if stopping.is_set():
                    break
                put((True, event))
        except BaseException as exc:
            put((False, exc))

    reader = threading.Thread(target=read, name="sglang-sse-reader", daemon=True)
    reader.start()
    try:
        while True:
            try:
                ok, value = items.get(timeout=poll_seconds)
            except queue.Empty:
                yield None
                continue
            if not ok:
                raise value
            yield value
    finally:
        stopping.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        reader.join(timeout=1)


def _load_snapshot(value):
    """Validate scheduler counters, not a cached throughput/HTTP-health hint."""
    if not isinstance(value, list) or not value:
        return {}
    result = {}
    for row in value:
        if not isinstance(row, dict):
            return {}
        rank = row.get("dp_rank")
        if rank is not None and (type(rank) is not int or rank < 0):
            return {}
        if rank in result:
            return {}
        values = [row.get(k) for k in ("num_reqs", "num_waiting_reqs", "num_tokens")]
        tic = row.get("ts_tic")
        if (any(type(v) is not int or v < 0 for v in values)
            or values[0] < values[1] or isinstance(tic, bool)
            or not isinstance(tic, (int, float)) or not math.isfinite(tic)):
            return {}
        result[rank] = (*values, tic)
    return result


def _queued_peer_progress(previous, current):
    # /get_load.num_reqs includes waiting requests. Require real token growth
    # on the same scheduler while requests are BOTH running and queued.
    # A timestamp alone, static load, 200 OK, or stale throughput is not proof.
    return any(rank in previous and now[0] > now[1] > 0
               and now[2] > previous[rank][2] and now[3] > previous[rank][3]
               for rank, now in current.items())


class GenerationStreamingMixin:
    def _stream_scheduler_load(self, timeout):
        from .residual_preflight import _bounded_get_attempt

        def read(url, seconds):
            # This endpoint is implemented even when Prometheus is disabled.
            # Share the client's origin-scoped authentication and no-redirect policy.
            with self._open_request(urllib.request.Request(url), timeout=seconds) as response:
                payload = response.read(65537)
            if len(payload) > 65536:
                raise ValueError("SGLang load reply is too large")
            return _load_snapshot(json.loads(payload))

        return _bounded_get_attempt(self.base_url + "/get_load", timeout,
            getattr(self, "_residual_stopping", None), reader=read, kind="stream-load")

    def configure_generation_streaming(self, *, artifact_dir, incremental=False,
                                       event_sink=None, cancelled=None):
        if type(incremental) is not bool:
            raise ValueError("SGLang incremental_streaming_output must be boolean")
        self._stream_dir = Path(artifact_dir) / "generation-streams"
        self._stream_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._stream_incremental = incremental
        self._stream_sink = event_sink or (lambda event: None)
        self._stream_cancelled = cancelled or (lambda: False)
        self._stream_context = {}

    def set_generation_stream_context(self, **context):
        # A client is private to one backend and its decodes are serialized.
        self._stream_context = dict(context)

    def _check_stream_open(self):
        stopping = getattr(self, "_residual_stopping", None)
        if self._stream_cancelled() or (stopping is not None and stopping.is_set()):
            raise RuntimeError("SGLang streaming generation cancelled by backend shutdown")

    def _post(self, route, payload, *, request_timeout=None):
        if (not hasattr(self, "_stream_dir") or route != "/generate"
            or payload.get("sampling_params", {}).get("max_new_tokens", 0) <= 0):
            return super()._post(route, payload, request_timeout=request_timeout)
        return self._post_streaming_generate(payload, request_timeout=request_timeout)

    def _post_streaming_generate(self, payload, *, request_timeout=None):
        self._check_stream_open()
        rid = payload.get("rid")
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
            raise ValueError("streaming generation requires a private cancellable request ID")
        outbound = dict(payload, stream=True)
        request = urllib.request.Request(self.base_url + "/generate",
            data=json.dumps(outbound, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"}, method="POST")
        idle_timeout = self.request_timeout if request_timeout is None else request_timeout
        path = self._stream_dir / f"{rid}.jsonl"
        # Never overwrite an old request, and never publish raw text/tokens to a
        # shared pool state or monitor summary. Archive deltas, not O(n²) prefixes.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        tokens, text, meta = [], "", {}
        started = last_progress = last_notification = time.monotonic()
        last_load_check = started - 5.0
        previous_load = {}
        queue_extensions = 0
        frames = 0
        receipt = dict(policy=POLICY, request_id=rid, artifact=str(path),
                       incremental=self._stream_incremental, idle_timeout_seconds=idle_timeout,
                       queue_wait_policy=QUEUE_WAIT_POLICY, **self._stream_context)

        with os.fdopen(fd, "w", encoding="utf-8") as archive:
            def record(event, **fields):
                try:
                    archive.write(json.dumps(dict(event=event, **fields), ensure_ascii=False,
                                             allow_nan=False, separators=(",", ":")) + "\n")
                    archive.flush()
                except OSError as exc:
                    # A disk error is not a recoverable network interruption.
                    raise RuntimeError("Cannot persist SGLang streaming output") from exc

            def notify(status, **fields):
                progress = dict(receipt, status=status, output_tokens=len(tokens), frames=frames,
                    input_tokens=meta.get("prompt_tokens"), seconds=time.monotonic() - started, **fields)
                # Raw output durability is mandatory; optional progress reporting
                # must not interrupt an otherwise valid stream.
                try:
                    self._stream_sink(dict(event="generation_stream_" + status, **progress))
                except Exception:
                    pass
                return progress

            record("started", **receipt, max_new_tokens=outbound["sampling_params"]["max_new_tokens"])
            notify("started")
            try:
                try:
                    with self._open_request(request, timeout=idle_timeout) as response, closing(
                            _interruptible_events(response, poll_seconds=min(0.5, idle_timeout / 4))) as events:
                        if "text/event-stream" not in response.headers.get("Content-Type", ""):
                            raise StreamProtocolError("SGLang ignored stream=True (expected text/event-stream)")
                        for raw in events:
                            self._check_stream_open()
                            now = time.monotonic()
                            if raw is None and not tokens and now - last_load_check >= min(5.0, idle_timeout / 4):
                                last_load_check = now
                                try:
                                    current_load = self._stream_scheduler_load(min(2.0, idle_timeout / 4))
                                except Exception:
                                    current_load = {}  # no evidence: keep the original deadline
                                self._check_stream_open()
                                if _queued_peer_progress(previous_load, current_load):
                                    last_progress = time.monotonic()
                                    queue_extensions += 1
                                    progress = notify("queued", scheduler_load=current_load,
                                                      queue_extensions=queue_extensions)
                                    record("queued", **progress)
                                previous_load = current_load
                            if now - last_progress > idle_timeout:
                                raise TimeoutError("SGLang stream made no token progress within the idle timeout")
                            if raw is None:
                                continue
                            if raw == b"[DONE]":
                                raise StreamInterrupted("SGLang sent DONE without a terminal generation receipt")
                            try:
                                chunk = json.loads(raw)
                            except (UnicodeDecodeError, ValueError) as exc:
                                raise StreamProtocolError("SGLang sent invalid SSE JSON") from exc
                            if not isinstance(chunk, dict) or "error" in chunk:
                                raise StreamProtocolError(f"SGLang SSE error: {chunk!r}")
                            ids, chunk_meta = chunk.get("output_ids"), chunk.get("meta_info")
                            if (not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids)
                                or not isinstance(chunk_meta, dict)):
                                raise StreamProtocolError("SGLang SSE requires integer output_ids and meta_info")
                            if self._stream_incremental:
                                delta = ids
                            else:
                                if ids[:len(tokens)] != tokens:
                                    raise StreamProtocolError("SGLang cumulative stream changed its token prefix")
                                delta = ids[len(tokens):]
                            chunk_text = chunk.get("text", "")
                            if not isinstance(chunk_text, str):
                                raise StreamProtocolError("SGLang stream text must be a string")
                            text_delta = chunk_text if self._stream_incremental else chunk_text[len(text):]
                            text_reset = not self._stream_incremental and not chunk_text.startswith(text)
                            offset = len(tokens)
                            tokens.extend(delta)
                            text = text + chunk_text if self._stream_incremental else chunk_text
                            meta.update(chunk_meta)
                            frames += 1
                            if delta:
                                last_progress = now
                            record("chunk", offset=offset, token_ids=delta,
                                   **({"text": text} if text_reset else {"text_delta": text_delta}),
                                   meta_info=chunk_meta, seconds=now - started)
                            reason = meta.get("finish_reason")
                            if reason is not None:
                                kind = reason.get("type") if isinstance(reason, dict) else reason
                                if kind in {"abort", "aborted", "cancel", "canceled", "cancelled", "error"}:
                                    if isinstance(reason, dict) and reason.get("status_code") == 400:
                                        raise StreamProtocolError(f"SGLang rejected the generation: {reason!r}")
                                    raise StreamInterrupted(f"SGLang ended the stream: {reason!r}")
                                if kind not in {"stop", "length", "max_tokens", "max_new_tokens", "token_limit"}:
                                    raise StreamProtocolError(f"Unknown SGLang finish reason: {reason!r}")
                                # A complete terminal receipt is sufficient; a
                                # lost trailing SSE DONE cannot invalidate it.
                                receipt = notify("completed")
                                break
                            if now - last_notification >= 5:
                                notify("progress")
                                last_notification = now
                except urllib.error.HTTPError as exc:
                    detail = exc.read(8192).decode("utf-8", errors="replace")
                    raise StreamProtocolError(f"SGLang /generate returned HTTP {exc.code}: {detail}") from exc
                except (StreamInterrupted, urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                    self._check_stream_open()  # Ctrl+C/timeout shutdown never continues the agent.
                    if getattr(self, "server_memory", False):
                        # The enclosing cancellable request aborts this rid. Do
                        # not return partial tokens that trigger automatic resume
                        # against a possibly invalid epoch or uncertain lease.
                        raise RuntimeError("Stateful SGLang stream interrupted; explicit recovery required") from exc
                    if not tokens:
                        raise RuntimeError("SGLang stream interrupted before any output tokens") from exc
                    # Abort ONLY this rid. An acknowledged abort is a request to
                    # drain, not evidence that the scheduler has already drained.
                    # If control transport fails, do not start a successor decode.
                    self._abort_request(rid)
                    self._check_stream_open()
                    receipt = notify("interrupted", error_type=type(exc).__name__,
                                     error=str(exc), abort_acknowledged=True)
                    meta["finish_reason"] = dict(type="abort", message="generation stream interrupted")
                self._check_stream_open()
                meta["generation_stream"] = receipt
                record("finished", **receipt, finish_reason=meta.get("finish_reason"))
                os.fsync(archive.fileno())
                return dict(output_ids=tokens, text=text, meta_info=meta)
            except BaseException as exc:
                record("failed", **notify("failed", error_type=type(exc).__name__, error=str(exc)))
                raise
