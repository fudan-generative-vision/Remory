# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""DASYNC v3 JSON-lines IPC over Unix domain sockets.

The protocol intentionally carries only Python scalars and token-id arrays.  It
must never carry tensors, CUDA IPC handles, KV-cache references, or distributed
process-group metadata.  This keeps the verifier engine world and draft engine
world independent.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
import traceback
from typing import Any, Dict, Optional


class DAsyncIPCError(RuntimeError):
    pass


def _put_drop_oldest(q: "queue.Queue[Dict[str, Any]]", msg: Dict[str, Any]) -> None:
    try:
        q.put_nowait(msg)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except Exception:
        pass
    try:
        q.put_nowait(msg)
    except queue.Full:
        pass


def _encode_msg(msg: Dict[str, Any]) -> bytes:
    return (json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _decode_line(line: bytes) -> Dict[str, Any]:
    obj = json.loads(line.decode("utf-8"))
    if not isinstance(obj, dict):
        raise DAsyncIPCError(f"DASYNC IPC expected a JSON object, got {type(obj).__name__}")
    return obj


class DAsyncUnixSocketClient:
    """Verifier-side client for one external DASYNC draft engine."""

    def __init__(
        self,
        path: str,
        *,
        connect_timeout_s: float,
        queue_size: int = 4096,
    ) -> None:
        if not path:
            raise ValueError("DASYNC IPC path must be non-empty")
        self.path = path
        self._rx_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(4, queue_size))
        self._send_lock = threading.Lock()
        self._closed = threading.Event()
        self._sock: Optional[socket.socket] = None
        self._reader: Optional[threading.Thread] = None
        self._connect(connect_timeout_s=float(connect_timeout_s))

    def _connect(self, *, connect_timeout_s: float) -> None:
        deadline = time.perf_counter() + max(0.0, connect_timeout_s)
        last_exc: Optional[BaseException] = None
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(self.path)
                self._sock = sock
                self._reader = threading.Thread(
                    target=self._reader_loop,
                    name="dasync-ipc-reader",
                    daemon=True,
                )
                self._reader.start()
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last_exc = exc
                try:
                    sock.close()
                except Exception:
                    pass
                if time.perf_counter() >= deadline:
                    raise DAsyncIPCError(
                        f"Timed out connecting to DASYNC draft engine at {self.path!r}. "
                        "Start the draft engine first or increase --dasync-ipc-connect-timeout-ms."
                    ) from last_exc
                time.sleep(0.05)

    def _reader_loop(self) -> None:
        assert self._sock is not None
        try:
            with self._sock.makefile("rb") as fin:
                while not self._closed.is_set():
                    line = fin.readline()
                    if not line:
                        if not self._closed.is_set():
                            _put_drop_oldest(
                                self._rx_q,
                                {
                                    "type": "error",
                                    "source": "dasync_ipc_client",
                                    "traceback": "DASYNC draft engine disconnected.",
                                },
                            )
                        return
                    try:
                        _put_drop_oldest(self._rx_q, _decode_line(line))
                    except Exception:
                        _put_drop_oldest(
                            self._rx_q,
                            {
                                "type": "error",
                                "source": "dasync_ipc_client",
                                "traceback": traceback.format_exc(),
                            },
                        )
        except Exception:
            if not self._closed.is_set():
                _put_drop_oldest(
                    self._rx_q,
                    {
                        "type": "error",
                        "source": "dasync_ipc_client",
                        "traceback": traceback.format_exc(),
                    },
                )

    def send(self, msg: Dict[str, Any]) -> None:
        if self._closed.is_set() or self._sock is None:
            raise DAsyncIPCError("DASYNC IPC client is closed")
        data = _encode_msg(msg)
        with self._send_lock:
            try:
                self._sock.sendall(data)
            except OSError as exc:
                raise DAsyncIPCError("Failed to send DASYNC IPC message") from exc

    def recv_nowait(self) -> Dict[str, Any]:
        return self._rx_q.get_nowait()

    def recv(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._rx_q.get(timeout=timeout)

    def close(self) -> None:
        self._closed.set()
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass


class DAsyncUnixSocketServer:
    """Draft-engine-side JSON-lines server.

    A single verifier connection is active at a time.  If a verifier reconnects,
    the newer connection replaces the older one.  The draft engine loop receives
    verifier commands through ``recv``/``recv_nowait`` and publishes proposals
    through ``send``.
    """

    CLIENT_CONNECTED = "_ipc_client_connected"
    CLIENT_DISCONNECTED = "_ipc_client_disconnected"

    def __init__(self, path: str, *, queue_size: int = 4096) -> None:
        if not path:
            raise ValueError("DASYNC IPC path must be non-empty")
        self.path = path
        self._rx_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(4, queue_size))
        self._closed = threading.Event()
        self._send_lock = threading.Lock()
        self._conn_lock = threading.Lock()
        self._listener: Optional[socket.socket] = None
        self._conn: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(self.path)
        listener.listen(1)
        self._listener = listener
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="dasync-ipc-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def _replace_conn(self, conn: Optional[socket.socket]) -> None:
        with self._conn_lock:
            old = self._conn
            self._conn = conn
        if old is not None:
            try:
                old.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                old.close()
            except Exception:
                pass

    def _accept_loop(self) -> None:
        assert self._listener is not None
        while not self._closed.is_set():
            try:
                conn, _ = self._listener.accept()
            except OSError:
                if not self._closed.is_set():
                    _put_drop_oldest(
                        self._rx_q,
                        {
                            "type": "error",
                            "source": "dasync_ipc_server",
                            "traceback": traceback.format_exc(),
                        },
                    )
                return
            self._replace_conn(conn)
            _put_drop_oldest(self._rx_q, {"type": self.CLIENT_CONNECTED})
            self._read_conn(conn)

    def _read_conn(self, conn: socket.socket) -> None:
        try:
            with conn.makefile("rb") as fin:
                while not self._closed.is_set():
                    line = fin.readline()
                    if not line:
                        break
                    try:
                        _put_drop_oldest(self._rx_q, _decode_line(line))
                    except Exception:
                        _put_drop_oldest(
                            self._rx_q,
                            {
                                "type": "error",
                                "source": "dasync_ipc_server",
                                "traceback": traceback.format_exc(),
                            },
                        )
        finally:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None
            try:
                conn.close()
            except Exception:
                pass
            _put_drop_oldest(self._rx_q, {"type": self.CLIENT_DISCONNECTED})

    def send(self, msg: Dict[str, Any]) -> bool:
        with self._conn_lock:
            conn = self._conn
        if conn is None:
            return False
        data = _encode_msg(msg)
        with self._send_lock:
            try:
                conn.sendall(data)
                return True
            except OSError:
                self._replace_conn(None)
                return False

    def recv_nowait(self) -> Dict[str, Any]:
        return self._rx_q.get_nowait()

    def recv(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self._rx_q.get(timeout=timeout)

    def close(self) -> None:
        self._closed.set()
        self._replace_conn(None)
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
