"""Tools-only MCP facade; official simulator and rubric stay in the controller."""
from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time

from .runtime import official_python, interpreter_environment


class OfficialSession:
    def __init__(self, runtime_root, domain, task_id, artifacts):
        self.lock = threading.Lock()
        self.failed = None
        self._buffer = b""
        self._stderr = (Path(artifacts) / "official-stderr.log").open("w")
        self.process = subprocess.Popen([
            str(official_python(runtime_root)), "-u",
            str(Path(__file__).with_name("official_session.py")),
            "--domain", domain, "--task-id", task_id, "--artifacts", str(artifacts),
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            cwd="/tmp", env={k: v for k, v in interpreter_environment().items()
                             if k != "AUTOMATIONBENCH_STRICT_ASSERTIONS"}, bufsize=0)
        try:
            self.public = self._read(180)["ready"]
        except BaseException:
            self.close()
            raise

    def _read(self, timeout=120):
        deadline = time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Official AutomationBench process did not respond")
                part = os.read(self.process.stdout.fileno(), 65536)
                if not part:
                    raise RuntimeError("Official AutomationBench controller pipe closed")
                self._buffer += part
                if len(self._buffer) > 32 * 1024 * 1024:
                    raise RuntimeError("Oversized official controller response")
        data, self._buffer = self._buffer.split(b"\n", 1)
        response = json.loads(data)
        if "error" in response:
            raise RuntimeError(f"Official AutomationBench error: {response['error']}")
        return response

    def request(self, method, **params):
        with self.lock:
            if self.failed:
                raise RuntimeError(self.failed)
            try:
                data = memoryview((json.dumps({"method": method, **params}) + "\n").encode())
                while data:
                    data = data[os.write(self.process.stdin.fileno(), data):]
                return self._read()["result"]
            except Exception as exc:
                # Do not retry a state-mutating tool with uncertain completion.
                self.failed = f"{type(exc).__name__}: {exc}"
                raise

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self._stderr.close()


class MCPServer:
    def __init__(self, session, token):
        self.session = session
        self.token = token
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value=None):
                data = json.dumps(value, ensure_ascii=False).encode() if value is not None else b""
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply(405)

            def do_POST(self):
                if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + owner.token):
                    self.reply(401)
                    return
                if self.path != "/mcp":
                    self.reply(404)
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4 * 1024 * 1024:
                        raise ValueError("Invalid request size")
                    self.connection.settimeout(30)
                    request = json.loads(self.rfile.read(size))
                    status, response = owner.handle(request)
                    self.reply(status, response)
                except (ValueError, TypeError):
                    self.reply(400)
                except Exception:
                    # Controller diagnostics are private, never expose paths,
                    # assertion content, or a traceback to the evaluated agent.
                    self.reply(500, {"error": "Official simulator unavailable; execution stopped"})

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.http.server_port}/mcp"

    def handle(self, request):
        request_id = request.get("id")
        def result(value):
            return 200, {"jsonrpc": "2.0", "id": request_id, "result": value}
        def error(code, message):
            return 200, {"jsonrpc": "2.0", "id": request_id,
                         "error": {"code": code, "message": message}}
        if request.get("jsonrpc") != "2.0":
            return error(-32600, "Invalid JSON-RPC request")
        method = request.get("method")
        if method in {"notifications/initialized", "notifications/cancelled"}:
            return 202, None
        if method == "initialize":
            return result({"protocolVersion": "2025-06-18", "capabilities": {"tools": {"listChanged": False}},
                           "serverInfo": {"name": "automationbench-official-api", "version": "1.0.0"}})
        if method == "ping":
            return result({})
        if method == "tools/list":
            return result({"tools": self.session.public["tools"]})
        if method == "tools/call":
            params = request.get("params", {})
            if not isinstance(params, dict) or not isinstance(params.get("arguments", {}), dict):
                return error(-32602, "Tool arguments must be an object")
            name = params.get("name")
            if name not in {t["name"] for t in self.session.public["tools"]}:
                return result({"content": [{"type": "text", "text": "Unknown tool"}], "isError": True})
            return result(self.session.request("call", name=name, arguments=params.get("arguments", {})))
        # Critically, no evaluate/get_state/read_assertions MCP methods.
        return error(-32601, "Method not found")

    def start(self):
        self.thread.start()

    def close(self):
        if self.thread.is_alive():
            self.http.shutdown()
            self.thread.join()
        self.http.server_close()
