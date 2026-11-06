"""Per-trial document MCPs with explicit guest-file staging, not host access.

The server schemas/implementations are shared with AutomationBench/JobBench.
PRoot paths are snapshotted into a private CPU sandbox. Saved outputs are
published back to /tmp/r2eact-documents for shell/view_image consumption.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
from urllib.parse import urlsplit

from automation_bench_eval.bridge import MCPServer, OfficialSession
from shared_runtime.runner import document_servers, local_runtime, safe_environment, sandbox_command


GUEST_OUTPUT = "/tmp/r2eact-documents"
MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
INSTRUCTIONS = (
    "For external references use the search MCP (Serper) and shell HTTP tools. "
    "Use Docling MCP to convert documents, then export_docling_document_to_markdown to read them. "
    "If Docling fails or extraction is unusable, use MinerU MCP as a fallback; "
    "MinerU uploads the selected documents to its parsing service. "
    "For images, render documents/pages to local files and call view_image with the file path. "
    "Document tools accept absolute paths inside this task environment, or HTTP(S) URLs. "
    "Saved document outputs are available under /tmp/r2eact-documents. "
    "If a source is a symlink, copy its contents to a regular task file before conversion."
)


@contextlib.contextmanager
def rooted_parent(root, name, *, create=False):
    """Open every path component beneath root without following guest links."""
    path = PurePosixPath(name)
    if not path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
        raise ValueError("Document paths must be absolute task paths without '..'")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:-1]:
            if create:
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd, path.name
    finally:
        os.close(fd)


class DocumentSession:
    # Reuse the bounded pipe reader already exercised by AutomationBench.
    _read = OfficialSession._read

    def __init__(self, name, spec, runtime, rootfs, workspace, uid, artifacts):
        self.name, self.rootfs, self.workspace, self.uid = name, Path(rootfs), workspace, uid
        self.lock, self._buffer = threading.Lock(), b""
        self._source_map = {}
        self.failed = None
        self._stderr = (artifacts / f"{name}-mcp.log").open("w")
        script = workspace / "document_worker.py"
        if not script.exists():
            shutil.copyfile(Path(__file__).with_name("document_worker.py"), script)
            script.chmod(0o444)
        argv = sandbox_command(workspace, uid, Path(spec["command"]), runtime,
                               [spec["command"], str(script)])
        env = safe_environment(workspace, "private-document-child")
        env["CACHE_DIR"] = str(workspace / "exports" / "docling")
        env["OUTPUT_DIR"] = str(workspace / "exports" / "mineru")
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, cwd=workspace, env=env, start_new_session=True, bufsize=0)
        try:
            self._write(spec)
            self.public = self._read(180)["ready"]
        except BaseException:
            self.close()
            raise

    def _write(self, value):
        data = memoryview((json.dumps(value) + "\n").encode())
        while data:
            data = data[os.write(self.process.stdin.fileno(), data):]

    def _source(self, source):
        if not isinstance(source, str):
            raise ValueError("Document source must be a file path or HTTP(S) URL")
        if urlsplit(source).scheme in {"http", "https"}:
            return source
        # Do not interpret file:// URLs or a host-visible absolute path directly.
        with rooted_parent(self.rootfs, source) as (parent, name):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as src:
            info = os.fstat(src.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_DOCUMENT_BYTES:
                raise ValueError("Document must be a regular task file no larger than 256 MiB")
            # Content-addressed snapshots also avoid Docling's path-keyed stale cache.
            temporary = ".source-" + secrets.token_hex(16)
            with rooted_parent(self.workspace, "/inputs/" + temporary) as (target_parent, _):
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o444, dir_fd=target_parent)
                try:
                    digest, size = hashlib.sha256(), 0
                    with os.fdopen(fd, "wb") as dest:
                        os.fchmod(dest.fileno(), 0o444)
                        while block := src.read(1024 * 1024):
                            size += len(block)
                            if size > MAX_DOCUMENT_BYTES:
                                raise ValueError("Document grew beyond the input limit")
                            digest.update(block)
                            dest.write(block)
                    target_name = digest.hexdigest() + Path(source).suffix
                    os.rename(temporary, target_name, src_dir_fd=target_parent, dst_dir_fd=target_parent)
                    target = self.workspace / "inputs" / target_name
                    self._source_map[str(target)] = source
                    return str(target)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(temporary, dir_fd=target_parent)

    def _arguments(self, name, arguments):
        args = deepcopy(arguments)
        if self.name == "docling" and name == "convert_document_into_docling_document":
            args["source"] = self._source(args.get("source"))
        if self.name == "mineru" and name == "parse_documents":
            sources = args.get("file_sources")
            if not isinstance(sources, list):
                raise ValueError("file_sources must be an array")
            args["file_sources"] = [({**s, "source": self._source(s.get("source"))} if isinstance(s, dict)
                                     else self._source(s)) for s in sources]
            destination = args.get("output_dir")
            if destination and (not isinstance(destination, str)
                                or not destination.startswith(GUEST_OUTPUT + "/exports/")):
                raise ValueError("Document output_dir must be under " + GUEST_OUTPUT + "/exports/")
            relative = PurePosixPath(destination).relative_to(GUEST_OUTPUT) if destination else PurePosixPath("exports/mineru")
            if ".." in relative.parts:
                raise ValueError("Invalid output directory")
            args["output_dir"] = str(self.workspace / relative)
        return args

    def _publish(self, result):
        # Only explicitly staged outputs, not caches, credentials or the sandbox
        # program, can enter the task filesystem. Never follow an output link.
        roots = [self.workspace / "exports"]
        for root in roots:
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Document output contains a symlink")
                if not path.is_file():
                    continue
                if path.stat().st_size > MAX_DOCUMENT_BYTES:
                    raise ValueError("Document output exceeds the file limit")
                guest = GUEST_OUTPUT + "/" + path.relative_to(self.workspace).as_posix()
                with rooted_parent(self.rootfs, guest, create=True) as (parent, name):
                    staging = ".document-" + secrets.token_hex(12)
                    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                 0o644, dir_fd=parent)
                    try:
                        with os.fdopen(fd, "wb") as dest:
                            # The parser owns exports/. Protect the privileged
                            # publisher from symlink-swap races in that tree.
                            relative = "/" + path.relative_to(self.workspace).as_posix()
                            with rooted_parent(self.workspace, relative) as (src_parent, src_name):
                                source_fd = os.open(src_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                                    dir_fd=src_parent)
                            with os.fdopen(source_fd, "rb") as src:
                                info = os.fstat(src.fileno())
                                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_DOCUMENT_BYTES:
                                    raise ValueError("Unsafe document output")
                                size = 0
                                while block := src.read(1024 * 1024):
                                    size += len(block)
                                    if size > MAX_DOCUMENT_BYTES:
                                        raise ValueError("Document output grew beyond the file limit")
                                    dest.write(block)
                        os.rename(staging, name, src_dir_fd=parent, dst_dir_fd=parent)
                    finally:
                        with contextlib.suppress(FileNotFoundError):
                            os.unlink(staging, dir_fd=parent)
        # Retain the upstream structured/text result; translate private staging
        # prefixes only, so saved-file references are usable by guest shell tools.
        encoded = json.dumps(result, ensure_ascii=False)
        for source, guest in self._source_map.items():
            # JSON-escape both paths; filenames may contain quotes/backslashes.
            encoded = encoded.replace(json.dumps(source)[1:-1], json.dumps(guest)[1:-1])
        return json.loads(encoded.replace(str(self.workspace), GUEST_OUTPUT))

    def request(self, method, *, name, arguments):
        with self.lock:
            try:
                if self.failed:
                    raise RuntimeError(self.failed)
                args = self._arguments(name, arguments)
                try:
                    self._write({"name": name, "arguments": args})
                    result = self._read(900)["result"]
                except Exception:
                    # A late response must never be mistaken for a later call.
                    self.failed = "Document subprocess transport failed; this MCP session is retired"
                    self.close()
                    raise
                return self._publish(result)
            except Exception as exc:
                return {"isError": True, "content": [{"type": "text", "text":
                    f"Document tool failed ({type(exc).__name__}): {str(exc).replace(str(self.workspace), GUEST_OUTPUT)}"}]}

    def close(self):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self._stderr.close()


class DocumentHTTPServer(MCPServer):
    def handle(self, request):
        status, response = super().handle(request)
        if request.get("method") == "initialize" and response and "result" in response:
            response["result"]["serverInfo"] = {"name": self.session.name, "version": "jobbench-tools-v1"}
        return status, response


class DocumentTools:
    def __init__(self, rootfs, artifacts):
        self.rootfs, self.artifacts = Path(rootfs), Path(artifacts)
        self.token = secrets.token_urlsafe(32)
        self.sessions, self.servers = [], {}
        self.workspace = None
        self.closed = threading.Event()

    def start(self):
        _, runtime = local_runtime()
        self.artifacts.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.workspace = Path(tempfile.mkdtemp(prefix="tb-document-tools-"))
        uid = 60000 + secrets.randbelow(5000) if os.geteuid() == 0 else os.getuid()
        # The child can write only its output/cache subdirectories, not replace
        # the controller-owned input directory or adapter program with a link.
        self.workspace.chmod(0o755)
        for name in ("tmp", "inputs", "exports", ".codex", ".cache"):
            (self.workspace / name).mkdir()
        (self.workspace / "inputs").chmod(0o755)
        for path in self.workspace.iterdir():
            if path.name == "inputs":
                continue
            os.chown(path, uid, uid)
        try:
            for name, spec in document_servers(runtime).items():
                session = DocumentSession(name, spec, runtime, self.rootfs, self.workspace, uid, self.artifacts)
                self.sessions.append(session)
                if self.closed.is_set():
                    raise RuntimeError("Document tools closed during startup")
                server = DocumentHTTPServer(session, self.token)
                self.servers[name] = server
                server.start()
        except BaseException:
            self.close()
            raise
        return self

    def configs(self):
        return {name: {"url": server.url, "bearer_token_env_var": "OPENAI_API_KEY",
                       "required": True, "startup_timeout_sec": 180, "tool_timeout_sec": 900}
                for name, server in self.servers.items()}

    def set_token(self, token):
        self.token = token
        for server in self.servers.values():
            server.token = token

    def close(self):
        self.closed.set()
        # Kill children first: pending HTTP calls must unblock before shutdown.
        for session in self.sessions:
            session.close()
        for server in self.servers.values():
            server.close()
        self.sessions.clear()
        self.servers.clear()
        # Keep the small private staging workspace for failed-conversion diagnosis.
