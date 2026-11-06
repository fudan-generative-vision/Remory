"""Host-side package-command bridge for commands launched inside PRoot.

A small static guest
wrapper is bind-mounted over an explicit set of package-manager paths in the
PRoot view.  The wrapper forwards only argv, cwd, a bounded stdin payload and a
small environment allowlist to this broker over a private Unix socket.  The
broker independently validates the request, maps argv[0] to a fixed guest
absolute path, enters the trial rootfs with ``chroot(2)``, and executes the
physical rootfs binary.  PRoot's bind overlay therefore never applies to the
broker child and cannot recurse back into the wrapper.

Every package command is nested under the trusted static host supervisor used
by the PRoot environment.  That supervisor is a child subreaper and removes
setsid/double-fork descendants after the package command completes.  A broker
never reports cleanup complete merely because the package command's original
process group disappeared.

This remains a compatibility component for trusted benchmark environments,
not a security sandbox.  Callers own its lifecycle and add the bind arguments
returned by ``build_proot_bind_arguments`` to the PRoot command line.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import os
import shutil
import signal
import socket
import stat
import struct
import subprocess
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final


PROTOCOL_VERSION: Final = 1
REQUEST_MAGIC: Final = b"HPKGBR1\0"
RESPONSE_MAGIC: Final = b"HPKGRS1\0"
HEADER_SIZE: Final = 12
MAX_REQUEST_BYTES: Final = 8 * 1024 * 1024
MAX_RESPONSE_BYTES: Final = 64 * 1024 * 1024
MAX_STRING_BYTES: Final = 1024 * 1024
MAX_ARGUMENTS: Final = 512
MAX_ENVIRONMENT_ENTRIES: Final = 32
HANDSHAKE_TIMEOUT_SECONDS: Final = 10.0
RESPONSE_WRITE_TIMEOUT_SECONDS: Final = 10.0
SUPERVISOR_STOP_TIMEOUT_SECONDS: Final = 5.0
GUEST_SOCKET_PATH: Final = PurePosixPath("/run/harbor-package-bridge/control.sock")
_MODULE_DIR: Final = Path(__file__).resolve().parent
GUEST_WRAPPER_SOURCE: Final = _MODULE_DIR / "proot_package_bridge_wrapper.c"
CHROOT_LAUNCHER_SOURCE: Final = _MODULE_DIR / "proot_package_bridge_chroot_launcher.c"

# Both the wrapper and broker enforce this exact basename allowlist.  Values
# are guest absolute paths; no request-controlled path reaches execve().
COMMAND_PATHS: Mapping[str, PurePosixPath] = MappingProxyType(
    {
        "apt": PurePosixPath("/usr/bin/apt"),
        "apt-get": PurePosixPath("/usr/bin/apt-get"),
        "apt-cache": PurePosixPath("/usr/bin/apt-cache"),
        "apt-mark": PurePosixPath("/usr/bin/apt-mark"),
        "dpkg": PurePosixPath("/usr/bin/dpkg"),
        "dpkg-query": PurePosixPath("/usr/bin/dpkg-query"),
        "dpkg-deb": PurePosixPath("/usr/bin/dpkg-deb"),
        "dpkg-divert": PurePosixPath("/usr/bin/dpkg-divert"),
        "dpkg-statoverride": PurePosixPath("/usr/bin/dpkg-statoverride"),
        "debconf-set-selections": PurePosixPath("/usr/bin/debconf-set-selections"),
        "dpkg-reconfigure": PurePosixPath("/usr/sbin/dpkg-reconfigure"),
    }
)

# Query-only commands may run concurrently.  The conservative remainder is
# serialized per broker/rootfs because apt/dpkg share mutable databases and
# maintainer-script state.
READ_ONLY_COMMANDS: Final = frozenset({"apt-cache", "dpkg-query"})
MUTATING_COMMANDS: Final = frozenset(COMMAND_PATHS) - READ_ONLY_COMMANDS

PRESERVED_ENVIRONMENT: Final = (
    "DEBIAN_FRONTEND",
    "DEBIAN_PRIORITY",
    "DEBCONF_NONINTERACTIVE_SEEN",
    "APT_LISTCHANGES_FRONTEND",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
)
PRESERVED_ENVIRONMENT_SET: Final = frozenset(PRESERVED_ENVIRONMENT)
BASE_ENVIRONMENT: Mapping[str, str] = MappingProxyType(
    {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "SHELL": "/bin/sh",
    }
)


class PackageBridgeError(RuntimeError):
    """Base class for a package bridge failure."""


class ProtocolError(PackageBridgeError):
    """The peer sent a malformed or unauthorized wire request."""


class PeerDisconnected(PackageBridgeError):
    """The guest peer disappeared or wrote beyond its single request."""


@dataclass(frozen=True, slots=True)
class PackageRequest:
    argv: tuple[str, ...]
    cwd: str
    env: Mapping[str, str]
    stdin: bytes = b""

    @property
    def command_name(self) -> str:
        return _command_basename(self.argv[0])

    @property
    def mutating(self) -> bool:
        return self.command_name in MUTATING_COMMANDS


@dataclass(frozen=True, slots=True)
class PackageResponse:
    return_code: int
    stdout: bytes
    stderr: bytes


def _command_basename(value: str) -> str:
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise ProtocolError("argv[0] is empty or contains a forbidden byte")
    # PurePosixPath.name deliberately rejects a trailing slash by producing an
    # empty name; argv[0] is a label, never a path selected by the peer.
    name = PurePosixPath(value).name
    if not name:
        raise ProtocolError("argv[0] has no command basename")
    return name


def _validate_text(value: str, *, field: str) -> bytes:
    if not isinstance(value, str) or "\x00" in value:
        raise ProtocolError(f"{field} must be NUL-free text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ProtocolError(f"{field} is not valid UTF-8") from exc
    if len(encoded) > MAX_STRING_BYTES:
        raise ProtocolError(f"{field} exceeds the string length limit")
    return encoded


def _validate_cwd(value: str) -> None:
    encoded = _validate_text(value, field="cwd")
    if not encoded:
        raise ProtocolError("cwd is empty")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ProtocolError("cwd must be an absolute normalized guest path")


def validate_request(request: PackageRequest) -> PackageRequest:
    if not isinstance(request.argv, tuple):
        request = PackageRequest(
            argv=tuple(request.argv),
            cwd=request.cwd,
            env=dict(request.env),
            stdin=request.stdin,
        )
    if not request.argv or len(request.argv) > MAX_ARGUMENTS:
        raise ProtocolError("request has an invalid argument count")
    command_name = _command_basename(request.argv[0])
    if command_name not in COMMAND_PATHS:
        raise ProtocolError(f"package command is not allowed: {command_name!r}")
    for index, argument in enumerate(request.argv):
        _validate_text(argument, field=f"argv[{index}]")
    _validate_cwd(request.cwd)
    if len(request.env) > MAX_ENVIRONMENT_ENTRIES:
        raise ProtocolError("request has too many environment entries")
    normalized_env: dict[str, str] = {}
    for raw_key, raw_value in request.env.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ProtocolError("environment keys and values must be text")
        if raw_key not in PRESERVED_ENVIRONMENT_SET:
            raise ProtocolError(f"environment key is not allowed: {raw_key!r}")
        _validate_text(raw_key, field="environment key")
        _validate_text(raw_value, field=f"environment[{raw_key}]")
        normalized_env[raw_key] = raw_value
    if not isinstance(request.stdin, bytes):
        raise ProtocolError("stdin must be bytes")
    return PackageRequest(
        argv=tuple(request.argv),
        cwd=request.cwd,
        env=MappingProxyType(normalized_env),
        stdin=request.stdin,
    )


def _pack_u16(value: int) -> bytes:
    return struct.pack("!H", value)


def _pack_u32(value: int) -> bytes:
    return struct.pack("!I", value)


def _pack_text(value: str, *, field: str) -> bytes:
    encoded = _validate_text(value, field=field)
    return _pack_u32(len(encoded)) + encoded


def _pack_blob(value: bytes, *, field: str) -> bytes:
    if not isinstance(value, bytes):
        raise ProtocolError(f"{field} must be bytes")
    if len(value) > MAX_REQUEST_BYTES:
        raise ProtocolError(f"{field} exceeds the request length limit")
    return _pack_u32(len(value)) + value


def encode_request(request: PackageRequest) -> bytes:
    request = validate_request(request)
    body = bytearray(_pack_u16(len(request.argv)))
    for index, argument in enumerate(request.argv):
        body.extend(_pack_text(argument, field=f"argv[{index}]"))
    body.extend(_pack_text(request.cwd, field="cwd"))
    body.extend(_pack_u16(len(request.env)))
    for key, value in request.env.items():
        body.extend(_pack_text(key, field="environment key"))
        body.extend(_pack_text(value, field=f"environment[{key}]"))
    body.extend(_pack_blob(request.stdin, field="stdin"))
    if len(body) > MAX_REQUEST_BYTES:
        raise ProtocolError("encoded request exceeds the request length limit")
    return REQUEST_MAGIC + _pack_u32(len(body)) + bytes(body)


class _BodyReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.offset = 0

    def take(self, size: int) -> bytes:
        if size < 0 or self.offset + size > len(self.payload):
            raise ProtocolError("request body is truncated")
        value = self.payload[self.offset : self.offset + size]
        self.offset += size
        return value

    def u16(self) -> int:
        return struct.unpack("!H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("!I", self.take(4))[0]

    def blob(self, *, field: str, maximum: int = MAX_STRING_BYTES) -> bytes:
        size = self.u32()
        if size > maximum:
            raise ProtocolError(f"{field} exceeds its length limit")
        return self.take(size)

    def text(self, *, field: str) -> str:
        raw = self.blob(field=field)
        if b"\x00" in raw:
            raise ProtocolError(f"{field} contains NUL")
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ProtocolError(f"{field} is not valid UTF-8") from exc

    def finish(self) -> None:
        if self.offset != len(self.payload):
            raise ProtocolError("request body contains trailing bytes")


def decode_request_frame(frame: bytes) -> PackageRequest:
    if len(frame) < HEADER_SIZE:
        raise ProtocolError("request header is truncated")
    if frame[:8] != REQUEST_MAGIC:
        raise ProtocolError("request magic/version mismatch")
    body_size = struct.unpack("!I", frame[8:12])[0]
    if body_size > MAX_REQUEST_BYTES:
        raise ProtocolError("announced request exceeds the request length limit")
    if len(frame) != HEADER_SIZE + body_size:
        raise ProtocolError("request frame length does not match its header")
    reader = _BodyReader(frame[HEADER_SIZE:])
    argument_count = reader.u16()
    if argument_count < 1 or argument_count > MAX_ARGUMENTS:
        raise ProtocolError("request has an invalid argument count")
    argv = tuple(reader.text(field=f"argv[{index}]") for index in range(argument_count))
    cwd = reader.text(field="cwd")
    environment_count = reader.u16()
    if environment_count > MAX_ENVIRONMENT_ENTRIES:
        raise ProtocolError("request has too many environment entries")
    env: dict[str, str] = {}
    for _ in range(environment_count):
        key = reader.text(field="environment key")
        value = reader.text(field=f"environment[{key}]")
        if key in env:
            raise ProtocolError(f"duplicate environment key: {key!r}")
        env[key] = value
    stdin = reader.blob(field="stdin", maximum=MAX_REQUEST_BYTES)
    reader.finish()
    return validate_request(PackageRequest(argv=argv, cwd=cwd, env=env, stdin=stdin))


def encode_response(response: PackageResponse) -> bytes:
    if not isinstance(response.return_code, int) or isinstance(
        response.return_code, bool
    ):
        raise ProtocolError("response return code must be an integer")
    if not -(2**31) <= response.return_code < 2**31:
        raise ProtocolError("response return code is outside int32")
    if not isinstance(response.stdout, bytes) or not isinstance(response.stderr, bytes):
        raise ProtocolError("response output fields must be bytes")
    body = (
        struct.pack(
            "!iII", response.return_code, len(response.stdout), len(response.stderr)
        )
        + response.stdout
        + response.stderr
    )
    if len(body) > MAX_RESPONSE_BYTES:
        raise ProtocolError("encoded response exceeds the response length limit")
    return RESPONSE_MAGIC + _pack_u32(len(body)) + body


def decode_response_frame(frame: bytes) -> PackageResponse:
    if len(frame) < HEADER_SIZE or frame[:8] != RESPONSE_MAGIC:
        raise ProtocolError("response magic/version mismatch or truncated header")
    body_size = struct.unpack("!I", frame[8:12])[0]
    if body_size > MAX_RESPONSE_BYTES or len(frame) != HEADER_SIZE + body_size:
        raise ProtocolError("response frame length is invalid")
    body = frame[HEADER_SIZE:]
    if len(body) < 12:
        raise ProtocolError("response body is truncated")
    return_code, stdout_size, stderr_size = struct.unpack("!iII", body[:12])
    if 12 + stdout_size + stderr_size != len(body):
        raise ProtocolError("response output lengths are invalid")
    return PackageResponse(
        return_code=return_code,
        stdout=body[12 : 12 + stdout_size],
        stderr=body[12 + stdout_size :],
    )


def _rootfs_target(rootfs: Path, guest_path: PurePosixPath) -> Path:
    return rootfs.joinpath(*guest_path.parts[1:])


def _build_static_helper(
    *,
    source: Path,
    cache_dir: Path,
    name: str,
    description: str,
    compiler: str | None = None,
) -> Path:
    """Compile one content-addressed helper without exposing a partial binary."""

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    destination_dir = Path(cache_dir).expanduser().absolute()
    destination_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory_info = destination_dir.lstat()
    if (
        not stat.S_ISDIR(directory_info.st_mode)
        or stat.S_ISLNK(directory_info.st_mode)
        or directory_info.st_uid != os.getuid()
    ):
        raise PackageBridgeError(
            f"{description} cache must be a real directory owned by this uid"
        )
    # Existing shared caches may predate the bridge.  Tighten an owned cache
    # before trusting or creating any executable within it.
    destination_dir.chmod(0o700)
    if stat.S_IMODE(destination_dir.stat().st_mode) != 0o700:
        raise PackageBridgeError(f"{description} cache is not mode 0700")
    binary = destination_dir / f"{name}-{source_hash}"
    compiler_path = compiler or shutil.which("gcc")
    if not compiler_path:
        raise PackageBridgeError(f"gcc is required to compile the {description}")
    lock_path = destination_dir / f".{binary.name}.lock"
    lock_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        lock_info = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != os.getuid()
            or lock_info.st_nlink != 1
        ):
            raise PackageBridgeError(f"{description} cache lock is unsafe")
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        temporary = destination_dir / f".{binary.name}-{uuid.uuid4().hex}.tmp"
        try:
            completed = subprocess.run(
                [
                    compiler_path,
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-static",
                    "-o",
                    str(temporary),
                    str(source),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode != 0:
                raise PackageBridgeError(
                    f"failed to compile the {description}:\n"
                    + completed.stderr.decode(errors="replace")
                )
            temporary.chmod(0o700)
            expected_digest = hashlib.sha256(temporary.read_bytes()).digest()
            if binary.exists() or binary.is_symlink():
                cached_info = binary.lstat()
                cached_safe = bool(
                    stat.S_ISREG(cached_info.st_mode)
                    and not stat.S_ISLNK(cached_info.st_mode)
                    and cached_info.st_uid == os.getuid()
                    and cached_info.st_nlink == 1
                    and stat.S_IMODE(cached_info.st_mode) == 0o700
                )
                if cached_safe:
                    cached_digest = hashlib.sha256(binary.read_bytes()).digest()
                    if cached_digest == expected_digest:
                        return binary
            os.replace(temporary, binary)
            installed_info = binary.lstat()
            if (
                not stat.S_ISREG(installed_info.st_mode)
                or installed_info.st_uid != os.getuid()
                or installed_info.st_nlink != 1
                or stat.S_IMODE(installed_info.st_mode) != 0o700
                or hashlib.sha256(binary.read_bytes()).digest() != expected_digest
            ):
                raise PackageBridgeError(
                    f"installed {description} failed integrity verification"
                )
            return binary
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        os.close(lock_fd)


def build_guest_wrapper(*, cache_dir: Path, compiler: str | None = None) -> Path:
    """Compile and content-address the static helper bind-mounted in the guest."""

    return _build_static_helper(
        source=GUEST_WRAPPER_SOURCE,
        cache_dir=cache_dir,
        name="proot-package-guest-wrapper",
        description="package bridge guest wrapper",
        compiler=compiler,
    )


def build_chroot_launcher(*, cache_dir: Path, compiler: str | None = None) -> Path:
    """Compile and content-address the trusted host-side chroot launcher."""

    return _build_static_helper(
        source=CHROOT_LAUNCHER_SOURCE,
        cache_dir=cache_dir,
        name="proot-package-chroot-launcher",
        description="package bridge chroot launcher",
        compiler=compiler,
    )


def prepare_guest_bind_directory(rootfs: Path) -> Path:
    """Create only the safe guest parent needed by the PRoot socket bind."""

    root = Path(rootfs).expanduser().resolve()
    if root == Path("/") or root.is_symlink() or not root.is_dir():
        raise PackageBridgeError("rootfs must be a non-root real directory")
    parent = _rootfs_target(root, GUEST_SOCKET_PATH.parent)
    cursor = root
    for component in GUEST_SOCKET_PATH.parent.parts[1:]:
        cursor = cursor / component
        if cursor.exists() or cursor.is_symlink():
            info = cursor.lstat()
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise PackageBridgeError(
                    f"guest package bridge parent is unsafe: /{component}"
                )
        else:
            cursor.mkdir(mode=0o755)
    parent.chmod(0o755)
    return parent


def build_proot_bind_arguments(
    *,
    rootfs: Path,
    wrapper_path: Path,
    socket_path: Path,
    command_paths: Mapping[str, PurePosixPath] = COMMAND_PATHS,
) -> list[str]:
    """Return PRoot-only overlays without modifying the physical rootfs.

    Only commands physically present as regular files are overlaid.  The
    broker's direct chroot consequently resolves the original rootfs binaries,
    while every descendant in the PRoot view resolves the wrapper.
    """

    root = Path(rootfs).expanduser().resolve()
    wrapper = Path(wrapper_path).expanduser().resolve()
    host_socket = Path(socket_path).expanduser().absolute()
    if root == Path("/") or root.is_symlink() or not root.is_dir():
        raise PackageBridgeError("rootfs must be a non-root real directory")
    if wrapper.is_symlink() or not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        raise PackageBridgeError("guest package wrapper must be an executable file")
    try:
        socket_info = host_socket.stat()
    except OSError as exc:
        raise PackageBridgeError("package bridge socket does not exist") from exc
    if not stat.S_ISSOCK(socket_info.st_mode):
        raise PackageBridgeError("package bridge endpoint is not a Unix socket")
    guest_parent = _rootfs_target(root, GUEST_SOCKET_PATH.parent)
    try:
        guest_parent_info = guest_parent.lstat()
    except OSError as exc:
        raise PackageBridgeError(
            "guest socket parent is absent; call prepare_guest_bind_directory first"
        ) from exc
    if not stat.S_ISDIR(guest_parent_info.st_mode) or stat.S_ISLNK(
        guest_parent_info.st_mode
    ):
        raise PackageBridgeError("guest socket parent is not a safe directory")

    arguments: list[str] = []
    wrapper_info = wrapper.stat()
    for command_name, guest_path in command_paths.items():
        if command_name != guest_path.name or not guest_path.is_absolute():
            raise PackageBridgeError(
                "command mapping is not basename/absolute-path fixed"
            )
        target = _rootfs_target(root, guest_path)
        try:
            target_info = target.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(target_info.st_mode):
            raise PackageBridgeError(
                f"rootfs command target is not regular: {guest_path}"
            )
        if (target_info.st_dev, target_info.st_ino) == (
            wrapper_info.st_dev,
            wrapper_info.st_ino,
        ):
            raise PackageBridgeError(
                f"rootfs command already aliases the package wrapper: {guest_path}"
            )
        arguments.extend(("-b", f"{wrapper}:{guest_path}"))
    if not arguments:
        raise PackageBridgeError("rootfs has none of the approved package commands")
    arguments.extend(("-b", f"{host_socket}:{GUEST_SOCKET_PATH}"))
    return arguments


class PackageCommandBroker:
    """One private Unix-socket broker bound to one immutable rootfs path."""

    def __init__(
        self,
        *,
        rootfs: Path,
        socket_path: Path,
        launcher_path: Path,
        supervisor_path: Path,
        allowed_peer_uid: int | None = None,
        command_paths: Mapping[str, PurePosixPath] = COMMAND_PATHS,
        handshake_timeout_seconds: float = HANDSHAKE_TIMEOUT_SECONDS,
        allow_socket_within_rootfs_for_tests: bool = False,
        package_lock: asyncio.Lock | None = None,
    ) -> None:
        root = Path(rootfs).expanduser().resolve()
        endpoint = Path(socket_path).expanduser().absolute()
        launcher = Path(launcher_path).expanduser().resolve()
        supervisor = Path(supervisor_path).expanduser().resolve()
        if root == Path("/") or root.is_symlink() or not root.is_dir():
            raise PackageBridgeError("rootfs must be a non-root real directory")
        if not allow_socket_within_rootfs_for_tests and endpoint.is_relative_to(root):
            raise PackageBridgeError("broker socket must remain outside the rootfs")
        if handshake_timeout_seconds <= 0:
            raise PackageBridgeError("handshake timeout must be positive")
        if (
            launcher.is_symlink()
            or not launcher.is_file()
            or not os.access(launcher, os.X_OK)
        ):
            raise PackageBridgeError("chroot launcher must be an executable file")
        if launcher.is_relative_to(root):
            raise PackageBridgeError("chroot launcher must remain outside the rootfs")
        if (
            supervisor.is_symlink()
            or not supervisor.is_file()
            or not os.access(supervisor, os.X_OK)
        ):
            raise PackageBridgeError("host supervisor must be an executable file")
        if supervisor.is_relative_to(root):
            raise PackageBridgeError("host supervisor must remain outside the rootfs")
        mappings = dict(command_paths)
        if not mappings or set(mappings) - set(COMMAND_PATHS):
            raise PackageBridgeError("broker command mapping exceeds the allowlist")
        for name, path in mappings.items():
            if name != path.name or not path.is_absolute():
                raise PackageBridgeError("broker command mapping is not fixed")
        self.rootfs = root
        self.socket_path = endpoint
        self.launcher_path = launcher
        self.supervisor_path = supervisor
        self.allowed_peer_uid = (
            os.getuid() if allowed_peer_uid is None else allowed_peer_uid
        )
        self.command_paths = MappingProxyType(mappings)
        self.handshake_timeout_seconds = float(handshake_timeout_seconds)
        # Full serialization is deliberately stricter than apt/dpkg require.
        # It prevents read-only dpkg queries observing an in-flight database
        # replacement and avoids a writer-starvation-prone custom RW lock.
        self._package_lock = (
            package_lock if package_lock is not None else asyncio.Lock()
        )
        self._server: asyncio.AbstractServer | None = None
        self._handlers: set[asyncio.Task[None]] = set()
        self._processes: set[asyncio.subprocess.Process] = set()
        self._process_pidfds: dict[asyncio.subprocess.Process, int] = {}
        self._process_procfds: dict[asyncio.subprocess.Process, int] = {}
        self.events: list[dict[str, object]] = []
        self._fatal_error: str | None = None
        self._stopping = False

    @property
    def is_running(self) -> bool:
        server = self._server
        return bool(server is not None and server.is_serving())

    @property
    def cleanup_complete(self) -> bool:
        return bool(
            self._fatal_error is None
            and self._server is None
            and not self._handlers
            and not self._processes
            and not self._process_pidfds
            and not self._process_procfds
            and not (self.socket_path.exists() or self.socket_path.is_symlink())
        )

    @property
    def healthy(self) -> bool:
        return self._fatal_error is None

    @property
    def fatal_error(self) -> str | None:
        return self._fatal_error

    async def __aenter__(self) -> PackageCommandBroker:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    def _record(self, event: str, **detail: object) -> None:
        self.events.append({"event": event, **detail})

    def _mark_fatal(self, reason: str) -> None:
        if self._fatal_error is None:
            self._fatal_error = reason[:1_024]
            self._record("broker_fatal", error_type=self._fatal_error)

    def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._handlers.add(task)
        task.add_done_callback(self._consume_handler_result)

    def _consume_handler_result(self, task: asyncio.Task[None]) -> None:
        self._handlers.discard(task)
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.exception()

    async def start(self) -> None:
        if self._server is not None:
            raise PackageBridgeError("package bridge broker is already started")
        if not self.healthy:
            raise PackageBridgeError(
                "package bridge broker cannot restart after an internal failure"
            )
        self._stopping = False
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent_info = self.socket_path.parent.lstat()
        if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
            raise PackageBridgeError("broker socket parent is not a real directory")
        self.socket_path.parent.chmod(0o700)
        if self.socket_path.exists() or self.socket_path.is_symlink():
            info = self.socket_path.lstat()
            if not stat.S_ISSOCK(info.st_mode):
                raise PackageBridgeError("refusing to replace a non-socket endpoint")
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._accept,
            path=str(self.socket_path),
            start_serving=True,
        )
        self.socket_path.chmod(0o600)
        self._record("broker_started", socket_path=str(self.socket_path))

    async def stop(self) -> None:
        self._stopping = True
        server, self._server = self._server, None
        if server is not None:
            server.close()
        handlers = tuple(self._handlers)
        cleanup_errors: list[BaseException] = []
        # Ask each dedicated subreaper to retire its own tree before canceling
        # the handler that is concurrently draining its protocol pipes.
        for process in tuple(self._processes):
            try:
                await self._terminate_supervised_process(process)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if handlers:
            handler_group = asyncio.gather(*handlers, return_exceptions=True)
            try:
                await asyncio.wait_for(
                    asyncio.shield(handler_group),
                    timeout=SUPERVISOR_STOP_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                for task in handlers:
                    task.cancel()
                try:
                    await asyncio.wait_for(
                        handler_group, timeout=SUPERVISOR_STOP_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError as exc:
                    self._mark_fatal("broker_handler_cleanup_unproven")
                    cleanup_errors.append(exc)
            self._handlers.difference_update(handlers)
        if server is not None:
            try:
                await asyncio.wait_for(server.wait_closed(), timeout=1.0)
            except asyncio.TimeoutError as exc:
                self._mark_fatal("broker_listener_cleanup_unproven")
                cleanup_errors.append(exc)
        for process in tuple(self._processes):
            try:
                await self._terminate_supervised_process(process)
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            if self.socket_path.is_socket():
                self.socket_path.unlink()
        except OSError:
            pass
        self._record("broker_stopped")
        if cleanup_errors or self._processes:
            self._mark_fatal("package_process_cleanup_unproven")
            raise PackageBridgeError(
                "package bridge stopped without proving package-process cleanup"
            ) from (cleanup_errors[0] if cleanup_errors else None)

    def _peer_credentials(self, writer: asyncio.StreamWriter) -> tuple[int, int, int]:
        transport_socket = writer.get_extra_info("socket")
        if transport_socket is None or not hasattr(socket, "SO_PEERCRED"):
            raise ProtocolError("Unix peer credentials are unavailable")
        raw = transport_socket.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        )
        return struct.unpack("3i", raw)

    async def _read_request(self, reader: asyncio.StreamReader) -> PackageRequest:
        header = await reader.readexactly(HEADER_SIZE)
        if header[:8] != REQUEST_MAGIC:
            raise ProtocolError("request magic/version mismatch")
        body_size = struct.unpack("!I", header[8:])[0]
        if body_size > MAX_REQUEST_BYTES:
            raise ProtocolError("announced request exceeds the request length limit")
        body = await reader.readexactly(body_size)
        return decode_request_frame(header + body)

    async def _write_response(
        self, writer: asyncio.StreamWriter, response: PackageResponse
    ) -> None:
        writer.write(encode_response(response))
        await writer.drain()

    async def _write_rejection(
        self, writer: asyncio.StreamWriter, error: BaseException
    ) -> None:
        message = f"package bridge rejected request: {type(error).__name__}: {error}\n"
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                self._write_response(
                    writer,
                    PackageResponse(
                        return_code=126, stdout=b"", stderr=message.encode()
                    ),
                ),
                timeout=RESPONSE_WRITE_TIMEOUT_SECONDS,
            )

    @staticmethod
    async def _peer_event(reader: asyncio.StreamReader) -> bytes:
        # EOF means disconnect.  Any byte means protocol smuggling because one
        # connection carries exactly one request and one response.
        return await reader.read(1)

    async def _acquire_package_lock(self, peer_event: asyncio.Task[bytes]) -> bool:
        acquire = asyncio.create_task(self._package_lock.acquire())
        try:
            done, _ = await asyncio.wait(
                {acquire, peer_event}, return_when=asyncio.FIRST_COMPLETED
            )
            if peer_event in done:
                # Both futures may become ready in the same loop iteration. If
                # acquire already succeeded, explicitly release it; canceling a
                # completed Task would otherwise leak the broker-wide lock.
                if (
                    acquire.done()
                    and not acquire.cancelled()
                    and acquire.exception() is None
                ):
                    if acquire.result():
                        self._package_lock.release()
                else:
                    acquire.cancel()
                    with contextlib.suppress(BaseException):
                        await acquire
                raise PeerDisconnected("peer left or sent trailing request bytes")
            await acquire
            return True
        except BaseException:
            if not acquire.done():
                acquire.cancel()
                with contextlib.suppress(BaseException):
                    await acquire
            raise

    def _command_target(self, command_name: str) -> PurePosixPath:
        target = self.command_paths.get(command_name)
        if target is None:
            raise ProtocolError(f"package command is not configured: {command_name!r}")
        physical = _rootfs_target(self.rootfs, target)
        try:
            info = physical.stat(follow_symlinks=False)
        except OSError as exc:
            raise PackageBridgeError(
                f"rootfs command is unavailable: {target}"
            ) from exc
        if not stat.S_ISREG(info.st_mode):
            raise PackageBridgeError(f"rootfs command is not regular: {target}")
        return target

    async def _spawn(
        self, request: PackageRequest, target: PurePosixPath
    ) -> asyncio.subprocess.Process:
        environment = dict(BASE_ENVIRONMENT)
        environment.update(request.env)
        # The static launcher performs chroot/chdir/execve after fork.  This
        # avoids Python preexec_fn, which is unsafe once the host process has
        # any threads.  No shell interprets any request field.
        gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
        try:
            process = await asyncio.create_subprocess_exec(
                str(self.supervisor_path),
                "--supervise",
                str(self.launcher_path),
                "--identity-gate-fd",
                str(gate_read),
                str(self.rootfs),
                request.cwd,
                str(target),
                *request.argv,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                pass_fds=(gate_read,),
            )
        except BaseException:
            os.close(gate_read)
            os.close(gate_write)
            raise
        os.close(gate_read)
        self._processes.add(process)
        pidfd: int | None = None
        procfd: int | None = None
        identity_registered = False
        try:
            pidfd = os.pidfd_open(process.pid, 0)
            self._prove_live_child_pidfd(pidfd)
            procfd = os.open(
                f"/proc/{process.pid}",
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            # Both descriptors are acquired immediately after spawn. The pidfd
            # pins identity; the proc dirfd is the only permitted anchor for
            # any future ownership inspection (never reopen by numeric PID).
            stat_fd = os.open(
                "stat", os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=procfd
            )
            try:
                stat_pid = int(os.read(stat_fd, 64).split(b" ", 1)[0])
            finally:
                os.close(stat_fd)
            if stat_pid != process.pid:
                raise PackageBridgeError(
                    "host package supervisor proc identity does not match its pidfd"
                )
            self._prove_live_child_pidfd(pidfd)
            self._process_pidfds[process] = pidfd
            self._process_procfds[process] = procfd
            identity_registered = True
            written = os.write(gate_write, b"G")
            if written != 1:
                raise PackageBridgeError(
                    "host package supervisor identity gate was not released"
                )
        except (OSError, ValueError, PackageBridgeError) as exc:
            # Once registered, retain the identity descriptors.  They are the
            # only safe way for stop() to retire or quarantine this owner; a
            # cleanup path must never reopen the numeric PID after a failure.
            if pidfd is not None and not identity_registered:
                with contextlib.suppress(OSError):
                    os.close(pidfd)
            if procfd is not None and not identity_registered:
                with contextlib.suppress(OSError):
                    os.close(procfd)
            self._mark_fatal("package_supervisor_pidfd_open_failed")
            raise PackageBridgeError(
                "unable to pin host package supervisor process identity"
            ) from exc
        finally:
            os.close(gate_write)
        return process

    @staticmethod
    def _prove_live_child_pidfd(pidfd: int) -> None:
        """Prove the pinned identity is still this process's live child.

        A pidfd alone prevents signalling a reused numeric PID, but it does not
        prove the descriptor refers to a child still owned by this broker.
        ``WNOWAIT`` performs that ownership/liveness check without consuming
        the exit status expected by asyncio's child watcher.
        """

        try:
            result = os.waitid(
                os.P_PIDFD,
                pidfd,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except ChildProcessError as exc:
            raise PackageBridgeError(
                "host package supervisor pidfd is not a broker-owned child"
            ) from exc
        if result is not None:
            raise PackageBridgeError(
                "host package supervisor exited before identity pinning completed"
            )

    async def _communicate_bounded(
        self, process: asyncio.subprocess.Process, stdin: bytes
    ) -> tuple[bytes, bytes]:
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        remaining = [MAX_RESPONSE_BYTES - 12]

        async def feed() -> None:
            try:
                process.stdin.write(stdin)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()

        async def collect(stream: asyncio.StreamReader, field: str) -> bytes:
            chunks: list[bytes] = []
            while True:
                chunk = await stream.read(64 * 1024)
                if not chunk:
                    return b"".join(chunks)
                remaining[0] -= len(chunk)
                if remaining[0] < 0:
                    raise PackageBridgeError(
                        f"combined command output exceeds {MAX_RESPONSE_BYTES} bytes"
                    )
                chunks.append(chunk)

        stdin_task = asyncio.create_task(feed())
        stdout_task = asyncio.create_task(collect(process.stdout, "stdout"))
        stderr_task = asyncio.create_task(collect(process.stderr, "stderr"))
        wait_task = asyncio.create_task(process.wait())
        tasks = (stdin_task, stdout_task, stderr_task, wait_task)
        try:
            await asyncio.gather(*tasks)
            return stdout_task.result(), stderr_task.result()
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    @staticmethod
    def _close_process_pipes(process: asyncio.subprocess.Process) -> None:
        # asyncio's Process.wait() reaps the child but does not guarantee that
        # canceled StreamReader transports have observed EOF.  Close all pipe
        # transports explicitly so their __del__ cannot touch a closed loop.
        transport = getattr(process, "_transport", None)
        getter = getattr(transport, "get_pipe_transport", None)
        if callable(getter):
            for descriptor in (0, 1, 2):
                with contextlib.suppress(Exception):
                    pipe = getter(descriptor)
                    if pipe is not None:
                        pipe.close()

    @classmethod
    def _close_process_transport(cls, process: asyncio.subprocess.Process) -> None:
        cls._close_process_pipes(process)
        transport = getattr(process, "_transport", None)
        close_transport = getattr(transport, "close", None)
        if callable(close_transport):
            with contextlib.suppress(Exception):
                close_transport()

    async def _retire_completed_process(
        self, process: asyncio.subprocess.Process
    ) -> None:
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                timeout=SUPERVISOR_STOP_TIMEOUT_SECONDS,
            )
        except (asyncio.TimeoutError, ProcessLookupError) as exc:
            raise PackageBridgeError(
                "host package supervisor did not publish its exit status"
            ) from exc
        if process.returncode is None:
            raise PackageBridgeError("host package supervisor exit is unconfirmed")
        self._close_process_transport(process)
        self._processes.discard(process)
        pidfd = self._process_pidfds.pop(process, None)
        if pidfd is not None:
            os.close(pidfd)
        procfd = self._process_procfds.pop(process, None)
        if procfd is not None:
            os.close(procfd)

    async def _terminate_supervised_process(
        self, process: asyncio.subprocess.Process
    ) -> None:
        if process.returncode is not None:
            await self._retire_completed_process(process)
            return
        pidfd = self._process_pidfds.get(process)
        procfd = self._process_procfds.get(process)
        if pidfd is None or procfd is None:
            self._mark_fatal("package_supervisor_pidfd_missing")
            raise PackageBridgeError(
                "host package supervisor process identity is not pinned"
            )
        # Signal only the subreaper.  Killing its process group would kill the
        # one process that can adopt and reap a setsid/double-fork descendant.
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
            signal.pidfd_send_signal(pidfd, signal.SIGCONT)
        except ProcessLookupError:
            pass
        except (OSError, PermissionError) as exc:
            self._mark_fatal("package_supervisor_pidfd_signal_failed")
            raise PackageBridgeError(
                "permission denied signaling host package supervisor"
            ) from exc
        # An output-limit cancellation can leave asyncio's read transport
        # paused.  Closing only the protocol pipes lets the primary observe
        # EOF/SIGPIPE and lets Process.wait() finish.  Do not close the process
        # transport itself here: it may kill the owner subreaper before that
        # owner has adopted and reaped a setsid/double-fork descendant.
        self._close_process_pipes(process)
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()),
                timeout=SUPERVISOR_STOP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            # Never kill the owner subreaper first. Without an acknowledged
            # supervisor exit, quarantine the process/rootfs for diagnosis.
            self._mark_fatal("package_process_cleanup_unproven")
            raise PackageBridgeError(
                "host package supervisor cleanup could not be proven"
            )
        await self._retire_completed_process(process)

    async def _kill_process_group(self, process: asyncio.subprocess.Process) -> None:
        """Compatibility name with descendant-safe supervisor semantics."""

        await self._terminate_supervised_process(process)

    async def _run_request(
        self,
        request: PackageRequest,
        peer_event: asyncio.Task[bytes],
    ) -> PackageResponse:
        if self._stopping:
            raise asyncio.CancelledError
        target = self._command_target(request.command_name)
        process = await self._spawn(request, target)
        if self._stopping:
            await self._kill_process_group(process)
            raise asyncio.CancelledError
        communicate = asyncio.create_task(
            self._communicate_bounded(process, request.stdin)
        )
        try:
            done, _ = await asyncio.wait(
                {communicate, peer_event}, return_when=asyncio.FIRST_COMPLETED
            )
            if peer_event in done:
                communicate.cancel()
                with contextlib.suppress(BaseException):
                    await communicate
                await self._kill_process_group(process)
                raise PeerDisconnected("peer left or sent trailing request bytes")
            stdout, stderr = await communicate
            await self._retire_completed_process(process)
            if self._stopping:
                raise asyncio.CancelledError
            if process.returncode is None or process.returncode < 0:
                self._mark_fatal("package_supervisor_abnormal_exit")
                raise PackageBridgeError("host package supervisor exited abnormally")
            if (
                b"proot_rpc_server fatal:" in stderr
                or b"proot_rpc_server supervisor exec failed" in stderr
            ):
                self._mark_fatal("package_supervisor_internal_failure")
                raise PackageBridgeError("host package supervisor reported failure")
            response = PackageResponse(
                return_code=process.returncode or 0,
                stdout=stdout,
                stderr=stderr,
            )
            # Validate the bound before attempting any socket write.
            encode_response(response)
            return response
        except BaseException:
            if not communicate.done():
                communicate.cancel()
            with contextlib.suppress(BaseException):
                await communicate
            await self._kill_process_group(process)
            raise

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer_event: asyncio.Task[bytes] | None = None
        mutation_lock_held = False
        command_name: str | None = None
        try:
            pid, uid, gid = self._peer_credentials(writer)
            if uid != self.allowed_peer_uid:
                raise ProtocolError(f"Unix peer uid {uid} is not authorized")
            if not self.healthy:
                raise PackageBridgeError(
                    "package bridge is unhealthy after an internal failure"
                )
            request = await asyncio.wait_for(
                self._read_request(reader), timeout=self.handshake_timeout_seconds
            )
            command_name = request.command_name
            peer_event = asyncio.create_task(self._peer_event(reader))
            mutation_lock_held = await self._acquire_package_lock(peer_event)
            response = await self._run_request(request, peer_event)
            # Package database mutation is complete once the supervised command
            # and all descendants are gone. Do not let a slow/non-reading peer
            # hold the rootfs-wide package lock during response delivery.
            self._package_lock.release()
            mutation_lock_held = False
            if peer_event.done():
                raise PeerDisconnected("peer left or sent trailing request bytes")
            peer_event.cancel()
            with contextlib.suppress(BaseException):
                await peer_event
            peer_event = None
            try:
                await asyncio.wait_for(
                    self._write_response(writer, response),
                    timeout=RESPONSE_WRITE_TIMEOUT_SECONDS,
                )
            except (
                asyncio.TimeoutError,
                BrokenPipeError,
                ConnectionResetError,
            ) as exc:
                self._record(
                    "response_delivery_failed",
                    command=command_name,
                    error_type=type(exc).__name__,
                )
                return
            self._record(
                "command_completed",
                command=command_name,
                mutating=request.mutating,
                return_code=response.return_code,
                peer_pid=pid,
                peer_uid=uid,
                peer_gid=gid,
            )
        except PeerDisconnected as exc:
            self._record("peer_disconnected", command=command_name, detail=str(exc))
        except (
            ProtocolError,
            asyncio.IncompleteReadError,
            asyncio.TimeoutError,
        ) as exc:
            self._record(
                "request_rejected", command=command_name, error_type=type(exc).__name__
            )
            await self._write_rejection(writer, exc)
        except PackageBridgeError as exc:
            self._mark_fatal(type(exc).__name__)
            self._record(
                "request_rejected", command=command_name, error_type=type(exc).__name__
            )
            await self._write_rejection(writer, exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._mark_fatal(type(exc).__name__)
            self._record(
                "request_internal_error",
                command=command_name,
                error_type=type(exc).__name__,
            )
            await self._write_rejection(writer, exc)
        finally:
            if mutation_lock_held:
                self._package_lock.release()
            if peer_event is not None:
                peer_event.cancel()
                with contextlib.suppress(BaseException):
                    await peer_event
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), timeout=1.0)


async def request_broker(socket_path: Path, request: PackageRequest) -> PackageResponse:
    """Small bounded client used by isolated tests and integration probes."""

    reader, writer = await asyncio.open_unix_connection(str(socket_path))
    try:
        writer.write(encode_request(request))
        await writer.drain()
        header = await reader.readexactly(HEADER_SIZE)
        if header[:8] != RESPONSE_MAGIC:
            raise ProtocolError("broker returned an invalid response header")
        size = struct.unpack("!I", header[8:])[0]
        if size > MAX_RESPONSE_BYTES:
            raise ProtocolError("broker response exceeds its length limit")
        return decode_response_frame(header + await reader.readexactly(size))
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


__all__ = [
    "BASE_ENVIRONMENT",
    "CHROOT_LAUNCHER_SOURCE",
    "COMMAND_PATHS",
    "GUEST_WRAPPER_SOURCE",
    "GUEST_SOCKET_PATH",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "MUTATING_COMMANDS",
    "PRESERVED_ENVIRONMENT",
    "PackageBridgeError",
    "PackageCommandBroker",
    "PackageRequest",
    "PackageResponse",
    "PeerDisconnected",
    "ProtocolError",
    "READ_ONLY_COMMANDS",
    "build_proot_bind_arguments",
    "build_chroot_launcher",
    "build_guest_wrapper",
    "decode_request_frame",
    "decode_response_frame",
    "encode_request",
    "encode_response",
    "request_broker",
    "prepare_guest_bind_directory",
    "validate_request",
]
