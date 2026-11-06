"""Dockerless Harbor environment for trusted, single-container tasks.

Images are pulled with skopeo, unpacked with umoci, and executed under a
long-lived PRoot process.  This exists for cluster jobs where Docker-in-Docker
and Apptainer are unavailable.  It deliberately rejects GPU, Compose, image
build, and restricted-network tasks and is not a hardened sandbox.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, override

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from eval.terminal_bench.proot_package_bridge import (
    COMMAND_PATHS,
    GUEST_SOCKET_PATH,
    PackageBridgeError,
    PackageCommandBroker,
    build_chroot_launcher,
    build_guest_wrapper,
    build_proot_bind_arguments,
    prepare_guest_bind_directory,
)


_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_CACHE = Path("/tmp/harbor-proot-cache")
_RPC_SERVER_SOURCE = _MODULE_DIR / "proot_rpc_server.c"
_STATX_PROBE_SOURCE = _MODULE_DIR / "proot_statx_probe.c"
_FACCESSAT2_PROBE_SOURCE = _MODULE_DIR / "proot_faccessat2_probe.c"
_PROOT_BINARY_ENV = "HARBOR_PROOT_BINARY"
PROOT_GUEST_UTILITY_PROFILE = "tb20-file-xxd-rg-v1"
PROOT_GUEST_UTILITY_POLICY = "apt_profile_then_proot_rpc_verification_v1"
PROOT_GUEST_UTILITY_COMMANDS = ("file", "xxd", "rg")
PROOT_GUEST_UTILITY_APT_PACKAGES = ("file", "xxd", "ripgrep")
PROOT_GUEST_UTILITY_RECEIPT = "proot-guest-utilities.json"
_PROOT_GUEST_UTILITY_INSTALL_TIMEOUT_SEC = 600
_PROOT_GUEST_UTILITY_APT_ATTEMPTS = 3
_PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS = (1.0, 2.0)
# APT's HTTP method uses one Timeout value for both connection establishment
# and an inactive response read.  Keep that network-operation timeout small,
# disable APT's own opaque retry loop, and let the audited outer loop below own
# the three attempts.  Six worst-case update/install attempts then remain well
# inside Harbor's 600-second environment-start boundary.
_PROOT_GUEST_UTILITY_APT_NETWORK_TIMEOUT_SEC = 20
_PROOT_GUEST_UTILITY_APT_ATTEMPT_TIMEOUT_SEC = 75
_PROOT_GUEST_UTILITY_APT_INNER_RETRIES = 0
_PROOT_GUEST_UTILITY_APT_OPTIONS = (
    f"-o Acquire::Retries={_PROOT_GUEST_UTILITY_APT_INNER_RETRIES} "
    "-o Acquire::http::Timeout="
    f"{_PROOT_GUEST_UTILITY_APT_NETWORK_TIMEOUT_SEC} "
    "-o Acquire::https::Timeout="
    f"{_PROOT_GUEST_UTILITY_APT_NETWORK_TIMEOUT_SEC}"
)
_PROOT_GUEST_UTILITY_APT_TRANSPORT_PATTERNS = (
    (
        "apt_http_transient",
        re.compile(r"(?m)^[EW]:\s*Failed to fetch\s+https?://\S+\s+(?:408|429|5[0-9]{2})\b", re.I),
    ),
    ("command_timeout", re.compile(r"\bcommand timed out\b", re.I)),
    ("connection_failed", re.compile(r"\bconnection failed\b", re.I)),
    (
        "connection_timeout",
        re.compile(r"\b(?:connection|connect) timed out\b", re.I),
    ),
    ("could_not_connect", re.compile(r"\bcould not connect to\b", re.I)),
    (
        "temporary_dns_failure",
        re.compile(
            r"\btemporary failure (?:in name resolution|resolving)\b", re.I
        ),
    ),
    ("tls_handshake_timeout", re.compile(r"\btls handshake timeout\b", re.I)),
    ("unexpected_eof", re.compile(r"\bunexpected eof\b", re.I)),
    (
        "connection_reset",
        re.compile(r"\bconnection reset(?: by peer)?\b", re.I),
    ),
    (
        "proxy_http_transient",
        re.compile(
            r"\bproxy\b[^\n]*(?:status\s*)?(?:408|429|5[0-9]{2})\b",
            re.I,
        ),
    ),
)
_PROOT_GUEST_UTILITY_APT_UPDATE_INCOMPLETE_PATTERNS = (
    (
        "failed_to_fetch",
        re.compile(r"(?m)^W:\s*Failed to fetch\s+\S+"),
    ),
    (
        "index_files_ignored",
        re.compile(r"(?m)^W:\s*Some index files failed to download\b"),
    ),
)
_PROOT_GUEST_UTILITY_APT_UPDATE_SEMANTIC_FAILURE = (
    "apt_update_incomplete_indexes"
)
_PROOT_GUEST_UTILITY_RECEIPT_SCHEMA = (
    "terminal_bench_proot_guest_utilities_v2"
)
PROOT_IMAGE_PULL_RECEIPT = "proot-image-pull.json"
_PROOT_IMAGE_PULL_RECEIPT_SCHEMA = "terminal_bench_proot_image_pull_v1"
_PROOT_IMAGE_PULL_POLICY = "bounded_outer_retry_reuse_partial_oci_v1"
_PROOT_IMAGE_PULL_ATTEMPTS = 3
_PROOT_IMAGE_PULL_BACKOFF_SECONDS = (1.0, 2.0)
_PROOT_IMAGE_PULL_SKOPEO_RETRY_TIMES = 3
_PROOT_IMAGE_PULL_TRANSPORT_PATTERNS = (
    ("tls_handshake_timeout", re.compile(r"\btls handshake timeout\b", re.I)),
    ("unexpected_eof", re.compile(r"\bunexpected eof\b", re.I)),
    (
        "connection_reset",
        re.compile(r"\bconnection reset(?: by peer)?\b", re.I),
    ),
    ("io_timeout", re.compile(r"\bi/o timeout\b", re.I)),
    (
        "response_header_timeout",
        re.compile(r"\btimeout awaiting response headers\b", re.I),
    ),
    (
        "connection_timeout",
        re.compile(r"\b(?:connection|connect) timed out\b", re.I),
    ),
    (
        "temporary_dns_failure",
        re.compile(r"\btemporary failure in name resolution\b", re.I),
    ),
    ("dns_host_not_found", re.compile(r"\bno such host\b", re.I)),
    (
        "network_unreachable",
        re.compile(r"\bnetwork is unreachable\b", re.I),
    ),
    ("broken_pipe", re.compile(r"\bbroken pipe\b", re.I)),
    ("connection_refused", re.compile(r"\bconnection refused\b", re.I)),
    (
        "registry_http_transient",
        re.compile(
            r"\b(?:status(?: code)?|http(?: response)? status)\s*"
            r"(?:408|429|5[0-9]{2})\b|\btoo many requests\b|\btoomanyrequests\b",
            re.I,
        ),
    ),
)
_PROOT_RPC_TIMEOUT_RESPONSE_GRACE_SEC = 10.0
_PROOT_RPC_EXIT_OBSERVATION_GRACE_SEC = 0.25
_PROOT_RPC_HARD_STOP_GRACE_SEC = 3.0
_DIRECT_EXEC_HARD_STOP_GRACE_SEC = 2.0
_PROOT_RPC_MAX_FIELD_SIZE = 128 * 1024 * 1024
_PROOT_RUNTIME_RECEIPT = "proot-runtime.json"
_PROOT_RUNTIME_LOG = "proot-server.log"
_VERIFIER_TIMEOUT_RECEIPT = "verifier-timeout-resilience.json"
_VERIFIER_TIMEOUT_RECEIPT_SCHEMA = "terminal_bench_verifier_timeout_v1"
_VERIFIER_TIMEOUT_PARTIAL_MAX_BYTES = 64 * 1024 * 1024
# Retrying an arbitrary verifier after it has mutated the environment can change
# what is being scored.  Automatic retry is therefore limited to reviewed,
# immutable test scripts and only while their dependency/bootstrap process is
# still active.  The hash below is Terminal-Bench 2.1 pytorch-model-cli's
# official test.sh (task package sha256:6b7d27e1...).
_SAFE_NETWORK_RETRY_TEST_SHA256S = frozenset(
    {"43fb67e537c7adc2df11c2367f47c876cb5d16a7c1ae16d0339c3a3364bb2b06"}
)
_VERIFIER_NETWORK_PROCESS_NAMES = frozenset(
    {"apt", "apt-get", "curl", "uv", "uvx", "wget"}
)
_VERIFIER_TEST_PROCESS_PREFIXES = ("python", "pytest", "cli_tool")
_HOST_RESOLV_CONF = Path("/etc/resolv.conf")
_PROOT_RUNTIME_HOSTS = """\
# Generated privately by Harbor's PRoot backend for Docker runtime parity.
127.0.0.1\tlocalhost
::1\tlocalhost ip6-localhost ip6-loopback
fe00::0\tip6-localnet
ff00::0\tip6-mcastprefix
ff02::1\tip6-allnodes
ff02::2\tip6-allrouters
"""
PROOT_PACKAGE_BRIDGE_POLICY = "pinned_profile_proot_bind_host_chroot_v1"
PROOT_PACKAGE_BRIDGE_RECEIPT = "proot-package-bridge.json"
_PROOT_PACKAGE_BRIDGE_FEATURE_ENV = "HARBOR_PROOT_PACKAGE_BRIDGE"
_PROOT_PACKAGE_BRIDGE_REQUIRED_COMMANDS = frozenset({"apt-get", "dpkg", "dpkg-query"})
_PROOT_PACKAGE_BRIDGE_PROBE_TIMEOUT_SEC = 10
_PROOT_GUEST_UTILITY_VERSION_COMMANDS = {
    "file": "file --version",
    "xxd": "xxd -v",
    "rg": "rg --version",
}
_PROOT_GUEST_UTILITY_SUPPORTED_DISTROS = frozenset({"debian", "ubuntu"})
_SUPERVISOR_READY_ATTESTATION = b"R"
_SUPERVISOR_STATUS_ATTESTATION = b"S"
_SUPERVISOR_CLEANUP_ATTESTATION = b"K"
_SUPERVISOR_IDENTITY_GATE_RELEASE = b"G"
_SUPERVISOR_READY_TIMEOUT_SEC = 3.0
_SUPERVISOR_CONTROL_FRAME_SIZE = 7
_CAP_SYS_PTRACE_BIT = 19
_LINUX_CAPABILITY_FIELDS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
_TAGGED_SHA256_IMAGE_RE = re.compile(
    r"^(?P<repository>[^@]+):(?P<tag>[^/:@]+)@"
    r"(?P<digest>sha256:[0-9A-Fa-f]{64})$"
)


class _SupervisorIdentity:
    """Host-owned, PID-reuse-safe identity for one cleanup supervisor."""

    __slots__ = (
        "process",
        "pid",
        "pidfd",
        "procfd",
        "cleanup_ack_fd",
        "start_gate_fd",
        "ready_attested",
        "cleanup_attested",
        "control_eof",
        "control_buffer",
        "primary_wait_status",
        "control_invalid",
        "closed",
    )

    def __init__(
        self,
        *,
        process: asyncio.subprocess.Process,
        pid: int,
        pidfd: int,
        procfd: int,
        cleanup_ack_fd: int,
        start_gate_fd: int,
    ) -> None:
        self.process = process
        self.pid = pid
        self.pidfd = pidfd
        self.procfd = procfd
        self.cleanup_ack_fd = cleanup_ack_fd
        self.start_gate_fd = start_gate_fd
        self.ready_attested = False
        self.cleanup_attested = False
        self.control_eof = False
        self.control_buffer = bytearray()
        self.primary_wait_status: int | None = None
        self.control_invalid = False
        self.closed = False


class ProotEnvironment(BaseEnvironment):
    """A best-effort OCI/PRoot backend for Harbor 0.21."""

    _image_locks: dict[str, asyncio.Lock] = {}

    def __init__(
        self,
        *args: Any,
        keep_containers: bool = False,
        guest_utility_profile: str | None = None,
        verifier_checkpoint_path: str | None = None,
        verifier_checkpoint_sha256: str | None = None,
        verifier_checkpoint_task: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if guest_utility_profile not in {None, PROOT_GUEST_UTILITY_PROFILE}:
            raise ValueError(
                "Unsupported PRoot guest utility profile: "
                f"{guest_utility_profile!r}; expected "
                f"{PROOT_GUEST_UTILITY_PROFILE!r}"
            )
        self._keep_containers = keep_containers
        if bool(verifier_checkpoint_path) != bool(verifier_checkpoint_sha256):
            raise ValueError("Verifier checkpoint path and digest must be supplied together")
        self._verifier_checkpoint_path = verifier_checkpoint_path
        self._verifier_checkpoint_sha256 = verifier_checkpoint_sha256
        self._verifier_checkpoint_task = verifier_checkpoint_task
        self._restored_verifier_checkpoint = None
        self._guest_utility_profile = guest_utility_profile
        self._cache_dir = Path(
            os.environ.get("HARBOR_PROOT_CACHE", str(_DEFAULT_CACHE))
        ).resolve()
        self._bundle_dir: Path | None = None
        self._rootfs: Path | None = None
        self._rpc_process: asyncio.subprocess.Process | None = None
        self._rpc_identity: _SupervisorIdentity | None = None
        self._rpc_cleanup_required = False
        self._quarantined_supervisors: list[_SupervisorIdentity] = []
        self._untracked_supervisors: list[asyncio.subprocess.Process] = []
        self._rpc_log_handle: Any | None = None
        self._rpc_log_path: Path | None = None
        self._rpc_poisoned = False
        self._rpc_poison_reason: str | None = None
        self._rpc_diagnostic_events: list[dict[str, Any]] = []
        self._cleanup_incomplete = False
        self._rpc_lock = asyncio.Lock()
        self._verifier_timeout_count = 0
        self._verifier_timeout_retry_used = False
        self._verifier_timeout_records: list[dict[str, Any]] = []
        self._last_verifier_timeout: dict[str, Any] | None = None
        self._active_verifier_timeout_retry_record: dict[str, Any] | None = None
        package_bridge_setting = os.environ.get(_PROOT_PACKAGE_BRIDGE_FEATURE_ENV)
        if package_bridge_setting not in {None, "0", "1"}:
            raise ValueError(
                f"{_PROOT_PACKAGE_BRIDGE_FEATURE_ENV} must be unset, '0', or '1'"
            )
        if (
            package_bridge_setting == "1"
            and guest_utility_profile != PROOT_GUEST_UTILITY_PROFILE
        ):
            raise ValueError(
                f"{_PROOT_PACKAGE_BRIDGE_FEATURE_ENV}=1 requires the pinned "
                f"guest utility profile {PROOT_GUEST_UTILITY_PROFILE!r}"
            )
        package_bridge_requested = bool(
            package_bridge_setting != "0"
            and guest_utility_profile == PROOT_GUEST_UTILITY_PROFILE
        )
        self._package_bridge_enabled = package_bridge_requested
        self._package_bridge_broker: PackageCommandBroker | None = None
        self._package_bridge_runtime_dir: Path | None = None
        self._package_bridge_socket_path: Path | None = None
        self._package_bridge_wrapper_path: Path | None = None
        self._package_bridge_launcher_path: Path | None = None
        self._package_bridge_bind_args: tuple[str, ...] = ()
        self._package_bridge_bound_commands: tuple[str, ...] = ()
        self._package_bridge_ready = False
        self._package_bridge_accepting_commands = False
        self._package_bridge_status = (
            "not_started" if self._package_bridge_enabled else "disabled"
        )
        self._package_bridge_diagnostic_events: list[dict[str, Any]] = []
        self._package_command_lock = asyncio.Lock()
        self._unbounded_wire_exec: contextvars.ContextVar[bool] = (
            contextvars.ContextVar(
                f"proot_unbounded_wire_exec_{id(self)}", default=False
            )
        )
        self._image_workdir = "/"
        self._image_env: dict[str, str] = {}
        self._source_image: str | None = None
        self._oci_manifest_digest: str | None = None
        self._image_pull_receipt: dict[str, Any] | None = None
        self._guest_utility_receipt: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    @override
    def type() -> str:
        return "proot"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    @classmethod
    @override
    def preflight(cls) -> None:
        try:
            cls._require_safe_cleanup_capability_boundary()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        missing = [
            name
            for name in ("skopeo", "umoci", "gcc", "chroot")
            if not shutil.which(name)
        ]
        if missing:
            raise SystemExit(
                "The custom PRoot environment requires: " + ", ".join(missing)
            )
        proot_binary = cls._configured_proot_binary()
        gcc_binary = Path(shutil.which("gcc") or "gcc")
        cls._verify_proot_statx(proot_binary, gcc_binary)
        cls._verify_proot_faccessat2(proot_binary, gcc_binary)
        helper_cache = (
            Path(os.environ.get("HARBOR_PROOT_CACHE", str(_DEFAULT_CACHE))).resolve()
            / "helpers"
        )
        if os.environ.get(_PROOT_PACKAGE_BRIDGE_FEATURE_ENV) != "0":
            try:
                build_guest_wrapper(cache_dir=helper_cache, compiler=str(gcc_binary))
                build_chroot_launcher(cache_dir=helper_cache, compiler=str(gcc_binary))
            except PackageBridgeError as exc:
                raise SystemExit(
                    f"PRoot package bridge preflight failed: {exc}"
                ) from exc

    @staticmethod
    def _require_safe_cleanup_capability_boundary() -> None:
        """Reject hosts where a child could ptrace/reopen its cleanup owner.

        ``PR_SET_DUMPABLE=0`` protects the supervisor's ``/proc/PID/fd`` only
        while descendants lack ``CAP_SYS_PTRACE``.  Checking every inheritable,
        permitted, effective, bounding, and ambient set also prevents a command
        from acquiring the capability later through a file capability.
        """

        try:
            # Linux capabilities are per-thread.  ``thread-self`` names the
            # actual thread that will fork, unlike ``/proc/self/status`` which
            # always describes the thread-group leader.
            payload = Path("/proc/thread-self/status").read_text(encoding="ascii")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                "Cannot prove the host cleanup capability boundary: "
                "/proc/thread-self/status is unavailable"
            ) from exc
        values: dict[str, int] = {}
        for line in payload.splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            if key not in _LINUX_CAPABILITY_FIELDS:
                continue
            fields = raw_value.split()
            if len(fields) != 1 or not re.fullmatch(r"[0-9A-Fa-f]+", fields[0]):
                raise RuntimeError(
                    f"Cannot prove the host cleanup capability boundary: "
                    f"malformed {key}"
                )
            if key in values:
                raise RuntimeError(
                    f"Cannot prove the host cleanup capability boundary: "
                    f"duplicate {key}"
                )
            values[key] = int(fields[0], 16)
        missing = [key for key in _LINUX_CAPABILITY_FIELDS if key not in values]
        if missing:
            raise RuntimeError(
                "Cannot prove the host cleanup capability boundary: missing "
                + ", ".join(missing)
            )
        unsafe = [
            key for key, value in values.items() if value & (1 << _CAP_SYS_PTRACE_BIT)
        ]
        if unsafe:
            raise RuntimeError(
                "The PRoot cleanup supervisor requires CAP_SYS_PTRACE to be "
                "absent from all host capability sets; present in " + ", ".join(unsafe)
            )

    @classmethod
    def _configured_proot_binary(cls) -> Path:
        configured = os.environ.get(_PROOT_BINARY_ENV)
        if not configured:
            raise SystemExit(
                f"{_PROOT_BINARY_ENV} must point to a statx/faccessat2-capable "
                "PRoot binary. Older builds return false ENOENT results for "
                "guest file metadata or executable checks; use the pinned "
                "PRoot supplied by the pool launcher."
            )
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            resolved = shutil.which(configured)
            if resolved is None:
                raise SystemExit(
                    f"{_PROOT_BINARY_ENV}={configured!r} is not an executable file "
                    "or a command on PATH."
                )
            candidate = Path(resolved)
        candidate = candidate.resolve()
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise SystemExit(
                f"{_PROOT_BINARY_ENV}={configured!r} does not resolve to an "
                "executable file."
            )
        return candidate

    @classmethod
    def _verify_proot_statx(cls, proot_binary: Path, gcc_binary: Path) -> None:
        """Fail closed unless PRoot translates statx paths into its guest rootfs."""

        cls._verify_proot_guest_syscall(
            proot_binary, gcc_binary, "statx", _STATX_PROBE_SOURCE
        )

    @classmethod
    def _verify_proot_faccessat2(cls, proot_binary: Path, gcc_binary: Path) -> None:
        """Check the guest-path syscall used by modern bash's test -x."""

        cls._verify_proot_guest_syscall(
            proot_binary, gcc_binary, "faccessat2", _FACCESSAT2_PROBE_SOURCE
        )

    @classmethod
    def _verify_proot_guest_syscall(
        cls, proot_binary: Path, gcc_binary: Path, syscall: str, source: Path
    ) -> None:
        with tempfile.TemporaryDirectory(prefix=f"harbor-proot-{syscall}-") as temporary:
            rootfs = Path(temporary) / "rootfs"
            (rootfs / "tmp").mkdir(parents=True)
            probe = rootfs / f".harbor-proot-{syscall}-probe"
            compile_result = subprocess.run(
                [
                    str(gcc_binary),
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-static",
                    "-o",
                    str(probe),
                    str(source),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if compile_result.returncode != 0:
                raise SystemExit(
                    f"Unable to compile the PRoot {syscall} preflight probe:\n"
                    + compile_result.stderr.decode(errors="replace")
                )
            try:
                probe_result = subprocess.run(
                    [
                        str(proot_binary),
                        "-0",
                        "-r",
                        str(rootfs),
                        "-w",
                        "/",
                        f"/{probe.name}",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=20,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                raise SystemExit(
                    f"{_PROOT_BINARY_ENV}={str(proot_binary)!r} timed out during "
                    f"the {syscall} semantics probe."
                ) from None
            except OSError as exc:
                raise SystemExit(
                    f"{_PROOT_BINARY_ENV}={str(proot_binary)!r} could not run "
                    f"the {syscall} semantics probe: {exc}"
                ) from exc
            stdout = probe_result.stdout.decode(errors="replace").strip()
            stderr = probe_result.stderr.decode(errors="replace").strip()
            if probe_result.returncode != 0 or stdout != f"proot-{syscall}-ready":
                detail = stderr or stdout or f"exit status {probe_result.returncode}"
                raise SystemExit(
                    f"{_PROOT_BINARY_ENV}={str(proot_binary)!r} failed the guest "
                    f"{syscall} semantics probe: {detail}. Restart through the "
                    "pool launcher to provision the pinned PRoot with statx "
                    "and faccessat2 guest-path translation support."
                )

    @override
    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image:
            raise ValueError(
                "The PRoot environment only supports tasks with a prebuilt "
                "[environment].docker_image."
            )
        if self.extra_docker_compose_paths:
            raise ValueError("The PRoot environment does not support Docker Compose.")

    @staticmethod
    def _safe_component(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
        return cleaned[:80] or "trial"

    @staticmethod
    def _normalize_skopeo_image_reference(image: str) -> str:
        """Return an immutable image reference accepted by older Skopeo.

        Harbor registry tasks can pin an image as ``repository:tag@sha256:digest``.
        That is a valid Docker reference, but Skopeo 1.13 rejects the combined
        tag-and-digest spelling.  The digest already identifies the exact image,
        so dropping only the redundant tag preserves immutability while making
        the reference usable by both old and new Skopeo releases.

        Be deliberately conservative: ordinary tags, digest-only references,
        malformed digests, and non-SHA256 algorithms are returned unchanged.
        """

        match = _TAGGED_SHA256_IMAGE_RE.fullmatch(image)
        if match is None:
            return image
        return f"{match.group('repository')}@{match.group('digest')}"

    @staticmethod
    def _image_pull_transport_signals(value: str) -> tuple[str, ...]:
        """Return only reviewed transient registry transport signatures."""

        return tuple(
            name
            for name, pattern in _PROOT_IMAGE_PULL_TRANSPORT_PATTERNS
            if pattern.search(value)
        )

    @staticmethod
    def _guest_utility_apt_transport_signals(value: str) -> tuple[str, ...]:
        """Return reviewed APT/proxy transport signatures from host output."""

        return tuple(
            name
            for name, pattern in _PROOT_GUEST_UTILITY_APT_TRANSPORT_PATTERNS
            if pattern.search(value)
        )

    @staticmethod
    def _guest_utility_apt_update_incomplete_signals(
        value: str,
    ) -> tuple[str, ...]:
        """Detect APT's exit-0 partial-index warning.

        ``apt-get update`` deliberately exits zero when at least one configured
        index could be reused, even if another configured index could not be
        fetched.  Continuing to installation after that warning can turn an
        egress outage into a misleading ``Unable to locate package`` error.
        """

        return tuple(
            name
            for name, pattern in (
                _PROOT_GUEST_UTILITY_APT_UPDATE_INCOMPLETE_PATTERNS
            )
            if pattern.search(value)
        )

    def _write_image_pull_receipt(self) -> None:
        """Durably persist host-only pull provenance without registry secrets."""

        receipt = self._image_pull_receipt
        if receipt is None:
            return
        artifact_dir = self._runtime_artifact_dir()
        if artifact_dir is None:
            raise RuntimeError("PRoot image-pull receipt directory is unavailable")
        path = artifact_dir / PROOT_IMAGE_PULL_RECEIPT
        temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
        receipt["updated_at"] = self._utc_now()
        payload = (
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
            directory_fd = os.open(artifact_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    async def _pull_image_with_retry(
        self,
        *,
        pull_image: str,
        oci_dir: Path,
        complete_marker: Path,
    ) -> None:
        """Pull one OCI image with a bounded outer transport retry loop.

        Skopeo's internal retry can still exhaust while reconnecting a large
        blob.  Re-running ``copy`` against the same OCI layout lets Skopeo reuse
        content-addressed blobs that were already completed.  The directory is
        intentionally not removed between outer attempts (or between failed
        workers); the complete marker remains the sole cache-ready authority.
        """

        started_at = self._utc_now()
        pull_image_sha256 = hashlib.sha256(pull_image.encode("utf-8")).hexdigest()
        source_image = self._source_image or pull_image
        self._image_pull_receipt = {
            "schema": _PROOT_IMAGE_PULL_RECEIPT_SCHEMA,
            "schema_version": 1,
            "session_id": self.session_id,
            "policy": _PROOT_IMAGE_PULL_POLICY,
            "failure_policy": "closed",
            "status": "running",
            "stage": "skopeo_copy",
            "started_at": started_at,
            "source_image_sha256": hashlib.sha256(
                source_image.encode("utf-8")
            ).hexdigest(),
            "pull_image_sha256": pull_image_sha256,
            "image_cache_key": pull_image_sha256[:20],
            "max_attempts": _PROOT_IMAGE_PULL_ATTEMPTS,
            "backoff_seconds": list(_PROOT_IMAGE_PULL_BACKOFF_SECONDS),
            "skopeo_inner_retry_times": _PROOT_IMAGE_PULL_SKOPEO_RETRY_TIMES,
            "partial_oci_reuse": True,
            "partial_oci_present_at_start": oci_dir.exists(),
            "attempts": [],
            "events": [
                {"event": "image_pull_started", "recorded_at": started_at}
            ],
        }
        self._write_image_pull_receipt()
        attempts = self._image_pull_receipt["attempts"]
        events = self._image_pull_receipt["events"]

        # Operator-prefetched images on the cluster's shared filesystem avoid
        # repeating a failed Docker Hub transfer on every worker. No proxy is
        # changed: this affects only the image import, before agent startup.
        from terminal_bench_eval.docker_hub_recovery import shared_oci_source, verify_oci
        shared_source = await asyncio.to_thread(shared_oci_source, pull_image)
        source_reference = f"docker://{pull_image}"
        if shared_source is not None:
            shared_path, shared_digest = shared_source
            source_reference = f"oci:{shared_path}:image"
            self._image_pull_receipt.update(
                source_kind="verified_shared_oci",
                shared_oci_manifest_digest=shared_digest,
            )
            self._write_image_pull_receipt()

        for attempt_number in range(1, _PROOT_IMAGE_PULL_ATTEMPTS + 1):
            attempt_started_at = self._utc_now()
            attempt: dict[str, Any] = {
                "attempt": attempt_number,
                "max_attempts": _PROOT_IMAGE_PULL_ATTEMPTS,
                "status": "running",
                "started_at": attempt_started_at,
            }
            attempts.append(attempt)
            self._image_pull_receipt["last_attempt"] = attempt_number
            self._write_image_pull_receipt()
            try:
                await self._run_host(
                    shutil.which("skopeo") or "skopeo",
                    "copy",
                    "--retry-times",
                    str(_PROOT_IMAGE_PULL_SKOPEO_RETRY_TIMES),
                    source_reference,
                    f"oci:{oci_dir}:image",
                    timeout_sec=1800,
                )
                if shared_source is not None:
                    await asyncio.to_thread(verify_oci, oci_dir, shared_digest)
            except BaseException as exc:
                finished_at = self._utc_now()
                error_text = str(exc)
                transport_signals = (
                    self._image_pull_transport_signals(error_text)
                    if isinstance(exc, RuntimeError)
                    else ()
                )
                retry_scheduled = bool(
                    transport_signals
                    and attempt_number < _PROOT_IMAGE_PULL_ATTEMPTS
                )
                attempt.update(
                    {
                        "status": "failed",
                        "finished_at": finished_at,
                        "error_type": type(exc).__name__,
                        "error_sha256": hashlib.sha256(
                            error_text.encode("utf-8")
                        ).hexdigest(),
                        "transient_transport_error": bool(transport_signals),
                        "transport_signals": list(transport_signals),
                        "retry_scheduled": retry_scheduled,
                    }
                )
                if retry_scheduled:
                    backoff = _PROOT_IMAGE_PULL_BACKOFF_SECONDS[attempt_number - 1]
                    attempt.update(
                        {
                            "backoff_seconds": backoff,
                            "next_attempt": attempt_number + 1,
                        }
                    )
                    events.append(
                        {
                            "event": "image_pull_attempt_failed",
                            "attempt": attempt_number,
                            "recorded_at": finished_at,
                        }
                    )
                    self._write_image_pull_receipt()
                    try:
                        await asyncio.sleep(backoff)
                    except BaseException as sleep_exc:
                        cancelled_at = self._utc_now()
                        self._image_pull_receipt.update(
                            {
                                "status": "cancelled",
                                "finished_at": cancelled_at,
                                "cancellation_type": type(sleep_exc).__name__,
                            }
                        )
                        events.append(
                            {
                                "event": "image_pull_cancelled",
                                "recorded_at": cancelled_at,
                            }
                        )
                        self._write_image_pull_receipt()
                        raise
                    continue

                terminal_status = (
                    "cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed"
                )
                self._image_pull_receipt.update(
                    {
                        "status": terminal_status,
                        "finished_at": finished_at,
                        "transport_failure": bool(transport_signals),
                        "attempts_exhausted": bool(
                            transport_signals
                            and attempt_number == _PROOT_IMAGE_PULL_ATTEMPTS
                        ),
                        "final_error_type": type(exc).__name__,
                        "final_error_sha256": attempt["error_sha256"],
                        "transport_signals": list(transport_signals),
                    }
                )
                events.append(
                    {
                        "event": (
                            "image_pull_cancelled"
                            if terminal_status == "cancelled"
                            else "image_pull_failed"
                        ),
                        "attempt": attempt_number,
                        "recorded_at": finished_at,
                    }
                )
                self._write_image_pull_receipt()
                raise

            finished_at = self._utc_now()
            attempt.update(
                {
                    "status": "succeeded",
                    "finished_at": finished_at,
                }
            )
            self._image_pull_receipt.update(
                {
                    "status": "ok",
                    "stage": "complete",
                    "finished_at": finished_at,
                    "successful_attempt": attempt_number,
                }
            )
            events.append(
                {
                    "event": "image_pull_succeeded",
                    "attempt": attempt_number,
                    "recorded_at": finished_at,
                }
            )
            self._write_image_pull_receipt()
            complete_marker.write_text(pull_image + "\n", encoding="utf-8")
            return

        raise AssertionError("bounded PRoot image-pull retry loop fell through")

    async def _run_host(
        self, *args: str, timeout_sec: int | None = None
    ) -> tuple[bytes, bytes]:
        spawning = asyncio.create_task(asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        ))
        process = None
        communication = None
        try:
            # Cancellation must not strand a just-created skopeo/umoci child,
            # nor let an old download keep writing a cache used by the retry.
            process = await asyncio.shield(spawning)
            communication = asyncio.create_task(process.communicate())
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communication), timeout=timeout_sec
            )
        except BaseException as exc:
            if process is None:
                process = await asyncio.shield(spawning)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            if communication is None:
                communication = asyncio.create_task(process.communicate())
            await asyncio.shield(communication)
            if isinstance(exc, asyncio.TimeoutError):
                raise RuntimeError(
                    f"Host command timed out after {timeout_sec}s: {args[0]}"
                ) from None
            raise
        if process.returncode:
            raise RuntimeError(
                f"Host command failed ({process.returncode}): {' '.join(args)}\n"
                f"{stderr.decode(errors='replace')}"
            )
        return stdout, stderr

    @override
    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError(
                "The PRoot environment cannot build Dockerfiles; use the declared "
                "prebuilt image without --force-build."
            )

        if self._verifier_checkpoint_path:
            from terminal_bench_eval.verifier_checkpoint import restore_checkpoint
            marker = Path(self._verifier_checkpoint_path)
            metadata = json.loads(marker.read_text())
            if metadata.get("task") != self._verifier_checkpoint_task:
                raise RuntimeError("Verifier checkpoint frozen task identity disagrees")
            self._bundle_dir = self._cache_dir / "runs" / (
                self._safe_component(self.session_id) + "-restore-" + uuid.uuid4().hex[:8]
            )
            self._restored_verifier_checkpoint = await asyncio.to_thread(
                restore_checkpoint, marker, self._verifier_checkpoint_sha256,
                self._bundle_dir, self._verifier_checkpoint_task["name"],
            )
            self._rootfs = self._bundle_dir / "rootfs"
            self._source_image = self.task_env_config.docker_image
            self._oci_manifest_digest = metadata["image_manifest"]
            await self._start_bundle_runtime(upload_environment=False)
            return

        image = self.task_env_config.docker_image
        assert image is not None
        self._source_image = image
        pull_image = self._normalize_skopeo_image_reference(image)
        image_key = hashlib.sha256(pull_image.encode()).hexdigest()[:20]
        images_dir = self._cache_dir / "images"
        runs_dir = self._cache_dir / "runs"
        oci_dir = images_dir / image_key
        complete_marker = images_dir / f"{image_key}.complete"
        images_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        lock = self._image_locks.setdefault(image_key, asyncio.Lock())
        async with lock:
            if not complete_marker.is_file():
                await self._pull_image_with_retry(
                    pull_image=pull_image,
                    oci_dir=oci_dir,
                    complete_marker=complete_marker,
                )

        try:
            oci_index = json.loads((oci_dir / "index.json").read_text(encoding="utf-8"))
            manifests = oci_index.get("manifests", [])
            self._oci_manifest_digest = str(manifests[0]["digest"])
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Unable to resolve the cached OCI manifest digest for {image}: {exc}"
            ) from exc

        run_name = self._safe_component(self.session_id)
        self._bundle_dir = runs_dir / f"{run_name}-{uuid.uuid4().hex[:8]}"
        await self._run_host(
            shutil.which("umoci") or "umoci",
            "unpack",
            "--image",
            f"{oci_dir}:image",
            str(self._bundle_dir),
            timeout_sec=1800,
        )
        self._rootfs = self._bundle_dir / "rootfs"

        await self._start_bundle_runtime()

    async def _start_bundle_runtime(self, *, upload_environment: bool = True) -> None:

        config = json.loads((self._bundle_dir / "config.json").read_text())
        process_config = config.get("process", {})
        self._image_workdir = (
            self.task_env_config.workdir or process_config.get("cwd") or "/"
        )
        self._image_env = {}
        for item in process_config.get("env", []):
            if "=" in item:
                key, value = item.split("=", 1)
                self._image_env[key] = value
        self._image_env.update(self._startup_env())
        # This cluster requires its host proxy for usable public egress. Docker
        # normally receives equivalent daemon/client proxy configuration; pass
        # only the conventional proxy variables into this Dockerless backend.
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        ):
            if value := os.environ.get(key):
                self._image_env.setdefault(key, value)
        self._image_env.setdefault("SHELL", "/bin/bash")

        self._prepare_rootfs()
        await self._install_guest_utilities()
        stage = "package_bridge_start"
        try:
            await self._start_package_bridge()
            stage = "proot_rpc_start"
            await self._start_rpc_server()
            stage = "package_bridge_probe"
            await self._verify_package_bridge()
            stage = "guest_utility_verification"
            await self._verify_guest_utilities()
            stage = "environment_upload"
            if upload_environment:
                await self._upload_environment_dir_after_start()
            self._package_bridge_accepting_commands = self._package_bridge_enabled
        except BaseException as exc:
            if stage in {
                "package_bridge_start",
                "proot_rpc_start",
                "package_bridge_probe",
            }:
                self._set_guest_utility_failure(stage, exc)
            cleanup = asyncio.create_task(self._shutdown_runtime_components())
            try:
                await asyncio.shield(cleanup)
            except BaseException:
                if not cleanup.done():
                    cleanup.cancel()
                with contextlib.suppress(BaseException):
                    await cleanup
            self._close_rpc_log_handle()
            raise

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _output_tail(value: str | None, maximum: int = 4_096) -> str:
        text = value or ""
        return text[-maximum:]

    def _guest_utility_receipt_path(self) -> Path:
        suffix = self._safe_component(self.session_id)
        return self.trial_paths.trial_dir / (
            f"{Path(PROOT_GUEST_UTILITY_RECEIPT).stem}-{suffix}.json"
        )

    def _write_guest_utility_receipt(self) -> None:
        if self._guest_utility_receipt is None:
            return
        path = self._guest_utility_receipt_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
        self._guest_utility_receipt["updated_at"] = self._utc_now()
        payload = (
            json.dumps(
                self._guest_utility_receipt,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _runtime_artifact_dir(self) -> Path | None:
        trial_paths = getattr(self, "trial_paths", None)
        trial_dir = getattr(trial_paths, "trial_dir", None)
        if trial_dir is None:
            return None
        try:
            path = Path(trial_dir)
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            return None

    @staticmethod
    def _verifier_script_path(command: str) -> str | None:
        """Return the test path only for Harbor's exact redirected verifier form."""

        match = re.fullmatch(
            r"\((?P<script>/tests/[^\s()]+)\)\s*>\s*"
            r"/logs/verifier/test-stdout\.txt\s+2>&1",
            command.strip(),
        )
        return match.group("script") if match is not None else None

    def _verifier_script_sha256(self, command: str) -> str | None:
        script = self._verifier_script_path(command)
        if script is None:
            return None
        try:
            source = self._rootfs_path(script)
            info = source.stat(follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                return None
            return hashlib.sha256(source.read_bytes()).hexdigest()
        except OSError:
            return None

    @staticmethod
    def _atomic_copy_regular_tail(
        source: Path,
        destination: Path,
        *,
        maximum_bytes: int,
    ) -> dict[str, Any]:
        """Copy one changing regular file without following a guest symlink."""

        source_fd = os.open(
            source,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        temporary = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
        try:
            info = os.fstat(source_fd)
            if not stat.S_ISREG(info.st_mode):
                raise OSError(f"refusing to copy non-regular file: {source}")
            source_size = max(0, int(info.st_size))
            start = max(0, source_size - maximum_bytes)
            if start:
                os.lseek(source_fd, start, os.SEEK_SET)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination_fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
            )
            copied = 0
            try:
                remaining = min(source_size, maximum_bytes)
                while remaining:
                    chunk = os.read(source_fd, min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination_fd, view)
                        view = view[written:]
                    copied += len(chunk)
                    remaining -= len(chunk)
                os.fsync(destination_fd)
            finally:
                os.close(destination_fd)
            os.replace(temporary, destination)
            directory_fd = os.open(
                destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return {
                "bytes_copied": copied,
                "source_bytes_at_snapshot": source_size,
                "tail_truncated": bool(start),
            }
        finally:
            os.close(source_fd)
            temporary.unlink(missing_ok=True)

    @classmethod
    def _verifier_descendant_process_snapshot(
        cls, identity: _SupervisorIdentity | None
    ) -> dict[str, Any]:
        """Collect bounded, argument-free process evidence for retry gating.

        Command arguments can contain credentials, so only basenames are inspected;
        the durable receipt further reduces them to stage categories and counts.
        Any procfs race makes the snapshot incomplete and unable to authorize retry.
        """

        if not isinstance(identity, _SupervisorIdentity) or identity.closed:
            return {"complete": False, "process_names": []}
        children, failed = cls._pinned_direct_children(identity)
        if failed:
            return {"complete": False, "process_names": []}
        queue = list(children)
        seen: set[int] = set()
        names: set[str] = set()
        complete = True
        deadline = time.monotonic() + 0.25
        while queue and len(seen) < 256 and time.monotonic() < deadline:
            pid = queue.pop(0)
            if pid in seen:
                continue
            seen.add(pid)
            proc_dir = Path("/proc") / str(pid)
            try:
                comm = (
                    (proc_dir / "comm")
                    .read_text(encoding="utf-8", errors="replace")
                    .strip()
                )
                cmdline = (proc_dir / "cmdline").read_bytes().split(b"\0", 1)[0]
                child_text = (proc_dir / "task" / str(pid) / "children").read_text(
                    encoding="ascii"
                )
            except OSError:
                complete = False
                continue
            for raw_name in (comm, os.path.basename(os.fsdecode(cmdline))):
                normalized = raw_name.strip().lower()
                if normalized:
                    names.add(normalized)
            try:
                queue.extend(int(value) for value in child_text.split())
            except ValueError:
                complete = False
        if queue:
            complete = False
        return {"complete": complete, "process_names": sorted(names)}

    @staticmethod
    def _network_bootstrap_retry_eligible(snapshot: dict[str, Any]) -> bool:
        if snapshot.get("complete") is not True:
            return False
        raw_names = snapshot.get("process_names")
        if not isinstance(raw_names, list) or not all(
            isinstance(value, str) for value in raw_names
        ):
            return False
        names = {value.lower() for value in raw_names}
        if any(name.startswith(_VERIFIER_TEST_PROCESS_PREFIXES) for name in names):
            return False
        return bool(names & _VERIFIER_NETWORK_PROCESS_NAMES)

    @staticmethod
    def _sanitize_verifier_process_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        """Persist stage evidence without attacker-controlled process names."""

        raw_names = snapshot.get("process_names")
        names = (
            {value.lower() for value in raw_names if isinstance(value, str)}
            if isinstance(raw_names, list)
            else set()
        )
        network = sorted(names & _VERIFIER_NETWORK_PROCESS_NAMES)
        tests = sorted(
            prefix
            for prefix in _VERIFIER_TEST_PROCESS_PREFIXES
            if any(name.startswith(prefix) for name in names)
        )
        recognized = set(network)
        recognized.update(
            name
            for name in names
            if name.startswith(_VERIFIER_TEST_PROCESS_PREFIXES)
        )
        return {
            "complete": snapshot.get("complete") is True,
            "network_processes": network,
            "test_processes": tests,
            "other_process_count": max(0, len(names - recognized)),
        }

    def _write_verifier_timeout_receipt(self) -> None:
        trial_paths = getattr(self, "trial_paths", None)
        verifier_dir = getattr(trial_paths, "verifier_dir", None)
        if verifier_dir is None:
            return
        destination_dir = Path(verifier_dir)
        if destination_dir.is_symlink():
            return
        payload = {
            "schema": _VERIFIER_TIMEOUT_RECEIPT_SCHEMA,
            "policy": "persist_all_retry_reviewed_network_bootstrap_once_v1",
            "session_id": getattr(self, "session_id", None),
            "records": list(getattr(self, "_verifier_timeout_records", ())),
            "updated_at": self._utc_now(),
        }
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / _VERIFIER_TIMEOUT_RECEIPT
            temporary = destination.with_name(
                f".{destination.name}-{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary.open("x", encoding="utf-8") as handle:
                    json.dump(
                        payload,
                        handle,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.chmod(0o600)
                os.replace(temporary, destination)
                directory_fd = os.open(
                    destination_dir,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError:
            # Timeout diagnostics must never hide Harbor's original timeout.
            return

    def _capture_verifier_timeout(self, command: str) -> None:
        """Persist verifier output and conservative retry evidence before teardown."""

        if self._verifier_script_path(command) is None:
            return
        count = int(getattr(self, "_verifier_timeout_count", 0)) + 1
        self._verifier_timeout_count = count
        script_sha256 = self._verifier_script_sha256(command)
        process_snapshot = self._verifier_descendant_process_snapshot(
            getattr(self, "_rpc_identity", None)
        )
        retry_eligible = bool(
            count == 1
            and script_sha256 in _SAFE_NETWORK_RETRY_TEST_SHA256S
            and self._network_bootstrap_retry_eligible(process_snapshot)
        )
        record: dict[str, Any] = {
            "timeout_attempt": count,
            "recorded_at": self._utc_now(),
            "script_sha256": script_sha256,
            "process_snapshot": self._sanitize_verifier_process_snapshot(
                process_snapshot
            ),
            "retry_eligible": retry_eligible,
            "retry_state": "eligible" if retry_eligible else "not_eligible",
        }
        trial_paths = getattr(self, "trial_paths", None)
        verifier_dir = getattr(trial_paths, "verifier_dir", None)
        rootfs = getattr(self, "_rootfs", None)
        if verifier_dir is not None and isinstance(rootfs, Path):
            destination_dir = Path(verifier_dir)
            if not destination_dir.is_symlink():
                source = rootfs / "logs" / "verifier" / "test-stdout.txt"
                archive_name = f"test-stdout.timeout-{count}.partial.txt"
                try:
                    copy_result = self._atomic_copy_regular_tail(
                        source,
                        destination_dir / archive_name,
                        maximum_bytes=_VERIFIER_TIMEOUT_PARTIAL_MAX_BYTES,
                    )
                    # Keep Harbor/UI's canonical path useful when no retry occurs.
                    self._atomic_copy_regular_tail(
                        source,
                        destination_dir / "test-stdout.txt",
                        maximum_bytes=_VERIFIER_TIMEOUT_PARTIAL_MAX_BYTES,
                    )
                    record["partial_stdout"] = {
                        "path": archive_name,
                        **copy_result,
                    }
                except OSError as exc:
                    record["partial_stdout"] = {
                        "error": type(exc).__name__,
                        "path": archive_name,
                    }
        records = getattr(self, "_verifier_timeout_records", None)
        if not isinstance(records, list):
            records = []
            self._verifier_timeout_records = records
        records.append(record)
        self._last_verifier_timeout = record
        self._write_verifier_timeout_receipt()

    @staticmethod
    def _clear_verifier_retry_outputs(root: Path) -> None:
        """Empty one owned verifier directory without following guest symlinks."""

        if root.is_symlink():
            raise OSError(f"refusing to clear symlinked verifier directory: {root}")
        root.mkdir(parents=True, exist_ok=True)
        for child in root.iterdir():
            if child.is_symlink() or not child.is_dir():
                child.unlink(missing_ok=True)
            else:
                shutil.rmtree(child)

    def _restore_canonical_partial_stdout(self, record: dict[str, Any]) -> None:
        partial = record.get("partial_stdout")
        trial_paths = getattr(self, "trial_paths", None)
        verifier_dir = getattr(trial_paths, "verifier_dir", None)
        if not isinstance(partial, dict) or verifier_dir is None:
            return
        name = partial.get("path")
        if not isinstance(name, str) or Path(name).name != name:
            return
        host_dir = Path(verifier_dir)
        if host_dir.is_symlink():
            return
        try:
            self._atomic_copy_regular_tail(
                host_dir / name,
                host_dir / "test-stdout.txt",
                maximum_bytes=_VERIFIER_TIMEOUT_PARTIAL_MAX_BYTES,
            )
        except OSError:
            pass

    async def prepare_verifier_timeout_retry(self, timeout_sec: float | None) -> bool:
        """Restart a clean RPC channel for one reviewed bootstrap-only retry."""

        del timeout_sec  # Each Harbor invocation keeps its original timeout intact.
        record = getattr(self, "_last_verifier_timeout", None)
        if (
            not isinstance(record, dict)
            or record.get("retry_eligible") is not True
            or bool(getattr(self, "_verifier_timeout_retry_used", False))
        ):
            return False
        self._verifier_timeout_retry_used = True
        record["retry_state"] = "preparing"

        process = getattr(self, "_rpc_process", None)
        identity = getattr(self, "_rpc_identity", None)
        bridge = getattr(self, "_package_bridge_broker", None)
        package_lock = getattr(self, "_package_command_lock", None)
        bridge_healthy = bool(
            not getattr(self, "_package_bridge_enabled", False)
            or (
                bridge is not None
                and bridge.healthy
                and getattr(self, "_package_bridge_ready", False)
                and getattr(self, "_package_bridge_accepting_commands", False)
                and package_lock is not None
                and not package_lock.locked()
            )
        )
        if (
            getattr(self, "_rpc_poison_reason", None) != "wire_exec_cancelled"
            or bool(getattr(self, "_cleanup_incomplete", False))
            or process is None
            or getattr(process, "returncode", None) is None
            or not isinstance(identity, _SupervisorIdentity)
            or not self._rpc_cleanup_complete()
            or not bridge_healthy
        ):
            record["retry_state"] = "rejected_after_cleanup"
            self._write_verifier_timeout_receipt()
            return False

        try:
            rootfs = getattr(self, "_rootfs", None)
            if not isinstance(rootfs, Path):
                raise OSError("PRoot rootfs is unavailable")
            self._clear_verifier_retry_outputs(rootfs / "logs" / "verifier")
            trial_paths = getattr(self, "trial_paths", None)
            verifier_dir = getattr(trial_paths, "verifier_dir", None)
            if verifier_dir is not None:
                host_dir = Path(verifier_dir)
                for name in (
                    "test-stdout.txt",
                    "test-stderr.txt",
                    "reward.txt",
                    "reward.json",
                    "ctrf.json",
                ):
                    candidate = host_dir / name
                    if candidate.is_dir() and not candidate.is_symlink():
                        raise OSError(f"unexpected verifier output directory: {name}")
                    candidate.unlink(missing_ok=True)

            self._close_process_pipes(process)
            self._release_supervisor_identity(identity)
            self._rpc_process = None
            self._rpc_identity = None
            self._close_rpc_log_handle()
            await self._start_rpc_server()
            await self._verify_package_bridge()
        except asyncio.CancelledError as exc:
            record["retry_state"] = "restart_failed"
            record["retry_error"] = type(exc).__name__
            self._restore_canonical_partial_stdout(record)
            self._write_verifier_timeout_receipt()
            raise
        except Exception as exc:
            record["retry_state"] = "restart_failed"
            record["retry_error"] = type(exc).__name__
            self._restore_canonical_partial_stdout(record)
            self._write_verifier_timeout_receipt()
            return False

        record["retry_state"] = "running"
        record["retry_started_at"] = self._utc_now()
        self._active_verifier_timeout_retry_record = record
        self._write_verifier_timeout_receipt()
        return True

    async def prepare_shared_verifier_after_agent_timeout(self) -> bool:
        """Restore the RPC channel that an agent-phase cancellation retired.

        Harbor deliberately continues to the shared verifier after an agent
        timeout so the work already present in the task environment can still
        be scored.  Cancelling an in-flight RPC command must poison that byte
        stream, but it does not invalidate the unpacked rootfs.  Reuse the
        rootfs only after the cleanup supervisor proves that the old process
        tree is gone, then start a fresh command server for verification.

        A timeout that happened while the agent was outside ``exec`` leaves a
        healthy RPC channel and needs no restart.  Every ambiguous cleanup or
        package-bridge state fails closed and lets Harbor's normal verifier
        error handling decide the result.
        """

        process = getattr(self, "_rpc_process", None)
        if (
            not bool(getattr(self, "_rpc_poisoned", False))
            and process is not None
            and getattr(process, "returncode", None) is None
            and not bool(getattr(self, "_cleanup_incomplete", False))
        ):
            return True

        identity = getattr(self, "_rpc_identity", None)
        bridge = getattr(self, "_package_bridge_broker", None)
        package_lock = getattr(self, "_package_command_lock", None)
        bridge_healthy = bool(
            not getattr(self, "_package_bridge_enabled", False)
            or (
                bridge is not None
                and bridge.healthy
                and getattr(self, "_package_bridge_ready", False)
                and getattr(self, "_package_bridge_accepting_commands", False)
                and package_lock is not None
                and not package_lock.locked()
            )
        )
        if (
            getattr(self, "_rpc_poison_reason", None) != "wire_exec_cancelled"
            or bool(getattr(self, "_cleanup_incomplete", False))
            or process is None
            or getattr(process, "returncode", None) is None
            or not isinstance(identity, _SupervisorIdentity)
            or not self._rpc_cleanup_complete()
            or not bridge_healthy
        ):
            self._persist_rpc_diagnostics(
                "agent_timeout_verifier_rpc_restart_rejected",
                detail="old RPC cleanup or package bridge was not proven healthy",
            )
            return False

        try:
            rootfs = getattr(self, "_rootfs", None)
            if not isinstance(rootfs, Path) or not rootfs.is_dir():
                raise OSError("PRoot rootfs is unavailable")
            self._close_process_pipes(process)
            self._release_supervisor_identity(identity)
            self._rpc_process = None
            self._rpc_identity = None
            self._close_rpc_log_handle()
            await self._start_rpc_server()
            await self._verify_package_bridge()
        except asyncio.CancelledError:
            self._persist_rpc_diagnostics(
                "agent_timeout_verifier_rpc_restart_cancelled"
            )
            raise
        except Exception as exc:
            self._rpc_poisoned = True
            self._rpc_poison_reason = "agent_timeout_verifier_rpc_restart_failed"
            self._persist_rpc_diagnostics(
                "agent_timeout_verifier_rpc_restart_failed",
                detail=type(exc).__name__,
            )
            return False

        self._persist_rpc_diagnostics("agent_timeout_verifier_rpc_restarted")
        return True

    def finish_verifier_timeout_retry(self, *, succeeded: bool) -> None:
        record = getattr(self, "_active_verifier_timeout_retry_record", None)
        if not isinstance(record, dict) or record.get("retry_state") != "running":
            return
        record["retry_state"] = "succeeded" if succeeded else "failed"
        record["retry_finished_at"] = self._utc_now()
        self._active_verifier_timeout_retry_record = None
        self._write_verifier_timeout_receipt()

    def _rpc_returncode(self) -> int | None:
        process = getattr(self, "_rpc_process", None)
        identity = getattr(self, "_rpc_identity", None)
        if isinstance(identity, _SupervisorIdentity):
            if self._consume_cleanup_attestation(identity):
                primary_value = self._supervised_primary_returncode(identity)
                if primary_value is not None:
                    return primary_value
        value = getattr(process, "returncode", None)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def _persist_rpc_diagnostics(
        self,
        event: str,
        *,
        detail: str | None = None,
    ) -> None:
        """Best-effort durable PRoot evidence before the unpacked rootfs is removed."""

        now = self._utc_now()
        events = getattr(self, "_rpc_diagnostic_events", None)
        if not isinstance(events, list):
            events = []
            self._rpc_diagnostic_events = events
        record: dict[str, Any] = {"event": event, "recorded_at": now}
        if detail:
            record["detail"] = detail
        events.append(record)

        artifact_dir = self._runtime_artifact_dir()
        if artifact_dir is None:
            return
        log_source = getattr(self, "_rpc_log_path", None)
        log_handle = getattr(self, "_rpc_log_handle", None)
        log_target = artifact_dir / _PROOT_RUNTIME_LOG
        log_name: str | None = log_target.name if log_target.is_file() else None
        try:
            if log_handle is not None and not log_handle.closed:
                log_handle.flush()
                os.fsync(log_handle.fileno())
            if isinstance(log_source, Path) and log_source.is_file():
                temporary_log = log_target.with_name(
                    f".{log_target.name}-{uuid.uuid4().hex}.tmp"
                )
                try:
                    shutil.copyfile(log_source, temporary_log)
                    temporary_log.chmod(0o600)
                    with temporary_log.open("rb") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary_log, log_target)
                    log_name = log_target.name
                finally:
                    temporary_log.unlink(missing_ok=True)
        except OSError:
            # Diagnostics must never replace the original environment exception.
            pass

        returncode = self._rpc_returncode()
        cleanup_incomplete = bool(getattr(self, "_cleanup_incomplete", False))
        rpc_cleanup_complete = self._rpc_cleanup_complete()
        payload = {
            "schema": "terminal_bench_proot_runtime_v1",
            "session_id": getattr(self, "session_id", None),
            "rpc_pid": getattr(getattr(self, "_rpc_process", None), "pid", None),
            "rpc_returncode": returncode,
            "rpc_signal": (
                -returncode if returncode is not None and returncode < 0 else None
            ),
            "rpc_poisoned": bool(getattr(self, "_rpc_poisoned", False)),
            "rpc_poison_reason": getattr(self, "_rpc_poison_reason", None),
            "cleanup_incomplete": cleanup_incomplete,
            "rpc_cleanup_complete": rpc_cleanup_complete,
            "cleanup_complete": not cleanup_incomplete and rpc_cleanup_complete,
            "server_log": log_name,
            "events": list(events),
            "updated_at": now,
        }
        receipt = artifact_dir / _PROOT_RUNTIME_RECEIPT
        temporary = receipt.with_name(f".{receipt.name}-{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, receipt)
            directory_fd = os.open(artifact_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _bridge_helper_sha256(path: Path | None) -> str | None:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest() if path else None
        except OSError:
            return None

    def _package_bridge_cleanup_complete(self) -> bool:
        broker = getattr(self, "_package_bridge_broker", None)
        endpoint = getattr(self, "_package_bridge_socket_path", None)
        runtime_dir = getattr(self, "_package_bridge_runtime_dir", None)
        package_lock = getattr(self, "_package_command_lock", None)
        return bool(
            (broker is None or broker.cleanup_complete)
            and (endpoint is None or not (endpoint.exists() or endpoint.is_symlink()))
            and (
                runtime_dir is None
                or not (runtime_dir.exists() or runtime_dir.is_symlink())
            )
            and (package_lock is None or not package_lock.locked())
        )

    def _rpc_cleanup_complete(self) -> bool:
        process = getattr(self, "_rpc_process", None)
        if process is None:
            return not bool(getattr(self, "_rpc_cleanup_required", False))
        identity = getattr(self, "_rpc_identity", None)
        if not isinstance(identity, _SupervisorIdentity):
            return False
        return self._supervisor_cleanup_complete(process)

    def _record_cleanup_incomplete(
        self,
        event: str,
        *,
        detail: str | None = None,
    ) -> None:
        """Make an unproven process-tree cleanup durably block bundle deletion."""

        self._cleanup_incomplete = True
        self._persist_rpc_diagnostics(event, detail=detail)

    def _sanitized_package_bridge_events(self) -> list[dict[str, Any]]:
        broker = getattr(self, "_package_bridge_broker", None)
        source = list(getattr(broker, "events", ())) if broker is not None else []
        allowed = {
            "event",
            "command",
            "mutating",
            "return_code",
            "peer_pid",
            "peer_uid",
            "peer_gid",
            "error_type",
            "detail",
        }
        sanitized: list[dict[str, Any]] = []
        for raw in source:
            if not isinstance(raw, dict):
                continue
            record: dict[str, Any] = {}
            for key in allowed:
                value = raw.get(key)
                if isinstance(value, (str, int, bool)) or value is None:
                    record[key] = value[:1_024] if isinstance(value, str) else value
            if record:
                sanitized.append(record)
        return sanitized

    def _persist_package_bridge_diagnostics(
        self,
        status: str,
        event: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        """Persist bounded bridge provenance without argv, input, output, or env."""

        if not bool(getattr(self, "_package_bridge_enabled", False)):
            return
        now = self._utc_now()
        self._package_bridge_status = status
        events = getattr(self, "_package_bridge_diagnostic_events", None)
        if not isinstance(events, list):
            events = []
            self._package_bridge_diagnostic_events = events
        record: dict[str, Any] = {"event": event, "recorded_at": now}
        if error is not None:
            record["error_type"] = type(error).__name__
        events.append(record)
        artifact_dir = self._runtime_artifact_dir()
        if artifact_dir is None:
            return

        payload: dict[str, Any] = {
            "schema": "terminal_bench_proot_package_bridge_v1",
            "session_id": getattr(self, "session_id", None),
            "enabled": True,
            "policy": PROOT_PACKAGE_BRIDGE_POLICY,
            "failure_policy": "closed",
            "status": status,
            "ready": bool(getattr(self, "_package_bridge_ready", False)),
            "healthy": bool(
                getattr(getattr(self, "_package_bridge_broker", None), "healthy", True)
            ),
            "fatal_error": getattr(
                getattr(self, "_package_bridge_broker", None), "fatal_error", None
            ),
            "cleanup_complete": self._package_bridge_cleanup_complete(),
            "rpc_cleanup_complete": self._rpc_cleanup_complete(),
            "guest_socket_path": str(GUEST_SOCKET_PATH),
            "bound_commands": list(getattr(self, "_package_bridge_bound_commands", ())),
            "wrapper_sha256": self._bridge_helper_sha256(
                getattr(self, "_package_bridge_wrapper_path", None)
            ),
            "launcher_sha256": self._bridge_helper_sha256(
                getattr(self, "_package_bridge_launcher_path", None)
            ),
            "source_image_sha256": (
                hashlib.sha256(str(source_image).encode("utf-8")).hexdigest()
                if (source_image := getattr(self, "_source_image", None))
                else None
            ),
            "oci_manifest_digest": getattr(self, "_oci_manifest_digest", None),
            "stdin_policy": "non_tty_bounded_until_eof_tty_empty",
            "events": list(events),
            "broker_events": self._sanitized_package_bridge_events(),
            "updated_at": now,
        }
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error)[:4_096],
            }
        receipt = artifact_dir / PROOT_PACKAGE_BRIDGE_RECEIPT
        temporary = receipt.with_name(f".{receipt.name}-{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, receipt)
            directory_fd = os.open(artifact_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        finally:
            temporary.unlink(missing_ok=True)

    def _set_guest_utility_failure(self, stage: str, exc: BaseException) -> None:
        if self._guest_utility_receipt is None:
            return
        self._guest_utility_receipt.update(
            {
                "status": "failed",
                "stage": stage,
                "finished_at": self._utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        self._write_guest_utility_receipt()

    @staticmethod
    def _read_os_release(path: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line or raw_line.startswith("#") or "=" not in raw_line:
                continue
            key, raw_value = raw_line.split("=", 1)
            try:
                parsed = shlex.split(raw_value)
            except ValueError:
                parsed = []
            values[key] = parsed[0] if len(parsed) == 1 else raw_value.strip('"')
        return values

    async def _run_guest_utility_install_stage(
        self,
        *,
        stage: str,
        command: str,
        max_attempts: int = 1,
        retry_backoff_seconds: tuple[float, ...] = (),
        timeout_sec: int = _PROOT_GUEST_UTILITY_INSTALL_TIMEOUT_SEC,
    ) -> ExecResult:
        if self._guest_utility_receipt is None:
            raise RuntimeError("guest utility receipt was not initialized")
        if (
            max_attempts < 1
            or len(retry_backoff_seconds) != max_attempts - 1
            or any(delay < 0 for delay in retry_backoff_seconds)
            or isinstance(timeout_sec, bool)
            or timeout_sec <= 0
        ):
            raise RuntimeError(
                "invalid bounded retry policy for PRoot guest utility stage "
                f"{stage!r}"
            )
        attempt_records = self._guest_utility_receipt.get("install_attempts")
        if not isinstance(attempt_records, list):
            raise RuntimeError("guest utility receipt has no attempt journal")

        for attempt_number in range(1, max_attempts + 1):
            started_at = self._utc_now()
            attempt_record: dict[str, Any] = {
                "stage": stage,
                "attempt": attempt_number,
                "max_attempts": max_attempts,
                "command": command,
                "status": "running",
                "started_at": started_at,
                "timeout_seconds": timeout_sec,
            }
            attempt_records.append(attempt_record)
            self._guest_utility_receipt.update(
                {
                    "stage": stage,
                    "last_command": command,
                    "last_command_attempt": attempt_number,
                    "last_command_started_at": started_at,
                }
            )
            self._write_guest_utility_receipt()
            try:
                result = await self._direct_exec(
                    command,
                    "/",
                    self._image_env,
                    timeout_sec,
                )
            except BaseException as exc:
                attempt_record.update(
                    {
                        "status": "exception",
                        "finished_at": self._utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                self._write_guest_utility_receipt()
                raise

            result_record = {
                "return_code": result.return_code,
                "stdout_tail": self._output_tail(result.stdout),
                "stderr_tail": self._output_tail(result.stderr),
            }
            combined_output = "\n".join(
                value
                for value in (
                    result_record["stdout_tail"],
                    result_record["stderr_tail"],
                )
                if value
            )
            transport_signals = (
                self._guest_utility_apt_transport_signals(combined_output)
                if stage in {"apt_update", "apt_install"}
                else ()
            )
            incomplete_update_signals = (
                self._guest_utility_apt_update_incomplete_signals(
                    combined_output
                )
                if stage == "apt_update" and result.return_code == 0
                else ()
            )
            semantic_failure = bool(incomplete_update_signals)
            if transport_signals:
                result_record["transport_signals"] = list(transport_signals)
            if semantic_failure:
                result_record.update(
                    {
                        "semantic_failure": (
                            _PROOT_GUEST_UTILITY_APT_UPDATE_SEMANTIC_FAILURE
                        ),
                        "semantic_failure_signals": list(
                            incomplete_update_signals
                        ),
                    }
                )
            attempt_failed = result.return_code != 0 or semantic_failure
            attempt_record.update(
                {
                    "status": "failed" if attempt_failed else "succeeded",
                    "finished_at": self._utc_now(),
                    "result": result_record,
                }
            )
            self._guest_utility_receipt["last_command_result"] = result_record
            if not attempt_failed:
                self._write_guest_utility_receipt()
                return result

            if attempt_number < max_attempts:
                backoff_seconds = retry_backoff_seconds[attempt_number - 1]
                attempt_record.update(
                    {
                        "retry_scheduled": True,
                        "backoff_seconds": backoff_seconds,
                        "next_attempt": attempt_number + 1,
                    }
                )
                self._write_guest_utility_receipt()
                await asyncio.sleep(backoff_seconds)
                continue

            attempt_record["retry_scheduled"] = False
            stage_transport_signals: list[str] = []
            for recorded_attempt in attempt_records:
                if recorded_attempt.get("stage") != stage:
                    continue
                recorded_result = recorded_attempt.get("result")
                if not isinstance(recorded_result, dict):
                    continue
                for signal_name in recorded_result.get("transport_signals", []):
                    if signal_name not in stage_transport_signals:
                        stage_transport_signals.append(signal_name)
            self._guest_utility_receipt.update(
                {
                    "transport_failure": bool(stage_transport_signals),
                    "transport_signals": stage_transport_signals,
                }
            )
            self._write_guest_utility_receipt()
            detail = self._output_tail(result.stderr or result.stdout).strip()
            if semantic_failure:
                raise RuntimeError(
                    f"PRoot guest utility {stage} failed despite exit status 0: "
                    f"{detail}"
                )
            raise RuntimeError(
                f"PRoot guest utility {stage} failed with exit status "
                f"{result.return_code}: {detail}"
            )

        raise AssertionError("bounded PRoot guest utility retry loop fell through")

    async def _install_guest_utilities(self) -> None:
        if self._guest_utility_profile is None:
            return
        if self._rootfs is None:
            raise RuntimeError("PRoot rootfs is unavailable for utility installation")
        self._guest_utility_receipt = {
            "schema": _PROOT_GUEST_UTILITY_RECEIPT_SCHEMA,
            "schema_version": 2,
            "policy": PROOT_GUEST_UTILITY_POLICY,
            "profile": self._guest_utility_profile,
            "session_id": self.session_id,
            "status": "running",
            "stage": "distro_preflight",
            "started_at": self._utc_now(),
            "source_image": self._source_image,
            "oci_manifest_digest": self._oci_manifest_digest,
            "requested_commands": list(PROOT_GUEST_UTILITY_COMMANDS),
            "apt_packages": list(PROOT_GUEST_UTILITY_APT_PACKAGES),
            "install_transport": "direct_chroot_compatibility",
            "verification_transport": "long_lived_proot_rpc",
            "failure_policy": "closed",
            "benchmark_environment_delta": True,
            "package_versions_unpinned": True,
            "install_attempts": [],
            "apt_retry_policy": {
                "stages": ["apt_update", "apt_install"],
                "max_attempts": _PROOT_GUEST_UTILITY_APT_ATTEMPTS,
                "backoff_seconds": list(
                    _PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS
                ),
                "inner_acquire_retries": _PROOT_GUEST_UTILITY_APT_INNER_RETRIES,
                "http_connect_and_read_timeout_seconds": (
                    _PROOT_GUEST_UTILITY_APT_NETWORK_TIMEOUT_SEC
                ),
                "https_connect_and_read_timeout_seconds": (
                    _PROOT_GUEST_UTILITY_APT_NETWORK_TIMEOUT_SEC
                ),
                "per_attempt_timeout_seconds": (
                    _PROOT_GUEST_UTILITY_APT_ATTEMPT_TIMEOUT_SEC
                ),
            },
        }
        self._write_guest_utility_receipt()
        stage = "distro_preflight"
        apt_transport_config = None
        try:
            os_release = self._read_os_release(self._rootfs / "etc/os-release")
            distro_id = os_release.get("ID", "").lower()
            if distro_id not in _PROOT_GUEST_UTILITY_SUPPORTED_DISTROS:
                raise RuntimeError(
                    "The configured guest utility profile requires a Debian/Ubuntu "
                    f"rootfs, got ID={distro_id!r}"
                )
            for executable in ("usr/bin/apt-get", "usr/bin/dpkg-query"):
                if not (self._rootfs / executable).exists():
                    raise RuntimeError(
                        f"The configured guest utility profile requires /{executable}"
                    )
            self._guest_utility_receipt["distro"] = {
                "id": distro_id,
                "version_id": os_release.get("VERSION_ID"),
                "pretty_name": os_release.get("PRETTY_NAME"),
            }
            stage = "architecture_preflight"
            architecture = await self._run_guest_utility_install_stage(
                stage=stage,
                command="dpkg --print-architecture",
            )
            architecture_name = (architecture.stdout or "").strip()
            if architecture_name != "amd64":
                raise RuntimeError(
                    "The configured guest utility profile requires amd64, got "
                    f"{architecture_name!r}"
                )
            self._guest_utility_receipt["architecture"] = architecture_name

            from eval.terminal_bench.proot_apt_sources import prepare_bullseye_security_snapshot
            source_policy = prepare_bullseye_security_snapshot(self._rootfs, os_release)
            if source_policy is not None:
                self._guest_utility_receipt["apt_source_policy"] = source_policy
                self._write_guest_utility_receipt()

            from eval.terminal_bench.proot_apt_transport import prepare_apt_direct_transport
            transport_task = asyncio.create_task(asyncio.to_thread(
                prepare_apt_direct_transport, self._rootfs
            ))
            try:
                apt_transport_config, transport_policy = await asyncio.shield(transport_task)
            except BaseException:
                # A cancelled to_thread keeps running. Collect its owned file
                # so the finally below cannot leave a policy in the guest.
                apt_transport_config, _ = await asyncio.shield(transport_task)
                raise
            self._guest_utility_receipt["apt_transport_policy"] = transport_policy
            self._write_guest_utility_receipt()

            stage = "apt_update"
            await self._run_guest_utility_install_stage(
                stage=stage,
                command=(
                    "DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none "
                    f"apt-get {_PROOT_GUEST_UTILITY_APT_OPTIONS} update"
                ),
                max_attempts=_PROOT_GUEST_UTILITY_APT_ATTEMPTS,
                retry_backoff_seconds=(
                    _PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS
                ),
                timeout_sec=_PROOT_GUEST_UTILITY_APT_ATTEMPT_TIMEOUT_SEC,
            )
            stage = "apt_install"
            await self._run_guest_utility_install_stage(
                stage=stage,
                command=(
                    "DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none "
                    f"apt-get {_PROOT_GUEST_UTILITY_APT_OPTIONS} install -y "
                    "--no-install-recommends --no-upgrade file xxd ripgrep"
                ),
                max_attempts=_PROOT_GUEST_UTILITY_APT_ATTEMPTS,
                retry_backoff_seconds=(
                    _PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS
                ),
                timeout_sec=_PROOT_GUEST_UTILITY_APT_ATTEMPT_TIMEOUT_SEC,
            )
            stage = "dpkg_query"
            package_result = await self._run_guest_utility_install_stage(
                stage=stage,
                command=(
                    "dpkg-query -W -f='${Package}\\t${Version}\\n' " "file xxd ripgrep"
                ),
            )
            package_versions: dict[str, str] = {}
            for line in (package_result.stdout or "").splitlines():
                package, separator, version = line.partition("\t")
                if separator and package and version:
                    package_versions[package] = version
            if set(package_versions) != set(PROOT_GUEST_UTILITY_APT_PACKAGES):
                raise RuntimeError(
                    "dpkg-query did not prove all requested guest utility packages: "
                    f"{package_versions!r}"
                )
            self._guest_utility_receipt.update(
                {
                    "package_versions": package_versions,
                    "status": "installed_pending_proot_verification",
                    "stage": "proot_rpc_start",
                }
            )
            self._write_guest_utility_receipt()
        except asyncio.CancelledError as exc:
            self._set_guest_utility_failure(stage, exc)
            raise
        except Exception as exc:
            self._set_guest_utility_failure(stage, exc)
            raise
        finally:
            if apt_transport_config is not None:
                apt_transport_config.unlink(missing_ok=True)

    async def _verify_guest_utilities(self) -> None:
        if self._guest_utility_profile is None:
            return
        if self._guest_utility_receipt is None:
            raise RuntimeError("guest utility install receipt is unavailable")
        stage = "proot_rpc_verification"
        try:
            binaries: dict[str, dict[str, str]] = {}
            for command_name in PROOT_GUEST_UTILITY_COMMANDS:
                resolved = await self._wire_exec(
                    f"command -v {command_name}",
                    "/",
                    self._image_env,
                    timeout_sec=10,
                )
                binary_path = (resolved.stdout or "").strip()
                if resolved.return_code != 0 or not binary_path.startswith("/"):
                    raise RuntimeError(
                        f"command -v {command_name} failed inside the PRoot view"
                    )
                digest = await self._wire_exec(
                    f"sha256sum -- {shlex.quote(binary_path)}",
                    "/",
                    self._image_env,
                    timeout_sec=10,
                )
                sha256 = (digest.stdout or "").split(maxsplit=1)[0]
                if (
                    digest.return_code != 0
                    or len(sha256) != 64
                    or any(character not in "0123456789abcdef" for character in sha256)
                ):
                    raise RuntimeError(
                        f"sha256 verification failed for guest command {command_name}"
                    )
                version_result = await self._wire_exec(
                    _PROOT_GUEST_UTILITY_VERSION_COMMANDS[command_name],
                    "/",
                    self._image_env,
                    timeout_sec=10,
                )
                version_lines = [
                    line.strip()
                    for line in (version_result.stdout or "").splitlines()
                    + (version_result.stderr or "").splitlines()
                    if line.strip()
                ]
                if version_result.return_code != 0 or not version_lines:
                    raise RuntimeError(
                        f"version verification failed for guest command {command_name}"
                    )
                binaries[command_name] = {
                    "path": binary_path,
                    "sha256": sha256,
                    "version_line": version_lines[0],
                }
            self._guest_utility_receipt.update(
                {
                    "binaries": binaries,
                    "status": "ok",
                    "stage": "complete",
                    "finished_at": self._utc_now(),
                }
            )
            self._write_guest_utility_receipt()
        except asyncio.CancelledError as exc:
            self._set_guest_utility_failure(stage, exc)
            raise
        except Exception as exc:
            self._set_guest_utility_failure(stage, exc)
            raise

    def _prepare_rootfs(self) -> None:
        assert self._rootfs is not None
        self._prepare_network_files()
        for relative in (
            "dev",
            "dev/pts",
            "logs",
            "logs/agent",
            "logs/verifier",
            "logs/artifacts",
            "tests",
            "solution",
            "harbor/skills",
            "proc/self",
            "tmp",
            ".harbor",
        ):
            (self._rootfs / relative).mkdir(parents=True, exist_ok=True)
        (self._rootfs / "tmp").chmod(0o1777)
        devices = {
            "null": (1, 3),
            "zero": (1, 5),
            "random": (1, 8),
            "urandom": (1, 9),
            "tty": (5, 0),
            "ptmx": (5, 2),
        }
        for name, (major, minor) in devices.items():
            device_path = self._rootfs / "dev" / name
            if device_path.exists() or device_path.is_symlink():
                device_path.unlink()
            os.mknod(
                device_path,
                stat.S_IFCHR | 0o666,
                os.makedev(major, minor),
            )
            device_path.chmod(0o666)
        # The package-manager compatibility path below still executes through
        # chroot, where procfs cannot be mounted in this nested container. Keep
        # the historical uv fallback for commands launched by package hooks.
        # All ordinary agent/verifier commands execute through PRoot and see
        # the real host procfs instead (see _start_rpc_server).
        proc_self_exe = self._rootfs / "proc/self/exe"
        proc_self_exe.unlink(missing_ok=True)
        proc_self_exe.symlink_to("/root/.local/bin/uv")
        shutil.copy2(
            self._build_rpc_server(), self._rootfs / ".harbor/proot_rpc_server"
        )
        if bool(getattr(self, "_package_bridge_enabled", False)):
            prepare_guest_bind_directory(self._rootfs)

    @staticmethod
    def _replace_private_runtime_file(target: Path, payload: str) -> None:
        """Atomically replace one OCI runtime-managed file inside a rootfs.

        OCI layers commonly contain empty placeholders or symlinks for files
        that Docker supplies at container creation time.  Never write through
        an image-controlled symlink: materialize a private regular file in the
        unpacked per-trial rootfs instead.
        """

        if target.exists() and target.is_dir() and not target.is_symlink():
            raise RuntimeError(f"PRoot runtime file target is a directory: {target}")
        temporary = target.with_name(f".{target.name}-{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o644)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _prepare_network_files(self) -> None:
        """Create the private network files normally injected by Docker."""

        assert self._rootfs is not None
        etc_dir = self._rootfs / "etc"
        if etc_dir.is_symlink() or (etc_dir.exists() and not etc_dir.is_dir()):
            raise RuntimeError("PRoot rootfs /etc must be a real directory")
        etc_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolver = _HOST_RESOLV_CONF.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(
                "PRoot could not snapshot the host resolver configuration"
            ) from exc
        if not any(
            re.match(r"nameserver\s+\S+", line.lstrip())
            for line in resolver.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", ";"))
        ):
            raise RuntimeError(
                "PRoot requires a host resolver configuration with a nameserver"
            )
        if not resolver.endswith("\n"):
            resolver += "\n"

        self._replace_private_runtime_file(etc_dir / "hosts", _PROOT_RUNTIME_HOSTS)
        self._replace_private_runtime_file(etc_dir / "resolv.conf", resolver)

    def _build_rpc_server(self) -> Path:
        source_hash = hashlib.sha256(_RPC_SERVER_SOURCE.read_bytes()).hexdigest()[:16]
        build_dir = self._cache_dir / "helpers"
        binary = build_dir / f"proot_rpc_server-{source_hash}"
        if binary.is_file():
            return binary
        build_dir.mkdir(parents=True, exist_ok=True)
        temporary = build_dir / f".{binary.name}-{uuid.uuid4().hex}.tmp"
        result = subprocess.run(
            [
                shutil.which("gcc") or "gcc",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-static",
                "-o",
                str(temporary),
                str(_RPC_SERVER_SOURCE),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Failed to compile the static PRoot helper:\n"
                + result.stderr.decode(errors="replace")
            )
        temporary.chmod(0o755)
        temporary.replace(binary)
        return binary

    async def _shutdown_package_bridge(self) -> None:
        if not bool(getattr(self, "_package_bridge_enabled", False)):
            return
        if self._package_bridge_cleanup_complete():
            self._package_bridge_ready = False
            self._package_bridge_accepting_commands = False
            return
        self._package_bridge_ready = False
        self._package_bridge_accepting_commands = False
        self._persist_package_bridge_diagnostics("stopping", "broker_stop_started")
        broker = getattr(self, "_package_bridge_broker", None)
        try:
            if broker is not None:
                await broker.stop()
            package_lock = getattr(self, "_package_command_lock", None)
            if package_lock is not None:
                async with package_lock:
                    pass
            runtime_dir = getattr(self, "_package_bridge_runtime_dir", None)
            if runtime_dir is not None and runtime_dir.exists():
                resolved = runtime_dir.resolve()
                if (
                    resolved.parent != Path("/tmp")
                    or not resolved.name.startswith("harbor-pkg-")
                    or runtime_dir.is_symlink()
                ):
                    raise PackageBridgeError(
                        "refusing to remove an unexpected package bridge runtime path"
                    )
                shutil.rmtree(resolved)
            self._package_bridge_runtime_dir = None
            self._package_bridge_socket_path = None
            self._package_bridge_bind_args = ()
        except BaseException as exc:
            self._persist_package_bridge_diagnostics(
                "cleanup_failed", "broker_stop_failed", error=exc
            )
            raise
        self._persist_package_bridge_diagnostics("stopped", "broker_stopped")

    async def _start_package_bridge(self) -> None:
        if not bool(getattr(self, "_package_bridge_enabled", False)):
            return
        if bool(getattr(self, "_package_bridge_ready", False)):
            raise PackageBridgeError("PRoot package bridge is already ready")
        self._package_bridge_accepting_commands = False
        assert self._rootfs is not None
        self._persist_package_bridge_diagnostics("starting", "startup_started")
        try:
            runtime_dir = Path(
                tempfile.mkdtemp(prefix="harbor-pkg-", dir="/tmp")
            ).resolve()
            runtime_dir.chmod(0o700)
            endpoint = runtime_dir / "control.sock"
            self._package_bridge_runtime_dir = runtime_dir
            self._package_bridge_socket_path = endpoint
            if len(os.fsencode(endpoint)) >= 104:
                raise PackageBridgeError("package bridge Unix socket path is too long")
            helper_cache = self._cache_dir / "helpers"
            wrapper = build_guest_wrapper(cache_dir=helper_cache)
            launcher = build_chroot_launcher(cache_dir=helper_cache)
            self._package_bridge_wrapper_path = wrapper
            self._package_bridge_launcher_path = launcher
            prepare_guest_bind_directory(self._rootfs)

            for command_name in _PROOT_PACKAGE_BRIDGE_REQUIRED_COMMANDS:
                guest_path = COMMAND_PATHS[command_name]
                physical = self._rootfs.joinpath(*guest_path.parts[1:])
                try:
                    info = physical.stat(follow_symlinks=False)
                except OSError as exc:
                    raise PackageBridgeError(
                        f"required rootfs package command is unavailable: {guest_path}"
                    ) from exc
                if not stat.S_ISREG(info.st_mode):
                    raise PackageBridgeError(
                        f"required rootfs package command is not regular: {guest_path}"
                    )

            broker = PackageCommandBroker(
                rootfs=self._rootfs,
                socket_path=endpoint,
                launcher_path=launcher,
                supervisor_path=self._build_rpc_server(),
                package_lock=self._package_command_lock,
            )
            self._package_bridge_broker = broker
            await broker.start()
            bind_args = build_proot_bind_arguments(
                rootfs=self._rootfs,
                wrapper_path=wrapper,
                socket_path=endpoint,
            )
            bound_commands: list[str] = []
            for command_name, guest_path in COMMAND_PATHS.items():
                physical = self._rootfs.joinpath(*guest_path.parts[1:])
                try:
                    info = physical.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    bound_commands.append(command_name)
            self._package_bridge_bind_args = tuple(bind_args)
            self._package_bridge_bound_commands = tuple(sorted(bound_commands))
            self._package_bridge_ready = True
            self._persist_package_bridge_diagnostics("ready", "broker_ready")
        except asyncio.CancelledError as exc:
            self._package_bridge_ready = False
            cleanup = asyncio.create_task(self._shutdown_package_bridge())
            with contextlib.suppress(BaseException):
                await asyncio.shield(cleanup)
            self._persist_package_bridge_diagnostics(
                "cancelled", "startup_cancelled", error=exc
            )
            raise
        except Exception as exc:
            self._package_bridge_ready = False
            cleanup = asyncio.create_task(self._shutdown_package_bridge())
            with contextlib.suppress(BaseException):
                await asyncio.shield(cleanup)
            self._persist_package_bridge_diagnostics(
                "failed", "startup_failed", error=exc
            )
            if isinstance(exc, PackageBridgeError):
                raise
            raise PackageBridgeError(
                f"PRoot package bridge startup failed: {exc}"
            ) from exc

    async def _verify_package_bridge(self) -> None:
        if not bool(getattr(self, "_package_bridge_enabled", False)):
            return
        broker = getattr(self, "_package_bridge_broker", None)
        endpoint = getattr(self, "_package_bridge_socket_path", None)
        if (
            broker is None
            or not broker.is_running
            or not broker.healthy
            or endpoint is None
            or not endpoint.is_socket()
            or not bool(getattr(self, "_package_bridge_ready", False))
        ):
            raise PackageBridgeError("PRoot package bridge is not ready for its probe")
        first_event = len(broker.events)
        result = await self._wire_exec(
            "/usr/bin/dpkg-query --version",
            "/",
            self._image_env,
            timeout_sec=_PROOT_PACKAGE_BRIDGE_PROBE_TIMEOUT_SEC,
        )
        deadline = time.monotonic() + 1.0
        completed = False
        while time.monotonic() < deadline:
            completed = any(
                event.get("event") == "command_completed"
                and event.get("command") == "dpkg-query"
                and event.get("return_code") == 0
                for event in broker.events[first_event:]
            )
            if completed:
                break
            await asyncio.sleep(0.01)
        if result.return_code != 0 or not completed:
            raise PackageBridgeError(
                "PRoot package bridge dpkg-query probe did not traverse the broker"
            )
        if not broker.healthy:
            raise PackageBridgeError(
                "PRoot package bridge became unhealthy during its probe"
            )
        self._persist_package_bridge_diagnostics("ready", "probe_completed")

    @staticmethod
    def _pidfd_is_live_child(pidfd: int) -> bool:
        """Prove that a pinned pidfd still names our live, waitable child."""

        try:
            result = os.waitid(
                os.P_PIDFD,
                pidfd,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except (AttributeError, ChildProcessError, PermissionError, OSError):
            return False
        return result is None

    @classmethod
    def _pin_supervisor_identity(
        cls,
        process: asyncio.subprocess.Process,
        cleanup_ack_fd: int,
        start_gate_fd: int,
    ) -> _SupervisorIdentity:
        """Pin process identity before any later await can expose PID reuse."""

        pid = process.pid
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 1:
            raise RuntimeError("cleanup supervisor returned an invalid PID")
        pidfd: int | None = None
        procfd: int | None = None
        try:
            pidfd = os.pidfd_open(pid, 0)
            if not cls._pidfd_is_live_child(pidfd):
                raise RuntimeError(
                    "cleanup supervisor exited before its identity was pinned"
                )
            procfd = os.open(
                f"/proc/{pid}",
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
            )
            # The old process could have exited and its numeric PID could have
            # been reused between pidfd_open() and open(/proc/PID).  Rechecking
            # the original pidfd rejects that race before publishing procfd.
            if not cls._pidfd_is_live_child(pidfd):
                raise RuntimeError(
                    "cleanup supervisor identity changed while pinning /proc"
                )
            return _SupervisorIdentity(
                process=process,
                pid=pid,
                pidfd=pidfd,
                procfd=procfd,
                cleanup_ack_fd=cleanup_ack_fd,
                start_gate_fd=start_gate_fd,
            )
        except BaseException:
            if procfd is not None:
                os.close(procfd)
            if pidfd is not None:
                os.close(pidfd)
            raise

    @staticmethod
    def _consume_supervisor_control(identity: _SupervisorIdentity) -> None:
        if identity.closed or identity.control_invalid or identity.control_eof:
            return
        if identity.cleanup_ack_fd < 0:
            identity.control_invalid = True
            return
        while True:
            try:
                payload = os.read(identity.cleanup_ack_fd, 64)
            except BlockingIOError:
                return
            except OSError:
                identity.control_invalid = True
                return
            if not payload:
                with contextlib.suppress(OSError):
                    os.close(identity.cleanup_ack_fd)
                identity.cleanup_ack_fd = -1
                identity.control_eof = True
                if len(identity.control_buffer) != _SUPERVISOR_CONTROL_FRAME_SIZE:
                    identity.control_invalid = True
                return
            identity.control_buffer.extend(payload)
            frame = identity.control_buffer
            if len(frame) > _SUPERVISOR_CONTROL_FRAME_SIZE:
                identity.control_invalid = True
                return
            if len(frame) >= 1:
                if bytes(frame[0:1]) != _SUPERVISOR_READY_ATTESTATION:
                    identity.control_invalid = True
                    return
                identity.ready_attested = True
            if len(frame) >= 2 and bytes(frame[1:2]) != _SUPERVISOR_STATUS_ATTESTATION:
                identity.control_invalid = True
                return
            if len(frame) >= 6 and identity.primary_wait_status is None:
                wait_status = struct.unpack("!I", bytes(frame[2:6]))[0]
                if wait_status > 0xFFFF or not (
                    os.WIFEXITED(wait_status) or os.WIFSIGNALED(wait_status)
                ):
                    identity.control_invalid = True
                    return
                identity.primary_wait_status = wait_status
            if len(frame) == _SUPERVISOR_CONTROL_FRAME_SIZE:
                if bytes(frame[6:7]) != _SUPERVISOR_CLEANUP_ATTESTATION:
                    identity.control_invalid = True
                    return
                identity.cleanup_attested = True

    @classmethod
    def _consume_cleanup_attestation(cls, identity: _SupervisorIdentity) -> bool:
        cls._consume_supervisor_control(identity)
        return bool(
            identity.cleanup_attested
            and identity.control_eof
            and identity.primary_wait_status is not None
            and getattr(identity.process, "returncode", None) == 0
            and not identity.control_invalid
        )

    @staticmethod
    def _supervised_primary_returncode(identity: _SupervisorIdentity) -> int | None:
        wait_status = identity.primary_wait_status
        if wait_status is None:
            return None
        if os.WIFEXITED(wait_status):
            return os.WEXITSTATUS(wait_status)
        if os.WIFSIGNALED(wait_status):
            return -os.WTERMSIG(wait_status)
        return None

    @classmethod
    async def _wait_supervisor_ready(cls, identity: _SupervisorIdentity) -> bool:
        deadline = asyncio.get_running_loop().time() + _SUPERVISOR_READY_TIMEOUT_SEC
        while asyncio.get_running_loop().time() < deadline:
            cls._consume_supervisor_control(identity)
            if identity.ready_attested and not identity.control_invalid:
                return True
            if (
                identity.control_invalid
                or identity.cleanup_ack_fd < 0
                or getattr(identity.process, "returncode", None) is not None
            ):
                return False
            await asyncio.sleep(0.01)
        return False

    @staticmethod
    def _finish_identity_gate(identity: _SupervisorIdentity, *, release: bool) -> bool:
        descriptor = identity.start_gate_fd
        if descriptor < 0:
            return not release
        succeeded = True
        if release:
            try:
                succeeded = os.write(descriptor, _SUPERVISOR_IDENTITY_GATE_RELEASE) == 1
            except OSError:
                succeeded = False
        with contextlib.suppress(OSError):
            os.close(descriptor)
        identity.start_gate_fd = -1
        return succeeded

    @staticmethod
    def _release_supervisor_identity(identity: _SupervisorIdentity) -> None:
        if identity.closed:
            return
        for descriptor_name in (
            "start_gate_fd",
            "cleanup_ack_fd",
            "procfd",
            "pidfd",
        ):
            descriptor = getattr(identity, descriptor_name)
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                setattr(identity, descriptor_name, -1)
        identity.closed = True

    def _retain_or_release_supervisor_identity(
        self, identity: _SupervisorIdentity
    ) -> None:
        if identity.closed:
            return
        if getattr(identity.process, "returncode", None) is not None:
            self._release_supervisor_identity(identity)
            return
        quarantined = getattr(self, "_quarantined_supervisors", None)
        if not isinstance(quarantined, list):
            quarantined = []
            self._quarantined_supervisors = quarantined
        if identity not in quarantined:
            quarantined.append(identity)

    @classmethod
    def _supervisor_cleanup_complete(cls, process: Any) -> bool:
        identity = getattr(process, "_r2eact_supervisor_identity", None)
        if not isinstance(identity, _SupervisorIdentity):
            return False
        cleanup_proven = cls._consume_cleanup_attestation(identity)
        return getattr(process, "returncode", None) is not None and cleanup_proven

    @classmethod
    async def _observe_supervisor_cleanup(cls, process: Any) -> bool:
        wait = getattr(process, "wait", None)
        if callable(wait) and getattr(process, "returncode", None) is None:
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(
                    asyncio.shield(wait()), timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC
                )
        return cls._supervisor_cleanup_complete(process)

    async def _retire_cancelled_spawn(self, process: Any) -> bool:
        self._signal_process(process, signal.SIGTERM)
        wait = getattr(process, "wait", None)
        if callable(wait):
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(
                    asyncio.shield(wait()), timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC
                )
        cleanup_proven = self._supervisor_cleanup_complete(process)
        if not cleanup_proven and getattr(process, "returncode", None) is None:
            cleanup_proven = await self._kill_process_group(process)
        return cleanup_proven

    async def _spawn_cleanup_supervisor(
        self,
        *command: str,
        **kwargs: Any,
    ) -> tuple[asyncio.subprocess.Process, _SupervisorIdentity]:
        """Spawn the host subreaper with an unforgeable cleanup ACK channel."""

        self._require_safe_cleanup_capability_boundary()
        cleanup_read_fd, cleanup_write_fd = os.pipe2(os.O_CLOEXEC)
        try:
            gate_read_fd, gate_write_fd = os.pipe2(os.O_CLOEXEC)
        except BaseException:
            os.close(cleanup_read_fd)
            os.close(cleanup_write_fd)
            raise
        os.set_blocking(cleanup_read_fd, False)
        process: asyncio.subprocess.Process | None = None
        identity: _SupervisorIdentity | None = None
        interrupted: asyncio.CancelledError | None = None
        try:
            spawn_task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    str(self._build_rpc_server()),
                    "--supervise-control-fds",
                    str(cleanup_write_fd),
                    str(gate_read_fd),
                    *command,
                    pass_fds=(cleanup_write_fd, gate_read_fd),
                    **kwargs,
                )
            )
        except BaseException:
            os.close(cleanup_read_fd)
            os.close(cleanup_write_fd)
            os.close(gate_read_fd)
            os.close(gate_write_fd)
            raise
        try:
            try:
                process = await asyncio.shield(spawn_task)
            except asyncio.CancelledError as exc:
                # Do not lose a child in the cancellation window between fork
                # and asyncio publishing Process.  Finish the shielded spawn,
                # pin its identity, retire it, and only then re-raise.
                interrupted = exc
                settle_task = asyncio.create_task(
                    asyncio.wait_for(
                        asyncio.shield(spawn_task),
                        timeout=_SUPERVISOR_READY_TIMEOUT_SEC,
                    )
                )
                while not settle_task.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.shield(settle_task)
                try:
                    process = settle_task.result()
                except asyncio.TimeoutError:
                    spawn_task.cancel()
                    with contextlib.suppress(BaseException):
                        await spawn_task
                    self._record_cleanup_incomplete(
                        "supervisor_spawn_cancellation_window_unresolved"
                    )
                    raise interrupted
        except BaseException:
            os.close(cleanup_read_fd)
            os.close(gate_write_fd)
            raise
        finally:
            os.close(cleanup_write_fd)
            os.close(gate_read_fd)
        try:
            identity = self._pin_supervisor_identity(
                process, cleanup_read_fd, gate_write_fd
            )
        except BaseException as exc:
            os.close(cleanup_read_fd)
            os.close(gate_write_fd)
            if process is not None:
                self._close_process_pipes(process)
                untracked = getattr(self, "_untracked_supervisors", None)
                if not isinstance(untracked, list):
                    untracked = []
                    self._untracked_supervisors = untracked
                untracked.append(process)
            self._record_cleanup_incomplete(
                "supervisor_identity_pin_failed", detail=type(exc).__name__
            )
            raise RuntimeError(
                "Host cleanup supervisor identity could not be pinned"
            ) from exc
        setattr(process, "_r2eact_supervisor_identity", identity)
        ready_task = asyncio.create_task(self._wait_supervisor_ready(identity))
        try:
            ready = await asyncio.shield(ready_task)
        except asyncio.CancelledError as exc:
            if interrupted is None:
                interrupted = exc
            while not ready_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(ready_task)
            ready = ready_task.result()
        if not ready:
            self._finish_identity_gate(identity, release=False)
            self._close_process_pipes(process)
            self._record_cleanup_incomplete("supervisor_readiness_attestation_missing")
            self._retain_or_release_supervisor_identity(identity)
            if interrupted is not None:
                raise interrupted
            raise RuntimeError("Host cleanup supervisor did not attest readiness")
        if interrupted is not None:
            self._finish_identity_gate(identity, release=False)
            cleanup_task = asyncio.create_task(self._retire_cancelled_spawn(process))
            while not cleanup_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(cleanup_task)
            cleanup_proven = cleanup_task.result()
            if not cleanup_proven:
                self._record_cleanup_incomplete(
                    "supervisor_spawn_cancelled_cleanup_incomplete"
                )
            self._retain_or_release_supervisor_identity(identity)
            raise interrupted
        if not self._finish_identity_gate(identity, release=True):
            self._record_cleanup_incomplete("supervisor_identity_gate_release_failed")
            self._retain_or_release_supervisor_identity(identity)
            raise RuntimeError(
                "Host cleanup supervisor identity gate could not be released"
            )
        return process, identity

    async def _start_rpc_server(self) -> None:
        assert self._bundle_dir is not None and self._rootfs is not None
        log_path = self._bundle_dir / "proot-server.log"
        self._rpc_log_path = log_path
        self._rpc_poisoned = False
        self._rpc_poison_reason = None
        self._rpc_diagnostic_events = []
        self._rpc_log_handle = log_path.open("ab")

        server_env = {
            "PATH": self._image_env.get(
                "PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "HOME": self._image_env.get("HOME", "/root"),
            "TERM": self._image_env.get("TERM", "xterm-256color"),
            "LANG": self._image_env.get("LANG", "C.UTF-8"),
        }
        proot_args = [
            str(self._configured_proot_binary()),
            "-0",
            "-r",
            str(self._rootfs),
            "-b",
            "/dev",
            "-b",
            "/proc",
        ]
        if bool(getattr(self, "_package_bridge_enabled", False)):
            broker = getattr(self, "_package_bridge_broker", None)
            endpoint = getattr(self, "_package_bridge_socket_path", None)
            if (
                broker is None
                or not broker.is_running
                or not broker.healthy
                or endpoint is None
                or not endpoint.is_socket()
                or not bool(getattr(self, "_package_bridge_ready", False))
                or not getattr(self, "_package_bridge_bind_args", ())
            ):
                raise PackageBridgeError(
                    "refusing to start PRoot without its pinned package bridge"
                )
            proot_args.extend(self._package_bridge_bind_args)
        proot_args.extend(("-w", "/", "/.harbor/proot_rpc_server"))
        self._rpc_cleanup_required = True
        self._rpc_process, self._rpc_identity = await self._spawn_cleanup_supervisor(
            *proot_args,
            env=server_env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self._rpc_log_handle,
            start_new_session=True,
        )
        await asyncio.sleep(0.1)
        if self._rpc_process.returncode is not None:
            raise RuntimeError(f"PRoot command server exited early; see {log_path}")
        probe = await self._wire_exec(
            "printf proot-ready", "/", self._image_env, timeout_sec=5
        )
        if probe.return_code != 0 or probe.stdout != "proot-ready":
            raise RuntimeError(f"PRoot command server failed its probe; see {log_path}")

    @staticmethod
    def _close_process_pipes(process: Any) -> None:
        """Close asyncio subprocess pipes, including pipes held by escaped children."""

        stdin = getattr(process, "stdin", None)
        if stdin is not None:
            with contextlib.suppress(Exception):
                stdin.close()
        transport = getattr(process, "_transport", None)
        getter = getattr(transport, "get_pipe_transport", None)
        if callable(getter):
            for descriptor in (0, 1, 2):
                with contextlib.suppress(Exception):
                    pipe_transport = getter(descriptor)
                    if pipe_transport is not None:
                        pipe_transport.close()

    def _close_rpc_log_handle(self) -> None:
        handle = getattr(self, "_rpc_log_handle", None)
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
        self._rpc_log_handle = None

    @classmethod
    def _pinned_direct_children(
        cls, identity: _SupervisorIdentity
    ) -> tuple[tuple[int, ...], bool]:
        """Read children relative to a pinned /proc dirfd, never a naked PID."""

        if identity.closed or not cls._pidfd_is_live_child(identity.pidfd):
            return (), True
        descriptor: int | None = None
        try:
            descriptor = os.open(
                f"task/{identity.pid}/children",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=identity.procfd,
            )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 65_536)
                if not chunk:
                    break
                total += len(chunk)
                if total > 1024 * 1024:
                    return (), True
                chunks.append(chunk)
            payload = b"".join(chunks).decode("ascii")
        except (UnicodeDecodeError, OSError):
            return (), True
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if not cls._pidfd_is_live_child(identity.pidfd):
            return (), True
        children: list[int] = []
        for field in payload.split():
            try:
                child = int(field)
            except ValueError:
                return tuple(children), True
            if child > 1:
                children.append(child)
        return tuple(children), False

    @classmethod
    async def _kill_process_group(cls, process: Any) -> bool:
        """Use pidfds to retire one subreaper-owned tree without PID reuse races.

        The C owner normally performs recursive cleanup itself.  This is only a
        bounded best-effort retirement path: ``/proc/.../children`` snapshots are
        not a kernel containment primitive and can race child reparenting.  Thus
        this path deliberately returns ``False`` after a forced owner kill and can
        never authorize rootfs deletion.  Only the C owner's complete protected
        control record is accepted as cleanup proof.
        """

        identity = getattr(process, "_r2eact_supervisor_identity", None)
        if not isinstance(identity, _SupervisorIdentity) or identity.closed:
            return False
        if getattr(process, "returncode", None) is not None:
            return cls._consume_cleanup_attestation(identity)

        for _ in range(25):
            children, failed = cls._pinned_direct_children(identity)
            if failed:
                return await cls._observe_supervisor_cleanup(process)
            if not children:
                break
            for child in children:
                try:
                    child_fd = os.pidfd_open(child, 0)
                except ProcessLookupError:
                    continue
                except (AttributeError, PermissionError, OSError):
                    return False
                try:
                    current_children, current_failed = cls._pinned_direct_children(
                        identity
                    )
                    if current_failed or child not in current_children:
                        return await cls._observe_supervisor_cleanup(process)
                    try:
                        signal.pidfd_send_signal(child_fd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except (AttributeError, PermissionError, OSError):
                        return False
                finally:
                    os.close(child_fd)
            try:
                signal.pidfd_send_signal(identity.pidfd, signal.SIGCONT)
            except ProcessLookupError:
                return await cls._observe_supervisor_cleanup(process)
            except (AttributeError, PermissionError, OSError):
                return False
            await asyncio.sleep(0.02)

        remaining, failed = cls._pinned_direct_children(identity)
        if remaining or failed:
            return await cls._observe_supervisor_cleanup(process)
        try:
            signal.pidfd_send_signal(identity.pidfd, signal.SIGKILL)
        except ProcessLookupError:
            return await cls._observe_supervisor_cleanup(process)
        except (AttributeError, PermissionError, OSError):
            return False
        wait = getattr(process, "wait", None)
        if not callable(wait):
            return False
        try:
            await asyncio.wait_for(
                asyncio.shield(wait()), timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC
            )
        except (asyncio.TimeoutError, ProcessLookupError):
            return False
        if getattr(process, "returncode", None) is None:
            return False
        # The observed children were retired, but procfs enumeration cannot
        # prove that no concurrent reparent escaped the final snapshot.
        return False

    @staticmethod
    def _signal_process(process: Any, signum: int) -> None:
        if getattr(process, "returncode", None) is not None:
            return
        identity = getattr(process, "_r2eact_supervisor_identity", None)
        if not isinstance(identity, _SupervisorIdentity) or identity.closed:
            return
        if not ProotEnvironment._pidfd_is_live_child(identity.pidfd):
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            signal.pidfd_send_signal(identity.pidfd, signum)

    async def _observe_rpc_exit(self, timeout: float) -> None:
        process = getattr(self, "_rpc_process", None)
        wait = getattr(process, "wait", None)
        if (
            process is None
            or not callable(wait)
            or getattr(process, "returncode", None) is not None
        ):
            return
        with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
            await asyncio.wait_for(asyncio.shield(wait()), timeout=max(0.01, timeout))

    async def _poison_rpc_transport(
        self,
        reason: str,
        *,
        observe_natural_exit: bool = False,
    ) -> None:
        """Make a canceled/truncated byte stream permanently unusable and retire it."""

        self._rpc_poisoned = True
        self._rpc_poison_reason = reason
        process = getattr(self, "_rpc_process", None)
        hard_cleanup_complete = True
        if process is not None and observe_natural_exit:
            await self._observe_rpc_exit(_PROOT_RPC_EXIT_OBSERVATION_GRACE_SEC)
        if process is not None and getattr(process, "returncode", None) is None:
            self._signal_process(process, signal.SIGTERM)
            await self._observe_rpc_exit(_PROOT_RPC_HARD_STOP_GRACE_SEC)
        if process is not None and getattr(process, "returncode", None) is None:
            hard_cleanup_complete = await self._kill_process_group(process)
            await self._observe_rpc_exit(_PROOT_RPC_HARD_STOP_GRACE_SEC)
        if process is not None:
            self._close_process_pipes(process)
        cleanup_complete = (
            hard_cleanup_complete
            and self._rpc_cleanup_complete()
            and not bool(getattr(self, "_cleanup_incomplete", False))
        )
        if cleanup_complete:
            self._persist_rpc_diagnostics("rpc_transport_poisoned", detail=reason)
        else:
            self._record_cleanup_incomplete(
                "rpc_transport_cleanup_incomplete", detail=reason
            )

    @staticmethod
    def _wire_string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("!I", len(encoded)) + encoded

    @contextlib.contextmanager
    def unbounded_wire_exec(self):
        """Disable per-command RPC deadlines only for the current async context.

        Long-running interactive agents use this inside Harbor's independently
        cancellable agent phase.  Startup probes, installation, artifact handling,
        and the verifier remain bounded because the ContextVar does not leak beyond
        that phase or into concurrent tasks.
        """

        token = self._unbounded_wire_exec.set(True)
        try:
            yield
        finally:
            self._unbounded_wire_exec.reset(token)

    async def _wire_exec(
        self,
        command: str,
        cwd: str,
        env: dict[str, str],
        timeout_sec: int | None,
    ) -> ExecResult:
        if getattr(self, "_rpc_poisoned", False):
            raise RuntimeError("PRoot command server is not running")
        if self._rpc_process is None or self._rpc_process.returncode is not None:
            self._rpc_poisoned = True
            self._rpc_poison_reason = "rpc_process_not_running"
            self._persist_rpc_diagnostics(
                "rpc_process_not_running", detail="wire exec rejected before request"
            )
            raise RuntimeError("PRoot command server is not running")
        assert self._rpc_process.stdin is not None
        assert self._rpc_process.stdout is not None
        timeout_ms = min(max(int((timeout_sec or 0) * 1000), 0), 0xFFFFFFFF)
        request = bytearray(b"\x01")
        request.extend(struct.pack("!I", timeout_ms))
        request.extend(self._wire_string(cwd))
        request.extend(self._wire_string(command))
        request.extend(struct.pack("!I", len(env)))
        for key, value in env.items():
            request.extend(self._wire_string(str(key)))
            request.extend(self._wire_string(str(value)))

        async with self._rpc_lock:
            # A prior request can poison the shared byte stream while this caller is
            # queued on the lock. Recheck at the write boundary so no request can be
            # appended after a canceled or truncated response.
            if getattr(self, "_rpc_poisoned", False):
                raise RuntimeError("PRoot command server is not running")
            if self._rpc_process.returncode is not None:
                self._rpc_poisoned = True
                self._rpc_poison_reason = "rpc_process_not_running"
                self._persist_rpc_diagnostics(
                    "rpc_process_not_running",
                    detail="wire exec rejected at request boundary",
                )
                raise RuntimeError("PRoot command server is not running")
            try:
                self._rpc_process.stdin.write(request)
                await self._rpc_process.stdin.drain()

                async def read_response() -> tuple[int, bytes, bytes]:
                    header = await self._rpc_process.stdout.readexactly(12)
                    return_code, stdout_size, stderr_size = struct.unpack(
                        "!iII", header
                    )
                    if (
                        stdout_size > _PROOT_RPC_MAX_FIELD_SIZE
                        or stderr_size > _PROOT_RPC_MAX_FIELD_SIZE
                    ):
                        raise RuntimeError(
                            "PRoot command server returned an invalid response size"
                        )
                    stdout = await self._rpc_process.stdout.readexactly(stdout_size)
                    stderr = await self._rpc_process.stdout.readexactly(stderr_size)
                    return return_code, stdout, stderr

                if timeout_sec is None:
                    # A missing command timeout is intentionally unbounded.  The
                    # surrounding Harbor phase remains responsible for cancellation;
                    # silently substituting a one-hour RPC deadline breaks that
                    # contract for long-running interactive agents such as Codex.
                    return_code, stdout, stderr = await read_response()
                else:
                    # The command server enforces timeout_sec and normally answers
                    # with status 124.  This small host-side grace is only a watchdog
                    # for a wedged server, not another command time budget.
                    response_timeout = (
                        float(timeout_sec) + _PROOT_RPC_TIMEOUT_RESPONSE_GRACE_SEC
                    )
                    return_code, stdout, stderr = await asyncio.wait_for(
                        read_response(), timeout=response_timeout
                    )
            except asyncio.CancelledError:
                # Harbor's verifier timeout cancels this coroutine before
                # Verifier.verify() can download /logs/verifier.  Snapshot the
                # still-live rootfs first; poisoning the RPC transport retires the
                # process tree, and normal environment cleanup may then delete it.
                with contextlib.suppress(Exception):
                    self._capture_verifier_timeout(command)
                await self._poison_rpc_transport("wire_exec_cancelled")
                raise
            except asyncio.IncompleteReadError as exc:
                await self._poison_rpc_transport(
                    "wire_response_eof", observe_natural_exit=True
                )
                raise RuntimeError(
                    "Lost contact with the PRoot command server"
                ) from exc
            except asyncio.TimeoutError as exc:
                returncode = self._rpc_process.returncode
                await self._poison_rpc_transport("wire_timeout_response_grace_expired")
                raise RuntimeError(
                    "PRoot command server missed its timeout-response grace period "
                    f"after a {float(timeout_sec):g}-second command timeout "
                    f"(server returncode={returncode!r})"
                ) from exc
            except (BrokenPipeError, ConnectionResetError) as exc:
                await self._poison_rpc_transport(
                    "wire_request_transport_closed", observe_natural_exit=True
                )
                raise RuntimeError(
                    "Lost contact with the PRoot command server"
                ) from exc
            except RuntimeError:
                await self._poison_rpc_transport("wire_protocol_invalid_response")
                raise
        return ExecResult(
            stdout=stdout.decode(errors="replace") or None,
            stderr=stderr.decode(errors="replace") or None,
            return_code=return_code,
        )

    def _rootfs_path(self, container_path: str | PurePosixPath) -> Path:
        if self._rootfs is None:
            raise RuntimeError("Environment has not started")
        posix_path = PurePosixPath(str(container_path))
        if ".." in posix_path.parts:
            raise ValueError(f"Unsafe container path: {container_path}")
        return self._rootfs.joinpath(*posix_path.parts[1:])

    def _require_healthy_package_bridge(self, stage: str) -> None:
        if not bool(getattr(self, "_package_bridge_enabled", False)):
            return
        broker = getattr(self, "_package_bridge_broker", None)
        if broker is None or not broker.healthy:
            reason = getattr(broker, "fatal_error", None) or "broker_unavailable"
            error = PackageBridgeError(
                f"PRoot package bridge infrastructure failed during {stage}: {reason}"
            )
            self._package_bridge_ready = False
            self._package_bridge_accepting_commands = False
            self._persist_package_bridge_diagnostics(
                "unhealthy", "broker_health_check_failed", error=error
            )
            raise error

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        del user  # TB2.1 images run as root; PRoot presents uid 0 inside the rootfs.
        merged_env = {**self._image_env, **(self._merge_env(env) or {})}
        self._require_healthy_package_bridge("exec_preflight")
        effective_cwd = cwd or self._image_workdir
        effective_timeout_sec = None if self._unbounded_wire_exec.get() else timeout_sec
        package_manager_command = re.match(
            r"^(?:set\s+-o\s+pipefail\s*;\s*)?"
            r"(?:[A-Z_][A-Z0-9_]*=[^ ]+\s+)*(?:sudo\s+)?"
            r"(?:apt-get|apt|dnf|yum|apk|pacman|zypper)\s",
            command.lstrip(),
        )
        # Execute both agent and verifier commands inside the same long-lived
        # PRoot view. In particular, the /dev and /proc binds above provide a
        # working devpts (/dev/ptmx and /dev/pts/*), process discovery, socket
        # inspection, and a truthful /proc/self/exe. The direct chroot path is
        # retained only as an explicit bridge-disabled compatibility path.
        # With the pinned bridge enabled, the bound guest wrapper delegates the
        # package mutation to the host chroot broker without nesting PRoot.
        if package_manager_command:
            if bool(getattr(self, "_package_bridge_enabled", False)):
                if not bool(getattr(self, "_package_bridge_accepting_commands", False)):
                    raise PackageBridgeError(
                        "PRoot package bridge is not accepting package commands"
                    )
                result = await self._wire_exec(
                    command, effective_cwd, merged_env, effective_timeout_sec
                )
            else:
                package_lock = getattr(self, "_package_command_lock", None)
                if package_lock is None:
                    result = await self._direct_exec(
                        command, effective_cwd, merged_env, effective_timeout_sec
                    )
                else:
                    async with package_lock:
                        result = await self._direct_exec(
                            command, effective_cwd, merged_env, effective_timeout_sec
                        )
        else:
            result = await self._wire_exec(
                command, effective_cwd, merged_env, effective_timeout_sec
            )
        self._require_healthy_package_bridge("exec_completion")
        return result

    async def _direct_exec(
        self,
        command: str,
        cwd: str,
        env: dict[str, str],
        timeout_sec: int | None,
    ) -> ExecResult:
        assert self._rootfs is not None
        wrapped = f"cd -- {shlex.quote(cwd)} && {command}"
        process, identity = await self._spawn_cleanup_supervisor(
            shutil.which("chroot") or "chroot",
            str(self._rootfs),
            "/bin/bash",
            "-lc",
            wrapped,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communicate_task = asyncio.create_task(process.communicate())
        try:
            if timeout_sec is None:
                stdout, stderr = await communicate_task
            else:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communicate_task), timeout=timeout_sec
                )
        except asyncio.TimeoutError:
            self._signal_process(process, signal.SIGTERM)
            forced_pipe_close = False
            hard_cleanup_complete = True
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=_DIRECT_EXEC_HARD_STOP_GRACE_SEC,
                )
            except asyncio.CancelledError:
                hard_cleanup_complete = await self._kill_process_group(process)
                self._close_process_pipes(process)
                communicate_task.cancel()
                with contextlib.suppress(BaseException):
                    await communicate_task
                if not hard_cleanup_complete or not self._supervisor_cleanup_complete(
                    process
                ):
                    self._record_cleanup_incomplete(
                        "direct_exec_cleanup_incomplete",
                        detail="cancelled during timeout cleanup",
                    )
                self._retain_or_release_supervisor_identity(identity)
                raise
            except (asyncio.TimeoutError, OSError):
                # A daemon can setsid/double-fork and retain these descriptors after
                # escaping the direct child's process group. Closing our transports
                # is the hard boundary that prevents communicate() from hanging.
                forced_pipe_close = True
                hard_cleanup_complete = await self._kill_process_group(process)
                self._close_process_pipes(process)
                communicate_task.cancel()
                with contextlib.suppress(BaseException):
                    await communicate_task
                stdout, stderr = b"", b""
            wait = getattr(process, "wait", None)
            if callable(wait) and getattr(process, "returncode", None) is None:
                try:
                    await asyncio.wait_for(
                        wait(), timeout=_DIRECT_EXEC_HARD_STOP_GRACE_SEC
                    )
                except asyncio.CancelledError:
                    if (
                        not hard_cleanup_complete
                        or not self._supervisor_cleanup_complete(process)
                    ):
                        self._record_cleanup_incomplete(
                            "direct_exec_cleanup_incomplete",
                            detail="cancelled while observing timeout cleanup",
                        )
                    self._retain_or_release_supervisor_identity(identity)
                    raise
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
            cleanup_proven = self._supervisor_cleanup_complete(process)
            if not hard_cleanup_complete or not cleanup_proven:
                self._record_cleanup_incomplete(
                    "direct_exec_cleanup_incomplete",
                    detail="timed-out process tree retirement was not proven",
                )
                self._retain_or_release_supervisor_identity(identity)
                raise RuntimeError(
                    "Direct command timed out, but its process-tree cleanup "
                    "could not be proven"
                )
            timeout_message = "Command timed out\n"
            if forced_pipe_close:
                timeout_message += "Detached output pipes were forcibly closed\n"
            result = ExecResult(
                stdout=stdout.decode(errors="replace") or None,
                stderr=(stderr.decode(errors="replace") + timeout_message),
                return_code=124,
            )
            self._retain_or_release_supervisor_identity(identity)
            return result
        except asyncio.CancelledError:
            self._signal_process(process, signal.SIGTERM)
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=_DIRECT_EXEC_HARD_STOP_GRACE_SEC,
                )
            hard_cleanup_complete = True
            if getattr(process, "returncode", None) is None:
                hard_cleanup_complete = await self._kill_process_group(process)
                self._close_process_pipes(process)
                communicate_task.cancel()
                with contextlib.suppress(BaseException):
                    await communicate_task
            if not hard_cleanup_complete or not self._supervisor_cleanup_complete(
                process
            ):
                self._record_cleanup_incomplete(
                    "direct_exec_cleanup_incomplete",
                    detail="cancelled process tree retirement was not proven",
                )
            self._retain_or_release_supervisor_identity(identity)
            raise
        if not self._supervisor_cleanup_complete(process):
            self._record_cleanup_incomplete(
                "direct_exec_cleanup_incomplete",
                detail="cleanup attestation missing after command completion",
            )
            self._retain_or_release_supervisor_identity(identity)
            raise RuntimeError(
                "Direct command completed without a valid host cleanup attestation"
            )
        primary_returncode = self._supervised_primary_returncode(identity)
        result = ExecResult(
            stdout=stdout.decode(errors="replace") or None,
            stderr=stderr.decode(errors="replace") or None,
            return_code=primary_returncode if primary_returncode is not None else 1,
        )
        self._retain_or_release_supervisor_identity(identity)
        return result

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        target = self._rootfs_path(target_path)
        if target.is_dir():
            target = target / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        target = self._rootfs_path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = self._rootfs_path(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = self._rootfs_path(source_dir)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=True)

    async def _shutdown_runtime_components(self) -> None:
        """Retire guest PRoot first, then the host package-command broker."""

        self._package_bridge_accepting_commands = False
        try:
            await self._shutdown_quarantined_supervisors()
            await self._shutdown_rpc_runtime()
        finally:
            await self._shutdown_package_bridge()

    async def _shutdown_quarantined_supervisors(self) -> None:
        for process in list(getattr(self, "_untracked_supervisors", ())):
            self._close_process_pipes(process)
            wait = getattr(process, "wait", None)
            if callable(wait) and getattr(process, "returncode", None) is None:
                with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                    await asyncio.wait_for(
                        asyncio.shield(wait()),
                        timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC,
                    )
            if getattr(process, "returncode", None) is not None:
                with contextlib.suppress(ValueError):
                    self._untracked_supervisors.remove(process)
            self._record_cleanup_incomplete("untracked_supervisor_cleanup_unproven")
        identities = list(getattr(self, "_quarantined_supervisors", ()))
        for identity in identities:
            process = identity.process
            if getattr(process, "returncode", None) is None:
                self._signal_process(process, signal.SIGTERM)
                wait = getattr(process, "wait", None)
                if callable(wait):
                    with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                        await asyncio.wait_for(
                            asyncio.shield(wait()),
                            timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC,
                        )
            cleanup_proven = self._supervisor_cleanup_complete(process)
            if not cleanup_proven and getattr(process, "returncode", None) is None:
                cleanup_proven = await self._kill_process_group(process)
            if cleanup_proven:
                self._release_supervisor_identity(identity)
                with contextlib.suppress(ValueError):
                    self._quarantined_supervisors.remove(identity)
            else:
                self._record_cleanup_incomplete(
                    "quarantined_supervisor_cleanup_incomplete"
                )
                if getattr(process, "returncode", None) is not None:
                    self._release_supervisor_identity(identity)
                    with contextlib.suppress(ValueError):
                        self._quarantined_supervisors.remove(identity)

    @override
    async def stop(self, delete: bool) -> None:
        """Retire the trial process tree even when Harbor cancels finalization."""

        shutdown_task = asyncio.create_task(self._shutdown_runtime_components())
        interrupted: asyncio.CancelledError | None = None
        try:
            await asyncio.shield(shutdown_task)
        except asyncio.CancelledError as exc:
            interrupted = exc
            process = getattr(self, "_rpc_process", None)
            if process is not None:
                self._signal_process(process, signal.SIGTERM)
                self._close_process_pipes(process)
            # Signaling the host-side subreaper starts recursive retirement without
            # killing the supervisor that must adopt escaped sessions. Give the
            # shielded cleanup one short chance to reap and persist its return code.
            with contextlib.suppress(
                asyncio.TimeoutError, asyncio.CancelledError, Exception
            ):
                await asyncio.wait_for(
                    asyncio.shield(shutdown_task),
                    timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC,
                )
        finally:
            if not shutdown_task.done():
                shutdown_task.cancel()
                with contextlib.suppress(BaseException):
                    await shutdown_task
            process = getattr(self, "_rpc_process", None)
            hard_cleanup_complete = True
            if process is not None and getattr(process, "returncode", None) is None:
                hard_cleanup_complete = await self._kill_process_group(process)
                self._close_process_pipes(process)
            self._rpc_poisoned = True
            if getattr(self, "_rpc_poison_reason", None) is None:
                self._rpc_poison_reason = "environment_stopped"
            bridge_cleanup_complete = self._package_bridge_cleanup_complete()
            rpc_cleanup_complete = self._rpc_cleanup_complete()
            if (
                not hard_cleanup_complete
                or not bridge_cleanup_complete
                or not rpc_cleanup_complete
            ):
                self._cleanup_incomplete = True
            runtime_cleanup_complete = (
                not bool(getattr(self, "_cleanup_incomplete", False))
                and bridge_cleanup_complete
                and rpc_cleanup_complete
            )
            if runtime_cleanup_complete:
                self._persist_rpc_diagnostics("environment_stop_finalized")
            else:
                self._record_cleanup_incomplete("environment_stop_incomplete")
            if bool(getattr(self, "_package_bridge_enabled", False)):
                if runtime_cleanup_complete:
                    self._persist_package_bridge_diagnostics(
                        "stopped", "environment_stop_finalized"
                    )
                else:
                    self._persist_package_bridge_diagnostics(
                        "cleanup_incomplete", "environment_stop_incomplete"
                    )
            self._close_rpc_log_handle()
            if (
                delete
                and not self._keep_containers
                and self._bundle_dir is not None
                and runtime_cleanup_complete
            ):
                shutil.rmtree(self._bundle_dir, ignore_errors=True)
            identity = getattr(self, "_rpc_identity", None)
            process = getattr(self, "_rpc_process", None)
            if (
                isinstance(identity, _SupervisorIdentity)
                and process is not None
                and getattr(process, "returncode", None) is not None
            ):
                self._release_supervisor_identity(identity)
        if interrupted is not None:
            raise interrupted

    async def _shutdown_rpc_runtime(self) -> None:
        process = getattr(self, "_rpc_process", None)
        if process is None:
            self._persist_rpc_diagnostics("environment_stop_without_rpc")
            return
        if process.returncode is None and not getattr(self, "_rpc_poisoned", False):
            try:
                await self._wire_exec(
                    "tmux kill-server >/dev/null 2>&1 || true",
                    "/",
                    self._image_env,
                    timeout_sec=5,
                )
            except Exception:
                pass
        if process.returncode is None and not getattr(self, "_rpc_poisoned", False):
            try:
                assert process.stdin is not None
                process.stdin.write(b"\x00")
                await process.stdin.drain()
                process.stdin.close()
            except Exception:
                pass
        try:
            if process.returncode is None:
                await asyncio.wait_for(
                    asyncio.shield(process.wait()),
                    timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC,
                )
        except asyncio.TimeoutError:
            self._signal_process(process, signal.SIGTERM)
            with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                await asyncio.wait_for(
                    asyncio.shield(process.wait()),
                    timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC,
                )
        finally:
            # The host-side supervisor exits only after it has reaped descendants.
            # SIGKILL is reserved for the bounded fallback when that contract stalls.
            hard_cleanup_complete = True
            if process.returncode is None:
                hard_cleanup_complete = await self._kill_process_group(process)
            self._close_process_pipes(process)
            try:
                if process.returncode is None:
                    await asyncio.wait_for(
                        asyncio.shield(process.wait()),
                        timeout=_PROOT_RPC_HARD_STOP_GRACE_SEC,
                    )
            except asyncio.CancelledError:
                self._rpc_poisoned = True
                if getattr(self, "_rpc_poison_reason", None) is None:
                    self._rpc_poison_reason = "environment_stopped"
                self._record_cleanup_incomplete(
                    "rpc_runtime_cleanup_incomplete",
                    detail="shutdown was cancelled before cleanup was proven",
                )
                raise
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
            self._rpc_poisoned = True
            if getattr(self, "_rpc_poison_reason", None) is None:
                self._rpc_poison_reason = "environment_stopped"
            cleanup_complete = (
                hard_cleanup_complete
                and self._rpc_cleanup_complete()
                and not bool(getattr(self, "_cleanup_incomplete", False))
            )
            if cleanup_complete:
                self._persist_rpc_diagnostics("rpc_runtime_stopped")
            else:
                self._record_cleanup_incomplete("rpc_runtime_cleanup_incomplete")
