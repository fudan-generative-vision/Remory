"""Private, crash-tolerant artifacts for local BrowseComp evaluations.

The helpers in this module deliberately keep sample content below an opaque
per-sample directory.  Job-level state is intended for aggregate monitoring
only and rejects common plaintext question/answer fields.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
OPAQUE_SAMPLE_ID_RE = re.compile(r"^bc_[0-9a-f]{32}$")
_JOB_LEVEL_SENSITIVE_KEYS = frozenset(
    {
        "answer",
        "correct_answer",
        "ground_truth",
        "input",
        "problem",
        "prompt",
        "question",
        "reference_answer",
        "response",
        "response_text",
    }
)


class ArtifactError(RuntimeError):
    """Base class for artifact contract violations."""


class ArtifactCorruptionError(ArtifactError):
    """A committed artifact cannot be decoded using its declared schema."""


class SensitiveJobPayloadError(ArtifactError):
    """A job-level monitor payload contains sample plaintext."""


def ensure_private_dir(path: str | Path) -> Path:
    """Create *path* and any missing parents with mode 0700.

    Existing ancestors are not chmod'ed, but the requested directory is always
    brought back to the private mode.  Symlinks are rejected at the privacy
    boundary.
    """

    target = Path(path)
    if target.is_symlink():
        raise ArtifactError(
            f"private artifact directory may not be a symlink: {target}"
        )

    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent

    target.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=True)
    if not target.is_dir():
        raise ArtifactError(f"artifact path is not a directory: {target}")
    for created in reversed(missing):
        os.chmod(created, PRIVATE_DIR_MODE)
    os.chmod(target, PRIVATE_DIR_MODE)
    return target


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("artifact write made no progress")
        view = view[written:]


def atomic_write_bytes(
    path: str | Path, payload: bytes | bytearray | memoryview
) -> Path:
    """Durably replace one private file with *payload*."""

    target = Path(path)
    ensure_private_dir(target.parent)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = os.open(temporary, flags, PRIVATE_FILE_MODE)
    try:
        _write_all(descriptor, bytes(payload))
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)

    try:
        os.replace(temporary, target)
        os.chmod(target, PRIVATE_FILE_MODE)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_text(path: str | Path, payload: str) -> Path:
    return atomic_write_bytes(path, payload.encode("utf-8"))


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return atomic_write_text(path, rendered + "\n")


def read_json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCorruptionError(f"unable to read JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactCorruptionError(f"JSON artifact is not an object: {path}")
    return dict(value)


def durable_append_jsonl(path: str | Path, record: Mapping[str, Any]) -> Path:
    """Append one complete JSON object and fsync before returning.

    A process-level advisory lock prevents records from interleaving when
    multiple workers share a job event stream.
    """

    target = Path(path)
    ensure_private_dir(target.parent)
    payload = (
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, PRIVATE_FILE_MODE)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    _fsync_directory(target.parent)
    return target


def read_complete_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read committed object records, ignoring malformed and partial lines."""

    target = Path(path)
    records: list[dict[str, Any]] = []
    try:
        handle = target.open("rb")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ArtifactCorruptionError(
            f"unable to read JSONL artifact: {target}"
        ) from exc

    with handle:
        for raw_line in handle:
            if not raw_line.endswith(b"\n"):
                continue
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(value, Mapping):
                continue
            records.append(dict(value))
    return records


def sha256_bytes(payload: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(payload)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_opaque_sample_id(
    encrypted_problem: str | bytes,
    *,
    namespace: str = "openai-browsecomp-v1",
) -> str:
    """Derive a stable ID without decrypting or embedding the question."""

    ciphertext = (
        encrypted_problem.encode("utf-8")
        if isinstance(encrypted_problem, str)
        else bytes(encrypted_problem)
    )
    digest = hashlib.sha256(namespace.encode("utf-8") + b"\0" + ciphertext).hexdigest()
    return f"bc_{digest[:32]}"


def validate_opaque_sample_id(sample_id: str) -> str:
    if OPAQUE_SAMPLE_ID_RE.fullmatch(sample_id) is None:
        raise ArtifactError(f"invalid opaque BrowseComp sample id: {sample_id!r}")
    return sample_id


@dataclass(frozen=True, slots=True)
class SampleArtifactPaths:
    root: Path

    def ensure(self) -> "SampleArtifactPaths":
        ensure_private_dir(self.root)
        ensure_private_dir(self.page_store_dir)
        return self

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def dispatch_path(self) -> Path:
        """Scheduler-owned state, deliberately separate from agent state."""

        return self.root / "dispatch.json"

    @property
    def claim_lock_path(self) -> Path:
        """Advisory lock held for the complete lifetime of one claimed sample."""

        return self.root / ".claim.lock"

    @property
    def luna_review_lock_path(self) -> Path:
        """Historical path serializing the independent answer review."""

        return self.root / ".luna-review.lock"

    @property
    def luna_review_path(self) -> Path:
        """Independent post-commit review; never part of the runner result."""

        return self.root / "luna-review.json"

    @property
    def result_path(self) -> Path:
        return self.root / "result.json"

    @property
    def trajectory_path(self) -> Path:
        return self.root / "trajectory.json"

    @property
    def tool_events_path(self) -> Path:
        return self.root / "tool-events.jsonl"

    @property
    def memory_path(self) -> Path:
        return self.root / "memory.jsonl"

    @property
    def page_store_dir(self) -> Path:
        return self.root / "page_store"

    @property
    def page_manifest_path(self) -> Path:
        return self.page_store_dir / "manifest.jsonl"


@dataclass(frozen=True, slots=True)
class JobArtifactPaths:
    root: Path

    def ensure(self) -> "JobArtifactPaths":
        ensure_private_dir(self.root)
        ensure_private_dir(self.samples_dir)
        return self

    def ensure_pool(self) -> "JobArtifactPaths":
        self.ensure()
        ensure_private_dir(self.workers_dir)
        return self

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def lock_path(self) -> Path:
        return self.root / "lock.json"

    @property
    def pool_path(self) -> Path:
        return self.root / "pool.json"

    @property
    def runner_lock_path(self) -> Path:
        return self.root / ".runner.lock"

    @property
    def pool_init_lock_path(self) -> Path:
        return self.root / ".pool-init.lock"

    @property
    def pool_aggregate_lock_path(self) -> Path:
        return self.root / ".pool-aggregate.lock"

    @property
    def luna_reviewer_lock_path(self) -> Path:
        """Historical path for one long-lived reviewer authority per job."""

        return self.root / ".luna-reviewer.lock"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def result_path(self) -> Path:
        return self.root / "result.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def samples_dir(self) -> Path:
        return self.root / "samples"

    @property
    def workers_dir(self) -> Path:
        return self.root / "workers"

    @property
    def agent_library_dir(self) -> Path:
        """Controller-owned skills/tools store shared by the whole pool."""

        return self.root / "agent_library"

    def sample(self, sample_id: str) -> SampleArtifactPaths:
        validate_opaque_sample_id(sample_id)
        return SampleArtifactPaths(self.samples_dir / sample_id)


def _assert_monitor_safe(value: Any, *, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _JOB_LEVEL_SENSITIVE_KEYS:
                raise SensitiveJobPayloadError(
                    f"job-level monitor payload may not contain {key!r} at {location}"
                )
            _assert_monitor_safe(nested, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_monitor_safe(nested, location=f"{location}[{index}]")


def write_job_state(paths: JobArtifactPaths, payload: Mapping[str, Any]) -> Path:
    _assert_monitor_safe(payload)
    return atomic_write_json(paths.state_path, dict(payload))


def write_job_result(paths: JobArtifactPaths, payload: Mapping[str, Any]) -> Path:
    _assert_monitor_safe(payload)
    return atomic_write_json(paths.result_path, dict(payload))


def append_job_event(paths: JobArtifactPaths, payload: Mapping[str, Any]) -> Path:
    _assert_monitor_safe(payload)
    return durable_append_jsonl(paths.events_path, payload)


def write_sample_state(paths: SampleArtifactPaths, payload: Mapping[str, Any]) -> Path:
    return atomic_write_json(paths.state_path, dict(payload))


def write_sample_result(paths: SampleArtifactPaths, payload: Mapping[str, Any]) -> Path:
    return atomic_write_json(paths.result_path, dict(payload))


def write_trajectory(paths: SampleArtifactPaths, payload: Mapping[str, Any]) -> Path:
    return atomic_write_json(paths.trajectory_path, dict(payload))


class ResumeAction(str, Enum):
    START = "start"
    RESUME = "resume"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class ResumeDecision:
    action: ResumeAction
    state: Mapping[str, Any] | None = None
    result: Mapping[str, Any] | None = None


def sample_resume_decision(paths: SampleArtifactPaths) -> ResumeDecision:
    """Classify a sample using an atomic result as the terminal commit marker."""

    if paths.result_path.exists():
        return ResumeDecision(
            action=ResumeAction.SKIP,
            result=read_json_object(paths.result_path),
        )
    if paths.state_path.exists():
        return ResumeDecision(
            action=ResumeAction.RESUME,
            state=read_json_object(paths.state_path),
        )
    return ResumeDecision(action=ResumeAction.START)
