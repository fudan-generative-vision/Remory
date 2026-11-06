"""Pinned JobBench dataset discovery and public-workspace projection.

Only ``task_folder`` is considered agent-visible.  Rubrics, task cards and
``files_required_to_search`` remain outside the copied workspace and are not
represented in :class:`JobBenchTask`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import (
    EXPECTED_TASK_COUNTS,
    OFFICIAL_DATASET_REPOSITORY,
    OFFICIAL_DATASET_REVISION,
    OFFICIAL_EVAL_COMMIT,
    SUPPORTED_SPLITS,
)


DATASET_MARKER_FILENAME = ".jobbench-source.json"
DATASET_MARKER_SCHEMA = "jobbench_pinned_dataset_source_v1"
TASK_INSTRUCTIONS_FILENAME = "TASK_INSTRUCTIONS.txt"
_TASK_NAME_RE = re.compile(r"task[0-9]+\Z")
_SAFE_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class DatasetError(RuntimeError):
    """The local JobBench source does not match the pinned public dataset."""


@dataclass(frozen=True, slots=True)
class JobBenchTask:
    """An agent-visible task projection.

    ``task_folder`` is the only source path retained.  In particular, there is
    no task-root field through which a runner could accidentally discover
    private benchmark metadata.
    """

    task_id: str
    profession: str
    task_name: str
    task_folder: Path
    relative_task_folder: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    dataset_root: Path
    split: str
    tasks: tuple[JobBenchTask, ...]
    snapshot_sha256: str


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(_canonical_json(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_dataset_marker(dataset_root: str | Path) -> Path:
    """Commit a revision marker after an exact-revision download.

    This function does not download anything.  The caller is responsible for
    materializing ``OFFICIAL_DATASET_REVISION`` before writing the marker.
    Existing conflicting markers are rejected rather than overwritten.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise DatasetError(f"dataset root does not exist: {root}")
    marker = root / DATASET_MARKER_FILENAME
    payload = {
        "schema": DATASET_MARKER_SCHEMA,
        "dataset_repository": OFFICIAL_DATASET_REPOSITORY,
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        "eval_commit": OFFICIAL_EVAL_COMMIT,
    }
    if marker.exists():
        try:
            current = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"invalid JobBench source marker: {marker}") from exc
        if current != payload:
            raise DatasetError("refusing to replace a conflicting dataset marker")
        return marker
    _atomic_write_json(marker, payload)
    return marker


def validate_dataset_marker(dataset_root: str | Path) -> Mapping[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    marker = root / DATASET_MARKER_FILENAME
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(
            f"pinned dataset marker is missing: {marker}; materialize revision "
            f"{OFFICIAL_DATASET_REVISION} and write the marker"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"invalid pinned dataset marker: {marker}") from exc
    expected = {
        "schema": DATASET_MARKER_SCHEMA,
        "dataset_repository": OFFICIAL_DATASET_REPOSITORY,
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        "eval_commit": OFFICIAL_EVAL_COMMIT,
    }
    if payload != expected:
        raise DatasetError("local dataset marker does not match the pinned release")
    return payload


def _regular_files(root: Path) -> Iterable[tuple[str, Path]]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise DatasetError(f"task_folder contains a symbolic link: {relative}")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise DatasetError(f"task_folder contains a non-regular file: {relative}")
        yield relative, path


def task_folder_digest(task_folder: str | Path) -> str:
    root = Path(task_folder).resolve()
    if not root.is_dir() or root.is_symlink():
        raise DatasetError(f"task_folder is not a real directory: {root}")
    digest = hashlib.sha256()
    for relative, path in _regular_files(root):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _discover_task(
    dataset_root: Path, split: str, profession_dir: Path, task_dir: Path
) -> JobBenchTask:
    profession = profession_dir.name
    task_name = task_dir.name
    if not _SAFE_COMPONENT_RE.fullmatch(profession):
        raise DatasetError(f"unsafe profession directory: {profession!r}")
    if not _TASK_NAME_RE.fullmatch(task_name):
        raise DatasetError(f"unsafe task directory: {task_name!r}")
    folder = task_dir / "task_folder"
    if folder.is_symlink() or not folder.is_dir():
        raise DatasetError(f"missing real task_folder for {profession}/{task_name}")
    instructions = folder / TASK_INSTRUCTIONS_FILENAME
    if instructions.is_symlink() or not instructions.is_file():
        raise DatasetError(f"missing {TASK_INSTRUCTIONS_FILENAME} for {profession}/{task_name}")
    if not instructions.read_bytes().strip():
        raise DatasetError(f"empty {TASK_INSTRUCTIONS_FILENAME} for {profession}/{task_name}")
    relative = folder.relative_to(dataset_root).as_posix()
    task_id = f"{profession}__{task_name}"
    return JobBenchTask(
        task_id=task_id,
        profession=profession,
        task_name=task_name,
        task_folder=folder.resolve(),
        relative_task_folder=relative,
        source_sha256=task_folder_digest(folder),
    )


def discover_tasks(
    dataset_root: str | Path,
    *,
    split: str = "main",
    verify_marker: bool = True,
    expected_count: int | None = None,
) -> DatasetSnapshot:
    """Discover the exact ``dataset/<split>/*/taskN/task_folder`` layout."""

    if split not in SUPPORTED_SPLITS:
        raise DatasetError(f"unsupported JobBench split: {split!r}")
    root = Path(dataset_root).expanduser().resolve()
    if verify_marker:
        validate_dataset_marker(root)
    split_root = root / split
    if split_root.is_symlink() or not split_root.is_dir():
        raise DatasetError(f"JobBench split directory does not exist: {split_root}")
    tasks: list[JobBenchTask] = []
    for profession_dir in sorted(split_root.iterdir(), key=lambda item: item.name):
        if profession_dir.name.startswith("."):
            continue
        if profession_dir.is_symlink() or not profession_dir.is_dir():
            raise DatasetError(f"unexpected split entry: {profession_dir}")
        for task_dir in sorted(profession_dir.iterdir(), key=lambda item: item.name):
            if task_dir.name.startswith("."):
                continue
            if task_dir.is_symlink() or not task_dir.is_dir():
                raise DatasetError(f"unexpected profession entry: {task_dir}")
            tasks.append(_discover_task(root, split, profession_dir, task_dir))
    required_count = EXPECTED_TASK_COUNTS[split] if expected_count is None else expected_count
    if len(tasks) != required_count:
        raise DatasetError(
            f"JobBench {split} must contain {required_count} tasks; found {len(tasks)}"
        )
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise DatasetError("JobBench task ids are not unique")
    manifest = [
        {
            "task_id": task.task_id,
            "relative_task_folder": task.relative_task_folder,
            "source_sha256": task.source_sha256,
        }
        for task in tasks
    ]
    snapshot_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DatasetSnapshot(
        dataset_root=root,
        split=split,
        tasks=tuple(tasks),
        snapshot_sha256=snapshot_sha256,
    )


def prepare_private_workspace(task: JobBenchTask, destination: str | Path) -> Path:
    """Copy only a task's public ``task_folder`` into a private workspace."""

    source = task.task_folder.resolve()
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise DatasetError(f"workspace destination already exists: {target}")
    if source == target or source in target.parents or target in source.parents:
        raise DatasetError("workspace and source paths must not contain one another")
    # Validate before copying so a symlink can never traverse into RUBRICS,
    # task_card, files_required_to_search, or any path outside task_folder.
    list(_regular_files(source))
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    shutil.copytree(source, target, symlinks=True)
    for path in [target, *target.rglob("*")]:
        if path.is_symlink():
            raise DatasetError("workspace unexpectedly contains a symbolic link")
        if path.is_dir():
            os.chmod(path, 0o700)
        elif path.is_file():
            source_mode = path.stat().st_mode
            os.chmod(path, 0o700 if source_mode & 0o111 else 0o600)
    if task_folder_digest(target) != task.source_sha256:
        raise DatasetError("copied task workspace failed its source digest check")
    return target
