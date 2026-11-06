"""Offline resolution for content-addressed Harbor task packages.

Harbor's package client normally asks the registry to resolve every task ref,
including refs that are already exact ``sha256:`` content addresses.  A
distributed evaluation therefore fails before creating a trial whenever the
registry is briefly unavailable, even if the requested package is cached.

Published Harbor task archives have a content-addressed storage path.  For an
exact digest we can derive that path without querying the registry, while the
normal Harbor download/extraction code remains responsible for materializing
the package.  Callers should verify the extracted directory against the pinned
digest before launching Harbor; :func:`verify_pinned_package_task` provides
that check.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


PINNED_PACKAGE_RESOLUTION_POLICY = (
    "harbor_exact_sha256_local_cache_and_direct_storage_v1"
)
_SHA256_REF_RE = re.compile(r"sha256:([0-9a-f]{64})")
_PACKAGE_SEGMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_OFFLINE_VERSION_ID_PREFIX = "offline-sha256:"


class PinnedTaskVerificationError(RuntimeError):
    """A pinned task could not be reconstructed or failed content validation."""


def _exact_digest(task_id: Any) -> str | None:
    ref = getattr(task_id, "ref", None)
    if not isinstance(ref, str):
        return None
    match = _SHA256_REF_RE.fullmatch(ref)
    if match is None:
        return None
    org = getattr(task_id, "org", None)
    name = getattr(task_id, "name", None)
    if (
        not isinstance(org, str)
        or _PACKAGE_SEGMENT_RE.fullmatch(org) is None
        or not isinstance(name, str)
        or _PACKAGE_SEGMENT_RE.fullmatch(name) is None
    ):
        raise PinnedTaskVerificationError(
            "content-addressed Harbor task has an unsafe package name"
        )
    return match.group(1)


def install_pinned_package_resolution(
    task_client_type: type[Any] | None = None,
    registry_db_type: type[Any] | None = None,
) -> None:
    """Make exact package refs bypass Harbor's mutable registry lookup.

    Non-digest refs retain Harbor's original behavior.  Installation is
    process-local and idempotent, so the pool controller and the Harbor CLI
    subprocess can both install it safely.
    """

    if task_client_type is None:
        from harbor.tasks.client import TaskClient

        task_client_type = TaskClient
    if registry_db_type is None:
        from harbor.db.client import RegistryDB

        registry_db_type = RegistryDB
    current = getattr(task_client_type, "_resolve_package_version")
    if getattr(current, "_r2eact_pinned_task_policy", None) != (
        PINNED_PACKAGE_RESOLUTION_POLICY
    ):

        async def resolve_package_version(self: Any, task_id: Any) -> Any:
            digest = _exact_digest(task_id)
            if digest is None:
                return await current(self, task_id)

            # This is the layout used by Harbor's task publisher.  The registry
            # row contains the same path, but contributes no information once the
            # immutable org/name/content digest triple is known.
            from harbor.tasks.client import _ResolvedPackage

            return _ResolvedPackage(
                id=f"{_OFFLINE_VERSION_ID_PREFIX}{digest}",
                archive_path=(
                    f"packages/{task_id.org}/{task_id.name}/{digest}/dist.tar.gz"
                ),
                content_hash=digest,
            )

        resolve_package_version._r2eact_pinned_task_policy = (  # type: ignore[attr-defined]
            PINNED_PACKAGE_RESOLUTION_POLICY
        )
        setattr(task_client_type, "_resolve_package_version", resolve_package_version)

    # Harbor records package download telemetry after a cache miss.  A direct
    # content-addressed resolution deliberately has no registry row UUID, so a
    # write for its synthetic id can only fail (and unnecessarily reintroduce
    # a registry dependency).  Preserve normal telemetry for all real ids.
    current_record = getattr(registry_db_type, "record_task_download")
    if getattr(current_record, "_r2eact_pinned_task_policy", None) != (
        PINNED_PACKAGE_RESOLUTION_POLICY
    ):

        async def record_task_download(self: Any, task_version_id: str) -> Any:
            if isinstance(task_version_id, str) and task_version_id.startswith(
                _OFFLINE_VERSION_ID_PREFIX
            ):
                return None
            return await current_record(self, task_version_id)

        record_task_download._r2eact_pinned_task_policy = (  # type: ignore[attr-defined]
            PINNED_PACKAGE_RESOLUTION_POLICY
        )
        setattr(registry_db_type, "record_task_download", record_task_download)


def verify_pinned_package_task(task_config: Any, task_path: Path) -> str:
    """Validate a materialized package directory against its exact task ref.

    Returns the unprefixed digest.  Validation intentionally uses Harbor's own
    package hash algorithm, so a merely well-shaped or stale cache directory
    cannot satisfy the immutable pool contract.
    """

    from harbor.models.task.task import Task
    from harbor.models.trial.config import TaskConfig
    from harbor.publisher.packager import Packager

    try:
        task = (
            task_config
            if isinstance(task_config, TaskConfig)
            else TaskConfig.model_validate(task_config)
        )
        task_id = task.get_task_id()
    except Exception as exc:
        raise PinnedTaskVerificationError(
            "could not reconstruct pinned Harbor task config"
        ) from exc
    digest = _exact_digest(task_id)
    if digest is None:
        raise PinnedTaskVerificationError(
            "Harbor package task is not pinned by an exact sha256 ref"
        )
    resolved = Path(task_path).expanduser().resolve()
    if not Task.is_valid_dir(resolved):
        raise PinnedTaskVerificationError(
            f"materialized Harbor task is invalid: {resolved}"
        )
    try:
        observed, _files = Packager.compute_content_hash(resolved)
    except Exception as exc:
        raise PinnedTaskVerificationError(
            f"could not hash materialized Harbor task: {resolved}"
        ) from exc
    if observed != digest:
        raise PinnedTaskVerificationError(
            "materialized Harbor task content hash mismatch: "
            f"expected {digest}, observed {observed}"
        )
    return digest


__all__ = [
    "PINNED_PACKAGE_RESOLUTION_POLICY",
    "PinnedTaskVerificationError",
    "install_pinned_package_resolution",
    "verify_pinned_package_task",
]
