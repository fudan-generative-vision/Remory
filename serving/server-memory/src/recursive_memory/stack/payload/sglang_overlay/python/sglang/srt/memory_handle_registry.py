"""Process-local GPU tensors addressed by server-minted memory handles.

The registry is intentionally local to one model-worker/TP-rank process.  All
TP ranks derive the same public descriptor from a shared server epoch and
operation identity while retaining their own device-local tensor.  Nothing in
this module enables the feature implicitly: a serving hook must explicitly
configure the global registry before publication or resolution can succeed.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch


GPU_MEMORY_HANDLE_CAPABILITY = "specforge_tp_gpu_memory_handle_v1"
GPU_MEMORY_HANDLE_SCHEMA = "specforge_tp_gpu_memory_handle_v1"
GPU_MEMORY_HANDLE_PUBLISH_SCHEMA = "specforge_tp_gpu_memory_handle_publish_v1"
GPU_MEMORY_HANDLE_RELEASE_SCHEMA = "specforge_tp_gpu_memory_handle_release_v1"
GPU_MEMORY_HANDLE_META_KEY = "specforge_gpu_memory_handle"
GPU_MEMORY_HANDLE_SERVER_INFO_SCHEMA = (
    "specforge_tp_gpu_memory_handle_server_info_v1"
)
GPU_MEMORY_HANDLE_SERVER_EPOCH_ENV = "SGLANG_SPECFORGE_MEMORY_SERVER_EPOCH"
GPU_MEMORY_IDENTITY_KIND = "specforge_transition_provenance_v1"

_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,256}\Z")
_HANDLE_FIELDS = frozenset(
    {
        "schema",
        "server_epoch",
        "handle_id",
        "identity_kind",
        "memory_identity",
        "request_digest",
        "descriptor_digest",
        "shape",
        "dtype",
        "nbytes",
    }
)


class MemoryHandleRegistryError(RuntimeError):
    """A handle operation violated the registry's fail-closed contract."""


@dataclass(frozen=True)
class ResolvedMemory:
    tensor: torch.Tensor
    memory_identity: str
    request_digest: str
    descriptor_digest: str
    handle_id: str


@dataclass
class _Entry:
    tensor: torch.Tensor
    descriptor: dict[str, Any]
    op_id: str
    generation: int
    expires_at: float


@dataclass(frozen=True)
class _ReleaseReplay:
    handle_id: str
    released_generation: Optional[int]
    response: dict[str, Any]
    expires_at: float


def _require_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise MemoryHandleRegistryError(
            f"{name} must be 1-256 characters from [A-Za-z0-9._:-]"
        )
    return value


def _require_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise MemoryHandleRegistryError(
            f"{name} must use canonical sha256:<64 lowercase hex> form"
        )
    return value


def _canonical_dtype(dtype: torch.dtype) -> str:
    value = str(dtype)
    return value[6:] if value.startswith("torch.") else value


def _digest_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class TPMemoryHandleRegistry:
    """Thread-safe tensor registry owned by one TP-rank process."""

    def __init__(
        self,
        *,
        server_epoch: str,
        tp_rank: int,
        ttl_seconds: float,
        max_entries: int,
        max_bytes: int,
        hook_abi_digest: str,
        release_replay_ttl_seconds: float = 60.0,
        release_replay_max_entries: int = 4096,
        require_cuda: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.server_epoch = _require_identifier(server_epoch, name="server_epoch")
        if isinstance(tp_rank, bool) or not isinstance(tp_rank, int) or tp_rank < 0:
            raise ValueError("tp_rank must be a nonnegative integer")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or float(ttl_seconds) <= 0
        ):
            raise ValueError("ttl_seconds must be positive")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(release_replay_ttl_seconds, bool)
            or not isinstance(release_replay_ttl_seconds, (int, float))
            or float(release_replay_ttl_seconds) <= 0
        ):
            raise ValueError("release_replay_ttl_seconds must be positive")
        if (
            isinstance(release_replay_max_entries, bool)
            or not isinstance(release_replay_max_entries, int)
            or release_replay_max_entries <= 0
        ):
            raise ValueError("release_replay_max_entries must be positive")
        if not isinstance(require_cuda, bool):
            raise TypeError("require_cuda must be boolean")

        self.tp_rank = tp_rank
        self.hook_abi_digest = _require_digest(
            hook_abi_digest, name="hook_abi_digest"
        )
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.release_replay_ttl_seconds = float(release_replay_ttl_seconds)
        self.release_replay_max_entries = release_replay_max_entries
        self.require_cuda = require_cuda
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._publish_ops: dict[str, str] = {}
        self._release_ops: OrderedDict[str, _ReleaseReplay] = OrderedDict()
        self._total_bytes = 0
        self._next_generation = 1

    def _descriptor_base(
        self,
        *,
        op_id: str,
        memory_identity: str,
        request_digest: str,
        tensor: torch.Tensor,
    ) -> dict[str, Any]:
        handle_seed = {
            "server_epoch": self.server_epoch,
            "op_id": op_id,
            "memory_identity": memory_identity,
            "request_digest": request_digest,
        }
        handle_id = "mh_" + _digest_json(handle_seed)[7:]
        return {
            "schema": GPU_MEMORY_HANDLE_SCHEMA,
            "server_epoch": self.server_epoch,
            "handle_id": handle_id,
            "identity_kind": GPU_MEMORY_IDENTITY_KIND,
            "memory_identity": memory_identity,
            "request_digest": request_digest,
            "shape": [int(value) for value in tensor.shape],
            "dtype": _canonical_dtype(tensor.dtype),
            "nbytes": int(tensor.numel() * tensor.element_size()),
        }

    @staticmethod
    def _with_descriptor_digest(base: Mapping[str, Any]) -> dict[str, Any]:
        descriptor = dict(base)
        descriptor["descriptor_digest"] = _digest_json(descriptor)
        return descriptor

    def _validate_descriptor(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != _HANDLE_FIELDS:
            raise MemoryHandleRegistryError(
                "memory handle must contain exactly the v1 descriptor fields"
            )
        descriptor = dict(value)
        if descriptor["schema"] != GPU_MEMORY_HANDLE_SCHEMA:
            raise MemoryHandleRegistryError("unsupported memory handle schema")
        if descriptor["server_epoch"] != self.server_epoch:
            raise MemoryHandleRegistryError("memory handle belongs to another server epoch")
        if descriptor["identity_kind"] != GPU_MEMORY_IDENTITY_KIND:
            raise MemoryHandleRegistryError("unsupported memory identity kind")
        _require_identifier(descriptor["handle_id"], name="handle_id")
        _require_digest(descriptor["memory_identity"], name="memory_identity")
        _require_digest(descriptor["request_digest"], name="request_digest")
        supplied_digest = _require_digest(
            descriptor["descriptor_digest"], name="descriptor_digest"
        )
        base = {key: value for key, value in descriptor.items() if key != "descriptor_digest"}
        if _digest_json(base) != supplied_digest:
            raise MemoryHandleRegistryError("memory handle descriptor digest mismatch")
        shape = descriptor["shape"]
        if (
            isinstance(shape, (str, bytes))
            or not isinstance(shape, Sequence)
            or len(shape) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        ):
            raise MemoryHandleRegistryError("memory handle shape must be two positive integers")
        if not isinstance(descriptor["dtype"], str) or not descriptor["dtype"]:
            raise MemoryHandleRegistryError("memory handle dtype must be nonempty")
        nbytes = descriptor["nbytes"]
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
            raise MemoryHandleRegistryError("memory handle nbytes must be positive")
        return descriptor

    def _cleanup_expired_locked(self, now: float) -> int:
        expired = [
            handle_id
            for handle_id, entry in self._entries.items()
            if entry.expires_at <= now
        ]
        for handle_id in expired:
            entry = self._entries.pop(handle_id)
            self._total_bytes -= int(entry.descriptor["nbytes"])
            if self._publish_ops.get(entry.op_id) == handle_id:
                del self._publish_ops[entry.op_id]
        self._cleanup_release_replays_locked(now)
        return len(expired)

    def _cleanup_release_replays_locked(self, now: float) -> int:
        removed = 0
        while self._release_ops:
            _op_id, replay = next(iter(self._release_ops.items()))
            if replay.expires_at > now:
                break
            self._release_ops.popitem(last=False)
            removed += 1
        return removed

    def _get_release_replay_locked(
        self, op_id: str, *, now: float
    ) -> Optional[_ReleaseReplay]:
        replay = self._release_ops.get(op_id)
        if replay is None:
            return None
        refreshed = _ReleaseReplay(
            handle_id=replay.handle_id,
            released_generation=replay.released_generation,
            response=replay.response,
            expires_at=now + self.release_replay_ttl_seconds,
        )
        self._release_ops[op_id] = refreshed
        self._release_ops.move_to_end(op_id)
        return refreshed

    def _store_release_replay_locked(
        self, op_id: str, replay: _ReleaseReplay
    ) -> None:
        self._release_ops[op_id] = replay
        self._release_ops.move_to_end(op_id)
        while len(self._release_ops) > self.release_replay_max_entries:
            self._release_ops.popitem(last=False)

    def cleanup_expired(self) -> int:
        with self._lock:
            return self._cleanup_expired_locked(self._monotonic())

    def rotate_epoch(self, server_epoch: str) -> dict[str, Any]:
        """Invalidate every handle and begin a fresh model-memory epoch."""

        server_epoch = _require_identifier(server_epoch, name="server_epoch")
        with self._lock:
            if server_epoch == self.server_epoch:
                raise MemoryHandleRegistryError(
                    "memory handle epoch rotation requires a new epoch"
                )
            self._entries.clear()
            self._publish_ops.clear()
            self._release_ops.clear()
            self._total_bytes = 0
            self._next_generation = 1
            self.server_epoch = server_epoch
            return {
                "server_epoch": self.server_epoch,
                "entries": 0,
                "bytes": 0,
                "release_replays": 0,
            }

    def publish(
        self,
        *,
        op_id: str,
        memory_identity: str,
        request_digest: str,
        tensor: torch.Tensor,
    ) -> dict[str, Any]:
        op_id = _require_identifier(op_id, name="op_id")
        memory_identity = _require_digest(memory_identity, name="memory_identity")
        request_digest = _require_digest(request_digest, name="request_digest")
        if not isinstance(tensor, torch.Tensor):
            raise MemoryHandleRegistryError("published memory must be a torch.Tensor")
        if tensor.ndim != 2 or tensor.numel() <= 0 or not tensor.is_floating_point():
            raise MemoryHandleRegistryError(
                "published memory must be a nonempty rank-two floating tensor"
            )
        if self.require_cuda and tensor.device.type != "cuda":
            raise MemoryHandleRegistryError("published memory must remain on CUDA")
        if not tensor.is_contiguous():
            raise MemoryHandleRegistryError("published memory tensor must be contiguous")
        descriptor = self._with_descriptor_digest(
            self._descriptor_base(
                op_id=op_id,
                memory_identity=memory_identity,
                request_digest=request_digest,
                tensor=tensor,
            )
        )
        handle_id = descriptor["handle_id"]

        with self._lock:
            now = self._monotonic()
            self._cleanup_expired_locked(now)
            previous_handle_id = self._publish_ops.get(op_id)
            if previous_handle_id is not None:
                previous = self._entries.get(previous_handle_id)
                if previous is None:  # pragma: no cover - lock invariant
                    raise MemoryHandleRegistryError("publish operation index is corrupt")
                if previous.descriptor != descriptor:
                    raise MemoryHandleRegistryError(
                        "op_id was already published with a different memory identity"
                    )
                return dict(previous.descriptor)
            if handle_id in self._entries:  # pragma: no cover - digest collision defense
                raise MemoryHandleRegistryError("memory handle collision")
            nbytes = int(descriptor["nbytes"])
            if len(self._entries) >= self.max_entries:
                raise MemoryHandleRegistryError("memory handle entry quota exceeded")
            if nbytes > self.max_bytes - self._total_bytes:
                raise MemoryHandleRegistryError("memory handle byte quota exceeded")
            # The registry owns an immutable allocation.  Holding a detached
            # view would let a publisher mutate storage after its stable
            # identity had been minted.
            owned_tensor = tensor.detach().clone(
                memory_format=torch.contiguous_format
            )
            entry = _Entry(
                tensor=owned_tensor,
                descriptor=descriptor,
                op_id=op_id,
                generation=self._next_generation,
                expires_at=now + self.ttl_seconds,
            )
            self._next_generation += 1
            self._entries[handle_id] = entry
            self._publish_ops[op_id] = handle_id
            self._total_bytes += nbytes
            return dict(descriptor)

    def resolve_and_renew(self, handle: Mapping[str, Any]) -> ResolvedMemory:
        """Resolve a tensor and renew its sliding lease under one lock."""

        descriptor = self._validate_descriptor(handle)
        with self._lock:
            now = self._monotonic()
            self._cleanup_expired_locked(now)
            entry = self._entries.get(descriptor["handle_id"])
            if entry is None:
                raise MemoryHandleRegistryError("memory handle is missing or expired")
            if entry.descriptor != descriptor:
                raise MemoryHandleRegistryError("memory handle metadata mismatch")
            tensor = entry.tensor
            if list(tensor.shape) != descriptor["shape"]:
                raise MemoryHandleRegistryError("registered tensor shape changed")
            if _canonical_dtype(tensor.dtype) != descriptor["dtype"]:
                raise MemoryHandleRegistryError("registered tensor dtype changed")
            if int(tensor.numel() * tensor.element_size()) != descriptor["nbytes"]:
                raise MemoryHandleRegistryError("registered tensor storage size changed")
            if self.require_cuda and tensor.device.type != "cuda":
                raise MemoryHandleRegistryError("registered tensor left CUDA")
            entry.expires_at = now + self.ttl_seconds
            return ResolvedMemory(
                tensor=tensor,
                memory_identity=descriptor["memory_identity"],
                request_digest=descriptor["request_digest"],
                descriptor_digest=descriptor["descriptor_digest"],
                handle_id=descriptor["handle_id"],
            )

    def resolve(self, handle: Mapping[str, Any]) -> ResolvedMemory:
        return self.resolve_and_renew(handle)

    def renew(self, handle: Mapping[str, Any]) -> dict[str, Any]:
        """Extend the external lease for a still-live physical generation."""

        descriptor = self._validate_descriptor(handle)
        with self._lock:
            now = self._monotonic()
            self._cleanup_expired_locked(now)
            entry = self._entries.get(descriptor["handle_id"])
            if entry is None:
                raise MemoryHandleRegistryError(
                    "memory handle is missing or expired during lease renewal"
                )
            if entry.descriptor != descriptor:
                raise MemoryHandleRegistryError("memory handle metadata mismatch")
            entry.expires_at = now + self.ttl_seconds
            return {
                "server_epoch": self.server_epoch,
                "handle_id": descriptor["handle_id"],
                "descriptor_digest": descriptor["descriptor_digest"],
                "lease_ttl_seconds": self.ttl_seconds,
            }

    def release(
        self, *, op_id: str, handle: Mapping[str, Any]
    ) -> dict[str, Any]:
        op_id = _require_identifier(op_id, name="op_id")
        descriptor = self._validate_descriptor(handle)
        handle_id = descriptor["handle_id"]
        with self._lock:
            now = self._monotonic()
            self._cleanup_expired_locked(now)
            response = self._preview_release_locked(
                op_id=op_id,
                descriptor=descriptor,
                now=now,
            )
            replay = self._get_release_replay_locked(op_id, now=now)
            entry = self._entries.get(handle_id)
            if replay is not None and entry is None:
                return dict(replay.response)
            if (
                replay is not None
                and entry is not None
                and replay.released_generation == entry.generation
            ):
                raise MemoryHandleRegistryError(
                    "release replay generation is unexpectedly still live"
                )
            if entry is not None:
                del self._entries[handle_id]
                self._total_bytes -= int(entry.descriptor["nbytes"])
                if self._publish_ops.get(entry.op_id) == handle_id:
                    del self._publish_ops[entry.op_id]
            self._store_release_replay_locked(
                op_id,
                _ReleaseReplay(
                    handle_id=handle_id,
                    released_generation=(
                        None if entry is None else entry.generation
                    ),
                    response=response,
                    expires_at=now + self.release_replay_ttl_seconds,
                ),
            )
            return dict(response)

    def _preview_release_locked(
        self,
        *,
        op_id: str,
        descriptor: Mapping[str, Any],
        now: float,
    ) -> dict[str, Any]:
        handle_id = descriptor["handle_id"]
        replay = self._get_release_replay_locked(op_id, now=now)
        if replay is not None:
            if replay.handle_id != handle_id:
                raise MemoryHandleRegistryError(
                    "release op_id was reused for a different handle"
                )
        entry = self._entries.get(handle_id)
        if replay is not None and entry is None:
            return dict(replay.response)
        if (
            replay is not None
            and entry is not None
            and replay.released_generation == entry.generation
        ):
            raise MemoryHandleRegistryError(
                "release replay generation is unexpectedly still live"
            )
        if entry is not None and entry.descriptor != descriptor:
            raise MemoryHandleRegistryError("memory handle metadata mismatch")
        return {
            "schema": GPU_MEMORY_HANDLE_RELEASE_SCHEMA,
            "op_id": op_id,
            "server_epoch": self.server_epoch,
            "released": entry is not None,
            "already_released": entry is None,
        }

    def preview_release(
        self, *, op_id: str, handle: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate a release without mutating live entries (TP two-phase use)."""

        op_id = _require_identifier(op_id, name="op_id")
        descriptor = self._validate_descriptor(handle)
        with self._lock:
            now = self._monotonic()
            self._cleanup_expired_locked(now)
            return self._preview_release_locked(
                op_id=op_id,
                descriptor=descriptor,
                now=now,
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._cleanup_expired_locked(self._monotonic())
            return {
                "server_epoch": self.server_epoch,
                "tp_rank": self.tp_rank,
                "hook_abi_digest": self.hook_abi_digest,
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "release_replay_ttl_seconds": self.release_replay_ttl_seconds,
                "release_replay_max_entries": self.release_replay_max_entries,
                "release_replays": len(self._release_ops),
                "entries": len(self._entries),
                "bytes": self._total_bytes,
            }


_GLOBAL_LOCK = threading.Lock()
_GLOBAL_REGISTRY: Optional[TPMemoryHandleRegistry] = None


def configure_global_memory_handle_registry(
    *,
    server_epoch: str,
    tp_rank: int,
    ttl_seconds: float,
    max_entries: int,
    max_bytes: int,
    hook_abi_digest: str,
    release_replay_ttl_seconds: float = 60.0,
    release_replay_max_entries: int = 4096,
    require_cuda: bool = True,
) -> TPMemoryHandleRegistry:
    global _GLOBAL_REGISTRY
    candidate = TPMemoryHandleRegistry(
        server_epoch=server_epoch,
        tp_rank=tp_rank,
        ttl_seconds=ttl_seconds,
        max_entries=max_entries,
        max_bytes=max_bytes,
        hook_abi_digest=hook_abi_digest,
        release_replay_ttl_seconds=release_replay_ttl_seconds,
        release_replay_max_entries=release_replay_max_entries,
        require_cuda=require_cuda,
    )
    with _GLOBAL_LOCK:
        if _GLOBAL_REGISTRY is None:
            _GLOBAL_REGISTRY = candidate
        else:
            existing = _GLOBAL_REGISTRY
            expected = (
                candidate.server_epoch,
                candidate.tp_rank,
                candidate.ttl_seconds,
                candidate.max_entries,
                candidate.max_bytes,
                candidate.hook_abi_digest,
                candidate.release_replay_ttl_seconds,
                candidate.release_replay_max_entries,
                candidate.require_cuda,
            )
            actual = (
                existing.server_epoch,
                existing.tp_rank,
                existing.ttl_seconds,
                existing.max_entries,
                existing.max_bytes,
                existing.hook_abi_digest,
                existing.release_replay_ttl_seconds,
                existing.release_replay_max_entries,
                existing.require_cuda,
            )
            if actual != expected:
                raise MemoryHandleRegistryError(
                    "global memory handle registry is already configured differently"
                )
        return _GLOBAL_REGISTRY


def get_global_memory_handle_registry() -> TPMemoryHandleRegistry:
    with _GLOBAL_LOCK:
        if _GLOBAL_REGISTRY is None:
            raise MemoryHandleRegistryError(
                "GPU memory handle registry is not configured in this process"
            )
        return _GLOBAL_REGISTRY


def global_memory_handle_registry_is_configured() -> bool:
    with _GLOBAL_LOCK:
        return _GLOBAL_REGISTRY is not None


def _reset_global_memory_handle_registry_for_tests() -> None:
    global _GLOBAL_REGISTRY
    with _GLOBAL_LOCK:
        _GLOBAL_REGISTRY = None
