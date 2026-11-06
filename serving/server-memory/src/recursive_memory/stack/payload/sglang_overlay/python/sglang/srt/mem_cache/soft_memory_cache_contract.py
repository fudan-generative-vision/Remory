"""Small contracts for safe continuous-memory prefix-cache reuse.

This module deliberately has no dependency on the SpecForge hook or on a
particular model implementation.  It owns the two pieces that must agree
before a continuous-memory request may use a token-keyed radix cache:

* a server-derived namespace that identifies the exact memory rows and their
  full-request token positions; and
* a request-aligned mapping from those full-request positions to SGLang's
  packed EXTEND coordinates after a prefix-cache hit.

Callers remain responsible for resolving handles and for scattering the
selected embedding rows.  In particular, a client-supplied digest must never
be passed as ``memory_identity``: it must come from the server registry, or be
computed from the inline rows with :func:`digest_inline_memory_rows`.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Sequence


SOFT_MEMORY_CACHE_NAMESPACE_ABI = "soft_memory_cache_namespace_v1"
SOFT_MEMORY_PREFIX_REUSE_BATCH_CAPABILITY = (
    "soft_memory_prefix_reuse_batch_v1"
)
SOFT_MEMORY_REUSE_CACHE_POLICY = "soft_reuse"


def _strict_nonnegative_positions(
    values: Sequence[int], *, name: str = "memory_positions"
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    if not values:
        raise ValueError(f"{name} must not be empty")
    result: list[int] = []
    previous = -1
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name}[{index}] must be an integer")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name}[{index}] must be nonnegative")
        if value <= previous:
            raise ValueError(f"{name} must be unique and strictly increasing")
        result.append(value)
        previous = value
    return tuple(result)


def digest_inline_memory_rows(
    rows: Sequence[Sequence[float]],
) -> str:
    """Return a server-computed identity for exact JSON-decoded memory rows.

    Values are normalized to finite IEEE-754 binary64 before hashing.  This is
    intentionally conservative: two wire representations that later round to
    the same model dtype may produce a harmless cache miss, but different wire
    values can never alias merely because a client chose the same cache key.
    """

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("memory_embeddings must be a rank-two sequence")
    if not rows:
        raise ValueError("memory_embeddings must not be empty")

    digest = hashlib.sha256()
    digest.update(b"specforge:inline-memory-rows:v1\0")
    digest.update(struct.pack(">Q", len(rows)))
    width: int | None = None
    for row_index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"memory_embeddings[{row_index}] must be a row")
        if width is None:
            width = len(row)
            if width <= 0:
                raise ValueError("memory embedding rows must not be empty")
            digest.update(struct.pack(">Q", width))
        elif len(row) != width:
            raise ValueError("memory embedding rows must have a uniform width")
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "memory_embeddings"
                    f"[{row_index}][{column_index}] must be numeric"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("memory_embeddings must contain finite values")
            digest.update(struct.pack(">d", value))
    return "inline-sha256:" + digest.hexdigest()


def build_soft_memory_cache_namespace(
    *, memory_identity: str, memory_positions: Sequence[int]
) -> str:
    """Bind one server-authoritative memory identity to its scatter positions."""

    if not isinstance(memory_identity, str) or not memory_identity:
        raise ValueError("memory_identity must be a nonempty server-derived string")
    if len(memory_identity.encode("utf-8")) > 4096:
        raise ValueError("memory_identity is unreasonably large")
    positions = _strict_nonnegative_positions(memory_positions)
    digest = hashlib.sha256()
    digest.update(SOFT_MEMORY_CACHE_NAMESPACE_ABI.encode("ascii"))
    digest.update(b"\0")
    identity_bytes = memory_identity.encode("utf-8")
    digest.update(struct.pack(">Q", len(identity_bytes)))
    digest.update(identity_bytes)
    digest.update(struct.pack(">Q", len(positions)))
    for position in positions:
        digest.update(struct.pack(">Q", position))
    return "specforge-soft-memory:" + digest.hexdigest()


@dataclass(frozen=True)
class PackedSoftMemoryRows:
    """Embedding rows that still need scatter for one request's EXTEND."""

    request_index: int
    embedding_row_indices: tuple[int, ...]
    packed_token_positions: tuple[int, ...]


@dataclass(frozen=True)
class _ServerSoftMemoryExtraKey:
    """Scheduler-internal radix namespace that cannot arrive over JSON.

    ``GenerateReqInput.extra_key`` only accepts JSON strings. Keeping the
    authoritative soft-memory namespace in a distinct hashable Python type
    prevents an ordinary discrete request from pre-populating the same radix
    branch even when a client knows a handle's public memory identity.
    """

    namespace: str
    lora_id: str


def build_server_soft_memory_extra_key(
    *, namespace: str, lora_id: str | None
) -> object:
    """Return the opaque cache key used only after scheduler admission."""

    if not isinstance(namespace, str) or not namespace.startswith(
        "specforge-soft-memory:"
    ):
        raise ValueError("invalid server-derived soft-memory namespace")
    if lora_id is not None and not isinstance(lora_id, str):
        raise TypeError("lora_id must be a string or None")
    return _ServerSoftMemoryExtraKey(
        namespace=namespace,
        lora_id=lora_id or "",
    )


def plan_packed_soft_memory_rows(
    *,
    memory_positions_per_request: Sequence[Sequence[int] | None],
    extend_prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    request_total_lens: Sequence[int] | None = None,
) -> tuple[PackedSoftMemoryRows, ...]:
    """Map full-request memory coordinates into a packed EXTEND batch.

    A position below ``extend_prefix_len`` is already represented by the
    cache entry selected under the same authoritative memory namespace and is
    therefore omitted.  Positions in the uncached suffix are translated by
    subtracting that request's prefix length and adding the packed offset of
    preceding requests.
    """

    if not (
        len(memory_positions_per_request)
        == len(extend_prefix_lens)
        == len(extend_seq_lens)
    ):
        raise ValueError("soft-memory batch fields must be request-aligned")

    if request_total_lens is not None and len(request_total_lens) != len(extend_seq_lens):
        raise ValueError("request totals must be request-aligned")
    packed_offset = 0
    result: list[PackedSoftMemoryRows] = []
    for request_index, (raw_positions, raw_prefix, raw_extend) in enumerate(
        zip(
            memory_positions_per_request,
            extend_prefix_lens,
            extend_seq_lens,
        )
    ):
        for name, value in (
            ("extend_prefix_lens", raw_prefix),
            ("extend_seq_lens", raw_extend),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name}[{request_index}] must be an integer")
            if value < 0:
                raise ValueError(f"{name}[{request_index}] must be nonnegative")
        prefix_len = int(raw_prefix)
        extend_len = int(raw_extend)
        chunk_end = prefix_len + extend_len
        full_length = chunk_end if request_total_lens is None else request_total_lens[request_index]
        if type(full_length) is not int or full_length < chunk_end:
            raise ValueError("request total must cover its EXTEND chunk")

        row_indices: list[int] = []
        packed_positions: list[int] = []
        if raw_positions is not None:
            positions = _strict_nonnegative_positions(
                raw_positions,
                name=f"memory_positions_per_request[{request_index}]",
            )
            for embedding_row_index, position in enumerate(positions):
                if position >= full_length:
                    raise ValueError(
                        "memory position lies outside its full request: "
                        f"request={request_index}, position={position}, "
                        f"length={full_length}"
                    )
                if position < prefix_len or position >= chunk_end:
                    continue
                row_indices.append(embedding_row_index)
                packed_positions.append(
                    packed_offset + position - prefix_len
                )

        result.append(
            PackedSoftMemoryRows(
                request_index=request_index,
                embedding_row_indices=tuple(row_indices),
                packed_token_positions=tuple(packed_positions),
            )
        )
        packed_offset += extend_len
    return tuple(result)


__all__ = [
    "PackedSoftMemoryRows",
    "SOFT_MEMORY_CACHE_NAMESPACE_ABI",
    "SOFT_MEMORY_PREFIX_REUSE_BATCH_CAPABILITY",
    "SOFT_MEMORY_REUSE_CACHE_POLICY",
    "build_soft_memory_cache_namespace",
    "build_server_soft_memory_extra_key",
    "digest_inline_memory_rows",
    "plan_packed_soft_memory_rows",
]
