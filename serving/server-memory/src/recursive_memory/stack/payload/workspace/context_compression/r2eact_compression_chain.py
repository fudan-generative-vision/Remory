"""Strict wire contracts for scheduler-serialized recursive compression.

The chain protocol is deliberately a control-plane protocol.  Every
transition is still submitted through SGLang's ordinary scheduler; the route
must never call a model forward (or a compressor hook) reentrantly.  GPU
memory is passed between transitions by an opaque, server-minted descriptor,
so intermediate tensors never cross the HTTP boundary.

Nothing in this module enables the protocol.  Existing harnesses continue to
use one ``/generate`` call per transition until they explicitly select the
versioned server-chain policy and attest the matching server capabilities.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


RECURSIVE_COMPRESSION_CHAIN_CAPABILITY = "specforge_recursive_compression_chain_v1"
RECURSIVE_COMPRESSION_CHAIN_ROUTE = "/specforge/memory/chain"
RECURSIVE_COMPRESSION_CHAIN_REQUEST_SCHEMA = (
    "specforge_recursive_compression_chain_request_v1"
)
RECURSIVE_COMPRESSION_CHAIN_RESPONSE_SCHEMA = (
    "specforge_recursive_compression_chain_response_v1"
)
RECURSIVE_COMPRESSION_CHAIN_EXECUTION_POLICY = (
    "scheduler_serial_no_reentrant_forward_v1"
)
RECURSIVE_COMPRESSION_CHAIN_COMMIT_POLICY = "atomic_final_handle_v1"
GPU_MEMORY_HANDLE_CAPABILITY = "specforge_tp_gpu_memory_handle_v1"

MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1 = "legacy_cpu_roundtrip_v1"
MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1 = "tp_gpu_handle_scheduler_chain_v1"
SOFT_MEMORY_PREFIX_REUSE_BATCH_CAPABILITY = "soft_memory_prefix_reuse_batch_v1"
SOFT_MEMORY_PREFIX_REUSE_WIRE_POLICY = "soft_reuse"
REQUEST_CACHE_BYPASS_CAPABILITY = "request_cache_bypass_v1"
MEMORY_TRANSPORT_TP_GPU_CHAIN_REQUIRED_CAPABILITIES = frozenset(
    {
        GPU_MEMORY_HANDLE_CAPABILITY,
        RECURSIVE_COMPRESSION_CHAIN_CAPABILITY,
        SOFT_MEMORY_PREFIX_REUSE_BATCH_CAPABILITY,
        REQUEST_CACHE_BYPASS_CAPABILITY,
    }
)

# Compatibility names for callers developed against the narrow chain draft.
RECURSIVE_COMPRESSION_POLICY_LEGACY_PER_TRANSITION_V1 = (
    MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1
)
RECURSIVE_COMPRESSION_POLICY_SERVER_SCHEDULER_CHAIN_V1 = (
    MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1
)
RECURSIVE_COMPRESSION_POLICIES = frozenset(
    {
        RECURSIVE_COMPRESSION_POLICY_LEGACY_PER_TRANSITION_V1,
        RECURSIVE_COMPRESSION_POLICY_SERVER_SCHEDULER_CHAIN_V1,
    }
)

GPU_MEMORY_HANDLE_SCHEMA = "specforge_tp_gpu_memory_handle_v1"
GPU_MEMORY_HANDLE_PUBLISH_SCHEMA = "specforge_tp_gpu_memory_handle_publish_v1"
GPU_MEMORY_HANDLE_RELEASE_SCHEMA = "specforge_tp_gpu_memory_handle_release_v1"
GPU_MEMORY_HANDLE_RELEASE_ROUTE = "/release_memory_handle"
GPU_MEMORY_HANDLE_META_KEY = "specforge_gpu_memory_handle"
GPU_MEMORY_IDENTITY_KIND = "specforge_transition_provenance_v1"
GPU_MEMORY_TRANSITION_REQUEST_SCHEMA = "specforge_memory_transition_request_v1"
GPU_MEMORY_IDENTITY_SCHEMA = "specforge_memory_identity_v1"
GPU_MEMORY_HANDLE_SERVER_INFO_SCHEMA = "specforge_tp_gpu_memory_handle_server_info_v1"
_GPU_MEMORY_HANDLE_SERVER_INFO_FIELDS = frozenset(
    {
        "schema",
        "server_epoch",
        "identity_kind",
        "request_digest_schema",
        "memory_identity_schema",
        "hook_abi_digest",
        "ttl_seconds",
        "max_entries_per_tp_rank",
        "max_bytes_per_tp_rank",
    }
)

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
_CHAIN_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "operation_id",
        "execution_policy",
        "commit_policy",
        "server_epoch",
        "hook_abi_digest",
        "expected_hidden_size",
        "compression_ratio",
        "custom_logit_processor",
        "cache_policy",
        "parent_memory_handle",
        "transitions",
    }
)
_TRANSITION_FIELDS = frozenset(
    {
        "transition_index",
        "source_input_ids",
        "evidence_spans",
        "source_token_lengths",
        "expected_slot_count",
        "recursive_source_contract",
    }
)
_PUBLISH_FIELDS = frozenset({"schema", "op_id", "memory_identity", "request_digest"})
_CHAIN_RESPONSE_FIELDS = frozenset(
    {
        "schema",
        "operation_id",
        "execution_policy",
        "commit_policy",
        "final_memory_handle",
        "transition_receipts",
    }
)
_TRANSITION_RECEIPT_FIELDS = frozenset(
    {
        "transition_index",
        "memory_identity",
        "request_digest",
        "descriptor_digest",
        "shape",
    }
)


class RecursiveCompressionChainContractError(RuntimeError):
    """The recursive compression chain violated its versioned wire ABI."""


@dataclass(frozen=True)
class MemoryTransportAttestation:
    requested: str
    effective: str
    required_capabilities: tuple[str, ...]
    observed_capabilities: tuple[str, ...]
    server_epoch: str | None
    hook_abi_digest: str | None
    max_running_requests: int | None

    def to_json(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "effective": self.effective,
            "required_capabilities": list(self.required_capabilities),
            "observed_capabilities": list(self.observed_capabilities),
            "server_epoch": self.server_epoch,
            "hook_abi_digest": self.hook_abi_digest,
            "max_running_requests": self.max_running_requests,
        }


def _digest_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise RecursiveCompressionChainContractError(
            f"{name} must be 1-256 characters from [A-Za-z0-9._:-]"
        )
    return value


def _require_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise RecursiveCompressionChainContractError(
            f"{name} must use canonical sha256:<64 lowercase hex> form"
        )
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecursiveCompressionChainContractError(
            f"{name} must be a positive integer"
        )
    return int(value)


def attest_memory_transport_server_info(
    server_info: Mapping[str, Any],
    *,
    requested_policy: str,
) -> MemoryTransportAttestation:
    """Validate the one coherent transport handshake used by new harnesses."""

    if requested_policy not in RECURSIVE_COMPRESSION_POLICIES:
        raise ValueError(f"unsupported memory transport policy: {requested_policy!r}")
    if requested_policy == MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1:
        return MemoryTransportAttestation(
            requested=requested_policy,
            effective=requested_policy,
            required_capabilities=(),
            observed_capabilities=(),
            server_epoch=None,
            hook_abi_digest=None,
            max_running_requests=None,
        )
    if not isinstance(server_info, Mapping):
        raise RecursiveCompressionChainContractError("server_info must be a mapping")
    capabilities_raw = server_info.get("request_capabilities")
    if (
        not isinstance(capabilities_raw, list)
        or any(not isinstance(item, str) for item in capabilities_raw)
        or len(set(capabilities_raw)) != len(capabilities_raw)
    ):
        raise RecursiveCompressionChainContractError(
            "server_info request_capabilities must be a unique string list"
        )
    capabilities = frozenset(capabilities_raw)
    missing = MEMORY_TRANSPORT_TP_GPU_CHAIN_REQUIRED_CAPABILITIES - capabilities
    if missing:
        raise RecursiveCompressionChainContractError(
            "memory transport server is missing required capabilities: "
            + ", ".join(sorted(missing))
        )
    max_running = server_info.get("max_running_requests")
    if (
        isinstance(max_running, bool)
        or not isinstance(max_running, int)
        or max_running < 2
    ):
        raise RecursiveCompressionChainContractError(
            "GPU-chain memory transport requires max_running_requests>=2"
        )
    handle_info = server_info.get("specforge_gpu_memory_handle")
    if (
        not isinstance(handle_info, Mapping)
        or set(handle_info) != _GPU_MEMORY_HANDLE_SERVER_INFO_FIELDS
    ):
        raise RecursiveCompressionChainContractError(
            "server_info has no exact v1 GPU-memory-handle attestation"
        )
    if handle_info["schema"] != GPU_MEMORY_HANDLE_SERVER_INFO_SCHEMA:
        raise RecursiveCompressionChainContractError(
            "unsupported GPU-memory-handle server-info schema"
        )
    if handle_info["identity_kind"] != GPU_MEMORY_IDENTITY_KIND:
        raise RecursiveCompressionChainContractError(
            "unsupported GPU-memory identity kind"
        )
    if handle_info["request_digest_schema"] != GPU_MEMORY_TRANSITION_REQUEST_SCHEMA:
        raise RecursiveCompressionChainContractError(
            "GPU-memory request digest schema mismatch"
        )
    if handle_info["memory_identity_schema"] != GPU_MEMORY_IDENTITY_SCHEMA:
        raise RecursiveCompressionChainContractError(
            "GPU-memory identity schema mismatch"
        )
    server_epoch = _require_identifier(handle_info["server_epoch"], name="server_epoch")
    hook_abi_digest = _require_digest(
        handle_info["hook_abi_digest"], name="hook_abi_digest"
    )
    ttl = handle_info["ttl_seconds"]
    if (
        isinstance(ttl, bool)
        or not isinstance(ttl, (int, float))
        or not math.isfinite(float(ttl))
        or float(ttl) <= 0
    ):
        raise RecursiveCompressionChainContractError(
            "GPU-memory handle ttl_seconds must be positive"
        )
    _positive_int(
        handle_info["max_entries_per_tp_rank"],
        name="max_entries_per_tp_rank",
    )
    _positive_int(
        handle_info["max_bytes_per_tp_rank"],
        name="max_bytes_per_tp_rank",
    )
    return MemoryTransportAttestation(
        requested=requested_policy,
        effective=requested_policy,
        required_capabilities=tuple(
            sorted(MEMORY_TRANSPORT_TP_GPU_CHAIN_REQUIRED_CAPABILITIES)
        ),
        observed_capabilities=tuple(sorted(capabilities)),
        server_epoch=server_epoch,
        hook_abi_digest=hook_abi_digest,
        max_running_requests=max_running,
    )


@dataclass(frozen=True)
class TPGPUMemoryHandle:
    """CPU-side validation view of an opaque server GPU-memory handle."""

    descriptor: Mapping[str, Any]

    @classmethod
    def parse(
        cls,
        value: Any,
        *,
        expected_server_epoch: str | None = None,
        expected_slots: int | None = None,
        expected_hidden_size: int | None = None,
        expected_memory_identity: str | None = None,
        expected_request_digest: str | None = None,
    ) -> TPGPUMemoryHandle:
        if not isinstance(value, Mapping) or set(value) != _HANDLE_FIELDS:
            raise RecursiveCompressionChainContractError(
                "memory handle must contain exactly the v1 descriptor fields"
            )
        descriptor = dict(value)
        if descriptor["schema"] != GPU_MEMORY_HANDLE_SCHEMA:
            raise RecursiveCompressionChainContractError(
                "unsupported memory handle schema"
            )
        server_epoch = _require_identifier(
            descriptor["server_epoch"], name="server_epoch"
        )
        if expected_server_epoch is not None and server_epoch != expected_server_epoch:
            raise RecursiveCompressionChainContractError(
                "memory handle belongs to another server epoch"
            )
        _require_identifier(descriptor["handle_id"], name="handle_id")
        if descriptor["identity_kind"] != GPU_MEMORY_IDENTITY_KIND:
            raise RecursiveCompressionChainContractError(
                "unsupported memory identity kind"
            )
        memory_identity = _require_digest(
            descriptor["memory_identity"], name="memory_identity"
        )
        request_digest = _require_digest(
            descriptor["request_digest"], name="request_digest"
        )
        supplied_digest = _require_digest(
            descriptor["descriptor_digest"], name="descriptor_digest"
        )
        descriptor_base = {
            key: item for key, item in descriptor.items() if key != "descriptor_digest"
        }
        if _digest_json(descriptor_base) != supplied_digest:
            raise RecursiveCompressionChainContractError(
                "memory handle descriptor digest mismatch"
            )
        shape = descriptor["shape"]
        if (
            isinstance(shape, (str, bytes))
            or not isinstance(shape, Sequence)
            or len(shape) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape
            )
        ):
            raise RecursiveCompressionChainContractError(
                "memory handle shape must be two positive integers"
            )
        normalized_shape = [int(shape[0]), int(shape[1])]
        if expected_slots is not None and normalized_shape[0] != _positive_int(
            expected_slots, name="expected_slots"
        ):
            raise RecursiveCompressionChainContractError(
                "memory handle has the wrong slot count"
            )
        if expected_hidden_size is not None and normalized_shape[1] != _positive_int(
            expected_hidden_size, name="expected_hidden_size"
        ):
            raise RecursiveCompressionChainContractError(
                "memory handle has the wrong hidden width"
            )
        if not isinstance(descriptor["dtype"], str) or not descriptor["dtype"]:
            raise RecursiveCompressionChainContractError(
                "memory handle dtype must be nonempty"
            )
        _positive_int(descriptor["nbytes"], name="memory handle nbytes")
        if (
            expected_memory_identity is not None
            and memory_identity != expected_memory_identity
        ):
            raise RecursiveCompressionChainContractError(
                "memory handle has the wrong memory identity"
            )
        if (
            expected_request_digest is not None
            and request_digest != expected_request_digest
        ):
            raise RecursiveCompressionChainContractError(
                "memory handle has the wrong request digest"
            )
        descriptor["shape"] = normalized_shape
        return cls(descriptor=descriptor)

    @property
    def server_epoch(self) -> str:
        return str(self.descriptor["server_epoch"])

    @property
    def handle_id(self) -> str:
        return str(self.descriptor["handle_id"])

    @property
    def memory_identity(self) -> str:
        return str(self.descriptor["memory_identity"])

    @property
    def request_digest(self) -> str:
        return str(self.descriptor["request_digest"])

    @property
    def shape(self) -> tuple[int, int]:
        value = self.descriptor["shape"]
        return int(value[0]), int(value[1])

    def to_json(self) -> dict[str, Any]:
        value = dict(self.descriptor)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True)
class RecursiveCompressionTransition:
    transition_index: int
    source_input_ids: tuple[int, ...]
    evidence_span: tuple[int, int]
    source_token_length: int
    expected_slot_count: int
    recursive_source_contract: str | None

    @classmethod
    def parse(
        cls,
        value: Any,
        *,
        compression_ratio: int,
        previous_memory_slots: int,
        expected_transition_index: int,
    ) -> RecursiveCompressionTransition:
        if not isinstance(value, Mapping) or set(value) != _TRANSITION_FIELDS:
            raise RecursiveCompressionChainContractError(
                "chain transition must contain exactly the v1 fields"
            )
        index = value["transition_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index != expected_transition_index
        ):
            raise RecursiveCompressionChainContractError(
                "chain transition indices must be consecutive from zero"
            )
        source = value["source_input_ids"]
        if (
            isinstance(source, (str, bytes))
            or not isinstance(source, Sequence)
            or not source
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in source
            )
        ):
            raise RecursiveCompressionChainContractError(
                "source_input_ids must be nonempty nonnegative integers"
            )
        spans = value["evidence_spans"]
        if (
            isinstance(spans, (str, bytes))
            or not isinstance(spans, Sequence)
            or len(spans) != 1
            or isinstance(spans[0], (str, bytes))
            or not isinstance(spans[0], Sequence)
            or len(spans[0]) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in spans[0]
            )
        ):
            raise RecursiveCompressionChainContractError(
                "recursive chain transition requires one integer evidence span"
            )
        start, end = int(spans[0][0]), int(spans[0][1])
        if start <= 0 or end != len(source) or end <= start:
            raise RecursiveCompressionChainContractError(
                "evidence span must cover the nonempty suffix after a nonempty prefix"
            )
        lengths = value["source_token_lengths"]
        if (
            isinstance(lengths, (str, bytes))
            or not isinstance(lengths, Sequence)
            or len(lengths) != 1
        ):
            raise RecursiveCompressionChainContractError(
                "source_token_lengths must contain one item"
            )
        source_length = _positive_int(lengths[0], name="source_token_lengths[0]")
        if source_length != end - start:
            raise RecursiveCompressionChainContractError(
                "source token length must equal the evidence span width"
            )
        if previous_memory_slots < 0 or previous_memory_slots >= source_length:
            raise RecursiveCompressionChainContractError(
                "every transition must contain a nonempty new discrete suffix"
            )
        expected_slots = _positive_int(
            value["expected_slot_count"], name="expected_slot_count"
        )
        derived_slots = (source_length + compression_ratio - 1) // compression_ratio
        if expected_slots != derived_slots:
            raise RecursiveCompressionChainContractError(
                "expected_slot_count disagrees with the compression ratio"
            )
        contract = value["recursive_source_contract"]
        if contract is not None and (not isinstance(contract, str) or not contract):
            raise RecursiveCompressionChainContractError(
                "recursive_source_contract must be a nonempty string or null"
            )
        return cls(
            transition_index=int(index),
            source_input_ids=tuple(int(item) for item in source),
            evidence_span=(start, end),
            source_token_length=source_length,
            expected_slot_count=expected_slots,
            recursive_source_contract=contract,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "transition_index": self.transition_index,
            "source_input_ids": list(self.source_input_ids),
            "evidence_spans": [list(self.evidence_span)],
            "source_token_lengths": [self.source_token_length],
            "expected_slot_count": self.expected_slot_count,
            "recursive_source_contract": self.recursive_source_contract,
        }


@dataclass(frozen=True)
class RecursiveCompressionChainRequest:
    operation_id: str
    server_epoch: str
    hook_abi_digest: str
    expected_hidden_size: int
    compression_ratio: int
    custom_logit_processor: str
    cache_policy: str
    parent_memory_handle: TPGPUMemoryHandle | None
    transitions: tuple[RecursiveCompressionTransition, ...]

    @classmethod
    def parse(cls, value: Any) -> RecursiveCompressionChainRequest:
        if not isinstance(value, Mapping) or set(value) != _CHAIN_REQUEST_FIELDS:
            raise RecursiveCompressionChainContractError(
                "chain request must contain exactly the v1 fields"
            )
        if value["schema"] != RECURSIVE_COMPRESSION_CHAIN_REQUEST_SCHEMA:
            raise RecursiveCompressionChainContractError(
                "unsupported recursive compression chain request schema"
            )
        if value["execution_policy"] != RECURSIVE_COMPRESSION_CHAIN_EXECUTION_POLICY:
            raise RecursiveCompressionChainContractError(
                "unsupported recursive compression execution policy"
            )
        if value["commit_policy"] != RECURSIVE_COMPRESSION_CHAIN_COMMIT_POLICY:
            raise RecursiveCompressionChainContractError(
                "unsupported recursive compression commit policy"
            )
        operation_id = _require_identifier(value["operation_id"], name="operation_id")
        server_epoch = _require_identifier(value["server_epoch"], name="server_epoch")
        hook_abi_digest = _require_digest(
            value["hook_abi_digest"], name="hook_abi_digest"
        )
        hidden_size = _positive_int(
            value["expected_hidden_size"], name="expected_hidden_size"
        )
        ratio = _positive_int(value["compression_ratio"], name="compression_ratio")
        processor = value["custom_logit_processor"]
        if not isinstance(processor, str) or not processor:
            raise RecursiveCompressionChainContractError(
                "custom_logit_processor must be a nonempty string"
            )
        cache_policy = value["cache_policy"]
        if cache_policy != "bypass":
            raise RecursiveCompressionChainContractError(
                "v1 chain requires request-local cache_policy='bypass'"
            )
        parent_raw = value["parent_memory_handle"]
        parent = (
            None
            if parent_raw is None
            else TPGPUMemoryHandle.parse(
                parent_raw,
                expected_server_epoch=server_epoch,
                expected_hidden_size=hidden_size,
            )
        )
        transition_values = value["transitions"]
        if (
            isinstance(transition_values, (str, bytes))
            or not isinstance(transition_values, Sequence)
            or len(transition_values) < 2
        ):
            raise RecursiveCompressionChainContractError(
                "server chain v1 requires at least two transitions"
            )
        previous_slots = 0 if parent is None else parent.shape[0]
        transitions: list[RecursiveCompressionTransition] = []
        for index, item in enumerate(transition_values):
            transition = RecursiveCompressionTransition.parse(
                item,
                compression_ratio=ratio,
                previous_memory_slots=previous_slots,
                expected_transition_index=index,
            )
            transitions.append(transition)
            previous_slots = transition.expected_slot_count
        return cls(
            operation_id=operation_id,
            server_epoch=server_epoch,
            hook_abi_digest=hook_abi_digest,
            expected_hidden_size=hidden_size,
            compression_ratio=ratio,
            custom_logit_processor=processor,
            cache_policy=cache_policy,
            parent_memory_handle=parent,
            transitions=tuple(transitions),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": RECURSIVE_COMPRESSION_CHAIN_REQUEST_SCHEMA,
            "operation_id": self.operation_id,
            "execution_policy": RECURSIVE_COMPRESSION_CHAIN_EXECUTION_POLICY,
            "commit_policy": RECURSIVE_COMPRESSION_CHAIN_COMMIT_POLICY,
            "server_epoch": self.server_epoch,
            "hook_abi_digest": self.hook_abi_digest,
            "expected_hidden_size": self.expected_hidden_size,
            "compression_ratio": self.compression_ratio,
            "custom_logit_processor": self.custom_logit_processor,
            "cache_policy": self.cache_policy,
            "parent_memory_handle": (
                None
                if self.parent_memory_handle is None
                else self.parent_memory_handle.to_json()
            ),
            "transitions": [item.to_json() for item in self.transitions],
        }


@dataclass(frozen=True)
class RecursiveCompressionChainResult:
    operation_id: str
    final_memory_handle: TPGPUMemoryHandle
    transition_receipts: tuple[Mapping[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": RECURSIVE_COMPRESSION_CHAIN_RESPONSE_SCHEMA,
            "operation_id": self.operation_id,
            "execution_policy": RECURSIVE_COMPRESSION_CHAIN_EXECUTION_POLICY,
            "commit_policy": RECURSIVE_COMPRESSION_CHAIN_COMMIT_POLICY,
            "final_memory_handle": self.final_memory_handle.to_json(),
            "transition_receipts": [dict(item) for item in self.transition_receipts],
        }


def derive_recursive_compression_transition_claim(
    chain: RecursiveCompressionChainRequest,
    transition: RecursiveCompressionTransition,
    *,
    previous_memory: Mapping[str, Any] | None,
    previous_memory_slots: int,
) -> dict[str, str]:
    """Derive the hook-authoritative claim after the parent is known.

    This mirrors :func:`compute_memory_handle_request_digest` and
    :func:`compute_memory_handle_identity` in ``sglang_compressor_hook``.  A
    fixed-vector test locks the two implementations together so the endpoint
    can remain a small, dependency-light SGLang integration.
    """

    start = transition.evidence_span[0]
    positions = (
        []
        if previous_memory is None
        else list(range(start, start + previous_memory_slots))
    )
    previous_provenance = None
    if previous_memory is not None:
        previous_provenance = {
            "server_epoch": chain.server_epoch,
            "identity_kind": GPU_MEMORY_IDENTITY_KIND,
            "memory_identity": _require_digest(
                previous_memory.get("memory_identity"),
                name="previous memory identity",
            ),
            "descriptor_digest": _require_digest(
                previous_memory.get("descriptor_digest"),
                name="previous descriptor digest",
            ),
        }
    request_digest = _digest_json(
        {
            "schema": GPU_MEMORY_TRANSITION_REQUEST_SCHEMA,
            "custom_param_version": 1,
            "mode": "encode",
            "input_ids": list(transition.source_input_ids),
            "evidence_spans": [list(transition.evidence_span)],
            "source_token_lengths": [transition.source_token_length],
            "memory_positions": positions,
            "previous_memory": previous_provenance,
            "recursive_source_contract": transition.recursive_source_contract,
        }
    )
    memory_identity = _digest_json(
        {
            "schema": GPU_MEMORY_IDENTITY_SCHEMA,
            "identity_kind": GPU_MEMORY_IDENTITY_KIND,
            "server_epoch": chain.server_epoch,
            "request_digest": request_digest,
            "hook_abi_digest": chain.hook_abi_digest,
        }
    )
    op_digest = _digest_json(
        {
            "schema": "specforge_recursive_compression_chain_child_op_v1",
            "operation_id": chain.operation_id,
            "transition_index": transition.transition_index,
        }
    )
    return {
        "schema": GPU_MEMORY_HANDLE_PUBLISH_SCHEMA,
        "op_id": "chain_" + op_digest[7:],
        "memory_identity": memory_identity,
        "request_digest": request_digest,
    }


def parse_recursive_compression_chain_response(
    value: Any,
    *,
    request: RecursiveCompressionChainRequest,
) -> RecursiveCompressionChainResult:
    if not isinstance(value, Mapping) or set(value) != _CHAIN_RESPONSE_FIELDS:
        raise RecursiveCompressionChainContractError(
            "chain response must contain exactly the v1 fields"
        )
    if value["schema"] != RECURSIVE_COMPRESSION_CHAIN_RESPONSE_SCHEMA:
        raise RecursiveCompressionChainContractError(
            "unsupported recursive compression chain response schema"
        )
    if value["operation_id"] != request.operation_id:
        raise RecursiveCompressionChainContractError(
            "chain response operation_id mismatch"
        )
    if value["execution_policy"] != RECURSIVE_COMPRESSION_CHAIN_EXECUTION_POLICY:
        raise RecursiveCompressionChainContractError(
            "chain response execution policy mismatch"
        )
    if value["commit_policy"] != RECURSIVE_COMPRESSION_CHAIN_COMMIT_POLICY:
        raise RecursiveCompressionChainContractError(
            "chain response commit policy mismatch"
        )
    receipts_raw = value["transition_receipts"]
    if (
        isinstance(receipts_raw, (str, bytes))
        or not isinstance(receipts_raw, Sequence)
        or len(receipts_raw) != len(request.transitions)
    ):
        raise RecursiveCompressionChainContractError(
            "chain response has the wrong transition receipt count"
        )
    receipts: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = (
        None
        if request.parent_memory_handle is None
        else request.parent_memory_handle.to_json()
    )
    previous_slots = (
        0
        if request.parent_memory_handle is None
        else request.parent_memory_handle.shape[0]
    )
    for transition, receipt_raw in zip(request.transitions, receipts_raw):
        if (
            not isinstance(receipt_raw, Mapping)
            or set(receipt_raw) != _TRANSITION_RECEIPT_FIELDS
        ):
            raise RecursiveCompressionChainContractError(
                "transition receipt must contain exactly the v1 fields"
            )
        if receipt_raw["transition_index"] != transition.transition_index:
            raise RecursiveCompressionChainContractError(
                "transition receipt index mismatch"
            )
        claim = derive_recursive_compression_transition_claim(
            request,
            transition,
            previous_memory=previous,
            previous_memory_slots=previous_slots,
        )
        if receipt_raw["memory_identity"] != claim["memory_identity"]:
            raise RecursiveCompressionChainContractError(
                "transition receipt memory identity mismatch"
            )
        if receipt_raw["request_digest"] != claim["request_digest"]:
            raise RecursiveCompressionChainContractError(
                "transition receipt request digest mismatch"
            )
        if receipt_raw["shape"] != [
            transition.expected_slot_count,
            request.expected_hidden_size,
        ]:
            raise RecursiveCompressionChainContractError(
                "transition receipt shape mismatch"
            )
        normalized_receipt = dict(receipt_raw)
        _require_digest(
            normalized_receipt["descriptor_digest"],
            name="transition receipt descriptor_digest",
        )
        receipts.append(normalized_receipt)
        previous = normalized_receipt
        previous_slots = transition.expected_slot_count
    final_transition = request.transitions[-1]
    final_receipt = receipts[-1]
    final_handle = TPGPUMemoryHandle.parse(
        value["final_memory_handle"],
        expected_server_epoch=(
            None
            if request.parent_memory_handle is None
            else request.parent_memory_handle.server_epoch
        ),
        expected_slots=final_transition.expected_slot_count,
        expected_hidden_size=request.expected_hidden_size,
        expected_memory_identity=final_receipt["memory_identity"],
        expected_request_digest=final_receipt["request_digest"],
    )
    if (
        final_handle.descriptor["descriptor_digest"]
        != final_receipt["descriptor_digest"]
    ):
        raise RecursiveCompressionChainContractError(
            "final handle descriptor digest disagrees with its transition receipt"
        )
    return RecursiveCompressionChainResult(
        operation_id=request.operation_id,
        final_memory_handle=final_handle,
        transition_receipts=tuple(receipts),
    )


SchedulerSubmit = Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]]
HandleRelease = Callable[[Mapping[str, Any]], Awaitable[Any] | Any]
SchedulerAbort = Callable[[str], Awaitable[Any] | Any]
ChainExecution = Callable[[], Awaitable[Mapping[str, Any]]]


@dataclass
class _RecursiveCompressionChainFlight:
    payload_digest: str
    task: asyncio.Task[dict[str, Any]]
    waiters: int = 0
    completed_at: float | None = None
    orphan_cancel_task: asyncio.Task[None] | None = None
    orphan_cancelled: bool = False


class RecursiveCompressionChainSingleFlight:
    """Operation-scoped singleflight for the HTTP chain endpoint.

    A stable child ``op_id`` makes each publish idempotent, but it does not by
    itself make two whole-chain executions safe: either execution could
    release an intermediate while the other is about to consume it.  This
    registry admits one execution for an ``operation_id`` and lets exact
    payload retries await the same task.  A cancelled HTTP waiter is shielded
    from the shared task, so disconnecting one client cannot abort work still
    observed by another retry.

    Successful results are retained briefly for lost-response retries.  The
    HTTP release route invalidates the retained result when the final handle
    is explicitly released; the time bound must not exceed the server handle
    TTL.  A zero-waiter execution gets a bounded reconnect grace period and is
    then cancelled; every execution also has an independent server deadline.
    Failed executions are removed once their current waiters drain and can
    then be retried from scratch.

    The registry is intentionally event-loop local.  Servers with multiple
    independent HTTP frontend processes must not advertise this capability;
    the SGLang route enforces that fail-closed condition.
    """

    def __init__(
        self,
        *,
        completed_result_ttl_seconds: float = 60.0,
        max_completed_results: int = 1024,
        execution_timeout_seconds: float = 1800.0,
        orphan_grace_seconds: float = 30.0,
    ) -> None:
        if (
            isinstance(completed_result_ttl_seconds, bool)
            or not isinstance(completed_result_ttl_seconds, (int, float))
            or not math.isfinite(float(completed_result_ttl_seconds))
            or float(completed_result_ttl_seconds) <= 0
        ):
            raise ValueError("completed_result_ttl_seconds must be positive")
        if (
            isinstance(max_completed_results, bool)
            or not isinstance(max_completed_results, int)
            or max_completed_results <= 0
        ):
            raise ValueError("max_completed_results must be a positive integer")
        if (
            isinstance(execution_timeout_seconds, bool)
            or not isinstance(execution_timeout_seconds, (int, float))
            or not math.isfinite(float(execution_timeout_seconds))
            or float(execution_timeout_seconds) <= 0
        ):
            raise ValueError("execution_timeout_seconds must be positive")
        if (
            isinstance(orphan_grace_seconds, bool)
            or not isinstance(orphan_grace_seconds, (int, float))
            or not math.isfinite(float(orphan_grace_seconds))
            or float(orphan_grace_seconds) < 0
        ):
            raise ValueError("orphan_grace_seconds must be nonnegative")
        self._completed_result_ttl_seconds = float(completed_result_ttl_seconds)
        self._max_completed_results = int(max_completed_results)
        self._execution_timeout_seconds = float(execution_timeout_seconds)
        self._orphan_grace_seconds = float(orphan_grace_seconds)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None
        self._flights: dict[str, _RecursiveCompressionChainFlight] = {}

    def _bind_running_loop(self) -> tuple[asyncio.AbstractEventLoop, asyncio.Lock]:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
            self._lock = asyncio.Lock()
        elif self._loop is not loop:
            if self._flights:
                raise RuntimeError(
                    "recursive compression singleflight cannot cross event loops"
                )
            self._loop = loop
            self._lock = asyncio.Lock()
        if self._lock is None:  # pragma: no cover - defensive invariant
            raise AssertionError("singleflight loop has no lock")
        return loop, self._lock

    def _prune_completed_locked(self, *, now: float) -> None:
        expired = [
            operation_id
            for operation_id, flight in self._flights.items()
            if flight.completed_at is not None
            and flight.waiters == 0
            and now - flight.completed_at >= self._completed_result_ttl_seconds
        ]
        for operation_id in expired:
            self._flights.pop(operation_id, None)

        completed = sorted(
            (
                (flight.completed_at, operation_id)
                for operation_id, flight in self._flights.items()
                if flight.completed_at is not None and flight.waiters == 0
            ),
            key=lambda item: item[0],
        )
        overflow = len(completed) - self._max_completed_results
        for _, operation_id in completed[: max(0, overflow)]:
            self._flights.pop(operation_id, None)

    async def _execute(self, execute: ChainExecution) -> dict[str, Any]:
        try:
            result = await asyncio.wait_for(
                execute(),
                timeout=self._execution_timeout_seconds,
            )
        except TimeoutError as error:
            raise RecursiveCompressionChainContractError(
                "recursive compression chain exceeded its server execution deadline"
            ) from error
        except asyncio.CancelledError as error:
            # The shared task is never cancelled by an individual waiter.
            # This branch therefore means zero-waiter orphan cleanup or
            # server lifecycle shutdown, after the executor had a chance to
            # abort its active child and release owned candidates.
            raise RecursiveCompressionChainContractError(
                "recursive compression chain execution was cancelled by the server"
            ) from error
        if not isinstance(result, Mapping):
            raise RecursiveCompressionChainContractError(
                "chain execution returned a non-object result"
            )
        return copy.deepcopy(dict(result))

    async def _record_completion(
        self,
        *,
        operation_id: str,
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        loop, lock = self._bind_running_loop()
        try:
            failure = task.cancelled() or task.exception() is not None
        except asyncio.CancelledError:  # pragma: no cover - task.cancelled covers it
            failure = True
        async with lock:
            flight = self._flights.get(operation_id)
            if flight is None or flight.task is not task:
                return
            if flight.orphan_cancel_task is not None:
                flight.orphan_cancel_task.cancel()
                flight.orphan_cancel_task = None
            if failure:
                if flight.waiters == 0:
                    self._flights.pop(operation_id, None)
                return
            if flight.completed_at is None:
                flight.completed_at = loop.time()
            self._prune_completed_locked(now=loop.time())

    async def _cancel_if_still_orphaned(
        self,
        *,
        operation_id: str,
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        try:
            await asyncio.sleep(self._orphan_grace_seconds)
        except asyncio.CancelledError:
            return
        _, lock = self._bind_running_loop()
        async with lock:
            flight = self._flights.get(operation_id)
            if flight is None or flight.task is not task:
                return
            flight.orphan_cancel_task = None
            if flight.waiters == 0 and not task.done():
                flight.orphan_cancelled = True
                task.cancel("recursive compression chain lost every HTTP waiter")

    def _schedule_completion(
        self,
        operation_id: str,
        task: asyncio.Task[dict[str, Any]],
    ) -> None:
        # Retrieving the exception here prevents an orphaned leader (all HTTP
        # waiters disconnected) from producing "Task exception was never
        # retrieved".  Awaiting the task later still observes the exception.
        if not task.cancelled():
            task.exception()
        loop = task.get_loop()
        if not loop.is_closed():
            loop.create_task(
                self._record_completion(operation_id=operation_id, task=task)
            )

    async def run(
        self,
        request_payload: Mapping[str, Any],
        *,
        execute: ChainExecution,
    ) -> dict[str, Any]:
        """Run or join one exact operation without propagating waiter cancel."""

        chain = RecursiveCompressionChainRequest.parse(request_payload)
        payload_digest = _digest_json(chain.to_json())
        loop, lock = self._bind_running_loop()
        async with lock:
            self._prune_completed_locked(now=loop.time())
            flight = self._flights.get(chain.operation_id)
            if flight is not None and flight.payload_digest != payload_digest:
                raise RecursiveCompressionChainContractError(
                    "operation_id is already bound to a different chain payload"
                )
            if flight is None:
                task = loop.create_task(self._execute(execute))
                flight = _RecursiveCompressionChainFlight(
                    payload_digest=payload_digest,
                    task=task,
                )
                self._flights[chain.operation_id] = flight
                task.add_done_callback(
                    lambda done, operation_id=chain.operation_id: (
                        self._schedule_completion(operation_id, done)
                    )
                )
            elif flight.orphan_cancel_task is not None:
                flight.orphan_cancel_task.cancel()
                flight.orphan_cancel_task = None
            flight.waiters += 1

        retry_after_orphan_cleanup = False
        try:
            result = await asyncio.shield(flight.task)
        except RecursiveCompressionChainContractError:
            if flight.orphan_cancelled:
                retry_after_orphan_cleanup = True
            else:
                raise
        finally:
            _, lock = self._bind_running_loop()
            async with lock:
                current = self._flights.get(chain.operation_id)
                if current is flight:
                    if current.waiters <= 0:  # pragma: no cover - defensive invariant
                        raise AssertionError("singleflight waiter count underflow")
                    current.waiters -= 1
                    if current.task.done():
                        if current.orphan_cancel_task is not None:
                            current.orphan_cancel_task.cancel()
                            current.orphan_cancel_task = None
                        if (
                            current.task.cancelled()
                            or current.task.exception() is not None
                        ):
                            if current.waiters == 0:
                                self._flights.pop(chain.operation_id, None)
                        elif current.completed_at is None:
                            current.completed_at = loop.time()
                    elif current.waiters == 0 and current.orphan_cancel_task is None:
                        current.orphan_cancel_task = loop.create_task(
                            self._cancel_if_still_orphaned(
                                operation_id=chain.operation_id,
                                task=current.task,
                            )
                        )
        if retry_after_orphan_cleanup:
            return await self.run(request_payload, execute=execute)
        return copy.deepcopy(result)

    async def invalidate_final_handle(
        self,
        handle: Mapping[str, Any],
    ) -> None:
        """Forget replay receipts whose final descriptor was released."""

        descriptor_digest = handle.get("descriptor_digest")
        if not isinstance(descriptor_digest, str):
            return
        _, lock = self._bind_running_loop()
        async with lock:
            for operation_id, flight in list(self._flights.items()):
                if not flight.task.done() or flight.task.cancelled():
                    continue
                try:
                    result = flight.task.result()
                except BaseException:  # noqa: BLE001, S112 - inspect failed task
                    continue
                final_handle = result.get("final_memory_handle")
                if (
                    isinstance(final_handle, Mapping)
                    and final_handle.get("descriptor_digest") == descriptor_digest
                ):
                    self._flights.pop(operation_id, None)


async def _maybe_await(value: Awaitable[Any] | Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _scheduler_payload(
    chain: RecursiveCompressionChainRequest,
    transition: RecursiveCompressionTransition,
    parent: TPGPUMemoryHandle | None,
    publish: Mapping[str, str],
) -> dict[str, Any]:
    compressor: dict[str, Any] = {
        "version": 1,
        "mode": "encode",
        "evidence_spans": [list(transition.evidence_span)],
        "source_token_lengths": [transition.source_token_length],
        "memory_handle_publish": dict(publish),
    }
    if transition.recursive_source_contract is not None:
        compressor["recursive_source_contract"] = transition.recursive_source_contract
    if parent is not None:
        start = transition.evidence_span[0]
        compressor["memory_positions"] = list(range(start, start + parent.shape[0]))
        compressor["memory_handle"] = parent.to_json()
    return {
        "rid": "specforge_" + publish["op_id"],
        "input_ids": list(transition.source_input_ids),
        "extra_key": (
            f"specforge-chain-{chain.operation_id}-{transition.transition_index}"
        ),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 0,
            "custom_params": {"specforge_context_compressor": compressor},
        },
        "return_hidden_states": False,
        "custom_logit_processor": chain.custom_logit_processor,
        "cache_policy": chain.cache_policy,
    }


def _extract_published_handle(
    response: Any,
    *,
    server_epoch: str | None,
    expected_memory_identity: str,
    expected_request_digest: str,
) -> TPGPUMemoryHandle:
    if not isinstance(response, Mapping):
        raise RecursiveCompressionChainContractError(
            "scheduler returned a non-object chain transition response"
        )
    meta = response.get("meta_info")
    if not isinstance(meta, Mapping):
        raise RecursiveCompressionChainContractError(
            "scheduler chain transition response has no meta_info object"
        )
    return TPGPUMemoryHandle.parse(
        meta.get(GPU_MEMORY_HANDLE_META_KEY),
        expected_server_epoch=server_epoch,
        expected_memory_identity=expected_memory_identity,
        expected_request_digest=expected_request_digest,
    )


async def execute_recursive_compression_chain(
    request_payload: Mapping[str, Any],
    *,
    submit_scheduler_request: SchedulerSubmit,
    release_memory_handle: HandleRelease,
    authoritative_server_epoch: str,
    authoritative_hook_abi_digest: str,
    abort_scheduler_request: SchedulerAbort | None = None,
) -> dict[str, Any]:
    """Execute a chain only through sequential ordinary scheduler requests.

    Candidate handles are private to this invocation.  After a child is
    published and validated, its immediately preceding chain-owned handle is
    released before the next transition is submitted.  Thus live candidate
    ownership is O(1), independent of chain length.  On failure the at-most
    two handles in that sliding window are released and the caller's parent
    handle is left untouched.
    """

    chain = RecursiveCompressionChainRequest.parse(request_payload)
    if chain.server_epoch != _require_identifier(
        authoritative_server_epoch, name="authoritative_server_epoch"
    ):
        raise RecursiveCompressionChainContractError(
            "chain request targets a stale server epoch"
        )
    if chain.hook_abi_digest != _require_digest(
        authoritative_hook_abi_digest, name="authoritative_hook_abi_digest"
    ):
        raise RecursiveCompressionChainContractError(
            "chain request targets a different compressor hook ABI"
        )
    current = chain.parent_memory_handle
    server_epoch = chain.server_epoch
    # In insertion order: normally just ``current``; briefly previous+child
    # between a successful publish and release of the previous intermediate.
    owned_candidates: dict[str, TPGPUMemoryHandle] = {}
    receipts: list[dict[str, Any]] = []
    active_child_rid: str | None = None
    try:
        for transition in chain.transitions:
            previous_slots = 0 if current is None else current.shape[0]
            publish = derive_recursive_compression_transition_claim(
                chain,
                transition,
                previous_memory=(None if current is None else current.to_json()),
                previous_memory_slots=previous_slots,
            )
            scheduler_payload = _scheduler_payload(
                chain,
                transition,
                current,
                publish,
            )
            active_child_rid = str(scheduler_payload["rid"])
            response = await submit_scheduler_request(scheduler_payload)
            active_child_rid = None
            candidate = _extract_published_handle(
                response,
                server_epoch=server_epoch,
                expected_memory_identity=publish["memory_identity"],
                expected_request_digest=publish["request_digest"],
            )
            candidate_digest = str(candidate.descriptor["descriptor_digest"])
            owned_candidates[candidate_digest] = candidate
            # The descriptor is now known well enough for the idempotent
            # release protocol.  Perform transition-specific checks only
            # after taking cleanup ownership, otherwise an authentic but
            # wrong-shape/identity receipt would leak its published tensor.
            candidate = TPGPUMemoryHandle.parse(
                candidate.to_json(),
                expected_server_epoch=server_epoch,
                expected_slots=transition.expected_slot_count,
                expected_hidden_size=chain.expected_hidden_size,
                expected_memory_identity=publish["memory_identity"],
                expected_request_digest=publish["request_digest"],
            )
            # Append the server-published descriptor before checking for a
            # forbidden tensor payload.  If the scheduler violates the
            # handle-only response ABI, the exception path can now release
            # the known candidate instead of leaking it until registry TTL.
            meta = response["meta_info"]
            forbidden = {"hidden_states", "memory_embeddings"}.intersection(meta)
            if forbidden:
                raise RecursiveCompressionChainContractError(
                    "scheduler chain transition leaked tensor payload fields: "
                    + ", ".join(sorted(forbidden))
                )
            previous = current
            if previous is not None:
                previous_digest = str(previous.descriptor["descriptor_digest"])
                if previous_digest in owned_candidates:
                    await _maybe_await(release_memory_handle(previous.to_json()))
                    owned_candidates.pop(previous_digest, None)
            current = candidate
            receipts.append(
                {
                    "transition_index": transition.transition_index,
                    "memory_identity": candidate.memory_identity,
                    "request_digest": candidate.request_digest,
                    "descriptor_digest": candidate.descriptor["descriptor_digest"],
                    "shape": list(candidate.shape),
                }
            )
    except BaseException as error:
        cleanup_errors: list[BaseException] = []
        if active_child_rid is not None and abort_scheduler_request is not None:
            try:
                await _maybe_await(abort_scheduler_request(active_child_rid))
            except BaseException as abort_error:  # noqa: BLE001
                cleanup_errors.append(abort_error)
        for candidate in reversed(tuple(owned_candidates.values())):
            try:
                await _maybe_await(release_memory_handle(candidate.to_json()))
            except BaseException as release_error:  # noqa: BLE001
                cleanup_errors.append(release_error)
        if cleanup_errors:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(
                    "chain abort/candidate cleanup also failed: "
                    + "; ".join(str(item) for item in cleanup_errors)
                )
        raise
    if (
        current is None or not owned_candidates
    ):  # pragma: no cover - parser requires two
        raise AssertionError("validated chain produced no candidate handle")
    final_digest = str(current.descriptor["descriptor_digest"])
    if tuple(owned_candidates) != (final_digest,):  # pragma: no cover - invariant
        raise AssertionError("validated chain retained a non-final candidate handle")
    return RecursiveCompressionChainResult(
        operation_id=chain.operation_id,
        final_memory_handle=current,
        transition_receipts=tuple(receipts),
    ).to_json()


__all__ = [
    "GPU_MEMORY_HANDLE_CAPABILITY",
    "GPU_MEMORY_HANDLE_META_KEY",
    "GPU_MEMORY_HANDLE_PUBLISH_SCHEMA",
    "GPU_MEMORY_HANDLE_RELEASE_ROUTE",
    "GPU_MEMORY_HANDLE_RELEASE_SCHEMA",
    "GPU_MEMORY_HANDLE_SCHEMA",
    "GPU_MEMORY_HANDLE_SERVER_INFO_SCHEMA",
    "GPU_MEMORY_IDENTITY_KIND",
    "MEMORY_TRANSPORT_POLICY_LEGACY_CPU_ROUNDTRIP_V1",
    "MEMORY_TRANSPORT_POLICY_TP_GPU_CHAIN_V1",
    "MEMORY_TRANSPORT_TP_GPU_CHAIN_REQUIRED_CAPABILITIES",
    "RECURSIVE_COMPRESSION_CHAIN_CAPABILITY",
    "RECURSIVE_COMPRESSION_CHAIN_COMMIT_POLICY",
    "RECURSIVE_COMPRESSION_CHAIN_EXECUTION_POLICY",
    "RECURSIVE_COMPRESSION_CHAIN_REQUEST_SCHEMA",
    "RECURSIVE_COMPRESSION_CHAIN_RESPONSE_SCHEMA",
    "RECURSIVE_COMPRESSION_CHAIN_ROUTE",
    "RECURSIVE_COMPRESSION_POLICIES",
    "RECURSIVE_COMPRESSION_POLICY_LEGACY_PER_TRANSITION_V1",
    "RECURSIVE_COMPRESSION_POLICY_SERVER_SCHEDULER_CHAIN_V1",
    "REQUEST_CACHE_BYPASS_CAPABILITY",
    "SOFT_MEMORY_PREFIX_REUSE_BATCH_CAPABILITY",
    "SOFT_MEMORY_PREFIX_REUSE_WIRE_POLICY",
    "MemoryTransportAttestation",
    "RecursiveCompressionChainContractError",
    "RecursiveCompressionChainRequest",
    "RecursiveCompressionChainResult",
    "RecursiveCompressionChainSingleFlight",
    "RecursiveCompressionTransition",
    "TPGPUMemoryHandle",
    "attest_memory_transport_server_info",
    "derive_recursive_compression_transition_claim",
    "execute_recursive_compression_chain",
    "parse_recursive_compression_chain_response",
]
