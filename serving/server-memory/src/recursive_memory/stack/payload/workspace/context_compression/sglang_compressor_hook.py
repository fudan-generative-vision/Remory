"""Fail-closed SGLang hook core for GrepSeek context compression.

The hook is intentionally a small bridge rather than a second implementation
of the model.  SGLang owns the frozen target prefill and exposes the selected
Qwen3.5 layer outputs as ``aux_hidden_states``.  This module gathers the
checkpoint-defined source spans (legacy stdout segments or the complete
post-prefix recursive source), runs the existing
:class:`GrepSeekDFlashCompressor`, and replaces
``LogitsProcessorOutput.hidden_states`` with the packed valid memory slots.

No SGLang module is imported at module import time.  This keeps the normal
SpecForge training/evaluation environment usable and also makes the contract
testable with light-weight stand-ins.  The returned hook object has an
explicit :meth:`SpecForgeContextCompressorHook.setup_model` lifecycle method;
the SGLang integration must call it on the loaded target model before the hook
can process an encode request.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch import nn

from context_compression.grepseek_serving_contract import (
    CAUSAL_OBSERVATION_CACHE_V1,
    QWEN36_TOKEN_CLOCK_V1,
    RECURSIVE_HISTORY_V2,
    RECURSIVE_WORKING_MEMORY_V3,
    WORKING_MEMORY_V3_SOURCE_CONTRACT,
    resolve_grepseek_checkpoint_serving_contract,
)


CUSTOM_PARAM_KEY = "specforge_context_compressor"
GENERATION_DEGENERATION_GUARD_CUSTOM_PARAM_KEY = (
    "specforge_generation_degeneration_guard"
)
GENERATION_DEGENERATION_GUARD_SERVER_ABI = (
    "specforge_exact_periodic_guard_global_class_v1"
)
CUSTOM_PARAM_VERSION = 1
ENCODE_MODE = "encode"
RECOVER_MODE = "recover"
GPU_MEMORY_HANDLE_CAPABILITY = "specforge_tp_gpu_memory_handle_v1"
GPU_MEMORY_HANDLE_SCHEMA = "specforge_tp_gpu_memory_handle_v1"
GPU_MEMORY_HANDLE_PUBLISH_SCHEMA = "specforge_tp_gpu_memory_handle_publish_v1"
GPU_MEMORY_HANDLE_META_KEY = "specforge_gpu_memory_handle"
GPU_MEMORY_IDENTITY_KIND = "specforge_transition_provenance_v1"
GPU_MEMORY_TRANSITION_REQUEST_SCHEMA = "specforge_memory_transition_request_v1"
GPU_MEMORY_IDENTITY_SCHEMA = "specforge_memory_identity_v1"
GPU_MEMORY_HANDLE_SERVER_INFO_SCHEMA = (
    "specforge_tp_gpu_memory_handle_server_info_v1"
)
GPU_MEMORY_HANDLE_POLICY = GPU_MEMORY_HANDLE_CAPABILITY
GPU_MEMORY_HANDLE_SERVER_EPOCH_ENV = "SGLANG_SPECFORGE_MEMORY_SERVER_EPOCH"
_GPU_MEMORY_HANDLE_FIELDS = frozenset(
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


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _canonical_sha256_digest(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must use sha256:<64 lowercase hex> form")
    return value


def normalize_memory_handle_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one server-minted JSON descriptor without resolving its tensor."""

    if not isinstance(value, Mapping) or set(value) != _GPU_MEMORY_HANDLE_FIELDS:
        raise ValueError("memory_handle must contain exactly the v1 descriptor fields")
    descriptor = dict(value)
    if descriptor["schema"] != GPU_MEMORY_HANDLE_SCHEMA:
        raise ValueError("unsupported memory_handle schema")
    if descriptor["identity_kind"] != GPU_MEMORY_IDENTITY_KIND:
        raise ValueError("unsupported memory_handle identity_kind")
    for name in ("server_epoch", "handle_id", "dtype"):
        if not isinstance(descriptor[name], str) or not descriptor[name]:
            raise ValueError(f"memory_handle {name} must be a nonempty string")
    for name in ("memory_identity", "request_digest", "descriptor_digest"):
        _canonical_sha256_digest(descriptor[name], name=f"memory_handle {name}")
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
        raise ValueError("memory_handle shape must contain two positive integers")
    nbytes = descriptor["nbytes"]
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
        raise ValueError("memory_handle nbytes must be a positive integer")
    base = {
        key: item
        for key, item in descriptor.items()
        if key != "descriptor_digest"
    }
    if _canonical_sha256(base) != descriptor["descriptor_digest"]:
        raise ValueError("memory_handle descriptor_digest mismatch")
    descriptor["shape"] = [int(item) for item in shape]
    return descriptor


def compute_memory_handle_request_digest(
    *,
    input_ids: Sequence[int],
    evidence_spans: Sequence[Sequence[int]],
    source_token_lengths: Sequence[int],
    memory_positions: Optional[Sequence[int]] = None,
    previous_memory_handle: Optional[Mapping[str, Any]] = None,
    recursive_source_contract: Optional[str] = None,
) -> str:
    """Hash the actual transition provenance using a stable public schema."""

    if isinstance(input_ids, (str, bytes)) or not isinstance(input_ids, Sequence):
        raise ValueError("input_ids must be a sequence")
    normalized_input_ids: list[int] = []
    for index, token_id in enumerate(input_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"input_ids[{index}] must be an integer")
        normalized_input_ids.append(int(token_id))
    if not normalized_input_ids:
        raise ValueError("input_ids must not be empty")
    spans, lengths = _normalize_evidence_contract(
        evidence_spans, source_token_lengths
    )
    if memory_positions is None:
        positions: list[int] = []
    else:
        if isinstance(memory_positions, (str, bytes)) or not isinstance(
            memory_positions, Sequence
        ):
            raise ValueError("memory_positions must be a sequence")
        positions = []
        previous_position = -1
        for index, raw_position in enumerate(memory_positions):
            if isinstance(raw_position, bool) or not isinstance(raw_position, int):
                raise ValueError(f"memory_positions[{index}] must be an integer")
            position = int(raw_position)
            if position <= previous_position:
                raise ValueError(
                    "memory_positions must be nonnegative and strictly increasing"
                )
            positions.append(position)
            previous_position = position
    previous_memory = None
    if previous_memory_handle is not None:
        descriptor = normalize_memory_handle_descriptor(previous_memory_handle)
        previous_memory = {
            "server_epoch": descriptor["server_epoch"],
            "identity_kind": descriptor["identity_kind"],
            "memory_identity": descriptor["memory_identity"],
            "descriptor_digest": descriptor["descriptor_digest"],
        }
    if recursive_source_contract is not None and (
        not isinstance(recursive_source_contract, str)
        or not recursive_source_contract
    ):
        raise ValueError("recursive_source_contract must be a nonempty string")
    provenance = {
        "schema": GPU_MEMORY_TRANSITION_REQUEST_SCHEMA,
        "custom_param_version": CUSTOM_PARAM_VERSION,
        "mode": ENCODE_MODE,
        "input_ids": normalized_input_ids,
        "evidence_spans": spans,
        "source_token_lengths": lengths,
        "memory_positions": positions,
        "previous_memory": previous_memory,
        "recursive_source_contract": recursive_source_contract,
    }
    return _canonical_sha256(provenance)


def compute_memory_handle_identity(
    *, server_epoch: str, request_digest: str, hook_abi_digest: str
) -> str:
    if not isinstance(server_epoch, str) or not server_epoch:
        raise ValueError("server_epoch must be a nonempty string")
    request_digest = _canonical_sha256_digest(
        request_digest, name="request_digest"
    )
    hook_abi_digest = _canonical_sha256_digest(
        hook_abi_digest, name="hook_abi_digest"
    )
    return _canonical_sha256(
        {
            "schema": GPU_MEMORY_IDENTITY_SCHEMA,
            "identity_kind": GPU_MEMORY_IDENTITY_KIND,
            "server_epoch": server_epoch,
            "request_digest": request_digest,
            "hook_abi_digest": hook_abi_digest,
        }
    )


class SpecForgeCompressorContractError(RuntimeError):
    """Raised when a serving request could violate the trained model ABI."""


@dataclass(frozen=True)
class CompressorCheckpointMetadata:
    checkpoint: Path
    state_path: Path
    raw_config: Mapping[str, Any]
    target_layer_ids: tuple[int, ...]
    target_hidden_size: int
    compression_ratio: int
    local_window: int
    output_embedding_rms: float
    context_organization: str
    checkpoint_schema: str


CompressorLoader = Callable[
    [CompressorCheckpointMetadata, torch.device, torch.dtype], nn.Module
]


def _compute_hook_abi_digest(metadata: CompressorCheckpointMetadata) -> str:
    state_stat = metadata.state_path.stat()
    return _canonical_sha256(
        {
            "schema": "specforge_context_compressor_hook_abi_v1",
            "custom_param_version": CUSTOM_PARAM_VERSION,
            "checkpoint_config_digest": _canonical_sha256(
                dict(metadata.raw_config)
            ),
            "checkpoint_artifact": {
                "resolved_path": str(metadata.state_path.resolve()),
                "size": int(state_stat.st_size),
                "mtime_ns": int(state_stat.st_mtime_ns),
            },
            "checkpoint_schema": metadata.checkpoint_schema,
            "context_organization": metadata.context_organization,
            "target_layer_ids": list(metadata.target_layer_ids),
            "target_hidden_size": metadata.target_hidden_size,
            "compression_ratio": metadata.compression_ratio,
            "local_window": metadata.local_window,
            "output_embedding_rms": metadata.output_embedding_rms,
        }
    )


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _normalize_evidence_contract(
    evidence_spans: Sequence[Sequence[int]],
    source_token_lengths: Sequence[int],
) -> tuple[list[list[int]], list[int]]:
    if isinstance(evidence_spans, (str, bytes)) or not isinstance(
        evidence_spans, Sequence
    ):
        raise ValueError("evidence_spans must be a sequence of [start, end] rows")
    if isinstance(source_token_lengths, (str, bytes)) or not isinstance(
        source_token_lengths, Sequence
    ):
        raise ValueError("source_token_lengths must be a sequence of integers")
    if not evidence_spans:
        raise ValueError("encode mode requires at least one evidence span")
    if len(evidence_spans) != len(source_token_lengths):
        raise ValueError("evidence spans and source token lengths differ in arity")

    normalized_spans: list[list[int]] = []
    previous_end = -1
    for index, span in enumerate(evidence_spans):
        if isinstance(span, (str, bytes)) or not isinstance(span, Sequence):
            raise ValueError(f"evidence_spans[{index}] must be [start, end]")
        if len(span) != 2:
            raise ValueError(f"evidence_spans[{index}] must contain two integers")
        start, end = span
        if any(isinstance(value, bool) or not isinstance(value, int) for value in span):
            raise ValueError(f"evidence_spans[{index}] must contain integers")
        start = int(start)
        end = int(end)
        if start < 0 or end <= start:
            raise ValueError(f"evidence_spans[{index}] is empty or negative")
        if start < previous_end:
            raise ValueError("evidence spans must be ordered and disjoint")
        normalized_spans.append([start, end])
        previous_end = end

    normalized_lengths = [
        _positive_int(value, name=f"source_token_lengths[{index}]")
        for index, value in enumerate(source_token_lengths)
    ]
    return normalized_spans, normalized_lengths


def build_encode_custom_params(
    evidence_spans: Sequence[Sequence[int]],
    source_token_lengths: Sequence[int],
    *,
    memory_handle_publish: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the JSON-safe ``custom_params`` for one source-only prefill."""

    spans, lengths = _normalize_evidence_contract(evidence_spans, source_token_lengths)
    payload: dict[str, Any] = {
        "version": CUSTOM_PARAM_VERSION,
        "mode": ENCODE_MODE,
        "evidence_spans": spans,
        "source_token_lengths": lengths,
    }
    if memory_handle_publish is not None:
        payload["memory_handle_publish"] = normalize_memory_handle_publish_claim(
            memory_handle_publish
        )
    return {CUSTOM_PARAM_KEY: payload}


def normalize_memory_handle_publish_claim(
    value: Mapping[str, Any],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "op_id",
        "memory_identity",
        "request_digest",
    }:
        raise ValueError(
            "memory_handle_publish must contain schema, op_id, "
            "memory_identity, and request_digest"
        )
    if value["schema"] != GPU_MEMORY_HANDLE_PUBLISH_SCHEMA:
        raise ValueError("unsupported memory_handle_publish schema")
    op_id = value["op_id"]
    if not isinstance(op_id, str) or not op_id or len(op_id) > 256:
        raise ValueError("memory_handle_publish op_id must be 1-256 characters")
    memory_identity = _canonical_sha256_digest(
        value["memory_identity"], name="memory_handle_publish memory_identity"
    )
    request_digest = _canonical_sha256_digest(
        value["request_digest"], name="memory_handle_publish request_digest"
    )
    return {
        "schema": GPU_MEMORY_HANDLE_PUBLISH_SCHEMA,
        "op_id": op_id,
        "memory_identity": memory_identity,
        "request_digest": request_digest,
    }


def _normalize_memory_contract(
    *,
    memory_positions: Sequence[int],
    memory_embeddings: Sequence[Sequence[float]],
    hidden_size: int = 4096,
) -> tuple[list[int], list[list[float]]]:
    if isinstance(memory_positions, (str, bytes)) or not isinstance(
        memory_positions, Sequence
    ):
        raise ValueError("memory_positions must be a one-dimensional sequence")
    if isinstance(memory_embeddings, (str, bytes)) or not isinstance(
        memory_embeddings, Sequence
    ):
        raise ValueError("memory_embeddings must be a rank-two sequence")
    if not memory_positions:
        raise ValueError("memory_positions must not be empty")
    if len(memory_positions) != len(memory_embeddings):
        raise ValueError("memory positions and embeddings differ in row count")

    normalized_positions: list[int] = []
    previous_position = -1
    for index, value in enumerate(memory_positions):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"memory_positions[{index}] must be an integer")
        position = int(value)
        if position < 0:
            raise ValueError(f"memory_positions[{index}] must be nonnegative")
        if position <= previous_position:
            raise ValueError("memory_positions must be unique and strictly increasing")
        normalized_positions.append(position)
        previous_position = position

    normalized_embeddings: list[list[float]] = []
    for row_index, row in enumerate(memory_embeddings):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"memory_embeddings[{row_index}] must be a row")
        if len(row) != hidden_size:
            raise ValueError(
                f"memory_embeddings[{row_index}] must have hidden width {hidden_size}"
            )
        normalized_row: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"memory_embeddings[{row_index}][{column_index}] must be numeric"
                )
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError("memory_embeddings must contain only finite values")
            normalized_row.append(converted)
        normalized_embeddings.append(normalized_row)

    return normalized_positions, normalized_embeddings


def build_recursive_encode_custom_params(
    evidence_spans: Sequence[Sequence[int]],
    source_token_lengths: Sequence[int],
    *,
    memory_positions: Optional[Sequence[int]] = None,
    memory_embeddings: Optional[Sequence[Sequence[float]]] = None,
    memory_handle: Optional[Mapping[str, Any]] = None,
    memory_handle_publish: Optional[Mapping[str, Any]] = None,
    recursive_source_contract: Optional[str] = None,
    hidden_size: int = 4096,
) -> dict[str, Any]:
    """Build one exact ``P + M_prev + incremental_suffix`` encode payload.

    The first recursive transition has no previous memory and therefore uses
    only the ordinary encode fields.  Later transitions add sparse overrides
    for the contiguous ``M_prev`` rows.  The checkpoint-bound hook validates
    that the single evidence span covers every post-``P`` physical row.
    """

    spans, lengths = _normalize_evidence_contract(evidence_spans, source_token_lengths)
    if len(spans) != 1:
        raise ValueError("recursive encode requires exactly one physical source span")
    payload: dict[str, Any] = {
        "version": CUSTOM_PARAM_VERSION,
        "mode": ENCODE_MODE,
        "evidence_spans": spans,
        "source_token_lengths": lengths,
    }
    if recursive_source_contract is not None:
        if (
            not isinstance(recursive_source_contract, str)
            or not recursive_source_contract
        ):
            raise ValueError("recursive_source_contract must be a nonempty string")
        payload["recursive_source_contract"] = recursive_source_contract
    if memory_embeddings is not None and memory_handle is not None:
        raise ValueError("recursive encode accepts memory_embeddings or memory_handle, not both")
    has_memory_value = memory_embeddings is not None or memory_handle is not None
    if (memory_positions is not None) != has_memory_value:
        raise ValueError(
            "recursive encode memory_positions and one memory value must be set together"
        )
    if memory_positions is not None and memory_embeddings is not None:
        positions, embeddings = _normalize_memory_contract(
            memory_positions=memory_positions,
            memory_embeddings=memory_embeddings,
            hidden_size=hidden_size,
        )
        start, end = spans[0]
        if positions != list(range(start, start + len(positions))):
            raise ValueError(
                "recursive previous memory must be contiguous at the source-span start"
            )
        if start + len(positions) >= end:
            raise ValueError(
                "recursive encode requires a nonempty incremental discrete suffix"
            )
        payload["memory_positions"] = positions
        payload["memory_embeddings"] = embeddings
    elif memory_positions is not None and memory_handle is not None:
        descriptor = normalize_memory_handle_descriptor(memory_handle)
        positions = [int(value) for value in memory_positions]
        start, end = spans[0]
        if positions != list(range(start, start + len(positions))):
            raise ValueError(
                "recursive previous memory must be contiguous at the source-span start"
            )
        if start + len(positions) >= end:
            raise ValueError(
                "recursive encode requires a nonempty incremental discrete suffix"
            )
        if descriptor["shape"][0] != len(positions):
            raise ValueError(
                "recursive memory handle row count must match memory_positions"
            )
        if descriptor["shape"][1] != hidden_size:
            raise ValueError("recursive memory handle hidden width mismatch")
        payload["memory_positions"] = positions
        payload["memory_handle"] = descriptor
    if memory_handle_publish is not None:
        payload["memory_handle_publish"] = normalize_memory_handle_publish_claim(
            memory_handle_publish
        )
    return {CUSTOM_PARAM_KEY: payload}


def build_recover_custom_params(
    *,
    memory_positions: Sequence[int],
    memory_embeddings: Optional[Sequence[Sequence[float]]] = None,
    memory_handle: Optional[Mapping[str, Any]] = None,
    hidden_size: int = 4096,
) -> dict[str, Any]:
    """Build JSON-safe sparse embedding overrides for target recovery.

    Positions refer to the batch-one, unpadded recovery prefix and must be in
    the same order as the packed memory rows returned by encode mode.  The
    Qwen target-side sparse-scatter integration consumes this payload before
    its embedding forward; the logits hook itself deliberately leaves recover
    forwards unchanged.
    """

    if (memory_embeddings is None) == (memory_handle is None):
        raise ValueError(
            "recover requires exactly one of memory_embeddings or memory_handle"
        )
    if memory_embeddings is not None:
        positions, embeddings = _normalize_memory_contract(
            memory_positions=memory_positions,
            memory_embeddings=memory_embeddings,
            hidden_size=hidden_size,
        )
        memory_value = {"memory_embeddings": embeddings}
    else:
        assert memory_handle is not None
        descriptor = normalize_memory_handle_descriptor(memory_handle)
        positions = [int(value) for value in memory_positions]
        if descriptor["shape"] != [len(positions), hidden_size]:
            raise ValueError(
                "recover memory handle shape must match positions and hidden size"
            )
        memory_value = {"memory_handle": descriptor}
    return {
        CUSTOM_PARAM_KEY: {
            "version": CUSTOM_PARAM_VERSION,
            "mode": RECOVER_MODE,
            "memory_positions": positions,
            **memory_value,
        }
    }


class SpecForgeContextCompressorNoOpLogitProcessor:
    """Duck-typed SGLang ``CustomLogitProcessor`` that preserves logits.

    SGLang only copies request ``custom_params`` into ``ForwardBatch`` when a
    custom logit processor is present.  This no-op processor exists solely to
    activate that transport.  It deliberately avoids inheriting from SGLang's
    base class so importing this module never imports SGLang.  SGLang's wire
    format only requires a dill-serializable callable with this signature.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor:
        del custom_param_list
        return logits

    @classmethod
    def to_str(cls) -> str:
        """Serialize using the wire format understood by SGLang."""

        try:
            import dill
        except ImportError as exc:  # pragma: no cover - serving dependency
            raise RuntimeError(
                "dill is required to serialize the SGLang no-op logit processor"
            ) from exc
        return json.dumps({"callable": dill.dumps(cls).hex()})


def noop_custom_logit_processor_str() -> str:
    """Return the serialized no-op processor for an SGLang request."""

    return SpecForgeContextCompressorNoOpLogitProcessor.to_str()


class SpecForgeCompositeMetadataLogitProcessor:
    """Metadata transport with an opt-in exact-period generation guard.

    This must remain a module-level class.  The BrowseComp controller currently
    runs under Python 3.12 while the SGLang scheduler runs under Python 3.11;
    serializing a local class would embed version-specific Python bytecode and
    can crash the scheduler interpreter.  Dill therefore transports only this
    importable class reference, and the scheduler executes code compiled by its
    own interpreter.  A server restart is required after changing this class.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[list[dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits
        for batch_index, params in enumerate(custom_param_list):
            if not isinstance(params, dict):
                continue
            guard = params.get(GENERATION_DEGENERATION_GUARD_CUSTOM_PARAM_KEY)
            if not isinstance(guard, dict) or guard.get("enabled") is not True:
                continue
            request = params.get("__req__")
            output_ids = getattr(request, "output_ids", None)
            if not isinstance(output_ids, list):
                continue
            try:
                check_interval = int(guard["check_interval_tokens"])
                minimum_span = int(guard["minimum_periodic_span_tokens"])
                maximum_period = int(guard["maximum_period_tokens"])
                minimum_repeats = int(guard["minimum_repeat_count"])
                eos_token_id = int(guard["eos_token_id"])
            except (KeyError, TypeError, ValueError):
                continue
            token_count = len(output_ids)
            if (
                check_interval <= 0
                or minimum_span <= 0
                or maximum_period <= 0
                or minimum_repeats < 2
                or token_count < minimum_span
                or token_count % check_interval != 0
            ):
                continue
            largest_period = min(maximum_period, token_count // minimum_repeats)
            detected = False
            for period in range(1, largest_period + 1):
                required_repeats = max(
                    minimum_repeats,
                    (minimum_span + period - 1) // period,
                )
                required_span = period * required_repeats
                if token_count < required_span:
                    continue
                pattern = output_ids[token_count - period : token_count]
                matched = True
                for repeat_index in range(2, required_repeats + 1):
                    end = token_count - period * (repeat_index - 1)
                    start = end - period
                    if output_ids[start:end] != pattern:
                        matched = False
                        break
                if matched:
                    detected = True
                    break
            if not detected or not 0 <= eos_token_id < logits.shape[-1]:
                continue
            logits[batch_index, :] = -float("inf")
            logits[batch_index, eos_token_id] = 0.0
        return logits

    @classmethod
    def to_str(cls) -> str:
        try:
            import dill
        except ImportError as exc:  # pragma: no cover - serving dependency
            raise RuntimeError(
                "dill is required to serialize the SGLang metadata logit processor"
            ) from exc
        return json.dumps({"callable": dill.dumps(cls).hex()})


def compressor_metadata_logit_processor_string() -> str:
    """Serialize the cross-interpreter-safe metadata/guard processor."""

    return SpecForgeCompositeMetadataLogitProcessor.to_str()


def _read_checkpoint_metadata(checkpoint: str | Path) -> CompressorCheckpointMetadata:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(
            f"context-compressor checkpoint directory does not exist: {checkpoint_path}"
        )
    config_path = checkpoint_path / "config.json"
    state_path = checkpoint_path / "compressor.pt"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config does not exist: {config_path}")
    if not state_path.is_file():
        raise FileNotFoundError(f"checkpoint state does not exist: {state_path}")
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read checkpoint config: {config_path}") from exc
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint config must contain a JSON object")
    serving_contract = resolve_grepseek_checkpoint_serving_contract(raw_config)

    compressor_config = raw_config.get("compressor")
    if not isinstance(compressor_config, dict):
        raise ValueError("checkpoint config is missing compressor metadata")
    if compressor_config.get("architecture") != "GrepSeekDFlashCompressor":
        raise ValueError("checkpoint architecture is not GrepSeekDFlashCompressor")
    if compressor_config.get("slot_padding_side") != "left":
        raise ValueError("checkpoint slot padding must be left")
    qwen36_token_clock = serving_contract.context_organization == QWEN36_TOKEN_CLOCK_V1
    expected_layer_types = ["sliding_attention"] * (4 if qwen36_token_clock else 5) + [
        "full_attention"
    ]
    if compressor_config.get("layer_types") != expected_layer_types:
        raise ValueError("checkpoint compressor layer_types do not match training")
    expected_compressor_layers = 5 if qwen36_token_clock else 6
    if compressor_config.get("num_hidden_layers") != expected_compressor_layers:
        raise ValueError(
            "checkpoint compressor layer count does not match its serving schema"
        )

    raw_layer_ids = compressor_config.get("target_layer_ids")
    if not isinstance(raw_layer_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_layer_ids
    ):
        raise ValueError("checkpoint target_layer_ids must be a list of integers")
    target_layer_ids = tuple(int(value) for value in raw_layer_ids)
    expected_target_layers = 5 if qwen36_token_clock else 8
    if (
        len(target_layer_ids) != expected_target_layers
        or len(set(target_layer_ids)) != expected_target_layers
    ):
        if qwen36_token_clock:
            raise ValueError(
                "Qwen3.6 server hook requires exactly five unique target layers"
            )
        raise ValueError("server hook requires exactly eight unique target layers")
    if any(value < 0 for value in target_layer_ids):
        raise ValueError("checkpoint target layer ids must be nonnegative")

    target_hidden_size = _positive_int(
        compressor_config.get("target_hidden_size"), name="target_hidden_size"
    )
    hidden_size = _positive_int(
        compressor_config.get("hidden_size"), name="compressor.hidden_size"
    )
    expected_hidden_size = 5120 if qwen36_token_clock else 4096
    if target_hidden_size != expected_hidden_size or hidden_size != target_hidden_size:
        if qwen36_token_clock:
            raise ValueError(
                "Qwen3.6 server hook requires matching 5120-wide target/compressor"
            )
        raise ValueError("server hook requires matching 4096-wide target/compressor")
    expected_source_width = len(target_layer_ids) * target_hidden_size
    if compressor_config.get("selected_source_width") != expected_source_width:
        raise ValueError("checkpoint selected_source_width is inconsistent")

    compression_ratio = _positive_int(
        compressor_config.get("compression_ratio"), name="compression_ratio"
    )
    if compression_ratio != serving_contract.compression_ratio:
        raise ValueError("checkpoint serving/compressor compression ratios disagree")
    local_window = _positive_int(
        compressor_config.get("local_window"), name="local_window"
    )
    if local_window < compression_ratio:
        raise ValueError("checkpoint local_window is narrower than compression_ratio")
    output_embedding_rms = compressor_config.get("output_embedding_rms")
    if (
        isinstance(output_embedding_rms, bool)
        or not isinstance(output_embedding_rms, (int, float))
        or not math.isfinite(float(output_embedding_rms))
        or float(output_embedding_rms) <= 0
    ):
        raise ValueError("checkpoint output_embedding_rms must be finite and positive")

    return CompressorCheckpointMetadata(
        checkpoint=checkpoint_path,
        state_path=state_path,
        raw_config=raw_config,
        target_layer_ids=target_layer_ids,
        target_hidden_size=target_hidden_size,
        compression_ratio=compression_ratio,
        local_window=local_window,
        output_embedding_rms=float(output_embedding_rms),
        context_organization=serving_contract.context_organization,
        checkpoint_schema=serving_contract.checkpoint_schema,
    )


def _default_compressor_loader(
    metadata: CompressorCheckpointMetadata,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    # Keep the training/modeling dependency lazy.  In particular, this import
    # does not happen for clients that only construct request parameters.
    from context_compression.grepseek_modeling import (
        GrepSeekDFlashCompressor,
        build_qwen35_9b_dflash_config,
    )

    compressor_config = metadata.raw_config["compressor"]
    if metadata.context_organization == QWEN36_TOKEN_CLOCK_V1:
        compressor = GrepSeekDFlashCompressor.canonical_qwen36_27b(
            output_embedding_rms=metadata.output_embedding_rms,
            gradient_checkpointing=False,
        )
    else:
        draft_config = build_qwen35_9b_dflash_config(
            target_hidden_size=metadata.target_hidden_size,
            compression_ratio=metadata.compression_ratio,
        )
        compressor = GrepSeekDFlashCompressor(
            draft_config,
            target_hidden_size=metadata.target_hidden_size,
            target_layer_ids=metadata.target_layer_ids,
            compression_ratio=metadata.compression_ratio,
            local_window=metadata.local_window,
            output_embedding_rms=metadata.output_embedding_rms,
            gradient_checkpointing=False,
        )
    state = torch.load(metadata.state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("compressor.pt must contain a state-dict mapping")
    compressor.load_state_dict(state, strict=True)
    compressor.to(device=device, dtype=dtype)
    compressor.requires_grad_(False)
    for parameter in compressor.parameters():
        parameter.grad = None
    compressor.eval()
    # This is intentionally checked after construction as protection against
    # future checkpoint fields silently diverging from the modeled geometry.
    exported = compressor.export_config()
    for field in (
        "compression_ratio",
        "local_window",
        "target_hidden_size",
        "target_layer_ids",
    ):
        expected = compressor_config[field]
        if exported[field] != expected:
            raise ValueError(
                f"loaded compressor disagrees with checkpoint field {field}"
            )
    return compressor


def _model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in model.parameters():
        if torch.is_floating_point(parameter):
            return parameter.device, parameter.dtype
    raise TypeError("target model exposes no floating parameter for device/dtype")


def _target_text_config(model: nn.Module) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise TypeError("target model must expose config")
    return getattr(config, "text_config", config)


def _is_capture_full(capture_mode: Any) -> bool:
    predicate = getattr(capture_mode, "is_full", None)
    return bool(predicate()) if callable(predicate) else False


def _is_extend_mode(forward_mode: Any) -> bool:
    predicate = getattr(forward_mode, "is_extend", None)
    return bool(predicate()) if callable(predicate) else False


def _as_single_cpu_length(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim != 1 or value.numel() != 1:
            raise SpecForgeCompressorContractError(f"{name} must contain one row")
        return int(value.item())
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise SpecForgeCompressorContractError(f"{name} must contain one row")
    raw = value[0]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SpecForgeCompressorContractError(f"{name} must contain an integer")
    return int(raw)


class SpecForgeContextCompressorHook:
    """Stateful logits-processor hook for one prefill-only encode request."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        compressor_loader: Optional[CompressorLoader] = None,
    ) -> None:
        if not isinstance(config, Mapping):
            raise TypeError("hook config must be a mapping")
        unknown = set(config) - {
            "checkpoint",
            "generation_degeneration_guard_server_abi",
            "sync_token_ids_across_tp",
            "gpu_memory_handle_policy",
            "gpu_memory_handle_ttl_seconds",
            "gpu_memory_handle_max_entries_per_tp_rank",
            "gpu_memory_handle_max_bytes_per_tp_rank",
        }
        if unknown:
            raise ValueError(
                f"unknown context-compressor hook config: {sorted(unknown)}"
            )
        checkpoint = config.get("checkpoint")
        if not isinstance(checkpoint, (str, Path)) or not str(checkpoint):
            raise ValueError("hook config requires a nonempty checkpoint path")
        guard_server_abi = config.get("generation_degeneration_guard_server_abi")
        if (
            guard_server_abi is not None
            and guard_server_abi != GENERATION_DEGENERATION_GUARD_SERVER_ABI
        ):
            raise ValueError(
                "unsupported generation degeneration guard server ABI: "
                f"expected {GENERATION_DEGENERATION_GUARD_SERVER_ABI!r}"
            )
        self.generation_degeneration_guard_server_abi = guard_server_abi
        sync_token_ids_across_tp = config.get("sync_token_ids_across_tp")
        if (
            sync_token_ids_across_tp is not None
            and sync_token_ids_across_tp is not True
        ):
            raise ValueError("sync_token_ids_across_tp hook attestation must be true")
        if (
            sync_token_ids_across_tp is True
            and os.environ.get("SYNC_TOKEN_IDS_ACROSS_TP") != "1"
        ):
            raise ValueError(
                "sync_token_ids_across_tp requires the real server environment "
                "SYNC_TOKEN_IDS_ACROSS_TP=1"
            )
        self.sync_token_ids_across_tp = sync_token_ids_across_tp
        self.metadata = _read_checkpoint_metadata(checkpoint)
        self.hook_abi_digest = _compute_hook_abi_digest(self.metadata)
        memory_handle_policy = config.get("gpu_memory_handle_policy")
        if memory_handle_policy not in {None, GPU_MEMORY_HANDLE_POLICY}:
            raise ValueError(
                f"unsupported gpu_memory_handle_policy: {memory_handle_policy!r}"
            )
        self.gpu_memory_handle_policy = memory_handle_policy
        raw_memory_handle_ttl = config.get(
            "gpu_memory_handle_ttl_seconds", 900.0
        )
        if isinstance(raw_memory_handle_ttl, bool) or not isinstance(
            raw_memory_handle_ttl, (int, float)
        ):
            raise ValueError("gpu_memory_handle_ttl_seconds must be numeric")
        self.gpu_memory_handle_ttl_seconds = float(raw_memory_handle_ttl)
        self.gpu_memory_handle_max_entries_per_tp_rank = _positive_int(
            config.get("gpu_memory_handle_max_entries_per_tp_rank", 64),
            name="gpu_memory_handle_max_entries_per_tp_rank",
        )
        self.gpu_memory_handle_max_bytes_per_tp_rank = _positive_int(
            config.get("gpu_memory_handle_max_bytes_per_tp_rank", 1 << 30),
            name="gpu_memory_handle_max_bytes_per_tp_rank",
        )
        if (
            not math.isfinite(self.gpu_memory_handle_ttl_seconds)
            or self.gpu_memory_handle_ttl_seconds <= 0
        ):
            raise ValueError("gpu_memory_handle_ttl_seconds must be positive")
        if memory_handle_policy is None and any(
            key in config
            for key in (
                "gpu_memory_handle_ttl_seconds",
                "gpu_memory_handle_max_entries_per_tp_rank",
                "gpu_memory_handle_max_bytes_per_tp_rank",
            )
        ):
            raise ValueError(
                "GPU memory handle limits require gpu_memory_handle_policy opt-in"
            )
        self._compressor_loader = compressor_loader or _default_compressor_loader
        self.compressor: Optional[nn.Module] = None
        self._target_model: Optional[nn.Module] = None
        self._memory_handle_registry: Any = None

    def setup_model(self, model: nn.Module) -> "SpecForgeContextCompressorHook":
        """Validate the target, enable exact layer capture, and load weights."""

        if not isinstance(model, nn.Module):
            raise TypeError("setup_model requires a torch.nn.Module")
        if self._target_model is not None:
            if self._target_model is not model:
                raise RuntimeError("hook is already bound to a different target model")
            return self

        text_config = _target_text_config(model)
        hidden_size = getattr(text_config, "hidden_size", None)
        num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        if hidden_size != self.metadata.target_hidden_size:
            raise ValueError(
                "target hidden size does not match compressor checkpoint: "
                f"{hidden_size!r} != {self.metadata.target_hidden_size}"
            )
        if (
            isinstance(num_hidden_layers, bool)
            or not isinstance(num_hidden_layers, int)
            or num_hidden_layers <= max(self.metadata.target_layer_ids)
        ):
            raise ValueError("target does not contain every requested capture layer")
        capture_setter = getattr(model, "set_dflash_layers_to_capture", None)
        if not callable(capture_setter):
            raise TypeError("target model does not support DFlash layer capture")

        device, dtype = _model_device_and_dtype(model)
        compressor = self._compressor_loader(self.metadata, device, dtype)
        if not isinstance(compressor, nn.Module):
            raise TypeError("compressor loader must return torch.nn.Module")
        compressor.requires_grad_(False)
        compressor.eval()
        memory_handle_registry = None
        if self.gpu_memory_handle_policy is not None:
            if self.sync_token_ids_across_tp is not True:
                raise ValueError(
                    "GPU memory handles require sync_token_ids_across_tp=true"
                )
            if device.type != "cuda":
                raise ValueError("GPU memory handles require a CUDA target model")
            server_epoch = os.environ.get(GPU_MEMORY_HANDLE_SERVER_EPOCH_ENV)
            if not server_epoch:
                raise ValueError(
                    f"GPU memory handles require {GPU_MEMORY_HANDLE_SERVER_EPOCH_ENV}"
                )
            try:
                from sglang.srt.memory_handle_registry import (
                    configure_global_memory_handle_registry,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "SGLang GPU memory handle registry is unavailable"
                ) from exc
            tp_rank = (
                int(torch.distributed.get_rank())
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            memory_handle_registry = configure_global_memory_handle_registry(
                server_epoch=server_epoch,
                tp_rank=tp_rank,
                ttl_seconds=self.gpu_memory_handle_ttl_seconds,
                max_entries=self.gpu_memory_handle_max_entries_per_tp_rank,
                max_bytes=self.gpu_memory_handle_max_bytes_per_tp_rank,
                hook_abi_digest=self.hook_abi_digest,
                require_cuda=True,
            )
        capture_setter(list(self.metadata.target_layer_ids))
        self.compressor = compressor
        self._target_model = model
        self._memory_handle_registry = memory_handle_registry
        return self

    def _request_payload(self, forward_batch: Any) -> Optional[Mapping[str, Any]]:
        sampling_info = getattr(forward_batch, "sampling_info", None)
        custom_params = getattr(sampling_info, "custom_params", None)
        if custom_params is None:
            return None
        if not isinstance(custom_params, (list, tuple)):
            raise SpecForgeCompressorContractError(
                "ForwardBatch custom_params must be a per-request sequence"
            )

        marked: list[tuple[int, Any]] = []
        for index, params in enumerate(custom_params):
            if isinstance(params, Mapping) and CUSTOM_PARAM_KEY in params:
                marked.append((index, params[CUSTOM_PARAM_KEY]))
        if not marked:
            return None
        normalized: list[tuple[int, Mapping[str, Any]]] = []
        for index, payload in marked:
            if not isinstance(payload, Mapping):
                raise SpecForgeCompressorContractError(
                    f"custom_params[{index}][{CUSTOM_PARAM_KEY!r}] must be a mapping"
                )
            version = payload.get("version")
            if version != CUSTOM_PARAM_VERSION:
                raise SpecForgeCompressorContractError(
                    f"unsupported compressor custom-param version: {version!r}"
                )
            mode = payload.get("mode")
            if mode not in {ENCODE_MODE, RECOVER_MODE}:
                raise SpecForgeCompressorContractError(
                    f"unsupported compressor request mode: {mode!r}"
                )
            normalized.append((index, payload))

        if all(payload["mode"] == RECOVER_MODE for _, payload in normalized):
            # Recover only supplies embeddings to the model wrapper; this hook
            # is a no-op and therefore supports request-aligned batching and
            # mixing with ordinary discrete requests.
            for index, payload in normalized:
                payload_fields = set(payload)
                has_embeddings = "memory_embeddings" in payload_fields
                has_handle = "memory_handle" in payload_fields
                expected = {
                    "version",
                    "mode",
                    "memory_positions",
                    "memory_embeddings" if has_embeddings else "memory_handle",
                }
                if has_embeddings == has_handle or payload_fields != expected:
                    raise SpecForgeCompressorContractError(
                        "recover requires exactly one memory_embeddings or memory_handle"
                    )
                if has_handle:
                    try:
                        normalize_memory_handle_descriptor(payload["memory_handle"])
                    except (TypeError, ValueError) as exc:
                        raise SpecForgeCompressorContractError(str(exc)) from exc
            return None

        if len(normalized) != 1:
            raise SpecForgeCompressorContractError(
                "compressor encode cannot share a forward with another request"
            )
        index, payload = normalized[0]
        mode = payload["mode"]
        if mode == RECOVER_MODE:  # pragma: no cover - handled by all-recover
            payload_fields = set(payload)
            has_embeddings = "memory_embeddings" in payload_fields
            has_handle = "memory_handle" in payload_fields
            expected = {
                "version",
                "mode",
                "memory_positions",
                "memory_embeddings" if has_embeddings else "memory_handle",
            }
            if has_embeddings == has_handle or payload_fields != expected:
                raise SpecForgeCompressorContractError(
                    "recover requires exactly one memory_embeddings or memory_handle"
                )
            if has_handle:
                try:
                    normalize_memory_handle_descriptor(payload["memory_handle"])
                except (TypeError, ValueError) as exc:
                    raise SpecForgeCompressorContractError(str(exc)) from exc
            return None

        if index != 0 or len(custom_params) != 1:
            raise SpecForgeCompressorContractError(
                "compressor encode supports batch size one only"
            )
        encode_fields = {
            "version",
            "mode",
            "evidence_spans",
            "source_token_lengths",
        }
        payload_fields = set(payload)
        if "memory_handle_publish" in payload_fields:
            try:
                normalize_memory_handle_publish_claim(
                    payload["memory_handle_publish"]
                )
            except (TypeError, ValueError) as exc:
                raise SpecForgeCompressorContractError(str(exc)) from exc
            encode_fields.add("memory_handle_publish")
        if self.metadata.context_organization == CAUSAL_OBSERVATION_CACHE_V1:
            allowed_encode_fields = encode_fields
        elif self.metadata.context_organization in {
            RECURSIVE_HISTORY_V2,
            RECURSIVE_WORKING_MEMORY_V3,
            QWEN36_TOKEN_CLOCK_V1,
        }:
            has_positions = "memory_positions" in payload_fields
            has_embeddings = "memory_embeddings" in payload_fields
            has_handle = "memory_handle" in payload_fields
            if has_embeddings and has_handle:
                raise SpecForgeCompressorContractError(
                    "recursive encode cannot mix memory_embeddings and memory_handle"
                )
            has_memory_value = has_embeddings or has_handle
            if has_positions != has_memory_value:
                raise SpecForgeCompressorContractError(
                    "recursive encode memory positions and value must be provided together"
                )
            allowed_encode_fields = encode_fields | (
                {
                    "memory_positions",
                    "memory_embeddings" if has_embeddings else "memory_handle",
                }
                if has_positions
                else set()
            )
            if self.metadata.context_organization == RECURSIVE_WORKING_MEMORY_V3:
                if (
                    payload.get("recursive_source_contract")
                    != WORKING_MEMORY_V3_SOURCE_CONTRACT
                ):
                    raise SpecForgeCompressorContractError(
                        "working-memory-v3 encode requires the semantic working-memory "
                        "source contract"
                    )
                allowed_encode_fields.add("recursive_source_contract")
            elif "recursive_source_contract" in payload_fields:
                raise SpecForgeCompressorContractError(
                    "recursive-history-v2 encode cannot claim a working-memory-v3 "
                    "semantic source contract"
                )
        else:  # pragma: no cover - normalized by checkpoint classification
            raise SpecForgeCompressorContractError(
                "checkpoint has an unknown context organization"
            )
        if payload_fields != allowed_encode_fields:
            raise SpecForgeCompressorContractError(
                "encode custom params contain missing or unsupported fields"
            )
        return payload

    @staticmethod
    def _admitted_memory_tensor(
        forward_batch: Any, descriptor: Mapping[str, Any]
    ) -> torch.Tensor:
        """Return the scheduler-resolved strong reference for this request.

        Resolving the external handle again inside model forward would reapply
        process-local TTL and could make one TP rank fail after its peers had
        already entered the compressor.  Scheduler admission instead resolves
        collectively and stores a request-lifetime tensor reference.
        """

        custom_params = getattr(
            getattr(forward_batch, "sampling_info", None),
            "custom_params",
            None,
        )
        if not isinstance(custom_params, (list, tuple)) or len(custom_params) != 1:
            raise SpecForgeCompressorContractError(
                "recursive memory_handle requires one admitted scheduler request"
            )
        params = custom_params[0]
        request = params.get("__req__") if isinstance(params, Mapping) else None
        if request is None:
            raise SpecForgeCompressorContractError(
                "recursive memory_handle lacks scheduler admission state"
            )
        admitted_descriptor = getattr(request, "memory_handle_descriptor", None)
        if admitted_descriptor != dict(descriptor):
            raise SpecForgeCompressorContractError(
                "recursive memory_handle disagrees with scheduler admission"
            )
        tensor = getattr(request, "memory_handle_tensor_ref", None)
        if not isinstance(tensor, torch.Tensor):
            raise SpecForgeCompressorContractError(
                "recursive memory_handle admission did not retain its tensor"
            )
        if getattr(request, "memory_handle_identity", None) != descriptor[
            "memory_identity"
        ]:
            raise SpecForgeCompressorContractError(
                "recursive memory identity disagrees with scheduler admission"
            )
        canonical_dtype = str(tensor.dtype)
        if canonical_dtype.startswith("torch."):
            canonical_dtype = canonical_dtype[6:]
        if (
            tensor.ndim != 2
            or list(tensor.shape) != descriptor["shape"]
            or canonical_dtype != descriptor["dtype"]
            or int(tensor.numel() * tensor.element_size())
            != descriptor["nbytes"]
        ):
            raise SpecForgeCompressorContractError(
                "admitted recursive memory tensor disagrees with its descriptor"
            )
        return tensor

    def _validate_encode_forward(
        self,
        inputs: tuple[Any, ...],
        forward_batch: Any,
        *,
        allow_cpu_aux: bool = False,
    ) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        if self.compressor is None or self._target_model is None:
            raise SpecForgeCompressorContractError(
                "setup_model must complete before an encode request"
            )
        if getattr(forward_batch, "batch_size", None) != 1:
            raise SpecForgeCompressorContractError(
                "compressor encode supports ForwardBatch.batch_size == 1 only"
            )
        if getattr(forward_batch, "is_prefill_only", None) is not True:
            raise SpecForgeCompressorContractError(
                "compressor encode must be a prefill-only request"
            )
        if not _is_extend_mode(getattr(forward_batch, "forward_mode", None)):
            raise SpecForgeCompressorContractError(
                "compressor encode requires SGLang extend/prefill mode"
            )
        if not _is_capture_full(getattr(forward_batch, "capture_hidden_mode", None)):
            raise SpecForgeCompressorContractError(
                "compressor encode requires CaptureHiddenMode.FULL"
            )
        if getattr(forward_batch, "spec_info", None) is not None:
            raise SpecForgeCompressorContractError(
                "compressor encode cannot run inside speculative decoding"
            )
        if getattr(forward_batch, "return_logprob", False):
            raise SpecForgeCompressorContractError(
                "compressor encode must not request token logprobs"
            )
        if getattr(forward_batch, "input_embeds", None) is not None:
            raise SpecForgeCompressorContractError(
                "compressor source encoding requires discrete input_ids"
            )
        padded_static_len = getattr(forward_batch, "padded_static_len", -1)
        if padded_static_len not in {-1, None}:
            raise SpecForgeCompressorContractError(
                "compressor encode does not support static-padded prefill"
            )
        if len(inputs) < 5:
            raise SpecForgeCompressorContractError(
                "logits processor did not receive aux_hidden_states"
            )
        input_ids = inputs[0]
        aux_hidden_states = inputs[4]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            raise SpecForgeCompressorContractError(
                "SGLang packed source input_ids must be rank one"
            )
        token_count = int(input_ids.numel())
        if token_count <= 0:
            raise SpecForgeCompressorContractError("source prefill cannot be empty")
        extend_length = _as_single_cpu_length(
            getattr(forward_batch, "extend_seq_lens_cpu", None),
            name="extend_seq_lens_cpu",
        )
        prefix_length = _as_single_cpu_length(
            getattr(forward_batch, "extend_prefix_lens_cpu", None),
            name="extend_prefix_lens_cpu",
        )
        if extend_length != token_count or prefix_length != 0:
            raise SpecForgeCompressorContractError(
                "encode requires one complete, unchunked, cache-free source prefill"
            )
        if getattr(forward_batch, "extend_num_tokens", token_count) != token_count:
            raise SpecForgeCompressorContractError(
                "extend_num_tokens disagrees with packed source width"
            )
        if not isinstance(aux_hidden_states, (list, tuple)):
            raise SpecForgeCompressorContractError(
                "aux_hidden_states must be an ordered layer sequence"
            )
        if len(aux_hidden_states) != len(self.metadata.target_layer_ids):
            raise SpecForgeCompressorContractError(
                "target returned the wrong number of captured hidden layers"
            )
        reference: Optional[torch.Tensor] = None
        for index, hidden in enumerate(aux_hidden_states):
            if (
                not isinstance(hidden, torch.Tensor)
                or hidden.ndim != 2
                or hidden.shape != (token_count, self.metadata.target_hidden_size)
                or not torch.is_floating_point(hidden)
            ):
                raise SpecForgeCompressorContractError(
                    f"aux_hidden_states[{index}] must have shape "
                    f"[{token_count}, {self.metadata.target_hidden_size}]"
                )
            if reference is None:
                reference = hidden
            elif hidden.device != reference.device or hidden.dtype != reference.dtype:
                raise SpecForgeCompressorContractError(
                    "captured hidden layers must share a device and dtype"
                )
        assert reference is not None
        first_parameter = next(self.compressor.parameters(), None)
        if first_parameter is None:
            raise SpecForgeCompressorContractError("compressor has no parameters")
        if first_parameter.device != reference.device and not (
            allow_cpu_aux and reference.device.type == "cpu"
        ):
            raise SpecForgeCompressorContractError(
                "compressor and target hidden states are on different devices"
            )
        return input_ids, aux_hidden_states

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> Any:
        del module
        if not isinstance(inputs, tuple) or len(inputs) < 4:
            raise SpecForgeCompressorContractError(
                "hook must be attached to the SGLang logits processor"
            )
        forward_batch = inputs[3]
        payload = self._request_payload(forward_batch)
        if payload is None or payload["mode"] == RECOVER_MODE:
            return output

        input_ids, aux_hidden_states = self._validate_encode_forward(
            inputs, forward_batch
        )
        try:
            spans, source_lengths = _normalize_evidence_contract(
                payload["evidence_spans"], payload["source_token_lengths"]
            )
        except (TypeError, ValueError) as exc:
            raise SpecForgeCompressorContractError(str(exc)) from exc
        token_count = int(aux_hidden_states[0].shape[0])
        if spans[-1][1] > token_count:
            raise SpecForgeCompressorContractError(
                "evidence span lies outside captured source hidden states"
            )
        if self.metadata.context_organization in {
            RECURSIVE_HISTORY_V2,
            RECURSIVE_WORKING_MEMORY_V3,
            QWEN36_TOKEN_CLOCK_V1,
        }:
            if len(spans) != 1:
                raise SpecForgeCompressorContractError(
                    "recursive encode requires exactly one physical source span"
                )
            start, end = spans[0]
            if start <= 0 or end != token_count:
                raise SpecForgeCompressorContractError(
                    "recursive encode span must cover every row after a nonempty raw prefix"
                )
            if source_lengths != [end - start]:
                raise SpecForgeCompressorContractError(
                    "recursive source length must equal all physical post-prefix rows"
                )
            if "memory_positions" in payload:
                if "memory_embeddings" in payload:
                    try:
                        positions, embeddings = _normalize_memory_contract(
                            memory_positions=payload["memory_positions"],
                            memory_embeddings=payload["memory_embeddings"],
                            hidden_size=self.metadata.target_hidden_size,
                        )
                    except (TypeError, ValueError) as exc:
                        raise SpecForgeCompressorContractError(str(exc)) from exc
                else:
                    if self._memory_handle_registry is None:
                        raise SpecForgeCompressorContractError(
                            "recursive memory_handle requires server registry opt-in"
                        )
                    try:
                        descriptor = normalize_memory_handle_descriptor(
                            payload["memory_handle"]
                        )
                    except (RuntimeError, TypeError, ValueError) as exc:
                        raise SpecForgeCompressorContractError(str(exc)) from exc
                    positions = [int(value) for value in payload["memory_positions"]]
                    embeddings = self._admitted_memory_tensor(
                        forward_batch, descriptor
                    )
                if positions != list(range(start, start + len(positions))):
                    raise SpecForgeCompressorContractError(
                        "recursive previous memory must be contiguous at the source-span start"
                    )
                if start + len(embeddings) >= end:
                    raise SpecForgeCompressorContractError(
                        "recursive encode requires a nonempty incremental discrete suffix"
                    )

        segment_count = len(spans)
        max_span_length = max(end - start for start, end in spans)
        feature_width = len(aux_hidden_states) * self.metadata.target_hidden_size
        reference = aux_hidden_states[0]
        padded_features = reference.new_zeros(
            (segment_count, max_span_length, feature_width)
        )
        source_attention_mask = torch.zeros(
            (segment_count, max_span_length),
            dtype=torch.bool,
            device=reference.device,
        )
        for segment_index, (start, end) in enumerate(spans):
            span_features = torch.cat(
                [hidden[start:end] for hidden in aux_hidden_states], dim=-1
            )
            width = end - start
            padded_features[segment_index, :width].copy_(span_features)
            source_attention_mask[segment_index, :width] = True
        canonical_lengths = torch.tensor(
            source_lengths, dtype=torch.long, device=reference.device
        )

        assert self.compressor is not None
        with torch.inference_mode():
            compressed = self.compressor(
                padded_features,
                source_attention_mask,
                source_token_lengths=canonical_lengths,
            )
        embeddings = getattr(compressed, "embeddings", None)
        attention_mask = getattr(compressed, "attention_mask", None)
        slot_counts = getattr(compressed, "slot_counts", None)
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.ndim != 3
            or embeddings.shape[0] != segment_count
            or embeddings.shape[-1] != self.metadata.target_hidden_size
            or not torch.is_floating_point(embeddings)
        ):
            raise SpecForgeCompressorContractError(
                "compressor returned invalid memory embeddings"
            )
        if (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != embeddings.shape[:2]
        ):
            raise SpecForgeCompressorContractError(
                "compressor returned an invalid memory attention mask"
            )
        if (
            not isinstance(slot_counts, torch.Tensor)
            or slot_counts.ndim != 1
            or slot_counts.shape[0] != segment_count
        ):
            raise SpecForgeCompressorContractError(
                "compressor returned invalid slot counts"
            )
        expected_counts = torch.tensor(
            [
                (length + self.metadata.compression_ratio - 1)
                // self.metadata.compression_ratio
                for length in source_lengths
            ],
            dtype=torch.long,
            device=slot_counts.device,
        )
        if not torch.equal(slot_counts.long(), expected_counts):
            raise SpecForgeCompressorContractError(
                "compressor slot counts violate the checkpoint compression ratio"
            )
        memory_mask = attention_mask.bool()
        if not torch.equal(memory_mask.sum(dim=-1).to(slot_counts.device), slot_counts):
            raise SpecForgeCompressorContractError(
                "compressor memory mask disagrees with slot counts"
            )
        valid_memory = embeddings[memory_mask]
        if valid_memory.shape != (
            int(expected_counts.sum().item()),
            self.metadata.target_hidden_size,
        ):
            raise SpecForgeCompressorContractError(
                "packed memory shape disagrees with valid slot count"
            )
        if not bool(torch.isfinite(valid_memory).all().item()):
            raise SpecForgeCompressorContractError(
                "compressor returned non-finite memory embeddings"
            )
        if not hasattr(output, "hidden_states"):
            raise SpecForgeCompressorContractError(
                "hook output is not an SGLang LogitsProcessorOutput"
            )
        publish_claim_raw = payload.get("memory_handle_publish")
        if publish_claim_raw is not None:
            if self._memory_handle_registry is None:
                raise SpecForgeCompressorContractError(
                    "memory_handle_publish requires explicit server registry opt-in"
                )
            if "memory_embeddings" in payload:
                raise SpecForgeCompressorContractError(
                    "a published recursive transition must reference its previous "
                    "server memory by handle"
                )
            descriptor: Optional[dict[str, Any]] = None
            local_publish_error: Optional[str] = None
            try:
                publish_claim = normalize_memory_handle_publish_claim(
                    publish_claim_raw
                )
                actual_input_ids = [
                    int(value)
                    for value in input_ids.detach().to(device="cpu").tolist()
                ]
                request_digest = compute_memory_handle_request_digest(
                    input_ids=actual_input_ids,
                    evidence_spans=spans,
                    source_token_lengths=source_lengths,
                    memory_positions=payload.get("memory_positions"),
                    previous_memory_handle=payload.get("memory_handle"),
                    recursive_source_contract=payload.get(
                        "recursive_source_contract"
                    ),
                )
                server_epoch = self._memory_handle_registry.server_epoch
                memory_identity = compute_memory_handle_identity(
                    server_epoch=server_epoch,
                    request_digest=request_digest,
                    hook_abi_digest=self.hook_abi_digest,
                )
                if publish_claim["request_digest"] != request_digest:
                    raise ValueError(
                        "memory_handle_publish request_digest does not match "
                        "server-observed transition"
                    )
                if publish_claim["memory_identity"] != memory_identity:
                    raise ValueError(
                        "memory_handle_publish memory_identity does not match "
                        "server-authoritative provenance"
                    )
                descriptor = self._memory_handle_registry.publish(
                    op_id=publish_claim["op_id"],
                    memory_identity=memory_identity,
                    request_digest=request_digest,
                    tensor=valid_memory,
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                # Every TP rank must reach the coordination collective even if
                # its local quota, claim validation, or publish failed.  Raising
                # here would strand successful peers inside all_gather_object.
                local_publish_error = f"{type(exc).__name__}: {exc}"

            local_outcome = {
                "success": local_publish_error is None,
                "descriptor": descriptor,
                "message": local_publish_error or "",
            }
            gathered_outcomes = [local_outcome]
            distributed_publish = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            try:
                if distributed_publish:
                    gathered_outcomes = [
                        None
                    ] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(
                        gathered_outcomes, local_outcome
                    )
            except RuntimeError as exc:
                gathered_outcomes = []
                local_publish_error = (
                    "TP memory-handle publish coordination failed: " + str(exc)
                )

            coordination_error: Optional[str] = local_publish_error
            if coordination_error is None:
                if not all(
                    isinstance(candidate, Mapping)
                    and candidate.get("success") is True
                    and candidate.get("descriptor") is not None
                    for candidate in gathered_outcomes
                ):
                    coordination_error = (
                        "TP memory-handle publish failed atomically: "
                        f"{gathered_outcomes!r}"
                    )
                elif any(
                    candidate["descriptor"] != descriptor
                    for candidate in gathered_outcomes
                ):
                    coordination_error = (
                        "TP ranks published divergent memory descriptors"
                    )

            if coordination_error is not None:
                rollback_error = None
                if descriptor is not None:
                    # A rollback operation is intentionally unique.  Reusing a
                    # deterministic release op_id after a later republish would
                    # replay the old tombstone instead of deleting the new entry.
                    rollback_op_id = (
                        "rollback."
                        + descriptor["descriptor_digest"][7:]
                        + "."
                        + uuid.uuid4().hex
                    )
                    try:
                        self._memory_handle_registry.release(
                            op_id=rollback_op_id, handle=descriptor
                        )
                    except (RuntimeError, TypeError, ValueError) as exc:
                        rollback_error = str(exc)
                if rollback_error is not None:
                    coordination_error += f"; local rollback failed: {rollback_error}"
                raise SpecForgeCompressorContractError(coordination_error)
            assert descriptor is not None
            metadata_error: Optional[str] = None
            try:
                customized_info = getattr(output, "customized_info", None)
                if customized_info is None:
                    customized_info = {}
                elif not isinstance(customized_info, dict):
                    raise TypeError(
                        "SGLang logits customized_info must be a dictionary"
                    )
                else:
                    customized_info = dict(customized_info)
                if GPU_MEMORY_HANDLE_META_KEY in customized_info:
                    raise ValueError(
                        "memory handle customized-info key is already occupied"
                    )
                customized_info[GPU_MEMORY_HANDLE_META_KEY] = [descriptor]
                output.customized_info = customized_info
                output.hidden_states = None
            except Exception as exc:
                metadata_error = f"{type(exc).__name__}: {exc}"

            metadata_outcome = {
                "success": metadata_error is None,
                "message": metadata_error or "",
            }
            metadata_outcomes = [metadata_outcome]
            try:
                if distributed_publish:
                    metadata_outcomes = [
                        None
                    ] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(
                        metadata_outcomes, metadata_outcome
                    )
            except Exception as exc:
                metadata_error = (
                    "TP memory-handle metadata coordination failed: " + str(exc)
                )
                metadata_outcomes = []
            if metadata_error is None and not all(
                isinstance(candidate, Mapping)
                and candidate.get("success") is True
                for candidate in metadata_outcomes
            ):
                metadata_error = (
                    "TP memory-handle metadata publication failed atomically: "
                    f"{metadata_outcomes!r}"
                )
            if metadata_error is not None:
                rollback_error = None
                rollback_op_id = (
                    "rollback."
                    + descriptor["descriptor_digest"][7:]
                    + "."
                    + uuid.uuid4().hex
                )
                try:
                    self._memory_handle_registry.release(
                        op_id=rollback_op_id, handle=descriptor
                    )
                except Exception as exc:
                    rollback_error = str(exc)
                if rollback_error is not None:
                    metadata_error += f"; local rollback failed: {rollback_error}"
                raise SpecForgeCompressorContractError(metadata_error)
        else:
            try:
                output.hidden_states = valid_memory.detach()
            except (AttributeError, TypeError) as exc:
                raise SpecForgeCompressorContractError(
                    "SGLang logits output does not allow hidden-state replacement"
                ) from exc
        return output


def make_sglang_context_compressor_hook(
    config: Mapping[str, Any],
) -> SpecForgeContextCompressorHook:
    """Forward-hook factory referenced by SGLang ``--forward-hooks`` JSON."""

    return SpecForgeContextCompressorHook(config)


def make_r2eact_compressor_hook(
    config: Mapping[str, Any],
) -> SpecForgeContextCompressorHook:
    """Canonical r^2eAct alias for the SGLang context-compressor factory.

    The callable intentionally delegates to the established implementation so
    checkpoint metadata, state-dict keys, and request ABIs remain unchanged.
    """

    return make_sglang_context_compressor_hook(config)


def make_grepseek_compressor_hook(
    config: Mapping[str, Any],
) -> SpecForgeContextCompressorHook:
    """Backward-compatible alias for existing launcher configurations."""

    return make_sglang_context_compressor_hook(config)


R2EACT_HOOK_FACTORY = (
    "context_compression.sglang_compressor_hook:make_r2eact_compressor_hook"
)
CANONICAL_HOOK_FACTORY = R2EACT_HOOK_FACTORY
LEGACY_GREPSEEK_HOOK_FACTORY = (
    "context_compression.sglang_compressor_hook:make_grepseek_compressor_hook"
)


__all__ = [
    "CUSTOM_PARAM_KEY",
    "CUSTOM_PARAM_VERSION",
    "CANONICAL_HOOK_FACTORY",
    "CompressorCheckpointMetadata",
    "ENCODE_MODE",
    "GENERATION_DEGENERATION_GUARD_CUSTOM_PARAM_KEY",
    "GENERATION_DEGENERATION_GUARD_SERVER_ABI",
    "GPU_MEMORY_HANDLE_CAPABILITY",
    "GPU_MEMORY_HANDLE_META_KEY",
    "GPU_MEMORY_HANDLE_POLICY",
    "GPU_MEMORY_HANDLE_PUBLISH_SCHEMA",
    "GPU_MEMORY_HANDLE_SCHEMA",
    "GPU_MEMORY_HANDLE_SERVER_INFO_SCHEMA",
    "GPU_MEMORY_IDENTITY_KIND",
    "GPU_MEMORY_IDENTITY_SCHEMA",
    "GPU_MEMORY_TRANSITION_REQUEST_SCHEMA",
    "LEGACY_GREPSEEK_HOOK_FACTORY",
    "RECOVER_MODE",
    "R2EACT_HOOK_FACTORY",
    "SpecForgeCompressorContractError",
    "SpecForgeCompositeMetadataLogitProcessor",
    "SpecForgeContextCompressorHook",
    "SpecForgeContextCompressorNoOpLogitProcessor",
    "build_encode_custom_params",
    "build_recursive_encode_custom_params",
    "build_recover_custom_params",
    "compute_memory_handle_identity",
    "compute_memory_handle_request_digest",
    "compressor_metadata_logit_processor_string",
    "make_grepseek_compressor_hook",
    "make_r2eact_compressor_hook",
    "make_sglang_context_compressor_hook",
    "noop_custom_logit_processor_str",
    "normalize_memory_handle_descriptor",
    "normalize_memory_handle_publish_claim",
]
