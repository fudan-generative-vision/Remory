"""Pretrain the Qwen3.8 compressor on MMFineReasoning text repetition.

Each row contributes every non-empty deterministic repeat event among
``question``, ``original_answer``, and ``caption``.  A question containing only
the removed ``<image>`` marker is empty; only that field slot is excluded while
the row's other fields remain.  Candidate-field order rotates across epochs.
The raw text is compressed and the frozen Qwen3.8 reader reconstructs it with
position-weighted hard-label cross entropy.  No VQA response and no teacher KL
are part of this objective.  The assistant prefix is the unscored template-
equivalent ``<think>\n\n</think>\n\n`` block.

Long sources are split into consecutive, exact tokenizer-ID chunks of at most
16 blocks without a decode/re-tokenize round trip.  Long targets rotate over
exact token windows within their source chunk.  Source slot capacity is padded
in 1024-token blocks, so every block produces exactly 64 soft tokens at
compression ratio 16.

Only the compressor is trainable.  The full multimodal Qwen3.8 target is TP2;
the compressor is FSDP FULL_SHARD over the same two ranks.  AdamW moments are
kept on CPU between optimizer boundaries to leave room for multimodal forwards.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.utils.tensorboard import SummaryWriter

from context_compression.grepseek_modeling import GrepSeekDFlashCompressor
from context_compression.losses import frozen_lm_head_chunked_token_ce
from context_compression.mmfine_reasoning_data import (
    REPEAT_FIELDS,
    REPEAT_MANIFEST_SCHEMA,
    ReadyShardQueue,
    ShardReceipt,
    iter_shard_repeat_events,
    load_validated_shard_manifest,
    verify_shard_receipt,
)
from context_compression.qwen38_multimodal_target import (
    FrozenQwen38MultimodalTarget,
    RenderedQwen38Target,
    load_qwen38_multimodal_model,
    load_qwen38_processor,
    render_qwen38_repeat_target,
)

FORMAT_VERSION = 4
CHECKPOINT_SCHEMA = "qwen38_mmfine_repeat_ce_tp2_fsdp_v4"
OBJECTIVE_SCHEMA = "qwen38_mmfine_repeat_position_weighted_ce_v2"
LEGACY_FORMAT_VERSION = 3
LEGACY_CHECKPOINT_SCHEMA = "qwen38_mmfine_repeat_ce_tp2_fsdp_v3"
LEGACY_OBJECTIVE_SCHEMA = "qwen38_mmfine_repeat_position_weighted_ce_v1"
LEGACY_OVERFLOW_MIGRATION_SCHEMA = "mmfine_repeat_v3_to_token_chunked_nonempty_v4_v2"
REPEAT_CHUNK_PLAN_SCHEMA = "mmfine_repeat_exact_token_chunk_plan_v1"
ADAMW_STATE_KEYS = frozenset({"step", "exp_avg", "exp_avg_sq"})
TOKENIZER_CONTRACT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)


@dataclass(frozen=True)
class TargetTokenWindow:
    """A deterministic scored suffix and bounded unscored causal history."""

    full_token_count: int
    window_index: int
    window_count: int
    start: int
    stop: int
    prefix_start: int
    prefix_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class RepeatTokenChunk:
    """One exact, decode-free slice of an original repeat event."""

    event_id: str
    chunk_index: int
    chunk_count: int
    token_start: int
    token_stop: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class QueueCursor:
    epoch: int
    shard_index: int
    event_index: int
    chunk_index: int


@dataclass
class ObjectiveMetrics:
    loss: float
    ce: float
    token_accuracy: float
    target_tokens: int
    target_offset: int
    source_tokens: int
    padded_source_tokens: int
    soft_tokens: int


@dataclass
class UpdateAccumulator:
    samples: int = 0
    repeat_loss: float = 0.0
    repeat_ce: float = 0.0
    repeat_token_accuracy: float = 0.0
    repeat_tokens: int = 0
    source_tokens: int = 0
    soft_tokens: int = 0

    def add(self, repeat: ObjectiveMetrics) -> None:
        self.samples += 1
        self.repeat_loss += repeat.loss
        self.repeat_ce += repeat.ce
        self.repeat_token_accuracy += repeat.token_accuracy
        self.repeat_tokens += repeat.target_tokens
        self.source_tokens += repeat.source_tokens
        self.soft_tokens += repeat.soft_tokens

    def means(self) -> dict[str, float | int]:
        if self.samples <= 0:
            raise ValueError("cannot summarize an empty update")
        inverse = 1.0 / self.samples
        return {
            "samples": self.samples,
            "repeat_loss": self.repeat_loss * inverse,
            "repeat_ce": self.repeat_ce * inverse,
            "repeat_token_accuracy": self.repeat_token_accuracy * inverse,
            "repeat_tokens": self.repeat_tokens,
            "source_tokens": self.source_tokens,
            "soft_tokens": self.soft_tokens,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokenizer_files_receipt(target_model: Path) -> dict[str, Any]:
    """Content-bind every local file that can affect repeat token identities."""

    root = target_model.expanduser().resolve()
    files: dict[str, dict[str, Any]] = {}
    for name in TOKENIZER_CONTRACT_FILES:
        path = root / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"tokenizer contract path is not a regular file: {path}")
        files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if "tokenizer_config.json" not in files or not (
        {"tokenizer.json"} <= set(files) or {"vocab.json", "merges.txt"} <= set(files)
    ):
        raise ValueError("target model has no complete local tokenizer contract")
    return {
        "schema": "local_tokenizer_files_sha256_v1",
        "files": files,
        "files_sha256": _canonical_json_sha256(files),
    }


def _write_tensorboard_summary(
    writer: Any,
    summary: Mapping[str, Any],
    *,
    global_step: int,
) -> str | None:
    """Write diagnostic scalars without making telemetry training-critical."""

    try:
        for key, value in summary.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(f"train/{key}", value, global_step)
        writer.flush()
    except Exception as error:  # noqa: BLE001 - third-party async writer errors vary.
        message = f"{type(error).__name__}: {error}"
        try:
            writer.close()
        except Exception as close_error:  # noqa: BLE001
            message += f"; close={type(close_error).__name__}: {close_error}"
        return message
    return None


def _atomic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def deterministic_token_window_from_ids(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    event_id: str,
    epoch: int,
    window_tokens: int,
    context_tokens: int,
) -> TargetTokenWindow:
    """Select an epoch-rotated window without decoding or retokenizing IDs."""

    for name, value, minimum in (
        ("epoch", epoch, 0),
        ("window_tokens", window_tokens, 1),
        ("context_tokens", context_tokens, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError("target token IDs must be one-dimensional")
        token_ids = token_ids.detach().cpu().tolist()
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError("target token IDs must be a sequence")
    token_ids = list(token_ids)
    if not token_ids:
        raise ValueError("target token IDs must not be empty")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise ValueError("target token sequence contains an invalid token id")
    window_count = math.ceil(len(token_ids) / window_tokens)
    seed = int(hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:16], 16)
    window_index = (seed + epoch) % window_count
    start = window_index * window_tokens
    stop = min(start + window_tokens, len(token_ids))
    prefix_start = max(0, start - context_tokens)
    return TargetTokenWindow(
        full_token_count=len(token_ids),
        window_index=window_index,
        window_count=window_count,
        start=start,
        stop=stop,
        prefix_start=prefix_start,
        prefix_ids=tuple(token_ids[prefix_start:start]),
        target_ids=tuple(token_ids[start:stop]),
    )


def deterministic_token_window(
    tokenizer: Any,
    text: str,
    *,
    event_id: str,
    epoch: int,
    window_tokens: int,
    context_tokens: int,
) -> TargetTokenWindow:
    """Tokenize once, then select an epoch-rotated exact target window."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("window text must be non-empty")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("target text tokenized to an empty or invalid sequence")
    return deterministic_token_window_from_ids(
        token_ids,
        event_id=event_id,
        epoch=epoch,
        window_tokens=window_tokens,
        context_tokens=context_tokens,
    )


def deterministic_repeat_token_chunks(
    tokenizer: Any,
    text: str,
    *,
    event_id: str,
    maximum_source_tokens: int,
) -> tuple[RepeatTokenChunk, ...]:
    """Split a repeat event into exact token-ID chunks with stable identities."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("repeat text must be non-empty")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return repeat_token_chunks_from_ids(
        token_ids,
        event_id=event_id,
        maximum_source_tokens=maximum_source_tokens,
    )


def repeat_token_chunks_from_ids(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    event_id: str,
    maximum_source_tokens: int,
) -> tuple[RepeatTokenChunk, ...]:
    """Split already-tokenized IDs without decode, encode, or token loss."""

    if (
        isinstance(maximum_source_tokens, bool)
        or not isinstance(maximum_source_tokens, int)
        or maximum_source_tokens <= 0
    ):
        raise ValueError("maximum_source_tokens must be a positive integer")
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim != 1:
            raise ValueError("repeat token IDs must be one-dimensional")
        token_ids = token_ids.detach().cpu().tolist()
    if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
        raise TypeError("repeat token IDs must be a sequence")
    token_ids = list(token_ids)
    if not token_ids:
        raise ValueError("repeat text tokenized to an empty or invalid sequence")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ):
        raise ValueError("repeat tokenizer returned an invalid token id")
    chunk_count = math.ceil(len(token_ids) / maximum_source_tokens)
    chunks: list[RepeatTokenChunk] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * maximum_source_tokens
        stop = min(start + maximum_source_tokens, len(token_ids))
        chunk_event_id = (
            event_id
            if chunk_count == 1
            else f"{event_id}:chunk:{chunk_index + 1:06d}-of-{chunk_count:06d}"
        )
        chunks.append(
            RepeatTokenChunk(
                event_id=chunk_event_id,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                token_start=start,
                token_stop=stop,
                token_ids=tuple(token_ids[start:stop]),
            )
        )
    if tuple(token for chunk in chunks for token in chunk.token_ids) != tuple(
        token_ids
    ):
        raise RuntimeError("repeat token chunking did not preserve the exact sequence")
    return tuple(chunks)


def _batch_repeat_token_ids(
    tokenizer: Any,
    events: Sequence[Any],
) -> tuple[tuple[int, ...], ...]:
    """Batch-tokenize preflight events with an exact single-encode cross-check."""

    if not events:
        return ()
    texts = [event.text for event in events]
    if callable(tokenizer):
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        token_batches = (
            encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        )
    else:
        token_batches = [
            tokenizer.encode(text, add_special_tokens=False) for text in texts
        ]
    if not isinstance(token_batches, list) or len(token_batches) != len(events):
        raise ValueError("batch tokenizer returned an invalid repeat plan")
    result = tuple(tuple(ids) for ids in token_batches)
    for index in {0, len(texts) - 1}:
        if result[index] != tuple(
            tokenizer.encode(texts[index], add_special_tokens=False)
        ):
            raise ValueError("batch and single repeat tokenization differ")
    return result


def _event_batches(values: Iterator[Any], batch_size: int = 512) -> Iterator[list[Any]]:
    """Bounded batching helper for the exhaustive CPU tokenizer preflight."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def bounded_causal_position_weights(
    length: int,
    *,
    absolute_start: int,
    alpha: float,
    tau: float,
    power: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return early-token emphasis with a non-decaying unit floor.

    Unlike a raw DFlash-style power decay, late tokens never approach zero.
    Normalizing each scored window to mean one preserves objective scale.
    """

    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    if (
        isinstance(absolute_start, bool)
        or not isinstance(absolute_start, int)
        or absolute_start < 0
    ):
        raise ValueError("absolute_start must be a nonnegative integer")
    for name, value, positive in (
        ("alpha", alpha, False),
        ("tau", tau, True),
        ("power", power, True),
    ):
        if not math.isfinite(float(value)) or (
            float(value) <= 0 if positive else float(value) < 0
        ):
            raise ValueError(f"{name} has an invalid value")
    positions = torch.arange(
        absolute_start,
        absolute_start + length,
        dtype=torch.float32,
        device=device,
    )
    weights = 1.0 + float(alpha) * torch.pow(
        1.0 + positions / float(tau), -float(power)
    )
    return weights / weights.mean()


def padded_slot_token_length(
    real_tokens: int, *, block_tokens: int, maximum_blocks: int
) -> int:
    if (
        isinstance(real_tokens, bool)
        or not isinstance(real_tokens, int)
        or real_tokens <= 0
    ):
        raise ValueError("real_tokens must be positive")
    if block_tokens <= 0 or maximum_blocks <= 0:
        raise ValueError("block_tokens and maximum_blocks must be positive")
    blocks = math.ceil(real_tokens / block_tokens)
    if blocks > maximum_blocks:
        raise ValueError(
            f"source requires {blocks} blocks, above maximum {maximum_blocks}"
        )
    return blocks * block_tokens


@dataclass
class LowPeakAdamMomentResidency:
    """Keep AdamW exp_avg/exp_avg_sq on CPU between optimizer uses."""

    restore_calls: int = 0
    offload_calls: int = 0
    forward_checks: int = 0
    cache_trim_calls: int = 0
    cpu_moment_tensors: int = 0
    cpu_moment_bytes: int = 0
    maximum_forward_cuda_moment_bytes: int = 0
    offload_reasons: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _items(
        optimizer: torch.optim.Optimizer, *, tolerate_partial: bool = False
    ) -> list[tuple[torch.Tensor, dict[str, Any]]]:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise RuntimeError("low-peak moment residency requires AdamW")
        result: list[tuple[torch.Tensor, dict[str, Any]]] = []
        for group in optimizer.param_groups:
            parameters = group.get("params")
            if not isinstance(parameters, list):
                raise RuntimeError("AdamW parameter group is malformed")
            for parameter in parameters:
                if not isinstance(parameter, torch.Tensor):
                    raise RuntimeError("AdamW parameter is not a tensor")
                state = optimizer.state.get(parameter)
                if not state:
                    continue
                if not isinstance(state, dict):
                    raise RuntimeError("AdamW state is not a dictionary")
                keys = set(state)
                if (not tolerate_partial and keys != ADAMW_STATE_KEYS) or (
                    tolerate_partial and not keys <= ADAMW_STATE_KEYS
                ):
                    raise RuntimeError(f"unexpected AdamW state keys: {sorted(keys)}")
                for name in ("exp_avg", "exp_avg_sq"):
                    moment = state.get(name)
                    if moment is None and tolerate_partial:
                        continue
                    if not isinstance(moment, torch.Tensor):
                        raise RuntimeError(f"AdamW {name} is not a tensor")
                    if (
                        moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype
                    ):
                        raise RuntimeError(f"AdamW {name} does not match its parameter")
                result.append((parameter, state))
        return result

    def restore(self, optimizer: torch.optim.Optimizer) -> None:
        for parameter, state in self._items(optimizer):
            for name in ("exp_avg", "exp_avg_sq"):
                moment = state[name]
                if parameter.device.type == "cpu":
                    if moment.device.type != "cpu":
                        raise RuntimeError("CPU parameter has a non-CPU AdamW moment")
                elif moment.device.type == "cpu":
                    state[name] = moment.to(parameter.device, non_blocking=False)
                elif moment.device != parameter.device:
                    raise RuntimeError("AdamW moment has an unexpected device")
        self.restore_calls += 1

    def offload(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        reason: str,
        tolerate_partial: bool = False,
    ) -> None:
        cpu_tensors = 0
        cpu_bytes = 0
        for parameter, state in self._items(
            optimizer, tolerate_partial=tolerate_partial
        ):
            for name in ("exp_avg", "exp_avg_sq"):
                if name not in state:
                    continue
                moment = state[name]
                if moment.device.type != "cpu":
                    if (
                        parameter.device.type == "cpu"
                        or moment.device != parameter.device
                    ):
                        raise RuntimeError("AdamW moment has an unexpected device")
                    state[name] = moment.to("cpu", non_blocking=False)
                    moment = state[name]
                cpu_tensors += 1
                cpu_bytes += moment.numel() * moment.element_size()
        self.offload_calls += 1
        self.offload_reasons[reason] = self.offload_reasons.get(reason, 0) + 1
        self.cpu_moment_tensors = cpu_tensors
        self.cpu_moment_bytes = cpu_bytes

    def prepare_forward(self, optimizer: torch.optim.Optimizer) -> None:
        cuda_bytes = 0
        for _parameter, state in self._items(optimizer):
            for name in ("exp_avg", "exp_avg_sq"):
                moment = state[name]
                if moment.device.type != "cpu":
                    cuda_bytes += moment.numel() * moment.element_size()
        self.maximum_forward_cuda_moment_bytes = max(
            self.maximum_forward_cuda_moment_bytes, cuda_bytes
        )
        if cuda_bytes:
            raise RuntimeError("forward entered with CUDA-resident AdamW moments")
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        self.forward_checks += 1
        self.cache_trim_calls += 1

    def receipt(self) -> dict[str, Any]:
        return {
            "policy": "cpu_between_steps_restore_after_activation_release_v1",
            "moment_keys": ["exp_avg", "exp_avg_sq"],
            "restore_calls": self.restore_calls,
            "offload_calls": self.offload_calls,
            "offload_reasons": dict(sorted(self.offload_reasons.items())),
            "forward_checks": self.forward_checks,
            "cache_trim_calls": self.cache_trim_calls,
            "cpu_moment_tensors": self.cpu_moment_tensors,
            "cpu_moment_bytes": self.cpu_moment_bytes,
            "maximum_forward_cuda_moment_bytes": self.maximum_forward_cuda_moment_bytes,
        }


def _compress_rendered(
    rendered: RenderedQwen38Target,
    raw_features: torch.Tensor,
    *,
    compressor: FSDP,
    block_tokens: int,
    maximum_blocks: int,
    compression_ratio: int,
) -> tuple[torch.Tensor, int, int]:
    real_tokens = rendered.source_span.width
    padded_tokens = padded_slot_token_length(
        real_tokens,
        block_tokens=block_tokens,
        maximum_blocks=maximum_blocks,
    )
    mask = torch.ones(
        raw_features.shape[:2], dtype=torch.bool, device=raw_features.device
    )
    compressed = compressor(
        raw_features,
        mask,
        source_token_lengths=torch.tensor(
            [real_tokens], dtype=torch.long, device=raw_features.device
        ),
        slot_token_lengths=torch.tensor(
            [padded_tokens], dtype=torch.long, device=raw_features.device
        ),
    )
    memory = compressed.embeddings[compressed.attention_mask][None]
    expected_slots = padded_tokens // compression_ratio
    if memory.shape[1] != expected_slots:
        raise RuntimeError(
            f"compressor produced {memory.shape[1]} slots, expected {expected_slots}"
        )
    return memory, real_tokens, padded_tokens


def _forward_objective(
    rendered: RenderedQwen38Target,
    window: TargetTokenWindow,
    *,
    target: FrozenQwen38MultimodalTarget,
    compressor: FSDP,
    block_tokens: int,
    maximum_blocks: int,
    compression_ratio: int,
    position_alpha: float,
    position_tau: float,
    position_power: float,
    ce_chunk_size: int,
) -> tuple[torch.Tensor, ObjectiveMetrics]:
    raw_features = target.raw_source_features(rendered)
    memory, source_tokens, padded_source_tokens = _compress_rendered(
        rendered,
        raw_features,
        compressor=compressor,
        block_tokens=block_tokens,
        maximum_blocks=maximum_blocks,
        compression_ratio=compression_ratio,
    )
    del raw_features
    student = target.student_from_memory(rendered, memory, grad=True)
    weights = bounded_causal_position_weights(
        student.target_ids.shape[1],
        absolute_start=window.start,
        alpha=position_alpha,
        tau=position_tau,
        power=position_power,
        device=student.target_ids.device,
    )[None]
    ce = frozen_lm_head_chunked_token_ce(
        student.hidden_states,
        student.target_ids,
        target.lm_head,
        prefix_width=student.prefix_width,
        token_weights=weights,
        chunk_size=ce_chunk_size,
    )
    total = ce.loss
    metrics = ObjectiveMetrics(
        loss=float(total.detach().item()),
        ce=float(ce.loss.detach().item()),
        token_accuracy=float(ce.token_accuracy.item()),
        target_tokens=int(ce.token_count.item()),
        target_offset=window.start,
        source_tokens=source_tokens,
        padded_source_tokens=padded_source_tokens,
        soft_tokens=int(memory.shape[1]),
    )
    return total, metrics


def _normalize_gradients(
    parameters: Iterator[torch.nn.Parameter], samples: int
) -> None:
    if samples <= 0:
        raise ValueError("gradient normalization requires at least one sample")
    scale = 1.0 / float(samples)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def _step_optimizer(
    optimizer: torch.optim.AdamW,
    residency: LowPeakAdamMomentResidency,
) -> None:
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    try:
        residency.restore(optimizer)
        optimizer.step()
    finally:
        residency.offload(optimizer, reason="after_step", tolerate_partial=True)
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    optimizer.zero_grad(set_to_none=True)


def _parameters_finite(module: torch.nn.Module) -> bool:
    local = torch.ones(
        (), device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.int32
    )
    with torch.no_grad():
        for parameter in module.parameters():
            if not bool(torch.isfinite(parameter).all().item()):
                local.zero_()
                break
    dist.all_reduce(local, op=dist.ReduceOp.MIN)
    return bool(local.item())


def _checkpoint_configs() -> tuple[ShardedStateDictConfig, ShardedOptimStateDictConfig]:
    return (
        ShardedStateDictConfig(offload_to_cpu=True),
        ShardedOptimStateDictConfig(offload_to_cpu=True),
    )


def _checkpoint_cache_barrier() -> None:
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    dist.barrier()


def _save_checkpoint(
    output_dir: Path,
    *,
    compressor: FSDP,
    optimizer: torch.optim.AdamW,
    residency: LowPeakAdamMomentResidency,
    rank: int,
    global_step: int,
    cursor: QueueCursor,
    run_contract_sha256: str,
) -> Path:
    destination = output_dir / f"checkpoint-{global_step:08d}"
    staging = output_dir / f".{destination.name}.staging"
    if rank == 0:
        if destination.exists() or staging.exists():
            raise FileExistsError(f"checkpoint destination exists: {destination}")
        staging.mkdir()
    _checkpoint_cache_barrier()
    try:
        residency.prepare_forward(optimizer)
        state_config, optim_config = _checkpoint_configs()
        with FSDP.state_dict_type(
            compressor,
            StateDictType.SHARDED_STATE_DICT,
            state_config,
            optim_config,
        ):
            model_state = compressor.state_dict()
            _checkpoint_cache_barrier()
            optimizer_state = FSDP.optim_state_dict(compressor, optimizer)
            _checkpoint_cache_barrier()
    finally:
        residency.offload(optimizer, reason="after_save", tolerate_partial=True)
        _checkpoint_cache_barrier()
    payload = {
        "format_version": FORMAT_VERSION,
        "schema": CHECKPOINT_SCHEMA,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "global_step": global_step,
        "run_contract_sha256": run_contract_sha256,
        "model": model_state,
        "optimizer": optimizer_state,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
        "rng_python": random.getstate(),
    }
    torch.save(payload, staging / f"rank-{rank:05d}.pt")
    dist.barrier()
    if rank == 0:
        _atomic_json(
            staging / "run_config.json",
            json.loads((output_dir / "run_config.json").read_text(encoding="utf-8")),
        )
        _atomic_json(
            staging / "trainer_state.json",
            {
                "format_version": FORMAT_VERSION,
                "schema": CHECKPOINT_SCHEMA,
                "global_step": global_step,
                "cursor": dataclasses.asdict(cursor),
                "run_contract_sha256": run_contract_sha256,
                "rank_files": [
                    f"rank-{index:05d}.pt" for index in range(dist.get_world_size())
                ],
            },
        )
        os.replace(staging, destination)
    dist.barrier()
    return destination


def _load_checkpoint(
    checkpoint: Path,
    *,
    compressor: FSDP,
    optimizer: torch.optim.AdamW,
    residency: LowPeakAdamMomentResidency,
    rank: int,
    checkpoint_format_version: int,
    checkpoint_schema: str,
    checkpoint_run_contract_sha256: str,
) -> tuple[int, QueueCursor]:
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    if (
        trainer.get("format_version") != checkpoint_format_version
        or trainer.get("schema") != checkpoint_schema
        or trainer.get("run_contract_sha256") != checkpoint_run_contract_sha256
    ):
        raise ValueError("checkpoint trainer contract mismatch")
    payload = torch.load(
        checkpoint / f"rank-{rank:05d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        payload.get("format_version") != checkpoint_format_version
        or payload.get("schema") != checkpoint_schema
        or payload.get("rank") != rank
        or payload.get("world_size") != dist.get_world_size()
        or payload.get("global_step") != trainer.get("global_step")
        or payload.get("run_contract_sha256") != checkpoint_run_contract_sha256
    ):
        raise ValueError("checkpoint rank contract mismatch")
    state_config, optim_config = _checkpoint_configs()
    try:
        with FSDP.state_dict_type(
            compressor,
            StateDictType.SHARDED_STATE_DICT,
            state_config,
            optim_config,
        ):
            compressor.load_state_dict(payload["model"])
            converted = FSDP.optim_state_dict_to_load(
                compressor, optimizer, payload["optimizer"]
            )
            optimizer.load_state_dict(converted)
            del converted
            payload.pop("optimizer", None)
    finally:
        residency.offload(optimizer, reason="after_resume", tolerate_partial=True)
        torch.cuda.empty_cache()
    torch.set_rng_state(payload["rng_cpu"])
    torch.cuda.set_rng_state(payload["rng_cuda"])
    random.setstate(payload["rng_python"])
    cursor_value = trainer.get("cursor")
    if not isinstance(cursor_value, dict):
        raise ValueError("checkpoint has no queue cursor")
    if "chunk_index" in cursor_value:
        chunk_index = int(cursor_value["chunk_index"])
    elif checkpoint_format_version == LEGACY_FORMAT_VERSION:
        chunk_index = 0
    else:
        raise ValueError("v4 checkpoint cursor has no chunk_index")
    cursor = QueueCursor(
        epoch=int(cursor_value["epoch"]),
        shard_index=int(cursor_value["shard_index"]),
        event_index=int(cursor_value["event_index"]),
        chunk_index=chunk_index,
    )
    if (
        min(cursor.epoch, cursor.shard_index, cursor.event_index, cursor.chunk_index)
        < 0
    ):
        raise ValueError("checkpoint cursor contains a negative component")
    return int(trainer["global_step"]), cursor


def _prune_checkpoints(output_dir: Path, *, keep_last: int, rank: int) -> None:
    """Retain only the newest exact checkpoint directories in this run root."""

    if keep_last <= 0:
        return
    if rank == 0:
        candidates: list[tuple[int, Path]] = []
        for path in output_dir.iterdir():
            match = re.fullmatch(r"checkpoint-([0-9]{8})", path.name)
            if match is None or path.is_symlink() or not path.is_dir():
                continue
            if path.parent.resolve() != output_dir.resolve():
                raise RuntimeError("checkpoint retention target escaped output root")
            candidates.append((int(match.group(1)), path))
        for _step, path in sorted(candidates)[:-keep_last]:
            shutil.rmtree(path)
    dist.barrier()


def _next_cursor(
    *,
    epoch: int,
    shard_index: int,
    event_index: int,
    chunk_index: int,
    chunks_in_event: int,
    last_event_in_shard: bool,
    shard_count: int,
) -> QueueCursor:
    if chunk_index + 1 < chunks_in_event:
        return QueueCursor(epoch, shard_index, event_index, chunk_index + 1)
    if not last_event_in_shard:
        return QueueCursor(epoch, shard_index, event_index + 1, 0)
    if shard_index + 1 < shard_count:
        return QueueCursor(epoch, shard_index + 1, 0, 0)
    return QueueCursor(epoch + 1, 0, 0, 0)


def _with_last(values: Iterator[Any]) -> Iterator[tuple[Any, bool]]:
    """Stream values with a final-item marker without materializing a shard."""

    iterator = iter(values)
    try:
        previous = next(iterator)
    except StopIteration:
        return
    for value in iterator:
        yield previous, False
        previous = value
    yield previous, True


def _verify_receipt_distributed(receipt: ShardReceipt, root: Path, rank: int) -> None:
    result: list[str | None] = [None]
    if rank == 0:
        try:
            verify_shard_receipt(
                receipt,
                data_root=root,
                image_sample_count=0,
                repeat_only=True,
            )
        except Exception as error:  # broadcast failure before either rank reads it.
            result[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(result, src=0)
    if result[0] is not None:
        raise RuntimeError(f"ready-shard receipt verification failed: {result[0]}")


def _portable_config(init_compressor: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = init_compressor.parent / "config.json"
    manifest_path = init_compressor.parent / "manifest.json"
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        config.get("target_family") != "qwen38"
        or config.get("schema") != manifest.get("export_schema")
        or manifest.get("compressor_file") != init_compressor.name
        or manifest.get("config_file") != config_path.name
        or manifest.get("config_sha256") != _canonical_json_sha256(config)
        or manifest.get("compressor_sha256") != _sha256(init_compressor)
    ):
        raise ValueError("portable compressor export contract is invalid")
    return config, manifest


def _load_portable_weights(
    module: GrepSeekDFlashCompressor,
    init_compressor: Path,
    *,
    config: Mapping[str, Any],
) -> None:
    if module.export_config() != config.get("compressor"):
        raise ValueError("portable compressor architecture does not match")
    state = torch.load(init_compressor, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise TypeError("portable compressor must be a tensor state dictionary")
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("portable compressor strict load returned incompatible keys")


def _learning_rate(step: int, *, base: float, warmup: int) -> float:
    return base if warmup <= 0 else base * min(1.0, (step + 1) / warmup)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--init-compressor", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-additional-steps", type=int, default=0)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--source-block-tokens", type=int, default=1024)
    parser.add_argument("--compression-ratio", type=int, default=16)
    parser.add_argument("--max-source-blocks", type=int, default=16)
    parser.add_argument("--target-window-tokens", type=int, default=256)
    parser.add_argument("--target-context-tokens", type=int, default=256)
    parser.add_argument("--position-weight-alpha", type=float, default=1.0)
    parser.add_argument("--position-weight-tau", type=float, default=64.0)
    parser.add_argument("--position-weight-power", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--ce-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument("--verify-parameters-finite-after-step", action="store_true")
    parser.add_argument("--disable-target-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--disable-compressor-gradient-checkpointing", action="store_true"
    )
    parser.add_argument("--validate-input-only", action="store_true")
    parser.add_argument("--validate-repeat-chunks-only", action="store_true")
    parser.add_argument("--repeat-chunk-plan-output", type=Path)
    parser.add_argument("--repeat-chunk-plan-receipt", type=Path)
    parser.add_argument(
        "--allow-legacy-v3-token-chunk-resume",
        action="store_true",
        help="allow the single audited v3 -> exact-token-chunked v4 migration",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "grad_accumulation",
        "source_block_tokens",
        "compression_ratio",
        "max_source_blocks",
        "target_window_tokens",
        "ce_chunk_size",
        "log_interval",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.target_context_tokens < 0 or args.warmup_steps < 0:
        raise ValueError("target context and warmup must be nonnegative")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("--max-steps must be -1 or positive")
    if args.max_additional_steps < 0:
        raise ValueError("--max-additional-steps must be nonnegative")
    if args.max_steps > 0 and args.max_additional_steps > 0:
        raise ValueError("max-steps and max-additional-steps are mutually exclusive")
    if args.validate_input_only and args.validate_repeat_chunks_only:
        raise ValueError("input-only validation modes are mutually exclusive")
    if (
        args.repeat_chunk_plan_output is not None
        and not args.validate_repeat_chunks_only
    ):
        raise ValueError(
            "--repeat-chunk-plan-output requires --validate-repeat-chunks-only"
        )
    if (
        not args.validate_input_only
        and not args.validate_repeat_chunks_only
        and args.repeat_chunk_plan_receipt is None
    ):
        raise ValueError("training requires --repeat-chunk-plan-receipt")
    if args.save_interval < 0 or args.keep_last_checkpoints < 0:
        raise ValueError("checkpoint interval/retention must be nonnegative")
    if args.source_block_tokens % args.compression_ratio:
        raise ValueError("source block size must be divisible by compression ratio")
    if args.source_block_tokens != 1024 or args.compression_ratio != 16:
        raise ValueError("MMFineReasoning repeat currently pins 1024 -> 64 compression")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("optimizer hyperparameters are invalid")
    bounded_causal_position_weights(
        2,
        absolute_start=0,
        alpha=args.position_weight_alpha,
        tau=args.position_weight_tau,
        power=args.position_weight_power,
    )
    if (args.init_compressor is None) == (args.resume_from is None):
        raise ValueError("provide exactly one of --init-compressor or --resume-from")


def _args_contract(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for name, value in tuple(result.items()):
        if isinstance(value, Path):
            result[name] = str(value.expanduser().resolve())
    for name in (
        "output_dir",
        "tensorboard_dir",
        "resume_from",
        "init_compressor",
        "max_steps",
        "max_additional_steps",
        "save_interval",
        "keep_last_checkpoints",
        "log_interval",
        "skip_final_checkpoint",
        "validate_input_only",
        "validate_repeat_chunks_only",
        "repeat_chunk_plan_output",
        "repeat_chunk_plan_receipt",
        "allow_legacy_v3_token_chunk_resume",
    ):
        result.pop(name, None)
    return result


def _source_run_contract(checkpoint: Path) -> dict[str, Any]:
    path = checkpoint.resolve() / "run_config.json"
    if not path.exists():
        path = checkpoint.resolve().parent / "run_config.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"resume source has no valid run_config.json: {path}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("resume source run_config must be an object")
    contract = dict(value)
    recorded_sha256 = contract.pop("run_contract_sha256", None)
    for runtime_key in (
        "output_dir",
        "tensorboard_dir",
        "resume_from",
        "init_compressor",
    ):
        contract.pop(runtime_key, None)
    if recorded_sha256 != _canonical_json_sha256(contract):
        raise ValueError("resume source run contract self-hash is invalid")
    return contract


def _preserve_checkpoint_run_configs(output_dir: Path) -> None:
    """Keep older checkpoints resumable when an in-place run upgrades its contract."""

    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract_hash = config.get("run_contract_sha256")
    if not isinstance(contract_hash, str):
        raise ValueError("existing output run_config has no contract identity")
    for checkpoint in sorted(output_dir.glob("checkpoint-*")):
        if checkpoint.is_symlink() or not checkpoint.is_dir():
            raise ValueError(f"invalid existing checkpoint directory: {checkpoint}")
        header_path = checkpoint / "trainer_state.json"
        if not header_path.is_file():
            continue
        header = json.loads(header_path.read_text(encoding="utf-8"))
        if header.get("run_contract_sha256") != contract_hash:
            continue
        snapshot = checkpoint / "run_config.json"
        if snapshot.exists():
            if (
                _canonical_json_sha256(_source_run_contract(checkpoint))
                != contract_hash
            ):
                raise ValueError(f"checkpoint run_config snapshot differs: {snapshot}")
        else:
            _atomic_json(snapshot, config)


def _open_tensorboard_writer(path: Path, *, resume_step: int | None) -> SummaryWriter:
    # Keep the checkpoint's metrics; hide only abandoned steps after it.
    return SummaryWriter(
        log_dir=str(path),
        purge_step=resume_step + 1 if resume_step is not None else None,
    )


def _legacy_manifest_extension(
    source_contract: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ReadyShardQueue]:
    """Prove that the current frozen queue only adds immutable source shards."""

    source_manifest_path = source_contract.get("manifest")
    if not isinstance(source_manifest_path, str):
        raise ValueError("legacy run contract has no manifest path")
    source_manifest = load_validated_shard_manifest(source_manifest_path)
    if source_manifest.get("manifest_sha256") != source_contract.get(
        "manifest_sha256"
    ) or source_manifest.get("counts") != source_contract.get("manifest_counts"):
        raise ValueError("legacy manifest differs from its parent run contract")
    counts = current_manifest.get("counts")
    policy = current_manifest.get("policy")
    selection = policy.get("selection") if isinstance(policy, dict) else None
    if (
        current_manifest.get("schema") != REPEAT_MANIFEST_SCHEMA
        or not isinstance(counts, dict)
        or counts.get("deferred_shards") != 0
        or counts.get("rejected_shards") != 0
        or counts.get("discovered_candidates") != counts.get("ready_shards")
        or not isinstance(selection, dict)
        or selection.get("mode") != "all_discovered_finalized_parquet"
    ):
        raise ValueError("current manifest is not a complete finalized snapshot")
    if (
        Path(str(source_manifest["data_root"])).resolve()
        != Path(str(current_manifest["data_root"])).resolve()
    ):
        raise ValueError("extended manifest changed the data root")
    parent = {
        item.path: item
        for item in (
            ShardReceipt.from_mapping(value) for value in source_manifest["shards"]
        )
    }
    current = {
        item.path: item
        for item in (
            ShardReceipt.from_mapping(value) for value in current_manifest["shards"]
        )
    }
    for path, parent_receipt in parent.items():
        current_receipt = current.get(path)
        if current_receipt is None:
            raise ValueError(f"extended manifest removed parent shard: {path}")
        parent_value = parent_receipt.as_dict()
        current_value = current_receipt.as_dict()
        parent_value.pop("image_decode_samples", None)
        current_value.pop("image_decode_samples", None)
        if current_value != parent_value:
            raise ValueError(f"extended manifest changed parent shard receipt: {path}")
    additions = tuple(current[path] for path in sorted(set(current) - set(parent)))
    if not additions:
        raise ValueError("legacy queue migration expected at least one added shard")
    added_rows = sum(receipt.rows for receipt in additions)
    receipt = {
        "schema": "immutable_manifest_superset_v1",
        "parent_manifest_sha256": source_manifest["manifest_sha256"],
        "parent_ready_shards": source_manifest["counts"]["ready_shards"],
        "parent_ready_rows": source_manifest["counts"]["ready_rows"],
        "current_manifest_sha256": current_manifest["manifest_sha256"],
        "current_ready_shards": counts["ready_shards"],
        "current_ready_rows": counts["ready_rows"],
        "added_shards": len(additions),
        "added_rows": added_rows,
        "added_receipts_sha256": _canonical_json_sha256(
            [item.as_dict() for item in additions]
        ),
        "result": "all_parent_shard_receipts_unchanged_and_new_shards_added",
    }
    return receipt, ReadyShardQueue.from_manifest(source_manifest)


def _checkpoint_header(
    checkpoint: Path,
    *,
    expected_format_version: int,
    expected_schema: str,
    expected_run_contract_sha256: str,
) -> dict[str, Any]:
    """Validate the small immutable checkpoint header before model allocation."""

    path = checkpoint.resolve() / "trainer_state.json"
    try:
        header = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"resume checkpoint has no valid trainer_state.json: {path}"
        ) from error
    if not isinstance(header, dict):
        raise TypeError("resume checkpoint trainer state must be an object")
    if (
        header.get("format_version") != expected_format_version
        or header.get("schema") != expected_schema
        or header.get("run_contract_sha256") != expected_run_contract_sha256
    ):
        raise ValueError("resume checkpoint header contract mismatch")
    if header.get("rank_files") != ["rank-00000.pt", "rank-00001.pt"]:
        raise ValueError(
            "resume checkpoint must contain exactly two canonical rank files"
        )
    for name in header["rank_files"]:
        rank_path = checkpoint.resolve() / name
        if (
            rank_path.is_symlink()
            or not rank_path.is_file()
            or rank_path.stat().st_size <= 0
        ):
            raise ValueError(f"resume checkpoint rank file is invalid: {rank_path}")
    cursor = header.get("cursor")
    if not isinstance(cursor, dict):
        raise ValueError("resume checkpoint has no cursor object")
    required_cursor = {"epoch", "shard_index", "event_index"}
    expected_cursor = (
        required_cursor | {"chunk_index"}
        if expected_format_version == FORMAT_VERSION
        else required_cursor
    )
    if set(cursor) != expected_cursor:
        raise ValueError(
            "resume checkpoint cursor fields do not match its format version"
        )
    if any(
        isinstance(cursor[name], bool)
        or not isinstance(cursor[name], int)
        or cursor[name] < 0
        for name in expected_cursor
    ):
        raise ValueError("resume checkpoint cursor contains an invalid component")
    if (
        isinstance(header.get("global_step"), bool)
        or not isinstance(header.get("global_step"), int)
        or header["global_step"] < 0
    ):
        raise ValueError("resume checkpoint global_step is invalid")
    return header


def _audit_legacy_cursor_prefix(
    queue: ReadyShardQueue,
    tokenizer: Any,
    *,
    cursor: QueueCursor,
    maximum_source_tokens: int,
) -> dict[str, Any]:
    """Prove that a v3 cursor did not skip chunks newly introduced by v4."""

    if cursor.epoch != 0 or cursor.chunk_index != 0:
        raise ValueError(
            "legacy chunk migration is restricted to an epoch-0 base cursor"
        )
    if cursor.shard_index >= len(queue.shards):
        raise ValueError("legacy cursor shard is outside the frozen queue")
    digest = hashlib.sha256()
    checked = 0
    maximum_observed = 0
    for shard_index, receipt in enumerate(queue.shards[: cursor.shard_index + 1]):
        stop = cursor.event_index if shard_index == cursor.shard_index else None
        events = iter_shard_repeat_events(
            receipt,
            data_root=queue.root,
            repeat_rotation=cursor.epoch,
            verify_receipt=False,
            skip_empty_fields=True,
        )
        observed_in_cursor_shard = 0
        for event_index, event in enumerate(events):
            if stop is not None and event_index >= stop:
                break
            chunks = deterministic_repeat_token_chunks(
                tokenizer,
                event.text,
                event_id=event.event_id,
                maximum_source_tokens=maximum_source_tokens,
            )
            if len(chunks) != 1:
                raise ValueError(
                    "legacy cursor prefix contains an event that v4 would split: "
                    f"shard={shard_index} event={event_index} id={event.event_id}"
                )
            token_count = len(chunks[0].token_ids)
            maximum_observed = max(maximum_observed, token_count)
            digest.update(
                json.dumps(
                    [shard_index, event_index, event.event_id, token_count],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
            checked += 1
            if shard_index == cursor.shard_index:
                observed_in_cursor_shard += 1
        if stop is not None and observed_in_cursor_shard != stop:
            raise ValueError("legacy cursor event is outside its shard")
    return {
        "events_checked": checked,
        "maximum_source_tokens_observed": maximum_observed,
        "identity_sha256": digest.hexdigest(),
        "result": "all_prior_base_events_are_single_chunk",
    }


def _bind_legacy_optimizer_boundary(
    audit: Mapping[str, Any],
    *,
    global_step: int,
    grad_accumulation: int,
) -> dict[str, Any]:
    """Bind a cursor-prefix audit to an empty legacy gradient accumulator."""

    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
        or isinstance(grad_accumulation, bool)
        or not isinstance(grad_accumulation, int)
        or grad_accumulation <= 0
    ):
        raise ValueError("legacy optimizer-boundary inputs are invalid")
    observed = audit.get("events_checked")
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        raise ValueError("legacy cursor-prefix audit has an invalid event count")
    expected = global_step * grad_accumulation
    if observed != expected:
        raise ValueError(
            "legacy cursor is not at the optimizer boundary implied by "
            "global_step and gradient accumulation: "
            f"cursor={observed} expected={expected}"
        )
    return {
        **audit,
        "gradient_accumulation": grad_accumulation,
        "expected_consumed_base_events": expected,
        "optimizer_accumulator_empty": True,
    }


def _as_legacy_v3_contract(
    current: Mapping[str, Any],
    *,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the v4 contract onto the exact historical v3 contract shape."""

    projected = json.loads(json.dumps(current, ensure_ascii=False, allow_nan=False))
    projected["format_version"] = LEGACY_FORMAT_VERSION
    projected["schema"] = LEGACY_CHECKPOINT_SCHEMA
    projected["objective_schema"] = LEGACY_OBJECTIVE_SCHEMA
    projected["cursor_unit"] = "flattened_repeat_event_within_shard"
    projected["data_projection"] = (
        "source_id_question_original_answer_caption_only_no_image_column"
    )
    projected.pop("migration", None)
    projected.pop("repeat_chunk_plan", None)
    projected.pop("base_repeat_events_per_epoch", None)
    projected["repeat_events_per_epoch"] = int(
        projected["manifest_counts"]["ready_rows"]
    ) * len(REPEAT_FIELDS)
    projected["repeat_objective"][
        "field_schedule"
    ] = "all_three_each_epoch_stable_hash_rotated_order"
    projected["repeat_objective"]["events_per_row"] = len(REPEAT_FIELDS)
    projected["repeat_objective"].pop("candidate_events_per_row", None)
    projected["repeat_objective"].pop("empty_normalized_field_policy", None)
    projected["repeat_objective"].pop("token_chunking", None)
    projected["source_layout"].pop("maximum_source_tokens_per_chunk", None)
    projected["position_weighting"].pop("position_coordinate", None)
    if source is not None:
        for key in ("manifest", "manifest_sha256", "manifest_counts"):
            projected[key] = source[key]
        projected["repeat_events_per_epoch"] = source["repeat_events_per_epoch"]
        projected["args"]["manifest"] = source["args"]["manifest"]
    return projected


def repeat_chunk_plan_summary(
    queue: ReadyShardQueue,
    tokenizer: Any,
    *,
    maximum_source_tokens: int,
) -> dict[str, Any]:
    """Exhaustively validate and summarize the epoch-invariant chunk plan."""

    digest = hashlib.sha256()
    base_events = 0
    token_chunks = 0
    split_base_events = 0
    chunk_count_histogram: dict[int, int] = {}
    maximum_original_tokens = 0
    maximum_event_id: str | None = None
    maximum_chunk_tokens = 0
    base_events_by_shard: list[int] = []
    field_event_counts = {field: 0 for field in REPEAT_FIELDS}
    for receipt in queue.shards:
        shard_base_events = 0
        events = iter_shard_repeat_events(
            receipt,
            data_root=queue.root,
            repeat_rotation=0,
            verify_receipt=False,
            skip_empty_fields=True,
        )
        for event_batch in _event_batches(events):
            token_batch = _batch_repeat_token_ids(tokenizer, event_batch)
            for event, token_ids in zip(event_batch, token_batch, strict=True):
                chunks = repeat_token_chunks_from_ids(
                    token_ids,
                    event_id=event.event_id,
                    maximum_source_tokens=maximum_source_tokens,
                )
                original_tokens = sum(len(chunk.token_ids) for chunk in chunks)
                if original_tokens > maximum_original_tokens:
                    maximum_original_tokens = original_tokens
                    maximum_event_id = event.event_id
                maximum_chunk_tokens = max(
                    maximum_chunk_tokens, *(len(chunk.token_ids) for chunk in chunks)
                )
                base_events += 1
                shard_base_events += 1
                field_event_counts[event.field] += 1
                token_chunks += len(chunks)
                split_base_events += int(len(chunks) > 1)
                chunk_count_histogram[len(chunks)] = (
                    chunk_count_histogram.get(len(chunks), 0) + 1
                )
                digest.update(
                    json.dumps(
                        [event.event_id, [len(chunk.token_ids) for chunk in chunks]],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                digest.update(b"\n")
        base_events_by_shard.append(shard_base_events)
    return {
        "status": "VALIDATED_REPEAT_TOKEN_CHUNK_PLAN",
        "base_events_per_epoch": base_events,
        "base_events_by_shard": base_events_by_shard,
        "field_event_counts": field_event_counts,
        "token_chunks_per_epoch": token_chunks,
        "split_base_events": split_base_events,
        "chunk_count_histogram": {
            str(key): value for key, value in sorted(chunk_count_histogram.items())
        },
        "maximum_source_tokens_per_chunk": maximum_source_tokens,
        "maximum_original_tokens": maximum_original_tokens,
        "maximum_original_event_id": maximum_event_id,
        "maximum_observed_chunk_tokens": maximum_chunk_tokens,
        "plan_sha256": digest.hexdigest(),
    }


def _load_repeat_chunk_plan_receipt(
    path: Path,
    *,
    manifest_sha256: str,
    tokenizer_receipt: Mapping[str, Any],
    maximum_source_tokens: int,
    shard_count: int,
    manifest_ready_rows: int,
) -> dict[str, Any]:
    """Validate the exhaustive CPU plan receipt used by a GPU run."""

    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        raise ValueError(f"repeat chunk plan is not a regular file: {path}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid repeat chunk plan receipt: {resolved}") from error
    if not isinstance(value, dict):
        raise ValueError("repeat chunk plan receipt must be an object")
    if (
        value.get("schema") != REPEAT_CHUNK_PLAN_SCHEMA
        or value.get("status") != "VALIDATED_REPEAT_TOKEN_CHUNK_PLAN"
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("tokenizer") != tokenizer_receipt
        or value.get("maximum_source_tokens_per_chunk") != maximum_source_tokens
        or value.get("empty_normalized_field_policy")
        != "exclude_field_keep_other_row_fields"
    ):
        raise ValueError("repeat chunk plan identity does not match this run")

    def receipt_int(name: str, *, minimum: int = 0) -> int:
        observed = value.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < minimum
        ):
            raise ValueError(f"repeat chunk plan has invalid {name}")
        return observed

    base_events = receipt_int("base_events_per_epoch", minimum=1)
    token_chunks = receipt_int("token_chunks_per_epoch", minimum=base_events)
    split_events = receipt_int("split_base_events")
    excluded_fields = receipt_int("excluded_empty_normalized_fields")
    ready_rows = receipt_int("ready_rows", minimum=1)
    candidate_events = receipt_int("candidate_field_events_per_epoch", minimum=1)
    if (
        ready_rows != manifest_ready_rows
        or candidate_events != ready_rows * len(REPEAT_FIELDS)
        or candidate_events - base_events != excluded_fields
    ):
        raise ValueError("repeat chunk plan row/event totals are inconsistent")
    by_shard = value.get("base_events_by_shard")
    if (
        not isinstance(by_shard, list)
        or len(by_shard) != shard_count
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in by_shard
        )
        or sum(by_shard) != base_events
    ):
        raise ValueError("repeat chunk plan has invalid per-shard event counts")
    field_counts = value.get("field_event_counts")
    if (
        not isinstance(field_counts, dict)
        or set(field_counts) != set(REPEAT_FIELDS)
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in field_counts.values()
        )
        or sum(field_counts.values()) != base_events
    ):
        raise ValueError("repeat chunk plan has invalid per-field event counts")
    histogram = value.get("chunk_count_histogram")
    if not isinstance(histogram, dict) or not histogram:
        raise ValueError("repeat chunk plan has no chunk-count histogram")
    try:
        parsed_histogram = {int(key): count for key, count in histogram.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("repeat chunk plan histogram keys are invalid") from error
    if (
        any(
            key <= 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for key, count in parsed_histogram.items()
        )
        or sum(parsed_histogram.values()) != base_events
        or sum(key * count for key, count in parsed_histogram.items()) != token_chunks
        or sum(count for key, count in parsed_histogram.items() if key > 1)
        != split_events
    ):
        raise ValueError("repeat chunk plan histogram is inconsistent")
    if excluded_fields != sum(ready_rows - count for count in field_counts.values()):
        raise ValueError("repeat chunk plan empty-field count is inconsistent")
    maximum_observed = receipt_int("maximum_observed_chunk_tokens", minimum=1)
    if maximum_observed > maximum_source_tokens:
        raise ValueError("repeat chunk plan contains an over-capacity chunk")
    plan_sha256 = value.get("plan_sha256")
    if (
        not isinstance(plan_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", plan_sha256) is None
    ):
        raise ValueError("repeat chunk plan identity hash is invalid")
    return {
        "receipt_file_sha256": _sha256(resolved),
        "plan": value,
    }


def train(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    manifest = load_validated_shard_manifest(args.manifest)
    if manifest.get("schema") != REPEAT_MANIFEST_SCHEMA:
        raise ValueError("repeat training requires a repeat-only manifest")
    queue = ReadyShardQueue.from_manifest(manifest)
    if not queue.shards:
        raise ValueError("validated MMFineReasoning queue is empty")
    if args.validate_input_only:
        print(
            json.dumps(
                {
                    "status": "VALIDATED_INPUT_ONLY",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "shards": len(queue.shards),
                    "rows": manifest["counts"]["ready_rows"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.validate_repeat_chunks_only:
        processor = load_qwen38_processor(args.target_model)
        tokenizer_receipt = _tokenizer_files_receipt(args.target_model)
        summary = repeat_chunk_plan_summary(
            queue,
            processor.tokenizer,
            maximum_source_tokens=(args.source_block_tokens * args.max_source_blocks),
        )
        ready_rows = int(manifest["counts"]["ready_rows"])
        summary.update(
            {
                "schema": REPEAT_CHUNK_PLAN_SCHEMA,
                "manifest_sha256": manifest["manifest_sha256"],
                "ready_rows": ready_rows,
                "candidate_field_events_per_epoch": ready_rows * len(REPEAT_FIELDS),
                "excluded_empty_normalized_fields": ready_rows * len(REPEAT_FIELDS)
                - int(summary["base_events_per_epoch"]),
                "empty_normalized_field_policy": (
                    "exclude_field_keep_other_row_fields"
                ),
                "tokenizer": tokenizer_receipt,
            }
        )
        if args.repeat_chunk_plan_output is not None:
            plan_output = args.repeat_chunk_plan_output.expanduser()
            if plan_output.exists() or plan_output.is_symlink():
                raise FileExistsError(
                    f"repeat chunk plan output already exists: {plan_output}"
                )
            _atomic_json(plan_output, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("MMFineReasoning repeat requires exactly two ranks")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    tensorboard_dir = (
        args.tensorboard_dir.expanduser().resolve()
        if args.tensorboard_dir is not None
        else output_dir / "tensorboard"
    )
    if rank == 0:
        if args.resume_from is None and output_dir.exists():
            raise FileExistsError(
                f"fresh output directory already exists: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=args.resume_from is not None)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    source_contract = (
        _source_run_contract(args.resume_from) if args.resume_from is not None else None
    )
    processor = load_qwen38_processor(args.target_model)
    tokenizer_receipt = _tokenizer_files_receipt(args.target_model)
    if args.repeat_chunk_plan_receipt is None:
        raise RuntimeError("repeat chunk plan receipt validation was bypassed")
    repeat_chunk_plan = _load_repeat_chunk_plan_receipt(
        args.repeat_chunk_plan_receipt,
        manifest_sha256=manifest["manifest_sha256"],
        tokenizer_receipt=tokenizer_receipt,
        maximum_source_tokens=(args.source_block_tokens * args.max_source_blocks),
        shard_count=len(queue.shards),
        manifest_ready_rows=int(manifest["counts"]["ready_rows"]),
    )
    planned_base_events_by_shard = tuple(
        int(value) for value in repeat_chunk_plan["plan"]["base_events_by_shard"]
    )
    source_run_contract_sha256: str | None = None
    source_checkpoint_format = FORMAT_VERSION
    source_checkpoint_schema = CHECKPOINT_SCHEMA
    source_header: dict[str, Any] | None = None
    migration_receipt: dict[str, Any] | None = None
    if source_contract is not None:
        assert args.resume_from is not None
        source_run_contract_sha256 = _canonical_json_sha256(source_contract)
        source_checkpoint_format = int(source_contract.get("format_version", -1))
        source_checkpoint_schema = str(source_contract.get("schema", ""))
        source_header = _checkpoint_header(
            args.resume_from.resolve(),
            expected_format_version=source_checkpoint_format,
            expected_schema=source_checkpoint_schema,
            expected_run_contract_sha256=source_run_contract_sha256,
        )
        if source_checkpoint_format == LEGACY_FORMAT_VERSION:
            if (
                not args.allow_legacy_v3_token_chunk_resume
                or source_checkpoint_schema != LEGACY_CHECKPOINT_SCHEMA
                or source_contract.get("objective_schema") != LEGACY_OBJECTIVE_SCHEMA
            ):
                raise ValueError(
                    "legacy v3 resume requires the explicit token-chunk migration gate"
                )
            queue_extension, legacy_queue = _legacy_manifest_extension(
                source_contract,
                manifest,
            )
            legacy_cursor_value = source_header["cursor"]
            legacy_cursor = QueueCursor(
                epoch=int(legacy_cursor_value["epoch"]),
                shard_index=int(legacy_cursor_value["shard_index"]),
                event_index=int(legacy_cursor_value["event_index"]),
                chunk_index=0,
            )
            audit_box: list[dict[str, Any] | None] = [None]
            if rank == 0:
                try:
                    legacy_args = source_contract.get("args")
                    if not isinstance(legacy_args, dict):
                        raise ValueError("legacy run contract has no args object")
                    legacy_grad_accumulation = legacy_args.get("grad_accumulation")
                    if (
                        isinstance(legacy_grad_accumulation, bool)
                        or not isinstance(legacy_grad_accumulation, int)
                        or legacy_grad_accumulation <= 0
                    ):
                        raise ValueError(
                            "legacy run contract has invalid gradient accumulation"
                        )
                    legacy_prefix_audit = _audit_legacy_cursor_prefix(
                        legacy_queue,
                        processor.tokenizer,
                        cursor=legacy_cursor,
                        maximum_source_tokens=(
                            args.source_block_tokens * args.max_source_blocks
                        ),
                    )
                    current_prefix_audit = _audit_legacy_cursor_prefix(
                        queue,
                        processor.tokenizer,
                        cursor=legacy_cursor,
                        maximum_source_tokens=(
                            args.source_block_tokens * args.max_source_blocks
                        ),
                    )
                    if current_prefix_audit != legacy_prefix_audit:
                        raise ValueError(
                            "extended manifest changed the consumed event prefix"
                        )
                    audit = _bind_legacy_optimizer_boundary(
                        current_prefix_audit,
                        global_step=int(source_header["global_step"]),
                        grad_accumulation=legacy_grad_accumulation,
                    )
                    audit["identity_matches_parent_manifest"] = True
                    audit_box[0] = {
                        "audit": audit,
                        "rank_file_sha256": {
                            name: _sha256(args.resume_from.resolve() / name)
                            for name in source_header["rank_files"]
                        },
                    }
                except (
                    Exception
                ) as error:  # broadcast before either rank allocates model.
                    audit_box[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(audit_box, src=0)
            audit_result = audit_box[0]
            if not isinstance(audit_result, dict) or "error" in audit_result:
                raise ValueError(
                    "legacy v3 cursor-prefix audit failed: "
                    + str(
                        audit_result.get("error")
                        if isinstance(audit_result, dict)
                        else audit_result
                    )
                )
            checkpoint = args.resume_from.resolve()
            migration_receipt = {
                "schema": LEGACY_OVERFLOW_MIGRATION_SCHEMA,
                "parent_run_contract_sha256": source_run_contract_sha256,
                "parent_checkpoint": {
                    "path": str(checkpoint),
                    "global_step": int(source_header["global_step"]),
                    "cursor": dict(source_header["cursor"]),
                    "trainer_state_sha256": _sha256(checkpoint / "trainer_state.json"),
                    "rank_file_sizes": {
                        name: (checkpoint / name).stat().st_size
                        for name in source_header["rank_files"]
                    },
                    "rank_file_sha256": audit_result["rank_file_sha256"],
                },
                "cursor_upgrade": "missing_chunk_index_becomes_zero",
                "prefix_audit": audit_result["audit"],
                "queue_extension": queue_extension,
                "semantic_upgrade": {
                    "exact_source_token_ids": True,
                    "decode_or_retokenize": False,
                    "template_boundary_retokenization_removed": True,
                    "empty_field_slots_excluded": True,
                    "legacy_tokenizer_identity_limit": (
                        "parent_v3_bound_only_the_local_target_model_path; "
                        "v4_content-binds_current_tokenizer_files"
                    ),
                },
            }
        elif (
            source_checkpoint_format == FORMAT_VERSION
            and source_checkpoint_schema == CHECKPOINT_SCHEMA
            and source_contract.get("objective_schema") == OBJECTIVE_SCHEMA
        ):
            inherited = source_contract.get("migration")
            if inherited is not None and not isinstance(inherited, dict):
                raise ValueError("resume source migration receipt is malformed")
            migration_receipt = inherited
        else:
            raise ValueError("resume source uses an unsupported checkpoint contract")
    model = load_qwen38_multimodal_model(
        args.target_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        tp_plan="auto",
    )
    target = FrozenQwen38MultimodalTarget(
        model,
        gradient_checkpointing=not args.disable_target_gradient_checkpointing,
    )

    portable_box: list[tuple[dict[str, Any], dict[str, Any]] | None] = [None]
    if rank == 0:
        if args.init_compressor is not None:
            portable_box[0] = _portable_config(args.init_compressor.resolve())
    dist.broadcast_object_list(portable_box, src=0)
    portable = portable_box[0]
    if args.init_compressor is not None and portable is None:
        raise RuntimeError("portable init receipt broadcast failed")
    compressor_config = (
        portable[0]["compressor"]
        if portable is not None
        else source_contract.get("compressor") if source_contract is not None else None
    )
    if not isinstance(compressor_config, dict):
        raise RuntimeError("compressor architecture receipt is missing")
    output_rms = (
        float(compressor_config["output_embedding_rms"])
        if compressor_config is not None
        else 0.015053687617182732
    )
    compressor_base = GrepSeekDFlashCompressor.canonical_qwen36_27b(
        output_embedding_rms=output_rms,
        compression_ratio=args.compression_ratio,
        local_window=2048,
        gradient_checkpointing=not args.disable_compressor_gradient_checkpointing,
    ).float()
    if rank == 0 and args.init_compressor is not None:
        _load_portable_weights(
            compressor_base,
            args.init_compressor.resolve(),
            config=portable[0],  # type: ignore[index]
        )
    compressor = FSDP(
        compressor_base,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        use_orig_params=True,
        sync_module_states=True,
        limit_all_gathers=True,
    )
    if any(parameter.dtype != torch.float32 for parameter in compressor.parameters()):
        raise RuntimeError("compressor FSDP master parameters must remain FP32")
    compressor.train()
    optimizer = torch.optim.AdamW(
        compressor.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    residency = LowPeakAdamMomentResidency()

    init_identity: Any = None
    if portable is not None:
        init_identity = {
            "path": str(args.init_compressor.resolve()),
            "sha256": portable[1]["compressor_sha256"],
            "config_sha256": portable[1]["config_sha256"],
        }
    elif source_contract is not None:
        init_identity = source_contract.get("compressor_initialization")
        if not isinstance(init_identity, dict):
            raise ValueError("resume source lost portable initialization identity")
    run_contract = {
        "format_version": FORMAT_VERSION,
        "schema": CHECKPOINT_SCHEMA,
        "objective_schema": OBJECTIVE_SCHEMA,
        "target_model": str(args.target_model.expanduser().resolve()),
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_counts": manifest["counts"],
        "base_repeat_events_per_epoch": repeat_chunk_plan["plan"][
            "base_events_per_epoch"
        ],
        "repeat_chunk_plan": repeat_chunk_plan,
        "compressor_initialization": init_identity,
        "compressor": compressor.module.export_config(),
        "repeat_objective": {
            "fields": list(REPEAT_FIELDS),
            "qwen3vl_235b_thinking_response_excluded": True,
            "field_schedule": (
                "all_nonempty_normalized_fields_each_epoch_stable_hash_rotated_order"
            ),
            "candidate_events_per_row": len(REPEAT_FIELDS),
            "empty_normalized_field_policy": ("exclude_field_keep_other_row_fields"),
            "loss": "bounded_position_weighted_hard_label_cross_entropy_only",
            "teacher_distribution": None,
            "image_and_vqa_branch": None,
            "shared_user_instruction": "Repeat the compressed text exactly.",
            "assistant_prefix": "<think>\\n\\n</think>\\n\\n",
            "assistant_prefix_scored": False,
            "token_chunking": {
                "unit": "exact_original_token_ids",
                "maximum_source_tokens": (
                    args.source_block_tokens * args.max_source_blocks
                ),
                "decode_or_retokenize": False,
                "tokenizer": tokenizer_receipt,
                "singleton_event_id": "unchanged",
                "multi_chunk_event_id": (
                    "{base}:chunk:{one_based_index:06d}-of-{chunk_count:06d}"
                ),
                "target": "same_chunk_ids",
                "scoring": "one_epoch_rotated_window_within_each_chunk",
                "coverage_guarantee": (
                    "all_source_ids_exact_each_epoch; one_scored_target_window_"
                    "per_chunk_per_epoch"
                ),
                "gradient_accumulation_unit": "token_chunk",
            },
        },
        "data_projection": (
            "source_id_question_original_answer_caption_only_no_image_column_"
            "empty_field_slots_excluded"
        ),
        "cursor_unit": "base_repeat_event_plus_exact_token_chunk_within_shard",
        "source_layout": {
            "block_tokens": args.source_block_tokens,
            "soft_tokens_per_block": args.source_block_tokens // args.compression_ratio,
            "slot_allocation": "ceil(real_source_tokens/1024)*64",
            "source_positions": "real_unpadded_tokens",
            "maximum_source_tokens_per_chunk": (
                args.source_block_tokens * args.max_source_blocks
            ),
        },
        "position_weighting": {
            "formula": "normalize_window_mean(1+alpha*(1+absolute_position/tau)^(-power))",
            "floor_before_normalization": 1.0,
            "alpha": args.position_weight_alpha,
            "tau": args.position_weight_tau,
            "power": args.position_weight_power,
            "position_coordinate": "chunk_local_target_token_index",
        },
        "distributed": "TP2_frozen_Qwen3.8_text_path_plus_FULL_SHARD_compressor",
        "args": _args_contract(args),
    }
    if migration_receipt is not None:
        run_contract["migration"] = migration_receipt
    if source_contract is not None:
        if source_checkpoint_format == LEGACY_FORMAT_VERSION:
            if (
                _as_legacy_v3_contract(run_contract, source=source_contract)
                != source_contract
            ):
                raise ValueError(
                    "legacy v3 source differs beyond the audited token-chunk upgrade"
                )
        elif run_contract != source_contract:
            raise ValueError("v4 resume source run contract differs from this run")
    run_contract_sha256 = _canonical_json_sha256(run_contract)
    run_config = {
        **run_contract,
        "run_contract_sha256": run_contract_sha256,
        "output_dir": str(output_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "init_compressor": (
            str(args.init_compressor.resolve()) if args.init_compressor else None
        ),
    }
    global_step = 0
    cursor = QueueCursor(0, 0, 0, 0)
    if args.resume_from is not None:
        if source_run_contract_sha256 is None:
            raise RuntimeError("resume source contract hash was not resolved")
        global_step, cursor = _load_checkpoint(
            args.resume_from.resolve(),
            compressor=compressor,
            optimizer=optimizer,
            residency=residency,
            rank=rank,
            checkpoint_format_version=source_checkpoint_format,
            checkpoint_schema=source_checkpoint_schema,
            checkpoint_run_contract_sha256=source_run_contract_sha256,
        )
    if rank == 0:
        _preserve_checkpoint_run_configs(output_dir)
        _atomic_json(output_dir / "run_config.json", run_config)
    writer = (
        _open_tensorboard_writer(
            tensorboard_dir,
            resume_step=global_step if args.resume_from is not None else None,
        )
        if rank == 0
        else None
    )
    tensorboard_error: str | None = None
    initial_global_step = global_step
    additional_step_limit = (
        initial_global_step + args.max_additional_steps
        if args.max_additional_steps > 0
        else None
    )

    def step_limit_reached() -> bool:
        return (args.max_steps > 0 and global_step >= args.max_steps) or (
            additional_step_limit is not None and global_step >= additional_step_limit
        )

    accumulator = UpdateAccumulator()
    current_cursor = cursor
    stop = step_limit_reached()
    last_checkpoint: Path | None = None
    metrics_path = output_dir / "metrics.jsonl"
    started = time.monotonic()

    def complete_update() -> None:
        nonlocal accumulator, global_step, last_checkpoint, writer, tensorboard_error
        if accumulator.samples <= 0:
            return
        _normalize_gradients(compressor.parameters(), accumulator.samples)
        lr = _learning_rate(
            global_step, base=args.learning_rate, warmup=args.warmup_steps
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        grad_norm = float(compressor.clip_grad_norm_(1.0).item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError("compressor gradient norm is non-finite")
        _step_optimizer(optimizer, residency)
        global_step += 1
        parameters_finite = (
            _parameters_finite(compressor)
            if args.verify_parameters_finite_after_step
            else None
        )
        if parameters_finite is False:
            raise FloatingPointError("compressor parameters are non-finite after step")
        summary = {
            "global_step": global_step,
            "cursor": dataclasses.asdict(current_cursor),
            "learning_rate": lr,
            "grad_norm": grad_norm,
            "parameters_finite": parameters_finite,
            "elapsed_seconds": time.monotonic() - started,
            "gpu_allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "gpu_reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "moment_residency": residency.receipt(),
            **accumulator.means(),
        }
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(summary, sort_keys=True, allow_nan=False) + "\n"
                )
                handle.flush()
            if writer is not None:
                tensorboard_error = _write_tensorboard_summary(
                    writer,
                    summary,
                    global_step=global_step,
                )
                if tensorboard_error is not None:
                    print(
                        json.dumps(
                            {
                                "warning": "tensorboard_disabled_after_write_failure",
                                "global_step": global_step,
                                "error": tensorboard_error,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    writer = None
            if global_step % args.log_interval == 0:
                print(json.dumps(summary, sort_keys=True), flush=True)
        if args.save_interval and global_step % args.save_interval == 0:
            last_checkpoint = _save_checkpoint(
                output_dir,
                compressor=compressor,
                optimizer=optimizer,
                residency=residency,
                rank=rank,
                global_step=global_step,
                cursor=current_cursor,
                run_contract_sha256=run_contract_sha256,
            )
            _prune_checkpoints(
                output_dir,
                keep_last=args.keep_last_checkpoints,
                rank=rank,
            )
        accumulator = UpdateAccumulator()

    try:
        for epoch in range(cursor.epoch, args.epochs):
            if stop:
                break
            first_shard = cursor.shard_index if epoch == cursor.epoch else 0
            for shard_index in range(first_shard, len(queue.shards)):
                receipt = queue.shards[shard_index]
                _verify_receipt_distributed(receipt, queue.root, rank)
                first_event = (
                    cursor.event_index
                    if epoch == cursor.epoch and shard_index == cursor.shard_index
                    else 0
                )
                expected_base_events = planned_base_events_by_shard[shard_index]
                if first_event >= expected_base_events:
                    raise ValueError("checkpoint event cursor is outside its shard")
                events = iter_shard_repeat_events(
                    receipt,
                    data_root=queue.root,
                    repeat_rotation=epoch,
                    verify_receipt=False,
                    skip_empty_fields=True,
                )
                for event_index, (repeat_event, last_event_in_shard) in enumerate(
                    _with_last(events)
                ):
                    if event_index >= expected_base_events or last_event_in_shard != (
                        event_index + 1 == expected_base_events
                    ):
                        raise RuntimeError(
                            "runtime repeat event count differs from the frozen plan"
                        )
                    if event_index < first_event:
                        continue
                    repeat_chunks = deterministic_repeat_token_chunks(
                        processor.tokenizer,
                        repeat_event.text,
                        event_id=repeat_event.event_id,
                        maximum_source_tokens=(
                            args.source_block_tokens * args.max_source_blocks
                        ),
                    )
                    first_chunk = (
                        cursor.chunk_index
                        if (
                            epoch == cursor.epoch
                            and shard_index == cursor.shard_index
                            and event_index == cursor.event_index
                        )
                        else 0
                    )
                    if first_chunk >= len(repeat_chunks):
                        raise ValueError(
                            "checkpoint chunk cursor is outside its base event"
                        )
                    for repeat_chunk in repeat_chunks[first_chunk:]:
                        next_cursor = _next_cursor(
                            epoch=epoch,
                            shard_index=shard_index,
                            event_index=event_index,
                            chunk_index=repeat_chunk.chunk_index,
                            chunks_in_event=repeat_chunk.chunk_count,
                            last_event_in_shard=last_event_in_shard,
                            shard_count=len(queue.shards),
                        )
                        repeat_window = deterministic_token_window_from_ids(
                            repeat_chunk.token_ids,
                            event_id=repeat_chunk.event_id,
                            epoch=epoch,
                            window_tokens=args.target_window_tokens,
                            context_tokens=args.target_context_tokens,
                        )
                        identity = (
                            repeat_event.event_id,
                            repeat_chunk.chunk_index,
                            repeat_chunk.chunk_count,
                            repeat_chunk.token_start,
                            repeat_chunk.token_stop,
                            repeat_chunk.event_id,
                            repeat_window.start,
                            repeat_window.stop,
                        )
                        gathered: list[Any] = [None] * dist.get_world_size()
                        dist.all_gather_object(gathered, identity)
                        if any(item != identity for item in gathered):
                            raise RuntimeError(
                                "TP ranks selected different repeat token chunks"
                            )

                        residency.prepare_forward(optimizer)
                        repeat_rendered = render_qwen38_repeat_target(
                            processor,
                            source_token_ids=repeat_chunk.token_ids,
                            target_token_ids=repeat_window.target_ids,
                            target_prefix_token_ids=repeat_window.prefix_ids,
                        )
                        rendered_source_ids = tuple(
                            repeat_rendered.raw_model_inputs["input_ids"][
                                0,
                                repeat_rendered.source_span.start : repeat_rendered.source_span.stop,
                            ].tolist()
                        )
                        if rendered_source_ids != repeat_chunk.token_ids:
                            raise RuntimeError(
                                "rendered repeat source differs from its exact token chunk"
                            )
                        repeat_loss, repeat_metrics = _forward_objective(
                            repeat_rendered,
                            repeat_window,
                            target=target,
                            compressor=compressor,
                            block_tokens=args.source_block_tokens,
                            maximum_blocks=args.max_source_blocks,
                            compression_ratio=args.compression_ratio,
                            position_alpha=args.position_weight_alpha,
                            position_tau=args.position_weight_tau,
                            position_power=args.position_weight_power,
                            ce_chunk_size=args.ce_chunk_size,
                        )
                        repeat_loss.backward()
                        del repeat_loss, repeat_rendered
                        accumulator.add(repeat_metrics)
                        current_cursor = next_cursor
                        if accumulator.samples >= args.grad_accumulation:
                            complete_update()
                            if step_limit_reached():
                                stop = True
                                break
                    if stop:
                        break
                if stop:
                    break
            if accumulator.samples:
                complete_update()
                if step_limit_reached():
                    stop = True
            if stop:
                break

        if not args.skip_final_checkpoint and (
            last_checkpoint is None
            or last_checkpoint.name != f"checkpoint-{global_step:08d}"
        ):
            last_checkpoint = _save_checkpoint(
                output_dir,
                compressor=compressor,
                optimizer=optimizer,
                residency=residency,
                rank=rank,
                global_step=global_step,
                cursor=current_cursor,
                run_contract_sha256=run_contract_sha256,
            )
            _prune_checkpoints(
                output_dir,
                keep_last=args.keep_last_checkpoints,
                rank=rank,
            )
        rank_receipts: list[Any] = [None] * dist.get_world_size()
        local_receipt = {"rank": rank, **residency.receipt()}
        dist.all_gather_object(rank_receipts, local_receipt)
        if rank == 0:
            _atomic_json(
                output_dir / "result.json",
                {
                    "status": "PASS",
                    "global_step": global_step,
                    "cursor": dataclasses.asdict(current_cursor),
                    "run_contract_sha256": run_contract_sha256,
                    "last_checkpoint": (
                        str(last_checkpoint) if last_checkpoint else None
                    ),
                    "moment_residency_by_rank": rank_receipts,
                    "objective": "text_repeat_only",
                    "supervision": "bounded_position_weighted_hard_label_CE_only",
                    "tensorboard_error": tensorboard_error,
                    "parameters_finite_after_step": (
                        True if args.verify_parameters_finite_after_step else None
                    ),
                },
            )
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception as error:
                print(
                    json.dumps(
                        {
                            "warning": "tensorboard_close_failure_ignored",
                            "error": f"{type(error).__name__}: {error}",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        dist.destroy_process_group()
    return 0


def main() -> int:
    return train()


if __name__ == "__main__":
    raise SystemExit(main())
