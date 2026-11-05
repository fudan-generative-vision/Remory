"""Query-conditioned, variable-rate DFlash compression for GrepSeek.

Only the compressor is trainable.  A frozen Qwen3.5 target first encodes the
canonical (uncompressed) history with ``output_hidden_states=True``.  The
caller slices the rows belonging to each grep response and this module turns
those rows into ``ceil(evidence_tokens / compression_ratio)`` continuous
tokens.  The frozen target is then run a second time on a context in which the
grep responses have been replaced by those continuous tokens.

The compressor deliberately preserves the parameter shapes of the public
Qwen3.5-9B DFlash draft model: an eight-hidden-state concatenation is projected
by ``fc`` and consumed by six Qwen3 decoder layers (4096 hidden, 32/8 heads,
128 head dimension, 12288 FFN).  Its attention *layout* is compression-specific:
the first five layers read overlapping local evidence windows and neighbouring
slots; the sixth layer globally mixes valid slots without rereading the source.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from context_compression.losses import frozen_lm_head_chunked_token_ce


DEFAULT_QWEN35_TARGET_LAYER_IDS = (1, 5, 9, 13, 17, 21, 25, 29)
DEFAULT_QWEN36_27B_TARGET_LAYER_IDS = (1, 16, 31, 46, 61)
DEFAULT_QWEN36_35B_A3B_TARGET_LAYER_IDS = (1, 6, 11, 16, 22, 27, 32, 37)


def build_qwen35_9b_dflash_config(
    *,
    target_hidden_size: int = 4096,
    vocab_size: int = 248320,
    compression_ratio: int = 16,
) -> Qwen3Config:
    """Return the exact public Qwen3.5-9B-DFlash draft geometry.

    ``target_hidden_size`` controls only the source projection width.  The
    canonical draft hidden size is always 4096.  Tests and small experiments
    may instead pass a hand-written tiny :class:`Qwen3Config` to the compressor.
    """

    for name, value in (
        ("target_hidden_size", target_hidden_size),
        ("vocab_size", vocab_size),
        ("compression_ratio", compression_ratio),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=6,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=True,
        sliding_window=4096,
        max_window_layers=6,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
    )
    config.num_target_layers = 32
    config.target_hidden_size = int(target_hidden_size)
    config.block_size = int(compression_ratio)
    config.dflash_config = {
        "block_size": int(compression_ratio),
        "mask_token_id": 248077,
        "target_layer_ids": list(DEFAULT_QWEN35_TARGET_LAYER_IDS),
    }
    return config


def build_qwen36_27b_dflash_config(
    *,
    target_hidden_size: int = 5120,
    vocab_size: int = 248320,
    compression_ratio: int = 16,
) -> Qwen3Config:
    """Return the public Qwen3.6-27B-DFlash backbone geometry.

    The published draft checkpoint is used only as an architecture contract;
    context-compression experiments initialize every trainable tensor from
    scratch.  ``block_size`` is recorded as the nominal physical compression
    ratio while the number of output slots remains data dependent.
    """

    for name, value in (
        ("target_hidden_size", target_hidden_size),
        ("vocab_size", vocab_size),
        ("compression_ratio", compression_ratio),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=True,
        sliding_window=2048,
        max_window_layers=5,
        layer_types=["sliding_attention"] * 4 + ["full_attention"],
    )
    config.num_target_layers = 64
    config.target_hidden_size = int(target_hidden_size)
    config.block_size = int(compression_ratio)
    config.dflash_config = {
        "block_size": int(compression_ratio),
        "mask_token_id": 248070,
        "target_layer_ids": list(DEFAULT_QWEN36_27B_TARGET_LAYER_IDS),
    }
    return config


def build_qwen36_35b_a3b_dflash_config(
    *,
    target_hidden_size: int = 2048,
    vocab_size: int = 248320,
    compression_ratio: int = 16,
) -> Qwen3Config:
    """Return the public Qwen3.6-35B-A3B-DFlash backbone geometry.

    The matching z-lab checkpoint defines a dense six-layer Qwen3 draft even
    though its frozen target is MoE.  It is an architecture contract only:
    :class:`GrepSeekDFlashCompressor` initializes all trainable tensors from
    scratch and never loads the published draft weights.
    """

    for name, value in (
        ("target_hidden_size", target_hidden_size),
        ("vocab_size", vocab_size),
        ("compression_ratio", compression_ratio),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=2048,
        intermediate_size=6144,
        num_hidden_layers=6,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000.0,
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=True,
        sliding_window=4096,
        max_window_layers=6,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        bos_token_id=None,
        eos_token_id=248044,
        pad_token_id=None,
    )
    config.num_target_layers = 40
    config.target_hidden_size = int(target_hidden_size)
    config.block_size = int(compression_ratio)
    config.dflash_config = {
        "block_size": int(compression_ratio),
        "mask_token_id": 248077,
        "target_layer_ids": list(DEFAULT_QWEN36_35B_A3B_TARGET_LAYER_IDS),
    }
    return config


def extract_dflash_hidden_features(
    all_hidden_states: Sequence[torch.Tensor],
    target_layer_ids: Sequence[int] = DEFAULT_QWEN35_TARGET_LAYER_IDS,
) -> torch.Tensor:
    """Select target features with the native DFlash/HF layer convention.

    HF returns the embedding output at index zero, so DFlash layer id ``i`` is
    exactly ``all_hidden_states[i + 1]``.  This function intentionally does not
    accept the post-final-norm sentinel used by some unrelated dFlow variants.
    """

    if not isinstance(all_hidden_states, (tuple, list)):
        raise TypeError("all_hidden_states must be a tuple or list of tensors")
    if len(all_hidden_states) < 2:
        raise ValueError("target returned no decoder hidden states")
    ids = tuple(int(layer_id) for layer_id in target_layer_ids)
    if not ids:
        raise ValueError("target_layer_ids must not be empty")
    decoder_layers = len(all_hidden_states) - 1
    if any(layer_id < 0 or layer_id >= decoder_layers for layer_id in ids):
        raise ValueError(f"target_layer_ids must be in [0, {decoder_layers - 1}]")
    selected = [all_hidden_states[layer_id + 1] for layer_id in ids]
    reference_shape = selected[0].shape
    if any(state.shape != reference_shape for state in selected):
        raise ValueError("selected target hidden states have inconsistent shapes")
    return torch.cat(selected, dim=-1)


def _rotate_half(hidden: torch.Tensor) -> torch.Tensor:
    first, second = hidden.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    hidden: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return hidden * cos.unsqueeze(1) + _rotate_half(hidden) * sin.unsqueeze(1)


def _repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden
    batch, key_heads, length, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(
        batch, key_heads, repeats, length, head_dim
    )
    return hidden.reshape(batch, key_heads * repeats, length, head_dim)


class CompressionAttentionLayout(NamedTuple):
    """Masks and compact positions for one padded evidence batch."""

    local_keep_mask: torch.Tensor
    global_keep_mask: torch.Tensor
    slot_attention_mask: torch.Tensor
    slot_counts: torch.Tensor
    source_position_ids: torch.Tensor
    slot_position_ids: torch.Tensor


def build_local_compression_layout(
    source_attention_mask: torch.Tensor,
    *,
    compression_ratio: int = 16,
    local_window: int = 256,
    source_token_lengths: Optional[torch.Tensor] = None,
    slot_token_lengths: Optional[torch.Tensor] = None,
) -> CompressionAttentionLayout:
    """Build overlapping local-read and global-slot masks.

    Source padding may occur anywhere: locality is defined over the compact
    rank of valid evidence tokens.  Slots are left padded in the returned
    batch, and invalid query rows are allowed to read only their own slot key.
    That diagonal escape hatch prevents all-masked SDPA rows; the compressor
    explicitly zeros those rows after every decoder layer.
    """

    if not isinstance(source_attention_mask, torch.Tensor):
        raise TypeError("source_attention_mask must be a torch.Tensor")
    if source_attention_mask.ndim != 2:
        raise ValueError("source_attention_mask must have shape [batch, source]")
    for name, value in (
        ("compression_ratio", compression_ratio),
        ("local_window", local_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if local_window < compression_ratio:
        raise ValueError(
            "local_window must be at least compression_ratio so every source "
            "token is visible to its compression slot"
        )

    source_mask = source_attention_mask.bool()
    device = source_mask.device
    batch, source_width = source_mask.shape
    source_span_lengths = source_mask.sum(dim=-1)
    if source_token_lengths is None:
        source_lengths = source_span_lengths
    else:
        if (
            not isinstance(source_token_lengths, torch.Tensor)
            or source_token_lengths.ndim != 1
            or source_token_lengths.shape[0] != batch
        ):
            raise ValueError("source_token_lengths must have shape [batch]")
        if source_token_lengths.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("source_token_lengths must use an integer dtype")
        if source_token_lengths.device != device:
            raise ValueError("source_token_lengths and source mask must share a device")
        source_lengths = source_token_lengths.long()
        if bool((source_lengths < 0).any().item()):
            raise ValueError("source_token_lengths must be nonnegative")
        if bool(((source_lengths == 0) != (source_span_lengths == 0)).any().item()):
            raise ValueError(
                "source_token_lengths and valid source spans must be empty together"
            )
    if slot_token_lengths is None:
        slot_lengths = source_lengths
    else:
        if (
            not isinstance(slot_token_lengths, torch.Tensor)
            or slot_token_lengths.ndim != 1
            or slot_token_lengths.shape[0] != batch
        ):
            raise ValueError("slot_token_lengths must have shape [batch]")
        if slot_token_lengths.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("slot_token_lengths must use an integer dtype")
        if slot_token_lengths.device != device:
            raise ValueError(
                "slot_token_lengths and source mask must share a device"
            )
        slot_lengths = slot_token_lengths.long()
        if bool((slot_lengths < source_lengths).any().item()):
            raise ValueError(
                "slot_token_lengths cannot be shorter than source_token_lengths"
            )
        if bool(((slot_lengths == 0) != (source_lengths == 0)).any().item()):
            raise ValueError(
                "slot and source token lengths must be empty together"
            )
        if bool((slot_lengths > source_lengths).any().item()) and local_window < 2048:
            raise ValueError(
                "padded slot allocation requires local_window >= 2048"
            )
    slot_counts = torch.div(
        slot_lengths + compression_ratio - 1,
        compression_ratio,
        rounding_mode="floor",
    )
    slot_width = int(slot_counts.max().item()) if batch else 0

    source_ranks = source_mask.long().cumsum(dim=-1) - 1
    source_ranks.masked_fill_(~source_mask, 0)
    # JSON/chat-template escaping can make the encoded source span wider than
    # the canonical raw stdout used to define the compression ratio.  Map each
    # feature-row centre into canonical-token coordinates so locality and RoPE
    # remain tied to the actual evidence rather than serialization overhead.
    span_denominator = source_span_lengths.clamp_min(1).to(torch.float32)[:, None]
    canonical_scale = source_lengths.to(torch.float32)[:, None] / span_denominator
    source_coordinates = (source_ranks.to(torch.float32) + 0.5) * canonical_scale - 0.5
    source_coordinates.masked_fill_(~source_mask, 0.0)
    source_positions = source_coordinates.round().long()
    maximum_position = (source_lengths - 1).clamp_min(0)[:, None]
    source_positions = torch.minimum(source_positions.clamp_min(0), maximum_position)
    source_positions.masked_fill_(~source_mask, 0)
    if slot_width == 0:
        empty_slots = torch.zeros((batch, 0), dtype=torch.bool, device=device)
        return CompressionAttentionLayout(
            local_keep_mask=torch.zeros(
                (batch, 1, 0, source_width), dtype=torch.bool, device=device
            ),
            global_keep_mask=torch.zeros(
                (batch, 1, 0, 0), dtype=torch.bool, device=device
            ),
            slot_attention_mask=empty_slots,
            slot_counts=slot_counts,
            source_position_ids=source_positions,
            slot_position_ids=torch.zeros((batch, 0), dtype=torch.long, device=device),
        )

    padded_slot_index = torch.arange(slot_width, device=device)[None, :]
    left_padding = slot_width - slot_counts[:, None]
    slot_rank = padded_slot_index - left_padding
    slot_mask = slot_rank >= 0

    # Each slot is anchored at the centre of the evidence block it represents.
    # In particular, centre the final partial block over its *actual* extent.
    # Clipping a full-stride centre to the final token would move a 1:1024 slot
    # hundreds of positions away from a short source and can leave it with no
    # source keys at all.
    block_start = slot_rank.clamp_min(0) * compression_ratio
    final_position = (slot_lengths - 1).clamp_min(0)[:, None]
    block_end = torch.minimum(
        block_start + compression_ratio - 1,
        final_position,
    )
    anchor_float_2d = (
        block_start.to(torch.float32) + block_end.to(torch.float32)
    ) / 2.0
    slot_positions = torch.floor(anchor_float_2d + 0.5).long()
    slot_positions.masked_fill_(~slot_mask, 0)

    half_window = local_window / 2.0
    source_coordinate = source_coordinates[:, None, :]
    anchor_float = anchor_float_2d[:, :, None]
    source_local = (
        source_mask[:, None, :]
        & slot_mask[:, :, None]
        & ((source_coordinate - anchor_float).abs() < half_window)
    )

    # Neighbouring slots exchange information within the same raw-token
    # receptive field.  The sixth layer later performs the sole global mix.
    query_anchor = anchor_float
    key_anchor = anchor_float.transpose(1, 2)
    slot_local = (
        slot_mask[:, :, None]
        & slot_mask[:, None, :]
        & ((query_anchor - key_anchor).abs() < half_window)
    )
    local_keep = torch.cat((source_local, slot_local), dim=-1).unsqueeze(1)

    global_keep = (slot_mask[:, :, None] & slot_mask[:, None, :]).unsqueeze(1)

    # Safe self-unmask for invalid padded query rows.
    diagonal = torch.eye(slot_width, dtype=torch.bool, device=device)[None, :, :]
    invalid_diagonal = (~slot_mask)[:, :, None] & diagonal
    local_keep[:, :, :, source_width:] |= invalid_diagonal.unsqueeze(1)
    global_keep |= invalid_diagonal.unsqueeze(1)

    return CompressionAttentionLayout(
        local_keep_mask=local_keep,
        global_keep_mask=global_keep,
        slot_attention_mask=slot_mask,
        slot_counts=slot_counts,
        source_position_ids=source_positions,
        slot_position_ids=slot_positions,
    )


class GrepSeekDFlashAttention(nn.Module):
    """Native-shape DFlash attention with an explicit compression mask."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_attention_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.num_attention_heads)
        )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        bias = bool(getattr(config, "attention_bias", False))
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        eps = float(config.rms_norm_eps)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_hidden: Optional[torch.Tensor],
        keep_mask: torch.Tensor,
        slot_cos: torch.Tensor,
        slot_sin: torch.Tensor,
        source_cos: Optional[torch.Tensor],
        source_sin: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, slots, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, slots, self.num_attention_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        query = _apply_rotary(query, slot_cos, slot_sin)

        slot_key = self.k_proj(hidden_states).view(
            batch, slots, self.num_key_value_heads, self.head_dim
        )
        slot_key = self.k_norm(slot_key).transpose(1, 2)
        slot_key = _apply_rotary(slot_key, slot_cos, slot_sin)
        slot_value = (
            self.v_proj(hidden_states)
            .view(batch, slots, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        if source_hidden is None:
            key = slot_key
            value = slot_value
        else:
            source_width = source_hidden.shape[1]
            source_key = self.k_proj(source_hidden).view(
                batch, source_width, self.num_key_value_heads, self.head_dim
            )
            source_key = self.k_norm(source_key).transpose(1, 2)
            if source_cos is None or source_sin is None:
                raise ValueError("source RoPE tensors are required with source_hidden")
            source_key = _apply_rotary(source_key, source_cos, source_sin)
            source_value = (
                self.v_proj(source_hidden)
                .view(batch, source_width, self.num_key_value_heads, self.head_dim)
                .transpose(1, 2)
            )
            key = torch.cat((source_key, slot_key), dim=-2)
            value = torch.cat((source_value, slot_value), dim=-2)

        key = _repeat_kv(key, self.num_key_value_groups)
        value = _repeat_kv(value, self.num_key_value_groups)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=keep_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
            scale=self.head_dim**-0.5,
        )
        attended = attended.transpose(1, 2).reshape(batch, slots, -1)
        return self.o_proj(attended)


class GrepSeekDFlashDecoderLayer(nn.Module):
    """A parameter-shape-isomorphic Qwen3 DFlash decoder layer."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = GrepSeekDFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        source_hidden: Optional[torch.Tensor],
        keep_mask: torch.Tensor,
        slot_cos: torch.Tensor,
        slot_sin: torch.Tensor,
        source_cos: Optional[torch.Tensor],
        source_sin: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            normalized,
            source_hidden,
            keep_mask,
            slot_cos,
            slot_sin,
            source_cos,
            source_sin,
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class CompressedContextOutput(NamedTuple):
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    slot_counts: torch.Tensor


class GrepSeekDFlashCompressor(nn.Module):
    """Variable-length 1:N compressor with native DFlash backbone shapes."""

    def __init__(
        self,
        draft_config: Qwen3Config,
        *,
        target_hidden_size: int,
        target_layer_ids: Sequence[int] = DEFAULT_QWEN35_TARGET_LAYER_IDS,
        compression_ratio: int = 16,
        local_window: int = 256,
        output_embedding_rms: float = 0.02,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        num_layers = int(draft_config.num_hidden_layers)
        if num_layers < 2:
            raise ValueError("the DFlash compressor must have at least two layers")
        if int(draft_config.hidden_size) != int(target_hidden_size):
            raise ValueError(
                "draft and target hidden sizes must match for continuous-token recovery"
            )
        ids = tuple(int(layer_id) for layer_id in target_layer_ids)
        if not ids or len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("target_layer_ids must be unique nonnegative integers")
        for name, value in (
            ("target_hidden_size", target_hidden_size),
            ("compression_ratio", compression_ratio),
            ("local_window", local_window),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if local_window < compression_ratio:
            raise ValueError(
                "local_window must be at least compression_ratio so every source "
                "token is visible to its compression slot"
            )
        if not math.isfinite(float(output_embedding_rms)) or output_embedding_rms <= 0:
            raise ValueError("output_embedding_rms must be finite and positive")

        self.config = copy.deepcopy(draft_config)
        self.target_hidden_size = int(target_hidden_size)
        self.target_layer_ids = ids
        self.compression_ratio = int(compression_ratio)
        self.local_window = int(local_window)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        hidden_size = int(self.config.hidden_size)

        # These names and matrix shapes match native DFlash exactly.
        self.fc = nn.Linear(len(ids) * self.target_hidden_size, hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(
            hidden_size, eps=float(self.config.rms_norm_eps)
        )
        self.layers = nn.ModuleList(
            [
                GrepSeekDFlashDecoderLayer(self.config, layer_idx)
                for layer_idx in range(num_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(hidden_size, eps=float(self.config.rms_norm_eps))
        self.rotary_emb = Qwen3RotaryEmbedding(self.config)

        # Native DFlash receives repeated target MASK embeddings.  Here the
        # same single seed is trainable because only the compressor is trained;
        # RoPE at evidence anchors breaks the otherwise exact symmetry.
        self.mask_embedding = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.register_buffer(
            "output_embedding_rms",
            torch.tensor(float(output_embedding_rms), dtype=torch.float32),
            persistent=True,
        )

        initializer_range = float(getattr(self.config, "initializer_range", 0.02))
        self.apply(lambda module: self._initialize_module(module, initializer_range))
        nn.init.normal_(self.mask_embedding, mean=0.0, std=initializer_range)

    @staticmethod
    def _initialize_module(module: nn.Module, std: float) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @classmethod
    def canonical_qwen35_9b(
        cls,
        *,
        output_embedding_rms: float = 0.02,
        compression_ratio: int = 16,
        local_window: int = 256,
        gradient_checkpointing: bool = True,
    ) -> "GrepSeekDFlashCompressor":
        config = build_qwen35_9b_dflash_config(
            target_hidden_size=4096,
            compression_ratio=compression_ratio,
        )
        return cls(
            config,
            target_hidden_size=4096,
            compression_ratio=compression_ratio,
            local_window=local_window,
            output_embedding_rms=output_embedding_rms,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    def canonical_qwen36_27b(
        cls,
        *,
        output_embedding_rms: float = 0.02,
        compression_ratio: int = 16,
        local_window: int = 2048,
        gradient_checkpointing: bool = True,
    ) -> "GrepSeekDFlashCompressor":
        config = build_qwen36_27b_dflash_config(
            target_hidden_size=5120,
            compression_ratio=compression_ratio,
        )
        return cls(
            config,
            target_hidden_size=5120,
            target_layer_ids=DEFAULT_QWEN36_27B_TARGET_LAYER_IDS,
            compression_ratio=compression_ratio,
            local_window=local_window,
            output_embedding_rms=output_embedding_rms,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    def canonical_qwen36_35b_a3b(
        cls,
        *,
        output_embedding_rms: float = 0.02,
        compression_ratio: int = 16,
        local_window: int = 4096,
        gradient_checkpointing: bool = True,
    ) -> "GrepSeekDFlashCompressor":
        """Build a randomly initialized Qwen3.6-35B-A3B DFlash compressor."""

        config = build_qwen36_35b_a3b_dflash_config(
            target_hidden_size=2048,
            compression_ratio=compression_ratio,
        )
        return cls(
            config,
            target_hidden_size=2048,
            target_layer_ids=DEFAULT_QWEN36_35B_A3B_TARGET_LAYER_IDS,
            compression_ratio=compression_ratio,
            local_window=local_window,
            output_embedding_rms=output_embedding_rms,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(
        self,
        source_features: torch.Tensor,
        source_attention_mask: torch.Tensor,
        source_token_lengths: Optional[torch.Tensor] = None,
        slot_token_lengths: Optional[torch.Tensor] = None,
    ) -> CompressedContextOutput:
        if source_features.ndim != 3 or not torch.is_floating_point(source_features):
            raise ValueError(
                "source_features must be floating [segments, source, selected_hidden]"
            )
        if source_attention_mask.shape != source_features.shape[:2]:
            raise ValueError("source feature and attention-mask shapes differ")
        expected_width = len(self.target_layer_ids) * self.target_hidden_size
        if source_features.shape[-1] != expected_width:
            raise ValueError(
                f"source feature width must be {expected_width}, got "
                f"{source_features.shape[-1]}"
            )
        if source_features.device != self.fc.weight.device:
            raise ValueError("source features and compressor must share a device")

        source_features = source_features.to(dtype=self.fc.weight.dtype)
        layout = build_local_compression_layout(
            source_attention_mask,
            compression_ratio=self.compression_ratio,
            local_window=self.local_window,
            source_token_lengths=source_token_lengths,
            slot_token_lengths=slot_token_lengths,
        )
        batch, slot_width = layout.slot_attention_mask.shape
        if slot_width == 0:
            empty = source_features.new_zeros((batch, 0, int(self.config.hidden_size)))
            return CompressedContextOutput(
                empty, layout.slot_attention_mask, layout.slot_counts
            )

        source_hidden = self.hidden_norm(self.fc(source_features))
        hidden_states = self.mask_embedding.expand(batch, slot_width, -1)
        slot_cos, slot_sin = self.rotary_emb(hidden_states, layout.slot_position_ids)
        source_cos, source_sin = self.rotary_emb(
            source_hidden, layout.source_position_ids
        )
        slot_row_mask = layout.slot_attention_mask.unsqueeze(-1)

        # Local overlapping reads, exactly followed by one global slot-only
        # mixer. Invalid rows are zero after every residual block.  This maps
        # public 6-layer Qwen3.5/Qwen3.6-35B-A3B drafts to 5+1 and the public
        # 5-layer Qwen3.6-27B draft to 4+1 without changing parameter geometry.
        for layer in self.layers[:-1]:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    source_hidden,
                    layout.local_keep_mask,
                    slot_cos,
                    slot_sin,
                    source_cos,
                    source_sin,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    source_hidden,
                    layout.local_keep_mask,
                    slot_cos,
                    slot_sin,
                    source_cos,
                    source_sin,
                )
            hidden_states = hidden_states * slot_row_mask

        final_layer = self.layers[-1]
        if self.gradient_checkpointing and self.training:
            hidden_states = checkpoint(
                final_layer,
                hidden_states,
                None,
                layout.global_keep_mask,
                slot_cos,
                slot_sin,
                None,
                None,
                use_reentrant=False,
            )
        else:
            hidden_states = final_layer(
                hidden_states,
                None,
                layout.global_keep_mask,
                slot_cos,
                slot_sin,
                None,
                None,
            )
        hidden_states = hidden_states * slot_row_mask
        scale = self.output_embedding_rms.to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        embeddings = self.norm(hidden_states) * scale
        embeddings = embeddings * slot_row_mask
        return CompressedContextOutput(
            embeddings=embeddings,
            attention_mask=layout.slot_attention_mask,
            slot_counts=layout.slot_counts,
        )

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": type(self).__name__,
            "target_layer_ids": list(self.target_layer_ids),
            "target_hidden_size": self.target_hidden_size,
            "selected_source_width": len(self.target_layer_ids)
            * self.target_hidden_size,
            "compression_ratio": self.compression_ratio,
            "local_window": self.local_window,
            "hidden_size": int(self.config.hidden_size),
            "intermediate_size": int(self.config.intermediate_size),
            "num_hidden_layers": int(self.config.num_hidden_layers),
            "num_attention_heads": int(self.config.num_attention_heads),
            "num_key_value_heads": int(self.config.num_key_value_heads),
            "head_dim": int(self.config.head_dim),
            "layer_types": ["sliding_attention"]
            * (int(self.config.num_hidden_layers) - 1)
            + ["full_attention"],
            "gradient_checkpointing": self.gradient_checkpointing,
            "output_embedding_rms": float(self.output_embedding_rms.item()),
            "slot_padding_side": "left",
        }


@dataclass(frozen=True)
class RecoveryLossOutput:
    loss: torch.Tensor
    token_accuracy: torch.Tensor
    token_count: torch.Tensor
    prefix_width: int


class GrepSeekGenerationOutput(NamedTuple):
    """Greedy continuation ids plus each sample's first-stop length."""

    token_ids: torch.Tensor
    lengths: torch.Tensor
    finished: torch.Tensor


def freeze_target_model(target_model: nn.Module) -> nn.Module:
    """Freeze parameters while retaining autograd through continuous inputs."""

    target_model.requires_grad_(False)
    for parameter in target_model.parameters():
        parameter.grad = None
    return target_model


class GrepSeekFrozenTargetRuntime:
    """Frozen Qwen3.5 source encoding and differentiable student recovery."""

    def __init__(
        self,
        target_model: nn.Module,
        *,
        target_layer_ids: Sequence[int] = DEFAULT_QWEN35_TARGET_LAYER_IDS,
        gradient_checkpointing: bool = True,
    ) -> None:
        self.target = freeze_target_model(target_model)
        self.target_layer_ids = tuple(int(i) for i in target_layer_ids)
        if not self.target_layer_ids:
            raise ValueError("target_layer_ids must not be empty")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            try:
                self.target.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.target.gradient_checkpointing_enable()
            self.target.train()
        else:
            self.target.eval()

    @property
    def text_config(self) -> Any:
        return getattr(self.target.config, "text_config", self.target.config)

    @property
    def backbone(self) -> nn.Module:
        model = getattr(self.target, "model", None)
        if model is None:
            raise TypeError("target model must expose `.model`")
        # Qwen3.5 ForConditionalGeneration owns a multimodal Qwen3_5Model;
        # its text-only decoder is `.model.language_model`.  Text-only causal
        # LMs expose their decoder directly as `.model`.
        return getattr(model, "language_model", model)

    @property
    def lm_head(self) -> nn.Module:
        head = None
        getter = getattr(self.target, "get_output_embeddings", None)
        if callable(getter):
            head = getter()
        if head is None:
            head = getattr(self.target, "lm_head", None)
        if not isinstance(head, nn.Module):
            raise TypeError("target model has no frozen lm_head")
        return head

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.target.get_input_embeddings()
        if not isinstance(embedding, nn.Module):
            raise TypeError("target model has no input embedding")
        return embedding(input_ids)

    def encode_source(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a full canonical raw history once, without source gradients."""

        if (
            source_input_ids.ndim != 2
            or source_input_ids.shape != source_attention_mask.shape
        ):
            raise ValueError("source ids and mask must have the same rank-2 shape")
        mask = source_attention_mask.bool()
        positions = mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(~mask, 0)
        was_training = self.target.training
        self.target.eval()
        try:
            with torch.no_grad():
                output = self.backbone(
                    input_ids=source_input_ids,
                    attention_mask=mask,
                    position_ids=positions,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            self.target.train(was_training)
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("frozen target did not return output_hidden_states")
        return extract_dflash_hidden_features(
            hidden_states, self.target_layer_ids
        ).detach()

    def _run_recovery(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(~attention_mask, 0)
        output = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=positions,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]
        return hidden

    def token_loss_interleaved(
        self,
        compressed_embeddings: torch.Tensor,
        compressed_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        memory_positions: torch.Tensor,
        *,
        chunk_size: int = 256,
        ignore_index: int = -100,
    ) -> RecoveryLossOutput:
        """CE for histories containing any number of compressed grep segments.

        ``memory_positions`` is ``[valid_slots, 2]`` containing ``(batch,
        sequence_position)`` rows.  Its order must be segment-major and then
        valid-slot order; boolean flattening of the left-padded compressed batch
        is scattered into those rows.  All target operations remain grad-enabled
        even though every target parameter is frozen.
        """

        if compressed_embeddings.ndim != 3:
            raise ValueError("compressed_embeddings must be [segments, slots, hidden]")
        if compressed_attention_mask.shape != compressed_embeddings.shape[:2]:
            raise ValueError("compressed embedding and mask shapes differ")
        if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input ids and attention mask must have equal rank-2 shapes"
            )
        if labels.shape != input_ids.shape:
            raise ValueError("labels and input ids shapes differ")
        if input_ids.shape[1] < 2:
            raise ValueError("recovery sequence must contain at least two tokens")
        if compressed_embeddings.shape[-1] != int(self.text_config.hidden_size):
            raise ValueError("compressed embedding hidden size does not match target")
        if memory_positions.ndim != 2 or memory_positions.shape[1] != 2:
            raise ValueError("memory_positions must have shape [valid_slots, 2]")
        if memory_positions.dtype not in {torch.int32, torch.int64}:
            raise TypeError("memory_positions must use an integer dtype")
        if memory_positions.device != input_ids.device:
            raise ValueError(
                "memory_positions and recovery tensors must share a device"
            )
        memory_mask = compressed_attention_mask.bool()
        valid_memory = compressed_embeddings[memory_mask]
        if memory_positions.shape[0] != valid_memory.shape[0]:
            raise ValueError(
                "memory_positions count must equal the number of valid compressed slots"
            )

        batch_positions = memory_positions[:, 0].long()
        token_positions = memory_positions[:, 1].long()
        batch, sequence = input_ids.shape
        if memory_positions.numel() and bool(
            (
                (batch_positions < 0)
                | (batch_positions >= batch)
                | (token_positions < 0)
                | (token_positions >= sequence)
            )
            .any()
            .item()
        ):
            raise ValueError("memory_positions contains an out-of-range coordinate")
        if memory_positions.numel():
            linear = batch_positions * sequence + token_positions
            if torch.unique(linear).numel() != linear.numel():
                raise ValueError("memory_positions must be unique")
            if not bool(
                attention_mask.bool()[batch_positions, token_positions].all().item()
            ):
                raise ValueError("every memory position must be an attended token")

        inputs_embeds = self._embed(input_ids).clone()
        if valid_memory.numel():
            inputs_embeds[batch_positions, token_positions] = valid_memory.to(
                dtype=inputs_embeds.dtype
            )
        hidden = self._run_recovery(inputs_embeds, attention_mask.bool())
        # Standard causal-LM alignment: row t predicts label t+1.
        ce = frozen_lm_head_chunked_token_ce(
            hidden[:, :-1, :],
            labels[:, 1:],
            self.lm_head,
            prefix_width=1,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )
        return RecoveryLossOutput(
            loss=ce.loss,
            token_accuracy=ce.token_accuracy,
            token_count=ce.token_count,
            prefix_width=0,
        )

    # Alias chosen so training code can read naturally.
    interleaved_token_loss = token_loss_interleaved

    def token_loss(
        self,
        compressed_embeddings: torch.Tensor,
        compressed_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_labels: torch.Tensor,
        *,
        chunk_size: int = 256,
        ignore_index: int = -100,
    ) -> RecoveryLossOutput:
        """CE for a left-padded compressed prefix followed by gold tokens."""

        if compressed_embeddings.ndim != 3:
            raise ValueError("compressed_embeddings must be [batch, slots, hidden]")
        if compressed_attention_mask.shape != compressed_embeddings.shape[:2]:
            raise ValueError("compressed embedding and mask shapes differ")
        if target_input_ids.ndim != 2 or target_input_ids.shape != target_labels.shape:
            raise ValueError("target ids and labels must have equal rank-2 shapes")
        if target_input_ids.shape[0] != compressed_embeddings.shape[0]:
            raise ValueError("memory and target batch sizes differ")
        if target_input_ids.shape[1] < 1:
            raise ValueError("target sequence must not be empty")
        if not bool(compressed_attention_mask.bool()[:, -1].all().item()):
            raise ValueError("every compressed prefix must contain a valid final slot")

        target_mask = target_labels != int(ignore_index)
        target_embeddings = self._embed(target_input_ids)
        inputs_embeds = torch.cat(
            (
                compressed_embeddings.to(dtype=target_embeddings.dtype),
                target_embeddings,
            ),
            dim=1,
        )
        full_mask = torch.cat((compressed_attention_mask.bool(), target_mask), dim=1)
        hidden = self._run_recovery(inputs_embeds, full_mask)
        prefix_width = int(compressed_embeddings.shape[1])
        ce = frozen_lm_head_chunked_token_ce(
            hidden,
            target_labels,
            self.lm_head,
            prefix_width=prefix_width,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )
        return RecoveryLossOutput(
            loss=ce.loss,
            token_accuracy=ce.token_accuracy,
            token_count=ce.token_count,
            prefix_width=prefix_width,
        )

    def _greedy_from_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int],
    ) -> GrepSeekGenerationOutput:
        """Greedy decode one fresh recovery cache from a soft-token prefill."""

        if inputs_embeds.ndim != 3 or not torch.is_floating_point(inputs_embeds):
            raise ValueError(
                "generation inputs must be floating [batch, prefix, hidden]"
            )
        if attention_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError("generation embeddings and attention mask shapes differ")
        if inputs_embeds.shape[1] <= 0:
            raise ValueError("generation prefix must not be empty")
        if inputs_embeds.shape[-1] != int(self.text_config.hidden_size):
            raise ValueError("generation embedding hidden size does not match target")
        if isinstance(max_new_tokens, bool) or int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if not bool(attention_mask.bool()[:, -1].all().item()):
            raise ValueError("every generation prefix must end in an attended token")

        attention_mask = attention_mask.bool()
        batch, prefix_width = attention_mask.shape
        positions = attention_mask.long().cumsum(dim=-1) - 1
        positions.masked_fill_(~attention_mask, 0)
        # cache_position indexes physical cache columns, while position_ids
        # remain compact per sample under left padding.  Qwen3.5's hybrid
        # attention/linear-attention cache requires keeping these concepts
        # distinct for variable-length batches.
        cache_position = torch.arange(
            prefix_width, device=inputs_embeds.device, dtype=torch.long
        )

        was_training = self.target.training
        self.target.eval()
        generated: list[torch.Tensor] = []
        finished = torch.zeros(batch, dtype=torch.bool, device=inputs_embeds.device)
        lengths = torch.zeros(batch, dtype=torch.long, device=inputs_embeds.device)
        try:
            output = self.target(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=positions,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            recovery_cache = output.past_key_values
            if recovery_cache is None:
                raise RuntimeError("target prefill did not return a recovery KV cache")
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if eos_token_id is not None:
                just_finished = next_token.squeeze(-1) == int(eos_token_id)
                lengths[just_finished] = 1
                finished |= just_finished

            for generated_index in range(1, int(max_new_tokens)):
                if bool(finished.all().item()):
                    break
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (batch, 1),
                            dtype=torch.bool,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                next_position = attention_mask.sum(dim=-1, keepdim=True) - 1
                cache_position = torch.tensor(
                    [prefix_width + generated_index - 1],
                    dtype=torch.long,
                    device=attention_mask.device,
                )
                output = self.target(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    position_ids=next_position,
                    cache_position=cache_position,
                    past_key_values=recovery_cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                recovery_cache = output.past_key_values
                if recovery_cache is None:
                    raise RuntimeError("target decode step dropped its recovery cache")
                next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                if eos_token_id is not None:
                    next_token = torch.where(
                        finished[:, None],
                        torch.full_like(next_token, int(eos_token_id)),
                        next_token,
                    )
                    just_finished = ~finished & (
                        next_token.squeeze(-1) == int(eos_token_id)
                    )
                    lengths[just_finished] = generated_index + 1
                    finished |= just_finished
                generated.append(next_token)
        finally:
            self.target.train(was_training)

        token_ids = torch.cat(generated, dim=1)
        lengths.masked_fill_(~finished, token_ids.shape[1])
        return GrepSeekGenerationOutput(
            token_ids=token_ids,
            lengths=lengths,
            finished=finished,
        )

    @torch.inference_mode()
    def greedy_generate_interleaved(
        self,
        compressed_embeddings: torch.Tensor,
        compressed_attention_mask: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        memory_positions: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int],
    ) -> GrepSeekGenerationOutput:
        """Generate after scattering ragged soft memories into a prefix."""

        if compressed_embeddings.ndim != 3:
            raise ValueError("compressed_embeddings must be [segments, slots, hidden]")
        if compressed_attention_mask.shape != compressed_embeddings.shape[:2]:
            raise ValueError("compressed embedding and mask shapes differ")
        if (
            prefix_input_ids.ndim != 2
            or prefix_input_ids.shape != prefix_attention_mask.shape
        ):
            raise ValueError(
                "prefix ids and attention mask must have equal rank-2 shapes"
            )
        if memory_positions.ndim != 2 or memory_positions.shape[1] != 2:
            raise ValueError("memory_positions must have shape [valid_slots, 2]")
        if memory_positions.device != prefix_input_ids.device:
            raise ValueError("memory positions and prefix tensors must share a device")
        valid_memory = compressed_embeddings[compressed_attention_mask.bool()]
        if valid_memory.shape[0] != memory_positions.shape[0]:
            raise ValueError("valid memory and scatter position counts differ")
        batch_positions = memory_positions[:, 0].long()
        token_positions = memory_positions[:, 1].long()
        batch, width = prefix_input_ids.shape
        if memory_positions.numel() and bool(
            (
                (batch_positions < 0)
                | (batch_positions >= batch)
                | (token_positions < 0)
                | (token_positions >= width)
            )
            .any()
            .item()
        ):
            raise ValueError("memory_positions contains an out-of-range coordinate")
        if memory_positions.numel():
            linear = batch_positions * width + token_positions
            if torch.unique(linear).numel() != linear.numel():
                raise ValueError("memory_positions must be unique")
            if not bool(
                prefix_attention_mask.bool()[batch_positions, token_positions]
                .all()
                .item()
            ):
                raise ValueError("every memory position must be attended")
        inputs_embeds = self._embed(prefix_input_ids).clone()
        if valid_memory.numel():
            inputs_embeds[batch_positions, token_positions] = valid_memory.to(
                dtype=inputs_embeds.dtype
            )
        return self._greedy_from_embeddings(
            inputs_embeds,
            prefix_attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

    @torch.inference_mode()
    def greedy_generate_prompt(
        self,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int],
    ) -> GrepSeekGenerationOutput:
        """Generate from a fully discrete (oracle) prompt with a fresh cache."""

        if (
            prefix_input_ids.ndim != 2
            or prefix_input_ids.shape != prefix_attention_mask.shape
        ):
            raise ValueError(
                "prefix ids and attention mask must have equal rank-2 shapes"
            )
        return self._greedy_from_embeddings(
            self._embed(prefix_input_ids),
            prefix_attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )


__all__ = [
    "DEFAULT_QWEN35_TARGET_LAYER_IDS",
    "DEFAULT_QWEN36_27B_TARGET_LAYER_IDS",
    "DEFAULT_QWEN36_35B_A3B_TARGET_LAYER_IDS",
    "CompressedContextOutput",
    "CompressionAttentionLayout",
    "GrepSeekDFlashAttention",
    "GrepSeekDFlashCompressor",
    "GrepSeekDFlashDecoderLayer",
    "GrepSeekFrozenTargetRuntime",
    "GrepSeekGenerationOutput",
    "RecoveryLossOutput",
    "build_local_compression_layout",
    "build_qwen35_9b_dflash_config",
    "build_qwen36_27b_dflash_config",
    "build_qwen36_35b_a3b_dflash_config",
    "extract_dflash_hidden_features",
    "freeze_target_model",
]
