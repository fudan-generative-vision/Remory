"""Strict tensor layout for compressing a fixed-width source into 16 queries."""

from __future__ import annotations

from typing import NamedTuple

import torch

# Kept as the legacy/default width so existing callers and 1024-token
# checkpoints remain valid. New compressors pass their configured width
# explicitly to the layout builder.
SOURCE_KV_WIDTH = 1024
COMPRESSED_QUERY_COUNT = 16


class DFlashCompressionLayout(NamedTuple):
    """DFlash keep mask and full source-plus-query position IDs."""

    keep_mask: torch.Tensor
    position_ids: torch.Tensor


def build_dflash_16_query_layout(
    source_attention_mask: torch.Tensor,
    *,
    source_tokens: int = SOURCE_KV_WIDTH,
) -> DFlashCompressionLayout:
    """Build the one-block DFlash layout for right-padded source KV.

    ``source_attention_mask`` uses ``True == keep`` and must have shape
    ``[batch, source_tokens]``. Every one of the 16 queries sees the valid
    source prefix and the complete 16-query block. Query positions start at
    each sample's unpadded source length, while the physical source width
    remains fixed within one compressor/run for batching.

    The legacy default is 1024 source tokens. A 256-token compressor, for
    example, returns a keep mask shaped ``[batch, 1, 16, 272]`` and position
    IDs shaped ``[batch, 272]``.
    """

    if isinstance(source_tokens, bool) or not isinstance(source_tokens, int):
        raise TypeError("source_tokens must be an integer")
    if source_tokens <= 0:
        raise ValueError("source_tokens must be positive")
    if not isinstance(source_attention_mask, torch.Tensor):
        raise TypeError("source_attention_mask must be a torch.Tensor")
    if source_attention_mask.dtype is not torch.bool:
        raise TypeError("source_attention_mask must be bool with True == keep")
    if source_attention_mask.ndim != 2:
        raise ValueError(
            f"source_attention_mask must have shape [batch, {source_tokens}]"
        )
    batch_size, source_width = source_attention_mask.shape
    if batch_size <= 0 or source_width != source_tokens:
        raise ValueError(
            f"source_attention_mask must have shape [batch, {source_tokens}]"
        )

    source_lengths = source_attention_mask.sum(dim=-1, dtype=torch.long)
    if bool((source_lengths <= 0).any().item()):
        raise ValueError("every source must contain at least one valid token")
    source_offsets = torch.arange(
        source_width,
        dtype=torch.long,
        device=source_attention_mask.device,
    )
    expected_right_padding = source_offsets.view(1, -1) < source_lengths.view(-1, 1)
    if not torch.equal(source_attention_mask, expected_right_padding):
        raise ValueError(
            "source_attention_mask must be a contiguous right-padded prefix"
        )

    source_keep = source_attention_mask[:, None, None, :].expand(
        batch_size, 1, COMPRESSED_QUERY_COUNT, source_width
    )
    same_block_keep = torch.ones(
        (
            batch_size,
            1,
            COMPRESSED_QUERY_COUNT,
            COMPRESSED_QUERY_COUNT,
        ),
        dtype=torch.bool,
        device=source_attention_mask.device,
    )
    keep_mask = torch.cat((source_keep, same_block_keep), dim=-1)

    source_position_ids = source_offsets.view(1, -1).expand(batch_size, -1)
    query_offsets = torch.arange(
        COMPRESSED_QUERY_COUNT,
        dtype=torch.long,
        device=source_attention_mask.device,
    )
    query_position_ids = source_lengths.view(-1, 1) + query_offsets.view(1, -1)
    position_ids = torch.cat((source_position_ids, query_position_ids), dim=-1)
    return DFlashCompressionLayout(
        keep_mask=keep_mask.contiguous(),
        position_ids=position_ids.contiguous(),
    )


__all__ = [
    "COMPRESSED_QUERY_COUNT",
    "DFlashCompressionLayout",
    "SOURCE_KV_WIDTH",
    "build_dflash_16_query_layout",
]
