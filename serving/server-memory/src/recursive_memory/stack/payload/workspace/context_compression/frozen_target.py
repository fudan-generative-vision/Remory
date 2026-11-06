"""Shared inference helpers; benchmark/training entry points are not distributed."""

from __future__ import annotations


from typing import Any, Iterator, Mapping, NamedTuple, Optional, Sequence


import torch


class FlattenedEvidence(NamedTuple):
    features: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor


def _flatten_encoded_evidence(
    source_features: torch.Tensor,
    batch: Mapping[str, Any],
) -> FlattenedEvidence:
    """Gather ordered stdout spans after one frozen prefill per sample.

    Expected data ABI:

    * ``source_features``: ``[B, history_width, 8 * target_hidden]``;
    * ``source_evidence_positions``: ordered ``[Ne, 2]`` rows of
      ``(sample_index, history_position)``;
    * ``source_evidence_offsets``: ``[Ns + 1]`` offsets delimiting the stdout
      tokens belonging to each flattened tool-response segment.

    The result is right padded ``[Ns, max_stdout_tokens, feature_width]``.  It
    is this segment batch—not the full histories—that enters the compressor.
    """

    if source_features.ndim != 3:
        raise ValueError("source encoder must return [batch, sequence, feature]")
    required = (
        "source_evidence_positions",
        "source_evidence_offsets",
        "source_lengths",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(
            "prepared-data ABI is missing one-prefill evidence metadata: "
            + ", ".join(missing)
        )
    positions = batch["source_evidence_positions"].long()
    offsets = batch["source_evidence_offsets"].long()
    # The source prompt contains a JSON string, so quoting/escaping means its
    # hidden-state span can contain a different number of tokenizer rows than
    # the canonical raw stdout.  Offsets describe prompt rows; source_lengths
    # remains the canonical length used to enforce the advertised 1:N rate.
    expected_lengths = batch["source_span_lengths"].long()
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("source_evidence_positions must have shape [tokens, 2]")
    if offsets.ndim != 1 or offsets.numel() < 2:
        raise ValueError("source_evidence_offsets must have shape [segments + 1]")
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != positions.shape[0]:
        raise ValueError("source evidence offsets do not cover evidence positions")
    lengths = offsets[1:] - offsets[:-1]
    if bool((lengths <= 0).any().item()):
        raise ValueError("every compressed stdout segment must contain evidence")
    if not torch.equal(lengths, expected_lengths):
        raise ValueError("source_span_lengths disagree with source_evidence_offsets")
    if positions.device != source_features.device:
        raise ValueError("source evidence metadata and features must share a device")
    if bool(
        (
            (positions[:, 0] < 0)
            | (positions[:, 0] >= source_features.shape[0])
            | (positions[:, 1] < 0)
            | (positions[:, 1] >= source_features.shape[1])
        )
        .any()
        .item()
    ):
        raise ValueError("source_evidence_positions index outside source features")

    gathered = source_features[positions[:, 0], positions[:, 1]]
    segment_count = int(lengths.numel())
    max_length = int(lengths.max().item())
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_count, device=source_features.device), lengths
    )
    within_segment = torch.arange(
        gathered.shape[0], device=source_features.device
    ) - torch.repeat_interleave(offsets[:-1], lengths)
    padded = source_features.new_zeros(
        (segment_count, max_length, source_features.shape[-1])
    )
    padded[segment_ids, within_segment] = gathered
    attention_mask = torch.arange(max_length, device=source_features.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)
    return FlattenedEvidence(padded, attention_mask, lengths)


def _validate_memory_contract(compressed: Any, batch: Mapping[str, Any]) -> None:
    embeddings = compressed.embeddings
    mask = compressed.attention_mask.bool()
    counts = compressed.slot_counts.long()
    expected = batch["memory_lengths"].long()
    if embeddings.ndim != 3 or mask.shape != embeddings.shape[:2]:
        raise ValueError("compressor output embeddings/mask shapes are inconsistent")
    if counts.shape != expected.shape or not torch.equal(counts, expected):
        raise ValueError("compressor slot counts disagree with prepared memory lengths")
    if not torch.equal(mask.sum(dim=-1), counts):
        raise ValueError("compressor attention mask disagrees with its slot counts")
    if int(mask.sum().item()) != int(batch["memory_positions"].shape[0]):
        raise ValueError("compressor slots and student memory positions differ")
