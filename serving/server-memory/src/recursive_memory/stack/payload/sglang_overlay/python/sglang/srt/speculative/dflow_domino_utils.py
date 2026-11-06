from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn


DFLOW_DOMINO_CANDIDATE_POOL_SIZE = 2048
DFLOW_DOMINO_RESIDUAL_RANK = 256

# A policy threshold for the logical gathered base-logit tensor.  It is not a
# peak-memory guarantee: the caller still owns the draft hidden states and the
# low-rank head's weights.
DFLOW_DOMINO_TP_FULL_BASE_LOGITS_MAX_BYTES = 96 * 1024 * 1024

DFlowDominoGreedyMode = Literal["auto", "full_vocab", "candidate_pool"]
DFlowDominoTraceHook = Callable[[str, torch.Tensor], None]


@runtime_checkable
class DFlowDominoTPGroup(Protocol):
    """The small subset of GroupCoordinator used by the Domino core."""

    world_size: int

    def all_gather_into_tensor(
        self, output: torch.Tensor, input: torch.Tensor
    ) -> None: ...

    def all_reduce(self, input: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class DFlowDominoVocabShard:
    """Contiguous original-vocabulary rows owned by one TP rank.

    ``num_org_padded`` is the physical lm-head row count used by collectives;
    rows in ``[num_org, num_org_padded)`` are padding and must never win.
    """

    org_vocab_start: int
    num_org: int
    num_org_padded: int

    @classmethod
    def unsharded(cls, vocab_size: int) -> "DFlowDominoVocabShard":
        return cls(
            org_vocab_start=0, num_org=int(vocab_size), num_org_padded=int(vocab_size)
        )

    @property
    def org_vocab_end(self) -> int:
        return int(self.org_vocab_start + self.num_org)

    def validate(
        self,
        *,
        vocab_size: int,
        tp_size: int,
        tp_rank: int,
    ) -> None:
        vocab_size = int(vocab_size)
        tp_size = int(tp_size)
        tp_rank = int(tp_rank)
        if self.num_org <= 0 or self.num_org_padded < self.num_org:
            raise ValueError("DFlow-Domino vocab shard has invalid row counts.")
        if self.org_vocab_start < 0 or self.org_vocab_end > vocab_size:
            raise ValueError("DFlow-Domino vocab shard is outside the vocabulary.")
        expected_start = tp_rank * self.num_org_padded
        expected_end = min(expected_start + self.num_org_padded, vocab_size)
        if (
            self.org_vocab_start != expected_start
            or self.org_vocab_end != expected_end
            or self.num_org != expected_end - expected_start
        ):
            raise ValueError(
                "DFlow-Domino vocab shard does not match its contiguous TP rank."
            )
        if tp_size * self.num_org_padded < vocab_size:
            raise ValueError("DFlow-Domino TP shards do not cover the vocabulary.")


def _trace(
    trace_hook: DFlowDominoTraceHook | None,
    name: str,
    value: torch.Tensor,
) -> None:
    if trace_hook is not None:
        trace_hook(name, value)


def _dflow_domino_gru_cell(
    prefix_gru: nn.GRU,
    input: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Run one sampled-prefix update without per-token RNN weight packing."""

    return torch.ops.aten.gru_cell.default(
        input,
        hidden,
        prefix_gru.weight_ih_l0,
        prefix_gru.weight_hh_l0,
        prefix_gru.bias_ih_l0 if prefix_gru.bias else None,
        prefix_gru.bias_hh_l0 if prefix_gru.bias else None,
    )


def _all_reduce_sum(tp_group: DFlowDominoTPGroup, value: torch.Tensor) -> torch.Tensor:
    reduced = tp_group.all_reduce(value)
    # GroupCoordinator returns a tensor.  Supporting an in-place/None fake is
    # useful for CPU correctness tests and does not alter the serving path.
    return value if reduced is None else reduced


def dflow_domino_parallel_base_logits(
    proposal_hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    """Project every proposal position with exactly one batched ``F.linear``.

    Args:
        proposal_hidden: ``[batch, proposals, hidden]`` decoded DFlow states.
        lm_head_weight: Full or local vocab shard, ``[vocab_rows, hidden]``.

    Returns:
        Step-major logits with shape ``[proposals, batch, vocab_rows]``.
    """

    if proposal_hidden.ndim != 3:
        raise ValueError(
            "proposal_hidden must have shape [batch, proposals, hidden], got "
            f"{tuple(proposal_hidden.shape)}."
        )
    if lm_head_weight.ndim != 2:
        raise ValueError("lm_head_weight must be a rank-2 tensor.")
    batch_size, num_proposals, hidden_size = proposal_hidden.shape
    if int(lm_head_weight.shape[1]) != int(hidden_size):
        raise ValueError(
            "DFlow-Domino lm-head hidden size does not match proposal states."
        )
    hidden = proposal_hidden
    if hidden.dtype != lm_head_weight.dtype:
        hidden = hidden.to(dtype=lm_head_weight.dtype)
    flat = (
        hidden.transpose(0, 1)
        .contiguous()
        .view(int(num_proposals) * int(batch_size), int(hidden_size))
    )
    return F.linear(flat, lm_head_weight).view(
        int(num_proposals), int(batch_size), int(lm_head_weight.shape[0])
    )


def _tp_global_argmax(
    local_logits: torch.Tensor,
    *,
    shard: DFlowDominoVocabShard,
    tp_group: DFlowDominoTPGroup,
) -> torch.Tensor:
    """Exact global argmax over contiguous vocab shards, including tie order."""

    local_max, local_position = torch.max(local_logits[:, : shard.num_org], dim=-1)
    local_ids = local_position.to(torch.int64) + int(shard.org_vocab_start)
    tp_size = int(tp_group.world_size)
    batch_size = int(local_logits.shape[0])
    gathered_max = torch.empty(
        (tp_size * batch_size,), dtype=local_max.dtype, device=local_max.device
    )
    gathered_ids = torch.empty(
        (tp_size * batch_size,), dtype=torch.int64, device=local_max.device
    )
    tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
    tp_group.all_gather_into_tensor(gathered_ids, local_ids.contiguous())
    gathered_max = gathered_max.view(tp_size, batch_size)
    gathered_ids = gathered_ids.view(tp_size, batch_size)
    best_rank = torch.argmax(gathered_max, dim=0, keepdim=True)
    return torch.gather(gathered_ids, 0, best_rank).squeeze(0)


def _tp_candidate_state(
    local_feedback_logits: torch.Tensor,
    *,
    candidate_pool_size: int,
    shard: DFlowDominoVocabShard,
    tp_group: DFlowDominoTPGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one global, block-shared candidate pool from vocab shards."""

    num_steps, batch_size, local_vocab_size = local_feedback_logits.shape
    tp_size = int(tp_group.world_size)
    local_k = min(int(candidate_pool_size), int(local_vocab_size))
    if int(candidate_pool_size) > tp_size * local_k:
        raise ValueError(
            "DFlow-Domino TP shards do not cover the requested candidate pool."
        )

    pool_scores = local_feedback_logits.amax(dim=0)
    if shard.num_org < local_vocab_size:
        pool_scores[:, shard.num_org :].fill_(float("-inf"))
    local_scores, local_positions = torch.topk(
        pool_scores, k=local_k, dim=-1, sorted=False
    )
    local_ids = local_positions.to(torch.int64) + int(shard.org_vocab_start)

    gathered_scores = torch.empty(
        (tp_size * batch_size, local_k),
        dtype=local_scores.dtype,
        device=local_scores.device,
    )
    gathered_ids = torch.empty(
        (tp_size * batch_size, local_k),
        dtype=torch.int64,
        device=local_scores.device,
    )
    tp_group.all_gather_into_tensor(gathered_scores, local_scores.contiguous())
    tp_group.all_gather_into_tensor(gathered_ids, local_ids.contiguous())
    gathered_scores = (
        gathered_scores.view(tp_size, batch_size, local_k)
        .permute(1, 0, 2)
        .reshape(batch_size, tp_size * local_k)
    )
    gathered_ids = (
        gathered_ids.view(tp_size, batch_size, local_k)
        .permute(1, 0, 2)
        .reshape(batch_size, tp_size * local_k)
    )
    global_positions = torch.topk(
        gathered_scores,
        k=int(candidate_pool_size),
        dim=-1,
        sorted=False,
    ).indices
    candidate_ids = torch.gather(gathered_ids, 1, global_positions).contiguous()

    owned = (candidate_ids >= int(shard.org_vocab_start)) & (
        candidate_ids < int(shard.org_vocab_end)
    )
    local_positions = (candidate_ids - int(shard.org_vocab_start)).clamp(
        0, max(shard.num_org - 1, 0)
    )
    candidate_base = torch.gather(
        local_feedback_logits.transpose(0, 1),
        2,
        local_positions[:, None, :].expand(-1, num_steps, -1),
    )
    # Every original-vocab id has exactly one owner; SUM reconstructs its
    # base logits without materializing the full vocabulary on every rank.
    candidate_base.masked_fill_(~owned[:, None, :], 0)
    candidate_base = _all_reduce_sum(tp_group, candidate_base.contiguous())
    return candidate_ids, candidate_base.transpose(0, 1)


def _gather_full_base_logits(
    local_logits: torch.Tensor,
    *,
    vocab_size: int,
    tp_group: DFlowDominoTPGroup,
) -> torch.Tensor:
    num_proposals, batch_size, local_vocab_size = local_logits.shape
    local_t = local_logits.view(
        num_proposals * batch_size, local_vocab_size
    ).T.contiguous()
    gathered = torch.empty(
        (int(tp_group.world_size) * local_vocab_size, num_proposals * batch_size),
        dtype=local_logits.dtype,
        device=local_logits.device,
    )
    tp_group.all_gather_into_tensor(gathered, local_t)
    return (
        gathered[: int(vocab_size)]
        .T.contiguous()
        .view(num_proposals, batch_size, int(vocab_size))
    )


def _candidate_state_unsharded(
    feedback_logits: torch.Tensor,
    *,
    candidate_pool_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_steps = int(feedback_logits.shape[0])
    candidate_ids = torch.topk(
        feedback_logits.amax(dim=0),
        k=int(candidate_pool_size),
        dim=-1,
        sorted=False,
    ).indices.contiguous()
    candidate_base = torch.gather(
        feedback_logits.transpose(0, 1),
        2,
        candidate_ids[:, None, :].expand(-1, num_steps, -1),
    ).transpose(0, 1)
    return candidate_ids, candidate_base


def resolve_dflow_domino_vocab_shard(
    *,
    lm_head: nn.Module,
    vocab_size: int,
    tp_size: int,
    tp_rank: int,
) -> DFlowDominoVocabShard:
    """Resolve and fail-close SGLang lm-head original-vocab shard metadata."""

    lm_head_weight = getattr(lm_head, "weight", None)
    if not isinstance(lm_head_weight, torch.Tensor):
        raise ValueError("DFlow-Domino requires an lm_head weight tensor.")
    shard_metadata = getattr(lm_head, "shard_indices", None)
    if shard_metadata is None:
        if int(tp_size) != 1:
            raise ValueError("DFlow-Domino requires lm_head shard metadata for TP>1.")
        if int(lm_head_weight.shape[0]) != int(vocab_size):
            raise ValueError(
                "DFlow-Domino unsharded lm_head rows must equal vocab_size."
            )
        return DFlowDominoVocabShard.unsharded(vocab_size)

    if (
        int(getattr(lm_head, "num_added_embeddings", 0)) != 0
        or int(getattr(shard_metadata, "num_added_elements", 0)) != 0
    ):
        raise ValueError("DFlow-Domino does not support added-vocab lm_head shards.")
    required = (
        "org_vocab_start_index",
        "org_vocab_end_index",
        "num_org_elements",
        "num_org_elements_padded",
    )
    missing = [name for name in required if not hasattr(shard_metadata, name)]
    if missing:
        raise ValueError(
            "DFlow-Domino lm_head shard metadata is missing: " + ", ".join(missing)
        )
    if int(getattr(lm_head, "org_vocab_size", vocab_size)) != int(vocab_size):
        raise ValueError("DFlow-Domino lm_head original vocab size is inconsistent.")
    if int(getattr(lm_head, "tp_size", tp_size)) != int(tp_size):
        raise ValueError("DFlow-Domino lm_head TP size is inconsistent.")

    shard = DFlowDominoVocabShard(
        org_vocab_start=int(shard_metadata.org_vocab_start_index),
        num_org=int(shard_metadata.num_org_elements),
        num_org_padded=int(shard_metadata.num_org_elements_padded),
    )
    if int(shard_metadata.org_vocab_end_index) != shard.org_vocab_end:
        raise ValueError("DFlow-Domino lm_head shard end index is inconsistent.")
    shard.validate(vocab_size=vocab_size, tp_size=tp_size, tp_rank=tp_rank)
    if int(lm_head_weight.shape[0]) < shard.num_org_padded:
        raise ValueError("DFlow-Domino lm_head is smaller than its padded shard.")
    return shard


def validate_dflow_domino_runtime(
    *,
    tp_size: int,
    tp_rank: int,
    target_vocab_size: int,
    draft_vocab_size: int,
    hidden_size: int,
    target_embedding: nn.Module,
    lm_head: nn.Module,
    prefix_gru: nn.GRU,
    residual_down: nn.Linear,
    residual_up: nn.Linear,
    pure_draft_prefix_len: int,
    residual_rank: int = DFLOW_DOMINO_RESIDUAL_RANK,
) -> DFlowDominoVocabShard:
    """Validate the narrow, official-prefix DFlow-Domino serving contract.

    CPU and floating-point dtype are intentionally unrestricted so the exact
    runtime math can be tested without a GPU.  All participating weights must
    nevertheless share one dtype and one device.
    """

    if isinstance(pure_draft_prefix_len, bool) or int(pure_draft_prefix_len) != 1:
        raise ValueError("DFlow-Domino requires pure_draft_prefix_len exactly 1.")
    if int(residual_rank) != DFLOW_DOMINO_RESIDUAL_RANK:
        raise ValueError(
            "DFlow-Domino requires rank-256 logit residuals, got "
            f"rank={residual_rank}."
        )
    tp_size = int(tp_size)
    tp_rank = int(tp_rank)
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(
            f"DFlow-Domino invalid TP coordinates rank={tp_rank}, size={tp_size}."
        )
    if int(target_vocab_size) != int(draft_vocab_size):
        raise ValueError(
            "DFlow-Domino requires identical target and draft vocab sizes."
        )

    embedding_weight = getattr(target_embedding, "weight", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    if not isinstance(embedding_weight, torch.Tensor) or not isinstance(
        lm_head_weight, torch.Tensor
    ):
        raise ValueError(
            "DFlow-Domino requires target embedding and lm_head weight tensors."
        )
    shard = resolve_dflow_domino_vocab_shard(
        lm_head=lm_head,
        vocab_size=target_vocab_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )

    embedding_shard = getattr(target_embedding, "shard_indices", None)
    if embedding_shard is not None:
        if (
            int(getattr(target_embedding, "num_added_embeddings", 0)) != 0
            or int(getattr(embedding_shard, "num_added_elements", 0)) != 0
        ):
            raise ValueError(
                "DFlow-Domino does not support added-vocab embedding shards."
            )
        if int(getattr(target_embedding, "org_vocab_size", target_vocab_size)) != int(
            target_vocab_size
        ):
            raise ValueError(
                "DFlow-Domino embedding original vocab size is inconsistent."
            )
        embedding_tp_size = int(getattr(target_embedding, "tp_size", tp_size))
        if embedding_tp_size not in (1, tp_size):
            raise ValueError("DFlow-Domino embedding TP size is inconsistent.")
        required_rows = (
            int(target_vocab_size)
            if embedding_tp_size == 1
            else int(getattr(embedding_shard, "num_org_elements_padded"))
        )
        if int(embedding_weight.shape[0]) < required_rows:
            raise ValueError(
                "DFlow-Domino embedding is smaller than its vocabulary shard."
            )
    elif int(embedding_weight.shape[0]) < int(target_vocab_size):
        if tp_size != 1:
            raise ValueError("DFlow-Domino requires embedding shard metadata for TP>1.")
        raise ValueError("DFlow-Domino embedding has fewer rows than vocab_size.")

    hidden_size = int(hidden_size)
    if (
        int(embedding_weight.shape[-1]) != hidden_size
        or int(lm_head_weight.shape[-1]) != hidden_size
    ):
        raise ValueError("DFlow-Domino embedding/lm-head hidden size is inconsistent.")
    if not isinstance(prefix_gru, nn.GRU):
        raise ValueError("DFlow-Domino prefix_gru must be torch.nn.GRU.")
    if (
        int(prefix_gru.input_size) != hidden_size
        or int(prefix_gru.num_layers) != 1
        or bool(prefix_gru.bidirectional)
        or not bool(prefix_gru.batch_first)
    ):
        raise ValueError(
            "DFlow-Domino requires a one-layer, batch-first, unidirectional GRU."
        )
    if (
        int(residual_down.in_features) != hidden_size + int(prefix_gru.hidden_size)
        or int(residual_down.out_features) != DFLOW_DOMINO_RESIDUAL_RANK
    ):
        raise ValueError("DFlow-Domino residual_down shape is inconsistent.")
    if int(residual_up.in_features) != DFLOW_DOMINO_RESIDUAL_RANK or int(
        residual_up.out_features
    ) != int(target_vocab_size):
        raise ValueError("DFlow-Domino residual_up must be [vocab, 256].")

    weights = [
        embedding_weight,
        lm_head_weight,
        prefix_gru.weight_ih_l0,
        prefix_gru.weight_hh_l0,
        residual_down.weight,
        residual_up.weight,
    ]
    if prefix_gru.bias:
        weights.extend((prefix_gru.bias_ih_l0, prefix_gru.bias_hh_l0))
    if residual_down.bias is not None:
        weights.append(residual_down.bias)
    if residual_up.bias is not None:
        weights.append(residual_up.bias)
    dtypes = {weight.dtype for weight in weights}
    devices = {weight.device for weight in weights}
    if len(dtypes) != 1:
        raise ValueError("DFlow-Domino runtime weights must share one dtype.")
    if len(devices) != 1:
        raise ValueError("DFlow-Domino runtime weights must share one device.")
    return shard


@torch.no_grad()
def dflow_domino_greedy_rollout(
    *,
    draft_hidden: torch.Tensor,
    verified_ids: torch.Tensor,
    target_embedding: nn.Module,
    lm_head_weight: torch.Tensor,
    prefix_gru: nn.GRU,
    residual_down: nn.Linear,
    residual_up: nn.Linear,
    vocab_size: int,
    pure_draft_prefix_len: int = 1,
    candidate_pool_size: int = DFLOW_DOMINO_CANDIDATE_POOL_SIZE,
    greedy_mode: DFlowDominoGreedyMode = "auto",
    tp_group: DFlowDominoTPGroup | None = None,
    vocab_shard: DFlowDominoVocabShard | None = None,
    full_base_logits_max_bytes: int = DFLOW_DOMINO_TP_FULL_BASE_LOGITS_MAX_BYTES,
    trace_hook: DFlowDominoTraceHook | None = None,
) -> torch.Tensor:
    """Greedily decode a DFlow-Domino proposal chain.

    ``draft_hidden`` is ``[batch, block, hidden]`` and position zero is the
    verified root.  The lm head is evaluated once for positions ``1..block-1``.
    Slot 1 is *strictly* the base-lm-head argmax.  Only after sampling slot 1
    does the GRU consume ``[verified root, sampled slot 1]``; the rank-256
    residual may therefore affect slots 2 and later, never slot 1.

    The low-rank up projection is replicated/full-vocabulary.  The target lm
    head may be vocab-sharded; ``tp_group`` plus ``vocab_shard`` then provide
    exact full-vocab argmax and global candidate-pool reconstruction.
    """

    if isinstance(pure_draft_prefix_len, bool) or int(pure_draft_prefix_len) != 1:
        raise ValueError("DFlow-Domino requires pure_draft_prefix_len exactly 1.")
    if greedy_mode not in ("auto", "full_vocab", "candidate_pool"):
        raise ValueError(f"Unknown DFlow-Domino greedy_mode={greedy_mode!r}.")
    if draft_hidden.ndim != 3:
        raise ValueError(
            "draft_hidden must have shape [batch, block, hidden], got "
            f"{tuple(draft_hidden.shape)}."
        )
    batch_size, block_size, hidden_size = draft_hidden.shape
    if int(block_size) <= 1:
        raise ValueError("DFlow-Domino requires block_size > 1.")
    if verified_ids.shape != (batch_size,):
        raise ValueError(
            f"verified_ids must have shape ({batch_size},), got "
            f"{tuple(verified_ids.shape)}."
        )
    if verified_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("DFlow-Domino verified_ids must be integer token ids.")
    vocab_size = int(vocab_size)
    candidate_pool_size = int(candidate_pool_size)
    if vocab_size <= 0:
        raise ValueError("DFlow-Domino vocab_size must be positive.")
    if candidate_pool_size < 0:
        raise ValueError("DFlow-Domino candidate_pool_size must be non-negative.")
    candidate_pool_size = min(candidate_pool_size, vocab_size)
    if greedy_mode == "candidate_pool" and not (0 < candidate_pool_size < vocab_size):
        raise ValueError(
            "candidate_pool mode requires 0 < candidate_pool_size < vocab_size."
        )
    if (
        int(residual_down.out_features) != DFLOW_DOMINO_RESIDUAL_RANK
        or int(residual_up.in_features) != DFLOW_DOMINO_RESIDUAL_RANK
    ):
        raise ValueError("DFlow-Domino rollout requires a rank-256 residual.")
    if int(residual_up.out_features) != vocab_size:
        raise ValueError("DFlow-Domino residual_up must cover the full vocabulary.")
    if int(residual_down.in_features) != int(hidden_size) + int(prefix_gru.hidden_size):
        raise ValueError("DFlow-Domino residual_down input shape is inconsistent.")

    tp_size = int(tp_group.world_size) if tp_group is not None else 1
    if tp_size < 1:
        raise ValueError("DFlow-Domino TP world size must be positive.")
    if tp_size > 1 and vocab_shard is None:
        raise ValueError("DFlow-Domino TP rollout requires vocab_shard metadata.")
    if vocab_shard is None:
        vocab_shard = DFlowDominoVocabShard.unsharded(vocab_size)
    local_vocab_size = int(vocab_shard.num_org_padded) if tp_size > 1 else vocab_size
    if int(lm_head_weight.shape[0]) < local_vocab_size:
        raise ValueError("DFlow-Domino lm_head is smaller than its vocab shard.")

    # Position zero is the verified root, not a proposal.  This is the single
    # physical base lm-head projection for the entire block.
    proposal_hidden = draft_hidden[:, 1:, :]
    num_proposals = int(proposal_hidden.shape[1])
    local_base_logits = dflow_domino_parallel_base_logits(
        proposal_hidden, lm_head_weight[:local_vocab_size]
    )
    _trace(trace_hook, "local_base_logits", local_base_logits)

    logical_full_bytes = (
        num_proposals * int(batch_size) * vocab_size * local_base_logits.element_size()
    )
    candidate_available = 0 < candidate_pool_size < vocab_size
    if greedy_mode == "full_vocab":
        use_candidate_pool = False
    elif greedy_mode == "candidate_pool":
        use_candidate_pool = True
    else:
        use_candidate_pool = candidate_available and (
            tp_size == 1 or logical_full_bytes > int(full_base_logits_max_bytes)
        )

    first_ids: torch.Tensor
    base_logits: torch.Tensor | None
    candidate_ids: torch.Tensor | None = None
    candidate_base: torch.Tensor | None = None
    if tp_size == 1:
        base_logits = local_base_logits[:, :, :vocab_size]
        first_ids = torch.argmax(base_logits[0], dim=-1).to(torch.long)
        if use_candidate_pool and num_proposals > 1:
            candidate_ids, candidate_base = _candidate_state_unsharded(
                base_logits[1:], candidate_pool_size=candidate_pool_size
            )
    elif use_candidate_pool:
        first_ids = _tp_global_argmax(
            local_base_logits[0],
            shard=vocab_shard,
            tp_group=tp_group,
        )
        base_logits = None
        if num_proposals > 1:
            candidate_ids, candidate_base = _tp_candidate_state(
                local_base_logits[1:],
                candidate_pool_size=candidate_pool_size,
                shard=vocab_shard,
                tp_group=tp_group,
            )
    else:
        base_logits = _gather_full_base_logits(
            local_base_logits,
            vocab_size=vocab_size,
            tp_group=tp_group,
        )
        first_ids = torch.argmax(base_logits[0], dim=-1).to(torch.long)

    _trace(trace_hook, "first_ids", first_ids)
    proposals = [first_ids]
    if num_proposals == 1:
        result = first_ids[:, None]
        _trace(trace_hook, "proposals", result)
        return result

    candidate_weight = None
    candidate_bias = None
    if candidate_ids is not None:
        candidate_weight = F.embedding(candidate_ids, residual_up.weight)
        if residual_up.bias is not None:
            candidate_bias = F.embedding(
                candidate_ids, residual_up.bias.unsqueeze(-1)
            ).squeeze(-1)
        _trace(trace_hook, "candidate_ids", candidate_ids)

    # Strict pure_draft_prefix_len=1: the GRU is not consulted until slot 1
    # has been sampled from base logits.  Then root and sampled slot 1 are
    # consumed in causal order before correcting slot 2.
    prefix_ids = torch.stack((verified_ids.to(torch.long), first_ids), dim=1)
    prefix_embeddings = target_embedding(prefix_ids)
    gru_dtype = prefix_gru.weight_ih_l0.dtype
    if prefix_embeddings.dtype != gru_dtype:
        prefix_embeddings = prefix_embeddings.to(dtype=gru_dtype)
    _, gru_hidden = prefix_gru(prefix_embeddings)

    residual_dtype = residual_down.weight.dtype
    for index in range(1, num_proposals):
        step_hidden = proposal_hidden[:, index, :]
        features = torch.cat(
            (
                step_hidden.to(dtype=residual_dtype),
                gru_hidden[0].to(dtype=residual_dtype),
            ),
            dim=-1,
        )
        correction_hidden = F.silu(residual_down(features))
        if candidate_ids is None:
            if base_logits is None:
                raise RuntimeError("DFlow-Domino full-vocab base logits are missing.")
            correction = residual_up(correction_hidden)
            next_ids = torch.argmax(
                base_logits[index] + correction.to(dtype=base_logits.dtype),
                dim=-1,
            ).to(torch.long)
        else:
            if candidate_base is None or candidate_weight is None:
                raise RuntimeError("DFlow-Domino candidate state is incomplete.")
            correction = torch.bmm(
                candidate_weight,
                correction_hidden.to(dtype=candidate_weight.dtype).unsqueeze(-1),
            ).squeeze(-1)
            if candidate_bias is not None:
                correction = correction + candidate_bias
            candidate_position = torch.argmax(
                candidate_base[index - 1] + correction.to(dtype=candidate_base.dtype),
                dim=-1,
            )
            next_ids = torch.gather(
                candidate_ids, 1, candidate_position[:, None]
            ).squeeze(1)
        proposals.append(next_ids)
        if index + 1 < num_proposals:
            next_embedding = target_embedding(next_ids)
            if next_embedding.dtype != gru_dtype:
                next_embedding = next_embedding.to(dtype=gru_dtype)
            gru_hidden = _dflow_domino_gru_cell(
                prefix_gru, next_embedding, gru_hidden[0]
            )[None]

    result = torch.stack(proposals, dim=1)
    _trace(trace_hook, "proposals", result)
    return result


__all__ = [
    "DFLOW_DOMINO_CANDIDATE_POOL_SIZE",
    "DFLOW_DOMINO_RESIDUAL_RANK",
    "DFLOW_DOMINO_TP_FULL_BASE_LOGITS_MAX_BYTES",
    "DFlowDominoGreedyMode",
    "DFlowDominoTPGroup",
    "DFlowDominoTraceHook",
    "DFlowDominoVocabShard",
    "dflow_domino_greedy_rollout",
    "dflow_domino_parallel_base_logits",
    "resolve_dflow_domino_vocab_shard",
    "validate_dflow_domino_runtime",
]
