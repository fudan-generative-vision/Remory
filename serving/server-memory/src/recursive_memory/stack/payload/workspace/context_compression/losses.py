"""Bounded-memory token loss through a frozen language-model head."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


class ChunkedTokenCEOutput(NamedTuple):
    """Mean token CE, detached top-1 accuracy, and active-token count."""

    loss: torch.Tensor
    token_accuracy: torch.Tensor
    token_count: torch.Tensor


class ChunkedForwardKLOutput(NamedTuple):
    """Exact mean forward KL and detached teacher/student diagnostics."""

    loss: torch.Tensor
    top1_agreement: torch.Tensor
    teacher_token_accuracy: torch.Tensor
    student_token_accuracy: torch.Tensor
    teacher_entropy: torch.Tensor
    token_count: torch.Tensor


def _active_token_weights(
    token_weights: torch.Tensor | None,
    *,
    target_ids: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Return validated FP32 weights for flattened active target positions."""

    if token_weights is None:
        return torch.ones(
            int(active.sum().item()), device=target_ids.device, dtype=torch.float32
        )
    if not isinstance(token_weights, torch.Tensor):
        raise TypeError("token_weights must be a torch.Tensor or None")
    if token_weights.shape != target_ids.shape:
        raise ValueError("token_weights must have the same shape as target_ids")
    if not torch.is_floating_point(token_weights):
        raise TypeError("token_weights must use a floating dtype")
    if token_weights.device != target_ids.device:
        raise ValueError("token_weights and target_ids must share a device")
    weights = token_weights.detach().reshape(-1).float()[active]
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("active token_weights must be finite")
    if bool((weights < 0).any().item()):
        raise ValueError("active token_weights must be nonnegative")
    if not bool((weights.sum() > 0).item()):
        raise ValueError("active token_weights must have positive total weight")
    return weights


def frozen_lm_head_chunked_token_ce(
    recovery_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    lm_head: nn.Module,
    *,
    prefix_width: int,
    token_weights: torch.Tensor | None = None,
    chunk_size: int = 256,
    ignore_index: int = -100,
) -> ChunkedTokenCEOutput:
    """Compute teacher-forced token CE without materializing all logits.

    The alignment is deliberately explicit: hidden/logit row
    ``prefix_width - 1 + j`` predicts ``target_ids[..., j]``.  Thus the last
    prefix row predicts target token zero, exactly matching causal-LM teacher
    forcing.  Ignored target rows are removed before projection.

    Projection is chunked over active token positions.  When the recovery path
    is differentiable, each chunk is activation-checkpointed so
    full-vocabulary logits from all chunks are not retained simultaneously for
    backward.  The LM head must already be frozen.  Its operations remain
    grad-enabled so gradients flow through ``recovery_hidden`` to the
    compressor.
    """

    if not isinstance(recovery_hidden, torch.Tensor):
        raise TypeError("recovery_hidden must be a torch.Tensor")
    if recovery_hidden.ndim != 3 or not torch.is_floating_point(recovery_hidden):
        raise ValueError("recovery_hidden must be floating [batch, sequence, hidden]")
    if not isinstance(target_ids, torch.Tensor) or target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, target]")
    if target_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("target_ids must use int32 or int64")
    if target_ids.device != recovery_hidden.device:
        raise ValueError("target_ids and recovery_hidden must share a device")
    if target_ids.shape[0] != recovery_hidden.shape[0]:
        raise ValueError("target_ids and recovery_hidden batch sizes differ")
    if target_ids.shape[1] <= 0:
        raise ValueError("target_ids must contain at least one target position")
    if isinstance(prefix_width, bool) or not isinstance(prefix_width, int):
        raise TypeError("prefix_width must be a plain int")
    if prefix_width <= 0:
        raise ValueError("prefix_width must be positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a plain int")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be a plain int")
    if not isinstance(lm_head, nn.Module):
        raise TypeError("lm_head must be a torch.nn.Module")
    if any(parameter.requires_grad for parameter in lm_head.parameters()):
        raise ValueError("lm_head must be frozen before computing token CE")

    target_width = int(target_ids.shape[1])
    aligned_start = prefix_width - 1
    aligned_stop = aligned_start + target_width
    if aligned_stop > recovery_hidden.shape[1]:
        raise ValueError("recovery_hidden is too short for prefix_width and target_ids")

    aligned_hidden = recovery_hidden[:, aligned_start:aligned_stop, :]
    hidden_flat = aligned_hidden.reshape(-1, aligned_hidden.shape[-1])
    target_flat = target_ids.reshape(-1).long()
    active = target_flat != int(ignore_index)
    hidden_active = hidden_flat[active]
    target_active = target_flat[active]
    weight_active = _active_token_weights(
        token_weights, target_ids=target_ids, active=active
    )
    active_count = int(target_active.numel())
    if active_count == 0:
        raise ValueError("target_ids contain no active tokens")

    def ce_chunk(
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        weight_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = lm_head(hidden_chunk)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise TypeError("lm_head must return [tokens, vocabulary] logits")
        if logits.shape[0] != hidden_chunk.shape[0] or logits.shape[1] < 2:
            raise ValueError("lm_head returned an invalid logits shape")
        if bool(
            ((target_chunk < 0) | (target_chunk >= logits.shape[1])).any().item()
        ):
            raise ValueError("target_ids contain an invalid vocabulary id")
        logits_float = logits.float()
        per_token = F.cross_entropy(logits_float, target_chunk, reduction="none")
        loss_sum = (per_token * weight_chunk).sum()
        with torch.no_grad():
            correct = (logits_float.argmax(dim=-1) == target_chunk).sum()
        return loss_sum, correct.detach()

    loss_numerators = []
    correct_count = target_active.new_zeros(())
    for start in range(0, active_count, chunk_size):
        stop = min(start + chunk_size, active_count)
        chunk_inputs = (
            hidden_active[start:stop],
            target_active[start:stop],
            weight_active[start:stop],
        )
        if torch.is_grad_enabled() and chunk_inputs[0].requires_grad:
            chunk_loss, chunk_correct = activation_checkpoint(
                ce_chunk, *chunk_inputs, use_reentrant=False
            )
        else:
            chunk_loss, chunk_correct = ce_chunk(*chunk_inputs)
        loss_numerators.append(chunk_loss)
        correct_count = correct_count + chunk_correct

    token_count = target_active.new_tensor(active_count)
    loss = torch.stack(loss_numerators).sum() / weight_active.sum()
    token_accuracy = correct_count.float() / token_count.float()
    return ChunkedTokenCEOutput(
        loss=loss,
        token_accuracy=token_accuracy.detach(),
        token_count=token_count.detach(),
    )


def frozen_lm_head_chunked_forward_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    lm_head: nn.Module,
    *,
    student_prefix_width: int,
    teacher_prefix_width: int,
    token_weights: torch.Tensor | None = None,
    chunk_size: int = 32,
    ignore_index: int = -100,
) -> ChunkedForwardKLOutput:
    """Compute exact full-vocabulary ``KL(teacher || student)`` at temperature 1.

    Both hidden-state sequences are teacher-forced with the same continuation.
    Hidden row ``prefix_width - 1 + j`` predicts continuation token ``j``.
    The two prefixes may have different lengths, but their target width and
    active-token mask are shared exactly.

    Projection is chunked over active token positions.  When the student path
    is differentiable, each chunk is activation-checkpointed so full-vocabulary
    logits from all chunks are not retained simultaneously for backward.  The
    teacher projection and distribution are always evaluated under
    :func:`torch.no_grad` and are explicitly detached.  The frozen head remains
    grad-enabled on the student path, preserving gradients to ``student_hidden``.

    ``target_ids`` is used only for teacher/student top-1 accuracy diagnostics;
    it never contributes a gold-token CE term to :attr:`ChunkedForwardKLOutput.loss`.
    """

    for name, hidden in (
        ("student_hidden", student_hidden),
        ("teacher_hidden", teacher_hidden),
    ):
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if hidden.ndim != 3 or not torch.is_floating_point(hidden):
            raise ValueError(f"{name} must be floating [batch, sequence, hidden]")
    if student_hidden.device != teacher_hidden.device:
        raise ValueError("teacher and student hidden states must share a device")
    if student_hidden.shape[0] != teacher_hidden.shape[0]:
        raise ValueError("teacher and student batch sizes differ")
    if student_hidden.shape[2] != teacher_hidden.shape[2]:
        raise ValueError("teacher and student hidden widths differ")
    if not isinstance(target_ids, torch.Tensor) or target_ids.ndim != 2:
        raise ValueError("target_ids must have shape [batch, target]")
    if target_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("target_ids must use int32 or int64")
    if target_ids.device != student_hidden.device:
        raise ValueError("target_ids and hidden states must share a device")
    if target_ids.shape[0] != student_hidden.shape[0]:
        raise ValueError("target_ids and hidden-state batch sizes differ")
    if target_ids.shape[1] <= 0:
        raise ValueError("target_ids must contain at least one target position")
    for name, width in (
        ("student_prefix_width", student_prefix_width),
        ("teacher_prefix_width", teacher_prefix_width),
    ):
        if isinstance(width, bool) or not isinstance(width, int):
            raise TypeError(f"{name} must be a plain int")
        if width <= 0:
            raise ValueError(f"{name} must be positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be a plain int")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be a plain int")
    if not isinstance(lm_head, nn.Module):
        raise TypeError("lm_head must be a torch.nn.Module")
    if any(parameter.requires_grad for parameter in lm_head.parameters()):
        raise ValueError("lm_head must be frozen before computing forward KL")

    target_width = int(target_ids.shape[1])
    student_start = student_prefix_width - 1
    teacher_start = teacher_prefix_width - 1
    if student_start + target_width > student_hidden.shape[1]:
        raise ValueError(
            "student_hidden is too short for student_prefix_width and target_ids"
        )
    if teacher_start + target_width > teacher_hidden.shape[1]:
        raise ValueError(
            "teacher_hidden is too short for teacher_prefix_width and target_ids"
        )

    student_aligned = student_hidden[
        :, student_start : student_start + target_width, :
    ]
    teacher_aligned = teacher_hidden[
        :, teacher_start : teacher_start + target_width, :
    ].detach()
    target_flat = target_ids.reshape(-1).long()
    active = target_flat != int(ignore_index)
    student_active = student_aligned.reshape(-1, student_aligned.shape[-1])[active]
    teacher_active = teacher_aligned.reshape(-1, teacher_aligned.shape[-1])[active]
    target_active = target_flat[active]
    weight_active = _active_token_weights(
        token_weights, target_ids=target_ids, active=active
    )
    active_count = int(target_active.numel())
    if active_count == 0:
        raise ValueError("target_ids contain no active tokens")

    def kl_chunk(
        student_chunk: torch.Tensor,
        teacher_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        weight_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        student_logits = lm_head(student_chunk)
        if not isinstance(student_logits, torch.Tensor) or student_logits.ndim != 2:
            raise TypeError("lm_head must return [tokens, vocabulary] logits")
        if student_logits.shape[0] != student_chunk.shape[0]:
            raise ValueError("lm_head returned an invalid student token dimension")
        if student_logits.shape[1] < 2:
            raise ValueError("lm_head vocabulary must contain at least two tokens")
        student_log_prob = F.log_softmax(student_logits.float(), dim=-1)

        with torch.no_grad():
            teacher_logits = lm_head(teacher_chunk.detach())
            if (
                not isinstance(teacher_logits, torch.Tensor)
                or teacher_logits.shape != student_logits.shape
            ):
                raise ValueError(
                    "teacher and student lm_head projections must have equal shape"
                )
            if bool(
                ((target_chunk < 0) | (target_chunk >= teacher_logits.shape[1]))
                .any()
                .item()
            ):
                raise ValueError("target_ids contain an invalid vocabulary id")
            teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
            teacher_prob = teacher_log_prob.exp()

        per_token_kl = (
            teacher_prob * (teacher_log_prob - student_log_prob)
        ).sum(dim=-1)
        kl_sum = (per_token_kl * weight_chunk).sum()
        with torch.no_grad():
            teacher_top1 = teacher_logits.argmax(dim=-1)
            student_top1 = student_logits.argmax(dim=-1)
            agreement = (teacher_top1 == student_top1).sum()
            teacher_correct = (teacher_top1 == target_chunk).sum()
            student_correct = (student_top1 == target_chunk).sum()
            entropy_sum = (
                -(teacher_prob * teacher_log_prob).sum(dim=-1)
            ).sum()
        return (
            kl_sum,
            agreement.detach(),
            teacher_correct.detach(),
            student_correct.detach(),
            entropy_sum.detach(),
        )

    kl_numerator = student_active.float().sum() * 0.0
    agreement_count = target_active.new_zeros(())
    teacher_correct_count = target_active.new_zeros(())
    student_correct_count = target_active.new_zeros(())
    entropy_numerator = student_active.new_zeros((), dtype=torch.float32)
    for start in range(0, active_count, chunk_size):
        stop = min(start + chunk_size, active_count)
        chunk_inputs = (
            student_active[start:stop],
            teacher_active[start:stop],
            target_active[start:stop],
            weight_active[start:stop],
        )
        if torch.is_grad_enabled() and chunk_inputs[0].requires_grad:
            chunk_output = activation_checkpoint(
                kl_chunk, *chunk_inputs, use_reentrant=False
            )
        else:
            chunk_output = kl_chunk(*chunk_inputs)
        (
            chunk_kl,
            chunk_agreement,
            chunk_teacher_correct,
            chunk_student_correct,
            chunk_entropy,
        ) = chunk_output
        kl_numerator = kl_numerator + chunk_kl
        agreement_count = agreement_count + chunk_agreement
        teacher_correct_count = teacher_correct_count + chunk_teacher_correct
        student_correct_count = student_correct_count + chunk_student_correct
        entropy_numerator = entropy_numerator + chunk_entropy

    token_count = target_active.new_tensor(active_count)
    loss_denominator = weight_active.sum()
    diagnostic_denominator = token_count.float()
    return ChunkedForwardKLOutput(
        loss=kl_numerator / loss_denominator,
        top1_agreement=(
            agreement_count.float() / diagnostic_denominator
        ).detach(),
        teacher_token_accuracy=(
            teacher_correct_count.float() / diagnostic_denominator
        ).detach(),
        student_token_accuracy=(
            student_correct_count.float() / diagnostic_denominator
        ).detach(),
        teacher_entropy=(
            entropy_numerator.float() / diagnostic_denominator
        ).detach(),
        token_count=token_count.detach(),
    )


__all__ = [
    "ChunkedForwardKLOutput",
    "ChunkedTokenCEOutput",
    "frozen_lm_head_chunked_forward_kl",
    "frozen_lm_head_chunked_token_ce",
]
