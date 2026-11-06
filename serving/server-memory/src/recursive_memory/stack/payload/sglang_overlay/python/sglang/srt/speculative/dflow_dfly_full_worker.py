"""Isolated DFLASH worker for predecessor-conditioned full DFlow/DFly.

The inherited proposal path performs one physical draft-backbone forward for
the whole 16-slot block.  This subclass replaces only the head stage: the 15
future slots are visited in order, and each HSC invocation consumes the target
embedding of the token actually sampled at the previous slot.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    is_dflash_sampling_verify_available,
    resolve_dflash_verify_mask_policy,
)
from sglang.srt.speculative.dflash_worker import DFlashWorker


_TP_PARITY_STEPS_ENV = "SGLANG_DFLOW_DFLY_FULL_TP2_PARITY_STEPS"


def _parse_nonnegative_int_env(name: str, *, environ=None) -> int:
    if environ is None:
        environ = os.environ
    raw = str(environ.get(name, "0")).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a non-negative integer, got {raw!r}."
        ) from exc
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


@torch.no_grad()
def _sequential_hsc_greedy(
    *,
    decoded_future_hidden: torch.Tensor,
    verified_root_ids: torch.Tensor,
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    apply_hidden_correction: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sample_hidden: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a block while maintaining the exact predecessor-token chain.

    Args:
        decoded_future_hidden: ``[batch, future_slots, hidden]`` from the sole
            draft-backbone forward after denormalization/decode adaptation.
        verified_root_ids: ``[batch]`` committed token IDs.  These condition
            the first future slot.

    Returns:
        ``(sampled_ids, corrected_hidden)`` with shapes ``[B, M]`` and
        ``[B, M, H]``.  The helper contains no draft-model invocation.
    """

    if decoded_future_hidden.ndim != 3:
        raise ValueError(
            "Full DFlow/DFly decoded future hidden must be [batch, slots, hidden], "
            f"got {tuple(decoded_future_hidden.shape)}."
        )
    batch_size, future_slots, hidden_size = decoded_future_hidden.shape
    if batch_size <= 0 or future_slots <= 0 or hidden_size <= 0:
        raise ValueError(
            "Full DFlow/DFly decoded future dimensions must be positive, got "
            f"{tuple(decoded_future_hidden.shape)}."
        )
    if verified_root_ids.ndim != 1 or int(verified_root_ids.numel()) != int(batch_size):
        raise ValueError(
            "Full DFlow/DFly verified roots must be [batch], got "
            f"{tuple(verified_root_ids.shape)} for batch={batch_size}."
        )
    previous_ids = verified_root_ids.to(
        device=decoded_future_hidden.device, dtype=torch.long
    )
    sampled: list[torch.Tensor] = []
    corrected: list[torch.Tensor] = []
    for slot_idx in range(int(future_slots)):
        previous_embeddings = embed_tokens(previous_ids).detach()
        slot_hidden = apply_hidden_correction(
            decoded_future_hidden[:, slot_idx, :], previous_embeddings
        )
        if tuple(slot_hidden.shape) != (int(batch_size), int(hidden_size)):
            raise RuntimeError(
                "Full DFlow/DFly HSC returned wrong slot shape: expected "
                f"{(int(batch_size), int(hidden_size))}, got "
                f"{tuple(slot_hidden.shape)}."
            )
        next_ids = sample_hidden(slot_hidden)
        if (
            next_ids.ndim != 1
            or int(next_ids.numel()) != int(batch_size)
            or next_ids.dtype != torch.long
        ):
            raise RuntimeError(
                "Full DFlow/DFly sampler must return int64 [batch], got "
                f"shape={tuple(next_ids.shape)}, dtype={next_ids.dtype}."
            )
        corrected.append(slot_hidden)
        sampled.append(next_ids)
        # The next HSC sees the actual action produced above, not a parallel
        # preliminary prediction and not a teacher-forced target.
        previous_ids = next_ids
    return torch.stack(sampled, dim=1), torch.stack(corrected, dim=1)


def _assert_tp_replicated_bitwise(value: torch.Tensor, *, label: str) -> None:
    """Fail the optional real-checkpoint gate on any TP-rank bit drift."""

    tp_group = get_tp_group()
    tp_size = int(tp_group.world_size)
    if tp_size == 1:
        return
    flat = value.contiguous().view(-1)
    gathered = torch.empty(
        tp_size * int(flat.numel()), device=flat.device, dtype=flat.dtype
    )
    tp_group.all_gather_into_tensor(gathered, flat)
    gathered = gathered.view(tp_size, int(flat.numel()))
    reference = gathered[0]
    mismatched = [
        rank for rank in range(1, tp_size) if not torch.equal(gathered[rank], reference)
    ]
    if mismatched:
        max_delta = None
        if value.is_floating_point():
            max_delta = max(
                float((gathered[rank].float() - reference.float()).abs().max().item())
                for rank in mismatched
            )
        raise RuntimeError(
            "Full DFlow/DFly TP parity gate failed for "
            f"{label}: mismatched_ranks={mismatched}, max_abs_delta={max_delta}."
        )


class DFlowDFlyFullWorker(DFlashWorker):
    """DFLASH worker with sequential HSC/head sampling after one backbone."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        violations: list[str] = []
        if not bool(getattr(self.draft_model, "is_dflow_dfly_full_model", False)):
            violations.append("draft architecture must be DFlowDFlyFullDraftModel")
        if (
            str(getattr(self.server_args, "speculative_algorithm", "")).upper()
            != "DFLASH"
        ):
            violations.append("speculative_algorithm must be DFLASH")
        if int(self.block_size) != 16:
            violations.append(f"block_size must be 16, got {self.block_size}")
        if int(self.flow_noise_samples) != 1:
            violations.append(
                f"flow_noise_samples must be 1, got {self.flow_noise_samples}"
            )
        if str(self.flow_noise_aggregation) != "latent_mean":
            violations.append(
                "flow_noise_aggregation must be latent_mean, got "
                f"{self.flow_noise_aggregation!r}"
            )
        if self.active_judge_enabled:
            violations.append("active judge is unsupported")
        if self._training_proposal_topk is not None:
            violations.append("training categorical proposal sampling is unsupported")
        if self._collaborative_gate_mode is not None:
            violations.append("collaborative-source proposals are unsupported")
        apply_correction = getattr(self.draft_model, "apply_hidden_correction", None)
        if not callable(apply_correction):
            violations.append("draft model has no apply_hidden_correction()")
        if violations:
            raise ValueError(
                "Full DFlow/DFly worker contract failed: " + "; ".join(violations) + "."
            )
        self._full_tp_parity_steps = _parse_nonnegative_int_env(_TP_PARITY_STEPS_ENV)
        self._full_tp_parity_completed = 0
        if self._full_tp_parity_steps > 0 and int(get_tp_group().world_size) != 2:
            raise ValueError(
                f"{_TP_PARITY_STEPS_ENV}>0 is an exact TP=2 gate; got "
                f"TP={int(get_tp_group().world_size)}."
            )

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ) -> None:
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return
        if batch.has_grammar:
            raise RuntimeError(
                "Full DFlow/DFly received a grammar-constrained batch; the "
                "scheduler must reject it for DFLASH."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            if (
                not is_dflash_sampling_verify_available()
                and not self._warned_sampling_fallback
                and self.tp_rank == 0
            ):
                # Match native DFLASH's verifier fallback notice.  Proposal
                # generation remains deterministic greedy by contract.
                import logging

                logging.getLogger(__name__).warning(
                    "Full DFlow/DFly non-greedy verification is unavailable; "
                    "falling back to greedy verification."
                )
                self._warned_sampling_fallback = True

        batch_size = batch.batch_size()
        # Sole physical draft-backbone invocation in this transaction.
        decoded_hidden, positions_2d, block_ids, lm_head = (
            self._run_draft_block_forward(batch, draft_input)
        )
        parity_active = self._full_tp_parity_completed < self._full_tp_parity_steps
        if parity_active:
            _assert_tp_replicated_bitwise(
                decoded_hidden[:, 1:, :], label="decoded_backbone_future_hidden"
            )

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        apply_correction = self.draft_model.apply_hidden_correction
        self._timing.mark("draft_head_start")
        draft_next, corrected_hidden = _sequential_hsc_greedy(
            decoded_future_hidden=decoded_hidden[:, 1:, :],
            verified_root_ids=block_ids[:, 0],
            embed_tokens=embed_module,
            apply_hidden_correction=apply_correction,
            sample_hidden=lambda hidden: self._greedy_sample_from_vocab_parallel_head(
                hidden_states=hidden, lm_head=lm_head
            ),
        )
        self._timing.mark("draft_head_end")
        if parity_active:
            _assert_tp_replicated_bitwise(
                corrected_hidden, label="predecessor_corrected_future_hidden"
            )
            _assert_tp_replicated_bitwise(draft_next, label="sampled_future_ids")
            self._full_tp_parity_completed += 1

        draft_tokens = self._draft_block_tokens_buf[:batch_size]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions_2d.reshape(-1),
            draft_token_num=self.block_size,
            proposal_hidden=(
                corrected_hidden if self._trace_draft_hidden_enabled else None
            ),
            proposal_behavior_logprobs=None,
            proposal_behavior_support_ids=None,
            proposal_behavior="deterministic_greedy",
            proposal_behavior_topk=None,
            proposal_behavior_temperature=None,
            acceptance_mode="strict",
            collaborative_gate_mode=None,
            collaborative_fixed_draft_tokens=None,
            policy_version=getattr(self.server_args, "weight_version", None),
        )
        _, build_custom_mask = resolve_dflash_verify_mask_policy(
            self.model_runner.attn_backend
        )
        verify_input.prepare_for_verify(
            batch, self.page_size, build_custom_mask=build_custom_mask
        )
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False


__all__ = ["DFlowDFlyFullWorker"]
