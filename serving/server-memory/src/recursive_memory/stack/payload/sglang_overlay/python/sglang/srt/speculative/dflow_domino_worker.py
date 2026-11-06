"""Isolated exact-greedy worker for plain one-step DFlow-Domino.

The inherited worker performs the sole physical DFlow block forward.  This
subclass replaces only proposal selection with the causal Domino head, then
hands the unchanged ``[verified root, 15 proposals]`` block to the ordinary
DFlash verifier.  The first implementation deliberately supports only exact
full-vocabulary greedy decoding.
"""

from __future__ import annotations

import logging
import math
import os

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
from sglang.srt.speculative.dflash_worker import DFlashWorker
from sglang.srt.speculative.dflow_domino_utils import (
    DFlowDominoVocabShard,
    dflow_domino_greedy_rollout,
    validate_dflow_domino_runtime,
)


_TP_PARITY_STEPS_ENV = "SGLANG_DFLOW_DOMINO_TP_PARITY_STEPS"
DOMINO_RUNTIME_CONTRACT_LOG = (
    "DFlow-Domino runtime contract enabled: mode=full_vocab_exact tp=2 "
    "block=16 m=1 proposals=15 sequential_kv=1 overlap=0 one_backbone=1 "
    "candidate_pool_size=0 verifier_bonus=1 full_tail_gaussian=1 "
    "padding=whole_block_zero_only loss_mask=supervision_only_not_latent "
    "block_attention=prefix_lt_anchor_plus_bidirectional_same_block"
)
logger = logging.getLogger(__name__)


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


def _assert_tp_identical(value: torch.Tensor, *, label: str) -> None:
    """Fail a bounded startup gate if replicated TP results differ bitwise."""

    tp_group = get_tp_group()
    tp_size = int(tp_group.world_size)
    if tp_size == 1:
        return
    flat = value.contiguous().view(-1)
    gathered = torch.empty(
        tp_size * int(flat.numel()),
        device=flat.device,
        dtype=flat.dtype,
    )
    tp_group.all_gather_into_tensor(gathered, flat)
    gathered = gathered.view(tp_size, int(flat.numel()))
    reference = gathered[0]
    mismatched = [
        rank for rank in range(1, tp_size) if not torch.equal(gathered[rank], reference)
    ]
    if mismatched:
        raise RuntimeError(
            "Plain DFlow-Domino TP parity gate failed for "
            f"{label}: mismatched_ranks={mismatched}."
        )


def _worker_contract_violations(worker, *, environ=None) -> list[str]:
    """Return every unsupported runtime option before serving any request."""

    if environ is None:
        environ = os.environ
    violations: list[str] = []
    draft_model = worker.draft_model
    if getattr(draft_model, "is_dflow_domino_model", None) is not True:
        violations.append("draft architecture must be DFlowDominoDraftModel")
    if getattr(draft_model, "domino_source_arm", None) != "base":
        violations.append("draft checkpoint must be the plain/base Stage-L arm")
    if (
        str(getattr(worker.server_args, "speculative_algorithm", "")).upper()
        != "DFLASH"
    ):
        violations.append("speculative_algorithm must be DFLASH")
    if str(worker.flow_mode) != "one_step":
        violations.append(f"flow_mode must be one_step, got {worker.flow_mode!r}")
    if int(worker.flow_steps) != 1:
        violations.append(f"flow_steps must be 1, got {worker.flow_steps}")
    if int(worker.block_size) != 16:
        violations.append(f"block_size must be 16, got {worker.block_size}")
    if int(worker.flow_noise_samples) != 1:
        violations.append(
            f"flow_noise_samples must be 1, got {worker.flow_noise_samples}"
        )
    if str(worker.flow_noise_aggregation) != "latent_mean":
        violations.append(
            "flow_noise_aggregation must be latent_mean, got "
            f"{worker.flow_noise_aggregation!r}"
        )
    if (
        not math.isfinite(float(worker.flow_noise_scale))
        or float(worker.flow_noise_scale) != 2.0
    ):
        violations.append(
            f"flow_noise_scale must be exactly 2.0, got {worker.flow_noise_scale}"
        )
    if worker.use_compact_draft_cache:
        violations.append("draft_window_size/compact draft cache is unsupported")
    if worker.active_judge_enabled:
        violations.append("active multi-block judge is unsupported")
    if worker._training_proposal_topk is not None:
        violations.append("training categorical proposal sampling is unsupported")
    if worker._collaborative_gate_mode is not None:
        violations.append("collaborative-source proposals are unsupported")
    if worker._trace_draft_hidden_enabled:
        violations.append("training draft-hidden trace is unsupported")
    if worker._strict_one_forward_enabled:
        violations.append("legacy Native MASK strict-one-forward mode is unsupported")
    if not worker._dflow_strict_one_backbone_enabled:
        violations.append(
            "SGLANG_DFLOW_STRICT_ONE_BACKBONE=1 is required for the formal runtime"
        )
    if environ.get("SGLANG_DFLASH_DISABLE_FUSED_KV_MATERIALIZATION") != "1":
        violations.append(
            "SGLANG_DFLASH_DISABLE_FUSED_KV_MATERIALIZATION=1 is required"
        )
    if getattr(worker.server_args, "disable_overlap_schedule", None) is not True:
        violations.append("disable_overlap_schedule must be true")
    return violations


class DFlowDominoWorker(DFlashWorker):
    """One-backbone, causal-head, exact-full-vocabulary DFlash worker."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        violations = _worker_contract_violations(self)
        if violations:
            raise ValueError(
                "Plain DFlow-Domino worker contract failed: "
                + "; ".join(violations)
                + "."
            )

        target_model = self.target_worker.model_runner.model
        target_embedding = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not isinstance(
            getattr(lm_head, "weight", None), torch.Tensor
        ):
            raise ValueError(
                "Plain DFlow-Domino requires a target lm_head weight tensor."
            )
        if not torch.is_floating_point(lm_head.weight):
            raise ValueError(
                "Plain DFlow-Domino exact base projection does not support a "
                "quantized lm_head."
            )
        if getattr(lm_head, "bias", None) is not None:
            raise ValueError("Plain DFlow-Domino requires a bias-free target lm_head.")

        prefix_gru = getattr(self.draft_model, "prefix_gru", None)
        residual_down = getattr(self.draft_model, "residual_down", None)
        residual_up = getattr(self.draft_model, "residual_up", None)
        tp_group = get_tp_group()
        if int(tp_group.world_size) != 2:
            raise ValueError(
                "Plain DFlow-Domino v1 requires tensor parallel size 2, got "
                f"{tp_group.world_size}."
            )
        vocab_shard = validate_dflow_domino_runtime(
            tp_size=int(tp_group.world_size),
            tp_rank=int(self.tp_rank),
            target_vocab_size=int(self.model_runner.model_config.vocab_size),
            draft_vocab_size=int(self.draft_model_runner.model_config.vocab_size),
            hidden_size=int(self.draft_model.config.hidden_size),
            target_embedding=target_embedding,
            lm_head=lm_head,
            prefix_gru=prefix_gru,
            residual_down=residual_down,
            residual_up=residual_up,
            pure_draft_prefix_len=int(self.draft_model.pure_draft_prefix_len),
            residual_rank=int(self.draft_model.residual_rank),
        )
        self._domino_target_embedding = target_embedding
        self._domino_lm_head = lm_head
        self._domino_prefix_gru = prefix_gru
        self._domino_residual_down = residual_down
        self._domino_residual_up = residual_up
        self._domino_tp_group = tp_group
        self._domino_vocab_shard: DFlowDominoVocabShard = vocab_shard
        self._domino_tp_parity_steps = _parse_nonnegative_int_env(_TP_PARITY_STEPS_ENV)
        self._domino_tp_parity_completed = 0
        if int(self.tp_rank) == 0:
            logger.info(DOMINO_RUNTIME_CONTRACT_LOG)

    @torch.no_grad()
    def _prepare_for_speculative_decoding(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return
        if batch.has_grammar:
            raise RuntimeError(
                "Plain DFlow-Domino received a grammar-constrained batch; the "
                "scheduler must reject grammar for DFLASH."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise RuntimeError(
                "Plain DFlow-Domino v1 supports greedy requests only. Sequential "
                "categorical q and exact q-aware verification are not implemented."
            )

        batch_size = batch.batch_size()
        # The inherited transaction owns and counts the sole physical DFlow
        # backbone call, including prepare_hidden_for_lm_head().
        draft_hidden, positions_2d, block_ids, lm_head = self._run_draft_block_forward(
            batch, draft_input
        )
        if lm_head is not self._domino_lm_head:
            raise RuntimeError(
                "Plain DFlow-Domino target lm_head changed after startup validation."
            )

        self._timing.mark("draft_head_start")
        draft_next = dflow_domino_greedy_rollout(
            draft_hidden=draft_hidden,
            verified_ids=block_ids[:, 0],
            target_embedding=self._domino_target_embedding,
            lm_head_weight=lm_head.weight,
            prefix_gru=self._domino_prefix_gru,
            residual_down=self._domino_residual_down,
            residual_up=self._domino_residual_up,
            vocab_size=int(self.model_runner.model_config.vocab_size),
            pure_draft_prefix_len=1,
            candidate_pool_size=0,
            greedy_mode="full_vocab",
            tp_group=self._domino_tp_group,
            vocab_shard=self._domino_vocab_shard,
            trace_hook=None,
        )
        self._timing.mark("draft_head_end")
        expected_shape = (int(batch_size), int(self.block_size) - 1)
        if tuple(draft_next.shape) != expected_shape or draft_next.dtype != torch.long:
            raise RuntimeError(
                "Plain DFlow-Domino rollout returned an invalid proposal block: "
                f"expected int64 {expected_shape}, got shape={tuple(draft_next.shape)}, "
                f"dtype={draft_next.dtype}."
            )

        if self._domino_tp_parity_completed < self._domino_tp_parity_steps:
            _assert_tp_identical(draft_next, label="sampled_proposal_ids")
            self._domino_tp_parity_completed += 1

        draft_tokens = self._draft_block_tokens_buf[:batch_size]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        # Domino returns only slot1..slot15.  The ordinary verifier below owns
        # the correction/bonus token, including the all-15-accepted case.
        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions_2d.reshape(-1),
            draft_token_num=self.block_size,
            proposal_hidden=None,
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
            batch,
            self.page_size,
            build_custom_mask=build_custom_mask,
        )
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False


__all__ = [
    "DOMINO_RUNTIME_CONTRACT_LOG",
    "DFlowDominoWorker",
]
