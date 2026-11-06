from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ddtree_info import DDTreeVerifyInput
from sglang.srt.speculative.ddtree_utils import (
    DDTreeBuildResult,
    build_candidate_union_tree,
    build_ddtree_tree,
    compile_ddtree_inputs,
)
from sglang.srt.speculative.dflash_info import DFlashDraftInput
from sglang.srt.speculative.dflash_worker import DFlashWorker

logger = logging.getLogger(__name__)


class DDTreeWorker(DFlashWorker):
    """Fixed-width DDTree backport on top of the PR22077 DFlash worker.

    Tree construction and commit remain eager. When global CUDA Graph is enabled,
    the shared DFlash draft block and fixed-width target verify forward are replayed.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        self.tree_budget = int(server_args.speculative_ddtree_budget)
        self.tree_topk = int(server_args.speculative_ddtree_topk)
        self.adaptive_topk = bool(
            getattr(server_args, "speculative_ddtree_adaptive_topk", False)
        )
        self.topk_mass = float(
            getattr(server_args, "speculative_ddtree_topk_mass", 0.98)
        )
        self.topk_min = int(
            getattr(server_args, "speculative_ddtree_topk_min", 2)
        )
        self.max_tree_nodes = self.tree_budget + 1
        self._logged_adaptive_topk = False
        self._contextual_jit_same_state_sc_dual_proposal = bool(
            self.__dict__.get(
                "_dflow_contextual_jit_same_state_sc_dual_proposal",
                False,
            )
        )
        self._contextual_jit_nfe_endpoint_candidate_union = bool(
            self.__dict__.get(
                "_dflow_contextual_jit_nfe_endpoint_candidate_union",
                False,
            )
        )
        noise_candidate_union = (
            str(getattr(self, "flow_noise_aggregation", "mean")).lower()
            == "candidate_union"
        )
        self._candidate_union_source = (
            "contextual_jit_same_state_sc"
            if self._contextual_jit_same_state_sc_dual_proposal
            else (
                "contextual_jit_nfe_endpoints"
                if self._contextual_jit_nfe_endpoint_candidate_union
                else ("noise_ensemble" if noise_candidate_union else None)
            )
        )
        self._candidate_union_enabled = self._candidate_union_source is not None
        self._candidate_union_proposal_count = (
            2
            if self._contextual_jit_same_state_sc_dual_proposal
            else (
                int(getattr(self, "flow_steps", 1))
                if self._contextual_jit_nfe_endpoint_candidate_union
                else int(getattr(self, "flow_noise_samples", 1))
            )
        )
        self._candidate_union_cycles = 0
        self._candidate_union_requests = 0
        self._candidate_union_unique_chain_sum = 0
        self._candidate_union_actual_node_sum = 0
        self._last_candidate_union_stats = None

        if self.page_size != 1:
            raise RuntimeError(
                "DDTree backport v1 requires page_size=1, got "
                f"page_size={self.page_size}."
            )
        if self.tree_budget < 1:
            raise ValueError(
                f"DDTree tree budget must be positive, got {self.tree_budget}."
            )
        if self.tree_topk < 1:
            raise ValueError(
                f"DDTree candidate top-k must be positive, got {self.tree_topk}."
            )
        if self.topk_min < 1 or self.topk_min > self.tree_topk:
            raise ValueError(
                "DDTree adaptive min top-k must be in [1, tree_topk], got "
                f"min={self.topk_min}, tree_topk={self.tree_topk}."
            )
        if not 0.0 < self.topk_mass <= 1.0:
            raise ValueError(
                f"DDTree top-k mass must be in (0, 1], got {self.topk_mass}."
            )
        if self._candidate_union_enabled:
            proposal_count = int(self._candidate_union_proposal_count)
            if proposal_count <= 1:
                raise ValueError(
                    "DDTree candidate_union requires more than one candidate "
                    f"chain, got M={proposal_count}."
                )
            expected_budget = proposal_count * (int(self.block_size) - 1)
            if self.tree_budget != expected_budget:
                raise ValueError(
                    "DDTree candidate_union requires tree_budget == M * (B - 1) "
                    "so every sample chain has a fixed verifier slot budget: "
                    f"budget={self.tree_budget}, M={proposal_count}, "
                    f"B={self.block_size}, expected={expected_budget}."
                )

        if self.tp_rank == 0:
            logger.info(
                "Initialized DDTree worker (fixed-width CUDA-graph capable v2). "
                "block_size=%s, draft_horizon=%s, tree_budget=%s, "
                "max_tree_nodes=%s, topk_max=%s, adaptive_topk=%s, "
                "topk_mass=%s, topk_min=%s, candidate_union=%s, "
                "candidate_union_source=%s, candidate_count=%s, "
                "noise_samples=%s, cuda_graph=%s",
                self.block_size,
                self.block_size - 1,
                self.tree_budget,
                self.max_tree_nodes,
                self.tree_topk,
                self.adaptive_topk,
                self.topk_mass,
                self.topk_min,
                self._candidate_union_enabled,
                self._candidate_union_source,
                self._candidate_union_proposal_count,
                int(getattr(self, "flow_noise_samples", 1)),
                not server_args.disable_cuda_graph,
            )

    def _get_candidate_union_sample_hidden(self) -> torch.Tensor:
        """Read the latest prepared per-candidate hidden states from DFlashWorker.

        DFlashWorker owns the proposal execution and the exact prepare-for-LM-
        head boundary. Its consuming accessor clears the transaction-local
        publication, preventing a later verifier round from accidentally
        reusing stale candidate states.
        """

        sample_hidden = self.take_dflow_candidate_sample_hiddens()
        if sample_hidden is None:
            raise RuntimeError(
                "DDTree candidate_union requires DFlashWorker to expose the "
                "latest prepared candidate hidden states after its proposal "
                "forward(s)."
            )
        if not isinstance(sample_hidden, torch.Tensor):
            raise RuntimeError(
                "DDTree candidate_union sample hidden state must be a tensor, got "
                f"{type(sample_hidden).__name__}."
            )
        return sample_hidden

    def _candidate_union_chains_from_sample_hidden(
        self,
        *,
        sample_hidden: torch.Tensor,
        batch_size: int,
        lm_head,
    ) -> torch.Tensor:
        """Project ``[M, bs, B, H]`` prepared states into ``[M, bs, B-1]`` chains."""

        if sample_hidden.ndim != 4:
            raise RuntimeError(
                "DDTree candidate_union expects prepared sample hidden states "
                f"with shape [M, bs, B, H], got {tuple(sample_hidden.shape)}."
            )
        proposal_count, hidden_bs, hidden_block, hidden_size = map(
            int, sample_hidden.shape
        )
        expected_proposals = int(
            self.__dict__.get(
                "_candidate_union_proposal_count",
                self.__dict__.get("flow_noise_samples", 1),
            )
        )
        if proposal_count != expected_proposals:
            raise RuntimeError(
                "DDTree candidate_union sample-count mismatch: "
                f"hidden M={proposal_count}, configured M={expected_proposals}, "
                f"source={self.__dict__.get('_candidate_union_source')!r}."
            )
        if hidden_bs != int(batch_size):
            raise RuntimeError(
                "DDTree candidate_union batch mismatch: "
                f"hidden batch={hidden_bs}, scheduler batch={batch_size}."
            )
        if hidden_block != int(self.block_size):
            raise RuntimeError(
                "DDTree candidate_union block mismatch: "
                f"hidden B={hidden_block}, configured B={self.block_size}."
            )
        if hidden_size <= 0:
            raise RuntimeError(
                "DDTree candidate_union hidden dimension must be positive."
            )

        proposal_hidden = sample_hidden[:, :, 1:, :]
        horizon = int(self.block_size) - 1
        expected_tokens = proposal_count * int(batch_size) * horizon
        # Candidate union should issue one large shared-head projection for the
        # common M=16, C<=16, B=16 operating point (3840 rows), rather than the
        # default 256-row loop. Cap the temporary TP1 logits near 1.2 GiB.
        projection_chunk_size = min(4096, max(256, expected_tokens))
        candidate_tokens = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=proposal_hidden.reshape(-1, hidden_size),
            lm_head=lm_head,
            chunk_size=projection_chunk_size,
        )
        if int(candidate_tokens.numel()) != expected_tokens:
            raise RuntimeError(
                "DDTree candidate_union LM-head output size mismatch: "
                f"got={candidate_tokens.numel()}, expected={expected_tokens}."
            )
        return candidate_tokens.view(proposal_count, int(batch_size), horizon)

    def _record_candidate_union_stats(self, tree: DDTreeBuildResult) -> None:
        """Persist bounded structured diagnostics in worker state and server logs."""

        if tree.actual_node_counts is None or tree.unique_chain_counts is None:
            raise RuntimeError(
                "DDTree candidate_union tree is missing union diagnostics."
            )
        actual_node_counts = tree.actual_node_counts.to(
            device="cpu", dtype=torch.long
        )
        unique_chain_counts = tree.unique_chain_counts.to(
            device="cpu", dtype=torch.long
        )
        request_count = int(actual_node_counts.numel())
        if request_count != int(unique_chain_counts.numel()):
            raise RuntimeError(
                "DDTree candidate_union diagnostic batch mismatch: "
                f"nodes={request_count}, chains={unique_chain_counts.numel()}."
            )

        self._candidate_union_cycles += 1
        self._candidate_union_requests += request_count
        self._candidate_union_actual_node_sum += int(actual_node_counts.sum().item())
        self._candidate_union_unique_chain_sum += int(
            unique_chain_counts.sum().item()
        )
        stats = {
            "cycle": self._candidate_union_cycles,
            "batch_size": request_count,
            "unique_chain_count_mean": float(
                unique_chain_counts.float().mean().item()
            ),
            "unique_chain_count_min": int(unique_chain_counts.min().item()),
            "unique_chain_count_max": int(unique_chain_counts.max().item()),
            "actual_trie_nodes_mean": float(
                actual_node_counts.float().mean().item()
            ),
            "actual_trie_nodes_min": int(actual_node_counts.min().item()),
            "actual_trie_nodes_max": int(actual_node_counts.max().item()),
            "padded_verify_width": int(self.max_tree_nodes),
        }
        self._last_candidate_union_stats = stats

        if self.tp_rank == 0 and (
            self._candidate_union_cycles == 1
            or self._candidate_union_cycles % 100 == 0
        ):
            logger.info(
                "DDTree candidate_union stats: cycle=%d batch=%d "
                "unique_chains=%.2f[%d,%d] actual_trie_nodes=%.2f[%d,%d] "
                "padded_verify_width=%d cumulative_unique_chains=%.2f "
                "cumulative_actual_trie_nodes=%.2f",
                stats["cycle"],
                stats["batch_size"],
                stats["unique_chain_count_mean"],
                stats["unique_chain_count_min"],
                stats["unique_chain_count_max"],
                stats["actual_trie_nodes_mean"],
                stats["actual_trie_nodes_min"],
                stats["actual_trie_nodes_max"],
                stats["padded_verify_width"],
                self._candidate_union_unique_chain_sum
                / max(1, self._candidate_union_requests),
                self._candidate_union_actual_node_sum
                / max(1, self._candidate_union_requests),
            )

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return
        if batch.has_grammar:
            raise RuntimeError("DDTree backport v1 does not support grammar constraints.")
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise RuntimeError("DDTree backport v1 supports greedy decoding only.")

        bs = batch.batch_size()
        draft_hidden, _, block_ids, lm_head = self._run_draft_block_forward(
            batch, draft_input
        )
        if self._candidate_union_enabled:
            sample_hidden = self._get_candidate_union_sample_hidden()
            candidate_token_chains = self._candidate_union_chains_from_sample_hidden(
                sample_hidden=sample_hidden,
                batch_size=bs,
                lm_head=lm_head,
            )
            tree = build_candidate_union_tree(
                candidate_token_chains=candidate_token_chains,
                tree_budget=self.tree_budget,
                device=batch.device,
            )
            self._record_candidate_union_stats(tree)
        else:
            proposal_hidden = draft_hidden[:, 1:, :]
            draft_horizon = int(proposal_hidden.shape[1])

            top_log_probs, top_token_ids = (
                self._topk_log_probs_from_vocab_parallel_head(
                    hidden_states=proposal_hidden.reshape(
                        -1, proposal_hidden.shape[-1]
                    ),
                    lm_head=lm_head,
                    topk=self.tree_topk,
                )
            )
            top_log_probs = top_log_probs.view(bs, draft_horizon, -1)
            top_token_ids = top_token_ids.view(bs, draft_horizon, -1)

            # Keep Kmax static for CUDA Graph compatibility. Only active_k values
            # change between rounds; the Python heap remains outside the graph.
            if self.adaptive_topk:
                cumulative_mass = top_log_probs.float().exp().cumsum(dim=-1)
                reached = cumulative_mass >= self.topk_mass
                first_reached = reached.float().argmax(dim=-1).to(torch.long) + 1
                active_k = torch.where(
                    reached.any(dim=-1),
                    first_reached,
                    torch.full_like(first_reached, top_log_probs.shape[-1]),
                )
                active_k = active_k.clamp(
                    min=self.topk_min, max=top_log_probs.shape[-1]
                )
            else:
                active_k = torch.full(
                    top_log_probs.shape[:2],
                    top_log_probs.shape[-1],
                    dtype=torch.long,
                    device=top_log_probs.device,
                )

            if not self._logged_adaptive_topk and self.tp_rank == 0:
                logger.info(
                    "DDTree candidate truncation: adaptive=%s, Kmax=%d, "
                    "Kmin=%d, mass=%.4f, observed_mean_K=%.2f, "
                    "observed_max_K=%d",
                    self.adaptive_topk,
                    self.tree_topk,
                    self.topk_min,
                    self.topk_mass,
                    float(active_k.float().mean().item()),
                    int(active_k.max().item()),
                )
                self._logged_adaptive_topk = True

            tree = build_ddtree_tree(
                top_log_probs=top_log_probs,
                top_token_ids=top_token_ids,
                tree_budget=self.tree_budget,
                device=batch.device,
                active_k=active_k,
            )
        verify_ids, verify_positions = compile_ddtree_inputs(
            root_token_ids=block_ids[:, 0],
            start_positions=batch.seq_lens,
            tree=tree,
        )

        verify_input = DDTreeVerifyInput(
            draft_token=verify_ids.reshape(-1),
            positions=verify_positions.reshape(-1),
            draft_token_num=self.max_tree_nodes,
            tree_budget=self.tree_budget,
            visibility=tree.visibility,
            child_maps=tree.child_maps,
            retrive_next_token=tree.retrieve_next_token,
            retrive_next_sibling=tree.retrieve_next_sibling,
            parents=tree.parents,
            topk=2,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        verify_input.prepare_for_verify(batch, self.page_size)

        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _update_target_mamba_state_after_ddtree_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
        verify_input: DDTreeVerifyInput,
    ) -> None:
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return
        if verify_input.accepted_node_indices is None:
            raise RuntimeError(
                "DDTree verify did not publish accepted node indices for Mamba commit."
            )

        accepted_paths = verify_input.accepted_node_indices
        row_ids = torch.arange(
            commit_lens.shape[0], dtype=torch.long, device=commit_lens.device
        )
        last_ordinals = commit_lens.to(torch.long) - 1
        accepted_steps = accepted_paths[row_ids, last_ordinals]

        mamba_steps_to_track = None
        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ordinal = torch.clamp(
                tracking_point - seq_lens_pre_verify - 1, min=0
            ).to(torch.long)
            can_track_mask = to_track_mask & (
                to_track_ordinal < commit_lens.to(to_track_ordinal.dtype)
            )
            safe_ordinal = torch.minimum(
                to_track_ordinal,
                (commit_lens.to(torch.long) - 1).clamp_min(0),
            )
            tracked_nodes = accepted_paths[row_ids, safe_ordinal]
            mamba_steps_to_track = torch.where(
                can_track_mask,
                tracked_nodes,
                torch.full_like(tracked_nodes, -1),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        # Reuse DFlash's prefill and ModelWorkerBatch fallback verbatim.
        if (
            isinstance(batch, ModelWorkerBatch)
            or batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
        ):
            return super().forward_batch_generation(batch, **kwargs)

        if getattr(batch, "return_logprob", False):
            raise RuntimeError("DDTree backport v1 does not support return_logprob.")

        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DDTree decode requires DFlashDraftInput state after prefill."
            )

        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        if not isinstance(verify_input, DDTreeVerifyInput):
            raise RuntimeError(
                f"Expected DDTreeVerifyInput, got {type(verify_input).__name__}."
            )

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_ddtree_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
                verify_input=verify_input,
            )

        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        # Contextual hidden-space DFlow takes the post-final-norm target
        # feature h[p-1] as the next verified root anchor.  Without this
        # refresh, DDTree would correctly commit tokens but use the previous
        # round's root feature beginning with its second proposal round.
        draft_input.router_target_hidden = self._maybe_last_target_hidden_per_request(
            next_target_hidden, commit_lens
        )
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DDTree verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
