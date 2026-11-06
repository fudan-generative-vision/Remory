from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.ddtree_utils import follow_verified_tree
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


@dataclass
class DDTreeVerifyInput(SpecInput):
    """Target-model verify input for a fixed-width DDTree.

    The runtime uses a fixed verify width, so target-model execution may be
    CUDA-graph replayed while tree construction and path/KV commit remain eager:
      * greedy decoding only;
      * page_size == 1;
      * fixed ``draft_token_num == tree_budget + 1`` rows per request;
      * dynamic masks/topology are copied into graph-owned buffers before replay.

    Node 0 is the already verified root token. Remaining nodes are DFlash
    proposals selected by the DDTree heap.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    tree_budget: int
    visibility: torch.Tensor  # [bs, N, N]
    child_maps: List[Dict[int, Dict[int, int]]]
    retrive_next_token: torch.Tensor  # spelling follows existing SGLang Eagle API
    retrive_next_sibling: torch.Tensor
    parents: Optional[torch.Tensor] = None
    topk: int = 2  # >1 enables tree topology in HybridLinearAttnBackend
    custom_mask: Optional[torch.Tensor] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    num_tokens_per_batch: int = -1

    # Populated by verify(); consumed by DDTreeWorker for hybrid/Mamba commit.
    accepted_node_indices: Optional[torch.Tensor] = None
    accepted_path_lens: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DDTREE_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)
        if int(self.draft_token_num) != int(self.tree_budget) + 1:
            raise ValueError(
                "DDTree requires draft_token_num == tree_budget + 1, got "
                f"draft_token_num={self.draft_token_num}, tree_budget={self.tree_budget}."
            )

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return
        if page_size != 1:
            raise RuntimeError(
                "DDTree backport v1 only supports page_size=1. "
                f"Got page_size={page_size}."
            )

        bs = batch.batch_size()
        expected_tokens = bs * int(self.draft_token_num)
        if int(self.draft_token.numel()) != expected_tokens:
            raise RuntimeError(
                "DDTree draft-token shape mismatch: "
                f"numel={self.draft_token.numel()}, expected={expected_tokens}."
            )
        if tuple(self.visibility.shape) != (
            bs,
            int(self.draft_token_num),
            int(self.draft_token_num),
        ):
            raise RuntimeError(
                "DDTree visibility shape mismatch: "
                f"got={tuple(self.visibility.shape)}, expected={(bs, self.draft_token_num, self.draft_token_num)}."
            )

        batch.input_ids = self.draft_token
        batch.out_cache_loc = alloc_token_slots(batch.tree_cache, expected_tokens)
        if batch.out_cache_loc is None:
            raise RuntimeError(
                f"DDTree target verify OOM while allocating {expected_tokens} slots."
            )

        end_offset = batch.seq_lens + int(self.draft_token_num)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # Build FlashInfer's flattened allow-mask. Every tree row can attend to
        # the historical prefix and only to its ancestors/self inside the tree.
        mask_chunks: List[torch.Tensor] = []
        for batch_idx, prefix_len in enumerate(batch.seq_lens_cpu.tolist()):
            prefix_len_i = int(prefix_len)
            prefix_allow = torch.ones(
                (self.draft_token_num, prefix_len_i),
                dtype=torch.bool,
                device=batch.device,
            )
            tree_allow = self.visibility[batch_idx].to(
                device=batch.device, dtype=torch.bool
            )
            mask_chunks.append(torch.cat([prefix_allow, tree_allow], dim=1).flatten())
        self.custom_mask = (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

        # Match Eagle target-verify setup for Qwen3.5 hybrid linear attention.
        if get_global_server_args().enable_mamba_extra_buffer():
            batch.mamba_track_indices = torch.tensor(
                [
                    req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.mamba_track_mask = None
            batch.mamba_track_seqlens = None

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )

        if self.custom_mask is None:
            raise RuntimeError("DDTree verify requires a compiled custom attention mask.")
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * bs
        )
        mask = self.custom_mask
        if mask.numel() < mask_numel:
            # CUDA-graph replay may pad the request batch to a captured batch
            # size. Dummy requests use an allow-all mask; their outputs are
            # discarded by the graph runner.
            mask = torch.cat(
                [
                    mask,
                    torch.full(
                        (mask_numel - mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )
        elif mask.numel() > mask_numel:
            # Capture-time dummy masks may be backed by a max-batch buffer.
            mask = mask[:mask_numel]
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []
        if page_size != 1:
            raise RuntimeError("DDTree backport v1 verify requires page_size=1.")
        if batch.has_grammar:
            raise RuntimeError("DDTree backport v1 does not support grammar constraints.")
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise RuntimeError("DDTree backport v1 supports greedy decoding only.")

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device
        num_nodes = int(self.draft_token_num)

        # Match the normal greedy target path before tree traversal. Even when
        # temperature is zero, requests may carry repetition penalties, logit
        # bias, or a custom logit processor. Grammar is rejected above because
        # tree-wide grammar state branching is not implemented in v1.
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DDTree verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )
            if sampling_info.has_custom_logit_processor:
                raise RuntimeError(
                    "DDTree backport v1 does not support custom logit processors; "
                    "their state would need to be evaluated separately on every tree path."
                )
            if sampling_info.penalizer_orchestrator.is_required:
                raise RuntimeError(
                    "DDTree backport v1 does not support repetition/frequency/presence "
                    "penalties because the penalty history differs across tree paths."
                )
            if sampling_info.logit_bias is not None:
                linear_bias = torch.zeros(
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_bias)
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(linear_bias, num_nodes, dim=0)
                )

        target_predict = torch.argmax(
            logits_output.next_token_logits, dim=-1
        ).view(bs, num_nodes)
        target_predict_cpu = target_predict.to(device="cpu", dtype=torch.long)
        draft_tokens_2d = self.draft_token.view(bs, num_nodes)
        draft_tokens_cpu = draft_tokens_2d.to(device="cpu", dtype=torch.long)

        accepted_paths: List[List[int]] = []
        commit_paths: List[List[int]] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []
        accept_length_per_req_cpu: List[int] = []

        for batch_idx, req in enumerate(batch.reqs):
            path, bonus = follow_verified_tree(
                child_map=self.child_maps[batch_idx],
                target_predictions=target_predict_cpu[batch_idx].tolist(),
            )
            accepted_paths.append(path)

            proposed = [
                int(draft_tokens_cpu[batch_idx, node_idx].item())
                for node_idx in path[1:]
            ]
            proposed.append(int(bonus))

            appended = 0
            for token_id in proposed:
                req.output_ids.append(int(token_id))
                appended += 1
                req.check_finished()
                if req.finished():
                    break

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DDTree verify cannot determine the current token for an empty request."
                )

            # Same accounting convention as DFlash: output append count equals
            # the number of verify rows whose KV/hidden state is committed.
            commit_len = int(appended)
            if commit_len <= 0 or commit_len > len(path):
                raise RuntimeError(
                    "DDTree commit/path mismatch: "
                    f"commit_len={commit_len}, accepted_path_len={len(path)}."
                )
            committed_path = path[:commit_len]
            commit_paths.append(committed_path)
            commit_lens_cpu.append(commit_len)
            new_verified_list.append(new_verified_token)
            accepted_drafts = max(0, appended - 1)
            accept_length_per_req_cpu.append(accepted_drafts)

            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accepted_drafts
            if hasattr(req, "update_spec_acceptance_histogram"):
                req.update_spec_acceptance_histogram(accepted_drafts)

        commit_lens = torch.tensor(
            commit_lens_cpu, dtype=torch.int32, device=device
        )
        new_verified_id = torch.tensor(
            new_verified_list, dtype=torch.int64, device=device
        )

        # Keep arbitrary accepted tree slots and remap them to consecutive logical
        # sequence positions. This avoids per-layer KV copies when page_size == 1.
        out_cache_loc = batch.out_cache_loc.view(bs, num_nodes)
        keep_mask = torch.zeros((bs, num_nodes), dtype=torch.bool, device=device)
        selected_loc_segments: List[torch.Tensor] = []
        accepted_node_tensor = torch.full(
            (bs, num_nodes), -1, dtype=torch.long, device=device
        )
        for batch_idx, path in enumerate(commit_paths):
            path_tensor = torch.tensor(path, dtype=torch.long, device=device)
            keep_mask[batch_idx, path_tensor] = True
            selected_loc_segments.append(out_cache_loc[batch_idx, path_tensor])
            accepted_node_tensor[batch_idx, : len(path)] = path_tensor

        batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
        batch.out_cache_loc = torch.cat(selected_loc_segments, dim=0)
        self.accepted_node_indices = accepted_node_tensor
        self.accepted_path_lens = commit_lens.to(torch.long)

        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DDTree verify requires target hidden states.")
        hidden = hidden.view(bs, num_nodes, -1)
        hidden_segments: List[torch.Tensor] = []
        for batch_idx, path in enumerate(commit_paths):
            path_tensor = torch.tensor(path, dtype=torch.long, device=hidden.device)
            hidden_segments.append(hidden[batch_idx].index_select(0, path_tensor))
        next_target_hidden = torch.cat(hidden_segments, dim=0)
        logits_output.hidden_states = None

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        )
