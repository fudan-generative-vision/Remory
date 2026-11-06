from __future__ import annotations

import json
import logging
import os
import queue
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import (
    SpecInput,
    SpecInputType,
    SpeculativeAlgorithm,
)
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.dasync_ipc import DAsyncIPCError, DAsyncUnixSocketClient
from sglang.srt.speculative.dasync_draft_runtime import _dasync_normalize_proposal_mode
from sglang.srt.speculative.dasync_local_engine import DAsyncLocalDraftEngine

logger = logging.getLogger(__name__)


def _dasync_env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dasync_dump_enabled() -> bool:
    return _dasync_env_flag("SGLANG_DASYNC_DUMP", False)


def _dasync_dump_path() -> str:
    return os.environ.get("SGLANG_DASYNC_DUMP_FILE", "/tmp/sglang_dasync_trace.jsonl")


def _dasync_dump_max_items() -> int:
    try:
        return max(1, int(os.environ.get("SGLANG_DASYNC_DUMP_MAX_ITEMS", "128")))
    except Exception:
        return 128


def _dasync_clip_list(values, limit: Optional[int] = None):
    if values is None:
        return []
    if limit is None:
        limit = _dasync_dump_max_items()
    out = []
    for x in list(values)[: int(limit)]:
        try:
            if isinstance(x, float):
                out.append(float(x))
            else:
                out.append(int(x))
        except Exception:
            out.append(x)
    return out


def _dasync_decode_ids(tokenizer, ids) -> Optional[List[str]]:
    if tokenizer is None:
        return None
    if not _dasync_env_flag("SGLANG_DASYNC_DUMP_DECODE", True):
        return None
    try:
        return [
            tokenizer.decode([int(x)], skip_special_tokens=False)
            for x in _dasync_clip_list(ids)
        ]
    except Exception as exc:
        return [f"<decode-error:{type(exc).__name__}>"]


def _dasync_dump_event(event: Dict[str, Any]) -> None:
    if not _dasync_dump_enabled():
        return
    try:
        path = _dasync_dump_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        event = dict(event)
        event.setdefault("ts", time.time())
        event.setdefault("pid", os.getpid())
        with open(path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        logger.debug("failed to write DASYNC dump event", exc_info=True)


def _dasync_load_dump_tokenizer(tokenizer_path: Optional[str], *, trust_remote_code: bool):
    if not _dasync_dump_enabled():
        return None
    if not _dasync_env_flag("SGLANG_DASYNC_DUMP_DECODE", True):
        return None
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=bool(trust_remote_code),
        )
        _dasync_dump_event(
            {
                "role": "tokenizer",
                "event": "tokenizer_ready",
                "tokenizer_path": tokenizer_path,
                "vocab_size": len(tokenizer),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            }
        )
        return tokenizer
    except Exception:
        _dasync_dump_event(
            {
                "role": "tokenizer",
                "event": "tokenizer_load_failed",
                "tokenizer_path": tokenizer_path,
                "traceback": traceback.format_exc(),
            }
        )
        return None


@dataclass
class DAsyncDraftInput(SpecInput):
    """Scheduler-side state for DASYNC.

    ``verified_id`` is the current token that has already been emitted to the
    client but has not yet been materialized into the target KV cache.  Each
    decode iteration verifies this root token plus zero or more raw-token draft
    proposals, then produces a new ``verified_id`` bonus token for the next
    round.
    """

    verified_id: torch.Tensor
    epochs: List[int]
    target_hidden: Optional[torch.Tensor] = None
    ctx_lens: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DASYNC_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        idx = [int(x) for x in new_indices.detach().cpu().tolist()]
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        self.verified_id = self.verified_id[new_indices]
        self.epochs = [self.epochs[i] for i in idx]

        if old_ctx_lens is None:
            self.ctx_lens = None
            self.target_hidden = None if old_target_hidden is None else old_target_hidden[:0]
            return

        self.ctx_lens = old_ctx_lens[new_indices]
        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

        offsets = torch.zeros(
            (int(old_ctx_lens.shape[0]) + 1,),
            dtype=torch.int64,
            device=old_ctx_lens.device,
        )
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))
        seg_start = offsets[:-1][new_indices]
        seg_lens = self.ctx_lens.to(torch.int64)
        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
        if max_len <= 0:
            self.target_hidden = old_target_hidden[:0]
            return

        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[None, :]
        pos2d = seg_start[:, None] + r
        mask = r < seg_lens[:, None]
        flat_pos = pos2d[mask]
        self.target_hidden = (
            old_target_hidden.index_select(0, flat_pos)
            if flat_pos.numel() > 0
            else old_target_hidden[:0]
        )

    def merge_batch(self, spec_info: "DAsyncDraftInput"):
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.epochs.extend(spec_info.epochs)
        if self.ctx_lens is None:
            self.ctx_lens = spec_info.ctx_lens
        elif spec_info.ctx_lens is not None:
            self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)

        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0:
            self.target_hidden = torch.cat([self.target_hidden, spec_info.target_hidden], dim=0)


@dataclass
class DAsyncVerifyInput(SpecInput):
    """Linear target verifier for raw-token async block proposals.

    This is intentionally close to DFlash's chain verifier.  The current
    v3 runtime is token-only; target-hidden conditioning is configured as an
    explicit next-generation mode so it cannot be silently mistaken for this
    checkpoint/protocol.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    valid_draft_lens: torch.Tensor
    topk: int = 1
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DASYNC_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(batch.tree_cache, len(batch.input_ids))
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if not build_custom_mask:
            self.custom_mask = None
            return

        if self.draft_token_num <= 0:
            raise ValueError(
                f"DASYNC draft_token_num must be positive, got {self.draft_token_num}."
            )
        mask_chunks: List[torch.Tensor] = []
        q_len = int(self.draft_token_num)
        q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
        for prefix_len in batch.seq_lens_cpu.tolist():
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + q_len
            k_idx = torch.arange(kv_len, device=batch.device, dtype=torch.int32).unsqueeze(0)
            allow = k_idx <= (prefix_len_i + q_idx)
            mask_chunks.append(allow.flatten())
        self.custom_mask = (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

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
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
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
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]:
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), None, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DASYNC verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )
            if not sampling_info.is_all_greedy:
                raise RuntimeError(
                    "DASYNC currently supports greedy target sampling only. "
                    "Run with temperature=0/top_k=1-style sampling for the verifier."
                )
            if sampling_info.has_custom_logit_processor:
                apply_custom_logit_processor(
                    logits_output.next_token_logits,
                    sampling_info,
                    num_tokens_in_batch=self.draft_token_num,
                )
            if (
                sampling_info.penalizer_orchestrator.is_required
                or sampling_info.logit_bias is not None
            ):
                linear_penalty = torch.zeros(
                    (bs, logits_output.next_token_logits.shape[1]),
                    dtype=torch.float32,
                    device=device,
                )
                sampling_info.apply_logits_bias(linear_penalty)
                logits_output.next_token_logits.add_(
                    torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
                )

        expected_logits = bs * int(self.draft_token_num)
        next_token_logits = logits_output.next_token_logits
        if next_token_logits.shape[0] != expected_logits:
            if next_token_logits.shape[0] < expected_logits:
                raise RuntimeError(
                    "DASYNC verifier produced fewer logits than the active verify width: "
                    f"logits={next_token_logits.shape[0]}, expected={expected_logits}, "
                    f"bs={bs}, draft_token_num={self.draft_token_num}."
                )
            # CUDA Graph replay may return the fixed capture width even when an
            # older variable-width path built a smaller DAsyncVerifyInput.  Keep
            # the leading active tokens and ignore graph padding.  The normal
            # fast path below now keeps DASYNC fixed-width, so this is defensive.
            next_token_logits = next_token_logits[:expected_logits]

        candidates = self.draft_token.view(bs, self.draft_token_num)
        target_predict = torch.argmax(next_token_logits, dim=-1).view(
            bs, self.draft_token_num
        )
        if _dasync_dump_enabled():
            try:
                self._debug_candidates = candidates.detach().cpu().tolist()
                self._debug_target_predict = target_predict.detach().cpu().tolist()
            except Exception:
                self._debug_candidates = None
                self._debug_target_predict = None
        valid_lens = self.valid_draft_lens.to(device=device, dtype=torch.int64)
        valid_lens.clamp_(min=0, max=max(0, self.draft_token_num - 1))

        if self.draft_token_num == 1:
            accept_len = torch.zeros((bs,), dtype=torch.int32, device=device)
        else:
            pos = torch.arange(
                self.draft_token_num - 1, dtype=torch.int64, device=device
            ).unsqueeze(0)
            valid_mask = pos < valid_lens.unsqueeze(1)
            matches = (candidates[:, 1:] == target_predict[:, :-1]) & valid_mask
            accept_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
        bonus = target_predict[torch.arange(bs, device=device), accept_len.to(torch.long)]
        if _dasync_dump_enabled():
            try:
                self._debug_valid_lens = valid_lens.detach().cpu().tolist()
                self._debug_matches = (
                    matches.detach().cpu().tolist() if self.draft_token_num > 1 else []
                )
                self._debug_accept_len_raw = accept_len.detach().cpu().tolist()
                self._debug_bonus = bonus.detach().cpu().tolist()
            except Exception:
                pass

        packed = torch.cat(
            [candidates[:, 1:], accept_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
        ).cpu()
        valid_lens_cpu = valid_lens.detach().cpu().tolist()

        max_acc = self.draft_token_num - 1
        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []
        debug_accept_rows: List[Dict[str, Any]] = []

        for i, req in enumerate(batch.reqs):
            acc_len = int(packed[i, max_acc].item())
            acc_len = min(acc_len, int(valid_lens_cpu[i]))
            proposed = packed[i, :acc_len].tolist() + [int(packed[i, max_acc + 1].item())]

            appended = 0
            for token_id in proposed:
                token_id = int(token_id)
                req.output_ids.append(token_id)
                appended += 1
                req.check_finished()
                if req.finished():
                    break
                if req.grammar is not None:
                    req.grammar.accept_token(token_id)

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DASYNC verify cannot determine current token: both "
                    "output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            accept_length_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]
            if _dasync_dump_enabled():
                debug_accept_rows.append(
                    {
                        "rid": str(getattr(req, "rid", i)),
                        "row": int(i),
                        "valid_future_len": int(valid_lens_cpu[i]),
                        "accepted_future_len": int(acc_len),
                        "commit_len": int(appended),
                        "proposed_commit_ids": [int(x) for x in proposed],
                        "bonus_id": int(packed[i, max_acc + 1].item()),
                        "new_verified_id": int(new_verified_token),
                        "finished": bool(req.finished()),
                    }
                )

        if _dasync_dump_enabled():
            self._debug_accept_rows = debug_accept_rows

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(new_verified_list, dtype=torch.int64, device=device)

        if page_size == 1:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            keep_mask = (
                torch.arange(self.draft_token_num, device=device)[None, :]
                < commit_lens[:, None]
            )
            batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
            batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            row_offsets = torch.arange(self.draft_token_num, device=device)[None, :]
            extended_lens = batch.seq_lens + self.draft_token_num
            new_lens = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
            aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size
            keep_lens = torch.minimum(aligned_new_lens, extended_lens)
            keep_slots = (keep_lens - batch.seq_lens).to(torch.int64)
            keep_slots.clamp_(min=0, max=int(self.draft_token_num))
            free_mask = row_offsets >= keep_slots[:, None]
            batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])

            keep_mask = row_offsets < commit_lens[:, None]
            batch.out_cache_loc = out_cache_loc[keep_mask]

        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

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

        next_target_hidden: Optional[torch.Tensor] = None
        if self.capture_hidden_mode == CaptureHiddenMode.FULL:
            hidden = logits_output.hidden_states
            if hidden is None:
                raise RuntimeError(
                    "DASYNC target-hidden condition requires hidden states from "
                    "target verify, but logits_output.hidden_states is None."
                )
            if int(hidden.shape[0]) < expected_logits:
                raise RuntimeError(
                    "DASYNC target-hidden condition got fewer hidden rows than "
                    f"verify tokens: hidden={int(hidden.shape[0])}, expected={expected_logits}."
                )
            hidden = hidden[:expected_logits].view(bs, self.draft_token_num, -1)
            segments: List[torch.Tensor] = []
            for i, ln in enumerate(commit_lens_cpu):
                if int(ln) > 0:
                    segments.append(hidden[i, : int(ln), :])
            next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]

        if logits_output.hidden_states is not None:
            logits_output.hidden_states = None

        return new_verified_id, commit_lens, next_target_hidden, accept_length_per_req_cpu


@dataclass
class _DAsyncProposalMatch:
    msg: Optional[Dict[str, Any]]
    tokens: List[int]
    probs: List[float]
    matched: bool
    match_kind: str
    prefix_advanced: int = 0
    proposal_epoch: Optional[int] = None
    proposal_prefix_len: Optional[int] = None
    candidates_seen: int = 0
    discarded: Optional[List[Dict[str, Any]]] = None


class DAsyncWorker:
    """DASYNC speculative verifier worker.

    The target model stays inside SGLang on the normal scheduler device. Draft
    proposals arrive either from the standalone Unix-socket draft engine or from
    the local rank-0 draft thread plus TP broadcast. The verifier consumes only
    token IDs/proposal epochs; raw-token DASYNC draft KV is private to the draft
    engine and is never transferred into target KV.
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
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        self.device = target_worker.device

        self.max_future_tokens = int(server_args.speculative_dasync_draft_max_tokens)
        self.verify_width_cap = self.max_future_tokens + 1
        self.min_verify_tokens = min(
            self.max_future_tokens,
            max(0, int(getattr(server_args, "speculative_dasync_draft_min_tokens", 0) or 0)),
        )
        self.poll_timeout_s = float(server_args.speculative_dasync_draft_poll_timeout_ms) / 1000.0
        adaptive_wait_ms = float(
            os.environ.get(
                "SGLANG_DASYNC_ADAPTIVE_WAIT_MAX_MS",
                getattr(server_args, "speculative_dasync_adaptive_wait_max_ms", 80.0),
            )
        )
        self.adaptive_wait_max_s = max(self.poll_timeout_s, adaptive_wait_ms / 1000.0)
        self.pad_token_id = 0
        self.condition_mode = str(
            getattr(server_args, "speculative_dasync_condition_mode", "none") or "none"
        ).strip().lower().replace("-", "_")
        self.condition_layer_ids = list(
            getattr(server_args, "speculative_dasync_condition_layer_ids", None) or []
        )
        self.proposal_mode = _dasync_normalize_proposal_mode(
            getattr(
                server_args,
                "speculative_dasync_draft_proposal_mode",
                "masked_block",
            )
        )
        self.engine_mode = str(
            getattr(server_args, "speculative_dasync_engine_mode", "external")
            or "external"
        ).strip().lower().replace("-", "_")
        self._tp_size = int(
            getattr(self.model_runner, "tp_size", getattr(server_args, "tp_size", 1))
            or 1
        )
        self._local_draft_leader = (
            self.engine_mode != "local" or int(self.tp_rank) == 0
        )
        if self.engine_mode == "local":
            self._draft_topology = (
                "local_rank0_broadcast" if self._tp_size > 1 else "local_single_rank"
            )
        else:
            self._draft_topology = "external_socket"
        self._latest_proposals: Dict[Tuple[str, int], Dict[str, Any]] = {}
        self._completed_proposals: Dict[Tuple[str, int], int] = {}
        self._warned_no_proposal = False
        self._draft_failed = False
        self._draft_ready = False
        self._draft_inflight: Dict[str, Tuple[int, float]] = {}
        self._consecutive_fallbacks = 0
        self._logged_first_verify = False
        self._control_step = 0
        self._proposal_verify_ct = 0
        self._fallback_decode_ct = 0
        self._zero_future_skip_ct = 0
        self._partial_proposal_skip_ct = 0
        self._proposal_accept_sum = 0
        self._proposal_ready_sum = 0
        self._proposal_stale_exhausted_ct = 0

        qsize = int(server_args.speculative_dasync_draft_queue_size)
        if self.engine_mode == "local":
            draft_model_path = getattr(server_args, "speculative_draft_model_path", None)
            if not draft_model_path:
                raise ValueError(
                    "DASYNC local engine requires --speculative-draft-model-path."
                )
            if self.condition_mode != "none":
                raise RuntimeError(
                    "DASYNC local engine is currently token-only. "
                    "target_hidden conditioning needs a condition-trained drafter "
                    "and TP-local tensor plumbing."
                )
            if self._local_draft_leader:
                self._ipc = DAsyncLocalDraftEngine(
                    {
                        "model_path": str(draft_model_path),
                        "target_model_path": str(server_args.model_path),
                        "tokenizer_path": str(
                            server_args.tokenizer_path or server_args.model_path
                        ),
                        "device": str(self.device),
                        "dtype": (
                            server_args.speculative_dasync_draft_dtype
                            or getattr(server_args, "dtype", None)
                            or "auto"
                        ),
                        "trust_remote_code": bool(server_args.trust_remote_code),
                        "attn_implementation": getattr(
                            server_args,
                            "speculative_dasync_draft_attn_implementation",
                            None,
                        ),
                        "max_tokens": int(self.max_future_tokens),
                        "min_tokens": int(
                            server_args.speculative_dasync_draft_min_tokens
                        ),
                        "confidence_threshold": float(
                            server_args.speculative_dasync_draft_confidence_threshold
                        ),
                        "temperature": float(
                            server_args.speculative_dasync_draft_temperature
                        ),
                        "top_p": float(server_args.speculative_dasync_draft_top_p),
                        "publish_stride": int(
                            server_args.speculative_dasync_draft_publish_stride
                        ),
                        "proposal_mode": self.proposal_mode,
                        "batch_size": int(
                            getattr(
                                server_args,
                                "speculative_dasync_draft_batch_size",
                                1,
                            )
                        ),
                        "ar_anchor_enabled": bool(
                            getattr(
                                server_args,
                                "speculative_dasync_draft_ar_anchor",
                                False,
                            )
                        ),
                        "ar_anchor_threshold": float(
                            getattr(
                                server_args,
                                "speculative_dasync_draft_ar_anchor_threshold",
                                0.0,
                            )
                        ),
                        "ar_anchor_max_steps": int(
                            getattr(
                                server_args,
                                "speculative_dasync_draft_ar_anchor_max_steps",
                                0,
                            )
                        ),
                        "condition_mode": self.condition_mode,
                        "condition_layer_ids": self.condition_layer_ids,
                    },
                    queue_size=max(4, qsize),
                )
            else:
                self._ipc = None
        else:
            self._ipc = DAsyncUnixSocketClient(
                str(server_args.speculative_dasync_ipc_path),
                connect_timeout_s=float(server_args.speculative_dasync_ipc_connect_timeout_ms) / 1000.0,
                queue_size=max(4, qsize),
            )

        self._dump_tokenizer = _dasync_load_dump_tokenizer(
            server_args.tokenizer_path or server_args.model_path,
            trust_remote_code=bool(server_args.trust_remote_code),
        )
        self._wait_for_draft_startup()

        if self.tp_rank == 0:
            logger.info(
                "Initialized DASYNC verifier. engine_mode=%s, topology=%s, "
                "tp_size=%s, local_draft_leader=%s, target_gpu=%s, ipc_path=%s, "
                "max_future_tokens=%s, min_verify_tokens=%s, confidence_threshold=%s, "
                "proposal_mode=%s",
                self.engine_mode,
                self._draft_topology,
                self._tp_size,
                self._local_draft_leader,
                gpu_id,
                server_args.speculative_dasync_ipc_path,
                self.max_future_tokens,
                self.min_verify_tokens,
                server_args.speculative_dasync_draft_confidence_threshold,
                self.proposal_mode,
            )
        _dasync_dump_event(
            {
                "role": "verifier",
                "event": "worker_init",
                "engine_mode": self.engine_mode,
                "draft_topology": self._draft_topology,
                "tp_size": int(self._tp_size),
                "tp_rank": int(self.tp_rank),
                "local_draft_leader": bool(self._local_draft_leader),
                "target_gpu": gpu_id,
                "target_device": str(self.device),
                "ipc_path": str(server_args.speculative_dasync_ipc_path),
                "tokenizer_path": server_args.tokenizer_path or server_args.model_path,
                "max_future_tokens": int(self.max_future_tokens),
                "min_verify_tokens": int(self.min_verify_tokens),
                "verify_width_cap": int(self.verify_width_cap),
                "poll_timeout_ms": float(self.poll_timeout_s * 1000.0),
                "adaptive_wait_max_ms": float(self.adaptive_wait_max_s * 1000.0),
                "confidence_threshold": float(server_args.speculative_dasync_draft_confidence_threshold),
                "proposal_mode": self.proposal_mode,
                "condition_mode": self.condition_mode,
                "condition_layer_ids": self.condition_layer_ids,
                "protocol": (
                    "dasync-v4-local-thread"
                    if self.engine_mode == "local"
                    else "dasync-v3-jsonl-unix"
                ),
            }
        )

    def _validate_ready_condition(self, msg: Dict[str, Any]) -> None:
        draft_mode = str(msg.get("condition_mode") or "none").strip().lower().replace("-", "_")
        draft_layers = [int(x) for x in (msg.get("condition_layer_ids") or [])]
        if draft_mode != self.condition_mode or draft_layers != self.condition_layer_ids:
            self._draft_failed = True
            raise RuntimeError(
                "DASYNC verifier/drafter condition mismatch: "
                f"verifier=({self.condition_mode}, {self.condition_layer_ids}), "
                f"drafter=({draft_mode}, {draft_layers}). Refusing to run a "
                "token-only drafter under a conditioned verifier configuration."
            )
        draft_proposal_mode = _dasync_normalize_proposal_mode(
            msg.get("proposal_mode") or "masked_block"
        )
        if draft_proposal_mode != self.proposal_mode:
            self._draft_failed = True
            raise RuntimeError(
                "DASYNC verifier/drafter proposal mode mismatch: "
                f"verifier={self.proposal_mode}, drafter={draft_proposal_mode}. "
                "Start both sides with the same DASYNC proposal mode."
            )

    def _wait_for_draft_startup(self) -> None:
        """Wait until the active DASYNC drafter reports ready."""

        timeout_ms = os.environ.get("SGLANG_DASYNC_DRAFT_STARTUP_TIMEOUT_MS")
        if timeout_ms is not None:
            timeout_s = float(timeout_ms) / 1000.0
        else:
            timeout_s = float(os.environ.get("SGLANG_DASYNC_STARTUP_TIMEOUT_S", "120"))
        if timeout_s <= 0:
            return

        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            messages = self._recv_draft_messages(
                timeout=max(0.0, deadline - time.perf_counter()),
                wait=True,
                context="startup",
            )
            if not messages:
                break
            for msg in messages:
                typ = self._handle_draft_message(
                    msg, from_wait=True, startup_wait=True
                )
                if typ == "ready":
                    return

        if self.tp_rank == 0:
            logger.warning(
                "DASYNC draft engine did not report ready within %.1f s; "
                "continuing asynchronously. engine_mode=%s, local_leader=%s",
                timeout_s,
                self.engine_mode,
                self._local_draft_leader,
            )

    def __getattr__(self, name):
        return getattr(self.target_worker, name)

    def _decode_ids(self, ids):
        return _dasync_decode_ids(self._dump_tokenizer, ids)

    def _dump_event(self, event: str, **payload: Any) -> None:
        payload.setdefault("role", "verifier")
        payload.setdefault("event", event)
        _dasync_dump_event(payload)

    def _sync_local_draft_messages(self, messages: Optional[List[Dict[str, Any]]]):
        if not (self.engine_mode == "local" and self._tp_size > 1):
            return messages or []
        return get_tp_group().broadcast_object(messages, src=0) or []

    def _recv_draft_messages(
        self,
        *,
        timeout: float = 0.0,
        wait: bool = False,
        context: str = "drain",
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.engine_mode == "local" and not self._local_draft_leader:
            return self._sync_local_draft_messages(None)

        if self._ipc is None:
            return self._sync_local_draft_messages(messages)

        try:
            if wait:
                messages.append(self._ipc.recv(timeout=max(0.0, float(timeout))))
            while True:
                try:
                    messages.append(self._ipc.recv_nowait())
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        except DAsyncIPCError as exc:
            self._draft_failed = True
            logger.warning(
                "DASYNC draft IPC receive failed during %s: %s", context, exc
            )
            messages.append(
                {
                    "type": "error",
                    "source": "dasync_verifier_recv",
                    "traceback": str(exc),
                }
            )

        return self._sync_local_draft_messages(messages)

    def _handle_draft_message(
        self,
        msg: Dict[str, Any],
        *,
        from_wait: bool = False,
        startup_wait: bool = False,
    ) -> Optional[str]:
        typ = msg.get("type")
        if typ == "ready":
            self._validate_ready_condition(msg)
            self._draft_ready = True
            if self.tp_rank == 0:
                logger.info(
                    "DASYNC draft engine ready. topology=%s, device=%s, backend=%s, model=%s, base=%s, protocol=%s, proposal_mode=%s",
                    self._draft_topology,
                    msg.get("device"),
                    msg.get("draft_backend"),
                    msg.get("model_path"),
                    msg.get("base_model_path"),
                    msg.get("protocol"),
                    msg.get("proposal_mode") or "masked_block",
                )
            self._dump_event(
                "draft_ready_seen",
                draft_topology=self._draft_topology,
                tp_size=int(self._tp_size),
                tp_rank=int(self.tp_rank),
                local_draft_leader=bool(self._local_draft_leader),
                device=msg.get("device"),
                draft_backend=msg.get("draft_backend"),
                model_path=msg.get("model_path"),
                base_model_path=msg.get("base_model_path"),
                tokenizer_path=msg.get("tokenizer_path"),
                eos_token_id=msg.get("eos_token_id"),
                condition_mode=msg.get("condition_mode"),
                condition_layer_ids=msg.get("condition_layer_ids"),
                proposal_mode=msg.get("proposal_mode") or "masked_block",
                protocol=msg.get("protocol"),
                from_wait=bool(from_wait),
                startup_wait=bool(startup_wait),
            )
        elif typ == "error":
            self._draft_failed = True
            logger.error("DASYNC draft engine failed:\n%s", msg.get("traceback"))
            self._dump_event(
                "draft_error",
                traceback=msg.get("traceback"),
                from_wait=bool(from_wait),
                startup_wait=bool(startup_wait),
            )
            if startup_wait:
                raise RuntimeError(
                    "DASYNC draft engine failed during startup; see traceback above."
                )
        elif typ in {"draft_started", "draft_done"}:
            self._handle_draft_status(msg)
        elif typ == "proposal":
            self._remember_proposal(msg, from_wait=from_wait)
        return typ

    def _put_cmd(self, msg: Dict[str, Any]) -> None:
        if self._draft_failed:
            return
        if self.engine_mode == "local" and not self._local_draft_leader:
            return
        if self._ipc is None:
            return
        try:
            self._ipc.send(msg)
        except DAsyncIPCError as exc:
            self._draft_failed = True
            logger.warning("DASYNC draft IPC send failed for %s: %s", msg.get("op"), exc)

    def _handle_draft_status(self, msg: Dict[str, Any]) -> None:
        typ = msg.get("type")
        rid = str(msg.get("rid", ""))
        if not rid:
            return
        try:
            epoch = int(msg.get("epoch", -1))
        except Exception:
            epoch = -1
        if typ == "draft_started":
            self._draft_inflight[rid] = (epoch, time.perf_counter())
            self._dump_event(
                "scheduler_draft_started",
                rid=rid,
                epoch=epoch,
                prefix_len=msg.get("prefix_len"),
            )
        elif typ == "draft_done":
            old = self._draft_inflight.get(rid)
            if old is not None and int(old[0]) <= epoch:
                self._draft_inflight.pop(rid, None)
            try:
                proposal_len = int(msg.get("proposal_len", 0) or 0)
            except Exception:
                proposal_len = 0
            key = (rid, int(epoch))
            self._completed_proposals[key] = max(0, proposal_len)
            existing = self._latest_proposals.get(key)
            if existing is not None and len(existing.get("tokens", [])) >= proposal_len:
                existing["_dasync_final"] = True
            self._dump_event(
                "scheduler_draft_done",
                rid=rid,
                epoch=epoch,
                prefix_len=msg.get("prefix_len"),
                proposal_len=proposal_len,
            )

    def _batch_has_inflight_draft(self, batch: ScheduleBatch, state: DAsyncDraftInput) -> bool:
        now = time.perf_counter()
        stale_after_s = max(0.5, self.adaptive_wait_max_s * 10.0)
        for req, epoch in zip(batch.reqs, state.epochs, strict=True):
            item = self._draft_inflight.get(str(req.rid))
            if item is None:
                continue
            inflight_epoch, started = item
            if now - float(started) > stale_after_s:
                self._draft_inflight.pop(str(req.rid), None)
                continue
            if int(inflight_epoch) >= int(epoch):
                return True
        return False

    def _drain_results(self) -> None:
        for msg in self._recv_draft_messages(context="drain"):
            self._handle_draft_message(msg)

    def _remember_proposal(self, msg: Dict[str, Any], *, from_wait: bool = False) -> None:
        key = (str(msg["rid"]), int(msg["epoch"]))
        prev = self._latest_proposals.get(key)
        prev_len = len(prev.get("tokens", [])) if prev is not None else -1
        cur_len = len(msg.get("tokens", []))
        completed_len = self._completed_proposals.get(key)
        if completed_len is not None and cur_len >= int(completed_len):
            msg["_dasync_final"] = True
        if prev is None or cur_len >= prev_len:
            self._latest_proposals[key] = msg
        elif completed_len is not None and prev is not None and prev_len >= int(completed_len):
            prev["_dasync_final"] = True

        toks = [int(x) for x in msg.get("tokens", [])]
        self._dump_event(
            "scheduler_recv_proposal",
            rid=str(msg.get("rid")),
            epoch=int(msg.get("epoch", -1)),
            prefix_len=int(msg.get("prefix_len", -1)),
            prefix_tail_ids=_dasync_clip_list(msg.get("prefix_tail_ids", [])),
            prefix_tail_text=self._decode_ids(msg.get("prefix_tail_ids", [])),
            proposal_len=len(toks),
            token_ids=_dasync_clip_list(toks),
            token_text=self._decode_ids(toks),
            probs=_dasync_clip_list(msg.get("probs", [])),
            from_wait=bool(from_wait),
            replaced_previous=bool(prev is not None and cur_len >= prev_len),
        )

        # Keep the map bounded.  We preserve newer proposals by deleting from the
        # insertion order front, which is sufficient because every request sends
        # monotonic epochs and new proposals are inserted at the end.
        if len(self._latest_proposals) > 4096:
            for old_key in list(self._latest_proposals.keys())[:1024]:
                self._latest_proposals.pop(old_key, None)
                self._completed_proposals.pop(old_key, None)

    def _proposal_tokens_probs(self, msg: Dict[str, Any]) -> Tuple[List[int], List[float]]:
        toks = [int(x) for x in msg.get("tokens", [])[: self.max_future_tokens]]
        probs = [float(x) for x in msg.get("probs", [])[: len(toks)]]
        if len(probs) < len(toks):
            probs.extend([0.0] * (len(toks) - len(probs)))
        return toks, probs

    @staticmethod
    def _proposal_base_tail_matches(msg: Dict[str, Any], current_prefix: List[int]) -> bool:
        try:
            base_len = int(msg.get("prefix_len", -1))
        except Exception:
            return False
        if base_len < 0 or base_len > len(current_prefix):
            return False
        tail = msg.get("prefix_tail_ids") or []
        if not tail:
            # Proposals produced before this hotfix do not carry a base-prefix
            # fingerprint.  They can still be used safely when their proposed
            # path matches the current suffix, so keep backward compatibility.
            return True
        tail = [int(x) for x in tail]
        if len(tail) > base_len:
            tail = tail[-base_len:]
        start = base_len - len(tail)
        return current_prefix[start:base_len] == tail

    def _select_proposal_for_req(
        self,
        req,
        *,
        current_epoch: int,
        current_prefix_ids: List[int],
        prune_unusable: bool = True,
    ) -> _DAsyncProposalMatch:
        """Return the best proposal suffix whose base lies on current prefix path.

        An async proposal may arrive one or more verifier iterations late.  If
        the verifier has advanced from the proposal base by tokens that exactly
        match the proposal's own prefix, the still-unverified suffix remains a
        valid speculative candidate.  This converts the matching criterion from
        strict epoch equality to path membership.
        """

        rid = str(req.rid)
        current_len = len(current_prefix_ids)
        candidates_seen = 0
        discarded: List[Dict[str, Any]] = []
        delete_keys: List[Tuple[str, int]] = []
        best: Optional[Tuple[Tuple[int, int, int], _DAsyncProposalMatch]] = None

        for key, msg in list(self._latest_proposals.items()):
            if key[0] != rid:
                continue
            candidates_seen += 1
            try:
                proposal_epoch = int(msg.get("epoch", key[1]))
                base_len = int(msg.get("prefix_len", -1))
            except Exception:
                delete_keys.append(key)
                discarded.append({"epoch": key[1], "reason": "bad_metadata"})
                continue

            toks, probs = self._proposal_tokens_probs(msg)
            completed_len = self._completed_proposals.get(key)
            proposal_final = bool(msg.get("_dasync_final")) or (
                completed_len is not None and len(toks) >= int(completed_len)
            )
            if proposal_final:
                msg["_dasync_final"] = True
            reason = None
            if base_len < 0:
                reason = "bad_prefix_len"
            elif base_len > current_len:
                reason = "base_after_current"
            elif not self._proposal_base_tail_matches(msg, current_prefix_ids):
                reason = "base_tail_mismatch"
            else:
                advanced = current_len - base_len
                if advanced >= len(toks):
                    reason = "exhausted"
                elif current_prefix_ids[base_len:current_len] != toks[:advanced]:
                    reason = "path_mismatch"
                else:
                    suffix_toks = toks[advanced:]
                    suffix_probs = probs[advanced:]
                    suffix_ready = (
                        len(suffix_toks) >= int(self.min_verify_tokens)
                        or proposal_final
                        or int(self.min_verify_tokens) <= 0
                    )
                    if not suffix_ready:
                        discarded.append(
                            {
                                "epoch": int(proposal_epoch),
                                "prefix_len": int(base_len),
                                "proposal_len": len(toks),
                                "suffix_len": len(suffix_toks),
                                "reason": "partial_not_ready",
                                "proposal_final": bool(proposal_final),
                                "min_verify_tokens": int(self.min_verify_tokens),
                            }
                        )
                        continue
                    if proposal_epoch == int(current_epoch) and advanced == 0:
                        match_kind = "exact_epoch"
                    elif advanced == 0:
                        match_kind = "exact_prefix_stale_epoch"
                    else:
                        match_kind = "stale_suffix"
                    # Prefer the proposal with the most recent base prefix.  It
                    # is conditioned on more verified text; suffix length is the
                    # secondary objective.
                    score = (base_len, len(suffix_toks), proposal_epoch)
                    candidate = _DAsyncProposalMatch(
                        msg=msg,
                        tokens=suffix_toks,
                        probs=suffix_probs,
                        matched=True,
                        match_kind=match_kind,
                        prefix_advanced=advanced,
                        proposal_epoch=proposal_epoch,
                        proposal_prefix_len=base_len,
                        candidates_seen=candidates_seen,
                    )
                    if best is None or score > best[0]:
                        best = (score, candidate)
                    continue

            discarded.append(
                {
                    "epoch": int(proposal_epoch),
                    "prefix_len": int(base_len),
                    "proposal_len": len(toks),
                    "reason": reason,
                }
            )
            delete_keys.append(key)

        if prune_unusable:
            for key in delete_keys:
                self._latest_proposals.pop(key, None)
                self._completed_proposals.pop(key, None)

        if best is not None:
            match = best[1]
            match.candidates_seen = candidates_seen
            match.discarded = discarded[:8]
            return match

        return _DAsyncProposalMatch(
            msg=None,
            tokens=[],
            probs=[],
            matched=False,
            match_kind="none",
            proposal_epoch=None,
            candidates_seen=candidates_seen,
            discarded=discarded[:8],
        )

    def _wait_for_results(self) -> None:
        if self.poll_timeout_s <= 0 or self._draft_failed:
            self._drain_results()
            return
        deadline = time.perf_counter() + self.poll_timeout_s
        self._drain_results()
        while time.perf_counter() < deadline:
            messages = self._recv_draft_messages(
                timeout=max(0.0, deadline - time.perf_counter()),
                wait=True,
                context="wait",
            )
            if not messages:
                break
            for msg in messages:
                typ = self._handle_draft_message(msg, from_wait=True)
                if typ == "proposal" or self._draft_failed:
                    return

    @staticmethod
    def _prefix_ids(req, *, append_token: Optional[int] = None) -> List[int]:
        ids = [int(x) for x in req.origin_input_ids]
        ids.extend(int(x) for x in req.output_ids)
        if append_token is not None:
            token = int(append_token)
            if len(ids) == 0 or ids[-1] != token:
                ids.append(token)
        return ids

    def _send_upsert(self, req, *, epoch: int, prefix_ids: List[int]) -> None:
        if req.finished():
            self._put_cmd({"op": "remove", "rid": str(req.rid)})
            return
        prefix_tail = [int(x) for x in prefix_ids[-_dasync_dump_max_items():]]
        evt = {
            "rid": str(req.rid),
            "epoch": int(epoch),
            "prefix_len": len(prefix_ids),
            "prefix_tail_ids": prefix_tail,
            "prefix_tail_text": self._decode_ids(prefix_tail),
        }
        if _dasync_env_flag("SGLANG_DASYNC_DUMP_FULL_PREFIX", False):
            evt["prefix_ids"] = [int(x) for x in prefix_ids]
        self._dump_event("scheduler_send_upsert", **evt)
        self._put_cmd(
            {
                "op": "upsert",
                "rid": str(req.rid),
                "epoch": int(epoch),
                "prefix_ids": [int(x) for x in prefix_ids],
            }
        )

    def clear_cache_pool(self):
        self._latest_proposals.clear()
        self._put_cmd({"op": "clear"})

    def shutdown(self):
        try:
            self._put_cmd({"op": "shutdown"})
        finally:
            if hasattr(self, "_ipc") and self._ipc is not None:
                self._ipc.close()

    def _condition_enabled(self) -> bool:
        return self.condition_mode == "target_hidden"

    def _bootstrap_after_prefill(
        self,
        batch: ScheduleBatch,
        next_token_ids: torch.Tensor,
        *,
        target_hidden: Optional[torch.Tensor] = None,
        ctx_lens: Optional[torch.Tensor] = None,
    ):
        next_cpu = [int(x) for x in next_token_ids.detach().cpu().tolist()]
        epochs = [0 for _ in batch.reqs]
        state = DAsyncDraftInput(
            verified_id=next_token_ids.to(torch.int64),
            epochs=epochs,
            target_hidden=target_hidden,
            ctx_lens=ctx_lens,
        )
        for req, token_id, epoch in zip(batch.reqs, next_cpu, epochs, strict=True):
            self._send_upsert(
                req,
                epoch=epoch,
                prefix_ids=self._prefix_ids(req, append_token=int(token_id)),
            )
        batch.spec_info = state

    def _collect_proposal_matches(
        self,
        batch: ScheduleBatch,
        state: DAsyncDraftInput,
        *,
        prune_unusable: bool,
    ) -> Tuple[List[_DAsyncProposalMatch], List[Dict[str, Any]], int]:
        """Match the latest async proposal tape against the current verified path."""

        matches: List[_DAsyncProposalMatch] = []
        proposal_debug_rows: List[Dict[str, Any]] = []
        max_future = 0
        for req, epoch in zip(batch.reqs, state.epochs, strict=True):
            current_prefix_ids = self._prefix_ids(req)
            prefix_len = len(current_prefix_ids)
            match = self._select_proposal_for_req(
                req,
                current_epoch=int(epoch),
                current_prefix_ids=current_prefix_ids,
                prune_unusable=prune_unusable,
            )
            toks = match.tokens[: self.max_future_tokens]
            probs = match.probs[: len(toks)]
            if prune_unusable and match.discarded:
                self._proposal_stale_exhausted_ct += sum(
                    1 for item in match.discarded if item.get("reason") == "exhausted"
                )
            max_future = max(max_future, len(toks))
            matches.append(match)
            proposal_debug_rows.append(
                {
                    "rid": str(req.rid),
                    "epoch": int(epoch),
                    "prefix_len": int(prefix_len),
                    "proposal_found": match.msg is not None,
                    "proposal_matched_prefix": bool(match.matched),
                    "proposal_match_kind": match.match_kind,
                    "proposal_epoch": match.proposal_epoch,
                    "proposal_prefix_len": match.proposal_prefix_len,
                    "prefix_advanced_on_proposal": int(match.prefix_advanced),
                    "proposal_candidates_seen": int(match.candidates_seen),
                    "proposal_discarded": match.discarded or [],
                    "proposal_len": len(toks),
                    "proposal_final": bool(match.msg.get("_dasync_final")) if match.msg is not None else False,
                    "min_verify_tokens": int(self.min_verify_tokens),
                    "token_ids": _dasync_clip_list(toks),
                    "token_text": self._decode_ids(toks),
                    "probs": _dasync_clip_list(probs),
                }
            )
        return matches, proposal_debug_rows, max_future

    def _wait_until_any_usable_proposal(
        self,
        batch: ScheduleBatch,
        state: DAsyncDraftInput,
    ) -> Tuple[List[_DAsyncProposalMatch], List[Dict[str, Any]], int]:
        """Drain/wait for a proposal that still has an unverified suffix.

        DASYNC must not run fixed-width target verification with a zero-length
        future.  A zero-future speculative verify produces the same user-visible
        token as greedy decode, but pays the wide verifier bookkeeping cost and
        continuously makes async proposals stale.  This routine either returns a
        usable suffix, or lets the caller run an ordinary one-token target decode.
        """

        self._drain_results()
        matches, rows, max_future = self._collect_proposal_matches(
            batch, state, prune_unusable=False
        )
        bs = batch.batch_size()
        ready_rows = sum(1 for match in matches if len(match.tokens) > 0)
        if (
            ready_rows == bs
            or self.poll_timeout_s <= 0
            or self._draft_failed
        ):
            # Do a pruning pass only when we are about to consume or give up.
            return self._collect_proposal_matches(batch, state, prune_unusable=True)

        start_time = time.perf_counter()
        wait_s = self.poll_timeout_s
        bootstrap_wait = self._proposal_verify_ct == 0 and self._fallback_decode_ct == 0
        if self._consecutive_fallbacks > 0:
            wait_s = min(
                self.adaptive_wait_max_s,
                self.poll_timeout_s * float(2 ** min(self._consecutive_fallbacks, 4)),
            )
        inflight_wait = self._batch_has_inflight_draft(batch, state)
        if inflight_wait or bootstrap_wait:
            wait_s = max(wait_s, self.adaptive_wait_max_s)
        deadline = start_time + wait_s
        polls = 0
        while time.perf_counter() < deadline:
            timeout = max(0.0, deadline - time.perf_counter())
            messages = self._recv_draft_messages(
                timeout=timeout,
                wait=True,
                context="proposal_wait",
            )
            if not messages:
                break
            polls += len(messages)
            for msg in messages:
                self._handle_draft_message(msg, from_wait=True)
            if self._draft_failed:
                break

            matches, rows, max_future = self._collect_proposal_matches(
                batch, state, prune_unusable=False
            )
            ready_rows = sum(1 for match in matches if len(match.tokens) > 0)
            if ready_rows == bs:
                break

        wait_ms = (time.perf_counter() - start_time) * 1000.0
        matches, rows, max_future = self._collect_proposal_matches(
            batch, state, prune_unusable=True
        )
        ready_rows = sum(1 for match in matches if len(match.tokens) > 0)
        self._dump_event(
            "verifier_proposal_wait",
            waited_ms=float(wait_ms),
            polls=int(polls),
            ready_rows=int(ready_rows),
            batch_size=int(bs),
            max_future_ready=int(max_future),
            timeout_ms=float(wait_s * 1000.0),
            consecutive_fallbacks=int(self._consecutive_fallbacks),
            inflight_wait=bool(inflight_wait),
            bootstrap_wait=bool(bootstrap_wait),
        )
        return matches, rows, max_future

    def _build_verify_input(
        self, batch: ScheduleBatch, state: DAsyncDraftInput
    ) -> Optional[DAsyncVerifyInput]:
        bs = batch.batch_size()
        if state.verified_id.shape[0] != bs or len(state.epochs) != bs:
            raise RuntimeError(
                "DASYNC state/batch size mismatch: "
                f"verified_id={tuple(state.verified_id.shape)}, epochs={len(state.epochs)}, bs={bs}."
            )

        matches, proposal_debug_rows, max_future = self._wait_until_any_usable_proposal(
            batch, state
        )
        if max_future == 0:
            self._zero_future_skip_ct += 1
            if not self._warned_no_proposal and self.tp_rank == 0:
                logger.warning(
                    "DASYNC has no ready draft proposal; using ordinary one-token target decode instead of zero-future speculative verify."
                )
                self._warned_no_proposal = True
            if _dasync_dump_enabled():
                state_verified = state.verified_id.detach().cpu().tolist()
                for i, row in enumerate(proposal_debug_rows):
                    row["root_id"] = int(state_verified[i]) if i < len(state_verified) else None
                    row["root_text"] = self._decode_ids([row["root_id"]]) if row["root_id"] is not None else None
                    row["candidate_ids"] = [row["root_id"]] if row["root_id"] is not None else []
                    row["candidate_text"] = self._decode_ids(row["candidate_ids"])
                    row["valid_future_len_after_padding"] = 0
                self._dump_event(
                    "verifier_no_future_fallback",
                    batch_size=int(bs),
                    verify_width_cap=int(self.verify_width_cap),
                    max_future_ready=0,
                    rows=proposal_debug_rows,
                )
            return None

        proposals: List[List[int]] = []
        valid_lens: List[int] = []
        for match in matches:
            toks = match.tokens[: self.max_future_tokens]
            proposals.append(toks)
            valid_lens.append(len(toks))

        if any(length <= 0 for length in valid_lens):
            # A fixed-width target-verify forward is only useful for rows that
            # have real future tokens. Mixing zero-future rows into the same
            # verify batch pays the full verifier cost for [root]+padding rows
            # and makes proposal/accept metrics misleading. Until DASYNC has a
            # safe sub-batch split+merge path, fall back the whole batch.
            self._partial_proposal_skip_ct += 1
            if _dasync_dump_enabled():
                state_verified = state.verified_id.detach().cpu().tolist()
                for i, row in enumerate(proposal_debug_rows):
                    row["root_id"] = int(state_verified[i]) if i < len(state_verified) else None
                    row["root_text"] = self._decode_ids([row["root_id"]]) if row["root_id"] is not None else None
                    row["candidate_ids"] = [row["root_id"]] if row["root_id"] is not None else []
                    row["candidate_text"] = self._decode_ids(row["candidate_ids"])
                    row["valid_future_len_after_padding"] = int(valid_lens[i])
                self._dump_event(
                    "verifier_partial_proposal_fallback",
                    batch_size=int(bs),
                    verify_width_cap=int(self.verify_width_cap),
                    max_future_ready=int(max_future),
                    ready_rows=int(sum(1 for length in valid_lens if length > 0)),
                    rows=proposal_debug_rows,
                )
            return None

        self._proposal_ready_sum += sum(valid_lens)
        draft_token_num = int(self.verify_width_cap)

        draft_tokens = torch.full(
            (bs, draft_token_num),
            int(self.pad_token_id),
            dtype=torch.long,
            device=batch.device,
        )
        draft_tokens[:, 0].copy_(state.verified_id.to(device=batch.device, dtype=torch.long))
        for i, toks in enumerate(proposals):
            n = min(len(toks), draft_token_num - 1)
            if n > 0:
                draft_tokens[i, 1 : 1 + n] = torch.tensor(
                    toks[:n], dtype=torch.long, device=batch.device
                )
                valid_lens[i] = n
            else:
                valid_lens[i] = 0

        offsets = torch.arange(draft_token_num, dtype=torch.int64, device=batch.device)
        positions = (batch.seq_lens.to(torch.int64).unsqueeze(1) + offsets).reshape(-1)
        valid_lens_tensor = torch.tensor(valid_lens, dtype=torch.int32, device=batch.device)
        if _dasync_dump_enabled():
            candidate_rows = draft_tokens.detach().cpu().tolist()
            state_verified = state.verified_id.detach().cpu().tolist()
            for i, row in enumerate(proposal_debug_rows):
                row["root_id"] = int(state_verified[i]) if i < len(state_verified) else None
                row["root_text"] = self._decode_ids([row["root_id"]]) if row["root_id"] is not None else None
                row["candidate_ids"] = _dasync_clip_list(candidate_rows[i])
                row["candidate_text"] = self._decode_ids(candidate_rows[i])
                row["valid_future_len_after_padding"] = int(valid_lens[i])
            self._dump_event(
                "verifier_build_input",
                batch_size=int(bs),
                draft_token_num=int(draft_token_num),
                verify_width_cap=int(self.verify_width_cap),
                max_future_ready=int(max_future),
                rows=proposal_debug_rows,
            )
        return DAsyncVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=draft_token_num,
            valid_draft_lens=valid_lens_tensor,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return

        accepted_steps = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None
        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = batch.seq_lens // mamba_track_interval * mamba_track_interval
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (to_track_ith < commit_lens.to(to_track_ith.dtype))
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def _normalize_fallback_next_token_ids(
        self,
        batch: ScheduleBatch,
        next_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return exactly one fallback next-token id per live request.

        The DASYNC fallback asks the target worker to run an ordinary
        one-token decode. On Qwen3.5/GDN, a stale TARGET_VERIFY/fixed-width
        graph path can still return a flattened verify-shaped tensor
        (bs * verify_width) instead of bs ids. For fallback semantics we only
        want the first target token per request.
        """

        bs = len(batch.reqs)
        if bs <= 0:
            return next_token_ids.new_empty((0,), dtype=torch.long)

        flat = next_token_ids.detach().to(device=batch.device, dtype=torch.long).view(-1)
        if flat.numel() == bs:
            return flat

        if flat.numel() % bs == 0:
            width = flat.numel() // bs
            if self.tp_rank == 0:
                logger.warning(
                    "DASYNC fallback target decode returned verify-shaped next_token_ids "
                    "with %d values for bs=%d; using the first token per request "
                    "and ignoring width=%d tail values.",
                    int(flat.numel()),
                    int(bs),
                    int(width),
                )
            return flat.view(bs, width)[:, 0].contiguous()

        if flat.numel() > bs:
            if self.tp_rank == 0:
                logger.warning(
                    "DASYNC fallback target decode returned %d next_token_ids for bs=%d; "
                    "truncating to one token per request.",
                    int(flat.numel()),
                    int(bs),
                )
            return flat[:bs].contiguous()

        raise RuntimeError(
            "DASYNC fallback target decode returned fewer next_token_ids than requests: "
            f"numel={int(flat.numel())}, bs={int(bs)}."
        )

    def _append_sampled_tokens_after_fallback(
        self,
        batch: ScheduleBatch,
        next_token_ids: torch.Tensor,
    ) -> None:
        next_token_ids = self._normalize_fallback_next_token_ids(batch, next_token_ids)
        next_cpu = [int(x) for x in next_token_ids.detach().cpu().tolist()]
        for req, token_id in zip(batch.reqs, next_cpu, strict=True):
            req.output_ids.append(int(token_id))
            req.check_finished()
            if not req.finished() and req.grammar is not None:
                req.grammar.accept_token(int(token_id))
            req.spec_verify_ct += 1

    def _run_target_decode_fallback(
        self,
        batch: ScheduleBatch,
        state: DAsyncDraftInput,
        **kwargs,
    ) -> GenerationBatchResult:
        """Commit the current root token with the normal target decode path.

        This is the zero-future replacement.  It preserves the same DASYNC token
        state invariant as verifier commit_len=1: the current ``verified_id`` is
        materialized into the target KV cache, and the sampled target bonus token
        becomes the next unmaterialized ``verified_id``.  Unlike the old path it
        does not construct a padded fixed-width verifier input.
        """

        bs = batch.batch_size()
        old_spec_algorithm = batch.spec_algorithm
        old_spec_info = batch.spec_info
        old_return_hidden_states = batch.return_hidden_states
        old_forward_mode = batch.forward_mode
        try:
            batch.spec_algorithm = SpeculativeAlgorithm.NONE
            batch.spec_info = None
            batch.return_hidden_states = False
            batch.forward_mode = ForwardMode.DECODE
            batch.output_ids = state.verified_id.to(device=batch.device, dtype=torch.long)
            batch.prepare_for_decode()
            # Some hybrid/GDN paths infer metadata from forward_mode after
            # prepare_for_decode. Be explicit: fallback is ordinary one-token
            # target decode, never target-verify with spec_info=None.
            batch.forward_mode = ForwardMode.DECODE
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.forward_mode = ForwardMode.DECODE
            model_worker_batch.spec_info = None
            if self._condition_enabled():
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        finally:
            batch.spec_algorithm = old_spec_algorithm
            batch.spec_info = old_spec_info
            batch.return_hidden_states = old_return_hidden_states
            if old_forward_mode.is_idle():
                batch.forward_mode = old_forward_mode
            else:
                batch.forward_mode = ForwardMode.DECODE

        target_kwargs = dict(kwargs)
        # The scheduler calls speculative workers with is_verify=True on some
        # paths.  This fallback is intentionally ordinary decode, so do not let
        # that flag re-enable target-verify metadata inside TpModelWorker.
        target_kwargs.pop("is_verify", None)
        # DASYNC target verification captures a fixed-width TARGET_VERIFY graph.
        # This fallback is ordinary one-token decode; the DASYNC graph runner guard
        # rejects mode/width mismatches so the verify graph cannot be replayed here.
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, **target_kwargs
        )
        new_verified_id = batch_result.next_token_ids.to(
            device=batch.device, dtype=torch.int64
        )
        new_verified_id = self._normalize_fallback_next_token_ids(batch, new_verified_id)
        self._append_sampled_tokens_after_fallback(batch, new_verified_id)

        if self._condition_enabled():
            target_hidden = batch_result.logits_output.hidden_states
            if target_hidden is None:
                raise RuntimeError(
                    "DASYNC target-hidden condition requires hidden states from "
                    "ordinary fallback decode, but got None."
                )
            target_hidden = target_hidden[:bs]
            state.target_hidden = target_hidden
            state.ctx_lens = torch.ones((bs,), dtype=torch.int32, device=batch.device)
            batch_result.logits_output.hidden_states = None

        state.verified_id = new_verified_id
        for i, req in enumerate(batch.reqs):
            state.epochs[i] += 1
            self._send_upsert(
                req,
                epoch=state.epochs[i],
                prefix_ids=self._prefix_ids(req),
            )

        batch.spec_info = state
        batch.forward_mode = ForwardMode.DECODE
        self._fallback_decode_ct += 1
        self._consecutive_fallbacks += 1
        self._control_step += 1
        self._maybe_log_control_stats()
        if _dasync_dump_enabled():
            next_cpu = [int(x) for x in new_verified_id.detach().cpu().tolist()]
            self._dump_event(
                "verifier_fallback_decode_result",
                batch_size=int(bs),
                next_token_ids=_dasync_clip_list(next_cpu),
                next_token_text=self._decode_ids(next_cpu),
                can_run_cuda_graph=bool(batch_result.can_run_cuda_graph),
            )

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=0,
            accept_length_per_req_cpu=[0 for _ in range(bs)],
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    def _maybe_log_control_stats(self) -> None:
        interval = max(1, int(getattr(self.server_args, "decode_log_interval", 40)))
        if self.tp_rank != 0 or self._control_step % interval != 0:
            return
        proposal_ct = max(1, self._proposal_verify_ct)
        logger.info(
            "DASYNC control stats: steps=%s proposal_verify=%s fallback_decode=%s "
            "zero_future_skipped=%s partial_proposal_skipped=%s "
            "avg_ready_future=%.2f avg_proposal_accept=%.2f stale_exhausted=%s",
            self._control_step,
            self._proposal_verify_ct,
            self._fallback_decode_ct,
            self._zero_future_skip_ct,
            self._partial_proposal_skip_ct,
            self._proposal_ready_sum / proposal_ct,
            self._proposal_accept_sum / proposal_ct,
            self._proposal_stale_exhausted_ct,
        )

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if isinstance(batch, ModelWorkerBatch):
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "Invariant broken: DASYNC batch requested return_logprob, but scheduler should have rejected this request."
            )
        if getattr(batch, "has_grammar", False):
            raise RuntimeError(
                "Invariant broken: DASYNC batch has grammar constraints, but scheduler should have rejected this request."
            )

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            if self._condition_enabled():
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            if not batch.is_prefill_only:
                target_hidden = None
                ctx_lens = None
                if self._condition_enabled():
                    target_hidden = batch_result.logits_output.hidden_states
                    if target_hidden is None:
                        raise RuntimeError(
                            "DASYNC target-hidden condition requires hidden states "
                            "from prefill, but got None."
                        )
                    if model_worker_batch.extend_seq_lens is None:
                        raise RuntimeError(
                            "DASYNC target-hidden condition expected extend_seq_lens "
                            "to be populated during prefill."
                        )
                    ctx_lens = model_worker_batch.extend_seq_lens
                    if not isinstance(ctx_lens, torch.Tensor):
                        ctx_lens = torch.tensor(
                            ctx_lens, dtype=torch.int32, device=batch.device
                        )
                    else:
                        ctx_lens = ctx_lens.to(device=batch.device, dtype=torch.int32)
                    batch_result.logits_output.hidden_states = None
                self._bootstrap_after_prefill(
                    batch,
                    batch_result.next_token_ids,
                    target_hidden=target_hidden,
                    ctx_lens=ctx_lens,
                )
            return batch_result

        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise RuntimeError(
                "DASYNC currently supports greedy target sampling only."
            )

        state = batch.spec_info
        if not isinstance(state, DAsyncDraftInput):
            raise RuntimeError(
                "DASYNC decode requires DAsyncDraftInput state on the running batch. "
                "This usually means the request did not complete prefill."
            )

        verify_input = self._build_verify_input(batch, state)
        if verify_input is None:
            return self._run_target_decode_fallback(batch, state, **kwargs)

        verify_input.capture_hidden_mode = (
            CaptureHiddenMode.FULL if self._condition_enabled() else CaptureHiddenMode.NULL
        )
        verify_input.prepare_for_verify(batch, self.page_size, build_custom_mask=True)
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.return_hidden_states = False

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = batch.seq_lens.clone() if need_mamba_verify_commit else None

        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=batch_result.logits_output,
            page_size=self.page_size,
        )
        if _dasync_dump_enabled():
            candidates = getattr(verify_input, "_debug_candidates", None)
            target_predict = getattr(verify_input, "_debug_target_predict", None)
            matches = getattr(verify_input, "_debug_matches", None)
            raw_accept = getattr(verify_input, "_debug_accept_len_raw", None)
            bonus = getattr(verify_input, "_debug_bonus", None)
            accept_rows = getattr(verify_input, "_debug_accept_rows", [])
            commit_lens_cpu = [int(x) for x in commit_lens.detach().cpu().tolist()]
            new_verified_cpu = [int(x) for x in new_verified_id.detach().cpu().tolist()]
            rows = []
            for i, req in enumerate(batch.reqs):
                cand = candidates[i] if candidates is not None and i < len(candidates) else []
                pred = target_predict[i] if target_predict is not None and i < len(target_predict) else []
                row = {
                    "rid": str(getattr(req, "rid", i)),
                    "row": int(i),
                    "candidate_ids_verified": _dasync_clip_list(cand),
                    "candidate_text_verified": self._decode_ids(cand),
                    "target_argmax_ids": _dasync_clip_list(pred),
                    "target_argmax_text": self._decode_ids(pred),
                    "match_mask_future": matches[i] if matches is not None and i < len(matches) else None,
                    "raw_accepted_future_len": None if raw_accept is None or i >= len(raw_accept) else int(raw_accept[i]),
                    "bonus_id": None if bonus is None or i >= len(bonus) else int(bonus[i]),
                    "bonus_text": None if bonus is None or i >= len(bonus) else self._decode_ids([bonus[i]]),
                    "commit_len": commit_lens_cpu[i],
                    "accept_length": int(accept_length_per_req_cpu[i]),
                    "new_verified_id": new_verified_cpu[i],
                }
                if i < len(accept_rows):
                    row["accept_row"] = accept_rows[i]
                    row["accept_row"]["proposed_commit_text"] = self._decode_ids(
                        row["accept_row"].get("proposed_commit_ids", [])
                    )
                rows.append(row)
            self._dump_event(
                "verifier_result",
                batch_size=int(len(rows)),
                draft_token_num=int(verify_input.draft_token_num),
                accept_length_per_req=[int(x) for x in accept_length_per_req_cpu],
                rows=rows,
            )

        self._proposal_verify_ct += 1
        self._consecutive_fallbacks = 0
        self._proposal_accept_sum += sum(int(x) for x in accept_length_per_req_cpu)
        self._control_step += 1
        self._maybe_log_control_stats()

        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        state.verified_id = new_verified_id
        if self._condition_enabled():
            if next_target_hidden is None:
                raise RuntimeError(
                    "DASYNC target-hidden condition expected verify hidden segments, got None."
                )
            state.target_hidden = next_target_hidden
            state.ctx_lens = commit_lens
        for i, req in enumerate(batch.reqs):
            state.epochs[i] += 1
            self._send_upsert(
                req,
                epoch=state.epochs[i],
                prefix_ids=self._prefix_ids(req),
            )

        batch.spec_info = state
        batch.forward_mode = ForwardMode.DECODE

        if not self._logged_first_verify and self.tp_rank == 0:
            logger.info(
                "DASYNC verify completed. accept_length_per_req=%s",
                accept_length_per_req_cpu,
            )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=batch_result.logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=sum(accept_length_per_req_cpu),
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )
