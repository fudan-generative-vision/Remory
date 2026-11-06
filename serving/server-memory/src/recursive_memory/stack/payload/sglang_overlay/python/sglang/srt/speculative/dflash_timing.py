"""Low-overhead CUDA timing for the DFlash decode pipeline.

Enable with ``SGLANG_DFLASH_TIMING=1``. The optional
``SGLANG_DFLASH_TIMING_WARMUP_STEPS`` and
``SGLANG_DFLASH_TIMING_LOG_INTERVAL`` environment variables default to 20 and
100 decode steps, respectively.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Optional

import torch


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def _env_nonnegative_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[index]


class _CudaTrace:
    def __init__(self, device: torch.device):
        self.device = device
        self.events: dict[str, torch.cuda.Event] = {}
        self.recorded: set[str] = set()
        self.batch_size = 0
        self.accepted_tokens = 0
        self.judge_proposed_tokens = 0
        self.judge_blocks = 0
        self.judge_first_fail_blocks = 0
        self.judge_zero_proposal_requests = 0
        self.verify_width = 0
        self.can_run_cuda_graph = False
        self.discard = False

    def reset(self, *, batch_size: int) -> None:
        self.recorded.clear()
        self.batch_size = int(batch_size)
        self.accepted_tokens = 0
        self.judge_proposed_tokens = 0
        self.judge_blocks = 0
        self.judge_first_fail_blocks = 0
        self.judge_zero_proposal_requests = 0
        self.verify_width = 0
        self.can_run_cuda_graph = False
        self.discard = False

    def record(self, name: str) -> None:
        event = self.events.get(name)
        if event is None:
            event = torch.cuda.Event(enable_timing=True)
            self.events[name] = event
        event.record(torch.cuda.current_stream(self.device))
        self.recorded.add(name)

    def elapsed(self, start: str, end: str) -> float:
        if start not in self.recorded or end not in self.recorded:
            return 0.0
        return float(self.events[start].elapsed_time(self.events[end]))

    def elapsed_indexed(self, stage: str) -> tuple[float, bool]:
        """Sum indexed spans and report whether every observed pair is complete."""
        start_prefix = f"{stage}_start_"
        end_prefix = f"{stage}_end_"
        start_indices = {
            name[len(start_prefix) :]
            for name in self.recorded
            if name.startswith(start_prefix)
        }
        end_indices = {
            name[len(end_prefix) :]
            for name in self.recorded
            if name.startswith(end_prefix)
        }
        if start_indices != end_indices:
            return 0.0, False
        try:
            ordered_indices = sorted(start_indices, key=int)
        except ValueError:
            return 0.0, False
        total = sum(
            self.elapsed(
                f"{stage}_start_{index}",
                f"{stage}_end_{index}",
            )
            for index in ordered_indices
        )
        return total, True


class DFlashCudaTimingProfiler:
    """Collect per-stage DFlash GPU timings without per-step synchronization."""

    def __init__(
        self,
        *,
        device: torch.device,
        tp_rank: int,
        block_size: int,
    ) -> None:
        requested = _env_flag("SGLANG_DFLASH_TIMING", False)
        self.device = torch.device(device)
        self.block_size = int(block_size)
        self.flow_steps = max(
            _env_nonnegative_int("SGLANG_DFLASH_FLOW_STEPS", 1), 1
        )
        self.enabled = bool(
            requested
            and int(tp_rank) == 0
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.warmup_steps = _env_nonnegative_int(
            "SGLANG_DFLASH_TIMING_WARMUP_STEPS", 20
        )
        self.log_interval = _env_nonnegative_int(
            "SGLANG_DFLASH_TIMING_LOG_INTERVAL", 100
        )
        if requested and self.log_interval == 0:
            raise ValueError("SGLANG_DFLASH_TIMING_LOG_INTERVAL must be positive")

        self._seen_steps = 0
        self._measured_steps = 0
        self._active: Optional[_CudaTrace] = None
        self._pending: deque[_CudaTrace] = deque()
        self._pool: list[_CudaTrace] = []
        self._window: list[dict[str, float]] = []
        self._cumulative_sums: dict[str, float] = {}
        self._cumulative_verify_buckets: dict[int, dict[str, float]] = {}
        self._warned_incomplete_trace = False

        if self.enabled:
            logger.info(
                "DFLASH CUDA timing enabled on TP rank 0. warmup_steps=%d, "
                "log_interval=%d, block_size=%d. Timing is decode-only and "
                "excludes CPU-only gaps.",
                self.warmup_steps,
                self.log_interval,
                self.block_size,
            )

    def start_step(self, *, batch_size: int) -> None:
        if not self.enabled:
            return
        self._drain_completed()
        self._seen_steps += 1
        if self._seen_steps <= self.warmup_steps:
            return
        if self._active is not None:
            self.abort_step()

        trace = self._pool.pop() if self._pool else _CudaTrace(self.device)
        trace.reset(batch_size=batch_size)
        trace.record("decode_start")
        self._active = trace

    def mark(self, name: str) -> None:
        if self._active is not None:
            self._active.record(name)

    def finish_step(
        self,
        *,
        accepted_tokens: int,
        judge_proposed_tokens: int = 0,
        judge_blocks: int = 0,
        judge_first_fail_blocks: int = 0,
        judge_zero_proposal_requests: int = 0,
        verify_width: int = 0,
        can_run_cuda_graph: bool = False,
    ) -> None:
        trace = self._active
        if trace is None:
            return
        trace.accepted_tokens = int(accepted_tokens)
        trace.judge_proposed_tokens = int(judge_proposed_tokens)
        trace.judge_blocks = int(judge_blocks)
        trace.judge_first_fail_blocks = int(judge_first_fail_blocks)
        trace.judge_zero_proposal_requests = int(judge_zero_proposal_requests)
        trace.verify_width = int(verify_width)
        trace.can_run_cuda_graph = bool(can_run_cuda_graph)
        trace.record("decode_end")
        self._pending.append(trace)
        self._active = None
        self._drain_completed()

    def abort_step(self) -> None:
        trace = self._active
        if trace is None:
            return
        trace.discard = True
        trace.record("decode_end")
        self._pending.append(trace)
        self._active = None
        self._drain_completed()

    def _drain_completed(self) -> None:
        while self._pending:
            trace = self._pending[0]
            if not trace.events["decode_end"].query():
                break
            self._pending.popleft()
            if not trace.discard:
                sample = self._make_sample(trace)
                if sample is not None:
                    self._window.append(sample)
                    self._measured_steps += 1
                    for key, value in sample.items():
                        self._cumulative_sums[key] = (
                            self._cumulative_sums.get(key, 0.0) + value
                        )
                    width = int(sample["verify_width"])
                    bucket = self._cumulative_verify_buckets.setdefault(
                        width,
                        {
                            "steps": 0.0,
                            "requests": 0.0,
                            "verifier_ms": 0.0,
                            "graph_steps": 0.0,
                        },
                    )
                    bucket["steps"] += 1.0
                    bucket["requests"] += sample["batch_size"]
                    bucket["verifier_ms"] += sample["verifier"]
                    bucket["graph_steps"] += sample["can_run_cuda_graph"]
            self._pool.append(trace)

        if len(self._window) >= self.log_interval:
            self._log_window()

    def _make_sample(self, trace: _CudaTrace) -> Optional[dict[str, float]]:
        required = {
            "decode_start",
            "verifier_start",
            "verifier_end",
            "decode_end",
        }
        if not required.issubset(trace.recorded):
            if not self._warned_incomplete_trace:
                logger.warning(
                    "Dropping an incomplete DFLASH timing trace. recorded=%s",
                    sorted(trace.recorded),
                )
                self._warned_incomplete_trace = True
            return None

        sample = {
            "batch_size": float(trace.batch_size),
            "accepted_tokens": float(trace.accepted_tokens),
            "judge_proposed_tokens": float(trace.judge_proposed_tokens),
            "judge_blocks": float(trace.judge_blocks),
            "judge_first_fail_blocks": float(trace.judge_first_fail_blocks),
            "judge_zero_proposal_requests": float(
                trace.judge_zero_proposal_requests
            ),
            "verify_width": float(trace.verify_width),
            "can_run_cuda_graph": float(trace.can_run_cuda_graph),
            "total": trace.elapsed("decode_start", "decode_end"),
            "draft_total": trace.elapsed("decode_start", "verifier_start"),
            "verifier": trace.elapsed("verifier_start", "verifier_end"),
            "post_total": trace.elapsed("verifier_end", "decode_end"),
            "draft_kv_pre": trace.elapsed("draft_kv_pre_start", "draft_kv_pre_end"),
            "draft_backbone": trace.elapsed(
                "draft_backbone_start", "draft_backbone_end"
            ),
            "draft_head": trace.elapsed("draft_head_start", "draft_head_end"),
            "flow_total": trace.elapsed("flow_total_start", "flow_total_end"),
            "flow_init": trace.elapsed("flow_init_start", "flow_init_end"),
            "flow_denoise": trace.elapsed(
                "flow_denoise_start", "flow_denoise_end"
            ),
            "flow_decode": trace.elapsed("flow_decode_start", "flow_decode_end"),
            "accept": trace.elapsed("accept_start", "accept_end"),
            "mamba_commit": trace.elapsed("mamba_commit_start", "mamba_commit_end"),
            "draft_kv_post": trace.elapsed("draft_kv_post_start", "draft_kv_post_end"),
        }
        sample["flow_denoise_step"] = (
            sample["flow_denoise"] / float(self.flow_steps)
        )
        sample["flow_other"] = max(
            0.0,
            sample["flow_total"]
            - sample["flow_init"]
            - sample["flow_denoise"]
            - sample["flow_decode"],
        )
        # Active drafter/judge invokes multiple draft blocks in a single decode
        # step.  Generic event names are intentionally retained for legacy
        # DFlash, while indexed spans make every active block additive instead
        # of silently reporting only the last event recording.
        indexed_stages = (
            "active_draft",
            "active_draft_backbone",
            "active_draft_head",
            "active_judge",
            "active_judge_model",
        )
        indexed_elapsed: dict[str, float] = {}
        for stage in indexed_stages:
            elapsed, complete = trace.elapsed_indexed(stage)
            if not complete:
                if not self._warned_incomplete_trace:
                    logger.warning(
                        "Dropping DFLASH timing trace with an unmatched indexed "
                        "stage marker: stage=%s recorded=%s",
                        stage,
                        sorted(trace.recorded),
                    )
                    self._warned_incomplete_trace = True
                return None
            indexed_elapsed[stage] = elapsed

        sample["active_draft_total"] = indexed_elapsed["active_draft"]
        sample["active_draft_backbone"] = indexed_elapsed[
            "active_draft_backbone"
        ]
        sample["active_draft_head"] = indexed_elapsed["active_draft_head"]
        sample["active_drafter_model"] = (
            sample["active_draft_backbone"] + sample["active_draft_head"]
        )
        sample["active_judge_total"] = indexed_elapsed["active_judge"]
        sample["active_judge_model"] = indexed_elapsed["active_judge_model"]
        active_total = (
            sample["active_draft_total"] + sample["active_judge_total"]
        )
        if active_total > 0.0:
            # Keep the legacy detail fields meaningful in active mode too:
            # indexed spans are additive, unlike repeatedly recorded generic
            # CUDA events, which would otherwise expose only the last block.
            sample["draft_backbone"] = sample["active_draft_backbone"]
            sample["draft_head"] = sample["active_draft_head"]
        accounted_draft_total = (
            active_total
            if active_total > 0.0
            else sample["draft_backbone"] + sample["draft_head"]
        )
        sample["draft_other"] = max(
            0.0,
            sample["draft_total"]
            - sample["draft_kv_pre"]
            - accounted_draft_total,
        )
        sample["post_other"] = max(
            0.0,
            sample["post_total"]
            - sample["accept"]
            - sample["mamba_commit"]
            - sample["draft_kv_post"],
        )
        return sample

    @staticmethod
    def _format_verify_bucket_stats(
        stats: dict[int, dict[str, float]],
    ) -> str:
        fields = []
        for width in sorted(stats):
            bucket = stats[width]
            steps = max(bucket["steps"], 1.0)
            fields.append(
                "W%d:n=%d,batch=%.2f,verifier=%.3fms,graph=%.1f%%"
                % (
                    width,
                    int(bucket["steps"]),
                    bucket["requests"] / steps,
                    bucket["verifier_ms"] / steps,
                    100.0 * bucket["graph_steps"] / steps,
                )
            )
        return ";".join(fields) if fields else "none"

    @classmethod
    def _summarize_verify_buckets(
        cls,
        samples: list[dict[str, float]],
    ) -> str:
        stats: dict[int, dict[str, float]] = {}
        for sample in samples:
            width = int(sample["verify_width"])
            bucket = stats.setdefault(
                width,
                {
                    "steps": 0.0,
                    "requests": 0.0,
                    "verifier_ms": 0.0,
                    "graph_steps": 0.0,
                },
            )
            bucket["steps"] += 1.0
            bucket["requests"] += sample["batch_size"]
            bucket["verifier_ms"] += sample["verifier"]
            bucket["graph_steps"] += sample["can_run_cuda_graph"]
        return cls._format_verify_bucket_stats(stats)

    def _log_window(self) -> None:
        samples = self._window[: self.log_interval]
        del self._window[: self.log_interval]

        def mean(name: str) -> float:
            return sum(sample[name] for sample in samples) / len(samples)

        total = mean("total")
        draft_total = mean("draft_total")
        verifier = mean("verifier")
        post_total = mean("post_total")
        denominator = max(total, 1e-9)
        active_draft_total = mean("active_draft_total")
        active_drafter_model = mean("active_drafter_model")
        active_judge_total = mean("active_judge_total")
        active_judge_model = mean("active_judge_model")
        is_active_sample = active_draft_total + active_judge_total > 0.0
        drafter_compute = (
            active_drafter_model
            if is_active_sample
            else mean("draft_backbone") + mean("draft_head")
        )
        judge_compute = active_judge_model if is_active_sample else 0.0
        model_compute = max(drafter_compute + judge_compute + verifier, 1e-9)
        request_steps = sum(sample["batch_size"] for sample in samples)
        accepted_draft_tokens = sum(sample["accepted_tokens"] for sample in samples)
        accepted_draft_per_request_step = accepted_draft_tokens / max(
            request_steps, 1e-9
        )
        judge_proposed_tokens = sum(
            sample["judge_proposed_tokens"] for sample in samples
        )
        judge_blocks = sum(sample["judge_blocks"] for sample in samples)
        judge_first_fail_blocks = sum(
            sample["judge_first_fail_blocks"] for sample in samples
        )
        judge_zero_proposal_requests = sum(
            sample["judge_zero_proposal_requests"] for sample in samples
        )
        judge_proposed_per_request_step = judge_proposed_tokens / max(
            request_steps, 1e-9
        )
        target_accept_over_judge_proposed = accepted_draft_tokens / max(
            judge_proposed_tokens, 1e-9
        )
        judge_blocks_per_request_step = judge_blocks / max(request_steps, 1e-9)
        judge_stop_first_rate = judge_first_fail_blocks / max(judge_blocks, 1e-9)
        judge_zero_proposal_rate = judge_zero_proposal_requests / max(
            request_steps, 1e-9
        )
        accept_length = (accepted_draft_tokens + request_steps) / max(
            request_steps, 1e-9
        )
        output_tokens = accepted_draft_tokens + request_steps
        total_window_ms = sum(sample["total"] for sample in samples)
        decode_gpu_tokens_per_second = (
            output_tokens * 1000.0 / max(total_window_ms, 1e-9)
        )
        total_values = [sample["total"] for sample in samples]

        logger.info(
            "DFLASH_TIMING_CUDA steps=%d window=%d batch_avg=%.2f "
            "accepted_draft_per_request_step=%.3f accept_length=%.3f "
            "active_quality[proposed_per_req=%.3f target_over_proposed=%.3f "
            "blocks_per_req=%.3f stop_first_rate=%.3f zero_proposal_rate=%.3f] "
            "decode_gpu_output_tokens_per_s=%.2f total_ms=%.3f "
            "p50_ms=%.3f p95_ms=%.3f | draft_ms=%.3f (%.1f%%) "
            "verifier_ms=%.3f (%.1f%%) post_ms=%.3f (%.1f%%) | "
            "model_compute_split[drafter=%.3f (%.1f%%) judge=%.3f (%.1f%%) "
            "verifier=%.3f (%.1f%%)] | verify_buckets[%s] | "
            "draft_detail_ms[kv_pre=%.3f backbone=%.3f head=%.3f other=%.3f] | "
            "flow_detail_ms[total=%.3f init=%.3f denoise=%.3f "
            "denoise_step=%.3f decode=%.3f other=%.3f] | "
            "active_detail_ms[draft_block=%.3f drafter_model=%.3f "
            "judge_total=%.3f judge_model=%.3f] | "
            "post_detail_ms[accept=%.3f mamba=%.3f kv_post=%.3f other=%.3f]",
            self._measured_steps,
            len(samples),
            mean("batch_size"),
            accepted_draft_per_request_step,
            accept_length,
            judge_proposed_per_request_step,
            target_accept_over_judge_proposed,
            judge_blocks_per_request_step,
            judge_stop_first_rate,
            judge_zero_proposal_rate,
            decode_gpu_tokens_per_second,
            total,
            _percentile(total_values, 0.50),
            _percentile(total_values, 0.95),
            draft_total,
            100.0 * draft_total / denominator,
            verifier,
            100.0 * verifier / denominator,
            post_total,
            100.0 * post_total / denominator,
            drafter_compute,
            100.0 * drafter_compute / model_compute,
            judge_compute,
            100.0 * judge_compute / model_compute,
            verifier,
            100.0 * verifier / model_compute,
            self._summarize_verify_buckets(samples),
            mean("draft_kv_pre"),
            mean("draft_backbone"),
            mean("draft_head"),
            mean("draft_other"),
            mean("flow_total"),
            mean("flow_init"),
            mean("flow_denoise"),
            mean("flow_denoise_step"),
            mean("flow_decode"),
            mean("flow_other"),
            active_draft_total,
            active_drafter_model,
            active_judge_total,
            active_judge_model,
            mean("accept"),
            mean("mamba_commit"),
            mean("draft_kv_post"),
            mean("post_other"),
        )

        def cumulative_mean(name: str) -> float:
            return self._cumulative_sums[name] / self._measured_steps

        cumulative_total = cumulative_mean("total")
        cumulative_draft = cumulative_mean("draft_total")
        cumulative_verifier = cumulative_mean("verifier")
        cumulative_post = cumulative_mean("post_total")
        cumulative_denominator = max(cumulative_total, 1e-9)
        cumulative_active_draft_total = cumulative_mean("active_draft_total")
        cumulative_active_drafter_model = cumulative_mean(
            "active_drafter_model"
        )
        cumulative_active_judge_total = cumulative_mean("active_judge_total")
        cumulative_active_judge_model = cumulative_mean("active_judge_model")
        cumulative_is_active = (
            cumulative_active_draft_total + cumulative_active_judge_total > 0.0
        )
        cumulative_drafter_compute = (
            cumulative_active_drafter_model
            if cumulative_is_active
            else cumulative_mean("draft_backbone")
            + cumulative_mean("draft_head")
        )
        cumulative_judge_compute = (
            cumulative_active_judge_model if cumulative_is_active else 0.0
        )
        cumulative_model_compute = max(
            cumulative_drafter_compute
            + cumulative_judge_compute
            + cumulative_verifier,
            1e-9,
        )
        cumulative_request_steps = self._cumulative_sums["batch_size"]
        cumulative_accepted_draft_tokens = self._cumulative_sums["accepted_tokens"]
        cumulative_accepted_draft_per_request_step = (
            cumulative_accepted_draft_tokens / max(cumulative_request_steps, 1e-9)
        )
        cumulative_judge_proposed_tokens = self._cumulative_sums[
            "judge_proposed_tokens"
        ]
        cumulative_judge_blocks = self._cumulative_sums["judge_blocks"]
        cumulative_judge_first_fail_blocks = self._cumulative_sums[
            "judge_first_fail_blocks"
        ]
        cumulative_judge_zero_proposal_requests = self._cumulative_sums[
            "judge_zero_proposal_requests"
        ]
        cumulative_judge_proposed_per_request_step = (
            cumulative_judge_proposed_tokens
            / max(cumulative_request_steps, 1e-9)
        )
        cumulative_target_accept_over_judge_proposed = (
            cumulative_accepted_draft_tokens
            / max(cumulative_judge_proposed_tokens, 1e-9)
        )
        cumulative_judge_blocks_per_request_step = cumulative_judge_blocks / max(
            cumulative_request_steps, 1e-9
        )
        cumulative_judge_stop_first_rate = (
            cumulative_judge_first_fail_blocks
            / max(cumulative_judge_blocks, 1e-9)
        )
        cumulative_judge_zero_proposal_rate = (
            cumulative_judge_zero_proposal_requests
            / max(cumulative_request_steps, 1e-9)
        )
        cumulative_accept_length = (
            cumulative_accepted_draft_tokens + cumulative_request_steps
        ) / max(cumulative_request_steps, 1e-9)
        cumulative_decode_gpu_tokens_per_second = (
            (cumulative_accepted_draft_tokens + cumulative_request_steps)
            * 1000.0
            / max(self._cumulative_sums["total"], 1e-9)
        )
        logger.info(
            "DFLASH_TIMING_CUDA_CUMULATIVE steps=%d batch_avg=%.2f "
            "accepted_draft_per_request_step=%.3f accept_length=%.3f "
            "active_quality[proposed_per_req=%.3f target_over_proposed=%.3f "
            "blocks_per_req=%.3f stop_first_rate=%.3f zero_proposal_rate=%.3f] "
            "decode_gpu_output_tokens_per_s=%.2f total_ms=%.3f | "
            "draft_ms=%.3f (%.1f%%) verifier_ms=%.3f (%.1f%%) "
            "post_ms=%.3f (%.1f%%) | model_compute_split[drafter=%.3f (%.1f%%) "
            "judge=%.3f (%.1f%%) verifier=%.3f (%.1f%%)] | "
            "verify_buckets[%s] | "
            "draft_detail_ms[kv_pre=%.3f backbone=%.3f head=%.3f other=%.3f] | "
            "flow_detail_ms[total=%.3f init=%.3f denoise=%.3f "
            "denoise_step=%.3f decode=%.3f other=%.3f] | "
            "active_detail_ms[draft_block=%.3f drafter_model=%.3f "
            "judge_total=%.3f judge_model=%.3f] | "
            "post_detail_ms[accept=%.3f mamba=%.3f kv_post=%.3f other=%.3f]",
            self._measured_steps,
            cumulative_mean("batch_size"),
            cumulative_accepted_draft_per_request_step,
            cumulative_accept_length,
            cumulative_judge_proposed_per_request_step,
            cumulative_target_accept_over_judge_proposed,
            cumulative_judge_blocks_per_request_step,
            cumulative_judge_stop_first_rate,
            cumulative_judge_zero_proposal_rate,
            cumulative_decode_gpu_tokens_per_second,
            cumulative_total,
            cumulative_draft,
            100.0 * cumulative_draft / cumulative_denominator,
            cumulative_verifier,
            100.0 * cumulative_verifier / cumulative_denominator,
            cumulative_post,
            100.0 * cumulative_post / cumulative_denominator,
            cumulative_drafter_compute,
            100.0 * cumulative_drafter_compute / cumulative_model_compute,
            cumulative_judge_compute,
            100.0 * cumulative_judge_compute / cumulative_model_compute,
            cumulative_verifier,
            100.0 * cumulative_verifier / cumulative_model_compute,
            self._format_verify_bucket_stats(self._cumulative_verify_buckets),
            cumulative_mean("draft_kv_pre"),
            cumulative_mean("draft_backbone"),
            cumulative_mean("draft_head"),
            cumulative_mean("draft_other"),
            cumulative_mean("flow_total"),
            cumulative_mean("flow_init"),
            cumulative_mean("flow_denoise"),
            cumulative_mean("flow_denoise_step"),
            cumulative_mean("flow_decode"),
            cumulative_mean("flow_other"),
            cumulative_active_draft_total,
            cumulative_active_drafter_model,
            cumulative_active_judge_total,
            cumulative_active_judge_model,
            cumulative_mean("accept"),
            cumulative_mean("mamba_commit"),
            cumulative_mean("draft_kv_post"),
            cumulative_mean("post_other"),
        )
