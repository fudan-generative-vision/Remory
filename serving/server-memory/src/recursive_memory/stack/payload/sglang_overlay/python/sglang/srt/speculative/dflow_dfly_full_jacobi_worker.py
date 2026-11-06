"""Optional fixed-round Jacobi runtime for full DFlow/DFly HSC.

This module is deliberately separate from the frozen sequential Stage-J
runtime.  It consumes the same strictly validated ``DFlowDFlyFullDraftModel``
checkpoint and keeps the sole physical draft-backbone invocation, but replaces
the 15-step Gauss-Seidel HSC/head chain by a fixed number of block-parallel
Jacobi refinements.

For decoded future states ``h`` the proposal is

``q0 = head(h)``

followed by ``R`` refinements

``q_r = head(HSC(h, embed([verified_root, q_(r-1)[:-1]))))``.

The only supported round counts are 1, 2, 4, 8, and 15.  Launch is also gated
by a content-addressed Stage-J step-500 efficacy manifest, so this diagnostic
runtime cannot silently become the primary Stage-J evaluation path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashDraftInput, DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import (
    is_dflash_sampling_verify_available,
    resolve_dflash_verify_mask_policy,
)
from sglang.srt.speculative.dflow_dfly_full_worker import (
    DFlowDFlyFullWorker,
    _assert_tp_replicated_bitwise,
)


logger = logging.getLogger(__name__)

_JACOBI_ROUNDS_ENV = "SGLANG_DFLOW_DFLY_FULL_JACOBI_ROUNDS"
_JACOBI_GATE_PATH_ENV = "SGLANG_DFLOW_DFLY_FULL_JACOBI_GATE_MANIFEST"
_JACOBI_GATE_SHA_ENV = "SGLANG_DFLOW_DFLY_FULL_JACOBI_GATE_SHA256"
_JACOBI_TIMING_ENV = "SGLANG_DFLOW_DFLY_FULL_JACOBI_TIMING"
_JACOBI_ALLOWED_ROUNDS = frozenset({1, 2, 4, 8, 15})
_JACOBI_GATE_SCHEMA = "dflow-dfly-full-jacobi-stage-j500-efficacy-gate-v1"
_JACOBI_GATE_RULE = "delta_al_ge_0.15_and_paired_ci95_low_gt_0"
_JACOBI_TIMING_SCHEMA = "dflow-dfly-full-jacobi-timing-v1"
_JACOBI_TIMING_LOG_PREFIX = "DFLOW_DFLY_JACOBI_TIMING_JSON"
_JACOBI_EVAL_CONTRACT = {
    "dataset": "math500",
    "num_prompts": 500,
    "concurrency": 16,
    "enable_thinking": False,
    "tensor_parallel_size": 2,
    "flow_noise_samples": 1,
    "random_seed": 42,
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_jacobi_rounds(raw: object) -> int:
    """Parse the experimental refinement count without permissive coercions."""

    if isinstance(raw, bool):
        raise ValueError(
            f"{_JACOBI_ROUNDS_ENV} must be one of "
            f"{sorted(_JACOBI_ALLOWED_ROUNDS)}, got boolean {raw!r}."
        )
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_JACOBI_ROUNDS_ENV} must be one of "
            f"{sorted(_JACOBI_ALLOWED_ROUNDS)}, got {raw!r}."
        ) from exc
    if str(raw).strip() != str(value) or value not in _JACOBI_ALLOWED_ROUNDS:
        raise ValueError(
            f"{_JACOBI_ROUNDS_ENV} must be one of "
            f"{sorted(_JACOBI_ALLOWED_ROUNDS)}, got {raw!r}."
        )
    return value


def validate_stage_j500_efficacy_gate(
    *,
    manifest_path: str | os.PathLike[str],
    expected_sha256: str,
    draft_model_path: str | os.PathLike[str],
    verify_referenced_files: bool = True,
) -> dict[str, Any]:
    """Validate the content-addressed prerequisite for optional Jacobi runs."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Jacobi efficacy gate manifest is missing: {path}")
    if not _is_sha256(expected_sha256):
        raise ValueError(
            f"{_JACOBI_GATE_SHA_ENV} must be a lowercase SHA256, "
            f"got {expected_sha256!r}."
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Jacobi efficacy gate manifest SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Jacobi efficacy gate JSON: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Jacobi efficacy gate must be a JSON object.")

    errors: list[str] = []
    if payload.get("schema") != _JACOBI_GATE_SCHEMA:
        errors.append(f"schema={payload.get('schema')!r}")
    checkpoint_step = payload.get("checkpoint_step")
    if isinstance(checkpoint_step, bool) or checkpoint_step != 500:
        errors.append(f"checkpoint_step={checkpoint_step!r}")
    if payload.get("passed") is not True:
        errors.append(f"passed={payload.get('passed')!r}")
    if payload.get("rule") != _JACOBI_GATE_RULE:
        errors.append(f"rule={payload.get('rule')!r}")

    checkpoint_path = payload.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.startswith("/"):
        errors.append(f"checkpoint_path={checkpoint_path!r}")
    else:
        recorded = Path(checkpoint_path).expanduser().resolve()
        requested = Path(draft_model_path).expanduser().resolve()
        if recorded != requested:
            errors.append(
                f"checkpoint_path={str(recorded)!r}, requested={str(requested)!r}"
            )

    for key in ("checkpoint_payload_sha256", "evaluation_artifact_sha256"):
        if not _is_sha256(payload.get(key)):
            errors.append(f"{key}={payload.get(key)!r}")

    evaluation_path = payload.get("evaluation_artifact_path")
    if not isinstance(evaluation_path, str) or not evaluation_path.startswith("/"):
        errors.append(f"evaluation_artifact_path={evaluation_path!r}")

    if verify_referenced_files and isinstance(checkpoint_path, str):
        checkpoint_file = (
            Path(checkpoint_path).expanduser().resolve() / "model.safetensors"
        )
        if not checkpoint_file.is_file():
            errors.append(f"checkpoint_payload_missing={str(checkpoint_file)!r}")
        elif _is_sha256(payload.get("checkpoint_payload_sha256")):
            actual_checkpoint_sha = _sha256_file(checkpoint_file)
            if actual_checkpoint_sha != payload["checkpoint_payload_sha256"]:
                errors.append(
                    "checkpoint_payload_sha256="
                    f"{payload['checkpoint_payload_sha256']!r}, "
                    f"actual={actual_checkpoint_sha!r}"
                )
    if verify_referenced_files and isinstance(evaluation_path, str):
        artifact = Path(evaluation_path).expanduser().resolve()
        if not artifact.is_file():
            errors.append(f"evaluation_artifact_missing={str(artifact)!r}")
        elif _is_sha256(payload.get("evaluation_artifact_sha256")):
            actual_evaluation_sha = _sha256_file(artifact)
            if actual_evaluation_sha != payload["evaluation_artifact_sha256"]:
                errors.append(
                    "evaluation_artifact_sha256="
                    f"{payload['evaluation_artifact_sha256']!r}, "
                    f"actual={actual_evaluation_sha!r}"
                )

    delta = payload.get("delta_accept_length")
    if (
        isinstance(delta, bool)
        or not isinstance(delta, (int, float))
        or not math.isfinite(float(delta))
        or float(delta) < 0.15
    ):
        errors.append(f"delta_accept_length={delta!r}")

    interval = payload.get("paired_ci95")
    if (
        not isinstance(interval, list)
        or len(interval) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in interval
        )
    ):
        errors.append(f"paired_ci95={interval!r}")
    else:
        low, high = (float(interval[0]), float(interval[1]))
        if not (0.0 < low <= high):
            errors.append(f"paired_ci95={interval!r}")

    evaluation_contract = payload.get("evaluation_contract")
    if evaluation_contract != _JACOBI_EVAL_CONTRACT:
        errors.append(f"evaluation_contract={evaluation_contract!r}")

    allowed_keys = frozenset(
        {
            "schema",
            "checkpoint_step",
            "checkpoint_path",
            "checkpoint_payload_sha256",
            "evaluation_artifact_path",
            "evaluation_artifact_sha256",
            "delta_accept_length",
            "paired_ci95",
            "passed",
            "rule",
            "evaluation_contract",
        }
    )
    unknown = sorted(set(payload) - allowed_keys)
    missing = sorted(allowed_keys - set(payload))
    if unknown:
        errors.append(f"unknown_keys={unknown}")
    if missing:
        errors.append(f"missing_keys={missing}")
    if errors:
        raise ValueError(
            "Stage-J500 efficacy gate does not authorize Jacobi runtime: "
            + "; ".join(errors)
            + "."
        )
    return dict(payload)


def validate_jacobi_gate_from_env(
    *,
    draft_model_path: str | os.PathLike[str],
    environ=None,
    verify_referenced_files: bool = True,
) -> dict[str, Any]:
    if environ is None:
        environ = os.environ
    manifest_path = str(environ.get(_JACOBI_GATE_PATH_ENV, "")).strip()
    expected_sha256 = str(environ.get(_JACOBI_GATE_SHA_ENV, "")).strip()
    if not manifest_path:
        raise ValueError(f"{_JACOBI_GATE_PATH_ENV} is required.")
    if not expected_sha256:
        raise ValueError(f"{_JACOBI_GATE_SHA_ENV} is required.")
    return validate_stage_j500_efficacy_gate(
        manifest_path=manifest_path,
        expected_sha256=expected_sha256,
        draft_model_path=draft_model_path,
        verify_referenced_files=verify_referenced_files,
    )


@dataclass(frozen=True)
class JacobiPassCounts:
    backbone_forwards: int
    base_head_passes: int
    correction_head_passes: int
    head_passes: int
    embedding_passes: int
    hidden_correction_passes: int
    expected_tp2_collectives: int

    @classmethod
    def for_rounds(cls, rounds: int) -> "JacobiPassCounts":
        rounds = parse_jacobi_rounds(rounds)
        return cls(
            backbone_forwards=1,
            base_head_passes=1,
            correction_head_passes=rounds,
            head_passes=1 + rounds,
            embedding_passes=rounds,
            hidden_correction_passes=rounds,
            # Each vocab-parallel greedy head performs max/id all-gathers;
            # each sharded target embedding lookup performs one all-reduce.
            expected_tp2_collectives=2 * (1 + rounds) + rounds,
        )


def jacobi_timing_marker_order(rounds: int) -> tuple[str, ...]:
    """Return the exact marker ABI consumed by the Jacobi timing profiler."""

    rounds = parse_jacobi_rounds(rounds)
    markers = ["jacobi_base_head_start", "jacobi_base_head_end"]
    for round_index in range(1, rounds + 1):
        markers.extend(
            (
                f"jacobi_embed_start_{round_index}",
                f"jacobi_embed_end_{round_index}",
                f"jacobi_hsc_start_{round_index}",
                f"jacobi_hsc_end_{round_index}",
                f"jacobi_head_start_{round_index}",
                f"jacobi_head_end_{round_index}",
            )
        )
    return tuple(markers)


def _jacobi_timing_sample_from_elapsed(
    *,
    rounds: int,
    batch_size: int,
    elapsed: Callable[[str, str], float],
) -> dict[str, float]:
    """Consume every Jacobi marker pair and produce one flat timing sample.

    Keeping this conversion independent of CUDA makes the profiler's marker
    coverage testable on CPU.  The callable is normally backed by two CUDA
    events, but a deterministic test oracle can supply the same interface.
    """

    rounds = parse_jacobi_rounds(rounds)

    def duration(start: str, end: str) -> float:
        value = float(elapsed(start, end))
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(
                "Jacobi CUDA timing produced an invalid duration for "
                f"{start!r}->{end!r}: {value!r}."
            )
        return value

    sample: dict[str, float] = {
        "batch_size": float(batch_size),
        "refinement_total_ms": duration(
            "jacobi_base_head_start", f"jacobi_head_end_{rounds}"
        ),
        "base_head_ms": duration(
            "jacobi_base_head_start", "jacobi_base_head_end"
        ),
    }
    embedding_total = 0.0
    hsc_total = 0.0
    correction_head_total = 0.0
    for round_index in range(1, rounds + 1):
        embed_ms = duration(
            f"jacobi_embed_start_{round_index}",
            f"jacobi_embed_end_{round_index}",
        )
        hsc_ms = duration(
            f"jacobi_hsc_start_{round_index}",
            f"jacobi_hsc_end_{round_index}",
        )
        head_ms = duration(
            f"jacobi_head_start_{round_index}",
            f"jacobi_head_end_{round_index}",
        )
        sample[f"round_{round_index}_embed_ms"] = embed_ms
        sample[f"round_{round_index}_hsc_ms"] = hsc_ms
        sample[f"round_{round_index}_head_ms"] = head_ms
        sample[f"round_{round_index}_component_sum_ms"] = (
            embed_ms + hsc_ms + head_ms
        )
        embedding_total += embed_ms
        hsc_total += hsc_ms
        correction_head_total += head_ms

    sample["embedding_total_ms"] = embedding_total
    sample["hsc_total_ms"] = hsc_total
    sample["correction_head_total_ms"] = correction_head_total
    sample["component_sum_ms"] = (
        sample["base_head_ms"]
        + embedding_total
        + hsc_total
        + correction_head_total
    )
    # This exposes concatenation, launch gaps, and any other work between the
    # explicitly timed kernels instead of silently assigning it to HSC/head.
    sample["unattributed_ms"] = max(
        0.0, sample["refinement_total_ms"] - sample["component_sum_ms"]
    )
    return sample


def _parse_boolean_flag(raw: object, *, name: str) -> bool:
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}.")


def _parse_nonnegative_timing_int(
    name: str, *, default: int, environ: Mapping[str, object]
) -> int:
    raw = environ.get(name, default)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a non-negative integer, got {raw!r}."
        ) from exc
    if str(raw).strip() != str(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {raw!r}.")
    return value


class _JacobiCudaTrace:
    """Reusable event set for one asynchronous Jacobi refinement trace."""

    _ABORT_COMPLETION = "__jacobi_profiler_abort_completion"

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.events: dict[str, Any] = {}
        self.recorded: set[str] = set()
        self.marker_order: tuple[str, ...] = ()
        self.next_marker = 0
        self.batch_size = 0
        self.discard = False

    def reset(self, *, batch_size: int, rounds: int) -> None:
        self.recorded.clear()
        self.marker_order = jacobi_timing_marker_order(rounds)
        self.next_marker = 0
        self.batch_size = int(batch_size)
        self.discard = False

    def _record_event(self, name: str) -> None:
        event = self.events.get(name)
        if event is None:
            event = torch.cuda.Event(enable_timing=True)
            self.events[name] = event
        event.record(torch.cuda.current_stream(self.device))
        self.recorded.add(name)

    def record_marker(self, name: str) -> None:
        expected = (
            self.marker_order[self.next_marker]
            if self.next_marker < len(self.marker_order)
            else None
        )
        if name != expected:
            raise RuntimeError(
                "Jacobi timing marker sequence changed: "
                f"expected={expected!r}, observed={name!r}, "
                f"marker_index={self.next_marker}."
            )
        self._record_event(name)
        self.next_marker += 1

    def require_complete(self) -> None:
        if self.next_marker != len(self.marker_order):
            missing = self.marker_order[self.next_marker :]
            raise RuntimeError(
                "Jacobi timing trace is incomplete: "
                f"recorded={self.next_marker}, expected={len(self.marker_order)}, "
                f"missing={list(missing)}."
            )

    def mark_aborted(self) -> None:
        self.discard = True
        self._record_event(self._ABORT_COMPLETION)

    def completion_event(self):
        label = self._ABORT_COMPLETION if self.discard else self.marker_order[-1]
        return self.events[label]

    def elapsed(self, start: str, end: str) -> float:
        if start not in self.recorded or end not in self.recorded:
            raise RuntimeError(
                "Jacobi timing attempted to consume unrecorded markers: "
                f"start={start!r}, end={end!r}."
            )
        return float(self.events[start].elapsed_time(self.events[end]))


class JacobiCudaTimingProfiler:
    """Independent per-pass Jacobi CUDA timer with structured JSON logging.

    The generic DFLASH profiler continues to own the enclosing ``draft_head``
    span.  This profiler is intentionally separate because the generic sample
    builder does not consume the ``jacobi_*`` marker ABI.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        tp_rank: int,
        rounds: int,
        environ: Mapping[str, object] | None = None,
    ) -> None:
        if environ is None:
            environ = os.environ
        self.device = torch.device(device)
        self.rounds = parse_jacobi_rounds(rounds)
        custom_flag = environ.get(_JACOBI_TIMING_ENV)
        inherited_flag = environ.get("SGLANG_DFLASH_TIMING", "0")
        requested = _parse_boolean_flag(
            inherited_flag if custom_flag is None else custom_flag,
            name=(
                "SGLANG_DFLASH_TIMING"
                if custom_flag is None
                else _JACOBI_TIMING_ENV
            ),
        )
        self.warmup_steps = _parse_nonnegative_timing_int(
            "SGLANG_DFLASH_TIMING_WARMUP_STEPS",
            default=20,
            environ=environ,
        )
        self.log_interval = _parse_nonnegative_timing_int(
            "SGLANG_DFLASH_TIMING_LOG_INTERVAL",
            default=100,
            environ=environ,
        )
        if requested and self.log_interval == 0:
            raise ValueError("SGLANG_DFLASH_TIMING_LOG_INTERVAL must be positive.")
        self.enabled = bool(
            requested
            and int(tp_rank) == 0
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        )
        self._seen_steps = 0
        self._measured_steps = 0
        self._active: _JacobiCudaTrace | None = None
        self._pending: deque[_JacobiCudaTrace] = deque()
        self._pool: list[_JacobiCudaTrace] = []
        self._cumulative_sums: dict[str, float] = {}
        self._samples_since_log = 0

        if self.enabled:
            logger.info(
                "Independent Jacobi CUDA timing enabled on TP rank 0: "
                "rounds=%d warmup_steps=%d log_interval=%d schema=%s.",
                self.rounds,
                self.warmup_steps,
                self.log_interval,
                _JACOBI_TIMING_SCHEMA,
            )

    def start_step(self, *, batch_size: int) -> None:
        if not self.enabled:
            return
        self._drain_completed()
        self._seen_steps += 1
        if self._seen_steps <= self.warmup_steps:
            return
        if self._active is not None:
            raise RuntimeError("Jacobi timing start_step called with an active trace.")
        trace = self._pool.pop() if self._pool else _JacobiCudaTrace(self.device)
        trace.reset(batch_size=batch_size, rounds=self.rounds)
        self._active = trace

    def mark(self, name: str) -> None:
        if self._active is not None:
            self._active.record_marker(name)

    def finish_step(self) -> None:
        trace = self._active
        if trace is None:
            return
        try:
            trace.require_complete()
        except Exception:
            self.abort_step()
            raise
        self._active = None
        self._pending.append(trace)
        self._drain_completed()

    def abort_step(self) -> None:
        trace = self._active
        if trace is None:
            return
        self._active = None
        try:
            trace.mark_aborted()
        except Exception:
            # Preserve the original serving exception.  The trace is not
            # returned to the pool because its last in-flight event is unknown.
            logger.warning("Failed to finalize an aborted Jacobi timing trace.")
            return
        self._pending.append(trace)
        self._drain_completed()

    def _drain_completed(self) -> None:
        while self._pending:
            trace = self._pending[0]
            if not trace.completion_event().query():
                break
            self._pending.popleft()
            if not trace.discard:
                sample = _jacobi_timing_sample_from_elapsed(
                    rounds=self.rounds,
                    batch_size=trace.batch_size,
                    elapsed=trace.elapsed,
                )
                self._measured_steps += 1
                self._samples_since_log += 1
                for key, value in sample.items():
                    self._cumulative_sums[key] = (
                        self._cumulative_sums.get(key, 0.0) + value
                    )
            self._pool.append(trace)
        if self._samples_since_log >= self.log_interval:
            self._log_cumulative_json()
            self._samples_since_log %= self.log_interval

    def _mean(self, key: str) -> float:
        return self._cumulative_sums.get(key, 0.0) / max(self._measured_steps, 1)

    def _sum(self, key: str) -> float:
        return self._cumulative_sums.get(key, 0.0)

    def _log_cumulative_json(self) -> None:
        mean_ms = {
            "refinement_total": self._mean("refinement_total_ms"),
            "base_head": self._mean("base_head_ms"),
            "embedding_total": self._mean("embedding_total_ms"),
            "hsc_total": self._mean("hsc_total_ms"),
            "correction_head_total": self._mean("correction_head_total_ms"),
            "component_sum": self._mean("component_sum_ms"),
            "unattributed": self._mean("unattributed_ms"),
            "rounds": [
                {
                    "round": round_index,
                    "embed": self._mean(f"round_{round_index}_embed_ms"),
                    "hsc": self._mean(f"round_{round_index}_hsc_ms"),
                    "head": self._mean(f"round_{round_index}_head_ms"),
                    "component_sum": self._mean(
                        f"round_{round_index}_component_sum_ms"
                    ),
                }
                for round_index in range(1, self.rounds + 1)
            ],
        }
        cumulative_ms = {
            "refinement_total": self._sum("refinement_total_ms"),
            "base_head": self._sum("base_head_ms"),
            "embedding_total": self._sum("embedding_total_ms"),
            "hsc_total": self._sum("hsc_total_ms"),
            "correction_head_total": self._sum("correction_head_total_ms"),
            "component_sum": self._sum("component_sum_ms"),
            "unattributed": self._sum("unattributed_ms"),
            "rounds": [
                {
                    "round": round_index,
                    "embed": self._sum(f"round_{round_index}_embed_ms"),
                    "hsc": self._sum(f"round_{round_index}_hsc_ms"),
                    "head": self._sum(f"round_{round_index}_head_ms"),
                    "component_sum": self._sum(
                        f"round_{round_index}_component_sum_ms"
                    ),
                }
                for round_index in range(1, self.rounds + 1)
            ],
        }
        payload = {
            "schema": _JACOBI_TIMING_SCHEMA,
            "scope": "jacobi_head_refinement_only",
            "mode": "asynchronous_cuda_events",
            "rounds": self.rounds,
            "measured_steps": self._measured_steps,
            "mean_batch_size": self._mean("batch_size"),
            "pass_counts": asdict(JacobiPassCounts.for_rounds(self.rounds)),
            "mean_ms": mean_ms,
            "cumulative_ms": cumulative_ms,
        }
        logger.info(
            "%s %s",
            _JACOBI_TIMING_LOG_PREFIX,
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )


def _validate_sample_ids(
    value: torch.Tensor, *, batch_size: int, future_slots: int, label: str
) -> torch.Tensor:
    expected = int(batch_size) * int(future_slots)
    if value.ndim != 1 or int(value.numel()) != expected or value.dtype != torch.long:
        raise RuntimeError(
            f"Jacobi {label} sampler must return int64 [{expected}], got "
            f"shape={tuple(value.shape)}, dtype={value.dtype}."
        )
    return value.view(batch_size, future_slots)


@torch.no_grad()
def _jacobi_hsc_greedy(
    *,
    decoded_future_hidden: torch.Tensor,
    verified_root_ids: torch.Tensor,
    rounds: int,
    embed_tokens: Callable[[torch.Tensor], torch.Tensor],
    apply_hidden_correction: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sample_hidden: Callable[[torch.Tensor], torch.Tensor],
    mark: Callable[[str], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, JacobiPassCounts]:
    """Run fixed-round block-parallel predecessor refinement after one backbone."""

    rounds = parse_jacobi_rounds(rounds)
    if decoded_future_hidden.ndim != 3:
        raise ValueError(
            "Jacobi decoded future hidden must be [batch, slots, hidden], got "
            f"{tuple(decoded_future_hidden.shape)}."
        )
    batch_size, future_slots, hidden_size = decoded_future_hidden.shape
    if batch_size <= 0 or future_slots != 15 or hidden_size <= 0:
        raise ValueError(
            "Jacobi runtime requires positive [batch, 15, hidden], got "
            f"{tuple(decoded_future_hidden.shape)}."
        )
    if verified_root_ids.ndim != 1 or int(verified_root_ids.numel()) != batch_size:
        raise ValueError(
            "Jacobi verified roots must be [batch], got "
            f"{tuple(verified_root_ids.shape)} for batch={batch_size}."
        )
    roots = verified_root_ids.to(
        device=decoded_future_hidden.device, dtype=torch.long
    )
    base_flat = decoded_future_hidden.reshape(batch_size * future_slots, hidden_size)

    def record(label: str) -> None:
        if mark is not None:
            mark(label)

    record("jacobi_base_head_start")
    proposal_ids = _validate_sample_ids(
        sample_hidden(base_flat),
        batch_size=batch_size,
        future_slots=future_slots,
        label="base-head",
    )
    record("jacobi_base_head_end")

    corrected = decoded_future_hidden
    for round_index in range(1, rounds + 1):
        previous_ids = torch.cat((roots[:, None], proposal_ids[:, :-1]), dim=1)
        record(f"jacobi_embed_start_{round_index}")
        previous_embeddings = embed_tokens(previous_ids).detach()
        record(f"jacobi_embed_end_{round_index}")
        if previous_embeddings.ndim != 3 or tuple(previous_embeddings.shape[:2]) != (
            int(batch_size),
            int(future_slots),
        ):
            raise RuntimeError(
                "Jacobi target embedding returned wrong leading shape: expected "
                f"{(int(batch_size), int(future_slots))}, got "
                f"{tuple(previous_embeddings.shape)}."
            )

        record(f"jacobi_hsc_start_{round_index}")
        corrected_flat = apply_hidden_correction(
            base_flat,
            previous_embeddings.reshape(
                batch_size * future_slots, previous_embeddings.shape[-1]
            ),
        )
        record(f"jacobi_hsc_end_{round_index}")
        if tuple(corrected_flat.shape) != tuple(base_flat.shape):
            raise RuntimeError(
                "Jacobi HSC returned wrong flattened shape: expected "
                f"{tuple(base_flat.shape)}, got {tuple(corrected_flat.shape)}."
            )
        corrected = corrected_flat.view(batch_size, future_slots, hidden_size)

        record(f"jacobi_head_start_{round_index}")
        proposal_ids = _validate_sample_ids(
            sample_hidden(corrected_flat),
            batch_size=batch_size,
            future_slots=future_slots,
            label=f"round-{round_index}",
        )
        record(f"jacobi_head_end_{round_index}")

    return proposal_ids, corrected, JacobiPassCounts.for_rounds(rounds)


class DFlowDFlyFullJacobiWorker(DFlowDFlyFullWorker):
    """Efficacy-gated optional Jacobi HSC worker for TP=2 greedy DFLASH."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._jacobi_rounds = parse_jacobi_rounds(
            os.environ.get(_JACOBI_ROUNDS_ENV, "")
        )
        if int(get_tp_group().world_size) != 2:
            raise ValueError(
                "Full DFlow/DFly Jacobi runtime is a fail-closed TP=2 experiment; "
                f"got TP={int(get_tp_group().world_size)}."
            )
        self._jacobi_gate = validate_jacobi_gate_from_env(
            draft_model_path=self.server_args.speculative_draft_model_path,
            # The spawn parent already hashes both referenced artifacts before
            # allocating model processes.  Rechecking multi-GB model payloads
            # independently on every TP rank would add startup I/O but no new
            # trust boundary; ranks still revalidate the gate file itself.
            verify_referenced_files=False,
        )
        self._jacobi_pass_counts = JacobiPassCounts.for_rounds(
            self._jacobi_rounds
        )
        self._jacobi_timing = JacobiCudaTimingProfiler(
            device=self.device,
            tp_rank=self.tp_rank,
            rounds=self._jacobi_rounds,
        )
        self._jacobi_runtime_invocations = 0
        if self.tp_rank == 0:
            logger.info(
                "Optional Stage-J Jacobi runtime authorized: rounds=%d "
                "backbone=%d head=%d (base=%d correction=%d) embed=%d hsc=%d "
                "expected_tp2_collectives=%d gate_evaluation_sha256=%s",
                self._jacobi_rounds,
                self._jacobi_pass_counts.backbone_forwards,
                self._jacobi_pass_counts.head_passes,
                self._jacobi_pass_counts.base_head_passes,
                self._jacobi_pass_counts.correction_head_passes,
                self._jacobi_pass_counts.embedding_passes,
                self._jacobi_pass_counts.hidden_correction_passes,
                self._jacobi_pass_counts.expected_tp2_collectives,
                self._jacobi_gate["evaluation_artifact_sha256"],
            )

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ) -> None:
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return
        if batch.has_grammar:
            raise RuntimeError(
                "Full DFlow/DFly Jacobi received a grammar-constrained batch; "
                "the scheduler must reject it for DFLASH."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            if (
                not is_dflash_sampling_verify_available()
                and not self._warned_sampling_fallback
                and self.tp_rank == 0
            ):
                logger.warning(
                    "Full DFlow/DFly Jacobi non-greedy verification is unavailable; "
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
                decoded_hidden[:, 1:, :], label="jacobi_decoded_backbone_future_hidden"
            )

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        apply_correction = self.draft_model.apply_hidden_correction
        self._jacobi_timing.start_step(batch_size=batch_size)
        self._timing.mark("draft_head_start")
        try:
            draft_next, corrected_hidden, observed_counts = _jacobi_hsc_greedy(
                decoded_future_hidden=decoded_hidden[:, 1:, :],
                verified_root_ids=block_ids[:, 0],
                rounds=self._jacobi_rounds,
                embed_tokens=embed_module,
                apply_hidden_correction=apply_correction,
                sample_hidden=(
                    lambda hidden: self._greedy_sample_from_vocab_parallel_head(
                        hidden_states=hidden, lm_head=lm_head
                    )
                ),
                mark=self._jacobi_timing.mark,
            )
        except Exception:
            self._jacobi_timing.abort_step()
            raise
        self._timing.mark("draft_head_end")
        self._jacobi_timing.finish_step()
        if observed_counts != self._jacobi_pass_counts:
            raise RuntimeError(
                "Jacobi pass accounting changed at runtime: expected "
                f"{self._jacobi_pass_counts}, got {observed_counts}."
            )
        self._jacobi_runtime_invocations += 1
        if parity_active:
            _assert_tp_replicated_bitwise(
                corrected_hidden, label="jacobi_predecessor_corrected_future_hidden"
            )
            _assert_tp_replicated_bitwise(
                draft_next, label="jacobi_sampled_future_ids"
            )
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


__all__ = [
    "DFlowDFlyFullJacobiWorker",
    "JacobiCudaTimingProfiler",
    "JacobiPassCounts",
    "_jacobi_hsc_greedy",
    "_jacobi_timing_sample_from_elapsed",
    "jacobi_timing_marker_order",
    "parse_jacobi_rounds",
    "validate_jacobi_gate_from_env",
    "validate_stage_j500_efficacy_gate",
]
