"""Isolated, opt-in rollout tracing for pure one-step DFlow.

This module is intentionally additive.  It subclasses the ordinary DFlash
worker and verifier without changing their decoding or acceptance semantics.
Rank zero of the tensor-parallel group writes one immutable ``.pt`` payload per
selected verify cycle; the request trace receives only the resulting path and
SHA256.

The pilot stores the exact post-anchor-overwrite BF16 latent instead of an RNG
state.  Proposal ``j`` is aligned with target-logit row ``j``; verify row 15 is
the full-accept bonus row and is deliberately excluded from teacher logits.
Hard targets are computed from the original verifier logits in float32 before
the separate BF16 replay-logit copy is made.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dflash_info import (
    JOINT_CYCLE_TRACE_KEY,
    DFlashDraftInput,
    DFlashVerifyInput,
)
from sglang.srt.speculative.dflash_worker import DFlashWorker


TRACE_DIR_ENV = "SGLANG_DFLOW_OPD_TRACE_DIR"
TRACE_MAX_CYCLES_ENV = "SGLANG_DFLOW_OPD_TRACE_MAX_CYCLES"
TRACE_MAX_PER_REQUEST_ENV = "SGLANG_DFLOW_OPD_TRACE_MAX_CYCLES_PER_REQUEST"
TRACE_POLICY_SHA_ENV = "SGLANG_DFLOW_OPD_TRACE_POLICY_SHA256"
TRACE_TARGET_SHA_ENV = "SGLANG_DFLOW_OPD_TRACE_TARGET_SHA256"
TRACE_RID_ALLOWLIST_FILE_ENV = "SGLANG_DFLOW_OPD_TRACE_RID_ALLOWLIST_FILE"

TRACE_SCHEMA = "sglang-pure-dflow-opd-cycle-v2"
TRACE_RECORD_KEY = "dflow_opd_trace_record"
TRACE_BLOCK_SIZE = 16
TRACE_PROPOSALS = TRACE_BLOCK_SIZE - 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _required_positive_int(environ: Mapping[str, str], name: str) -> int:
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        raise ValueError(f"{name} is required to opt in to DFlow OPD tracing.")
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _required_sha256(environ: Mapping[str, str], name: str) -> str:
    value = str(environ.get(name, "")).strip().lower()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be an explicit lowercase SHA256 hex digest.")
    return value


def _load_rid_allowlist(path: Path) -> tuple[frozenset[str], str]:
    if not path.is_absolute():
        raise ValueError(f"{TRACE_RID_ALLOWLIST_FILE_ENV} must be an absolute path.")
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            f"{TRACE_RID_ALLOWLIST_FILE_ENV} must be a regular non-symlink file."
        )
    raw_bytes = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            value = json.loads(raw_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid RID allowlist JSON {path}: {exc}") from exc
        if not isinstance(value, list):
            raise ValueError("RID allowlist JSON must be an array of strings.")
        values = value
    elif suffix == ".csv":
        try:
            rows = list(csv.reader(raw_bytes.decode("utf-8").splitlines()))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ValueError(f"Invalid RID allowlist CSV {path}: {exc}") from exc
        if not rows or rows[0] != ["request_id"]:
            raise ValueError("RID allowlist CSV must have only a request_id column.")
        if any(len(row) != 1 for row in rows[1:]):
            raise ValueError("RID allowlist CSV rows must contain exactly one field.")
        values = [row[0] for row in rows[1:]]
    else:
        raise ValueError("RID allowlist path must end in .json or .csv.")
    if not values:
        raise ValueError("RID allowlist must contain at least one request id.")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in values
    ):
        raise ValueError("Every RID allowlist value must be a non-empty trimmed string.")
    if len(set(values)) != len(values):
        raise ValueError("RID allowlist values must be unique.")
    return frozenset(values), hashlib.sha256(raw_bytes).hexdigest()


@dataclass(frozen=True)
class DFlowOPDTraceConfig:
    trace_dir: Path
    max_cycles: int
    max_cycles_per_request: int
    policy_sha256: str
    target_sha256: str
    rid_allowlist_file: Path
    rid_allowlist: frozenset[str]
    rid_allowlist_file_sha256: str

    @classmethod
    def from_environ(
        cls, environ: Mapping[str, str] | None = None
    ) -> "DFlowOPDTraceConfig":
        environ = os.environ if environ is None else environ
        raw_dir = str(environ.get(TRACE_DIR_ENV, "")).strip()
        if not raw_dir:
            raise ValueError(f"{TRACE_DIR_ENV} is required to opt in to tracing.")
        trace_dir = Path(raw_dir)
        if not trace_dir.is_absolute():
            raise ValueError(f"{TRACE_DIR_ENV} must be an absolute path.")
        raw_allowlist = str(environ.get(TRACE_RID_ALLOWLIST_FILE_ENV, "")).strip()
        if not raw_allowlist:
            raise ValueError(
                f"{TRACE_RID_ALLOWLIST_FILE_ENV} is required to opt in to tracing."
            )
        allowlist_file = Path(raw_allowlist)
        rid_allowlist, allowlist_sha256 = _load_rid_allowlist(allowlist_file)
        return cls(
            trace_dir=trace_dir,
            max_cycles=_required_positive_int(environ, TRACE_MAX_CYCLES_ENV),
            max_cycles_per_request=_required_positive_int(
                environ, TRACE_MAX_PER_REQUEST_ENV
            ),
            policy_sha256=_required_sha256(environ, TRACE_POLICY_SHA_ENV),
            target_sha256=_required_sha256(environ, TRACE_TARGET_SHA_ENV),
            rid_allowlist_file=allowlist_file,
            rid_allowlist=rid_allowlist,
            rid_allowlist_file_sha256=allowlist_sha256,
        )


@dataclass(frozen=True)
class DFlowOPDTraceReservation:
    global_cycle_index: int
    request_cycle_index: int
    request_id: str


def _request_id(req: Any) -> str:
    rid = str(getattr(req, "rid", "")).strip()
    if not rid:
        raise RuntimeError("DFlow OPD tracing requires every request to have rid.")
    return rid


def _prefix_token_ids_before_cycle(req: Any) -> tuple[list[int], int, int]:
    prompt = [int(token_id) for token_id in getattr(req, "origin_input_ids", [])]
    output = [int(token_id) for token_id in getattr(req, "output_ids", [])]
    prefix = prompt + output
    if not prefix:
        raise RuntimeError("DFlow OPD cycle has an empty token prefix.")
    return prefix, len(prompt), len(output)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AtomicDFlowOPDTraceWriter:
    """Bounded write-once payload publisher owned by TP rank zero."""

    def __init__(self, config: DFlowOPDTraceConfig, *, abi: Mapping[str, Any]):
        self.config = config
        self.abi = dict(abi)
        parent = config.trace_dir.parent
        if not parent.is_dir():
            raise ValueError(f"Trace directory parent does not exist: {parent}.")
        if config.trace_dir.exists() or config.trace_dir.is_symlink():
            raise FileExistsError(
                "DFlow OPD trace directory must be newly created and must not "
                f"already exist: {config.trace_dir}."
            )
        config.trace_dir.mkdir(mode=0o700)
        self._global_reserved = 0
        self._request_reserved: dict[str, int] = {}

    def reserve_batch(
        self, reqs: Sequence[Any]
    ) -> list[DFlowOPDTraceReservation | None]:
        reservations: list[DFlowOPDTraceReservation | None] = []
        for req in reqs:
            rid = _request_id(req)
            if rid not in self.config.rid_allowlist:
                raise RuntimeError(
                    f"Request rid={rid!r} is not in the frozen OPD trace allowlist."
                )
            request_count = self._request_reserved.get(rid, 0)
            if (
                self._global_reserved >= self.config.max_cycles
                or request_count >= self.config.max_cycles_per_request
            ):
                reservations.append(None)
                continue
            reservation = DFlowOPDTraceReservation(
                global_cycle_index=self._global_reserved,
                request_cycle_index=request_count,
                request_id=rid,
            )
            self._global_reserved += 1
            self._request_reserved[rid] = request_count + 1
            reservations.append(reservation)
        return reservations

    def write_payload(
        self,
        reservation: DFlowOPDTraceReservation,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        rid_digest = hashlib.sha256(reservation.request_id.encode("utf-8")).hexdigest()[
            :16
        ]
        filename = (
            f"cycle-{reservation.global_cycle_index:08d}-"
            f"{rid_digest}-{reservation.request_cycle_index:06d}.pt"
        )
        final_path = self.config.trace_dir / filename
        temp_path = self.config.trace_dir / f".{filename}.tmp-{os.getpid()}"
        if final_path.exists() or temp_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite OPD trace payload {final_path}."
            )

        try:
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                torch.save(dict(payload), handle)
                handle.flush()
                os.fsync(handle.fileno())
            raw_file_sha256 = _sha256_file(temp_path)
            # A hard link publishes the complete inode atomically and fails if
            # the destination exists.  Unlike os.replace(), it never overwrites.
            os.link(temp_path, final_path)
            temp_path.unlink()
            dir_fd = os.open(self.config.trace_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return {
            "schema": TRACE_SCHEMA,
            "path": str(final_path),
            "raw_file_sha256": raw_file_sha256,
            "global_cycle_index": reservation.global_cycle_index,
            "request_cycle_index": reservation.request_cycle_index,
        }


def _proposal_aligned_target_rows(
    *,
    next_token_logits: torch.Tensor,
    draft_token: torch.Tensor,
    batch_size: int,
    block_size: int = TRACE_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return proposal ids and their exactly aligned verifier-logit rows.

    DFlash accepts while ``candidates[:, 1:] == target_predict[:, :-1]``.
    Therefore proposal index ``j`` uses verifier row ``j``.  The last verifier
    row predicts the correction/bonus and is not a proposal-policy target.
    """

    if int(block_size) != TRACE_BLOCK_SIZE:
        raise ValueError(f"OPD tracing requires block_size={TRACE_BLOCK_SIZE}.")
    if next_token_logits.ndim != 2:
        raise ValueError("Verifier logits must have shape [batch*block, vocab].")
    expected_rows = int(batch_size) * int(block_size)
    if int(next_token_logits.shape[0]) != expected_rows:
        raise ValueError(
            "Verifier logits row count is not batch*block: "
            f"got {next_token_logits.shape[0]}, expected {expected_rows}."
        )
    if draft_token.numel() != expected_rows:
        raise ValueError("Draft token count is not batch*block.")
    candidates = draft_token.view(batch_size, block_size)
    proposal_ids = candidates[:, 1:].to(torch.long)
    vocab_size = int(next_token_logits.shape[-1])
    if bool(((proposal_ids < 0) | (proposal_ids >= vocab_size)).any().item()):
        raise ValueError("A proposal token id is outside the target-logit vocabulary.")
    target_rows = next_token_logits.view(batch_size, block_size, vocab_size)[
        :, : block_size - 1, :
    ]
    if target_rows.shape[:2] != proposal_ids.shape:
        raise RuntimeError("Proposal/logit row alignment invariant failed.")
    return proposal_ids, target_rows


def _validate_greedy_trace_batch(batch: ScheduleBatch) -> None:
    if batch.has_grammar:
        raise RuntimeError("Pure-DFlow OPD tracing forbids grammar constraints.")
    if getattr(batch, "return_logprob", False):
        raise RuntimeError("Pure-DFlow OPD tracing forbids return_logprob requests.")
    sampling_info = batch.sampling_info
    if sampling_info is not None:
        if not sampling_info.is_all_greedy:
            raise RuntimeError("Pure-DFlow OPD tracing supports greedy requests only.")
        if bool(getattr(sampling_info, "has_custom_logit_processor", False)):
            raise RuntimeError(
                "Pure-DFlow OPD tracing forbids custom logit processors."
            )
        if getattr(sampling_info, "logit_bias", None) is not None:
            raise RuntimeError("Pure-DFlow OPD tracing forbids logit bias.")
        penalizer = getattr(sampling_info, "penalizer_orchestrator", None)
        if penalizer is not None and bool(getattr(penalizer, "is_required", False)):
            raise RuntimeError("Pure-DFlow OPD tracing forbids sampling penalties.")


def _worker_contract_violations(worker: Any) -> list[str]:
    violations: list[str] = []
    if not bool(getattr(worker, "_is_dflow_model", False)):
        violations.append("draft model must be a DFlow model")
    if not bool(getattr(worker, "_dflow_one_step_direct", False)):
        violations.append("draft checkpoint must declare one_step_direct")
    if bool(getattr(worker.draft_model, "is_dflow_domino_model", False)):
        violations.append("Domino is forbidden; this collector is pure-DFlow only")
    if str(getattr(worker, "flow_mode", "")) != "one_step":
        violations.append("flow_mode must be one_step")
    if int(getattr(worker, "flow_steps", -1)) != 1:
        violations.append("flow_steps must be 1")
    if int(getattr(worker, "block_size", -1)) != TRACE_BLOCK_SIZE:
        violations.append(f"block_size must be {TRACE_BLOCK_SIZE}")
    if int(getattr(worker, "flow_noise_samples", -1)) != 1:
        violations.append("flow_noise_samples must be 1")
    if str(getattr(worker, "flow_noise_aggregation", "")) != "latent_mean":
        violations.append("flow_noise_aggregation must be latent_mean")
    if float(getattr(worker, "flow_noise_scale", float("nan"))) != 2.0:
        violations.append("flow_noise_scale must be exactly 2.0")
    if bool(getattr(worker, "use_compact_draft_cache", False)):
        violations.append("compact draft cache is unsupported")
    if bool(getattr(worker, "active_judge_enabled", False)):
        violations.append("active multi-block judge is unsupported")
    if getattr(worker, "_training_proposal_topk", None) is not None:
        violations.append("categorical proposal sampling is unsupported")
    if getattr(worker, "_collaborative_gate_mode", None) is not None:
        violations.append("collaborative source mode is unsupported")
    if bool(getattr(worker, "_trace_draft_hidden_enabled", False)):
        violations.append("legacy draft-hidden tracing must be disabled")
    if not bool(getattr(worker, "_dflow_strict_one_backbone_enabled", False)):
        violations.append("strict one-backbone enforcement is required")
    server_args = getattr(worker, "server_args", None)
    if str(getattr(server_args, "speculative_algorithm", "")).upper() != "DFLASH":
        violations.append("speculative_algorithm must be DFLASH")
    if int(getattr(server_args, "dp_size", 1)) != 1:
        violations.append("data parallel size must be 1 for a global trace cap")
    if getattr(server_args, "disable_overlap_schedule", None) is not True:
        violations.append("disable_overlap_schedule must be true")
    return violations


def _normalization_sha256(draft_model: Any) -> str:
    """Hash the loaded persistent normalization buffers canonically.

    Each tensor contributes an 8-byte little-endian metadata length, canonical
    JSON metadata (name/dtype/shape), an 8-byte raw-data length, then contiguous
    raw bytes.  Hashing the loaded buffers, rather than a source stats file,
    makes the serving and replay checks independent of checkpoint packaging.
    """

    digest = hashlib.sha256()
    for name in ("latent_mean", "latent_std"):
        value = getattr(draft_model, name, None)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"DFlow draft model is missing persistent {name}.")
        tensor = value.detach().to(device="cpu").contiguous()
        if tensor.dtype != torch.float32:
            raise ValueError(f"DFlow normalization buffer {name} must be float32.")
        metadata = json.dumps(
            {
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "name": name,
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        # reshape(-1) makes scalar normalization buffers byte-viewable while
        # metadata above preserves their original rank/shape.
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "little", signed=False))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "little", signed=False))
        digest.update(raw)
    return digest.hexdigest()


def _runtime_abi(worker: Any) -> dict[str, Any]:
    return {
        "noise": {
            "distribution": "normal",
            "scale": 2.0,
            "samples": 1,
            "aggregation": "none",
            "full_tail_gaussian": True,
            "block_size": TRACE_BLOCK_SIZE,
            "dtype": "bfloat16",
            "slot0": "verified_root_included",
        },
        "attention": "prefix_lt_anchor_plus_bidirectional_same_block",
        "normalization": {
            "kind": "affine",
            "formula": "(x-mean)/std",
            "stats_dtype": "float32",
        },
        "normalization_sha256": _normalization_sha256(worker.draft_model),
    }


class DFlowOPDTraceVerifyInput(DFlashVerifyInput):
    """Ordinary DFlash verification plus bounded immutable OPD capture."""

    def verify(self, *, batch, logits_output, page_size):
        reservations = self._opd_reservations
        selected = [
            index for index, value in enumerate(reservations) if value is not None
        ]
        captured: dict[int, dict[str, Any]] = {}
        if selected:
            batch_size = batch.batch_size()
            proposal_ids, target_rows = _proposal_aligned_target_rows(
                next_token_logits=logits_output.next_token_logits,
                draft_token=self.draft_token,
                batch_size=batch_size,
                block_size=self.draft_token_num,
            )
            positions = self.positions.view(batch_size, self.draft_token_num)
            candidates = self.draft_token.view(batch_size, self.draft_token_num)
            for index in selected:
                prefix = self._opd_prefixes[index]
                root_id = int(candidates[index, 0].item())
                if int(prefix["token_ids"][-1]) != root_id:
                    raise RuntimeError(
                        "Stored pre-cycle prefix does not end in the verified root."
                    )
                latent = self._opd_latents[index]
                if latent.dtype != torch.bfloat16:
                    raise RuntimeError("Exact DFlow OPD latent must remain BF16.")
                # Do not recover hard targets from the persisted BF16 logits:
                # low-margin float32 rows can change argmax after quantization.
                # This is the exact greedy operation used by the base verifier.
                original_rows = target_rows[index].detach()
                teacher_top1_ids = original_rows.float().argmax(dim=-1).to(
                    device="cpu", dtype=torch.long
                )
                full_accept_bonus_id = int(
                    logits_output.next_token_logits.view(
                        batch_size, self.draft_token_num, -1
                    )[index, -1]
                    .detach()
                    .float()
                    .argmax()
                    .item()
                )
                captured[index] = {
                    "proposal_ids": proposal_ids[index].detach().to("cpu", torch.long),
                    "target_logits": original_rows.detach().to(
                        device="cpu", dtype=torch.bfloat16
                    ),
                    "teacher_top1_ids": teacher_top1_ids,
                    "full_accept_bonus_id": full_accept_bonus_id,
                    "positions": positions[index].detach().to("cpu", torch.long),
                    "verified_root_id": root_id,
                    "latent_z": latent,
                    "prefix": prefix,
                }

        result = super().verify(
            batch=batch, logits_output=logits_output, page_size=page_size
        )
        accepted_per_req = result[3]
        if len(accepted_per_req) != batch.batch_size():
            raise RuntimeError("DFlow OPD acceptance result is not batch-aligned.")

        for index in selected:
            req = batch.reqs[index]
            trace = getattr(req, "customized_info", None)
            cycle_trace = (
                trace.get(JOINT_CYCLE_TRACE_KEY) if isinstance(trace, dict) else None
            )
            if not isinstance(cycle_trace, list) or not cycle_trace:
                raise RuntimeError("DFlow verifier did not publish its cycle trace.")
            cycle_entry = cycle_trace[-1]
            reservation = reservations[index]
            state = captured[index]
            if int(cycle_entry.get("cycle_id", -1)) != reservation.request_cycle_index:
                raise RuntimeError(
                    "Base verifier cycle id is not aligned with the OPD reservation."
                )
            if list(cycle_entry.get("proposal_ids", ())) != state[
                "proposal_ids"
            ].tolist():
                raise RuntimeError(
                    "Base verifier proposal ids are not aligned with the captured block."
                )
            raw_accepted = int(cycle_entry["verifier_accepted_length"])
            if raw_accepted < 0 or raw_accepted > TRACE_PROPOSALS:
                raise RuntimeError(
                    f"Invalid verifier accepted length in cycle trace: {raw_accepted}."
                )
            correction_or_bonus_id = cycle_entry.get("correction_or_bonus_id")
            if isinstance(correction_or_bonus_id, bool) or not isinstance(
                correction_or_bonus_id, int
            ):
                raise RuntimeError(
                    "DFlow OPD tracing requires a real correction/bonus token id."
                )
            expected_verifier_id = (
                int(state["teacher_top1_ids"][raw_accepted].item())
                if raw_accepted < TRACE_PROPOSALS
                else state["full_accept_bonus_id"]
            )
            if correction_or_bonus_id != expected_verifier_id:
                kind = "correction" if raw_accepted < TRACE_PROPOSALS else "bonus"
                raise RuntimeError(
                    f"Cycle trace {kind} token {correction_or_bonus_id} does not "
                    f"equal the original-logit greedy top-1 {expected_verifier_id}."
                )
            committed_accepted = int(accepted_per_req[index])
            if int(cycle_entry.get("accepted_length", -1)) != committed_accepted:
                raise RuntimeError(
                    "Base verifier committed acceptance is not result-aligned."
                )
            expected_rejection = (
                raw_accepted if raw_accepted < TRACE_PROPOSALS else None
            )
            if cycle_entry.get("first_rejection") != expected_rejection:
                raise RuntimeError(
                    "Base verifier first-rejection boundary is inconsistent."
                )
            payload = {
                # Stable cross-repository replay contract.  Keep these names in
                # lockstep with SpecForge's dflow_on_policy_replay loader.
                "schema_version": TRACE_SCHEMA,
                "policy_sha256": self._opd_writer.config.policy_sha256,
                "target_sha256": self._opd_writer.config.target_sha256,
                "abi": dict(self._opd_writer.abi),
                "request_id": reservation.request_id,
                "cycle_id": reservation.request_cycle_index,
                "context_token_ids": torch.tensor(
                    state["prefix"]["token_ids"][:-1], dtype=torch.long
                ),
                "anchor_offset": len(state["prefix"]["token_ids"]) - 1,
                "verified_root_id": state["verified_root_id"],
                "proposal_ids": state["proposal_ids"],
                "accepted_length": raw_accepted,
                "exact_latent": state["latent_z"],
                "teacher_logits": state["target_logits"],
                "teacher_top1_ids": state["teacher_top1_ids"],
                "correction_or_bonus_id": correction_or_bonus_id,
                # Additional provenance/audit fields do not change the replay
                # contract above.
                "policy_version": self.policy_version,
                "global_cycle_index": reservation.global_cycle_index,
                "request_cycle_index": reservation.request_cycle_index,
                "prompt_length": state["prefix"]["prompt_length"],
                "output_length_before_cycle": state["prefix"]["output_length"],
                "target_logits_source_dtype": str(
                    logits_output.next_token_logits.dtype
                ).removeprefix("torch."),
                "committed_accepted_length": committed_accepted,
                "first_rejection": expected_rejection,
                "positions": state["positions"],
            }
            record = self._opd_writer.write_payload(reservation, payload)
            if TRACE_RECORD_KEY in cycle_entry:
                raise RuntimeError("DFlow OPD trace record was already attached.")
            cycle_entry[TRACE_RECORD_KEY] = record
        return result


class DFlowOPDTraceWorker(DFlashWorker):
    """Pure one-step DFlow worker with exact, bounded TP-rank-zero traces."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        violations = _worker_contract_violations(self)
        if violations:
            raise ValueError(
                "Pure-DFlow OPD trace worker contract failed: " + "; ".join(violations)
            )
        config = DFlowOPDTraceConfig.from_environ()
        self._opd_trace_writer = (
            AtomicDFlowOPDTraceWriter(config, abi=_runtime_abi(self))
            if int(self.tp_rank) == 0
            else None
        )
        self._opd_pending_reservations = None
        self._opd_pending_latents = None

    def _make_dflow_initial_latent(self, *, bs: int, input_embeds: torch.Tensor):
        z, anchor_norm = super()._make_dflow_initial_latent(
            bs=bs, input_embeds=input_embeds
        )
        reservations = self._opd_pending_reservations
        if reservations is not None and any(
            value is not None for value in reservations
        ):
            latent = z.view(int(bs), TRACE_BLOCK_SIZE, -1)
            if latent.dtype != torch.bfloat16:
                raise RuntimeError(
                    "Pure-DFlow OPD pilot requires the final serving latent to be BF16."
                )
            self._opd_pending_latents = [
                (
                    latent[index].detach().to(device="cpu", copy=True)
                    if reservation is not None
                    else None
                )
                for index, reservation in enumerate(reservations)
            ]
        return z, anchor_norm

    @torch.no_grad()
    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return super()._prepare_for_speculative_decoding(batch, draft_input)
        _validate_greedy_trace_batch(batch)

        writer = self._opd_trace_writer
        reservations = (
            writer.reserve_batch(batch.reqs)
            if writer is not None
            else [None] * batch.batch_size()
        )
        prefixes = []
        for req, reservation in zip(batch.reqs, reservations):
            if reservation is None:
                prefixes.append(None)
                continue
            token_ids, prompt_length, output_length = _prefix_token_ids_before_cycle(
                req
            )
            prefixes.append(
                {
                    "token_ids": token_ids,
                    "prompt_length": prompt_length,
                    "output_length": output_length,
                }
            )
        self._opd_pending_reservations = reservations
        self._opd_pending_latents = None
        try:
            super()._prepare_for_speculative_decoding(batch, draft_input)
            verify_input = batch.spec_info
            if not isinstance(verify_input, DFlashVerifyInput):
                raise RuntimeError(
                    "DFlow OPD prepare did not produce DFlashVerifyInput."
                )
            if int(verify_input.draft_token_num) != TRACE_BLOCK_SIZE:
                raise RuntimeError(
                    "DFlow OPD verify width changed after proposal prepare."
                )
            if verify_input.proposal_behavior != "deterministic_greedy":
                raise RuntimeError("DFlow OPD proposal behavior is not greedy.")
            if verify_input.proposal_behavior_logprobs is not None:
                raise RuntimeError(
                    "Greedy DFlow OPD proposals unexpectedly have logprobs."
                )
            if any(value is not None for value in reservations):
                if self._opd_pending_latents is None:
                    raise RuntimeError(
                        "DFlow OPD proposal forward did not expose latent z."
                    )
                for index, reservation in enumerate(reservations):
                    if (
                        reservation is not None
                        and self._opd_pending_latents[index] is None
                    ):
                        raise RuntimeError("DFlow OPD selected cycle has no latent z.")

            # DFlashVerifyInput is an ordinary Python dataclass without slots.
            # Switching to a no-field subclass preserves every allocation and
            # attention field prepared by the parent while overriding verify().
            verify_input.__class__ = DFlowOPDTraceVerifyInput
            verify_input._opd_writer = writer
            verify_input._opd_reservations = reservations
            verify_input._opd_prefixes = prefixes
            verify_input._opd_latents = self._opd_pending_latents
        finally:
            self._opd_pending_reservations = None
            self._opd_pending_latents = None


__all__ = [
    "AtomicDFlowOPDTraceWriter",
    "DFlowOPDTraceConfig",
    "DFlowOPDTraceReservation",
    "DFlowOPDTraceVerifyInput",
    "DFlowOPDTraceWorker",
    "TRACE_BLOCK_SIZE",
    "TRACE_PROPOSALS",
    "TRACE_RECORD_KEY",
    "TRACE_SCHEMA",
    "_normalization_sha256",
]
