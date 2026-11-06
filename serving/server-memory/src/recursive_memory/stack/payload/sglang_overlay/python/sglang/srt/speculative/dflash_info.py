from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import random
from typing import Any, List, Sequence, Tuple

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_accept_len_and_bonus,
    compute_dflash_sampling_accept_len_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func


JOINT_CYCLE_TRACE_KEY = "joint_cycle_trace"
_DRAFT_HIDDEN_RESERVOIR_CAPACITY = 32
_DRAFT_HIDDEN_SAMPLING_SCHEME = "uniform_cycle_reservoir_ht_v1"


def _clear_cycle_draft_hidden(cycle_trace: List[dict], cycle_id: int) -> None:
    entry = cycle_trace[cycle_id]
    entry["draft_hidden"] = None
    entry["proposal_behavior_support_ids"] = None
    entry["draft_hidden_dtype"] = None
    entry["draft_hidden_capture"] = "reservoir_not_selected"
    entry["draft_hidden_selected"] = False
    entry["draft_hidden_inclusion_probability"] = None
    entry["draft_hidden_ht_weight"] = None
    entry["draft_hidden_population_size"] = None
    entry["draft_hidden_reservoir_capacity"] = _DRAFT_HIDDEN_RESERVOIR_CAPACITY


def _retained_behavior_support_ids(
    *,
    proposal_ids: Sequence[int],
    proposal_behavior: str,
    proposal_behavior_topk: int | None,
    proposal_hidden: Any | None,
    proposal_behavior_support_ids: (
        torch.Tensor | Sequence[Sequence[int]] | None
    ),
) -> torch.Tensor | None:
    """Validate and compact the old-policy support needed by PPO replay.

    Support rows are retained iff their aligned hidden rows are retained.  Each
    categorical row must be the exact, duplicate-free behavior top-k set and
    must contain its sampled action exactly once.
    """

    if proposal_hidden is None:
        if proposal_behavior_support_ids is not None:
            raise ValueError(
                "proposal_behavior_support_ids may only accompany retained "
                "proposal_hidden rows."
            )
        return None

    behavior = str(proposal_behavior).strip().lower()
    if behavior != "categorical_topk":
        if proposal_behavior_support_ids is not None:
            raise ValueError(
                "Only categorical_topk proposals may carry behavior support ids."
            )
        return None
    if (
        isinstance(proposal_behavior_topk, bool)
        or not isinstance(proposal_behavior_topk, Integral)
        or int(proposal_behavior_topk) < 2
    ):
        raise ValueError(
            "categorical_topk proposal support requires integer topk >= 2."
        )
    if proposal_behavior_support_ids is None:
        raise ValueError(
            "categorical_topk proposals with retained hidden rows require the "
            "exact behavior support ids."
        )

    if torch.is_tensor(proposal_behavior_support_ids):
        if (
            proposal_behavior_support_ids.dtype == torch.bool
            or proposal_behavior_support_ids.is_floating_point()
            or proposal_behavior_support_ids.is_complex()
        ):
            raise ValueError(
                "proposal_behavior_support_ids tensor must have integer dtype."
            )
        support = proposal_behavior_support_ids.detach().to(
            device="cpu",
            dtype=torch.long,
            copy=True,
        )
    else:
        try:
            support = torch.tensor(
                proposal_behavior_support_ids,
                dtype=torch.long,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "proposal_behavior_support_ids must be a rectangular integer "
                "matrix."
            ) from exc

    proposal_count = len(proposal_ids)
    expected_shape = (proposal_count, int(proposal_behavior_topk))
    if support.ndim != 2 or tuple(support.shape) != expected_shape:
        raise ValueError(
            "proposal_behavior_support_ids must align with proposals and topk: "
            f"expected={expected_shape}, got={tuple(support.shape)}."
        )
    if bool((support < 0).any()):
        raise ValueError("proposal behavior support ids must be non-negative.")
    if int(support.shape[1]) > 1:
        sorted_support = support.sort(dim=1).values
        if bool((sorted_support[:, 1:] == sorted_support[:, :-1]).any()):
            raise ValueError(
                "Each proposal behavior support row must contain unique token ids."
            )
    actions = torch.tensor(proposal_ids, dtype=torch.long).reshape(-1, 1)
    if not bool(((support == actions).sum(dim=1) == 1).all()):
        raise ValueError(
            "Each proposal behavior support row must contain its sampled action "
            "exactly once."
        )
    return support.contiguous()


def _draft_hidden_reservoir_rng(req: Any) -> random.Random:
    """Return one request-local RNG for standard uniform reservoir sampling."""

    rng = getattr(req, "_joint_trace_hidden_reservoir_rng", None)
    if rng is not None:
        return rng
    # Keep the realized reservoir reproducible from trace metadata without
    # coupling it to model-sampling RNG streams or TP collectives.
    seed = random.SystemRandom().getrandbits(64)
    rng = random.Random(seed)
    req._joint_trace_hidden_reservoir_seed = seed
    req._joint_trace_hidden_reservoir_rng = rng
    return rng


def _update_retained_hidden_sampling_metadata(
    *,
    req: Any,
    cycle_trace: List[dict],
    population_size: int,
) -> None:
    retained_ids = getattr(req, "_joint_trace_hidden_reservoir_cycle_ids", [])
    capacity = int(_DRAFT_HIDDEN_RESERVOIR_CAPACITY)
    retained_count = min(population_size, capacity)
    inclusion_probability = (
        float(retained_count) / float(population_size)
        if population_size > 0
        else 0.0
    )
    ht_weight = (
        1.0 / inclusion_probability if inclusion_probability > 0.0 else 0.0
    )
    seed = getattr(req, "_joint_trace_hidden_reservoir_seed", None)
    for cycle_id in retained_ids:
        entry = cycle_trace[cycle_id]
        entry["draft_hidden_capture"] = "uniform_reservoir_selected"
        entry["draft_hidden_selected"] = True
        entry["draft_hidden_sampling_scheme"] = _DRAFT_HIDDEN_SAMPLING_SCHEME
        entry["draft_hidden_inclusion_probability"] = inclusion_probability
        entry["draft_hidden_ht_weight"] = ht_weight
        entry["draft_hidden_population_size"] = population_size
        entry["draft_hidden_reservoir_capacity"] = capacity
        entry["draft_hidden_reservoir_seed"] = seed


def _retain_bounded_cycle_draft_hidden(
    *,
    req: Any,
    cycle_trace: List[dict],
    trace_entry: dict,
) -> None:
    """Uniformly retain at most 32 eligible cycle snapshots.

    This is Algorithm R: after observing ``n`` eligible cycles, every cycle is
    retained with probability ``min(1, capacity / n)``.  Retained cycles carry
    the final inclusion probability and its Horvitz--Thompson inverse so the
    trainer can reconstruct an unbiased all-cycle action total.
    """

    cycle_id = int(trace_entry["cycle_id"])
    capacity = int(_DRAFT_HIDDEN_RESERVOIR_CAPACITY)
    if capacity <= 0:
        raise ValueError("draft hidden reservoir capacity must be positive")

    reservoir_ids = getattr(req, "_joint_trace_hidden_reservoir_cycle_ids", None)
    if reservoir_ids is None:
        reservoir_ids = []
        req._joint_trace_hidden_reservoir_cycle_ids = reservoir_ids
    population_size = (
        int(getattr(req, "_joint_trace_hidden_population_size", 0)) + 1
    )
    req._joint_trace_hidden_population_size = population_size

    if len(reservoir_ids) < capacity:
        reservoir_ids.append(cycle_id)
    else:
        replacement = _draft_hidden_reservoir_rng(req).randrange(population_size)
        if replacement < capacity:
            evicted_cycle_id = reservoir_ids[replacement]
            _clear_cycle_draft_hidden(cycle_trace, evicted_cycle_id)
            reservoir_ids[replacement] = cycle_id
        else:
            _clear_cycle_draft_hidden(cycle_trace, cycle_id)

    _update_retained_hidden_sampling_metadata(
        req=req,
        cycle_trace=cycle_trace,
        population_size=population_size,
    )


def build_proposal_credit_masks(
    *,
    proposal_count: int,
    accepted_length: int,
    effective_first_rejection: int | None,
) -> Tuple[List[int], List[int]]:
    """Build causal task/cost credit masks from one authoritative boundary."""

    k = int(proposal_count)
    accepted = int(accepted_length)
    if k < 0 or accepted < 0 or accepted > k:
        raise ValueError(
            f"proposal credit bounds are invalid: K={k}, accepted_length={accepted}."
        )
    if effective_first_rejection is not None:
        rejection = int(effective_first_rejection)
        if rejection != accepted or rejection >= k:
            raise ValueError(
                "effective_first_rejection must equal accepted_length and lie "
                f"inside the proposal block: rejection={rejection}, "
                f"accepted_length={accepted}, K={k}."
            )
    else:
        rejection = None

    task_mask = [int(position < accepted) for position in range(k)]
    cost_mask = list(task_mask)
    if rejection is not None:
        cost_mask[rejection] = 1
    return task_mask, cost_mask


def commit_dflash_cycle_and_record_trace(
    *,
    req: Any,
    proposal_ids: Sequence[int],
    verifier_accepted_length: int,
    correction_or_bonus_id: int,
    proposal_behavior_logprobs: Sequence[float] | None = None,
    proposal_behavior_support_ids: (
        torch.Tensor | Sequence[Sequence[int]] | None
    ) = None,
    proposal_behavior: str = "deterministic_greedy",
    proposal_behavior_topk: int | None = None,
    proposal_behavior_temperature: float | None = None,
    proposal_hidden: torch.Tensor | Sequence[Sequence[float]] | None = None,
    proposal_hidden_dtype: str | None = None,
    mode: str = "collaborative",
    policy_version: str | None = None,
    verifier_forward_width: int | None = None,
) -> Tuple[int, int]:
    """Commit one DFlash cycle and append an output-serializable trace entry.

    ``verifier_accepted_length`` is the verifier's raw accepted draft prefix.
    The returned accepted length can be smaller when a stop condition (notably
    EOS) fires inside that prefix.  In that case no verifier bonus is emitted.
    Tensor draft hidden states stay as bounded CPU tensors while the request is
    live.  The scheduler materializes only retained snapshots to Python lists
    when it emits the final response.

    Returns:
        ``(committed_token_count, committed_draft_token_count)``.
    """

    proposals = [int(token_id) for token_id in proposal_ids]
    k = len(proposals)
    mode = str(mode).strip().lower()
    if mode not in {"collaborative", "collaborative_source"}:
        raise ValueError(
            "joint cycle trace mode must be collaborative or "
            f"collaborative_source, got {mode!r}."
        )
    if verifier_forward_width is None:
        verifier_forward_width = k + 1
    verifier_forward_width = int(verifier_forward_width)
    if verifier_forward_width < k + 1:
        raise ValueError(
            "verifier_forward_width must include the anchor and every proposal: "
            f"width={verifier_forward_width}, K={k}."
        )
    raw_accepted = int(verifier_accepted_length)
    if raw_accepted < 0 or raw_accepted > k:
        raise ValueError(
            "verifier_accepted_length must be within the proposal block: "
            f"accepted={raw_accepted}, K={k}."
        )

    if proposal_behavior_logprobs is None:
        serialized_logprobs = None
    else:
        if len(proposal_behavior_logprobs) != k:
            raise ValueError(
                "proposal_behavior_logprobs must align with proposal_ids: "
                f"got {len(proposal_behavior_logprobs)} values for K={k}."
            )
        serialized_logprobs = [float(logprob) for logprob in proposal_behavior_logprobs]

    # A greedy argmax is not a sample from a categorical behavior policy.  Do
    # not manufacture zeros (or model log-softmax values) and call them q_old.
    if proposal_behavior == "deterministic_greedy" and serialized_logprobs is not None:
        raise ValueError(
            "deterministic_greedy proposals must record behavior logprobs as null."
        )
    if proposal_behavior == "categorical_topk":
        if serialized_logprobs is None:
            raise ValueError(
                "categorical_topk proposals require exact behavior logprobs."
            )
        if proposal_behavior_topk is None or proposal_behavior_temperature is None:
            raise ValueError(
                "categorical_topk proposals require topk and temperature metadata."
            )

    if proposal_hidden is None:
        retained_hidden = None
    elif isinstance(proposal_hidden, torch.Tensor):
        if proposal_hidden.ndim != 2 or int(proposal_hidden.shape[0]) != k:
            raise ValueError(
                "proposal_hidden must align with proposal_ids: "
                f"got shape={tuple(proposal_hidden.shape)} for K={k}."
            )
        if proposal_hidden_dtype is None:
            proposal_hidden_dtype = str(proposal_hidden.dtype).removeprefix("torch.")
        # Own a compact CPU copy for this request. Keeping the tensor avoids
        # constructing tens of thousands of Python float objects every cycle.
        retained_hidden = proposal_hidden.detach().to(device="cpu", copy=True)
    else:
        if len(proposal_hidden) != k:
            raise ValueError(
                "proposal_hidden must align with proposal_ids: "
                f"got {len(proposal_hidden)} rows for K={k}."
            )
        # Preserve the direct helper API used by small tests and callers that
        # already provide ordinary Python values.
        retained_hidden = [
            [float(value) for value in hidden_row] for hidden_row in proposal_hidden
        ]
    retained_behavior_support_ids = _retained_behavior_support_ids(
        proposal_ids=proposals,
        proposal_behavior=proposal_behavior,
        proposal_behavior_topk=proposal_behavior_topk,
        proposal_hidden=retained_hidden,
        proposal_behavior_support_ids=proposal_behavior_support_ids,
    )

    emitted_ids: List[int] = []
    emitted_sources: List[str] = []
    committed_draft_tokens = 0
    stopped_in_draft_prefix = False

    def append_token(token_id: int, source: str) -> bool:
        token_id = int(token_id)
        req.output_ids.append(token_id)
        emitted_ids.append(token_id)
        emitted_sources.append(source)
        req.check_finished()
        if req.finished():
            return True
        if req.grammar is not None:
            req.grammar.accept_token(token_id)
        return False

    for token_id in proposals[:raw_accepted]:
        committed_draft_tokens += 1
        if append_token(token_id, "drafter"):
            stopped_in_draft_prefix = True
            break

    verifier_token_id = None
    verifier_token_kind = None
    if not stopped_in_draft_prefix:
        verifier_token_id = int(correction_or_bonus_id)
        verifier_token_kind = "correction" if raw_accepted < k else "bonus"
        append_token(verifier_token_id, "verifier")

    # Zero-based verifier mismatch position in proposal_ids. A stop condition
    # may prevent that mismatch from becoming an emitted correction, but the
    # verifier result remains useful trajectory information.
    first_rejection = raw_accepted if raw_accepted < k else None
    effective_first_rejection = None if stopped_in_draft_prefix else first_rejection
    proposal_task_credit_mask, proposal_cost_credit_mask = build_proposal_credit_masks(
        proposal_count=k,
        accepted_length=committed_draft_tokens,
        effective_first_rejection=effective_first_rejection,
    )

    if req.customized_info is None:
        req.customized_info = {}
    cycle_trace = req.customized_info.setdefault(JOINT_CYCLE_TRACE_KEY, [])
    trace_entry = {
        "cycle_id": len(cycle_trace),
        "mode": mode,
        # ``mode`` is the source-ownership transport mode required by the joint
        # trainer. Token choice still follows strict speculative acceptance.
        "acceptance_mode": "strict",
        "policy_version": (str(policy_version) if policy_version is not None else None),
        "K": k,
        "verified_width": k,
        "verifier_forward_width": verifier_forward_width,
        "proposal_ids": proposals,
        # Compatibility name used by the trajectory schema. This is exactly
        # candidates[:, 1:1+K], never the authoritative slot-0 anchor.
        "draft_tokens": proposals,
        "proposal_behavior": str(proposal_behavior),
        "draft_policy_kind": str(proposal_behavior),
        "proposal_behavior_logprobs": serialized_logprobs,
        "draft_logprobs": serialized_logprobs,
        # Explicit alias used by policy-gradient rollout consumers.
        "proposal_old_logprobs": serialized_logprobs,
        "proposal_behavior_support_ids": retained_behavior_support_ids,
        "proposal_logprobs_valid": serialized_logprobs is not None,
        "proposal_behavior_topk": (
            int(proposal_behavior_topk) if proposal_behavior_topk is not None else None
        ),
        "proposal_behavior_temperature": (
            float(proposal_behavior_temperature)
            if proposal_behavior_temperature is not None
            else None
        ),
        "draft_hidden": retained_hidden,
        "draft_hidden_dtype": (
            str(proposal_hidden_dtype) if retained_hidden is not None else None
        ),
        "draft_hidden_capture": (
            "captured" if retained_hidden is not None else "disabled"
        ),
        "draft_hidden_selected": retained_hidden is not None,
        "draft_hidden_sampling_scheme": (
            _DRAFT_HIDDEN_SAMPLING_SCHEME
            if retained_hidden is not None
            else None
        ),
        "draft_hidden_aligned_to": "proposal_ids",
        "verifier_accepted_length": raw_accepted,
        "accepted_length": committed_draft_tokens,
        "first_rejection": first_rejection,
        "first_rejection_position": first_rejection,
        "effective_first_rejection": effective_first_rejection,
        "proposal_credit_mask": proposal_task_credit_mask,
        "proposal_task_credit_mask": proposal_task_credit_mask,
        "proposal_cost_credit_mask": proposal_cost_credit_mask,
        "emitted_ids": emitted_ids,
        "emitted_tokens": emitted_ids,
        "emitted_sources": emitted_sources,
        "emitted_token_sources": emitted_sources,
        "correction_or_bonus_id": verifier_token_id,
        "correction_token_id": (
            verifier_token_id if verifier_token_kind == "correction" else None
        ),
        "bonus_token_id": (
            verifier_token_id if verifier_token_kind == "bonus" else None
        ),
        "waste": k - committed_draft_tokens,
        "wasted_draft_positions": k - committed_draft_tokens,
    }
    cycle_trace.append(trace_entry)
    if retained_hidden is not None:
        _retain_bounded_cycle_draft_hidden(
            req=req,
            cycle_trace=cycle_trace,
            trace_entry=trace_entry,
        )

    return len(emitted_ids), committed_draft_tokens


def sample_direct_verifier_token(
    *,
    next_token_logits: torch.Tensor,
    selected_draft_lengths: torch.Tensor,
    draft_token_num: int,
    sampling_info: Any | None,
) -> Tuple[torch.Tensor, torch.Tensor | None, str]:
    """Sample one token directly from ``p_V`` at each gated verify position.

    Row ``m`` of a target verify forward is conditioned on the anchor and the
    first ``m`` proposed tokens.  This helper selects that row, applies the same
    temperature/top-k/top-p/min-p behavior contract as SGLang's PyTorch sampler,
    and returns the exact filtered behavior log probability.  It deliberately
    does not perform speculative rejection or residual ``max(p-q, 0)`` sampling.
    """

    width = int(draft_token_num)
    if width <= 0:
        raise ValueError(f"draft_token_num must be positive, got {width}.")
    if next_token_logits.ndim != 2:
        raise ValueError(
            "next_token_logits must have shape [batch * width, vocab], got "
            f"{tuple(next_token_logits.shape)}."
        )
    selected = selected_draft_lengths.to(
        device=next_token_logits.device, dtype=torch.long
    ).reshape(-1)
    bs = int(selected.numel())
    if int(next_token_logits.shape[0]) != bs * width:
        raise ValueError(
            "next_token_logits row count does not match batch * verify width: "
            f"rows={int(next_token_logits.shape[0])}, batch={bs}, width={width}."
        )
    if bool(((selected < 0) | (selected >= width)).any()):
        raise ValueError(
            "selected_draft_lengths must index a verifier-logit row in "
            f"[0, {width - 1}]."
        )

    flat_rows = torch.arange(bs, device=next_token_logits.device) * width + selected
    selected_logits = next_token_logits.index_select(0, flat_rows)
    if sampling_info is None or sampling_info.is_all_greedy:
        return (
            torch.argmax(selected_logits, dim=-1).to(dtype=torch.int64),
            None,
            "deterministic_greedy",
        )
    if len(sampling_info) != bs:
        raise ValueError(
            "sampling_info size does not match selected verifier rows: "
            f"len={len(sampling_info)}, batch={bs}."
        )
    if getattr(sampling_info, "sampling_seed", None) is not None:
        raise RuntimeError(
            "collaborative source sampling does not yet support per-request "
            "deterministic sampling_seed; disable deterministic inference or "
            "use strict decoding."
        )

    temperatures = sampling_info.temperatures.to(
        device=selected_logits.device, dtype=torch.float32
    ).reshape(bs, 1)
    if bool((~torch.isfinite(temperatures) | (temperatures <= 0)).any()):
        raise ValueError("Verifier sampling temperatures must be positive and finite.")
    probs = torch.softmax(selected_logits.float() / temperatures, dim=-1)
    vocab_size = int(probs.shape[-1])
    top_ks = sampling_info.top_ks.to(
        device=selected_logits.device, dtype=torch.long
    ).reshape(bs, 1)
    top_ks = top_ks.clamp(min=1, max=vocab_size)
    top_ps = sampling_info.top_ps.to(
        device=selected_logits.device, dtype=torch.float32
    ).reshape(bs, 1)
    if bool((~torch.isfinite(top_ps) | (top_ps <= 0) | (top_ps > 1)).any()):
        raise ValueError("Verifier top_p values must be finite and in (0, 1].")
    min_ps = sampling_info.min_ps.to(
        device=selected_logits.device, dtype=torch.float32
    ).reshape(bs, 1)
    if bool((~torch.isfinite(min_ps) | (min_ps < 0) | (min_ps > 1)).any()):
        raise ValueError("Verifier min_p values must be finite and in [0, 1].")

    unfiltered = bool(
        ((top_ks >= vocab_size) & (top_ps >= 1.0) & (min_ps <= 0.0)).all()
    )
    if unfiltered:
        behavior_probs = probs
        sorted_ids = None
    else:
        sorted_probs, sorted_ids = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        ranks = torch.arange(vocab_size, device=selected_logits.device).reshape(
            1, -1
        )
        sorted_probs.masked_fill_(ranks >= top_ks, 0.0)
        # Match SGLang's inclusive nucleus boundary: retain the first token that
        # crosses top_p by testing cumulative mass before the current token.
        sorted_probs.masked_fill_(
            (cumulative_probs - sorted_probs) > top_ps,
            0.0,
        )
        sorted_probs.masked_fill_(
            sorted_probs < sorted_probs[:, :1] * min_ps,
            0.0,
        )
        normalizer = sorted_probs.sum(dim=-1, keepdim=True)
        if bool((~torch.isfinite(normalizer) | (normalizer <= 0)).any()):
            raise RuntimeError("Verifier filtering removed every token from a row.")
        behavior_probs = sorted_probs / normalizer

    # Advance every TP rank's RNG identically, then make rank zero's action and
    # likelihood authoritative.  This mirrors the stochastic Drafter path and
    # prevents later rank-dependent batch shrinkage from desynchronizing RNG.
    sampled_ranks = torch.multinomial(behavior_probs, num_samples=1)
    tp_group = get_tp_group()
    tp_group.broadcast(sampled_ranks, src=0)
    sampled_ids = (
        sampled_ranks.squeeze(1)
        if sorted_ids is None
        else torch.gather(sorted_ids, 1, sampled_ranks).squeeze(1)
    )
    sampled_logprobs = torch.log(
        torch.gather(behavior_probs, 1, sampled_ranks).squeeze(1)
    )
    tp_group.broadcast(sampled_ids, src=0)
    tp_group.broadcast(sampled_logprobs, src=0)
    return (
        sampled_ids.to(dtype=torch.int64),
        sampled_logprobs,
        "categorical",
    )


def gather_prefill_verifier_behavior_logprobs(
    *,
    sampler_output_scores: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    sampling_info: Any | None,
) -> Tuple[torch.Tensor | None, str]:
    """Recover the exact behavior likelihood of DFlash's prefill V action.

    In SGLang's ordinary stochastic CUDA path, ``Sampler.forward`` replaces
    ``LogitsProcessorOutput.next_token_logits`` in place with the
    temperature-scaled softmax probabilities before returning the sampled
    token.  For the unfiltered categorical policy we use in joint training,
    those values are precisely the weights consumed by ``torch.multinomial``.
    The weights are normalized explicitly here because multinomial itself does
    not require its input rows to sum to one.

    Filtered and alternate sampler paths fail closed.  Reconstructing a
    plausible log-softmax after sampling is not sufficient evidence of the
    behavior policy and would make PPO importance ratios invalid.
    """

    if sampler_output_scores.ndim != 2:
        raise ValueError(
            "sampler_output_scores must have shape [batch, vocab], got "
            f"{tuple(sampler_output_scores.shape)}."
        )
    sampled = sampled_token_ids.to(
        device=sampler_output_scores.device, dtype=torch.long
    ).reshape(-1)
    batch_size, vocab_size = sampler_output_scores.shape
    if int(sampled.numel()) != int(batch_size):
        raise ValueError(
            "sampled_token_ids must contain one token per sampler row: "
            f"tokens={int(sampled.numel())}, rows={int(batch_size)}."
        )
    if bool(((sampled < 0) | (sampled >= int(vocab_size))).any()):
        raise ValueError("sampled_token_ids contains an out-of-vocabulary id.")
    if sampling_info is None:
        raise RuntimeError(
            "DFlash collaborative prefill requires SamplingBatchInfo to prove "
            "the Verifier behavior policy."
        )
    if len(sampling_info) != int(batch_size):
        raise ValueError(
            "sampling_info size does not match prefill sampler rows: "
            f"len={len(sampling_info)}, batch={int(batch_size)}."
        )
    if sampling_info.is_all_greedy:
        return None, "deterministic_greedy"
    if (
        bool(getattr(sampling_info, "need_top_k_sampling", False))
        or bool(getattr(sampling_info, "need_top_p_sampling", False))
        or bool(getattr(sampling_info, "need_min_p_sampling", False))
    ):
        raise RuntimeError(
            "Exact collaborative prefill p_old currently requires unfiltered "
            "Verifier sampling (top_k=-1, top_p=1, min_p=0)."
        )

    weights = sampler_output_scores.float()
    if bool((~torch.isfinite(weights) | (weights < 0)).any()):
        raise RuntimeError(
            "Expected the post-sampler Verifier probability tensor, but found "
            "negative or non-finite values."
        )
    row_sums = weights.sum(dim=-1)
    if bool((~torch.isfinite(row_sums) | (row_sums <= 0)).any()):
        raise RuntimeError("Verifier sampler probability rows have invalid mass.")
    # Distinguish the in-place probability output from an unmodified logits
    # tensor.  A loose tolerance admits low-precision softmax rounding but does
    # not silently reinterpret arbitrary logits as behavior weights.
    if bool((weights.max(dim=-1).values > 1.01).any()) or bool(
        ((row_sums - 1.0).abs() > 0.05).any()
    ):
        raise RuntimeError(
            "DFlash collaborative prefill could not prove that sampler output "
            "contains categorical probabilities; refusing inferred p_old."
        )

    selected_weights = torch.gather(weights, 1, sampled[:, None]).squeeze(1)
    if bool((~torch.isfinite(selected_weights) | (selected_weights <= 0)).any()):
        raise RuntimeError(
            "The sampled Verifier token has zero or invalid behavior probability."
        )
    return torch.log(selected_weights) - torch.log(row_sums), "categorical"


def build_collaborative_verifier_prefill_cycle(
    *,
    verifier_token_id: int,
    verifier_behavior_logprob: float | None,
    verifier_behavior: str,
    verifier_behavior_temperature: float | None = None,
    verifier_behavior_topk: int | None = None,
    verifier_behavior_top_p: float | None = None,
    verifier_behavior_min_p: float | None = None,
    policy_version: str | None = None,
) -> dict:
    """Build, but do not commit, the initial source-owned Verifier cycle.

    The target model samples the first generated token during prefill, before a
    DFlash proposal block exists.  This is therefore a genuine ``p_V`` action
    with ``K=0``.  The scheduler assigns the final cycle id and attaches this
    entry only alongside the actual ``req.output_ids.append`` operation.
    """

    token_id = int(verifier_token_id)
    if verifier_behavior == "categorical":
        if (
            verifier_behavior_logprob is None
            or not math.isfinite(float(verifier_behavior_logprob))
            or float(verifier_behavior_logprob) > 0.0
        ):
            raise ValueError(
                "A categorical prefill Verifier action requires an exact finite "
                "behavior logprob."
            )
        if (
            verifier_behavior_temperature is None
            or verifier_behavior_topk is None
            or verifier_behavior_top_p is None
            or verifier_behavior_min_p is None
        ):
            raise ValueError(
                "Categorical prefill actions require temperature/top-k/top-p/min-p "
                "metadata."
            )
    elif verifier_behavior == "deterministic_greedy":
        if verifier_behavior_logprob is not None:
            raise ValueError(
                "A deterministic prefill Verifier action must record behavior "
                "logprob as null."
            )
    else:
        raise ValueError(
            f"Unsupported prefill Verifier behavior {verifier_behavior!r}."
        )

    return {
        # The scheduler replaces this placeholder at the commit boundary.
        "cycle_id": None,
        "mode": "collaborative_source",
        "acceptance_mode": "collaborative_source",
        "policy_version": (
            str(policy_version) if policy_version is not None else None
        ),
        "K": 0,
        "verified_width": 0,
        "verifier_forward_width": 1,
        "proposal_ids": [],
        "draft_tokens": [],
        "proposal_behavior": "not_applicable",
        "draft_policy_kind": "not_applicable",
        "proposal_behavior_logprobs": None,
        "draft_logprobs": None,
        "proposal_old_logprobs": None,
        "proposal_behavior_support_ids": None,
        "proposal_logprobs_valid": False,
        "proposal_behavior_topk": None,
        "proposal_behavior_temperature": None,
        "draft_hidden": None,
        "draft_hidden_dtype": None,
        "draft_hidden_capture": "disabled",
        "draft_hidden_selected": False,
        "draft_hidden_sampling_scheme": None,
        "draft_hidden_aligned_to": "proposal_ids",
        "source_gate": {
            "kind": "verifier_prefill",
            "mode": "fixed",
            "planned_draft_tokens": 0,
            "max_draft_tokens": 0,
            "threshold": None,
            "stochastic": False,
            "logprob": None,
        },
        "selected_draft_length": 0,
        "verifier_accepted_length": None,
        "accepted_length": 0,
        "first_rejection": None,
        "first_rejection_position": None,
        "effective_first_rejection": None,
        "proposal_credit_mask": [],
        "proposal_task_credit_mask": [],
        "proposal_cost_credit_mask": [],
        "emitted_ids": [token_id],
        "emitted_tokens": [token_id],
        "emitted_sources": ["verifier"],
        "emitted_token_sources": ["verifier"],
        "verifier_token_id": token_id,
        "verifier_token_kind": "direct_sample",
        "verifier_behavior": str(verifier_behavior),
        "verifier_behavior_logprob": (
            float(verifier_behavior_logprob)
            if verifier_behavior_logprob is not None
            else None
        ),
        "verifier_behavior_temperature": (
            float(verifier_behavior_temperature)
            if verifier_behavior_temperature is not None
            else None
        ),
        "verifier_behavior_topk": (
            int(verifier_behavior_topk)
            if verifier_behavior_topk is not None
            else None
        ),
        "verifier_behavior_top_p": (
            float(verifier_behavior_top_p)
            if verifier_behavior_top_p is not None
            else None
        ),
        "verifier_behavior_min_p": (
            float(verifier_behavior_min_p)
            if verifier_behavior_min_p is not None
            else None
        ),
        "correction_or_bonus_id": None,
        "correction_token_id": None,
        "bonus_token_id": None,
        "waste": 0,
        "wasted_draft_positions": 0,
    }


def select_collaborative_draft_lengths(
    *,
    candidates: torch.Tensor,
    next_token_logits: torch.Tensor,
    gate_mode: str,
    fixed_draft_tokens: int | None = None,
    valid_draft_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select a deterministic q-prefix without residual acceptance."""

    if candidates.ndim != 2:
        raise ValueError(
            f"candidates must have shape [batch, width], got {tuple(candidates.shape)}."
        )
    bs, width = candidates.shape
    max_draft_tokens = int(width) - 1
    if max_draft_tokens < 0:
        raise ValueError("candidates must include an anchor token.")
    if tuple(next_token_logits.shape[:1]) != (int(bs) * int(width),):
        raise ValueError(
            "next_token_logits must contain one row per candidate slot: "
            f"got rows={int(next_token_logits.shape[0])}, batch={bs}, width={width}."
        )
    if valid_draft_lens is None:
        valid = torch.full(
            (bs,),
            max_draft_tokens,
            dtype=torch.int64,
            device=candidates.device,
        )
    else:
        valid = valid_draft_lens.to(
            device=candidates.device, dtype=torch.int64
        ).clamp(min=0, max=max_draft_tokens)

    normalized_mode = str(gate_mode).strip().lower()
    if normalized_mode == "fixed":
        if fixed_draft_tokens is None:
            raise ValueError("The fixed collaborative gate requires a draft length.")
        fixed = int(fixed_draft_tokens)
        if fixed < 0 or fixed > max_draft_tokens:
            raise ValueError(
                "fixed_draft_tokens must be within the proposal block: "
                f"fixed={fixed}, K={max_draft_tokens}."
            )
        selected = torch.full_like(valid, fixed)
        return torch.minimum(selected, valid)
    if normalized_mode == "greedy_match":
        target_predict = torch.argmax(next_token_logits, dim=-1).view(bs, width)
        proposal_matches = candidates[:, 1:] == target_predict[:, :max_draft_tokens]
        selected = torch.cumprod(
            proposal_matches.to(dtype=torch.int64), dim=1
        ).sum(dim=1)
        return torch.minimum(selected, valid)
    raise ValueError(
        f"Collaborative gate mode must be fixed or greedy_match, got {gate_mode!r}."
    )


def commit_collaborative_dflash_cycle_and_record_trace(
    *,
    req: Any,
    proposal_ids: Sequence[int],
    selected_draft_length: int,
    verifier_token_id: int,
    verifier_behavior_logprob: float | None,
    verifier_behavior: str,
    verifier_behavior_temperature: float | None = None,
    verifier_behavior_topk: int | None = None,
    verifier_behavior_top_p: float | None = None,
    verifier_behavior_min_p: float | None = None,
    proposal_behavior_logprobs: Sequence[float] | None = None,
    proposal_behavior_support_ids: (
        torch.Tensor | Sequence[Sequence[int]] | None
    ) = None,
    proposal_behavior: str = "deterministic_greedy",
    proposal_behavior_topk: int | None = None,
    proposal_behavior_temperature: float | None = None,
    proposal_hidden: torch.Tensor | Sequence[Sequence[float]] | None = None,
    proposal_hidden_dtype: str | None = None,
    source_gate_kind: str = "fixed_draft_prefix_then_verifier",
    source_gate_mode: str = "fixed",
    source_gate_threshold: float | None = None,
    source_gate_max_draft_tokens: int | None = None,
    source_gate_stochastic: bool = False,
    source_gate_logprob: float | None = None,
    policy_version: str | None = None,
    verifier_forward_width: int | None = None,
) -> Tuple[int, int]:
    """Commit ``q_D`` actions followed by one direct ``p_V`` action.

    For a gate selecting ``m`` draft actions, the response policy factorization
    for this cycle is ``prod_{i=1..m} q_D(d_i) * p_V(v | d_1..d_m)``.  A stop
    condition inside the draft prefix truncates the cycle before the Verifier
    action.  No speculative acceptance probability or residual distribution is
    part of this kernel.
    """

    proposals = [int(token_id) for token_id in proposal_ids]
    k = len(proposals)
    planned_draft_tokens = int(selected_draft_length)
    if planned_draft_tokens < 0 or planned_draft_tokens > k:
        raise ValueError(
            "selected_draft_length must lie inside the proposal block: "
            f"selected={planned_draft_tokens}, K={k}."
        )
    source_gate_mode = str(source_gate_mode).strip().lower()
    if source_gate_mode not in {"fixed", "greedy_match"}:
        raise ValueError(
            "source_gate_mode must be fixed or greedy_match, got "
            f"{source_gate_mode!r}."
        )
    if source_gate_max_draft_tokens is None:
        source_gate_max_draft_tokens = k
    source_gate_max_draft_tokens = int(source_gate_max_draft_tokens)
    if (
        source_gate_max_draft_tokens < planned_draft_tokens
        or source_gate_max_draft_tokens > k
    ):
        raise ValueError(
            "source_gate_max_draft_tokens must cover the selected prefix and "
            f"stay within K: selected={planned_draft_tokens}, "
            f"max={source_gate_max_draft_tokens}, K={k}."
        )
    if source_gate_mode == "greedy_match":
        if source_gate_threshold is None:
            source_gate_threshold = 1.0
        if float(source_gate_threshold) != 1.0:
            raise ValueError(
                "The deterministic greedy_match gate currently requires threshold=1.0."
            )
    elif source_gate_threshold is not None:
        raise ValueError("The fixed source gate must record threshold as null.")
    if verifier_forward_width is None:
        verifier_forward_width = k + 1
    verifier_forward_width = int(verifier_forward_width)
    if verifier_forward_width < k + 1:
        raise ValueError(
            "verifier_forward_width must include the anchor and every proposal: "
            f"width={verifier_forward_width}, K={k}."
        )
    if not isinstance(source_gate_stochastic, bool):
        raise ValueError("source_gate_stochastic must be an explicit boolean.")
    if source_gate_stochastic:
        if (
            source_gate_logprob is None
            or not math.isfinite(float(source_gate_logprob))
            or float(source_gate_logprob) > 0.0
        ):
            raise ValueError(
                "A stochastic source gate requires an exact finite logprob."
            )
    elif source_gate_logprob is not None:
        raise ValueError(
            "A deterministic source gate must record its logprob as null."
        )

    if proposal_behavior_logprobs is None:
        serialized_draft_logprobs = None
    else:
        if len(proposal_behavior_logprobs) != k:
            raise ValueError(
                "proposal_behavior_logprobs must align with proposal_ids: "
                f"got {len(proposal_behavior_logprobs)} values for K={k}."
            )
        serialized_draft_logprobs = [
            float(logprob) for logprob in proposal_behavior_logprobs
        ]
        if not all(
            math.isfinite(logprob) and logprob <= 0.0
            for logprob in serialized_draft_logprobs
        ):
            raise ValueError(
                "proposal_behavior_logprobs must be finite log probabilities."
            )
    draft_credit_count = planned_draft_tokens
    if source_gate_mode == "greedy_match" and planned_draft_tokens < k:
        # The first non-matching q action is not emitted, but it caused the
        # gate to stop and therefore owns cost/cooperation credit.
        draft_credit_count += 1
    if draft_credit_count:
        if proposal_behavior != "categorical_topk":
            raise ValueError(
                "Collaborative Drafter actions require categorical_topk behavior."
            )
        if serialized_draft_logprobs is None:
            raise ValueError(
                "Collaborative Drafter actions require exact q behavior logprobs."
            )
        if proposal_behavior_topk is None or proposal_behavior_temperature is None:
            raise ValueError(
                "categorical_topk proposals require topk and temperature metadata."
            )

    if verifier_behavior == "categorical":
        if (
            verifier_behavior_logprob is None
            or not math.isfinite(float(verifier_behavior_logprob))
            or float(verifier_behavior_logprob) > 0.0
        ):
            raise ValueError(
                "A categorical Verifier action requires an exact finite p logprob."
            )
        if (
            verifier_behavior_temperature is None
            or verifier_behavior_topk is None
            or verifier_behavior_top_p is None
            or verifier_behavior_min_p is None
        ):
            raise ValueError(
                "Categorical Verifier actions require temperature/top-k/top-p/min-p metadata."
            )
    elif verifier_behavior == "deterministic_greedy":
        if verifier_behavior_logprob is not None:
            raise ValueError(
                "A deterministic Verifier action must record behavior logprob as null."
            )
    else:
        raise ValueError(f"Unsupported Verifier behavior {verifier_behavior!r}.")

    if proposal_hidden is None:
        retained_hidden = None
    elif isinstance(proposal_hidden, torch.Tensor):
        if proposal_hidden.ndim != 2 or int(proposal_hidden.shape[0]) != k:
            raise ValueError(
                "proposal_hidden must align with proposal_ids: "
                f"got shape={tuple(proposal_hidden.shape)} for K={k}."
            )
        if proposal_hidden_dtype is None:
            proposal_hidden_dtype = str(proposal_hidden.dtype).removeprefix("torch.")
        retained_hidden = proposal_hidden.detach().to(device="cpu", copy=True)
    else:
        if len(proposal_hidden) != k:
            raise ValueError(
                "proposal_hidden must align with proposal_ids: "
                f"got {len(proposal_hidden)} rows for K={k}."
            )
        retained_hidden = [
            [float(value) for value in hidden_row] for hidden_row in proposal_hidden
        ]
    retained_behavior_support_ids = _retained_behavior_support_ids(
        proposal_ids=proposals,
        proposal_behavior=proposal_behavior,
        proposal_behavior_topk=proposal_behavior_topk,
        proposal_hidden=retained_hidden,
        proposal_behavior_support_ids=proposal_behavior_support_ids,
    )

    emitted_ids: List[int] = []
    emitted_sources: List[str] = []
    committed_draft_tokens = 0
    stopped_in_draft_prefix = False

    def append_token(token_id: int, source: str) -> bool:
        token_id = int(token_id)
        req.output_ids.append(token_id)
        emitted_ids.append(token_id)
        emitted_sources.append(source)
        req.check_finished()
        if req.finished():
            return True
        if req.grammar is not None:
            req.grammar.accept_token(token_id)
        return False

    for token_id in proposals[:planned_draft_tokens]:
        committed_draft_tokens += 1
        if append_token(token_id, "drafter"):
            stopped_in_draft_prefix = True
            break

    emitted_verifier_token_id = None
    if not stopped_in_draft_prefix:
        emitted_verifier_token_id = int(verifier_token_id)
        append_token(emitted_verifier_token_id, "verifier")

    proposal_task_credit_mask = [
        int(position < committed_draft_tokens) for position in range(k)
    ]
    proposal_cost_credit_mask = list(proposal_task_credit_mask)
    if (
        source_gate_mode == "greedy_match"
        and not stopped_in_draft_prefix
        and planned_draft_tokens < k
    ):
        proposal_cost_credit_mask[planned_draft_tokens] = 1
    emitted_verifier_logprob = (
        float(verifier_behavior_logprob)
        if emitted_verifier_token_id is not None
        and verifier_behavior_logprob is not None
        else None
    )

    if req.customized_info is None:
        req.customized_info = {}
    cycle_trace = req.customized_info.setdefault(JOINT_CYCLE_TRACE_KEY, [])
    trace_entry = {
        "cycle_id": len(cycle_trace),
        "mode": "collaborative_source",
        "acceptance_mode": "collaborative_source",
        "policy_version": (str(policy_version) if policy_version is not None else None),
        "K": k,
        "verified_width": k,
        "verifier_forward_width": verifier_forward_width,
        "proposal_ids": proposals,
        "draft_tokens": proposals,
        "proposal_behavior": str(proposal_behavior),
        "draft_policy_kind": str(proposal_behavior),
        "proposal_behavior_logprobs": serialized_draft_logprobs,
        "draft_logprobs": serialized_draft_logprobs,
        "proposal_old_logprobs": serialized_draft_logprobs,
        "proposal_behavior_support_ids": retained_behavior_support_ids,
        "proposal_logprobs_valid": serialized_draft_logprobs is not None,
        "proposal_behavior_topk": (
            int(proposal_behavior_topk) if proposal_behavior_topk is not None else None
        ),
        "proposal_behavior_temperature": (
            float(proposal_behavior_temperature)
            if proposal_behavior_temperature is not None
            else None
        ),
        "draft_hidden": retained_hidden,
        "draft_hidden_dtype": (
            str(proposal_hidden_dtype) if retained_hidden is not None else None
        ),
        "draft_hidden_capture": (
            "captured" if retained_hidden is not None else "disabled"
        ),
        "draft_hidden_selected": retained_hidden is not None,
        "draft_hidden_sampling_scheme": (
            _DRAFT_HIDDEN_SAMPLING_SCHEME
            if retained_hidden is not None
            else None
        ),
        "draft_hidden_aligned_to": "proposal_ids",
        "source_gate": {
            "kind": str(source_gate_kind),
            "mode": source_gate_mode,
            "planned_draft_tokens": planned_draft_tokens,
            "max_draft_tokens": source_gate_max_draft_tokens,
            "threshold": (
                float(source_gate_threshold)
                if source_gate_threshold is not None
                else None
            ),
            "stochastic": source_gate_stochastic,
            "logprob": (
                float(source_gate_logprob)
                if source_gate_logprob is not None
                else None
            ),
        },
        "selected_draft_length": planned_draft_tokens,
        "verifier_accepted_length": None,
        "accepted_length": committed_draft_tokens,
        "first_rejection": None,
        "first_rejection_position": None,
        "effective_first_rejection": None,
        "proposal_credit_mask": proposal_task_credit_mask,
        "proposal_task_credit_mask": proposal_task_credit_mask,
        "proposal_cost_credit_mask": proposal_cost_credit_mask,
        "emitted_ids": emitted_ids,
        "emitted_tokens": emitted_ids,
        "emitted_sources": emitted_sources,
        "emitted_token_sources": emitted_sources,
        "verifier_token_id": emitted_verifier_token_id,
        "verifier_token_kind": (
            "direct_sample" if emitted_verifier_token_id is not None else None
        ),
        "verifier_behavior": (
            str(verifier_behavior)
            if emitted_verifier_token_id is not None
            else None
        ),
        "verifier_behavior_logprob": emitted_verifier_logprob,
        "verifier_behavior_temperature": (
            float(verifier_behavior_temperature)
            if emitted_verifier_token_id is not None
            and verifier_behavior_temperature is not None
            else None
        ),
        "verifier_behavior_topk": (
            int(verifier_behavior_topk)
            if emitted_verifier_token_id is not None
            and verifier_behavior_topk is not None
            else None
        ),
        "verifier_behavior_top_p": (
            float(verifier_behavior_top_p)
            if emitted_verifier_token_id is not None
            and verifier_behavior_top_p is not None
            else None
        ),
        "verifier_behavior_min_p": (
            float(verifier_behavior_min_p)
            if emitted_verifier_token_id is not None
            and verifier_behavior_min_p is not None
            else None
        ),
        "correction_or_bonus_id": None,
        "correction_token_id": None,
        "bonus_token_id": None,
        "waste": k - committed_draft_tokens,
        "wasted_draft_positions": k - committed_draft_tokens,
    }
    cycle_trace.append(trace_entry)
    if retained_hidden is not None:
        _retain_bounded_cycle_draft_hidden(
            req=req,
            cycle_trace=cycle_trace,
            trace_entry=trace_entry,
        )
    return len(emitted_ids), committed_draft_tokens


def _compute_paged_keep_slots(
    *,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    draft_token_num: int,
    page_size: int,
) -> torch.Tensor:
    """Compute how many draft slots per request must remain allocated.

    The allocator frees at page granularity for paged mode, so we can only release
    full pages from the tail after verify.
    """

    if page_size <= 1:
        raise ValueError(f"Expected page_size > 1, got {page_size}.")

    seq_dtype = prefix_lens.dtype
    extended_lens = prefix_lens + int(draft_token_num)
    new_lens = prefix_lens + commit_lens.to(seq_dtype)
    aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size
    keep_lens = torch.minimum(aligned_new_lens, extended_lens)
    keep_slots = (keep_lens - prefix_lens).to(torch.int64)
    keep_slots.clamp_(min=0, max=int(draft_token_num))
    return keep_slots


@dataclass
class DFlashDraftInput(SpecInput):
    """Per-batch DFlash draft state for spec-v1 (non-overlap) scheduling.

    This object is stored on `ScheduleBatch.spec_info` between decode iterations.
    It is NOT sent to model attention backends; the DFlash worker uses it to run
    the draft model and to track draft-side cache progress.

    When draft windowing is disabled, `draft_seq_lens` matches the committed target
    prefix length already materialized in the draft KV cache. When windowing is
    enabled, `draft_seq_lens` is the logical resident length in the draft worker's
    compact req-to-token mapping. In paged mode this may exceed the requested
    window by up to `page_size - 1` so the local page table remains valid. `ctx_lens`
    tracks newly committed target tokens that still need draft KV materialization.
    """

    # Current token to start the next DFlash block (one per request).
    verified_id: torch.Tensor

    # Flattened context features for tokens that need to be appended into the draft cache.
    # Shape: [sum(ctx_lens), K * hidden_size], where K is the number of target-layer
    # hidden-state features concatenated per token (len(dflash_config.target_layer_ids),
    # or default K == draft_num_layers for existing checkpoints).
    target_hidden: torch.Tensor

    # Context lengths per request, used to slice `target_hidden`. Device tensor (int32).
    ctx_lens: torch.Tensor

    # How many committed tokens are visible to the draft worker per request.
    draft_seq_lens: torch.Tensor

    # Last raw target feature row for each request. Sparse full-tower DFlash
    # needs the anchor-1 target state for its leakage-free block router even
    # after target_hidden has been projected into (and cleared from) the draft
    # KV cache. Shape: [batch, K * hidden_size].
    router_target_hidden: torch.Tensor | None = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Draft state does not change token accounting.
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden
        old_router_target_hidden = self.router_target_hidden

        self.verified_id = self.verified_id[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]
        if old_router_target_hidden is not None:
            self.router_target_hidden = old_router_target_hidden[new_indices]

        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

        # Rebuild target_hidden for the filtered batch using vectorized indexing.
        old_bs = int(old_ctx_lens.shape[0])
        offsets = torch.zeros(
            (old_bs + 1,), dtype=torch.int64, device=old_ctx_lens.device
        )
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))

        start = offsets[:-1]
        seg_start = start[new_indices]
        seg_lens = old_ctx_lens[new_indices].to(torch.int64)

        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
        if max_len <= 0:
            self.target_hidden = old_target_hidden[:0]
            return

        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[
            None, :
        ]
        pos2d = seg_start[:, None] + r
        mask = r < seg_lens[:, None]
        flat_pos = pos2d[mask]
        self.target_hidden = (
            old_target_hidden.index_select(0, flat_pos)
            if flat_pos.numel() > 0
            else old_target_hidden[:0]
        )

    def merge_batch(self, spec_info: "DFlashDraftInput"):
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif (
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(
                [self.target_hidden, spec_info.target_hidden], dim=0
            )
        if self.router_target_hidden is None:
            self.router_target_hidden = spec_info.router_target_hidden
        elif spec_info.router_target_hidden is not None:
            self.router_target_hidden = torch.cat(
                [self.router_target_hidden, spec_info.router_target_hidden], dim=0
            )


@dataclass
class DFlashVerifyInput(SpecInput):
    """Inputs for a target-model verify forward in DFlash (spec-v1).

    The verify forward is run with `ForwardMode.TARGET_VERIFY` so that the target
    model returns logits for all tokens in the block, enabling accept-length
    computation.
    """

    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    # Number of real draft proposals (excluding the anchor) in each fixed-width
    # row.  Active drafter/judge chaining pads shorter rows so target CUDA graph
    # shapes remain static.  Native one-block DFlash leaves this as ``None``.
    valid_draft_lens: torch.Tensor | None = None
    # Optional LM-head input rows aligned with candidates[:, 1:]. This remains
    # None unless explicitly enabled for training trace collection.
    proposal_hidden: torch.Tensor | None = None
    # Exact behavior-policy log probabilities for candidates[:, 1:]. Greedy
    # proposals leave this unset rather than fabricating q_old.
    proposal_behavior_logprobs: torch.Tensor | None = None
    # Exact old-policy top-k token set for every proposal row. This is captured
    # only alongside proposal_hidden and is the normalization support used by
    # Drafter PPO replay.
    proposal_behavior_support_ids: torch.Tensor | None = None
    proposal_behavior: str = "deterministic_greedy"
    proposal_behavior_topk: int | None = None
    proposal_behavior_temperature: float | None = None
    # Strict is the default and preserves the original speculative-decoding
    # distribution. Collaborative source mode is an explicit composite policy:
    # a deterministic gate selects q_D actions, then one action is drawn from p_V.
    acceptance_mode: str = "strict"
    collaborative_gate_mode: str | None = None
    collaborative_fixed_draft_tokens: int | None = None
    policy_version: str | None = None
    # Kept for compatibility with attention backends that gate tree metadata by `topk > 1`.
    # DFLASH verify is linear (non-tree), so this is always 1.
    topk: int = 1
    # Custom attention "allow mask" for TARGET_VERIFY in backends that require it (e.g. triton).
    # Semantics follow SGLang speculative conventions: True means the (q, k) pair is allowed.
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Shape info for padding (e.g., DP attention / CUDA graph).
    num_tokens_per_batch: int = -1

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_VERIFY)
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
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
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
                f"DFLASH draft_token_num must be positive, got {self.draft_token_num}."
            )
        mask_chunks: List[torch.Tensor] = []
        q_len = int(self.draft_token_num)
        q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
        for prefix_len in batch.seq_lens_cpu.tolist():
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + q_len
            k_idx = torch.arange(
                kv_len, device=batch.device, dtype=torch.int32
            ).unsqueeze(0)
            # Allow attending to the full prefix and to tokens up to (and including) the
            # current query position within the verify block (standard causal masking).
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
                # FIXME(attn): temporary fix for custom mask padding with cuda graph
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
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[int],
        List[int],
    ]:
        """DFlash verification for greedy and non-greedy sampling.

        Returns:
            new_verified_id: int64 tensor [bs] (the new current token per request)
            commit_lens: int32 tensor [bs] (how many verify-input tokens are committed)
            next_target_hidden: tensor [sum(commit_lens), feature_dim]
            accept_length_per_req_cpu: list[int] (accepted draft tokens per request)
            commit_lens_cpu: list[int] (actual committed rows per request)
        """
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, [], []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device

        sampling_info = batch.sampling_info
        if sampling_info is not None:
            if len(sampling_info) != bs:
                raise RuntimeError(
                    "DFLASH verify sampling_info size mismatch: "
                    f"len(sampling_info)={len(sampling_info)}, bs={bs}."
                )

            # Keep speculative verify semantics consistent with normal sampling path.
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
            if self.valid_draft_lens is not None and not sampling_info.is_all_greedy:
                raise RuntimeError(
                    "Active DFLASH drafter/judge verification currently supports "
                    "greedy target sampling only."
                )

        candidates = self.draft_token.view(bs, self.draft_token_num)
        max_acc = self.draft_token_num - 1
        if self.valid_draft_lens is None:
            valid_draft_lens = torch.full(
                (bs,),
                max_acc,
                dtype=torch.int64,
                device=candidates.device,
            )
        else:
            valid_draft_lens = self.valid_draft_lens.to(
                device=candidates.device, dtype=torch.int64
            ).clamp(min=0, max=max_acc)

        acceptance_mode = str(self.acceptance_mode).strip().lower()
        collaborative = acceptance_mode == "collaborative_source"
        if acceptance_mode not in {"strict", "collaborative_source"}:
            raise ValueError(
                "DFLASH acceptance_mode must be strict or collaborative_source, "
                f"got {self.acceptance_mode!r}."
            )

        verifier_behavior = None
        verifier_behavior_logprobs_cpu = None
        verifier_temperatures_cpu = None
        verifier_topks_cpu = None
        verifier_top_ps_cpu = None
        verifier_min_ps_cpu = None
        source_gate_kind = None
        source_gate_mode = None
        source_gate_threshold = None
        source_gate_max_draft_tokens = None

        if collaborative:
            if self.valid_draft_lens is not None:
                raise RuntimeError(
                    "Collaborative source verification is incompatible with "
                    "active multi-block DFLASH padding."
                )
            gate_mode = str(self.collaborative_gate_mode or "").strip().lower()
            if gate_mode == "fixed":
                if self.collaborative_fixed_draft_tokens is None:
                    raise ValueError(
                        "The fixed collaborative gate requires an explicit draft length."
                    )
                fixed = int(self.collaborative_fixed_draft_tokens)
                if fixed < 0 or fixed > max_acc:
                    raise ValueError(
                        "Collaborative fixed draft tokens must be within the "
                        f"proposal block [0, {max_acc}], got {fixed}."
                    )
                accept_len = select_collaborative_draft_lengths(
                    candidates=candidates,
                    next_token_logits=logits_output.next_token_logits,
                    gate_mode=gate_mode,
                    fixed_draft_tokens=fixed,
                    valid_draft_lens=valid_draft_lens,
                )
                source_gate_kind = "fixed_draft_prefix_then_verifier"
                source_gate_mode = "fixed"
                source_gate_max_draft_tokens = fixed
            elif gate_mode == "greedy_match":
                accept_len = select_collaborative_draft_lengths(
                    candidates=candidates,
                    next_token_logits=logits_output.next_token_logits,
                    gate_mode=gate_mode,
                    valid_draft_lens=valid_draft_lens,
                )
                source_gate_kind = "greedy_match_then_verifier"
                source_gate_mode = "greedy_match"
                source_gate_threshold = 1.0
                source_gate_max_draft_tokens = max_acc
            else:
                raise ValueError(
                    "Collaborative DFLASH gate mode must be fixed or "
                    f"greedy_match, got {self.collaborative_gate_mode!r}."
                )

            bonus, verifier_behavior_logprobs, verifier_behavior = (
                sample_direct_verifier_token(
                    next_token_logits=logits_output.next_token_logits,
                    selected_draft_lengths=accept_len,
                    draft_token_num=self.draft_token_num,
                    sampling_info=sampling_info,
                )
            )
            if verifier_behavior_logprobs is not None:
                verifier_behavior_logprobs_cpu = (
                    verifier_behavior_logprobs.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .tolist()
                )
                verifier_temperatures_cpu = (
                    sampling_info.temperatures.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .reshape(-1)
                    .tolist()
                )
                verifier_topks_cpu = (
                    sampling_info.top_ks.detach()
                    .to(device="cpu", dtype=torch.int64)
                    .reshape(-1)
                    .tolist()
                )
                verifier_top_ps_cpu = (
                    sampling_info.top_ps.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .reshape(-1)
                    .tolist()
                )
                verifier_min_ps_cpu = (
                    sampling_info.min_ps.detach()
                    .to(device="cpu", dtype=torch.float32)
                    .reshape(-1)
                    .tolist()
                )
        elif (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_accept_len_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
            )
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, self.draft_token_num
            )
            accept_len, bonus = compute_dflash_accept_len_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
                valid_draft_lens=self.valid_draft_lens,
            )

        # Single D2H transfer: candidates[1:] + real K + selected/accepted + V token.
        packed = torch.cat(
            [
                candidates[:, 1:],
                valid_draft_lens.unsqueeze(1),
                accept_len.unsqueeze(1),
                bonus.unsqueeze(1),
            ],
            dim=1,
        ).cpu()

        accept_length_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_verified_list: List[int] = []
        proposal_hidden_cpu = None
        proposal_hidden_dtype = None
        if self.proposal_hidden is not None:
            expected_shape = (bs, max_acc)
            if (
                self.proposal_hidden.ndim != 3
                or tuple(self.proposal_hidden.shape[:2]) != expected_shape
            ):
                raise ValueError(
                    "proposal_hidden must have shape [batch, K, hidden_size]: "
                    f"expected prefix={expected_shape}, "
                    f"got shape={tuple(self.proposal_hidden.shape)}."
                )
            proposal_hidden_dtype = str(self.proposal_hidden.dtype).removeprefix(
                "torch."
            )
            # Preserve the compact model dtype and defer Python-list
            # materialization until the final response is emitted.
            proposal_hidden_cpu = self.proposal_hidden.detach().to(device="cpu")
        proposal_behavior_logprobs_cpu = None
        if self.proposal_behavior_logprobs is not None:
            expected_shape = (bs, max_acc)
            if tuple(self.proposal_behavior_logprobs.shape) != expected_shape:
                raise ValueError(
                    "proposal_behavior_logprobs must have shape [batch, K]: "
                    f"expected={expected_shape}, "
                    f"got shape={tuple(self.proposal_behavior_logprobs.shape)}."
                )
            proposal_behavior_logprobs_cpu = (
                self.proposal_behavior_logprobs.detach()
                .to(device="cpu", dtype=torch.float32)
                .tolist()
            )
        proposal_behavior_support_ids_cpu = None
        if self.proposal_behavior_support_ids is not None:
            if self.proposal_hidden is None:
                raise ValueError(
                    "proposal behavior support ids require aligned proposal_hidden."
                )
            if (
                isinstance(self.proposal_behavior_topk, bool)
                or not isinstance(self.proposal_behavior_topk, Integral)
                or int(self.proposal_behavior_topk) < 2
            ):
                raise ValueError(
                    "proposal behavior support ids require integer topk >= 2."
                )
            expected_shape = (
                bs,
                max_acc,
                int(self.proposal_behavior_topk),
            )
            if tuple(self.proposal_behavior_support_ids.shape) != expected_shape:
                raise ValueError(
                    "proposal_behavior_support_ids must have shape "
                    "[batch, K, topk]: "
                    f"expected={expected_shape}, "
                    f"got={tuple(self.proposal_behavior_support_ids.shape)}."
                )
            proposal_behavior_support_ids_cpu = (
                self.proposal_behavior_support_ids.detach()
                .to(device="cpu", dtype=torch.long)
            )
        elif (
            self.proposal_hidden is not None
            and self.proposal_behavior == "categorical_topk"
        ):
            raise ValueError(
                "categorical_topk proposal replay requires behavior support ids "
                "whenever proposal_hidden is captured."
            )

        for i, req in enumerate(batch.reqs):
            k = int(packed[i, max_acc].item())
            acc_len = int(packed[i, max_acc + 1].item())
            bonus_token_id = int(packed[i, max_acc + 2].item())
            proposal_ids = packed[i, :k].tolist()
            common_trace_kwargs = {
                "req": req,
                "proposal_ids": proposal_ids,
                "proposal_behavior_logprobs": (
                    proposal_behavior_logprobs_cpu[i][:k]
                    if proposal_behavior_logprobs_cpu is not None
                    else None
                ),
                "proposal_behavior_support_ids": (
                    proposal_behavior_support_ids_cpu[i, :k]
                    if proposal_behavior_support_ids_cpu is not None
                    else None
                ),
                "proposal_behavior": self.proposal_behavior,
                "proposal_behavior_topk": self.proposal_behavior_topk,
                "proposal_behavior_temperature": self.proposal_behavior_temperature,
                "proposal_hidden": (
                    proposal_hidden_cpu[i, :k]
                    if proposal_hidden_cpu is not None
                    else None
                ),
                "proposal_hidden_dtype": proposal_hidden_dtype,
                "policy_version": self.policy_version,
                "verifier_forward_width": self.draft_token_num,
            }
            if collaborative:
                appended, accepted_draft_tokens = (
                    commit_collaborative_dflash_cycle_and_record_trace(
                        **common_trace_kwargs,
                        selected_draft_length=acc_len,
                        verifier_token_id=bonus_token_id,
                        verifier_behavior_logprob=(
                            verifier_behavior_logprobs_cpu[i]
                            if verifier_behavior_logprobs_cpu is not None
                            else None
                        ),
                        verifier_behavior=str(verifier_behavior),
                        verifier_behavior_temperature=(
                            verifier_temperatures_cpu[i]
                            if verifier_temperatures_cpu is not None
                            else None
                        ),
                        verifier_behavior_topk=(
                            verifier_topks_cpu[i]
                            if verifier_topks_cpu is not None
                            else None
                        ),
                        verifier_behavior_top_p=(
                            verifier_top_ps_cpu[i]
                            if verifier_top_ps_cpu is not None
                            else None
                        ),
                        verifier_behavior_min_p=(
                            verifier_min_ps_cpu[i]
                            if verifier_min_ps_cpu is not None
                            else None
                        ),
                        source_gate_kind=str(source_gate_kind),
                        source_gate_mode=str(source_gate_mode),
                        source_gate_threshold=source_gate_threshold,
                        source_gate_max_draft_tokens=source_gate_max_draft_tokens,
                    )
                )
            else:
                appended, accepted_draft_tokens = (
                    commit_dflash_cycle_and_record_trace(
                        **common_trace_kwargs,
                        verifier_accepted_length=acc_len,
                        correction_or_bonus_id=bonus_token_id,
                    )
                )

            if req.output_ids:
                new_verified_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                # If no token was appended in this verify step, keep the current token unchanged.
                new_verified_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DFLASH verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_verified_list.append(new_verified_token)
            accept_length_per_req_cpu.append(accepted_draft_tokens)
            req.spec_verify_ct += 1
            req.spec_accepted_tokens += accept_length_per_req_cpu[-1]
            req.update_spec_acceptance_histogram(accept_length_per_req_cpu[-1])

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_verified_id = torch.tensor(
            new_verified_list, dtype=torch.int64, device=device
        )

        # Free uncommitted KV cache slots and compact out_cache_loc.
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
            keep_slots = _compute_paged_keep_slots(
                prefix_lens=batch.seq_lens,
                commit_lens=commit_lens,
                draft_token_num=self.draft_token_num,
                page_size=page_size,
            )
            free_mask = row_offsets >= keep_slots[:, None]
            batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])

            keep_mask = row_offsets < commit_lens[:, None]
            batch.out_cache_loc = out_cache_loc[keep_mask]

        # Update req-level KV cache accounting.
        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

        # Update req_to_token pool mapping for newly committed tokens.
        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        # Update batch seq lens.
        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        # Keep seq_lens_sum in sync; flashinfer indices updaters rely on this for buffer sizing.
        batch.seq_lens_sum += sum(commit_lens_cpu)

        # Build next-step context features from the committed verify-input tokens.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, self.draft_token_num, -1)
        segments: List[torch.Tensor] = []
        for i, ln in enumerate(commit_lens_cpu):
            if ln > 0:
                segments.append(hidden[i, :ln, :])
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]

        # Avoid confusing downstream consumers (spec-v1 decode doesn't use this).
        logits_output.hidden_states = None

        return (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
            commit_lens_cpu,
        )
