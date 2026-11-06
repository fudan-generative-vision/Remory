import logging
import math
import os
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from typing import Optional, Union

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import evict_from_tree_cache, get_last_loc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.models.dflow_token_embedding_contract import (
    validate_token_embedding_objective_abi,
)
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.dflash_info import (
    DFlashDraftInput,
    DFlashVerifyInput,
    build_collaborative_verifier_prefill_cycle,
    gather_prefill_verifier_behavior_logprobs,
)
from sglang.srt.speculative.dflash_timing import DFlashCudaTimingProfiler
from sglang.srt.speculative.dflash_utils import (
    append_joint_judge_accepted_prefix_,
    can_dflash_use_fused_qkv_proj,
    compact_active_dflash_verify_tensors,
    compute_joint_judge_accept_len,
    get_active_dflash_verify_width_buckets,
    is_dflash_sampling_verify_available,
    normalize_joint_judge_decoder_mode,
    parse_dflash_draft_config,
    resolve_dflash_verify_mask_policy,
    select_active_dflash_verify_width,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_FusedKVMaterializeHelper = None

_STRICT_ONE_FORWARD_ENV = "SGLANG_DFLASH_STRICT_ONE_FORWARD"
_DFLOW_STRICT_ONE_BACKBONE_ENV = "SGLANG_DFLOW_STRICT_ONE_BACKBONE"
_DISABLE_FUSED_KV_ENV = "SGLANG_DFLASH_DISABLE_FUSED_KV_MATERIALIZATION"
_TRACE_DRAFT_HIDDEN_ENV = "SGLANG_DFLASH_TRACE_DRAFT_HIDDEN"
_TRAINING_PROPOSAL_TOPK_ENV = "SGLANG_DFLASH_TRAINING_PROPOSAL_TOPK"
_TRAINING_PROPOSAL_TEMPERATURE_ENV = (
    "SGLANG_DFLASH_TRAINING_PROPOSAL_TEMPERATURE"
)
_COLLABORATIVE_GATE_MODE_ENV = "SGLANG_DFLASH_COLLABORATIVE_GATE_MODE"
_COLLABORATIVE_FIXED_DRAFT_TOKENS_ENV = (
    "SGLANG_DFLASH_COLLABORATIVE_FIXED_DRAFT_TOKENS"
)
_VALIDATE_JUDGE_GRAPH_STEPS_ENV = "SGLANG_DFLASH_VALIDATE_JUDGE_GRAPH_STEPS"
_FLOW_MODE_ENV = "SGLANG_DFLASH_FLOW_MODE"
_FLOW_SOLVER_ENV = "SGLANG_DFLASH_FLOW_SOLVER"
_FLOW_NOISE_SAMPLES_ENV = "SGLANG_DFLASH_FLOW_NOISE_SAMPLES"
_FLOW_NOISE_AGGREGATION_ENV = "SGLANG_DFLASH_FLOW_NOISE_AGGREGATION"
_DFLOW_SAME_TIME_SELF_COND_CORRECTION_ENV = (
    "SGLANG_DFLOW_SAME_TIME_SELF_COND_CORRECTION"
)
_CONTEXTUAL_JIT_NFE1_TRUNCATION_ENV = (
    "SGLANG_DFLOW_ALLOW_CONTEXTUAL_JIT_NFE1_TRUNCATION"
)
_CONTEXTUAL_JIT_NFE32_EULER_RESOLUTION_ABLATION_ENV = (
    "SGLANG_DFLOW_ALLOW_CONTEXTUAL_JIT_NFE32_EULER_RESOLUTION_ABLATION"
)
_CONTEXTUAL_JIT_SAME_STATE_SC_DUAL_PROPOSAL_ENV = (
    "SGLANG_DFLOW_CONTEXTUAL_JIT_SAME_STATE_SC_DUAL_PROPOSAL"
)
_CONTEXTUAL_JIT_NFE_ENDPOINT_CANDIDATE_UNION_ENV = (
    "SGLANG_DFLOW_CONTEXTUAL_JIT_NFE_ENDPOINT_CANDIDATE_UNION"
)


def _normalize_dflash_flow_mode(raw_mode, *, source: str) -> str:
    if raw_mode is None:
        raw_mode = "off"
    flow_mode = str(raw_mode).strip().lower().replace("-", "_")
    if flow_mode not in {
        "off",
        "denoise",
        "denoise_decode",
        "one_step",
        "token_embedding_elf",
    }:
        raise ValueError(
            f"Unsupported DFLASH flow mode from {source}. Expected off, "
            "denoise, denoise_decode, one_step, or token_embedding_elf; "
            f"got {flow_mode!r}."
        )
    return flow_mode


def _normalize_dflash_flow_solver(raw_solver, *, source: str) -> str:
    """Normalize the continuous-token flow integrator.

    ``flow_steps`` always denotes *physical draft-model evaluations*, rather
    than time intervals.  Consequently ``heun`` consumes two evaluations per
    interval, while the historical ``euler`` solver consumes one.  Keeping
    this distinction here makes NFE comparisons auditable in the server log
    and the per-proposal forward counter.
    """

    if raw_solver is None:
        raw_solver = "euler"
    solver = str(raw_solver).strip().lower().replace("-", "_")
    if solver not in {"euler", "heun"}:
        raise ValueError(
            f"Unsupported DFLASH flow solver from {source}. Expected euler or "
            f"heun; got {solver!r}."
        )
    return solver


def _resolve_dflash_flow_mode(
    *,
    server_mode,
    server_mode_from_cli: bool,
    environ: Optional[Mapping] = None,
) -> str:
    """Resolve flow mode with explicit CLI/config taking precedence over env.

    The environment variable predates the CLI option and remains a fallback
    only when argparse supplied its implicit default. This prevents a stale
    process environment from changing an explicit ``--...-flow-mode off``.
    """

    if environ is None:
        environ = os.environ
    if not isinstance(server_mode_from_cli, bool):
        raise ValueError(
            "DFLASH flow-mode CLI provenance must be boolean, got "
            f"{server_mode_from_cli!r}."
        )
    if server_mode_from_cli or _FLOW_MODE_ENV not in environ:
        return _normalize_dflash_flow_mode(server_mode, source="server args")
    return _normalize_dflash_flow_mode(
        environ[_FLOW_MODE_ENV], source=_FLOW_MODE_ENV
    )


def _lookup_dflow_verified_root_embedding(
    embed_module,
    root_ids: torch.Tensor,
    *,
    batch_size: int,
    hidden_size: int,
) -> torch.Tensor:
    """Perform the sole target-embedding lookup for one base proposal batch."""

    if root_ids.ndim != 1 or int(root_ids.numel()) != int(batch_size):
        raise RuntimeError(
            "Verified-root token IDs must have shape [base_batch], got "
            f"{tuple(root_ids.shape)} for batch={batch_size}."
        )
    # This lookup deliberately precedes any M-noise expansion.  Besides being
    # cheaper, base-batch shape is part of the VRC serving ABI and prevents a
    # virtual sample row from acquiring a different root.
    with torch.inference_mode():
        root_embedding = embed_module(root_ids).detach()
    expected_shape = (int(batch_size), int(hidden_size))
    if tuple(root_embedding.shape) != expected_shape:
        raise RuntimeError(
            "Target embedding returned the wrong verified-root shape: expected "
            f"{expected_shape}, got {tuple(root_embedding.shape)}."
        )
    return root_embedding


def _validate_dflow_verified_root_algorithm(
    *,
    enabled: bool,
    speculative_algorithm,
) -> None:
    """Restrict the first VRC serving ABI to its audited linear-block path."""

    if not isinstance(enabled, bool):
        raise RuntimeError(
            "DFlow verified-root runtime marker must resolve to a boolean."
        )
    if not enabled:
        return
    algorithm = str(speculative_algorithm or "").strip().upper()
    if algorithm != "DFLASH":
        raise ValueError(
            "Verified-root DFlow currently supports only "
            "--speculative-algorithm DFLASH; got "
            f"{speculative_algorithm!r}. DDTREE is rejected because its "
            "post-verify path does not refresh the contextual h[p-1] root."
        )


def _validate_dflow_one_step_noise_contract(
    hf_config,
    runtime_noise_scale: float,
) -> float:
    """Require an exact checkpoint/runtime sigma match for one-step DFlow."""

    raw_dflow_config = getattr(hf_config, "dflow_config", None)
    if raw_dflow_config is None:
        raise ValueError(
            "One-step DFlow checkpoint is missing hf_config.dflow_config; "
            "cannot validate its input-noise contract."
        )
    if isinstance(raw_dflow_config, Mapping):
        dflow_config = raw_dflow_config
    else:
        try:
            dflow_config = dict(raw_dflow_config)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "One-step DFlow hf_config.dflow_config must be a mapping; "
                f"got {type(raw_dflow_config).__name__}."
            ) from exc
    if "one_step_input_noise_scale" not in dflow_config:
        raise ValueError(
            "One-step DFlow checkpoint dflow_config is missing required "
            "one_step_input_noise_scale."
        )

    def validate_value(value, *, source: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"One-step DFlow {source} noise scale must be a real number, "
                f"got {value!r}."
            )
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"One-step DFlow {source} noise scale must be non-negative "
                f"and finite, got {value!r}."
            )
        return value

    checkpoint_noise_scale = validate_value(
        dflow_config["one_step_input_noise_scale"],
        source="checkpoint",
    )
    runtime_noise_scale = validate_value(
        runtime_noise_scale,
        source="runtime",
    )
    if not math.isclose(
        checkpoint_noise_scale,
        runtime_noise_scale,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(
            "One-step DFlow checkpoint/runtime noise-scale mismatch: "
            f"checkpoint one_step_input_noise_scale={checkpoint_noise_scale}, "
            f"runtime flow_noise_scale={runtime_noise_scale}."
        )
    return checkpoint_noise_scale


def _draft_model_requires_router_target_hidden(
    *,
    draft_model,
    is_dflow_model: bool,
    clean_latent_source: str,
) -> bool:
    """Whether proposal construction consumes one target row per request."""

    return bool(
        (
            bool(is_dflow_model)
            and str(clean_latent_source) == "target_last_hidden"
        )
        or callable(getattr(draft_model, "select_experts", None))
    )


def _reshape_dflow_noise_sample_hidden(
    hidden_states: torch.Tensor,
    *,
    noise_samples: int,
    batch_size: int,
    block_size: int,
) -> torch.Tensor:
    """Expose folded one-step outputs as ``[M, batch, block, hidden]``.

    Keeping this sample axis intact until the aggregation boundary lets a
    candidate-union decoder reuse the same one-backbone result without
    reconstructing samples from logits.
    """

    noise_samples = int(noise_samples)
    batch_size = int(batch_size)
    block_size = int(block_size)
    if noise_samples <= 0 or batch_size <= 0 or block_size <= 0:
        raise ValueError(
            "DFlow sample hidden dimensions must be positive, got "
            f"M={noise_samples}, batch={batch_size}, block={block_size}."
        )
    if hidden_states.ndim != 2:
        raise RuntimeError(
            "DFlow folded backbone hidden states must be 2-D, got "
            f"shape={tuple(hidden_states.shape)}."
        )
    expected_rows = noise_samples * batch_size * block_size
    if int(hidden_states.shape[0]) != expected_rows:
        raise RuntimeError(
            "DFlow folded backbone hidden row mismatch: expected "
            f"{expected_rows} (= {noise_samples} * {batch_size} * {block_size}), "
            f"got {tuple(hidden_states.shape)}."
        )
    return hidden_states.view(
        noise_samples, batch_size, block_size, int(hidden_states.shape[-1])
    )


def _mean_dflow_sample_latents(sample_hidden_states: torch.Tensor) -> torch.Tensor:
    """Average predicted latents, deliberately before decode norm / LM head."""

    if sample_hidden_states.ndim != 4:
        raise RuntimeError(
            "DFlow latent aggregation expects [M, batch, block, hidden], got "
            f"shape={tuple(sample_hidden_states.shape)}."
        )
    if int(sample_hidden_states.shape[0]) == 1:
        # Preserve the exact M=1 serving path: this is a view, not a reduction.
        return sample_hidden_states[0]
    return sample_hidden_states.mean(dim=0)


def _expand_dflow_ensemble_req_pool_indices(
    base_req_pool_indices: torch.Tensor,
    *,
    base_pool_size: int,
    noise_samples: int,
) -> torch.Tensor:
    """Map sample-major folded rows onto disjoint private request-table rows."""

    base_pool_size = int(base_pool_size)
    noise_samples = int(noise_samples)
    if base_pool_size <= 0 or noise_samples <= 0:
        raise ValueError(
            "DFlow ensemble request layout requires positive pool size and "
            f"sample count, got pool={base_pool_size}, samples={noise_samples}."
        )
    base = base_req_pool_indices.to(dtype=torch.int64)
    if base.ndim != 1:
        raise RuntimeError(
            "DFlow ensemble request indices must be 1-D, got "
            f"shape={tuple(base.shape)}."
        )
    if bool(((base < 0) | (base >= base_pool_size)).any().item()):
        raise RuntimeError(
            "DFlow base request index is outside the target request pool: "
            f"indices={base.tolist()}, pool_size={base_pool_size}."
        )
    offsets = (
        torch.arange(noise_samples, device=base.device, dtype=torch.int64)
        * base_pool_size
    )
    return (base.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)


def _strict_one_forward_enabled(environ=None) -> bool:
    """Parse the opt-in strict serving contract without silently accepting typos."""
    if environ is None:
        environ = os.environ
    raw_value = str(environ.get(_STRICT_ONE_FORWARD_ENV, "")).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_STRICT_ONE_FORWARD_ENV} must be unset, 0, or 1; got {raw_value!r}."
    )


def _dflow_strict_one_backbone_enabled(environ=None) -> bool:
    """Parse the explicit DFlow one-step physical-backbone contract."""
    if environ is None:
        environ = os.environ
    raw_value = str(environ.get(_DFLOW_STRICT_ONE_BACKBONE_ENV, "")).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_DFLOW_STRICT_ONE_BACKBONE_ENV} must be unset, 0, or 1; "
        f"got {raw_value!r}."
    )


def _dflow_same_time_self_cond_correction_enabled(environ=None) -> bool:
    """Parse the opt-in, training-style same-time self-conditioning corrector."""
    if environ is None:
        environ = os.environ
    raw_value = str(
        environ.get(_DFLOW_SAME_TIME_SELF_COND_CORRECTION_ENV, "")
    ).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_DFLOW_SAME_TIME_SELF_COND_CORRECTION_ENV} must be unset, 0, or 1; "
        f"got {raw_value!r}."
    )


def _contextual_jit_nfe1_truncation_enabled(environ=None) -> bool:
    """Parse the explicit NFE=1 serving-resolution switch for contextual JiT.

    A contextual-hidden continuous-JiT checkpoint records its intended Euler
    resolution in the checkpoint ABI.  Its JiT denoiser is trained to predict
    the endpoint from a continuous range of noisy states, so a single
    t=0-to-endpoint call is semantically valid.  The switch remains explicit
    to prevent an accidental resolution change through a stale environment
    variable or a generic launcher.
    """

    if environ is None:
        environ = os.environ
    raw_value = str(
        environ.get(_CONTEXTUAL_JIT_NFE1_TRUNCATION_ENV, "")
    ).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_CONTEXTUAL_JIT_NFE1_TRUNCATION_ENV} must be unset, 0, or 1; "
        f"got {raw_value!r}."
    )


def _contextual_jit_nfe32_euler_resolution_ablation_enabled(environ=None) -> bool:
    """Parse the deliberately narrow NFE=32 Euler-resolution ablation switch.

    The contextual-hidden JiT checkpoint declares its native serving NFE in
    its checkpoint ABI.  NFE=32 is useful to measure whether a finer Euler
    grid improves this continuous-time endpoint predictor, but it is not a
    checkpoint-native serving configuration.  Keep the override opt-in and
    exact so an arbitrary ``flow_steps`` value cannot silently be reported as
    train/serve matched.
    """

    if environ is None:
        environ = os.environ
    raw_value = str(
        environ.get(_CONTEXTUAL_JIT_NFE32_EULER_RESOLUTION_ABLATION_ENV, "")
    ).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_CONTEXTUAL_JIT_NFE32_EULER_RESOLUTION_ABLATION_ENV} must be "
        f"unset, 0, or 1; got {raw_value!r}."
    )


def _contextual_jit_same_state_sc_dual_proposal_enabled(environ=None) -> bool:
    """Parse the explicit contextual-JiT SC dual-proposal switch.

    This is deliberately independent of the DFlow noise-ensemble controls.
    It produces two semantically distinct predictions at the *same* ``(z, t)``:
    an unconditioned pilot and a pilot-conditioned correction.  DDTree consumes
    them as two sibling candidate chains, rather than treating them as two
    independent Gaussian samples.
    """

    if environ is None:
        environ = os.environ
    raw_value = str(
        environ.get(_CONTEXTUAL_JIT_SAME_STATE_SC_DUAL_PROPOSAL_ENV, "")
    ).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_CONTEXTUAL_JIT_SAME_STATE_SC_DUAL_PROPOSAL_ENV} must be unset, "
        f"0, or 1; got {raw_value!r}."
    )


def _contextual_jit_nfe_endpoint_candidate_union_enabled(environ=None) -> bool:
    """Parse an opt-in endpoint-ensemble verifier topology for contextual JiT.

    A K-step contextual JiT Euler rollout produces an x_1 estimate at every
    denoiser evaluation. This switch preserves all K *endpoint predictions*
    (rather than intermediate z_{t+1} states) as sibling candidate chains and
    lets DDTree verify their prefix union in one target forward.
    """

    if environ is None:
        environ = os.environ
    raw_value = str(
        environ.get(_CONTEXTUAL_JIT_NFE_ENDPOINT_CANDIDATE_UNION_ENV, "")
    ).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_CONTEXTUAL_JIT_NFE_ENDPOINT_CANDIDATE_UNION_ENV} must be unset, "
        f"0, or 1; got {raw_value!r}."
    )


def _disable_fused_kv_materialization(environ=None) -> bool:
    """Opt out of the fused context-KV path for numerical diagnosis."""
    if environ is None:
        environ = os.environ
    raw_value = str(environ.get(_DISABLE_FUSED_KV_ENV, "")).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_DISABLE_FUSED_KV_ENV} must be unset, 0, or 1; got {raw_value!r}."
    )


def _trace_draft_hidden_enabled(environ=None) -> bool:
    """Enable bounded training traces of the hidden rows consumed by the LM head."""
    if environ is None:
        environ = os.environ
    raw_value = str(environ.get(_TRACE_DRAFT_HIDDEN_ENV, "")).strip()
    if raw_value in {"", "0"}:
        return False
    if raw_value == "1":
        return True
    raise ValueError(
        f"{_TRACE_DRAFT_HIDDEN_ENV} must be unset, 0, or 1; got {raw_value!r}."
    )


def _judge_graph_validation_steps(environ=None) -> int:
    """Return the bounded number of graph/eager judge shadow comparisons."""
    if environ is None:
        environ = os.environ
    raw_value = str(environ.get(_VALIDATE_JUDGE_GRAPH_STEPS_ENV, "0")).strip()
    try:
        steps = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{_VALIDATE_JUDGE_GRAPH_STEPS_ENV} must be a non-negative integer; "
            f"got {raw_value!r}."
        ) from exc
    if steps < 0:
        raise ValueError(
            f"{_VALIDATE_JUDGE_GRAPH_STEPS_ENV} must be a non-negative integer; "
            f"got {steps}."
        )
    return steps


def _training_proposal_sampling_config(environ=None) -> tuple[Optional[int], float]:
    """Parse the opt-in truncated-categorical proposal policy.

    ``topk=None`` is the default deterministic greedy path. Values >=2 enable
    training-only sampling with exact behavior log probabilities.
    """

    if environ is None:
        environ = os.environ
    raw_topk = str(environ.get(_TRAINING_PROPOSAL_TOPK_ENV, "")).strip()
    if raw_topk in {"", "0"}:
        topk = None
    else:
        try:
            topk = int(raw_topk)
        except ValueError as exc:
            raise ValueError(
                f"{_TRAINING_PROPOSAL_TOPK_ENV} must be 0 or an integer >= 2; "
                f"got {raw_topk!r}."
            ) from exc
        if topk < 2:
            raise ValueError(
                f"{_TRAINING_PROPOSAL_TOPK_ENV} must be 0 or >= 2; got {topk}."
            )

    raw_temperature = str(
        environ.get(_TRAINING_PROPOSAL_TEMPERATURE_ENV, "1.0")
    ).strip()
    try:
        temperature = float(raw_temperature)
    except ValueError as exc:
        raise ValueError(
            f"{_TRAINING_PROPOSAL_TEMPERATURE_ENV} must be positive and finite; "
            f"got {raw_temperature!r}."
        ) from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"{_TRAINING_PROPOSAL_TEMPERATURE_ENV} must be positive and finite; "
            f"got {temperature}."
        )
    return topk, temperature


def _collaborative_source_gate_config(
    environ=None,
) -> tuple[Optional[str], int]:
    """Parse the explicit composite-policy gate; unset preserves strict mode."""

    if environ is None:
        environ = os.environ
    raw_mode = str(environ.get(_COLLABORATIVE_GATE_MODE_ENV, "")).strip().lower()
    raw_fixed = str(
        environ.get(_COLLABORATIVE_FIXED_DRAFT_TOKENS_ENV, "1")
    ).strip()
    try:
        fixed_draft_tokens = int(raw_fixed)
    except ValueError as exc:
        raise ValueError(
            f"{_COLLABORATIVE_FIXED_DRAFT_TOKENS_ENV} must be a non-negative "
            f"integer, got {raw_fixed!r}."
        ) from exc
    if fixed_draft_tokens < 0:
        raise ValueError(
            f"{_COLLABORATIVE_FIXED_DRAFT_TOKENS_ENV} must be non-negative, "
            f"got {fixed_draft_tokens}."
        )

    if raw_mode in {"", "0", "off", "strict"}:
        return None, fixed_draft_tokens
    if raw_mode == "greedy_match":
        return "greedy_match", fixed_draft_tokens
    if raw_mode == "fixed":
        return "fixed", fixed_draft_tokens
    if raw_mode.startswith("fixed:"):
        raw_inline = raw_mode.partition(":")[2]
        try:
            inline_fixed = int(raw_inline)
        except ValueError as exc:
            raise ValueError(
                f"{_COLLABORATIVE_GATE_MODE_ENV}=fixed:N requires a "
                f"non-negative integer N, got {raw_inline!r}."
            ) from exc
        if inline_fixed < 0:
            raise ValueError(
                f"{_COLLABORATIVE_GATE_MODE_ENV}=fixed:N requires N >= 0, "
                f"got {inline_fixed}."
            )
        return "fixed", inline_fixed
    raise ValueError(
        f"{_COLLABORATIVE_GATE_MODE_ENV} must be unset, strict, "
        f"greedy_match, fixed, or fixed:N; got {raw_mode!r}."
    )


def _renormalize_topk_behavior_log_probs(
    topk_log_probs: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    """Return exact log q for temperature-scaled, top-k-truncated sampling."""

    if topk_log_probs.ndim != 2 or int(topk_log_probs.shape[-1]) <= 0:
        raise ValueError(
            "topk_log_probs must have shape [tokens, candidates], got "
            f"{tuple(topk_log_probs.shape)}."
        )
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}.")
    scaled = topk_log_probs.float() / float(temperature)
    return scaled - torch.logsumexp(scaled, dim=-1, keepdim=True)


def _validate_strict_one_forward_runtime_options(
    *,
    enabled: bool,
    flow_mode: str,
    draft_window_size: Optional[int],
    pos_mask_embeddings_path: Optional[str],
) -> None:
    """Reject serving-only proposal transforms before the draft model is loaded."""
    if not enabled:
        return

    violations = []
    if flow_mode != "off":
        violations.append(f"flow_mode must be 'off', got {flow_mode!r}")
    if draft_window_size is not None:
        violations.append(f"draft_window_size must be None, got {draft_window_size!r}")
    if pos_mask_embeddings_path is not None and str(pos_mask_embeddings_path).strip():
        violations.append(
            "SGLANG_DFLASH_POS_MASK_EMB_PATH must be unset so serving uses the "
            "checkpoint MASK embedding"
        )
    if violations:
        raise ValueError(
            "Strict one-forward DFLASH runtime contract failed: "
            + "; ".join(violations)
        )


def _validate_dflow_one_backbone_runtime_options(
    *,
    enabled: bool,
    native_strict_enabled: bool,
    flow_mode: str,
    speculative_algorithm,
    draft_window_size: Optional[int],
    pos_mask_embeddings_path: Optional[str],
) -> None:
    """Validate the DFlow ABI without invoking Native MASK/checkpoint rules."""
    if not enabled:
        return
    violations = []
    if native_strict_enabled:
        violations.append(f"{_STRICT_ONE_FORWARD_ENV} must be 0")
    if flow_mode != "one_step":
        violations.append(f"flow_mode must be 'one_step', got {flow_mode!r}")
    if str(speculative_algorithm or "").strip().upper() != "DFLASH":
        violations.append(
            "speculative_algorithm must be DFLASH, got "
            f"{speculative_algorithm!r}"
        )
    if draft_window_size is not None:
        violations.append(f"draft_window_size must be None, got {draft_window_size!r}")
    if pos_mask_embeddings_path is not None and str(pos_mask_embeddings_path).strip():
        violations.append("SGLANG_DFLASH_POS_MASK_EMB_PATH must be unset")
    if violations:
        raise ValueError(
            "Strict one-backbone DFlow runtime contract failed: "
            + "; ".join(violations)
        )


def _validate_dflow_one_backbone_checkpoint_contract(
    *,
    enabled: bool,
    is_dflow_model: bool,
    runtime_one_step_direct: bool,
    dflow_config,
    flow_steps: int,
    active_judge_enabled: bool,
) -> None:
    """Bind the explicit DFlow runtime assertion to its checkpoint ABI."""
    if not enabled:
        return
    violations = []
    if not is_dflow_model:
        violations.append("draft checkpoint must be a DFlow model")
    if runtime_one_step_direct is not True:
        violations.append("runtime model one_step_direct must be exactly true")
    if not isinstance(dflow_config, Mapping):
        violations.append("checkpoint dflow_config must be a mapping")
    else:
        checkpoint_one_step_direct = dflow_config.get("one_step_direct")
        if checkpoint_one_step_direct is not True:
            violations.append("checkpoint one_step_direct must be exactly true")
        serving_drafter_forwards = dflow_config.get("serving_drafter_forwards")
        if (
            isinstance(serving_drafter_forwards, bool)
            or not isinstance(serving_drafter_forwards, int)
            or serving_drafter_forwards != 1
        ):
            violations.append(
                "checkpoint serving_drafter_forwards must be integer 1, got "
                f"{serving_drafter_forwards!r}"
            )
    if isinstance(flow_steps, bool) or not isinstance(flow_steps, int) or flow_steps != 1:
        violations.append(f"flow_steps must be 1, got {flow_steps}")
    if active_judge_enabled:
        violations.append("active multi-block judge must be disabled")
    if violations:
        raise ValueError(
            "Strict one-backbone DFlow checkpoint contract failed: "
            + "; ".join(violations)
        )


def _validate_token_embedding_elf_runtime_contract(
    *,
    enabled: bool,
    speculative_algorithm,
    flow_mode: str,
    flow_steps: int,
    flow_t_min: float,
    flow_t_max: float,
    flow_t_eps: float,
    flow_noise_scale: float,
    flow_noise_samples: int,
    flow_noise_aggregation: str,
    flow_schedule: str,
    block_size: int,
    draft_window_size: Optional[int],
    active_judge_enabled: bool,
    strict_native_enabled: bool,
    strict_one_backbone_enabled: bool,
    draft_hf_config,
    target_hf_config,
    target_input_embeddings,
) -> None:
    """Fail closed on every train/serve invariant of token-embedding ELF."""

    if not enabled:
        return
    dflow_config = getattr(draft_hf_config, "dflow_config", None)
    violations: list[str] = []
    if not isinstance(dflow_config, Mapping):
        violations.append("checkpoint dflow_config must be a mapping")
        dflow_config = {}

    algorithm = str(speculative_algorithm or "").strip().upper()
    if algorithm not in {"DFLASH", "DDTREE"}:
        violations.append(
            "speculative_algorithm must be DFLASH or DDTREE, got "
            f"{speculative_algorithm!r}"
        )
    if flow_mode != "token_embedding_elf":
        violations.append(
            f"flow_mode must be 'token_embedding_elf', got {flow_mode!r}"
        )
    if int(flow_steps) not in {1, 2, 4, 8, 16, 32, 64}:
        violations.append(
            f"NFE must be one of 1, 2, 4, 8, 16, 32, 64, got {flow_steps}"
        )
    exact_runtime = {
        "flow_t_min": (float(flow_t_min), 0.0),
        "flow_t_max": (float(flow_t_max), 1.0),
        "flow_t_eps": (float(flow_t_eps), 0.05),
        "flow_noise_scale": (float(flow_noise_scale), 2.0),
        "flow_noise_samples": (int(flow_noise_samples), 1),
        "flow_noise_aggregation": (str(flow_noise_aggregation), "latent_mean"),
        "flow_schedule": (str(flow_schedule), "linear"),
        # The checkpoint owns its proposal width.  Ordinary token-ELF uses
        # B=16 while the fixed-MASK ablation intentionally uses B=32; do not
        # silently coerce either one at serving time.
        "block_size": (int(block_size), int(getattr(draft_hf_config, "block_size", -1))),
    }
    for name, (actual, expected) in exact_runtime.items():
        if actual != expected:
            violations.append(f"{name} must be {expected!r}, got {actual!r}")
    if draft_window_size is not None:
        violations.append("draft_window_size must be disabled")
    if active_judge_enabled:
        violations.append("active multi-block judge must be disabled")
    if strict_native_enabled:
        violations.append(f"{_STRICT_ONE_FORWARD_ENV} must be 0")
    if strict_one_backbone_enabled:
        violations.append(f"{_DFLOW_STRICT_ONE_BACKBONE_ENV} must be 0")

    checkpoint_contract = {
        "clean_latent_source": "token_embedding",
        "decode_adapter_type": "identity",
        "layerwise_noise_conditioning": False,
        "normalization_type": "scalar",
        "one_step_direct": False,
        "output_norm_type": "identity",
        "retrieval_temperature": 1.0,
        "target_embedding_contract": "raw_tied_input_embedding",
        "token_embedding_method": "elf",
        "token_embedding_output": True,
    }
    for name, expected in checkpoint_contract.items():
        actual = dflow_config.get(name)
        if actual != expected:
            violations.append(
                f"checkpoint {name} must be {expected!r}, got {actual!r}"
            )
    fixed_mask_source = (
        dflow_config.get("initial_state_type")
        == "frozen_target_mask_embedding"
    )
    if fixed_mask_source:
        fixed_mask_contract = {
            "training_input_source": "frozen_target_mask_embedding",
            "serving_rollout_start_state": (
                "verified_anchor_plus_frozen_target_mask_embedding"
            ),
            "training_input_noise_active": False,
            "serving_input_noise_active": False,
            "source_mask_trainable": False,
            "source_mask_storage": "persistent_non_parameter_buffer",
            "source_mask_coordinate_space": "normalized_target_token_embedding",
        }
        for name, expected in fixed_mask_contract.items():
            actual = dflow_config.get(name)
            if actual != expected:
                violations.append(
                    f"fixed-MASK checkpoint {name} must be {expected!r}, "
                    f"got {actual!r}"
                )
        mask_token_id = dflow_config.get("mask_token_id")
        if isinstance(mask_token_id, bool) or not isinstance(mask_token_id, int):
            violations.append(
                "fixed-MASK checkpoint mask_token_id must be an integer, got "
                f"{mask_token_id!r}"
            )
    try:
        validate_token_embedding_objective_abi(
            dflow_config,
            runtime_flow_steps=flow_steps,
        )
    except ValueError as exc:
        violations.append(str(exc))
    serving_self_conditioning = dflow_config.get(
        "serving_self_conditioning", True
    )
    if not isinstance(serving_self_conditioning, bool):
        violations.append(
            "checkpoint serving_self_conditioning must be a JSON boolean, got "
            f"{dflow_config.get('serving_self_conditioning')!r}"
        )
    if dflow_config.get("serving_denoiser_noise_scale") != 2.0:
        violations.append(
            "checkpoint serving_denoiser_noise_scale must be 2.0, got "
            f"{dflow_config.get('serving_denoiser_noise_scale')!r}"
        )
    if dflow_config.get("serving_t_eps") != 0.05:
        violations.append(
            "checkpoint serving_t_eps must be 0.05, got "
            f"{dflow_config.get('serving_t_eps')!r}"
        )

    valid_vocab_size = dflow_config.get("valid_vocab_size")
    if isinstance(valid_vocab_size, bool) or not isinstance(valid_vocab_size, int):
        violations.append(
            f"checkpoint valid_vocab_size must be an integer, got {valid_vocab_size!r}"
        )
        valid_vocab_size = -1
    target_vocab_size = int(getattr(target_hf_config, "vocab_size", -1))
    if not 1 <= int(valid_vocab_size) <= target_vocab_size:
        violations.append(
            "valid_vocab_size must fit target vocab: "
            f"valid={valid_vocab_size}, target={target_vocab_size}"
        )
    if getattr(target_hf_config, "tie_word_embeddings", None) is not True:
        violations.append("target tie_word_embeddings must be exactly true")
    if not hasattr(target_input_embeddings, "weight") or not hasattr(
        target_input_embeddings, "shard_indices"
    ):
        violations.append(
            "target input embeddings must expose TP-sharded weight and shard_indices"
        )
    else:
        embed_vocab_size = int(
            getattr(target_input_embeddings, "org_vocab_size", target_vocab_size)
        )
        if embed_vocab_size != target_vocab_size:
            violations.append(
                "target input embedding rows differ from config vocab: "
                f"embedding={embed_vocab_size}, config={target_vocab_size}"
            )

    if violations:
        raise ValueError(
            "Token-embedding ELF train/serve contract failed: "
            + "; ".join(violations)
        )


def _validate_dflow_same_time_self_cond_correction(
    *,
    enabled: bool,
    flow_mode: str,
    flow_steps: int,
    speculative_algorithm,
) -> None:
    """Fail closed outside the one-step linear DFLASH experiment.

    The corrector adds a same-(z, t) pilot call, so a solver configured with
    one Euler step performs two physical draft-model evaluations.
    """
    if not enabled:
        return
    violations = []
    if flow_mode != "token_embedding_elf":
        violations.append(
            f"flow_mode must be 'token_embedding_elf', got {flow_mode!r}"
        )
    if isinstance(flow_steps, bool) or int(flow_steps) != 1:
        violations.append(f"flow_steps must be 1, got {flow_steps!r}")
    if str(speculative_algorithm or "").strip().upper() != "DFLASH":
        violations.append(
            "speculative_algorithm must be DFLASH, got "
            f"{speculative_algorithm!r}"
        )
    if violations:
        raise ValueError(
            "Same-time DFlow self-conditioning correction contract failed: "
            + "; ".join(violations)
        )


def _validate_strict_one_forward_checkpoint_contract(
    *,
    enabled: bool,
    runtime_block_size: int,
    config_block_size: Optional[int],
    config_mask_token_id: Optional[int],
    resolved_mask_token_id: int,
    pos_mask_embeddings_present: bool,
    configured_target_layer_ids: Optional[list[int]],
    runtime_target_layer_ids: Optional[list[int]],
) -> None:
    """Validate checkpoint fields that define the training/serving ABI."""
    if not enabled:
        return

    violations = []
    if config_block_size is None:
        violations.append("dflash_config.block_size must be explicit")
    elif int(config_block_size) != int(runtime_block_size):
        violations.append(
            "block size mismatch: "
            f"checkpoint={int(config_block_size)}, runtime={int(runtime_block_size)}"
        )

    if config_mask_token_id is None:
        violations.append("dflash_config.mask_token_id must be explicit")
    elif int(config_mask_token_id) != int(resolved_mask_token_id):
        violations.append(
            "mask token id mismatch: "
            f"checkpoint={int(config_mask_token_id)}, "
            f"runtime={int(resolved_mask_token_id)}"
        )

    if pos_mask_embeddings_present:
        violations.append("position-specific MASK embeddings must be disabled")

    if configured_target_layer_ids is None:
        violations.append("dflash_config.target_layer_ids must be explicit")
    else:
        configured_target_layer_ids = [int(x) for x in configured_target_layer_ids]
        if configured_target_layer_ids != sorted(set(configured_target_layer_ids)):
            violations.append(
                "dflash_config.target_layer_ids must be sorted and unique, got "
                f"{configured_target_layer_ids}"
            )
        if runtime_target_layer_ids is None:
            violations.append("runtime target capture layer ids are unavailable")
        else:
            runtime_target_layer_ids = [int(x) for x in runtime_target_layer_ids]
            if runtime_target_layer_ids != configured_target_layer_ids:
                violations.append(
                    "target layer id mismatch: "
                    f"checkpoint={configured_target_layer_ids}, "
                    f"runtime={runtime_target_layer_ids}"
                )

    if violations:
        raise ValueError(
            "Strict one-forward DFLASH checkpoint contract failed: "
            + "; ".join(violations)
        )


class _StrictOneForwardCounter:
    """Count an exact number of draft-runner calls per proposal transaction."""

    def __init__(
        self,
        *,
        enabled: bool,
        contract_label: str = "Strict one-forward DFLASH",
        expected_calls: int = 1,
    ):
        self.enabled = bool(enabled)
        self.contract_label = str(contract_label)
        self.expected_calls = int(expected_calls)
        if self.expected_calls <= 0:
            raise ValueError(
                f"{self.contract_label} expected_calls must be positive, got "
                f"{self.expected_calls}."
            )
        self.active = False
        self.calls = 0

    def begin(self) -> None:
        if not self.enabled:
            return
        if self.active:
            raise RuntimeError(
                f"{self.contract_label} proposal scopes must not be nested."
            )
        self.active = True
        self.calls = 0

    def record_call(self) -> None:
        if not self.enabled:
            return
        if not self.active:
            raise RuntimeError(
                f"{self.contract_label} draft runner was called outside a "
                "strict proposal scope."
            )
        self.calls += 1
        if self.calls > self.expected_calls:
            raise RuntimeError(
                f"{self.contract_label} attempted more than "
                f"{self.expected_calls} actual draft runner calls in one "
                f"proposal (calls={self.calls})."
            )

    def finish(self) -> None:
        if not self.enabled:
            return
        if not self.active:
            raise RuntimeError(f"No {self.contract_label} proposal is active.")
        calls = self.calls
        self.active = False
        if calls != self.expected_calls:
            raise RuntimeError(
                f"{self.contract_label} requires exactly "
                f"{self.expected_calls} actual draft runner calls per proposal, "
                f"got {calls}."
            )

    def abort(self) -> None:
        if self.enabled:
            self.active = False


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class DFlashWorker:
    """DFlash speculative decoding worker (spec-v1, tp>=1/pp=1)."""

    def _initialize_draft_forward_counters(self) -> None:
        """Create independent Native and DFlow physical-forward guards."""

        self._native_draft_forward_counter = _StrictOneForwardCounter(
            enabled=self._strict_one_forward_enabled,
            contract_label="Strict one-forward DFLASH",
            expected_calls=1,
        )
        self._dflow_backbone_forward_counter = _StrictOneForwardCounter(
            # one_step_direct is an immutable checkpoint ABI. Keep enforcing it
            # even outside the formal launcher; the explicit environment switch
            # additionally validates every runtime/checkpoint prerequisite.
            enabled=(
                self._dflow_strict_one_backbone_enabled
                or (
                    self._is_dflow_model
                    and (
                        self._dflow_one_step_direct
                        or self._is_dflow_token_embedding_model
                        or bool(
                            getattr(
                                self,
                                "_dflow_direct_lm_head_terminal",
                                False,
                            )
                        )
                    )
                )
            ),
            contract_label=(
                "Token-embedding ELF exact physical-forward"
                if self._is_dflow_token_embedding_model
                else (
                    "Contextual hidden JiT same-state SC dual-proposal "
                    "physical-forward"
                    if bool(
                        self.__dict__.get(
                            "_dflow_contextual_jit_same_state_sc_dual_proposal",
                            False,
                        )
                    )
                    else "Contextual hidden JiT exact physical-forward"
                    if bool(
                        getattr(self, "_dflow_direct_lm_head_terminal", False)
                    )
                    else "Strict one-backbone DFlow"
                )
            ),
            expected_calls=(
                int(self.flow_steps)
                + int(
                    bool(
                        getattr(
                            self,
                            "_same_time_self_cond_correction_enabled",
                            False,
                        )
                    )
                )
                if self._is_dflow_token_embedding_model
                else (
                    int(self.flow_steps)
                    + int(
                        bool(
                            self.__dict__.get(
                                "_dflow_contextual_jit_same_state_sc_dual_proposal",
                                False,
                            )
                        )
                    )
                    if bool(
                        getattr(self, "_dflow_direct_lm_head_terminal", False)
                    )
                    else 1
                )
            ),
        )

    def _begin_draft_forward_counters(self) -> None:
        self._native_draft_forward_counter.begin()
        self._dflow_backbone_forward_counter.begin()

    def _finish_draft_forward_counters(self) -> None:
        self._native_draft_forward_counter.finish()
        self._dflow_backbone_forward_counter.finish()

    def _abort_draft_forward_counters(self) -> None:
        self._native_draft_forward_counter.abort()
        self._dflow_backbone_forward_counter.abort()

    def _record_draft_forward_call(self) -> None:
        # Both counters record before the runner invocation, so either contract
        # blocks a second physical call rather than merely detecting it later.
        self._native_draft_forward_counter.record_call()
        self._dflow_backbone_forward_counter.record_call()

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
        self.draft_window_size: Optional[int] = (
            int(server_args.speculative_dflash_draft_window_size)
            if server_args.speculative_dflash_draft_window_size is not None
            else None
        )
        self.use_compact_draft_cache = self.draft_window_size is not None
        self.device = target_worker.device
        self._strict_one_forward_enabled = _strict_one_forward_enabled()
        self._dflow_strict_one_backbone_enabled = (
            _dflow_strict_one_backbone_enabled()
        )
        self._trace_draft_hidden_enabled = _trace_draft_hidden_enabled()
        (
            self._training_proposal_topk,
            self._training_proposal_temperature,
        ) = _training_proposal_sampling_config()
        (
            self._collaborative_gate_mode,
            self._collaborative_fixed_draft_tokens,
        ) = _collaborative_source_gate_config()

        self._warned_sampling_fallback = False
        self._logged_first_verify = False
        self.flow_mode = _resolve_dflash_flow_mode(
            server_mode=getattr(
                server_args, "speculative_dflash_flow_mode", "off"
            ),
            server_mode_from_cli=getattr(
                server_args,
                "_speculative_dflash_flow_mode_from_cli",
                False,
            ),
        )
        self._same_time_self_cond_correction_enabled = (
            _dflow_same_time_self_cond_correction_enabled()
        )
        _validate_strict_one_forward_runtime_options(
            enabled=self._strict_one_forward_enabled,
            flow_mode=self.flow_mode,
            draft_window_size=self.draft_window_size,
            pos_mask_embeddings_path=os.environ.get("SGLANG_DFLASH_POS_MASK_EMB_PATH"),
        )
        _validate_dflow_one_backbone_runtime_options(
            enabled=self._dflow_strict_one_backbone_enabled,
            native_strict_enabled=self._strict_one_forward_enabled,
            flow_mode=self.flow_mode,
            speculative_algorithm=server_args.speculative_algorithm,
            draft_window_size=self.draft_window_size,
            pos_mask_embeddings_path=os.environ.get("SGLANG_DFLASH_POS_MASK_EMB_PATH"),
        )
        self.flow_steps = int(
            os.environ.get(
                "SGLANG_DFLASH_FLOW_STEPS",
                getattr(server_args, "speculative_dflash_flow_steps", 1),
            )
        )
        if self.flow_steps <= 0:
            raise ValueError(
                "SGLANG_DFLASH_FLOW_STEPS / --speculative-dflash-flow-steps "
                f"must be positive, got {self.flow_steps}."
            )
        self.flow_solver = _normalize_dflash_flow_solver(
            os.environ.get(_FLOW_SOLVER_ENV, "euler"),
            source=_FLOW_SOLVER_ENV,
        )
        if self.flow_solver == "heun":
            if self.flow_mode != "token_embedding_elf":
                raise ValueError(
                    "Heun flow integration requires "
                    "flow_mode='token_embedding_elf', got "
                    f"{self.flow_mode!r}."
                )
            if self.flow_steps < 2 or self.flow_steps % 2 != 0:
                raise ValueError(
                    "Heun flow integration defines NFE as physical draft "
                    "forwards and therefore requires an even NFE >= 2, got "
                    f"{self.flow_steps}."
                )
        _validate_dflow_same_time_self_cond_correction(
            enabled=self._same_time_self_cond_correction_enabled,
            flow_mode=self.flow_mode,
            flow_steps=self.flow_steps,
            speculative_algorithm=server_args.speculative_algorithm,
        )
        self.flow_t_min = float(
            os.environ.get(
                "SGLANG_DFLASH_FLOW_T_MIN",
                getattr(server_args, "speculative_dflash_flow_t_min", 0.0),
            )
        )
        self.flow_t_max = float(
            os.environ.get(
                "SGLANG_DFLASH_FLOW_T_MAX",
                getattr(server_args, "speculative_dflash_flow_t_max", 0.95),
            )
        )
        if not (0.0 <= self.flow_t_min < self.flow_t_max <= 1.0):
            raise ValueError(
                "Flow-DFlash requires 0.0 <= t_min < t_max <= 1.0, "
                f"got t_min={self.flow_t_min}, t_max={self.flow_t_max}."
            )
        self.flow_t_eps = float(
            os.environ.get(
                "SGLANG_DFLASH_FLOW_T_EPS",
                getattr(server_args, "speculative_dflash_flow_t_eps", 0.05),
            )
        )
        if not math.isfinite(self.flow_t_eps) or self.flow_t_eps <= 0.0:
            raise ValueError(
                "SGLANG_DFLASH_FLOW_T_EPS / --speculative-dflash-flow-t-eps "
                f"must be positive and finite, got {self.flow_t_eps}."
            )
        self.flow_noise_scale = float(
            os.environ.get(
                "SGLANG_DFLASH_FLOW_NOISE_SCALE",
                getattr(server_args, "speculative_dflash_flow_noise_scale", 2.0),
            )
        )
        if not math.isfinite(self.flow_noise_scale) or self.flow_noise_scale < 0.0:
            raise ValueError(
                "SGLANG_DFLASH_FLOW_NOISE_SCALE / "
                "--speculative-dflash-flow-noise-scale must be non-negative and "
                f"finite, got {self.flow_noise_scale}."
            )
        try:
            self.flow_noise_samples = int(
                os.environ.get(
                    _FLOW_NOISE_SAMPLES_ENV,
                    getattr(
                        server_args,
                        "speculative_dflash_flow_noise_samples",
                        1,
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{_FLOW_NOISE_SAMPLES_ENV} / "
                "--speculative-dflash-flow-noise-samples must be a positive "
                "integer."
            ) from exc
        if self.flow_noise_samples <= 0:
            raise ValueError(
                f"{_FLOW_NOISE_SAMPLES_ENV} / "
                "--speculative-dflash-flow-noise-samples must be positive, "
                f"got {self.flow_noise_samples}."
            )
        if self.flow_noise_samples > 1 and self.flow_mode != "one_step":
            raise ValueError(
                "Multi-noise DFlow inference is supported only for "
                f"flow_mode='one_step', got mode={self.flow_mode!r}, "
                f"samples={self.flow_noise_samples}."
            )
        if self.flow_noise_samples > 1 and self.use_compact_draft_cache:
            raise ValueError(
                "Multi-noise DFlow inference is incompatible with compact draft "
                "windowing because the two features require different private "
                "request-table layouts."
            )
        self.flow_noise_aggregation = (
            str(
                os.environ.get(
                    _FLOW_NOISE_AGGREGATION_ENV,
                    getattr(
                        server_args,
                        "speculative_dflash_flow_noise_aggregation",
                        "latent_mean",
                    ),
                )
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.flow_noise_aggregation not in {
            "latent_mean",
            "candidate_union",
        }:
            raise ValueError(
                f"{_FLOW_NOISE_AGGREGATION_ENV} / "
                "--speculative-dflash-flow-noise-aggregation must be "
                "latent_mean or candidate_union, got "
                f"{self.flow_noise_aggregation!r}."
            )
        if (
            self.flow_noise_aggregation == "candidate_union"
            and str(server_args.speculative_algorithm).upper() != "DDTREE"
        ):
            raise ValueError(
                "candidate_union noise aggregation is supported only by "
                "DDTREE; plain DFLASH must use latent_mean."
            )
        if (
            self.flow_noise_aggregation == "candidate_union"
            and self.flow_noise_samples == 1
        ):
            self.flow_noise_aggregation = "latent_mean"
        self._use_flow_ensemble_req_pool = self.flow_noise_samples > 1
        self._pending_dflow_candidate_sample_hiddens: Optional[torch.Tensor] = None
        self.flow_schedule = str(
            os.environ.get(
                "SGLANG_DFLASH_FLOW_SCHEDULE",
                getattr(server_args, "speculative_dflash_flow_schedule", "linear"),
            )
        ).lower()
        if self.flow_schedule != "linear":
            raise ValueError(
                "Flow-DFlash currently only supports a linear schedule, got "
                f"{self.flow_schedule!r}."
            )
        self.elf_latest_endpoint_bridge_mix = float(
            os.environ.get(
                "SGLANG_DFLASH_ELF_LATEST_ENDPOINT_BRIDGE_MIX",
                getattr(
                    server_args,
                    "speculative_dflash_elf_latest_endpoint_bridge_mix",
                    0.0,
                ),
            )
        )
        if not math.isfinite(self.elf_latest_endpoint_bridge_mix) or not (
            0.0 <= self.elf_latest_endpoint_bridge_mix <= 1.0
        ):
            raise ValueError(
                "SGLANG_DFLASH_ELF_LATEST_ENDPOINT_BRIDGE_MIX / "
                "--speculative-dflash-elf-latest-endpoint-bridge-mix must be "
                "finite and in [0, 1], got "
                f"{self.elf_latest_endpoint_bridge_mix}."
            )
        if (
            self.elf_latest_endpoint_bridge_mix > 0.0
            and self.flow_mode != "token_embedding_elf"
        ):
            raise ValueError(
                "Latest-endpoint bridge alignment requires "
                "flow_mode='token_embedding_elf'."
            )
        self.elf_euler_velocity_scale = float(
            os.environ.get(
                "SGLANG_DFLASH_ELF_EULER_VELOCITY_SCALE",
                getattr(
                    server_args,
                    "speculative_dflash_elf_euler_velocity_scale",
                    1.0,
                ),
            )
        )
        if not math.isfinite(self.elf_euler_velocity_scale) or not (
            self.elf_euler_velocity_scale > 0.0
        ):
            raise ValueError(
                "SGLANG_DFLASH_ELF_EULER_VELOCITY_SCALE / "
                "--speculative-dflash-elf-euler-velocity-scale must be "
                "finite and > 0, got "
                f"{self.elf_euler_velocity_scale}."
            )
        if (
            self.elf_euler_velocity_scale != 1.0
            and self.flow_mode != "token_embedding_elf"
        ):
            raise ValueError(
                "ELF Euler velocity scaling requires "
                "flow_mode='token_embedding_elf'."
            )
        if self.flow_solver == "heun":
            if self.elf_latest_endpoint_bridge_mix != 0.0:
                raise ValueError(
                    "Heun flow integration is incompatible with "
                    "ELF latest-endpoint bridge mixing; set "
                    "SGLANG_DFLASH_ELF_LATEST_ENDPOINT_BRIDGE_MIX=0."
                )
            if self.elf_euler_velocity_scale != 1.0:
                raise ValueError(
                    "Heun flow integration is incompatible with Euler "
                    "velocity scaling; set "
                    "SGLANG_DFLASH_ELF_EULER_VELOCITY_SCALE=1."
                )

        # Draft runner (separate KV cache + attention backend).
        # Without draft windowing, the draft worker aliases the target request->token
        # mapping and allocation state. With draft windowing enabled, the draft worker
        # keeps a private compact req->token table over the same global KV index space,
        # so radix-cache/prefix-hit KV remains reusable while draft attention sees only
        # the recent window.
        target_req_to_token_pool, target_token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        self._flow_ensemble_base_pool_size = int(target_req_to_token_pool.size)
        shared_req_to_token_pool = (
            None
            if (
                self.use_compact_draft_cache
                or self._use_flow_ensemble_req_pool
            )
            else target_req_to_token_pool
        )
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_memory_pool_config = target_worker.model_runner.memory_pool_config
        if self._use_flow_ensemble_req_pool:
            # Only the request-to-token table is enlarged. The token allocator
            # remains shared with the target and every folded proposal block is
            # allocated transactionally below, then released by restore_state.
            draft_memory_pool_config = deepcopy(draft_memory_pool_config)
            draft_memory_pool_config.max_running_requests = (
                self._flow_ensemble_base_pool_size * self.flow_noise_samples
            )
            draft_server_args.max_running_requests = (
                draft_memory_pool_config.max_running_requests
            )
        draft_backend = draft_server_args.speculative_draft_attention_backend
        # ``torch_native`` is intentionally supported as a correctness oracle.
        # It implements ENCODER_ONLY/non-causal extend attention and is useful
        # for separating DFlash model/KV bugs from optimized backend bugs.
        supported_draft_backends = ("flashinfer", "fa3", "fa4", "torch_native")
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            draft_backend = "flashinfer"
        elif draft_backend == "trtllm_mha":
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires non-causal attention. Falling back to "
                "'flashinfer'."
            )
            draft_backend = "flashinfer"
        elif draft_backend not in supported_draft_backends:
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to 'flashinfer'.",
                supported_draft_backends,
                draft_backend,
            )
            draft_backend = "flashinfer"
        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        self._draft_attention_backend = str(draft_backend)
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
        self.draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
            req_to_token_pool=shared_req_to_token_pool,
            token_to_kv_pool_allocator=target_token_to_kv_pool_allocator,
            memory_pool_config=draft_memory_pool_config,
        )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self.draft_worker.model_runner
        self.draft_model = self.draft_model_runner.model
        self._is_dflow_model = bool(
            getattr(self.draft_model, "is_dflow_draft_model", False)
        )
        self._is_dflow_token_embedding_model = bool(
            getattr(self.draft_model, "is_dflow_token_embedding_model", False)
        )
        self._dflow_fixed_mask_initial_state = bool(
            getattr(self.draft_model, "fixed_mask_initial_state", False)
        )
        if self._is_dflow_token_embedding_model and not self._is_dflow_model:
            raise RuntimeError(
                "Token-embedding ELF runtime marker requires a DFlow model."
            )
        if self._dflow_fixed_mask_initial_state and not self._is_dflow_token_embedding_model:
            raise RuntimeError(
                "Fixed-MASK ELF runtime marker requires a token-embedding DFlow model."
            )
        # Self-conditioning is a checkpoint-owned Euler-solver contract for
        # both token-embedding and contextual-hidden DFlow.  In particular,
        # the contextual-hidden SC0 branches are trained with a zero condition
        # at *every* time point; feeding the preceding x-prediction back after
        # the first step would silently evaluate a different model.
        self._dflow_serving_self_conditioning = bool(
            getattr(self.draft_model, "serving_self_conditioning", True)
        )
        # Retain the historical attribute name for token-embedding-only code
        # below.  Contextual-hidden code uses the general contract above.
        self._token_embedding_serving_self_conditioning = (
            self._dflow_serving_self_conditioning
        )
        self._dflow_one_step_direct = bool(
            getattr(self.draft_model, "one_step_direct", False)
        )
        self._dflow_direct_lm_head_terminal = bool(
            getattr(self.draft_model, "serving_terminal_mode", "decode_forward")
            == "direct_lm_head"
        )
        self._dflow_contextual_jit_nfe1_truncation = False
        self._dflow_contextual_jit_nfe32_euler_resolution_ablation = False
        self._dflow_contextual_jit_same_state_sc_dual_proposal = False
        self._dflow_contextual_jit_nfe_endpoint_candidate_union = False
        self._dflow_clean_latent_source = str(
            getattr(self.draft_model, "clean_latent_source", "token_embedding")
        )
        self._requires_router_target_hidden = (
            _draft_model_requires_router_target_hidden(
                draft_model=self.draft_model,
                is_dflow_model=self._is_dflow_model,
                clean_latent_source=self._dflow_clean_latent_source,
            )
        )
        self._dflow_fixed_conditioning_folded = bool(
            getattr(self.draft_model, "fixed_conditioning_folded", False)
        )
        self._dflow_verified_root_conditioning = bool(
            getattr(self.draft_model, "verified_root_conditioning", False)
        )
        _validate_dflow_verified_root_algorithm(
            enabled=self._dflow_verified_root_conditioning,
            speculative_algorithm=server_args.speculative_algorithm,
        )
        draft_hf_config = self.draft_model_runner.model_config.hf_config
        self._dflow_checkpoint_noise_scale = None
        if self._is_dflow_model and self._dflow_one_step_direct:
            self._dflow_checkpoint_noise_scale = (
                _validate_dflow_one_step_noise_contract(
                    draft_hf_config,
                    self.flow_noise_scale,
                )
            )
        expected_dflow_mode = (
            "token_embedding_elf"
            if self._is_dflow_token_embedding_model
            else ("one_step" if self._dflow_one_step_direct else "denoise_decode")
        )
        if self._is_dflow_model and self.flow_mode != expected_dflow_mode:
            raise ValueError(
                "This DFlowDraftModel checkpoint requires "
                f"--speculative-dflash-flow-mode={expected_dflow_mode}. "
                "Using another path would violate its trained Gaussian-latent, "
                "conditioning, and forward-count contract."
            )
        if self._dflow_direct_lm_head_terminal and (
            not self._is_dflow_model
            or self._is_dflow_token_embedding_model
            or self._dflow_one_step_direct
            or self.flow_mode != "denoise_decode"
        ):
            raise ValueError(
                "direct_lm_head is only valid for the iterative contextual "
                "DFlowDraftModel denoise_decode serving path."
            )
        if self._dflow_direct_lm_head_terminal:
            expected_steps = int(
                getattr(self.draft_model, "serving_num_flow_steps", -1)
            )
            requested_nfe1_truncation = _contextual_jit_nfe1_truncation_enabled()
            requested_nfe32_euler_resolution_ablation = (
                _contextual_jit_nfe32_euler_resolution_ablation_enabled()
            )
            requested_same_state_sc_dual_proposal = (
                _contextual_jit_same_state_sc_dual_proposal_enabled()
            )
            requested_nfe_endpoint_candidate_union = (
                _contextual_jit_nfe_endpoint_candidate_union_enabled()
            )
            self._dflow_contextual_jit_nfe1_truncation = bool(
                requested_nfe1_truncation
                and expected_steps == 8
                and self.flow_steps == 1
            )
            self._dflow_contextual_jit_nfe32_euler_resolution_ablation = bool(
                requested_nfe32_euler_resolution_ablation
                and not requested_nfe1_truncation
                and expected_steps == 8
                and self.flow_steps == 32
                and self.flow_solver == "euler"
                and str(server_args.speculative_algorithm).upper() == "DFLASH"
            )
            self._dflow_contextual_jit_same_state_sc_dual_proposal = bool(
                requested_same_state_sc_dual_proposal
                and self._dflow_contextual_jit_nfe1_truncation
                and str(server_args.speculative_algorithm).upper() == "DDTREE"
            )
            self._dflow_contextual_jit_nfe_endpoint_candidate_union = bool(
                requested_nfe_endpoint_candidate_union
                and self.flow_steps == expected_steps
                and str(server_args.speculative_algorithm).upper() == "DDTREE"
            )
            expected_noise_scale = float(
                getattr(self.draft_model, "serving_denoiser_noise_scale", float("nan"))
            )
            expected_t_eps = float(
                getattr(self.draft_model, "serving_t_eps", float("nan"))
            )
            expected_self_conditioning = bool(
                getattr(self.draft_model, "serving_self_conditioning", False)
            )
            violations = []
            if self.flow_solver != "euler":
                violations.append(f"solver=euler (got {self.flow_solver!r})")
            if (
                self.flow_steps != expected_steps
                and not self._dflow_contextual_jit_nfe1_truncation
                and not self._dflow_contextual_jit_nfe32_euler_resolution_ablation
            ):
                violations.append(
                    "flow_steps="
                    f"{expected_steps} (got {self.flow_steps})"
                )
            if requested_nfe1_truncation and not self._dflow_contextual_jit_nfe1_truncation:
                violations.append(
                    f"{_CONTEXTUAL_JIT_NFE1_TRUNCATION_ENV}=1 only permits "
                    "the explicit checkpoint NFE=8 -> runtime NFE=1 Euler "
                    f"ablation (got checkpoint NFE={expected_steps}, "
                    f"runtime NFE={self.flow_steps})"
                )
            if (
                requested_nfe32_euler_resolution_ablation
                and not self._dflow_contextual_jit_nfe32_euler_resolution_ablation
            ):
                violations.append(
                    f"{_CONTEXTUAL_JIT_NFE32_EULER_RESOLUTION_ABLATION_ENV}=1 "
                    "only permits the explicit checkpoint NFE=8 -> runtime "
                    "NFE=32 Euler-resolution ablation with "
                    "speculative_algorithm=DFLASH "
                    f"(got checkpoint NFE={expected_steps}, "
                    f"runtime NFE={self.flow_steps}, solver={self.flow_solver!r}, "
                    f"algorithm={server_args.speculative_algorithm!r})"
                )
            if requested_nfe1_truncation and requested_nfe32_euler_resolution_ablation:
                violations.append(
                    "NFE=1 truncation and NFE=32 Euler-resolution ablation "
                    "switches are mutually exclusive"
                )
            if (
                requested_same_state_sc_dual_proposal
                and not self._dflow_contextual_jit_same_state_sc_dual_proposal
            ):
                violations.append(
                    f"{_CONTEXTUAL_JIT_SAME_STATE_SC_DUAL_PROPOSAL_ENV}=1 "
                    "requires the explicit contextual-JiT checkpoint NFE=8 "
                    "-> runtime NFE=1 switch and speculative_algorithm=DDTREE"
                )
            if (
                requested_nfe_endpoint_candidate_union
                and not self._dflow_contextual_jit_nfe_endpoint_candidate_union
            ):
                violations.append(
                    f"{_CONTEXTUAL_JIT_NFE_ENDPOINT_CANDIDATE_UNION_ENV}=1 "
                    "requires the checkpoint's exact NFE and "
                    "speculative_algorithm=DDTREE"
                )
            if (
                self._dflow_contextual_jit_same_state_sc_dual_proposal
                and self._dflow_contextual_jit_nfe_endpoint_candidate_union
            ):
                violations.append(
                    "same-state SC dual proposal and NFE endpoint candidate "
                    "union are mutually exclusive"
                )
            if not math.isclose(
                self.flow_t_min, 0.0, rel_tol=0.0, abs_tol=0.0
            ) or not math.isclose(self.flow_t_max, 1.0, rel_tol=0.0, abs_tol=0.0):
                violations.append(
                    f"linear time grid [0.0, 1.0] (got [{self.flow_t_min}, {self.flow_t_max}])"
                )
            if not math.isclose(
                self.flow_noise_scale,
                expected_noise_scale,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                violations.append(
                    "noise_scale="
                    f"{expected_noise_scale} (got {self.flow_noise_scale})"
                )
            if not math.isclose(
                self.flow_t_eps,
                expected_t_eps,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                violations.append(
                    f"t_eps={expected_t_eps} (got {self.flow_t_eps})"
                )
            if self.flow_noise_samples != 1:
                violations.append(
                    f"flow_noise_samples=1 (got {self.flow_noise_samples})"
                )
            if self.flow_noise_aggregation != "latent_mean":
                violations.append(
                    "flow_noise_aggregation=latent_mean "
                    f"(got {self.flow_noise_aggregation!r})"
                )
            if violations:
                raise ValueError(
                    "Contextual hidden continuous-JiT serving contract mismatch: "
                    + "; ".join(violations)
                    + "."
                )
        if self._use_flow_ensemble_req_pool and (
            not self._is_dflow_model or not self._dflow_one_step_direct
        ):
            raise ValueError(
                "Multi-noise folded inference requires a one_step_direct "
                "DFlowDraftModel checkpoint."
            )
        if self._use_flow_ensemble_req_pool:
            required_req_rows = (
                self._flow_ensemble_base_pool_size * self.flow_noise_samples
            )
            actual_req_rows = int(
                self.draft_model_runner.req_to_token_pool.req_to_token.shape[0]
            )
            if actual_req_rows < required_req_rows:
                raise RuntimeError(
                    "DFlow folded request table is smaller than required: "
                    f"required={required_req_rows}, actual={actual_req_rows}."
                )
        draft_config = parse_dflash_draft_config(
            draft_hf_config=draft_hf_config
        )
        if (
            self._is_dflow_model
            and self._dflow_clean_latent_source == "target_last_hidden"
        ):
            target_depth = int(draft_config.num_target_layers or -1)
            target_layer_ids = list(draft_config.target_layer_ids or [])
            if target_depth <= 0 or not target_layer_ids or target_layer_ids[-1] != target_depth:
                raise ValueError(
                    "Contextual-x0 DFlow serving requires its last target feature "
                    "to be the post-final-norm sentinel. Expected final "
                    f"target_layer_ids entry {target_depth}, got {target_layer_ids}."
                )
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(draft_config.resolve_block_size(default=16))
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = draft_config.block_size
            if model_block_size is None:
                model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                if self._strict_one_forward_enabled or self._is_dflow_model:
                    raise ValueError(
                        "DFLASH checkpoint/runtime block size mismatch: "
                        "speculative_num_draft_tokens="
                        f"{self.block_size}, draft config block_size={model_block_size}."
                    )
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )

        if self._collaborative_gate_mode is not None:
            max_draft_tokens = self.block_size - 1
            violations = []
            if self._training_proposal_topk is None:
                violations.append(
                    f"{_TRAINING_PROPOSAL_TOPK_ENV} must enable exact q sampling"
                )
            if (
                self._collaborative_gate_mode == "fixed"
                and self._collaborative_fixed_draft_tokens > max_draft_tokens
            ):
                violations.append(
                    "fixed draft tokens must fit the proposal block: "
                    f"fixed={self._collaborative_fixed_draft_tokens}, "
                    f"K={max_draft_tokens}"
                )
            if violations:
                raise ValueError(
                    "Collaborative source DFLASH configuration is invalid: "
                    + "; ".join(violations)
                    + "."
                )

        self.active_judge_enabled = bool(
            getattr(server_args, "speculative_dflash_active_judge", False)
        )
        self.active_judge_max_blocks = int(
            getattr(server_args, "speculative_dflash_active_max_blocks", 4)
        )
        self.active_judge_threshold = float(
            getattr(server_args, "speculative_dflash_judge_threshold", 0.5)
        )
        self.active_judge_decoder_mode = normalize_joint_judge_decoder_mode(
            getattr(
                server_args,
                "speculative_dflash_judge_decoder_mode",
                "threshold",
            )
        )
        self.active_judge_boundary_early_cost = float(
            getattr(
                server_args,
                "speculative_dflash_judge_boundary_early_cost",
                1.0,
            )
        )
        self.active_judge_boundary_late_cost = float(
            getattr(
                server_args,
                "speculative_dflash_judge_boundary_late_cost",
                1.0,
            )
        )
        self.active_judge_boundary_quantile = float(
            getattr(
                server_args,
                "speculative_dflash_judge_boundary_quantile",
                0.5,
            )
        )
        self.active_verify_bucketing_enabled = bool(
            getattr(
                server_args,
                "speculative_dflash_active_verify_bucketing",
                False,
            )
        )
        if self.active_verify_bucketing_enabled and not self.active_judge_enabled:
            raise ValueError(
                "Active DFLASH target verify bucketing requires active judge."
            )
        self._judge_graph_validation_steps = _judge_graph_validation_steps()
        self._judge_graph_validation_count = 0
        self.active_verify_width = 1 + self.active_judge_max_blocks * (
            self.block_size - 1
        )
        if self.active_judge_enabled:
            violations = []
            resolved_verify_width = int(
                server_args.resolve_speculative_target_verify_width()
            )
            if resolved_verify_width != self.active_verify_width:
                violations.append(
                    "target verify width mismatch: worker="
                    f"{self.active_verify_width}, server={resolved_verify_width}"
                )
            if not bool(draft_config.joint_judge_enabled):
                violations.append(
                    "checkpoint dflash_config.joint_judge_enabled must be true"
                )
            if not bool(getattr(self.draft_model, "joint_judge_enabled", False)):
                violations.append("runtime model did not instantiate the joint judge")
            if self.block_size < 2:
                violations.append(f"block_size must be >= 2, got {self.block_size}")
            if self.active_judge_max_blocks < 1:
                violations.append(
                    "speculative_dflash_active_max_blocks must be >= 1, got "
                    f"{self.active_judge_max_blocks}"
                )
            if not (
                math.isfinite(self.active_judge_threshold)
                and 0.0 <= self.active_judge_threshold <= 1.0
            ):
                violations.append(
                    "speculative_dflash_judge_threshold must be finite and in "
                    f"[0, 1], got {self.active_judge_threshold}"
                )
            if (
                not math.isfinite(self.active_judge_boundary_early_cost)
                or not math.isfinite(self.active_judge_boundary_late_cost)
                or self.active_judge_boundary_early_cost < 0.0
                or self.active_judge_boundary_late_cost < 0.0
                or (
                    self.active_judge_boundary_early_cost
                    + self.active_judge_boundary_late_cost
                    <= 0.0
                )
            ):
                violations.append(
                    "judge boundary early/late costs must be finite, "
                    "non-negative, and not both zero"
                )
            if not (
                math.isfinite(self.active_judge_boundary_quantile)
                and 0.0 <= self.active_judge_boundary_quantile <= 1.0
            ):
                violations.append(
                    "judge boundary quantile must be finite and in [0, 1], got "
                    f"{self.active_judge_boundary_quantile}"
                )
            if self.page_size != 1:
                violations.append(
                    f"page_size must be 1 for transactional active chaining, got {self.page_size}"
                )
            if self.use_compact_draft_cache:
                violations.append("draft_window_size must be disabled")
            if self.flow_mode != "off":
                violations.append(f"flow_mode must be 'off', got {self.flow_mode!r}")
            if self._is_dflow_model:
                violations.append("DFlow checkpoints are not joint-judge checkpoints")
            if self._strict_one_forward_enabled:
                violations.append(
                    f"{_STRICT_ONE_FORWARD_ENV}=1 is incompatible with a multi-block chain"
                )
            if self._training_proposal_topk is not None:
                violations.append(
                    f"{_TRAINING_PROPOSAL_TOPK_ENV} is not supported by active judge"
                )
            if self._collaborative_gate_mode is not None:
                violations.append(
                    f"{_COLLABORATIVE_GATE_MODE_ENV} is not supported by active judge"
                )
            if int(draft_config.sparse_num_experts) > 1:
                violations.append("sparse drafter experts are not supported")
            if self._draft_attention_backend != "flashinfer":
                violations.append(
                    "the first active-chain runtime supports only the FlashInfer "
                    f"draft attention backend, got {self._draft_attention_backend!r}"
                )
            target_backend_name, _ = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )
            if target_backend_name != "FlashInferAttnBackend":
                violations.append(
                    "the first active-chain runtime supports only a FlashInfer "
                    "target full-attention backend, got "
                    f"{target_backend_name!r}"
                )
            if violations:
                raise ValueError(
                    "Active DFLASH drafter/judge runtime contract failed: "
                    + "; ".join(violations)
                )

        self._timing = DFlashCudaTimingProfiler(
            device=self.device,
            tp_rank=self.tp_rank,
            block_size=self.block_size,
        )
        self._last_active_judge_block_accept_lens: list[list[int]] = []
        self._last_active_judge_proposal_lens: list[int] = []
        self._last_active_judge_blocks_run: list[int] = []
        self._last_active_judge_first_fail_blocks: list[int] = []

        if self._is_dflow_model:
            # DFlow starts non-anchor positions from Gaussian latents and does
            # not define a tokenizer MASK token.  Keep a valid placeholder ID
            # only for the reusable verification buffer; proposal positions are
            # overwritten by sampled draft IDs before target verification.
            self._mask_token = "<DFlow Gaussian latent>"
            self._mask_token_id_override = None
            self._mask_token_id = 0
        else:
            self._mask_token = draft_config.mask_token
            self._mask_token_id_override = draft_config.mask_token_id
            self._mask_token_id = self._resolve_mask_token_id(
                mask_token=self._mask_token,
                mask_token_id=self._mask_token_id_override,
            )
        if (
            self._is_dflow_model
            and str(os.environ.get("SGLANG_DFLASH_POS_MASK_EMB_PATH", "")).strip()
        ):
            raise ValueError(
                "DFlowDraftModel does not use position-specific MASK embeddings; "
                "unset SGLANG_DFLASH_POS_MASK_EMB_PATH to preserve training/serving "
                "consistency."
            )
        self._pos_mask_embeddings = self._load_pos_mask_embeddings()
        configured_target_layer_ids = (
            None
            if draft_config.target_layer_ids is None
            else [int(x) for x in draft_config.target_layer_ids]
        )
        runtime_target_layer_ids = getattr(
            self.model_runner, "dflash_target_layer_ids", None
        )
        if runtime_target_layer_ids is not None:
            runtime_target_layer_ids = [int(x) for x in runtime_target_layer_ids]
        if self._is_dflow_model and (
            configured_target_layer_ids is None
            or runtime_target_layer_ids != configured_target_layer_ids
        ):
            raise ValueError(
                "DFlow target feature capture does not match the checkpoint: "
                f"checkpoint={configured_target_layer_ids}, "
                f"runtime={runtime_target_layer_ids}."
            )
        _validate_strict_one_forward_checkpoint_contract(
            enabled=self._strict_one_forward_enabled,
            runtime_block_size=self.block_size,
            config_block_size=draft_config.block_size,
            config_mask_token_id=draft_config.mask_token_id,
            resolved_mask_token_id=self._mask_token_id,
            pos_mask_embeddings_present=self._pos_mask_embeddings is not None,
            configured_target_layer_ids=configured_target_layer_ids,
            runtime_target_layer_ids=runtime_target_layer_ids,
        )
        _validate_dflow_one_backbone_checkpoint_contract(
            enabled=self._dflow_strict_one_backbone_enabled,
            is_dflow_model=self._is_dflow_model,
            runtime_one_step_direct=self._dflow_one_step_direct,
            dflow_config=getattr(draft_hf_config, "dflow_config", None),
            flow_steps=self.flow_steps,
            active_judge_enabled=self.active_judge_enabled,
        )
        target_input_embeddings = (
            self.target_worker.model_runner.model.get_input_embeddings()
        )
        _validate_token_embedding_elf_runtime_contract(
            enabled=self._is_dflow_token_embedding_model,
            speculative_algorithm=server_args.speculative_algorithm,
            flow_mode=self.flow_mode,
            flow_steps=self.flow_steps,
            flow_t_min=self.flow_t_min,
            flow_t_max=self.flow_t_max,
            flow_t_eps=self.flow_t_eps,
            flow_noise_scale=self.flow_noise_scale,
            flow_noise_samples=self.flow_noise_samples,
            flow_noise_aggregation=self.flow_noise_aggregation,
            flow_schedule=self.flow_schedule,
            block_size=self.block_size,
            draft_window_size=self.draft_window_size,
            active_judge_enabled=self.active_judge_enabled,
            strict_native_enabled=self._strict_one_forward_enabled,
            strict_one_backbone_enabled=self._dflow_strict_one_backbone_enabled,
            draft_hf_config=draft_hf_config,
            target_hf_config=self.model_runner.model_config.hf_config,
            target_input_embeddings=target_input_embeddings,
        )
        if (
            self._is_dflow_token_embedding_model
            and self._same_time_self_cond_correction_enabled
            and not self._token_embedding_serving_self_conditioning
        ):
            raise ValueError(
                "Same-time self-conditioning correction is incompatible with "
                "a token-embedding ELF checkpoint whose "
                "serving_self_conditioning=false."
            )
        self._token_embedding_l2_geometry = None
        self._initialize_draft_forward_counters()
        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )
            if self.active_judge_enabled:
                logger.info(
                    "Active DFLASH drafter/judge chain enabled. block_size=%s, "
                    "max_blocks=%s, target_verify_width=%s, judge_threshold=%s, "
                    "judge_decoder=%s, boundary_costs[early=%s late=%s], "
                    "boundary_quantile=%s, "
                    "runtime_verify_bucketing=%s, verify_width_buckets=%s",
                    self.block_size,
                    self.active_judge_max_blocks,
                    self.active_verify_width,
                    self.active_judge_threshold,
                    self.active_judge_decoder_mode,
                    self.active_judge_boundary_early_cost,
                    self.active_judge_boundary_late_cost,
                    self.active_judge_boundary_quantile,
                    self.active_verify_bucketing_enabled,
                    get_active_dflash_verify_width_buckets(
                        block_size=self.block_size,
                        max_verify_width=self.active_verify_width,
                    ),
                )
            if self._strict_one_forward_enabled:
                logger.info(
                    "Strict one-forward DFLASH serving contract enabled. "
                    "block_size=%s, mask_token_id=%s, target_layer_ids=%s",
                    self.block_size,
                    self._mask_token_id,
                    runtime_target_layer_ids,
                )
            if self._dflow_strict_one_backbone_enabled:
                logger.info(
                    "Strict one-backbone DFlow serving contract enabled. "
                    "checkpoint_one_step_direct=%s, flow_mode=%s, flow_steps=%s",
                    self._dflow_one_step_direct,
                    self.flow_mode,
                    self.flow_steps,
                )
            if self.flow_mode != "off":
                logger.info(
                    "Flow-DFLASH draft proposal enabled. mode=%s, steps=%s, "
                    "t_min=%s, t_max=%s, t_eps=%s, noise_scale=%s, "
                    "noise_samples=%s, noise_aggregation=%s, schedule=%s, "
                    "solver=%s, native_dflow=%s. "
                    "Verifier remains unchanged.",
                    self.flow_mode,
                    self.flow_steps,
                    self.flow_t_min,
                    self.flow_t_max,
                    self.flow_t_eps,
                    self.flow_noise_scale,
                    self.flow_noise_samples,
                    self.flow_noise_aggregation,
                    self.flow_schedule,
                    self.flow_solver,
                    self._is_dflow_model,
                )
            if self._dflow_direct_lm_head_terminal:
                logger.info(
                    "Contextual hidden continuous-JiT serving enabled: "
                    "solver=euler NFE=%s checkpoint_default_nfe=%s "
                    "runtime_nfe_override=%s "
                    "same_state_sc_dual_proposal=%s "
                    "endpoint=target_post_final_norm_hidden "
                    "serving_self_conditioning=%s "
                    "physical_draft_forwards=%s "
                    "no_extra_decode_forward=true.",
                    self.flow_steps,
                    int(getattr(self.draft_model, "serving_num_flow_steps", -1)),
                    (
                        "nfe1_truncation"
                        if bool(
                            getattr(
                                self,
                                "_dflow_contextual_jit_nfe1_truncation",
                                False,
                            )
                        )
                        else "nfe32_euler_resolution_ablation"
                        if bool(
                            getattr(
                                self,
                                "_dflow_contextual_jit_nfe32_euler_resolution_ablation",
                                False,
                            )
                        )
                        else "none"
                    ),
                    bool(
                        self.__dict__.get(
                            "_dflow_contextual_jit_same_state_sc_dual_proposal",
                            False,
                        )
                    ),
                    bool(
                        getattr(self.draft_model, "serving_self_conditioning", False)
                    ),
                    int(self.flow_steps)
                    + int(
                        bool(
                            self.__dict__.get(
                                "_dflow_contextual_jit_same_state_sc_dual_proposal",
                                False,
                            )
                        )
                    ),
                )
            if self._is_dflow_token_embedding_model:
                checkpoint_default_nfe = int(
                    getattr(self.draft_model, "serving_num_flow_steps", 8)
                )
                logger.info(
                    "Token-embedding ELF serving enabled: solver=%s NFE=%s "
                    "integration_intervals=%s "
                    "checkpoint_default_nfe=%s solver_resolution_ablation=%s "
                    "endpoint=%s(input_embedding) valid_vocab=%s "
                    "serving_self_conditioning=%s "
                    "same_time_self_cond_correction=%s "
                    "physical_draft_forwards=%s no_extra_decode_forward=true "
                    "latest_endpoint_bridge_mix=%s elf_euler_velocity_scale=%s.",
                    self.flow_solver,
                    self.flow_steps,
                    (
                        self.flow_steps // 2
                        if self.flow_solver == "heun"
                        else self.flow_steps
                    ),
                    checkpoint_default_nfe,
                    self.flow_steps != checkpoint_default_nfe,
                    getattr(self.draft_model, "retrieval_metric", None),
                    getattr(self.draft_model, "valid_vocab_size", None),
                    self._token_embedding_serving_self_conditioning,
                    self._same_time_self_cond_correction_enabled,
                    int(self.flow_steps)
                    + int(self._same_time_self_cond_correction_enabled),
                    self.elf_latest_endpoint_bridge_mix,
                    self.elf_euler_velocity_scale,
                )
                if self._same_time_self_cond_correction_enabled:
                    logger.info(
                        "Same-time self-conditioning correction enabled: "
                        "solver_NFE=1 physical_draft_forwards=2"
                    )
            if self._training_proposal_topk is not None:
                logger.warning(
                    "Training-only stochastic DFLASH proposals enabled. "
                    "topk=%s, temperature=%s. Proposal q_old is recorded in "
                    "joint_cycle_trace.",
                    self._training_proposal_topk,
                    self._training_proposal_temperature,
                )
            if self._collaborative_gate_mode is not None:
                logger.warning(
                    "Collaborative source DFLASH enabled. gate_mode=%s, "
                    "fixed_draft_tokens=%s. This is a composite q_D/p_V policy "
                    "and does not preserve strict speculative output.",
                    self._collaborative_gate_mode,
                    self._collaborative_fixed_draft_tokens,
                )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_model_input_ids_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size], may carry sparse expert IDs
        )
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._active_verify_tokens_buf: Optional[torch.Tensor] = None
        self._active_verify_positions_buf: Optional[torch.Tensor] = None
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0
        self._head_lora_vocab_shard_validated = False
        self._logged_first_sparse_route = False

        self._use_fused_kv_materialize = (
            is_cuda() and not _disable_fused_kv_materialization()
        )
        if not self._use_fused_kv_materialize and self.tp_rank == 0 and is_cuda():
            logger.info(
                "DFLASH fused KV materialization disabled by %s.",
                _DISABLE_FUSED_KV_ENV,
            )
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        try:
            if not bool(
                getattr(
                    self.draft_model,
                    "supports_fused_kv_materialization",
                    True,
                )
            ):
                if self.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled by model %s.",
                        self.draft_model.__class__.__name__,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return
            layers = self.draft_model.layers
            fused_disable_reason: Optional[str] = None

            if len(layers) == 0:
                fused_disable_reason = "no layers found"

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    fused_disable_reason = f"{reason}: layer={layer_idx}"
                    break

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break

            if fused_disable_reason is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            rotary_emb = first_attn.rotary_emb

            self._fused_kv_helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
            )
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False
            self._fused_kv_helper = None

    def _load_pos_mask_embeddings(self):
        """Load learned position-specific DFlash MASK embeddings from env.

        Env:
            SGLANG_DFLASH_POS_MASK_EMB_PATH=/path/to/pos_mask_embeddings.pt

        Supported payloads:
            Tensor[block_size - 1, hidden_size]
            Tensor[block_size, hidden_size]  # anchor row is ignored
            {"pos_mask_embeddings": tensor, ...}
        """
        path = os.environ.get("SGLANG_DFLASH_POS_MASK_EMB_PATH", None)
        if path is None or str(path).strip() == "":
            return None

        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SGLANG_DFLASH_POS_MASK_EMB_PATH points to a missing file: {path}"
            )

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            if "pos_mask_embeddings" not in payload:
                raise KeyError(
                    f"{path} is a dict but does not contain 'pos_mask_embeddings'. "
                    f"Available keys: {sorted(payload.keys())}"
                )
            emb = payload["pos_mask_embeddings"]
            payload_block_size = payload.get("block_size", None)
            payload_mask_id = payload.get("mask_token_id", None)
            if payload_block_size is not None and int(payload_block_size) != int(
                self.block_size
            ):
                raise ValueError(
                    "Loaded pos-mask block_size mismatch: "
                    f"file={payload_block_size}, server={self.block_size}"
                )
            if payload_mask_id is not None and int(payload_mask_id) != int(
                self._mask_token_id
            ):
                logger.warning(
                    "Loaded pos-mask mask_token_id differs from server mask token: "
                    "file=%s, server=%s. This can be okay if you intentionally reuse "
                    "the learned vectors with a different reserved token.",
                    payload_mask_id,
                    self._mask_token_id,
                )
        else:
            emb = payload

        if not torch.is_tensor(emb):
            raise TypeError(f"pos_mask_embeddings must be a tensor, got {type(emb)}")

        emb = emb.detach().float().contiguous()
        if emb.ndim != 2:
            raise ValueError(
                f"pos_mask_embeddings must be rank-2, got shape={tuple(emb.shape)}"
            )

        if int(emb.shape[0]) == int(self.block_size):
            # Allow files that include an anchor row. In DFlash the anchor is real
            # token embedding and should never be overridden.
            emb = emb[1:].contiguous()

        if int(emb.shape[0]) != int(self.block_size) - 1:
            raise ValueError(
                "pos_mask_embeddings first dimension must be block_size - 1 "
                f"({int(self.block_size) - 1}) or block_size ({self.block_size}); "
                f"got {tuple(emb.shape)}"
            )

        emb = emb.to(device=self.device)
        if self.tp_rank == 0:
            logger.info(
                "Loaded DFLASH position-specific mask embeddings from %s: shape=%s",
                path,
                tuple(emb.shape),
            )
        return emb

    def _override_noise_embedding_with_pos_mask(
        self, noise_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Replace DFlash MASK slots with learned position-specific embeddings."""
        pos_emb = getattr(self, "_pos_mask_embeddings", None)
        if pos_emb is None:
            return noise_embedding

        if noise_embedding.ndim != 3:
            raise RuntimeError(
                "DFLASH pos-mask override expects noise_embedding shape "
                f"[batch, block_size, hidden], got {tuple(noise_embedding.shape)}"
            )

        if int(noise_embedding.shape[1]) != int(self.block_size):
            raise RuntimeError(
                "DFLASH pos-mask override block size mismatch: "
                f"noise_embedding={tuple(noise_embedding.shape)}, block_size={self.block_size}"
            )

        if int(pos_emb.shape[-1]) != int(noise_embedding.shape[-1]):
            raise RuntimeError(
                "DFLASH pos-mask hidden size mismatch: "
                f"pos_mask={tuple(pos_emb.shape)}, noise_embedding={tuple(noise_embedding.shape)}"
            )

        pos_emb = pos_emb.to(device=noise_embedding.device, dtype=noise_embedding.dtype)
        noise_embedding = noise_embedding.clone()
        noise_embedding[:, 1:, :].copy_(
            pos_emb.unsqueeze(0).expand(noise_embedding.shape[0], -1, -1)
        )
        return noise_embedding

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_model_input_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker.
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # The target worker owns the shared KV allocator/cache. For the compact
        # sliding-window path, the draft req->token view is rebuilt from committed
        # target state before each draft forward, so there is nothing persistent
        # to flush here.
        pass

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        pos2d: torch.Tensor,
        mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if pos2d.ndim != 2:
            raise RuntimeError(
                f"{context} expected 2D positions, got shape={tuple(pos2d.shape)}."
            )
        if mask.shape != pos2d.shape:
            raise RuntimeError(
                f"{context} mask/position shape mismatch: {tuple(mask.shape)} vs {tuple(pos2d.shape)}."
            )

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        table_width = int(req_to_token.shape[1])
        if table_width <= 0:
            if bool(mask.any().item()):
                raise RuntimeError(
                    f"{context} req_to_token table is empty but gather mask is non-empty."
                )
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        # Only the masked-off rectangular padding can be out of range in the normal
        # ragged-batch case. Replace those don't-care columns with a valid in-range
        # position before the gather so the kernel only sees real positions.
        safe_pos2d = pos2d.masked_fill(~mask, 0)
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)

    def _gather_req_to_token_segments(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        start: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        if lengths.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        offsets = torch.arange(
            max_len, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        if start is None:
            pos2d = offsets.expand(req_pool_indices.shape[0], -1)
        else:
            pos2d = start.to(torch.int64).unsqueeze(1) + offsets
        mask = offsets < lengths.unsqueeze(1)
        return self._gather_req_to_token_masked(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            pos2d=pos2d,
            mask=mask,
            context="DFLASH req_to_token segment gather",
        )

    def _sync_flow_ensemble_base_prefix(
        self,
        *,
        batch: ScheduleBatch,
        draft_seq_lens: torch.Tensor,
    ) -> None:
        """Mirror committed prefix indices into sample-0's private draft rows."""

        if not self._use_flow_ensemble_req_pool:
            return
        draft_seq_lens = draft_seq_lens.to(
            device=self.device, dtype=torch.int32
        )
        prefix_cache_loc = self._gather_req_to_token_segments(
            req_to_token=batch.req_to_token_pool.req_to_token,
            req_pool_indices=batch.req_pool_indices,
            start=None,
            lengths=draft_seq_lens,
        )
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            self.draft_model_runner.req_to_token_pool.req_to_token,
            torch.zeros_like(draft_seq_lens),
            draft_seq_lens,
            prefix_cache_loc,
            batch.batch_size(),
        )

    def _compute_compact_draft_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        assert self.draft_window_size is not None
        visible_lens = torch.clamp(
            seq_lens.to(dtype=torch.int32, device=self.device),
            max=int(self.draft_window_size),
        )
        if self.page_size <= 1:
            return visible_lens

        # Paged FA backends derive the page table from local token positions, so the
        # compact suffix must start on a page boundary. Keep up to page_size - 1 extra
        # tokens on the left to preserve valid local page structure.
        seq_lens_i64 = seq_lens.to(torch.int64)
        visible_lens_i64 = visible_lens.to(torch.int64)
        visible_start = seq_lens_i64 - visible_lens_i64
        aligned_start = visible_start - torch.remainder(visible_start, self.page_size)
        return (seq_lens_i64 - aligned_start).to(torch.int32)

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)

    def _run_draft_block_forward(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
        *,
        root_ids: Optional[torch.Tensor] = None,
        target_prefix_lens: Optional[torch.Tensor] = None,
        draft_prefix_lens: Optional[torch.Tensor] = None,
        draft_prefix_lens_cpu: Optional[torch.Tensor] = None,
        append_target_context: bool = True,
        restore_allocator: bool = True,
        return_forward_batch: bool = False,
        timing_block_idx: Optional[int] = None,
    ):
        """Run the shared DFlash block forward and return hidden states.

        DDTree reuses exactly this proposal path, including PosMask and the
        optional Flow-DFlash mode, but consumes top-k distributions instead of
        immediately taking a per-position argmax.
        """

        bs = batch.batch_size()
        # Candidate-union hiddens are transaction-local. DDTree must consume
        # them after this call via take_dflow_candidate_sample_hiddens().
        self._pending_dflow_candidate_sample_hiddens = None

        if append_target_context:
            # Append newly committed target features into every draft-side KV
            # tower before constructing the first block in a transaction.
            self._timing.mark("draft_kv_pre_start")
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            self._timing.mark("draft_kv_pre_end")

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if (
            lm_head is None
            or not hasattr(lm_head, "weight")
            or not hasattr(lm_head, "shard_indices")
        ):
            raise RuntimeError(
                "DFlash/DDTree requires the target model to expose a vocab-parallel "
                "`lm_head` with `weight` and `shard_indices` attributes."
            )

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        block_ids.fill_(int(self._mask_token_id))
        if (
            root_ids is not None
            and self.__dict__.get("_dflow_verified_root_conditioning", False)
        ):
            raise RuntimeError(
                "Verified-root DFlow does not support synthetic/overridden "
                "block roots; each root must be draft_input.verified_id from "
                "the completed verifier transaction."
            )
        if root_ids is None:
            root_ids = draft_input.verified_id
        if root_ids.ndim != 1 or int(root_ids.numel()) != int(bs):
            raise RuntimeError(
                "DFLASH block roots must have shape [batch], got "
                f"{tuple(root_ids.shape)} for batch={bs}."
            )
        block_ids[:, 0].copy_(root_ids.to(device=self.device, dtype=torch.long))

        verified_root_embedding = None
        if bool(self.__dict__.get("_dflow_verified_root_conditioning", False)):
            verified_root_embedding = _lookup_dflow_verified_root_embedding(
                embed_module,
                block_ids[:, 0],
                batch_size=bs,
                hidden_size=int(self.draft_model.config.hidden_size),
            )

        if self._is_dflow_model:
            # The reference DFlow sampler embeds only the verified anchor.  All
            # remaining positions are initialized from synchronized Gaussian
            # latent noise in _run_flow_dflash_draft_forward.
            if self._dflow_clean_latent_source == "target_last_hidden":
                router_target_hidden = draft_input.router_target_hidden
                if router_target_hidden is None:
                    raise RuntimeError(
                        "Contextual-x0 DFlow draft state is missing the target "
                        "hidden that predicts the verified anchor."
                    )
                hidden_size = int(self.draft_model.config.hidden_size)
                if int(router_target_hidden.shape[-1]) < hidden_size:
                    raise RuntimeError(
                        "Contextual-x0 target feature is narrower than one hidden "
                        f"state: shape={tuple(router_target_hidden.shape)}, "
                        f"hidden_size={hidden_size}."
                    )
                # The config/runtime ABI guarantees that the final H channels
                # are the post-final-norm h[p-1] whose LM-head logits produced
                # the current verified token y[p].
                anchor_embedding = router_target_hidden[:, -hidden_size:].unsqueeze(1)
            else:
                anchor_embedding = embed_module(block_ids[:, :1])
            noise_embedding = anchor_embedding.new_zeros(
                (bs, self.block_size, int(anchor_embedding.shape[-1]))
            )
            noise_embedding[:, :1, :].copy_(anchor_embedding)
        else:
            noise_embedding = embed_module(block_ids)
            noise_embedding = self._override_noise_embedding_with_pos_mask(
                noise_embedding
            )
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        model_input_ids = block_ids
        select_experts = getattr(self.draft_model, "select_experts", None)
        if callable(select_experts):
            router_target_hidden = draft_input.router_target_hidden
            if router_target_hidden is None:
                raise RuntimeError(
                    "Sparse DFlash draft state is missing router_target_hidden."
                )
            expert_ids = select_experts(
                anchor_embeddings=noise_embedding[:, 0, :],
                target_hidden=router_target_hidden,
            )
            assert self._draft_model_input_ids_buf is not None
            model_input_ids = self._draft_model_input_ids_buf[:bs]
            model_input_ids.copy_(block_ids)
            # The native SparseDFlashDraftModel has no embedding table and uses
            # this first slot solely as a CUDA-graph-friendly route carrier.
            model_input_ids[:, 0].copy_(expert_ids)
            if not self._logged_first_sparse_route and self.tp_rank == 0:
                route_counts = torch.bincount(
                    expert_ids, minlength=int(self.draft_model.num_experts)
                ).tolist()
                logger.info(
                    "Sparse DFLASH routed first block: expert_ids=%s counts=%s",
                    expert_ids.tolist(),
                    route_counts,
                )
                self._logged_first_sparse_route = True

        if target_prefix_lens is None:
            target_prefix_lens = batch.seq_lens
        if draft_prefix_lens is None:
            draft_prefix_lens = draft_input.draft_seq_lens
        if draft_prefix_lens.dtype != torch.int32:
            draft_prefix_lens = draft_prefix_lens.to(torch.int32)
        if draft_prefix_lens.device != self.device:
            draft_prefix_lens = draft_prefix_lens.to(self.device, non_blocking=True)

        positions_2d = self._draft_block_positions_buf[:bs]
        torch.add(
            target_prefix_lens.unsqueeze(1), self._block_pos_offsets, out=positions_2d
        )

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if draft_prefix_lens_cpu is None:
            seq_lens_cpu.copy_(
                draft_prefix_lens.to(device="cpu", dtype=torch.int32)
            )
        else:
            if (
                draft_prefix_lens_cpu.device.type != "cpu"
                or draft_prefix_lens_cpu.ndim != 1
                or int(draft_prefix_lens_cpu.numel()) != int(bs)
            ):
                raise RuntimeError(
                    "DFLASH draft_prefix_lens_cpu must be a CPU tensor with "
                    f"shape [batch], got {tuple(draft_prefix_lens_cpu.shape)} "
                    f"on {draft_prefix_lens_cpu.device} for batch={bs}."
                )
            seq_lens_cpu.copy_(draft_prefix_lens_cpu.to(dtype=torch.int32))

        noise_samples = (
            int(self.flow_noise_samples)
            if self._use_flow_ensemble_req_pool
            else 1
        )
        forward_bs = int(bs) * noise_samples
        if self._use_flow_ensemble_req_pool:
            # Sample-major layout:
            #   [sample0:req0..reqN, sample1:req0..reqN, ...]
            # Each virtual row shares the committed prefix KV indices but gets
            # disjoint transient slots for this proposal block.
            forward_req_pool_indices = _expand_dflow_ensemble_req_pool_indices(
                batch.req_pool_indices,
                base_pool_size=self._flow_ensemble_base_pool_size,
                noise_samples=noise_samples,
            )
            forward_draft_prefix_lens = (
                draft_prefix_lens.unsqueeze(0)
                .expand(noise_samples, -1)
                .reshape(-1)
                .contiguous()
            )
            forward_seq_lens_cpu = (
                seq_lens_cpu.unsqueeze(0)
                .expand(noise_samples, -1)
                .reshape(-1)
                .contiguous()
            )
            forward_block_ids = (
                model_input_ids.unsqueeze(0)
                .expand(noise_samples, -1, -1)
                .reshape(forward_bs, self.block_size)
                .contiguous()
            )
            forward_positions_2d = (
                positions_2d.unsqueeze(0)
                .expand(noise_samples, -1, -1)
                .reshape(forward_bs, self.block_size)
                .contiguous()
            )

            base_prefix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.draft_model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=batch.req_pool_indices,
                start=None,
                lengths=draft_prefix_lens,
            )
            ensemble_prefix_cache_loc = base_prefix_cache_loc.repeat(noise_samples)
            assign_req_to_token_pool_func(
                forward_req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(forward_draft_prefix_lens),
                forward_draft_prefix_lens,
                ensemble_prefix_cache_loc,
                forward_bs,
            )
        else:
            forward_req_pool_indices = batch.req_pool_indices
            forward_draft_prefix_lens = draft_prefix_lens
            forward_seq_lens_cpu = seq_lens_cpu
            forward_block_ids = model_input_ids
            forward_positions_2d = positions_2d

        forward_positions = forward_positions_2d.reshape(-1)
        forward_block_start = forward_draft_prefix_lens
        if self._use_flow_ensemble_req_pool:
            forward_block_end = forward_block_start + int(self.block_size)
        else:
            # Preserve the original M=1 buffer and in-place add path.
            forward_block_end = self._draft_block_end_buf[:bs]
            torch.add(
                forward_block_start,
                int(self.block_size),
                out=forward_block_end,
            )
        forward_block_end_cpu = forward_seq_lens_cpu + int(self.block_size)
        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        transient_tokens = forward_bs * self.block_size
        # The folded M-noise proposal is one physical forward, but it needs M
        # disjoint transient KV blocks at the same instant. The target
        # scheduler reserves this peak in new_tokens_required_next_decode().
        # Evict again at the actual allocation boundary because a prefill can
        # join the running batch after the preceding decode-memory check.
        #
        # Eviction must happen before backup_state(): restoring a pre-eviction
        # allocator snapshot after mutating the radix tree would resurrect
        # slots that the tree has already released.
        if restore_allocator:
            transient_reserve = transient_tokens
            if self.page_size > 1:
                transient_reserve += forward_bs * self.page_size
            evict_from_tree_cache(batch.tree_cache, transient_reserve)
        token_to_kv_pool_state_backup = (
            allocator.backup_state() if restore_allocator else None
        )
        try:
            if self.page_size == 1:
                block_cache_loc = allocator.alloc(transient_tokens)
            else:
                last_loc = get_last_loc(
                    self.draft_model_runner.req_to_token_pool.req_to_token,
                    forward_req_pool_indices,
                    forward_block_start,
                )
                block_cache_loc = allocator.alloc_extend(
                    forward_block_start,
                    forward_seq_lens_cpu,
                    forward_block_end,
                    forward_block_end_cpu,
                    last_loc,
                    transient_tokens,
                )
            if block_cache_loc is None:
                raise RuntimeError(
                    "DFlash/DDTree draft OOM when allocating "
                    f"{transient_tokens} block tokens "
                    f"(batch={bs}, noise_samples={noise_samples})."
                )

            assign_req_to_token_pool_func(
                forward_req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                forward_block_start,
                forward_block_end,
                block_cache_loc,
                forward_bs,
            )

            draft_spec_info = self._draft_block_spec_info
            if self._draft_attention_backend == "torch_native":
                # Unlike optimized speculative backends, torch_native does not
                # consume DFlashVerifyInput to extend prefix lengths by the fixed
                # block.  Spell out ordinary extend metadata so it can serve as
                # an independent non-causal SDPA correctness oracle.
                seq_lens = forward_block_end
                model_seq_lens_cpu = forward_block_end_cpu
                seq_lens_sum = int(forward_block_end_cpu.sum().item())
                extend_prefix_lens = forward_draft_prefix_lens
                extend_seq_lens = torch.full_like(
                    forward_draft_prefix_lens, int(self.block_size)
                )
            else:
                seq_lens = forward_draft_prefix_lens
                model_seq_lens_cpu = forward_seq_lens_cpu
                seq_lens_sum = int(forward_seq_lens_cpu.sum().item())
                extend_prefix_lens = None
                extend_seq_lens = None
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=forward_bs,
                input_ids=forward_block_ids.flatten(),
                req_pool_indices=forward_req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=block_cache_loc,
                seq_lens_sum=seq_lens_sum,
                seq_lens_cpu=model_seq_lens_cpu,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                positions=forward_positions,
                req_to_token_pool=self.draft_model_runner.req_to_token_pool,
                token_to_kv_pool=self.draft_model_runner.token_to_kv_pool,
                attn_backend=self.draft_model_runner.attn_backend,
                input_embeds=input_embeds,
                # The draft block itself still uses DFlash's non-causal fixed-block
                # semantics. DDTree only changes proposal selection and target verify.
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=draft_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )

            self._timing.mark("draft_backbone_start")
            if timing_block_idx is not None:
                self._timing.mark(
                    f"active_draft_backbone_start_{timing_block_idx}"
                )
            self._begin_draft_forward_counters()
            try:
                with torch.inference_mode():
                    draft_logits_output = self._run_flow_dflash_draft_forward(
                        forward_batch=forward_batch,
                        bs=bs,
                        input_embeds=input_embeds,
                        verified_root_embedding=verified_root_embedding,
                    )
                self._finish_draft_forward_counters()
            finally:
                # Preserve the original forward exception while making the next
                # scheduler transaction start from a clean strict-counter state.
                self._abort_draft_forward_counters()
            self._timing.mark("draft_backbone_end")
            if timing_block_idx is not None:
                self._timing.mark(
                    f"active_draft_backbone_end_{timing_block_idx}"
                )
        finally:
            if token_to_kv_pool_state_backup is not None:
                allocator.restore_state(token_to_kv_pool_state_backup)

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFlash/DDTree draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, self.block_size, -1)
        if return_forward_batch:
            return draft_hidden, positions_2d, block_ids, lm_head, forward_batch
        return draft_hidden, positions_2d, block_ids, lm_head

    def _ensure_active_verify_buffers(self, bs: int) -> None:
        width = int(self.active_verify_width)
        current_cap = (
            0
            if self._active_verify_tokens_buf is None
            else int(self._active_verify_tokens_buf.shape[0])
        )
        current_width = (
            0
            if self._active_verify_tokens_buf is None
            else int(self._active_verify_tokens_buf.shape[1])
        )
        if current_cap >= int(bs) and current_width == width:
            return
        new_cap = max(int(bs), current_cap * 2 if current_cap > 0 else int(bs))
        self._active_verify_tokens_buf = torch.empty(
            (new_cap, width), dtype=torch.long, device=self.device
        )
        self._active_verify_positions_buf = torch.empty(
            (new_cap, width), dtype=torch.int64, device=self.device
        )

    def _decode_active_judge_boundary(
        self,
        *,
        judge_logits: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Decode K without mutating graph/eager judge output buffers."""

        valid_horizons = active.to(dtype=torch.int64) * (self.block_size - 1)
        return compute_joint_judge_accept_len(
            judge_logits=judge_logits,
            threshold=self.active_judge_threshold,
            mode=self.active_judge_decoder_mode,
            valid_horizons=valid_horizons,
            early_cost=self.active_judge_boundary_early_cost,
            late_cost=self.active_judge_boundary_late_cost,
            quantile=self.active_judge_boundary_quantile,
        )

    def _prepare_active_judge_speculative_decoding(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Build one variable-length proposal by chaining draft/judge blocks.

        The draft allocator is transactional: temporary synthetic-prefix KV is
        visible to later blocks in this proposal, then rolled back before the
        target allocates its fixed-width verification block.
        """
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise RuntimeError(
                "Active DFLASH drafter/judge currently supports greedy decoding "
                "only; its first-fail labels are defined against verifier argmax."
            )

        bs = batch.batch_size()
        self._ensure_active_verify_buffers(bs)
        assert self._active_verify_tokens_buf is not None
        assert self._active_verify_positions_buf is not None
        assert self._draft_block_tokens_buf is not None or bs >= 0

        verify_tokens = self._active_verify_tokens_buf[:bs]
        verify_tokens.fill_(int(self._mask_token_id))
        initial_root = draft_input.verified_id.to(
            device=self.device, dtype=torch.long
        )
        verify_tokens[:, 0].copy_(initial_root)

        accumulated = torch.zeros((bs,), dtype=torch.int32, device=self.device)
        active = torch.ones((bs,), dtype=torch.bool, device=self.device)
        current_root = initial_root.clone()
        base_target_lens = batch.seq_lens.to(
            device=self.device, dtype=torch.int32
        )
        judge_block_accept_lens: list[list[int]] = []
        judge_blocks_run = [0] * bs
        judge_first_fail_blocks = [0] * bs
        active_cpu = [True] * bs
        accumulated_cpu = [0] * bs
        proposal_hidden = None

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        # Context KV is permanent state, not part of the synthetic-block
        # allocator transaction.  Materialize it once outside the loop so the
        # indexed drafter timing spans only the actual block model + LM head.
        self._timing.mark("draft_kv_pre_start")
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        self._timing.mark("draft_kv_pre_end")
        base_draft_lens = draft_input.draft_seq_lens.to(
            device=self.device, dtype=torch.int32
        ).clone()
        # Active mode rejects compact draft caching, so the just-materialized
        # draft prefix exactly follows the scheduler's existing CPU seq_lens.
        # Reusing it avoids two additional GPU-host synchronizations per block.
        base_draft_lens_cpu = batch.seq_lens_cpu.to(dtype=torch.int32)

        allocator = self.draft_model_runner.token_to_kv_pool_allocator
        # Active judge keeps accepted synthetic rows across several draft
        # blocks inside one outer allocator transaction. Make all necessary
        # radix eviction permanent before taking that transaction snapshot;
        # inner calls use restore_allocator=False and must not mutate the tree.
        active_transient_reserve = bs * int(self.active_verify_width)
        if self.page_size > 1:
            active_transient_reserve += bs * self.page_size
        evict_from_tree_cache(batch.tree_cache, active_transient_reserve)
        transaction_state = allocator.backup_state()
        try:
            for block_idx in range(self.active_judge_max_blocks):
                target_prefix_lens = base_target_lens + accumulated
                draft_prefix_lens = base_draft_lens + accumulated
                draft_prefix_lens_cpu = base_draft_lens_cpu + torch.tensor(
                    accumulated_cpu, dtype=torch.int32
                )

                self._timing.mark(f"active_draft_start_{block_idx}")
                (
                    draft_hidden,
                    positions_2d,
                    block_ids,
                    lm_head,
                    forward_batch,
                ) = self._run_draft_block_forward(
                    batch,
                    draft_input,
                    root_ids=current_root,
                    target_prefix_lens=target_prefix_lens,
                    draft_prefix_lens=draft_prefix_lens,
                    draft_prefix_lens_cpu=draft_prefix_lens_cpu,
                    append_target_context=False,
                    restore_allocator=False,
                    return_forward_batch=True,
                    timing_block_idx=block_idx,
                )

                self._timing.mark("draft_head_start")
                self._timing.mark(f"active_draft_head_start_{block_idx}")
                draft_next = self._greedy_sample_from_vocab_parallel_head(
                    hidden_states=draft_hidden[:, 1:, :].reshape(
                        -1, draft_hidden.shape[-1]
                    ),
                    lm_head=lm_head,
                ).view(bs, self.block_size - 1)
                self._timing.mark(f"active_draft_head_end_{block_idx}")
                self._timing.mark("draft_head_end")
                self._timing.mark(f"active_draft_end_{block_idx}")
                assert self._draft_block_tokens_buf is not None
                draft_tokens = self._draft_block_tokens_buf[:bs]
                draft_tokens[:, 0].copy_(block_ids[:, 0])
                draft_tokens[:, 1:].copy_(draft_next)
                # Draft and judge graphs deliberately share a CUDA graph memory
                # pool.  Preserve draft hidden states only for the optional
                # trace path, which consumes them after judge replay.
                draft_hidden_for_trace = (
                    draft_hidden.clone()
                    if self._trace_draft_hidden_enabled
                    else draft_hidden
                )

                # Training replaces every MASK slot by the hard draft token
                # embedding before the independent isomorphic judge tower.
                self._timing.mark(f"active_judge_start_{block_idx}")
                judge_input_embeds = embed_module(draft_tokens).reshape(
                    -1, draft_hidden.shape[-1]
                )
                self._timing.mark(f"active_judge_model_start_{block_idx}")
                with torch.inference_mode():
                    graph_runner = self.draft_model_runner.graph_runner
                    used_judge_graph = bool(
                        graph_runner is not None
                        and graph_runner.can_replay_dflash_judge(forward_batch)
                    )
                    if used_judge_graph:
                        judge_logits = graph_runner.replay_dflash_judge(
                            forward_batch,
                            input_ids=draft_tokens.reshape(-1),
                            input_embeds=judge_input_embeds,
                        ).view(bs, self.block_size)
                        if (
                            self._judge_graph_validation_count
                            < self._judge_graph_validation_steps
                        ):
                            # Clone before eager replay: paired CUDA graphs share
                            # a private pool, so their output buffers only have a
                            # bounded lifetime.
                            graph_logits = judge_logits.clone()
                            self.draft_model_runner.attn_backend.init_forward_metadata(
                                forward_batch
                            )
                            eager_logits = self.draft_model.joint_judge_forward(
                                input_ids=draft_tokens.reshape(-1),
                                positions=positions_2d.reshape(-1),
                                forward_batch=forward_batch,
                                input_embeds=judge_input_embeds,
                            ).view(bs, self.block_size)
                            self._judge_graph_validation_count += 1

                            graph_finite = torch.isfinite(graph_logits).all()
                            eager_finite = torch.isfinite(eager_logits).all()
                            graph_accept = self._decode_active_judge_boundary(
                                judge_logits=graph_logits,
                                active=active,
                            )
                            eager_accept = self._decode_active_judge_boundary(
                                judge_logits=eager_logits,
                                active=active,
                            )
                            accept_mismatch = graph_accept != eager_accept
                            delta = (
                                graph_logits.float() - eager_logits.float()
                            ).abs()
                            max_abs = float(delta.max().item())
                            mean_abs = float(delta.mean().item())
                            has_nonfinite = not bool(
                                (graph_finite & eager_finite).item()
                            )
                            has_accept_mismatch = bool(
                                accept_mismatch.any().item()
                            )
                            captured_bs = int(getattr(graph_runner, "bs", bs))
                            if self.tp_rank == 0:
                                logger.info(
                                    "DFLASH_JUDGE_GRAPH_VALIDATE step=%s/%s "
                                    "raw_bs=%s captured_bs=%s max_abs=%.6g "
                                    "mean_abs=%.6g accept_mismatch=%s "
                                    "decoder=%s graph_accept=%s eager_accept=%s",
                                    self._judge_graph_validation_count,
                                    self._judge_graph_validation_steps,
                                    bs,
                                    captured_bs,
                                    max_abs,
                                    mean_abs,
                                    int(accept_mismatch.sum().item()),
                                    self.active_judge_decoder_mode,
                                    graph_accept.tolist(),
                                    eager_accept.tolist(),
                                )
                            if has_nonfinite or has_accept_mismatch:
                                raise RuntimeError(
                                    "DFlash judge CUDA graph/eager validation "
                                    "failed: "
                                    f"nonfinite={has_nonfinite}, "
                                    "accept_mismatch="
                                    f"{int(accept_mismatch.sum().item())}, "
                                    f"max_abs={max_abs:.6g}, "
                                    f"mean_abs={mean_abs:.6g}, "
                                    f"raw_bs={bs}, captured_bs={captured_bs}."
                                )
                            # Serving decisions must continue to use the graph
                            # result being validated, not the shadow eager pass.
                            judge_logits = graph_logits
                    else:
                        # An eager draft pass has raw-size metadata, while an
                        # unpaired graph replay may have padded metadata.  The
                        # fallback therefore rebuilds the exact eager metadata.
                        self.draft_model_runner.attn_backend.init_forward_metadata(
                            forward_batch
                        )
                        judge_logits = self.draft_model.joint_judge_forward(
                            input_ids=draft_tokens.reshape(-1),
                            positions=positions_2d.reshape(-1),
                            forward_batch=forward_batch,
                            input_embeds=judge_input_embeds,
                        ).view(bs, self.block_size)
                self._timing.mark(f"active_judge_model_end_{block_idx}")
                self._timing.mark(f"active_judge_end_{block_idx}")
                accepted = self._decode_active_judge_boundary(
                    judge_logits=judge_logits,
                    active=active,
                )
                (
                    accepted,
                    current_root,
                    accumulated,
                    active,
                ) = append_joint_judge_accepted_prefix_(
                    verify_tokens=verify_tokens,
                    draft_tokens=draft_tokens,
                    accumulated_lens=accumulated,
                    active=active,
                    accepted_lens=accepted,
                )
                accepted_cpu = accepted.detach().to(device="cpu").tolist()
                if self._trace_draft_hidden_enabled:
                    if proposal_hidden is None:
                        proposal_hidden = torch.empty(
                            (
                                bs,
                                self.active_verify_width - 1,
                                int(draft_hidden.shape[-1]),
                            ),
                            dtype=draft_hidden.dtype,
                            device=draft_hidden.device,
                        )
                    for row, was_active in enumerate(active_cpu):
                        accepted_in_block = int(accepted_cpu[row])
                        if not was_active or accepted_in_block <= 0:
                            continue
                        start = int(accumulated_cpu[row])
                        proposal_hidden[row, start : start + accepted_in_block].copy_(
                            draft_hidden_for_trace[
                                row, 1 : 1 + accepted_in_block
                            ]
                        )
                judge_block_accept_lens.append(accepted_cpu)
                for row, was_active in enumerate(active_cpu):
                    if not was_active:
                        continue
                    judge_blocks_run[row] += 1
                    accumulated_cpu[row] += int(accepted_cpu[row])
                    if accepted_cpu[row] == 0:
                        judge_first_fail_blocks[row] += 1
                        active_cpu[row] = False

                # Only slots strictly before d_k form the next synthetic
                # prefix.  Slot k is d_k's old MASK-state KV and must be
                # re-materialized as the next hard-token anchor; everything
                # from k onward can be returned immediately.  This bounds peak
                # transient allocation by the fixed target verify width W
                # instead of max_blocks * block_size.
                block_cache_loc = forward_batch.out_cache_loc.view(
                    bs, self.block_size
                )
                retain = (
                    torch.arange(
                        self.block_size, device=self.device, dtype=torch.int32
                    ).unsqueeze(0)
                    < accepted.unsqueeze(1)
                )
                allocator.free(block_cache_loc[~retain])

                if not any(length > 0 for length in accepted_cpu):
                    break
        finally:
            allocator.restore_state(transaction_state)

        target_verify_width = int(self.active_verify_width)
        if self.active_verify_bucketing_enabled:
            target_verify_width = select_active_dflash_verify_width(
                max_accumulated_draft_tokens=max(accumulated_cpu, default=0),
                block_size=self.block_size,
                max_verify_width=self.active_verify_width,
            )
        if max(accumulated_cpu, default=0) >= target_verify_width:
            raise RuntimeError(
                "Active DFLASH proposal does not fit selected target verify "
                f"width: proposal_lens={accumulated_cpu}, "
                f"verify_width={target_verify_width}."
            )

        verify_positions = self._active_verify_positions_buf[:bs]
        verify_offsets = torch.arange(
            target_verify_width, dtype=torch.int64, device=self.device
        )
        torch.add(
            base_target_lens.to(torch.int64).unsqueeze(1),
            verify_offsets.unsqueeze(0),
            out=verify_positions[:, :target_verify_width],
        )
        if self.active_verify_bucketing_enabled:
            (
                target_verify_tokens,
                target_verify_positions,
                proposal_hidden,
            ) = compact_active_dflash_verify_tensors(
                verify_tokens=verify_tokens,
                verify_positions=verify_positions,
                proposal_hidden=proposal_hidden,
                verify_width=target_verify_width,
            )
        else:
            target_verify_tokens = verify_tokens
            target_verify_positions = verify_positions
        verify_input = DFlashVerifyInput(
            draft_token=target_verify_tokens.reshape(-1),
            positions=target_verify_positions.reshape(-1),
            draft_token_num=target_verify_width,
            valid_draft_lens=accumulated,
            proposal_hidden=proposal_hidden,
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

        self._last_active_judge_block_accept_lens = judge_block_accept_lens
        self._last_active_judge_proposal_lens = accumulated_cpu
        self._last_active_judge_blocks_run = judge_blocks_run
        self._last_active_judge_first_fail_blocks = judge_first_fail_blocks
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = verify_input
        batch.return_hidden_states = False

    def _prepare_for_speculative_decoding(
        self, batch: ScheduleBatch, draft_input: DFlashDraftInput
    ):
        if batch.forward_mode.is_extend() or batch.forward_mode.is_idle():
            return

        if batch.has_grammar:
            raise RuntimeError(
                "Invariant broken: DFLASH batch has grammar constraints, but scheduler should have rejected this request."
            )
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            if (
                self._collaborative_gate_mode is None
                and
                not is_dflash_sampling_verify_available()
                and not self._warned_sampling_fallback
                and self.tp_rank == 0
            ):
                logger.warning(
                    "DFLASH non-greedy verification is unavailable on this build/device; "
                    "falling back to greedy argmax verification."
                )
                self._warned_sampling_fallback = True

        if self.active_judge_enabled:
            self._prepare_active_judge_speculative_decoding(batch, draft_input)
            return

        bs = batch.batch_size()
        draft_hidden, positions_2d, block_ids, lm_head = self._run_draft_block_forward(
            batch, draft_input
        )
        self._timing.mark("draft_head_start")
        (
            draft_next_flat,
            proposal_behavior_logprobs,
            proposal_behavior,
            proposal_behavior_topk,
            proposal_behavior_support_ids,
        ) = self._sample_dflash_future_proposals(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        )
        draft_next = draft_next_flat.view(bs, self.block_size - 1)
        if proposal_behavior_logprobs is not None:
            proposal_behavior_logprobs = proposal_behavior_logprobs.view(
                bs, self.block_size - 1
            )
        if proposal_behavior_support_ids is not None:
            proposal_behavior_support_ids = proposal_behavior_support_ids.view(
                bs,
                self.block_size - 1,
                proposal_behavior_support_ids.shape[-1],
            )
        self._timing.mark("draft_head_end")
        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)
        positions = positions_2d.reshape(-1)

        verify_input = DFlashVerifyInput(
            draft_token=draft_tokens.reshape(-1),
            positions=positions,
            draft_token_num=self.block_size,
            proposal_hidden=(
                draft_hidden[:, 1:, :] if self._trace_draft_hidden_enabled else None
            ),
            proposal_behavior_logprobs=proposal_behavior_logprobs,
            proposal_behavior_support_ids=(
                proposal_behavior_support_ids
                if self._trace_draft_hidden_enabled
                else None
            ),
            proposal_behavior=proposal_behavior,
            proposal_behavior_topk=proposal_behavior_topk,
            proposal_behavior_temperature=(
                self._training_proposal_temperature
                if proposal_behavior_logprobs is not None
                else None
            ),
            acceptance_mode=(
                "collaborative_source"
                if self._collaborative_gate_mode is not None
                else "strict"
            ),
            collaborative_gate_mode=self._collaborative_gate_mode,
            collaborative_fixed_draft_tokens=(
                self._collaborative_fixed_draft_tokens
                if self._collaborative_gate_mode == "fixed"
                else None
            ),
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

    def _set_flow_condition(
        self,
        *,
        forward_batch: ForwardBatch,
        num_tokens: int,
        flow_t: float,
        flow_mode_id: int,
        device: torch.device,
    ) -> None:
        """Attach Flow-DFlash conditioning tensors to the draft forward batch.

        The draft model serving code should read these from either
        ``forward_batch.spec_info`` or directly from ``forward_batch``:
          - dflash_flow_t:     [num_tokens, 1], float32
          - dflash_flow_mode:  [num_tokens], long;
                                 0=denoise, 1=decode, 2=one-step direct
        """
        flow_t_tensor = torch.full(
            (int(num_tokens), 1),
            float(flow_t),
            dtype=torch.float32,
            device=device,
        )
        flow_mode_tensor = torch.full(
            (int(num_tokens),),
            int(flow_mode_id),
            dtype=torch.long,
            device=device,
        )

        spec_info = getattr(forward_batch, "spec_info", None)
        if spec_info is not None:
            setattr(spec_info, "dflash_flow_t", flow_t_tensor)
            setattr(spec_info, "dflash_flow_mode", flow_mode_tensor)

        # Also attach directly to forward_batch for draft model implementations
        # that do not look inside spec_info.
        setattr(forward_batch, "dflash_flow_t", flow_t_tensor)
        setattr(forward_batch, "dflash_flow_mode", flow_mode_tensor)

    def _clear_flow_condition(self, forward_batch: ForwardBatch) -> None:
        spec_info = getattr(forward_batch, "spec_info", None)
        if spec_info is not None:
            setattr(spec_info, "dflash_flow_t", None)
            setattr(spec_info, "dflash_flow_mode", None)
        setattr(forward_batch, "dflash_flow_t", None)
        setattr(forward_batch, "dflash_flow_mode", None)

    def _draft_forward_with_flow_condition(
        self,
        *,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
        flow_t: float,
        flow_mode_id: int,
        self_cond: Optional[torch.Tensor] = None,
        verified_root_embedding: Optional[torch.Tensor] = None,
    ):
        num_tokens = int(input_embeds.shape[0])
        self._set_flow_condition(
            forward_batch=forward_batch,
            num_tokens=num_tokens,
            flow_t=float(flow_t),
            flow_mode_id=int(flow_mode_id),
            device=input_embeds.device,
        )
        prepared_input = input_embeds
        prepare_flow_input = getattr(self.draft_model, "prepare_flow_input", None)
        if callable(prepare_flow_input):
            # Run the replicated conditioning MLP outside CUDA Graph replay.
            # SGLang then copies this prepared tensor into its existing dynamic
            # input_embeds graph buffer, so t/mode/self-conditioning never get
            # frozen at graph-capture values.
            root_kwargs = {}
            if verified_root_embedding is not None:
                root_kwargs["verified_root_embedding"] = verified_root_embedding
            prepared_input = prepare_flow_input(
                input_embeds,
                flow_t=float(flow_t),
                flow_mode_id=int(flow_mode_id),
                self_cond=self_cond,
                **root_kwargs,
            )
        forward_batch.input_embeds = prepared_input
        output = self._run_counted_draft_model_forward(forward_batch)
        if output.hidden_states is None:
            mode_name = "denoise" if int(flow_mode_id) == 0 else "decode"
            raise RuntimeError(
                f"Flow-DFLASH {mode_name} pass returned no hidden states."
            )
        return output

    def _run_counted_draft_model_forward(self, forward_batch: ForwardBatch):
        """Run the actual draft runner and record this physical proposal call."""
        self._record_draft_forward_call()
        return self.draft_model_runner.forward(forward_batch).logits_output

    def _make_flow_schedule(self, *, device: torch.device) -> torch.Tensor:
        if self.flow_schedule != "linear":
            raise RuntimeError(
                f"Unsupported Flow-DFlash schedule {self.flow_schedule!r}."
            )
        # NFE denotes physical draft-model evaluations.  Heun uses a
        # predictor and a corrector evaluation per time interval.
        num_intervals = (
            int(self.flow_steps) // 2
            if getattr(self, "flow_solver", "euler") == "heun"
            else int(self.flow_steps)
        )
        return torch.linspace(
            float(self.flow_t_min),
            float(self.flow_t_max),
            num_intervals + 1,
            dtype=torch.float32,
            device=device,
        )

    def _make_dflow_initial_latent(
        self,
        *,
        bs: int,
        input_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the training-matched DFlow latent and normalized anchor.

        Tensor-parallel ranks must start from identical random latents because
        the DFlow hidden state is replicated at the entrance to each TP block.
        """
        hidden_size = int(input_embeds.shape[-1])
        raw_block = input_embeds.view(int(bs), int(self.block_size), hidden_size)
        normalize_latent = getattr(self.draft_model, "normalize_latent", None)
        if not callable(normalize_latent):
            raise RuntimeError("DFlowDraftModel does not implement normalize_latent().")
        anchor_norm = normalize_latent(raw_block[:, :1, :]).contiguous()

        if self._dflow_fixed_mask_initial_state:
            make_initial_state = getattr(
                self.draft_model, "make_fixed_mask_initial_state", None
            )
            if not callable(make_initial_state):
                raise RuntimeError(
                    "Fixed-MASK DFlow model does not implement "
                    "make_fixed_mask_initial_state()."
                )
            if int(self.flow_noise_samples) != 1:
                raise RuntimeError(
                    "Fixed-MASK token-embedding ELF requires exactly one "
                    "initial state; noise ensembles are unsupported."
                )
            z = make_initial_state(
                anchor_norm, block_size=int(self.block_size)
            ).contiguous()
            expected_shape = (int(bs), int(self.block_size), hidden_size)
            if tuple(z.shape) != expected_shape:
                raise RuntimeError(
                    "Fixed-MASK DFlow initial state shape mismatch: "
                    f"expected {expected_shape}, got {tuple(z.shape)}."
                )
            return z.view(-1, hidden_size), anchor_norm

        tp_group = get_tp_group()
        noise_samples = int(self.flow_noise_samples)
        # Keep M=1's exact shape and RNG call for backward-compatible serving.
        latent_shape = (
            (int(bs), int(self.block_size), hidden_size)
            if noise_samples == 1
            else (
                noise_samples,
                int(bs),
                int(self.block_size),
                hidden_size,
            )
        )
        if int(tp_group.rank_in_group) == 0:
            z = torch.randn(
                latent_shape,
                device=input_embeds.device,
                dtype=anchor_norm.dtype,
            )
            z.mul_(float(self.flow_noise_scale))
        else:
            z = torch.empty(
                latent_shape,
                device=input_embeds.device,
                dtype=anchor_norm.dtype,
            )
        tp_group.broadcast(z, src=0)
        if noise_samples == 1:
            z[:, :1, :].copy_(anchor_norm)
        else:
            z[:, :, :1, :].copy_(anchor_norm.unsqueeze(0))
        return z.view(-1, hidden_size), anchor_norm

    def take_dflow_candidate_sample_hiddens(self) -> Optional[torch.Tensor]:
        """Consume prepared per-noise hiddens for DDTree candidate union.

        Returns ``[M, batch, block, hidden]``. These are outputs of
        ``prepare_hidden_for_lm_head`` but have not passed through ``lm_head``.
        The tensor is intentionally transaction-local so no M-fold state is
        retained across verifier rounds.
        """

        sample_hiddens = self._pending_dflow_candidate_sample_hiddens
        self._pending_dflow_candidate_sample_hiddens = None
        if (
            self.__dict__.get("flow_noise_aggregation") == "candidate_union"
            or bool(
                self.__dict__.get(
                    "_dflow_contextual_jit_same_state_sc_dual_proposal",
                    False,
                )
            )
            or bool(
                self.__dict__.get(
                    "_dflow_contextual_jit_nfe_endpoint_candidate_union",
                    False,
                )
            )
        ) and sample_hiddens is None:
            raise RuntimeError(
                "DDTree candidate-union proposal mode requested but the DFlow "
                "backbone did not publish prepared per-candidate hiddens."
            )
        return sample_hiddens

    def _run_flow_dflash_draft_forward(
        self,
        *,
        forward_batch: ForwardBatch,
        bs: int,
        input_embeds: torch.Tensor,
        verified_root_embedding: Optional[torch.Tensor] = None,
    ):
        """Run the DFlash draft block with optional K-step Flow-DFlash proposal.

        off:
            original DFlash path, one draft forward.
        denoise:
            one time-conditioned denoise pass; useful for ablation.
        denoise_decode:
            K Euler denoise steps from t_min to t_max, followed by a final
            decode pass at t=1.0. The target verifier is unchanged.
        token_embedding_elf:
            Exactly K x-prediction/Euler forwards.  The last uniform-Euler
            step is assigned directly from x_pred, and the normalized endpoint
            is returned for L2 token-embedding retrieval.  There is no decode
            forward.
        one_step:
            one direct contextual-x0 prediction from anchored Gaussian noise.
            This is the exact serving contract of one-step-direct checkpoints.

        Native DFlash flow ablations retain anchor+MASK embeddings.  A native
        DFlowDraftModel instead follows SpecForge exactly: normalized anchor +
        synchronized Gaussian latent, x-prediction Euler integration with
        self-conditioning, then a separate decode-mode pass and adapter.
        """
        verified_root_enabled = bool(
            self.__dict__.get("_dflow_verified_root_conditioning", False)
        )
        if verified_root_enabled != (verified_root_embedding is not None):
            requirement = "requires" if verified_root_enabled else "forbids"
            raise RuntimeError(
                f"This DFlow worker {requirement} a verified-root embedding."
            )
        if verified_root_enabled and self.flow_mode != "one_step":
            raise RuntimeError(
                "Verified-root DFlow checkpoints require flow_mode=one_step."
            )

        if self.flow_mode == "off":
            self._clear_flow_condition(forward_batch)
            forward_batch.input_embeds = input_embeds
            return self._run_counted_draft_model_forward(forward_batch)

        expected_tokens = int(bs) * int(self.block_size)
        if int(input_embeds.shape[0]) != expected_tokens:
            raise RuntimeError(
                "Flow-DFLASH input embedding shape mismatch: expected first dim "
                f"{expected_tokens}, got {tuple(input_embeds.shape)}."
            )

        if self.flow_mode == "denoise":
            return self._draft_forward_with_flow_condition(
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                flow_t=float(self.flow_t_min),
                flow_mode_id=0,
            )

        if self.flow_mode == "one_step":
            if not self._is_dflow_model or not self._dflow_one_step_direct:
                raise RuntimeError(
                    "flow_mode=one_step requires a one_step_direct "
                    "DFlowDraftModel checkpoint."
                )
            self._timing.mark("flow_total_start")
            self._timing.mark("flow_init_start")
            z, _anchor_norm = self._make_dflow_initial_latent(
                bs=bs,
                input_embeds=input_embeds,
            )
            self._timing.mark("flow_init_end")
            self._timing.mark("flow_denoise_start")
            direct_output = self._draft_forward_with_flow_condition(
                forward_batch=forward_batch,
                input_embeds=z,
                flow_t=0.0,
                flow_mode_id=2,
                self_cond=(
                    None
                    if self._dflow_fixed_conditioning_folded
                    else torch.zeros_like(z)
                ),
                verified_root_embedding=verified_root_embedding,
            )
            self._timing.mark("flow_denoise_end")
            prepare_hidden_for_lm_head = getattr(
                self.draft_model, "prepare_hidden_for_lm_head", None
            )
            if not callable(prepare_hidden_for_lm_head):
                raise RuntimeError(
                    "DFlowDraftModel does not implement prepare_hidden_for_lm_head()."
                )
            sample_latents = _reshape_dflow_noise_sample_hidden(
                direct_output.hidden_states,
                noise_samples=self.flow_noise_samples,
                batch_size=bs,
                block_size=self.block_size,
            )
            # Preserve the reversible [M, batch, block, hidden] boundary for
            # candidate-union decoding. No vocabulary projection has happened.
            setattr(direct_output, "dflow_sample_latents", sample_latents)
            if self.flow_noise_aggregation == "latent_mean":
                mean_latent = _mean_dflow_sample_latents(sample_latents)
                direct_output.hidden_states = prepare_hidden_for_lm_head(
                    mean_latent.reshape(-1, mean_latent.shape[-1])
                )
            elif self.flow_noise_aggregation == "candidate_union":
                prepared_samples = prepare_hidden_for_lm_head(
                    sample_latents.reshape(-1, sample_latents.shape[-1])
                ).view_as(sample_latents)
                self._pending_dflow_candidate_sample_hiddens = prepared_samples
                # Keep the legacy return shape valid. DDTree must use the
                # explicit transaction-local sample interface above instead of
                # this sample-0 compatibility view.
                direct_output.hidden_states = prepared_samples[0].reshape(
                    -1, prepared_samples.shape[-1]
                )
            else:
                raise RuntimeError(
                    "Unknown DFlow noise aggregation "
                    f"{self.flow_noise_aggregation!r}."
                )
            self._timing.mark("flow_total_end")
            return direct_output

        if self.flow_mode not in {"denoise_decode", "token_embedding_elf"}:
            raise RuntimeError(f"Unknown DFLASH flow_mode={self.flow_mode!r}")
        token_embedding_elf = self.flow_mode == "token_embedding_elf"
        direct_lm_head_terminal = bool(
            self._is_dflow_model
            and getattr(self, "_dflow_direct_lm_head_terminal", False)
        )
        if token_embedding_elf and not self._is_dflow_token_embedding_model:
            raise RuntimeError(
                "flow_mode=token_embedding_elf requires a "
                "DFlowTokenEmbeddingDraftModel checkpoint."
            )

        self._timing.mark("flow_total_start")
        self._timing.mark("flow_init_start")
        if self._is_dflow_model:
            z, anchor_norm = self._make_dflow_initial_latent(
                bs=bs,
                input_embeds=input_embeds,
            )
            self_cond = torch.zeros_like(z)
        else:
            z = input_embeds.contiguous()
            anchor_norm = None
            self_cond = None
        schedule = self._make_flow_schedule(device=z.device)
        # Retain epsilon only for the opt-in training-free bridge solver. The
        # normal beta=0 path holds no extra state and executes its original
        # operations unchanged.
        initial_noise = (
            z
            if token_embedding_elf
            and self.__dict__.get("elf_latest_endpoint_bridge_mix", 0.0) > 0.0
            else None
        )
        self._timing.mark("flow_init_end")

        if (
            direct_lm_head_terminal
            and bool(
                self.__dict__.get(
                    "_dflow_contextual_jit_same_state_sc_dual_proposal",
                    False,
                )
            )
        ):
            # This exactly mirrors the *self-conditioned* training branch at
            # a single endpoint-resolution state: both evaluations see the
            # same anchored Gaussian z_0 and t=0.  They are alternative
            # continuations from the same verified root, not a 30-token linear
            # trajectory.  Publish both prepared blocks for DDTree to build a
            # two-chain union tree and preserve the conditioned block as the
            # legacy return value.
            if int(self.flow_steps) != 1:
                raise RuntimeError(
                    "Contextual JiT same-state SC dual proposal requires "
                    f"NFE=1, got NFE={self.flow_steps}."
                )
            t0 = float(schedule[0].item())
            if t0 != 0.0:
                raise RuntimeError(
                    "Contextual JiT same-state SC dual proposal requires "
                    f"t_0=0.0, got {t0}."
                )
            self._timing.mark("flow_denoise_start")
            pilot_output = self._draft_forward_with_flow_condition(
                forward_batch=forward_batch,
                input_embeds=z,
                flow_t=t0,
                flow_mode_id=0,
                self_cond=torch.zeros_like(z),
            )
            pilot_pred = pilot_output.hidden_states
            if int(pilot_pred.shape[0]) != expected_tokens:
                raise RuntimeError(
                    "Contextual JiT SC pilot hidden shape mismatch: expected "
                    f"first dim {expected_tokens}, got {tuple(pilot_pred.shape)}."
                )
            # CUDA-graph replay reuses the draft runner output buffer. Take the
            # pilot snapshot *before* the conditioned replay so branch A does
            # not silently alias and get overwritten by branch B.
            pilot_snapshot = pilot_pred.detach().clone()
            conditioned_output = self._draft_forward_with_flow_condition(
                forward_batch=forward_batch,
                input_embeds=z,
                flow_t=t0,
                flow_mode_id=0,
                self_cond=pilot_snapshot,
            )
            conditioned_pred = conditioned_output.hidden_states
            if int(conditioned_pred.shape[0]) != expected_tokens:
                raise RuntimeError(
                    "Contextual JiT SC-conditioned hidden shape mismatch: "
                    f"expected first dim {expected_tokens}, got "
                    f"{tuple(conditioned_pred.shape)}."
                )
            assert anchor_norm is not None
            # Keep both candidate tensors immutable while restoring the
            # verified anchor below.  In particular, ``pilot_snapshot`` was
            # captured before replay #2, whose CUDA-graph output buffer may
            # alias ``pilot_pred``.
            pilot_endpoint = pilot_snapshot.clone()
            conditioned_endpoint = conditioned_pred.detach().clone()
            pilot_endpoint.view(bs, self.block_size, -1)[:, :1, :].copy_(
                anchor_norm
            )
            conditioned_endpoint.view(bs, self.block_size, -1)[:, :1, :].copy_(
                anchor_norm
            )
            prepare_hidden_for_lm_head = getattr(
                self.draft_model, "prepare_hidden_for_lm_head", None
            )
            if not callable(prepare_hidden_for_lm_head):
                raise RuntimeError(
                    "DFlowDraftModel does not implement "
                    "prepare_hidden_for_lm_head()."
                )
            candidate_endpoints = torch.stack(
                (pilot_endpoint, conditioned_endpoint), dim=0
            )
            prepared_candidates = prepare_hidden_for_lm_head(
                candidate_endpoints.reshape(-1, candidate_endpoints.shape[-1])
            ).view(
                2,
                bs,
                self.block_size,
                candidate_endpoints.shape[-1],
            )
            self._pending_dflow_candidate_sample_hiddens = prepared_candidates
            conditioned_output.hidden_states = prepared_candidates[1].reshape(
                -1, prepared_candidates.shape[-1]
            )
            self._timing.mark("flow_denoise_end")
            self._timing.mark("flow_total_end")
            return conditioned_output

        collect_contextual_jit_endpoint_candidates = bool(
            direct_lm_head_terminal
            and self.__dict__.get(
                "_dflow_contextual_jit_nfe_endpoint_candidate_union", False
            )
        )
        contextual_jit_endpoint_candidates = (
            [] if collect_contextual_jit_endpoint_candidates else None
        )

        # K-step Euler integration using x-prediction:
        #   v(z_t, t) = (x_pred - z_t) / (1 - t)
        #   z_{t+dt} = z_t + dt * v(z_t, t)
        self._timing.mark("flow_denoise_start")
        if token_embedding_elf and getattr(self, "flow_solver", "euler") == "heun":
            # Explicit trapezoidal (Heun) integration.  ``flow_steps`` is NFE,
            # so each of the K/2 macro intervals receives exactly two physical
            # draft-model calls: an Euler predictor at (z_t, t), followed by a
            # corrector at (z_predictor, t + dt).  In particular, NFE=4 means
            # two Heun intervals, not four intervals/eight forwards.
            #
            # The final corrector is intentionally evaluated at t=1.0.  Its
            # endpoint-parameterized velocity uses the checkpoint's existing
            # ``t_eps`` clamp, making this a clearly labelled solver ablation
            # rather than the historical endpoint-Euler fast path.
            for interval_idx in range(int(self.flow_steps) // 2):
                t0 = float(schedule[interval_idx].item())
                t1 = float(schedule[interval_idx + 1].item())
                dt = t1 - t0

                predictor_output = self._draft_forward_with_flow_condition(
                    forward_batch=forward_batch,
                    input_embeds=z,
                    flow_t=t0,
                    flow_mode_id=0,
                    self_cond=self_cond,
                )
                x_pred_predictor = predictor_output.hidden_states
                if int(x_pred_predictor.shape[0]) != expected_tokens:
                    raise RuntimeError(
                        "Flow-DFLASH Heun predictor hidden shape mismatch: "
                        f"expected first dim {expected_tokens}, got "
                        f"{tuple(x_pred_predictor.shape)}."
                    )
                predictor_denom = max(1.0 - t0, float(self.flow_t_eps))
                predictor_velocity = (x_pred_predictor - z) / predictor_denom
                z_predictor = (z + dt * predictor_velocity).contiguous()
                assert anchor_norm is not None
                z_predictor.view(bs, self.block_size, -1)[:, :1, :].copy_(
                    anchor_norm
                )

                # This is the serving-consistent SC ordering: a call at the
                # predictor state consumes the immediately preceding endpoint;
                # after the correction, the next interval consumes the
                # correction endpoint.  SC-off checkpoints retain their
                # persistent all-zero state for both calls.
                corrector_self_cond = (
                    x_pred_predictor.detach()
                    if self._token_embedding_serving_self_conditioning
                    else self_cond
                )
                denoise_output = self._draft_forward_with_flow_condition(
                    forward_batch=forward_batch,
                    input_embeds=z_predictor,
                    flow_t=t1,
                    flow_mode_id=0,
                    self_cond=corrector_self_cond,
                )
                x_pred_corrector = denoise_output.hidden_states
                if int(x_pred_corrector.shape[0]) != expected_tokens:
                    raise RuntimeError(
                        "Flow-DFLASH Heun corrector hidden shape mismatch: "
                        f"expected first dim {expected_tokens}, got "
                        f"{tuple(x_pred_corrector.shape)}."
                    )
                corrector_denom = max(1.0 - t1, float(self.flow_t_eps))
                corrector_velocity = (
                    x_pred_corrector - z_predictor
                ) / corrector_denom
                z = (
                    z
                    + 0.5 * dt * (predictor_velocity + corrector_velocity)
                ).contiguous()
                z.view(bs, self.block_size, -1)[:, :1, :].copy_(anchor_norm)
                if self._token_embedding_serving_self_conditioning:
                    self_cond = x_pred_corrector.detach()

            self._timing.mark("flow_denoise_end")
            # Token-embedding proposal construction consumes normalized raw
            # token-embedding latents directly; there is still no decode pass.
            denoise_output.hidden_states = z
            self._timing.mark("flow_total_end")
            return denoise_output

        for step_idx in range(int(self.flow_steps)):
            t0 = float(schedule[step_idx].item())
            t1 = float(schedule[step_idx + 1].item())
            # ``sc0`` checkpoints keep the zero tensor initialized above for
            # every Euler call: the update below intentionally does not
            # replace it with the preceding x-prediction.
            step_self_cond = self_cond
            if token_embedding_elf and bool(
                getattr(
                    self,
                    "_same_time_self_cond_correction_enabled",
                    False,
                )
            ):
                # Match the training-time self-conditioning branch exactly:
                # first predict from the unchanged (z_t, t) with zero/None
                # self-conditioning, detach that endpoint, then correct from
                # the same (z_t, t).  Do not perform an Euler update between
                # these two physical draft-model evaluations.
                pilot_output = self._draft_forward_with_flow_condition(
                    forward_batch=forward_batch,
                    input_embeds=z,
                    flow_t=t0,
                    flow_mode_id=0,
                    self_cond=None,
                )
                pilot_pred = pilot_output.hidden_states
                if int(pilot_pred.shape[0]) != expected_tokens:
                    raise RuntimeError(
                        "Flow-DFLASH self-conditioning pilot hidden shape "
                        "mismatch: expected first dim "
                        f"{expected_tokens}, got {tuple(pilot_pred.shape)}."
                    )
                step_self_cond = pilot_pred.detach()
            denoise_output = self._draft_forward_with_flow_condition(
                forward_batch=forward_batch,
                input_embeds=z,
                flow_t=t0,
                flow_mode_id=0,
                self_cond=step_self_cond,
            )
            x_pred = denoise_output.hidden_states
            if int(x_pred.shape[0]) != expected_tokens:
                raise RuntimeError(
                    "Flow-DFLASH denoise hidden shape mismatch: expected first dim "
                    f"{expected_tokens}, got {tuple(x_pred.shape)}."
                )

            if contextual_jit_endpoint_candidates is not None:
                # Every contextual JiT Euler call directly predicts x_1. CUDA
                # graph replay reuses the output buffer on the next call, so
                # snapshot this endpoint before the Euler update / next replay.
                # In particular, z_{t+1} is an intermediate noisy state and
                # must not be published as a draft candidate.
                endpoint_prediction = x_pred.detach().clone()
                assert anchor_norm is not None
                endpoint_prediction.view(bs, self.block_size, -1)[:, :1, :].copy_(
                    anchor_norm
                )
                contextual_jit_endpoint_candidates.append(endpoint_prediction)

            if (
                (token_embedding_elf or direct_lm_head_terminal)
                and step_idx + 1 == int(self.flow_steps)
            ):
                # For the uniform grid, dt == 1 - t on the last interval.
                # This is algebraically Euler's update.  Both the raw
                # token-embedding endpoint and contextual JiT direct-LM-head
                # endpoint use the direct assignment in SpecForge to avoid a
                # final BF16 roundoff before proposal construction.
                z = x_pred.contiguous()
            else:
                denom = max(1.0 - t0, float(self.flow_t_eps))
                dt = t1 - t0
                velocity = (x_pred - z) / denom
                z = (
                    z + self.elf_euler_velocity_scale * dt * velocity
                ).contiguous()
                if (
                    token_embedding_elf
                    and initial_noise is not None
                    and self.elf_latest_endpoint_bridge_mix > 0.0
                    and step_idx > 0
                ):
                    # A student rollout at nominal t can be slower than the
                    # teacher bridge used during training. Rebuild that bridge
                    # around the *same forward's* endpoint prediction and mix
                    # it in without an extra draft call or any future token.
                    # Step 0 is algebraically identical to Euler when t_min=0,
                    # so leave its baseline BF16 operation order untouched.
                    bridge = (1.0 - t1) * initial_noise + t1 * x_pred
                    beta = self.elf_latest_endpoint_bridge_mix
                    z = ((1.0 - beta) * z + beta * bridge).contiguous()
            if self._is_dflow_model:
                assert anchor_norm is not None
                z.view(bs, self.block_size, -1)[:, :1, :].copy_(anchor_norm)
                if self._dflow_serving_self_conditioning:
                    self_cond = x_pred.detach()
        self._timing.mark("flow_denoise_end")

        if token_embedding_elf:
            # ``z`` is still normalized raw token-embedding space.  Proposal
            # construction denormalizes and performs the checkpoint-selected
            # vocabulary projection.  Returning here is the critical
            # no-extra-decode-forward train/serve invariant.
            denoise_output.hidden_states = z
            self._timing.mark("flow_total_end")
            return denoise_output

        if direct_lm_head_terminal:
            # The contextual continuous-JiT contract has exactly K denoiser
            # calls and no decode-mode call.  `z` is normalized target
            # post-final-norm hidden space, so only undo its scalar
            # normalization and apply the (validated identity) decoder before
            # the frozen target LM head.
            prepare_hidden_for_lm_head = getattr(
                self.draft_model, "prepare_hidden_for_lm_head", None
            )
            if not callable(prepare_hidden_for_lm_head):
                raise RuntimeError(
                    "DFlowDraftModel does not implement "
                    "prepare_hidden_for_lm_head()."
                )
            if contextual_jit_endpoint_candidates is not None:
                if len(contextual_jit_endpoint_candidates) != int(self.flow_steps):
                    raise RuntimeError(
                        "Contextual JiT endpoint candidate count mismatch: "
                        f"collected={len(contextual_jit_endpoint_candidates)}, "
                        f"NFE={self.flow_steps}."
                    )
                candidate_endpoints = torch.stack(
                    contextual_jit_endpoint_candidates, dim=0
                )
                prepared_candidates = prepare_hidden_for_lm_head(
                    candidate_endpoints.reshape(
                        -1, candidate_endpoints.shape[-1]
                    )
                ).view(
                    int(self.flow_steps),
                    bs,
                    self.block_size,
                    candidate_endpoints.shape[-1],
                )
                self._pending_dflow_candidate_sample_hiddens = prepared_candidates
                # The return keeps the final Euler endpoint for compatibility;
                # DDTree consumes the explicit M-way tensor above as sibling
                # chains, rather than a fictitious 8x-long linear proposal.
                denoise_output.hidden_states = prepared_candidates[-1].reshape(
                    -1, prepared_candidates.shape[-1]
                )
            else:
                denoise_output.hidden_states = prepare_hidden_for_lm_head(z)
            self._timing.mark("flow_total_end")
            return denoise_output

        self._timing.mark("flow_decode_start")
        decode_output = self._draft_forward_with_flow_condition(
            forward_batch=forward_batch,
            input_embeds=z,
            flow_t=1.0,
            flow_mode_id=1,
            self_cond=(torch.zeros_like(z) if self._is_dflow_model else None),
        )
        if int(decode_output.hidden_states.shape[0]) != expected_tokens:
            raise RuntimeError(
                "Flow-DFLASH decode hidden shape mismatch: expected first dim "
                f"{expected_tokens}, got {tuple(decode_output.hidden_states.shape)}."
            )
        if self._is_dflow_model:
            prepare_hidden_for_lm_head = getattr(
                self.draft_model, "prepare_hidden_for_lm_head", None
            )
            if not callable(prepare_hidden_for_lm_head):
                raise RuntimeError(
                    "DFlowDraftModel does not implement prepare_hidden_for_lm_head()."
                )
            decode_output.hidden_states = prepare_hidden_for_lm_head(
                decode_output.hidden_states
            )
        self._timing.mark("flow_decode_end")
        self._timing.mark("flow_total_end")
        return decode_output

    def _validate_head_lora_vocab_shard(self, lm_head) -> None:
        if self._head_lora_vocab_shard_validated:
            return
        if int(getattr(self.draft_model, "head_lora_rank", 0) or 0) <= 0:
            self._head_lora_vocab_shard_validated = True
            return

        head_lora_b = getattr(self.draft_model, "head_lora_b", None)
        if head_lora_b is None or not hasattr(head_lora_b, "weight"):
            raise RuntimeError(
                "DFLASH config enables Head-LoRA but the draft model has no B weight."
            )
        if int(head_lora_b.weight.shape[0]) != int(lm_head.weight.shape[0]):
            raise RuntimeError(
                "DFLASH Head-LoRA B and target LM head use different local vocab "
                f"sizes: B={tuple(head_lora_b.weight.shape)}, "
                f"target={tuple(lm_head.weight.shape)}"
            )
        b_shard = getattr(head_lora_b, "shard_indices", None)
        target_shard = getattr(lm_head, "shard_indices", None)
        if b_shard is None or target_shard is None:
            raise RuntimeError(
                "DFLASH Head-LoRA requires shard_indices on both B and target head."
            )
        shard_fields = (
            "org_vocab_start_index",
            "org_vocab_end_index",
            "added_vocab_start_index",
            "added_vocab_end_index",
            "num_org_elements_padded",
            "num_added_elements_padded",
        )
        mismatches = {
            field: (int(getattr(b_shard, field)), int(getattr(target_shard, field)))
            for field in shard_fields
            if int(getattr(b_shard, field)) != int(getattr(target_shard, field))
        }
        if mismatches:
            raise RuntimeError(
                "DFLASH Head-LoRA B does not match the target LM-head vocab shard: "
                f"{mismatches}"
            )
        self._head_lora_vocab_shard_validated = True
        if self.tp_rank == 0:
            logger.info(
                "DFLASH Head-LoRA enabled without merge. rank=%s, alpha=%s, "
                "local_vocab_rows=%s. Verifier remains unchanged.",
                self.draft_model.head_lora_rank,
                self.draft_model.head_lora_alpha,
                int(head_lora_b.weight.shape[0]),
            )

    def _apply_head_lora_to_local_logits(
        self,
        *,
        base_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head,
        output_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        if int(getattr(self.draft_model, "head_lora_rank", 0) or 0) <= 0:
            return base_logits
        self._validate_head_lora_vocab_shard(lm_head)
        apply_head_lora = getattr(
            self.draft_model, "apply_head_lora_to_local_logits", None
        )
        if apply_head_lora is None:
            raise RuntimeError(
                "DFLASH Head-LoRA checkpoint was loaded by a draft model that "
                "does not implement unmerged local-logit projection."
            )
        return apply_head_lora(base_logits, hidden_states, output_slice)

    def _get_token_embedding_l2_geometry(self, embed_module):
        """Cache the checkpoint's centered-L2 geometry on each TP shard."""

        if not self._is_dflow_token_embedding_model:
            raise RuntimeError("L2 token geometry requested for a non-token ELF model.")
        if not hasattr(embed_module, "weight") or not hasattr(
            embed_module, "shard_indices"
        ):
            raise RuntimeError(
                "Token-embedding ELF requires vocab-parallel target input embeddings."
            )
        weight = embed_module.weight
        shard = embed_module.shard_indices
        valid_vocab_size = int(self.draft_model.valid_vocab_size)
        table_vocab_size = int(
            getattr(embed_module, "org_vocab_size", valid_vocab_size)
        )
        org_start = int(shard.org_vocab_start_index)
        org_end = int(shard.org_vocab_end_index)
        num_org = int(shard.num_org_elements)
        cache_key = (
            int(weight.data_ptr()),
            tuple(weight.shape),
            weight.dtype,
            weight.device,
            valid_vocab_size,
            table_vocab_size,
            org_start,
            org_end,
        )
        cached = self._token_embedding_l2_geometry
        if cached is not None and cached["key"] == cache_key:
            return cached

        if not 1 <= valid_vocab_size <= table_vocab_size:
            raise RuntimeError(
                "Invalid token-embedding vocabulary geometry: "
                f"valid={valid_vocab_size}, table={table_vocab_size}."
            )
        local_valid = max(0, min(org_end, valid_vocab_size) - org_start)
        if local_valid > num_org:
            raise RuntimeError(
                "Token-embedding local valid rows exceed the physical shard: "
                f"valid={local_valid}, physical={num_org}."
            )

        if valid_vocab_size < table_vocab_size:
            center_start, center_end = valid_vocab_size, table_vocab_size
            center_source = "padded_tail_mean"
        else:
            center_start, center_end = 0, valid_vocab_size
            center_source = "valid_vocab_mean"
        local_center_start = max(org_start, center_start)
        local_center_end = min(org_end, center_end)
        center_sum = torch.zeros(
            (int(weight.shape[1]),), dtype=torch.float32, device=weight.device
        )
        center_count = torch.zeros((), dtype=torch.float32, device=weight.device)
        if local_center_end > local_center_start:
            begin = local_center_start - org_start
            end = local_center_end - org_start
            # The reference computes the center in FP32 and never casts this
            # geometry cache through mixed-precision buffer policies.
            center_sum.copy_(weight[begin:end].detach().float().sum(dim=0))
            center_count.fill_(float(end - begin))
        tp_group = get_tp_group()
        if int(tp_group.world_size) > 1:
            dist.all_reduce(
                center_sum, op=dist.ReduceOp.SUM, group=tp_group.device_group
            )
            dist.all_reduce(
                center_count, op=dist.ReduceOp.SUM, group=tp_group.device_group
            )
        if float(center_count.item()) <= 0.0:
            raise RuntimeError("Token-embedding L2 center has no source rows.")
        center = center_sum / center_count
        local_weight = weight[:local_valid].detach()
        # Match the reference's bounded 4096-row geometry build.  Casting the
        # complete 151k x hidden table to FP32 at once causes a needless
        # multi-gigabyte initialization spike.
        row_norm_chunks = []
        for start in range(0, local_valid, 4096):
            end = min(start + 4096, local_valid)
            row_norm_chunks.append(
                0.5
                * (local_weight[start:end].float() - center)
                .square()
                .sum(dim=-1)
            )
        row_half_squared_norm = torch.cat(row_norm_chunks, dim=0)
        cached = {
            "key": cache_key,
            "center": center,
            "row_half_squared_norm": row_half_squared_norm,
            "local_valid": local_valid,
            "org_start": org_start,
            "org_end": org_end,
            "table_vocab_size": table_vocab_size,
            "center_source": center_source,
        }
        self._token_embedding_l2_geometry = cached
        if self.tp_rank == 0:
            logger.info(
                "Initialized token-embedding L2 vocabulary geometry: "
                "valid_vocab=%s table_vocab=%s center=%s exact_rerank_k=32.",
                valid_vocab_size,
                table_vocab_size,
                center_source,
            )
        return cached

    @staticmethod
    def _token_embedding_fp32_mm(
        query: torch.Tensor, weight_t: torch.Tensor
    ) -> torch.Tensor:
        """Use TensorCore inputs with FP32 output where the runtime supports it."""

        if query.device.type == "cuda" and query.dtype in {
            torch.float16,
            torch.bfloat16,
        }:
            try:
                return torch.mm(query, weight_t, out_dtype=torch.float32)
            except TypeError:
                pass
        if query.dtype in {torch.float16, torch.bfloat16}:
            return torch.mm(query.float(), weight_t.float())
        return torch.mm(query, weight_t)

    def _topk_log_probs_from_token_embedding_l2(
        self,
        *,
        hidden_states: torch.Tensor,
        topk: int,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project normalized ELF endpoints to global centered-L2 scores.

        Returns normalized global top-k log probabilities, token IDs, and the
        denormalized raw queries.  The score is

          (q-c) E_v^T - 0.5 ||E_v-c||^2,

        which differs from a tied LM head only by the prototype-norm bias and
        is exactly equivalent to negative half squared L2 up to a query-only
        constant.
        """

        if hidden_states.ndim != 2:
            raise ValueError(
                "Token-embedding ELF states must be [tokens, hidden], got "
                f"{tuple(hidden_states.shape)}."
            )
        if int(topk) <= 0:
            raise ValueError(f"Token-embedding topk must be positive, got {topk}.")
        denormalize = getattr(self.draft_model, "denormalize_latent", None)
        if not callable(denormalize):
            raise RuntimeError("Token-embedding ELF model cannot denormalize endpoint.")
        raw_query = denormalize(hidden_states)
        embed_module = self.target_worker.model_runner.model.get_input_embeddings()
        geometry = self._get_token_embedding_l2_geometry(embed_module)
        local_valid = int(geometry["local_valid"])
        if local_valid <= 0:
            raise RuntimeError("Token-embedding ELF found an empty local vocab shard.")
        weight = embed_module.weight[:local_valid].detach()
        center = geometry["center"]
        centered_query = (raw_query.float() - center).to(dtype=weight.dtype)
        local_logits = self._token_embedding_fp32_mm(
            centered_query.contiguous(), weight.T
        )
        local_logits.sub_(geometry["row_half_squared_norm"].unsqueeze(0))
        temperature = float(self.draft_model.retrieval_temperature)
        local_logits.div_(temperature)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        process_group = tp_group.device_group
        log_z = None
        if normalize:
            # DDTree compares prefixes of different depths and adaptive top-k
            # consumes cumulative probability mass, so it needs the global
            # full-vocabulary normalizer.  Plain deterministic DFLASH skips
            # these exponentials/collectives and only retrieves top-1.
            local_max = local_logits.amax(dim=-1)
            global_max = local_max.clone()
            if tp_size > 1:
                dist.all_reduce(
                    global_max, op=dist.ReduceOp.MAX, group=process_group
                )
            local_exp_sum = torch.exp(
                local_logits - global_max.unsqueeze(-1)
            ).sum(-1)
            global_exp_sum = local_exp_sum.clone()
            if tp_size > 1:
                dist.all_reduce(
                    global_exp_sum, op=dist.ReduceOp.SUM, group=process_group
                )
            log_z = global_max + torch.log(global_exp_sum.clamp_min_(1e-30))

        min_valid = torch.tensor(
            local_valid, dtype=torch.int64, device=local_logits.device
        )
        if tp_size > 1:
            dist.all_reduce(min_valid, op=dist.ReduceOp.MIN, group=process_group)
        local_k = min(int(topk), int(min_valid.item()))
        local_top_logits, local_indices = torch.topk(
            local_logits, k=local_k, dim=-1, sorted=True
        )
        local_global_ids = local_indices.to(torch.long) + int(
            geometry["org_start"]
        )
        if tp_size == 1:
            selected_scores = local_top_logits
            if log_z is not None:
                selected_scores = selected_scores - log_z.unsqueeze(-1)
            return (
                selected_scores,
                local_global_ids,
                raw_query,
            )

        gathered_logits = tp_group.all_gather(local_top_logits, dim=-1)
        gathered_ids = tp_group.all_gather(local_global_ids, dim=-1)
        global_k = min(int(topk), int(gathered_logits.shape[-1]))
        top_logits, top_indices = torch.topk(
            gathered_logits, k=global_k, dim=-1, sorted=True
        )
        top_ids = torch.gather(gathered_ids, 1, top_indices)
        if log_z is not None:
            top_logits = top_logits - log_z.unsqueeze(-1)
        return top_logits, top_ids, raw_query

    def _topk_log_probs_from_token_embedding_dot(
        self,
        *,
        hidden_states: torch.Tensor,
        topk: int,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ELF endpoints with the trained tied-input-embedding head.

        The dot+CE checkpoint family is trained with the frozen target input
        embedding table as its (tied) vocabulary head.  Its exact score is

          denormalize(x_pred) @ E_v.T / temperature,

        over the logical vocabulary only.  In particular, this deliberately
        has no L2 center, row-norm bias, or explicit-distance rerank.
        """

        if hidden_states.ndim != 2:
            raise ValueError(
                "Token-embedding ELF states must be [tokens, hidden], got "
                f"{tuple(hidden_states.shape)}."
            )
        if int(topk) <= 0:
            raise ValueError(f"Token-embedding topk must be positive, got {topk}.")
        denormalize = getattr(self.draft_model, "denormalize_latent", None)
        if not callable(denormalize):
            raise RuntimeError("Token-embedding ELF model cannot denormalize endpoint.")
        raw_query = denormalize(hidden_states)
        embed_module = self.target_worker.model_runner.model.get_input_embeddings()
        if not hasattr(embed_module, "weight") or not hasattr(
            embed_module, "shard_indices"
        ):
            raise RuntimeError(
                "Token-embedding ELF requires vocab-parallel target input embeddings."
            )
        weight = embed_module.weight
        shard = embed_module.shard_indices
        valid_vocab_size = int(self.draft_model.valid_vocab_size)
        table_vocab_size = int(
            getattr(embed_module, "org_vocab_size", valid_vocab_size)
        )
        org_start = int(shard.org_vocab_start_index)
        org_end = int(shard.org_vocab_end_index)
        num_org = int(shard.num_org_elements)
        if not 1 <= valid_vocab_size <= table_vocab_size:
            raise RuntimeError(
                "Invalid token-embedding vocabulary geometry: "
                f"valid={valid_vocab_size}, table={table_vocab_size}."
            )
        local_valid = max(0, min(org_end, valid_vocab_size) - org_start)
        if local_valid <= 0 or local_valid > num_org:
            raise RuntimeError(
                "Token-embedding dot projection has an invalid local vocab shard: "
                f"valid={local_valid}, physical={num_org}."
            )
        local_weight = weight[:local_valid].detach()
        query = raw_query.to(dtype=local_weight.dtype).contiguous()
        local_logits = self._token_embedding_fp32_mm(query, local_weight.T)
        local_logits.div_(float(self.draft_model.retrieval_temperature))

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        process_group = tp_group.device_group
        log_z = None
        if normalize:
            local_max = local_logits.amax(dim=-1)
            global_max = local_max.clone()
            if tp_size > 1:
                dist.all_reduce(
                    global_max, op=dist.ReduceOp.MAX, group=process_group
                )
            local_exp_sum = torch.exp(
                local_logits - global_max.unsqueeze(-1)
            ).sum(-1)
            global_exp_sum = local_exp_sum.clone()
            if tp_size > 1:
                dist.all_reduce(
                    global_exp_sum, op=dist.ReduceOp.SUM, group=process_group
                )
            log_z = global_max + torch.log(global_exp_sum.clamp_min_(1e-30))

        min_valid = torch.tensor(
            local_valid, dtype=torch.int64, device=local_logits.device
        )
        if tp_size > 1:
            dist.all_reduce(min_valid, op=dist.ReduceOp.MIN, group=process_group)
        local_k = min(int(topk), int(min_valid.item()))
        local_top_logits, local_indices = torch.topk(
            local_logits, k=local_k, dim=-1, sorted=True
        )
        local_global_ids = local_indices.to(torch.long) + org_start
        if tp_size == 1:
            selected_scores = local_top_logits
            if log_z is not None:
                selected_scores = selected_scores - log_z.unsqueeze(-1)
            return selected_scores, local_global_ids, raw_query

        gathered_logits = tp_group.all_gather(local_top_logits, dim=-1)
        gathered_ids = tp_group.all_gather(local_global_ids, dim=-1)
        global_k = min(int(topk), int(gathered_logits.shape[-1]))
        top_logits, top_indices = torch.topk(
            gathered_logits, k=global_k, dim=-1, sorted=True
        )
        top_ids = torch.gather(gathered_ids, 1, top_indices)
        if log_z is not None:
            top_logits = top_logits - log_z.unsqueeze(-1)
        return top_logits, top_ids, raw_query

    def _topk_log_probs_from_token_embedding(
        self,
        *,
        hidden_states: torch.Tensor,
        topk: int,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to the exact retrieval rule recorded by the checkpoint."""

        metric = str(getattr(self.draft_model, "retrieval_metric", "")).lower()
        if metric == "l2":
            return self._topk_log_probs_from_token_embedding_l2(
                hidden_states=hidden_states, topk=topk, normalize=normalize
            )
        if metric == "dot":
            return self._topk_log_probs_from_token_embedding_dot(
                hidden_states=hidden_states, topk=topk, normalize=normalize
            )
        raise RuntimeError(f"Unsupported token-embedding retrieval metric: {metric!r}")

    def _greedy_sample_from_token_embedding_l2(
        self, *, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Training-matched L2 top-1 with a global top-32 FP32 rerank."""

        _log_probs, candidate_ids, raw_query = (
            self._topk_log_probs_from_token_embedding_l2(
                hidden_states=hidden_states,
                topk=32,
                normalize=False,
            )
        )
        embed_module = self.target_worker.model_runner.model.get_input_embeddings()
        geometry = self._get_token_embedding_l2_geometry(embed_module)
        org_start = int(geometry["org_start"])
        local_valid = int(geometry["local_valid"])
        local_end = org_start + local_valid
        max_id = torch.iinfo(torch.long).max
        tp_group = get_tp_group()
        outputs = []
        for start in range(0, int(raw_query.shape[0]), 128):
            end = min(start + 128, int(raw_query.shape[0]))
            ids = candidate_ids[start:end]
            owned = (ids >= org_start) & (ids < local_end)
            local_indices = (ids - org_start).clamp(
                min=0, max=max(local_valid - 1, 0)
            )
            candidate_weight = (
                embed_module.weight[local_indices].detach().float()
            )
            squared_distance = (
                raw_query[start:end].float().unsqueeze(1) - candidate_weight
            ).square().sum(dim=-1)
            squared_distance.masked_fill_(~owned, float("inf"))
            local_best_distance = squared_distance.amin(dim=-1)
            local_ties = squared_distance == local_best_distance.unsqueeze(-1)
            local_best_id = ids.masked_fill(~local_ties, max_id).amin(dim=-1)
            if int(tp_group.world_size) == 1:
                outputs.append(local_best_id)
                continue
            gathered_distances = tp_group.all_gather(
                local_best_distance.unsqueeze(-1), dim=-1
            )
            gathered_ids = tp_group.all_gather(
                local_best_id.unsqueeze(-1), dim=-1
            )
            best_distance = gathered_distances.amin(dim=-1)
            global_ties = gathered_distances == best_distance.unsqueeze(-1)
            outputs.append(
                gathered_ids.masked_fill(~global_ties, max_id).amin(dim=-1)
            )
        return torch.cat(outputs, dim=0)

    def _greedy_sample_from_token_embedding(
        self, *, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Greedy proposal under the checkpoint-selected embedding decoder."""

        metric = str(getattr(self.draft_model, "retrieval_metric", "")).lower()
        if metric == "l2":
            return self._greedy_sample_from_token_embedding_l2(
                hidden_states=hidden_states
            )
        if metric == "dot":
            _scores, token_ids, _raw_query = (
                self._topk_log_probs_from_token_embedding_dot(
                    hidden_states=hidden_states,
                    topk=1,
                    normalize=False,
                )
            )
            return token_ids.squeeze(-1)
        raise RuntimeError(f"Unsupported token-embedding retrieval metric: {metric!r}")

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DFLASH greedy sampling requires a vocab-parallel head with `weight` and `shard_indices`."
            )

        shard = lm_head.shard_indices
        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        num_tokens = int(hidden_states.shape[0])
        out_token_ids = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        if tp_size == 1 and num_added == 0:
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    base_logits = self._apply_head_lora_to_local_logits(
                        base_logits=base_logits,
                        hidden_states=hs,
                        lm_head=lm_head,
                        output_slice=slice(0, num_org),
                    )
                    out_token_ids[start:end] = (
                        torch.argmax(base_logits, dim=-1).to(torch.long)
                        + org_vocab_start
                    )
                else:
                    out_token_ids[start:end] = 0
            return out_token_ids

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            # Base vocab logits.
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                base_logits = self._apply_head_lora_to_local_logits(
                    base_logits=base_logits,
                    hidden_states=hs,
                    lm_head=lm_head,
                    output_slice=slice(0, num_org),
                )
                local_max, local_arg = torch.max(base_logits, dim=-1)
            else:
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_logits = self._apply_head_lora_to_local_logits(
                    base_logits=added_logits,
                    hidden_states=hs,
                    lm_head=lm_head,
                    output_slice=slice(added_slice_start, added_slice_end),
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:
                out_token_ids[start:end] = global_ids.to(torch.long)
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            needed = tp_size * chunk_len
            chunk_cap = int(chunk_size)
            if (
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                cap = tp_size * chunk_cap
                self._draft_greedy_gathered_max_buf = torch.empty(
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]

            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_token_ids[start:end].copy_(selected_ids.view(-1))

        return out_token_ids

    def _topk_log_probs_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global top-k normalized log probabilities and token IDs.

        This avoids materializing the full global vocabulary on every TP rank.
        Each rank computes a masked local log-sum-exp and local top-k; NCCL then
        reduces the partition function and gathers only the candidates.
        """

        if self._is_dflow_token_embedding_model:
            top_log_probs, top_ids, _ = (
                self._topk_log_probs_from_token_embedding(
                    hidden_states=hidden_states,
                    topk=topk,
                )
            )
            return top_log_probs, top_ids

        if hidden_states.ndim != 2:
            raise ValueError(
                f"Expected [tokens, hidden] draft states, got {tuple(hidden_states.shape)}."
            )
        if int(topk) <= 0:
            raise ValueError(f"DDTree topk must be positive, got {topk}.")
        if hidden_states.numel() == 0:
            return (
                torch.empty(
                    (0, int(topk)), dtype=torch.float32, device=hidden_states.device
                ),
                torch.empty(
                    (0, int(topk)), dtype=torch.long, device=hidden_states.device
                ),
            )
        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "DDTree top-k extraction requires a vocab-parallel head with "
                "`weight` and `shard_indices`."
            )

        weight = lm_head.weight
        shard = lm_head.shard_indices
        hs = (
            hidden_states
            if hidden_states.dtype == weight.dtype
            else hidden_states.to(weight.dtype)
        )
        local_logits = torch.matmul(hs, weight.T)
        local_logits = self._apply_head_lora_to_local_logits(
            base_logits=local_logits,
            hidden_states=hs,
            lm_head=lm_head,
        )
        local_logits = local_logits.float()

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        valid = torch.zeros(
            (int(local_logits.shape[-1]),), dtype=torch.bool, device=local_logits.device
        )
        if num_org > 0:
            valid[:num_org] = True
        if num_added > 0:
            valid[num_org_padded : num_org_padded + num_added] = True
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            raise RuntimeError("DDTree found an empty local vocabulary shard.")
        local_logits.masked_fill_(~valid.unsqueeze(0), float("-inf"))

        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        process_group = tp_group.device_group

        # Stable global log-normalizer.
        local_max = local_logits.amax(dim=-1)
        global_max = local_max.clone()
        if tp_size > 1:
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=process_group)
        local_exp_sum = torch.exp(local_logits - global_max.unsqueeze(-1)).sum(dim=-1)
        global_exp_sum = local_exp_sum.clone()
        if tp_size > 1:
            dist.all_reduce(global_exp_sum, op=dist.ReduceOp.SUM, group=process_group)
        log_z = global_max + torch.log(global_exp_sum.clamp_min_(1e-30))

        # all_gather_into_tensor requires the same candidate width on every TP
        # rank. Vocab shards are normally balanced, but use the global minimum
        # valid count to remain correct for uneven or added-vocab shards.
        min_valid_count = torch.tensor(
            valid_count, dtype=torch.int64, device=local_logits.device
        )
        if tp_size > 1:
            dist.all_reduce(min_valid_count, op=dist.ReduceOp.MIN, group=process_group)
        local_k = min(int(topk), int(min_valid_count.item()))
        if local_k <= 0:
            raise RuntimeError(
                "DDTree could not select any TP-local vocabulary candidates."
            )
        local_top_logits, local_indices = torch.topk(
            local_logits, k=local_k, dim=-1, sorted=True
        )
        local_global_ids = torch.empty_like(local_indices, dtype=torch.long)
        is_base = local_indices < num_org
        local_global_ids[is_base] = org_vocab_start + local_indices[is_base]
        if num_added > 0:
            is_added = ~is_base
            local_global_ids[is_added] = added_vocab_start + (
                local_indices[is_added] - num_org_padded
            )

        if tp_size == 1:
            global_k = min(int(topk), local_k)
            return (
                local_top_logits[:, :global_k] - log_z.unsqueeze(-1),
                local_global_ids[:, :global_k],
            )

        num_tokens = int(hidden_states.shape[0])
        flat_count = num_tokens * local_k
        gathered_logits_flat = torch.empty(
            (tp_size * flat_count,),
            dtype=local_top_logits.dtype,
            device=local_top_logits.device,
        )
        gathered_ids_flat = torch.empty(
            (tp_size * flat_count,),
            dtype=local_global_ids.dtype,
            device=local_global_ids.device,
        )
        tp_group.all_gather_into_tensor(
            gathered_logits_flat, local_top_logits.contiguous().view(-1)
        )
        tp_group.all_gather_into_tensor(
            gathered_ids_flat, local_global_ids.contiguous().view(-1)
        )
        gathered_logits = gathered_logits_flat.view(tp_size, num_tokens, local_k)
        gathered_ids = gathered_ids_flat.view(tp_size, num_tokens, local_k)
        candidate_logits = gathered_logits.permute(1, 0, 2).reshape(
            num_tokens, tp_size * local_k
        )
        candidate_ids = gathered_ids.permute(1, 0, 2).reshape(
            num_tokens, tp_size * local_k
        )
        global_k = min(int(topk), int(candidate_logits.shape[-1]))
        top_logits, top_indices = torch.topk(
            candidate_logits, k=global_k, dim=-1, sorted=True
        )
        top_ids = torch.gather(candidate_ids, 1, top_indices)
        return top_logits - log_z.unsqueeze(-1), top_ids

    def _sample_dflash_future_proposals(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        str,
        Optional[int],
        Optional[torch.Tensor],
    ]:
        """Select proposal actions and return exact q_old plus its support.

        For the training-only categorical kernel, ``behavior_support_ids`` is
        the exact old-policy top-k set used to normalize each sampled action.
        PPO replay must reuse this set instead of recomputing a new top-k set:
        the latter can drop a behavior action after an optimizer step and
        makes the old/new likelihood ratio incomparable.
        """

        if self._training_proposal_topk is None:
            if self._is_dflow_token_embedding_model:
                return (
                    self._greedy_sample_from_token_embedding(
                        hidden_states=hidden_states
                    ),
                    None,
                    "deterministic_token_embedding_"
                    f"{self.draft_model.retrieval_metric}",
                    None,
                    None,
                )
            return (
                self._greedy_sample_from_vocab_parallel_head(
                    hidden_states=hidden_states,
                    lm_head=lm_head,
                ),
                None,
                "deterministic_greedy",
                None,
                None,
            )

        top_log_probs, top_ids = self._topk_log_probs_from_vocab_parallel_head(
            hidden_states=hidden_states,
            lm_head=lm_head,
            topk=self._training_proposal_topk,
        )
        behavior_log_probs = _renormalize_topk_behavior_log_probs(
            top_log_probs,
            temperature=self._training_proposal_temperature,
        )
        tp_group = get_tp_group()
        # Every TP rank must advance its sampling RNG by the same amount.
        # Strict stochastic verification draws random values independently on
        # every rank later in the same decode step.  Sampling only on rank zero
        # makes those RNG streams diverge; once a request finishes and the
        # batch shrinks, ranks can then disagree on acceptance and enter
        # different collective shapes/order.  Draw the same-shaped local
        # sample everywhere, then keep rank zero's action authoritative.
        sampled_indices = torch.multinomial(
            behavior_log_probs.exp(), num_samples=1
        )
        tp_group.broadcast(sampled_indices, src=0)
        sampled_ids = torch.gather(top_ids, 1, sampled_indices).squeeze(1)
        sampled_old_logprobs = torch.gather(
            behavior_log_probs, 1, sampled_indices
        ).squeeze(1)
        # Keep the contract robust to any future rank-local top-k tie handling.
        tp_group.broadcast(sampled_ids, src=0)
        tp_group.broadcast(sampled_old_logprobs, src=0)
        behavior_support_ids = top_ids.contiguous()
        tp_group.broadcast(behavior_support_ids, src=0)
        return (
            sampled_ids,
            sampled_old_logprobs,
            "categorical_topk",
            int(top_ids.shape[-1]),
            behavior_support_ids,
        )

    def _append_target_hidden_to_draft_kv(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInput,
    ) -> None:
        """Materialize the target hidden-state features into the draft KV cache.

        This must be run before exposing new tokens to radix cache (prefix hits), otherwise
        another request could reuse target KV indices without having draft KV values.
        """

        bs = batch.batch_size()
        device = self.model_runner.device

        if draft_input.target_hidden is None:
            raise RuntimeError(
                "DFLASH draft state missing target_hidden context features."
            )
        if draft_input.ctx_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH ctx_lens length mismatch: got {draft_input.ctx_lens.numel()} for bs={bs}."
            )
        if draft_input.draft_seq_lens.numel() != bs:
            raise RuntimeError(
                f"DFLASH draft_seq_lens length mismatch: got {draft_input.draft_seq_lens.numel()} for bs={bs}."
            )

        total_ctx = int(draft_input.target_hidden.shape[0])
        if total_ctx <= 0:
            if self._use_flow_ensemble_req_pool:
                draft_seq_lens = batch.seq_lens.to(
                    device=self.device, dtype=torch.int32
                )
                self._sync_flow_ensemble_base_prefix(
                    batch=batch,
                    draft_seq_lens=draft_seq_lens,
                )
                draft_input.draft_seq_lens = draft_seq_lens
            draft_input.ctx_lens = torch.zeros_like(draft_input.ctx_lens)
            draft_input.target_hidden = draft_input.target_hidden[:0]
            return

        target_req_to_token = batch.req_to_token_pool.req_to_token
        draft_req_to_token = self.draft_model_runner.req_to_token_pool.req_to_token

        req_pool_indices = batch.req_pool_indices
        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)

        ctx_lens = draft_input.ctx_lens
        if ctx_lens.dtype != torch.int32:
            ctx_lens = ctx_lens.to(torch.int32)
        if ctx_lens.device != device:
            ctx_lens = ctx_lens.to(device, non_blocking=True)
        ctx_start = batch.seq_lens.to(torch.int64) - ctx_lens.to(torch.int64)

        if bs == 1:
            # Fast path for single request.
            max_ctx = int(total_ctx)
            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            pos2d = ctx_start[:, None] + r[None, :]  # [1, ctx]
            cache2d = target_req_to_token[req_pool_indices[:, None], pos2d]  # [1, ctx]
            ctx_cache_loc = cache2d.reshape(-1).to(torch.int64)  # [ctx]
            ctx_positions = pos2d.reshape(-1)  # [ctx]
        else:
            # Decode ctx_lens is bounded by the fixed verify width, so avoid a
            # per-step max().item() synchronization.
            if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
                max_ctx = int(ctx_lens.max().item())
            else:
                max_ctx = int(
                    self.active_verify_width
                    if self.active_judge_enabled
                    else self.block_size
                )
            if max_ctx <= 0:
                raise RuntimeError(f"DFLASH invalid max_ctx={max_ctx} for KV append.")

            if max_ctx <= self._block_pos_offsets.numel():
                r = self._block_pos_offsets[:max_ctx]
            else:
                r = torch.arange(max_ctx, device=device, dtype=torch.int64)
            r = r[None, :]  # [1, max_ctx]
            pos2d = ctx_start[:, None] + r  # [bs, max_ctx]
            mask = r < ctx_lens[:, None]

            # Batched gather of cache locations and positions.
            ctx_cache_loc = self._gather_req_to_token_masked(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                pos2d=pos2d,
                mask=mask,
                context="DFLASH target hidden KV append",
            )  # [sum(ctx_lens)]
            ctx_positions = pos2d[mask]  # [sum(ctx_lens)]

        with torch.inference_mode():
            iter_kv_groups = getattr(self.draft_model, "iter_dflash_kv_groups", None)
            if callable(iter_kv_groups) and bool(
                getattr(self.draft_model, "has_multiple_dflash_kv_groups", False)
            ):
                # Sparse full-tower experts have different fc/hidden_norm and
                # K/V projections. Materialize every independent KV stack.
                for layers, ctx_hidden in iter_kv_groups(draft_input.target_hidden):
                    if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                        raise RuntimeError(
                            "Sparse DFLASH ctx_hidden/cache_loc mismatch: "
                            f"{ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                        )
                    self._append_target_hidden_sequential(
                        ctx_hidden,
                        ctx_positions,
                        ctx_cache_loc,
                        layers=layers,
                    )
            else:
                ctx_hidden = self.draft_model.project_target_hidden(
                    draft_input.target_hidden
                )  # [sum(ctx), hidden]
                if ctx_hidden.shape[0] != ctx_cache_loc.numel():
                    raise RuntimeError(
                        f"DFLASH ctx_hidden/cache_loc mismatch: {ctx_hidden.shape[0]} vs {ctx_cache_loc.numel()}."
                    )

                if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                    try:
                        self._append_target_hidden_fused(
                            ctx_hidden, ctx_positions, ctx_cache_loc
                        )
                    except Exception as e:
                        logger.warning(
                            "DFLASH fused KV append failed; falling back to sequential path: %s",
                            e,
                        )
                        self._use_fused_kv_materialize = False
                        self._fused_kv_helper = None
                        self._append_target_hidden_sequential(
                            ctx_hidden, ctx_positions, ctx_cache_loc
                        )
                else:
                    self._append_target_hidden_sequential(
                        ctx_hidden, ctx_positions, ctx_cache_loc
                    )

        if self.use_compact_draft_cache:
            new_draft_seq_lens = self._compute_compact_draft_seq_lens(batch.seq_lens)
            suffix_start = batch.seq_lens.to(torch.int64) - new_draft_seq_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=target_req_to_token,
                req_pool_indices=req_pool_indices,
                start=suffix_start,
                lengths=new_draft_seq_lens,
            )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                draft_req_to_token,
                torch.zeros_like(new_draft_seq_lens),
                new_draft_seq_lens,
                suffix_cache_loc,
                bs,
            )
            draft_input.draft_seq_lens = new_draft_seq_lens
        elif self._use_flow_ensemble_req_pool:
            new_draft_seq_lens = batch.seq_lens.to(
                device=self.device, dtype=torch.int32
            )
            self._sync_flow_ensemble_base_prefix(
                batch=batch,
                draft_seq_lens=new_draft_seq_lens,
            )
            draft_input.draft_seq_lens = new_draft_seq_lens
        else:
            draft_input.draft_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        draft_input.ctx_lens = torch.zeros_like(ctx_lens)
        draft_input.target_hidden = draft_input.target_hidden[:0]

    @staticmethod
    def _last_target_hidden_per_request(
        target_hidden: torch.Tensor,
        ctx_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Gather each ragged request segment's final raw target feature row."""
        if target_hidden is None or target_hidden.ndim != 2:
            raise RuntimeError(
                "DFLASH router context requires a 2D target_hidden tensor."
            )
        lengths = ctx_lens.to(device=target_hidden.device, dtype=torch.int64)
        if lengths.ndim != 1 or int(lengths.numel()) == 0:
            raise RuntimeError("DFLASH router context requires non-empty 1D ctx_lens.")
        if bool((lengths <= 0).any().item()):
            raise RuntimeError(
                "DFLASH router context requires every request to contribute at "
                f"least one target feature row, got ctx_lens={lengths.tolist()}."
            )
        end_indices = lengths.cumsum(0) - 1
        expected_rows = int(lengths.sum().item())
        if expected_rows != int(target_hidden.shape[0]):
            raise RuntimeError(
                "DFLASH target_hidden/ctx_lens mismatch while building router "
                f"context: rows={target_hidden.shape[0]}, sum(ctx_lens)={expected_rows}."
            )
        return target_hidden.index_select(0, end_indices).contiguous()

    def _maybe_last_target_hidden_per_request(
        self,
        target_hidden: torch.Tensor | None,
        ctx_lens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Build router/context state only for checkpoints that consume it.

        Native dense DFlash never reads this field. Skipping the gather there
        also avoids two GPU-to-host synchronizations in every verify cycle.
        """

        if not self._requires_router_target_hidden:
            return None
        return self._last_target_hidden_per_request(target_hidden, ctx_lens)

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
        *,
        layers=None,
    ) -> None:
        if layers is None:
            layers = self.draft_model.layers
        for layer in layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_norm(k)
            k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        layers = self.draft_model.layers

        def _write_layer_kv(
            layer_idx: int, cache_k: torch.Tensor, cache_v: torch.Tensor
        ) -> None:
            attn = layers[layer_idx].self_attn.attn
            token_to_kv_pool.set_kv_buffer(
                attn,
                ctx_cache_loc,
                cache_k,
                cache_v,
                attn.k_scale,
                attn.v_scale,
            )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
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
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
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

    def forward_batch_generation(
        self,
        batch: Union[ScheduleBatch, ModelWorkerBatch],
        **kwargs,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise RuntimeError(
                "Invariant broken: DFLASH batch requested return_logprob, but scheduler should have rejected this request."
            )

        if isinstance(batch, ModelWorkerBatch):
            # Should not happen for spec-v1 (non-overlap) scheduling, but keep a sane fallback.
            return self.target_worker.forward_batch_generation(batch, **kwargs)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL

            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch, **kwargs
            )
            logits_output, next_token_ids = (
                batch_result.logits_output,
                batch_result.next_token_ids,
            )
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_seq_lens is None
                or model_worker_batch.extend_prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_seq_lens / extend_prefix_lens to be populated in extend mode, but got None."
                )

            joint_prefill_cycles_cpu = None
            if self._collaborative_gate_mode is not None:
                sampling_info = model_worker_batch.sampling_info
                verifier_logprobs, verifier_behavior = (
                    gather_prefill_verifier_behavior_logprobs(
                        sampler_output_scores=logits_output.next_token_logits,
                        sampled_token_ids=next_token_ids,
                        sampling_info=sampling_info,
                    )
                )
                token_ids_cpu = (
                    next_token_ids.detach()
                    .to(device="cpu", dtype=torch.int64)
                    .reshape(-1)
                    .tolist()
                )
                if verifier_logprobs is None:
                    verifier_logprobs_cpu = [None] * len(token_ids_cpu)
                    verifier_temperatures_cpu = [None] * len(token_ids_cpu)
                    verifier_topks_cpu = [None] * len(token_ids_cpu)
                    verifier_top_ps_cpu = [None] * len(token_ids_cpu)
                    verifier_min_ps_cpu = [None] * len(token_ids_cpu)
                else:
                    verifier_logprobs_cpu = (
                        verifier_logprobs.detach()
                        .to(device="cpu", dtype=torch.float32)
                        .reshape(-1)
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
                joint_prefill_cycles_cpu = [
                    build_collaborative_verifier_prefill_cycle(
                        verifier_token_id=token_id,
                        verifier_behavior_logprob=verifier_logprobs_cpu[i],
                        verifier_behavior=verifier_behavior,
                        verifier_behavior_temperature=verifier_temperatures_cpu[i],
                        verifier_behavior_topk=verifier_topks_cpu[i],
                        verifier_behavior_top_p=verifier_top_ps_cpu[i],
                        verifier_behavior_min_p=verifier_min_ps_cpu[i],
                        policy_version=getattr(
                            self.server_args, "weight_version", None
                        ),
                    )
                    for i, token_id in enumerate(token_ids_cpu)
                ]

            # Materialize the prompt tokens into the draft KV cache immediately. This is required
            # for radix cache support, since the scheduler may update radix after prefill returns.
            device = next_token_ids.device

            def _to_int32_device_tensor(x, *, device=device):
                if isinstance(x, torch.Tensor):
                    if x.device != device:
                        x = x.to(device, non_blocking=True)
                    return x if x.dtype == torch.int32 else x.to(torch.int32)
                return torch.tensor(x, dtype=torch.int32, device=device)

            extend_seq_lens = _to_int32_device_tensor(
                model_worker_batch.extend_seq_lens
            )
            draft_input = DFlashDraftInput(
                verified_id=next_token_ids.to(torch.int64),
                target_hidden=logits_output.hidden_states,
                ctx_lens=extend_seq_lens,
                draft_seq_lens=(
                    torch.zeros_like(extend_seq_lens)
                    if self.use_compact_draft_cache
                    else _to_int32_device_tensor(model_worker_batch.extend_prefix_lens)
                ),
                router_target_hidden=self._maybe_last_target_hidden_per_request(
                    logits_output.hidden_states, extend_seq_lens
                ),
            )
            self._append_target_hidden_to_draft_kv(batch, draft_input)
            batch.spec_info = draft_input

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                joint_prefill_cycles_cpu=joint_prefill_cycles_cpu,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        # Decode / target-verify stage.
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInput):
            raise RuntimeError(
                "DFLASH decode requires DFlashDraftInput state on the running batch. "
                "This usually means the request did not complete the prefill stage."
            )

        self._timing.start_step(batch_size=batch.batch_size())
        self._prepare_for_speculative_decoding(batch, draft_input)

        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.forward_mode.is_target_verify()
        verify_input = model_worker_batch.spec_info
        assert isinstance(verify_input, DFlashVerifyInput)
        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            batch.seq_lens.clone() if need_mamba_verify_commit else None
        )

        self._timing.mark("verifier_start")
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True, **kwargs
        )
        self._timing.mark("verifier_end")
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        self._timing.mark("accept_start")
        (
            new_verified_id,
            commit_lens,
            next_target_hidden,
            accept_length_per_req_cpu,
            commit_lens_cpu,
        ) = verify_input.verify(
            batch=batch,
            logits_output=logits_output,
            page_size=self.page_size,
        )
        self._timing.mark("accept_end")
        self._timing.mark("mamba_commit_start")
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )
        self._timing.mark("mamba_commit_end")

        # Update draft state for the next iteration. Also materialize the committed verify tokens
        # into the draft KV cache immediately so radix cache entries are safe to reuse.
        draft_input.verified_id = new_verified_id
        draft_input.target_hidden = next_target_hidden
        draft_input.ctx_lens = commit_lens
        draft_input.router_target_hidden = self._maybe_last_target_hidden_per_request(
            next_target_hidden, commit_lens
        )
        self._timing.mark("draft_kv_post_start")
        self._append_target_hidden_to_draft_kv(batch, draft_input)
        self._timing.mark("draft_kv_post_end")
        batch.spec_info = draft_input
        batch.forward_mode = ForwardMode.DECODE

        num_accepted_tokens = sum(accept_length_per_req_cpu)
        if self.active_judge_enabled:
            judge_proposed_tokens = sum(self._last_active_judge_proposal_lens)
            judge_blocks = sum(self._last_active_judge_blocks_run)
            judge_first_fail_blocks = sum(
                self._last_active_judge_first_fail_blocks
            )
            judge_zero_proposal_requests = sum(
                length == 0 for length in self._last_active_judge_proposal_lens
            )
        else:
            judge_proposed_tokens = 0
            judge_blocks = 0
            judge_first_fail_blocks = 0
            judge_zero_proposal_requests = 0
        self._timing.finish_step(
            accepted_tokens=num_accepted_tokens,
            judge_proposed_tokens=judge_proposed_tokens,
            judge_blocks=judge_blocks,
            judge_first_fail_blocks=judge_first_fail_blocks,
            judge_zero_proposal_requests=judge_zero_proposal_requests,
            verify_width=int(verify_input.draft_token_num),
            can_run_cuda_graph=bool(can_run_cuda_graph),
        )
        if not self._logged_first_verify and self.tp_rank == 0:
            if self.active_judge_enabled:
                logger.info(
                    "Active DFLASH first verify completed. "
                    "judge_block_accept_lens=%s, judge_proposal_lens=%s, "
                    "judge_blocks_run=%s, target_verify_width=%s, "
                    "target_cuda_graph=%s, target_accept_lens=%s",
                    self._last_active_judge_block_accept_lens,
                    self._last_active_judge_proposal_lens,
                    self._last_active_judge_blocks_run,
                    int(verify_input.draft_token_num),
                    bool(can_run_cuda_graph),
                    accept_length_per_req_cpu,
                )
            else:
                logger.info(
                    "DFLASH verify completed. accept_length_per_req=%s",
                    accept_length_per_req_cpu,
                )
            self._logged_first_verify = True

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=new_verified_id,
            num_accepted_tokens=num_accepted_tokens,
            accept_length_per_req_cpu=accept_length_per_req_cpu,
            committed_rows_per_req_cpu=commit_lens_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
        )
