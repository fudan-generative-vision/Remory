"""Native full DFlow/DFly runtime: layer fusion plus hidden correction.

The draft transformer remains block parallel and is evaluated exactly once.
Its decoded endpoint is corrected outside the backbone by
``HiddenStatesCorrection``.  The worker must then visit future slots in order:
slot 1 is conditioned on the verified root token and every later slot is
conditioned on the token actually sampled at the preceding slot.

The HSC tensors are deliberately ordinary, replicated ``nn.Parameter``
objects.  At TP=2 every rank owns the complete 32,773,120-parameter HSC (for
Qwen3-4B) and the complete 54-parameter fusion state.  Only the inherited
attention/MLP tensors use SGLang's tensor-parallel sharding.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflow import DFlowDraftModel
from sglang.srt.models.dflow_dfly_hybrid import DFlowDFlyHybridDraftModel
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config


_FULL_ARCHITECTURE = "DFlowDFlyFullDraftModel"
_FULL_AUTO_MAP = {"AutoModel": "dflow_dfly_full.DFlowDFlyFullDraftModel"}
_STAGE_J_TRAINING_CONFIG_SCHEMA = "specforge-dflow-stage-j-full-hsc-training-config-v1"
_STAGE_J_SERVING_CONFIG_SCHEMA = "specforge-dflow-stage-j-full-hsc-serving-config-v1"
_STAGE_J_TARGET_LAYER_IDS = (1, 6, 10, 15, 19, 24, 28, 36)
_STAGE_J_SERVING_CONTRACT = {
    "schema": _STAGE_J_SERVING_CONFIG_SCHEMA,
    "architecture": _FULL_ARCHITECTURE,
    "drafter_backbone_forwards_per_block": 1,
    "draft_block_size": 16,
    "generated_slots_per_block": 15,
    "layer_fusion_target_views": 8,
    "layer_fusion_draft_layers": 6,
    "hidden_correction_enabled": True,
    "hidden_correction_order": (
        "denormalize_then_decode_adapter_then_hsc_then_frozen_lm_head"
    ),
    "hidden_correction_inference_conditioning": (
        "slot1_verified_root_embedding_then_slot2_to_15_previous_sampled_token_embedding"
    ),
    "hidden_correction_teacher_forcing_is_training_only": True,
    "hidden_correction_steps_per_block": 15,
    "linear_m1_surviving_prefix_equivalence": (
        "runtime_previous_sampled_token_equals_ground_truth_before_first_mismatch"
    ),
    "post_first_mismatch_states": "untrained_and_irrelevant_to_linear_m1_acceptance",
    "step250_role": "stop_only_if_delta_al_le_-0.10_and_paired_ci95_high_lt_0",
    "step500_gate": "continue_joint_only_if_delta_al_ge_0.15_and_paired_ci95_low_gt_0",
    "minimum_failed_hsc_followup": "alpha0_exact_survival_not_scheduled_sampling",
    "dpace_alpha_zero_semantics": (
        "exact_survival_credit_stopgrad_reverse_cumsum_prefix_probability"
    ),
    "dpace_alpha_half_semantics": "smoothed_surrogate",
    "frozen_verifier_embedding_required": True,
    "contextual_x0_decode_space": "verifier_post_final_norm_hidden",
    "hidden_correction_output_projection": (
        "direct_frozen_lm_head_without_second_final_norm"
    ),
    "verifier_final_norm_runtime_reapplication_forbidden": True,
    "frozen_verifier_embedding_use": "hsc_previous_token_condition_only",
    "frozen_verifier_lm_head_required": True,
    "reject_base_or_fusion_only_runtime": True,
}
_STAGE_J_CONFIG_KEYS = frozenset(
    {
        "stage_j_training_config_schema",
        "stage_j_serving_config_schema",
        "stage_j_full_dfly_hsc",
        "stage_j_source_spec_sha256",
        "stage_j_source_spec_canonical_sha256",
        "stage_j_winner",
        "stage_j_source_arm",
        "stage_j_source_checkpoint",
        "stage_j_source_training_stage",
        "stage_j_source_training_objective",
        "stage_j_source_weights_only",
        "stage_j_objective_normalization",
        "stage_j_step0_endpoint_exact_source",
        "stage_j_step0_logits_exact_source",
        "stage_j_hsc_teacher_forced_previous_ids",
        "stage_j_hsc_supervised_ids",
        "stage_j_linear_m1_surviving_prefix_equivalence",
        "stage_j_post_first_mismatch_states",
        "stage_j_step250_role",
        "stage_j_step500_gate",
        "stage_j_minimum_failed_hsc_followup",
        "stage_j_dpace_alpha_zero_semantics",
        "stage_j_dpace_alpha_half_semantics",
        "stage_j_contextual_x0_decode_space",
        "stage_j_hidden_correction_output_projection",
        "stage_j_verifier_final_norm_runtime_reapplication_forbidden",
        "stage_j_optimizer_schedule",
        "stage_j_hsc_only_steps",
        "stage_j_hsc_phase1_peak_lr",
        "stage_j_hsc_phase1_warmup_steps",
        "stage_j_hsc_phase2_start_lr",
        "stage_j_hsc_phase2_min_lr",
        "stage_j_shared_phase1_lr",
        "stage_j_shared_phase1_gradients_discarded",
        "stage_j_shared_phase2_peak_lr",
        "stage_j_shared_phase2_warmup_steps",
        "stage_j_shared_phase2_min_lr",
        "stage_j_optimizer_reset",
        "stage_j_scheduler_reset",
        "stage_j_rng_reset",
        "stage_j_cursor_reset",
        "stage_j_seed",
        "stage_j_max_train_steps",
        "stage_j_checkpoint_steps",
        "stage_j_serving_contract",
    }
)
_HSC_WEIGHT_KEYS = frozenset(
    {
        "hidden_correction.hidden_norm.weight",
        "hidden_correction.embed_norm.weight",
        "hidden_correction.gate_proj.weight",
        "hidden_correction.up_proj.weight",
        "hidden_correction.down_proj.weight",
    }
)
_HSC_CONFIG_KEYS = frozenset(
    {
        "hidden_correction_hidden_size",
        "hidden_correction_embed_size",
        "hidden_correction_intermediate_size",
        "hidden_correction_initializer_std",
        "hidden_correction_initialization_seed",
        "hidden_correction_input_space",
        "hidden_correction_token_conditioning",
        "hidden_correction_warm_start",
        "hidden_correction_preserves_single_drafter_forward",
    }
)


def _normalize_weight_name(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def _value_equal(actual, expected) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, Integral)
            and not isinstance(actual, bool)
            and int(actual) == expected
        )
    if isinstance(expected, float):
        return (
            isinstance(actual, Real)
            and not isinstance(actual, bool)
            and math.isfinite(float(actual))
            and float(actual) == expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_value_equal(a, b) for a, b in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _validate_full_config(
    config,
    *,
    num_draft_layers: int,
    target_layer_ids: tuple[int, ...],
) -> tuple[int, int]:
    """Validate the complete full-HSC serving ABI and return ``(E, I)``."""

    architectures = getattr(config, "architectures", None)
    if architectures != [_FULL_ARCHITECTURE]:
        raise ValueError(
            "Full DFlow/DFly runtime requires architectures="
            f"[{_FULL_ARCHITECTURE!r}], got {architectures!r}."
        )
    if getattr(config, "auto_map", None) != _FULL_AUTO_MAP:
        raise ValueError(
            "Full DFlow/DFly runtime requires auto_map="
            f"{_FULL_AUTO_MAP!r}, got {getattr(config, 'auto_map', None)!r}."
        )
    raw = getattr(config, "dflow_config", None)
    if not isinstance(raw, Mapping):
        raise ValueError("Full DFlow/DFly checkpoint requires mapping dflow_config.")
    dflow = dict(raw)
    hidden_size = int(config.hidden_size)
    embed_size = int(dflow.get("hidden_correction_embed_size", -1))
    intermediate_size = int(dflow.get("hidden_correction_intermediate_size", -1))

    expected = {
        "training_stage": "one_forward_stage_j_full_dfly_hsc_local_dpace",
        "training_objective": (
            "full_dfly_hsc_teacher_forced_previous_token_frozen_head_local_dpace"
        ),
        "hybrid_target_conditioning": "dfly_residual",
        "stage_e_dfly_hybrid": True,
        "layer_fusion_warm_start": "zero_gate_exact_base",
        "layer_fusion_num_target_layers": len(target_layer_ids),
        "layer_fusion_target_layer_ids": list(target_layer_ids),
        "layer_fusion_preserves_target_feature_abi": True,
        "target_layer_ids": list(_STAGE_J_TARGET_LAYER_IDS),
        "full_dfly_adaptation": True,
        "enable_hidden_correction": True,
        "hidden_correction_hidden_size": hidden_size,
        "hidden_correction_embed_size": hidden_size,
        "hidden_correction_intermediate_size": 2560,
        "hidden_correction_initializer_std": 0.02,
        "hidden_correction_initialization_seed": 47011,
        "hidden_correction_input_space": "denormalize_then_decode_adapter",
        "hidden_correction_token_conditioning": (
            "teacher_forced_previous_token_embedding"
        ),
        "hidden_correction_warm_start": "zero_down_proj_exact_endpoint",
        "hidden_correction_preserves_single_drafter_forward": True,
        "serving_drafter_forwards": 1,
        "stage_j_training_config_schema": _STAGE_J_TRAINING_CONFIG_SCHEMA,
        "stage_j_serving_config_schema": _STAGE_J_SERVING_CONFIG_SCHEMA,
        "stage_j_full_dfly_hsc": True,
        "stage_j_source_weights_only": True,
        "stage_j_objective_normalization": "local_microbatch",
        "stage_j_step0_endpoint_exact_source": True,
        "stage_j_step0_logits_exact_source": True,
        "stage_j_hsc_teacher_forced_previous_ids": "target_ids[..., :-1]",
        "stage_j_hsc_supervised_ids": "target_ids[..., 1:]",
        "stage_j_linear_m1_surviving_prefix_equivalence": (
            "runtime_previous_sampled_token_equals_ground_truth_before_first_mismatch"
        ),
        "stage_j_post_first_mismatch_states": (
            "untrained_and_irrelevant_to_linear_m1_acceptance"
        ),
        "stage_j_step250_role": (
            "stop_only_if_delta_al_le_-0.10_and_paired_ci95_high_lt_0"
        ),
        "stage_j_step500_gate": (
            "continue_joint_only_if_delta_al_ge_0.15_and_paired_ci95_low_gt_0"
        ),
        "stage_j_minimum_failed_hsc_followup": (
            "alpha0_exact_survival_not_scheduled_sampling"
        ),
        "stage_j_dpace_alpha_zero_semantics": (
            "exact_survival_credit_stopgrad_reverse_cumsum_prefix_probability"
        ),
        "stage_j_dpace_alpha_half_semantics": "smoothed_surrogate",
        "stage_j_contextual_x0_decode_space": "verifier_post_final_norm_hidden",
        "stage_j_hidden_correction_output_projection": (
            "direct_frozen_lm_head_without_second_final_norm"
        ),
        "stage_j_verifier_final_norm_runtime_reapplication_forbidden": True,
        "stage_j_optimizer_schedule": "hsc_only_500_then_low_lr_joint_to_2000",
        "stage_j_hsc_only_steps": 500,
        "stage_j_hsc_phase1_peak_lr": 1e-4,
        "stage_j_hsc_phase1_warmup_steps": 25,
        "stage_j_hsc_phase2_start_lr": 5e-5,
        "stage_j_hsc_phase2_min_lr": 1e-5,
        "stage_j_shared_phase1_lr": 0.0,
        "stage_j_shared_phase1_gradients_discarded": True,
        "stage_j_shared_phase2_peak_lr": 1e-5,
        "stage_j_shared_phase2_warmup_steps": 50,
        "stage_j_shared_phase2_min_lr": 2e-6,
        "stage_j_optimizer_reset": True,
        "stage_j_scheduler_reset": True,
        "stage_j_rng_reset": True,
        "stage_j_cursor_reset": True,
        "stage_j_seed": 48,
        "stage_j_max_train_steps": 2000,
        "stage_j_checkpoint_steps": [250, 500, 1000, 1500, 2000],
        "stage_j_serving_contract": _STAGE_J_SERVING_CONTRACT,
        "loss_type": "dpace",
        "dpace_alpha": 0.5,
        "unique_anchors_per_sequence": 256,
        "generator_samples_per_condition": 1,
        "full_tail_gaussian": True,
        "denoiser_noise_scale": 2.0,
        "frozen_target_model": True,
        "frozen_target_embedding": True,
        "frozen_target_final_norm": True,
        "frozen_target_lm_head": True,
    }
    mismatches = {
        key: (dflow.get(key), value)
        for key, value in expected.items()
        if not _value_equal(dflow.get(key), value)
    }
    if not _value_equal(
        getattr(config, "num_hidden_layers", None), int(num_draft_layers)
    ):
        mismatches["num_hidden_layers"] = (
            getattr(config, "num_hidden_layers", None),
            int(num_draft_layers),
        )
    qwen3_4b_geometry = {
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 6,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "block_size": 16,
        "vocab_size": 151936,
    }
    for key, value in qwen3_4b_geometry.items():
        if not _value_equal(getattr(config, key, None), value):
            mismatches[key] = (getattr(config, key, None), value)
    if target_layer_ids != _STAGE_J_TARGET_LAYER_IDS:
        mismatches["resolved_target_layer_ids"] = (
            list(target_layer_ids),
            list(_STAGE_J_TARGET_LAYER_IDS),
        )
    source_arm = dflow.get("stage_j_source_arm")
    if source_arm not in {"base", "zero_gated_layer_fusion"}:
        mismatches["stage_j_source_arm"] = (
            source_arm,
            "base|zero_gated_layer_fusion",
        )
    winner = dflow.get("stage_j_winner")
    winner_source_arms = {
        "stage_d_incumbent": "base",
        "stage_i_base_250": "base",
        "stage_i_fusion_250": "zero_gated_layer_fusion",
        "stage_i_base_500": "base",
        "stage_i_fusion_500": "zero_gated_layer_fusion",
    }
    if winner not in winner_source_arms:
        mismatches["stage_j_winner"] = (
            winner,
            sorted(winner_source_arms),
        )
    elif winner_source_arms[winner] != source_arm:
        mismatches["stage_j_winner_source_arm"] = (
            (winner, source_arm),
            (winner, winner_source_arms[winner]),
        )
    for key in ("stage_j_source_spec_sha256", "stage_j_source_spec_canonical_sha256"):
        value = dflow.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            mismatches[key] = (value, "lowercase sha256")
    source_checkpoint = dflow.get("stage_j_source_checkpoint")
    if not isinstance(source_checkpoint, str) or not source_checkpoint.startswith("/"):
        mismatches["stage_j_source_checkpoint"] = (
            source_checkpoint,
            "absolute path recorded by trainer",
        )
    for key in ("stage_j_source_training_stage", "stage_j_source_training_objective"):
        value = dflow.get(key)
        if not isinstance(value, str) or not value:
            mismatches[key] = (value, "non-empty source provenance string")
    if embed_size != hidden_size:
        mismatches["hidden_correction_embed_size"] = (embed_size, hidden_size)
    if intermediate_size != 2560:
        mismatches["hidden_correction_intermediate_size"] = (
            intermediate_size,
            2560,
        )
    initializer_std = dflow.get("hidden_correction_initializer_std")
    if (
        isinstance(initializer_std, bool)
        or not isinstance(initializer_std, Real)
        or not math.isfinite(float(initializer_std))
        or float(initializer_std) <= 0.0
    ):
        mismatches["hidden_correction_initializer_std"] = (
            initializer_std,
            "positive finite real",
        )
    initialization_seed = dflow.get("hidden_correction_initialization_seed")
    if (
        isinstance(initialization_seed, bool)
        or not isinstance(initialization_seed, Integral)
        or int(initialization_seed) < 0
    ):
        mismatches["hidden_correction_initialization_seed"] = (
            initialization_seed,
            "non-negative integer",
        )
    unknown_hsc_keys = sorted(
        key
        for key in dflow
        if key.startswith("hidden_correction_") and key not in _HSC_CONFIG_KEYS
    )
    if unknown_hsc_keys:
        mismatches["unknown_hidden_correction_keys"] = (
            unknown_hsc_keys,
            "none",
        )
    unknown_stage_j_keys = sorted(
        key
        for key in dflow
        if key.startswith("stage_j_") and key not in _STAGE_J_CONFIG_KEYS
    )
    if unknown_stage_j_keys:
        mismatches["unknown_stage_j_keys"] = (unknown_stage_j_keys, "none")
    if mismatches:
        raise ValueError(
            "Full DFlow/DFly config contract mismatch (actual, expected): "
            f"{mismatches}."
        )
    return embed_size, intermediate_size


class _ExactRMSNorm(nn.Module):
    """Qwen3 training-side RMSNorm formula with an explicit FP32 reduction."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        value = hidden_states.float()
        value = value * torch.rsqrt(
            value.square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * value.to(dtype=input_dtype)


class HiddenStatesCorrection(nn.Module):
    """Residual two-norm SwiGLU matching SpecForge's training module."""

    def __init__(
        self,
        *,
        hidden_size: int,
        embed_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embed_size = int(embed_size)
        self.intermediate_size = int(intermediate_size)
        self.hidden_norm = _ExactRMSNorm(self.hidden_size, rms_norm_eps)
        self.embed_norm = _ExactRMSNorm(self.embed_size, rms_norm_eps)
        input_size = self.hidden_size + self.embed_size
        # These are intentionally full, replicated dense layers on every TP rank.
        self.gate_proj = nn.Linear(input_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(input_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(
                "Full DFlow/DFly HSC hidden_states must be [batch, hidden], got "
                f"{tuple(hidden_states.shape)}."
            )
        if previous_token_embeddings.ndim != 2:
            raise ValueError(
                "Full DFlow/DFly previous-token embeddings must be "
                f"[batch, embed], got {tuple(previous_token_embeddings.shape)}."
            )
        if hidden_states.shape[:-1] != previous_token_embeddings.shape[:-1]:
            raise ValueError(
                "Full DFlow/DFly HSC batch mismatch: hidden="
                f"{tuple(hidden_states.shape)}, embedding="
                f"{tuple(previous_token_embeddings.shape)}."
            )
        if int(hidden_states.shape[-1]) != self.hidden_size:
            raise ValueError(
                "Full DFlow/DFly HSC hidden width mismatch: expected "
                f"{self.hidden_size}, got {hidden_states.shape[-1]}."
            )
        if int(previous_token_embeddings.shape[-1]) != self.embed_size:
            raise ValueError(
                "Full DFlow/DFly HSC embedding width mismatch: expected "
                f"{self.embed_size}, got {previous_token_embeddings.shape[-1]}."
            )
        compute_dtype = self.gate_proj.weight.dtype
        residual_dtype = hidden_states.dtype
        h = self.hidden_norm(hidden_states.to(dtype=compute_dtype))
        e = self.embed_norm(
            previous_token_embeddings.to(
                device=hidden_states.device, dtype=compute_dtype
            )
        )
        value = torch.cat((h, e), dim=-1)
        correction = self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))
        return hidden_states + correction.to(dtype=residual_dtype)

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())


def _expected_full_checkpoint_shapes(
    config,
    *,
    num_draft_layers: int,
    num_target_features: int,
    hsc_embed_size: int,
    hsc_intermediate_size: int,
) -> dict[str, tuple[int, ...]]:
    """Return the exact SpecForge Qwen3 state-dict inventory and full shapes."""

    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    head_dim = int(
        getattr(config, "head_dim", hidden // int(config.num_attention_heads))
    )
    q_width = int(config.num_attention_heads) * head_dim
    kv_width = int(config.num_key_value_heads) * head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "context_proj.weight": (hidden, num_target_features * hidden),
        "fc.weight": (hidden, num_target_features * hidden),
        "hidden_norm.weight": (hidden,),
        "latent_mean": (),
        "latent_std": (),
        "mode_embed.weight": (4, hidden),
        "out_norm.weight": (hidden,),
        "self_cond_proj.bias": (hidden,),
        "self_cond_proj.weight": (hidden, 2 * hidden),
        "time_mlp.0.bias": (4 * hidden,),
        "time_mlp.0.weight": (4 * hidden, hidden),
        "time_mlp.2.bias": (hidden,),
        "time_mlp.2.weight": (hidden, 4 * hidden),
        "layer_fusion_weights": (num_draft_layers, num_target_features),
        "layer_fusion_gates": (num_draft_layers,),
        "hidden_correction.hidden_norm.weight": (hidden,),
        "hidden_correction.embed_norm.weight": (hsc_embed_size,),
        "hidden_correction.gate_proj.weight": (
            hsc_intermediate_size,
            hidden + hsc_embed_size,
        ),
        "hidden_correction.up_proj.weight": (
            hsc_intermediate_size,
            hidden + hsc_embed_size,
        ),
        "hidden_correction.down_proj.weight": (
            hidden,
            hsc_intermediate_size,
        ),
    }
    dflow = dict(getattr(config, "dflow_config", {}) or {})
    layerwise = bool(dflow.get("layerwise_noise_conditioning", False))
    shapes["layerwise_denoise_t"] = (num_draft_layers,) if layerwise else (0,)
    normalization_type = str(dflow.get("normalization_type", "scalar"))
    if normalization_type == "channel":
        shapes["latent_mean"] = (hidden,)
        shapes["latent_std"] = (hidden,)
    elif normalization_type != "scalar":
        raise ValueError(
            "Full DFlow/DFly supports only scalar/channel latent normalization, "
            f"got {normalization_type!r}."
        )
    decode_type = dflow.get("decode_adapter_type", "rmsnorm")
    if decode_type == "rmsnorm":
        shapes["decode_norm.weight"] = (hidden,)
    elif decode_type == "linear":
        shapes["decode_norm.weight"] = (hidden, hidden)
        shapes["decode_norm.bias"] = (hidden,)
    elif decode_type not in (None, "none", "identity", "no_adapter"):
        raise ValueError(
            f"Unsupported Full DFlow/DFly decode_adapter_type={decode_type!r}."
        )
    for layer_idx in range(num_draft_layers):
        prefix = f"layers.{layer_idx}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight": (q_width, hidden),
                f"{prefix}.self_attn.k_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.v_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, q_width),
                f"{prefix}.self_attn.q_norm.weight": (head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (head_dim,),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return shapes


def _partition_full_weights(
    weights: Iterable[Tuple[str, torch.Tensor]],
    *,
    expected_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[list[Tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Validate all names/shapes before modifying any model parameter."""

    ordinary: list[Tuple[str, torch.Tensor]] = []
    hsc: dict[str, torch.Tensor] = {}
    payloads: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    for raw_name, value in weights:
        name = _normalize_weight_name(raw_name)
        if name in seen:
            raise ValueError(
                f"Full DFlow/DFly checkpoint contains duplicate tensor {name!r}."
            )
        seen.add(name)
        if name not in expected_shapes:
            raise ValueError(
                "Full DFlow/DFly checkpoint contains an unknown tensor; "
                f"refusing a partial/silent load: {name!r}."
            )
        expected = tuple(int(dim) for dim in expected_shapes[name])
        actual = tuple(int(dim) for dim in value.shape)
        if actual != expected:
            raise ValueError(
                f"Full DFlow/DFly tensor {name!r} shape mismatch: expected "
                f"{expected}, got {actual}."
            )
        if name in _HSC_WEIGHT_KEYS:
            hsc[name] = value
        else:
            ordinary.append((raw_name, value))
        payloads[name] = value
    missing = set(expected_shapes) - seen
    if missing:
        raise ValueError(
            f"Full DFlow/DFly checkpoint is missing required tensors: {sorted(missing)}"
        )
    if set(hsc) != _HSC_WEIGHT_KEYS:
        raise ValueError(
            "Full DFlow/DFly checkpoint has incomplete HSC tensors: "
            f"missing={sorted(_HSC_WEIGHT_KEYS - set(hsc))}."
        )
    # SpecForge serializes the same context projection under a compatibility
    # alias.  Requiring byte-identical payloads closes the otherwise
    # order-dependent "last alias wins" load path.
    if not torch.equal(payloads["context_proj.weight"], payloads["fc.weight"]):
        raise ValueError(
            "Full DFlow/DFly context_proj.weight and fc.weight aliases differ; "
            "refusing order-dependent loading."
        )
    return ordinary, hsc


class DFlowDFlyFullDraftModel(DFlowDFlyHybridDraftModel):
    """One-forward DFlow/DFly backbone with replicated post-decode HSC."""

    is_dflow_dfly_full_model = True

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        # Do not call the hybrid constructor: it correctly rejects HSC configs.
        # Build the common DFlow base directly, then add the same two fusion
        # tensors under the independently validated full-HSC contract.
        DFlowDraftModel.__init__(
            self, config=config, quant_config=quant_config, prefix=prefix
        )
        self.has_multiple_dflash_kv_groups = True
        self.supports_fused_kv_materialization = False
        if getattr(self, "joint_judge_enabled", False):
            raise ValueError("Full DFlow/DFly does not support joint judge.")

        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        target_num_layers = (
            int(draft_config.num_target_layers)
            if draft_config.num_target_layers is not None
            else int(config.num_hidden_layers)
        )
        target_layer_ids = tuple(
            int(layer_id)
            for layer_id in draft_config.resolve_target_layer_ids(
                target_num_layers=target_num_layers,
                draft_num_layers=int(config.num_hidden_layers),
            )
        )
        if len(target_layer_ids) != int(self.num_context_features):
            raise ValueError(
                "Full DFlow/DFly resolved target-feature count mismatch: "
                f"ids={target_layer_ids}, num_context_features="
                f"{self.num_context_features}."
            )
        embed_size, intermediate_size = _validate_full_config(
            config,
            num_draft_layers=len(self.layers),
            target_layer_ids=target_layer_ids,
        )
        self.target_layer_ids = target_layer_ids
        self.layer_fusion_weights = nn.Parameter(
            torch.zeros(len(self.layers), len(target_layer_ids))
        )
        self.layer_fusion_gates = nn.Parameter(torch.zeros(len(self.layers)))
        self._layer_fusion_zero_gates = (True,) * len(self.layers)
        self.hidden_correction = HiddenStatesCorrection(
            hidden_size=int(config.hidden_size),
            embed_size=embed_size,
            intermediate_size=intermediate_size,
            rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
        )
        self.hidden_correction_embed_size = embed_size
        self.hidden_correction_intermediate_size = intermediate_size

    def apply_hidden_correction(
        self,
        decoded_hidden_states: torch.Tensor,
        previous_token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Apply HSC only; this method never invokes the draft backbone."""

        return self.hidden_correction(decoded_hidden_states, previous_token_embeddings)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Strictly load the complete full checkpoint before serving."""

        expected_shapes = _expected_full_checkpoint_shapes(
            self.config,
            num_draft_layers=len(self.layers),
            num_target_features=int(self.num_context_features),
            hsc_embed_size=self.hidden_correction_embed_size,
            hsc_intermediate_size=self.hidden_correction_intermediate_size,
        )
        ordinary, hsc = _partition_full_weights(
            list(weights), expected_shapes=expected_shapes
        )
        loaded = super().load_weights(ordinary)
        params = dict(self.named_parameters())
        for name in sorted(_HSC_WEIGHT_KEYS):
            parameter = params[name]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, hsc[name])
            loaded.add(name)
        return loaded


EntryClass = DFlowDFlyFullDraftModel


__all__ = [
    "DFlowDFlyFullDraftModel",
    "EntryClass",
    "HiddenStatesCorrection",
]
