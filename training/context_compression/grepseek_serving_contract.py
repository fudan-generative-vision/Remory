"""Shared serving contracts for legacy and recursive GrepSeek checkpoints.

This module is deliberately model- and SGLang-free.  The GPU hook, the live
gateway, and offline evaluation all import the same fail-closed checkpoint
classifier and data-path resolver so a checkpoint cannot silently select a
different context organization in different entry points.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


CAUSAL_OBSERVATION_CACHE_V1 = "causal_observation_cache_v1"
RECURSIVE_HISTORY_V2 = "recursive_history_v2"
RECURSIVE_HISTORY_WITH_RAW_WORKING_MEMORY_ABLATION_V1 = (
    "recursive_history_with_raw_working_memory_ablation_v1"
)
LATEST_TOOL_EXCHANGE_RAW_V1 = "latest_tool_exchange_raw_v1"
RECURSIVE_WORKING_MEMORY_V3 = "recursive_working_memory_v3"
QWEN36_TOKEN_CLOCK_V1 = "qwen36_token_clock_v1"
QWEN36_ROUND_RECURSIVE_V1 = "qwen36_round_recursive_v1"
ONE_ROUND_LAG_RECURSIVE_WORKING_MEMORY_V3 = (
    "one_round_lag_recursive_working_memory_v3"
)
REPLACE_PREVIOUS_MEMORY_KEEP_LATEST_RAW_WORKING_MEMORY_V3 = (
    "replace_previous_memory_keep_latest_raw_working_memory_v3"
)
LEGACY_CHECKPOINT_SCHEMA = "grepseek_context_compression_training_v1"
RECURSIVE_SFT_CHECKPOINT_SCHEMA = "grepseek_recursive_training_v2"
RECURSIVE_STREAMING_CHECKPOINT_SCHEMA = (
    "grepseek_recursive_streaming_onpolicy_training_v1"
)
WORKING_MEMORY_V3_CHECKPOINT_SCHEMA = (
    "grepseek_recursive_working_memory_training_v3"
)
WORKING_MEMORY_PURE_KL_V4_CHECKPOINT_SCHEMA = (
    "grepseek_recursive_working_memory_pure_kl_training_v4"
)
QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA = "qwen36_27b_token_clock_serving_v1"
QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA = (
    "qwen38_27b_r2eact_token_clock_serving_v1"
)
WORKING_MEMORY_CHECKPOINT_SCHEMAS = frozenset(
    {
        WORKING_MEMORY_V3_CHECKPOINT_SCHEMA,
        WORKING_MEMORY_PURE_KL_V4_CHECKPOINT_SCHEMA,
    }
)
WORKING_MEMORY_V3_SOURCE_CONTRACT = "semantic_working_memory_v3"
TOKEN_TRIGGERED_WORKING_MEMORY_POLICY = (
    "m_plus_unprotected_raw_working_memory_threshold_v2"
)
QWEN36_ROUND_WORKING_MEMORY_POLICY = (
    "latest_k_complete_tool_response_rounds_raw_v1"
)
QWEN36_ROUND_RECURSIVE_STATE_POLICY = (
    "replace_previous_memory_at_complete_tool_response_boundary_v1"
)
QWEN36_ROUND_RECURSIVE_SOURCE_CONTRACT = (
    "complete_react_round_chunked_at_checkpoint_block_size_v1"
)
QWEN36_ROUND_TOOL_RESULT_FIT_POLICY = (
    "server_headroom_before_first_compression_then_checkpoint_headroom_v2"
)
QWEN36_ROUND_EXPERIMENTAL_TOOL_RESULT_FIT_POLICY = (
    "server_generation_headroom_before_and_after_soft_memory_experimental_v1"
)
QWEN36_ROUND_GENERATION_RESERVE_TOKENS = 1_024
QWEN36_ROUND_DYNAMIC_GENERATION_BUDGET_POLICY = (
    "remaining_context_before_first_compression_then_checkpoint_8192_cap"
)
QWEN36_ROUND_FIXED_GENERATION_BUDGET_POLICY = (
    "configured_cap_before_first_compression_then_checkpoint_8192_cap"
)
QWEN36_POST_COMPRESSION_GENERATION_POLICY_CHECKPOINT_CAP = (
    "checkpoint_8192_cap"
)
QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL = (
    "sglang_remaining_context_experimental"
)
QWEN36_POST_COMPRESSION_GENERATION_POLICIES = frozenset(
    {
        QWEN36_POST_COMPRESSION_GENERATION_POLICY_CHECKPOINT_CAP,
        QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL,
    }
)
QWEN36_ROUND_EXPERIMENTAL_GENERATION_BUDGET_POLICY = (
    "sglang_remaining_context_for_postcompression_recovery_experimental_v1"
)
TOKEN_BUDGET_WORKING_MEMORY_WINDOW_POLICY = (
    "protected_latest_complete_exchanges_token_budget_v2"
)
WORKING_MEMORY_V3_TOOL_RESULT_FIT_POLICY = (
    "fit_tool_result_to_semantic_and_protected_raw_budgets_v2"
)
DEFAULT_WORKING_MEMORY_TOKENS = 2_048
DEFAULT_COMPRESSION_TRIGGER_TOKENS = 4_096
MAX_WORKING_MEMORY_TOKENS = 8_191
MAX_COMPRESSION_TRIGGER_TOKENS = 8_191
WORKING_MEMORY_V3_GENERATION_RESERVE_TOKENS = 1_024
GREPSEEK_COMPRESSOR_SERVED_MODEL_PREFIX = "grepseek-compressor-r"

_LEGACY_SOURCE_ENCODING = "one_full_raw_history_prefill_per_sample"
_GENERATION_BOUNDARY_IDS = (248046, 198, 248045, 74455, 198, 248068, 198)
_ASSISTANT_OPENING_IDS = (248045, 74455, 198, 248068, 198)


@dataclass(frozen=True)
class GrepSeekCheckpointServingContract:
    """Normalized fields that dispatch every serving/evaluation entry point."""

    context_organization: str
    checkpoint_schema: str
    compression_ratio: int

    @property
    def recursive(self) -> bool:
        return self.context_organization in {
            RECURSIVE_HISTORY_V2,
            RECURSIVE_WORKING_MEMORY_V3,
            QWEN36_TOKEN_CLOCK_V1,
            QWEN36_ROUND_RECURSIVE_V1,
        }

    @property
    def working_memory_v3(self) -> bool:
        return self.context_organization == RECURSIVE_WORKING_MEMORY_V3

    def to_json(self) -> dict[str, Any]:
        return {
            "context_organization": self.context_organization,
            "checkpoint_schema": self.checkpoint_schema,
            "compression_ratio": self.compression_ratio,
        }


@dataclass(frozen=True)
class ResolvedGrepSeekLiveData:
    """Prompt source data, absent for serving-only fixed-prompt checkpoints."""

    prepared_data: Optional[Path]
    raw_parquet: Optional[Path]
    split_config: Mapping[str, Any]


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_working_memory_token_budget(
    working_memory_tokens: Any,
    compression_trigger_tokens: Any,
) -> tuple[int, int]:
    """Validate protected raw-W and left-context compaction budgets.

    ``working_memory_tokens`` protects the newest complete raw exchanges.
    ``compression_trigger_tokens`` applies only to ``M + unprotected raw W``.
    The strict integer checks prevent booleans, floats, and numeric strings
    from silently changing a live rollout policy at the HTTP boundary.
    """

    for name, value in (
        ("working_memory_tokens", working_memory_tokens),
        ("compression_trigger_tokens", compression_trigger_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    working = int(working_memory_tokens)
    trigger = int(compression_trigger_tokens)
    if working <= 0:
        raise ValueError("working_memory_tokens must be positive")
    if working > MAX_WORKING_MEMORY_TOKENS:
        raise ValueError(
            "working_memory_tokens must not exceed "
            f"{MAX_WORKING_MEMORY_TOKENS}"
        )
    if trigger <= 0:
        raise ValueError("compression_trigger_tokens must be positive")
    if trigger > MAX_COMPRESSION_TRIGGER_TOKENS:
        raise ValueError(
            "compression_trigger_tokens must not exceed "
            f"{MAX_COMPRESSION_TRIGGER_TOKENS}"
        )
    return working, trigger


def default_grepseek_compressor_served_model_name(
    compression_ratio: int,
) -> str:
    """Return the stable default SGLang alias for a verified ratio."""

    ratio = _positive_int(compression_ratio, name="compression_ratio")
    return f"{GREPSEEK_COMPRESSOR_SERVED_MODEL_PREFIX}{ratio}"


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint config is missing {name}")
    return value


def _validate_recursive_sft(config: Mapping[str, Any]) -> None:
    if config.get("format_version") != 2:
        raise ValueError("recursive SFT checkpoint requires format_version=2")
    recurrence = _mapping(config.get("recurrence_abi"), name="recurrence_abi")
    required = {
        "previous_soft_memory_participates_in_next_compression": True,
        "raw_prefix_policy": "persistent_native_system_user",
        "compressed_rows": "all_physical_rows_after_P_in_mixed_target_encoding",
        "slot_count": "ceil(physical_source_rows/128)",
    }
    for name, expected in required.items():
        if recurrence.get(name) != expected:
            raise ValueError(
                f"recursive SFT checkpoint has incompatible recurrence_abi.{name}"
            )
    if tuple(recurrence.get("generation_boundary_ids", ())) != (
        _GENERATION_BOUNDARY_IDS
    ):
        raise ValueError(
            "recursive SFT checkpoint has incompatible generation boundary ids"
        )


def _validate_recursive_streaming(config: Mapping[str, Any]) -> None:
    if config.get("format_version") != 1:
        raise ValueError("recursive streaming checkpoint requires format_version=1")
    initialization = _mapping(
        config.get("initialization"), name="initialization"
    )
    if initialization.get("config_schema") != RECURSIVE_SFT_CHECKPOINT_SCHEMA:
        raise ValueError(
            "recursive streaming checkpoint is not initialized from recursive-v2"
        )
    run_contract = _mapping(config.get("run_contract"), name="run_contract")
    admission = _mapping(run_contract.get("admission"), name="run_contract.admission")
    if admission.get("recovery") != "P+M_k+G7+target":
        raise ValueError(
            "recursive streaming checkpoint has incompatible recovery contract"
        )
    source = admission.get("source")
    if not isinstance(source, str) or "recursive_source_widths" not in source:
        raise ValueError(
            "recursive streaming checkpoint has incompatible source contract"
        )


def _validate_working_memory_v3(
    config: Mapping[str, Any], *, ratio: int, args: Mapping[str, Any]
) -> None:
    """Validate the checkpoint-native P/M/semantic-W/raw-W/G5 serving ABI."""

    schema = config.get("schema")
    pure_kl = schema == WORKING_MEMORY_PURE_KL_V4_CHECKPOINT_SCHEMA
    expected_format_version = 4 if pure_kl else 3
    if config.get("format_version") != expected_format_version:
        raise ValueError(
            "working-memory checkpoint requires format_version="
            f"{expected_format_version} for schema {schema!r}"
        )
    if config.get("data_contract") != "working_memory_v3":
        raise ValueError(
            "working-memory-v3 checkpoint requires data_contract=working_memory_v3"
        )
    if args.get("data_contract") != "working_memory_v3":
        raise ValueError(
            "working-memory-v3 checkpoint args require data_contract=working_memory_v3"
        )
    if ratio != 16:
        raise ValueError(
            "recursive_working_memory_v3 requires physical compression_ratio=16"
        )
    if args.get("loss_supervision") != "full_assistant":
        raise ValueError(
            "working-memory-v3 checkpoint requires full_assistant supervision"
        )
    if args.get("recurrent_gradient") != "full":
        raise ValueError("working-memory-v3 checkpoint requires full recurrent gradient")
    if pure_kl:
        if (
            config.get("training_objective") != "raw_teacher_full_vocab_kl"
            or config.get("training_objective_mode")
            != "raw_teacher_full_vocab_kl"
            or args.get("training_objective") != "raw_teacher_full_vocab_kl"
            or args.get("kl_temperature") != 1.0
            or config.get("loss_supervision_mode") != "none_gold_ce"
        ):
            raise ValueError(
                "working-memory pure-KL checkpoint has an incompatible objective"
            )
        objective = _mapping(config.get("objective"), name="objective")
        expected_objective = {
            "name": (
                "raw_teacher_to_compressed_student_exact_full_vocabulary_"
                "forward_kl_only"
            ),
            "direction": "KL(p_teacher_raw || p_student_compressed)",
            "temperature": 1.0,
            "vocabulary": "complete_frozen_target_vocabulary",
            "gold_cross_entropy_weight": 0.0,
        }
        for name, expected in expected_objective.items():
            if objective.get(name) != expected:
                raise ValueError(
                    f"working-memory pure-KL checkpoint has incompatible objective.{name}"
                )
        teacher_forcing = _mapping(
            config.get("teacher_forcing_contract"),
            name="teacher_forcing_contract",
        )
        if (
            teacher_forcing.get("gold_cross_entropy_optimized") is not False
            or teacher_forcing.get("path")
            != "same_sft_target_ids_for_teacher_and_student"
        ):
            raise ValueError(
                "working-memory pure-KL checkpoint has incompatible teacher forcing"
            )

    max_encoder_tokens = _positive_int(
        args.get("max_encoder_tokens"), name="args.max_encoder_tokens"
    )
    if max_encoder_tokens != 8192:
        raise ValueError(
            "working-memory-v3 checkpoint requires max_encoder_tokens=8192"
        )
    recurrence = _mapping(config.get("recurrence_abi"), name="recurrence_abi")
    required = {
        "name": ONE_ROUND_LAG_RECURSIVE_WORKING_MEMORY_V3,
        "prompt_layout": "P_plus_optional_M_plus_latest_raw_W_plus_G5",
        "P": (
            "complete_native_system_plus_user_through_user_im_end_newline;"
            "always_discrete_and_never_compressed"
        ),
        "M": "recursive_soft_memory_from_older_semantic_working_rounds",
        "W": "latest_complete_native_assistant_tool_exchange_kept_raw",
        "update_1": "M1=C(P_conditioning,semantic_W1)",
        "update_k": "Mk=C(P_conditioning,M_prev,semantic_Wk)",
        "recovery_k": "target_k_from_P_plus_Mk_plus_raw_Wk_plus_1_plus_G5",
        "compressed_rows": (
            "previous_soft_memory_plus_semantic_assistant_and_tool_content;"
            "no_intermediate_chat_template_rows"
        ),
        "slot_count": "ceil(physical_source_rows/16)",
        "previous_soft_memory_participates_in_next_compression": True,
        "raw_prefix_and_latest_working_memory_are_not_compressor_source": True,
        "recurrent_gradient": "full",
    }
    for name, expected in required.items():
        if recurrence.get(name) != expected:
            raise ValueError(
                f"working-memory-v3 checkpoint has incompatible recurrence_abi.{name}"
            )

    g5 = _mapping(recurrence.get("G5"), name="recurrence_abi.G5")
    expected_g5 = {
        "text": "<|im_start|>assistant\n<think>\n",
        "ids": list(_ASSISTANT_OPENING_IDS),
        "regular_tokens": len(_ASSISTANT_OPENING_IDS),
        "relationship_to_memory_slots": (
            "native_generation_template_only_not_compressor_output"
        ),
    }
    if dict(g5) != expected_g5:
        raise ValueError("working-memory-v3 checkpoint has incompatible recurrence_abi.G5")

    source_cap = _mapping(
        recurrence.get("source_cap"), name="recurrence_abi.source_cap"
    )
    expected_source_cap: dict[str, Any] = {
        "max_encoder_tokens": max_encoder_tokens,
        "compressor_source_counted_rows": "P_plus_M_prev_plus_semantic_W",
        "recovery_counted_rows": (
            "P_plus_M_current_plus_latest_raw_W_plus_G5_plus_full_target"
        ),
        "policy": "fail_closed_never_crop",
        "telemetry": "train/max_source_encoder_tokens_and_train/max_recovery_tokens",
    }
    if pure_kl:
        expected_source_cap["raw_teacher_counted_rows"] = (
            "P_plus_all_raw_W_plus_G5_plus_full_target"
        )
        expected_source_cap["telemetry"] = (
            "train/max_source_encoder_tokens_train/max_student_recovery_tokens_"
            "and_train/max_teacher_recovery_tokens"
        )
    if dict(source_cap) != expected_source_cap:
        raise ValueError(
            "working-memory-v3 checkpoint has incompatible recurrence_abi.source_cap"
        )


def _validate_qwen36_token_clock(
    config: Mapping[str, Any], *, ratio: int, args: Mapping[str, Any]
) -> None:
    """Validate the exported 27B fixed-token-clock inference ABI."""

    if config.get("format_version") != 1 or ratio != 16:
        raise ValueError("Qwen3.6 token-clock serving requires format_version=1 and r16")
    compressor = _mapping(config.get("compressor"), name="compressor metadata")
    expected_geometry: dict[str, Any] = {
        "architecture": "GrepSeekDFlashCompressor",
        "target_hidden_size": 5120,
        "hidden_size": 5120,
        "target_layer_ids": [1, 16, 31, 46, 61],
        "selected_source_width": 25600,
        "num_hidden_layers": 5,
        "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
        "local_window": 2048,
        "slot_padding_side": "left",
    }
    for name, expected in expected_geometry.items():
        if compressor.get(name) != expected:
            raise ValueError(f"Qwen3.6 token-clock compressor geometry mismatch: {name}")
    if args.get("compression_ratio") != 16 or args.get("max_encoder_tokens") != 8192:
        raise ValueError("Qwen3.6 token-clock args require r16 and max_encoder_tokens=8192")
    abi = _mapping(config.get("token_clock_serving_abi"), name="token_clock_serving_abi")
    required = {
        "name": "recursive_fixed_token_clock_v1",
        "persistent_prefix": "native_system_and_initial_user_never_compressed",
        "eviction_block_tokens": 1024,
        "raw_working_memory_tokens": 1024,
        "update_0": "M0=C(P_conditioning,B0)",
        "update_k": "Mk=C(P_conditioning,M_prev,Bk)",
        "recovery": "P_plus_M_current_plus_latest_1024_raw_history_tokens",
        "slot_count": "ceil((previous_memory_slots+1024)/16)",
        "boundary_policy": "left_to_right_contiguous_token_blocks",
        "crop_policy": "never_crop_fail_closed",
    }
    for name, expected in required.items():
        if abi.get(name) != expected:
            raise ValueError(f"Qwen3.6 token-clock ABI mismatch: {name}")
    deployment = _mapping(config.get("deployment_export"), name="deployment_export")
    if deployment.get("source_curriculum_stage") != "epoch_1_one_shot_only":
        raise ValueError("Qwen3.6 step-1750 exploratory provenance is missing")
    # Early Qwen3.6 exports carried this serving-ablation label.  It is no
    # longer emitted, but accepting the one historical value keeps those
    # already-published artifacts readable without weakening any execution
    # geometry or token-clock invariant.
    legacy_status = deployment.get("recursive_serving_status")
    if legacy_status is not None and legacy_status != (
        "exploratory_out_of_training_distribution"
    ):
        raise ValueError("Qwen3.6 token-clock provenance is incompatible")


def _validate_qwen38_r2eact_token_clock(
    config: Mapping[str, Any], *, ratio: int, args: Mapping[str, Any]
) -> None:
    """Validate the stable Qwen3.8 27B r^2eAct deployment ABI."""

    if config.get("format_version") != 1 or ratio != 16:
        raise ValueError(
            "Qwen3.8 r2eAct token-clock serving requires format_version=1 and r16"
        )
    if config.get("target_family") != "qwen38":
        raise ValueError("Qwen3.8 r2eAct deployment requires target_family=qwen38")
    compressor = _mapping(config.get("compressor"), name="compressor metadata")
    expected_geometry: dict[str, Any] = {
        "architecture": "GrepSeekDFlashCompressor",
        "compression_ratio": 16,
        "target_hidden_size": 5120,
        "hidden_size": 5120,
        "target_layer_ids": [1, 16, 31, 46, 61],
        "selected_source_width": 25600,
        "num_hidden_layers": 5,
        "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
        "local_window": 2048,
        "slot_padding_side": "left",
    }
    for name, expected in expected_geometry.items():
        if compressor.get(name) != expected:
            raise ValueError(
                f"Qwen3.8 r2eAct token-clock compressor geometry mismatch: {name}"
            )
    if (
        args.get("compression_ratio") != 16
        or args.get("max_encoder_tokens") != 12288
        or args.get("max_target_sequence_tokens") != 12288
    ):
        raise ValueError(
            "Qwen3.8 r2eAct token-clock args require r16 and 12288-token "
            "source/target caps"
        )
    target_model = config.get("target_model")
    if (
        not isinstance(target_model, str)
        or not target_model
        or args.get("target_model") != target_model
        or not Path(target_model).is_absolute()
    ):
        raise ValueError(
            "Qwen3.8 r2eAct deployment must bind one absolute full target model"
        )
    abi = _mapping(
        config.get("token_clock_serving_abi"), name="token_clock_serving_abi"
    )
    required = {
        "name": "recursive_fixed_token_clock_v1",
        "persistent_prefix": "native_system_and_initial_user_never_compressed",
        "eviction_block_tokens": 1024,
        "raw_working_memory_tokens": 1024,
        "update_0": "M0=C(P_conditioning,B0)",
        "update_k": "Mk=C(P_conditioning,M_prev,Bk)",
        "recovery": "P_plus_M_current_plus_latest_1024_raw_history_tokens",
        "slot_count": "ceil((previous_memory_slots+1024)/16)",
        "boundary_policy": "left_to_right_contiguous_token_blocks",
        "crop_policy": "never_crop_fail_closed",
    }
    for name, expected in required.items():
        if abi.get(name) != expected:
            raise ValueError(f"Qwen3.8 r2eAct token-clock ABI mismatch: {name}")
    deployment = _mapping(config.get("deployment_export"), name="deployment_export")
    stage = deployment.get("source_curriculum_stage")
    if stage == "mmfine_repeat_ce_v4":
        source = _mapping(config.get("source_training_config"), name="source_training_config")
        contract = dict(source)
        for key in ("run_contract_sha256", "output_dir", "tensorboard_dir", "resume_from", "init_compressor"):
            contract.pop(key, None)
        contract_hash = hashlib.sha256(json.dumps(
            contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        source_args = _mapping(source.get("args"), name="source training args")
        if (
            source.get("schema") != "qwen38_mmfine_repeat_ce_tp2_fsdp_v4"
            or source.get("format_version") != 4
            or source.get("objective_schema") != "qwen38_mmfine_repeat_position_weighted_ce_v2"
            or deployment.get("source_schema") != source["schema"]
            or deployment.get("source_objective") != source["objective_schema"]
            or source.get("run_contract_sha256") != contract_hash
            or deployment.get("source_run_contract_sha256") != contract_hash
            or source.get("compressor") != compressor
            or source_args.get("target_model") != target_model
            or type(deployment.get("source_global_step")) is not int
            or deployment["source_global_step"] <= 0
        ):
            raise ValueError("MMFine serving source provenance is inconsistent")
    elif stage != "epoch_1_one_shot_only":
        raise ValueError("Qwen3.8 r2eAct source curriculum provenance is missing")
    for forbidden in ("recursive_serving_status", "serving_ablation"):
        if forbidden in deployment or forbidden in config:
            raise ValueError(
                f"Qwen3.8 r2eAct deployment must not contain {forbidden}"
            )


def resolve_grepseek_checkpoint_serving_contract(
    config: Mapping[str, Any],
) -> GrepSeekCheckpointServingContract:
    """Classify one verified checkpoint config and reject ambiguous variants."""

    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    schema = config.get("schema")
    expected_target_model_type = (
        "qwen3_5_multimodal"
        if schema == QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA
        else "qwen3_5_text"
    )
    if config.get("target_model_type") != expected_target_model_type:
        raise ValueError(
            "checkpoint target_model_type must be " f"{expected_target_model_type}"
        )
    compressor = _mapping(config.get("compressor"), name="compressor metadata")
    ratio = _positive_int(
        compressor.get("compression_ratio"), name="compressor.compression_ratio"
    )
    args = _mapping(config.get("args"), name="args")
    args_ratio = args.get("compression_ratio")
    if args_ratio is not None and _positive_int(
        args_ratio, name="args.compression_ratio"
    ) != ratio:
        raise ValueError("checkpoint args/compressor compression ratios disagree")

    if schema == QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA:
        _validate_qwen38_r2eact_token_clock(config, ratio=ratio, args=args)
        return GrepSeekCheckpointServingContract(
            # Qwen3.8 has a distinct, stable checkpoint schema but deliberately
            # reuses the established token-clock runtime dispatch.
            context_organization=QWEN36_TOKEN_CLOCK_V1,
            checkpoint_schema=QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
            compression_ratio=ratio,
        )
    if schema == QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA:
        _validate_qwen36_token_clock(config, ratio=ratio, args=args)
        return GrepSeekCheckpointServingContract(
            context_organization=QWEN36_TOKEN_CLOCK_V1,
            checkpoint_schema=QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
            compression_ratio=ratio,
        )
    if (
        config.get("format_version") == 1
        and config.get("source_encoding") == _LEGACY_SOURCE_ENCODING
        and schema in {None, LEGACY_CHECKPOINT_SCHEMA}
    ):
        return GrepSeekCheckpointServingContract(
            context_organization=CAUSAL_OBSERVATION_CACHE_V1,
            checkpoint_schema=LEGACY_CHECKPOINT_SCHEMA,
            compression_ratio=ratio,
        )

    if schema in WORKING_MEMORY_CHECKPOINT_SCHEMAS:
        _validate_working_memory_v3(config, ratio=ratio, args=args)
        return GrepSeekCheckpointServingContract(
            context_organization=RECURSIVE_WORKING_MEMORY_V3,
            checkpoint_schema=str(schema),
            compression_ratio=ratio,
        )
    if schema == RECURSIVE_SFT_CHECKPOINT_SCHEMA:
        _validate_recursive_sft(config)
    elif schema == RECURSIVE_STREAMING_CHECKPOINT_SCHEMA:
        _validate_recursive_streaming(config)
    else:
        raise ValueError(
            "unsupported context-compression checkpoint serving schema: "
            f"{schema!r}"
        )
    if ratio != 128:
        raise ValueError("recursive_history_v2 requires physical compression_ratio=128")
    return GrepSeekCheckpointServingContract(
        context_organization=RECURSIVE_HISTORY_V2,
        checkpoint_schema=str(schema),
        compression_ratio=ratio,
    )


def _resolve_existing_file(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _normalized_split_config(value: Mapping[str, Any]) -> dict[str, Any]:
    fraction = value.get("validation_fraction")
    # Streaming checkpoints may carry both the learner RNG seed and the
    # dataset split seed.  The latter owns held-out membership.  Prepared-data
    # manifests use the historical plain ``seed`` key and fall back to it.
    seed = value.get("split_seed", value.get("seed"))
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise ValueError("data split validation_fraction must be numeric")
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("data split validation_fraction must be in [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("data split seed must be an integer")
    return {"validation_fraction": fraction, "seed": int(seed)}


def resolve_grepseek_live_data_paths(
    saved_args: Mapping[str, Any],
    prepared_override: Optional[str | Path] = None,
    raw_override: Optional[str | Path] = None,
) -> ResolvedGrepSeekLiveData:
    """Resolve legacy/SFT prepared data or streaming on-policy raw data.

    Streaming on-policy checkpoints intentionally contain ``input_parquet``
    instead of ``prepared_data``.  Their split fields live directly in saved
    args, while legacy and recursive-SFT checkpoints bind them through the
    prepared-data manifest.
    """

    if not isinstance(saved_args, Mapping):
        raise TypeError("saved checkpoint args must be a mapping")
    prepared_value = prepared_override or saved_args.get("prepared_data")
    if prepared_value is not None:
        prepared = Path(prepared_value).expanduser().resolve()
        manifest_path = prepared / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"prepared manifest does not exist: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"failed to read prepared manifest: {manifest_path}"
            ) from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("prepared manifest must contain a JSON object")
        raw_value = raw_override or manifest.get("source_parquet")
        if raw_value is None:
            raise ValueError(
                "prepared manifest has no source_parquet; pass raw_override"
            )
        raw_path = _resolve_existing_file(raw_value, name="raw GrepSeek parquet")
        split = _mapping(manifest.get("config"), name="prepared split config")
        return ResolvedGrepSeekLiveData(
            prepared_data=prepared,
            raw_parquet=raw_path,
            split_config=_normalized_split_config(split),
        )

    raw_value = raw_override or saved_args.get("input_parquet")
    if raw_value is None:
        raise ValueError(
            "checkpoint args contain neither prepared_data nor input_parquet"
        )
    return ResolvedGrepSeekLiveData(
        prepared_data=None,
        raw_parquet=_resolve_existing_file(raw_value, name="raw GrepSeek parquet"),
        split_config=_normalized_split_config(saved_args),
    )


__all__ = [
    "CAUSAL_OBSERVATION_CACHE_V1",
    "GREPSEEK_COMPRESSOR_SERVED_MODEL_PREFIX",
    "LATEST_TOOL_EXCHANGE_RAW_V1",
    "LEGACY_CHECKPOINT_SCHEMA",
    "ONE_ROUND_LAG_RECURSIVE_WORKING_MEMORY_V3",
    "QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA",
    "QWEN36_TOKEN_CLOCK_V1",
    "QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA",
    "QWEN36_ROUND_RECURSIVE_V1",
    "QWEN36_ROUND_WORKING_MEMORY_POLICY",
    "QWEN36_ROUND_RECURSIVE_STATE_POLICY",
    "QWEN36_ROUND_RECURSIVE_SOURCE_CONTRACT",
    "QWEN36_ROUND_TOOL_RESULT_FIT_POLICY",
    "QWEN36_ROUND_EXPERIMENTAL_TOOL_RESULT_FIT_POLICY",
    "QWEN36_ROUND_GENERATION_RESERVE_TOKENS",
    "QWEN36_ROUND_DYNAMIC_GENERATION_BUDGET_POLICY",
    "QWEN36_ROUND_FIXED_GENERATION_BUDGET_POLICY",
    "QWEN36_POST_COMPRESSION_GENERATION_POLICIES",
    "QWEN36_POST_COMPRESSION_GENERATION_POLICY_CHECKPOINT_CAP",
    "QWEN36_POST_COMPRESSION_GENERATION_POLICY_SGLANG_EXPERIMENTAL",
    "QWEN36_ROUND_EXPERIMENTAL_GENERATION_BUDGET_POLICY",
    "REPLACE_PREVIOUS_MEMORY_KEEP_LATEST_RAW_WORKING_MEMORY_V3",
    "RECURSIVE_HISTORY_V2",
    "RECURSIVE_HISTORY_WITH_RAW_WORKING_MEMORY_ABLATION_V1",
    "RECURSIVE_WORKING_MEMORY_V3",
    "RECURSIVE_SFT_CHECKPOINT_SCHEMA",
    "RECURSIVE_STREAMING_CHECKPOINT_SCHEMA",
    "WORKING_MEMORY_V3_CHECKPOINT_SCHEMA",
    "WORKING_MEMORY_PURE_KL_V4_CHECKPOINT_SCHEMA",
    "WORKING_MEMORY_CHECKPOINT_SCHEMAS",
    "WORKING_MEMORY_V3_SOURCE_CONTRACT",
    "GrepSeekCheckpointServingContract",
    "ResolvedGrepSeekLiveData",
    "default_grepseek_compressor_served_model_name",
    "resolve_grepseek_checkpoint_serving_contract",
    "resolve_grepseek_live_data_paths",
]
