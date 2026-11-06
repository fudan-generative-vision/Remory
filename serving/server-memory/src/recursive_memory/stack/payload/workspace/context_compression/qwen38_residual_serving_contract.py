"""Independent, fail-closed ABI for summary-conditioned residual serving.

Metadata validation deliberately imports neither torch nor the training entry
point, so launchers can attest a checkpoint before claiming a task or GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

RESIDUAL_CHECKPOINT_SCHEMA = "qwen38_27b_summary_residual_serving_v1"
RESIDUAL_TRAINING_SCHEMA = "qwen38_compact_residual_context_sorted_response_rkl_tp2_fsdp_v3"
RESIDUAL_MANIFEST_SCHEMA = "context_compression_ddp_v1"
RESIDUAL_SERVING_ABI = {
    "name": "summary_conditioned_residual_pyramid_v1",
    "persistent_prefix": "native_system_and_tools",
    "source_features": "selected_layer_outputs_under_full_raw_causal_prefix",
    "summary_features": "final_normalized_hidden_summary_body_rows_under_native_prefix_and_summary_envelope",
    "parent_features": "selected_layer_outputs_under_native_prefix_plus_adjacent_child_soft_rows",
    "input_depth": "raw_is_zero_parent_uses_child_depth",
    "partial_block": "real_source_rows_with_virtual_1024_slot_allocation_no_synthetic_features",
    "summary_change": "rebuild_complete_raw_history_same_summary_all_levels",
    "recovery": "native_prefix_plus_summary_envelope_plus_memory_envelope_plus_continuation",
    "crop_policy": "never_crop_fail_closed",
}
_GEOMETRY = {
    "architecture": "DFlash2SummaryResidualCompressor", "compression_ratio": 16,
    "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408,
    "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
    "local_window": 2048, "max_depth": 8, "num_attention_heads": 32,
    "num_hidden_layers": 5, "num_key_value_heads": 8,
    "selected_source_width": 25600, "slot_padding_side": "left",
    "summary_conditioning": "independent_slot_Q_summary_KV_after_source_attention",
    "depth_coordinate": "input_depth_raw_is_zero", "target_hidden_size": 5120,
    "target_layer_ids": [5, 19, 33, 47, 61],
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_residual_model_config(config: Mapping[str, Any]) -> None:
    for key, expected in _GEOMETRY.items():
        if config.get(key) != expected or isinstance(config.get(key), bool):
            raise ValueError(f"residual compressor geometry mismatch: {key}")
    for key in ("output_embedding_rms", "summary_gate_init"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"invalid residual compressor {key}")
    if config["output_embedding_rms"] <= 0:
        raise ValueError("output_embedding_rms must be positive")


def validate_summary_contract(summary: Mapping[str, Any]) -> None:
    if summary.get("schema") != "qwen38_compact_residual_fullprefix_summary_v1":
        raise ValueError("unsupported residual summary contract")
    if summary.get("mode") != "full_causal_prefix_rebuild_no_previous_summary":
        raise ValueError("residual summary must be rebuilt from full causal history")
    maximum = _positive_int(summary.get("max_summary_tokens"), "max_summary_tokens")
    target = _positive_int(summary.get("target_summary_tokens"), "target_summary_tokens")
    if target > maximum or summary.get("temperature") != 0.0:
        raise ValueError("invalid residual summary sampling/budget")
    _positive_int(summary.get("end_token_id"), "end_token_id")
    for key in ("generation_suffix_ids", "student_summary_before_ids", "student_summary_after_ids",
                "student_memory_before_ids", "student_memory_after_ids"):
        ids = summary.get(key)
        if not isinstance(ids, list) or not ids or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
            raise ValueError(f"invalid residual summary envelope: {key}")
    if not isinstance(summary.get("prompt"), str) or not summary["prompt"]:
        raise ValueError("residual summary prompt is missing")
    files = summary.get("tokenizer_files")
    if not isinstance(files, dict) or not {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"} <= files.keys():
        raise ValueError("residual summary tokenizer receipts are missing")
    for name, digest in files.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("tokenizer receipt filename must be local")
        _digest(digest, f"tokenizer {name}")


def _validate_payload(config: Mapping[str, Any]) -> None:
    compressor, residual, summary = (config.get(k) for k in ("compressor", "residual", "summary_contract"))
    if not all(isinstance(value, Mapping) for value in (compressor, residual, summary)):
        raise ValueError("residual checkpoint is missing compressor/residual/summary metadata")
    validate_residual_model_config(compressor)
    validate_summary_contract(summary)
    expected = {"block_tokens": 1024, "compression_ratio": 16,
                "policy": "budget_driven_oldest_adjacent_equal_depth_pyramid",
                "summary_version": "full_causal_history_rebuild_every_response_same_summary_all_levels",
                "raw_prefix": "native_system_and_tools", "window_tokens": None}
    if config.get("online_training", {}).get("schema") == "qwen38_online_residual_v1":
        expected.update(summary_version="previous_summary_memory_and_new_actual_rounds",
                        raw_prefix="native_system_tools_and_original_user_task")
        branch_protocol = config['online_training'].get('branch_protocol')
        if branch_protocol:
            if branch_protocol != 'actual_rollout_frontier_independent_branches_schema_summary_v3':
                raise ValueError('unsupported online branch protocol')
            expected['summary_version'] = branch_protocol
    if any(residual.get(k) != v for k, v in expected.items()):
        raise ValueError("residual pyramid contract mismatch")
    budget = _positive_int(residual.get("memory_budget"), "residual memory budget")
    if budget < 64 or budget % 64:
        raise ValueError("residual memory budget must be a multiple of 64")
    if config.get("target_lora") is not False or config.get("target_trainable_parameters") != 0:
        raise ValueError("residual serving requires an unmodified frozen target")
    if not isinstance(config.get("target_model"), str) or not config["target_model"]:
        raise ValueError("residual target model is missing")


def load_residual_training_contract(checkpoint: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = Path(checkpoint).resolve()
    config = read_json(checkpoint / "run_config.json")
    trainer = read_json(checkpoint / "trainer_state.json")
    if any(row.get("schema") != RESIDUAL_TRAINING_SCHEMA or row.get("format_version") != 2 for row in (config, trainer)):
        raise ValueError("unsupported residual training checkpoint")
    material = dict(config)
    for key in ("run_contract_sha256", "output_dir", "tensorboard_dir", "resume_from", "init_compressor"):
        material.pop(key, None)
    digest = canonical_sha256(material)
    if config.get("run_contract_sha256") != digest or trainer.get("run_contract_sha256") != digest:
        raise ValueError("residual training contract identity mismatch")
    _positive_int(trainer.get("global_step"), "source global step")
    if trainer.get("rank_files") != ["rank-00000.pt", "rank-00001.pt"]:
        raise ValueError("residual export requires exactly two source rank files")
    _validate_payload(config)
    return config, trainer


@dataclass(frozen=True)
class ResidualServingContract:
    checkpoint_schema: str
    block_tokens: int
    compression_ratio: int
    memory_budget: int
    max_depth: int
    summary_contract: Mapping[str, Any]

    @property
    def slots_per_node(self) -> int:
        return self.block_tokens // self.compression_ratio


def resolve_residual_serving_contract(config: Mapping[str, Any]) -> ResidualServingContract:
    if config.get("schema") != RESIDUAL_CHECKPOINT_SCHEMA or config.get("format_version") != 1:
        raise ValueError("not a residual serving checkpoint")
    _validate_payload(config)
    if config.get("residual_serving_abi") != RESIDUAL_SERVING_ABI:
        raise ValueError("residual serving ABI mismatch")
    provenance = config.get("deployment_export")
    if not isinstance(provenance, Mapping):
        raise ValueError("residual serving provenance is missing")
    _digest(provenance.get("source_run_contract_sha256"), "source contract")
    _positive_int(provenance.get("source_global_step"), "source step")
    if provenance.get("source_schema") not in {RESIDUAL_TRAINING_SCHEMA,
            "qwen38_residual_throughput_training_v1", "qwen38_online_residual_training_v1", "qwen38_pretrained_dflash2_offline_stage2_v1", "qwen38_pretrained_dflash2_offline_stage2_shuffled_v2"}:
        raise ValueError("residual source schema mismatch")
    if config.get("token_clock_serving_abi") is not None:
        raise ValueError("residual checkpoint cannot claim legacy token-clock semantics")
    return ResidualServingContract(RESIDUAL_CHECKPOINT_SCHEMA, 1024, 16,
                                   config["residual"]["memory_budget"],
                                   config["compressor"]["max_depth"], config["summary_contract"])


def load_residual_serving_contract(checkpoint: str | Path, *, verify_weights: bool = False) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    config, manifest = read_json(checkpoint / "config.json"), read_json(checkpoint / "manifest.json")
    expected = {"schema": RESIDUAL_MANIFEST_SCHEMA, "export_schema": RESIDUAL_CHECKPOINT_SCHEMA,
                "config_file": "config.json",
                "rank_state_files": [], "frozen_target_saved": False}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("residual serving manifest mismatch")
    if type(manifest.get("world_size")) is not int or manifest["world_size"] not in {1, 2, 4}:
        raise ValueError("invalid residual export world size")
    if manifest.get("config_sha256") != canonical_sha256(config):
        raise ValueError("residual serving config checksum mismatch")
    digest = _digest(manifest.get("compressor_sha256"), "compressor checksum")
    from .remory_public_release import serving_weight_path, validate_serving_export
    weights = serving_weight_path(checkpoint)
    if weights.name == "model.safetensors":
        validate_serving_export(manifest)
        digest = manifest["safetensors_sha256"]
    if not weights.is_file():
        raise ValueError("residual compressor weights are missing")
    if verify_weights and file_sha256(weights) != digest:
        raise ValueError("residual compressor checksum mismatch")
    resolve_residual_serving_contract(config)
    return config


def verify_residual_target(config: Mapping[str, Any], target_model: str | Path | None = None) -> Path:
    target = Path(target_model or config["target_model"]).expanduser().resolve()
    model_config = read_json(target / "config.json")
    text = model_config.get("text_config", model_config)
    if model_config.get("model_type") != "qwen3_5" or text.get("hidden_size") != 5120 or text.get("num_hidden_layers") != 64:
        raise ValueError("residual serving target must be the complete Qwen3.8-27B multimodal model")
    for name, digest in config["summary_contract"]["tokenizer_files"].items():
        if file_sha256(target / name) != digest:
            raise ValueError(f"residual target tokenizer identity mismatch: {name}")
    return target


def build_summary_residual_compressor(config, *, device="cpu", dtype=None, checkpoint=None):
    import torch
    from context_compression.qwen38_dflash2_residual import build_residual
    values = config["compressor"]
    validate_residual_model_config(values)
    architecture = config["dflash2_architecture_config"]
    if checkpoint is not None:
        from .remory_public_release import architecture_path
        architecture = architecture_path(checkpoint, config)
    with torch.device(device):
        model = build_residual(architecture,
            output_embedding_rms=values["output_embedding_rms"], gradient_checkpointing=False)
    return model.to(dtype=dtype) if dtype is not None else model
