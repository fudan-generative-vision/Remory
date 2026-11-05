"""Export two-rank Qwen3.6/Qwen3.8 r^2eAct training state for inference.

The Qwen3.6 trainer intentionally writes FSDP sharded, optimizer-complete
checkpoints.  Serving must not load optimizer/RNG state and expects one direct
``compressor.pt`` state dict.  This exporter is therefore a two-process CPU
collective: every rank reads its own training shard, writes a model-only local
part, and rank zero reconstructs and atomically publishes the full state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from context_compression.grepseek_modeling import GrepSeekDFlashCompressor
from context_compression.mindforge_ce_manifest import (
    BASELINE_PORTABLE_STAGE,
    MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA,
    MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA_V2,
    MINDFORGE_CE_PORTABLE_STAGE,
    MINDFORGE_FRONTIER_CE_DATA_CONTRACT,
    exact_h4096_training_manifest_receipt,
)
from context_compression.mindforge_ce_manifest import (
    SCHEMA as MINDFORGE_DECISION_MANIFEST_SCHEMA,
)
from context_compression.mindforge_ce_manifest import (
    VIEW_SCHEMA as MINDFORGE_DECISION_MANIFEST_VIEW_SCHEMA,
)
from context_compression.qwen36_recursive_data import (
    COMPRESSION_RATIO,
    EVICTION_BLOCK_TOKENS,
    RAW_WORKING_MEMORY_TOKENS,
)
from context_compression.r2eact_serving_contract import (
    QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
    QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
)
from context_compression.train_qwen36_recursive_compressor import (
    FORMAT_VERSION,
    SCHEMA_BY_TARGET_FAMILY,
    _checkpoint_schema_for_data_contract,
)

EXPORT_SCHEMA = QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA
EXPORT_SCHEMA_BY_TARGET_FAMILY = {
    "qwen36": QWEN36_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
    "qwen38": QWEN38_R2EACT_TOKEN_CLOCK_CHECKPOINT_SCHEMA,
}
LEGACY_QWEN36_SOURCE_SCHEMA = "qwen36_27b_recursive_compressor_tp2_fsdp_v1"
SOURCE_TOKEN_CAP_BY_TARGET_FAMILY = {"qwen36": 8_192, "qwen38": 12_288}
MANIFEST_SCHEMA = "context_compression_ddp_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-model-override",
        type=Path,
        default=None,
        help=(
            "Full multimodal Qwen3.8 target directory to bind into the deployment; "
            "it must be the source of the training text-only snapshot"
        ),
    )
    return parser


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _plain_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a plain integer")
    return int(value)


def _lower_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_run_contract_sha256(run_config: Mapping[str, Any]) -> str:
    material = dict(run_config)
    for name in (
        "run_contract_sha256",
        "output_dir",
        "resume_from",
        "init_compressor",
        "resume_cursor_override",
    ):
        material.pop(name, None)
    return _canonical_json_sha256(material)


def _mindforge_ce_export_lineage(
    run_config: Mapping[str, Any], trainer: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return strict CE provenance, or ``None`` for the legacy KL export.

    The legacy branch deliberately performs no new serialization: published
    step-1000 KL exports keep their byte-level config/manifest semantics.  A
    MindForge source, however, must prove the complete CE run contract before
    it may be labelled as such in a portable export.
    """

    data_contract = run_config.get("data_contract")
    is_mindforge = (
        isinstance(data_contract, Mapping)
        and data_contract.get("name") == MINDFORGE_FRONTIER_CE_DATA_CONTRACT
    )
    if not is_mindforge:
        objective = run_config.get("objective")
        source_schema = run_config.get("schema")
        args = run_config.get("args")
        declares_mindforge = (
            isinstance(source_schema, str)
            and source_schema.endswith("__mindforge_glm52_frontier_actions_v1")
        ) or (
            isinstance(args, Mapping)
            and args.get("data_contract") == MINDFORGE_FRONTIER_CE_DATA_CONTRACT
        )
        declares_frontier_ce = isinstance(objective, Mapping) and objective.get(
            "name"
        ) in {"ce", "ce_kl"}
        if declares_mindforge or declares_frontier_ce:
            raise ValueError(
                "frontier CE source is missing the MindForge data-contract lineage"
            )
        return None
    if run_config.get("target_family") != "qwen38":
        raise ValueError("MindForge CE export requires target_family=qwen38")

    source_run_contract = _lower_sha256(
        run_config.get("run_contract_sha256"), name="source run contract"
    )
    if source_run_contract != _canonical_run_contract_sha256(run_config):
        raise ValueError("MindForge CE source run contract self-hash mismatch")
    if trainer.get("run_contract_sha256") != source_run_contract:
        raise ValueError("MindForge CE trainer/run contract mismatch")

    expected_objective = {
        "name": "ce",
        "ce": "complete_native_frontier_assistant_action_only",
        "kl": None,
        "kl_loss_weight": None,
        "prompt_and_tool_observation_labels": -100,
    }
    if run_config.get("objective") != expected_objective:
        raise ValueError("MindForge portable export requires objective=ce exactly")
    expected_curriculum = {
        "all_epochs": (
            "recursive_stopgrad_cold_start_window; replay_only_states_over_"
            "sequence_cap; every_eligible_1024_eviction_directly_action_"
            "supervised; no_BPTT"
        )
    }
    if run_config.get("curriculum") != expected_curriculum:
        raise ValueError("MindForge CE source curriculum/stage is incompatible")

    manifest = data_contract.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("MindForge CE run has no primary manifest receipt")
    manifest_schema = manifest.get("schema")
    if (
        manifest_schema
        not in {
            MINDFORGE_DECISION_MANIFEST_SCHEMA,
            MINDFORGE_DECISION_MANIFEST_VIEW_SCHEMA,
        }
        or manifest.get("format_version") != 1
    ):
        raise ValueError("MindForge CE source used an unsupported manifest")
    manifest_artifact_sha = _lower_sha256(
        manifest.get("artifact_sha256"), name="training manifest artifact"
    )
    manifest_contract_sha = _lower_sha256(
        manifest.get("contract_sha256"), name="training manifest contract"
    )
    policy = manifest.get("policy")
    training_history_tokens = (
        4_096 if manifest_schema == MINDFORGE_DECISION_MANIFEST_VIEW_SCHEMA else 8_192
    )
    if not isinstance(policy, Mapping) or {
        "history_window_tokens": policy.get("history_window_tokens"),
        "train_trajectories": policy.get("train_trajectories"),
        "actions_per_train_trajectory": policy.get("actions_per_train_trajectory"),
        "validation_trajectories": policy.get("validation_trajectories"),
        "actions_per_validation_trajectory": policy.get(
            "actions_per_validation_trajectory"
        ),
    } != {
        "history_window_tokens": training_history_tokens,
        "train_trajectories": 512,
        "actions_per_train_trajectory": 4,
        "validation_trajectories": 16,
        "actions_per_validation_trajectory": 2,
    }:
        raise ValueError("MindForge CE source manifest has foreign history/counts")
    training_manifest: dict[str, Any] | None = None
    if manifest_schema == MINDFORGE_DECISION_MANIFEST_VIEW_SCHEMA:
        manifest_path_value = manifest.get("path")
        if not isinstance(manifest_path_value, str) or not manifest_path_value:
            raise ValueError("MindForge 4K training manifest path is missing")
        manifest_path = Path(manifest_path_value).expanduser().resolve()
        if not manifest_path.is_file() or _sha256(manifest_path) != (
            manifest_artifact_sha
        ):
            raise ValueError("MindForge 4K training manifest file/pin mismatch")
        full_manifest = _json(manifest_path)
        training_manifest = exact_h4096_training_manifest_receipt(
            full_manifest, artifact_sha256=manifest_artifact_sha
        )
        for name in (
            "schema",
            "format_version",
            "contract_sha256",
            "source",
            "tokenizer",
            "checkpoint",
            "adapter",
            "policy",
        ):
            if manifest.get(name) != full_manifest.get(name):
                raise ValueError(
                    f"MindForge 4K compact/full manifest field differs: {name}"
                )
        if (
            data_contract.get("pilot_profile") != "full-view-h4096-v1"
            or not isinstance(data_contract.get("pilot_geometry"), Mapping)
            or data_contract["pilot_geometry"].get("history_window_tokens") != 4_096
            or not isinstance(data_contract.get("history_window"), Mapping)
            or data_contract["history_window"].get("tokens") != 4_096
        ):
            raise ValueError("MindForge 4K training geometry/profile is inconsistent")

    baseline = manifest.get("checkpoint")
    initialization = run_config.get("compressor_initialization")
    if not isinstance(baseline, Mapping) or not isinstance(initialization, Mapping):
        raise ValueError("MindForge CE source has no baseline initialization receipt")
    baseline_receipt_sha = _lower_sha256(
        baseline.get("receipt_sha256"), name="baseline initialization receipt"
    )
    baseline_material = dict(baseline)
    baseline_material.pop("receipt_sha256", None)
    if _canonical_json_sha256(baseline_material) != baseline_receipt_sha:
        raise ValueError("MindForge baseline initialization receipt self-hash mismatch")
    baseline_compressor_sha = _lower_sha256(
        baseline.get("compressor_sha256"), name="baseline compressor"
    )
    baseline_target_sha = _lower_sha256(
        baseline.get("source_target_content_identity_sha256"),
        name="baseline target identity",
    )
    if (
        baseline.get("source_global_step") != 1_000
        or baseline.get("source_curriculum_stage") != BASELINE_PORTABLE_STAGE
        or initialization.get("checkpoint_receipt_sha256") != baseline_receipt_sha
        or initialization.get("artifact_sha256") != baseline_compressor_sha
        or initialization.get("source_global_step") != 1_000
        or initialization.get("source_curriculum_stage") != BASELINE_PORTABLE_STAGE
        or initialization.get("source_target_content_identity_sha256")
        != baseline_target_sha
        or initialization.get("source_target_content_identity")
        != baseline.get("source_target_content_identity")
        or run_config.get("target_content_identity")
        != baseline.get("source_target_content_identity")
    ):
        raise ValueError("MindForge CE baseline initialization lineage is inconsistent")

    implementation = run_config.get("implementation_identity")
    if not isinstance(implementation, Mapping):
        raise ValueError("MindForge CE source has no implementation identity")
    local_files = implementation.get("local_files")
    if not isinstance(local_files, Mapping) or not {
        "trainer",
        "data",
        "compressor",
        "loss",
        "parity_validator",
        "target_loader",
        "mindforge_data",
        "mindforge_manifest",
    }.issubset(local_files):
        raise ValueError("MindForge CE implementation identity is incomplete")
    for name, entry in local_files.items():
        if (
            not isinstance(name, str)
            or not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry.get("path"), str)
        ):
            raise ValueError("MindForge CE local implementation identity is malformed")
        _lower_sha256(entry.get("sha256"), name=f"implementation file {name}")
    adapter = manifest.get("adapter")
    if (
        not isinstance(adapter, Mapping)
        or manifest.get("adapter_source_sha256")
        != local_files["mindforge_data"]["sha256"]
        or adapter.get("implementation_sha256")
        != local_files["mindforge_data"]["sha256"]
    ):
        raise ValueError("MindForge adapter and implementation identities disagree")
    implementation_sha = _canonical_json_sha256(implementation)

    common_lineage = {
        "objective": "ce",
        "baseline_initialization": {
            "checkpoint_receipt_sha256": baseline_receipt_sha,
            "compressor_sha256": baseline_compressor_sha,
            "source_global_step": 1_000,
            "source_curriculum_stage": BASELINE_PORTABLE_STAGE,
            "source_target_content_identity_sha256": baseline_target_sha,
        },
        "implementation_identity_sha256": implementation_sha,
    }
    if training_manifest is not None:
        return {
            "schema": MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA_V2,
            **common_lineage,
            "training_history_window_tokens": 4_096,
            "training_manifest": training_manifest,
        }
    return {
        "schema": MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA,
        **common_lineage,
        "primary_manifest": {
            "artifact_sha256": manifest_artifact_sha,
            "contract_sha256": manifest_contract_sha,
        },
    }


def _source_target_family(
    trainer: Mapping[str, Any], run_config: Mapping[str, Any]
) -> str:
    trainer_identity = (trainer.get("schema"), trainer.get("format_version"))
    run_identity = (run_config.get("schema"), run_config.get("format_version"))
    if trainer_identity != run_identity:
        raise ValueError("source checkpoint and run config schema identities disagree")
    supported = {
        (LEGACY_QWEN36_SOURCE_SCHEMA, 1): "qwen36",
        (SCHEMA_BY_TARGET_FAMILY["qwen36"], FORMAT_VERSION): "qwen36",
        (SCHEMA_BY_TARGET_FAMILY["qwen38"], FORMAT_VERSION): "qwen38",
        (
            _checkpoint_schema_for_data_contract(
                "qwen38", MINDFORGE_FRONTIER_CE_DATA_CONTRACT
            ),
            FORMAT_VERSION,
        ): "qwen38",
    }
    target_family = supported.get(trainer_identity)
    if target_family is None:
        raise ValueError(
            "source checkpoint schema is not a supported Qwen3.6/Qwen3.8 "
            "r2eAct FSDP trainer schema"
        )
    args = run_config.get("args")
    args_family = args.get("target_family") if isinstance(args, Mapping) else None
    for name, declared in (
        ("target_family", run_config.get("target_family")),
        ("args.target_family", args_family),
    ):
        if declared is not None and declared != target_family:
            raise ValueError(
                f"source {name} disagrees with the checkpoint schema target family"
            )
    return target_family


def _target_text_snapshot_path(run_config: Mapping[str, Any]) -> Path:
    target_model = run_config.get("target_model")
    if not isinstance(target_model, str) or not target_model:
        raise ValueError("run config is missing the training target model")
    target_path = Path(target_model).expanduser().resolve()
    identity = run_config.get("target_content_identity")
    if isinstance(identity, Mapping):
        identity_path = identity.get("resolved_path")
        if not isinstance(identity_path, str) or not identity_path:
            raise ValueError(
                "run config target_content_identity is missing resolved_path"
            )
        if Path(identity_path).expanduser().resolve() != target_path:
            raise ValueError(
                "run config training target and target content identity disagree"
            )
    return target_path


def _text_tower_geometry(
    config: Mapping[str, Any], *, nested: bool, name: str
) -> tuple[int, int]:
    text_config: Mapping[str, Any]
    if nested:
        value = config.get("text_config")
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} is missing text_config")
        text_config = value
    else:
        text_config = config
    if text_config.get("model_type") != "qwen3_5_text":
        raise ValueError(f"{name} text tower must use model_type=qwen3_5_text")
    hidden_size = _plain_int(text_config.get("hidden_size"), name=f"{name}.hidden_size")
    num_hidden_layers = _plain_int(
        text_config.get("num_hidden_layers"),
        name=f"{name}.num_hidden_layers",
    )
    return hidden_size, num_hidden_layers


def _resolve_deployment_target_model(
    run_config: Mapping[str, Any],
    *,
    target_family: str,
    target_model_override: Path | None,
) -> Path:
    training_target = _target_text_snapshot_path(run_config)
    training_config_path = training_target / "config.json"
    if not training_config_path.is_file():
        raise FileNotFoundError(
            f"training target snapshot config does not exist: {training_config_path}"
        )
    training_config = _json(training_config_path)
    if training_config.get("architectures") != ["Qwen3_5ForCausalLM"]:
        raise ValueError("training target snapshot must use Qwen3_5ForCausalLM")
    training_geometry = _text_tower_geometry(
        training_config, nested=False, name="training target snapshot"
    )
    if training_geometry != (5120, 64):
        raise ValueError(
            "training target snapshot must have hidden_size=5120 and 64 layers"
        )

    if target_family == "qwen36":
        if target_model_override is not None:
            raise ValueError(
                "--target-model-override is supported only for target_family=qwen38"
            )
        return training_target
    if target_family != "qwen38":  # pragma: no cover - guarded by source classifier
        raise ValueError(f"unsupported r2eAct target family: {target_family}")
    if target_model_override is None:
        raise ValueError(
            "Qwen3.8 export requires --target-model-override pointing to the full "
            "multimodal target"
        )

    deployment_target = target_model_override.expanduser().resolve()
    deployment_config_path = deployment_target / "config.json"
    if not deployment_config_path.is_file():
        raise FileNotFoundError(
            f"deployment target config does not exist: {deployment_config_path}"
        )
    deployment_config = _json(deployment_config_path)
    if deployment_config.get("model_type") != "qwen3_5" or deployment_config.get(
        "architectures"
    ) != ["Qwen3_5ForConditionalGeneration"]:
        raise ValueError(
            "Qwen3.8 deployment target must be the full "
            "Qwen3_5ForConditionalGeneration model"
        )
    deployment_geometry = _text_tower_geometry(
        deployment_config, nested=True, name="Qwen3.8 deployment target"
    )
    if deployment_geometry != training_geometry:
        raise ValueError(
            "Qwen3.8 deployment target text_config hidden size/layer count differs "
            "from the training target snapshot"
        )

    snapshot_manifest_path = training_target / "text_snapshot_manifest.json"
    if not snapshot_manifest_path.is_file():
        raise FileNotFoundError(
            "Qwen3.8 training target snapshot manifest does not exist: "
            f"{snapshot_manifest_path}"
        )
    snapshot_manifest = _json(snapshot_manifest_path)
    if (
        snapshot_manifest.get("schema") != "qwen3_5_text_only_symlink_snapshot"
        or snapshot_manifest.get("format_version") != 2
    ):
        raise ValueError("Qwen3.8 training target snapshot manifest is incompatible")
    source = snapshot_manifest.get("source")
    if (
        not isinstance(source, str)
        or Path(source).expanduser().resolve() != deployment_target
    ):
        raise ValueError(
            "--target-model-override is not the full source bound by the training "
            "target snapshot"
        )
    expected_manifest_sha = run_config.get("target_text_snapshot_manifest_sha256")
    if (
        not isinstance(expected_manifest_sha, str)
        or _sha256(snapshot_manifest_path) != expected_manifest_sha
    ):
        raise ValueError(
            "Qwen3.8 training target snapshot manifest hash differs from run config"
        )
    if snapshot_manifest.get("derived_config_sha256") != _sha256(training_config_path):
        raise ValueError(
            "Qwen3.8 training text config differs from its snapshot receipt"
        )
    if snapshot_manifest.get("source_config_sha256") != _sha256(deployment_config_path):
        raise ValueError("Qwen3.8 full target config differs from its snapshot receipt")
    return deployment_target


def _validate_source(
    checkpoint: Path,
    run_config_path: Path,
    *,
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    trainer = _json(checkpoint / "trainer_state.json")
    run_config = _json(run_config_path)
    target_family = _source_target_family(trainer, run_config)
    if trainer.get("run_contract_sha256") != run_config.get("run_contract_sha256"):
        raise ValueError("source checkpoint and run config contracts disagree")
    if _plain_int(trainer.get("global_step"), name="global_step") <= 0:
        raise ValueError("source checkpoint global_step must be positive")
    rank_files = trainer.get("rank_files")
    if not isinstance(rank_files, list) or rank_files != [
        f"rank-{index:05d}.pt" for index in range(world_size)
    ]:
        raise ValueError("source checkpoint rank file contract is incomplete")
    rank_path = checkpoint / rank_files[rank]
    if not rank_path.is_file() or rank_path.stat().st_size <= 0:
        raise FileNotFoundError(f"source rank state is missing: {rank_path}")
    compressor = run_config.get("compressor")
    if not isinstance(compressor, dict):
        raise ValueError("run config is missing compressor geometry")
    required_geometry = {
        "architecture": "GrepSeekDFlashCompressor",
        "compression_ratio": COMPRESSION_RATIO,
        "target_hidden_size": 5120,
        "hidden_size": 5120,
        "target_layer_ids": [1, 16, 31, 46, 61],
        "selected_source_width": 25600,
        "head_dim": 128,
        "intermediate_size": 17408,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 5,
        "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
        "local_window": 2048,
        "slot_padding_side": "left",
    }
    for name, expected in required_geometry.items():
        if compressor.get(name) != expected:
            raise ValueError(
                f"unsupported {target_family} r2eAct compressor geometry: {name}"
            )
    if run_config.get("compression_ratio") != COMPRESSION_RATIO:
        raise ValueError("run config compression ratio differs from r16 geometry")
    if run_config.get("target_hidden_size", 5120) != 5120:
        raise ValueError("run config target hidden size differs from 5120")
    if run_config.get("target_layer_ids") != [1, 16, 31, 46, 61]:
        raise ValueError("run config target layer ids differ from r2eAct geometry")
    if run_config.get("eviction_block_tokens") != EVICTION_BLOCK_TOKENS:
        raise ValueError("run config eviction block differs from the token-clock ABI")
    if run_config.get("raw_working_memory_tokens") != RAW_WORKING_MEMORY_TOKENS:
        raise ValueError("run config working memory differs from the token-clock ABI")
    run_args = run_config.get("args")
    if not isinstance(run_args, Mapping):
        raise ValueError("run config is missing args")
    expected_cap = SOURCE_TOKEN_CAP_BY_TARGET_FAMILY[target_family]
    if run_args.get("max_target_sequence_tokens") != expected_cap:
        raise ValueError(
            f"{target_family} r2eAct training target cap must be {expected_cap}"
        )
    if target_family == "qwen38":
        if run_args.get("max_teacher_sequence_tokens") != expected_cap:
            raise ValueError("Qwen3.8 r2eAct training source/teacher cap must be 12288")
        context_budget = run_config.get("context_budget")
        ce_lineage = _mindforge_ce_export_lineage(run_config, trainer)
        if ce_lineage is not None:
            expected_ce_context_budget = {
                "target_native_context_tokens": 262_144,
                "teacher_sequence_hard_cap_tokens": None,
                "teacher_execution_mode": "disabled_for_ce_only",
                "teacher_monolithic_forward_cap_tokens": None,
                "teacher_cached_prefill_chunk_tokens": None,
                "teacher_cached_chunk_schedule": None,
                "student_single_forward_tokens": expected_cap,
                "teacher_cache": "disabled_no_raw_actor_teacher_forward",
                "linear_attention_chunk_patch": {
                    "enabled": False,
                    "patched_layers": 0,
                },
                "full_attention_cached_continuation_patch": {
                    "enabled": False,
                    "patched_layers": 0,
                },
                "sequence_parallelism": None,
            }
            if context_budget != expected_ce_context_budget:
                raise ValueError(
                    "MindForge CE-only context budget must disable every raw-teacher "
                    "execution path and bind the student cap to 12288"
                )
        elif not isinstance(context_budget, Mapping) or any(
            context_budget.get(name) != expected_cap
            for name in (
                "student_single_forward_tokens",
                "teacher_monolithic_forward_cap_tokens",
                "teacher_sequence_hard_cap_tokens",
            )
        ):
            raise ValueError(
                "Qwen3.8 r2eAct context budget must bind all source/target caps "
                "to 12288"
            )
    return trainer, run_config, rank_path


def _local_model_part(model: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in model.items():
        local_shards = getattr(value, "local_shards", None)
        metadata_method = getattr(value, "metadata", None)
        if callable(local_shards) and callable(metadata_method):
            metadata = metadata_method()
            shards = []
            for shard in local_shards():
                shard_metadata = shard.metadata
                shards.append(
                    {
                        "offsets": tuple(
                            int(item) for item in shard_metadata.shard_offsets
                        ),
                        "sizes": tuple(
                            int(item) for item in shard_metadata.shard_sizes
                        ),
                        "tensor": shard.tensor.detach().cpu().contiguous(),
                    }
                )
            result[name] = {
                "kind": "sharded",
                "shape": tuple(int(item) for item in metadata.size),
                "shards": shards,
            }
        elif isinstance(value, torch.Tensor):
            result[name] = {
                "kind": "replicated",
                "shape": tuple(int(item) for item in value.shape),
                "tensor": value.detach().cpu().contiguous(),
            }
        else:
            raise TypeError(
                f"unsupported model state value for {name!r}: {type(value)!r}"
            )
    return result


def _assemble(parts: Sequence[Mapping[str, Any]]) -> OrderedDict[str, torch.Tensor]:
    if not parts:
        raise ValueError("no local model parts were provided")
    key_set = set(parts[0])
    if any(set(part) != key_set for part in parts[1:]):
        raise ValueError("rank-local model state keys disagree")
    full_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in sorted(key_set):
        entries = [part[name] for part in parts]
        kinds = {entry.get("kind") for entry in entries}
        shapes = {tuple(entry.get("shape", ())) for entry in entries}
        if len(kinds) != 1 or len(shapes) != 1:
            raise ValueError(f"rank-local state metadata disagrees for {name!r}")
        shape = next(iter(shapes))
        kind = next(iter(kinds))
        if kind == "replicated":
            tensors = [entry.get("tensor") for entry in entries]
            if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
                raise TypeError(f"replicated tensor is missing for {name!r}")
            reference = tensors[0]
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                raise ValueError(f"replicated tensor differs across ranks for {name!r}")
            full_state[name] = reference
            continue
        if kind != "sharded":
            raise ValueError(f"unknown model state kind for {name!r}: {kind!r}")
        shard_records = [
            record for entry in entries for record in entry.get("shards", ())
        ]
        if not shard_records:
            raise ValueError(f"no physical shards were found for {name!r}")
        dtype = shard_records[0]["tensor"].dtype
        full = torch.empty(shape, dtype=dtype)
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        covered = 0
        for record in shard_records:
            offsets = tuple(record["offsets"])
            sizes = tuple(record["sizes"])
            tensor = record["tensor"]
            if tensor.dtype != dtype or tuple(tensor.shape) != sizes:
                raise ValueError(f"physical shard metadata is invalid for {name!r}")
            identity = (offsets, sizes)
            if identity in seen:
                raise ValueError(f"duplicate physical shard for {name!r}: {identity}")
            seen.add(identity)
            slices = tuple(
                slice(offset, offset + size) for offset, size in zip(offsets, sizes)
            )
            full[slices].copy_(tensor)
            covered += math.prod(sizes)
        if covered != math.prod(shape):
            raise ValueError(
                "physical shards do not exactly cover "
                f"{name!r}: {covered} != {math.prod(shape)}"
            )
        full_state[name] = full
    return full_state


def _validate_full_state(state: Mapping[str, torch.Tensor], output_rms: float) -> None:
    with torch.device("meta"):
        expected = GrepSeekDFlashCompressor.canonical_qwen36_27b(
            output_embedding_rms=output_rms,
            gradient_checkpointing=False,
        ).state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise ValueError(
            "exported state keys differ from the model: "
            f"missing={missing}, extra={extra}"
        )
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise ValueError(
                f"exported state shape differs for {name!r}: "
                f"{tuple(tensor.shape)} != {tuple(expected[name].shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"exported state contains a non-finite tensor: {name!r}")


def _deployment_config(
    run_config: Mapping[str, Any],
    trainer: Mapping[str, Any],
    checkpoint: Path,
    rank_files: Sequence[Path],
    *,
    target_family: str,
    deployment_target_model: Path,
) -> dict[str, Any]:
    export_schema = EXPORT_SCHEMA_BY_TARGET_FAMILY[target_family]
    max_encoder_tokens = SOURCE_TOKEN_CAP_BY_TARGET_FAMILY[target_family]
    target_model = str(deployment_target_model)
    ce_lineage = _mindforge_ce_export_lineage(run_config, trainer)
    source_provenance: dict[str, Any] = {
        "source_checkpoint": str(checkpoint),
        "source_global_step": int(trainer["global_step"]),
        "source_curriculum_stage": (
            MINDFORGE_CE_PORTABLE_STAGE
            if ce_lineage is not None
            else BASELINE_PORTABLE_STAGE
        ),
        "source_run_contract_sha256": str(trainer["run_contract_sha256"]),
        "source_rank_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in rank_files
        ],
    }
    if ce_lineage is not None:
        source_provenance.update(
            {
                "source_objective": "ce",
                "source_lineage": ce_lineage,
            }
        )
    config = dict(run_config)
    # These were historical serving experiment labels, not checkpoint-native
    # execution contracts.  New exports must not republish them.
    config.pop("recursive_serving_status", None)
    config.pop("serving_ablation", None)
    config.update(
        {
            "schema": export_schema,
            "format_version": 1,
            "target_family": target_family,
            "target_model": target_model,
            "target_model_type": (
                "qwen3_5_multimodal" if target_family == "qwen38" else "qwen3_5_text"
            ),
            "target_hidden_size": 5120,
            "args": {
                **dict(run_config["args"]),
                "target_model": target_model,
                "compression_ratio": COMPRESSION_RATIO,
                "max_encoder_tokens": max_encoder_tokens,
                "max_target_sequence_tokens": max_encoder_tokens,
            },
            "token_clock_serving_abi": {
                "name": "recursive_fixed_token_clock_v1",
                "persistent_prefix": "native_system_and_initial_user_never_compressed",
                "eviction_block_tokens": EVICTION_BLOCK_TOKENS,
                "raw_working_memory_tokens": RAW_WORKING_MEMORY_TOKENS,
                "update_0": "M0=C(P_conditioning,B0)",
                "update_k": "Mk=C(P_conditioning,M_prev,Bk)",
                "recovery": "P_plus_M_current_plus_latest_1024_raw_history_tokens",
                "slot_count": "ceil((previous_memory_slots+1024)/16)",
                "boundary_policy": "left_to_right_contiguous_token_blocks",
                "crop_policy": "never_crop_fail_closed",
            },
            "deployment_export": source_provenance,
        }
    )
    return config


def export(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("r2eAct FSDP export requires exactly two ranks")
    checkpoint = args.checkpoint.expanduser().resolve()
    run_config_path = (
        args.run_config.expanduser().resolve()
        if args.run_config is not None
        else checkpoint.parent / "run_config.json"
    )
    output_dir = args.output_dir.expanduser().resolve()
    trainer, run_config, rank_path = _validate_source(
        checkpoint, run_config_path, rank=rank, world_size=world_size
    )
    target_family = _source_target_family(trainer, run_config)
    deployment_target_model = _resolve_deployment_target_model(
        run_config,
        target_family=target_family,
        target_model_override=args.target_model_override,
    )

    staging_box: list[str | None] = [None]
    if rank == 0:
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"refusing to overwrite export destination: {output_dir}"
            )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = (
            output_dir.parent / f".{output_dir.name}.incomplete-{uuid.uuid4().hex}"
        )
        staging.mkdir(mode=0o700)
        (staging / "parts").mkdir(mode=0o700)
        staging_box[0] = str(staging)
    dist.broadcast_object_list(staging_box, src=0)
    staging = Path(str(staging_box[0]))
    try:
        payload = torch.load(rank_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != trainer.get("schema")
            or payload.get("format_version") != trainer.get("format_version")
            or payload.get("rank") != rank
            or payload.get("world_size") != world_size
            or not isinstance(payload.get("model"), Mapping)
        ):
            raise ValueError(f"invalid source rank payload: {rank_path}")
        local_part = _local_model_part(payload["model"])
        part_path = staging / "parts" / f"rank-{rank:05d}.pt"
        with part_path.open("xb") as handle:
            torch.save(local_part, handle)
            handle.flush()
            os.fsync(handle.fileno())
        del payload, local_part
        dist.barrier()

        if rank == 0:
            parts = [
                torch.load(
                    staging / "parts" / f"rank-{index:05d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for index in range(world_size)
            ]
            full_state = _assemble(parts)
            output_rms = float(run_config["compressor"]["output_embedding_rms"])
            _validate_full_state(full_state, output_rms)
            compressor_path = staging / "compressor.pt"
            with compressor_path.open("xb") as handle:
                torch.save(full_state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            rank_files = [
                checkpoint / f"rank-{index:05d}.pt" for index in range(world_size)
            ]
            config = _deployment_config(
                run_config,
                trainer,
                checkpoint,
                rank_files,
                target_family=target_family,
                deployment_target_model=deployment_target_model,
            )
            _write_json(staging / "config.json", config)
            manifest = {
                "schema": MANIFEST_SCHEMA,
                "world_size": world_size,
                "compressor_file": "compressor.pt",
                "compressor_sha256": _sha256(compressor_path),
                "config_file": "config.json",
                "config_sha256": _canonical_json_sha256(config),
                "rank_state_files": [],
                "frozen_target_saved": False,
                "export_schema": config["schema"],
            }
            _write_json(staging / "manifest.json", manifest)
            shutil.rmtree(staging / "parts")
            _fsync_directory(staging)
            os.rename(staging, output_dir)
            _fsync_directory(output_dir.parent)
            print(
                json.dumps(
                    {"status": "exported", "output_dir": str(output_dir), **manifest},
                    indent=2,
                )
            )
        dist.barrier()
    except BaseException:
        # Keep failed staging directories for diagnosis.  They are never valid
        # checkpoints because the final destination is published only by rename.
        raise
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(export())
