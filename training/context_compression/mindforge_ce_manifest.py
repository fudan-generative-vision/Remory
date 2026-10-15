"""Auditable action-CE manifests for the pinned MindForge GLM-5.2 data.

The manifest contains byte offsets, hashes, token counts, and causal event IDs;
it never copies the 679 MB source conversations.  Training can therefore seek
directly to a selected row while still failing closed if either the dataset,
adapter, tokenizer, selection policy, or selected event changes.

Gold actions use the same autoregressive contract as
``mindforge_glm52_data``: tokenize the generation prompt, then independently
encode the complete textual continuation.  Jointly tokenizing the completed
chat is deliberately *not* an invariant because the Qwen3.8 tokenizer can
merge differently across that newline seam.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from context_compression.mindforge_glm52_data import (
    IGNORE_INDEX,
    MINDFORGE_CONTRACT,
    MINDFORGE_DATASET_FILE,
    MINDFORGE_DATASET_FILE_BYTES,
    MINDFORGE_DATASET_FILE_SHA256,
    MINDFORGE_DATASET_REPO,
    MINDFORGE_DATASET_REVISION,
    MINDFORGE_DATASET_TRAJECTORIES,
    MINDFORGE_MAX_ROW_BYTES,
    MINDFORGE_MAX_SEQUENCE_TOKENS,
    QWEN38_ENABLE_THINKING,
    QWEN38_PRESERVE_THINKING,
    MindForgeEventTooLongError,
    MindForgeTrajectory,
    parse_mindforge_jsonl_line,
    render_mindforge_decision_event,
)
from context_compression.qwen36_recursive_data import recurrent_slot_counts

FORMAT_VERSION = 1
SCHEMA = "mindforge_glm52_qwen38_ce_decision_manifest_v1"
VIEW_SCHEMA = "mindforge_glm52_qwen38_ce_decision_manifest_view_v1"
COMPATIBILITY_REBIND_SCHEMA = (
    "mindforge_glm52_qwen38_ce_compatibility_checkpoint_rebind_v1"
)
COMPATIBILITY_REBIND_RECEIPT_SCHEMA = (
    "mindforge_glm52_qwen38_ce_compatibility_rebind_receipt_v1"
)
COMPATIBILITY_REBIND_RECEIPT_SCHEMA_V2 = (
    "mindforge_glm52_qwen38_ce_compatibility_rebind_receipt_v2"
)

BASELINE_PORTABLE_STAGE = "epoch_1_one_shot_only"
MINDFORGE_CE_PORTABLE_STAGE = "mindforge_frontier_ce_recursive_all_epochs"
MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA = "mindforge_glm52_qwen38_ce_portable_lineage_v1"
MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA_V2 = (
    "mindforge_glm52_qwen38_ce_portable_lineage_v2"
)
MINDFORGE_FRONTIER_CE_DATA_CONTRACT = "mindforge_glm52_frontier_actions_v1"

# These are release artifacts, not merely self-hashed manifests.  In
# particular, VIEW_SCHEMA is a derivation format and must never become a way to
# bless an arbitrary locally rewritten training cohort.  The 4K training view
# and the 8K evaluation manifest below select the same 512x4 + 16x2 actions;
# they differ only in the history-dependent geometry recorded on each event.
MINDFORGE_PRIMARY_H8192_ARTIFACT_SHA256 = (
    "14f4e4492e2add6ad02e6a5d475ddb70990294d40c841cd9f5ef94cdecf1f498"
)
MINDFORGE_PRIMARY_H8192_CONTRACT_SHA256 = (
    "d0b6f57bd1aa95052e54a35b09bf7319cce0b6496e834fae08cab61a583e1917"
)
MINDFORGE_FULL_VIEW_H4096_ARTIFACT_SHA256 = (
    "352df26d29e0ee085cb4ee7d5142a674f0dd4b757c2687196500b9f83e93d832"
)
MINDFORGE_FULL_VIEW_H4096_CONTRACT_SHA256 = (
    "e80f356308f663602e1ff55436ba3a1421b9d0a451d9fc57147d290fb1fcb2d3"
)
MINDFORGE_FULL_ACTION_COHORT_SHA256 = (
    "45ebaa8fb073ef937adadb0e8a0d4175dacfbee5ef4311e86995d84176918147"
)
MINDFORGE_RELEASE_BASELINE_RECEIPT_SHA256 = (
    "84864c81f5fce24ae7af9770deac282b9d3c99f771f8192825885a86310779e5"
)
MINDFORGE_RELEASE_BASELINE_COMPRESSOR_SHA256 = (
    "6137e1da636e6d1b1805d3018bcc4f7388eab25469569d6eca4007c91019b3c1"
)
MINDFORGE_RELEASE_TARGET_IDENTITY_SHA256 = (
    "e57c472cb3ac2c70190bd8aa645fb89f4038abaec603d979d765bb93366c3550"
)

DEFAULT_SELECTION_SEED = 42
DEFAULT_TRAIN_TRAJECTORIES = 512
DEFAULT_ACTIONS_PER_TRAIN_TRAJECTORY = 4
DEFAULT_VALIDATION_TRAJECTORIES = 16
DEFAULT_ACTIONS_PER_VALIDATION_TRAJECTORY = 2
DEFAULT_HISTORY_WINDOW_TOKENS = 8_192
DEFAULT_MAX_ACTION_TOKENS = 1_024
DEFAULT_MAX_STUDENT_SEQUENCE_TOKENS = 12_288
EVICTION_BLOCK_TOKENS = 1_024
FINAL_WORKING_MEMORY_TOKENS = 1_024
COMPRESSION_RATIO = 16

CLEAN_PILOT_EXCLUDED_ANOMALIES = frozenset(
    {
        "generation_output_limit_recovery",
        "duplicate_think_close_cleaned",
        "tool_arguments_outside_declared_schema",
    }
)
HARNESS_SUBMIT_LITERAL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
SUBMISSION_SENTINEL_REJECT_REASON = "target_contains_submission_sentinel"
SUBMISSION_SENTINEL_AUDIT_EVENTS = 238
SUBMISSION_SENTINEL_AUDIT_ROWS = 215
SUBMISSION_SENTINEL_AUDIT_RECORDS_SHA256 = (
    "b6ab0d912198facc16968b217d96afe0ad1d81bb25073fc7f63146c1a2992571"
)
TOKENIZER_RECEIPT_FILES = (
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


@dataclass(frozen=True)
class SourceLine:
    line_1based: int
    byte_offset: int
    byte_length: int
    raw_jsonl_line_sha256: str
    raw_json_payload_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a plain int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _as_ids(value: Any, *, label: str) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        isinstance(value, Sequence)
        and value
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        if len(value) != 1:
            raise ValueError(f"{label} unexpectedly returned a batch")
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} did not return token ids")
    result: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise TypeError(f"{label} returned an invalid token id")
        result.append(token)
    if not result:
        raise ValueError(f"{label} returned no token ids")
    return tuple(result)


def _with_contract_sha256(material: Mapping[str, Any]) -> dict[str, Any]:
    if "contract_sha256" in material:
        raise ValueError("unhashed manifest material contains contract_sha256")
    result = dict(material)
    result["contract_sha256"] = _canonical_sha256(result)
    return result


def _verify_contract_sha256(manifest: Mapping[str, Any]) -> str:
    observed = _sha256("contract_sha256", manifest.get("contract_sha256"))
    material = dict(manifest)
    material.pop("contract_sha256", None)
    expected = _canonical_sha256(material)
    if observed != expected:
        raise ValueError(
            f"manifest contract_sha256 mismatch: got {observed}, expected {expected}"
        )
    return observed


def _full_manifest_counts() -> dict[str, int]:
    return {
        "train_trajectories": DEFAULT_TRAIN_TRAJECTORIES,
        "train_actions": (
            DEFAULT_TRAIN_TRAJECTORIES * DEFAULT_ACTIONS_PER_TRAIN_TRAJECTORY
        ),
        "validation_trajectories": DEFAULT_VALIDATION_TRAJECTORIES,
        "validation_actions": (
            DEFAULT_VALIDATION_TRAJECTORIES * DEFAULT_ACTIONS_PER_VALIDATION_TRAJECTORY
        ),
    }


def _manifest_counts_receipt(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Return the four exact cohort counts from a full or compact receipt."""

    counts = manifest.get("counts")
    if isinstance(counts, Mapping):
        material = {
            name: _plain_int(f"manifest counts {name}", counts.get(name))
            for name in _full_manifest_counts()
        }
        if set(counts) != set(material):
            raise ValueError("training manifest counts receipt has foreign fields")
        return material
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("training manifest receipt has no policy")
    train_trajectories = _plain_int(
        "training manifest train trajectories", policy.get("train_trajectories")
    )
    train_actions_per_trajectory = _plain_int(
        "training manifest actions per train trajectory",
        policy.get("actions_per_train_trajectory"),
    )
    validation_trajectories = _plain_int(
        "training manifest validation trajectories",
        policy.get("validation_trajectories"),
    )
    validation_actions_per_trajectory = _plain_int(
        "training manifest actions per validation trajectory",
        policy.get("actions_per_validation_trajectory"),
    )
    material = {
        "train_trajectories": train_trajectories,
        "train_actions": train_trajectories * train_actions_per_trajectory,
        "validation_trajectories": validation_trajectories,
        "validation_actions": (
            validation_trajectories * validation_actions_per_trajectory
        ),
    }
    # Trainer receipts also carry these two redundant production counters.
    for field, expected in (
        ("train_trajectories", train_trajectories),
        ("selected_actions", material["train_actions"]),
    ):
        if field in manifest and manifest[field] != expected:
            raise ValueError(f"training manifest {field} counter changed")
    return material


def mindforge_action_cohort_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash ordered action identities and targets, excluding history geometry."""

    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("action cohort manifest has no rows")
    material: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"action cohort row {row_index} is not an object")
        events = row.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"action cohort row {row_index} has no events")
        material.append(
            {
                "split": row.get("split"),
                "training_order": row.get("training_order"),
                "raw_jsonl_line_sha256": row.get("raw_jsonl_line_sha256"),
                "source_trajectory_sha256": row.get("source_trajectory_sha256"),
                "normalized_trajectory_sha256": row.get("normalized_trajectory_sha256"),
                "events": [
                    {
                        name: event.get(name)
                        for name in (
                            "selected_event_instance_id",
                            "event_id",
                            "cutpoint_index",
                            "target_sha256",
                            "target_tokens",
                            "rendered_sha256",
                            "context_sha256",
                        )
                    }
                    for event in events
                    if isinstance(event, Mapping)
                ],
            }
        )
        if len(material[-1]["events"]) != len(events):
            raise ValueError(f"action cohort row {row_index} has a foreign event")
    return _canonical_sha256(material)


def _exact_manifest_lineage_receipt(
    manifest: Mapping[str, Any],
    *,
    artifact_sha256: str,
    expected_schema: str,
    expected_artifact_sha256: str,
    expected_contract_sha256: str,
    expected_history_window_tokens: int,
    expected_action_cohort_sha256: str,
) -> dict[str, Any]:
    """Validate one immutable release manifest and return its portable pin."""

    if (
        manifest.get("schema") != expected_schema
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("training manifest schema/version changed")
    artifact_sha256 = _sha256("training manifest artifact", artifact_sha256)
    contract_sha256 = _sha256(
        "training manifest contract", manifest.get("contract_sha256")
    )
    if (
        artifact_sha256 != expected_artifact_sha256
        or contract_sha256 != expected_contract_sha256
    ):
        raise ValueError("training manifest is not an exact pinned release artifact")
    policy = manifest.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("training manifest receipt has no policy")
    history_window_tokens = _plain_int(
        "training manifest history window", policy.get("history_window_tokens")
    )
    counts = _manifest_counts_receipt(manifest)
    if (
        history_window_tokens != expected_history_window_tokens
        or counts != _full_manifest_counts()
    ):
        raise ValueError("training manifest history/count geometry changed")
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or {
        "receipt_sha256": checkpoint.get("receipt_sha256"),
        "compressor_sha256": checkpoint.get("compressor_sha256"),
        "source_global_step": checkpoint.get("source_global_step"),
        "source_curriculum_stage": checkpoint.get("source_curriculum_stage"),
        "source_target_content_identity_sha256": checkpoint.get(
            "source_target_content_identity_sha256"
        ),
    } != {
        "receipt_sha256": MINDFORGE_RELEASE_BASELINE_RECEIPT_SHA256,
        "compressor_sha256": MINDFORGE_RELEASE_BASELINE_COMPRESSOR_SHA256,
        "source_global_step": 1_000,
        "source_curriculum_stage": BASELINE_PORTABLE_STAGE,
        "source_target_content_identity_sha256": (
            MINDFORGE_RELEASE_TARGET_IDENTITY_SHA256
        ),
    }:
        raise ValueError("training manifest release baseline checkpoint changed")
    if "rows" in manifest:
        observed_cohort = mindforge_action_cohort_sha256(manifest)
        if observed_cohort != expected_action_cohort_sha256:
            raise ValueError("training manifest action cohort changed")
    return {
        "schema": expected_schema,
        "artifact_sha256": artifact_sha256,
        "contract_sha256": contract_sha256,
        "history_window_tokens": history_window_tokens,
        "counts": counts,
        "action_cohort_sha256": expected_action_cohort_sha256,
    }


def exact_h4096_training_manifest_receipt(
    manifest: Mapping[str, Any], *, artifact_sha256: str
) -> dict[str, Any]:
    """Pin the sole VIEW artifact authorized as a 4K CE training source."""

    return _exact_manifest_lineage_receipt(
        manifest,
        artifact_sha256=artifact_sha256,
        expected_schema=VIEW_SCHEMA,
        expected_artifact_sha256=MINDFORGE_FULL_VIEW_H4096_ARTIFACT_SHA256,
        expected_contract_sha256=MINDFORGE_FULL_VIEW_H4096_CONTRACT_SHA256,
        expected_history_window_tokens=4_096,
        expected_action_cohort_sha256=MINDFORGE_FULL_ACTION_COHORT_SHA256,
    )


def exact_h8192_evaluation_manifest_receipt(
    manifest: Mapping[str, Any], *, artifact_sha256: str | None = None
) -> dict[str, Any]:
    """Pin the original 8K primary manifest used for compatibility scoring."""

    if artifact_sha256 is None:
        artifact_sha256 = _canonical_manifest_artifact_sha256(manifest)
    return _exact_manifest_lineage_receipt(
        manifest,
        artifact_sha256=artifact_sha256,
        expected_schema=SCHEMA,
        expected_artifact_sha256=MINDFORGE_PRIMARY_H8192_ARTIFACT_SHA256,
        expected_contract_sha256=MINDFORGE_PRIMARY_H8192_CONTRACT_SHA256,
        expected_history_window_tokens=DEFAULT_HISTORY_WINDOW_TOKENS,
        expected_action_cohort_sha256=MINDFORGE_FULL_ACTION_COHORT_SHA256,
    )


def _h4096_to_h8192_geometry_receipt() -> dict[str, Any]:
    """Describe every content field changed by the pinned history derivation."""

    return {
        "kind": "history_window_geometry_only",
        "training_history_window_tokens": 4_096,
        "evaluation_history_window_tokens": 8_192,
        "history_derived_fields": [
            "policy.history_window_tokens",
            "rows[].events[].history_window_tokens",
            "rows[].events[].compression_cutpoint_count",
            "rows[].events[].raw_oracle_sequence_tokens",
            "rows[].events[].maximum_student_sequence_tokens",
        ],
    }


def tokenizer_receipt(tokenizer: Any, tokenizer_path: str | Path) -> dict[str, Any]:
    """Hash the actual local tokenizer and Qwen chat-template identity."""

    root = Path(tokenizer_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"tokenizer path is not a directory: {root}")
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("tokenizer has no non-empty chat_template")
    files: dict[str, dict[str, Any]] = {}
    for name in TOKENIZER_RECEIPT_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"tokenizer receipt file is missing: {path}")
        files[name] = {"size_bytes": path.stat().st_size, "sha256": _file_sha256(path)}
    material = {
        "resolved_path": str(root),
        "target_content_sha256": _canonical_sha256(files),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "files": files,
        "tokenizer_class": (
            f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}"
        ),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
    }
    material["receipt_sha256"] = _canonical_sha256(material)
    return material


def _validate_mindforge_ce_portable_lineage(
    config: Mapping[str, Any], deployment: Mapping[str, Any]
) -> dict[str, Any]:
    """Cross-check every claim that distinguishes a CE portable export."""

    source_step = _plain_int(
        "checkpoint CE source_global_step",
        deployment.get("source_global_step"),
        minimum=1,
    )
    source_run_contract = _sha256(
        "checkpoint CE source_run_contract_sha256",
        deployment.get("source_run_contract_sha256"),
    )
    if (
        deployment.get("source_objective") != "ce"
        or config.get("run_contract_sha256") != source_run_contract
    ):
        raise ValueError("checkpoint CE objective/run contract lineage mismatch")
    expected_objective = {
        "name": "ce",
        "ce": "complete_native_frontier_assistant_action_only",
        "kl": None,
        "kl_loss_weight": None,
        "prompt_and_tool_observation_labels": -100,
    }
    if config.get("objective") != expected_objective:
        raise ValueError("checkpoint MindForge lineage requires objective=ce exactly")
    if config.get("curriculum") != {
        "all_epochs": (
            "recursive_stopgrad_cold_start_window; replay_only_states_over_"
            "sequence_cap; every_eligible_1024_eviction_directly_action_"
            "supervised; no_BPTT"
        )
    }:
        raise ValueError("checkpoint MindForge CE curriculum/stage changed")

    data_contract = config.get("data_contract")
    if (
        not isinstance(data_contract, Mapping)
        or data_contract.get("name") != MINDFORGE_FRONTIER_CE_DATA_CONTRACT
    ):
        raise ValueError("checkpoint has no MindForge CE data contract")
    manifest = data_contract.get("manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") not in {SCHEMA, VIEW_SCHEMA}
        or manifest.get("format_version") != FORMAT_VERSION
    ):
        raise ValueError("checkpoint CE lineage used an unsupported manifest")
    manifest_schema = str(manifest["schema"])
    manifest_artifact_sha = _sha256(
        "checkpoint training manifest artifact", manifest.get("artifact_sha256")
    )
    manifest_contract_sha = _sha256(
        "checkpoint training manifest contract", manifest.get("contract_sha256")
    )
    training_history_tokens = 4_096 if manifest_schema == VIEW_SCHEMA else 8_192
    policy = manifest.get("policy")
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
        "train_trajectories": DEFAULT_TRAIN_TRAJECTORIES,
        "actions_per_train_trajectory": DEFAULT_ACTIONS_PER_TRAIN_TRAJECTORY,
        "validation_trajectories": DEFAULT_VALIDATION_TRAJECTORIES,
        "actions_per_validation_trajectory": (
            DEFAULT_ACTIONS_PER_VALIDATION_TRAJECTORY
        ),
    }:
        raise ValueError("checkpoint CE lineage manifest has foreign history/counts")
    training_manifest: dict[str, Any] | None = None
    if manifest_schema == VIEW_SCHEMA:
        training_manifest = exact_h4096_training_manifest_receipt(
            manifest, artifact_sha256=manifest_artifact_sha
        )
        pilot_geometry = data_contract.get("pilot_geometry")
        history_window = data_contract.get("history_window")
        if (
            data_contract.get("pilot_profile") != "full-view-h4096-v1"
            or not isinstance(pilot_geometry, Mapping)
            or pilot_geometry.get("history_window_tokens") != 4_096
            or not isinstance(history_window, Mapping)
            or history_window.get("tokens") != 4_096
        ):
            raise ValueError("checkpoint CE 4K training geometry/profile changed")

    baseline = manifest.get("checkpoint")
    initialization = config.get("compressor_initialization")
    if not isinstance(baseline, Mapping) or not isinstance(initialization, Mapping):
        raise ValueError("checkpoint CE lineage has no baseline initialization")
    baseline_receipt_sha = _sha256(
        "checkpoint baseline receipt", baseline.get("receipt_sha256")
    )
    baseline_material = dict(baseline)
    baseline_material.pop("receipt_sha256", None)
    if baseline_receipt_sha != _canonical_sha256(baseline_material):
        raise ValueError("checkpoint baseline initialization self-hash mismatch")
    baseline_compressor_sha = _sha256(
        "checkpoint baseline compressor", baseline.get("compressor_sha256")
    )
    baseline_target_sha = _sha256(
        "checkpoint baseline target identity",
        baseline.get("source_target_content_identity_sha256"),
    )
    target_identity = config.get("target_content_identity")
    if (
        baseline.get("source_global_step") != 1_000
        or baseline.get("source_curriculum_stage") != BASELINE_PORTABLE_STAGE
        or baseline_target_sha
        != _canonical_sha256(baseline.get("source_target_content_identity"))
        or baseline.get("source_target_content_identity") != target_identity
        or initialization.get("checkpoint_receipt_sha256") != baseline_receipt_sha
        or initialization.get("artifact_sha256") != baseline_compressor_sha
        or initialization.get("source_global_step") != 1_000
        or initialization.get("source_curriculum_stage") != BASELINE_PORTABLE_STAGE
        or initialization.get("source_target_content_identity") != target_identity
        or initialization.get("source_target_content_identity_sha256")
        != baseline_target_sha
    ):
        raise ValueError("checkpoint CE baseline initialization lineage mismatch")

    implementation = config.get("implementation_identity")
    if not isinstance(implementation, Mapping):
        raise ValueError("checkpoint CE implementation identity is missing")
    local_files = implementation.get("local_files")
    required_local_files = {
        "trainer",
        "data",
        "compressor",
        "loss",
        "parity_validator",
        "target_loader",
        "mindforge_data",
        "mindforge_manifest",
    }
    if not isinstance(local_files, Mapping) or not required_local_files.issubset(
        local_files
    ):
        raise ValueError("checkpoint CE implementation identity is incomplete")
    for name, entry in local_files.items():
        if (
            not isinstance(name, str)
            or not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
        ):
            raise ValueError("checkpoint CE implementation file receipt is malformed")
        _sha256(f"checkpoint implementation file {name}", entry.get("sha256"))
    adapter = manifest.get("adapter")
    if (
        not isinstance(adapter, Mapping)
        or manifest.get("adapter_source_sha256")
        != local_files["mindforge_data"]["sha256"]
        or adapter.get("implementation_sha256")
        != local_files["mindforge_data"]["sha256"]
    ):
        raise ValueError("checkpoint CE adapter/implementation identity mismatch")
    implementation_sha = _canonical_sha256(implementation)

    lineage = deployment.get("source_lineage")
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
        expected_lineage = {
            "schema": MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA_V2,
            **common_lineage,
            "training_history_window_tokens": 4_096,
            "training_manifest": training_manifest,
        }
    else:
        expected_lineage = {
            "schema": MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA,
            **common_lineage,
            "primary_manifest": {
                "artifact_sha256": manifest_artifact_sha,
                "contract_sha256": manifest_contract_sha,
            },
        }
    if lineage != expected_lineage:
        raise ValueError("checkpoint MindForge CE portable lineage receipt mismatch")
    return {
        **json.loads(json.dumps(expected_lineage, ensure_ascii=False, allow_nan=False)),
        "source_global_step": source_step,
        "source_run_contract_sha256": source_run_contract,
    }


def checkpoint_receipt(checkpoint_path: str | Path) -> dict[str, Any]:
    """Verify and pin a baseline or MindForge-CE portable checkpoint.

    Baseline step-1000 receipts intentionally retain their original v1 field
    set and hash.  A CE export receives a distinct, strictly cross-checked
    lineage extension instead of being mislabeled as the KL one-shot stage.
    """

    root = Path(checkpoint_path).expanduser().resolve()
    manifest_path = root / "manifest.json"
    config_path = root / "config.json"
    try:
        exported = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid checkpoint receipt at {root}") from error
    if not isinstance(exported, Mapping) or not isinstance(config, Mapping):
        raise ValueError("checkpoint manifest/config must be JSON objects")
    expected_export_abi = {
        "schema": "context_compression_ddp_v1",
        "world_size": 2,
        "config_file": "config.json",
        "frozen_target_saved": False,
        "rank_state_files": [],
        "export_schema": "qwen38_27b_r2eact_token_clock_serving_v1",
    }
    for key, expected in expected_export_abi.items():
        if exported.get(key) != expected:
            raise ValueError(
                f"checkpoint export ABI mismatch for {key}: "
                f"{exported.get(key)!r} != {expected!r}"
            )
    if exported.get("compressor_file") != "compressor.pt":
        raise ValueError("checkpoint manifest must name compressor.pt")
    if config.get("schema") != expected_export_abi["export_schema"]:
        raise ValueError("checkpoint config schema differs from export_schema")
    if config.get("target_family") != "qwen38":
        raise ValueError("checkpoint target_family must be qwen38")
    compressor_path = root / "compressor.pt"
    if not compressor_path.is_file():
        raise ValueError(f"checkpoint compressor is missing: {compressor_path}")
    config_file_sha = _file_sha256(config_path)
    config_sha = _canonical_sha256(config)
    compressor_sha = _file_sha256(compressor_path)
    if exported.get("config_sha256") != config_sha:
        raise ValueError("checkpoint config checksum mismatch")
    if exported.get("compressor_sha256") != compressor_sha:
        raise ValueError("checkpoint compressor checksum mismatch")
    deployment = config.get("deployment_export", {})
    if not isinstance(deployment, Mapping):
        raise ValueError("checkpoint deployment_export must be an object")
    source_stage = deployment.get("source_curriculum_stage")
    if source_stage == BASELINE_PORTABLE_STAGE:
        if deployment.get("source_global_step") != 1_000:
            raise ValueError("checkpoint must be the frozen step-1000 initialization")
        if "source_objective" in deployment or "source_lineage" in deployment:
            raise ValueError("baseline checkpoint must not claim a CE lineage")
        source_objective = config.get("objective")
        source_data_contract = config.get("data_contract")
        if (
            isinstance(source_objective, Mapping)
            and source_objective.get("name") in {"ce", "ce_kl"}
        ) or (
            isinstance(source_data_contract, Mapping)
            and source_data_contract.get("name") == MINDFORGE_FRONTIER_CE_DATA_CONTRACT
        ):
            raise ValueError("baseline checkpoint is mislabeled MindForge CE state")
        ce_lineage: dict[str, Any] | None = None
    elif source_stage == MINDFORGE_CE_PORTABLE_STAGE:
        ce_lineage = _validate_mindforge_ce_portable_lineage(config, deployment)
    else:
        raise ValueError("checkpoint source curriculum stage is unsupported")
    source_target_identity = config.get("target_content_identity")
    identity_keys = {
        "resolved_path",
        "weights",
        "metadata",
        "weight_content_sha256",
        "metadata_content_sha256",
        "target_content_sha256",
    }
    if (
        not isinstance(source_target_identity, Mapping)
        or set(source_target_identity) != identity_keys
    ):
        raise ValueError("checkpoint has no exact source target-content identity")
    for list_name, digest_name in (
        ("weights", "weight_content_sha256"),
        ("metadata", "metadata_content_sha256"),
    ):
        entries = source_target_identity[list_name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"checkpoint target identity {list_name} is empty")
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"name", "sha256"}:
                raise ValueError(f"checkpoint target identity {list_name} is malformed")
            if not isinstance(entry["name"], str) or not entry["name"]:
                raise ValueError("checkpoint target identity contains an empty name")
            _sha256("checkpoint target file digest", entry["sha256"])
        if _canonical_sha256(entries) != source_target_identity[digest_name]:
            raise ValueError(f"checkpoint target identity {digest_name} mismatch")
    identity_material = dict(source_target_identity)
    declared_target_sha = _sha256(
        "checkpoint target_content_sha256",
        identity_material.pop("target_content_sha256"),
    )
    if _canonical_sha256(identity_material) != declared_target_sha:
        raise ValueError("checkpoint target_content_sha256 mismatch")
    source_target_identity_copy = json.loads(
        json.dumps(source_target_identity, ensure_ascii=False, allow_nan=False)
    )
    material = {
        "resolved_path": str(root),
        "manifest_sha256": _file_sha256(manifest_path),
        "config_sha256": config_sha,
        "config_file_sha256": config_file_sha,
        "compressor_sha256": compressor_sha,
        "compressor_size_bytes": compressor_path.stat().st_size,
        "export_schema": exported.get("export_schema"),
        "source_global_step": (
            deployment.get("source_global_step")
            if isinstance(deployment, Mapping)
            else None
        ),
        "source_curriculum_stage": (
            deployment.get("source_curriculum_stage")
            if isinstance(deployment, Mapping)
            else None
        ),
        "source_target_content_identity": source_target_identity_copy,
        "source_target_content_identity_sha256": _canonical_sha256(
            source_target_identity_copy
        ),
    }
    if ce_lineage is not None:
        material.update(
            {
                "source_objective": "ce",
                "mindforge_ce_lineage": ce_lineage,
            }
        )
    material["receipt_sha256"] = _canonical_sha256(material)
    return material


def index_verified_source(
    source_path: str | Path,
    *,
    expected_size_bytes: int,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[SourceLine, ...]:
    """Verify the complete JSONL and return immutable random-access pointers."""

    source = Path(source_path)
    expected_size_bytes = _plain_int(
        "expected_size_bytes", expected_size_bytes, minimum=1
    )
    expected_sha256 = _sha256("expected_sha256", expected_sha256)
    expected_rows = _plain_int("expected_rows", expected_rows, minimum=1)
    if not source.is_file() or source.stat().st_size != expected_size_bytes:
        observed = source.stat().st_size if source.exists() else None
        raise ValueError(
            f"source size mismatch: got {observed}, expected {expected_size_bytes}"
        )
    whole_digest = hashlib.sha256()
    rows: list[SourceLine] = []
    seen_line_hashes: set[str] = set()
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            raw_line = handle.readline(MINDFORGE_MAX_ROW_BYTES + 1)
            if not raw_line:
                break
            whole_digest.update(raw_line)
            line_number = len(rows) + 1
            if len(raw_line) > MINDFORGE_MAX_ROW_BYTES:
                raise ValueError(f"source line {line_number} exceeds its byte bound")
            if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
                raise ValueError(f"source line {line_number} has invalid framing")
            line_sha = hashlib.sha256(raw_line).hexdigest()
            if line_sha in seen_line_hashes:
                raise ValueError(f"duplicate raw source line at {line_number}")
            seen_line_hashes.add(line_sha)
            rows.append(
                SourceLine(
                    line_1based=line_number,
                    byte_offset=offset,
                    byte_length=len(raw_line),
                    raw_jsonl_line_sha256=line_sha,
                    raw_json_payload_sha256=hashlib.sha256(raw_line[:-1]).hexdigest(),
                )
            )
    observed_sha = whole_digest.hexdigest()
    if observed_sha != expected_sha256:
        raise ValueError(
            f"source SHA-256 mismatch: got {observed_sha}, expected {expected_sha256}"
        )
    if len(rows) != expected_rows:
        raise ValueError(
            f"source row count mismatch: got {len(rows)}, expected {expected_rows}"
        )
    return tuple(rows)


def _read_source_line(source_path: Path, source_line: SourceLine) -> bytes:
    with source_path.open("rb") as handle:
        handle.seek(source_line.byte_offset)
        raw_line = handle.read(source_line.byte_length)
    if len(raw_line) != source_line.byte_length:
        raise ValueError("source row seek returned the wrong byte length")
    if hashlib.sha256(raw_line).hexdigest() != source_line.raw_jsonl_line_sha256:
        raise ValueError("source row checksum changed after indexing")
    if hashlib.sha256(raw_line[:-1]).hexdigest() != source_line.raw_json_payload_sha256:
        raise ValueError("source row payload checksum changed after indexing")
    return raw_line


def _chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]],
    tokenize: bool,
    add_generation_prompt: bool,
) -> Any:
    return tokenizer.apply_chat_template(
        list(messages),
        tools=list(tools),
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=QWEN38_ENABLE_THINKING,
        preserve_thinking=QWEN38_PRESERVE_THINKING,
        return_dict=False,
    )


def _event_record(
    trajectory: MindForgeTrajectory,
    event_index: int,
    tokenizer: Any,
    *,
    history_window_tokens: int,
    max_action_tokens: int,
    max_student_sequence_tokens: int,
) -> tuple[dict[str, Any] | None, str | None]:
    event = trajectory.events[event_index]
    if set(event.anomaly_flags).intersection(CLEAN_PILOT_EXCLUDED_ANOMALIES):
        return None, "excluded_anomaly"
    # The adapter removes the eight exact terminal rollout controls.  A
    # separate audit found this literal in otherwise non-terminal content and
    # tool arguments; those are also rollout-contaminated for the clean pilot.
    if HARNESS_SUBMIT_LITERAL in json.dumps(
        event.target_message,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ):
        return None, SUBMISSION_SENTINEL_REJECT_REASON
    try:
        rendered = render_mindforge_decision_event(
            event, tokenizer, max_sequence_tokens=MINDFORGE_MAX_SEQUENCE_TOKENS
        )
    except MindForgeEventTooLongError:
        return None, "raw_prompt_over_native_context"
    if rendered.target_tokens > max_action_tokens:
        return None, "action_over_max_tokens"
    if (
        rendered.labels[: rendered.target_start]
        != (IGNORE_INDEX,) * rendered.target_start
    ):
        raise RuntimeError("adapter leaked CE labels into the generation prompt")
    if (
        rendered.labels[rendered.target_start :]
        != rendered.input_ids[rendered.target_start :]
    ):
        raise RuntimeError("adapter action labels differ from the complete action")

    prefix_ids = _as_ids(
        _chat_template(
            tokenizer,
            event.context_messages[:2],
            tools=event.tools,
            tokenize=True,
            add_generation_prompt=False,
        ),
        label="Qwen3.8 raw prefix",
    )
    prompt_ids = rendered.input_ids[: rendered.target_start]
    if prompt_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Qwen3.8 generation prompt does not preserve raw-prefix ids")
    history_tokens = rendered.prompt_tokens - len(prefix_ids)
    if history_tokens < history_window_tokens:
        return None, "history_shorter_than_window"
    compression_source_tokens = history_window_tokens - FINAL_WORKING_MEMORY_TOKENS
    if compression_source_tokens <= 0 or (
        compression_source_tokens % EVICTION_BLOCK_TOKENS
    ):
        raise ValueError(
            "history window must contain whole eviction blocks plus final working "
            "memory"
        )
    compression_cutpoint_count = compression_source_tokens // EVICTION_BLOCK_TOKENS
    slot_counts = recurrent_slot_counts(
        compression_cutpoint_count,
        block_tokens=EVICTION_BLOCK_TOKENS,
        compression_ratio=COMPRESSION_RATIO,
    )
    if not slot_counts:
        raise ValueError("selected action has no eligible compression cutpoint")
    raw_oracle_tokens = len(prefix_ids) + history_window_tokens + rendered.target_tokens
    maximum_student_tokens = (
        len(prefix_ids)
        + slot_counts[0]
        + history_window_tokens
        - EVICTION_BLOCK_TOKENS
        + rendered.target_tokens
    )
    final_memory_student_tokens = (
        len(prefix_ids)
        + slot_counts[-1]
        + FINAL_WORKING_MEMORY_TOKENS
        + rendered.target_tokens
    )
    no_old_history_tokens = (
        len(prefix_ids) + FINAL_WORKING_MEMORY_TOKENS + rendered.target_tokens
    )
    if raw_oracle_tokens > max_student_sequence_tokens:
        return None, "matched_raw_oracle_over_max_sequence"
    if maximum_student_tokens > max_student_sequence_tokens:
        return None, "compression_cutpoint_over_max_sequence"

    target_calls = event.target_message.get("tool_calls", ())
    return (
        {
            "event_id": event.event_id,
            "cutpoint_index": event.cutpoint_index,
            "trajectory_decision_count": event.weight.cutpoint_count,
            "context_sha256": event.context_sha256,
            "target_sha256": event.target_sha256,
            "rendered_sha256": rendered.rendered_sha256,
            "prompt_tokens": rendered.prompt_tokens,
            "target_tokens": rendered.target_tokens,
            "raw_prefix_tokens": len(prefix_ids),
            "history_tokens": history_tokens,
            "history_window_tokens": history_window_tokens,
            "compression_cutpoint_count": compression_cutpoint_count,
            "final_memory_tokens": slot_counts[-1],
            "raw_oracle_sequence_tokens": raw_oracle_tokens,
            "maximum_student_sequence_tokens": maximum_student_tokens,
            "final_memory_student_sequence_tokens": final_memory_student_tokens,
            "no_old_history_sequence_tokens": no_old_history_tokens,
            "anomaly_flags": list(event.anomaly_flags),
            "context_submit_reminder_count": event.context_submit_reminder_count,
            "generation_output_limit_feedback_count": (
                event.generation_output_limit_feedback_count
            ),
            "target_has_tool_calls": bool(target_calls),
            "target_tool_call_count": len(target_calls),
            "selected_event_weight": 0.0,
        },
        None,
    )


def _selection_key(seed: int, split: str, line_sha256: str) -> str:
    return hashlib.sha256(
        f"{SCHEMA}|{seed}|{split}|{line_sha256}".encode("ascii")
    ).hexdigest()


def _select_split(
    source_path: Path,
    source_lines: Sequence[SourceLine],
    tokenizer: Any,
    *,
    split: str,
    seed: int,
    trajectory_count: int,
    actions_per_trajectory: int,
    excluded_line_hashes: set[str],
    excluded_event_ids: set[str],
    history_window_tokens: int,
    max_action_tokens: int,
    max_student_sequence_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates = sorted(
        source_lines,
        key=lambda row: _selection_key(seed, split, row.raw_jsonl_line_sha256),
    )
    selected: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}
    seen_event_instances: set[str] = set()
    seen_event_ids: set[str] = set(excluded_event_ids)
    for source_line in candidates:
        if source_line.raw_jsonl_line_sha256 in excluded_line_hashes:
            continue
        raw_line = _read_source_line(source_path, source_line)
        trajectory = parse_mindforge_jsonl_line(
            raw_line, line_number=source_line.line_1based
        )
        if trajectory.raw_jsonl_line_sha256 != source_line.raw_jsonl_line_sha256:
            raise RuntimeError("adapter raw-line identity differs from source index")
        if trajectory.raw_json_payload_sha256 != source_line.raw_json_payload_sha256:
            raise RuntimeError("adapter raw-payload identity differs from source index")

        records: list[dict[str, Any]] = []
        candidate_event_ids: set[str] = set()
        for event_index in range(len(trajectory.events) - 1, -1, -1):
            record, rejection = _event_record(
                trajectory,
                event_index,
                tokenizer,
                history_window_tokens=history_window_tokens,
                max_action_tokens=max_action_tokens,
                max_student_sequence_tokens=max_student_sequence_tokens,
            )
            if record is None:
                rejection_counts[rejection or "unknown"] = (
                    rejection_counts.get(rejection or "unknown", 0) + 1
                )
                continue
            if (
                str(record["event_id"]) in seen_event_ids
                or str(record["event_id"]) in candidate_event_ids
            ):
                rejection_counts["duplicate_causal_event_id"] = (
                    rejection_counts.get("duplicate_causal_event_id", 0) + 1
                )
                continue
            instance_id = hashlib.sha256(
                (
                    source_line.raw_jsonl_line_sha256 + ":" + str(record["event_id"])
                ).encode("ascii")
            ).hexdigest()
            if instance_id in seen_event_instances:
                raise ValueError(f"duplicate selected event instance: {instance_id}")
            record["selected_event_instance_id"] = instance_id
            records.append(record)
            candidate_event_ids.add(str(record["event_id"]))
            if len(records) == actions_per_trajectory:
                break
        if len(records) != actions_per_trajectory:
            rejection_counts["trajectory_has_too_few_eligible_actions"] = (
                rejection_counts.get("trajectory_has_too_few_eligible_actions", 0) + 1
            )
            continue
        records.sort(key=lambda record: int(record["cutpoint_index"]))
        for record in records:
            record["selected_event_weight"] = 1.0 / len(records)
            seen_event_instances.add(str(record["selected_event_instance_id"]))
            seen_event_ids.add(str(record["event_id"]))
        selected.append(
            {
                "split": split,
                "training_order": len(selected),
                "selection_key": _selection_key(
                    seed, split, source_line.raw_jsonl_line_sha256
                ),
                "line_1based": source_line.line_1based,
                "byte_offset": source_line.byte_offset,
                "byte_length": source_line.byte_length,
                "raw_jsonl_line_sha256": source_line.raw_jsonl_line_sha256,
                "raw_json_payload_sha256": source_line.raw_json_payload_sha256,
                "source_trajectory_sha256": trajectory.source_trajectory_sha256,
                "normalized_trajectory_sha256": (
                    trajectory.normalized_trajectory_sha256
                ),
                "trajectory_anomaly_flags": list(trajectory.anomaly_flags),
                "dropped_empty_terminal_assistant": (
                    trajectory.dropped_empty_terminal_assistant
                ),
                "events": records,
            }
        )
        if len(selected) == trajectory_count:
            break
    if len(selected) != trajectory_count:
        raise ValueError(
            f"could select only {len(selected)}/{trajectory_count} {split} trajectories"
        )
    return selected, dict(sorted(rejection_counts.items()))


def _policy(
    *,
    history_window_tokens: int,
    max_action_tokens: int,
    max_student_sequence_tokens: int,
    selection_seed: int,
    train_trajectories: int,
    actions_per_train_trajectory: int,
    validation_trajectories: int,
    actions_per_validation_trajectory: int,
) -> dict[str, Any]:
    return {
        "history_window_tokens": history_window_tokens,
        "max_action_tokens": max_action_tokens,
        "max_student_sequence_tokens": max_student_sequence_tokens,
        "selection_seed": selection_seed,
        "eviction_block_tokens": EVICTION_BLOCK_TOKENS,
        "final_working_memory_tokens": FINAL_WORKING_MEMORY_TOKENS,
        "compression_ratio": COMPRESSION_RATIO,
        "train_trajectories": train_trajectories,
        "actions_per_train_trajectory": actions_per_train_trajectory,
        "validation_trajectories": validation_trajectories,
        "actions_per_validation_trajectory": actions_per_validation_trajectory,
        "selection": "hash_rank_per_split_then_last_clean_complete_actions_v1",
        "clean_pilot_excluded_anomalies": sorted(CLEAN_PILOT_EXCLUDED_ANOMALIES),
        "action_truncation": "forbidden",
        "action_weighting": "unit_trajectory_then_uniform_selected_action",
        "compression_cutpoint_weighting": "uniform_1_over_K_within_action",
        "split_is_task_disjoint": False,
        "split_limitation": (
            "the public messages-only release has no trajectory-to-instance mapping"
        ),
        "history_before_window_excluded": True,
        "initial_memory": None,
        "long_trajectory_position_used_for_action_selection_only": True,
        "target_harness_submit_literal_excluded": HARNESS_SUBMIT_LITERAL,
        "target_submission_sentinel_reject_reason": (SUBMISSION_SENTINEL_REJECT_REASON),
        "target_submission_sentinel_audit_events": (SUBMISSION_SENTINEL_AUDIT_EVENTS),
        "target_submission_sentinel_audit_rows": SUBMISSION_SENTINEL_AUDIT_ROWS,
        "target_submission_sentinel_audit_records_sha256": (
            SUBMISSION_SENTINEL_AUDIT_RECORDS_SHA256
        ),
    }


def build_ce_manifest(
    source_path: str | Path,
    tokenizer: Any,
    *,
    tokenizer_identity: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    expected_size_bytes: int = MINDFORGE_DATASET_FILE_BYTES,
    expected_sha256: str = MINDFORGE_DATASET_FILE_SHA256,
    expected_rows: int = MINDFORGE_DATASET_TRAJECTORIES,
    selection_seed: int = DEFAULT_SELECTION_SEED,
    train_trajectories: int = DEFAULT_TRAIN_TRAJECTORIES,
    actions_per_train_trajectory: int = DEFAULT_ACTIONS_PER_TRAIN_TRAJECTORY,
    validation_trajectories: int = DEFAULT_VALIDATION_TRAJECTORIES,
    actions_per_validation_trajectory: int = (
        DEFAULT_ACTIONS_PER_VALIDATION_TRAJECTORY
    ),
    history_window_tokens: int = DEFAULT_HISTORY_WINDOW_TOKENS,
    max_action_tokens: int = DEFAULT_MAX_ACTION_TOKENS,
    max_student_sequence_tokens: int = DEFAULT_MAX_STUDENT_SEQUENCE_TOKENS,
) -> dict[str, Any]:
    """Build the deterministic full train + fixed validation manifest."""

    for name, value in (
        ("selection_seed", selection_seed),
        ("train_trajectories", train_trajectories),
        ("actions_per_train_trajectory", actions_per_train_trajectory),
        ("validation_trajectories", validation_trajectories),
        ("actions_per_validation_trajectory", actions_per_validation_trajectory),
        ("history_window_tokens", history_window_tokens),
        ("max_action_tokens", max_action_tokens),
        ("max_student_sequence_tokens", max_student_sequence_tokens),
    ):
        _plain_int(name, value, minimum=1 if name != "selection_seed" else 0)
    if history_window_tokens < 2 * EVICTION_BLOCK_TOKENS:
        raise ValueError("history_window_tokens must create a compression event")
    if (history_window_tokens - FINAL_WORKING_MEMORY_TOKENS) % EVICTION_BLOCK_TOKENS:
        raise ValueError("history window is not aligned to 1024-token evictions")
    if not isinstance(tokenizer_identity, Mapping) or not isinstance(
        checkpoint_identity, Mapping
    ):
        raise TypeError("tokenizer/checkpoint identities must be mappings")
    source = Path(source_path).expanduser().resolve()
    source_lines = index_verified_source(
        source,
        expected_size_bytes=expected_size_bytes,
        expected_sha256=expected_sha256,
        expected_rows=expected_rows,
    )
    validation, validation_rejections = _select_split(
        source,
        source_lines,
        tokenizer,
        split="validation",
        seed=selection_seed,
        trajectory_count=validation_trajectories,
        actions_per_trajectory=actions_per_validation_trajectory,
        excluded_line_hashes=set(),
        excluded_event_ids=set(),
        history_window_tokens=history_window_tokens,
        max_action_tokens=max_action_tokens,
        max_student_sequence_tokens=max_student_sequence_tokens,
    )
    validation_hashes = {str(row["raw_jsonl_line_sha256"]) for row in validation}
    validation_event_ids = {
        str(event["event_id"]) for row in validation for event in row["events"]
    }
    train, train_rejections = _select_split(
        source,
        source_lines,
        tokenizer,
        split="train",
        seed=selection_seed,
        trajectory_count=train_trajectories,
        actions_per_trajectory=actions_per_train_trajectory,
        excluded_line_hashes=validation_hashes,
        excluded_event_ids=validation_event_ids,
        history_window_tokens=history_window_tokens,
        max_action_tokens=max_action_tokens,
        max_student_sequence_tokens=max_student_sequence_tokens,
    )
    rows = [*train, *validation]
    event_instances = [
        str(event["selected_event_instance_id"])
        for row in rows
        for event in row["events"]
    ]
    if len(event_instances) != len(set(event_instances)):
        raise ValueError("selected event instance IDs are not globally unique")
    event_ids = [str(event["event_id"]) for row in rows for event in row["events"]]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("selected causal event IDs are not globally unique")
    adapter_path = Path(__file__).with_name("mindforge_glm52_data.py")
    policy = _policy(
        history_window_tokens=history_window_tokens,
        max_action_tokens=max_action_tokens,
        max_student_sequence_tokens=max_student_sequence_tokens,
        selection_seed=selection_seed,
        train_trajectories=train_trajectories,
        actions_per_train_trajectory=actions_per_train_trajectory,
        validation_trajectories=validation_trajectories,
        actions_per_validation_trajectory=actions_per_validation_trajectory,
    )
    material = {
        "schema": SCHEMA,
        "format_version": FORMAT_VERSION,
        "source": {
            "dataset_repo": MINDFORGE_DATASET_REPO,
            "dataset_revision": MINDFORGE_DATASET_REVISION,
            "file": MINDFORGE_DATASET_FILE,
            "path_basename": source.name,
            "size_bytes": expected_size_bytes,
            "sha256": expected_sha256,
            "row_count": expected_rows,
        },
        "tokenizer": dict(tokenizer_identity),
        "checkpoint": dict(checkpoint_identity),
        "adapter": {
            "contract": MINDFORGE_CONTRACT,
            "implementation_sha256": _file_sha256(adapter_path),
        },
        "policy": policy,
        "rows": rows,
        "counts": {
            "train_trajectories": len(train),
            "train_actions": sum(len(row["events"]) for row in train),
            "validation_trajectories": len(validation),
            "validation_actions": sum(len(row["events"]) for row in validation),
        },
        "selection_rejections": {
            "train": train_rejections,
            "validation": validation_rejections,
        },
        "gold_action_contract": (
            "generation_prompt_tokenize_plus_independent_complete_continuation_encode"
        ),
        "joint_completed_chat_token_prefix_required": False,
    }
    return _with_contract_sha256(material)


def _require_keys(
    value: Any, required: set[str], *, label: str, allow_extra: bool = False
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if not allow_extra and set(value) != required:
        raise ValueError(
            f"{label} has unexpected fields: {sorted(set(value)-required)}"
        )
    return value


def _validate_ce_checkpoint_lineage_receipt(
    checkpoint_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    raw_lineage = checkpoint_identity.get("mindforge_ce_lineage")
    if not isinstance(raw_lineage, Mapping):
        raise ValueError("manifest.checkpoint.mindforge_ce_lineage must be an object")
    lineage_schema = raw_lineage.get("schema")
    lineage_specific_fields: set[str]
    if lineage_schema == MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA:
        lineage_specific_fields = {"primary_manifest"}
    elif lineage_schema == MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA_V2:
        lineage_specific_fields = {
            "training_history_window_tokens",
            "training_manifest",
        }
    else:
        raise ValueError("manifest checkpoint CE lineage schema changed")
    lineage = _require_keys(
        raw_lineage,
        {
            "schema",
            "objective",
            "baseline_initialization",
            "implementation_identity_sha256",
            "source_global_step",
            "source_run_contract_sha256",
        }
        | lineage_specific_fields,
        label="manifest.checkpoint.mindforge_ce_lineage",
    )
    if (
        lineage["objective"] != "ce"
        or checkpoint_identity.get("source_objective") != "ce"
        or lineage["source_global_step"]
        != checkpoint_identity.get("source_global_step")
    ):
        raise ValueError("manifest checkpoint CE lineage schema/objective/step changed")
    _plain_int(
        "manifest.checkpoint.mindforge_ce_lineage.source_global_step",
        lineage["source_global_step"],
        minimum=1,
    )
    _sha256(
        "manifest.checkpoint.mindforge_ce_lineage.source_run_contract_sha256",
        lineage["source_run_contract_sha256"],
    )
    _sha256(
        "manifest.checkpoint.mindforge_ce_lineage.implementation_identity_sha256",
        lineage["implementation_identity_sha256"],
    )
    if lineage_schema == MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA:
        primary = _require_keys(
            lineage["primary_manifest"],
            {"artifact_sha256", "contract_sha256"},
            label="manifest.checkpoint.mindforge_ce_lineage.primary_manifest",
        )
        _sha256(
            "manifest checkpoint primary manifest artifact",
            primary["artifact_sha256"],
        )
        _sha256(
            "manifest checkpoint primary manifest contract",
            primary["contract_sha256"],
        )
    else:
        if lineage["training_history_window_tokens"] != 4_096:
            raise ValueError("manifest checkpoint CE training horizon changed")
        training = _require_keys(
            lineage["training_manifest"],
            {
                "schema",
                "artifact_sha256",
                "contract_sha256",
                "history_window_tokens",
                "counts",
                "action_cohort_sha256",
            },
            label="manifest.checkpoint.mindforge_ce_lineage.training_manifest",
        )
        expected_training = {
            "schema": VIEW_SCHEMA,
            "artifact_sha256": MINDFORGE_FULL_VIEW_H4096_ARTIFACT_SHA256,
            "contract_sha256": MINDFORGE_FULL_VIEW_H4096_CONTRACT_SHA256,
            "history_window_tokens": 4_096,
            "counts": _full_manifest_counts(),
            "action_cohort_sha256": MINDFORGE_FULL_ACTION_COHORT_SHA256,
        }
        if training != expected_training:
            raise ValueError("manifest checkpoint CE training manifest pin changed")
    baseline = _require_keys(
        lineage["baseline_initialization"],
        {
            "checkpoint_receipt_sha256",
            "compressor_sha256",
            "source_global_step",
            "source_curriculum_stage",
            "source_target_content_identity_sha256",
        },
        label="manifest.checkpoint.mindforge_ce_lineage.baseline_initialization",
    )
    for name in (
        "checkpoint_receipt_sha256",
        "compressor_sha256",
        "source_target_content_identity_sha256",
    ):
        _sha256(f"manifest checkpoint baseline {name}", baseline[name])
    if (
        baseline["source_global_step"] != 1_000
        or baseline["source_curriculum_stage"] != BASELINE_PORTABLE_STAGE
        or baseline["source_target_content_identity_sha256"]
        != checkpoint_identity.get("source_target_content_identity_sha256")
    ):
        raise ValueError("manifest checkpoint CE baseline lineage changed")
    return lineage


def validate_ce_manifest(
    manifest: Mapping[str, Any], *, require_pinned_source: bool = True
) -> dict[str, Any]:
    """Strictly validate the v1 ABI, self hash, policy, order, and exact set."""

    manifest_schema = manifest.get("schema") if isinstance(manifest, Mapping) else None
    required_top = {
        "schema",
        "format_version",
        "source",
        "tokenizer",
        "checkpoint",
        "adapter",
        "policy",
        "rows",
        "counts",
        "selection_rejections",
        "gold_action_contract",
        "joint_completed_chat_token_prefix_required",
        "contract_sha256",
    }
    if manifest_schema == COMPATIBILITY_REBIND_SCHEMA:
        required_top.add("checkpoint_rebind")
    root = _require_keys(manifest, required_top, label="manifest")
    if (
        root["schema"]
        not in {
            SCHEMA,
            VIEW_SCHEMA,
            COMPATIBILITY_REBIND_SCHEMA,
        }
        or root["format_version"] != 1
    ):
        raise ValueError("unsupported CE manifest schema/version")
    _verify_contract_sha256(root)
    source = _require_keys(
        root["source"],
        {
            "dataset_repo",
            "dataset_revision",
            "file",
            "path_basename",
            "size_bytes",
            "sha256",
            "row_count",
        },
        label="manifest.source",
    )
    _sha256("manifest.source.sha256", source["sha256"])
    if require_pinned_source:
        pinned_source = {
            "dataset_repo": MINDFORGE_DATASET_REPO,
            "dataset_revision": MINDFORGE_DATASET_REVISION,
            "file": MINDFORGE_DATASET_FILE,
            "path_basename": MINDFORGE_DATASET_FILE,
            "size_bytes": MINDFORGE_DATASET_FILE_BYTES,
            "sha256": MINDFORGE_DATASET_FILE_SHA256,
            "row_count": MINDFORGE_DATASET_TRAJECTORIES,
        }
        if source != pinned_source:
            raise ValueError(
                "manifest source receipt is not the pinned MindForge object"
            )
    tokenizer_identity = _require_keys(
        root["tokenizer"],
        {"resolved_path", "target_content_sha256", "chat_template_sha256"},
        label="manifest.tokenizer",
        allow_extra=True,
    )
    if (
        not isinstance(tokenizer_identity["resolved_path"], str)
        or not tokenizer_identity["resolved_path"]
    ):
        raise ValueError("manifest tokenizer resolved_path is empty")
    _sha256(
        "manifest.tokenizer.target_content_sha256",
        tokenizer_identity["target_content_sha256"],
    )
    _sha256(
        "manifest.tokenizer.chat_template_sha256",
        tokenizer_identity["chat_template_sha256"],
    )
    if require_pinned_source and "receipt_sha256" not in tokenizer_identity:
        raise ValueError("production tokenizer receipt_sha256 is required")
    if "receipt_sha256" in tokenizer_identity:
        expected_tokenizer_fields = {
            "resolved_path",
            "target_content_sha256",
            "chat_template_sha256",
            "files",
            "tokenizer_class",
            "vocab_size",
            "receipt_sha256",
        }
        if set(tokenizer_identity) != expected_tokenizer_fields:
            raise ValueError("manifest tokenizer receipt has the wrong v1 fields")
        files = _require_keys(
            tokenizer_identity["files"],
            set(TOKENIZER_RECEIPT_FILES),
            label="manifest.tokenizer.files",
        )
        for name in TOKENIZER_RECEIPT_FILES:
            entry = _require_keys(
                files[name],
                {"size_bytes", "sha256"},
                label=f"manifest.tokenizer.files.{name}",
            )
            _plain_int(
                f"manifest.tokenizer.files.{name}.size_bytes",
                entry["size_bytes"],
                minimum=1,
            )
            _sha256(f"manifest.tokenizer.files.{name}.sha256", entry["sha256"])
        if tokenizer_identity["target_content_sha256"] != _canonical_sha256(files):
            raise ValueError("manifest tokenizer content checksum mismatch")
        if (
            not isinstance(tokenizer_identity["tokenizer_class"], str)
            or not tokenizer_identity["tokenizer_class"]
        ):
            raise ValueError("manifest tokenizer_class is empty")
        _plain_int(
            "manifest.tokenizer.vocab_size",
            tokenizer_identity["vocab_size"],
            minimum=1,
        )
        observed = _sha256(
            "manifest.tokenizer.receipt_sha256", tokenizer_identity["receipt_sha256"]
        )
        receipt_material = dict(tokenizer_identity)
        receipt_material.pop("receipt_sha256")
        if observed != _canonical_sha256(receipt_material):
            raise ValueError("manifest tokenizer receipt checksum mismatch")
    checkpoint_identity = _require_keys(
        root["checkpoint"],
        {"compressor_sha256"},
        label="manifest.checkpoint",
        allow_extra=True,
    )
    _sha256(
        "manifest.checkpoint.compressor_sha256",
        checkpoint_identity["compressor_sha256"],
    )
    if require_pinned_source and "receipt_sha256" not in checkpoint_identity:
        raise ValueError("production checkpoint receipt_sha256 is required")
    if "receipt_sha256" in checkpoint_identity:
        baseline_checkpoint_fields = {
            "resolved_path",
            "manifest_sha256",
            "config_sha256",
            "config_file_sha256",
            "compressor_sha256",
            "compressor_size_bytes",
            "export_schema",
            "source_global_step",
            "source_curriculum_stage",
            "source_target_content_identity",
            "source_target_content_identity_sha256",
            "receipt_sha256",
        }
        checkpoint_stage = checkpoint_identity.get("source_curriculum_stage")
        expected_checkpoint_fields = set(baseline_checkpoint_fields)
        if checkpoint_stage == MINDFORGE_CE_PORTABLE_STAGE:
            expected_checkpoint_fields.update(
                {"source_objective", "mindforge_ce_lineage"}
            )
        if set(checkpoint_identity) != expected_checkpoint_fields:
            raise ValueError("manifest checkpoint receipt has the wrong v1 fields")
        for digest_name in (
            "manifest_sha256",
            "config_sha256",
            "config_file_sha256",
        ):
            _sha256(
                f"manifest.checkpoint.{digest_name}",
                checkpoint_identity[digest_name],
            )
        _plain_int(
            "manifest.checkpoint.compressor_size_bytes",
            checkpoint_identity["compressor_size_bytes"],
            minimum=1,
        )
        if checkpoint_identity["export_schema"] != (
            "qwen38_27b_r2eact_token_clock_serving_v1"
        ):
            raise ValueError("manifest checkpoint export ABI changed")
        if checkpoint_stage == BASELINE_PORTABLE_STAGE:
            if checkpoint_identity["source_global_step"] != 1_000:
                raise ValueError("manifest baseline checkpoint step changed")
        elif checkpoint_stage == MINDFORGE_CE_PORTABLE_STAGE:
            if root["schema"] != COMPATIBILITY_REBIND_SCHEMA:
                raise ValueError(
                    "CE checkpoints require the compatibility rebind schema"
                )
            _plain_int(
                "manifest.checkpoint.source_global_step",
                checkpoint_identity["source_global_step"],
                minimum=1,
            )
            if checkpoint_identity.get("source_objective") != "ce":
                raise ValueError("manifest CE checkpoint objective changed")
            _validate_ce_checkpoint_lineage_receipt(checkpoint_identity)
        else:
            raise ValueError("manifest checkpoint source stage changed")
        target_identity = _require_keys(
            checkpoint_identity["source_target_content_identity"],
            {
                "resolved_path",
                "weights",
                "metadata",
                "weight_content_sha256",
                "metadata_content_sha256",
                "target_content_sha256",
            },
            label="manifest.checkpoint.source_target_content_identity",
        )
        if (
            not isinstance(target_identity["resolved_path"], str)
            or not target_identity["resolved_path"]
        ):
            raise ValueError("manifest checkpoint target resolved_path is empty")
        for entries_name, content_digest_name in (
            ("weights", "weight_content_sha256"),
            ("metadata", "metadata_content_sha256"),
        ):
            entries = target_identity[entries_name]
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"manifest checkpoint target {entries_name} is empty")
            for entry_index, entry_value in enumerate(entries):
                entry = _require_keys(
                    entry_value,
                    {"name", "sha256"},
                    label=(
                        "manifest.checkpoint.source_target_content_identity."
                        f"{entries_name}[{entry_index}]"
                    ),
                )
                if not isinstance(entry["name"], str) or not entry["name"]:
                    raise ValueError("manifest checkpoint target file name is empty")
                _sha256("manifest checkpoint target file digest", entry["sha256"])
            if target_identity[content_digest_name] != _canonical_sha256(entries):
                raise ValueError(
                    f"manifest checkpoint target {content_digest_name} mismatch"
                )
        target_material = dict(target_identity)
        target_content_sha = _sha256(
            "manifest checkpoint target_content_sha256",
            target_material.pop("target_content_sha256"),
        )
        if target_content_sha != _canonical_sha256(target_material):
            raise ValueError("manifest checkpoint target content checksum mismatch")
        declared_identity_sha = _sha256(
            "manifest.checkpoint.source_target_content_identity_sha256",
            checkpoint_identity["source_target_content_identity_sha256"],
        )
        if declared_identity_sha != _canonical_sha256(target_identity):
            raise ValueError("manifest checkpoint target identity checksum mismatch")
        observed = _sha256(
            "manifest.checkpoint.receipt_sha256",
            checkpoint_identity["receipt_sha256"],
        )
        receipt_material = dict(checkpoint_identity)
        receipt_material.pop("receipt_sha256")
        if observed != _canonical_sha256(receipt_material):
            raise ValueError("manifest checkpoint receipt checksum mismatch")
    adapter = _require_keys(
        root["adapter"],
        {"contract", "implementation_sha256"},
        label="manifest.adapter",
    )
    if adapter["contract"] != MINDFORGE_CONTRACT:
        raise ValueError("manifest adapter contract differs from the active adapter")
    _sha256("manifest.adapter.implementation_sha256", adapter["implementation_sha256"])
    if require_pinned_source:
        active_adapter = Path(__file__).with_name("mindforge_glm52_data.py")
        if adapter["implementation_sha256"] != _file_sha256(active_adapter):
            raise ValueError("manifest adapter implementation changed")
    policy = _require_keys(
        root["policy"],
        {
            "history_window_tokens",
            "max_action_tokens",
            "max_student_sequence_tokens",
            "selection_seed",
            "eviction_block_tokens",
            "final_working_memory_tokens",
            "compression_ratio",
            "train_trajectories",
            "actions_per_train_trajectory",
            "validation_trajectories",
            "actions_per_validation_trajectory",
            "selection",
            "clean_pilot_excluded_anomalies",
            "action_truncation",
            "action_weighting",
            "compression_cutpoint_weighting",
            "split_is_task_disjoint",
            "split_limitation",
            "history_before_window_excluded",
            "initial_memory",
            "long_trajectory_position_used_for_action_selection_only",
            "target_harness_submit_literal_excluded",
            "target_submission_sentinel_reject_reason",
            "target_submission_sentinel_audit_events",
            "target_submission_sentinel_audit_rows",
            "target_submission_sentinel_audit_records_sha256",
        },
        label="manifest.policy",
    )
    if policy["eviction_block_tokens"] != EVICTION_BLOCK_TOKENS:
        raise ValueError("manifest eviction-block contract changed")
    if policy["final_working_memory_tokens"] != FINAL_WORKING_MEMORY_TOKENS:
        raise ValueError("manifest working-memory contract changed")
    if policy["compression_ratio"] != COMPRESSION_RATIO:
        raise ValueError("manifest compression ratio changed")
    if policy["action_truncation"] != "forbidden":
        raise ValueError("manifest permits action truncation")
    if policy["selection"] != "hash_rank_per_split_then_last_clean_complete_actions_v1":
        raise ValueError("manifest selection policy changed")
    if policy["clean_pilot_excluded_anomalies"] != sorted(
        CLEAN_PILOT_EXCLUDED_ANOMALIES
    ):
        raise ValueError("manifest clean anomaly exclusions changed")
    if policy["action_weighting"] != "unit_trajectory_then_uniform_selected_action":
        raise ValueError("manifest action weighting changed")
    if policy["compression_cutpoint_weighting"] != "uniform_1_over_K_within_action":
        raise ValueError("manifest compression-cutpoint weighting changed")
    if policy["history_before_window_excluded"] is not True:
        raise ValueError("v1 must exclude history before the cold-start window")
    if policy["initial_memory"] is not None:
        raise ValueError("v1 window must start from M=None")
    if policy["long_trajectory_position_used_for_action_selection_only"] is not True:
        raise ValueError("v1 may use long position only for action selection")
    if policy["target_harness_submit_literal_excluded"] != HARNESS_SUBMIT_LITERAL:
        raise ValueError("v1 target harness-control exclusion changed")
    if (
        policy["target_submission_sentinel_reject_reason"]
        != SUBMISSION_SENTINEL_REJECT_REASON
        or policy["target_submission_sentinel_audit_events"]
        != SUBMISSION_SENTINEL_AUDIT_EVENTS
        or policy["target_submission_sentinel_audit_rows"]
        != SUBMISSION_SENTINEL_AUDIT_ROWS
        or policy["target_submission_sentinel_audit_records_sha256"]
        != SUBMISSION_SENTINEL_AUDIT_RECORDS_SHA256
    ):
        raise ValueError("v1 target submission-sentinel audit receipt changed")
    rows = root["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest.rows must be a non-empty array")
    row_keys = {
        "split",
        "training_order",
        "selection_key",
        "line_1based",
        "byte_offset",
        "byte_length",
        "raw_jsonl_line_sha256",
        "raw_json_payload_sha256",
        "source_trajectory_sha256",
        "normalized_trajectory_sha256",
        "trajectory_anomaly_flags",
        "dropped_empty_terminal_assistant",
        "events",
    }
    event_keys = {
        "event_id",
        "selected_event_instance_id",
        "cutpoint_index",
        "trajectory_decision_count",
        "context_sha256",
        "target_sha256",
        "rendered_sha256",
        "prompt_tokens",
        "target_tokens",
        "raw_prefix_tokens",
        "history_tokens",
        "history_window_tokens",
        "compression_cutpoint_count",
        "final_memory_tokens",
        "raw_oracle_sequence_tokens",
        "maximum_student_sequence_tokens",
        "final_memory_student_sequence_tokens",
        "no_old_history_sequence_tokens",
        "anomaly_flags",
        "context_submit_reminder_count",
        "generation_output_limit_feedback_count",
        "target_has_tool_calls",
        "target_tool_call_count",
        "selected_event_weight",
    }
    seen_lines: set[str] = set()
    seen_event_instances: set[str] = set()
    seen_event_ids: set[str] = set()
    expected_order = {"train": 0, "validation": 0}
    counts = {"train": 0, "validation": 0}
    action_counts = {"train": 0, "validation": 0}
    for row_index, row_value in enumerate(rows):
        row = _require_keys(row_value, row_keys, label=f"rows[{row_index}]")
        split = row["split"]
        if split not in expected_order:
            raise ValueError(f"rows[{row_index}].split is invalid")
        if row["training_order"] != expected_order[split]:
            raise ValueError(f"rows[{row_index}] has a non-contiguous training_order")
        expected_order[split] += 1
        line_sha = _sha256(
            f"rows[{row_index}].raw_jsonl_line_sha256",
            row["raw_jsonl_line_sha256"],
        )
        expected_selection_key = _selection_key(
            int(policy["selection_seed"]), str(split), line_sha
        )
        if row["selection_key"] != expected_selection_key:
            raise ValueError("manifest row selection_key is inconsistent")
        _sha256(
            f"rows[{row_index}].raw_json_payload_sha256",
            row["raw_json_payload_sha256"],
        )
        _sha256(
            f"rows[{row_index}].source_trajectory_sha256",
            row["source_trajectory_sha256"],
        )
        if line_sha in seen_lines:
            raise ValueError("manifest selects a source row more than once")
        seen_lines.add(line_sha)
        events = row["events"]
        if not isinstance(events, list) or not events:
            raise ValueError(f"rows[{row_index}].events must be non-empty")
        expected_actions_per_row = int(
            policy[
                (
                    "actions_per_train_trajectory"
                    if split == "train"
                    else "actions_per_validation_trajectory"
                )
            ]
        )
        if len(events) != expected_actions_per_row:
            raise ValueError("manifest row selected-action count differs from policy")
        cutpoints: list[int] = []
        weight_sum = 0.0
        for event_index, event_value in enumerate(events):
            event = _require_keys(
                event_value,
                event_keys,
                label=f"rows[{row_index}].events[{event_index}]",
            )
            event_id = _sha256("event_id", event["event_id"])
            if event_id in seen_event_ids:
                raise ValueError("manifest repeats a causal event ID globally")
            seen_event_ids.add(event_id)
            instance_id = _sha256(
                "selected_event_instance_id", event["selected_event_instance_id"]
            )
            expected_instance_id = hashlib.sha256(
                (line_sha + ":" + event_id).encode("ascii")
            ).hexdigest()
            if instance_id != expected_instance_id:
                raise ValueError("selected event instance identity is inconsistent")
            if instance_id in seen_event_instances:
                raise ValueError("manifest selects an event instance more than once")
            seen_event_instances.add(instance_id)
            if event["anomaly_flags"]:
                raise ValueError("clean pilot manifest contains an anomalous event")
            if event["history_window_tokens"] != policy["history_window_tokens"]:
                raise ValueError("event history window differs from policy")
            if event["target_tokens"] > policy["max_action_tokens"]:
                raise ValueError("manifest event exceeds max_action_tokens")
            if (
                event["raw_oracle_sequence_tokens"]
                > policy["max_student_sequence_tokens"]
                or event["maximum_student_sequence_tokens"]
                > policy["max_student_sequence_tokens"]
            ):
                raise ValueError("manifest event exceeds sequence budget")
            cutpoints.append(_plain_int("cutpoint_index", event["cutpoint_index"]))
            weight = event["selected_event_weight"]
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) <= 0
            ):
                raise ValueError("selected_event_weight must be finite and positive")
            weight_sum += float(weight)
        if cutpoints != sorted(cutpoints) or len(cutpoints) != len(set(cutpoints)):
            raise ValueError("row event cutpoints are not unique chronological values")
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("selected action weights do not sum to one per trajectory")
        counts[split] += 1
        action_counts[split] += len(events)
    declared = root["counts"]
    expected_counts = {
        "train_trajectories": counts["train"],
        "train_actions": action_counts["train"],
        "validation_trajectories": counts["validation"],
        "validation_actions": action_counts["validation"],
    }
    if declared != expected_counts:
        raise ValueError(
            f"manifest counts mismatch: {declared!r} != {expected_counts!r}"
        )
    if (
        policy["train_trajectories"] != counts["train"]
        or policy["validation_trajectories"] != counts["validation"]
        or policy["actions_per_train_trajectory"] * counts["train"]
        != action_counts["train"]
        or policy["actions_per_validation_trajectory"] * counts["validation"]
        != action_counts["validation"]
    ):
        raise ValueError("manifest policy counts differ from selected rows")
    if (
        root["schema"] in {SCHEMA, COMPATIBILITY_REBIND_SCHEMA}
        and require_pinned_source
    ):
        primary_counts = {
            "train_trajectories": DEFAULT_TRAIN_TRAJECTORIES,
            "actions_per_train_trajectory": (DEFAULT_ACTIONS_PER_TRAIN_TRAJECTORY),
            "validation_trajectories": DEFAULT_VALIDATION_TRAJECTORIES,
            "actions_per_validation_trajectory": (
                DEFAULT_ACTIONS_PER_VALIDATION_TRAJECTORY
            ),
        }
        for key, expected in primary_counts.items():
            if policy[key] != expected:
                raise ValueError(f"primary manifest policy {key} must be {expected}")
    if (
        root["gold_action_contract"]
        != "generation_prompt_tokenize_plus_independent_complete_continuation_encode"
        or root["joint_completed_chat_token_prefix_required"] is not False
    ):
        raise ValueError("manifest gold autoregressive action contract changed")
    if root["schema"] == COMPATIBILITY_REBIND_SCHEMA:
        _validate_compatibility_rebind_receipt(
            root, require_pinned_source=require_pinned_source
        )
    return json.loads(json.dumps(root, ensure_ascii=False, allow_nan=False))


def compatibility_cohort_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash the checkpoint-independent validation cohort and scoring policy."""

    validation_rows = [
        row
        for row in manifest.get("rows", ())
        if isinstance(row, Mapping) and row.get("split") == "validation"
    ]
    if not validation_rows:
        raise ValueError("compatibility cohort has no validation rows")
    material = {
        "source": manifest.get("source"),
        "tokenizer": manifest.get("tokenizer"),
        "adapter": manifest.get("adapter"),
        "policy": manifest.get("policy"),
        "validation_rows": validation_rows,
        "gold_action_contract": manifest.get("gold_action_contract"),
        "joint_completed_chat_token_prefix_required": manifest.get(
            "joint_completed_chat_token_prefix_required"
        ),
    }
    return _canonical_sha256(material)


def _canonical_manifest_artifact_sha256(manifest: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rebind_compatibility_manifest(
    source_manifest: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    *,
    require_pinned_source: bool = True,
) -> dict[str, Any]:
    """Clone an exact primary 8K cohort onto one portable checkpoint receipt."""

    checked = validate_ce_manifest(
        source_manifest, require_pinned_source=require_pinned_source
    )
    if checked["schema"] != SCHEMA:
        raise ValueError("compatibility rebind requires the full primary manifest")
    destination_probe = json.loads(
        json.dumps(checked, ensure_ascii=False, allow_nan=False)
    )
    destination_probe["checkpoint"] = json.loads(
        json.dumps(checkpoint_identity, ensure_ascii=False, allow_nan=False)
    )
    destination_probe.pop("contract_sha256")
    destination_probe = _with_contract_sha256(destination_probe)
    if not isinstance(destination_probe["checkpoint"], Mapping):
        raise TypeError("rebind checkpoint identity must be a mapping")
    source_target = checked["checkpoint"]["source_target_content_identity_sha256"]
    if (
        destination_probe["checkpoint"]["source_target_content_identity_sha256"]
        != source_target
    ):
        raise ValueError("rebind checkpoint target identity differs from the cohort")
    v2_training_manifest: Mapping[str, Any] | None = None
    v2_evaluation_manifest: dict[str, Any] | None = None
    if (
        destination_probe["checkpoint"].get("source_curriculum_stage")
        == MINDFORGE_CE_PORTABLE_STAGE
    ):
        lineage = destination_probe["checkpoint"]["mindforge_ce_lineage"]
        if lineage["schema"] == MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA:
            primary = lineage["primary_manifest"]
            if primary != {
                "artifact_sha256": _canonical_manifest_artifact_sha256(checked),
                "contract_sha256": checked["contract_sha256"],
            }:
                raise ValueError(
                    "CE checkpoint was trained from a different primary manifest"
                )
        elif lineage["schema"] == MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA_V2:
            # A 4K-trained compressor is deliberately evaluated on the original
            # 8K primary cohort.  Exact release pins make the cross-horizon
            # bridge auditable and prevent a self-rehashed VIEW from entering.
            v2_training_manifest = lineage["training_manifest"]
            v2_evaluation_manifest = exact_h8192_evaluation_manifest_receipt(checked)
            expected_baseline_initialization = {
                "checkpoint_receipt_sha256": checked["checkpoint"]["receipt_sha256"],
                "compressor_sha256": checked["checkpoint"]["compressor_sha256"],
                "source_global_step": checked["checkpoint"]["source_global_step"],
                "source_curriculum_stage": checked["checkpoint"][
                    "source_curriculum_stage"
                ],
                "source_target_content_identity_sha256": checked["checkpoint"][
                    "source_target_content_identity_sha256"
                ],
            }
            if lineage["baseline_initialization"] != expected_baseline_initialization:
                raise ValueError(
                    "4K CE baseline lineage differs from the 8K source checkpoint"
                )
            if (
                v2_training_manifest["counts"] != v2_evaluation_manifest["counts"]
                or v2_training_manifest["action_cohort_sha256"]
                != v2_evaluation_manifest["action_cohort_sha256"]
                or v2_training_manifest["history_window_tokens"] != 4_096
                or v2_evaluation_manifest["history_window_tokens"] != 8_192
            ):
                raise ValueError(
                    "4K training and 8K evaluation action cohorts are not identical"
                )
        else:  # guarded by checkpoint receipt validation, kept fail-closed here.
            raise ValueError("CE checkpoint has an unsupported portable lineage")

    receipt_material: dict[str, Any] = {
        "schema": (
            COMPATIBILITY_REBIND_RECEIPT_SCHEMA_V2
            if v2_training_manifest is not None
            else COMPATIBILITY_REBIND_RECEIPT_SCHEMA
        ),
        "source_manifest_schema": SCHEMA,
        "source_manifest_artifact_sha256": _canonical_manifest_artifact_sha256(checked),
        "source_manifest_contract_sha256": checked["contract_sha256"],
        "source_checkpoint": checked["checkpoint"],
        "destination_checkpoint_receipt_sha256": destination_probe["checkpoint"][
            "receipt_sha256"
        ],
        "compatibility_cohort_sha256": compatibility_cohort_sha256(checked),
    }
    if v2_training_manifest is not None:
        if v2_evaluation_manifest is None:
            raise RuntimeError("4K-to-8K rebind lost its evaluation manifest receipt")
        receipt_material.update(
            {
                "training_manifest": json.loads(
                    json.dumps(
                        v2_training_manifest, ensure_ascii=False, allow_nan=False
                    )
                ),
                "evaluation_manifest": v2_evaluation_manifest,
                "allowed_geometry_difference": (_h4096_to_h8192_geometry_receipt()),
            }
        )
    receipt = dict(receipt_material)
    receipt["receipt_sha256"] = _canonical_sha256(receipt_material)

    rebound = json.loads(json.dumps(checked, ensure_ascii=False, allow_nan=False))
    rebound.pop("contract_sha256")
    rebound["schema"] = COMPATIBILITY_REBIND_SCHEMA
    rebound["checkpoint"] = destination_probe["checkpoint"]
    rebound["checkpoint_rebind"] = receipt
    result = _with_contract_sha256(rebound)
    return validate_ce_manifest(result, require_pinned_source=require_pinned_source)


def _validate_compatibility_rebind_receipt(
    manifest: Mapping[str, Any], *, require_pinned_source: bool
) -> None:
    raw_receipt = manifest.get("checkpoint_rebind")
    if not isinstance(raw_receipt, Mapping):
        raise ValueError("manifest.checkpoint_rebind must be an object")
    receipt_schema = raw_receipt.get("schema")
    v2_fields: set[str] = set()
    if receipt_schema == COMPATIBILITY_REBIND_RECEIPT_SCHEMA_V2:
        v2_fields = {
            "training_manifest",
            "evaluation_manifest",
            "allowed_geometry_difference",
        }
    receipt = _require_keys(
        raw_receipt,
        {
            "schema",
            "source_manifest_schema",
            "source_manifest_artifact_sha256",
            "source_manifest_contract_sha256",
            "source_checkpoint",
            "destination_checkpoint_receipt_sha256",
            "compatibility_cohort_sha256",
            "receipt_sha256",
        }
        | v2_fields,
        label="manifest.checkpoint_rebind",
    )
    if (
        receipt["schema"]
        not in {
            COMPATIBILITY_REBIND_RECEIPT_SCHEMA,
            COMPATIBILITY_REBIND_RECEIPT_SCHEMA_V2,
        }
        or receipt["source_manifest_schema"] != SCHEMA
    ):
        raise ValueError("unsupported compatibility rebind receipt schema")
    observed_receipt_sha = _sha256(
        "manifest.checkpoint_rebind.receipt_sha256", receipt["receipt_sha256"]
    )
    receipt_material = dict(receipt)
    receipt_material.pop("receipt_sha256")
    if observed_receipt_sha != _canonical_sha256(receipt_material):
        raise ValueError("compatibility rebind receipt self-hash mismatch")
    if (
        receipt["destination_checkpoint_receipt_sha256"]
        != manifest["checkpoint"]["receipt_sha256"]
    ):
        raise ValueError("compatibility rebind destination checkpoint changed")

    source = json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
    source.pop("contract_sha256")
    source.pop("checkpoint_rebind")
    source["schema"] = SCHEMA
    source["checkpoint"] = receipt["source_checkpoint"]
    source = _with_contract_sha256(source)
    if source["contract_sha256"] != receipt["source_manifest_contract_sha256"]:
        raise ValueError("compatibility rebind source contract reconstruction failed")
    checked_source = validate_ce_manifest(
        source, require_pinned_source=require_pinned_source
    )
    if (
        _canonical_manifest_artifact_sha256(checked_source)
        != receipt["source_manifest_artifact_sha256"]
    ):
        raise ValueError("compatibility rebind source artifact reconstruction failed")
    cohort_sha = compatibility_cohort_sha256(checked_source)
    if (
        receipt["compatibility_cohort_sha256"] != cohort_sha
        or compatibility_cohort_sha256(manifest) != cohort_sha
    ):
        raise ValueError("compatibility rebind cohort changed")
    source_target_sha = checked_source["checkpoint"][
        "source_target_content_identity_sha256"
    ]
    if manifest["checkpoint"]["source_target_content_identity_sha256"] != (
        source_target_sha
    ):
        raise ValueError("compatibility rebind target identity changed")
    if (
        manifest["checkpoint"].get("source_curriculum_stage")
        == MINDFORGE_CE_PORTABLE_STAGE
    ):
        lineage = manifest["checkpoint"]["mindforge_ce_lineage"]
        if lineage["schema"] == MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA:
            if receipt_schema != COMPATIBILITY_REBIND_RECEIPT_SCHEMA:
                raise ValueError("legacy 8K CE lineage requires the v1 rebind receipt")
            primary = lineage["primary_manifest"]
            if primary != {
                "artifact_sha256": receipt["source_manifest_artifact_sha256"],
                "contract_sha256": receipt["source_manifest_contract_sha256"],
            }:
                raise ValueError("compatibility CE lineage binds a different manifest")
        elif lineage["schema"] == MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA_V2:
            if receipt_schema != COMPATIBILITY_REBIND_RECEIPT_SCHEMA_V2:
                raise ValueError("4K CE lineage requires the v2 rebind receipt")
            expected_baseline_initialization = {
                "checkpoint_receipt_sha256": checked_source["checkpoint"][
                    "receipt_sha256"
                ],
                "compressor_sha256": checked_source["checkpoint"]["compressor_sha256"],
                "source_global_step": checked_source["checkpoint"][
                    "source_global_step"
                ],
                "source_curriculum_stage": checked_source["checkpoint"][
                    "source_curriculum_stage"
                ],
                "source_target_content_identity_sha256": checked_source["checkpoint"][
                    "source_target_content_identity_sha256"
                ],
            }
            if lineage["baseline_initialization"] != expected_baseline_initialization:
                raise ValueError(
                    "4K CE baseline lineage differs from the 8K source checkpoint"
                )
            expected_evaluation = exact_h8192_evaluation_manifest_receipt(
                checked_source
            )
            training = _require_keys(
                receipt["training_manifest"],
                {
                    "schema",
                    "artifact_sha256",
                    "contract_sha256",
                    "history_window_tokens",
                    "counts",
                    "action_cohort_sha256",
                },
                label="manifest.checkpoint_rebind.training_manifest",
            )
            evaluation = _require_keys(
                receipt["evaluation_manifest"],
                {
                    "schema",
                    "artifact_sha256",
                    "contract_sha256",
                    "history_window_tokens",
                    "counts",
                    "action_cohort_sha256",
                },
                label="manifest.checkpoint_rebind.evaluation_manifest",
            )
            if (
                training != lineage["training_manifest"]
                or evaluation != expected_evaluation
                or training["counts"] != evaluation["counts"]
                or training["action_cohort_sha256"]
                != evaluation["action_cohort_sha256"]
                or receipt["allowed_geometry_difference"]
                != _h4096_to_h8192_geometry_receipt()
            ):
                raise ValueError("4K training / 8K evaluation rebind lineage changed")
        else:
            raise ValueError("compatibility CE lineage schema is unsupported")
    elif receipt_schema != COMPATIBILITY_REBIND_RECEIPT_SCHEMA:
        raise ValueError("baseline checkpoints require the legacy v1 rebind receipt")


def load_and_validate_ce_manifest(
    path: str | Path,
    *,
    expected_source_path: str | Path | None = None,
    require_pinned_source: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid CE manifest JSON: {manifest_path}") from error
    manifest = validate_ce_manifest(value, require_pinned_source=require_pinned_source)
    receipt = manifest["source"]
    if require_pinned_source:
        pinned = {
            "dataset_repo": MINDFORGE_DATASET_REPO,
            "dataset_revision": MINDFORGE_DATASET_REVISION,
            "file": MINDFORGE_DATASET_FILE,
            "path_basename": MINDFORGE_DATASET_FILE,
            "size_bytes": MINDFORGE_DATASET_FILE_BYTES,
            "sha256": MINDFORGE_DATASET_FILE_SHA256,
            "row_count": MINDFORGE_DATASET_TRAJECTORIES,
        }
        if receipt != pinned:
            raise ValueError(
                "manifest source receipt is not the pinned MindForge object"
            )
        adapter_path = Path(__file__).with_name("mindforge_glm52_data.py")
        if manifest["adapter"]["implementation_sha256"] != _file_sha256(adapter_path):
            raise ValueError("manifest adapter implementation changed")
    if expected_source_path is not None:
        source = Path(expected_source_path).expanduser().resolve()
        if source.name != receipt["path_basename"]:
            raise ValueError("manifest source basename differs from requested source")
        if not source.is_file() or source.stat().st_size != receipt["size_bytes"]:
            raise ValueError("manifest source file size mismatch")
        if _file_sha256(source) != receipt["sha256"]:
            raise ValueError("manifest source file SHA-256 mismatch")
    return manifest


def iter_manifest_rows(
    manifest: Mapping[str, Any],
    split: str,
    *,
    require_pinned_source: bool = True,
) -> Iterator[Mapping[str, Any]]:
    checked = validate_ce_manifest(
        manifest, require_pinned_source=require_pinned_source
    )
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    selected = [row for row in checked["rows"] if row["split"] == split]
    selected.sort(key=lambda row: int(row["training_order"]))
    yield from selected


def load_manifest_trajectory(
    source_path: str | Path, row: Mapping[str, Any]
) -> MindForgeTrajectory:
    """Seek one manifest row and revalidate raw plus normalized identities."""

    required = {
        "line_1based",
        "byte_offset",
        "byte_length",
        "raw_jsonl_line_sha256",
        "raw_json_payload_sha256",
        "source_trajectory_sha256",
        "normalized_trajectory_sha256",
        "events",
    }
    _require_keys(row, required, label="manifest row", allow_extra=True)
    pointer = SourceLine(
        line_1based=_plain_int("line_1based", row["line_1based"], minimum=1),
        byte_offset=_plain_int("byte_offset", row["byte_offset"]),
        byte_length=_plain_int("byte_length", row["byte_length"], minimum=1),
        raw_jsonl_line_sha256=_sha256(
            "raw_jsonl_line_sha256", row["raw_jsonl_line_sha256"]
        ),
        raw_json_payload_sha256=_sha256(
            "raw_json_payload_sha256", row["raw_json_payload_sha256"]
        ),
    )
    raw_line = _read_source_line(Path(source_path), pointer)
    trajectory = parse_mindforge_jsonl_line(raw_line, line_number=pointer.line_1based)
    if trajectory.source_trajectory_sha256 != row["source_trajectory_sha256"]:
        raise ValueError("manifest source trajectory checksum mismatch")
    if trajectory.normalized_trajectory_sha256 != row["normalized_trajectory_sha256"]:
        raise ValueError("manifest normalized trajectory checksum mismatch")
    expected_event_ids = [str(event["event_id"]) for event in row["events"]]
    available = {event.event_id for event in trajectory.events}
    if len(expected_event_ids) != len(set(expected_event_ids)) or not set(
        expected_event_ids
    ).issubset(available):
        raise ValueError("manifest selected event set is absent from source trajectory")
    return trajectory


def derive_history_window_view(
    manifest: Mapping[str, Any],
    history_window_tokens: int,
    *,
    require_pinned_source: bool = True,
) -> dict[str, Any]:
    """Derive a hash-distinct 2K/4K view over the identical selected actions."""

    checked = validate_ce_manifest(
        manifest, require_pinned_source=require_pinned_source
    )
    history_window_tokens = _plain_int(
        "history_window_tokens",
        history_window_tokens,
        minimum=2 * EVICTION_BLOCK_TOKENS,
    )
    if (history_window_tokens - FINAL_WORKING_MEMORY_TOKENS) % EVICTION_BLOCK_TOKENS:
        raise ValueError("history-window view is not eviction-block aligned")
    view = json.loads(json.dumps(checked, ensure_ascii=False, allow_nan=False))
    view.pop("contract_sha256")
    view["schema"] = VIEW_SCHEMA
    view["policy"]["history_window_tokens"] = history_window_tokens
    cutpoint_count = (
        history_window_tokens - FINAL_WORKING_MEMORY_TOKENS
    ) // EVICTION_BLOCK_TOKENS
    slots = recurrent_slot_counts(
        cutpoint_count,
        block_tokens=EVICTION_BLOCK_TOKENS,
        compression_ratio=COMPRESSION_RATIO,
    )
    for row in view["rows"]:
        for event in row["events"]:
            if event["history_tokens"] < history_window_tokens:
                raise ValueError(
                    "selected action is shorter than requested history view"
                )
            prefix_tokens = int(event["raw_prefix_tokens"])
            target_tokens = int(event["target_tokens"])
            event["history_window_tokens"] = history_window_tokens
            event["compression_cutpoint_count"] = cutpoint_count
            event["final_memory_tokens"] = slots[-1]
            event["raw_oracle_sequence_tokens"] = (
                prefix_tokens + history_window_tokens + target_tokens
            )
            event["maximum_student_sequence_tokens"] = (
                prefix_tokens
                + slots[0]
                + history_window_tokens
                - EVICTION_BLOCK_TOKENS
                + target_tokens
            )
            event["final_memory_student_sequence_tokens"] = (
                prefix_tokens + slots[-1] + FINAL_WORKING_MEMORY_TOKENS + target_tokens
            )
            event["no_old_history_sequence_tokens"] = (
                prefix_tokens + FINAL_WORKING_MEMORY_TOKENS + target_tokens
            )
            if (
                max(
                    event["raw_oracle_sequence_tokens"],
                    event["maximum_student_sequence_tokens"],
                )
                > view["policy"]["max_student_sequence_tokens"]
            ):
                raise ValueError("derived history view exceeds sequence budget")
    return _with_contract_sha256(view)


def derive_smoke_view(
    manifest: Mapping[str, Any],
    *,
    train_trajectories: int = 8,
    actions_per_train_trajectory: int = 2,
    validation_trajectories: int = 4,
    actions_per_validation_trajectory: int = 2,
    require_pinned_source: bool = True,
) -> dict[str, Any]:
    """Create a deterministic late-action subset for CPU/tiny-GPU plumbing."""

    checked = validate_ce_manifest(
        manifest, require_pinned_source=require_pinned_source
    )
    requested = {
        "train": (
            _plain_int("train_trajectories", train_trajectories, minimum=1),
            _plain_int(
                "actions_per_train_trajectory",
                actions_per_train_trajectory,
                minimum=1,
            ),
        ),
        "validation": (
            _plain_int("validation_trajectories", validation_trajectories, minimum=1),
            _plain_int(
                "actions_per_validation_trajectory",
                actions_per_validation_trajectory,
                minimum=1,
            ),
        ),
    }
    selected_rows: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        row_count, action_count = requested[split]
        candidates = [row for row in checked["rows"] if row["split"] == split]
        candidates.sort(key=lambda row: int(row["training_order"]))
        if len(candidates) < row_count:
            raise ValueError(f"full manifest has too few {split} rows for smoke view")
        for new_order, source_row in enumerate(candidates[:row_count]):
            if len(source_row["events"]) < action_count:
                raise ValueError(
                    f"full manifest has too few {split} actions for smoke view"
                )
            row = json.loads(json.dumps(source_row, ensure_ascii=False))
            row["training_order"] = new_order
            # Retain the chronologically latest selected actions.
            row["events"] = row["events"][-action_count:]
            for event in row["events"]:
                event["selected_event_weight"] = 1.0 / action_count
            selected_rows.append(row)
    view = json.loads(json.dumps(checked, ensure_ascii=False, allow_nan=False))
    view.pop("contract_sha256")
    view["schema"] = VIEW_SCHEMA
    view["rows"] = selected_rows
    view["policy"]["train_trajectories"] = train_trajectories
    view["policy"]["actions_per_train_trajectory"] = actions_per_train_trajectory
    view["policy"]["validation_trajectories"] = validation_trajectories
    view["policy"][
        "actions_per_validation_trajectory"
    ] = actions_per_validation_trajectory
    view["counts"] = {
        "train_trajectories": train_trajectories,
        "train_actions": train_trajectories * actions_per_train_trajectory,
        "validation_trajectories": validation_trajectories,
        "validation_actions": (
            validation_trajectories * actions_per_validation_trajectory
        ),
    }
    return _with_contract_sha256(view)


def verify_manifest_row_events(
    source_path: str | Path,
    row: Mapping[str, Any],
    tokenizer: Any,
    policy: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Re-render every selected action and match every hashed/count field.

    This is the trainer-facing semantic verifier.  ``load_manifest_trajectory``
    intentionally needs no tokenizer and proves the random-access source
    identity; this function additionally proves the adapter, complete-action
    rendering, cold-start window layout, and exact selected event set.
    """

    trajectory = load_manifest_trajectory(source_path, row)
    required_policy = {
        "history_window_tokens",
        "max_action_tokens",
        "max_student_sequence_tokens",
        "history_before_window_excluded",
        "initial_memory",
        "long_trajectory_position_used_for_action_selection_only",
        "target_harness_submit_literal_excluded",
        "target_submission_sentinel_reject_reason",
        "target_submission_sentinel_audit_events",
        "target_submission_sentinel_audit_rows",
        "target_submission_sentinel_audit_records_sha256",
    }
    _require_keys(
        policy,
        required_policy,
        label="manifest policy for event verification",
        allow_extra=True,
    )
    if (
        policy["history_before_window_excluded"] is not True
        or policy["initial_memory"] is not None
        or policy["long_trajectory_position_used_for_action_selection_only"] is not True
        or policy["target_harness_submit_literal_excluded"] != HARNESS_SUBMIT_LITERAL
        or policy["target_submission_sentinel_reject_reason"]
        != SUBMISSION_SENTINEL_REJECT_REASON
        or policy["target_submission_sentinel_audit_events"]
        != SUBMISSION_SENTINEL_AUDIT_EVENTS
        or policy["target_submission_sentinel_audit_rows"]
        != SUBMISSION_SENTINEL_AUDIT_ROWS
        or policy["target_submission_sentinel_audit_records_sha256"]
        != SUBMISSION_SENTINEL_AUDIT_RECORDS_SHA256
    ):
        raise ValueError("manifest cold-start window policy changed")
    event_by_id = {event.event_id: event for event in trajectory.events}
    if len(event_by_id) != len(trajectory.events):
        raise ValueError("source trajectory has duplicate causal event IDs")
    verified: list[Any] = []
    for selected in row["events"]:
        event_id = str(selected["event_id"])
        event = event_by_id.get(event_id)
        if event is None or event.cutpoint_index != selected["cutpoint_index"]:
            raise ValueError("selected event ID/cutpoint is absent from trajectory")
        actual, rejection = _event_record(
            trajectory,
            event.cutpoint_index,
            tokenizer,
            history_window_tokens=int(policy["history_window_tokens"]),
            max_action_tokens=int(policy["max_action_tokens"]),
            max_student_sequence_tokens=int(policy["max_student_sequence_tokens"]),
        )
        if actual is None:
            raise ValueError(
                f"selected event is no longer clean/eligible: {event_id}: {rejection}"
            )
        actual["selected_event_instance_id"] = hashlib.sha256(
            (str(row["raw_jsonl_line_sha256"]) + ":" + str(actual["event_id"])).encode(
                "ascii"
            )
        ).hexdigest()
        actual["selected_event_weight"] = selected["selected_event_weight"]
        if actual != selected:
            differing = sorted(
                key
                for key in set(actual).union(selected)
                if actual.get(key) != selected.get(key)
            )
            raise ValueError(
                f"selected event record differs after reconstruction: {differing}"
            )
        verified.append(event)
    if len(verified) != len(row["events"]):
        raise RuntimeError("selected event reconstruction was incomplete")
    return tuple(verified)


def write_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    require_pinned_source: bool = True,
) -> None:
    checked = validate_ce_manifest(
        manifest, require_pinned_source=require_pinned_source
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            checked, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--train-trajectories", type=int, default=512)
    parser.add_argument("--train-actions", type=int, default=4)
    parser.add_argument("--validation-trajectories", type=int, default=16)
    parser.add_argument("--validation-actions", type=int, default=2)
    parser.add_argument(
        "--history-window-tokens", type=int, default=DEFAULT_HISTORY_WINDOW_TOKENS
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - production environment only.
        raise RuntimeError("manifest CLI requires transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    manifest = build_ce_manifest(
        args.source,
        tokenizer,
        tokenizer_identity=tokenizer_receipt(tokenizer, args.tokenizer),
        checkpoint_identity=checkpoint_receipt(args.checkpoint),
        selection_seed=args.selection_seed,
        train_trajectories=args.train_trajectories,
        actions_per_train_trajectory=args.train_actions,
        validation_trajectories=args.validation_trajectories,
        actions_per_validation_trajectory=args.validation_actions,
        history_window_tokens=args.history_window_tokens,
    )
    write_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "contract_sha256": manifest["contract_sha256"],
                "counts": manifest["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BASELINE_PORTABLE_STAGE",
    "COMPRESSION_RATIO",
    "COMPATIBILITY_REBIND_SCHEMA",
    "COMPATIBILITY_REBIND_RECEIPT_SCHEMA_V2",
    "DEFAULT_HISTORY_WINDOW_TOKENS",
    "DEFAULT_MAX_ACTION_TOKENS",
    "DEFAULT_MAX_STUDENT_SEQUENCE_TOKENS",
    "EVICTION_BLOCK_TOKENS",
    "FINAL_WORKING_MEMORY_TOKENS",
    "FORMAT_VERSION",
    "MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA",
    "MINDFORGE_CE_PORTABLE_LINEAGE_SCHEMA_V2",
    "MINDFORGE_CE_PORTABLE_STAGE",
    "MINDFORGE_FULL_ACTION_COHORT_SHA256",
    "MINDFORGE_FULL_VIEW_H4096_ARTIFACT_SHA256",
    "MINDFORGE_FULL_VIEW_H4096_CONTRACT_SHA256",
    "MINDFORGE_PRIMARY_H8192_ARTIFACT_SHA256",
    "MINDFORGE_PRIMARY_H8192_CONTRACT_SHA256",
    "MINDFORGE_RELEASE_BASELINE_COMPRESSOR_SHA256",
    "MINDFORGE_RELEASE_BASELINE_RECEIPT_SHA256",
    "MINDFORGE_RELEASE_TARGET_IDENTITY_SHA256",
    "SCHEMA",
    "SourceLine",
    "build_ce_manifest",
    "checkpoint_receipt",
    "compatibility_cohort_sha256",
    "derive_history_window_view",
    "derive_smoke_view",
    "exact_h4096_training_manifest_receipt",
    "exact_h8192_evaluation_manifest_receipt",
    "index_verified_source",
    "iter_manifest_rows",
    "load_and_validate_ce_manifest",
    "load_manifest_trajectory",
    "mindforge_action_cohort_sha256",
    "rebind_compatibility_manifest",
    "tokenizer_receipt",
    "validate_ce_manifest",
    "verify_manifest_row_events",
    "write_manifest",
]
