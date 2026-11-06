#!/usr/bin/env python3
"""Static, CPU-only preflight for a plain Stage-L DFlow-Domino checkpoint.

This file is executed directly by the dedicated launcher.  Keep it independent
of torch, transformers, SGLang imports, CUDA discovery, sockets, and server
startup: its job is only to authenticate JSON/files and inspect safetensors
headers before the serving process is allowed to see a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open


ARCHITECTURE = "DFlowDominoDraftModel"
REMOTE_AUTO_MODEL = "dflow_dfly_domino.DFlowDominoDraftModel"
TRAINING_STAGE = "one_forward_stage_l_dflow_domino"
MARKER_SCHEMA = "specforge-atomic-checkpoint-v1"
PROVENANCE_SCHEMA = "specforge-dflow-stage-l-domino-v1"
TOTAL_HORIZON = 33_300
CHECKPOINT_GATES = [250, 500, 2_000, 5_000, 10_000, 15_000, 25_000, 33_300]
PREFIX_HIDDEN_SIZE = 1024
RESIDUAL_RANK = 256
LATENT_ABI = {
    "full_tail_gaussian": True,
    "stage_l_padding_policy": "whole_block_zero_only",
    "stage_l_loss_mask_scope": "supervision_only_not_latent",
    "stage_l_block_attention": (
        "prefix_lt_anchor_plus_bidirectional_same_block"
    ),
}
HEAD_KEYS = frozenset(
    {
        "prefix_gru.weight_ih_l0",
        "prefix_gru.weight_hh_l0",
        "residual_down.weight",
        "residual_up.weight",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path.name}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{path.name} must contain one JSON object")
        return {}
    return value


def _exact(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_exact(item, wanted) for item, wanted in zip(actual, expected))
        )
    return type(actual) is type(expected) and actual == expected


def _expected_head_shapes(hidden_size: int, vocab_size: int) -> dict[str, list[int]]:
    return {
        "prefix_gru.weight_ih_l0": [3 * PREFIX_HIDDEN_SIZE, hidden_size],
        "prefix_gru.weight_hh_l0": [
            3 * PREFIX_HIDDEN_SIZE,
            PREFIX_HIDDEN_SIZE,
        ],
        "residual_down.weight": [
            RESIDUAL_RANK,
            hidden_size + PREFIX_HIDDEN_SIZE,
        ],
        "residual_up.weight": [vocab_size, RESIDUAL_RANK],
    }


def _normalize_tensor_name(name: str) -> str:
    return name[len("model.") :] if name.startswith("model.") else name


def _validate_config(
    target: Mapping[str, Any],
    draft: Mapping[str, Any],
    marker: Mapping[str, Any],
    errors: list[str],
) -> tuple[int, int]:
    if draft.get("architectures") != [ARCHITECTURE]:
        errors.append(f"architectures={draft.get('architectures')!r}")
    auto_map = draft.get("auto_map")
    auto_model = auto_map.get("AutoModel") if isinstance(auto_map, Mapping) else None
    if auto_model != REMOTE_AUTO_MODEL:
        errors.append(f"auto_map.AutoModel={auto_model!r}")
    if int(draft.get("block_size", -1)) != 16:
        errors.append(f"block_size={draft.get('block_size')!r}, expected=16")
    if "Qwen3.5" in json.dumps(draft, ensure_ascii=False):
        errors.append("forbidden Qwen3.5 alias in draft config")

    hidden_size = int(draft.get("hidden_size", -1))
    vocab_size = int(draft.get("vocab_size", -1))
    if hidden_size != 2560 or vocab_size != 151936:
        errors.append(
            f"draft dimensions={(hidden_size, vocab_size)}, expected=(2560, 151936)"
        )
    target_text = target.get("text_config") or target
    if not isinstance(target_text, Mapping):
        errors.append("target text_config is not an object")
    else:
        if int(target_text.get("hidden_size", -2)) != hidden_size:
            errors.append("target/draft hidden_size mismatch")
        if int(target_text.get("vocab_size", -2)) != vocab_size:
            errors.append("target/draft vocab_size mismatch")

    flow = draft.get("dflow_config")
    if not isinstance(flow, Mapping):
        errors.append("dflow_config is not an object")
        flow = {}
    expected_shapes = _expected_head_shapes(hidden_size, vocab_size)
    flow_contract = {
        "stage_l_domino": True,
        "stage_l_source_arm": "base",
        "pure_draft_prefix_len": 1,
        "stage_l_correction_start_slot": 2,
        "one_step_direct": True,
        "serving_drafter_forwards": 1,
        "stage_l_head_storage": "unified_model_safetensors",
        "stage_l_head_state_keys": sorted(HEAD_KEYS),
        "stage_l_prefix_hidden_size": PREFIX_HIDDEN_SIZE,
        "stage_l_residual_rank": RESIDUAL_RANK,
        "stage_l_slot1_base_only": True,
        "training_stage": TRAINING_STAGE,
        **LATENT_ABI,
    }
    for key, wanted in flow_contract.items():
        if not _exact(flow.get(key), wanted):
            errors.append(f"dflow_config.{key}={flow.get(key)!r}, expected={wanted!r}")
    if (
        flow.get("stage_l_head_shapes") is not None
        and flow.get("stage_l_head_shapes") != expected_shapes
    ):
        errors.append("dflow_config.stage_l_head_shapes differs from the native ABI")
    forbidden_flow = sorted(
        key
        for key in flow
        if key.startswith("layer_fusion_")
        or (
            key
            in {
                "hybrid_target_conditioning",
                "stage_e_dfly_hybrid",
                "full_dfly_adaptation",
                "enable_hidden_correction",
            }
            and flow.get(key) not in (None, False)
        )
    )
    if forbidden_flow:
        errors.append(f"plain config carries fusion/HSC markers={forbidden_flow}")

    marker_contract = {
        "schema": MARKER_SCHEMA,
        "status": "complete",
        "world_size": 2,
        "training_stage": TRAINING_STAGE,
        "source_arm": "base",
        "architecture": ARCHITECTURE,
        "seed": 49,
        "head_storage": "unified_model_safetensors",
        "head_state_keys": sorted(HEAD_KEYS),
        "head_tensor_shapes": expected_shapes,
        "total_horizon": TOTAL_HORIZON,
        "checkpoint_gates": CHECKPOINT_GATES,
        "full_tail_gaussian": True,
    }
    for key, wanted in marker_contract.items():
        if not _exact(marker.get(key), wanted):
            errors.append(
                f"checkpoint marker {key}={marker.get(key)!r}, expected={wanted!r}"
            )
    global_step = marker.get("global_step")
    if (
        not isinstance(global_step, int)
        or isinstance(global_step, bool)
        or global_step not in CHECKPOINT_GATES
    ):
        errors.append(f"checkpoint marker global_step={global_step!r} is not a gate")
    return hidden_size, vocab_size


def _validate_marker_inventory(
    draft_root: Path,
    marker: Mapping[str, Any],
    errors: list[str],
) -> tuple[list[str], dict[str, str]]:
    model_files = marker.get("model_files")
    if (
        not isinstance(model_files, list)
        or not model_files
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith("model")
            or not name.endswith(".safetensors")
            for name in model_files
        )
    ):
        errors.append(f"invalid marker model_files={model_files!r}")
        model_files = []
    elif model_files != sorted(set(model_files)):
        errors.append("marker model_files must be sorted and duplicate-free")

    actual_models = sorted(path.name for path in draft_root.glob("model*.safetensors"))
    if model_files != actual_models:
        errors.append(
            f"marker/actual model inventory mismatch: {model_files!r} != {actual_models!r}"
        )

    payload_hashes = marker.get("payload_sha256")
    if not isinstance(payload_hashes, dict):
        errors.append("checkpoint marker payload_sha256 is not an object")
        payload_hashes = {}
    actual_payloads = sorted(
        path.name
        for path in draft_root.iterdir()
        if path.is_file() and path.name != "checkpoint_complete.json"
    )
    if sorted(payload_hashes) != actual_payloads:
        errors.append("marker payload_sha256 is not the exact regular-file inventory")
    for name, digest in payload_hashes.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            errors.append(f"malformed payload pin for {name!r}")

    provenance_name = marker.get("provenance_file")
    runtime_payloads = {"config.json", *model_files}
    if (
        isinstance(provenance_name, str)
        and Path(provenance_name).name == provenance_name
    ):
        runtime_payloads.add(provenance_name)
        if marker.get("provenance_sha256") != payload_hashes.get(provenance_name):
            errors.append("provenance_sha256 does not match payload_sha256")
    else:
        errors.append(f"invalid marker provenance_file={provenance_name!r}")
    index_name = "model.safetensors.index.json"
    if len(model_files) > 1:
        runtime_payloads.add(index_name)
    elif index_name in actual_payloads:
        errors.append("unsharded model unexpectedly carries a shard index")

    for name in sorted(runtime_payloads):
        path = draft_root / name
        if not path.is_file():
            errors.append(f"missing runtime payload {name}")
            continue
        expected = payload_hashes.get(name)
        if expected is None:
            errors.append(f"runtime payload lacks marker hash: {name}")
        elif _sha256(path) != expected:
            errors.append(f"runtime payload hash mismatch: {name}")
    return list(model_files), dict(payload_hashes)


def _validate_tensor_inventory(
    draft_root: Path,
    model_files: list[str],
    hidden_size: int,
    vocab_size: int,
    errors: list[str],
) -> None:
    raw_to_file: dict[str, str] = {}
    normalized_to_shape: dict[str, tuple[int, ...]] = {}
    for name in model_files:
        try:
            with safe_open(draft_root / name, framework="numpy") as weights:
                for raw_name in weights.keys():
                    if raw_name in raw_to_file:
                        errors.append(f"duplicate raw tensor {raw_name!r}")
                        continue
                    raw_to_file[raw_name] = name
                    normalized = _normalize_tensor_name(raw_name)
                    if normalized in normalized_to_shape:
                        errors.append(f"duplicate normalized tensor {normalized!r}")
                        continue
                    normalized_to_shape[normalized] = tuple(
                        int(dim) for dim in weights.get_slice(raw_name).get_shape()
                    )
        except Exception as exc:
            errors.append(f"cannot inspect {name}: {exc}")

    expected_shapes = {
        key: tuple(shape)
        for key, shape in _expected_head_shapes(hidden_size, vocab_size).items()
    }
    observed_head = set(normalized_to_shape).intersection(HEAD_KEYS)
    if observed_head != set(HEAD_KEYS):
        errors.append(f"head tensor inventory={sorted(observed_head)}")
    for key, wanted in expected_shapes.items():
        actual = normalized_to_shape.get(key)
        if actual is not None and actual != wanted:
            errors.append(f"{key} shape={actual}, expected={wanted}")
    unsupported_head = sorted(
        key
        for key in normalized_to_shape
        if key.startswith(("prefix_gru.", "residual_down.", "residual_up."))
        and key not in HEAD_KEYS
    )
    if unsupported_head:
        errors.append(f"unsupported Domino head tensors={unsupported_head}")
    forbidden = sorted(
        key
        for key in normalized_to_shape
        if key.startswith(("layer_fusion_", "hidden_correction.", "embed_proj."))
    )
    if forbidden:
        errors.append(f"plain checkpoint carries fusion/HSC tensors={forbidden}")

    index_path = draft_root / "model.safetensors.index.json"
    if len(model_files) > 1 and index_path.is_file():
        index = _read_json(index_path, errors)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            errors.append("model shard index weight_map is not an object")
        else:
            if set(weight_map) != set(raw_to_file):
                errors.append(
                    "model shard index key inventory differs from safetensors"
                )
            bad_routes = sorted(
                key
                for key, name in weight_map.items()
                if raw_to_file.get(key) != name or name not in model_files
            )
            if bad_routes:
                errors.append(
                    f"model shard index routes tensors incorrectly={bad_routes[:8]}"
                )


def validate_checkpoint(target_root: Path, draft_root: Path) -> None:
    """Validate the fixed plain-arm ABI, raising one aggregated error."""

    target_root = target_root.expanduser().resolve()
    draft_root = draft_root.expanduser().resolve()
    errors: list[str] = []
    target = _read_json(target_root / "config.json", errors)
    draft = _read_json(draft_root / "config.json", errors)
    marker = _read_json(draft_root / "checkpoint_complete.json", errors)
    hidden_size, vocab_size = _validate_config(target, draft, marker, errors)
    model_files, _ = _validate_marker_inventory(draft_root, marker, errors)
    _validate_tensor_inventory(
        draft_root,
        model_files,
        hidden_size,
        vocab_size,
        errors,
    )

    sidecars = sorted(
        path.name
        for path in draft_root.glob("*domino*.safetensors")
        if path.name not in model_files
    )
    if sidecars:
        errors.append(f"Domino sidecars are forbidden={sidecars}")

    provenance_name = marker.get("provenance_file")
    if isinstance(provenance_name, str) and (draft_root / provenance_name).is_file():
        provenance = _read_json(draft_root / provenance_name, errors)
        if provenance.get("schema") != PROVENANCE_SCHEMA:
            errors.append(
                f"unexpected Stage-L provenance schema={provenance.get('schema')!r}"
            )
        if provenance.get("training_stage") != TRAINING_STAGE:
            errors.append("Stage-L provenance training_stage mismatch")
        if provenance.get("global_step") != marker.get("global_step"):
            errors.append("Stage-L provenance/marker global_step mismatch")
        for key, wanted in LATENT_ABI.items():
            if not _exact(provenance.get(key), wanted):
                errors.append(
                    f"Stage-L provenance {key}={provenance.get(key)!r}, "
                    f"expected={wanted!r}"
                )

    if errors:
        raise ValueError(
            "Plain DFlow-Domino preflight failed:\n- " + "\n- ".join(errors)
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_model", type=Path)
    parser.add_argument("draft_model", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    validate_checkpoint(args.target_model, args.draft_model)
    print("Plain DFlow-Domino static preflight OK: marker-authoritative unified head")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "validate_checkpoint"]
