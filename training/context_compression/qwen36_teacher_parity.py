"""Validate cached-teacher parity evidence before a formal Qwen3.5-family run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


PARITY_FORMAT_VERSION = 2
PARITY_SCHEMA = "qwen35_cached_teacher_output_parity_v2"
SUCCESS_SENTINEL_SCHEMA = "qwen35_cached_teacher_formal_ready_v1"
SUCCESS_SENTINEL_NAME = "FORMAL_READY.json"
LINEAR_ATTENTION_CHUNK_ALIGNMENT = 64

_RUN_CONFIG_METADATA_FIELDS = frozenset(
    {
        "run_contract_sha256",
        "output_dir",
        "resume_from",
        "resume_cursor_override",
    }
)


def cached_teacher_chunk_ranges(
    total_tokens: int,
    *,
    chunk_tokens: int,
    protected_suffix_tokens: int,
    linear_attention_alignment: int = LINEAR_ATTENTION_CHUNK_ALIGNMENT,
) -> tuple[tuple[int, int], ...]:
    """Plan aligned cache chunks with the prediction suffix in the final call.

    Qwen3.5/3.8's FLA GatedDeltaNet kernel uses 64-token internal blocks.  The
    final forward is made as wide as possible and its start is rounded up to a
    global block boundary.  Earlier chunks cover the remaining prefix.  This
    keeps every teacher prediction row in one final forward while respecting
    both the linear-attention alignment and the per-forward token cap.
    """

    if total_tokens <= 0 or chunk_tokens <= 0 or protected_suffix_tokens <= 0:
        raise ValueError("cached-teacher chunk-plan lengths must be positive")
    if protected_suffix_tokens > chunk_tokens:
        raise ValueError("protected teacher suffix cannot exceed one cache chunk")
    if (
        linear_attention_alignment <= 0
        or chunk_tokens % linear_attention_alignment != 0
    ):
        raise ValueError(
            "teacher cache chunks must align with the linear-attention block"
        )
    if total_tokens <= chunk_tokens:
        return ((0, total_tokens),)
    protected_start = total_tokens - protected_suffix_tokens
    minimum_final_start = total_tokens - chunk_tokens
    final_start = (
        (minimum_final_start + linear_attention_alignment - 1)
        // linear_attention_alignment
        * linear_attention_alignment
    )
    if final_start > protected_start:
        raise ValueError("unable to align the protected teacher suffix")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < final_start:
        stop = min(start + chunk_tokens, final_start)
        ranges.append((start, stop))
        start = stop
    ranges.append((final_start, total_tokens))
    if any(stop <= start or stop - start > chunk_tokens for start, stop in ranges):
        raise RuntimeError("cached-teacher chunk plan violates its width contract")
    if ranges[-1][0] > protected_start:
        raise RuntimeError("cached-teacher prediction suffix remains split")
    return tuple(ranges)


# These are deliberately data- and tokenizer-specific probes.  A merely
# self-consistent smoke is not evidence for the production contract: it must
# have exercised the reviewed rows, causal endpoints, and cache boundaries.
CANONICAL_PARITY_PROFILES: dict[int, dict[str, Any]] = {
    512: {
        "kl_tokens": 16,
        "lm_head_chunk_size": 4,
        "max_target_sequence_tokens": 4_096,
        "max_teacher_sequence_tokens": 262_144,
        "max_history_tokens": 262_144,
        "selector": {
            "line_index": 84,
            "example_id": "32",
            "endpoint_event_index": 0,
            "selected_event_indices": [0],
            "raw_line_sha256": (
                "14b8ed362bf8cefd7478d65fab270b454bd14209416d80519fbe493488fdc426"
            ),
            "teacher_prefix_ids_sha256": (
                "cc8623db84f120ba8a88ad3adfa1b22b25f736e114ca7b4850e764e79edf74ef"
            ),
            "target_ids_sha256": (
                "3f106d4180baf83f8ba73c0db693fa2bede433b0370aecbc70065c7fb10bf574"
            ),
            "logical_teacher_forcing_ids_sha256": (
                "ad6efa6ee3a8e2bb1c337a225d37845340cdf1ba2e14fc551465f285ffa78a00"
            ),
            "cached_teacher_input_ids_sha256": (
                "e64c6cd66f3fd34683a54b1afd7f66be072470ae73524868b8f224e2d98bbb9c"
            ),
            "cached_teacher_chunk_ranges": [
                [0, 512],
                [512, 1_024],
                [1_024, 1_536],
                [1_536, 1_792],
                [1_792, 2_295],
            ],
            "teacher_sequence_tokens": 2_295,
            "student_sequence_tokens": 1_335,
            "teacher_cached_chunks": 5,
        },
    },
    8_192: {
        "kl_tokens": 256,
        "lm_head_chunk_size": 16,
        "max_target_sequence_tokens": 8_192,
        "max_teacher_sequence_tokens": 262_144,
        "max_history_tokens": 262_144,
        "selector": {
            "line_index": 189,
            "example_id": "41",
            "endpoint_event_index": 5,
            "selected_event_indices": [0, 1, 2, 3, 4, 5],
            "raw_line_sha256": (
                "58c785f9a127cd929cf89905dfef4cdad3b2851ea2d5219bdc0cd5764d73dfa8"
            ),
            "teacher_prefix_ids_sha256": (
                "279f00288018a0ebe38e2ed048652bfa90383f5c5d0d8c410816fb90c10c4374"
            ),
            "target_ids_sha256": (
                "a7e40cb27f7c16c03334a06d9b2673c55b2747578ffd17aa9ddbe0714cd4c891"
            ),
            "logical_teacher_forcing_ids_sha256": (
                "6e641570d495e36e20ec1b6db2da2d4c3d3581d3bbb997ea6601351cf1d3e63b"
            ),
            "cached_teacher_input_ids_sha256": (
                "32145d05af0bf1c73da27c39d0e14d5d50e715b347a766c4558d3f2be77c0e8b"
            ),
            "cached_teacher_chunk_ranges": [[0, 256], [256, 8_436]],
            "teacher_sequence_tokens": 8_436,
            "student_sequence_tokens": 2_361,
            "teacher_cached_chunks": 2,
        },
    },
}


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    try:
        return _canonical_json(actual) == _canonical_json(expected)
    except (TypeError, ValueError):
        return False


def _validate_run_config(
    path: Path,
    *,
    artifact_path: Path,
    artifact: dict[str, Any],
    result: dict[str, Any],
    expected_profile: dict[str, Any],
    expected_data: str,
    data_sha256: str,
    expected_target: str,
    target_manifest_sha256: str,
    target_content_identity: dict[str, Any],
    implementation_identity: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError("cached-teacher parity run config is missing")
    run_config = _read_json_object(path, description="cached-teacher parity run config")
    missing_metadata = _RUN_CONFIG_METADATA_FIELDS.difference(run_config)
    if missing_metadata:
        raise ValueError(
            "cached-teacher parity run config metadata is incomplete: "
            f"{sorted(missing_metadata)}"
        )
    configured_output = run_config.get("output_dir")
    if not isinstance(configured_output, str) or str(
        Path(configured_output).expanduser().resolve()
    ) != str(artifact_path.parent):
        raise ValueError("cached-teacher parity run config output directory mismatch")
    run_contract = {
        key: value
        for key, value in run_config.items()
        if key not in _RUN_CONFIG_METADATA_FIELDS
    }
    recomputed_run_sha = _canonical_sha256(run_contract)
    recorded_run_sha = run_config.get("run_contract_sha256")
    if not _is_sha256(recorded_run_sha) or recorded_run_sha != recomputed_run_sha:
        raise ValueError("cached-teacher parity run-contract digest mismatch")
    if artifact.get("run_contract_sha256") != recomputed_run_sha:
        raise ValueError("cached-teacher parity artifact is not bound to run config")
    if result.get("run_contract_sha256") != recomputed_run_sha:
        raise ValueError("cached-teacher parity result is not bound to run config")

    selector = expected_profile["selector"]
    if not _strict_json_equal(
        run_config.get("cached_teacher_parity_selector"), selector
    ):
        raise ValueError("cached-teacher parity run config selector mismatch")
    if run_config.get("implementation_identity") != implementation_identity:
        raise ValueError("cached-teacher parity run config implementation is stale")
    if run_config.get("target_family") != "qwen38":
        raise ValueError("cached-teacher parity run config target family mismatch")
    if (
        run_config.get("data") != expected_data
        or run_config.get("data_sha256") != data_sha256
    ):
        raise ValueError("cached-teacher parity run config data identity mismatch")
    if (
        run_config.get("target_model") != expected_target
        or run_config.get("target_text_snapshot_manifest_sha256")
        != target_manifest_sha256
        or run_config.get("target_content_identity") != target_content_identity
    ):
        raise ValueError("cached-teacher parity run config target identity mismatch")
    if run_config.get("kl_tokens_per_event") != expected_profile["kl_tokens"]:
        raise ValueError("cached-teacher parity run config KL width mismatch")
    contract_args = run_config.get("args")
    if not isinstance(contract_args, dict):
        raise ValueError("cached-teacher parity run config arguments are missing")
    expected_args = {
        "verify_cached_teacher": True,
        "curriculum_mode": "recursive",
        "epochs": 1,
        "grad_accumulation": 1,
        "teacher_prefill_chunk_tokens": artifact["teacher_prefill_chunk_tokens"],
        "kl_tokens": expected_profile["kl_tokens"],
        "lm_head_chunk_size": expected_profile["lm_head_chunk_size"],
        "max_target_sequence_tokens": expected_profile["max_target_sequence_tokens"],
        "max_teacher_sequence_tokens": expected_profile["max_teacher_sequence_tokens"],
        "max_history_tokens": expected_profile["max_history_tokens"],
    }
    if any(
        not _strict_json_equal(contract_args.get(name), expected)
        for name, expected in expected_args.items()
    ):
        raise ValueError("cached-teacher parity run config probe arguments mismatch")
    return run_config, recomputed_run_sha


def _expected_success_sentinel(
    *,
    artifact_path: Path,
    result_path: Path,
    run_config_path: Path,
    run_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "schema": SUCCESS_SENTINEL_SCHEMA,
        "status": "ready",
        "artifact_sha256": _sha256(artifact_path),
        "result_sha256": _sha256(result_path),
        "run_config_sha256": _sha256(run_config_path),
        "run_contract_sha256": run_contract_sha256,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably publish a JSON object without exposing a partial sentinel."""

    path = path.expanduser().resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"sentinel parent directory is missing: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def validate_cached_teacher_parity_artifacts(
    artifacts: Sequence[Path],
    *,
    required_chunk_tokens: Sequence[int],
    data_path: Path,
    data_sha256: str,
    target_model: Path,
    target_manifest_sha256: str,
    target_content_identity: dict[str, Any],
    implementation_identity: dict[str, Any],
    maximum_output_distribution_kl: float,
    minimum_top1_agreement: float,
    require_success_sentinel: bool = True,
) -> list[dict[str, Any]]:
    """Fail closed unless artifacts prove the exact current runtime contract."""

    if (
        not _is_finite_number(maximum_output_distribution_kl)
        or maximum_output_distribution_kl < 0
        or not _is_finite_number(minimum_top1_agreement)
        or not 0 <= minimum_top1_agreement <= 1
    ):
        raise ValueError("cached-teacher parity thresholds are invalid")
    if not isinstance(require_success_sentinel, bool):
        raise TypeError("require_success_sentinel must be bool")
    required_values: list[int] = []
    for value in required_chunk_tokens:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("required parity chunk sizes must be positive integers")
        required_values.append(value)
    required = tuple(sorted(required_values))
    if not required:
        raise ValueError("required parity chunk sizes must be positive integers")
    if len(set(required)) != len(required):
        raise ValueError("required parity chunk sizes must be unique")
    unsupported = set(required).difference(CANONICAL_PARITY_PROFILES)
    if unsupported:
        raise ValueError(
            f"unsupported cached-teacher parity chunks: {sorted(unsupported)}"
        )
    resolved_artifacts = tuple(Path(path).expanduser().resolve() for path in artifacts)
    if len(resolved_artifacts) != len(required):
        raise ValueError(
            "one cached-teacher parity artifact is required per chunk size"
        )
    if len(set(resolved_artifacts)) != len(resolved_artifacts):
        raise ValueError("cached-teacher parity artifacts must be unique")
    expected_data = str(data_path.expanduser().resolve())
    expected_target = str(target_model.expanduser().resolve())
    receipts: list[dict[str, Any]] = []
    observed_chunks: list[int] = []
    for path in resolved_artifacts:
        if not path.is_file():
            raise FileNotFoundError(
                f"cached-teacher parity artifact is missing: {path}"
            )
        artifact = _read_json_object(path, description="cached-teacher parity artifact")
        artifact_format_version = artifact.get("format_version")
        if (
            isinstance(artifact_format_version, bool)
            or not isinstance(artifact_format_version, int)
            or artifact_format_version != PARITY_FORMAT_VERSION
            or artifact.get("schema") != PARITY_SCHEMA
            or artifact.get("status") != "passed"
        ):
            raise ValueError("cached-teacher parity artifact schema/status mismatch")
        if artifact.get("direction") != "KL(monolithic_teacher||cached_teacher)":
            raise ValueError("cached-teacher parity KL direction mismatch")
        if artifact.get("hidden_difference_is_diagnostic_only") is not True:
            raise ValueError("cached-teacher hidden-difference contract mismatch")
        if artifact.get("implementation_identity") != implementation_identity:
            raise ValueError("cached-teacher parity implementation identity is stale")
        if artifact.get("target_family") != "qwen38":
            raise ValueError("cached-teacher parity target family mismatch")
        if (
            artifact.get("data") != expected_data
            or artifact.get("data_sha256") != data_sha256
        ):
            raise ValueError("cached-teacher parity data identity mismatch")
        if (
            artifact.get("target_model") != expected_target
            or artifact.get("target_text_snapshot_manifest_sha256")
            != target_manifest_sha256
        ):
            raise ValueError("cached-teacher parity target identity mismatch")
        if artifact.get("target_content_identity") != target_content_identity:
            raise ValueError("cached-teacher parity target content is stale")
        artifact_maximum_kl = artifact.get("maximum_output_distribution_kl")
        artifact_minimum_agreement = artifact.get("minimum_top1_agreement")
        if (
            not _is_finite_number(artifact_maximum_kl)
            or artifact_maximum_kl != maximum_output_distribution_kl
        ):
            raise ValueError("cached-teacher parity KL threshold mismatch")
        if (
            not _is_finite_number(artifact_minimum_agreement)
            or artifact_minimum_agreement != minimum_top1_agreement
        ):
            raise ValueError("cached-teacher parity top-1 threshold mismatch")
        chunk_tokens = artifact.get("teacher_prefill_chunk_tokens")
        if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int):
            raise ValueError("cached-teacher parity chunk size is invalid")
        profile = CANONICAL_PARITY_PROFILES.get(chunk_tokens)
        if profile is None:
            raise ValueError("cached-teacher parity chunk size is not canonical")
        observed_chunks.append(chunk_tokens)
        kl_tokens = artifact.get("kl_tokens")
        lm_head_chunk_size = artifact.get("lm_head_chunk_size")
        if (
            isinstance(kl_tokens, bool)
            or not isinstance(kl_tokens, int)
            or kl_tokens != profile["kl_tokens"]
        ):
            raise ValueError("cached-teacher parity KL token width is not canonical")
        if (
            isinstance(lm_head_chunk_size, bool)
            or not isinstance(lm_head_chunk_size, int)
            or lm_head_chunk_size != profile["lm_head_chunk_size"]
        ):
            raise ValueError("cached-teacher parity LM-head chunk is not canonical")
        selector = artifact.get("selector")
        if not _strict_json_equal(selector, profile["selector"]):
            raise ValueError("cached-teacher parity fixed probe mismatch")
        rank_records = artifact.get("rank_records")
        if not isinstance(rank_records, list) or len(rank_records) != 2:
            raise ValueError("cached-teacher parity rank evidence is missing")
        ranks = [
            record.get("rank") if isinstance(record, dict) else None
            for record in rank_records
        ]
        if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks):
            raise ValueError("cached-teacher parity rank evidence is invalid")
        if set(ranks) != {0, 1}:
            raise ValueError("cached-teacher parity rank evidence is incomplete")
        records_by_rank = {record["rank"]: record for record in rank_records}
        events_by_rank = [records_by_rank[rank].get("events") for rank in (0, 1)]
        if any(not isinstance(events, list) or not events for events in events_by_rank):
            raise ValueError("cached-teacher parity rank events are missing")
        if not _strict_json_equal(events_by_rank[0], events_by_rank[1]):
            raise ValueError("cached-teacher parity metrics differ across ranks")
        events = events_by_rank[0]
        if len(events) != 1:
            raise ValueError("cached-teacher parity rank event set is not canonical")
        endpoint_records = [
            event
            for event in events
            if isinstance(event, dict)
            and event.get("line_index") == selector["line_index"]
            and event.get("event_index") == selector["endpoint_event_index"]
        ]
        if len(endpoint_records) != 1:
            raise ValueError("cached-teacher parity endpoint evidence is not unique")
        endpoint = endpoint_records[0]
        endpoint_contract = {
            "line_index": selector["line_index"],
            "example_id": selector["example_id"],
            "event_index": selector["endpoint_event_index"],
            "teacher_sequence_tokens": selector["teacher_sequence_tokens"],
            "student_sequence_tokens": selector["student_sequence_tokens"],
            "teacher_cached_chunks": selector["teacher_cached_chunks"],
        }
        if any(
            not _strict_json_equal(endpoint.get(field), expected)
            for field, expected in endpoint_contract.items()
        ):
            raise ValueError(
                "cached-teacher parity endpoint does not match fixed probe"
            )
        for field in (
            "hidden_max_abs_difference",
            "hidden_mean_abs_difference",
            "output_distribution_kl",
            "top1_agreement",
        ):
            if not _is_finite_number(endpoint.get(field)):
                raise ValueError(f"cached-teacher parity {field} is non-finite")
        output_kl = endpoint["output_distribution_kl"]
        agreement = endpoint["top1_agreement"]
        cached_chunks = endpoint["teacher_cached_chunks"]
        if output_kl < -1e-6 or output_kl > maximum_output_distribution_kl:
            raise ValueError("cached-teacher parity KL exceeds its contract")
        if agreement < minimum_top1_agreement or agreement > 1:
            raise ValueError(
                "cached-teacher parity top-1 agreement violates its contract"
            )
        planned_ranges = selector["cached_teacher_chunk_ranges"]
        if not isinstance(planned_ranges, list) or not planned_ranges:
            raise ValueError("cached-teacher parity chunk plan is missing")
        expected_start = 0
        for index, bounds in enumerate(planned_ranges):
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in bounds
                )
            ):
                raise ValueError("cached-teacher parity chunk plan is invalid")
            start, stop = bounds
            if (
                start != expected_start
                or stop <= start
                or stop - start > chunk_tokens
                or (index < len(planned_ranges) - 1 and stop % 64 != 0)
            ):
                raise ValueError("cached-teacher parity chunk plan is inconsistent")
            expected_start = stop
        if expected_start != selector["teacher_sequence_tokens"]:
            raise ValueError("cached-teacher parity chunk plan width is inconsistent")
        protected_start = (
            selector["teacher_sequence_tokens"] - artifact["kl_tokens"] - 1
        )
        if planned_ranges[-1][0] > protected_start:
            raise ValueError("cached-teacher parity prediction suffix is split")
        expected_ranges = [
            [start, stop]
            for start, stop in cached_teacher_chunk_ranges(
                selector["teacher_sequence_tokens"],
                chunk_tokens=chunk_tokens,
                protected_suffix_tokens=artifact["kl_tokens"] + 1,
            )
        ]
        if not _strict_json_equal(planned_ranges, expected_ranges):
            raise ValueError("cached-teacher parity chunk plan differs from planner")
        expected_chunks = len(expected_ranges)
        if cached_chunks < 2 or cached_chunks != expected_chunks:
            raise ValueError("cached-teacher parity cached-chunk count is inconsistent")

        result_path = path.with_name("result.json")
        if not result_path.is_file():
            raise FileNotFoundError("cached-teacher parity smoke result is missing")
        result_sha256 = _sha256(result_path)
        if artifact.get("result_sha256") != result_sha256:
            raise ValueError("cached-teacher parity smoke result digest mismatch")
        result = _read_json_object(
            result_path, description="cached-teacher parity smoke result"
        )
        health = result.get("last_update")
        if (
            result.get("status") != "completed"
            or isinstance(result.get("global_step"), bool)
            or result.get("global_step") != 1
            or not isinstance(health, dict)
            or not all(
                _is_finite_number(health.get(name))
                for name in ("training_kl", "gradient_norm")
            )
            or result.get("parameters_finite_after_step") is not True
        ):
            raise ValueError("cached-teacher parity smoke health contract failed")

        run_config_path = path.with_name("run_config.json")
        _run_config, run_sha = _validate_run_config(
            run_config_path,
            artifact_path=path,
            artifact=artifact,
            result=result,
            expected_profile=profile,
            expected_data=expected_data,
            data_sha256=data_sha256,
            expected_target=expected_target,
            target_manifest_sha256=target_manifest_sha256,
            target_content_identity=target_content_identity,
            implementation_identity=implementation_identity,
        )

        sentinel_path = path.with_name(SUCCESS_SENTINEL_NAME)
        sentinel_sha256: str | None = None
        if require_success_sentinel:
            if not sentinel_path.is_file():
                raise FileNotFoundError(
                    "cached-teacher parity success sentinel is missing"
                )
            sentinel = _read_json_object(
                sentinel_path,
                description="cached-teacher parity success sentinel",
            )
            expected_sentinel = _expected_success_sentinel(
                artifact_path=path,
                result_path=result_path,
                run_config_path=run_config_path,
                run_contract_sha256=run_sha,
            )
            if not _strict_json_equal(sentinel, expected_sentinel):
                raise ValueError("cached-teacher parity success sentinel mismatch")
            sentinel_sha256 = _sha256(sentinel_path)

        receipt = {
            "path": str(path),
            "sha256": _sha256(path),
            "result_path": str(result_path),
            "result_sha256": result_sha256,
            "run_config_path": str(run_config_path),
            "run_config_sha256": _sha256(run_config_path),
            "teacher_prefill_chunk_tokens": chunk_tokens,
            "kl_tokens": kl_tokens,
            "lm_head_chunk_size": lm_head_chunk_size,
            "selector": selector,
            "endpoint": endpoint,
            "run_contract_sha256": run_sha,
        }
        if sentinel_sha256 is not None:
            receipt.update(
                {
                    "success_sentinel_path": str(sentinel_path),
                    "success_sentinel_sha256": sentinel_sha256,
                }
            )
        receipts.append(receipt)
    if tuple(sorted(observed_chunks)) != required:
        raise ValueError(
            f"cached-teacher parity chunks mismatch: expected {required}, "
            f"got {tuple(sorted(observed_chunks))}"
        )
    return sorted(receipts, key=lambda item: item["teacher_prefill_chunk_tokens"])


def _write_success_sentinel(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    artifact_path = Path(receipt["path"])
    canonical_path = artifact_path.with_name(SUCCESS_SENTINEL_NAME)
    path = path.expanduser().resolve()
    if path != canonical_path:
        raise ValueError(
            f"success sentinel must be written to the artifact sibling {canonical_path}"
        )
    result_path = Path(receipt["result_path"])
    run_config_path = Path(receipt["run_config_path"])
    expected_file_hashes = {
        artifact_path: receipt["sha256"],
        result_path: receipt["result_sha256"],
        run_config_path: receipt["run_config_sha256"],
    }
    if any(
        _sha256(file_path) != digest
        for file_path, digest in expected_file_hashes.items()
    ):
        raise RuntimeError(
            "cached-teacher parity evidence changed before sentinel issue"
        )
    sentinel = _expected_success_sentinel(
        artifact_path=artifact_path,
        result_path=result_path,
        run_config_path=run_config_path,
        run_contract_sha256=receipt["run_contract_sha256"],
    )
    _atomic_json(path, sentinel)
    if (
        _read_json_object(path, description="cached-teacher parity success sentinel")
        != sentinel
    ):
        raise RuntimeError("cached-teacher parity success sentinel publish failed")
    return sentinel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parser.add_argument(
        "--required-chunk-tokens", type=int, action="append", required=True
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--maximum-output-distribution-kl", type=float, default=1e-4)
    parser.add_argument("--minimum-top1-agreement", type=float, default=1.0)
    parser.add_argument(
        "--write-success-sentinel",
        type=Path,
        help=(
            "After validating one completed smoke without a sentinel, atomically "
            "publish its canonical FORMAL_READY.json."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from context_compression.train_qwen36_recursive_compressor import (
        _implementation_identity,
        _sha256 as trainer_sha256,
        _target_content_identity,
    )

    if args.write_success_sentinel is not None and len(args.artifact) != 1:
        raise ValueError("success-sentinel issue requires exactly one artifact")
    receipts = validate_cached_teacher_parity_artifacts(
        args.artifact,
        required_chunk_tokens=args.required_chunk_tokens,
        data_path=args.data,
        data_sha256=trainer_sha256(args.data),
        target_model=args.target_model,
        target_manifest_sha256=trainer_sha256(
            args.target_model / "text_snapshot_manifest.json"
        ),
        target_content_identity=_target_content_identity(args.target_model),
        implementation_identity=_implementation_identity(),
        maximum_output_distribution_kl=args.maximum_output_distribution_kl,
        minimum_top1_agreement=args.minimum_top1_agreement,
        require_success_sentinel=args.write_success_sentinel is None,
    )
    response: dict[str, Any] = {"status": "passed", "receipts": receipts}
    if args.write_success_sentinel is not None:
        response["success_sentinel"] = _write_success_sentinel(
            args.write_success_sentinel, receipts[0]
        )
        response["status"] = "success_sentinel_written"
    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_PARITY_PROFILES",
    "LINEAR_ATTENTION_CHUNK_ALIGNMENT",
    "PARITY_FORMAT_VERSION",
    "PARITY_SCHEMA",
    "SUCCESS_SENTINEL_NAME",
    "SUCCESS_SENTINEL_SCHEMA",
    "cached_teacher_chunk_ranges",
    "validate_cached_teacher_parity_artifacts",
]
