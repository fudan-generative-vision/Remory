"""Fail-closed preflight for the official Terminal-Bench 2.1 v0.5.0 run.

The agent, checkpoint, PRoot, and live SGLang contracts are shared with the
Terminal-Bench 2.0 v0.5.0 preflight. This module changes only the immutable
dataset identity and requires the complete five-attempt official protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from harbor.models.job.config import JobConfig
from harbor.registry.client.package import PackageDatasetClient

from terminal_bench_eval import tb20_v050_preflight as common


PREFLIGHT_SCHEMA = "terminal_bench_r2eact_v050_tb21_preflight_v1"
DATASET_NAME = "terminal-bench/terminal-bench-2-1"
DATASET_DIGEST = (
    "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
)
DATASET_VERSION_ID = "f92eea12-ff70-4d30-ace0-003abf294998"
DATASET_TASK_COUNT = 89

OFFICIAL_ATTEMPTS = common.OFFICIAL_ATTEMPTS
CONCURRENT_TRIALS = common.CONCURRENT_TRIALS
AGENT_TIMEOUT_SECONDS = common.AGENT_TIMEOUT_SECONDS
MAX_NEW_TOKENS = common.MAX_NEW_TOKENS
WORKING_MEMORY_TURNS = common.WORKING_MEMORY_TURNS
TERMINUS_R2EACT_V050_IMPORT_PATH = common.TERMINUS_R2EACT_V050_IMPORT_PATH
TERMINUS_R2EACT_V050_NAME = common.TERMINUS_R2EACT_V050_NAME
TERMINUS_R2EACT_V050_VERSION = common.TERMINUS_R2EACT_V050_VERSION
TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST = common.TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST
TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS = (
    common.TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS
)
QWEN38_MODEL_NAME = common.QWEN38_MODEL_NAME
QWEN38_TARGET_FAMILY = common.QWEN38_TARGET_FAMILY
QWEN38_CHECKPOINT_SCHEMA = common.QWEN38_CHECKPOINT_SCHEMA

runtime_attestation = common.runtime_attestation
_attest_runtime = common._attest_runtime
_checkpoint_contract_report = common._checkpoint_contract_report
_proot_report = common._proot_report
_runtime_spec = common._runtime_spec


def _validate_job_config(path: Path) -> JobConfig:
    """Validate the exact complete Terminal-Bench 2.1 official protocol."""

    config = common._validate_job_config_for_dataset(
        path,
        dataset_name=DATASET_NAME,
        dataset_digest=DATASET_DIGEST,
        dataset_label="TB2.1",
    )
    if config.n_attempts != OFFICIAL_ATTEMPTS:
        raise RuntimeError("TB2.1 official config must use five attempts per task")
    dataset = config.datasets[0]
    if dataset.task_names or dataset.exclude_task_names or dataset.n_tasks is not None:
        raise RuntimeError("TB2.1 official config must select the complete dataset")
    return config


def _scope_report(config: JobConfig) -> dict[str, Any]:
    report = common._scope_report(config)
    if report.get("official_task_count") != DATASET_TASK_COUNT:
        raise RuntimeError("shared preflight reports an unexpected task count")
    return report


async def _dataset_report() -> dict[str, Any]:
    reference = f"{DATASET_NAME}@{DATASET_DIGEST}"
    metadata = await PackageDatasetClient().get_dataset_metadata(reference)
    if metadata.version != DATASET_DIGEST:
        raise RuntimeError("Harbor resolved TB2.1 to an unexpected digest")
    if len(metadata.task_ids) != DATASET_TASK_COUNT:
        raise RuntimeError(
            f"pinned TB2.1 has {len(metadata.task_ids)} tasks, expected 89"
        )
    observed_version_id = str(metadata.dataset_version_id)
    if observed_version_id != DATASET_VERSION_ID:
        raise RuntimeError("Harbor resolved TB2.1 to an unexpected dataset version")
    return {
        "reference": reference,
        "resolved_digest": metadata.version,
        "task_count": len(metadata.task_ids),
        "dataset_version_id": observed_version_id,
    }


async def build_report(
    config_path: Path,
    sglang_url: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = _validate_job_config(config_path)
    spec = _runtime_spec(config, sglang_url)
    dataset, proot, checkpoint, runtime = await asyncio.gather(
        _dataset_report(),
        asyncio.to_thread(_proot_report),
        asyncio.to_thread(_checkpoint_contract_report, config),
        asyncio.to_thread(_attest_runtime, spec),
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "ok",
        "config_path": str(config_path),
        "protocol": {
            "dataset_name": DATASET_NAME,
            "dataset_digest": DATASET_DIGEST,
            **_scope_report(config),
            "concurrent_trials": CONCURRENT_TRIALS,
            "agent_timeout_seconds": AGENT_TIMEOUT_SECONDS,
            "score_aggregation": "mean_pass_at_1_across_five_attempts",
        },
        "dataset": dataset,
        "agent": {
            "name": TERMINUS_R2EACT_V050_NAME,
            "version": TERMINUS_R2EACT_V050_VERSION,
            "import_path": TERMINUS_R2EACT_V050_IMPORT_PATH,
            "model_name": config.agents[0].model_name,
            "working_memory_turns": WORKING_MEMORY_TURNS,
            "max_turns": None,
            "max_new_tokens": MAX_NEW_TOKENS,
            "remaining_context_output_ceiling_tokens": MAX_NEW_TOKENS,
            "expected_target_family": QWEN38_TARGET_FAMILY,
            "expected_checkpoint_schema": QWEN38_CHECKPOINT_SCHEMA,
            "controller_owned_log_protection": {
                "include_logs": list(TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST),
                "exclude_logs": sorted(TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS),
            },
        },
        "checkpoint": checkpoint,
        "runtime_attestation": runtime,
        "proot": proot,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sglang-url")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = asyncio.run(build_report(args.config, args.sglang_url))
    if args.output is not None:
        common._atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PREFLIGHT_SCHEMA",
    "TERMINUS_R2EACT_V050_IMPORT_PATH",
    "_checkpoint_contract_report",
    "_validate_job_config",
    "build_report",
    "main",
]
