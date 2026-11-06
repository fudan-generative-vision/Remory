"""Fail-closed preflight for the TB2.1 Codex CLI + r^2eAct 0.1.0 pool."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harbor.models.job.config import JobConfig

from terminal_bench_eval import tb20_v050_preflight as common
from terminal_bench_eval import tb21_v050_preflight as tb21
from terminal_bench_eval import tb21_v053_preflight as v053
from terminal_bench_eval.codex_r2eact_agent import (
    CODEX_R2EACT_POC_VERSION,
    PINNED_CODEX_BINARY,
    PINNED_CODEX_SHA256,
    PINNED_CODEX_VERSION,
)
from terminal_bench_eval.codex_r2eact_artifacts import (
    CONTROLLER_ARTIFACT_DIRNAME,
)
from terminal_bench_eval.codex_r2eact_gateway import (
    CODEX_R2EACT_WEB_SEARCH_MCP_SERVER,
    CODEX_R2EACT_WEB_SEARCH_MCP_TOOL,
    CODEX_R2EACT_WEB_SEARCH_NAMESPACE,
    CODEX_R2EACT_WEB_SEARCH_ENDPOINT,
    CODEX_R2EACT_WEB_SEARCH_POLICY,
    CODEX_R2EACT_VISION_MAX_ACTIVE_IMAGE_BYTES,
    CODEX_R2EACT_VISION_MAX_ACTIVE_IMAGES,
    CODEX_R2EACT_VISION_MAX_IMAGE_BYTES,
    CODEX_R2EACT_VISION_MAX_IMAGES_PER_OUTPUT,
    CODEX_R2EACT_VISION_POLICY,
    CODEX_R2EACT_OVERSIZED_ANSWER_RECOVERY_POLICY,
    CODEX_R2EACT_OVERSIZED_ANSWER_RETAIN_BODY_TOKENS,
    CODEX_R2EACT_OVERSIZED_ANSWER_THRESHOLD_TOKENS,
    GATEWAY_ROUNDS_FILENAME,
    GATEWAY_ROUND_TRACE_SCHEMA,
    GATEWAY_IMAGE_RECEIPTS_FILENAME,
    GATEWAY_IMAGE_RECEIPT_SCHEMA,
    GATEWAY_TOOL_RESPONSES_FILENAME,
    GATEWAY_TOOL_RESPONSE_TRACE_SCHEMA,
    GATEWAY_TOOL_RESPONSE_TRACE_SCHEMA_V1,
    GATEWAY_WEB_SEARCH_RECEIPTS_FILENAME,
    GATEWAY_WEB_SEARCH_RECEIPT_SCHEMA,
    SUPPORTED_CODEX_TOOLS,
    SUPPORTED_CODEX_VISION_TOOLS,
    activate_pinned_host_rg,
    supported_codex_tool_names,
)
from terminal_bench_eval.harbor_unbounded_agent import (
    HARBOR_UNBOUNDED_CLI_MODULE,
    TB21_PER_TASK_AGENT_TIMEOUT_SECONDS,
    TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
    UNBOUNDED_AGENT_TIMEOUT_POLICY,
    agent_import_uses_unbounded_timeout,
)


PREFLIGHT_SCHEMA = "terminal_bench_codex_r2eact_v010_tb21_preflight_v1"
CODEX_R2EACT_IMPORT_PATH = "terminal_bench_eval.codex_r2eact_agent:CodexR2EActAgent"
CODEX_R2EACT_NAME = "codex-cli+r2eact"
SINGLE_ATTEMPT = 1
OFFICIAL_ATTEMPTS = common.OFFICIAL_ATTEMPTS
ALLOWED_ATTEMPTS = frozenset((SINGLE_ATTEMPT, OFFICIAL_ATTEMPTS))
WORKING_MEMORY_TURNS = 12
CHECKPOINT_1024_SERVING_EVICTION_POLICY = "checkpoint_1024_v1"
SERVING_EVICTION_POLICY = "qwen38_serving_8192_accumulation_canary_v1"
SERVING_EVICTION_ACCUMULATION_TOKENS = 8_192
EXPECTED_MODEL_PATH = Path(
    "/inspire/ssd/project/sais-bio/public/chenbaoyou/pretrained_models/"
    "Qwen/Qwen3.8-27B"
)
CODEX_LOG_ALLOWLIST = (
    "codex.txt",
    "sessions/**",
    "codex-r2eact-gateway-events.jsonl",
    "codex-r2eact-gateway-receipts.json",
    "memory.jsonl",
)
CODEX_VISION_LOG_ALLOWLIST = (
    *CODEX_LOG_ALLOWLIST,
    GATEWAY_IMAGE_RECEIPTS_FILENAME,
)
CODEX_VISION_WEB_PLAN_LOG_ALLOWLIST = (
    *CODEX_VISION_LOG_ALLOWLIST,
    GATEWAY_WEB_SEARCH_RECEIPTS_FILENAME,
)


def _require_exact(kwargs: Mapping[str, Any], key: str, expected: Any) -> None:
    observed = kwargs.get(key)
    if type(observed) is not type(expected) or observed != expected:
        raise RuntimeError(
            f"Codex r^2eAct 0.1.0 agent kwarg {key!r} must equal {expected!r}"
        )


def _validate_job_config_for_dataset(
    path: Path,
    *,
    dataset_name: str,
    dataset_digest: str,
    dataset_label: str,
    working_memory_turns: int = WORKING_MEMORY_TURNS,
    serving_eviction_policy: str = SERVING_EVICTION_POLICY,
    require_explicit_serving_eviction_policy: bool = False,
    include_logs: Sequence[str] = CODEX_LOG_ALLOWLIST,
) -> JobConfig:
    """Validate the shared Codex+r^2eAct contract for one pinned dataset.

    Dataset releases deliberately live in their own preflight modules.  Keeping
    the agent/runtime checks here prevents a newer Terminal-Bench release from
    silently drifting away from the already-attested Codex gateway contract.
    """

    raw = common._load_yaml(Path(path))
    try:
        config = JobConfig.model_validate(raw)
    except Exception as exc:
        raise RuntimeError("job config does not satisfy Harbor's schema") from exc

    if config.n_attempts not in ALLOWED_ATTEMPTS:
        raise RuntimeError(
            f"{dataset_label} Codex r^2eAct requires one diagnostic or five official "
            "attempts per task"
        )
    if config.n_concurrent_trials != common.CONCURRENT_TRIALS:
        raise RuntimeError("the Codex r^2eAct pool requires concurrency one")
    if len(config.datasets) != 1:
        raise RuntimeError("job config must contain exactly one dataset")
    dataset = config.datasets[0]
    if dataset.name != dataset_name or dataset.ref != dataset_digest:
        raise RuntimeError(
            f"job config does not pin the approved {dataset_label} digest"
        )
    if dataset.task_names or dataset.exclude_task_names or dataset.n_tasks is not None:
        raise RuntimeError(
            f"{dataset_label} Codex r^2eAct must select the complete dataset"
        )
    if len(config.agents) != 1:
        raise RuntimeError("job config must contain exactly one agent")

    agent = config.agents[0]
    if agent.import_path != CODEX_R2EACT_IMPORT_PATH:
        raise RuntimeError("job config must select CodexR2EActAgent 0.1.0")
    if agent.model_name != common.QWEN38_MODEL_NAME:
        raise RuntimeError("job config must select the pinned Qwen3.8 model")
    if agent.override_timeout_sec != common.AGENT_TIMEOUT_SECONDS:
        raise RuntimeError(
            "job config must retain the frozen 10,800-second compatibility value"
        )
    if not agent_import_uses_unbounded_timeout(agent.import_path):
        raise RuntimeError("Codex r^2eAct is not covered by the pool timeout wrapper")
    if config.timeout_multiplier != 1.0:
        raise RuntimeError("TB2.1 8h wrapper requires timeout_multiplier=1.0")
    if config.agent_timeout_multiplier is not None:
        raise RuntimeError("TB2.1 requires the default agent timeout multiplier")
    if agent.max_timeout_sec is not None:
        raise RuntimeError("TB2.1 8h wrapper must not configure a Harbor max timeout")
    if tuple(agent.include_logs) != tuple(include_logs):
        raise RuntimeError("Codex r^2eAct requires the exact artifact allowlist")
    if agent.exclude_logs:
        raise RuntimeError("Codex r^2eAct must not exclude its controller artifacts")

    kwargs = agent.kwargs
    exact = {
        "codex_binary": str(PINNED_CODEX_BINARY),
        "codex_binary_sha256": PINNED_CODEX_SHA256,
        "gateway_host": "127.0.0.1",
        "gateway_port": 0,
        "max_new_tokens": common.MAX_NEW_TOKENS,
        "remaining_context_output_ceiling_tokens": common.MAX_NEW_TOKENS,
        "working_memory_turns": working_memory_turns,
        "request_timeout": common.REQUEST_TIMEOUT_SECONDS,
        "capacity_pressure_reserve_tokens": 1_024,
        "process_lock": "/tmp/r2eact-terminal-agent-sglang.lock",
    }
    for key, expected in exact.items():
        _require_exact(kwargs, key, expected)
    if require_explicit_serving_eviction_policy and (
        "serving_eviction_policy" not in kwargs
    ):
        raise RuntimeError(
            "Codex r^2eAct 0.1.0 agent kwarg 'serving_eviction_policy' "
            "must be explicit"
        )
    observed_serving_policy = kwargs.get(
        "serving_eviction_policy", SERVING_EVICTION_POLICY
    )
    if (
        type(observed_serving_policy) is not str
        or observed_serving_policy != serving_eviction_policy
    ):
        raise RuntimeError(
            "Codex r^2eAct 0.1.0 agent kwarg 'serving_eviction_policy' "
            f"must equal {serving_eviction_policy!r}"
        )
    for key in ("specforge_root", "checkpoint", "corpus_dir", "codex_binary"):
        common._absolute_path(kwargs.get(key), field=key)
    process_lock = common._absolute_path(
        kwargs.get("process_lock"), field="process_lock"
    )
    if not process_lock.is_relative_to(Path("/tmp")):
        raise RuntimeError("process_lock must remain under /tmp")
    sglang_url = kwargs.get("sglang_url")
    if not isinstance(sglang_url, str) or not sglang_url.startswith(
        ("http://", "https://")
    ):
        raise RuntimeError("sglang_url must be a nonempty HTTP(S) origin")

    environment = config.environment
    if environment.import_path != common.PROOT_ENVIRONMENT_IMPORT_PATH:
        raise RuntimeError("job config must use the approved local PRoot provider")
    if environment.kwargs.get("guest_utility_profile") != (
        common.PROOT_GUEST_UTILITY_PROFILE
    ):
        raise RuntimeError("job config must use the exact PRoot guest utility profile")
    if environment.override_cpus != common.OFFICIAL_CPUS:
        raise RuntimeError("job config must record 32 CPUs")
    if environment.override_memory_mb != common.OFFICIAL_MEMORY_MB:
        raise RuntimeError("job config must record 48 GiB RAM")
    return config


def _validate_job_config(path: Path) -> JobConfig:
    return _validate_job_config_for_dataset(
        path,
        dataset_name=tb21.DATASET_NAME,
        dataset_digest=tb21.DATASET_DIGEST,
        dataset_label="TB2.1",
    )


def _attestation_config(config: JobConfig) -> JobConfig:
    """Add verifier-only expectations without forwarding them to the agent."""

    payload = copy.deepcopy(config.model_dump(mode="json", exclude_none=True))
    kwargs = payload["agents"][0].setdefault("kwargs", {})
    kwargs.update(
        {
            "expected_served_model": common.QWEN38_MODEL_NAME,
            "expected_target_family": common.QWEN38_TARGET_FAMILY,
            "expected_checkpoint_schema": common.QWEN38_CHECKPOINT_SCHEMA,
            "expected_model_path": str(EXPECTED_MODEL_PATH),
            "sync_token_ids_across_tp": True,
        }
    )
    return JobConfig.model_validate(payload)


def _checkpoint_contract_report(config: JobConfig) -> dict[str, Any]:
    return common._checkpoint_contract_report(_attestation_config(config))


def _runtime_spec(
    config: JobConfig, sglang_url: str | None
) -> common.runtime_attestation.RuntimeAttestationSpec:
    return common._runtime_spec(_attestation_config(config), sglang_url)


def _codex_binary_report(config: JobConfig) -> dict[str, Any]:
    kwargs = config.agents[0].kwargs
    binary = common._absolute_path(kwargs["codex_binary"], field="codex_binary")
    try:
        metadata = binary.lstat()
    except OSError as exc:
        raise RuntimeError("pinned Codex binary is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("pinned Codex binary must be a safe regular file")
    digest = hashlib.sha256()
    try:
        with binary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError("could not hash the pinned Codex binary") from exc
    observed = digest.hexdigest()
    if observed != PINNED_CODEX_SHA256:
        raise RuntimeError("pinned Codex binary checksum mismatch")
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("pinned Codex binary is not executable") from exc
    rendered = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or PINNED_CODEX_VERSION not in rendered.split():
        raise RuntimeError("pinned Codex binary version mismatch")
    return {
        "path": str(binary),
        "sha256": observed,
        "version": PINNED_CODEX_VERSION,
        "version_output": rendered,
        "size_bytes": metadata.st_size,
    }


def _gateway_report(
    *,
    working_memory_turns: int = WORKING_MEMORY_TURNS,
    serving_eviction_policy: str = SERVING_EVICTION_POLICY,
    serving_eviction_block_tokens: int | None = None,
    serving_eviction_accumulation_tokens: int | None = (
        SERVING_EVICTION_ACCUMULATION_TOKENS
    ),
    vision_enabled: bool = False,
    web_search_enabled: bool = False,
    update_plan_enabled: bool = False,
) -> dict[str, Any]:
    from terminal_bench_eval import codex_r2eact_gateway as gateway

    source = Path(str(gateway.__file__)).resolve()
    project_root = Path(__file__).resolve().parents[1]
    if (
        source.is_symlink()
        or not source.is_file()
        or not source.is_relative_to(project_root)
    ):
        raise RuntimeError("Codex r^2eAct gateway source is not project-owned")
    return {
        "module": str(source),
        "module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "wire_api": "responses",
        "web_search": (
            CODEX_R2EACT_WEB_SEARCH_POLICY if web_search_enabled else "disabled"
        ),
        "terminal_tools": list(
            supported_codex_tool_names(
                vision_enabled=vision_enabled,
                web_search_enabled=web_search_enabled,
                update_plan_enabled=update_plan_enabled,
            )
        ),
        "private_round_trace": {
            "filename": GATEWAY_ROUNDS_FILENAME,
            "schema": GATEWAY_ROUND_TRACE_SCHEMA,
            "storage": "host_agent_logs_hidden_controller_subdir",
            "storage_subdirectory": CONTROLLER_ARTIFACT_DIRNAME,
            "file_mode": "0600",
            "guest_visibility_during_agent_execution": False,
            "post_agent_provider_mirroring": "provider_dependent",
            "responses_wire_unchanged": True,
            "model_context_unchanged": True,
            "fields": [
                "episode_id",
                "round_index",
                "assistant_content",
                "reasoning",
                "visible_text",
                "tool_calls",
                "input_tokens",
                "output_tokens",
                "finish_reason",
                "sha256",
                "observed_at",
            ],
        },
        "private_tool_response_trace": {
            "filename": GATEWAY_TOOL_RESPONSES_FILENAME,
            "schema": GATEWAY_TOOL_RESPONSE_TRACE_SCHEMA,
            "legacy_schemas": [GATEWAY_TOOL_RESPONSE_TRACE_SCHEMA_V1],
            "storage": "host_agent_logs_hidden_controller_subdir",
            "storage_subdirectory": CONTROLLER_ARTIFACT_DIRNAME,
            "file_mode": "0600",
            "guest_visibility_during_agent_execution": False,
            "post_agent_provider_mirroring": "provider_dependent",
            "responses_wire_unchanged": True,
            "model_context_unchanged": True,
            "fields": [
                "episode_id",
                "round_index",
                "responses",
                "model_context_content",
                "model_context_content_sha256",
                "receipt_sha256",
                "observed_at",
            ],
        },
        "working_memory_turns": working_memory_turns,
        **(
            {
                "vision": {
                    "enabled": True,
                    "policy": CODEX_R2EACT_VISION_POLICY,
                    "initial_prompt_images": False,
                    "raw_suffix_images": True,
                    "pixel_lifetime": "until_owner_round_is_compressed",
                    "limits": {
                        "decoded_bytes_per_image": (
                            CODEX_R2EACT_VISION_MAX_IMAGE_BYTES
                        ),
                        "images_per_view_image_output": (
                            CODEX_R2EACT_VISION_MAX_IMAGES_PER_OUTPUT
                        ),
                        "active_images": CODEX_R2EACT_VISION_MAX_ACTIVE_IMAGES,
                        "active_decoded_bytes": (
                            CODEX_R2EACT_VISION_MAX_ACTIVE_IMAGE_BYTES
                        ),
                    },
                    "image_receipts": {
                        "filename": GATEWAY_IMAGE_RECEIPTS_FILENAME,
                        "schema": GATEWAY_IMAGE_RECEIPT_SCHEMA,
                        "storage": "host_agent_logs_hidden_controller_subdir",
                        "storage_subdirectory": CONTROLLER_ARTIFACT_DIRNAME,
                        "guest_visibility_during_agent_execution": False,
                        "post_agent_provider_mirroring": "provider_dependent",
                        "payload_storage": "sha256_mime_size_path_only",
                        "base64_persisted": False,
                    },
                }
            }
            if vision_enabled
            else {}
        ),
        **(
            {
                "web_search_contract": {
                    "enabled": True,
                    "orchestration": "codex_native_mcp",
                    "mcp_server": CODEX_R2EACT_WEB_SEARCH_MCP_SERVER,
                    "responses_namespace": CODEX_R2EACT_WEB_SEARCH_NAMESPACE,
                    "mcp_tool": CODEX_R2EACT_WEB_SEARCH_MCP_TOOL,
                    "model_visible_tool": "web_search",
                    "provider": "serper_search",
                    "endpoint": CODEX_R2EACT_WEB_SEARCH_ENDPOINT,
                    "endpoint_override": False,
                    "provider_native_web_search": "disabled",
                    "serper_scrape": False,
                    "credential_env": "SERPER_API_KEY",
                    "credential_present": bool(os.environ.get("SERPER_API_KEY")),
                    "receipt": {
                        "filename": GATEWAY_WEB_SEARCH_RECEIPTS_FILENAME,
                        "schema": GATEWAY_WEB_SEARCH_RECEIPT_SCHEMA,
                        "storage": "host_agent_logs_hidden_controller_subdir",
                        "storage_subdirectory": CONTROLLER_ARTIFACT_DIRNAME,
                        "guest_visibility_during_agent_execution": False,
                        "post_agent_provider_mirroring": "provider_dependent",
                        "api_key_persisted": False,
                    },
                }
            }
            if web_search_enabled
            else {}
        ),
        **(
            {
                "planning": {
                    "enabled": True,
                    "tool": "update_plan",
                    "implementation": "codex_native_function",
                    "mcp": False,
                }
            }
            if update_plan_enabled
            else {}
        ),
        "serving_eviction_policy": serving_eviction_policy,
        "serving_eviction_block_tokens": serving_eviction_block_tokens,
        "serving_eviction_accumulation_tokens": (serving_eviction_accumulation_tokens),
        "oversized_answer_recovery": {
            "enabled": True,
            "policy": CODEX_R2EACT_OVERSIZED_ANSWER_RECOVERY_POLICY,
            "soft_threshold_tokens": (CODEX_R2EACT_OVERSIZED_ANSWER_THRESHOLD_TOKENS),
            "retained_body_tokens": (CODEX_R2EACT_OVERSIZED_ANSWER_RETAIN_BODY_TOKENS),
            "retained_side": "head",
        },
    }


async def build_report(
    config_path: Path, sglang_url: str | None = None
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = _validate_job_config(config_path)
    spec = _runtime_spec(config, sglang_url)
    configured_url = str(config.agents[0].kwargs["sglang_url"]).rstrip("/")
    dataset, proot, checkpoint, runtime, transport, serving, binary, host_rg = (
        await asyncio.gather(
            tb21._dataset_report(),
            asyncio.to_thread(common._proot_report),
            asyncio.to_thread(_checkpoint_contract_report, config),
            asyncio.to_thread(common._attest_runtime, spec),
            asyncio.to_thread(v053._memory_capability_report, configured_url),
            asyncio.to_thread(v053._serving_eviction_api_report, config),
            asyncio.to_thread(_codex_binary_report, config),
            asyncio.to_thread(activate_pinned_host_rg),
        )
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "ok",
        "config_path": str(config_path),
        "protocol": {
            "dataset_name": tb21.DATASET_NAME,
            "dataset_digest": tb21.DATASET_DIGEST,
            **tb21._scope_report(config),
            "concurrent_trials": common.CONCURRENT_TRIALS,
            "agent_timeout_seconds": TB21_PER_TASK_AGENT_TIMEOUT_SECONDS,
            "agent_timeout_disabled": False,
            "agent_timeout_policy": TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
            "agent_timeout_source": "r2eact.tb21.uniform_per_task_8h",
            "timeout_multiplier": config.timeout_multiplier,
            "agent_timeout_multiplier": config.agent_timeout_multiplier,
            "agent_timeout_execution_boundary": HARBOR_UNBOUNDED_CLI_MODULE,
            "configured_compatibility_timeout_seconds": (common.AGENT_TIMEOUT_SECONDS),
            "total_trials": tb21.DATASET_TASK_COUNT * config.n_attempts,
            "score_aggregation": (
                "mean_pass_at_1_across_five_attempts"
                if config.n_attempts == OFFICIAL_ATTEMPTS
                else "single_attempt_diagnostic_only"
            ),
        },
        "dataset": dataset,
        "agent": {
            "name": CODEX_R2EACT_NAME,
            "version": CODEX_R2EACT_POC_VERSION,
            "import_path": CODEX_R2EACT_IMPORT_PATH,
            "model_name": config.agents[0].model_name,
            "artifacts": list(CODEX_LOG_ALLOWLIST),
            "gateway": _gateway_report(),
            "serving_eviction_policy": serving,
            "memory_transport_policy": transport,
            "codex_binary": binary,
            "host_memory_shell": {"rg": host_rg},
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
    "CODEX_R2EACT_IMPORT_PATH",
    "PREFLIGHT_SCHEMA",
    "_codex_binary_report",
    "_validate_job_config",
    "_validate_job_config_for_dataset",
    "build_report",
    "main",
]
