"""Fail-closed preflight for the TB2.1 v0.5.3 single-attempt run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harbor.models.job.config import JobConfig

from terminal_bench_eval import tb20_v050_preflight as common
from terminal_bench_eval import tb21_v050_preflight as tb21
from terminal_bench_eval import tb21_v052_preflight as v052_preflight
from terminal_bench_eval.harbor_unbounded_agent import (
    HARBOR_UNBOUNDED_CLI_MODULE,
    TB21_PER_TASK_AGENT_TIMEOUT_SECONDS,
    TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
    UNBOUNDED_AGENT_TIMEOUT_POLICY,
    agent_import_uses_unbounded_timeout,
)
from terminal_bench_eval.terminus_r2eact_agent_v052 import (
    TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY,
    TERMINUS_R2EACT_V052_NATIVE_HISTORY_POLICY,
    TERMINUS_R2EACT_V052_SGLANG_CACHE_POLICY,
)
from terminal_bench_eval.terminus_r2eact_agent_v053 import (
    TERMINUS_R2EACT_V053_NAME,
    TERMINUS_R2EACT_V053_OVERSIZED_ACTION_POLICY,
    TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RECOVERY_POLICY,
    TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RETAIN_BODY_TOKENS,
    TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_THRESHOLD_TOKENS,
    TERMINUS_R2EACT_V053_OVERSIZED_CONTROLLER_RESPONSE_POLICY,
    TERMINUS_R2EACT_V053_OVERSIZED_OBSERVATION_POLICY,
    TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS,
    TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY,
    TERMINUS_R2EACT_V053_VERSION,
    TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS,
)


PREFLIGHT_SCHEMA = "terminal_bench_r2eact_v053_tb21_preflight_v1"
SINGLE_ATTEMPT = 1
TERMINUS_R2EACT_V053_IMPORT_PATH = (
    "terminal_bench_eval.terminus_r2eact_agent_v053:TerminusR2EActAgentV053"
)


def _require_exact(kwargs: Mapping[str, Any], key: str, expected: Any) -> None:
    observed = kwargs.get(key)
    if type(observed) is not type(expected) or observed != expected:
        raise RuntimeError(f"v0.5.3 agent kwarg {key!r} must equal {expected!r}")


def _validate_job_config(path: Path) -> JobConfig:
    raw = common._load_yaml(Path(path))
    try:
        config = JobConfig.model_validate(raw)
    except Exception as exc:
        raise RuntimeError("job config does not satisfy Harbor's schema") from exc
    if config.n_attempts != SINGLE_ATTEMPT:
        raise RuntimeError("TB2.1 v0.5.3 config requires one attempt per task")
    if config.n_concurrent_trials != common.CONCURRENT_TRIALS:
        raise RuntimeError("the official TB2.1 config requires concurrency one")
    if len(config.datasets) != 1:
        raise RuntimeError("job config must contain exactly one dataset")
    dataset = config.datasets[0]
    if dataset.name != tb21.DATASET_NAME or dataset.ref != tb21.DATASET_DIGEST:
        raise RuntimeError("job config does not pin the approved TB2.1 digest")
    if dataset.task_names or dataset.exclude_task_names or dataset.n_tasks is not None:
        raise RuntimeError("TB2.1 official config must select the complete dataset")
    if len(config.agents) != 1:
        raise RuntimeError("job config must contain exactly one agent")
    agent = config.agents[0]
    if agent.import_path != TERMINUS_R2EACT_V053_IMPORT_PATH:
        raise RuntimeError("job config must select the v0.5.3 agent")
    if agent.model_name != common.QWEN38_MODEL_NAME:
        raise RuntimeError("job config must select the pinned Qwen3.8 model")
    if agent.override_timeout_sec != common.AGENT_TIMEOUT_SECONDS:
        raise RuntimeError(
            "job config must retain the frozen 10,800-second compatibility value"
        )
    if not agent_import_uses_unbounded_timeout(agent.import_path):
        raise RuntimeError("v0.5.3 agent is not covered by the pool timeout wrapper")
    if config.timeout_multiplier != 1.0:
        raise RuntimeError("TB2.1 8h wrapper requires timeout_multiplier=1.0")
    if config.agent_timeout_multiplier is not None:
        raise RuntimeError("TB2.1 requires the default agent timeout multiplier")
    if agent.max_timeout_sec is not None:
        raise RuntimeError("TB2.1 8h wrapper must not configure a Harbor max timeout")
    if tuple(agent.include_logs) != common.TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST:
        raise RuntimeError("v0.5.3 requires the exact guest-log allowlist")
    missing = common.TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS - set(agent.exclude_logs)
    if missing:
        raise RuntimeError(
            "v0.5.3 controller-owned log protection is incomplete: "
            + ", ".join(sorted(missing))
        )

    kwargs = agent.kwargs
    if kwargs.get("expected_served_model") != agent.model_name:
        raise RuntimeError("expected_served_model must equal model_name")
    exact = {
        "expected_target_family": common.QWEN38_TARGET_FAMILY,
        "expected_checkpoint_schema": common.QWEN38_CHECKPOINT_SCHEMA,
        "sync_token_ids_across_tp": True,
        "working_memory_turns": TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS,
        "qwen38_serving_8192_accumulation_canary": True,
        "max_new_tokens": common.MAX_NEW_TOKENS,
        "remaining_context_output_ceiling_tokens": common.MAX_NEW_TOKENS,
        "parser_name": "json",
        "enable_summarize": False,
        "collect_rollout_details": False,
        "record_terminal_session": False,
        "use_remaining_context_for_generation": True,
        "oversized_complete_round_eviction": True,
        "capacity_pressure_reserve_tokens": 1_024,
        "stagnation_cue_rounds": 6,
        "exact_repeat_cue_count": 3,
        "request_timeout": common.REQUEST_TIMEOUT_SECONDS,
        "post_compression_generation_policy": ("sglang_remaining_context_experimental"),
        "native_history_policy": TERMINUS_R2EACT_V052_NATIVE_HISTORY_POLICY,
        "sglang_cache_policy": TERMINUS_R2EACT_V052_SGLANG_CACHE_POLICY,
        "memory_transport_policy": TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY,
    }
    for key, expected in exact.items():
        _require_exact(kwargs, key, expected)
    if "max_turns" not in kwargs or kwargs.get("max_turns") is not None:
        raise RuntimeError("v0.5.3 must have no fixed max_turns limit")
    for key in ("specforge_root", "checkpoint", "corpus_dir", "expected_model_path"):
        common._absolute_path(kwargs.get(key), field=key)
    sglang_url = kwargs.get("sglang_url")
    if not isinstance(sglang_url, str) or not sglang_url.strip():
        raise RuntimeError("sglang_url must be a nonempty HTTP(S) origin")

    environment = config.environment
    if environment.import_path != common.PROOT_ENVIRONMENT_IMPORT_PATH:
        raise RuntimeError("job config must use the approved PRoot provider")
    if environment.kwargs.get("guest_utility_profile") != (
        common.PROOT_GUEST_UTILITY_PROFILE
    ):
        raise RuntimeError("job config must use the exact guest utility profile")
    if environment.override_cpus != common.OFFICIAL_CPUS:
        raise RuntimeError("job config must record 32 CPUs")
    if environment.override_memory_mb != common.OFFICIAL_MEMORY_MB:
        raise RuntimeError("job config must record 48 GiB RAM")
    return config


def _memory_capability_report(sglang_url: str) -> dict[str, Any]:
    """Reuse the inherited, frozen v0.5.2 TP-GPU server attestation."""

    return v052_preflight._memory_capability_report(sglang_url)


def _serving_eviction_api_report(
    config: JobConfig,
    *,
    requested_policy: str = TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY,
    accumulation_target_tokens: int | None = (
        TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS
    ),
) -> dict[str, Any]:
    """Attest the client-local SpecForge API that owns the v0.5.3 canary."""

    kwargs = config.agents[0].kwargs
    specforge_root = common._absolute_path(
        kwargs["specforge_root"], field="specforge_root"
    )
    source = specforge_root / "context_compression" / "grepseek_sglang_live.py"
    if source.is_symlink() or not source.is_file():
        raise RuntimeError("configured SpecForge has no safe canary implementation")
    root_text = str(specforge_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("context_compression.grepseek_sglang_live")
    except Exception as exc:
        raise RuntimeError(
            "could not import the SpecForge canary implementation"
        ) from exc
    module_file_value = getattr(module, "__file__", None)
    if not isinstance(module_file_value, str):
        raise RuntimeError("SpecForge canary module has no source path")
    module_file = Path(module_file_value).resolve()
    if not module_file.is_relative_to(specforge_root):
        raise RuntimeError(
            "SpecForge canary module was imported outside configured root"
        )

    checks = {
        "default_policy": (
            getattr(module, "CHECKPOINT_1024_SERVING_EVICTION_POLICY", None),
            "checkpoint_1024_v1",
        ),
        "canary_policy": (
            getattr(
                module,
                "QWEN38_8192_ACCUMULATION_SERVING_EVICTION_POLICY",
                None,
            ),
            TERMINUS_R2EACT_V053_SERVING_EVICTION_POLICY,
        ),
        "accumulation_target_tokens": (
            getattr(
                module,
                "QWEN38_SERVING_EVICTION_ACCUMULATION_TOKENS",
                None,
            ),
            TERMINUS_R2EACT_V053_SERVING_EVICTION_ACCUMULATION_TOKENS,
        ),
    }
    for label, (observed, expected) in checks.items():
        if type(observed) is not type(expected) or observed != expected:
            raise RuntimeError(f"SpecForge v0.5.3 {label} contract mismatch")
    allowed_policies = {checks["default_policy"][1], checks["canary_policy"][1]}
    if requested_policy not in allowed_policies:
        raise RuntimeError("requested serving-eviction policy is unsupported")
    if requested_policy == checks["default_policy"][1]:
        if accumulation_target_tokens is not None:
            raise RuntimeError(
                "checkpoint_1024_v1 must not declare an accumulation target"
            )
    elif accumulation_target_tokens != checks["accumulation_target_tokens"][1]:
        raise RuntimeError("8192 canary accumulation target mismatch")
    generator_type = getattr(module, "Qwen36SGLangTokenClockRolloutGenerator", None)
    configure = getattr(generator_type, "configure_serving_eviction_policy", None)
    if not isinstance(generator_type, type) or not callable(configure):
        raise RuntimeError("SpecForge lacks the v0.5.3 serving-eviction policy API")
    checkpoint_block_fields = (
        "eviction_block_tokens",
        "raw_working_memory_tokens",
        "checkpoint_eviction_block_tokens",
        "checkpoint_raw_working_memory_tokens",
    )
    checkpoint_block_tokens = {
        field: getattr(generator_type, field, None) for field in checkpoint_block_fields
    }
    if any(value != 1_024 for value in checkpoint_block_tokens.values()):
        raise RuntimeError("SpecForge checkpoint token-clock block contract mismatch")
    return {
        "attested": True,
        "module": str(module_file),
        "module_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
        "default_policy": "checkpoint_1024_v1",
        "requested_policy": requested_policy,
        "accumulation_target_tokens": accumulation_target_tokens,
        "configuration_api": "configure_serving_eviction_policy",
        "episode_snapshot_required": True,
        "checkpoint_block_tokens": checkpoint_block_tokens,
    }


async def build_report(
    config_path: Path,
    sglang_url: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = _validate_job_config(config_path)
    spec = common._runtime_spec(config, sglang_url)
    configured_url = str(config.agents[0].kwargs["sglang_url"]).rstrip("/")
    serving_eviction = _serving_eviction_api_report(config)
    dataset, proot, checkpoint, runtime, transport = await asyncio.gather(
        tb21._dataset_report(),
        asyncio.to_thread(common._proot_report),
        asyncio.to_thread(common._checkpoint_contract_report, config),
        asyncio.to_thread(common._attest_runtime, spec),
        asyncio.to_thread(_memory_capability_report, configured_url),
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
            "configured_compatibility_timeout_seconds": common.AGENT_TIMEOUT_SECONDS,
            "score_aggregation": "single_attempt_diagnostic_only",
        },
        "dataset": dataset,
        "agent": {
            "name": TERMINUS_R2EACT_V053_NAME,
            "version": TERMINUS_R2EACT_V053_VERSION,
            "import_path": TERMINUS_R2EACT_V053_IMPORT_PATH,
            "model_name": config.agents[0].model_name,
            "working_memory_turns": TERMINUS_R2EACT_V053_WORKING_MEMORY_TURNS,
            "qwen38_serving_8192_accumulation_canary": True,
            "serving_eviction_policy": serving_eviction,
            "oversized_answer_recovery": {
                "enabled": True,
                "policy": TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RECOVERY_POLICY,
                "soft_threshold_tokens": (
                    TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_THRESHOLD_TOKENS
                ),
                "retained_body_tokens": (
                    TERMINUS_R2EACT_V053_OVERSIZED_ANSWER_RETAIN_BODY_TOKENS
                ),
                "retained_side": "head",
                "scope": "finished_eos_nonaction_answer_only",
                "valid_action_policy": TERMINUS_R2EACT_V053_OVERSIZED_ACTION_POLICY,
                "harbor_response_policy": (
                    TERMINUS_R2EACT_V053_OVERSIZED_CONTROLLER_RESPONSE_POLICY
                ),
                "next_round_observation_policy": (
                    TERMINUS_R2EACT_V053_OVERSIZED_OBSERVATION_POLICY
                ),
                "physical_token_accounting": "raw_generation",
                "context_admission_accounting": "projected_head_plus_eos",
            },
            "native_history_policy": {
                "requested": TERMINUS_R2EACT_V052_NATIVE_HISTORY_POLICY,
                "effective": TERMINUS_R2EACT_V052_NATIVE_HISTORY_POLICY,
            },
            "sglang_cache_policy": {
                "requested": TERMINUS_R2EACT_V052_SGLANG_CACHE_POLICY,
                "effective": TERMINUS_R2EACT_V052_SGLANG_CACHE_POLICY,
            },
            "memory_transport_policy": transport,
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
    "SINGLE_ATTEMPT",
    "TERMINUS_R2EACT_V053_IMPORT_PATH",
    "_memory_capability_report",
    "_serving_eviction_api_report",
    "_validate_job_config",
    "build_report",
    "main",
]
