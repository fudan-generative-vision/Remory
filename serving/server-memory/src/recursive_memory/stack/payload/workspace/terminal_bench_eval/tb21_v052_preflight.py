"""Fail-closed preflight for Terminal-Bench 2.1 r^2eAct v0.5.2."""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harbor.models.job.config import JobConfig

from terminal_bench_eval import tb20_v050_preflight as common
from terminal_bench_eval import tb21_v050_preflight as tb21
from terminal_bench_eval.terminus_r2eact_agent_v052 import (
    TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY,
    TERMINUS_R2EACT_V052_NAME,
    TERMINUS_R2EACT_V052_NATIVE_HISTORY_POLICY,
    TERMINUS_R2EACT_V052_REQUIRED_CAPABILITIES,
    TERMINUS_R2EACT_V052_SGLANG_CACHE_POLICY,
    TERMINUS_R2EACT_V052_VERSION,
)


PREFLIGHT_SCHEMA = "terminal_bench_r2eact_v052_tb21_preflight_v1"
TERMINUS_R2EACT_V052_IMPORT_PATH = (
    "terminal_bench_eval.terminus_r2eact_agent_v052:TerminusR2EActAgentV052"
)


def _require_exact(kwargs: Mapping[str, Any], key: str, expected: Any) -> None:
    observed = kwargs.get(key)
    if type(observed) is not type(expected) or observed != expected:
        raise RuntimeError(f"v0.5.2 agent kwarg {key!r} must equal {expected!r}")


def _validate_job_config(path: Path) -> JobConfig:
    raw = common._load_yaml(Path(path))
    try:
        config = JobConfig.model_validate(raw)
    except Exception as exc:
        raise RuntimeError("job config does not satisfy Harbor's schema") from exc
    if config.n_attempts != common.OFFICIAL_ATTEMPTS:
        raise RuntimeError("TB2.1 v0.5.2 official config requires five attempts")
    # One trial per pool worker remains the official accounting policy.  The
    # SGLang server itself must support >=2 running requests for recover batch
    # scheduling and for multiple local clients when explicitly used.
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
    if agent.import_path != TERMINUS_R2EACT_V052_IMPORT_PATH:
        raise RuntimeError("job config must select the v0.5.2 agent")
    if agent.model_name != common.QWEN38_MODEL_NAME:
        raise RuntimeError("job config must select the pinned Qwen3.8 model")
    if agent.override_timeout_sec != common.AGENT_TIMEOUT_SECONDS:
        raise RuntimeError("job config must set the 10,800-second agent timeout")
    if tuple(agent.include_logs) != common.TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST:
        raise RuntimeError("v0.5.2 requires the exact guest-log allowlist")
    missing = common.TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS - set(
        agent.exclude_logs
    )
    if missing:
        raise RuntimeError(
            "v0.5.2 controller-owned log protection is incomplete: "
            + ", ".join(sorted(missing))
        )

    kwargs = agent.kwargs
    if kwargs.get("expected_served_model") != agent.model_name:
        raise RuntimeError("expected_served_model must equal model_name")
    exact = {
        "expected_target_family": common.QWEN38_TARGET_FAMILY,
        "expected_checkpoint_schema": common.QWEN38_CHECKPOINT_SCHEMA,
        "sync_token_ids_across_tp": True,
        "working_memory_turns": common.WORKING_MEMORY_TURNS,
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
        "post_compression_generation_policy": (
            "sglang_remaining_context_experimental"
        ),
        "native_history_policy": TERMINUS_R2EACT_V052_NATIVE_HISTORY_POLICY,
        "sglang_cache_policy": TERMINUS_R2EACT_V052_SGLANG_CACHE_POLICY,
        "memory_transport_policy": TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY,
    }
    for key, expected in exact.items():
        _require_exact(kwargs, key, expected)
    if "max_turns" not in kwargs or kwargs.get("max_turns") is not None:
        raise RuntimeError("v0.5.2 must have no fixed max_turns limit")
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
    request = urllib.request.Request(
        str(sglang_url).rstrip("/") + "/server_info",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            status = int(getattr(response, "status", 0))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("SGLang v0.5.2 capability request failed") from exc
    if status != 200 or len(raw) > 2 * 1024 * 1024:
        raise RuntimeError("SGLang v0.5.2 capability response is invalid")
    try:
        info = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("SGLang /server_info returned invalid JSON") from exc
    if not isinstance(info, Mapping):
        raise RuntimeError("SGLang /server_info returned a non-object")

    capabilities = info.get("request_capabilities")
    max_running = info.get("max_running_requests")
    handle = info.get("specforge_gpu_memory_handle")
    if (
        not isinstance(capabilities, list)
        or any(not isinstance(value, str) for value in capabilities)
        or not TERMINUS_R2EACT_V052_REQUIRED_CAPABILITIES.issubset(capabilities)
        or isinstance(max_running, bool)
        or not isinstance(max_running, int)
        or max_running < 2
        or not isinstance(handle, Mapping)
        or handle.get("schema")
        != "specforge_tp_gpu_memory_handle_server_info_v1"
        or not isinstance(handle.get("server_epoch"), str)
        or not handle.get("server_epoch")
        or not isinstance(handle.get("hook_abi_digest"), str)
        or not str(handle.get("hook_abi_digest")).startswith("sha256:")
    ):
        raise RuntimeError("SGLang does not satisfy the v0.5.2 transport ABI")
    return {
        "attested": True,
        "effective_policy": TERMINUS_R2EACT_V052_MEMORY_TRANSPORT_POLICY,
        "capabilities": sorted(set(capabilities)),
        "max_running_requests": max_running,
        "server_epoch": handle["server_epoch"],
        "hook_abi_digest": handle["hook_abi_digest"],
        "handle_registry": dict(handle),
    }


async def build_report(
    config_path: Path,
    sglang_url: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = _validate_job_config(config_path)
    spec = common._runtime_spec(config, sglang_url)
    configured_url = str(config.agents[0].kwargs["sglang_url"]).rstrip("/")
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
            "agent_timeout_seconds": common.AGENT_TIMEOUT_SECONDS,
            "score_aggregation": "mean_pass_at_1_across_five_attempts",
        },
        "dataset": dataset,
        "agent": {
            "name": TERMINUS_R2EACT_V052_NAME,
            "version": TERMINUS_R2EACT_V052_VERSION,
            "import_path": TERMINUS_R2EACT_V052_IMPORT_PATH,
            "model_name": config.agents[0].model_name,
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
    "TERMINUS_R2EACT_V052_IMPORT_PATH",
    "_memory_capability_report",
    "_validate_job_config",
    "build_report",
    "main",
]
