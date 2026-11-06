"""Preflight the pinned Qwen3.6 Terminal-Bench 2.0 r^2eAct job."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml
from harbor.models.job.config import JobConfig
from harbor.registry.client.package import PackageDatasetClient

from eval.terminal_bench.proot_environment import (
    PROOT_GUEST_UTILITY_APT_PACKAGES,
    PROOT_GUEST_UTILITY_COMMANDS,
    PROOT_GUEST_UTILITY_POLICY,
    PROOT_GUEST_UTILITY_PROFILE,
    PROOT_GUEST_UTILITY_RECEIPT,
    ProotEnvironment,
)
from terminal_bench_eval.r2eact_agent import (
    QWEN36_TB20_CONTEXT_TOKENS,
    QWEN36_TB20_MAX_NEW_TOKENS,
    QWEN36_TB20_PROTOCOL_ID,
    QWEN36_TB20_SAMPLING,
    R2EACT_ADAPTER_VERSION,
    R2EACT_MEMORY_FILENAME,
)
from terminal_bench_eval.terminus_r2eact_agent import (
    TERMINUS_R2EACT_DEFAULT_EXACT_REPEAT_CUE_COUNT,
    TERMINUS_R2EACT_DEFAULT_STAGNATION_CUE_ROUNDS,
    TERMINUS_R2EACT_DEFAULT_WORKING_MEMORY_TURNS,
    TERMINUS_R2EACT_CONTEXT_ABI,
    TERMINUS_R2EACT_EVENTS_FILENAME,
    TERMINUS_R2EACT_NAME,
    TERMINUS_R2EACT_OVERSIZED_ROUND_EVICTION_POLICY,
    TERMINUS_R2EACT_PROGRESS_GUARD_POLICY,
    TERMINUS_R2EACT_PROMPT_ABI,
    TERMINUS_R2EACT_RECEIPTS_FILENAME,
    TERMINUS_R2EACT_ROUND_REMINDER_POLICY,
    TERMINUS_R2EACT_VERSION,
    TERMINUS_TERMINAL_TRANSPORT_ABI,
    TERMINUS_TERMINAL_TRANSPORT_POLICY,
    TERMINUS_TERMINAL_TRANSPORT_RECEIPTS_FILENAME,
    TERMINUS_TOOL_OBSERVATION_INFORMATION_LINES_POLICY,
    TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING,
    TERMINUS_TOOL_OBSERVATION_EXACT_MAX_BYTES,
    TERMINUS_TOOL_OBSERVATION_MANIFEST_FIRST_LINE_MAX_BYTES,
    TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_BYTES,
    TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES,
    TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY,
    _terminal_transport_metadata,
)


DATASET_NAME = "terminal-bench/terminal-bench-2"
DATASET_DIGEST = (
    "sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b"
)
DATASET_TASK_COUNT = 89
OFFICIAL_ATTEMPTS = 5
OFFICIAL_AGENT_TIMEOUT_SECONDS = 10_800
OFFICIAL_CPUS = 32
OFFICIAL_MEMORY_MB = 49_152
R2EACT_IMPORT_PATH = "terminal_bench_eval.r2eact_agent:R2EActAgent"
TERMINUS_R2EACT_IMPORT_PATH = (
    "terminal_bench_eval.terminus_r2eact_agent:TerminusR2EActAgent"
)
PROOT_ENVIRONMENT_IMPORT_PATH = (
    "eval.terminal_bench.proot_environment:ProotEnvironment"
)
TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS = frozenset(
    {
        R2EACT_MEMORY_FILENAME,
        TERMINUS_R2EACT_EVENTS_FILENAME,
        TERMINUS_R2EACT_RECEIPTS_FILENAME,
        TERMINUS_TERMINAL_TRANSPORT_RECEIPTS_FILENAME,
        "trajectory.json",
    }
)
TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST = ("terminus_2.pane",)


def _positive_integer(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"SGLang /server_info has no positive {key!r}")
    return int(value)


def _read_server_info(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url.rstrip("/") + "/server_info", method="GET")
    with urllib.request.urlopen(request, timeout=30.0) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise RuntimeError("SGLang /server_info returned a non-object")
    return value


def _read_cgroup_limit(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _proot_report() -> dict[str, Any]:
    """Run the syscall-level verifier preflight and record its exact binary."""

    try:
        ProotEnvironment.preflight()
        binary = ProotEnvironment._configured_proot_binary()
    except SystemExit as exc:
        # Environment providers historically report fail-closed setup errors via
        # SystemExit.  This verifier runs inside asyncio.to_thread(), where a
        # BaseException aborts loop teardown and produces a misleading "Task
        # exception was never retrieved" traceback.  Keep the failure fatal, but
        # surface it as an ordinary preflight error at this library boundary.
        detail = str(exc).strip() or "PRoot environment preflight failed"
        raise RuntimeError(detail) from None
    result = subprocess.run(
        [str(binary), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
        check=False,
    )
    rendered = result.stdout.decode(errors="replace")
    version_match = re.search(r"\bv?(\d+\.\d+\.\d+(?:-[0-9a-z]+)?)\b", rendered)
    if result.returncode != 0 or version_match is None:
        raise RuntimeError(f"unable to identify configured PRoot: {rendered.strip()}")
    return {
        "binary": str(binary),
        "version": version_match.group(1),
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "statx_semantics_probe": "passed",
        "faccessat2_semantics_probe": "passed",
        "runtime_binds": ["/dev", "/proc"],
        "ordinary_exec_transport": "long_lived_proot_rpc",
        "package_manager_transport": "direct_chroot_compatibility",
        "guest_utility_provisioning": {
            "profile": PROOT_GUEST_UTILITY_PROFILE,
            "policy": PROOT_GUEST_UTILITY_POLICY,
            "provisioned_for_every_trial": True,
            "commands": list(PROOT_GUEST_UTILITY_COMMANDS),
            "apt_packages": list(PROOT_GUEST_UTILITY_APT_PACKAGES),
            "fail_closed_receipt": {
                "base_filename": PROOT_GUEST_UTILITY_RECEIPT,
                "location": "host_trial_directory_with_session_suffix",
                "failure_policy": "closed",
                "required_before_agent_start": True,
            },
            "modified_benchmark_environment": True,
            "package_versions_unpinned": True,
            "package_versions_recorded_in_receipt": True,
        },
    }


def _specforge_runtime_report(root_value: Any) -> dict[str, Any]:
    """Bind the mutable local SpecForge runtime to a checked capability/hash."""

    root = Path(str(root_value)).resolve()
    if not (root / "context_compression" / "r2eact_sglang_live.py").is_file():
        raise RuntimeError(f"invalid SpecForge root: {root}")
    probe = r'''
import hashlib
import json
from pathlib import Path
from context_compression import grepseek_sglang_live as implementation
from context_compression import r2eact_sglang_live as facade

facade_file = Path(facade.__file__).resolve()
implementation_file = Path(implementation.__file__).resolve()
print(json.dumps({
    "facade_file": str(facade_file),
    "facade_sha256": hashlib.sha256(facade_file.read_bytes()).hexdigest(),
    "implementation_file": str(implementation_file),
    "implementation_sha256": hashlib.sha256(
        implementation_file.read_bytes()
    ).hexdigest(),
    "advertised_policy": getattr(
        facade, "QWEN36_OVERSIZED_COMPLETE_ROUND_EVICTION_POLICY", None
    ),
    "generator_policy": getattr(
        facade.Qwen36SGLangTokenClockRolloutGenerator,
        "oversized_complete_round_eviction_policy",
        None,
    ),
}, separators=(",", ":")))
'''
    environment = dict(os.environ)
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(root) + (
        os.pathsep + existing_path if existing_path else ""
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(root),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"SpecForge capability probe failed: {stderr}")
    lines = [line for line in result.stdout.decode().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("SpecForge capability probe returned no report")
    try:
        report = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("SpecForge capability probe returned invalid JSON") from exc
    expected = TERMINUS_R2EACT_OVERSIZED_ROUND_EVICTION_POLICY
    if report.get("advertised_policy") != expected:
        raise RuntimeError("SpecForge facade lacks the pinned latest-successor grace")
    if report.get("generator_policy") != expected:
        raise RuntimeError("SpecForge generator lacks the pinned latest-successor grace")
    for key in ("facade_file", "implementation_file"):
        source = Path(str(report.get(key, ""))).resolve()
        if not source.is_relative_to(root):
            raise RuntimeError(f"SpecForge probe imported {source} outside {root}")
    return {
        "root": str(root),
        **report,
        "capability_check": "passed",
    }


def _validate_job_config(path: Path) -> JobConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = JobConfig.model_validate(raw)
    if config.n_attempts != OFFICIAL_ATTEMPTS:
        raise RuntimeError("job config must use five attempts per task")
    if config.n_concurrent_trials != 1:
        raise RuntimeError("the single-request SGLang server requires concurrency one")
    if len(config.datasets) != 1:
        raise RuntimeError("job config must contain exactly one dataset")
    dataset = config.datasets[0]
    if dataset.name != DATASET_NAME or dataset.ref != DATASET_DIGEST:
        raise RuntimeError("job config does not pin the approved TB2.0 digest")
    if len(config.agents) != 1:
        raise RuntimeError("job config must contain exactly one agent")
    agent = config.agents[0]
    if agent.import_path not in {R2EACT_IMPORT_PATH, TERMINUS_R2EACT_IMPORT_PATH}:
        raise RuntimeError("job config does not select an approved local r^2eAct agent")
    if agent.override_timeout_sec != OFFICIAL_AGENT_TIMEOUT_SECONDS:
        raise RuntimeError("job config must override the agent timeout to three hours")
    if int(agent.kwargs.get("max_new_tokens", 0)) != QWEN36_TB20_MAX_NEW_TOKENS:
        raise RuntimeError("job config does not request the official 81,920-token cap")
    if (
        int(agent.kwargs.get("remaining_context_output_ceiling_tokens", 0))
        != QWEN36_TB20_MAX_NEW_TOKENS
    ):
        raise RuntimeError("job config does not enforce the 81,920-token ceiling")
    expected_working_memory_turns = (
        TERMINUS_R2EACT_DEFAULT_WORKING_MEMORY_TURNS
        if agent.import_path == TERMINUS_R2EACT_IMPORT_PATH
        else 3
    )
    if (
        int(agent.kwargs.get("working_memory_turns", 0))
        != expected_working_memory_turns
    ):
        raise RuntimeError(
            "job config must retain "
            f"{expected_working_memory_turns} raw working-memory rounds"
        )
    round_limit_name = (
        "max_turns"
        if agent.import_path == TERMINUS_R2EACT_IMPORT_PATH
        else "max_rounds"
    )
    if agent.kwargs.get(round_limit_name) is not None:
        raise RuntimeError(
            f"job config must leave {round_limit_name} unset and rely on the trial timeout"
        )
    if agent.import_path == TERMINUS_R2EACT_IMPORT_PATH:
        if tuple(agent.include_logs) != TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST:
            required = ", ".join(TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST)
            raise RuntimeError(
                "Terminus r^2eAct must use the exact non-mounted guest-log "
                f"include_logs allowlist, in order: {required}"
            )
        missing_log_exclusions = TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS - set(
            agent.exclude_logs
        )
        if missing_log_exclusions:
            rendered = ", ".join(sorted(missing_log_exclusions))
            raise RuntimeError(
                "Terminus r^2eAct must protect controller-owned agent logs from "
                f"the non-mounted guest download: missing exclude_logs: {rendered}"
            )
        if agent.kwargs.get("parser_name") != "json":
            raise RuntimeError("Terminus r^2eAct must use the JSON parser")
        if agent.kwargs.get("enable_summarize") is not False:
            raise RuntimeError("Terminus summarization must be disabled")
        if int(agent.kwargs.get("capacity_pressure_reserve_tokens", 0)) != 1024:
            raise RuntimeError("Terminus r^2eAct must reserve 1,024 admission tokens")
        if agent.kwargs.get("oversized_complete_round_eviction") is not True:
            raise RuntimeError(
                "Terminus r^2eAct must enable whole oversized-round eviction"
            )
        if int(agent.kwargs.get("stagnation_cue_rounds", 0)) != (
            TERMINUS_R2EACT_DEFAULT_STAGNATION_CUE_ROUNDS
        ):
            raise RuntimeError(
                "Terminus r^2eAct must cue archive recall after six "
                "rounds without concrete task progress"
            )
        if int(agent.kwargs.get("exact_repeat_cue_count", 0)) != (
            TERMINUS_R2EACT_DEFAULT_EXACT_REPEAT_CUE_COUNT
        ):
            raise RuntimeError(
                "Terminus r^2eAct must cue on the third exact inspection"
            )
        if agent.kwargs.get("collect_rollout_details") is not False:
            raise RuntimeError(
                "Terminus r^2eAct must disable completion token-id collection"
            )
        if agent.kwargs.get("record_terminal_session") is not False:
            raise RuntimeError(
                "Terminus r^2eAct must keep inline terminal recording disabled; "
                "the direct tmux pipe-pane transcript is the approved artifact"
            )
    if config.environment.override_cpus != OFFICIAL_CPUS:
        raise RuntimeError("job config must record 32 CPUs")
    if config.environment.override_memory_mb != OFFICIAL_MEMORY_MB:
        raise RuntimeError("job config must record 48 GiB RAM")
    if config.environment.import_path != PROOT_ENVIRONMENT_IMPORT_PATH:
        raise RuntimeError(
            "job config must use the approved local PRoot environment provider"
        )
    guest_utility_profile = config.environment.kwargs.get("guest_utility_profile")
    if guest_utility_profile != PROOT_GUEST_UTILITY_PROFILE:
        raise RuntimeError(
            "job config must provision the exact PRoot guest utility profile "
            f"{PROOT_GUEST_UTILITY_PROFILE!r}; observed {guest_utility_profile!r}"
        )
    return config


async def _dataset_report() -> dict[str, Any]:
    reference = f"{DATASET_NAME}@{DATASET_DIGEST}"
    metadata = await PackageDatasetClient().get_dataset_metadata(reference)
    if metadata.version != DATASET_DIGEST:
        raise RuntimeError("Harbor resolved the dataset to an unexpected digest")
    if len(metadata.task_ids) != DATASET_TASK_COUNT:
        raise RuntimeError(
            f"pinned TB2.0 has {len(metadata.task_ids)} tasks, expected 89"
        )
    return {
        "reference": reference,
        "resolved_digest": metadata.version,
        "task_count": len(metadata.task_ids),
        "dataset_version_id": metadata.dataset_version_id,
    }


async def build_report(config_path: Path, sglang_url: str) -> dict[str, Any]:
    config = _validate_job_config(config_path)
    agent_config = config.agents[0]
    agent_import_path = agent_config.import_path
    local_name, local_version = (
        (TERMINUS_R2EACT_NAME, TERMINUS_R2EACT_VERSION)
        if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
        else ("r2eact", R2EACT_ADAPTER_VERSION)
    )
    specforge_task = (
        asyncio.to_thread(
            _specforge_runtime_report,
            agent_config.kwargs.get("specforge_root"),
        )
        if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
        else asyncio.sleep(0, result=None)
    )
    dataset, server_info, proot, specforge_runtime = await asyncio.gather(
        _dataset_report(),
        asyncio.to_thread(_read_server_info, sglang_url),
        asyncio.to_thread(_proot_report),
        specforge_task,
    )
    server_context = _positive_integer(server_info, "context_length")
    if server_context != QWEN36_TB20_CONTEXT_TOKENS:
        raise RuntimeError(
            "SGLang context_length must be 262,144 for this protocol; "
            f"observed {server_context:,}"
        )
    token_pool = _positive_integer(server_info, "max_total_num_tokens")
    maximum_input = _positive_integer(server_info, "max_req_input_len")
    empty_prefix_capacity = max(0, min(server_context, token_pool) - 2)
    effective_cap = min(QWEN36_TB20_MAX_NEW_TOKENS, empty_prefix_capacity)
    return {
        "status": "ok",
        "protocol_target": QWEN36_TB20_PROTOCOL_ID,
        "dataset": dataset,
        "job": {
            "attempts_per_task": config.n_attempts,
            "total_trials": DATASET_TASK_COUNT * config.n_attempts,
            "concurrent_trials": config.n_concurrent_trials,
            "agent_timeout_seconds": config.agents[0].override_timeout_sec,
            "requested_cpus": config.environment.override_cpus,
            "requested_memory_mb": config.environment.override_memory_mb,
            "score_aggregation": "mean_pass_at_1_across_five_attempts",
        },
        "model": {
            "served_model_name": server_info.get("served_model_name"),
            "sampling": dict(QWEN36_TB20_SAMPLING),
            "official_context_tokens": QWEN36_TB20_CONTEXT_TOKENS,
            "official_requested_max_new_tokens": QWEN36_TB20_MAX_NEW_TOKENS,
        },
        "local_harness": {
            "name": local_name,
            "version": local_version,
            "import_path": agent_import_path,
            "server_context_length": server_context,
            "server_configured_max_total_tokens": server_info.get("max_total_tokens"),
            "server_max_total_num_tokens": token_pool,
            "server_max_req_input_len": maximum_input,
            "empty_prefix_effective_max_new_tokens": effective_cap,
            "official_max_tokens_capacity_satisfied": (
                empty_prefix_capacity >= QWEN36_TB20_MAX_NEW_TOKENS
            ),
            "official_max_tokens_capacity_shortfall_tokens": max(
                0, QWEN36_TB20_MAX_NEW_TOKENS - empty_prefix_capacity
            ),
            "token_pool_classification": "local_harness_runtime",
            "environment_provider": config.environment.import_path,
            "prompt_abi": (
                TERMINUS_R2EACT_PROMPT_ABI
                if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
                else None
            ),
            "context_abi": (
                TERMINUS_R2EACT_CONTEXT_ABI
                if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
                else None
            ),
            "terminal_transport": (
                {
                    **_terminal_transport_metadata(),
                    "configured_record_terminal_session": (
                        agent_config.kwargs.get("record_terminal_session")
                    ),
                    "abi_constant": TERMINUS_TERMINAL_TRANSPORT_ABI,
                    "policy_constant": TERMINUS_TERMINAL_TRANSPORT_POLICY,
                }
                if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
                else None
            ),
            "specforge_runtime": specforge_runtime,
            "progress_guard": (
                {
                    "policy": TERMINUS_R2EACT_PROGRESS_GUARD_POLICY,
                    "exact_repeat_cue_count": agent_config.kwargs.get(
                        "exact_repeat_cue_count"
                    ),
                    "stagnation_cue_rounds": agent_config.kwargs.get(
                        "stagnation_cue_rounds"
                    ),
                    "stagnation_semantics": (
                        "consecutive_rounds_without_explicit_task_artifact_edit_"
                        "build_test_or_task_run"
                    ),
                    "cue_latching": "threshold_multiples_only",
                    "memory_lookup_effect": "preserves_stagnation_clock",
                    "blocking": False,
                    "round_limit": None,
                }
                if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
                else None
            ),
            "round_reminder": (
                {
                    "policy": TERMINUS_R2EACT_ROUND_REMINDER_POLICY,
                    "emission": "every_reached_model_round",
                    "reminder_round_cutoff": None,
                    "configured_max_turns": agent_config.kwargs.get(
                        "max_turns"
                    ),
                    "archive_range": "contiguous_complete_prefix_1_to_n",
                    "archive_complete_round_roles": ["assistant", "tool"],
                    "archive_source_of_truth": "persisted_memory_jsonl",
                    "current_round_source": "transactional_llm_round_index",
                    "placement": "tool_stdout_prefix_or_initial_user_prefix",
                    "delivery_verification": "exact_prefix_fail_closed",
                    "retrieval": "dedicated_grep_or_narrow_sed_action",
                    "mixed_archive_and_task_commands_allowed": False,
                    "raw_history_control_enabled": False,
                }
                if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
                else None
            ),
            "oversized_complete_round_eviction": (
                {
                    "enabled": agent_config.kwargs.get(
                        "oversized_complete_round_eviction"
                    ),
                    "policy": (
                        TERMINUS_R2EACT_OVERSIZED_ROUND_EVICTION_POLICY
                    ),
                    "working_memory_turns": agent_config.kwargs.get(
                        "working_memory_turns"
                    ),
                    "threshold_formula": (
                        "floor(min(server_context_length,"
                        "server_max_total_num_tokens)/(working_memory_turns+1))"
                    ),
                    "whole_rounds_only": True,
                    "latest_round_successor_grace": True,
                    "latest_round_capacity_fallback": (
                        "evict_older_first_then_latest_only_if_still_required"
                    ),
                    "changes_generation_budget": False,
                }
                if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
                else None
            ),
            "tool_observation_projection": (
                {
                    "policy": TERMINUS_TOOL_OBSERVATION_PROJECTION_POLICY,
                    "information_lines_policy": (
                        TERMINUS_TOOL_OBSERVATION_INFORMATION_LINES_POLICY
                    ),
                    "batch_hash_encoding": (
                        TERMINUS_TOOL_OBSERVATION_BATCH_HASH_ENCODING
                    ),
                    "exact_projection_max_bytes": (
                        TERMINUS_TOOL_OBSERVATION_EXACT_MAX_BYTES
                    ),
                    "manifest_first_line_max_bytes": (
                        TERMINUS_TOOL_OBSERVATION_MANIFEST_FIRST_LINE_MAX_BYTES
                    ),
                    "manifest_max_entries": (
                        TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_ENTRIES
                    ),
                    "manifest_max_bytes": (
                        TERMINUS_TOOL_OBSERVATION_MANIFEST_MAX_BYTES
                    ),
                    "shared_with_raw_control": True,
                }
                if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
                else None
            ),
            "controller_owned_agent_log_protection": (
                {
                    "policy": "harbor_agent_include_allowlist_and_exclude_logs_v2",
                    "guest_source_directory": "/logs/agent",
                    "required_include_logs": list(TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST),
                    "configured_include_logs": list(agent_config.include_logs),
                    "required_exclude_logs": sorted(
                        TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS
                    ),
                    "configured_exclude_logs": sorted(agent_config.exclude_logs),
                }
                if agent_import_path == TERMINUS_R2EACT_IMPORT_PATH
                else None
            ),
            "proot": proot,
            "resource_enforcement": (
                "requested_in_job_metadata; PRoot does not hard-enforce RAM"
            ),
            "outer_cgroup_cpu_max": _read_cgroup_limit(Path("/sys/fs/cgroup/cpu.max")),
            "outer_cgroup_memory_max": _read_cgroup_limit(
                Path("/sys/fs/cgroup/memory.max")
            ),
            "visible_cpu_count": len(os.sched_getaffinity(0)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(build_report(args.config.resolve(), args.sglang_url))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
