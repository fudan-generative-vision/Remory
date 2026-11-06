"""Fail-closed preflight for the Qwen3.8 Terminal-Bench 2.0 v0.5.0 harness.

The v0.4.x preflight remains an immutable historical contract.  This module
accepts only the v0.5.0 import path, validates either a one-attempt diagnostic
or five-attempt official protocol, resolves the compressor checkpoint through
SpecForge's canonical ``r2eact_serving_contract`` facade, and delegates live
SGLang identity checks to the secret-free runtime attestor.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from harbor.models.job.config import JobConfig
from evaluation_checkpoint import resolve_eval_checkpoint

from terminal_bench_eval import r2eact_runtime_attestation as runtime_attestation
from terminal_bench_eval.tb20_official_preflight import (
    DATASET_DIGEST,
    DATASET_NAME,
    DATASET_TASK_COUNT,
    PROOT_ENVIRONMENT_IMPORT_PATH,
    PROOT_GUEST_UTILITY_PROFILE,
    _dataset_report,
    _proot_report,
)


PREFLIGHT_SCHEMA = "terminal_bench_r2eact_v050_preflight_v1"
TERMINUS_R2EACT_V050_IMPORT_PATH = (
    "terminal_bench_eval.terminus_r2eact_agent_v050:TerminusR2EActAgentV050"
)
TERMINUS_R2EACT_V050_NAME = "r2eact-terminus-2"
TERMINUS_R2EACT_V050_VERSION = "0.5.0"

QWEN38_MODEL_NAME = "r2eact-qwen38-27b-compressor-r16-step1000"
QWEN38_TARGET_FAMILY = "qwen38"
QWEN38_CHECKPOINT_SCHEMA = "qwen38_27b_r2eact_token_clock_serving_v1"
GENERATION_DEGENERATION_GUARD_SERVER_ABI = (
    "specforge_exact_periodic_guard_global_class_v1"
)

ALLOWED_ATTEMPTS = frozenset({1, 5})
OFFICIAL_ATTEMPTS = 5
DIAGNOSTIC_ATTEMPTS = 1
CONCURRENT_TRIALS = 1
AGENT_TIMEOUT_SECONDS = 10_800
MAX_NEW_TOKENS = 81_920
WORKING_MEMORY_TURNS = 8
REQUEST_TIMEOUT_SECONDS = 1_800
OFFICIAL_CPUS = 32
OFFICIAL_MEMORY_MB = 49_152
CHECKPOINT_CONFIG_MAX_BYTES = 16 * 1024 * 1024

TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST = ("terminus_2.pane",)
TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS = frozenset(
    {
        "memory.jsonl",
        "r2eact-terminus-2-events.jsonl",
        "r2eact-terminus-2-receipts.json",
        "terminus-terminal-transport-events.jsonl",
        "trajectory.json",
    }
)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"could not read job config: {path}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("job config must be a YAML object")
    return value


def _absolute_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise RuntimeError(f"{field} must be an absolute filesystem path")
    rendered = os.fspath(value)
    if not isinstance(rendered, str) or not rendered.strip():
        raise RuntimeError(f"{field} must be an absolute filesystem path")
    if any(character in rendered for character in "\x00\r\n"):
        raise RuntimeError(f"{field} is not a safe filesystem path")
    raw = Path(rendered).expanduser()
    if not raw.is_absolute():
        raise RuntimeError(f"{field} must be an absolute filesystem path")
    return raw.resolve()


def _require_exact(kwargs: Mapping[str, Any], key: str, expected: Any) -> None:
    observed = kwargs.get(key)
    if type(observed) is not type(expected) or observed != expected:
        raise RuntimeError(f"v0.5.0 agent kwarg {key!r} must equal {expected!r}")


def _validate_job_config_for_dataset(
    path: Path,
    *,
    dataset_name: str,
    dataset_digest: str,
    dataset_label: str,
) -> JobConfig:
    """Validate the v0.5.0 protocol against one exact dataset identity."""

    raw = _load_yaml(Path(path))
    try:
        config = JobConfig.model_validate(raw)
    except Exception as exc:
        raise RuntimeError("job config does not satisfy Harbor's schema") from exc

    if config.n_attempts not in ALLOWED_ATTEMPTS:
        raise RuntimeError(
            "job config must use one diagnostic or five official attempts"
        )
    if config.n_concurrent_trials != CONCURRENT_TRIALS:
        raise RuntimeError("the single-request SGLang server requires concurrency one")
    if len(config.datasets) != 1:
        raise RuntimeError("job config must contain exactly one dataset")
    dataset = config.datasets[0]
    if dataset.name != dataset_name or dataset.ref != dataset_digest:
        raise RuntimeError(
            f"job config does not pin the approved {dataset_label} digest"
        )
    if len(config.agents) != 1:
        raise RuntimeError("job config must contain exactly one agent")

    agent = config.agents[0]
    if agent.import_path != TERMINUS_R2EACT_V050_IMPORT_PATH:
        raise RuntimeError("job config must select the v0.5.0 Terminus r^2eAct agent")
    if agent.model_name != QWEN38_MODEL_NAME:
        raise RuntimeError("job config must select the pinned Qwen3.8 served model")
    if agent.override_timeout_sec != AGENT_TIMEOUT_SECONDS:
        raise RuntimeError("job config must set the agent timeout to 10,800 seconds")
    if tuple(agent.include_logs) != TERMINUS_R2EACT_GUEST_LOG_ALLOWLIST:
        raise RuntimeError("v0.5.0 requires the exact guest-log allowlist")
    missing_exclusions = TERMINUS_R2EACT_CONTROLLER_LOG_EXCLUSIONS - set(
        agent.exclude_logs
    )
    if missing_exclusions:
        raise RuntimeError(
            "v0.5.0 controller-owned log protection is incomplete: "
            + ", ".join(sorted(missing_exclusions))
        )

    kwargs = agent.kwargs
    if kwargs.get("expected_served_model") != agent.model_name:
        raise RuntimeError("expected_served_model must equal the Harbor model_name")
    _require_exact(kwargs, "expected_target_family", QWEN38_TARGET_FAMILY)
    _require_exact(kwargs, "expected_checkpoint_schema", QWEN38_CHECKPOINT_SCHEMA)
    _require_exact(kwargs, "sync_token_ids_across_tp", True)
    _require_exact(kwargs, "working_memory_turns", WORKING_MEMORY_TURNS)
    _require_exact(kwargs, "max_new_tokens", MAX_NEW_TOKENS)
    _require_exact(
        kwargs,
        "remaining_context_output_ceiling_tokens",
        MAX_NEW_TOKENS,
    )
    _require_exact(kwargs, "parser_name", "json")
    _require_exact(kwargs, "enable_summarize", False)
    _require_exact(kwargs, "collect_rollout_details", False)
    _require_exact(kwargs, "record_terminal_session", False)
    _require_exact(kwargs, "use_remaining_context_for_generation", True)
    _require_exact(kwargs, "oversized_complete_round_eviction", True)
    _require_exact(kwargs, "capacity_pressure_reserve_tokens", 1_024)
    _require_exact(kwargs, "stagnation_cue_rounds", 6)
    _require_exact(kwargs, "exact_repeat_cue_count", 3)
    _require_exact(kwargs, "request_timeout", REQUEST_TIMEOUT_SECONDS)
    _require_exact(
        kwargs,
        "post_compression_generation_policy",
        "sglang_remaining_context_experimental",
    )
    if "max_turns" not in kwargs or kwargs.get("max_turns") is not None:
        raise RuntimeError("v0.5.0 must have no fixed max_turns limit")

    for key in (
        "specforge_root",
        "checkpoint",
        "corpus_dir",
        "expected_model_path",
    ):
        _absolute_path(kwargs.get(key), field=key)
    sglang_url = kwargs.get("sglang_url")
    if not isinstance(sglang_url, str) or not sglang_url.strip():
        raise RuntimeError("sglang_url must be a nonempty HTTP(S) origin")

    environment = config.environment
    if environment.import_path != PROOT_ENVIRONMENT_IMPORT_PATH:
        raise RuntimeError("job config must use the approved local PRoot provider")
    if environment.kwargs.get("guest_utility_profile") != (PROOT_GUEST_UTILITY_PROFILE):
        raise RuntimeError("job config must use the exact PRoot guest utility profile")
    if environment.override_cpus != OFFICIAL_CPUS:
        raise RuntimeError("job config must record 32 CPUs")
    if environment.override_memory_mb != OFFICIAL_MEMORY_MB:
        raise RuntimeError("job config must record 48 GiB RAM")
    return config


def _validate_job_config(path: Path) -> JobConfig:
    """Load and validate the immutable Terminal-Bench 2.0 protocol fields."""

    return _validate_job_config_for_dataset(
        path,
        dataset_name=DATASET_NAME,
        dataset_digest=DATASET_DIGEST,
        dataset_label="TB2.0",
    )


def _scope_report(config: JobConfig) -> dict[str, Any]:
    dataset = config.datasets[0]
    task_names = list(dataset.task_names or ())
    exclusions = list(dataset.exclude_task_names or ())
    limited = bool(task_names or exclusions or dataset.n_tasks is not None)
    attempt_regime = (
        "official_five_attempt"
        if config.n_attempts == OFFICIAL_ATTEMPTS
        else "diagnostic_single_attempt"
    )
    classification = (
        "official"
        if config.n_attempts == OFFICIAL_ATTEMPTS and not limited
        else "diagnostic"
    )
    return {
        "classification": classification,
        "attempt_regime": attempt_regime,
        "attempts_per_task": config.n_attempts,
        "task_scope": "selected" if limited else "full_dataset",
        "selected_task_names": task_names,
        "excluded_task_names": exclusions,
        "n_tasks_limit": dataset.n_tasks,
        "official_task_count": DATASET_TASK_COUNT,
    }


def _load_canonical_serving_contract(specforge_root: Path) -> ModuleType:
    source = specforge_root / "context_compression" / "r2eact_serving_contract.py"
    if not source.is_file():
        raise RuntimeError(
            "configured SpecForge has no canonical r2eact_serving_contract"
        )
    root_text = str(specforge_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    try:
        module = importlib.import_module("context_compression.r2eact_serving_contract")
    except Exception as exc:
        raise RuntimeError(
            "could not import canonical r2eact_serving_contract"
        ) from exc
    module_file_value = getattr(module, "__file__", None)
    if not isinstance(module_file_value, str):
        raise RuntimeError("canonical r2eact_serving_contract has no source path")
    module_file = Path(module_file_value).resolve()
    if not module_file.is_relative_to(specforge_root):
        raise RuntimeError(
            "canonical r2eact_serving_contract was imported outside configured SpecForge"
        )
    resolver = getattr(module, "resolve_r2eact_checkpoint_serving_contract", None)
    if not callable(resolver):
        raise RuntimeError("canonical r2eact serving resolver is unavailable")
    return module


def _read_checkpoint_config(checkpoint: Path) -> tuple[dict[str, Any], bytes]:
    config_path = checkpoint / "config.json"
    try:
        metadata = config_path.lstat()
    except OSError as exc:
        raise RuntimeError("checkpoint config.json is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("checkpoint config.json must be a safe regular file")
    if metadata.st_size <= 0 or metadata.st_size > CHECKPOINT_CONFIG_MAX_BYTES:
        raise RuntimeError("checkpoint config.json has an invalid size")
    try:
        payload = config_path.read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("checkpoint config.json is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("checkpoint config.json must contain an object")
    return value, payload


def _checkpoint_contract_report(config: JobConfig) -> dict[str, Any]:
    kwargs = config.agents[0].kwargs
    specforge_root = _absolute_path(kwargs["specforge_root"], field="specforge_root")
    checkpoint = resolve_eval_checkpoint(kwargs["checkpoint"])
    expected_model_path = _absolute_path(
        kwargs["expected_model_path"], field="expected_model_path"
    )
    checkpoint_config, payload = _read_checkpoint_config(checkpoint)
    canonical = _load_canonical_serving_contract(specforge_root)
    resolver = canonical.resolve_r2eact_checkpoint_serving_contract
    try:
        contract = resolver(checkpoint_config)
    except Exception as exc:
        raise RuntimeError(
            "canonical r2eact checkpoint contract rejected config"
        ) from exc

    expected_schema = kwargs["expected_checkpoint_schema"]
    if checkpoint_config.get("schema") != expected_schema:
        raise RuntimeError(
            "checkpoint config schema differs from expected_checkpoint_schema"
        )
    if getattr(contract, "checkpoint_schema", None) != expected_schema:
        raise RuntimeError("canonical checkpoint serving schema mismatch")
    expected_family = kwargs["expected_target_family"]
    if checkpoint_config.get("target_family") != expected_family:
        raise RuntimeError("checkpoint target_family differs from expectation")
    target_model = _absolute_path(
        checkpoint_config.get("target_model"), field="checkpoint target_model"
    )
    if target_model != expected_model_path:
        raise RuntimeError("checkpoint target_model differs from expected_model_path")
    ratio = getattr(contract, "compression_ratio", None)
    if type(ratio) is not int or ratio != 16:
        raise RuntimeError(
            "canonical checkpoint contract must resolve compression ratio 16"
        )
    context_organization = getattr(contract, "context_organization", None)
    if not isinstance(context_organization, str) or not context_organization:
        raise RuntimeError("canonical checkpoint contract has no context organization")

    canonical_file = Path(str(canonical.__file__)).resolve()
    return {
        "checkpoint": str(checkpoint),
        "config_path": str(checkpoint / "config.json"),
        "config_sha256": hashlib.sha256(payload).hexdigest(),
        "canonical_module": str(canonical_file),
        "canonical_module_sha256": hashlib.sha256(
            canonical_file.read_bytes()
        ).hexdigest(),
        "checkpoint_schema": expected_schema,
        "target_family": expected_family,
        "target_model": str(target_model),
        "compression_ratio": ratio,
        "context_organization": context_organization,
        "canonical_contract_resolved": True,
    }


def _runtime_spec(
    config: JobConfig, sglang_url: str | None
) -> runtime_attestation.RuntimeAttestationSpec:
    kwargs = config.agents[0].kwargs
    configured_url = str(kwargs["sglang_url"]).rstrip("/")
    requested_url = (
        configured_url if sglang_url is None else str(sglang_url).rstrip("/")
    )
    if requested_url != configured_url:
        raise RuntimeError("preflight SGLang URL differs from the agent configuration")
    return runtime_attestation.RuntimeAttestationSpec(
        sglang_url=requested_url,
        expected_served_model=str(kwargs["expected_served_model"]),
        expected_forward_hook_factory=(runtime_attestation.R2EACT_FORWARD_HOOK_FACTORY),
        expected_checkpoint=str(resolve_eval_checkpoint(kwargs["checkpoint"])),
        expected_generation_degeneration_guard_server_abi=(
            GENERATION_DEGENERATION_GUARD_SERVER_ABI
        ),
        expected_model_path=str(kwargs["expected_model_path"]),
        expected_sync_token_ids_across_tp=True,
        require_custom_logit_processor=True,
    )


def _attest_runtime(
    spec: runtime_attestation.RuntimeAttestationSpec,
) -> dict[str, Any]:
    value = runtime_attestation.attest_r2eact_runtime(spec)
    if not isinstance(value, Mapping) or value.get("ok") is not True:
        raise RuntimeError("r2eact runtime attestor returned no successful receipt")
    if value.get("schema") != runtime_attestation.RUNTIME_ATTESTATION_SCHEMA:
        raise RuntimeError("r2eact runtime attestation schema mismatch")
    return dict(value)


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
            "score_aggregation": (
                "mean_pass_at_1_across_five_attempts"
                if config.n_attempts == OFFICIAL_ATTEMPTS
                else "single_attempt_diagnostic_only"
            ),
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


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sglang-url")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = asyncio.run(build_report(args.config, args.sglang_url))
    if args.output is not None:
        _atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PREFLIGHT_SCHEMA",
    "TERMINUS_R2EACT_V050_IMPORT_PATH",
    "_checkpoint_contract_report",
    "_validate_job_config",
    "_validate_job_config_for_dataset",
    "build_report",
    "main",
]
