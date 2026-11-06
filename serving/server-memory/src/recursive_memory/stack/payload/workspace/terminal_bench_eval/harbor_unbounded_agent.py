"""Harbor CLI boundary for reviewed r^2eAct agent timeout policies.

Harbor normally falls back from a missing job override to the timeout embedded
in each task.  Therefore removing ``override_timeout_sec`` does not express an
unbounded agent phase.  This narrow wrapper changes the resolved timeout to
``None`` only for exact, reviewed import paths outside Terminal-Bench 2.1.
TB2.1 deliberately ignores both the legacy job override and heterogeneous task
defaults: every question receives its own independent 24-hour agent phase.
Environment construction, agent setup,
individual model/tool requests, verifier execution, pool leases, and worker
shutdown retain their independent safety timeouts.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from terminal_bench_eval.harbor_pinned_task_cache import (
    install_pinned_package_resolution,
)


UNBOUNDED_AGENT_TIMEOUT_POLICY = "terminal_bench_no_total_agent_timeout_v1"
TB21_DATASET_SOURCE = "terminal-bench/terminal-bench-2-1"
TB21_OFFICIAL_AGENT_TIMEOUT_POLICY = (
    "terminal_bench_2_1_official_per_task_agent_timeout_v1"
)
TB21_PER_TASK_AGENT_TIMEOUT_SECONDS = 24 * 60 * 60
TB21_LEGACY_8H_AGENT_TIMEOUT_POLICY = (
    "terminal_bench_2_1_uniform_per_task_8h_agent_timeout_v1"
)
TB21_UNIFORM_AGENT_TIMEOUT_POLICY = (
    "terminal_bench_2_1_uniform_per_task_24h_agent_timeout_v1"
)
TB21_UNIFORM_AGENT_TIMEOUT_SOURCE = "r2eact.tb21.uniform_per_task_24h"
UNBOUNDED_AGENT_IMPORT_PATHS = frozenset(
    {
        "terminal_bench_eval.codex_r2eact_agent:CodexNativeCompactR2EActAgent",
        "terminal_bench_eval.codex_r2eact_agent:CodexR2EActAgent",
        "terminal_bench_eval.terminus_r2eact_agent_v053:TerminusR2EActAgentV053",
    }
)
HARBOR_UNBOUNDED_CLI_MODULE = "terminal_bench_eval.harbor_unbounded_agent"
AGENT_TIME_BUDGET_RECEIPT_SCHEMA = "terminal_bench_agent_time_budget_receipt_v1"
AGENT_TIME_BUDGET_RECEIPT_NAME = "agent-time-budget.json"
VERIFIER_TIMEOUT_RETRY_POLICY = "proot_reviewed_network_bootstrap_once_v1"
AGENT_TIMEOUT_VERIFIER_RECOVERY_POLICY = (
    "proot_clean_rpc_restart_after_tb21_agent_timeout_v1"
)


def agent_import_uses_unbounded_timeout(import_path: Any) -> bool:
    return isinstance(import_path, str) and import_path in UNBOUNDED_AGENT_IMPORT_PATHS


def _trial_task_source(trial: Any) -> str | None:
    task = getattr(getattr(trial, "config", None), "task", None)
    source = getattr(task, "source", None)
    return source if isinstance(source, str) else None


def _resolved_multiplier(trial: Any) -> float:
    configured = getattr(
        getattr(trial, "config", None), "agent_timeout_multiplier", None
    )
    if configured is not None:
        return float(configured)
    return float(getattr(getattr(trial, "config", None), "timeout_multiplier", 1.0))


def _uniform_tb21_agent_timeout(trial: Any) -> tuple[float | None, float]:
    """Return ``(declared task default, independent 24-hour budget)``."""

    task_agent = getattr(
        getattr(getattr(trial, "task", None), "config", None), "agent", None
    )
    base = getattr(task_agent, "timeout_sec", None)
    declared = float(base) if base is not None else None
    return declared, float(TB21_PER_TASK_AGENT_TIMEOUT_SECONDS)


def install_unbounded_agent_timeout(
    trial_type: type[Any] | None = None,
    *,
    receipt_argv: list[str] | None = None,
) -> None:
    """Patch one Harbor process before it constructs any trial.

    ``trial_type`` is injectable so the policy and idempotence can be tested
    without mutating Harbor globally in the test process.
    """

    if trial_type is None:
        from harbor.trial.trial import Trial

        trial_type = Trial
    current = getattr(trial_type, "_compute_agent_timeout_sec")
    if getattr(current, "_r2eact_unbounded_timeout_policy", None) == (
        UNBOUNDED_AGENT_TIMEOUT_POLICY
    ):
        return

    receipt_arguments = None if receipt_argv is None else list(receipt_argv)

    def compute_agent_timeout(self: Any) -> float | None:
        agent = getattr(getattr(self, "config", None), "agent", None)
        if agent_import_uses_unbounded_timeout(getattr(agent, "import_path", None)):
            if _trial_task_source(self) == TB21_DATASET_SOURCE:
                task_timeout, effective = _uniform_tb21_agent_timeout(self)
                if receipt_arguments is not None:
                    _record_tb21_agent_timeout_receipt(
                        self,
                        receipt_arguments,
                        task_timeout_seconds=task_timeout,
                        effective_timeout_seconds=effective,
                    )
                return effective
            return None
        return current(self)

    compute_agent_timeout._r2eact_unbounded_timeout_policy = (  # type: ignore[attr-defined]
        UNBOUNDED_AGENT_TIMEOUT_POLICY
    )
    setattr(trial_type, "_compute_agent_timeout_sec", compute_agent_timeout)


def install_verifier_timeout_resilience(trial_type: type[Any] | None = None) -> None:
    """Retry once only when the environment proves the timeout was safe to retry.

    The PRoot backend persists partial stdout for every recognized verifier
    cancellation.  It authorizes a retry only for a reviewed official test-script
    hash that was still in network/bootstrap, after proving the canceled process
    tree was retired and restarting the RPC channel.  Other environments and all
    ambiguous/test-stage timeouts retain Harbor's original behavior.
    """

    if trial_type is None:
        from harbor.trial.trial import Trial

        trial_type = Trial
    current = getattr(trial_type, "_run_shared_verifier")
    if getattr(current, "_r2eact_verifier_timeout_retry_policy", None) == (
        VERIFIER_TIMEOUT_RETRY_POLICY
    ):
        return

    async def run_shared_verifier(self: Any, **kwargs: Any) -> Any:
        try:
            return await current(self, **kwargs)
        except asyncio.TimeoutError:
            environment = getattr(self, "agent_environment", None)
            prepare = getattr(environment, "prepare_verifier_timeout_retry", None)
            retry_ready = False
            if callable(prepare):
                try:
                    retry_ready = bool(await prepare(kwargs.get("timeout_sec")))
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Resilience machinery must never replace Harbor's timeout.
                    retry_ready = False
            if not retry_ready:
                raise

        finish = getattr(environment, "finish_verifier_timeout_retry", None)
        try:
            result = await current(self, **kwargs)
        except BaseException:
            if callable(finish):
                try:
                    finish(succeeded=False)
                except Exception:
                    pass
            raise
        if callable(finish):
            try:
                finish(succeeded=True)
            except Exception:
                pass
        return result

    run_shared_verifier._r2eact_verifier_timeout_retry_policy = (  # type: ignore[attr-defined]
        VERIFIER_TIMEOUT_RETRY_POLICY
    )
    setattr(trial_type, "_run_shared_verifier", run_shared_verifier)


def install_agent_timeout_verifier_recovery(
    trial_type: type[Any] | None = None,
) -> None:
    """Restore a clean PRoot command channel before scoring timed-out work.

    Harbor's ``SingleStepTrial._run_agent`` records ``AgentTimeoutError`` and
    intentionally continues to its verifier.  The custom PRoot backend must
    retire an RPC byte stream when the outer phase cancels an in-flight
    command, so this narrow hook asks that environment to restart only a
    proven-clean transport while preserving the task rootfs.
    """

    if trial_type is None:
        from harbor.trial.single_step import SingleStepTrial

        trial_type = SingleStepTrial
    current = getattr(trial_type, "_run_shared_verifier")
    if getattr(current, "_r2eact_agent_timeout_verifier_recovery", None) == (
        AGENT_TIMEOUT_VERIFIER_RECOVERY_POLICY
    ):
        return

    async def run_shared_verifier(self: Any, **kwargs: Any) -> Any:
        exception = getattr(getattr(self, "result", None), "exception_info", None)
        exception_type = (
            exception.get("exception_type")
            if isinstance(exception, dict)
            else getattr(exception, "exception_type", None)
        )
        agent = getattr(getattr(self, "config", None), "agent", None)
        if (
            exception_type != "AgentTimeoutError"
            or _trial_task_source(self) != TB21_DATASET_SOURCE
            or not agent_import_uses_unbounded_timeout(
                getattr(agent, "import_path", None)
            )
            or bool(
                getattr(
                    self,
                    "_r2eact_agent_timeout_verifier_recovery_attempted",
                    False,
                )
            )
        ):
            return await current(self, **kwargs)

        self._r2eact_agent_timeout_verifier_recovery_attempted = True

        prepare = getattr(
            getattr(self, "agent_environment", None),
            "prepare_shared_verifier_after_agent_timeout",
            None,
        )
        if callable(prepare):
            try:
                await prepare()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Recovery must never replace Harbor's original AgentTimeoutError.
                pass
        return await current(self, **kwargs)

    run_shared_verifier._r2eact_agent_timeout_verifier_recovery = (  # type: ignore[attr-defined]
        AGENT_TIMEOUT_VERIFIER_RECOVERY_POLICY
    )
    setattr(trial_type, "_run_shared_verifier", run_shared_verifier)


def _argument_value(argv: list[str], option: str) -> str | None:
    try:
        index = argv.index(option)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _invocation_config_and_cell(argv: list[str]) -> tuple[Path, Path] | None:
    """Resolve a pool cell without trusting arbitrary output paths."""

    config_value = _argument_value(argv, "--config")
    if config_value is not None:
        config = Path(config_value).expanduser().resolve()
        cell = config.parent
    else:
        job_value = _argument_value(argv, "--job-path")
        if job_value is None:
            return None
        job = Path(job_value).expanduser().resolve()
        config = job / "config.json"
        cell = job.parent
    if not (cell / "attempts").is_dir() or not config.is_file():
        return None
    return config, cell


def _unbounded_agent_from_config(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    agent = payload.get("agent")
    if not isinstance(agent, dict):
        agents = payload.get("agents")
        if (
            not isinstance(agents, list)
            or len(agents) != 1
            or not isinstance(agents[0], dict)
        ):
            return None
        agent = agents[0]
    if not agent_import_uses_unbounded_timeout(agent.get("import_path")):
        return None
    return agent


def _config_uses_tb21_task(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    task = payload.get("task") if isinstance(payload, dict) else None
    if isinstance(task, dict) and task.get("source") == TB21_DATASET_SOURCE:
        return True
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if isinstance(tasks, list) and any(
        isinstance(task_entry, dict) and task_entry.get("source") == TB21_DATASET_SOURCE
        for task_entry in tasks
    ):
        return True
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    return bool(
        isinstance(datasets, list)
        and any(
            isinstance(dataset, dict) and dataset.get("name") == TB21_DATASET_SOURCE
            for dataset in datasets
        )
    )


def _active_attempt_dir(cell: Path) -> Path | None:
    attempts = cell / "attempts"
    try:
        candidates = sorted(
            path
            for path in attempts.glob("execution-*")
            if path.is_dir() and not path.is_symlink()
        )
    except OSError:
        return None
    for candidate in reversed(candidates):
        try:
            record = json.loads((candidate / "run.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("status") == "running":
            return candidate
    return None


def _write_agent_timeout_receipt(attempt: Path, receipt: dict[str, Any]) -> None:
    destination = attempt / AGENT_TIME_BUDGET_RECEIPT_NAME
    temporary = attempt / f".{AGENT_TIME_BUDGET_RECEIPT_NAME}.{os.getpid()}.tmp"
    try:
        temporary.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _record_tb21_agent_timeout_receipt(
    trial: Any,
    argv: list[str],
    *,
    task_timeout_seconds: float | None,
    effective_timeout_seconds: float | None,
) -> None:
    """Record the concrete task budget after Harbor has materialized the task."""

    resolved = _invocation_config_and_cell(argv)
    if resolved is None:
        return
    config, cell = resolved
    agent = _unbounded_agent_from_config(config)
    attempt = _active_attempt_dir(cell)
    if agent is None or attempt is None:
        return
    trial_config = getattr(trial, "config", None)
    trial_agent = getattr(trial_config, "agent", None)
    task = getattr(trial_config, "task", None)
    receipt = {
        "schema": AGENT_TIME_BUDGET_RECEIPT_SCHEMA,
        "policy": TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
        "task_source": TB21_DATASET_SOURCE,
        "task_name": getattr(task, "name", None),
        "task_timeout_seconds": task_timeout_seconds,
        "effective_total_agent_timeout_seconds": effective_timeout_seconds,
        "effective_timeout_source": TB21_UNIFORM_AGENT_TIMEOUT_SOURCE,
        "configured_harbor_timeout_seconds": getattr(
            trial_agent, "override_timeout_sec", agent.get("override_timeout_sec")
        ),
        "max_timeout_seconds": getattr(trial_agent, "max_timeout_sec", None),
        "timeout_multiplier": getattr(trial_config, "timeout_multiplier", 1.0),
        "agent_timeout_multiplier": getattr(
            trial_config, "agent_timeout_multiplier", None
        ),
        "resolved_multiplier": _resolved_multiplier(trial),
        "execution_boundary": HARBOR_UNBOUNDED_CLI_MODULE,
        "pid": os.getpid(),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _write_agent_timeout_receipt(attempt, receipt)


def record_unbounded_agent_timeout_receipt(argv: list[str] | None = None) -> None:
    """Persist evidence that this concrete Harbor process installed the policy.

    The receipt is observability-only: failure to write it must not change the
    evaluation.  The timeout patch itself remains the execution authority.
    """

    arguments = list(sys.argv[1:] if argv is None else argv)
    resolved = _invocation_config_and_cell(arguments)
    if resolved is None:
        return
    config, cell = resolved
    agent = _unbounded_agent_from_config(config)
    attempt = _active_attempt_dir(cell)
    if agent is None or attempt is None:
        return
    # TB2.1's concrete value does not exist until Harbor materializes the task.
    # The patched Trial method writes the receipt at that point; never publish a
    # misleading pre-run "unbounded" receipt for this dataset.
    if _config_uses_tb21_task(config):
        return
    receipt = {
        "schema": AGENT_TIME_BUDGET_RECEIPT_SCHEMA,
        "policy": UNBOUNDED_AGENT_TIMEOUT_POLICY,
        "effective_total_agent_timeout_seconds": None,
        "configured_harbor_timeout_seconds": agent.get("override_timeout_sec"),
        "execution_boundary": HARBOR_UNBOUNDED_CLI_MODULE,
        "pid": os.getpid(),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    _write_agent_timeout_receipt(attempt, receipt)


def main() -> None:
    # The pool freezes exact sha256 task refs.  Resolve those refs directly to
    # Harbor's content-addressed package cache/storage instead of making every
    # single-cell CLI process depend on the mutable registry API.
    install_pinned_package_resolution()
    install_unbounded_agent_timeout(receipt_argv=list(sys.argv[1:]))
    install_verifier_timeout_resilience()
    from terminal_bench_eval.verifier_checkpoint import install_verifier_checkpointing
    install_verifier_checkpointing()
    # Compose on top of the verifier-timeout wrapper so SingleStepTrial keeps
    # both policies: restore after an agent timeout, then retain the reviewed
    # one-shot verifier-timeout retry.
    install_agent_timeout_verifier_recovery()
    record_unbounded_agent_timeout_receipt()
    from harbor.cli.main import app

    app()


if __name__ == "__main__":
    main()


__all__ = [
    "AGENT_TIME_BUDGET_RECEIPT_NAME",
    "AGENT_TIME_BUDGET_RECEIPT_SCHEMA",
    "AGENT_TIMEOUT_VERIFIER_RECOVERY_POLICY",
    "HARBOR_UNBOUNDED_CLI_MODULE",
    "TB21_DATASET_SOURCE",
    "TB21_OFFICIAL_AGENT_TIMEOUT_POLICY",
    "TB21_PER_TASK_AGENT_TIMEOUT_SECONDS",
    "TB21_UNIFORM_AGENT_TIMEOUT_POLICY",
    "TB21_UNIFORM_AGENT_TIMEOUT_SOURCE",
    "TB21_LEGACY_8H_AGENT_TIMEOUT_POLICY",
    "UNBOUNDED_AGENT_IMPORT_PATHS",
    "UNBOUNDED_AGENT_TIMEOUT_POLICY",
    "VERIFIER_TIMEOUT_RETRY_POLICY",
    "agent_import_uses_unbounded_timeout",
    "install_agent_timeout_verifier_recovery",
    "install_unbounded_agent_timeout",
    "install_verifier_timeout_resilience",
    "main",
    "record_unbounded_agent_timeout_receipt",
]
