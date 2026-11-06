"""Crash-safe shared-filesystem dispatcher for Terminal-Bench Harbor trials.

Each immutable cell is one ``(pinned task, official attempt)``.  A per-cell
``flock`` is the sole ownership authority; heartbeat leases exist only for
observability.  Every cell owns one deterministic single-trial Harbor job so a
worker replacement can adopt a completed result or use Harbor's native resume
without silently manufacturing another official attempt.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import traceback
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
import httpx
from harbor.job_plan import JobPlan
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import TaskConfig
from harbor.models.trial.result import TrialResult
from harbor.tasks.client import TaskClient

from terminal_bench_eval.harbor_pinned_task_cache import (
    PINNED_PACKAGE_RESOLUTION_POLICY,
    PinnedTaskVerificationError,
    install_pinned_package_resolution,
    verify_pinned_package_task,
)
from terminal_bench_eval.worker_network_circuit import NetworkCircuit, image_target
from terminal_bench_eval.verifier_bootstrap_replay import (
    POLICY as VERIFIER_BOOTSTRAP_REPLAY_POLICY,
    POLICIES as VERIFIER_BOOTSTRAP_REPLAY_POLICIES,
    LOG_KEY as VERIFIER_BOOTSTRAP_LOG_KEY,
    bootstrap_failure as _verifier_bootstrap_failure,
    validate_completed_ctrf,
)
from terminal_bench_eval.codex_r2eact_artifacts import (
    CONTROLLER_ARTIFACT_DIRNAME,
    GATEWAY_EVENTS_FILENAME,
    GATEWAY_RECEIPT_SUMMARY_SCHEMA,
    GATEWAY_RECEIPTS_FILENAME,
    GATEWAY_RECEIPTS_JSONL_FILENAME,
    GATEWAY_RECEIPTS_SUMMARY_FILENAME,
)
from terminal_bench_eval.codex_r2eact_receipts import (
    GatewayReceiptPersistenceError,
    decode_receipt_record,
)
from terminal_bench_eval.harbor_unbounded_agent import (
    AGENT_TIME_BUDGET_RECEIPT_NAME,
    AGENT_TIME_BUDGET_RECEIPT_SCHEMA,
    HARBOR_UNBOUNDED_CLI_MODULE,
    TB21_DATASET_SOURCE,
    TB21_OFFICIAL_AGENT_TIMEOUT_POLICY,
    TB21_LEGACY_8H_AGENT_TIMEOUT_POLICY,
    TB21_PER_TASK_AGENT_TIMEOUT_SECONDS,
    TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
    TB21_UNIFORM_AGENT_TIMEOUT_SOURCE,
    UNBOUNDED_AGENT_TIMEOUT_POLICY,
    agent_import_uses_unbounded_timeout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOBS_ROOT = PROJECT_ROOT / "eval" / "terminal_bench" / "jobs"
DEFAULT_HARBOR = (
    PROJECT_ROOT / "eval" / "terminal_bench" / "compressor_venv" / "bin" / "harbor"
)
POOL_SCHEMA = "terminal_bench_shared_pool_v1"
POOL_STATE_SCHEMA = "terminal_bench_pool_state_v1"
POOL_RESULT_SCHEMA = "terminal_bench_pool_result_v1"
CLAIM_PRIORITY_SCHEMA = "terminal_bench_pool_claim_priority_v1"
CELL_DISPATCH_SCHEMA = "terminal_bench_pool_dispatch_v1"
CELL_STATE_SCHEMA = "terminal_bench_pool_cell_state_v1"
CELL_RESULT_SCHEMA = "terminal_bench_pool_cell_result_v1"
CELL_EXTERNAL_RESERVATION_SCHEMA = "terminal_bench_pool_external_trial_reservation_v1"
EXTERNAL_TRIAL_IMPORT_RECEIPT_SCHEMA = "terminal_bench_pool_external_trial_import_v1"
VERIFIER_REPLAY_MANIFEST_SCHEMA = "terminal_bench_verifier_replay_manifest_v1"
VERIFIER_REPLAY_RECEIPT_SCHEMA = "terminal_bench_verifier_replay_commit_v1"
VERIFIER_REPLAY_COMMIT_DIR_NAME = "verifier-replay-commit-v1"
VERIFIED_AGENT_SIGTERM_RECONCILIATION_SCHEMA = (
    "terminal_bench_verified_agent_sigterm_reconciliation_v1"
)
VERIFIED_AGENT_SIGTERM_RECONCILIATION_DIR_NAME = (
    "verified-agent-sigterm-reconciliation-v1"
)
ATTEMPT_SELECTION_SCHEMA = "terminal_bench_pool_attempt_selection_v1"
QUEUED_ATTEMPT_RETIREMENT_SCHEMA = "terminal_bench_pool_queued_attempt_retirement_v1"
QUEUED_ATTEMPT_RETIREMENT_POLICY = "keep_single_official_attempt_v1"
QUEUED_ATTEMPT_RETIREMENT_DIR_NAME = "queued-attempt-retirement-v1"
TASK_SELECTION_SCHEMA = "terminal_bench_pool_task_selection_v1"
QUEUED_TASK_RETIREMENT_SCHEMA = "terminal_bench_pool_queued_task_retirement_v1"
QUEUED_TASK_RETIREMENT_POLICY = "retire_selected_queued_tasks_v1"
QUEUED_TASK_RETIREMENT_DIR_NAME = "queued-task-retirement-v1"
HARBOR_RESUME_CONFIG_INCIDENT = "harbor_normalized_resume_config_mismatch_v1"
HARBOR_RESUME_CONFIG_ERROR_MESSAGE = (
    "existing Harbor job config violates the cell contract"
)
HARBOR_RESUME_CONFIG_TERMINAL_ARCHIVE = (
    "committed-terminal-harbor-resume-config-incident"
)
HARBOR_RESUME_CONFIG_JOB_SNAPSHOT = "harbor-job-incomplete-before-resume-config-repair"
PINNED_CANDIDATE_TIMEOUT_ZERO_KIND = "pinned_harbor_candidate_timeout_zero"
PINNED_CANDIDATE_TIMEOUT_ZERO_POLICY = (
    "exact_model_extraction_candidate_timeout_zero_v1"
)
VERIFIER_REPLAY_KINDS = frozenset(
    {"host_pytest", "harbor_replay", PINNED_CANDIDATE_TIMEOUT_ZERO_KIND}
)
VERIFIER_REPLAY_EVIDENCE_CATEGORIES = (
    "artifact",
    "test",
    "report",
    "replay",
)
MAX_VERIFIER_REPLAY_EVIDENCE_FILES = 64
MAX_VERIFIER_REPLAY_TOTAL_BYTES = 256 * 1024 * 1024
COMPOSITE_VERIFIER_TIMEOUT_REPLAY_POLICY = (
    "pinned_secondary_verifier_timeout_after_agent_exit_137_v1"
)
COMPOSITE_VERIFIER_TIMEOUT_RECEIPT_NAME = "verifier-timeout-resilience.json"
COMPOSITE_VERIFIER_TIMEOUT_RECEIPT_SCHEMA = "terminal_bench_verifier_timeout_v1"
COMPOSITE_VERIFIER_TIMEOUT_RECEIPT_POLICY = (
    "persist_all_retry_reviewed_network_bootstrap_once_v1"
)
COMPOSITE_VERIFIER_TIMEOUT_HOST_LOG_MAX_BYTES = 2 * 1024 * 1024
COMPOSITE_VERIFIER_TIMEOUT_PARTIAL_MAX_BYTES = 64 * 1024 * 1024
WORKER_SCHEMA = "terminal_bench_pool_worker_v1"
WORKER_CLAIM_PRIORITY_POLICY = "priority_tasks_before_remaining_v1"
WORKER_EXCEPTION_SCHEMA = "terminal_bench_pool_worker_exception_v1"
RUN_RECORD_SCHEMA = "terminal_bench_pool_execution_v1"
TASK_CACHE_RECEIPT_SCHEMA = "terminal_bench_task_cache_ready_v1"
TASK_CACHE_TRANSPORT_SCHEMA = "terminal_bench_task_cache_transport_retry_v1"
TASK_CACHE_TRANSPORT_ATTEMPTS = 3
TASK_CACHE_TRANSPORT_BACKOFF_SECONDS = 1.0
PROOT_ARCHIVE_RECEIPT_SCHEMA = "terminal_bench_proot_infrastructure_harbor_archive_v1"
PROOT_AUTO_ARCHIVE_NAME = "harbor-job-retryable-proot-infrastructure"
PROOT_ARCHIVE_RECEIPT_NAME = "proot-infrastructure.json"
PROOT_AUTO_DETECTION_PHASE = "precommit_auto_retry"
PROOT_RUNTIME_RECEIPT_NAME = "proot-runtime.json"
PROOT_RUNTIME_RECEIPT_SCHEMA = "terminal_bench_proot_runtime_v1"
PROOT_IMAGE_PULL_RECEIPT_NAME = "proot-image-pull.json"
PROOT_IMAGE_PULL_RECEIPT_SCHEMA = "terminal_bench_proot_image_pull_v1"
PROOT_IMAGE_PULL_POLICY = "bounded_outer_retry_reuse_partial_oci_v1"
PROOT_IMAGE_PULL_ATTEMPTS = 3
PROOT_IMAGE_PULL_BACKOFF_SECONDS = (1.0, 2.0)
PROOT_IMAGE_PULL_SKOPEO_RETRY_TIMES = 3
PROOT_PACKAGE_BRIDGE_RECEIPT_NAME = "proot-package-bridge.json"
PROOT_PACKAGE_BRIDGE_RECEIPT_SCHEMA = "terminal_bench_proot_package_bridge_v1"
PROOT_PACKAGE_BRIDGE_POLICY = "pinned_profile_proot_bind_host_chroot_v1"
PROOT_GUEST_UTILITY_RECEIPT_STEM = "proot-guest-utilities"
PROOT_GUEST_UTILITY_RECEIPT_SCHEMA = "terminal_bench_proot_guest_utilities_v2"
PROOT_GUEST_UTILITY_POLICY = "apt_profile_then_proot_rpc_verification_v1"
PROOT_GUEST_UTILITY_PROFILE = "tb20-file-xxd-rg-v1"
PROOT_GUEST_UTILITY_APT_ATTEMPTS = 3
PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS = (1.0, 2.0)
SGLANG_AUTO_ARCHIVE_NAME = "harbor-job-retryable-sglang"
SGLANG_ARCHIVE_RECEIPT_NAME = "retryable-sglang-infrastructure.json"
SGLANG_ARCHIVE_RECEIPT_SCHEMA = "terminal_bench_retryable_sglang_harbor_archive_v1"
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,126}")
_SAFE_WORKER_RE = re.compile(r"[^A-Za-z0-9._-]+")


class PoolError(RuntimeError):
    """The immutable pool or one of its committed artifacts is invalid."""


class ExternalInterruption(BaseException):
    """A scheduler/user signal interrupted a worker without consuming retry budget."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"worker interrupted by signal {signum}")
        self.signum = signum


class RetryableSGLangTrialError(RuntimeError):
    """Harbor recorded a serving transport failure as a terminal trial."""

    def __init__(self, failure: Mapping[str, Any], archive_path: Path) -> None:
        self.failure = dict(failure)
        self.archive_path = Path(archive_path)
        super().__init__(
            "retryable SGLang infrastructure failure: "
            f"{self.failure.get('exception_type', 'unknown')}: "
            f"{self.failure.get('message', '')}"
        )


class RetryablePRootTrialError(RuntimeError):
    """A strictly identified local PRoot/runtime infrastructure failure."""

    def __init__(self, failure: Mapping[str, Any], archive_path: Path) -> None:
        self.failure = dict(failure)
        self.archive_path = Path(archive_path)
        super().__init__(
            "retryable PRoot infrastructure failure: "
            f"{self.failure.get('exception_type', 'unknown')}: "
            f"{self.failure.get('message', '')}"
        )


class RetryableTaskCacheTransportError(RuntimeError):
    """A pinned task could not be fetched because object storage was unreachable."""

    def __init__(self, task_ref: str, exc: BaseException, *, attempts: int) -> None:
        leaves = (
            _flatten_exception_group(exc)
            if isinstance(exc, BaseExceptionGroup)
            else [exc]
        )
        self.failure = {
            "infrastructure_kind": "task_cache_transport",
            "task_ref": task_ref,
            "download_attempts": int(attempts),
            "exceptions": [
                {"type": type(leaf).__name__, "message": str(leaf)} for leaf in leaves
            ],
        }
        super().__init__(
            "retryable pinned task-cache transport failure after "
            f"{attempts} attempts: {task_ref}"
        )


class SGLangUnavailable(RuntimeError):
    """The worker must yield so its launcher can recover the serving process."""


_RETRYABLE_SGLANG_RUNTIME_MESSAGE_PREFIXES = (
    "sglang /generate request failed:",
    "sglang /generate returned http ",
    "sglang /generate returned a non-object response",
    "sglang /generate failed:",
    "sglang cache flush request failed:",
    "sglang cache flush returned http ",
    "sglang encode response has no ",
    "sglang encode returned ",
    "sglang health request failed:",
    "sglang health returned http ",
    "sglang server info request failed:",
    "sglang server info returned http ",
)
_STRICT_SGLANG_AGENT_IMPORT_PATHS = frozenset(
    {
        "terminal_bench_eval.terminus_r2eact_agent_v050:TerminusR2EActAgentV050",
        "terminal_bench_eval.terminus_r2eact_agent_v051:TerminusR2EActAgentV051",
        "terminal_bench_eval.terminus_r2eact_agent_v052:TerminusR2EActAgentV052",
        "terminal_bench_eval.terminus_r2eact_agent_v053:TerminusR2EActAgentV053",
    }
)
_LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH = (
    "eval.terminal_bench.proot_environment:ProotEnvironment"
)
_PROOT_RPC_FAILURE_MESSAGES = {
    "PRoot command server is not running": "proot_rpc_process_not_running",
    "Lost contact with the PRoot command server": "proot_rpc_transport_lost",
}
_PROOT_IMAGE_PULL_TRANSPORT_PATTERNS = (
    ("tls_handshake_timeout", re.compile(r"\btls handshake timeout\b", re.I)),
    ("unexpected_eof", re.compile(r"\bunexpected eof\b", re.I)),
    (
        "connection_reset",
        re.compile(r"\bconnection reset(?: by peer)?\b", re.I),
    ),
    ("io_timeout", re.compile(r"\bi/o timeout\b", re.I)),
    (
        "response_header_timeout",
        re.compile(r"\btimeout awaiting response headers\b", re.I),
    ),
    (
        "connection_timeout",
        re.compile(r"\b(?:connection|connect) timed out\b", re.I),
    ),
    (
        "temporary_dns_failure",
        re.compile(r"\btemporary failure in name resolution\b", re.I),
    ),
    ("dns_host_not_found", re.compile(r"\bno such host\b", re.I)),
    (
        "network_unreachable",
        re.compile(r"\bnetwork is unreachable\b", re.I),
    ),
    ("broken_pipe", re.compile(r"\bbroken pipe\b", re.I)),
    ("connection_refused", re.compile(r"\bconnection refused\b", re.I)),
    (
        "registry_http_transient",
        re.compile(
            r"\b(?:status(?: code)?|http(?: response)? status)\s*"
            r"(?:408|429|5[0-9]{2})\b|\btoo many requests\b|\btoomanyrequests\b",
            re.I,
        ),
    ),
)
_PROOT_RUNTIME_FAILURE_EVENTS = frozenset(
    {
        "direct_exec_cleanup_incomplete",
        "environment_stop_incomplete",
        "rpc_process_not_running",
        "rpc_runtime_cleanup_incomplete",
        "rpc_transport_cleanup_incomplete",
        "rpc_transport_poisoned",
    }
)
_PROOT_RUNTIME_FAILURE_REASONS = frozenset(
    {
        "rpc_process_not_running",
        "wire_exec_cancelled",
        "wire_protocol_invalid_response",
        "wire_request_transport_closed",
        "wire_response_eof",
        "wire_timeout_response_grace_expired",
    }
)
_PROOT_CLEANUP_FAILURE_EVENTS = frozenset(
    {
        "direct_exec_cleanup_incomplete",
        "environment_stop_incomplete",
        "rpc_runtime_cleanup_incomplete",
        "rpc_transport_cleanup_incomplete",
    }
)
_PACKAGE_BRIDGE_FAILURE_STATUSES = frozenset(
    {"cancelled", "cleanup_failed", "cleanup_incomplete", "failed", "unhealthy"}
)
_PACKAGE_BRIDGE_FAILURE_EVENTS = frozenset(
    {
        "broker_health_check_failed",
        "broker_stop_failed",
        "environment_stop_incomplete",
        "startup_cancelled",
        "startup_failed",
    }
)
_PACKAGE_BRIDGE_CLEANUP_STATUSES = frozenset({"cleanup_failed", "cleanup_incomplete"})
_PACKAGE_BRIDGE_CLEANUP_EVENTS = frozenset(
    {"broker_stop_failed", "environment_stop_incomplete"}
)
_PROOT_GUEST_UTILITY_APT_COMMANDS = {
    "apt_update": (
        "DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none "
        "apt-get -o Acquire::Retries=3 update"
    ),
    "apt_install": (
        "DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none "
        "apt-get -o Acquire::Retries=3 install -y "
        "--no-install-recommends --no-upgrade file xxd ripgrep"
    ),
}
_PROOT_GUEST_UTILITY_BOUNDED_APT_NETWORK_TIMEOUT_SEC = 20
_PROOT_GUEST_UTILITY_BOUNDED_APT_ATTEMPT_TIMEOUT_SEC = 75
_PROOT_GUEST_UTILITY_BOUNDED_APT_COMMANDS = {
    "apt_update": (
        "DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none "
        "apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=20 "
        "-o Acquire::https::Timeout=20 update"
    ),
    "apt_install": (
        "DEBIAN_FRONTEND=noninteractive APT_LISTCHANGES_FRONTEND=none "
        "apt-get -o Acquire::Retries=0 -o Acquire::http::Timeout=20 "
        "-o Acquire::https::Timeout=20 install -y "
        "--no-install-recommends --no-upgrade file xxd ripgrep"
    ),
}
_PROOT_GUEST_UTILITY_APT_TRANSPORT_PATTERNS = (
    (
        "apt_http_transient",
        re.compile(r"(?m)^[EW]:\s*Failed to fetch\s+https?://\S+\s+(?:408|429|5[0-9]{2})\b", re.I),
    ),
    ("command_timeout", re.compile(r"\bcommand timed out\b", re.I)),
    ("connection_failed", re.compile(r"\bconnection failed\b", re.I)),
    (
        "connection_timeout",
        re.compile(r"\b(?:connection|connect) timed out\b", re.I),
    ),
    ("could_not_connect", re.compile(r"\bcould not connect to\b", re.I)),
    (
        "temporary_dns_failure",
        re.compile(
            r"\btemporary failure (?:in name resolution|resolving)\b", re.I
        ),
    ),
    ("tls_handshake_timeout", re.compile(r"\btls handshake timeout\b", re.I)),
    ("unexpected_eof", re.compile(r"\bunexpected eof\b", re.I)),
    (
        "connection_reset",
        re.compile(r"\bconnection reset(?: by peer)?\b", re.I),
    ),
    (
        "proxy_http_transient",
        re.compile(
            r"\bproxy\b[^\n]*(?:status\s*)?(?:408|429|5[0-9]{2})\b",
            re.I,
        ),
    ),
)
_PROOT_GUEST_UTILITY_APT_UPDATE_INCOMPLETE_PATTERNS = (
    (
        "failed_to_fetch",
        re.compile(r"(?m)^W:\s*Failed to fetch\s+\S+"),
    ),
    (
        "index_files_ignored",
        re.compile(r"(?m)^W:\s*Some index files failed to download\b"),
    ),
)
_PROOT_GUEST_UTILITY_APT_UPDATE_SEMANTIC_FAILURE = (
    "apt_update_incomplete_indexes"
)
_PROOT_GUEST_UTILITY_ENVIRONMENT_START_TIMEOUT_MESSAGE = (
    "Environment start timed out after 600.0 seconds"
)
_CODEX_R2EACT_AGENT_IMPORT_PATH = (
    "terminal_bench_eval.codex_r2eact_agent:CodexR2EActAgent"
)
_VERIFIED_AGENT_TIMEOUT_REWARD_POLICY = (
    "preserve_tb21_verifier_reward_after_agent_timeout_v1"
)
_CODEX_HOST_RG_INCIDENT = "codex_host_rg_missing_2026-08-28_v1"
_CODEX_HOST_RG_ERROR_MESSAGE = (
    "GrepSeek corpus preflight failed: exit_code=127; "
    "stderr=\"execution error: whitelisted command 'rg' is not installed\"; "
    "unexpected stdout=''; information_lines disagree with the single matched id"
)
_CODEX_HIGH_DEMAND_MESSAGE = (
    "We're currently experiencing high demand, which may cause temporary errors."
)
_CODEX_GATEWAY_TIMEOUT_INCIDENT = "codex_recursive_memory_chain_timeout_2026-08-31_v1"
_CODEX_GATEWAY_TIMEOUT_CLASSIFIER = (
    "codex_gateway_recursive_memory_chain_timeout_after_progress_v1"
)
_CODEX_GATEWAY_TIMEOUT_TEST_NAMES = (
    "test_outputs.py::test_vm_execution",
    "test_outputs.py::test_frame_bmp_exists",
    "test_outputs.py::test_frame_bmp_similar_to_reference",
)
# This repair is deliberately bounded to the two audited, already committed
# attempt-1 cells.  The hashes freeze every authority-bearing projection and
# host-owned incident artifact.  Tests replace this map with fixture hashes;
# production callers cannot broaden the cohort by merely manufacturing a
# similar traceback or Codex transcript.
_CODEX_GATEWAY_TIMEOUT_INCIDENT_CELLS: dict[str, dict[str, Any]] = {
    "tb_1e7148eb973d0092d64cbdd33a40f289": {
        "task_name": "terminal-bench/make-doom-for-mips",
        "task_ref": (
            "sha256:2d83dd3dee8e0f055e09973934cf0d7e3169a9cd90704cba5c8940b170be9498"
        ),
        "source": "terminal-bench/terminal-bench-2-1",
        "official_attempt_index": 1,
        "verifier_test_names": _CODEX_GATEWAY_TIMEOUT_TEST_NAMES,
        "artifact_sha256": {
            "result.json": (
                "baff0fb31cb9159571f2d5af48deadadce4908fc2e96c86b6fc130312ac2529b"
            ),
            "state.json": (
                "53a8a142daeeb0c883b3c34f6846bca93949906e4fff3a6360911a7bd44b63b3"
            ),
            "dispatch.json": (
                "bfa7345523226fcb64f720dfd2416bccf751690c44a20e9e0e04dcdcaa01c719"
            ),
            "run.json": (
                "baa7d18d4b5fac922c1b6814054e73a2fa13f1488937b50d4ebb88a757a06259"
            ),
            "trial/result.json": (
                "35559c4142784918a01ead1c62b39ca40ab97f90a23d4f58604d0a709a7d210c"
            ),
            "agent/codex.txt": (
                "2d06e8f692c25d96ffdb2719e6ff5ffb543c529f3da1dc3eef7473d4b0e22c5a"
            ),
            "agent/codex-r2eact-gateway-events.jsonl": (
                "bd85fa7c8bdf839d5c8d3a928be7f96f9362fefb4e10147c184cf51d9c83fc98"
            ),
            "agent/codex-r2eact-gateway-receipts.json": (
                "40484d0c48382517d594737087828d833574d70c06007421e09df3eb2567ede2"
            ),
            "verifier/reward.txt": (
                "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
            ),
            "verifier/ctrf.json": (
                "dda22a3ef988d524d052af2209f9fcd424547f2fb02cf3b42fc716d25cb8183a"
            ),
            "proot-runtime.json": (
                "ecdc7b707ca49f82485cc44e333264e9fac08acbb2e29eb0aa4979d9e04e9986"
            ),
            "proot-package-bridge.json": (
                "febfda1a876e49f3f411c3f72190911c68d84e3bed030ba0e18521069ff84e01"
            ),
        },
    },
    "tb_b0c8f713c22248a495c0c86617b4e42a": {
        "task_name": "terminal-bench/make-mips-interpreter",
        "task_ref": (
            "sha256:41a55da0abec5d7b32a0c2321f8b18e84000ca8074ae62c6874d6ed4a3a1cd3c"
        ),
        "source": "terminal-bench/terminal-bench-2-1",
        "official_attempt_index": 1,
        "verifier_test_names": _CODEX_GATEWAY_TIMEOUT_TEST_NAMES,
        "artifact_sha256": {
            "result.json": (
                "9327d4c00e015261620e5303dbe4cf242e28fe2c6a9e1ea378e58eb683d3b57b"
            ),
            "state.json": (
                "f82413007bc817dc7cbbd0d5495024e08e7b7667701c66683d6d7d556d8ca5ce"
            ),
            "dispatch.json": (
                "36ec0db5dd34007a91ff458d82503914720789cf25f3277edbdd918fc0ab397a"
            ),
            "run.json": (
                "1138326e5f9251e00bd1a0ddce3a6acd16153d04e5eda403cded7f6500864d55"
            ),
            "trial/result.json": (
                "f4bf6e31cd2f7baa6a6418081e5f3fe2e3d74950874b1e7b4c13dc928a9da549"
            ),
            "agent/codex.txt": (
                "c533d6d9d97020d6fdca8dfd4886f8ce60df062bd99fce0c1664677bd864ed87"
            ),
            "agent/codex-r2eact-gateway-events.jsonl": (
                "2626509c9b2e627f253977c8f8b943bd2474a65a07ef2da8aede43afdda7c0ff"
            ),
            "agent/codex-r2eact-gateway-receipts.json": (
                "6f35097e793a34e1f796976e05a7aebd29e7eb66bc68a01465b9f0487ec577d0"
            ),
            "verifier/reward.txt": (
                "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa"
            ),
            "verifier/ctrf.json": (
                "2963be8b1078184091c3922185ccb2ec1bb65f8cd4a466a5d83c761766839e28"
            ),
            "proot-runtime.json": (
                "a199bfd3c38af4daa66fe0ab00982759885581e6cb97c35d4bbee27ee88db812"
            ),
            "proot-package-bridge.json": (
                "edd5f08e77e8209b17cc7f4f43124c6fef9bc3174c929b81b7db7f78711eaf66"
            ),
        },
    },
}
_CODEX_GATEWAY_TIMEOUT_ARTIFACT_KEYS = frozenset(
    {
        "result.json",
        "state.json",
        "dispatch.json",
        "run.json",
        "trial/result.json",
        "agent/codex.txt",
        "agent/codex-r2eact-gateway-events.jsonl",
        "agent/codex-r2eact-gateway-receipts.json",
        "verifier/reward.txt",
        "verifier/ctrf.json",
        "proot-runtime.json",
        "proot-package-bridge.json",
    }
)
# Harbor preserves the first exception raised by a trial.  In this one audited
# attempt the agent command exited 137, then Harbor still entered the shared
# verifier and its 900-second timeout was recorded only in the host-owned trial
# log and timeout receipt.  This allowlist is intentionally usable only by the
# verifier-replay commit gate below; ordinary scoring and retry repair do not
# consult it.  Every authority-bearing byte is frozen so a similar-looking log
# cannot broaden replay eligibility.
_COMPOSITE_VERIFIER_TIMEOUT_REPLAY_ARTIFACT_KEYS = frozenset(
    {
        "result.json",
        "state.json",
        "dispatch.json",
        "run.json",
        "trial-result.json",
        "trial/config.json",
        "trial/lock.json",
        "trial/trial.log",
        "trial/verifier/verifier-timeout-resilience.json",
        "trial/verifier/test-stdout.timeout-1.partial.txt",
        "trial/proot-runtime.json",
        "trial/proot-package-bridge.json",
    }
)
_COMPOSITE_VERIFIER_TIMEOUT_REPLAY_CELLS: dict[str, dict[str, Any]] = {
    "tb_7c0093759a8408468c842a4e930d8b90": {
        "job_name": (
            "qwen38-tb21-codex-r2eact-vision-web-plan-wm4-block1024-"
            "v012-attempt1"
        ),
        "config_contract_hash": (
            "32a7f6949d22258f83d8ec59a853ed07d8a421c82d03855726cc299fd83fffdd"
        ),
        "task_name": "terminal-bench/model-extraction-relu-logits",
        "task_ref": (
            "sha256:1ae5045ad68b5d34c3398b612066a07c4a08b6dc330d28868ec4021e17c94b17"
        ),
        "source": "terminal-bench/terminal-bench-2-1",
        "official_attempt_index": 1,
        "execution_attempt": 1,
        "trial_name": "model-extraction-relu-logits__9k9hcna",
        "trial_id": "06297b2b-866b-40c4-b972-0f77b52f7b22",
        "task_checksum": (
            "8b3048320aa04676089f9240563c2d7d381be03d8810bcf5f5d336cab538f523"
        ),
        "configured_verifier_timeout_seconds": 900.0,
        "verifier_script_sha256": (
            "3b2d8f5e0353c2bab3515d6387468a863a4f9f241ee131c73f545f3fdf9e2d52"
        ),
        "process_snapshot": {
            "complete": True,
            "network_processes": ["uv"],
            "test_processes": ["python"],
            "other_process_count": 7,
        },
        "artifact_sha256": {
            "result.json": (
                "1302206eb9420b6a870393b290a9327d38b4c430e65118a5a99b3c995383885a"
            ),
            "state.json": (
                "3c10e6e664a1cec3cb5677fcfd4d50096e57c6e3b815e4d12360400dc6175622"
            ),
            "dispatch.json": (
                "f00734e971b2c89e909c3af828c0205b328e66708bc06b44fd90110e7611d9fd"
            ),
            "run.json": (
                "72e219283ef0ab60a8c56940b16baa87fd7f86743f45ba5688ad29de0907cc31"
            ),
            "trial-result.json": (
                "f66d8c3f4aac54deca7c3de7b038de75cbdc240c57d4393b27cb49884e3967a9"
            ),
            "trial/config.json": (
                "638e8e53823106f3d29c8712d699bc3a240612bb4334a32b3bd0d447f18c8185"
            ),
            "trial/lock.json": (
                "08dff24fa68e59418dc342ea14a3808cbfcc694f3d14b5d85a354df0b2ffe6f9"
            ),
            "trial/trial.log": (
                "6bf6fdf10e0abb98387d42f8a0aa9f174d4848c6ec249fd2f34d11f7710a1585"
            ),
            "trial/verifier/verifier-timeout-resilience.json": (
                "b3a7a569ce11deffe863dc31ab0549d0f3743f1649c0e1edcf083e0491cef9aa"
            ),
            "trial/verifier/test-stdout.timeout-1.partial.txt": (
                "502f0c9e4b9ca8aff203323ffb3562057c597c0641583268794eaefc0a6c40fb"
            ),
            "trial/proot-runtime.json": (
                "3386b2bc8bcfcba2d267f368cb26f0828f78a32625227ac0f48f34ffc2739266"
            ),
            "trial/proot-package-bridge.json": (
                "78777067d3a889ba2887663b3f0f4a6b21518484a31fa22797de4d8ef1208058"
            ),
        },
    }
}
# The same immutable candidate above was replayed once with a model-free agent.
# The official verifier again exhausted its 900-second budget after pytest had
# entered the one test, whose only blocking operation is
# ``python3 /app/steal.py``.  All setup downloads had completed before pytest
# started.  The durable timeout snapshot calls the surviving uvx parent
# ``uv``; for this exact run only, the official test bytes and the completed
# stdout prefix disambiguate it from dependency bootstrap.  This allowlist is
# intentionally not a general "verifier timeout means zero" rule: every input,
# output, task, process snapshot, and original model-session byte is frozen.
_PINNED_CANDIDATE_TIMEOUT_ZERO_EVIDENCE_CATEGORIES = {
    "artifact": frozenset({"candidate-steal-py"}),
    "test": frozenset(
        {"official-task-toml", "official-test-sh", "official-test-outputs-py"}
    ),
    "report": frozenset(
        {
            "replay-result-json",
            "replay-trial-config-json",
            "replay-job-config-json",
            "replay-lock-json",
            "replay-trial-log",
            "replay-timeout-receipt-json",
            "replay-timeout-stdout",
            "replay-proot-runtime-json",
            "replay-proot-package-bridge-json",
            "replay-proot-image-pull-json",
            "replay-proot-guest-utilities-json",
            "source-extraction-manifest-json",
            "source-model-session-jsonl",
            "source-job-config-json",
        }
    ),
    "replay": frozenset({"replay-agent-receipt-json"}),
}
_PINNED_CANDIDATE_TIMEOUT_ZERO_EVIDENCE_KEYS = frozenset(
    name
    for names in _PINNED_CANDIDATE_TIMEOUT_ZERO_EVIDENCE_CATEGORIES.values()
    for name in names
)
_PINNED_CANDIDATE_TIMEOUT_ZERO_CELLS: dict[str, dict[str, Any]] = {
    "tb_7c0093759a8408468c842a4e930d8b90": {
        "job_name": (
            "qwen38-tb21-codex-r2eact-vision-web-plan-wm4-block1024-"
            "v012-attempt1"
        ),
        "config_contract_hash": (
            "32a7f6949d22258f83d8ec59a853ed07d8a421c82d03855726cc299fd83fffdd"
        ),
        "task_name": "terminal-bench/model-extraction-relu-logits",
        "task_ref": (
            "sha256:1ae5045ad68b5d34c3398b612066a07c4a08b6dc330d28868ec4021e17c94b17"
        ),
        "source": "terminal-bench/terminal-bench-2-1",
        "official_attempt_index": 1,
        "execution_attempt": 1,
        "source_trial_name": "model-extraction-relu-logits__9k9hcna",
        "replay_trial_name": "model-extraction-relu-logits__8cWPy7z",
        "replay_trial_id": "3ef66ab5-133c-41df-8a76-0cbf3e053996",
        "task_checksum": (
            "8b3048320aa04676089f9240563c2d7d381be03d8810bcf5f5d336cab538f523"
        ),
        "configured_verifier_timeout_seconds": 900.0,
        "candidate_sha256": (
            "92f9135d011770662b5b08f19a554a8d8b5d1f7e5e61a4ca8298e94524c645e7"
        ),
        "source_session_zero_based_ordinal": 629,
        "source_command_sha256": (
            "4f3a9fd5c88c1aebe21150e9576304033239b7686c772893a4260b5e5b999c28"
        ),
        "process_snapshot": {
            "complete": True,
            "network_processes": ["uv"],
            "test_processes": [],
            "other_process_count": 6,
        },
        "evidence_sha256": {
            "candidate-steal-py": (
                "92f9135d011770662b5b08f19a554a8d8b5d1f7e5e61a4ca8298e94524c645e7"
            ),
            "official-task-toml": (
                "eb64dfaf8ae88868911ae28855cdc4f767de464001e2704919015b595be6dc29"
            ),
            "official-test-sh": (
                "3b2d8f5e0353c2bab3515d6387468a863a4f9f241ee131c73f545f3fdf9e2d52"
            ),
            "official-test-outputs-py": (
                "e5189c7eb9942ede55e1b48a80388590b80a486b75497613f257b3373ad9b2c9"
            ),
            "replay-result-json": (
                "4dbfada3ffd57bb469f21e428a8dc374d37bbad78d24b92cdaf61ba3aa87e997"
            ),
            "replay-trial-config-json": (
                "ee61c793e178f50449686634064785ea612ce5f2b1ccf73f2a337de39df8cedf"
            ),
            "replay-job-config-json": (
                "cc48b04fce21226da164ff037d20d74ee887ef685625bddc2ab36baf34f74d9b"
            ),
            "replay-lock-json": (
                "95ca9eb52d059baadd296290e3075937ad0258b7a5f178d2a9e6f3994af44d3b"
            ),
            "replay-trial-log": (
                "b5a42d90a25299ee4bd8513218e14825519fa0ef84bc83a7f75b0ab8e47378e1"
            ),
            "replay-timeout-receipt-json": (
                "af6e15373b9285997f72b95fde4ba2d39bf2da4237755cb57001c02fbd29872c"
            ),
            "replay-timeout-stdout": (
                "ea2e26316d6dbde3d00625152003bff61fe1167f32db921763d52c0ff9b669e8"
            ),
            "replay-proot-runtime-json": (
                "c6b414fda010654b312cd906fa70a03b033560fefa1ed12d9bd16b8f86449b18"
            ),
            "replay-proot-package-bridge-json": (
                "9b857fd2e950a8aae70a49eedd48c9f87f96a7633956e0aacb27530ec5ba742c"
            ),
            "replay-proot-image-pull-json": (
                "02238538c787f39308db884a5ec679d273b17b08f77a0efd5cbcf90299ea7bbc"
            ),
            "replay-proot-guest-utilities-json": (
                "eb47a5aa0ff3ff2348a15c641e99872d15c23495a09ba41b4b8a42d2665cc8dc"
            ),
            "source-extraction-manifest-json": (
                "abd04770228aed5973da9b8c500d40028485871f3fcfed36a5f2d5d0042570c6"
            ),
            "source-model-session-jsonl": (
                "d42617f7f97bc9c157125db7b90c1fe36e8a93040c502389c7671a60f9c7464e"
            ),
            "source-job-config-json": (
                "6e09cc4606640520381731aa173a8dba7085aa04d4b73d0906438cd4b8e878e3"
            ),
            "replay-agent-receipt-json": (
                "40af89b88efe76b266e29574fb0753729cf1c47ad157847398959804622ae79f"
            ),
        },
    }
}
_CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT = "codex_gateway_request_timeout_v1"
_CODEX_GATEWAY_REQUEST_TIMEOUT_CLASSIFIER = (
    "host_owned_recursive_memory_gateway_timeout_after_progress_v1"
)
_CODEX_GATEWAY_REQUEST_TIMEOUT_TERMINAL_ARCHIVE = (
    "committed-terminal-codex-gateway-request-timeout"
)
_CODEX_GATEWAY_REQUEST_TIMEOUT_HARBOR_ARCHIVE = (
    "harbor-job-committed-codex-gateway-request-timeout"
)
_CODEX_GATEWAY_REQUEST_TIMEOUT_HARBOR_RECEIPT = "codex-gateway-request-timeout.json"
_CODEX_GATEWAY_REQUEST_TIMEOUT_BASE_ARTIFACT_KEYS = frozenset(
    {
        "result.json",
        "state.json",
        "dispatch.json",
        "run.json",
        "trial/result.json",
        "agent/codex.txt",
        f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/{GATEWAY_EVENTS_FILENAME}",
        "verifier/reward.txt",
        "verifier/ctrf.json",
        "proot-runtime.json",
        "proot-package-bridge.json",
    }
)
_CODEX_GATEWAY_REQUEST_TIMEOUT_LEGACY_ARTIFACT_KEYS = frozenset(
    {
        *_CODEX_GATEWAY_REQUEST_TIMEOUT_BASE_ARTIFACT_KEYS,
        f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/{GATEWAY_RECEIPTS_FILENAME}",
    }
)
_CODEX_GATEWAY_REQUEST_TIMEOUT_CANONICAL_ARTIFACT_KEYS = frozenset(
    {
        *_CODEX_GATEWAY_REQUEST_TIMEOUT_BASE_ARTIFACT_KEYS,
        f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/{GATEWAY_RECEIPTS_JSONL_FILENAME}",
        f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/{GATEWAY_RECEIPTS_SUMMARY_FILENAME}",
    }
)
_CODEX_GATEWAY_REQUEST_TIMEOUT_ARTIFACT_KEY_SETS = frozenset(
    {
        _CODEX_GATEWAY_REQUEST_TIMEOUT_LEGACY_ARTIFACT_KEYS,
        _CODEX_GATEWAY_REQUEST_TIMEOUT_CANONICAL_ARTIFACT_KEYS,
    }
)
_CODEX_GATEWAY_ARTIFACT_NAMES = frozenset(
    {
        "codex-r2eact-gateway-events.jsonl",
        "codex-r2eact-gateway-receipts.jsonl",
        "codex-r2eact-gateway-receipts.json",
        "codex-r2eact-gateway-receipts-summary.json",
        "codex-r2eact-rounds.jsonl",
        "memory.jsonl",
    }
)
_CODEX_LOCAL_STORAGE_INCIDENT = "codex_local_storage_infrastructure_v1"
_CODEX_LOCAL_STORAGE_TERMINAL_ARCHIVE = (
    "committed-terminal-codex-local-storage-infrastructure"
)
_CODEX_LOCAL_STORAGE_HARBOR_ARCHIVE = (
    "harbor-job-committed-codex-local-storage-infrastructure"
)
_CODEX_LOCAL_STORAGE_HARBOR_RECEIPT = "codex-local-storage-infrastructure.json"
_DISK_QUOTA_ERROR_MESSAGE = "[Errno 122] Disk quota exceeded"
_LEGACY_REACT_MEMORY_CAP_ERROR_MESSAGE = (
    "ReAct memory corpus exceeds the bounded 32 MiB episode limit"
)
_LOCAL_STORAGE_GATEWAY_EVENT_MAX_BYTES = 512 * 1024 * 1024
_LOCAL_STORAGE_GATEWAY_MATCHING_LINE_MAX_BYTES = 2 * 1024 * 1024
_LOCAL_STORAGE_LOG_MAX_BYTES = 32 * 1024 * 1024
_PROOT_LOCALHOST_INCIDENT = "proot_localhost_background_2026-08-31_v1"
_PROOT_LOCALHOST_INCIDENT_TASKS: dict[str, dict[str, Any]] = {
    "terminal-bench/headless-terminal": {
        "task_ref": (
            "sha256:203953871ebdae4efbf163af9499849368dab5e219b70d447e5ee9701ad382d9"
        ),
        "classifier": "verifier_localhost_8000_name_resolution_v1",
        "tests": {
            "test_outputs.py::test_import": "passed",
            "test_outputs.py::test_send_non_interactive_command": "passed",
            "test_outputs.py::test_send_interactive_command": "passed",
            "test_outputs.py::test_cancel_command": "passed",
            "test_outputs.py::test_startup_files": "passed",
            "test_outputs.py::test_shell_state_persists_between_commands": "passed",
            "test_outputs.py::test_background_commands": "failed",
        },
        "failed_tests": ("test_outputs.py::test_background_commands",),
        "localhost_port": 8000,
    },
    "terminal-bench/hf-model-inference": {
        "task_ref": (
            "sha256:01370f37e61286f920dfca1e471496640ebcc8d06e1cc48d874c790715dbd4ad"
        ),
        "classifier": "verifier_localhost_5000_name_resolution_v1",
        "tests": {
            "test_outputs.py::test_model_downloaded": "passed",
            "test_outputs.py::test_flask_api_running": "passed",
            "test_outputs.py::test_sentiment_endpoint": "failed",
            "test_outputs.py::test_api_error_handling": "failed",
        },
        "failed_tests": (
            "test_outputs.py::test_sentiment_endpoint",
            "test_outputs.py::test_api_error_handling",
        ),
        "localhost_port": 5000,
    },
    "terminal-bench/qemu-startup": {
        "task_ref": (
            "sha256:8e58263747da7dc688ad470688fb72825c4ba2c1c40443fa0f52963645bbd999"
        ),
        "classifier": "qemu_background_service_lost_before_verifier_v1",
        "tests": {"test_outputs.py::test_version": "failed"},
        "failed_tests": ("test_outputs.py::test_version",),
        "codex_output_sha256": (
            "ff8c4473518cfb3a9f94fa5671187d235caeec1235c152c35e5e54441e222c6e"
        ),
    },
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _future_time(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=max(0.0, seconds))).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _safe_job_name(value: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME_RE.fullmatch(value) is None:
        raise PoolError("job name must match [A-Za-z0-9][A-Za-z0-9._-]{0,126}")
    return value


def _safe_worker_file(value: str) -> str:
    rendered = _SAFE_WORKER_RE.sub("-", value.strip()).strip(".-") or "worker"
    suffix = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{rendered[:80]}-{suffix}.json"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=PRIVATE_DIR_MODE)
    if path.is_symlink() or not path.is_dir():
        raise PoolError(f"unsafe artifact directory: {path}")
    try:
        path.chmod(PRIVATE_DIR_MODE)
    except OSError:
        pass
    return path


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # The file itself was already fsynced. Some network filesystems do
            # not support directory fsync; do not report a false post-commit
            # failure after os.replace has made the JSON authoritative.
            pass
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    _ensure_dir(path.parent)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace one private regular file with the exact byte payload."""

    path = Path(path)
    _ensure_dir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise PoolError(f"artifact must not be a symlink: {path}")
    try:
        info = path.stat()
    except OSError as exc:
        raise PoolError(f"could not stat artifact: {path}") from exc
    if not path.is_file() or info.st_size > max_bytes:
        raise PoolError(f"invalid or oversized artifact: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoolError(f"could not parse artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise PoolError(f"artifact is not a JSON object: {path}")
    return dict(value)


def _open_lock(path: Path, *, blocking: bool) -> int | None:
    _ensure_dir(path.parent)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    os.fchmod(descriptor, PRIVATE_FILE_MODE)
    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(descriptor, operation)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _close_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _append_event(root: Path, record: Mapping[str, Any]) -> None:
    lock = _open_lock(root / ".events.lock", blocking=True)
    assert lock is not None
    try:
        path = root / "events.jsonl"
        flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
        descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
        try:
            payload = _canonical_bytes(dict(record)) + b"\n"
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        _close_lock(lock)


def _best_effort_event(root: Path, record: Mapping[str, Any]) -> None:
    try:
        _append_event(root, record)
    except (OSError, PoolError):
        # Compact state/result commits remain authoritative; an audit journal
        # outage must not change an official attempt or its retry accounting.
        pass


@dataclass(frozen=True, slots=True)
class PoolPaths:
    root: Path

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def marker(self) -> Path:
        return self.root / "pool.json"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def cells_dir(self) -> Path:
        return self.root / "cells"

    @property
    def workers_dir(self) -> Path:
        return self.root / "workers"

    @property
    def init_lock(self) -> Path:
        return self.root / ".pool-init.lock"

    @property
    def aggregate_lock(self) -> Path:
        return self.root / ".pool-aggregate.lock"

    @property
    def attempt_selection(self) -> Path:
        return self.root / "attempt-selection.json"

    @property
    def attempt_retirement_lock(self) -> Path:
        return self.root / ".attempt-retirement.lock"

    @property
    def task_selection(self) -> Path:
        return self.root / "task-selection.json"

    @property
    def task_retirement_lock(self) -> Path:
        return self.root / ".task-retirement.lock"

    def ensure(self) -> "PoolPaths":
        _ensure_dir(self.root)
        _ensure_dir(self.cells_dir)
        _ensure_dir(self.workers_dir)
        return self


@dataclass(frozen=True, slots=True)
class CellSpec:
    cell_id: str
    task_name: str
    task_ref: str
    source: str
    official_attempt_index: int
    task_config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CellPaths:
    root: Path

    @property
    def claim_lock(self) -> Path:
        return self.root / ".claim.lock"

    @property
    def dispatch(self) -> Path:
        return self.root / "dispatch.json"

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def external_reservation(self) -> Path:
        return self.root / "external-reservation.json"

    @property
    def exception(self) -> Path:
        return self.root / "exception.json"

    @property
    def infrastructure(self) -> Path:
        return self.root / "infrastructure.json"

    @property
    def task_cache_transport(self) -> Path:
        return self.root / "task-cache-transport.json"

    @property
    def harbor_resume_config_repair(self) -> Path:
        return self.root / "harbor-resume-config-repair.json"

    @property
    def job_config(self) -> Path:
        return self.root / "job-config.json"

    @property
    def harbor_job(self) -> Path:
        return self.root / "harbor-job"

    @property
    def attempts_dir(self) -> Path:
        return self.root / "attempts"

    def ensure(self) -> "CellPaths":
        _ensure_dir(self.root)
        _ensure_dir(self.attempts_dir)
        return self


@dataclass(frozen=True, slots=True)
class PoolSpec:
    paths: PoolPaths
    config: Mapping[str, Any]
    marker: Mapping[str, Any]
    cells: tuple[CellSpec, ...]
    contract_hash: str

    def cell_paths(self, cell: CellSpec) -> CellPaths:
        return CellPaths(self.paths.cells_dir / cell.cell_id)


@dataclass(frozen=True, slots=True)
class ClaimedCell:
    cell: CellSpec
    paths: CellPaths
    descriptor: int
    execution_attempt: int
    dispatch: Mapping[str, Any]


def _cell_payload(cell: CellSpec) -> dict[str, Any]:
    return {
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "source": cell.source,
        "official_attempt_index": cell.official_attempt_index,
        "task_config": dict(cell.task_config),
    }


def _cell_from_payload(value: Mapping[str, Any]) -> CellSpec:
    cell_id = value.get("cell_id")
    task_name = value.get("task_name")
    task_ref = value.get("task_ref")
    source = value.get("source")
    official_attempt = value.get("official_attempt_index")
    task_config = value.get("task_config")
    if (
        not isinstance(cell_id, str)
        or re.fullmatch(r"tb_[0-9a-f]{32}", cell_id) is None
        or not isinstance(task_name, str)
        or not task_name
        or not isinstance(task_ref, str)
        or not task_ref.startswith("sha256:")
        or not isinstance(source, str)
        or not source
        or not isinstance(official_attempt, int)
        or isinstance(official_attempt, bool)
        or official_attempt < 1
        or not isinstance(task_config, Mapping)
    ):
        raise PoolError("pool marker contains an invalid cell")
    expected_id = (
        "tb_"
        + _digest(
            {
                "task_config": dict(task_config),
                "official_attempt_index": official_attempt,
            }
        )[:32]
    )
    if cell_id != expected_id:
        raise PoolError(f"cell id digest mismatch for {task_name}")
    return CellSpec(
        cell_id=cell_id,
        task_name=task_name,
        task_ref=task_ref,
        source=source,
        official_attempt_index=official_attempt,
        task_config=dict(task_config),
    )


def _claim_priority_task_names(
    marker: Mapping[str, Any], cells: Sequence[CellSpec]
) -> tuple[str, ...]:
    raw = marker.get("claim_priority")
    if raw is None:
        return ()
    if not isinstance(raw, Mapping) or set(raw) != {
        "contract_sha256",
        "policy",
        "schema",
        "source_job_name",
        "source_manifest_path",
        "source_manifest_sha256",
        "tasks",
    }:
        raise PoolError("pool claim-priority contract is malformed")
    body = {key: value for key, value in raw.items() if key != "contract_sha256"}
    if raw.get("contract_sha256") != _digest(body):
        raise PoolError("pool claim-priority contract digest mismatch")
    if raw.get("schema") != CLAIM_PRIORITY_SCHEMA:
        raise PoolError("unknown pool claim-priority schema")
    if raw.get("policy") != WORKER_CLAIM_PRIORITY_POLICY:
        raise PoolError("unknown pool claim-priority policy")
    manifest_path = raw.get("source_manifest_path")
    manifest_sha256 = raw.get("source_manifest_sha256")
    source_job_name = raw.get("source_job_name")
    if (
        not isinstance(manifest_path, str)
        or not Path(manifest_path).is_absolute()
        or any(character in manifest_path for character in ("\n", "\r", "\x00"))
        or not isinstance(manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
        or not isinstance(source_job_name, str)
        or _safe_job_name(source_job_name) != source_job_name
    ):
        raise PoolError("pool claim-priority source identity is invalid")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise PoolError("pool claim-priority task list is empty")
    known_refs: dict[str, str] = {}
    for cell in cells:
        existing = known_refs.setdefault(cell.task_name, cell.task_ref)
        if existing != cell.task_ref:
            raise PoolError("pool task name maps to multiple task refs")
    names: list[str] = []
    for task in tasks:
        if not isinstance(task, Mapping) or set(task) != {"task_name", "task_ref"}:
            raise PoolError("pool claim-priority task record is malformed")
        task_name = task.get("task_name")
        task_ref = task.get("task_ref")
        if (
            not isinstance(task_name, str)
            or not isinstance(task_ref, str)
            or known_refs.get(task_name) != task_ref
        ):
            raise PoolError("pool claim-priority task identity is invalid")
        names.append(task_name)
    if len(set(names)) != len(names):
        raise PoolError("pool claim-priority task names are duplicated")
    return tuple(names)


def _load_yaml_config(path: Path) -> tuple[JobConfig, bytes]:
    path = Path(path).expanduser().resolve()
    try:
        raw_bytes = path.read_bytes()
        raw = yaml.safe_load(raw_bytes)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise PoolError(f"could not read Harbor config: {path}") from exc
    if not isinstance(raw, Mapping):
        raise PoolError("Harbor config must be a YAML/JSON object")
    try:
        config = JobConfig.model_validate(raw)
    except Exception as exc:
        raise PoolError("Harbor config does not satisfy JobConfig") from exc
    return config, raw_bytes


def _canonical_pool_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only Harbor's set-valued retry fields, not ordered lists.

    JobConfig serializes these sets as JSON arrays whose order varies between
    processes. Existing pool files/hashes remain immutable; this representation
    is used for new contracts and semantic comparisons during reinitialization.
    """
    payload = copy.deepcopy(dict(config))
    retry = payload.get("retry")
    if isinstance(retry, dict):
        for field in ("include_exceptions", "exclude_exceptions"):
            values = retry.get(field)
            if isinstance(values, list) and all(isinstance(value, str) for value in values):
                retry[field] = sorted(values)
    return payload


async def _resolve_tasks(config: JobConfig) -> list[TaskConfig]:
    return await JobPlan.resolve_task_configs(config)


def _select_tasks(
    tasks: Sequence[TaskConfig],
    *,
    all_tasks: bool,
    task_names: Sequence[str],
    num_tasks: int | None,
) -> list[TaskConfig]:
    ordered = sorted(tasks, key=lambda task: str(task.name))
    by_name: dict[str, TaskConfig] = {}
    for task in ordered:
        name = str(task.name or "")
        if not name or name in by_name:
            raise PoolError(f"resolved task name is missing or duplicated: {name!r}")
        by_name[name] = task
    if all_tasks:
        selected = ordered
    elif task_names:
        missing = [name for name in task_names if name not in by_name]
        if missing:
            raise PoolError("unknown task name(s): " + ", ".join(missing))
        selected = [by_name[name] for name in task_names]
    elif num_tasks is not None:
        if num_tasks < 1:
            raise PoolError("--num-tasks must be positive")
        selected = ordered[:num_tasks]
    else:
        raise PoolError("choose exactly one of --all, --task-name, or --num-tasks")
    if not selected:
        raise PoolError("pool selection is empty")
    return selected


def _task_contract(task: TaskConfig) -> dict[str, Any]:
    payload = task.model_dump(mode="json", exclude_none=True)
    name = payload.get("name")
    ref = payload.get("ref")
    source = payload.get("source")
    if (
        not isinstance(name, str)
        or not name
        or not isinstance(ref, str)
        or not ref.startswith("sha256:")
        or not isinstance(source, str)
        or not source
    ):
        raise PoolError(f"task is not an exact pinned package task: {name!r}")
    return payload


def _build_claim_priority_contract(
    selected_tasks: Sequence[TaskConfig],
    *,
    priority_task_names: Sequence[str],
    source_manifest: Path | None,
    source_manifest_sha256: str | None,
    source_job_name: str | None,
) -> dict[str, Any] | None:
    names = tuple(priority_task_names)
    sources = (source_manifest, source_manifest_sha256, source_job_name)
    if not names:
        if any(value is not None for value in sources):
            raise PoolError(
                "claim-priority source metadata requires --priority-task-name"
            )
        return None
    if len(set(names)) != len(names) or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise PoolError("claim-priority task names must be unique nonempty strings")
    if any(value is None for value in sources):
        raise PoolError(
            "claim priority requires source manifest, SHA-256, and source job"
        )
    assert source_manifest is not None
    assert source_manifest_sha256 is not None
    assert source_job_name is not None
    manifest_path = Path(source_manifest).expanduser().resolve()
    if not manifest_path.is_absolute() or any(
        character in str(manifest_path) for character in ("\n", "\r", "\x00")
    ):
        raise PoolError("claim-priority source manifest must be a safe absolute path")
    if re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None:
        raise PoolError("claim-priority source manifest SHA-256 is invalid")
    source_job = _safe_job_name(source_job_name)
    by_name = {str(task.name): task for task in selected_tasks}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise PoolError(
            "claim-priority task names are not in the selected pool: "
            + ", ".join(unknown)
        )
    tasks = [
        {
            "task_name": name,
            "task_ref": str(_task_contract(by_name[name])["ref"]),
        }
        for name in names
    ]
    body = {
        "schema": CLAIM_PRIORITY_SCHEMA,
        "policy": WORKER_CLAIM_PRIORITY_POLICY,
        "tasks": tasks,
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "source_job_name": source_job,
    }
    return {**body, "contract_sha256": _digest(body)}


def initialize_pool(
    *,
    job_name: str,
    config_path: Path,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    all_tasks: bool = False,
    task_names: Sequence[str] = (),
    num_tasks: int | None = None,
    attempts: int | None = None,
    priority_task_names: Sequence[str] = (),
    priority_source_manifest: Path | None = None,
    priority_source_manifest_sha256: str | None = None,
    priority_source_job_name: str | None = None,
    task_resolver: Callable[[JobConfig], Sequence[TaskConfig]] | None = None,
) -> Path:
    """Freeze a new matrix, or validate/resume an existing frozen one offline.

    For an existing pool, --all means all tasks in its immutable snapshot, not
    a refresh of the remote registry. An explicit resolver still requests full
    resolution/comparison (used by callers deliberately validating a new list).
    """

    name = _safe_job_name(job_name)
    root = Path(jobs_root).expanduser().resolve()
    paths = PoolPaths(root / name).ensure()
    lock = _open_lock(paths.init_lock, blocking=True)
    assert lock is not None
    try:
        config, source_bytes = _load_yaml_config(config_path)
        if len(config.agents) != 1:
            raise PoolError(
                "distributed Terminal-Bench pool requires exactly one agent"
            )
        selected_attempts = config.n_attempts if attempts is None else attempts
        if selected_attempts not in {1, 5}:
            raise PoolError("official attempts must be 1 (diagnostic) or 5 (official)")
        # Normalize before any remote lookup. Joining a live, immutable pool
        # must not depend on Harbor's registry being reachable from this host.
        pool_config = config.model_copy(
            update={
                "job_name": name,
                "jobs_dir": root,
                "n_attempts": selected_attempts,
                "n_concurrent_trials": 1,
            }
        )
        config_payload = _canonical_pool_config(
            pool_config.model_dump(mode="json", exclude_none=True)
        )
        existing = None
        if paths.marker.exists() or paths.config.exists():
            if not paths.marker.exists() or not paths.config.exists():
                raise PoolError(
                    "pool initialization is incomplete; refuse to overwrite"
                )
            existing = load_pool_spec(job_name=name, jobs_root=root)
            if _canonical_pool_config(existing.config) != config_payload:
                raise PoolError(
                    "job name already belongs to a different immutable pool"
                )
        if existing is not None and task_resolver is None:
            resolved_tasks = [
                TaskConfig.model_validate(cell.task_config)
                for cell in existing.cells
                if cell.official_attempt_index == 1
            ]
            if num_tasks is not None and num_tasks > len(resolved_tasks):
                raise PoolError("cannot expand an immutable pool with --num-tasks")
        elif task_resolver is None:
            resolved_tasks = asyncio.run(_resolve_tasks(config))
        else:
            resolved_tasks = list(task_resolver(config))
        selected_tasks = _select_tasks(
            resolved_tasks,
            all_tasks=all_tasks,
            task_names=task_names,
            num_tasks=num_tasks,
        )
        claim_priority = _build_claim_priority_contract(
            selected_tasks,
            priority_task_names=priority_task_names,
            source_manifest=priority_source_manifest,
            source_manifest_sha256=priority_source_manifest_sha256,
            source_job_name=priority_source_job_name,
        )
        contract_hash = _digest(config_payload)
        cells: list[CellSpec] = []
        for task in selected_tasks:
            task_payload = _task_contract(task)
            for official_attempt in range(1, selected_attempts + 1):
                cell_id = (
                    "tb_"
                    + _digest(
                        {
                            "task_config": task_payload,
                            "official_attempt_index": official_attempt,
                        }
                    )[:32]
                )
                cells.append(
                    CellSpec(
                        cell_id=cell_id,
                        task_name=str(task_payload["name"]),
                        task_ref=str(task_payload["ref"]),
                        source=str(task_payload["source"]),
                        official_attempt_index=official_attempt,
                        task_config=task_payload,
                    )
                )
        cell_payloads = [_cell_payload(cell) for cell in cells]
        selection_hash = _digest(cell_payloads)
        marker = {
            "schema": POOL_SCHEMA,
            "job_name": name,
            "mode": "shared-filesystem-flock",
            "ownership_authority": "per-cell-flock",
            "lease_is_ownership_authority": False,
            "config_contract_hash": contract_hash,
            "source_config_path": str(Path(config_path).expanduser().resolve()),
            "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "selection_sha256": selection_hash,
            "task_count": len(selected_tasks),
            "cell_count": len(cells),
            "n_attempts": selected_attempts,
            "cells": cell_payloads,
            "initialized_at": _utc_now(),
        }
        if claim_priority is not None:
            marker["claim_priority"] = claim_priority
        if existing is not None:
            if (
                _canonical_pool_config(existing.config) != config_payload
                or existing.marker.get("selection_sha256") != selection_hash
                or existing.marker.get("claim_priority") != claim_priority
            ):
                raise PoolError(
                    "job name already belongs to a different immutable pool"
                )
            if task_resolver is None:
                print(
                    f"[terminal-bench-pool] verified existing pool offline: "
                    f"{len(selected_tasks)} tasks, {selected_attempts} attempt(s)/task; "
                    "frozen task refs and results unchanged",
                    file=sys.stderr,
                )
            return paths.root
        _atomic_write_json(paths.config, config_payload)
        for cell in cells:
            cell_paths = CellPaths(paths.cells_dir / cell.cell_id).ensure()
            _atomic_write_json(
                cell_paths.dispatch,
                {
                    "schema": CELL_DISPATCH_SCHEMA,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "task_ref": cell.task_ref,
                    "official_attempt_index": cell.official_attempt_index,
                    "config_contract_hash": contract_hash,
                    "status": "queued",
                    "execution_attempt": 0,
                    "enqueued_at": marker["initialized_at"],
                    "updated_at": marker["initialized_at"],
                },
            )
        _atomic_write_json(paths.marker, marker)
        _append_event(
            paths.root,
            {
                "event": "pool_initialized",
                "cell_count": len(cells),
                "task_count": len(selected_tasks),
                "recorded_at": _utc_now(),
            },
        )
    finally:
        _close_lock(lock)
    spec = load_pool_spec(job_name=name, jobs_root=root)
    aggregate_pool(spec)
    return paths.root


def load_pool_spec(*, job_name: str, jobs_root: Path = DEFAULT_JOBS_ROOT) -> PoolSpec:
    name = _safe_job_name(job_name)
    paths = PoolPaths(Path(jobs_root).expanduser().resolve() / name)
    if not paths.marker.is_file() or not paths.config.is_file():
        raise PoolError(f"distributed pool is not initialized: {paths.root}")
    marker = _read_json(paths.marker)
    config = _read_json(paths.config)
    if marker.get("schema") != POOL_SCHEMA or marker.get("job_name") != name:
        raise PoolError("unknown pool schema or job identity")
    contract_hash = _digest(config)
    if marker.get("config_contract_hash") != contract_hash:
        raise PoolError("pool config contract hash mismatch")
    raw_cells = marker.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise PoolError("pool marker has no cells")
    cells = tuple(
        _cell_from_payload(value) for value in raw_cells if isinstance(value, Mapping)
    )
    if len(cells) != len(raw_cells) or len({cell.cell_id for cell in cells}) != len(
        cells
    ):
        raise PoolError("pool marker contains duplicate or malformed cells")
    if marker.get("cell_count") != len(cells):
        raise PoolError("pool marker cell count mismatch")
    if marker.get("selection_sha256") != _digest(
        [_cell_payload(cell) for cell in cells]
    ):
        raise PoolError("pool selection digest mismatch")
    _claim_priority_task_names(marker, cells)
    configured_attempts = config.get("n_attempts")
    if (
        marker.get("n_attempts") != configured_attempts
        or not isinstance(configured_attempts, int)
        or isinstance(configured_attempts, bool)
        or configured_attempts < 1
    ):
        raise PoolError("pool attempt contract mismatch")
    by_task: dict[str, set[int]] = {}
    for cell in cells:
        by_task.setdefault(cell.task_name, set()).add(cell.official_attempt_index)
    expected_attempts = set(range(1, configured_attempts + 1))
    if marker.get("task_count") != len(by_task) or any(
        attempts != expected_attempts for attempts in by_task.values()
    ):
        raise PoolError("pool task/official-attempt matrix is incomplete")
    paths.ensure()
    return PoolSpec(
        paths=paths,
        config=config,
        marker=marker,
        cells=cells,
        contract_hash=contract_hash,
    )


def _attempt_selection(spec: PoolSpec) -> dict[str, Any] | None:
    """Return the optional additive scoring selection without changing the pool."""

    path = spec.paths.attempt_selection
    if not path.is_file():
        return None
    value = _read_json(path, max_bytes=1024 * 1024)
    keep = value.get("keep_official_attempt_index")
    excluded = value.get("excluded_official_attempt_indices")
    configured = spec.marker.get("n_attempts")
    if (
        value.get("schema") != ATTEMPT_SELECTION_SCHEMA
        or value.get("job_name") != spec.paths.root.name
        or value.get("config_contract_hash") != spec.contract_hash
        or value.get("selection_sha256") != spec.marker.get("selection_sha256")
        or value.get("policy") != QUEUED_ATTEMPT_RETIREMENT_POLICY
        or keep != 1
        or not isinstance(configured, int)
        or isinstance(configured, bool)
        or excluded != list(range(2, configured + 1))
        or _parse_time(value.get("activated_at")) is None
    ):
        raise PoolError("invalid pool attempt-selection artifact")
    return value


def _task_selection(spec: PoolSpec) -> dict[str, Any] | None:
    """Return the optional immutable task-level scoring selection."""

    path = spec.paths.task_selection
    if not path.is_file():
        return None
    value = _read_json(path, max_bytes=1024 * 1024)
    excluded = value.get("excluded_task_names")
    known = {cell.task_name for cell in spec.cells}
    if (
        value.get("schema") != TASK_SELECTION_SCHEMA
        or value.get("job_name") != spec.paths.root.name
        or value.get("config_contract_hash") != spec.contract_hash
        or value.get("selection_sha256") != spec.marker.get("selection_sha256")
        or value.get("policy") != QUEUED_TASK_RETIREMENT_POLICY
        or not isinstance(excluded, list)
        or not excluded
        or any(not isinstance(name, str) or not name for name in excluded)
        or excluded != sorted(set(excluded))
        or not set(excluded) < known
        or not isinstance(value.get("reason"), str)
        or not value["reason"].strip()
        or value["reason"] != value["reason"].strip()
        or "\n" in value["reason"]
        or "\r" in value["reason"]
        or len(value["reason"]) > 512
        or value.get("accuracy_denominator") != "score_eligible_cells_only"
        or value.get("immutable_pool_marker_changed") is not False
        or value.get("immutable_pool_config_changed") is not False
        or _parse_time(value.get("activated_at")) is None
    ):
        raise PoolError("invalid pool task-selection artifact")
    return value


def _cell_is_score_eligible(
    cell: CellSpec,
    attempt_selection: Mapping[str, Any] | None,
    task_selection: Mapping[str, Any] | None = None,
) -> bool:
    if (
        attempt_selection is not None
        and cell.official_attempt_index
        != attempt_selection["keep_official_attempt_index"]
    ):
        return False
    return task_selection is None or cell.task_name not in set(
        task_selection["excluded_task_names"]
    )


def _retirement_root(paths: CellPaths) -> Path:
    return paths.root / "retirement" / QUEUED_ATTEMPT_RETIREMENT_DIR_NAME


def _task_retirement_root(paths: CellPaths) -> Path:
    return paths.root / "retirement" / QUEUED_TASK_RETIREMENT_DIR_NAME


def _read_bounded_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    path = Path(path)
    if path.is_symlink():
        raise PoolError(f"artifact must not be a symlink: {path}")
    try:
        info = path.stat()
    except OSError as exc:
        raise PoolError(f"could not stat artifact: {path}") from exc
    if not path.is_file() or info.st_size > max_bytes:
        raise PoolError(f"invalid or oversized artifact: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PoolError(f"could not read artifact: {path}") from exc


def _validate_retired_result(
    spec: PoolSpec, cell: CellSpec, result: Mapping[str, Any]
) -> None:
    retirement = result.get("retirement")
    if (
        result.get("status") != "completed"
        or result.get("outcome") != "retired"
        or result.get("rewards") != {}
        or result.get("exception_info") is not None
        or not isinstance(retirement, Mapping)
        or retirement.get("schema") != QUEUED_ATTEMPT_RETIREMENT_SCHEMA
        or retirement.get("policy") != QUEUED_ATTEMPT_RETIREMENT_POLICY
        or retirement.get("keep_official_attempt_index") != 1
        or retirement.get("score_eligible") is not False
        or retirement.get("model_called_by_retirement") is not False
    ):
        raise PoolError(f"invalid retired terminal result for {cell.cell_id}")
    paths = spec.cell_paths(cell)
    expected_receipt = _retirement_root(paths) / "receipt.json"
    relative_receipt = retirement.get("receipt_path")
    if not isinstance(relative_receipt, str) or relative_receipt != str(
        expected_receipt.relative_to(spec.paths.root)
    ):
        raise PoolError(f"invalid retirement receipt path for {cell.cell_id}")
    receipt_bytes = _read_bounded_regular_bytes(
        expected_receipt, max_bytes=2 * 1024 * 1024
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != retirement.get("receipt_sha256"):
        raise PoolError(f"retirement receipt digest mismatch for {cell.cell_id}")
    try:
        receipt_value = json.loads(receipt_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PoolError(f"invalid retirement receipt for {cell.cell_id}") from exc
    if not isinstance(receipt_value, Mapping):
        raise PoolError(f"invalid retirement receipt for {cell.cell_id}")
    receipt = dict(receipt_value)
    if (
        receipt.get("schema") != QUEUED_ATTEMPT_RETIREMENT_SCHEMA
        or receipt.get("policy") != QUEUED_ATTEMPT_RETIREMENT_POLICY
        or receipt.get("cell_id") != cell.cell_id
        or receipt.get("task_name") != cell.task_name
        or receipt.get("task_ref") != cell.task_ref
        or receipt.get("official_attempt_index") != cell.official_attempt_index
        or receipt.get("config_contract_hash") != spec.contract_hash
        or receipt.get("keep_official_attempt_index") != 1
        or receipt.get("score_eligible") is not False
        or receipt.get("model_called_by_retirement") is not False
        or _parse_time(receipt.get("retired_at")) is None
    ):
        raise PoolError(f"retirement receipt identity mismatch for {cell.cell_id}")
    originals = receipt.get("original_artifacts")
    if not isinstance(originals, Mapping) or "dispatch.json" not in originals:
        raise PoolError(
            f"retirement receipt lacks dispatch snapshot for {cell.cell_id}"
        )
    for name, record in originals.items():
        if name not in {"dispatch.json", "state.json"} or not isinstance(
            record, Mapping
        ):
            raise PoolError(f"invalid retirement snapshot index for {cell.cell_id}")
        relative = record.get("snapshot_path")
        digest = record.get("sha256")
        source_relative = record.get("source_path")
        size_bytes = record.get("size_bytes")
        expected_snapshot = _retirement_root(paths) / f"original-{name}"
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative != str(expected_snapshot.relative_to(spec.paths.root))
            or source_relative != str((paths.root / name).relative_to(spec.paths.root))
            or not isinstance(digest, str)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise PoolError(f"invalid retirement snapshot index for {cell.cell_id}")
        snapshot = spec.paths.root / relative
        try:
            snapshot.relative_to(_retirement_root(paths))
        except ValueError as exc:
            raise PoolError(f"unsafe retirement snapshot for {cell.cell_id}") from exc
        payload = _read_bounded_regular_bytes(snapshot, max_bytes=MAX_ARTIFACT_BYTES)
        if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != digest:
            raise PoolError(f"retirement snapshot digest mismatch for {cell.cell_id}")


def _validate_task_retired_result(
    spec: PoolSpec, cell: CellSpec, result: Mapping[str, Any]
) -> None:
    """Validate a task-selection retirement and its immutable evidence chain."""

    selection = _task_selection(spec)
    retirement = result.get("retirement")
    if (
        selection is None
        or cell.task_name not in selection["excluded_task_names"]
        or result.get("status") != "completed"
        or result.get("outcome") != "retired"
        or result.get("rewards") != {}
        or result.get("exception_info") is not None
        or not isinstance(retirement, Mapping)
        or retirement.get("schema") != QUEUED_TASK_RETIREMENT_SCHEMA
        or retirement.get("policy") != QUEUED_TASK_RETIREMENT_POLICY
        or retirement.get("score_eligible") is not False
        or retirement.get("model_called_by_retirement") is not False
        or retirement.get("reason") != selection["reason"]
        or retirement.get("task_selection_path")
        != str(spec.paths.task_selection.relative_to(spec.paths.root))
    ):
        raise PoolError(f"invalid task-retired terminal result for {cell.cell_id}")
    paths = spec.cell_paths(cell)
    expected_receipt = _task_retirement_root(paths) / "receipt.json"
    relative_receipt = retirement.get("receipt_path")
    if not isinstance(relative_receipt, str) or relative_receipt != str(
        expected_receipt.relative_to(spec.paths.root)
    ):
        raise PoolError(f"invalid task-retirement receipt path for {cell.cell_id}")
    selection_bytes = _read_bounded_regular_bytes(
        spec.paths.task_selection, max_bytes=1024 * 1024
    )
    selection_digest = hashlib.sha256(selection_bytes).hexdigest()
    if retirement.get("task_selection_sha256") != selection_digest:
        raise PoolError(f"task-selection digest mismatch for {cell.cell_id}")
    receipt_bytes = _read_bounded_regular_bytes(
        expected_receipt, max_bytes=2 * 1024 * 1024
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != retirement.get("receipt_sha256"):
        raise PoolError(f"task-retirement receipt digest mismatch for {cell.cell_id}")
    try:
        receipt_value = json.loads(receipt_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PoolError(f"invalid task-retirement receipt for {cell.cell_id}") from exc
    if not isinstance(receipt_value, Mapping):
        raise PoolError(f"invalid task-retirement receipt for {cell.cell_id}")
    receipt = dict(receipt_value)
    if (
        receipt.get("schema") != QUEUED_TASK_RETIREMENT_SCHEMA
        or receipt.get("policy") != QUEUED_TASK_RETIREMENT_POLICY
        or receipt.get("cell_id") != cell.cell_id
        or receipt.get("task_name") != cell.task_name
        or receipt.get("task_ref") != cell.task_ref
        or receipt.get("official_attempt_index") != cell.official_attempt_index
        or receipt.get("config_contract_hash") != spec.contract_hash
        or receipt.get("selection_sha256") != spec.marker.get("selection_sha256")
        or receipt.get("task_selection_path")
        != str(spec.paths.task_selection.relative_to(spec.paths.root))
        or receipt.get("task_selection_sha256") != selection_digest
        or receipt.get("reason") != selection["reason"]
        or receipt.get("score_eligible") is not False
        or receipt.get("model_called_by_retirement") is not False
        or _parse_time(receipt.get("retired_at")) is None
    ):
        raise PoolError(f"task-retirement receipt identity mismatch for {cell.cell_id}")
    originals = receipt.get("original_artifacts")
    if not isinstance(originals, Mapping) or "dispatch.json" not in originals:
        raise PoolError(
            f"task-retirement receipt lacks dispatch snapshot for {cell.cell_id}"
        )
    for name, record in originals.items():
        if name not in {"dispatch.json", "state.json"} or not isinstance(
            record, Mapping
        ):
            raise PoolError(
                f"invalid task-retirement snapshot index for {cell.cell_id}"
            )
        relative = record.get("snapshot_path")
        digest = record.get("sha256")
        source_relative = record.get("source_path")
        size_bytes = record.get("size_bytes")
        expected_snapshot = _task_retirement_root(paths) / f"original-{name}"
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative != str(expected_snapshot.relative_to(spec.paths.root))
            or source_relative != str((paths.root / name).relative_to(spec.paths.root))
            or not isinstance(digest, str)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise PoolError(
                f"invalid task-retirement snapshot index for {cell.cell_id}"
            )
        snapshot = spec.paths.root / relative
        try:
            snapshot.relative_to(_task_retirement_root(paths))
        except ValueError as exc:
            raise PoolError(
                f"unsafe task-retirement snapshot for {cell.cell_id}"
            ) from exc
        payload = _read_bounded_regular_bytes(snapshot, max_bytes=MAX_ARTIFACT_BYTES)
        if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != digest:
            raise PoolError(
                f"task-retirement snapshot digest mismatch for {cell.cell_id}"
            )


def _cell_result(spec: PoolSpec, cell: CellSpec) -> dict[str, Any] | None:
    path = spec.cell_paths(cell).result
    if not path.is_file():
        return None
    value = _read_json(path)
    if (
        value.get("schema") != CELL_RESULT_SCHEMA
        or value.get("cell_id") != cell.cell_id
        or value.get("task_name") != cell.task_name
        or value.get("task_ref") != cell.task_ref
        or value.get("official_attempt_index") != cell.official_attempt_index
        or value.get("config_contract_hash") != spec.contract_hash
        or value.get("status") not in {"completed", "pool_error"}
    ):
        raise PoolError(f"invalid terminal result for {cell.cell_id}")
    if value.get("outcome") == "retired":
        retirement = value.get("retirement")
        schema = retirement.get("schema") if isinstance(retirement, Mapping) else None
        if schema == QUEUED_ATTEMPT_RETIREMENT_SCHEMA:
            _validate_retired_result(spec, cell, value)
        elif schema == QUEUED_TASK_RETIREMENT_SCHEMA:
            _validate_task_retired_result(spec, cell, value)
        else:
            raise PoolError(f"invalid retired terminal result for {cell.cell_id}")
    return value


def _dispatch_status(paths: CellPaths) -> tuple[str, dict[str, Any]]:
    if not paths.dispatch.is_file():
        return "queued", {}
    dispatch = _read_json(paths.dispatch)
    if dispatch.get("status") == "reserved_external":
        return "reserved_external", dispatch
    if dispatch.get("status") != "running":
        return "queued", dispatch
    claim = dispatch.get("claim")
    expires = claim.get("lease_expires_at") if isinstance(claim, Mapping) else None
    expiration = _parse_time(expires)
    if expiration is None or expiration <= datetime.now(UTC):
        return "stale", dispatch
    return "running", dispatch


def _scoreable_cell_reward(
    spec: PoolSpec, cell: CellSpec, result: Mapping[str, Any]
) -> float | None:
    """Apply the attribution gate again when rebuilding aggregate metrics.

    This second boundary protects a mixed-version pool: a terminal projection
    written by an older worker cannot become model zero merely because it
    contains a verifier reward alongside an agent/environment exception.
    """

    if isinstance(result.get("agent_timeout_policy"), Mapping):
        from terminal_bench_eval.agent_timeout_policy import validate_reward
        return validate_reward(spec, cell, result)
    rewards = result.get("rewards")
    reward = rewards.get("reward") if isinstance(rewards, Mapping) else None
    if (
        not isinstance(reward, (int, float))
        or isinstance(reward, bool)
        or not math.isfinite(float(reward))
        or result.get("status") != "completed"
        or result.get("outcome") != ("passed" if float(reward) > 0 else "failed")
    ):
        return None
    if isinstance(result.get("verification_recovery"), Mapping):
        from terminal_bench_eval.verifier_queue import validate_recovery_reward
        try:
            return validate_recovery_reward(spec, cell, result)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return None
    if isinstance(result.get("verifier_replay"), Mapping):
        return _scoreable_verifier_replay_reward(spec, cell, result)
    relative_trial = result.get("trial_path")
    if not isinstance(relative_trial, str) or not relative_trial:
        return None
    relative_path = Path(relative_trial)
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.parts in {(), (".",)}
    ):
        return None
    trial_path = spec.paths.root / relative_path
    expected_job = spec.cell_paths(cell).harbor_job
    if (
        trial_path.parent != expected_job
        or trial_path.is_symlink()
        or not trial_path.is_dir()
        or expected_job.is_symlink()
        or not expected_job.is_dir()
    ):
        return None
    result_path = trial_path / "result.json"
    if result_path.is_symlink() or not result_path.is_file():
        return None
    try:
        synthetic = ClaimedCell(
            cell=cell,
            paths=spec.cell_paths(cell),
            descriptor=-1,
            execution_attempt=_valid_execution(result.get("execution_attempt")),
            dispatch={},
        )
        terminal_trials = _harbor_trial_results(synthetic)
        if len(terminal_trials) != 1 or terminal_trials[0][0] != trial_path:
            return None
        trial = TrialResult.model_validate(terminal_trials[0][1]).model_dump(
            mode="json", exclude_none=True
        )
        _validate_harbor_trial(spec, synthetic, trial)
    except Exception:
        return None
    trial_rewards = _result_rewards(trial)
    if (
        trial.get("trial_name") != trial_path.name
        or result.get("trial_name") != trial_path.name
        or trial.get("trial_name") != result.get("trial_name")
        or trial_rewards != dict(rewards)
        or float(trial_rewards.get("reward", math.nan)) != float(reward)
    ):
        return None
    trial_exception = trial.get("exception_info")
    projected_exception = result.get("exception_info")
    if _digest(trial_exception) != _digest(projected_exception):
        return None
    has_exception = isinstance(trial_exception, Mapping) and bool(trial_exception)
    has_host_failure = any(
        classifier(trial_path, trial) is not None
        for classifier in (_runtime_receipt_failure, _package_bridge_receipt_failure)
    )
    if not has_exception and not has_host_failure:
        return float(reward)
    verified_agent_timeout = _verified_reward_after_agent_timeout(trial_path, trial)
    if verified_agent_timeout is not None:
        if _digest(verified_agent_timeout) != _digest(
            result.get("verified_agent_timeout_warning")
        ):
            return None
        return float(reward)
    verified_agent_exit = _verified_success_after_agent_sigterm(trial_path, trial)
    if verified_agent_exit is not None:
        if _digest(verified_agent_exit) != _digest(
            result.get("verified_agent_exit_warning")
        ):
            return None
        return float(reward)
    warning = _local_cleanup_warning(trial_path, trial)
    if (
        warning is None
        or not _exception_is_local_cleanup_only(trial)
        or _digest(warning) != _digest(result.get("infrastructure_cleanup_warning"))
    ):
        return None
    return float(reward)


def _active_worker_count(paths: PoolPaths) -> int:
    if not paths.workers_dir.is_dir():
        return 0
    now = datetime.now(UTC)
    count = 0
    for path in paths.workers_dir.glob("*.json"):
        try:
            worker = _read_json(path, max_bytes=2 * 1024 * 1024)
        except PoolError:
            continue
        expiration = _parse_time(worker.get("lease_expires_at"))
        if (
            expiration is not None
            and expiration > now
            and worker.get("status") in {"starting", "running", "idle", "network_unavailable"}
        ):
            count += 1
    return count


def aggregate_pool(spec: PoolSpec) -> dict[str, Any]:
    """Rebuild compact state; a cell result is the sole terminal authority."""

    lock = _open_lock(spec.paths.aggregate_lock, blocking=True)
    assert lock is not None
    try:
        attempt_selection = _attempt_selection(spec)
        task_selection = _task_selection(spec)
        eligible_cell_count = sum(
            _cell_is_score_eligible(cell, attempt_selection, task_selection)
            for cell in spec.cells
        )
        terminal_count = 0
        eligible_terminal_count = 0
        excluded_terminal_count = 0
        retired_count = 0
        completed_count = 0
        pool_error_count = 0
        eligible_pool_error_count = 0
        dispatch_counts: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        eligible_outcomes: Counter[str] = Counter()
        artifact_errors = 0
        eligible_artifact_errors = 0
        observed_rewards: list[float] = []
        attempt_one_rewards: list[float] = []
        attempt_one_terminal_count = 0
        attempt_one_artifact_errors = 0
        all_slot_reward = 0.0
        input_tokens = 0
        output_tokens = 0
        per_task_rewards: dict[str, list[float]] = {}
        for cell in spec.cells:
            eligible = _cell_is_score_eligible(cell, attempt_selection, task_selection)
            try:
                result = _cell_result(spec, cell)
            except PoolError:
                artifact_errors += 1
                if eligible:
                    eligible_artifact_errors += 1
                if cell.official_attempt_index == 1:
                    attempt_one_artifact_errors += 1
                continue
            if result is not None:
                if result.get("outcome") == "verification_pending":
                    dispatch_counts["verification_pending"] += 1
                    input_tokens += int(result.get("n_input_tokens") or 0)
                    output_tokens += int(result.get("n_output_tokens") or 0)
                    continue
                terminal_count += 1
                if eligible:
                    eligible_terminal_count += 1
                else:
                    excluded_terminal_count += 1
                if cell.official_attempt_index == 1:
                    attempt_one_terminal_count += 1
                status = str(result["status"])
                if status == "completed":
                    completed_count += 1
                else:
                    pool_error_count += 1
                    if eligible:
                        eligible_pool_error_count += 1
                outcome = str(result.get("outcome") or status)
                outcomes[outcome] += 1
                if outcome == "retired":
                    retired_count += 1
                if eligible:
                    eligible_outcomes[outcome] += 1
                reward = _scoreable_cell_reward(spec, cell, result)
                if cell.official_attempt_index == 1 and reward is not None:
                    attempt_one_rewards.append(reward)
                if eligible:
                    numeric = reward if reward is not None else 0.0
                    if reward is not None:
                        observed_rewards.append(reward)
                    all_slot_reward += numeric
                    per_task_rewards.setdefault(cell.task_name, []).append(numeric)
                input_tokens += int(result.get("n_input_tokens") or 0)
                output_tokens += int(result.get("n_output_tokens") or 0)
                continue
            try:
                dispatch_status, _ = _dispatch_status(spec.cell_paths(cell))
            except PoolError:
                artifact_errors += 1
                if eligible:
                    eligible_artifact_errors += 1
                if cell.official_attempt_index == 1:
                    attempt_one_artifact_errors += 1
                continue
            dispatch_counts[dispatch_status] += 1

        total = len(spec.cells)
        completed = terminal_count == total and artifact_errors == 0
        initialized_at = str(spec.marker.get("initialized_at") or _utc_now())
        started = _parse_time(initialized_at)
        elapsed = (
            max(0.0, (datetime.now(UTC) - started).total_seconds())
            if started is not None
            else 0.0
        )
        pass_any_count = sum(
            1
            for rewards in per_task_rewards.values()
            if any(reward > 0 for reward in rewards)
        )
        state: dict[str, Any] = {
            "schema": POOL_STATE_SCHEMA,
            "mode": "shared-filesystem-distributed",
            "job_name": spec.paths.root.name,
            "config_contract_hash": spec.contract_hash,
            "status": (
                "completed_with_pool_errors"
                if completed and eligible_pool_error_count
                else "completed" if completed else "running"
            ),
            "cell_count": total,
            "terminal_count": terminal_count,
            "completed_count": completed_count,
            "verification_pending_count": dispatch_counts["verification_pending"],
            "pool_error_count": pool_error_count,
            "score_eligible_cell_count": eligible_cell_count,
            "score_eligible_terminal_count": eligible_terminal_count,
            "score_eligible_pool_error_count": eligible_pool_error_count,
            "score_eligible_artifact_error_count": eligible_artifact_errors,
            "excluded_cell_count": total - eligible_cell_count,
            "excluded_terminal_count": excluded_terminal_count,
            "retired_count": retired_count,
            "keep_official_attempt_index": (
                attempt_selection.get("keep_official_attempt_index")
                if attempt_selection is not None
                else None
            ),
            "excluded_task_names": (
                list(task_selection["excluded_task_names"])
                if task_selection is not None
                else []
            ),
            "task_selection_reason": (
                task_selection.get("reason") if task_selection is not None else None
            ),
            "score_eligible_task_count": len(
                {
                    cell.task_name
                    for cell in spec.cells
                    if _cell_is_score_eligible(cell, attempt_selection, task_selection)
                }
            ),
            "attempt_1_cell_count": sum(
                cell.official_attempt_index == 1 for cell in spec.cells
            ),
            "attempt_1_terminal_count": attempt_one_terminal_count,
            "attempt_1_artifact_error_count": attempt_one_artifact_errors,
            "attempt_1_observed_reward_count": len(attempt_one_rewards),
            "attempt_1_mean_reward": (
                sum(attempt_one_rewards) / len(attempt_one_rewards)
                if attempt_one_rewards
                else None
            ),
            "queued_count": dispatch_counts.get("queued", 0),
            "reserved_external_count": dispatch_counts.get("reserved_external", 0),
            "running_count": dispatch_counts.get("running", 0),
            "stale_count": dispatch_counts.get("stale", 0),
            "artifact_error_count": artifact_errors,
            "active_worker_count": _active_worker_count(spec.paths),
            "observed_reward_count": len(observed_rewards),
            "observed_mean_reward": (
                sum(observed_rewards) / len(observed_rewards)
                if observed_rewards
                else None
            ),
            # Do not silently coerce an infrastructure/pool error with no
            # verifier reward into model reward zero.  The official all-slot
            # metric exists only when every immutable slot is scoreable.
            "all_slot_mean_reward": (
                all_slot_reward / eligible_cell_count
                if eligible_cell_count
                and len(observed_rewards) == eligible_cell_count
                and eligible_artifact_errors == 0
                else None
            ),
            "tasks_with_any_success": pass_any_count,
            "n_input_tokens": input_tokens,
            "n_output_tokens": output_tokens,
            "outcome_counts": dict(outcomes),
            "score_eligible_outcome_counts": dict(eligible_outcomes),
            "started_at": initialized_at,
            "updated_at": _utc_now(),
            "elapsed_seconds": elapsed,
        }
        if completed:
            state["finished_at"] = _utc_now()
        _atomic_write_json(spec.paths.state, state)
        if completed:
            result = {
                "schema": POOL_RESULT_SCHEMA,
                **state,
                "selection_sha256": spec.marker["selection_sha256"],
                "n_attempts": spec.marker["n_attempts"],
                "task_count": spec.marker["task_count"],
            }
            _atomic_write_json(spec.paths.result, result)
        return state
    finally:
        _close_lock(lock)


def _publish_attempt_selection(
    spec: PoolSpec, *, keep_official_attempt_index: int
) -> tuple[dict[str, Any], bool]:
    if keep_official_attempt_index != 1:
        raise PoolError("queued-attempt retirement currently requires --keep-attempt 1")
    existing = _attempt_selection(spec)
    if existing is not None:
        if existing["keep_official_attempt_index"] != keep_official_attempt_index:
            raise PoolError("pool already has a different attempt selection")
        return existing, False
    configured = spec.marker.get("n_attempts")
    if not isinstance(configured, int) or isinstance(configured, bool):
        raise PoolError("pool has no valid official-attempt count")
    now = _utc_now()
    selection = {
        "schema": ATTEMPT_SELECTION_SCHEMA,
        "policy": QUEUED_ATTEMPT_RETIREMENT_POLICY,
        "job_name": spec.paths.root.name,
        "config_contract_hash": spec.contract_hash,
        "selection_sha256": spec.marker.get("selection_sha256"),
        "keep_official_attempt_index": keep_official_attempt_index,
        "excluded_official_attempt_indices": list(range(2, configured + 1)),
        "reason": "operator_selected_single_official_attempt",
        "accuracy_denominator": "score_eligible_cells_only",
        "immutable_pool_marker_changed": False,
        "immutable_pool_config_changed": False,
        "activated_at": now,
    }
    _atomic_write_json(spec.paths.attempt_selection, selection)
    return selection, True


def _retirement_projection(
    spec: PoolSpec,
    cell: CellSpec,
    paths: CellPaths,
    result: Mapping[str, Any],
) -> None:
    """Finish or repair state/dispatch after result.json made retirement terminal."""

    execution = _valid_execution(result.get("execution_attempt"))
    retirement = dict(result["retirement"])
    now = str(result.get("committed_at") or result.get("finished_at") or _utc_now())
    state = {
        "schema": CELL_STATE_SCHEMA,
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
        "status": "completed",
        "outcome": "retired",
        "execution_attempt": execution,
        "retirement": retirement,
        "updated_at": now,
        "finished_at": result.get("finished_at") or now,
    }
    prior_dispatch = _read_json(paths.dispatch) if paths.dispatch.is_file() else {}
    dispatch = dict(prior_dispatch)
    dispatch.update(
        {
            "schema": CELL_DISPATCH_SCHEMA,
            "cell_id": cell.cell_id,
            "task_name": cell.task_name,
            "task_ref": cell.task_ref,
            "official_attempt_index": cell.official_attempt_index,
            "config_contract_hash": spec.contract_hash,
            "status": "terminal",
            "outcome": "retired",
            "execution_attempt": execution,
            "retirement": retirement,
            "updated_at": now,
        }
    )
    dispatch.pop("claim", None)
    _atomic_write_json(paths.state, state)
    _atomic_write_json(paths.dispatch, dispatch)


def _snapshot_retirement_originals(
    spec: PoolSpec, cell: CellSpec, paths: CellPaths
) -> dict[str, dict[str, Any]]:
    return _snapshot_retirement_originals_at(
        spec, cell, paths, destination=_retirement_root(paths)
    )


def _snapshot_task_retirement_originals(
    spec: PoolSpec, cell: CellSpec, paths: CellPaths
) -> dict[str, dict[str, Any]]:
    return _snapshot_retirement_originals_at(
        spec, cell, paths, destination=_task_retirement_root(paths)
    )


def _snapshot_retirement_originals_at(
    spec: PoolSpec,
    cell: CellSpec,
    paths: CellPaths,
    *,
    destination: Path,
) -> dict[str, dict[str, Any]]:
    destination = _ensure_dir(destination)
    originals: dict[str, dict[str, Any]] = {}
    for source_name, snapshot_name in (
        ("dispatch.json", "original-dispatch.json"),
        ("state.json", "original-state.json"),
    ):
        source = paths.root / source_name
        if not source.is_file():
            if source_name == "dispatch.json":
                raise PoolError(f"queued cell has no dispatch artifact: {cell.cell_id}")
            continue
        # Parse before copying so a malformed control artifact cannot be
        # blessed merely by moving it under a retirement directory.
        parsed = _read_json(source)
        if (
            parsed.get("cell_id") != cell.cell_id
            or parsed.get("official_attempt_index") != cell.official_attempt_index
            or parsed.get("config_contract_hash") != spec.contract_hash
        ):
            raise PoolError(f"retirement source identity mismatch: {cell.cell_id}")
        payload = _read_bounded_regular_bytes(source, max_bytes=MAX_ARTIFACT_BYTES)
        snapshot = destination / snapshot_name
        _atomic_write_bytes(snapshot, payload)
        originals[source_name] = {
            "source_path": str(source.relative_to(spec.paths.root)),
            "snapshot_path": str(snapshot.relative_to(spec.paths.root)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return originals


def retire_queued_attempts(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    keep_official_attempt_index: int = 1,
) -> dict[str, Any]:
    """Retire unlocked duplicate queue slots while preserving immutable history.

    The additive attempt-selection artifact changes only metric eligibility;
    ``pool.json`` and ``config.json`` remain byte-for-byte untouched. A retired
    ``result.json`` is intentionally published before its state/dispatch
    projection, so even workers that imported an older module will skip it.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    operation_lock = _open_lock(spec.paths.attempt_retirement_lock, blocking=True)
    assert operation_lock is not None
    retired: list[str] = []
    already_retired: list[str] = []
    already_terminal: list[str] = []
    skipped_owned: list[str] = []
    skipped_reserved: list[str] = []
    kept = [
        cell.cell_id
        for cell in spec.cells
        if cell.official_attempt_index == keep_official_attempt_index
    ]
    try:
        selection, selection_created = _publish_attempt_selection(
            spec, keep_official_attempt_index=keep_official_attempt_index
        )
        for cell in spec.cells:
            # This branch deliberately performs no cell-path read or write.
            if cell.official_attempt_index == keep_official_attempt_index:
                continue
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                skipped_owned.append(cell.cell_id)
                continue
            try:
                result = _cell_result(spec, cell)
                if result is not None:
                    if result.get("outcome") == "retired":
                        _retirement_projection(spec, cell, paths, result)
                        already_retired.append(cell.cell_id)
                    else:
                        already_terminal.append(cell.cell_id)
                    continue
                if paths.external_reservation.is_file():
                    _read_external_reservation(spec, cell)
                    skipped_reserved.append(cell.cell_id)
                    continue

                originals = _snapshot_retirement_originals(spec, cell, paths)
                dispatch = _read_json(paths.dispatch)
                execution = _read_execution(paths, dispatch)
                retired_at = _utc_now()
                receipt_path = _retirement_root(paths) / "receipt.json"
                receipt = {
                    "schema": QUEUED_ATTEMPT_RETIREMENT_SCHEMA,
                    "policy": QUEUED_ATTEMPT_RETIREMENT_POLICY,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "task_ref": cell.task_ref,
                    "official_attempt_index": cell.official_attempt_index,
                    "config_contract_hash": spec.contract_hash,
                    "keep_official_attempt_index": keep_official_attempt_index,
                    "score_eligible": False,
                    "model_called_by_retirement": False,
                    "prior_execution_attempt": execution,
                    "original_artifacts": originals,
                    "attempts_directory_preserved": True,
                    "harbor_job_preserved": True,
                    "reason": "operator_selected_single_official_attempt",
                    "retired_at": retired_at,
                }
                _atomic_write_json(receipt_path, receipt)
                receipt_bytes = _read_bounded_regular_bytes(
                    receipt_path, max_bytes=2 * 1024 * 1024
                )
                retirement = {
                    "schema": QUEUED_ATTEMPT_RETIREMENT_SCHEMA,
                    "policy": QUEUED_ATTEMPT_RETIREMENT_POLICY,
                    "keep_official_attempt_index": keep_official_attempt_index,
                    "score_eligible": False,
                    "model_called_by_retirement": False,
                    "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                    "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                }
                terminal_result = {
                    "schema": CELL_RESULT_SCHEMA,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "task_ref": cell.task_ref,
                    "source": cell.source,
                    "official_attempt_index": cell.official_attempt_index,
                    "config_contract_hash": spec.contract_hash,
                    "status": "completed",
                    "outcome": "retired",
                    "execution_attempt": execution,
                    "rewards": {},
                    "exception_info": None,
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "retirement": retirement,
                    "finished_at": retired_at,
                    "committed_at": retired_at,
                }
                _validate_retired_result(spec, cell, terminal_result)
                # Sole terminal authority first: old imported worker code will
                # observe this marker before this flock can be acquired again.
                _atomic_write_json(paths.result, terminal_result)
                _retirement_projection(spec, cell, paths, terminal_result)
                retired.append(cell.cell_id)
                _best_effort_event(
                    spec.paths.root,
                    {
                        "event": "queued_duplicate_attempt_retired",
                        "cell_id": cell.cell_id,
                        "task_name": cell.task_name,
                        "official_attempt_index": cell.official_attempt_index,
                        "keep_official_attempt_index": keep_official_attempt_index,
                        "execution_attempt": execution,
                        "receipt_path": retirement["receipt_path"],
                        "recorded_at": retired_at,
                    },
                )
            finally:
                _close_lock(descriptor)
    finally:
        _close_lock(operation_lock)
    state = aggregate_pool(spec)
    return {
        "keep_official_attempt_index": keep_official_attempt_index,
        "attempt_selection_created": selection_created,
        "attempt_selection_path": str(
            spec.paths.attempt_selection.relative_to(spec.paths.root)
        ),
        "kept_cell_ids": kept,
        "retired_cell_ids": retired,
        "already_retired_cell_ids": already_retired,
        "already_terminal_cell_ids": already_terminal,
        "skipped_owned_cell_ids": skipped_owned,
        "skipped_reserved_cell_ids": skipped_reserved,
        **state,
    }


def _publish_task_selection(
    spec: PoolSpec,
    *,
    task_names: Sequence[str],
    reason: str,
) -> tuple[dict[str, Any], bool]:
    raw_task_names = tuple(task_names)
    if any(not isinstance(name, str) or not name for name in raw_task_names):
        raise PoolError("queued-task retirement task names must be nonempty strings")
    requested = sorted(set(raw_task_names))
    if not requested:
        raise PoolError("queued-task retirement requires at least one --task-name")
    known = {cell.task_name for cell in spec.cells}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise PoolError(
            "queued-task retirement task names are not in this pool: "
            + ", ".join(unknown)
        )
    if set(requested) == known:
        raise PoolError("queued-task retirement may not exclude every task")
    normalized_reason = reason.strip() if isinstance(reason, str) else ""
    if (
        not normalized_reason
        or "\n" in normalized_reason
        or "\r" in normalized_reason
        or len(normalized_reason) > 512
    ):
        raise PoolError(
            "queued-task retirement reason must be a nonempty single line "
            "of at most 512 characters"
        )
    existing = _task_selection(spec)
    if existing is not None:
        if (
            existing["excluded_task_names"] != requested
            or existing["reason"] != normalized_reason
        ):
            raise PoolError("pool already has a different task selection")
        return existing, False
    now = _utc_now()
    selection = {
        "schema": TASK_SELECTION_SCHEMA,
        "policy": QUEUED_TASK_RETIREMENT_POLICY,
        "job_name": spec.paths.root.name,
        "config_contract_hash": spec.contract_hash,
        "selection_sha256": spec.marker.get("selection_sha256"),
        "excluded_task_names": requested,
        "reason": normalized_reason,
        "accuracy_denominator": "score_eligible_cells_only",
        "immutable_pool_marker_changed": False,
        "immutable_pool_config_changed": False,
        "activated_at": now,
    }
    _atomic_write_json(spec.paths.task_selection, selection)
    # Re-read through the strict validator before any cell is changed.
    validated = _task_selection(spec)
    if validated is None:
        raise PoolError("task selection did not become durable")
    return validated, True


def _task_retirement_projection(
    spec: PoolSpec,
    cell: CellSpec,
    paths: CellPaths,
    result: Mapping[str, Any],
) -> None:
    """Repair state/dispatch after task-retirement result became terminal."""

    retirement = result.get("retirement")
    if (
        not isinstance(retirement, Mapping)
        or retirement.get("schema") != QUEUED_TASK_RETIREMENT_SCHEMA
    ):
        raise PoolError(f"invalid task retirement projection for {cell.cell_id}")
    _retirement_projection(spec, cell, paths, result)


def retire_queued_tasks(
    *,
    job_name: str,
    task_names: Sequence[str],
    reason: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
) -> dict[str, Any]:
    """Exclude named tasks and retire only their unlocked queued cells.

    ``pool.json`` and ``config.json`` remain immutable.  The task-selection
    artifact is published before cell mutation, so a crash is recoverable by
    rerunning the same command and current workers refuse subsequent claims.
    Each result is the first terminal commit, followed by repairable
    state/dispatch projections.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    operation_lock = _open_lock(spec.paths.task_retirement_lock, blocking=True)
    assert operation_lock is not None
    retired: list[str] = []
    already_retired: list[str] = []
    already_terminal: list[str] = []
    skipped_owned: list[str] = []
    skipped_reserved: list[str] = []
    try:
        selection, selection_created = _publish_task_selection(
            spec, task_names=task_names, reason=reason
        )
        excluded_names = set(selection["excluded_task_names"])
        selected_cells = [
            cell for cell in spec.cells if cell.task_name in excluded_names
        ]
        kept = [
            cell.cell_id for cell in spec.cells if cell.task_name not in excluded_names
        ]
        selection_bytes = _read_bounded_regular_bytes(
            spec.paths.task_selection, max_bytes=1024 * 1024
        )
        selection_digest = hashlib.sha256(selection_bytes).hexdigest()
        selection_relative = str(spec.paths.task_selection.relative_to(spec.paths.root))
        for cell in selected_cells:
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                skipped_owned.append(cell.cell_id)
                continue
            try:
                result = _cell_result(spec, cell)
                if result is not None:
                    retirement = result.get("retirement")
                    if (
                        result.get("outcome") == "retired"
                        and isinstance(retirement, Mapping)
                        and retirement.get("schema") == QUEUED_TASK_RETIREMENT_SCHEMA
                    ):
                        _task_retirement_projection(spec, cell, paths, result)
                        already_retired.append(cell.cell_id)
                    else:
                        already_terminal.append(cell.cell_id)
                    continue
                if paths.external_reservation.is_file():
                    _read_external_reservation(spec, cell)
                    skipped_reserved.append(cell.cell_id)
                    continue

                originals = _snapshot_task_retirement_originals(spec, cell, paths)
                dispatch = _read_json(paths.dispatch)
                execution = _read_execution(paths, dispatch)
                retired_at = _utc_now()
                receipt_path = _task_retirement_root(paths) / "receipt.json"
                receipt = {
                    "schema": QUEUED_TASK_RETIREMENT_SCHEMA,
                    "policy": QUEUED_TASK_RETIREMENT_POLICY,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "task_ref": cell.task_ref,
                    "official_attempt_index": cell.official_attempt_index,
                    "config_contract_hash": spec.contract_hash,
                    "selection_sha256": spec.marker.get("selection_sha256"),
                    "task_selection_path": selection_relative,
                    "task_selection_sha256": selection_digest,
                    "score_eligible": False,
                    "model_called_by_retirement": False,
                    "prior_execution_attempt": execution,
                    "original_artifacts": originals,
                    "attempts_directory_preserved": True,
                    "harbor_job_preserved": True,
                    "reason": selection["reason"],
                    "retired_at": retired_at,
                }
                _atomic_write_json(receipt_path, receipt)
                receipt_bytes = _read_bounded_regular_bytes(
                    receipt_path, max_bytes=2 * 1024 * 1024
                )
                retirement = {
                    "schema": QUEUED_TASK_RETIREMENT_SCHEMA,
                    "policy": QUEUED_TASK_RETIREMENT_POLICY,
                    "score_eligible": False,
                    "model_called_by_retirement": False,
                    "reason": selection["reason"],
                    "task_selection_path": selection_relative,
                    "task_selection_sha256": selection_digest,
                    "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                    "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                }
                terminal_result = {
                    "schema": CELL_RESULT_SCHEMA,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "task_ref": cell.task_ref,
                    "source": cell.source,
                    "official_attempt_index": cell.official_attempt_index,
                    "config_contract_hash": spec.contract_hash,
                    "status": "completed",
                    "outcome": "retired",
                    "execution_attempt": execution,
                    "rewards": {},
                    "exception_info": None,
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "retirement": retirement,
                    "finished_at": retired_at,
                    "committed_at": retired_at,
                }
                _validate_task_retired_result(spec, cell, terminal_result)
                # Result is the sole terminal authority and must be visible
                # before the repairable state/dispatch projections.
                _atomic_write_json(paths.result, terminal_result)
                _task_retirement_projection(spec, cell, paths, terminal_result)
                retired.append(cell.cell_id)
                _best_effort_event(
                    spec.paths.root,
                    {
                        "event": "queued_task_retired",
                        "cell_id": cell.cell_id,
                        "task_name": cell.task_name,
                        "official_attempt_index": cell.official_attempt_index,
                        "execution_attempt": execution,
                        "reason": selection["reason"],
                        "receipt_path": retirement["receipt_path"],
                        "recorded_at": retired_at,
                    },
                )
            finally:
                _close_lock(descriptor)
    finally:
        _close_lock(operation_lock)
    state = aggregate_pool(spec)
    return {
        "excluded_task_names": list(selection["excluded_task_names"]),
        "task_selection_reason": selection["reason"],
        "task_selection_created": selection_created,
        "task_selection_path": str(
            spec.paths.task_selection.relative_to(spec.paths.root)
        ),
        "kept_cell_ids": kept,
        "retired_cell_ids": retired,
        "already_retired_cell_ids": already_retired,
        "already_terminal_cell_ids": already_terminal,
        "skipped_owned_cell_ids": skipped_owned,
        "skipped_reserved_cell_ids": skipped_reserved,
        "partial": bool(skipped_owned or skipped_reserved),
        **state,
    }


def _valid_execution(value: Any) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def _read_execution(paths: CellPaths, dispatch: Mapping[str, Any] | None = None) -> int:
    state_value = 0
    if paths.state.is_file():
        try:
            state_value = _valid_execution(
                _read_json(paths.state).get("execution_attempt")
            )
        except PoolError:
            state_value = 0
    dispatch_value = _valid_execution(
        dispatch.get("execution_attempt") if isinstance(dispatch, Mapping) else None
    )
    return max(state_value, dispatch_value)


def _read_worker_exception(paths: CellPaths, cell: CellSpec) -> tuple[int, str]:
    if not paths.exception.is_file():
        return 0, "unknown"
    value = _read_json(paths.exception)
    count = value.get("worker_exception_count")
    exception_type = value.get("exception_type") or value.get("type")
    if (
        value.get("cell_id") != cell.cell_id
        or value.get("schema") != WORKER_EXCEPTION_SCHEMA
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or not isinstance(exception_type, str)
        or not exception_type
    ):
        raise PoolError(f"invalid worker exception record for {cell.cell_id}")
    if _harbor_resume_config_exception_was_repaired(paths, cell, value):
        return 0, "repaired_harbor_resume_config_incident"
    return count, exception_type


def _harbor_resume_config_exception_evidence(
    cell: CellSpec, value: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Recognize only the pool's former raw-vs-normalized config false mismatch."""

    traceback_text = value.get("traceback")
    execution = _valid_execution(value.get("execution_attempt"))
    count = value.get("worker_exception_count")
    if (
        value.get("schema") != WORKER_EXCEPTION_SCHEMA
        or value.get("cell_id") != cell.cell_id
        or value.get("task_name") != cell.task_name
        or value.get("official_attempt_index") != cell.official_attempt_index
        or value.get("exception_type") != "PoolError"
        or value.get("message") != HARBOR_RESUME_CONFIG_ERROR_MESSAGE
        or execution < 1
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or _parse_time(value.get("recorded_at")) is None
        or not isinstance(traceback_text, str)
        or not traceback_text.startswith("Traceback (most recent call last):\n")
        or not traceback_text.endswith(
            f"PoolError: {HARBOR_RESUME_CONFIG_ERROR_MESSAGE}\n"
        )
        or re.search(
            r'File "[^"\n]*terminal_bench_pool\.py", line [0-9]+, '
            r"in _execute_harbor\n",
            traceback_text,
        )
        is None
    ):
        return None
    return {
        "classifier": HARBOR_RESUME_CONFIG_INCIDENT,
        "execution_attempt": execution,
        "worker_exception_count": count,
        "exception_sha256": hashlib.sha256(_canonical_bytes(dict(value))).hexdigest(),
    }


def _harbor_resume_config_exception_was_repaired(
    paths: CellPaths, cell: CellSpec, exception: Mapping[str, Any]
) -> bool:
    """Let a durable explicit repair clear exactly one persisted exception."""

    evidence = _harbor_resume_config_exception_evidence(cell, exception)
    receipt_path = paths.harbor_resume_config_repair
    if evidence is None or not receipt_path.is_file() or receipt_path.is_symlink():
        return False
    try:
        receipt = _read_json(receipt_path, max_bytes=2 * 1024 * 1024)
    except PoolError:
        return False
    execution = evidence["execution_attempt"]
    expected_terminal = (
        paths.attempts_dir
        / f"execution-{execution:04d}"
        / HARBOR_RESUME_CONFIG_TERMINAL_ARCHIVE
    )
    archived_exception = expected_terminal / "exception.json"
    archived_result = expected_terminal / "result.json"
    if (
        receipt.get("schema") != "terminal_bench_harbor_resume_config_repair_v1"
        or receipt.get("incident") != HARBOR_RESUME_CONFIG_INCIDENT
        or receipt.get("cell_id") != cell.cell_id
        or receipt.get("task_name") != cell.task_name
        or receipt.get("task_ref") != cell.task_ref
        or receipt.get("official_attempt_index") != cell.official_attempt_index
        or receipt.get("execution_attempt") != execution
        or receipt.get("exception_sha256") != evidence["exception_sha256"]
        or receipt.get("official_attempt_consumed_by_repair") is not False
        or _parse_time(receipt.get("requeued_at")) is None
        or receipt.get("terminal_archive_path")
        != str(expected_terminal.relative_to(paths.root))
        or not archived_exception.is_file()
        or archived_exception.is_symlink()
        or not archived_result.is_file()
        or archived_result.is_symlink()
    ):
        return False
    try:
        archived = _read_json(archived_exception, max_bytes=MAX_ARTIFACT_BYTES)
    except PoolError:
        return False
    return (
        hashlib.sha256(_canonical_bytes(archived)).hexdigest()
        == evidence["exception_sha256"]
    )


def _read_infrastructure_retry(
    paths: CellPaths, cell: CellSpec
) -> tuple[int, list[dict[str, Any]], int, int, int | None, dict[str, Any] | None]:
    """Read infrastructure history and the legacy-named shared strict budget."""

    if not paths.infrastructure.is_file():
        return 0, [], 0, 0, None, None
    value = _read_json(paths.infrastructure)
    count = value.get("infrastructure_retry_count")
    raw_history = value.get("history")
    proot_failures = value.get("proot_infrastructure_failure_count", 0)
    proot_requeues = value.get("proot_infrastructure_requeue_count", 0)
    max_proot_retries = value.get("max_proot_infrastructure_retries")
    last_proot_failure = value.get("last_proot_infrastructure_failure")
    if (
        value.get("cell_id") != cell.cell_id
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or not isinstance(raw_history, list)
        or not isinstance(proot_failures, int)
        or isinstance(proot_failures, bool)
        or proot_failures < 0
        or not isinstance(proot_requeues, int)
        or isinstance(proot_requeues, bool)
        or proot_requeues < 0
        or proot_requeues > proot_failures
        or (
            max_proot_retries is not None
            and (
                not isinstance(max_proot_retries, int)
                or isinstance(max_proot_retries, bool)
                or max_proot_retries < 0
            )
        )
        or (
            last_proot_failure is not None
            and not isinstance(last_proot_failure, Mapping)
        )
    ):
        raise PoolError(f"invalid infrastructure retry record for {cell.cell_id}")
    history = [dict(item) for item in raw_history if isinstance(item, Mapping)]
    return (
        count,
        history,
        proot_failures,
        proot_requeues,
        max_proot_retries,
        dict(last_proot_failure) if isinstance(last_proot_failure, Mapping) else None,
    )


def _select_cell(
    spec: PoolSpec,
    *,
    task_name: str,
    official_attempt_index: int = 1,
) -> CellSpec:
    matches = [
        cell
        for cell in spec.cells
        if cell.task_name == task_name
        and cell.official_attempt_index == official_attempt_index
    ]
    if len(matches) != 1:
        raise PoolError(
            "external trial target must identify exactly one immutable pool cell"
        )
    return matches[0]


def _resolve_external_trial_dir(spec: PoolSpec, source_trial_dir: Path) -> Path:
    """Resolve one direct Harbor trial without accepting arbitrary host paths."""

    source = Path(source_trial_dir).expanduser().resolve()
    jobs_root = spec.paths.root.parent.resolve()
    if source.is_symlink() or not source.is_dir():
        raise PoolError(f"external Harbor trial directory does not exist: {source}")
    try:
        source.relative_to(jobs_root)
    except ValueError as exc:
        raise PoolError(
            "external Harbor trial must be inside the pool jobs root"
        ) from exc
    if source == spec.paths.root or spec.paths.root in source.parents:
        raise PoolError("a pool cannot import a trial from itself")
    # Direct Harbor jobs and shared pool jobs are both immediate children of
    # jobs_root. Requiring the trial to be one directory below its source job
    # keeps the import boundary narrow and auditable.
    if source.parent.parent != jobs_root:
        raise PoolError(
            "external Harbor trial must be an immediate child of a direct job"
        )
    return source


def _resolve_external_job_dir(spec: PoolSpec, source_job_dir: Path) -> Path:
    source = Path(source_job_dir).expanduser().resolve()
    jobs_root = spec.paths.root.parent.resolve()
    if source.is_symlink() or not source.is_dir() or source.parent != jobs_root:
        raise PoolError(
            "external Harbor job must be a direct job in the pool jobs root"
        )
    if source == spec.paths.root:
        raise PoolError("a pool cannot reserve its own controller")
    return source


def _read_external_reservation(
    spec: PoolSpec,
    cell: CellSpec,
) -> dict[str, Any]:
    path = spec.cell_paths(cell).external_reservation
    value = _read_json(path, max_bytes=2 * 1024 * 1024)
    reservation_id = value.get("reservation_id")
    source_trial_dir = value.get("source_trial_dir")
    reservation_kind = value.get("reservation_kind")
    if (
        value.get("schema") != CELL_EXTERNAL_RESERVATION_SCHEMA
        or value.get("cell_id") != cell.cell_id
        or value.get("task_name") != cell.task_name
        or value.get("task_ref") != cell.task_ref
        or value.get("official_attempt_index") != cell.official_attempt_index
        or value.get("config_contract_hash") != spec.contract_hash
        or value.get("status") != "active"
        or not isinstance(reservation_id, str)
        or re.fullmatch(r"[0-9a-f]{32}", reservation_id) is None
        or reservation_kind not in {"external_trial", "source_controller_pending_task"}
        or (
            reservation_kind == "external_trial"
            and (not isinstance(source_trial_dir, str) or not source_trial_dir)
        )
        or (
            reservation_kind == "source_controller_pending_task"
            and source_trial_dir is not None
        )
        or _parse_time(value.get("reserved_at")) is None
    ):
        raise PoolError(f"invalid external reservation for {cell.cell_id}")
    return value


def reserve_external_harbor_trial(
    *,
    job_name: str,
    task_name: str,
    source_trial_dir: Path,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    official_attempt_index: int = 1,
    reservation_id: str | None = None,
) -> dict[str, Any]:
    """Prevent workers claiming a cell while a direct Harbor trial finishes.

    Reservation creation and worker claiming use the same per-cell flock.  A
    caller must therefore reserve before dispatching workers; if a worker
    already owns the cell this operation fails instead of waiting indefinitely.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    cell = _select_cell(
        spec,
        task_name=task_name,
        official_attempt_index=official_attempt_index,
    )
    source = _resolve_external_trial_dir(spec, source_trial_dir)
    paths = spec.cell_paths(cell).ensure()
    descriptor = _open_lock(paths.claim_lock, blocking=False)
    if descriptor is None:
        raise PoolError("cannot reserve an external trial after a worker claimed it")
    try:
        if _cell_result(spec, cell) is not None:
            raise PoolError("cannot reserve an already-terminal pool cell")
        if not _cell_is_score_eligible(
            cell, _attempt_selection(spec), _task_selection(spec)
        ):
            raise PoolError("cannot reserve a cell excluded by pool selection")
        if paths.external_reservation.is_file():
            existing = _read_external_reservation(spec, cell)
            if existing.get("source_trial_dir") != str(source):
                raise PoolError(
                    "pool cell already has a different external reservation"
                )
            if (
                reservation_id is not None
                and existing.get("reservation_id") != reservation_id
            ):
                raise PoolError("external reservation id does not match")
            return existing
        identifier = reservation_id or uuid.uuid4().hex
        if re.fullmatch(r"[0-9a-f]{32}", identifier) is None:
            raise PoolError("external reservation id must be 32 lowercase hex digits")
        now = _utc_now()
        reservation = {
            "schema": CELL_EXTERNAL_RESERVATION_SCHEMA,
            "reservation_id": identifier,
            "reservation_kind": "external_trial",
            "status": "active",
            "cell_id": cell.cell_id,
            "task_name": cell.task_name,
            "task_ref": cell.task_ref,
            "official_attempt_index": cell.official_attempt_index,
            "config_contract_hash": spec.contract_hash,
            "source_job_dir": str(source.parent),
            "source_trial_dir": str(source),
            "reserved_at": now,
        }
        prior_dispatch = _read_json(paths.dispatch) if paths.dispatch.is_file() else {}
        dispatch = dict(prior_dispatch)
        dispatch.update(
            {
                "schema": CELL_DISPATCH_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "reserved_external",
                "external_reservation_id": identifier,
                "external_source_trial_dir": str(source),
                "updated_at": now,
            }
        )
        dispatch.pop("claim", None)
        # Reservation is written first. A crash can therefore only leave a
        # blocked cell, never an accidentally claimable duplicate attempt.
        _atomic_write_json(paths.external_reservation, reservation)
        _atomic_write_json(paths.dispatch, dispatch)
        _best_effort_event(
            spec.paths.root,
            {
                "event": "external_trial_reserved",
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "reservation_id": identifier,
                "source_trial_dir": str(source),
                "recorded_at": now,
            },
        )
        return reservation
    finally:
        _close_lock(descriptor)


def reserve_external_controller_cell(
    *,
    job_name: str,
    task_name: str,
    source_job_dir: Path,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    official_attempt_index: int = 1,
    reservation_id: str | None = None,
) -> dict[str, Any]:
    """Hold a fresh cell while an older controller may still start that task."""

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    cell = _select_cell(
        spec,
        task_name=task_name,
        official_attempt_index=official_attempt_index,
    )
    source_job = _resolve_external_job_dir(spec, source_job_dir)
    paths = spec.cell_paths(cell).ensure()
    descriptor = _open_lock(paths.claim_lock, blocking=False)
    if descriptor is None:
        raise PoolError("cannot reserve a controller cell after a worker claimed it")
    try:
        if _cell_result(spec, cell) is not None:
            raise PoolError("cannot reserve an already-terminal pool cell")
        if not _cell_is_score_eligible(
            cell, _attempt_selection(spec), _task_selection(spec)
        ):
            raise PoolError("cannot reserve a cell excluded by pool selection")
        if paths.external_reservation.is_file():
            existing = _read_external_reservation(spec, cell)
            if existing.get(
                "reservation_kind"
            ) != "source_controller_pending_task" or existing.get(
                "source_job_dir"
            ) != str(
                source_job
            ):
                raise PoolError(
                    "pool cell already has a different external reservation"
                )
            if (
                reservation_id is not None
                and existing.get("reservation_id") != reservation_id
            ):
                raise PoolError("external reservation id does not match")
            return existing
        identifier = reservation_id or uuid.uuid4().hex
        if re.fullmatch(r"[0-9a-f]{32}", identifier) is None:
            raise PoolError("external reservation id must be 32 lowercase hex digits")
        now = _utc_now()
        reservation = {
            "schema": CELL_EXTERNAL_RESERVATION_SCHEMA,
            "reservation_id": identifier,
            "reservation_kind": "source_controller_pending_task",
            "status": "active",
            "cell_id": cell.cell_id,
            "task_name": cell.task_name,
            "task_ref": cell.task_ref,
            "official_attempt_index": cell.official_attempt_index,
            "config_contract_hash": spec.contract_hash,
            "source_job_dir": str(source_job),
            "source_trial_dir": None,
            "reserved_at": now,
        }
        prior_dispatch = _read_json(paths.dispatch) if paths.dispatch.is_file() else {}
        dispatch = dict(prior_dispatch)
        dispatch.update(
            {
                "schema": CELL_DISPATCH_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "reserved_external",
                "external_reservation_id": identifier,
                "external_source_job_dir": str(source_job),
                "updated_at": now,
            }
        )
        dispatch.pop("claim", None)
        _atomic_write_json(paths.external_reservation, reservation)
        _atomic_write_json(paths.dispatch, dispatch)
        _best_effort_event(
            spec.paths.root,
            {
                "event": "external_controller_cell_reserved",
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "reservation_id": identifier,
                "source_job_dir": str(source_job),
                "recorded_at": now,
            },
        )
        return reservation
    finally:
        _close_lock(descriptor)


def release_external_harbor_trial_reservation(
    *,
    job_name: str,
    task_name: str,
    reservation_id: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    official_attempt_index: int = 1,
) -> dict[str, Any]:
    """Explicitly return a reserved cell to the fresh Codex queue."""

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    cell = _select_cell(
        spec,
        task_name=task_name,
        official_attempt_index=official_attempt_index,
    )
    paths = spec.cell_paths(cell).ensure()
    descriptor = _open_lock(paths.claim_lock, blocking=False)
    if descriptor is None:
        raise PoolError("cannot release an external reservation while cell is owned")
    try:
        if _cell_result(spec, cell) is not None:
            raise PoolError("cannot release a reservation from a terminal cell")
        reservation = _read_external_reservation(spec, cell)
        if reservation.get("reservation_id") != reservation_id:
            raise PoolError("external reservation id does not match")
        now = _utc_now()
        dispatch = _read_json(paths.dispatch) if paths.dispatch.is_file() else {}
        dispatch.update(
            {
                "schema": CELL_DISPATCH_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "updated_at": now,
            }
        )
        for key in (
            "claim",
            "external_reservation_id",
            "external_source_trial_dir",
            "external_source_job_dir",
        ):
            dispatch.pop(key, None)
        paths.external_reservation.unlink()
        _fsync_directory(paths.root)
        _atomic_write_json(paths.dispatch, dispatch)
        _best_effort_event(
            spec.paths.root,
            {
                "event": "external_trial_reservation_released",
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "reservation_id": reservation_id,
                "recorded_at": now,
            },
        )
    finally:
        _close_lock(descriptor)
    return {**aggregate_pool(spec), "cell_status": "queued"}


def _retry_exhausted_result(
    spec: PoolSpec,
    cell: CellSpec,
    paths: CellPaths,
    *,
    execution_attempt: int,
    exception_type: str,
    worker_exception_count: int,
) -> None:
    now = _utc_now()
    state = {
        "schema": CELL_STATE_SCHEMA,
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
        "status": "pool_error",
        "outcome": "unclassified_worker_exception",
        "execution_attempt": execution_attempt,
        "worker_exception_count": worker_exception_count,
        "exception_type": exception_type,
        "updated_at": now,
        "finished_at": now,
    }
    dispatch = {
        "schema": CELL_DISPATCH_SCHEMA,
        **{
            key: state[key]
            for key in (
                "cell_id",
                "task_name",
                "task_ref",
                "official_attempt_index",
                "config_contract_hash",
                "execution_attempt",
                "worker_exception_count",
            )
        },
        "status": "terminal",
        "outcome": state["outcome"],
        "updated_at": now,
    }
    result = {
        "schema": CELL_RESULT_SCHEMA,
        **{
            key: state[key]
            for key in (
                "cell_id",
                "task_name",
                "task_ref",
                "official_attempt_index",
                "config_contract_hash",
                "execution_attempt",
                "worker_exception_count",
                "exception_type",
                "outcome",
                "finished_at",
            )
        },
        "status": "pool_error",
        "rewards": {},
        "n_input_tokens": 0,
        "n_output_tokens": 0,
        "committed_at": now,
    }
    # Result is deliberately last: it is the sole terminal commit marker.
    _atomic_write_json(paths.state, state)
    _atomic_write_json(paths.dispatch, dispatch)
    _atomic_write_json(paths.result, result)


def _proot_retry_exhausted_result(
    spec: PoolSpec,
    cell: CellSpec,
    paths: CellPaths,
    *,
    execution_attempt: int,
    proot_failure_count: int,
    proot_requeue_count: int,
    max_proot_retries: int,
    failure: Mapping[str, Any],
) -> None:
    """Commit an explicit pool error after the shared strict-infra cap."""

    now = _utc_now()
    infrastructure_kind = str(failure.get("infrastructure_kind") or "proot")
    worker_exception_count, _worker_exception_type = _read_worker_exception(paths, cell)
    shared = {
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
        "execution_attempt": execution_attempt,
        "worker_exception_count": worker_exception_count,
        "proot_infrastructure_failure_count": proot_failure_count,
        "proot_infrastructure_requeue_count": proot_requeue_count,
        "max_proot_infrastructure_retries": max_proot_retries,
        "strict_infrastructure_failure_count": proot_failure_count,
        "strict_infrastructure_requeue_count": proot_requeue_count,
        "max_strict_infrastructure_retries": max_proot_retries,
        "last_infrastructure_failure": dict(failure),
    }
    state = {
        "schema": CELL_STATE_SCHEMA,
        **shared,
        "status": "pool_error",
        "outcome": f"{infrastructure_kind}_infrastructure_retry_exhausted",
        "updated_at": now,
        "finished_at": now,
    }
    dispatch = {
        "schema": CELL_DISPATCH_SCHEMA,
        **shared,
        "status": "terminal",
        "outcome": state["outcome"],
        "updated_at": now,
    }
    result = {
        "schema": CELL_RESULT_SCHEMA,
        **shared,
        "status": "pool_error",
        "outcome": state["outcome"],
        "exception_type": (
            "RetryableSGLangTrialError"
            if infrastructure_kind == "sglang"
            else "RetryablePRootTrialError"
        ),
        "rewards": {},
        "n_input_tokens": 0,
        "n_output_tokens": 0,
        "finished_at": now,
        "committed_at": now,
    }
    # Result remains the sole terminal commit marker.
    _atomic_write_json(paths.state, state)
    _atomic_write_json(paths.dispatch, dispatch)
    _atomic_write_json(paths.result, result)


def _try_claim(
    spec: PoolSpec,
    cell: CellSpec,
    *,
    worker_id: str,
    device_label: str,
    instance_id: str,
    lease_seconds: float,
    max_worker_exceptions: int,
    max_proot_infrastructure_retries: int = 2,
) -> ClaimedCell | None:
    paths = spec.cell_paths(cell).ensure()
    descriptor = _open_lock(paths.claim_lock, blocking=False)
    if descriptor is None:
        return None
    release = True
    try:
        if _cell_result(spec, cell) is not None:
            return None
        from terminal_bench_eval.agent_timeout_policy import commit_existing
        if commit_existing(spec, cell, descriptor):
            return None
        # An additive attempt selection can be activated while an excluded
        # cell is already owned.  Once that owner releases the flock, do not
        # let a retry or an older queued dispatch make the cell claimable
        # again; the retirement pass will preserve a real terminal result or
        # commit an explicit retired result for the now-unowned cell.
        if not _cell_is_score_eligible(
            cell, _attempt_selection(spec), _task_selection(spec)
        ):
            return None
        # A reservation is durable dispatch authority for a terminal trial
        # that is completing in a pre-existing Harbor job.  The reservation
        # is created while holding this same claim lock, so a worker can never
        # slip between reservation creation and this check.
        if paths.external_reservation.is_file():
            _read_external_reservation(spec, cell)
            return None
        prior_dispatch: dict[str, Any] = {}
        if paths.dispatch.is_file():
            prior_dispatch = _read_json(paths.dispatch)
        if max_proot_infrastructure_retries < 0:
            raise PoolError("max PRoot infrastructure retries must be nonnegative")
        if _reconcile_unjournaled_proot_archives(
            spec,
            cell,
            paths,
            descriptor=descriptor,
            requested_max_proot_retries=max_proot_infrastructure_retries,
        ):
            return None
        (
            _infrastructure_count,
            _infrastructure_history,
            proot_failure_count,
            proot_requeue_count,
            recorded_max_proot_retries,
            last_proot_failure,
        ) = _read_infrastructure_retry(paths, cell)
        effective_max_proot_retries = (
            recorded_max_proot_retries
            if recorded_max_proot_retries is not None
            else max_proot_infrastructure_retries
        )
        if proot_failure_count > effective_max_proot_retries:
            if last_proot_failure is None:
                raise PoolError(
                    f"exhausted PRoot retry record has no failure: {cell.cell_id}"
                )
            _proot_retry_exhausted_result(
                spec,
                cell,
                paths,
                execution_attempt=_read_execution(paths, prior_dispatch),
                proot_failure_count=proot_failure_count,
                proot_requeue_count=proot_requeue_count,
                max_proot_retries=effective_max_proot_retries,
                failure=last_proot_failure,
            )
            return None
        retry_at = _parse_time(prior_dispatch.get("retry_not_before"))
        if retry_at is not None and retry_at > datetime.now(UTC):
            return None
        previous_execution = _read_execution(paths, prior_dispatch)
        exception_count, exception_type = _read_worker_exception(paths, cell)
        if exception_count >= 1:
            _retry_exhausted_result(
                spec,
                cell,
                paths,
                execution_attempt=previous_execution,
                exception_type=exception_type,
                worker_exception_count=exception_count,
            )
            return None
        execution = previous_execution + 1
        now = _utc_now()
        claim = {
            "lease_id": uuid.uuid4().hex,
            "worker_id": worker_id,
            "instance_id": instance_id,
            "device": device_label,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "claimed_at": now,
            "heartbeat_at": now,
            "lease_expires_at": _future_time(lease_seconds),
            "harbor_job_path": str(paths.harbor_job.relative_to(spec.paths.root)),
        }
        dispatch: dict[str, Any] = {
            "schema": CELL_DISPATCH_SCHEMA,
            "cell_id": cell.cell_id,
            "task_name": cell.task_name,
            "task_ref": cell.task_ref,
            "official_attempt_index": cell.official_attempt_index,
            "config_contract_hash": spec.contract_hash,
            "status": "running",
            "execution_attempt": execution,
            "worker_exception_count": exception_count,
            "enqueued_at": prior_dispatch.get("enqueued_at")
            or spec.marker.get("initialized_at")
            or now,
            "updated_at": now,
            "harbor_job_path": claim["harbor_job_path"],
            "claim": claim,
        }
        previous_claim = prior_dispatch.get("claim")
        if isinstance(previous_claim, Mapping):
            dispatch["reclaimed_from"] = {
                key: previous_claim.get(key)
                for key in ("worker_id", "instance_id", "lease_id", "heartbeat_at")
            }
        _atomic_write_json(paths.dispatch, dispatch)
        attempt_dir = _ensure_dir(paths.attempts_dir / f"execution-{execution:04d}")
        _atomic_write_json(
            attempt_dir / "run.json",
            {
                "schema": RUN_RECORD_SCHEMA,
                "cell_id": cell.cell_id,
                "execution_attempt": execution,
                "worker_id": worker_id,
                "instance_id": instance_id,
                "status": "claimed",
                "claimed_at": now,
            },
        )
        release = False
        return ClaimedCell(
            cell=cell,
            paths=paths,
            descriptor=descriptor,
            execution_attempt=execution,
            dispatch=dispatch,
        )
    finally:
        if release:
            _close_lock(descriptor)


class _Heartbeat:
    def __init__(
        self,
        *,
        claim: ClaimedCell,
        worker_path: Path,
        worker_id: str,
        device_label: str,
        instance_id: str,
        interval_seconds: float,
        lease_seconds: float,
        environment_backend: str = "frozen",
    ) -> None:
        self.claim = claim
        self.worker_path = worker_path
        self.worker_id = worker_id
        self.device_label = device_label
        self.instance_id = instance_id
        self.interval_seconds = interval_seconds
        self.lease_seconds = lease_seconds
        self.environment_backend = environment_backend
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._loop,
            name=f"terminal-bench-pool-heartbeat-{instance_id[:8]}",
            daemon=True,
        )

    def _payloads(self) -> tuple[dict[str, Any], dict[str, Any]]:
        now = _utc_now()
        expiration = _future_time(self.lease_seconds)
        dispatch = dict(self.claim.dispatch)
        raw_claim = dispatch.get("claim")
        claim = dict(raw_claim) if isinstance(raw_claim, Mapping) else {}
        claim.update({"heartbeat_at": now, "lease_expires_at": expiration})
        dispatch.update({"status": "running", "updated_at": now, "claim": claim})
        worker = {
            "schema": WORKER_SCHEMA,
            "worker_id": self.worker_id,
            "instance_id": self.instance_id,
            "status": "running",
            "device": self.device_label,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "cell_id": self.claim.cell.cell_id,
            "task_name": self.claim.cell.task_name,
            "official_attempt_index": self.claim.cell.official_attempt_index,
            "execution_attempt": self.claim.execution_attempt,
            "environment_backend": self.environment_backend,
            "heartbeat_at": now,
            "lease_expires_at": expiration,
        }
        return dispatch, worker

    def tick(self) -> None:
        dispatch, worker = self._payloads()
        _atomic_write_json(self.claim.paths.dispatch, dispatch)
        _atomic_write_json(self.worker_path, worker)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.tick()
            except BaseException:
                # flock ownership remains valid even if an observational write fails.
                continue

    def start(self) -> None:
        self.tick()
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=max(5.0, self.interval_seconds + 1.0))


def _execution_memory_policy(spec: PoolSpec, claim: ClaimedCell) -> dict[str, Any] | None:
    from evaluation_memory import codex_r2eact_agent, requested_memory_profile, memory_profile_receipt

    agent = spec.config["agents"][0]
    if not codex_r2eact_agent(agent):
        return None
    path = claim.paths.root / "memory-policy.json"
    if path.is_symlink():
        raise PoolError("execution memory policy must not be a symlink")
    if path.is_file():
        stored = _read_json(path)
        expected = {**memory_profile_receipt(stored.get("profile")),
                    "base_config_sha256": _digest(spec.config)}
        if stored != expected:
            raise PoolError("execution memory policy does not match the frozen pool")
        return expected
    # Existing executions keep their original memory contract across restarts.
    if claim.paths.job_config.exists():
        return None
    profile = requested_memory_profile()
    if profile is None:
        return None
    return {**memory_profile_receipt(profile), "base_config_sha256": _digest(spec.config)}


def _validate_science_execution_backend(spec: PoolSpec, environment_backend: str) -> None:
    if any(cell.source == "terminal-bench-science/terminal-bench-science" for cell in spec.cells):
        if environment_backend != "docker":
            raise PoolError(
                "TB-Science requires the Docker execution backend for its official "
                "Dockerfiles and separate verifier network policies; PRoot is not "
                "supported. No task was claimed. Use eval_terminal_bench_sc_codex.sh "
                "on a Docker-capable worker."
            )


def _single_job_config(
    spec: PoolSpec,
    claim: ClaimedCell,
    *,
    sglang_url: str,
    environment_backend: str = "frozen",
) -> dict[str, Any]:
    _validate_science_execution_backend(spec, environment_backend)
    config = copy.deepcopy(dict(spec.config))
    config["job_name"] = "harbor-job"
    config["jobs_dir"] = str(claim.paths.root)
    config["n_attempts"] = 1
    config["n_concurrent_trials"] = 1
    config["quiet"] = True
    config["datasets"] = []
    config["tasks"] = [dict(claim.cell.task_config)]
    retry = config.get("retry")
    retry_payload = dict(retry) if isinstance(retry, Mapping) else {}
    retry_payload["max_retries"] = 0
    config["retry"] = retry_payload
    agents = config.get("agents")
    if (
        not isinstance(agents, list)
        or len(agents) != 1
        or not isinstance(agents[0], Mapping)
    ):
        raise PoolError("frozen pool config no longer contains one agent")
    agent = dict(agents[0])
    kwargs = agent.get("kwargs")
    kwargs_payload = dict(kwargs) if isinstance(kwargs, Mapping) else {}
    kwargs_payload["sglang_url"] = sglang_url
    if "checkpoint" in kwargs_payload:
        # Deployment selection is recorded in this execution's Harbor config;
        # the frozen pool config and completed results retain their identity.
        from evaluation_checkpoint import resolve_eval_checkpoint

        kwargs_payload["checkpoint"] = str(resolve_eval_checkpoint(kwargs_payload["checkpoint"]))
    memory_policy = _execution_memory_policy(spec, claim)
    if memory_policy is not None:
        from evaluation_memory import memory_profile_values

        kwargs_payload.update(memory_profile_values(memory_policy["profile"]))
    agent["kwargs"] = kwargs_payload
    if (
        environment_backend in {"docker", "docker-gpu"}
        and agent.get("import_path")
        in {
            "terminal_bench_eval.codex_r2eact_agent:CodexR2EActAgent",
            "terminal_bench_eval.science_preflight:ScienceCodexAgent",
        }
    ):
        # The Responses gateway and its MCP endpoint live on the worker host.
        # Harbor snapshots this top-level field into the *agent phase* network
        # plan, so restricted-network tasks can reach Codex without relaxing
        # the environment baseline, solution, or verifier phases.
        raw_hosts = agent.get("extra_allowed_hosts", [])
        if not isinstance(raw_hosts, list) or any(
            not isinstance(host, str) for host in raw_hosts
        ):
            raise PoolError("Codex Docker agent extra_allowed_hosts must be strings")
        allowed_hosts = list(raw_hosts)
        if "host.docker.internal" not in allowed_hosts:
            allowed_hosts.append("host.docker.internal")
        agent["extra_allowed_hosts"] = allowed_hosts
    config["agents"] = [agent]
    if environment_backend == "docker":
        # The immutable pool freezes the model, task refs, and scoring regime,
        # while worker capability routing selects the execution provider.  A
        # Docker worker must use each task's own resource declaration instead
        # of inheriting the legacy global PRoot 32-CPU/48-GiB override.
        config["environment"] = {
            "import_path": (
                "eval.terminal_bench.docker_gpu_environment:"
                "LocalDockerR2EActEnvironment"
            )
        }
    elif environment_backend == "docker-gpu":
        config["environment"] = {
            "import_path": (
                "eval.terminal_bench.docker_gpu_environment:"
                "LocalDockerGPUEnvironment"
            )
        }
    elif environment_backend != "frozen":
        raise PoolError(
            "worker environment backend must be one of: " "frozen, docker, docker-gpu"
        )
    # Validate the derived config before allowing Harbor to see it.
    try:
        JobConfig.model_validate(config)
    except Exception as exc:
        raise PoolError("derived single-cell Harbor config is invalid") from exc
    return config


def _uses_tb21_uniform_task_budget(spec: PoolSpec) -> bool:
    datasets = spec.config.get("datasets")
    if isinstance(datasets, list) and any(
        isinstance(dataset, Mapping) and dataset.get("name") == TB21_DATASET_SOURCE
        for dataset in datasets
    ):
        return True
    return bool(spec.cells) and all(
        cell.source == TB21_DATASET_SOURCE for cell in spec.cells
    )


def _agent_time_budget(spec: PoolSpec) -> dict[str, Any]:
    agents = spec.config.get("agents")
    agent = (
        agents[0]
        if isinstance(agents, list) and agents and isinstance(agents[0], Mapping)
        else {}
    )
    configured = agent.get("override_timeout_sec")
    unbounded = agent_import_uses_unbounded_timeout(agent.get("import_path"))
    if unbounded and _uses_tb21_uniform_task_budget(spec):
        policy = TB21_UNIFORM_AGENT_TIMEOUT_POLICY
        effective = TB21_PER_TASK_AGENT_TIMEOUT_SECONDS
        source = TB21_UNIFORM_AGENT_TIMEOUT_SOURCE
    elif unbounded:
        policy = UNBOUNDED_AGENT_TIMEOUT_POLICY
        effective = None
        source = "pool_harbor_wrapper"
    elif configured is None:
        policy = "harbor_task_timeout_fallback"
        effective = None
        source = "task_config"
    else:
        policy = "harbor_configured_timeout"
        effective = configured
        source = "job_override"
    return {
        "policy": policy,
        "configured_harbor_timeout_seconds": configured,
        "effective_total_agent_timeout_seconds": effective,
        "effective_timeout_source": source,
    }


def _harbor_command_prefix(spec: PoolSpec, harbor_bin: Path) -> list[str]:
    agents = spec.config.get("agents")
    agent = (
        agents[0]
        if isinstance(agents, list) and agents and isinstance(agents[0], Mapping)
        else {}
    )
    if agent_import_uses_unbounded_timeout(agent.get("import_path")):
        return [sys.executable, "-m", HARBOR_UNBOUNDED_CLI_MODULE]
    return [str(harbor_bin)]


def _execution_agent_time_budget(spec: PoolSpec, claim: ClaimedCell) -> dict[str, Any]:
    """Project the policy evidenced by this concrete execution attempt."""

    attempt = claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    receipt_path = attempt / AGENT_TIME_BUDGET_RECEIPT_NAME
    if receipt_path.is_file() and not receipt_path.is_symlink():
        try:
            receipt = _read_json(receipt_path, max_bytes=1024 * 1024)
        except PoolError:
            receipt = {}
        if receipt.get("schema") == AGENT_TIME_BUDGET_RECEIPT_SCHEMA and receipt.get(
            "policy"
        ) in {
            UNBOUNDED_AGENT_TIMEOUT_POLICY,
            TB21_OFFICIAL_AGENT_TIMEOUT_POLICY,
            TB21_LEGACY_8H_AGENT_TIMEOUT_POLICY,
            TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
        }:
            policy = receipt["policy"]
            effective = receipt.get("effective_total_agent_timeout_seconds")
            return {
                "policy": policy,
                "configured_harbor_timeout_seconds": receipt.get(
                    "configured_harbor_timeout_seconds"
                ),
                "effective_total_agent_timeout_seconds": effective,
                "effective_timeout_source": "execution_receipt",
                "execution_boundary": receipt.get("execution_boundary"),
                **(
                    {
                        "task_timeout_seconds": receipt.get("task_timeout_seconds"),
                        "timeout_multiplier": receipt.get("timeout_multiplier"),
                        "agent_timeout_multiplier": receipt.get(
                            "agent_timeout_multiplier"
                        ),
                        "resolved_multiplier": receipt.get("resolved_multiplier"),
                        "max_timeout_seconds": receipt.get("max_timeout_seconds"),
                    }
                    if policy
                    in {
                        TB21_OFFICIAL_AGENT_TIMEOUT_POLICY,
                        TB21_LEGACY_8H_AGENT_TIMEOUT_POLICY,
                        TB21_UNIFORM_AGENT_TIMEOUT_POLICY,
                    }
                    else {}
                ),
            }
    run_path = attempt / "run.json"
    run = _read_json(run_path, max_bytes=1024 * 1024) if run_path.is_file() else {}
    recorded = run.get("agent_time_budget")
    if isinstance(recorded, Mapping):
        return dict(recorded)
    command = run.get("command")
    if isinstance(command, list) and HARBOR_UNBOUNDED_CLI_MODULE in command:
        return _agent_time_budget(spec)
    planned = _agent_time_budget(spec)
    configured = planned.get("configured_harbor_timeout_seconds")
    return {
        "policy": (
            "harbor_configured_timeout"
            if configured is not None
            else "harbor_task_timeout_fallback"
        ),
        "configured_harbor_timeout_seconds": configured,
        "effective_total_agent_timeout_seconds": configured,
        "effective_timeout_source": (
            "legacy_harbor_command" if configured is not None else "task_config"
        ),
    }


def _ensure_job_config(
    spec: PoolSpec,
    claim: ClaimedCell,
    *,
    sglang_url: str,
    environment_backend: str = "frozen",
) -> None:
    memory_policy = _execution_memory_policy(spec, claim)
    memory_path = claim.paths.root / "memory-policy.json"
    if memory_policy is not None and not memory_path.exists():
        # Commit before job-config so retries can derive exactly the same config
        # even when the next controller does not inherit this launcher's env.
        _atomic_write_json(memory_path, memory_policy)
    expected = _single_job_config(
        spec,
        claim,
        sglang_url=sglang_url,
        environment_backend=environment_backend,
    )
    if claim.paths.job_config.is_file():
        stored = _read_json(claim.paths.job_config)
        if stored != expected:
            # A fresh execution may be claimed by another attested replica.
            # Only its serving origin may change, and never while Harbor still
            # has a live directory to resume. Keep the old config as evidence.
            rerouted = copy.deepcopy(stored)
            try:
                old_url = rerouted["agents"][0]["kwargs"]["sglang_url"]
                rerouted["agents"][0]["kwargs"]["sglang_url"] = sglang_url
            except (KeyError, IndexError, TypeError):
                raise PoolError(f"derived config changed for {claim.cell.cell_id}") from None
            if (rerouted != expected or claim.execution_attempt < 2
                or claim.paths.harbor_job.exists() or claim.paths.harbor_job.is_symlink()
                or claim.paths.result.exists()):
                raise PoolError(f"derived config changed for {claim.cell.cell_id}")
            directory = _ensure_dir(claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}")
            archive = directory / "previous-endpoint-job-config.json"
            if archive.exists() and _read_json(archive) != stored:
                raise PoolError("conflicting endpoint config archive")
            _atomic_write_json(archive, stored)
            _atomic_write_json(directory / "endpoint-reroute.json", {
                "schema": "terminal_bench_fresh_execution_endpoint_reroute_v1",
                "cell_id": claim.cell.cell_id,
                "official_attempt_index": claim.cell.official_attempt_index,
                "execution_attempt": claim.execution_attempt,
                "old_endpoint": old_url, "new_endpoint": sglang_url,
                "config_contract_hash": spec.contract_hash,
                "recorded_at": _utc_now(),
            })
            _atomic_write_json(claim.paths.job_config, expected)
        return
    _atomic_write_json(claim.paths.job_config, expected)


def _harbor_resume_config_matches_contract(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Accept only the two durable encodings Harbor can produce for one config.

    ``job-config.json`` is the pool-authored cell contract.  Harbor 0.21 parses
    that config and persists ``harbor-job/config.json`` using Pydantic's
    ``exclude_defaults=True`` serialization, so byte-for-byte JSON equality is
    too strict after an interrupted first run.  Conversely, comparing parsed
    models alone is too permissive because JobConfig ignores unknown fields and
    performs type coercion.  Restrict resume compatibility to either the exact
    pool config (legacy/raw jobs) or the exact deterministic Harbor
    serialization of that config.  Unknown keys, partial rewrites, coercive
    values, and changes to the agent/task/environment contract remain rejected.
    """

    if dict(observed) == dict(expected):
        return True
    try:
        canonical = JobConfig.model_validate(dict(expected)).model_dump(
            mode="json", exclude_defaults=True
        )
    except Exception:
        return False
    return dict(observed) == canonical


def _local_task_cache_lock(task_ref: str) -> Path:
    root = Path(
        os.environ.get(
            "TERMINAL_BENCH_POOL_CACHE_LOCK_ROOT",
            str(Path(tempfile.gettempdir()) / "terminal-bench-pool-task-cache-locks"),
        )
    ).expanduser()
    return root / f"{hashlib.sha256(task_ref.encode()).hexdigest()}.lock"


def _is_transient_task_cache_transport_error(exc: BaseException) -> bool:
    """Accept only transport leaves, including Python 3.12 TaskGroup wrappers."""

    if isinstance(exc, BaseExceptionGroup):
        leaves = [
            leaf
            for child in exc.exceptions
            for leaf in (
                [child]
                if not isinstance(child, BaseExceptionGroup)
                else _flatten_exception_group(child)
            )
        ]
    else:
        leaves = [exc]
    return bool(leaves) and all(
        isinstance(
            leaf,
            (httpx.TransportError, ConnectionError, TimeoutError, ssl.SSLError),
        )
        for leaf in leaves
    )


def _flatten_exception_group(exc: BaseExceptionGroup) -> list[BaseException]:
    leaves: list[BaseException] = []
    for child in exc.exceptions:
        if isinstance(child, BaseExceptionGroup):
            leaves.extend(_flatten_exception_group(child))
        else:
            leaves.append(child)
    return leaves


def _prewarm_task_cache(task_config: Mapping[str, Any]) -> Path:
    """Materialize and verify one exact package task without registry lookup.

    The content-addressed resolver is installed both here and in the Harbor CLI
    wrapper.  This first pass serializes extraction and verifies Harbor's own
    package digest before the CLI is allowed to consume the cache directory.
    """

    try:
        task = TaskConfig.model_validate(task_config)
        task_id = task.get_task_id()
    except Exception as exc:
        raise PoolError("could not reconstruct pinned task config") from exc
    task_ref = str(task_config.get("ref") or "")
    install_pinned_package_resolution()
    lock_path = _local_task_cache_lock(task_ref)
    lock = _open_lock(lock_path, blocking=True)
    assert lock is not None
    try:
        receipt_path = lock_path.with_suffix(".ready.json")
        expected = {
            "schema": TASK_CACHE_RECEIPT_SCHEMA,
            "task_name": str(task_config.get("name")),
            "task_ref": task_ref,
            "resolution_policy": PINNED_PACKAGE_RESOLUTION_POLICY,
        }
        if receipt_path.is_file():
            try:
                receipt = _read_json(receipt_path, max_bytes=1024 * 1024)
            except PoolError:
                receipt = {}
            cached_path = receipt.get("task_path")
            if all(
                receipt.get(key) == value for key, value in expected.items()
            ) and isinstance(cached_path, str):
                try:
                    verify_pinned_package_task(task, Path(cached_path))
                except PinnedTaskVerificationError:
                    pass
                else:
                    return Path(cached_path).expanduser().resolve()

        async def download(*, overwrite: bool) -> Path:
            result = await TaskClient().download_tasks(
                task_ids=[task_id], overwrite=overwrite, output_dir=None
            )
            if not result.results or result.results[0] is None:
                raise PoolError("Harbor task cache prewarm returned no task")
            task_path = Path(result.results[0].path).expanduser().resolve()
            verify_pinned_package_task(task, task_path)
            return task_path

        def download_with_transport_retry(*, overwrite: bool) -> Path:
            for attempt in range(1, TASK_CACHE_TRANSPORT_ATTEMPTS + 1):
                try:
                    return asyncio.run(download(overwrite=overwrite))
                except Exception as exc:
                    transient = _is_transient_task_cache_transport_error(exc)
                    if not transient:
                        raise
                    if attempt >= TASK_CACHE_TRANSPORT_ATTEMPTS:
                        raise RetryableTaskCacheTransportError(
                            task_ref,
                            exc,
                            attempts=TASK_CACHE_TRANSPORT_ATTEMPTS,
                        ) from exc
                    time.sleep(TASK_CACHE_TRANSPORT_BACKOFF_SECONDS * attempt)
            raise AssertionError("unreachable task-cache retry state")

        try:
            task_path = download_with_transport_retry(overwrite=False)
        except PinnedTaskVerificationError:
            # A process may have died during extraction or an old cache entry
            # may not match the pool's exact ref.  Only integrity failures
            # trigger destructive rematerialization; transport/registry errors
            # are preserved rather than obscured by a duplicate retry.
            task_path = download_with_transport_retry(overwrite=True)
        _atomic_write_json(
            receipt_path,
            {
                **expected,
                "verified_content_hash": task_ref,
                "task_path": str(task_path),
                "completed_at": _utc_now(),
            },
        )
        return task_path
    finally:
        _close_lock(lock)


def _harbor_trial_results(claim: ClaimedCell) -> list[tuple[Path, dict[str, Any]]]:
    job = claim.paths.harbor_job
    if not job.is_dir() or job.is_symlink():
        return []
    results: list[tuple[Path, dict[str, Any]]] = []
    for child in sorted(job.iterdir(), key=lambda path: path.name):
        if child.is_symlink() or not child.is_dir():
            continue
        result_path = child / "result.json"
        if not result_path.is_file():
            continue
        value = _read_json(result_path)
        try:
            TrialResult.model_validate(value)
        except Exception as exc:
            raise PoolError(f"invalid Harbor TrialResult: {result_path}") from exc
        results.append((child, value))
    return results


def _validate_harbor_trial(
    spec: PoolSpec, claim: ClaimedCell, trial: Mapping[str, Any]
) -> None:
    if trial.get("task_name") != claim.cell.task_name:
        raise PoolError("Harbor result task name does not match pool cell")
    if (
        trial.get("source") != claim.cell.source
        or _parse_time(trial.get("finished_at")) is None
    ):
        raise PoolError("Harbor result source or terminal timestamp is invalid")
    task_id = trial.get("task_id")
    result_ref = task_id.get("ref") if isinstance(task_id, Mapping) else None
    config = trial.get("config")
    config_task = config.get("task") if isinstance(config, Mapping) else None
    config_ref = config_task.get("ref") if isinstance(config_task, Mapping) else None
    if result_ref != claim.cell.task_ref or config_ref != claim.cell.task_ref:
        raise PoolError("Harbor result task ref does not match pinned pool cell")
    base_agents = spec.config.get("agents")
    expected_agent = (
        base_agents[0]
        if isinstance(base_agents, list)
        and base_agents
        and isinstance(base_agents[0], Mapping)
        else {}
    )
    trial_agent = config.get("agent") if isinstance(config, Mapping) else None
    if not isinstance(trial_agent, Mapping) or (
        trial_agent.get("import_path") != expected_agent.get("import_path")
        or trial_agent.get("model_name") != expected_agent.get("model_name")
    ):
        raise PoolError("Harbor result agent/model does not match pool contract")


def _load_validated_external_trial(
    spec: PoolSpec,
    cell: CellSpec,
    source_trial_dir: Path,
) -> tuple[Path, dict[str, Any], str]:
    source = _resolve_external_trial_dir(spec, source_trial_dir)
    result_path = source / "result.json"
    if result_path.is_symlink() or not result_path.is_file():
        raise PoolError("external Harbor trial is not terminal yet")
    trial = _read_json(result_path)
    try:
        trial = TrialResult.model_validate(trial).model_dump(
            mode="json", exclude_none=True
        )
    except Exception as exc:
        raise PoolError(f"invalid external Harbor TrialResult: {result_path}") from exc
    synthetic = ClaimedCell(
        cell=cell,
        paths=spec.cell_paths(cell),
        descriptor=-1,
        execution_attempt=1,
        dispatch={},
    )
    _validate_harbor_trial(spec, synthetic, trial)
    try:
        result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PoolError("could not hash external Harbor result") from exc
    return source, trial, result_sha256


def _external_trial_infrastructure_failure(
    trial: Mapping[str, Any],
) -> dict[str, str] | None:
    for classifier in (
        _retryable_sglang_trial_failure,
        _proot_rpc_trial_failure,
        _external_interruption_trial_failure,
    ):
        failure = classifier(trial)
        if failure is not None:
            return {
                "classifier": classifier.__name__,
                **failure,
            }
    return None


def _external_trial_agent_time_budget(trial: Mapping[str, Any]) -> dict[str, Any]:
    config = trial.get("config")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    configured = (
        agent.get("override_timeout_sec") if isinstance(agent, Mapping) else None
    )
    return {
        "policy": "external_harbor_trial_config_v1",
        "configured_harbor_timeout_seconds": configured,
        "effective_total_agent_timeout_seconds": configured,
        "effective_timeout_source": "imported_trial_config",
        "imported_external_execution": True,
    }


def validate_external_harbor_trial(
    *,
    job_name: str,
    task_name: str,
    source_trial_dir: Path,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    official_attempt_index: int = 1,
) -> dict[str, Any]:
    """Validate a terminal direct-job result against one immutable pool cell."""

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    cell = _select_cell(
        spec,
        task_name=task_name,
        official_attempt_index=official_attempt_index,
    )
    source, trial, result_sha256 = _load_validated_external_trial(
        spec, cell, source_trial_dir
    )
    rewards = _result_rewards(trial)
    return {
        "status": "terminal_validated",
        "job_name": job_name,
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "source_trial_dir": str(source),
        "source_result_sha256": result_sha256,
        "reward": rewards.get("reward"),
        "infrastructure_failure": _external_trial_infrastructure_failure(trial),
        "finished_at": trial.get("finished_at"),
    }


def _assert_regular_snapshot_tree(source: Path) -> None:
    """Reject links/devices so an import cannot escape its staging directory."""

    for directory, names, filenames in os.walk(source, followlinks=False):
        root = Path(directory)
        if root.is_symlink() or not root.is_dir():
            raise PoolError(f"unsafe directory in external trial: {root}")
        for name in [*names, *filenames]:
            path = root / name
            if path.is_symlink():
                raise PoolError(f"external trial contains a symlink: {path}")
            if path.is_dir() or path.is_file():
                continue
            raise PoolError(f"external trial contains a special file: {path}")


def _make_external_trial_snapshot(
    spec: PoolSpec,
    claim: ClaimedCell,
    *,
    source: Path,
    source_result_sha256: str,
    reservation: Mapping[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    """Copy and atomically publish one direct trial as a one-trial Harbor job."""

    destination = claim.paths.harbor_job
    if destination.is_dir() and not destination.is_symlink():
        receipt_path = destination / "external-import.json"
        if not receipt_path.is_file():
            raise PoolError("cell Harbor job already exists and is not an import")
        receipt = _read_json(receipt_path, max_bytes=2 * 1024 * 1024)
        if (
            receipt.get("schema") != EXTERNAL_TRIAL_IMPORT_RECEIPT_SCHEMA
            or receipt.get("cell_id") != claim.cell.cell_id
            or receipt.get("source_trial_dir") != str(source)
            or receipt.get("source_result_sha256") != source_result_sha256
        ):
            raise PoolError("existing external Harbor import receipt does not match")
        copied_name = receipt.get("copied_trial_name")
        if not isinstance(copied_name, str) or not copied_name:
            raise PoolError("external Harbor import receipt has no copied trial")
        copied = destination / copied_name
        copied_trial = _read_json(copied / "result.json")
        try:
            copied_trial = TrialResult.model_validate(copied_trial).model_dump(
                mode="json", exclude_none=True
            )
        except Exception as exc:
            raise PoolError("existing external Harbor import is invalid") from exc
        _validate_harbor_trial(spec, claim, copied_trial)
        return copied, copied_trial
    if destination.exists() or destination.is_symlink():
        raise PoolError("cell Harbor job path exists but is not a safe directory")

    _assert_regular_snapshot_tree(source)
    staging = claim.paths.root / f".harbor-job.external-import-{uuid.uuid4().hex}"
    copied = staging / source.name
    receipt: dict[str, Any] = {}
    try:
        _ensure_dir(staging)
        shutil.copytree(source, copied, symlinks=False)
        for directory, _, filenames in os.walk(staging):
            try:
                Path(directory).chmod(PRIVATE_DIR_MODE)
            except OSError:
                pass
            for name in filenames:
                try:
                    (Path(directory) / name).chmod(PRIVATE_FILE_MODE)
                except OSError:
                    pass
        copied_result_path = copied / "result.json"
        copied_result_sha256 = hashlib.sha256(
            copied_result_path.read_bytes()
        ).hexdigest()
        # The terminal commit marker must not change while it is snapshotted.
        current_source_sha256 = hashlib.sha256(
            (source / "result.json").read_bytes()
        ).hexdigest()
        if (
            copied_result_sha256 != source_result_sha256
            or current_source_sha256 != source_result_sha256
        ):
            raise PoolError("external Harbor result changed during import")
        copied_trial = _read_json(copied_result_path)
        try:
            copied_trial = TrialResult.model_validate(copied_trial).model_dump(
                mode="json", exclude_none=True
            )
        except Exception as exc:
            raise PoolError("copied external Harbor TrialResult is invalid") from exc
        _validate_harbor_trial(spec, claim, copied_trial)
        receipt = {
            "schema": EXTERNAL_TRIAL_IMPORT_RECEIPT_SCHEMA,
            "cell_id": claim.cell.cell_id,
            "task_name": claim.cell.task_name,
            "task_ref": claim.cell.task_ref,
            "official_attempt_index": claim.cell.official_attempt_index,
            "config_contract_hash": spec.contract_hash,
            "execution_attempt": claim.execution_attempt,
            "source_job_dir": str(source.parent),
            "source_trial_dir": str(source),
            "source_result_sha256": source_result_sha256,
            "copied_trial_name": source.name,
            "reservation_id": (
                reservation.get("reservation_id")
                if isinstance(reservation, Mapping)
                else None
            ),
            "imported_at": _utc_now(),
        }
        _atomic_write_json(staging / "external-import.json", receipt)
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return destination / source.name, copied_trial
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def import_external_harbor_trial(
    *,
    job_name: str,
    task_name: str,
    source_trial_dir: Path,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    official_attempt_index: int = 1,
    reservation_id: str | None = None,
    allowed_rewards: Sequence[float] | None = None,
    reject_known_infrastructure: bool = True,
    worker_id: str = "external-harbor-import",
    device_label: str = "preserved-direct-harbor-job",
) -> dict[str, Any]:
    """Commit a pre-existing terminal trial as this pool cell's only attempt."""

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    cell = _select_cell(
        spec,
        task_name=task_name,
        official_attempt_index=official_attempt_index,
    )
    source, trial, source_result_sha256 = _load_validated_external_trial(
        spec, cell, source_trial_dir
    )
    failure = _external_trial_infrastructure_failure(trial)
    if reject_known_infrastructure and failure is not None:
        raise PoolError(
            "external Harbor trial is retryable infrastructure, not a model result: "
            f"{failure.get('classifier')}"
        )
    reward = _result_rewards(trial).get("reward")
    if allowed_rewards is not None and not any(
        reward == float(allowed) for allowed in allowed_rewards
    ):
        raise PoolError(
            f"external Harbor trial reward {reward!r} is outside the import policy"
        )
    paths = spec.cell_paths(cell).ensure()
    descriptor = _open_lock(paths.claim_lock, blocking=False)
    if descriptor is None:
        raise PoolError("cannot import external trial while a worker owns the cell")
    try:
        existing_result = _cell_result(spec, cell)
        if existing_result is not None:
            receipt_path = paths.harbor_job / "external-import.json"
            if receipt_path.is_file():
                receipt = _read_json(receipt_path, max_bytes=2 * 1024 * 1024)
                if (
                    receipt.get("schema") == EXTERNAL_TRIAL_IMPORT_RECEIPT_SCHEMA
                    and receipt.get("source_trial_dir") == str(source)
                    and receipt.get("source_result_sha256") == source_result_sha256
                ):
                    return {
                        "status": "already_imported",
                        "job_name": job_name,
                        "cell_id": cell.cell_id,
                        "task_name": cell.task_name,
                        "cell_result": existing_result,
                    }
            raise PoolError("pool cell is already terminal from a different execution")
        reservation: Mapping[str, Any] | None = None
        if paths.external_reservation.is_file():
            reservation = _read_external_reservation(spec, cell)
            if reservation.get(
                "reservation_kind"
            ) != "external_trial" or reservation.get("source_trial_dir") != str(source):
                raise PoolError("external trial does not match the cell reservation")
            if (
                reservation_id is not None
                and reservation.get("reservation_id") != reservation_id
            ):
                raise PoolError("external reservation id does not match")
        elif reservation_id is not None:
            raise PoolError("external reservation does not exist")
        if reservation is None and not _cell_is_score_eligible(
            cell, _attempt_selection(spec), _task_selection(spec)
        ):
            raise PoolError(
                "cannot import an unreserved cell excluded by pool selection"
            )

        prior_dispatch = _read_json(paths.dispatch) if paths.dispatch.is_file() else {}
        execution = _read_execution(paths, prior_dispatch) + 1
        now = _utc_now()
        dispatch = dict(prior_dispatch)
        dispatch.update(
            {
                "schema": CELL_DISPATCH_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "running",
                "execution_attempt": execution,
                "updated_at": now,
                "harbor_job_path": str(paths.harbor_job.relative_to(spec.paths.root)),
                "claim": {
                    "lease_id": uuid.uuid4().hex,
                    "worker_id": worker_id,
                    "instance_id": "external-import",
                    "device": device_label,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "claimed_at": now,
                    "heartbeat_at": now,
                    "lease_expires_at": _future_time(300),
                    "harbor_job_path": str(
                        paths.harbor_job.relative_to(spec.paths.root)
                    ),
                },
            }
        )
        claim = ClaimedCell(
            cell=cell,
            paths=paths,
            descriptor=descriptor,
            execution_attempt=execution,
            dispatch=dispatch,
        )
        copied_trial_path, copied_trial = _make_external_trial_snapshot(
            spec,
            claim,
            source=source,
            source_result_sha256=source_result_sha256,
            reservation=reservation,
        )
        attempt = _ensure_dir(paths.attempts_dir / f"execution-{execution:04d}")
        run = {
            "schema": RUN_RECORD_SCHEMA,
            "cell_id": cell.cell_id,
            "execution_attempt": execution,
            "status": "exited",
            "action": "external_harbor_trial_import",
            "source_trial_dir": str(source),
            "source_result_sha256": source_result_sha256,
            "agent_time_budget": _external_trial_agent_time_budget(copied_trial),
            "started_at": now,
            "finished_at": _utc_now(),
        }
        _atomic_write_json(attempt / "run.json", run)
        if reservation is not None:
            _atomic_write_json(attempt / "external-reservation.json", reservation)
        _atomic_write_json(paths.dispatch, dispatch)
        result = _cell_result_from_trial(
            spec,
            claim,
            copied_trial_path,
            copied_trial,
            worker_id=worker_id,
            device_label=device_label,
        )
        result["agent_time_budget"] = _external_trial_agent_time_budget(copied_trial)
        result["external_import"] = {
            "schema": EXTERNAL_TRIAL_IMPORT_RECEIPT_SCHEMA,
            "source_trial_dir": str(source),
            "source_result_sha256": source_result_sha256,
            "reservation_id": (
                reservation.get("reservation_id")
                if isinstance(reservation, Mapping)
                else None
            ),
        }
        _commit_cell_result(spec, claim, result)
        if paths.external_reservation.is_file():
            paths.external_reservation.unlink()
            _fsync_directory(paths.root)
        _best_effort_event(
            spec.paths.root,
            {
                "event": "external_trial_imported",
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "source_trial_dir": str(source),
                "source_result_sha256": source_result_sha256,
                "reward": reward,
                "recorded_at": _utc_now(),
            },
        )
    finally:
        _close_lock(descriptor)
    state = aggregate_pool(spec)
    return {
        "status": "imported",
        "job_name": job_name,
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "cell_result": result,
        "pool_state": state,
    }


def _terminal_harbor_trial(
    spec: PoolSpec, claim: ClaimedCell
) -> tuple[Path, dict[str, Any]] | None:
    results = _harbor_trial_results(claim)
    if not results:
        return None
    if len(results) != 1:
        raise PoolError("single-cell Harbor job contains multiple terminal trials")
    trial_path, result = results[0]
    _validate_harbor_trial(spec, claim, result)
    return trial_path, result


def _run_subprocess_with_claim(
    command: Sequence[str],
    *,
    claim_descriptor: int,
    cwd: Path,
    log_path: Path,
) -> int:
    _ensure_dir(log_path.parent)
    interrupted = 0
    interrupted_at: float | None = None
    process: subprocess.Popen[bytes] | None = None
    old_handlers: dict[int, Any] = {}

    def handle(signum: int, _frame: Any) -> None:
        nonlocal interrupted, interrupted_at
        interrupted = signum
        interrupted_at = interrupted_at or time.monotonic()
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle)
    try:
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(claim_descriptor,),
                start_new_session=True,
            )
            if interrupted:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
            while True:
                try:
                    return_code = process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    if (
                        interrupted_at is not None
                        and time.monotonic() - interrupted_at >= 30.0
                    ):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            process.kill()
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    if interrupted:
        raise ExternalInterruption(interrupted)
    return return_code


def _result_rewards(trial: Mapping[str, Any]) -> dict[str, float]:
    verifier = trial.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    if not isinstance(rewards, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in rewards.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def _retryable_sglang_trial_failure(
    trial: Mapping[str, Any],
) -> dict[str, str] | None:
    """Recognize only a typed failure raised at our local serving boundary.

    An exception class or an SGLang-looking message is not attribution
    evidence: model/tool code can raise either.  Automatic retry therefore
    requires the pinned local Terminus adapter, its exact ``_generate_sync``
    traceback frame, and its dedicated wrapper exception.  Older message-only
    artifacts remain terminal and unscoreable; they are never resampled.
    """

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = str(exception.get("exception_type", "")).strip()
    message = str(exception.get("exception_message", "")).strip()
    traceback_text = exception.get("exception_traceback")
    config = trial.get("config")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    import_path = agent.get("import_path") if isinstance(agent, Mapping) else None
    if (
        import_path not in _STRICT_SGLANG_AGENT_IMPORT_PATHS
        or exception_type != "RetryableSGLangInfrastructureError"
        or not isinstance(traceback_text, str)
        or not message.lower().startswith(_RETRYABLE_SGLANG_RUNTIME_MESSAGE_PREFIXES)
        or re.search(
            r'File "[^"\n]*terminus_r2eact_agent_v050\.py", line [0-9]+, '
            r"in _generate_sync\n",
            traceback_text,
        )
        is None
        or not traceback_text.endswith(
            "RetryableSGLangInfrastructureError: " + message + "\n"
        )
    ):
        return None
    return {
        "exception_type": exception_type,
        "message": message,
        "cause": "sglang_transport_failure",
        "infrastructure_kind": "sglang",
        "evidence": "local_terminus_generate_sync_traceback_v1",
    }


def _proot_rpc_trial_failure(
    trial: Mapping[str, Any],
) -> dict[str, str] | None:
    """Recognize only the two local PRoot RPC transport terminal failures.

    The exception strings alone are not sufficient: an agent can print or
    raise arbitrary text.  Repair therefore also requires the serialized
    exception to originate at the local environment's ``_wire_exec`` frame.
    The lost-contact variant must retain its low-level asyncio transport cause.
    """

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type != "RuntimeError"
        or not isinstance(message, str)
        or message not in _PROOT_RPC_FAILURE_MESSAGES
        or not isinstance(traceback_text, str)
    ):
        return None
    if re.search(
        r'File "[^"\n]*proot_environment\.py", line [0-9]+, in _wire_exec\n',
        traceback_text,
    ) is None or not traceback_text.endswith(f"RuntimeError: {message}\n"):
        return None
    cause = _PROOT_RPC_FAILURE_MESSAGES[message]
    if cause == "proot_rpc_transport_lost" and (
        re.search(
            r"(?m)^(?:asyncio\.exceptions\.)?"
            r"(?:IncompleteReadError|TimeoutError)(?::|$)",
            traceback_text,
        )
        is None
    ):
        return None
    return {
        "exception_type": "RuntimeError",
        "message": message,
        "cause": cause,
        "infrastructure_kind": "proot",
        "evidence": "local_proot_wire_traceback_v1",
    }


def _environment_import_path(value: Mapping[str, Any]) -> str | None:
    config = value.get("config")
    environment = config.get("environment") if isinstance(config, Mapping) else None
    import_path = (
        environment.get("import_path") if isinstance(environment, Mapping) else None
    )
    return import_path if isinstance(import_path, str) else None


def _receipt_event_names(value: Mapping[str, Any]) -> frozenset[str] | None:
    raw_events = value.get("events")
    if not isinstance(raw_events, list):
        return None
    names: set[str] = set()
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            return None
        name = raw.get("event")
        recorded_at = raw.get("recorded_at")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(recorded_at, str)
            or _parse_time(recorded_at) is None
        ):
            return None
        names.add(name)
    return frozenset(names)


def _receipt_failure_time(
    value: Mapping[str, Any], names: frozenset[str]
) -> str | None:
    """Return the earliest matching, already schema-validated event time."""

    raw_events = value.get("events")
    if not isinstance(raw_events, list):
        return None
    observed: list[datetime] = []
    for raw in raw_events:
        if not isinstance(raw, Mapping) or raw.get("event") not in names:
            continue
        parsed = _parse_time(raw.get("recorded_at"))
        if parsed is None:
            return None
        observed.append(parsed)
    if not observed:
        return None
    return min(observed).isoformat()


def _trial_exception_fields(trial: Mapping[str, Any]) -> tuple[str, str]:
    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping):
        return "unknown", ""
    return (
        str(exception.get("exception_type") or "unknown"),
        str(exception.get("exception_message") or ""),
    )


def _image_pull_transport_signals(value: str) -> tuple[str, ...]:
    """Return only reviewed, retry-safe registry transport signatures."""

    return tuple(
        name
        for name, pattern in _PROOT_IMAGE_PULL_TRANSPORT_PATTERNS
        if pattern.search(value)
    )


def _trial_stopped_before_agent_and_verifier(trial: Mapping[str, Any]) -> bool:
    """Prove Harbor never entered a model- or verifier-controlled phase."""

    return bool(
        trial.get("verifier_environment_mode") == "shared"
        and trial.get("agent_setup") is None
        and trial.get("agent_execution") is None
        and trial.get("agent_result") is None
        and trial.get("step_results") is None
        and trial.get("verifier") is None
        and trial.get("verifier_result") is None
        and not _result_rewards(trial)
    )


def _validated_local_receipt(
    trial_path: Path,
    trial: Mapping[str, Any],
    *,
    name: str,
    schema: str,
) -> tuple[dict[str, Any], frozenset[str]] | None:
    """Read one host-owned environment receipt using fail-closed provenance.

    The model runs inside the guest rootfs and cannot write the Harbor trial
    directory.  Even so, a receipt is trusted only when its fixed filename,
    schema, session identity, event shape, and containing trial directory all
    agree with the validated Harbor TrialResult.  A malformed optional receipt
    is ignored rather than becoming evidence for an infrastructure retry.
    """

    if _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH:
        return None
    trial_name = trial.get("trial_name")
    if (
        not isinstance(trial_name, str)
        or not trial_name
        or Path(trial_name).name != trial_name
        or Path(trial_path).name != trial_name
        or Path(trial_path).is_symlink()
        or not Path(trial_path).is_dir()
    ):
        return None
    receipt_path = Path(trial_path) / name
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None
    try:
        if receipt_path.stat().st_mode & 0o022:
            return None
        receipt = _read_json(receipt_path, max_bytes=2 * 1024 * 1024)
    except (OSError, PoolError):
        return None
    events = _receipt_event_names(receipt)
    if (
        receipt.get("schema") != schema
        or receipt.get("session_id") != f"{trial_name}__env"
        or _parse_time(receipt.get("updated_at")) is None
        or events is None
    ):
        return None
    return receipt, events


def _legacy_image_pull_transport_trial_failure(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, str] | None:
    """Recognize the pre-receipt Skopeo incident without trusting its text alone.

    The old environment emitted no image-pull receipt, but it did durably emit
    two independent host-owned lifecycle receipts during ``stop``.  A legacy
    retry is allowed only when those receipts prove that no rootfs/RPC/package
    bridge ever became ready, the package-bridge image hash equals the exact
    image in the local Skopeo command, and Skopeo's final fatal record contains
    a reviewed transient transport signature.  Once the dedicated image-pull
    receipt exists, even malformed, this compatibility path is disabled.
    """

    if (
        _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
        or trial.get("source") != TB21_DATASET_SOURCE
        or not _trial_stopped_before_agent_and_verifier(trial)
    ):
        return None
    config = trial.get("config")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    if (
        not isinstance(agent, Mapping)
        or agent.get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
    ):
        return None

    image_receipt_path = Path(trial_path) / PROOT_IMAGE_PULL_RECEIPT_NAME
    if image_receipt_path.exists() or image_receipt_path.is_symlink():
        return None

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type != "RuntimeError"
        or not isinstance(message, str)
        or not message
        or "\r" in message
        or not isinstance(traceback_text, str)
        or not traceback_text.startswith("Traceback (most recent call last):\n")
        or not traceback_text.endswith(f"RuntimeError: {message}\n")
    ):
        return None

    lines = message.splitlines()
    command_match = (
        re.fullmatch(
            r"Host command failed \(1\): /usr/bin/skopeo copy --retry-times 3 "
            r"docker://(?P<image>[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,4095}) "
            r"oci:/tmp/harbor-proot-cache/images/(?P<cache_key>[0-9a-f]{20}):image",
            lines[0],
        )
        if lines
        else None
    )
    log_pattern = re.compile(
        r'^time="(?P<recorded_at>[^"\r\n]+)" '
        r'level=(?P<level>info|warning|error|fatal) msg=".*"$'
    )
    log_matches = [log_pattern.fullmatch(line) for line in lines[1:]]
    if (
        command_match is None
        or not log_matches
        or any(match is None for match in log_matches)
        or log_matches[-1] is None
        or log_matches[-1].group("level") != "fatal"
    ):
        return None
    fatal_line = lines[-1]
    transport_signals = _image_pull_transport_signals(fatal_line)
    if not transport_signals:
        return None
    image = command_match.group("image")
    image_sha256 = hashlib.sha256(image.encode("utf-8")).hexdigest()
    if command_match.group("cache_key") != image_sha256[:20]:
        return None

    required_frames = (
        r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, in run\n',
        r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, in _prepare\n',
        r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, '
        r'in _setup_agent_environment\n',
        r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, '
        r'in _start_agent_environment\n',
        r'File "[^"\n]*/eval/terminal_bench/proot_environment\.py", '
        r'line [0-9]+, in start\n',
        r'File "[^"\n]*/eval/terminal_bench/proot_environment\.py", '
        r'line [0-9]+, in _run_host\n',
    )
    if any(re.search(frame, traceback_text) is None for frame in required_frames):
        return None

    trial_started_at = _parse_time(trial.get("started_at"))
    trial_finished_at = _parse_time(trial.get("finished_at"))
    exception_at = _parse_time(exception.get("occurred_at"))
    environment_setup = trial.get("environment_setup")
    environment_started_at = (
        _parse_time(environment_setup.get("started_at"))
        if isinstance(environment_setup, Mapping)
        else None
    )
    environment_finished_at = (
        _parse_time(environment_setup.get("finished_at"))
        if isinstance(environment_setup, Mapping)
        else None
    )
    if (
        trial_started_at is None
        or environment_started_at is None
        or environment_finished_at is None
        or exception_at is None
        or trial_finished_at is None
        or not (
            trial_started_at
            <= environment_started_at
            <= environment_finished_at
            <= exception_at
            <= trial_finished_at
        )
    ):
        return None

    bridge_validated = _validated_local_receipt(
        trial_path,
        trial,
        name=PROOT_PACKAGE_BRIDGE_RECEIPT_NAME,
        schema=PROOT_PACKAGE_BRIDGE_RECEIPT_SCHEMA,
    )
    runtime_validated = _validated_local_receipt(
        trial_path,
        trial,
        name=PROOT_RUNTIME_RECEIPT_NAME,
        schema=PROOT_RUNTIME_RECEIPT_SCHEMA,
    )
    if bridge_validated is None or runtime_validated is None:
        return None
    bridge, _bridge_event_names = bridge_validated
    runtime, _runtime_event_names = runtime_validated
    bridge_events = bridge.get("events")
    runtime_events = runtime.get("events")
    if (
        bridge.get("enabled") is not True
        or bridge.get("policy") != PROOT_PACKAGE_BRIDGE_POLICY
        or bridge.get("failure_policy") != "closed"
        or bridge.get("status") != "stopped"
        or bridge.get("ready") is not False
        or bridge.get("healthy") is not True
        or bridge.get("cleanup_complete") is not True
        or bridge.get("rpc_cleanup_complete") is not True
        or bridge.get("guest_socket_path")
        != "/run/harbor-package-bridge/control.sock"
        or bridge.get("bound_commands") != []
        or bridge.get("broker_events") != []
        or bridge.get("fatal_error") is not None
        or bridge.get("wrapper_sha256") is not None
        or bridge.get("launcher_sha256") is not None
        or bridge.get("oci_manifest_digest") is not None
        or bridge.get("source_image_sha256") != image_sha256
        or bridge.get("stdin_policy") != "non_tty_bounded_until_eof_tty_empty"
        or not isinstance(bridge_events, list)
        or [event.get("event") for event in bridge_events if isinstance(event, Mapping)]
        != [
            "broker_health_check_failed",
            "broker_health_check_failed",
            "environment_stop_finalized",
        ]
        or any(
            not isinstance(event, Mapping)
            or (
                index < 2
                and event.get("error_type") != "PackageBridgeError"
            )
            or (index == 2 and set(event) != {"event", "recorded_at"})
            for index, event in enumerate(bridge_events)
        )
        or runtime.get("rpc_pid") is not None
        or runtime.get("rpc_returncode") is not None
        or runtime.get("rpc_signal") is not None
        or runtime.get("rpc_poisoned") is not True
        or runtime.get("rpc_poison_reason") != "environment_stopped"
        or runtime.get("cleanup_incomplete") is not False
        or runtime.get("rpc_cleanup_complete") is not True
        or runtime.get("cleanup_complete") is not True
        or runtime.get("server_log") is not None
        or not isinstance(runtime_events, list)
        or [event.get("event") for event in runtime_events if isinstance(event, Mapping)]
        != ["environment_stop_without_rpc", "environment_stop_finalized"]
        or any(
            not isinstance(event, Mapping)
            or set(event) != {"event", "recorded_at"}
            for event in runtime_events
        )
    ):
        return None

    lifecycle_times: list[datetime] = []
    for event in [*bridge_events, *runtime_events]:
        recorded_at = _parse_time(event.get("recorded_at"))
        if recorded_at is None:
            return None
        lifecycle_times.append(recorded_at)
    bridge_updated_at = _parse_time(bridge.get("updated_at"))
    runtime_updated_at = _parse_time(runtime.get("updated_at"))
    if (
        bridge_updated_at is None
        or runtime_updated_at is None
        or bridge_updated_at
        != _parse_time(bridge_events[-1].get("recorded_at"))
        or runtime_updated_at
        != _parse_time(runtime_events[-1].get("recorded_at"))
        or any(
            recorded_at < exception_at or recorded_at > trial_finished_at
            for recorded_at in lifecycle_times
        )
    ):
        return None

    return {
        "exception_type": "RuntimeError",
        "message": message,
        "cause": "proot_image_pull_transport_failure",
        "receipt": PROOT_PACKAGE_BRIDGE_RECEIPT_NAME,
        "failure_recorded_at": exception_at.isoformat(),
        "transport_signals": ",".join(transport_signals),
        "infrastructure_kind": "proot",
        "evidence": "host_lifecycle_receipts_and_local_skopeo_traceback_v1",
    }


def _image_pull_receipt_failure(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, str] | None:
    """Classify only a fully exhausted, receipt-bound Skopeo transport pull."""

    if not _trial_stopped_before_agent_and_verifier(trial):
        return None
    validated = _validated_local_receipt(
        trial_path,
        trial,
        name=PROOT_IMAGE_PULL_RECEIPT_NAME,
        schema=PROOT_IMAGE_PULL_RECEIPT_SCHEMA,
    )
    if validated is None:
        return None
    receipt, _event_names = validated
    attempts = receipt.get("attempts")
    events = receipt.get("events")
    source_image_sha256 = receipt.get("source_image_sha256")
    pull_image_sha256 = receipt.get("pull_image_sha256")
    transport_signals = receipt.get("transport_signals")
    started_at = _parse_time(receipt.get("started_at"))
    finished_at = _parse_time(receipt.get("finished_at"))
    exception = trial.get("exception_info")
    exception_message = (
        exception.get("exception_message") if isinstance(exception, Mapping) else None
    )
    exception_at = (
        _parse_time(exception.get("occurred_at"))
        if isinstance(exception, Mapping)
        else None
    )
    trial_started_at = _parse_time(trial.get("started_at"))
    trial_finished_at = _parse_time(trial.get("finished_at"))
    environment_setup = trial.get("environment_setup")
    environment_started_at = (
        _parse_time(environment_setup.get("started_at"))
        if isinstance(environment_setup, Mapping)
        else None
    )
    environment_finished_at = (
        _parse_time(environment_setup.get("finished_at"))
        if isinstance(environment_setup, Mapping)
        else None
    )
    allowed_signals = {name for name, _pattern in _PROOT_IMAGE_PULL_TRANSPORT_PATTERNS}
    if (
        receipt.get("schema_version") != 1
        or receipt.get("policy") != PROOT_IMAGE_PULL_POLICY
        or receipt.get("failure_policy") != "closed"
        or receipt.get("status") != "failed"
        or receipt.get("stage") != "skopeo_copy"
        or receipt.get("max_attempts") != PROOT_IMAGE_PULL_ATTEMPTS
        or receipt.get("backoff_seconds")
        != list(PROOT_IMAGE_PULL_BACKOFF_SECONDS)
        or receipt.get("skopeo_inner_retry_times")
        != PROOT_IMAGE_PULL_SKOPEO_RETRY_TIMES
        or receipt.get("partial_oci_reuse") is not True
        or not isinstance(receipt.get("partial_oci_present_at_start"), bool)
        or receipt.get("last_attempt") != PROOT_IMAGE_PULL_ATTEMPTS
        or receipt.get("transport_failure") is not True
        or receipt.get("attempts_exhausted") is not True
        or receipt.get("final_error_type") != "RuntimeError"
        or not isinstance(source_image_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", source_image_sha256) is None
        or not isinstance(pull_image_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", pull_image_sha256) is None
        or receipt.get("image_cache_key") != pull_image_sha256[:20]
        or not isinstance(transport_signals, list)
        or not transport_signals
        or len(transport_signals) != len(set(transport_signals))
        or any(signal not in allowed_signals for signal in transport_signals)
        or not isinstance(exception_message, str)
        or receipt.get("final_error_sha256")
        != hashlib.sha256(exception_message.encode("utf-8")).hexdigest()
        or not isinstance(attempts, list)
        or len(attempts) != PROOT_IMAGE_PULL_ATTEMPTS
        or not isinstance(events, list)
        or [event.get("event") for event in events if isinstance(event, Mapping)]
        != [
            "image_pull_started",
            "image_pull_attempt_failed",
            "image_pull_attempt_failed",
            "image_pull_failed",
        ]
        or any(not isinstance(event, Mapping) for event in events)
        or started_at is None
        or finished_at is None
        or exception_at is None
        or trial_started_at is None
        or trial_finished_at is None
        or environment_started_at is None
        or environment_finished_at is None
        or not (
            trial_started_at
            <= environment_started_at
            <= started_at
            <= finished_at
            <= environment_finished_at
            <= exception_at
            <= trial_finished_at
        )
    ):
        return None

    for index, attempt in enumerate(attempts, start=1):
        attempt_started_at = (
            _parse_time(attempt.get("started_at"))
            if isinstance(attempt, Mapping)
            else None
        )
        attempt_finished_at = (
            _parse_time(attempt.get("finished_at"))
            if isinstance(attempt, Mapping)
            else None
        )
        attempt_signals = (
            attempt.get("transport_signals")
            if isinstance(attempt, Mapping)
            else None
        )
        retry_scheduled = index < PROOT_IMAGE_PULL_ATTEMPTS
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("attempt") != index
            or attempt.get("max_attempts") != PROOT_IMAGE_PULL_ATTEMPTS
            or attempt.get("status") != "failed"
            or attempt.get("error_type") != "RuntimeError"
            or re.fullmatch(r"[0-9a-f]{64}", str(attempt.get("error_sha256") or ""))
            is None
            or attempt.get("transient_transport_error") is not True
            or not isinstance(attempt_signals, list)
            or not attempt_signals
            or len(attempt_signals) != len(set(attempt_signals))
            or any(signal not in allowed_signals for signal in attempt_signals)
            or attempt.get("retry_scheduled") is not retry_scheduled
            or attempt_started_at is None
            or attempt_finished_at is None
            or not (
                started_at
                <= attempt_started_at
                <= attempt_finished_at
                <= finished_at
            )
            or (
                retry_scheduled
                and (
                    attempt.get("backoff_seconds")
                    != PROOT_IMAGE_PULL_BACKOFF_SECONDS[index - 1]
                    or attempt.get("next_attempt") != index + 1
                )
            )
            or (
                not retry_scheduled
                and (
                    "backoff_seconds" in attempt or "next_attempt" in attempt
                )
            )
        ):
            return None
    if attempts[-1].get("error_sha256") != receipt.get("final_error_sha256"):
        return None

    event_times = [_parse_time(event.get("recorded_at")) for event in events]
    if (
        any(recorded_at is None for recorded_at in event_times)
        or event_times[0] != started_at
        or event_times[-1] != finished_at
        or any(
            earlier > later
            for earlier, later in zip(event_times, event_times[1:], strict=False)
        )
        or events[1].get("attempt") != 1
        or events[2].get("attempt") != 2
        or events[3].get("attempt") != 3
    ):
        return None

    return {
        "exception_type": "RuntimeError",
        "message": exception_message,
        "cause": "proot_image_pull_transport_failure",
        "receipt": PROOT_IMAGE_PULL_RECEIPT_NAME,
        "failure_recorded_at": finished_at.isoformat(),
        "transport_signals": ",".join(transport_signals),
    }


def _runtime_receipt_failure(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, str] | None:
    validated = _validated_local_receipt(
        trial_path,
        trial,
        name=PROOT_RUNTIME_RECEIPT_NAME,
        schema=PROOT_RUNTIME_RECEIPT_SCHEMA,
    )
    if validated is None:
        return None
    receipt, events = validated
    exception_type, exception_message = _trial_exception_fields(trial)
    reason = receipt.get("rpc_poison_reason")
    failure_event_names = events & _PROOT_RUNTIME_FAILURE_EVENTS
    failure_events = sorted(failure_event_names)
    explicit_incomplete = (
        receipt.get("cleanup_incomplete") is True
        or ("cleanup_complete" in receipt and receipt.get("cleanup_complete") is False)
        or (
            "rpc_cleanup_complete" in receipt
            and receipt.get("rpc_cleanup_complete") is False
        )
    )
    if (
        not explicit_incomplete
        and reason not in _PROOT_RUNTIME_FAILURE_REASONS
        and not failure_events
    ):
        return None
    # A mutable status field without a corresponding append-only diagnostic
    # event is not enough to authorize another official model draw.
    failure_recorded_at = _receipt_failure_time(receipt, failure_event_names)
    if failure_recorded_at is None:
        return None
    cause = (
        "proot_runtime_cleanup_incomplete"
        if explicit_incomplete or bool(events & _PROOT_CLEANUP_FAILURE_EVENTS)
        else "proot_rpc_runtime_failure"
    )
    return {
        "exception_type": exception_type,
        "message": exception_message,
        "cause": cause,
        "receipt": PROOT_RUNTIME_RECEIPT_NAME,
        "failure_recorded_at": failure_recorded_at,
    }


def _package_bridge_receipt_failure(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, str] | None:
    validated = _validated_local_receipt(
        trial_path,
        trial,
        name=PROOT_PACKAGE_BRIDGE_RECEIPT_NAME,
        schema=PROOT_PACKAGE_BRIDGE_RECEIPT_SCHEMA,
    )
    if validated is None:
        return None
    receipt, events = validated
    exception_type, exception_message = _trial_exception_fields(trial)
    status = receipt.get("status")
    bound_commands = receipt.get("bound_commands")
    if (
        receipt.get("enabled") is not True
        or receipt.get("policy") != PROOT_PACKAGE_BRIDGE_POLICY
        or receipt.get("failure_policy") != "closed"
        or not isinstance(status, str)
        or not isinstance(receipt.get("ready"), bool)
        or not isinstance(receipt.get("cleanup_complete"), bool)
        or not isinstance(receipt.get("rpc_cleanup_complete"), bool)
        or receipt.get("guest_socket_path") != "/run/harbor-package-bridge/control.sock"
        or not isinstance(bound_commands, list)
        or any(
            not isinstance(command, str) or not command for command in bound_commands
        )
        or len(bound_commands) != len(set(bound_commands))
    ):
        return None
    failure_event_names = events & _PACKAGE_BRIDGE_FAILURE_EVENTS
    failure_events = failure_event_names
    cleanup_incomplete = (
        receipt.get("cleanup_complete") is False
        or status in _PACKAGE_BRIDGE_CLEANUP_STATUSES
        or bool(events & _PACKAGE_BRIDGE_CLEANUP_EVENTS)
    )
    if status not in _PACKAGE_BRIDGE_FAILURE_STATUSES and not failure_events:
        return None
    failure_recorded_at = _receipt_failure_time(receipt, failure_event_names)
    if failure_recorded_at is None:
        return None
    return {
        "exception_type": exception_type,
        "message": exception_message,
        "cause": (
            "proot_package_bridge_cleanup_incomplete"
            if cleanup_incomplete
            else "proot_package_bridge_failure"
        ),
        "receipt": PROOT_PACKAGE_BRIDGE_RECEIPT_NAME,
        "failure_recorded_at": failure_recorded_at,
    }


def _guest_utility_environment_start_timeout_receipt_failure(
    trial: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    receipt_name: str,
    environment_role: str,
    session_id: str,
    environment_kwargs: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Bind Harbor's outer start timeout to one in-flight legacy APT stage.

    A bare ``CancelledError`` is not evidence: Harbor uses cancellation in
    several normal shutdown paths, and model output can contain arbitrary
    exception text.  This legacy repair path accepts only the host-owned v2
    guest-utility receipt left while Harbor cancelled the *main* environment's
    first APT attempt at its fixed 600-second start boundary.  The matching
    local traceback is checked separately by
    ``_exception_matches_local_receipt_failure``.
    """

    stage = receipt.get("stage")
    command = (
        _PROOT_GUEST_UTILITY_APT_COMMANDS.get(stage)
        if isinstance(stage, str)
        else None
    )
    attempts = receipt.get("install_attempts")
    distro = receipt.get("distro")
    manifest_digest = receipt.get("oci_manifest_digest")
    source_image = receipt.get("source_image")
    started_at = _parse_time(receipt.get("started_at"))
    command_started_at = _parse_time(receipt.get("last_command_started_at"))
    finished_at = _parse_time(receipt.get("finished_at"))
    updated_at = _parse_time(receipt.get("updated_at"))
    trial_started_at = _parse_time(trial.get("started_at"))
    trial_finished_at = _parse_time(trial.get("finished_at"))
    exception = trial.get("exception_info")
    exception_at = (
        _parse_time(exception.get("occurred_at"))
        if isinstance(exception, Mapping)
        else None
    )
    environment_setup = trial.get("environment_setup")
    environment_setup_started_at = (
        _parse_time(environment_setup.get("started_at"))
        if isinstance(environment_setup, Mapping)
        else None
    )
    environment_setup_finished_at = (
        _parse_time(environment_setup.get("finished_at"))
        if isinstance(environment_setup, Mapping)
        else None
    )
    if (
        environment_role != "main"
        or trial.get("verifier_environment_mode") != "shared"
        or trial.get("agent_setup") is not None
        or trial.get("agent_execution") is not None
        or trial.get("agent_result") is not None
        or trial.get("step_results") is not None
        or trial.get("verifier") is not None
        or trial.get("verifier_result") is not None
        or _result_rewards(trial)
        or receipt.get("schema_version") != 2
        or receipt.get("schema") != PROOT_GUEST_UTILITY_RECEIPT_SCHEMA
        or receipt.get("policy") != PROOT_GUEST_UTILITY_POLICY
        or receipt.get("profile") != PROOT_GUEST_UTILITY_PROFILE
        or receipt.get("session_id") != session_id
        or receipt.get("status") != "failed"
        or command is None
        or receipt.get("last_command") != command
        or receipt.get("last_command_attempt") != 1
        or receipt.get("error_type") != "CancelledError"
        or receipt.get("error") != ""
        or receipt.get("failure_policy") != "closed"
        or receipt.get("install_transport") != "direct_chroot_compatibility"
        or receipt.get("verification_transport") != "long_lived_proot_rpc"
        or receipt.get("benchmark_environment_delta") is not True
        or receipt.get("package_versions_unpinned") is not True
        or receipt.get("requested_commands") != ["file", "xxd", "rg"]
        or receipt.get("apt_packages") != ["file", "xxd", "ripgrep"]
        or not isinstance(environment_kwargs, Mapping)
        or environment_kwargs.get("guest_utility_profile")
        != PROOT_GUEST_UTILITY_PROFILE
        or not isinstance(distro, Mapping)
        or distro.get("id") not in {"debian", "ubuntu"}
        or receipt.get("architecture") != "amd64"
        or not isinstance(manifest_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", manifest_digest) is None
        or not isinstance(source_image, str)
        or not source_image
        or len(source_image) > 4096
        or any(ord(character) < 0x20 for character in source_image)
        or receipt.get("apt_retry_policy")
        != {
            "stages": ["apt_update", "apt_install"],
            "max_attempts": PROOT_GUEST_UTILITY_APT_ATTEMPTS,
            "backoff_seconds": list(PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS),
            "inner_acquire_retries": 3,
        }
        or not isinstance(attempts, list)
        or started_at is None
        or command_started_at is None
        or finished_at is None
        or updated_at is None
        or trial_started_at is None
        or trial_finished_at is None
        or exception_at is None
        or environment_setup_started_at is None
        or environment_setup_finished_at is None
        or not (
            trial_started_at
            <= environment_setup_started_at
            <= started_at
            <= command_started_at
            <= finished_at
            <= environment_setup_finished_at
            <= exception_at
            <= trial_finished_at
        )
        or updated_at < finished_at
    ):
        return None

    expected_stage_order = (
        ["architecture_preflight", "apt_update"]
        if stage == "apt_update"
        else ["architecture_preflight", "apt_update", "apt_install"]
    )
    if len(attempts) != len(expected_stage_order):
        return None
    for index, (attempt, expected_stage) in enumerate(
        zip(attempts, expected_stage_order, strict=True)
    ):
        if not isinstance(attempt, Mapping):
            return None
        attempt_started_at = _parse_time(attempt.get("started_at"))
        attempt_finished_at = _parse_time(attempt.get("finished_at"))
        if (
            attempt.get("stage") != expected_stage
            or attempt.get("attempt") != 1
            or attempt_started_at is None
            or attempt_finished_at is None
            or not (started_at <= attempt_started_at <= attempt_finished_at <= finished_at)
            or (index > 0 and attempt_started_at < _parse_time(attempts[index - 1].get("finished_at")))
        ):
            return None
        if expected_stage == "architecture_preflight":
            result = attempt.get("result")
            if (
                attempt.get("max_attempts") != 1
                or attempt.get("command") != "dpkg --print-architecture"
                or attempt.get("status") != "succeeded"
                or not isinstance(result, Mapping)
                or result.get("return_code") != 0
                or str(result.get("stdout_tail") or "").strip() != "amd64"
                or not isinstance(result.get("stderr_tail"), str)
            ):
                return None
        elif expected_stage != stage:
            result = attempt.get("result")
            if (
                expected_stage != "apt_update"
                or attempt.get("max_attempts") != PROOT_GUEST_UTILITY_APT_ATTEMPTS
                or attempt.get("command")
                != _PROOT_GUEST_UTILITY_APT_COMMANDS["apt_update"]
                or attempt.get("status") != "succeeded"
                or not isinstance(result, Mapping)
                or result.get("return_code") != 0
                or not isinstance(result.get("stdout_tail"), str)
                or not isinstance(result.get("stderr_tail"), str)
            ):
                return None
        elif (
            attempt.get("max_attempts") != PROOT_GUEST_UTILITY_APT_ATTEMPTS
            or attempt.get("command") != command
            or attempt.get("status") != "exception"
            or attempt.get("error_type") != "CancelledError"
            or attempt.get("error") != ""
            or attempt.get("result") is not None
            or attempt.get("retry_scheduled") is not None
            or attempt_started_at != command_started_at
        ):
            return None

    prior_result = attempts[-2].get("result")
    if not isinstance(prior_result, Mapping) or receipt.get(
        "last_command_result"
    ) != dict(prior_result):
        return None
    exception_type, exception_message = _trial_exception_fields(trial)
    if (
        exception_type != "EnvironmentStartTimeoutError"
        or exception_message
        != _PROOT_GUEST_UTILITY_ENVIRONMENT_START_TIMEOUT_MESSAGE
    ):
        return None
    return {
        "exception_type": exception_type,
        "message": exception_message,
        "cause": f"proot_guest_utility_{stage}_environment_start_timeout",
        "receipt": receipt_name,
        "environment_role": environment_role,
        "failure_recorded_at": finished_at.isoformat(),
    }


def _guest_utility_apt_output_signals(
    result: Mapping[str, Any],
) -> tuple[str, ...]:
    """Derive reviewed transport signals from one host-owned APT result."""

    stdout = result.get("stdout_tail")
    stderr = result.get("stderr_tail")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return ()
    output = f"{stdout}\n{stderr}"
    return tuple(
        name
        for name, pattern in _PROOT_GUEST_UTILITY_APT_TRANSPORT_PATTERNS
        if pattern.search(output)
    )


def _recorded_apt_transport_matches(recorded: Any, inferred: Sequence[str]) -> bool:
    """Accept legacy receipts missing only the newly recognized HTTP status.

    Preserve the original host receipt; never relabel a plain signature error,
    a 404, or an unknown/missing pre-existing signal as a network failure.
    """

    expected = list(inferred)
    legacy = [signal for signal in expected if signal != "apt_http_transient"]
    return isinstance(recorded, list) and (recorded == expected or recorded == legacy)


def _guest_utility_apt_update_incomplete_signals(
    result: Mapping[str, Any],
) -> tuple[str, ...]:
    """Validate APT's exit-0 partial-index warning from captured host output."""

    stdout = result.get("stdout_tail")
    stderr = result.get("stderr_tail")
    if not isinstance(stdout, str) or not isinstance(stderr, str):
        return ()
    output = f"{stdout}\n{stderr}"
    return tuple(
        name
        for name, pattern in (
            _PROOT_GUEST_UTILITY_APT_UPDATE_INCOMPLETE_PATTERNS
        )
        if pattern.search(output)
    )


def _guest_utility_failed_apt_result_is_valid(
    result: Mapping[str, Any],
    *,
    stage: str,
) -> bool:
    """Accept a real nonzero exit or a receipt-bound exit-0 update failure."""

    return_code = result.get("return_code")
    if (
        not isinstance(return_code, int)
        or isinstance(return_code, bool)
        or not isinstance(result.get("stdout_tail"), str)
        or not isinstance(result.get("stderr_tail"), str)
    ):
        return False
    semantic_failure = result.get("semantic_failure")
    semantic_signals = result.get("semantic_failure_signals")
    if return_code > 0:
        return semantic_failure is None and semantic_signals is None
    if return_code != 0 or stage != "apt_update":
        return False
    inferred = _guest_utility_apt_update_incomplete_signals(result)
    return (
        bool(inferred)
        and semantic_failure
        == _PROOT_GUEST_UTILITY_APT_UPDATE_SEMANTIC_FAILURE
        and semantic_signals == list(inferred)
    )


def _guest_utility_receipt_failure(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, str] | None:
    """Classify an exhausted, host-owned apt provisioning receipt.

    Version 1 is retained solely so already-committed failures can be repaired;
    version 2 additionally proves that both bounded backoffs were attempted.
    Neither version is sufficient without the matching local traceback checked
    by ``_exception_matches_local_receipt_failure``.
    """

    if _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH:
        return None
    trial_name = trial.get("trial_name")
    trial_dir = Path(trial_path)
    if (
        not isinstance(trial_name, str)
        or not trial_name
        or Path(trial_name).name != trial_name
        or trial_dir.name != trial_name
        or trial_dir.is_symlink()
        or not trial_dir.is_dir()
    ):
        return None
    role_sessions = {"main": f"{trial_name}__env"}
    if trial.get("verifier_environment_mode") == "separate":
        role_sessions["separate_verifier"] = f"{trial_name}__verifier__trial"
    expected_receipts: dict[str, tuple[str, str]] = {}
    for role, candidate_session in role_sessions.items():
        safe_session = re.sub(r"[^a-zA-Z0-9_.-]+", "-", candidate_session).strip("-.")
        safe_session = safe_session[:80] or "trial"
        candidate_name = f"{PROOT_GUEST_UTILITY_RECEIPT_STEM}-{safe_session}.json"
        if candidate_name in expected_receipts:
            return None
        expected_receipts[candidate_name] = (role, candidate_session)

    observed_receipts: list[tuple[str, str, str, dict[str, Any]]] = []
    try:
        guest_receipt_paths = sorted(
            path
            for path in trial_dir.iterdir()
            if path.name.startswith(f"{PROOT_GUEST_UTILITY_RECEIPT_STEM}-")
            and path.name.endswith(".json")
        )
    except OSError:
        return None
    for candidate_path in guest_receipt_paths:
        expected = expected_receipts.get(candidate_path.name)
        if (
            expected is None
            or candidate_path.is_symlink()
            or not candidate_path.is_file()
        ):
            return None
        role, candidate_session = expected
        try:
            if candidate_path.stat().st_mode & 0o022:
                return None
            candidate_receipt = _read_json(candidate_path, max_bytes=2 * 1024 * 1024)
        except (OSError, PoolError):
            return None
        if (
            candidate_receipt.get("session_id") != candidate_session
            or candidate_receipt.get("schema_version") not in {1, 2}
            or (
                candidate_receipt.get("schema_version") == 1
                and candidate_receipt.get("schema") is not None
            )
            or (
                candidate_receipt.get("schema_version") == 2
                and candidate_receipt.get("schema")
                != PROOT_GUEST_UTILITY_RECEIPT_SCHEMA
            )
            or candidate_receipt.get("policy") != PROOT_GUEST_UTILITY_POLICY
            or candidate_receipt.get("profile") != PROOT_GUEST_UTILITY_PROFILE
            or candidate_receipt.get("failure_policy") != "closed"
        ):
            return None
        observed_receipts.append(
            (role, candidate_session, candidate_path.name, candidate_receipt)
        )

    failed_receipts = [
        observed
        for observed in observed_receipts
        if observed[3].get("status") == "failed"
    ]
    if len(failed_receipts) != 1:
        return None
    environment_role, session_id, receipt_name, receipt = failed_receipts[0]
    nonfailure_receipts = [
        observed
        for observed in observed_receipts
        if observed[3].get("status") != "failed"
    ]
    if any(
        observed[3].get("status") != "ok"
        or observed[3].get("stage") != "complete"
        or _parse_time(observed[3].get("finished_at")) is None
        for observed in nonfailure_receipts
    ):
        return None
    if environment_role == "separate_verifier":
        if {observed[0] for observed in observed_receipts} != {
            "main",
            "separate_verifier",
        } or trial.get("verifier_environment_mode") != "separate":
            return None
    elif any(observed[0] == "separate_verifier" for observed in observed_receipts):
        return None

    environment = trial.get("config")
    environment = (
        environment.get("environment") if isinstance(environment, Mapping) else None
    )
    environment_kwargs = (
        environment.get("kwargs") if isinstance(environment, Mapping) else None
    )
    environment_start_timeout = (
        _guest_utility_environment_start_timeout_receipt_failure(
            trial,
            receipt,
            receipt_name=receipt_name,
            environment_role=environment_role,
            session_id=session_id,
            environment_kwargs=environment_kwargs,
        )
    )
    if environment_start_timeout is not None:
        return environment_start_timeout
    stage = receipt.get("stage")
    observed_command = receipt.get("last_command")
    accepted_commands = (
        {
            _PROOT_GUEST_UTILITY_APT_COMMANDS[stage],
            _PROOT_GUEST_UTILITY_BOUNDED_APT_COMMANDS[stage],
        }
        if isinstance(stage, str)
        and stage in _PROOT_GUEST_UTILITY_APT_COMMANDS
        and stage in _PROOT_GUEST_UTILITY_BOUNDED_APT_COMMANDS
        else set()
    )
    command = observed_command if observed_command in accepted_commands else None
    command_result = receipt.get("last_command_result")
    distro = receipt.get("distro")
    return_code = (
        command_result.get("return_code")
        if isinstance(command_result, Mapping)
        else None
    )
    stdout_tail = (
        command_result.get("stdout_tail")
        if isinstance(command_result, Mapping)
        else None
    )
    stderr_tail = (
        command_result.get("stderr_tail")
        if isinstance(command_result, Mapping)
        else None
    )
    schema_version = receipt.get("schema_version")
    receipt_attempts = receipt.get("install_attempts")
    transport_attempts = (
        receipt_attempts
        if schema_version == 2 and isinstance(receipt_attempts, list)
        else []
    )
    inferred_transport_signals: list[str] = []
    for candidate_attempt in transport_attempts:
        candidate_result = (
            candidate_attempt.get("result")
            if isinstance(candidate_attempt, Mapping)
            else None
        )
        if not isinstance(candidate_result, Mapping):
            continue
        for signal_name in _guest_utility_apt_output_signals(candidate_result):
            if signal_name not in inferred_transport_signals:
                inferred_transport_signals.append(signal_name)
    if not transport_attempts and isinstance(command_result, Mapping):
        inferred_transport_signals.extend(
            _guest_utility_apt_output_signals(command_result)
        )
    final_transport_signals = (
        _guest_utility_apt_output_signals(command_result)
        if isinstance(command_result, Mapping)
        else ()
    )
    # Before the exit-0 guard existed, APT could report failed indexes as
    # warnings and then fail the immediately following install with an
    # apparently deterministic "Unable to locate package".  Preserve repair
    # compatibility only for that exact host-owned causal chain.
    legacy_partial_update_cause = False
    if (
        stage == "apt_install"
        and isinstance(stdout_tail, str)
        and isinstance(stderr_tail, str)
        and re.search(
            r"\bUnable to locate package\b", f"{stdout_tail}\n{stderr_tail}"
        )
    ):
        legacy_partial_update_cause = any(
            isinstance(candidate_attempt, Mapping)
            and candidate_attempt.get("stage") == "apt_update"
            and isinstance(candidate_attempt.get("result"), Mapping)
            and candidate_attempt["result"].get("return_code") == 0
            and bool(
                _guest_utility_apt_update_incomplete_signals(
                    candidate_attempt["result"]
                )
            )
            for candidate_attempt in transport_attempts
        )
    recorded_transport_failure = receipt.get("transport_failure")
    recorded_transport_signals = receipt.get("transport_signals")
    has_recorded_transport = (
        "transport_failure" in receipt or "transport_signals" in receipt
    )
    started_at = _parse_time(receipt.get("started_at"))
    command_started_at = _parse_time(receipt.get("last_command_started_at"))
    finished_at = _parse_time(receipt.get("finished_at"))
    exception = trial.get("exception_info")
    exception_at = (
        _parse_time(exception.get("occurred_at"))
        if isinstance(exception, Mapping)
        else None
    )
    trial_started_at = _parse_time(trial.get("started_at"))
    trial_finished_at = _parse_time(trial.get("finished_at"))
    # End-of-life metadata is not a network timeout. Admit only the known
    # Debian 11 security expiry, before any agent/verifier work, while retaining
    # the receipt, three-attempt and traceback checks below. This permits an
    # explicit archive/requeue after the snapshot-source repair, not reward retry.
    security_expiry = re.compile(
        r"^E: Release file for https?://(?:deb\.debian\.org/debian-security|"
        r"security\.debian\.org(?:/debian-security)?)/dists/bullseye-security/"
        r"(?:InRelease|Release) is expired \(invalid since [^\n]+\)\. "
        r"Updates for this repository will not be applied\.$", re.M
    )
    agent_timing = trial.get("agent_execution") or {}
    eol_security_expiry = (
        stage == "apt_update" and schema_version == 2
        and isinstance(distro, Mapping) and distro.get("id") == "debian"
        and distro.get("version_id") == "11"
        and environment_role == "main" and trial.get("verifier_result") is None
        and trial.get("agent_result") is None
        and isinstance(agent_timing, Mapping) and not agent_timing.get("started_at")
        and started_at is not None and started_at >= datetime(2026, 9, 1, tzinfo=UTC)
        and isinstance(command_result, Mapping) and command_result.get("return_code") == 100
        and isinstance(stderr_tail, str) and security_expiry.fullmatch(stderr_tail.strip()) is not None
        and all(
            isinstance(item.get("result"), Mapping)
            and item["result"].get("return_code") == 100
            and isinstance(item["result"].get("stderr_tail"), str)
            and security_expiry.fullmatch(item["result"]["stderr_tail"].strip()) is not None
            for item in transport_attempts
            if isinstance(item, Mapping) and item.get("stage") == "apt_update"
        )
    )
    verifier_timing = trial.get("verifier")
    verifier_started_at = (
        _parse_time(verifier_timing.get("started_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    verifier_finished_at = (
        _parse_time(verifier_timing.get("finished_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    oci_manifest_digest = receipt.get("oci_manifest_digest")
    if (
        receipt.get("schema_version") not in {1, 2}
        or receipt.get("policy") != PROOT_GUEST_UTILITY_POLICY
        or receipt.get("profile") != PROOT_GUEST_UTILITY_PROFILE
        or receipt.get("session_id") != session_id
        or receipt.get("status") != "failed"
        or command is None
        or receipt.get("last_command") != command
        or receipt.get("failure_policy") != "closed"
        or receipt.get("install_transport") != "direct_chroot_compatibility"
        or receipt.get("verification_transport") != "long_lived_proot_rpc"
        or receipt.get("benchmark_environment_delta") is not True
        or receipt.get("package_versions_unpinned") is not True
        or receipt.get("requested_commands") != ["file", "xxd", "rg"]
        or receipt.get("apt_packages") != ["file", "xxd", "ripgrep"]
        or not isinstance(environment_kwargs, Mapping)
        or environment_kwargs.get("guest_utility_profile")
        != PROOT_GUEST_UTILITY_PROFILE
        or not isinstance(distro, Mapping)
        or distro.get("id") not in {"debian", "ubuntu"}
        or receipt.get("architecture") != "amd64"
        or not isinstance(oci_manifest_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", oci_manifest_digest) is None
        or not isinstance(receipt.get("source_image"), str)
        or not receipt.get("source_image")
        or len(receipt.get("source_image")) > 4096
        or any(
            ord(character) < 0x20
            for character in receipt.get("source_image")
        )
        or not isinstance(command_result, Mapping)
        or not _guest_utility_failed_apt_result_is_valid(
            command_result, stage=str(stage)
        )
        or not isinstance(stdout_tail, str)
        or not isinstance(stderr_tail, str)
        or (not inferred_transport_signals and not eol_security_expiry)
        or (not final_transport_signals and not legacy_partial_update_cause and not eol_security_expiry)
        or (
            has_recorded_transport
            and (
                recorded_transport_failure is not bool(recorded_transport_signals)
                or not _recorded_apt_transport_matches(
                    recorded_transport_signals, inferred_transport_signals
                )
            )
        )
        or started_at is None
        or command_started_at is None
        or finished_at is None
        or exception_at is None
        or trial_started_at is None
        or trial_finished_at is None
        or trial_started_at > started_at
        or not (started_at <= command_started_at <= finished_at)
        or not (finished_at <= exception_at <= trial_finished_at)
        or (
            environment_role == "separate_verifier"
            and (
                verifier_started_at is None
                or verifier_finished_at is None
                or not (
                    verifier_started_at
                    <= started_at
                    <= finished_at
                    <= verifier_finished_at
                )
            )
        )
    ):
        return None

    detail = (stderr_tail or stdout_tail).strip()
    expected_error = (
        f"PRoot guest utility {stage} failed despite exit status 0: {detail}"
        if return_code == 0
        else (
            f"PRoot guest utility {stage} failed with exit status "
            f"{return_code}: {detail}"
        )
    )
    exception_type, exception_message = _trial_exception_fields(trial)
    if (
        receipt.get("error_type") != "RuntimeError"
        or receipt.get("error") != expected_error
        or exception_type != "RuntimeError"
        or exception_message != expected_error
    ):
        return None

    if schema_version == 1:
        # The original receipt had no named schema or attempt journal.
        if receipt.get("schema") is not None:
            return None
    else:
        updated_at = _parse_time(receipt.get("updated_at"))
        retry_policy = receipt.get("apt_retry_policy")
        attempts = receipt_attempts
        expected_retry_policy = (
            {
                "stages": ["apt_update", "apt_install"],
                "max_attempts": PROOT_GUEST_UTILITY_APT_ATTEMPTS,
                "backoff_seconds": list(PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS),
                "inner_acquire_retries": 0,
                "http_connect_and_read_timeout_seconds": (
                    _PROOT_GUEST_UTILITY_BOUNDED_APT_NETWORK_TIMEOUT_SEC
                ),
                "https_connect_and_read_timeout_seconds": (
                    _PROOT_GUEST_UTILITY_BOUNDED_APT_NETWORK_TIMEOUT_SEC
                ),
                "per_attempt_timeout_seconds": (
                    _PROOT_GUEST_UTILITY_BOUNDED_APT_ATTEMPT_TIMEOUT_SEC
                ),
            }
            if command == _PROOT_GUEST_UTILITY_BOUNDED_APT_COMMANDS.get(stage)
            else {
                "stages": ["apt_update", "apt_install"],
                "max_attempts": PROOT_GUEST_UTILITY_APT_ATTEMPTS,
                "backoff_seconds": list(PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS),
                "inner_acquire_retries": 3,
            }
        )
        if (
            receipt.get("schema") != PROOT_GUEST_UTILITY_RECEIPT_SCHEMA
            or updated_at is None
            or updated_at < finished_at
            or retry_policy != expected_retry_policy
            or not isinstance(attempts, list)
        ):
            return None
        stage_attempts = [
            attempt
            for attempt in attempts
            if isinstance(attempt, Mapping) and attempt.get("stage") == stage
        ]
        if len(stage_attempts) != PROOT_GUEST_UTILITY_APT_ATTEMPTS:
            return None
        for index, attempt in enumerate(stage_attempts, start=1):
            attempt_result = attempt.get("result")
            attempt_started_at = _parse_time(attempt.get("started_at"))
            attempt_finished_at = _parse_time(attempt.get("finished_at"))
            attempt_return_code = (
                attempt_result.get("return_code")
                if isinstance(attempt_result, Mapping)
                else None
            )
            expected_retry = index < PROOT_GUEST_UTILITY_APT_ATTEMPTS
            if (
                attempt.get("attempt") != index
                or attempt.get("max_attempts") != PROOT_GUEST_UTILITY_APT_ATTEMPTS
                or attempt.get("command") != command
                or (
                    command
                    == _PROOT_GUEST_UTILITY_BOUNDED_APT_COMMANDS.get(stage)
                    and attempt.get("timeout_seconds")
                    != _PROOT_GUEST_UTILITY_BOUNDED_APT_ATTEMPT_TIMEOUT_SEC
                )
                or attempt.get("status") != "failed"
                or not isinstance(attempt_result, Mapping)
                or not _guest_utility_failed_apt_result_is_valid(
                    attempt_result, stage=str(stage)
                )
                or not isinstance(attempt_result.get("stdout_tail"), str)
                or not isinstance(attempt_result.get("stderr_tail"), str)
                or (
                    "transport_signals" in attempt_result
                    and not _recorded_apt_transport_matches(
                        attempt_result.get("transport_signals"),
                        _guest_utility_apt_output_signals(attempt_result),
                    )
                )
                or attempt_started_at is None
                or attempt_finished_at is None
                or not (
                    started_at
                    <= attempt_started_at
                    <= attempt_finished_at
                    <= finished_at
                )
                or attempt.get("retry_scheduled") is not expected_retry
            ):
                return None
            if expected_retry and (
                attempt.get("backoff_seconds")
                != PROOT_GUEST_UTILITY_APT_BACKOFF_SECONDS[index - 1]
                or attempt.get("next_attempt") != index + 1
            ):
                return None
        if receipt.get(
            "last_command_attempt"
        ) != PROOT_GUEST_UTILITY_APT_ATTEMPTS or dict(
            stage_attempts[-1]["result"]
        ) != dict(
            command_result
        ):
            return None

    return {
        "exception_type": exception_type,
        "message": exception_message,
        "cause": f"proot_guest_utility_{stage}_failure",
        "receipt": receipt_name,
        "environment_role": environment_role,
        "failure_recorded_at": finished_at.isoformat(),
        "transport_signals": ",".join(inferred_transport_signals),
        **({"repository_failure": "bullseye_security_eol_expired_release"} if eol_security_expiry else {}),
    }


def _exception_matches_local_receipt_failure(
    trial: Mapping[str, Any], failure: Mapping[str, Any]
) -> bool:
    """Bind a trusted receipt to the exception that made the trial terminal."""

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return False
    traceback_text = exception.get("exception_traceback")
    exception_type = str(exception.get("exception_type") or "")
    if not isinstance(traceback_text, str):
        return False
    cause = failure.get("cause")
    if cause == "proot_image_pull_transport_failure":
        message = exception.get("exception_message")
        return (
            exception_type == "RuntimeError"
            and isinstance(message, str)
            and message == failure.get("message")
            and traceback_text.endswith(f"RuntimeError: {message}\n")
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in start\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in _pull_image_with_retry\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in _run_host\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, '
                r"in _setup_agent_environment\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, '
                r"in _start_agent_environment\n",
                traceback_text,
            )
            is not None
        )
    if cause in {
        "proot_runtime_cleanup_incomplete",
        "proot_package_bridge_cleanup_incomplete",
    }:
        return _exception_is_local_cleanup_only(
            trial
        ) or _exception_is_local_proot_runtime(trial)
    if cause == "proot_package_bridge_failure":
        return exception_type == "PackageBridgeError" and (
            re.search(
                r'File "[^"\n]*(?:proot_environment|proot_package_bridge)\.py", '
                r"line [0-9]+, in [^\n]+\n",
                traceback_text,
            )
            is not None
        )
    if cause == "proot_rpc_runtime_failure":
        return exception_type in {"RuntimeError", "PackageBridgeError"} and (
            re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in (?:_wire_exec|exec|start|_start_rpc_server)\n",
                traceback_text,
            )
            is not None
        )
    if cause in {
        "proot_guest_utility_apt_update_environment_start_timeout",
        "proot_guest_utility_apt_install_environment_start_timeout",
    }:
        message = exception.get("exception_message")
        return (
            exception_type == "EnvironmentStartTimeoutError"
            and message == _PROOT_GUEST_UTILITY_ENVIRONMENT_START_TIMEOUT_MESSAGE
            and message == failure.get("message")
            and traceback_text.endswith(
                "harbor.trial.errors.EnvironmentStartTimeoutError: "
                f"{message}\n"
            )
            and "asyncio.exceptions.CancelledError\n" in traceback_text
            and "The above exception was the direct cause of the following "
            "exception:\n" in traceback_text
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in start\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in _install_guest_utilities\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in _run_guest_utility_install_stage\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in _direct_exec\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, '
                r"in _start_agent_environment\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, '
                r"in _setup_agent_environment\n",
                traceback_text,
            )
            is not None
            and "raise EnvironmentStartTimeoutError(" in traceback_text
        )
    if cause in {
        "proot_guest_utility_apt_update_failure",
        "proot_guest_utility_apt_install_failure",
    }:
        message = exception.get("exception_message")
        environment_role = failure.get("environment_role")
        separate_verifier_trace = (
            re.search(
                r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, '
                r"in _separate_verifier_env\n",
                traceback_text,
            )
            is not None
        )
        separate_verifier_run_trace = (
            re.search(
                r'File "[^"\n]*/harbor/trial/trial\.py", line [0-9]+, '
                r"in _run_separate_verifier\n",
                traceback_text,
            )
            is not None
        )
        return (
            exception_type == "RuntimeError"
            and isinstance(message, str)
            and message == failure.get("message")
            and traceback_text.endswith(f"RuntimeError: {message}\n")
            and (
                (
                    environment_role == "separate_verifier"
                    and separate_verifier_trace
                    and separate_verifier_run_trace
                )
                or (
                    environment_role == "main"
                    and not separate_verifier_trace
                    and not separate_verifier_run_trace
                )
            )
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in _install_guest_utilities\n",
                traceback_text,
            )
            is not None
            and re.search(
                r'File "[^"\n]*proot_environment\.py", line [0-9]+, '
                r"in _run_guest_utility_install_stage\n",
                traceback_text,
            )
            is not None
        )
    return False


def _image_pull_environment_timeout_failure(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, str] | None:
    """A host-cancelled image pull is setup failure, never an agent timeout."""
    if not _trial_stopped_before_agent_and_verifier(trial):
        return None
    exception = trial.get("exception_info") or {}
    message = str(exception.get("exception_message") or "")
    trace = str(exception.get("exception_traceback") or "")
    if (exception.get("exception_type") != "EnvironmentStartTimeoutError"
        or re.fullmatch(r"Environment start timed out after [0-9]+(?:\.[0-9]+)? seconds", message) is None
        or not trace.rstrip().endswith("harbor.trial.errors.EnvironmentStartTimeoutError: " + message)
        or not all(re.search(r'File "[^"\n]*/proot_environment\.py", line [0-9]+, in ' + name + r"\n", trace)
                   for name in ("start", "_pull_image_with_retry", "_run_host"))):
        return None
    validated = _validated_local_receipt(trial_path, trial,
        name=PROOT_IMAGE_PULL_RECEIPT_NAME, schema=PROOT_IMAGE_PULL_RECEIPT_SCHEMA)
    if validated is None:
        return None
    receipt, _ = validated
    attempts = receipt.get("attempts")
    if (receipt.get("schema_version") != 1 or receipt.get("policy") != PROOT_IMAGE_PULL_POLICY
        or receipt.get("failure_policy") != "closed" or receipt.get("status") != "cancelled"
        or receipt.get("stage") != "skopeo_copy" or receipt.get("final_error_type") != "CancelledError"
        or receipt.get("partial_oci_reuse") is not True
        or not isinstance(attempts, list) or not 1 <= len(attempts) <= PROOT_IMAGE_PULL_ATTEMPTS
        or receipt.get("last_attempt") != len(attempts)
        or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("source_image_sha256"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("pull_image_sha256"))) is None
        or receipt.get("image_cache_key") != receipt["pull_image_sha256"][:20]):
        return None
    phase = trial.get("environment_setup") or {}
    times = [_parse_time(v) for v in (trial.get("started_at"), phase.get("started_at"),
        receipt.get("started_at"), receipt.get("finished_at"), phase.get("finished_at"),
        exception.get("occurred_at"), trial.get("finished_at"))]
    if any(t is None for t in times) or times != sorted(times):
        return None
    allowed = {name for name, _ in _PROOT_IMAGE_PULL_TRANSPORT_PATTERNS}
    previous = times[2]
    signals = set()
    for index, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, Mapping):
            return None
        start, end = (_parse_time(attempt.get(k)) for k in ("started_at", "finished_at"))
        if (start is None or end is None or not previous <= start <= end <= times[3]
            or attempt.get("attempt") != index or attempt.get("status") != "failed"
            or attempt.get("max_attempts") != PROOT_IMAGE_PULL_ATTEMPTS):
            return None
        previous = end
        if index == len(attempts):
            if (attempt.get("error_type") != "CancelledError" or attempt.get("retry_scheduled") is not False
                or attempt.get("error_sha256") != receipt.get("final_error_sha256")):
                return None
        else:
            observed = attempt.get("transport_signals")
            if (attempt.get("error_type") != "RuntimeError" or attempt.get("retry_scheduled") is not True
                or attempt.get("transient_transport_error") is not True or not isinstance(observed, list)
                or not observed or any(item not in allowed for item in observed)
                or attempt.get("backoff_seconds") != PROOT_IMAGE_PULL_BACKOFF_SECONDS[index - 1]
                or attempt.get("next_attempt") != index + 1):
                return None
            signals.update(observed)
    return dict(exception_type="EnvironmentStartTimeoutError", message=message,
        cause="proot_image_pull_environment_start_timeout", receipt=PROOT_IMAGE_PULL_RECEIPT_NAME,
        failure_recorded_at=times[3].isoformat(), transport_signals=",".join(sorted(signals)),
        infrastructure_kind="proot", evidence="host_image_pull_cancelled_before_agent_v1")


def _local_proot_infrastructure_trial_failure(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, str] | None:
    """Classify host-proven local PRoot failures, never model-authored text."""

    if _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH:
        return None
    legacy = _proot_rpc_trial_failure(trial)
    if legacy is not None:
        return legacy
    legacy_image_pull = _legacy_image_pull_transport_trial_failure(
        trial_path, trial
    )
    if legacy_image_pull is not None:
        return legacy_image_pull
    image_timeout = _image_pull_environment_timeout_failure(trial_path, trial)
    if image_timeout is not None:
        return image_timeout
    for classifier in (
        _image_pull_receipt_failure,
        _guest_utility_receipt_failure,
        _runtime_receipt_failure,
        _package_bridge_receipt_failure,
    ):
        failure = classifier(trial_path, trial)
        if failure is not None and _exception_matches_local_receipt_failure(
            trial, failure
        ):
            failure["infrastructure_kind"] = "proot"
            failure["evidence"] = "host_receipt_and_local_exception_v1"
            return failure
    return None


def _exception_is_local_proot_runtime(trial: Mapping[str, Any]) -> bool:
    """Require a receipt-bound exception to originate in pinned host runtime code."""

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return False
    exception_type = str(exception.get("exception_type") or "")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type not in {"PackageBridgeError", "RuntimeError"}
        or not isinstance(message, str)
        or not isinstance(traceback_text, str)
        or not traceback_text.endswith(f"{exception_type}: {message}\n")
    ):
        return False
    return (
        re.search(
            r'File "[^"\n]*(?:proot_environment|proot_package_bridge)\.py", '
            r"line [0-9]+, in "
            r"(?:start|exec|_build_rpc_server|_direct_exec|"
            r"_install_guest_utilities|_prepare_rootfs|"
            r"_run_guest_utility_install_stage|_spawn_cleanup_supervisor|"
            r"_start_package_bridge|_start_rpc_server|_wire_exec)\n",
            traceback_text,
        )
        is not None
    )


def _local_cleanup_warning(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return host-proven cleanup damage that happened after verification."""

    if "reward" not in _result_rewards(trial):
        return None
    verifier_timing = trial.get("verifier")
    verifier_finished = (
        _parse_time(verifier_timing.get("finished_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    if verifier_finished is None:
        return None
    trial_finished = _parse_time(trial.get("finished_at"))
    if trial_finished is None or trial_finished <= verifier_finished:
        return None
    exception = trial.get("exception_info")
    exception_at: datetime | None = None
    if isinstance(exception, Mapping) and exception:
        exception_at = _parse_time(exception.get("occurred_at"))
        if (
            exception_at is None
            or exception_at <= verifier_finished
            or exception_at > trial_finished
        ):
            return None
    evidence: list[dict[str, str]] = []
    runtime = _runtime_receipt_failure(trial_path, trial)
    if runtime is not None and runtime.get("cause") == (
        "proot_runtime_cleanup_incomplete"
    ):
        evidence.append(runtime)
    bridge = _package_bridge_receipt_failure(trial_path, trial)
    if bridge is not None and bridge.get("cause") == (
        "proot_package_bridge_cleanup_incomplete"
    ):
        evidence.append(bridge)
    if not evidence or any(
        (recorded := _parse_time(item.get("failure_recorded_at"))) is None
        or recorded <= verifier_finished
        or recorded > trial_finished
        or (exception_at is not None and recorded > exception_at)
        for item in evidence
    ):
        return None
    return {
        "policy": "preserve_verified_reward_without_resampling_v1",
        "evidence": evidence,
    }


def _exception_is_local_cleanup_only(trial: Mapping[str, Any]) -> bool:
    """Require the recorded exception itself to originate in local stop()."""

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return True
    traceback_text = exception.get("exception_traceback")
    message = exception.get("exception_message")
    if not isinstance(traceback_text, str) or not isinstance(message, str):
        return False
    if not traceback_text.endswith(
        f"{str(exception.get('exception_type') or '')}: {message}\n"
    ):
        return False
    return (
        re.search(
            r'File "[^"\n]*proot_environment\.py", line [0-9]+, in '
            r"(?:stop|_shutdown_package_bridge|_shutdown_rpc_runtime)\n",
            traceback_text,
        )
        is not None
    )


def _external_interruption_trial_failure(
    trial: Mapping[str, Any],
) -> dict[str, str] | None:
    """Recognize only Harbor cancellations caused by its SIGTERM handler.

    ``CancelledError`` is otherwise too broad to retry safely: an agent or an
    environment may raise it for deterministic reasons. Harbor's external
    signal path is distinguishable because its serialized traceback contains
    the CLI signal handler raising ``KeyboardInterrupt`` before asyncio
    cancels the in-flight agent call.
    """

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = str(exception.get("exception_type", "")).strip()
    traceback_text = str(exception.get("exception_traceback", ""))
    if (
        exception_type != "CancelledError"
        or "_handle_sigterm" not in traceback_text
        or "raise KeyboardInterrupt" not in traceback_text
        or "asyncio.exceptions.CancelledError" not in traceback_text
    ):
        return None
    return {
        "exception_type": exception_type,
        "message": str(exception.get("exception_message", "")).strip(),
        "cause": "harbor_external_sigterm",
    }


def _agent_timeout_trial_failure(
    trial: Mapping[str, Any],
) -> dict[str, str] | None:
    """Recognize only Harbor's former total-agent timeout terminal result.

    V0.5.3 now disables the total agent-phase timeout at the Harbor execution
    boundary.  Cells committed before that policy was installed may be retried
    without consuming another official attempt, but only when both the pool
    result and its pinned Harbor trial contain Harbor's typed timeout evidence.
    """

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = str(exception.get("exception_type", "")).strip()
    message = str(exception.get("exception_message", "")).strip()
    traceback_text = str(exception.get("exception_traceback", ""))
    if (
        exception_type != "AgentTimeoutError"
        or re.fullmatch(
            r"Agent execution timed out after [0-9]+(?:\.[0-9]+)? seconds",
            message,
        )
        is None
        or "harbor.trial.errors.AgentTimeoutError" not in traceback_text
    ):
        return None
    return {
        "exception_type": exception_type,
        "message": message,
        "cause": "harbor_total_agent_timeout",
    }


def _verified_reward_after_agent_timeout(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Prove that Harbor verified a TB2.1 task after its agent timed out.

    Harbor deliberately preserves ``AgentTimeoutError`` on the terminal result
    even when the verifier subsequently completes.  Treat that reward as
    authoritative only when the typed Harbor timeout, phase ordering, exact
    verifier payload, and clean local runtime all agree.  In particular, this
    helper never manufactures a zero when no verifier result exists.
    """

    if trial.get("source") != TB21_DATASET_SOURCE:
        return None
    config = trial.get("config")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    if (
        not isinstance(agent, Mapping)
        or agent.get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
        or _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
        or trial.get("verifier_environment_mode") != "shared"
    ):
        return None

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    timeout_match = (
        re.fullmatch(
            r"Agent execution timed out after ([0-9]+(?:\.[0-9]+)?) seconds",
            message,
        )
        if isinstance(message, str)
        else None
    )
    if timeout_match is None:
        return None
    timeout_seconds = float(timeout_match.group(1))
    if (
        exception_type != "AgentTimeoutError"
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or not isinstance(traceback_text, str)
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]single_step\.py", '
            r"line [0-9]+, in _run_agent\n",
            traceback_text,
        )
        is None
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]trial\.py", '
            r"line [0-9]+, in _run_agent_phase\n",
            traceback_text,
        )
        is None
        or "asyncio.exceptions.CancelledError" not in traceback_text
        or "TimeoutError" not in traceback_text
        or "raise AgentTimeoutError(" not in traceback_text
        or not traceback_text.endswith(
            "harbor.trial.errors.AgentTimeoutError: " + message + "\n"
        )
    ):
        return None

    verifier_result = trial.get("verifier_result")
    raw_rewards = (
        verifier_result.get("rewards") if isinstance(verifier_result, Mapping) else None
    )
    if (
        not isinstance(raw_rewards, Mapping)
        or not raw_rewards
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for key, value in raw_rewards.items()
        )
    ):
        return None
    normalized_rewards = {str(key): float(value) for key, value in raw_rewards.items()}
    reward = normalized_rewards.get("reward")
    if normalized_rewards != _result_rewards(trial) or reward not in {0.0, 1.0}:
        return None

    agent_timing = trial.get("agent_execution")
    verifier_timing = trial.get("verifier")
    trial_started = _parse_time(trial.get("started_at"))
    agent_started = (
        _parse_time(agent_timing.get("started_at"))
        if isinstance(agent_timing, Mapping)
        else None
    )
    agent_finished = (
        _parse_time(agent_timing.get("finished_at"))
        if isinstance(agent_timing, Mapping)
        else None
    )
    exception_at = _parse_time(exception.get("occurred_at"))
    verifier_started = (
        _parse_time(verifier_timing.get("started_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    verifier_finished = (
        _parse_time(verifier_timing.get("finished_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    trial_finished = _parse_time(trial.get("finished_at"))
    if any(
        value is None
        for value in (
            trial_started,
            agent_started,
            agent_finished,
            exception_at,
            verifier_started,
            verifier_finished,
            trial_finished,
        )
    ) or not (
        trial_started
        <= agent_started
        < agent_finished
        <= exception_at
        < verifier_started
        < verifier_finished
        <= trial_finished
    ):
        return None
    agent_elapsed = (agent_finished - agent_started).total_seconds()
    if not (
        agent_elapsed >= timeout_seconds * 0.99
        and agent_elapsed <= timeout_seconds + max(60.0, timeout_seconds * 0.05)
    ):
        return None

    trial_path = Path(trial_path)
    if trial_path.is_symlink() or not trial_path.is_dir():
        return None
    if any(
        classifier(trial_path, trial) is not None
        for classifier in (_runtime_receipt_failure, _package_bridge_receipt_failure)
    ):
        return None

    return {
        "policy": _VERIFIED_AGENT_TIMEOUT_REWARD_POLICY,
        "exception_type": "AgentTimeoutError",
        "timeout_seconds": timeout_seconds,
        "agent_started_at": agent_timing.get("started_at"),
        "agent_finished_at": agent_timing.get("finished_at"),
        "exception_occurred_at": exception.get("occurred_at"),
        "verifier_started_at": verifier_timing.get("started_at"),
        "verifier_finished_at": verifier_timing.get("finished_at"),
        "verifier_environment_mode": "shared",
        "reward": reward,
        "verifier_result_sha256": _digest(verifier_result),
    }


def _verifier_timeout_trial_failure(
    trial: Mapping[str, Any],
) -> dict[str, str] | None:
    """Recognize only Harbor's typed verifier-phase timeout terminal result."""

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = str(exception.get("exception_type", "")).strip()
    message = str(exception.get("exception_message", "")).strip()
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type != "VerifierTimeoutError"
        or re.fullmatch(
            r"Verifier execution timed out after [0-9]+(?:\.[0-9]+)? seconds",
            message,
        )
        is None
        or not isinstance(traceback_text, str)
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]single_step\.py", '
            r"line [0-9]+, in _run_verifier\n",
            traceback_text,
        )
        is None
        or "raise VerifierTimeoutError(" not in traceback_text
        or not traceback_text.endswith(
            "harbor.trial.errors.VerifierTimeoutError: " + message + "\n"
        )
    ):
        return None
    return {
        "exception_type": exception_type,
        "message": message,
        "cause": "harbor_verifier_timeout",
    }


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read one bounded, non-link incident artifact."""

    path = Path(path)
    if path.is_symlink():
        raise PoolError(f"incident artifact must not be a symlink: {path}")
    try:
        info = path.stat()
    except OSError as exc:
        raise PoolError(f"could not stat incident artifact: {path}") from exc
    if not path.is_file() or info.st_size > max_bytes:
        raise PoolError(f"invalid or oversized incident artifact: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PoolError(f"could not read incident artifact: {path}") from exc


def _clean_proot_incident_receipts(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Bind the incident to one clean, host-owned legacy PRoot lifecycle."""

    runtime_validated = _validated_local_receipt(
        trial_path,
        trial,
        name=PROOT_RUNTIME_RECEIPT_NAME,
        schema=PROOT_RUNTIME_RECEIPT_SCHEMA,
    )
    bridge_validated = _validated_local_receipt(
        trial_path,
        trial,
        name=PROOT_PACKAGE_BRIDGE_RECEIPT_NAME,
        schema=PROOT_PACKAGE_BRIDGE_RECEIPT_SCHEMA,
    )
    if runtime_validated is None or bridge_validated is None:
        return None
    runtime, runtime_events = runtime_validated
    bridge, bridge_events = bridge_validated
    if (
        runtime.get("cleanup_complete") is not True
        or runtime.get("cleanup_incomplete") is not False
        or runtime.get("rpc_cleanup_complete") is not True
        or runtime.get("rpc_poisoned") is not True
        or runtime.get("rpc_poison_reason") != "environment_stopped"
        or runtime.get("rpc_returncode") != 0
        or runtime.get("rpc_signal") is not None
        or runtime_events
        != frozenset({"rpc_runtime_stopped", "environment_stop_finalized"})
        or bridge.get("enabled") is not True
        or bridge.get("healthy") is not True
        or bridge.get("ready") is not False
        or bridge.get("status") != "stopped"
        or bridge.get("cleanup_complete") is not True
        or bridge.get("rpc_cleanup_complete") is not True
        or bridge.get("fatal_error") is not None
        or bridge.get("policy") != PROOT_PACKAGE_BRIDGE_POLICY
        or bridge.get("failure_policy") != "closed"
        or bridge.get("guest_socket_path") != "/run/harbor-package-bridge/control.sock"
        or not {
            "startup_started",
            "broker_ready",
            "probe_completed",
            "broker_stop_started",
            "broker_stopped",
            "environment_stop_finalized",
        }.issubset(bridge_events)
    ):
        return None
    verifier_timing = trial.get("verifier")
    verifier_finished = (
        _parse_time(verifier_timing.get("finished_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    trial_finished = _parse_time(trial.get("finished_at"))
    runtime_updated = _parse_time(runtime.get("updated_at"))
    bridge_updated = _parse_time(bridge.get("updated_at"))
    if (
        verifier_finished is None
        or trial_finished is None
        or runtime_updated is None
        or bridge_updated is None
        or not (verifier_finished < runtime_updated <= bridge_updated <= trial_finished)
    ):
        return None
    try:
        runtime_payload = _read_regular_bytes(
            Path(trial_path) / PROOT_RUNTIME_RECEIPT_NAME,
            max_bytes=2 * 1024 * 1024,
        )
        bridge_payload = _read_regular_bytes(
            Path(trial_path) / PROOT_PACKAGE_BRIDGE_RECEIPT_NAME,
            max_bytes=2 * 1024 * 1024,
        )
    except PoolError:
        return None
    return {
        "environment_import_path": _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH,
        "runtime_receipt_sha256": hashlib.sha256(runtime_payload).hexdigest(),
        "package_bridge_receipt_sha256": hashlib.sha256(bridge_payload).hexdigest(),
        "runtime_updated_at": runtime.get("updated_at"),
        "package_bridge_updated_at": bridge.get("updated_at"),
    }


def _qemu_agent_liveness_evidence(
    agent_dir: Path, *, expected_sha256: Any
) -> dict[str, Any] | None:
    """Require the audited Codex transcript and its QEMU liveness outputs."""

    output_path = Path(agent_dir) / "codex.txt"
    try:
        payload = _read_regular_bytes(output_path, max_bytes=2 * 1024 * 1024)
        lines = payload.decode("utf-8").splitlines()
    except (PoolError, UnicodeError):
        return None
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or observed_sha256 != expected_sha256
        or len(lines) < 5
        or not lines[0].startswith(
            "WARNING: proceeding, even though we could not create PATH aliases: "
        )
        or lines[1] != "Reading additional input from stdin..."
    ):
        return None
    records: list[Mapping[str, Any]] = []
    ignored_line_count = 0
    for line in lines[2:]:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # The audited transcript contains raw Codex ERROR continuations and
            # one command's binary-ish stdout outside the JSON event stream.
            # The exact digest above authenticates those lines; evidence below
            # is extracted only from independently parseable event records.
            ignored_line_count += 1
            continue
        if isinstance(record, Mapping):
            records.append(record)
        else:
            ignored_line_count += 1
    completed_outputs: list[str] = []
    for record in records:
        item = record.get("item")
        if (
            record.get("type") == "item.completed"
            and isinstance(item, Mapping)
            and item.get("type") == "command_execution"
            and item.get("status") == "completed"
            and item.get("exit_code") == 0
            and isinstance(item.get("aggregated_output"), str)
        ):
            completed_outputs.append(str(item["aggregated_output"]))
    boot_proven = any(
        all(
            marker in output
            for marker in (
                "=== LOGIN PROMPT SEEN ===",
                "Welcome to Alpine Linux 3.19",
                "Kernel 6.6.4-1-lts",
                "localhost login:",
            )
        )
        for output in completed_outputs
    )
    processes_proven = any(
        re.search(r"qemu pid [0-9]+ alive: yes", output) is not None
        and re.search(r"relay(?: v2)? pid [0-9]+ alive: yes", output) is not None
        for output in completed_outputs
    )
    interactive_proven = any(
        all(
            marker in output
            for marker in (
                "LIVE bytes received:",
                "Password:",
                "Login incorrect",
                "localhost login:",
            )
        )
        for output in completed_outputs
    )
    if not (boot_proven and processes_proven and interactive_proven):
        return None
    return {
        "codex_output_sha256": observed_sha256,
        "parsed_record_count": len(records),
        "ignored_nonrecord_line_count": ignored_line_count,
        "completed_command_count": len(completed_outputs),
        "boot_prompt_proven": True,
        "background_processes_proven": True,
        "interactive_relay_proven": True,
    }


def _proot_localhost_verifier_evidence(
    trial_path: Path,
    trial: Mapping[str, Any],
    *,
    task_contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate the exact official-verifier symptom for one incident task."""

    if (
        trial.get("exception_info") is not None
        or _result_rewards(trial) != {"reward": 0.0}
        or _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
    ):
        return None
    verifier_timing = trial.get("verifier")
    verifier_started = (
        _parse_time(verifier_timing.get("started_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    verifier_finished = (
        _parse_time(verifier_timing.get("finished_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    if (
        verifier_started is None
        or verifier_finished is None
        or verifier_finished <= verifier_started
    ):
        return None
    verifier_dir = Path(trial_path) / "verifier"
    try:
        reward_payload = _read_regular_bytes(verifier_dir / "reward.txt", max_bytes=128)
        report_payload = _read_regular_bytes(
            verifier_dir / "ctrf.json", max_bytes=8 * 1024 * 1024
        )
        stdout_payload = _read_regular_bytes(
            verifier_dir / "test-stdout.txt", max_bytes=8 * 1024 * 1024
        )
        report = json.loads(report_payload)
        stdout = stdout_payload.decode("utf-8")
    except (PoolError, UnicodeError, json.JSONDecodeError):
        return None
    if reward_payload != b"0\n" or not isinstance(report, Mapping):
        return None
    results = report.get("results")
    if not isinstance(results, Mapping) or set(results) != {"tool", "summary", "tests"}:
        return None
    tool = results.get("tool")
    summary = results.get("summary")
    tests = results.get("tests")
    if not isinstance(tool, Mapping) or dict(tool) != {
        "name": "pytest",
        "version": "8.4.1",
    }:
        return None
    if not isinstance(summary, Mapping) or not isinstance(tests, list):
        return None
    expected_tests = task_contract.get("tests")
    failed_names = task_contract.get("failed_tests")
    if not isinstance(expected_tests, Mapping) or not isinstance(failed_names, tuple):
        return None
    expected_failed = sum(status == "failed" for status in expected_tests.values())
    expected_passed = sum(status == "passed" for status in expected_tests.values())
    if (
        summary.get("tests") != len(expected_tests)
        or summary.get("passed") != expected_passed
        or summary.get("failed") != expected_failed
        or any(summary.get(key) != 0 for key in ("skipped", "pending", "other"))
        or len(tests) != len(expected_tests)
    ):
        return None
    observed: dict[str, str] = {}
    failed_traces: dict[str, str] = {}
    for test in tests:
        if not isinstance(test, Mapping):
            return None
        name = test.get("name")
        status = test.get("status")
        if (
            not isinstance(name, str)
            or name in observed
            or status not in {"passed", "failed"}
            or test.get("file_path") != "test_outputs.py"
        ):
            return None
        observed[name] = str(status)
        if status == "failed":
            trace = test.get("trace")
            if not isinstance(trace, str):
                return None
            failed_traces[name] = trace
    if observed != dict(expected_tests) or tuple(failed_traces) != failed_names:
        return None

    task_name = str(trial.get("task_name") or "")
    port = task_contract.get("localhost_port")
    if task_name in {
        "terminal-bench/headless-terminal",
        "terminal-bench/hf-model-inference",
    }:
        if not isinstance(port, int):
            return None
        for trace in failed_traces.values():
            if not all(
                marker in trace
                for marker in (
                    f"host='localhost', port={port}",
                    "socket.gaierror: [Errno -2] Name or service not known",
                    "urllib3.exceptions.NameResolutionError: HTTPConnection(",
                    "requests.exceptions.ConnectionError: HTTPConnectionPool(",
                )
            ):
                return None
        agent_evidence = None
    elif task_name == "terminal-bench/qemu-startup":
        trace = failed_traces.get("test_outputs.py::test_version", "")
        if not all(
            marker in trace
            for marker in (
                "FileNotFoundError: [Errno 2] No such file or directory: "
                "'/tmp/data.txt'",
                "/tests/test_outputs.py:58: FileNotFoundError",
            )
        ) or not all(
            marker in stdout
            for marker in (
                "send: spawn id exp4 not open",
                '(file "run.exp" line 22)',
            )
        ):
            return None
        agent_evidence = _qemu_agent_liveness_evidence(
            Path(trial_path) / "agent",
            expected_sha256=task_contract.get("codex_output_sha256"),
        )
        if agent_evidence is None:
            return None
    else:
        return None
    report_start = summary.get("start")
    report_stop = summary.get("stop")
    if (
        not isinstance(report_start, (int, float))
        or isinstance(report_start, bool)
        or not math.isfinite(float(report_start))
        or not isinstance(report_stop, (int, float))
        or isinstance(report_stop, bool)
        or not math.isfinite(float(report_stop))
        or float(report_stop) <= float(report_start)
    ):
        return None
    try:
        report_started = datetime.fromtimestamp(float(report_start), UTC)
        report_finished = datetime.fromtimestamp(float(report_stop), UTC)
    except (OverflowError, OSError, ValueError):
        return None
    if not (verifier_started <= report_started < report_finished <= verifier_finished):
        return None
    clean_receipts = _clean_proot_incident_receipts(trial_path, trial)
    if clean_receipts is None:
        return None
    return {
        "incident": _PROOT_LOCALHOST_INCIDENT,
        "classifier": task_contract["classifier"],
        "reward_artifact_sha256": hashlib.sha256(reward_payload).hexdigest(),
        "test_report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "test_stdout_sha256": hashlib.sha256(stdout_payload).hexdigest(),
        "tests": len(expected_tests),
        "passed": expected_passed,
        "failed": expected_failed,
        "failed_test_names": list(failed_names),
        "clean_proot_receipts": clean_receipts,
        "agent_liveness": agent_evidence,
    }


def _verified_success_after_agent_sigterm(
    trial_path: Path, trial: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Validate a verifier pass recorded after a narrowly proven agent SIGTERM.

    Harbor deliberately proceeds to verification after an agent command exits
    non-zero.  A signal exit can therefore coexist with a completely valid
    task success.  Preserve that success only when independent, host-owned
    verifier artifacts agree with the terminal TrialResult, the phase timing
    proves verification happened after the agent exception, and no local
    environment failure receipt exists.  The returned evidence is persisted
    in the cell result and recomputed by aggregation so a projected reward or
    a subsequently modified verifier artifact cannot become authoritative.
    """

    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type != "NonZeroAgentExitCodeError"
        or not isinstance(message, str)
        or not message.startswith("Command failed (exit 143): ")
        or not isinstance(traceback_text, str)
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]single_step\.py", '
            r"line [0-9]+, in _run_agent\n",
            traceback_text,
        )
        is None
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]agents[/\\]installed[/\\]base\.py", '
            r"line [0-9]+, in _exec\n",
            traceback_text,
        )
        is None
        or "raise self._classify_exec_error(command, result)\n" not in traceback_text
        or not traceback_text.endswith(
            "harbor.agents.installed.base.NonZeroAgentExitCodeError: " + message + "\n"
        )
    ):
        return None

    verifier_result = trial.get("verifier_result")
    raw_rewards = (
        verifier_result.get("rewards") if isinstance(verifier_result, Mapping) else None
    )
    if (
        not isinstance(raw_rewards, Mapping)
        or not raw_rewards
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for key, value in raw_rewards.items()
        )
    ):
        return None
    normalized_rewards = {str(key): float(value) for key, value in raw_rewards.items()}
    if (
        normalized_rewards != _result_rewards(trial)
        or normalized_rewards.get("reward") != 1.0
        or trial.get("verifier_environment_mode") != "shared"
    ):
        return None

    agent_timing = trial.get("agent_execution")
    verifier_timing = trial.get("verifier")
    trial_started = _parse_time(trial.get("started_at"))
    agent_started = (
        _parse_time(agent_timing.get("started_at"))
        if isinstance(agent_timing, Mapping)
        else None
    )
    agent_finished = (
        _parse_time(agent_timing.get("finished_at"))
        if isinstance(agent_timing, Mapping)
        else None
    )
    exception_at = _parse_time(exception.get("occurred_at"))
    verifier_started = (
        _parse_time(verifier_timing.get("started_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    verifier_finished = (
        _parse_time(verifier_timing.get("finished_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    trial_finished = _parse_time(trial.get("finished_at"))
    if any(
        value is None
        for value in (
            trial_started,
            agent_started,
            agent_finished,
            exception_at,
            verifier_started,
            verifier_finished,
            trial_finished,
        )
    ) or not (
        trial_started
        <= agent_started
        < agent_finished
        <= exception_at
        < verifier_started
        < verifier_finished
        <= trial_finished
    ):
        return None

    trial_path = Path(trial_path)
    if trial_path.is_symlink() or not trial_path.is_dir():
        return None
    if any(
        classifier(trial_path, trial) is not None
        for classifier in (_runtime_receipt_failure, _package_bridge_receipt_failure)
    ):
        return None

    reward_path = trial_path / "verifier" / "reward.txt"
    report_path = trial_path / "verifier" / "ctrf.json"
    try:
        reward_payload = _read_regular_bytes(reward_path, max_bytes=128)
        report_payload = _read_regular_bytes(report_path, max_bytes=8 * 1024 * 1024)
        reward_text = reward_payload.decode("ascii").strip()
        report = json.loads(report_payload)
    except (PoolError, UnicodeError, json.JSONDecodeError):
        return None
    if re.fullmatch(r"1(?:\.0*)?", reward_text) is None:
        return None

    results = report.get("results") if isinstance(report, Mapping) else None
    summary = results.get("summary") if isinstance(results, Mapping) else None
    tests = results.get("tests") if isinstance(results, Mapping) else None
    if not isinstance(summary, Mapping) or not isinstance(tests, list):
        return None
    count_fields = ("tests", "passed", "failed", "skipped", "pending", "other")
    if any(
        not isinstance(summary.get(key), int)
        or isinstance(summary.get(key), bool)
        or summary.get(key) < 0
        for key in count_fields
    ):
        return None
    test_count = int(summary["tests"])
    if (
        test_count <= 0
        or summary["passed"] != test_count
        or any(summary[key] != 0 for key in ("failed", "skipped", "pending", "other"))
        or len(tests) != test_count
        or any(
            not isinstance(test, Mapping)
            or not isinstance(test.get("name"), str)
            or not test.get("name")
            or test.get("status") != "passed"
            for test in tests
        )
        or len({str(test["name"]) for test in tests}) != test_count
    ):
        return None

    report_start_raw = summary.get("start")
    report_stop_raw = summary.get("stop")
    if (
        not isinstance(report_start_raw, (int, float))
        or isinstance(report_start_raw, bool)
        or not math.isfinite(float(report_start_raw))
        or not isinstance(report_stop_raw, (int, float))
        or isinstance(report_stop_raw, bool)
        or not math.isfinite(float(report_stop_raw))
    ):
        return None
    try:
        report_started = datetime.fromtimestamp(float(report_start_raw), UTC)
        report_finished = datetime.fromtimestamp(float(report_stop_raw), UTC)
    except (OverflowError, OSError, ValueError):
        return None
    if not (verifier_started <= report_started < report_finished <= verifier_finished):
        return None

    return {
        "policy": "preserve_complete_verifier_pass_after_agent_sigterm_v1",
        "agent_exit_code": 143,
        "agent_signal": int(signal.SIGTERM),
        "exception_type": exception_type,
        "exception_occurred_at": exception.get("occurred_at"),
        "verifier_started_at": verifier_timing.get("started_at"),
        "verifier_finished_at": verifier_timing.get("finished_at"),
        "reward": 1.0,
        "reward_artifact": "verifier/reward.txt",
        "reward_artifact_sha256": hashlib.sha256(reward_payload).hexdigest(),
        "test_report": "verifier/ctrf.json",
        "test_report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "tests": test_count,
        "passed": int(summary["passed"]),
    }


def _codex_high_demand_output_evidence(agent_dir: Path) -> dict[str, Any] | None:
    """Recognize the exact four-event Codex output emitted for HTTP 500."""

    output_path = Path(agent_dir) / "codex.txt"
    if not output_path.is_file() or output_path.is_symlink():
        return None
    payload = _read_regular_bytes(output_path, max_bytes=2 * 1024 * 1024)
    try:
        text = payload.decode("utf-8")
    except UnicodeError:
        return None
    lines = text.splitlines()
    if len(lines) != 6:
        return None
    if (
        not lines[0].startswith(
            "WARNING: proceeding, even though we could not create PATH aliases: "
        )
        or lines[1] != "Reading additional input from stdin..."
    ):
        return None
    try:
        events = [json.loads(line) for line in lines[2:]]
    except json.JSONDecodeError:
        return None
    if not all(isinstance(event, Mapping) for event in events):
        return None
    thread, started, failed_request, failed_turn = events
    thread_id = thread.get("thread_id")
    if (
        set(thread) != {"type", "thread_id"}
        or thread.get("type") != "thread.started"
        or not isinstance(thread_id, str)
        or re.fullmatch(r"[0-9a-f-]{16,64}", thread_id) is None
        or dict(started) != {"type": "turn.started"}
        or dict(failed_request)
        != {"type": "error", "message": _CODEX_HIGH_DEMAND_MESSAGE}
        or dict(failed_turn)
        != {
            "type": "turn.failed",
            "error": {"message": _CODEX_HIGH_DEMAND_MESSAGE},
        }
    ):
        return None
    return {
        "codex_output_sha256": hashlib.sha256(payload).hexdigest(),
        "codex_thread_id": thread_id,
    }


def _codex_high_demand_exception_evidence(
    value: Mapping[str, Any], *, codex_output: bytes
) -> dict[str, Any] | None:
    exception = value.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type != "NonZeroAgentExitCodeError"
        or not isinstance(message, str)
        or not isinstance(traceback_text, str)
        or not message.startswith("Command failed (exit 1): ")
        or " codex exec " not in message
        or message.count(_CODEX_HIGH_DEMAND_MESSAGE) != 2
        or not message.endswith("stderr: None")
        or "harbor.agents.installed.base.NonZeroAgentExitCodeError"
        not in traceback_text
        or not traceback_text.endswith(
            "harbor.agents.installed.base.NonZeroAgentExitCodeError: " + message + "\n"
        )
    ):
        return None
    try:
        rendered_output = codex_output.decode("utf-8").rstrip("\n")
    except UnicodeError:
        return None
    if f"stdout: {rendered_output}\n\nstderr: None" not in message:
        return None
    return {
        "exception_type": "NonZeroAgentExitCodeError",
        "exception_message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "exception_traceback_sha256": hashlib.sha256(
            traceback_text.encode("utf-8")
        ).hexdigest(),
    }


def _codex_truncated_high_demand_exception_evidence(
    value: Mapping[str, Any], *, codex_output: bytes
) -> dict[str, Any] | None:
    """Bind Harbor's fixed 250/749-character stdout projection to codex.txt."""

    exception = value.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type != "NonZeroAgentExitCodeError"
        or not isinstance(message, str)
        or not message.startswith("Command failed (exit 1): ")
        or " codex exec " not in message
        or message.count(_CODEX_HIGH_DEMAND_MESSAGE) != 2
        or not message.endswith("\n\nstderr: None")
        or not isinstance(traceback_text, str)
        or not traceback_text.endswith(
            "harbor.agents.installed.base.NonZeroAgentExitCodeError: " + message + "\n"
        )
    ):
        return None
    try:
        rendered_output = codex_output.decode("utf-8").rstrip("\n")
    except UnicodeError:
        return None
    marker_matches = list(
        re.finditer(r"\.\.\. \[([0-9]+) chars truncated\] \.\.\.", message)
    )
    if len(marker_matches) != 1:
        return None
    marker = marker_matches[0]
    before = message[: marker.start()]
    after = message[marker.end() :]
    stdout_offset = before.rfind("stdout: ")
    if stdout_offset < 0:
        return None
    prefix = before[stdout_offset + len("stdout: ") :]
    suffix = after[: -len("\n\nstderr: None")]
    removed = int(marker.group(1))
    # Harbor's render truncation preserves exactly the first 250 and final 749
    # stdout characters, with one separator space on each side of the marker.
    if (
        len(prefix) != 251
        or not prefix.endswith(" ")
        or len(suffix) != 750
        or not suffix.startswith(" ")
        or rendered_output[:250] != prefix[:-1]
        or rendered_output[-749:] != suffix[1:]
        or len(rendered_output) - 250 - 749 != removed
    ):
        return None
    return {
        "exception_type": "NonZeroAgentExitCodeError",
        "stdout_projection": "harbor_first_250_last_749_v1",
        "stdout_truncated_characters": removed,
        "exception_message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "exception_traceback_sha256": hashlib.sha256(
            traceback_text.encode("utf-8")
        ).hexdigest(),
    }


def _zero_token_codex_failure(value: Mapping[str, Any]) -> bool:
    if "n_input_tokens" in value or "n_output_tokens" in value:
        return value.get("n_input_tokens") == 0 and value.get("n_output_tokens") == 0
    agent_result = value.get("agent_result")
    if not isinstance(agent_result, Mapping):
        return False
    if "n_input_tokens" not in agent_result or "n_output_tokens" not in agent_result:
        return False
    return agent_result.get("n_input_tokens") in (None, 0) and agent_result.get(
        "n_output_tokens"
    ) in (None, 0)


def _zero_reward_codex_failure(value: Mapping[str, Any]) -> bool:
    raw = value.get("rewards")
    if isinstance(raw, Mapping) and not raw:
        rejected = value.get("rejected_verifier_rewards")
        raw = rejected if isinstance(rejected, Mapping) else raw
    rewards = (
        {
            str(key): float(item)
            for key, item in raw.items()
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        }
        if isinstance(raw, Mapping)
        else _result_rewards(value)
    )
    return rewards == {"reward": 0.0}


def _gateway_rg_incident_evidence(agent_dir: Path) -> dict[str, Any] | None:
    """Classify diagnosed and explicitly selected pre-diagnostic cohorts."""

    agent_dir = Path(agent_dir)
    if agent_dir.is_symlink() or not agent_dir.is_dir():
        return None
    gateway_entries = {
        path.name
        for path in agent_dir.iterdir()
        if path.name.startswith("codex-r2eact-") or path.name == "memory.jsonl"
    }
    if not gateway_entries.issubset(_CODEX_GATEWAY_ARTIFACT_NAMES):
        return None
    event_path = agent_dir / "codex-r2eact-gateway-events.jsonl"
    other_gateway_entries = gateway_entries - {event_path.name}
    if other_gateway_entries:
        return None
    if event_path.is_symlink():
        return None
    if not event_path.exists():
        if gateway_entries:
            return None
        return {
            "classifier": "prediagnostic_zero_token_high_demand_no_gateway_v1",
            "gateway_event_count": 0,
            "gateway_events_sha256": None,
        }
    payload = _read_regular_bytes(event_path, max_bytes=2 * 1024 * 1024)
    records: list[dict[str, Any]] = []
    try:
        for line in payload.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                return None
            records.append(dict(value))
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not records:
        return None
    for record in records:
        traceback_text = record.get("traceback")
        if (
            set(record)
            != {
                "event",
                "observed_at",
                "error_type",
                "error_message",
                "traceback",
            }
            or record.get("event") != "gateway_request_failed"
            or not isinstance(record.get("observed_at"), (int, float))
            or isinstance(record.get("observed_at"), bool)
            or record.get("error_type") != "RuntimeError"
            or record.get("error_message") != _CODEX_HOST_RG_ERROR_MESSAGE
            or not isinstance(traceback_text, str)
            or " in configure\n" not in traceback_text
            or "load_sglang_live_bundle" not in traceback_text
            or "preflight_corpus_tool(tool_runner)" not in traceback_text
            or "grepseek_tool.py" not in traceback_text
            or not traceback_text.endswith(
                "RuntimeError: " + _CODEX_HOST_RG_ERROR_MESSAGE + "\n"
            )
        ):
            return None
    return {
        "classifier": "diagnosed_gateway_host_rg_event_v1",
        "gateway_event_count": len(records),
        "gateway_events_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _codex_host_rg_incident_failure(
    value: Mapping[str, Any], *, agent_dir: Path
) -> dict[str, Any] | None:
    """Recognize only the 2026-08-28 missing-host-rg infrastructure incident."""

    if not _zero_token_codex_failure(value) or not _zero_reward_codex_failure(value):
        return None
    output_evidence = _codex_high_demand_output_evidence(agent_dir)
    if output_evidence is None:
        return None
    codex_output = _read_regular_bytes(
        Path(agent_dir) / "codex.txt", max_bytes=2 * 1024 * 1024
    )
    exception_evidence = _codex_high_demand_exception_evidence(
        value, codex_output=codex_output
    )
    gateway_evidence = _gateway_rg_incident_evidence(agent_dir)
    if exception_evidence is None or gateway_evidence is None:
        return None
    return {
        "incident": _CODEX_HOST_RG_INCIDENT,
        **gateway_evidence,
        **output_evidence,
        **exception_evidence,
    }


def _codex_gateway_timeout_incident_evidence(
    trial_path: Path,
    trial: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Prove one frozen long-trajectory recursive-memory gateway timeout.

    The generic Codex ``high demand`` rendering is not attribution evidence.
    This classifier additionally binds the terminal TrialResult to the exact
    archived agent output, the complete gateway event journal, every completed
    round receipt, the post-exception verifier report, and clean host-owned
    PRoot lifecycle receipts.  The incident contract pins all bytes so even a
    semantically similar artifact outside the audited cohort is rejected.
    """

    trial_path = Path(trial_path)
    expected_hashes = contract.get("artifact_sha256")
    expected_tests_raw = contract.get("verifier_test_names")
    if (
        trial_path.is_symlink()
        or not trial_path.is_dir()
        or not isinstance(expected_hashes, Mapping)
        or set(expected_hashes) != _CODEX_GATEWAY_TIMEOUT_ARTIFACT_KEYS
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for key, value in expected_hashes.items()
        )
        or not isinstance(expected_tests_raw, (list, tuple))
        or not expected_tests_raw
        or any(not isinstance(name, str) or not name for name in expected_tests_raw)
        or len(set(expected_tests_raw)) != len(expected_tests_raw)
    ):
        return None
    expected_tests = tuple(expected_tests_raw)
    artifact_paths = {
        "trial/result.json": trial_path / "result.json",
        "agent/codex.txt": trial_path / "agent" / "codex.txt",
        "agent/codex-r2eact-gateway-events.jsonl": (
            trial_path / "agent" / "codex-r2eact-gateway-events.jsonl"
        ),
        "agent/codex-r2eact-gateway-receipts.json": (
            trial_path / "agent" / "codex-r2eact-gateway-receipts.json"
        ),
        "verifier/reward.txt": trial_path / "verifier" / "reward.txt",
        "verifier/ctrf.json": trial_path / "verifier" / "ctrf.json",
        "proot-runtime.json": trial_path / PROOT_RUNTIME_RECEIPT_NAME,
        "proot-package-bridge.json": (trial_path / PROOT_PACKAGE_BRIDGE_RECEIPT_NAME),
    }
    try:
        payloads = {
            name: _read_regular_bytes(path, max_bytes=MAX_ARTIFACT_BYTES)
            for name, path in artifact_paths.items()
        }
    except PoolError:
        return None
    observed_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    if any(observed_hashes[name] != expected_hashes[name] for name in payloads):
        return None
    try:
        raw_trial = json.loads(payloads["trial/result.json"])
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw_trial, Mapping) or dict(raw_trial) != dict(trial):
        return None

    exception = trial.get("exception_info")
    agent_result = trial.get("agent_result")
    agent_timing = trial.get("agent_execution")
    verifier_timing = trial.get("verifier")
    if (
        not isinstance(exception, Mapping)
        or not isinstance(agent_result, Mapping)
        or not isinstance(agent_timing, Mapping)
        or not isinstance(verifier_timing, Mapping)
        or _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
        or trial.get("verifier_environment_mode") != "shared"
        or _result_rewards(trial) != {"reward": 0.0}
    ):
        return None
    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type != "NonZeroAgentExitCodeError"
        or not isinstance(message, str)
        or not message.startswith("Command failed (exit 1): ")
        or " codex exec " not in message
        or not isinstance(traceback_text, str)
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]single_step\.py", '
            r"line [0-9]+, in _run_agent\n",
            traceback_text,
        )
        is None
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]agents[/\\]installed[/\\]base\.py", '
            r"line [0-9]+, in _exec\n",
            traceback_text,
        )
        is None
        or "raise self._classify_exec_error(command, result)\n" not in traceback_text
        or not traceback_text.endswith(
            "harbor.agents.installed.base.NonZeroAgentExitCodeError: " + message + "\n"
        )
    ):
        return None

    codex_payload = payloads["agent/codex.txt"]
    try:
        codex_text = codex_payload.decode("utf-8")
        codex_lines = codex_text.splitlines()
    except UnicodeError:
        return None
    codex_events: list[Mapping[str, Any]] = []
    codex_diagnostic_line_count = 0
    for line in codex_lines[2:]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Codex can interleave its own timestamped router diagnostics with
            # the JSONL stream (and a rejected multiline shell command can
            # span several physical lines).  The frozen output hash binds
            # those bytes; only parsed JSON objects are used as phase events.
            codex_diagnostic_line_count += 1
            continue
        if not isinstance(event, Mapping):
            return None
        codex_events.append(event)
    if (
        len(codex_lines) < 8
        or not codex_lines[0].startswith(
            "WARNING: proceeding, even though we could not create PATH aliases: "
        )
        or codex_lines[1] != "Reading additional input from stdin..."
        or len(codex_events) < 6
        or codex_events[0].get("type") != "thread.started"
        or not isinstance(codex_events[0].get("thread_id"), str)
        or re.fullmatch(r"[0-9a-f-]{16,64}", codex_events[0]["thread_id"]) is None
        or dict(codex_events[1]) != {"type": "turn.started"}
        or dict(codex_events[-2])
        != {"type": "error", "message": _CODEX_HIGH_DEMAND_MESSAGE}
        or dict(codex_events[-1])
        != {
            "type": "turn.failed",
            "error": {"message": _CODEX_HIGH_DEMAND_MESSAGE},
        }
        or sum(event.get("type") == "error" for event in codex_events) != 1
        or sum(event.get("type") == "turn.failed" for event in codex_events) != 1
        or not any(
            event.get("type") == "item.completed"
            and isinstance(event.get("item"), Mapping)
            and event["item"].get("type") == "command_execution"
            and event["item"].get("status") == "completed"
            and event["item"].get("exit_code") == 0
            for event in codex_events[:-2]
        )
    ):
        return None
    exception_evidence = _codex_truncated_high_demand_exception_evidence(
        trial, codex_output=codex_payload
    )
    if exception_evidence is None:
        return None

    events_payload = payloads["agent/codex-r2eact-gateway-events.jsonl"]
    gateway_events: list[dict[str, Any]] = []
    try:
        for line in events_payload.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, Mapping):
                return None
            gateway_events.append(dict(event))
    except (UnicodeError, json.JSONDecodeError):
        return None
    allowed_events = {
        "gateway_backend_configured",
        "compression_finished",
        "model_tool_validation_recovered",
        "response_committed",
        "gateway_request_failed",
    }
    event_counts: dict[str, int] = {}
    for event in gateway_events:
        name = event.get("event")
        if not isinstance(name, str) or name not in allowed_events:
            return None
        event_counts[name] = event_counts.get(name, 0) + 1
    failures = [
        event
        for event in gateway_events
        if event.get("event") == "gateway_request_failed"
    ]
    if (
        len(failures) != 1
        or not gateway_events
        or gateway_events[-1] != failures[0]
        or event_counts.get("gateway_backend_configured") != 1
    ):
        return None
    gateway_failure = failures[0]
    gateway_traceback = gateway_failure.get("traceback")
    gateway_observed_raw = gateway_failure.get("observed_at")
    if (
        set(gateway_failure)
        != {"event", "observed_at", "error_type", "error_message", "traceback"}
        or gateway_failure.get("error_type") != "TimeoutError"
        or gateway_failure.get("error_message") != "timed out"
        or not isinstance(gateway_observed_raw, (int, float))
        or isinstance(gateway_observed_raw, bool)
        or not math.isfinite(float(gateway_observed_raw))
        or not isinstance(gateway_traceback, str)
        or not gateway_traceback.endswith("TimeoutError: timed out\n")
    ):
        return None
    gateway_trace_markers = (
        "in do_POST\n",
        "self.server.gateway.handle(request)",
        "in generate_next\n",
        "in _generate_next_locked\n",
        "in materialize_pending_gpu_memory\n",
        "self.client.encode_recursive_memory_chain(",
        "in encode_recursive_memory_chain\n",
        "response = self._post(RECURSIVE_COMPRESSION_CHAIN_ROUTE, request.to_json())",
        "in _post\n",
        "with urllib.request.urlopen(",
        "TimeoutError: timed out\n",
    )
    marker_position = -1
    for marker in gateway_trace_markers:
        marker_position = gateway_traceback.find(marker, marker_position + 1)
        if marker_position < 0:
            return None

    receipts_payload = payloads["agent/codex-r2eact-gateway-receipts.json"]
    try:
        receipts = json.loads(receipts_payload)
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(receipts, list) or len(receipts) < 2:
        return None
    valid_receipts: list[Mapping[str, Any]] = []
    recovered_receipts: list[Mapping[str, Any]] = []
    for index, receipt in enumerate(receipts, start=1):
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("round_index") != index
            or not isinstance(receipt.get("elapsed_seconds"), (int, float))
            or isinstance(receipt.get("elapsed_seconds"), bool)
            or not math.isfinite(float(receipt["elapsed_seconds"]))
            or float(receipt["elapsed_seconds"]) <= 0
            or not isinstance(receipt.get("input_tokens"), int)
            or isinstance(receipt.get("input_tokens"), bool)
            or receipt["input_tokens"] <= 0
            or not isinstance(receipt.get("output_tokens"), int)
            or isinstance(receipt.get("output_tokens"), bool)
            or receipt["output_tokens"] <= 0
            or receipt.get("finished_on_eos") is not True
            or receipt.get("hit_generation_cap") is not False
            or not isinstance(receipt.get("complete_valid"), bool)
        ):
            return None
        (valid_receipts if receipt["complete_valid"] else recovered_receipts).append(
            receipt
        )
    input_tokens = agent_result.get("n_input_tokens")
    output_tokens = agent_result.get("n_output_tokens")
    if (
        not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or input_tokens <= 0
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens <= 0
        or not valid_receipts
        or receipts[-1].get("complete_valid") is not True
        or len(receipts) != event_counts.get("compression_finished", 0)
        or len(valid_receipts) != event_counts.get("response_committed", 0)
        or len(recovered_receipts)
        != event_counts.get("model_tool_validation_recovered", 0)
        or sum(int(receipt["input_tokens"]) for receipt in valid_receipts)
        != input_tokens
        or sum(int(receipt["output_tokens"]) for receipt in valid_receipts)
        != output_tokens
    ):
        return None

    reward_payload = payloads["verifier/reward.txt"]
    report_payload = payloads["verifier/ctrf.json"]
    try:
        reward_text = reward_payload.decode("ascii").strip()
        report = json.loads(report_payload)
    except (UnicodeError, json.JSONDecodeError):
        return None
    results = report.get("results") if isinstance(report, Mapping) else None
    summary = results.get("summary") if isinstance(results, Mapping) else None
    tests = results.get("tests") if isinstance(results, Mapping) else None
    count_fields = ("tests", "passed", "failed", "skipped", "pending", "other")
    if (
        re.fullmatch(r"0(?:\.0*)?", reward_text) is None
        or not isinstance(summary, Mapping)
        or not isinstance(tests, list)
        or any(
            not isinstance(summary.get(key), int)
            or isinstance(summary.get(key), bool)
            or summary[key] < 0
            for key in count_fields
        )
        or summary["tests"] <= 0
        or summary["passed"] != 0
        or summary["failed"] != summary["tests"]
        or any(summary[key] != 0 for key in ("skipped", "pending", "other"))
        or len(tests) != summary["tests"]
        or tuple(
            test.get("name") if isinstance(test, Mapping) else None for test in tests
        )
        != expected_tests
        or any(
            not isinstance(test, Mapping) or test.get("status") != "failed"
            for test in tests
        )
    ):
        return None

    trial_started = _parse_time(trial.get("started_at"))
    agent_started = _parse_time(agent_timing.get("started_at"))
    agent_finished = _parse_time(agent_timing.get("finished_at"))
    exception_at = _parse_time(exception.get("occurred_at"))
    verifier_started = _parse_time(verifier_timing.get("started_at"))
    verifier_finished = _parse_time(verifier_timing.get("finished_at"))
    trial_finished = _parse_time(trial.get("finished_at"))
    report_start_raw = summary.get("start")
    report_stop_raw = summary.get("stop")
    if (
        any(
            value is None
            for value in (
                trial_started,
                agent_started,
                agent_finished,
                exception_at,
                verifier_started,
                verifier_finished,
                trial_finished,
            )
        )
        or not isinstance(report_start_raw, (int, float))
        or isinstance(report_start_raw, bool)
        or not math.isfinite(float(report_start_raw))
        or not isinstance(report_stop_raw, (int, float))
        or isinstance(report_stop_raw, bool)
        or not math.isfinite(float(report_stop_raw))
    ):
        return None
    try:
        gateway_observed = datetime.fromtimestamp(float(gateway_observed_raw), UTC)
        report_started = datetime.fromtimestamp(float(report_start_raw), UTC)
        report_finished = datetime.fromtimestamp(float(report_stop_raw), UTC)
    except (OverflowError, OSError, ValueError):
        return None
    assert trial_started is not None
    assert agent_started is not None
    assert agent_finished is not None
    assert exception_at is not None
    assert verifier_started is not None
    assert verifier_finished is not None
    assert trial_finished is not None
    if not (
        trial_started
        <= agent_started
        < gateway_observed
        < agent_finished
        <= exception_at
        < verifier_started
        <= report_started
        < report_finished
        <= verifier_finished
        <= trial_finished
    ):
        return None

    clean_receipts = _clean_proot_incident_receipts(trial_path, trial)
    if clean_receipts is None:
        return None
    return {
        "incident": _CODEX_GATEWAY_TIMEOUT_INCIDENT,
        "classifier": _CODEX_GATEWAY_TIMEOUT_CLASSIFIER,
        "exception_type": exception_type,
        "agent_exit_code": 1,
        "gateway_error_type": "TimeoutError",
        "gateway_error_message": "timed out",
        "gateway_failure_count": 1,
        "gateway_failure_observed_at": gateway_observed.isoformat(),
        "completed_round_receipt_count": len(receipts),
        "committed_response_count": len(valid_receipts),
        "recovered_invalid_response_count": len(recovered_receipts),
        "codex_diagnostic_line_count": codex_diagnostic_line_count,
        "n_input_tokens": input_tokens,
        "n_output_tokens": output_tokens,
        "verifier_reward": 0.0,
        "tests": int(summary["tests"]),
        "passed": 0,
        "failed": int(summary["failed"]),
        "failed_test_names": list(expected_tests),
        "artifact_sha256": observed_hashes,
        "agent_result_sha256": _digest(agent_result),
        "exception_evidence": exception_evidence,
        "clean_proot_receipts": clean_receipts,
    }


def _decode_dynamic_gateway_receipts(
    artifact_payloads: Mapping[str, bytes],
) -> tuple[list[dict[str, Any]], str]:
    """Decode the canonical journal or the in-flight legacy receipt array."""

    prefix = f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/"
    journal_key = prefix + GATEWAY_RECEIPTS_JSONL_FILENAME
    summary_key = prefix + GATEWAY_RECEIPTS_SUMMARY_FILENAME
    legacy_key = prefix + GATEWAY_RECEIPTS_FILENAME
    if journal_key in artifact_payloads or summary_key in artifact_payloads:
        if journal_key not in artifact_payloads or summary_key not in artifact_payloads:
            raise ValueError("canonical gateway receipts are incomplete")
        receipts: list[dict[str, Any]] = []
        previous: str | None = None
        try:
            for line in artifact_payloads[journal_key].splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("gateway receipt journal record is not an object")
                receipt, previous = decode_receipt_record(
                    value,
                    expected_index=len(receipts) + 1,
                    expected_previous_sha256=previous,
                )
                receipts.append(receipt)
            summary = json.loads(artifact_payloads[summary_key])
        except (
            UnicodeError,
            json.JSONDecodeError,
            GatewayReceiptPersistenceError,
        ) as exc:
            raise ValueError("canonical gateway receipts are invalid") from exc
        if not isinstance(summary, Mapping):
            raise ValueError("gateway receipt summary is not an object")

        latest_round = 0
        latest: dict[str, Any] = {}
        input_tokens = 0
        output_tokens = 0
        has_input_tokens = False
        has_output_tokens = False
        for receipt in receipts:
            round_index = receipt.get("round_index")
            if isinstance(round_index, int) and not isinstance(round_index, bool):
                if round_index >= latest_round:
                    latest_round = round_index
                    latest = receipt
            input_value = receipt.get("input_tokens")
            if isinstance(input_value, (int, float)) and not isinstance(
                input_value, bool
            ):
                input_tokens += int(input_value)
                has_input_tokens = True
            output_value = receipt.get("output_tokens", receipt.get("token_count"))
            if isinstance(output_value, (int, float)) and not isinstance(
                output_value, bool
            ):
                output_tokens += int(output_value)
                has_output_tokens = True
        latest_output = latest.get("output_tokens", latest.get("token_count"))
        compression = latest.get("compression")
        expected_summary = {
            "schema": GATEWAY_RECEIPT_SUMMARY_SCHEMA,
            "source_filename": GATEWAY_RECEIPTS_JSONL_FILENAME,
            "source_size_bytes": len(artifact_payloads[journal_key]),
            "records": len(receipts),
            "last_record_sha256": previous,
            "latest_round": latest_round,
            "input_tokens": input_tokens if has_input_tokens else None,
            "output_tokens": output_tokens if has_output_tokens else None,
            "latest_generation": (
                {
                    "round_index": latest.get("round_index"),
                    "elapsed_seconds": latest.get("elapsed_seconds"),
                    "input_tokens": latest.get("input_tokens"),
                    "token_count": latest_output,
                    "finished_on_eos": latest.get("finished_on_eos"),
                    "hit_generation_cap": latest.get("hit_generation_cap"),
                    "complete_valid": latest.get("complete_valid"),
                    "finish_reason": latest.get("finish_reason"),
                    "tool_calls": latest.get("tool_calls"),
                }
                if latest
                else {}
            ),
            "latest_compression": (
                dict(compression) if isinstance(compression, Mapping) else {}
            ),
        }
        if dict(summary) != expected_summary:
            raise ValueError("gateway receipt summary disagrees with its journal")
        return receipts, "canonical_jsonl"

    if legacy_key not in artifact_payloads:
        raise ValueError("gateway receipt artifact is missing")
    try:
        value = json.loads(artifact_payloads[legacy_key])
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("legacy gateway receipt array is invalid") from exc
    if not isinstance(value, list) or any(
        not isinstance(receipt, Mapping) for receipt in value
    ):
        raise ValueError("legacy gateway receipt artifact is not an object array")
    return [dict(receipt) for receipt in value], "legacy_json"


def _codex_gateway_request_timeout_evidence(
    trial_path: Path,
    trial: Mapping[str, Any],
    *,
    artifact_payloads: Mapping[str, bytes],
) -> dict[str, Any] | None:
    """Prove a timeout recorded by the host-owned Codex gateway controller.

    Codex renders every HTTP-500 response as a generic ``high demand`` error.
    That text is therefore only a consistency check.  Retry authority comes
    from the private ``.r2eact-controller`` journal, whose terminal event must
    be one typed timeout in the recursive-memory transport after committed
    model progress.  The pool caller reads every authority-bearing artifact
    once under the cell flock and supplies those exact bytes here.
    """

    trial_path = Path(trial_path)
    agent_dir = trial_path / "agent"
    controller_dir = agent_dir / CONTROLLER_ARTIFACT_DIRNAME
    artifact_keys = frozenset(artifact_payloads)
    if (
        artifact_keys not in _CODEX_GATEWAY_REQUEST_TIMEOUT_ARTIFACT_KEY_SETS
        or trial_path.is_symlink()
        or not trial_path.is_dir()
        or agent_dir.is_symlink()
        or not agent_dir.is_dir()
        or controller_dir.is_symlink()
        or not controller_dir.is_dir()
    ):
        return None
    try:
        controller_stat = controller_dir.stat()
    except OSError:
        return None
    if (
        controller_stat.st_uid != os.geteuid()
        or controller_stat.st_mode & 0o777 != PRIVATE_DIR_MODE
    ):
        return None

    try:
        raw_trial = json.loads(artifact_payloads["trial/result.json"])
    except (UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw_trial, Mapping) or dict(raw_trial) != dict(trial):
        return None

    exception = trial.get("exception_info")
    agent_result = trial.get("agent_result")
    agent_timing = trial.get("agent_execution")
    verifier_timing = trial.get("verifier")
    trial_config = trial.get("config")
    trial_agent = (
        trial_config.get("agent") if isinstance(trial_config, Mapping) else None
    )
    agent_kwargs = (
        trial_agent.get("kwargs") if isinstance(trial_agent, Mapping) else None
    )
    request_timeout = (
        # CodexR2EActAgent's default is 1800s; current residual manifests omit
        # the kwarg. Streaming evidence below must independently attest it.
        agent_kwargs.get("request_timeout", 1800.0)
        if isinstance(agent_kwargs, Mapping)
        else None
    )
    if (
        not isinstance(exception, Mapping)
        or not isinstance(agent_result, Mapping)
        or not isinstance(agent_timing, Mapping)
        or not isinstance(verifier_timing, Mapping)
        or not isinstance(trial_agent, Mapping)
        or trial_agent.get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
        or not isinstance(request_timeout, (int, float))
        or isinstance(request_timeout, bool)
        or not math.isfinite(float(request_timeout))
        or float(request_timeout) <= 0
        or _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
        or trial.get("verifier_environment_mode") != "shared"
        or _result_rewards(trial) != {"reward": 0.0}
    ):
        return None

    exception_type = exception.get("exception_type")
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception_type != "NonZeroAgentExitCodeError"
        or not isinstance(message, str)
        or not message.startswith("Command failed (exit 1): ")
        or " codex exec " not in message
        or not isinstance(traceback_text, str)
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]single_step\.py", '
            r"line [0-9]+, in _run_agent\n",
            traceback_text,
        )
        is None
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]agents[/\\]installed[/\\]base\.py", '
            r"line [0-9]+, in _exec\n",
            traceback_text,
        )
        is None
        or "raise self._classify_exec_error(command, result)\n" not in traceback_text
        or not traceback_text.endswith(
            "harbor.agents.installed.base.NonZeroAgentExitCodeError: " + message + "\n"
        )
    ):
        return None

    codex_payload = artifact_payloads["agent/codex.txt"]
    try:
        codex_lines = codex_payload.decode("utf-8").splitlines()
    except UnicodeError:
        return None
    codex_events: list[Mapping[str, Any]] = []
    diagnostic_lines = 0
    for line in codex_lines[2:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            diagnostic_lines += 1
            continue
        if not isinstance(value, Mapping):
            return None
        codex_events.append(value)
    if (
        len(codex_lines) < 8
        or not codex_lines[0].startswith(
            "WARNING: proceeding, even though we could not create PATH aliases: "
        )
        or codex_lines[1] != "Reading additional input from stdin..."
        or len(codex_events) < 6
        or codex_events[0].get("type") != "thread.started"
        or not isinstance(codex_events[0].get("thread_id"), str)
        or re.fullmatch(r"[0-9a-f-]{16,64}", codex_events[0]["thread_id"]) is None
        or dict(codex_events[1]) != {"type": "turn.started"}
        or dict(codex_events[-2])
        != {"type": "error", "message": _CODEX_HIGH_DEMAND_MESSAGE}
        or dict(codex_events[-1])
        != {
            "type": "turn.failed",
            "error": {"message": _CODEX_HIGH_DEMAND_MESSAGE},
        }
        or sum(event.get("type") == "error" for event in codex_events) != 1
        or sum(event.get("type") == "turn.failed" for event in codex_events) != 1
        or not any(
            event.get("type") == "item.completed"
            and isinstance(event.get("item"), Mapping)
            and event["item"].get("type") == "command_execution"
            and event["item"].get("status") == "completed"
            and event["item"].get("exit_code") == 0
            for event in codex_events[:-2]
        )
    ):
        return None
    exception_evidence = _codex_truncated_high_demand_exception_evidence(
        trial, codex_output=codex_payload
    )
    if exception_evidence is None:
        return None

    events_key = f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/{GATEWAY_EVENTS_FILENAME}"
    events_payload = artifact_payloads[events_key]
    gateway_events: list[dict[str, Any]] = []
    try:
        for line in events_payload.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                return None
            gateway_events.append(dict(value))
    except (UnicodeError, json.JSONDecodeError):
        return None
    allowed_events = {
        "gateway_backend_configured",
        "compression_finished",
        "raw_suffix_images_released",
        "model_tool_validation_recovered",
        "response_committed",
        "response_replayed",
        "prompt_fingerprint_drift_accepted",
        "prompt_fingerprint_drift_rejected",
        "gateway_request_failed",
        "generation_stream_started",
        "generation_stream_progress",
        "generation_stream_completed",
        "generation_stream_failed",
        "generation_stream_queued",
    }
    observed_times: list[float] = []
    event_counts: Counter[str] = Counter()
    for event in gateway_events:
        name = event.get("event")
        observed_at = event.get("observed_at")
        if (
            not isinstance(name, str)
            or name not in allowed_events
            or not isinstance(observed_at, (int, float))
            or isinstance(observed_at, bool)
            or not math.isfinite(float(observed_at))
            or float(observed_at) <= 0
        ):
            return None
        observed_times.append(float(observed_at))
        event_counts[name] += 1
    failures = [
        event
        for event in gateway_events
        if event.get("event") == "gateway_request_failed"
    ]
    if (
        not gateway_events
        or any(right < left for left, right in zip(observed_times, observed_times[1:]))
        or gateway_events[0].get("event") != "gateway_backend_configured"
        or event_counts["gateway_backend_configured"] != 1
        or event_counts["compression_finished"] < 1
        or event_counts["response_committed"] < 1
        or len(failures) != 1
        or gateway_events[-1] != failures[0]
    ):
        return None
    gateway_failure = failures[0]
    gateway_traceback = gateway_failure.get("traceback")
    gateway_observed_raw = gateway_failure.get("observed_at")
    stream_timeout = (
        gateway_failure.get("error_type") == "RuntimeError"
        and gateway_failure.get("error_message")
        == "SGLang stream interrupted before any output tokens"
    )
    if (
        set(gateway_failure)
        != {"event", "observed_at", "error_type", "error_message", "traceback"}
        or (not stream_timeout and gateway_failure.get("error_type") != "TimeoutError")
        or (not stream_timeout and gateway_failure.get("error_message") != "timed out")
        or not isinstance(gateway_traceback, str)
        or not gateway_traceback.endswith(
            "RuntimeError: SGLang stream interrupted before any output tokens\n"
            if stream_timeout else "TimeoutError: timed out\n")
    ):
        return None
    gateway_trace_markers = (
        "shared_runtime/sglang_streaming.py",
        "in _post_streaming_generate\n",
        "TimeoutError: timed out\n",
        "The above exception was the direct cause of the following exception:",
        "shared_runtime/sglang_streaming.py",
        "in _post_streaming_generate\n",
        "RuntimeError: SGLang stream interrupted before any output tokens\n",
    ) if stream_timeout else (
        "in do_POST\n",
        "self.server.gateway.handle(request)",
        "in handle\n",
        "self.backend.generate(**generate_kwargs)",
        "in generate\n",
        "self._generator.generate_next(",
        "in generate_next\n",
        "in _generate_next_locked\n",
        "in materialize_pending_gpu_memory\n",
        "self.client.encode_recursive_memory_chain(",
        "in encode_recursive_memory_chain\n",
        "response = self._post(RECURSIVE_COMPRESSION_CHAIN_ROUTE, request.to_json())",
        "in _post\n",
        "with urllib.request.urlopen(",
        "TimeoutError: timed out\n",
    )
    marker_position = -1
    for marker in gateway_trace_markers:
        marker_position = gateway_traceback.find(marker, marker_position + 1)
        if marker_position < 0:
            return None

    if stream_timeout:
        starts = [e for e in gateway_events if e.get("event") == "generation_stream_started"]
        stream_failures = [e for e in gateway_events if e.get("event") == "generation_stream_failed"]
        if not starts or len(stream_failures) != 1:
            return None
        started, failed = starts[-1], stream_failures[0]
        rid = started.get("request_id")
        if (not isinstance(rid, str) or re.fullmatch(r"[A-Za-z0-9_-]+", rid) is None
            or failed.get("request_id") != rid
            or started.get("round_index") != event_counts["response_committed"] + 1
            or failed.get("round_index") != started.get("round_index")
            or failed.get("phase") != "generation" or started.get("phase") != "generation"
            or failed.get("episode_id") != started.get("episode_id")
            or failed.get("output_tokens") != 0 or failed.get("frames") != 0
            or failed.get("error") != gateway_failure["error_message"]
            or started.get("idle_timeout_seconds") != request_timeout
            or failed.get("idle_timeout_seconds") != request_timeout
            or failed["observed_at"] - started["observed_at"] < 0.95 * request_timeout
            or gateway_events[-2] != failed
            or any(e.get("request_id") == rid and e.get("event") in {
                "generation_stream_progress", "generation_stream_completed"} for e in gateway_events)):
            return None

    try:
        receipts, receipt_storage = _decode_dynamic_gateway_receipts(artifact_payloads)
    except ValueError:
        return None
    if not receipts:
        return None
    valid_receipts: list[Mapping[str, Any]] = []
    recovered_receipts: list[Mapping[str, Any]] = []
    for index, receipt in enumerate(receipts, start=1):
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("round_index") != index
            or not isinstance(receipt.get("elapsed_seconds"), (int, float))
            or isinstance(receipt.get("elapsed_seconds"), bool)
            or not math.isfinite(float(receipt["elapsed_seconds"]))
            or float(receipt["elapsed_seconds"]) <= 0
            or not isinstance(receipt.get("input_tokens"), int)
            or isinstance(receipt.get("input_tokens"), bool)
            or receipt["input_tokens"] <= 0
            or not isinstance(receipt.get("output_tokens"), int)
            or isinstance(receipt.get("output_tokens"), bool)
            or receipt["output_tokens"] <= 0
            or receipt.get("finished_on_eos") is not True
            or receipt.get("hit_generation_cap") is not False
            or not isinstance(receipt.get("complete_valid"), bool)
        ):
            return None
        (valid_receipts if receipt["complete_valid"] else recovered_receipts).append(
            receipt
        )
    input_tokens = agent_result.get("n_input_tokens")
    output_tokens = agent_result.get("n_output_tokens")
    if (
        not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or input_tokens <= 0
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens <= 0
        or not valid_receipts
        or receipts[-1].get("complete_valid") is not True
        or len(receipts) + int(stream_timeout) != event_counts["compression_finished"]
        or len(valid_receipts) != event_counts["response_committed"]
        or len(recovered_receipts) != event_counts["model_tool_validation_recovered"]
        or sum(int(receipt["input_tokens"]) for receipt in valid_receipts)
        != input_tokens
        or sum(int(receipt["output_tokens"]) for receipt in valid_receipts)
        != output_tokens
    ):
        return None

    reward_payload = artifact_payloads["verifier/reward.txt"]
    report_payload = artifact_payloads["verifier/ctrf.json"]
    try:
        reward_text = reward_payload.decode("ascii").strip()
        report = json.loads(report_payload)
    except (UnicodeError, json.JSONDecodeError):
        return None
    results = report.get("results") if isinstance(report, Mapping) else None
    summary = results.get("summary") if isinstance(results, Mapping) else None
    tests = results.get("tests") if isinstance(results, Mapping) else None
    count_fields = ("tests", "passed", "failed", "skipped", "pending", "other")
    test_status_counts = (
        Counter(str(test.get("status")) for test in tests if isinstance(test, Mapping))
        if isinstance(tests, list)
        else Counter()
    )
    if (
        re.fullmatch(r"0(?:\.0*)?", reward_text) is None
        or not isinstance(summary, Mapping)
        or not isinstance(tests, list)
        or any(
            not isinstance(summary.get(key), int)
            or isinstance(summary.get(key), bool)
            or summary[key] < 0
            for key in count_fields
        )
        or summary["tests"] <= 0
        or sum(int(summary[key]) for key in count_fields[1:]) != summary["tests"]
        or len(tests) != summary["tests"]
        or any(
            not isinstance(test, Mapping)
            or not isinstance(test.get("name"), str)
            or not test["name"]
            or test.get("status")
            not in {"passed", "failed", "skipped", "pending", "other"}
            for test in tests
        )
        or any(
            test_status_counts[status] != summary[status]
            for status in ("passed", "failed", "skipped", "pending", "other")
        )
    ):
        return None

    trial_started = _parse_time(trial.get("started_at"))
    agent_started = _parse_time(agent_timing.get("started_at"))
    agent_finished = _parse_time(agent_timing.get("finished_at"))
    exception_at = _parse_time(exception.get("occurred_at"))
    verifier_started = _parse_time(verifier_timing.get("started_at"))
    verifier_finished = _parse_time(verifier_timing.get("finished_at"))
    trial_finished = _parse_time(trial.get("finished_at"))
    report_start_raw = summary.get("start")
    report_stop_raw = summary.get("stop")
    if (
        any(
            value is None
            for value in (
                trial_started,
                agent_started,
                agent_finished,
                exception_at,
                verifier_started,
                verifier_finished,
                trial_finished,
            )
        )
        or not isinstance(gateway_observed_raw, (int, float))
        or isinstance(gateway_observed_raw, bool)
        or not math.isfinite(float(gateway_observed_raw))
        or not isinstance(report_start_raw, (int, float))
        or isinstance(report_start_raw, bool)
        or not math.isfinite(float(report_start_raw))
        or not isinstance(report_stop_raw, (int, float))
        or isinstance(report_stop_raw, bool)
        or not math.isfinite(float(report_stop_raw))
    ):
        return None
    try:
        gateway_observed = datetime.fromtimestamp(float(gateway_observed_raw), UTC)
        report_started = datetime.fromtimestamp(float(report_start_raw), UTC)
        report_finished = datetime.fromtimestamp(float(report_stop_raw), UTC)
    except (OverflowError, OSError, ValueError):
        return None
    assert trial_started is not None
    assert agent_started is not None
    assert agent_finished is not None
    assert exception_at is not None
    assert verifier_started is not None
    assert verifier_finished is not None
    assert trial_finished is not None
    if not (
        trial_started
        <= agent_started
        < gateway_observed
        < agent_finished
        <= exception_at
        < verifier_started
        <= report_started
        < report_finished
        <= verifier_finished
        <= trial_finished
    ):
        return None

    clean_receipts = _clean_proot_incident_receipts(trial_path, trial)
    if clean_receipts is None:
        return None
    authority_hashes = {
        key: hashlib.sha256(artifact_payloads[key]).hexdigest()
        for key in sorted(artifact_keys)
    }
    return {
        "incident": _CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT,
        "classifier": ("codex_sglang_stream_first_token_timeout_v1" if stream_timeout
                       else _CODEX_GATEWAY_REQUEST_TIMEOUT_CLASSIFIER),
        "controller_artifact_directory": CONTROLLER_ARTIFACT_DIRNAME,
        "controller_directory_mode": oct(PRIVATE_DIR_MODE),
        "exception_type": exception_type,
        "agent_exit_code": 1,
        "gateway_error_type": gateway_failure["error_type"],
        "gateway_error_message": gateway_failure["error_message"],
        "gateway_failure_count": 1,
        "gateway_receipt_storage": receipt_storage,
        "gateway_failure_observed_at": gateway_observed.isoformat(),
        "configured_request_timeout_seconds": float(request_timeout),
        "completed_round_receipt_count": len(receipts),
        "committed_response_count": len(valid_receipts),
        "recovered_invalid_response_count": len(recovered_receipts),
        "codex_diagnostic_line_count": diagnostic_lines,
        "n_input_tokens": input_tokens,
        "n_output_tokens": output_tokens,
        "verifier_reward": 0.0,
        "verifier_tests": int(summary["tests"]),
        "verifier_passed": int(summary["passed"]),
        "verifier_failed": int(summary["failed"]),
        "authority_artifact_sha256": authority_hashes,
        "agent_result_sha256": _digest(agent_result),
        "exception_evidence": exception_evidence,
        "clean_proot_receipts": clean_receipts,
    }


def _archive_retryable_sglang_harbor_job(
    claim: ClaimedCell,
    *,
    trial_path: Path,
    failure: Mapping[str, Any],
) -> Path:
    """Atomically remove a failed Harbor job from the resumable live path."""

    source = claim.paths.harbor_job
    if not source.is_dir() or source.is_symlink():
        raise PoolError("retryable SGLang trial has no safe Harbor job directory")
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / SGLANG_AUTO_ARCHIVE_NAME
    if archive.exists() or archive.is_symlink():
        raise PoolError(f"retryable SGLang Harbor archive already exists: {archive}")
    try:
        relative_trial = str(Path(trial_path).relative_to(source))
    except ValueError as exc:
        raise PoolError("retryable SGLang trial is outside its Harbor job") from exc
    receipt = {
        "schema": SGLANG_ARCHIVE_RECEIPT_SCHEMA,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "trial_path": relative_trial,
        "failure": dict(failure),
        "archived_at": _utc_now(),
    }
    # Put the receipt inside the source before rename, so a successful rename
    # can never leave an unaudited archive. Both paths are on the same cell FS.
    _atomic_write_json(source / SGLANG_ARCHIVE_RECEIPT_NAME, receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive retryable SGLang Harbor job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _archive_proot_infrastructure_harbor_job(
    claim: ClaimedCell,
    *,
    trial_path: Path,
    failure: Mapping[str, Any],
    archive_name: str = "harbor-job-committed-proot-infrastructure",
    detection_phase: str = "committed_repair",
) -> Path:
    """Atomically retire a complete Harbor job containing a PRoot failure."""

    source = claim.paths.harbor_job
    if not source.is_dir() or source.is_symlink():
        raise PoolError("PRoot infrastructure trial has no safe Harbor job directory")
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / archive_name
    if archive.exists() or archive.is_symlink():
        raise PoolError(
            f"PRoot infrastructure Harbor archive already exists: {archive}"
        )
    try:
        relative_trial = str(Path(trial_path).relative_to(source))
    except ValueError as exc:
        raise PoolError("PRoot infrastructure trial is outside its Harbor job") from exc
    receipt = {
        "schema": PROOT_ARCHIVE_RECEIPT_SCHEMA,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "trial_path": relative_trial,
        "failure": dict(failure),
        "detection_phase": detection_phase,
        "archived_at": _utc_now(),
    }
    # The receipt moves with the job, so every visible archive is self-auditing.
    _atomic_write_json(source / PROOT_ARCHIVE_RECEIPT_NAME, receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive PRoot infrastructure Harbor job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _archive_external_interruption_harbor_job(
    claim: ClaimedCell,
    interruption: ExternalInterruption,
    *,
    archive_name: str = "harbor-job-external-interruption",
) -> Path | None:
    """Remove a signal-cancelled Harbor job before the cell is claimable.

    Harbor writes a terminal ``CancelledError`` while shutting down. Leaving
    that deterministic job path live makes the next execution adopt the stale
    terminal result instead of starting a fresh trial. The rename is atomic
    on the cell filesystem and happens while the claim flock is still held.
    """

    source = claim.paths.harbor_job
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / archive_name
    if not source.exists():
        if archive.is_dir() and not archive.is_symlink():
            return archive
        return None
    if not source.is_dir() or source.is_symlink():
        raise PoolError("external interruption has no safe Harbor job directory")
    if archive.exists() or archive.is_symlink():
        raise PoolError(
            f"external interruption Harbor archive already exists: {archive}"
        )
    receipt = {
        "schema": "terminal_bench_external_interruption_harbor_archive_v1",
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "signal": interruption.signum,
        "archived_at": _utc_now(),
    }
    _atomic_write_json(source / "external-interruption.json", receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive externally interrupted Harbor job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _archive_agent_timeout_harbor_job(
    claim: ClaimedCell,
    *,
    trial_path: Path,
    failure: Mapping[str, Any],
) -> Path:
    """Atomically retire one terminal Harbor job produced by agent timeout."""

    source = claim.paths.harbor_job
    if not source.is_dir() or source.is_symlink():
        raise PoolError("agent-timeout trial has no safe Harbor job directory")
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / "harbor-job-committed-agent-timeout"
    if archive.exists() or archive.is_symlink():
        raise PoolError(f"agent-timeout Harbor archive already exists: {archive}")
    try:
        relative_trial = str(Path(trial_path).relative_to(source))
    except ValueError as exc:
        raise PoolError("agent-timeout trial is outside its Harbor job") from exc
    receipt = {
        "schema": "terminal_bench_agent_timeout_harbor_archive_v1",
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "trial_path": relative_trial,
        "failure": dict(failure),
        "archived_at": _utc_now(),
    }
    _atomic_write_json(source / "agent-timeout.json", receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive agent-timeout Harbor job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _archive_verifier_timeout_harbor_job(
    claim: ClaimedCell,
    *,
    trial_path: Path,
    failure: Mapping[str, Any],
) -> Path:
    """Atomically retire one terminal Harbor verifier-timeout job."""

    source = claim.paths.harbor_job
    if not source.is_dir() or source.is_symlink():
        raise PoolError("verifier-timeout trial has no safe Harbor job directory")
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / "harbor-job-committed-verifier-timeout"
    if archive.exists() or archive.is_symlink():
        raise PoolError(f"verifier-timeout Harbor archive already exists: {archive}")
    try:
        relative_trial = str(Path(trial_path).relative_to(source))
    except ValueError as exc:
        raise PoolError("verifier-timeout trial is outside its Harbor job") from exc
    receipt = {
        "schema": "terminal_bench_verifier_timeout_harbor_archive_v1",
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "trial_path": relative_trial,
        "failure": dict(failure),
        "archived_at": _utc_now(),
    }
    _atomic_write_json(source / "verifier-timeout.json", receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive verifier-timeout Harbor job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _archive_codex_host_rg_harbor_job(
    claim: ClaimedCell,
    *,
    trial_path: Path,
    evidence: Mapping[str, Any],
) -> Path:
    """Atomically retire one Harbor job from the missing-host-rg incident."""

    source = claim.paths.harbor_job
    if not source.is_dir() or source.is_symlink():
        raise PoolError("Codex host-rg incident has no safe Harbor job directory")
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / "harbor-job-committed-codex-host-rg-incident"
    if archive.exists() or archive.is_symlink():
        raise PoolError(f"Codex host-rg Harbor archive already exists: {archive}")
    try:
        relative_trial = str(Path(trial_path).relative_to(source))
    except ValueError as exc:
        raise PoolError("Codex host-rg trial is outside its Harbor job") from exc
    receipt = {
        "schema": "terminal_bench_codex_host_rg_harbor_archive_v1",
        "incident": _CODEX_HOST_RG_INCIDENT,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "trial_path": relative_trial,
        "evidence": dict(evidence),
        "official_attempt_consumed_by_repair": False,
        "archived_at": _utc_now(),
    }
    _atomic_write_json(source / "codex-host-rg-incident.json", receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive Codex host-rg Harbor job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _archive_codex_gateway_timeout_harbor_job(
    claim: ClaimedCell,
    *,
    trial_path: Path,
    evidence: Mapping[str, Any],
) -> Path:
    """Atomically retire one exact recursive-memory gateway-timeout job."""

    source = claim.paths.harbor_job
    if not source.is_dir() or source.is_symlink():
        raise PoolError("Codex gateway-timeout incident has no safe Harbor job")
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / "harbor-job-committed-codex-gateway-timeout-incident"
    if archive.exists() or archive.is_symlink():
        raise PoolError(f"Codex gateway-timeout archive already exists: {archive}")
    try:
        relative_trial = str(Path(trial_path).relative_to(source))
    except ValueError as exc:
        raise PoolError(
            "Codex gateway-timeout trial is outside its Harbor job"
        ) from exc
    receipt = {
        "schema": "terminal_bench_codex_gateway_timeout_harbor_archive_v1",
        "incident": _CODEX_GATEWAY_TIMEOUT_INCIDENT,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "task_ref": claim.cell.task_ref,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "trial_path": relative_trial,
        "evidence": dict(evidence),
        "official_attempt_consumed_by_repair": False,
        "archived_at": _utc_now(),
    }
    _atomic_write_json(source / "codex-gateway-timeout-incident.json", receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive Codex gateway-timeout Harbor job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _archive_codex_gateway_request_timeout_harbor_job(
    claim: ClaimedCell,
    *,
    trial_path: Path,
    evidence: Mapping[str, Any],
) -> Path:
    """Atomically retire a host-controller-attributed gateway timeout job."""

    source = claim.paths.harbor_job
    if not source.is_dir() or source.is_symlink():
        raise PoolError("Codex gateway request timeout has no safe Harbor job")
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / _CODEX_GATEWAY_REQUEST_TIMEOUT_HARBOR_ARCHIVE
    if archive.exists() or archive.is_symlink():
        raise PoolError(
            f"Codex gateway request timeout archive already exists: {archive}"
        )
    try:
        relative_trial = str(Path(trial_path).relative_to(source))
    except ValueError as exc:
        raise PoolError(
            "Codex gateway request timeout trial is outside its Harbor job"
        ) from exc
    receipt = {
        "schema": "terminal_bench_codex_gateway_request_timeout_harbor_archive_v1",
        "incident": _CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "task_ref": claim.cell.task_ref,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "trial_path": relative_trial,
        "evidence": dict(evidence),
        "official_attempt_consumed_by_repair": False,
        "archived_at": _utc_now(),
    }
    # The authority receipt moves in the same rename as the complete Harbor
    # job, so a visible archive is never detached from its classification.
    _atomic_write_json(source / _CODEX_GATEWAY_REQUEST_TIMEOUT_HARBOR_RECEIPT, receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive Codex gateway request timeout job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _archive_proot_localhost_harbor_job(
    claim: ClaimedCell,
    *,
    trial_path: Path,
    evidence: Mapping[str, Any],
) -> Path:
    """Atomically retire one Harbor job from the diagnosed PRoot incident."""

    source = claim.paths.harbor_job
    if not source.is_dir() or source.is_symlink():
        raise PoolError("PRoot localhost incident has no safe Harbor job directory")
    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    archive = attempt_dir / "harbor-job-committed-proot-localhost-incident"
    if archive.exists() or archive.is_symlink():
        raise PoolError(f"PRoot localhost Harbor archive already exists: {archive}")
    try:
        relative_trial = str(Path(trial_path).relative_to(source))
    except ValueError as exc:
        raise PoolError("PRoot localhost trial is outside its Harbor job") from exc
    receipt = {
        "schema": "terminal_bench_proot_localhost_harbor_archive_v1",
        "incident": _PROOT_LOCALHOST_INCIDENT,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "official_attempt_index": claim.cell.official_attempt_index,
        "execution_attempt": claim.execution_attempt,
        "trial_path": relative_trial,
        "evidence": dict(evidence),
        "official_attempt_consumed_by_repair": False,
        "archived_at": _utc_now(),
    }
    _atomic_write_json(source / "proot-localhost-incident.json", receipt)
    try:
        os.replace(source, archive)
    except OSError as exc:
        raise PoolError("could not archive PRoot localhost Harbor job") from exc
    _fsync_directory(source.parent)
    _fsync_directory(archive.parent)
    return archive


def _raise_if_retryable_sglang_trial(
    claim: ClaimedCell,
    trial_path: Path,
    trial: Mapping[str, Any],
) -> None:
    failure = _retryable_sglang_trial_failure(trial)
    if (
        failure is None
        or trial.get("verifier_result") is not None
        or bool(_result_rewards(trial))
    ):
        return
    archive = _archive_retryable_sglang_harbor_job(
        claim,
        trial_path=trial_path,
        failure=failure,
    )
    raise RetryableSGLangTrialError(failure, archive)


def _raise_if_retryable_proot_trial(
    claim: ClaimedCell,
    trial_path: Path,
    trial: Mapping[str, Any],
) -> None:
    """Retire only an unverified, host-proven local PRoot failure.

    A legacy RPC failure needs the exact local ``_wire_exec`` traceback.  New
    runtime/package/guest-utility failures instead need a host-owned receipt
    whose schema, environment role, and session agree with the pinned Harbor
    trial.  In all cases the trial must have reached neither a verifier result
    nor any reward.  Model-authored text alone can therefore never request
    another official execution.
    """

    failure = _local_proot_infrastructure_trial_failure(trial_path, trial)
    if failure is None:
        return
    if (
        _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
        or trial.get("verifier_result") is not None
        or _result_rewards(trial)
    ):
        return
    archive = _archive_proot_infrastructure_harbor_job(
        claim,
        trial_path=trial_path,
        failure=failure,
        archive_name=PROOT_AUTO_ARCHIVE_NAME,
        detection_phase=PROOT_AUTO_DETECTION_PHASE,
    )
    raise RetryablePRootTrialError(failure, archive)


def _cell_result_from_trial(
    spec: PoolSpec,
    claim: ClaimedCell,
    trial_path: Path,
    trial: Mapping[str, Any],
    *,
    worker_id: str,
    device_label: str,
) -> dict[str, Any]:
    rewards = _result_rewards(trial)
    verifier_rewards = dict(rewards)
    exception = trial.get("exception_info")
    has_exception = isinstance(exception, Mapping) and bool(exception)
    reward = rewards.get("reward")
    cleanup_warning = _local_cleanup_warning(trial_path, trial)
    cleanup_only_exception = cleanup_warning is not None and (
        _exception_is_local_cleanup_only(trial)
    )
    verified_agent_timeout = _verified_reward_after_agent_timeout(trial_path, trial)
    verified_agent_exit = _verified_success_after_agent_sigterm(trial_path, trial)
    from terminal_bench_eval.verifier_checkpoint import pending_record
    verification_pending = pending_record(trial_path)
    has_host_failure = any(
        classifier(trial_path, trial) is not None
        for classifier in (_runtime_receipt_failure, _package_bridge_receipt_failure)
    )
    scoreable = reward is not None and (
        (not has_exception and not has_host_failure)
        or cleanup_only_exception
        or verified_agent_timeout is not None
        or verified_agent_exit is not None
    )
    if scoreable:
        status = "completed"
        outcome = "passed" if reward > 0 else "failed"
    else:
        # Only a completed verifier is allowed to manufacture a model score.
        # An agent/environment exception, a missing primary reward, or a
        # non-finite reward is an unscoreable trial failure, never implicit 0.
        # Keep the historical terminal-state vocabulary so explicit repair
        # tools can still recognize pre-gate artifacts.  Empty ``rewards`` and
        # the aggregate scoreability gate, rather than a synthetic pool error,
        # are the authority that prevents this slot from becoming model 0.
        status = "completed"
        rewards = {}
        if has_exception:
            outcome = "errored"
        else:
            outcome = "completed_without_reward"
    if verification_pending is not None:
        scoreable = False
        rewards = {}
        outcome = "verification_pending"
    agent_result = trial.get("agent_result")
    agent_payload = agent_result if isinstance(agent_result, Mapping) else {}
    result = {
        "schema": CELL_RESULT_SCHEMA,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "task_ref": claim.cell.task_ref,
        "source": claim.cell.source,
        "official_attempt_index": claim.cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
        "status": status,
        "outcome": outcome,
        "execution_attempt": claim.execution_attempt,
        "agent_time_budget": _execution_agent_time_budget(spec, claim),
        "worker_id": worker_id,
        "device": device_label,
        "harbor_job_path": str(claim.paths.harbor_job.relative_to(spec.paths.root)),
        "trial_path": str(trial_path.relative_to(spec.paths.root)),
        "trial_name": trial.get("trial_name"),
        "rewards": rewards,
        "rejected_verifier_rewards": (
            verifier_rewards if verifier_rewards and not scoreable else None
        ),
        "exception_info": dict(exception) if isinstance(exception, Mapping) else None,
        "infrastructure_cleanup_warning": cleanup_warning,
        "verified_agent_timeout_warning": verified_agent_timeout,
        "verified_agent_exit_warning": verified_agent_exit,
        "n_input_tokens": int(agent_payload.get("n_input_tokens") or 0),
        "n_output_tokens": int(agent_payload.get("n_output_tokens") or 0),
        "started_at": trial.get("started_at"),
        "finished_at": trial.get("finished_at"),
        "committed_at": _utc_now(),
    }
    from terminal_bench_eval.agent_timeout_policy import project
    return project(spec, claim, trial_path, trial, result)


def _execute_harbor(
    spec: PoolSpec,
    claim: ClaimedCell,
    *,
    harbor_bin: Path,
    sglang_url: str,
    worker_id: str,
    device_label: str,
    prewarm_task_cache: bool,
    environment_backend: str = "frozen",
) -> dict[str, Any]:
    _ensure_job_config(
        spec,
        claim,
        sglang_url=sglang_url,
        environment_backend=environment_backend,
    )
    existing = _terminal_harbor_trial(spec, claim)
    if existing is not None:
        path, result = existing
        from terminal_bench_eval.verifier_checkpoint import pending_record
        from terminal_bench_eval.agent_timeout_policy import applies
        if pending_record(path) is None and not applies(spec, result):
            _raise_if_retryable_sglang_trial(claim, path, result)
            _raise_if_retryable_proot_trial(claim, path, result)
        return _cell_result_from_trial(
            spec,
            claim,
            path,
            result,
            worker_id=worker_id,
            device_label=device_label,
        )
    if prewarm_task_cache:
        _prewarm_task_cache(claim.cell.task_config)
    attempt_dir = claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    log_path = attempt_dir / "harbor.log"
    if (claim.paths.harbor_job / "config.json").is_file():
        observed_job_config = _read_json(claim.paths.harbor_job / "config.json")
        expected_job_config = _read_json(claim.paths.job_config)
        if not _harbor_resume_config_matches_contract(
            observed_job_config, expected_job_config
        ):
            raise PoolError("existing Harbor job config violates the cell contract")
        command = [
            *_harbor_command_prefix(spec, harbor_bin),
            "job",
            "resume",
            "--job-path",
            str(claim.paths.harbor_job),
            "--yes",
        ]
        action = "resume"
    elif claim.paths.harbor_job.exists():
        raise PoolError("Harbor job directory exists without a resumable config")
    else:
        command = [
            *_harbor_command_prefix(spec, harbor_bin),
            "run",
            "--config",
            str(claim.paths.job_config),
            "--job-name",
            "harbor-job",
            "--jobs-dir",
            str(claim.paths.root),
            "--n-attempts",
            "1",
            "--n-concurrent",
            "1",
            "--max-retries",
            "0",
            "--yes",
            "--quiet",
        ]
        action = "run"
    run_record = {
        "schema": RUN_RECORD_SCHEMA,
        "cell_id": claim.cell.cell_id,
        "execution_attempt": claim.execution_attempt,
        "status": "running",
        "action": action,
        "command": command,
        "agent_time_budget": _agent_time_budget(spec),
        "environment_backend": environment_backend,
        "started_at": _utc_now(),
    }
    _atomic_write_json(attempt_dir / "run.json", run_record)
    return_code = _run_subprocess_with_claim(
        command,
        claim_descriptor=claim.descriptor,
        cwd=PROJECT_ROOT,
        log_path=log_path,
    )
    _atomic_write_json(
        attempt_dir / "run.json",
        {
            **run_record,
            "status": "exited",
            "return_code": return_code,
            "finished_at": _utc_now(),
        },
    )
    terminal = _terminal_harbor_trial(spec, claim)
    if terminal is None:
        raise PoolError(
            f"Harbor exited {return_code} without one terminal trial result"
        )
    path, result = terminal
    from terminal_bench_eval.verifier_checkpoint import pending_record
    from terminal_bench_eval.agent_timeout_policy import applies
    if pending_record(path) is None and not applies(spec, result):
        _raise_if_retryable_sglang_trial(claim, path, result)
        _raise_if_retryable_proot_trial(claim, path, result)
    return _cell_result_from_trial(
        spec,
        claim,
        path,
        result,
        worker_id=worker_id,
        device_label=device_label,
    )


def _commit_cell_result(
    spec: PoolSpec, claim: ClaimedCell, result: Mapping[str, Any]
) -> None:
    now = _utc_now()
    result_status = result.get("status")
    if result_status not in {"completed", "pool_error"}:
        raise PoolError("cell result has no valid terminal status")
    state = {
        "schema": CELL_STATE_SCHEMA,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "task_ref": claim.cell.task_ref,
        "official_attempt_index": claim.cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
        "status": result_status,
        "outcome": result.get("outcome"),
        "execution_attempt": claim.execution_attempt,
        "agent_time_budget": dict(
            result.get("agent_time_budget")
            if isinstance(result.get("agent_time_budget"), Mapping)
            else _execution_agent_time_budget(spec, claim)
        ),
        "updated_at": now,
        "finished_at": now,
    }
    dispatch = dict(claim.dispatch)
    dispatch.update(
        {
            "status": "terminal",
            "outcome": result.get("outcome"),
            "updated_at": now,
        }
    )
    verifier_replay = result.get("verifier_replay")
    if isinstance(verifier_replay, Mapping):
        state["verifier_replay"] = dict(verifier_replay)
        dispatch["verifier_replay"] = dict(verifier_replay)
    dispatch.pop("claim", None)
    _atomic_write_json(claim.paths.state, state)
    _atomic_write_json(claim.paths.dispatch, dispatch)
    _atomic_write_json(claim.paths.result, dict(result))


def _record_worker_exception(
    spec: PoolSpec,
    claim: ClaimedCell,
    exc: Exception,
    *,
    retry_backoff_seconds: float,
    max_worker_exceptions: int,
) -> int:
    """Commit an unknown pool/harness exception without another model draw.

    ``max_worker_exceptions`` remains in the public CLI for compatibility with
    existing launch scripts, but unknown exceptions are deliberately not a
    retry class. Only a separately classified, host-proven infrastructure
    failure may consume the immutable infrastructure retry budget.
    """

    del max_worker_exceptions, retry_backoff_seconds
    previous, _ = _read_worker_exception(claim.paths, claim.cell)
    count = previous + 1
    now = _utc_now()
    _atomic_write_json(
        claim.paths.exception,
        {
            "schema": WORKER_EXCEPTION_SCHEMA,
            "cell_id": claim.cell.cell_id,
            "task_name": claim.cell.task_name,
            "official_attempt_index": claim.cell.official_attempt_index,
            "execution_attempt": claim.execution_attempt,
            "worker_exception_count": count,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=40),
            "recorded_at": now,
        },
    )
    _retry_exhausted_result(
        spec,
        claim.cell,
        claim.paths,
        execution_attempt=claim.execution_attempt,
        exception_type=type(exc).__name__,
        worker_exception_count=count,
    )
    return count


def _record_task_cache_transport_retry(
    spec: PoolSpec,
    claim: ClaimedCell,
    exc: RetryableTaskCacheTransportError,
    *,
    retry_backoff_seconds: float,
) -> tuple[int, str]:
    """Return one proven pre-model storage outage to the queue indefinitely."""

    paths = claim.paths
    if paths.result.exists():
        raise PoolError("task-cache transport retry cannot replace a terminal result")
    if paths.harbor_job.exists():
        raise PoolError("task-cache transport retry started after Harbor materialized")
    payload: dict[str, Any] = {
        "schema": TASK_CACHE_TRANSPORT_SCHEMA,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "task_ref": claim.cell.task_ref,
        "official_attempt_index": claim.cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
        "retry_count": 0,
        "history": [],
    }
    if paths.task_cache_transport.is_file():
        observed = _read_json(paths.task_cache_transport, max_bytes=MAX_ARTIFACT_BYTES)
        for key in (
            "schema",
            "cell_id",
            "task_name",
            "task_ref",
            "official_attempt_index",
            "config_contract_hash",
        ):
            if observed.get(key) != payload[key]:
                raise PoolError("task-cache transport retry journal is inconsistent")
        if not isinstance(observed.get("history"), list):
            raise PoolError("task-cache transport retry journal has invalid history")
        payload = observed
    previous = payload.get("retry_count")
    if not isinstance(previous, int) or isinstance(previous, bool) or previous < 0:
        raise PoolError("task-cache transport retry journal has invalid count")
    count = previous + 1
    delay = min(
        300.0,
        max(1.0, float(retry_backoff_seconds)) * (2 ** min(count - 1, 5)),
    )
    retry_at = _future_time(delay)
    now = _utc_now()
    event = {
        "execution_attempt": claim.execution_attempt,
        "retry_count": count,
        "retry_not_before": retry_at,
        "failure": dict(exc.failure),
        "recorded_at": now,
    }
    payload["retry_count"] = count
    payload["last_failure"] = dict(exc.failure)
    payload["retry_not_before"] = retry_at
    payload["updated_at"] = now
    payload["history"].append(event)
    _atomic_write_json(paths.task_cache_transport, payload)

    attempt_dir = _ensure_dir(
        paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    _atomic_write_json(
        attempt_dir / "task-cache-transport.json",
        {
            "schema": TASK_CACHE_TRANSPORT_SCHEMA,
            "cell_id": claim.cell.cell_id,
            "execution_attempt": claim.execution_attempt,
            "retry_count": count,
            "retry_not_before": retry_at,
            "failure": dict(exc.failure),
            "recorded_at": now,
        },
    )
    run_path = attempt_dir / "run.json"
    if run_path.is_file():
        run = _read_json(run_path, max_bytes=1024 * 1024)
        run.update(
            {
                "status": "task_cache_transport_requeued",
                "task_cache_transport_retry_count": count,
                "retry_not_before": retry_at,
                "finished_at": now,
            }
        )
        _atomic_write_json(run_path, run)

    dispatch = dict(claim.dispatch)
    dispatch.update(
        {
            "status": "queued",
            "task_cache_transport_retry_count": count,
            "last_task_cache_transport_failure": dict(exc.failure),
            "retry_not_before": retry_at,
            "updated_at": now,
        }
    )
    for key in ("claim", "outcome", "harbor_job_path"):
        dispatch.pop(key, None)
    state = {
        "schema": CELL_STATE_SCHEMA,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "task_ref": claim.cell.task_ref,
        "official_attempt_index": claim.cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
        "status": "queued",
        "execution_attempt": claim.execution_attempt,
        "task_cache_transport_retry_count": count,
        "last_task_cache_transport_failure": dict(exc.failure),
        "retry_not_before": retry_at,
        "updated_at": now,
    }
    paths.exception.unlink(missing_ok=True)
    _atomic_write_json(paths.dispatch, dispatch)
    _atomic_write_json(paths.state, state)
    return count, retry_at


def _record_bounded_proot_infrastructure_retry(
    spec: PoolSpec,
    claim: ClaimedCell,
    exc: RetryableSGLangTrialError | RetryablePRootTrialError,
    *,
    retry_backoff_seconds: float,
    max_proot_infrastructure_retries: int,
) -> tuple[int, int, int, int, bool]:
    """Record one host-proven failure against one shared immutable budget.

    Both local PRoot and local SGLang failures have already had their Harbor
    jobs archived before this function runs. The legacy CLI/JSON field names
    retain ``proot`` for launch-script compatibility, but the counter is now
    shared by both strict infrastructure kinds. The default two permits two
    total requeues; a third matching failure becomes an explicit pool error.
    Unknown worker exceptions never enter this budget.
    """

    if max_proot_infrastructure_retries < 0:
        raise PoolError("max PRoot infrastructure retries must be nonnegative")
    (
        previous_infrastructure_count,
        history,
        previous_proot_failures,
        previous_proot_requeues,
        recorded_max_proot_retries,
        _last_proot_failure,
    ) = _read_infrastructure_retry(claim.paths, claim.cell)
    effective_max_proot_retries = (
        recorded_max_proot_retries
        if recorded_max_proot_retries is not None
        else max_proot_infrastructure_retries
    )
    # Older journals counted SGLang only in the overall counter. Fold those
    # entries into the shared strict budget without forgetting an exhausted
    # PRoot failure (which was counted as a failure but not a requeue).
    legacy_non_proot = max(0, previous_infrastructure_count - previous_proot_failures)
    previous_strict_failures = previous_proot_failures + legacy_non_proot
    previous_strict_requeues = previous_proot_requeues + legacy_non_proot
    infrastructure_count = previous_infrastructure_count + 1
    proot_failure_count = previous_strict_failures + 1
    exhausted = proot_failure_count > effective_max_proot_retries
    proot_requeue_count = previous_strict_requeues + (0 if exhausted else 1)
    now = _utc_now()
    archive_path = str(exc.archive_path.relative_to(spec.paths.root))
    history.append(
        {
            "execution_attempt": claim.execution_attempt,
            "failure": dict(exc.failure),
            "harbor_archive_path": archive_path,
            "retry_budget": {
                "policy": "shared_strict_infrastructure_requeues_v1",
                "max_requeues": effective_max_proot_retries,
                "failure_count": proot_failure_count,
                "requeue_count": proot_requeue_count,
                "exhausted": exhausted,
            },
            "recorded_at": now,
        }
    )

    # This is the authoritative strict-infrastructure budget journal. It is durable before the
    # queued/terminal projection, so a replacement claimant can reconcile a
    # worker death in the narrow window between these writes.
    _atomic_write_json(
        claim.paths.infrastructure,
        {
            "schema": "terminal_bench_pool_infrastructure_retry_v1",
            "cell_id": claim.cell.cell_id,
            "task_name": claim.cell.task_name,
            "official_attempt_index": claim.cell.official_attempt_index,
            "infrastructure_retry_count": infrastructure_count,
            "proot_infrastructure_failure_count": proot_failure_count,
            "proot_infrastructure_requeue_count": proot_requeue_count,
            "max_proot_infrastructure_retries": effective_max_proot_retries,
            "strict_infrastructure_failure_count": proot_failure_count,
            "strict_infrastructure_requeue_count": proot_requeue_count,
            "max_strict_infrastructure_retries": effective_max_proot_retries,
            "last_proot_infrastructure_failure": dict(exc.failure),
            "history": history,
            "updated_at": now,
        },
    )

    if exhausted:
        _proot_retry_exhausted_result(
            spec,
            claim.cell,
            claim.paths,
            execution_attempt=claim.execution_attempt,
            proot_failure_count=proot_failure_count,
            proot_requeue_count=proot_requeue_count,
            max_proot_retries=effective_max_proot_retries,
            failure=exc.failure,
        )
        return (
            infrastructure_count,
            proot_failure_count,
            proot_requeue_count,
            effective_max_proot_retries,
            True,
        )

    dispatch = dict(claim.dispatch)
    dispatch.update(
        {
            "status": "queued",
            "worker_exception_count": int(
                claim.dispatch.get("worker_exception_count") or 0
            ),
            "infrastructure_retry_count": infrastructure_count,
            "proot_infrastructure_failure_count": proot_failure_count,
            "proot_infrastructure_requeue_count": proot_requeue_count,
            "max_proot_infrastructure_retries": effective_max_proot_retries,
            "strict_infrastructure_failure_count": proot_failure_count,
            "strict_infrastructure_requeue_count": proot_requeue_count,
            "max_strict_infrastructure_retries": effective_max_proot_retries,
            "last_infrastructure_failure": dict(exc.failure),
            "last_infrastructure_archive": archive_path,
            "retry_not_before": _future_time(retry_backoff_seconds),
            "updated_at": now,
        }
    )
    for key in ("claim", "outcome", "harbor_job_path"):
        dispatch.pop(key, None)
    _atomic_write_json(claim.paths.dispatch, dispatch)
    _atomic_write_json(
        claim.paths.state,
        {
            "schema": CELL_STATE_SCHEMA,
            "cell_id": claim.cell.cell_id,
            "task_name": claim.cell.task_name,
            "task_ref": claim.cell.task_ref,
            "official_attempt_index": claim.cell.official_attempt_index,
            "config_contract_hash": spec.contract_hash,
            "status": "queued",
            "execution_attempt": claim.execution_attempt,
            "worker_exception_count": int(
                claim.dispatch.get("worker_exception_count") or 0
            ),
            "infrastructure_retry_count": infrastructure_count,
            "proot_infrastructure_failure_count": proot_failure_count,
            "proot_infrastructure_requeue_count": proot_requeue_count,
            "max_proot_infrastructure_retries": effective_max_proot_retries,
            "strict_infrastructure_failure_count": proot_failure_count,
            "strict_infrastructure_requeue_count": proot_requeue_count,
            "max_strict_infrastructure_retries": effective_max_proot_retries,
            "last_infrastructure_failure": dict(exc.failure),
            "last_infrastructure_archive": archive_path,
            "updated_at": now,
        },
    )
    return (
        infrastructure_count,
        proot_failure_count,
        proot_requeue_count,
        effective_max_proot_retries,
        False,
    )


def _validated_unjournaled_proot_archive(
    spec: PoolSpec,
    cell: CellSpec,
    paths: CellPaths,
    archive: Path,
) -> tuple[int, dict[str, Any]]:
    """Validate an auto-retry archive as strictly as the precommit path."""

    if archive.is_symlink() or not archive.is_dir():
        raise PoolError(f"unsafe PRoot retry archive: {archive}")
    match = re.fullmatch(r"execution-([0-9]+)", archive.parent.name)
    if match is None:
        raise PoolError(f"PRoot retry archive has no execution owner: {archive}")
    execution = int(match.group(1))
    if execution < 1 or archive.parent.name != f"execution-{execution:04d}":
        raise PoolError(f"PRoot retry archive execution is not canonical: {archive}")
    receipt_path = archive / PROOT_ARCHIVE_RECEIPT_NAME
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise PoolError(f"PRoot retry archive has no safe receipt: {archive}")
    receipt = _read_json(receipt_path, max_bytes=1024 * 1024)
    failure = receipt.get("failure")
    relative_trial = receipt.get("trial_path")
    if (
        receipt.get("schema") != PROOT_ARCHIVE_RECEIPT_SCHEMA
        or receipt.get("detection_phase") != PROOT_AUTO_DETECTION_PHASE
        or receipt.get("cell_id") != cell.cell_id
        or receipt.get("task_name") != cell.task_name
        or receipt.get("official_attempt_index") != cell.official_attempt_index
        or receipt.get("execution_attempt") != execution
        or _parse_time(receipt.get("archived_at")) is None
        or not isinstance(failure, Mapping)
        or not isinstance(relative_trial, str)
    ):
        raise PoolError(f"invalid PRoot retry archive receipt: {archive}")
    relative_path = Path(relative_trial)
    if (
        relative_path.is_absolute()
        or len(relative_path.parts) != 1
        or relative_path.parts[0] in ("", ".", "..")
    ):
        raise PoolError(f"unsafe PRoot retry archive trial path: {archive}")
    trial_dir = archive / relative_path
    result_path = trial_dir / "result.json"
    if (
        trial_dir.is_symlink()
        or not trial_dir.is_dir()
        or result_path.is_symlink()
        or not result_path.is_file()
    ):
        raise PoolError(f"PRoot retry archive trial is incomplete: {archive}")
    trial = _read_json(result_path)
    try:
        trial = TrialResult.model_validate(trial).model_dump(
            mode="json", exclude_none=True
        )
    except Exception as exc:
        raise PoolError(f"invalid archived PRoot Harbor trial: {archive}") from exc
    synthetic = ClaimedCell(
        cell=cell,
        paths=paths,
        descriptor=-1,
        execution_attempt=execution,
        dispatch={},
    )
    _validate_harbor_trial(spec, synthetic, trial)
    classified = _local_proot_infrastructure_trial_failure(trial_dir, trial)
    if (
        classified is None
        or dict(failure) != classified
        or _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
        or trial.get("verifier_result") is not None
        or _result_rewards(trial)
    ):
        raise PoolError(f"archived trial is not a strict PRoot failure: {archive}")
    return execution, classified


def _validated_unjournaled_sglang_archive(
    spec: PoolSpec,
    cell: CellSpec,
    paths: CellPaths,
    archive: Path,
) -> tuple[int, dict[str, Any]]:
    """Validate a strict SGLang archive before recovering its budget journal."""

    if archive.is_symlink() or not archive.is_dir():
        raise PoolError(f"unsafe SGLang retry archive: {archive}")
    match = re.fullmatch(r"execution-([0-9]+)", archive.parent.name)
    if match is None:
        raise PoolError(f"SGLang retry archive has no execution owner: {archive}")
    execution = int(match.group(1))
    if execution < 1 or archive.parent.name != f"execution-{execution:04d}":
        raise PoolError(f"SGLang retry archive execution is not canonical: {archive}")
    receipt_path = archive / SGLANG_ARCHIVE_RECEIPT_NAME
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise PoolError(f"SGLang retry archive has no safe receipt: {archive}")
    receipt = _read_json(receipt_path, max_bytes=1024 * 1024)
    failure = receipt.get("failure")
    relative_trial = receipt.get("trial_path")
    if (
        receipt.get("schema") != SGLANG_ARCHIVE_RECEIPT_SCHEMA
        or receipt.get("cell_id") != cell.cell_id
        or receipt.get("task_name") != cell.task_name
        or receipt.get("official_attempt_index") != cell.official_attempt_index
        or receipt.get("execution_attempt") != execution
        or _parse_time(receipt.get("archived_at")) is None
        or not isinstance(failure, Mapping)
        or not isinstance(relative_trial, str)
    ):
        raise PoolError(f"invalid SGLang retry archive receipt: {archive}")
    relative_path = Path(relative_trial)
    if (
        relative_path.is_absolute()
        or len(relative_path.parts) != 1
        or relative_path.parts[0] in ("", ".", "..")
    ):
        raise PoolError(f"unsafe SGLang retry archive trial path: {archive}")
    trial_dir = archive / relative_path
    result_path = trial_dir / "result.json"
    if (
        trial_dir.is_symlink()
        or not trial_dir.is_dir()
        or result_path.is_symlink()
        or not result_path.is_file()
    ):
        raise PoolError(f"SGLang retry archive trial is incomplete: {archive}")
    trial = _read_json(result_path)
    try:
        trial = TrialResult.model_validate(trial).model_dump(
            mode="json", exclude_none=True
        )
    except Exception as exc:
        raise PoolError(f"invalid archived SGLang Harbor trial: {archive}") from exc
    synthetic = ClaimedCell(
        cell=cell,
        paths=paths,
        descriptor=-1,
        execution_attempt=execution,
        dispatch={},
    )
    _validate_harbor_trial(spec, synthetic, trial)
    classified = _retryable_sglang_trial_failure(trial)
    if (
        classified is None
        or dict(failure) != classified
        or trial.get("verifier_result") is not None
        or _result_rewards(trial)
    ):
        raise PoolError(f"archived trial is not a strict SGLang failure: {archive}")
    return execution, classified


def _reconcile_unjournaled_proot_archives(
    spec: PoolSpec,
    cell: CellSpec,
    paths: CellPaths,
    *,
    descriptor: int,
    requested_max_proot_retries: int,
) -> bool:
    """Recover strict-infra archive-to-journal crash windows under cell flock."""

    (
        _infrastructure_count,
        history,
        proot_failure_count,
        proot_requeue_count,
        recorded_max_proot_retries,
        last_proot_failure,
    ) = _read_infrastructure_retry(paths, cell)
    effective_cap = (
        recorded_max_proot_retries
        if recorded_max_proot_retries is not None
        else requested_max_proot_retries
    )
    recorded_archives = {
        str(item.get("harbor_archive_path"))
        for item in history
        if isinstance(item.get("harbor_archive_path"), str)
    }
    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    if paths.attempts_dir.is_dir() and not paths.attempts_dir.is_symlink():
        for attempt in sorted(paths.attempts_dir.iterdir(), key=lambda path: path.name):
            if attempt.is_symlink():
                if any(
                    (attempt / name).exists()
                    for name in (PROOT_AUTO_ARCHIVE_NAME, SGLANG_AUTO_ARCHIVE_NAME)
                ):
                    raise PoolError(
                        f"infrastructure retry archive has a symlinked attempt: {attempt}"
                    )
                continue
            if not attempt.is_dir():
                continue
            for archive_name, validator in (
                (PROOT_AUTO_ARCHIVE_NAME, _validated_unjournaled_proot_archive),
                (SGLANG_AUTO_ARCHIVE_NAME, _validated_unjournaled_sglang_archive),
            ):
                archive = attempt / archive_name
                if not archive.exists() and not archive.is_symlink():
                    continue
                relative_archive = str(archive.relative_to(spec.paths.root))
                if relative_archive in recorded_archives:
                    continue
                execution, failure = validator(spec, cell, paths, archive)
                candidates.append((execution, archive, failure))

    # Validate all missing archives before mutating the first journal entry.
    candidates.sort(key=lambda item: item[0])
    if paths.harbor_job.exists() and candidates:
        raise PoolError(
            f"unjournaled PRoot archive races a live Harbor job: {cell.cell_id}"
        )
    mutated = False
    for execution, archive, failure in candidates:
        dispatch = _read_json(paths.dispatch) if paths.dispatch.is_file() else {}
        claim = ClaimedCell(
            cell=cell,
            paths=paths,
            descriptor=descriptor,
            execution_attempt=execution,
            dispatch=dispatch,
        )
        exc: RetryableSGLangTrialError | RetryablePRootTrialError
        if failure.get("infrastructure_kind") == "sglang":
            exc = RetryableSGLangTrialError(failure, archive)
        else:
            exc = RetryablePRootTrialError(failure, archive)
        (
            infrastructure_count,
            proot_failure_count,
            proot_requeue_count,
            effective_cap,
            exhausted,
        ) = _record_bounded_proot_infrastructure_retry(
            spec,
            claim,
            exc,
            retry_backoff_seconds=0,
            max_proot_infrastructure_retries=effective_cap,
        )
        _best_effort_event(
            spec.paths.root,
            {
                "event": (
                    "cell_infrastructure_archive_reconciled_exhausted"
                    if exhausted
                    else "cell_infrastructure_archive_reconciled_requeued"
                ),
                "cell_id": cell.cell_id,
                "official_attempt_index": cell.official_attempt_index,
                "execution_attempt": execution,
                "infrastructure_retry_count": infrastructure_count,
                "proot_infrastructure_failure_count": proot_failure_count,
                "proot_infrastructure_requeue_count": proot_requeue_count,
                "max_proot_infrastructure_retries": effective_cap,
                "harbor_archive_path": str(archive.relative_to(spec.paths.root)),
                "recorded_at": _utc_now(),
            },
        )
        mutated = True

    if mutated:
        return True

    # A crash can also occur after the journal write but before its queued or
    # exhausted projection. Repair that projection without charging again.
    if proot_failure_count < 1:
        return False
    dispatch = _read_json(paths.dispatch) if paths.dispatch.is_file() else {}
    state = _read_json(paths.state) if paths.state.is_file() else {}
    if proot_failure_count > effective_cap:
        if last_proot_failure is None:
            raise PoolError(
                f"exhausted PRoot retry journal has no failure: {cell.cell_id}"
            )
        _proot_retry_exhausted_result(
            spec,
            cell,
            paths,
            execution_attempt=_read_execution(paths, dispatch),
            proot_failure_count=proot_failure_count,
            proot_requeue_count=proot_requeue_count,
            max_proot_retries=effective_cap,
            failure=last_proot_failure,
        )
        return True
    if dispatch.get("status") == "queued" and state.get("status") == "queued":
        return False
    if dispatch.get("status") not in (None, "running", "queued") or state.get(
        "status"
    ) not in (None, "running", "queued"):
        raise PoolError(
            f"PRoot retry journal projection is inconsistent: {cell.cell_id}"
        )
    latest = next(
        (
            item
            for item in reversed(history)
            if isinstance(item.get("retry_budget"), Mapping)
            and item["retry_budget"].get("policy")
            in {
                "independent_proot_infrastructure_requeues_v1",
                "shared_strict_infrastructure_requeues_v1",
            }
        ),
        None,
    )
    if latest is None or last_proot_failure is None:
        raise PoolError(
            f"PRoot retry journal lacks projection evidence: {cell.cell_id}"
        )
    execution = _valid_execution(latest.get("execution_attempt"))
    archive_path = latest.get("harbor_archive_path")
    if execution < 1 or not isinstance(archive_path, str):
        raise PoolError(f"PRoot retry journal has invalid projection: {cell.cell_id}")
    now = _utc_now()
    dispatch.update(
        {
            "status": "queued",
            "execution_attempt": execution,
            "worker_exception_count": int(dispatch.get("worker_exception_count") or 0),
            "infrastructure_retry_count": _infrastructure_count,
            "proot_infrastructure_failure_count": proot_failure_count,
            "proot_infrastructure_requeue_count": proot_requeue_count,
            "max_proot_infrastructure_retries": effective_cap,
            "last_infrastructure_failure": last_proot_failure,
            "last_infrastructure_archive": archive_path,
            "retry_not_before": now,
            "updated_at": now,
        }
    )
    for key in ("claim", "outcome", "harbor_job_path"):
        dispatch.pop(key, None)
    _atomic_write_json(paths.dispatch, dispatch)
    _atomic_write_json(
        paths.state,
        {
            "schema": CELL_STATE_SCHEMA,
            "cell_id": cell.cell_id,
            "task_name": cell.task_name,
            "task_ref": cell.task_ref,
            "official_attempt_index": cell.official_attempt_index,
            "config_contract_hash": spec.contract_hash,
            "status": "queued",
            "execution_attempt": execution,
            "worker_exception_count": int(dispatch.get("worker_exception_count") or 0),
            "infrastructure_retry_count": _infrastructure_count,
            "proot_infrastructure_failure_count": proot_failure_count,
            "proot_infrastructure_requeue_count": proot_requeue_count,
            "max_proot_infrastructure_retries": effective_cap,
            "last_infrastructure_failure": last_proot_failure,
            "last_infrastructure_archive": archive_path,
            "updated_at": now,
        },
    )
    _best_effort_event(
        spec.paths.root,
        {
            "event": "cell_proot_journal_projection_reconciled",
            "cell_id": cell.cell_id,
            "execution_attempt": execution,
            "recorded_at": now,
        },
    )
    return True


def _sglang_is_healthy(sglang_url: str, *, timeout_seconds: float = 5.0) -> bool:
    """Bounded readiness admission; a transient busy probe is not a dead server."""
    from shared_runtime.residual_preflight import wait_for_health
    try:
        wait_for_health(sglang_url, total_timeout=180,
                        request_timeout=max(0.1, timeout_seconds))
        return True
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _publish_verified_agent_sigterm_reconciliation(
    spec: PoolSpec,
    claim: ClaimedCell,
    *,
    evidence: Mapping[str, Any],
    raw: Mapping[str, bytes],
) -> tuple[Path, dict[str, Any], str]:
    """Durably snapshot the old attribution before replacing its commit marker."""

    attempt_dir = _ensure_dir(
        claim.paths.attempts_dir / f"execution-{claim.execution_attempt:04d}"
    )
    destination = attempt_dir / VERIFIED_AGENT_SIGTERM_RECONCILIATION_DIR_NAME
    expected_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in raw.items()
    }
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise PoolError("unsafe verified-agent-exit reconciliation directory")
        receipt_path = destination / "receipt.json"
        receipt = _read_json(receipt_path, max_bytes=MAX_ARTIFACT_BYTES)
        original = receipt.get("original")
        if (
            receipt.get("schema") != VERIFIED_AGENT_SIGTERM_RECONCILIATION_SCHEMA
            or receipt.get("cell_id") != claim.cell.cell_id
            or receipt.get("execution_attempt") != claim.execution_attempt
            or _digest(receipt.get("verified_agent_exit_evidence")) != _digest(evidence)
            or not isinstance(original, Mapping)
            or any(
                not isinstance(original.get(name), Mapping)
                or original[name].get("sha256") != digest
                for name, digest in expected_hashes.items()
            )
        ):
            raise PoolError("existing verified-agent-exit receipt does not match")
        receipt_bytes = _read_regular_bytes(receipt_path, max_bytes=MAX_ARTIFACT_BYTES)
        return receipt_path, receipt, hashlib.sha256(receipt_bytes).hexdigest()

    staging = attempt_dir / (
        f".{VERIFIED_AGENT_SIGTERM_RECONCILIATION_DIR_NAME}-{uuid.uuid4().hex}"
    )
    try:
        _ensure_dir(staging)
        original_dir = _ensure_dir(staging / "original")
        original_records: dict[str, Any] = {}
        for name, payload in raw.items():
            _atomic_write_bytes(original_dir / name, payload)
            final_path = destination / "original" / name
            original_records[name] = {
                "path": str(final_path.relative_to(spec.paths.root)),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        receipt = {
            "schema": VERIFIED_AGENT_SIGTERM_RECONCILIATION_SCHEMA,
            "cell_id": claim.cell.cell_id,
            "task_name": claim.cell.task_name,
            "task_ref": claim.cell.task_ref,
            "official_attempt_index": claim.cell.official_attempt_index,
            "execution_attempt": claim.execution_attempt,
            "config_contract_hash": spec.contract_hash,
            "policy": "retroactive_attribution_only_no_rerun_v1",
            "model_called": False,
            "verifier_rerun": False,
            "official_attempt_consumed_by_reconciliation": False,
            "verified_agent_exit_evidence": dict(evidence),
            "original": original_records,
            "prepared_at": _utc_now(),
        }
        _atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
        _fsync_directory(attempt_dir)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt_path = destination / "receipt.json"
    receipt_bytes = _read_regular_bytes(receipt_path, max_bytes=MAX_ARTIFACT_BYTES)
    return receipt_path, receipt, hashlib.sha256(receipt_bytes).hexdigest()


def _validate_verified_agent_sigterm_reconciliation(
    spec: PoolSpec,
    cell: CellSpec,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    projection = result.get("attribution_reconciliation")
    if not isinstance(projection, Mapping):
        raise PoolError("cell result has no agent-SIGTERM reconciliation projection")
    execution = _valid_execution(result.get("execution_attempt"))
    commit_dir = (
        spec.cell_paths(cell).attempts_dir
        / f"execution-{execution:04d}"
        / VERIFIED_AGENT_SIGTERM_RECONCILIATION_DIR_NAME
    )
    receipt_path = commit_dir / "receipt.json"
    expected_relative = str(receipt_path.relative_to(spec.paths.root))
    if (
        execution < 1
        or projection.get("schema") != VERIFIED_AGENT_SIGTERM_RECONCILIATION_SCHEMA
        or projection.get("receipt_path") != expected_relative
        or projection.get("policy") != "retroactive_attribution_only_no_rerun_v1"
    ):
        raise PoolError("invalid agent-SIGTERM reconciliation projection")
    receipt_bytes = _read_regular_bytes(receipt_path, max_bytes=MAX_ARTIFACT_BYTES)
    if hashlib.sha256(receipt_bytes).hexdigest() != projection.get("receipt_sha256"):
        raise PoolError("agent-SIGTERM reconciliation receipt digest mismatch")
    receipt = _read_json(receipt_path, max_bytes=MAX_ARTIFACT_BYTES)
    identity = {
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "execution_attempt": execution,
        "config_contract_hash": spec.contract_hash,
    }
    if (
        receipt.get("schema") != VERIFIED_AGENT_SIGTERM_RECONCILIATION_SCHEMA
        or any(receipt.get(key) != value for key, value in identity.items())
        or receipt.get("policy") != "retroactive_attribution_only_no_rerun_v1"
        or receipt.get("model_called") is not False
        or receipt.get("verifier_rerun") is not False
        or receipt.get("official_attempt_consumed_by_reconciliation") is not False
        or _digest(receipt.get("verified_agent_exit_evidence"))
        != _digest(result.get("verified_agent_exit_warning"))
    ):
        raise PoolError("agent-SIGTERM reconciliation receipt identity mismatch")
    original = receipt.get("original")
    required = {
        "result.json",
        "state.json",
        "dispatch.json",
        "trial-result.json",
        "reward.txt",
        "ctrf.json",
    }
    if not isinstance(original, Mapping) or set(original) != required:
        raise PoolError("agent-SIGTERM reconciliation snapshot set is invalid")
    snapshots = {
        name: _contained_replay_snapshot(spec, commit_dir, original[name])
        for name in required
    }
    prior_result = _json_mapping_from_bytes(
        snapshots["result.json"], label="original cell result"
    )
    if (
        prior_result.get("outcome") != "errored"
        or prior_result.get("rewards") != {}
        or prior_result.get("rejected_verifier_rewards") != {"reward": 1.0}
        or hashlib.sha256(snapshots["result.json"]).hexdigest()
        != projection.get("original_result_sha256")
    ):
        raise PoolError("agent-SIGTERM reconciliation original result is invalid")
    return receipt


def reconcile_committed_verified_agent_sigterm(
    *,
    job_name: str,
    cell_id: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
) -> dict[str, Any]:
    """Atomically reattribute one strictly proven, already verified SIGTERM pass.

    This command never reruns the model or verifier and never scans the pool.
    The caller must name one cell.  Its old result, projections, Harbor result,
    and verifier artifacts are snapshotted under the execution attempt before
    ``result.json`` is atomically replaced as the sole score authority.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    matches = [cell for cell in spec.cells if cell.cell_id == cell_id]
    if len(matches) != 1:
        raise PoolError(f"unknown cell id: {cell_id}")
    cell = matches[0]
    paths = spec.cell_paths(cell).ensure()
    descriptor = _open_lock(paths.claim_lock, blocking=False)
    if descriptor is None:
        raise PoolError(f"cell is currently owned by a worker: {cell_id}")
    status = "reconciled"
    try:
        result = _cell_result(spec, cell)
        if result is None:
            raise PoolError(f"cell is not terminal: {cell_id}")
        execution = _valid_execution(result.get("execution_attempt"))
        if execution < 1:
            raise PoolError(f"cell result has no execution attempt: {cell_id}")
        state = _read_json(paths.state)
        dispatch = _read_json(paths.dispatch)
        claim = ClaimedCell(
            cell=cell,
            paths=paths,
            descriptor=descriptor,
            execution_attempt=execution,
            dispatch=dispatch,
        )
        terminal = _terminal_harbor_trial(spec, claim)
        if terminal is None:
            raise PoolError(f"cell has no unique terminal Harbor trial: {cell_id}")
        trial_path, trial = terminal
        evidence = _verified_success_after_agent_sigterm(trial_path, trial)
        if evidence is None:
            raise PoolError(
                f"cell lacks complete verified agent-SIGTERM evidence: {cell_id}"
            )

        if result.get("outcome") == "passed":
            if (
                result.get("rewards") != {"reward": 1.0}
                or result.get("rejected_verifier_rewards") is not None
                or _digest(result.get("verified_agent_exit_warning"))
                != _digest(evidence)
                or _scoreable_cell_reward(spec, cell, result) != 1.0
            ):
                raise PoolError(
                    f"existing reconciled agent-SIGTERM result is invalid: {cell_id}"
                )
            _validate_verified_agent_sigterm_reconciliation(spec, cell, result)
            status = "already_reconciled"
            if (
                state.get("status") != "completed"
                or state.get("outcome") != "passed"
                or dispatch.get("status") != "terminal"
                or dispatch.get("outcome") != "passed"
            ):
                _commit_cell_result(spec, claim, result)
        else:
            identity = {
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
            }
            if (
                result.get("status") != "completed"
                or result.get("outcome") != "errored"
                or result.get("rewards") != {}
                or result.get("rejected_verifier_rewards") != {"reward": 1.0}
                or result.get("verified_agent_exit_warning") is not None
                or any(state.get(key) != value for key, value in identity.items())
                or any(dispatch.get(key) != value for key, value in identity.items())
                or state.get("status") != "completed"
                or state.get("outcome") != "errored"
                or dispatch.get("status") != "terminal"
                or dispatch.get("outcome") != "errored"
                or state.get("execution_attempt") != execution
                or dispatch.get("execution_attempt") != execution
                or result.get("trial_path")
                != str(trial_path.relative_to(spec.paths.root))
                or result.get("trial_name") != trial.get("trial_name")
                or _digest(result.get("exception_info"))
                != _digest(trial.get("exception_info"))
                or _result_rewards(trial) != {"reward": 1.0}
                or paths.exception.exists()
                or paths.external_reservation.exists()
            ):
                raise PoolError(
                    f"committed agent-SIGTERM attribution is inconsistent: {cell_id}"
                )
            raw = {
                "result.json": _read_regular_bytes(
                    paths.result, max_bytes=MAX_ARTIFACT_BYTES
                ),
                "state.json": _read_regular_bytes(
                    paths.state, max_bytes=MAX_ARTIFACT_BYTES
                ),
                "dispatch.json": _read_regular_bytes(
                    paths.dispatch, max_bytes=MAX_ARTIFACT_BYTES
                ),
                "trial-result.json": _read_regular_bytes(
                    trial_path / "result.json", max_bytes=MAX_ARTIFACT_BYTES
                ),
                "reward.txt": _read_regular_bytes(
                    trial_path / "verifier" / "reward.txt", max_bytes=128
                ),
                "ctrf.json": _read_regular_bytes(
                    trial_path / "verifier" / "ctrf.json",
                    max_bytes=8 * 1024 * 1024,
                ),
            }
            receipt_path, _receipt, receipt_sha256 = (
                _publish_verified_agent_sigterm_reconciliation(
                    spec, claim, evidence=evidence, raw=raw
                )
            )
            now = _utc_now()
            reconciled = dict(result)
            reconciled.update(
                {
                    "status": "completed",
                    "outcome": "passed",
                    "rewards": {"reward": 1.0},
                    "rejected_verifier_rewards": None,
                    "infrastructure_cleanup_warning": None,
                    "verified_agent_exit_warning": evidence,
                    "committed_at": now,
                    "attribution_reconciliation": {
                        "schema": VERIFIED_AGENT_SIGTERM_RECONCILIATION_SCHEMA,
                        "policy": "retroactive_attribution_only_no_rerun_v1",
                        "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                        "receipt_sha256": receipt_sha256,
                        "original_result_sha256": hashlib.sha256(
                            raw["result.json"]
                        ).hexdigest(),
                        "reconciled_at": now,
                    },
                }
            )
            if _scoreable_cell_reward(spec, cell, reconciled) != 1.0:
                raise PoolError("prepared agent-SIGTERM reconciliation is unscoreable")
            # The cell is already terminal, so atomically replace its sole
            # score authority first.  State/dispatch are recoverable projections.
            _atomic_write_json(paths.result, reconciled)
            _commit_cell_result(spec, claim, reconciled)
            _validate_verified_agent_sigterm_reconciliation(spec, cell, reconciled)
            if _scoreable_cell_reward(spec, cell, reconciled) != 1.0:
                raise PoolError("committed agent-SIGTERM reconciliation is unscoreable")
            result = reconciled
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "verified_agent_sigterm_reconciled",
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "reward": 1.0,
                    "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                    "receipt_sha256": receipt_sha256,
                    "model_called": False,
                    "verifier_rerun": False,
                    "recorded_at": now,
                },
            )
    finally:
        _close_lock(descriptor)
    pool_state = aggregate_pool(spec)
    return {
        "status": status,
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "cell_result": result,
        "pool_state": pool_state,
    }


def requeue_committed_sglang_infrastructure(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Archive and requeue committed cells proven to be SGLang failures."""

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested = set(cell_ids)
    known = {cell.cell_id for cell in spec.cells}
    unknown = requested - known
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    repaired: list[str] = []
    skipped: list[str] = []
    for cell in spec.cells:
        if requested and cell.cell_id not in requested:
            continue
        paths = spec.cell_paths(cell).ensure()
        descriptor = _open_lock(paths.claim_lock, blocking=False)
        if descriptor is None:
            skipped.append(cell.cell_id)
            continue
        try:
            result = _cell_result(spec, cell)
            if result is None:
                skipped.append(cell.cell_id)
                continue
            if result.get("outcome") != "errored" or result.get("rewards") != {}:
                skipped.append(cell.cell_id)
                continue
            execution = _valid_execution(result.get("execution_attempt"))
            if execution < 1:
                raise PoolError(
                    f"committed infrastructure result has no execution: {cell.cell_id}"
                )
            dispatch = _read_json(paths.dispatch) if paths.dispatch.is_file() else {}
            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    f"committed infrastructure cell has no Harbor trial: {cell.cell_id}"
                )
            trial_path, trial = terminal
            harbor_failure = _retryable_sglang_trial_failure(trial)
            if (
                harbor_failure is None
                or trial.get("verifier_result") is not None
                or _result_rewards(trial)
                or _digest(trial.get("exception_info"))
                != _digest(result.get("exception_info"))
                or result.get("trial_path")
                != str(trial_path.relative_to(spec.paths.root))
            ):
                raise PoolError(
                    f"committed SGLang failure lacks strict Harbor evidence: {cell.cell_id}"
                )
            failure = harbor_failure
            attempt_dir = _ensure_dir(paths.attempts_dir / f"execution-{execution:04d}")
            terminal_archive = attempt_dir / "committed-terminal-retryable-sglang"
            if terminal_archive.exists() or terminal_archive.is_symlink():
                raise PoolError(
                    f"committed infrastructure archive already exists: {terminal_archive}"
                )
            harbor_archive = _archive_retryable_sglang_harbor_job(
                claim,
                trial_path=trial_path,
                failure=failure,
            )
            _ensure_dir(terminal_archive)
            for source in (paths.result, paths.state, paths.dispatch):
                if source.is_file() and not source.is_symlink():
                    os.replace(source, terminal_archive / source.name)
            _fsync_directory(paths.root)
            retry = RetryableSGLangTrialError(failure, harbor_archive)
            count, failures, requeues, cap, exhausted = (
                _record_bounded_proot_infrastructure_retry(
                    spec,
                    claim,
                    retry,
                    retry_backoff_seconds=0,
                    max_proot_infrastructure_retries=2,
                )
            )
            _atomic_write_json(
                terminal_archive / "requeue.json",
                {
                    "schema": "terminal_bench_committed_sglang_requeue_v1",
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "infrastructure_retry_count": count,
                    "strict_infrastructure_failure_count": failures,
                    "strict_infrastructure_requeue_count": requeues,
                    "max_strict_infrastructure_retries": cap,
                    "retry_exhausted": exhausted,
                    "failure": failure,
                    "harbor_archive_path": str(
                        harbor_archive.relative_to(spec.paths.root)
                    ),
                    "requeued_at": _utc_now(),
                },
            )
            repaired.append(cell.cell_id)
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_sglang_infrastructure_requeued",
                    "cell_id": cell.cell_id,
                    "execution_attempt": execution,
                    "official_attempt_index": cell.official_attempt_index,
                    "recorded_at": _utc_now(),
                },
            )
        finally:
            _close_lock(descriptor)
    state = aggregate_pool(spec)
    return {"repaired_cell_ids": repaired, "skipped_cell_ids": skipped, **state}


def requeue_committed_proot_infrastructure(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Explicitly repair committed local PRoot/runtime infrastructure failures.

    This operation is intentionally cell-scoped and fail-closed.  It never
    scans and mutates an entire pool: callers must name every candidate cell,
    and both the pool commit and its pinned Harbor trial must independently
    satisfy the strict traceback or host-receipt signature before any marker is
    moved.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested = set(cell_ids)
    if not requested:
        raise PoolError("repair-proot-infrastructure requires at least one cell id")
    known = {cell.cell_id for cell in spec.cells}
    unknown = requested - known
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    spec_environment = spec.config.get("environment")
    if (
        not isinstance(spec_environment, Mapping)
        or spec_environment.get("import_path") != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
    ):
        raise PoolError("pool does not use the local PRoot environment contract")

    repaired: list[str] = []
    skipped: list[str] = []
    locked_descriptors: list[int] = []
    validated: list[dict[str, Any]] = []
    try:
        # Validate every unlocked target before the first archive is created.
        # Keeping each flock through the mutation phase closes the validation-
        # to-commit race without treating heartbeat leases as ownership.
        for cell in spec.cells:
            if cell.cell_id not in requested:
                continue
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                skipped.append(cell.cell_id)
                continue
            locked_descriptors.append(descriptor)
            result = _cell_result(spec, cell)
            if result is None:
                skipped.append(cell.cell_id)
                continue
            projected_failure = _proot_rpc_trial_failure(result)
            if result.get("outcome") != "errored" or result.get("rewards") != {}:
                if projected_failure is None:
                    skipped.append(cell.cell_id)
                    continue
                raise PoolError(
                    f"committed PRoot failure has an outcome or reward: {cell.cell_id}"
                )
            execution = _valid_execution(result.get("execution_attempt"))
            if execution < 1:
                raise PoolError(
                    f"committed PRoot infrastructure result has no execution: "
                    f"{cell.cell_id}"
                )
            state = _read_json(paths.state)
            dispatch = _read_json(paths.dispatch)
            if (
                state.get("status") != "completed"
                or state.get("outcome") != "errored"
                or state.get("execution_attempt") != execution
                or dispatch.get("status") != "terminal"
                or dispatch.get("outcome") != "errored"
                or dispatch.get("execution_attempt") != execution
            ):
                raise PoolError(
                    f"committed PRoot infrastructure metadata is inconsistent: "
                    f"{cell.cell_id}"
                )
            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    f"committed PRoot infrastructure cell has no Harbor trial: "
                    f"{cell.cell_id}"
                )
            trial_path, trial = terminal
            harbor_failure = _local_proot_infrastructure_trial_failure(
                trial_path, trial
            )
            if harbor_failure is None:
                if projected_failure is None:
                    skipped.append(cell.cell_id)
                    continue
                raise PoolError(
                    f"Harbor PRoot failure lacks the required environment/reward "
                    f"evidence: {cell.cell_id}"
                )
            if projected_failure is not None and harbor_failure != projected_failure:
                raise PoolError(
                    f"pool and Harbor PRoot infrastructure failures disagree: "
                    f"{cell.cell_id}"
                )
            projected_exception = result.get("exception_info")
            harbor_exception = trial.get("exception_info")
            if (
                not isinstance(projected_exception, Mapping)
                or not isinstance(harbor_exception, Mapping)
                or _digest(dict(projected_exception)) != _digest(dict(harbor_exception))
                or result.get("trial_path")
                != str(trial_path.relative_to(spec.paths.root))
            ):
                raise PoolError(
                    f"pool and Harbor PRoot exception provenance disagree: "
                    f"{cell.cell_id}"
                )
            if (
                _environment_import_path(trial) != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
                or trial.get("verifier_result") is not None
                or _result_rewards(trial)
            ):
                raise PoolError(
                    f"Harbor PRoot failure lacks the required environment/reward "
                    f"evidence: {cell.cell_id}"
                )
            attempt_dir = paths.attempts_dir / f"execution-{execution:04d}"
            terminal_archive = attempt_dir / "committed-terminal-proot-infrastructure"
            harbor_archive = attempt_dir / "harbor-job-committed-proot-infrastructure"
            for archive in (terminal_archive, harbor_archive):
                if archive.exists() or archive.is_symlink():
                    raise PoolError(
                        f"committed PRoot infrastructure archive already exists: "
                        f"{archive}"
                    )
            (
                prior_infrastructure_count,
                _history,
                prior_strict_failures,
                _prior_strict_requeues,
                recorded_strict_cap,
                _last_strict_failure,
            ) = _read_infrastructure_retry(paths, cell)
            legacy_non_proot = max(
                0, prior_infrastructure_count - prior_strict_failures
            )
            next_strict_failure = prior_strict_failures + legacy_non_proot + 1
            effective_strict_cap = (
                recorded_strict_cap if recorded_strict_cap is not None else 2
            )
            if next_strict_failure > effective_strict_cap:
                raise PoolError(
                    "committed PRoot repair would exceed the immutable strict "
                    f"infrastructure retry cap for {cell.cell_id}"
                )
            validated.append(
                {
                    "claim": claim,
                    "state": state,
                    "dispatch": dispatch,
                    "failure": harbor_failure,
                    "trial_path": trial_path,
                    "attempt_dir": attempt_dir,
                    "terminal_archive": terminal_archive,
                }
            )

        for candidate in validated:
            claim = candidate["claim"]
            cell = claim.cell
            paths = claim.paths
            execution = claim.execution_attempt
            state = candidate["state"]
            dispatch = candidate["dispatch"]
            failure = candidate["failure"]
            trial_path = candidate["trial_path"]
            _ensure_dir(candidate["attempt_dir"])
            terminal_archive = candidate["terminal_archive"]
            _ensure_dir(terminal_archive)
            _atomic_write_json(terminal_archive / "state.json", state)
            _atomic_write_json(terminal_archive / "dispatch.json", dispatch)
            harbor_archive = _archive_proot_infrastructure_harbor_job(
                claim,
                trial_path=trial_path,
                failure=failure,
            )
            retry = RetryablePRootTrialError(failure, harbor_archive)
            count, failures, requeues, cap, exhausted = (
                _record_bounded_proot_infrastructure_retry(
                    spec,
                    claim,
                    retry,
                    retry_backoff_seconds=0,
                    max_proot_infrastructure_retries=2,
                )
            )
            if exhausted:
                raise PoolError(
                    "validated committed PRoot repair unexpectedly exhausted its cap"
                )
            # result.json is the sole terminal commit marker. Retire it only
            # after the complete Harbor archive and queued metadata are durable.
            os.replace(paths.result, terminal_archive / "result.json")
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)
            _atomic_write_json(
                terminal_archive / "requeue.json",
                {
                    "schema": (
                        "terminal_bench_committed_proot_infrastructure_requeue_v1"
                    ),
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "infrastructure_retry_count": count,
                    "strict_infrastructure_failure_count": failures,
                    "strict_infrastructure_requeue_count": requeues,
                    "max_strict_infrastructure_retries": cap,
                    "retry_exhausted": False,
                    "failure": dict(failure),
                    "harbor_archive_path": str(
                        harbor_archive.relative_to(spec.paths.root)
                    ),
                    "requeue_policy": "proot_rpc_transport_failure_v1",
                    "requeued_at": _utc_now(),
                },
            )
            repaired.append(cell.cell_id)
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_proot_infrastructure_requeued",
                    "cell_id": cell.cell_id,
                    "execution_attempt": execution,
                    "official_attempt_index": cell.official_attempt_index,
                    "recorded_at": _utc_now(),
                },
            )
    finally:
        for descriptor in reversed(locked_descriptors):
            _close_lock(descriptor)
    state = aggregate_pool(spec)
    return {"repaired_cell_ids": repaired, "skipped_cell_ids": skipped, **state}


def requeue_committed_proot_localhost_incident(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Archive and requeue only the three diagnosed legacy PRoot false failures.

    This is not a generic reward-zero retry.  Every caller-named cell is locked
    and fully validated before the first mutation.  Authorization is bounded by
    the exact pinned task refs and by independent pool, Harbor, PRoot-receipt,
    verifier-report, and (for QEMU) live agent-command evidence.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested_order = list(cell_ids)
    if not requested_order:
        raise PoolError("repair-proot-localhost requires at least one cell id")
    if len(set(requested_order)) != len(requested_order):
        raise PoolError("repair-proot-localhost received a duplicate cell id")
    cells_by_id = {cell.cell_id: cell for cell in spec.cells}
    unknown = set(requested_order) - set(cells_by_id)
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    agents = spec.config.get("agents")
    environment = spec.config.get("environment")
    if (
        not isinstance(agents, list)
        or len(agents) != 1
        or not isinstance(agents[0], Mapping)
        or agents[0].get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
        or not isinstance(environment, Mapping)
        or environment.get("import_path") != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
    ):
        raise PoolError("pool does not use the diagnosed Codex/PRoot contract")

    locked_descriptors: list[int] = []
    candidates: list[dict[str, Any]] = []
    try:
        for cell_id in requested_order:
            cell = cells_by_id[cell_id]
            task_contract = _PROOT_LOCALHOST_INCIDENT_TASKS.get(cell.task_name)
            if (
                task_contract is None
                or cell.task_ref != task_contract.get("task_ref")
                or cell.source != "terminal-bench/terminal-bench-2-1"
                or cell.official_attempt_index != 1
            ):
                raise PoolError(
                    f"cell is outside the exact PRoot localhost incident: {cell_id}"
                )
            if not _cell_is_score_eligible(
                cell, _attempt_selection(spec), _task_selection(spec)
            ):
                raise PoolError(f"incident cell is excluded from scoring: {cell_id}")
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                raise PoolError(f"requested cell is currently owned: {cell_id}")
            locked_descriptors.append(descriptor)
            if any(
                path.exists() or path.is_symlink()
                for path in (
                    paths.external_reservation,
                    paths.exception,
                    paths.infrastructure,
                    paths.task_cache_transport,
                )
            ):
                raise PoolError(
                    f"incident cell has unrelated retry/reservation provenance: "
                    f"{cell_id}"
                )
            result = _cell_result(spec, cell)
            if result is None:
                raise PoolError(f"requested cell has no terminal result: {cell_id}")
            execution = _valid_execution(result.get("execution_attempt"))
            if execution != 1:
                raise PoolError(
                    f"PRoot localhost incident is not original execution 1: {cell_id}"
                )
            if (
                result.get("status") != "completed"
                or result.get("outcome") != "failed"
                or result.get("rewards") != {"reward": 0.0}
                or result.get("exception_info") is not None
                or result.get("rejected_verifier_rewards") is not None
                or result.get("infrastructure_cleanup_warning") is not None
                or result.get("verified_agent_exit_warning") is not None
                or result.get("official_attempt_index") != 1
                or result.get("source") != cell.source
                or result.get("external_import") is not None
                or result.get("verifier_replay") is not None
            ):
                raise PoolError(
                    f"committed PRoot localhost result is inconsistent: {cell_id}"
                )
            state = _read_json(paths.state)
            dispatch = _read_json(paths.dispatch)
            raw_repair_count = dispatch.get("proot_localhost_repair_count", 0)
            raw_worker_exception_count = dispatch.get("worker_exception_count", 0)
            if (
                state.get("status") != "completed"
                or state.get("outcome") != "failed"
                or state.get("execution_attempt") != 1
                or state.get("official_attempt_index") != 1
                or dispatch.get("status") != "terminal"
                or dispatch.get("outcome") != "failed"
                or dispatch.get("execution_attempt") != 1
                or dispatch.get("official_attempt_index") != 1
                or raw_repair_count != 0
                or raw_worker_exception_count != 0
            ):
                raise PoolError(
                    f"committed PRoot localhost metadata is inconsistent: {cell_id}"
                )
            run_path = paths.attempts_dir / "execution-0001" / "run.json"
            run = _read_json(run_path, max_bytes=2 * 1024 * 1024)
            if (
                run.get("schema") != RUN_RECORD_SCHEMA
                or run.get("cell_id") != cell.cell_id
                or run.get("execution_attempt") != 1
                or run.get("action") != "run"
                or run.get("status") != "exited"
                or run.get("return_code") != 1
                or run.get("environment_backend") not in {None, "frozen"}
            ):
                raise PoolError(
                    f"PRoot localhost execution receipt is inconsistent: {cell_id}"
                )
            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    f"committed PRoot localhost cell has no Harbor trial: {cell_id}"
                )
            trial_path, trial = terminal
            agent_result = trial.get("agent_result")
            expected_trial_path = str(trial_path.relative_to(spec.paths.root))
            expected_job_path = str(paths.harbor_job.relative_to(spec.paths.root))
            if (
                trial.get("trial_name") != trial_path.name
                or trial.get("exception_info") is not None
                or _result_rewards(trial) != {"reward": 0.0}
                or result.get("trial_name") != trial_path.name
                or result.get("trial_path") != expected_trial_path
                or result.get("harbor_job_path") != expected_job_path
                or result.get("started_at") != trial.get("started_at")
                or result.get("finished_at") != trial.get("finished_at")
                or not isinstance(agent_result, Mapping)
                or result.get("n_input_tokens")
                != int(agent_result.get("n_input_tokens") or 0)
                or result.get("n_output_tokens")
                != int(agent_result.get("n_output_tokens") or 0)
            ):
                raise PoolError(
                    f"pool and Harbor PRoot localhost provenance disagree: {cell_id}"
                )
            evidence = _proot_localhost_verifier_evidence(
                trial_path, trial, task_contract=task_contract
            )
            if evidence is None:
                raise PoolError(
                    f"cell does not match the exact PRoot localhost incident: {cell_id}"
                )
            attempt_dir = paths.attempts_dir / "execution-0001"
            terminal_archive = (
                attempt_dir / "committed-terminal-proot-localhost-incident"
            )
            harbor_archive = (
                attempt_dir / "harbor-job-committed-proot-localhost-incident"
            )
            for archive in (terminal_archive, harbor_archive):
                if archive.exists() or archive.is_symlink():
                    raise PoolError(
                        f"PRoot localhost incident archive already exists: {archive}"
                    )
            result_bytes = _read_regular_bytes(
                paths.result, max_bytes=MAX_ARTIFACT_BYTES
            )
            state_bytes = _read_regular_bytes(paths.state, max_bytes=MAX_ARTIFACT_BYTES)
            dispatch_bytes = _read_regular_bytes(
                paths.dispatch, max_bytes=MAX_ARTIFACT_BYTES
            )
            candidates.append(
                {
                    "claim": claim,
                    "state": state,
                    "dispatch": dispatch,
                    "trial_path": trial_path,
                    "evidence": evidence,
                    "attempt_dir": attempt_dir,
                    "terminal_archive": terminal_archive,
                    "worker_exception_count": raw_worker_exception_count,
                    "artifact_payloads": {
                        "result.json": result_bytes,
                        "state.json": state_bytes,
                        "dispatch.json": dispatch_bytes,
                    },
                    "artifact_sha256": {
                        "result.json": hashlib.sha256(result_bytes).hexdigest(),
                        "state.json": hashlib.sha256(state_bytes).hexdigest(),
                        "dispatch.json": hashlib.sha256(dispatch_bytes).hexdigest(),
                    },
                }
            )

        repaired: list[str] = []
        classifications: dict[str, str] = {}
        for candidate in candidates:
            claim = candidate["claim"]
            cell = claim.cell
            paths = claim.paths
            execution = claim.execution_attempt
            dispatch = candidate["dispatch"]
            evidence = candidate["evidence"]
            _ensure_dir(candidate["attempt_dir"])
            terminal_archive = _ensure_dir(candidate["terminal_archive"])
            payloads = candidate["artifact_payloads"]
            _atomic_write_bytes(terminal_archive / "state.json", payloads["state.json"])
            _atomic_write_bytes(
                terminal_archive / "dispatch.json", payloads["dispatch.json"]
            )
            harbor_archive = _archive_proot_localhost_harbor_job(
                claim,
                trial_path=candidate["trial_path"],
                evidence=evidence,
            )
            now = _utc_now()
            harbor_relative = str(harbor_archive.relative_to(spec.paths.root))
            queued_dispatch = dict(dispatch)
            queued_dispatch.update(
                {
                    "status": "queued",
                    "proot_localhost_repair_count": 1,
                    "last_proot_localhost_incident": _PROOT_LOCALHOST_INCIDENT,
                    "last_proot_localhost_classifier": evidence["classifier"],
                    "last_proot_localhost_execution_attempt": execution,
                    "last_proot_localhost_archive": harbor_relative,
                    "updated_at": now,
                }
            )
            for key in ("claim", "outcome", "harbor_job_path", "retry_not_before"):
                queued_dispatch.pop(key, None)
            queued_state = {
                "schema": CELL_STATE_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "execution_attempt": execution,
                "worker_exception_count": candidate["worker_exception_count"],
                "proot_localhost_repair_count": 1,
                "last_proot_localhost_incident": _PROOT_LOCALHOST_INCIDENT,
                "last_proot_localhost_classifier": evidence["classifier"],
                "last_proot_localhost_execution_attempt": execution,
                "last_proot_localhost_archive": harbor_relative,
                "updated_at": now,
            }
            _atomic_write_json(paths.dispatch, queued_dispatch)
            _atomic_write_json(paths.state, queued_state)
            # result.json is the sole terminal authority. Move it only after
            # the complete Harbor archive and queued projections are durable.
            os.replace(paths.result, terminal_archive / "result.json")
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)
            receipt_path = terminal_archive / "requeue.json"
            receipt = {
                "schema": "terminal_bench_committed_proot_localhost_requeue_v1",
                "incident": _PROOT_LOCALHOST_INCIDENT,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "execution_attempt": execution,
                "proot_localhost_repair_count": 1,
                "evidence": dict(evidence),
                "archived_artifact_sha256": candidate["artifact_sha256"],
                "harbor_archive_path": harbor_relative,
                "terminal_archive_path": str(
                    terminal_archive.relative_to(spec.paths.root)
                ),
                "official_attempt_consumed_by_repair": False,
                "requeue_policy": _PROOT_LOCALHOST_INCIDENT,
                "requeued_at": now,
            }
            _atomic_write_json(receipt_path, receipt)
            repaired.append(cell.cell_id)
            classifications[cell.cell_id] = str(evidence["classifier"])
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_proot_localhost_incident_requeued",
                    "incident": _PROOT_LOCALHOST_INCIDENT,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "classifier": evidence["classifier"],
                    "official_attempt_consumed_by_repair": False,
                    "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                    "recorded_at": now,
                },
            )
    finally:
        for descriptor in reversed(locked_descriptors):
            _close_lock(descriptor)
    pool_state = aggregate_pool(spec)
    return {
        "incident": _PROOT_LOCALHOST_INCIDENT,
        "repaired_cell_ids": repaired,
        "classifications": classifications,
        **pool_state,
    }


def requeue_committed_codex_host_rg_incident(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Archive and requeue explicitly named cells from one diagnosed incident.

    This is intentionally not a generic Codex retry.  Every requested cell is
    locked and validated before the first mutation.  Newer cells must carry the
    exact persisted missing-host-``rg`` gateway exception.  The bounded cohort
    committed before that diagnostic journal existed is accepted only when it
    has the exact zero-token/high-demand transcript and no gateway artifacts.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested_order = list(cell_ids)
    if not requested_order:
        raise PoolError("repair-codex-host-rg requires at least one cell id")
    if len(set(requested_order)) != len(requested_order):
        raise PoolError("repair-codex-host-rg received a duplicate cell id")
    cells_by_id = {cell.cell_id: cell for cell in spec.cells}
    unknown = set(requested_order) - set(cells_by_id)
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    agents = spec.config.get("agents")
    if (
        not isinstance(agents, list)
        or len(agents) != 1
        or not isinstance(agents[0], Mapping)
        or agents[0].get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
    ):
        raise PoolError("pool does not use the Codex r^2eAct agent contract")

    locked_descriptors: list[int] = []
    candidates: list[dict[str, Any]] = []
    try:
        # Use caller order as an explicit authorization list. No unrequested
        # cell artifact is inspected, classified, archived, or requeued.
        for cell_id in requested_order:
            cell = cells_by_id[cell_id]
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                raise PoolError(f"requested cell is currently owned: {cell_id}")
            locked_descriptors.append(descriptor)

            result = _cell_result(spec, cell)
            if result is None:
                raise PoolError(f"requested cell has no terminal result: {cell_id}")
            execution = _valid_execution(result.get("execution_attempt"))
            if execution < 1:
                raise PoolError(f"Codex host-rg result has no execution: {cell_id}")
            state = _read_json(paths.state)
            dispatch = _read_json(paths.dispatch)
            if (
                result.get("status") != "completed"
                or result.get("outcome") != "errored"
                or result.get("official_attempt_index") != cell.official_attempt_index
                or state.get("status") != "completed"
                or state.get("outcome") != "errored"
                or state.get("execution_attempt") != execution
                or state.get("official_attempt_index") != cell.official_attempt_index
                or dispatch.get("status") != "terminal"
                or dispatch.get("outcome") != "errored"
                or dispatch.get("execution_attempt") != execution
                or dispatch.get("official_attempt_index") != cell.official_attempt_index
            ):
                raise PoolError(
                    f"committed Codex host-rg metadata is inconsistent: {cell_id}"
                )
            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    f"committed Codex host-rg cell has no Harbor trial: {cell_id}"
                )
            trial_path, trial = terminal
            trial_config = trial.get("config")
            trial_agent = (
                trial_config.get("agent") if isinstance(trial_config, Mapping) else None
            )
            if (
                not isinstance(trial_agent, Mapping)
                or trial_agent.get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
                or trial.get("exception_info") != result.get("exception_info")
            ):
                raise PoolError(
                    f"Harbor Codex agent/failure evidence disagrees: {cell_id}"
                )
            agent_dir = trial_path / "agent"
            pool_evidence = _codex_host_rg_incident_failure(result, agent_dir=agent_dir)
            harbor_evidence = _codex_host_rg_incident_failure(
                trial, agent_dir=agent_dir
            )
            if pool_evidence is None or harbor_evidence != pool_evidence:
                raise PoolError(
                    f"cell does not match the exact Codex host-rg incident: {cell_id}"
                )

            attempt_dir = paths.attempts_dir / f"execution-{execution:04d}"
            terminal_archive = attempt_dir / "committed-terminal-codex-host-rg-incident"
            harbor_archive = attempt_dir / "harbor-job-committed-codex-host-rg-incident"
            for archive in (terminal_archive, harbor_archive):
                if archive.exists() or archive.is_symlink():
                    raise PoolError(
                        f"Codex host-rg incident archive already exists: {archive}"
                    )
            raw_repair_count = dispatch.get("codex_host_rg_repair_count", 0)
            raw_worker_exception_count = dispatch.get("worker_exception_count", 0)
            if (
                not isinstance(raw_repair_count, int)
                or isinstance(raw_repair_count, bool)
                or raw_repair_count < 0
                or not isinstance(raw_worker_exception_count, int)
                or isinstance(raw_worker_exception_count, bool)
                or raw_worker_exception_count < 0
            ):
                raise PoolError(f"invalid Codex host-rg retry metadata: {cell.cell_id}")
            result_bytes = _read_regular_bytes(
                paths.result, max_bytes=MAX_ARTIFACT_BYTES
            )
            state_bytes = _read_regular_bytes(paths.state, max_bytes=MAX_ARTIFACT_BYTES)
            dispatch_bytes = _read_regular_bytes(
                paths.dispatch, max_bytes=MAX_ARTIFACT_BYTES
            )
            candidates.append(
                {
                    "claim": claim,
                    "result": result,
                    "state": state,
                    "dispatch": dispatch,
                    "trial_path": trial_path,
                    "evidence": pool_evidence,
                    "repair_count": raw_repair_count + 1,
                    "worker_exception_count": raw_worker_exception_count,
                    "attempt_dir": attempt_dir,
                    "terminal_archive": terminal_archive,
                    "artifact_sha256": {
                        "result.json": hashlib.sha256(result_bytes).hexdigest(),
                        "state.json": hashlib.sha256(state_bytes).hexdigest(),
                        "dispatch.json": hashlib.sha256(dispatch_bytes).hexdigest(),
                    },
                }
            )

        repaired: list[str] = []
        classifications: dict[str, str] = {}
        for candidate in candidates:
            claim = candidate["claim"]
            cell = claim.cell
            paths = claim.paths
            execution = claim.execution_attempt
            state = candidate["state"]
            dispatch = candidate["dispatch"]
            evidence = candidate["evidence"]
            attempt_dir = _ensure_dir(candidate["attempt_dir"])
            terminal_archive = candidate["terminal_archive"]
            _ensure_dir(terminal_archive)
            # Keep byte-equivalent committed metadata in the recovery archive.
            _atomic_write_json(terminal_archive / "state.json", state)
            _atomic_write_json(terminal_archive / "dispatch.json", dispatch)
            harbor_archive = _archive_codex_host_rg_harbor_job(
                claim,
                trial_path=candidate["trial_path"],
                evidence=evidence,
            )

            repair_count = candidate["repair_count"]
            now = _utc_now()
            harbor_relative = str(harbor_archive.relative_to(spec.paths.root))
            queued_dispatch = dict(dispatch)
            queued_dispatch.update(
                {
                    "status": "queued",
                    "codex_host_rg_repair_count": repair_count,
                    "last_codex_host_rg_incident": _CODEX_HOST_RG_INCIDENT,
                    "last_codex_host_rg_classifier": evidence["classifier"],
                    "last_codex_host_rg_execution_attempt": execution,
                    "last_codex_host_rg_archive": harbor_relative,
                    "updated_at": now,
                }
            )
            for key in ("claim", "outcome", "harbor_job_path", "retry_not_before"):
                queued_dispatch.pop(key, None)
            queued_state = {
                "schema": CELL_STATE_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "execution_attempt": execution,
                "worker_exception_count": candidate["worker_exception_count"],
                "codex_host_rg_repair_count": repair_count,
                "last_codex_host_rg_incident": _CODEX_HOST_RG_INCIDENT,
                "last_codex_host_rg_classifier": evidence["classifier"],
                "last_codex_host_rg_execution_attempt": execution,
                "last_codex_host_rg_archive": harbor_relative,
                "updated_at": now,
            }
            _atomic_write_json(paths.dispatch, queued_dispatch)
            _atomic_write_json(paths.state, queued_state)
            # result.json is the sole terminal authority. Retire it only after
            # the full Harbor archive and fresh queued metadata are durable.
            os.replace(paths.result, terminal_archive / "result.json")
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)
            receipt_path = terminal_archive / "requeue.json"
            receipt = {
                "schema": "terminal_bench_committed_codex_host_rg_requeue_v1",
                "incident": _CODEX_HOST_RG_INCIDENT,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "official_attempt_index": cell.official_attempt_index,
                "execution_attempt": execution,
                "codex_host_rg_repair_count": repair_count,
                "evidence": dict(evidence),
                "archived_artifact_sha256": candidate["artifact_sha256"],
                "harbor_archive_path": harbor_relative,
                "terminal_archive_path": str(
                    terminal_archive.relative_to(spec.paths.root)
                ),
                "official_attempt_consumed_by_repair": False,
                "requeue_policy": _CODEX_HOST_RG_INCIDENT,
                "requeued_at": now,
            }
            _atomic_write_json(receipt_path, receipt)
            repaired.append(cell.cell_id)
            classifications[cell.cell_id] = str(evidence["classifier"])
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_codex_host_rg_incident_requeued",
                    "incident": _CODEX_HOST_RG_INCIDENT,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "classifier": evidence["classifier"],
                    "official_attempt_consumed_by_repair": False,
                    "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                    "recorded_at": now,
                },
            )
    finally:
        for descriptor in reversed(locked_descriptors):
            _close_lock(descriptor)
    pool_state = aggregate_pool(spec)
    return {
        "incident": _CODEX_HOST_RG_INCIDENT,
        "repaired_cell_ids": repaired,
        "classifications": classifications,
        **pool_state,
    }


def requeue_committed_codex_gateway_timeout_incident(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Archive and requeue only explicitly named, hash-pinned timeout cells.

    This is a one-off attribution repair for the audited 2026-08-31 cohort,
    not a generic zero-reward or Codex-high-demand retry.  Every requested cell
    is locked and fully validated before the first mutation.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested_order = list(cell_ids)
    if not requested_order:
        raise PoolError("repair-codex-gateway-timeout requires a cell id")
    if len(set(requested_order)) != len(requested_order):
        raise PoolError("repair-codex-gateway-timeout received a duplicate cell id")
    cells_by_id = {cell.cell_id: cell for cell in spec.cells}
    unknown = set(requested_order) - set(cells_by_id)
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    outside_incident = set(requested_order) - set(_CODEX_GATEWAY_TIMEOUT_INCIDENT_CELLS)
    if outside_incident:
        raise PoolError(
            "cell ids are outside the frozen Codex gateway-timeout incident: "
            + ", ".join(sorted(outside_incident))
        )
    agents = spec.config.get("agents")
    environment = spec.config.get("environment")
    if (
        not isinstance(agents, list)
        or len(agents) != 1
        or not isinstance(agents[0], Mapping)
        or agents[0].get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
        or not isinstance(environment, Mapping)
        or environment.get("import_path") != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
    ):
        raise PoolError("pool does not use the audited Codex/PRoot contract")

    locked_descriptors: list[int] = []
    candidates: list[dict[str, Any]] = []
    try:
        for cell_id in requested_order:
            cell = cells_by_id[cell_id]
            contract = _CODEX_GATEWAY_TIMEOUT_INCIDENT_CELLS[cell_id]
            expected_hashes = contract.get("artifact_sha256")
            if (
                cell.task_name != contract.get("task_name")
                or cell.task_ref != contract.get("task_ref")
                or cell.source != contract.get("source")
                or cell.official_attempt_index != contract.get("official_attempt_index")
                or cell.official_attempt_index != 1
                or not isinstance(expected_hashes, Mapping)
                or set(expected_hashes) != _CODEX_GATEWAY_TIMEOUT_ARTIFACT_KEYS
                or not _cell_is_score_eligible(
                    cell, _attempt_selection(spec), _task_selection(spec)
                )
            ):
                raise PoolError(
                    f"cell is outside the exact gateway-timeout contract: {cell_id}"
                )
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                raise PoolError(f"requested cell is currently owned: {cell_id}")
            locked_descriptors.append(descriptor)
            if any(
                path.exists() or path.is_symlink()
                for path in (
                    paths.external_reservation,
                    paths.exception,
                    paths.infrastructure,
                    paths.task_cache_transport,
                )
            ):
                raise PoolError(
                    f"gateway-timeout cell has unrelated retry provenance: {cell_id}"
                )

            result = _cell_result(spec, cell)
            if result is None:
                raise PoolError(f"requested cell has no terminal result: {cell_id}")
            execution = _valid_execution(result.get("execution_attempt"))
            if (
                execution != 1
                or result.get("schema") != CELL_RESULT_SCHEMA
                or result.get("cell_id") != cell.cell_id
                or result.get("task_name") != cell.task_name
                or result.get("task_ref") != cell.task_ref
                or result.get("source") != cell.source
                or result.get("official_attempt_index") != 1
                or result.get("config_contract_hash") != spec.contract_hash
                or result.get("status") != "completed"
                or result.get("outcome") != "errored"
                or result.get("rewards") != {}
                or result.get("rejected_verifier_rewards") != {"reward": 0.0}
                or result.get("verified_agent_exit_warning") is not None
                or result.get("infrastructure_cleanup_warning") is not None
                or result.get("external_import") is not None
                or result.get("verifier_replay") is not None
            ):
                raise PoolError(
                    f"committed gateway-timeout result is inconsistent: {cell_id}"
                )
            state = _read_json(paths.state)
            dispatch = _read_json(paths.dispatch)
            raw_repair_count = dispatch.get("codex_gateway_timeout_repair_count", 0)
            raw_worker_exception_count = dispatch.get("worker_exception_count", 0)
            if (
                state.get("status") != "completed"
                or state.get("outcome") != "errored"
                or state.get("execution_attempt") != 1
                or state.get("official_attempt_index") != 1
                or dispatch.get("status") != "terminal"
                or dispatch.get("outcome") != "errored"
                or dispatch.get("execution_attempt") != 1
                or dispatch.get("official_attempt_index") != 1
                or raw_repair_count != 0
                or raw_worker_exception_count != 0
            ):
                raise PoolError(
                    f"committed gateway-timeout metadata is inconsistent: {cell_id}"
                )
            run_path = paths.attempts_dir / "execution-0001" / "run.json"
            run = _read_json(run_path, max_bytes=MAX_ARTIFACT_BYTES)
            if (
                run.get("schema") != RUN_RECORD_SCHEMA
                or run.get("cell_id") != cell.cell_id
                or run.get("execution_attempt") != 1
                or run.get("action") != "run"
                or run.get("status") != "exited"
                or run.get("return_code") != 1
                or run.get("environment_backend") not in {None, "frozen"}
            ):
                raise PoolError(
                    f"gateway-timeout execution receipt is inconsistent: {cell_id}"
                )
            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    f"gateway-timeout cell has no unique Harbor trial: {cell_id}"
                )
            trial_path, trial = terminal
            trial_config = trial.get("config")
            trial_agent = (
                trial_config.get("agent") if isinstance(trial_config, Mapping) else None
            )
            agent_result = trial.get("agent_result")
            if (
                not isinstance(trial_agent, Mapping)
                or trial_agent.get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
                or trial.get("trial_name") != trial_path.name
                or trial.get("task_name") != cell.task_name
                or trial.get("source") != cell.source
                or _result_rewards(trial) != {"reward": 0.0}
                or _digest(trial.get("exception_info"))
                != _digest(result.get("exception_info"))
                or result.get("trial_name") != trial_path.name
                or result.get("trial_path")
                != str(trial_path.relative_to(spec.paths.root))
                or result.get("harbor_job_path")
                != str(paths.harbor_job.relative_to(spec.paths.root))
                or result.get("started_at") != trial.get("started_at")
                or result.get("finished_at") != trial.get("finished_at")
                or not isinstance(agent_result, Mapping)
                or result.get("n_input_tokens")
                != int(agent_result.get("n_input_tokens") or 0)
                or result.get("n_output_tokens")
                != int(agent_result.get("n_output_tokens") or 0)
                or _verified_success_after_agent_sigterm(trial_path, trial) is not None
            ):
                raise PoolError(
                    f"pool and Harbor gateway-timeout provenance disagree: {cell_id}"
                )

            artifact_paths = {
                "result.json": paths.result,
                "state.json": paths.state,
                "dispatch.json": paths.dispatch,
                "run.json": run_path,
                "trial/result.json": trial_path / "result.json",
                "agent/codex.txt": trial_path / "agent" / "codex.txt",
                "agent/codex-r2eact-gateway-events.jsonl": (
                    trial_path / "agent" / "codex-r2eact-gateway-events.jsonl"
                ),
                "agent/codex-r2eact-gateway-receipts.json": (
                    trial_path / "agent" / "codex-r2eact-gateway-receipts.json"
                ),
                "verifier/reward.txt": trial_path / "verifier" / "reward.txt",
                "verifier/ctrf.json": trial_path / "verifier" / "ctrf.json",
                "proot-runtime.json": trial_path / PROOT_RUNTIME_RECEIPT_NAME,
                "proot-package-bridge.json": (
                    trial_path / PROOT_PACKAGE_BRIDGE_RECEIPT_NAME
                ),
            }
            try:
                artifact_payloads = {
                    name: _read_regular_bytes(path, max_bytes=MAX_ARTIFACT_BYTES)
                    for name, path in artifact_paths.items()
                }
            except PoolError as exc:
                raise PoolError(
                    f"gateway-timeout artifact is unsafe: {cell_id}"
                ) from exc
            artifact_sha256 = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in artifact_payloads.items()
            }
            if dict(expected_hashes) != artifact_sha256:
                raise PoolError(f"gateway-timeout artifact hashes changed: {cell_id}")
            evidence = _codex_gateway_timeout_incident_evidence(
                trial_path, trial, contract=contract
            )
            if evidence is None:
                raise PoolError(f"cell lacks exact gateway-timeout evidence: {cell_id}")

            attempt_dir = paths.attempts_dir / "execution-0001"
            terminal_archive = (
                attempt_dir / "committed-terminal-codex-gateway-timeout-incident"
            )
            harbor_archive = (
                attempt_dir / "harbor-job-committed-codex-gateway-timeout-incident"
            )
            for archive in (terminal_archive, harbor_archive):
                if archive.exists() or archive.is_symlink():
                    raise PoolError(
                        f"gateway-timeout incident archive exists: {archive}"
                    )
            candidates.append(
                {
                    "claim": claim,
                    "state": state,
                    "dispatch": dispatch,
                    "trial_path": trial_path,
                    "evidence": evidence,
                    "contract_sha256": _digest(contract),
                    "artifact_payloads": artifact_payloads,
                    "artifact_sha256": artifact_sha256,
                    "attempt_dir": attempt_dir,
                    "terminal_archive": terminal_archive,
                }
            )

        repaired: list[str] = []
        for candidate in candidates:
            claim = candidate["claim"]
            cell = claim.cell
            paths = claim.paths
            execution = claim.execution_attempt
            evidence = candidate["evidence"]
            artifact_payloads = candidate["artifact_payloads"]
            _ensure_dir(candidate["attempt_dir"])
            terminal_archive = _ensure_dir(candidate["terminal_archive"])
            _atomic_write_bytes(
                terminal_archive / "state.json", artifact_payloads["state.json"]
            )
            _atomic_write_bytes(
                terminal_archive / "dispatch.json",
                artifact_payloads["dispatch.json"],
            )
            harbor_archive = _archive_codex_gateway_timeout_harbor_job(
                claim,
                trial_path=candidate["trial_path"],
                evidence=evidence,
            )

            now = _utc_now()
            harbor_relative = str(harbor_archive.relative_to(spec.paths.root))
            queued_dispatch = dict(candidate["dispatch"])
            queued_dispatch.update(
                {
                    "status": "queued",
                    "codex_gateway_timeout_repair_count": 1,
                    "last_codex_gateway_timeout_incident": (
                        _CODEX_GATEWAY_TIMEOUT_INCIDENT
                    ),
                    "last_codex_gateway_timeout_execution_attempt": execution,
                    "last_codex_gateway_timeout_archive": harbor_relative,
                    "updated_at": now,
                }
            )
            for key in ("claim", "outcome", "harbor_job_path", "retry_not_before"):
                queued_dispatch.pop(key, None)
            queued_state = {
                "schema": CELL_STATE_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "execution_attempt": execution,
                "worker_exception_count": 0,
                "codex_gateway_timeout_repair_count": 1,
                "last_codex_gateway_timeout_incident": (
                    _CODEX_GATEWAY_TIMEOUT_INCIDENT
                ),
                "last_codex_gateway_timeout_execution_attempt": execution,
                "last_codex_gateway_timeout_archive": harbor_relative,
                "updated_at": now,
            }
            _atomic_write_json(paths.dispatch, queued_dispatch)
            _atomic_write_json(paths.state, queued_state)
            # result.json remains the terminal authority until the complete
            # Harbor archive and fresh queue projections are durable.
            os.replace(paths.result, terminal_archive / "result.json")
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)
            receipt_path = terminal_archive / "requeue.json"
            receipt = {
                "schema": ("terminal_bench_committed_codex_gateway_timeout_requeue_v1"),
                "incident": _CODEX_GATEWAY_TIMEOUT_INCIDENT,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "execution_attempt": execution,
                "codex_gateway_timeout_repair_count": 1,
                "incident_contract_sha256": candidate["contract_sha256"],
                "evidence": dict(evidence),
                "archived_artifact_sha256": candidate["artifact_sha256"],
                "harbor_archive_path": harbor_relative,
                "terminal_archive_path": str(
                    terminal_archive.relative_to(spec.paths.root)
                ),
                "official_attempt_consumed_by_repair": False,
                "requeue_policy": _CODEX_GATEWAY_TIMEOUT_INCIDENT,
                "requeued_at": now,
            }
            _atomic_write_json(receipt_path, receipt)
            repaired.append(cell.cell_id)
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_codex_gateway_timeout_incident_requeued",
                    "incident": _CODEX_GATEWAY_TIMEOUT_INCIDENT,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "classifier": evidence["classifier"],
                    "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                    "official_attempt_consumed_by_repair": False,
                    "recorded_at": now,
                },
            )
    finally:
        for descriptor in reversed(locked_descriptors):
            _close_lock(descriptor)
    pool_state = aggregate_pool(spec)
    return {
        "incident": _CODEX_GATEWAY_TIMEOUT_INCIDENT,
        "repaired_cell_ids": repaired,
        **pool_state,
    }


def requeue_committed_codex_gateway_request_timeout(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Archive and requeue explicit host-controller gateway timeouts.

    This is deliberately cell-id-only.  Every requested cell is locked and
    every pool, Harbor, controller, verifier, and runtime artifact is validated
    before the first archive or queue projection is written.  Unlike the
    historical hash-pinned incident repair, this path supports any positive
    execution number while requiring private controller evidence generated by
    the current Codex r^2eAct harness.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested_order = list(cell_ids)
    if not requested_order:
        raise PoolError(
            "repair-codex-gateway-request-timeout requires at least one cell id"
        )
    if len(set(requested_order)) != len(requested_order):
        raise PoolError(
            "repair-codex-gateway-request-timeout received a duplicate cell id"
        )
    cells_by_id = {cell.cell_id: cell for cell in spec.cells}
    requested = set(requested_order)
    unknown = requested - set(cells_by_id)
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    agents = spec.config.get("agents")
    environment = spec.config.get("environment")
    if (
        not isinstance(agents, list)
        or len(agents) != 1
        or not isinstance(agents[0], Mapping)
        or agents[0].get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
        or not isinstance(environment, Mapping)
        or environment.get("import_path") != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
    ):
        raise PoolError("pool does not use the current Codex/PRoot contract")

    locked_descriptors: list[int] = []
    candidates: list[dict[str, Any]] = []
    try:
        # Pool order is a deterministic multi-cell lock order.  Holding every
        # descriptor through validation and mutation prevents both deadlocks
        # and a worker claim between the two phases.
        for cell in spec.cells:
            if cell.cell_id not in requested:
                continue
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                raise PoolError(f"requested cell is currently owned: {cell.cell_id}")
            locked_descriptors.append(descriptor)
            if not _cell_is_score_eligible(
                cell, _attempt_selection(spec), _task_selection(spec)
            ):
                raise PoolError(f"requested cell is not score eligible: {cell.cell_id}")
            if any(
                path.exists() or path.is_symlink()
                for path in (
                    paths.external_reservation,
                    paths.exception,
                    paths.infrastructure,
                    paths.task_cache_transport,
                )
            ):
                raise PoolError(
                    "gateway request timeout cell has unrelated retry provenance: "
                    f"{cell.cell_id}"
                )

            result = _cell_result(spec, cell)
            if result is None:
                raise PoolError(
                    f"requested cell has no terminal result: {cell.cell_id}"
                )
            execution = _valid_execution(result.get("execution_attempt"))
            if execution < 1:
                raise PoolError(
                    f"gateway request timeout has no execution: {cell.cell_id}"
                )
            identity = {
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
            }
            if (
                result.get("schema") != CELL_RESULT_SCHEMA
                or any(result.get(key) != value for key, value in identity.items())
                or result.get("source") != cell.source
                or result.get("status") != "completed"
                or result.get("outcome") != "errored"
                or result.get("rewards") != {}
                or result.get("rejected_verifier_rewards") != {"reward": 0.0}
                or result.get("verified_agent_exit_warning") is not None
                or result.get("infrastructure_cleanup_warning") is not None
                or result.get("external_import") is not None
                or result.get("verifier_replay") is not None
            ):
                raise PoolError(
                    "committed gateway request timeout result is inconsistent: "
                    f"{cell.cell_id}"
                )
            state = _read_json(paths.state)
            dispatch = _read_json(paths.dispatch)
            common_metadata = (state, dispatch)
            if any(
                record.get("execution_attempt") != execution
                or any(record.get(key) != value for key, value in identity.items())
                for record in common_metadata
            ) or (
                state.get("schema") != CELL_STATE_SCHEMA
                or state.get("status") != "completed"
                or state.get("outcome") != "errored"
                or dispatch.get("schema") != CELL_DISPATCH_SCHEMA
                or dispatch.get("status") != "terminal"
                or dispatch.get("outcome") != "errored"
            ):
                raise PoolError(
                    "committed gateway request timeout metadata is inconsistent: "
                    f"{cell.cell_id}"
                )
            raw_worker_exception_count = dispatch.get("worker_exception_count", 0)
            raw_repair_count = dispatch.get(
                "codex_gateway_request_timeout_repair_count", 0
            )
            if (
                not isinstance(raw_worker_exception_count, int)
                or isinstance(raw_worker_exception_count, bool)
                or raw_worker_exception_count < 0
                or not isinstance(raw_repair_count, int)
                or isinstance(raw_repair_count, bool)
                or raw_repair_count < 0
            ):
                raise PoolError(
                    f"invalid gateway request timeout retry metadata: {cell.cell_id}"
                )

            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    "committed gateway request timeout has no unique Harbor trial: "
                    f"{cell.cell_id}"
                )
            trial_path, trial = terminal
            trial_config = trial.get("config")
            trial_agent = (
                trial_config.get("agent") if isinstance(trial_config, Mapping) else None
            )
            agent_result = trial.get("agent_result")
            if (
                not isinstance(trial_agent, Mapping)
                or trial_agent.get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
                or trial.get("trial_name") != trial_path.name
                or trial.get("task_name") != cell.task_name
                or trial.get("source") != cell.source
                or _result_rewards(trial) != {"reward": 0.0}
                or _digest(trial.get("exception_info"))
                != _digest(result.get("exception_info"))
                or result.get("trial_name") != trial_path.name
                or result.get("trial_path")
                != str(trial_path.relative_to(spec.paths.root))
                or result.get("harbor_job_path")
                != str(paths.harbor_job.relative_to(spec.paths.root))
                or result.get("started_at") != trial.get("started_at")
                or result.get("finished_at") != trial.get("finished_at")
                or not isinstance(agent_result, Mapping)
                or result.get("n_input_tokens")
                != int(agent_result.get("n_input_tokens") or 0)
                or result.get("n_output_tokens")
                != int(agent_result.get("n_output_tokens") or 0)
                or _verified_success_after_agent_sigterm(trial_path, trial) is not None
            ):
                raise PoolError(
                    "pool and Harbor gateway request timeout provenance disagree: "
                    f"{cell.cell_id}"
                )

            run_path = paths.attempts_dir / f"execution-{execution:04d}" / "run.json"
            controller_dir = trial_path / "agent" / CONTROLLER_ARTIFACT_DIRNAME
            canonical_journal = controller_dir / GATEWAY_RECEIPTS_JSONL_FILENAME
            canonical_summary = controller_dir / GATEWAY_RECEIPTS_SUMMARY_FILENAME
            canonical_receipts_seen = any(
                path.exists() or path.is_symlink()
                for path in (canonical_journal, canonical_summary)
            )
            receipt_artifact_paths = (
                {
                    (
                        f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/"
                        f"{GATEWAY_RECEIPTS_JSONL_FILENAME}"
                    ): canonical_journal,
                    (
                        f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/"
                        f"{GATEWAY_RECEIPTS_SUMMARY_FILENAME}"
                    ): canonical_summary,
                }
                if canonical_receipts_seen
                else {
                    (
                        f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/"
                        f"{GATEWAY_RECEIPTS_FILENAME}"
                    ): controller_dir
                    / GATEWAY_RECEIPTS_FILENAME,
                }
            )
            artifact_paths = {
                "result.json": paths.result,
                "state.json": paths.state,
                "dispatch.json": paths.dispatch,
                "run.json": run_path,
                "trial/result.json": trial_path / "result.json",
                "agent/codex.txt": trial_path / "agent" / "codex.txt",
                (f"agent/{CONTROLLER_ARTIFACT_DIRNAME}/{GATEWAY_EVENTS_FILENAME}"): (
                    controller_dir / GATEWAY_EVENTS_FILENAME
                ),
                **receipt_artifact_paths,
                "verifier/reward.txt": trial_path / "verifier" / "reward.txt",
                "verifier/ctrf.json": trial_path / "verifier" / "ctrf.json",
                "proot-runtime.json": trial_path / PROOT_RUNTIME_RECEIPT_NAME,
                "proot-package-bridge.json": (
                    trial_path / PROOT_PACKAGE_BRIDGE_RECEIPT_NAME
                ),
            }
            if frozenset(artifact_paths) not in (
                _CODEX_GATEWAY_REQUEST_TIMEOUT_ARTIFACT_KEY_SETS
            ):
                raise AssertionError("gateway request timeout artifact contract drift")
            try:
                artifact_payloads = {
                    name: _read_regular_bytes(path, max_bytes=MAX_ARTIFACT_BYTES)
                    for name, path in artifact_paths.items()
                }
            except PoolError as exc:
                raise PoolError(
                    f"gateway request timeout artifact is unsafe: {cell.cell_id}"
                ) from exc
            try:
                raw_result = json.loads(artifact_payloads["result.json"])
                raw_state = json.loads(artifact_payloads["state.json"])
                raw_dispatch = json.loads(artifact_payloads["dispatch.json"])
                run = json.loads(artifact_payloads["run.json"])
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise PoolError(
                    f"gateway request timeout artifact is invalid: {cell.cell_id}"
                ) from exc
            if (
                not isinstance(raw_result, Mapping)
                or dict(raw_result) != dict(result)
                or not isinstance(raw_state, Mapping)
                or dict(raw_state) != dict(state)
                or not isinstance(raw_dispatch, Mapping)
                or dict(raw_dispatch) != dict(dispatch)
                or not isinstance(run, Mapping)
                or run.get("schema") != RUN_RECORD_SCHEMA
                or run.get("cell_id") != cell.cell_id
                or run.get("execution_attempt") != execution
                or run.get("action") != "run"
                or run.get("status") != "exited"
                or run.get("return_code") != 1
                or run.get("environment_backend") not in {None, "frozen"}
            ):
                raise PoolError(
                    "gateway request timeout execution receipt is inconsistent: "
                    f"{cell.cell_id}"
                )
            evidence = _codex_gateway_request_timeout_evidence(
                trial_path,
                trial,
                artifact_payloads=artifact_payloads,
            )
            if evidence is None:
                raise PoolError(
                    "cell lacks strict host-controller gateway request timeout "
                    f"evidence: {cell.cell_id}"
                )

            attempt_dir = paths.attempts_dir / f"execution-{execution:04d}"
            terminal_archive = (
                attempt_dir / _CODEX_GATEWAY_REQUEST_TIMEOUT_TERMINAL_ARCHIVE
            )
            harbor_archive = attempt_dir / _CODEX_GATEWAY_REQUEST_TIMEOUT_HARBOR_ARCHIVE
            for archive in (terminal_archive, harbor_archive):
                if archive.exists() or archive.is_symlink():
                    raise PoolError(
                        f"gateway request timeout archive already exists: {archive}"
                    )
            candidates.append(
                {
                    "claim": claim,
                    "state": state,
                    "dispatch": dispatch,
                    "trial_path": trial_path,
                    "evidence": evidence,
                    "artifact_payloads": artifact_payloads,
                    "artifact_sha256": {
                        name: hashlib.sha256(payload).hexdigest()
                        for name, payload in artifact_payloads.items()
                    },
                    "attempt_dir": attempt_dir,
                    "terminal_archive": terminal_archive,
                    "repair_count": raw_repair_count + 1,
                    "worker_exception_count": raw_worker_exception_count,
                }
            )

        repaired: list[str] = []
        classifications: dict[str, str] = {}
        for candidate in candidates:
            claim = candidate["claim"]
            cell = claim.cell
            paths = claim.paths
            execution = claim.execution_attempt
            evidence = candidate["evidence"]
            artifact_payloads = candidate["artifact_payloads"]
            _ensure_dir(candidate["attempt_dir"])
            terminal_archive = _ensure_dir(candidate["terminal_archive"])
            _atomic_write_bytes(
                terminal_archive / "state.json", artifact_payloads["state.json"]
            )
            _atomic_write_bytes(
                terminal_archive / "dispatch.json", artifact_payloads["dispatch.json"]
            )
            harbor_archive = _archive_codex_gateway_request_timeout_harbor_job(
                claim,
                trial_path=candidate["trial_path"],
                evidence=evidence,
            )

            now = _utc_now()
            repair_count = candidate["repair_count"]
            harbor_relative = str(harbor_archive.relative_to(spec.paths.root))
            queued_dispatch = dict(candidate["dispatch"])
            queued_dispatch.update(
                {
                    "status": "queued",
                    "codex_gateway_request_timeout_repair_count": repair_count,
                    "last_codex_gateway_request_timeout_incident": (
                        _CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT
                    ),
                    "last_codex_gateway_request_timeout_classifier": evidence[
                        "classifier"
                    ],
                    "last_codex_gateway_request_timeout_execution_attempt": execution,
                    "last_codex_gateway_request_timeout_archive": harbor_relative,
                    "updated_at": now,
                }
            )
            for key in ("claim", "outcome", "harbor_job_path", "retry_not_before"):
                queued_dispatch.pop(key, None)
            queued_state = {
                "schema": CELL_STATE_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "execution_attempt": execution,
                "worker_exception_count": candidate["worker_exception_count"],
                "codex_gateway_request_timeout_repair_count": repair_count,
                "last_codex_gateway_request_timeout_incident": (
                    _CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT
                ),
                "last_codex_gateway_request_timeout_classifier": evidence["classifier"],
                "last_codex_gateway_request_timeout_execution_attempt": execution,
                "last_codex_gateway_request_timeout_archive": harbor_relative,
                "updated_at": now,
            }
            _atomic_write_json(paths.dispatch, queued_dispatch)
            _atomic_write_json(paths.state, queued_state)
            # result.json is the sole terminal authority.  Move it last, only
            # after the complete Harbor archive and queued projections exist.
            os.replace(paths.result, terminal_archive / "result.json")
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)
            receipt_path = terminal_archive / "requeue.json"
            receipt = {
                "schema": (
                    "terminal_bench_committed_codex_gateway_request_timeout_requeue_v1"
                ),
                "incident": _CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "execution_attempt": execution,
                "codex_gateway_request_timeout_repair_count": repair_count,
                "evidence": dict(evidence),
                "archived_artifact_sha256": candidate["artifact_sha256"],
                "harbor_archive_path": harbor_relative,
                "terminal_archive_path": str(
                    terminal_archive.relative_to(spec.paths.root)
                ),
                "official_attempt_consumed_by_repair": False,
                "requeue_policy": _CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT,
                "requeued_at": now,
            }
            _atomic_write_json(receipt_path, receipt)
            repaired.append(cell.cell_id)
            classifications[cell.cell_id] = str(evidence["classifier"])
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_codex_gateway_request_timeout_requeued",
                    "incident": _CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "classifier": evidence["classifier"],
                    "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                    "official_attempt_consumed_by_repair": False,
                    "recorded_at": now,
                },
            )
    finally:
        for descriptor in reversed(locked_descriptors):
            _close_lock(descriptor)
    pool_state = aggregate_pool(spec)
    return {
        "incident": _CODEX_GATEWAY_REQUEST_TIMEOUT_INCIDENT,
        "repaired_cell_ids": repaired,
        "classifications": classifications,
        **pool_state,
    }


def _local_storage_gateway_evidence(
    trial_path: Path,
) -> list[dict[str, Any]]:
    """Read only typed host-controller failures from the private journal."""

    path = (
        trial_path
        / "agent"
        / CONTROLLER_ARTIFACT_DIRNAME
        / GATEWAY_EVENTS_FILENAME
    )
    if not path.exists():
        return []
    if path.is_symlink():
        raise PoolError(f"local-storage gateway journal is a symlink: {path}")
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PoolError(
            f"could not stat local-storage gateway journal: {path}"
        ) from exc
    if not path.is_file() or before.st_size > _LOCAL_STORAGE_GATEWAY_EVENT_MAX_BYTES:
        raise PoolError(
            f"invalid or oversized local-storage gateway journal: {path}"
        )

    digest = hashlib.sha256()
    matches: list[dict[str, Any]] = []
    last_nonempty_line = 0
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                digest.update(line)
                if line.strip():
                    last_nonempty_line = line_number
                if (
                    b'"gateway_request_failed"' not in line
                    or (
                        b"Disk quota exceeded" not in line
                        and b"ReAct memory corpus exceeds" not in line
                    )
                ):
                    continue
                if len(line) > _LOCAL_STORAGE_GATEWAY_MATCHING_LINE_MAX_BYTES:
                    raise PoolError(
                        "matching local-storage gateway event is oversized: "
                        f"{path}:{line_number}"
                    )
                try:
                    event = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise PoolError(
                        "matching local-storage gateway event is invalid JSON: "
                        f"{path}:{line_number}"
                    ) from exc
                if not isinstance(event, Mapping):
                    continue
                message = event.get("error_message")
                error_type = event.get("error_type")
                traceback_text = event.get("traceback")
                classifier: str | None = None
                if (
                    event.get("event") == "gateway_request_failed"
                    and error_type == "OSError"
                    and message == _DISK_QUOTA_ERROR_MESSAGE
                    and isinstance(traceback_text, str)
                    and "terminal_bench_eval/codex_r2eact_gateway.py" in traceback_text
                    and traceback_text.endswith(
                        f"OSError: {_DISK_QUOTA_ERROR_MESSAGE}\n"
                    )
                ):
                    classifier = "host_gateway_disk_quota_v1"
                elif (
                    event.get("event") == "gateway_request_failed"
                    and error_type == "ValueError"
                    and message == _LEGACY_REACT_MEMORY_CAP_ERROR_MESSAGE
                    and isinstance(traceback_text, str)
                    and "/context_compression/grepseek_tool.py" in traceback_text
                    and "in _write_episode_memory_records\n" in traceback_text
                    and traceback_text.endswith(
                        f"ValueError: {_LEGACY_REACT_MEMORY_CAP_ERROR_MESSAGE}\n"
                    )
                ):
                    classifier = "host_gateway_legacy_memory_mirror_cap_v1"
                if classifier is not None:
                    matches.append(
                        {
                            "classifier": classifier,
                            "source": str(path.relative_to(trial_path)),
                            "line_number": line_number,
                            "event_sha256": _digest(dict(event)),
                            "error_type": error_type,
                            "error_message": message,
                        }
                    )
    except OSError as exc:
        raise PoolError(
            f"could not read local-storage gateway journal: {path}"
        ) from exc
    try:
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise PoolError(
            f"could not restat local-storage gateway journal: {path}"
        ) from exc
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise PoolError(f"local-storage gateway journal changed while read: {path}")
    journal_sha256 = digest.hexdigest()
    # A quota/cap failure that the controller recovered from is not a reason
    # to discard a later model-authored terminal outcome.  The typed failure
    # must therefore be the journal's final committed event.
    matches = [
        match
        for match in matches
        if match["line_number"] == last_nonempty_line
    ]
    for match in matches:
        match["journal_size_bytes"] = before.st_size
        match["journal_sha256"] = journal_sha256
    return matches


def _local_storage_direct_exception_evidence(
    value: Mapping[str, Any] | None,
    *,
    source: str,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    exception = value.get("exception_info")
    if not isinstance(exception, Mapping):
        return None
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception.get("exception_type") != "OSError"
        or message != _DISK_QUOTA_ERROR_MESSAGE
        or not isinstance(traceback_text, str)
        or not traceback_text.endswith(f"OSError: {_DISK_QUOTA_ERROR_MESSAGE}\n")
        or not any(
            marker in traceback_text
            for marker in (
                "/site-packages/harbor/",
                "/eval/terminal_bench/proot_environment.py",
                "/terminal_bench_eval/codex_r2eact_",
            )
        )
    ):
        return None
    return {
        "classifier": "host_exception_disk_quota_v1",
        "source": source,
        "error_type": "OSError",
        "error_message": _DISK_QUOTA_ERROR_MESSAGE,
        "exception_sha256": _digest(dict(exception)),
    }


def _local_storage_log_evidence(
    *,
    spec: PoolSpec,
    paths: CellPaths,
    trial_path: Path,
    execution: int,
) -> list[dict[str, Any]]:
    sources = (
        paths.harbor_job / "job.log",
        trial_path / "trial.log",
        paths.attempts_dir / f"execution-{execution:04d}" / "harbor.log",
    )
    matches: list[dict[str, Any]] = []
    for path in sources:
        if not path.exists():
            continue
        payload = _read_regular_bytes(path, max_bytes=_LOCAL_STORAGE_LOG_MAX_BYTES)
        text = payload.decode("utf-8", errors="replace")
        if (
            _DISK_QUOTA_ERROR_MESSAGE not in text
            and "OSError(122, 'Disk quota exceeded')" not in text
        ):
            continue
        fatal_python = (
            "object repr     : OSError(122, 'Disk quota exceeded')" in text
        )
        host_traceback = (
            f"OSError: {_DISK_QUOTA_ERROR_MESSAGE}" in text
            and "Traceback (most recent call last):" in text
            and any(
                marker in text
                for marker in (
                    "/site-packages/harbor/",
                    "/eval/terminal_bench/proot_environment.py",
                    "/terminal_bench_eval/codex_r2eact_",
                )
            )
        )
        verifier_copy_failure = (
            "shutil.Error:" in text
            and "DownloadVerifierDirError" in text
            and "Failed to download verifier directory from environment" in text
        )
        if not (fatal_python or host_traceback or verifier_copy_failure):
            continue
        matches.append(
            {
                "classifier": "host_log_disk_quota_v1",
                "source": str(path.relative_to(spec.paths.root)),
                "log_sha256": hashlib.sha256(payload).hexdigest(),
                "log_size_bytes": len(payload),
                "fatal_python_diagnostic": fatal_python,
                "host_traceback": host_traceback,
                "verifier_copy_failure": verifier_copy_failure,
            }
        )
    return matches


def _local_storage_trial_context(
    spec: PoolSpec,
    claim: ClaimedCell,
) -> tuple[Path, dict[str, Any] | None]:
    paths = claim.paths
    if paths.harbor_job.is_symlink() or not paths.harbor_job.is_dir():
        raise PoolError(
            f"local-storage incident has no safe Harbor job: {claim.cell.cell_id}"
        )
    stored = _read_json(paths.job_config)
    _stored_single_job_contract(spec, claim, stored)
    observed = _read_json(paths.harbor_job / "config.json")
    if not _harbor_resume_config_matches_contract(observed, stored):
        raise PoolError(
            "local-storage Harbor config violates the cell contract: "
            f"{claim.cell.cell_id}"
        )
    trials = [
        child
        for child in sorted(paths.harbor_job.iterdir(), key=lambda value: value.name)
        if child.is_dir()
        and not child.is_symlink()
        and (child / "config.json").is_file()
    ]
    if len(trials) != 1:
        raise PoolError(
            "local-storage Harbor job must contain exactly one trial: "
            f"{claim.cell.cell_id}"
        )
    trial_path = trials[0]
    trial_config = _read_json(trial_path / "config.json")
    task = trial_config.get("task")
    agent = trial_config.get("agent")
    environment = trial_config.get("environment")
    if (
        trial_config.get("trial_name") != trial_path.name
        or not isinstance(task, Mapping)
        or task.get("name") != claim.cell.task_name
        or task.get("ref") != claim.cell.task_ref
        or task.get("source") != claim.cell.source
        or not isinstance(agent, Mapping)
        or agent.get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
        or not isinstance(environment, Mapping)
        or environment.get("import_path") != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
    ):
        raise PoolError(
            f"local-storage trial config violates cell identity: {claim.cell.cell_id}"
        )

    result_path = trial_path / "result.json"
    if not result_path.exists():
        return trial_path, None
    if result_path.is_symlink() or not result_path.is_file():
        raise PoolError(f"local-storage trial result is unsafe: {result_path}")
    try:
        size = result_path.stat().st_size
    except OSError as exc:
        raise PoolError(
            f"could not stat local-storage trial result: {result_path}"
        ) from exc
    if size == 0:
        return trial_path, None
    trial = _read_json(result_path)
    try:
        TrialResult.model_validate(trial)
    except Exception as exc:
        raise PoolError(
            f"invalid nonempty local-storage Harbor result: {result_path}"
        ) from exc
    _validate_harbor_trial(spec, claim, trial)
    return trial_path, trial


def _codex_local_storage_candidate(
    spec: PoolSpec,
    cell: CellSpec,
    descriptor: int,
) -> dict[str, Any]:
    paths = spec.cell_paths(cell).ensure()
    if not _cell_is_score_eligible(
        cell, _attempt_selection(spec), _task_selection(spec)
    ):
        raise PoolError(f"requested cell is not score eligible: {cell.cell_id}")
    if any(
        path.exists() or path.is_symlink()
        for path in (
            paths.external_reservation,
            paths.infrastructure,
            paths.task_cache_transport,
            paths.harbor_resume_config_repair,
        )
    ):
        raise PoolError(
            f"local-storage cell has unrelated retry provenance: {cell.cell_id}"
        )

    state = _read_json(paths.state)
    dispatch = _read_json(paths.dispatch)
    execution = _valid_execution(state.get("execution_attempt"))
    identity = {
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
    }
    if (
        execution < 1
        or dispatch.get("execution_attempt") != execution
        or any(
            record.get(key) != value
            for record in (state, dispatch)
            for key, value in identity.items()
        )
        or state.get("schema") != CELL_STATE_SCHEMA
        or dispatch.get("schema") != CELL_DISPATCH_SCHEMA
        or dispatch.get("status") != "terminal"
    ):
        raise PoolError(f"local-storage cell metadata is inconsistent: {cell.cell_id}")

    result = _cell_result(spec, cell)
    exception: dict[str, Any] | None = None
    if paths.exception.exists() or paths.exception.is_symlink():
        exception = _read_json(paths.exception)
    if state.get("status") == "completed":
        if (
            result is None
            or state.get("outcome") != "errored"
            or dispatch.get("outcome") != "errored"
            or result.get("status") != "completed"
            or result.get("outcome") != "errored"
            or result.get("rewards") != {}
            or result.get("execution_attempt") != execution
            or exception is not None
        ):
            raise PoolError(
                f"completed local-storage result is inconsistent: {cell.cell_id}"
            )
    elif state.get("status") == "pool_error":
        if (
            state.get("outcome") != "unclassified_worker_exception"
            or dispatch.get("outcome") != "unclassified_worker_exception"
            or exception is None
            or exception.get("schema") != WORKER_EXCEPTION_SCHEMA
            or exception.get("cell_id") != cell.cell_id
            or exception.get("task_name") != cell.task_name
            or exception.get("official_attempt_index")
            != cell.official_attempt_index
            or exception.get("execution_attempt") != execution
            or exception.get("worker_exception_count") != 1
            or state.get("worker_exception_count") != 1
            or dispatch.get("worker_exception_count") != 1
            or (
                result is not None
                and (
                    result.get("status") != "pool_error"
                    or result.get("outcome") != "unclassified_worker_exception"
                    or result.get("rewards") != {}
                    or result.get("execution_attempt") != execution
                    or result.get("worker_exception_count") != 1
                )
            )
        ):
            raise PoolError(
                f"pool-error local-storage result is inconsistent: {cell.cell_id}"
            )
    else:
        raise PoolError(f"requested cell is not terminal: {cell.cell_id}")

    if result is not None and (
        result.get("external_import") is not None
        or result.get("verifier_replay") is not None
        or result.get("verified_agent_exit_warning") is not None
        or result.get("verified_agent_timeout_warning") is not None
    ):
        raise PoolError(
            f"local-storage result has unrelated attribution: {cell.cell_id}"
        )

    claim = ClaimedCell(
        cell=cell,
        paths=paths,
        descriptor=descriptor,
        execution_attempt=execution,
        dispatch=dispatch,
    )
    trial_path, trial = _local_storage_trial_context(spec, claim)
    if trial is not None and result is not None and result.get("status") == "completed":
        if (
            result.get("trial_name") != trial_path.name
            or result.get("trial_path")
            != str(trial_path.relative_to(spec.paths.root))
            or _digest(result.get("exception_info"))
            != _digest(trial.get("exception_info"))
        ):
            raise PoolError(
                f"pool and Harbor local-storage provenance disagree: {cell.cell_id}"
            )

    evidence: list[dict[str, Any]] = []
    evidence.extend(_local_storage_gateway_evidence(trial_path))
    for source, value in (("result.json", result), ("trial/result.json", trial)):
        direct = _local_storage_direct_exception_evidence(value, source=source)
        if direct is not None:
            evidence.append(direct)
    # Host logs are a fallback only when quota exhaustion prevented Harbor
    # from committing a parseable trial result.  A completed trial must carry
    # typed final-journal or direct exception evidence, otherwise an old log
    # line could incorrectly overwrite a later model-authored failure.
    if state.get("status") == "pool_error" and trial is None:
        evidence.extend(
            _local_storage_log_evidence(
                spec=spec,
                paths=paths,
                trial_path=trial_path,
                execution=execution,
            )
        )
    classifications = {str(item["classifier"]) for item in evidence}
    memory_cap = "host_gateway_legacy_memory_mirror_cap_v1" in classifications
    quota = any("disk_quota" in value for value in classifications)
    if memory_cap and quota:
        raise PoolError(f"local-storage incident is ambiguous: {cell.cell_id}")
    if not (memory_cap or quota):
        raise PoolError(
            f"cell lacks strict Codex local-storage evidence: {cell.cell_id}"
        )
    incident_kind = "legacy_memory_mirror_cap" if memory_cap else "disk_quota"

    attempt_dir = paths.attempts_dir / f"execution-{execution:04d}"
    terminal_archive = attempt_dir / _CODEX_LOCAL_STORAGE_TERMINAL_ARCHIVE
    harbor_archive = attempt_dir / _CODEX_LOCAL_STORAGE_HARBOR_ARCHIVE
    for archive in (terminal_archive, harbor_archive):
        if archive.exists() or archive.is_symlink():
            raise PoolError(f"local-storage incident archive already exists: {archive}")
    artifact_paths = {
        "state.json": paths.state,
        "dispatch.json": paths.dispatch,
    }
    if paths.result.is_file() and not paths.result.is_symlink():
        artifact_paths["result.json"] = paths.result
    if paths.exception.is_file() and not paths.exception.is_symlink():
        artifact_paths["exception.json"] = paths.exception
    run_path = attempt_dir / "run.json"
    if run_path.is_file() and not run_path.is_symlink():
        artifact_paths["run.json"] = run_path
    artifact_payloads = {
        name: _read_regular_bytes(path, max_bytes=MAX_ARTIFACT_BYTES)
        for name, path in artifact_paths.items()
    }
    repair_count = dispatch.get("codex_local_storage_repair_count", 0)
    if (
        not isinstance(repair_count, int)
        or isinstance(repair_count, bool)
        or repair_count < 0
    ):
        raise PoolError(f"invalid local-storage repair count: {cell.cell_id}")
    return {
        "claim": claim,
        "state": state,
        "dispatch": dispatch,
        "trial_path": trial_path,
        "incident_kind": incident_kind,
        "evidence": evidence,
        "attempt_dir": attempt_dir,
        "terminal_archive": terminal_archive,
        "harbor_archive": harbor_archive,
        "artifact_payloads": artifact_payloads,
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in artifact_payloads.items()
        },
        "repair_count": repair_count + 1,
    }


def requeue_committed_codex_local_storage_infrastructure(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Archive and requeue explicit, host-proven Codex storage incidents.

    The only accepted incidents are the exact local/project quota error and
    the retired 32 MiB ReAct memory-mirror guard.  Every requested cell flock
    is acquired in immutable pool order before any candidate is validated or
    any artifact is written, so a mixed/owned batch cannot be partly repaired.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested_order = list(cell_ids)
    if not requested_order:
        raise PoolError(
            "repair-codex-local-storage requires at least one cell id"
        )
    if len(set(requested_order)) != len(requested_order):
        raise PoolError("repair-codex-local-storage received a duplicate cell id")
    cells_by_id = {cell.cell_id: cell for cell in spec.cells}
    requested = set(requested_order)
    unknown = requested - set(cells_by_id)
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    agents = spec.config.get("agents")
    environment = spec.config.get("environment")
    if (
        not isinstance(agents, list)
        or len(agents) != 1
        or not isinstance(agents[0], Mapping)
        or agents[0].get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
        or not isinstance(environment, Mapping)
        or environment.get("import_path") != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
    ):
        raise PoolError("pool does not use the current Codex/PRoot contract")

    locked_descriptors: list[int] = []
    descriptors_by_cell: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    try:
        # This phase is intentionally lock-only.  Do not validate (and above
        # all do not write) cell A before proving that cell B is unowned.
        for cell in spec.cells:
            if cell.cell_id not in requested:
                continue
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                raise PoolError(f"requested cell is currently owned: {cell.cell_id}")
            locked_descriptors.append(descriptor)
            descriptors_by_cell[cell.cell_id] = descriptor

        # Validate the complete batch under all flocks before the first write.
        for cell in spec.cells:
            if cell.cell_id in requested:
                candidates.append(
                    _codex_local_storage_candidate(
                        spec, cell, descriptors_by_cell[cell.cell_id]
                    )
                )

        repaired: list[str] = []
        classifications: dict[str, list[str]] = {}
        incident_kinds: dict[str, str] = {}
        for candidate in candidates:
            claim = candidate["claim"]
            cell = claim.cell
            paths = claim.paths
            execution = claim.execution_attempt
            artifact_payloads = candidate["artifact_payloads"]

            # Recheck every small mutable authority against the bytes that
            # were validated while all flocks were held.
            live_artifacts = {
                "state.json": paths.state,
                "dispatch.json": paths.dispatch,
            }
            for optional_name, optional_path in (
                ("result.json", paths.result),
                ("exception.json", paths.exception),
                (
                    "run.json",
                    candidate["attempt_dir"] / "run.json",
                ),
            ):
                if optional_name in artifact_payloads:
                    live_artifacts[optional_name] = optional_path
            for name, path in live_artifacts.items():
                if (
                    _read_regular_bytes(path, max_bytes=MAX_ARTIFACT_BYTES)
                    != artifact_payloads[name]
                ):
                    raise PoolError(
                        f"local-storage artifact changed while locked: {cell.cell_id}"
                    )

            _ensure_dir(candidate["attempt_dir"])
            terminal_archive = _ensure_dir(candidate["terminal_archive"])
            # result/exception are moved below; the other authorities are
            # immutable byte-for-byte snapshots.
            for name in ("state.json", "dispatch.json", "run.json"):
                payload = artifact_payloads.get(name)
                if payload is not None:
                    _atomic_write_bytes(terminal_archive / name, payload)

            evidence = [dict(item) for item in candidate["evidence"]]
            now = _utc_now()
            try:
                relative_trial = str(candidate["trial_path"].relative_to(paths.harbor_job))
            except ValueError as exc:
                raise PoolError(
                    f"local-storage trial escaped Harbor job: {cell.cell_id}"
                ) from exc
            harbor_receipt = {
                "schema": "terminal_bench_codex_local_storage_harbor_archive_v1",
                "incident": _CODEX_LOCAL_STORAGE_INCIDENT,
                "incident_kind": candidate["incident_kind"],
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "execution_attempt": execution,
                "trial_path": relative_trial,
                "evidence": evidence,
                "official_attempt_consumed_by_repair": False,
                "archived_at": now,
            }
            _atomic_write_json(
                paths.harbor_job / _CODEX_LOCAL_STORAGE_HARBOR_RECEIPT,
                harbor_receipt,
            )
            try:
                os.replace(paths.harbor_job, candidate["harbor_archive"])
            except OSError as exc:
                raise PoolError(
                    f"could not archive local-storage Harbor job: {cell.cell_id}"
                ) from exc
            _fsync_directory(paths.root)
            _fsync_directory(candidate["harbor_archive"].parent)

            harbor_relative = str(
                candidate["harbor_archive"].relative_to(spec.paths.root)
            )
            terminal_relative = str(terminal_archive.relative_to(spec.paths.root))
            repair_count = candidate["repair_count"]
            receipt = {
                "schema": "terminal_bench_committed_codex_local_storage_requeue_v1",
                "incident": _CODEX_LOCAL_STORAGE_INCIDENT,
                "incident_kind": candidate["incident_kind"],
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "execution_attempt": execution,
                "codex_local_storage_repair_count": repair_count,
                "evidence": evidence,
                "archived_artifact_sha256": candidate["artifact_sha256"],
                "harbor_archive_path": harbor_relative,
                "terminal_archive_path": terminal_relative,
                "official_attempt_consumed_by_repair": False,
                "requeue_policy": _CODEX_LOCAL_STORAGE_INCIDENT,
                "requeued_at": now,
            }
            _atomic_write_json(terminal_archive / "requeue.json", receipt)

            queued_dispatch = dict(candidate["dispatch"])
            queued_dispatch.update(
                {
                    "status": "queued",
                    "worker_exception_count": 0,
                    "codex_local_storage_repair_count": repair_count,
                    "last_codex_local_storage_incident": (
                        _CODEX_LOCAL_STORAGE_INCIDENT
                    ),
                    "last_codex_local_storage_incident_kind": candidate[
                        "incident_kind"
                    ],
                    "last_codex_local_storage_execution_attempt": execution,
                    "last_codex_local_storage_archive": harbor_relative,
                    "updated_at": now,
                }
            )
            for key in (
                "claim",
                "outcome",
                "harbor_job_path",
                "retry_not_before",
            ):
                queued_dispatch.pop(key, None)
            queued_state = {
                "schema": CELL_STATE_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "execution_attempt": execution,
                "worker_exception_count": 0,
                "codex_local_storage_repair_count": repair_count,
                "last_codex_local_storage_incident": (
                    _CODEX_LOCAL_STORAGE_INCIDENT
                ),
                "last_codex_local_storage_incident_kind": candidate[
                    "incident_kind"
                ],
                "last_codex_local_storage_execution_attempt": execution,
                "last_codex_local_storage_archive": harbor_relative,
                "updated_at": now,
            }
            _atomic_write_json(paths.dispatch, queued_dispatch)
            _atomic_write_json(paths.state, queued_state)

            # The flock prevents a claim while both terminal authorities are
            # retired.  Move rather than unlink so all original bytes remain
            # auditable.  result.json is moved last when both exist.
            for name, path in (
                ("exception.json", paths.exception),
                ("result.json", paths.result),
            ):
                payload = artifact_payloads.get(name)
                if payload is None:
                    continue
                if (
                    _read_regular_bytes(path, max_bytes=MAX_ARTIFACT_BYTES)
                    != payload
                ):
                    raise PoolError(
                        f"local-storage terminal artifact changed: {cell.cell_id}"
                    )
                os.replace(path, terminal_archive / name)
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)

            repaired.append(cell.cell_id)
            classifications[cell.cell_id] = sorted(
                {str(item["classifier"]) for item in evidence}
            )
            incident_kinds[cell.cell_id] = str(candidate["incident_kind"])
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_codex_local_storage_requeued",
                    "incident": _CODEX_LOCAL_STORAGE_INCIDENT,
                    "incident_kind": candidate["incident_kind"],
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "receipt_path": str(
                        (terminal_archive / "requeue.json").relative_to(
                            spec.paths.root
                        )
                    ),
                    "official_attempt_consumed_by_repair": False,
                    "recorded_at": now,
                },
            )
    finally:
        for descriptor in reversed(locked_descriptors):
            _close_lock(descriptor)

    pool_state = aggregate_pool(spec)
    return {
        "incident": _CODEX_LOCAL_STORAGE_INCIDENT,
        "repaired_cell_ids": repaired,
        "incident_kinds": incident_kinds,
        "classifications": classifications,
        **pool_state,
    }


def _regular_tree_manifest(root: Path) -> dict[str, Any]:
    """Hash every regular file in a private tree without following links."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise PoolError(f"snapshot source is not a safe directory: {root}")
    _assert_regular_snapshot_tree(root)
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for directory, names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        names.sort()
        filenames.sort()
        for name in names:
            child = current / name
            if child.is_symlink() or not child.is_dir():
                raise PoolError(f"unsafe directory in snapshot source: {child}")
            directories.append(child.relative_to(root).as_posix())
        for name in filenames:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise PoolError(f"unsafe file in snapshot source: {path}")
            try:
                before = path.stat(follow_symlinks=False)
                digest = hashlib.sha256()
                size = 0
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                after = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise PoolError(f"could not hash snapshot artifact: {path}") from exc
            stable_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            if (
                stable_identity
                != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                or size != before.st_size
            ):
                raise PoolError(f"snapshot artifact changed while hashing: {path}")
            relative = path.relative_to(root).as_posix()
            files.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": digest.hexdigest(),
                }
            )
            total_bytes += size
    directories.sort()
    files.sort(key=lambda item: str(item["path"]))
    content = {"directories": directories, "files": files}
    return {
        **content,
        "directory_count": len(directories),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": _digest(content),
    }


def _snapshot_incomplete_harbor_job(
    source: Path,
    destination: Path,
    *,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically retain a byte-complete incomplete Harbor job before resume."""

    source = Path(source)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise PoolError(f"Harbor resume snapshot already exists: {destination}")
    _ensure_dir(destination.parent)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        shutil.copytree(source, staging, symlinks=True)
        _assert_regular_snapshot_tree(staging)
        for directory, _, filenames in os.walk(staging, followlinks=False):
            try:
                Path(directory).chmod(PRIVATE_DIR_MODE)
            except OSError:
                pass
            for name in filenames:
                try:
                    (Path(directory) / name).chmod(PRIVATE_FILE_MODE)
                except OSError:
                    pass
        source_after = _regular_tree_manifest(source)
        copied = _regular_tree_manifest(staging)
        if dict(expected_manifest) != source_after or source_after != copied:
            raise PoolError("incomplete Harbor job changed while it was snapshotted")
        os.replace(staging, destination)
        _fsync_directory(destination.parent)
        return copied
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _stored_single_job_contract(
    spec: PoolSpec,
    claim: ClaimedCell,
    stored: Mapping[str, Any],
) -> tuple[str, str]:
    """Recover the serving URL/backend only if the stored contract is exact."""

    agents = stored.get("agents")
    agent = (
        agents[0]
        if isinstance(agents, list)
        and len(agents) == 1
        and isinstance(agents[0], Mapping)
        else None
    )
    kwargs = agent.get("kwargs") if isinstance(agent, Mapping) else None
    sglang_url = kwargs.get("sglang_url") if isinstance(kwargs, Mapping) else None
    if not isinstance(sglang_url, str) or not sglang_url.strip():
        raise PoolError("stored cell config has no SGLang URL")
    matching_backends = [
        backend
        for backend in ("frozen", "docker", "docker-gpu")
        if _single_job_config(
            spec,
            claim,
            sglang_url=sglang_url,
            environment_backend=backend,
        )
        == dict(stored)
    ]
    if len(matching_backends) != 1:
        raise PoolError("stored cell config is not one exact pool-derived contract")
    return sglang_url, matching_backends[0]


def _unfinished_harbor_job_evidence(
    claim: ClaimedCell,
) -> dict[str, Any]:
    """Require one incomplete single-cell trial and no terminal Harbor trial."""

    job = claim.paths.harbor_job
    if job.is_symlink() or not job.is_dir():
        raise PoolError("Harbor resume incident has no safe job directory")
    if _harbor_trial_results(claim):
        raise PoolError("Harbor resume incident already has a terminal trial")
    incomplete_trials = [
        child
        for child in sorted(job.iterdir(), key=lambda value: value.name)
        if child.is_dir()
        and not child.is_symlink()
        and (child / "config.json").is_file()
        and not (child / "result.json").exists()
    ]
    if len(incomplete_trials) != 1:
        raise PoolError(
            "Harbor resume incident must contain exactly one incomplete trial"
        )
    trial = incomplete_trials[0]
    if (trial / "config.json").is_symlink():
        raise PoolError("incomplete Harbor trial config is unsafe")
    _read_json(trial / "config.json")
    summary_path = job / "result.json"
    if summary_path.exists() or summary_path.is_symlink():
        summary = _read_json(summary_path, max_bytes=2 * 1024 * 1024)
        stats = summary.get("stats")
        terminal_counts = (
            "n_completed_trials",
            "n_errored_trials",
            "n_cancelled_trials",
        )
        if (
            summary.get("finished_at") is not None
            or summary.get("n_total_trials") != 1
            or not isinstance(stats, Mapping)
            or any(stats.get(name) != 0 for name in terminal_counts)
        ):
            raise PoolError("Harbor job summary already records a terminal trial")
    manifest = _regular_tree_manifest(job)
    return {
        "incomplete_trial_name": trial.name,
        "manifest": manifest,
    }


def _harbor_resume_config_candidate(
    spec: PoolSpec,
    cell: CellSpec,
    descriptor: int,
) -> dict[str, Any]:
    """Strictly classify one committed raw-vs-normalized config incident."""

    paths = spec.cell_paths(cell).ensure()
    if not _cell_is_score_eligible(
        cell, _attempt_selection(spec), _task_selection(spec)
    ):
        raise PoolError(f"requested cell is excluded from scoring: {cell.cell_id}")
    if paths.external_reservation.exists() or paths.external_reservation.is_symlink():
        raise PoolError(f"requested cell has an external reservation: {cell.cell_id}")
    if (
        paths.harbor_resume_config_repair.exists()
        or paths.harbor_resume_config_repair.is_symlink()
    ):
        raise PoolError(f"Harbor resume config repair already exists: {cell.cell_id}")

    artifact_paths = {
        "result.json": paths.result,
        "state.json": paths.state,
        "dispatch.json": paths.dispatch,
        "exception.json": paths.exception,
    }
    artifact_payloads = {
        name: _read_regular_bytes(path, max_bytes=MAX_ARTIFACT_BYTES)
        for name, path in artifact_paths.items()
    }
    result = _cell_result(spec, cell)
    if result is None:
        raise PoolError(f"requested cell has no terminal result: {cell.cell_id}")
    state = _read_json(paths.state)
    dispatch = _read_json(paths.dispatch)
    exception = _read_json(paths.exception)
    evidence = _harbor_resume_config_exception_evidence(cell, exception)
    execution = _valid_execution(result.get("execution_attempt"))
    identity = {
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
    }
    if (
        evidence is None
        or execution < 1
        or evidence["execution_attempt"] != execution
        or evidence["worker_exception_count"] != 1
        or result.get("status") != "pool_error"
        or result.get("outcome") != "unclassified_worker_exception"
        or result.get("exception_type") != "PoolError"
        or result.get("worker_exception_count") != 1
        or result.get("rewards") != {}
        or result.get("n_input_tokens") != 0
        or result.get("n_output_tokens") != 0
        or _parse_time(result.get("finished_at")) is None
        or _parse_time(result.get("committed_at")) is None
        or any(result.get(key) != value for key, value in identity.items())
    ):
        raise PoolError(
            f"committed Harbor resume config result is inconsistent: {cell.cell_id}"
        )
    common_metadata = (state, dispatch)
    if any(
        record.get("execution_attempt") != execution
        or record.get("worker_exception_count") != 1
        or any(record.get(key) != value for key, value in identity.items())
        for record in common_metadata
    ) or (
        state.get("schema") != CELL_STATE_SCHEMA
        or state.get("status") != "pool_error"
        or state.get("outcome") != "unclassified_worker_exception"
        or state.get("exception_type") != "PoolError"
        or dispatch.get("schema") != CELL_DISPATCH_SCHEMA
        or dispatch.get("status") != "terminal"
        or dispatch.get("outcome") != "unclassified_worker_exception"
    ):
        raise PoolError(
            f"committed Harbor resume config metadata is inconsistent: {cell.cell_id}"
        )

    claim = ClaimedCell(
        cell=cell,
        paths=paths,
        descriptor=descriptor,
        execution_attempt=execution,
        dispatch=dispatch,
    )
    stored_job_config = _read_json(paths.job_config)
    sglang_url, environment_backend = _stored_single_job_contract(
        spec, claim, stored_job_config
    )
    observed_path = paths.harbor_job / "config.json"
    observed_job_config = _read_json(observed_path)
    if observed_job_config == stored_job_config or not (
        _harbor_resume_config_matches_contract(observed_job_config, stored_job_config)
    ):
        raise PoolError(
            f"observed Harbor config is not the normalized resume form: {cell.cell_id}"
        )
    harbor = _unfinished_harbor_job_evidence(claim)
    attempt_dir = paths.attempts_dir / f"execution-{execution:04d}"
    terminal_archive = attempt_dir / HARBOR_RESUME_CONFIG_TERMINAL_ARCHIVE
    harbor_snapshot = attempt_dir / HARBOR_RESUME_CONFIG_JOB_SNAPSHOT
    for archive in (terminal_archive, harbor_snapshot):
        if archive.exists() or archive.is_symlink():
            raise PoolError(f"Harbor resume config archive already exists: {archive}")
    return {
        "claim": claim,
        "result": result,
        "state": state,
        "dispatch": dispatch,
        "exception": exception,
        "evidence": evidence,
        "artifact_payloads": artifact_payloads,
        "artifact_sha256": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in artifact_payloads.items()
        },
        "stored_job_config": stored_job_config,
        "observed_job_config": observed_job_config,
        "sglang_url": sglang_url,
        "environment_backend": environment_backend,
        "incomplete_trial_name": harbor["incomplete_trial_name"],
        "harbor_manifest": harbor["manifest"],
        "attempt_dir": attempt_dir,
        "terminal_archive": terminal_archive,
        "harbor_snapshot": harbor_snapshot,
    }


def requeue_committed_harbor_resume_config_incident(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Archive and requeue explicit false Harbor resume-config pool errors.

    The incomplete Harbor job remains live for native resume, but a complete
    immutable snapshot is published first because Harbor removes an unfinished
    trial directory before rebuilding it.  Rewriting only the live config to
    the exact pool-authored encoding also lets already-running legacy workers
    pass their former raw JSON equality gate.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested_order = list(cell_ids)
    if not requested_order:
        raise PoolError("repair-harbor-resume-config requires at least one cell id")
    if len(set(requested_order)) != len(requested_order):
        raise PoolError("repair-harbor-resume-config received a duplicate cell id")
    cells_by_id = {cell.cell_id: cell for cell in spec.cells}
    unknown = set(requested_order) - set(cells_by_id)
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")

    locked_descriptors: list[int] = []
    candidates: list[dict[str, Any]] = []
    try:
        # Lock and validate the entire explicit batch before the first write.
        for cell_id in requested_order:
            cell = cells_by_id[cell_id]
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                raise PoolError(f"requested cell is currently owned: {cell_id}")
            locked_descriptors.append(descriptor)
            candidates.append(_harbor_resume_config_candidate(spec, cell, descriptor))

        repaired: list[str] = []
        for candidate in candidates:
            claim = candidate["claim"]
            cell = claim.cell
            paths = claim.paths
            execution = claim.execution_attempt
            _ensure_dir(candidate["attempt_dir"])
            snapshot_manifest = _snapshot_incomplete_harbor_job(
                paths.harbor_job,
                candidate["harbor_snapshot"],
                expected_manifest=candidate["harbor_manifest"],
            )
            terminal_archive = _ensure_dir(candidate["terminal_archive"])
            for name, payload in candidate["artifact_payloads"].items():
                _atomic_write_bytes(terminal_archive / name, payload)
            _atomic_write_json(
                terminal_archive / "harbor-job-snapshot-manifest.json",
                snapshot_manifest,
            )

            # The snapshot retains Harbor's original normalized encoding.
            # Normalize the live resume boundary back to the pool-authored raw
            # contract so both pre-fix and post-fix workers can resume it.
            _atomic_write_json(
                paths.harbor_job / "config.json",
                candidate["stored_job_config"],
            )
            if (
                _read_json(paths.harbor_job / "config.json")
                != candidate["stored_job_config"]
            ):
                raise PoolError("live Harbor resume config rewrite did not persist")

            now = _utc_now()
            terminal_relative = str(terminal_archive.relative_to(paths.root))
            snapshot_relative = str(
                candidate["harbor_snapshot"].relative_to(paths.root)
            )
            receipt = {
                "schema": "terminal_bench_harbor_resume_config_repair_v1",
                "incident": HARBOR_RESUME_CONFIG_INCIDENT,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "execution_attempt": execution,
                "worker_exception_count": 1,
                "exception_sha256": candidate["evidence"]["exception_sha256"],
                "archived_artifact_sha256": candidate["artifact_sha256"],
                "observed_harbor_config_sha256": _digest(
                    candidate["observed_job_config"]
                ),
                "restored_cell_config_sha256": _digest(candidate["stored_job_config"]),
                "incomplete_trial_name": candidate["incomplete_trial_name"],
                "harbor_snapshot_manifest": snapshot_manifest,
                "harbor_snapshot_path": snapshot_relative,
                "terminal_archive_path": terminal_relative,
                "sglang_url": candidate["sglang_url"],
                "environment_backend": candidate["environment_backend"],
                "official_attempt_consumed_by_repair": False,
                "requeue_policy": HARBOR_RESUME_CONFIG_INCIDENT,
                "requeued_at": now,
            }
            _atomic_write_json(terminal_archive / "requeue.json", receipt)
            _atomic_write_json(paths.harbor_resume_config_repair, receipt)

            queued_dispatch = dict(candidate["dispatch"])
            queued_dispatch.update(
                {
                    "status": "queued",
                    "worker_exception_count": 0,
                    "last_harbor_resume_config_incident": (
                        HARBOR_RESUME_CONFIG_INCIDENT
                    ),
                    "last_harbor_resume_config_execution_attempt": execution,
                    "last_harbor_resume_config_snapshot": snapshot_relative,
                    "updated_at": now,
                }
            )
            for key in (
                "claim",
                "outcome",
                "harbor_job_path",
                "retry_not_before",
            ):
                queued_dispatch.pop(key, None)
            queued_state = {
                "schema": CELL_STATE_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "execution_attempt": execution,
                "worker_exception_count": 0,
                "last_harbor_resume_config_incident": (HARBOR_RESUME_CONFIG_INCIDENT),
                "last_harbor_resume_config_execution_attempt": execution,
                "last_harbor_resume_config_snapshot": snapshot_relative,
                "updated_at": now,
            }
            _atomic_write_json(paths.dispatch, queued_dispatch)
            _atomic_write_json(paths.state, queued_state)

            # Keep result.json as the terminal authority until every recovery
            # artifact and queued projection is durable.  The live exception
            # must then be removed too: workers started before this fix do not
            # understand the additive clearance receipt.
            if (
                paths.exception.read_bytes()
                != candidate["artifact_payloads"]["exception.json"]
            ):
                raise PoolError("worker exception changed during config repair")
            paths.exception.unlink()
            _fsync_directory(paths.root)
            if (
                paths.result.read_bytes()
                != candidate["artifact_payloads"]["result.json"]
            ):
                raise PoolError("terminal result changed during config repair")
            paths.result.unlink()
            _fsync_directory(paths.root)
            repaired.append(cell.cell_id)
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_harbor_resume_config_incident_requeued",
                    "incident": HARBOR_RESUME_CONFIG_INCIDENT,
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "harbor_snapshot_path": snapshot_relative,
                    "terminal_archive_path": terminal_relative,
                    "official_attempt_consumed_by_repair": False,
                    "recorded_at": now,
                },
            )
    finally:
        for descriptor in reversed(locked_descriptors):
            _close_lock(descriptor)
    return {
        "incident": HARBOR_RESUME_CONFIG_INCIDENT,
        "repaired_cell_ids": repaired,
        **aggregate_pool(spec),
    }


def requeue_committed_agent_timeout(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Archive and requeue committed cells from Harbor's retired time budget.

    This is deliberately not a generic timeout retry.  The pool result and the
    single pinned Harbor trial must independently attest Harbor's typed
    ``AgentTimeoutError`` before any terminal marker is changed.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    from terminal_bench_eval.agent_timeout_policy import load_policy
    if load_policy(spec) is not None:
        raise PoolError("AgentTimeoutError is terminal with reward=0; this pool forbids agent timeout retries")
    requested = set(cell_ids)
    known = {cell.cell_id for cell in spec.cells}
    unknown = requested - known
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    repaired: list[str] = []
    skipped: list[str] = []
    for cell in spec.cells:
        if requested and cell.cell_id not in requested:
            continue
        paths = spec.cell_paths(cell).ensure()
        descriptor = _open_lock(paths.claim_lock, blocking=False)
        if descriptor is None:
            skipped.append(cell.cell_id)
            continue
        try:
            result = _cell_result(spec, cell)
            if result is None:
                skipped.append(cell.cell_id)
                continue
            failure = _agent_timeout_trial_failure(result)
            if failure is None:
                skipped.append(cell.cell_id)
                continue
            execution = _valid_execution(result.get("execution_attempt"))
            if execution < 1:
                raise PoolError(
                    f"committed agent timeout has no execution: {cell.cell_id}"
                )
            state = _read_json(paths.state)
            dispatch = _read_json(paths.dispatch)
            if (
                state.get("status") != "completed"
                or state.get("execution_attempt") != execution
                or dispatch.get("status") != "terminal"
                or dispatch.get("execution_attempt") != execution
            ):
                raise PoolError(
                    f"committed agent-timeout metadata is inconsistent: {cell.cell_id}"
                )
            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    f"committed agent timeout has no Harbor trial: {cell.cell_id}"
                )
            trial_path, trial = terminal
            harbor_failure = _agent_timeout_trial_failure(trial)
            if harbor_failure != failure:
                raise PoolError(
                    f"pool and Harbor agent timeouts disagree: {cell.cell_id}"
                )

            attempt_dir = _ensure_dir(paths.attempts_dir / f"execution-{execution:04d}")
            terminal_archive = attempt_dir / "committed-terminal-agent-timeout"
            if terminal_archive.exists() or terminal_archive.is_symlink():
                raise PoolError(
                    f"committed agent-timeout archive already exists: {terminal_archive}"
                )
            _ensure_dir(terminal_archive)
            _atomic_write_json(terminal_archive / "state.json", state)
            _atomic_write_json(terminal_archive / "dispatch.json", dispatch)
            harbor_archive = _archive_agent_timeout_harbor_job(
                claim,
                trial_path=trial_path,
                failure=failure,
            )

            now = _utc_now()
            queued_dispatch = dict(dispatch)
            queued_dispatch.update(
                {
                    "status": "queued",
                    "last_agent_timeout_execution_attempt": execution,
                    "last_agent_timeout_failure": dict(failure),
                    "last_agent_timeout_archive": str(
                        harbor_archive.relative_to(spec.paths.root)
                    ),
                    "updated_at": now,
                }
            )
            for key in (
                "claim",
                "outcome",
                "harbor_job_path",
                "retry_not_before",
            ):
                queued_dispatch.pop(key, None)
            queued_state = {
                "schema": CELL_STATE_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "execution_attempt": execution,
                "worker_exception_count": int(
                    dispatch.get("worker_exception_count") or 0
                ),
                "last_agent_timeout_execution_attempt": execution,
                "last_agent_timeout_failure": dict(failure),
                "last_agent_timeout_archive": str(
                    harbor_archive.relative_to(spec.paths.root)
                ),
                "updated_at": now,
            }
            _atomic_write_json(paths.dispatch, queued_dispatch)
            _atomic_write_json(paths.state, queued_state)
            # result.json is the terminal commit marker.  Move it only after
            # the fresh queued state is durable, while the cell flock is held.
            os.replace(paths.result, terminal_archive / "result.json")
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)
            _atomic_write_json(
                terminal_archive / "requeue.json",
                {
                    "schema": "terminal_bench_committed_agent_timeout_requeue_v1",
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "failure": dict(failure),
                    "harbor_archive_path": str(
                        harbor_archive.relative_to(spec.paths.root)
                    ),
                    "requeue_policy": "retired_total_agent_timeout_v1",
                    "requeued_at": _utc_now(),
                },
            )
            repaired.append(cell.cell_id)
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_agent_timeout_requeued",
                    "cell_id": cell.cell_id,
                    "execution_attempt": execution,
                    "official_attempt_index": cell.official_attempt_index,
                    "recorded_at": _utc_now(),
                },
            )
        finally:
            _close_lock(descriptor)
    state = aggregate_pool(spec)
    return {"repaired_cell_ids": repaired, "skipped_cell_ids": skipped, **state}


def _verifier_replay_reward(value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) not in {0.0, 1.0}
    ):
        raise PoolError("verifier replay reward must be exactly 0 or 1")
    return float(value)


def _json_mapping_from_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PoolError(f"invalid JSON verifier replay {label}") from exc
    if not isinstance(value, Mapping):
        raise PoolError(f"verifier replay {label} must be a JSON object")
    return dict(value)


def _write_private_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    _ensure_dir(path.parent)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short verifier replay snapshot write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verifier_replay_source_path(manifest_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PoolError("verifier replay evidence has an invalid source_path")
    raw = Path(value).expanduser()
    candidate = raw if raw.is_absolute() else manifest_path.parent / raw
    if candidate.is_symlink():
        raise PoolError(f"verifier replay evidence must not be a symlink: {candidate}")
    try:
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise PoolError(
            f"verifier replay evidence does not exist: {candidate}"
        ) from exc


def _validate_pinned_candidate_timeout_zero_replay(
    cell: CellSpec,
    entry: Mapping[str, Any],
    evidence_payloads: Mapping[str, bytes],
) -> None:
    """Accept one frozen model-free replay whose candidate exhausted the verifier.

    This is deliberately an exact-evidence policy, not a reusable timeout
    classifier.  In particular, the timeout snapshot's surviving ``uv`` name is
    ambiguous in isolation.  It becomes candidate attribution only together
    with the frozen official test, completed pytest bootstrap output, exact
    source-session extraction, and clean environment receipts below.
    """

    contract = _PINNED_CANDIDATE_TIMEOUT_ZERO_CELLS.get(cell.cell_id)
    identity = {
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "source": cell.source,
        "official_attempt_index": cell.official_attempt_index,
        "execution_attempt": entry.get("execution_attempt"),
        "config_contract_hash": entry.get("config_contract_hash"),
    }
    if (
        not isinstance(contract, Mapping)
        or any(contract.get(key) != value for key, value in identity.items())
        or entry.get("kind") != PINNED_CANDIDATE_TIMEOUT_ZERO_KIND
        or entry.get("timeout_zero_policy")
        != PINNED_CANDIDATE_TIMEOUT_ZERO_POLICY
        or _verifier_replay_reward(entry.get("reward")) != 0.0
        or entry.get("model_called") is not False
        or entry.get("pytest_exit_code") is not None
        or entry.get("runner_receipt_evidence")
        != "replay-agent-receipt-json"
        or entry.get("harbor_result_evidence") != "replay-result-json"
    ):
        raise PoolError("pinned candidate-timeout-zero identity mismatch")

    manifest_evidence = entry.get("evidence")
    if not isinstance(manifest_evidence, Mapping):
        raise PoolError("pinned candidate-timeout-zero evidence is missing")
    for category, expected_names in (
        _PINNED_CANDIDATE_TIMEOUT_ZERO_EVIDENCE_CATEGORIES.items()
    ):
        items = manifest_evidence.get(category)
        observed_names = (
            {item.get("name") for item in items if isinstance(item, Mapping)}
            if isinstance(items, list)
            else set()
        )
        if observed_names != expected_names or len(items) != len(expected_names):
            raise PoolError(
                "pinned candidate-timeout-zero evidence categories disagree"
            )
    if set(evidence_payloads) != _PINNED_CANDIDATE_TIMEOUT_ZERO_EVIDENCE_KEYS:
        raise PoolError("pinned candidate-timeout-zero evidence set disagrees")
    expected_hashes = contract.get("evidence_sha256")
    observed_hashes = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in evidence_payloads.items()
    }
    if (
        not isinstance(expected_hashes, Mapping)
        or set(expected_hashes) != _PINNED_CANDIDATE_TIMEOUT_ZERO_EVIDENCE_KEYS
        or observed_hashes != dict(expected_hashes)
    ):
        raise PoolError("pinned candidate-timeout-zero evidence digest mismatch")

    def json_evidence(name: str) -> dict[str, Any]:
        return _json_mapping_from_bytes(evidence_payloads[name], label=name)

    candidate = evidence_payloads["candidate-steal-py"]
    extraction = json_evidence("source-extraction-manifest-json")
    source_config = json_evidence("source-job-config-json")
    runner = json_evidence("replay-agent-receipt-json")
    replay_config = json_evidence("replay-job-config-json")
    replay_trial_config = json_evidence("replay-trial-config-json")
    replay_lock = json_evidence("replay-lock-json")
    replay_result_raw = json_evidence("replay-result-json")
    timeout_receipt = json_evidence("replay-timeout-receipt-json")
    image_pull = json_evidence("replay-proot-image-pull-json")
    guest_utilities = json_evidence("replay-proot-guest-utilities-json")

    # Prove that the replayed /app/steal.py is the exact final heredoc emitted
    # by the original model session, rather than a repaired human artifact.
    try:
        session_rows = evidence_payloads["source-model-session-jsonl"].splitlines()
        ordinal = int(contract["source_session_zero_based_ordinal"])
        session_row = json.loads(session_rows[ordinal])
        session_item = session_row["payload"]
        command = json.loads(session_item["arguments"])["cmd"]
        command_bytes = command.encode("utf-8")
        opener = "cat > steal.py <<'PYEOF'\n"
        terminator = "\nPYEOF"
        if command.count(opener) != 1 or command.count(terminator) != 1:
            raise ValueError("unexpected heredoc shape")
        start = command.index(opener) + len(opener)
        end = command.index(terminator, start)
        extracted_candidate = (command[start:end] + "\n").encode("utf-8")
    except (IndexError, KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise PoolError("pinned source-session extraction is invalid") from exc
    if (
        session_row.get("type") != "response_item"
        or session_item.get("type") != "function_call"
        or session_item.get("name") != "exec_command"
        or hashlib.sha256(command_bytes).hexdigest()
        != contract.get("source_command_sha256")
        or extracted_candidate != candidate
        or hashlib.sha256(candidate).hexdigest() != contract.get("candidate_sha256")
        or extraction.get("schema")
        != "terminal_bench_exact_verifier_replay_artifact_v1"
        or extraction.get("job_name") != contract.get("job_name")
        or extraction.get("cell_id") != cell.cell_id
        or extraction.get("task_name") != cell.task_name
        or extraction.get("official_attempt_index") != cell.official_attempt_index
        or extraction.get("execution_attempt") != entry.get("execution_attempt")
        or extraction.get("session_zero_based_ordinal") != ordinal
        or extraction.get("command_sha256") != contract.get("source_command_sha256")
        or extraction.get("deliverable_sha256") != contract.get("candidate_sha256")
        or extraction.get("model_called") is not False
        or extraction.get("source_config_sha256")
        != observed_hashes["source-job-config-json"]
        or extraction.get("session_sha256")
        != observed_hashes["source-model-session-jsonl"]
        or source_config.get("job_name") != "harbor-job"
        or not isinstance(source_config.get("tasks"), list)
        or len(source_config["tasks"]) != 1
        or source_config["tasks"][0].get("name") != cell.task_name
        or source_config["tasks"][0].get("ref") != cell.task_ref
        or source_config["tasks"][0].get("source") != cell.source
    ):
        raise PoolError("pinned source candidate provenance disagrees")

    try:
        task_metadata = tomllib.loads(
            evidence_payloads["official-task-toml"].decode("utf-8")
        )
        test_shell = evidence_payloads["official-test-sh"].decode("utf-8")
        test_python = evidence_payloads["official-test-outputs-py"].decode("utf-8")
        partial_stdout = evidence_payloads["replay-timeout-stdout"].decode("utf-8")
        replay_log = evidence_payloads["replay-trial-log"].decode("utf-8")
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise PoolError("pinned official-test evidence is invalid") from exc
    verifier_metadata = task_metadata.get("verifier")
    task_metadata_identity = task_metadata.get("task")
    timeout_seconds = float(contract["configured_verifier_timeout_seconds"])
    if (
        not isinstance(verifier_metadata, Mapping)
        or float(verifier_metadata.get("timeout_sec", -1)) != timeout_seconds
        or not isinstance(task_metadata_identity, Mapping)
        or task_metadata_identity.get("name") != cell.task_name
        or test_shell.count("/tests/test_outputs.py") != 1
        or test_shell.count("uvx \\\n") != 1
        or test_python.count('os.popen("python3 /app/steal.py").read()') != 1
        or test_python.count('Path("/app/stolen_A1.npy")') != 1
        or partial_stdout.count("collected 1 item") != 1
        or partial_stdout.count("../tests/test_outputs.py ") != 1
        or not partial_stdout.rstrip().endswith("../tests/test_outputs.py")
        or "Installed 7 packages in 306ms" not in partial_stdout
        or any(
            marker in partial_stdout.split("collected 1 item", 1)[1]
            for marker in ("Downloading ", "Get:", "Installing ")
        )
        or replay_log.count(
            "Verifier execution timed out after 900.0 seconds"
        )
        != 1
    ):
        raise PoolError("pinned official test did not enter the candidate cleanly")

    if (
        runner.get("schema") != "terminal_bench_verifier_replay_agent_v1"
        or runner.get("model_called") is not False
        or runner.get("sha256") != contract.get("candidate_sha256")
        or runner.get("target_path") != "/app/steal.py"
        or replay_config.get("job_name")
        != "verifier-replay-model-extraction-tb_7c009"
        or replay_config.get("n_attempts") != 1
        or replay_config.get("n_concurrent_trials") != 1
        or replay_config.get("timeout_multiplier") != 1.0
        or not isinstance(replay_config.get("retry"), Mapping)
        or replay_config["retry"].get("max_retries") != 0
        or not isinstance(replay_config.get("agents"), list)
        or len(replay_config["agents"]) != 1
        or replay_config["agents"][0].get("import_path")
        != "terminal_bench_eval.verifier_replay_agent:VerifierReplayAgent"
        or replay_trial_config.get("task", {}).get("ref") != cell.task_ref
        or replay_trial_config.get("task", {}).get("source") != cell.source
        or replay_trial_config.get("agent", {}).get("import_path")
        != "terminal_bench_eval.verifier_replay_agent:VerifierReplayAgent"
        or replay_lock.get("schema_version") != 2
        or replay_lock.get("task", {}).get("digest") != cell.task_ref
        or replay_lock.get("task", {}).get("source") != cell.source
        or replay_lock.get("timeout_multiplier") != 1.0
    ):
        raise PoolError("pinned model-free replay configuration is invalid")

    try:
        replay_result = TrialResult.model_validate(replay_result_raw).model_dump(
            mode="json", exclude_none=False
        )
    except Exception as exc:
        raise PoolError("invalid pinned timeout replay TrialResult") from exc
    replay_exception = replay_result.get("exception_info")
    replay_agent = replay_result.get("config", {}).get("agent", {})
    replay_task = replay_result.get("config", {}).get("task", {})
    replay_verifier = replay_result.get("config", {}).get("verifier", {})
    verifier_timing = replay_result.get("verifier")
    verifier_started = (
        _parse_time(verifier_timing.get("started_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    verifier_finished = (
        _parse_time(verifier_timing.get("finished_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    if (
        replay_result.get("trial_name") != contract.get("replay_trial_name")
        or replay_result.get("id") != contract.get("replay_trial_id")
        or replay_result.get("task_name") != cell.task_name
        or replay_result.get("source") != cell.source
        or replay_result.get("task_checksum") != contract.get("task_checksum")
        or replay_result.get("verifier_result") is not None
        or _result_rewards(replay_result)
        or not isinstance(replay_exception, Mapping)
        or replay_exception.get("exception_type") != "VerifierTimeoutError"
        or replay_exception.get("exception_message")
        != f"Verifier execution timed out after {timeout_seconds:.1f} seconds"
        or replay_agent.get("import_path")
        != "terminal_bench_eval.verifier_replay_agent:VerifierReplayAgent"
        or replay_agent.get("kwargs", {}).get("expected_sha256")
        != contract.get("candidate_sha256")
        or replay_task.get("ref") != cell.task_ref
        or replay_task.get("source") != cell.source
        or replay_verifier.get("disable") is not False
        or replay_verifier.get("override_timeout_sec") is not None
        or replay_verifier.get("max_timeout_sec") is not None
        or verifier_started is None
        or verifier_finished is None
        or not (
            timeout_seconds * 0.99
            <= (verifier_finished - verifier_started).total_seconds()
            <= timeout_seconds + max(5.0, timeout_seconds * 0.01)
        )
    ):
        raise PoolError("pinned Harbor timeout replay result is invalid")

    records = timeout_receipt.get("records")
    timeout_record = (
        records[0]
        if isinstance(records, list)
        and len(records) == 1
        and isinstance(records[0], Mapping)
        else None
    )
    partial = (
        timeout_record.get("partial_stdout")
        if isinstance(timeout_record, Mapping)
        else None
    )
    timeout_recorded = (
        _parse_time(timeout_record.get("recorded_at"))
        if isinstance(timeout_record, Mapping)
        else None
    )
    if (
        timeout_receipt.get("schema")
        != COMPOSITE_VERIFIER_TIMEOUT_RECEIPT_SCHEMA
        or timeout_receipt.get("policy")
        != COMPOSITE_VERIFIER_TIMEOUT_RECEIPT_POLICY
        or timeout_receipt.get("session_id")
        != f"{contract['replay_trial_name']}__env"
        or not isinstance(timeout_record, Mapping)
        or timeout_record.get("timeout_attempt") != 1
        or timeout_record.get("script_sha256")
        != observed_hashes["official-test-sh"]
        or timeout_record.get("process_snapshot")
        != contract.get("process_snapshot")
        or timeout_record.get("retry_eligible") is not False
        or timeout_record.get("retry_state") != "not_eligible"
        or not isinstance(partial, Mapping)
        or partial.get("path") != "test-stdout.timeout-1.partial.txt"
        or partial.get("bytes_copied")
        != len(evidence_payloads["replay-timeout-stdout"])
        or partial.get("source_bytes_at_snapshot")
        != len(evidence_payloads["replay-timeout-stdout"])
        or partial.get("tail_truncated") is not False
        or timeout_recorded is None
        or verifier_started is None
        or not (
            timeout_seconds * 0.99
            <= (timeout_recorded - verifier_started).total_seconds()
            <= timeout_seconds + max(5.0, timeout_seconds * 0.01)
        )
    ):
        raise PoolError("pinned candidate timeout process evidence is invalid")

    # Validate clean teardown using the same strict receipt parser as the
    # original composite failure gate.
    proot = _composite_timeout_clean_proot_evidence(
        {
            "trial/proot-runtime.json": evidence_payloads[
                "replay-proot-runtime-json"
            ],
            "trial/proot-package-bridge.json": evidence_payloads[
                "replay-proot-package-bridge-json"
            ],
        },
        trial_name=str(contract["replay_trial_name"]),
    )
    image_attempts = image_pull.get("attempts")
    utility_attempts = guest_utilities.get("install_attempts")
    if (
        proot is None
        or image_pull.get("schema") != PROOT_IMAGE_PULL_RECEIPT_SCHEMA
        or image_pull.get("policy") != PROOT_IMAGE_PULL_POLICY
        or image_pull.get("failure_policy") != "closed"
        or image_pull.get("session_id")
        != f"{contract['replay_trial_name']}__env"
        or image_pull.get("status") != "ok"
        or image_pull.get("stage") != "complete"
        or image_pull.get("successful_attempt") != 1
        or not isinstance(image_attempts, list)
        or len(image_attempts) != 1
        or image_attempts[0].get("status") != "succeeded"
        or guest_utilities.get("schema")
        != PROOT_GUEST_UTILITY_RECEIPT_SCHEMA
        or guest_utilities.get("policy") != PROOT_GUEST_UTILITY_POLICY
        or guest_utilities.get("failure_policy") != "closed"
        or guest_utilities.get("session_id")
        != f"{contract['replay_trial_name']}__env"
        or guest_utilities.get("status") != "ok"
        or guest_utilities.get("stage") != "complete"
        or not isinstance(utility_attempts, list)
        or not utility_attempts
        or any(
            attempt.get("status") != "succeeded"
            or attempt.get("result", {}).get("return_code") != 0
            for attempt in utility_attempts
            if isinstance(attempt, Mapping)
        )
        or any(not isinstance(attempt, Mapping) for attempt in utility_attempts)
    ):
        raise PoolError("pinned candidate timeout has unclean infrastructure")


def _validate_verifier_replay_execution_evidence(
    cell: CellSpec,
    entry: Mapping[str, Any],
    evidence_payloads: Mapping[str, bytes],
) -> None:
    reward = _verifier_replay_reward(entry.get("reward"))
    if entry.get("model_called") is not False:
        raise PoolError("verifier replay manifest must attest model_called=false")
    runner_name = entry.get("runner_receipt_evidence")
    if not isinstance(runner_name, str) or runner_name not in evidence_payloads:
        raise PoolError("verifier replay runner receipt evidence is missing")
    runner = _json_mapping_from_bytes(
        evidence_payloads[runner_name], label="runner receipt"
    )
    if runner.get("model_called") is not False:
        raise PoolError("verifier replay runner receipt did not attest no model call")

    kind = entry.get("kind")
    if (entry.get("original_failure_policy") in VERIFIER_BOOTSTRAP_REPLAY_POLICIES
            and kind != "harbor_replay"):
        raise PoolError("bootstrap repair requires a model-free Harbor replay")
    if kind == "host_pytest":
        exit_code = entry.get("pytest_exit_code")
        if (
            not isinstance(exit_code, int)
            or isinstance(exit_code, bool)
            or exit_code not in {0, 1}
        ):
            raise PoolError("host pytest replay exit code must be 0 or 1")
        expected_reward = 1.0 if exit_code == 0 else 0.0
        if reward != expected_reward:
            raise PoolError("host pytest exit code and replay reward disagree")
        if (
            runner.get("pytest_exit_code") != exit_code
            or runner.get("timed_out") is not False
            or runner.get("signal") is not None
        ):
            raise PoolError("host pytest runner receipt is not a clean terminal run")
        if entry.get("harbor_result_evidence") is not None:
            raise PoolError("host pytest replay must not name a Harbor result")
        return

    if kind == PINNED_CANDIDATE_TIMEOUT_ZERO_KIND:
        _validate_pinned_candidate_timeout_zero_replay(
            cell, entry, evidence_payloads
        )
        return

    if kind != "harbor_replay":
        raise PoolError("unknown verifier replay kind")
    if entry.get("pytest_exit_code") is not None:
        raise PoolError("Harbor verifier replay must not set pytest_exit_code")
    result_name = entry.get("harbor_result_evidence")
    if not isinstance(result_name, str) or result_name not in evidence_payloads:
        raise PoolError("Harbor verifier replay result evidence is missing")
    result = _json_mapping_from_bytes(
        evidence_payloads[result_name], label="Harbor TrialResult"
    )
    try:
        trial = TrialResult.model_validate(result).model_dump(
            mode="json", exclude_none=True
        )
    except Exception as exc:
        raise PoolError("invalid Harbor verifier replay TrialResult") from exc
    if (
        trial.get("task_name") != cell.task_name
        or _parse_time(trial.get("finished_at")) is None
        or trial.get("exception_info") is not None
        or _result_rewards(trial) != {"reward": reward}
    ):
        raise PoolError("Harbor verifier replay result disagrees with the manifest")
    config = trial.get("config")
    task_config = config.get("task") if isinstance(config, Mapping) else None
    if isinstance(task_config, Mapping):
        replay_ref = task_config.get("ref")
        replay_source = task_config.get("source")
        if replay_ref is not None and replay_ref != cell.task_ref:
            raise PoolError("Harbor verifier replay task ref does not match the cell")
        if replay_source is not None and replay_source != cell.source:
            raise PoolError("Harbor verifier replay source does not match the cell")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    replay_agent = None
    if isinstance(agent, Mapping):
        replay_agent = agent.get("import_path") or agent.get("name")
    if replay_agent != (
        "terminal_bench_eval.verifier_replay_agent:VerifierReplayAgent"
    ):
        raise PoolError("Harbor replay did not use VerifierReplayAgent")
    kwargs = agent.get("kwargs") if isinstance(agent, Mapping) else None
    expected_sha256 = (
        kwargs.get("expected_sha256") if isinstance(kwargs, Mapping) else None
    )
    artifact_hashes = {
        str(item.get("sha256"))
        for item in entry.get("evidence", {}).get("artifact", [])
        if isinstance(item, Mapping)
    }
    if expected_sha256 not in artifact_hashes:
        raise PoolError("Harbor replay artifact digest is not in the manifest")
    if entry.get("original_failure_policy") in VERIFIER_BOOTSTRAP_REPLAY_POLICIES:
        # Bootstrap can emit a misleading zero reward even without running
        # pytest. Require a completed, internally consistent test report.
        report_name = entry.get("ctrf_report_evidence")
        report_names = {
            item.get("name") for item in entry.get("evidence", {}).get("report", [])
            if isinstance(item, Mapping)
        }
        if report_name not in report_names or report_name not in evidence_payloads:
            raise PoolError("bootstrap replay requires completed CTRF evidence")
        try:
            validate_completed_ctrf(evidence_payloads[report_name], reward)
        except ValueError as exc:
            raise PoolError(f"bootstrap replay tests did not complete: {exc}") from exc
        if (not isinstance(task_config, Mapping) or not isinstance(kwargs, Mapping)
                or runner.get("schema") != "terminal_bench_verifier_replay_agent_v1"
                or runner.get("sha256") != expected_sha256
                or runner.get("target_path") != kwargs.get("target_path")
                or task_config.get("ref") != cell.task_ref
                or task_config.get("source") != cell.source):
            raise PoolError("bootstrap replay runner or frozen task identity disagrees")
        artifact_pairs = {
            (item.get("target_path"), item.get("sha256"))
            for item in entry.get("evidence", {}).get("artifact", [])
            if isinstance(item, Mapping)
        }
        extra = kwargs.get("additional_files") or []
        if not isinstance(extra, list):
            raise PoolError("bootstrap replay additional files must be a list")
        expected_extra = [{"target_path": f.get("target_path"), "sha256": f.get("expected_sha256")}
                          for f in extra if isinstance(f, Mapping)]
        if (len(expected_extra) != len(extra)
                or (runner.get("additional_files") or []) != expected_extra
                or (kwargs.get("target_path"), expected_sha256) not in artifact_pairs
                or any((f["target_path"], f["sha256"]) not in artifact_pairs for f in expected_extra)):
            raise PoolError("bootstrap replay additional file evidence disagrees")
        script_hash = kwargs.get("restoration_script_sha256")
        if kwargs.get("restoration_script_path") or script_hash:
            script_name = entry.get("restoration_script_evidence")
            restoration = runner.get("restoration")
            if (not isinstance(script_name, str) or script_name not in evidence_payloads
                    or hashlib.sha256(evidence_payloads[script_name]).hexdigest() != script_hash
                    or not isinstance(restoration, Mapping)
                    or restoration.get("sha256") != script_hash
                    or restoration.get("return_code") != 0):
                raise PoolError("bootstrap restoration script lacks pinned successful evidence")
        collected = runner.get("collected_files") or []
        collect_paths = kwargs.get("collect_paths") or []
        if (not isinstance(collected, list) or not isinstance(collect_paths, list)
                or any(not isinstance(p, str) for p in collect_paths)):
            raise PoolError("bootstrap restored output metadata is invalid")
        if ({f.get("target_path") for f in collected if isinstance(f, Mapping)}
                != set(collect_paths)
                or any(not isinstance(f, Mapping) or (f.get("target_path"), f.get("sha256"))
                       not in artifact_pairs for f in collected)):
            raise PoolError("bootstrap restored output evidence disagrees")


def _load_verifier_replay_manifest(
    spec: PoolSpec, manifest_path: Path
) -> tuple[Path, bytes, list[dict[str, Any]]]:
    source = Path(manifest_path).expanduser()
    if source.is_symlink():
        raise PoolError("verifier replay manifest must not be a symlink")
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise PoolError("verifier replay manifest does not exist") from exc
    raw_manifest = _read_regular_bytes(source, max_bytes=MAX_ARTIFACT_BYTES)
    manifest = _json_mapping_from_bytes(raw_manifest, label="manifest")
    entries = manifest.get("entries")
    if (
        manifest.get("schema") != VERIFIER_REPLAY_MANIFEST_SCHEMA
        or manifest.get("job_name") != spec.paths.root.name
        or manifest.get("config_contract_hash") != spec.contract_hash
        or not isinstance(entries, list)
        or not entries
    ):
        raise PoolError("verifier replay manifest does not match the pool")

    cells = {cell.cell_id: cell for cell in spec.cells}
    prepared: list[dict[str, Any]] = []
    seen_cells: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise PoolError("verifier replay manifest entry must be an object")
        entry = dict(raw_entry)
        cell_id = entry.get("cell_id")
        if not isinstance(cell_id, str) or cell_id not in cells:
            raise PoolError(f"unknown verifier replay cell id: {cell_id!r}")
        if cell_id in seen_cells:
            raise PoolError(f"duplicate verifier replay cell id: {cell_id}")
        seen_cells.add(cell_id)
        cell = cells[cell_id]
        execution = entry.get("execution_attempt")
        if (
            entry.get("task_name") != cell.task_name
            or entry.get("task_ref") != cell.task_ref
            or entry.get("official_attempt_index") != cell.official_attempt_index
            or entry.get("config_contract_hash", spec.contract_hash)
            != spec.contract_hash
            or not isinstance(execution, int)
            or isinstance(execution, bool)
            or execution < 1
            or entry.get("kind") not in VERIFIER_REPLAY_KINDS
        ):
            raise PoolError(f"verifier replay identity mismatch for {cell_id}")
        _verifier_replay_reward(entry.get("reward"))

        evidence = entry.get("evidence")
        if not isinstance(evidence, Mapping):
            raise PoolError(f"verifier replay evidence is missing for {cell_id}")
        copied: list[dict[str, Any]] = []
        payloads: dict[str, bytes] = {}
        total_bytes = 0
        for category in VERIFIER_REPLAY_EVIDENCE_CATEGORIES:
            items = evidence.get(category)
            if not isinstance(items, list) or not items:
                raise PoolError(
                    f"verifier replay {category} evidence is missing for {cell_id}"
                )
            for item in items:
                if not isinstance(item, Mapping):
                    raise PoolError("verifier replay evidence entry must be an object")
                name = item.get("name")
                expected_sha256 = item.get("sha256")
                if (
                    not isinstance(name, str)
                    or _SAFE_NAME_RE.fullmatch(name) is None
                    or name in payloads
                    or not isinstance(expected_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
                ):
                    raise PoolError(
                        "verifier replay evidence name or digest is invalid"
                    )
                target_path = item.get("target_path")
                if category == "artifact":
                    target = Path(target_path) if isinstance(target_path, str) else None
                    if (
                        target is None
                        or not target.is_absolute()
                        or ".." in target.parts
                        or len(target.parts) < 3
                        or target.parts[1] != "app"
                    ):
                        raise PoolError(
                            "verifier replay artifact target must be under /app"
                        )
                elif target_path is not None and not isinstance(target_path, str):
                    raise PoolError("verifier replay target_path must be a string")
                resolved = _verifier_replay_source_path(source, item.get("source_path"))
                payload = _read_regular_bytes(resolved, max_bytes=MAX_ARTIFACT_BYTES)
                observed_sha256 = hashlib.sha256(payload).hexdigest()
                if observed_sha256 != expected_sha256:
                    raise PoolError(
                        f"verifier replay evidence digest mismatch: {resolved}"
                    )
                total_bytes += len(payload)
                if (
                    len(copied) >= MAX_VERIFIER_REPLAY_EVIDENCE_FILES
                    or total_bytes > MAX_VERIFIER_REPLAY_TOTAL_BYTES
                ):
                    raise PoolError("verifier replay evidence exceeds the snapshot cap")
                payloads[name] = payload
                copied.append(
                    {
                        "category": category,
                        "name": name,
                        "source_path": str(item.get("source_path")),
                        "resolved_source_path": str(resolved),
                        "sha256": observed_sha256,
                        "bytes": len(payload),
                        "target_path": target_path,
                        "payload": payload,
                    }
                )
        _validate_verifier_replay_execution_evidence(cell, entry, payloads)
        prepared.append(
            {
                "cell": cell,
                "entry": entry,
                "entry_sha256": _digest(entry),
                "evidence": copied,
            }
        )
    return source, raw_manifest, prepared


def _composite_verifier_timeout_contract(
    spec: PoolSpec,
    cell: CellSpec,
    *,
    execution: int,
) -> Mapping[str, Any] | None:
    """Return the immutable allowlist entry for the one audited composite failure."""

    contract = _COMPOSITE_VERIFIER_TIMEOUT_REPLAY_CELLS.get(cell.cell_id)
    if not isinstance(contract, Mapping):
        return None
    identity = {
        "job_name": spec.paths.root.name,
        "config_contract_hash": spec.contract_hash,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "source": cell.source,
        "official_attempt_index": cell.official_attempt_index,
        "execution_attempt": execution,
    }
    expected_hashes = contract.get("artifact_sha256")
    timeout_seconds = contract.get("configured_verifier_timeout_seconds")
    process_snapshot = contract.get("process_snapshot")
    if (
        any(contract.get(key) != value for key, value in identity.items())
        or not isinstance(contract.get("trial_name"), str)
        or not contract.get("trial_name")
        or not isinstance(contract.get("trial_id"), str)
        or not contract.get("trial_id")
        or not isinstance(contract.get("task_checksum"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(contract.get("task_checksum"))) is None
        or not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0
        or not isinstance(contract.get("verifier_script_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(contract.get("verifier_script_sha256"))
        )
        is None
        or not isinstance(process_snapshot, Mapping)
        or not isinstance(expected_hashes, Mapping)
        or set(expected_hashes) != _COMPOSITE_VERIFIER_TIMEOUT_REPLAY_ARTIFACT_KEYS
        or any(
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in expected_hashes.values()
        )
    ):
        return None
    return contract


def _composite_verifier_timeout_artifact_paths(
    paths: CellPaths,
    trial_path: Path,
    *,
    execution: int,
) -> dict[str, Path]:
    return {
        "result.json": paths.result,
        "state.json": paths.state,
        "dispatch.json": paths.dispatch,
        "run.json": paths.attempts_dir / f"execution-{execution:04d}" / "run.json",
        "trial-result.json": trial_path / "result.json",
        "trial/config.json": trial_path / "config.json",
        "trial/lock.json": trial_path / "lock.json",
        "trial/trial.log": trial_path / "trial.log",
        "trial/verifier/verifier-timeout-resilience.json": (
            trial_path / "verifier" / COMPOSITE_VERIFIER_TIMEOUT_RECEIPT_NAME
        ),
        "trial/verifier/test-stdout.timeout-1.partial.txt": (
            trial_path / "verifier" / "test-stdout.timeout-1.partial.txt"
        ),
        "trial/proot-runtime.json": trial_path / PROOT_RUNTIME_RECEIPT_NAME,
        "trial/proot-package-bridge.json": (
            trial_path / PROOT_PACKAGE_BRIDGE_RECEIPT_NAME
        ),
    }


def _read_composite_verifier_timeout_artifacts(
    paths: CellPaths,
    trial_path: Path,
    *,
    execution: int,
    known_payloads: Mapping[str, bytes] | None = None,
) -> dict[str, bytes] | None:
    """Read the bounded host-owned evidence without following mutable links."""

    trial_path = Path(trial_path)
    verifier_dir = trial_path / "verifier"
    attempt_dir = paths.attempts_dir / f"execution-{execution:04d}"
    if any(
        candidate.is_symlink() or not candidate.is_dir()
        for candidate in (trial_path, verifier_dir, attempt_dir)
    ):
        return None
    artifact_paths = _composite_verifier_timeout_artifact_paths(
        paths, trial_path, execution=execution
    )
    supplied = dict(known_payloads or {})
    if not set(supplied).issubset(artifact_paths):
        return None
    payloads: dict[str, bytes] = {}
    try:
        for name, path in artifact_paths.items():
            if name in supplied:
                payload = supplied[name]
                if not isinstance(payload, bytes):
                    return None
            else:
                maximum = (
                    COMPOSITE_VERIFIER_TIMEOUT_HOST_LOG_MAX_BYTES
                    if name == "trial/trial.log"
                    else (
                        COMPOSITE_VERIFIER_TIMEOUT_PARTIAL_MAX_BYTES
                        if name.endswith(".partial.txt")
                        else MAX_ARTIFACT_BYTES
                    )
                )
                payload = _read_regular_bytes(path, max_bytes=maximum)
            info = path.stat(follow_symlinks=False)
            if info.st_mode & 0o022:
                return None
            payloads[name] = payload
    except (OSError, PoolError):
        return None
    return payloads


def _composite_agent_exit_137_evidence(
    trial: Mapping[str, Any],
) -> dict[str, Any] | None:
    exception = trial.get("exception_info")
    if not isinstance(exception, Mapping) or not exception:
        return None
    message = exception.get("exception_message")
    traceback_text = exception.get("exception_traceback")
    if (
        exception.get("exception_type") != "NonZeroAgentExitCodeError"
        or not isinstance(message, str)
        or not message.startswith("Command failed (exit 137): ")
        or " codex exec " not in message
        or not message.endswith("\n\nstderr: None")
        or not isinstance(traceback_text, str)
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]single_step\.py", '
            r"line [0-9]+, in _run_agent\n",
            traceback_text,
        )
        is None
        or re.search(
            r'File "[^"\n]*[/\\]harbor[/\\]agents[/\\]installed[/\\]base\.py", '
            r"line [0-9]+, in _exec\n",
            traceback_text,
        )
        is None
        or "raise self._classify_exec_error(command, result)\n" not in traceback_text
        or not traceback_text.endswith(
            "harbor.agents.installed.base.NonZeroAgentExitCodeError: "
            + message
            + "\n"
        )
        or _parse_time(exception.get("occurred_at")) is None
    ):
        return None
    return {
        "exception_type": "NonZeroAgentExitCodeError",
        "agent_exit_code": 137,
        "exception_message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "exception_traceback_sha256": hashlib.sha256(
            traceback_text.encode("utf-8")
        ).hexdigest(),
        "exception_occurred_at": exception.get("occurred_at"),
    }


def _composite_secondary_timeout_log_evidence(
    payload: bytes,
    *,
    trial_name: str,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    try:
        text = payload.decode("utf-8")
    except UnicodeError:
        return None
    message = f"Verifier execution timed out after {timeout_seconds:.1f} seconds"
    failure_line = f"Trial {trial_name} failed: {message}"
    skip_line = "Skipping exception record because trial already has exception_info"
    final_line = f"harbor.trial.errors.VerifierTimeoutError: {message}\n"
    marker = f"{failure_line}\n{skip_line}\n"
    if (
        not payload
        or len(payload) > COMPOSITE_VERIFIER_TIMEOUT_HOST_LOG_MAX_BYTES
        or "\r" in text
        or text.count(failure_line) != 1
        or text.count(skip_line) != 1
        or text.count("raise VerifierTimeoutError(\n") != 1
        or text.count(final_line) != 1
        or marker not in text
        or not text.endswith(final_line)
    ):
        return None
    traceback_text = text.split(marker, 1)[1]
    required_frames = (
        r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]trial\.py", '
        r"line [0-9]+, in run\n",
        r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]single_step\.py", '
        r"line [0-9]+, in _run\n",
        r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]single_step\.py", '
        r"line [0-9]+, in _run_verifier\n",
        r'File "[^"\n]*[/\\]harbor[/\\]trial[/\\]trial\.py", '
        r"line [0-9]+, in _run_shared_verifier\n",
    )
    if (
        not traceback_text.startswith("Traceback (most recent call last):\n")
        or traceback_text.count(
            "The above exception was the direct cause of the following exception:\n"
        )
        != 2
        or any(re.search(frame, traceback_text) is None for frame in required_frames)
        or "asyncio.exceptions.CancelledError\n" not in traceback_text
        or "raise TimeoutError from exc_val\nTimeoutError\n" not in traceback_text
    ):
        return None
    return {
        "exception_type": "VerifierTimeoutError",
        "message": message,
        "timeout_seconds": timeout_seconds,
        "skip_preserved_primary_exception": True,
        "trial_log_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _composite_timeout_receipt_evidence(
    payloads: Mapping[str, bytes],
    *,
    trial_name: str,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        receipt = _json_mapping_from_bytes(
            payloads["trial/verifier/verifier-timeout-resilience.json"],
            label="composite timeout receipt",
        )
    except (KeyError, PoolError):
        return None
    records = receipt.get("records")
    if (
        set(receipt) != {"schema", "policy", "session_id", "records", "updated_at"}
        or receipt.get("schema") != COMPOSITE_VERIFIER_TIMEOUT_RECEIPT_SCHEMA
        or receipt.get("policy") != COMPOSITE_VERIFIER_TIMEOUT_RECEIPT_POLICY
        or receipt.get("session_id") != f"{trial_name}__env"
        or not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], Mapping)
        or _parse_time(receipt.get("updated_at")) is None
    ):
        return None
    record = records[0]
    partial = record.get("partial_stdout")
    process_snapshot = record.get("process_snapshot")
    partial_payload = payloads.get(
        "trial/verifier/test-stdout.timeout-1.partial.txt"
    )
    if (
        set(record)
        != {
            "timeout_attempt",
            "recorded_at",
            "script_sha256",
            "process_snapshot",
            "retry_eligible",
            "retry_state",
            "partial_stdout",
        }
        or record.get("timeout_attempt") != 1
        or _parse_time(record.get("recorded_at")) is None
        or record.get("script_sha256") != contract.get("verifier_script_sha256")
        or record.get("retry_eligible") is not False
        or record.get("retry_state") != "not_eligible"
        or not isinstance(process_snapshot, Mapping)
        or dict(process_snapshot) != dict(contract.get("process_snapshot", {}))
        or process_snapshot.get("complete") is not True
        or not isinstance(process_snapshot.get("network_processes"), list)
        or not isinstance(process_snapshot.get("test_processes"), list)
        or not process_snapshot.get("test_processes")
        or not isinstance(process_snapshot.get("other_process_count"), int)
        or isinstance(process_snapshot.get("other_process_count"), bool)
        or process_snapshot.get("other_process_count") < 0
        or not isinstance(partial, Mapping)
        or set(partial)
        != {"path", "bytes_copied", "source_bytes_at_snapshot", "tail_truncated"}
        or partial.get("path") != "test-stdout.timeout-1.partial.txt"
        or partial.get("tail_truncated") is not False
        or not isinstance(partial_payload, bytes)
        or not partial_payload
        or len(partial_payload) > COMPOSITE_VERIFIER_TIMEOUT_PARTIAL_MAX_BYTES
        or partial.get("bytes_copied") != len(partial_payload)
        or partial.get("source_bytes_at_snapshot") != len(partial_payload)
    ):
        return None
    return {
        "schema": receipt["schema"],
        "receipt_sha256": hashlib.sha256(
            payloads["trial/verifier/verifier-timeout-resilience.json"]
        ).hexdigest(),
        "recorded_at": record["recorded_at"],
        "updated_at": receipt["updated_at"],
        "verifier_script_sha256": record["script_sha256"],
        "process_snapshot_sha256": _digest(process_snapshot),
        "partial_stdout_sha256": hashlib.sha256(partial_payload).hexdigest(),
        "partial_stdout_bytes": len(partial_payload),
    }


def _composite_timeout_clean_proot_evidence(
    payloads: Mapping[str, bytes],
    *,
    trial_name: str,
) -> dict[str, Any] | None:
    try:
        runtime = _json_mapping_from_bytes(
            payloads["trial/proot-runtime.json"], label="composite PRoot runtime"
        )
        bridge = _json_mapping_from_bytes(
            payloads["trial/proot-package-bridge.json"],
            label="composite package bridge",
        )
    except (KeyError, PoolError):
        return None
    runtime_events = runtime.get("events")
    bridge_events = bridge.get("events")
    broker_events = bridge.get("broker_events")
    expected_runtime_events = [
        "rpc_transport_poisoned",
        "rpc_runtime_stopped",
        "environment_stop_finalized",
    ]
    expected_bridge_events = [
        "startup_started",
        "broker_ready",
        "probe_completed",
        "broker_stop_started",
        "broker_stopped",
        "environment_stop_finalized",
    ]
    if (
        runtime.get("schema") != PROOT_RUNTIME_RECEIPT_SCHEMA
        or runtime.get("session_id") != f"{trial_name}__env"
        or runtime.get("cleanup_complete") is not True
        or runtime.get("cleanup_incomplete") is not False
        or runtime.get("rpc_cleanup_complete") is not True
        or runtime.get("rpc_poisoned") is not True
        or runtime.get("rpc_poison_reason") != "wire_exec_cancelled"
        or runtime.get("rpc_returncode") != -int(signal.SIGKILL)
        or runtime.get("rpc_signal") != int(signal.SIGKILL)
        or not isinstance(runtime.get("rpc_pid"), int)
        or isinstance(runtime.get("rpc_pid"), bool)
        or runtime.get("rpc_pid") <= 1
        or not isinstance(runtime_events, list)
        or [
            event.get("event") if isinstance(event, Mapping) else None
            for event in runtime_events
        ]
        != expected_runtime_events
        or not isinstance(runtime_events[0], Mapping)
        or runtime_events[0].get("detail") != "wire_exec_cancelled"
        or bridge.get("schema") != PROOT_PACKAGE_BRIDGE_RECEIPT_SCHEMA
        or bridge.get("session_id") != f"{trial_name}__env"
        or bridge.get("enabled") is not True
        or bridge.get("healthy") is not True
        or bridge.get("ready") is not False
        or bridge.get("status") != "stopped"
        or bridge.get("cleanup_complete") is not True
        or bridge.get("rpc_cleanup_complete") is not True
        or bridge.get("fatal_error") is not None
        or bridge.get("policy") != PROOT_PACKAGE_BRIDGE_POLICY
        or bridge.get("failure_policy") != "closed"
        or bridge.get("guest_socket_path")
        != "/run/harbor-package-bridge/control.sock"
        or not isinstance(bridge_events, list)
        or [
            event.get("event") if isinstance(event, Mapping) else None
            for event in bridge_events
        ]
        != expected_bridge_events
        or not isinstance(broker_events, list)
        or len(broker_events) < 3
        or not isinstance(broker_events[0], Mapping)
        or broker_events[0].get("event") != "broker_started"
        or not isinstance(broker_events[-1], Mapping)
        or broker_events[-1].get("event") != "broker_stopped"
        or any(
            not isinstance(event, Mapping)
            or event.get("event") != "command_completed"
            or event.get("return_code") != 0
            or event.get("error_type") is not None
            for event in broker_events[1:-1]
        )
        or not any(
            isinstance(event, Mapping) and event.get("command") == "dpkg-query"
            for event in broker_events[1:-1]
        )
    ):
        return None
    runtime_times = [
        _parse_time(event.get("recorded_at"))
        for event in runtime_events
        if isinstance(event, Mapping)
    ]
    bridge_times = [
        _parse_time(event.get("recorded_at"))
        for event in bridge_events
        if isinstance(event, Mapping)
    ]
    runtime_updated = _parse_time(runtime.get("updated_at"))
    bridge_updated = _parse_time(bridge.get("updated_at"))
    if (
        len(runtime_times) != len(runtime_events)
        or len(bridge_times) != len(bridge_events)
        or any(value is None for value in (*runtime_times, *bridge_times))
        or runtime_updated is None
        or bridge_updated is None
        or any(
            earlier > later
            for earlier, later in zip(runtime_times, runtime_times[1:], strict=False)
        )
        or any(
            earlier > later
            for earlier, later in zip(bridge_times, bridge_times[1:], strict=False)
        )
        or runtime_times[-1] != runtime_updated
        or bridge_times[-1] != bridge_updated
        or runtime_updated > bridge_updated
    ):
        return None
    return {
        "runtime_receipt_sha256": hashlib.sha256(
            payloads["trial/proot-runtime.json"]
        ).hexdigest(),
        "package_bridge_receipt_sha256": hashlib.sha256(
            payloads["trial/proot-package-bridge.json"]
        ).hexdigest(),
        "rpc_poison_reason": "wire_exec_cancelled",
        "rpc_cleanup_complete": True,
        "package_bridge_cleanup_complete": True,
        "runtime_event_times": runtime_times,
        "runtime_updated_at": runtime_updated,
        "package_bridge_updated_at": bridge_updated,
    }


def _composite_verifier_timeout_failure(
    spec: PoolSpec,
    cell: CellSpec,
    *,
    execution: int,
    result: Mapping[str, Any],
    state: Mapping[str, Any],
    dispatch: Mapping[str, Any],
    trial: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> dict[str, Any] | None:
    """Validate the allowlisted exit-137 plus secondary verifier timeout."""

    contract = _composite_verifier_timeout_contract(spec, cell, execution=execution)
    if (
        contract is None
        or set(payloads) != _COMPOSITE_VERIFIER_TIMEOUT_REPLAY_ARTIFACT_KEYS
    ):
        return None
    expected_hashes = contract["artifact_sha256"]
    observed_hashes = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    if any(observed_hashes[name] != expected_hashes[name] for name in payloads):
        return None
    try:
        raw_result = _json_mapping_from_bytes(
            payloads["result.json"], label="composite original result"
        )
        raw_state = _json_mapping_from_bytes(
            payloads["state.json"], label="composite original state"
        )
        raw_dispatch = _json_mapping_from_bytes(
            payloads["dispatch.json"], label="composite original dispatch"
        )
        raw_trial = _json_mapping_from_bytes(
            payloads["trial-result.json"], label="composite Harbor result"
        )
        run = _json_mapping_from_bytes(payloads["run.json"], label="composite run")
        trial_config = _json_mapping_from_bytes(
            payloads["trial/config.json"], label="composite trial config"
        )
        trial_lock = _json_mapping_from_bytes(
            payloads["trial/lock.json"], label="composite trial lock"
        )
    except PoolError:
        return None
    identity = {
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
    }
    if (
        raw_result != dict(result)
        or raw_state != dict(state)
        or raw_dispatch != dict(dispatch)
        or raw_trial != dict(trial)
        or any(result.get(key) != value for key, value in identity.items())
        or any(state.get(key) != value for key, value in identity.items())
        or any(dispatch.get(key) != value for key, value in identity.items())
        or result.get("execution_attempt") != execution
        or state.get("execution_attempt") != execution
        or dispatch.get("execution_attempt") != execution
        or result.get("status") != "completed"
        or state.get("status") != "completed"
        or dispatch.get("status") != "terminal"
        or result.get("outcome") != "errored"
        or state.get("outcome") != "errored"
        or dispatch.get("outcome") != "errored"
        or result.get("rewards") != {}
        or result.get("verifier_result") is not None
        or trial.get("verifier_result") is not None
        or _result_rewards(trial)
        or trial.get("verifier_environment_mode") != "shared"
        or trial.get("trial_name") != contract.get("trial_name")
        or trial.get("id") != contract.get("trial_id")
        or trial.get("task_checksum") != contract.get("task_checksum")
        or trial.get("task_name") != cell.task_name
        or trial.get("source") != cell.source
        or _digest(result.get("exception_info"))
        != _digest(trial.get("exception_info"))
        or run.get("schema") != RUN_RECORD_SCHEMA
        or run.get("action") != "run"
        or run.get("cell_id") != cell.cell_id
        or run.get("execution_attempt") != execution
        or run.get("environment_backend") != "frozen"
        or run.get("status") != "exited"
        or run.get("return_code") != 1
    ):
        return None
    task_id = trial.get("task_id")
    config = trial.get("config")
    config_task = config.get("task") if isinstance(config, Mapping) else None
    config_agent = config.get("agent") if isinstance(config, Mapping) else None
    config_environment = (
        config.get("environment") if isinstance(config, Mapping) else None
    )
    config_verifier = config.get("verifier") if isinstance(config, Mapping) else None
    trial_config_task = trial_config.get("task")
    lock_task = trial_lock.get("task")
    timeout_multiplier = (
        config.get("timeout_multiplier") if isinstance(config, Mapping) else None
    )
    lock_timeout_multiplier = trial_lock.get("timeout_multiplier")
    if (
        not isinstance(task_id, Mapping)
        or task_id.get("ref") != cell.task_ref
        or not isinstance(config, Mapping)
        or not isinstance(config_task, Mapping)
        or config_task.get("name") != cell.task_name
        or config_task.get("ref") != cell.task_ref
        or config_task.get("source") != cell.source
        or not isinstance(config_agent, Mapping)
        or config_agent.get("import_path") != _CODEX_R2EACT_AGENT_IMPORT_PATH
        or not isinstance(config_environment, Mapping)
        or config_environment.get("import_path") != _LOCAL_PROOT_ENVIRONMENT_IMPORT_PATH
        or not isinstance(config_verifier, Mapping)
        or config_verifier.get("disable") is not False
        or config_verifier.get("override_timeout_sec") is not None
        or config_verifier.get("max_timeout_sec") is not None
        or not isinstance(timeout_multiplier, (int, float))
        or isinstance(timeout_multiplier, bool)
        or float(timeout_multiplier) != 1.0
        or config.get("verifier_timeout_multiplier") is not None
        or not isinstance(trial_config_task, Mapping)
        or trial_config_task.get("name") != cell.task_name
        or trial_config_task.get("ref") != cell.task_ref
        or trial_config_task.get("source") != cell.source
        or trial_config.get("trial_name") != contract.get("trial_name")
        or trial_lock.get("schema_version") != 2
        or not isinstance(lock_task, Mapping)
        or lock_task.get("name") != cell.task_name
        or lock_task.get("type") != "package"
        or lock_task.get("digest") != cell.task_ref
        or lock_task.get("source") != cell.source
        or not isinstance(lock_timeout_multiplier, (int, float))
        or isinstance(lock_timeout_multiplier, bool)
        or float(lock_timeout_multiplier) != 1.0
    ):
        return None

    primary = _composite_agent_exit_137_evidence(trial)
    timeout_seconds = float(contract["configured_verifier_timeout_seconds"])
    secondary = _composite_secondary_timeout_log_evidence(
        payloads["trial/trial.log"],
        trial_name=str(contract["trial_name"]),
        timeout_seconds=timeout_seconds,
    )
    timeout_receipt = _composite_timeout_receipt_evidence(
        payloads, trial_name=str(contract["trial_name"]), contract=contract
    )
    proot = _composite_timeout_clean_proot_evidence(
        payloads, trial_name=str(contract["trial_name"])
    )
    if primary is None or secondary is None or timeout_receipt is None or proot is None:
        return None

    exception = trial["exception_info"]
    agent_timing = trial.get("agent_execution")
    verifier_timing = trial.get("verifier")
    trial_started = _parse_time(trial.get("started_at"))
    agent_started = (
        _parse_time(agent_timing.get("started_at"))
        if isinstance(agent_timing, Mapping)
        else None
    )
    agent_finished = (
        _parse_time(agent_timing.get("finished_at"))
        if isinstance(agent_timing, Mapping)
        else None
    )
    exception_at = _parse_time(exception.get("occurred_at"))
    verifier_started = (
        _parse_time(verifier_timing.get("started_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    verifier_finished = (
        _parse_time(verifier_timing.get("finished_at"))
        if isinstance(verifier_timing, Mapping)
        else None
    )
    trial_finished = _parse_time(trial.get("finished_at"))
    timeout_recorded = _parse_time(timeout_receipt.get("recorded_at"))
    timeout_receipt_updated = _parse_time(timeout_receipt.get("updated_at"))
    run_started = _parse_time(run.get("started_at"))
    run_finished = _parse_time(run.get("finished_at"))
    if any(
        value is None
        for value in (
            trial_started,
            agent_started,
            agent_finished,
            exception_at,
            verifier_started,
            timeout_recorded,
            timeout_receipt_updated,
            verifier_finished,
            trial_finished,
            run_started,
            run_finished,
        )
    ):
        return None
    assert verifier_started is not None
    assert verifier_finished is not None
    assert timeout_recorded is not None
    assert timeout_receipt_updated is not None
    assert trial_started is not None
    assert trial_finished is not None
    assert run_started is not None
    assert run_finished is not None
    assert agent_started is not None
    assert agent_finished is not None
    assert exception_at is not None
    verifier_elapsed = (verifier_finished - verifier_started).total_seconds()
    record_elapsed = (timeout_recorded - verifier_started).total_seconds()
    if (
        not (
            run_started
            <= trial_started
            <= agent_started
            < agent_finished
            <= exception_at
            < verifier_started
            <= timeout_recorded
            <= timeout_receipt_updated
            <= verifier_finished
            <= proot["runtime_updated_at"]
            <= proot["package_bridge_updated_at"]
            <= trial_finished
            <= run_finished
        )
        or not (
            timeout_seconds * 0.99
            <= verifier_elapsed
            <= timeout_seconds + max(5.0, timeout_seconds * 0.01)
        )
        or not (
            timeout_seconds * 0.99
            <= record_elapsed
            <= timeout_seconds + max(5.0, timeout_seconds * 0.01)
        )
        or proot["runtime_event_times"][0] < timeout_recorded
        or proot["runtime_event_times"][0] > verifier_finished
    ):
        return None
    return {
        "policy": COMPOSITE_VERIFIER_TIMEOUT_REPLAY_POLICY,
        "cause": "secondary_harbor_verifier_timeout_after_agent_exit_137",
        "primary_exception": primary,
        "secondary_exception": secondary,
        "configured_verifier_timeout_seconds": timeout_seconds,
        "configured_verifier_timeout_source": "pinned_task_ref_and_checksum",
        "timeout_receipt": timeout_receipt,
        "clean_proot_receipts": {
            key: value
            for key, value in proot.items()
            if key
            not in {
                "runtime_event_times",
                "runtime_updated_at",
                "package_bridge_updated_at",
            }
        },
        "authority_artifact_sha256": observed_hashes,
    }


def _committed_verifier_timeout_candidate(
    spec: PoolSpec,
    cell: CellSpec,
    *,
    descriptor: int,
    expected_execution: int,
    allow_bootstrap: bool = False,
) -> dict[str, Any]:
    paths = spec.cell_paths(cell).ensure()
    result_bytes = _read_regular_bytes(paths.result, max_bytes=MAX_ARTIFACT_BYTES)
    state_bytes = _read_regular_bytes(paths.state, max_bytes=MAX_ARTIFACT_BYTES)
    dispatch_bytes = _read_regular_bytes(paths.dispatch, max_bytes=MAX_ARTIFACT_BYTES)
    result = _json_mapping_from_bytes(result_bytes, label="original cell result")
    state = _json_mapping_from_bytes(state_bytes, label="original cell state")
    dispatch = _json_mapping_from_bytes(dispatch_bytes, label="original cell dispatch")
    primary_failure = _verifier_timeout_trial_failure(result)
    bootstrap = bool(allow_bootstrap and result.get("outcome") == "failed")
    expected_outcome = "failed" if bootstrap else "errored"
    expected_rewards = {"reward": 0.0} if bootstrap else {}
    execution = _valid_execution(result.get("execution_attempt"))
    identity = {
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
    }
    if (
        result.get("schema") != CELL_RESULT_SCHEMA
        or any(result.get(key) != value for key, value in identity.items())
        or result.get("status") != "completed"
        or result.get("outcome") != expected_outcome
        or result.get("rewards") != expected_rewards
        or execution != expected_execution
    ):
        raise PoolError(
            f"cell is not the requested committed verifier timeout: {cell.cell_id}"
        )
    if (
        state.get("schema") != CELL_STATE_SCHEMA
        or dispatch.get("schema") != CELL_DISPATCH_SCHEMA
        or any(state.get(key) != value for key, value in identity.items())
        or any(dispatch.get(key) != value for key, value in identity.items())
        or state.get("status") != "completed"
        or dispatch.get("status") != "terminal"
        or state.get("outcome") != expected_outcome
        or dispatch.get("outcome") != expected_outcome
        or state.get("execution_attempt") != execution
        or dispatch.get("execution_attempt") != execution
        or paths.exception.exists()
        or paths.external_reservation.exists()
    ):
        raise PoolError(
            f"committed verifier-timeout metadata is inconsistent: {cell.cell_id}"
        )
    claim = ClaimedCell(
        cell=cell,
        paths=paths,
        descriptor=descriptor,
        execution_attempt=execution,
        dispatch=dispatch,
    )
    terminal = _terminal_harbor_trial(spec, claim)
    if terminal is None:
        raise PoolError(f"verifier-timeout cell has no Harbor trial: {cell.cell_id}")
    trial_path, trial = terminal
    trial_result_path = trial_path / "result.json"
    trial_result_bytes = _read_regular_bytes(
        trial_result_path, max_bytes=MAX_ARTIFACT_BYTES
    )
    harbor_primary_failure = _verifier_timeout_trial_failure(trial)
    relative_trial = str(trial_path.relative_to(spec.paths.root))
    if (
        (not bootstrap and trial.get("verifier_result") is not None)
        or _result_rewards(trial) != expected_rewards
        or _digest(result.get("exception_info")) != _digest(trial.get("exception_info"))
        or result.get("trial_path") != relative_trial
        or result.get("harbor_job_path")
        != str(paths.harbor_job.relative_to(spec.paths.root))
        or result.get("trial_name") != trial.get("trial_name")
        or trial.get("trial_name") != trial_path.name
    ):
        raise PoolError(
            f"pool and Harbor verifier-timeout provenance disagree: {cell.cell_id}"
        )
    raw = {
        "result.json": result_bytes,
        "state.json": state_bytes,
        "dispatch.json": dispatch_bytes,
        "trial-result.json": trial_result_bytes,
    }
    if bootstrap:
        ctrf = trial_path / "verifier" / "ctrf.json"
        if ctrf.exists() or ctrf.is_symlink():
            raise PoolError("bootstrap replay cannot override an original test report")
        log = _read_regular_bytes(
            trial_path / "verifier" / "test-stdout.txt", max_bytes=MAX_ARTIFACT_BYTES
        )
        failure = _verifier_bootstrap_failure(result, trial, log)
        if failure is None:
            raise PoolError("original failure is not a proven verifier bootstrap failure")
        raw[VERIFIER_BOOTSTRAP_LOG_KEY] = log
    elif primary_failure is not None:
        if harbor_primary_failure != primary_failure:
            raise PoolError(
                f"pool and Harbor verifier-timeout provenance disagree: {cell.cell_id}"
            )
        failure: Mapping[str, Any] = primary_failure
    else:
        known_payloads = {
            "result.json": result_bytes,
            "state.json": state_bytes,
            "dispatch.json": dispatch_bytes,
            "trial-result.json": trial_result_bytes,
        }
        composite_payloads = _read_composite_verifier_timeout_artifacts(
            paths,
            trial_path,
            execution=execution,
            known_payloads=known_payloads,
        )
        composite_failure = (
            _composite_verifier_timeout_failure(
                spec,
                cell,
                execution=execution,
                result=result,
                state=state,
                dispatch=dispatch,
                trial=trial,
                payloads=composite_payloads,
            )
            if composite_payloads is not None
            else None
        )
        if composite_failure is None:
            raise PoolError(
                f"cell is not the requested committed verifier timeout: {cell.cell_id}"
            )
        failure = composite_failure
        raw = dict(composite_payloads)
    return {
        "claim": claim,
        "result": result,
        "state": state,
        "dispatch": dispatch,
        "trial": trial,
        "trial_path": trial_path,
        "failure": failure,
        "raw": raw,
    }


def _snapshot_record(spec: PoolSpec, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(spec.paths.root)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _publish_verifier_replay_receipt(
    spec: PoolSpec,
    candidate: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> tuple[Path, dict[str, Any], str]:
    claim = candidate["claim"]
    assert isinstance(claim, ClaimedCell)
    execution = claim.execution_attempt
    attempt_dir = _ensure_dir(claim.paths.attempts_dir / f"execution-{execution:04d}")
    destination = attempt_dir / VERIFIER_REPLAY_COMMIT_DIR_NAME
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise PoolError("unsafe verifier replay commit directory")
        receipt_path = destination / "receipt.json"
        receipt = _read_json(receipt_path, max_bytes=MAX_ARTIFACT_BYTES)
        original = receipt.get("original")
        expected_original = {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in candidate["raw"].items()
        }
        if (
            receipt.get("schema") != VERIFIER_REPLAY_RECEIPT_SCHEMA
            or receipt.get("cell_id") != claim.cell.cell_id
            or receipt.get("execution_attempt") != execution
            or receipt.get("manifest_sha256") != manifest_sha256
            or receipt.get("entry_sha256") != prepared["entry_sha256"]
            or not isinstance(original, Mapping)
            or any(
                not isinstance(original.get(name), Mapping)
                or original[name].get("sha256") != digest
                for name, digest in expected_original.items()
            )
        ):
            raise PoolError("existing verifier replay receipt does not match")
        receipt_sha256 = hashlib.sha256(
            _read_regular_bytes(receipt_path, max_bytes=MAX_ARTIFACT_BYTES)
        ).hexdigest()
        return receipt_path, receipt, receipt_sha256

    staging = attempt_dir / f".{VERIFIER_REPLAY_COMMIT_DIR_NAME}-{uuid.uuid4().hex}"
    entry = prepared["entry"]
    receipt: dict[str, Any] = {}
    try:
        _ensure_dir(staging)
        manifest_snapshot = staging / "manifest.json"
        _write_private_bytes(manifest_snapshot, manifest_bytes)
        original_dir = _ensure_dir(staging / "original")
        original_records: dict[str, Any] = {}
        for name, payload in candidate["raw"].items():
            snapshot = original_dir / name
            _write_private_bytes(snapshot, payload)
            final_snapshot = destination / "original" / name
            original_records[name] = _snapshot_record(spec, final_snapshot, payload)

        evidence_records: list[dict[str, Any]] = []
        for index, evidence in enumerate(prepared["evidence"]):
            category = evidence["category"]
            name = evidence["name"]
            payload = evidence["payload"]
            snapshot = _ensure_dir(staging / "evidence" / category) / (
                f"{index:02d}-{name}.bin"
            )
            _write_private_bytes(snapshot, payload)
            final_snapshot = (
                destination / "evidence" / category / f"{index:02d}-{name}.bin"
            )
            record = _snapshot_record(spec, final_snapshot, payload)
            record.update(
                {
                    "category": category,
                    "name": name,
                    "source_path": evidence["source_path"],
                    "resolved_source_path": evidence["resolved_source_path"],
                    "target_path": evidence["target_path"],
                }
            )
            evidence_records.append(record)

        now = _utc_now()
        receipt = {
            "schema": VERIFIER_REPLAY_RECEIPT_SCHEMA,
            "cell_id": claim.cell.cell_id,
            "task_name": claim.cell.task_name,
            "task_ref": claim.cell.task_ref,
            "official_attempt_index": claim.cell.official_attempt_index,
            "execution_attempt": execution,
            "config_contract_hash": spec.contract_hash,
            "kind": entry["kind"],
            "reward": _verifier_replay_reward(entry["reward"]),
            "model_called": False,
            "pytest_exit_code": entry.get("pytest_exit_code"),
            "runner_receipt_evidence": entry["runner_receipt_evidence"],
            "harbor_result_evidence": entry.get("harbor_result_evidence"),
            "manifest_source_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "manifest_snapshot": _snapshot_record(
                spec, destination / "manifest.json", manifest_bytes
            ),
            "entry_sha256": prepared["entry_sha256"],
            "original": original_records,
            "original_trial_path": str(
                candidate["trial_path"].relative_to(spec.paths.root)
            ),
            "original_timeout_failure": dict(candidate["failure"]),
            "evidence": evidence_records,
            "official_attempt_consumed_by_replay": False,
            "replay_policy": "same_agent_state_verifier_only_v1",
            "prepared_at": now,
        }
        _atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
        _fsync_directory(attempt_dir)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt_path = destination / "receipt.json"
    receipt_sha256 = hashlib.sha256(
        _read_regular_bytes(receipt_path, max_bytes=MAX_ARTIFACT_BYTES)
    ).hexdigest()
    return receipt_path, receipt, receipt_sha256


def _contained_replay_snapshot(
    spec: PoolSpec,
    commit_dir: Path,
    record: Mapping[str, Any],
) -> bytes:
    relative = record.get("path")
    expected_sha256 = record.get("sha256")
    expected_bytes = record.get("bytes")
    if (
        not isinstance(relative, str)
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise PoolError("invalid verifier replay snapshot record")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise PoolError("verifier replay snapshot path escapes the pool")
    candidate = spec.paths.root / relative_path
    try:
        candidate.relative_to(commit_dir)
    except ValueError as exc:
        raise PoolError(
            "verifier replay snapshot is outside its commit directory"
        ) from exc
    if candidate.is_symlink():
        raise PoolError("verifier replay snapshot must not be a symlink")
    payload = _read_regular_bytes(candidate, max_bytes=MAX_ARTIFACT_BYTES)
    if (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise PoolError("verifier replay snapshot digest mismatch")
    return payload


def _validated_verifier_replay_reward(
    spec: PoolSpec,
    cell: CellSpec,
    result: Mapping[str, Any],
) -> float:
    replay = result.get("verifier_replay")
    if not isinstance(replay, Mapping):
        raise PoolError("cell result has no verifier replay projection")
    reward = _verifier_replay_reward(
        result.get("rewards", {}).get("reward")
        if isinstance(result.get("rewards"), Mapping)
        else None
    )
    execution = _valid_execution(result.get("execution_attempt"))
    if (
        result.get("schema") != CELL_RESULT_SCHEMA
        or result.get("cell_id") != cell.cell_id
        or result.get("task_name") != cell.task_name
        or result.get("task_ref") != cell.task_ref
        or result.get("official_attempt_index") != cell.official_attempt_index
        or result.get("config_contract_hash") != spec.contract_hash
        or result.get("status") != "completed"
        or result.get("outcome") != ("passed" if reward > 0 else "failed")
    ):
        raise PoolError("invalid verifier replay terminal result")
    expected_commit_dir = (
        spec.cell_paths(cell).attempts_dir
        / f"execution-{execution:04d}"
        / VERIFIER_REPLAY_COMMIT_DIR_NAME
    )
    expected_receipt_path = expected_commit_dir / "receipt.json"
    expected_relative = str(expected_receipt_path.relative_to(spec.paths.root))
    if (
        execution < 1
        or replay.get("schema") != VERIFIER_REPLAY_RECEIPT_SCHEMA
        or replay.get("receipt_path") != expected_relative
        or replay.get("model_called") is not False
        or result.get("exception_info") is not None
        or result.get("rejected_verifier_rewards") is not None
        or result.get("infrastructure_cleanup_warning") is not None
    ):
        raise PoolError("invalid verifier replay cell projection")
    receipt_bytes = _read_regular_bytes(
        expected_receipt_path, max_bytes=MAX_ARTIFACT_BYTES
    )
    if hashlib.sha256(receipt_bytes).hexdigest() != replay.get("receipt_sha256"):
        raise PoolError("verifier replay receipt digest mismatch")
    receipt = _json_mapping_from_bytes(receipt_bytes, label="commit receipt")
    identity = {
        "cell_id": cell.cell_id,
        "task_name": cell.task_name,
        "task_ref": cell.task_ref,
        "official_attempt_index": cell.official_attempt_index,
        "execution_attempt": execution,
        "config_contract_hash": spec.contract_hash,
    }
    if (
        receipt.get("schema") != VERIFIER_REPLAY_RECEIPT_SCHEMA
        or any(receipt.get(key) != value for key, value in identity.items())
        or receipt.get("reward") != reward
        or receipt.get("kind") != replay.get("kind")
        or receipt.get("model_called") is not False
        or receipt.get("official_attempt_consumed_by_replay") is not False
        or receipt.get("replay_policy") != "same_agent_state_verifier_only_v1"
    ):
        raise PoolError("verifier replay receipt identity mismatch")

    manifest_record = receipt.get("manifest_snapshot")
    if not isinstance(manifest_record, Mapping):
        raise PoolError("verifier replay receipt has no manifest snapshot")
    manifest_bytes = _contained_replay_snapshot(
        spec, expected_commit_dir, manifest_record
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != receipt.get("manifest_sha256"):
        raise PoolError("verifier replay manifest snapshot hash mismatch")
    manifest = _json_mapping_from_bytes(manifest_bytes, label="snapshot manifest")
    entries = manifest.get("entries")
    if (
        manifest.get("schema") != VERIFIER_REPLAY_MANIFEST_SCHEMA
        or manifest.get("job_name") != spec.paths.root.name
        or manifest.get("config_contract_hash") != spec.contract_hash
        or not isinstance(entries, list)
    ):
        raise PoolError("verifier replay snapshot manifest does not match the pool")
    matching_entries = [
        dict(item)
        for item in entries
        if isinstance(item, Mapping) and item.get("cell_id") == cell.cell_id
    ]
    if len(matching_entries) != 1 or _digest(matching_entries[0]) != receipt.get(
        "entry_sha256"
    ):
        raise PoolError("verifier replay manifest entry digest mismatch")
    entry = matching_entries[0]
    if (
        entry.get("task_name") != cell.task_name
        or entry.get("task_ref") != cell.task_ref
        or entry.get("official_attempt_index") != cell.official_attempt_index
        or entry.get("execution_attempt") != execution
        or entry.get("kind") != receipt.get("kind")
        or _verifier_replay_reward(entry.get("reward")) != reward
        or entry.get("model_called") is not False
    ):
        raise PoolError("verifier replay manifest entry identity mismatch")

    receipt_evidence = receipt.get("evidence")
    manifest_evidence = entry.get("evidence")
    if not isinstance(receipt_evidence, list) or not isinstance(
        manifest_evidence, Mapping
    ):
        raise PoolError("verifier replay evidence index is invalid")
    expected_evidence: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for category in VERIFIER_REPLAY_EVIDENCE_CATEGORIES:
        items = manifest_evidence.get(category)
        if not isinstance(items, list) or not items:
            raise PoolError("verifier replay manifest evidence is incomplete")
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
                raise PoolError("verifier replay manifest evidence is invalid")
            name = str(item["name"])
            if name in expected_evidence:
                raise PoolError("duplicate verifier replay evidence name")
            expected_evidence[name] = (category, item)
    payloads: dict[str, bytes] = {}
    for record in receipt_evidence:
        if not isinstance(record, Mapping):
            raise PoolError("verifier replay receipt evidence is invalid")
        name = record.get("name")
        expected = expected_evidence.pop(name, None) if isinstance(name, str) else None
        if expected is None:
            raise PoolError("unexpected verifier replay evidence snapshot")
        category, manifest_item = expected
        if (
            record.get("category") != category
            or record.get("source_path") != manifest_item.get("source_path")
            or record.get("sha256") != manifest_item.get("sha256")
            or record.get("target_path") != manifest_item.get("target_path")
        ):
            raise PoolError("verifier replay evidence metadata mismatch")
        payloads[name] = _contained_replay_snapshot(spec, expected_commit_dir, record)
    if expected_evidence:
        raise PoolError("verifier replay receipt omitted evidence")
    _validate_verifier_replay_execution_evidence(cell, entry, payloads)

    original = receipt.get("original")
    if not isinstance(original, Mapping):
        raise PoolError("verifier replay receipt omitted original artifacts")
    recorded_failure = receipt.get("original_timeout_failure")
    bootstrap = bool(
        isinstance(recorded_failure, Mapping)
        and recorded_failure.get("policy") in VERIFIER_BOOTSTRAP_REPLAY_POLICIES
    )
    if bootstrap and entry.get("original_failure_policy") != recorded_failure.get("policy"):
        raise PoolError("bootstrap replay manifest must require completed test evidence")
    composite = bool(
        isinstance(recorded_failure, Mapping)
        and recorded_failure.get("policy") == COMPOSITE_VERIFIER_TIMEOUT_REPLAY_POLICY
        and recorded_failure.get("cause")
        == "secondary_harbor_verifier_timeout_after_agent_exit_137"
    )
    expected_original_names = (
        _COMPOSITE_VERIFIER_TIMEOUT_REPLAY_ARTIFACT_KEYS
        if composite
        else frozenset(
            {"result.json", "state.json", "dispatch.json", "trial-result.json"}
            | ({VERIFIER_BOOTSTRAP_LOG_KEY} if bootstrap else set())
        )
    )
    if (composite or bootstrap) and set(original) != expected_original_names:
        raise PoolError("verifier replay composite artifact index is incomplete")
    original_payloads: dict[str, bytes] = {}
    for name in expected_original_names:
        record = original.get(name)
        if not isinstance(record, Mapping):
            raise PoolError("verifier replay original artifact index is incomplete")
        original_payloads[name] = _contained_replay_snapshot(
            spec, expected_commit_dir, record
        )
    original_result = _json_mapping_from_bytes(
        original_payloads["result.json"], label="archived original result"
    )
    original_state = _json_mapping_from_bytes(
        original_payloads["state.json"], label="archived original state"
    )
    original_dispatch = _json_mapping_from_bytes(
        original_payloads["dispatch.json"], label="archived original dispatch"
    )
    original_trial = _json_mapping_from_bytes(
        original_payloads["trial-result.json"], label="archived original trial"
    )
    if bootstrap:
        failure = _verifier_bootstrap_failure(
            original_result, original_trial, original_payloads[VERIFIER_BOOTSTRAP_LOG_KEY]
        )
    elif composite:
        failure = _composite_verifier_timeout_failure(
            spec,
            cell,
            execution=execution,
            result=original_result,
            state=original_state,
            dispatch=original_dispatch,
            trial=original_trial,
            payloads=original_payloads,
        )
    else:
        failure = _verifier_timeout_trial_failure(original_result)
    if (
        failure is None
        or failure != recorded_failure
        or original_result.get("status") != "completed"
        or original_result.get("outcome") != ("failed" if bootstrap else "errored")
        or original_result.get("rewards") != ({"reward": 0.0} if bootstrap else {})
        or original_state.get("status") != "completed"
        or original_state.get("outcome") != ("failed" if bootstrap else "errored")
        or original_dispatch.get("status") != "terminal"
        or original_dispatch.get("outcome") != ("failed" if bootstrap else "errored")
        or _digest(original_result.get("exception_info"))
        != _digest(original_trial.get("exception_info"))
    ):
        raise PoolError("verifier replay archived timeout provenance is invalid")

    paths = spec.cell_paths(cell)
    synthetic = ClaimedCell(
        cell=cell,
        paths=paths,
        descriptor=-1,
        execution_attempt=execution,
        dispatch={},
    )
    terminal = _terminal_harbor_trial(spec, synthetic)
    if terminal is None:
        raise PoolError("verifier replay source Harbor trial disappeared")
    trial_path, current_trial = terminal
    current_trial_bytes = _read_regular_bytes(
        trial_path / "result.json", max_bytes=MAX_ARTIFACT_BYTES
    )
    if bootstrap:
        ctrf = trial_path / "verifier" / "ctrf.json"
        if ctrf.exists() or ctrf.is_symlink():
            raise PoolError("source bootstrap trial gained a test report after replay")
        current_log = _read_regular_bytes(
            trial_path / "verifier" / "test-stdout.txt", max_bytes=MAX_ARTIFACT_BYTES
        )
        current_failure = _verifier_bootstrap_failure(original_result, current_trial, current_log)
    elif composite:
        live_payloads = _read_composite_verifier_timeout_artifacts(
            paths,
            trial_path,
            execution=execution,
            known_payloads={
                "result.json": original_payloads["result.json"],
                "state.json": original_payloads["state.json"],
                "dispatch.json": original_payloads["dispatch.json"],
            },
        )
        current_failure = (
            _composite_verifier_timeout_failure(
                spec,
                cell,
                execution=execution,
                result=original_result,
                state=original_state,
                dispatch=original_dispatch,
                trial=current_trial,
                payloads=live_payloads,
            )
            if live_payloads is not None
            else None
        )
    else:
        current_failure = _verifier_timeout_trial_failure(current_trial)
    if (
        str(trial_path.relative_to(spec.paths.root))
        != receipt.get("original_trial_path")
        or result.get("trial_path") != receipt.get("original_trial_path")
        or result.get("trial_name") != current_trial.get("trial_name")
        or hashlib.sha256(current_trial_bytes).hexdigest()
        != original["trial-result.json"].get("sha256")
        or current_failure != failure
        or (not bootstrap and current_trial.get("verifier_result") is not None)
        or _result_rewards(current_trial) != ({"reward": 0.0} if bootstrap else {})
    ):
        raise PoolError("current Harbor timeout trial changed after replay")
    return reward


def _scoreable_verifier_replay_reward(
    spec: PoolSpec,
    cell: CellSpec,
    result: Mapping[str, Any],
) -> float | None:
    try:
        return _validated_verifier_replay_reward(spec, cell, result)
    except (OSError, PoolError, ValueError):
        return None


def commit_verifier_replay_results(
    *,
    job_name: str,
    manifest_path: Path,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
) -> dict[str, Any]:
    """Commit verifier-only rewards without another model or execution draw."""

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    manifest_source, manifest_bytes, prepared_entries = _load_verifier_replay_manifest(
        spec, manifest_path
    )
    by_cell = {entry["cell"].cell_id: entry for entry in prepared_entries}
    locked: list[int] = []
    candidates: list[dict[str, Any]] = []
    already_committed: list[str] = []
    try:
        for cell in spec.cells:
            prepared = by_cell.get(cell.cell_id)
            if prepared is None:
                continue
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                raise PoolError(
                    f"requested replay cell is currently owned: {cell.cell_id}"
                )
            locked.append(descriptor)
            current = _cell_result(spec, cell)
            if current is not None and isinstance(
                current.get("verifier_replay"), Mapping
            ):
                reward = _validated_verifier_replay_reward(spec, cell, current)
                replay = current["verifier_replay"]
                receipt = _read_json(
                    spec.paths.root / replay["receipt_path"],
                    max_bytes=MAX_ARTIFACT_BYTES,
                )
                if (
                    receipt.get("manifest_sha256")
                    != hashlib.sha256(manifest_bytes).hexdigest()
                    or receipt.get("entry_sha256") != prepared["entry_sha256"]
                    or reward != _verifier_replay_reward(prepared["entry"]["reward"])
                ):
                    raise PoolError(
                        f"cell already has a different verifier replay: {cell.cell_id}"
                    )
                already_committed.append(cell.cell_id)
                continue
            candidate = _committed_verifier_timeout_candidate(
                spec,
                cell,
                descriptor=descriptor,
                expected_execution=prepared["entry"]["execution_attempt"],
                allow_bootstrap=(prepared["entry"].get("original_failure_policy")
                                 in VERIFIER_BOOTSTRAP_REPLAY_POLICIES),
            )
            policy = prepared["entry"].get("original_failure_policy")
            if (policy in VERIFIER_BOOTSTRAP_REPLAY_POLICIES
                    and candidate["failure"].get("policy") != policy):
                raise PoolError("bootstrap replay policy disagrees with original failure")
            candidate["prepared"] = prepared
            candidates.append(candidate)

        committed: list[str] = []
        for candidate in candidates:
            claim = candidate["claim"]
            prepared = candidate["prepared"]
            receipt_path, receipt, receipt_sha256 = _publish_verifier_replay_receipt(
                spec,
                candidate,
                prepared,
                manifest_path=manifest_source,
                manifest_bytes=manifest_bytes,
            )
            original_result = candidate["result"]
            entry = prepared["entry"]
            now = _utc_now()
            projection = {
                "schema": VERIFIER_REPLAY_RECEIPT_SCHEMA,
                "kind": entry["kind"],
                "model_called": False,
                "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                "receipt_sha256": receipt_sha256,
                "original_trial_result_sha256": receipt["original"][
                    "trial-result.json"
                ]["sha256"],
            }
            result = dict(original_result)
            reward = _verifier_replay_reward(entry["reward"])
            result.update(
                {
                    "status": "completed",
                    "outcome": "passed" if reward > 0 else "failed",
                    "rewards": {"reward": reward},
                    "exception_info": None,
                    "infrastructure_cleanup_warning": None,
                    "finished_at": receipt.get("prepared_at") or now,
                    "committed_at": now,
                    "verifier_replay": projection,
                }
            )
            result.pop("rejected_verifier_rewards", None)
            if _validated_verifier_replay_reward(spec, claim.cell, result) != reward:
                raise PoolError(
                    "prepared verifier replay did not pass scoreability gate"
                )
            _commit_cell_result(spec, claim, result)
            if _validated_verifier_replay_reward(spec, claim.cell, result) != reward:
                raise PoolError(
                    "committed verifier replay did not pass scoreability gate"
                )
            committed.append(claim.cell.cell_id)
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "verifier_replay_committed",
                    "cell_id": claim.cell.cell_id,
                    "task_name": claim.cell.task_name,
                    "official_attempt_index": claim.cell.official_attempt_index,
                    "execution_attempt": claim.execution_attempt,
                    "reward": reward,
                    "receipt_path": projection["receipt_path"],
                    "receipt_sha256": receipt_sha256,
                    "official_attempt_consumed_by_replay": False,
                    "recorded_at": now,
                },
            )
    finally:
        for descriptor in reversed(locked):
            _close_lock(descriptor)
    pool_state = aggregate_pool(spec)
    return {
        "committed_cell_ids": committed,
        "already_committed_cell_ids": already_committed,
        **pool_state,
    }


def requeue_committed_verifier_timeout(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str],
) -> dict[str, Any]:
    """Archive and requeue explicitly named committed verifier timeouts.

    Every requested cell is locked and validated before the first artifact is
    moved.  A typed timeout alone is insufficient: the committed pool result
    and its single pinned Harbor trial must have identical exception
    provenance, and neither may contain a verifier result or reward.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested_order = list(cell_ids)
    if not requested_order:
        raise PoolError("repair-verifier-timeout requires at least one cell id")
    if len(set(requested_order)) != len(requested_order):
        raise PoolError("repair-verifier-timeout received a duplicate cell id")
    requested = set(requested_order)
    known = {cell.cell_id for cell in spec.cells}
    unknown = requested - known
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")

    locked_descriptors: list[int] = []
    candidates: list[dict[str, Any]] = []
    repaired: list[str] = []
    try:
        # Pool order gives one deterministic multi-cell lock order. Keep every
        # flock through validation and mutation to close ownership races.
        for cell in spec.cells:
            if cell.cell_id not in requested:
                continue
            paths = spec.cell_paths(cell).ensure()
            descriptor = _open_lock(paths.claim_lock, blocking=False)
            if descriptor is None:
                raise PoolError(f"requested cell is currently owned: {cell.cell_id}")
            locked_descriptors.append(descriptor)

            result = _cell_result(spec, cell)
            if result is None:
                raise PoolError(
                    f"requested cell has no terminal result: {cell.cell_id}"
                )
            failure = _verifier_timeout_trial_failure(result)
            if (
                result.get("status") != "completed"
                or result.get("outcome") != "errored"
                or result.get("rewards") != {}
                or failure is None
            ):
                raise PoolError(
                    "committed result lacks strict verifier-timeout evidence: "
                    f"{cell.cell_id}"
                )
            execution = _valid_execution(result.get("execution_attempt"))
            if execution < 1:
                raise PoolError(
                    f"committed verifier timeout has no execution: {cell.cell_id}"
                )
            state = _read_json(paths.state)
            dispatch = _read_json(paths.dispatch)
            if (
                state.get("schema") != CELL_STATE_SCHEMA
                or state.get("cell_id") != cell.cell_id
                or state.get("official_attempt_index") != cell.official_attempt_index
                or state.get("config_contract_hash") != spec.contract_hash
                or state.get("status") != "completed"
                or state.get("outcome") != "errored"
                or state.get("execution_attempt") != execution
                or dispatch.get("schema") != CELL_DISPATCH_SCHEMA
                or dispatch.get("cell_id") != cell.cell_id
                or dispatch.get("official_attempt_index") != cell.official_attempt_index
                or dispatch.get("config_contract_hash") != spec.contract_hash
                or dispatch.get("status") != "terminal"
                or dispatch.get("outcome") != "errored"
                or dispatch.get("execution_attempt") != execution
            ):
                raise PoolError(
                    "committed verifier-timeout metadata is inconsistent: "
                    f"{cell.cell_id}"
                )
            if paths.exception.exists() or paths.exception.is_symlink():
                raise PoolError(
                    "committed verifier-timeout cell has worker exception metadata: "
                    f"{cell.cell_id}"
                )
            if (
                paths.external_reservation.exists()
                or paths.external_reservation.is_symlink()
            ):
                raise PoolError(
                    "committed verifier-timeout cell has an external reservation: "
                    f"{cell.cell_id}"
                )

            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    f"committed verifier timeout has no Harbor trial: {cell.cell_id}"
                )
            trial_path, trial = terminal
            harbor_failure = _verifier_timeout_trial_failure(trial)
            pool_exception = result.get("exception_info")
            harbor_exception = trial.get("exception_info")
            relative_trial = str(trial_path.relative_to(spec.paths.root))
            relative_job = str(paths.harbor_job.relative_to(spec.paths.root))
            if (
                harbor_failure != failure
                or trial.get("verifier_result") is not None
                or _result_rewards(trial) != {}
                or not isinstance(pool_exception, Mapping)
                or not isinstance(harbor_exception, Mapping)
                or _digest(dict(pool_exception)) != _digest(dict(harbor_exception))
                or result.get("trial_path") != relative_trial
                or result.get("harbor_job_path") != relative_job
                or result.get("trial_name") != trial.get("trial_name")
                or trial.get("trial_name") != trial_path.name
            ):
                raise PoolError(
                    "pool and Harbor verifier-timeout provenance disagree: "
                    f"{cell.cell_id}"
                )

            attempt_dir = paths.attempts_dir / f"execution-{execution:04d}"
            terminal_archive = attempt_dir / "committed-terminal-verifier-timeout"
            harbor_archive = attempt_dir / "harbor-job-committed-verifier-timeout"
            for archive in (terminal_archive, harbor_archive):
                if archive.exists() or archive.is_symlink():
                    raise PoolError(
                        f"committed verifier-timeout archive already exists: {archive}"
                    )
            worker_exception_count = dispatch.get("worker_exception_count", 0)
            repair_count = dispatch.get("verifier_timeout_repair_count", 0)
            if (
                not isinstance(worker_exception_count, int)
                or isinstance(worker_exception_count, bool)
                or worker_exception_count < 0
                or not isinstance(repair_count, int)
                or isinstance(repair_count, bool)
                or repair_count < 0
            ):
                raise PoolError(
                    f"invalid verifier-timeout retry metadata: {cell.cell_id}"
                )
            candidates.append(
                {
                    "claim": claim,
                    "state": state,
                    "dispatch": dispatch,
                    "failure": failure,
                    "trial_path": trial_path,
                    "attempt_dir": attempt_dir,
                    "terminal_archive": terminal_archive,
                    "worker_exception_count": worker_exception_count,
                    "repair_count": repair_count + 1,
                }
            )

        for candidate in candidates:
            claim = candidate["claim"]
            cell = claim.cell
            paths = claim.paths
            execution = claim.execution_attempt
            state = candidate["state"]
            dispatch = candidate["dispatch"]
            failure = candidate["failure"]
            _ensure_dir(candidate["attempt_dir"])
            terminal_archive = candidate["terminal_archive"]
            _ensure_dir(terminal_archive)
            _atomic_write_json(terminal_archive / "state.json", state)
            _atomic_write_json(terminal_archive / "dispatch.json", dispatch)
            harbor_archive = _archive_verifier_timeout_harbor_job(
                claim,
                trial_path=candidate["trial_path"],
                failure=failure,
            )

            now = _utc_now()
            harbor_relative = str(harbor_archive.relative_to(spec.paths.root))
            repair_count = candidate["repair_count"]
            queued_dispatch = dict(dispatch)
            queued_dispatch.update(
                {
                    "status": "queued",
                    "verifier_timeout_repair_count": repair_count,
                    "last_verifier_timeout_execution_attempt": execution,
                    "last_verifier_timeout_failure": dict(failure),
                    "last_verifier_timeout_archive": harbor_relative,
                    "updated_at": now,
                }
            )
            for key in ("claim", "outcome", "harbor_job_path", "retry_not_before"):
                queued_dispatch.pop(key, None)
            queued_state = {
                "schema": CELL_STATE_SCHEMA,
                "cell_id": cell.cell_id,
                "task_name": cell.task_name,
                "task_ref": cell.task_ref,
                "official_attempt_index": cell.official_attempt_index,
                "config_contract_hash": spec.contract_hash,
                "status": "queued",
                "execution_attempt": execution,
                "worker_exception_count": candidate["worker_exception_count"],
                "verifier_timeout_repair_count": repair_count,
                "last_verifier_timeout_execution_attempt": execution,
                "last_verifier_timeout_failure": dict(failure),
                "last_verifier_timeout_archive": harbor_relative,
                "updated_at": now,
            }
            _atomic_write_json(paths.dispatch, queued_dispatch)
            _atomic_write_json(paths.state, queued_state)
            # result.json is the sole terminal marker. Retire it last, after
            # the complete Harbor archive and queued metadata are durable.
            os.replace(paths.result, terminal_archive / "result.json")
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)
            receipt_path = terminal_archive / "requeue.json"
            _atomic_write_json(
                receipt_path,
                {
                    "schema": "terminal_bench_committed_verifier_timeout_requeue_v1",
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "verifier_timeout_repair_count": repair_count,
                    "failure": dict(failure),
                    "harbor_archive_path": harbor_relative,
                    "terminal_archive_path": str(
                        terminal_archive.relative_to(spec.paths.root)
                    ),
                    "official_attempt_consumed_by_repair": False,
                    "requeue_policy": "explicit_committed_verifier_timeout_v1",
                    "requeued_at": now,
                },
            )
            repaired.append(cell.cell_id)
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_verifier_timeout_requeued",
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "official_attempt_consumed_by_repair": False,
                    "receipt_path": str(receipt_path.relative_to(spec.paths.root)),
                    "recorded_at": now,
                },
            )
    finally:
        for descriptor in reversed(locked_descriptors):
            _close_lock(descriptor)
    pool_state = aggregate_pool(spec)
    return {"repaired_cell_ids": repaired, "skipped_cell_ids": [], **pool_state}


def requeue_committed_external_interruption(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    cell_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Archive and requeue committed Harbor SIGTERM cancellations.

    This repair is intentionally narrower than a generic ``CancelledError``
    retry. Both the pool result and the pinned Harbor trial must contain the
    serialized Harbor SIGTERM traceback before the official cell is changed.
    """

    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    requested = set(cell_ids)
    known = {cell.cell_id for cell in spec.cells}
    unknown = requested - known
    if unknown:
        raise PoolError(f"unknown cell ids: {', '.join(sorted(unknown))}")
    repaired: list[str] = []
    skipped: list[str] = []
    for cell in spec.cells:
        if requested and cell.cell_id not in requested:
            continue
        paths = spec.cell_paths(cell).ensure()
        descriptor = _open_lock(paths.claim_lock, blocking=False)
        if descriptor is None:
            skipped.append(cell.cell_id)
            continue
        try:
            result = _cell_result(spec, cell)
            if result is None:
                skipped.append(cell.cell_id)
                continue
            failure = _external_interruption_trial_failure(result)
            if failure is None:
                skipped.append(cell.cell_id)
                continue
            execution = _valid_execution(result.get("execution_attempt"))
            if execution < 1:
                raise PoolError(
                    "committed external interruption has no execution: "
                    f"{cell.cell_id}"
                )
            state = _read_json(paths.state)
            dispatch = _read_json(paths.dispatch)
            if (
                state.get("status") != "completed"
                or state.get("execution_attempt") != execution
                or dispatch.get("status") != "terminal"
                or dispatch.get("execution_attempt") != execution
            ):
                raise PoolError(
                    f"committed interruption metadata is inconsistent: {cell.cell_id}"
                )
            claim = ClaimedCell(
                cell=cell,
                paths=paths,
                descriptor=descriptor,
                execution_attempt=execution,
                dispatch=dispatch,
            )
            terminal = _terminal_harbor_trial(spec, claim)
            if terminal is None:
                raise PoolError(
                    f"committed interruption has no Harbor trial: {cell.cell_id}"
                )
            _, trial = terminal
            harbor_failure = _external_interruption_trial_failure(trial)
            if harbor_failure != failure:
                raise PoolError(
                    f"pool and Harbor interruption failures disagree: {cell.cell_id}"
                )
            attempt_dir = _ensure_dir(paths.attempts_dir / f"execution-{execution:04d}")
            terminal_archive = attempt_dir / "committed-terminal-external-interruption"
            if terminal_archive.exists() or terminal_archive.is_symlink():
                raise PoolError(
                    f"committed interruption archive already exists: {terminal_archive}"
                )
            _ensure_dir(terminal_archive)
            _atomic_write_json(terminal_archive / "state.json", state)
            _atomic_write_json(terminal_archive / "dispatch.json", dispatch)

            interruption = ExternalInterruption(signal.SIGTERM)
            harbor_archive = _archive_external_interruption_harbor_job(
                claim,
                interruption,
                archive_name="harbor-job-committed-external-interruption",
            )
            if harbor_archive is None:
                raise PoolError(
                    f"committed interruption Harbor archive is absent: {cell.cell_id}"
                )
            _record_interruption(
                spec,
                claim,
                interruption,
                archive_harbor_job=False,
                harbor_archive=harbor_archive,
            )
            # result.json is the terminal commit marker. Move it only after
            # queued state/dispatch and the stale Harbor job are durable.
            os.replace(paths.result, terminal_archive / "result.json")
            _fsync_directory(paths.root)
            _fsync_directory(terminal_archive)
            _atomic_write_json(
                terminal_archive / "requeue.json",
                {
                    "schema": "terminal_bench_committed_external_interruption_requeue_v1",
                    "cell_id": cell.cell_id,
                    "task_name": cell.task_name,
                    "official_attempt_index": cell.official_attempt_index,
                    "execution_attempt": execution,
                    "failure": failure,
                    "harbor_archive_path": str(
                        harbor_archive.relative_to(spec.paths.root)
                    ),
                    "requeued_at": _utc_now(),
                },
            )
            repaired.append(cell.cell_id)
            _best_effort_event(
                spec.paths.root,
                {
                    "event": "committed_external_interruption_requeued",
                    "cell_id": cell.cell_id,
                    "execution_attempt": execution,
                    "official_attempt_index": cell.official_attempt_index,
                    "recorded_at": _utc_now(),
                },
            )
        finally:
            _close_lock(descriptor)
    state = aggregate_pool(spec)
    return {"repaired_cell_ids": repaired, "skipped_cell_ids": skipped, **state}


def _record_interruption(
    spec: PoolSpec,
    claim: ClaimedCell,
    interruption: ExternalInterruption,
    *,
    archive_harbor_job: bool = True,
    harbor_archive: Path | None = None,
) -> Path | None:
    if archive_harbor_job:
        if harbor_archive is not None:
            raise PoolError("external interruption archive was specified twice")
        harbor_archive = _archive_external_interruption_harbor_job(claim, interruption)
    now = _utc_now()
    dispatch = dict(claim.dispatch)
    dispatch.update(
        {
            "status": "queued",
            "last_interruption_signal": interruption.signum,
            "last_interruption_execution_attempt": claim.execution_attempt,
            "updated_at": now,
        }
    )
    if harbor_archive is not None:
        dispatch["last_interruption_archive"] = str(
            harbor_archive.relative_to(spec.paths.root)
        )
    dispatch.pop("claim", None)
    dispatch.pop("outcome", None)
    dispatch.pop("harbor_job_path", None)
    _atomic_write_json(claim.paths.dispatch, dispatch)
    state: dict[str, Any] = {
        "schema": CELL_STATE_SCHEMA,
        "cell_id": claim.cell.cell_id,
        "task_name": claim.cell.task_name,
        "task_ref": claim.cell.task_ref,
        "official_attempt_index": claim.cell.official_attempt_index,
        "config_contract_hash": spec.contract_hash,
        "status": "queued",
        "execution_attempt": claim.execution_attempt,
        "worker_exception_count": int(
            claim.dispatch.get("worker_exception_count") or 0
        ),
        "last_interruption_signal": interruption.signum,
        "updated_at": now,
    }
    if harbor_archive is not None:
        state["last_interruption_archive"] = str(
            harbor_archive.relative_to(spec.paths.root)
        )
    _atomic_write_json(claim.paths.state, state)
    return harbor_archive


def _write_worker(
    path: Path,
    *,
    worker_id: str,
    device_label: str,
    instance_id: str,
    status: str,
    lease_seconds: float,
    detail: Mapping[str, Any] | None = None,
) -> None:
    now = _utc_now()
    _atomic_write_json(
        path,
        {
            "schema": WORKER_SCHEMA,
            "worker_id": worker_id,
            "instance_id": instance_id,
            "status": status,
            "device": device_label,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "heartbeat_at": now,
            "lease_expires_at": _future_time(lease_seconds),
            **dict(detail or {}),
        },
    )


def _default_worker_id() -> str:
    scheduler = next(
        (
            os.environ[name].strip()
            for name in ("TJOB_ID", "SLURM_JOB_ID", "JOB_ID", "K8S_JOB_NAME")
            if os.environ.get(name, "").strip()
        ),
        "interactive",
    )
    rank = next(
        (
            os.environ[name].strip()
            for name in ("RANK", "INDEX", "SLURM_PROCID")
            if os.environ.get(name, "").strip()
        ),
        "0",
    )
    return f"{scheduler}-{socket.gethostname()}-rank-{rank}"


def _default_device_label() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return f"cuda-{visible}" if visible else "platform-assigned"


def _run_worker_preflight(
    spec: PoolSpec,
    *,
    module: str,
    sglang_url: str,
    worker_id: str,
    environment_backend: str = "frozen",
) -> None:
    output_dir = _ensure_dir(spec.paths.root / "preflight")
    output = output_dir / _safe_worker_file(worker_id)
    command = [
        sys.executable,
        "-m",
        module,
        "--config",
        str(spec.paths.config),
        "--sglang-url",
        sglang_url,
        "--output",
        str(output),
    ]
    preflight_env = dict(os.environ)
    preflight_env["TERMINAL_BENCH_POOL_ENVIRONMENT_BACKEND"] = environment_backend
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=preflight_env,
        check=False,
    )
    if completed.returncode != 0:
        raise PoolError(f"worker preflight failed with status {completed.returncode}")
    _attest_execution_memory(spec, output)


def _attest_execution_memory(spec: PoolSpec, output: Path) -> None:
    """Attest the launch-time profile in addition to the historical pool ABI."""
    from evaluation_memory import (
        codex_r2eact_agent, requested_memory_profile, memory_profile_receipt,
        WORKING_MEMORY_TURNS, SERVING_EVICTION_POLICY, ACCUMULATION_TOKENS,
    )

    agent = spec.config["agents"][0]
    if not codex_r2eact_agent(agent):
        return
    profile = requested_memory_profile()
    if profile is None:
        return
    from terminal_bench_eval import tb21_codex_r2eact_preflight as codex

    config = JobConfig.model_validate(spec.config)
    serving = codex.v053._serving_eviction_api_report(
        config, requested_policy=SERVING_EVICTION_POLICY,
        accumulation_target_tokens=ACCUMULATION_TOKENS,
    )
    kwargs = agent.get("kwargs", {})
    gateway = codex._gateway_report(
        working_memory_turns=WORKING_MEMORY_TURNS,
        serving_eviction_policy=SERVING_EVICTION_POLICY,
        serving_eviction_block_tokens=None,
        serving_eviction_accumulation_tokens=ACCUMULATION_TOKENS,
        **{key: kwargs.get(key, False) for key in
           ("vision_enabled", "web_search_enabled", "update_plan_enabled")},
    )
    _atomic_write_json(output.with_suffix(".memory.json"), {
        **memory_profile_receipt(profile), "serving": serving, "gateway": gateway,
        "base_config_sha256": _digest(spec.config),
    })


def _ordered_worker_cells(
    worker_cells: Sequence[CellSpec],
    *,
    worker_id: str,
    priority_task_names: Sequence[str] = (),
) -> tuple[CellSpec, ...]:
    """Rotate within priority tiers without changing legacy no-priority order."""

    cells = tuple(worker_cells)
    if not cells:
        return ()
    priority_names = tuple(priority_task_names)
    if not priority_names:
        offset = int(hashlib.sha256(worker_id.encode()).hexdigest(), 16) % len(cells)
        return cells[offset:] + cells[:offset]

    priority_set = frozenset(priority_names)
    priority_cells = tuple(cell for cell in cells if cell.task_name in priority_set)
    remaining_cells = tuple(
        cell for cell in cells if cell.task_name not in priority_set
    )

    def rotate(bucket: tuple[CellSpec, ...]) -> tuple[CellSpec, ...]:
        if not bucket:
            return ()
        offset = int(hashlib.sha256(worker_id.encode()).hexdigest(), 16) % len(bucket)
        return bucket[offset:] + bucket[:offset]

    return rotate(priority_cells) + rotate(remaining_cells)


def run_worker(
    *,
    job_name: str,
    jobs_root: Path = DEFAULT_JOBS_ROOT,
    worker_id: str | None = None,
    device_label: str | None = None,
    harbor_bin: Path = DEFAULT_HARBOR,
    sglang_url: str = "http://127.0.0.1:30000",
    preflight_module: str = "terminal_bench_eval.tb20_v050_preflight",
    skip_preflight: bool = False,
    skip_task_prewarm: bool = False,
    heartbeat_seconds: float = 15.0,
    lease_seconds: float = 90.0,
    poll_seconds: float = 2.0,
    idle_exit_seconds: float = 120.0,
    retry_backoff_seconds: float = 15.0,
    max_worker_exceptions: int = 3,
    max_proot_infrastructure_retries: int = 2,
    exclude_task_names: Sequence[str] = (),
    priority_task_names: Sequence[str] = (),
    environment_backend: str = "frozen",
    once: bool = False,
    execute_cell: Callable[[PoolSpec, ClaimedCell], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Claim cells until completion/idle exit; safe for many shared workers."""

    if heartbeat_seconds <= 0 or lease_seconds <= heartbeat_seconds * 2:
        raise PoolError("lease must be more than twice the positive heartbeat interval")
    if poll_seconds <= 0 or idle_exit_seconds < 0 or retry_backoff_seconds < 0:
        raise PoolError("poll must be positive and idle/backoff must be nonnegative")
    if max_worker_exceptions < 0:
        raise PoolError("max worker exceptions must be nonnegative")
    if max_proot_infrastructure_retries < 0:
        raise PoolError("max PRoot infrastructure retries must be nonnegative")
    if environment_backend not in {"frozen", "docker", "docker-gpu"}:
        raise PoolError(
            "worker environment backend must be one of: " "frozen, docker, docker-gpu"
        )
    if environment_backend != "frozen" and skip_preflight:
        raise PoolError("Docker workers may not bypass the claim-before preflight")
    shared_route = r"(?:/groups/[A-Za-z0-9_.:-]{1,128})?(?:/backends/[A-Za-z0-9_.:-]{1,128})?"
    if not re.fullmatch(r"https?://[^\s/?#@]+" + shared_route, sglang_url.rstrip("/")):
        raise PoolError("SGLang URL must be a credential-free HTTP(S) origin or shared gateway route")
    harbor = Path(harbor_bin).expanduser().resolve()
    if execute_cell is None and (
        not harbor.is_file() or not os.access(harbor, os.X_OK)
    ):
        raise PoolError(f"Harbor executable does not exist: {harbor}")
    spec = load_pool_spec(job_name=job_name, jobs_root=jobs_root)
    _validate_science_execution_backend(spec, environment_backend)
    if (
        any(cell.source == "terminal-bench-science/terminal-bench-science" for cell in spec.cells)
        and preflight_module != "terminal_bench_eval.science_preflight"
    ):
        raise PoolError("TB-Science requires its Docker capability preflight before any task claim")
    excluded_tasks = tuple(exclude_task_names)
    if len(set(excluded_tasks)) != len(excluded_tasks):
        raise PoolError("worker task exclusions must be unique")
    known_task_names = {cell.task_name for cell in spec.cells}
    unknown_exclusions = sorted(set(excluded_tasks) - known_task_names)
    if unknown_exclusions:
        raise PoolError(
            "worker task exclusions are not in this pool: "
            + ", ".join(unknown_exclusions)
        )
    configured_priority_tasks = _claim_priority_task_names(spec.marker, spec.cells)
    requested_priority_tasks = tuple(priority_task_names)
    if len(set(requested_priority_tasks)) != len(requested_priority_tasks):
        raise PoolError("worker priority task names must be unique")
    unknown_priorities = sorted(set(requested_priority_tasks) - known_task_names)
    if unknown_priorities:
        raise PoolError(
            "worker priority task names are not in this pool: "
            + ", ".join(unknown_priorities)
        )
    if (
        configured_priority_tasks
        and requested_priority_tasks
        and requested_priority_tasks != configured_priority_tasks
    ):
        raise PoolError("worker priority task names differ from the pool contract")
    priority_tasks = configured_priority_tasks or requested_priority_tasks
    priority_contract = spec.marker.get("claim_priority")
    excluded_task_set = frozenset(excluded_tasks)
    worker_cells = tuple(
        cell for cell in spec.cells if cell.task_name not in excluded_task_set
    )
    if not worker_cells:
        raise PoolError("worker task exclusions remove every cell in the pool")
    worker = worker_id or _default_worker_id()
    device = device_label or _default_device_label()
    if not worker.strip() or not device.strip() or "\n" in worker or "\n" in device:
        raise PoolError(
            "worker id and device label must be nonempty single-line values"
        )
    instance_id = uuid.uuid4().hex
    worker_path = spec.paths.workers_dir / _safe_worker_file(worker)
    _write_worker(
        worker_path,
        worker_id=worker,
        device_label=device,
        instance_id=instance_id,
        status="starting",
        lease_seconds=lease_seconds,
        detail={
            "environment_backend": environment_backend,
            "claim_priority_policy": (
                WORKER_CLAIM_PRIORITY_POLICY if priority_tasks else None
            ),
            "priority_task_count": len(priority_tasks),
            "claim_priority_contract_sha256": (
                priority_contract.get("contract_sha256")
                if isinstance(priority_contract, Mapping)
                else None
            ),
        },
    )
    if not skip_preflight:
        _run_worker_preflight(
            spec,
            module=preflight_module,
            sglang_url=sglang_url,
            worker_id=worker,
            environment_backend=environment_backend,
        )
    ordered = _ordered_worker_cells(
        worker_cells,
        worker_id=worker,
        priority_task_names=priority_tasks,
    )
    idle_started: float | None = None
    processed = 0
    network_circuit = NetworkCircuit()
    network_heartbeat_at = 0.0

    def network_heartbeat(detail: Mapping[str, Any]) -> None:
        nonlocal network_heartbeat_at
        if time.monotonic() - network_heartbeat_at < heartbeat_seconds:
            return
        network_heartbeat_at = time.monotonic()
        _write_worker(
            worker_path, worker_id=worker, device_label=device,
            instance_id=instance_id, status="network_unavailable",
            lease_seconds=lease_seconds,
            detail={"processed_count": processed, **detail},
        )

    try:
        while True:
            pool_state = aggregate_pool(spec)
            if pool_state["terminal_count"] == pool_state["cell_count"]:
                _write_worker(
                    worker_path,
                    worker_id=worker,
                    device_label=device,
                    instance_id=instance_id,
                    status="completed",
                    lease_seconds=lease_seconds,
                    detail={"processed_count": processed},
                )
                return aggregate_pool(spec)
            if execute_cell is None and not _sglang_is_healthy(sglang_url):
                _write_worker(
                    worker_path,
                    worker_id=worker,
                    device_label=device,
                    instance_id=instance_id,
                    status="sglang_unavailable",
                    lease_seconds=lease_seconds,
                    detail={"processed_count": processed, "sglang_url": sglang_url},
                )
                raise SGLangUnavailable(
                    f"SGLang health check failed before claiming a cell: {sglang_url}"
                )
            claimed: ClaimedCell | None = None
            # A failed egress must not consume retries on unrelated cells.
            # The gate is shared by same-host slots, before any cell ownership.
            network_circuit.wait(network_heartbeat)
            for cell in ordered:
                claimed = _try_claim(
                    spec,
                    cell,
                    worker_id=worker,
                    device_label=device,
                    instance_id=instance_id,
                    lease_seconds=lease_seconds,
                    max_worker_exceptions=max_worker_exceptions,
                    max_proot_infrastructure_retries=max_proot_infrastructure_retries,
                )
                if claimed is not None:
                    break
            if claimed is None:
                now = time.monotonic()
                if idle_started is None:
                    idle_started = now
                _write_worker(
                    worker_path,
                    worker_id=worker,
                    device_label=device,
                    instance_id=instance_id,
                    status="idle",
                    lease_seconds=lease_seconds,
                    detail={"processed_count": processed},
                )
                if once or (
                    idle_exit_seconds > 0 and now - idle_started >= idle_exit_seconds
                ):
                    state = aggregate_pool(spec)
                    _write_worker(
                        worker_path,
                        worker_id=worker,
                        device_label=device,
                        instance_id=instance_id,
                        status="exited_idle",
                        lease_seconds=lease_seconds,
                        detail={"processed_count": processed},
                    )
                    return aggregate_pool(spec)
                time.sleep(poll_seconds)
                continue
            idle_started = None
            heartbeat = _Heartbeat(
                claim=claimed,
                worker_path=worker_path,
                worker_id=worker,
                device_label=device,
                instance_id=instance_id,
                interval_seconds=heartbeat_seconds,
                lease_seconds=lease_seconds,
                environment_backend=environment_backend,
            )
            infrastructure_failure: (
                RetryableSGLangTrialError
                | RetryablePRootTrialError
                | RetryableTaskCacheTransportError
                | None
            ) = None
            infrastructure_exhausted = False
            try:
                heartbeat.start()
                if execute_cell is None:
                    result = _execute_harbor(
                        spec,
                        claimed,
                        harbor_bin=harbor,
                        sglang_url=sglang_url.rstrip("/"),
                        worker_id=worker,
                        device_label=device,
                        prewarm_task_cache=not skip_task_prewarm,
                        environment_backend=environment_backend,
                    )
                else:
                    result = dict(execute_cell(spec, claimed))
                heartbeat.stop()
                _commit_cell_result(spec, claimed, result)
                processed += 1
                _best_effort_event(
                    spec.paths.root,
                    {
                        "event": "cell_committed",
                        "cell_id": claimed.cell.cell_id,
                        "task_name": claimed.cell.task_name,
                        "official_attempt_index": claimed.cell.official_attempt_index,
                        "execution_attempt": claimed.execution_attempt,
                        "worker_id": worker,
                        "recorded_at": _utc_now(),
                    },
                )
            except ExternalInterruption as exc:
                heartbeat.stop()
                harbor_archive = _record_interruption(spec, claimed, exc)
                _best_effort_event(
                    spec.paths.root,
                    {
                        "event": "cell_external_interruption_requeued",
                        "cell_id": claimed.cell.cell_id,
                        "execution_attempt": claimed.execution_attempt,
                        "official_attempt_index": (claimed.cell.official_attempt_index),
                        "signal": exc.signum,
                        "harbor_archive_path": (
                            str(harbor_archive.relative_to(spec.paths.root))
                            if harbor_archive is not None
                            else None
                        ),
                        "worker_id": worker,
                        "recorded_at": _utc_now(),
                    },
                )
                _write_worker(
                    worker_path,
                    worker_id=worker,
                    device_label=device,
                    instance_id=instance_id,
                    status="interrupted",
                    lease_seconds=lease_seconds,
                    detail={"signal": exc.signum, "processed_count": processed},
                )
                aggregate_pool(spec)
                raise
            except RetryableSGLangTrialError as exc:
                heartbeat.stop()
                infrastructure_failure = exc
                (
                    infrastructure_count,
                    strict_failure_count,
                    strict_requeue_count,
                    effective_max_retries,
                    exhausted,
                ) = _record_bounded_proot_infrastructure_retry(
                    spec,
                    claimed,
                    exc,
                    retry_backoff_seconds=retry_backoff_seconds,
                    max_proot_infrastructure_retries=(max_proot_infrastructure_retries),
                )
                infrastructure_exhausted = exhausted
                _best_effort_event(
                    spec.paths.root,
                    {
                        "event": (
                            "cell_sglang_infrastructure_retry_exhausted"
                            if exhausted
                            else "cell_sglang_infrastructure_requeued"
                        ),
                        "cell_id": claimed.cell.cell_id,
                        "execution_attempt": claimed.execution_attempt,
                        "infrastructure_retry_count": infrastructure_count,
                        "strict_infrastructure_failure_count": (strict_failure_count),
                        "strict_infrastructure_requeue_count": strict_requeue_count,
                        "max_strict_infrastructure_retries": effective_max_retries,
                        "exception_type": exc.failure.get("exception_type"),
                        "worker_id": worker,
                        "recorded_at": _utc_now(),
                    },
                )
            except RetryablePRootTrialError as exc:
                heartbeat.stop()
                infrastructure_failure = exc
                recovery_target = image_target(exc.failure)
                if recovery_target is not None:
                    network_circuit.trip(recovery_target)
                (
                    infrastructure_count,
                    proot_failure_count,
                    proot_requeue_count,
                    effective_max_proot_retries,
                    exhausted,
                ) = _record_bounded_proot_infrastructure_retry(
                    spec,
                    claimed,
                    exc,
                    retry_backoff_seconds=retry_backoff_seconds,
                    max_proot_infrastructure_retries=(max_proot_infrastructure_retries),
                )
                infrastructure_exhausted = exhausted
                _best_effort_event(
                    spec.paths.root,
                    {
                        "event": (
                            "cell_proot_infrastructure_retry_exhausted"
                            if exhausted
                            else "cell_proot_infrastructure_requeued"
                        ),
                        "cell_id": claimed.cell.cell_id,
                        "execution_attempt": claimed.execution_attempt,
                        "official_attempt_index": (claimed.cell.official_attempt_index),
                        "infrastructure_retry_count": infrastructure_count,
                        "proot_infrastructure_failure_count": proot_failure_count,
                        "proot_infrastructure_requeue_count": proot_requeue_count,
                        "max_proot_infrastructure_retries": (
                            effective_max_proot_retries
                        ),
                        "requested_max_proot_infrastructure_retries": (
                            max_proot_infrastructure_retries
                        ),
                        "worker_exception_count": int(
                            claimed.dispatch.get("worker_exception_count") or 0
                        ),
                        "failure": dict(exc.failure),
                        "harbor_archive_path": str(
                            exc.archive_path.relative_to(spec.paths.root)
                        ),
                        "worker_id": worker,
                        "recorded_at": _utc_now(),
                    },
                )
            except RetryableTaskCacheTransportError as exc:
                heartbeat.stop()
                infrastructure_failure = exc
                network_circuit.trip({"kind": "task", "config": dict(claimed.cell.task_config)})
                count, retry_at = _record_task_cache_transport_retry(
                    spec,
                    claimed,
                    exc,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                _best_effort_event(
                    spec.paths.root,
                    {
                        "event": "cell_task_cache_transport_requeued",
                        "cell_id": claimed.cell.cell_id,
                        "task_name": claimed.cell.task_name,
                        "official_attempt_index": (claimed.cell.official_attempt_index),
                        "execution_attempt": claimed.execution_attempt,
                        "task_cache_transport_retry_count": count,
                        "retry_not_before": retry_at,
                        "failure": dict(exc.failure),
                        "worker_id": worker,
                        "recorded_at": _utc_now(),
                    },
                )
            except Exception as exc:
                heartbeat.stop()
                count = _record_worker_exception(
                    spec,
                    claimed,
                    exc,
                    retry_backoff_seconds=retry_backoff_seconds,
                    max_worker_exceptions=max_worker_exceptions,
                )
                _best_effort_event(
                    spec.paths.root,
                    {
                        "event": "cell_worker_exception",
                        "cell_id": claimed.cell.cell_id,
                        "execution_attempt": claimed.execution_attempt,
                        "worker_exception_count": count,
                        "exception_type": type(exc).__name__,
                        "worker_id": worker,
                        "recorded_at": _utc_now(),
                    },
                )
            finally:
                _close_lock(claimed.descriptor)
            state = aggregate_pool(spec)
            if infrastructure_failure is not None:
                if (
                    isinstance(infrastructure_failure, RetryableSGLangTrialError)
                    and not infrastructure_exhausted
                    and not _sglang_is_healthy(sglang_url)
                ):
                    _write_worker(
                        worker_path,
                        worker_id=worker,
                        device_label=device,
                        instance_id=instance_id,
                        status="sglang_unavailable",
                        lease_seconds=lease_seconds,
                        detail={
                            "processed_count": processed,
                            "sglang_url": sglang_url,
                            "cell_id": claimed.cell.cell_id,
                        },
                    )
                    raise SGLangUnavailable(
                        "SGLang became unavailable while executing "
                        f"{claimed.cell.cell_id}"
                    )
                if once:
                    _write_worker(
                        worker_path,
                        worker_id=worker,
                        device_label=device,
                        instance_id=instance_id,
                        status="exited_once",
                        lease_seconds=lease_seconds,
                        detail={"processed_count": processed},
                    )
                    return aggregate_pool(spec)
                if retry_backoff_seconds > 0:
                    time.sleep(retry_backoff_seconds)
                continue
            if once:
                _write_worker(
                    worker_path,
                    worker_id=worker,
                    device_label=device,
                    instance_id=instance_id,
                    status="exited_once",
                    lease_seconds=lease_seconds,
                    detail={"processed_count": processed},
                )
                return aggregate_pool(spec)
    except ExternalInterruption:
        raise
    except SGLangUnavailable:
        raise
    except BaseException:
        _write_worker(
            worker_path,
            worker_id=worker,
            device_label=device,
            instance_id=instance_id,
            status="aborted",
            lease_seconds=lease_seconds,
            detail={"processed_count": processed},
        )
        raise


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="initialize an immutable pool")
    init.add_argument("--job-name", required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    selection = init.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", dest="all_tasks")
    selection.add_argument("--task-name", action="append", default=[])
    selection.add_argument("--num-tasks", type=int)
    init.add_argument("--attempts", type=int, choices=(1, 5))
    init.add_argument("--priority-task-name", action="append", default=[])
    init.add_argument("--priority-source-manifest", type=Path)
    init.add_argument("--priority-source-manifest-sha256")
    init.add_argument("--priority-source-job")

    worker = subparsers.add_parser("worker", help="run one distributed worker")
    worker.add_argument("--job-name", required=True)
    worker.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    worker.add_argument("--worker-id")
    worker.add_argument("--device-label")
    worker.add_argument("--harbor-bin", type=Path, default=DEFAULT_HARBOR)
    worker.add_argument(
        "--sglang-url",
        default=os.environ.get(
            "TERMINAL_BENCH_POOL_SGLANG_URL", "http://127.0.0.1:30000"
        ),
    )
    worker.add_argument(
        "--preflight-module", default="terminal_bench_eval.tb20_v050_preflight"
    )
    worker.add_argument("--skip-preflight", action="store_true")
    worker.add_argument("--skip-task-prewarm", action="store_true")
    worker.add_argument("--heartbeat-seconds", type=float, default=15.0)
    worker.add_argument("--lease-seconds", type=float, default=90.0)
    worker.add_argument("--poll-seconds", type=float, default=2.0)
    worker.add_argument("--idle-exit-seconds", type=float, default=120.0)
    worker.add_argument("--retry-backoff-seconds", type=float, default=15.0)
    worker.add_argument("--max-worker-exceptions", type=int, default=3)
    worker.add_argument(
        "--max-proot-infrastructure-retries",
        type=int,
        default=2,
        help=(
            "maximum automatic PRoot RPC requeues per official cell "
            "(default: 2; a third matching failure becomes a pool error)"
        ),
    )
    worker.add_argument(
        "--exclude-task-name",
        action="append",
        default=[],
        help=(
            "task name this worker backend must not claim; repeat for each "
            "backend-incompatible task (the cells remain queued in the pool)"
        ),
    )
    worker.add_argument(
        "--priority-task-name",
        action="append",
        default=[],
        help=(
            "task name this worker should attempt before the remaining pool; "
            "repeat to define one priority tier"
        ),
    )
    worker.add_argument(
        "--environment-backend",
        choices=("frozen", "docker", "docker-gpu"),
        default="frozen",
        help=(
            "execution provider for cells claimed by this worker; 'frozen' "
            "uses the pool config, while Docker profiles replace only the "
            "environment provider and retain the immutable task/model contract"
        ),
    )
    worker.add_argument("--once", action="store_true")

    status = subparsers.add_parser("status", help="show compact pool state")
    status.add_argument("--job-name", required=True)
    status.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    status.add_argument("--refresh", action="store_true")
    retire_attempts = subparsers.add_parser(
        "retire-queued-attempts",
        help=(
            "exclude attempts 2-5 and terminally retire only their unlocked "
            "nonterminal queue cells"
        ),
    )
    retire_attempts.add_argument("--job-name", required=True)
    retire_attempts.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    retire_attempts.add_argument(
        "--keep-attempt",
        type=int,
        default=1,
        help="official attempt to retain (currently must be 1)",
    )
    retire_tasks = subparsers.add_parser(
        "retire-queued-tasks",
        help=(
            "exclude named tasks and terminally retire only their unlocked "
            "nonterminal queue cells"
        ),
    )
    retire_tasks.add_argument("--job-name", required=True)
    retire_tasks.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    retire_tasks.add_argument("--task-name", action="append", required=True)
    retire_tasks.add_argument(
        "--reason",
        required=True,
        help="single-line auditable reason for excluding these tasks",
    )
    repair = subparsers.add_parser(
        "repair-sglang-infrastructure",
        help="archive and requeue committed SGLang transport failures",
    )
    repair.add_argument("--job-name", required=True)
    repair.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    repair.add_argument("--cell-id", action="append", default=[])
    repair_proot = subparsers.add_parser(
        "repair-proot-infrastructure",
        help=(
            "archive and requeue an explicit committed local PRoot "
            "infrastructure failure"
        ),
    )
    repair_proot.add_argument("--job-name", required=True)
    repair_proot.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    repair_proot.add_argument("--cell-id", action="append", required=True)
    repair_proot_localhost = subparsers.add_parser(
        "repair-proot-localhost",
        help=(
            "archive and requeue explicitly named cells from the diagnosed "
            "2026-08-31 legacy PRoot localhost/background incident"
        ),
    )
    repair_proot_localhost.add_argument("--job-name", required=True)
    repair_proot_localhost.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    repair_proot_localhost.add_argument("--cell-id", action="append", required=True)
    repair_codex_host_rg = subparsers.add_parser(
        "repair-codex-host-rg",
        help=(
            "archive and requeue explicitly named cells from the diagnosed "
            "2026-08-28 missing-host-rg incident"
        ),
    )
    repair_codex_host_rg.add_argument("--job-name", required=True)
    repair_codex_host_rg.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    repair_codex_host_rg.add_argument("--cell-id", action="append", required=True)
    repair_codex_gateway_timeout = subparsers.add_parser(
        "repair-codex-gateway-timeout",
        help=(
            "archive and requeue explicitly named cells from the hash-pinned "
            "2026-08-31 recursive-memory gateway-timeout incident"
        ),
    )
    repair_codex_gateway_timeout.add_argument("--job-name", required=True)
    repair_codex_gateway_timeout.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    repair_codex_gateway_timeout.add_argument(
        "--cell-id", action="append", required=True
    )
    repair_codex_gateway_request_timeout = subparsers.add_parser(
        "repair-codex-gateway-request-timeout",
        help=(
            "archive and requeue explicitly named Codex cells with a typed "
            "timeout in the private gateway controller journal"
        ),
    )
    repair_codex_gateway_request_timeout.add_argument("--job-name", required=True)
    repair_codex_gateway_request_timeout.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    repair_codex_gateway_request_timeout.add_argument(
        "--cell-id", action="append", required=True
    )
    repair_codex_local_storage = subparsers.add_parser(
        "repair-codex-local-storage",
        help=(
            "archive and requeue explicitly named Codex/PRoot cells with "
            "exact host disk-quota or legacy 32 MiB memory-mirror evidence"
        ),
    )
    repair_codex_local_storage.add_argument("--job-name", required=True)
    repair_codex_local_storage.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    repair_codex_local_storage.add_argument(
        "--cell-id", action="append", required=True
    )
    repair_harbor_resume_config = subparsers.add_parser(
        "repair-harbor-resume-config",
        help=(
            "snapshot and requeue explicitly named false Harbor "
            "resume-config pool errors"
        ),
    )
    repair_harbor_resume_config.add_argument("--job-name", required=True)
    repair_harbor_resume_config.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    repair_harbor_resume_config.add_argument(
        "--cell-id", action="append", required=True
    )
    repair_timeout = subparsers.add_parser(
        "repair-agent-timeout",
        help="archive and requeue committed retired Harbor agent timeouts",
    )
    repair_timeout.add_argument("--job-name", required=True)
    repair_timeout.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    repair_timeout.add_argument("--cell-id", action="append", default=[])
    repair_verifier_timeout = subparsers.add_parser(
        "repair-verifier-timeout",
        help="archive and requeue explicitly named committed verifier timeouts",
    )
    repair_verifier_timeout.add_argument("--job-name", required=True)
    repair_verifier_timeout.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    repair_verifier_timeout.add_argument("--cell-id", action="append", required=True)
    commit_verifier_replay = subparsers.add_parser(
        "commit-verifier-replay",
        help="commit a strict batch of verifier-only replay rewards",
    )
    commit_verifier_replay.add_argument("--job-name", required=True)
    commit_verifier_replay.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    commit_verifier_replay.add_argument("--manifest", type=Path, required=True)
    repair_interruption = subparsers.add_parser(
        "repair-external-interruption",
        help="archive and requeue committed Harbor SIGTERM cancellations",
    )
    repair_interruption.add_argument("--job-name", required=True)
    repair_interruption.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    repair_interruption.add_argument("--cell-id", action="append", default=[])
    reconcile_agent_sigterm = subparsers.add_parser(
        "reconcile-verified-agent-sigterm",
        help="reattribute one explicit exit-143 cell with a complete verifier pass",
    )
    reconcile_agent_sigterm.add_argument("--job-name", required=True)
    reconcile_agent_sigterm.add_argument(
        "--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT
    )
    reconcile_agent_sigterm.add_argument("--cell-id", required=True)
    reserve_external = subparsers.add_parser(
        "reserve-external-trial",
        help="reserve a cell for a terminal trial completing in a direct Harbor job",
    )
    reserve_external.add_argument("--job-name", required=True)
    reserve_external.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    reserve_external.add_argument("--task-name", required=True)
    reserve_external.add_argument("--official-attempt-index", type=int, default=1)
    reserve_external.add_argument("--source-trial-dir", type=Path, required=True)
    reserve_external.add_argument("--reservation-id")
    reserve_controller = subparsers.add_parser(
        "reserve-external-controller-cell",
        help="hold a cell that an older direct Harbor controller may still start",
    )
    reserve_controller.add_argument("--job-name", required=True)
    reserve_controller.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    reserve_controller.add_argument("--task-name", required=True)
    reserve_controller.add_argument("--official-attempt-index", type=int, default=1)
    reserve_controller.add_argument("--source-job-dir", type=Path, required=True)
    reserve_controller.add_argument("--reservation-id")
    validate_external = subparsers.add_parser(
        "validate-external-trial",
        help="validate but do not import one terminal direct Harbor result",
    )
    validate_external.add_argument("--job-name", required=True)
    validate_external.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    validate_external.add_argument("--task-name", required=True)
    validate_external.add_argument("--official-attempt-index", type=int, default=1)
    validate_external.add_argument("--source-trial-dir", type=Path, required=True)
    import_external = subparsers.add_parser(
        "import-external-trial",
        help="strictly import one terminal direct Harbor result into its pool cell",
    )
    import_external.add_argument("--job-name", required=True)
    import_external.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    import_external.add_argument("--task-name", required=True)
    import_external.add_argument("--official-attempt-index", type=int, default=1)
    import_external.add_argument("--source-trial-dir", type=Path, required=True)
    import_external.add_argument("--reservation-id")
    import_external.add_argument("--allowed-reward", type=float, action="append")
    import_external.add_argument(
        "--allow-known-infrastructure-error", action="store_true"
    )
    release_external = subparsers.add_parser(
        "release-external-reservation",
        help="return a guarded external cell to the fresh worker queue",
    )
    release_external.add_argument("--job-name", required=True)
    release_external.add_argument("--jobs-root", type=Path, default=DEFAULT_JOBS_ROOT)
    release_external.add_argument("--task-name", required=True)
    release_external.add_argument("--official-attempt-index", type=int, default=1)
    release_external.add_argument("--reservation-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            path = initialize_pool(
                job_name=args.job_name,
                config_path=args.config,
                jobs_root=args.jobs_root,
                all_tasks=args.all_tasks,
                task_names=args.task_name,
                num_tasks=args.num_tasks,
                attempts=args.attempts,
                priority_task_names=args.priority_task_name,
                priority_source_manifest=args.priority_source_manifest,
                priority_source_manifest_sha256=(args.priority_source_manifest_sha256),
                priority_source_job_name=args.priority_source_job,
            )
            spec = load_pool_spec(job_name=args.job_name, jobs_root=args.jobs_root)
            _print_json({"pool_path": str(path), **aggregate_pool(spec)})
            return 0
        if args.command == "worker":
            state = run_worker(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                worker_id=args.worker_id,
                device_label=args.device_label,
                harbor_bin=args.harbor_bin,
                sglang_url=args.sglang_url,
                preflight_module=args.preflight_module,
                skip_preflight=args.skip_preflight,
                skip_task_prewarm=args.skip_task_prewarm,
                heartbeat_seconds=args.heartbeat_seconds,
                lease_seconds=args.lease_seconds,
                poll_seconds=args.poll_seconds,
                idle_exit_seconds=args.idle_exit_seconds,
                retry_backoff_seconds=args.retry_backoff_seconds,
                max_worker_exceptions=args.max_worker_exceptions,
                max_proot_infrastructure_retries=(
                    args.max_proot_infrastructure_retries
                ),
                exclude_task_names=args.exclude_task_name,
                priority_task_names=args.priority_task_name,
                environment_backend=args.environment_backend,
                once=args.once,
            )
            _print_json(state)
            return 0
        if args.command == "retire-queued-attempts":
            state = retire_queued_attempts(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                keep_official_attempt_index=args.keep_attempt,
            )
            _print_json(state)
            return 0
        if args.command == "retire-queued-tasks":
            state = retire_queued_tasks(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                task_names=args.task_name,
                reason=args.reason,
            )
            _print_json(state)
            return 0
        if args.command == "repair-sglang-infrastructure":
            state = requeue_committed_sglang_infrastructure(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-proot-infrastructure":
            state = requeue_committed_proot_infrastructure(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-proot-localhost":
            state = requeue_committed_proot_localhost_incident(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-codex-host-rg":
            state = requeue_committed_codex_host_rg_incident(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-codex-gateway-timeout":
            state = requeue_committed_codex_gateway_timeout_incident(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-codex-gateway-request-timeout":
            state = requeue_committed_codex_gateway_request_timeout(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-codex-local-storage":
            state = requeue_committed_codex_local_storage_infrastructure(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-harbor-resume-config":
            state = requeue_committed_harbor_resume_config_incident(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-agent-timeout":
            state = requeue_committed_agent_timeout(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "repair-verifier-timeout":
            state = requeue_committed_verifier_timeout(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "commit-verifier-replay":
            state = commit_verifier_replay_results(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                manifest_path=args.manifest,
            )
            _print_json(state)
            return 0
        if args.command == "repair-external-interruption":
            state = requeue_committed_external_interruption(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_ids=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "reconcile-verified-agent-sigterm":
            state = reconcile_committed_verified_agent_sigterm(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                cell_id=args.cell_id,
            )
            _print_json(state)
            return 0
        if args.command == "reserve-external-trial":
            reservation = reserve_external_harbor_trial(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                task_name=args.task_name,
                official_attempt_index=args.official_attempt_index,
                source_trial_dir=args.source_trial_dir,
                reservation_id=args.reservation_id,
            )
            _print_json(reservation)
            return 0
        if args.command == "reserve-external-controller-cell":
            reservation = reserve_external_controller_cell(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                task_name=args.task_name,
                official_attempt_index=args.official_attempt_index,
                source_job_dir=args.source_job_dir,
                reservation_id=args.reservation_id,
            )
            _print_json(reservation)
            return 0
        if args.command == "validate-external-trial":
            report = validate_external_harbor_trial(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                task_name=args.task_name,
                official_attempt_index=args.official_attempt_index,
                source_trial_dir=args.source_trial_dir,
            )
            _print_json(report)
            return 0
        if args.command == "import-external-trial":
            report = import_external_harbor_trial(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                task_name=args.task_name,
                official_attempt_index=args.official_attempt_index,
                source_trial_dir=args.source_trial_dir,
                reservation_id=args.reservation_id,
                allowed_rewards=args.allowed_reward,
                reject_known_infrastructure=(not args.allow_known_infrastructure_error),
            )
            _print_json(report)
            return 0
        if args.command == "release-external-reservation":
            report = release_external_harbor_trial_reservation(
                job_name=args.job_name,
                jobs_root=args.jobs_root,
                task_name=args.task_name,
                official_attempt_index=args.official_attempt_index,
                reservation_id=args.reservation_id,
            )
            _print_json(report)
            return 0
        spec = load_pool_spec(job_name=args.job_name, jobs_root=args.jobs_root)
        state = aggregate_pool(spec) if args.refresh else _read_json(spec.paths.state)
        _print_json(state)
        return 0
    except ExternalInterruption as exc:
        return 128 + int(exc.signum)
    except SGLangUnavailable as exc:
        print(f"terminal-bench SGLang unavailable: {exc}", file=sys.stderr)
        return 75
    except (PoolError, OSError, ValueError) as exc:
        print(f"terminal-bench pool error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_SELECTION_SCHEMA",
    "CLAIM_PRIORITY_SCHEMA",
    "CELL_EXTERNAL_RESERVATION_SCHEMA",
    "CELL_RESULT_SCHEMA",
    "EXTERNAL_TRIAL_IMPORT_RECEIPT_SCHEMA",
    "POOL_SCHEMA",
    "QUEUED_ATTEMPT_RETIREMENT_DIR_NAME",
    "QUEUED_ATTEMPT_RETIREMENT_POLICY",
    "QUEUED_ATTEMPT_RETIREMENT_SCHEMA",
    "TASK_SELECTION_SCHEMA",
    "WORKER_CLAIM_PRIORITY_POLICY",
    "QUEUED_TASK_RETIREMENT_DIR_NAME",
    "QUEUED_TASK_RETIREMENT_POLICY",
    "QUEUED_TASK_RETIREMENT_SCHEMA",
    "VERIFIER_REPLAY_MANIFEST_SCHEMA",
    "VERIFIER_REPLAY_RECEIPT_SCHEMA",
    "ClaimedCell",
    "ExternalInterruption",
    "PoolError",
    "RetryablePRootTrialError",
    "RetryableSGLangTrialError",
    "RetryableTaskCacheTransportError",
    "SGLangUnavailable",
    "aggregate_pool",
    "commit_verifier_replay_results",
    "initialize_pool",
    "import_external_harbor_trial",
    "load_pool_spec",
    "main",
    "requeue_committed_agent_timeout",
    "requeue_committed_codex_gateway_request_timeout",
    "requeue_committed_codex_gateway_timeout_incident",
    "requeue_committed_codex_host_rg_incident",
    "requeue_committed_codex_local_storage_infrastructure",
    "requeue_committed_external_interruption",
    "requeue_committed_harbor_resume_config_incident",
    "requeue_committed_proot_infrastructure",
    "requeue_committed_proot_localhost_incident",
    "requeue_committed_sglang_infrastructure",
    "requeue_committed_verifier_timeout",
    "release_external_harbor_trial_reservation",
    "reserve_external_controller_cell",
    "reserve_external_harbor_trial",
    "retire_queued_attempts",
    "retire_queued_tasks",
    "run_worker",
    "validate_external_harbor_trial",
]
