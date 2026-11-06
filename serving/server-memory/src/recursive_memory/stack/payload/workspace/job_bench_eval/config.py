"""Immutable configuration contract for JobBench generation pools."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_ROOT = WORKSPACE_ROOT / "eval" / "job_bench"
DEFAULT_DATASET_ROOT = DEFAULT_RUNTIME_ROOT / "source" / "dataset"

# These are immutable experiment inputs rather than moving branch names.
OFFICIAL_EVAL_REPOSITORY = "https://github.com/Job-Bench/job-bench-eval.git"
OFFICIAL_EVAL_COMMIT = "f938276c815e0fb2281b2262262dbba52a9c3e0b"
OFFICIAL_DATASET_REPOSITORY = "JobBench/job-bench"
OFFICIAL_DATASET_REVISION = "7ff2673ac78f42e492b204f5fffe0b8ccc11bf7d"

SUPPORTED_SPLITS = frozenset({"main", "easy"})
EXPECTED_TASK_COUNTS = {"main": 65, "easy": 63}
DEFAULT_SPLIT = "main"
DEFAULT_TASK_TIMEOUT_SECONDS = 86_400.0
OFFICIAL_ATTEMPTS_PER_TASK = 1

CONFIG_SCHEMA = "jobbench_codex_pool_config_v1"
DEFAULT_RUNNER_ENTRYPOINT = "job_bench_eval.runner:run_jobbench_task"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class JobBenchConfig:
    """Secret-free configuration pinned into every shared pool.

    API keys and other credentials intentionally have no fields here.  A
    launcher injects them into the runner process at execution time, keeping
    pool metadata safe to inspect and synchronize.
    """

    schema: str = CONFIG_SCHEMA
    runtime_root: str = str(DEFAULT_RUNTIME_ROOT)
    dataset_root: str = str(DEFAULT_DATASET_ROOT)
    split: str = DEFAULT_SPLIT
    official_eval_repository: str = OFFICIAL_EVAL_REPOSITORY
    official_eval_commit: str = OFFICIAL_EVAL_COMMIT
    official_dataset_repository: str = OFFICIAL_DATASET_REPOSITORY
    official_dataset_revision: str = OFFICIAL_DATASET_REVISION
    timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS
    attempts_per_task: int = OFFICIAL_ATTEMPTS_PER_TASK
    runner_entrypoint: str = DEFAULT_RUNNER_ENTRYPOINT
    harness_name: str = "Codex CLI + r²eAct"
    harness_mode: str = "native_compact"
    working_memory_turns: int = 12
    serving_eviction_policy: str = "checkpoint_1024_v1"
    native_compaction_text_summary: bool = True
    compressor_enabled: bool = True
    residual_compaction: bool = True
    residual_checkpoint: str = ""

    def validate(self) -> "JobBenchConfig":
        from shared_runtime.config import HarnessConfig
        HarnessConfig(**{name: getattr(self, name) for name in HarnessConfig.__dataclass_fields__}).validate()
        if self.schema != CONFIG_SCHEMA:
            raise ValueError(f"unsupported JobBench config schema: {self.schema!r}")
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported JobBench split: {self.split!r}")
        fixed = (
            ("official_eval_repository", OFFICIAL_EVAL_REPOSITORY),
            ("official_eval_commit", OFFICIAL_EVAL_COMMIT),
            ("official_dataset_repository", OFFICIAL_DATASET_REPOSITORY),
            ("official_dataset_revision", OFFICIAL_DATASET_REVISION),
            ("attempts_per_task", OFFICIAL_ATTEMPTS_PER_TASK),
        )
        for name, expected in fixed:
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be pinned to {expected!r}")
        if isinstance(self.timeout_seconds, bool) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        for name in (
            "runtime_root",
            "dataset_root",
            "runner_entrypoint",
            "harness_name",
            "harness_mode",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not Path(self.runtime_root).expanduser().is_absolute():
            raise ValueError("runtime_root must be absolute")
        if not Path(self.dataset_root).expanduser().is_absolute():
            raise ValueError("dataset_root must be absolute")
        module, separator, attribute = self.runner_entrypoint.partition(":")
        if not separator or not module.strip() or not attribute.strip():
            raise ValueError("runner_entrypoint must have the form module:callable")
        return self

    @property
    def expected_task_count(self) -> int:
        return EXPECTED_TASK_COUNTS[self.split]

    @property
    def contract_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_json())).hexdigest()

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        # False must remain explicit: the default on deserialization is True.
        if not self.residual_checkpoint:
            payload.pop("residual_checkpoint")
        if self.compressor_enabled:
            payload.pop("compressor_enabled")  # Preserve all existing pool hashes.
        if not self.native_compaction_text_summary:
            # Preserve the exact hashes of already initialized v1 pools.
            payload.pop("native_compaction_text_summary")
        return payload

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "JobBenchConfig":
        if not isinstance(value, Mapping):
            raise TypeError("JobBench config must be an object")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError("unknown JobBench config fields: " + ", ".join(unknown))
        return cls(**dict(value)).validate()
