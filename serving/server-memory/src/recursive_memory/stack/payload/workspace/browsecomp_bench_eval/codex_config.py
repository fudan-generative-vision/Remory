"""BrowseComp provenance and unbounded task budget, sharing Codex runtime fields."""
from dataclasses import dataclass

from job_bench_eval.config import CONFIG_SCHEMA, JobBenchConfig


@dataclass(frozen=True)
class BrowseCompCodexConfig(JobBenchConfig):
    schema: str = "browsecomp100_codex_runtime_v1"
    timeout_seconds: float | None = None

    @property
    def expected_task_count(self):
        return 100

    def validate(self):
        if self.schema != "browsecomp100_codex_runtime_v1":
            raise ValueError("unexpected BrowseComp Codex runtime schema")
        # Reuse inference validation, but do not inherit JobBench's 24h deadline.
        runtime = {name: getattr(self, name) for name in JobBenchConfig.__dataclass_fields__}
        runtime["schema"] = CONFIG_SCHEMA
        runtime["timeout_seconds"] = 1 if self.timeout_seconds is None else self.timeout_seconds
        JobBenchConfig(**runtime).validate()
        return self

    def to_json(self):
        payload = JobBenchConfig.to_json(self)
        # Dataset and judge provenance live in the BrowseComp pool manifest.
        # Do not mislabel this run with unused JobBench revision/split metadata.
        for name in ("official_eval_repository", "official_eval_commit", "official_dataset_repository",
                     "official_dataset_revision", "split", "dataset_root", "runner_entrypoint"):
            payload.pop(name, None)
        return payload
