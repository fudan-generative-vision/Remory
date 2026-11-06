"""JobBench evaluation support for the local Codex/r²eAct harness.

The package deliberately separates public task inputs from private benchmark
metadata.  Pool workers only hand a copied ``task_folder`` workspace to a
runner; rubric and reference-answer material never enters the runner request.
"""

from .config import (
    DEFAULT_TASK_TIMEOUT_SECONDS,
    OFFICIAL_DATASET_REVISION,
    OFFICIAL_EVAL_COMMIT,
    JobBenchConfig,
)
from .dataset import JobBenchTask, discover_tasks, prepare_private_workspace

__all__ = [
    "DEFAULT_TASK_TIMEOUT_SECONDS",
    "OFFICIAL_DATASET_REVISION",
    "OFFICIAL_EVAL_COMMIT",
    "JobBenchConfig",
    "JobBenchTask",
    "discover_tasks",
    "prepare_private_workspace",
]
