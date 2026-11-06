"""Download exact official revisions; never execute the upstream destructive setup.sh."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import DEFAULT_RUNTIME_ROOT, OFFICIAL_DATASET_REPOSITORY, OFFICIAL_DATASET_REVISION, OFFICIAL_EVAL_COMMIT, OFFICIAL_EVAL_REPOSITORY
from .dataset import discover_tasks, write_dataset_marker, validate_dataset_marker
from shared_runtime.storage import lock


def prepare(runtime_root: Path, split="main", offline=False):
    runtime_root = runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Private source and controller artifacts must not be visible to the agent UID.
    runtime_root.chmod(0o700)
    source = runtime_root / "source"
    repo = source / "official-eval"
    dataset = source / "dataset"
    with lock(runtime_root / "bootstrap.lock"):
        if not (repo / ".git").exists():
            if offline:
                raise RuntimeError(f"offline official evaluator is missing: {repo}")
            repo.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", OFFICIAL_EVAL_REPOSITORY, str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "checkout", "--detach", OFFICIAL_EVAL_COMMIT], check=True)
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != OFFICIAL_EVAL_COMMIT:
            raise RuntimeError("official evaluator revision mismatch; refusing to change it")
        if (dataset / split).is_dir():
            validate_dataset_marker(dataset)
        else:
            if offline:
                raise RuntimeError(f"offline dataset split is missing: {dataset / split}")
            from huggingface_hub import snapshot_download
            prefix = "dataset" if split == "main" else "dataset_easy"
            staging = source / "hf-download"
            print(f"Downloading pinned JobBench {split} to {staging}", flush=True)
            snapshot_download(
                repo_id=OFFICIAL_DATASET_REPOSITORY, repo_type="dataset",
                revision=OFFICIAL_DATASET_REVISION, allow_patterns=[f"{prefix}/**"],
                local_dir=staging, max_workers=4,
            )
            dataset.mkdir(exist_ok=True)
            os.replace(staging / prefix, dataset / split)
            write_dataset_marker(dataset)
        snapshot = discover_tasks(dataset, split=split)
    print(f"JobBench {split}: {len(snapshot.tasks)} tasks; source={snapshot.snapshot_sha256}", flush=True)
    return snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    p.add_argument("--split", choices=["main", "easy"], default="main")
    p.add_argument("--offline-dataset", action="store_true")
    a = p.parse_args()
    prepare(a.runtime_root, a.split, a.offline_dataset)


if __name__ == "__main__":
    main()
