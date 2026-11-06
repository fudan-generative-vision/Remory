"""Private pre-verifier filesystem checkpoints and bounded preparation recovery.

Never retries assertions and never calls a model. A filesystem checkpoint is
not advertised as a VM/process checkpoint. Unknown live processes fail closed
for unattended restore, while same-environment preparation retries remain useful.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import signal
import tarfile
import threading
import time
import uuid
from datetime import UTC, datetime

SCHEMA = "tb_verifier_checkpoint_v1"
PENDING_SCHEMA = "tb_verification_pending_v1"
DIRECTORY = "verification-recovery"
REPLAY_AGENT = "terminal_bench_eval.verifier_checkpoint:CheckpointVerifierAgent"
POLICY = "tb21_pre_verifier_checkpoint_prepare_retry_v1"
_BASE_PROCESSES = {"bash", "sh", "dash", "tmux", "tmux: server", "proot",
                   "proot_rpc_server", "proot_rpc_serve", "proot_rpc_serve~"}
_STATEFUL_TASKS = {"mailman", "install-windows-3.11", "qemu-startup", "qemu-alpine-ssh"}


def now():
    return datetime.now(UTC).isoformat()


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as f:
            os.chmod(temporary, 0o600)
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def preparation_failure(log: bytes, *, report_exists=False):
    """Positive proof only: no report/test-start and a failed runtime bootstrap."""
    text = log.decode("utf-8", errors="replace")
    if report_exists or any(s in text for s in (
            "test session starts", "collected ", "short test summary info", "FAILED /tests/")):
        return None
    transport = any(s.lower() in text.lower() for s in (
        "502  Bad Gateway", "503 Service Unavailable", "Connection timed out",
        "Connection reset", "Could not resolve", "operation timed out",
        "temporary failure in name resolution", "network is unreachable"))
    missing = re.search(r"(?m)^/tests/test\.sh: line \d+: (?:uvx|uv|pytest): command not found$", text)
    download = re.search(
        r"(?m)^(?:error:|\s+Caused by:) Failed to download https://github\.com/astral-sh/"
        r"python-build-standalone/releases/download/[^\s]+/cpython-[^\s]+\.tar\.gz$", text)
    if transport and ((missing and ("E: Failed to fetch " in text or "curl:" in text))
                      or (download and "Caused by: error sending request" in text)):
        return {"cause": "verifier_runtime_preparation_failed", "log_sha256": hashlib.sha256(log).hexdigest()}
    return None


def checkpoint(environment, trial_dir: Path, config: dict, cancel=None):
    """Snapshot the host-owned PRoot rootfs before any official verifier mutation."""
    dest = trial_dir / DIRECTORY
    dest.mkdir(mode=0o700, exist_ok=True)
    marker = dest / "checkpoint.json"
    if marker.exists():
        return json.loads(marker.read_text())
    root = environment._rootfs
    if not isinstance(root, Path) or root.is_symlink() or not root.is_dir():
        raise RuntimeError("No safe PRoot rootfs to checkpoint")
    process = environment._verifier_descendant_process_snapshot(environment._rpc_identity)
    names = set(process.get("process_names") or [])
    infrastructure = set(_BASE_PROCESSES)
    proot_name = Path(environment._configured_proot_binary()).name.lower()
    infrastructure.update({proot_name, proot_name[:15]})  # Linux comm truncates at 15 bytes.
    unknown = names - infrastructure
    task = config["task"]
    task_name = str(task.get("name", "")).rsplit("/", 1)[-1]
    safe = process.get("complete") is True and not unknown and task_name not in _STATEFUL_TASKS
    record = {"schema": SCHEMA, "policy": POLICY, "created_at": now(),
              "task": task, "scope": "filesystem_only_not_process_or_vm_state",
              "auto_restore": safe, "unknown_process_count": len(unknown),
              "process_scan_complete": process.get("complete") is True,
              "original_host": os.uname().nodename, "original_rootfs": str(root),
              "image_manifest": environment._oci_manifest_digest,
              "bundle_config": json.loads((environment._bundle_dir / "config.json").read_text()),
              "trial_config": config, "status": "creating"}
    atomic_json(marker, record)
    temporary = dest / "rootfs.tar.gz.partial"
    archive = dest / "rootfs.tar.gz"
    try:
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Checkpoint creation cancelled")
        # No symlink dereferencing and no virtual filesystems. Runtime helpers
        # are reconstructed by the same pinned PRoot startup code on restore.
        with (dest / "snapshot.stderr.txt").open("xb") as log:
            os.chmod(log.name, 0o600)
            with temporary.open("xb") as output:
                os.chmod(temporary, 0o600)
                command = ["tar", "--sparse", "--xattrs", "--xattrs-include=*", "--acls", "--numeric-owner",
                           "--one-file-system", "--exclude=./proc", "--exclude=./dev",
                           "--exclude=./sys", "--exclude=./.harbor", "-I", "gzip -1",
                           "-cf", "-", "-C", str(root), "."]
                run = subprocess.Popen(command, stdout=output, stderr=log, start_new_session=True)
                deadline = time.monotonic() + 1800
                try:
                    while run.poll() is None:
                        if (cancel is not None and cancel.is_set()) or time.monotonic() > deadline:
                            raise RuntimeError("Checkpoint creation interrupted or timed out")
                        try: run.wait(timeout=0.5)
                        except subprocess.TimeoutExpired: pass
                finally:
                    if run.poll() is None:
                        os.killpg(run.pid, signal.SIGTERM)
                        try: run.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(run.pid, signal.SIGKILL); run.wait(timeout=5)
                output.flush(); os.fsync(output.fileno())
            if run.returncode != 0:
                raise RuntimeError(f"Rootfs snapshot was not consistent (tar exit {run.returncode})")
        os.replace(temporary, archive)
        record.update(status="ready", archive_sha256=digest(archive), archive_bytes=archive.stat().st_size)
    except Exception as exc:
        # Do not suppress actual verification just because backup failed.
        # The original rootfs is retained on a later preparation failure.
        record.update(status="unavailable", auto_restore=False, error=type(exc).__name__)
    atomic_json(marker, record)
    return record


def restore_checkpoint(marker: Path, expected_sha256: str, bundle: Path, task_name: str):
    """Restore a trusted, pinned archive into a NEW empty directory only."""
    marker = Path(marker)
    if marker.is_symlink() or digest(marker) != expected_sha256:
        raise RuntimeError("Verifier checkpoint metadata changed")
    record = json.loads(marker.read_text())
    if (record.get("schema") != SCHEMA or record.get("status") != "ready"
            or record.get("auto_restore") is not True or record["task"].get("name") != task_name):
        raise RuntimeError("Checkpoint requires manual/stateful restoration")
    archive = marker.parent / "rootfs.tar.gz"
    if archive.is_symlink() or digest(archive) != record["archive_sha256"]:
        raise RuntimeError("Verifier checkpoint archive changed")
    if bundle.exists():
        raise RuntimeError("Checkpoint restore requires a new empty bundle")
    bundle.mkdir(mode=0o700, parents=True)
    root = bundle / "rootfs"; root.mkdir()
    # Reject traversal and writes through symlink parents. Absolute symlink
    # targets are legitimate inside a rootfs, but are never followed by tar.
    with tarfile.open(archive, "r:gz") as f:
        members = f.getmembers()
    links = set()
    for m in members:
        p = Path(m.name)
        if (p.is_absolute() or ".." in p.parts or m.isdev() or m.isfifo()
                or (not p.parts and not m.isdir())):
            raise RuntimeError("Unsafe member in verifier checkpoint")
        if m.issym(): links.add(p)
        if m.islnk() and (Path(m.linkname).is_absolute() or ".." in Path(m.linkname).parts):
            raise RuntimeError("Unsafe hardlink in verifier checkpoint")
    for m in members:
        if any(parent in links for parent in Path(m.name).parents):
            raise RuntimeError("Checkpoint contains a write through a symlink")
        if m.islnk() and (Path(m.linkname) in links or any(parent in links for parent in Path(m.linkname).parents)):
            raise RuntimeError("Checkpoint contains a hardlink through a symlink")
    subprocess.run(["tar", "--sparse", "--xattrs", "--xattrs-include=*", "--acls", "--numeric-owner",
                    "-xpf", str(archive), "-C", str(root)], check=True, timeout=1800)
    atomic_json(bundle / "config.json", record["bundle_config"])
    return record


class VerifierPreparationPending(RuntimeError):
    """Agent output retained; no legitimate verifier score was produced."""


def pending_record(trial_dir):
    path = Path(trial_dir) / DIRECTORY / "pending.json"
    if not path.exists(): return None
    if path.is_symlink(): raise RuntimeError("Unsafe pending verifier record")
    value = json.loads(path.read_text())
    if value.get("schema") != PENDING_SCHEMA: raise RuntimeError("Invalid pending verifier record")
    return value if value.get("status") != "completed" else None


def install_verifier_checkpointing(trial_type=None):
    if trial_type is None:
        from harbor.trial.trial import Trial
        trial_type = Trial
    original = trial_type._run_shared_verifier
    if getattr(original, "_checkpoint_policy", None) == POLICY: return

    async def run(self, **kwargs):
        config = self.config.model_dump(mode="json")
        env = self.agent_environment
        if (config["task"].get("source") != "terminal-bench/terminal-bench-2-1"
                or config["agent"].get("import_path") == REPLAY_AGENT
                or not hasattr(env, "_rootfs")):
            return await original(self, **kwargs)
        trial_dir = Path(self.paths.trial_dir)
        directory = trial_dir / DIRECTORY
        keep_after_success = env._keep_containers
        # Any unexpected backup/logging/verification error must retain the
        # original filesystem, even when the shared disk cannot accept metadata.
        env._keep_containers = True
        cancelled = threading.Event()
        backup_task = asyncio.create_task(asyncio.to_thread(checkpoint, env, trial_dir, config, cancelled))
        try:
            backup = await asyncio.shield(backup_task)
        except asyncio.CancelledError:
            env._keep_containers = True
            cancelled.set()
            # Do not let teardown delete a rootfs while the archive thread reads it.
            while not backup_task.done():
                try: await asyncio.shield(backup_task)
                except asyncio.CancelledError: pass
            raise
        except Exception as exc:
            backup = {"status": "unavailable", "auto_restore": False, "error": type(exc).__name__}
            try: atomic_json(directory / "checkpoint-error.json", backup)
            except OSError: pass
        marker = directory / "checkpoint.json"
        attempts = []
        for index in range(3):
            error = None; value = None
            try:
                value = await original(self, **kwargs)
            except asyncio.CancelledError:
                env._keep_containers = True
                raise
            except Exception as exc:
                error = exc
            logs = Path(self.paths.verifier_dir)
            log_path = logs / "test-stdout.txt"
            log = log_path.read_bytes() if log_path.is_file() else b""
            failure = preparation_failure(log, report_exists=(logs / "ctrf.json").exists())
            if (failure is None and type(error).__name__ == "AddTestsDirError"
                    and not (logs / "ctrf.json").exists()):
                failure = {"cause": "official_tests_upload_failed", "log_sha256": hashlib.sha256(log).hexdigest()}
            if (failure is None and isinstance(error, asyncio.TimeoutError)
                    and not (logs / "ctrf.json").exists()):
                timed = getattr(env, "_last_verifier_timeout", None) or {}
                process = timed.get("process_snapshot") or {}
                if (any(r.get("retry_eligible") for r in getattr(env, "_verifier_timeout_records", []))
                        and process.get("complete") is True and process.get("network_processes")
                        and not process.get("test_processes")
                        and not any(s in log for s in (b"test session starts", b"collected "))):
                    failure = {"cause": "proven_preparation_timeout", "log_sha256": hashlib.sha256(log).hexdigest()}
            if failure is None:
                if error is not None:
                    env._keep_containers = True
                    raise error
                atomic_json(directory / "verification.json", {"status": "verified", "attempts": attempts,
                                                             "model_called": False, "finished_at": now()})
                env._keep_containers = keep_after_success
                return value
            attempt_dir = directory / f"preparation-{index + 1}"
            attempt_dir.mkdir(mode=0o700, exist_ok=True)
            for file in logs.iterdir():
                if file.is_file() and not file.is_symlink(): shutil.copy2(file, attempt_dir / file.name)
            attempts.append({"attempt": index + 1, "failure": failure, "finished_at": now()})
            atomic_json(directory / "preparation-attempts.json", attempts)
            if index < 2 and not getattr(env, "_rpc_poisoned", True):
                # The complete bootstrap shell has exited. Preserve live task
                # services; do not restart the RPC server for this retry.
                env._clear_verifier_retry_outputs(env._rootfs / "logs/verifier")
                env._clear_verifier_retry_outputs(logs)
                await asyncio.sleep(2 ** index)
                continue
            break
        env._keep_containers = True
        atomic_json(directory / "pending.json", {
            "schema": PENDING_SCHEMA, "policy": POLICY, "status": "pending",
            "created_at": now(), "task": config["task"], "model_called": False,
            "checkpoint_sha256": digest(marker) if marker.is_file() else None,
            "auto_restore": backup.get("status") == "ready" and backup.get("auto_restore") is True,
            "attempts": attempts, "agent_exception": self.result.exception_info.model_dump(mode="json")
            if self.result.exception_info is not None else None,
        })
        raise VerifierPreparationPending("Verifier preparation failed; final state retained for verifier-only recovery")

    run._checkpoint_policy = POLICY
    trial_type._run_shared_verifier = run


# Imported only when Harbor instantiates the model-free recovery agent.
from harbor.agents.base import BaseAgent


class CheckpointVerifierAgent(BaseAgent):
    @staticmethod
    def name(): return "checkpoint-verifier"
    def version(self): return "1.0.0"
    async def setup(self, environment): pass
    async def run(self, instruction, environment, context):
        receipt = getattr(environment, "_restored_verifier_checkpoint", None)
        if receipt is None: raise RuntimeError("Verifier replay did not restore its checkpoint")
        atomic_json(self.logs_dir / "checkpoint-replay.json", {
            "schema": "tb_checkpoint_replay_agent_v1", "model_called": False,
            "archive_sha256": receipt["archive_sha256"], "task": receipt["task"],
        })
