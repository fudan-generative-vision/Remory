"""Install and attest the document boundary before starting expensive services."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

BWRAP_SHA = "52231e1caf55bcbc667b269f49c63599a6f7db4767ae6a039580d0ff853db712"
BWRAP_DEB_SHA = "1b506492bd9c7fd0cdb4f02ac822f1d3e336b0aead5113c1239baf8db5db562a"
BWRAP_URL = "https://archive.ubuntu.com/ubuntu/pool/main/b/bubblewrap/bubblewrap_0.9.0-1ubuntu0.1_amd64.deb"


def landlock_abi():
    return ctypes.CDLL(None, use_errno=True).syscall(444, 0, 0, 1)


def install(prefix, requested="auto"):
    from .cuda import download
    from .deploy import sha

    backend = (
        ("landlock" if landlock_abi() >= 1 else "bubblewrap")
        if requested == "auto"
        else requested
    )
    if backend == "landlock":
        if landlock_abi() < 1:
            raise RuntimeError(
                "Landlock is unavailable; use document_sandbox=auto or bubblewrap"
            )
        return {"backend": backend}
    binary = prefix / "bin/bwrap"
    if not binary.exists():
        local = shutil.which("bwrap")
        if local and sha(local) == BWRAP_SHA:
            shutil.copyfile(local, binary)
        else:
            archive = download(
                BWRAP_URL, BWRAP_DEB_SHA, prefix / "downloads/bubblewrap-0.9.0.deb"
            )
            with tempfile.TemporaryDirectory(prefix=".bwrap-", dir=prefix) as name:
                subprocess.run(["dpkg-deb", "-x", str(archive), name], check=True)
                source = Path(name) / "usr/bin/bwrap"
                if sha(source) != BWRAP_SHA:
                    raise RuntimeError("bubblewrap binary checksum mismatch")
                shutil.copyfile(source, binary)
        binary.chmod(0o755)
    if binary.is_symlink() or sha(binary) != BWRAP_SHA:
        raise RuntimeError("bubblewrap differs from the pinned binary")
    # Fail during installation on hosts lacking the binary's shared libraries.
    subprocess.run([str(binary), "--version"], check=True, capture_output=True)
    return {"backend": backend, "binary": str(binary), "sha256": BWRAP_SHA}


def check():
    from shared_runtime.runner import sandbox_command

    if os.environ.get("R2EACT_DOCUMENT_SANDBOX") == "bubblewrap":
        from shared_runtime.bubblewrap import boundary_probe

        return {"backend": "bubblewrap", **boundary_probe()}
    # Probe the real UID/Landlock boundary, rather than only querying its ABI.
    with tempfile.TemporaryDirectory(prefix="rm-landlock-check-") as name:
        root = Path(name)
        root.chmod(0o755)
        workspace = root / "workspace"
        workspace.mkdir(mode=0o700)
        outside = root / "outside"
        outside.write_text("unavailable canary")
        outside.chmod(0o644)
        uid = 60000 + os.getpid() % 5000 if os.geteuid() == 0 else os.getuid()
        if os.geteuid() == 0:
            os.chown(workspace, uid, uid)
        argv = sandbox_command(
            workspace,
            uid,
            Path("/bin/true"),
            Path("/usr"),
            [
                "/bin/sh",
                "-c",
                '! cat "$1" >/dev/null 2>&1 && touch probe',
                "probe",
                str(outside),
            ],
        )
        subprocess.run(argv, check=True, capture_output=True, timeout=30, cwd=workspace)
        if not (workspace / "probe").is_file():
            raise RuntimeError("document boundary probe did not complete")
    return {"backend": "landlock", "status": "ok"}
