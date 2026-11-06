"""Private, content-addressed snapshots of pixels already admitted by the gateway."""
from __future__ import annotations

import base64
import hashlib
import os
import re
import tempfile
from pathlib import Path

IMAGE_STORE_DIRNAME = "codex-image-store"
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def persist_image(artifact_dir: Path, image_data: str, sha256: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("invalid image digest")
    encoded = image_data.partition(",")[2]
    if len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise ValueError("image exceeds snapshot size limit")
    data = base64.b64decode(encoded, validate=True)
    if len(data) > MAX_IMAGE_BYTES or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("image snapshot digest mismatch")
    directory = artifact_dir / IMAGE_STORE_DIRNAME
    if directory.is_symlink():
        raise ValueError("image store must not be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".image-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / sha256)
    finally:
        Path(temporary).unlink(missing_ok=True)
