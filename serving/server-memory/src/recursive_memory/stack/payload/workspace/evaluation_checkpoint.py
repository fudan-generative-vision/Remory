"""Resolve the worker deployment without rewriting frozen evaluation pools."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_EVAL_CHECKPOINT = Path(
    "/inspire/ssd/project/sais-bio/public/chenbaoyou/workspace/SpecForge/outputs/"
    "qwen38-mmfine-repeat-ce-emptythink-all3-20260904-v3-step4500-serving"
)


def resolve_eval_checkpoint(configured: str | Path = DEFAULT_EVAL_CHECKPOINT) -> Path:
    """Apply an explicit launch-time pin; retain historical configs otherwise.

    Call this only at execution/preflight boundaries, never when computing a
    frozen pool's identity. Live server attestation must still match this path.
    """
    value = os.environ.get("R2EACT_CHECKPOINT") or os.fspath(configured)
    if not value.strip() or any(char in value for char in "\x00\r\n"):
        raise ValueError("evaluation compressor checkpoint must be a safe absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("evaluation compressor checkpoint must be an absolute path")
    return path.resolve()
