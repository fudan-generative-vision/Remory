"""Shared inference helpers; benchmark/training entry points are not distributed."""

from __future__ import annotations


import hashlib


import json


from pathlib import Path


from typing import Any, Mapping, Optional, Sequence


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_checkpoint_config(checkpoint: Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    required = ("compressor.pt", "config.json", "manifest.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete checkpoint {checkpoint}: missing {missing}"
        )
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "context_compression_ddp_v1":
        raise ValueError(f"unsupported checkpoint schema: {manifest.get('schema')!r}")
    digest = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    if manifest.get("config_sha256") != digest:
        raise ValueError("checkpoint config checksum does not match manifest")
    if manifest.get("frozen_target_saved") is not False:
        raise ValueError("checkpoint must declare the frozen target as excluded")
    return config


def _same_path_or_id(left: str, right: str) -> bool:
    if left == right:
        return True
    a, b = Path(left).expanduser(), Path(right).expanduser()
    return a.exists() and b.exists() and a.resolve() == b.resolve()
