"""Portable compressor contract and target metadata identity.

The server verifies compressor bytes when loading its hook. The target must
still be provisioned from the pinned, unmodified model revision; this metadata
identity is not a remote proof of all target weight bytes.
"""
from pathlib import Path
import re

FIELD = "remory_artifact_identity"
SCHEMA = "remory_artifact_identity_v1"
_loaded_identity = None


def artifact_identity(checkpoint, config=None):
    from context_compression.qwen38_residual_serving_contract import (
        load_residual_serving_contract, verify_residual_target, read_json,
        file_sha256, canonical_sha256,
    )
    checkpoint = Path(checkpoint)
    config = config or load_residual_serving_contract(checkpoint)
    target = verify_residual_target(config)
    contract = {k: v for k, v in config.items() if k != "target_model"}
    return dict(schema=SCHEMA,
        compressor_sha256=read_json(checkpoint / "manifest.json")["compressor_sha256"],
        contract_sha256=canonical_sha256(contract),
        target_metadata_sha256=canonical_sha256({
            "config": file_sha256(target / "config.json"),
            "tokenizer": config["summary_contract"]["tokenizer_files"],
        }))


def validate_identity(value):
    if (not isinstance(value, dict) or set(value) != {
            "schema", "compressor_sha256", "contract_sha256", "target_metadata_sha256"}
            or value["schema"] != SCHEMA
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
                   for k, v in value.items() if k != "schema")):
        raise ValueError("invalid portable Remory artifact identity")
    return dict(value)


def register_loaded_identity(checkpoint, config):
    global _loaded_identity
    value = artifact_identity(checkpoint, config)
    if _loaded_identity is not None and _loaded_identity != value:
        raise ValueError("a worker cannot advertise multiple residual identities")
    _loaded_identity = value


def loaded_identity():
    return None if _loaded_identity is None else dict(_loaded_identity)


def check_server_identity(info, checkpoint, config):
    if FIELD not in info:
        return False  # Legacy endpoints retain their original path checks.
    observed = validate_identity(info[FIELD])
    expected = artifact_identity(checkpoint, config)
    if observed != expected:
        from .h20_equivalence import verified_pair
        if not verified_pair(expected, info, config):
            raise RuntimeError("remote Remory compressor/contract/target metadata differs from the local pin")
    return True
