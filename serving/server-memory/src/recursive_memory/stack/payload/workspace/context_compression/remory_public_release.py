"""Fixed public export equivalence for online step 250 (base Stage-II-14925).

The source PT digest identifies the model; the safetensors digest verifies the
actual public file. Never accept an arbitrary source-to-export digest mapping
from an upstream server or a checkpoint's self-reported manifest.
"""
import hashlib
import json
from pathlib import Path

RELEASE = dict(
    repository="fudan-generative-ai/Qwen3.8-27B-REMORY-1.9B",
    revision="5ada60e61d5425012b92a7c2b2796fc0b0f29241",
    compressor_sha256="cf1fe99c2f51f330b4b93055b0da4f930e1b01d1841d0b9b9825a8e4dc386732",
    safetensors_sha256="59d36bcec81c44bb314b63afea4dae79d8cda53a8918a5364143d8ecb2a499e5",
    config_sha256="bd37e28b4bcf367be0101186b87a6b939ac1558585a4a6a4ae0d16bfdcf8fc60",
)
ARCHITECTURE = Path(__file__).with_name("remory_dflash2_config.json")


def verify_public_snapshot(checkpoint, *, verify_weights=True):
    from .qwen38_residual_serving_contract import read_json, file_sha256
    checkpoint = Path(checkpoint)
    manifest = read_json(checkpoint / "manifest.json")
    config = read_json(checkpoint / "config.json")
    if (manifest.get("schema") != "remory.inference.v1"
            or any(manifest.get(k) != RELEASE[k] for k in
                   ("compressor_sha256", "safetensors_sha256", "config_sha256"))):
        raise ValueError("public snapshot is not the pinned online-step-250 export")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if hashlib.sha256(canonical.encode()).hexdigest() != RELEASE["config_sha256"]:
        raise ValueError("public configuration checksum mismatch")
    path = checkpoint / "model.safetensors"
    if not path.is_file() or (verify_weights and file_sha256(path) != RELEASE["safetensors_sha256"]):
        raise ValueError("public safetensors checksum mismatch")
    return manifest


def validate_serving_export(manifest):
    if (manifest.get("compressor_file") != "model.safetensors"
            or manifest.get("compressor_sha256") != RELEASE["compressor_sha256"]
            or manifest.get("safetensors_sha256") != RELEASE["safetensors_sha256"]
            or manifest.get("public_revision") != RELEASE["revision"]):
        raise ValueError("untrusted public export mapping; only pinned online step 250 is supported")


def serving_weight_path(checkpoint):
    from .qwen38_residual_serving_contract import read_json
    checkpoint = Path(checkpoint)
    name = read_json(checkpoint / "manifest.json").get("compressor_file")
    if name not in {"compressor.pt", "model.safetensors"}:
        raise ValueError("unsupported residual weight file")
    return checkpoint / name


def architecture_path(checkpoint, config):
    from .qwen38_residual_serving_contract import canonical_sha256, read_json
    # Use a portable adjacent copy where available. Old local checkpoints keep
    # their original path as a fallback, and both must match our fixed config.
    adjacent = Path(checkpoint) / "dflash2_config.json"
    path = adjacent if adjacent.is_file() else Path(config["dflash2_architecture_config"])
    if canonical_sha256(read_json(path)) != canonical_sha256(read_json(ARCHITECTURE)):
        raise ValueError("DFlash2 architecture configuration differs from the serving pin")
    return path
