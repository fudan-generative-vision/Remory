#!/usr/bin/env python3
"""Verify and relocate the pinned online-step-250 export into a new runtime.

Accepts the original serving export or the pinned public safetensors snapshot.
No downloads, dependency installation, GPU allocation or service changes.
Weights are linked without rewriting tensor bytes; keep the source.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "src/recursive_memory/stack/payload/workspace"
sys.path.insert(0, str(WORKSPACE))
from context_compression.qwen38_residual_serving_contract import (
    load_residual_serving_contract, canonical_sha256, read_json,
)
from context_compression.remory_identity import artifact_identity
from context_compression.remory_public_release import (
    ARCHITECTURE, RELEASE, verify_public_snapshot, serving_weight_path,
)


def prepare(*, checkpoint, target, prefix, destination, pins, public=False):
    checkpoint, target, prefix, destination = [Path(p).expanduser().resolve()
                                              for p in (checkpoint, target, prefix, destination)]
    if public:
        verify_public_snapshot(checkpoint)
        metadata = ROOT / "handoff/h20/compressor-metadata"
        config = read_json(metadata / "config.json")
        manifest = read_json(metadata / "manifest.json")
        manifest.update(compressor_file="model.safetensors",
            safetensors_sha256=RELEASE["safetensors_sha256"], public_revision=RELEASE["revision"])
        weights = checkpoint / "model.safetensors"
    else:
        config = load_residual_serving_contract(checkpoint, verify_weights=True)
        manifest = read_json(checkpoint / "manifest.json")
        weights = serving_weight_path(checkpoint)
    if (manifest["compressor_sha256"] != pins["artifact_identity"]["compressor_sha256"]
            or canonical_sha256(config) != pins["source_config_sha256"]):
        raise ValueError("source must be the unmodified, pinned online-step-250 serving export")
    # Only relocate target_model; all training/protocol provenance stays intact.
    config["target_model"] = str(target)
    manifest["config_sha256"] = canonical_sha256(config)
    if artifact_identity(checkpoint, config) != pins["artifact_identity"]:
        raise ValueError("relocated target metadata/contract differs from the deployment pin")
    for required in (prefix / "server/bin/python", prefix / "sglang/python/sglang/launch_server.py"):
        if not required.is_file():
            raise ValueError(f"provision the pinned engine/runtime first: {required}")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    serving = destination / "serving"
    serving.mkdir(mode=0o700)
    # A symlink works across filesystems and avoids an extra multi-GB transfer.
    (serving / manifest["compressor_file"]).symlink_to(weights)
    # The original config records an absolute source-host architecture path.
    # Retain that provenance; the hook resolves this adjacent verified copy.
    shutil.copyfile(ARCHITECTURE, serving / "dflash2_config.json")
    for name, value in (("config.json", config), ("manifest.json", manifest)):
        (serving / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    state = dict(schema="recursive_memory_stack_v1", prefix=str(prefix),
        workspace=str(WORKSPACE), controller_root=str(ROOT / "src"),
        target=str(target), checkpoint=str(serving),
        served_model_name="qwen38-remory-online-250", remory_release=dict(
            name="online-250", revision=RELEASE["revision"], online_step=250, base_stage2_step=14925),
        server=dict(port=30300, devices="0,1,2,3", tp=4, memory_fraction=0.75,
                    max_running_requests=2, context_length=262144,
                    max_total_tokens=262144, cuda_graph=False))
    if (prefix / "h20-engine.json").is_file():
        engine = read_json(prefix / "h20-engine.json")
        state.update(cuda_home=engine["cuda_home"], cuda_build=engine["cuda_build"])
    (destination / "runtime.json").write_text(json.dumps(state, indent=2) + "\n")
    (destination / "relocation.json").write_text(json.dumps(dict(
        source_checkpoint=str(checkpoint), source_config_sha256=pins["source_config_sha256"],
        weight_file=manifest["compressor_file"], public_revision=manifest.get("public_revision"),
        artifact_identity=artifact_identity(serving), target_revision=pins["target"]["revision"],
        target_weight_integrity="operator must verify official pinned weights before serving"), indent=2) + "\n")
    return destination / "runtime.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Original online-step-250 serving export")
    source.add_argument("--public-checkpoint", type=Path, help="Public revision 5ada60e... snapshot directory")
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pins", type=Path, default=ROOT / "handoff/h20/pins.json")
    args = p.parse_args()
    print(prepare(checkpoint=args.public_checkpoint or args.checkpoint, target=args.target, prefix=args.prefix,
                  destination=args.output, pins=read_json(args.pins), public=args.public_checkpoint is not None))


if __name__ == "__main__":
    main()
