#!/usr/bin/env python3
"""Prepare a NEW unified-pool config; never restart or edit a running gateway."""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recursive_memory.stack.gateway import load_config, BACKEND_ROUTE


def build_config(source, *, url, key_file, checkpoint, name, model, capacity,
                 stateless_only=False, allow_http=False, memory_pinned_only=False, verified_artifact_pair=None):
    if memory_pinned_only and stateless_only:
        raise ValueError("pinned memory requires a full backend")
    if verified_artifact_pair and not memory_pinned_only:
        raise ValueError("verified artifact pairs require memory-pinned routing")
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    parsed = urlsplit(url)
    if (parsed.scheme != "https" and not (parsed.scheme == "http" and (allow_http or parsed.hostname in {"localhost", "127.0.0.1", "::1"}))
            or not BACKEND_ROUTE.fullmatch(parsed.path) or "/backends/" not in parsed.path
            or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("remote URL needs HTTPS and a pinned /backends/NAME route (HTTP requires --allow-http except on loopback)")
    checkpoint = Path(checkpoint).expanduser().resolve()
    local = json.loads((checkpoint / "config.json").read_text())
    config = load_config(source)
    # Only map local legacy engines for this explicitly selected model pin.
    for row in config["backends"]:
        if row["model"] == model:
            if row.get("model_path") and row["model_path"] != local["target_model"]:
                raise ValueError("existing public model points at a different local target")
            row["local_checkpoint"] = str(checkpoint)
    config["backends"].append(dict(name=name, url=url, model=model,
        max_inflight=capacity, api_key_file=str(Path(key_file).expanduser().resolve()),
        local_checkpoint=str(checkpoint), supports_local_encoder_versions=False,
        stateless_only=stateless_only, memory_pinned_only=memory_pinned_only))
    if verified_artifact_pair:
        config["backends"][-1]["verified_artifact_pair"] = verified_artifact_pair
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--base-url", required=True, help="Remote pinned native URL; optional trailing /v1 is removed")
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--local-checkpoint", type=Path, required=True)
    p.add_argument("--name", default="h20")
    p.add_argument("--model", default="Qwen3.8-27B")
    p.add_argument("--max-inflight", type=int, required=True)
    p.add_argument("--stateless-only", action="store_true", help="Admit ordinary generation only; require matching base identity, not compressor identity")
    p.add_argument("--memory-pinned-only", action="store_true", help="Route ordinary generation freely; require an explicit backend for memory")
    p.add_argument("--verified-artifact-pair", choices=["h20_online250_verified_tensor_pair_20261109_v1"], help="Opt in to the exact documented, proof-pinned artifact pair")
    p.add_argument("--allow-http", action="store_true", help="Explicitly allow a remote HTTP URL")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = build_config(args.config, url=args.base_url, key_file=args.api_key_file,
        checkpoint=args.local_checkpoint, name=args.name, model=args.model, capacity=args.max_inflight,
        stateless_only=args.stateless_only, allow_http=args.allow_http,
        memory_pinned_only=args.memory_pinned_only, verified_artifact_pair=args.verified_artifact_pair)
    # Exclusive creation also protects the source config from accidental overwrite.
    with args.output.open("x") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    try:
        load_config(args.output)
    except BaseException:
        args.output.unlink()
        raise
    print(json.dumps(dict(config=str(args.output.resolve()), restart_performed=False,
                         backend=args.name, native_url=result["backends"][-1]["url"])))


if __name__ == "__main__":
    main()
