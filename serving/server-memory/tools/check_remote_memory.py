#!/usr/bin/env python3
"""Run a small real Remory protocol smoke test against an existing pinned server.

Consumes inference capacity but starts/stops no services. Keeps only a JSON
receipt, and releases this test's handles/snapshots. Run with the server venv
or the local controller venv (torch + transformers are required).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/recursive_memory/stack/payload/workspace"))


def check(url, checkpoint, timeout):
    from context_compression.remory_sglang_client import ServerResidualSGLangClient
    from context_compression.remory_identity import FIELD
    from transformers import AutoTokenizer
    c = ServerResidualSGLangClient(url.rstrip("/"), request_timeout=timeout)
    began = time.monotonic()
    try:
        info = c.server_info()
        receipt = c.attest_residual_server(checkpoint, info)
        cfg = c._residual_config
        c.eos_token_id = cfg["summary_contract"]["end_token_id"]
        tok = AutoTokenizer.from_pretrained(cfg["target_model"], local_files_only=True)
        prefix = tok.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", add_special_tokens=False)
        history = tok.encode("<|im_start|>user\nRemember CEDAR-719.<|im_end|>\n" * 16, add_special_tokens=False)
        summary = tok.encode("The user asked to remember CEDAR-719.", add_special_tokens=False)
        first = c.build_residual_memory(prefix, history, summary, contract=cfg["summary_contract"])
        old = first["memory_embeddings"]
        new = c.build_residual_memory(prefix, [tok.eos_token_id] * old.shape[0] + history, summary,
            contract=cfg["summary_contract"], source_memory=old,
            source_memory_positions=list(range(len(prefix), len(prefix) + old.shape[0])),
            source_memory_depths=first["receipt"]["slot_depths"])["memory_embeddings"]
        c.release_memory(old)
        snapshot = c.save_memory(new)
        restored = c.load_memory(snapshot)
        combined = c.compose_memory([new, restored])
        c._control("renew", handles=[combined.descriptor], lease=True)
        suffix = tok.encode("<|im_start|>user\nWhat code?<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        result = c.generate_with_memory(prefix + [tok.eos_token_id] * combined.shape[0] + suffix,
            memory_positions=list(range(len(prefix), len(prefix) + combined.shape[0])),
            memory_embeddings=combined, expected_hidden_size=5120, max_new_tokens=32)
        return dict(success=True, protocol=receipt["transport"], elapsed_seconds=time.monotonic()-began,
            artifact_identity=info.get(FIELD), server_epoch=c._memory_epoch,
            backend=url, max_running_requests=info["max_running_requests"],
            memory_shape=list(new.shape), output_tokens=len(result.token_ids),
            decoded=tok.decode(result.token_ids), quality_benchmark=False)
    finally:
        # No user journal references this smoke test's snapshots.
        try:
            c.checkpoint_saved([])
            for memory in list(c.owned_memory().values()):
                c.release_memory(memory)
        finally:
            c.close_memory()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", required=True, help="Native /backends/NAME route, without /v1")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["REMORY_SGLANG_URL"] = args.base_url
    if args.api_key_file:
        os.environ["REMORY_SGLANG_API_KEY_FILE"] = str(args.api_key_file.resolve())
    try:
        result = check(args.base_url, args.checkpoint, args.timeout)
    except Exception as exc:
        args.output.write_text(json.dumps(dict(success=False, error=str(exc)), indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
